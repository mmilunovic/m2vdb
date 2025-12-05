from typing import List, Optional, Dict
import numpy as np
import os
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

from .base import Index

class PQIndex(Index):
    """
    Product Quantization (PQ) index implementation.
    
    Product Quantization compresses vectors by splitting them into subvectors and
    quantizing each subvector independently using learned codebooks. This achieves
    significant memory savings while maintaining somewhat reasonable search accuracy.
    
    Memory usage: O(n * m * log2(k)) bits instead of O(n * d * 32) bits
    where n=num_vectors, m=num_subvectors, k=clusters_per_subvector, d=dimensionality
    
    Trade-offs:
    - Much lower memory footprint (e.g., 128D vectors: 512 bytes -> 8 bytes with m=8, k=256)
    - Approximate search (not exact like brute force)
    - Build time includes k-means clustering overhead
    """
    def __init__(self, n_subvectors: int, n_clusters: int, metric: str = 'euclidean'):
        """
        Initialize Product Quantization index.
        
        Args:
            n_subvectors: Number of subvectors to split each vector into (m in PQ literature).
                         Higher values = more compression but also more approximation error.
            n_clusters: Number of clusters per subvector (k in PQ literature).
                       Typically 256 (8 bits per subvector). Must be >= 1.
            metric: Distance metric ('cosine' or 'euclidean')
        """
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.metric = metric

        # Codebooks: learned cluster centroids, shape (n_subvectors, n_clusters, subvector_dim)
        # Each codebook[m] contains k centroids for the m-th subvector slice
        self.codebooks: Optional[np.ndarray] = None
        
        # K-means models: store trained models for fast predict() during encoding
        self.kmeans_models: Optional[List[KMeans]] = None
        
        # Quantized codes: compressed vector representations, shape (n_vectors, n_subvectors)
        # Each code[i, m] is an integer in [0, k-1] representing which centroid
        # the m-th subvector of vector i is closest to
        self.quantized_codes: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # Dimensionality of each subvector (computed during build)
        self.subvector_dim: Optional[int] = None
        
        # Set up metric-specific functions (strategy pattern)
        if metric == 'cosine':
            self._normalize = self._normalize_cosine
            self._compute_lookup_table = self._compute_lookup_table_cosine
        else:
            self._normalize = self._normalize_noop
            self._compute_lookup_table = self._compute_lookup_table_euclidean
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built (codebooks trained)."""
        return self.codebooks is not None
    
    def _normalize_noop(self, vectors: np.ndarray) -> np.ndarray:
        """No-op normalization for euclidean metric."""
        return vectors
    
    def _normalize_cosine(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine metric."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms
    
    def _compute_lookup_table_euclidean(self, codebooks: np.ndarray, query_subvecs: np.ndarray) -> np.ndarray:
        """Compute lookup table for euclidean distance."""
        # codebooks: (m, k, d), query_subvecs: (m, 1, d) -> (m, k, d)
        diff = codebooks - query_subvecs[:, np.newaxis, :]
        return np.sum(diff * diff, axis=2)  # (m, k)
    
    def _compute_lookup_table_cosine(self, codebooks: np.ndarray, query_subvecs: np.ndarray) -> np.ndarray:
        """Compute lookup table for cosine distance."""
        # codebooks: (m, k, d), query_subvecs: (m, d) -> (m, k)
        similarities = np.einsum('mki,mi->mk', codebooks, query_subvecs)
        return 1 - similarities

    def _compute_distances(self, centroids: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Compute distances between centroids and query based on metric.
        """
        if self.metric == 'cosine':
            similarities = np.dot(centroids, query)
            return 1 - similarities
        else:
            diff = centroids - query
            return np.sum(diff * diff, axis=1)

    def _encode_vector(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vector(s) into PQ codes by finding nearest centroid for each subvector.
        
        Uses sklearn's predict() for fast encoding. Parallelizes for batch encoding.
        
        Args:
            vectors: Single vector of shape (dim,) or batch of shape (n, dim)
            
        Returns:
            Codes of shape (n_subvectors,) for single vector or (n, n_subvectors) for batch
        """
        # Handle both single vector and batch
        is_single = vectors.ndim == 1
        if is_single:
            vectors = vectors.reshape(1, -1)
        
        n = vectors.shape[0]
        codes = np.empty((n, self.n_subvectors), dtype=np.int32)
        
        # OPTIMIZED: Parallelize encoding for large batches
        if n > 100:  # Only parallelize for batches (avoid overhead for small n)
            def encode_subvector(m):
                sub_vectors = vectors[:, m * self.subvector_dim: (m + 1) * self.subvector_dim]
                return m, self.kmeans_models[m].predict(sub_vectors)
            
            with ThreadPoolExecutor(max_workers=self.n_subvectors) as executor:
                results = executor.map(encode_subvector, range(self.n_subvectors))
                for m, pred in results:
                    codes[:, m] = pred
        else:
            # Sequential for single vectors or small batches
            for m in range(self.n_subvectors):
                sub_vectors = vectors[:, m * self.subvector_dim: (m + 1) * self.subvector_dim]
                codes[:, m] = self.kmeans_models[m].predict(sub_vectors)
        
        return codes[0] if is_single else codes

    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build the PQ index by learning codebooks and quantizing vectors.
        
        This performs k-means clustering on each subvector slice independently,
        then encodes all input vectors using the learned codebooks.
        
        Args:
            vectors: numpy array of shape (n, dim) containing all vectors
            ids: list of string IDs corresponding to each vector
        """
        # Validate inputs
        assert len(ids) == vectors.shape[0], \
            f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        assert len(set(ids)) == len(ids), "Duplicate IDs found in the input"
        
        d = vectors.shape[1]
        assert d % self.n_subvectors == 0, \
            f"Dimensionality ({d}) must be divisible by n_subvectors ({self.n_subvectors})"
        
        self.subvector_dim = d // self.n_subvectors

        # Normalize vectors using the metric-specific strategy
        vectors = self._normalize(vectors)

        # Learn codebooks: train k-means for each subvector in parallel
        self.codebooks = np.empty((self.n_subvectors, self.n_clusters, self.subvector_dim))
        self.kmeans_models = [None] * self.n_subvectors
        
        def train_kmeans(m):
            sub_vectors = vectors[:, m * self.subvector_dim: (m + 1) * self.subvector_dim]
            
            # Sample for training if dataset is large (speeds up k-means)
            sample_size = min(300_000, len(sub_vectors))
            if len(sub_vectors) > sample_size:
                indices = np.random.choice(len(sub_vectors), sample_size, replace=False)
                sub_vectors_sample = sub_vectors[indices]
            else:
                sub_vectors_sample = sub_vectors
            
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=10,
                max_iter=100,
                random_state=42
            )
            kmeans.fit(sub_vectors_sample)
            
            # Normalize centroids using the metric-specific strategy
            centers = self._normalize(kmeans.cluster_centers_.copy())
            kmeans.cluster_centers_ = centers
            
            return m, centers, kmeans
        
        # Train all subvectors in parallel
        with ThreadPoolExecutor(max_workers=self.n_subvectors) as executor:
            results = executor.map(train_kmeans, range(self.n_subvectors))
            for m, centers, kmeans in results:
                self.codebooks[m] = centers
                self.kmeans_models[m] = kmeans

        # Quantize all vectors using the unified encoding function
        self.quantized_codes = self._encode_vector(vectors)
        
        # Build ID mappings
        self.ids = ids
        self._id_to_idx = {id: idx for idx, id in enumerate(ids)}
            
    def search(self, query: np.ndarray, k: int) -> List[tuple[str, float]]:
        """
        Search for k nearest neighbors using asymmetric distance computation.
        
        Asymmetric distance: compute exact distances from query subvectors to codebook
        centroids, then approximate distances to database vectors using their codes.
        This is more accurate than symmetric distance (quantizing the query too).
        
        Args:
            query: query vector of shape (dim,)
            k: number of nearest neighbors to return
            
        Returns:
            List of (id, distance) tuples sorted by distance (closest first)
        """
        # Edge case handling
        if self.codebooks is None or len(self.ids) == 0 or k == 0:
            return []
        
        # Normalize query using the metric-specific strategy
        query = self._normalize(query.reshape(1, -1))[0]
        
        query_subvecs = query.reshape(self.n_subvectors, self.subvector_dim)
        
        # Compute lookup table using the metric-specific strategy
        lookup_table = self._compute_lookup_table(self.codebooks, query_subvecs)

        m_indices = np.arange(self.n_subvectors)[:, np.newaxis]  # (m, 1)
        code_indices = self.quantized_codes.T  # (m, n_vectors)
        
        distances_per_subvec = lookup_table[m_indices, code_indices]
        
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum(distances_per_subvec, axis=0))
        else:
            distances = np.sum(distances_per_subvec, axis=0)
        
        # argpartition is O(n) vs argsort's O(n log n)
        n = len(distances)
        k = min(k, n)  # Ensure k doesn't exceed array size
        
        # Get k smallest distances (unsorted)
        partition_indices = np.argpartition(distances, k-1)[:k]
        top_k_indices = partition_indices[np.argsort(distances[partition_indices])]
        
        return [(self.ids[idx], float(distances[idx])) for idx in top_k_indices]

    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a single vector to the index after initial build.
        
        The vector is quantized using the existing codebooks and added to the index.
        Note: This requires the index to be built first (codebooks must exist).
        
        Args:
            id: unique string ID for this vector
            vector: vector of shape (dim,) to add
        """
        if id in self._id_to_idx:
            raise ValueError(f"ID '{id}' already exists in the index")
        
        if self.codebooks is None:
            raise RuntimeError("Index must be built before adding vectors. Call build() first.")
        
        # Normalize using the metric-specific strategy
        vector = self._normalize(vector.reshape(1, -1))[0]
        
        # Encode the vector
        codes = self._encode_vector(vector)
        
        # Determine the index where this vector will live
        new_idx = len(self.ids)
        
        # Append to the codes array
        if self.quantized_codes is None:
            self.quantized_codes = codes.reshape(1, -1)
        else:
            self.quantized_codes = np.vstack([self.quantized_codes, codes])
        
        # Update both mappings
        self.ids.append(id)
        self._id_to_idx[id] = new_idx
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the index by ID.
        
        Uses swap-and-pop strategy: swap the deleted element with the last element,
        then remove the last element. This is O(1) but doesn't preserve insertion order.
        
        Args:
            id: the ID of the vector to delete
            
        Returns:
            True if the vector was found and deleted, False otherwise
        """
        if id not in self._id_to_idx:
            return False
        
        idx = self._id_to_idx[id]
        last_idx = len(self.ids) - 1
        
        # If deleting the last element, no swap needed
        if idx == last_idx:
            self.quantized_codes = self.quantized_codes[:-1]
            self.ids.pop()
            del self._id_to_idx[id]
            return True
        
        # Swap with last element, then pop
        last_id = self.ids[last_idx]
        
        # Swap in quantized codes array
        self.quantized_codes[idx] = self.quantized_codes[last_idx]
        self.quantized_codes = self.quantized_codes[:-1]
        
        # Swap in ids list
        self.ids[idx] = last_id
        self.ids.pop()
        
        # Update mappings: last element moved to deleted position
        self._id_to_idx[last_id] = idx
        del self._id_to_idx[id]
        
        return True
    
    def size(self) -> int:
        """Return the number of vectors currently in the index."""
        return len(self.ids)
    
    def memory_usage(self) -> int:
        """Calculate memory usage of PQ index structures in bytes."""
        total = 0
        
        # Codebooks: (n_subvectors, n_clusters, subvector_dim) float32
        if self.codebooks is not None:
            total += self.codebooks.nbytes
        
        # Quantized codes: (n_vectors, n_subvectors) int32
        # This is the BIG memory saving of PQ!
        if self.quantized_codes is not None:
            total += self.quantized_codes.nbytes
        
        return total
    
    def save_artifacts(self, artifacts_dir: str) -> None:
        """
        Save trained PQ artifacts to disk.
        
        Saves:
        - codebooks.npy: the trained codebooks (centroids)
        - quantized_codes.npy: the encoded vectors (tiny memory footprint!)
        - ids.npy: the vector IDs
        - metadata.npz: other important state (n_subvectors, n_clusters, subvector_dim)
        
        If index is not yet built (not enough training samples), saves empty state.
        """
        if self.codebooks is None:
            # Index not built yet - save empty state marker
            # This happens when we have < n_clusters vectors for PQ training
            np.savez(
                os.path.join(artifacts_dir, "pq_metadata.npz"),
                n_subvectors=self.n_subvectors,
                n_clusters=self.n_clusters,
                subvector_dim=self.subvector_dim,
                is_built=False  # Marker that index is not trained yet
            )
            return
        
        np.save(os.path.join(artifacts_dir, "pq_codebooks.npy"), self.codebooks)
        
        # Save quantized codes and IDs (these are tiny but slow to recompute)
        if self.quantized_codes is not None:
            np.save(os.path.join(artifacts_dir, "pq_quantized_codes.npy"), self.quantized_codes)
        if self.ids:
            np.save(os.path.join(artifacts_dir, "pq_ids.npy"), np.array(self.ids))
        
        # Save metadata
        np.savez(
            os.path.join(artifacts_dir, "pq_metadata.npz"),
            n_subvectors=self.n_subvectors,
            n_clusters=self.n_clusters,
            subvector_dim=self.subvector_dim,
        )
    
    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Load trained PQ artifacts from disk.
        
        Reconstructs the codebooks, quantized codes, IDs, and kmeans models without retraining.
        If index was not built (not enough samples), loads empty state.
        """
        codebooks_path = os.path.join(artifacts_dir, "pq_codebooks.npy")
        metadata_path = os.path.join(artifacts_dir, "pq_metadata.npz")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("PQ metadata not found")
        
        # Load metadata first to check if index was built
        metadata = np.load(metadata_path)
        
        # Check if index was built when saved
        if 'is_built' in metadata and not metadata['is_built']:
            # Index was not trained when saved (not enough samples)
            # Keep codebooks as None - will train when enough vectors are added
            self.codebooks = None
            return
        
        # Index was built - load codebooks
        if not os.path.exists(codebooks_path):
            raise FileNotFoundError("PQ codebooks not found")
        
        self.codebooks = np.load(codebooks_path)
        metadata = np.load(metadata_path)
        self.subvector_dim = int(metadata['subvector_dim'])
        
        # Load quantized codes and IDs if they exist
        quantized_codes_path = os.path.join(artifacts_dir, "pq_quantized_codes.npy")
        ids_path = os.path.join(artifacts_dir, "pq_ids.npy")
        
        if os.path.exists(quantized_codes_path):
            self.quantized_codes = np.load(quantized_codes_path)
        
        if os.path.exists(ids_path):
            ids_array = np.load(ids_path)
            self.ids = [str(id) for id in ids_array]
            self._id_to_idx = {id: idx for idx, id in enumerate(self.ids)}
        
        # Reconstruct kmeans models from codebooks for fast encoding
        self.kmeans_models = []
        for m in range(self.n_subvectors):
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1)
            # Fake fit by directly setting cluster centers
            kmeans.cluster_centers_ = self.codebooks[m]
            kmeans._n_threads = 1
            self.kmeans_models.append(kmeans)