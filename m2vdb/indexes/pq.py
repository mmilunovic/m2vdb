



from typing import List, Optional, Dict
import numpy as np
from sklearn.cluster import KMeans

from .base import Index

class PQIndex(Index):
    """
    Product Quantization (PQ) index implementation.
    
    Product Quantization compresses vectors by splitting them into subvectors and
    quantizing each subvector independently using learned codebooks. This achieves
    significant memory savings while maintaining reasonable search accuracy.
    
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
        
        # Quantized codes: compressed vector representations, shape (n_vectors, n_subvectors)
        # Each code[i, m] is an integer in [0, k-1] representing which centroid
        # the m-th subvector of vector i is closest to
        self.quantized_codes: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # Dimensionality of each subvector (computed during build)
        self.subvector_dim: Optional[int] = None
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built (codebooks trained)."""
        return self.codebooks is not None

    def _compute_distances(self, centroids: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Compute distances between centroids and query based on metric."""
        if self.metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            centroid_norms = np.linalg.norm(centroids, axis=1)
            query_norm = np.linalg.norm(query)
            similarities = np.dot(centroids, query) / (centroid_norms * query_norm + 1e-10)
            return 1 - similarities
        else:
            # Euclidean distance
            return np.linalg.norm(centroids - query, axis=1)

    def _encode_vector(self, vector: np.ndarray) -> np.ndarray:
        """Encode a vector into PQ codes by finding nearest centroid for each subvector."""
        codes = np.zeros(self.n_subvectors, dtype=np.int32)
        for m in range(self.n_subvectors):
            sub_vector = vector[m * self.subvector_dim: (m + 1) * self.subvector_dim]
            distances = self._compute_distances(self.codebooks[m], sub_vector)
            codes[m] = np.argmin(distances)
        return codes

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

        # Normalize vectors if using cosine metric
        if self.metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            vectors = vectors / norms

        # Learn codebooks: allocate numpy array directly
        self.codebooks = np.empty((self.n_subvectors, self.n_clusters, self.subvector_dim))
        for m in range(self.n_subvectors):
            sub_vectors = vectors[:, m * self.subvector_dim: (m + 1) * self.subvector_dim]
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            kmeans.fit(sub_vectors)
            self.codebooks[m] = kmeans.cluster_centers_

        # Quantize all vectors and build ID mappings in one pass
        n = vectors.shape[0]
        self.quantized_codes = np.empty((n, self.n_subvectors), dtype=np.int32)
        self.ids = ids
        self._id_to_idx = {}
        
        for idx in range(n):
            self.quantized_codes[idx] = self._encode_vector(vectors[idx])
            self._id_to_idx[ids[idx]] = idx
            
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
        
        # Build distance lookup table: for each subvector, compute distances
        # from query subvector to all centroids in that codebook
        lookup_table = np.empty((self.n_subvectors, self.n_clusters))
        for m in range(self.n_subvectors):
            sub_vector = query[m * self.subvector_dim: (m + 1) * self.subvector_dim]
            lookup_table[m] = self._compute_distances(self.codebooks[m], sub_vector)

        # Compute approximate distances to all database vectors
        # For each vector, sum up the distances of its quantized subvectors
        approx_distances = []
        for idx, codes in enumerate(self.quantized_codes):
            dist = sum(lookup_table[m, codes[m]] for m in range(self.n_subvectors))
            approx_distances.append((self.ids[idx], dist))

        # Sort and return top k
        approx_distances.sort(key=lambda x: x[1])
        return approx_distances[:k]

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
        
        # Normalize if using cosine metric (must match build behavior)
        if self.metric == 'cosine':
            norm = np.linalg.norm(vector)
            if norm == 0:
                norm = 1e-10
            vector = vector / norm
        
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