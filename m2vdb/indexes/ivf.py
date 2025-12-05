from typing import List, Optional, Dict, Tuple
import numpy as np
import os
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import warnings

from .base import Index


class IVFIndex(Index):
    """
    Inverted File (IVF) index for approximate nearest neighbor search.

    IVF is a clustering-based approximate nearest neighbor search algorithm that:
    1. Partitions the vector space into n_clusters regions using k-means
    2. Stores vectors in inverted lists, one per cluster
    3. At search time, probes the nprobe nearest clusters to the query
    
    The index divides the vector space into clusters (Voronoi cells) and maintains
    an inverted list for each cluster. At search time, it only searches vectors
    within the nprobe nearest clusters, dramatically reducing the search space.
    
    Memory: O(n * d * 4) bytes for float32 vectors + O(n_clusters * d * 4) for centroids
    Search: O(n_clusters * d + (n/n_clusters) * nprobe * d) per query
    """
    
    def __init__(
        self,
        metric: str = 'euclidean',
        n_clusters: Optional[int] = None,
        nprobe: Optional[int] = None,
        kmeans_max_iter: int = 100,
        kmeans_n_init: int = 1
    ):
        """
        Initialize IVF index.
        
        Args:
            metric: Distance metric ('cosine' or 'euclidean')
            n_clusters: Number of clusters (Voronoi cells) to partition space into.
                       If None, will be set to sqrt(n_vectors) during build.
                       Typical range: sqrt(n) to n/10 for good speed/accuracy trade-off.
            nprobe: Number of nearest clusters to search at query time.
                   If None, will be set to sqrt(n_clusters) during build.
                   Higher values = more accurate but slower. Typical range: 1-32.
            kmeans_max_iter: Maximum iterations for k-means clustering (default: 100)
            kmeans_n_init: Number of k-means initializations (best result kept)
        """
        self.metric = metric
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_n_init = kmeans_n_init
        
        # Cluster centroids: shape (n_clusters, dimension), float32 for efficiency
        # For cosine: normalized to unit length
        self.centroids: Optional[np.ndarray] = None
        
        # Cached centroid norms squared for L2 distance optimization
        # Only used for euclidean metric (None for cosine since norms = 1)
        self.centroid_norms_sq: Optional[np.ndarray] = None
        
        # Inverted lists: one per cluster
        # Each inverted list stores vectors and their IDs as separate numpy arrays
        # for efficient vectorized distance computation
        # Structure: {cluster_id: {'vectors': ndarray, 'ids': ndarray}}
        self.inverted_lists: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Global ID to cluster mapping for O(1) lookups during delete
        self._id_to_cluster: Dict[str, int] = {}
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built (centroids learned)."""
        return self.centroids is not None
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms
    
    def _compute_distances_to_centroids(self, query: np.ndarray) -> np.ndarray:
        """
        Compute distances from query to all centroids.
        
        For cosine: Both query and centroids are normalized, so distance = 1 - dot product
        For L2: Use precomputed centroid norms: ||c - q||^2 = ||c||^2 + ||q||^2 - 2*c·q
        
        Args:
            query: Query vector of shape (dimension,), already normalized if cosine
            
        Returns:
            Distances array of shape (n_clusters,)
        """
        if self.metric == 'cosine':
            # Cosine distance = 1 - dot product (both normalized to unit length)
            similarities = self.centroids @ query
            return 1 - similarities
        else:
            # Euclidean: ||c - q||^2 = ||c||^2 + ||q||^2 - 2*c·q
            query_norm_sq = np.sum(query ** 2)
            dot_products = self.centroids @ query
            distances_sq = self.centroid_norms_sq + query_norm_sq - 2 * dot_products
            # Clamp to avoid numerical errors
            distances_sq = np.maximum(distances_sq, 0)
            return np.sqrt(distances_sq)
    
    def _compute_distances_in_cluster(
        self, 
        cluster_vectors: np.ndarray, 
        query: np.ndarray,
        cached_norms_sq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute distances from query to all vectors in a cluster.
        
        Args:
            cluster_vectors: Array of shape (n_vectors, dimension)
            query: Query vector of shape (dimension,)
            cached_norms_sq: Precomputed vector norms squared (for euclidean)
            
        Returns:
            Distances array of shape (n_vectors,)
        """
        if self.metric == 'cosine':
            # Cosine distance = 1 - dot product (both normalized)
            similarities = cluster_vectors @ query
            return 1 - similarities
        else:
            # Euclidean: ||v - q||^2 = ||v||^2 + ||q||^2 - 2*v·q
            # Use cached norms if available, otherwise compute
            if cached_norms_sq is not None:
                v_norms_sq = cached_norms_sq
            else:
                v_norms_sq = np.sum(cluster_vectors ** 2, axis=1)
            
            q_norm_sq = np.sum(query ** 2)
            dot_products = cluster_vectors @ query
            distances_sq = v_norms_sq + q_norm_sq - 2 * dot_products
            distances_sq = np.maximum(distances_sq, 0)
            return np.sqrt(distances_sq)
    
    def _assign_to_clusters(self, vectors: np.ndarray) -> np.ndarray:
        """
        Assign vectors to their nearest clusters using centroids.
        
        This computes distances to all centroids and assigns each vector
        to the nearest one. We don't need the KMeans object after training.
        
        Args:
            vectors: Array of shape (n_vectors, dimension), already normalized if cosine
            
        Returns:
            Cluster assignments of shape (n_vectors,)
        """
        if self.centroids is None:
            raise RuntimeError("Index not built - call build() first")
        
        # Compute distances to all centroids for each vector
        assignments = []
        for vec in vectors:
            distances = self._compute_distances_to_centroids(vec)
            assignments.append(np.argmin(distances))
        
        return np.array(assignments, dtype=np.int32)
    
    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build the IVF index by clustering vectors and populating inverted lists.
        
        Steps:
        1. Determine optimal n_clusters if not provided (sqrt(n_vectors))
        2. Learn cluster centroids using k-means
        3. Assign each vector to its nearest cluster
        4. Populate inverted lists with vectors and IDs
        
        Args:
            vectors: Array of shape (n_vectors, dimension), will be converted to float32
            ids: List of string IDs corresponding to each vector
        """
        # Validate inputs
        assert len(ids) == vectors.shape[0], \
            f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        assert len(set(ids)) == len(ids), "Duplicate IDs found in the input"
        
        n, d = vectors.shape
        
        # Auto-determine n_clusters if not provided: sqrt(n_vectors)
        if self.n_clusters is None:
            # Cap at 10000 for very large datasets to keep clustering fast
            self.n_clusters = min(int(np.sqrt(n)), 10000)
            self.n_clusters = max(self.n_clusters, 1)  # At least 1 cluster
        
        # Ensure we don't have more clusters than vectors
        if self.n_clusters > n:
            warnings.warn(
                f"n_clusters ({self.n_clusters}) > n_vectors ({n}). "
                f"Reducing to n_clusters = {n}"
            )
            self.n_clusters = n
        
        # Auto-determine nprobe if not provided: sqrt(n_clusters)
        if self.nprobe is None:
            self.nprobe = max(1, int(np.sqrt(self.n_clusters)))
        
        print(f"  IVF: Building index with {self.n_clusters} clusters, nprobe={self.nprobe}")
        
        # Convert to float32 for memory efficiency
        vectors = vectors.astype(np.float32)
        
        # Normalize vectors if using cosine metric
        if self.metric == 'cosine':
            vectors = self._normalize_vectors(vectors)
        
        # Learn cluster centroids using k-means
        print("  IVF: Running k-means clustering (this may take a minute)...")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.kmeans_max_iter,
            n_init=self.kmeans_n_init,
            random_state=42,
            algorithm='elkan',  # Faster for Euclidean
            verbose=0  # Keep it quiet to avoid clutter
        )
        
        cluster_assignments = kmeans.fit_predict(vectors)
        
        # Store centroids as float32 and normalize if cosine
        # We don't need to keep the kmeans object after this
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        if self.metric == 'cosine':
            self.centroids = self._normalize_vectors(self.centroids)
            self.centroid_norms_sq = None  # Not needed (all norms = 1)
        else:
            # Precompute centroid norms squared for L2 distance optimization
            self.centroid_norms_sq = np.sum(self.centroids ** 2, axis=1)
        
        # Populate inverted lists
        # Group vectors by cluster for efficient array construction
        self.inverted_lists = {}
        self._id_to_cluster = {}
        
        for cluster_id in range(self.n_clusters):
            # Find all vectors assigned to this cluster
            mask = cluster_assignments == cluster_id
            cluster_vectors = vectors[mask]
            cluster_ids = np.array([ids[i] for i in np.where(mask)[0]], dtype=object)
            
            # Store as numpy arrays for vectorized operations
            if len(cluster_vectors) > 0:
                inv_list = {
                    'vectors': cluster_vectors,  # Shape: (n_in_cluster, dimension)
                    'ids': cluster_ids            # Shape: (n_in_cluster,)
                }
                
                # Precompute vector norms squared for L2 distance optimization
                if self.metric == 'euclidean':
                    inv_list['norms_sq'] = np.sum(cluster_vectors ** 2, axis=1)
                
                self.inverted_lists[cluster_id] = inv_list
                
                # Update ID to cluster mapping
                for id_str in cluster_ids:
                    self._id_to_cluster[id_str] = cluster_id
            else:
                # Empty cluster
                inv_list = {
                    'vectors': np.empty((0, d), dtype=np.float32),
                    'ids': np.empty(0, dtype=object)
                }
                if self.metric == 'euclidean':
                    inv_list['norms_sq'] = np.empty(0, dtype=np.float32)
                
                self.inverted_lists[cluster_id] = inv_list
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors using IVF with nprobe clusters.
        
        Steps:
        1. Find nprobe nearest clusters to query
        2. Search vectors in those clusters
        3. Return top-k results across all probed clusters
        
        Args:
            query: Query vector of shape (dimension,)
            k: Number of nearest neighbors to return
            
        Returns:
            List of (id, distance) tuples sorted by distance
        """
        if not self.is_built or k == 0:
            return []
        
        # Convert to float32 and normalize if needed
        query = query.astype(np.float32)
        if self.metric == 'cosine':
            query = self._normalize_vectors(query.reshape(1, -1))[0]
        
        # Find nprobe nearest clusters using argpartition for efficiency
        centroid_distances = self._compute_distances_to_centroids(query)
        
        # Use argpartition for O(n) selection of nprobe clusters
        nprobe = min(self.nprobe, self.n_clusters)
        if nprobe < self.n_clusters:
            probe_indices = np.argpartition(centroid_distances, nprobe - 1)[:nprobe]
        else:
            probe_indices = np.arange(self.n_clusters)
        
        # Convert to Python list to avoid numpy int issues with threading
        probe_indices = probe_indices.tolist()
        
        # Search within selected clusters
        all_candidates = []
        
        # Only parallelize if searching MANY clusters (> 16) to avoid overhead
        # Most IVF configs have nprobe ~ 10-30, sequential is usually fine
        if nprobe > 16:
            # Parallel search for multiple clusters
            def search_cluster(cluster_id):
                inv_list = self.inverted_lists.get(cluster_id)
                if inv_list is None or len(inv_list['vectors']) == 0:
                    return []
                
                distances = self._compute_distances_in_cluster(
                    inv_list['vectors'], 
                    query,
                    cached_norms_sq=inv_list.get('norms_sq')
                )
                return list(zip(inv_list['ids'], distances))
            
            with ThreadPoolExecutor(max_workers=min(nprobe, 8)) as executor:
                results = executor.map(search_cluster, probe_indices)
                for result in results:
                    all_candidates.extend(result)
        else:
            # Sequential search for few clusters (avoid threading overhead)
            for cluster_id in probe_indices:
                inv_list = self.inverted_lists.get(cluster_id)
                if inv_list is None or len(inv_list['vectors']) == 0:
                    continue
                
                distances = self._compute_distances_in_cluster(
                    inv_list['vectors'], 
                    query,
                    cached_norms_sq=inv_list.get('norms_sq')
                )
                all_candidates.extend(zip(inv_list['ids'], distances))
        
        # If no candidates found, return empty
        if not all_candidates:
            return []
        
        # Select top-k from all candidates using argpartition
        candidate_distances = np.array([d for _, d in all_candidates], dtype=np.float32)
        k = min(k, len(candidate_distances))
        
        if k < len(candidate_distances):
            top_k_idx = np.argpartition(candidate_distances, k - 1)[:k]
            top_k_idx = top_k_idx[np.argsort(candidate_distances[top_k_idx])]
        else:
            top_k_idx = np.argsort(candidate_distances)
        
        return [(all_candidates[i][0], float(all_candidates[i][1])) for i in top_k_idx]
    
    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a single vector to the index after initial build.
        
        The vector is assigned to its nearest cluster and added to that
        cluster's inverted list.
        
        Args:
            id: Unique string ID for this vector
            vector: Vector of shape (dimension,) to add
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before adding vectors. Call build() first.")
        
        if id in self._id_to_cluster:
            raise ValueError(f"ID '{id}' already exists in the index")
        
        # Convert to float32 and normalize if needed
        vector = vector.astype(np.float32).reshape(1, -1)
        if self.metric == 'cosine':
            vector = self._normalize_vectors(vector)
        vector = vector[0]  # Back to 1D
        
        # Assign to nearest cluster
        cluster_id = self._assign_to_clusters(vector.reshape(1, -1))[0]
        
        # Add to inverted list
        inv_list = self.inverted_lists[cluster_id]
        inv_list['vectors'] = np.vstack([inv_list['vectors'], vector])
        inv_list['ids'] = np.append(inv_list['ids'], id)
        
        # Update cached norms for euclidean
        if self.metric == 'euclidean':
            norm_sq = np.sum(vector ** 2)
            inv_list['norms_sq'] = np.append(inv_list['norms_sq'], norm_sq)
        
        # Update mapping
        self._id_to_cluster[id] = cluster_id
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the index by ID.
        
        Uses O(1) lookup to find the cluster, then removes from inverted list.
        Uses swap-and-pop for O(1) deletion.
        
        Args:
            id: The ID of the vector to delete
            
        Returns:
            True if the vector was found and deleted, False otherwise
        """
        if id not in self._id_to_cluster:
            return False
        
        cluster_id = self._id_to_cluster[id]
        inv_list = self.inverted_lists[cluster_id]
        
        # Find index within cluster's inverted list
        idx = np.where(inv_list['ids'] == id)[0]
        if len(idx) == 0:
            return False
        idx = idx[0]
        
        # Swap-and-pop for O(1) deletion
        last_idx = len(inv_list['ids']) - 1
        
        if idx != last_idx:
            # Swap with last element
            inv_list['vectors'][idx] = inv_list['vectors'][last_idx]
            inv_list['ids'][idx] = inv_list['ids'][last_idx]
            if self.metric == 'euclidean':
                inv_list['norms_sq'][idx] = inv_list['norms_sq'][last_idx]
        
        # Remove last element
        inv_list['vectors'] = inv_list['vectors'][:-1]
        inv_list['ids'] = inv_list['ids'][:-1]
        if self.metric == 'euclidean':
            inv_list['norms_sq'] = inv_list['norms_sq'][:-1]
        
        # Update mapping
        del self._id_to_cluster[id]
        
        return True
    
    def size(self) -> int:
        """Return the total number of vectors in the index."""
        return len(self._id_to_cluster)
    
    def memory_usage(self) -> int:
        """Calculate memory usage of IVF index structures in bytes."""
        total = 0
        
        # Centroids: (n_clusters, dimension) float32
        if self.centroids is not None:
            total += self.centroids.nbytes
        
        # Cached centroid norms (if euclidean)
        if self.centroid_norms_sq is not None:
            total += self.centroid_norms_sq.nbytes
        
        # Inverted lists: all vectors stored across all clusters
        for inv_list in self.inverted_lists.values():
            if 'vectors' in inv_list:
                total += inv_list['vectors'].nbytes
            if 'norms_sq' in inv_list:
                total += inv_list['norms_sq'].nbytes
        
        return total
    
    def save_artifacts(self, artifacts_dir: str) -> None:
        """
        Save trained IVF artifacts to disk.
        
        Saves:
        - centroids.npy: the cluster centroids
        - metadata.npz: n_clusters, nprobe, centroid_norms_sq
        - inverted_lists/: all inverted lists (vectors, ids, norms_sq per cluster)
        - id_to_cluster.npz: mapping for fast lookups
        """
        if self.centroids is None:
            raise RuntimeError("Cannot save artifacts: index not built")
        
        np.save(os.path.join(artifacts_dir, "ivf_centroids.npy"), self.centroids)
        
        # Save metadata
        metadata = {
            'n_clusters': self.n_clusters,
            'nprobe': self.nprobe,
        }
        if self.centroid_norms_sq is not None:
            metadata['centroid_norms_sq'] = self.centroid_norms_sq
        
        np.savez(os.path.join(artifacts_dir, "ivf_metadata.npz"), **metadata)
        
        # Save inverted lists
        invlists_dir = os.path.join(artifacts_dir, "ivf_inverted_lists")
        os.makedirs(invlists_dir, exist_ok=True)
        
        for cluster_id, inv_list in self.inverted_lists.items():
            cluster_file = os.path.join(invlists_dir, f"cluster_{cluster_id}.npz")
            save_dict = {
                'vectors': inv_list['vectors'],
                'ids': inv_list['ids']
            }
            if 'norms_sq' in inv_list:
                save_dict['norms_sq'] = inv_list['norms_sq']
            np.savez(cluster_file, **save_dict)
        
        # Save ID to cluster mapping for fast lookups
        if self._id_to_cluster:
            ids = list(self._id_to_cluster.keys())
            clusters = [self._id_to_cluster[id] for id in ids]
            np.savez(
                os.path.join(artifacts_dir, "ivf_id_to_cluster.npz"),
                ids=np.array(ids),
                clusters=np.array(clusters)
            )
    
    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Load trained IVF artifacts from disk.
        
        Reconstructs the centroids, inverted lists, and ID mappings without retraining.
        """
        centroids_path = os.path.join(artifacts_dir, "ivf_centroids.npy")
        metadata_path = os.path.join(artifacts_dir, "ivf_metadata.npz")
        
        if not os.path.exists(centroids_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("IVF artifacts not found")
        
        # Load centroids
        self.centroids = np.load(centroids_path)
        
        # Load metadata
        metadata = np.load(metadata_path)
        self.n_clusters = int(metadata['n_clusters'])
        self.nprobe = int(metadata['nprobe'])
        
        if 'centroid_norms_sq' in metadata:
            self.centroid_norms_sq = metadata['centroid_norms_sq']
        
        # Load inverted lists
        self.inverted_lists = {}
        invlists_dir = os.path.join(artifacts_dir, "ivf_inverted_lists")
        
        if os.path.exists(invlists_dir):
            for cluster_id in range(self.n_clusters):
                cluster_file = os.path.join(invlists_dir, f"cluster_{cluster_id}.npz")
                if os.path.exists(cluster_file):
                    data = np.load(cluster_file, allow_pickle=True)
                    inv_list = {
                        'vectors': data['vectors'],
                        'ids': data['ids']
                    }
                    if 'norms_sq' in data:
                        inv_list['norms_sq'] = data['norms_sq']
                    self.inverted_lists[cluster_id] = inv_list
        
        # Load ID to cluster mapping
        id_to_cluster_path = os.path.join(artifacts_dir, "ivf_id_to_cluster.npz")
        if os.path.exists(id_to_cluster_path):
            data = np.load(id_to_cluster_path, allow_pickle=True)
            ids = data['ids']
            clusters = data['clusters']
            self._id_to_cluster = {str(id): int(cluster) for id, cluster in zip(ids, clusters)}

