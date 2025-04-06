# m2vdb/index.py

import numpy as np
from m2vdb.distance import euclidean_distance, cosine_similarity
from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.cluster import KMeans
from collections import defaultdict


class BaseIndex(ABC):
    """Abstract base class for all vector indexes"""
    def __init__(self, dim: int, metric: str = 'euclidean', **kwargs):
        """
        Initialize base index
        
        Args:
            dim: Dimensionality of vectors
            metric: Distance metric ('euclidean' or 'cosine')
            **kwargs: Additional parameters for derived classes
        """
        self.dim = dim
        self.metric = metric
        self._metric_fn = euclidean_distance if metric == 'euclidean' else cosine_similarity

    @abstractmethod
    def add(self, vecs: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index"""
        pass

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """Search for k nearest neighbors"""
        pass

class BruteForceIndex(BaseIndex):
    """Exact nearest neighbor search using brute force"""
    def __init__(self, dim: int, metric: str = 'euclidean', **kwargs):
        """
        Initialize brute force index
        
        Args:
            dim: Dimensionality of vectors
            metric: Distance metric ('euclidean' or 'cosine')
            **kwargs: Not used, but included for API consistency
        """
        super().__init__(dim, metric)
        
        # Check for unexpected parameters
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for BruteForceIndex: {unexpected}")
            
        # Store vectors directly in a numpy array for efficient computation
        self._vectors_array = np.zeros((0, dim), dtype=np.float32)
        self.ids = []

    def add(self, vecs: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """
        Add vectors to the index
        
        Args:
            vecs: Vectors to add (n x dim)
            ids: Optional list of IDs for the vectors
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (n, {self.dim}), got {vecs.shape}")
            
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(vecs)))
            
        # Pre-allocate new array and copy data
        new_size = len(self.ids) + len(vecs)
        new_array = np.empty((new_size, self.dim), dtype=np.float32)
        if len(self.ids) > 0:
            new_array[:len(self.ids)] = self._vectors_array
        new_array[len(self.ids):] = vecs
        
        # Update storage
        self._vectors_array = new_array
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Search for k nearest neighbors
        
        Args:
            queries: Query vectors (n x dim)
            k: Number of nearest neighbors to return
        Returns:
            Array of indices to nearest neighbors (n x k)
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected queries of shape (n, {self.dim}), got {queries.shape}")
            
        if len(self.ids) == 0:
            return np.zeros((len(queries), k), dtype=np.int64)
            
        # Compute distances efficiently using optimized distance function
        dists = self._metric_fn(queries, self._vectors_array)
        
        # Get top k indices
        k = min(k, len(self.ids))  # Don't request more neighbors than we have vectors
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        
        # Sort the k neighbors by distance
        rows = np.arange(len(queries))[:, None]
        idx = idx[rows, np.argsort(dists[rows, idx])]
        
        # Convert to array for faster indexing
        return np.array(self.ids)[idx]

class IVFIndex(BaseIndex):
    """
    Inverted File Index (IVF) for approximate nearest neighbor search using coarse quantization.
    """
    def __init__(self, dim: int, metric: str = 'euclidean', **kwargs):
        """
        Initialize IVF index.

        Args:
            dim: Dimensionality of vectors
            metric: Distance metric ('euclidean' or 'cosine')
            **kwargs:
                n_clusters: Number of clusters (default=64)
                n_probe: Number of clusters to search at query time (default=5)
                random_seed: Random seed for clustering (default=1312)
        """
        super().__init__(dim, metric)
        self.n_clusters = kwargs.pop('n_clusters', 64)
        self.n_probe = kwargs.pop('n_probe', 5)
        self.random_seed = kwargs.pop('random_seed', 1312)
        
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for IVFIndex: {unexpected}")
        
        self.centroids = None
        self.inverted_lists = defaultdict(list)
        self._vector_map = {}  # vector_id -> vector (optional)
        self.ids = []
        self._is_trained = False

    def train(self, vecs: np.ndarray) -> None:
        """
        Train the index by computing cluster centroids.
        Must be called before adding vectors.

        Args:
            vecs: Training vectors (n x dim)
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")

        # Compute centroids using k-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed, n_init="auto")
        kmeans.fit(vecs)
        self.centroids = kmeans.cluster_centers_
        self._is_trained = True

    def add(self, vecs: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """
        Add vectors to the index. The index must be trained first.

        Args:
            vecs: Vectors to add (n x dim)
            ids: Optional list of IDs for the vectors
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")
        
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(vecs)))

        # Train if not already trained
        if not self._is_trained:
            self.train(vecs)
        
        # Assign vectors to nearest centroids
        distances = self._metric_fn(vecs, self.centroids)
        cluster_ids = np.argmin(distances, axis=1)

        # Add vectors to inverted lists
        for i, cluster_id in enumerate(cluster_ids):
            self.inverted_lists[cluster_id].append((ids[i], vecs[i]))
            self._vector_map[ids[i]] = vecs[i]
        
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected queries of shape (n, {self.dim}), got {queries.shape}")
        
        if self.centroids is None or not self.inverted_lists:
            return np.zeros((len(queries), k), dtype=np.int64)

        results = []
        
        # Pre-compute centroid distances for all queries at once
        all_centroid_dists = self._metric_fn(queries, self.centroids)
        top_clusters_all = np.argpartition(all_centroid_dists, self.n_probe, axis=1)[:, :self.n_probe]

        for qi, q in enumerate(queries):
            # Use pre-computed top clusters
            top_clusters = top_clusters_all[qi]
            
            # Gather candidates more efficiently
            candidates = []
            candidate_count = sum(len(self.inverted_lists[cid]) for cid in top_clusters)
            if candidate_count == 0:
                results.append([0] * k)
                continue
                
            # Pre-allocate arrays for better memory efficiency
            candidate_ids = np.empty(candidate_count, dtype=np.int64)
            candidate_vecs = np.empty((candidate_count, self.dim), dtype=np.float32)
            
            # Fill arrays without list operations
            idx = 0
            for cid in top_clusters:
                cluster_candidates = self.inverted_lists[cid]
                chunk_size = len(cluster_candidates)
                for j, (id_, vec) in enumerate(cluster_candidates):
                    candidate_ids[idx + j] = id_
                    candidate_vecs[idx + j] = vec
                idx += chunk_size
            
            # Compute distances and get top-k in one step
            dists = self._metric_fn(q[None, :], candidate_vecs)[0]
            k_actual = min(k, len(dists))
            top_k_idx = np.argpartition(dists, k_actual)[:k_actual]
            top_k_sorted = top_k_idx[np.argsort(dists[top_k_idx])]
            
            results.append(candidate_ids[top_k_sorted].tolist())

        return np.array(results, dtype=np.int64)
