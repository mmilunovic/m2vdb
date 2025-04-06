# m2vdb/index.py

import numpy as np
from m2vdb.distance import euclidean_distance, cosine_similarity
import random
from abc import ABC, abstractmethod
from typing import List, Optional

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

class ANNIndex(BaseIndex):
    """Approximate nearest neighbor search using random sampling"""
    def __init__(self, dim: int, metric: str = 'euclidean', **kwargs):
        """
        Initialize ANN index
        
        Args:
            dim: Dimensionality of vectors
            metric: Distance metric ('euclidean' or 'cosine')
            **kwargs: Additional parameters:
                num_candidates: Number of candidates to sample (default=100)
                random_seed: Random seed for sampling (default=1312)
        """
        super().__init__(dim, metric)
        
        # Extract parameters with defaults
        self.num_candidates = kwargs.pop('num_candidates', 100)
        self.random_seed = kwargs.pop('random_seed', 1312)
        self._random_gen = random.Random(self.random_seed)
        
        # Check for unexpected parameters
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for ANNIndex: {unexpected}")
            
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
        Search for k approximate nearest neighbors using random sampling
        
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
            
        # Sample random indices once for all queries
        n_samples = min(self.num_candidates, len(self.ids))
        candidate_indices = np.array(self._random_gen.sample(range(len(self.ids)), n_samples))
        
        # Get candidate vectors once
        candidate_vectors = self._vectors_array[candidate_indices]
        
        # Compute all distances at once
        dists = self._metric_fn(queries, candidate_vectors)
        
        # Get top k for all queries at once
        k_actual = min(k, n_samples)
        idx = np.argpartition(dists, k_actual, axis=1)[:, :k_actual]
        
        # Sort top k by distance
        rows = np.arange(len(queries))[:, None]
        idx = idx[rows, np.argsort(dists[rows, idx])]
        
        # Map back to original indices
        return np.array(self.ids)[candidate_indices[idx]]
