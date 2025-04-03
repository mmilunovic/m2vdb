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
    def search(self, queries: np.ndarray, k: int = 10) -> List[List[int]]:
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
            
        self.vectors = []
        self.ids = []

    def add(self, vecs: np.ndarray, ids: Optional[List[int]] = None) -> None:
        vecs = np.array(vecs)
        if ids is None:
            ids = list(range(len(self.vectors), len(self.vectors) + len(vecs)))
        self.vectors.append(vecs)
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int = 10) -> List[List[int]]:
        queries = np.array(queries)
        all_vectors = np.vstack(self.vectors)
        dists = self._metric_fn(queries, all_vectors)
        idx = np.argsort(dists, axis=1)[:, :k]
        return [[self.ids[i] for i in row] for row in idx]

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
        self.random_seed = kwargs.pop('random_seed', 1312)  # Store the seed value as an integer
        self._random_gen = random.Random(self.random_seed)  # Create Random object from seed
        
        # Check for unexpected parameters
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for ANNIndex: {unexpected}")
            
        self.vectors = []
        self.ids = []
        
    def add(self, vecs: np.ndarray, ids: Optional[List[int]] = None) -> None:
        vecs = np.array(vecs)
        if ids is None:
            ids = list(range(len(self.vectors), len(self.vectors) + len(vecs)))
        self.vectors.append(vecs)
        self.ids.extend(ids)
        
    def search(self, queries: np.ndarray, k: int = 10) -> List[List[int]]:
        queries = np.array(queries)
        all_vectors = np.vstack(self.vectors)
        total_vectors = len(all_vectors)
        
        results = []
        for query in queries:
            # Randomly sample candidates (more than k but less than total)
            num_samples = min(self.num_candidates, total_vectors)
            candidate_indices = self._random_gen.sample(range(total_vectors), num_samples)
            
            # Calculate distances only for sampled candidates
            candidate_vectors = all_vectors[candidate_indices]
            dists = self._metric_fn(query.reshape(1, -1), candidate_vectors)[0]
            
            # Get top k among candidates
            top_k_local = np.argsort(dists)[:k]
            top_k_global = [candidate_indices[i] for i in top_k_local]
            results.append([self.ids[i] for i in top_k_global])
            
        return results
