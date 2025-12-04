"""
Minimal FAISS wrappers for benchmarking comparison.

Matches m2vdb's Collection interface so benchmark code works unchanged.
Uses FAISS directly without unnecessary overhead.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import faiss


class FAISSIndexWrapper:
    """Minimal FAISS wrapper matching m2vdb Collection interface."""
    
    def __init__(self, dimension: int, metric: str = "euclidean"):
        """
        Initialize FAISS index wrapper.
        
        Args:
            dimension: Vector dimensionality
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.dimension = dimension
        self.metric = metric
        self._vectors: Dict[str, np.ndarray] = {}  # Used by benchmark code
        self.index: Optional[faiss.Index] = None
        self.index_params: Dict[str, Any] = {}
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from _vectors dict (called by benchmark code)."""
        raise NotImplementedError("Subclasses must implement _rebuild_index")
    
    def search(self, query: np.ndarray, k: int = 10, return_metadata: bool = False) -> List[Any]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of neighbors to return
            return_metadata: Unused, for interface compatibility
            
        Returns:
            List of search results with id and distance
        """
        if self.index is None:
            return []
        
        query = query.reshape(1, -1).astype(np.float32).copy()
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k)
        
        # Convert FAISS results to match m2vdb format
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append(type('Result', (), {
                    'id': f'vec_{idx}',
                    'distance': float(dist)
                })())
        
        return results
    
    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return len(self._vectors)


class FAISSBruteForce(FAISSIndexWrapper):
    """FAISS Brute Force (Flat) index wrapper."""
    
    def __init__(self, dimension: int, metric: str = "euclidean"):
        super().__init__(dimension, metric)
        self.index_params = {"metric": metric}
        
    def _rebuild_index(self) -> None:
        """Build FAISS flat index from _vectors dict."""
        if not self._vectors:
            return
        
        # Extract vectors in order by ID (vec_0, vec_1, ...)
        ids_and_vecs = [(int(id.split('_')[1]), vec) for id, vec in self._vectors.items()]
        ids_and_vecs.sort(key=lambda x: x[0])
        vectors = np.array([vec for _, vec in ids_and_vecs], dtype=np.float32)
        
        # Create index
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(vectors)  # In-place normalization
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(vectors)


class FAISSPQ(FAISSIndexWrapper):
    """FAISS Product Quantization index wrapper."""
    
    def __init__(self, dimension: int, metric: str = "euclidean", 
                 n_subvectors: int = 8, n_clusters: int = 256):
        """
        Initialize FAISS PQ index.
        
        Args:
            dimension: Vector dimensionality
            metric: Distance metric ('euclidean' or 'cosine')
            n_subvectors: Number of subvectors (m in PQ)
            n_clusters: Number of clusters per subvector (k in PQ)
        """
        super().__init__(dimension, metric)
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.index_params = {
            "metric": metric,
            "n_subvectors": n_subvectors,
            "n_clusters": n_clusters
        }
        
    def _rebuild_index(self) -> None:
        """Build FAISS PQ index from _vectors dict."""
        if not self._vectors:
            return
        
        # Extract vectors in order by ID (vec_0, vec_1, ...)
        ids_and_vecs = [(int(id.split('_')[1]), vec) for id, vec in self._vectors.items()]
        ids_and_vecs.sort(key=lambda x: x[0])
        vectors = np.array([vec for _, vec in ids_and_vecs], dtype=np.float32)
        
        # Create PQ index
        nbits = int(np.log2(self.n_clusters))
        
        if self.metric == "cosine":
            self.index = faiss.IndexPQ(self.dimension, self.n_subvectors, nbits, faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(vectors)  # In-place normalization
        else:
            self.index = faiss.IndexPQ(self.dimension, self.n_subvectors, nbits)
        
        # Train and add
        self.index.train(vectors)
        self.index.add(vectors)


def create_faiss_index(index_type: str, dimension: int, metric: str = "euclidean", 
                       index_params: Optional[Dict[str, Any]] = None) -> FAISSIndexWrapper:
    """
    Factory function to create FAISS index wrappers.
    
    Args:
        index_type: Type of index ('brute_force' or 'pq')
        dimension: Vector dimensionality
        metric: Distance metric ('euclidean' or 'cosine')
        index_params: Additional parameters for the index
        
    Returns:
        FAISSIndexWrapper instance
    """
    index_params = index_params or {}
    
    if index_type == "brute_force" or index_type == "flat":
        return FAISSBruteForce(dimension, metric)
    elif index_type == "pq":
        n_subvectors = index_params.get("n_subvectors", 8)
        n_clusters = index_params.get("n_clusters", 256)
        return FAISSPQ(dimension, metric, n_subvectors, n_clusters)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
