"""
High-level vector database API.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from .indexes import Index, BruteForceIndex, PQIndex
from .models import SearchResult


class VectorDatabase:
    """
    Vector database with metadata support and pluggable index backends.
    
    Separates concerns: VectorDatabase manages IDs/metadata/API,
    Index handles vector storage and search.
    """
    
    def __init__(
        self, 
        dimension: int, 
        metric: str = 'cosine',
        index_type: str = 'brute_force'
    ):
        """
        Args:
            dimension: Vector dimensionality
            metric: 'cosine' or 'euclidean'
            index_type: 'brute_force', 'pq', or 'hnsw'
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.index = self._create_index(index_type, metric)
        
    def _create_index(self, index_type: str, metric: str) -> Index:
        """Factory for index implementations."""
        if index_type == 'brute_force':
            return BruteForceIndex(metric=metric)
        elif index_type == 'pq':
            # Default PQ params: 8 subvectors, 256 clusters (8 bits per subvector)
            return PQIndex(n_subvectors=8, n_clusters=256, metric=metric)
        elif index_type == 'hnsw':
            raise NotImplementedError("HNSW not yet implemented")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def upsert(
        self, 
        id: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert or update a vector in the database."""
        assert vector.shape == (self.dimension,), \
            f"Vector dimension {vector.shape} doesn't match {self.dimension}"
        assert id not in self.index._id_to_idx, f"ID '{id}' already exists"
        
        if metadata is not None:
            self._metadata[id] = metadata
        
        self.index.add(id, vector.copy())
    
    def upsert_batch(
        self, 
        ids: List[str], 
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Bulk upsert vectors using index.build() for better performance."""
        assert len(ids) == vectors.shape[0], \
            f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        assert vectors.shape[1] == self.dimension, \
            f"Vector dimension {vectors.shape[1]} doesn't match {self.dimension}"
        
        # Store metadata
        if metadata:
            for i, id in enumerate(ids):
                if i < len(metadata) and metadata[i]:
                    self._metadata[id] = metadata[i]
        
        # Build index in one shot
        self.index.build(vectors, ids)
    
    def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if found and deleted."""
        self._metadata.pop(id, None)
        return self.index.delete(id)
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10,
        return_metadata: bool = True
    ) -> List[SearchResult]:
        """Find k nearest neighbors."""
        assert query.shape == (self.dimension,), \
            f"Query dimension {query.shape} doesn't match {self.dimension}"
        
        raw_results = self.index.search(query, k)
        
        return [
            SearchResult(
                id=id,
                distance=distance,
                metadata=self._metadata.get(id) if return_metadata else None
            )
            for id, distance in raw_results
        ]
    
    def size(self) -> int:
        """Number of vectors in the database."""
        return self.index.size()
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return (
            f"VectorDatabase(dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"index_type={self.index_type}, "
            f"size={self.size()})"
        )
