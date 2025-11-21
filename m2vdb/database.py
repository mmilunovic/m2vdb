"""
High-level vector database API.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from .indexes import Index, BruteForceIndex, PQIndex, RustBruteForceIndex
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
        index_type: str = 'brute_force',
        rebuild_strategy: str = 'eager',
        index_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            dimension: Vector dimensionality
            metric: 'cosine' or 'euclidean'
            index_type: 'brute_force', 'pq', or 'hnsw'
            rebuild_strategy: When to rebuild index
                - 'eager': Rebuild on every upsert (default)
                - 'threshold': Rebuild every N vectors (TODO: not yet implemented)
            index_params: Optional parameters for the index (e.g., {'n_subvectors': 8, 'n_clusters': 256})
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.rebuild_strategy = rebuild_strategy
        self.index_params = index_params or {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.index = self._create_index(index_type, metric, self.index_params)
        
        # Store all vectors for rebuilding
        self._vectors: Dict[str, np.ndarray] = {}
        self._upserts_since_rebuild = 0
        
    def _create_index(self, index_type: str, metric: str, index_params: Dict[str, Any]) -> Index:
        """Factory for index implementations."""
        if index_type == 'brute_force':
            return BruteForceIndex(metric=metric)
        elif index_type == 'rust_brute_force':
            return RustBruteForceIndex(metric=metric)
        elif index_type == 'pq':
            return PQIndex(metric=metric, **index_params)
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
        
        # Check if this is an update
        if id in self._vectors:
            raise ValueError(f"ID '{id}' already exists. Delete first to update.")
        
        # Store vector and metadata
        self._vectors[id] = vector
        if metadata is not None:
            self._metadata[id] = metadata
        self._upserts_since_rebuild += 1
        
        # Decide whether to rebuild or add incrementally
        if self._should_rebuild():
            self._rebuild_index()
        else:
            # Incremental add to existing index
            self.index.add(id, vector)
    
    def _rebuild_index(self) -> None:
        """
        Rebuild the entire index from stored vectors.
        
        Called based on rebuild strategy. For PQ, this retrains k-means.
        For BruteForce, this just reorganizes the array.
        """
        if len(self._vectors) == 0:
            return
        
        # Extract all vectors and IDs in consistent order
        ids = list(self._vectors.keys())
        vectors = np.array([self._vectors[id] for id in ids])
        
        # Rebuild the index
        self.index.build(vectors, ids)
        self._upserts_since_rebuild = 0
    
    def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if found and deleted."""
        if id not in self._vectors:
            return False
        
        # Remove from storage
        del self._vectors[id]
        self._metadata.pop(id, None)
        
        # Rebuild index without this vector
        self._rebuild_index()
        
        return True
    
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
    
    def fetch(self, id: str) -> Optional[tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """
        Fetch a vector by ID.
        
        Returns:
            Tuple of (vector, metadata) if found, None if not found
        """
        if id not in self._vectors:
            return None
        return self._vectors[id], self._metadata.get(id)
    
    def __len__(self) -> int:
        """Number of vectors in the database."""
        return self.index.size()
    
    def __repr__(self) -> str:
        return (
            f"VectorDatabase(dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"index_type={self.index_type}, "
            f"size={len(self)})"
        )
    
    def _should_rebuild(self) -> bool:
        """
        Determine if index should be rebuilt based on strategy.
        
        Returns:
            True if index should be rebuilt, False for incremental add
        """
        # Always build if index is empty
        if not self.index.is_built:
            return True
        
        # TODO: Implement threshold-based rebuilding
        # For now, always rebuild (eager strategy)
        return True
