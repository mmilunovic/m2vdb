"""
High-level vector collection API.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from .indexes import Index, BruteForceIndex, PQIndex, IVFIndex, HAS_RUST
from .models import SearchResult

# Import Rust index if available
if HAS_RUST:
    from .indexes import RustBruteForceIndex


class Collection:
    """
    Vector collection with metadata support and pluggable index backends.
    
    Separates concerns: Collection manages IDs/metadata/API,
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
            if not HAS_RUST:
                raise ValueError(
                    "Rust indexes not available. To enable Rust indexes:\n"
                    "  1. Install Rust: https://rustup.rs/\n"
                    "  2. Build extensions: cd rust && maturin develop --release\n"
                    "  3. Or use 'brute_force' index type instead"
                )
            return RustBruteForceIndex(metric=metric)
        elif index_type == 'pq':
            return PQIndex(metric=metric, **index_params)
        elif index_type == 'ivf':
            return IVFIndex(metric=metric, **index_params)
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
        """Insert or update a vector in the collection."""
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
        """Number of vectors in the collection."""
        return self.index.size()
    
    def __repr__(self) -> str:
        return (
            f"Collection(dimension={self.dimension}, "
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this collection instance.
        
        Returns a dictionary with:
        - num_vectors: total number of vectors
        - dimension: vector dimensionality
        - metric: distance metric used
        - index_type: type of search index
        - memory: breakdown of memory usage in bytes
            - vectors_bytes: raw vector storage
            - index_bytes: index data structures (PQ codebooks, IVF clusters, etc.)
            - metadata_bytes: stored metadata
            - total_bytes: sum of all above
        """
        import sys
        
        # TODO: This implementation only works when everything is in-memory!
        # When we implement persistence (disk-backed storage), we need to rewrite this.
        # We CANNOT load all vectors into RAM just to calculate stats - that defeats the purpose.
        # 
        # Future implementation should:
        # 1. Track memory usage incrementally as vectors are added/removed
        # 2. Store metadata sizes separately (or use os.path.getsize() on disk files)
        # 3. Use psutil or similar to get actual process memory usage
        # 4. For disk-backed vectors: track file sizes, not in-memory numpy array sizes
        # 5. Consider maintaining a running stats dict that updates on upsert/delete
        #    instead of recalculating from scratch every time
        
        # Calculate vectors memory (raw numpy arrays)
        vectors_bytes = sum(v.nbytes for v in self._vectors.values())
        
        # Calculate index memory (index-specific data structures)
        # For brute force: stores vectors internally
        # For PQ: codebooks + quantized codes
        # For IVF: cluster centroids + inverted lists
        index_bytes = 0
        if hasattr(self.index, 'memory_usage'):
            index_bytes = self.index.memory_usage()
        
        # Calculate metadata memory (Python dicts/objects)
        metadata_bytes = sum(
            sys.getsizeof(m) for m in self._metadata.values() if m
        )
        
        total_bytes = vectors_bytes + index_bytes + metadata_bytes
        
        return {
            "num_vectors": len(self),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "memory": {
                "vectors_bytes": vectors_bytes,
                "index_bytes": index_bytes,
                "metadata_bytes": metadata_bytes,
                "total_bytes": total_bytes,
                # Convenience MB conversions
                "vectors_mb": round(vectors_bytes / 1024 / 1024, 2),
                "index_mb": round(index_bytes / 1024 / 1024, 2),
                "metadata_mb": round(metadata_bytes / 1024 / 1024, 2),
                "total_mb": round(total_bytes / 1024 / 1024, 2),
            }
        }


