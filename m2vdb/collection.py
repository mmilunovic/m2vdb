"""
High-level vector collection API.
"""

# TODO: Index build/rebuild strategy improvements
# - Decide if we should expose /train endpoint to users for manual control
# - Implement 'threshold' rebuild strategy (rebuild every N vectors)
# - Implement 'manual' rebuild strategy (only rebuild on explicit .build() call)
# - Consider batch upsert optimization (add N vectors, rebuild once)
# - Consider async/background rebuild for large indexes
# Related: Currently only 'eager' (rebuild on every upsert) is implemented

from typing import List, Optional, Dict, Any
import numpy as np

from .indexes import Index, BruteForceIndex, PQIndex, IVFIndex, HAS_RUST
from .models import SearchResult

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
        index_params: Optional[Dict[str, Any]] = None,
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
        self.index = self._create_index(index_type, metric, self.index_params)
        
        # Storage for vectors and metadata (always in memory for 1M vectors)
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
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
        """
        Insert or update a vector in the collection.
        
        WARNING: For PQ indexes with many vectors, this is inefficient because it rebuilds
        the index on every insert. For bulk loading, collect your vectors and call
        batch_upsert() instead, or wait until you have >= n_clusters vectors before upserting.
        """
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
            # Incremental add to existing index (only works if index already built)
            self.index.add(id, vector)
    
    def batch_upsert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> int:
        """
        Batch insert/update multiple vectors efficiently.
        
        This is the PROPER way to load many vectors at once, especially for
        indexes like PQ that need training data. Stores all vectors first,
        then rebuilds index once at the end.
        
        Args:
            ids: List of unique string IDs
            vectors: List of numpy arrays with shape (dimension,)
            metadata: Optional list of metadata dicts (None entries allowed)
        
        Returns:
            Number of vectors upserted
        
        Example:
            >>> db.batch_upsert(
            ...     ids=["id1", "id2"],
            ...     vectors=[np.array([...]), np.array([...])],
            ...     metadata=[{"category": "image"}, {"category": "text"}]
            ... )
        """
        if len(ids) != len(vectors):
            raise ValueError(f"ids and vectors must have same length ({len(ids)} vs {len(vectors)})")
        
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError(f"metadata must have same length as ids ({len(metadata)} vs {len(ids)})")
        
        count = 0
        for i, (id, vector) in enumerate(zip(ids, vectors)):
            assert vector.shape == (self.dimension,), \
                f"Vector {i} dimension {vector.shape} doesn't match {self.dimension}"
            
            if id in self._vectors:
                raise ValueError(f"ID '{id}' already exists. Delete first to update.")
            
            # Store vector and metadata (no rebuild yet)
            self._vectors[id] = vector
            if metadata is not None and metadata[i] is not None:
                self._metadata[id] = metadata[i]
            count += 1
        
        # Now rebuild once with all the new vectors
        self._rebuild_index()
        
        return count
    
    def _rebuild_index(self) -> None:
        """
        Rebuild the entire index from stored vectors.
        
        Called based on rebuild strategy. For PQ, this retrains k-means.
        For BruteForce, this just reorganizes the array.
        
        TODO: HACK ALERT - PQ training minimum samples check
        PQ index requires n_samples >= n_clusters for k-means training.
        Currently we skip rebuild if not enough samples, which means:
        - Search will fail or return empty results until we have enough vectors
        - Vectors are still stored in _vectors, just not indexed yet
        - Not ideal for production use
        
        Proper solutions to implement:
        1. ✅ Batch API: batch_upsert() method implemented above.
           This is the RIGHT way - store all vectors, rebuild once.
        2. Manual build: Add rebuild_strategy='manual' + explicit build() method
           so users control when to train. Good for bulk loading workflows.
        3. Lazy training: Store vectors without index, train on first search.
           Risky - search could be very slow unexpectedly.
        4. Fallback index: Use BruteForce until enough samples for PQ.
           Complex but provides smooth UX.
        
        For now: Skip rebuild if not enough samples (testing hack).
        Vectors are stored and will be indexed once we hit the threshold.
        """
        if len(self._vectors) == 0:
            return
        
        # HACK: Check if we have enough samples for PQ training
        if self.index_type == 'pq':
            n_clusters = self.index_params.get('n_clusters', 256)
            if len(self._vectors) < n_clusters:
                # Not enough samples to train PQ - skip rebuild
                # Vectors are stored, will be indexed when we have enough
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
    
    def fetch(self, id: str) -> Optional[tuple[np.ndarray, dict]]:
        """
        Fetch a vector by ID.
        
        Returns:
            Tuple of (vector, metadata_dict) if found, None if not found.
            metadata_dict is empty {} if no metadata was stored.
        """
        if id not in self._vectors:
            return None
        return self._vectors[id], self._metadata.get(id, {})
    
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
        # Always build if index is empty (first vector)
        if not self.index.is_built:
            return True
        
        # For PQ: check if we have enough samples for initial training
        # Once trained, use incremental add
        if self.index_type == 'pq':
            n_clusters = self.index_params.get('n_clusters', 256)
            # If index not built yet and we now have enough samples, rebuild
            if len(self._vectors) >= n_clusters and not self.index.is_built:
                return True
            # Otherwise use incremental add (or skip if not enough samples yet)
            return False
        
        # For other index types (BruteForce, IVF):
        # Rebuild on every upsert for now (eager strategy)
        # TODO: Implement threshold-based rebuilding
        return False  # Use incremental add for better performance
    
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


