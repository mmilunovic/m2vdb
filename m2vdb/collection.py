"""
High-level vector collection API.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
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
        storage_path: Optional[str] = None,
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
            storage_path: Optional path where collection is persisted on disk (for stats calculation)
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.rebuild_strategy = rebuild_strategy
        self.index_params = index_params or {}
        self.storage_path = storage_path
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
        
        Args:
            ids: List of unique string IDs
            vectors: List of numpy arrays with shape (dimension,)
            metadata: Optional list of metadata dicts (None entries allowed)
        
        Returns:
            Number of vectors upserted
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
        """
        if len(self._vectors) == 0:
            return
        
        # TODO temporary: Check if we have enough samples for PQ training
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
        Get comprehensive statistics about this collection.
        
        Returns sizes in MiB (binary, 1024-based) which matches what OS shows.
        """
        # Memory usage
        vectors_bytes = sum(v.nbytes for v in self._vectors.values())
        index_bytes = self.index.memory_usage() if hasattr(self.index, 'memory_usage') else 0
        metadata_bytes = sum(sys.getsizeof(m) for m in self._metadata.values() if m)
        memory_total = vectors_bytes + index_bytes + metadata_bytes
        
        stats = {
            "num_vectors": len(self),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "memory_mib": {
                "vectors": round(vectors_bytes / (1024 ** 2), 2),
                "index": round(index_bytes / (1024 ** 2), 2),
                "metadata": round(metadata_bytes / (1024 ** 2), 2),
                "total": round(memory_total / (1024 ** 2), 2),
            }
        }
        
        # Disk usage (if persisted)
        if self.storage_path:
            storage_path = Path(self.storage_path)
            if storage_path.exists():
                vectors_file = storage_path / "vectors.npz"
                metadata_file = storage_path / "metadata.json"
                manifest_file = storage_path / "manifest.json"
                artifacts_dir = storage_path / "artifacts"
                
                vectors_disk = vectors_file.stat().st_size if vectors_file.exists() else 0
                metadata_disk = metadata_file.stat().st_size if metadata_file.exists() else 0
                manifest_disk = manifest_file.stat().st_size if manifest_file.exists() else 0
                
                artifacts_disk = 0
                if artifacts_dir.exists() and artifacts_dir.is_dir():
                    artifacts_disk = sum(f.stat().st_size for f in artifacts_dir.rglob('*') if f.is_file())
                
                disk_total = vectors_disk + metadata_disk + manifest_disk + artifacts_disk
                
                stats["disk_mib"] = {
                    "vectors": round(vectors_disk / (1024 ** 2), 2),
                    "metadata": round(metadata_disk / (1024 ** 2), 2),
                    "manifest": round(manifest_disk / (1024 ** 2), 2),
                    "index_artifacts": round(artifacts_disk / (1024 ** 2), 2),
                    "total": round(disk_total / (1024 ** 2), 2),
                }
        
        return stats


