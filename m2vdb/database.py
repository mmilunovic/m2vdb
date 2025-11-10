"""
High-level vector database API.
"""

from typing import List, Optional, Dict, Any
import numpy as np
<<<<<<< HEAD

from .index import Index, BruteForceIndex
from .models import SearchResult


class VectorDatabase:
    """
    Vector database with metadata support and pluggable index backends.
=======
import os
from m2vdb.indexes import create_index, registered_indexes
from m2vdb.storage import IndexManager, FileStorage

class VectorDatabase:
    """High-level facade that owns an index and optional metadata."""
>>>>>>> bc1fe633faa3e73c7863b2fb5fc417afdcf871fb
    
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
<<<<<<< HEAD
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.index = self._create_index(index_type, metric)
        
    def _create_index(self, index_type: str, metric: str) -> Index:
        """Factory for index implementations."""
        if index_type == 'brute_force':
            return BruteForceIndex(metric=metric)
        elif index_type == 'pq':
            raise NotImplementedError("Product Quantization not yet implemented")
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
=======
        self.storage_path = storage_path
        self.metadata = []

        # Initialize storage manager
        self.storage_manager = IndexManager(FileStorage())
        
        # Either load existing index or create a new one
        if load_existing:
            self.load()
        else:
            if dim is None:
                raise ValueError("'dim' parameter is required when creating a new index")
                
            self.dim = dim
            if index_type not in registered_indexes():
                raise ValueError(
                    f"Unknown index type: {index_type}. Available: {', '.join(registered_indexes())}"
                )
            self.index = create_index(index_type, dim=dim, **kwargs)

    def add(self, vectors: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None, 
            ids: Optional[List[int]] = None) -> None:
        """
        Add vectors and optional metadata to the database
>>>>>>> bc1fe633faa3e73c7863b2fb5fc417afdcf871fb
        
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
        
<<<<<<< HEAD
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
=======
        # Set metadata for each vector at the corresponding ID index
        for i, id_val in enumerate(ids):
            self.metadata[id_val] = metadata_list[i]
            
        # Add to index
        self.index.add(vectors, ids=ids)

    def search(self, queries: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors and return results with metadata
        
        Args:
            queries: Query vectors
            k: Number of results to return
            
        Returns:
            List of results with IDs, distances, and metadata
        """
        # Get nearest neighbor IDs and scores from index
        result_ids, scores = self.index.search(queries, k=k)
        
        # Format results with metadata
        results = []
        for query_results, query_scores in zip(result_ids, scores):
            query_formatted = []
            for id_val, score in zip(query_results, query_scores):
                # If we have metadata for this ID, include it
                metadata = self.metadata[id_val] if id_val < len(self.metadata) else {}
                query_formatted.append({
                    "id": int(id_val),
                    "metadata": metadata,
                    "score": float(score)
                })
            results.append(query_formatted)
            
        return results

    def save(self) -> None:
        """
        Persist the entire database state to disk
        
        This high-level method saves the complete database state, including:
        1. The vector index (using IndexManager)
        2. All metadata associated with vectors
        """
        # Save index (vectors, IDs, and index configuration)  
        self.storage_manager.save_index(self.index, self.storage_path)
        
        # Save metadata separately
        metadata_path = f"{self.storage_path}/metadata.json"
        self.storage_manager.storage.save_vector_metadata(self.metadata, metadata_path)
           
            
    def load(self) -> None:
        """Load the database from storage path"""
        # Check if files exist
        config_file = os.path.join(self.storage_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No index found at {self.storage_path}")
            
        # Load index
        self.index = self.storage_manager.load_index(self.storage_path)
        self.dim = self.index.dim

        # Try to load metadata if it exists
        metadata_path = f"{self.storage_path}/metadata.json"
        if os.path.exists(metadata_path):
            self.metadata = self.storage_manager.storage.load_vector_metadata(metadata_path)
        else:
            self.metadata = []


# Backwards compatibility with the original whimsical name
V3cT0rDaTaBas3 = VectorDatabase
>>>>>>> bc1fe633faa3e73c7863b2fb5fc417afdcf871fb
