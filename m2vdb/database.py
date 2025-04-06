# m2vdb/database.py

from typing import List, Dict, Any, Optional
import numpy as np
import os
from m2vdb.index import BruteForceIndex, IVFIndex
from m2vdb.storage import IndexManager, FileStorage

class V3cT0rDaTaBas3:
    """Vector database with support for exact and approximate nearest neighbor search"""
    
    def __init__(self, dim: int = None, index_type: str = 'brute_force', 
                 storage_path: str = "data", load_existing: bool = False, **kwargs):
        """
        Initialize a vector database with flexible configuration
        
        Args:
            dim: Dimensionality of vectors (required if creating a new index)
            index_type: Type of index to use ('brute_force' or 'ann', default='brute_force')
            storage_path: Path to store the database (default='data')
            load_existing: Whether to load an existing index from storage_path (default=False)
            **kwargs: Additional configuration options for index:
                # Index-specific parameters:
                metric: Distance metric ('euclidean' or 'cosine', default='euclidean')
                
                # For IVFIndex:
                # TODO: Add IVFIndex parameters
                num_candidates: Number of candidates to sample (default=100)
                random_seed: Random seed for sampling (default=1312)
        """
        # Store parameters
        self.index_type = index_type
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
            # Create the appropriate index
            if index_type == 'brute_force':
                self.index = BruteForceIndex(dim=dim, **kwargs)
            elif index_type == 'ivf':
                self.index = IVFIndex(dim=dim, **kwargs)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

    def add(self, vectors: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None, 
            ids: Optional[List[int]] = None) -> None:
        """
        Add vectors and optional metadata to the database
        
        Args:
            vectors: Vectors to add
            metadata_list: Optional metadata for each vector
            ids: Optional IDs for the vectors
        """
        vectors = np.array(vectors).astype('float32')
        
        # Verify dimensions match
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimensionality mismatch: expected {self.dim}, got {vectors.shape[1]}")
        
        # Handle metadata
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(vectors))]
        
        # When adding vectors to an existing database, ensure new metadata aligns with IDs
        if ids is None:
            # Calculate where new IDs will start based on the current index state
            start_id = len(self.index.ids) if hasattr(self.index, 'ids') and self.index.ids else 0
            ids = list(range(start_id, start_id + len(vectors)))
        
        # Add metadata so it will align with IDs
        if len(self.metadata) < max(ids) + 1:
            # Extend metadata list with empty dicts if needed to match IDs
            self.metadata.extend([{}] * (max(ids) + 1 - len(self.metadata)))
        
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
        # Get nearest neighbor IDs from index
        result_ids = self.index.search(queries, k=k)
        
        # Format results with metadata
        results = []
        for query_results in result_ids:
            query_formatted = []
            for id_val in query_results:
                # If we have metadata for this ID, include it
                metadata = self.metadata[id_val] if id_val < len(self.metadata) else {}
                query_formatted.append({
                    "id": id_val,
                    "metadata": metadata
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
        self.storage_manager.storage.save_metadata(self.metadata, metadata_path)
           
            
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
            self.metadata = self.storage_manager.storage.load_metadata(metadata_path)
        else:
            self.metadata = []