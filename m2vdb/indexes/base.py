from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Index(ABC):
    """
    Abstract base class for vector index implementations.
    """
    
    @abstractmethod
    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build the search index structure from a batch of vectors.
        
        This is typically called once after loading your initial dataset. For
        algorithms like HNSW that have expensive index construction, this is
        where that work happens. For brute force, this just stores the vectors.
        
        Args:
            vectors: numpy array of shape (n, dim) containing all vectors
            ids: list of string IDs corresponding to each vector
        """
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[tuple[str, float]]:
        """
        Search for k nearest neighbors of the query vector.
        
        The index searches its internal data structures and returns results
        as ID-distance pairs. The IDs are the same string IDs that were passed
        during build() or add().
        
        Args:
            query: query vector of shape (dim,)
            k: number of nearest neighbors to return
            
        Returns:
            List of (id, distance) tuples sorted by distance (closest first)
        """
        pass
    
    @abstractmethod
    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a single vector to the index after initial build.
        
        This allows incremental updates to the index. Different algorithms
        handle this with different efficiency:
        - Brute force: O(1) append operation
        - Product Quantization: need to quantize and add to codebook regions
        - HNSW: need to insert into graph with link updates
        
        Args:
            id: unique string ID for this vector
            vector: vector of shape (dim,) to add
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the index by ID.
        
        Args:
            id: the ID of the vector to delete
            
        Returns:
            True if the vector was found and deleted, False otherwise
        """
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors currently in the index."""
        pass
