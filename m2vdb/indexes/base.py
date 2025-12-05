from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Index(ABC):
    """
    Abstract base class for vector index implementations.
    
    All indexes must track whether they have been built via the is_built property.
    This allows Collection to decide when to call build() vs add().
    """
    
    @property
    @abstractmethod
    def is_built(self) -> bool:
        """
        Check if the index has been built/trained.
        
        Returns:
            True if build() has been called and index is ready for use
        """
        pass
    
    @abstractmethod
    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Build/rebuild the search index structure from a batch of vectors.
        
        This is the primary way to construct the index. For PQ, this trains
        k-means clusters. For HNSW, this builds the graph. For brute force,
        this just stores the vectors.
        
        After calling this, is_built must return True.
        
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
        Add a single vector to a built index.
        
        This allows incremental updates after build(). Behavior:
        - If index not built: should raise RuntimeError
        - If index built: adds vector using existing structure
        
        Different algorithms handle this with different efficiency:
        - Brute force: O(1) append operation
        - Product Quantization: quantize using existing codebooks and append
        - HNSW: insert into graph with link updates
        
        Args:
            id: unique string ID for this vector
            vector: vector of shape (dim,) to add
            
        Raises:
            RuntimeError: if index not built (is_built == False)
            ValueError: if ID already exists
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
    
    def memory_usage(self) -> int:
        """
        Calculate approximate memory usage of index structures in bytes.
        
        This should NOT include the raw vectors themselves (those are tracked
        separately in Collection). Only count index-specific structures like:
        - PQ: codebooks, quantized codes
        - IVF: centroids, inverted lists
        - BruteForce: the vectors array (since it IS the index)
        
        Returns:
            Approximate memory usage in bytes
        """
        return 0  # Default implementation for indexes with no overhead
    
    def save_artifacts(self, artifacts_dir: str) -> None:
        """
        Save trained index artifacts (codebooks, centroids, etc.) to disk.
        
        Optional method for indexes that have trainable components.
        Default implementation does nothing (for indexes like brute_force).
        
        Args:
            artifacts_dir: Directory to save artifacts to
        """
        pass
    
    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Load trained index artifacts from disk.
        
        Optional method for indexes that have trainable components.
        Default implementation does nothing (for indexes like brute_force).
        
        Args:
            artifacts_dir: Directory to load artifacts from
        """
        pass
