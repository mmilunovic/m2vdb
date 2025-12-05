
from typing import List, Optional, Dict
import numpy as np

from .base import Index

class BruteForceIndex(Index):
    """
    Brute force nearest neighbor search.
    
    This is the simplest possible approach - compute distances to every vector
    and return the k smallest. It's O(n*d) for each query where n is the number
    of vectors and d is dimensionality.
    
    While slow for large datasets, brute force has several advantages:
    - It's perfectly accurate - no approximation errors
    - It handles dynamic insertions and deletions with zero overhead
    - For small datasets (under 10k vectors), it's actually competitive

    It also serves as the baseline for benchmarking faster algorithms.

    The implementation maintains bidirectional ID-to-index mappings for O(1)
    lookups in both directions, making operations like delete efficient.
    """
    
    def __init__(self, metric: str = 'cosine'):
        """
        Initialize brute force search.
        
        Args:
            metric: distance metric to use ('cosine' or 'euclidean')
        """
        self.metric = metric
        
        # Store vectors as a numpy array for efficient vectorized distance computation
        self.vectors: Optional[np.ndarray] = None
        
        # Store IDs as a list where position i corresponds to vectors[i]
        # This serves as our idx_to_id mapping
        self.ids: List[str] = []
        
        # Bidirectional mapping for O(1) lookups
        # This dictionary maps each ID to its position in the vectors array
        # Without this, operations like delete would require O(n) linear search
        self._id_to_idx: Dict[str, int] = {}
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built (has vectors)."""
        return self.vectors is not None and len(self.vectors) > 0
        
    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Store the initial set of vectors and IDs.
        
        This builds both the forward mapping (id to index) and sets up the
        reverse mapping (index to id) through the ids list. The dual mapping
        makes all ID-based operations fast.
        """
        assert len(ids) == vectors.shape[0], \
            f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        assert len(set(ids)) == len(ids), "Duplicate IDs found in the input"
        
        # Copy to prevent external modifications - numpy arrays are mutable and passed by reference,
        # so without copying, external changes to the input arrays could corrupt our index
        self.vectors = vectors.copy()
        self.ids = ids.copy()
        
        # Build the id_to_idx mapping for O(1) lookups
        self._id_to_idx = {id: idx for idx, id in enumerate(ids)}
        
    def search(self, query: np.ndarray, k: int) -> List[tuple[str, float]]:
        """
        Compute distances to all vectors and return the k nearest.
        
        The idx_to_id conversion at the end is O(1) per result because we just
        index into the ids list, which is the whole reason we maintain that list
        in parallel with the vectors array.
        """
        if self.vectors is None or len(self.vectors) == 0 or k == 0:
            return []
        
        # Compute distances based on the metric
        if self.metric == 'cosine':
            # Cosine similarity: (A·B)/(||A||·||B||)
            # We convert to distance: 1 - similarity
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                query_norm = 1e-10  # Avoid division by zero
            
            vector_norms = np.linalg.norm(self.vectors, axis=1)
            vector_norms = np.where(vector_norms == 0, 1e-10, vector_norms)
            
            # Dot product divided by norms gives cosine similarity
            similarities = np.dot(self.vectors, query) / (vector_norms * query_norm)
            distances = 1 - similarities
        elif self.metric == 'euclidean':
            # L2 distance: sqrt(sum((A-B)^2))
            # We don't need the sqrt since we're only comparing distances
            distances = np.linalg.norm(self.vectors - query, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Find the k smallest distances efficiently
        # We use argpartition instead of argsort because it's O(n) instead of O(n log n)
        # when k << n, which is the common case
        k = min(k, len(distances))
        
        # argpartition puts the k smallest elements at the front but doesn't sort them
        partition_indices = np.argpartition(distances, k-1)[:k]
        
        # Now sort just those k elements
        sorted_indices = partition_indices[np.argsort(distances[partition_indices])]
        
        # Convert indices to IDs using our idx_to_id mapping (the ids list)
        # This is O(1) per result because list indexing is constant time
        results = [
            (self.ids[idx], float(distances[idx]))
            for idx in sorted_indices
        ]
        
        return results
    
    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a new vector to the index.
        
        We update both mappings to keep them synchronized. The new vector goes
        at the end of the array, and we record its position in id_to_idx.
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before adding vectors. Call build() first.")
            
        if id in self._id_to_idx:
            raise ValueError(f"ID '{id}' already exists in the index")
        
        # Determine the index where this vector will live
        new_idx = len(self.ids)
        
        # Append to the vectors array
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1).copy()
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        # Update both mappings
        self.ids.append(id)
        self._id_to_idx[id] = new_idx
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from the index in O(1) time.
        
        Uses swap-and-pop: swap the deleted element with the last element,
        then remove the last element. This avoids shifting all subsequent elements.
        Trade-off: vectors are no longer in insertion order, but for similarity
        search this doesn't matter.
        """
        if id not in self._id_to_idx:
            return False
        
        idx = self._id_to_idx[id]
        last_idx = len(self.ids) - 1
        
        # If deleting the last element, no swap needed
        if idx == last_idx:
            self.vectors = self.vectors[:-1]
            self.ids.pop()
            del self._id_to_idx[id]
            return True
        
        # Swap with last element, then pop
        last_id = self.ids[last_idx]
        
        # Swap in vectors array
        self.vectors[idx] = self.vectors[last_idx]
        self.vectors = self.vectors[:-1]
        
        # Swap in ids list
        self.ids[idx] = last_id
        self.ids.pop()
        
        # Update mappings: last element moved to deleted position
        self._id_to_idx[last_id] = idx
        del self._id_to_idx[id]
        
        return True
    
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return len(self.ids)
    
    def memory_usage(self) -> int:
        """Calculate memory usage of the index structures in bytes."""
        if self.vectors is None:
            return 0
        # BruteForce stores the full vectors array as its index
        return self.vectors.nbytes
    
    def save_artifacts(self, artifacts_dir: str) -> None:
        """
        Save trained artifacts to disk.
        
        Brute force has no trainable components (no codebooks, centroids, etc.),
        so this method does nothing. Implemented for consistency with the Index interface.
        
        Args:
            artifacts_dir: Directory to save artifacts to (unused)
        """
        pass
    
    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Load trained artifacts from disk.
        
        Brute force has no trainable components (no codebooks, centroids, etc.),
        so this method does nothing. Implemented for consistency with the Index interface.
        
        Args:
            artifacts_dir: Directory to load artifacts from (unused)
        """
        pass
    