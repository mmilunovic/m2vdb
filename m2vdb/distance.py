# m2vdb/distance.py

import numpy as np
import numba

@numba.njit(parallel=True)
def euclidean_distance(a, b):
    result = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    for i in numba.prange(a.shape[0]):
        for j in range(b.shape[0]):
            diff = a[i] - b[j]
            result[i, j] = np.dot(diff, diff)
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of vectors efficiently.
    
    Args:
        a: First set of vectors (n x d)
        b: Second set of vectors (m x d)
    Returns:
        Similarity matrix (n x m)
    """
    # Normalize vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - np.dot(a_norm, b_norm.T)

def ip_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Inner Product (IP) distance between two sets of vectors efficiently.
    IP distance is defined as the negative dot product between vectors.
    
    Args:
        a: First set of vectors (n x d)
        b: Second set of vectors (m x d)
    Returns:
        Distance matrix (n x m)
    """
    # Compute negative dot product
    return -np.dot(a, b.T)  # (n x m)
