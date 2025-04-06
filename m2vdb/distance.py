# m2vdb/distance.py

import numpy as np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance between two sets of vectors efficiently.
    Uses the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    
    Args:
        a: First set of vectors (n x d)
        b: Second set of vectors (m x d)
    Returns:
        Distance matrix (n x m)
    """
    # Compute squared norms
    a_norm_sq = np.sum(a**2, axis=1, keepdims=True)  # (n x 1)
    b_norm_sq = np.sum(b**2, axis=1)  # (m,)
    
    # Compute dot product
    dot_product = np.dot(a, b.T)  # (n x m)
    
    # Return squared distances
    return a_norm_sq + b_norm_sq - 2 * dot_product

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
