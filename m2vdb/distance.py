
# m2vdb/distance.py

import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a[:, None] - b[None, :], axis=2)

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - np.dot(a_norm, b_norm.T)
