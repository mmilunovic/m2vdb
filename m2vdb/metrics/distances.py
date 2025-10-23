"""Vector distance utilities used by indexes and benchmarks."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba
except ImportError:  # pragma: no cover - optional dependency
    numba = None


if numba is not None:  # pragma: no cover - exercised in environments with numba

    @numba.njit(parallel=True)
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the squared L2 distance matrix between ``a`` and ``b``."""

        result = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
        for i in numba.prange(a.shape[0]):
            for j in range(b.shape[0]):
                diff = a[i] - b[j]
                result[i, j] = np.dot(diff, diff)
        return result

else:

    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.einsum("ijk,ijk->ij", diff, diff).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine distance (1 - similarity) between ``a`` and ``b``."""

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - np.dot(a_norm, b_norm.T)


def ip_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the negative dot product between ``a`` and ``b``."""

    return -np.dot(a, b.T)


__all__ = ["euclidean_distance", "cosine_similarity", "ip_distance"]
