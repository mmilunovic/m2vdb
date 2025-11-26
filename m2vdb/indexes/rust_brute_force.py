from typing import List, Tuple
import numpy as np

from .base import Index
import rust_indexes


class RustBruteForceIndex(Index):
    """
    Brute force nearest neighbor index implemented in Rust.

    This is a thin wrapper around rust_indexes.BruteForceIndex that uses
    zero-copy NumPy array access for maximum performance.
    """

    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        # This calls the #[new] constructor on the PyO3 class in Rust
        self._inner = rust_indexes.BruteForceIndex(metric)

    @property
    def is_built(self) -> bool:
        return self._inner.is_built

    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
            )

        # Pass NumPy array directly - Rust will read it via zero-copy
        # Ensure it's float32 and C-contiguous for optimal performance
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        self._inner.build(vectors_f32, ids)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self.is_built or k == 0:
            return []

        # Pass NumPy array directly - no conversion needed!
        query_f32 = np.ascontiguousarray(query, dtype=np.float32)
        return self._inner.search(query_f32, int(k))

    def add(self, id: str, vector: np.ndarray) -> None:
        if not self.is_built:
            raise RuntimeError("Index must be built before adding vectors. Call build() first.")

        # Pass NumPy array directly
        vector_f32 = np.ascontiguousarray(vector, dtype=np.float32)
        self._inner.add(id, vector_f32)

    def delete(self, id: str) -> bool:
        return self._inner.delete(id)

    def size(self) -> int:
        return self._inner.size()
