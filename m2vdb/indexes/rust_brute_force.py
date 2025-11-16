# indexes/rust_brute_force.py

from typing import List, Tuple
import numpy as np

from .base import Index
import rust_indexes  # this is the Rust extension module you just built


class RustBruteForceIndex(Index):
    """
    Brute force nearest neighbor index implemented in Rust.

    This is a thin wrapper around rust_indexes.BruteForceIndex so that
    it fits into the same Index interface as the pure Python version.
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

        # Convert NumPy → Python list-of-lists → Rust Vec<Vec<f32>>
        vectors_list: List[List[float]] = vectors.astype(np.float32).tolist()
        self._inner.build(vectors_list, ids)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self.is_built or k == 0:
            return []

        query_list: List[float] = query.astype(np.float32).tolist()
        # Rust returns List[Tuple[str, float]]
        return self._inner.search(query_list, int(k))

    def add(self, id: str, vector: np.ndarray) -> None:
        if not self.is_built:
            raise RuntimeError("Index must be built before adding vectors. Call build() first.")

        vector_list: List[float] = vector.astype(np.float32).tolist()
        self._inner.add(id, vector_list)

    def delete(self, id: str) -> bool:
        return self._inner.delete(id)

    def size(self) -> int:
        return self._inner.size()
