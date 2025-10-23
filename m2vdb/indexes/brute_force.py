"""Brute force index implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from m2vdb.core import BaseIndex


class BruteForceIndex(BaseIndex):
    """Exact nearest neighbour search using brute force scans."""

    def __init__(self, dim: int, metric: str = "euclidean", **kwargs) -> None:
        super().__init__(dim, metric)

        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for BruteForceIndex: {unexpected}")

        self._vectors_array = np.zeros((0, dim), dtype=np.float32)
        self.ids: List[int] = []

    def add(
        self,
        vecs: np.ndarray,
        ids: Optional[List[int]] = None,
        metadata: Optional[Dict[int, Dict]] = None,
    ) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (n, {self.dim}), got {vecs.shape}")

        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(vecs)))

        new_size = len(self.ids) + len(vecs)
        new_array = np.empty((new_size, self.dim), dtype=np.float32)
        if self.ids:
            new_array[: len(self.ids)] = self._vectors_array
        new_array[len(self.ids) :] = vecs

        self._vectors_array = new_array
        self.ids.extend(ids)

        if metadata:
            self.metadata.update(metadata)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected queries of shape (n, {self.dim}), got {queries.shape}")

        if not self.ids:
            return (
                np.zeros((len(queries), k), dtype=np.int64),
                np.zeros((len(queries), k), dtype=np.float32),
            )

        dists = self._metric_fn(queries, self._vectors_array)
        k = min(k, len(self.ids))
        partition_k = k if k < len(self.ids) else k - 1
        idx = np.argpartition(dists, partition_k, axis=1)[:, :k]
        rows = np.arange(len(queries))[:, None]
        idx = idx[rows, np.argsort(dists[rows, idx])]
        scores = np.take_along_axis(dists, idx, axis=1)

        return np.array(self.ids)[idx], scores
