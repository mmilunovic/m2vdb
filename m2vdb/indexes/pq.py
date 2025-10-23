"""Product quantization index implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from m2vdb.core import BaseIndex
from m2vdb.quantization import ProductQuantizer


class PQIndex(BaseIndex):
    """Product quantisation index for compressed search."""

    def __init__(self, dim: int, metric: str = "euclidean", **kwargs) -> None:
        super().__init__(dim, metric)
        self.num_subspaces = kwargs.pop("num_subspaces", 4)
        self.centroids_per_subspace = kwargs.pop("centroids_per_subspace", 256)
        seed = kwargs.pop("seed", 42)
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for PQIndex: {unexpected}")

        self.pq = ProductQuantizer(
            dim=dim,
            num_subspaces=self.num_subspaces,
            centroids_per_subspace=self.centroids_per_subspace,
            seed=seed,
        )
        self.codes: Optional[np.ndarray] = None
        self.ids: List[int] = []
        self._is_trained = False

    def train(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        self.pq.fit(vecs)
        self._is_trained = True

    def add(
        self,
        vecs: np.ndarray,
        ids: Optional[List[int]] = None,
        metadata: Optional[Dict[int, Dict]] = None,
    ) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")

        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(vecs)))

        if not self._is_trained:
            self.train(vecs)

        new_codes = self.pq.encode(vecs)
        if self.codes is None:
            self.codes = new_codes
        else:
            self.codes = np.vstack([self.codes, new_codes])

        self.ids.extend(ids)

        if metadata:
            self.metadata.update(metadata)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected queries of shape (n, {self.dim}), got {queries.shape}")

        if self.codes is None or not self.ids:
            return (
                np.zeros((len(queries), k), dtype=np.int64),
                np.zeros((len(queries), k), dtype=np.float32),
            )

        n_queries = queries.shape[0]
        n_db = self.codes.shape[0]
        lookup_tables = self.pq.build_lookup_table(queries)

        dists = np.zeros((n_queries, n_db), dtype=np.float32)
        for m in range(self.num_subspaces):
            dists += lookup_tables[:, m][:, self.codes[:, m]]

        k = min(k, n_db)
        top_k_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        sorted_top_k_idx = top_k_idx[rows, np.argsort(dists[rows, top_k_idx], axis=1)]

        return np.array(self.ids)[sorted_top_k_idx], dists[rows, sorted_top_k_idx]

    def search_with_metadata(self, queries: np.ndarray, k: int = 10):
        ids, scores = self.search(queries, k)
        return ids, scores, [[self.metadata.get(int(i)) for i in row] for row in ids]
