"""Inverted file index implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from m2vdb.core import BaseIndex


class IVFIndex(BaseIndex):
    """Inverted file index supporting approximate nearest neighbour search."""

    def __init__(self, dim: int, metric: str = "euclidean", **kwargs) -> None:
        super().__init__(dim, metric)
        self.n_clusters = kwargs.pop("n_clusters", 64)
        self.n_probe = kwargs.pop("n_probe", 5)
        self.random_seed = kwargs.pop("random_seed", 1312)

        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected parameters for IVFIndex: {unexpected}")

        self.ids: List[int] = []
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[tuple[int, np.ndarray]]] = defaultdict(list)
        self._vector_map: Dict[int, np.ndarray] = {}
        self._is_trained = False

    def train(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")

        if self.metric == "cosine":
            vecs_norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = np.nan_to_num(vecs_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed, n_init="auto")
        kmeans.fit(vecs)
        self.centroids = kmeans.cluster_centers_
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

        if self.centroids is None:
            raise RuntimeError("IVFIndex must be trained before adding vectors")

        distances = self._metric_fn(vecs, self.centroids)
        cluster_ids = np.argmin(distances, axis=1)

        for i, cluster_id in enumerate(cluster_ids):
            self.inverted_lists[cluster_id].append((ids[i], vecs[i]))
            self._vector_map[ids[i]] = vecs[i]

        if metadata:
            self.metadata.update(metadata)

        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected queries of shape (n, {self.dim}), got {queries.shape}")

        if self.centroids is None or not self.inverted_lists:
            return (
                np.zeros((len(queries), k), dtype=np.int64),
                np.zeros((len(queries), k), dtype=np.float32),
            )

        n_queries = queries.shape[0]
        indices = np.full((n_queries, k), -1, dtype=np.int64)
        scores = np.full((n_queries, k), np.inf, dtype=np.float32)

        all_centroid_dists = self._metric_fn(queries, self.centroids)
        n_probe = min(self.n_probe, self.n_clusters)
        top_clusters_all = np.argpartition(all_centroid_dists, n_probe, axis=1)[:, :n_probe]

        for qi, q in enumerate(queries):
            top_clusters = top_clusters_all[qi]
            candidates: List[int] = []
            candidate_vecs: List[np.ndarray] = []
            for cid in top_clusters:
                entries = self.inverted_lists.get(int(cid))
                if not entries:
                    continue
                for vec_id, vec in entries:
                    candidates.append(vec_id)
                    candidate_vecs.append(vec)

            if not candidates:
                continue

            candidate_vecs_array = np.stack(candidate_vecs)
            dists = self._metric_fn(q[None, :], candidate_vecs_array)[0]
            k_actual = min(k, len(candidates))
            top_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            order = np.argsort(dists[top_idx])
            top_idx = top_idx[order]

            indices[qi, :k_actual] = np.array(candidates, dtype=np.int64)[top_idx]
            scores[qi, :k_actual] = dists[top_idx]

        return indices, scores

    def search_with_metadata(self, queries: np.ndarray, k: int = 10):
        ids, scores = self.search(queries, k)
        return ids, scores, [[self.metadata.get(int(i)) for i in row] for row in ids]
