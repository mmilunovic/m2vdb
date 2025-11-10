"""Product quantisation primitives used by PQ-based indexes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ProductQuantizer:
    def __init__(self, dim: int, num_subspaces: int = 4, centroids_per_subspace: int = 256, seed: int = 42):
        if dim % num_subspaces != 0:
            raise ValueError("Dimensionality must be divisible by num_subspaces.")

        self.dim = dim
        self.num_subspaces = num_subspaces
        self.centroids_per_subspace = centroids_per_subspace
        self.subdim = dim // num_subspaces
        self.codebooks: List[np.ndarray] = []
        self.seed = seed

    def fit(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        self.codebooks = []

        def fit_one_subspace(m: int) -> np.ndarray:
            subvecs = vecs[:, m * self.subdim : (m + 1) * self.subdim]
            kmeans = KMeans(n_clusters=self.centroids_per_subspace, random_state=self.seed + m, n_init="auto")
            kmeans.fit(subvecs)
            return kmeans.cluster_centers_

        with ThreadPoolExecutor() as pool:
            self.codebooks = list(pool.map(fit_one_subspace, range(self.num_subspaces)))

    def encode(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        n = vecs.shape[0]
        codes = np.empty((n, self.num_subspaces), dtype=np.uint8)

        for m in range(self.num_subspaces):
            subvecs = vecs[:, m * self.subdim : (m + 1) * self.subdim]
            centroids = self.codebooks[m]
            dists = np.linalg.norm(subvecs[:, None, :] - centroids[None, :, :], axis=2)
            codes[:, m] = np.argmin(dists, axis=1)

        return codes

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        n = len(codes)
        vecs = np.empty((n, self.dim), dtype=np.float32)

        for m in range(self.num_subspaces):
            centroids = self.codebooks[m]
            vecs[:, m * self.subdim : (m + 1) * self.subdim] = centroids[codes[:, m]]

        return vecs

    def build_lookup_table(self, queries: np.ndarray) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = queries.shape[0]
        lookup = np.empty((n_queries, self.num_subspaces, self.centroids_per_subspace), dtype=np.float32)

        for m in range(self.num_subspaces):
            query_subvecs = queries[:, m * self.subdim : (m + 1) * self.subdim]
            centroids = self.codebooks[m]
            diff = query_subvecs[:, None, :] - centroids[None, :, :]
            lookup[:, m, :] = np.linalg.norm(diff, axis=2)

        return lookup


__all__ = ["ProductQuantizer"]
