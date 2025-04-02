# m2vdb/index.py

import numpy as np
from m2vdb.distance import euclidean_distance, cosine_similarity

class BruteForceIndex:
    def __init__(self, dim, metric='euclidean'):
        self.dim = dim
        self.metric = metric
        self.vectors = []
        self.ids = []
        self._metric_fn = euclidean_distance if metric == 'euclidean' else cosine_similarity

    def add(self, vecs, ids=None):
        vecs = np.array(vecs)
        if ids is None:
            ids = list(range(len(self.vectors), len(self.vectors) + len(vecs)))
        self.vectors.append(vecs)
        self.ids.extend(ids)

    def search(self, queries, k=10):
        queries = np.array(queries)
        all_vectors = np.vstack(self.vectors)
        dists = self._metric_fn(queries, all_vectors)
        idx = np.argsort(dists, axis=1)[:, :k]
        return [[self.ids[i] for i in row] for row in idx]
