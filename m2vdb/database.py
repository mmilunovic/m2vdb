# m2vdb/database.py

from m2vdb.index import BruteForceIndex
from m2vdb.storage import InMemoryStore
import numpy as np

class VectorDatabase:
    def __init__(self, dim, metric='euclidean', storage_path="data"):
        self.index = BruteForceIndex(dim, metric)
        self.store = InMemoryStore()
        self.dim = dim
        self.storage_path = storage_path

    def add(self, vectors, metadata_list=None, ids=None):
        vectors = np.array(vectors).astype('float32')
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(vectors))]

        for vec, meta in zip(vectors, metadata_list):
            self.store.add(vec, meta)

        self.index.add(vectors, ids=ids)

    def search(self, queries, k=10):
        return self.index.search(queries, k)

    def save(self):
        self.store.save(self.storage_path)

    def load(self):
        self.store.load(self.storage_path)
        vectors = np.array(self.store.vectors).astype('float32')
        self.index = BruteForceIndex(self.dim)  # re-init index
        self.index.add(vectors)
