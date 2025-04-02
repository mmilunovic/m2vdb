# m2vdb/storage.py

import numpy as np
import json
import os

class InMemoryStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector, meta=None):
        self.vectors.append(vector)
        self.metadata.append(meta or {})

    def save(self, path="data"):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), np.array(self.vectors))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)

    def load(self, path="data"):
        self.vectors = np.load(os.path.join(path, "vectors.npy")).tolist()
        with open(os.path.join(path, "metadata.json")) as f:
            self.metadata = json.load(f)
