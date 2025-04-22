# m2vdb/storage.py

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from m2vdb.index import BaseIndex, BruteForceIndex, IVFIndex


def _should_memmap(path, threshold_gb=1.0):
    size_gb = os.path.getsize(path) / (1024 ** 3)
    return size_gb > threshold_gb

class BaseStorage(ABC):
    """Abstract base class for storage implementations"""
    
    @abstractmethod
    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        """Save vectors to storage"""
        pass
    
    @abstractmethod
    def load_vectors(self, path: str, memmap: Optional[bool] = None) -> np.ndarray:
        """Load vectors from storage"""
        pass

    @abstractmethod
    def save_config(self, config: Dict[str, Any], path: str) -> None:
        """Save config to storage"""
        pass
    
    @abstractmethod
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load config from storage"""
        pass

    @abstractmethod
    def save_vector_metadata(self, metadata: Dict[int, Dict], path: str) -> None:
        """Save vector metadata to storage"""
        pass
    
    @abstractmethod
    def load_vector_metadata(self, path: str) -> Dict[int, Dict]:
        """Load vector metadata from storage"""
        pass
    
    
class FileStorage(BaseStorage):
    """File-based storage implementation"""
    
    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vectors)
    
    def load_vectors(self, path: str, memmap: Optional[bool] = None) -> np.ndarray:
        if memmap is None:
            memmap = _should_memmap(path)
        return np.load(path, mmap_mode="r" if memmap else None)

    # Config is fine as-is
    def save_config(self, config: Dict[str, Any], path: str) -> None:
        with open(path, "w") as f:
            json.dump(config, f)

    def load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    # Inverted lists: cluster_id (int) → List[int]
    def save_inverted_lists(self, inverted: Dict[int, List[int]], path: str) -> None:
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in inverted.items()}, f)

    def load_inverted_lists(self, path: str) -> Dict[int, List[int]]:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    # Vector metadata: vector_id (int) → Dict[str, Any]
    def save_vector_metadata(self, metadata: Dict[int, Dict], path: str) -> None:
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in metadata.items()}, f)

    def load_vector_metadata(self, path: str) -> Dict[int, Dict]:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

class IndexManager:
    """Manager class for saving and loading indexes"""
    
    def __init__(self, storage=None):
        # TODO: What if I want to use cloud instead of local storage?
        # from m2vdb.storage import FileStorage
        self.storage = storage or FileStorage()

    def save_index(self, index: BaseIndex, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        config = {
            "index_type": index.__class__.__name__,
            "dim": index.dim,
            "metric": index.metric,
            "ids": index.ids if hasattr(index, "ids") else None,
        }

        if isinstance(index, BruteForceIndex):
            self.storage.save_vectors(index._vectors_array, os.path.join(path, "vectors.npy"))

        elif isinstance(index, IVFIndex):
            # Save IVF metadata
            config["n_clusters"] = index.n_clusters
            config["n_probe"] = index.n_probe
            config["is_trained"] = index._is_trained

            # Save config, this is ok to be json
            self.storage.save_config(config, os.path.join(path, "config.json"))

            # Save metadata if it exists
            if index.metadata:
                self.storage.save_vector_metadata(index.metadata, os.path.join(path, "vector_metadata.json"))


            # Save centroids
            self.storage.save_vectors(index.centroids, os.path.join(path, "centroids.npy"))

            # Flatten vectors and ids from inverted lists
            all_vecs = []
            all_ids = []
            inverted_map = {}

            for cluster_id, entries in index.inverted_lists.items():
                cluster_ids = []
                for vec_id, vec in entries:
                    all_ids.append(vec_id)
                    all_vecs.append(vec)
                    cluster_ids.append(vec_id)
                inverted_map[str(cluster_id)] = cluster_ids

            all_vecs = np.stack(all_vecs)
            self.storage.save_vectors(all_vecs, os.path.join(path, "vectors.npy"))
            self.storage.save_vectors(np.array(all_ids, dtype=np.int64), os.path.join(path, "ids.npy"))
            self.storage.save_inverted_lists(inverted_map, os.path.join(path, "inverted_lists.json"))

        else:
            raise ValueError(f"Unsupported index type: {index.__class__.__name__}")

        

    def load_index(self, path: str) -> BaseIndex:
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("Missing config.json")

        config = self.storage.load_config(config_path)
        index_type = config.pop("index_type")
        dim = config.pop("dim")
        metric = config.pop("metric")
        ids = config.pop("ids", None)

        index = None

        if index_type == "BruteForceIndex":
            index = BruteForceIndex(dim=dim, metric=metric)
            vectors = self.storage.load_vectors(os.path.join(path, "vectors.npy"))
            index._vectors_array = vectors
            index.ids = ids or list(range(len(vectors)))

        elif index_type == "IVFIndex":
            # All IVF-specific params must be in config
            missing = [k for k in ("n_clusters", "n_probe", "is_trained") if k not in config]
            if missing:
                raise ValueError(f"Missing config fields for IVFIndex: {missing}")
            
            index = IVFIndex(dim=dim, metric=metric, **{
                "n_clusters": config["n_clusters"],
                "n_probe": config["n_probe"]
            })
            index._is_trained = config["is_trained"]

            # Load vectors and IDs
            vectors = self.storage.load_vectors(os.path.join(path, "vectors.npy"))
            ids = self.storage.load_vectors(os.path.join(path, "ids.npy")).tolist()
            index.ids = ids
            index._vector_map = {id_: vec for id_, vec in zip(ids, vectors)}

            # Load centroids
            index.centroids = self.storage.load_vectors(os.path.join(path, "centroids.npy"))

            # Load and reconstruct inverted lists
            raw_lists = self.storage.load_inverted_lists(os.path.join(path, "inverted_lists.json"))
            for cluster_id_str, id_list in raw_lists.items():
                cluster_id = int(cluster_id_str)
                for vec_id in id_list:
                    vec = index._vector_map.get(vec_id)
                    if vec is not None:
                        index.inverted_lists[cluster_id].append((vec_id, vec))

            # Load vector metadata if it exists
            meta_path = os.path.join(path, "vector_metadata.json")
            if os.path.exists(meta_path):
                index.metadata = self.storage.load_vector_metadata(meta_path)
            else:
                index.metadata = {}

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return index
