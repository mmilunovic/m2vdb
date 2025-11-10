"""Persistence helpers for vector indexes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from m2vdb.core import BaseIndex
from m2vdb.indexes import create_index, get_index_class, index_name_for


def _should_memmap(path: str, threshold_gb: float = 1.0) -> bool:
    size_gb = os.path.getsize(path) / (1024 ** 3)
    return size_gb > threshold_gb


class BaseStorage:
    """Abstract base class for storage implementations."""

    def save_vectors(self, vectors: np.ndarray, path: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def load_vectors(self, path: str, memmap: Optional[bool] = None) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def save_config(self, config: Dict[str, Any], path: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def load_config(self, path: str) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def save_vector_metadata(self, metadata: Any, path: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def load_vector_metadata(self, path: str) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class FileStorage(BaseStorage):
    """File-based storage implementation using NumPy arrays and JSON."""

    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vectors)

    def load_vectors(self, path: str, memmap: Optional[bool] = None) -> np.ndarray:
        if memmap is None:
            memmap = _should_memmap(path)
        return np.load(path, mmap_mode="r" if memmap else None)

    def save_config(self, config: Dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_inverted_lists(self, inverted: Dict[int, List[int]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in inverted.items()}, f)

    def load_inverted_lists(self, path: str) -> Dict[int, List[int]]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def save_vector_metadata(self, metadata: Any, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def load_vector_metadata(self, path: str) -> Any:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): v for k, v in data.items()}
        return data


@dataclass
class IndexSerializer:
    save: Callable[[BaseIndex, "IndexManager", str], Dict[str, Any]]
    load: Callable[["IndexManager", str, Dict[str, Any]], BaseIndex]


_SERIALIZERS: Dict[Type[BaseIndex], IndexSerializer] = {}


def register_index_serializer(index_cls: Type[BaseIndex], serializer: IndexSerializer) -> None:
    _SERIALIZERS[index_cls] = serializer


def _serializer_for(index: BaseIndex) -> IndexSerializer:
    index_cls = type(index)
    if index_cls not in _SERIALIZERS:  # pragma: no cover - defensive
        raise ValueError(f"No serializer registered for {index_cls.__name__}")
    return _SERIALIZERS[index_cls]


def _serializer_for_name(index_key: str) -> Tuple[Type[BaseIndex], IndexSerializer]:
    index_cls = get_index_class(index_key)
    if index_cls not in _SERIALIZERS:  # pragma: no cover - defensive
        raise ValueError(f"No serializer registered for index '{index_key}'")
    return index_cls, _SERIALIZERS[index_cls]


class IndexManager:
    """Manager class for saving and loading indexes."""

    def __init__(self, storage: Optional[BaseStorage] = None):
        self.storage = storage or FileStorage()

    def save_index(self, index: BaseIndex, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        base_config = {
            "index_key": index_name_for(type(index)),
            "index_type": index.__class__.__name__,
            "dim": index.dim,
            "metric": index.metric,
            "ids": getattr(index, "ids", None),
        }

        serializer = _serializer_for(index)
        extra_config = serializer.save(index, self, path)
        config = {**base_config, **extra_config}
        self.storage.save_config(config, os.path.join(path, "config.json"))

    def load_index(self, path: str) -> BaseIndex:
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("Missing config.json")

        config = self.storage.load_config(config_path)
        index_key = config["index_key"]
        _, serializer = _serializer_for_name(index_key)
        return serializer.load(self, path, config)


def _save_brute_force(index: BaseIndex, manager: IndexManager, path: str) -> Dict[str, Any]:
    assert hasattr(index, "_vectors_array")
    manager.storage.save_vectors(index._vectors_array, os.path.join(path, "vectors.npy"))
    if getattr(index, "metadata", None):
        manager.storage.save_vector_metadata(index.metadata, os.path.join(path, "metadata.json"))
    return {}


def _load_brute_force(manager: IndexManager, path: str, config: Dict[str, Any]) -> BaseIndex:
    index = create_index(config["index_key"], dim=config["dim"], metric=config["metric"])
    vectors = manager.storage.load_vectors(os.path.join(path, "vectors.npy"))
    index._vectors_array = vectors
    ids = config.get("ids") or list(range(len(vectors)))
    index.ids = ids
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        index.metadata = manager.storage.load_vector_metadata(meta_path)
    return index


def _save_ivf(index: BaseIndex, manager: IndexManager, path: str) -> Dict[str, Any]:
    if getattr(index, "centroids", None) is not None:
        manager.storage.save_vectors(index.centroids, os.path.join(path, "centroids.npy"))
    inverted_map: Dict[int, List[int]] = {}
    all_vecs = []
    all_ids: List[int] = []

    for cluster_id, entries in index.inverted_lists.items():
        cluster_ids: List[int] = []
        for vec_id, vec in entries:
            all_ids.append(vec_id)
            all_vecs.append(vec)
            cluster_ids.append(vec_id)
        inverted_map[cluster_id] = cluster_ids

    if all_vecs:
        manager.storage.save_vectors(np.stack(all_vecs), os.path.join(path, "vectors.npy"))
        manager.storage.save_vectors(np.array(all_ids, dtype=np.int64), os.path.join(path, "ids.npy"))
    manager.storage.save_inverted_lists(inverted_map, os.path.join(path, "inverted_lists.json"))

    if getattr(index, "metadata", None):
        manager.storage.save_vector_metadata(index.metadata, os.path.join(path, "vector_metadata.json"))

    return {
        "n_clusters": index.n_clusters,
        "n_probe": index.n_probe,
        "is_trained": index._is_trained,
    }


def _load_ivf(manager: IndexManager, path: str, config: Dict[str, Any]) -> BaseIndex:
    required = ["n_clusters", "n_probe", "is_trained"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config fields for IVFIndex: {missing}")

    index = create_index(
        config["index_key"],
        dim=config["dim"],
        metric=config["metric"],
        n_clusters=config["n_clusters"],
        n_probe=config["n_probe"],
    )
    index._is_trained = config["is_trained"]
    if os.path.exists(os.path.join(path, "centroids.npy")):
        index.centroids = manager.storage.load_vectors(os.path.join(path, "centroids.npy"))

    vectors_path = os.path.join(path, "vectors.npy")
    ids_path = os.path.join(path, "ids.npy")
    if os.path.exists(vectors_path) and os.path.exists(ids_path):
        vectors = manager.storage.load_vectors(vectors_path)
        ids = manager.storage.load_vectors(ids_path).tolist()
        index.ids = ids
        index._vector_map = {id_: vec for id_, vec in zip(ids, vectors)}
        inverted_raw = manager.storage.load_inverted_lists(os.path.join(path, "inverted_lists.json"))
        index.inverted_lists = {cid: [(vid, index._vector_map[vid]) for vid in vids] for cid, vids in inverted_raw.items()}

    meta_path = os.path.join(path, "vector_metadata.json")
    if os.path.exists(meta_path):
        index.metadata = manager.storage.load_vector_metadata(meta_path)

    return index


def _save_pq(index: BaseIndex, manager: IndexManager, path: str) -> Dict[str, Any]:
    if index.pq.codebooks:
        codebooks = np.stack(index.pq.codebooks)
        manager.storage.save_vectors(codebooks, os.path.join(path, "codebooks.npy"))
    if index.codes is not None:
        manager.storage.save_vectors(index.codes, os.path.join(path, "codes.npy"))
    if getattr(index, "metadata", None):
        manager.storage.save_vector_metadata(index.metadata, os.path.join(path, "metadata.json"))
    return {
        "num_subspaces": index.num_subspaces,
        "centroids_per_subspace": index.centroids_per_subspace,
        "is_trained": index._is_trained,
    }


def _load_pq(manager: IndexManager, path: str, config: Dict[str, Any]) -> BaseIndex:
    index = create_index(
        config["index_key"],
        dim=config["dim"],
        metric=config["metric"],
        num_subspaces=config["num_subspaces"],
        centroids_per_subspace=config["centroids_per_subspace"],
    )
    index._is_trained = config.get("is_trained", False)
    codebooks_path = os.path.join(path, "codebooks.npy")
    if os.path.exists(codebooks_path):
        codebooks = manager.storage.load_vectors(codebooks_path)
        index.pq.codebooks = [codebooks[i] for i in range(codebooks.shape[0])]
    codes_path = os.path.join(path, "codes.npy")
    if os.path.exists(codes_path):
        index.codes = manager.storage.load_vectors(codes_path)
    index.ids = config.get("ids", []) or []
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        index.metadata = manager.storage.load_vector_metadata(meta_path)
    return index


register_index_serializer(get_index_class("brute_force"), IndexSerializer(save=_save_brute_force, load=_load_brute_force))
register_index_serializer(get_index_class("ivf"), IndexSerializer(save=_save_ivf, load=_load_ivf))
register_index_serializer(get_index_class("pq"), IndexSerializer(save=_save_pq, load=_load_pq))

__all__ = [
    "BaseStorage",
    "FileStorage",
    "IndexManager",
    "register_index_serializer",
]
