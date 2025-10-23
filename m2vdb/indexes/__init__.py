"""Index implementations and registry helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Type

from m2vdb.core import BaseIndex

from .brute_force import BruteForceIndex
from .ivf import IVFIndex
from .pq import PQIndex

_INDEX_REGISTRY: Dict[str, Type[BaseIndex]] = {}


def register_index(name: str, index_cls: Type[BaseIndex]) -> None:
    key = name.lower()
    _INDEX_REGISTRY[key] = index_cls


def get_index_class(name: str) -> Type[BaseIndex]:
    try:
        return _INDEX_REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown index '{name}'. Available: {sorted(_INDEX_REGISTRY)}") from exc


def create_index(name: str, *args, **kwargs) -> BaseIndex:
    index_cls = get_index_class(name)
    return index_cls(*args, **kwargs)


def registered_indexes() -> Iterable[str]:
    return sorted(_INDEX_REGISTRY)


def index_name_for(cls: Type[BaseIndex]) -> str:
    for name, registered_cls in _INDEX_REGISTRY.items():
        if registered_cls is cls:
            return name
    raise ValueError(f"Index class {cls.__name__} is not registered")


register_index("brute_force", BruteForceIndex)
register_index("bruteforce", BruteForceIndex)
register_index("bf", BruteForceIndex)
register_index("ivf", IVFIndex)
register_index("pq", PQIndex)

__all__ = [
    "BruteForceIndex",
    "IVFIndex",
    "PQIndex",
    "create_index",
    "get_index_class",
    "index_name_for",
    "register_index",
    "registered_indexes",
]
