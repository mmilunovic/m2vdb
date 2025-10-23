"""m2vdb – educational vector database playground."""

from .database import VectorDatabase, V3cT0rDaTaBas3
from .indexes import (
    BruteForceIndex,
    IVFIndex,
    PQIndex,
    create_index,
    get_index_class,
    index_name_for,
    register_index,
    registered_indexes,
)

__all__ = [
    "VectorDatabase",
    "V3cT0rDaTaBas3",
    "BruteForceIndex",
    "IVFIndex",
    "PQIndex",
    "create_index",
    "get_index_class",
    "index_name_for",
    "register_index",
    "registered_indexes",
]
