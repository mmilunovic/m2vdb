<<<<<<< HEAD
"""
m2vdb: A simple, fast vector database in Python.
"""

from .client import M2VDBClient
from .database import VectorDatabase
from .models import SearchResult, Vector
from .index import Index, BruteForceIndex

__version__ = "0.1.0"

__all__ = [
    "M2VDBClient",
    "VectorDatabase",
    "SearchResult",
    "Vector",
    "Index",
    "BruteForceIndex",
=======
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
>>>>>>> bc1fe633faa3e73c7863b2fb5fc417afdcf871fb
]
