"""
m2vdb: A simple, fast vector database in Python.
"""

from .client import M2VDBClient
from .database import VectorDatabase
from .models import SearchResult, Vector
from .indexes import Index, BruteForceIndex, PQIndex, RustBruteForceIndex

__version__ = "0.1.0"

__all__ = [
    "M2VDBClient",
    "VectorDatabase",
    "SearchResult",
    "Vector",
    "Index",
    "BruteForceIndex",
    "PQIndex",
    "RustBruteForceIndex"
]
