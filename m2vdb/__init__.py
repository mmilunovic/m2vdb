"""
m2vdb - A simple vector database implementation.

This package provides vector database functionality with multiple index
implementations for efficient similarity search.
"""

from .database import VectorDatabase
from .models import SearchResult, Vector
from .index import Index, BruteForceIndex

__version__ = "0.1.0"

__all__ = [
    "VectorDatabase",
    "SearchResult",
    "Vector",
    "Index",
    "BruteForceIndex",
]
