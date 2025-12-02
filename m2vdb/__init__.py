"""
m2vdb: A simple, fast vector database in Python.
"""

from .client import M2VDBClient
from .collection import Collection
from .models import SearchResult, Vector
from .indexes import Index, BruteForceIndex, PQIndex, IVFIndex
from .storage import CollectionManager

__version__ = "0.1.0"

__all__ = [
    "M2VDBClient",
    "Collection",
    "SearchResult",
    "Vector",
    "Index",
    "BruteForceIndex",
    "PQIndex",
    "IVFIndex",
    "CollectionManager",
]

# Optional Rust index
try:
    from .indexes import RustBruteForceIndex, HAS_RUST  # noqa: F401
    __all__.extend(["RustBruteForceIndex", "HAS_RUST"])
except ImportError:
    HAS_RUST = False
