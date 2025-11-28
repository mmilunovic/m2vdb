"""
Index implementations for m2vdb.
"""

from .base import Index
from .brute_force import BruteForceIndex
from .pq import PQIndex
from .ivf import IVFIndex

# Optional Rust index - only available if rust_indexes is installed
try:
    from .rust_brute_force import RustBruteForceIndex
    HAS_RUST = True
except ImportError:
    RustBruteForceIndex = None
    HAS_RUST = False

__all__ = [
    "Index",
    "BruteForceIndex",
    "PQIndex",
    "IVFIndex",
    "HAS_RUST",
]

# Only export RustBruteForceIndex if available
if HAS_RUST:
    __all__.append("RustBruteForceIndex")
