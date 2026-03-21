"""
Custom exceptions for m2vdb.
"""


class M2VDBError(Exception):
    """Base exception for all m2vdb errors."""
    pass


class CollectionNotFoundError(M2VDBError):
    """Raised when a collection does not exist."""
    pass


class DuplicateCollectionError(M2VDBError):
    """Raised when trying to create a collection that already exists."""
    pass


class DimensionMismatchError(M2VDBError):
    """Raised when vector dimensions don't match the collection's dimension."""
    pass


class IndexNotBuiltError(M2VDBError):
    """Raised when attempting operations on an unbuilt index."""
    pass


# Backwards compatibility alias
CollectionNotFound = CollectionNotFoundError
