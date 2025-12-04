"""
Pydantic models for the vector database API.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class CollectionNotFound(Exception):
    """Raised when a collection does not exist."""
    pass


class SearchResult(BaseModel):
    """Search result with ID, distance, and optional metadata."""
    id: str
    distance: float
    metadata: dict[str, Any] | None = None


class CreateIndexRequest(BaseModel):
    """Create a new index."""
    dimension: int = Field(gt=0, le=10000, description="Vector dimensionality")
    metric: Literal["cosine", "euclidean"] = Field(default="cosine", description="Distance metric")
    index_type: Literal["brute_force", "pq", "rust_brute_force", "ivf"] = Field(default="brute_force", description="Index implementation")
    index_params: dict[str, Any] | None = Field(default=None, description="Optional index-specific parameters (e.g., n_subvectors, n_clusters for PQ)")


class IndexInfo(BaseModel):
    """Index metadata."""
    name: str
    dimension: int
    metric: str
    index_type: str
    size: int


class Vector(BaseModel):
    """Single vector for upsert operations."""
    id: str = Field(min_length=1)
    vector: list[float]
    metadata: dict[str, Any] | None = None


class UpsertRequest(BaseModel):
    """Upsert one or more vectors."""
    vectors: list[Vector] = Field(min_length=1)


class UpsertResponse(BaseModel):
    """Upsert operation result."""
    upserted_count: int


class SearchRequest(BaseModel):
    """Search query."""
    vector: list[float]
    k: int = Field(default=10, gt=0)
    include_metadata: bool = Field(default=True)


class SearchResponse(BaseModel):
    """Search results."""
    matches: list[SearchResult]
    query_time_ms: float


class DeleteRequest(BaseModel):
    """Delete vectors by ID."""
    ids: list[str] = Field(min_length=1)


class DeleteResponse(BaseModel):
    """Delete operation result."""
    deleted_count: int


class FetchResponse(BaseModel):
    """Fetched vector data."""
    id: str
    vector: list[float]
    metadata: dict[str, Any] | None = None
