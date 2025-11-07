"""
FastAPI server for VectorLite vector database.

This module provides a REST API for creating indexes, inserting vectors,
searching, and deleting vectors. It's designed to be simple and easy to
understand while providing the core functionality of a vector database.

The server keeps all indexes in memory for now. Later you can add persistence
by saving indexes to disk periodically or on shutdown, and loading them on startup.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import numpy as np
from contextlib import asynccontextmanager

from m2vdb.index import VectorIndex, SearchResult

class CreateIndexRequest(BaseModel):
    """Request to create a new index."""
    name: str = Field(..., description="Unique name for the index", min_length=1)
    dimension: int = Field(..., description="Dimensionality of vectors", gt=0)
    metric: str = Field(default="cosine", description="Distance metric (cosine or euclidean)")
    search_algorithm: str = Field(default="brute_force", description="Search algorithm to use")
    
    @validator('metric')
    def validate_metric(cls, v):
        if v not in ['cosine', 'euclidean']:
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        return v
    
    @validator('search_algorithm')
    def validate_algorithm(cls, v):
        if v not in ['brute_force', 'pq', 'hnsw']:
            raise ValueError("search_algorithm must be 'brute_force', 'pq', or 'hnsw'")
        return v


class CreateIndexResponse(BaseModel):
    """Response after creating an index."""
    name: str
    dimension: int
    metric: str
    search_algorithm: str
    message: str


class VectorInsert(BaseModel):
    """A single vector to insert."""
    id: str = Field(..., description="Unique identifier for this vector")
    vector: List[float] = Field(..., description="The vector values")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class InsertVectorsRequest(BaseModel):
    """Request to insert one or more vectors."""
    vectors: List[VectorInsert] = Field(..., description="List of vectors to insert", min_items=1)


class InsertVectorsResponse(BaseModel):
    """Response after inserting vectors."""
    inserted_count: int
    message: str


class SearchRequest(BaseModel):
    """Request to search for nearest neighbors."""
    vector: List[float] = Field(..., description="Query vector")
    k: int = Field(default=10, description="Number of nearest neighbors to return", gt=0)
    return_metadata: bool = Field(default=True, description="Whether to include metadata in results")


class SearchResultResponse(BaseModel):
    """A single search result."""
    id: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response containing search results."""
    results: List[SearchResultResponse]
    query_time_ms: float


class DeleteVectorRequest(BaseModel):
    """Request to delete one or more vectors."""
    ids: List[str] = Field(..., description="List of vector IDs to delete", min_items=1)


class DeleteVectorResponse(BaseModel):
    """Response after deleting vectors."""
    deleted_count: int
    message: str


class IndexInfo(BaseModel):
    """Information about an index."""
    name: str
    dimension: int
    metric: str
    search_algorithm: str
    size: int
    is_built: bool


class ListIndexesResponse(BaseModel):
    """Response listing all indexes."""
    indexes: List[IndexInfo]


class BuildIndexResponse(BaseModel):
    """Response after building an index."""
    name: str
    size: int
    message: str


# ============================================================================
# Global State
# ============================================================================

# Dictionary mapping index names to VectorIndex instances
# This is our in-memory storage of all indexes
# In a production system, you'd want to persist these to disk and implement
# proper concurrency control, but for your learning project this is fine
indexes: Dict[str, VectorIndex] = {}


# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    
    This function runs when the server starts up and shuts down. You can use
    it to load indexes from disk on startup and save them on shutdown. For now
    it's just a placeholder that yields control to the application.
    """
    # Startup: could load indexes from disk here
    print("VectorLite server starting up...")
    print(f"Loaded {len(indexes)} indexes from memory")
    
    yield  # Server runs here
    
    # Shutdown: could save indexes to disk here
    print("VectorLite server shutting down...")
    print(f"Had {len(indexes)} indexes in memory")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="VectorLite",
    description="A simple vector database for learning and experimentation",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from web browsers
# This is necessary if you want to build a web UI that runs on a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint that returns basic information about the server."""
    return {
        "name": "VectorLite",
        "version": "0.1.0",
        "description": "A simple vector database",
        "endpoints": {
            "create_index": "POST /indexes",
            "list_indexes": "GET /indexes",
            "get_index": "GET /indexes/{name}",
            "delete_index": "DELETE /indexes/{name}",
            "insert_vectors": "POST /indexes/{name}/vectors",
            "search": "POST /indexes/{name}/search",
            "delete_vectors": "DELETE /indexes/{name}/vectors",
            "build_index": "POST /indexes/{name}/build"
        }
    }


@app.post("/indexes", response_model=CreateIndexResponse, status_code=status.HTTP_201_CREATED)
async def create_index(request: CreateIndexRequest):
    """
    Create a new vector index.
    
    The index starts empty and unbuild. You'll need to insert vectors and then
    call the build endpoint before you can search. This separation allows you
    to bulk-load large datasets efficiently.
    """
    if request.name in indexes:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Index '{request.name}' already exists"
        )
    
    try:
        index = VectorIndex(
            dimension=request.dimension,
            metric=request.metric,
            search_algorithm=request.search_algorithm
        )
        indexes[request.name] = index
        
        return CreateIndexResponse(
            name=request.name,
            dimension=request.dimension,
            metric=request.metric,
            search_algorithm=request.search_algorithm,
            message=f"Index '{request.name}' created successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {str(e)}"
        )


@app.get("/indexes", response_model=ListIndexesResponse)
async def list_indexes():
    """
    List all indexes currently in the database.
    
    This returns metadata about each index including its size and whether
    it has been built yet. Useful for understanding what's in your database
    and for debugging.
    """
    index_list = []
    for name, index in indexes.items():
        index_list.append(IndexInfo(
            name=name,
            dimension=index.dimension,
            metric=index.metric,
            search_algorithm=index.search_algorithm_name,
            size=index.size(),
            is_built=index._is_built
        ))
    
    return ListIndexesResponse(indexes=index_list)


@app.get("/indexes/{name}", response_model=IndexInfo)
async def get_index(name: str):
    """
    Get detailed information about a specific index.
    
    Returns metadata like dimension, metric, algorithm, current size, and
    build status. This is useful for checking the state of an index before
    performing operations on it.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    index = indexes[name]
    return IndexInfo(
        name=name,
        dimension=index.dimension,
        metric=index.metric,
        search_algorithm=index.search_algorithm_name,
        size=index.size(),
        is_built=index._is_built
    )


@app.delete("/indexes/{name}")
async def delete_index(name: str):
    """
    Delete an index and all its data.
    
    This is irreversible, so use with caution. In a production system you'd
    want to add confirmation or soft-delete mechanisms, but for your learning
    project this simple approach is fine.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    del indexes[name]
    return {"message": f"Index '{name}' deleted successfully"}


@app.post("/indexes/{name}/vectors", response_model=InsertVectorsResponse)
async def insert_vectors(name: str, request: InsertVectorsRequest):
    """
    Insert one or more vectors into an index.
    
    If the index hasn't been built yet, vectors are added to a pending buffer.
    If it has been built, vectors are immediately added to the search index and
    become searchable right away. This dual behavior lets you bulk-load data
    efficiently while also supporting incremental updates.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    index = indexes[name]
    inserted_count = 0
    errors = []
    
    for vec_insert in request.vectors:
        try:
            # Convert list to numpy array
            vector = np.array(vec_insert.vector, dtype=np.float32)
            
            # Insert into index
            index.insert(
                id=vec_insert.id,
                vector=vector,
                metadata=vec_insert.metadata
            )
            inserted_count += 1
            
        except ValueError as e:
            # Collect errors but continue processing other vectors
            errors.append(f"Failed to insert '{vec_insert.id}': {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error inserting '{vec_insert.id}': {str(e)}")
    
    # If we had errors but also some successes, return partial success
    if errors and inserted_count > 0:
        return InsertVectorsResponse(
            inserted_count=inserted_count,
            message=f"Inserted {inserted_count} vectors with {len(errors)} errors. Errors: {'; '.join(errors[:5])}"
        )
    
    # If we had only errors, raise an exception
    if errors and inserted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to insert any vectors. Errors: {'; '.join(errors[:5])}"
        )
    
    # All successes
    return InsertVectorsResponse(
        inserted_count=inserted_count,
        message=f"Successfully inserted {inserted_count} vectors"
    )


@app.post("/indexes/{name}/build", response_model=BuildIndexResponse)
async def build_index(name: str):
    """
    Build the search index from all inserted vectors.
    
    This creates the data structures needed for efficient search. For brute
    force this is quick, but for HNSW it can take a while on large datasets.
    You must call this before searching if you've inserted vectors.
    
    After building, subsequent vector insertions go directly into the search
    index rather than a pending buffer, so they're immediately searchable.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    index = indexes[name]
    
    try:
        index.build()
        return BuildIndexResponse(
            name=name,
            size=index.size(),
            message=f"Index '{name}' built successfully with {index.size()} vectors"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build index: {str(e)}"
        )


@app.post("/indexes/{name}/search", response_model=SearchResponse)
async def search(name: str, request: SearchRequest):
    """
    Search for nearest neighbors in an index.
    
    This is the core operation of your vector database. It finds the k vectors
    most similar to your query vector according to the configured distance metric.
    Returns results sorted by distance with the closest vectors first.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    index = indexes[name]
    
    try:
        # Convert query list to numpy array
        query_vector = np.array(request.vector, dtype=np.float32)
        
        # Measure search time
        import time
        start_time = time.time()
        
        # Perform search
        results = index.search(
            query=query_vector,
            k=request.k,
            return_metadata=request.return_metadata
        )
        
        end_time = time.time()
        query_time_ms = (end_time - start_time) * 1000
        
        # Convert to response format
        response_results = [
            SearchResultResponse(
                id=result.id,
                distance=result.distance,
                metadata=result.metadata
            )
            for result in results
        ]
        
        return SearchResponse(
            results=response_results,
            query_time_ms=query_time_ms
        )
        
    except RuntimeError as e:
        # This catches the "index not built" error from VectorIndex
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query vector: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.delete("/indexes/{name}/vectors", response_model=DeleteVectorResponse)
async def delete_vectors(name: str, request: DeleteVectorRequest):
    """
    Delete one or more vectors from an index by their IDs.
    
    This removes the vectors from the index so they won't appear in future
    search results. The operation is immediate - if the index is built, the
    vectors are removed from the search structures right away.
    """
    if name not in indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
    
    index = indexes[name]
    deleted_count = 0
    
    for vec_id in request.ids:
        if index.delete(vec_id):
            deleted_count += 1
    
    return DeleteVectorResponse(
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} of {len(request.ids)} vectors"
    )


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint for monitoring.
    
    Returns the number of indexes and basic server status. Useful for
    deployment tools and monitoring systems to verify the server is running.
    """
    return {
        "status": "healthy",
        "indexes_count": len(indexes),
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the server when this file is executed directly
    # In production you'd use a proper ASGI server with multiple workers
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )