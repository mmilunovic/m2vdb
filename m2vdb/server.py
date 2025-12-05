"""
FastAPI server for m2vdb vector database.

Provides a REST API for managing vector indexes and performing similarity search.
Supports multi-tenant access with API key authentication.
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
from contextlib import asynccontextmanager
import time
from pathlib import Path
import os
 
from .collection import Collection
from .storage import CollectionManager
from .models import (
    CreateIndexRequest,
    IndexInfo,
    UpsertRequest,
    UpsertResponse,
    SearchRequest,
    SearchResponse,
    DeleteResponse,
    FetchResponse,
    CollectionNotFound,
)


security = HTTPBearer()

API_KEYS = {
    "sk-test-user1": "user1",
    "sk-test-user2": "user2"
}

DATA_ROOT = Path(os.getenv("M2VDB_DATA_DIR", Path(__file__).parent.parent / "data")) / "collections"
collection_manager = CollectionManager(DATA_ROOT)


def get_current_user(auth: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate API key and return user ID."""
    api_key = auth.credentials
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return API_KEYS[api_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    total_indexes = sum(
        len(collection_manager.list_collections(user_id))
        for user_id in API_KEYS.values()
    )
    print(f"m2vdb server starting... {total_indexes} indexes on disk")
    yield
    total_indexes = sum(
        len(collection_manager.list_collections(user_id))
        for user_id in API_KEYS.values()
    )
    print(f"m2vdb server shutting down... {total_indexes} indexes on disk")


app = FastAPI(
    title="m2vdb",
    description="Vector database with REST API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/indexes/{name}", response_model=IndexInfo, status_code=status.HTTP_201_CREATED)
async def create_index(
    name: str, 
    request: CreateIndexRequest,
    user_id: str = Depends(get_current_user)
):
    """Create a new index."""
    try:
        db = collection_manager.create_collection(
            user_id=user_id,
            name=name,
            dimension=request.dimension,
            metric=request.metric,
            index_type=request.index_type,
            index_params=request.index_params or {}
        )
        
        return IndexInfo(
            name=name,
            dimension=db.dimension,
            metric=db.metric,
            index_type=db.index_type,
            size=0
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))


@app.get("/indexes")
async def list_indexes(user_id: str = Depends(get_current_user)):
    """List all indexes for the authenticated user."""
    index_names = collection_manager.list_collections(user_id)
    
    indexes_info = []
    for name in index_names:
        db = collection_manager.get_collection(user_id, name)
        indexes_info.append(IndexInfo(
            name=name,
            dimension=db.dimension,
            metric=db.metric,
            index_type=db.index_type,
            size=len(db)
        ))
    
    return {"indexes": indexes_info}


@app.get("/indexes/{name}", response_model=IndexInfo)
async def get_index(name: str, user_id: str = Depends(get_current_user)):
    """Get index info."""
    try:
        db = collection_manager.get_collection(user_id, name)
        return IndexInfo(
            name=name,
            dimension=db.dimension,
            metric=db.metric,
            index_type=db.index_type,
            size=len(db)
        )
    except CollectionNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")


@app.delete("/indexes/{name}")
async def delete_index(name: str, user_id: str = Depends(get_current_user)):
    """Delete an index."""
    try:
        collection_manager.delete_collection(user_id, name)
        return {"message": f"Index '{name}' deleted"}
    except CollectionNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")


def _get_index(name: str, user_id: str) -> Collection:
    """Helper to get index or raise 404."""
    try:
        return collection_manager.get_collection(user_id, name)
    except CollectionNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")


@app.post("/indexes/{name}/vectors", response_model=UpsertResponse)
async def upsert_vectors(
    name: str, 
    request: UpsertRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Upsert (insert/update) vectors.
    
    Uses batch_upsert internally for optimal performance:
    - All vectors are stored first
    - Index is rebuilt/trained once at the end
    - Much faster for PQ indexes (1 rebuild instead of N rebuilds)
    """
    _get_index(name, user_id)  # Verify index exists
    
    try:
        # Extract all vectors from request
        ids = [vec.id for vec in request.vectors]
        vectors = [np.array(vec.vector, dtype=np.float32) for vec in request.vectors]
        metadata = [vec.metadata for vec in request.vectors]
        
        # Use batch_upsert for optimal performance
        upserted = collection_manager.batch_upsert(user_id, name, ids, vectors, metadata)
        
        return UpsertResponse(upserted_count=upserted)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/indexes/{name}/search", response_model=SearchResponse)
async def search_vectors(
    name: str, 
    request: SearchRequest,
    user_id: str = Depends(get_current_user)
):
    """Search for similar vectors."""
    db = _get_index(name, user_id)
    
    try:
        query = np.array(request.vector, dtype=np.float32)
        
        start = time.perf_counter()
        results = db.search(query, request.k, request.include_metadata)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            matches=results,
            query_time_ms=elapsed_ms
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.delete("/indexes/{name}/vectors/{id}", response_model=DeleteResponse)
async def delete_vector(
    name: str,
    id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a single vector by ID."""
    db = _get_index(name, user_id)
    
    deleted = db.delete(id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector '{id}' not found"
        )
    
    # Auto-save after mutation
    collection_manager._save_collection(user_id, name, db)
    
    return DeleteResponse(deleted_count=1)


@app.get("/indexes/{name}/vectors/{id}", response_model=FetchResponse)
async def fetch_vector(
    name: str, 
    id: str,
    user_id: str = Depends(get_current_user)
):
    """Fetch a vector by ID."""
    db = _get_index(name, user_id)
    
    result = db.fetch(id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Vector '{id}' not found")
    
    vector, metadata = result
    return FetchResponse(id=id, vector=vector.tolist(), metadata=metadata)


@app.get("/")
async def root():
    """API info."""
    total_indexes = sum(
        len(collection_manager.list_collections(user_id))
        for user_id in API_KEYS.values()
    )
    return {
        "name": "m2vdb",
        "version": "0.1.0",
        "total_indexes": total_indexes,
        "users": len(API_KEYS)
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/stats")
async def stats(user_id: str = Depends(get_current_user)):
    """Resource usage for the authenticated user."""
    total_vectors = 0
    total_memory_mib = 0.0
    total_disk_mib = 0.0
    index_details = []
    
    index_names = collection_manager.list_collections(user_id)
    
    for name in index_names:
        db = collection_manager.get_collection(user_id, name)
        db_stats = db.get_stats()
        
        total_vectors += db_stats["num_vectors"]
        total_memory_mib += db_stats["memory_mib"]["total"]
        
        if "disk_mib" in db_stats:
            total_disk_mib += db_stats["disk_mib"]["total"]
        
        index_details.append({
            "name": name,
            "type": db_stats["index_type"],
            "dimension": db_stats["dimension"],
            "metric": db_stats["metric"],
            "num_vectors": db_stats["num_vectors"],
            "memory_mib": db_stats["memory_mib"],
            "disk_mib": db_stats.get("disk_mib")
        })
    
    return {
        "user": user_id,
        "indexes": {
            "total": len(index_names),
            "details": index_details
        },
        "vectors": {
            "total": total_vectors
        },
        "memory_mib": round(total_memory_mib, 2),
        "disk_mib": round(total_disk_mib, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
