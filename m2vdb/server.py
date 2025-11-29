"""
FastAPI server for m2vdb vector database.

Pinecone-style API with multi-index support:
- Control plane: Create/list/delete indexes
- Data plane: Vector operations per index
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
from contextlib import asynccontextmanager
import time

from .collection import Collection
from .models import (
    CreateIndexRequest,
    IndexInfo,
    UpsertRequest,
    UpsertResponse,
    SearchRequest,
    SearchResponse,
    DeleteResponse,
    FetchResponse,
)


security = HTTPBearer()

API_KEYS = {
    "sk-test-user1": "user1",
    "sk-test-user2": "user2"
}

indexes: dict[str, dict[str, Collection]] = {
    "user1": {},
    "user2": {}
}

# Server startup time for uptime calculation
SERVER_START_TIME = time.time()


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
    total_indexes = sum(len(user_indexes) for user_indexes in indexes.values())
    print(f"m2vdb server starting... {total_indexes} indexes loaded across {len(indexes)} users")
    yield
    total_indexes = sum(len(user_indexes) for user_indexes in indexes.values())
    print(f"m2vdb server shutting down... {total_indexes} indexes in memory")


app = FastAPI(
    title="m2vdb",
    description="Vector database with Pinecone-style API",
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
    if name in indexes[user_id]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Index '{name}' already exists"
        )
    
    try:
        db = Collection(
            dimension=request.dimension,
            metric=request.metric,
            index_type=request.index_type
        )
        indexes[user_id][name] = db
        
        return IndexInfo(
            name=name,
            dimension=db.dimension,
            metric=db.metric,
            index_type=db.index_type,
            size=0
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))


@app.get("/indexes")
async def list_indexes(user_id: str = Depends(get_current_user)):
    """List all indexes for the authenticated user."""
    return {
        "indexes": [
            IndexInfo(
                name=name,
                dimension=db.dimension,
                metric=db.metric,
                index_type=db.index_type,
                size=len(db)
            )
            for name, db in indexes[user_id].items()
        ]
    }


@app.get("/indexes/{name}", response_model=IndexInfo)
async def get_index(name: str, user_id: str = Depends(get_current_user)):
    """Get index info."""
    if name not in indexes[user_id]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")
    
    db = indexes[user_id][name]
    return IndexInfo(
        name=name,
        dimension=db.dimension,
        metric=db.metric,
        index_type=db.index_type,
        size=len(db)
    )


@app.delete("/indexes/{name}")
async def delete_index(name: str, user_id: str = Depends(get_current_user)):
    """Delete an index."""
    if name not in indexes[user_id]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")
    
    del indexes[user_id][name]
    return {"message": f"Index '{name}' deleted"}


def _get_index(name: str, user_id: str) -> Collection:
    """Helper to get index or raise 404."""
    if name not in indexes[user_id]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")
    return indexes[user_id][name]


@app.post("/indexes/{name}/vectors", response_model=UpsertResponse)
async def upsert_vectors(
    name: str, 
    request: UpsertRequest,
    user_id: str = Depends(get_current_user)
):
    """Upsert (insert/update) vectors."""
    db = _get_index(name, user_id)
    
    upserted = 0
    errors = []
    
    for vec in request.vectors:
        try:
            vector_array = np.array(vec.vector, dtype=np.float32)
            db.upsert(vec.id, vector_array, vec.metadata)
            upserted += 1
        except ValueError as e:
            errors.append(f"{vec.id}: {str(e)}")
    
    if errors and upserted == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="; ".join(errors[:5]))
    
    return UpsertResponse(upserted_count=upserted)

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
    total_indexes = sum(len(user_indexes) for user_indexes in indexes.values())
    return {
        "name": "m2vdb",
        "version": "0.1.0",
        "total_indexes": total_indexes,
        "users": len(indexes)
    }


@app.get("/health")
async def health():
    """Simple health check - just confirms the service is alive"""
    return {"status": "healthy"}


@app.get("/stats")
async def stats(user_id: str = Depends(get_current_user)):
    """Detailed resource usage for the authenticated user"""
    # Aggregate stats across all user's indexes
    total_vectors = 0
    total_vectors_bytes = 0
    total_index_bytes = 0
    total_metadata_bytes = 0
    index_details = []
    
    for name, db in indexes[user_id].items():
        # Get stats from database instance
        db_stats = db.get_stats()
        
        total_vectors += db_stats["num_vectors"]
        total_vectors_bytes += db_stats["memory"]["vectors_bytes"]
        total_index_bytes += db_stats["memory"]["index_bytes"]
        total_metadata_bytes += db_stats["memory"]["metadata_bytes"]
        
        index_details.append({
            "name": name,
            "type": db_stats["index_type"],
            "dimension": db_stats["dimension"],
            "metric": db_stats["metric"],
            "num_vectors": db_stats["num_vectors"],
            "memory": {
                "vectors_mb": db_stats["memory"]["vectors_mb"],
                "index_mb": db_stats["memory"]["index_mb"],
                "metadata_mb": db_stats["memory"]["metadata_mb"],
                "total_mb": db_stats["memory"]["total_mb"]
            }
        })
    
    return {
        "user": user_id,
        "indexes": {
            "total": len(indexes[user_id]),
            "details": index_details
        },
        "vectors": {
            "total": total_vectors
        },
        "memory": {
            "vectors_mb": round(total_vectors_bytes / 1024 / 1024, 2),
            "indexes_mb": round(total_index_bytes / 1024 / 1024, 2),
            "metadata_mb": round(total_metadata_bytes / 1024 / 1024, 2),
            "total_mb": round((total_vectors_bytes + total_index_bytes + total_metadata_bytes) / 1024 / 1024, 2)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
