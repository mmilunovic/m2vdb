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

from database import VectorDatabase
from models import (
    CreateIndexRequest,
    IndexInfo,
    UpsertRequest,
    UpsertResponse,
    SearchRequest,
    SearchResponse,
    DeleteRequest,
    DeleteResponse,
    FetchResponse,
)


security = HTTPBearer()

API_KEYS = {
    "sk-test-user1": "user1",
    "sk-test-user2": "user2"
}

indexes: dict[str, dict[str, VectorDatabase]] = {
    "user1": {},
    "user2": {}
}


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
    
    assert request.metric in ['cosine', 'euclidean'], "Invalid metric"
    assert request.index_type in ['brute_force', 'pq', 'hnsw'], "Invalid index_type"
    
    try:
        db = VectorDatabase(
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
                size=db.size()
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
        size=db.size()
    )


@app.delete("/indexes/{name}")
async def delete_index(name: str, user_id: str = Depends(get_current_user)):
    """Delete an index."""
    if name not in indexes[user_id]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Index '{name}' not found")
    
    del indexes[user_id][name]
    return {"message": f"Index '{name}' deleted"}


def _get_index(name: str, user_id: str) -> VectorDatabase:
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
            db.insert(vec.id, vector_array, vec.metadata)
            upserted += 1
        except (ValueError, AssertionError) as e:
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
    except (ValueError, AssertionError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/indexes/{name}/vectors/delete", response_model=DeleteResponse)
async def delete_vectors(
    name: str, 
    request: DeleteRequest,
    user_id: str = Depends(get_current_user)
):
    """Delete vectors by ID."""
    db = _get_index(name, user_id)
    
    deleted = sum(1 for id in request.ids if db.delete(id))
    
    return DeleteResponse(deleted_count=deleted)


@app.get("/indexes/{name}/vectors/{id}", response_model=FetchResponse)
async def fetch_vector(
    name: str, 
    id: str,
    user_id: str = Depends(get_current_user)
):
    """Fetch a vector by ID."""
    db = _get_index(name, user_id)
    
    # Get vector from index
    if id not in db.index._id_to_idx:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Vector '{id}' not found")
    
    idx = db.index._id_to_idx[id]
    vector = db.index.vectors[idx].tolist()
    metadata = db._metadata.get(id)
    
    return FetchResponse(id=id, vector=vector, metadata=metadata)


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
    """Health check."""
    total_indexes = sum(len(user_indexes) for user_indexes in indexes.values())
    return {"status": "healthy", "total_indexes": total_indexes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
