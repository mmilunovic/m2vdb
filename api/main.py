from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv
import openai
import numpy as np
from m2vdb.database import V3cT0rDaTaBas3

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI(title="M2VDB API")

# Configure CORS for API clients.
# Allow callers specified in M2VDB_ALLOWED_ORIGINS env var or fall back to all origins
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("M2VDB_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector database with OpenAI's embedding dimension (1536 for text-embedding-3-small)
# DB_PATH = "_data/vector_db"
DB_PATH = "_data/product_descriptions"
os.makedirs(DB_PATH, exist_ok=True)  # Ensure directory exists

try:
    # Try to load existing database
    db = V3cT0rDaTaBas3(
        storage_path=DB_PATH,
        load_existing=True
    )
    print(f"Loaded existing database with {len(db.index.ids)} vectors")
except FileNotFoundError:
    # Create new database if none exists
    db = V3cT0rDaTaBas3(
        dim=1536,
        index_type='brute_force',
        storage_path=DB_PATH,
        metric='cosine'
    )
    # Save the initial empty database to ensure all required files are created
    db.save()
    print("Created new database")

class TextInput(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class SearchInput(BaseModel):
    query: str
    k: int = 5

async def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text using OpenAI's API."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")

@app.post("/add_text")
async def add_text(input_data: TextInput):
    """Add text and its metadata to the vector database."""
    # Get embedding for the text
    vector = await get_embedding(input_data.text)
    
    # Add vector to databasew
    db.add(
        vectors=vector.reshape(1, -1),  # Reshape to (1, dim)
        metadata_list=[{
            "text": input_data.text,
            **input_data.metadata
        }]
    )
    
    # Save database to disk
    db.save()
    
    return {
        "id": len(db.index.ids) - 1,
        "status": "success",
        "message": f"Added vector of dimension {vector.shape[0]} to the {db.index_type} index",
        "index_stats": {
            "total_vectors": len(db.index.ids),
            "dimension": db.dim,
            "metric": db.index.metric
        }
    }

@app.post("/search_text")
async def search_text(input_data: SearchInput):
    """Search for similar texts using vector similarity."""
    if not db.index.ids:
        raise HTTPException(status_code=400, detail="No vectors in database")
    
    # Validate k parameter
    if input_data.k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")
    
    # Adjust k if it's larger than available vectors
    adjusted_k = min(input_data.k, len(db.index.ids))
    notification = None
    if adjusted_k < input_data.k:
        notification = f"Requested k={input_data.k} was larger than available vectors ({len(db.index.ids)}). Adjusted to k={adjusted_k}"
    
    # Get embedding for the query
    query_vector = await get_embedding(input_data.query)
    
    # Search for similar vectors
    similar_ids, scores = db.index.search(
        queries=query_vector.reshape(1, -1),  # Reshape to (1, dim)
        k=adjusted_k
    )
    
    # Prepare results
    results = []
    for idx, score in zip(similar_ids[0], scores[0]):  # Get first row since we only have one query
        metadata = db.metadata[idx]
        results.append({
            "id": int(idx),
            "text": metadata.get("text", ""),
            "metadata": {k: v for k, v in metadata.items() if k != "text"},
            "similarity_score": float(score)
        })
    
    return {
        "query": input_data.query,
        "results": results,
        "stats": {
            "total_vectors_searched": len(db.index.ids),
            "k": adjusted_k,
            "metric": db.index.metric
        },
        "notification": notification
    }

@app.get("/vectors")
async def get_vectors():
    """Get all vectors and their metadata from the database."""
    if not db.index.ids:
        return {
            "vectors": [],
            "stats": {
                "total_vectors": 0,
                "dimension": db.dim,
                "metric": db.index.metric
            }
        }
    
    # Get all vectors and metadata
    vectors = []
    for idx in range(len(db.index.ids)):
        vector = db.index._vectors_array[idx]
        metadata = db.metadata[idx]
        vectors.append({
            "id": int(idx),
            "text": metadata.get("text", ""),
            "metadata": {k: v for k, v in metadata.items() if k != "text"},
            "vector": vector.tolist()
        })
    
    return {
        "vectors": vectors,
        "stats": {
            "total_vectors": len(db.index.ids),
            "dimension": db.dim,
            "metric": db.index.metric
        }
    }

@app.get("/neighbors/{vector_id}")
async def get_neighbors(vector_id: int, k: int = 5):
    """Get k nearest neighbors for a specific vector.
    
    Args:
        vector_id: ID of the vector to find neighbors for
        k: Number of nearest neighbors to return (default: 10)
    """
    if not db.index.ids:
        raise HTTPException(status_code=400, detail="No vectors in database")
    
    # Validate vector_id
    if vector_id < 0 or vector_id >= len(db.index.ids):
        raise HTTPException(status_code=400, detail=f"Invalid vector ID: {vector_id}")
    
    # Validate k parameter
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")
    
    # Adjust k if it's larger than available vectors
    adjusted_k = min(k, len(db.index.ids) - 1)  # -1 because we'll remove the vector itself
    notification = None
    if adjusted_k < k:
        notification = f"Requested k={k} was larger than available vectors ({len(db.index.ids)}). Adjusted to k={adjusted_k}"
    
    # Get the vector
    vector = db.index._vectors_array[vector_id].reshape(1, -1)
    
    # Search for similar vectors
    similar_ids, scores = db.index.search(
        queries=vector,
        k=adjusted_k + 1  # +1 because the vector will be its own nearest neighbor
    )
    
    # Remove the vector itself from results
    mask = similar_ids[0] != vector_id
    similar_ids = similar_ids[0][mask]
    scores = scores[0][mask]
    
    # Prepare results
    neighbors = []
    for idx, score in zip(similar_ids, scores):
        metadata = db.metadata[idx]
        neighbors.append({
            "id": int(idx),
            "text": metadata.get("text", ""),
            "metadata": {k: v for k, v in metadata.items() if k != "text"},
            "distance": float(score)
        })
    
    return {
        "vector_id": vector_id,
        "neighbors": neighbors,
        "stats": {
            "total_vectors_searched": len(db.index.ids),
            "k": adjusted_k,
            "metric": db.index.metric
        },
        "notification": notification
    } 