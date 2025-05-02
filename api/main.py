from fastapi import FastAPI, HTTPException
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

# Initialize vector database with OpenAI's embedding dimension (1536 for text-embedding-3-small)
DB_PATH = "data/vector_db"
try:
    # Try to load existing database
    db = V3cT0rDaTaBas3(
        storage_path=DB_PATH,
        load_existing=True
    )
except FileNotFoundError:
    # Create new database if none exists
    db = V3cT0rDaTaBas3(
        dim=1536,
        index_type='brute_force',
        storage_path=DB_PATH,
        metric='cosine'
    )

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
    
    # Add vector to database
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
    if input_data.k > len(db.index.ids):
        raise HTTPException(
            status_code=400, 
            detail=f"Requested k={input_data.k} is larger than the number of vectors in the index ({len(db.index.ids)})"
        )
    
    # Get embedding for the query
    query_vector = await get_embedding(input_data.query)
    
    # Search for similar vectors
    similar_ids, scores = db.index.search(
        queries=query_vector.reshape(1, -1),  # Reshape to (1, dim)
        k=input_data.k
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
            "k": input_data.k,
            "metric": db.index.metric
        }
    } 