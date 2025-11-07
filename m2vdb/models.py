"""
Pydantic models for the vector database API.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class SearchResult(BaseModel):
    """
    Represents a single search result with ID, distance, and optional metadata.
    
    This is returned by search operations and includes the vector ID,
    its distance from the query, and any associated metadata.
    """
    id: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None
