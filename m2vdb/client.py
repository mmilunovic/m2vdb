"""
Python SDK for m2vdb vector database.
"""

import httpx
from typing import List, Dict, Any
import numpy as np

from .models import SearchResult


class _IndexHandle:
    """Handle for operations on a specific index."""
    
    def __init__(self, name: str, client: 'M2VDBClient'):
        self.name = name
        self.client = client
    
    def upsert(
        self, 
        vectors: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Upsert vectors to this index.
        
        Args:
            vectors: List of dicts with 'id', 'vector', 'metadata' keys
            batch_size: Number of vectors per batch request
            
        Returns:
            Total number of vectors upserted
        """
        total_upserted = 0
        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                response = self.client._session.post(
                    f"/indexes/{self.name}/vectors",
                    json={"vectors": batch}
                )
                response.raise_for_status()
                total_upserted += response.json()['upserted_count']
            return total_upserted
        except httpx.HTTPStatusError as e:
            self.client._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def query(
        self,
        vector: List[float] | np.ndarray,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Query for similar vectors in this index.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        try:
            response = self.client._session.post(
                f"/indexes/{self.name}/search",
                json={
                    "vector": vector,
                    "k": top_k,
                    "include_metadata": include_metadata
                }
            )
            response.raise_for_status()
            response_data = response.json()
            return [SearchResult(**match) for match in response_data['matches']]
        except httpx.HTTPStatusError as e:
            self.client._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector from this index.
        
        Args:
            id: Vector ID to delete
            
        Returns:
            True if vector was deleted, False if not found
        """
        try:
            response = self.client._session.delete(
                f"/indexes/{self.name}/vectors/{id}"
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            self.client._handle_error(e)
            raise
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def fetch(self, vector_id: str) -> Dict[str, Any]:
        """
        Fetch a vector by ID from this index.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Dict with 'id', 'vector', 'metadata' keys
        """
        try:
            response = self.client._session.get(f"/indexes/{self.name}/vectors/{vector_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self.client._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def describe(self) -> Dict[str, Any]:
        """
        Get index info and statistics.
        
        Returns:
            Dict with index metadata
        """
        try:
            response = self.client._session.get(f"/indexes/{self.name}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self.client._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e


class M2VDBClient:
    """Python client for m2vdb vector database."""
    
    def __init__(
        self,
        api_key: str,
        host: str = "http://localhost:8000"
    ):
        """
        Initialize m2vdb client.
        
        Args:
            api_key: API key for authentication
            host: Base URL of m2vdb server
        """
        self.api_key = api_key
        self.host = host.rstrip('/')
        self._session = httpx.Client(
            base_url=self.host,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    def _handle_error(self, e: httpx.HTTPStatusError):
        """Convert HTTP errors to Python exceptions."""
        if e.response.status_code == 401:
            raise ValueError("Invalid API key") from e
        elif e.response.status_code == 404:
            detail = e.response.json().get('detail', 'Not found')
            raise ValueError(detail) from e
        elif e.response.status_code == 409:
            detail = e.response.json().get('detail', 'Conflict')
            raise ValueError(detail) from e
        elif e.response.status_code == 422:
            # Pydantic validation error
            errors = e.response.json().get('detail', [])
            if errors:
                first_error = errors[0]
                field = first_error.get('loc', [])[-1] if first_error.get('loc') else 'field'
                msg = first_error.get('msg', 'Validation error')
                raise ValueError(f"Invalid {field}: {msg}") from e
            raise ValueError("Validation error") from e
        else:
            raise ValueError(f"HTTP {e.response.status_code}: {e.response.text}") from e
    
    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "brute_force",
        index_params: dict[str, Any] | None = None
    ) -> '_IndexHandle':
        """
        Create a new index.
        
        Args:
            name: Index name (unique per user)
            dimension: Vector dimensionality
            metric: Distance metric ('cosine' or 'euclidean')
            index_type: Index implementation ('brute_force', 'pq', 'rust_brute_force', 'ivf')
            index_params: Optional index-specific parameters:
                - PQ: n_subvectors (int), n_clusters (int)
                - IVF: n_lists (int)
            
        Returns:
            Index handle for operations
        """
        payload = {
            "dimension": dimension,
            "metric": metric,
            "index_type": index_type
        }
        if index_params is not None:
            payload["index_params"] = index_params
            
        try:
            response = self._session.post(
                f"/indexes/{name}",
                json=payload
            )
            response.raise_for_status()
            return _IndexHandle(name, self)
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def Index(self, name: str) -> '_IndexHandle':
        """
        Get handle to an existing index.
        
        Args:
            name: Index name
            
        Returns:
            Index handle for operations
        """
        try:
            # Verify index exists
            response = self._session.get(f"/indexes/{name}")
            response.raise_for_status()
            return _IndexHandle(name, self)
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all indexes for the authenticated user.
        
        Returns:
            List of index info dicts
        """
        try:
            response = self._session.get("/indexes")
            response.raise_for_status()
            return response.json()['indexes']
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def delete_index(self, name: str):
        """
        Delete an index permanently.
        
        Args:
            name: Index name to delete
            
        Warning: This permanently deletes the index and all its data.
        """
        try:
            response = self._session.delete(f"/indexes/{name}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e
    
    def health(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Dict with health status
        """
        try:
            response = self._session.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
            raise  # Ensure we never fall through
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}") from e


