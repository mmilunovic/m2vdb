"""
Persistence layer for m2vdb collections.

CollectionManager: manages collections with disk persistence and in-memory LRU cache
"""

from pathlib import Path
from typing import Optional, List
from cachetools import LRUCache
import json
import numpy as np
import shutil

from .collection import Collection
from .models import CollectionNotFound


# Maximum number of collections to keep in RAM
MAX_CACHED_COLLECTIONS = 5

VectorsDict = dict[str, np.ndarray]
MetadataDict = dict[str, dict]


class CollectionManager:
    """
    Manages collections with disk persistence and in-memory LRU cache.
    
    Directory structure:
        data_root/
            user_id/
                collection_name/
                    manifest.json
                    artifacts/
                    vectors.npz
                    metadata.json
    """
    
    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        # Cache: (user_id, collection_name) -> Collection
        # LRUCache automatically evicts least recently used items when maxsize is exceeded
        self._collections: LRUCache[tuple[str, str], Collection] = LRUCache(maxsize=MAX_CACHED_COLLECTIONS)
    
    def _save_manifest(self, user_id: str, collection_name: str, manifest: dict) -> None:
        collection_dir = self.data_root / user_id / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = collection_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _load_manifest(self, user_id: str, collection_name: str) -> dict:
        manifest_path = self.data_root / user_id / collection_name / "manifest.json"
        if not manifest_path.exists():
            raise CollectionNotFound(
                f"Collection '{collection_name}' not found for user '{user_id}'"
            )
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def _save_vectors_and_metadata(
        self,
        user_id: str,
        collection_name: str,
        vectors: VectorsDict,
        metadata: MetadataDict,
    ) -> None:
        collection_dir = self.data_root / user_id / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        if vectors:
            ids = list(vectors.keys())
            vectors_array = np.stack([vectors[id] for id in ids])
            np.savez(
                collection_dir / "vectors.npz",
                ids=np.array(ids),
                vectors=vectors_array
            )
        else:
            np.savez(
                collection_dir / "vectors.npz",
                ids=np.array([]),
                vectors=np.array([]).reshape(0, 0)
            )
        
        metadata_path = collection_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_vectors_and_metadata(
        self,
        user_id: str,
        collection_name: str,
    ) -> tuple[VectorsDict, MetadataDict]:
        collection_dir = self.data_root / user_id / collection_name
        vectors_path = collection_dir / "vectors.npz"
        
        if not vectors_path.exists():
            raise CollectionNotFound(
                f"Collection '{collection_name}' not found for user '{user_id}'"
            )
        
        data = np.load(vectors_path, allow_pickle=True)
        ids = data['ids']
        vectors_array = data['vectors']
        
        vectors_dict = {}
        if len(ids) > 0:
            for id, vector in zip(ids, vectors_array):
                vectors_dict[str(id)] = vector
        
        metadata_path = collection_dir / "metadata.json"
        if not metadata_path.exists():
            raise CollectionNotFound(
                f"Collection '{collection_name}' not found for user '{user_id}'"
            )
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return vectors_dict, metadata_dict
    
    def list_collections(self, user_id: str) -> list[str]:
        user_dir = self.data_root / user_id
        if not user_dir.exists():
            return []
        
        collections = []
        for item in user_dir.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                collections.append(item.name)
        
        return sorted(collections)
    
    def create_collection(
        self,
        user_id: str,
        name: str,
        dimension: int,
        metric: str,
        index_type: str,
        index_params: Optional[dict] = None,
    ) -> Collection:
        key = (user_id, name)
        
        # Check if already exists (cache or disk)
        if key in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        
        manifest_path = self.data_root / user_id / name / "manifest.json"
        if manifest_path.exists():
            raise ValueError(f"Collection '{name}' already exists")
        
        collection = Collection(
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            index_params=index_params or {},
            storage_path=str(self.data_root / user_id / name),
        )
        
        manifest = {
            'dimension': collection.dimension,
            'metric': collection.metric,
            'index_type': collection.index_type,
            'index_params': collection.index_params,
        }
        self._save_manifest(user_id, name, manifest)
        self._save_vectors_and_metadata(user_id, name, collection._vectors, collection._metadata)
        
        self._collections[key] = collection
        return collection
    
    def get_collection(self, user_id: str, name: str) -> Collection:
        key = (user_id, name)
        
        if key in self._collections:
            return self._collections[key]
        
        # Load from disk (will raise CollectionNotFound if doesn't exist)
        manifest = self._load_manifest(user_id, name)
        
        collection = Collection(
            dimension=manifest['dimension'],
            metric=manifest['metric'],
            index_type=manifest['index_type'],
            index_params=manifest.get('index_params'),
            storage_path=str(self.data_root / user_id / name),
        )
        
        vectors_dict, metadata_dict = self._load_vectors_and_metadata(user_id, name)
        collection._vectors = vectors_dict
        collection._metadata = metadata_dict
        
        artifacts_dir = self.data_root / user_id / name / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        collection.index.load_artifacts(str(artifacts_dir))
        
        if len(vectors_dict) > 0 and not collection.index.is_built:
            ids = list(vectors_dict.keys())
            vectors_array = np.stack([vectors_dict[id] for id in ids])
            collection.index.build(vectors_array, ids)
        
        self._collections[key] = collection
        return collection
    
    def delete_collection(self, user_id: str, name: str) -> None:
        key = (user_id, name)
        
        # Check if exists (cache or disk)
        exists_in_cache = key in self._collections
        
        if not exists_in_cache:
            # Check disk
            manifest_path = self.data_root / user_id / name / "manifest.json"
            if not manifest_path.exists():
                raise CollectionNotFound(f"Collection '{name}' not found for user '{user_id}'")
        
        if exists_in_cache:
            del self._collections[key]
        
        collection_dir = self.data_root / user_id / name
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
        
        user_dir = self.data_root / user_id
        if user_dir.exists() and not any(user_dir.iterdir()):
            user_dir.rmdir()
    
    def upsert(self, user_id: str, name: str, id: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Upsert a single vector and auto-save to disk."""
        collection = self.get_collection(user_id, name)
        collection.upsert(id, vector, metadata)
        self._save_collection(user_id, name, collection)
    
    def batch_upsert(
        self, 
        user_id: str, 
        name: str,
        ids: List[str],
        vectors: List[np.ndarray],
        metadata: Optional[List[Optional[dict]]] = None
    ) -> int:
        """
        Batch upsert multiple vectors efficiently and auto-save to disk.
        
        This is much more efficient than calling upsert() in a loop because:
        1. Index is rebuilt only once after all vectors are added
        2. Collection is saved to disk only once
        3. For PQ indexes with many vectors, this avoids repeated k-means training
        
        Args:
            user_id: User identifier
            name: Collection name
            ids: List of unique vector IDs
            vectors: List of numpy arrays
            metadata: Optional list of metadata dicts
        
        Returns:
            Number of vectors upserted
        """
        collection = self.get_collection(user_id, name)
        count = collection.batch_upsert(ids, vectors, metadata)
        self._save_collection(user_id, name, collection)
        return count
    
    def _save_collection(self, user_id: str, name: str, collection: Collection) -> None:
        manifest = {
            'dimension': collection.dimension,
            'metric': collection.metric,
            'index_type': collection.index_type,
            'index_params': collection.index_params,
        }
        self._save_manifest(user_id, name, manifest)
        
        artifacts_dir = self.data_root / user_id / name / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        collection.index.save_artifacts(str(artifacts_dir))
        
        self._save_vectors_and_metadata(
            user_id,
            name,
            collection._vectors,
            collection._metadata
        )
