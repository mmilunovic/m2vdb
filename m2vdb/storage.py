# m2vdb/storage.py

import numpy as np
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from m2vdb.index import BaseIndex, BruteForceIndex, ANNIndex

class BaseStorage(ABC):
    """Abstract base class for storage implementations"""
    
    @abstractmethod
    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        """Save vectors to storage"""
        pass
    
    @abstractmethod
    def load_vectors(self, path: str) -> np.ndarray:
        """Load vectors from storage"""
        pass
    
    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> None:
        """Save metadata to storage"""
        pass
    
    @abstractmethod
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from storage"""
        pass

class FileStorage(BaseStorage):
    """File-based storage implementation"""
    
    def save_vectors(self, vectors: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vectors)
    
    def load_vectors(self, path: str) -> np.ndarray:
        return np.load(path)
    
    def save_metadata(self, metadata: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f)
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)

class IndexManager:
    """Manager class for saving and loading indexes"""
    
    def __init__(self, storage: BaseStorage = None):
        self.storage = storage or FileStorage()
    
    def save_index(self, index: BaseIndex, path: str) -> None:
        """Save an index to storage"""
        os.makedirs(path, exist_ok=True)
        
        # Save vectors - concatenate all vector arrays in the list
        if hasattr(index, 'vectors') and index.vectors:
            vectors = np.vstack(index.vectors) if len(index.vectors) > 0 else np.array([]).reshape(0, index.dim)
            self.storage.save_vectors(vectors, os.path.join(path, "vectors.npy"))
        
        # Save base configuration
        config = {
            'dim': index.dim,
            'metric': index.metric,
            'ids': index.ids if hasattr(index, 'ids') and index.ids else None,
            'index_type': index.__class__.__name__
        }
        
        # Save index-specific parameters
        if isinstance(index, ANNIndex):
            config['num_candidates'] = index.num_candidates
            # Save the integer seed value
            if hasattr(index, 'random_seed'):
                config['random_seed'] = index.random_seed
            
        self.storage.save_metadata(config, os.path.join(path, "config.json"))
    
    def load_index(self, path: str) -> BaseIndex:
        """Load an index from storage"""
        # Load configuration
        config = self.storage.load_metadata(os.path.join(path, "config.json"))
        
        # Extract base parameters
        index_type = config.pop('index_type')
        dim = config.pop('dim')
        metric = config.pop('metric')
        ids = config.pop('ids', None)
        
        # Create appropriate index using appropriate parameters
        if index_type == 'BruteForceIndex':
            # BruteForceIndex doesn't take additional parameters
            index = BruteForceIndex(dim=dim, metric=metric)
        elif index_type == 'ANNIndex':
            # Extract only the parameters ANNIndex expects
            ann_params = {}
            
            # Add parameters if they exist in the config
            if 'num_candidates' in config:
                ann_params['num_candidates'] = config.pop('num_candidates')
            
            if 'random_seed' in config:
                ann_params['random_seed'] = config.pop('random_seed')
            
            index = ANNIndex(dim=dim, metric=metric, **ann_params)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        # Load vectors and ids if they exist
        if os.path.exists(os.path.join(path, "vectors.npy")):
            vectors = self.storage.load_vectors(os.path.join(path, "vectors.npy"))
            # Store as a single array in the vectors list
            if len(vectors) > 0:
                index.vectors = [vectors]
                # Regenerate IDs if none were saved
                if ids is None:
                    ids = list(range(len(vectors)))
                index.ids = ids
        
        return index
