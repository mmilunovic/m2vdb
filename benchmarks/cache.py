import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type
from dataclasses import asdict

from benchmarks.benchmark import BenchmarkResult

CACHE_DIR = Path(__file__).parent / "cache"

class BenchmarkCache:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)

    def _get_code_hash(self, cls: Type[Any]) -> str:
        """Compute SHA256 hash of the class source code."""
        try:
            source = inspect.getsource(cls)
            return hashlib.sha256(source.encode('utf-8')).hexdigest()
        except (OSError, TypeError):
            return "unknown"

    def _compute_key(self, config: Dict[str, Any], code_hash: str) -> str:
        """Generate a unique cache key."""
        # Sort keys to ensure consistent JSON string
        config_str = json.dumps(config, sort_keys=True)
        combined = f"{config_str}|{code_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _get_file_path(self, dataset: str, index: str, key: str) -> Path:
        """Create a human-readable filename with the hash."""
        # Sanitize names for filesystem
        safe_dataset = "".join(c for c in dataset if c.isalnum() or c in ('-', '_'))
        safe_index = "".join(c for c in index if c.isalnum() or c in ('-', '_'))
        
        # Use first 12 chars of hash for filename to keep it manageable
        filename = f"{safe_dataset}__{safe_index}__{key[:12]}.json"
        return self.cache_dir / filename

    def _build_config(self, index_name, dataset_name, n_vectors, dimension, k, n_queries, seed, index_params: Optional[Dict[str, Any]] = None):
        config = {
            "index_name": index_name,
            "dataset": dataset_name,
            "n_vectors": n_vectors,
            "dimension": dimension,
            "k": k,
            "n_queries": n_queries,
            "seed": seed,
        }
        
        if index_params:
            config['index_params'] = index_params
            
        return config

    def get(self, db, index_name, dataset, k, n_queries, seed) -> Optional[BenchmarkResult]:
        """Retrieve result from cache if it exists."""
        index_params = {**db.index_params}
        index_params['metric'] = db.metric

        config = self._build_config(
            index_name, dataset.name, len(dataset.base_vectors), 
            dataset.dimension, k, n_queries, seed,
            index_params=index_params
        )
        
        # Get the code hash from the actual index instance
        code_hash = self._get_code_hash(type(db.index))
        key = self._compute_key(config, code_hash)
        
        cache_file = self._get_file_path(dataset.name, index_name, key)
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Remove cache info field (used for cache validation/auditing) 
            # before creating the BenchmarkResult object
            data.pop("_cache_info", None)
                
            return BenchmarkResult(**data)
        except Exception:
            return None

    def save(self, result: BenchmarkResult, db, n_queries, seed) -> None:
        """Save result to cache."""
        index_params = {**db.index_params}
        index_params['metric'] = db.metric

        config = self._build_config(
            result.index_name, result.dataset_name, result.n_vectors,
            result.dimension, result.k_searched, n_queries, seed,
            index_params=index_params
        )
        
        code_hash = self._get_code_hash(type(db.index))
        key = self._compute_key(config, code_hash)
        
        cache_file = self._get_file_path(result.dataset_name, result.index_name, key)
        
        # Prepare JSON data
        data = asdict(result)
        data["_cache_info"] = {
            "hash": key,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "code_hash": code_hash
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
