"""
Metrics computation for benchmarking vector search algorithms.
"""

import sys
from typing import List, Dict
import numpy as np
import psutil
import os

from m2vdb import Collection


def compute_recall(
    predicted: List[List[str]],
    ground_truth: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int
) -> float:
    """
    Compute recall@k: fraction of true top-k neighbors that were retrieved.
    
    For each query:
    - True neighbors: ground_truth[i, :k] (the k nearest neighbors)
    - Predicted: top k results from search
    - Recall@k = (# true neighbors found in predictions) / k
    
    Averaged across all queries.
    
    Args:
        predicted: List of result lists from search, each containing IDs
        ground_truth: Array of shape (n_queries, gt_k) with ground truth indices
        id_to_idx: Mapping from vector IDs to their indices
        k: Number of results to consider
    
    Returns:
        Average recall@k across all queries (0.0 to 1.0)
    """
    n_queries = len(predicted)
    assert n_queries == len(ground_truth), "Mismatch between predictions and ground truth"
    
    total_recall = 0.0
    
    for i, pred_ids in enumerate(predicted):
        # Get predicted indices (convert IDs back to indices)
        pred_indices = [id_to_idx.get(id, -1) for id in pred_ids[:k]]
        pred_set = set(pred_indices)
        
        # Get true top-k indices for this query
        # Filter out -1 which indicates "no valid neighbor" (when dataset is limited)
        true_indices_raw = ground_truth[i, :k].tolist()
        true_indices = set([idx for idx in true_indices_raw if idx >= 0])
        
        # Count how many of the true k neighbors we found
        intersection = len(pred_set & true_indices)
        
        # Recall = (# found) / (# we should have found)
        # If ground truth has fewer than k valid neighbors (limited dataset),
        # we normalize by the actual number of valid neighbors
        n_true = len(true_indices)
        recall = intersection / n_true if n_true > 0 else 0.0
        total_recall += recall
    
    return total_recall / n_queries if n_queries > 0 else 0.0


def compute_latency_stats(query_times: List[float]) -> Dict[str, float]:
    """
    Compute latency percentiles from query times.
    
    Args:
        query_times: List of query latencies in seconds
    
    Returns:
        Dict with p50, p90, p95, p99 in milliseconds
    """
    if not query_times:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    times_ms = np.array(query_times) * 1000  # Convert to milliseconds
    
    return {
        "p50": float(np.percentile(times_ms, 50)),
        "p90": float(np.percentile(times_ms, 90)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
        "mean": float(np.mean(times_ms))
    }


def measure_memory(pid: int = None) -> Dict[str, float]:
    """
    Measure memory usage of current process.
    
    Args:
        pid: Process ID to measure (defaults to current process)
    
    Returns:
        Dict with memory metrics in MB:
        - rss_mb: Resident Set Size - actual RAM used (not swapped to disk)
        - vms_mb: Virtual Memory Size - includes swapped memory
    """
    if pid is None:
        pid = os.getpid()
    
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 ** 2),  # Resident Set Size
        "vms_mb": mem_info.vms / (1024 ** 2),  # Virtual Memory Size
    }


def compute_qps(n_queries: int, total_time: float) -> float:
    """
    Compute queries per second.
    
    Args:
        n_queries: Number of queries executed
        total_time: Total time in seconds
    
    Returns:
        Queries per second
    """
    return n_queries / total_time if total_time > 0 else 0.0


def measure_index_memory(db: Collection) -> Dict[str, float]:
    """
    Measure memory used by index data structures only.
    
    This measures only the index's internal structures, not db._vectors
    (which is a Collection implementation detail):
    
    - BruteForce: vectors array stored in index.vectors
    - PQ: codebooks + quantized codes (much smaller than full vectors!)
    - RustBruteForce: vectors + norms stored in Rust
    
    Args:
        db: Collection instance to measure
    
    Returns:
        Dict with:
        - index_mb: Total index memory in MB
        - bytes_per_vector: Average index bytes per vector
    """
    index_bytes = 0
    
    # Get the index
    index = db.index
    
    # BruteForce: stores full vectors array
    if hasattr(index, 'vectors') and index.vectors is not None:
        index_bytes += index.vectors.nbytes
    
    # PQ: stores codebooks and quantized codes
    if hasattr(index, 'codebooks') and index.codebooks is not None:
        index_bytes += index.codebooks.nbytes
    
    if hasattr(index, 'quantized_codes') and index.quantized_codes is not None:
        index_bytes += index.quantized_codes.nbytes
    
    # IVF: stores centroids, inverted lists, and cached norms
    if hasattr(index, 'centroids') and index.centroids is not None:
        index_bytes += index.centroids.nbytes
    
    if hasattr(index, 'centroid_norms_sq') and index.centroid_norms_sq is not None:
        index_bytes += index.centroid_norms_sq.nbytes
    
    if hasattr(index, 'inverted_lists') and index.inverted_lists:
        for cluster_id, inv_list in index.inverted_lists.items():
            # Vectors in each cluster
            if 'vectors' in inv_list:
                index_bytes += inv_list['vectors'].nbytes
            # IDs in each cluster
            if 'ids' in inv_list:
                index_bytes += inv_list['ids'].nbytes * 8  # Approximate object array
            # Cached norms for euclidean
            if 'norms_sq' in inv_list:
                index_bytes += inv_list['norms_sq'].nbytes
    
    # IVF: ID to cluster mapping
    if hasattr(index, '_id_to_cluster') and index._id_to_cluster:
        index_bytes += sys.getsizeof(index._id_to_cluster)
        for k, v in index._id_to_cluster.items():
            index_bytes += sys.getsizeof(k) + sys.getsizeof(v)
    
    # RustBruteForce or other indexes might have vector_norms
    if hasattr(index, 'vector_norms') and index.vector_norms is not None:
        if hasattr(index.vector_norms, 'nbytes'):
            index_bytes += index.vector_norms.nbytes
        else:
            index_bytes += sys.getsizeof(index.vector_norms)
    
    # IDs list (all indexes need this for mapping results back)
    if hasattr(index, 'ids') and index.ids:
        index_bytes += sys.getsizeof(index.ids)
        for id_str in index.ids:
            index_bytes += sys.getsizeof(id_str)
    
    # ID to index mapping (for efficient lookups)
    if hasattr(index, '_id_to_idx') and index._id_to_idx:
        index_bytes += sys.getsizeof(index._id_to_idx)
        for k, v in index._id_to_idx.items():
            index_bytes += sys.getsizeof(k) + sys.getsizeof(v)
    
    n_vectors = len(db) if len(db) > 0 else 1
    
    return {
        'index_mb': index_bytes / (1024 ** 2),
        'bytes_per_vector': index_bytes / n_vectors
    }

