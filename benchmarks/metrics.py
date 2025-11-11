"""
Metrics computation for benchmarking vector search algorithms.
"""

import time
from typing import List, Tuple, Dict
import numpy as np
import psutil
import os


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


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed is not None else 0.0


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
