"""
Benchmarking suite for m2vdb vector database.

This package contains dataset loaders, metrics computation, and benchmark runners.
Use run_benchmarks.py as the main entry point.
"""

# Only expose utilities that might be imported by other code
from .datasets import load_sift1m, load_fasttext, Dataset
from .metrics import compute_recall, compute_latency_stats, measure_memory
from .benchmark import BenchmarkRunner, BenchmarkResult
from .cache import BenchmarkCache

__all__ = [
    "Dataset",
    "load_sift1m",
    "load_fasttext",
    "compute_recall",
    "compute_latency_stats",
    "measure_memory",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkCache",
]
