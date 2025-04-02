# benchmarks/benchmark.py

import numpy as np
import faiss
from m2vdb.index import BruteForceIndex
from time import perf_counter
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k

def format_time(seconds):
    """Format time in appropriate units"""
    ms = seconds * 1000
    if ms < 1:
        return f"{ms * 1000:.2f}µs"
    return f"{ms:.2f}ms"

def run_benchmark():
    # Print benchmark configuration
    print("=" * 60)
    print("VECTOR SIMILARITY SEARCH BENCHMARK")
    print("=" * 60)
    print("Dataset: SIFT1M (128-dimensional, L2 distance)")
    
    # Load dataset
    print("\nLoading dataset...")
    xb, xq, gt = load_sift1m()
    print(f"Database size: {len(xb)} vectors")
    print(f"Query size:    {len(xq)} queries")
    print(f"Dimensions:    {xb.shape[1]}")
    print("\nIndexes:")
    print("-" * 60)
    print("1. FAISS:  IndexFlatL2 (exact search, brute force)")
    print("2. m2vdb:  BruteForceIndex (exact search)")
    print("-" * 60)

    # FAISS baseline
    faiss_index = faiss.IndexFlatL2(128)
    start = perf_counter()
    faiss_index.add(xb)
    faiss_build_time = perf_counter() - start
    
    start = perf_counter()
    _, faiss_results = faiss_index.search(xq, 10)
    faiss_search_time = perf_counter() - start
    faiss_recall = recall_at_k(faiss_results, gt, k=10)

    # m2vdb brute-force
    bf = BruteForceIndex(dim=128)
    start = perf_counter()
    bf.add(xb)
    m2v_build_time = perf_counter() - start
    
    start = perf_counter()
    m2v_results = bf.search(xq, k=10)
    m2v_search_time = perf_counter() - start
    m2v_recall = recall_at_k(m2v_results, gt, k=10)

    # Calculate relative performance
    search_slowdown = (m2v_search_time / faiss_search_time - 1) * 100
    build_slowdown = (m2v_build_time / faiss_build_time - 1) * 100

    # Print results
    print("\nResults:")
    print("=" * 80)
    print(f"{'Method':<10} {'Build Time':<15} {'Search Time':<15} {'Recall@10':<10} {'vs FAISS'}")
    print("-" * 80)
    print(f"{'FAISS':<10} {format_time(faiss_build_time):<15} {format_time(faiss_search_time):<15} {faiss_recall:.4f}")
    print(f"{'m2vdb':<10} {format_time(m2v_build_time):<15} {format_time(m2v_search_time):<15} {m2v_recall:.4f}    {search_slowdown:+.1f}%")
    print("=" * 80)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("-" * 60)
    print(f"Index Build: m2vdb is {build_slowdown:+.1f}% vs FAISS")
    print(f"Search Time: m2vdb is {search_slowdown:+.1f}% vs FAISS")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
