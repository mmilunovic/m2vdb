# benchmarks/benchmark.py

import numpy as np
import faiss
from m2vdb.index import BruteForceIndex, ANNIndex
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
    print("VECTOR SIMILARITY SEARCH BENCHMARK")
    print("\nDataset: SIFT1M")
    
    # Load dataset
    xb, xq, gt = load_sift1m()
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")
    
    print("\nIndexes:")
    print("FAISS:  IndexFlatL2 (exact search, brute force)")
    print("m2vdb:  BruteForceIndex (exact search)")
    print("m2vdb:  ANNIndex (approximate search, random sampling)")

    # FAISS baseline
    print("\nBuilding FAISS index...")
    faiss_index = faiss.IndexFlatL2(128)
    start = perf_counter()
    faiss_index.add(xb)
    faiss_build_time = perf_counter() - start
    
    print("Running FAISS search...")
    start = perf_counter()
    _, faiss_results = faiss_index.search(xq, 10)
    faiss_search_time = perf_counter() - start
    faiss_recall = recall_at_k(faiss_results, gt, k=10)

    # m2vdb brute-force
    print("\nBuilding m2vdb brute-force index...")
    bf = BruteForceIndex(dim=128)
    start = perf_counter()
    bf.add(xb)
    m2v_bf_build_time = perf_counter() - start
    
    print("Running m2vdb brute-force search...")
    start = perf_counter()
    m2v_bf_results = bf.search(xq, k=10)
    m2v_bf_search_time = perf_counter() - start
    m2v_bf_recall = recall_at_k(m2v_bf_results, gt, k=10)

    # m2vdb ANN
    print("\nBuilding m2vdb ANN index...")
    ann = ANNIndex(dim=128, num_candidates=1000)  # Sample 1000 candidates per query
    start = perf_counter()
    ann.add(xb)
    m2v_ann_build_time = perf_counter() - start
    
    print("Running m2vdb ANN search...")
    start = perf_counter()
    m2v_ann_results = ann.search(xq, k=10)
    m2v_ann_search_time = perf_counter() - start
    m2v_ann_recall = recall_at_k(m2v_ann_results, gt, k=10)

    # Calculate relative performance
    bf_search_slowdown = (m2v_bf_search_time / faiss_search_time - 1) * 100
    bf_build_slowdown = (m2v_bf_build_time / faiss_build_time - 1) * 100
    ann_search_slowdown = (m2v_ann_search_time / faiss_search_time - 1) * 100
    ann_build_slowdown = (m2v_ann_build_time / faiss_build_time - 1) * 100

    # Print results table
    print("\nResults:")
    print(f"{'Method':<15} {'Build':<12} {'Search':<12} {'Recall@10':<10} {'vs FAISS'}")
    print(f"{'FAISS':<15} {format_time(faiss_build_time):<12} {format_time(faiss_search_time):<12} {faiss_recall:.4f}")
    print(f"{'m2vdb (BF)':<15} {format_time(m2v_bf_build_time):<12} {format_time(m2v_bf_search_time):<12} {m2v_bf_recall:.4f}    {bf_search_slowdown:+.1f}%")
    print(f"{'m2vdb (ANN)':<15} {format_time(m2v_ann_build_time):<12} {format_time(m2v_ann_search_time):<12} {m2v_ann_recall:.4f}    {ann_search_slowdown:+.1f}%")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Index Build:")
    print(f"  m2vdb (BF):  {bf_build_slowdown:+.1f}% vs FAISS")
    print(f"  m2vdb (ANN): {ann_build_slowdown:+.1f}% vs FAISS")
    print(f"Search Time:")
    print(f"  m2vdb (BF):  {bf_search_slowdown:+.1f}% vs FAISS")
    print(f"  m2vdb (ANN): {ann_search_slowdown:+.1f}% vs FAISS")

if __name__ == "__main__":
    run_benchmark()
