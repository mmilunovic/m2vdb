# benchmarks/benchmark.py

import numpy as np
import faiss
import psutil
import os
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

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB

def run_query_with_timing(index, queries, k):
    times = []
    results = []
    for q in queries:
        start = perf_counter()
        result = index.search([q], k=k)[0]
        end = perf_counter()
        times.append((end - start) * 1000)  # ms
        results.append(result)
    return results, times

def run_benchmark():
    print("VECTOR SIMILARITY SEARCH BENCHMARK")
    print("\nDataset: SIFT1M")
    
    xb, xq, gt = load_sift1m()
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")

    print("\nIndexes:")
    print("FAISS:  IndexFlatL2 (exact search, brute force)")
    print("m2vdb:  BruteForceIndex (exact search)")
    print("m2vdb:  ANNIndex (approximate search, random sampling)")

    # FAISS
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

    # m2vdb Brute Force
    print("\nBuilding m2vdb brute-force index...")
    bf = BruteForceIndex(dim=128)
    start = perf_counter()
    bf.add(xb)
    m2v_bf_build_time = perf_counter() - start

    print("Running m2vdb brute-force search...")
    m2v_bf_results, bf_query_times = run_query_with_timing(bf, xq, k=10)
    m2v_bf_search_time = sum(t / 1000 for t in bf_query_times)  # convert ms to seconds
    m2v_bf_recall = recall_at_k(m2v_bf_results, gt, k=10)
    m2v_bf_throughput = 1000 / np.mean(bf_query_times)
    m2v_bf_variance = np.std(bf_query_times)

    # m2vdb ANN
    print("\nBuilding m2vdb ANN index...")
    ann = ANNIndex(dim=128, num_candidates=1000)
    start = perf_counter()
    ann.add(xb)
    m2v_ann_build_time = perf_counter() - start

    print("Running m2vdb ANN search...")
    m2v_ann_results, ann_query_times = run_query_with_timing(ann, xq, k=10)
    m2v_ann_search_time = sum(t / 1000 for t in ann_query_times)
    m2v_ann_recall = recall_at_k(m2v_ann_results, gt, k=10)
    m2v_ann_throughput = 1000 / np.mean(ann_query_times)
    m2v_ann_variance = np.std(ann_query_times)

    # Relative performance
    bf_search_slowdown = (m2v_bf_search_time / faiss_search_time - 1) * 100
    ann_search_slowdown = (m2v_ann_search_time / faiss_search_time - 1) * 100

    # Memory usage
    mem_usage = get_memory_usage()

    # TODO: implement vector saving and measure disk footprint
    disk_footprint = "N/A"

    # Print results
    print("\nResults:")
    print(f"{'Method':<15} {'Build':<12} {'Search':<12} {'Recall@10':<10} {'Throughput':<10} {'Var(ms)':<10} {'RAM(MB)':<10} {'Disk':<10} {'vs FAISS'}")
    print(f"{'FAISS':<15} {format_time(faiss_build_time):<12} {format_time(faiss_search_time):<12} {faiss_recall:.4f}    {'—':<10} {'—':<10} {'—':<10} {'—':<10}")
    print(f"{'m2vdb (BF)':<15} {format_time(m2v_bf_build_time):<12} {format_time(m2v_bf_search_time):<12} {m2v_bf_recall:.4f}    {m2v_bf_throughput:.1f}     {m2v_bf_variance:.2f}     {mem_usage:.1f}     {disk_footprint:<10} {bf_search_slowdown:+.1f}%")
    print(f"{'m2vdb (ANN)':<15} {format_time(m2v_ann_build_time):<12} {format_time(m2v_ann_search_time):<12} {m2v_ann_recall:.4f}    {m2v_ann_throughput:.1f}     {m2v_ann_variance:.2f}     {mem_usage:.1f}     {disk_footprint:<10} {ann_search_slowdown:+.1f}%")

    print("\nPerformance Summary:")
    print(f"  Build Time (BF):  {format_time(m2v_bf_build_time)}  ({bf_search_slowdown:+.1f}% slower than FAISS)")
    print(f"  Build Time (ANN): {format_time(m2v_ann_build_time)}  ({ann_search_slowdown:+.1f}% slower than FAISS)")

if __name__ == "__main__":
    run_benchmark()
