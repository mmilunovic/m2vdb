# benchmarks/benchmark.py

import numpy as np
import faiss
import psutil
import os
import re
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

def generate_markdown_table(results_dict):
    """Generates a Markdown table from benchmark results, preserving README formatting."""
    header = (
        "| Method         | 🛠️ Build Time | ⚡ Search Time | 🎯 Recall@10 | 🚀 Throughput (q/s) | 📉 Variance (ms) | 🔍 vs FAISS        | 😬 Embarassment Factor™       |\n"
        "|----------------|----------------|----------------|--------------|---------------------|------------------|---------------------|-------------------------------|"
    )
    
    faiss_res = results_dict.get('FAISS', {})
    bf_res = results_dict.get('m2vdb (BF)', {})
    ann_res = results_dict.get('m2vdb (ANN)', {})
    
    # Format results with appropriate padding and precision
    faiss_build = faiss_res.get('build_time', 'N/A')
    faiss_search = faiss_res.get('search_time', 'N/A')
    faiss_recall = f"{faiss_res.get('recall', 0):.4f}"
    
    bf_build = bf_res.get('build_time', 'N/A')
    bf_search = bf_res.get('search_time', 'N/A')
    bf_recall = f"{bf_res.get('recall', 0):.4f}"
    bf_throughput = f"{bf_res.get('throughput', 0):.1f}"
    bf_variance = f"{bf_res.get('variance', 0):.2f}"
    bf_vs_faiss = f"🔺 +{bf_res.get('vs_faiss_percent', 0):.1f}%"

    ann_build = ann_res.get('build_time', 'N/A')
    ann_search = ann_res.get('search_time', 'N/A')
    ann_recall = f"{ann_res.get('recall', 0):.4f}"
    ann_throughput = f"{ann_res.get('throughput', 0):.1f}"
    ann_variance = f"{ann_res.get('variance', 0):.2f}"
    ann_vs_faiss = f"🔺 +{ann_res.get('vs_faiss_percent', 0):.1f}%"

    # Hardcoded text and structure matching README.md
    # Using padding based on header lengths
    faiss_row = f"| {'**FAISS**':<14} | {faiss_build:<14} | {faiss_search:<15} | {faiss_recall:<14} | {'—':<21} | {'—':<18} | {'—':<21} | {'😎 *"Just works."*':<31} |"
    bf_row =    f"| {'**m2vdb (BF)**':<14} | {bf_build:<14} | {bf_search:<15} | {bf_recall:<14} | {bf_throughput:<21} | {bf_variance:<18} | {bf_vs_faiss:<21} | {'😬 *"Please don\'t look."*':<31} |" # Adjusted quote
    ann_row =   f"| {'**m2vdb (ANN)**':<14} | {ann_build:<14} | {ann_search:<15} | {ann_recall:<14} | {ann_throughput:<21} | {ann_variance:<18} | {ann_vs_faiss:<21} | {'😐 *"Kind of works?"*':<31} |"

    return "\n".join([header, faiss_row, bf_row, ann_row])

def update_readme_benchmark(markdown_table):
    """Updates the benchmark section in README.md."""
    readme_path = "README.md"
    start_marker = "<!-- BENCHMARK_START -->"
    end_marker = "<!-- BENCHMARK_END -->"
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {readme_path} not found.")
        return

    # Use regex to find the content between markers, including the markers themselves
    # Use re.DOTALL to make '.' match newlines
    pattern = re.compile(f"({re.escape(start_marker)}).*?({re.escape(end_marker)})", re.DOTALL)
    
    # Construct the replacement string, keeping the markers and adding the new table
    replacement = f"{start_marker}\n{markdown_table}\n{end_marker}"
    
    new_content, num_replacements = pattern.subn(replacement, content)
    
    if num_replacements > 0:
        try:
            with open(readme_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully updated benchmark table in {readme_path}")
        except IOError:
            print(f"Error: Could not write to {readme_path}.")
    else:
        print(f"Error: Benchmark markers ({start_marker}, {end_marker}) not found in {readme_path}.")

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

    benchmark_results_dict = {}

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

    benchmark_results_dict['FAISS'] = {
        "build_time": format_time(faiss_build_time),
        "search_time": format_time(faiss_search_time),
        "recall": faiss_recall,
    }

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
    m2v_bf_throughput = 1000 / np.mean(bf_query_times) if bf_query_times else 0
    m2v_bf_variance = np.std(bf_query_times) if bf_query_times else 0
    bf_search_slowdown = (m2v_bf_search_time / faiss_search_time - 1) * 100 if faiss_search_time > 0 else float('inf')

    benchmark_results_dict['m2vdb (BF)'] = {
        "build_time": format_time(m2v_bf_build_time),
        "search_time": format_time(m2v_bf_search_time),
        "recall": m2v_bf_recall,
        "throughput": m2v_bf_throughput,
        "variance": m2v_bf_variance,
        "vs_faiss_percent": bf_search_slowdown,
    }

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
    m2v_ann_throughput = 1000 / np.mean(ann_query_times) if ann_query_times else 0
    m2v_ann_variance = np.std(ann_query_times) if ann_query_times else 0
    ann_search_slowdown = (m2v_ann_search_time / faiss_search_time - 1) * 100 if faiss_search_time > 0 else float('inf')

    benchmark_results_dict['m2vdb (ANN)'] = {
        "build_time": format_time(m2v_ann_build_time),
        "search_time": format_time(m2v_ann_search_time),
        "recall": m2v_ann_recall,
        "throughput": m2v_ann_throughput,
        "variance": m2v_ann_variance,
        "vs_faiss_percent": ann_search_slowdown,
    }

    # Relative performance (kept for summary printout)
    bf_build_slowdown_factor = m2v_bf_build_time / faiss_build_time if faiss_build_time > 0 else float('inf')
    ann_build_slowdown_factor = m2v_ann_build_time / faiss_build_time if faiss_build_time > 0 else float('inf')

    # Memory usage
    mem_usage = get_memory_usage()

    # TODO: implement vector saving and measure disk footprint
    disk_footprint = "N/A" # Placeholder

    # Print results table (console)
    print("\nResults (Console):")
    print(f"{'Method':<15} {'Build':<12} {'Search':<12} {'Recall@10':<10} {'Throughput':<10} {'Var(ms)':<10} {'RAM(MB)':<10} {'Disk':<10} {'vs FAISS'}")
    print(f"{'FAISS':<15} {format_time(faiss_build_time):<12} {format_time(faiss_search_time):<12} {faiss_recall:.4f}    {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—'}")
    print(f"{'m2vdb (BF)':<15} {format_time(m2v_bf_build_time):<12} {format_time(m2v_bf_search_time):<12} {m2v_bf_recall:.4f}    {m2v_bf_throughput:<10.1f} {m2v_bf_variance:<10.2f} {mem_usage:<10.1f} {disk_footprint:<10} {bf_search_slowdown:+.1f}%")
    print(f"{'m2vdb (ANN)':<15} {format_time(m2v_ann_build_time):<12} {format_time(m2v_ann_search_time):<12} {m2v_ann_recall:.4f}    {m2v_ann_throughput:<10.1f} {m2v_ann_variance:<10.2f} {mem_usage:<10.1f} {disk_footprint:<10} {ann_search_slowdown:+.1f}%")

    print("\nPerformance Summary (Console):")
    print(f"  Build Time (BF):  {format_time(m2v_bf_build_time)} ({bf_build_slowdown_factor:.1f}x FAISS build time)")
    print(f"  Search Time (BF): {format_time(m2v_bf_search_time)} ({bf_search_slowdown:+.1f}%)")
    print(f"  Build Time (ANN): {format_time(m2v_ann_build_time)} ({ann_build_slowdown_factor:.1f}x FAISS build time)")
    print(f"  Search Time (ANN): {format_time(m2v_ann_search_time)} ({ann_search_slowdown:+.1f}%)")

    # Generate and Update Markdown Table in README
    markdown_table = generate_markdown_table(benchmark_results_dict)
    update_readme_benchmark(markdown_table)

if __name__ == "__main__":
    run_benchmark()
