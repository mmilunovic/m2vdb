import numpy as np
import faiss

from m2vdb.index import PQIndex, BruteForceIndex
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage

def run_benchmark(return_runner=False):
    runner = BenchmarkRunner(
        name="PQ Search Benchmark",
        description="Benchmarking m2vdb PQIndex vs FAISS PQ and BruteForce ground-truth"
    )

    print("Loading SIFT1M dataset...")
    xb, xq, gt = load_sift1m(limit_queries=1000, limit_vectors=100_000)
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")

    num_subspaces = 8
    centroids_per_subspace = 4

    # Step 1: Compute BruteForce ground-truth
    print("\nComputing BruteForce ground truth...")
    gt_index = BruteForceIndex(dim=xb.shape[1])
    gt_index.add(xb)
    ground_truth_results = gt_index.search(xq, k=10)

    # Step 2: Benchmark FAISS PQ
    print("\nBenchmarking FAISS PQ...")
    faiss_pq_index = faiss.IndexPQ(xb.shape[1], num_subspaces, centroids_per_subspace)

    runner.start_timer()
    faiss_pq_index.train(xb)
    faiss_pq_train_time = runner.stop_timer()

    runner.start_timer()
    faiss_pq_index.add(xb)
    faiss_pq_build_time = runner.stop_timer()

    runner.start_timer()
    _, faiss_pq_results = faiss_pq_index.search(xq, 10)
    faiss_pq_search_time = runner.stop_timer() / len(xq)

    faiss_pq_recall_10 = recall_at_k(faiss_pq_results, ground_truth_results, k=10)

    runner.add_result("FAISS PQ", {
        "train_time": format_time(faiss_pq_train_time),
        "build_time": format_time(faiss_pq_build_time),
        "search_time": format_time(faiss_pq_search_time),
        "recall@10": faiss_pq_recall_10,
        "memory_mb": get_memory_usage()
    })

    # Step 3: Benchmark m2vdb PQIndex
    print("\nBenchmarking m2vdb PQIndex...")
    pq_index = PQIndex(
        dim=xb.shape[1],
        num_subspaces=num_subspaces,
        centroids_per_subspace=centroids_per_subspace
    )

    runner.start_timer()
    pq_index.train(xb)
    m2vdb_pq_train_time = runner.stop_timer()

    runner.start_timer()
    pq_index.add(xb)
    m2vdb_pq_build_time = runner.stop_timer()

    runner.start_timer()
    m2vdb_pq_results = pq_index.search(xq, k=10)
    m2vdb_pq_search_time = runner.stop_timer() / len(xq)

    m2vdb_pq_recall_10 = recall_at_k(m2vdb_pq_results, ground_truth_results, k=10)

    build_slowdown = m2vdb_pq_build_time / faiss_pq_build_time if faiss_pq_build_time > 0 else float('inf')
    search_slowdown = m2vdb_pq_search_time / faiss_pq_search_time if faiss_pq_search_time > 0 else float('inf')

    runner.add_result("m2vdb PQIndex", {
        "train_time": format_time(m2vdb_pq_train_time),
        "build_time": format_time(m2vdb_pq_build_time),
        "search_time": format_time(m2vdb_pq_search_time),
        "recall@10": m2vdb_pq_recall_10,
        "memory_mb": get_memory_usage(),
        "build_vs_faiss": f"{build_slowdown:.1f}x slower",
        "search_vs_faiss": f"{search_slowdown:.1f}x slower"
    })

    if return_runner:
        return runner
    else:
        runner.print_results()

if __name__ == "__main__":
    run_benchmark()
