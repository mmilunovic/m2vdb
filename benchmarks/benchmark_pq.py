import numpy as np
import faiss  # <-- added
from m2vdb.index import PQIndex, BruteForceIndex
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage

def run_benchmark():
    runner = BenchmarkRunner(
        name="PQ Search Benchmark",
        description="Benchmarking FAISS PQ vs m2vdb PQIndex vs BruteForce ground-truth"
    )

    print("Loading SIFT1M dataset...")
    xb, xq, gt = load_sift1m(limit_queries=1000, limit_vectors=100_000)  # smaller, because PQ training
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")

    num_subspaces = 8
    centroids_per_subspace = 16  # => 2^4 = 16 centroids → 4 bits per subvector
    nbits = int(np.log2(centroids_per_subspace))

    ### Ground-truth BruteForce
    print("\nComputing BruteForce ground truth...")
    gt_index = BruteForceIndex(dim=xb.shape[1])
    gt_index.add(xb)
    ground_truth_results = gt_index.search(xq, k=10)

    ### Benchmark FAISS PQ
    print("\nBenchmarking FAISS PQ...")
    faiss_pq = faiss.IndexPQ(xb.shape[1], num_subspaces, nbits)

    runner.start_timer()
    faiss_pq.train(xb)
    faiss_train_time = runner.stop_timer()

    runner.start_timer()
    faiss_pq.add(xb)
    faiss_build_time = runner.stop_timer()

    runner.start_timer()
    _, faiss_results = faiss_pq.search(xq, 10)
    faiss_search_time = runner.stop_timer() / len(xq)

    faiss_recall_1 = recall_at_k(faiss_results, gt, k=1)
    faiss_recall_5 = recall_at_k(faiss_results, gt, k=5)
    faiss_recall_10 = recall_at_k(faiss_results, gt, k=10)
    faiss_throughput = len(xq) / faiss_search_time if faiss_search_time > 0 else 0

    runner.add_result("FAISS PQ", {
        "train_time": format_time(faiss_train_time),
        "build_time": format_time(faiss_build_time),
        "search_time": format_time(faiss_search_time),
        "recall@1": faiss_recall_1,
        "recall@5": faiss_recall_5,
        "recall@10": faiss_recall_10,
        "throughput": faiss_throughput,
        "memory_mb": get_memory_usage()
    })

    ### Benchmark m2vdb PQ
    print("\nBenchmarking m2vdb PQIndex...")
    pq_index = PQIndex(
        dim=xb.shape[1],
        num_subspaces=num_subspaces,
        centroids_per_subspace=centroids_per_subspace
    )

    runner.start_timer()
    pq_index.train(xb)
    pq_train_time = runner.stop_timer()

    runner.start_timer()
    pq_index.add(xb)
    pq_build_time = runner.stop_timer()

    runner.start_timer()
    pq_results = pq_index.search(xq, k=10)
    pq_search_time = runner.stop_timer() / len(xq)

    pq_recall_1 = recall_at_k(pq_results, ground_truth_results, k=1)
    pq_recall_5 = recall_at_k(pq_results, ground_truth_results, k=5)
    pq_recall_10 = recall_at_k(pq_results, ground_truth_results, k=10)
    pq_throughput = len(xq) / pq_search_time if pq_search_time > 0 else 0

    runner.add_result("m2vdb PQIndex", {
        "train_time": format_time(pq_train_time),
        "build_time": format_time(pq_build_time),
        "search_time": format_time(pq_search_time),
        "recall@1": pq_recall_1,
        "recall@5": pq_recall_5,    
        "recall@10": pq_recall_10,
        "throughput": pq_throughput,
        "memory_mb": get_memory_usage()
    })

    # Print results nicely
    runner.print_results()

if __name__ == "__main__":
    run_benchmark()
