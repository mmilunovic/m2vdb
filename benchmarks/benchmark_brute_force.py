import faiss
from m2vdb.index import BruteForceIndex
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage

def run_benchmark():
    """Run benchmark comparing FAISS FlatL2 vs m2vdb BruteForceIndex"""
    runner = BenchmarkRunner(
        name="Brute Force Search",
        description="Comparing exact search implementations: FAISS FlatL2 vs m2vdb BruteForceIndex"
    )
    
    print("Loading SIFT1M dataset...")
    xb, xq, gt = load_sift1m(limit_queries=1000, limit_vectors=1_000_000)
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")

    # FAISS FlatL2
    print("\nBenchmarking FAISS FlatL2...")
    faiss_index = faiss.IndexFlatL2(xb.shape[1])
    
    runner.start_timer()
    faiss_index.add(xb)
    faiss_build_time = runner.stop_timer()
    
    runner.start_timer()
    _, faiss_results = faiss_index.search(xq, 10)
    faiss_search_time = runner.stop_timer()
    
    faiss_recall = recall_at_k(faiss_results, gt, k=10)
    faiss_throughput = len(xq) / faiss_search_time if faiss_search_time > 0 else 0
    
    runner.add_result("FAISS FlatL2", {
        "build_time": format_time(faiss_build_time),
        "search_time": format_time(faiss_search_time),
        "recall": faiss_recall,
        "throughput": faiss_throughput,
        "memory_mb": get_memory_usage()
    })

    # m2vdb BruteForce
    print("\nBenchmarking m2vdb BruteForceIndex...")
    bf = BruteForceIndex(dim=xb.shape[1])
    
    runner.start_timer()
    bf.add(xb)
    m2v_build_time = runner.stop_timer()
    
    runner.start_timer()
    m2v_results = bf.search(xq, k=10)
    m2v_search_time = runner.stop_timer()
    
    m2v_recall = recall_at_k(m2v_results, gt, k=10)
    m2v_throughput = len(xq) / m2v_search_time if m2v_search_time > 0 else 0
    
    # Calculate relative performance
    build_slowdown = m2v_build_time / faiss_build_time if faiss_build_time > 0 else float('inf')
    search_slowdown = m2v_search_time / faiss_search_time if faiss_search_time > 0 else float('inf')
    
    runner.add_result("m2vdb BruteForce", {
        "build_time": format_time(m2v_build_time),
        "search_time": format_time(m2v_search_time),
        "recall": m2v_recall,
        "throughput": m2v_throughput,
        "memory_mb": get_memory_usage(),
        "build_vs_faiss": f"{build_slowdown:.1f}x slower",
        "search_vs_faiss": f"{search_slowdown:.1f}x slower"
    })

    # Print results
    runner.print_results()

if __name__ == "__main__":
    run_benchmark() 