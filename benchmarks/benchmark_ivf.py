import faiss
from m2vdb.index import IVFIndex
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage
from itertools import product

def run_benchmark():
    """Run benchmark comparing FAISS IVF vs m2vdb IVF implementation"""
    runner = BenchmarkRunner(
        name="IVF Search Configuration Comparison",
        description="Comparing IVF implementations with different nprobe and nlist values"
    )
    
    print("Loading SIFT1M dataset...")
    xb, xq, gt = load_sift1m(limit_queries=1000, limit_vectors=1_000_000)
    print(f"Database vectors: {len(xb):,}")
    print(f"Query vectors:    {len(xq):,}")
    print(f"Dimensions:       {xb.shape[1]}")

    # Configuration grid
    nlist_values = [32, 64, 128, 256, 512]  # number of clusters/cells
    nprobe_values = [4, 8, 12, 16]  # number of cells to visit during search

    # Run benchmarks for each configuration
    for nlist, nprobe in product(nlist_values, nprobe_values):
        print(f"\nTesting configuration: nlist={nlist}, nprobe={nprobe}")
        
        # FAISS IVF
        print("Benchmarking FAISS IVF...")
        quantizer = faiss.IndexFlatL2(xb.shape[1])
        faiss_index = faiss.IndexIVFFlat(quantizer, xb.shape[1], nlist)
        faiss_index.nprobe = nprobe
        
        runner.start_timer()
        faiss_index.train(xb)
        faiss_train_time = runner.stop_timer()
        
        runner.start_timer()
        faiss_index.add(xb)
        faiss_build_time = runner.stop_timer()
        
        runner.start_timer()
        _, faiss_results = faiss_index.search(xq, 10)
        faiss_search_time = runner.stop_timer()
        
        faiss_recall_1 = recall_at_k(faiss_results, gt, k=1)
        faiss_recall_5 = recall_at_k(faiss_results, gt, k=5)
        faiss_recall_10 = recall_at_k(faiss_results, gt, k=10)
        faiss_throughput = len(xq) / faiss_search_time if faiss_search_time > 0 else 0
        
        runner.add_result(f"FAISS IVF (nlist={nlist}, nprobe={nprobe})", {
            "train_time": format_time(faiss_train_time),
            "build_time": format_time(faiss_build_time),
            "search_time": format_time(faiss_search_time),
            "recall@1": faiss_recall_1,
            "recall@5": faiss_recall_5,
            "recall@10": faiss_recall_10,
            "throughput": faiss_throughput,
            "memory_mb": get_memory_usage()
        })

        # m2vdb IVF
        print("Benchmarking m2vdb IVF...")
        m2vdb_index = IVFIndex(xb.shape[1], n_clusters=nlist, n_probe=nprobe)
        
        runner.start_timer()
        m2vdb_index.train(xb)
        m2vdb_train_time = runner.stop_timer()
        
        runner.start_timer()
        m2vdb_index.add(xb)
        m2vdb_build_time = runner.stop_timer()

        runner.start_timer()
        m2vdb_results = m2vdb_index.search(xq, k=10)
        m2vdb_search_time = runner.stop_timer()

        m2vdb_recall_1 = recall_at_k(m2vdb_results, gt, k=1)
        m2vdb_recall_5 = recall_at_k(m2vdb_results, gt, k=5)
        m2vdb_recall_10 = recall_at_k(m2vdb_results, gt, k=10)
        m2vdb_throughput = len(xq) / m2vdb_search_time if m2vdb_search_time > 0 else 0

        # Calculate relative performance
        build_slowdown = m2vdb_build_time / faiss_build_time if faiss_build_time > 0 else float('inf')
        search_slowdown = m2vdb_search_time / faiss_search_time if faiss_search_time > 0 else float('inf')

        runner.add_result(f"m2vdb IVF (nlist={nlist}, nprobe={nprobe})", {
            "train_time": format_time(m2vdb_train_time),
            "build_time": format_time(m2vdb_build_time),
            "search_time": format_time(m2vdb_search_time),
            "recall@1": m2vdb_recall_1,
            "recall@5": m2vdb_recall_5,
            "recall@10": m2vdb_recall_10,
            "throughput": m2vdb_throughput,
            "memory_mb": get_memory_usage(),
            "build_vs_faiss": f"{build_slowdown:.1f}x slower",
            "search_vs_faiss": f"{search_slowdown:.1f}x slower"
        })
    
    # Print results
    runner.print_results()

if __name__ == "__main__":
    run_benchmark() 