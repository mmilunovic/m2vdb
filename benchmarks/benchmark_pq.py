from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from benchmarks import BenchmarkCase
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage
from m2vdb.indexes import BruteForceIndex, PQIndex


def _execute_case(runner: BenchmarkRunner) -> BenchmarkRunner:
    if faiss is None:  # pragma: no cover - defensive
        raise RuntimeError("faiss is required to run this benchmark")

    xb, xq, _ = load_sift1m(limit_queries=500, limit_vectors=50_000)
    num_subspaces = 8
    centroids_per_subspace = 4

    gt_index = BruteForceIndex(dim=xb.shape[1])
    gt_index.add(xb)
    ground_truth_results = gt_index.search(xq, k=10)

    faiss_index = faiss.IndexPQ(xb.shape[1], num_subspaces, centroids_per_subspace)

    runner.start_timer()
    faiss_index.train(xb)
    faiss_train_time = runner.stop_timer()

    runner.start_timer()
    faiss_index.add(xb)
    faiss_build_time = runner.stop_timer()

    runner.start_timer()
    _, faiss_results = faiss_index.search(xq, 10)
    faiss_search_time = runner.stop_timer()

    faiss_metrics = {
        "recall@10": recall_at_k(faiss_results, ground_truth_results, k=10),
        "train_time": format_time(faiss_train_time),
        "build_time": format_time(faiss_build_time),
        "search_time": format_time(faiss_search_time),
        "memory_mb": get_memory_usage(),
    }

    m2v_index = PQIndex(
        dim=xb.shape[1],
        num_subspaces=num_subspaces,
        centroids_per_subspace=centroids_per_subspace,
    )

    runner.start_timer()
    m2v_index.train(xb)
    m2v_train_time = runner.stop_timer()

    runner.start_timer()
    m2v_index.add(xb)
    m2v_build_time = runner.stop_timer()

    runner.start_timer()
    m2v_results = m2v_index.search(xq, k=10)
    m2v_search_time = runner.stop_timer()

    m2v_metrics = {
        "recall@10": recall_at_k(m2v_results, ground_truth_results, k=10),
        "train_time": format_time(m2v_train_time),
        "build_time": format_time(m2v_build_time),
        "search_time": format_time(m2v_search_time),
        "memory_mb": get_memory_usage(),
        "train_vs_faiss": f"{m2v_train_time / faiss_train_time:.1f}x slower" if faiss_train_time > 0 else "∞",
        "search_vs_faiss": f"{m2v_search_time / faiss_search_time:.1f}x slower" if faiss_search_time > 0 else "∞",
    }

    runner.add_result("FAISS PQ", faiss_metrics)
    runner.add_result("m2vdb PQIndex", m2v_metrics)

    return runner


def build_case() -> BenchmarkCase:
    return BenchmarkCase(
        name="PQ Search Benchmark",
        description="Benchmarking m2vdb PQIndex vs FAISS PQ and brute-force ground truth",
        func=_execute_case,
    )


def run_benchmark(return_runner: bool = False):
    case = build_case()
    runner = case.run()
    if return_runner:
        return runner
    runner.print_results()
    return runner


if __name__ == "__main__":
    run_benchmark()
