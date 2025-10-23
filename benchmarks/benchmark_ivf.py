from __future__ import annotations

import warnings
from itertools import product

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from benchmarks import BenchmarkCase
from benchmarks.datasets import load_sift1m
from benchmarks.metrics import recall_at_k
from benchmarks.utils import BenchmarkRunner, format_time, get_memory_usage
from m2vdb.indexes import IVFIndex

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.cluster._kmeans")


def _execute_case(runner: BenchmarkRunner) -> BenchmarkRunner:
    if faiss is None:  # pragma: no cover - defensive
        raise RuntimeError("faiss is required to run this benchmark")

    xb, xq, gt = load_sift1m(limit_queries=500, limit_vectors=20_000)

    for nlist, nprobe in product([32], [4]):
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

        faiss_metrics = {
            "recall@1": recall_at_k(faiss_results, gt, k=1),
            "recall@5": recall_at_k(faiss_results, gt, k=5),
            "recall@10": recall_at_k(faiss_results, gt, k=10),
            "train_time": format_time(faiss_train_time),
            "build_time": format_time(faiss_build_time),
            "search_time": format_time(faiss_search_time),
            "throughput": len(xq) / faiss_search_time if faiss_search_time > 0 else 0,
            "memory_mb": get_memory_usage(),
        }

        m2v_index = IVFIndex(xb.shape[1], n_clusters=nlist, n_probe=nprobe)

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
            "recall@1": recall_at_k(m2v_results, gt, k=1),
            "recall@5": recall_at_k(m2v_results, gt, k=5),
            "recall@10": recall_at_k(m2v_results, gt, k=10),
            "train_time": format_time(m2v_train_time),
            "build_time": format_time(m2v_build_time),
            "search_time": format_time(m2v_search_time),
            "throughput": len(xq) / m2v_search_time if m2v_search_time > 0 else 0,
            "memory_mb": get_memory_usage(),
            "train_vs_faiss": f"{m2v_train_time / faiss_train_time:.1f}x slower" if faiss_train_time > 0 else "∞",
            "search_vs_faiss": f"{m2v_search_time / faiss_search_time:.1f}x slower" if faiss_search_time > 0 else "∞",
        }

        runner.add_result(f"FAISS IVF (nlist={nlist}, nprobe={nprobe})", faiss_metrics)
        runner.add_result(f"m2vdb IVF (nlist={nlist}, nprobe={nprobe})", m2v_metrics)

    return runner


def build_case() -> BenchmarkCase:
    return BenchmarkCase(
        name="IVF Search Benchmark",
        description="Comparing IVF implementations with FAISS vs m2vdb.",
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
