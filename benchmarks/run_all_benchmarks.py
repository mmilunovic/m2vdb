from __future__ import annotations

from benchmarks import BenchmarkSuite
from benchmarks.benchmark_brute_force import build_case as brute_case
from benchmarks.benchmark_ivf import build_case as ivf_case
from benchmarks.benchmark_pq import build_case as pq_case


def run_all_benchmarks():
    suite = BenchmarkSuite([brute_case(), ivf_case(), pq_case()])
    report = suite.run()
    report.print()
    return report


if __name__ == "__main__":
    run_all_benchmarks()
