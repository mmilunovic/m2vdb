from benchmarks.benchmark_brute_force import run_benchmark as run_bruteforce
from benchmarks.benchmark_ivf import run_benchmark as run_ivf
from benchmarks.benchmark_pq import run_benchmark as run_pq

from benchmarks.utils import BenchmarkRunner

def run_all_benchmarks():
    print("\n🏁 Starting full benchmark suite...")

    global_runner = BenchmarkRunner(
        name="m2vdb Full Benchmark Suite",
        description="Comparison of BruteForce, IVF, and PQ indexes against FAISS."
    )

    # Run and collect
    print("\n🔎 Running BruteForce benchmark...")
    bruteforce_runner = run_bruteforce(return_runner=True)

    print("\n🏎️ Running IVF benchmark...")
    ivf_runner = run_ivf(return_runner=True)

    print("\n🧠 Running PQ benchmark...")
    pq_runner = run_pq(return_runner=True)

    # Combine results
    for runner in [bruteforce_runner, ivf_runner, pq_runner]:
        for label, result in runner.results.items():
            global_runner.add_result(label, result)

    print("\n📊 Full Benchmark Results:")
    global_runner.print_results()

if __name__ == "__main__":
    run_all_benchmarks()
