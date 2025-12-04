"""
Run comprehensive benchmarks on m2vdb indexes.

This script benchmarks multiple index types on SIFT1M and FastText datasets,
measuring build time, search latency, recall, and memory usage.

Always uses the full corpus for indexing, with configurable query count.

Usage:
    uv run python benchmarks/run_benchmarks.py [--n-queries N] [--sift] [--fasttext] [--compare-faiss]

Examples:
    # Run all m2vdb benchmarks with 1k queries (default)
    uv run python benchmarks/run_benchmarks.py
    
    # Include FAISS for comparison
    uv run python benchmarks/run_benchmarks.py --compare-faiss
    
    # Quick test with 100 queries
    uv run python benchmarks/run_benchmarks.py --n-queries 100
    
    # Run only SIFT1M benchmarks with FAISS comparison
    uv run python benchmarks/run_benchmarks.py --sift --compare-faiss
    
    # Run with 10k queries for full benchmark
    uv run python benchmarks/run_benchmarks.py --n-queries 10000
"""

import argparse
from rich.console import Console

from benchmarks.datasets import load_sift1m, load_fasttext
from benchmarks.benchmark import BenchmarkRunner
from m2vdb import Collection
from benchmarks.faiss_wrappers import create_faiss_index


def create_benchmark_configs(dimension: int, metric: str, compare_faiss: bool = False):
    """
    Create index configurations for benchmarking.
    
    Args:
        dimension: Vector dimensionality
        metric: Distance metric ('euclidean' or 'cosine')
        compare_faiss: Include FAISS indexes for comparison
    
    Returns:
        List of config dicts with 'name' and 'factory' keys
    """
    configs = []
    
    configs.append({
        'name': f'PyBruteForce-{metric}',
        'factory': lambda: Collection(
            dimension=dimension,
            metric=metric,
            index_type='brute_force'
        )
    })
    
    configs.append({
        'name': f'RustBruteForce-{metric}',
        'factory': lambda: Collection(
            dimension=dimension,
            metric=metric,
            index_type='rust_brute_force'
        )
    })
    
    # PQ: choose subvectors based on dimension
    # SIFT (128D): m=8 → 16D per subvector
    # FastText (300D): m=10 → 30D per subvector
    if dimension == 128:
        m = 8
    elif dimension == 300:
        m = 10
    else:
        m = 8
    
    configs.append({
        'name': f'PQ(m={m},k=256)-{metric}',
        'factory': lambda m=m: Collection(
            dimension=dimension,
            metric=metric,
            index_type='pq',
            index_params={'n_subvectors': m, 'n_clusters': 256}
        )
    })
    
    # IVF: use default smart parameters (auto-determined during build)
    # n_clusters will be set to sqrt(n_vectors) ≈ 1000 for 1M vectors
    # nprobe will be set to sqrt(n_clusters) ≈ 31 for balanced recall/speed
    configs.append({
        'name': f'IVF(auto)-{metric}',
        'factory': lambda: Collection(
            dimension=dimension,
            metric=metric,
            index_type='ivf',
            index_params={'n_clusters': 20, 'nprobe': 5}  # Use all defaults
        )
    })
    
    if compare_faiss:
        configs.append({
            'name': f'FAISS-Flat-{metric}',
            'factory': lambda: create_faiss_index('brute_force', dimension, metric)
        })
        configs.append({
            'name': f'FAISS-PQ(m={m},k=256)-{metric}',
            'factory': lambda m=m: create_faiss_index(
                'pq', dimension, metric,
                index_params={'n_subvectors': m, 'n_clusters': 256}
            )
        })
    
    return configs


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark m2vdb vector database indexes"
    )
    parser.add_argument(
        '--n-queries',
        type=int,
        default=1000,
        help='Number of queries to run (default: 1000)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of nearest neighbors to search for (default: 10)'
    )
    parser.add_argument(
        '--sift',
        action='store_true',
        help='Run only SIFT1M benchmarks'
    )
    parser.add_argument(
        '--fasttext',
        action='store_true',
        help='Run only FastText benchmarks'
    )
    parser.add_argument(
        '--compare-faiss',
        action='store_true',
        help='Include FAISS indexes for comparison'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible query sampling (default: 42)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force re-run of benchmarks, ignoring cached results'
    )
    
    args = parser.parse_args()
    
    # If neither flag is set, run both
    run_sift = args.sift or not args.fasttext
    run_fasttext = args.fasttext or not args.sift
    
    console = Console()
    runner = BenchmarkRunner(use_cache=not args.no_cache)
    
    console.print("[bold green]m2vdb Benchmark Suite[/bold green]")
    if args.compare_faiss:
        console.print("[bold cyan]Including FAISS indexes for comparison[/bold cyan]")
    console.print(f"Queries: {args.n_queries:,}")
    console.print(f"Search k: {args.k}")
    console.print(f"Random seed: {args.seed}\n")
    
    all_results = []
    
    # SIFT1M Benchmarks (128D, Euclidean)
    if run_sift:
        sift = load_sift1m()
        
        console.print("\n[bold cyan]=" * 10)
        console.print("[bold cyan]SIFT1M BENCHMARKS (128D, Euclidean)")
        console.print("[bold cyan]" + "=" * 10)
        
        sift_configs = create_benchmark_configs(
            dimension=sift.dimension,
            metric=sift.metric,
            compare_faiss=args.compare_faiss
        )
        
        sift_results = runner.compare_indexes(
            configs=sift_configs,
            dataset=sift,
            k=args.k,
            n_queries=args.n_queries,
            seed=args.seed
        )
        all_results.extend(sift_results)
    
    # FastText Benchmarks (300D, Cosine)
    if run_fasttext:
        fasttext = load_fasttext()
        
        console.print("\n[bold cyan]=" * 10)
        console.print("[bold cyan]FASTTEXT BENCHMARKS (300D, Cosine)")
        console.print("[bold cyan]" + "=" * 10)
        
        fasttext_configs = create_benchmark_configs(
            dimension=fasttext.dimension,
            metric=fasttext.metric,
            compare_faiss=args.compare_faiss
        )
        
        fasttext_results = runner.compare_indexes(
            configs=fasttext_configs,
            dataset=fasttext,
            k=args.k,
            n_queries=args.n_queries,
            seed=args.seed
        )
        all_results.extend(fasttext_results)
    
    # Summary
    console.print("\n[bold green]" + "=" * 60)
    console.print("[bold green]BENCHMARK COMPLETE")
    console.print("[bold green]" + "=" * 60)
    console.print(f"\nRan {len(all_results)} benchmark(s)")
    
    # Print all results again for easy comparison
    if len(all_results) > 1:
        console.print("\n[bold]Summary:[/bold]")
        runner.print_results(all_results)
    
    console.print("\n[bold green]✓ Done![/bold green]")
    console.print("\n[dim]Tip: Run with --n-queries 100 for faster testing")
    console.print("[dim]Tip: Run with --n-queries 10000 for full benchmark[/dim]")
    console.print("[dim]Tip: Run with --compare-faiss to include FAISS comparisons[/dim]")


if __name__ == "__main__":
    main()
