"""
Benchmark runner for m2vdb vector database.

Measures build time, search latency, recall, and memory usage
for different index types on various datasets.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from m2vdb import Collection
from .datasets import Dataset
# from benchmarks.cache import BenchmarkCache
from benchmarks.metrics import (
    compute_recall,
    compute_latency_stats,
    measure_index_memory,
    compute_qps
)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single index configuration."""
    index_name: str
    dataset_name: str
    
    # Build metrics
    build_time_ms: float
    
    # Search metrics
    search_latency: Dict[str, float]  # p50, p90, p95, p99, mean in ms
    qps: float
    
    # Quality metrics
    recall: float
    
    # Memory metrics (index structures only, not db._vectors)
    index_mb: float  # Total index memory
    bytes_per_vector: float  # Average bytes per vector
    
    # Config
    n_vectors: int
    dimension: int
    k_searched: int = 10
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.index_name} on {self.dataset_name}: "
            f"build={self.build_time_ms:.1f}ms, "
            f"p99={self.search_latency['p99']:.2f}ms, "
            f"recall={self.recall:.3f})"
        )


class BenchmarkRunner:
    """
    Run benchmarks on vector database indexes.
    
    Usage:
        runner = BenchmarkRunner()
        results = runner.benchmark_index(
            index_name="BruteForce",
            index_factory=lambda: Collection(...),
            dataset=sift_dataset,
            k=10
        )
        runner.print_results([results])
    """
    
    def __init__(self, use_cache: bool = True):
        """Initialize benchmark runner with its own console for output."""
        self.console = Console()
        if use_cache:
            from benchmarks.cache import BenchmarkCache
            self.cache = BenchmarkCache()
        else:
            self.cache = None
    
    def benchmark_index(
        self,
        index_name: str,
        index_factory: Callable[[], Collection],
        dataset: Dataset,
        k: int = 10,
        n_queries: Optional[int] = None,
        seed: int = 42
    ) -> BenchmarkResult:
        """
        Benchmark a single index configuration on a dataset.
        
        Args:
            index_name: Human-readable name for this configuration
            index_factory: Function that creates a Collection instance
            dataset: Dataset to benchmark on
            k: Number of neighbors to search for
            n_queries: Number of queries to run (None = all queries in dataset)
            seed: Random seed for reproducible query sampling
        
        Returns:
            BenchmarkResult with all metrics
        """
        self.console.print(f"\n[bold cyan]Benchmarking {index_name} on {dataset.name}[/bold cyan]")
        
        # Create index
        db = index_factory()

        # Check Cache
        if self.cache:
            cached_result = self.cache.get(
                db=db,
                index_name=index_name,
                dataset=dataset,
                k=k,
                n_queries=n_queries,
                seed=seed
            )
            if cached_result:
                self.console.print("  [green]✓ Found cached result[/green]")
                self.console.print(f"  ✓ Built in {cached_result.build_time_ms:.1f}ms (cached)")
                qps = cached_result.qps if cached_result.qps > 0 else 1.0
                self.console.print(f"  ✓ Searched in {cached_result.n_vectors / qps:.2f}s ({cached_result.qps:.1f} QPS) (cached)")
                self.console.print(f"  ✓ Recall@{k}: {cached_result.recall:.3f} (cached)")
                return cached_result
        
        # Build index (measure time)
        self.console.print(f"  Building index with {len(dataset.base_vectors):,} vectors...")
        build_start = time.perf_counter()
        
        # Upsert all base vectors
        ids = [f"vec_{i}" for i in range(len(dataset.base_vectors))]
        
        # Use batch upsert if available, otherwise loop
        # For now, we'll use the internal rebuild to simulate batch
        for i, (id, vec) in track(
            enumerate(zip(ids, dataset.base_vectors)),
            total=len(ids),
            description="  Indexing",
            console=self.console
        ):
            # Store directly in _vectors to avoid per-vector rebuilds
            db._vectors[id] = vec
        
        # Now rebuild once
        db._rebuild_index()
        
        build_time_ms = (time.perf_counter() - build_start) * 1000
        
        # Measure actual index memory (not RSS which is unreliable)
        mem_stats = measure_index_memory(db)
        
        self.console.print(f"  ✓ Built in {build_time_ms:.1f}ms")
        self.console.print(f"  ✓ Index memory: {mem_stats['index_mb']:.1f}MB ({mem_stats['bytes_per_vector']:.1f} bytes/vec)")
        
        # Prepare queries with reproducible sampling
        query_vectors = dataset.query_vectors
        ground_truth = dataset.ground_truth
        
        if n_queries is not None and n_queries < len(query_vectors):
            # Reproducible random sampling
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(query_vectors), size=n_queries, replace=False)
            query_vectors = query_vectors[indices]
            ground_truth = ground_truth[indices]
        
        # Create ID to index mapping for recall computation
        id_to_idx = {id: i for i, id in enumerate(ids)}
        
        # Search queries (measure latency)
        self.console.print(f"  Searching with {len(query_vectors):,} queries (k={k})...")
        query_times = []
        predictions = []
        
        for query_vec in track(
            query_vectors,
            description="  Querying",
            console=self.console
        ):
            # Search with requested k and measure latency
            query_start = time.perf_counter()
            results = db.search(query_vec, k=k, return_metadata=False)
            query_times.append(time.perf_counter() - query_start)
            predictions.append([r.id for r in results])
        
        # Compute metrics
        latency_stats = compute_latency_stats(query_times)
        total_time = sum(query_times)
        qps = compute_qps(len(query_vectors), total_time)
        
        self.console.print(f"  ✓ Searched {len(query_vectors):,} queries in {total_time:.2f}s ({qps:.1f} QPS)")
        self.console.print(f"  ✓ Latency: p50={latency_stats['p50']:.2f}ms, p99={latency_stats['p99']:.2f}ms")
        
        # Compute recall@k (uses the k results we searched for)
        recall = compute_recall(predictions, ground_truth, id_to_idx, k=k)
        
        self.console.print(f"  ✓ Recall@{k}: {recall:.3f}")
        
        result = BenchmarkResult(
            index_name=index_name,
            dataset_name=dataset.name,
            build_time_ms=build_time_ms,
            search_latency=latency_stats,
            qps=qps,
            recall=recall,
            index_mb=mem_stats['index_mb'],
            bytes_per_vector=mem_stats['bytes_per_vector'],
            n_vectors=len(dataset.base_vectors),
            dimension=dataset.dimension,
            k_searched=k
        )

        # Save to Cache
        if self.cache:
            self.cache.save(result, db, n_queries, seed)
            
        return result
    
    def print_results(self, results: List[BenchmarkResult]) -> None:
        """
        Print benchmark results in a Rich table.
        
        Args:
            results: List of benchmark results to display
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return
        
        # Group by dataset
        by_dataset: Dict[str, List[BenchmarkResult]] = {}
        for result in results:
            if result.dataset_name not in by_dataset:
                by_dataset[result.dataset_name] = []
            by_dataset[result.dataset_name].append(result)
        
        # Print table for each dataset
        for dataset_name, dataset_results in by_dataset.items():
            self.console.print(f"\n[bold]Results for {dataset_name.upper()} ({dataset_results[0].n_vectors:,} vectors, {dataset_results[0].dimension}D)[/bold]\n")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Index", style="cyan", width=20)
            table.add_column("Build (ms)", justify="right")
            table.add_column("Index (MB)", justify="right")
            table.add_column("Bytes/Vec", justify="right")
            table.add_column("QPS", justify="right")
            table.add_column("p50 (ms)", justify="right")
            table.add_column("p99 (ms)", justify="right")
            table.add_column("Recall@k", justify="right")
            
            for result in dataset_results:
                table.add_row(
                    result.index_name,
                    f"{result.build_time_ms:,.0f}",
                    f"{result.index_mb:.1f}",
                    f"{result.bytes_per_vector:.0f}",
                    f"{result.qps:,.0f}",
                    f"{result.search_latency['p50']:.2f}",
                    f"{result.search_latency['p99']:.2f}",
                    f"{result.recall:.3f}",
                )
            
            self.console.print(table)
    
    def compare_indexes(
        self,
        configs: List[Dict[str, Any]],
        dataset: Dataset,
        k: int = 10,
        n_queries: Optional[int] = None,
        seed: int = 42
    ) -> List[BenchmarkResult]:
        """
        Compare multiple index configurations on a single dataset.
        
        Args:
            configs: List of dicts with 'name' and 'factory' keys
            dataset: Dataset to benchmark on
            k: Number of neighbors to search for (default: 10)
            n_queries: Number of queries to run (None = all)
            seed: Random seed for reproducible query sampling
        
        Returns:
            List of benchmark results
        """
        results = []
        
        for config in configs:
            result = self.benchmark_index(
                index_name=config['name'],
                index_factory=config['factory'],
                dataset=dataset,
                k=k,
                n_queries=n_queries,
                seed=seed
            )
            results.append(result)
        
        self.print_results(results)
        return results
