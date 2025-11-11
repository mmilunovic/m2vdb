"""
Benchmark runner for m2vdb vector database.

Measures build time, search latency, recall, and memory usage
for different index types on various datasets.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from m2vdb import VectorDatabase
from .datasets import Dataset
from .metrics import (
    compute_recall,
    compute_latency_stats,
    measure_memory,
    Timer,
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
    
    # Memory metrics
    memory_mb: float
    memory_per_vector_bytes: float
    
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
            index_factory=lambda: VectorDatabase(...),
            dataset=sift_dataset,
            k=10
        )
        runner.print_results([results])
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize benchmark runner.
        
        Args:
            console: Rich console for output (creates one if not provided)
        """
        self.console = console or Console()
    
    def benchmark_index(
        self,
        index_name: str,
        index_factory: Callable[[], VectorDatabase],
        dataset: Dataset,
        k: int = 10,
        n_queries: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single index configuration on a dataset.
        
        Args:
            index_name: Human-readable name for this configuration
            index_factory: Function that creates a VectorDatabase instance
            dataset: Dataset to benchmark on
            k: Number of neighbors to search for
            n_queries: Number of queries to run (None = all queries in dataset)
        
        Returns:
            BenchmarkResult with all metrics
        """
        self.console.print(f"\n[bold cyan]Benchmarking {index_name} on {dataset.name}[/bold cyan]")
        
        # Create index
        db = index_factory()
        
        # Measure memory before indexing
        mem_before = measure_memory()
        
        # Build index (measure time)
        self.console.print(f"  Building index with {len(dataset.base_vectors):,} vectors...")
        with Timer() as build_timer:
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
        
        build_time_ms = build_timer.elapsed_ms()
        
        # Measure memory after indexing
        mem_after = measure_memory()
        memory_mb = mem_after['rss_mb'] - mem_before['rss_mb']
        memory_per_vector_bytes = (memory_mb * 1024 * 1024) / len(dataset.base_vectors)
        
        self.console.print(f"  ✓ Built in {build_time_ms:.1f}ms")
        self.console.print(f"  ✓ Memory: {memory_mb:.1f}MB ({memory_per_vector_bytes:.1f} bytes/vector)")
        
        # Prepare queries
        query_vectors = dataset.query_vectors
        if n_queries is not None:
            query_vectors = query_vectors[:n_queries]
        
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
            with Timer() as query_timer:
                results = db.search(query_vec, k=k, return_metadata=False)
            query_times.append(query_timer.elapsed)
            predictions.append([r.id for r in results])
        
        # Compute metrics
        latency_stats = compute_latency_stats(query_times)
        total_time = sum(query_times)
        qps = compute_qps(len(query_vectors), total_time)
        
        self.console.print(f"  ✓ Searched {len(query_vectors):,} queries in {total_time:.2f}s ({qps:.1f} QPS)")
        self.console.print(f"  ✓ Latency: p50={latency_stats['p50']:.2f}ms, p99={latency_stats['p99']:.2f}ms")
        
        # Compute recall@k (uses the k results we searched for)
        recall = compute_recall(predictions, dataset.ground_truth, id_to_idx, k=k)
        
        self.console.print(f"  ✓ Recall@{k}: {recall:.3f}")
        
        return BenchmarkResult(
            index_name=index_name,
            dataset_name=dataset.name,
            build_time_ms=build_time_ms,
            search_latency=latency_stats,
            qps=qps,
            recall=recall,
            memory_mb=memory_mb,
            memory_per_vector_bytes=memory_per_vector_bytes,
            n_vectors=len(dataset.base_vectors),
            dimension=dataset.dimension,
            k_searched=k
        )
    
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
            table.add_column("Index", style="cyan", width=15)
            table.add_column("Build (ms)", justify="right")
            table.add_column("Memory (MB)", justify="right")
            table.add_column("Bytes/Vec", justify="right")
            table.add_column("QPS", justify="right")
            table.add_column("p50 (ms)", justify="right")
            table.add_column("p90 (ms)", justify="right")
            table.add_column("p99 (ms)", justify="right")
            table.add_column("Recall", justify="right")
            
            for result in dataset_results:
                table.add_row(
                    result.index_name,
                    f"{result.build_time_ms:,.0f}",
                    f"{result.memory_mb:.1f}",
                    f"{result.memory_per_vector_bytes:.0f}",
                    f"{result.qps:,.0f}",
                    f"{result.search_latency['p50']:.2f}",
                    f"{result.search_latency['p90']:.2f}",
                    f"{result.search_latency['p99']:.2f}",
                    f"{result.recall:.3f}",
                )
            
            self.console.print(table)
    
    def compare_indexes(
        self,
        configs: List[Dict[str, Any]],
        dataset: Dataset,
        k: int = 10,
        limit: Optional[int] = None,
        n_queries: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """
        Compare multiple index configurations on a single dataset.
        
        Args:
            configs: List of dicts with 'name' and 'factory' keys
            dataset: Dataset to benchmark on
            k: Number of neighbors to search for (default: 10)
            limit: Limit number of base vectors to index (None = all)
            n_queries: Limit number of queries to run (None = all)
        
        Returns:
            List of benchmark results
        """
        # Apply limits to dataset
        limited_dataset = dataset
        
        if limit is not None or n_queries is not None:
            base_vectors = dataset.base_vectors[:limit] if limit else dataset.base_vectors
            query_vectors = dataset.query_vectors[:n_queries] if n_queries else dataset.query_vectors
            ground_truth = dataset.ground_truth[:n_queries] if n_queries else dataset.ground_truth
            
            # CRITICAL: If we limit base vectors, we need to filter ground truth
            # Ground truth contains indices into the FULL dataset, but we only have
            # a subset now. We must filter out any ground truth indices >= limit.
            if limit is not None:
                self.console.print(f"\n[yellow]⚠ Warning: Limiting base vectors to {limit:,}. Ground truth will be filtered.[/yellow]")
                self.console.print(f"[yellow]   Recall may be lower than 1.0 even for BruteForce if true neighbors are outside the limit.[/yellow]\n")
                
                # Filter ground truth to only include indices < limit
                filtered_gt = []
                for gt_row in ground_truth:
                    # Keep only valid indices
                    valid_neighbors = [idx for idx in gt_row if idx < limit]
                    # Pad with -1 to maintain shape (we'll ignore -1 in recall calculation)
                    while len(valid_neighbors) < gt_row.shape[0]:
                        valid_neighbors.append(-1)
                    filtered_gt.append(valid_neighbors)
                ground_truth = np.array(filtered_gt, dtype=np.int32)
            
            limited_dataset = Dataset(
                name=dataset.name,
                base_vectors=base_vectors,
                query_vectors=query_vectors,
                ground_truth=ground_truth,
                dimension=dataset.dimension,
                metric=dataset.metric
            )
        
        results = []
        
        for config in configs:
            result = self.benchmark_index(
                index_name=config['name'],
                index_factory=config['factory'],
                dataset=limited_dataset,
                k=k
            )
            results.append(result)
        
        self.print_results(results)
        return results
