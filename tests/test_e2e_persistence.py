"""
Setup script for persistence testing.

This script creates indexes with data that should persist across
docker-compose restarts. Run this BEFORE stopping docker-compose.

Usage:
    docker-compose up -d
    uv run python tests/test_e2e_persistence.py --create
    docker-compose down
    docker-compose up -d
    uv run python tests/test_persistence_verify.py
    
Note: Uses sk-test-user1 API key for authentication.
"""

import sys
import time
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from m2vdb import M2VDBClient


console = Console()


def wait_for_server(max_retries=30, retry_delay=1):
    """Wait for server to be ready."""
    console.print("[yellow]⏳ Waiting for server to be ready...[/yellow]")
    for i in range(max_retries):
        try:
            client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
            health = client.health()
            if health.get("status") == "healthy":
                console.print("[green]✓ Server ready![/green]")
                return
        except Exception:
            if i < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    raise RuntimeError("Server failed to start")


def load_test_vectors(n_vectors=100):
    """Load test vectors from SIFT1M."""
    console.print(f"[cyan]📊 Loading {n_vectors} test vectors from SIFT1M...[/cyan]")
    
    from benchmarks.datasets import load_sift1m
    
    dataset = load_sift1m()
    base_vectors = dataset.base_vectors[:n_vectors]
    
    console.print(f"[green]✓ Loaded {len(base_vectors)} vectors (128D, euclidean)[/green]")
    
    return base_vectors


def create_persistent_data():
    """Create indexes with data that should persist."""
    console.print("\n[bold cyan]Creating data for persistence testing...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    # Load test data
    base_vectors = load_test_vectors(n_vectors=100)
    
    # Create multiple index types
    index_configs = [
        ("persist-brute-force", "brute_force", None, "Brute Force"),
        ("persist-pq", "pq", {"n_subvectors": 8, "n_clusters": 16}, "Product Quantization"),
        ("persist-ivf", "ivf", {"n_clusters": 10, "nprobe": 3}, "IVF"),
    ]
    
    created_table = Table(title="Created Indexes for Persistence Testing")
    created_table.add_column("Index Name", style="cyan")
    created_table.add_column("Type", style="yellow")
    created_table.add_column("Vectors", justify="right")
    created_table.add_column("Status", style="green")
    
    for name, idx_type, params, description in index_configs:
        try:
            # Delete if exists (for re-running)
            try:
                client.delete_index(name)
                console.print(f"  [dim]Cleaned up existing {name}[/dim]")
            except Exception:
                pass
            
            # Create index
            index = client.create_index(
                name,
                dimension=128,
                metric="euclidean",
                index_type=idx_type,
                index_params=params
            )
            
            # Add vectors
            vectors = [
                {"id": f"vec-{i}", "vector": base_vectors[i].tolist()}
                for i in range(len(base_vectors))
            ]
            count = index.upsert(vectors)
            
            # Verify data was added
            index.describe()
            
            created_table.add_row(
                name,
                description,
                str(count),
                "✓ Created"
            )
            
        except Exception as e:
            created_table.add_row(
                name,
                description,
                "-",
                f"✗ Failed: {str(e)[:30]}"
            )
            raise
    
    console.print(created_table)
    console.print("\n[bold green]✓ Data created successfully![/bold green]")
    console.print("[yellow]Now run: docker-compose down && docker-compose up -d[/yellow]")
    console.print("[yellow]Then run: uv run python tests/test_persistence_verify.py[/yellow]")


def cleanup_persistent_data():
    """Clean up all persistence test data."""
    console.print("\n[bold cyan]Cleaning up persistence test data...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    index_names = ["persist-brute-force", "persist-pq", "persist-ivf"]
    
    for name in index_names:
        try:
            client.delete_index(name)
            console.print(f"  ✓ Deleted {name}")
        except Exception as e:
            console.print(f"  [dim]Could not delete {name}: {e}[/dim]")
    
    console.print("[green]✓ Cleanup complete![/green]")


def main():
    """Run persistence test setup or cleanup."""
    parser = argparse.ArgumentParser(
        description="Setup data for persistence testing"
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create data for persistence testing'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up persistence test data'
    )
    
    args = parser.parse_args()
    
    if not args.create and not args.cleanup:
        parser.print_help()
        return 1
    
    console.print(Panel.fit(
        "[bold green]m2vdb Persistence Test Setup[/bold green]\n"
        "[dim]Preparing data for persistence verification[/dim]",
        border_style="green"
    ))
    
    try:
        wait_for_server()
        
        if args.create:
            create_persistent_data()
        elif args.cleanup:
            cleanup_persistent_data()
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
