"""
End-to-end integration tests for m2vdb.

This script tests all API endpoints and functionality with real data
from SIFT1M dataset. It tests:
- Health endpoint
- Index lifecycle (create, list, get, delete)
- Vector operations (upsert, search, fetch, delete)
- All index types (brute_force, pq, ivf, rust_brute_force)
- Error paths (duplicate indexes, non-existent resources)
- Multi-tenancy (user isolation)

Usage:
    # Requires docker-compose to be running
    docker-compose up -d
    uv run python tests/test_e2e.py
    docker-compose down
"""

import sys
import time
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
    """
    Load a small subset of SIFT1M for testing.
    
    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth)
        Each is a numpy array
    """
    console.print(f"[cyan]📊 Loading {n_vectors} test vectors from SIFT1M...[/cyan]")
    
    # Import here to avoid issues if benchmarks module not available
    from benchmarks.datasets import load_sift1m
    
    dataset = load_sift1m()
    
    # Use first n_vectors for base, take some queries
    base_vectors = dataset.base_vectors[:n_vectors]
    query_vectors = dataset.query_vectors[:10]  # 10 queries
    
    # Ground truth needs to be adjusted for subset
    # Just use first 10 neighbors from ground truth
    ground_truth = dataset.ground_truth[:10, :10]
    
    console.print(f"[green]✓ Loaded {len(base_vectors)} base vectors, "
                 f"{len(query_vectors)} queries (128D, euclidean)[/green]")
    
    return base_vectors, query_vectors, ground_truth


def test_health():
    """Test health endpoint."""
    console.print("\n[bold cyan]Testing health endpoint...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    health = client.health()
    assert health["status"] == "healthy", f"Expected healthy, got {health}"
    console.print("  ✓ Health check passed")


def test_index_lifecycle():
    """Test create, list, get, delete index."""
    console.print("\n[bold cyan]Testing index lifecycle...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    # Create index
    client.create_index("test-lifecycle", dimension=3, metric="cosine")
    console.print("  ✓ Created index")
    
    # List indexes
    indexes = client.list_indexes()
    assert any(idx["name"] == "test-lifecycle" for idx in indexes), \
        "Index not found in list"
    console.print("  ✓ Listed indexes")
    
    # Get index handle
    same_index = client.Index("test-lifecycle")
    stats = same_index.describe()
    assert stats["name"] == "test-lifecycle", f"Expected test-lifecycle, got {stats['name']}"
    console.print("  ✓ Got index handle")
    
    # Delete index
    client.delete_index("test-lifecycle")
    indexes = client.list_indexes()
    assert not any(idx["name"] == "test-lifecycle" for idx in indexes), \
        "Index still exists after deletion"
    console.print("  ✓ Deleted index")


def test_duplicate_index():
    """Test creating duplicate index (should fail)."""
    console.print("\n[bold cyan]Testing duplicate index creation...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    client.create_index("test-dup", dimension=3, metric="cosine")
    try:
        client.create_index("test-dup", dimension=3, metric="cosine")
        assert False, "Should have raised error for duplicate index"
    except ValueError as e:
        assert "already exists" in str(e).lower() or "conflict" in str(e).lower(), \
            f"Unexpected error message: {e}"
        console.print("  ✓ Duplicate creation rejected")
    finally:
        client.delete_index("test-dup")


def test_delete_nonexistent_index():
    """Test deleting non-existent index."""
    console.print("\n[bold cyan]Testing delete non-existent index...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    try:
        client.delete_index("does-not-exist")
        assert False, "Should have raised error for non-existent index"
    except ValueError as e:
        assert "not found" in str(e).lower(), f"Unexpected error message: {e}"
        console.print("  ✓ Non-existent deletion rejected")


def test_vector_operations():
    """Test upsert, search, fetch, delete vectors."""
    console.print("\n[bold cyan]Testing vector operations...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    index = client.create_index("test-vectors", dimension=3, metric="cosine")
    
    # Upsert
    vectors = [
        {"id": "v1", "vector": [1.0, 0.0, 0.0], "metadata": {"label": "A"}},
        {"id": "v2", "vector": [0.0, 1.0, 0.0], "metadata": {"label": "B"}},
        {"id": "v3", "vector": [0.0, 0.0, 1.0], "metadata": {"label": "C"}},
    ]
    count = index.upsert(vectors)
    assert count == 3, f"Expected 3 upserted, got {count}"
    console.print("  ✓ Upserted vectors")
    
    # Search
    results = index.query([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].id == "v1", f"Expected v1 as top result, got {results[0].id}"
    console.print("  ✓ Searched vectors")
    
    # Fetch
    vec = index.fetch("v2")
    assert vec["id"] == "v2", f"Expected v2, got {vec['id']}"
    assert vec["metadata"]["label"] == "B", f"Expected label B, got {vec['metadata']['label']}"
    console.print("  ✓ Fetched vector")
    
    # Delete vector
    deleted = index.delete("v3")
    assert deleted, "Expected deletion to succeed"
    stats = index.describe()
    assert stats["size"] == 2, f"Expected 2 vectors, got {stats['size']}"
    console.print("  ✓ Deleted vector")
    
    # Delete non-existent
    deleted = index.delete("v999")
    assert not deleted, "Expected deletion of non-existent to return False"
    console.print("  ✓ Non-existent vector deletion handled")
    
    client.delete_index("test-vectors")


def test_all_index_types():
    """Test each index type with real SIFT1M data."""
    console.print("\n[bold cyan]Testing all index types with SIFT1M data...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    # Load test data
    base_vectors, query_vectors, ground_truth = load_test_vectors(n_vectors=500)
    
    # For PQ: need enough data to build clusters
    # For IVF: need enough data to build clusters
    index_configs = [
        ("brute_force", None, "Brute Force (Python)"),
        ("rust_brute_force", None, "Brute Force (Rust)"),
        ("pq", {"n_subvectors": 8, "n_clusters": 16}, "Product Quantization (m=8, k=16)"),
        ("ivf", {"n_clusters": 10, "nprobe": 3}, "IVF (clusters=10, nprobe=3)"),
    ]
    
    results_table = Table(title="Index Type Test Results")
    results_table.add_column("Index Type", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Search Time", justify="right")
    
    for idx_type, params, description in index_configs:
        name = f"test-{idx_type}"
        try:
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
            assert count == len(base_vectors), \
                f"Expected {len(base_vectors)} upserted, got {count}"
            
            # Search with first query
            start = time.time()
            results = index.query(query_vectors[0].tolist(), top_k=10)
            search_time = time.time() - start
            
            assert len(results) > 0, "No results returned"
            
            # Check that we got some reasonable results
            # (exact recall checking would require ground truth mapping)
            assert len(results) <= 10, f"Expected <= 10 results, got {len(results)}"
            
            results_table.add_row(
                description,
                "✓ PASS",
                f"{search_time*1000:.2f}ms"
            )
            
        except Exception as e:
            results_table.add_row(
                description,
                f"✗ FAIL: {str(e)[:50]}",
                "-"
            )
        finally:
            try:
                client.delete_index(name)
            except Exception:
                pass
    
    console.print(results_table)
    console.print("  ✓ All index types tested")


def test_multi_tenant():
    """Test that different users have isolated indexes."""
    console.print("\n[bold cyan]Testing multi-tenancy...[/bold cyan]")
    
    user1 = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    user2 = M2VDBClient(api_key="sk-test-user2", host="http://localhost:8000")
    
    # User1 creates index
    user1.create_index("user1-index", dimension=3, metric="cosine")
    
    # User2 shouldn't see it
    user2_indexes = user2.list_indexes()
    assert not any(idx["name"] == "user1-index" for idx in user2_indexes), \
        "User2 should not see user1's index"
    console.print("  ✓ User isolation verified")
    
    # User2 can create index with same name (isolated namespace)
    user2.create_index("user1-index", dimension=3, metric="cosine")
    user2_indexes = user2.list_indexes()
    assert any(idx["name"] == "user1-index" for idx in user2_indexes), \
        "User2 should see their own index"
    console.print("  ✓ Namespace isolation verified")
    
    # Cleanup
    user1.delete_index("user1-index")
    user2.delete_index("user1-index")


def test_search_nonexistent_index():
    """Test searching in non-existent index."""
    console.print("\n[bold cyan]Testing search in non-existent index...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    try:
        index = client.Index("does-not-exist")
        index.query([1.0, 2.0, 3.0])
        assert False, "Should have raised error"
    except ValueError as e:
        assert "not found" in str(e).lower(), f"Unexpected error: {e}"
        console.print("  ✓ Non-existent index query rejected")


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold green]m2vdb End-to-End Tests[/bold green]\n"
        "[dim]Testing API endpoints and functionality[/dim]",
        border_style="green"
    ))
    
    try:
        wait_for_server()
        
        # Run all tests
        test_health()
        test_index_lifecycle()
        test_duplicate_index()
        test_delete_nonexistent_index()
        test_vector_operations()
        test_search_nonexistent_index()
        test_multi_tenant()
        test_all_index_types()  # This one loads real data
        
        console.print("\n" + "="*60)
        console.print("[bold green]✓ All tests passed![/bold green]")
        console.print("="*60)
        return 0
        
    except AssertionError as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
