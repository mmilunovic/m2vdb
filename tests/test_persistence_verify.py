"""
Persistence tests for m2vdb.

This script tests that data persists after container restart.
It should be run AFTER docker-compose has been stopped and restarted.

The test assumes that test_e2e_persistence.py has already created
indexes with specific names and data.

Usage:
    # First run creates data
    docker-compose up -d
    uv run python tests/test_e2e_persistence.py --create
    docker-compose down
    
    # Second run verifies persistence
    docker-compose up -d
    uv run python tests/test_persistence_verify.py
    docker-compose down
    
Note: Uses sk-test-user1 API key for authentication.
"""

import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

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


def verify_persistence():
    """Verify that data persisted after restart."""
    console.print("\n[bold cyan]Verifying data persistence...[/bold cyan]")
    client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")
    
    # Check that indexes still exist
    indexes = client.list_indexes()
    expected_indexes = ["persist-brute-force", "persist-pq", "persist-ivf"]
    
    for expected_name in expected_indexes:
        assert any(idx["name"] == expected_name for idx in indexes), \
            f"Index {expected_name} not found after restart!"
        console.print(f"  ✓ Found index: {expected_name}")
    
    # Verify each index has data
    for idx_name in expected_indexes:
        index = client.Index(idx_name)
        stats = index.describe()
        
        assert stats["size"] == 100, \
            f"Index {idx_name} expected 100 vectors, got {stats['size']}"
        console.print(f"  ✓ Index {idx_name} has {stats['size']} vectors")
        
        # Try a search
        results = index.query([1.0] * 128, top_k=5)
        assert len(results) > 0, f"Index {idx_name} search returned no results"
        console.print(f"  ✓ Index {idx_name} search works (got {len(results)} results)")
        
        # Verify we can fetch a vector
        vec = index.fetch("vec-0")
        assert vec["id"] == "vec-0", f"Expected vec-0, got {vec['id']}"
        assert len(vec["vector"]) == 128, f"Expected 128D vector, got {len(vec['vector'])}D"
        console.print(f"  ✓ Index {idx_name} can fetch vectors")
    
    console.print("\n[bold green]✓ All persistence checks passed![/bold green]")
    console.print("[dim]Data successfully persisted across container restart[/dim]")


def main():
    """Run persistence verification tests."""
    console.print(Panel.fit(
        "[bold green]m2vdb Persistence Verification Tests[/bold green]\n"
        "[dim]Verifying data persisted after docker-compose restart[/dim]",
        border_style="green"
    ))
    
    try:
        wait_for_server()
        verify_persistence()
        
        console.print("\n" + "="*60)
        console.print("[bold green]✓ Persistence verification passed![/bold green]")
        console.print("="*60)
        return 0
        
    except AssertionError as e:
        console.print(f"\n[bold red]✗ Persistence test failed: {e}[/bold red]")
        console.print("[dim]Data may not have persisted correctly[/dim]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
