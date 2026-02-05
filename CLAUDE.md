# CLAUDE.md - AI Assistant Guide for m2vdb

## Project Overview

m2vdb is an educational vector database built from first principles to understand how vector search really works. It provides a complete implementation covering indexing algorithms, performance benchmarking, and production deployment patterns.

**Author**: Milos Milunovic
**License**: MIT
**Python Version**: 3.12+
**Package Version**: 1.0.0

## Repository Structure

```
m2vdb/
├── m2vdb/                    # Main Python package
│   ├── __init__.py           # Package exports (M2VDBClient, Collection, indexes)
│   ├── cli.py                # CLI server command (m2vdb-server)
│   ├── collection.py         # Collection API - high-level vector collection interface
│   ├── client.py             # Python SDK client for remote server access
│   ├── models.py             # Pydantic models for API requests/responses
│   ├── server.py             # FastAPI server with multi-tenant support
│   ├── storage.py            # Persistence layer with LRU cache and disk storage
│   └── indexes/              # Index implementations
│       ├── base.py           # Abstract Index base class
│       ├── brute_force.py    # Pure Python brute force (exact search)
│       ├── pq.py             # Product Quantization implementation
│       ├── ivf.py            # Inverted File implementation
│       └── rust_brute_force.py # Wrapper for Rust brute force (optional)
├── rust/                     # Rust extensions for performance
│   ├── Cargo.toml            # Rust package manifest
│   └── src/lib.rs            # Rust brute force + PyO3 bindings
├── tests/                    # Test suite
│   ├── test_e2e.py           # End-to-end integration tests
│   ├── test_e2e_persistence.py # Persistence tests
│   ├── run_e2e_tests.sh      # E2E test runner (uses Docker)
│   ├── run_persistence_tests.sh # Persistence test runner
│   └── run_all_tests.sh      # Complete test suite runner
├── benchmarks/               # Benchmarking suite
│   ├── benchmark.py          # Benchmark runner and result tracking
│   ├── datasets.py           # Dataset loaders (SIFT1M, FastText)
│   ├── metrics.py            # Metric calculations
│   ├── run_benchmarks.py     # Main benchmark execution script
│   └── faiss_wrappers.py     # FAISS baseline implementations
├── examples/                 # Usage examples
│   └── sdk_usage.py          # SDK client usage example
├── pyproject.toml            # Python project configuration
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-service Docker setup
└── README.md                 # Main documentation
```

## Quick Start Commands

### Development Setup

```bash
# Install dependencies with uv (recommended)
uv sync

# Install with pip
pip install -e .

# Build Rust extensions (optional, for faster search)
uv run maturin develop --release --manifest-path rust/Cargo.toml
```

### Running the Server

```bash
# Start server locally
m2vdb-server --port 8000

# With persistent storage
m2vdb-server --data-dir /path/to/data

# Development mode with auto-reload
m2vdb-server --reload

# Using Docker
docker compose up -d
```

### Running Tests

```bash
# Run all tests (requires Docker)
./tests/run_all_tests.sh

# Run E2E tests only
./tests/run_e2e_tests.sh

# Run persistence tests only
./tests/run_persistence_tests.sh
```

### Linting

```bash
# Check code with Ruff
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix
```

### Running Benchmarks

```bash
uv run python benchmarks/run_benchmarks.py
```

## Architecture

### Core Components

1. **Collection** (`m2vdb/collection.py`): High-level API managing vectors, metadata, and indexes
2. **Index implementations** (`m2vdb/indexes/`): Pluggable search algorithms
3. **Server** (`m2vdb/server.py`): FastAPI REST API with multi-tenant support
4. **Client** (`m2vdb/client.py`): Python SDK for remote server access
5. **Storage** (`m2vdb/storage.py`): Persistence layer with LRU cache

### Index Types

| Index | Description | Use Case |
|-------|-------------|----------|
| `brute_force` | Exact linear search | Small datasets, perfect accuracy |
| `rust_brute_force` | Rust-accelerated exact search | 5x faster than Python |
| `ivf` | Inverted File Index | Large datasets, good accuracy |
| `pq` | Product Quantization | Memory-constrained, compression |

### API Authentication

Test API keys (hardcoded in `server.py`):
- `sk-test-user1`
- `sk-test-user2`

## Code Conventions

### Naming

- **Classes**: PascalCase (`Collection`, `BruteForceIndex`, `M2VDBClient`)
- **Functions/Methods**: snake_case (`upsert()`, `batch_upsert()`, `is_built`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_CACHED_COLLECTIONS`)
- **Private methods**: `_method_name()`
- **Index classes**: Always suffix with `Index`

### Type Hints

Always use type hints:
```python
def search(self, query: np.ndarray, k: int) -> List[tuple[str, float]]:
    ...
```

### Docstrings

Use comprehensive docstrings with Args and Returns:
```python
def build(self, vectors: np.ndarray, ids: List[str]) -> None:
    """
    Build the search index structure from a batch of vectors.

    Args:
        vectors: numpy array of shape (n, dim) containing all vectors
        ids: list of string IDs corresponding to each vector
    """
```

### Index Implementation Pattern

All indexes inherit from `Index` ABC (`m2vdb/indexes/base.py`):

```python
class Index(ABC):
    @property
    @abstractmethod
    def is_built(self) -> bool: ...

    @abstractmethod
    def build(self, vectors: np.ndarray, ids: List[str]) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[tuple[str, float]]: ...

    @abstractmethod
    def add(self, id: str, vector: np.ndarray) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...

    @abstractmethod
    def size(self) -> int: ...
```

### Error Handling

- Use custom exceptions where appropriate (`CollectionNotFound`)
- Proper HTTP status codes in FastAPI endpoints
- Assertions for internal contracts
- Validation through Pydantic models

## Testing Guidelines

### Test Structure

- Tests use Docker Compose for the server environment
- E2E tests verify all API endpoints and index types
- Persistence tests verify data survives server restarts

### Running Individual Test Files

```bash
# Start services first
docker compose up -d

# Run specific test
uv run python tests/test_e2e.py
```

### Test API Keys

Use `sk-test-user1` or `sk-test-user2` for test requests.

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Lint & Smoke Tests**: Ruff linting, Rust build, import check
2. **Integration Tests**: Docker E2E tests
3. **Persistence Tests**: Disk storage and recovery tests

## Key Files to Understand

| File | Purpose |
|------|---------|
| `m2vdb/collection.py` | Core Collection API - start here for understanding the data model |
| `m2vdb/indexes/base.py` | Abstract base class defining the index interface |
| `m2vdb/server.py` | REST API endpoints and multi-tenancy |
| `m2vdb/storage.py` | Persistence and caching layer |
| `m2vdb/client.py` | SDK for consuming the API |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `M2VDB_DATA_DIR` | Directory for persistent storage | None (in-memory) |

## Docker

### Build and Run

```bash
docker compose build
docker compose up -d
```

### Ports

- `8000`: HTTP API server

### Volumes

- `m2vdb-data`: Persistent data storage at `/data`

## Rust Extension

The optional Rust extension provides 5x faster brute force search:

```bash
# Build Rust extension
uv run maturin develop --release --manifest-path rust/Cargo.toml

# Verify it's available
python -c "from m2vdb import HAS_RUST; print(f'Rust available: {HAS_RUST}')"
```

The Rust code uses PyO3 for Python bindings and supports zero-copy NumPy array access.

## Common Tasks

### Adding a New Index Type

1. Create `m2vdb/indexes/your_index.py` inheriting from `Index`
2. Implement all abstract methods
3. Export in `m2vdb/indexes/__init__.py`
4. Add to `Collection._create_index()` factory method
5. Add tests in `tests/test_e2e.py`

### Modifying API Endpoints

1. Edit `m2vdb/server.py` for endpoint changes
2. Update Pydantic models in `m2vdb/models.py` if needed
3. Update SDK in `m2vdb/client.py` to match
4. Add/update tests in `tests/test_e2e.py`

### Running Benchmarks

```bash
# Full benchmark suite
uv run python benchmarks/run_benchmarks.py

# Results are cached in benchmarks/cache/
```

## Troubleshooting

### Import Errors

```bash
# Ensure dependencies are installed
uv sync

# Check if Rust extension is available
python -c "from m2vdb import HAS_RUST; print(HAS_RUST)"
```

### Docker Issues

```bash
# View logs
docker compose logs

# Rebuild from scratch
docker compose down && docker compose build --no-cache && docker compose up -d
```

### Test Failures

```bash
# Check server health
curl http://localhost:8000/health

# Ensure Docker services are running
docker compose ps
```
