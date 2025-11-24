# m2vdb Roadmap

## High Priority

### Memory Benchmarking Improvements
**Status**: Not Implemented  
**Priority**: Medium

#### Problem
Current memory benchmarking only measures Python object sizes using `sys.getsizeof()`:
- Works for Python-based indexes (PyBruteForce, PQ)
- Returns N/A for Rust indexes and FAISS indexes
- No tracking of memory allocated outside Python (Rust heap, FAISS native memory)

#### Solution
Implement cross-language memory tracking:
- Process-level memory snapshots (e.g., using `psutil.Process().memory_info()`)
- Measure before/after index build to capture total memory delta
- Track both Python heap and system-level allocations
- Properly attribute memory to Rust and FAISS indexes

---

### Batch Operations & Training Flow
**Status**: Not Implemented  
**Priority**: High

#### Problem
Currently, bulk loading vectors through the API is inefficient:
- `upsert()` rebuilds the index after every single insert
- For PQ with 1M vectors: would retrain k-means 1 million times
- Benchmarks work around this by directly accessing `_vectors` dict (internal hack)
- Users have no way to efficiently bulk load data

#### Solution: Implement Batch Upsert
Add `batch_upsert()` method to:
- **VectorDatabase** (`m2vdb/database.py`)
  - Accept batch of vectors, IDs, and metadata
  - Store all in `_vectors` dict at once
  - Call `_rebuild_index()` once (trains PQ k-means once)
  
- **FastAPI Server** (`m2vdb/server.py`)
  - Add `POST /indexes/{name}/vectors/batch` endpoint
  - Accept list of vectors in single request
  
- **Client SDK** (`m2vdb/client.py`)
  - Add `batch_upsert()` method for bulk operations

#### Solution: Elegant Training Flow for Trainable Indexes
Design a better API for indexes that require training (PQ, IVF, HNSW):

**Option 1: Explicit train/build separation**
```python
# User explicitly stages vectors then trains
db = VectorDatabase(index_type='pq', dimension=128)
db.stage_vectors(ids, vectors)  # Store without building
db.stage_vectors(more_ids, more_vectors)  # Add more
db.train()  # Now train k-means on all staged vectors
```

**Option 2: Auto-batching with threshold**
```python
# Automatically batch and train after N vectors
db = VectorDatabase(
    index_type='pq',
    dimension=128,
    rebuild_strategy='threshold',
    rebuild_threshold=10000  # Train after every 10k vectors
)
db.upsert(...)  # Accumulates
db.upsert(...)  # Still accumulating
# Automatically trains after 10k upserts
```

**Option 3: Context manager for bulk operations**
```python
# Training deferred until context exit
with db.bulk_mode():
    for id, vec in data:
        db.upsert(id, vec)
# Trains once here
```

**Decision needed**: Choose which pattern fits best with the rest of the API design.

---

## Medium Priority

### More Index Types
- [ ] **IVF (Inverted File Index)**: Coarse quantization for faster search
- [ ] **HNSW (Hierarchical Navigable Small World)**: Graph-based for high recall
- [ ] **Rust ports**: Port PQ and other indexes to Rust for performance

### Comparative Benchmarks
- [x] **FAISS Integration**: Add `--compare-faiss` flag to benchmarks
- [ ] **More Libraries**: Compare against Annoy, ScaNN, hnswlib
- [ ] **Visualization**: Generate plots comparing recall/QPS/memory trade-offs

### Hyperparameter Tuning
- [ ] **PQ Parameter Sweeps**: Test different m (subvectors) and k (clusters)
- [ ] **Automated Tuning**: Suggest optimal parameters for dataset characteristics
- [ ] **Visualization**: Plot recall vs compression curves

---

## Low Priority

### Configuration Management
- [ ] **YAML/TOML Configs**: Define benchmark configurations in files
- [ ] **Experiment Tracking**: Save results with git commit hash, parameters
- [ ] **Result Comparison**: Tools to compare benchmark runs over time

### Model Context Protocol (MCP)
- [ ] **MCP Server**: Integrate m2vdb as MCP server (for the memes)
- [ ] **Tools**: Expose search, upsert, index management as MCP tools

### API Improvements
- [ ] **Pagination**: For large result sets
- [ ] **Filtering**: Metadata-based filtering during search
- [ ] **Async Operations**: Background index building for large datasets

---

## Notes

### Why Benchmarks Currently Work
Benchmarks bypass the `upsert()` inefficiency by:
```python
# Direct internal access (not public API)
for id, vec in vectors:
    db._vectors[id] = vec  # Store without rebuilding
db._rebuild_index()  # Train once
```

This gives accurate build times (trains PQ once) but:
- ❌ Not available to users via public API
- ❌ Breaks encapsulation (accesses private `_vectors`)
- ❌ Users calling `upsert()` in loop would be 1000x slower

Once `batch_upsert()` is implemented:
- ✅ Benchmarks can use public API
- ✅ Users get same efficient bulk loading
- ✅ Build times remain accurate
