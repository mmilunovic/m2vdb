# m2vdb Roadmap

## High Priority

### Persistence & Bulk Ingestion (Near-Term Plan)
**Status**: Not Implemented
**Priority**: High

See `docs/persistence_ingestion_plan.md` for details.

#### Goals
- Add minimal disk-backed storage for vectors/metadata/index artifacts so data survives restarts.
- Provide efficient bulk ingestion without retraining after every insert.
- Keep the API small: explicit build flow with optional deferral.

#### MVP Tasks
- Add persistence toggles (`persist`, `data_dir`, `flush_threshold`) to `VectorDatabase` and the server.
- Implement a staged write path (buffer → flush → serialize artifacts on successful `build()`).
- Add startup load/warmup path using manifests and persisted artifacts.
- Introduce `batch_upsert(..., rebuild=True | False | "defer")` across database, API, and client.
- Extend `/health` to report persistence status (data dir reachable, manifest valid).

#### Nice-to-haves (after MVP)
- Threshold-based background rebuilds for ingestion bursts.
- Bulk mode context manager to defer rebuilds until exit.
- WAL/snapshots/compaction for better durability and cleanup.

---

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
