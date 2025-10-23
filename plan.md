# m2vdb Development Plan

## 1. Fix core API inconsistencies
- Wire the `PQIndex` into `V3cT0rDaTaBas3` so the README example using `index_type="pq"` actually works. This requires teaching the database constructor to instantiate `PQIndex` and to persist its codes/metadata alongside the existing IVF and brute-force paths. 
- Normalize metadata handling so the list-based metadata tracked by `V3cT0rDaTaBas3` and the dict-based metadata stored inside individual indexes stay in sync. Consolidating the single source of truth will simplify persistence and search responses.

## 2. Strengthen persistence and storage utilities
- Extend `IndexManager` to persist and reload PQ indexes, including codebooks, codes, and lookup tables, to match the functionality already available for brute-force and IVF indexes.
- Add cleanup hooks to the test suite (and future CLI tools) to make sure temporary directories under `_data/` or `m2vdb_data/` are removed even on failure, preventing noisy state between runs.

## 3. Improve search quality and evaluation tooling
- Provide evaluation scripts that compare recall/latency across index types using shared datasets (SIFT1M, synthetic, etc.) and emit plots or tables that update the README benchmarks automatically.
- Implement configurable distance metrics beyond Euclidean/Cosine (e.g., dot product) and validate them through both unit tests and benchmark runs.

## 4. Developer experience and API ergonomics
- Introduce a light command-line or FastAPI-based management layer for ingesting data, listing indexes, and running ad-hoc queries without writing scripts each time.
- Expand automated tests to cover the full CRUD lifecycle of `V3cT0rDaTaBas3`, including adding, searching, saving, loading, and deleting vectors for each index type.

## 5. Stretch goals
- Investigate streaming or chunked ingestion paths so very large collections can be added without loading everything into memory at once.
- Explore optional GPU acceleration hooks (e.g., via CuPy or FAISS bindings) while keeping the CPU-first implementation as the default fallback.
