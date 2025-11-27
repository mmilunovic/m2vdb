# Persistence and Bulk Ingestion Plan

This document captures the near-term plan for adding disk-backed storage and efficient ingestion while keeping the API surface intentionally small. The goals match the current priorities:
- **Must haves:** local persistence for vectors/metadata, batch ingestion, and an explicit build strategy for trainable indexes.
- **Later:** rich metadata filtering, multi-tenancy, advanced auth, pagination/streaming, and rate limiting.

## Persistence (MVP)
Aim for a minimal but durable local backend that survives restarts without redesigning the API surface.

### Storage layout
- `data_dir/` configured per server instance (default: `~/.m2vdb`).
- Per-index subdirectories: `data_dir/{index_name}/`.
- Files per index:
  - `manifest.json`: dimension, index type, creation time, index-specific params, and schema/version fields.
  - `vectors.bin`: contiguous float32 array in row-major order.
  - `ids.txt`: newline-delimited IDs aligned to vectors.
  - `metadata.jsonl`: optional metadata blobs aligned to IDs (keep as opaque JSON; no filtering yet).
  - `index_artifacts/`: trainable artifacts (e.g., PQ codebooks, IVF coarse centroids) serialized after `build()`.

### Write path
- Gate persistence behind `persist=True` and `data_dir` options on `VectorDatabase` and the server.
- On `upsert`/`batch_upsert`:
  - Append ID/vector/metadata to an in-memory staging buffer.
  - Flush buffer to disk (ids/vectors/metadata files) on either buffer size threshold or explicit `flush()` (called by `build()`), reducing fsync cost.
  - Serialize index artifacts only when `build()` completes successfully.
- Optional follow-ups: WAL for crash safety, background snapshots, and compaction to reclaim tombstoned IDs.

### Read path
- On startup, load `manifest.json` to validate dimension/index type.
- Memory-map or stream `vectors.bin` + `ids.txt` into in-memory structures, then hydrate the index with `build()` or by loading persisted artifacts if present.
- If artifacts exist, skip retraining; otherwise, trigger `build()` once after load.

### API and operational toggles
- `VectorDatabase` config: `persist`, `data_dir`, `flush_threshold`, `auto_build_on_startup` (default true when artifacts exist).
- Server config: environment variables or CLI flags mirroring the database options.
- Observability: emit structured logs when flushing, loading, or building; extend `/health` to report persistence status (data dir reachable, manifest valid).

## Batch ingestion and build strategy
Enable users to load millions of vectors without retraining per insert while keeping control explicit (no magical rebuilds).

### Public API surface
- `VectorDatabase.batch_upsert(ids, vectors, metadata=None, *, rebuild=True | False | "defer")`:
  - Accept iterables/arrays; validate lengths and dimension once.
  - `rebuild` controls index training: `True` (default) triggers `_rebuild_index()` once after the batch; `False` skips; `"defer"` stages data for a later explicit `build()`.
- FastAPI: `POST /indexes/{name}/vectors/batch` with body `{ids, vectors, metadata?, rebuild?}` mirroring the Python API.
- Client SDK: `batch_upsert(..., rebuild=...)` forwarding to the server endpoint.

### Build/defer patterns
- Explicit build flow:
  ```python
  db.batch_upsert(ids, vecs, rebuild="defer")
  # ... more batches ...
  db.build()  # trains once using all staged vectors
  ```
- Threshold-based convenience (optional): database-level `rebuild_threshold` (e.g., every 50k upserts) that triggers a background build when exceeded.
- Bulk context helper (optional later): `with db.bulk_mode(): ...` to auto-defer rebuilds until exit.

### Backwards compatibility
- Keep `upsert()` for small updates: accept `rebuild` flag (default `True`) to align semantics with `batch_upsert()`.
- Maintain existing benchmark flows by switching them to `batch_upsert(..., rebuild="defer")` + `build()`.

## Packaging and deployment niceties (post-persistence)
Lightweight improvements to increase usability without tackling multi-tenant or auth features yet.

- **Packaging:** finalize `pyproject.toml` metadata, add versioning, expose console scripts (`m2vdb-server`) that start uvicorn with config flags, and publish wheels to PyPI (pure-Python + optional Rust extensions).
- **Docker:** add a minimal Dockerfile (uvicorn + `m2vdb[server]`), healthcheck, and sample `docker run` / `docker-compose` snippets with a writable volume for `data_dir`.
- **Health endpoints:** extend `/health` to include build/persistence status and disk space checks.

## Explicitly out of scope for this phase
- Metadata filtering/indexing, auth/multi-tenancy, pagination/streaming, rate limiting, and gRPC/OpenAI/Pinecone compatibility layers.
- Distributed replication/sharding; initial persistence assumes a single-node process.
