# m2vdb Overall Roadmap

**Goal:** Portfolio piece demonstrating deep vector DB understanding for systems/infra engineering roles.

## Act 1: Complete the Index Story
1. **HNSW from scratch** (manual implementation, not Claude)
2. **Filtered HNSW** (Qdrant-style adaptive: graph traversal for high selectivity, brute-force for low)
3. **Hybrid search with RRF** (dense + sparse, Reciprocal Rank Fusion)
4. **Pareto benchmark charts** (recall vs QPS curves, memory vs recall, build time scaling)

## Act 2: Storage Engine & Database Fundamentals
5. **WAL (Write-Ahead Log)** - durability, crash recovery, CRC32 checksums
6. **Segment-based storage** - immutable segments, memtable, compaction
7. **IVF-PQ composite index** - coarse IVF + residual PQ compression
8. **mmap vectors** - memory-mapped I/O, OS page cache

## Act 3: Concurrency & Polish
9. **Reader/writer concurrency** - RWLock, snapshot isolation
10. **Final benchmarking** - ann-benchmarks compatible, comprehensive charts

## Learning Resources
- **DDIA Chapter 3** (Storage & Retrieval) - WAL, segments, LSM-trees, compaction
- **CMU 15-445 Lectures 17-18** (Logging & Recovery) - Andy Pavlo on YouTube
- **SQLite WAL docs** - clearest WAL explanation
- **HNSW paper** (Malkov & Yashunin, 2018) - read before implementing
- **Qdrant filtered HNSW blog** - their adaptive approach

## Key Concepts Glossary
- **WAL**: Write operation to log BEFORE applying. Replay on crash to recover.
- **CRC32**: 4-byte checksum to detect data corruption. `zlib.crc32(data)`.
- **fsync**: Force data to physical disk. `os.fsync(fd)`.
- **Segments**: Immutable data chunks. Never modify, only create new + merge old.
- **mmap**: Access files as RAM. OS manages hot/cold pages automatically.
- **RRF**: `score(d) = sum(1/(k+rank_i(d)))` to fuse multiple ranked lists.
