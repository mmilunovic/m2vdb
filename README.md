# m2vdb

> A lightweight vector database built from scratch in Python — by someone who doesn't know how to build a vector database. YET!

**m2vdb** is a side project and learning tool designed to understand how vector search really works — from distance metrics to indexing to actual retrieval performance. It's fast(ish), educational, and incredibly not production-ready.

## Features

- Brute-force nearest neighbor search
- Simple approximate search
- Support for multiple distance metrics (Euclidean, Cosine)
- In-memory storage with optional persistence
- Basic benchmark suite (with comparisons to FAISS)
- Clean Python codebase focused on readability and hackability

## Why?

This project started during an internal Microsoft hackathon. I wanted to learn how systems like FAISS, Pinecone, and Weaviate actually work — not just how to use them. This is a deep dive into:
- Vector search algorithms
- Index design
- Performance benchmarking
- Mild psychological unraveling


---

## 🔬 Benchmark Results (SIFT1M, 128D vectors)

| Metric                 | `m2vdb` (BruteForce) | `FAISS` (Flat) | Notes |
|------------------------|----------------------|--------------|-------|
| **Recall@10**          | TBD                  | TBD          | Higher is better (1.0 = perfect) |
| **Avg Query Latency**  | TBD                  | TBD          | ms per query (lower is better) |
| **Throughput**         | TBD                  | TBD          | Queries/sec (higher = faster) |
| **Build Time**         | TBD                  | TBD          | Time to prepare index (if needed) |
| **Memory Usage**       | TBD                  | TBD          | Peak memory during search (MB) |
| **Disk Footprint**     | TBD                  | TBD          | Size of stored index, if saved |
| **Query Variance**     | TBD                  | TBD          | Std dev of query times (consistency) |
| **Embarassment Factor™** | 😬 TBD              | 😎 TBD        | Will this run on a friend’s laptop without excuses? |


## Install

```bash
git clone https://github.com/mmilunovic/m2vdb.git
cd m2vdb
pip install -r requirements.txt
```

## Example usage

```python
from m2vdb import VectorDatabase

db = VectorDatabase(dim=128, index_type="brute")

# Add data
db.add(vectors=[[...], [...]], metadata_list=[{...}, {...}])

# Search
results = db.search(queries=[[...]], k=5)

# Save & reload
db.save()
db2 = VectorDatabase(dim=128, index_type="brute")
db2.load()

```