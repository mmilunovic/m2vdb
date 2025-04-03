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


## 🔬 Benchmark Results

**Dataset:** SIFT1M  
**Database vectors:** 10,000  
**Query vectors:** 1,000  
**Dimensions:** 128  

| Method         | 🛠️ Build Time | ⚡ Search Time | 🎯 Recall@10 | 🚀 Throughput (q/s) | 📉 Variance (ms) | 🔍 vs FAISS        | 😬 Embarassment Factor™       |
|----------------|----------------|----------------|--------------|---------------------|------------------|---------------------|-------------------------------|
| **FAISS**      | 460.42µs        | 7.14ms         | 0.0197       | —                   | —                | —                   | 😎 *"Just works."*            |
| **m2vdb (BF)** | 458.54µs        | 1422.42ms      | 0.0197       | 703.0               | 0.09             | 🔺 +19808.4%        | 😬 *"Please don’t look."*     |
| **m2vdb (ANN)**| 359.75µs        | 575.61ms       | 0.0019       | 1737.3              | 0.04             | 🔺 +7956.3%         | 😐 *"Kind of works?"*         |

> **😬 Embarassment Factor™** — a completely subjective metric for how ashamed you should feel demoing this to another human.

## Install

```bash
git clone https://github.com/mmilunovic/m2vdb.git
cd m2vdb
pip install -r requirements.txt
```

## Example usage

```python
from m2vdb.database import V3cT0rDaTaBas3
import numpy as np

# Create a new vector database with brute force index
db = V3cT0rDaTaBas3(dim=128, index_type="brute_force", storage_path="my_vector_db")

# Add vectors with metadata
vectors = np.random.random((100, 128)).astype('float32')
metadata = [{"id": i, "doc": f"document_{i}", "category": f"cat_{i % 5}"} for i in range(100)]
db.add(vectors=vectors, metadata_list=metadata)

# Search for similar vectors
query = np.random.random((1, 128)).astype('float32')
results = db.search(queries=query, k=5)
print(f"Top match: {results[0][0]}")

# Save the database
db.save()

# Later, load the existing database
loaded_db = V3cT0rDaTaBas3(storage_path="my_vector_db", load_existing=True)
```