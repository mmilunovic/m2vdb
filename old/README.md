# m2vdb

> A lightweight vector database built from scratch in Python — by someone who doesn't know how to build a vector database. YET!

**m2vdb** is a side project and learning tool designed to understand how vector search really works — from distance metrics to indexing to actual retrieval performance. It's fast(ish), educational, and incredibly not production-ready.

---

## 🚀 Features

- 🔎 **Brute-force exact search**  
  The gold standard... if your gold is measured in CPU cycles.

- ⚡ **IVF-based approximate search**  
  Cluster your vectors. Pretend you’re not brute-forcing inside each cluster. Works well enough.

- 🧮 **Product Quantization (PQ) support**  
  Compress vectors like a boss. Save RAM. Trade accuracy for ✨vibes✨.

- 🧠 **Multiple distance metrics**  
  Choose from Euclidean (for people with taste) or Cosine (for people who forgot to normalize).

- 🧍 **In-memory indexing with persistence**  
  Save and load indexes like it’s 2006. No external dependencies, no secrets. Just files and feelings.

- 📊 **Benchmark suite with FAISS comparisons**  
  See how badly (or surprisingly not badly) we lose to the industry leader.

- 🧼 **Clean, hackable Python codebase**  
  Readable. Extendable. Slightly judgmental if you try to use pandas for metadata.

---

## Why?

This project started during an internal Microsoft hackathon. I wanted to learn how systems like FAISS, Pinecone, and Weaviate actually work — not just how to use them. This is a deep dive into:
- Vector search algorithms
- Index design
- Performance benchmarking
- Mild psychological unraveling


## 🔬 Benchmark Results

**Dataset:** SIFT1M  
**Database vectors:** 100,000  
**Query vectors:** 1,000  
**Dimensions:** 128  

<!-- BENCHMARK_START -->
| 🏷 Method                    | 🛠 Build Time | 🐢 Build vs FAISS | 🧠 Memory (MB) | 🎯 Recall@10 | ⚡ Search Time | 🐢 Search vs FAISS | ⏳ Train Time | 😬 Embarassment Factor™               |
|:----------------------------|--------------:|------------------:|---------------:|-------------:|---------------:|-------------------:|--------------:|:--------------------------------------|
| **FAISS FlatL2**            | 756.29µs      | N/A               | 691.3          | 0.0197       | 6.29ms         | N/A                | N/A           | 😎 *Just works.*                      |
| **m2vdb BruteForce**        | 419.54µs      | 0.6x slower       | 862.3          | 0.0197       | 563.58ms       | 89.6x slower       | N/A           | 😬 *Ashamed of how bad this is.*     |
| **FAISS IVF (nlist=32, 4)** | 1.10ms        | N/A               | 868.2          | 0.0195       | 3.08µs         | N/A                | 7.46ms        | 😎 *Just works.*                      |
| **m2vdb IVF (nlist=32, 4)** | 37.99ms       | 34.5x slower      | 889.6          | 0.0191       | 433.54µs       | 140.8x slower      | 86.80ms       | 🫣 *Legally shouldn't be published.* |
| **FAISS PQ**                | 16.29ms       | N/A               | 411.4          | 0.1303       | 107.65µs       | N/A                | 63.46ms       | 😎 *Just works.*                      |
| **m2vdb PQIndex**           | 89.63ms       | 5.5x slower       | 547.4          | 0.0138       | 2.44ms         | 22.6x slower       | 205.40ms      | 😌 *Actually not that bad.*          |
<!-- BENCHMARK_END -->
> **😬 Embarassment Factor™** — a completely subjective metric for how ashamed you should feel demoing this to another human.

## Install

```bash
git clone https://github.com/mmilunovic/m2vdb.git
cd m2vdb
pip install -r requirements.txt
```

## Example usage

```python
import numpy as np
from m2vdb.database import V3cT0rDaTaBas3

# 1. Create a new vector database using Product Quantization (PQ) index
db = V3cT0rDaTaBas3(
    dim=128,
    index_type="pq",
    storage_path="my_vector_db_pq",
    index_params={"num_subspaces": 8, "centroids_per_subspace": 16}
)

# 2. Add vectors with flexible metadata
vectors = np.random.random((500, 128)).astype('float32')
metadata = [{"doc_id": i, "topic": f"topic_{i % 10}", "score": np.random.random()} for i in range(500)]
db.add(vectors=vectors, metadata_list=metadata)

# 3. Perform a search
query = np.random.random((1, 128)).astype('float32')
results = db.search(queries=query, k=5)

# Results contain (id, score, metadata) per match
for match in results[0]:
    print(f"Match ID: {match['id']}, Score: {match['score']:.4f}, Metadata: {match['metadata']}")

# 4. Save the database to disk
db.save()

# 5. Reload it later without retraining
loaded_db = V3cT0rDaTaBas3(storage_path="my_vector_db_pq", load_existing=True)

# 6. Search again on the loaded database
new_results = loaded_db.search(queries=query, k=5)
print(f"Loaded DB top match: {new_results[0][0]}")
```
