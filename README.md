# m2vdb

> A tiny vector database built by someone who absolutely should not have been building a vector database… but did it anyway. 😅  
> It's fine. It works. Mostly.

## Motivation

I’ve worked as an applied scientist on AI systems with retrieval and yet, I never really understood **how** vector databases actually work.

Naturally, I decided to fix this by… attempting to write the whole thing in Rust from scratch, without knowing anything about database internals.  
Shockingly, that did not work out. So the plan pivoted:

Start with Python.  
Implement the core index types.  
Benchmark everything properly.  
Expose an API.  
Then slowly port pieces to Rust while wondering why they’re slower, fix them, write about it, break everything again, repeat.  

This project is simply me trying to understand vector search and databases from first principles, while having fun building something end-to-end that *feels* like a real vector DB.

---

## Features

### 🧱 Index Implementations
- Brute Force (Python)
- Brute Force (Rust)
- Product Quantization (PQ)
- *More Rust ports coming...*

### 🌐 API
- Minimal FastAPI server so you can actually query it
- MCP server planned (for the memes)

### 📊 Benchmarking
- Benchmarks on multiple datasets (SIFT1M, FastText, more coming)
- Latency (p50/p90/p99), recall, build time, memory, QPS
- Caching so identical runs are never re-executed  
- Each run saved as a JSON for later analysis / tables / plotting
---

## Installation

```bash
git clone https://github.com/<your-user>/m2vdb
cd m2vdb
uv sync
````

Rust is optional unless you want to use the Rust-backed indexes.

---

## Running Benchmarks

Just run:

```bash
uv run python benchmarks/run_benchmarks.py
```

It will:

* run the default datasets
* generate Rich tables
* save structured JSON logs
* cache identical runs

This is the main “playground” entry point.

---

## Using the API

Start the server:

```bash
uv run python server.py
```

Then you can index vectors and query nearest neighbors.
Simple, clean, realistic enough to showcase how a tiny vector DB behaves.

---

## Implemented Index Types

| Index Type           | Language | Notes                                    |
| -------------------- | -------- | ---------------------------------------- |
| BruteForce           | Python   | Baseline, clear & easy to inspect        |
| BruteForce           | Rust     | Faster (after I fixed my early mistakes) |
| Product Quantization | Python   | Configurable M / clusters                |
| IVF                  | Python   | Coming soon                              |
| HNSW                 | Python   | Coming after IVF                         |

---

## Benchmarks

All results below were generated on a **MacBook Air M4**, 16GB RAM, with:

* **1,000,000** base vectors
* **1,000** queries
* **k = 10**

### SIFT1M (1,000,000 vectors, 128D)

```markdown
| Index                    | Build(ms) | Index(MB) | Bytes/Vec | QPS | p50(ms) | p99(ms) | Recall@10 |
|--------------------------|-----------|-----------|-----------|-----|---------|---------|-----------|
| PyBruteForce-euclidean   | 746       | 649.0     | 681       | 7   | 140.47  | 204.02  | 1.000     |
| RustBruteForce-euclidean | 698       | 0.0       | 0         | 34  | 28.74   | 40.31   | 1.000     |
| PQ(m=8,k=256)-euclidean  | 425167*   | 191.5     | 201       | 26  | 38.47   | 51.56   | 0.332     |
```

---

### FASTTEXT (1,000,000 vectors, 300D)

```markdown
| Index                    | Build(ms) | Index(MB) | Bytes/Vec | QPS | p50(ms) | p99(ms) | Recall@10 |
|--------------------------|-----------|-----------|-----------|-----|---------|---------|-----------|
| PyBruteForce-cosine      | 707       | 1305.1    | 1369      | 5   | 183.27  | 310.86  | 1.000     |
| RustBruteForce-cosine    | 1074      | 0.0       | 0         | 9   | 115.97  | 128.29  | 1.000     |
| PQ(m=10,k=256)-cosine    | 559221*   | 199.5     | 209       | 22  | 45.67   | 56.49   | 0.283     |
```

---

## License

MIT. If you actually use it I'll be flattered 🥹
