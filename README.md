<div align="center">
  <img src="assets/m2vdb-logo-wide.png" alt="m2vdb logo" width="100%" style="max-width: 600px;"/>
  <h1>m2vdb</h1>
  
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
  [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![CI](https://github.com/mmilunovic/m2vdb/actions/workflows/ci.yml/badge.svg)](https://github.com/mmilunovic/m2vdb/actions/workflows/ci.yml)

  <p><strong>A "First Principles" Vector Database for Learning & Benchmarking</strong></p>
</div>

> This project is simply me trying to understand vector search and databases from first principles, while having fun building something end-to-end that *feels* like a real vector DB.
> I’ve worked as an applied scientist on AI systems with retrieval and yet, I never really understood **how** vector databases actually work. Until now :)

## ✨ Features

<table>
  <tr>
    <td valign="top" width="33%">
      <h4>🧱 Index Implementations</h4>
      <ul>
        <li>Brute Force (Python)</li>
        <li>Brute Force (Rust)</li>
        <li>Product Quantization (PQ)</li>
        <li><i>More Rust ports coming...</i></li>
      </ul>
    </td>
    <td valign="top" width="33%">
      <h4>🌐 API</h4>
      <ul>
        <li>Minimal FastAPI server</li>
        <li>MCP server planned (for the memes)</li>
      </ul>
    </td>
    <td valign="top" width="33%">
      <h4>📊 Benchmarking</h4>
      <ul>
        <li>Benchmarks on multiple datasets (SIFT1M, FastText, more coming)</li>
        <li>Latency, recall, build time, memory, QPS</li>
        <li>Caching benchmark runs & JSON results</li>
      </ul>
    </td>
  </tr>
</table>


## 🗺️ Roadmap

- [ ] **More Indexe**: Implement IVF and HNSW (Python first, Rust when I'm board).
- [ ] **Comparative Benchmarks**: Add FAISS baselines to compare my implementations.
- [ ] **Experiments**: Hyperparameter sweeps for PQ (and others) with visualization/graphs.
- [ ] **Configuration**: Better config management for running benchmark sweeps.
- [ ] **MCP Server**: Model Context Protocol integration (because why not?).
- [ ] **Rust Ports**: Porting more index types to Rust for speed.


## ⚡️ Quick Start

### Installation

```bash
git clone https://github.com/mmilunovic/m2vdb.git
cd m2vdb
uv sync
```

(Optional) Enable Rust-accelerated indexes: ```cd rust & maturin develop --release```

### Start the Server

```bash
uv run uvicorn m2vdb.server:app --reload
```

> 💡 **Tip:** Once the server is running, visit **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive API documentation (Swagger UI) to explore endpoints and test requests directly from your browser.

### Use the Client
```python
from m2vdb import M2VDBClient

# 1. Connect
client = M2VDBClient(api_key="sk-test-user1", host="http://localhost:8000")

# 2. Create Index
index = client.create_index(
    name="demo", 
    dimension=3, 
    metric="cosine",
    index_type="brute_force" # Options: "brute_force", "rust_brute_force", "pq"
)

# 3. Insert Data
index.upsert(
    vectors=[
        {"id": "A", "vector": [1.0, 0.0, 0.0], "metadata": {"label": "Red"}},
        {"id": "B", "vector": [0.0, 1.0, 0.0], "metadata": {"label": "Green"}},
    ]
)

# 4. Search
results = index.query(
    vector=[0.9, 0.1, 0.0],
    top_k=1
)
print(results) # Matches "A" (Red)
```


## 📊 Benchmarks

All results below were generated on a **MacBook Air M4**, 16GB RAM, with:

* **1,000,000** base vectors
* **1,000** queries
* **k = 10**

### SIFT1M (1M vectors, 128D)


| Index                    | Build(ms) | Index(MB) | Bytes/Vec | p99(ms) | Recall@10 |
|--------------------------|-----------|-----------|-----------|---------|-----------|
| PyBruteForce-euclidean   | 746       | 649.0     | 681       | 204.02  | 1.000     |
| RustBruteForce-euclidean | 698       | N/A       | N/A       | 40.31   | 1.000     |
| PQ(m=8,k=256)-euclidean  | 425167*   | 191.5     | 201       | 51.56   | 0.332     |


---

### FASTTEXT (1M vectors, 300D)


| Index                    | Build(ms) | Index(MB) | Bytes/Vec | p99(ms) | Recall@10 |
|--------------------------|-----------|-----------|-----------|---------|-----------|
| PyBruteForce-cosine      | 707       | 1305.1    | 1369      | 310.86  | 1.000     |
| RustBruteForce-cosine    | 1074      | N/A       | N/A       | 128.29  | 1.000     |
| PQ(m=10,k=256)-cosine    | 559221*   | 199.5     | 209       | 56.49   | 0.283     |


---

To reproduce results just run.

```bash
uv run python benchmarks/run_benchmarks.py
```

## 📜 License

MIT. If you actually use it I'll be flattered 🥹
