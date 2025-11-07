vectorlite/
├── vectorlite/
│   ├── __init__.py
│   ├── index.py          # VectorIndex class - core data structure
│   ├── storage.py        # Persistence (save/load to disk)
│   ├── server.py         # FastAPI app and all endpoints
│   └── config.py         # Configuration settings
├── tests/
│   ├── test_index.py     # Test insert, delete, search work correctly
│   └── test_server.py    # Test API endpoints return right responses
├── benchmarks/
│   ├── datasets.py       # Download SIFT1M, GIST1M
│   └── benchmark.py      # Run timing tests, compare to FAISS
├── examples/
│   └── quickstart.py     # Simple usage example
├── pyproject.toml
└── README.md