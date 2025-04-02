# tests/test_index.py

import numpy as np
from m2vdb.index import BruteForceIndex

def test_brute_force_search():
    index = BruteForceIndex(dim=2)
    index.add([[1, 1], [2, 2], [3, 3]])
    results = index.search([[2.1, 2.1]], k=2)
    assert results[0][0] in [1, 2]  # Either [2,2] or [3,3]
