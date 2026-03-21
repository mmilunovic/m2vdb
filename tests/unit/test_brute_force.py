"""
Unit tests for BruteForceIndex.
"""

import numpy as np
import pytest

from m2vdb.indexes.brute_force import BruteForceIndex


class TestBruteForceIndexBuild:
    def test_build_valid(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)
        assert idx.is_built
        assert idx.size() == 4

    def test_build_euclidean(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="euclidean")
        idx.build(vectors_3d, ids_3d)
        assert idx.is_built
        assert idx.size() == 4

    def test_build_duplicate_ids_raises(self, vectors_3d):
        idx = BruteForceIndex(metric="cosine")
        with pytest.raises(AssertionError):
            idx.build(vectors_3d, ["a", "a", "b", "c"])

    def test_build_mismatched_lengths_raises(self, vectors_3d):
        idx = BruteForceIndex(metric="cosine")
        with pytest.raises(AssertionError):
            idx.build(vectors_3d, ["a", "b"])

    def test_not_built_initially(self):
        idx = BruteForceIndex(metric="cosine")
        assert not idx.is_built
        assert idx.size() == 0


class TestBruteForceIndexSearch:
    def test_search_cosine_exact(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        # Query with [1, 0, 0] - nearest should be v0
        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert len(results) == 1
        assert results[0][0] == "v0"
        assert results[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_search_euclidean_exact(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="euclidean")
        idx.build(vectors_3d, ids_3d)

        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert len(results) == 1
        assert results[0][0] == "v0"
        assert results[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_search_returns_k_results(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3)
        assert len(results) == 3

    def test_search_k_greater_than_n(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=100)
        assert len(results) == 4  # Only 4 vectors exist

    def test_search_k_zero_returns_empty(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=0)
        assert len(results) == 0

    def test_search_results_sorted_by_distance(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=4)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_search_distances_non_negative(self, vectors_128d, ids_128d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_128d, ids_128d)

        query = vectors_128d[0]
        results = idx.search(query, k=10)
        for _, dist in results:
            assert dist >= -1e-6  # Allow tiny floating point errors

    def test_search_not_built_returns_empty(self):
        idx = BruteForceIndex(metric="cosine")
        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
        assert len(results) == 0


class TestBruteForceIndexAdd:
    def test_add_after_build(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d[:3], ids_3d[:3])
        assert idx.size() == 3

        idx.add("v3", vectors_3d[3])
        assert idx.size() == 4

    def test_add_duplicate_raises(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        with pytest.raises(ValueError, match="already exists"):
            idx.add("v0", vectors_3d[0])

    def test_add_before_build_raises(self):
        idx = BruteForceIndex(metric="cosine")
        with pytest.raises(RuntimeError):
            idx.add("v0", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_add_searchable(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d[:3], ids_3d[:3])

        new_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        idx.add("vnew", new_vec)

        results = idx.search(new_vec, k=1)
        assert results[0][0] == "vnew"


class TestBruteForceIndexDelete:
    def test_delete_existing(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        assert idx.delete("v0") is True
        assert idx.size() == 3

    def test_delete_nonexistent(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        assert idx.delete("nonexistent") is False
        assert idx.size() == 4

    def test_delete_then_search_excludes(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        idx.delete("v0")
        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=4)
        result_ids = [id for id, _ in results]
        assert "v0" not in result_ids

    def test_delete_all_one_by_one(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)

        for id in ids_3d:
            assert idx.delete(id) is True

        assert idx.size() == 0


class TestBruteForceIndexMemory:
    def test_memory_usage_positive(self, vectors_3d, ids_3d):
        idx = BruteForceIndex(metric="cosine")
        idx.build(vectors_3d, ids_3d)
        assert idx.memory_usage() > 0

    def test_memory_usage_empty(self):
        idx = BruteForceIndex(metric="cosine")
        assert idx.memory_usage() == 0
