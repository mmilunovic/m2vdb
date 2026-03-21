"""
Unit tests for IVFIndex.
"""

import numpy as np
import pytest

from m2vdb.indexes.ivf import IVFIndex


class TestIVFIndexBuild:
    def test_build_valid(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)
        assert idx.is_built
        assert idx.size() == 200

    def test_build_cosine(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="cosine", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)
        assert idx.is_built

    def test_build_auto_n_clusters(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean")
        idx.build(vectors_128d, ids_128d)
        # sqrt(200) ~ 14
        assert idx.n_clusters == int(np.sqrt(200))

    def test_build_auto_nprobe(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=16)
        idx.build(vectors_128d, ids_128d)
        assert idx.nprobe == max(1, int(np.sqrt(16)))  # sqrt(16) = 4

    def test_build_n_clusters_gt_n_vectors(self):
        vectors = np.random.rand(5, 128).astype(np.float32)
        ids = [f"v{i}" for i in range(5)]
        idx = IVFIndex(metric="euclidean", n_clusters=100)
        with pytest.warns(UserWarning):
            idx.build(vectors, ids)
        assert idx.n_clusters == 5

    def test_build_duplicate_ids_raises(self, vectors_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10)
        ids = [f"v{i}" for i in range(200)]
        ids[1] = ids[0]
        with pytest.raises(AssertionError):
            idx.build(vectors_128d, ids)

    def test_not_built_initially(self):
        idx = IVFIndex(metric="euclidean")
        assert not idx.is_built
        assert idx.size() == 0


class TestIVFIndexSearch:
    def test_search_returns_k_results(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=5)
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=5)
        assert len(results) == 5

    def test_search_k_greater_than_n(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=10)
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=1000)
        # May not return all 200 if nprobe doesn't cover all clusters
        assert len(results) <= 200

    def test_search_k_zero(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=0)
        assert len(results) == 0

    def test_search_results_sorted(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=5)
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=10)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_search_distances_non_negative(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=5)
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=10)
        for _, dist in results:
            assert dist >= -1e-6

    def test_search_not_built(self):
        idx = IVFIndex(metric="euclidean")
        results = idx.search(np.random.rand(128).astype(np.float32), k=5)
        assert len(results) == 0

    def test_higher_nprobe_more_results(self, vectors_128d, ids_128d):
        """Higher nprobe should search more clusters and potentially find better results."""
        idx_low = IVFIndex(metric="euclidean", n_clusters=20, nprobe=1)
        idx_low.build(vectors_128d.copy(), ids_128d.copy())

        idx_high = IVFIndex(metric="euclidean", n_clusters=20, nprobe=20)
        idx_high.build(vectors_128d.copy(), ids_128d.copy())

        query = vectors_128d[0]
        idx_low.search(query, k=10)  # low nprobe baseline
        results_high = idx_high.search(query, k=10)

        # With nprobe == n_clusters, should get exact results
        # Top-1 result should be the query itself
        assert results_high[0][0] == ids_128d[0]


class TestIVFIndexAdd:
    def test_add_after_build(self, vectors_128d, ids_128d, rng):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d[:100], ids_128d[:100])
        assert idx.size() == 100

        idx.add("new-vec", vectors_128d[100])
        assert idx.size() == 101

    def test_add_duplicate_raises(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)

        with pytest.raises(ValueError, match="already exists"):
            idx.add(ids_128d[0], vectors_128d[0])

    def test_add_before_build_raises(self):
        idx = IVFIndex(metric="euclidean")
        with pytest.raises(RuntimeError):
            idx.add("v0", np.random.rand(128).astype(np.float32))


class TestIVFIndexDelete:
    def test_delete_existing(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)

        assert idx.delete(ids_128d[0]) is True
        assert idx.size() == 199

    def test_delete_nonexistent(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)

        assert idx.delete("nonexistent") is False

    def test_delete_then_search_excludes(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=10)
        idx.build(vectors_128d, ids_128d)

        target_id = ids_128d[0]
        idx.delete(target_id)

        results = idx.search(vectors_128d[0], k=200)
        result_ids = [id for id, _ in results]
        assert target_id not in result_ids


class TestIVFIndexPersistence:
    def test_save_load_roundtrip(self, vectors_128d, ids_128d, tmp_path):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=5)
        idx.build(vectors_128d, ids_128d)

        artifacts_dir = str(tmp_path / "artifacts")
        import os
        os.makedirs(artifacts_dir)
        idx.save_artifacts(artifacts_dir)

        idx2 = IVFIndex(metric="euclidean")
        idx2.load_artifacts(artifacts_dir)
        assert idx2.is_built
        assert idx2.n_clusters == 10
        assert idx2.nprobe == 5

        # Search should produce same results
        query = vectors_128d[0]
        results1 = idx.search(query, k=5)
        results2 = idx2.search(query, k=5)
        assert [id for id, _ in results1] == [id for id, _ in results2]


class TestIVFIndexMemory:
    def test_memory_usage_positive(self, vectors_128d, ids_128d):
        idx = IVFIndex(metric="euclidean", n_clusters=10, nprobe=3)
        idx.build(vectors_128d, ids_128d)
        assert idx.memory_usage() > 0

    def test_memory_usage_empty(self):
        idx = IVFIndex(metric="euclidean")
        assert idx.memory_usage() == 0
