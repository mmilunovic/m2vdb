"""
Unit tests for PQIndex.
"""

import numpy as np
import pytest

from m2vdb.indexes.pq import PQIndex


class TestPQIndexBuild:
    def test_build_valid(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)
        assert idx.is_built
        assert idx.size() == 200

    def test_build_cosine(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="cosine")
        idx.build(vectors_128d, ids_128d)
        assert idx.is_built

    def test_build_duplicate_ids_raises(self, vectors_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        ids = [f"vec-{i}" for i in range(200)]
        ids[1] = ids[0]  # duplicate
        with pytest.raises(AssertionError):
            idx.build(vectors_128d, ids)

    def test_build_mismatched_lengths_raises(self, vectors_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        with pytest.raises(AssertionError):
            idx.build(vectors_128d, ["a", "b"])

    def test_build_indivisible_dimension_raises(self):
        idx = PQIndex(n_subvectors=7, n_clusters=8, metric="euclidean")
        vectors = np.random.rand(50, 128).astype(np.float32)
        ids = [f"v{i}" for i in range(50)]
        with pytest.raises(AssertionError):
            idx.build(vectors, ids)

    def test_not_built_initially(self):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        assert not idx.is_built

    def test_can_build_threshold(self):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        assert not idx.can_build(10)  # < n_clusters
        assert idx.can_build(16)  # == n_clusters
        assert idx.can_build(100)  # > n_clusters

    def test_codebooks_shape(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)
        assert idx.codebooks.shape == (8, 16, 16)  # (m, k, subvec_dim)

    def test_quantized_codes_shape(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)
        assert idx.quantized_codes.shape == (200, 8)  # (n_vectors, m)


class TestPQIndexSearch:
    def test_search_returns_k_results(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=5)
        assert len(results) == 5

    def test_search_k_greater_than_n(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=1000)
        assert len(results) == 200

    def test_search_k_zero(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=0)
        assert len(results) == 0

    def test_search_results_sorted(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=10)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_search_distances_non_negative(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        results = idx.search(vectors_128d[0], k=10)
        for _, dist in results:
            assert dist >= -1e-6

    def test_search_not_built(self):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        results = idx.search(np.random.rand(128).astype(np.float32), k=5)
        assert len(results) == 0


class TestPQIndexAdd:
    def test_add_after_build(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d[:100], ids_128d[:100])
        assert idx.size() == 100

        idx.add("new-vec", vectors_128d[100])
        assert idx.size() == 101

    def test_add_duplicate_raises(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        with pytest.raises(ValueError, match="already exists"):
            idx.add(ids_128d[0], vectors_128d[0])

    def test_add_before_build_raises(self):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        with pytest.raises(RuntimeError):
            idx.add("v0", np.random.rand(128).astype(np.float32))


class TestPQIndexDelete:
    def test_delete_existing(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        assert idx.delete(ids_128d[0]) is True
        assert idx.size() == 199

    def test_delete_nonexistent(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        assert idx.delete("nonexistent") is False

    def test_delete_then_search_excludes(self, vectors_128d, ids_128d):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        target_id = ids_128d[0]
        idx.delete(target_id)
        results = idx.search(vectors_128d[0], k=200)
        result_ids = [id for id, _ in results]
        assert target_id not in result_ids


class TestPQIndexPersistence:
    def test_save_load_roundtrip(self, vectors_128d, ids_128d, tmp_path):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx.build(vectors_128d, ids_128d)

        artifacts_dir = str(tmp_path / "artifacts")
        import os
        os.makedirs(artifacts_dir)
        idx.save_artifacts(artifacts_dir)

        idx2 = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx2.load_artifacts(artifacts_dir)
        assert idx2.is_built
        assert idx2.size() == 200

        # Search should produce same results
        query = vectors_128d[0]
        results1 = idx.search(query, k=5)
        results2 = idx2.search(query, k=5)
        assert [id for id, _ in results1] == [id for id, _ in results2]

    def test_save_not_built(self, tmp_path):
        idx = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        artifacts_dir = str(tmp_path / "artifacts")
        import os
        os.makedirs(artifacts_dir)
        idx.save_artifacts(artifacts_dir)

        idx2 = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        idx2.load_artifacts(artifacts_dir)
        assert not idx2.is_built


class TestPQIndexMemory:
    def test_memory_compression(self, vectors_128d, ids_128d):
        """PQ memory should be much less than brute force."""
        from m2vdb.indexes.brute_force import BruteForceIndex

        bf = BruteForceIndex(metric="euclidean")
        bf.build(vectors_128d, ids_128d)

        pq = PQIndex(n_subvectors=8, n_clusters=16, metric="euclidean")
        pq.build(vectors_128d, ids_128d)

        # PQ should use significantly less memory for the index
        assert pq.memory_usage() < bf.memory_usage()
