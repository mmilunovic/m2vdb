"""
Unit tests for CollectionManager (storage/persistence layer).
"""

import numpy as np
import pytest

from m2vdb.storage import CollectionManager, MAX_CACHED_COLLECTIONS
from m2vdb.exceptions import CollectionNotFoundError


class TestCollectionManagerCreate:
    def test_create_collection(self, tmp_storage):
        c = tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        assert c.dimension == 3
        assert c.metric == "cosine"

    def test_create_duplicate_raises(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        with pytest.raises(ValueError, match="already exists"):
            tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")

    def test_create_same_name_different_users(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        tmp_storage.create_collection("user2", "test", dimension=3, metric="cosine", index_type="brute_force")
        # Should not raise


class TestCollectionManagerList:
    def test_list_empty(self, tmp_storage):
        assert tmp_storage.list_collections("user1") == []

    def test_list_collections(self, tmp_storage):
        tmp_storage.create_collection("user1", "alpha", dimension=3, metric="cosine", index_type="brute_force")
        tmp_storage.create_collection("user1", "beta", dimension=3, metric="cosine", index_type="brute_force")

        collections = tmp_storage.list_collections("user1")
        assert sorted(collections) == ["alpha", "beta"]

    def test_list_user_isolation(self, tmp_storage):
        tmp_storage.create_collection("user1", "shared", dimension=3, metric="cosine", index_type="brute_force")
        assert tmp_storage.list_collections("user2") == []


class TestCollectionManagerGet:
    def test_get_collection(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        c = tmp_storage.get_collection("user1", "test")
        assert c.dimension == 3

    def test_get_nonexistent_raises(self, tmp_storage):
        with pytest.raises(CollectionNotFoundError):
            tmp_storage.get_collection("user1", "nope")

    def test_get_loads_from_disk(self, tmp_path):
        """Create with one manager, load with a fresh one."""
        mgr1 = CollectionManager(tmp_path / "data")
        mgr1.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")

        # Upsert some data
        mgr1.upsert(
            "user1", "test", "v1",
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            metadata={"label": "A"}
        )

        # Create fresh manager (empty cache)
        mgr2 = CollectionManager(tmp_path / "data")
        c = mgr2.get_collection("user1", "test")
        assert c.dimension == 3
        assert len(c._vectors) == 1
        assert "v1" in c._vectors


class TestCollectionManagerDelete:
    def test_delete_collection(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        tmp_storage.delete_collection("user1", "test")
        assert tmp_storage.list_collections("user1") == []

    def test_delete_nonexistent_raises(self, tmp_storage):
        with pytest.raises(CollectionNotFoundError):
            tmp_storage.delete_collection("user1", "nope")


class TestCollectionManagerUpsert:
    def test_upsert_single(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        tmp_storage.upsert(
            "user1", "test", "v1",
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        c = tmp_storage.get_collection("user1", "test")
        assert len(c._vectors) == 1

    def test_batch_upsert(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        count = tmp_storage.batch_upsert(
            "user1", "test",
            ids=["v1", "v2"],
            vectors=[
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ],
        )
        assert count == 2

    def test_upsert_persists_to_disk(self, tmp_path):
        mgr1 = CollectionManager(tmp_path / "data")
        mgr1.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        mgr1.batch_upsert(
            "user1", "test",
            ids=["v1", "v2"],
            vectors=[
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ],
        )

        # Fresh manager
        mgr2 = CollectionManager(tmp_path / "data")
        c = mgr2.get_collection("user1", "test")
        assert len(c._vectors) == 2

    def test_search_after_upsert(self, tmp_storage):
        tmp_storage.create_collection("user1", "test", dimension=3, metric="cosine", index_type="brute_force")
        tmp_storage.batch_upsert(
            "user1", "test",
            ids=["v1", "v2", "v3"],
            vectors=[
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ],
        )
        c = tmp_storage.get_collection("user1", "test")
        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].id == "v1"


class TestCollectionManagerPersistenceRoundtrip:
    def test_pq_persistence(self, tmp_path):
        rng = np.random.default_rng(42)
        vectors = rng.random((50, 128)).astype(np.float32)
        ids = [f"v{i}" for i in range(50)]

        mgr1 = CollectionManager(tmp_path / "data")
        mgr1.create_collection(
            "user1", "pq-test", dimension=128, metric="euclidean",
            index_type="pq", index_params={"n_subvectors": 8, "n_clusters": 16}
        )
        mgr1.batch_upsert("user1", "pq-test", ids, list(vectors))

        # Verify search works
        c1 = mgr1.get_collection("user1", "pq-test")
        results1 = c1.search(vectors[0], k=5)
        assert len(results1) == 5

        # Load from fresh manager
        mgr2 = CollectionManager(tmp_path / "data")
        c2 = mgr2.get_collection("user1", "pq-test")
        results2 = c2.search(vectors[0], k=5)
        assert len(results2) == 5

        # Results should match
        assert [r.id for r in results1] == [r.id for r in results2]

    def test_ivf_persistence(self, tmp_path):
        rng = np.random.default_rng(42)
        vectors = rng.random((50, 128)).astype(np.float32)
        ids = [f"v{i}" for i in range(50)]

        mgr1 = CollectionManager(tmp_path / "data")
        mgr1.create_collection(
            "user1", "ivf-test", dimension=128, metric="euclidean",
            index_type="ivf", index_params={"n_clusters": 5, "nprobe": 3}
        )
        mgr1.batch_upsert("user1", "ivf-test", ids, list(vectors))

        c1 = mgr1.get_collection("user1", "ivf-test")
        results1 = c1.search(vectors[0], k=5)

        mgr2 = CollectionManager(tmp_path / "data")
        c2 = mgr2.get_collection("user1", "ivf-test")
        results2 = c2.search(vectors[0], k=5)

        assert [r.id for r in results1] == [r.id for r in results2]


class TestCollectionManagerLRUCache:
    def test_cache_eviction(self, tmp_path):
        mgr = CollectionManager(tmp_path / "data")

        # Create more collections than cache size
        for i in range(MAX_CACHED_COLLECTIONS + 2):
            mgr.create_collection(
                "user1", f"col-{i}", dimension=3, metric="cosine", index_type="brute_force"
            )

        # All should still be accessible (loaded from disk after eviction)
        for i in range(MAX_CACHED_COLLECTIONS + 2):
            c = mgr.get_collection("user1", f"col-{i}")
            assert c.dimension == 3
