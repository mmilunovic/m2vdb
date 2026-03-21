"""
Unit tests for Collection.
"""

import numpy as np
import pytest

from m2vdb.collection import Collection
from m2vdb.exceptions import DimensionMismatchError


class TestCollectionCreate:
    def test_create_brute_force(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        assert c.dimension == 3
        assert c.metric == "cosine"
        assert c.index_type == "brute_force"
        assert len(c) == 0

    def test_create_pq(self):
        c = Collection(
            dimension=128, metric="euclidean", index_type="pq",
            index_params={"n_subvectors": 8, "n_clusters": 16}
        )
        assert c.index_type == "pq"

    def test_create_ivf(self):
        c = Collection(
            dimension=128, metric="euclidean", index_type="ivf",
            index_params={"n_clusters": 10, "nprobe": 3}
        )
        assert c.index_type == "ivf"

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown index type"):
            Collection(dimension=3, metric="cosine", index_type="unknown")

    def test_create_hnsw_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Collection(dimension=3, metric="cosine", index_type="hnsw")


class TestCollectionUpsert:
    def test_upsert_insert(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert len(c) == 1

    def test_upsert_with_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"label": "A"})
        vec, meta = c.fetch("v1")
        assert meta["label"] == "A"

    def test_upsert_update_existing(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"label": "A"})
        c.upsert("v1", np.array([0.0, 1.0, 0.0], dtype=np.float32), metadata={"label": "B"})
        assert len(c) == 1
        vec, meta = c.fetch("v1")
        np.testing.assert_array_almost_equal(vec, [0.0, 1.0, 0.0])
        assert meta["label"] == "B"

    def test_upsert_update_preserves_metadata_if_none(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"label": "A"})
        c.upsert("v1", np.array([0.0, 1.0, 0.0], dtype=np.float32))  # No metadata
        _, meta = c.fetch("v1")
        assert meta["label"] == "A"

    def test_upsert_dimension_mismatch(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        with pytest.raises(DimensionMismatchError):
            c.upsert("v1", np.array([1.0, 0.0], dtype=np.float32))


class TestCollectionBatchUpsert:
    def test_batch_upsert(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        vectors = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ]
        count = c.batch_upsert(["v1", "v2"], vectors)
        assert count == 2
        assert len(c) == 2

    def test_batch_upsert_with_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        vectors = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ]
        metadata = [{"label": "A"}, {"label": "B"}]
        c.batch_upsert(["v1", "v2"], vectors, metadata)
        _, meta = c.fetch("v2")
        assert meta["label"] == "B"

    def test_batch_upsert_mismatched_lengths(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        with pytest.raises(ValueError, match="same length"):
            c.batch_upsert(["v1"], [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ])

    def test_batch_upsert_dimension_mismatch(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        with pytest.raises(DimensionMismatchError):
            c.batch_upsert(["v1"], [np.array([1.0, 0.0], dtype=np.float32)])


class TestCollectionSearch:
    def test_search_basic(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.batch_upsert(
            ["v1", "v2", "v3"],
            [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ]
        )

        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert len(results) == 1
        assert results[0].id == "v1"

    def test_search_with_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"label": "A"})

        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1, return_metadata=True)
        assert results[0].metadata == {"label": "A"}

    def test_search_without_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"label": "A"})

        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1, return_metadata=False)
        assert results[0].metadata is None

    def test_search_dimension_mismatch(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        with pytest.raises(DimensionMismatchError):
            c.search(np.array([1.0, 0.0], dtype=np.float32), k=1)

    def test_search_empty_collection(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
        assert len(results) == 0


class TestCollectionDelete:
    def test_delete_existing(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert c.delete("v1") is True
        assert len(c) == 0

    def test_delete_nonexistent(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        assert c.delete("v1") is False

    def test_delete_excludes_from_search(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.batch_upsert(
            ["v1", "v2"],
            [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.9, 0.1, 0.0], dtype=np.float32),
            ]
        )
        c.delete("v1")
        results = c.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
        result_ids = [r.id for r in results]
        assert "v1" not in result_ids

    def test_delete_removes_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={"x": 1})
        c.delete("v1")
        assert c.fetch("v1") is None


class TestCollectionFetch:
    def test_fetch_existing(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c.upsert("v1", vec, metadata={"label": "A"})

        result = c.fetch("v1")
        assert result is not None
        np.testing.assert_array_equal(result[0], vec)
        assert result[1]["label"] == "A"

    def test_fetch_nonexistent(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        assert c.fetch("v1") is None

    def test_fetch_no_metadata(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        _, meta = c.fetch("v1")
        assert meta == {}


class TestCollectionStats:
    def test_get_stats_structure(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        c.upsert("v1", np.array([1.0, 0.0, 0.0], dtype=np.float32))

        stats = c.get_stats()
        assert "num_vectors" in stats
        assert "dimension" in stats
        assert "metric" in stats
        assert "index_type" in stats
        assert "memory_mib" in stats
        assert stats["num_vectors"] == 1
        assert stats["dimension"] == 3

    def test_get_stats_empty(self):
        c = Collection(dimension=3, metric="cosine", index_type="brute_force")
        stats = c.get_stats()
        assert stats["num_vectors"] == 0


class TestCollectionRepr:
    def test_repr(self):
        c = Collection(dimension=128, metric="cosine", index_type="brute_force")
        r = repr(c)
        assert "128" in r
        assert "cosine" in r
        assert "brute_force" in r
