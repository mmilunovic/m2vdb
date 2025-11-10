import numpy as np
import pytest

from m2vdb.indexes import IVFIndex
from m2vdb.storage import IndexManager


@pytest.fixture
def toy_dataset(dim: int = 16, n_vecs: int = 200, n_queries: int = 5):
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n_vecs, dim), dtype=np.float32)
    queries = rng.standard_normal((n_queries, dim), dtype=np.float32)
    ids = list(range(1_000, 1_000 + n_vecs))
    metadata = {i: {"label": f"item-{i}", "score": float(i % 10) / 10} for i in ids}
    return vecs, queries, ids, metadata


def test_ivf_roundtrip(tmp_path, toy_dataset):
    vecs, queries, ids, metadata = toy_dataset
    index = IVFIndex(dim=vecs.shape[1], metric="euclidean", n_clusters=10, n_probe=3)
    index.train(vecs)
    index.add(vecs, ids=ids, metadata=metadata)

    manager = IndexManager()
    manager.save_index(index, tmp_path.as_posix())

    restored = manager.load_index(tmp_path.as_posix())
    restored_results = restored.search(queries, k=5)
    original_results = index.search(queries, k=5)

    assert np.array_equal(restored_results[0], original_results[0])
    assert np.allclose(restored_results[1], original_results[1])
    assert restored.dim == index.dim
    assert restored.metric == index.metric
    assert sorted(restored.ids) == sorted(index.ids)
    assert np.allclose(restored.centroids, index.centroids)
    assert restored.n_clusters == index.n_clusters
    assert restored.n_probe == index.n_probe
    assert restored._is_trained == index._is_trained

    for cluster_id, entries in index.inverted_lists.items():
        restored_entries = restored.inverted_lists.get(cluster_id, [])
        assert sorted(e[0] for e in entries) == sorted(e[0] for e in restored_entries)

    assert restored.metadata == metadata
