import numpy as np
import os
import shutil
from m2vdb.index import IVFIndex
from m2vdb.storage import IndexManager

def test_ivf_index_persistence(
    dim=16,
    n_vecs=200,
    n_queries=5,
    save_path="m2vdb_data/test_ivf_index"
):
    """Test IVFIndex save/load lifecycle with full validation and cleanup."""

    # Ensure clean state
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    try:
        # Generate random data
        np.random.seed(42)
        vecs = np.random.randn(n_vecs, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        ids = list(range(1337, 1337 + n_vecs))

        # 🔎 Create dummy metadata
        metadata = {
            i: {"label": f"item_{i}", "score": float(i % 10) / 10}
            for i in ids
        }

        # Create and train index
        index = IVFIndex(dim=dim, metric="euclidean", n_clusters=10, n_probe=3)
        index.train(vecs)
        index.add(vecs, ids, metadata)
        results_before = index.search(queries, k=5)

        # Save and reload
        manager = IndexManager()
        manager.save_index(index, save_path)
        loaded = manager.load_index(save_path)
        results_after = loaded.search(queries, k=5)

        # Validations
        assert np.array_equal(results_before, results_after), "Search results do not match after load!"
        assert loaded.dim == index.dim
        assert loaded.metric == index.metric
        assert sorted(loaded.ids) == sorted(index.ids), "IDs mismatch (unordered)"
        assert np.allclose(index.centroids, loaded.centroids), "Centroids mismatch"
        assert index.n_clusters == loaded.n_clusters
        assert index.n_probe == loaded.n_probe
        assert index._is_trained == loaded._is_trained

        for cid in index.inverted_lists:
            before_ids = sorted(id_ for id_, _ in index.inverted_lists[cid])
            after_ids = sorted(id_ for id_, _ in loaded.inverted_lists.get(cid, []))
            assert before_ids == after_ids, f"Inverted list mismatch for cluster {cid}"

        # ✅ Metadata check
        assert loaded.metadata is not None, "Metadata missing after load"
        assert len(loaded.metadata) == len(metadata), "Metadata length mismatch"
        for vec_id in list(metadata.keys())[:10]:
            # print(vec_id, metadata[vec_id], loaded.metadata[vec_id])
            assert vec_id in loaded.metadata, f"Missing metadata for ID {vec_id}"
            assert loaded.metadata[vec_id] == metadata[vec_id], f"Metadata mismatch for ID {vec_id}"




        print("✅ IVFIndex persistence test passed.")

    finally:
        # Cleanup
        if os.path.exists(save_path):
            # shutil.rmtree(save_path)
            print(f"🧹 Cleaned up: {save_path}")


test_ivf_index_persistence()