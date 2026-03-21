"""
Recall regression tests.

Uses deterministic synthetic data to verify search quality.
No external datasets needed.
"""

import numpy as np
import pytest

from m2vdb.indexes.brute_force import BruteForceIndex
from m2vdb.indexes.pq import PQIndex
from m2vdb.indexes.ivf import IVFIndex


def compute_recall(results: list[tuple[str, float]], ground_truth_ids: list[str]) -> float:
    """Compute recall@k: fraction of ground truth IDs found in results."""
    result_ids = {id for id, _ in results}
    hits = len(result_ids & set(ground_truth_ids))
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, ids: list[str], k: int, metric: str) -> list[str]:
    """Compute exact nearest neighbors using brute force."""
    bf = BruteForceIndex(metric=metric)
    bf.build(vectors, ids)
    results = bf.search(query, k)
    return [id for id, _ in results]


@pytest.fixture
def test_data():
    """Deterministic test dataset: 500 vectors, 128D."""
    rng = np.random.default_rng(123)
    vectors = rng.random((500, 128)).astype(np.float32)
    ids = [f"v{i}" for i in range(500)]
    queries = rng.random((10, 128)).astype(np.float32)
    return vectors, ids, queries


class TestBruteForceRecall:
    def test_recall_is_perfect(self, test_data):
        """Brute force is exact search -- recall must be 1.0."""
        vectors, ids, queries = test_data
        for metric in ["cosine", "euclidean"]:
            idx = BruteForceIndex(metric=metric)
            idx.build(vectors, ids)

            for query in queries:
                gt = get_ground_truth(vectors, query, ids, k=10, metric=metric)
                results = idx.search(query, k=10)
                recall = compute_recall(results, gt)
                assert recall == 1.0, f"BruteForce recall should be 1.0, got {recall} for {metric}"


class TestPQRecall:
    def test_recall_above_threshold_euclidean(self, test_data):
        """PQ recall should be above a reasonable threshold."""
        vectors, ids, queries = test_data
        idx = PQIndex(n_subvectors=8, n_clusters=32, metric="euclidean")
        idx.build(vectors, ids)

        recalls = []
        for query in queries:
            gt = get_ground_truth(vectors, query, ids, k=10, metric="euclidean")
            results = idx.search(query, k=10)
            recalls.append(compute_recall(results, gt))

        avg_recall = np.mean(recalls)
        # Random 128D data is hard for PQ -- threshold is conservative
        assert avg_recall > 0.1, f"PQ euclidean avg recall@10 = {avg_recall:.3f}, expected > 0.1"

    def test_recall_above_threshold_cosine(self, test_data):
        """PQ recall with cosine metric."""
        vectors, ids, queries = test_data
        idx = PQIndex(n_subvectors=8, n_clusters=32, metric="cosine")
        idx.build(vectors, ids)

        recalls = []
        for query in queries:
            gt = get_ground_truth(vectors, query, ids, k=10, metric="cosine")
            results = idx.search(query, k=10)
            recalls.append(compute_recall(results, gt))

        avg_recall = np.mean(recalls)
        assert avg_recall > 0.1, f"PQ cosine avg recall@10 = {avg_recall:.3f}, expected > 0.1"


class TestIVFRecall:
    def test_recall_above_threshold(self, test_data):
        """IVF with reasonable nprobe should have good recall."""
        vectors, ids, queries = test_data
        n_clusters = int(np.sqrt(500))
        nprobe = max(1, int(np.sqrt(n_clusters)))

        idx = IVFIndex(metric="euclidean", n_clusters=n_clusters, nprobe=nprobe)
        idx.build(vectors, ids)

        recalls = []
        for query in queries:
            gt = get_ground_truth(vectors, query, ids, k=10, metric="euclidean")
            results = idx.search(query, k=10)
            recalls.append(compute_recall(results, gt))

        avg_recall = np.mean(recalls)
        assert avg_recall > 0.3, f"IVF avg recall@10 = {avg_recall:.3f}, expected > 0.3"

    def test_recall_increases_with_nprobe(self, test_data):
        """Higher nprobe should give equal or better recall."""
        vectors, ids, queries = test_data

        recalls_by_nprobe = {}
        for nprobe in [1, 5, 22]:  # 22 = all clusters for sqrt(500) ~ 22
            idx = IVFIndex(metric="euclidean", n_clusters=22, nprobe=nprobe)
            idx.build(vectors.copy(), ids.copy())

            recalls = []
            for query in queries:
                gt = get_ground_truth(vectors, query, ids, k=10, metric="euclidean")
                results = idx.search(query, k=10)
                recalls.append(compute_recall(results, gt))

            recalls_by_nprobe[nprobe] = np.mean(recalls)

        # Recall should be monotonically non-decreasing with nprobe
        assert recalls_by_nprobe[5] >= recalls_by_nprobe[1] - 0.05  # small tolerance
        assert recalls_by_nprobe[22] >= recalls_by_nprobe[5] - 0.05

    def test_full_nprobe_is_exact(self, test_data):
        """With nprobe == n_clusters, IVF should be equivalent to brute force."""
        vectors, ids, queries = test_data
        n_clusters = 10

        idx = IVFIndex(metric="euclidean", n_clusters=n_clusters, nprobe=n_clusters)
        idx.build(vectors, ids)

        for query in queries:
            gt = get_ground_truth(vectors, query, ids, k=10, metric="euclidean")
            results = idx.search(query, k=10)
            recall = compute_recall(results, gt)
            assert recall == 1.0, f"IVF with full nprobe should be exact, got recall={recall}"


class TestSearchInvariants:
    """Property-based tests that should hold for ALL indexes."""

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_distances_non_negative(self, test_data, metric):
        vectors, ids, queries = test_data

        indexes = [
            BruteForceIndex(metric=metric),
            PQIndex(n_subvectors=8, n_clusters=32, metric=metric),
            IVFIndex(metric=metric, n_clusters=10, nprobe=5),
        ]

        for idx in indexes:
            idx.build(vectors.copy(), ids.copy())
            for query in queries[:3]:
                results = idx.search(query, k=5)
                for _, dist in results:
                    assert dist >= -1e-5, f"{type(idx).__name__} returned negative distance: {dist}"

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_results_sorted_ascending(self, test_data, metric):
        vectors, ids, queries = test_data

        indexes = [
            BruteForceIndex(metric=metric),
            PQIndex(n_subvectors=8, n_clusters=32, metric=metric),
            IVFIndex(metric=metric, n_clusters=10, nprobe=5),
        ]

        for idx in indexes:
            idx.build(vectors.copy(), ids.copy())
            for query in queries[:3]:
                results = idx.search(query, k=10)
                distances = [d for _, d in results]
                assert distances == sorted(distances), \
                    f"{type(idx).__name__} results not sorted: {distances}"
