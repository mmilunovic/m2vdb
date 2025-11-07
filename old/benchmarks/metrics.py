def recall_at_k(results, ground_truth, k=10):
    """
    Recall@k: What fraction of top-k ground truth items were found in top-k results?
    Assumes ground_truth[i] contains *at least* k true nearest neighbors for each query.
    """
    assert len(results) == len(ground_truth), "Mismatch in number of queries"

    correct = 0
    for res, gt in zip(results, ground_truth):
        correct += len(set(res[:k]) & set(gt[:k]))

    return correct / (len(results) * k)
