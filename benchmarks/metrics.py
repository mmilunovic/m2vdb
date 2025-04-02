# benchmarks/metrics.py

def recall_at_k(results, ground_truth, k=10):
    total = len(results)
    correct = 0
    for r, gt in zip(results, ground_truth):
        correct += len(set(r) & set(gt[:k]))
    return correct / (total * k)
