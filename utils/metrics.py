"""
utils/metrics.py
================
Ranking metrics — because accuracy/F1 are wrong for recommendation.

NDCG@K: rewards putting the best matches highest.
Precision@K: of top K, how many were real DJ transitions.
Hit Rate@K: did ANY correct answer appear in top K.
"""

import numpy as np
from typing import List, Dict


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = relevances / np.log2(np.arange(2, len(relevances) + 2))
    return gains.sum()


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalised DCG@K. 1.0 = perfect ranking."""
    actual  = dcg_at_k(relevances, k)
    ideal   = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def precision_at_k(relevances: List[int], k: int) -> float:
    """Fraction of top-K that are relevant."""
    return sum(relevances[:k]) / k


def hit_rate_at_k(relevances: List[int], k: int) -> float:
    """1 if at least one relevant item in top-K, else 0."""
    return float(any(r > 0 for r in relevances[:k]))


def mean_reciprocal_rank(relevances: List[int]) -> float:
    """1/rank of first relevant item."""
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(
    query_ids: List[str],
    scores: np.ndarray,
    labels: np.ndarray,
    pair_index: List,
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Full ranking evaluation across all queries.

    For each query track, rank all candidates by score,
    then compute NDCG@K, Precision@K, Hit Rate@K.

    Args:
        query_ids:   unique anchor track IDs
        scores:      model probability scores (N,)
        labels:      ground truth 0/1 (N,)
        pair_index:  list of (track_a_id, track_b_id) matching scores/labels
        k_values:    list of K values to evaluate at

    Returns:
        dict of metric_name → mean value across all queries
    """
    k_values = k_values or [5, 10, 20]

    # Group by query track
    query_results = {}
    for i, (a, b) in enumerate(pair_index):
        if a not in query_results:
            query_results[a] = []
        query_results[a].append((scores[i], labels[i]))

    # Compute per-query metrics
    all_metrics = {f"ndcg@{k}": [] for k in k_values}
    all_metrics.update({f"precision@{k}": [] for k in k_values})
    all_metrics.update({f"hit_rate@{k}": [] for k in k_values})
    all_metrics["mrr"] = []

    for qid, items in query_results.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[0], reverse=True)  # sort by score desc
        relevances = [int(label) for _, label in items]

        for k in k_values:
            all_metrics[f"ndcg@{k}"].append(ndcg_at_k(relevances, k))
            all_metrics[f"precision@{k}"].append(precision_at_k(relevances, k))
            all_metrics[f"hit_rate@{k}"].append(hit_rate_at_k(relevances, k))
        all_metrics["mrr"].append(mean_reciprocal_rank(relevances))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}


def print_metrics_report(metrics: Dict[str, float], stage: str = ""):
    """Pretty-print the evaluation report."""
    header = f"── Evaluation Report {stage} "
    print(f"\n{header}{'─' * max(0, 60 - len(header))}")
    print(f"  {'Metric':<22} {'Score':>8}")
    print(f"  {'──────':<22} {'─────':>8}")

    priority = ["ndcg@10", "ndcg@5", "ndcg@20", "precision@10", "precision@5", "hit_rate@10", "mrr"]
    shown = set()
    for key in priority:
        if key in metrics:
            print(f"  {key:<22} {metrics[key]:>8.4f}")
            shown.add(key)
    for key, val in sorted(metrics.items()):
        if key not in shown:
            print(f"  {key:<22} {val:>8.4f}")
    print()
