"""Evaluation metrics for career recommendation system."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


def precision_at_k(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    k: int = 3
) -> float:
    """
    Compute Precision@K.
    
    Measures the proportion of relevant items in top-K recommendations.
    
    Args:
        y_true: Ground truth labels (1 = relevant, 0 = not)
        y_pred: Predicted ranks (sorted by score, higher = better)
        k: Number of top recommendations to consider
    
    Returns:
        Precision@K score
    """
    if k <= 0 or len(y_pred) == 0:
        return 0.0
    
    # Take top K predictions
    top_k_indices = y_pred[:k]
    
    # Count relevant items in top K
    relevant_in_top_k = sum(1 for idx in top_k_indices if y_true[idx] == 1)
    
    return relevant_in_top_k / k


def recall_at_k(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    k: int = 5
) -> float:
    """
    Compute Recall@K.
    
    Measures the proportion of relevant items retrieved in top-K.
    
    Args:
        y_true: Ground truth labels (1 = relevant, 0 = not)
        y_pred: Predicted ranks
        k: Number of top recommendations
    
    Returns:
        Recall@K score
    """
    total_relevant = sum(y_true)
    
    if total_relevant == 0 or k <= 0:
        return 0.0
    
    top_k_indices = y_pred[:k]
    relevant_in_top_k = sum(1 for idx in top_k_indices if y_true[idx] == 1)
    
    return relevant_in_top_k / total_relevant


def ndcg_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    Measures ranking quality with position-based discounting.
    
    Args:
        y_true: Ground truth relevance scores (e.g., 0 or 1)
        y_scores: Predicted relevance scores
        k: Number of top recommendations
    
    Returns:
        nDCG@K score
    """
    if len(y_true) == 0 or len(y_scores) == 0:
        return 0.0
    
    # Reshape for sklearn
    y_true_reshaped = np.asarray(y_true).reshape(1, -1)
    y_scores_reshaped = np.asarray(y_scores).reshape(1, -1)
    
    try:
        score = ndcg_score(y_true_reshaped, y_scores_reshaped, k=k)
        return float(score)
    except ValueError as e:
        LOGGER.warning(f"nDCG computation failed: {e}")
        return 0.0


def diversity_score(
    recommended_embeddings: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compute diversity of recommendations.
    
    Higher diversity = lower average pairwise similarity.
    
    Args:
        recommended_embeddings: Embeddings of recommended items (n_items, embedding_dim)
        metric: Distance metric ("cosine" or "euclidean")
    
    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    if len(recommended_embeddings) < 2:
        return 1.0  # Single item is maximally diverse
    
    if metric == "cosine":
        # Cosine similarity matrix
        similarity_matrix = cosine_similarity(recommended_embeddings)
        
        # Extract upper triangle (exclude diagonal)
        n = similarity_matrix.shape[0]
        pairwise_similarities = [
            similarity_matrix[i, j]
            for i in range(n)
            for j in range(i + 1, n)
        ]
        
        # Average similarity
        avg_similarity = np.mean(pairwise_similarities)
        
        # Diversity = 1 - similarity
        diversity = 1.0 - avg_similarity
        
    elif metric == "euclidean":
        # Pairwise Euclidean distances
        from sklearn.metrics.pairwise import euclidean_distances
        
        distance_matrix = euclidean_distances(recommended_embeddings)
        
        # Average distance
        n = distance_matrix.shape[0]
        pairwise_distances = [
            distance_matrix[i, j]
            for i in range(n)
            for j in range(i + 1, n)
        ]
        
        avg_distance = np.mean(pairwise_distances)
        
        # Normalize to 0-1 (assuming embeddings are L2-normalized)
        max_distance = 2.0  # Maximum L2 distance between unit vectors
        diversity = min(avg_distance / max_distance, 1.0)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(diversity)


def mean_reciprocal_rank(
    y_true: Sequence[int],
    y_pred: Sequence[int]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Measures rank of first relevant item.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted ranks
    
    Returns:
        MRR score
    """
    for rank, idx in enumerate(y_pred, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    
    return 0.0


def hit_rate_at_k(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    k: int = 5
) -> float:
    """
    Compute Hit Rate@K.
    
    Binary: 1 if at least one relevant item in top-K, 0 otherwise.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted ranks
        k: Number of top recommendations
    
    Returns:
        Hit rate (0 or 1)
    """
    top_k_indices = y_pred[:k]
    
    for idx in top_k_indices:
        if y_true[idx] == 1:
            return 1.0
    
    return 0.0


def evaluate_recommendations(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    recommended_embeddings: np.ndarray | None = None,
    k_values: list[int] | None = None
) -> dict[str, float]:
    """
    Comprehensive evaluation of recommendation quality.
    
    Args:
        y_true: Ground truth labels (n_items,)
        y_scores: Predicted scores (n_items,)
        recommended_embeddings: Embeddings for diversity (n_items, dim)
        k_values: K values for metrics (default: [3, 5, 10])
    
    Returns:
        Dictionary of metric name → value
    """
    if k_values is None:
        k_values = [3, 5, 10]
    
    # Sort by predicted scores
    ranked_indices = np.argsort(y_scores)[::-1]  # Descending
    
    metrics = {}
    
    for k in k_values:
        # Precision@K
        metrics[f"precision@{k}"] = precision_at_k(y_true, ranked_indices, k)
        
        # Recall@K
        metrics[f"recall@{k}"] = recall_at_k(y_true, ranked_indices, k)
        
        # nDCG@K
        metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_scores, k)
        
        # Hit Rate@K
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(y_true, ranked_indices, k)
    
    # MRR (not K-dependent)
    metrics["mrr"] = mean_reciprocal_rank(y_true, ranked_indices)
    
    # Diversity (if embeddings provided)
    if recommended_embeddings is not None:
        for k in k_values:
            top_k_embeddings = recommended_embeddings[ranked_indices[:k]]
            metrics[f"diversity@{k}"] = diversity_score(top_k_embeddings)
    
    return metrics


def evaluate_batch(
    y_true_batch: list[np.ndarray],
    y_scores_batch: list[np.ndarray],
    embeddings_batch: list[np.ndarray] | None = None,
    k_values: list[int] | None = None
) -> dict[str, float]:
    """
    Evaluate metrics across multiple users.
    
    Args:
        y_true_batch: List of ground truth arrays (one per user)
        y_scores_batch: List of predicted score arrays
        embeddings_batch: List of embedding arrays
        k_values: K values for metrics
    
    Returns:
        Dictionary of average metric values
    """
    all_metrics = []
    
    for i, (y_true, y_scores) in enumerate(zip(y_true_batch, y_scores_batch)):
        embeddings = embeddings_batch[i] if embeddings_batch else None
        
        user_metrics = evaluate_recommendations(y_true, y_scores, embeddings, k_values)
        all_metrics.append(user_metrics)
    
    # Average across users
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example evaluation
    np.random.seed(42)
    
    n_items = 20
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0] + [0] * 10)
    y_scores = np.random.rand(n_items)
    
    # Make some ground truth items score higher
    for idx in range(n_items):
        if y_true[idx] == 1:
            y_scores[idx] += 0.5
    
    # Generate random embeddings
    embeddings = np.random.randn(n_items, 128)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
    
    metrics = evaluate_recommendations(y_true, y_scores, embeddings)
    
    print("\n=== Recommendation Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
