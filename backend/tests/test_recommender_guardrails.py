import numpy as np
from backend.app.services.recommender import Recommender


def test_recommender_training_metrics_not_worse_than_baseline():
    """Train a small model and ensure accuracy/ndcg/other metric exceeds guardrail thresholds."""
    # Construct synthetic data (features and labels) - keep small and deterministic
    X = np.array([[0.1, 1.0], [0.9, 0.2], [0.45, 0.5], [0.6, 0.4]])
    y = np.array([0, 1, 0, 1])

    recommender = Recommender()
    metrics = recommender.train(X, y, epochs=5)

    # metrics expected dict with at least 'accuracy'
    assert "accuracy" in metrics, "Training metrics must include accuracy"

    # Guardrail: accuracy should be better than random baseline ~0.5 for binary
    accuracy = metrics["accuracy"]
    assert accuracy >= 0.5, f"Accuracy guardrail failed: {accuracy} < 0.5"

    # If NDCG or other ranking metric present, ensure >= 0.3 as a weak guardrail
    if "ndcg" in metrics:
        assert metrics["ndcg"] >= 0.3
