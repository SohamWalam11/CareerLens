"""Compatibility shim providing a lightweight Recommender used by tests.

The repo contains a full ML recommender under `ml/models/recommender.py` but
some tests expect a simple `backend.app.services.recommender.Recommender` with
a `.train(X, y, epochs=...)` API that returns a metrics dict containing at
least an "accuracy" key. This small shim trains a fast LogisticRegression on
the provided arrays and returns accuracy. It's intentionally simple and
deterministic for test usage.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Recommender:
    """Tiny test-friendly recommender wrapper.

    Trains a LogisticRegression on the supplied arrays and returns a metrics
    dictionary containing at least `accuracy` so tests' guardrails can run.
    """

    def __init__(self) -> None:
        # Use a deterministic solver and fixed random state where applicable
        self._model = LogisticRegression(random_state=42, max_iter=200)

    def train(self, X, y, epochs: int = 5) -> dict[str, float]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        # If data is tiny or degenerate, fall back to a trivial accuracy estimate
        if X_arr.size == 0 or y_arr.size == 0:
            return {"accuracy": 0.0}

        try:
            # Fit model on provided data (tests use very small synthetic sets)
            self._model.fit(X_arr, y_arr)
            preds = self._model.predict(X_arr)
            acc = float(accuracy_score(y_arr, preds))
        except Exception:
            # Training may fail for degenerate inputs; return 0 accuracy instead
            acc = 0.0

        return {"accuracy": acc}
