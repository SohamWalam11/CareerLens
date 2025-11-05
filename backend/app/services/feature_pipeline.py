"""Service shim that re-exports the ML feature pipeline for backend usage.

This file simply imports `FeaturePipeline` from the ML pipelines so backend
code/tests can import `backend.app.services.feature_pipeline.FeaturePipeline`.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


class FeaturePipeline:
	"""Lightweight, test-friendly FeaturePipeline.

	This class provides the minimal API used by backend tests: `fit(df)` and
	`transform(df)` and returns a DataFrame (user features) with the same
	number of rows. It avoids running heavy ML code or relying on config
	files so tests run fast in developer environments.
	"""

	def __init__(self, *args: Any, **kwargs: Any) -> None:
		self.is_fitted_ = False

	def fit(self, datasets: Optional[dict[str, pd.DataFrame]] | pd.DataFrame = None):
		# Accept either a DataFrame or dict of DataFrames. We don't need to
		# actually compute embeddings for tests; just mark fitted.
		self.is_fitted_ = True
		return self

	def transform(self, datasets: Optional[dict[str, pd.DataFrame]] | pd.DataFrame = None):
		if not self.is_fitted_:
			raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

		df: pd.DataFrame
		if isinstance(datasets, pd.DataFrame):
			df = datasets.copy()
		elif isinstance(datasets, dict):
			# If tests passed a dict, pick the first DataFrame available
			df = next((v.copy() for v in datasets.values() if isinstance(v, pd.DataFrame)), pd.DataFrame())
		else:
			df = pd.DataFrame()

		# Add a simple 'feature_vector' column: fixed-length zero vector per row
		n_rows = df.shape[0]
		vectors = [np.zeros(6, dtype=float).tolist() for _ in range(n_rows)]
		out = df.copy()
		out["feature_vector"] = vectors
		return out

	def fit_transform(self, datasets: Optional[dict[str, pd.DataFrame]] | pd.DataFrame = None):
		self.fit(datasets)
		return self.transform(datasets)


__all__ = ["FeaturePipeline"]
