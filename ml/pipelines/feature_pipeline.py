"""End-to-end feature pipeline for user and career representations."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data_catalog import load_all_datasets
from .embedder import configure_embedder
from .feature_engineering import (
    EDUCATION_HIERARCHY,
    UserProfile,
    build_career_vector,
    build_user_vector
)
from .graph_builder import (
    build_career_graph,
    compute_career_centrality,
    train_node2vec
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


class FeaturePipeline:
    """Sklearn-style pipeline that materialises ML-ready feature tables."""

    def __init__(
        self,
        config_path: str | Path | None = DEFAULT_CONFIG_PATH,
        *,
        config: dict[str, Any] | None = None,
        save_outputs: bool = True
    ) -> None:
        self.save_outputs = save_outputs
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config(config)

        # Configure sentence-transformer model
        model_cfg = self.config.get("models", {}).get("text_embedding", {})
        configure_embedder(
            model_cfg.get("name"),
            embedding_dim=model_cfg.get("dimension"),
            cache_dir=model_cfg.get("cache_dir")
        )

        self.graph_ = None
        self.graph_embeddings_: dict[str, np.ndarray] | None = None
        self.centrality_: dict[str, float] | None = None
        self.user_features_: pd.DataFrame | None = None
        self.career_features_: pd.DataFrame | None = None
        self.graph_embeddings_df_: pd.DataFrame | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, datasets: dict[str, pd.DataFrame] | None = None) -> "FeaturePipeline":
        """Fit pipeline components (graph + embeddings)."""
        datasets = datasets or load_all_datasets()

        graph_cfg = self.config.get("graph", {})
        graph_df = self._prepare_karrierewege_frame(datasets, graph_cfg)

        if graph_df.empty:
            LOGGER.warning("KARRIEREWEGE dataset is empty; graph embeddings will be zero vectors")
            self.graph_ = None
            self.graph_embeddings_ = {}
            self.centrality_ = {}
            self.graph_embeddings_df_ = pd.DataFrame(columns=["career_title", "embedding"])
        else:
            self.graph_ = build_career_graph(
                graph_df,
                person_col=graph_cfg.get("person_col", "person_id"),
                role_col=graph_cfg.get("role_col", "job_title"),
                date_col=graph_cfg.get("date_col", "start_date"),
                min_transitions=graph_cfg.get("min_transitions", 2)
            )

            self.centrality_ = compute_career_centrality(self.graph_)

            self.graph_embeddings_ = train_node2vec(
                self.graph_,
                dimensions=graph_cfg.get("embedding_dim", 128),
                walk_length=graph_cfg.get("walk_length", 10),
                num_walks=graph_cfg.get("num_walks", 50),
                p=graph_cfg.get("p", 1.0),
                q=graph_cfg.get("q", 0.5),
                workers=graph_cfg.get("workers", 4),
                window=graph_cfg.get("window", 5),
                min_count=graph_cfg.get("min_count", 1),
                epochs=graph_cfg.get("epochs", 5)
            )

            self.graph_embeddings_df_ = self._graph_embeddings_to_df(self.graph_embeddings_)

            if self.save_outputs:
                self._write_parquet("graph_embeddings", self.graph_embeddings_df_)

        self.is_fitted_ = True
        return self

    def transform(
        self,
        datasets: dict[str, pd.DataFrame] | None = None,
        *,
        save_outputs: bool | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Transform datasets into feature tables using fitted components."""
        if not self.is_fitted_:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

        datasets = datasets or load_all_datasets()
        job_embeddings = self.graph_embeddings_ or {}
        centrality = self.centrality_ or {}

        user_features = self._build_user_features(datasets, job_embeddings)
        career_features = self._build_career_features(datasets, job_embeddings, centrality)
        if self.graph_embeddings_df_ is not None:
            graph_embeddings = self.graph_embeddings_df_
        else:
            graph_embeddings = pd.DataFrame(columns=["career_title", "embedding"])

        self.user_features_ = user_features
        self.career_features_ = career_features

        should_save = self.save_outputs if save_outputs is None else save_outputs
        if should_save:
            self._write_parquet("user_features", user_features)
            self._write_parquet("career_features", career_features)

        return user_features, career_features, graph_embeddings

    def fit_transform(
        self,
        datasets: dict[str, pd.DataFrame] | None = None,
        *,
        save_outputs: bool | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convenience wrapper for fit() and transform()."""
        return self.fit(datasets).transform(datasets, save_outputs=save_outputs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_config(self, config_override: dict[str, Any] | None) -> dict[str, Any]:
        if config_override is not None:
            return deepcopy(config_override)

        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config

    def _prepare_karrierewege_frame(
        self,
        datasets: dict[str, pd.DataFrame],
        graph_cfg: dict[str, Any]
    ) -> pd.DataFrame:
        prefix = graph_cfg.get("dataset_prefix", "karrierewege")
        frames = [df for name, df in datasets.items() if name.startswith(prefix)]

        if not frames:
            LOGGER.warning("No karrierewege datasets found with prefix '%s'", prefix)
            return pd.DataFrame(columns=[
                graph_cfg.get("person_col", "person_id"),
                graph_cfg.get("role_col", "job_title"),
                graph_cfg.get("date_col", "start_date")
            ])

        combined = pd.concat(frames, ignore_index=True)
        # Ensure datetime ordering
        date_col = graph_cfg.get("date_col", "start_date")
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce")
        return combined

    def _graph_embeddings_to_df(self, embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
        records = [
            {"career_title": career, "embedding": emb.astype(np.float32).tolist()}
            for career, emb in embeddings.items()
        ]
        df = pd.DataFrame(records, columns=["career_title", "embedding"])
        return df

    def _build_user_features(
        self,
        datasets: dict[str, pd.DataFrame],
        job_embeddings: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        dataset_name = self.config.get("datasets", {}).get("user_profiles")
        if dataset_name is None or dataset_name not in datasets:
            raise KeyError("User dataset not found in catalog; check config.datasets.user_profiles")

        df = datasets[dataset_name].copy()
        columns = self.config.get("columns", {}).get("user", {})
        id_col = columns.get("id", "user_id")

        if id_col not in df.columns:
            df[id_col] = df.index.astype(str)

        records = []
        for user_id, group in df.groupby(id_col):
            row = group.iloc[-1]
            profile = self._row_to_user_profile(row, columns)
            vector = build_user_vector(profile, job_embeddings)
            records.append({
                id_col: str(user_id),
                "feature_vector": vector.astype(np.float32).tolist()
            })

        return pd.DataFrame(records, columns=[id_col, "feature_vector"])

    def _row_to_user_profile(self, row: pd.Series, columns: dict[str, str]) -> UserProfile:
        def _get(col_name: str, default: Any = None) -> Any:
            if col_name is None:
                return default
            return row.get(col_name, default)

        age = _get(columns.get("age"), 25)
        try:
            age = int(age)
        except (TypeError, ValueError):
            age = 25

        education = str(_get(columns.get("education_level"), "Bachelor's") or "Bachelor's")

        gpa = _get(columns.get("gpa"))
        gpa = float(gpa) if gpa not in (None, "", np.nan) else None

        interests = _get(columns.get("interests"))
        skills = _get(columns.get("skills"))

        past_roles_raw = _get(columns.get("past_roles"))
        past_roles = self._normalize_past_roles(past_roles_raw)

        years_exp = _get(columns.get("years_experience"), 0)
        try:
            years_exp = int(years_exp)
        except (TypeError, ValueError):
            years_exp = 0

        return UserProfile(
            age=age,
            education_level=education,
            grades_gpa=gpa,
            interests=interests,
            skills=skills,
            past_roles=past_roles,
            years_experience=years_exp
        )

    def _normalize_past_roles(self, value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return [str(role).strip() for role in value if str(role).strip()]
        text = str(value)
        for delimiter in [";", ",", "|", "/"]:
            if delimiter in text:
                return [part.strip() for part in text.split(delimiter) if part.strip()]
        return [text.strip()] if text.strip() else []

    def _build_career_features(
        self,
        datasets: dict[str, pd.DataFrame],
        job_embeddings: dict[str, np.ndarray],
        centrality: dict[str, float]
    ) -> pd.DataFrame:
        dataset_name = self.config.get("datasets", {}).get("careers")
        if dataset_name is None or dataset_name not in datasets:
            raise KeyError("Career dataset not found in catalog; check config.datasets.careers")

        df = datasets[dataset_name].copy()
        columns = self.config.get("columns", {}).get("career", {})
        id_col = columns.get("id", "career_id")
        title_col = columns.get("title", "career_title")

        if title_col not in df.columns:
            raise KeyError(f"Career title column '{title_col}' not found in dataset '{dataset_name}'")

        if id_col not in df.columns:
            df[id_col] = df[title_col].ffill().bfill()

        records = []
        seen_titles: set[str] = set()
        for _, row in df.iterrows():
            title = str(row.get(title_col, "")).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            career_embedding = job_embeddings.get(title)

            required_skills = row.get(columns.get("skills", "required_skills"))
            education_req = self._education_requirement_to_int(row.get(columns.get("education_requirement")))
            typical_gpa = row.get(columns.get("typical_gpa"), 3.0)
            try:
                typical_gpa = float(typical_gpa)
            except (TypeError, ValueError):
                typical_gpa = 3.0

            centrality_score = float(centrality.get(title, 0.0))

            vector = build_career_vector(
                career_title=title,
                career_embedding=career_embedding if career_embedding is not None else np.zeros(
                    self.config.get("graph", {}).get("embedding_dim", 128),
                    dtype=np.float32
                ),
                required_skills=required_skills,
                education_requirement=education_req,
                typical_gpa=typical_gpa,
                centrality_score=centrality_score
            )

            records.append({
                id_col: row.get(id_col, title),
                title_col: title,
                "feature_vector": vector.astype(np.float32).tolist()
            })

        return pd.DataFrame(records, columns=[id_col, title_col, "feature_vector"])

    def _education_requirement_to_int(self, value: Any) -> int:
        if value is None or value == "":
            return 2  # Default "some college"
        if isinstance(value, (int, float)):
            return int(value)
        value_str = str(value).lower().strip()
        return EDUCATION_HIERARCHY.get(value_str, 2)

    def _write_parquet(self, output_key: str, df: pd.DataFrame) -> None:
        outputs_cfg = self.config.get("outputs", {})
        if output_key not in outputs_cfg:
            LOGGER.warning("Output path for key '%s' not found in config", output_key)
            return

        path = Path(outputs_cfg[output_key])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(path, index=False)
        LOGGER.info("Wrote %s to %s", output_key, path)


__all__ = ["FeaturePipeline"]
