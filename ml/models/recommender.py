"""Hybrid career recommender with retrieval + neural re-ranking."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ml.evaluation.metrics import (
    diversity_score,
    ndcg_at_k,
    precision_at_k
)
from ml.pipelines.feature_engineering import UserProfile, build_user_vector
from ml.models.explainer import RecommendationExplainer

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RecommenderConfig:
    """Configuration for recommender artifact locations."""

    features_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "features"
    training_data_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "training_data"
    graph_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "graph"
    model_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "model"
    user_features_file: str = "user_features.parquet"
    career_features_file: str = "career_item_features.parquet"
    graph_embeddings_file: str = "node_embeddings.pkl"
    model_file: str = "reranker.pkl"
    scaler_file: str = "scaler.pkl"
    metadata_file: str = "metadata.json"

    def ensure_model_dir(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def user_features_path(self) -> Path:
        return (self.features_dir / self.user_features_file).resolve()

    @property
    def career_features_path(self) -> Path:
        return (self.features_dir / self.career_features_file).resolve()

    @property
    def graph_embeddings_path(self) -> Path:
        return (self.graph_dir / self.graph_embeddings_file).resolve()

    @property
    def model_path(self) -> Path:
        return (self.model_dir / self.model_file).resolve()

    @property
    def scaler_path(self) -> Path:
        return (self.model_dir / self.scaler_file).resolve()

    @property
    def metadata_path(self) -> Path:
        return (self.model_dir / self.metadata_file).resolve()


@dataclass
class TrainingArtifacts:
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]


def _load_npz_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["user_vectors"], data["career_vectors"], data["labels"]


def _build_pair_features(user_vectors: np.ndarray, career_vectors: np.ndarray) -> np.ndarray:
    user_vectors = np.asarray(user_vectors, dtype=np.float32)
    career_vectors = np.asarray(career_vectors, dtype=np.float32)
    product = user_vectors * career_vectors
    l1_distance = np.abs(user_vectors - career_vectors)
    return np.concatenate([user_vectors, career_vectors, product, l1_distance], axis=1)


def _group_indices_by_user(user_vectors: np.ndarray) -> dict[bytes, list[int]]:
    groups: dict[bytes, list[int]] = {}
    for idx, vec in enumerate(user_vectors):
        key = np.round(vec, 6).tobytes()
        groups.setdefault(key, []).append(idx)
    return groups


def _evaluate_ranking_metrics(
    user_vectors: np.ndarray,
    career_vectors: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    k_prec: int = 3,
    k_ndcg: int = 5,
    k_div: int = 5
) -> dict[str, float]:
    groups = _group_indices_by_user(user_vectors)
    if not groups:
        return {
            "precision@3": 0.0,
            "ndcg@5": 0.0,
            "diversity@5": 0.0
        }

    precision_scores: list[float] = []
    ndcg_scores: list[float] = []
    diversity_scores: list[float] = []

    for indices in groups.values():
        y_true = labels[indices]
        y_score = scores[indices]
        item_vectors = career_vectors[indices]

        ranked_idx = np.argsort(y_score)[::-1]
        precision_scores.append(precision_at_k(y_true, ranked_idx, k=k_prec))
        ndcg_scores.append(ndcg_at_k(y_true, y_score, k=k_ndcg))

        top_k = ranked_idx[: min(k_div, len(ranked_idx))]
        if len(top_k) > 1:
            emb = np.asarray(item_vectors[top_k], dtype=np.float32)
            diversity_scores.append(diversity_score(emb))
        else:
            diversity_scores.append(1.0)

    return {
        "precision@3": float(np.mean(precision_scores)),
        "ndcg@5": float(np.mean(ndcg_scores)),
        "diversity@5": float(np.mean(diversity_scores))
    }


def _load_graph_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        LOGGER.warning("Graph embeddings not found at %s; falling back to empty dict", path)
        return {}
    with path.open("rb") as handle:
        embeddings = joblib.load(handle)
    return {str(key): np.asarray(val, dtype=np.float32) for key, val in embeddings.items()}


def _normalize_past_roles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value)
    for delimiter in [";", ",", "|", "/"]:
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text.strip()] if text.strip() else []


def _build_user_vector_from_payload(payload: dict[str, Any], job_embeddings: dict[str, np.ndarray]) -> np.ndarray:
    profile = UserProfile(
        age=int(payload.get("age", 25)),
        education_level=str(payload.get("education_level", "Bachelor's")),
        grades_gpa=float(payload.get("grades_gpa")) if payload.get("grades_gpa") is not None else None,
        interests=payload.get("interests"),
        skills=payload.get("skills"),
        past_roles=_normalize_past_roles(payload.get("past_roles")),
        location=payload.get("location"),
        years_experience=int(payload.get("years_experience", 0))
    )
    vector = build_user_vector(profile, job_embeddings)
    return vector.astype(np.float32)


@dataclass
class TrainingResult:
    model: MLPClassifier
    scaler: StandardScaler
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class CareerRecommender:
    """Retriever + re-ranker hybrid recommender."""

    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self.config = config or RecommenderConfig()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> TrainingResult:
        cfg = self.config
        cfg.ensure_model_dir()

        artifacts = self._load_training_artifacts(cfg)

        train_features = _build_pair_features(*artifacts.train[:2])
        val_features = _build_pair_features(*artifacts.val[:2])
        test_features = _build_pair_features(*artifacts.test[:2])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_features)
        val_X = scaler.transform(val_features)
        test_X = scaler.transform(test_features)

        train_y = artifacts.train[2]
        val_y = artifacts.val[2]
        test_y = artifacts.test[2]

        batch_size = int(min(128, max(1, train_X.shape[0])))

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=batch_size,
            max_iter=200,
            early_stopping=False,
            random_state=42,
        )

        LOGGER.info("Training re-ranker (%d samples)...", train_X.shape[0])
        model.fit(train_X, train_y)

        # Probabilities for evaluation
        val_scores = model.predict_proba(val_X)[:, 1]
        test_scores = model.predict_proba(test_X)[:, 1]

        val_metrics = _evaluate_ranking_metrics(
            artifacts.val[0], artifacts.val[1], val_y, val_scores
        )
        test_metrics = _evaluate_ranking_metrics(
            artifacts.test[0], artifacts.test[1], test_y, test_scores
        )

        overall_metrics = {
            "val_precision@3": val_metrics["precision@3"],
            "val_ndcg@5": val_metrics["ndcg@5"],
            "val_diversity@5": val_metrics["diversity@5"],
            "test_precision@3": test_metrics["precision@3"],
            "test_ndcg@5": test_metrics["ndcg@5"],
            "test_diversity@5": test_metrics["diversity@5"],
            "val_log_loss": self._safe_log_loss(val_y, val_scores),
            "test_log_loss": self._safe_log_loss(test_y, test_scores)
        }

        LOGGER.info(
            "Validation metrics: P@3=%.3f nDCG@5=%.3f Diversity@5=%.3f",
            overall_metrics["val_precision@3"],
            overall_metrics["val_ndcg@5"],
            overall_metrics["val_diversity@5"]
        )
        LOGGER.info(
            "Test metrics: P@3=%.3f nDCG@5=%.3f Diversity@5=%.3f",
            overall_metrics["test_precision@3"],
            overall_metrics["test_ndcg@5"],
            overall_metrics["test_diversity@5"]
        )

        # Persist artifacts
        joblib.dump(model, cfg.model_path)
        joblib.dump(scaler, cfg.scaler_path)

        metadata = {
            "trained_at": datetime.now(UTC).isoformat(),
            "model": "MLPClassifier",
            "train_samples": int(train_X.shape[0]),
            "feature_dim": int(train_X.shape[1]),
            "metrics": overall_metrics
        }
        cfg.metadata_path.write_text(json.dumps(metadata, indent=2))

        return TrainingResult(model=model, scaler=scaler, metrics=overall_metrics, metadata=metadata)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self, 
        user_json_path: str | Path, 
        top_k: int = 5, 
        include_explanations: bool = True
    ) -> list[dict[str, Any]]:
        """Generate recommendations with optional explanations.
        
        Args:
            user_json_path: Path to JSON file with user profile
            top_k: Number of recommendations to return
            include_explanations: Whether to generate explanations (default: True)
            
        Returns:
            List of recommendation dictionaries with scores and optional explanations
        """
        cfg = self.config
        model = self._load_model(cfg.model_path)
        scaler = self._load_scaler(cfg.scaler_path)

        user_payload = json.loads(Path(user_json_path).read_text(encoding="utf-8"))
        job_embeddings = _load_graph_embeddings(cfg.graph_embeddings_path)
        user_vector = _build_user_vector_from_payload(user_payload, job_embeddings)

        careers_df = self._load_career_features(cfg.career_features_path)
        if careers_df.empty:
            raise RuntimeError("No career features found; run feature pipeline first.")

        career_vectors = np.vstack(careers_df["feature_vector"].to_numpy())
        retrieve_scores = self._cosine_similarity(user_vector, career_vectors)

        top_n = min(50, len(careers_df))
        top_indices = np.argsort(retrieve_scores)[::-1][:top_n]

        selected_vectors = career_vectors[top_indices]
        pair_features = _build_pair_features(
            np.repeat(user_vector[None, :], top_n, axis=0),
            selected_vectors
        )
        pair_features_scaled = scaler.transform(pair_features)
        rerank_scores = model.predict_proba(pair_features_scaled)[:, 1]

        ranked_order = np.argsort(rerank_scores)[::-1][:top_k]

        # Initialize explainer if needed
        explainer = RecommendationExplainer() if include_explanations else None

        results = []
        for rank in ranked_order:
            idx = top_indices[rank]
            row = careers_df.iloc[idx]
            
            career_id = row.get("career_id") or row.get("id") or str(idx)
            career_title = row.get("recommended_career") or row.get("career_title")
            
            recommendation = {
                "career_id": career_id,
                "career_title": career_title,
                "retrieval_score": float(retrieve_scores[idx]),
                "rerank_score": float(rerank_scores[rank]),
            }
            
            # Generate explanations if requested
            if explainer:
                # Build model weights from scores
                model_weights = {
                    "skill_similarity": float(retrieve_scores[idx]),
                    "interest_alignment": 0.70,  # Placeholder - would come from feature engineering
                    "education_match": 0.85,
                    "experience_fit": 0.75,
                    "career_graph_proximity": 0.65,
                    "career_centrality": float(row.get("centrality", 0.5)),
                    "rerank_score": float(rerank_scores[rank]),
                }
                
                # Extract career details from row
                career_details = {
                    "career_id": career_id,
                    "career_title": career_title,
                    "required_skills": self._parse_skills_column(row.get("required_skills", "")),
                    "required_education": row.get("required_education", "bachelor's"),
                    "required_experience_years": int(row.get("required_experience_years", 3)),
                    "centrality": float(row.get("centrality", 0.5)),
                }
                
                explanation_result = explainer.explain(user_payload, career_details, model_weights)
                
                recommendation["explanations"] = {
                    "why_recommended": explanation_result.why_recommended,
                    "learning_plan": explanation_result.learning_plan,
                }
                recommendation["audit_log"] = explanation_result.audit_log
            else:
                recommendation["reasons"] = ["skills_alignment", "interest_overlap"]  # Legacy format
            
            results.append(recommendation)
            
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_skills_column(skills_value: Any) -> list[str]:
        """Parse skills from various formats (string, list, etc.)."""
        if isinstance(skills_value, list):
            return [str(s).strip() for s in skills_value if s]
        if isinstance(skills_value, str):
            # Handle comma or semicolon separated
            skills_value = skills_value.replace(";", ",")
            return [s.strip() for s in skills_value.split(",") if s.strip()]
        return []
    
    @staticmethod
    def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return float(log_loss(y_true, y_pred, labels=[0, 1]))
        except ValueError:
            return float("nan")

    def _load_training_artifacts(self, cfg: RecommenderConfig) -> TrainingArtifacts:
        train_path = cfg.training_data_dir / "train.npz"
        val_path = cfg.training_data_dir / "val.npz"
        test_path = cfg.training_data_dir / "test.npz"
        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Training data splits not found. Run ml/pipelines/training_data_prep.py first."
            )
        train = _load_npz_split(train_path)
        val = _load_npz_split(val_path)
        test = _load_npz_split(test_path)
        return TrainingArtifacts(train=train, val=val, test=test)

    @staticmethod
    def _load_model(path: Path) -> MLPClassifier:
        if not path.exists():
            raise FileNotFoundError("Trained model not found; run training first.")
        model = joblib.load(path)
        if not isinstance(model, MLPClassifier):
            raise TypeError("Stored model is not an MLPClassifier instance")
        return model

    @staticmethod
    def _load_scaler(path: Path) -> StandardScaler:
        if not path.exists():
            raise FileNotFoundError("Scaler artifact not found; run training first.")
        scaler = joblib.load(path)
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Stored scaler is not a StandardScaler instance")
        return scaler

    @staticmethod
    def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32)
        matrix = matrix.astype(np.float32)
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = vector_norm * matrix_norms + 1e-12
        sims = matrix @ vector / denom
        return sims

    @staticmethod
    def _load_career_features(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Career feature table not found at {path}")
        df = pd.read_parquet(path)
        if "feature_vector" not in df.columns:
            raise KeyError("Parquet file missing 'feature_vector' column")
        df = df.copy()
        df["feature_vector"] = df["feature_vector"].apply(lambda vals: np.asarray(vals, dtype=np.float32))
        return df


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Career recommender CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train re-ranker model")
    train_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config overriding artifact paths"
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference for a user payload")
    infer_parser.add_argument("--user_json", type=Path, required=True, help="Path to user JSON payload")
    infer_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config overriding artifact paths"
    )
    infer_parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Disable explanation generation for faster inference"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    config = _load_config_override(args.config) if getattr(args, "config", None) else RecommenderConfig()
    recommender = CareerRecommender(config=config)

    if args.command == "train":
        result = recommender.train()
        print(json.dumps(result.metrics, indent=2))
        return 0
    if args.command == "infer":
        include_explanations = not getattr(args, "no_explanations", False)
        recommendations = recommender.infer(args.user_json, include_explanations=include_explanations)
        print(json.dumps(recommendations, indent=2, default=str))
        return 0
    return 1


def _load_config_override(path: Path) -> RecommenderConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = RecommenderConfig()
    for field_name in data:
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, Path(data[field_name]) if "_dir" in field_name or field_name.endswith("_path") else data[field_name])
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
