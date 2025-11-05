"""Tests for the hybrid recommender model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.models.recommender import CareerRecommender, RecommenderConfig


@pytest.fixture()
def temp_recommender_paths(tmp_path: Path) -> RecommenderConfig:
    features_dir = tmp_path / "features"
    training_dir = tmp_path / "training"
    graph_dir = tmp_path / "graph"
    model_dir = tmp_path / "model"

    features_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    config = RecommenderConfig(
        features_dir=features_dir,
        training_data_dir=training_dir,
        graph_dir=graph_dir,
        model_dir=model_dir,
        career_features_file="career_features.parquet",
        graph_embeddings_file="node_embeddings.pkl",
    )
    return config


@pytest.fixture()
def seed_training_artifacts(temp_recommender_paths: RecommenderConfig) -> None:
    cfg = temp_recommender_paths

    def _write_split(name: str, user_vectors, career_vectors, labels) -> None:
        path = cfg.training_data_dir / f"{name}.npz"
        np.savez(
            path,
            user_vectors=np.asarray(user_vectors, dtype=np.float32),
            career_vectors=np.asarray(career_vectors, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.float32)
        )

    # Simple dataset with two distinct user profiles
    train_users = [
        [0.9, 0.1],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.2, 0.8],
    ]
    train_careers = [
        [0.85, 0.15],
        [0.1, 0.9],
        [0.25, 0.75],
        [0.8, 0.2],
    ]
    train_labels = [1, 0, 1, 0]

    val_users = [
        [0.9, 0.1],
        [0.2, 0.8],
    ]
    val_careers = [
        [0.8, 0.2],
        [0.3, 0.7],
    ]
    val_labels = [1, 1]

    test_users = [
        [0.9, 0.1],
        [0.2, 0.8],
    ]
    test_careers = [
        [0.2, 0.8],
        [0.7, 0.3],
    ]
    test_labels = [0, 1]

    _write_split("train", train_users, train_careers, train_labels)
    _write_split("val", val_users, val_careers, val_labels)
    _write_split("test", test_users, test_careers, test_labels)

    # Career feature table for inference
    career_df = pd.DataFrame(
        {
            "career_id": ["c1", "c2", "c3"],
            "recommended_career": ["Data Scientist", "Product Manager", "UX Designer"],
            "feature_vector": [
                [0.82, 0.18],
                [0.15, 0.85],
                [0.4, 0.6],
            ],
        }
    )
    career_df.to_parquet(cfg.career_features_path, index=False)

    # Node embeddings placeholder (not critical for test)
    job_embeddings = {
        "Data Scientist": np.array([0.5, 0.5], dtype=np.float32),
        "Product Manager": np.array([0.3, 0.7], dtype=np.float32),
    }
    import joblib

    joblib.dump(job_embeddings, cfg.graph_embeddings_path)


@pytest.fixture()
def patch_user_vector(monkeypatch):
    def fake_builder(payload, job_embeddings):  # noqa: ANN001, ARG001
        return np.array([0.88, 0.12], dtype=np.float32)

    monkeypatch.setattr(
        "ml.models.recommender._build_user_vector_from_payload",
        fake_builder
    )


def test_training_persists_model(seed_training_artifacts, temp_recommender_paths):
    recommender = CareerRecommender(config=temp_recommender_paths)
    result = recommender.train()

    assert temp_recommender_paths.model_path.exists()
    assert temp_recommender_paths.scaler_path.exists()
    assert temp_recommender_paths.metadata_path.exists()

    assert "val_precision@3" in result.metrics
    assert 0.0 <= result.metrics["val_precision@3"] <= 1.0


def test_inference_returns_ranked_results(
    seed_training_artifacts,
    temp_recommender_paths,
    patch_user_vector,
    tmp_path
):
    recommender = CareerRecommender(config=temp_recommender_paths)
    recommender.train()

    user_payload = {
        "age": 28,
        "education_level": "Master's",
        "grades_gpa": 3.6,
        "interests": "machine learning",
        "skills": "python;statistics",
        "past_roles": "Analyst;Consultant",
        "years_experience": 4,
    }
    user_json = tmp_path / "user.json"
    user_json.write_text(json.dumps(user_payload), encoding="utf-8")

    recommendations = recommender.infer(user_json)

    assert 1 <= len(recommendations) <= 5
    first = recommendations[0]
    assert "career_id" in first
    assert "rerank_score" in first
    
    # Check for new explanation structure
    assert "explanations" in first
    assert "why_recommended" in first["explanations"]
    assert "learning_plan" in first["explanations"]
    assert isinstance(first["explanations"]["why_recommended"], list)
    assert isinstance(first["explanations"]["learning_plan"], list)
    
    # Check audit log
    assert "audit_log" in first
    assert "timestamp" in first["audit_log"]
    assert "template_ids" in first["audit_log"]
