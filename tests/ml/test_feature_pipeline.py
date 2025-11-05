"""Tests for the feature pipeline end-to-end transformations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from ml.pipelines.feature_pipeline import FeaturePipeline


@pytest.fixture()
def sample_datasets() -> dict[str, pd.DataFrame]:
    karrierewege = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "job_title": [
                "Intern Analyst",
                "Junior Data Scientist",
                "Senior Data Scientist",
                "Associate Product Manager",
                "Product Manager"
            ],
            "start_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-06-01",
                    "2021-01-01",
                    "2019-03-15",
                    "2020-09-01"
                ]
            ),
        }
    )

    profiles = pd.DataFrame(
        {
            "user_id": [101, 102],
            "age": [24, 34],
            "education_level": ["Bachelor's", "Master's"],
            "grades_gpa": [3.4, 3.8],
            "interests": ["machine learning research", "product leadership"],
            "skills": ["python;statistics", "leadership;roadmaps"],
            "past_roles": ["Intern Analyst;Junior Data Scientist", "Associate Product Manager;Consultant"],
            "years_experience": [3, 8],
            "recommended_career": ["Senior Data Scientist", "Product Manager"],
            "required_skills": ["python, experimentation", "stakeholder management"],
            "education_requirement": ["master's", "bachelor's"],
            "typical_gpa": [3.5, 3.3],
        }
    )

    return {
        "karrierewege_sample": karrierewege,
        "ai_based_career_recommendation_system": profiles,
    }


@pytest.fixture()
def pipeline_config(tmp_path) -> dict:
    outputs_dir = tmp_path / "features"
    model_cache = tmp_path / "model_cache"

    return {
        "models": {
            "text_embedding": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "cache_dir": str(model_cache),
            }
        },
        "graph": {
            "dataset_prefix": "karrierewege",
            "person_col": "person_id",
            "role_col": "job_title",
            "date_col": "start_date",
            "min_transitions": 1,
            "embedding_dim": 128,
            "walk_length": 4,
            "num_walks": 5,
            "p": 1.0,
            "q": 0.5,
            "window": 3,
            "min_count": 1,
            "epochs": 1,
            "workers": 1,
        },
        "datasets": {
            "user_profiles": "ai_based_career_recommendation_system",
            "careers": "ai_based_career_recommendation_system",
        },
        "columns": {
            "user": {
                "id": "user_id",
                "age": "age",
                "education_level": "education_level",
                "gpa": "grades_gpa",
                "interests": "interests",
                "skills": "skills",
                "past_roles": "past_roles",
                "years_experience": "years_experience",
            },
            "career": {
                "id": "career_id",
                "title": "recommended_career",
                "skills": "required_skills",
                "education_requirement": "education_requirement",
                "typical_gpa": "typical_gpa",
            },
        },
        "outputs": {
            "user_features": str(outputs_dir / "user_features.parquet"),
            "career_features": str(outputs_dir / "career_item_features.parquet"),
            "graph_embeddings": str(outputs_dir / "graph_embeddings.parquet"),
        },
    }


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    """Replace model-dependent components with deterministic shims."""

    def fake_embed_text(text: str | None, show_progress: bool = False) -> np.ndarray:  # noqa: ARG001
        return np.full(384, 0.1, dtype=np.float32)

    monkeypatch.setattr("ml.pipelines.feature_engineering.embed_text", fake_embed_text)

    def fake_train_node2vec(graph, **kwargs):  # noqa: ANN001
        dim = kwargs.get("dimensions", 128)
        return {node: np.full(dim, 0.2, dtype=np.float32) for node in graph.nodes()}

    monkeypatch.setattr("ml.pipelines.graph_builder.train_node2vec", fake_train_node2vec)
    monkeypatch.setattr("ml.pipelines.feature_pipeline.train_node2vec", fake_train_node2vec)


def test_feature_pipeline_generates_expected_shapes(sample_datasets, pipeline_config):
    pipeline = FeaturePipeline(config=pipeline_config, save_outputs=True)

    pipeline.fit(sample_datasets)
    user_df, career_df, graph_df = pipeline.transform(sample_datasets)

    # Verify user features
    expected_user_dim = 914
    assert len(user_df) == 2
    assert user_df["feature_vector"].apply(len).eq(expected_user_dim).all()
    assert user_df["feature_vector"].apply(lambda vec: isinstance(vec, list)).all()

    # Verify career features
    expected_career_dim = 515
    assert len(career_df) == 2
    assert career_df["feature_vector"].apply(len).eq(expected_career_dim).all()

    # Verify graph embeddings
    assert not graph_df.empty
    assert graph_df["embedding"].apply(len).eq(128).all()

    # Check parquet outputs exist and maintain dtype
    user_parquet = Path(pipeline_config["outputs"]["user_features"])
    career_parquet = Path(pipeline_config["outputs"]["career_features"])
    graph_parquet = Path(pipeline_config["outputs"]["graph_embeddings"])

    assert user_parquet.exists()
    assert career_parquet.exists()
    assert graph_parquet.exists()

    persisted_user = pd.read_parquet(user_parquet)
    persisted_career = pd.read_parquet(career_parquet)
    persisted_graph = pd.read_parquet(graph_parquet)

    assert persisted_user["feature_vector"].apply(len).iloc[0] == expected_user_dim
    assert persisted_career["feature_vector"].apply(len).iloc[0] == expected_career_dim
    assert persisted_graph["embedding"].apply(len).iloc[0] == 128

    assert persisted_user["feature_vector"].apply(lambda vec: all(isinstance(val, float) for val in vec)).all()
    assert persisted_career["feature_vector"].apply(lambda vec: all(isinstance(val, float) for val in vec)).all()
