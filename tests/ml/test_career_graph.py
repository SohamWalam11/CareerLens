"""Tests for career graph utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pipelines.career_graph import (  # noqa: E402
    CareerGraphModel,
    build_transition_graph,
    train_node2vec_embeddings
)


@pytest.fixture()
def sample_sequences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "t": [1, 2, 3, 1, 2],
            "job_title": [
                "Analyst",
                "Senior Analyst",
                "Data Scientist",
                "Analyst",
                "Product Manager"
            ],
        }
    )


def test_build_transition_graph_counts_edges(sample_sequences):
    graph = build_transition_graph(sample_sequences)

    assert graph.number_of_nodes() == 4
    assert graph.has_edge("Analyst", "Senior Analyst")
    assert graph.has_edge("Senior Analyst", "Data Scientist")
    assert graph.has_edge("Analyst", "Product Manager")
    assert graph["Analyst"]["Senior Analyst"]["weight"] == 1
    assert graph["Analyst"]["Product Manager"]["weight"] == 1


def test_career_graph_model_neighbors_and_probabilities(tmp_path, sample_sequences):
    graph = build_transition_graph(sample_sequences)

    embeddings = {
        node: np.linspace(0.1, 0.1 + 0.01 * idx, 8, dtype=np.float32)
        for idx, node in enumerate(graph.nodes())
    }
    model = CareerGraphModel(graph=graph, embeddings=embeddings)

    neighbors = model.career_neighbors("Senior Analyst", k=2)
    assert len(neighbors) == 2
    assert neighbors[0][0] != "Senior Analyst"
    assert 0.0 <= neighbors[0][1] <= 1.0

    prob = model.career_transition_prob("Analyst", "Senior Analyst")
    assert prob == pytest.approx(0.5, rel=1e-3)

    model.save(tmp_path)
    assert (tmp_path / "career_graph.pkl").exists()
    assert (tmp_path / "node_embeddings.pkl").exists()
    assert (tmp_path / "node_embeddings.csv").exists()


def test_fit_from_sequences_uses_custom_node2vec(monkeypatch, sample_sequences):
    def fake_train(graph: nx.DiGraph, **kwargs):  # noqa: ANN001
        dim = kwargs.get("dimensions", 16)
        return {node: np.full(dim, 0.3, dtype=np.float32) for node in graph.nodes()}

    monkeypatch.setattr(
        "ml.pipelines.career_graph.train_node2vec_embeddings",
        fake_train
    )

    model = CareerGraphModel.fit_from_sequences(sample_sequences, node2vec_params={"dimensions": 16})

    assert len(model.embeddings) == model.graph.number_of_nodes()
    vector = next(iter(model.embeddings.values()))
    assert vector.shape == (16,)
    assert np.allclose(vector, 0.3)
