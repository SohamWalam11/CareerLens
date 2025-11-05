"""Career graph construction and Node2Vec embeddings."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts" / "graph"


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def build_transition_graph(
    sequences: pd.DataFrame,
    person_col: str = "person_id",
    time_col: str = "t",
    job_col: str = "job_title",
    min_transitions: int = 1
) -> nx.DiGraph:
    """Build a directed weighted transition graph from career sequences."""
    _validate_columns(sequences, [person_col, time_col, job_col])

    graph = nx.DiGraph()
    transitions = 0

    for _, group in sequences.groupby(person_col):
        ordered = group.sort_values(time_col)
        titles = ordered[job_col].astype(str).str.strip().tolist()

        for idx in range(len(titles) - 1):
            src = titles[idx]
            dst = titles[idx + 1]
            if not src or not dst or src == dst:
                continue

            weight = graph[src][dst]["weight"] + 1 if graph.has_edge(src, dst) else 1
            graph.add_edge(src, dst, weight=weight)
            transitions += 1

    if min_transitions > 1:
        to_remove = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if data.get("weight", 0) < min_transitions
        ]
        graph.remove_edges_from(to_remove)

    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)

    LOGGER.info(
        "Career graph built with %d nodes, %d edges, %d transitions",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        transitions
    )
    return graph


def train_node2vec_embeddings(
    graph: nx.DiGraph,
    dimensions: int = 128,
    walk_length: int = 10,
    num_walks: int = 50,
    window: int = 5,
    min_count: int = 1,
    workers: int = 2,
    epochs: int = 5,
    p: float = 1.0,
    q: float = 0.5
) -> dict[str, np.ndarray]:
    """Train Node2Vec embeddings for graph nodes using gensim backend."""
    if graph.number_of_nodes() == 0:
        LOGGER.warning("Empty graph; returning no embeddings")
        return {}

    try:
        from node2vec import Node2Vec
    except ImportError as exc:  # noqa: TRY003
        raise ImportError(
            "node2vec package is required for embedding training. Install with: pip install node2vec"
        ) from exc

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
        quiet=True
    )

    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=4,
        epochs=epochs,
        workers=workers
    )

    embeddings: dict[str, np.ndarray] = {}
    for node in graph.nodes():
        try:
            embeddings[node] = model.wv.get_vector(node).astype(np.float32)
        except KeyError:
            embeddings[node] = np.zeros(dimensions, dtype=np.float32)
    LOGGER.info("Generated embeddings for %d careers", len(embeddings))
    return embeddings


@dataclass
class CareerGraphModel:
    """Container for graph structure, embeddings, and convenience APIs."""

    graph: nx.DiGraph
    embeddings: dict[str, np.ndarray]

    def career_neighbors(self, title: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k similar careers using cosine similarity."""
        if title not in self.embeddings:
            return []

        target = self.embeddings[title]
        if not target.any():
            return []

        target_norm = np.linalg.norm(target)
        results: list[tuple[str, float]] = []

        for other, vector in self.embeddings.items():
            if other == title:
                continue
            denom = target_norm * np.linalg.norm(vector)
            if denom == 0:
                continue
            score = float(np.dot(target, vector) / denom)
            results.append((other, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def career_transition_prob(self, src: str, dst: str) -> float:
        """Return transition probability src -> dst based on outgoing weights."""
        if not self.graph.has_edge(src, dst):
            return 0.0
        total_weight = 0.0
        for _, _, data in self.graph.out_edges(src, data=True):
            total_weight += float(data.get("weight", 0.0))
        if total_weight == 0:
            return 0.0
        weight = float(self.graph[src][dst].get("weight", 0.0))
        return weight / total_weight

    def save(self, output_dir: str | Path | None = None) -> None:
        """Persist graph, embeddings, and CSV vectors to disk."""
        output_path = Path(output_dir) if output_dir else ARTIFACT_DIR
        output_path.mkdir(parents=True, exist_ok=True)

        graph_path = output_path / "career_graph.pkl"
        embeddings_path = output_path / "node_embeddings.pkl"
        csv_path = output_path / "node_embeddings.csv"

        with graph_path.open("wb") as handle:
            pickle.dump(self.graph, handle)
        with embeddings_path.open("wb") as handle:
            pickle.dump(self.embeddings, handle)

        df = _embeddings_to_dataframe(self.embeddings)
        df.to_csv(csv_path, index=False)

        LOGGER.info("Saved graph artifacts to %s", output_path)

    @classmethod
    def fit_from_sequences(
        cls,
        sequences: pd.DataFrame,
        *,
        person_col: str = "person_id",
        time_col: str = "t",
        job_col: str = "job_title",
        min_transitions: int = 1,
        node2vec_params: dict[str, Any] | None = None
    ) -> "CareerGraphModel":
        graph = build_transition_graph(
            sequences,
            person_col=person_col,
            time_col=time_col,
            job_col=job_col,
            min_transitions=min_transitions
        )

        params = {
            "dimensions": 128,
            "walk_length": 10,
            "num_walks": 50,
            "window": 5,
            "min_count": 1,
            "workers": 2,
            "epochs": 5,
            "p": 1.0,
            "q": 0.5
        }
        if node2vec_params:
            params.update(node2vec_params)

        embeddings = train_node2vec_embeddings(graph, **params)
        return cls(graph=graph, embeddings=embeddings)

    @classmethod
    def load(cls, artifact_dir: str | Path | None = None) -> "CareerGraphModel":
        """Load model artifacts from disk."""
        path = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
        graph_path = path / "career_graph.pkl"
        embeddings_path = path / "node_embeddings.pkl"

        if not graph_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError("Graph artifacts not found; run save() first")

        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        with embeddings_path.open("rb") as handle:
            embeddings = pickle.load(handle)
        return cls(graph=graph, embeddings=embeddings)


def _embeddings_to_dataframe(embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    if not embeddings:
        return pd.DataFrame(columns=["career_title"])

    dim = len(next(iter(embeddings.values())))
    columns = ["career_title"] + [f"dim_{idx}" for idx in range(dim)]
    records: list[list[float]] = []

    for title, vector in embeddings.items():
        row = [title] + [float(val) for val in vector]
        records.append(row)
    return pd.DataFrame(records, columns=columns)


__all__ = [
    "build_transition_graph",
    "train_node2vec_embeddings",
    "CareerGraphModel"
]
