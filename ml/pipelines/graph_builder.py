"""Career transition graph construction and Node2Vec embedding generation."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def build_career_graph(
    karrierewege_df: pd.DataFrame,
    person_col: str = "person_id",
    role_col: str = "job_title",
    date_col: str = "start_date",
    min_transitions: int = 2
) -> nx.DiGraph:
    """
    Build directed graph of career transitions from sequential job history.
    
    Args:
        karrierewege_df: DataFrame with person_id, job_title, start_date columns
        person_col: Column name for person identifier
        role_col: Column name for job title
        date_col: Column name for job start date
        min_transitions: Minimum edge weight to include (filter rare transitions)
    
    Returns:
        Directed graph where nodes are job titles and edges are transitions
    """
    LOGGER.info("Building career transition graph from KARRIEREWEGE data")
    
    G = nx.DiGraph()
    transition_count = 0
    
    # Group by person and extract career sequences
    for person_id, group in karrierewege_df.groupby(person_col):
        # Sort by date to get chronological order
        roles = group.sort_values(date_col)[role_col].tolist()
        
        # Skip if person has only one role
        if len(roles) < 2:
            continue
        
        # Add edges for sequential transitions
        for i in range(len(roles) - 1):
            source = str(roles[i]).strip()
            target = str(roles[i + 1]).strip()
            
            # Skip self-loops and empty strings
            if not source or not target or source == target:
                continue
            
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
            
            transition_count += 1
    
    # Filter edges with low weight
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] < min_transitions
    ]
    G.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    LOGGER.info(
        f"Graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{transition_count} total transitions"
    )
    
    return G


def compute_career_centrality(G: nx.DiGraph) -> dict[str, float]:
    """
    Compute PageRank centrality scores for all career nodes.
    
    Args:
        G: Career transition graph
    
    Returns:
        Dictionary mapping job title to centrality score
    """
    LOGGER.info("Computing PageRank centrality scores")
    
    if G.number_of_nodes() == 0:
        return {}
    
    try:
        centrality = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception as exc:
        LOGGER.warning(f"PageRank failed: {exc}. Using degree centrality as fallback.")
        centrality = nx.degree_centrality(G)
    
    return centrality


def train_node2vec(
    G: nx.DiGraph,
    dimensions: int = 128,
    walk_length: int = 10,
    num_walks: int = 50,
    p: float = 1.0,
    q: float = 0.5,
    workers: int = 4,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 5
) -> dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings on career transition graph.
    
    Args:
        G: Career transition graph
        dimensions: Embedding dimensionality
        walk_length: Length of random walks
        num_walks: Number of walks per node
        p: Return parameter (BFS vs DFS)
        q: In-out parameter (local vs global exploration)
        workers: Number of parallel workers
        window: Context window size for Skip-gram
        min_count: Minimum node frequency
        epochs: Training epochs
    
    Returns:
        Dictionary mapping job title to embedding vector
    """
    LOGGER.info(f"Training Node2Vec embeddings (dim={dimensions})")
    
    if G.number_of_nodes() == 0:
        LOGGER.warning("Empty graph, returning empty embeddings")
        return {}
    
    try:
        from node2vec import Node2Vec
    except ImportError as exc:
        LOGGER.error("node2vec not installed. Run: pip install node2vec")
        raise ImportError(
            "node2vec required for graph embeddings. "
            "Install with: pip install node2vec"
        ) from exc
    
    # Initialize Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=False
    )
    
    # Train Skip-gram model
    LOGGER.info("Training Skip-gram model on random walks")
    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=4,
        epochs=epochs,
        workers=workers
    )
    
    # Extract embeddings
    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[str(node)] = model.wv[str(node)]
        except KeyError:
            LOGGER.warning(f"No embedding found for node: {node}")
            embeddings[str(node)] = np.zeros(dimensions, dtype=np.float32)
    
    LOGGER.info(f"Generated embeddings for {len(embeddings)} career nodes")
    return embeddings


def save_graph_artifacts(
    G: nx.DiGraph,
    embeddings: dict[str, np.ndarray],
    centrality: dict[str, float],
    output_dir: Path
) -> None:
    """
    Save graph, embeddings, and centrality scores to disk.
    
    Args:
        G: Career transition graph
        embeddings: Node2Vec embeddings
        centrality: PageRank scores
        output_dir: Directory to save artifacts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save graph
    graph_path = output_dir / "career_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    LOGGER.info(f"Saved graph to {graph_path}")
    
    # Save embeddings
    embeddings_path = output_dir / "node2vec_embeddings.pkl"
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    LOGGER.info(f"Saved embeddings to {embeddings_path}")
    
    # Save centrality
    centrality_path = output_dir / "career_centrality.pkl"
    with open(centrality_path, "wb") as f:
        pickle.dump(centrality, f)
    LOGGER.info(f"Saved centrality scores to {centrality_path}")
    
    # Save metadata
    metadata = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "embedding_dim": len(next(iter(embeddings.values()))) if embeddings else 0,
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    metadata_path = output_dir / "graph_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    LOGGER.info(f"Saved metadata to {metadata_path}")


def load_graph_artifacts(artifact_dir: Path) -> tuple[nx.DiGraph, dict[str, np.ndarray], dict[str, float]]:
    """
    Load saved graph artifacts.
    
    Args:
        artifact_dir: Directory containing saved artifacts
    
    Returns:
        Tuple of (graph, embeddings, centrality)
    """
    graph_path = artifact_dir / "career_graph.pkl"
    embeddings_path = artifact_dir / "node2vec_embeddings.pkl"
    centrality_path = artifact_dir / "career_centrality.pkl"
    
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    
    with open(centrality_path, "rb") as f:
        centrality = pickle.load(f)
    
    LOGGER.info(f"Loaded graph with {G.number_of_nodes()} nodes and {len(embeddings)} embeddings")
    return G, embeddings, centrality


def get_typical_trajectory(
    career: str,
    G: nx.DiGraph,
    max_steps: int = 3
) -> list[str]:
    """
    Extract typical career progression path from graph.
    
    Args:
        career: Starting job title
        G: Career transition graph
        max_steps: Maximum number of progression steps
    
    Returns:
        Ordered list of job titles representing typical progression
    """
    if career not in G.nodes():
        return [career]
    
    trajectory = [career]
    current = career
    
    for _ in range(max_steps):
        # Get outgoing edges
        if not G.out_edges(current):
            break
        
        # Find most common next role (highest weight)
        next_roles = [
            (target, G[current][target]["weight"])
            for _, target in G.out_edges(current)
        ]
        
        if not next_roles:
            break
        
        # Sort by weight descending
        next_roles.sort(key=lambda x: x[1], reverse=True)
        current = next_roles[0][0]
        trajectory.append(current)
    
    return trajectory
