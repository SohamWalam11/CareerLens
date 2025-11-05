"""Text embedding utilities using sentence-transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Lazy-load the model to avoid import-time overhead
_MODEL = None
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"


def configure_embedder(
    model_name: str | None = None,
    embedding_dim: int | None = None,
    cache_dir: Path | None = None
) -> None:
    """Configure model settings and clear cached instance.

    Args:
        model_name: HuggingFace model identifier.
        embedding_dim: Expected embedding dimensionality.
        cache_dir: Directory to store model cache.
    """

    global MODEL_NAME, EMBEDDING_DIM, CACHE_DIR, _MODEL

    if model_name:
        if model_name != MODEL_NAME:
            LOGGER.info("Updating sentence-transformer model to %s", model_name)
            MODEL_NAME = model_name
            _MODEL = None  # Clear cached model so it reloads

    if embedding_dim is not None:
        EMBEDDING_DIM = embedding_dim

    if cache_dir is not None:
        CACHE_DIR = Path(cache_dir)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_embedder():
    """Lazy-load the sentence-transformer model."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Loading embedding model: {MODEL_NAME}")
            _MODEL = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))
            LOGGER.info(f"Model loaded successfully (dim={EMBEDDING_DIM})")
        except ImportError as exc:
            LOGGER.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise ImportError(
                "sentence-transformers required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc
    return _MODEL


def embed_text(text: str | None, show_progress: bool = False) -> np.ndarray:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Input text (skills, interests, or job description)
        show_progress: Whether to show progress bar (useful for batch processing)
    
    Returns:
        384-dimensional embedding vector
    """
    if not text or pd.isna(text):
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # Normalize text
    cleaned = str(text).lower().strip()
    
    # Truncate to model's max length (512 tokens for MiniLM)
    if len(cleaned) > 2048:  # ~512 tokens worth of characters
        cleaned = cleaned[:2048]
    
    model = get_embedder()
    embedding = model.encode(cleaned, show_progress_bar=show_progress, convert_to_numpy=True)
    
    return embedding.astype(np.float32)


def embed_batch(texts: Sequence[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar
    
    Returns:
        Array of shape (len(texts), 384)
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    
    # Normalize all texts
    cleaned_texts = []
    for text in texts:
        if not text or pd.isna(text):
            cleaned_texts.append("")
        else:
            cleaned = str(text).lower().strip()
            if len(cleaned) > 2048:
                cleaned = cleaned[:2048]
            cleaned_texts.append(cleaned)
    
    model = get_embedder()
    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings.astype(np.float32)


def cache_embeddings(
    df: pd.DataFrame,
    text_column: str,
    output_path: Path,
    id_column: str = "id"
) -> pd.DataFrame:
    """
    Generate and cache embeddings for a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text to embed
        output_path: Path to save cached embeddings (parquet format)
        id_column: Name of ID column for joining
    
    Returns:
        DataFrame with id and embedding columns
    """
    LOGGER.info(f"Generating embeddings for {len(df)} records from column '{text_column}'")
    
    texts = df[text_column].tolist()
    embeddings = embed_batch(texts, show_progress=True)
    
    # Create result DataFrame
    result = pd.DataFrame({
        id_column: df[id_column] if id_column in df.columns else df.index,
        "embedding": [emb.tolist() for emb in embeddings]
    })
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    LOGGER.info(f"Cached embeddings to {output_path}")
    
    return result


def load_cached_embeddings(cache_path: Path) -> dict[str, np.ndarray]:
    """
    Load cached embeddings from parquet file.
    
    Args:
        cache_path: Path to cached embeddings file
    
    Returns:
        Dictionary mapping ID to embedding vector
    """
    if not cache_path.exists():
        LOGGER.warning(f"Cache file not found: {cache_path}")
        return {}
    
    df = pd.read_parquet(cache_path)
    
    # Convert to dictionary
    embeddings = {}
    for _, row in df.iterrows():
        id_val = row.iloc[0]  # First column is ID
        emb_list = row.iloc[1]  # Second column is embedding
        embeddings[str(id_val)] = np.array(emb_list, dtype=np.float32)
    
    LOGGER.info(f"Loaded {len(embeddings)} cached embeddings from {cache_path}")
    return embeddings
