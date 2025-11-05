"""Utility helpers for generating deterministic embeddings when a provider is unavailable."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


def generate_embedding(text: str, dimension: int) -> list[float]:
    """Create a deterministic embedding vector for the given text.

    This helper keeps the service usable without external embedding providers by
    deriving a pseudo-random yet repeatable vector from token hashes. The
    resulting vector is normalised to unit length to mimic cosine-usable
    embeddings.
    """
    dimension = max(1, dimension)
    if not text:
        return [0.0] * dimension

    vector = [0.0] * dimension
    tokens = [token for token in text.split() if token]
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        for idx in range(0, len(digest), 2):
            chunk = digest[idx : idx + 2]
            if len(chunk) < 2:
                continue
            bucket = (idx // 2 + int(chunk, 16)) % dimension
            vector[bucket] += 1.0

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector

    return [value / magnitude for value in vector]


def batch_generate_embeddings(texts: Iterable[str], dimension: int) -> list[list[float]]:
    """Vectorise a collection of texts using the deterministic embedding helper."""
    return [generate_embedding(text, dimension) for text in texts]
