"""Pinecone-backed semantic retriever with retries and rate limiting."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.services.embedding_utils import generate_embedding


@dataclass
class RetrievalResult:
    """Structured representation of a retrieved match."""

    doc_id: str
    score: float
    metadata: dict[str, Any]


@dataclass
class PineconeRetriever:
    """Query Pinecone indexes with deterministic embeddings and resilience controls."""

    _settings: Any = field(default_factory=get_settings, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _index: Any | None = field(default=None, init=False, repr=False)
    _last_query_ts: float = field(default=0.0, init=False, repr=False)

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self._settings.pinecone_api_key
        index_name = self._settings.pinecone_index_name
        if not api_key or not index_name:
            raise RuntimeError("Pinecone configuration is incomplete. Provide API key and index name.")

        try:
            pinecone_module = importlib.import_module("pinecone")
        except ImportError as exc:  # pragma: no cover - dependency outside tests
            raise RuntimeError("The 'pinecone-client' package is required to query Pinecone.") from exc

        self._client = pinecone_module.Pinecone(api_key=api_key)
        return self._client

    def _load_index(self) -> Any:
        if self._index is not None:
            return self._index

        client = self._load_client()
        self._index = client.Index(self._settings.pinecone_index_name)
        return self._index

    def _respect_rate_limit(self) -> None:
        rate_limit = max(1, self._settings.pinecone_rate_limit_per_minute)
        delay = 60.0 / rate_limit
        elapsed = time.time() - self._last_query_ts
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k documents for the supplied query."""
        if top_k <= 0:
            return []

        index = self._load_index()
        self._respect_rate_limit()

        query_vector = generate_embedding(query, self._settings.pinecone_dimension)
        namespace = self._settings.pinecone_namespace or None

        last_error: Exception | None = None
        for attempt in range(max(1, self._settings.pinecone_max_retries)):
            try:
                response = index.query(  # type: ignore[call-arg]
                    vector=query_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                )
                self._last_query_ts = time.time()
                matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
                results: list[RetrievalResult] = []
                for match in matches:
                    if match is None:
                        continue
                    match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "")
                    score = float(match.get("score", 0.0)) if isinstance(match, dict) else float(getattr(match, "score", 0.0))
                    metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
                    results.append(RetrievalResult(doc_id=match_id, score=score, metadata=metadata))
                return results
            except Exception as exc:  # pragma: no cover - network interaction
                last_error = exc
                backoff = min(2**attempt, self._settings.pinecone_timeout_seconds)
                time.sleep(backoff)

        raise RuntimeError("Pinecone query failed after retries") from last_error


_retriever_instance: PineconeRetriever | None = None


def get_retriever() -> PineconeRetriever:
    """Provide a cached PineconeRetriever instance for FastAPI dependencies."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PineconeRetriever()
    return _retriever_instance
