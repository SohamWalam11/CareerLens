"""Utilities for embedding and upserting knowledge base content into Pinecone."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from app.core.config import get_settings
from app.services.embedding_utils import batch_generate_embeddings


@dataclass(frozen=True)
class KnowledgeDocument:
    """Representation of a knowledge item ready for vectorisation."""

    doc_id: str
    text: str
    metadata: dict[str, str]


def _load_role_documents(base_path: Path) -> List[KnowledgeDocument]:
    documents: List[KnowledgeDocument] = []
    roles_dir = base_path / "roles"
    if not roles_dir.exists():
        return documents

    for path in sorted(roles_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            KnowledgeDocument(
                doc_id=f"role::{path.stem}",
                text=text,
                metadata={
                    "source_type": "role_page",
                    "path": str(path.relative_to(base_path.parent)),
                    "title": path.stem.replace("-", " ").title(),
                },
            )
        )
    return documents


def _load_insight_documents(base_path: Path) -> List[KnowledgeDocument]:
    insights_path = base_path / "insights" / "dataset_insights.json"
    if not insights_path.exists():
        return []

    insights_data = json.loads(insights_path.read_text(encoding="utf-8"))
    documents: List[KnowledgeDocument] = []

    for block in insights_data.get("insights", []):
        category = block.get("category", "unknown")
        data = block.get("data", {})
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        documents.append(
            KnowledgeDocument(
                doc_id=f"insight::{category}",
                text=f"Category: {category}\n{serialized}",
                metadata={
                    "source_type": "dataset_insight",
                    "category": category,
                    "path": str(insights_path.relative_to(base_path.parent)),
                },
            )
        )
    return documents


def _load_explanation_documents() -> List[KnowledgeDocument]:
    from app.services import explanations

    templates = getattr(explanations, "_WEIGHT_TEMPLATES", {})
    explainer_summary = ["Explanation templates and rationale cues:"]
    for name, template in templates.items():
        explainer_summary.append(f"- {name}: {template}")

    exemplar_text = "\n".join(explainer_summary)
    return [
        KnowledgeDocument(
            doc_id="exemplar::explanations",
            text=exemplar_text,
            metadata={
                "source_type": "explanation_exemplar",
                "path": "app/services/explanations.py",
            },
        )
    ]


def collect_documents(additional_paths: Iterable[Path] | None = None) -> List[KnowledgeDocument]:
    """Gather documents from the knowledge base and optional additional paths."""
    base_path = Path(__file__).resolve().parents[2] / "knowledge_base"
    documents = []
    documents.extend(_load_role_documents(base_path))
    documents.extend(_load_insight_documents(base_path))
    documents.extend(_load_explanation_documents())

    if additional_paths:
        for path in additional_paths:
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                documents.append(
                    KnowledgeDocument(
                        doc_id=f"extra::{path.stem}",
                        text=text,
                        metadata={
                            "source_type": "additional",
                            "path": str(path),
                        },
                    )
                )
    return documents


def ingest_knowledge_base(additional_paths: Iterable[Path] | None = None, batch_size: int | None = None) -> int:
    """Vectorise documents and upsert them into the configured Pinecone index."""
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        raise RuntimeError("Pinecone configuration is incomplete. Set API key and index name.")

    try:
        import importlib

        pinecone_module = importlib.import_module("pinecone")
    except ImportError as exc:  # pragma: no cover - external dependency
        raise RuntimeError("The 'pinecone-client' package is required for ingestion.") from exc

    client = pinecone_module.Pinecone(api_key=settings.pinecone_api_key)
    index = client.Index(settings.pinecone_index_name)

    documents = collect_documents(additional_paths)
    if not documents:
        return 0

    embeddings = batch_generate_embeddings((doc.text for doc in documents), settings.pinecone_dimension)

    rate_limit = max(1, settings.pinecone_rate_limit_per_minute)
    batch_size = batch_size or min(50, rate_limit)
    namespace = settings.pinecone_namespace or None

    upsert_payload = [
        {
            "id": doc.doc_id,
            "values": vector,
            "metadata": doc.metadata | {"content": doc.text},
        }
        for doc, vector in zip(documents, embeddings)
    ]

    sleep_interval = 60.0 / rate_limit
    total_upserted = 0

    for start in range(0, len(upsert_payload), batch_size):
        chunk = upsert_payload[start : start + batch_size]
        for attempt in range(settings.pinecone_max_retries):
            try:
                index.upsert(vectors=chunk, namespace=namespace)
                total_upserted += len(chunk)
                break
            except Exception as exc:  # pragma: no cover - network interaction
                if attempt == settings.pinecone_max_retries - 1:
                    raise
                time.sleep(min(sleep_interval, settings.pinecone_timeout_seconds))
        time.sleep(sleep_interval)

    return total_upserted


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    count = ingest_knowledge_base()
    print(f"Ingested {count} documents into Pinecone.")
