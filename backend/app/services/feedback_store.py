"""In-memory persistence helpers for feedback submissions."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from app.models.feedback import FeedbackRecord, FeedbackRequest


class FeedbackRepository:
    """Store feedback entries in memory until a database is available."""

    def __init__(self) -> None:
        self._feedback: list[FeedbackRecord] = []
        self._lock = Lock()

    def add(self, request: FeedbackRequest) -> FeedbackRecord:
        """Persist a feedback entry and return the record."""
        record = FeedbackRecord(
            **request.model_dump(),
            submitted_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._feedback.append(record)
        return record

    def list_all(self) -> list[FeedbackRecord]:
        """Return a snapshot of all stored feedback."""
        with self._lock:
            return list(self._feedback)


_feedback_repo: FeedbackRepository | None = None


def get_feedback_repository() -> FeedbackRepository:
    """FastAPI dependency provider for the feedback repository."""
    global _feedback_repo
    if _feedback_repo is None:
        _feedback_repo = FeedbackRepository()
    return _feedback_repo
