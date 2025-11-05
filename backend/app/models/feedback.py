"""Feedback request and response schemas for the CareerLens API."""

from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Payload for recording user feedback on recommendations."""

    user_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    relevant: bool
    comments: str | None = Field(default=None, max_length=1000)


class FeedbackRecord(FeedbackRequest):
    """Stored feedback entry with a timestamp."""

    submitted_at: datetime


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after feedback submission."""

    message: str = "Feedback recorded"
    entry: FeedbackRecord
