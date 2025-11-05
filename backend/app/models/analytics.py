"""Analytics event models and ORM definitions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import SettingsConfigDict
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AnalyticsEventType(str, Enum):
    """Supported analytics event types emitted by the platform."""

    PROFILE_COMPLETED = "profile_completed"
    REC_VIEWED = "rec_viewed"
    REC_CLICKED = "rec_clicked"
    FEEDBACK_SUBMITTED = "feedback_submitted"


class AnalyticsEvent(Base):
    """ORM representation of a captured analytics event."""

    __tablename__ = "analytics_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AnalyticsEventCreate(BaseModel):
    """Request payload for recording analytics events."""

    event_type: AnalyticsEventType
    user_id: str | None = None
    role: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rating: int | None = Field(default=None, ge=1, le=5)
    relevant: bool | None = None
    context: dict[str, Any] | None = None


class AnalyticsEventRead(BaseModel):
    """Serialized analytics event returned by the API."""

    model_config = SettingsConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: str
    event_type: AnalyticsEventType
    user_id: str | None = None
    role: str | None = None
    score: float | None = None
    rating: int | None = None
    relevant: bool | None = None
    context: dict[str, Any] | None = None
    created_at: datetime


class RoleMetric(BaseModel):
    """Aggregated engagement stats for a recommended role."""

    model_config = SettingsConfigDict(alias_generator=to_camel, populate_by_name=True)

    role: str
    views: int
    clicks: int


class ScoreMetric(BaseModel):
    """Average recommendation score per event type."""

    model_config = SettingsConfigDict(alias_generator=to_camel, populate_by_name=True)

    event_type: AnalyticsEventType
    average_score: float


class FeedbackHeatmapCell(BaseModel):
    """Heatmap bucket for feedback sentiment."""

    model_config = SettingsConfigDict(alias_generator=to_camel, populate_by_name=True)

    rating: int
    relevant: bool
    count: int


class AnalyticsTotals(BaseModel):
    """High-level counts for analytics events."""

    model_config = SettingsConfigDict(alias_generator=to_camel, populate_by_name=True)

    total_events: int
    by_type: dict[AnalyticsEventType, int]


class AnalyticsSummary(BaseModel):
    """Top-line analytics rolled up for the admin dashboard."""

    model_config = SettingsConfigDict(alias_generator=to_camel, populate_by_name=True)

    top_roles: list[RoleMetric] = Field(default_factory=list)
    average_scores: list[ScoreMetric] = Field(default_factory=list)
    feedback_heatmap: list[FeedbackHeatmapCell] = Field(default_factory=list)
    totals: AnalyticsTotals
