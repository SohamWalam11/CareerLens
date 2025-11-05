"""REST endpoints for analytics events."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.models.analytics import AnalyticsEventCreate, AnalyticsEventRead, AnalyticsSummary
from app.services.analytics_service import AnalyticsService, get_analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/events", response_model=AnalyticsEventRead, status_code=status.HTTP_201_CREATED)
def record_event(
    payload: AnalyticsEventCreate,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsEventRead:
    """Persist a single analytics event."""

    return service.track_event(payload)


@router.get("/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsSummary:
    """Return aggregate analytics for the admin dashboard."""

    return service.summarize()
