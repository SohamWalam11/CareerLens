"""Service layer for recording and summarizing analytics events."""

from __future__ import annotations

from collections import defaultdict

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analytics import (
    AnalyticsEvent,
    AnalyticsEventCreate,
    AnalyticsEventRead,
    AnalyticsEventType,
    AnalyticsSummary,
    AnalyticsTotals,
    FeedbackHeatmapCell,
    RoleMetric,
    ScoreMetric,
)


class AnalyticsService:
    """Encapsulate persistence and aggregation for analytics events."""

    def __init__(self, db: Session):
        self._db = db
        # Ensure the analytics_events table exists for lightweight deployments.
        AnalyticsEvent.__table__.create(bind=self._db.get_bind(), checkfirst=True)

    def track_event(self, payload: AnalyticsEventCreate) -> AnalyticsEventRead:
        """Persist a new analytics event and return the serialized record."""

        record = AnalyticsEvent(
            event_type=payload.event_type.value,
            user_id=payload.user_id,
            role=payload.role,
            score=payload.score,
            rating=payload.rating,
            relevant=payload.relevant,
            context=payload.context or {},
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)
        return AnalyticsEventRead.model_validate(record)

    def summarize(self) -> AnalyticsSummary:
        """Aggregate analytics data for admin consumption."""

        events = self._db.execute(select(AnalyticsEvent)).scalars().all()

        role_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "clicks": 0})
        score_accumulator: dict[AnalyticsEventType, list[float]] = defaultdict(list)
        feedback_counts: dict[tuple[int, bool], int] = defaultdict(int)
        totals: dict[AnalyticsEventType, int] = defaultdict(int)

        for event in events:
            event_type = AnalyticsEventType(event.event_type)
            totals[event_type] += 1

            if event.role:
                if event_type == AnalyticsEventType.REC_VIEWED:
                    role_metrics[event.role]["views"] += 1
                elif event_type == AnalyticsEventType.REC_CLICKED:
                    role_metrics[event.role]["clicks"] += 1

            if event.score is not None:
                score_accumulator[event_type].append(event.score)

            if event_type == AnalyticsEventType.FEEDBACK_SUBMITTED and event.rating is not None and event.relevant is not None:
                feedback_counts[(event.rating, event.relevant)] += 1

        top_roles = [
            RoleMetric(role=role, views=data["views"], clicks=data["clicks"])
            for role, data in sorted(
                role_metrics.items(), key=lambda item: (item[1]["views"] + item[1]["clicks"]), reverse=True
            )
        ][:10]

        average_scores = [
            ScoreMetric(event_type=event_type, average_score=round(sum(scores) / len(scores), 3))
            for event_type, scores in score_accumulator.items()
            if scores
        ]

        feedback_heatmap = [
            FeedbackHeatmapCell(rating=rating, relevant=relevant, count=count)
            for (rating, relevant), count in sorted(
                feedback_counts.items(), key=lambda item: (item[0][0], item[0][1]), reverse=True
            )
        ]

        summary = AnalyticsSummary(
            top_roles=top_roles,
            average_scores=average_scores,
            feedback_heatmap=feedback_heatmap,
            totals=AnalyticsTotals(
                total_events=sum(totals.values()),
                by_type={event_type: count for event_type, count in totals.items()},
            ),
        )
        return summary


def get_analytics_service(db: Session = Depends(get_db)) -> AnalyticsService:
    """FastAPI dependency hook for the analytics service."""

    return AnalyticsService(db)
