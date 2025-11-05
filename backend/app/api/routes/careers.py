"""Primary CareerLens API endpoints for profiles, recommendations, trajectories, and feedback."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.analytics import AnalyticsEventCreate, AnalyticsEventType
from app.models.feedback import FeedbackRequest, FeedbackResponse
from app.models.graph import TrajectoryResponse
from app.models.profile import (
    CareerRecommendationRequest,
    ProfileResponse,
    RecommendationAPIResponse,
    RecommendationQuery,
    UserProfilePayload,
)
from app.services.analytics_service import AnalyticsService, get_analytics_service
from app.services.career_graph import CareerGraphService, get_career_graph_service
from app.services.feedback_store import FeedbackRepository, get_feedback_repository
from app.services.profile_store import ProfileRepository, get_profile_repository
from app.services.recommendations import RecommendationService, get_recommendation_service

router = APIRouter(tags=["career-experience"])


@router.post("/profile", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED, tags=["profiles"])
def upsert_profile(
    payload: UserProfilePayload,
    repository: ProfileRepository = Depends(get_profile_repository),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> ProfileResponse:
    """Save or update a user profile."""
    stored = repository.upsert(payload)
    analytics_service.track_event(
        AnalyticsEventCreate(
            event_type=AnalyticsEventType.PROFILE_COMPLETED,
            user_id=payload.user_id,
        )
    )
    return ProfileResponse(user_profile=stored, message="Profile stored")


@router.post("/recommend", response_model=RecommendationAPIResponse, tags=["recommendations"])
def generate_recommendations(
    payload: RecommendationQuery,
    profile_repository: ProfileRepository = Depends(get_profile_repository),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> RecommendationAPIResponse:
    """Return top candidate roles along with explanations and skill gaps."""
    profile: CareerRecommendationRequest | None = None
    user_id = payload.user_id

    if payload.profile is not None:
        profile = payload.profile
        if user_id:
            repository_payload = UserProfilePayload(user_id=user_id, **payload.profile.model_dump())
            profile_repository.upsert(repository_payload)
    elif payload.user_id:
        stored = profile_repository.get(payload.user_id)
        if stored is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found for supplied user_id")
        profile = CareerRecommendationRequest(**stored.model_dump(exclude={"user_id"}))
        user_id = stored.user_id

    if profile is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A profile must be provided or retrievable via user_id")

    bundles = recommendation_service.generate_detailed_recommendations(profile)

    for bundle in bundles:
        analytics_service.track_event(
            AnalyticsEventCreate(
                event_type=AnalyticsEventType.REC_VIEWED,
                user_id=user_id,
                role=bundle.role.title,
                score=bundle.role.fit_score,
                context={"source": "detailed_recommendations"},
            )
        )

    snapshot = CareerRecommendationRequest(**profile.model_dump())

    return RecommendationAPIResponse(
        user_id=user_id or "anonymous",
        recommendations=bundles,
        total=len(bundles),
        profile_snapshot=snapshot,
    )


@router.get("/trajectory", response_model=TrajectoryResponse, tags=["trajectory"])
def get_trajectory(
    role: str = Query(..., min_length=2, description="Target role to inspect neighboring transitions"),
    service: CareerGraphService = Depends(get_career_graph_service),
) -> TrajectoryResponse:
    """Return neighboring roles and transition probabilities for a given role."""
    try:
        return service.get_trajectory(role)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED, tags=["feedback"])
def submit_feedback(
    payload: FeedbackRequest,
    repository: FeedbackRepository = Depends(get_feedback_repository),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> FeedbackResponse:
    """Persist user feedback on a recommendation."""
    record = repository.add(payload)
    analytics_service.track_event(
        AnalyticsEventCreate(
            event_type=AnalyticsEventType.FEEDBACK_SUBMITTED,
            user_id=payload.user_id,
            role=payload.role,
            rating=payload.rating,
            relevant=payload.relevant,
            context={"source": "feedback_endpoint"},
        )
    )
    return FeedbackResponse(entry=record)
