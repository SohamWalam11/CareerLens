"""Career recommendation endpoints."""

from fastapi import APIRouter, Depends

from app.models.analytics import AnalyticsEventCreate, AnalyticsEventType
from app.models.profile import CareerRecommendationRequest, CareerRecommendationResponse
from app.services.analytics_service import AnalyticsService, get_analytics_service
from app.services.recommendations import RecommendationService, get_recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/generate", response_model=CareerRecommendationResponse)
def generate_recommendations(
    request: CareerRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> CareerRecommendationResponse:
    """Return top career paths and supporting analytics for a given profile."""
    response = service.generate_recommendations(request)

    for role in response.recommendations:
        analytics_service.track_event(
            AnalyticsEventCreate(
                event_type=AnalyticsEventType.REC_VIEWED,
                role=role.title,
                score=role.fit_score,
                context={"source": "lightweight_recommendations"},
            )
        )

    return response
