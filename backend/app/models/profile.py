"""Data models for end-user profile and recommendations."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class SkillGap(BaseModel):
    """Represents the difference between required and current skill levels."""

    skill: str
    current_level: float = Field(ge=0.0, le=1.0, default=0.0)
    target_level: float = Field(ge=0.0, le=1.0, default=1.0)


class LearningResource(BaseModel):
    """Learning asset suggested to close a skill gap."""

    title: str
    provider: str | None = None
    url: str | None = None
    estimated_hours: int | None = None


class CareerPath(BaseModel):
    """Suggested career plan entry with trajectory metadata."""

    title: str
    fit_score: float = Field(ge=0.0, le=1.0)
    description: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    trajectory: list[str] = Field(default_factory=list)


class CareerRecommendationResponse(BaseModel):
    """Response payload containing recommended careers and guidance."""

    user_summary: dict[str, Any]
    recommendations: list[CareerPath]
    skill_gaps: list[SkillGap]
    learning_plan: list[LearningResource]


class CareerRecommendationRequest(BaseModel):
    """Input payload capturing core profile details for recommendations."""

    name: str
    age: int
    education_level: str
    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class UserProfilePayload(CareerRecommendationRequest):
    """Stored representation of a user profile keyed by user identifier."""

    user_id: str = Field(min_length=1)


class ProfileResponse(BaseModel):
    """Response schema for profile upsert operations."""

    user_profile: UserProfilePayload
    message: str = "Profile stored"


class RecommendationQuery(BaseModel):
    """Request payload for generating recommendations."""

    user_id: str | None = Field(default=None)
    profile: CareerRecommendationRequest | None = Field(default=None)

    @model_validator(mode="after")
    def validate_payload(self) -> "RecommendationQuery":
        """Ensure that either a stored user or inline profile is supplied."""
        if not self.user_id and not self.profile:
            msg = "Provide either user_id of a stored profile or an inline profile payload"
            raise ValueError(msg)
        return self


class RecommendationExplanation(BaseModel):
    """Structured explanation bundle returned alongside recommendations."""

    reasons: list[str]
    gaps: list[dict[str, str]]
    confidence: float


class RecommendationBundle(BaseModel):
    """Pairing of a recommended role and its explanation payload."""

    role: CareerPath
    explanation: RecommendationExplanation


class RecommendationAPIResponse(BaseModel):
    """Envelope returned by the public recommendation endpoint."""

    user_id: str
    recommendations: list[RecommendationBundle]
    total: int
    profile_snapshot: CareerRecommendationRequest
