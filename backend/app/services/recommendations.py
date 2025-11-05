"""Domain services for generating career recommendations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from app.models.profile import (
    CareerPath,
    CareerRecommendationRequest,
    CareerRecommendationResponse,
    LearningResource,
    RecommendationBundle,
    RecommendationExplanation,
    SkillGap,
)
from app.services.explanations import explain_recommendation


@dataclass
class RecommendationService:
    """Generate recommendations using simple heuristics until ML pipeline is ready."""

    default_roles: tuple[str, ...] = (
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "Software Engineer",
        "Product Manager",
        "UX Designer",
    )

    role_skill_library: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "Data Scientist": (
                "Python",
                "Machine Learning",
                "Statistics",
                "Data Visualization",
                "SQL",
            ),
            "Machine Learning Engineer": (
                "Python",
                "TensorFlow",
                "PyTorch",
                "ML Ops",
                "Kubernetes",
            ),
            "Data Analyst": (
                "SQL",
                "Dashboarding",
                "Excel",
                "Data Cleaning",
                "Storytelling",
            ),
            "Software Engineer": (
                "Algorithms",
                "Data Structures",
                "System Design",
                "Python",
                "APIs",
            ),
            "Product Manager": (
                "User Research",
                "Roadmapping",
                "Stakeholder Management",
                "Analytics",
                "Prioritization",
            ),
            "UX Designer": (
                "Wireframing",
                "Design Systems",
                "User Research",
                "Prototyping",
                "Accessibility",
            ),
        }
    )
    def generate_recommendations(
        self, request: CareerRecommendationRequest
    ) -> CareerRecommendationResponse:
        """Return placeholder recommendations derived from user interests and skills."""
        ranked_paths = self._rank_roles(request)
        skill_gaps = self._identify_skill_gaps(request)
        learning_plan = self._build_learning_plan(skill_gaps)

        summary = {
            "name": request.name,
            "interests": request.interests,
            "skills": request.skills,
            "goals": request.goals,
        }

        return CareerRecommendationResponse(
            user_summary=summary,
            recommendations=ranked_paths,
            skill_gaps=skill_gaps,
            learning_plan=learning_plan,
        )

    def generate_detailed_recommendations(
        self, request: CareerRecommendationRequest, limit: int = 5
    ) -> list[RecommendationBundle]:
        """Return enriched recommendation bundles with deterministic explanations."""

        base_response = self.generate_recommendations(request)
        user_vector = self._vectorize_skills(request.skills)

        bundles: list[RecommendationBundle] = []
        for role_path in base_response.recommendations[:limit]:
            role_skills = self.role_skill_library.get(
                role_path.title,
                ("Communication", "Project Management", "Continuous Learning"),
            )
            role_vector = self._vectorize_skills(role_skills)
            weights = self._compute_match_weights(request, role_path, role_skills)
            explanation_payload = explain_recommendation(
                user_vector,
                role_vector,
                request.skills,
                role_skills,
                weights,
            )
            bundles.append(
                RecommendationBundle(
                    role=role_path,
                    explanation=RecommendationExplanation(**explanation_payload),
                )
            )

        return bundles

    def _rank_roles(self, request: CareerRecommendationRequest) -> list[CareerPath]:
        """Generate role recommendations with naive scoring for now."""
        ranked: list[CareerPath] = []
        for idx, role in enumerate(self.default_roles):
            overlap = len({skill.lower() for skill in request.skills} & {role.lower()})
            fit_score = max(0.25, min(0.95, 0.4 + 0.1 * overlap + 0.05 * idx))
            ranked.append(
                CareerPath(
                    title=role,
                    fit_score=round(fit_score, 2),
                    description=f"Candidate trajectory suggestion for {role}.",
                    next_steps=["Complete tailored learning plan", "Schedule mentor session"],
                    trajectory=["Junior", "Mid-Level", "Senior"],
                )
            )
        return ranked

    def _identify_skill_gaps(self, request: CareerRecommendationRequest) -> list[SkillGap]:
        """Return placeholder skill gaps until the ML skill comparison is implemented."""
        required = {"python", "data analysis", "communication"}
        current = {skill.lower() for skill in request.skills}
        pending = required - current
        gaps = [
            SkillGap(skill=skill.title(), current_level=0.4, target_level=0.8)
            for skill in sorted(pending)
        ]
        return gaps

    def _build_learning_plan(self, gaps: Iterable[SkillGap]) -> list[LearningResource]:
        """Map skill gaps to curated learning resources."""
        plan: list[LearningResource] = []
        for gap in gaps:
            plan.append(
                LearningResource(
                    title=f"{gap.skill} Fundamentals",
                    provider="CareerLens Academy",
                    url="https://example.com",
                    estimated_hours=10,
                )
            )
        return plan

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    def _vectorize_skills(self, skills: Iterable[str], size: int = 6) -> list[float]:
        """Create deterministic pseudo-embeddings from skill names."""
        vector = [0.0] * size
        cleaned_skills = [skill.strip().lower() for skill in skills if skill.strip()]
        if not cleaned_skills:
            return vector

        for idx, skill in enumerate(sorted(set(cleaned_skills))):
            bucket = idx % size
            weight = 0.6 + 0.4 * self._hash_to_unit(skill)
            vector[bucket] += weight

        total = sum(vector)
        if total > 0:
            vector = [val / total for val in vector]
        return vector

    def _compute_match_weights(
        self,
        request: CareerRecommendationRequest,
        role_path: CareerPath,
        role_skills: Iterable[str],
    ) -> dict[str, float]:
        """Derive lightweight weights feeding the explanation engine."""
        user_skills = {skill.strip().lower() for skill in request.skills if skill.strip()}
        target_skills = {skill.strip().lower() for skill in role_skills}

        overlap_ratio = 0.0
        if target_skills:
            overlap_ratio = len(user_skills & target_skills) / len(target_skills)

        interest_score = min(1.0, 0.2 + 0.15 * len(request.interests))
        experience_score = min(1.0, 0.3 + 0.1 * len(request.goals))
        rerank_score = role_path.fit_score

        return {
            "skill_similarity": round(overlap_ratio, 2),
            "interest_alignment": round(interest_score, 2),
            "experience_fit": round(experience_score, 2),
            "rerank_score": round(rerank_score, 2),
        }

    @staticmethod
    def _hash_to_unit(value: str) -> float:
        """Hash a string into the range [0, 1]."""
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16)
        return bucket / 0xFFFFFFFF


def get_recommendation_service() -> RecommendationService:
    """FastAPI dependency provider for the recommendation service."""
    return RecommendationService()
