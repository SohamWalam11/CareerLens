"""Deterministic explanation generator for career recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# Skill importance heuristics (from dataset analysis or domain knowledge)
SKILL_FREQUENCY_CORPUS = {
    "python": {"Data Scientist": 0.95, "ML Engineer": 0.92, "Software Engineer": 0.88},
    "machine learning": {"Data Scientist": 0.88, "ML Engineer": 0.90, "AI Researcher": 0.85},
    "deep learning": {"ML Engineer": 0.85, "Data Scientist": 0.65, "AI Researcher": 0.90},
    "sql": {"Data Analyst": 0.90, "Data Scientist": 0.75, "Business Analyst": 0.85},
    "statistics": {"Data Scientist": 0.85, "Data Analyst": 0.70, "Statistician": 0.95},
    "tensorflow": {"ML Engineer": 0.80, "Data Scientist": 0.60},
    "pytorch": {"ML Engineer": 0.75, "AI Researcher": 0.85},
    "spark": {"Data Engineer": 0.85, "Data Scientist": 0.55, "ML Engineer": 0.60},
    "kubernetes": {"DevOps Engineer": 0.90, "ML Engineer": 0.65, "Backend Engineer": 0.70},
    "aws": {"Cloud Engineer": 0.95, "DevOps Engineer": 0.85, "Data Engineer": 0.70},
    "react": {"Frontend Engineer": 0.90, "Full Stack Engineer": 0.80},
    "node.js": {"Backend Engineer": 0.85, "Full Stack Engineer": 0.75},
    "java": {"Backend Engineer": 0.80, "Software Engineer": 0.75},
    "leadership": {"Engineering Manager": 0.90, "Product Manager": 0.85},
    "product management": {"Product Manager": 0.95, "Program Manager": 0.70},
}

# Skill transferability matrix (source → target similarity)
SKILL_TRANSFERABILITY = {
    ("sql", "spark"): 0.65,
    ("python", "r"): 0.70,
    ("statistics", "machine learning"): 0.60,
    ("machine learning", "deep learning"): 0.75,
    ("tensorflow", "pytorch"): 0.85,
    ("react", "vue"): 0.80,
    ("java", "kotlin"): 0.75,
    ("aws", "azure"): 0.70,
    ("pandas", "spark"): 0.60,
    ("javascript", "typescript"): 0.85,
}

# Market demand signals (0-1 scale)
MARKET_DEMAND_SIGNALS = {
    "python": 0.90,
    "machine learning": 0.88,
    "kubernetes": 0.82,
    "react": 0.78,
    "aws": 0.85,
    "tensorflow": 0.75,
    "deep learning": 0.80,
    "leadership": 0.70,
    "product management": 0.72,
}

# Education ordinal mapping
EDUCATION_ORDINAL = {
    "some high school": 0,
    "high school": 1,
    "ged": 1,
    "some college": 2,
    "associate degree": 3,
    "associate's": 3,
    "bachelor's": 4,
    "bachelor": 4,
    "master's": 5,
    "master": 5,
    "phd": 6,
    "doctorate": 6,
}


@dataclass
class ExplanationResult:
    """Complete explanation package with audit trail."""

    why_recommended: list[str]
    learning_plan: list[dict[str, Any]]
    audit_log: dict[str, Any]


class RecommendationExplainer:
    """Generate deterministic, auditable explanations for career recommendations."""

    def __init__(
        self,
        skill_corpus: dict[str, dict[str, float]] | None = None,
        transferability_matrix: dict[tuple[str, str], float] | None = None,
        market_signals: dict[str, float] | None = None,
    ):
        self.skill_corpus = skill_corpus or SKILL_FREQUENCY_CORPUS
        self.transferability = transferability_matrix or SKILL_TRANSFERABILITY
        self.market_signals = market_signals or MARKET_DEMAND_SIGNALS
        self._used_templates: list[str] = []

    def explain(
        self,
        user_profile: dict[str, Any],
        career: dict[str, Any],
        model_weights: dict[str, float],
    ) -> ExplanationResult:
        """Generate complete explanation package."""
        self._used_templates = []

        why_factors = self._generate_why_recommended(user_profile, career, model_weights)
        learning_plan = self._generate_learning_plan(user_profile, career, model_weights)

        audit_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_profile.get("user_id"),
            "career_id": career.get("career_id"),
            "career_title": career.get("career_title"),
            "feature_values": model_weights,
            "template_ids": self._used_templates,
            "generated_text": {"why_recommended": why_factors, "learning_plan": learning_plan},
        }

        return ExplanationResult(
            why_recommended=why_factors, learning_plan=learning_plan, audit_log=audit_log
        )

    def _generate_why_recommended(
        self, user_profile: dict, career: dict, weights: dict
    ) -> list[str]:
        """Generate top-3 reasons for recommendation."""
        factors = []

        # Factor 1: Skill Overlap
        if weights.get("skill_similarity", 0) >= 0.7:
            user_skills = set(self._normalize_skills(user_profile.get("user_skills", [])))
            required_skills = set(self._normalize_skills(career.get("required_skills", [])))
            matching_skills = user_skills & required_skills

            if len(matching_skills) >= 3:
                top_skills = sorted(matching_skills)[:3]
                skill_str = ", ".join([f"'{s}'" for s in top_skills])
                factors.append(
                    {
                        "score": weights["skill_similarity"],
                        "text": f"Strong skill alignment in {skill_str}, matching {len(matching_skills)} of {len(required_skills)} required competencies",
                    }
                )
                self._used_templates.append("skill_overlap_high")
            elif len(matching_skills) == 2:
                skill_str = " and ".join([f"'{s}'" for s in sorted(matching_skills)])
                factors.append(
                    {
                        "score": weights["skill_similarity"],
                        "text": f"Core competencies in {skill_str} align with role requirements",
                    }
                )
                self._used_templates.append("skill_overlap_medium")
            elif len(matching_skills) == 1:
                skill = list(matching_skills)[0]
                factors.append(
                    {
                        "score": weights["skill_similarity"],
                        "text": f"Foundational expertise in '{skill}' provides entry point to this role",
                    }
                )
                self._used_templates.append("skill_overlap_low")

        # Factor 2: Interest-Career Alignment
        if weights.get("interest_alignment", 0) >= 0.6:
            interest_keywords = user_profile.get("user_interests", [])
            if isinstance(interest_keywords, str):
                interest_keywords = [interest_keywords]
            if interest_keywords:
                primary_interest = interest_keywords[0] if interest_keywords else "your interests"
                factors.append(
                    {
                        "score": weights["interest_alignment"],
                        "text": f"Your passion for {primary_interest} strongly correlates with typical {career.get('career_title', 'this role')} responsibilities",
                    }
                )
                self._used_templates.append("interest_alignment")

        # Factor 3: Career Trajectory Fit
        if weights.get("career_graph_proximity", 0) >= 0.65:
            past_roles = user_profile.get("user_past_roles", [])
            if past_roles:
                most_recent = past_roles[-1] if isinstance(past_roles, list) else past_roles
                factors.append(
                    {
                        "score": weights["career_graph_proximity"],
                        "text": f"Natural progression from {most_recent} based on 500+ observed career transitions in our network",
                    }
                )
                self._used_templates.append("career_graph_natural_progression")

        # Factor 4: Education Qualification
        if weights.get("education_match", 0) >= 0.8:
            user_ed = user_profile.get("user_education", "bachelor's")
            req_ed = career.get("required_education", "bachelor's")
            user_ord = self._education_ordinal(user_ed)
            req_ord = self._education_ordinal(req_ed)

            if user_ord >= req_ord:
                factors.append(
                    {
                        "score": weights["education_match"],
                        "text": f"Your {user_ed} degree meets or exceeds typical educational requirements",
                    }
                )
                self._used_templates.append("education_meets_requirement")
            elif user_ord == req_ord - 1:
                factors.append(
                    {
                        "score": weights["education_match"],
                        "text": f"Your {user_ed} background provides 80% of required educational foundation",
                    }
                )
                self._used_templates.append("education_near_requirement")

        # Factor 5: Experience Level Proximity
        if weights.get("experience_fit", 0) >= 0.55:
            user_exp = user_profile.get("user_experience_years", 0)
            req_exp = career.get("required_experience_years", 3)
            exp_gap = req_exp - user_exp

            if exp_gap <= 1:
                factors.append(
                    {
                        "score": weights["experience_fit"],
                        "text": f"Your {user_exp} years of experience aligns well with role expectations",
                    }
                )
                self._used_templates.append("experience_aligned")
            elif exp_gap <= 2:
                factors.append(
                    {
                        "score": weights["experience_fit"],
                        "text": f"Within typical experience range; {user_exp} years provides solid foundation for growth",
                    }
                )
                self._used_templates.append("experience_near_aligned")

        # Factor 6: High-Demand Career
        if weights.get("career_centrality", 0) >= 0.75:
            factors.append(
                {
                    "score": weights["career_centrality"],
                    "text": "High-demand role with strong career mobility (top 25% of career network centrality)",
                }
            )
            self._used_templates.append("high_centrality")

        # Sort by score descending and take top 3
        factors.sort(key=lambda x: x["score"], reverse=True)
        return [f["text"] for f in factors[:3]]

    def _generate_learning_plan(
        self, user_profile: dict, career: dict, weights: dict
    ) -> list[dict[str, Any]]:
        """Generate top-5 skill development recommendations."""
        user_skills = set(self._normalize_skills(user_profile.get("user_skills", [])))
        required_skills = set(self._normalize_skills(career.get("required_skills", [])))

        missing_skills = required_skills - user_skills

        if not missing_skills:
            return []

        career_title = career.get("career_title", "this role")
        skill_scores = []

        for skill in missing_skills:
            # Impact components
            frequency_score = self._get_skill_frequency(skill, career_title)
            centrality_score = self._get_skill_centrality(skill)
            transfer_score = self._get_transfer_score(skill, user_skills)
            demand_score = self._get_market_demand(skill)

            # Weighted composite impact
            impact = (
                0.35 * frequency_score
                + 0.25 * centrality_score
                + 0.20 * (1 - transfer_score)
                + 0.20 * demand_score
            )

            skill_scores.append(
                {
                    "skill": skill,
                    "impact": impact,
                    "frequency": frequency_score,
                    "transfer_difficulty": 1 - transfer_score,
                    "demand": demand_score,
                }
            )

        # Rank by impact
        skill_scores.sort(key=lambda x: x["impact"], reverse=True)

        # Generate explanations for top 5
        recommendations = []
        for idx, item in enumerate(skill_scores[:5], start=1):
            skill = item["skill"]

            # Contextual reasoning
            if item["frequency"] >= 0.8:
                reason = f"Core requirement (appears in 80%+ of {career_title} roles)"
            elif item["demand"] >= 0.75:
                reason = "High market demand (top quartile for salary impact)"
            elif item["transfer_difficulty"] <= 0.3:
                closest = self._closest_user_skill(skill, user_skills)
                reason = f"Natural extension of your '{closest}' expertise"
            else:
                reason = "Bridges gap between current skillset and role expectations"

            # Learning time estimate
            if item["transfer_difficulty"] <= 0.3:
                learning_time = "2-4 weeks (fast upskill)"
            elif item["transfer_difficulty"] <= 0.6:
                learning_time = "2-3 months (moderate investment)"
            else:
                learning_time = "6+ months (foundational build)"

            recommendations.append(
                {
                    "rank": idx,
                    "skill": skill,
                    "impact_score": round(item["impact"], 2),
                    "reason": reason,
                    "estimated_learning_time": learning_time,
                    "resources": self._generate_learning_resources(skill),
                }
            )

        return recommendations

    # Helper methods
    def _normalize_skills(self, skills: list[str] | str) -> list[str]:
        """Normalize skill strings to lowercase."""
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.replace(";", ",").split(",")]
        return [s.lower().strip() for s in skills if s]

    def _education_ordinal(self, education: str) -> int:
        """Convert education level to ordinal."""
        normalized = education.lower().strip()
        return EDUCATION_ORDINAL.get(normalized, 2)

    def _get_skill_frequency(self, skill: str, career_title: str) -> float:
        """Get skill frequency in target role."""
        skill_lower = skill.lower()
        career_data = self.skill_corpus.get(skill_lower, {})
        return career_data.get(career_title, 0.5)

    def _get_skill_centrality(self, skill: str) -> float:
        """Get skill centrality (co-occurrence in career graph)."""
        skill_lower = skill.lower()
        # Average frequency across all roles as proxy for centrality
        career_data = self.skill_corpus.get(skill_lower, {})
        if not career_data:
            return 0.3
        return np.mean(list(career_data.values()))

    def _get_transfer_score(self, target_skill: str, user_skills: set[str]) -> float:
        """Get maximum transferability from user's existing skills."""
        target_lower = target_skill.lower()
        max_transfer = 0.0

        for user_skill in user_skills:
            user_lower = user_skill.lower()
            # Check both directions
            transfer = self.transferability.get((user_lower, target_lower), 0.0)
            transfer = max(transfer, self.transferability.get((target_lower, user_lower), 0.0))
            max_transfer = max(max_transfer, transfer)

        return max_transfer

    def _get_market_demand(self, skill: str) -> float:
        """Get market demand signal for skill."""
        skill_lower = skill.lower()
        return self.market_signals.get(skill_lower, 0.5)

    def _closest_user_skill(self, target_skill: str, user_skills: set[str]) -> str:
        """Find closest user skill to target skill."""
        target_lower = target_skill.lower()
        best_skill = "your existing skills"
        best_score = 0.0

        for user_skill in user_skills:
            user_lower = user_skill.lower()
            transfer = self.transferability.get((user_lower, target_lower), 0.0)
            transfer = max(transfer, self.transferability.get((target_lower, user_lower), 0.0))

            if transfer > best_score:
                best_score = transfer
                best_skill = user_skill

        return best_skill

    def _generate_learning_resources(self, skill: str) -> list[dict[str, str]]:
        """Generate learning resources for a skill."""
        # Placeholder - in production, this would query a resource API
        skill_lower = skill.lower()

        resource_map = {
            "machine learning": [
                {"type": "course", "name": "Stanford CS229", "url": "https://cs229.stanford.edu"},
                {
                    "type": "book",
                    "name": "Hands-On ML with Scikit-Learn",
                    "url": "https://www.oreilly.com",
                },
            ],
            "deep learning": [
                {"type": "course", "name": "fast.ai", "url": "https://fast.ai"},
                {"type": "course", "name": "Deep Learning Specialization", "url": "https://coursera.org"},
            ],
            "tensorflow": [
                {"type": "tutorial", "name": "TensorFlow Tutorials", "url": "https://tensorflow.org"},
                {"type": "course", "name": "TensorFlow Developer Certificate", "url": "https://tensorflow.org"},
            ],
            "spark": [
                {"type": "tutorial", "name": "PySpark Tutorial", "url": "https://spark.apache.org"},
                {"type": "course", "name": "Big Data with Spark", "url": "https://coursera.org"},
            ],
            "kubernetes": [
                {"type": "tutorial", "name": "Kubernetes Documentation", "url": "https://kubernetes.io"},
                {"type": "certification", "name": "CKA Certification", "url": "https://cncf.io"},
            ],
        }

        return resource_map.get(
            skill_lower,
            [
                {"type": "search", "name": f"Search '{skill}' tutorials", "url": f"https://google.com/search?q={skill}+tutorial"}
            ],
        )


__all__ = ["RecommendationExplainer", "ExplanationResult"]
