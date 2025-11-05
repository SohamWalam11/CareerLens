"""Deterministic templates for recommendation explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import math


_WEIGHT_TEMPLATES = {
    "skill_similarity": "Model weight for skill alignment scored {value:.2f}.",
    "interest_alignment": "Interest alignment contributed {value:.2f} to the match.",
    "career_graph_proximity": "Career-path proximity factor registered {value:.2f}.",
    "education_match": "Education fit factor scored {value:.2f}.",
    "experience_fit": "Experience alignment factor scored {value:.2f}.",
    "career_centrality": "Role demand factor scored {value:.2f}.",
    "rerank_score": "Re-ranker confidence was {value:.2f}.",
}


@dataclass(frozen=True)
class ExplanationPayload:
    """Structured explanation output."""

    reasons: list[str]
    gaps: list[dict[str, str]]
    confidence: float


def explain_recommendation(
    user_vec: Iterable[float],
    item_vec: Iterable[float],
    user_skills: Iterable[str],
    item_skills: Iterable[str],
    weights: Mapping[str, float],
) -> dict[str, object]:
    """Explain a recommendation using deterministic templates.

    Args:
        user_vec: Vector representing the user profile.
        item_vec: Vector representing the career/item profile.
        user_skills: Skills the user currently holds.
        item_skills: Skills expected for the recommended career.
        weights: Model weights or scores contributing to the match.

    Returns:
        Dictionary containing reasons (list[str]), gaps (list[dict]), and confidence (float 0-1).
    """

    user_vector = tuple(float(val) for val in user_vec)
    item_vector = tuple(float(val) for val in item_vec)

    similarity = _cosine_similarity(user_vector, item_vector)

    normalized_user, user_map = _normalize_skills(user_skills)
    normalized_item, item_map = _normalize_skills(item_skills)

    shared = sorted(set(normalized_user) & set(normalized_item))
    missing = [norm for norm in normalized_item if norm not in shared]

    reasons = _build_reasons(shared, item_map, normalized_item, similarity, weights)
    gaps = _build_gaps(missing, item_map)
    confidence = _calculate_confidence(similarity, weights)

    return {
        "reasons": reasons,
        "gaps": gaps,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Reason builders
# ---------------------------------------------------------------------------

def _build_reasons(
    shared: list[str],
    item_map: Mapping[str, str],
    ordered_item_norms: list[str],
    similarity: float,
    weights: Mapping[str, float],
) -> list[str]:
    reasons: list[str] = []

    matched_skills = [
        item_map[norm]
        for norm in ordered_item_norms
        if norm in shared
    ]
    if matched_skills:
        display_list = ", ".join(matched_skills[:3])
        reasons.append(
            f"Skill overlap: matched {len(matched_skills)} of {len(ordered_item_norms)} core skills ({display_list})."
        )
    else:
        reasons.append(
            "Skill overlap: no core skills matched yet; prioritise foundational alignment."
        )

    reasons.append(f"Profile similarity score: {similarity:.2f} (cosine).")

    if weights:
        key, value = max(weights.items(), key=lambda item: item[1])
        template = _WEIGHT_TEMPLATES.get(key, "Model factor '{name}' contributed {value:.2f}.")
        reasons.append(template.format(name=key.replace("_", " "), value=value))

    return reasons


# ---------------------------------------------------------------------------
# Gap builders
# ---------------------------------------------------------------------------

def _build_gaps(missing_norms: list[str], item_map: Mapping[str, str]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for norm in missing_norms:
        display = item_map[norm]
        gaps.append(
            {
                "skill": display,
                "reason": f"{display} is required for the role and is not yet in the profile.",
                "suggested_action": "Schedule focused practice within the next quarter.",
            }
        )
    return gaps


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def _calculate_confidence(similarity: float, weights: Mapping[str, float]) -> float:
    if not weights:
        return round(max(0.0, min(1.0, similarity)), 2)

    average_weight = sum(weights.values()) / len(weights)
    blended = 0.5 * similarity + 0.5 * average_weight
    return round(max(0.0, min(1.0, blended)), 2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_skills(skills: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    ordered_norms: list[str] = []
    mapping: dict[str, str] = {}
    for skill in skills:
        cleaned = " ".join(skill.strip().split())
        if not cleaned:
            continue
        norm = cleaned.lower()
        if norm not in mapping:
            mapping[norm] = cleaned
        ordered_norms.append(norm)
    return ordered_norms, mapping


def _cosine_similarity(user_vector: tuple[float, ...], item_vector: tuple[float, ...]) -> float:
    if not user_vector or not item_vector:
        return 0.0

    if len(user_vector) != len(item_vector):
        raise ValueError("Vectors must share the same dimensionality for cosine similarity computation.")

    dot = sum(u * i for u, i in zip(user_vector, item_vector))
    user_norm = math.sqrt(sum(u * u for u in user_vector))
    item_norm = math.sqrt(sum(i * i for i in item_vector))

    if user_norm == 0 or item_norm == 0:
        return 0.0

    similarity = dot / (user_norm * item_norm)
    return max(0.0, min(1.0, similarity))


__all__ = ["explain_recommendation", "ExplanationPayload"]
