"""Tests for deterministic recommendation explanations."""

from __future__ import annotations

from app.services.explanations import explain_recommendation


def test_explain_recommendation_generates_expected_reasons_and_confidence() -> None:
    user_vec = [0.5, 0.1, 0.4]
    item_vec = [0.4, 0.2, 0.3]
    user_skills = ["Python", "SQL", "Statistics"]
    item_skills = ["Python", "SQL", "Machine Learning"]
    weights = {
        "skill_similarity": 0.80,
        "interest_alignment": 0.60,
        "rerank_score": 0.75,
    }

    explanation = explain_recommendation(user_vec, item_vec, user_skills, item_skills, weights)

    assert explanation["confidence"] == 0.85
    assert explanation["reasons"] == [
        "Skill overlap: matched 2 of 3 core skills (Python, SQL).",
        "Profile similarity score: 0.97 (cosine).",
        "Model weight for skill alignment scored 0.80.",
    ]
    assert explanation["gaps"] == [
        {
            "skill": "Machine Learning",
            "reason": "Machine Learning is required for the role and is not yet in the profile.",
            "suggested_action": "Schedule focused practice within the next quarter.",
        }
    ]


def test_explain_recommendation_handles_no_skill_overlap() -> None:
    user_vec = [0.0, 1.0]
    item_vec = [1.0, 0.0]
    user_skills = ["Design", "Presentation"]
    item_skills = ["Python", "SQL"]
    weights = {}

    explanation = explain_recommendation(user_vec, item_vec, user_skills, item_skills, weights)

    assert explanation["confidence"] == 0.0
    assert explanation["reasons"] == [
        "Skill overlap: no core skills matched yet; prioritise foundational alignment.",
        "Profile similarity score: 0.00 (cosine).",
    ]
    assert explanation["gaps"] == [
        {
            "skill": "Python",
            "reason": "Python is required for the role and is not yet in the profile.",
            "suggested_action": "Schedule focused practice within the next quarter.",
        },
        {
            "skill": "SQL",
            "reason": "SQL is required for the role and is not yet in the profile.",
            "suggested_action": "Schedule focused practice within the next quarter.",
        },
    ]
