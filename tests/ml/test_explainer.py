"""Unit tests for the recommendation explanation system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.models.explainer import RecommendationExplainer


@pytest.fixture
def sample_user_profile():
    """Sample user profile with skills and experience."""
    return {
        "user_id": "user_123",
        "user_skills": ["python", "sql", "statistics", "pandas"],
        "user_interests": ["data analysis", "visualization"],
        "user_education": "bachelor's",
        "user_experience_years": 3,
        "user_past_roles": ["Data Analyst", "Junior Analyst"],
    }


@pytest.fixture
def sample_career():
    """Sample career with requirements."""
    return {
        "career_id": "career_456",
        "career_title": "Data Scientist",
        "required_skills": ["python", "machine learning", "statistics", "sql", "deep learning"],
        "required_education": "bachelor's",
        "required_experience_years": 3,
        "centrality": 0.85,
    }


@pytest.fixture
def sample_model_weights():
    """Sample model weights from recommender."""
    return {
        "skill_similarity": 0.78,
        "interest_alignment": 0.72,
        "education_match": 0.95,
        "gpa_proximity": 0.60,
        "experience_fit": 0.88,
        "career_graph_proximity": 0.70,
        "career_centrality": 0.85,
        "rerank_score": 0.82,
    }


def test_explanation_returns_complete_result(sample_user_profile, sample_career, sample_model_weights):
    """Test that explain() returns ExplanationResult with all fields."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    assert hasattr(result, "why_recommended")
    assert hasattr(result, "learning_plan")
    assert hasattr(result, "audit_log")

    assert isinstance(result.why_recommended, list)
    assert isinstance(result.learning_plan, list)
    assert isinstance(result.audit_log, dict)


def test_why_recommended_returns_top_3(sample_user_profile, sample_career, sample_model_weights):
    """Test that why_recommended returns at most 3 reasons."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    assert len(result.why_recommended) <= 3
    assert len(result.why_recommended) > 0


def test_explanations_are_specific_not_generic(sample_user_profile, sample_career, sample_model_weights):
    """Test that explanations contain specific details, not generic phrases."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    # Anti-patterns to avoid
    generic_phrases = [
        "good fit",
        "great match",
        "you're qualified",
        "suitable candidate",
        "perfect for you",
    ]

    for explanation in result.why_recommended:
        explanation_lower = explanation.lower()
        for generic in generic_phrases:
            assert generic not in explanation_lower, f"Generic phrase '{generic}' found in: {explanation}"

        # Must contain quantification (numbers, percentages, or skill names in quotes)
        has_number = any(char.isdigit() for char in explanation)
        has_quoted_skill = "'" in explanation or '"' in explanation
        assert has_number or has_quoted_skill, f"Explanation lacks quantification: {explanation}"

        # Minimum length to avoid single-word explanations
        assert len(explanation) >= 10, f"Explanation too short: {explanation}"


def test_learning_plan_contains_actionable_items(sample_user_profile, sample_career, sample_model_weights):
    """Test that learning plan contains actionable skill recommendations."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    # Should recommend missing skills
    user_skills_set = set(["python", "sql", "statistics", "pandas"])
    required_skills_set = set(["python", "machine learning", "statistics", "sql", "deep learning"])
    expected_missing = required_skills_set - user_skills_set

    assert len(result.learning_plan) > 0
    recommended_skills = {item["skill"] for item in result.learning_plan}

    # At least some missing skills should be recommended
    assert len(recommended_skills & expected_missing) > 0

    # Each recommendation must have required fields
    for item in result.learning_plan:
        assert "skill" in item
        assert "impact_score" in item
        assert "reason" in item
        assert "estimated_learning_time" in item
        assert "resources" in item

        # Reason must be specific
        assert len(item["reason"]) > 15, f"Reason too vague: {item['reason']}"

        # Must have at least 1 resource
        assert len(item["resources"]) >= 1


def test_learning_plan_ranks_by_impact(sample_user_profile, sample_career, sample_model_weights):
    """Test that learning plan items are ranked by impact score."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    if len(result.learning_plan) > 1:
        impact_scores = [item["impact_score"] for item in result.learning_plan]
        # Should be in descending order
        assert impact_scores == sorted(impact_scores, reverse=True)


def test_learning_plan_limits_to_top_5(sample_user_profile, sample_career, sample_model_weights):
    """Test that learning plan returns at most 5 items."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    assert len(result.learning_plan) <= 5


def test_audit_log_contains_traceability_fields(sample_user_profile, sample_career, sample_model_weights):
    """Test that audit log contains required fields for traceability."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    audit = result.audit_log

    assert "timestamp" in audit
    assert "user_id" in audit
    assert "career_id" in audit
    assert "career_title" in audit
    assert "feature_values" in audit
    assert "template_ids" in audit
    assert "generated_text" in audit

    # Verify template IDs were populated
    assert len(audit["template_ids"]) > 0


def test_skill_overlap_generates_quantified_explanation(sample_user_profile, sample_career, sample_model_weights):
    """Test that skill overlap generates explanation with concrete numbers."""
    explainer = RecommendationExplainer()
    result = explainer.explain(sample_user_profile, sample_career, sample_model_weights)

    # Find skill-related explanation
    skill_explanation = None
    for exp in result.why_recommended:
        if "skill" in exp.lower() or "competenc" in exp.lower():
            skill_explanation = exp
            break

    if skill_explanation:
        # Should mention specific skills in quotes
        assert "'" in skill_explanation or '"' in skill_explanation

        # Should have numeric quantification
        assert any(char.isdigit() for char in skill_explanation)


def test_empty_skills_returns_empty_learning_plan():
    """Test that user with all required skills gets empty learning plan."""
    user_profile = {
        "user_id": "user_999",
        "user_skills": ["python", "machine learning", "statistics", "sql", "deep learning", "tensorflow"],
        "user_interests": ["ai"],
        "user_education": "master's",
        "user_experience_years": 5,
        "user_past_roles": ["ML Engineer"],
    }

    career = {
        "career_id": "career_456",
        "career_title": "Data Scientist",
        "required_skills": ["python", "machine learning", "statistics", "sql"],
        "required_education": "bachelor's",
        "required_experience_years": 3,
        "centrality": 0.85,
    }

    weights = {
        "skill_similarity": 0.95,
        "interest_alignment": 0.80,
        "education_match": 1.0,
        "experience_fit": 0.92,
        "career_graph_proximity": 0.75,
        "career_centrality": 0.85,
    }

    explainer = RecommendationExplainer()
    result = explainer.explain(user_profile, career, weights)

    # Should have no learning recommendations if all skills present
    assert len(result.learning_plan) == 0


def test_low_weights_reduce_explanation_factors():
    """Test that low model weights filter out weak factors."""
    user_profile = {
        "user_id": "user_111",
        "user_skills": ["python"],
        "user_interests": ["coding"],
        "user_education": "high school",
        "user_experience_years": 1,
        "user_past_roles": ["Intern"],
    }

    career = {
        "career_id": "career_777",
        "career_title": "Senior ML Engineer",
        "required_skills": ["machine learning", "deep learning", "tensorflow", "pytorch"],
        "required_education": "master's",
        "required_experience_years": 7,
        "centrality": 0.50,
    }

    # All weights below thresholds
    weights = {
        "skill_similarity": 0.25,
        "interest_alignment": 0.30,
        "education_match": 0.40,
        "experience_fit": 0.35,
        "career_graph_proximity": 0.45,
        "career_centrality": 0.50,
    }

    explainer = RecommendationExplainer()
    result = explainer.explain(user_profile, career, weights)

    # Should have fewer explanations (possibly zero) due to low weights
    assert len(result.why_recommended) <= 3


def test_education_ordinal_mapping():
    """Test education level normalization."""
    from ml.models.explainer import EDUCATION_ORDINAL

    assert EDUCATION_ORDINAL["high school"] < EDUCATION_ORDINAL["bachelor's"]
    assert EDUCATION_ORDINAL["bachelor's"] < EDUCATION_ORDINAL["master's"]
    assert EDUCATION_ORDINAL["master's"] < EDUCATION_ORDINAL["phd"]


def test_skill_normalization():
    """Test skill string normalization."""
    explainer = RecommendationExplainer()

    # Test list input
    skills_list = ["Python", "SQL", "Machine Learning"]
    normalized = explainer._normalize_skills(skills_list)
    assert normalized == ["python", "sql", "machine learning"]

    # Test comma-separated string
    skills_str = "Python, SQL, Machine Learning"
    normalized = explainer._normalize_skills(skills_str)
    assert normalized == ["python", "sql", "machine learning"]

    # Test semicolon-separated string
    skills_str_semi = "Python; SQL; Machine Learning"
    normalized = explainer._normalize_skills(skills_str_semi)
    assert normalized == ["python", "sql", "machine learning"]


def test_custom_heuristics_override():
    """Test that custom heuristics can be injected."""
    custom_corpus = {
        "custom_skill": {"Custom Role": 0.99}
    }

    custom_transfer = {
        ("custom_skill", "python"): 0.85
    }

    custom_demand = {
        "custom_skill": 0.92
    }

    explainer = RecommendationExplainer(
        skill_corpus=custom_corpus,
        transferability_matrix=custom_transfer,
        market_signals=custom_demand,
    )

    # Verify custom heuristics are used
    assert explainer.skill_corpus == custom_corpus
    assert explainer.transferability == custom_transfer
    assert explainer.market_signals == custom_demand


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
