"""Tests for RAG service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.rag_service import CareerRAGService, QueryType


@pytest.fixture
def rag_service(tmp_path: Path) -> CareerRAGService:
    """Create RAG service with test knowledge base."""
    # Create test knowledge base
    kb_path = tmp_path / "roles"
    kb_path.mkdir()
    
    # Create test role file
    (kb_path / "data-scientist.md").write_text("""# Data Scientist

## Overview
Extracts insights from data using statistical analysis.

## Required Skills
Python, SQL, Statistics, Machine Learning

## Transition Paths

### From Data Analyst (12-month plan)
**Months 1-3**: ML Foundations
- Complete ML course
- Build projects

**Key Skills**: Machine Learning, Deep Learning
**Time Commitment**: 15-20 hours/week
""", encoding="utf-8")
    
    # Create insights file
    insights_path = tmp_path / "insights"
    insights_path.mkdir()
    
    insights_data = {
        "insights": [
            {
                "category": "transition_success_rate",
                "data": {
                    "Data Analyst → Data Scientist": {
                        "observed_transitions": 523,
                        "avg_time_months": 14,
                        "success_rate": 0.72,
                        "common_skills_added": ["machine learning", "deep learning"]
                    }
                }
            }
        ]
    }
    
    (insights_path / "dataset_insights.json").write_text(json.dumps(insights_data))
    
    return CareerRAGService(
        kb_path=kb_path,
        insights_path=insights_path / "dataset_insights.json"
    )


def test_deflection_salary_question(rag_service: CareerRAGService) -> None:
    """Test that salary questions are deflected."""
    response = rag_service.answer_query(
        query="What is the average salary for a Data Scientist?",
        user_id="test_user",
    )
    
    assert response.deflection is True
    assert response.query_type == QueryType.OUT_OF_SCOPE
    assert "Glassdoor" in response.answer or "salary varies" in response.answer.lower()
    assert response.sources == ["deflection_rules"]


def test_deflection_company_hiring(rag_service: CareerRAGService) -> None:
    """Test that company hiring questions are deflected."""
    response = rag_service.answer_query(
        query="Is Google hiring Data Scientists?",
        user_id="test_user",
    )
    
    assert response.deflection is True
    assert response.query_type == QueryType.OUT_OF_SCOPE
    assert "careers page" in response.answer.lower() or "linkedin" in response.answer.lower()


def test_deflection_resume_review(rag_service: CareerRAGService) -> None:
    """Test that resume review requests are deflected."""
    response = rag_service.answer_query(
        query="Can you review my resume?",
        user_id="test_user",
    )
    
    assert response.deflection is True
    assert "resume" in response.answer.lower()


def test_classify_why_recommended(rag_service: CareerRAGService) -> None:
    """Test query classification for 'why recommended'."""
    query_type = rag_service._classify_query("Why was Data Scientist recommended for me?")
    assert query_type == QueryType.WHY_RECOMMENDED


def test_classify_transition_plan(rag_service: CareerRAGService) -> None:
    """Test query classification for transition plans."""
    query_type = rag_service._classify_query("How do I transition from Data Analyst to Data Scientist?")
    assert query_type == QueryType.TRANSITION_PLAN


def test_classify_skill_gaps(rag_service: CareerRAGService) -> None:
    """Test query classification for skill gaps."""
    query_type = rag_service._classify_query("What skills should I learn?")
    assert query_type == QueryType.SKILL_GAPS


def test_why_recommended_with_explanation(rag_service: CareerRAGService) -> None:
    """Test 'why recommended' query with explanation dict."""
    explanation = {
        "career_title": "Data Scientist",
        "confidence": 0.85,
        "reasons": [
            "Skill overlap: matched 3 of 5 core skills (python, sql, statistics).",
            "Profile similarity score: 0.78 (cosine).",
        ],
        "gaps": [
            {
                "skill": "Machine Learning",
                "reason": "Required for the role",
                "suggested_action": "Complete course"
            }
        ]
    }
    
    response = rag_service.answer_query(
        query="Why was Data Scientist recommended for me?",
        user_id="test_user",
        explanation_dict=explanation,
    )
    
    assert response.query_type == QueryType.WHY_RECOMMENDED
    assert response.deflection is False
    assert "Data Scientist" in response.answer
    assert "0.85" in response.answer
    assert "Machine Learning" in response.answer
    assert response.word_count <= 180


def test_why_recommended_without_explanation(rag_service: CareerRAGService) -> None:
    """Test 'why recommended' without explanation dict."""
    response = rag_service.answer_query(
        query="Why was Data Scientist recommended?",
        user_id="test_user",
    )
    
    assert response.query_type == QueryType.WHY_RECOMMENDED
    assert "don't have enough information" in response.answer.lower()


def test_transition_plan_with_context(rag_service: CareerRAGService) -> None:
    """Test transition plan query with full context."""
    response = rag_service.answer_query(
        query="How do I transition from Data Analyst to Data Scientist in 12 months?",
        user_id="test_user",
        user_profile={"current_role": "Data Analyst"},
    )
    
    assert response.query_type == QueryType.TRANSITION_PLAN
    assert response.deflection is False
    assert "Months 1-3" in response.answer or "ML" in response.answer
    assert response.word_count <= 180


def test_skill_gaps_response(rag_service: CareerRAGService) -> None:
    """Test skill gaps query."""
    explanation = {
        "gaps": [
            {
                "skill": "Machine Learning",
                "suggested_action": "Complete course within next quarter"
            },
            {
                "skill": "Deep Learning",
                "suggested_action": "Practice with projects"
            }
        ]
    }
    
    response = rag_service.answer_query(
        query="What skills should I learn?",
        user_id="test_user",
        explanation_dict=explanation,
    )
    
    assert response.query_type == QueryType.SKILL_GAPS
    assert "Machine Learning" in response.answer
    assert "Deep Learning" in response.answer


def test_data_insights_transition_stats(rag_service: CareerRAGService) -> None:
    """Test data insights query for transition stats."""
    response = rag_service.answer_query(
        query="How common is the transition from Data Analyst to Data Scientist?",
        user_id="test_user",
    )
    
    assert response.query_type == QueryType.DATA_INSIGHTS
    assert "523" in response.answer or "transitions" in response.answer.lower()


def test_response_word_limit_enforced(rag_service: CareerRAGService) -> None:
    """Test that responses are truncated to 180 words."""
    # Create long explanation
    long_explanation = {
        "career_title": "Data Scientist",
        "confidence": 0.85,
        "reasons": [
            "Skill overlap: " + " ".join(["matched"] * 50),
            "Profile similarity: " + " ".join(["high"] * 50),
            "Experience fit: " + " ".join(["good"] * 50),
        ],
        "gaps": []
    }
    
    response = rag_service.answer_query(
        query="Why was this recommended?",
        user_id="test_user",
        explanation_dict=long_explanation,
    )
    
    assert response.word_count <= 180


def test_extract_transition_roles(rag_service: CareerRAGService) -> None:
    """Test role extraction from transition queries."""
    source, target = rag_service._extract_transition_roles(
        "How do I transition from Data Analyst to Data Scientist?",
        None
    )
    
    assert source == "data analyst"
    assert target == "data scientist"


def test_extract_career_title(rag_service: CareerRAGService) -> None:
    """Test career title extraction from query."""
    title = rag_service._extract_career_title(
        "Why was Data Scientist recommended for me?",
        None
    )
    
    assert title == "data scientist"


def test_sources_tracking(rag_service: CareerRAGService) -> None:
    """Test that response tracks which sources were used."""
    explanation = {
        "career_title": "Data Scientist",
        "confidence": 0.85,
        "reasons": ["Skill match"],
        "gaps": []
    }
    
    response = rag_service.answer_query(
        query="Why was Data Scientist recommended?",
        user_id="test_user",
        explanation_dict=explanation,
    )
    
    assert "explanation_dict" in response.sources
    assert "role_knowledge_base" in response.sources


def test_general_query_fallback(rag_service: CareerRAGService) -> None:
    """Test fallback response for unclassified queries."""
    response = rag_service.answer_query(
        query="Hello, how are you?",
        user_id="test_user",
    )
    
    assert response.query_type == QueryType.GENERAL
    assert "can help with" in response.answer.lower()


def test_stable_output_for_same_input(rag_service: CareerRAGService) -> None:
    """Test that same input produces same output (deterministic)."""
    explanation = {
        "career_title": "Data Scientist",
        "confidence": 0.85,
        "reasons": ["Skill match"],
        "gaps": [{"skill": "ML", "suggested_action": "Learn"}]
    }
    
    response1 = rag_service.answer_query(
        query="Why was Data Scientist recommended?",
        user_id="test_user",
        explanation_dict=explanation,
    )
    
    response2 = rag_service.answer_query(
        query="Why was Data Scientist recommended?",
        user_id="test_user",
        explanation_dict=explanation,
    )
    
    assert response1.answer == response2.answer
    assert response1.query_type == response2.query_type
    assert response1.sources == response2.sources
