"""Integration tests for primary CareerLens API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.feedback_store import get_feedback_repository
from app.services.profile_store import get_profile_repository
from main import app

client = TestClient(app)


def _reset_state() -> None:
    profile_repo = get_profile_repository()
    feedback_repo = get_feedback_repository()
    profile_repo._profiles.clear()  # type: ignore[attr-defined]
    feedback_repo._feedback.clear()  # type: ignore[attr-defined]


def test_profile_upsert() -> None:
    _reset_state()
    payload = {
        "user_id": "user-123",
        "name": "Avery",
        "age": 28,
        "education_level": "Bachelors",
        "interests": ["data", "design"],
        "skills": ["python", "sql"],
        "goals": ["lead projects"],
    }

    response = client.post("/api/v1/profile", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["user_profile"]["user_id"] == "user-123"
    assert data["user_profile"]["skills"] == ["python", "sql"]


def test_recommendations_with_stored_profile() -> None:
    _reset_state()
    profile_payload = {
        "user_id": "user-456",
        "name": "Jordan",
        "age": 30,
        "education_level": "Masters",
        "interests": ["analytics"],
        "skills": ["python", "statistics"],
        "goals": ["become data scientist"],
    }
    client.post("/api/v1/profile", json=profile_payload)

    response = client.post("/api/v1/recommend", json={"user_id": "user-456"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-456"
    assert data["total"] == 5
    assert len(data["recommendations"]) == 5
    first_bundle = data["recommendations"][0]
    assert "explanation" in first_bundle
    assert first_bundle["explanation"]["reasons"]


def test_recommendations_inline_profile_stores_when_user_id_present() -> None:
    _reset_state()
    payload = {
        "user_id": "inline-1",
        "profile": {
            "name": "Casey",
            "age": 32,
            "education_level": "Bachelors",
            "interests": ["ml"],
            "skills": ["python"],
            "goals": ["transition to ml"],
        },
    }

    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200

    stored = get_profile_repository().get("inline-1")
    assert stored is not None
    assert stored.skills == ["python"]


def test_trajectory_endpoint() -> None:
    _reset_state()
    response = client.get("/api/v1/trajectory", params={"role": "Data Scientist"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "Data Scientist"
    assert data["neighbors"]


def test_feedback_submission() -> None:
    _reset_state()
    payload = {
        "user_id": "user-789",
        "role": "Data Scientist",
        "rating": 5,
        "relevant": True,
        "comments": "Great match",
    }

    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["entry"]["role"] == "Data Scientist"
    assert data["entry"]["rating"] == 5
