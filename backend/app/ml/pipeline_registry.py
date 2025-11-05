"""Registry for ML model artifacts consumed by the API layer."""

from pathlib import Path

from app.core.config import get_settings

settings = get_settings()

ARTIFACT_ROOT = Path("/ml/artifacts")


def get_latest_recommender_path() -> Path:
    """Return the expected path for the most recent recommender artifact."""
    return ARTIFACT_ROOT / "recommender" / "latest.joblib"
