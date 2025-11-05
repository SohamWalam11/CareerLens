"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def get_health() -> dict[str, str]:
    """Basic liveness probe for infrastructure checks."""
    return {"status": "ok"}
