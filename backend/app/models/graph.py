"""Pydantic models representing the career trajectory graph responses."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrajectoryNeighbor(BaseModel):
    """Edge information describing a transition between two roles."""

    role: str
    direction: Literal["inbound", "outbound"]
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_time_months: float = Field(ge=0.0)
    observed_transitions: int = Field(ge=0)
    common_skills_added: list[str] = Field(default_factory=list)


class TrajectoryResponse(BaseModel):
    """Response returned for trajectory queries."""

    role: str
    centrality: float | None = Field(default=None, ge=0.0, le=1.0)
    neighbors: list[TrajectoryNeighbor]
    metadata: dict[str, Any] = Field(default_factory=dict)
