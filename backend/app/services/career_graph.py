"""Lightweight career trajectory lookup service using dataset insights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.models.graph import TrajectoryNeighbor, TrajectoryResponse


@dataclass
class CareerGraphService:
    """Serve trajectory neighbors derived from structured insights."""

    insights_path: Path | None = None
    _insights_cache: dict | None = field(default=None, init=False, repr=False)

    def _load_insights(self) -> dict:
        if self._insights_cache is not None:
            return self._insights_cache

        base_path = self.insights_path or Path(__file__).resolve().parents[2] / "knowledge_base" / "insights" / "dataset_insights.json"

        if not base_path.exists():
            msg = f"Insights dataset is not available at {base_path}"
            raise FileNotFoundError(msg)

        with base_path.open("r", encoding="utf-8") as fp:
            self._insights_cache = json.load(fp)
        return self._insights_cache

    def get_trajectory(self, role: str) -> TrajectoryResponse:
        insights = self._load_insights()
        role_lower = role.lower().strip()

        transitions = self._extract_transition_data(insights)
        neighbors: list[TrajectoryNeighbor] = []

        for key, stats in transitions.items():
            source, target = [part.strip() for part in key.split("→", maxsplit=1)]
            if source.lower() == role_lower:
                neighbors.append(self._build_neighbor(target, "outbound", stats))
            elif target.lower() == role_lower:
                neighbors.append(self._build_neighbor(source, "inbound", stats))

        if not neighbors:
            msg = f"No transitions recorded for role '{role}'"
            raise LookupError(msg)

        centrality = self._lookup_centrality(insights, role_lower)

        metadata = {
            "dataset_version": insights.get("dataset_version"),
            "total_records": insights.get("total_records"),
        }

        return TrajectoryResponse(role=role, centrality=centrality, neighbors=neighbors, metadata=metadata)

    def _extract_transition_data(self, insights: dict) -> dict[str, dict]:
        for block in insights.get("insights", []):
            if block.get("category") == "transition_success_rate":
                return block.get("data", {})
        return {}

    def _lookup_centrality(self, insights: dict, role_lower: str) -> float | None:
        for block in insights.get("insights", []):
            if block.get("category") == "role_demand":
                for role_name, data in block.get("data", {}).items():
                    if role_name.lower() == role_lower:
                        return float(data.get("centrality", 0.0))
        return None

    @staticmethod
    def _build_neighbor(role: str, direction: str, stats: dict) -> TrajectoryNeighbor:
        return TrajectoryNeighbor(
            role=role,
            direction=direction,  # type: ignore[arg-type]
            success_rate=float(stats.get("success_rate", 0.0)),
            avg_time_months=float(stats.get("avg_time_months", 0.0)),
            observed_transitions=int(stats.get("observed_transitions", 0)),
            common_skills_added=list(stats.get("common_skills_added", [])),
        )


_career_graph_service: CareerGraphService | None = None


def get_career_graph_service() -> CareerGraphService:
    """FastAPI dependency provider for the trajectory service."""
    global _career_graph_service
    if _career_graph_service is None:
        _career_graph_service = CareerGraphService()
    return _career_graph_service
