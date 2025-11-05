"""Simple in-memory repository for storing user profiles."""

from __future__ import annotations

from threading import Lock

from app.models.profile import UserProfilePayload


class ProfileRepository:
    """Thread-safe repository abstraction for user profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfilePayload] = {}
        self._lock = Lock()

    def upsert(self, profile: UserProfilePayload) -> UserProfilePayload:
        """Insert or update a user profile."""
        with self._lock:
            self._profiles[profile.user_id] = profile
            return profile

    def get(self, user_id: str) -> UserProfilePayload | None:
        """Retrieve a stored profile."""
        with self._lock:
            profile = self._profiles.get(user_id)
            return profile.model_copy(deep=True) if profile else None


_profile_repo: ProfileRepository | None = None


def get_profile_repository() -> ProfileRepository:
    """FastAPI dependency hook for the profile repository."""
    global _profile_repo
    if _profile_repo is None:
        _profile_repo = ProfileRepository()
    return _profile_repo
