"""Application configuration and settings management."""

from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the CareerLens backend."""

    project_name: str = "CareerLens API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg2://postgres:postgres@db:5432/careerlens"
    alembic_database_url: str | None = None

    cors_origins: list[str] = ["http://localhost:5173"]

    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None
    pinecone_dimension: int = 768
    pinecone_max_retries: int = 3
    pinecone_timeout_seconds: int = 10
    pinecone_rate_limit_per_minute: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def sqlalchemy_database_uri(self) -> str:
        """Return the SQLAlchemy connection string used by the ORM."""
        return self.alembic_database_url or self.database_url

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Support comma-separated origins via environment variables."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor to avoid redundant environment parsing."""
    return Settings()
