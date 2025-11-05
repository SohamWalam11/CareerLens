"""CareerLens FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.project_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Landing endpoint for quick smoke tests."""
    return {"message": "CareerLens API is running"}
