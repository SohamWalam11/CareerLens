"""Root API router for CareerLens."""

from fastapi import APIRouter

from app.api.routes import analytics, careers, health, recommendations

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(recommendations.router)
api_router.include_router(careers.router)
api_router.include_router(analytics.router)
