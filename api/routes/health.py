"""
Health check endpoint
"""

from fastapi import APIRouter
from api.schemas.models import HealthResponse
from api.services.model_loader import model_loader
from api.services.cache import cache
from api.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "modelsLoaded": model_loader.get_available_models(),
        "cacheEnabled": cache.enabled
    }
