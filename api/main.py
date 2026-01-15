"""
FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import settings
from api.routes import recommendations, health
import logging

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="Machine Learning Recommendation Service"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(recommendations.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Recommendation Service",
        "docs": "/docs",
        "health": "/health"
    }

@app.on_event("startup")
async def startup():
    logger.info("🚀 ML Service starting...")
    logger.info(f"✅ Models loaded: {len(settings.MODELS_DIR)}")

@app.on_event("shutdown")
async def shutdown():
    logger.info("👋 ML Service shutting down...")
