"""
API Configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os

class APISettings(BaseSettings):
    # Service
    SERVICE_NAME: str = "ML Recommendation API"
    SERVICE_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    DEBUG: bool = False
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    CACHE_TTL: int = 3600
    CACHE_ENABLED: bool = True
    
    # Models
    MODELS_DIR: str = "./trained_models"
    DEFAULT_MODEL: str = "als"
    DEFAULT_K: int = 10
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # ⭐ THÊM DÒNG NÀY - Cho phép extra fields từ .env
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # 🔥 QUAN TRỌNG: Ignore extra fields
    }

settings = APISettings()
