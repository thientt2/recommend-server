"""
Recommendation endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from api.schemas.models import (
    RecommendationRequest,
    RecommendationResponse,
    SimilarProduct
)
from api.services.recommender import recommender
from api.services.model_loader import model_loader
from api.services.cache import cache
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    model: str = Query('als', description="Model to use (default: als)"),
    exclude_seen: bool = Query(True, description="Exclude seen products")
):
    """Get top-K recommendations for a user using ALS model"""
    
    # Validate model
    available_models = model_loader.get_available_models()
    if available_models and model not in available_models:
        raise HTTPException(400, f"Model '{model}' not available. Available: {available_models}")
    
    recommendations = await recommender.get_recommendations(
        user_id=user_id,
        k=k,
        model_name=model,
        exclude_seen=exclude_seen
    )
    
    if not recommendations:
        raise HTTPException(404, "No recommendations found for this user")
    
    return {
        "userId": user_id,
        "recommendations": recommendations,
        "model": model,
        "count": len(recommendations)
    }

@router.get("/user/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_by_id(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    model: str = Query('als')
):
    """Get recommendations by user ID (alternative endpoint)"""
    return await get_recommendations(user_id=user_id, k=k, model=model, exclude_seen=True)

@router.get("/similar/{product_id}")
async def get_similar_items(
    product_id: int,
    k: int = Query(10, ge=1, le=100)
):
    """Get similar items"""
    
    similar = await recommender.get_similar_items(product_id, k)
    
    if not similar:
        raise HTTPException(404, "No similar items found")
    
    return {
        "productId": product_id,
        "similarItems": similar,
        "count": len(similar)
    }

@router.delete("/cache/user/{user_id}")
async def clear_user_cache(user_id: int):
    """Clear user cache"""
    cache.clear_user_cache(user_id)
    return {"message": f"Cache cleared for user {user_id}"}

@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": model_loader.get_available_models(),
        "count": len(model_loader.get_available_models())
    }
