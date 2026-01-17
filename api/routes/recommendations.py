"""
Recommendation endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from api.schemas.models import (
    RecommendationRequest,
    RecommendationResponse,
    SimilarProduct,
    SessionEvent,
    RealtimeRecommendationRequest,
    RealtimeRecommendationResponse
)
from api.services.recommender import recommender
from api.services.session_recommender import session_recommender
from api.services.model_loader import model_loader
from api.services.cache import cache
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int = Query(..., description="User ID to get recommendations for"),
    k: int = Query(20, ge=1, le=100, description="Number of recommendations"),
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
    
    # Return empty list for cold-start users instead of 404
    return {
        "userId": user_id,
        "recommendations": recommendations or [],
        "model": model,
        "count": len(recommendations) if recommendations else 0
    }

@router.get("/similar/{product_id}")
async def get_similar_items(
    product_id: int,
    k: int = Query(20, ge=1, le=100)
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


@router.post("/realtime", response_model=RealtimeRecommendationResponse)
async def get_realtime_recommendations(request: RealtimeRecommendationRequest):
    """
    Get real-time hybrid recommendations
    
    Combines:
    - ALS long-term preferences (if user exists in training data)
    - Session-based short memory (dynamic user vector from recent interactions)
    - Content cluster boosting (boost items in popular session clusters)
    
    This endpoint does NOT retrain the model - it uses vector operations for low latency.
    """
    
    # Convert Pydantic models to dicts for the session recommender
    session_events = [
        {
            "item_id": event.item_id,
            "action": event.action,
            "timestamp": event.timestamp
        }
        for event in request.session_events
    ]
    
    result = await session_recommender.get_realtime_recommendations(
        user_id=request.user_id,
        session_events=session_events,
        k=request.k,
        exclude_session_items=request.exclude_session_items,
        enable_cluster_boost=request.enable_cluster_boost,
        session_alpha=request.session_alpha
    )
    
    if "error" in result and not result.get("recommendations"):
        raise HTTPException(500, f"Recommendation error: {result.get('error')}")
    
    return {
        "userId": request.user_id,
        "recommendations": result.get("recommendations", []),
        "sessionItems": result.get("session_items", 0),
        "topClusters": result.get("top_clusters", []),
        "isColdStart": result.get("is_cold_start", False),
        "count": len(result.get("recommendations", []))
    }

