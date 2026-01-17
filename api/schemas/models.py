"""
Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    k: int = Field(20, ge=1, le=100)
    model: str = Field('als')
    exclude_seen: bool = Field(True)

class ProductRecommendation(BaseModel):
    productId: int
    score: float
    rank: int

class RecommendationResponse(BaseModel):
    userId: int
    recommendations: List[ProductRecommendation]
    model: str
    count: int

class SimilarProduct(BaseModel):
    productId: int
    name: str
    price: float
    rating: float
    image: Optional[str] = None
    similarityScore: float
    rank: int

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    modelsLoaded: List[str]
    cacheEnabled: bool


# === Session-based Recommendation Schemas ===

class SessionEvent(BaseModel):
    """Single user interaction event in a session"""
    item_id: int = Field(..., description="Product ID")
    action: str = Field("view", description="Action type: view, click, add_to_cart, purchase, add_to_wishlist")
    timestamp: Optional[int] = Field(None, description="Unix timestamp (optional)")


class RealtimeRecommendationRequest(BaseModel):
    """Request for real-time hybrid recommendations"""
    user_id: int = Field(..., description="User ID")
    session_events: List[SessionEvent] = Field(..., description="List of session interactions (most recent last)")
    k: int = Field(20, ge=1, le=100, description="Number of recommendations")
    exclude_session_items: bool = Field(True, description="Exclude items from current session")
    enable_cluster_boost: bool = Field(True, description="Apply content cluster boosting")
    session_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for session vector (Dynamic User Vector strength)")


class RealtimeRecommendation(BaseModel):
    """Single recommendation with cluster info"""
    productId: int
    score: float
    rank: int
    cluster: Optional[int] = None


class RealtimeRecommendationResponse(BaseModel):
    """Response with real-time hybrid recommendations"""
    userId: int
    recommendations: List[RealtimeRecommendation]
    sessionItems: int
    topClusters: List[int]
    isColdStart: bool
    count: int

