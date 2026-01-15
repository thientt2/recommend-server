"""
Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    k: int = Field(10, ge=1, le=100)
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
