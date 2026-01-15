"""
Recommendation engine using ALS model
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from api.services.model_loader import model_loader
from api.services.cache import cache
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Generate recommendations using ALS model"""
    
    def __init__(self):
        self.model_loader = model_loader
    
    async def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        model_name: str = 'als',
        exclude_seen: bool = True
    ) -> List[Dict[str, Any]]:
        """Get top-K recommendations for a user using ALS model"""
        
        # Check cache
        cache_key = f"rec:user:{user_id}:model:{model_name}:k:{k}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: user {user_id}")
            return cached
        
        try:
            # Get ALS model
            model = self.model_loader.model
            if model is None:
                raise ValueError("ALS model not loaded")
            
            # Convert user_id to internal index
            user_internal_id = self.model_loader.get_user_internal_id(user_id)
            
            if user_internal_id is None:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            # Get recommendations from ALS model
            recommendations_list = self._get_als_recommendations(
                user_internal_id, k, exclude_seen
            )
            
            if not recommendations_list:
                return []
            
            # Format response with scores
            recommendations = []
            for i, (internal_id, score) in enumerate(recommendations_list):
                product_id = self.model_loader.get_product_id(internal_id)
                if product_id is not None:
                    recommendations.append({
                        'productId': int(product_id),
                        'score': float(score),
                        'rank': i + 1
                    })
            
            # Cache
            cache.set(cache_key, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}", exc_info=True)
            return []
    
    def _get_als_recommendations(
        self, 
        user_internal_id: int, 
        k: int, 
        exclude_seen: bool
    ) -> List[Tuple[int, float]]:
        """Get recommendations from ALS model"""
        try:
            model = self.model_loader.model
            train_matrix = self.model_loader.train_matrix
            
            if hasattr(model, 'recommend'):
                try:
                    # implicit >= 0.6.0
                    ids, scores = model.recommend(
                        user_internal_id,
                        train_matrix[user_internal_id] if train_matrix is not None else None,
                        N=k,
                        filter_already_liked_items=exclude_seen
                    )
                    return list(zip(ids, scores))
                except TypeError:
                    try:
                        # implicit < 0.6.0
                        recommendations = model.recommend(
                            user_internal_id,
                            train_matrix,
                            N=k,
                            filter_already_liked_items=exclude_seen
                        )
                        return recommendations
                    except Exception:
                        return self._manual_als_recommend(user_internal_id, k, exclude_seen)
            else:
                return self._manual_als_recommend(user_internal_id, k, exclude_seen)
                
        except Exception as e:
            logger.error(f"ALS recommendation error: {e}", exc_info=True)
            return []
    
    def _manual_als_recommend(
        self, 
        user_internal_id: int, 
        k: int, 
        exclude_seen: bool
    ) -> List[Tuple[int, float]]:
        """Manual ALS recommendation if model.recommend doesn't work"""
        try:
            model = self.model_loader.model
            user_factors = model.user_factors[user_internal_id]
            item_factors = model.item_factors
            scores = item_factors.dot(user_factors)
            
            if exclude_seen:
                seen_items = self.model_loader.get_user_seen_items(user_internal_id)
                for item_id in seen_items:
                    if item_id < len(scores):
                        scores[item_id] = -np.inf
            
            top_k_indices = np.argsort(scores)[::-1][:k]
            return [(int(idx), float(scores[idx])) for idx in top_k_indices]
            
        except Exception as e:
            logger.error(f"Manual ALS recommend error: {e}")
            return []
    
    async def get_similar_items(self, product_id: int, k: int = 10) -> List[Dict[str, Any]]:
        """Get similar items based on product features/cluster"""
        cache_key = f"rec:similar:{product_id}:k:{k}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            product_features = self.model_loader.product_features
            if product_features is None:
                return []
            
            # Find product's cluster
            if 'product_id' in product_features.columns:
                product_row = product_features[product_features['product_id'] == product_id]
            else:
                product_row = product_features[product_features.index == product_id]
            
            if len(product_row) == 0:
                return []
            
            cluster = product_row.iloc[0].get('cluster', None)
            if cluster is None:
                return []
            
            # Get other products in the same cluster
            if 'product_id' in product_features.columns:
                same_cluster = product_features[
                    (product_features['cluster'] == cluster) & 
                    (product_features['product_id'] != product_id)
                ]
                similar_ids = same_cluster['product_id'].head(k).tolist()
            else:
                same_cluster = product_features[
                    (product_features['cluster'] == cluster) & 
                    (product_features.index != product_id)
                ]
                similar_ids = same_cluster.index[:k].tolist()
            
            result = []
            for i, pid in enumerate(similar_ids):
                result.append({
                    'productId': int(pid),
                    'similarityScore': 1.0 - (i * 0.05),
                    'rank': i + 1
                })
            
            cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Similar items error: {e}")
            return []

recommender = RecommendationEngine()
