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
        k: int = 20,
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
            model = self.model_loader.als_model
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
            model = self.model_loader.als_model
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
            model = self.model_loader.als_model
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
        """Get similar items based on product clustering and ALS embeddings"""
        cache_key = f"rec:similar:{product_id}:k:{k}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # 1. Map external ID to internal ID
            internal_id = self.model_loader.get_item_internal_id(product_id)
            if internal_id is None:
                logger.warning(f"Product {product_id} not found in model mappings")
                return []

            # 2. Get Cluster ID
            cluster_id = self.model_loader.get_item_cluster(internal_id)
            if cluster_id is None:
                # Fallback: if no cluster, just find similar by embedding global
                logger.info(f"Product {product_id} has no cluster, falling back to global similarity")
                # (Optional: implement global nearest neighbors if needed, but for now just return empty or proceed)
            
            # 3. Get candidates
            # If we have a cluster, get items in that cluster. If not, maybe skip or use all items?
            # Let's stick to cluster-based retrieval for efficiency if possible.
            candidates_internal_ids = []
            if cluster_id is not None:
                candidates_internal_ids = self.model_loader.get_items_by_cluster(cluster_id)
            else:
                 # If no cluster, maybe we shouldn't return anything or it's too expensive to scan all
                 return []
            
            # Filter out the query item
            candidates_internal_ids = [mid for mid in candidates_internal_ids if mid != internal_id]
            
            if not candidates_internal_ids:
                return []

            # 4. Rank by Cosine Similarity using ALS item factors
            item_factors = self.model_loader.item_factors
            if item_factors is not None and internal_id < len(item_factors):
                query_vector = item_factors[internal_id]
                norm_query = np.linalg.norm(query_vector)
                
                scores = []
                for candidate_id in candidates_internal_ids:
                    if candidate_id < len(item_factors):
                        candidate_vector = item_factors[candidate_id]
                        norm_candidate = np.linalg.norm(candidate_vector)
                        
                        if norm_query > 0 and norm_candidate > 0:
                            # Cosine similarity
                            score = np.dot(query_vector, candidate_vector) / (norm_query * norm_candidate)
                        else:
                            score = 0.0
                            
                        scores.append((candidate_id, score))
                
                # Sort by score desc
                scores.sort(key=lambda x: x[1], reverse=True)
                top_k = scores[:k]
                
                result = []
                for i, (mid, score) in enumerate(top_k):
                    pid = self.model_loader.get_product_id(mid)
                    if pid:
                        result.append({
                            'productId': int(pid),
                            'similarityScore': float(score),
                            'rank': i + 1
                        })
            else:
                # Fallback if no embeddings: just return first K items from cluster
                logger.warning(f"No item factors available for ranking similar items for {product_id}")
                top_k_ids = candidates_internal_ids[:k]
                result = []
                for i, mid in enumerate(top_k_ids):
                    pid = self.model_loader.get_product_id(mid)
                    if pid:
                        result.append({
                            'productId': int(pid),
                            'similarityScore': 1.0, 
                            'rank': i + 1
                        })
            
            cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Similar items error: {e}", exc_info=True)
            return []

recommender = RecommendationEngine()
