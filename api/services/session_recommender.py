"""
Session-based Hybrid Recommender

Combines:
- ALS long-term user preferences
- Session-based short memory (dynamic user vector)
- Content cluster boosting
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from api.services.model_loader import model_loader
from api.config import settings
import logging

logger = logging.getLogger(__name__)


class SessionRecommender:
    """
    Real-time hybrid recommendations with session-based short memory
    
    Key concepts:
    - session_vector: Weighted average of item embeddings from current session
    - dynamic_user_vector: Combination of ALS user vector + session vector
    - cluster_boost: Boost scores for items in popular session clusters
    """
    
    # Action weights for computing session vector
    ACTION_WEIGHTS: Dict[str, float] = {
        "view": 1.0,
        "click": 2.0,
        "add_to_cart": 4.0,
        "purchase": 6.0,
        "add_to_wishlist": 3.0,
        "remove_from_cart": -1.0
    }
    
    def __init__(self):
        self.loader = model_loader
        
        # Configuration from settings
        self.session_alpha = getattr(settings, 'SESSION_ALPHA', 0.3)
        self.time_decay_factor = getattr(settings, 'TIME_DECAY_FACTOR', 0.9)
        self.cluster_boost = getattr(settings, 'CLUSTER_BOOST', 0.15)
    
    def compute_session_vector(
        self, 
        session_events: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Compute weighted session vector from recent interactions
        
        Args:
            session_events: List of {item_id, action, timestamp?}
                           Most recent event should be LAST
        
        Returns:
            Weighted average embedding of session items, or None if no valid items
        """
        if not session_events:
            return None
        
        n_factors = self.loader.n_factors
        if n_factors == 0:
            logger.warning("No factors available for session vector computation")
            return None
        
        weighted_sum = np.zeros(n_factors, dtype=np.float32)
        total_weight = 0.0
        
        # Process events (most recent last, so we reverse for time decay)
        n_events = len(session_events)
        
        for i, event in enumerate(session_events):
            item_id = event.get('item_id')
            action = event.get('action', 'view')
            
            if item_id is None:
                continue
            
            # Get internal item code
            item_code = self.loader.get_item_internal_id(item_id)
            if item_code is None:
                logger.debug(f"Item {item_id} not found in model")
                continue
            
            # Get item embedding
            embedding = self.loader.get_item_embedding(item_code)
            if embedding is None:
                continue
            
            # Compute weight: action_weight × time_decay
            action_weight = self.ACTION_WEIGHTS.get(action, 1.0)
            
            # Time decay: more recent events (higher index) get higher weight
            # Position from end: n_events - 1 - i = 0 for last item
            position_from_end = n_events - 1 - i
            time_weight = self.time_decay_factor ** position_from_end
            
            combined_weight = action_weight * time_weight
            
            if combined_weight > 0:
                weighted_sum += embedding * combined_weight
                total_weight += combined_weight
        
        if total_weight == 0:
            return None
        
        session_vector = weighted_sum / total_weight
        return session_vector
    
    def compute_dynamic_user_vector(
        self,
        user_vector_als: Optional[np.ndarray],
        session_vector: Optional[np.ndarray],
        alpha: float = None
    ) -> Optional[np.ndarray]:
        """
        Combine ALS user vector with session vector
        
        Formula: normalize(user_vector_als + α × session_vector)
        
        Args:
            user_vector_als: Long-term user preferences from ALS (can be None for cold start)
            session_vector: Short-term intent from session
            alpha: Weight for session vector (default: self.session_alpha)
        
        Returns:
            Dynamic user vector combining both signals
        """
        if session_vector is None and user_vector_als is None:
            return None
        
        if session_vector is None:
            return user_vector_als
        
        if user_vector_als is None:
            # Cold start: use only session vector
            return self._normalize(session_vector)
        
        # Determine alpha
        actual_alpha = alpha if alpha is not None else self.session_alpha
        
        # Combine vectors
        combined = user_vector_als + actual_alpha * session_vector
        return self._normalize(combined)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def vote_clusters(
        self, 
        session_events: List[Dict[str, Any]],
        top_k: int = 2
    ) -> List[int]:
        """
        Vote for most popular clusters based on session items
        
        Args:
            session_events: List of session events
            top_k: Number of top clusters to return
        
        Returns:
            List of top cluster IDs
        """
        cluster_counts: Counter = Counter()
        
        for event in session_events:
            item_id = event.get('item_id')
            action = event.get('action', 'view')
            
            if item_id is None:
                continue
            
            item_code = self.loader.get_item_internal_id(item_id)
            if item_code is None:
                continue
            
            cluster_id = self.loader.get_item_cluster(item_code)
            if cluster_id is not None:
                # Weight by action
                weight = self.ACTION_WEIGHTS.get(action, 1.0)
                cluster_counts[cluster_id] += weight
        
        if not cluster_counts:
            return []
        
        # Return top clusters
        return [cluster_id for cluster_id, _ in cluster_counts.most_common(top_k)]
    
    def recalculate_user_vector(
        self,
        session_events: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Step 2: ALS Branch - Recalculate User Factors
        Solving Least Squares problem: P_temp = (Y^T * C * Y + lambda * I)^-1 * Y^T * C * p
        Simplified approximation: Weighted average of session item vectors (similar efficacy for short sessions)
        """
        return self.compute_session_vector(session_events)

    def retrieve_cluster_candidates(
        self,
        session_events: List[Dict[str, Any]],
        limit_per_cluster: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Content Branch - Retrieve candidates from same clusters
        """
        candidates = []
        top_clusters = self.vote_clusters(session_events, top_k=3)
        
        seen_items = set()
        for event in session_events:
            if event.get('item_id'):
                seen_items.add(event.get('item_id'))

        for cluster_id in top_clusters:
            cluster_items = self.loader.get_items_by_cluster(cluster_id)
            # Simple ranking: take first N items (assuming they are somewhat sorted or random)
            # In production, sort by popularity or rating
            count = 0
            for item_code in cluster_items:
                if count >= limit_per_cluster:
                    break
                
                prod_id = self.loader.get_product_id(item_code)
                if prod_id and prod_id not in seen_items:
                    candidates.append({
                        'productId': prod_id,
                        'score': 1.0, # Placeholder score for content match
                        'rank': 0,
                        'cluster': cluster_id,
                        'source': 'content_cluster'
                    })
                    seen_items.add(prod_id)
                    count += 1
        return candidates

    def interleave_results(
        self,
        als_results: List[Dict[str, Any]],
        content_results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Fusion - Interleave results
        """
        final_list = []
        i, j = 0, 0
        ids_in_list = set()
        
        while len(final_list) < k and (i < len(als_results) or j < len(content_results)):
            # Pick from ALS
            if i < len(als_results):
                item = als_results[i]
                if item['productId'] not in ids_in_list:
                    item['source'] = 'als_hybrid'
                    final_list.append(item)
                    ids_in_list.add(item['productId'])
                i += 1
            
            if len(final_list) >= k:
                break
                
            # Pick from Content
            if j < len(content_results):
                item = content_results[j]
                if item['productId'] not in ids_in_list:
                    final_list.append(item)
                    ids_in_list.add(item['productId'])
                j += 1
                
        # Update ranks
        for idx, item in enumerate(final_list):
            item['rank'] = idx + 1
            
        return final_list

    async def get_realtime_recommendations(
        self,
        user_id: int,
        session_events: List[Dict[str, Any]],
        k: int = 20,
        exclude_session_items: bool = True,
        enable_cluster_boost: bool = True,
        session_alpha: float = None
    ) -> Dict[str, Any]:
        """
        Get real-time hybrid recommendations using 4-step process
        
        Step 1: Signal Ingestion (Session Weights)
        Step 2: ALS Branch (Recalculate User Vector)
        Step 3: Content Branch (Cluster Retrieval)
        Step 4: Fusion & Diversification (Interleaving)
        """
        try:
            # === Step 1: Signal Ingestion ===
            # (Handled implicitly in compute_session_vector via ACTION_WEIGHTS)
            
            # === Step 2: ALS Branch ===
            # Recalculate user vector based on session
            user_internal_id = self.loader.get_user_internal_id(user_id)
            user_vector_als = self.loader.get_user_embedding(user_internal_id) if user_internal_id is not None else None
            
            session_vector = self.recalculate_user_vector(session_events)
            dynamic_vector = self.compute_dynamic_user_vector(user_vector_als, session_vector, alpha=session_alpha)
            
            als_recommendations = []
            if dynamic_vector is not None:
                # Score all items
                item_factors = self.loader.get_all_item_factors()
                if item_factors is not None:
                    scores = item_factors @ dynamic_vector
                    
                    # Apply exclusions
                    exclude_set = set()
                    if exclude_session_items:
                        for event in session_events:
                            if event.get('item_id'):
                                ic = self.loader.get_item_internal_id(event.get('item_id'))
                                if ic is not None: exclude_set.add(ic)
                    
                    if user_internal_id is not None:
                        exclude_set.update(self.loader.get_user_seen_items(user_internal_id))
                        
                    for idx in exclude_set:
                        if idx < len(scores): scores[idx] = -np.inf
                        
                    # Top K for ALS branch
                    top_indices = np.argsort(scores)[::-1][:k]
                    for idx in top_indices:
                        if np.isinf(scores[idx]): continue
                        pid = self.loader.get_product_id(int(idx))
                        if pid:
                            als_recommendations.append({
                                'productId': int(pid),
                                'score': float(scores[idx]),
                                'cluster': self.loader.get_item_cluster(int(idx))
                            })

            # === Step 3: Content Branch ===
            content_recommendations = []
            if enable_cluster_boost:
                content_recommendations = self.retrieve_cluster_candidates(session_events, limit_per_cluster=k//2)
                
            # === Step 4: Fusion ===
            final_recommendations = self.interleave_results(als_recommendations, content_recommendations, k)
            
            return {
                "recommendations": final_recommendations,
                "session_items": len(session_events),
                "is_cold_start": user_internal_id is None,
                "count": len(final_recommendations)
            }
            
        except Exception as e:
            logger.error(f"Realtime recommendation error: {e}", exc_info=True)
            return {"recommendations": [], "error": str(e)}

# Singleton instance
session_recommender = SessionRecommender()
