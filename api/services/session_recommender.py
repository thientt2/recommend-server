"""
Session-based Hybrid Recommender

Combines:
- ALS long-term user preferences
- Session-based short memory (dynamic user vector)
- Content cluster boosting
"""

import numpy as np
import random
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
        self.last_item_boost = getattr(settings, 'LAST_ITEM_BOOST', 2.0)  # Extra boost for last interacted item
    
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
            
            # Compute weight: action_weight × time_decay × last_item_boost
            action_weight = self.ACTION_WEIGHTS.get(action, 1.0)
            
            # Time decay: more recent events (higher index) get higher weight
            # Position from end: n_events - 1 - i = 0 for last item
            position_from_end = n_events - 1 - i
            time_weight = self.time_decay_factor ** position_from_end
            
            # Apply extra boost to the LAST item (most recent interaction)
            is_last_item = (i == n_events - 1)
            last_boost = self.last_item_boost if is_last_item else 1.0
            
            combined_weight = action_weight * time_weight * last_boost
            
            if is_last_item:
                logger.debug(f"Last item {item_id} boosted with {last_boost}x multiplier")
            
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
        
        Priority:
        1. Last item's cluster (most relevant to current interest)
        2. Other voted clusters from session
        """
        candidates = []
        
        # Build seen items set
        seen_items = set()
        for event in session_events:
            if event.get('item_id'):
                seen_items.add(event.get('item_id'))
        
        # Priority 1: Get cluster from LAST interacted item
        last_item_cluster = None
        if session_events:
            logger.info(f"Session has {len(session_events)} events, checking for last item cluster...")
            # Find the last event with a valid item_id (most recent)
            for event in reversed(session_events):
                item_id = event.get('item_id')
                logger.info(f"Checking event: item_id={item_id}, action={event.get('action')}")
                if item_id is not None:
                    item_code = self.loader.get_item_internal_id(item_id)
                    logger.info(f"  -> item_code (internal) = {item_code}")
                    if item_code is not None:
                        last_item_cluster = self.loader.get_item_cluster(item_code)
                        logger.info(f"  -> cluster = {last_item_cluster}")
                        if last_item_cluster is not None:
                            logger.info(f"Last item {item_id} belongs to cluster {last_item_cluster}")
                            break
                    else:
                        logger.warning(f"Item {item_id} NOT FOUND in model mappings!")
        
        # Priority 2: Get voted clusters from all session items
        voted_clusters = self.vote_clusters(session_events, top_k=3)
        
        # Build final cluster list: last_item_cluster first, then voted (without duplicates)
        clusters_to_use = []
        if last_item_cluster is not None:
            clusters_to_use.append(last_item_cluster)
        for c in voted_clusters:
            if c not in clusters_to_use:
                clusters_to_use.append(c)
        
        logger.info(f"Cluster priority order: {clusters_to_use}")
        
        # Retrieve items from each cluster with random shuffle for variety
        # PRIORITY: Get at least 5 items from last_item_cluster
        LAST_ITEM_CLUSTER_MIN = 5
        
        ids_added = set()
        for idx, cluster_id in enumerate(clusters_to_use):
            cluster_items = self.loader.get_items_by_cluster(cluster_id)
            
            # Shuffle items to get variety on each request
            cluster_items_list = list(cluster_items) if not isinstance(cluster_items, list) else cluster_items
            random.shuffle(cluster_items_list)
            
            # Determine limit for this cluster
            if idx == 0 and last_item_cluster is not None:
                # First cluster (last_item_cluster) gets at least 5 items
                current_limit = max(LAST_ITEM_CLUSTER_MIN, limit_per_cluster)
                logger.info(f"Retrieving {current_limit} items from last_item_cluster {cluster_id}")
            else:
                current_limit = limit_per_cluster
            
            count = 0
            for item_code in cluster_items_list:
                if count >= current_limit:
                    break
                
                prod_id = self.loader.get_product_id(item_code)
                if prod_id and prod_id not in seen_items and prod_id not in ids_added:
                    candidates.append({
                        'productId': prod_id,
                        'score': 1.0,
                        'rank': 0,
                        'cluster': cluster_id,
                        'source': 'content_cluster'
                    })
                    ids_added.add(prod_id)
                    count += 1
            
            logger.info(f"Cluster {cluster_id}: retrieved {count} items")
        
        logger.info(f"Retrieved {len(candidates)} cluster candidates total")
        return candidates

    def interleave_results(
        self,
        als_results: List[Dict[str, Any]],
        content_results: List[Dict[str, Any]],
        k: int = None  # k is now optional, if None returns all
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Fusion - Interleave results
        Ratio: 1 ALS : 3 Content (prioritize cluster-based for session relevance)
        If k is None, returns ALL combined results
        """
        total_possible = len(als_results) + len(content_results)
        max_results = k if k is not None else total_possible
        logger.info(f"Interleaving: {len(als_results)} ALS + {len(content_results)} Content -> max {max_results}")
        
        final_list = []
        i, j = 0, 0
        ids_in_list = set()
        
        while (i < len(als_results) or j < len(content_results)):
            # Check if we've reached the limit (if k is set)
            if k is not None and len(final_list) >= k:
                break
                
            # Pick 1 from ALS
            if i < len(als_results):
                item = als_results[i]
                if item['productId'] not in ids_in_list:
                    item['source'] = 'als_hybrid'
                    final_list.append(item)
                    ids_in_list.add(item['productId'])
                i += 1
            
            if k is not None and len(final_list) >= k:
                break
                
            # Pick 3 from Content (cluster-based) to prioritize session relevance
            for _ in range(3):
                if j < len(content_results):
                    if k is not None and len(final_list) >= k:
                        break
                    item = content_results[j]
                    if item['productId'] not in ids_in_list:
                        final_list.append(item)
                        ids_in_list.add(item['productId'])
                    j += 1
                
        # Update ranks
        for idx, item in enumerate(final_list):
            item['rank'] = idx + 1
            
        return final_list

    def merge_results(
        self,
        als_results: List[Dict[str, Any]],
        content_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Simple merge: ALS results first, then Content results
        Removes duplicates, assigns ranks
        
        Expected: 10 ALS + 10 Content = ~20 total (after dedup)
        """
        logger.info(f"Merging: {len(als_results)} ALS + {len(content_results)} Content")
        
        final_list = []
        ids_in_list = set()
        
        # Add all ALS results first
        for item in als_results:
            if item['productId'] not in ids_in_list:
                final_list.append(item)
                ids_in_list.add(item['productId'])
        
        # Add Content results (skip duplicates)
        for item in content_results:
            if item['productId'] not in ids_in_list:
                final_list.append(item)
                ids_in_list.add(item['productId'])
        
        # Update ranks
        for idx, item in enumerate(final_list):
            item['rank'] = idx + 1
        
        logger.info(f"Merged: {len(final_list)} total unique items")
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
                        
                    # Get TOP 10 items from ALS
                    top_indices = np.argsort(scores)[::-1][:10]
                    for idx in top_indices:
                        if np.isinf(scores[idx]): continue
                        pid = self.loader.get_product_id(int(idx))
                        if pid:
                            als_recommendations.append({
                                'productId': int(pid),
                                'score': float(scores[idx]),
                                'cluster': self.loader.get_item_cluster(int(idx)),
                                'source': 'als_hybrid'
                            })

            # === Step 3: Content Branch ===
            # Get 10 items from K-means clusters (based on session)
            content_recommendations = []
            if enable_cluster_boost:
                # Get ~10 items total from clusters (prioritize last_item_cluster)
                content_recommendations = self.retrieve_cluster_candidates(session_events, limit_per_cluster=5)
                # Limit to 10 items max from content
                content_recommendations = content_recommendations[:10]
                
            # === Step 4: Merge - Combine ALS (10) + Content (10) = 20 total ===
            final_recommendations = self.merge_results(als_recommendations, content_recommendations)
            
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
