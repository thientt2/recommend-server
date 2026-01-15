"""
Load trained recommendation_system.pkl model with ALS artifacts
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from api.config import settings
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage the trained ALS recommendation model"""
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        
        # Main ALS model artifacts
        self.model = None               # Mô hình ALS đã train
        self.train_matrix = None        # Ma trận train (để lọc item đã mua)
        self.user_map_inv = None        # Tra cứu User ID -> Code
        self.item_map = None            # Tra cứu Code -> Item ID (để trả về client)
        self.product_features = None    # DataFrame chứa cluster & distance
        self.user_pref_cluster = None   # Dict sở thích của user (User -> Cluster)
        
        self._load_recommendation_system()
    
    def _load_recommendation_system(self):
        """Load the recommendation_system.pkl file containing all artifacts"""
        filepath = os.path.join(self.models_dir, 'recommendation_system.pkl')
        
        logger.info("📂 Loading recommendation_system.pkl...")
        
        try:
            with open(filepath, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Extract artifacts
            self.model = artifacts.get('model')
            self.train_matrix = artifacts.get('train_matrix')
            self.user_map_inv = artifacts.get('user_map_inv')
            self.item_map = artifacts.get('item_map')
            self.product_features = artifacts.get('product_features')
            self.user_pref_cluster = artifacts.get('user_pref_cluster')
            
            logger.info("  ✅ ALS model loaded")
            logger.info(f"  ✅ Train matrix shape: {self.train_matrix.shape if self.train_matrix is not None else 'N/A'}")
            logger.info(f"  ✅ User map: {len(self.user_map_inv) if self.user_map_inv else 0} users")
            logger.info(f"  ✅ Item map: {len(self.item_map) if self.item_map else 0} items")
            logger.info(f"  ✅ Product features: {len(self.product_features) if self.product_features is not None else 0} products")
            logger.info(f"  ✅ User preferences: {len(self.user_pref_cluster) if self.user_pref_cluster else 0} users")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to load recommendation_system.pkl: {e}")
            raise
    
    def get_model(self, name: str = 'als') -> Optional[Any]:
        """Get loaded ALS model"""
        if name == 'als' or name == 'recommendation_system':
            return self.model
        return None
    
    def get_available_models(self) -> list:
        """Get available model names"""
        if self.model is not None:
            return ['als']
        return []
    
    def get_user_internal_id(self, user_id: int) -> Optional[int]:
        """Convert external user ID to internal model index"""
        if self.user_map_inv and user_id in self.user_map_inv:
            return self.user_map_inv[user_id]
        return None
    
    def get_product_id(self, internal_id: int) -> Optional[int]:
        """Convert internal item index to external product ID"""
        if self.item_map and internal_id in self.item_map:
            return self.item_map[internal_id]
        return None
    
    def get_user_seen_items(self, user_internal_id: int) -> set:
        """Get items that user has already interacted with"""
        if self.train_matrix is None:
            return set()
        
        try:
            # Get user row from train matrix (sparse matrix)
            user_row = self.train_matrix[user_internal_id]
            # Get non-zero item indices
            if hasattr(user_row, 'indices'):
                return set(user_row.indices)
            elif hasattr(user_row, 'nonzero'):
                return set(user_row.nonzero()[0])
            else:
                return set(np.nonzero(user_row)[0])
        except Exception as e:
            logger.warning(f"Error getting seen items: {e}")
            return set()
    
    def get_user_preferred_cluster(self, user_id: int) -> Optional[int]:
        """Get user's preferred cluster"""
        if self.user_pref_cluster and user_id in self.user_pref_cluster:
            return self.user_pref_cluster[user_id]
        return None


model_loader = ModelLoader()
