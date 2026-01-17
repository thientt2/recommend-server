"""
Load trained recommendation model with hybrid artifacts:
- ALS model (long-term preferences)
- K-Means clustering (content-based)
- Mappings for ID conversions
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Set
from api.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage the hybrid recommendation model"""
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        
        # === ALS Model Artifacts ===
        self.als_model = None           # Trained ALS model
        self.train_matrix = None        # User-item interaction matrix (for filtering)
        self.user_factors = None        # User embedding matrix (n_users, factors)
        self.item_factors = None        # Item embedding matrix (n_items, factors)
        
        # === K-Means Clustering Artifacts ===
        self.kmeans_model = None        # Trained K-Means model
        self.tfidf_vectorizer = None    # Text feature extractor
        self.price_scaler = None        # Price normalization
        self.image_pca = None           # Image dimension reduction
        self.image_scaler = None        # Image embedding normalization
        
        # === ID Mappings ===
        self.user_id_to_code: Dict[int, int] = {}   # Real User ID -> Model Index
        self.item_id_to_code: Dict[int, int] = {}   # Real Item ID -> Model Index
        self.item_code_to_id: Dict[int, int] = {}   # Model Index -> Real Item ID
        self.item_code_to_cluster: Dict[int, int] = {}  # Model Index -> Cluster ID
        
        # === Config ===
        self.configs: Dict[str, Any] = {}
        self.n_clusters: int = 30
        self.n_factors: int = 0
        self.n_items: int = 0
        self.n_users: int = 0
        
        # Load the model
        self._load_hybrid_model()
    
    def _load_hybrid_model(self):
        """Load the recommend_model.pkl containing all hybrid artifacts"""
        filepath = os.path.join(self.models_dir, getattr(settings, 'MODEL_FILE', 'recommend_model.pkl'))
        
        logger.info(f"📂 Loading hybrid model from {filepath}...")
        
        try:
            with open(filepath, 'rb') as f:
                artifacts = pickle.load(f)
            
            # === Load ALS Model ===
            self.als_model = artifacts.get('als_model')
            self.train_matrix = artifacts.get('train_matrix')
            
            if self.als_model is not None:
                self.user_factors = getattr(self.als_model, 'user_factors', None)
                self.item_factors = getattr(self.als_model, 'item_factors', None)
                
                if self.item_factors is not None:
                    self.n_items, self.n_factors = self.item_factors.shape
                if self.user_factors is not None:
                    self.n_users = self.user_factors.shape[0]
            
            logger.info(f"  ✅ ALS model loaded ({self.n_users} users, {self.n_items} items, {self.n_factors} factors)")
            
            # === Load K-Means Model ===
            self.kmeans_model = artifacts.get('kmeans_model')
            self.tfidf_vectorizer = artifacts.get('tfidf_vectorizer')
            self.price_scaler = artifacts.get('price_scaler')
            self.image_pca = artifacts.get('image_pca')
            self.image_scaler = artifacts.get('image_scaler')
            
            if self.kmeans_model is not None:
                self.n_clusters = getattr(self.kmeans_model, 'n_clusters', 30)
            logger.info(f"  ✅ K-Means model loaded ({self.n_clusters} clusters)")
            
            # === Load Mappings ===
            # FINAL pickle structure (ALL have misleading names!):
            # - user_id_to_code: matrix_index (0-936) -> user_id (large numbers)
            # - item_id_to_code: matrix_index (0-524) -> product_id (large numbers)
            # - item_code_to_id: same as item_id_to_code
            # - item_code_to_cluster: matrix_index (0-524) -> cluster_id ✓
            maps = artifacts.get('maps', {})
            
            # User mapping: pickle has matrix_idx -> user_id, we need user_id -> matrix_idx
            user_code_to_id = maps.get('user_id_to_code', {})  
            self.user_id_to_code = {
                user_id: matrix_idx 
                for matrix_idx, user_id in user_code_to_id.items()
            }
            
            # Item mapping: pickle ALSO has matrix_idx -> product_id, we need product_id -> matrix_idx
            item_code_to_id_pickle = maps.get('item_id_to_code', {})
            self.item_id_to_code = {
                product_id: matrix_idx 
                for matrix_idx, product_id in item_code_to_id_pickle.items()
            }
            
            # Reverse mapping: matrix_index -> product_id (use pickle directly)
            self.item_code_to_id = dict(item_code_to_id_pickle)
            
            # Cluster mapping: matrix_index -> cluster_id (correct!)
            self.item_code_to_cluster = maps.get('item_code_to_cluster', {})
            
            logger.info(f"  ✅ Mappings loaded:")
            logger.info(f"      - user_id_to_code: {len(self.user_id_to_code)} users (user_id -> matrix_idx)")
            logger.info(f"      - item_id_to_code: {len(self.item_id_to_code)} items (product_id -> matrix_idx)")
            logger.info(f"      - item_code_to_id: {len(self.item_code_to_id)} items (matrix_idx -> product_id)")
            logger.info(f"      - item_code_to_cluster: {len(self.item_code_to_cluster)} items")
            
            # === Load Configs ===
            self.configs = artifacts.get('configs', {})
            self.n_clusters = self.configs.get('N_CLUSTERS', self.n_clusters)
            logger.info(f"  ✅ Configs: {self.configs}")
            
        except FileNotFoundError:
            logger.error(f"  ❌ Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"  ❌ Failed to load model: {e}")
            raise
    
    # === ALS Methods ===
    
    def get_model(self, name: str = 'als') -> Optional[Any]:
        """Get loaded ALS model"""
        if name == 'als' or name == 'recommendation_system':
            return self.als_model
        return None
    
    def get_available_models(self) -> list:
        """Get available model names"""
        models = []
        if self.als_model is not None:
            models.append('als')
        if self.kmeans_model is not None:
            models.append('kmeans')
        return models
    
    def get_user_internal_id(self, user_id: int) -> Optional[int]:
        """Convert external user ID to internal model index"""
        return self.user_id_to_code.get(user_id)
    
    def get_item_internal_id(self, item_id: int) -> Optional[int]:
        """Convert external item ID to internal model index"""
        return self.item_id_to_code.get(item_id)
    
    def get_product_id(self, internal_id: int) -> Optional[int]:
        """Convert internal item index to external product ID"""
        return self.item_code_to_id.get(internal_id)
    
    def get_user_embedding(self, user_internal_id: int) -> Optional[np.ndarray]:
        """Get user factor vector from ALS model"""
        if self.user_factors is None or user_internal_id >= self.n_users:
            return None
        return self.user_factors[user_internal_id]
    
    def get_item_embedding(self, item_internal_id: int) -> Optional[np.ndarray]:
        """Get item factor vector from ALS model"""
        if self.item_factors is None or item_internal_id >= self.n_items:
            return None
        return self.item_factors[item_internal_id]
    
    def get_all_item_factors(self) -> Optional[np.ndarray]:
        """Get all item factors matrix"""
        return self.item_factors
    
    def get_all_item_codes(self) -> List[int]:
        """Get list of all valid item codes"""
        return list(self.item_code_to_id.keys())
    
    def get_user_seen_items(self, user_internal_id: int) -> Set[int]:
        """Get items that user has already interacted with"""
        if self.train_matrix is None:
            return set()
        
        try:
            user_row = self.train_matrix[user_internal_id]
            if hasattr(user_row, 'indices'):
                return set(user_row.indices)
            elif hasattr(user_row, 'nonzero'):
                return set(user_row.nonzero()[0])
            else:
                return set(np.nonzero(user_row)[0])
        except Exception as e:
            logger.warning(f"Error getting seen items: {e}")
            return set()
    
    # === K-Means Cluster Methods ===
    
    def get_item_cluster(self, item_internal_id: int) -> Optional[int]:
        """Get cluster ID for an item"""
        return self.item_code_to_cluster.get(item_internal_id)
    
    def get_items_by_cluster(self, cluster_id: int) -> List[int]:
        """Get all item codes belonging to a cluster"""
        return [
            item_code for item_code, cluster in self.item_code_to_cluster.items()
            if cluster == cluster_id
        ]
    
    def get_cluster_distribution(self) -> Dict[int, int]:
        """Get count of items per cluster"""
        distribution: Dict[int, int] = {}
        for cluster in self.item_code_to_cluster.values():
            distribution[cluster] = distribution.get(cluster, 0) + 1
        return distribution


# Singleton instance
model_loader = ModelLoader()
