"""
Redis cache
"""

import json
import redis
from typing import Optional, Any
from api.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache manager"""
    
    def __init__(self):
        self.enabled = settings.CACHE_ENABLED
        self.ttl = settings.CACHE_TTL
        
        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.client.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️  Redis failed: {e}. Cache disabled.")
                self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set to cache"""
        if not self.enabled:
            return False
        try:
            ttl = ttl or self.ttl
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key"""
        if not self.enabled:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        if not self.enabled:
            return 0
        try:
            keys = self.client.keys(pattern)
            return self.client.delete(*keys) if keys else 0
        except Exception:
            return 0
    
    def clear_user_cache(self, user_id: int):
        """Clear user cache"""
        pattern = f"rec:user:{user_id}:*"
        deleted = self.delete_pattern(pattern)
        logger.info(f"Cleared {deleted} cache entries for user {user_id}")

cache = RedisCache()
