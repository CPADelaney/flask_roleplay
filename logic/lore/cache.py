"""
Caching utility functions for the Lore System.
"""

import json
import logging
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

from ..config.settings import config

logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass

class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(config.ERROR_MESSAGES["cache_error"]) from e
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            raise CacheError(f"Cache get failed: {str(e)}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            value_json = json.dumps(value)
            if ttl:
                self.redis.setex(key, ttl, value_json)
            else:
                self.redis.set(key, value_json)
        except (RedisError, TypeError) as e:
            logger.error(f"Cache set error: {e}")
            raise CacheError(f"Cache set failed: {str(e)}")
    
    def delete(self, key: str) -> None:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            self.redis.delete(key)
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            raise CacheError(f"Cache delete failed: {str(e)}")
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            return bool(self.redis.exists(key))
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            raise CacheError(f"Cache exists check failed: {str(e)}")
    
    def clear_pattern(self, pattern: str) -> None:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except RedisError as e:
            logger.error(f"Cache clear pattern error: {e}")
            raise CacheError(f"Cache clear pattern failed: {str(e)}")

# Create global cache instance
cache = RedisCache()

def get_cached_value(key: str) -> Optional[Any]:
    """
    Get a value from cache with error handling.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        return cache.get(key)
    except CacheError:
        return None

def set_cached_value(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """
    Set a value in cache with error handling.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Optional time-to-live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cache.set(key, value, ttl or config.CACHE_TTL)
        return True
    except CacheError:
        return False

def invalidate_cache_pattern(pattern: str) -> bool:
    """
    Invalidate all cache entries matching a pattern.
    
    Args:
        pattern: Redis key pattern
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cache.clear_pattern(pattern)
        return True
    except CacheError:
        return False 