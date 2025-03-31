# logic/lore/utils/cache.py

"""
Caching utility functions for the Lore System.
Refactored to use async/await patterns.
"""

import json
import logging
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import redis
import asyncio
from redis.exceptions import RedisError

from ..config.settings import config

logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass

class AsyncRedisCache:
    """Redis-based cache implementation with async support."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            
            # Test connection
            self.redis.ping()
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(config.ERROR_MESSAGES["cache_error"]) from e
    
    async def get(self, key: str) -> Optional[Any]:
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
            # Run Redis commands in a thread to avoid blocking event loop
            value = await asyncio.to_thread(self.redis.get, key)
            return json.loads(value) if value else None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            raise CacheError(f"Cache get failed: {str(e)}")
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
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
                await asyncio.to_thread(self.redis.setex, key, ttl, value_json)
            else:
                await asyncio.to_thread(self.redis.set, key, value_json)
        except (RedisError, TypeError) as e:
            logger.error(f"Cache set error: {e}")
            raise CacheError(f"Cache set failed: {str(e)}")
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            await asyncio.to_thread(self.redis.delete, key)
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            raise CacheError(f"Cache delete failed: {str(e)}")
    
    async def exists(self, key: str) -> bool:
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
            return bool(await asyncio.to_thread(self.redis.exists, key))
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            raise CacheError(f"Cache exists check failed: {str(e)}")
    
    async def clear_pattern(self, pattern: str) -> None:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            keys = await asyncio.to_thread(self.redis.keys, pattern)
            if keys:
                await asyncio.to_thread(self.redis.delete, *keys)
        except RedisError as e:
            logger.error(f"Cache clear pattern error: {e}")
            raise CacheError(f"Cache clear pattern failed: {str(e)}")

# Create global cache instance
cache = AsyncRedisCache()

async def get_cached_value(key: str) -> Optional[Any]:
    """
    Get a value from cache with error handling.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        return await cache.get(key)
    except CacheError:
        return None

async def set_cached_value(key: str, value: Any, ttl: Optional[int] = None) -> bool:
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
        await cache.set(key, value, ttl or config.CACHE_TTL)
        return True
    except CacheError:
        return False

async def invalidate_cache_pattern(pattern: str) -> bool:
    """
    Invalidate all cache entries matching a pattern.
    
    Args:
        pattern: Redis key pattern
        
    Returns:
        True if successful, False otherwise
    """
    try:
        await cache.clear_pattern(pattern)
        return True
    except CacheError:
        return False

class LocalCache:
    """In-memory local cache implementation."""
    
    def __init__(self, max_size=1000):
        """Initialize local cache."""
        self.cache = {}
        self.max_size = max_size
        self.expiry = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the local cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        now = datetime.now()
        if key in self.cache:
            # Check if expired
            if key in self.expiry and now > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return None
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the local cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        # Check if we need to make room
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.expiry:
                del self.expiry[oldest_key]
                
        # Store the value
        self.cache[key] = value
        
        # Set expiry if provided
        if ttl:
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the local cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the local cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and not expired, False otherwise
        """
        now = datetime.now()
        if key in self.cache:
            # Check if expired
            if key in self.expiry and now > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return False
            return True
        return False
    
    async def clear_pattern(self, pattern: str) -> None:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (simple substring match)
        """
        # Get keys that match the pattern
        matching_keys = [k for k in self.cache.keys() if pattern in k]
        
        # Delete matching keys
        for key in matching_keys:
            await self.delete(key)

# Create a local cache instance for fallback
local_cache = LocalCache(max_size=config.CACHE_MAX_SIZE)

async def fallback_get_cached_value(key: str) -> Optional[Any]:
    """
    Try to get a value from Redis, fall back to local cache if Redis fails.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        # Try Redis first
        value = await get_cached_value(key)
        if value is not None:
            return value
            
        # Fall back to local cache
        return await local_cache.get(key)
    except Exception:
        # If all else fails, try local cache
        return await local_cache.get(key)

async def resilient_set_cached_value(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Try to set a value in Redis, also set in local cache as backup.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Optional time-to-live in seconds
    """
    try:
        # Try Redis
        await set_cached_value(key, value, ttl)
    except Exception:
        pass
        
    # Always set in local cache too
    await local_cache.set(key, value, ttl)
