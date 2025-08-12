# logic/lore/utils/cache.py

"""
Caching utility functions for the Lore System.
Refactored to use async/await patterns.
"""

import json
import logging
from typing import Any, Optional
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
            # Get Redis URL from environment (Render format)
            redis_url = os.getenv('REDIS_URL')
            
            if redis_url:
                # Use Redis URL if provided (for Render/production)
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=True
                )
                logger.info(f"Connecting to Redis via URL: {redis_url.split('@')[1] if '@' in redis_url else redis_url}")
            else:
                # Fall back to localhost for development
                self.redis = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                logger.info("Connecting to Redis at localhost:6379")
            
            # Test connection
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(config.ERROR_MESSAGES["cache_error"]) from e

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache."""
        try:
            value = await asyncio.to_thread(self.redis.get, key)
            return json.loads(value) if value else None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            raise CacheError(f"Cache get failed: {str(e)}")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in cache."""
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
        """Delete a value from cache."""
        try:
            await asyncio.to_thread(self.redis.delete, key)
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            raise CacheError(f"Cache delete failed: {str(e)}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            return bool(await asyncio.to_thread(self.redis.exists, key))
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            raise CacheError(f"Cache exists check failed: {str(e)}")

    async def clear_pattern(self, pattern: str) -> None:
        """Clear all keys matching a pattern."""
        try:
            keys = await asyncio.to_thread(self.redis.keys, pattern)
            if keys:
                await asyncio.to_thread(self.redis.delete, *keys)
        except RedisError as e:
            logger.error(f"Cache clear pattern error: {e}")
            raise CacheError(f"Cache clear pattern failed: {str(e)}")


class LocalCache:
    """In-memory local cache implementation."""

    def __init__(self, max_size: int = 1000):
        """Initialize local cache."""
        self.cache = {}
        self.max_size = max_size
        self.expiry = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the local cache."""
        now = datetime.now()
        if key in self.cache:
            if key in self.expiry and now > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return None
            return self.cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the local cache."""
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.expiry:
                del self.expiry[oldest_key]

        self.cache[key] = value

        if ttl:
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl)

    async def delete(self, key: str) -> None:
        """Delete a value from the local cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the local cache."""
        now = datetime.now()
        if key in self.cache:
            if key in self.expiry and now > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return False
            return True
        return False

    async def clear_pattern(self, pattern: str) -> None:
        """Clear all keys matching a pattern."""
        matching_keys = [k for k in self.cache.keys() if pattern in k]
        for key in matching_keys:
            await self.delete(key)


# Create caches
local_cache = LocalCache(max_size=config.CACHE_MAX_SIZE)
try:
    cache = AsyncRedisCache()
except CacheError:
    logger.warning("Redis cache unavailable, using local cache instead.")
    cache = local_cache


async def get_cached_value(key: str) -> Optional[Any]:
    """Get a value from cache with error handling."""
    try:
        return await cache.get(key)
    except CacheError:
        return None


async def set_cached_value(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set a value in cache with error handling."""
    try:
        await cache.set(key, value, ttl or config.CACHE_TTL)
        return True
    except CacheError:
        return False


async def invalidate_cache_pattern(pattern: str) -> bool:
    """Invalidate all cache entries matching a pattern."""
    try:
        await cache.clear_pattern(pattern)
        return True
    except CacheError:
        return False


async def fallback_get_cached_value(key: str) -> Optional[Any]:
    """Try to get a value from Redis, fall back to local cache if Redis fails."""
    try:
        value = await get_cached_value(key)
        if value is not None:
            return value
        return await local_cache.get(key)
    except Exception:
        return await local_cache.get(key)


async def resilient_set_cached_value(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Try to set a value in Redis, also set in local cache as backup."""
    try:
        await set_cached_value(key, value, ttl)
    except Exception:
        pass

    await local_cache.set(key, value, ttl)

