# utils/cache_manager.py

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
import json

logger = logging.getLogger(__name__)

class CacheManager:
    """Asynchronous cache manager that supports TTL and size limits."""
    
    def __init__(self, name: str = "default", max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache manager.
        
        Args:
            name: Name of the cache (for logging)
            max_size: Maximum number of entries
            ttl: Default TTL in seconds
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"Initialized cache '{name}' with max_size={max_size}, ttl={ttl}")
    
    async def get(self, key: str) -> Any:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        async with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            # Check if expired
            if entry['expires'] < time.time():
                del self.cache[key]
                self.misses += 1
                return None
            
            # Update access time
            entry['last_access'] = time.time()
            entry['access_count'] += 1
            
            self.hits += 1
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional, uses default if None)
        """
        expires = time.time() + (ttl if ttl is not None else self.default_ttl)
        
        async with self.lock:
            # Check if we need to evict items
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_items()
            
            self.cache[key] = {
                'value': value,
                'expires': expires,
                'created': time.time(),
                'last_access': time.time(),
                'access_count': 0
            }
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False if it wasn't in the cache
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and isn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists and isn't expired
        """
        async with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                return False
            
            # Check if expired
            if entry['expires'] < time.time():
                del self.cache[key]
                return False
            
            return True
    
    async def _evict_items(self) -> int:
        """
        Evict items to make room in the cache.
        
        Returns:
            Number of items evicted
        """
        # Evict 10% of cache or at least 1 item
        evict_count = max(1, int(self.max_size * 0.1))
        
        # Sort by last access time (oldest first)
        items_by_access = sorted(
            self.cache.items(),
            key=lambda x: (x[1]['last_access'], -x[1]['access_count'])
        )
        
        # Evict oldest items
        for i in range(min(evict_count, len(items_by_access))):
            key, _ = items_by_access[i]
            del self.cache[key]
            self.evictions += 1
        
        return evict_count
    
    async def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items cleared
        """
        async with self.lock:
            count = len(self.cache)
            self.cache.clear()
            return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        async with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'name': self.name,
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions
            }
