# context/unified_cache.py

"""
Unified caching system for RPG context management.

This module provides a flexible, multi-level caching system that can be used
across all context-related components.
"""

import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)

class CacheItem:
    """Represents a single cached item with metadata"""
    
    def __init__(
        self, 
        key: str, 
        value: Any, 
        timestamp: float = None,
        importance: float = 0.5
    ):
        self.key = key
        self.value = value
        self.timestamp = timestamp or time.time()
        self.last_access = self.timestamp
        self.access_count = 0
        self.importance = importance  # 0.0 to 1.0
    
    def access(self) -> None:
        """Record an access to this item"""
        self.last_access = time.time()
        self.access_count += 1
    
    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.timestamp
    
    def time_since_access(self) -> float:
        """Get time since last access in seconds"""
        return time.time() - self.last_access
    
    def is_stale(self, ttl: float) -> bool:
        """Check if item is stale based on ttl"""
        return self.age() > ttl
    
    def get_eviction_score(self) -> float:
        """
        Calculate eviction score (higher scores are evicted first)
        Factors: age, access frequency, importance
        """
        # Normalize age to hours
        age_factor = self.age() / 3600
        
        # Less accesses = higher score
        access_factor = 1.0 / (1.0 + self.access_count)
        
        # Normalize time since last access to hours
        recency_factor = self.time_since_access() / 3600
        
        # Combine factors (importance reduces score)
        return (0.4 * age_factor + 0.3 * access_factor + 0.3 * recency_factor) * (1.0 - self.importance)


class UnifiedCache:
    """
    Unified multi-level caching system with intelligent eviction.
    """
    
    def __init__(self):
        """Initialize the cache."""
        # L1: Short-term cache (1 minute TTL, 100 items)
        # L2: Medium-term cache (5 minutes TTL, 500 items)
        # L3: Long-term cache (30 minutes TTL, 2000 items)
        self.l1_cache = {}
        self.l2_cache = {}
        self.l3_cache = {}
        
        self.l1_ttl = 60  # 1 minute
        self.l2_ttl = 300  # 5 minutes
        self.l3_ttl = 1800  # 30 minutes
        
        self.l1_max_size = 100
        self.l2_max_size = 500
        self.l3_max_size = 2000
        
        # Metrics
        self.hits = 0
        self.misses = 0
        
        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def get(
        self, 
        key: str, 
        fetch_func: Callable, 
        cache_level: int = 1,
        importance: float = 0.5,
        ttl_override: Optional[float] = None
    ) -> Any:
        """
        Get an item from cache, or fetch and cache it if not found.
        
        Args:
            key: Cache key
            fetch_func: Async function to call if cache miss
            cache_level: Desired cache level (1-3)
            importance: Importance score (0.0 to 1.0)
            ttl_override: Optional TTL override in seconds
            
        Returns:
            The cached or fetched value
        """
        cache_level = min(max(1, cache_level), 3)
        
        # Try to get from all cache levels
        for level in range(cache_level, 0, -1):
            cache = self._get_cache_by_level(level)
            ttl = ttl_override or self._get_ttl_by_level(level)
            
            if key in cache:
                item = cache[key]
                
                if not item.is_stale(ttl):
                    # Cache hit
                    item.access()
                    self.hits += 1
                    
                    # Promote to higher level if needed
                    self._promote_item(item, level, cache_level)
                    
                    return item.value
                else:
                    # Stale item, remove it
                    del cache[key]
        
        # Cache miss
        self.misses += 1
        
        # Fetch data
        try:
            value = await fetch_func()
        except Exception as e:
            logger.error(f"Error fetching data for key {key}: {e}")
            raise
        
        # Store in cache at requested level
        item = CacheItem(
            key=key,
            value=value,
            importance=importance
        )
        
        cache = self._get_cache_by_level(cache_level)
        cache[key] = item
        
        # Check size limit
        self._check_size_limit(cache_level)
        
        # Maybe run cleanup
        await self._maybe_cleanup()
        
        return value
    
    def _get_cache_by_level(self, level: int) -> Dict[str, CacheItem]:
        """Get cache dictionary by level"""
        if level == 1:
            return self.l1_cache
        elif level == 2:
            return self.l2_cache
        else:
            return self.l3_cache
    
    def _get_ttl_by_level(self, level: int) -> float:
        """Get TTL by level"""
        if level == 1:
            return self.l1_ttl
        elif level == 2:
            return self.l2_ttl
        else:
            return self.l3_ttl
    
    def _get_max_size_by_level(self, level: int) -> int:
        """Get max size by level"""
        if level == 1:
            return self.l1_max_size
        elif level == 2:
            return self.l2_max_size
        else:
            return self.l3_max_size
    
    def _promote_item(self, item: CacheItem, current_level: int, max_level: int) -> None:
        """Promote an item to higher cache levels"""
        # Only promote items with multiple accesses
        if item.access_count < 2:
            return
        
        # Skip if already at highest requested level
        if current_level >= max_level:
            return
        
        # Add to one level higher
        higher_level = min(current_level + 1, 3)
        higher_cache = self._get_cache_by_level(higher_level)
        higher_cache[item.key] = item
        
        # Check size limit
        self._check_size_limit(higher_level)
    
    def _check_size_limit(self, level: int) -> None:
        """Check if a cache level exceeds its size limit and evict if necessary"""
        cache = self._get_cache_by_level(level)
        max_size = self._get_max_size_by_level(level)
        
        if len(cache) > max_size:
            # Calculate how many items to evict (10% of max size)
            evict_count = max(1, int(max_size * 0.1))
            self._evict_items(level, evict_count)
    
    def _evict_items(self, level: int, count: int) -> None:
        """Evict items from a cache level based on eviction scores"""
        cache = self._get_cache_by_level(level)
        
        # Calculate eviction scores
        scores = [(item.get_eviction_score(), key) for key, item in cache.items()]
        
        # Sort by score (higher scores are evicted first)
        scores.sort(reverse=True)
        
        # Evict items
        for _, key in scores[:count]:
            if key in cache:
                del cache[key]
    
    def invalidate(self, key_prefix: Optional[str] = None) -> int:
        """
        Invalidate cache entries by key prefix.
        
        Args:
            key_prefix: Prefix to match (None to invalidate all)
            
        Returns:
            Number of invalidated items
        """
        count = 0
        
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if key_prefix is None:
                # Invalidate all
                count += len(cache)
                cache.clear()
            else:
                # Invalidate by prefix
                keys_to_remove = [k for k in cache if k.startswith(key_prefix)]
                for key in keys_to_remove:
                    del cache[key]
                count += len(keys_to_remove)
        
        return count
    
    async def _maybe_cleanup(self) -> None:
        """Periodically run cleanup operations"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
            
        # Run cleanup asynchronously
        self.last_cleanup = now
        asyncio.create_task(self._cleanup())
    
    async def _cleanup(self) -> None:
        """Remove stale items"""
        start = time.time()
        removed = 0
        
        # Clean L1 cache
        stale_keys = [k for k, item in self.l1_cache.items() if item.is_stale(self.l1_ttl)]
        for key in stale_keys:
            del self.l1_cache[key]
            removed += 1
        
        # Clean L2 cache
        stale_keys = [k for k, item in self.l2_cache.items() if item.is_stale(self.l2_ttl)]
        for key in stale_keys:
            del self.l2_cache[key]
            removed += 1
        
        # Clean L3 cache
        stale_keys = [k for k, item in self.l3_cache.items() if item.is_stale(self.l3_ttl)]
        for key in stale_keys:
            del self.l3_cache[key]
            removed += 1
        
        duration = time.time() - start
        logger.debug(f"Cache cleanup completed in {duration:.3f}s: removed {removed} items")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_items": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
        }


# Singleton instance
context_cache = UnifiedCache()
