# context/unified_cache.py

"""
Unified caching system for RPG context management.

This module provides a flexible, multi-level caching system that can be used
across all context-related components. Refactored to integrate with the OpenAI Agents SDK.
"""

import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Callable, List, Union
import asyncio

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper, trace, custom_span
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class CacheItemModel(BaseModel):
    """Model for a cache item"""
    key: str
    value: Any
    timestamp: float = Field(default_factory=time.time)
    last_access: float = Field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0


class CacheStatsModel(BaseModel):
    """Model for cache statistics"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    l1_size: int = 0
    l2_size: int = 0
    l3_size: int = 0
    total_items: int = 0


class CacheOperationRequest(BaseModel):
    """Model for cache operations"""
    key: str
    cache_level: int = 1
    importance: float = 0.5
    ttl_override: Optional[int] = None


class CacheInvalidateRequest(BaseModel):
    """Model for cache invalidation"""
    key_prefix: Optional[str] = None


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
    
    def to_model(self) -> CacheItemModel:
        """Convert to pydantic model"""
        return CacheItemModel(
            key=self.key,
            value=self.value,
            timestamp=self.timestamp,
            last_access=self.last_access,
            access_count=self.access_count,
            importance=self.importance
        )


class UnifiedCache:
    """
    Unified multi-level caching system with intelligent eviction.
    Refactored to integrate with the OpenAI Agents SDK.
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
        
        # Cleanup task
        self._cleanup_task = None
        
        # Start background cleanup
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
    
    async def _background_cleanup(self):
        """Run cleanup periodically in the background"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup()
        except asyncio.CancelledError:
            # Task was cancelled
            logger.debug("Cache cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
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
        with trace(workflow_name="cache.get"):
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

    async def get_item(
        self, 
        ctx: RunContextWrapper,
        request: CacheOperationRequest,
        fetch_func: Optional[Callable] = None
    ) -> Any:
        """
        Get an item from cache
        
        Args:
            request: Cache operation request
            fetch_func: Function to fetch data if not in cache (if None, only check cache)
            
        Returns:
            The cached value or None if not found
        """
        with custom_span("cache_get_item"):
            cache_level = min(max(1, request.cache_level), 3)
            
            # Try to get from all cache levels
            for level in range(cache_level, 0, -1):
                cache = self._get_cache_by_level(level)
                ttl = request.ttl_override or self._get_ttl_by_level(level)
                
                if request.key in cache:
                    item = cache[request.key]
                    
                    if not item.is_stale(ttl):
                        # Cache hit
                        item.access()
                        self.hits += 1
                        
                        # Promote to higher level if needed
                        self._promote_item(item, level, cache_level)
                        
                        return item.value
                    else:
                        # Stale item, remove it
                        del cache[request.key]
            
            # Cache miss
            self.misses += 1
            
            # If no fetch function, return None
            if fetch_func is None:
                return None
            
            # Fetch data
            try:
                value = await fetch_func()
            except Exception as e:
                logger.error(f"Error fetching data for key {request.key}: {e}")
                raise
            
            # Store in cache at requested level
            item = CacheItem(
                key=request.key,
                value=value,
                importance=request.importance
            )
            
            cache = self._get_cache_by_level(cache_level)
            cache[request.key] = item
            
            # Check size limit
            self._check_size_limit(cache_level)
            
            # Maybe run cleanup
            await self._maybe_cleanup()
            
            return value
    
    @function_tool
    async def set_item(
        self, 
        ctx: RunContextWrapper,
        request: CacheOperationRequest,
        value: Any
    ) -> bool:
        """
        Set an item in the cache
        
        Args:
            request: Cache operation request
            value: Value to cache
            
        Returns:
            Success status
        """
        with custom_span("cache_set_item"):
            cache_level = min(max(1, request.cache_level), 3)
            
            # Store in cache at requested level
            item = CacheItem(
                key=request.key,
                value=value,
                importance=request.importance
            )
            
            cache = self._get_cache_by_level(cache_level)
            cache[request.key] = item
            
            # Check size limit
            self._check_size_limit(cache_level)
            
            return True
    
    @function_tool
    async def invalidate(
        self, 
        ctx: RunContextWrapper,
        request: CacheInvalidateRequest
    ) -> int:
        """
        Invalidate cache entries by key prefix.
        
        Args:
            request: Cache invalidation request
            
        Returns:
            Number of invalidated items
        """
        with custom_span("cache_invalidate"):
            count = 0
            
            for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
                if request.key_prefix is None:
                    # Invalidate all
                    count += len(cache)
                    cache.clear()
                else:
                    # Invalidate by prefix
                    keys_to_remove = [k for k in cache if k.startswith(request.key_prefix)]
                    for key in keys_to_remove:
                        del cache[key]
                    count += len(keys_to_remove)
            
            return count
    
    @function_tool
    async def get_stats(self, ctx: RunContextWrapper) -> CacheStatsModel:
        """
        Get cache performance metrics
        
        Returns:
            Cache statistics
        """
        with custom_span("cache_get_stats"):
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return CacheStatsModel(
                hits=self.hits,
                misses=self.misses,
                hit_rate=hit_rate,
                l1_size=len(self.l1_cache),
                l2_size=len(self.l2_cache),
                l3_size=len(self.l3_cache),
                total_items=len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
            )
    
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
        with custom_span("cache_cleanup"):
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


def create_cache_agent() -> Agent:
    """Create a cache agent using the OpenAI Agents SDK"""
    # Get the singleton instance
    cache = context_cache
    
    # Define the agent
    agent = Agent(
        name="Cache Manager",
        instructions="""
        You are a cache manager agent specialized in caching operations.
        Your tasks include:
        
        1. Getting items from cache
        2. Setting items in cache
        3. Invalidating cache entries
        4. Getting cache statistics
        
        When handling cache operations, prioritize efficiency and performance.
        """,
        tools=[
            cache.get_item,
            cache.set_item,
            cache.invalidate,
            cache.get_stats,
        ],
    )
    
    return agent


# Singleton instance
context_cache = UnifiedCache()

def get_cache_agent() -> Agent:
    """Get the cache agent"""
    return create_cache_agent()
