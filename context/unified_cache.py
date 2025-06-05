# context/unified_cache.py

"""
Unified caching system for RPG context management.

This module provides a flexible, multi-level caching system that can be used
across all context-related components. Refactored to integrate with the OpenAI Agents SDK.
"""

import time
import logging
import json
import asyncio
import hashlib
from typing import Dict, Any, Optional, Callable, List, Union

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
        
        self.l1_ttl = 60   # 1 minute
        self.l2_ttl = 300  # 5 minutes
        self.l3_ttl = 1800 # 30 minutes
        
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
    
    async def start_background_cleanup(self):
        """Explicitly start the background cleanup task once an event loop is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            logger.debug("Starting background cleanup task for UnifiedCache.")
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
        else:
            logger.debug("Background cleanup task is already running.")
    
    async def _background_cleanup(self):
        """Run cleanup periodically in the background."""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup()
        except asyncio.CancelledError:
            # Task was cancelled
            logger.debug("Cache cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
    async def _cleanup(self) -> None:
        """Remove stale items (triggered periodically)"""
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

    async def _maybe_cleanup(self) -> None:
        """Periodically run cleanup operations."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        # Manually trigger cleanup
        self.last_cleanup = now
        asyncio.create_task(self._cleanup())

    async def get(
        self,
        key: str,
        fetch_func: Callable,
        cache_level: int = 1,
        importance: float = 0.5,
        ttl_override: Optional[int] = None
    ) -> Any:
        """Get an item from cache or fetch it if not present."""
        request = CacheOperationRequest(
            key=key,
            cache_level=cache_level,
            importance=importance,
            ttl_override=ttl_override
        )
        return await self._get_item(request, fetch_func)
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_level: int = 1,
        importance: float = 0.5,
        ttl_override: Optional[int] = None
    ) -> bool:
        """Set an item in the cache."""
        request = CacheOperationRequest(
            key=key,
            cache_level=cache_level,
            importance=importance,
            ttl_override=ttl_override
        )
        return await self._set_item(request, value)
    
    async def delete(
        self,
        key_prefix: Optional[str] = None
    ) -> int:
        """Delete cache entries matching the key prefix."""
        request = CacheInvalidateRequest(key_prefix=key_prefix)
        return await self._invalidate(request)

    def _get_cache_by_level(self, level: int) -> Dict[str, "CacheItem"]:
        """Get cache dictionary by level."""
        if level == 1:
            return self.l1_cache
        elif level == 2:
            return self.l2_cache
        else:
            return self.l3_cache
    
    def _get_ttl_by_level(self, level: int) -> float:
        """Get TTL by level."""
        if level == 1:
            return self.l1_ttl
        elif level == 2:
            return self.l2_ttl
        else:
            return self.l3_ttl
    
    def _get_max_size_by_level(self, level: int) -> int:
        """Get max size by level."""
        if level == 1:
            return self.l1_max_size
        elif level == 2:
            return self.l2_max_size
        else:
            return self.l3_max_size
    
    def _promote_item(self, item: "CacheItem", current_level: int, max_level: int) -> None:
        """Promote an item to higher cache levels if it has enough accesses."""
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
        """Check if a cache level exceeds its size limit and evict if necessary."""
        cache = self._get_cache_by_level(level)
        max_size = self._get_max_size_by_level(level)
        
        if len(cache) > max_size:
            # Calculate how many items to evict (10% of max size)
            evict_count = max(1, int(max_size * 0.1))
            self._evict_items(level, evict_count)
    
    def _evict_items(self, level: int, count: int) -> None:
        """Evict items from a cache level based on eviction scores."""
        cache = self._get_cache_by_level(level)
        
        # Calculate eviction scores
        scores = [(item.get_eviction_score(), key) for key, item in cache.items()]
        
        # Sort by score (higher scores are evicted first)
        scores.sort(reverse=True)
        
        # Evict items
        for _, key in scores[:count]:
            if key in cache:
                del cache[key]
    
    # ---------------------------------------------------------------------
    #           INTERNAL METHODS (no @function_tool, have `self`)
    # ---------------------------------------------------------------------
    
    async def _get_item(
        self,
        request: CacheOperationRequest,
        fetch_func: Optional[Callable] = None
    ) -> Any:
        """Internal method to get or fetch an item from cache (no run_context)."""
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
            
            target_cache = self._get_cache_by_level(cache_level)
            target_cache[request.key] = item
            
            # Check size limit
            self._check_size_limit(cache_level)
            
            # Maybe run cleanup
            await self._maybe_cleanup()
            
            return value

    async def _set_item(
        self,
        request: CacheOperationRequest,
        value: Any
    ) -> bool:
        """Internal method to set an item in the cache."""
        with custom_span("cache_set_item"):
            cache_level = min(max(1, request.cache_level), 3)
            
            item = CacheItem(
                key=request.key,
                value=value,
                importance=request.importance
            )
            
            cache = self._get_cache_by_level(cache_level)
            cache[request.key] = item
            
            self._check_size_limit(cache_level)
            return True

    async def _invalidate(
        self,
        request: CacheInvalidateRequest
    ) -> int:
        """Internal method to invalidate cache entries (no run_context)."""
        with custom_span("cache_invalidate"):
            count = 0
            
            for cache_dict in [self.l1_cache, self.l2_cache, self.l3_cache]:
                if request.key_prefix is None:
                    # Invalidate all
                    count += len(cache_dict)
                    cache_dict.clear()
                else:
                    # Invalidate by prefix
                    keys_to_remove = [k for k in cache_dict if k.startswith(request.key_prefix)]
                    for key in keys_to_remove:
                        del cache_dict[key]
                    count += len(keys_to_remove)
            
            return count

    async def _get_stats(self) -> CacheStatsModel:
        """Internal method to get cache performance metrics."""
        with custom_span("cache_get_stats"):
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests else 0
            
            return CacheStatsModel(
                hits=self.hits,
                misses=self.misses,
                hit_rate=hit_rate,
                l1_size=len(self.l1_cache),
                l2_size=len(self.l2_cache),
                l3_size=len(self.l3_cache),
                total_items=len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
            )


# ---------------------------------------------------------------------
#         STANDALONE TOOL FUNCTIONS (run_context is first param)
# ---------------------------------------------------------------------

context_cache = UnifiedCache()  # The singleton instance

@function_tool(strict=False)
async def get_item_tool(
    ctx: RunContextWrapper,
    request: CacheOperationRequest
) -> Any:
    """
    Tool: get or fetch an item from the cache.
    """
    return await context_cache._get_item(request)

@function_tool(strict=False)
async def set_item_tool(
    ctx: RunContextWrapper,
    request: CacheOperationRequest,
    value: Any
) -> bool:
    """
    Tool: set an item in the cache.
    """
    return await context_cache._set_item(request, value)


@function_tool
async def invalidate_tool(
    ctx: RunContextWrapper,
    request: CacheInvalidateRequest
) -> int:
    """
    Tool: invalidate cache entries by key prefix.
    """
    return await context_cache._invalidate(request)


@function_tool
async def get_stats_tool(
    ctx: RunContextWrapper
) -> CacheStatsModel:
    """
    Tool: get cache statistics.
    """
    return await context_cache._get_stats()


def create_cache_agent() -> Agent:
    """Create a cache agent using the OpenAI Agents SDK."""
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
            get_item_tool,
            set_item_tool,
            invalidate_tool,
            get_stats_tool,
        ],
    )
    
    return agent


def get_cache_agent() -> Agent:
    """Get the singleton cache agent."""
    return create_cache_agent()
