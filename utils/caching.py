# utils/caching.py

import time
import logging
import threading
import asyncio  # Added missing import
from typing import Dict, Any, Optional, Callable, Tuple, Union, List, Set
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime
import json
import hashlib
import zlib
from prometheus_client import Counter, Histogram, Gauge
import redis

# Configure logging
logger = logging.getLogger(__name__)

# Prometheus metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Number of cache hits',
    ['cache_level']
)
CACHE_MISSES = Counter(
    'cache_misses_total',
    'Number of cache misses',
    ['cache_level']
)
CACHE_SIZE = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_level']
)
CACHE_EVICTIONS = Counter(
    'cache_evictions_total',
    'Number of cache evictions',
    ['cache_level', 'reason']
)
CACHE_LATENCY = Histogram(
    'cache_operation_latency_seconds',
    'Cache operation latency in seconds',
    ['operation']
)

@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    tags: Set[str]

class EvictionPolicy:
    """Base class for cache eviction policies."""
    
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        """Select items to evict to free up required space."""
        raise NotImplementedError

class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        sorted_items = sorted(items.items(), key=lambda x: x[1].last_accessed)
        to_evict = []
        freed_space = 0
        
        for key, item in sorted_items:
            if freed_space >= required_space:
                break
            to_evict.append(key)
            freed_space += item.size_bytes
        
        return to_evict

class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        sorted_items = sorted(items.items(), key=lambda x: x[1].access_count)
        to_evict = []
        freed_space = 0
        
        for key, item in sorted_items:
            if freed_space >= required_space:
                break
            to_evict.append(key)
            freed_space += item.size_bytes
        
        return to_evict

class TTLEvictionPolicy(EvictionPolicy):
    """Time To Live eviction policy."""
    
    def __init__(self, ttl: float):
        self.ttl = ttl
    
    def select_items_to_evict(self, items: Dict[str, CacheItem], required_space: int) -> List[str]:
        now = time.time()
        expired = []
        to_evict = []
        freed_space = 0
        
        # First, remove expired items
        for key, item in items.items():
            if now - item.created_at > self.ttl:
                expired.append(key)
                freed_space += item.size_bytes
        
        if freed_space >= required_space:
            return expired
        
        # If we still need space, use LRU for remaining items
        remaining_space = required_space - freed_space
        remaining_items = {k: v for k, v in items.items() if k not in expired}
        lru = LRUEvictionPolicy()
        additional_evictions = lru.select_items_to_evict(remaining_items, remaining_space)
        
        return expired + additional_evictions

class EnhancedCache:
    """Enhanced multi-level cache with adaptive TTLs and predictive prefetching."""
    
    def __init__(
        self,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None,
        eviction_policy: Optional[EvictionPolicy] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        
        # Cache levels
        self.l1_cache = OrderedDict()  # Very short-lived (seconds)
        self.l2_cache = OrderedDict()  # Medium-lived (minutes)
        self.l3_cache = OrderedDict()  # Long-lived (hours)
        
        # TTL settings
        self.l1_ttl = 60  # 1 minute
        self.l2_ttl = 300  # 5 minutes
        self.l3_ttl = 3600  # 1 hour
        
        # Redis connection for distributed caching
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Eviction policy
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        
        # Access patterns for prefetching
        self.access_patterns = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
        
        # Background tasks
        self.maintenance_task = None
        self.prefetch_task = None
    
    async def start(self):
        """Start background tasks."""
        if self.maintenance_task is None:
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        if self.prefetch_task is None:
            self.prefetch_task = asyncio.create_task(self._prefetch_loop())
    
    async def stop(self):
        """Stop background tasks."""
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass
    
    async def get(self, key: str, level: str = 'l1') -> Optional[Any]:
        """Get item from cache."""
        start_time = time.time()
        try:
            cache = getattr(self, f"{level}_cache")
            
            if key in cache:
                item = cache[key]
                # Update access metadata
                item.last_accessed = time.time()
                item.access_count += 1
                # Move to end (LRU)
                cache.move_to_end(key)
                
                CACHE_HITS.labels(cache_level=level).inc()
                self.stats['hits'] += 1
                
                # Record access pattern
                self._record_access_pattern(key)
                
                return item.value
            
            # Try Redis if available
            if self.redis_client:
                value = self.redis_client.get(f"{level}:{key}")
                if value:
                    # Cache locally
                    await self.set(key, json.loads(value), level)
                    return json.loads(value)
            
            CACHE_MISSES.labels(cache_level=level).inc()
            self.stats['misses'] += 1
            return None
            
        finally:
            CACHE_LATENCY.labels(operation='get').observe(time.time() - start_time)
    
    async def set(
        self,
        key: str,
        value: Any,
        level: str = 'l1',
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set item in cache."""
        start_time = time.time()
        try:
            cache = getattr(self, f"{level}_cache")
            
            # Calculate item size
            size_bytes = len(json.dumps(value).encode())
            
            # Check if we need to evict items
            if self.current_size_bytes + size_bytes > self.max_size_bytes:
                await self._evict_items(size_bytes, level)
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                tags=tags or set()
            )
            
            # Update cache
            cache[key] = item
            self.current_size_bytes += size_bytes
            
            # Update Redis if available
            if self.redis_client:
                ttl = getattr(self, f"{level}_ttl")
                self.redis_client.setex(
                    f"{level}:{key}",
                    ttl,
                    json.dumps(value)
                )
            
            # Update metrics
            CACHE_SIZE.labels(cache_level=level).set(self.current_size_bytes)
            
            return True
            
        finally:
            CACHE_LATENCY.labels(operation='set').observe(time.time() - start_time)
    
    async def invalidate(self, key: str, level: str = None):
        """Invalidate cache item."""
        start_time = time.time()
        try:
            if level:
                levels = [level]
            else:
                levels = ['l1', 'l2', 'l3']
            
            for l in levels:
                cache = getattr(self, f"{l}_cache")
                if key in cache:
                    item = cache.pop(key)
                    self.current_size_bytes -= item.size_bytes
                
                # Invalidate in Redis
                if self.redis_client:
                    self.redis_client.delete(f"{l}:{key}")
            
            # Update metrics
            for l in levels:
                CACHE_SIZE.labels(cache_level=l).set(self.current_size_bytes)
                
        finally:
            CACHE_LATENCY.labels(operation='invalidate').observe(time.time() - start_time)
    
    async def _evict_items(self, required_space: int, level: str):
        """Evict items to free up space."""
        cache = getattr(self, f"{level}_cache")
        
        # Get items to evict
        to_evict = self.eviction_policy.select_items_to_evict(
            cache,
            required_space
        )
        
        # Evict items
        for key in to_evict:
            item = cache.pop(key)
            self.current_size_bytes -= item.size_bytes
            
            # Remove from Redis
            if self.redis_client:
                self.redis_client.delete(f"{level}:{key}")
            
            CACHE_EVICTIONS.labels(
                cache_level=level,
                reason='size'
            ).inc()
            self.stats['evictions'] += 1
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for prefetching."""
        now = time.time()
        if not hasattr(self, '_last_access'):
            self._last_access = (None, 0)
        
        last_key, last_time = self._last_access
        
        if last_key and (now - last_time) < 5:  # Within 5 seconds
            # Record pattern
            pattern = (last_key, key)
            self.access_patterns[pattern] = self.access_patterns.get(pattern, 0) + 1
        
        self._last_access = (key, now)
    
    async def _maintenance_loop(self):
        """Background task for cache maintenance."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up expired items
                now = time.time()
                for level in ['l1', 'l2', 'l3']:
                    cache = getattr(self, f"{level}_cache")
                    ttl = getattr(self, f"{level}_ttl")
                    
                    expired = [
                        key for key, item in cache.items()
                        if now - item.created_at > ttl
                    ]
                    
                    for key in expired:
                        await self.invalidate(key, level)
                        CACHE_EVICTIONS.labels(
                            cache_level=level,
                            reason='ttl'
                        ).inc()
                
                # Clean up access patterns
                old_patterns = [
                    pattern for pattern, count in self.access_patterns.items()
                    if count < 3  # Remove patterns with low frequency
                ]
                for pattern in old_patterns:
                    del self.access_patterns[pattern]
                
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    async def _prefetch_loop(self):
        """Background task for predictive prefetching."""
        while True:
            try:
                await asyncio.sleep(1)  # Check frequently
                
                # Get current access patterns
                patterns = self.access_patterns.copy()
                
                for (key1, key2), count in patterns.items():
                    if count >= 3:  # Only prefetch frequently occurring patterns
                        # If key1 was recently accessed, prefetch key2
                        cache = self.l1_cache  # Use L1 cache for checking
                        if key1 in cache:
                            item = cache[key1]
                            if time.time() - item.last_accessed < 5:  # Within 5 seconds
                                # Prefetch key2 if not already cached
                                if key2 not in cache:
                                    # Trigger prefetch (implementation depends on your data source)
                                    logger.debug(f"Prefetching {key2} based on access pattern")
                
            except Exception as e:
                logger.error(f"Error in prefetch loop: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self.stats,
            'size_mb': self.current_size_bytes / (1024 * 1024),
            'items': {
                'l1': len(self.l1_cache),
                'l2': len(self.l2_cache),
                'l3': len(self.l3_cache)
            },
            'patterns': len(self.access_patterns)
        }

class MemoryCache:
    """
    Enhanced cache with TTL and size management.
    Thread-safe implementation with statistics tracking.
    """
    def __init__(self, name: str = "default", max_size: int = 100, default_ttl: int = 60):
        """
        Initialize cache.
        
        Args:
            name: Cache name for logging
            max_size: Maximum number of entries before eviction
            default_ttl: Default time-to-live in seconds
        """
        self.name = name
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, int] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = threading.RLock()
        self.last_cleanup = time.time()
        # Track memory usage
        self.estimated_memory_usage = 0
        logging.info(f"Initialized {name} cache with max_size={max_size}, ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
                
            # Check if expired
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                return None
                
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional custom TTL."""
        with self.lock:
            # Handle eviction if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()
                
            # Estimate memory usage (rough approximation)
            try:
                import sys
                value_size = sys.getsizeof(value)
                key_size = sys.getsizeof(key)
                
                # Update memory tracking
                if key in self.cache:
                    old_value_size = sys.getsizeof(self.cache[key])
                    self.estimated_memory_usage += (value_size - old_value_size)
                else:
                    self.estimated_memory_usage += (key_size + value_size)
            except:
                # Fallback if we can't estimate size
                self.estimated_memory_usage = 0
                
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.ttls[key] = ttl if ttl is not None else self.default_ttl
            
            # Run periodic cleanup if needed
            self._maybe_cleanup()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a key is expired."""
        timestamp = self.timestamps.get(key, 0)
        ttl = self.ttls.get(key, self.default_ttl)
        return time.time() - timestamp > ttl
    
    def _remove(self, key: str) -> None:
        """Remove a key from the cache."""
        if key in self.cache:
            # Update memory tracking
            try:
                import sys
                self.estimated_memory_usage -= (sys.getsizeof(key) + sys.getsizeof(self.cache[key]))
            except:
                pass
                
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.ttls:
            del self.ttls[key]
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self.timestamps:
            return
            
        # Find oldest key
        oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
        self._remove(oldest_key)
        self.evictions += 1
        
        if self.evictions % 100 == 0:
            logging.info(f"{self.name} cache: {self.evictions} total evictions")
    
    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        # Only run cleanup every 60 seconds
        now = time.time()
        if now - self.last_cleanup < 60:
            return
            
        self.last_cleanup = now
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        
        for key in expired_keys:
            self._remove(key)
            
        if expired_keys:
            logging.debug(f"{self.name} cache: cleaned up {len(expired_keys)} expired entries")
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.estimated_memory_usage = 0
            logging.info(f"{self.name} cache cleared")
    
    def remove_pattern(self, pattern: str) -> int:
        """Remove all keys matching a pattern."""
        with self.lock:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                self._remove(key)
            return len(keys_to_remove)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "evictions": self.evictions,
                "memory_usage_bytes": self.estimated_memory_usage,
                "memory_usage_mb": self.estimated_memory_usage / (1024 * 1024) if self.estimated_memory_usage > 0 else 0,
                "keys": list(self.cache.keys())
            }

# Create frequently used cache instances with configuration from environment
from os import environ

# Config-driven cache settings
NPC_CACHE_SIZE = int(environ.get("NPC_CACHE_SIZE", "50"))
NPC_CACHE_TTL = int(environ.get("NPC_CACHE_TTL", "30"))
LOCATION_CACHE_SIZE = int(environ.get("LOCATION_CACHE_SIZE", "20"))
LOCATION_CACHE_TTL = int(environ.get("LOCATION_CACHE_TTL", "120"))
AGGREGATOR_CACHE_SIZE = int(environ.get("AGGREGATOR_CACHE_SIZE", "10"))
AGGREGATOR_CACHE_TTL = int(environ.get("AGGREGATOR_CACHE_TTL", "15"))
TIME_CACHE_SIZE = int(environ.get("TIME_CACHE_SIZE", "5"))
TIME_CACHE_TTL = int(environ.get("TIME_CACHE_TTL", "10"))
COMPUTATION_CACHE_SIZE = int(environ.get("COMPUTATION_CACHE_SIZE", "50"))
COMPUTATION_CACHE_TTL = int(environ.get("COMPUTATION_CACHE_TTL", "300"))

# Create cache instances
NPC_CACHE = MemoryCache(name="npc", max_size=NPC_CACHE_SIZE, default_ttl=NPC_CACHE_TTL)
LOCATION_CACHE = MemoryCache(name="location", max_size=LOCATION_CACHE_SIZE, default_ttl=LOCATION_CACHE_TTL)
AGGREGATOR_CACHE = MemoryCache(name="aggregator", max_size=AGGREGATOR_CACHE_SIZE, default_ttl=AGGREGATOR_CACHE_TTL)
TIME_CACHE = MemoryCache(name="time", max_size=TIME_CACHE_SIZE, default_ttl=TIME_CACHE_TTL)


class ComputationCache:
    """Cache for expensive computations with function-based keys."""
    
    def __init__(self, name: str = "computation", max_size: int = 50, default_ttl: int = 300):
        """Initialize computation cache."""
        self.memory_cache = MemoryCache(name=name, max_size=max_size, default_ttl=default_ttl)
    
    async def cached_call(self, func: Callable, *args, ttl: Optional[int] = None, **kwargs) -> Any:
        """
        Call a function with caching based on function name and arguments.
        
        Args:
            func: Function to call
            *args: Positional arguments
            ttl: Optional custom TTL
            **kwargs: Keyword arguments
            
        Returns:
            Function result (from cache if available)
        """
        # Create cache key from function name and arguments
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        cache_key = ":".join(key_parts)
        
        # Check cache
        cached_result = self.memory_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Call function
        start_time = time.time()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        # Cache result
        self.memory_cache.set(cache_key, result, ttl)
        
        # Log for expensive operations
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Log if took more than 100ms
            logging.info(f"Cached computation for {func.__name__} took {elapsed:.3f}s")
        
        return result

# Global computation cache
COMPUTATION_CACHE = ComputationCache(
    max_size=COMPUTATION_CACHE_SIZE,
    default_ttl=COMPUTATION_CACHE_TTL
)

# Usage examples

# Create a context cache for NPCs with 50MB max size
npc_cache = EnhancedCache(max_size_mb=50)

async def get_npc_data(user_id, conv_id, npc_id):
    """Example of using the enhanced cache"""
    cache_key = f"npc:{user_id}:{conv_id}:{npc_id}"
    
    async def fetch_npc_from_db():
        # This would be your actual database query
        return await fetch_detailed_npc_data(user_id, conv_id, npc_id)
    
    # Related keys for smart invalidation (when we update this NPC, we might need to invalidate these)
    related_keys = [
        f"npc_list:{user_id}:{conv_id}",
        f"location:{user_id}:{conv_id}:{npc_location}"
    ]
    
    # Get from cache with smart prefetching and adaptive TTLs
    return await npc_cache.get(
        cache_key,
        fetch_npc_from_db,
        level='l3'  # Store in long-lived cache
    )

async def update_npc_location(user_id, conv_id, npc_id, new_location):
    """Example of smart cache invalidation"""
    # After updating the NPC in the database
    
    # Invalidate the specific NPC
    cache_key = f"npc:{user_id}:{conv_id}:{npc_id}"
    await npc_cache.invalidate(key=cache_key, level='l3')
    
    # Invalidate the location's NPC list
    location_key = f"location:{user_id}:{conv_id}:{new_location}"
    await npc_cache.invalidate(key=location_key, level='l3')

# Start background maintenance
async def startup():
    await npc_cache.start()

# Stop maintenance on shutdown
async def shutdown():
    await npc_cache.stop()

class Cache:
    """
    A simple caching system for storing and retrieving data.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
        self._size = 0
        
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Key to look up
            
        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Key to store under
            value: Value to store
        """
        # If cache is full, remove least recently used item
        if self._size >= self.max_size and key not in self._cache:
            self._remove_lru()
            
        self._cache[key] = value
        self._access_times[key] = time.time()
        if key not in self._cache:
            self._size += 1
            
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Key to delete
        """
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            self._size -= 1
            
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        self._access_times.clear()
        self._size = 0
        
    def _remove_lru(self) -> None:
        """Remove the least recently used item from the cache."""
        if not self._access_times:
            return
            
        # Find key with oldest access time
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self.delete(lru_key)
        
    @property
    def size(self) -> int:
        """Get current cache size."""
        return self._size
