# utils/caching.py

import time
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, Callable, Tuple, Union, List, Set
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime
import json
import hashlib
import zlib
import sys # Import sys for getsizeof

from prometheus_client import Counter, Histogram, Gauge
# Ensure redis is imported if used by EnhancedCache
try:
    import redis
except ImportError:
    redis = None # Handle cases where redis might not be installed

# Configure logging
name = __name__  # Define name for logger
logger = logging.getLogger(name)

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
        freed_space = 0

        # First, identify expired items based on creation time and TTL
        for key, item in items.items():
            if now - item.created_at > self.ttl:
                expired.append(key)
                freed_space += item.size_bytes

        if freed_space >= required_space:
            return expired # Return only expired items if enough space is freed

        # If more space is needed, use LRU on the remaining (non-expired) items
        remaining_space = required_space - freed_space
        remaining_items = {k: v for k, v in items.items() if k not in expired}

        if not remaining_items: # No non-expired items left
            return expired

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
        logger.info(f"Initializing EnhancedCache (max_size={max_size_mb}MB, redis={'yes' if redis_url else 'no'})...")
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0

        # Cache levels (using OrderedDict for LRU behavior within levels if needed)
        self.l1_cache = OrderedDict()  # In-memory short-term
        # self.l2_cache = OrderedDict() # Example: could add more levels
        # self.l3_cache = OrderedDict() # Example: could add more levels

        # Simplified TTL for this example (could be per-level)
        self.default_ttl = 300 # 5 minutes

        # Redis connection for distributed caching
        self.redis_client = None
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True) # decode for easier handling
                self.redis_client.ping()
                logger.info("Redis client connected successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
                self.redis_client = None
        else:
            logger.info("Redis client not configured or 'redis' library not installed.")


        # Eviction policy (default to LRU if none provided)
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()

        # Access patterns for prefetching (simplified)
        self.access_patterns = {}
        self._last_access: Tuple[Optional[str], float] = (None, 0)


        # Cache statistics
        self.stats = {
            'hits_l1': 0,
            'hits_redis': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0 # Tracks only L1 size for simplicity here
        }

        # Background tasks
        self.maintenance_task = None
        self.prefetch_task = None
        self.lock = asyncio.Lock() # Use asyncio lock for async methods
        logger.info("EnhancedCache initialized.")


    async def start(self):
        """Start background tasks."""
        logger.info("Starting EnhancedCache background tasks...")
        if self.maintenance_task is None or self.maintenance_task.done():
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.info("Maintenance task started.")
        # Prefetching loop can be added similarly if needed
        # if self.prefetch_task is None or self.prefetch_task.done():
        #     self.prefetch_task = asyncio.create_task(self._prefetch_loop())
        #     logger.info("Prefetch task started.")

    async def stop(self):
        """Stop background tasks."""
        logger.info("Stopping EnhancedCache background tasks...")
        tasks_to_cancel = []
        if self.maintenance_task and not self.maintenance_task.done():
             tasks_to_cancel.append(self.maintenance_task)
        # if self.prefetch_task and not self.prefetch_task.done():
        #     tasks_to_cancel.append(self.prefetch_task)

        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {task.get_name()} cancelled.")
            except Exception as e:
                logger.error(f"Error during task {task.get_name()} cancellation: {e}")
        self.maintenance_task = None
        # self.prefetch_task = None
        logger.info("EnhancedCache background tasks stopped.")


    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (L1 -> Redis)."""
        start_time = time.time()
        cache_level_hit = None
        try:
            async with self.lock:
                # Check L1
                if key in self.l1_cache:
                    item = self.l1_cache[key]
                    # Check L1 TTL (optional, can rely on maintenance)
                    if time.time() - item.created_at <= self.default_ttl: # Simple TTL check
                        item.last_accessed = time.time()
                        item.access_count += 1
                        self.l1_cache.move_to_end(key) # Move to end for LRU behavior within L1
                        CACHE_HITS.labels(cache_level='l1').inc()
                        self.stats['hits_l1'] += 1
                        self._record_access_pattern(key)
                        cache_level_hit = 'l1'
                        return item.value
                    else:
                        # Expired in L1, remove it
                        logger.debug(f"Item {key} expired in L1 cache.")
                        await self._evict_items_internal([key], 'l1', reason='ttl')


            # Check Redis (if L1 miss or expired)
            if self.redis_client:
                try:
                    value_str = await asyncio.to_thread(self.redis_client.get, key) # Use to_thread for sync redis client
                    if value_str:
                        logger.debug(f"Cache hit for {key} in Redis.")
                        value = json.loads(value_str) # Assuming JSON storage
                        # Add to L1 cache
                        await self.set(key, value, ttl=self.default_ttl) # Use default TTL when loading from Redis
                        CACHE_HITS.labels(cache_level='redis').inc()
                        self.stats['hits_redis'] += 1
                        cache_level_hit = 'redis'
                        return value
                except Exception as e:
                    logger.error(f"Redis GET error for key {key}: {e}")


            # Missed all levels
            logger.debug(f"Cache miss for key {key}.")
            CACHE_MISSES.labels(cache_level='all').inc() # Use a general label or specific levels checked
            self.stats['misses'] += 1
            return None

        finally:
            op_latency = time.time() - start_time
            CACHE_LATENCY.labels(operation='get').observe(op_latency)
            logger.debug(f"Get operation for key '{key}' took {op_latency:.4f}s. Hit: {cache_level_hit or 'None'}")


    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache (L1 and Redis)."""
        start_time = time.time()
        ttl = ttl if ttl is not None else self.default_ttl
        try:
            # Calculate item size (approximate)
            try:
                value_str = json.dumps(value)
                size_bytes = len(value_str.encode('utf-8'))
            except (TypeError, OverflowError) as e:
                logger.warning(f"Could not serialize value for key '{key}' to calculate size: {e}. Using size 0.")
                size_bytes = 0
                value_str = None # Cannot store if not serializable

            async with self.lock:
                # --- L1 Cache Handling ---
                # Check if update or new item
                existing_item = self.l1_cache.get(key)
                size_diff = size_bytes - (existing_item.size_bytes if existing_item else 0)

                # Check if we need to evict items from L1
                if self.current_size_bytes + size_diff > self.max_size_bytes:
                    required_space = (self.current_size_bytes + size_diff) - self.max_size_bytes
                    logger.info(f"L1 cache full. Need to evict {required_space} bytes for key '{key}'.")
                    await self._evict_items_internal(None, 'l1', reason='size', required_space=required_space)

                    # Re-check if enough space was freed
                    if self.current_size_bytes + size_diff > self.max_size_bytes:
                        logger.warning(f"Could not free enough space in L1 cache for key '{key}'. Item not added to L1.")
                        # Optionally, still try to set in Redis
                        # For now, we return False if L1 fails due to size
                        # return False # Decide policy: fail if L1 fails, or just skip L1?

                # Create or update L1 cache item
                now = time.time()
                item = CacheItem(
                    key=key,
                    value=value,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    size_bytes=size_bytes,
                    tags=set() # Add tag support if needed
                )

                # Update L1 cache
                if existing_item:
                    self.current_size_bytes -= existing_item.size_bytes # Remove old size
                self.l1_cache[key] = item
                self.current_size_bytes += size_bytes # Add new size
                self.l1_cache.move_to_end(key) # Mark as recently used

                # Update L1 metrics
                self.stats['size_bytes'] = self.current_size_bytes
                CACHE_SIZE.labels(cache_level='l1').set(self.current_size_bytes)
                logger.debug(f"Set key '{key}' in L1 cache. New L1 size: {self.current_size_bytes} bytes.")


            # --- Redis Cache Handling ---
            if self.redis_client and value_str is not None: # Only if serializable
                try:
                    # Use SETEX for key, ttl, value
                    await asyncio.to_thread(self.redis_client.setex, key, ttl, value_str)
                    logger.debug(f"Set key '{key}' in Redis with TTL {ttl}s.")
                except Exception as e:
                    logger.error(f"Redis SETEX error for key {key}: {e}")
                    # Decide if failure here should roll back L1 set or just log
                    # return False # Example: return False if Redis fails

            return True

        finally:
            op_latency = time.time() - start_time
            CACHE_LATENCY.labels(operation='set').observe(op_latency)
            logger.debug(f"Set operation for key '{key}' took {op_latency:.4f}s.")


    async def invalidate(self, key: str):
        """Invalidate cache item from L1 and Redis."""
        start_time = time.time()
        logger.debug(f"Invalidating key '{key}'...")
        try:
            # Invalidate L1
            async with self.lock:
                if key in self.l1_cache:
                    item = self.l1_cache.pop(key)
                    self.current_size_bytes -= item.size_bytes
                    self.stats['size_bytes'] = self.current_size_bytes
                    CACHE_SIZE.labels(cache_level='l1').set(self.current_size_bytes)
                    logger.debug(f"Invalidated key '{key}' from L1 cache.")
                else:
                    logger.debug(f"Key '{key}' not found in L1 cache for invalidation.")


            # Invalidate Redis
            if self.redis_client:
                try:
                    deleted_count = await asyncio.to_thread(self.redis_client.delete, key)
                    if deleted_count > 0:
                        logger.debug(f"Invalidated key '{key}' from Redis.")
                    else:
                        logger.debug(f"Key '{key}' not found in Redis for invalidation.")
                except Exception as e:
                    logger.error(f"Redis DELETE error for key {key}: {e}")

        finally:
             op_latency = time.time() - start_time
             CACHE_LATENCY.labels(operation='invalidate').observe(op_latency)
             logger.debug(f"Invalidate operation for key '{key}' took {op_latency:.4f}s.")


    async def _evict_items_internal(self, keys_to_evict: Optional[List[str]] = None, level: str = 'l1', reason: str = 'unknown', required_space: int = 0):
        """Internal helper to evict items. Assumes lock is held if modifying shared state like l1_cache."""
        if level != 'l1':
             logger.warning(f"Eviction requested for unsupported level: {level}")
             return # Currently only supports L1 eviction

        cache = self.l1_cache # Only L1 for now
        freed_space = 0

        if keys_to_evict is None:
             # If specific keys aren't provided, use eviction policy
             if not cache:
                 logger.debug("L1 cache is empty, nothing to evict.")
                 return # Nothing to evict
             if required_space <= 0:
                 logger.warning("Eviction called with policy but no required space.")
                 required_space = 1 # Evict at least one item maybe? Or return.

             logger.debug(f"Running eviction policy '{type(self.eviction_policy).__name__}' for L1 to free {required_space} bytes.")
             # Pass a copy to avoid modification issues if policy iterates differently
             keys_to_evict = self.eviction_policy.select_items_to_evict(dict(cache), required_space)
             logger.debug(f"Policy selected keys for eviction: {keys_to_evict}")


        if not keys_to_evict:
             logger.debug("No keys selected for eviction.")
             return

        # Evict selected items from L1
        for key in keys_to_evict:
            if key in cache:
                item = cache.pop(key)
                freed_space_item = item.size_bytes
                self.current_size_bytes -= freed_space_item
                freed_space += freed_space_item
                CACHE_EVICTIONS.labels(cache_level=level, reason=reason).inc()
                self.stats['evictions'] += 1
                logger.debug(f"Evicted key '{key}' from L1 (reason: {reason}). Freed {freed_space_item} bytes.")
            else:
                logger.warning(f"Attempted to evict key '{key}' which was not found in L1 cache.")


        # Update metrics after eviction
        self.stats['size_bytes'] = self.current_size_bytes
        CACHE_SIZE.labels(cache_level=level).set(self.current_size_bytes)
        logger.info(f"Eviction process completed for L1. Freed {freed_space} bytes. New size: {self.current_size_bytes} bytes.")


    def _record_access_pattern(self, key: str):
        """Record access pattern for potential prefetching."""
        # Simplified: Just track last access to enable basic sequential prefetching logic
        now = time.time()
        last_key, last_time = self._last_access
        if last_key and (now - last_time) < 5:  # Within 5 seconds suggests sequence
             pattern = (last_key, key)
             self.access_patterns[pattern] = self.access_patterns.get(pattern, 0) + 1
             # Optional: Prune old patterns periodically in maintenance
        self._last_access = (key, now)

    async def _maintenance_loop(self):
        """Background task for cache maintenance (TTL cleanup)."""
        logger.info("Cache maintenance loop started.")
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                logger.debug("Running cache maintenance (TTL check)...")
                now = time.time()
                expired_keys = []
                # Lock needed to safely iterate and modify
                async with self.lock:
                    # Check L1 TTL
                    # Create a list of keys to avoid modifying dict during iteration
                    l1_keys = list(self.l1_cache.keys())
                    for key in l1_keys:
                         # Check if key still exists (might have been evicted/invalidated)
                         item = self.l1_cache.get(key)
                         if item and now - item.created_at > self.default_ttl:
                             expired_keys.append(key)

                    if expired_keys:
                         logger.info(f"Found {len(expired_keys)} expired keys in L1: {expired_keys}")
                         await self._evict_items_internal(expired_keys, 'l1', reason='ttl')
                    else:
                         logger.debug("No expired keys found in L1.")

                # --- Add Redis TTL cleanup if needed (less critical if SETEX is used) ---
                # Redis usually handles its own TTLs via EXPIRE/SETEX

                # --- Optional: Access Pattern Cleanup ---
                # Example: Remove infrequent patterns
                # patterns_to_prune = [p for p, count in self.access_patterns.items() if count < 3]
                # for p in patterns_to_prune:
                #    del self.access_patterns[p]
                # logger.debug(f"Pruned {len(patterns_to_prune)} infrequent access patterns.")

            except asyncio.CancelledError:
                 logger.info("Cache maintenance loop cancelled.")
                 break # Exit loop immediately on cancellation
            except Exception as e:
                 # Log error and continue loop
                 logger.exception(f"Error in cache maintenance loop: {e}")
                 # Add a small delay to prevent fast looping on persistent errors
                 await asyncio.sleep(5)
    
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
        # Lock needed if accessing shared state like cache size/items directly
        # For simplicity, returning potentially slightly stale stats from self.stats dict
        return {
            **self.stats,
            'size_l1_mb': self.current_size_bytes / (1024 * 1024),
            'items_l1': len(self.l1_cache), # Snapshot of current len
            # Add stats for other levels if implemented
            'tracked_patterns': len(self.access_patterns)
        }


class MemoryCache:
    """
    Enhanced cache with TTL and size management.
    Thread-safe implementation with statistics tracking.
    """
    # FIX: Rename init to __init__
    def __init__(self, name: str = "default", max_size: int = 100, default_ttl: int = 60):
        """
        Initialize cache.

        Args:
            name: Cache name for logging
            max_size: Maximum number of entries before eviction (LRU based on timestamp)
            default_ttl: Default time-to-live in seconds
        """
        self.name = name
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {} # Store creation/update time
        self.ttls: Dict[str, int] = {} # Store specific TTL per key
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = threading.RLock() # Use RLock for reentrancy if needed
        self.last_cleanup = time.time()
        # Track memory usage (very rough estimate)
        self.estimated_memory_usage = 0
        logger.info(f"Initialized MemoryCache '{name}' (max_size={max_size}, default_ttl={default_ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                logger.debug(f"Cache miss for key '{key}' in '{self.name}' cache.")
                return None

            # Check if expired
            if self._is_expired(key):
                logger.debug(f"Key '{key}' found but expired in '{self.name}' cache. Removing.")
                self._remove(key) # Remove expired item
                self.misses += 1 # Count as miss if expired
                return None

            # Cache Hit
            self.hits += 1
            # Update timestamp to mark as recently accessed (for potential LRU eviction)
            self.timestamps[key] = time.time()
            logger.debug(f"Cache hit for key '{key}' in '{self.name}' cache.")
            return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional custom TTL."""
        with self.lock:
            # --- Eviction Logic ---
            # Check if adding a NEW key would exceed max_size
            if key not in self.cache and len(self.cache) >= self.max_size:
                logger.warning(f"'{self.name}' cache full (size={len(self.cache)}). Evicting oldest entry.")
                self._evict_oldest()

            # --- Size Estimation (Best effort) ---
            try:
                value_size = sys.getsizeof(value)
                key_size = sys.getsizeof(key)

                # Update memory tracking
                if key in self.cache:
                    # Subtract old value size, add new value size
                    old_value_size = sys.getsizeof(self.cache[key])
                    self.estimated_memory_usage += (value_size - old_value_size)
                else:
                    # Add size of new key and value
                    self.estimated_memory_usage += (key_size + value_size)
            except Exception as e:
                # Don't fail the set operation if size estimation fails
                logger.debug(f"Could not estimate size for key '{key}' in '{self.name}': {e}")
                # Optionally reset or avoid updating estimated_memory_usage

            # --- Store Data ---
            self.cache[key] = value
            self.timestamps[key] = time.time() # Record time of set/update
            self.ttls[key] = ttl if ttl is not None else self.default_ttl # Store specific or default TTL

            logger.debug(f"Set key '{key}' in '{self.name}' cache with TTL {self.ttls[key]}s.")

            # --- Periodic Cleanup ---
            # Run cleanup occasionally to remove expired items proactively
            self._maybe_cleanup()

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        with self.lock:
            if key in self.cache:
                logger.debug(f"Deleting key '{key}' from '{self.name}' cache.")
                self._remove(key) # Use the internal remove method
            else:
                 logger.debug(f"Attempted to delete non-existent key '{key}' from '{self.name}' cache.")


    def _is_expired(self, key: str) -> bool:
        """Check if a key is expired based on its timestamp and TTL."""
        # Assumes lock is held
        timestamp = self.timestamps.get(key)
        ttl = self.ttls.get(key)

        if timestamp is None or ttl is None:
             logger.warning(f"Missing timestamp or TTL for key '{key}' during expiry check in '{self.name}'.")
             return True # Treat as expired if metadata is missing

        is_expired = time.time() > (timestamp + ttl)
        # logger.debug(f"Expiry check for '{key}': now={time.time()}, ts={timestamp}, ttl={ttl}, expired={is_expired}")
        return is_expired


    def _remove(self, key: str) -> None:
        """Internal method to remove a key and its metadata. Assumes lock is held."""
        if key in self.cache:
            # Update memory tracking before deleting
            try:
                self.estimated_memory_usage -= (sys.getsizeof(key) + sys.getsizeof(self.cache[key]))
                # Ensure memory doesn't go negative due to estimation errors
                self.estimated_memory_usage = max(0, self.estimated_memory_usage)
            except Exception as e:
                logger.debug(f"Could not estimate size reduction for key '{key}' during removal: {e}")

            del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
            if key in self.ttls:
                del self.ttls[key]
            # logger.debug(f"Successfully removed key '{key}' and metadata from '{self.name}'.")

    def _evict_oldest(self) -> None:
        """Evict the least recently set/updated entry. Assumes lock is held."""
        if not self.timestamps:
            return # Nothing to evict

        # Find the key with the oldest timestamp (least recently set/updated)
        try:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            logger.info(f"'{self.name}' cache evicting oldest key: '{oldest_key}' (timestamp: {self.timestamps[oldest_key]})")
            self._remove(oldest_key)
            self.evictions += 1
            if self.evictions % 100 == 0: # Log periodically
                logger.info(f"'{self.name}' cache: {self.evictions} total evictions.")
        except ValueError:
            # Should not happen if self.timestamps is not empty, but handle defensively
            logger.error(f"'{self.name}' cache: Error finding oldest key for eviction despite non-empty timestamps.")


    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries. Assumes lock is held."""
        # Run cleanup approx every 60 seconds
        now = time.time()
        if now - self.last_cleanup < 60:
            return

        logger.debug(f"Running periodic cleanup for '{self.name}' cache...")
        self.last_cleanup = now
        # Iterate over a copy of keys to allow removal during iteration
        keys_to_check = list(self.cache.keys())
        expired_keys_found = []

        for key in keys_to_check:
            # Need to double-check existence as eviction might have removed it
            if key in self.cache and self._is_expired(key):
                expired_keys_found.append(key)
                self._remove(key) # Remove the expired key

        if expired_keys_found:
            logger.info(f"'{self.name}' cache: Cleaned up {len(expired_keys_found)} expired entries.")
        else:
            logger.debug(f"'{self.name}' cache: No expired entries found during cleanup.")


    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.estimated_memory_usage = 0
            self.hits = 0 # Reset stats as well
            self.misses = 0
            self.evictions = 0
            logger.info(f"'{self.name}' cache cleared.")

    def remove_pattern(self, pattern: str) -> int:
        """Remove all keys matching a pattern (substring check)."""
        with self.lock:
            # Find keys matching the pattern (case-sensitive substring check)
            keys_to_remove = [k for k in self.cache if pattern in k]
            count = len(keys_to_remove)
            if count > 0:
                logger.info(f"'{self.name}' cache: Removing {count} keys matching pattern '{pattern}': {keys_to_remove}")
                for key in keys_to_remove:
                    self._remove(key) # Use internal remove
            else:
                 logger.debug(f"'{self.name}' cache: No keys found matching pattern '{pattern}'.")
            return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            # Perform a quick cleanup of expired items before reporting stats for accuracy
            self._maybe_cleanup()
            
            total_accesses = self.hits + self.misses
            hit_ratio = self.hits / total_accesses if total_accesses > 0 else 0
            current_size = len(self.cache)

            return {
                "name": self.name,
                "size": current_size,
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": f"{hit_ratio:.2%}",
                "evictions": self.evictions,
                "memory_usage_bytes": self.estimated_memory_usage,
                "memory_usage_mb": f"{self.estimated_memory_usage / (1024 * 1024):.3f}" if self.estimated_memory_usage > 0 else "0.000",
                # "keys": list(self.cache.keys()) # Avoid returning all keys in production stats unless needed
            }


# --- Cache Instance Creation ---
from os import environ

# Config-driven cache settings from environment variables
NPC_CACHE_SIZE = int(environ.get("NPC_CACHE_SIZE", "500")) # Increased default
NPC_CACHE_TTL = int(environ.get("NPC_CACHE_TTL", "300")) # Increased default TTL (5 min)
LOCATION_CACHE_SIZE = int(environ.get("LOCATION_CACHE_SIZE", "100")) # Increased default
LOCATION_CACHE_TTL = int(environ.get("LOCATION_CACHE_TTL", "600")) # Increased default TTL (10 min)
AGGREGATOR_CACHE_SIZE = int(environ.get("AGGREGATOR_CACHE_SIZE", "50"))
AGGREGATOR_CACHE_TTL = int(environ.get("AGGREGATOR_CACHE_TTL", "60")) # Increased TTL (1 min)
TIME_CACHE_SIZE = int(environ.get("TIME_CACHE_SIZE", "5"))
TIME_CACHE_TTL = int(environ.get("TIME_CACHE_TTL", "10"))
COMPUTATION_CACHE_SIZE = int(environ.get("COMPUTATION_CACHE_SIZE", "100")) # Increased default
COMPUTATION_CACHE_TTL = int(environ.get("COMPUTATION_CACHE_TTL", "600")) # Increased default TTL (10 min)

# --- Initialize EnhancedCache Instance (if used) ---
# Make sure REDIS_URL env var is set if you want Redis support
redis_url_env = environ.get("REDIS_URL")
# Note: EnhancedCache is async, ensure it's started/stopped within an async context
# logger.info("Attempting to initialize EnhancedCache 'enhanced_main_cache'...")
# try:
#    # Example initialization, adjust params as needed
#    enhanced_main_cache = EnhancedCache(max_size_mb=100, redis_url=redis_url_env)
#    logger.info("Successfully initialized EnhancedCache 'enhanced_main_cache'.")
# except Exception as e:
#    logger.exception(f"CRITICAL ERROR initializing EnhancedCache 'enhanced_main_cache': {e}")
    # Decide if this should be fatal (raise) or just logged
    # raise # Re-raise to make it fatal

# --- Initialize MemoryCache Instances ---
logger.info("Initializing MemoryCache instances...")
try:
    # FIX: Ensure __init__ is called correctly now
    NPC_CACHE = MemoryCache(name="npc", max_size=NPC_CACHE_SIZE, default_ttl=NPC_CACHE_TTL)
    LOCATION_CACHE = MemoryCache(name="location", max_size=LOCATION_CACHE_SIZE, default_ttl=LOCATION_CACHE_TTL)
    AGGREGATOR_CACHE = MemoryCache(name="aggregator", max_size=AGGREGATOR_CACHE_SIZE, default_ttl=AGGREGATOR_CACHE_TTL)
    TIME_CACHE = MemoryCache(name="time", max_size=TIME_CACHE_SIZE, default_ttl=TIME_CACHE_TTL)
    # Note: ComputationCache uses MemoryCache internally
except Exception as e:
     logger.exception(f"CRITICAL ERROR initializing one or more MemoryCache instances: {e}")
     raise # Make initialization errors fatal


class ComputationCache:
    """Cache for expensive computations with function-based keys."""

    def __init__(self, name: str = "computation", max_size: int = 50, default_ttl: int = 300):
        """Initialize computation cache using MemoryCache."""
        logger.info(f"Initializing ComputationCache '{name}'...")
        # Use the MemoryCache class for the underlying storage
        self.memory_cache = MemoryCache(name=name, max_size=max_size, default_ttl=default_ttl)
        logger.info(f"ComputationCache '{name}' initialized.")


    async def cached_call(self, func: Callable, *args, ttl: Optional[int] = None, **kwargs) -> Any:
        """
        Call a function with caching based on function name and arguments.
        Handles both sync and async functions.
        """
        # Create a robust cache key
        try:
            key_parts = [func.__name__]
            # Use repr for args/kwargs to handle different types more reliably
            key_parts.extend([repr(arg) for arg in args])
            key_parts.extend([f"{k}={repr(v)}" for k, v in sorted(kwargs.items())])
            # Use hashlib for potentially long keys
            full_key_str = ":".join(key_parts)
            cache_key = f"comp:{func.__name__}:{hashlib.sha256(full_key_str.encode()).hexdigest()[:16]}"
        except Exception as e:
            logger.error(f"Error generating cache key for {func.__name__}: {e}. Caching disabled for this call.")
            # Fallback: execute without caching
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run synchronous function in a thread pool if called from async context
                # to avoid blocking the event loop. If called from sync context, just call directly.
                # Simple direct call here, needs adjustment if used heavily in async code.
                 return func(*args, **kwargs)


        # Check cache
        # Use the underlying memory_cache instance's get method
        cached_result = self.memory_cache.get(cache_key)
        if cached_result is not None:
             logger.debug(f"Computation cache hit for {func.__name__} (key: {cache_key})")
             return cached_result


        logger.debug(f"Computation cache miss for {func.__name__} (key: {cache_key}). Executing function.")
        # Call function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Consider running sync functions in thread pool if called from async context
                result = await asyncio.to_thread(func, *args, **kwargs)
                # If this cache is only used in sync code, the direct call is fine:
                # result = func(*args, **kwargs)

            elapsed = time.time() - start_time

            # Cache result using the underlying memory_cache instance's set method
            self.memory_cache.set(cache_key, result, ttl=ttl) # Pass specific ttl if provided

            # Log if execution was slow
            if elapsed > 0.1:
                logger.info(f"Cached computation for {func.__name__} took {elapsed:.3f}s (key: {cache_key})")
            else:
                logger.debug(f"Executed and cached {func.__name__} in {elapsed:.3f}s (key: {cache_key})")

            return result
        except Exception as e:
             logger.exception(f"Error executing function {func.__name__} during cached call: {e}")
             raise # Re-raise the exception after logging


# --- Initialize Global Computation Cache ---
logger.info("Initializing global COMPUTATION_CACHE...")
try:
    COMPUTATION_CACHE = ComputationCache(
        name="global_computation", # Give it a specific name
        max_size=COMPUTATION_CACHE_SIZE,
        default_ttl=COMPUTATION_CACHE_TTL
    )
    logger.info("Global COMPUTATION_CACHE initialized.")
except Exception as e:
     logger.exception(f"CRITICAL ERROR initializing COMPUTATION_CACHE: {e}")
     raise

# --- Remove redundant/example code ---
# Remove the second EnhancedCache initialization and usage examples (lines ~633-672 in original)
# Remove the simple Cache class and its instance/wrapper functions (lines ~676-725 in original)

logger.info("Finished loading utils.caching.")

# --- Ensure async functions for EnhancedCache are handled correctly ---
# Example of starting/stopping EnhancedCache if used at application level
# async def startup_event():
#     if 'enhanced_main_cache' in globals():
#         await enhanced_main_cache.start()
#
# async def shutdown_event():
#     if 'enhanced_main_cache' in globals():
#         await enhanced_main_cache.stop()
