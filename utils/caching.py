# utils/caching.py

import time
import logging
import threading
import asyncio  # Added missing import
from typing import Dict, Any, Optional, Callable, Tuple, Union, List

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
import time
import logging
import asyncio
import zlib
from typing import Dict, Any, Optional, Tuple, List, Callable, Set
import json
import hashlib

logger = logging.getLogger(__name__)

class CacheItem:
    """Cache item with metadata for smarter caching decisions"""
    
    def __init__(self, key: str, value: Any, level: int = 1):
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        self.level = level
        self.size = self._estimate_size(value)
        self.value_hash = self._compute_hash(value)
    
    def access(self) -> None:
        """Record an access to this cache item"""
        self.access_count += 1
        self.last_access = time.time()
    
    def update(self, value: Any) -> bool:
        """Update the value and return whether it changed"""
        new_hash = self._compute_hash(value)
        if new_hash != self.value_hash:
            self.value = value
            self.timestamp = time.time()
            self.value_hash = new_hash
            self.size = self._estimate_size(value)
            return True
        return False
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes"""
        try:
            return len(json.dumps(value).encode('utf-8'))
        except (TypeError, OverflowError):
            # Fallback for non-serializable objects
            return 1024  # Default size estimate
    
    def _compute_hash(self, value: Any) -> str:
        """Compute a hash for the value to detect changes"""
        try:
            serialized = json.dumps(value, sort_keys=True)
            return hashlib.md5(serialized.encode('utf-8')).hexdigest()
        except (TypeError, OverflowError):
            # Fallback for non-serializable objects
            return str(id(value))
    
    def compress(self) -> None:
        """Compress the value to save memory"""
        if not isinstance(self.value, bytes) and not hasattr(self.value, '_is_compressed'):
            try:
                serialized = json.dumps(self.value)
                self.value = zlib.compress(serialized.encode('utf-8'))
                self.value._is_compressed = True
            except (TypeError, AttributeError):
                # Could not compress, skip
                pass
    
    def decompress(self) -> None:
        """Decompress the value for use"""
        if isinstance(self.value, bytes) and hasattr(self.value, '_is_compressed'):
            try:
                decompressed = zlib.decompress(self.value).decode('utf-8')
                self.value = json.loads(decompressed)
                delattr(self.value, '_is_compressed')
            except (zlib.error, UnicodeDecodeError, json.JSONDecodeError):
                # Could not decompress, leave as is
                pass


class EnhancedContextCache:
    """Enhanced multi-level cache with adaptive TTLs and predictive prefetching"""
    
    def __init__(self, max_size_mb: float = 100):
        # Level 1: Very short-lived (seconds to minutes)
        self.l1_cache: Dict[str, CacheItem] = {}
        self.l1_ttl = 60  # Base TTL: 1 minute
        
        # Level 2: Medium-lived (minutes)
        self.l2_cache: Dict[str, CacheItem] = {}
        self.l2_ttl = 300  # Base TTL: 5 minutes
        
        # Level 3: Long-lived (hours)
        self.l3_cache: Dict[str, CacheItem] = {}
        self.l3_ttl = 3600  # Base TTL: 1 hour
        
        # Store related keys for partial invalidation
        self.key_relationships: Dict[str, Set[str]] = {}
        
        # Cache metrics and settings
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.last_maintenance = time.time()
        
        # Access patterns for prefetching
        self.access_patterns: Dict[str, Dict[str, int]] = {}
        
        # Background task for cache maintenance
        self.maintenance_task = None
    
    async def start_maintenance(self, interval: int = 300):
        """Start background cache maintenance"""
        if self.maintenance_task is None:
            self.maintenance_task = asyncio.create_task(self._maintenance_loop(interval))
    
    async def stop_maintenance(self):
        """Stop background maintenance task"""
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
            self.maintenance_task = None
    
    async def _maintenance_loop(self, interval: int):
        """Background loop to perform cache maintenance"""
        while True:
            try:
                await asyncio.sleep(interval)
                self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    def _perform_maintenance(self):
        """Perform cache maintenance tasks"""
        now = time.time()
        self.last_maintenance = now
        
        # Expire old entries
        self._expire_cache_entries(now)
        
        # Compress rarely accessed items in L3
        self._compress_rarely_accessed(now)
        
        # If we're over size limit, evict least valuable entries
        if self.current_size_bytes > self.max_size_bytes:
            self._evict_entries(self.current_size_bytes - self.max_size_bytes)
        
        # Update adaptive TTLs based on access patterns
        self._update_adaptive_ttls()
    
    def _expire_cache_entries(self, now: float):
        """Expire old cache entries"""
        for level, cache, ttl in [
            (1, self.l1_cache, self.l1_ttl),
            (2, self.l2_cache, self.l2_ttl),
            (3, self.l3_cache, self.l3_ttl)
        ]:
            keys_to_remove = []
            
            for key, item in cache.items():
                # Adjust TTL based on access frequency
                adjusted_ttl = ttl * (1 + 0.1 * min(item.access_count, 10))
                if now - item.timestamp > adjusted_ttl:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_item(cache, key)
    
    def _compress_rarely_accessed(self, now: float):
        """Compress items that are rarely accessed to save memory"""
        for key, item in self.l3_cache.items():
            # If item hasn't been accessed in 15 minutes, compress it
            if now - item.last_access > 900:  # 15 minutes
                item.compress()
    
    def _evict_entries(self, bytes_to_free: int):
        """Evict least valuable entries to free up space"""
        # Compute value score for each item (lower is less valuable)
        value_scores = []
        
        # Start with L1 as it's meant to be shortest-lived
        for level, cache in [(1, self.l1_cache), (2, self.l2_cache), (3, self.l3_cache)]:
            for key, item in cache.items():
                # Value formula: access_count / (age * size)
                age = time.time() - item.timestamp
                # Prevent division by zero
                if age < 0.1:
                    age = 0.1
                    
                # Normalize access count (1-11)
                normalized_access = 1 + 0.1 * min(item.access_count, 100)
                
                # Calculate value score (higher means more valuable)
                value = normalized_access / (age * item.size)
                
                # Adjust by level - L3 items get a small bonus
                if level == 3:
                    value *= 1.2
                
                value_scores.append((level, key, value, item.size))
        
        # Sort by value (ascending, so least valuable first)
        value_scores.sort(key=lambda x: x[2])
        
        # Evict items until we've freed enough space
        bytes_freed = 0
        for level, key, _, size in value_scores:
            if bytes_freed >= bytes_to_free:
                break
                
            if level == 1 and key in self.l1_cache:
                self._remove_item(self.l1_cache, key)
                bytes_freed += size
            elif level == 2 and key in self.l2_cache:
                self._remove_item(self.l2_cache, key)
                bytes_freed += size
            elif level == 3 and key in self.l3_cache:
                self._remove_item(self.l3_cache, key)
                bytes_freed += size
    
    def _update_adaptive_ttls(self):
        """Update TTLs based on cache performance and access patterns"""
        # If hit rate is high, we can extend TTLs
        total_requests = self.hit_count + self.miss_count
        if total_requests > 100:
            hit_rate = self.hit_count / total_requests
            
            # Adapt TTLs based on hit rate
            if hit_rate > 0.9:  # Excellent hit rate
                self.l1_ttl = min(self.l1_ttl * 1.1, 120)  # Max 2 minutes
                self.l2_ttl = min(self.l2_ttl * 1.1, 600)  # Max 10 minutes
                self.l3_ttl = min(self.l3_ttl * 1.1, 7200)  # Max 2 hours
            elif hit_rate < 0.7:  # Poor hit rate
                self.l1_ttl = max(self.l1_ttl * 0.9, 30)   # Min 30 seconds
                self.l2_ttl = max(self.l2_ttl * 0.9, 120)  # Min 2 minutes
                self.l3_ttl = max(self.l3_ttl * 0.9, 1800) # Min 30 minutes
            
            # Reset counters for next cycle
            self.hit_count = 0
            self.miss_count = 0
    
    def _record_access_pattern(self, key: str, next_key: str = None):
        """Record access patterns for prefetching"""
        if next_key and key != next_key:
            if key not in self.access_patterns:
                self.access_patterns[key] = {}
            
            if next_key in self.access_patterns[key]:
                self.access_patterns[key][next_key] += 1
            else:
                self.access_patterns[key][next_key] = 1
    
    def _remove_item(self, cache: Dict[str, CacheItem], key: str):
        """Remove an item from the cache and update metrics"""
        if key in cache:
            item = cache[key]
            self.current_size_bytes -= item.size
            del cache[key]
            
            # Also remove from key relationships
            if key in self.key_relationships:
                del self.key_relationships[key]
    
    def _add_item(self, cache: Dict[str, CacheItem], item: CacheItem):
        """Add an item to the cache and update metrics"""
        old_item = cache.get(item.key)
        if old_item:
            self.current_size_bytes -= old_item.size
        
        cache[item.key] = item
        self.current_size_bytes += item.size
    
    def relate_keys(self, primary_key: str, related_keys: List[str]):
        """
        Establish relationships between keys for smart invalidation
        
        Args:
            primary_key: The main key
            related_keys: List of keys related to the primary key
        """
        if primary_key not in self.key_relationships:
            self.key_relationships[primary_key] = set()
        
        for key in related_keys:
            if key != primary_key:
                self.key_relationships[primary_key].add(key)
    
    async def get(self, key: str, fetch_func: Callable, cache_level: int = 1,
                 ttl_override: Optional[int] = None, related_keys: List[str] = None) -> Any:
        """
        Get data from cache or fetch it.
        
        Args:
            key: Cache key
            fetch_func: Async function to fetch data if not in cache
            cache_level: Level to cache the result (1-3)
            ttl_override: Optional override for TTL
            related_keys: List of keys related to this data for smart invalidation
            
        Returns:
            Cached or freshly fetched data
        """
        now = time.time()
        last_key = getattr(self, '_last_access_key', None)
        self._last_access_key = key
        
        # Record access pattern for prefetching
        if last_key:
            self._record_access_pattern(last_key, key)
        
        # Setup relationships if provided
        if related_keys:
            self.relate_keys(key, related_keys)
        
        # Try L1 cache first
        if key in self.l1_cache:
            item = self.l1_cache[key]
            ttl = ttl_override or self.l1_ttl
            
            if now - item.timestamp < ttl:
                # Decompress if needed
                item.decompress()
                item.access()
                self.hit_count += 1
                return item.value
            else:
                self._remove_item(self.l1_cache, key)
        
        # Try L2 cache
        if key in self.l2_cache:
            item = self.l2_cache[key]
            ttl = ttl_override or self.l2_ttl
            
            if now - item.timestamp < ttl:
                # Decompress if needed
                item.decompress()
                item.access()
                # Promote to L1 cache
                self._add_item(self.l1_cache, CacheItem(key, item.value, level=1))
                self.hit_count += 1
                return item.value
            else:
                self._remove_item(self.l2_cache, key)
        
        # Try L3 cache
        if key in self.l3_cache:
            item = self.l3_cache[key]
            ttl = ttl_override or self.l3_ttl
            
            if now - item.timestamp < ttl:
                # Decompress if needed
                item.decompress()
                item.access()
                # Promote to L2 cache
                self._add_item(self.l2_cache, CacheItem(key, item.value, level=2))
                self.hit_count += 1
                return item.value
            else:
                self._remove_item(self.l3_cache, key)
        
        # Cache miss - fetch the data
        self.miss_count += 1
        data = await fetch_func()
        
        # Store in appropriate cache level
        item = CacheItem(key, data, level=cache_level)
        
        if cache_level >= 1:
            self._add_item(self.l1_cache, item)
        
        if cache_level >= 2:
            # Create a new item for L2 to avoid shared reference
            self._add_item(self.l2_cache, CacheItem(key, data, level=2))
        
        if cache_level >= 3:
            # Create a new item for L3 to avoid shared reference
            self._add_item(self.l3_cache, CacheItem(key, data, level=3))
        
        # Initiate prefetching if we have access patterns
        if key in self.access_patterns:
            asyncio.create_task(self._prefetch_related(key, data))
        
        return data
    
    async def _prefetch_related(self, key: str, current_data: Any):
        """
        Prefetch related items based on access patterns
        
        Args:
            key: Current key that was accessed
            current_data: Current data that was fetched
        """
        if key not in self.access_patterns:
            return
        
        # Find most likely next keys (up to 3)
        next_keys = sorted(
            self.access_patterns[key].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Prefetch each one
        for next_key, count in next_keys:
            # Only prefetch if count is significant
            if count < 3:
                continue
                
            # Check if already in cache
            if (next_key in self.l1_cache or
                next_key in self.l2_cache or
                next_key in self.l3_cache):
                continue
            
            # Schedule prefetch with lower priority
            async def prefetch_func():
                # Simulate a fetch function - this would need to be adapted for real use
                await asyncio.sleep(0.1)  # Delay to not block main operations
                return {"prefetched": True, "key": next_key}
            
            try:
                asyncio.create_task(self.get(next_key, prefetch_func, cache_level=2))
            except Exception as e:
                logger.warning(f"Error in prefetch: {e}")
    
    def invalidate(self, key_prefix: str = None, key: str = None, recursive: bool = True):
        """
        Invalidate cache entries.
        
        Args:
            key_prefix: Prefix to match keys for invalidation
            key: Exact key to invalidate
            recursive: Whether to invalidate related keys
        """
        invalidated = 0
        
        if key:
            # Invalidate exact key
            for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
                if key in cache:
                    self._remove_item(cache, key)
                    invalidated += 1
            
            # Recursively invalidate related keys
            if recursive and key in self.key_relationships:
                for related_key in self.key_relationships[key]:
                    invalidated += self.invalidate(key=related_key, recursive=False)
                
                # Clear relationships for this key
                self.key_relationships[key] = set()
        
        elif key_prefix:
            # Invalidate by prefix
            for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
                for cache_key in list(cache.keys()):
                    if cache_key.startswith(key_prefix):
                        self._remove_item(cache, cache_key)
                        invalidated += 1
                        
                        # Recursively invalidate related keys
                        if recursive and cache_key in self.key_relationships:
                            for related_key in self.key_relationships[cache_key]:
                                invalidated += self.invalidate(key=related_key, recursive=False)
                            
                            # Clear relationships for this key
                            self.key_relationships[cache_key] = set()
        
        return invalidated
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            "size_bytes": self.current_size_bytes,
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "l1_count": len(self.l1_cache),
            "l2_count": len(self.l2_cache),
            "l3_count": len(self.l3_cache),
            "l1_ttl": self.l1_ttl,
            "l2_ttl": self.l2_ttl,
            "l3_ttl": self.l3_ttl,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0,
            "last_maintenance": self.last_maintenance
        }


# Usage examples

# Create a context cache for NPCs with 50MB max size
npc_cache = EnhancedContextCache(max_size_mb=50)

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
        cache_level=3,  # Store in all cache levels (long-lived)
        related_keys=related_keys
    )

async def update_npc_location(user_id, conv_id, npc_id, new_location):
    """Example of smart cache invalidation"""
    # After updating the NPC in the database
    
    # Invalidate the specific NPC
    cache_key = f"npc:{user_id}:{conv_id}:{npc_id}"
    npc_cache.invalidate(key=cache_key)
    
    # Invalidate the location's NPC list
    location_key = f"location:{user_id}:{conv_id}:{new_location}"
    npc_cache.invalidate(key=location_key)

# Start background maintenance
async def startup():
    await npc_cache.start_maintenance(interval=300)  # Run every 5 minutes

# Stop maintenance on shutdown
async def shutdown():
    await npc_cache.stop_maintenance()
