# utils/caching.py

import time
import logging
import threading
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
                "keys": list(self.cache.keys())
            }

# Create frequently used cache instances
NPC_CACHE = MemoryCache(name="npc", max_size=50, default_ttl=30)  # 30 seconds TTL
LOCATION_CACHE = MemoryCache(name="location", max_size=20, default_ttl=120)  # 2 minutes TTL
AGGREGATOR_CACHE = MemoryCache(name="aggregator", max_size=10, default_ttl=15)  # 15 seconds TTL
TIME_CACHE = MemoryCache(name="time", max_size=5, default_ttl=10)  # 10 seconds TTL


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
COMPUTATION_CACHE = ComputationCache()
