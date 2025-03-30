# nyx/core/tool_cache.py

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class ToolResponseCache:
    """Cache for tool responses."""
    
    def __init__(self, max_cache_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cached response if available and not expired."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            # Entry expired
            del self.cache[key]
            return None
        
        logger.debug(f"Cache hit for key: {key}")
        return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """Cache a response."""
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        logger.debug(f"Cached response for key: {key}")

def cached_tool(cache_instance, key_func: Callable = None, ttl_seconds: Optional[int] = None):
    """Decorator for caching tool responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation based on function name and arguments
                args_str = ",".join(str(arg) for arg in args[1:])  # Skip ctx
                kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
                cache_key = f"{func.__name__}:{args_str}:{kwargs_str}"
            
            # Check cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
