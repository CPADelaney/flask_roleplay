# lore/core/cache.py

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional

class LoreCache:
    """Unified cache system for all lore types with improved organization"""
    
    def __init__(self, max_size=1000, ttl=7200):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.access_times = {}
        self._lock = asyncio.Lock()  # Thread-safety for async operations
    
    async def get(self, namespace, key, user_id=None, conversation_id=None):
        """Get an item from the cache with async support"""
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        
        async with self._lock:
            if full_key in self.cache:
                value, expiry = self.cache[full_key]
                if expiry > datetime.now().timestamp():
                    # Update access time for LRU
                    self.access_times[full_key] = datetime.now().timestamp()
                    return value
                # Remove expired item
                self._remove_key(full_key)
        return None
    
    async def set(self, namespace, key, value, ttl=None, user_id=None, conversation_id=None):
        """Set an item in the cache with async support"""
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        expiry = datetime.now().timestamp() + (ttl or self.default_ttl)
        
        async with self._lock:
            # Manage cache size - use LRU strategy
            if len(self.cache) >= self.max_size:
                # Find oldest accessed item
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                self._remove_key(oldest_key)
                
            self.cache[full_key] = (value, expiry)
            self.access_times[full_key] = datetime.now().timestamp()
    
    async def invalidate(self, namespace, key, user_id=None, conversation_id=None):
        """Invalidate a specific key with async support"""
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        async with self._lock:
            self._remove_key(full_key)
    
    async def invalidate_pattern(self, namespace, pattern, user_id=None, conversation_id=None):
        """Invalidate keys matching a pattern with async support"""
        namespace_pattern = f"{namespace}:"
        async with self._lock:
            keys_to_remove = []
            
            for key in self.cache.keys():
                if key.startswith(namespace_pattern):
                    # Extract the key part after the namespace
                    key_part = key[len(namespace_pattern):]
                    if re.search(pattern, key_part):
                        keys_to_remove.append(key)
                        
            for key in keys_to_remove:
                self._remove_key(key)
    
    async def clear_namespace(self, namespace):
        """Clear all keys in a namespace with async support"""
        namespace_prefix = f"{namespace}:"
        async with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(namespace_prefix)]
            for key in keys_to_remove:
                self._remove_key(key)
    
    def _create_key(self, namespace, key, user_id=None, conversation_id=None):
        """Create a full cache key with optional scoping"""
        scoped_key = key
        if user_id:
            scoped_key = f"{scoped_key}_{user_id}"
        if conversation_id:
            scoped_key = f"{scoped_key}_{conversation_id}"
        return f"{namespace}:{scoped_key}"
    
    def _remove_key(self, full_key):
        """Remove a key from both cache and access times"""
        if full_key in self.cache:
            del self.cache[full_key]
        if full_key in self.access_times:
            del self.access_times[full_key]

# Global cache instance
GLOBAL_LORE_CACHE = LoreCache()
