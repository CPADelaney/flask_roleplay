# context/unified_cache.py

"""
Unified caching system for RPG context management.

This module provides a flexible, multi-level caching system that can be used
across all context-related components, eliminating redundancy while providing
specialized functionality for different use cases.
"""

import time
import math
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Callable, Tuple, List, Set
import asyncio

logger = logging.getLogger(__name__)

class CacheItem:
    """Represents a single cached item with metadata"""
    
    def __init__(
        self, 
        key: str, 
        value: Any, 
        timestamp: float = None,
        importance: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        self.key = key
        self.value = value
        self.timestamp = timestamp or time.time()
        self.last_access = self.timestamp
        self.access_count = 0
        self.importance = importance  # 0.0 to 1.0
        self.metadata = metadata or {}
    
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
        age_factor = self.age() / 3600  # Normalize to hours
        access_factor = 1.0 / (1.0 + self.access_count * 0.1)  # Less accesses = higher score
        recency_factor = self.time_since_access() / 3600  # Normalize to hours
        
        # Combine factors (importance reduces score)
        return (0.4 * age_factor + 0.3 * access_factor + 0.3 * recency_factor) * (2.0 - self.importance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "importance": self.importance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """Create from dictionary"""
        item = cls(
            key=data["key"],
            value=data["value"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 1.0),
            metadata=data.get("metadata", {})
        )
        item.last_access = data["last_access"]
        item.access_count = data["access_count"]
        return item


class UnifiedCache:
    """
    Unified multi-level caching system with intelligent eviction.
    
    This cache supports multiple levels with different TTLs and size limits.
    It features importance-based prioritization, intelligent eviction,
    and optional compression for large values.
    """
    
    def __init__(self, levels: int = 3, config: Dict[str, Any] = None):
        """
        Initialize the cache.
        
        Args:
            levels: Number of cache levels (default: 3)
            config: Configuration dictionary with TTLs and size limits
        """
        self.levels = levels
        
        # Default configuration
        default_config = {
            "compression_threshold": 10000,  # Size in bytes to trigger compression
            "cache_metrics": True           # Whether to track metrics
        }
        
        # Default level configurations
        for i in range(1, levels + 1):
            # Level 1: 1 minute, 100 items
            # Level 2: 5 minutes, 500 items
            # Level 3: 1 hour, 2000 items
            default_config[f"l{i}_ttl"] = 60 * (5 ** (i - 1))
            default_config[f"l{i}_max_size"] = 100 * (5 ** (i - 1))
        
        # Apply custom config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize cache levels (dictionaries of CacheItem objects)
        self.cache_levels = [{} for _ in range(levels)]
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.promotions = 0
        self.evictions = 0
        self.inserts = 0
        
        # Compression stat tracking
        self.compressed_items = 0
        self.compression_savings = 0
        
        # Performance tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def get(
        self, 
        key: str, 
        fetch_func: Callable, 
        cache_level: int = 1,
        importance: float = 0.5,
        ttl_override: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = False
    ) -> Any:
        """
        Get an item from cache, or fetch and cache it if not found.
        
        Args:
            key: Cache key
            fetch_func: Async function to call if cache miss
            cache_level: Desired cache level (1 to levels)
            importance: Importance score (0.0 to 1.0)
            ttl_override: Optional TTL override in seconds
            metadata: Optional metadata to store with the item
            compress: Whether to compress large values
            
        Returns:
            The cached or fetched value
        """
        cache_level = min(max(1, cache_level), self.levels)
        
        # Try to get from cache, starting from highest level requested
        for level in range(cache_level, 0, -1):
            level_idx = level - 1
            cache = self.cache_levels[level_idx]
            
            if key in cache:
                item = cache[key]
                ttl = ttl_override or self.config[f"l{level}_ttl"]
                
                if not item.is_stale(ttl):
                    # Cache hit
                    item.access()
                    self.hits += 1
                    
                    # Promote to higher levels if needed
                    await self._promote_item(item, level_idx)
                    
                    # Return decompressed value if needed
                    if isinstance(item.value, dict) and item.value.get('_compressed'):
                        return self._decompress_value(item.value)
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
        
        # Prepare metadata
        meta = metadata or {}
        if importance is not None:
            meta['importance'] = importance
        
        # Compress if needed
        original_value = value
        compressed = False
        if compress and self._should_compress(value):
            value = self._compress_value(value)
            compressed = True
        
        # Store in cache
        for level in range(1, cache_level + 1):
            level_idx = level - 1
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                importance=importance,
                metadata=meta
            )
            
            # Add to cache
            self.cache_levels[level_idx][key] = item
            
            # Check size limit
            self._check_size_limit(level)
        
        self.inserts += 1
        
        # Maybe run cleanup
        await self._maybe_cleanup()
        
        # Return the original value (not compressed)
        return original_value
    
    async def _promote_item(self, item: CacheItem, current_level: int) -> None:
        """Promote an item to higher cache levels based on access patterns"""
        # Only promote items that have been accessed multiple times
        if item.access_count < 2:
            return
            
        # Check if promotion is warranted
        should_promote = (
            item.access_count > 5 or  # Frequently accessed
            item.importance > 0.7 or  # Important item
            time.time() - item.last_access < 60  # Recently accessed
        )
        
        if not should_promote:
            return
            
        # Promote to levels above current level
        for level in range(current_level + 1, self.levels):
            # Skip if already at highest requested level
            if level >= self.levels:
                break
                
            # Add to higher level
            self.cache_levels[level][item.key] = item
            
            # Check size limit
            self._check_size_limit(level + 1)
            
        self.promotions += 1
    
    def _check_size_limit(self, level: int) -> None:
        """Check if a cache level exceeds its size limit and evict if necessary"""
        level_idx = level - 1
        cache = self.cache_levels[level_idx]
        max_size = self.config[f"l{level}_max_size"]
        
        if len(cache) > max_size:
            # Calculate how many items to evict (20% of excess)
            evict_count = max(1, int((len(cache) - max_size) * 0.2))
            self._evict_items(level_idx, evict_count)
    
    def _evict_items(self, level_idx: int, count: int) -> None:
        """Evict items from a cache level based on eviction scores"""
        cache = self.cache_levels[level_idx]
        
        # Calculate eviction scores
        scores = [(item.get_eviction_score(), key) for key, item in cache.items()]
        
        # Sort by score (higher scores are evicted first)
        scores.sort(reverse=True)
        
        # Evict items
        for _, key in scores[:count]:
            if key in cache:
                del cache[key]
                self.evictions += 1
    
    def invalidate(self, key_prefix: Optional[str] = None) -> int:
        """
        Invalidate cache entries by key prefix.
        
        Args:
            key_prefix: Prefix to match (None to invalidate all)
            
        Returns:
            Number of invalidated items
        """
        count = 0
        
        for level_idx in range(self.levels):
            cache = self.cache_levels[level_idx]
            
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
        """Remove stale items and compact cache"""
        start = time.time()
        removed = 0
        
        for level_idx in range(self.levels):
            level = level_idx + 1
            cache = self.cache_levels[level_idx]
            ttl = self.config[f"l{level}_ttl"]
            
            # Remove stale items
            stale_keys = [k for k, item in cache.items() if item.is_stale(ttl)]
            for key in stale_keys:
                del cache[key]
                removed += 1
        
        duration = time.time() - start
        logger.debug(f"Cache cleanup completed in {duration:.3f}s: removed {removed} items")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        metrics = {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "insertions": self.inserts,
            "evictions": self.evictions,
            "promotions": self.promotions,
            "compressed_items": self.compressed_items,
            "compression_savings_bytes": self.compression_savings,
            "levels": {}
        }
        
        # Add level-specific metrics
        for level_idx in range(self.levels):
            level = level_idx + 1
            cache = self.cache_levels[level_idx]
            
            metrics["levels"][f"l{level}"] = {
                "size": len(cache),
                "max_size": self.config[f"l{level}_max_size"],
                "ttl": self.config[f"l{level}_ttl"],
                "utilization": len(cache) / self.config[f"l{level}_max_size"]
            }
        
        return metrics
    
    def _should_compress(self, value: Any) -> bool:
        """Check if a value should be compressed"""
        if not isinstance(value, (dict, list, str)):
            return False
            
        # Estimate size in bytes
        try:
            value_size = len(json.dumps(value).encode('utf-8'))
            return value_size > self.config["compression_threshold"]
        except Exception:
            return False
    
    def _compress_value(self, value: Any) -> Dict[str, Any]:
        """Compress a value for storage"""
        import zlib
        import base64
        
        try:
            # Convert to JSON and compress
            json_value = json.dumps(value)
            original_size = len(json_value)
            
            # Compress with zlib
            compressed = zlib.compress(json_value.encode('utf-8'))
            compressed_b64 = base64.b64encode(compressed).decode('ascii')
            
            compressed_size = len(compressed)
            self.compressed_items += 1
            self.compression_savings += (original_size - compressed_size)
            
            return {
                "_compressed": True,
                "format": "zlib+b64",
                "data": compressed_b64,
                "original_type": type(value).__name__
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value
    
    def _decompress_value(self, compressed: Dict[str, Any]) -> Any:
        """Decompress a stored value"""
        import zlib
        import base64
        
        try:
            # Only support zlib+b64 format for now
            if compressed.get("format") != "zlib+b64":
                logger.warning(f"Unknown compression format: {compressed.get('format')}")
                return compressed
                
            # Decode and decompress
            compressed_data = base64.b64decode(compressed["data"])
            json_value = zlib.decompress(compressed_data).decode('utf-8')
            
            # Parse JSON
            return json.loads(json_value)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compressed
