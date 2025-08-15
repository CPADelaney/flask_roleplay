# lore/core/cache.py

import asyncio
import re
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass
import json
import logging

from agents import Agent, function_tool, Runner, trace, ModelSettings
from agents.run import RunConfig

@dataclass
class CacheAnalytics:
    """Analytics data for cache performance."""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    size_bytes: int = 0
    keys_count: int = 0
    avg_access_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

class CacheItem:
    """Enhanced cache item with metadata for optimization."""

    def __init__(self, value: Any, expiry: float, size_bytes: int = 0, priority: int = 0, 
                 user_id: int = 0, conversation_id: int = 0):
        self.value = value
        self.expiry = expiry
        self.size_bytes = size_bytes
        self.access_count = 0
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.last_access_time = 0.0
        self.creation_time = 0.0
        self.creation_datetime = None
        self.priority = priority  # 0-10, higher means more important

    async def initialize_timestamps(self):
        """Initialize timestamps using game time."""
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        dt = await get_game_datetime(self.user_id, self.conversation_id)
        ts = dt.timestamp()
        self.last_access_time = ts
        self.creation_time = ts
        self.creation_datetime = dt

    async def access(self):
        """Record an access to this item."""
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        self.access_count += 1
        self.last_access_time = await get_game_timestamp(self.user_id, self.conversation_id)
    
    def get_access_frequency(self, current_time: float) -> float:
        """Calculate access frequency (accesses per hour)."""
        hours_since_creation = (current_time - self.creation_time) / 3600
        if hours_since_creation < 0.01:  # Avoid division by zero
            hours_since_creation = 0.01
        return self.access_count / hours_since_creation

class LoreCache:
    """Unified cache system with agent-driven optimization and analytics."""
    
    def __init__(self, max_size=1000, ttl=7200):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self._lock = asyncio.Lock()  # Thread-safety for async operations
        self.analytics = CacheAnalytics()
        self._init_optimization_agent()
        self._predictor_initialized = False
        self._prediction_model = None
        self._warm_up_candidates = set()
    
    def _init_optimization_agent(self):
        """Initialize the agent for cache optimization."""
        self.optimization_agent = Agent(
            name="CacheOptimizationAgent",
            instructions=(
                "You analyze cache performance metrics and provide recommendations "
                "for optimizing cache usage. This includes identifying patterns in "
                "cache misses, suggesting priority adjustments, and predicting "
                "which items should be pre-warmed."
            ),
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.0)
        )
    
    async def get(self, namespace, key, user_id=None, conversation_id=None):
        """Get an item from the cache with async support and analytics."""
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        uid = user_id or 0
        cid = conversation_id or 0
        start_time = await get_game_timestamp(uid, cid)
        full_key = self._create_key(namespace, key, user_id, conversation_id)

        async with self._lock:
            if full_key in self.cache:
                cache_item = self.cache[full_key]
                current_time = await get_game_timestamp(uid, cid)
                if cache_item.expiry > current_time:
                    # Update access statistics
                    await cache_item.access()

                    # Record hit in analytics
                    self.analytics.hit_count += 1

                    # Calculate access time for analytics (game time)
                    end_time = await get_game_timestamp(uid, cid)
                    access_time = end_time - start_time
                    self._update_avg_access_time(access_time)

                    return cache_item.value

                # Remove expired item
                self._remove_key(full_key)
                self.analytics.eviction_count += 1
        
        # Record miss in analytics
        self.analytics.miss_count += 1
        
        # Add to warm-up candidates
        self._add_warm_up_candidate(namespace, key, user_id, conversation_id)
        
        return None
    
    async def set(self, namespace, key, value, ttl=None, user_id=None, conversation_id=None, priority=0):
        """Set an item in the cache with async support, priority, and size tracking."""
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        uid = user_id or 0
        cid = conversation_id or 0
        current_time = await get_game_timestamp(uid, cid)
        expiry = current_time + (ttl or self.default_ttl)

        # Estimate size in bytes (rough approximation)
        size_bytes = self._estimate_size(value)

        async with self._lock:
            # Manage cache size - use priority-aware eviction strategy
            if len(self.cache) >= self.max_size:
                await self._evict_items(size_bytes)

            # Create cache item with metadata
            cache_item = CacheItem(
                value=value,
                expiry=expiry,
                size_bytes=size_bytes,
                priority=priority,
                user_id=uid,
                conversation_id=cid,
            )
            await cache_item.initialize_timestamps()

            self.cache[full_key] = cache_item

            # Update analytics
            self.analytics.size_bytes += size_bytes
            self.analytics.keys_count = len(self.cache)
    
    async def invalidate(self, namespace, key, user_id=None, conversation_id=None):
        """Invalidate a specific key with async support."""
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        async with self._lock:
            if full_key in self.cache:
                self.analytics.size_bytes -= self.cache[full_key].size_bytes
                self._remove_key(full_key)
    
    async def invalidate_pattern(self, namespace, pattern, user_id=None, conversation_id=None):
        """Invalidate keys matching a pattern with async support."""
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
                self.analytics.size_bytes -= self.cache[key].size_bytes
                self._remove_key(key)
    
    async def clear_namespace(self, namespace):
        """Clear all keys in a namespace with async support."""
        namespace_prefix = f"{namespace}:"
        async with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(namespace_prefix)]
            for key in keys_to_remove:
                self.analytics.size_bytes -= self.cache[key].size_bytes
                self._remove_key(key)
    
    def _create_key(self, namespace, key, user_id=None, conversation_id=None):
        """Create a full cache key with optional scoping."""
        scoped_key = key
        if user_id:
            scoped_key = f"{scoped_key}_{user_id}"
        if conversation_id:
            scoped_key = f"{scoped_key}_{conversation_id}"
        return f"{namespace}:{scoped_key}"
    
    def _remove_key(self, full_key):
        """Remove a key from the cache."""
        if full_key in self.cache:
            del self.cache[full_key]
            self.analytics.keys_count = len(self.cache)
    
    async def _evict_items(self, required_space=0):
        """
        Evict items based on priority, expiry, and access patterns.
        This is a more sophisticated eviction strategy than basic LRU.
        """
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        # First, remove any expired items
        current_times: Dict[str, float] = {}
        expired_keys = []
        
        # Gather all timestamp fetches in parallel for efficiency
        items = list(self.cache.items())
        if items:
            timestamps = await asyncio.gather(*[
                get_game_timestamp(item.user_id, item.conversation_id)
                for _, item in items
            ])
            
            for (key, item), current in zip(items, timestamps):
                current_times[key] = current
                if item.expiry <= current:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.analytics.size_bytes -= self.cache[key].size_bytes
            self._remove_key(key)
            self.analytics.eviction_count += 1

        # If we still need to evict, use a smart strategy
        if len(self.cache) >= self.max_size or required_space > 0:
            # Calculate scores for each item (lower is more evictable)
            scores = []
            for key, item in self.cache.items():
                current_time = current_times.get(key)
                if current_time is None:
                    current_time = await get_game_timestamp(item.user_id, item.conversation_id)
                    current_times[key] = current_time
                
                # Score formula: priority * frequency * recency
                frequency = item.get_access_frequency(current_time)
                recency = (current_time - item.last_access_time) / 3600  # Hours since last access
                recency_factor = 1 / (1 + recency)  # Higher for more recent access

                score = item.priority * (1 + frequency) * recency_factor
                scores.append((key, score, item.size_bytes))

            # Sort by score (ascending, so we evict lowest scores first)
            scores.sort(key=lambda x: x[1])

            # Evict until we have enough space
            space_cleared = 0
            for key, score, size in scores:
                if len(self.cache) < self.max_size and space_cleared >= required_space:
                    break

                self.analytics.size_bytes -= size
                self._remove_key(key)
                self.analytics.eviction_count += 1
                space_cleared += size
    
    def _estimate_size(self, value):
        """Estimate the size of a value in bytes."""
        try:
            # For simple types
            if isinstance(value, (int, float, bool, type(None))):
                return 8
            # For strings
            elif isinstance(value, str):
                return len(value.encode('utf-8'))
            # For lists, dicts, etc.
            else:
                # Serialize to JSON and count bytes
                serialized = json.dumps(value)
                return len(serialized.encode('utf-8'))
        except:
            # Fallback for non-serializable objects
            return 100  # Arbitrary small size
    
    def _update_avg_access_time(self, access_time):
        """Update the average access time statistic."""
        total_accesses = self.analytics.hit_count
        if total_accesses == 1:
            self.analytics.avg_access_time_ms = access_time * 1000
        else:
            # Weighted update
            self.analytics.avg_access_time_ms = (
                (self.analytics.avg_access_time_ms * (total_accesses - 1) + access_time * 1000) / total_accesses
            )
    
    def _add_warm_up_candidate(self, namespace, key, user_id, conversation_id):
        """Add a key to warm-up candidates for predictive caching."""
        candidate = (namespace, key, user_id, conversation_id)
        self._warm_up_candidates.add(candidate)
    
    @function_tool
    async def get_cache_analytics(self):
        """Get analytics data for cache performance."""
        return self.analytics.__dict__
    
    @function_tool
    async def optimize_cache(self):
        """
        Use the optimization agent to analyze and optimize cache performance.
        """
        with trace("CacheOptimization", metadata={"component": "LoreCache"}):
            # Get current analytics
            analytics = await self.get_cache_analytics()
            
            # Prepare prompt for optimization agent
            prompt = (
                f"Analyze these cache performance metrics and recommend optimizations:\n"
                f"{json.dumps(analytics, indent=2)}\n\n"
                "Return a JSON object with:\n"
                "- priority_adjustments: list of cache item patterns and their new priority levels\n"
                "- ttl_adjustments: list of cache item patterns and their new TTL values\n"
                "- pre_warm_recommendations: list of cache item patterns to pre-warm\n"
                "- general_recommendations: text with general cache optimization advice"
            )
            
            run_ctx = RunContextWrapper(context={"component": "LoreCache"})
            result = await Runner.run(self.optimization_agent, prompt, context=run_ctx.context)
            
            try:
                recommendations = json.loads(result.final_output)
                
                # Apply priority adjustments
                if "priority_adjustments" in recommendations:
                    for adj in recommendations["priority_adjustments"]:
                        await self._adjust_item_priorities(
                            adj["pattern"], 
                            adj["namespace"], 
                            adj["new_priority"]
                        )
                
                # Apply TTL adjustments
                if "ttl_adjustments" in recommendations:
                    for adj in recommendations["ttl_adjustments"]:
                        await self._adjust_item_ttls(
                            adj["pattern"], 
                            adj["namespace"], 
                            adj["new_ttl"]
                        )
                
                # Schedule pre-warming
                if "pre_warm_recommendations" in recommendations:
                    for rec in recommendations["pre_warm_recommendations"]:
                        if "factory_func" in rec:
                            await self._schedule_pre_warm(
                                rec["namespace"],
                                rec["pattern"],
                                rec["factory_func"]
                            )
                
                return recommendations
                
            except Exception as e:
                logging.error(f"Error applying cache optimizations: {e}")
                return {"error": str(e), "raw_output": result.final_output}
    
    async def _adjust_item_priorities(self, pattern, namespace, new_priority):
        """Adjust priorities for cache items matching a pattern."""
        async with self._lock:
            namespace_prefix = f"{namespace}:"
            for key, item in self.cache.items():
                if key.startswith(namespace_prefix):
                    key_part = key[len(namespace_prefix):]
                    if re.search(pattern, key_part):
                        item.priority = new_priority
    
    async def _adjust_item_ttls(self, pattern, namespace, new_ttl):
        """Adjust TTLs for cache items matching a pattern."""
        from logic.game_time_helper import get_game_timestamp, get_game_datetime
        async with self._lock:
            namespace_prefix = f"{namespace}:"
            for key, item in self.cache.items():
                if key.startswith(namespace_prefix):
                    key_part = key[len(namespace_prefix):]
                    if re.search(pattern, key_part):
                        current_time = await get_game_timestamp(item.user_id, item.conversation_id)
                        item.expiry = current_time + new_ttl
    
    async def _schedule_pre_warm(self, namespace, pattern, factory_func_name):
        """Schedule pre-warming for keys matching a pattern."""
        # In a real implementation, you would have a registry of factory functions
        # and call the appropriate one to pre-warm the cache
        logging.info(f"Scheduled pre-warming for {namespace}:{pattern} using {factory_func_name}")
    
    @function_tool
    async def warm_predictive_cache(self, warm_strategy="high_miss"):
        """
        Pre-warm cache based on predictive model and recent miss patterns.
        
        Args:
            warm_strategy: Strategy for warming ('high_miss', 'frequency', 'custom')
            
        Returns:
            Dictionary with warming results
        """
        if len(self._warm_up_candidates) == 0:
            return {"status": "no_candidates", "warmed_keys": 0}
        
        # In a real implementation, you would:
        # 1. Train a predictive model based on access patterns
        # 2. Use it to predict which keys are likely to be accessed soon
        # 3. Pre-load those keys into the cache
        
        # For this simplified example:
        warmed = 0
        for namespace, key, user_id, conversation_id in self._warm_up_candidates:
            # This would call a factory function to generate the value
            # Instead, we'll just log it
            logging.info(f"Would warm cache for {namespace}:{key}")
            warmed += 1
        
        self._warm_up_candidates.clear()
        return {"status": "success", "warmed_keys": warmed}

# Global cache instance
GLOBAL_LORE_CACHE = LoreCache()
