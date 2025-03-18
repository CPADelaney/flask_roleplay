"""
Lore Cache Management System

This module provides a specialized cache management system for lore data,
integrating with the core caching implementation from utils.caching.
"""

import logging
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
import asyncio
from utils.caching import EnhancedCache, CacheItem, EvictionPolicy, TTLEvictionPolicy

logger = logging.getLogger(__name__)

class LoreCacheManager:
    """
    Specialized cache manager for lore data with multi-level caching,
    predictive prefetching, and automatic invalidation.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 500,  # Larger default size for lore
        redis_url: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize multi-level cache with specialized TTLs for lore
        self.cache = EnhancedCache(
            max_size_mb=max_size_mb,
            redis_url=redis_url,
            eviction_policy=TTLEvictionPolicy(ttl=3600)  # 1 hour default TTL
        )
        
        # Lore-specific cache levels
        self.lore_levels = {
            'world': 'l3',  # Long-lived world lore
            'faction': 'l2',  # Medium-lived faction data
            'location': 'l2',  # Medium-lived location data
            'npc': 'l1',  # Short-lived NPC data
            'quest': 'l1',  # Short-lived quest data
            'event': 'l1',  # Short-lived event data
            'relationship': 'l2',  # Medium-lived relationship data
            'culture': 'l2',  # Medium-lived cultural data
            'history': 'l3',  # Long-lived historical data
            'magic': 'l3',  # Long-lived magic system data
        }
        
        # Cache tags for automatic invalidation
        self.tags = {
            'world': {'world_lore', 'foundation'},
            'faction': {'faction', 'organization', 'group'},
            'location': {'location', 'place', 'area'},
            'npc': {'npc', 'character', 'person'},
            'quest': {'quest', 'mission', 'task'},
            'event': {'event', 'happening', 'occurrence'},
            'relationship': {'relationship', 'connection', 'link'},
            'culture': {'culture', 'society', 'custom'},
            'history': {'history', 'past', 'timeline'},
            'magic': {'magic', 'spell', 'power'}
        }
        
        # Dependency tracking for automatic invalidation
        self.dependencies = {
            'world': {'faction', 'location', 'culture', 'history', 'magic'},
            'faction': {'npc', 'location', 'quest', 'event'},
            'location': {'npc', 'quest', 'event'},
            'npc': {'quest', 'event', 'relationship'},
            'quest': {'npc', 'location', 'event'},
            'event': {'npc', 'location', 'faction'},
            'relationship': {'npc'},
            'culture': {'faction', 'npc'},
            'history': {'event', 'faction', 'location'},
            'magic': {'npc', 'event', 'quest'}
        }
        
        # Cache warming patterns
        self.warming_patterns = {
            'world': ['faction', 'location', 'culture'],
            'faction': ['npc', 'location'],
            'location': ['npc', 'quest'],
            'npc': ['quest', 'relationship'],
            'quest': ['npc', 'location']
        }
        
        # Background tasks
        self.maintenance_task = None
        self.warming_task = None
    
    async def start(self):
        """Start the cache manager and background tasks."""
        await self.cache.start()
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.warming_task = asyncio.create_task(self._warming_loop())
    
    async def stop(self):
        """Stop the cache manager and background tasks."""
        await self.cache.stop()
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        if self.warming_task:
            self.warming_task.cancel()
            try:
                await self.warming_task
            except asyncio.CancelledError:
                pass
    
    async def get_lore(self, lore_type: str, lore_id: str) -> Optional[Any]:
        """Get lore data from cache."""
        if lore_type not in self.lore_levels:
            logger.error(f"Invalid lore type: {lore_type}")
            return None
            
        cache_level = self.lore_levels[lore_type]
        cache_key = f"{lore_type}:{lore_id}"
        
        # Try to get from cache
        value = await self.cache.get(cache_key, level=cache_level)
        if value is not None:
            return value
            
        # If not in cache, trigger cache warming for related items
        await self._warm_related_cache(lore_type, lore_id)
        
        return None
    
    async def set_lore(
        self,
        lore_type: str,
        lore_id: str,
        value: Any,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore data in cache."""
        if lore_type not in self.lore_levels:
            logger.error(f"Invalid lore type: {lore_type}")
            return False
            
        cache_level = self.lore_levels[lore_type]
        cache_key = f"{lore_type}:{lore_id}"
        
        # Combine default tags with provided tags
        default_tags = self.tags.get(lore_type, set())
        combined_tags = default_tags.union(tags or set())
        
        # Set in cache
        success = await self.cache.set(
            cache_key,
            value,
            level=cache_level,
            tags=combined_tags
        )
        
        if success:
            # Invalidate dependent items
            await self._invalidate_dependent_items(lore_type, lore_id)
            
        return success
    
    async def invalidate_lore(
        self,
        lore_type: str,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore data in cache."""
        if lore_type not in self.lore_levels:
            logger.error(f"Invalid lore type: {lore_type}")
            return
            
        cache_level = self.lore_levels[lore_type]
        
        if lore_id:
            # Invalidate specific lore item
            cache_key = f"{lore_type}:{lore_id}"
            await self.cache.invalidate(cache_key, level=cache_level)
            
            if recursive:
                # Invalidate dependent items
                await self._invalidate_dependent_items(lore_type, lore_id)
        else:
            # Invalidate all items of this type
            await self.cache.invalidate_pattern(f"{lore_type}:*", level=cache_level)
            
            if recursive:
                # Invalidate all dependent types
                for dep_type in self.dependencies.get(lore_type, set()):
                    await self.invalidate_lore(dep_type, recursive=True)
    
    async def _invalidate_dependent_items(self, lore_type: str, lore_id: str) -> None:
        """Invalidate items that depend on the given lore item."""
        for dep_type in self.dependencies.get(lore_type, set()):
            cache_level = self.lore_levels[dep_type]
            await self.cache.invalidate_pattern(f"{dep_type}:*", level=cache_level)
    
    async def _warm_related_cache(self, lore_type: str, lore_id: str) -> None:
        """Warm cache for related lore items."""
        warming_types = self.warming_patterns.get(lore_type, [])
        for warm_type in warming_types:
            cache_level = self.lore_levels[warm_type]
            # Trigger async cache warming for related items
            asyncio.create_task(self._warm_cache_items(warm_type, lore_id, cache_level))
    
    async def _warm_cache_items(
        self,
        lore_type: str,
        related_id: str,
        cache_level: str
    ) -> None:
        """Warm cache for specific items."""
        try:
            # This would be implemented to fetch related items from your data source
            # For example, if warming NPCs for a location:
            # related_items = await fetch_npcs_in_location(related_id)
            # for item in related_items:
            #     await self.set_lore(lore_type, item.id, item.data)
            pass
        except Exception as e:
            logger.error(f"Error warming cache for {lore_type}: {e}")
    
    async def _maintenance_loop(self):
        """Background task for cache maintenance."""
        while True:
            try:
                # Update cache statistics
                stats = self.cache.get_stats()
                logger.info(f"Cache stats: {stats}")
                
                # Check for cache health
                if stats['misses'] / (stats['hits'] + stats['misses']) > 0.3:
                    logger.warning("High cache miss rate detected")
                    # Implement cache optimization strategies
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)
    
    async def _warming_loop(self):
        """Background task for predictive cache warming."""
        while True:
            try:
                # Analyze access patterns and warm frequently accessed items
                access_patterns = self.cache.get_access_patterns()
                for pattern in access_patterns:
                    if pattern['frequency'] > 0.7:  # High frequency pattern
                        await self._warm_cache_items(
                            pattern['type'],
                            pattern['id'],
                            self.lore_levels[pattern['type']]
                        )
                
                await asyncio.sleep(600)  # Check every 10 minutes
            except Exception as e:
                logger.error(f"Error in warming loop: {e}")
                await asyncio.sleep(600)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        return self.cache.get_stats() 