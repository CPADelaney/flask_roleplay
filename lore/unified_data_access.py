"""
Unified Data Access with Resource Management

This module provides unified data access capabilities with integrated resource management.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from .base_manager import BaseManager
from .resource_manager import resource_manager

logger = logging.getLogger(__name__)

class UnifiedDataAccess(BaseManager):
    """Manager for unified data access with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.data_access = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the unified data access manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the unified data access manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_data(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('data', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None
    
    async def set_data(
        self,
        data_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('data', data_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting data: {e}")
            return False
    
    async def invalidate_data(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate data cache."""
        try:
            await self.invalidate_cached_data('data', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating data: {e}")
    
    async def get_data_history(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get data history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('data_history', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting data history: {e}")
            return None
    
    async def set_data_history(
        self,
        data_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set data history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('data_history', data_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting data history: {e}")
            return False
    
    async def invalidate_data_history(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate data history cache."""
        try:
            await self.invalidate_cached_data('data_history', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating data history: {e}")
    
    async def get_data_metadata(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get data metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('data_metadata', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting data metadata: {e}")
            return None
    
    async def set_data_metadata(
        self,
        data_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set data metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('data_metadata', data_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting data metadata: {e}")
            return False
    
    async def invalidate_data_metadata(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate data metadata cache."""
        try:
            await self.invalidate_cached_data('data_metadata', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating data metadata: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
unified_data_access = UnifiedDataAccess() 