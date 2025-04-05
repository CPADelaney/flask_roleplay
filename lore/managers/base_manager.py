# lore/managers/base_manager.py

from typing import Dict, List, Any, Optional, Type, Set
import logging
from datetime import datetime
from lore.database_access import DatabaseAccess
from lore.utils.caching import LoreCache
from lore.error_handler import ErrorHandler
from lore.monitoring import metrics
from lore.lore_cache_manager import LoreCacheManager
import asyncio

logger = logging.getLogger(__name__)

class BaseLoreManager:
    """Base class for all lore managers providing common functionality."""
    
    def __init__(self, user_id: int, conversation_id: int, cache_size: int = 100, ttl: int = 3600):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db = DatabaseAccess(user_id, conversation_id)
        self.cache = LoreCache(max_size=cache_size, ttl=ttl)
        self.error_handler = ErrorHandler()
        
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache with metrics tracking."""
        try:
            start_time = datetime.utcnow()
            data = self.cache.get(cache_key)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            if data:
                metrics.record_cache_operation(self.__class__.__name__, True)
            else:
                metrics.record_cache_operation(self.__class__.__name__, False)
                
            return data
        except Exception as e:
            self.error_handler.handle_error(e)
            return None
            
    async def _set_cached_data(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """Set data in cache with metrics tracking."""
        try:
            start_time = datetime.utcnow()
            self.cache.set(cache_key, data)
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.record_cache_operation(self.__class__.__name__, True)
            return True
        except Exception as e:
            self.error_handler.handle_error(e)
            return False
            
    async def _delete_cached_data(self, cache_key: str) -> bool:
        """Delete data from cache."""
        try:
            self.cache.delete(cache_key)
            return True
        except Exception as e:
            self.error_handler.handle_error(e)
            return False
            
    async def _execute_db_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute database query with metrics tracking."""
        try:
            start_time = datetime.utcnow()
            result = await self.db._execute_query(query, *args)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            metrics.record_db_query(
                operation=query.split()[0].lower(),
                table=self._get_table_name(query),
                duration=duration
            )
            
            return result
        except Exception as e:
            self.error_handler.handle_error(e)
            return []
            
    def _get_table_name(self, query: str) -> str:
        """Extract table name from SQL query."""
        query = query.lower()
        if 'from' in query:
            return query.split('from')[1].split()[0]
        return 'unknown'
        
    async def _batch_update(self, table: str, updates: List[Dict[str, Any]]) -> bool:
        """Perform batch update operation."""
        try:
            start_time = datetime.utcnow()
            result = await self.db.execute_many(
                f"UPDATE {table} SET $1 = $2 WHERE id = $3 AND user_id = $4",
                [(update, update['id'], self.user_id) for update in updates]
            )
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            metrics.record_db_query(
                operation='batch_update',
                table=table,
                duration=duration
            )
            
            return bool(result)
        except Exception as e:
            self.error_handler.handle_error(e)
            return False
            
    async def _validate_data(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """Validate data against schema."""
        try:
            from ..schemas import schema_validator
            return schema_validator.validate(data, schema_type)
        except Exception as e:
            self.error_handler.handle_error(e)
            raise

class BaseManager:
    """Base manager class with integrated caching support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize cache manager
        self.cache_manager = LoreCacheManager(
            user_id=user_id,
            conversation_id=conversation_id,
            max_size_mb=max_size_mb,
            redis_url=redis_url
        )
        
        # Cache configuration
        self.cache_config = {
            'ttl': 3600,  # 1 hour default TTL
            'max_size': max_size_mb,
            'redis_url': redis_url
        }
        
        # Background tasks
        self.maintenance_task = None
    
    async def start(self):
        """Start the manager and its cache."""
        await self.cache_manager.start()
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def stop(self):
        """Stop the manager and its cache."""
        await self.cache_manager.stop()
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def get_cached_data(
        self,
        data_type: str,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Any]:
        """Get data from cache or fetch if not available."""
        try:
            # Try to get from cache first
            cached_value = await self.cache_manager.get_lore(data_type, data_id)
            if cached_value is not None:
                return cached_value
            
            # If not in cache and fetch function provided, fetch and cache
            if fetch_func:
                value = await fetch_func()
                if value is not None:
                    await self.cache_manager.set_lore(data_type, data_id, value)
                return value
            
            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    async def set_cached_data(
        self,
        data_type: str,
        data_id: str,
        value: Any,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set data in cache."""
        try:
            return await self.cache_manager.set_lore(
                data_type,
                data_id,
                value,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Error setting cached data: {e}")
            return False
    
    async def invalidate_cached_data(
        self,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate cached data."""
        try:
            await self.cache_manager.invalidate_lore(
                data_type,
                data_id,
                recursive=recursive
            )
        except Exception as e:
            logger.error(f"Error invalidating cached data: {e}")
    
    async def _maintenance_loop(self):
        """Background task for maintenance."""
        while True:
            try:
                # Get cache statistics
                stats = self.cache_manager.get_cache_stats()
                
                # Log cache health
                if stats['misses'] / (stats['hits'] + stats['misses']) > 0.3:
                    logger.warning("High cache miss rate detected")
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_cache_stats() 
