"""
World Lore Manager with Resource Management

This module provides world lore management capabilities with integrated resource management.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from ..base_manager import BaseManager
from ..resource_manager import resource_manager

logger = logging.getLogger(__name__)

class WorldLoreManager(BaseManager):
    """Manager for world lore with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.world_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the world lore manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the world lore manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_world_data(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get world data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world data: {e}")
            return None
    
    async def set_world_data(
        self,
        world_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('world', world_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting world data: {e}")
            return False
    
    async def invalidate_world_data(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world data cache."""
        try:
            await self.invalidate_cached_data('world', world_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating world data: {e}")
    
    async def get_world_history(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get world history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_history', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world history: {e}")
            return None
    
    async def set_world_history(
        self,
        world_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('world_history', world_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting world history: {e}")
            return False
    
    async def invalidate_world_history(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world history cache."""
        try:
            await self.invalidate_cached_data('world_history', world_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating world history: {e}")
    
    async def get_world_events(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get world events from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_events', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world events: {e}")
            return None
    
    async def set_world_events(
        self,
        world_id: str,
        events: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world events in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('world_events', world_id, events, tags)
        except Exception as e:
            logger.error(f"Error setting world events: {e}")
            return False
    
    async def invalidate_world_events(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world events cache."""
        try:
            await self.invalidate_cached_data('world_events', world_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating world events: {e}")
    
    async def get_world_relationships(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get world relationships from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_relationships', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world relationships: {e}")
            return None
    
    async def set_world_relationships(
        self,
        world_id: str,
        relationships: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world relationships in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('world_relationships', world_id, relationships, tags)
        except Exception as e:
            logger.error(f"Error setting world relationships: {e}")
            return False
    
    async def invalidate_world_relationships(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world relationships cache."""
        try:
            await self.invalidate_cached_data('world_relationships', world_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating world relationships: {e}")
    
    async def get_world_metadata(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get world metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_metadata', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world metadata: {e}")
            return None
    
    async def set_world_metadata(
        self,
        world_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('world_metadata', world_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting world metadata: {e}")
            return False
    
    async def invalidate_world_metadata(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world metadata cache."""
        try:
            await self.invalidate_cached_data('world_metadata', world_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating world metadata: {e}")
    
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
    
    async def get_world_lore(self, world_id: int) -> Dict[str, Any]:
        """Retrieve comprehensive world lore including cultures, religions, and history."""
        cache_key = f"world_lore_{world_id}"
        cached_data = await self.get_cached_data('world', cache_key)
        if cached_data:
            return cached_data
            
        world_data = await self._execute_db_query(
            "SELECT * FROM worlds WHERE id = $1 AND user_id = $2",
            world_id, self.user_id
        )
        cultures = await self._execute_db_query(
            "SELECT * FROM cultures WHERE world_id = $1",
            world_id
        )
        religions = await self._execute_db_query(
            "SELECT * FROM religions WHERE world_id = $1",
            world_id
        )
        history = await self._execute_db_query(
            "SELECT * FROM world_history WHERE world_id = $1",
            world_id
        )
        
        result = {
            "world_details": world_data[0] if world_data else {},
            "cultures": cultures,
            "religions": religions,
            "history": history
        }
        
        await self.set_cached_data('world', cache_key, result)
        return result
        
    async def update_world_lore(self, world_id: int, updates: Dict[str, Any]) -> bool:
        """Update world lore with new information."""
        try:
            validated_data = await self._validate_data(updates, 'world')
            result = await self._execute_db_query(
                "UPDATE worlds SET $1 = $2 WHERE id = $3 AND user_id = $4",
                validated_data, world_id, self.user_id
            )
            await self.invalidate_cached_data('world', f"world_lore_{world_id}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error updating world lore: {str(e)}")
            return False
            
    async def get_cultural_context(self, culture_id: int) -> Dict[str, Any]:
        """Get detailed cultural context including traditions, customs, and beliefs."""
        return await self._execute_db_query(
            "SELECT * FROM cultural_details WHERE culture_id = $1",
            culture_id
        )
        
    async def get_religious_context(self, religion_id: int) -> Dict[str, Any]:
        """Get detailed religious context including beliefs, practices, and hierarchy."""
        return await self._execute_db_query(
            "SELECT * FROM religious_details WHERE religion_id = $1",
            religion_id
        )
        
    async def get_historical_events(self, world_id: int, time_period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve historical events, optionally filtered by time period."""
        query = "SELECT * FROM historical_events WHERE world_id = $1"
        params = [world_id]
        
        if time_period:
            query += " AND time_period = $2"
            params.append(time_period)
            
        return await self._execute_db_query(query, *params)
    
    async def _validate_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Validate data based on type and return cleaned data"""
        try:
            # Define validation schemas for different data types
            schemas = {
                'world': {
                    'required': ['name', 'description', 'history'],
                    'optional': ['tags', 'metadata'],
                    'types': {
                        'name': str,
                        'description': str,
                        'history': str,
                        'tags': list,
                        'metadata': dict
                    }
                },
                'culture': {
                    'required': ['name', 'description', 'traditions'],
                    'optional': ['beliefs', 'customs', 'language'],
                    'types': {
                        'name': str,
                        'description': str,
                        'traditions': list,
                        'beliefs': list,
                        'customs': list,
                        'language': str
                    }
                },
                'religion': {
                    'required': ['name', 'description', 'beliefs'],
                    'optional': ['practices', 'hierarchy', 'symbols'],
                    'types': {
                        'name': str,
                        'description': str,
                        'beliefs': list,
                        'practices': list,
                        'hierarchy': dict,
                        'symbols': list
                    }
                }
            }
            
            if data_type not in schemas:
                raise ValueError(f"Unknown data type: {data_type}")
            
            schema = schemas[data_type]
            validated_data = {}
            
            # Check required fields
            for field in schema['required']:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
                validated_data[field] = data[field]
            
            # Add optional fields if present
            for field in schema['optional']:
                if field in data:
                    validated_data[field] = data[field]
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise

# Create a singleton instance for easy access
world_lore_manager = WorldLoreManager()

async def get_world_lore(self, world_id: int) -> Dict[str, Any]:
    """Retrieve comprehensive world lore including cultures, religions, and history."""
    cache_key = f"world_lore_{world_id}"
    cached_data = await self.get_cached_data('world', cache_key)
    if cached_data:
        return cached_data
            
    world_data = await self._execute_db_query(
        "SELECT * FROM worlds WHERE id = $1 AND user_id = $2",
        world_id, self.user_id
    )
    cultures = await self._execute_db_query(
        "SELECT * FROM cultures WHERE world_id = $1",
        world_id
    )
    religions = await self._execute_db_query(
        "SELECT * FROM religions WHERE world_id = $1",
        world_id
    )
    history = await self._execute_db_query(
        "SELECT * FROM world_history WHERE world_id = $1",
        world_id
    )
    
    result = {
        "world_details": world_data[0] if world_data else {},
        "cultures": cultures,
        "religions": religions,
        "history": history
    }
    
    await self.set_cached_data('world', cache_key, result)
    return result
        
async def update_world_lore(self, world_id: int, updates: Dict[str, Any]) -> bool:
    """Update world lore with new information."""
    try:
        validated_data = await self._validate_data(updates, 'world')
        result = await self._execute_db_query(
            "UPDATE worlds SET $1 = $2 WHERE id = $3 AND user_id = $4",
            validated_data, world_id, self.user_id
        )
        await self.invalidate_cached_data('world', f"world_lore_{world_id}")
        return bool(result)
    except Exception as e:
        logger.error(f"Error updating world lore: {str(e)}")
        return False
            
async def get_cultural_context(self, culture_id: int) -> Dict[str, Any]:
    """Get detailed cultural context including traditions, customs, and beliefs."""
    return await self._execute_db_query(
        "SELECT * FROM cultural_details WHERE culture_id = $1",
        culture_id
    )
        
async def get_religious_context(self, religion_id: int) -> Dict[str, Any]:
    """Get detailed religious context including beliefs, practices, and hierarchy."""
    return await self._execute_db_query(
        "SELECT * FROM religious_details WHERE religion_id = $1",
        religion_id
    )
        
async def get_historical_events(self, world_id: int, time_period: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve historical events, optionally filtered by time period."""
    query = "SELECT * FROM historical_events WHERE world_id = $1"
    params = [world_id]
    
    if time_period:
        query += " AND time_period = $2"
        params.append(time_period)
        
    return await self._execute_db_query(query, *params)

async def _validate_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """Validate data based on type and return cleaned data"""
    try:
        # Define validation schemas for different data types
        schemas = {
            'world': {
                'required': ['name', 'description', 'history'],
                'optional': ['tags', 'metadata'],
                'types': {
                    'name': str,
                    'description': str,
                    'history': str,
                    'tags': list,
                    'metadata': dict
                }
            },
            'culture': {
                'required': ['name', 'description', 'traditions'],
                'optional': ['beliefs', 'customs', 'language'],
                'types': {
                    'name': str,
                    'description': str,
                    'traditions': list,
                    'beliefs': list,
                    'customs': list,
                    'language': str
                }
            },
            'religion': {
                'required': ['name', 'description', 'beliefs'],
                'optional': ['practices', 'hierarchy', 'holy_texts'],
                'types': {
                    'name': str,
                    'description': str,
                    'beliefs': list,
                    'practices': list,
                    'hierarchy': dict,
                    'holy_texts': list
                }
            },
            'history': {
                'required': ['event_name', 'description', 'date'],
                'optional': ['impact', 'participants', 'consequences'],
                'types': {
                    'event_name': str,
                    'description': str,
                    'date': str,
                    'impact': str,
                    'participants': list,
                    'consequences': list
                }
            }
        }
        
        # Get schema for data type
        schema = schemas.get(data_type)
        if not schema:
            raise ValueError(f"Unknown data type: {data_type}")
            
        # Validate required fields
        for field in schema['required']:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
        # Validate field types
        for field, value in data.items():
            if field in schema['types']:
                expected_type = schema['types'][field]
                if not isinstance(value, expected_type):
                    raise TypeError(f"Invalid type for {field}: expected {expected_type}, got {type(value)}")
                    
        # Remove any fields not in schema
        cleaned_data = {}
        for field in schema['required'] + schema['optional']:
            if field in data:
                cleaned_data[field] = data[field]
                
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error validating {data_type} data: {str(e)}")
        raise 

async def _execute_db_query(self, query: str, *args) -> Any:
    """Execute a database query with error handling and logging"""
    try:
        # Get connection from pool
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Log query for debugging
                logger.debug(f"Executing query: {query} with args: {args}")
                
                # Execute query
                if query.strip().upper().startswith('SELECT'):
                    # For SELECT queries, return all rows
                    return await conn.fetch(query, *args)
                else:
                    # For other queries, return the result
                    return await conn.execute(query, *args)
                    
    except Exception as e:
        logger.error(f"Database error executing query: {str(e)}")
        raise
            
async def get_connection_pool(self) -> asyncpg.Pool:
    """Get a connection pool for database operations"""
    if not hasattr(self, '_pool'):
        self._pool = await asyncpg.create_pool(dsn=DB_DSN)
    return self._pool 