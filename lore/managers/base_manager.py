# lore/managers/base_manager.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Set, Callable, Protocol, runtime_checkable

# Agents SDK imports
from agents import Agent, Runner, function_tool, trace, RunContextWrapper, GuardrailFunctionOutput, ModelSettings
from agents.run import RunConfig

from pydantic import BaseModel, Field

# Placeholders for dependencies
from lore.data_access import BaseDataAccess
from lore.core.cache import LoreCache  
from lore.error_manager import ErrorHandler
from lore.metrics import MetricsManager, metrics_manager as metrics

logger = logging.getLogger(__name__)

# Presumably you have references to these objects or modules:
#   - DatabaseAccess
#   - metrics
#   - etc.
# We'll treat them as placeholders.
class DatabaseAccess:
    """Placeholder for a real DB access class."""
    def __init__(self, user_id: int, conversation_id: int):
        pass
    
    async def _execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        # Dummy placeholder
        return []

    async def execute_many(self, query: str, updates: List[Any]) -> bool:
        # Dummy placeholder
        return True

# ------------------------------------------------------------------------
# Function Wrapper Classes to handle Callable in Pydantic models
# ------------------------------------------------------------------------
@runtime_checkable
class AsyncCallable(Protocol):
    """Protocol for async callables."""
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class FunctionWrapper(BaseModel):
    """Wrapper for callable functions to work with Pydantic schemas."""
    # Define this as a model to exclude from schema generation
    model_config = {
        "json_schema_extra": {"exclude": ["func"]}
    }
    
    func: Optional[Callable] = Field(default=None, exclude=True)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper itself callable."""
        if self.func is None:
            return None
        return self.func(*args, **kwargs)

# ------------------------------------------------------------------------
# Infrastructure Agent
# ------------------------------------------------------------------------
# We create an agent that can evaluate system stats and provide maintenance actions.
maintenance_agent = Agent(
    name="MaintenanceAgent",
    instructions=(
        "You analyze caching or database metrics and decide how to handle them. "
        "If the cache miss rate is too high, you might recommend clearing some keys or "
        "logging a warning. Return JSON with instructions if needed.\n\n"
        "Example:\n"
        "{\n"
        '  "action": "log_warning",\n'
        '  "message": "High cache miss rate detected"\n'
        "}"
    ),
    model="o3-mini",
    model_settings=ModelSettings(temperature=0.0)  # Typically 0 or low temp for straightforward logic
)

# ------------------------------------------------------------------------
# BaseLoreManager - now with function_tool and potential agent usage
# ------------------------------------------------------------------------
class BaseLoreManager:
    """Base class for all lore managers providing common functionality, agent-ified."""

    def __init__(self, user_id: int, conversation_id: int, cache_size: int = 100, ttl: int = 3600):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db = DatabaseAccess(user_id, conversation_id)
        self.cache = LoreCache(max_size=cache_size, ttl=ttl)
        self.error_handler = ErrorHandler()

    @function_tool
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache with metrics tracking (now exposed as a function tool).
        """
        try:
            start_time = datetime.utcnow()
            data = self.cache.get(cache_key)
            duration = (datetime.utcnow() - start_time).total_seconds()

            if data:
                await metrics.record_cache_operation(self.__class__.__name__, True)
            else:
                await metrics.record_cache_operation(self.__class__.__name__, False)

            return data
        except Exception as e:
            self.error_handler.handle_error(e)
            return None

    @function_tool
    async def _set_cached_data(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        Set data in cache with metrics tracking (now a function tool).
        """
        try:
            start_time = datetime.utcnow()
            self.cache.set(cache_key, data)
            duration = (datetime.utcnow() - start_time).total_seconds()
            await metrics.record_cache_operation(self.__class__.__name__, True)
            return True
        except Exception as e:
            self.error_handler.handle_error(e)
            return False

    @function_tool
    async def _delete_cached_data(self, cache_key: str) -> bool:
        """
        Delete data from cache (now a function tool).
        """
        try:
            self.cache.delete(cache_key)
            return True
        except Exception as e:
            self.error_handler.handle_error(e)
            return False

    @function_tool
    async def _execute_db_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute database query with metrics tracking (function tool).
        """
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
        query_lower = query.lower()
        if 'from' in query_lower:
            return query_lower.split('from')[1].split()[0]
        return 'unknown'

    @function_tool
    async def _batch_update(self, table: str, updates: List[Dict[str, Any]]) -> bool:
        """
        Perform batch update operation (function tool).
        """
        try:
            start_time = datetime.utcnow()
            # Example usage:
            # "UPDATE {table} SET column=$1 WHERE id=$2"
            # We'll assume you handle the logic in db.execute_many
            # This is placeholder code
            pairs_to_update = []
            for item in updates:
                # We'll assume item has keys 'column', 'value', 'id'
                # e.g. item = {"column": "description", "value": "New desc", "id": 123}
                column = item.get("column")
                value = item.get("value")
                row_id = item.get("id")
                pairs_to_update.append((column, value, row_id, self.user_id))

            result = await self.db.execute_many(
                f"UPDATE {table} SET $1 = $2 WHERE id = $3 AND user_id = $4",
                pairs_to_update
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

    @function_tool
    async def _validate_data(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate data against schema (function tool).
        """
        try:
            from lore.schemas import schema_validator  # example import
            return schema_validator.validate(data, schema_type)
        except Exception as e:
            self.error_handler.handle_error(e)
            raise

class LoreCacheManager:
    """
    Manager class for working with the LoreCache system.
    Provides a higher-level interface for the application-specific needs.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Convert MB to estimated entries (rough approximation)
        # Assuming average entry size of about 1KB
        estimated_entries = int(max_size_mb * 1024)
        
        # Use global cache instance or create a new one
        if redis_url:
            # If redis URL is provided, we'd set up Redis caching
            # This is a placeholder - actual Redis integration would go here
            self.cache = LoreCache(max_size=estimated_entries)
        else:
            # Use the global instance by default
            self.cache = GLOBAL_LORE_CACHE
    
    async def start(self):
        """Start the cache manager."""
        # Initialization tasks, like warming up the cache
        logger.info(f"Starting cache manager for user {self.user_id}")
    
    async def stop(self):
        """Stop the cache manager."""
        # Cleanup tasks
        logger.info(f"Stopping cache manager for user {self.user_id}")
    
    async def get_lore(self, data_type: str, data_id: str) -> Optional[Any]:
        """
        Get data from the cache.
        
        Args:
            data_type: The type of data (namespace)
            data_id: The ID of the data entry
            
        Returns:
            The cached data or None if not found
        """
        return await self.cache.get(
            namespace=data_type,
            key=data_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
    
    async def set_lore(
        self,
        data_type: str,
        data_id: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """
        Set data in the cache.
        
        Args:
            data_type: The type of data (namespace)
            data_id: The ID of the data entry
            value: The data to cache
            ttl: Optional time-to-live in seconds
            tags: Optional tags for categorizing the data
            
        Returns:
            True if successful, False otherwise
        """
        # Priority calculation based on tags
        priority = 0
        if tags:
            # Higher priority for important tags
            if "critical" in tags:
                priority = 10
            elif "important" in tags:
                priority = 7
            elif "frequently_accessed" in tags:
                priority = 5
        
        await self.cache.set(
            namespace=data_type,
            key=data_id,
            value=value,
            ttl=ttl,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            priority=priority
        )
        return True
    
    async def invalidate_lore(
        self,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """
        Invalidate cached data.
        
        Args:
            data_type: The type of data (namespace)
            data_id: Optional specific ID to invalidate
            recursive: If True, invalidate all entries with matching pattern
        """
        if data_id is not None:
            # Invalidate specific entry
            await self.cache.invalidate(
                namespace=data_type,
                key=data_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
        elif recursive:
            # Invalidate all entries in namespace
            await self.cache.clear_namespace(namespace=data_type)
        else:
            # Invalidate entries for current user in namespace
            pattern = f".*_{self.user_id}"
            if self.conversation_id:
                pattern += f"_{self.conversation_id}"
            await self.cache.invalidate_pattern(
                namespace=data_type,
                pattern=pattern
            )
    
    async def clear_all(self) -> None:
        """Clear all cached data for this user/conversation."""
        # Clear all namespaces for the current user/conversation
        pattern = f".*_{self.user_id}"
        if self.conversation_id:
            pattern += f"_{self.conversation_id}"
        
        # This is a simplified approach - a real implementation
        # might be more selective
        for namespace in self._get_all_namespaces():
            await self.cache.invalidate_pattern(
                namespace=namespace,
                pattern=pattern
            )
    
    def _get_all_namespaces(self) -> Set[str]:
        """
        Get all cache namespaces in use.
        This is a placeholder - real implementation would track namespaces.
        """
        # In a real implementation, you might store this in a registry
        return {"user_data", "conversation_data", "world_data", "entity_data"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        # Convert the CacheAnalytics data to a dict
        stats = vars(self.cache.analytics)
        
        # Add manager-specific stats
        stats.update({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return stats


# ---------------------------------------------------------------------------
# BaseManager
# ---------------------------------------------------------------------------
class BaseManager:
    """
    Base manager class with integrated caching support, also now partly agent-driven.
    """

    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id

        # Initialize a placeholder for the cache manager
        self.cache_manager = LoreCacheManager(
            user_id=user_id,
            conversation_id=conversation_id,
            max_size_mb=max_size_mb,
            redis_url=redis_url
        )

        # Cache config
        self.cache_config = {
            'ttl': 3600,
            'max_size': max_size_mb,
            'redis_url': redis_url
        }

        # Maintenance loop
        self.maintenance_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the manager and its cache, plus the agent-driven maintenance loop."""
        await self.cache_manager.start()
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        """Stop the manager and its cache, cancel maintenance."""
        await self.cache_manager.stop()
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

    # This method needs a special implementation to handle the Callable parameter
    # We'll use a non-decorated version for normal use and a decorated version for the agent
    async def get_cached_data_impl(
        self,
        data_type: str,
        data_id: str,
        fetch_func: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Implementation of get_cached_data that accepts a callable directly.
        This version is used internally and not exposed as a function tool.
        """
        try:
            cached_value = await self.cache_manager.get_lore(data_type, data_id)
            if cached_value is not None:
                return cached_value

            if fetch_func:
                value = await fetch_func()
                if value is not None:
                    await self.cache_manager.set_lore(data_type, data_id, value)
                return value

            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
            
    @function_tool
    async def get_cached_data(
        self,
        data_type: str,
        data_id: str,
        function_id: Optional[str] = None  # Use a function ID instead of a direct callable
    ) -> Optional[Any]:
        """
        Get data from cache or fetch if not available (function tool).
        This version is exposed as a function tool and uses function_id instead of direct callables.
        """
        try:
            cached_value = await self.cache_manager.get_lore(data_type, data_id)
            if cached_value is not None:
                return cached_value

            if function_id:
                # Implement a registry of functions that can be looked up by ID
                # This is a pattern to avoid passing Callables directly
                fetch_func = self._get_function_by_id(function_id)
                if fetch_func:
                    value = await fetch_func()
                    if value is not None:
                        await self.cache_manager.set_lore(data_type, data_id, value)
                    return value

            return None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
            
    def _get_function_by_id(self, function_id: str) -> Optional[Callable]:
        """
        Get a function by its ID from a registry.
        This is a placeholder - you would implement a real registry.
        """
        # Example implementation
        function_registry = {
            "get_user_data": self._fetch_user_data,
            "get_conversation_data": self._fetch_conversation_data,
            # Add more functions as needed
        }
        return function_registry.get(function_id)
        
    async def _fetch_user_data(self) -> Dict[str, Any]:
        """Example fetch function for user data."""
        # Placeholder implementation
        return {"user_id": self.user_id, "name": "Example User"}
        
    async def _fetch_conversation_data(self) -> Dict[str, Any]:
        """Example fetch function for conversation data."""
        # Placeholder implementation
        return {"conversation_id": self.conversation_id, "messages": []}

    @function_tool
    async def set_cached_data(
        self,
        data_type: str,
        data_id: str,
        value: Any,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set data in cache (function tool)."""
        try:
            return await self.cache_manager.set_lore(data_type, data_id, value, tags=tags)
        except Exception as e:
            logger.error(f"Error setting cached data: {e}")
            return False

    @function_tool
    async def invalidate_cached_data(
        self,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate cached data (function tool)."""
        try:
            await self.cache_manager.invalidate_lore(data_type, data_id, recursive=recursive)
        except Exception as e:
            logger.error(f"Error invalidating cached data: {e}")

    @function_tool
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (function tool)."""
        return self.cache_manager.get_cache_stats()

    @function_tool
    async def _maintenance_loop(self):
        """
        Agent-driven background task for maintenance. 
        We'll call the 'MaintenanceAgent' to interpret stats and advise next steps.
        """
        while True:
            try:
                stats = self.cache_manager.get_cache_stats()
                # Evaluate with MaintenanceAgent
                with trace(
                    "MaintenanceCheck",
                    metadata={"component": "BaseManagerMaintenance"}
                ):
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    prompt = (
                        "We have these cache stats:\n"
                        f"{json.dumps(stats, indent=2)}\n\n"
                        "Decide if any action is needed. Return JSON, e.g.:\n"
                        "{ \"action\": \"log_warning\", \"message\": \"High miss rate\" }\n"
                        "or { \"action\": \"none\" }"
                    )
                    run_config = RunConfig(workflow_name="MaintenanceAgentRun")
                    
                    result = await Runner.run(
                        starting_agent=maintenance_agent,
                        input=prompt,
                        context=run_ctx.context,
                        run_config=run_config
                    )
                    
                    try:
                        decision = json.loads(result.final_output)
                    except json.JSONDecodeError:
                        decision = {"action": "none"}
                    
                    if decision.get("action") == "log_warning":
                        msg = decision.get("message", "Maintenance warning triggered by agent.")
                        logger.warning(msg)
                    elif decision.get("action") == "clear_cache":
                        # For example, if the agent says to forcibly clear everything
                        logger.warning("Agent recommended clearing entire cache. Doing so now.")
                        await self.cache_manager.clear_all()
                    else:
                        # No action
                        pass

                await asyncio.sleep(300)  # Sleep 5 min
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)  # Sleep 5 min if error, then retry
