# lore/managers/base_manager.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Set, Callable

# Agents SDK imports
from agents import Agent, Runner, function_tool, trace, RunContextWrapper, GuardrailFunctionOutput, ModelSettings
from agents.run import RunConfig

from pydantic import BaseModel, Field

# Placeholders for dependencies
from lore.data_access import BaseDataAccess
from lore.core.cache import LoreCache  
from lore.error_manager import ErrorHandler
from lore.metrics import MetricsManager

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

# Placeholder for your metrics
metrics = MetricsManager()

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
                metrics.record_cache_operation(self.__class__.__name__, True)
            else:
                metrics.record_cache_operation(self.__class__.__name__, False)

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
            metrics.record_cache_operation(self.__class__.__name__, True)
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

    @function_tool
    async def get_cached_data(
        self,
        data_type: str,
        data_id: str,
        fetch_func: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get data from cache or fetch if not available (function tool).
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
