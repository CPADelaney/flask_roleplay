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

# Nyx governance integration
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Database connection
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

def _mgr(ctx: RunContextWrapper):
    mgr = ctx.context.get("manager")
    if mgr is None:
        raise RuntimeError("manager instance missing from ctx.context")
    return mgr

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
# Pydantic Models for Agent SDK Integration
# ------------------------------------------------------------------------
class CacheStats(BaseModel):
    """Cache statistics model"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None

class LoreComponentData(BaseModel):
    """Base model for lore component data"""
    id: str
    name: str
    type: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class QueryInput(BaseModel):
    """Input model for queries"""
    query: str
    min_relevance: float = 0.6
    limit: int = 5
    lore_types: Optional[List[str]] = None

class MaintenanceResult(BaseModel):
    """Result model for maintenance operations"""
    action: str
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

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
    model="o4-mini",
    model_settings=ModelSettings(temperature=0.0)  # Typically 0 or low temp for straightforward logic
)

# ------------------------------------------------------------------------
# Database Access Stub
# ------------------------------------------------------------------------
class DatabaseAccess:
    """Database access class."""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def _execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a database query."""
        logger.debug(f"Executing query: {query}")
        
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetch(query, *args)
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error executing database query: {str(e)}")
            return []

    async def execute_many(self, query: str, updates: List[Any]) -> bool:
        """Execute multiple updates in a batch."""
        logger.debug(f"Executing batch update: {query}")
        
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    for update in updates:
                        await conn.execute(query, *update)
                return True
        except Exception as e:
            logger.error(f"Error executing batch update: {str(e)}")
            return False

# ------------------------------------------------------------------------
# BaseLoreManager - now with full function_tool and agent usage
# ------------------------------------------------------------------------
class BaseLoreManager:
    """Base class for all lore managers providing common functionality, agent-ified."""

    def __init__(self, user_id: int, conversation_id: int, cache_size: int = 100, ttl: int = 3600):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db = DatabaseAccess(user_id, conversation_id)
        self._cache: Dict[str, Any] = {}
        self._cache_max = cache_size
        self.initialized = False
        self.trace_group_id = f"lore_{user_id}_{conversation_id}"
        self.trace_metadata = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "component": "BaseLoreManager"
        }
        
        # Agent definitions
        self.agents = {}
        
        # Governance integration
        self.governor = None
        self.directive_handler = None
        
        # Maintenance task
        self.maintenance_task: Optional[asyncio.Task] = None

    async def initialize_agents(self):
        """Initialize agent definitions using OpenAI Agent SDK."""
        try:
            # Foundation lore agent
            self.agents["foundation"] = Agent(
                name="FoundationLoreAgent",
                instructions=(
                    "You produce foundational world lore for a fantasy environment. "
                    "Generate detailed, cohesive world foundations including cosmology, "
                    "magic systems, social structures, and history."
                ),
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            # Faction agent
            self.agents["faction"] = Agent(
                name="FactionAgent",
                instructions=(
                    "You create detailed factions for a game world. For each faction, "
                    "provide a name, type, description, values, goals, headquarters, "
                    "rivals, allies, and hierarchy_type. Ensure factions have clear "
                    "motivations and logical relationships with each other."
                ),
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            # Location agent
            self.agents["location"] = Agent(
                name="LocationAgent",
                instructions=(
                    "You create detailed game locations. For each location, provide "
                    "a name, type, description, controlling faction, notable features, "
                    "hidden secrets, and strategic importance. Make locations feel lived-in "
                    "with history and character."
                ),
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            # Cultural agent
            self.agents["cultural"] = Agent(
                name="CulturalAgent",
                instructions=(
                    "You create cultural elements like traditions, customs, rituals, "
                    "and social practices for a game world. Each element should have a "
                    "name, type, description, who practices it, and historical origins."
                ),
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            # Validation agent
            self.agents["validation"] = Agent(
                name="ValidationAgent",
                instructions=(
                    "You validate lore for consistency and quality. Check for "
                    "contradictions, logical issues, and areas that need improvement. "
                    "Provide specific issues and recommendations."
                ),
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.4)
            )
            
            logger.info(f"Agents initialized for BaseLoreManager user {self.user_id}")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def initialize_governance(self):
        """Initialize Nyx governance integration."""
        try:
            # Import here to avoid circular imports
            from nyx.integrate import get_central_governance
            
            # Get central governance
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize directive handler
            self.directive_handler = DirectiveHandler(
                self.user_id,
                self.conversation_id,
                AgentType.NARRATIVE_CRAFTER,
                "base_lore_manager"
            )
            
            # Register with governance system
            await self.governor.register_agent(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="base_lore_manager",
                agent_instance=self
            )
            
            logger.info(f"Governance initialized for BaseLoreManager user {self.user_id}")
        except Exception as e:
            logger.error(f"Error initializing governance: {str(e)}")
            raise

    async def ensure_initialized(self):
        if self.initialized:
            return
        await self.initialize_agents()
        await self.initialize_governance()
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.initialized = True

    async def initialize_tables_from_definitions(self, defs: Dict[str, str]):
        async with get_db_connection_context() as conn:
            for name, sql in defs.items():
                try:
                    await conn.execute(sql)
                    logger.info("table %s ready", name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("init table %s failed: %s", name, exc)

    # internal helper used by subclasses
    async def _initialize_tables_for_class_impl(self, defs: Dict[str, str]):
        await self.initialize_tables_from_definitions(defs)

    @staticmethod
    @function_tool
    async def initialize_tables_for_class(ctx: RunContextWrapper, table_definitions: Dict[str, str]):
        mgr = _mgr(ctx)
        await mgr._initialize_tables_for_class_impl(table_definitions)
        return {"status": "ok"}

    def get_connection_pool(self):
        """
        Get an async context manager for a db connection.
        """
        return get_db_connection_context()

    async def _execute_query_impl(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Run an arbitrary SQL statement and return the rows as plain dicts.
    
        • Uses the classʼs shared connection‑pool helper
        • Catches and logs all errors; returns an empty list on failure
        """
        try:
            async with self.get_connection_pool() as conn:
                records = await conn.fetch(query, *args)
                return [dict(r) for r in records]
        except Exception as exc:
            logger.error("DB query failed (%s): %s", query, exc)
            return []

    def _cache_get(self, key: str):
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any):
        self._cache[key] = value
        if len(self._cache) > self._cache_max:
            # naive eviction: pop first key
            self._cache.pop(next(iter(self._cache)))

    def _cache_delete(self, key: str):
        self._cache.pop(key, None)

    @staticmethod
    async def _set_cached_data(ctx: RunContextWrapper, cache_key: str, data: Any) -> None:
        """
        Agent‑friendly cache setter that just forwards to the internal store.
        """
        mgr = _mgr(ctx)
        mgr._cache_set(cache_key, data)
        

    @staticmethod
    @function_tool
    async def get_cached_data(ctx: RunContextWrapper, cache_key: str):
        return _mgr(ctx)._cache_get(cache_key)

    @staticmethod
    @function_tool
    async def set_cached_data(ctx: RunContextWrapper, cache_key: str, data: Any):
        _mgr(ctx)._cache_set(cache_key, data)
        return True

    @staticmethod
    @function_tool
    async def delete_cached_data(ctx: RunContextWrapper, cache_key: str):
        _mgr(ctx)._cache_delete(cache_key)
        return True

    @staticmethod
    @function_tool
    async def execute_db_query(ctx: RunContextWrapper, query: str, *args):
        return await _mgr(ctx)._execute_query_impl(query, *args)

    @staticmethod
    @function_tool
    async def batch_update(ctx: RunContextWrapper, table: str, updates: List[Dict[str, Any]]):
        return await _mgr(ctx)._batch_update_impl(table, updates)

    async def _batch_update_impl(self, table: str, updates: List[Dict[str, Any]]) -> int:
        """
        Extremely generic batch updater.
        Each item must have:  {"column": "<col>", "value": <val>, "id": <row_id>}
        Returns the number of rows changed.
        """
        if not updates:
            return 0
    
        async with self.get_connection_pool() as conn:
            async with conn.transaction():
                for row in updates:
                    col  = row["column"]
                    sql  = f'UPDATE {table} SET {col} = $1 WHERE id = $2'
                    await conn.execute(sql, row["value"], row["id"])
        return len(updates)

    # ---------------- Validate data -----------------------------------
    async def _validate_data_impl(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """Dummy validation – replace with real logic / agent call."""
        # Example: ensure required keys exist
        required = {"foundation": ["cosmology", "magic_system"],
                    "faction": ["name", "type"]}.get(schema_type, [])
        issues = [k for k in required if k not in data]
        return {
            "is_valid": not issues,
            "issues": issues,
            "fixed_data": data,
        }

    @staticmethod
    @function_tool
    async def validate_data(ctx: RunContextWrapper, data: Dict[str, Any], schema_type: str):
        return await _mgr(ctx)._validate_data_impl(data, schema_type)
            
    def create_run_context(self, ctx):
        """Create a run context for agent execution."""
        if isinstance(ctx, RunContextWrapper):
            return ctx
        return RunContextWrapper(context=ctx)

    def _get_cache_stats(self) -> Dict[str, Any]:
        # This is your working implementation
        return {
            "size": len(self._cache),
            "max_size": self._cache_max,
            "utilization": len(self._cache) / self._cache_max if self._cache_max > 0 else 0,
            # You can add proper hit/miss rate if you track it in your cache
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        }

    # THIS is the ONLY agent-tool-exposed version!
    @function_tool
    async def get_cache_stats(ctx: RunContextWrapper) -> dict:
        # retrieve manager instance from context!
        manager = ctx.context["manager"]  # Or whatever key you use
        return manager._get_cache_stats()
    
    async def _maintenance_loop(self):
        while True:
            await asyncio.sleep(300)
            logger.debug("cache utilisation %s/%s", len(self._cache), self._cache_max)
    
    async def _maintenance_once(self):
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        stats = self._get_cache_stats()
        with trace(
            "MaintenanceCheck",
            metadata={"component": "BaseLoreManager"}
        ):
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
                if isinstance(result.final_output, dict):
                    decision = result.final_output
                else:
                    decision = json.loads(result.final_output)
            except json.JSONDecodeError:
                decision = {"action": "none"}
            if decision.get("action") == "log_warning":
                msg = decision.get("message", "Maintenance warning triggered by agent.")
                logger.warning(msg)
            elif decision.get("action") == "clear_cache":
                logger.warning("Agent recommended clearing entire cache. Doing so now.")
                self._cache.clear()
            # No else/else-pass needed
    
    @function_tool
    async def maintenance_tool(self):
        """Agent-accessible: run a single maintenance pass on demand."""
        await self._maintenance_once()
        return {"status": "completed"}

    async def close(self):
        """
        Close the manager and clean up resources.
        """
        # Cancel maintenance task if running
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

    def as_context(self) -> RunContextWrapper:
        """Create a context object embedding *this* manager."""
        return RunContextWrapper(context={"manager": self})

                
    # Enhanced lore generation methods
    @staticmethod
    @function_tool
    @with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_foundation_lore",
    action_description="Generating foundation lore for environment: {environment_desc}",
    id_from_context=lambda ctx: f"foundation_lore_{int(datetime.now().timestamp())}"
    )
    async def generate_foundation_lore(
    ctx: RunContextWrapper,
    environment_desc: str
    ) -> Dict[str, Any]:
        mgr = _mgr(ctx)
        await mgr.ensure_initialized()
        
        with trace(workflow_name="GenerateFoundationLore",
               metadata=mgr.trace_metadata):
               user_prompt = f"""
               Generate cohesive foundational world lore for this environment:
               {environment_desc}
                
               Return as JSON with these keys:
               - cosmology
               - magic_system
               - world_history
               - calendar_system
               - social_structure
               """
               result = await Runner.run(
                   mgr.agents["foundation"],
                   user_prompt,
                   context=ctx.context,
                   run_config=RunConfig(
                       workflow_name="Foundation Lore Generation",
                       trace_metadata=mgr.trace_metadata
                   )
               )
                
               foundation_lore = result.final_output
               await BaseLoreManager._set_cached_data(ctx, "foundation_lore_latest", foundation_lore)
               return foundation_lore
    
    @staticmethod
    @function_tool
    @with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_factions",
    action_description="Generating factions for environment",
    id_from_context=lambda ctx: f"factions_{int(datetime.now().timestamp())}"
    )
    async def generate_factions(
    ctx: RunContextWrapper,
    environment_desc: str,
    foundation_lore: Dict[str, Any],
    num_factions: int = 5
    ) -> List[Dict[str, Any]]:
        mgr = _mgr(ctx)
        await mgr.ensure_initialized()
        
        social_structure = foundation_lore.get("social_structure", "")
        world_history   = foundation_lore.get("world_history", "")
        
        user_prompt = f"""
        Generate {num_factions} distinct factions for this environment:
        
        Environment: {environment_desc}
        Social Structure: {social_structure}
        World History: {world_history}
        
        For each faction, provide:
        - name: A unique name
        - type: The faction type (guild, nation, cult, etc.)
        - description: A paragraph describing the faction
        - values: List of 3-5 values the faction holds
        - goals: List of 2-3 current goals
        - headquarters: Their main base of operations
        - rivals: List of 1-3 other factions they oppose
        - allies: List of 0-2 other factions they ally with
        - hierarchy_type: How they're organized internally
        
        Return as a JSON array of faction objects.
        Ensure factions have distinct personalities and interesting relationships.
        """
        
        result = await Runner.run(
            mgr.agents["faction"],
            user_prompt,
            context=ctx.context,
            run_config=RunConfig(
                workflow_name="Faction Generation",
                trace_metadata=mgr.trace_metadata
            )
        )
        
        factions = result.final_output
        await BaseLoreManager._set_cached_data(ctx, "factions_latest", factions)
        return factions
    
    @staticmethod
    @function_tool
    @with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_locations",
    action_description="Generating locations for environment",
    id_from_context=lambda ctx: f"locations_{int(datetime.now().timestamp())}"
    )
    async def generate_locations(
    ctx: RunContextWrapper,
    environment_desc: str,
    factions: List[Dict[str, Any]],
    num_locations: int = 8
    ) -> List[Dict[str, Any]]:
        mgr = _mgr(ctx)
        await mgr.ensure_initialized()
        
        faction_names = [f.get("name", "") for f in factions if isinstance(f, dict)]
        
        
        user_prompt = f"""
        Generate {num_locations} significant locations for this environment:
        
        Environment: {environment_desc}
        Factions: {', '.join(faction_names)}
        
        For each location, provide:
        - name: A unique name
        - type: The location type (city, dungeon, forest, etc.)
        - description: A detailed description
        - controlling_faction: Which faction controls this location (can be null)
        - notable_features: List of 3-5 interesting features
        - hidden_secrets: List of 1-3 secrets about this location
        - strategic_importance: A number from 1-10 indicating importance
        
        Return as a JSON array of location objects.
        Create a diverse mix of locations with unique characteristics.
        Some locations should be controlled by factions, others contested or neutral.
        """
        
        result = await Runner.run(
            mgr.agents["location"],
            user_prompt,
            context=ctx.context,
            run_config=RunConfig(
                workflow_name="Location Generation",
                trace_metadata=mgr.trace_metadata
            )
        )
        
        locations = result.final_output
        await BaseLoreManager._set_cached_data(ctx, "locations_latest", locations)
        return locations
        
    @staticmethod
    @function_tool
    @with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_complete_lore",
    action_description="Generating complete lore for environment: {environment_desc}",
    id_from_context=lambda ctx: f"complete_lore_{int(datetime.now().timestamp())}"
    )
    async def generate_complete_lore(
    ctx: RunContextWrapper,
    environment_desc: str
    ) -> Dict[str, Any]:
        mgr = _mgr(ctx)
        await mgr.ensure_initialized()
        
        with trace(workflow_name="GenerateCompleteLore",
                   metadata=mgr.trace_metadata):
        
            foundation = await BaseLoreManager.generate_foundation_lore(ctx, environment_desc)
            factions   = await BaseLoreManager.generate_factions(ctx, environment_desc, foundation)
            locations  = await BaseLoreManager.generate_locations(ctx, environment_desc, factions)
        
            complete_lore = {
                "environment_desc": environment_desc,
                "foundation": foundation,
                "factions": factions,
                "locations": locations,
                "generated_at": datetime.now().isoformat()
            }
        
        await BaseLoreManager._set_cached_data(ctx, "complete_lore_latest", complete_lore)
        return complete_lore
                


# ------------------------------------------------------------------------
# LoreCache class (simplified)
# ------------------------------------------------------------------------
class LoreCache:
    """Basic cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize the cache."""
        self._cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.analytics = CacheAnalytics()
    
    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        if key in self._cache:
            del self._cache[key]
            self.analytics.deletes += 1

    async def get(self, namespace: str, key: str, *, user_id:int, conversation_id:int):
        full = f"{namespace}_{key}_{user_id}_{conversation_id}"
        if full in self._cache:
            self.analytics.hits += 1
            return self._cache[full]
        self.analytics.misses += 1
        return None
    
    async def set(self, namespace: str, key: str, value: Any, *, user_id:int,
                  conversation_id:int, ttl:int|None=None, priority:int=0):
        full = f"{namespace}_{key}_{user_id}_{conversation_id}"
        self._cache[full] = value
        self.analytics.sets += 1
        if len(self._cache) > self.max_size:
            self._cache.pop(next(iter(self._cache)))
    
    async def invalidate(self, namespace: str, key: str, user_id: int, conversation_id: int) -> None:
        """Invalidate a cache entry."""
        full_key = f"{namespace}_{key}_{user_id}_{conversation_id}"
        await self.delete(full_key)
    
    async def clear_namespace(self, namespace: str) -> None:
        """Clear all entries in a namespace."""
        keys_to_remove = []
        for key in self._cache.keys():
            if key.startswith(f"{namespace}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self.delete(key)
    
    async def invalidate_pattern(self, namespace: str, pattern: str) -> None:
        """Invalidate all keys matching a pattern within a namespace."""
        prefix = f"{namespace}_"
        keys_to_remove = []
        for key in self._cache.keys():
            if key.startswith(prefix) and pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self.delete(key)

# Global cache instance
class CacheAnalytics:
    """Analytics for cache operations."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0

# Global cache instance
GLOBAL_LORE_CACHE = LoreCache()

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
        self._cache_manager = LoreCacheManager(
            user_id=user_id,
            conversation_id=conversation_id,
            max_size_mb=max_size_mb,
            redis_url=redis_url
        )

        # Cache config
        self._cache_config = {
            'ttl': 3600,
            'max_size': max_size_mb,
            'redis_url': redis_url
        }
        
        # Trace metadata
        self.trace_metadata = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "component": "BaseManager"
        }

        # Maintenance loop
        self.maintenance_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the manager and its cache, plus the agent-driven maintenance loop."""
        await self._cache_manager.start()
        
        # Create an agent context for the maintenance loop
        ctx = RunContextWrapper(context={})
        
        # Start the maintenance loop
        self.maintenance_task = asyncio.create_task(self._maintenance_loop(ctx))

    async def stop(self):
        """Stop the manager and its cache, cancel maintenance."""
        await self._cache_manager.stop()
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

    async def _maintenance_loop(self, ctx: RunContextWrapper) -> None:
        """
        Very simple heartbeat – extend later if you want real logic.
        """
        while True:
            await asyncio.sleep(300)          # every 5 min
            size = len(self._cache_manager._cache._cache)  # type: ignore
            logger.debug("BaseManager cache utilisation: %s entries", size)

    
    def _get_function_by_id(self, function_id: str) -> Optional[Callable]:
        """
        Get a function by its ID from a registry.
        This is a placeholder - you would implement a real registry.
        
        Args:
            function_id: ID of the function to retrieve
            
        Returns:
            Function callable or None if not found
        """
        # Example implementation
        function_registry = {
            "get_user_data": self._fetch_user_data,
            "get_conversation_data": self._fetch_conversation_data,
            # Add more functions as needed
        }
        return function_registry.get(function_id)
        
    async def _fetch_user_data(self) -> Dict[str, Any]:
        """
        Example fetch function for user data.
        
        Returns:
            User data
        """
        # Implement your actual user data fetch logic
        return {"user_id": self.user_id, "name": "Example User"}
        
    async def _fetch_conversation_data(self) -> Dict[str, Any]:
        """
        Example fetch function for conversation data.
        
        Returns:
            Conversation data
        """
        # Implement your actual conversation data fetch logic
        return {"conversation_id": self.conversation_id, "messages": []}

# ---------------------------------------------------------------------------
# LoreCacheManager
# ---------------------------------------------------------------------------
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
            self._cache = LoreCache(max_size=estimated_entries)
        else:
            # Use the global instance by default
            self._cache = GLOBAL_LORE_CACHE
    
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
        return await self._cache.get(
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
        
        await self._cache.set(
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
            await self._cache.invalidate(
                namespace=data_type,
                key=data_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
        elif recursive:
            # Invalidate all entries in namespace
            await self._cache.clear_namespace(namespace=data_type)
        else:
            # Invalidate entries for current user in namespace
            pattern = f".*_{self.user_id}"
            if self.conversation_id:
                pattern += f"_{self.conversation_id}"
            await self._cache.invalidate_pattern(
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
            await self._cache.invalidate_pattern(
                namespace=namespace,
                pattern=pattern
            )
    
    def _get_all_namespaces(self) -> Set[str]:
        """
        Get all cache namespaces in use.
        """
        # In a real implementation, you might store this in a registry
        return {"user_data", "conversation_data", "world_data", "entity_data"}
    
