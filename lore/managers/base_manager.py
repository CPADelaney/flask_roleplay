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
    model="o3-mini",
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
        self.cache = {}  # Simplified cache for demo
        self.cache_ttl = ttl
        self.max_cache_size = cache_size
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
                model="o3-mini",
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
                model="o3-mini",
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
                model="o3-mini",
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
                model="o3-mini",
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
                model="o3-mini",
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
        """Ensure system is initialized."""
        if not self.initialized:
            # Initialize database tables and any other required setup
            # Initialize agents
            await self.initialize_agents()
            
            # Initialize governance
            await self.initialize_governance()
            
            # Start maintenance loop
            self.maintenance_task = asyncio.create_task(self._())
            
            self.initialized = True
            logger.info(f"Initialized BaseLoreManager for user {self.user_id}")

    async def initialize_tables_from_definitions(self, table_definitions: Dict[str, str]):
        """Initialize tables from provided SQL definitions."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for table_name, sql in table_definitions.items():
                    try:
                        await conn.execute(sql)
                        logger.info(f"Initialized table {table_name}")
                    except Exception as e:
                        logger.error(f"Error initializing table {table_name}: {e}")

    def get_connection_pool(self):
        """
        Get an async context manager for a db connection.
        """
        return get_db_connection_context()

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_cached_data",
        action_description="Retrieving cached lore data",
        id_from_context=lambda ctx: f"lore_cache_{int(datetime.now().timestamp())}"
    )
    async def _get_cached_data(ctx: RunContextWrapper, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache with metrics tracking and governance oversight.
        
        Args:
            ctx: Run context wrapper
            cache_key: Key to retrieve from cache
            
        Returns:
            Cached data or None if not found
        """
        try:
            start_time = datetime.now()
            data = self.cache.get(cache_key)
            duration = (datetime.now() - start_time).total_seconds()

            logger.debug(f"Cache operation for {cache_key}, hit: {data is not None}, duration: {duration}s")

            return data
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="set_cached_data",
        action_description="Setting cached lore data",
        id_from_context=lambda ctx: f"lore_cache_{int(datetime.now().timestamp())}"
    )
    async def _set_cached_data(ctx: RunContextWrapper, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        Set data in cache with metrics tracking and governance oversight.
        
        Args:
            ctx: Run context wrapper
            cache_key: Key to store in cache
            data: Data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = datetime.now()
            self.cache[cache_key] = data
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Cache set for {cache_key}, duration: {duration}s")
            
            # Check if we need to evict old entries
            if len(self.cache) > self.max_cache_size:
                await self._evict_cache_entries()
                
            return True
        except Exception as e:
            logger.error(f"Error setting cached data: {e}")
            return False
    
    async def _evict_cache_entries(self):
        """Evict old cache entries to stay within size limit."""
        try:
            # Simple LRU-like eviction - remove oldest items
            # In a real implementation, you would track timestamps
            keys = list(self.cache.keys())
            # Remove oldest entries
            keys_to_remove = keys[:len(keys) - self.max_cache_size]
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
            logger.debug(f"Evicted {len(keys_to_remove)} cache entries")
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="delete_cached_data",
        action_description="Deleting cached lore data",
        id_from_context=lambda ctx: f"lore_cache_{int(datetime.now().timestamp())}"
    )
    async def _delete_cached_data(ctx: RunContextWrapper, cache_key: str) -> bool:
        """
        Delete data from cache with governance oversight.
        
        Args:
            ctx: Run context wrapper
            cache_key: Key to delete from cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_key in self.cache:
                del self.cache[cache_key]
            return True
        except Exception as e:
            logger.error(f"Error deleting cached data: {e}")
            return False

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="execute_db_query",
        action_description="Executing database query",
        id_from_context=lambda ctx: f"lore_db_{int(datetime.now().timestamp())}"
    )
    async def _execute_db_query(ctx: RunContextWrapper, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute database query with metrics tracking and governance oversight.
        
        Args:
            ctx: Run context wrapper
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            Query results
        """
        try:
            start_time = datetime.now()
            result = await self.db._execute_query(query, *args)
            duration = (datetime.now() - start_time).total_seconds()

            # Log metrics
            logger.debug(f"DB query executed: {query}, duration: {duration}s")
            return result
        except Exception as e:
            logger.error(f"Error executing DB query: {e}")
            return []

    def _get_table_name(self, query: str) -> str:
        """Extract table name from SQL query."""
        query_lower = query.lower()
        if 'from' in query_lower:
            return query_lower.split('from')[1].split()[0]
        return 'unknown'

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="batch_update",
        action_description="Performing batch database update",
        id_from_context=lambda ctx: f"lore_batch_{int(datetime.now().timestamp())}"
    )
    async def _batch_update(ctx: RunContextWrapper, table: str, updates: List[Dict[str, Any]]) -> bool:
        """
        Perform batch update operation with governance oversight.
        
        Args:
            ctx: Run context wrapper
            table: Table to update
            updates: List of updates to perform
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = datetime.now()
            # Example usage:
            # "UPDATE {table} SET column=$1 WHERE id=$2"
            # We'll assume you handle the logic in db.execute_many
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
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Batch update on {table}, duration: {duration}s")
            return bool(result)
        except Exception as e:
            logger.error(f"Error performing batch update: {e}")
            return False

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="validate_data",
        action_description="Validating lore data",
        id_from_context=lambda ctx: f"lore_validate_{int(datetime.now().timestamp())}"
    )
    async def _validate_data(ctx: RunContextWrapper, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate data against schema with governance oversight.
        
        Args:
            ctx: Run context wrapper
            data: Data to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Validated data
        """
        try:
            # We'll use the validation agent for this
            with trace(
                workflow_name="ValidateData",
                metadata=self.trace_metadata
            ):
                user_prompt = f"""
                Validate this data against the {schema_type} schema:
                {json.dumps(data, indent=2)}
                
                Check for:
                1. Missing required fields
                2. Incorrect data types
                3. Invalid values
                4. Logical inconsistencies
                
                Return JSON with these fields:
                - is_valid: Boolean indicating if data passes validation
                - issues: Array of specific issues found (empty if none)
                - fixed_data: The corrected data if possible
                """
                
                result = await Runner.run(
                    self.agents["validation"],
                    user_prompt,
                    context=ctx.context,
                    run_config=RunConfig(
                        workflow_name="Data Validation",
                        trace_metadata=self.trace_metadata
                    )
                )
                
                validation_result = result.final_output
                if isinstance(validation_result, dict) and validation_result.get("is_valid"):
                    # If valid, return the fixed data if provided, otherwise the original
                    return validation_result.get("fixed_data", data)
                else:
                    # If not valid, log issues and raise exception
                    issues = validation_result.get("issues", ["Unknown validation error"])
                    logger.warning(f"Validation failed for {schema_type}: {issues}")
                    raise ValueError(f"Validation failed: {issues[0] if issues else 'Unknown error'}")
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            raise

    # Helper methods for derived classes
    def get_cache(self, key: str) -> Any:
        """Get item from cache."""
        return self.cache.get(key)
    
    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with optional TTL."""
        self.cache[key] = value
        
        # If we're over cache size limit, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            # This is simplistic - actual implementation would track timestamps
            keys = list(self.cache.keys())
            for old_key in keys[:len(keys) - self.max_cache_size]:
                if old_key != key:  # Don't remove what we just added
                    self.cache.pop(old_key, None)
    
    def invalidate_cache(self, key: str) -> None:
        """Invalidate a specific cache key."""
        self.cache.pop(key, None)
    
    def invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate all cache keys matching a pattern."""
        keys_to_remove = []
        for cache_key in self.cache.keys():
            if pattern in cache_key:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            self.cache.pop(key, None)
            
    def create_run_context(self, ctx):
        """Create a run context for agent execution."""
        if isinstance(ctx, RunContextWrapper):
            return ctx
        return RunContextWrapper(context=ctx)

    @staticmethod
    @function_tool
    async def get_cache_stats(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            ctx: Run context wrapper
            
        Returns:
            Cache statistics
        """
        # Calculate hit/miss rate
        total_operations = len(self.cache)
        hit_rate = 0.0  # Would be calculated based on actual hits/misses tracking
        
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "utilization": len(self.cache) / self.max_cache_size if self.max_cache_size > 0 else 0,
            "hit_rate": hit_rate,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _maintenance_loop(self):
        while True:
            try:
                await self._maintenance_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
            await asyncio.sleep(300)  # Sleep 5 min
    
    async def _maintenance_once(self):
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        stats = await self.get_cache_stats(run_ctx)
        
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
                self.cache.clear()
            # No else/else-pass needed
    
    @function_tool
    async def maintenance_loop_tool(self):
        """
        Agent-exposed maintenance pass (runs ONCE).
        """
        result = await self._maintenance_once()
        return {"status": "completed", "result": result}
                
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
        """
        Generate foundation lore for a given environment with governance oversight.
        
        Args:
            ctx: Run context wrapper
            environment_desc: Environment description
            
        Returns:
            Foundation lore data
        """
        try:
            # Make sure we're initialized
            await self.ensure_initialized()
            
            # Use the agent to generate foundation lore
            with trace(
                workflow_name="GenerateFoundationLore",
                metadata=self.trace_metadata,
            ):
                user_prompt = f"""
                Generate cohesive foundational world lore for this environment:
                {environment_desc}

                Return as JSON with these keys:
                - cosmology (description of universe/planes/gods)
                - magic_system (how magic works, limitations, schools)
                - world_history (major eras and events)
                - calendar_system (how time is tracked)
                - social_structure (class systems, hierarchy)

                Be creative but ensure all elements are cohesive and interconnected.
                """
                
                result = await Runner.run(
                    self.agents["foundation"],
                    user_prompt,
                    context=ctx.context,
                    run_config=RunConfig(
                        workflow_name="Foundation Lore Generation",
                        trace_metadata=self.trace_metadata
                    )
                )
                
                # Cache the result
                foundation_lore = result.final_output
                await self._set_cached_data(ctx, "foundation_lore_latest", foundation_lore)
                
                return foundation_lore
        except Exception as e:
            logger.error(f"Error generating foundation lore: {str(e)}")
            return {
                "error": str(e),
                "cosmology": "Error generating cosmology",
                "magic_system": "Error generating magic system",
                "world_history": "Error generating world history",
                "calendar_system": "Error generating calendar system",
                "social_structure": "Error generating social structure"
            }

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
        """
        Generate factions for a given environment with governance oversight.
        
        Args:
            ctx: Run context wrapper
            environment_desc: Environment description
            foundation_lore: Foundation lore data
            num_factions: Number of factions to generate
            
        Returns:
            List of faction data
        """
        try:
            # Make sure we're initialized
            await self.ensure_initialized()
            
            # Use the agent to generate factions
            with trace(
                workflow_name="GenerateFactions",
                metadata=self.trace_metadata,
            ):
                social_structure = foundation_lore.get("social_structure", "")
                world_history = foundation_lore.get("world_history", "")
                
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
                    self.agents["faction"],
                    user_prompt,
                    context=ctx.context,
                    run_config=RunConfig(
                        workflow_name="Faction Generation",
                        trace_metadata=self.trace_metadata
                    )
                )
                
                # Process and cache the result
                factions = result.final_output
                
                # Cache the full faction list
                await self._set_cached_data(ctx, "factions_latest", factions)
                
                return factions
        except Exception as e:
            logger.error(f"Error generating factions: {str(e)}")
            return [{"error": str(e), "name": "Error generating factions"}]

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
        """
        Generate locations for a given environment with governance oversight.
        
        Args:
            ctx: Run context wrapper
            environment_desc: Environment description
            factions: List of faction data
            num_locations: Number of locations to generate
            
        Returns:
            List of location data
        """
        try:
            # Make sure we're initialized
            await self.ensure_initialized()
            
            # Use the agent to generate locations
            with trace(
                workflow_name="GenerateLocations",
                metadata=self.trace_metadata,
            ):
                faction_names = []
                for faction in factions:
                    if isinstance(faction, dict) and "name" in faction:
                        faction_names.append(faction["name"])
                
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
                    self.agents["location"],
                    user_prompt,
                    context=ctx.context,
                    run_config=RunConfig(
                        workflow_name="Location Generation",
                        trace_metadata=self.trace_metadata
                    )
                )
                
                # Process and cache the result
                locations = result.final_output
                
                # Cache the full location list
                await self._set_cached_data(ctx, "locations_latest", locations)
                
                return locations
        except Exception as e:
            logger.error(f"Error generating locations: {str(e)}")
            return [{"error": str(e), "name": "Error generating locations"}]

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
        """
        Generate complete lore for an environment with governance oversight.
        
        Args:
            ctx: Run context wrapper
            environment_desc: Environment description
            
        Returns:
            Complete lore data
        """
        try:
            # Make sure we're initialized
            await self.ensure_initialized()
            
            # Use trace to track the entire workflow
            with trace(
                workflow_name="GenerateCompleteLore",
                metadata=self.trace_metadata,
            ):
                # 1. Generate foundation lore
                foundation_lore = await self.generate_foundation_lore(ctx, environment_desc)
                
                # 2. Generate factions
                factions = await self.generate_factions(ctx, environment_desc, foundation_lore)
                
                # 3. Generate locations
                locations = await self.generate_locations(ctx, environment_desc, factions)
                
                # 4. Create the complete lore object
                complete_lore = {
                    "environment_desc": environment_desc,
                    "foundation": foundation_lore,
                    "factions": factions,
                    "locations": locations,
                    "generated_at": datetime.now().isoformat()
                }
                
                # Cache the complete lore
                await self._set_cached_data(ctx, "complete_lore_latest", complete_lore)
                
                return complete_lore
        except Exception as e:
            logger.error(f"Error generating complete lore: {str(e)}")
            return {
                "error": str(e),
                "environment_desc": environment_desc,
                "generated_at": datetime.now().isoformat()
            }
            
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

# ------------------------------------------------------------------------
# LoreCache class (simplified)
# ------------------------------------------------------------------------
class LoreCache:
    """Basic cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize the cache."""
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.analytics = CacheAnalytics()
    
    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        if key in self.cache:
            self.analytics.hits += 1
            return self.cache[key]
        self.analytics.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        self.cache[key] = value
        self.analytics.sets += 1
        
        # Simple eviction if over size
        if len(self.cache) > self.max_size:
            keys = list(self.cache.keys())
            for old_key in keys[:len(keys) - self.max_size]:
                if old_key != key:
                    self.delete(old_key)
    
    def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        if key in self.cache:
            del self.cache[key]
            self.analytics.deletes += 1

    async def get(self, namespace: str, key: str, user_id: int, conversation_id: int) -> Any:
        """Get a value with namespace and user context."""
        full_key = f"{namespace}_{key}_{user_id}_{conversation_id}"
        return self.get(full_key)
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None,
                user_id: int = 0, conversation_id: int = 0, priority: int = 0) -> None:
        """Set a value with namespace and user context."""
        full_key = f"{namespace}_{key}_{user_id}_{conversation_id}"
        self.set(full_key, value)
    
    async def invalidate(self, namespace: str, key: str, user_id: int, conversation_id: int) -> None:
        """Invalidate a cache entry."""
        full_key = f"{namespace}_{key}_{user_id}_{conversation_id}"
        self.delete(full_key)
    
    async def clear_namespace(self, namespace: str) -> None:
        """Clear all entries in a namespace."""
        keys_to_remove = []
        for key in self.cache.keys():
            if key.startswith(f"{namespace}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.delete(key)
    
    async def invalidate_pattern(self, namespace: str, pattern: str) -> None:
        """Invalidate all keys matching a pattern within a namespace."""
        prefix = f"{namespace}_"
        keys_to_remove = []
        for key in self.cache.keys():
            if key.startswith(prefix) and pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.delete(key)

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
        await self.cache_manager.start()
        
        # Create an agent context for the maintenance loop
        ctx = RunContextWrapper(context={})
        
        # Start the maintenance loop
        self.maintenance_task = asyncio.create_task(self._maintenance_loop(ctx))

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

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_cached_data",
        action_description="Getting cached data {data_type}/{data_id}",
        id_from_context=lambda ctx: f"get_cached_{ctx.data_type}_{ctx.data_id}"
    )
    async def get_cached_data(
        ctx: RunContextWrapper,
        data_type: str,
        data_id: str,
        function_id: Optional[str] = None  # Use a function ID instead of a direct callable
    ) -> Optional[Any]:
        """
        Get data from cache or fetch if not available (function tool).
        This version is exposed as a function tool and uses function_id instead of direct callables.
        
        Args:
            ctx: Run context wrapper
            data_type: Type of data to retrieve
            data_id: ID of the data
            function_id: Optional function ID to call if data is not in cache
        
        Returns:
            Cached data or None if not found
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

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="set_cached_data",
        action_description="Setting cached data {data_type}/{data_id}",
        id_from_context=lambda ctx: f"set_cached_{ctx.data_type}_{ctx.data_id}"
    )
    async def set_cached_data(
        ctx: RunContextWrapper,
        data_type: str,
        data_id: str,
        value: Any,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """
        Set data in cache with governance oversight.
        
        Args:
            ctx: Run context wrapper
            data_type: Type of data to store
            data_id: ID of the data
            value: Data to store
            tags: Optional tags for the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.cache_manager.set_lore(data_type, data_id, value, tags=tags)
        except Exception as e:
            logger.error(f"Error setting cached data: {e}")
            return False

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="invalidate_cached_data",
        action_description="Invalidating cached data {data_type}/{data_id}",
        id_from_context=lambda ctx: f"invalidate_cached_{ctx.data_type}_{ctx.data_id}"
    )
    async def invalidate_cached_data(
        ctx: RunContextWrapper,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """
        Invalidate cached data with governance oversight.
        
        Args:
            ctx: Run context wrapper
            data_type: Type of data to invalidate
            data_id: Optional ID of the data to invalidate
            recursive: If True, invalidate all entries with matching pattern
        """
        try:
            await self.cache_manager.invalidate_lore(data_type, data_id, recursive=recursive)
        except Exception as e:
            logger.error(f"Error invalidating cached data: {e}")

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_cache_stats",
        action_description="Getting cache statistics",
        id_from_context=lambda ctx: f"get_cache_stats_{int(datetime.now().timestamp())}"
    )
    async def get_cache_stats(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get cache statistics with governance oversight.
        
        Args:
            ctx: Run context wrapper
            
        Returns:
            Cache statistics
        """
        return self.cache_manager.get_cache_stats()

    @staticmethod
    @function_tool
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="maintenance_loop",
        action_description="Running maintenance loop",
        id_from_context=lambda ctx: f"maintenance_loop_{int(datetime.now().timestamp())}"
    )
    async def _maintenance_loop(ctx: RunContextWrapper):
        """
        Agent-driven background task for maintenance with governance oversight. 
        We'll call the 'MaintenanceAgent' to interpret stats and advise next steps.
        
        Args:
            ctx: Run context wrapper
        """
        while True:
            try:
                stats = self.cache_manager.get_cache_stats()
                
                # Evaluate with MaintenanceAgent
                with trace(
                    "MaintenanceCheck",
                    metadata={"component": "BaseManager"}
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
