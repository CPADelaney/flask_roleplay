# lore/core/base_manager.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Set, Callable, Protocol, runtime_checkable, Union, Tuple

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

# Embedding service
from embedding.vector_store import generate_embedding, compute_similarity

# Cache system
from lore.core.cache import GLOBAL_LORE_CACHE

logger = logging.getLogger(__name__)

def _mgr(ctx: RunContextWrapper):
    """Helper to retrieve manager instance from context."""
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
# BaseLoreManager - Consolidated Implementation
# ------------------------------------------------------------------------
class BaseLoreManager:
    """
    Consolidated base class for all lore managers providing common functionality.
    Integrates agent capabilities, database access, caching, and governance oversight.
    """

    def __init__(self, user_id: int, conversation_id: int, cache_size: int = 100, ttl: int = 3600):
        """
        Initialize the base lore manager.
        
        Args:
            user_id: ID of the user
            conversation_id: ID of the conversation
            cache_size: Maximum number of items to cache
            ttl: Default time-to-live for cache items in seconds
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._cache: Dict[str, Any] = {}
        self._cache_max = cache_size
        self.initialized = False
        self.cache_namespace = self.__class__.__name__.lower()
        
        # Governance integration
        self.governor = None
        self.directive_handler = None
        
        # Define standard table columns for common operations
        self._standard_columns = {
            'id': 'SERIAL PRIMARY KEY',
            'name': 'TEXT NOT NULL',
            'description': 'TEXT NOT NULL',
            'timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'embedding': 'VECTOR(1536)'
        }
        
        # Set up tracing
        self._setup_tracing()
        
        # Initialize agent definitions
        self.agents = {}
        
        # Maintenance task
        self.maintenance_task: Optional[asyncio.Task] = None

    def _setup_tracing(self):
        """Set up naming and metadata for traces from this manager instance."""
        self.trace_name = f"{self.__class__.__name__}Workflow"
        self.trace_group_id = f"lore_{self.user_id}_{self.conversation_id}"
        self.trace_metadata = {
            "user_id": str(self.user_id),
            "conversation_id": str(self.conversation_id),
            "component": self.__class__.__name__
        }

    async def ensure_initialized(self):
        """
        Full startup: agents → governance → tables → maintenance loop.
        """
        if self.initialized:
            return

        await self.initialize_agents()
        await self.initialize_governance()
        await self._initialize_tables()
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.initialized = True

    async def initialize_agents(self):
        """
        Initialize agent definitions. Override in subclasses to define specific agents.
        """
        try:
            # Example agent definition - subclasses should override this
            self.agents["foundation"] = Agent(
                name="FoundationAgent",
                instructions="You are a general-purpose agent for lore management.",
                model="o4-mini",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            logger.info(f"Agents initialized for {self.__class__.__name__} user {self.user_id}")
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
                f"{self.__class__.__name__.lower()}"
            )
            
            # Register with governance system
            await self.governor.register_agent(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id=f"{self.__class__.__name__.lower()}",
                agent_instance=self
            )
            
            logger.info(f"Governance initialized for {self.__class__.__name__} user {self.user_id}")
        except Exception as e:
            logger.error(f"Error initializing governance: {str(e)}")
            raise

    async def _initialize_tables(self):
        """
        Initialize database tables - to be implemented by derived classes.
        Base implementation does nothing.
        """
        pass

    async def initialize_tables_from_definitions(self, defs: Dict[str, str]):
        """
        Initialize tables using a dictionary of table definitions.
        Creates tables if they don't exist.
        
        Args:
            defs: Dictionary mapping table names to CREATE TABLE statements
        """
        async with get_db_connection_context() as conn:
            for name, sql in defs.items():
                try:
                    await conn.execute(sql)
                    logger.info(f"Table {name} ready")
                except Exception as exc:
                    logger.error(f"Init table {name} failed: {exc}")

    # This is an alias for backward compatibility
    async def initialize_tables_for_class(self, defs: Dict[str, str]):
        """Alias for initialize_tables_from_definitions for backward compatibility."""
        await self.initialize_tables_from_definitions(defs)

    async def _initialize_tables_for_class_impl(self, defs: Dict[str, str]):
        """Internal implementation for table initialization."""
        await self.initialize_tables_from_definitions(defs)
    
    # agent‑callable tool, with a distinct name
    @staticmethod
    @function_tool(strict_mode=False)
    async def initialize_tables_tool(ctx: RunContextWrapper, table_definitions: Dict[str, str]):
        """
        Agent-callable tool to initialize database tables.
        
        Args:
            ctx: Run context wrapper containing manager reference
            table_definitions: Dictionary of table definitions
            
        Returns:
            Status dictionary
        """
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
            logger.error(f"DB query failed ({query}): {exc}")
            return []

    # ---------------------------
    # Cache Methods - Local and Global
    # ---------------------------

    def _cache_get(self, key: str):
        """Get an item from the local cache."""
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any):
        """Set an item in the local cache with eviction if needed."""
        self._cache[key] = value
        if len(self._cache) > self._cache_max:
            # naive eviction: pop first key
            self._cache.pop(next(iter(self._cache)))

    def _cache_delete(self, key: str):
        """Delete an item from the local cache."""
        self._cache.pop(key, None)

    def get_cache(self, key: str) -> Any:
        """
        Get an item from the global cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found
        """
        return GLOBAL_LORE_CACHE.get(
            self.cache_namespace,
            key,
            self.user_id,
            self.conversation_id
        )
    
    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the global cache.
        
        Args:
            key: Cache key to set
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
        """
        GLOBAL_LORE_CACHE.set(
            self.cache_namespace,
            key,
            value,
            ttl,
            self.user_id,
            self.conversation_id
        )
    
    def invalidate_cache(self, key: str) -> None:
        """
        Invalidate a specific cache key.
        
        Args:
            key: Cache key to invalidate
        """
        GLOBAL_LORE_CACHE.invalidate(
            self.cache_namespace,
            key,
            self.user_id,
            self.conversation_id
        )
    
    def invalidate_cache_pattern(self, pattern: str) -> None:
        """
        Invalidate cache keys matching a pattern.
        
        Args:
            pattern: Pattern to match cache keys
        """
        GLOBAL_LORE_CACHE.invalidate_pattern(
            self.cache_namespace,
            pattern,
            self.user_id,
            self.conversation_id
        )
    
    def clear_cache(self) -> None:
        """Clear all cache entries for this manager."""
        GLOBAL_LORE_CACHE.clear_namespace(self.cache_namespace)

    # ---------------------------
    # Agent-callable cache tools
    # ---------------------------
    
    @staticmethod
    async def _set_cached_data(ctx: RunContextWrapper, cache_key: str, data: Any) -> None:
        """
        Agent‑friendly cache setter that just forwards to the internal store.
        
        Args:
            ctx: Run context wrapper containing manager reference
            cache_key: Key to store data under
            data: Data to cache
        """
        mgr = _mgr(ctx)
        mgr._cache_set(cache_key, data)

    @staticmethod
    @function_tool
    async def get_cached_data(ctx: RunContextWrapper, cache_key: str):
        """
        Agent-callable tool to get data from cache.
        
        Args:
            ctx: Run context wrapper containing manager reference
            cache_key: Key to retrieve
            
        Returns:
            Cached data or None
        """
        return _mgr(ctx)._cache_get(cache_key)

    @staticmethod
    @function_tool(strict_mode=False)
    async def set_cached_data(ctx: RunContextWrapper, cache_key: str, data: Any):
        """
        Agent-callable tool to set data in cache.
        
        Args:
            ctx: Run context wrapper containing manager reference
            cache_key: Key to store data under
            data: Data to cache
            
        Returns:
            True if successful
        """
        _mgr(ctx)._cache_set(cache_key, data)
        return True

    @staticmethod
    @function_tool
    async def delete_cached_data(ctx: RunContextWrapper, cache_key: str):
        """
        Agent-callable tool to delete data from cache.
        
        Args:
            ctx: Run context wrapper containing manager reference
            cache_key: Key to delete
            
        Returns:
            True if successful
        """
        _mgr(ctx)._cache_delete(cache_key)
        return True

    # ---------------------------
    # Database Operations - Agent Tools
    # ---------------------------

    @staticmethod
    @function_tool
    async def execute_db_query(ctx: RunContextWrapper, query: str, *args):
        """
        Agent-callable tool to execute a database query.
        
        Args:
            ctx: Run context wrapper containing manager reference
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        return await _mgr(ctx)._execute_query_impl(query, *args)

    @staticmethod
    @function_tool(strict_mode=False)
    async def batch_update(ctx: RunContextWrapper, table: str, updates: List[Dict[str, Any]]):
        """
        Agent-callable tool to perform batch updates.
        
        Args:
            ctx: Run context wrapper containing manager reference
            table: Table name to update
            updates: List of update dictionaries
            
        Returns:
            Number of updated rows
        """
        return await _mgr(ctx)._batch_update_impl(table, updates)

    async def _batch_update_impl(self, table: str, updates: List[Dict[str, Any]]) -> int:
        """
        Extremely generic batch updater.
        Each item must have:  {"column": "<col>", "value": <val>, "id": <row_id>}
        Returns the number of rows changed.
        
        Args:
            table: Table name to update
            updates: List of update dictionaries
            
        Returns:
            Number of updated rows
        """
        if not updates:
            return 0
    
        async with self.get_connection_pool() as conn:
            async with conn.transaction():
                for row in updates:
                    col = row["column"]
                    sql = f'UPDATE {table} SET {col} = $1 WHERE id = $2'
                    await conn.execute(sql, row["value"], row["id"])
        return len(updates)

    # ---------------------------
    # Data Validation
    # ---------------------------

    async def _validate_data_impl(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate data against a schema type.
        
        Args:
            data: Data to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Validation results
        """
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
    @function_tool(strict_mode=False)
    async def validate_data(ctx: RunContextWrapper, data: Dict[str, Any], schema_type: str):
        """
        Agent-callable tool to validate data.
        
        Args:
            ctx: Run context wrapper containing manager reference
            data: Data to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Validation results
        """
        return await _mgr(ctx)._validate_data_impl(data, schema_type)

    # ---------------------------
    # Cache Stats & Maintenance
    # ---------------------------
            
    def _get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Cache statistics dictionary
        """
        return {
            "size": len(self._cache),
            "max_size": self._cache_max,
            "utilization": len(self._cache) / self._cache_max if self._cache_max > 0 else 0,
            # You can add proper hit/miss rate if you track it in your cache
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    @function_tool
    async def get_cache_stats(ctx: RunContextWrapper) -> dict:
        """
        Agent-callable tool to get cache statistics.
        
        Args:
            ctx: Run context wrapper containing manager reference
            
        Returns:
            Cache statistics dictionary
        """
        return _mgr(ctx)._get_cache_stats()
    
    async def _maintenance_loop(self):
        """Background task that performs periodic maintenance."""
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            logger.debug(f"Cache utilization {len(self._cache)}/{self._cache_max}")
            try:
                await self._maintenance_once()
            except Exception as exc:
                logger.error(f"Error in maintenance loop: {exc}")
    
    async def _maintenance_once(self):
        """
        Run a single maintenance pass.
        Check cache statistics and take action if needed.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": self  # Include reference to self for agent tools
        })
        stats = self._get_cache_stats()
        with trace(
            "MaintenanceCheck",
            metadata={"component": self.__class__.__name__}
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

    @staticmethod
    @function_tool
    async def maintenance_tool(ctx: RunContextWrapper):
        """
        Agent-accessible: run a single maintenance pass on demand.
        
        Returns:
            Status dictionary
        """
        await self._maintenance_once()
        return {"status": "completed"}

    # ---------------------------
    # Governance & Registration
    # ---------------------------

    async def register_with_governance(
        self, 
        agent_type: AgentType = None, 
        agent_id: str = None, 
        directive_text: str = None, 
        scope: str = "world_building",
        priority: DirectivePriority = DirectivePriority.MEDIUM
    ):
        """
        Register with Nyx governance system with sensible defaults.
        
        Args:
            agent_type: Type of agent to register as
            agent_id: ID of the agent
            directive_text: Text of the directive
            scope: Scope of the directive
            priority: Priority of the directive
        """
        await self.ensure_initialized()
        
        # Default values if not provided
        agent_type = agent_type or AgentType.NARRATIVE_CRAFTER
        agent_id = agent_id or self.__class__.__name__.lower()
        directive_text = directive_text or f"Manage {self.__class__.__name__} for the world setting."
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=agent_type,
            agent_id=agent_id,
            agent_instance=self
        )
        
        # Issue a directive
        await self.governor.issue_directive(
            agent_type=agent_type,
            agent_id=agent_id,
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": directive_text,
                "scope": scope
            },
            priority=priority,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(
            f"{agent_id} registered with governance for "
            f"user {self.user_id}, conversation {self.conversation_id}"
        )

    # ---------------------------
    # Context Utilities
    # ---------------------------

    def create_run_context(self, ctx):
        """
        Create a run context for agent execution.
        
        Args:
            ctx: Context object or dictionary
            
        Returns:
            RunContextWrapper instance
        """
        if isinstance(ctx, RunContextWrapper):
            return ctx
        return RunContextWrapper(context=ctx)

    def as_context(self) -> RunContextWrapper:
        """
        Create a context object embedding this manager.
        
        Returns:
            RunContextWrapper instance with this manager in context
        """
        return RunContextWrapper(context={"manager": self})

    # ---------------------------
    # Embedding & Similarity
    # ---------------------------

    async def generate_and_store_embedding(
        self,
        text: str,
        conn,
        table_name: str,
        id_field: str = "id",
        id_value: Any = None
    ):
        """
        Generate an embedding for text and store it in the database.
        
        Args:
            text: Text to generate embedding for
            conn: Database connection
            table_name: Name of the table
            id_field: Name of the ID field
            id_value: Value of the ID field
        """
        try:
            embedding = await generate_embedding(text)
            await conn.execute(f"""
                UPDATE {table_name}
                SET embedding = $1
                WHERE {id_field} = $2
            """, embedding, id_value)
        except Exception as e:
            logger.error(
                f"Error generating embedding for {table_name}.{id_field}={id_value}: {e}"
            )

    # ---------------------------
    # Standardized CRUD Operations with Governance
    # ---------------------------

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="create")
    @function_tool(strict_mode=False)
    async def create_record(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Create a record with governance oversight.
        
        Args:
            table_name: Name of the table to insert into
            data: Dictionary of column names and values to insert
            
        Returns:
            ID of the created record
        """
        try:
            columns = list(data.keys())
            values = list(data.values())
            placeholders = [f"${i+1}" for i in range(len(values))]
            
            query = f"""
                INSERT INTO {table_name} 
                ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING id
            """
            
            async with get_db_connection_context() as conn:
                record_id = await conn.fetchval(query, *values)
                
                # Generate embedding if text data is provided
                if 'name' in data and 'description' in data and 'embedding' not in data:
                    embedding_text = f"{data['name']} {data['description']}"
                    await self.generate_and_store_embedding(
                        embedding_text,
                        conn,
                        table_name,
                        "id",
                        record_id
                    )
                
                return record_id
        except Exception as e:
            logger.error(f"Error creating record in {table_name}: {e}")
            raise

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="read")
    @function_tool
    async def get_record(self, table_name: str, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a record by ID with governance oversight.
        
        Args:
            table_name: Name of the table to query
            record_id: ID of the record to retrieve
            
        Returns:
            Record as a dictionary or None if not found
        """
        # Check cache first
        cache_key = f"{table_name}_{record_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        try:
            async with get_db_connection_context() as conn:
                record = await conn.fetchrow(f"""
                    SELECT * FROM {table_name}
                    WHERE id = $1
                """, record_id)
                
                if record:
                    result = dict(record)
                    self.set_cache(cache_key, result)
                    return result
                return None
        except Exception as e:
            logger.error(f"Error fetching record {record_id} from {table_name}: {e}")
            return None

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="update")
    @function_tool(strict_mode=False)
    async def update_record(self, table_name: str, record_id: int, data: Dict[str, Any]) -> bool:
        """
        Update a record with governance oversight.
        
        Args:
            table_name: Name of the table to update
            record_id: ID of the record to update
            data: Dictionary of column names and values to update
            
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            if not data:
                return False
                
            columns = list(data.keys())
            values = list(data.values())
            set_clause = ", ".join([f"{col} = ${i+1}" for i, col in enumerate(columns)])
            
            query = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE id = ${len(values) + 1}
            """
            
            async with get_db_connection_context() as conn:
                result = await conn.execute(query, *values, record_id)
                
                # Update embedding if text content changed
                if ('name' in data or 'description' in data):
                    record = await conn.fetchrow(
                        f"SELECT * FROM {table_name} WHERE id = $1",
                        record_id
                    )
                    if record:
                        record_dict = dict(record)
                        embedding_text = f"{record_dict.get('name', '')} {record_dict.get('description', '')}"
                        await self.generate_and_store_embedding(
                            embedding_text,
                            conn,
                            table_name,
                            "id",
                            record_id
                        )
                
                # Invalidate cache
                self.invalidate_cache(f"{table_name}_{record_id}")
                
                return result != "UPDATE 0"
        except Exception as e:
            logger.error(f"Error updating record {record_id} in {table_name}: {e}")
            return False

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="delete")
    @function_tool
    async def delete_record(self, table_name: str, record_id: int) -> bool:
        """
        Delete a record with governance oversight.
        
        Args:
            table_name: Name of the table to delete from
            record_id: ID of the record to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            async with get_db_connection_context() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {table_name}
                    WHERE id = $1
                """, record_id)
                
                self.invalidate_cache(f"{table_name}_{record_id}")
                return result != "DELETE 0"
        except Exception as e:
            logger.error(f"Error deleting record {record_id} from {table_name}: {e}")
            return False

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="query")
    @function_tool(strict_mode=False)
    async def query_records(
        self,
        table_name: str,
        conditions: Dict[str, Any] = None,
        limit: int = 100,
        order_by: str = None
    ) -> List[Dict[str, Any]]:
        """
        Query records with conditions and governance oversight.
        
        Args:
            table_name: Name of the table to query
            conditions: Dictionary of column names and values to match
            limit: Maximum number of records to return
            order_by: Column to order results by
            
        Returns:
            List of records as dictionaries
        """
        try:
            query = f"SELECT * FROM {table_name}"
            values = []
            
            if conditions:
                where_clauses = []
                for i, (col, val) in enumerate(conditions.items()):
                    where_clauses.append(f"{col} = ${i+1}")
                    values.append(val)
                query += f" WHERE {' AND '.join(where_clauses)}"
            
            if order_by:
                query += f" ORDER BY {order_by}"
                
            query += f" LIMIT {limit}"
            
            async with get_db_connection_context() as conn:
                records = await conn.fetch(query, *values)
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error querying records from {table_name}: {e}")
            return []

    @function_tool
    async def search_by_similarity(self, table_name: str, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search records by semantic similarity to the provided text.
        
        Args:
            table_name: Name of the table to search
            text: Query text to find similar content
            limit: Maximum number of results to return
            
        Returns:
            List of records sorted by similarity
        """
        try:
            # Generate embedding for the query text
            embedding = await generate_embedding(text)
            
            async with get_db_connection_context() as conn:
                # Check if the table has an embedding column
                has_embedding = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = '{table_name.lower()}' 
                          AND column_name = 'embedding'
                    );
                """)
                
                if not has_embedding:
                    return []
                
                # Perform similarity search
                records = await conn.fetch(f"""
                    SELECT *, 1 - (embedding <=> $1) as similarity
                    FROM {table_name}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1
                    LIMIT $2
                """, embedding, limit)
                
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error performing similarity search in {table_name}: {e}")
            return []

    # ---------------------------
    # Cleanup
    # ---------------------------

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
        
        # Clear cache
        self._cache.clear()
        
        logger.info(f"{self.__class__.__name__} for user {self.user_id} shut down")
