# lore/managers/base_manager.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Set, Callable, Protocol, runtime_checkable, Union, Tuple

# Agents SDK imports - v0.0.17 best practices
from agents import (
    Agent, 
    Runner, 
    function_tool, 
    trace, 
    RunContextWrapper, 
    ModelSettings,
    GuardrailFunctionOutput
)
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

# ------------------------------------------------------------------------
# Pydantic Models for Agent SDK Integration
# ------------------------------------------------------------------------

class TableRecord(BaseModel):
    """Model for database records to avoid Dict[str, Any] issues"""
    id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[str] = None
    embedding: Optional[List[float]] = None
    # Store extra fields as a dict instead of using extra='allow'
    extra_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

class CreateRecordInput(BaseModel):
    """Input model for creating records"""
    table_name: str
    name: str
    description: str
    # Store additional fields explicitly
    extra_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UpdateRecordInput(BaseModel):
    """Input model for updating records"""
    table_name: str
    record_id: int
    name: Optional[str] = None
    description: Optional[str] = None
    # Store additional fields explicitly
    extra_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QueryConditions(BaseModel):
    """Model for query conditions"""
    # Use specific fields instead of arbitrary dict
    field_name: str
    field_value: Union[str, int, float, bool, List[Union[str, int, float, bool]]]
    operator: str = "="  # =, >, <, >=, <=, !=, LIKE, IN

class QueryRecordsInput(BaseModel):
    """Input model for querying records"""
    table_name: str
    conditions: Optional[List[QueryConditions]] = None
    limit: int = 100
    order_by: Optional[str] = None

class CacheStats(BaseModel):
    """Cache statistics model"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    utilization: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None

class MaintenanceResult(BaseModel):
    """Result model for maintenance operations"""
    action: str
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ValidationResult(BaseModel):
    """Result model for data validation"""
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    fixed_data: Optional[Dict[str, Any]] = None

class BatchUpdateItem(BaseModel):
    """Model for batch update items"""
    column: str
    value: Union[str, int, float, bool, None]
    id: int

# ------------------------------------------------------------------------
# Table Definition Helper
# ------------------------------------------------------------------------

class TableDefinition(BaseModel):
    """Model for table definitions"""
    name: str
    columns: Dict[str, str]
    indexes: Optional[List[str]] = None
    foreign_keys: Optional[Dict[str, str]] = None

# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------

def _get_manager(ctx: RunContextWrapper) -> 'BaseLoreManager':
    """Helper to retrieve manager instance from context."""
    mgr = ctx.context.get("manager")
    if mgr is None:
        raise RuntimeError("manager instance missing from ctx.context")
    return mgr

# ------------------------------------------------------------------------
# Infrastructure Agent
# ------------------------------------------------------------------------

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
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.0)
)

# ------------------------------------------------------------------------
# BaseLoreManager - Refactored Implementation
# ------------------------------------------------------------------------

class BaseLoreManager:
    """
    Consolidated base class for all lore managers providing common functionality.
    Integrates agent capabilities, database access, caching, and governance oversight.
    
    IMPORTANT: Governance initialization has been removed from ensure_initialized()
    to prevent circular dependencies. Governance should be set externally via
    set_governor() and registered after all components are initialized.
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
        self._local_cache: Dict[str, Any] = {}
        self._cache_max = cache_size
        self._default_ttl = ttl
        self.initialized = False
        self._initializing = False  # Re-entry guard
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
        
        # Cache statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }

    def _setup_tracing(self):
        """Set up naming and metadata for traces from this manager instance."""
        self.trace_name = f"{self.__class__.__name__}Workflow"
        self.trace_group_id = f"conversation_{self.conversation_id}"
        self.trace_metadata = {
            "user_id": str(self.user_id),
            "conversation_id": str(self.conversation_id),
            "manager_type": self.__class__.__name__
        }

    def set_governor(self, governor):
        """
        Set the governor externally to avoid circular dependencies.
        This should be called by the LoreSystem after creating the manager.
        """
        self.governor = governor
        if governor:
            # Initialize directive handler when governor is set
            self.directive_handler = DirectiveHandler(
                self.user_id,
                self.conversation_id,
                self._get_agent_type(),
                self._get_agent_id()
            )

    async def ensure_initialized(self):
        """
        Ensure the manager is initialized.
        Note: Governance registration is now deferred until after initialization.
        """
        if self.initialized:
            return True
            
        # Re-entry guard
        if self._initializing:
            logger.warning(f"Re-entry into {self.__class__.__name__}.ensure_initialized()")
            return True
            
        self._initializing = True
        try:
            await self.initialize_agents()
            await self._initialize_tables()
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            self.initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False
            raise
        finally:
            self._initializing = False

    async def register_with_governance(
        self,
        agent_type: AgentType,
        agent_id: str,
        directive_text: str,
        scope: str = "world_building",
        priority: DirectivePriority = DirectivePriority.MEDIUM
    ) -> bool:
        """
        Register with Nyx governance system.
        
        Args:
            agent_type: Type of agent
            agent_id: Unique ID for this agent
            directive_text: Directive text describing agent's purpose
            scope: Scope of operations
            priority: Priority level
            
        Returns:
            True if registration successful
        """
        if not self.governor:
            logger.warning(f"Cannot register {self.__class__.__name__} - no governor set")
            return False
            
        try:
            # Store the agent type and ID for later use
            self._agent_type = agent_type
            self._agent_id = agent_id
            
            # Register with the governor
            result = await self.governor.register_agent(
                agent_type=agent_type,
                agent_instance=self,
                agent_id=agent_id
            )
            
            if result.get("success", False):
                # Also register the directive
                await self.governor.issue_directive(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    directive_type=DirectiveType.INFORMATION,
                    directive_data={
                        "text": directive_text,
                        "scope": scope
                    },
                    priority=priority,
                    duration_minutes=60 * 24 * 365  # 1 year
                )
                
                logger.info(f"{self.__class__.__name__} registered with governance as {agent_type}/{agent_id}")
                return True
            else:
                logger.error(f"Failed to register {self.__class__.__name__} with governance")
                return False
                
        except Exception as e:
            logger.error(f"Error registering {self.__class__.__name__} with governance: {e}")
            return False

    async def initialize_agents(self):
        """
        Initialize agent definitions. Override in subclasses to define specific agents.
        """
        try:
            # Base agent for general lore management
            self.agents["foundation"] = Agent(
                name="FoundationAgent",
                instructions="You are a general-purpose agent for lore management.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.7)
            )
            
            logger.info(f"Agents initialized for {self.__class__.__name__} user {self.user_id}")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise

    async def _initialize_tables(self):
        """
        Initialize database tables - to be implemented by derived classes.
        """
        pass

    async def initialize_tables_for_class(self, table_definitions: Dict[str, str]):
        """
        Initialize tables using a dictionary of table definitions.
        
        Args:
            table_definitions: Dictionary mapping table names to CREATE TABLE statements
        """
        async with get_db_connection_context() as conn:
            for table_name, create_statement in table_definitions.items():
                try:
                    # Check if table exists
                    table_exists = await conn.fetchval(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table_name.lower()}'
                        );
                    """)
                    
                    if not table_exists:
                        # Create the table
                        await conn.execute(create_statement)
                        logging.info(f"{table_name} table created")
                except Exception as e:
                    logger.error(f"Error creating table {table_name}: {e}")

    async def register_with_governance_deferred(self):
        """
        Register with governance system after initialization is complete.
        This should be called by the LoreSystem after all components are initialized.
        """
        if not self.governor:
            logger.warning(f"Cannot register {self.__class__.__name__} - no governor set")
            return False
            
        try:
            await self.governor.register_agent(
                agent_type=self._get_agent_type(),
                agent_id=self._get_agent_id(),
                agent_instance=self
            )
            logger.info(f"{self.__class__.__name__} registered with governance")
            return True
        except Exception as e:
            logger.error(f"Error registering {self.__class__.__name__} with governance: {e}")
            return False

    def _get_agent_type(self) -> AgentType:
        """
        Get the agent type for this manager.
        Override in subclasses to provide specific agent type.
        """
        return getattr(self, '_agent_type', AgentType.NARRATIVE_CRAFTER)
    
    def _get_agent_id(self) -> str:
        """
        Get the agent ID for this manager.
        Override in subclasses to provide specific agent ID.
        """
        return getattr(self, '_agent_id', self.__class__.__name__.lower())

    # ---------------------------
    # Helper Methods
    # ---------------------------

    def _dict_to_table_record(self, data: Dict[str, Any]) -> TableRecord:
        """
        Convert a dictionary to a TableRecord, handling extra fields.
        
        Args:
            data: Dictionary of record data
            
        Returns:
            TableRecord instance
        """
        # Extract known fields
        known_fields = {'id', 'name', 'description', 'timestamp', 'embedding'}
        record_data = {k: v for k, v in data.items() if k in known_fields}
        
        # Put remaining fields in extra_fields
        extra = {k: v for k, v in data.items() if k not in known_fields and k != 'similarity'}
        if extra:
            record_data['extra_fields'] = extra
            
        return TableRecord(**record_data)

    # ---------------------------
    # Standardized CRUD Operations
    # ---------------------------

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="create")
    async def create_record(self, input_data: CreateRecordInput) -> int:
        """
        Create a record with governance oversight.
        
        Args:
            input_data: CreateRecordInput model with table name and data
            
        Returns:
            ID of the created record
        """
        # Combine standard fields with extra fields
        data = {
            'name': input_data.name,
            'description': input_data.description,
            **input_data.extra_fields
        }
        
        return await self._create_record_internal(input_data.table_name, data)

    async def _create_record_internal(self, table_name: str, data: Dict[str, Any]) -> int:
        """Internal method for record creation with error handling."""
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
            logging.error(f"Error creating record in {table_name}: {e}")
            raise

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="read")
    async def get_record(self, table_name: str, record_id: int) -> Optional[TableRecord]:
        """
        Get a record by ID with governance oversight.
        
        Args:
            table_name: Name of the table to query
            record_id: ID of the record to retrieve
            
        Returns:
            TableRecord model or None if not found
        """
        # Check cache first
        cache_key = f"{table_name}_{record_id}"
        cached = self.get_cache(cache_key)
        if cached:
            self._cache_stats['hits'] += 1
            return self._dict_to_table_record(cached)
        
        self._cache_stats['misses'] += 1
        
        try:
            async with get_db_connection_context() as conn:
                record = await conn.fetchrow(f"""
                    SELECT * FROM {table_name}
                    WHERE id = $1
                """, record_id)
                
                if record:
                    result = dict(record)
                    self.set_cache(cache_key, result)
                    return self._dict_to_table_record(result)
                return None
        except Exception as e:
            logging.error(f"Error fetching record {record_id} from {table_name}: {e}")
            return None

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="update")
    async def update_record(self, input_data: UpdateRecordInput) -> bool:
        """
        Update a record with governance oversight.
        
        Args:
            input_data: UpdateRecordInput model with table name, record id and data
            
        Returns:
            True if update succeeded, False otherwise
        """
        # Build update data
        data = {}
        if input_data.name is not None:
            data['name'] = input_data.name
        if input_data.description is not None:
            data['description'] = input_data.description
        if input_data.extra_fields:
            data.update(input_data.extra_fields)
            
        if not data:
            return False
            
        return await self._update_record_internal(
            input_data.table_name, 
            input_data.record_id, 
            data
        )

    async def _update_record_internal(self, table_name: str, record_id: int, data: Dict[str, Any]) -> bool:
        """Internal method for record updates."""
        try:
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
            logging.error(f"Error updating record {record_id} in {table_name}: {e}")
            return False

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="delete")
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
                self._cache_stats['deletes'] += 1
                return result != "DELETE 0"
        except Exception as e:
            logging.error(f"Error deleting record {record_id} from {table_name}: {e}")
            return False

    @with_governance_permission(agent_type=AgentType.NARRATIVE_CRAFTER, action_type="query")
    async def query_records(self, input_data: QueryRecordsInput) -> List[TableRecord]:
        """
        Query records with conditions and governance oversight.
        
        Args:
            input_data: QueryRecordsInput model with query parameters
            
        Returns:
            List of TableRecord models
        """
        try:
            query = f"SELECT * FROM {input_data.table_name}"
            values = []
            
            if input_data.conditions:
                where_clauses = []
                for i, cond in enumerate(input_data.conditions):
                    if cond.operator.upper() == "IN":
                        # Handle IN operator specially
                        placeholders = [f"${j+len(values)+1}" for j in range(len(cond.field_value))]
                        where_clauses.append(f"{cond.field_name} IN ({', '.join(placeholders)})")
                        values.extend(cond.field_value)
                    else:
                        where_clauses.append(f"{cond.field_name} {cond.operator} ${len(values)+1}")
                        values.append(cond.field_value)
                query += f" WHERE {' AND '.join(where_clauses)}"
            
            if input_data.order_by:
                query += f" ORDER BY {input_data.order_by}"
                
            query += f" LIMIT {input_data.limit}"
            
            async with get_db_connection_context() as conn:
                records = await conn.fetch(query, *values)
                return [self._dict_to_table_record(dict(record)) for record in records]
        except Exception as e:
            logging.error(f"Error querying records from {input_data.table_name}: {e}")
            return []

    async def search_by_similarity(self, table_name: str, text: str, limit: int = 5) -> List[TableRecord]:
        """
        Search records by semantic similarity to the provided text.
        
        Args:
            table_name: Name of the table to search
            text: Query text to find similar content
            limit: Maximum number of results to return
            
        Returns:
            List of TableRecord models sorted by similarity
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
                
                results = []
                for record in records:
                    record_dict = dict(record)
                    # Store similarity score in extra_fields
                    table_record = self._dict_to_table_record(record_dict)
                    if 'similarity' in record_dict:
                        if table_record.extra_fields is None:
                            table_record.extra_fields = {}
                        table_record.extra_fields['similarity'] = record_dict['similarity']
                    results.append(table_record)
                return results
        except Exception as e:
            logging.error(f"Error performing similarity search in {table_name}: {e}")
            return []

    async def generate_and_store_embedding(
        self,
        text: str,
        conn,
        table_name: str,
        id_field: str,
        id_value: Any
    ):
        """
        Generate an embedding for text and store it in the database.
        
        Args:
            text: Text to generate embedding for
            conn: Database connection
            table_name: Name of the table
            id_field: Name of the ID field
            id_value: Value of the ID
        """
        try:
            embedding = await generate_embedding(text)
            await conn.execute(f"""
                UPDATE {table_name}
                SET embedding = $1
                WHERE {id_field} = $2
            """, embedding, id_value)
        except Exception as e:
            logging.error(
                f"Error generating embedding for {table_name}.{id_field}={id_value}: {e}"
            )

    # ---------------------------
    # Agent-Callable Tools
    # ---------------------------

    @staticmethod
    @function_tool(strict_mode=False)
    async def create_record_tool(ctx: RunContextWrapper, input_data: CreateRecordInput) -> int:
        """
        Agent-callable tool to create a record.
        
        Args:
            ctx: Run context wrapper containing manager reference
            input_data: CreateRecordInput model
            
        Returns:
            ID of created record
        """
        mgr = _get_manager(ctx)
        return await mgr.create_record(input_data)

    @staticmethod
    @function_tool
    async def get_record_tool(ctx: RunContextWrapper, table_name: str, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Agent-callable tool to get a record.
        
        Args:
            ctx: Run context wrapper containing manager reference
            table_name: Name of the table
            record_id: ID of the record
            
        Returns:
            Record data or None
        """
        mgr = _get_manager(ctx)
        record = await mgr.get_record(table_name, record_id)
        if record:
            # Flatten the record for agent use
            data = record.model_dump(exclude={'extra_fields'})
            if record.extra_fields:
                data.update(record.extra_fields)
            return data
        return None

    @staticmethod
    @function_tool(strict_mode=False)
    async def update_record_tool(ctx: RunContextWrapper, input_data: UpdateRecordInput) -> bool:
        """
        Agent-callable tool to update a record.
        
        Args:
            ctx: Run context wrapper containing manager reference
            input_data: UpdateRecordInput model
            
        Returns:
            True if successful
        """
        mgr = _get_manager(ctx)
        return await mgr.update_record(input_data)

    @staticmethod
    @function_tool(strict_mode=False)
    async def query_records_tool(ctx: RunContextWrapper, input_data: QueryRecordsInput) -> List[Dict[str, Any]]:
        """
        Agent-callable tool to query records.
        
        Args:
            ctx: Run context wrapper containing manager reference
            input_data: QueryRecordsInput model
            
        Returns:
            List of records
        """
        mgr = _get_manager(ctx)
        records = await mgr.query_records(input_data)
        # Flatten records for agent use
        result = []
        for record in records:
            data = record.model_dump(exclude={'extra_fields'})
            if record.extra_fields:
                data.update(record.extra_fields)
            result.append(data)
        return result

    @staticmethod
    @function_tool
    async def search_similarity_tool(ctx: RunContextWrapper, table_name: str, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Agent-callable tool to search by similarity.
        
        Args:
            ctx: Run context wrapper containing manager reference
            table_name: Name of the table
            text: Query text
            limit: Maximum results
            
        Returns:
            List of similar records
        """
        mgr = _get_manager(ctx)
        records = await mgr.search_by_similarity(table_name, text, limit)
        # Flatten records for agent use
        result = []
        for record in records:
            data = record.model_dump(exclude={'extra_fields'})
            if record.extra_fields:
                data.update(record.extra_fields)
            result.append(data)
        return result

    @staticmethod
    @function_tool
    async def get_cache_stats_tool(ctx: RunContextWrapper) -> CacheStats:
        """
        Agent-callable tool to get cache statistics.
        
        Args:
            ctx: Run context wrapper containing manager reference
            
        Returns:
            CacheStats model
        """
        mgr = _get_manager(ctx)
        stats = mgr._get_cache_stats_dict()
        return CacheStats(**stats)

    @staticmethod
    @function_tool(strict_mode=False)
    async def execute_raw_query_tool(ctx: RunContextWrapper, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Agent-callable tool to execute raw SQL queries.
        Use with caution - only for advanced operations.
        
        Args:
            ctx: Run context wrapper containing manager reference
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        mgr = _get_manager(ctx)
        return await mgr._execute_query_internal(query, *(params or []))

    @staticmethod
    @function_tool(strict_mode=False)
    async def batch_update_tool(ctx: RunContextWrapper, table_name: str, updates: List[BatchUpdateItem]) -> int:
        """
        Agent-callable tool to perform batch updates.
        
        Args:
            ctx: Run context wrapper containing manager reference
            table_name: Name of the table
            updates: List of BatchUpdateItem models
            
        Returns:
            Number of updated rows
        """
        mgr = _get_manager(ctx)
        update_dicts = [u.model_dump() for u in updates]
        return await mgr._batch_update_internal(table_name, update_dicts)

    @staticmethod
    @function_tool(strict_mode=False)
    async def validate_data_tool(ctx: RunContextWrapper, data: Dict[str, Any], schema_type: str) -> ValidationResult:
        """
        Agent-callable tool to validate data.
        
        Args:
            ctx: Run context wrapper containing manager reference
            data: Data to validate
            schema_type: Type of schema
            
        Returns:
            ValidationResult model
        """
        mgr = _get_manager(ctx)
        result = await mgr._validate_data_internal(data, schema_type)
        return ValidationResult(**result)

    # ---------------------------
    # Caching Methods
    # ---------------------------

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
            ttl or self._default_ttl,
            self.user_id,
            self.conversation_id
        )
        self._cache_stats['sets'] += 1
    
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
        self._local_cache.clear()

    # ---------------------------
    # Internal Methods
    # ---------------------------

    async def _execute_query_internal(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute a database query and return results.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            async with get_db_connection_context() as conn:
                records = await conn.fetch(query, *args)
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []

    async def _batch_update_internal(self, table: str, updates: List[Dict[str, Any]]) -> int:
        """
        Perform batch updates on a table.
        
        Args:
            table: Table name
            updates: List of update dictionaries with 'column', 'value', 'id'
            
        Returns:
            Number of updated rows
        """
        if not updates:
            return 0
        
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    for update in updates:
                        await conn.execute(
                            f"UPDATE {table} SET {update['column']} = $1 WHERE id = $2",
                            update['value'],
                            update['id']
                        )
            return len(updates)
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
            return 0

    async def _validate_data_internal(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate data against a schema type.
        
        Args:
            data: Data to validate
            schema_type: Type of schema
            
        Returns:
            Validation results
        """
        # Example validation - override in subclasses for specific schemas
        required_fields = {
            "foundation": ["cosmology", "magic_system"],
            "faction": ["name", "type"],
            "character": ["name", "role"]
        }.get(schema_type, [])
        
        missing = [field for field in required_fields if field not in data]
        
        return {
            "is_valid": len(missing) == 0,
            "issues": missing,
            "fixed_data": data
        }

    def _get_cache_stats_dict(self) -> Dict[str, Any]:
        """
        Get cache statistics as a dictionary.
        
        Returns:
            Cache statistics
        """
        return {
            **self._cache_stats,
            "size": len(self._local_cache),
            "max_size": self._cache_max,
            "utilization": len(self._local_cache) / self._cache_max if self._cache_max > 0 else 0,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        }

    # ---------------------------
    # Maintenance Loop
    # ---------------------------

    async def _maintenance_loop(self):
        """Background task that performs periodic maintenance."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._maintenance_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")

    async def _maintenance_once(self):
        """
        Run a single maintenance pass.
        """
        stats = self._get_cache_stats_dict()
        
        with trace(
            workflow_name="MaintenanceCheck",
            group_id=self.trace_group_id,
            metadata=self.trace_metadata
        ):
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "manager": self
            })
            
            prompt = (
                f"Cache statistics:\n{json.dumps(stats, indent=2)}\n\n"
                "Analyze and decide if action is needed. Return JSON response."
            )
            
            run_config = RunConfig(
                workflow_name="MaintenanceAgent",
                trace_metadata=self.trace_metadata
            )
            
            result = await Runner.run(
                starting_agent=maintenance_agent,
                input=prompt,
                context=run_ctx.context,
                run_config=run_config
            )
            
            try:
                decision = json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
                
                if decision.get("action") == "log_warning":
                    logger.warning(decision.get("message", "Maintenance warning"))
                elif decision.get("action") == "clear_cache":
                    logger.info("Clearing cache based on maintenance recommendation")
                    self._local_cache.clear()
                    self._cache_stats['evictions'] += len(self._local_cache)
            except Exception as e:
                logger.error(f"Error processing maintenance decision: {e}")

    # ---------------------------
    # Context and Agent Utilities
    # ---------------------------

    def create_run_context(self, additional_context: Optional[Dict[str, Any]] = None) -> RunContextWrapper:
        """
        Create a run context for agent execution.
        
        Args:
            additional_context: Additional context to include
            
        Returns:
            RunContextWrapper instance
        """
        context = {
            "manager": self,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        if additional_context:
            context.update(additional_context)
        return RunContextWrapper(context=context)

    async def execute_llm_prompt(
        self, 
        prompt: str, 
        agent_name: Optional[str] = None, 
        model: str = "gpt-4.1-nano",
        temperature: float = 0.7
    ) -> str:
        """
        Execute a prompt with an LLM agent.
        
        Args:
            prompt: Prompt text
            agent_name: Optional agent name
            model: Model to use
            temperature: Temperature setting
            
        Returns:
            Response text
        """
        with trace(
            workflow_name=self.trace_name,
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "prompt_type": "lore_generation",
                "agent_name": agent_name or f"{self.__class__.__name__}Agent",
                "model": model
            }
        ):
            # Use existing agent if available, otherwise create temporary one
            if agent_name and agent_name in self.agents:
                agent = self.agents[agent_name]
            else:
                agent = Agent(
                    name=agent_name or f"{self.__class__.__name__}Agent",
                    instructions=f"You help with {self.__class__.__name__} management.",
                    model=model,
                    model_settings=ModelSettings(temperature=temperature)
                )
            
            run_config = RunConfig(
                workflow_name=f"{self.__class__.__name__}Prompt",
                trace_metadata=self.trace_metadata
            )
            
            result = await Runner.run(
                starting_agent=agent,
                input=prompt,
                context=self.create_run_context().context,
                run_config=run_config
            )
            
            return result.final_output

    async def create_table_definition(
        self,
        table_name: str,
        extra_columns: Optional[Dict[str, str]] = None,
        include_standard: bool = True,
        foreign_keys: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a standardized table definition.
        
        Args:
            table_name: Name of the table
            extra_columns: Additional columns beyond standard
            include_standard: Whether to include standard columns
            foreign_keys: Foreign key constraints
            
        Returns:
            SQL CREATE TABLE statement
        """
        all_columns = {}
        
        if include_standard:
            all_columns.update(self._standard_columns)
        
        if extra_columns:
            all_columns.update(extra_columns)
        
        column_defs = [f"{col} {definition}" for col, definition in all_columns.items()]
        
        if foreign_keys:
            for column, reference in foreign_keys.items():
                column_defs.append(
                    f"FOREIGN KEY ({column}) REFERENCES {reference} ON DELETE CASCADE"
                )
        
        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_defs)}
            );
        """
        
        # Add index for embeddings if included
        if 'embedding' in all_columns:
            sql += f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_embedding
            ON {table_name} USING ivfflat (embedding vector_cosine_ops);
            """
        
        return sql

    # ---------------------------
    # Cleanup
    # ---------------------------

    async def close(self):
        """
        Close the manager and clean up resources.
        """
        # Cancel maintenance task
        if self.maintenance_task and not self.maintenance_task.done():
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Clear caches
        self._local_cache.clear()
        self.clear_cache()
        
        logger.info(f"{self.__class__.__name__} for user {self.user_id} closed")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'maintenance_task') and self.maintenance_task:
            self.maintenance_task.cancel()
