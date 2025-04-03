# lore/core/base_manager.py

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import asyncio
from datetime import datetime

from agents import Agent, ModelSettings, Runner
from agents.run_context import RunContextWrapper

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity

from lore.core.cache import GLOBAL_LORE_CACHE

class BaseLoreManager:
    """
    Enhanced base class for all lore management systems.
    Centralized database operations, authorization and table management.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the base lore manager.
        
        Args:
            user_id: ID of the user
            conversation_id: ID of the conversation
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.initialized = False
        self.cache_namespace = self.__class__.__name__.lower()
        self.db_pool = None
        
        # Define standard table columns for common operations
        self._standard_columns = {
            'id': 'SERIAL PRIMARY KEY',
            'name': 'TEXT NOT NULL',
            'description': 'TEXT NOT NULL',
            'timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'embedding': 'VECTOR(1536)'
        }
    
    async def ensure_initialized(self):
        """
        Ensure governance is initialized and any necessary tables exist.
        """
        if not self.initialized:
            await self._initialize_governance()
            await self._initialize_tables()
            await self._initialize_db_pool()
            self.initialized = True
    
    async def _initialize_governance(self):
        """Initialize Nyx governance connection."""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def _initialize_db_pool(self):
        """Initialize database connection pool."""
        if not self.db_pool:
            self.db_pool = await get_db_connection(self.user_id, self.conversation_id)
        return self.db_pool
    
    async def _initialize_tables(self):
        """Initialize database tables - to be implemented by derived classes."""
        pass
    
    async def initialize_tables_for_class(self, table_definitions: Dict[str, str]):
        """
        Initialize tables using a dictionary of table definitions.
        
        Args:
            table_definitions: Dictionary mapping table names to CREATE TABLE statements
        """
        db_pool = await self._initialize_db_pool()
        async with db_pool.acquire() as conn:
            for table_name, create_statement in table_definitions.items():
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

    # [Rest of BaseLoreManager methods...]
    # Enhanced governance registration with sensible defaults
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
        
        logging.info(f"{agent_id} registered with governance for user {self.user_id}, conversation {self.conversation_id}")
    
    # Standardized CRUD operations with error handling and permission checking
    
    @with_governance_permission
    async def create_record(self, table_name: str, data: Dict[str, Any]) -> int:
        """Create a record with governance oversight."""
        return await self._create_record_internal(table_name, data)
    
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
            
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                record_id = await conn.fetchval(query, *values)
                
                # Generate embedding if text data is provided
                if 'name' in data and 'description' in data and 'embedding' not in data:
                    embedding_text = f"{data['name']} {data['description']}"
                    await self.generate_and_store_embedding(embedding_text, conn, table_name, "id", record_id)
                
                return record_id
        except Exception as e:
            logging.error(f"Error creating record in {table_name}: {e}")
            raise
    
    @with_governance_permission
    async def get_record(self, table_name: str, record_id: int) -> Optional[Dict[str, Any]]:
        """Get a record by ID with governance oversight."""
        # Check cache first
        cache_key = f"{table_name}_{record_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
            
        try:
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                record = await conn.fetchrow(f"""
                    SELECT * FROM {table_name}
                    WHERE id = $1
                """, record_id)
                
                if record:
                    result = dict(record)
                    # Cache the result
                    self.set_cache(cache_key, result)
                    return result
                return None
        except Exception as e:
            logging.error(f"Error fetching record {record_id} from {table_name}: {e}")
            return None
    
    @with_governance_permission
    async def update_record(self, table_name: str, record_id: int, data: Dict[str, Any]) -> bool:
        """Update a record with governance oversight."""
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
            
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                result = await conn.execute(query, *values, record_id)
                
                # Update embedding if text content changed
                if ('name' in data or 'description' in data):
                    # Get full record data
                    record = await conn.fetchrow(f"SELECT * FROM {table_name} WHERE id = $1", record_id)
                    if record:
                        record_dict = dict(record)
                        embedding_text = f"{record_dict.get('name', '')} {record_dict.get('description', '')}"
                        await self.generate_and_store_embedding(embedding_text, conn, table_name, "id", record_id)
                
                # Invalidate cache
                self.invalidate_cache(f"{table_name}_{record_id}")
                
                return result != "UPDATE 0"
        except Exception as e:
            logging.error(f"Error updating record {record_id} in {table_name}: {e}")
            return False
    
    @with_governance_permission
    async def delete_record(self, table_name: str, record_id: int) -> bool:
        """Delete a record with governance oversight."""
        try:
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {table_name}
                    WHERE id = $1
                """, record_id)
                
                # Invalidate cache
                self.invalidate_cache(f"{table_name}_{record_id}")
                
                return result != "DELETE 0"
        except Exception as e:
            logging.error(f"Error deleting record {record_id} from {table_name}: {e}")
            return False
    
    @with_governance_permission
    async def query_records(self, table_name: str, conditions: Dict[str, Any] = None, 
                           limit: int = 100, order_by: str = None) -> List[Dict[str, Any]]:
        """Query records with conditions and governance oversight."""
        try:
            # Build query
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
            
            # Execute query
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                records = await conn.fetch(query, *values)
                return [dict(record) for record in records]
        except Exception as e:
            logging.error(f"Error querying records from {table_name}: {e}")
            return []
    
    async def search_by_similarity(self, table_name: str, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search records by semantic similarity to the provided text."""
        try:
            # Generate embedding for the query text
            embedding = await generate_embedding(text)
            
            db_pool = await self._initialize_db_pool()
            async with db_pool.acquire() as conn:
                # Check if the table has an embedding column
                has_embedding = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = '{table_name.lower()}' AND column_name = 'embedding'
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
            # Generate embedding
            embedding = await generate_embedding(text)
            
            # Update the table
            await conn.execute(f"""
                UPDATE {table_name}
                SET embedding = $1
                WHERE {id_field} = $2
            """, embedding, id_value)
        except Exception as e:
            logging.error(f"Error generating embedding for {table_name}.{id_field}={id_value}: {e}")
    
    # Standardized caching methods
    
    def get_cache(self, key: str) -> Any:
        """Get an item from the cache."""
        return GLOBAL_LORE_CACHE.get(self.cache_namespace, key, self.user_id, self.conversation_id)
    
    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set an item in the cache."""
        GLOBAL_LORE_CACHE.set(self.cache_namespace, key, value, ttl, self.user_id, self.conversation_id)
    
    def invalidate_cache(self, key: str) -> None:
        """Invalidate a specific cache key."""
        GLOBAL_LORE_CACHE.invalidate(self.cache_namespace, key, self.user_id, self.conversation_id)
    
    def invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache keys matching a pattern."""
        GLOBAL_LORE_CACHE.invalidate_pattern(self.cache_namespace, pattern, self.user_id, self.conversation_id)
    
    def clear_cache(self) -> None:
        """Clear all cache entries for this manager."""
        GLOBAL_LORE_CACHE.clear_namespace(self.cache_namespace)
    
    # Utility methods
    
    def create_run_context(self, ctx):
        """Create a run context from context."""
        if hasattr(ctx, 'context'):
            return ctx
        return RunContextWrapper(context=ctx)
    
    async def execute_llm_prompt(self, prompt: str, agent_name: str = None, model: str = "o3-mini") -> str:
        """Execute a prompt with an LLM agent."""
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Create agent
        agent = Agent(
            name=agent_name or f"{self.__class__.__name__}Agent",
            instructions=f"You help with generating content for {self.__class__.__name__}.",
            model=model
        )
        
        # Run prompt
        result = await Runner.run(agent, prompt, context=run_ctx.context)
        return result.final_output
    
    async def create_table_definition(self, table_name: str, extra_columns: Dict[str, str] = None,
                                     include_standard: bool = True, foreign_keys: Dict[str, str] = None) -> str:
        """
        Create a standardized table definition.
        
        Args:
            table_name: Name of the table
            extra_columns: Additional columns beyond the standard ones
            include_standard: Whether to include standard columns
            foreign_keys: Dictionary mapping column names to referenced tables/columns
            
        Returns:
            SQL CREATE TABLE statement
        """
        all_columns = {}
        
        # Add standard columns if requested
        if include_standard:
            all_columns.update(self._standard_columns)
        
        # Add extra columns
        if extra_columns:
            all_columns.update(extra_columns)
        
        # Build column definitions
        column_defs = [f"{col} {definition}" for col, definition in all_columns.items()]
        
        # Add foreign keys
        if foreign_keys:
            for column, reference in foreign_keys.items():
                column_defs.append(f"FOREIGN KEY ({column}) REFERENCES {reference} ON DELETE CASCADE")
        
        # Create the SQL statement
        sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(column_defs)}
            );
            
            CREATE INDEX IF NOT EXISTS idx_{table_name.lower()}_embedding 
            ON {table_name} USING ivfflat (embedding vector_cosine_ops);
        """
        
        return sql
