# lore/core/base_manager.py
"""
Base Manager Module with OpenAI Agent SDK Integration

This module provides the foundational agent-based architecture for lore management.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Type, Set, Callable, TypeVar, Union
from datetime import datetime

from agents import Agent, Runner, function_tool, AgentHooks, trace, Trace
from agents.run_context import RunContextWrapper
from agents.tracing import custom_span, function_span, agent_span
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

class LoreManagerHooks(AgentHooks):
    """Lifecycle hooks for lore manager agents."""
    
    async def on_start(self, context, agent):
        """Called when the agent starts execution."""
        logger.info(f"Agent {agent.name} starting execution")
    
    async def on_end(self, context, agent, output):
        """Called when the agent produces a final output."""
        logger.info(f"Agent {agent.name} completed execution")
    
    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked."""
        logger.info(f"Agent {agent.name} using tool {tool.name}")
    
    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool is invoked."""
        logger.info(f"Agent {agent.name} completed tool {tool.name}")
    
    async def on_handoff(self, context, agent, source):
        """Called when the agent is being handed off to."""
        logger.info(f"Handoff from {source.name} to {agent.name}")


class BaseLoreManager:
    """
    Base class for all lore managers with integrated OpenAI Agent SDK.
    Provides common functionality for database access, agent creation, and more.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self.hooks = LoreManagerHooks()
        self._pool = None
        self._default_agents = {}
        
    async def ensure_initialized(self):
        """Ensure system is initialized."""
        if not self.initialized:
            await self._initialize_tables()
            self.initialized = True
    
    async def _initialize_tables(self):
        """Initialize necessary database tables."""
        raise NotImplementedError("Subclasses must implement _initialize_tables")
    
    async def initialize_tables_for_class(self, table_definitions: Dict[str, str]):
        """Initialize tables from definitions dictionary."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for table_name, definition in table_definitions.items():
                    try:
                        # Check if table exists
                        exists = await conn.fetchval(
                            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                            table_name.lower()
                        )
                        
                        if not exists:
                            await conn.execute(definition)
                            logger.info(f"Created table {table_name}")
                        else:
                            logger.info(f"Table {table_name} already exists")
                    except Exception as e:
                        logger.error(f"Error creating table {table_name}: {e}")
    
    async def get_connection_pool(self):
        """Get database connection pool."""
        if self._pool is None:
            # This is a placeholder - in a real implementation, you would
            # use the actual connection details from your configuration
            import asyncpg
            self._pool = await asyncpg.create_pool(
                database="your_database",
                user="your_user",
                password="your_password",
                host="localhost"
            )
        return self._pool
    
    def create_run_context(self, ctx=None) -> RunContextWrapper:
        """Create a run context for agent execution."""
        context = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
        }
        
        # Add additional context if provided
        if ctx is not None:
            if hasattr(ctx, 'context') and ctx.context is not None:
                context.update(ctx.context)
            elif isinstance(ctx, dict):
                context.update(ctx)
        
        return RunContextWrapper(context=context)
    
    @function_tool
    async def _execute_db_query(self, ctx: RunContextWrapper, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute database query as a tool function.
        
        Args:
            ctx: Run context
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            List of dictionaries representing the query results
        """
        try:
            with function_span(name="database_query", input=query):
                async with await self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        logger.debug(f"Executing query: {query}")
                        if query.strip().upper().startswith('SELECT'):
                            result = await conn.fetch(query, *args)
                            return [dict(row) for row in result]
                        else:
                            result = await conn.execute(query, *args)
                            if isinstance(result, str):
                                return [{"result": result}]
                            return [{"affected_rows": result}]
        except Exception as e:
            logger.error(f"Database error: {e}")
            return [{"error": str(e)}]
    
    @function_tool
    async def generate_and_store_embedding(
        self, 
        ctx: RunContextWrapper, 
        text: str, 
        conn, 
        table_name: str, 
        id_column: str, 
        id_value: Any
    ) -> bool:
        """
        Generate an embedding for text and store it in the database.
        
        Args:
            ctx: Run context
            text: Text to generate embedding for
            conn: Database connection
            table_name: Table to update
            id_column: ID column name
            id_value: ID value
            
        Returns:
            Success status
        """
        try:
            from embedding.vector_store import generate_embedding
            
            with function_span(name="generate_embedding"):
                embedding = await generate_embedding(text)
                
                # Update the record with the embedding
                await conn.execute(
                    f"UPDATE {table_name} SET embedding = $1 WHERE {id_column} = $2",
                    embedding, id_value
                )
                
                return True
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return False
    
    def default_agent(self, name: str, instructions: str, tools=None, handoffs=None, output_type=None):
        """
        Create or retrieve a standard agent configuration.
        
        Args:
            name: Agent name
            instructions: Agent instructions
            tools: List of tools for the agent
            handoffs: List of handoffs for the agent
            output_type: Expected output type
            
        Returns:
            Configured Agent
        """
        # Check if we already have this agent cached
        if name in self._default_agents:
            return self._default_agents[name]
        
        # Set up default tools if none provided
        if tools is None:
            tools = [function_tool(self._execute_db_query)]
        
        # Create the agent
        agent = Agent(
            name=name,
            instructions=instructions,
            tools=tools,
            handoffs=handoffs or [],
            output_type=output_type,
            hooks=self.hooks
        )
        
        # Cache for future use
        self._default_agents[name] = agent
        return agent
    
    async def register_with_governance(
        self, 
        agent_type: Any, 
        agent_id: str, 
        directive_text: str, 
        scope: str, 
        priority: Any
    ):
        """
        Register with Nyx governance system.
        This is a placeholder for the original governance system.
        
        In a production system, this would be replaced with Agent SDK's
        built-in governance mechanisms.
        """
        logger.info(f"Registered {agent_id} with governance system")
        pass
    
    async def _validate_data(self, data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Validated data
        """
        try:
            # In a real implementation, you would use Pydantic models
            # Placeholder for now
            return data
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise
    
    @function_tool
    async def get_cache(self, ctx: RunContextWrapper, key: str) -> Any:
        """Get data from cache."""
        # Placeholder for cache implementation
        return None
    
    @function_tool
    async def set_cache(self, ctx: RunContextWrapper, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set data in cache."""
        # Placeholder for cache implementation
        return True
    
    @function_tool
    async def invalidate_cache(self, ctx: RunContextWrapper, key: str) -> bool:
        """Invalidate cache entry."""
        # Placeholder for cache implementation
        return True
    
    @function_tool
    async def invalidate_cache_pattern(self, ctx: RunContextWrapper, pattern: str) -> bool:
        """Invalidate cache entries matching pattern."""
        # Placeholder for cache implementation
        return True
