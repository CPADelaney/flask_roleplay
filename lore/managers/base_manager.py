# lore/core/base_manager.py
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from agents import Agent, Runner, function_tool, AgentHooks, trace
from agents.run_context import RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class BaseLoreManagerHooks(AgentHooks):
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

class BaseLoreManager:
    """Base class for all lore managers using Agent SDK integration."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self.hooks = BaseLoreManagerHooks()
    
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
        # Implementation for database initialization
        pass
        
    def create_run_context(self, ctx) -> RunContextWrapper:
        """Create a run context for agent execution."""
        return RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            **(ctx.context if hasattr(ctx, 'context') else {})
        })
    
    @function_tool
    async def _execute_db_query(self, ctx: RunContextWrapper, query: str, *args) -> List[Dict[str, Any]]:
        """Execute database query as a tool function."""
        try:
            async with await self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    logger.debug(f"Executing query: {query}")
                    if query.strip().upper().startswith('SELECT'):
                        result = await conn.fetch(query, *args)
                        return [dict(row) for row in result]
                    else:
                        return await conn.execute(query, *args)
        except Exception as e:
            logger.error(f"Database error: {e}")
            return []
    
    def default_agent_config(self, name: str, instructions: str, tools=None, handoffs=None):
        """Create a standard agent configuration."""
        return Agent(
            name=name,
            instructions=instructions,
            tools=tools or [function_tool(self._execute_db_query)],
            handoffs=handoffs or [],
            hooks=self.hooks
        )
