# lore/manager/world_lore_manager.py

"""
World Lore Manager with Resource Management (Agent-ified)

This module provides world lore management with integrated resource management,
allowing each core operation to be called as an agent function tool if desired.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from pydantic import BaseModel, Field

import asyncpg

# Agents SDK (import what you need)
from agents import Agent, function_tool, Runner, trace, ModelSettings, handoff
# Or: from agents import Agent, function_tool, Runner, ...
# (depending on your usage patterns)

# Because we might want to store or retrieve data from an LLM orchestrator
# we define a ResourceOpsAgent for demonstration:
# (You can remove it or rename it if you don't plan on LLM usage here.)
RESOURCE_OPS_AGENT = Agent(
    name="ResourceOpsAgent",
    instructions=(
        "You manage resources and data caching for world lore. "
        "Your tool methods allow for retrieving, setting, invalidating data in a cache, "
        "and checking resource usage. Keep everything thread-safe and consistent."
    ),
    model="o3-mini"
    # model_settings, etc., if needed
)

logger = logging.getLogger(__name__)

from lore.managers.base_manager import BaseManager
from lore.resource_manager import resource_manager

class WorldLoreManager(BaseManager):
    """
    Manager for world lore with resource management support.
    Now includes function_tool decorators so you can orchestrate
    these methods from an agent if desired.
    """
    
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
        
        # Optional: store your agent if you want direct usage
        self.resource_ops_agent = RESOURCE_OPS_AGENT

        self.inconsistency_resolution_agent = Agent(
            name="InconsistencyResolutionAgent",
            instructions="Analyze and resolve any inconsistencies in world lore elements.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7),
            output_type=InconsistencyResolutionAgent
        )

        self.world_documentation_agent = Agent(
            name="WorldDocumentationAgent",
            instructions="Generate readable summaries of world history and current state.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7),
            output_type=WorldDocumentationAgent
        )    

        self.world_query_agent = Agent(
            name="WorldQueryAgent",
            instructions="Process queries related to the world state and provide relevant information.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7),
            output_type=WorldQueryAgent
        )

    @function_tool
    async def query_world_state(self, query: str) -> str:
        """Handle a natural language query about the world state."""
        query_agent = self.world_query_agent.clone()
        query_agent.query = query
        result = await Runner.run(query_agent, query)
        return result.final_output    

    @function_tool
    async def resolve_world_inconsistencies(self, world_id: str) -> str:
        """Identify and resolve any inconsistencies in the world lore."""
        resolution_agent = self.inconsistency_resolution_agent.clone()
        resolution_agent.world_id = world_id
        result = await Runner.run(resolution_agent, f"Resolve inconsistencies for world {world_id}")
        return result.final_output

    @function_tool
    async def generate_world_summary(self, world_id: str, include_history: bool = True, include_current_state: bool = True) -> str:
        """Generate world documentation for history and current state."""
        doc_agent = self.world_documentation_agent.clone()
        doc_agent.world_id = world_id
        doc_agent.include_history = include_history
        doc_agent.include_current_state = include_current_state
        result = await Runner.run(doc_agent, f"Generate summary for world {world_id}")
        return result.final_output

    @function_tool
    async def start(self):
        """Start the world lore manager and resource management."""
        await super().start()
        await self.resource_manager.start()

    @function_tool
    async def stop(self):
        """Stop the world lore manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()

    @function_tool
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

    @function_tool
    async def set_world_data(
        self,
        world_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world data in cache."""
        try:
            await self.resource_manager._check_resource_availability('memory')
            return await self.set_cached_data('world', world_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting world data: {e}")
            return False

    @function_tool
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

    @function_tool
    async def get_world_history(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get world history from cache or fetch if not available."""
        try:
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_history', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world history: {e}")
            return None

    @function_tool
    async def set_world_history(
        self,
        world_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world history in cache."""
        try:
            await self.resource_manager._check_resource_availability('memory')
            return await self.set_cached_data('world_history', world_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting world history: {e}")
            return False

    @function_tool
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

    @function_tool
    async def get_world_events(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get world events from cache or fetch if not available."""
        try:
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_events', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world events: {e}")
            return None

    @function_tool
    async def set_world_events(
        self,
        world_id: str,
        events: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world events in cache."""
        try:
            await self.resource_manager._check_resource_availability('memory')
            return await self.set_cached_data('world_events', world_id, events, tags)
        except Exception as e:
            logger.error(f"Error setting world events: {e}")
            return False

    @function_tool
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

    @function_tool
    async def get_world_relationships(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get world relationships from cache or fetch if not available."""
        try:
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_relationships', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world relationships: {e}")
            return None

    @function_tool
    async def set_world_relationships(
        self,
        world_id: str,
        relationships: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world relationships in cache."""
        try:
            await self.resource_manager._check_resource_availability('memory')
            return await self.set_cached_data('world_relationships', world_id, relationships, tags)
        except Exception as e:
            logger.error(f"Error setting world relationships: {e}")
            return False

    @function_tool
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

    @function_tool
    async def get_world_metadata(
        self,
        world_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get world metadata from cache or fetch if not available."""
        try:
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('world_metadata', world_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting world metadata: {e}")
            return None

    @function_tool
    async def set_world_metadata(
        self,
        world_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world metadata in cache."""
        try:
            await self.resource_manager._check_resource_availability('memory')
            return await self.set_cached_data('world_metadata', world_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting world metadata: {e}")
            return False

    @function_tool
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

    @function_tool
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}

    @function_tool
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")

    @function_tool
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    @function_tool
    async def get_world_lore(self, world_id: int) -> Dict[str, Any]:
        """
        Retrieve comprehensive world lore including cultures, religions, and history.
        This example uses `_execute_db_query` from the parent class for DB calls.
        """
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

    @function_tool
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

    @function_tool
    async def get_cultural_context(self, culture_id: int) -> Dict[str, Any]:
        """Get detailed cultural context including traditions, customs, and beliefs."""
        return await self._execute_db_query(
            "SELECT * FROM cultural_details WHERE culture_id = $1",
            culture_id
        )

    @function_tool
    async def get_religious_context(self, religion_id: int) -> Dict[str, Any]:
        """Get detailed religious context including beliefs, practices, and hierarchy."""
        return await self._execute_db_query(
            "SELECT * FROM religious_details WHERE religion_id = $1",
            religion_id
        )

    @function_tool
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


    # Inherited from base: _validate_data, _execute_db_query, get_connection_pool, etc.
    # We can override or extend them if needed, or rely on the base implementation.

# Create a singleton instance if desired
world_lore_manager = WorldLoreManager(user_id=0, conversation_id=0)

class MasterCoordinationAgent:
    """
    Master agent for coordinating lore subsystems, ensuring consistency and coherence.
    """
    
    def __init__(self, world_lore_manager):
        self.world_lore_manager = world_lore_manager
        self.agent = Agent(
            name="LoreMasterAgent",
            instructions="""
            You are the master coordinator for a fantasy world lore system.
            Your responsibilities:
            1. Ensure narrative consistency across subsystems
            2. Manage dependencies between world elements
            3. Prioritize and schedule generation tasks
            4. Resolve conflicts between generated content
            5. Maintain matriarchal theming throughout
            """,
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7)
        )
        self.trace_id = None
    
    async def initialize(self, user_id: int, conversation_id: int):
        """Initialize the master coordination agent."""
        self.trace_id = f"master_coord_{user_id}_{conversation_id}"
        with trace("MasterCoordinationInit", group_id=self.trace_id):
            # Load existing world state
            world_data = await self.world_lore_manager.get_world_data("main")
            # Initialize coordination memory
            self.coordination_memory = {
                "subsystems": {
                    "politics": {"status": "ready", "last_update": datetime.now().isoformat()},
                    "religion": {"status": "ready", "last_update": datetime.now().isoformat()},
                    "culture": {"status": "ready", "last_update": datetime.now().isoformat()},
                    "dynamics": {"status": "ready", "last_update": datetime.now().isoformat()}
                },
                "pending_tasks": [],
                "dependency_graph": {},
                "consistency_issues": []
            }
            return {"status": "initialized", "world_data": world_data is not None}
    
    async def coordinate_task(self, task_description: str, subsystems: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a task across multiple subsystems."""
        with trace("TaskCoordination", group_id=self.trace_id, metadata={"task": task_description, "subsystems": subsystems}):
            run_ctx = context or {}
            
            # Ask the agent how to coordinate this task
            prompt = f"""
            I need to coordinate this task across multiple subsystems:
            
            TASK: {task_description}
            
            SUBSYSTEMS: {subsystems}
            
            CURRENT WORLD STATE:
            {json.dumps(await self.world_lore_manager.get_world_metadata("main"), indent=2)}
            
            For each subsystem, determine:
            1. What action it should take
            2. In what order the subsystems should execute
            3. How data should flow between them
            4. How to ensure consistency
            
            Return a detailed JSON execution plan.
            """
            
            result = await Runner.run(self.agent, prompt, context=run_ctx)
            
            try:
                execution_plan = json.loads(result.final_output)
                # Update the coordination memory
                self.coordination_memory["pending_tasks"].append({
                    "task": task_description,
                    "plan": execution_plan,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                })
                return execution_plan
            except json.JSONDecodeError:
                return {"error": "Failed to parse execution plan", "raw_output": result.final_output}
    
    async def validate_consistency(self, content: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Validate the consistency of newly generated content."""
        with trace("ConsistencyValidation", group_id=self.trace_id, metadata={"content_type": content_type}):
            # Get relevant existing content for comparison
            existing_data = await self._get_related_content(content, content_type)
            
            prompt = f"""
            Validate the consistency of this newly generated {content_type}:
            
            NEW CONTENT:
            {json.dumps(content, indent=2)}
            
            EXISTING RELATED CONTENT:
            {json.dumps(existing_data, indent=2)}
            
            Check for:
            1. Timeline inconsistencies
            2. Character/faction motivation contradictions
            3. World rule violations
            4. Thematic inconsistencies with matriarchal setting
            
            Return JSON with validation results and any issues found.
            """
            
            result = await Runner.run(self.agent, prompt, context={})
            
            try:
                validation = json.loads(result.final_output)
                # Update consistency issues if any found
                if not validation.get("is_consistent", True):
                    self.coordination_memory["consistency_issues"].append({
                        "content_type": content_type,
                        "content_id": content.get("id", "unknown"),
                        "issues": validation.get("issues", []),
                        "detected_at": datetime.now().isoformat()
                    })
                return validation
            except json.JSONDecodeError:
                return {"is_consistent": False, "error": "Failed to parse validation", "raw_output": result.final_output}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the coordination system."""
        return {
            "subsystems": self.coordination_memory["subsystems"],
            "pending_tasks": len(self.coordination_memory["pending_tasks"]),
            "consistency_issues": len(self.coordination_memory["consistency_issues"])
        }
    
    async def _get_related_content(self, content: Dict[str, Any], content_type: str) -> List[Dict[str, Any]]:
        """Get existing content related to the new content for consistency checking."""
        # Implementation would depend on specific content relationships
        # This is a placeholder
        return []

class UnifiedTraceSystem:
    """
    System for unified tracing across all lore subsystems.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.trace_id = f"lore_{user_id}_{conversation_id}"
        self.traces = {}
    
    def start_trace(self, operation: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new trace for an operation."""
        metadata = metadata or {}
        metadata.update({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        })
        
        trace_id = f"{operation}_{uuid.uuid4()}"
        with trace(operation, group_id=self.trace_id, metadata=metadata) as current_trace:
            self.traces[trace_id] = {
                "id": trace_id,
                "operation": operation,
                "metadata": metadata,
                "started_at": datetime.now().isoformat(),
                "status": "running",
                "steps": [],
                "trace_obj": current_trace
            }
            return trace_id
    
    def add_trace_step(self, trace_id: str, step_name: str, data: Dict[str, Any] = None):
        """Add a step to an existing trace."""
        if trace_id not in self.traces:
            return
        
        with trace(
            step_name, 
            group_id=self.traces[trace_id]["trace_obj"].id, 
            metadata=data
        ):
            self.traces[trace_id]["steps"].append({
                "name": step_name,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            })
    
    def end_trace(self, trace_id: str, status: str = "completed", result: Dict[str, Any] = None):
        """End a trace with a status and result."""
        if trace_id not in self.traces:
            return
        
        self.traces[trace_id]["status"] = status
        self.traces[trace_id]["ended_at"] = datetime.now().isoformat()
        self.traces[trace_id]["result"] = result or {}
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get the details of a specific trace."""
        return self.traces.get(trace_id)
    
    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get all currently active traces."""
        return [t for t in self.traces.values() if t["status"] == "running"]
    
    def export_trace(self, trace_id: str, format_type: str = "json") -> Dict[str, Any]:
        """Export a trace in the specified format."""
        if trace_id not in self.traces:
            return {"error": "Trace not found"}
        
        trace_data = self.traces[trace_id]
        
        if format_type == "json":
            return trace_data
        elif format_type == "timeline":
            # Format for timeline visualization
            events = []
            events.append({
                "time": trace_data["started_at"],
                "event": f"Started {trace_data['operation']}",
                "type": "start"
            })
            
            for step in trace_data["steps"]:
                events.append({
                    "time": step["timestamp"],
                    "event": step["name"],
                    "data": step["data"],
                    "type": "step"
                })
            
            if trace_data["status"] != "running":
                events.append({
                    "time": trace_data["ended_at"],
                    "event": f"Ended {trace_data['operation']} with status {trace_data['status']}",
                    "type": "end"
                })
            
            return {"timeline": events}
        else:
            return {"error": f"Unsupported format: {format_type}"}

class ContentValidationTool:
    """
    Tool for validating and ensuring consistency of lore content.
    """
    
    def __init__(self, world_lore_manager):
        self.world_lore_manager = world_lore_manager
        self.validator_agent = Agent(
            name="ContentValidatorAgent",
            instructions="""
            You validate fantasy world lore for consistency, completeness, and thematic coherence.
            Check for contradictions with existing lore, missing required elements,
            and alignment with the matriarchal theme.
            """,
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def validate_content(self, content: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Validate the provided content against consistency rules."""
        # Get validation schema for this content type
        schema = self._get_validation_schema(content_type)
        
        # Basic structural validation
        schema_validation = self._validate_against_schema(content, schema)
        if not schema_validation["valid"]:
            return schema_validation
        
        # Get related content for contextual validation
        related_content = await self._fetch_related_content(content, content_type)
        
        # Prompt the validator agent
        prompt = f"""
        Validate this {content_type} content for consistency and quality:
        
        CONTENT:
        {json.dumps(content, indent=2)}
        
        RELATED EXISTING CONTENT:
        {json.dumps(related_content, indent=2)}
        
        Check for:
        1. Internal consistency
        2. Consistency with existing lore
        3. Completeness of required elements
        4. Proper matriarchal theming
        5. Narrative quality and interest
        
        Return JSON with:
        - valid: boolean
        - issues: list of specific issues
        - improvement_suggestions: list of suggestions
        - matriarchal_score: 1-10 rating of how well it upholds matriarchal themes
        """
        
        result = await Runner.run(self.validator_agent, prompt, context={})
        
        try:
            validation_result = json.loads(result.final_output)
            return validation_result
        except json.JSONDecodeError:
            return {
                "valid": False,
                "issues": ["Failed to parse validation result"],
                "raw_output": result.final_output
            }
    
    def _get_validation_schema(self, content_type: str) -> Dict[str, Any]:
        """Get the validation schema for a content type."""
        schemas = {
            "nation": {
                "required_fields": ["name", "government_type", "description"],
                "optional_fields": ["matriarchy_level", "population_scale", "major_resources"],
                "types": {
                    "name": str,
                    "government_type": str,
                    "description": str,
                    "matriarchy_level": int
                }
            },
            "deity": {
                "required_fields": ["name", "gender", "domain", "description"],
                "optional_fields": ["iconography", "holy_symbol", "sacred_animals"],
                "types": {
                    "name": str,
                    "gender": str,
                    "domain": list,
                    "description": str
                }
            }
            # Add schemas for other content types
        }
        
        return schemas.get(content_type, {
            "required_fields": [],
            "optional_fields": [],
            "types": {}
        })
    
    def _validate_against_schema(self, content: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content against a structural schema."""
        issues = []
        
        # Check required fields
        for field in schema.get("required_fields", []):
            if field not in content:
                issues.append(f"Missing required field: {field}")
        
        # Check types
        for field, expected_type in schema.get("types", {}).items():
            if field in content and not isinstance(content[field], expected_type):
                issues.append(f"Field {field} should be of type {expected_type.__name__}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    async def _fetch_related_content(self, content: Dict[str, Any], content_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content related to the provided content for contextual validation."""
        # Implementation would vary based on content relationships
        # Placeholder example:
        related = {}
        
        if content_type == "nation":
            # Get neighboring nations
            if "neighboring_nations" in content:
                nations = []
                for neighbor in content["neighboring_nations"]:
                    # This is a simplified example; actual implementation would query the database
                    nation_data = await self.world_lore_manager.get_world_data(f"nation_{neighbor}")
                    if nation_data:
                        nations.append(nation_data)
                related["neighboring_nations"] = nations
        
        return related

class LoreRelationshipMapper:
    """
    Tool for creating and managing relationships between lore elements.
    """
    
    def __init__(self, world_lore_manager):
        self.world_lore_manager = world_lore_manager
        self.relationship_agent = Agent(
            name="RelationshipMapperAgent",
            instructions="""
            You analyze fantasy world lore elements and identify meaningful relationships between them.
            These could be causal relationships, thematic connections, contradictions, or influences.
            Create a network of relationships that shows how lore elements interact.
            """,
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def create_relationship_graph(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a relationship graph from a set of lore elements."""
        if not elements:
            return {"nodes": [], "edges": []}
        
        prompt = f"""
        Analyze these lore elements and create a relationship graph that shows how they connect:
        
        ELEMENTS:
        {json.dumps(elements, indent=2)}
        
        For each possible pair of elements, determine if a meaningful relationship exists.
        Consider:
        - Causal relationships (one created/influenced the other)
        - Thematic connections
        - Contradictions or conflicts
        - Geographical proximity
        - Hierarchical relationships
        
        Return a JSON graph with:
        - nodes: list of element IDs
        - edges: list of connections with source, target, and relationship type
        """
        
        result = await Runner.run(self.relationship_agent, prompt, context={})
        
        try:
            graph = json.loads(result.final_output)
            # Store the relationship graph
            await self._store_relationship_graph(graph)
            return graph
        except json.JSONDecodeError:
            return {"error": "Failed to parse relationship graph", "raw_output": result.final_output}
    
    async def find_related_elements(self, element_id: str, element_type: str, depth: int = 1) -> Dict[str, Any]:
        """Find lore elements related to the specified element."""
        # Get the element
        element = await self.world_lore_manager.get_world_data(f"{element_type}_{element_id}")
        if not element:
            return {"error": "Element not found"}
        
        # Get previously mapped relationships
        relationships = await self._get_element_relationships(element_id)
        
        # If we need to go deeper or no relationships exist, use the agent
        if depth > 1 or not relationships:
            # Get potential related elements based on type
            potential_related = await self._get_potential_related(element_type, element)
            
            prompt = f"""
            Find relationships between this element and potentially related elements:
            
            PRIMARY ELEMENT:
            {json.dumps(element, indent=2)}
            
            POTENTIAL RELATED ELEMENTS:
            {json.dumps(potential_related, indent=2)}
            
            For each potential related element, determine:
            1. If a meaningful relationship exists
            2. What type of relationship it is
            3. The strength of the relationship (1-10)
            
            Return JSON with an array of related elements and their relationship details.
            """
            
            result = await Runner.run(self.relationship_agent, prompt, context={})
            
            try:
                new_relationships = json.loads(result.final_output)
                # Store these new relationships
                await self._store_element_relationships(element_id, new_relationships)
                
                # Merge with existing relationships
                relationships = self._merge_relationships(relationships, new_relationships)
                
                # If depth > 1, recursively get related elements of related elements
                if depth > 1:
                    for related in relationships.get("related_elements", []):
                        related_id = related.get("id")
                        related_type = related.get("type")
                        if related_id and related_type:
                            second_level = await self.find_related_elements(related_id, related_type, depth=depth-1)
                            related["connections"] = second_level.get("related_elements", [])
            except json.JSONDecodeError:
                relationships = {"error": "Failed to parse relationships", "raw_output": result.final_output}
        
        return relationships
    
    async def _store_relationship_graph(self, graph: Dict[str, Any]) -> None:
        """Store a relationship graph in the database."""
        # Implementation would depend on database structure
        pass
    
    async def _get_element_relationships(self, element_id: str) -> Dict[str, Any]:
        """Get previously mapped relationships for an element."""
        # Implementation would depend on database structure
        return {"related_elements": []}
    
    async def _store_element_relationships(self, element_id: str, relationships: Dict[str, Any]) -> None:
        """Store relationships for an element."""
        # Implementation would depend on database structure
        pass
    
    async def _get_potential_related(self, element_type: str, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get potential related elements based on element type and content."""
        # Implementation would depend on specific lore relationships
        # This is a placeholder
        return []
    
    def _merge_relationships(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge existing and new relationship data."""
        if "related_elements" not in existing:
            existing["related_elements"] = []
        
        existing_ids = {r.get("id") for r in existing["related_elements"]}
        
        for related in new.get("related_elements", []):
            if related.get("id") not in existing_ids:
                existing["related_elements"].append(related)
        
        return existing
        
class WorldQueryAgent(BaseModel):
    """Simulate world query processing."""
    query: str

    @function_tool
    async def process_query(self, ctx: RunContextWrapper) -> str:
        """Process the procedural query and return relevant world data."""
        # Example: Simply return a mock response for the query for demonstration purposes
        return f"Processing query: {self.query}"

class WorldDocumentationAgent(BaseModel):
    """Generate readable summaries of world history and current state."""
    world_id: str
    include_history: bool = True
    include_current_state: bool = True

    @function_tool
    async def generate_documentation(self, ctx: RunContextWrapper) -> str:
        """Generate documentation for the world state and history."""
        documentation = f"World {self.world_id} Summary:\n"
        if self.include_history:
            history = await self.get_world_history(self.world_id)
            documentation += f"\nWorld History:\n{history}"

        if self.include_current_state:
            current_state = await self.get_world_data(self.world_id)
            documentation += f"\nCurrent State:\n{current_state}"

        return documentation

    async def get_world_history(self, world_id: str) -> str:
        """Fetch world history."""
        # In a real implementation, this would query the database or cache
        return f"History of world {world_id}"

    async def get_world_data(self, world_id: str) -> str:
        """Fetch current world state."""
        return f"Current state of world {world_id}"

class InconsistencyResolutionAgent(BaseModel):
    """Resolve inconsistencies between world lore elements."""
    world_id: str

    @function_tool
    async def resolve_inconsistencies(self, ctx: RunContextWrapper) -> str:
        """Analyze world lore and resolve any inconsistencies."""
        # For demonstration, return a mock resolution.
        return f"Analyzed inconsistencies in world {self.world_id} and proposed fixes."
