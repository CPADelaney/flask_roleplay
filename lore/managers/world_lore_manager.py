# lore/manager/world_lore_manager.py

"""
World Lore Manager with Resource Management (Agent-ified)

This module provides world lore management with integrated resource management,
allowing each core operation to be called as an agent function tool if desired.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Union, ClassVar
from datetime import datetime
import uuid
import json
import asyncio
import os
import asyncpg
import random
import re

# Agents SDK (import what you need)
from agents import Agent, function_tool, Runner, trace, ModelSettings, handoff
from agents.run import RunConfig
from agents.run_context import RunContextWrapper

# Import for dependencies
from pydantic import BaseModel, Field

# Database connection string
DB_DSN = os.getenv("DB_DSN") 

logger = logging.getLogger(__name__)

from lore.managers.base_manager import BaseManager
from lore.resource_manager import resource_manager

# Because we might want to store or retrieve data from an LLM orchestrator
# we define a ResourceOpsAgent for demonstration:
RESOURCE_OPS_AGENT = Agent(
    name="ResourceOpsAgent",
    instructions=(
        "You manage resources and data caching for world lore. "
        "Your tool methods allow for retrieving, setting, invalidating data in a cache, "
        "and checking resource usage. Keep everything thread-safe and consistent."
    ),
    model="o3-mini"
)

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

    # Implementation of cultural diffusion methods
    async def _apply_language_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply language diffusion effects between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if both nations have languages
                nation1_langs = await conn.fetch("""
                    SELECT * FROM Languages 
                    WHERE $1 = ANY(primary_regions) OR $1 = ANY(minority_regions)
                """, nation1_id)
                
                nation2_langs = await conn.fetch("""
                    SELECT * FROM Languages 
                    WHERE $1 = ANY(primary_regions) OR $1 = ANY(minority_regions)
                """, nation2_id)
                
                if not nation1_langs or not nation2_langs:
                    return
                
                # Apply vocabulary diffusion
                if "vocabulary" in effects:
                    for vocab_change in effects["vocabulary"]:
                        # Update common phrases or add new ones
                        for lang in nation1_langs:
                            lang_id = lang["id"]
                            common_phrases = lang.get("common_phrases", {})
                            if isinstance(common_phrases, str):
                                try:
                                    common_phrases = json.loads(common_phrases)
                                except:
                                    common_phrases = {}
                            
                            # Add new phrases from the other nation
                            for phrase, meaning in vocab_change.get("adopted_phrases", {}).items():
                                common_phrases[phrase] = meaning
                            
                            # Update the language
                            await conn.execute("""
                                UPDATE Languages
                                SET common_phrases = $1
                                WHERE id = $2
                            """, json.dumps(common_phrases), lang_id)
                
                # Record the cultural exchange
                exchange_id = await conn.fetchval("""
                    INSERT INTO CulturalExchanges (
                        nation1_id, nation2_id, exchange_type, exchange_details, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, nation1_id, nation2_id, "language_diffusion", json.dumps(effects), datetime.now())
                
                # Log the exchange in world history
                await conn.execute("""
                    INSERT INTO WorldHistory (
                        event_type, description, involved_entities, timestamp
                    )
                    VALUES ($1, $2, $3, $4)
                """, "cultural_exchange", 
                f"Language exchange occurred between nations {nation1_id} and {nation2_id}",
                json.dumps([nation1_id, nation2_id]), datetime.now())

    async def _apply_artistic_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply artistic and creative diffusion between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Record artistic exchange in cultural elements
                if "artistic_elements" in effects:
                    for element in effects["artistic_elements"]:
                        # Check if the cultural element already exists
                        existing = await conn.fetchrow("""
                            SELECT id FROM CulturalElements
                            WHERE name = $1 AND element_type = 'artistic'
                        """, element["name"])
                        
                        if existing:
                            # Update existing element
                            await conn.execute("""
                                UPDATE CulturalElements
                                SET description = $1, practiced_by = array_append(practiced_by, $2)
                                WHERE id = $3
                            """, element["description"], f"Nation {nation2_id}", existing["id"])
                        else:
                            # Create new cultural element
                            await conn.execute("""
                                INSERT INTO CulturalElements (
                                    name, element_type, description, practiced_by, significance,
                                    historical_origin
                                )
                                VALUES ($1, $2, $3, $4, $5, $6)
                            """, element["name"], "artistic", element["description"],
                            [f"Nation {nation1_id}", f"Nation {nation2_id}"],
                            element.get("significance", 5),
                            f"Cultural exchange between nations {nation1_id} and {nation2_id}")

    async def _apply_religious_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply religious practice and belief diffusion between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Track religious changes in the appropriate tables
                # This is a simplified implementation
                if "religious_practices" in effects:
                    for practice in effects["religious_practices"]:
                        # Add or update practice in religious tables
                        existing = await conn.fetchrow("""
                            SELECT id FROM ReligiousPractices
                            WHERE name = $1
                        """, practice["name"])
                        
                        if existing:
                            # Update existing practice
                            await conn.execute("""
                                UPDATE ReligiousPractices
                                SET description = $1, followers = array_append(followers, $2)
                                WHERE id = $3
                            """, practice["description"], f"Nation {nation2_id}", existing["id"])
                        else:
                            # Create new practice
                            await conn.execute("""
                                INSERT INTO ReligiousPractices (
                                    name, description, origin, followers, significance
                                )
                                VALUES ($1, $2, $3, $4, $5)
                            """, practice["name"], practice["description"],
                            f"Cultural exchange with Nation {nation1_id}",
                            [f"Nation {nation1_id}", f"Nation {nation2_id}"],
                            practice.get("significance", 5))

    async def _apply_fashion_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply fashion and clothing diffusion between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Implement fashion diffusion effects
                # This is a simplified implementation
                if "fashion_elements" in effects:
                    for element in effects["fashion_elements"]:
                        # Add to cultural elements table with fashion type
                        await conn.execute("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by, significance,
                                historical_origin
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (name, element_type) 
                            DO UPDATE SET description = EXCLUDED.description,
                                        practiced_by = array_append(CulturalElements.practiced_by, $7)
                        """, element["name"], "fashion", element["description"],
                        [f"Nation {nation2_id}"], element.get("significance", 5),
                        f"Adopted from Nation {nation1_id}", f"Nation {nation2_id}")

    async def _apply_cuisine_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply culinary and food diffusion between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Implement cuisine diffusion
                if "cuisine_elements" in effects:
                    for dish in effects["cuisine_elements"]:
                        # Add to culinary database or cultural elements
                        await conn.execute("""
                            INSERT INTO CulinaryTraditions (
                                name, nation_origin, description, ingredients, preparation,
                                cultural_significance, adopted_by
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (name) 
                            DO UPDATE SET adopted_by = array_append(CulinaryTraditions.adopted_by, $8)
                        """, dish["name"], nation1_id, dish["description"],
                        dish.get("ingredients", []), dish.get("preparation", ""),
                        dish.get("significance", ""), [nation2_id], nation2_id)

    async def _apply_customs_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply social customs and etiquette diffusion between nations."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Implement social customs diffusion
                if "social_customs" in effects:
                    for custom in effects["social_customs"]:
                        # Create or update social norms/customs
                        await conn.execute("""
                            INSERT INTO SocialCustoms (
                                name, nation_origin, description, context, formality_level,
                                adopted_by, adoption_date
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (name) 
                            DO UPDATE SET adopted_by = array_append(SocialCustoms.adopted_by, $8)
                        """, custom["name"], nation1_id, custom["description"],
                        custom.get("context", "social"), custom.get("formality_level", "medium"),
                        [nation2_id], datetime.now(), nation2_id)

    async def _update_plan_step(self, plan_id: str, step_index: int, outcome: Dict[str, Any]) -> None:
        """Update a plan step with its outcome."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # First, fetch the current plan
                plan_data = await conn.fetchrow("""
                    SELECT plan_data FROM NarrativePlans 
                    WHERE id = $1
                """, plan_id)
                
                if not plan_data:
                    logging.error(f"Plan {plan_id} not found")
                    return
                
                # Parse plan JSON
                try:
                    plan = json.loads(plan_data["plan_data"])
                except (json.JSONDecodeError, KeyError):
                    logging.error(f"Failed to parse plan data for {plan_id}")
                    return
                
                # Update the step with the outcome
                if "steps" in plan and 0 <= step_index < len(plan["steps"]):
                    plan["steps"][step_index]["status"] = "completed"
                    plan["steps"][step_index]["outcome"] = outcome
                    plan["steps"][step_index]["completed_at"] = datetime.now().isoformat()
                    
                    # Update the overall plan status if all steps are complete
                    all_completed = all(step.get("status") == "completed" for step in plan["steps"])
                    if all_completed:
                        plan["status"] = "completed"
                        plan["completed_at"] = datetime.now().isoformat()
                    
                    # Save the updated plan
                    await conn.execute("""
                        UPDATE NarrativePlans
                        SET plan_data = $1,
                            status = $2,
                            updated_at = $3
                        WHERE id = $4
                    """, json.dumps(plan), plan["status"], datetime.now(), plan_id)
                    
                    # Log the step completion
                    await conn.execute("""
                        INSERT INTO PlanExecutionLog (
                            plan_id, step_index, step_title, outcome_summary, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5)
                    """, plan_id, step_index, 
                       plan["steps"][step_index].get("title", f"Step {step_index}"),
                       json.dumps(outcome), datetime.now())
                else:
                    logging.error(f"Invalid step index {step_index} for plan {plan_id}")

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
        related_content = []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                if content_type == "faction":
                    # Find factions with shared territory, rivals, or allies
                    if "territory" in content:
                        territory_factions = await conn.fetch("""
                            SELECT id, name, type, description, territory 
                            FROM Factions
                            WHERE territory && $1
                        """, content.get("territory", []))
                        related_content.extend([dict(f) for f in territory_factions])
                    
                    # Find related by mentioned rivals
                    if "rivals" in content:
                        rival_factions = await conn.fetch("""
                            SELECT id, name, type, description 
                            FROM Factions
                            WHERE name = ANY($1)
                        """, content.get("rivals", []))
                        related_content.extend([dict(f) for f in rival_factions])
                    
                    # Find related by mentioned allies
                    if "allies" in content:
                        ally_factions = await conn.fetch("""
                            SELECT id, name, type, description 
                            FROM Factions
                            WHERE name = ANY($1)
                        """, content.get("allies", []))
                        related_content.extend([dict(f) for f in ally_factions])
                
                elif content_type == "location":
                    # Find locations in the same region or connected
                    if "region" in content:
                        region_locations = await conn.fetch("""
                            SELECT id, name, description, type, controlling_faction 
                            FROM Locations
                            WHERE region = $1
                        """, content.get("region"))
                        related_content.extend([dict(l) for l in region_locations])
                    
                    # Find locations controlled by the same faction
                    if "controlling_faction" in content:
                        faction_locations = await conn.fetch("""
                            SELECT id, name, description, type
                            FROM Locations
                            WHERE controlling_faction = $1
                        """, content.get("controlling_faction"))
                        related_content.extend([dict(l) for l in faction_locations])
                
                elif content_type == "historical_event":
                    # Find events in the same time period
                    if "date_description" in content:
                        related_events = await conn.fetch("""
                            SELECT id, name, description, date_description, significance 
                            FROM HistoricalEvents
                            WHERE date_description LIKE '%' || $1 || '%'
                        """, content.get("date_description", ""))
                        related_content.extend([dict(e) for e in related_events])
                    
                    # Find events with the same participating factions
                    if "participating_factions" in content:
                        faction_events = await conn.fetch("""
                            SELECT id, name, description, date_description, participating_factions
                            FROM HistoricalEvents
                            WHERE participating_factions && $1
                        """, content.get("participating_factions", []))
                        related_content.extend([dict(e) for e in faction_events])
                
                elif content_type == "character" or content_type == "notable_figure":
                    # Find characters with same affiliations
                    if "affiliations" in content:
                        affiliation_chars = await conn.fetch("""
                            SELECT id, name, description, affiliations 
                            FROM NotableFigures
                            WHERE affiliations && $1
                        """, content.get("affiliations", []))
                        related_content.extend([dict(c) for c in affiliation_chars])
                    
                    # Find characters with same titles or roles
                    if "titles" in content:
                        titled_chars = await conn.fetch("""
                            SELECT id, name, description, titles 
                            FROM NotableFigures
                            WHERE titles && $1
                        """, content.get("titles", []))
                        related_content.extend([dict(c) for c in titled_chars])
        
        # Remove duplicates based on id
        seen_ids = set()
        unique_related = []
        for item in related_content:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_related.append(item)
        
        return unique_related

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
        related = {}
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                if content_type == "nation":
                    # Get neighboring nations
                    if "neighboring_nations" in content:
                        nations = []
                        for neighbor in content["neighboring_nations"]:
                            nation_data = await conn.fetchrow("""
                                SELECT * FROM Nations
                                WHERE id = $1
                            """, neighbor)
                            if nation_data:
                                nations.append(dict(nation_data))
                        related["neighboring_nations"] = nations
                    
                    # Get religions practiced in the nation
                    religions = await conn.fetch("""
                        SELECT r.* 
                        FROM Religions r
                        JOIN NationReligions nr ON r.id = nr.religion_id
                        WHERE nr.nation_id = $1
                    """, content.get("id"))
                    if religions:
                        related["religions"] = [dict(r) for r in religions]
                    
                    # Get cultural traditions
                    traditions = await conn.fetch("""
                        SELECT ce.* 
                        FROM CulturalElements ce
                        WHERE $1 = ANY(ce.practiced_by)
                    """, content.get("name", ""))
                    if traditions:
                        related["cultural_traditions"] = [dict(t) for t in traditions]
                
                elif content_type == "religion":
                    # Get nations where this religion is practiced
                    nations = await conn.fetch("""
                        SELECT n.* 
                        FROM Nations n
                        JOIN NationReligions nr ON n.id = nr.nation_id
                        WHERE nr.religion_id = $1
                    """, content.get("id"))
                    if nations:
                        related["practicing_nations"] = [dict(n) for n in nations]
                    
                    # Get deities associated with this religion
                    deities = await conn.fetch("""
                        SELECT * FROM Deities
                        WHERE religion_id = $1
                    """, content.get("id"))
                    if deities:
                        related["deities"] = [dict(d) for d in deities]
                    
                    # Get religious practices
                    practices = await conn.fetch("""
                        SELECT * FROM ReligiousPractices
                        WHERE religion_id = $1
                    """, content.get("id"))
                    if practices:
                        related["practices"] = [dict(p) for p in practices]
                
                elif content_type == "faction":
                    # Get faction leaders
                    leaders = await conn.fetch("""
                        SELECT nf.* 
                        FROM NotableFigures nf
                        WHERE nf.id = ANY($1)
                    """, content.get("leadership", []))
                    if leaders:
                        related["leaders"] = [dict(l) for l in leaders]
                    
                    # Get controlled locations
                    locations = await conn.fetch("""
                        SELECT * FROM Locations
                        WHERE controlling_faction = $1
                    """, content.get("name", ""))
                    if locations:
                        related["controlled_locations"] = [dict(l) for l in locations]
                    
                    # Get rival factions
                    rivals = await conn.fetch("""
                        SELECT * FROM Factions
                        WHERE name = ANY($1)
                    """, content.get("rivals", []))
                    if rivals:
                        related["rival_factions"] = [dict(r) for r in rivals]
                
                elif content_type == "location":
                    # Get controlling faction
                    faction = await conn.fetchrow("""
                        SELECT * FROM Factions
                        WHERE name = $1
                    """, content.get("controlling_faction", ""))
                    if faction:
                        related["controlling_faction"] = dict(faction)
                    
                    # Get historical events at this location
                    events = await conn.fetch("""
                        SELECT * FROM HistoricalEvents
                        WHERE $1 = ANY(affected_locations)
                    """, content.get("name", ""))
                    if events:
                        related["historical_events"] = [dict(e) for e in events]
        
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
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Create transaction
                async with conn.transaction():
                    # Store the graph metadata
                    graph_id = await conn.fetchval("""
                        INSERT INTO LoreRelationshipGraphs (
                            user_id, creation_date, graph_name, description
                        ) VALUES ($1, $2, $3, $4)
                        RETURNING id
                    """, self.user_id, datetime.now(), graph.get("name", "Graph"), graph.get("description", ""))
                    
                    # Store all nodes
                    for node in graph.get("nodes", []):
                        await conn.execute("""
                            INSERT INTO LoreGraphNodes (
                                graph_id, node_id, lore_type, lore_id, label, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                        """, graph_id, node.get("id"), node.get("type"), 
                        node.get("lore_id"), node.get("label"), json.dumps(node.get("metadata", {})))
                    
                    # Store all edges
                    for edge in graph.get("edges", []):
                        await conn.execute("""
                            INSERT INTO LoreGraphEdges (
                                graph_id, source_id, target_id, relationship_type, 
                                strength, directional, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, graph_id, edge.get("source"), edge.get("target"),
                        edge.get("type"), edge.get("strength", 1.0), 
                        edge.get("directional", True), json.dumps(edge.get("metadata", {})))
    
    async def _get_element_relationships(self, element_id: str) -> Dict[str, Any]:
        """Get previously mapped relationships for an element."""
        relationships = {"related_elements": []}
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'loregraphedges'
                    );
                """)
                
                if not table_exists:
                    return relationships
                
                # Get all outgoing relationships
                outgoing = await conn.fetch("""
                    SELECT e.*, n.lore_type, n.label
                    FROM LoreGraphEdges e
                    JOIN LoreGraphNodes n ON e.target_id = n.node_id
                    WHERE e.source_id = $1
                """, element_id)
                
                # Get all incoming relationships
                incoming = await conn.fetch("""
                    SELECT e.*, n.lore_type, n.label
                    FROM LoreGraphEdges e
                    JOIN LoreGraphNodes n ON e.source_id = n.node_id
                    WHERE e.target_id = $1
                """, element_id)
                
                for rel in outgoing:
                    relationships["related_elements"].append({
                        "id": rel["target_id"],
                        "type": rel["lore_type"],
                        "name": rel["label"],
                        "relationship_type": rel["relationship_type"],
                        "relationship_strength": rel["strength"],
                        "direction": "outgoing"
                    })
                
                for rel in incoming:
                    relationships["related_elements"].append({
                        "id": rel["source_id"],
                        "type": rel["lore_type"],
                        "name": rel["label"],
                        "relationship_type": rel["relationship_type"],
                        "relationship_strength": rel["strength"],
                        "direction": "incoming"
                    })
        
        return relationships
    
    async def _store_element_relationships(self, element_id: str, relationships: Dict[str, Any]) -> None:
        """Store relationships for an element."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Create transaction
                async with conn.transaction():
                    # Find graph ID for this element, or create a new one
                    graph_id = await conn.fetchval("""
                        SELECT graph_id FROM LoreGraphNodes
                        WHERE node_id = $1
                        LIMIT 1
                    """, element_id)
                    
                    if not graph_id:
                        graph_id = await conn.fetchval("""
                            INSERT INTO LoreRelationshipGraphs (
                                user_id, creation_date, graph_name, description
                            ) VALUES ($1, $2, $3, $4)
                            RETURNING id
                        """, self.user_id, datetime.now(), "Generated Graph", "Automatically generated relationships")
                        
                        # Ensure the element node exists
                        await conn.execute("""
                            INSERT INTO LoreGraphNodes (
                                graph_id, node_id, lore_type, lore_id, label
                            ) VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (graph_id, node_id) DO NOTHING
                        """, graph_id, element_id, "unknown", element_id, f"Element {element_id}")
                    
                    # Add all related elements
                    for relation in relationships.get("related_elements", []):
                        related_id = relation.get("id")
                        relationship_type = relation.get("relationship_type", "related")
                        strength = relation.get("relationship_strength", 1.0)
                        direction = relation.get("direction", "outgoing")
                        
                        # Ensure the related node exists
                        await conn.execute("""
                            INSERT INTO LoreGraphNodes (
                                graph_id, node_id, lore_type, lore_id, label
                            ) VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (graph_id, node_id) DO NOTHING
                        """, graph_id, related_id, relation.get("type", "unknown"), 
                        related_id, relation.get("name", f"Element {related_id}"))
                        
                        # Add the edge in the proper direction
                        if direction == "outgoing":
                            source_id, target_id = element_id, related_id
                        else:
                            source_id, target_id = related_id, element_id
                        
                        await conn.execute("""
                            INSERT INTO LoreGraphEdges (
                                graph_id, source_id, target_id, relationship_type, 
                                strength, directional
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (graph_id, source_id, target_id) 
                            DO UPDATE SET relationship_type = EXCLUDED.relationship_type,
                                        strength = EXCLUDED.strength
                        """, graph_id, source_id, target_id, relationship_type, 
                        strength, True)
    
    async def _get_potential_related(self, element_type: str, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get potential related elements based on element type and content."""
        related_elements = []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                if element_type == "faction":
                    # Get factions with similar territory
                    if "territory" in element:
                        similar_territory = await conn.fetch("""
                            SELECT id, name, type, description, territory 
                            FROM Factions
                            WHERE id != $1 AND territory && $2
                            LIMIT 5
                        """, element.get("id", -1), element.get("territory", []))
                        
                        for faction in similar_territory:
                            related_elements.append({
                                "id": faction["id"],
                                "name": faction["name"],
                                "type": "faction",
                                "potential_relationship": "territorial"
                            })
                    
                    # Get factions with similar goals
                    if "goals" in element:
                        goals_str = " ".join(element.get("goals", []))
                        if goals_str:
                            similar_goals = await conn.fetch("""
                                SELECT id, name, type, description, goals 
                                FROM Factions
                                WHERE id != $1 AND 
                                      SIMILARITY(
                                        ARRAY_TO_STRING(goals, ' '), $2
                                      ) > 0.3
                                LIMIT 5
                            """, element.get("id", -1), goals_str)
                            
                            for faction in similar_goals:
                                related_elements.append({
                                    "id": faction["id"],
                                    "name": faction["name"],
                                    "type": "faction",
                                    "potential_relationship": "ideological"
                                })
                
                elif element_type == "location":
                    # Get nearby locations
                    if "region" in element:
                        nearby_locations = await conn.fetch("""
                            SELECT id, name, type, description, region
                            FROM Locations
                            WHERE id != $1 AND region = $2
                            LIMIT 5
                        """, element.get("id", -1), element.get("region"))
                        
                        for location in nearby_locations:
                            related_elements.append({
                                "id": location["id"],
                                "name": location["name"],
                                "type": "location",
                                "potential_relationship": "proximity"
                            })
                    
                    # Get locations controlled by same faction
                    if "controlling_faction" in element and element["controlling_faction"]:
                        faction_locations = await conn.fetch("""
                            SELECT id, name, type, description
                            FROM Locations
                            WHERE id != $1 AND controlling_faction = $2
                            LIMIT 5
                        """, element.get("id", -1), element.get("controlling_faction"))
                        
                        for location in faction_locations:
                            related_elements.append({
                                "id": location["id"],
                                "name": location["name"],
                                "type": "location",
                                "potential_relationship": "political"
                            })
                
                elif element_type == "historical_event":
                    # Get events in similar time period
                    if "date_description" in element:
                        similar_time = await conn.fetch("""
                            SELECT id, name, description, date_description
                            FROM HistoricalEvents
                            WHERE id != $1 AND 
                                  SIMILARITY(date_description, $2) > 0.3
                            LIMIT 5
                        """, element.get("id", -1), element.get("date_description", ""))
                        
                        for event in similar_time:
                            related_elements.append({
                                "id": event["id"],
                                "name": event["name"],
                                "type": "historical_event",
                                "potential_relationship": "temporal"
                            })
                    
                    # Get events with shared participants
                    if "participating_factions" in element:
                        shared_participants = await conn.fetch("""
                            SELECT id, name, description, participating_factions
                            FROM HistoricalEvents
                            WHERE id != $1 AND 
                                  participating_factions && $2
                            LIMIT 5
                        """, element.get("id", -1), element.get("participating_factions", []))
                        
                        for event in shared_participants:
                            related_elements.append({
                                "id": event["id"],
                                "name": event["name"],
                                "type": "historical_event",
                                "potential_relationship": "participant"
                            })
        
        # Ensure unique results
        seen_ids = set()
        unique_results = []
        for item in related_elements:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_results.append(item)
        
        return unique_results
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
    """Agent that processes queries about the world state."""
    query: str
    world_id: str = "main"
    process_query: ClassVar

    @function_tool
    async def process_query(self, ctx: Optional[RunContextWrapper] = None) -> str:
        """Process a natural language query about the world state."""
        # Extract key entities and intents from the query
        query_terms = self.query.lower().split()
        entity_types = ["faction", "location", "history", "character", "event", "magic"]
        
        # Determine query intent
        intent = "general"
        if any(term in query_terms for term in ["who", "person", "character", "leader"]):
            intent = "character"
        elif any(term in query_terms for term in ["where", "place", "location", "city"]):
            intent = "location"
        elif any(term in query_terms for term in ["when", "time", "date", "year", "history"]):
            intent = "timeline"
        elif any(term in query_terms for term in ["how", "works", "system", "magic"]):
            intent = "system"
        
        # Construct database query based on intent
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                if intent == "character":
                    results = await conn.fetch("""
                        SELECT * FROM NotableFigures 
                        WHERE world_id = $1 
                        ORDER BY significance DESC LIMIT 5
                    """, self.world_id)
                    
                    response = "Here are the notable characters in this world:\n\n"
                    for result in results:
                        response += f"- {result['name']}: {result['description'][:100]}...\n"
                
                elif intent == "location":
                    results = await conn.fetch("""
                        SELECT * FROM Locations 
                        WHERE world_id = $1 
                        ORDER BY significance DESC LIMIT 5
                    """, self.world_id)
                    
                    response = "Here are significant locations in this world:\n\n"
                    for result in results:
                        response += f"- {result['name']}: {result['description'][:100]}...\n"
                
                elif intent == "timeline":
                    results = await conn.fetch("""
                        SELECT * FROM HistoricalEvents 
                        WHERE world_id = $1 
                        ORDER BY date_order DESC LIMIT 5
                    """, self.world_id)
                    
                    response = "Key historical events in chronological order:\n\n"
                    for result in results:
                        response += f"- {result['date_description']}: {result['name']} - {result['description'][:100]}...\n"
                
                else:
                    # General world information
                    result = await conn.fetchrow("""
                        SELECT * FROM WorldLore 
                        WHERE id = $1 LIMIT 1
                    """, self.world_id)
                    
                    if result:
                        response = f"World '{result['name']}' overview:\n\n"
                        response += result['description']
                    else:
                        response = f"No general information found for world {self.world_id}"
        
        return response


class WorldDocumentationAgent(BaseModel):
    """Generate readable summaries of world history and current state."""
    world_id: str
    include_history: bool = True
    include_current_state: bool = True
    
    # Tell Pydantic these aren't fields
    generate_documentation: ClassVar
    get_world_history: ClassVar
    get_world_data: ClassVar
    
    @function_tool
    async def generate_documentation(self, ctx: Optional[RunContextWrapper] = None) -> str:
        """Generate documentation for the world state and history."""
        documentation = f"# World Documentation: {self.world_id}\n\n"
        
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                # Get world overview
                world_data = await conn.fetchrow("""
                    SELECT * FROM WorldLore WHERE id = $1
                """, self.world_id)
                
                if world_data:
                    documentation += f"## Overview\n\n{world_data['description']}\n\n"
                
                if self.include_history:
                    # Get historical events
                    history_data = await conn.fetch("""
                        SELECT * FROM HistoricalEvents 
                        WHERE world_id = $1
                        ORDER BY date_order
                    """, self.world_id)
                    
                    documentation += "## Historical Timeline\n\n"
                    for event in history_data:
                        documentation += f"### {event['name']} ({event['date_description']})\n\n"
                        documentation += f"{event['description']}\n\n"
                        documentation += f"**Significance**: {event['significance']}/10\n\n"

                if self.include_current_state:
                    # Get current factions
                    factions = await conn.fetch("""
                        SELECT * FROM Factions 
                        WHERE world_id = $1
                    """, self.world_id)
                    
                    documentation += "## Current Factions\n\n"
                    for faction in factions:
                        documentation += f"### {faction['name']}\n\n"
                        documentation += f"**Type**: {faction['type']}\n\n"
                        documentation += f"{faction['description']}\n\n"
                        
                        if faction.get('values'):
                            values = faction['values'] if isinstance(faction['values'], list) else json.loads(faction['values'])
                            documentation += "**Values**: " + ", ".join(values) + "\n\n"
                    
                    # Get locations
                    locations = await conn.fetch("""
                        SELECT * FROM Locations 
                        WHERE world_id = $1
                    """, self.world_id)
                    
                    documentation += "## Significant Locations\n\n"
                    for location in locations:
                        documentation += f"### {location['name']}\n\n"
                        documentation += f"**Type**: {location['type']}\n\n"
                        documentation += f"{location['description']}\n\n"
        
        return documentation

    async def get_world_history(self, world_id: str) -> str:
        """Fetch world history for the documentation agent."""
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                # Get the historical events
                events = await conn.fetch("""
                    SELECT * FROM HistoricalEvents
                    WHERE world_id = $1
                    ORDER BY date_order
                """, world_id)
                
                if not events:
                    return f"No historical records found for world {world_id}."
                
                history = f"## History of {world_id}\n\n"
                
                # Group events by time periods
                time_periods = {}
                for event in events:
                    period = event.get("time_period", "Unknown Era")
                    if period not in time_periods:
                        time_periods[period] = []
                    time_periods[period].append(dict(event))
                
                # Format the history by time periods
                for period, period_events in time_periods.items():
                    history += f"### {period}\n\n"
                    for event in period_events:
                        history += f"**{event['name']}** ({event['date_description']}): {event['description']}\n\n"
                        if event.get("consequences"):
                            history += f"*Consequences*: {', '.join(event['consequences'])}\n\n"
                
                return history

    async def get_world_data(self, world_id: str) -> str:
        """Fetch current world state for the documentation agent."""
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                # Get the world data
                world = await conn.fetchrow("""
                    SELECT * FROM worlds
                    WHERE id = $1
                """, world_id)
                
                if not world:
                    return f"No data found for world {world_id}."
                
                world_data = dict(world)
                
                # Get factions
                factions = await conn.fetch("""
                    SELECT * FROM Factions
                    WHERE world_id = $1
                """, world_id)
                
                # Get nations
                nations = await conn.fetch("""
                    SELECT * FROM Nations
                    WHERE world_id = $1
                """, world_id)
                
                # Get major locations
                locations = await conn.fetch("""
                    SELECT * FROM Locations
                    WHERE world_id = $1
                    ORDER BY significance DESC
                    LIMIT 10
                """, world_id)
                
                # Format the state information
                state = f"## Current State of {world_data.get('name', world_id)}\n\n"
                state += f"{world_data.get('description', 'No description available.')}\n\n"
                
                state += "### Major Powers\n\n"
                for nation in nations:
                    state += f"**{nation['name']}** ({nation['government_type']}): {nation['description']}\n\n"
                
                state += "### Active Factions\n\n"
                for faction in factions:
                    state += f"**{faction['name']}** ({faction['type']}): {faction['description']}\n\n"
                
                state += "### Notable Locations\n\n"
                for location in locations:
                    state += f"**{location['name']}** ({location['type']}): {location['description']}\n\n"
                
                return state

class InconsistencyResolutionAgent(BaseModel):
    """Resolve inconsistencies between world lore elements."""
    world_id: str
    
    # Tell Pydantic these aren't fields
    resolve_inconsistencies: ClassVar
    identify_inconsistencies: ClassVar
    resolve_single_inconsistency: ClassVar
    
    @function_tool
    async def resolve_inconsistencies(self, ctx: Optional[RunContextWrapper] = None) -> str:
        """Analyze world lore and resolve any inconsistencies."""
        inconsistencies = await self.identify_inconsistencies()
        
        if not inconsistencies:
            return f"No inconsistencies found in world {self.world_id}."
        
        # Process each inconsistency
        resolutions = []
        
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                for inconsistency in inconsistencies:
                    resolution = await self.resolve_single_inconsistency(conn, inconsistency)
                    resolutions.append({
                        "issue": inconsistency,
                        "resolution": resolution
                    })
                    
                    # Log the resolution in the database
                    await conn.execute("""
                        INSERT INTO LoreInconsistencyLog 
                        (world_id, issue_description, resolution, timestamp)
                        VALUES ($1, $2, $3, $4)
                    """, self.world_id, inconsistency["description"], resolution, datetime.now())
        
        # Format the response
        result = f"Resolved {len(resolutions)} inconsistencies in world {self.world_id}:\n\n"
        
        for i, res in enumerate(resolutions, 1):
            result += f"{i}. Issue: {res['issue']['description']}\n"
            result += f"   Resolution: {res['resolution']}\n\n"
        
        return result
    
    async def identify_inconsistencies(self) -> List[Dict[str, Any]]:
        """Identify inconsistencies in world lore."""
        inconsistencies = []
        
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                # Check for timeline inconsistencies
                timeline_issues = await conn.fetch("""
                    WITH event_pairs AS (
                        SELECT e1.id as id1, e2.id as id2, 
                               e1.name as name1, e2.name as name2,
                               e1.date_order as date1, e2.date_order as date2,
                               e1.description as desc1, e2.description as desc2
                        FROM HistoricalEvents e1
                        JOIN HistoricalEvents e2 ON e1.id < e2.id
                        WHERE e1.world_id = $1 AND e2.world_id = $1
                    )
                    SELECT * FROM event_pairs
                    WHERE (date1 > date2 AND date2 > 0)
                       OR (desc1 LIKE '%' || name2 || '%' AND date1 < date2)
                       OR (desc2 LIKE '%' || name1 || '%' AND date2 < date1)
                """, self.world_id)
                
                for issue in timeline_issues:
                    inconsistencies.append({
                        "type": "timeline",
                        "entities": [issue["id1"], issue["id2"]],
                        "description": f"Timeline inconsistency between '{issue['name1']}' and '{issue['name2']}'",
                        "details": {
                            "event1": {
                                "id": issue["id1"],
                                "name": issue["name1"],
                                "date_order": issue["date1"]
                            },
                            "event2": {
                                "id": issue["id2"],
                                "name": issue["name2"],
                                "date_order": issue["date2"]
                            }
                        }
                    })
                
                # Check for faction leadership inconsistencies
                faction_issues = await conn.fetch("""
                    WITH faction_leaders AS (
                        SELECT f.id as faction_id, f.name as faction_name,
                               n1.id as leader_id, n1.name as leader_name
                        FROM Factions f
                        JOIN NotableFigures n1 ON n1.id = ANY(f.leadership)
                        WHERE f.world_id = $1
                    )
                    SELECT fl1.faction_id, fl1.faction_name,
                           fl1.leader_id as leader1_id, fl1.leader_name as leader1_name,
                           fl2.leader_id as leader2_id, fl2.leader_name as leader2_name
                    FROM faction_leaders fl1
                    JOIN faction_leaders fl2 ON fl1.faction_id = fl2.faction_id AND fl1.leader_id < fl2.leader_id
                """, self.world_id)
                
                for issue in faction_issues:
                    inconsistencies.append({
                        "type": "leadership",
                        "entities": [issue["faction_id"], issue["leader1_id"], issue["leader2_id"]],
                        "description": f"Multiple leadership claims for faction '{issue['faction_name']}'",
                        "details": {
                            "faction": {
                                "id": issue["faction_id"],
                                "name": issue["faction_name"]
                            },
                            "leader1": {
                                "id": issue["leader1_id"],
                                "name": issue["leader1_name"]
                            },
                            "leader2": {
                                "id": issue["leader2_id"],
                                "name": issue["leader2_name"]
                            }
                        }
                    })
        
        return inconsistencies
    
    async def resolve_single_inconsistency(self, conn, inconsistency: Dict[str, Any]) -> str:
        """Resolve a single inconsistency and update the database."""
        if inconsistency["type"] == "timeline":
            # Resolve timeline inconsistency
            event1 = inconsistency["details"]["event1"]
            event2 = inconsistency["details"]["event2"]
            
            # Decide which event to adjust
            if event1["date_order"] > event2["date_order"]:
                # Move event1 to before event2
                new_date_order = event2["date_order"] - 1
                await conn.execute("""
                    UPDATE HistoricalEvents 
                    SET date_order = $1
                    WHERE id = $2
                """, new_date_order, event1["id"])
                
                return f"Adjusted date order of '{event1['name']}' to occur before '{event2['name']}'"
            else:
                # Events reference each other but dates are correct
                # Update descriptions to clarify reference
                event1_desc = await conn.fetchval("""
                    SELECT description FROM HistoricalEvents WHERE id = $1
                """, event1["id"])
                
                updated_desc = event1_desc + f"\n\nNote: This event occurred before '{event2['name']}' but references it due to prophecy/legend."
                
                await conn.execute("""
                    UPDATE HistoricalEvents 
                    SET description = $1
                    WHERE id = $2
                """, updated_desc, event1["id"])
                
                return f"Clarified that '{event1['name']}' references future event '{event2['name']}' in its description"
        
        elif inconsistency["type"] == "leadership":
            # Resolve faction leadership inconsistency
            faction = inconsistency["details"]["faction"]
            leader1 = inconsistency["details"]["leader1"]
            leader2 = inconsistency["details"]["leader2"]
            
            # Option 1: Make one leader primary, one advisor
            await conn.execute("""
                UPDATE Factions
                SET leadership = $1,
                    advisors = array_append(advisors, $2)
                WHERE id = $3
            """, [leader1["id"]], leader2["id"], faction["id"])
            
            return f"Resolved by making '{leader1['name']}' the primary leader and '{leader2['name']}' an advisor to faction '{faction['name']}'"
        
        else:
            return f"Unknown inconsistency type: {inconsistency['type']}"

# Create a singleton instance if desired
world_lore_manager = WorldLoreManager(user_id=0, conversation_id=0)
