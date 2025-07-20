# lore/lore_generator.py

"""
Lore Generator Components - Consolidated

This module provides components for generating and evolving lore content,
including dynamic generation, evolution, component generation, and governance integration.
"""

import logging
import json
import asyncio
import os
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime
from dataclasses import dataclass
from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_async_openai_client

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from agents.run_context import RunContextWrapper

from openai import AsyncOpenAI

# Import schemas
from .unified_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

# Nyx governance integration
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting

# Import data access layer
from .data_access import (
    NPCDataAccess,
    LocationDataAccess,
    FactionDataAccess,
    LoreKnowledgeAccess
)
from embedding.vector_store import generate_embedding

# Import error handling
from .error_manager import LoreError, ErrorHandler, handle_errors

logger = logging.getLogger(__name__)
_LORE_GENERATOR_INSTANCES: Dict[Tuple[int, int], "DynamicLoreGenerator"] = {}

#---------------------------
# Core Implementation Functions (not decorated)
#---------------------------

async def _generate_foundation_lore_impl(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Implementation: Generate foundation lore (cosmology, magic system, etc.) for a given environment.
    """
    # Handle context properly - it might be a RunContextWrapper or a dict
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate cohesive foundational world lore for this environment:
    {environment_desc}

    Return as JSON with keys:
    cosmology, magic_system, world_history, calendar_system, social_structure
    """
    
    foundation_lore_agent = get_foundation_lore_agent()
    result = await Runner.run(foundation_lore_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(FoundationLoreOutput)
    return final_output.dict()

async def _generate_factions_impl(ctx, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
    """
    Implementation: Generate 3-5 distinct factions referencing environment_desc + social_structure.
    """
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate 3-5 distinct factions for this environment:
    Environment: {environment_desc}
    Social Structure: {social_structure}
    
    Return JSON as an OBJECT with a "factions" array.
    Example: {{"factions": [{{...}}, {{...}}]}}
    """
    
    factions_agent = get_factions_agent()
    result = await Runner.run(factions_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(FactionsOutput)
    return [f.dict() for f in final_output.factions]

async def _generate_cultural_elements_impl(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Implementation: Generate cultural elements (traditions, taboos, etc.) referencing environment + faction names.
    """
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate 4-7 unique cultural elements for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON as an OBJECT with an "elements" array.
    Example: {{"elements": [{{...}}, {{...}}]}}
    """
    
    cultural_agent = get_cultural_agent()
    result = await Runner.run(cultural_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(CulturalElementsOutput)
    return [c.dict() for c in final_output.elements]

async def _generate_historical_events_impl(ctx, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Implementation: Generate historical events referencing environment, existing world_history, faction_names.
    """
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate 5-7 significant historical events:
    Environment: {environment_desc}
    Existing World History: {world_history}
    Factions: {faction_names}

    Return JSON as an OBJECT with an "events" array.
    Example: {{"events": [{{...}}, {{...}}]}}
    """
    
    history_agent = get_history_agent()
    result = await Runner.run(history_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(HistoricalEventsOutput)
    return [h.dict() for h in final_output.events]

async def _generate_locations_impl(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Implementation: Generate 5-8 significant locations referencing environment_desc + faction names.
    """
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate 5-8 significant locations for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON as an OBJECT with a "locations" array.
    Example: {{"locations": [{{...}}, {{...}}]}}
    """
    
    locations_agent = get_locations_agent()
    result = await Runner.run(locations_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(LocationsOutput)
    return [l.dict() for l in final_output.locations]

async def _generate_quest_hooks_impl(ctx, faction_names: str, location_names: str) -> List[Dict[str, Any]]:
    """
    Implementation: Generate 5-7 quest hooks referencing existing factions, locations, etc.
    """
    if hasattr(ctx, 'context'):
        run_ctx = RunContextWrapper(context=ctx.context)
    elif isinstance(ctx, dict):
        run_ctx = RunContextWrapper(context=ctx)
    else:
        run_ctx = ctx
    
    user_prompt = f"""
    Generate 5-7 engaging quest hooks:
    Factions: {faction_names}
    Locations: {location_names}

    Return JSON as an OBJECT with a "quests" array.
    Example: {{"quests": [{{...}}, {{...}}]}}
    """
    
    quests_agent = get_quests_agent()
    result = await Runner.run(quests_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(QuestsOutput)
    return [q.dict() for q in final_output.quests]

#---------------------------
# Function Tool Definitions with Nyx Governance
#---------------------------

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_foundation_lore",
    action_description="Generating foundation lore for environment: {environment_desc}",
    id_from_context=lambda ctx: f"foundation_lore_{ctx.context.get('conversation_id', 0)}"
)
async def generate_foundation_lore(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Generate foundation lore (cosmology, magic system, etc.) for a given environment.
    
    Args:
        environment_desc: Environment description
    """
    return await _generate_foundation_lore_impl(ctx, environment_desc)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_factions",
    action_description="Generating factions for environment: {environment_desc}",
    id_from_context=lambda ctx: f"factions_{ctx.context.get('conversation_id', 0)}"
)
async def generate_factions(ctx, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 distinct factions referencing environment_desc + social_structure.
    
    Args:
        environment_desc: Environment description
        social_structure: Social structure description
    """
    return await _generate_factions_impl(ctx, environment_desc, social_structure)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_cultural_elements",
    action_description="Generating cultural elements for environment: {environment_desc}",
    id_from_context=lambda ctx: f"cultural_{ctx.context.get('conversation_id', 0)}"
)
async def generate_cultural_elements(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate cultural elements (traditions, taboos, etc.) referencing environment + faction names.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    return await _generate_cultural_elements_impl(ctx, environment_desc, faction_names)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_historical_events",
    action_description="Generating historical events for environment: {environment_desc}",
    id_from_context=lambda ctx: f"history_{ctx.context.get('conversation_id', 0)}"
)
async def generate_historical_events(ctx, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate historical events referencing environment, existing world_history, faction_names.
    
    Args:
        environment_desc: Environment description
        world_history: Existing world history
        faction_names: Comma-separated faction names
    """
    return await _generate_historical_events_impl(ctx, environment_desc, world_history, faction_names)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_locations",
    action_description="Generating locations for environment: {environment_desc}",
    id_from_context=lambda ctx: f"locations_{ctx.context.get('conversation_id', 0)}"
)
async def generate_locations(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-8 significant locations referencing environment_desc + faction names.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    return await _generate_locations_impl(ctx, environment_desc, faction_names)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_quest_hooks",
    action_description="Generating quest hooks for factions and locations",
    id_from_context=lambda ctx: f"quests_{ctx.context.get('conversation_id', 0)}"
)
async def generate_quest_hooks(ctx, faction_names: str, location_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-7 quest hooks referencing existing factions, locations, etc.
    
    Args:
        faction_names: Comma-separated faction names
        location_names: Comma-separated location names
    """
    return await _generate_quest_hooks_impl(ctx, faction_names, location_names)

#---------------------------
# Component Generator Base Classes
#---------------------------

@dataclass
class ComponentConfig:
    """Configuration for component generation"""
    min_length: int = 100
    max_length: int = 500
    style: str = "descriptive"
    tone: str = "neutral"
    include_metadata: bool = True

class BaseGenerator:
    """Base class for all generator components."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None):
        """
        Initialize the base generator component.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
            governor: Optional pre-initialized governor to avoid circular deps
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor  # Accept injected governor
        self.initialized = False
        self._cache = {}
        
        # Initialize data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
    
    async def initialize(self) -> bool:
        """
        Initialize the generator component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            # Only get governance if not already provided
            if self.governor is None:
                from nyx.integrate import get_central_governance
                self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize data access components
            await self.npc_data.initialize()
            await self.location_data.initialize()
            await self.faction_data.initialize()
            await self.lore_knowledge.initialize()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            return False
    
    async def initialize_governance(self) -> bool:
        """Initialize governance connection."""
        try:
            from nyx.integrate import get_central_governance
            
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            return True
        except Exception as e:
            logger.error(f"Error initializing governance for {self.__class__.__name__}: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Cleanup data access components
            await self.npc_data.cleanup()
            await self.location_data.cleanup()
            await self.faction_data.cleanup()
            await self.lore_knowledge.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached component if available"""
        return self._cache.get(key)
    
    def _cache_component(self, key: str, component: Dict[str, Any]):
        """Cache a generated component"""
        self._cache[key] = component

class ComponentGenerator(BaseGenerator):
    """Base class for all component generators"""
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, 
                 config: Optional[ComponentConfig] = None):
        super().__init__(user_id, conversation_id)
        self.config = config or ComponentConfig()
    
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a component with the given context"""
        raise NotImplementedError

class CharacterGenerator(ComponentGenerator):
    """Generator for character components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"character_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "character",
            "name": context.get("name", "Unknown Character"),
            "description": await self._generate_description(context),
            "traits": await self._generate_traits(context),
            "background": await self._generate_background(context),
            "relationships": await self._generate_relationships(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a character."""
        try:
            # Extract relevant context
            name = context.get("name", "Unknown Character")
            role = context.get("role", "Unknown Role")
            background = context.get("background", {})
            
            # Build description components
            run_ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            
            # Sample implementation - would use more sophisticated generation in practice
            description = f"{name} is a {role}. "
            if "origin" in background:
                description += f"They come from {background['origin']}. "
            if "appearance" in context:
                description += context["appearance"]
            
            return description
        except Exception as e:
            logger.error(f"Error generating character description: {str(e)}")
            return f"Description for {context.get('name', 'Unknown Character')}"
    
    async def _generate_traits(self, context: Dict[str, Any]) -> List[str]:
        """Generate character traits."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_traits", ["Intelligent", "Resourceful", "Cautious"])
        except Exception as e:
            logger.error(f"Error generating character traits: {str(e)}")
            return ["Resourceful", "Adaptable"]
    
    async def _generate_background(self, context: Dict[str, Any]) -> str:
        """Generate character background."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_background", "A mysterious past shrouded in secrecy.")
        except Exception as e:
            logger.error(f"Error generating character background: {str(e)}")
            return "Background unknown."
    
    async def _generate_relationships(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate character relationships."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_relationships", [])
        except Exception as e:
            logger.error(f"Error generating character relationships: {str(e)}")
            return []

class LocationGenerator(ComponentGenerator):
    """Generator for location components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"location_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "location",
            "name": context.get("name", "Unknown Location"),
            "description": await self._generate_description(context),
            "climate": await self._generate_climate(context),
            "geography": await self._generate_geography(context),
            "culture": await self._generate_culture(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a location."""
        # Sample implementation
        try:
            name = context.get("name", "Unknown Location")
            location_type = context.get("type", "area")
            
            description = f"{name} is a {location_type}. "
            if "features" in context:
                description += f"It features {', '.join(context['features'])}. "
            if "atmosphere" in context:
                description += context["atmosphere"]
                
            return description
        except Exception as e:
            logger.error(f"Error generating location description: {str(e)}")
            return f"Description of {context.get('name', 'Unknown Location')}"
    
    async def _generate_climate(self, context: Dict[str, Any]) -> str:
        """Generate climate information."""
        # Sample implementation
        try:
            return context.get("climate", "Temperate")
        except Exception as e:
            logger.error(f"Error generating climate: {str(e)}")
            return "Temperate"
    
    async def _generate_geography(self, context: Dict[str, Any]) -> str:
        """Generate geographical information."""
        # Sample implementation
        try:
            return context.get("geography", "Varied terrain")
        except Exception as e:
            logger.error(f"Error generating geography: {str(e)}")
            return "Varied terrain"
    
    async def _generate_culture(self, context: Dict[str, Any]) -> str:
        """Generate cultural information."""
        # Sample implementation
        try:
            return context.get("culture", "Diverse")
        except Exception as e:
            logger.error(f"Error generating culture: {str(e)}")
            return "Diverse"

class EventGenerator(ComponentGenerator):
    """Generator for event components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"event_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "event",
            "name": context.get("name", "Unknown Event"),
            "description": await self._generate_description(context),
            "date": await self._generate_date(context),
            "participants": await self._generate_participants(context),
            "consequences": await self._generate_consequences(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for an event."""
        # Sample implementation
        try:
            name = context.get("name", "Unknown Event")
            event_type = context.get("type", "occurrence")
            
            description = f"{name} was a significant {event_type}. "
            if "details" in context:
                if "summary" in context["details"]:
                    description += context["details"]["summary"]
                
            return description
        except Exception as e:
            logger.error(f"Error generating event description: {str(e)}")
            return f"Description of {context.get('name', 'Unknown Event')}"
    
    async def _generate_date(self, context: Dict[str, Any]) -> str:
        """Generate date information."""
        # Sample implementation
        try:
            year = context.get("year")
            month = context.get("month")
            day = context.get("day")
            
            if all([year, month, day]):
                return f"{year}-{month:02d}-{day:02d}"
            elif year:
                return f"Year {year}"
            else:
                return "Unknown date"
        except Exception as e:
            logger.error(f"Error generating event date: {str(e)}")
            return "Unknown date"
    
    async def _generate_participants(self, context: Dict[str, Any]) -> List[str]:
        """Generate participant information."""
        # Sample implementation
        try:
            return context.get("participants", [])
        except Exception as e:
            logger.error(f"Error generating event participants: {str(e)}")
            return []
    
    async def _generate_consequences(self, context: Dict[str, Any]) -> List[str]:
        """Generate consequence information."""
        # Sample implementation
        try:
            return context.get("consequences", [])
        except Exception as e:
            logger.error(f"Error generating event consequences: {str(e)}")
            return []

class ComponentGeneratorFactory:
    """Factory for creating component generators"""
    @staticmethod
    def create_generator(component_type: str, user_id: Optional[int] = None, 
                         conversation_id: Optional[int] = None,
                         config: Optional[ComponentConfig] = None) -> ComponentGenerator:
        """Create a component generator of the specified type."""
        generators = {
            "character": CharacterGenerator,
            "location": LocationGenerator,
            "event": EventGenerator
        }
        
        generator_class = generators.get(component_type.lower())
        if not generator_class:
            raise ValueError(f"Unknown component type: {component_type}")
            
        return generator_class(user_id, conversation_id, config)

#---------------------------
# World and Setting Generators
#---------------------------

class WorldBuilder(BaseGenerator):
    """Generates foundation world lore."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None):
        """
        Initialize the world builder with optional governor.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            governor: Optional pre-initialized governor to avoid circular deps
        """
        super().__init__(user_id, conversation_id, governor)
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore (cosmology, magic system, world history, etc.)
        
        Args:
            environment_desc: Short textual description of the environment
            
        Returns:
            Dict containing the foundation lore fields
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="initialize_world_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"World lore initialization not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
            
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Call the implementation function instead of the function tool
        foundation_data = await _generate_foundation_lore_impl(run_ctx, environment_desc)

        # Store in database
        for category, desc in foundation_data.items():
            # e.g. "category" might be "cosmology", "magic_system", ...
            await self._store_world_lore(
                name=f"{category.title()} of {await self.get_setting_name()}",
                category=category,
                description=desc,
                significance=8,
                tags=[category, "foundation", "world_building"]
            )

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "initialize_world_lore",
                "description": f"Generated foundation lore for {environment_desc[:50]}"
            },
            result={
                "categories": list(foundation_data.keys()),
                "world_name": await self.get_setting_name()
            }
        )

        return foundation_data
    
    async def _store_world_lore(self, name: str, category: str, 
                              description: str, significance: int,
                              tags: List[str]) -> int:
        """
        Store world lore in the database.
        """
        try:
            query = """
                INSERT INTO WorldLore (
                    user_id, conversation_id, name, category,
                    description, significance, tags, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id
            """
            
            # tags should be passed as a list directly for TEXT[] column
            async with get_db_connection_context() as conn:
                lore_id = await conn.fetchval(
                    query,
                    self.user_id,
                    self.conversation_id,
                    name,
                    category,
                    description,
                    significance,
                    tags  # Pass as list, not JSON string
                )
                
                return lore_id
                
        except Exception as e:
            logger.error(f"Error storing world lore: {e}")
            return 0
    
    async def get_setting_name(self) -> str:
        """
        Get the current setting name.
        
        Returns:
            Setting name
        """
        try:
            query = """
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
                LIMIT 1
            """
            
            async with get_db_connection_context() as conn:
                    setting_name = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id
                    )
                    
                    if setting_name:
                        return setting_name
                    else:
                        return "The Setting"
                    
        except Exception as e:
            logger.error(f"Error getting setting name: {e}")
            return "The Setting"

class FactionGenerator(BaseGenerator):
    """Generates faction and related lore content."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None):
        """
        Initialize the faction generator with optional governor.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            governor: Optional pre-initialized governor to avoid circular deps
        """
        super().__init__(user_id, conversation_id, governor)
    
    async def generate_factions(self, environment_desc: str, 
                              world_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3-5 distinct factions referencing the environment description.
        
        Args:
            environment_desc: Text describing environment or setting
            world_lore: The dictionary from initialize_world_lore
            
        Returns:
            A list of faction dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_factions",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Faction generation not approved: {permission.get('reasoning')}")
            return []
            
        # Typically we want the 'social_structure' from foundation_data
        social_structure = world_lore.get("social_structure", "")

        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Call the implementation function
        factions_data = await _generate_factions_impl(run_ctx, environment_desc, social_structure)

        # Store each in the DB
        for faction in factions_data:
            try:
                await self._store_faction(faction)
            except Exception as e:
                logging.error(f"Error storing faction '{faction['name']}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_factions",
                "description": f"Generated {len(factions_data)} factions"
            },
            result={
                "faction_count": len(factions_data),
                "faction_names": [f.get("name", "Unknown") for f in factions_data]
            }
        )

        return factions_data
    
    async def generate_cultural_elements(self, environment_desc: str, 
                                      factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate cultural elements referencing environment + the names of the existing factions.
        
        Args:
            environment_desc: Text describing environment
            factions: List of faction dictionaries
            
        Returns:
            List of cultural element dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_cultural_elements",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Cultural elements generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name", "Unknown") for f in factions])

        # Call the implementation function
        cultural_data = await _generate_cultural_elements_impl(run_ctx, environment_desc, faction_names)

        # Store them
        for element in cultural_data:
            try:
                await self._store_cultural_element(element)
            except Exception as e:
                logging.error(f"Error storing cultural element '{element.get('name','unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_cultural_elements",
                "description": f"Generated {len(cultural_data)} cultural elements"
            },
            result={
                "element_count": len(cultural_data),
                "element_types": list(set([e.get("type", "unknown") for e in cultural_data]))
            }
        )

        return cultural_data
    
    async def generate_historical_events(self, environment_desc: str, 
                                      foundation_data: Dict[str, Any], 
                                      factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 major historical events referencing environment + existing 'world_history' + faction names.
        
        Args:
            environment_desc: Environment description text
            foundation_data: Foundation lore dictionary
            factions: List of faction dictionaries
            
        Returns:
            List of historical event dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_historical_events",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Historical events generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # We can feed in the previously generated 'world_history'
        world_history = foundation_data.get("world_history", "")
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        # Call the implementation function
        events_data = await _generate_historical_events_impl(run_ctx, environment_desc, world_history, faction_names)

        # Then store them
        for event in events_data:
            try:
                await self._store_historical_event(event)
            except Exception as e:
                logging.error(f"Error storing historical event '{event.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_historical_events",
                "description": f"Generated {len(events_data)} historical events"
            },
            result={
                "event_count": len(events_data),
                "significant_events": [e.get("name") for e in events_data if e.get("significance", 0) > 7]
            }
        )

        return events_data
    
    async def generate_locations(self, environment_desc: str, 
                              factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-8 significant locations referencing environment + faction names.
        
        Args:
            environment_desc: Environment description text
            factions: List of faction dictionaries
            
        Returns:
            List of location dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_locations",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Locations generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        # Call the implementation function
        locations_data = await _generate_locations_impl(run_ctx, environment_desc, faction_names)

        # Store each location
        for loc in locations_data:
            try:
                # Create the location record
                location_id = await self._store_location(loc)

                # Add location lore
                controlling_faction = loc.get("controlling_faction")
                hidden_secrets = loc.get("hidden_secrets", [])
                founding_story = f"Founded as a {loc['type']}."

                await self._store_location_lore(
                    location_id=location_id,
                    founding_story=founding_story,
                    hidden_secrets=hidden_secrets,
                    local_legends=[],
                    historical_significance=loc.get("strategic_importance", "")
                )

                # Record controlling_faction if needed
                if controlling_faction:
                    await self._connect_faction_to_location(location_id, controlling_faction)
            except Exception as e:
                logging.error(f"Error storing location '{loc.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_locations",
                "description": f"Generated {len(locations_data)} locations"
            },
            result={
                "location_count": len(locations_data),
                "location_types": list(set([l.get("type", "unknown") for l in locations_data]))
            }
        )

        return locations_data
    
    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], 
                                 locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 quest hooks referencing existing factions + location names.
        
        Args:
            factions: List of faction dictionaries
            locations: List of location dictionaries
            
        Returns:
            List of quest hook dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_quest_hooks",
            action_details={"faction_count": len(factions), "location_count": len(locations)}
        )
        
        if not permission["approved"]:
            logging.warning(f"Quest hooks generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])
        location_names = ", ".join([l.get("name","Unknown") for l in locations])

        # Call the implementation function
        quests_data = await _generate_quest_hooks_impl(run_ctx, faction_names, location_names)

        # Store them
        for quest in quests_data:
            try:
                await self._store_quest(quest)
            except Exception as e:
                logging.error(f"Error storing quest hook '{quest.get('quest_name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_quest_hooks",
                "description": f"Generated {len(quests_data)} quest hooks"
            },
            result={
                "quest_count": len(quests_data),
                "quest_difficulties": list(set([q.get("difficulty", 0) for q in quests_data]))
            }
        )

        return quests_data
    
    # Database storage methods - implementing based on schema and patterns from other modules
    async def _store_faction(self, faction_data: Dict[str, Any]) -> int:
        """Store a faction in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Check if faction already exists
                existing = await conn.fetchval("""
                    SELECT id FROM Factions 
                    WHERE user_id = $1 AND conversation_id = $2 AND name = $3
                """, self.user_id, self.conversation_id, faction_data.get('name'))
                
                if existing:
                    logger.info(f"Faction '{faction_data['name']}' already exists with id {existing}")
                    return existing
                
                # Generate embedding for faction
                embedding_text = f"{faction_data['name']} {faction_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Convert lists to JSON strings for JSONB columns
                values_json = json.dumps(faction_data.get('values', []))
                goals_json = json.dumps(faction_data.get('goals', []))
                resources_json = json.dumps(faction_data.get('resources', []))
                membership_requirements_json = json.dumps(faction_data.get('membership_requirements', []))
                secret_activities_json = json.dumps(faction_data.get('secret_activities', []))
                recruitment_methods_json = json.dumps(faction_data.get('recruitment_methods', []))
                
                # Insert faction
                faction_id = await conn.fetchval("""
                    INSERT INTO Factions (
                        user_id, conversation_id, name, type, description,
                        values, goals, hierarchy_type, resources, territory,
                        meeting_schedule, membership_requirements, 
                        public_reputation, secret_activities, power_level,
                        influence_scope, recruitment_methods, leadership_structure,
                        founding_story, embedding, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                        $13, $14, $15, $16, $17, $18, $19, $20, NOW()
                    ) RETURNING id
                """, 
                    self.user_id, self.conversation_id,
                    faction_data.get('name'),
                    faction_data.get('type', 'organization'),
                    faction_data.get('description'),
                    values_json,  # Pass as JSON string
                    goals_json,   # Pass as JSON string
                    faction_data.get('hierarchy_type', 'formal'),
                    resources_json,  # Pass as JSON string
                    faction_data.get('headquarters'),  # Using headquarters as territory
                    faction_data.get('meeting_schedule'),
                    membership_requirements_json,  # Pass as JSON string
                    faction_data.get('public_reputation', 'neutral'),
                    secret_activities_json,  # Pass as JSON string
                    faction_data.get('power_level', 5),
                    faction_data.get('influence_scope', 'local'),
                    recruitment_methods_json,  # Pass as JSON string
                    json.dumps(faction_data.get('leadership_structure', {})),  # Already JSONB
                    faction_data.get('founding_story', f"Founded as a {faction_data.get('type', 'organization')}."),
                    embedding
                )
                
                # Handle allies and rivals relationships
                if faction_data.get('allies'):
                    allies_json = json.dumps(faction_data['allies'])
                    await conn.execute("""
                        UPDATE Factions SET allies = $1 WHERE id = $2
                    """, allies_json, faction_id)  # Pass as JSON string
                
                if faction_data.get('rivals'):
                    rivals_json = json.dumps(faction_data['rivals'])
                    await conn.execute("""
                        UPDATE Factions SET rivals = $1 WHERE id = $2
                    """, rivals_json, faction_id)  # Pass as JSON string
                
                logger.info(f"Stored faction '{faction_data['name']}' with id {faction_id}")
                return faction_id
                
        except Exception as e:
            logger.error(f"Error storing faction: {e}")
            return 0
        
    async def _store_cultural_element(self, element_data: Dict[str, Any]) -> int:
        """Store a cultural element in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Generate embedding
                embedding_text = f"{element_data['name']} {element_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Ensure practiced_by is properly formatted
                practiced_by = element_data.get('practiced_by', [])
                if isinstance(practiced_by, str):
                    practiced_by = [practiced_by]
                
                # Convert list to JSON string for JSONB column
                practiced_by_json = json.dumps(practiced_by)
                
                # Insert cultural element
                element_id = await conn.fetchval("""
                    INSERT INTO CulturalElements (
                        user_id, conversation_id, name, element_type,
                        description, practiced_by, significance,
                        historical_origin, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                    self.user_id, self.conversation_id,
                    element_data.get('name'),
                    element_data.get('type', 'tradition'),
                    element_data.get('description'),
                    practiced_by_json,  # Pass as JSON string
                    element_data.get('significance', 5),
                    element_data.get('historical_origin', ''),
                    embedding
                )
                
                logger.info(f"Stored cultural element '{element_data['name']}' with id {element_id}")
                return element_id
                
        except Exception as e:
            logger.error(f"Error storing cultural element: {e}")
            return 0
    
    async def _store_historical_event(self, event_data: Dict[str, Any]) -> int:
        """Store a historical event in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Generate embedding
                embedding_text = f"{event_data['name']} {event_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Extract participating factions and convert to JSON
                participating_factions = event_data.get('participating_factions', [])
                involved_entities_json = json.dumps(participating_factions)
                
                # Convert other lists to JSON strings
                consequences_json = json.dumps(event_data.get('consequences', []))
                disputed_facts_json = json.dumps(event_data.get('disputed_facts', []))
                commemorations_json = json.dumps(event_data.get('commemorations', []))
                primary_sources_json = json.dumps(event_data.get('primary_sources', []))
                
                # Insert historical event
                event_id = await conn.fetchval("""
                    INSERT INTO HistoricalEvents (
                        user_id, conversation_id, name, description,
                        date_description, event_type, significance,
                        involved_entities, location, consequences,
                        cultural_impact, disputed_facts, commemorations,
                        primary_sources, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING id
                """,
                    self.user_id, self.conversation_id,
                    event_data.get('name'),
                    event_data.get('description'),
                    event_data.get('date_description', 'Unknown date'),
                    event_data.get('event_type', 'political'),
                    event_data.get('significance', 5),
                    involved_entities_json,  # Pass as JSON string
                    event_data.get('location'),
                    consequences_json,  # Pass as JSON string
                    event_data.get('cultural_impact', 'moderate'),
                    disputed_facts_json,  # Pass as JSON string
                    commemorations_json,  # Pass as JSON string
                    primary_sources_json,  # Pass as JSON string
                    embedding
                )
                
                logger.info(f"Stored historical event '{event_data['name']}' with id {event_id}")
                return event_id
                
        except Exception as e:
            logger.error(f"Error storing historical event: {e}")
            return 0
        
    async def _store_location(self, location_data: Dict[str, Any]) -> int:
        """Store a location in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Generate embedding
                embedding_text = f"{location_data['name']} {location_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Parse strategic_importance to ensure it's an integer
                strategic_value = location_data.get('strategic_importance', 5)
                if isinstance(strategic_value, str):
                    # Try to extract a number from the string or use default
                    try:
                        # Look for a number in the string
                        import re
                        numbers = re.findall(r'\d+', strategic_value)
                        if numbers:
                            strategic_value = int(numbers[0])
                            # Ensure it's within valid range (1-10)
                            strategic_value = max(1, min(10, strategic_value))
                        else:
                            # Default to 5 if no number found
                            strategic_value = 5
                    except:
                        strategic_value = 5
                elif not isinstance(strategic_value, int):
                    strategic_value = 5
                
                # Insert location
                location_id = await conn.fetchval("""
                    INSERT INTO Locations (
                        user_id, conversation_id, location_name, description,
                        location_type, parent_location, cultural_significance,
                        economic_importance, strategic_value, population_density,
                        notable_features, hidden_aspects, access_restrictions,
                        local_customs, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING id
                """,
                    self.user_id, self.conversation_id,
                    location_data.get('name'),
                    location_data.get('description'),
                    location_data.get('type', 'settlement'),
                    location_data.get('parent_location'),
                    location_data.get('cultural_significance', 'moderate'),
                    location_data.get('economic_importance', 'moderate'),
                    strategic_value,  # Now guaranteed to be an integer
                    location_data.get('population_density', 'moderate'),
                    location_data.get('notable_features', []),
                    location_data.get('hidden_secrets', []),  # mapped to hidden_aspects
                    location_data.get('access_restrictions', []),
                    location_data.get('local_customs', []),
                    embedding
                )
                    
                logger.info(f"Stored location '{location_data['name']}' with id {location_id}")
                return location_id
                    
        except Exception as e:
            logger.error(f"Error storing location: {e}")
            return 0
    
    async def _store_location_lore(self, location_id: int, founding_story: str,
                                  hidden_secrets: List[str], local_legends: List[str],
                                  historical_significance: str) -> int:
        """Store location lore in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Check if we have an existing LocationLore table or use LocalHistories
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'locationlore'
                    );
                """)
                
                if table_exists:
                    # Use LocationLore table if it exists
                    # Convert lists to JSON strings
                    hidden_secrets_json = json.dumps(hidden_secrets)
                    local_legends_json = json.dumps(local_legends)
                    
                    lore_id = await conn.fetchval("""
                        INSERT INTO LocationLore (
                            user_id, conversation_id, location_id,
                            founding_story, hidden_secrets, local_legends,
                            historical_significance
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                    """,
                        self.user_id, self.conversation_id, location_id,
                        founding_story, hidden_secrets_json, local_legends_json,
                        historical_significance
                    )
                else:
                    # Use LocalHistories table as fallback
                    # Generate embedding for the history
                    embedding_text = f"{founding_story} {historical_significance}"
                    embedding = await generate_embedding(embedding_text)
                    
                    # Convert lists to JSON strings for JSONB columns
                    connected_myths_json = json.dumps(local_legends)
                    related_landmarks_json = json.dumps([])
                    
                    lore_id = await conn.fetchval("""
                        INSERT INTO LocalHistories (
                            user_id, conversation_id, location_id,
                            event_name, description, date_description,
                            significance, impact_type, connected_myths,
                            related_landmarks, narrative_category, embedding
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                    """,
                        self.user_id, self.conversation_id, location_id,
                        "Founding Story",
                        founding_story,
                        "At the founding",
                        8,  # High significance for founding story
                        "foundational",
                        connected_myths_json,  # Pass as JSON string
                        related_landmarks_json,  # Pass as JSON string
                        "origin",
                        embedding
                    )
                    
                    # Store hidden secrets as separate entries if they exist
                    for secret in hidden_secrets:
                        if secret:
                            secret_embedding = await generate_embedding(secret)
                            await conn.execute("""
                                INSERT INTO LocalHistories (
                                    user_id, conversation_id, location_id,
                                    event_name, description, date_description,
                                    significance, impact_type, narrative_category, embedding
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            """,
                                self.user_id, self.conversation_id, location_id,
                                "Hidden Secret",
                                secret,
                                "Unknown",
                                7,  # High significance for secrets
                                "secret",
                                "mystery",
                                secret_embedding
                            )
                    
                    logger.info(f"Stored location lore for location {location_id}")
                    return lore_id
                    
        except Exception as e:
            logger.error(f"Error storing location lore: {e}")
            return 0
        
    async def _connect_faction_to_location(self, location_id: int, faction_name: str) -> bool:
        """Connect a faction to a location in the database."""
        try:
            async with get_db_connection_context() as conn:
                    # Find the faction by name
                    faction_id = await conn.fetchval("""
                        SELECT id FROM Factions 
                        WHERE user_id = $1 AND conversation_id = $2 AND name = $3
                    """, self.user_id, self.conversation_id, faction_name)
                    
                    if not faction_id:
                        logger.warning(f"Faction '{faction_name}' not found")
                        return False
                    
                    # Update the location's controlling faction
                    # Note: The schema shows 'controlling_faction' as TEXT in Locations table
                    await conn.execute("""
                        UPDATE Locations 
                        SET controlling_faction = $1
                        WHERE id = $2 AND user_id = $3 AND conversation_id = $4
                    """, faction_name, location_id, self.user_id, self.conversation_id)
                    
                    # Also update the faction's territory if needed
                    # Get current territory
                    current_territory = await conn.fetchval("""
                        SELECT territory FROM Factions WHERE id = $1
                    """, faction_id)
                    
                    # Get location name
                    location_name = await conn.fetchval("""
                        SELECT location_name FROM Locations WHERE id = $1
                    """, location_id)
                    
                    if location_name:
                        # Update faction territory to include this location
                        new_territory = f"{current_territory}, {location_name}" if current_territory else location_name
                        await conn.execute("""
                            UPDATE Factions SET territory = $1 WHERE id = $2
                        """, new_territory, faction_id)
                    
                    logger.info(f"Connected faction '{faction_name}' to location {location_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error connecting faction to location: {e}")
            return False
        
    async def _store_quest(self, quest_data: Dict[str, Any]) -> int:
        """Store a quest in the database."""
        try:
            async with get_db_connection_context() as conn:
                # Generate embedding
                quest_description = quest_data.get('description', '')
                objectives_text = ' '.join(quest_data.get('objectives', []))
                embedding_text = f"{quest_data['quest_name']} {quest_description} {objectives_text}"
                embedding = await generate_embedding(embedding_text)
                
                # Handle rewards - convert list to JSON string
                rewards = quest_data.get('rewards', ['Unknown rewards'])
                if isinstance(rewards, list):
                    rewards_json = json.dumps(rewards)
                else:
                    rewards_json = json.dumps([rewards])  # Wrap single string in list
                
                # Insert quest
                quest_id = await conn.fetchval("""
                    INSERT INTO Quests (
                        user_id, conversation_id, quest_name,
                        status, progress_detail, quest_giver,
                        reward, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING quest_id
                """,
                    self.user_id, self.conversation_id,
                    quest_data.get('quest_name'),
                    'Available',  # Default status
                    json.dumps({
                        'description': quest_data.get('description', ''),
                        'objectives': quest_data.get('objectives', []),
                        'location': quest_data.get('location', ''),
                        'difficulty': quest_data.get('difficulty', 5),
                        'lore_significance': quest_data.get('lore_significance', '')
                    }),
                    quest_data.get('quest_giver'),
                    rewards_json,  # Use the JSON-serialized rewards
                    embedding
                )
                
                logger.info(f"Stored quest '{quest_data['quest_name']}' with id {quest_id}")
                return quest_id
                
        except Exception as e:
            logger.error(f"Error storing quest: {e}")
            return 0
        
class LoreEvolution(BaseGenerator):
    """Handles lore evolution over time."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None):
        """
        Initialize the lore evolution component with optional governor.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            governor: Optional pre-initialized governor to avoid circular deps
        """
        super().__init__(user_id, conversation_id, governor)
        self.active_triggers = set()
        self.evolution_history = []
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Process any existing directives before proceeding
        await self._check_and_process_directives()
        
        # Evolve lore using the implemented evolution logic
        evolved_lore = await self._evolve_lore(
            {"event": event_description},
            {"type": "event_evolution"}
        )
        
        result = {
            "updated": True,
            "event": event_description,
            "affected_elements": evolved_lore.get("affected_elements", []),
            "evolution_history": evolved_lore.get("evolution_history", {}),
            "future_implications": evolved_lore.get("future_implications", {})
        }
        
        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result=result
        )
        
        return result
    
    async def _check_and_process_directives(self):
        """Check for and process any pending directives from Nyx."""
        # Get directives for lore generator
        directives = await self.governor.get_agent_directives(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator"
        )
        
        for directive in directives:
            directive_type = directive.get("type")
            
            # Process prohibition directives
            if directive_type == DirectiveType.PROHIBITION:
                prohibited_actions = directive.get("prohibited_actions", [])
                logging.info(f"Found prohibition directive: {prohibited_actions}")
                
                # Store prohibited actions (will be checked during permission checks)
                if not hasattr(self, 'prohibited_lore_actions'):
                    self.prohibited_lore_actions = []
                
                self.prohibited_lore_actions.extend(prohibited_actions)
            
            # Process action directives
            elif directive_type == DirectiveType.ACTION:
                instruction = directive.get("instruction", "")
                logging.info(f"Processing action directive: {instruction}")
                
                # Implement action directive processing as needed
                # We'll report back that we've processed it
                await self.governor.process_agent_action_report(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    agent_id="lore_generator",
                    action={
                        "type": "process_directive",
                        "description": f"Processed directive: {instruction[:50]}"
                    },
                    result={
                        "directive_id": directive.get("id"),
                        "processed": True
                    }
                )
    
    async def _evolve_lore(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evolve lore based on context and world state.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            Evolved lore
        """
        try:
            # Analyze potential evolution triggers
            triggers = await self._analyze_evolution_triggers(lore, context)
            
            if not triggers:
                logger.info("No evolution triggers found")
                return lore
                
            # Generate evolution plan
            evolution_plan = await self._generate_evolution_plan(
                lore,
                triggers,
                context
            )
            
            # Apply evolution
            evolved_lore = await self._apply_evolution(
                lore,
                evolution_plan,
                context
            )
            
            # Validate evolved lore
            validated_lore = await self._validate_evolution(
                evolved_lore,
                lore,
                context
            )
            
            # Enhance evolved lore
            enhanced_lore = await self._enhance_evolution(
                validated_lore,
                lore,
                context
            )
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to evolve lore: {str(e)}")
            raise
    
    async def _analyze_evolution_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze potential evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of identified triggers
        """
        try:
            triggers = []
            
            # Check event-based triggers
            event_triggers = await self._check_event_triggers(lore, context)
            triggers.extend(event_triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to analyze evolution triggers: {str(e)}")
            raise
    
    async def _check_event_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for event-based evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of event-based triggers
        """
        triggers = []
        
        # If we have an event context, create a trigger
        if "event" in context:
            triggers.append({
                "id": f"event_{datetime.now().timestamp()}",
                "type": "event",
                "description": context["event"],
                "priority": "high",
                "impact": random.uniform(0.5, 0.9)
            })
        
        return triggers
    
    async def _generate_evolution_plan(
        self,
        lore: Dict[str, Any],
        triggers: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evolution plan.
        
        Args:
            lore: Current lore state
            triggers: Identified triggers
            context: Evolution context
            
        Returns:
            Evolution plan
        """
        # Mock implementation - would be replaced with actual logic
        return {
            "steps": [
                {
                    "id": "step_1",
                    "description": "Update affected lore elements",
                    "dependencies": []
                },
                {
                    "id": "step_2",
                    "description": "Create new elements if needed",
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_3",
                    "description": "Update relationships",
                    "dependencies": ["step_1", "step_2"]
                }
            ],
            "timeline": ["step_1", "step_2", "step_3"],
            "impact_analysis": {}
        }
    
    async def _apply_evolution(
        self,
        lore: Dict[str, Any],
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply evolution plan to lore.
        
        Args:
            lore: Current lore state
            plan: Evolution plan
            context: Evolution context
            
        Returns:
            Evolved lore
        """
        # Mock implementation - would be replaced with actual logic
        # For now, just return the original lore with an added history entry
        evolved_lore = lore.copy()
        
        evolved_lore["affected_elements"] = ["Element 1", "Element 2"]
        evolved_lore["evolution_history"] = {
            "timestamp": datetime.now().isoformat(),
            "event": context.get("event", "Unknown event"),
            "impact": "moderate"
        }
        evolved_lore["future_implications"] = {
            "short_term": "Some immediate effects on local population",
            "long_term": "Potential shift in faction dynamics"
        }
        
        return evolved_lore
    
    async def _validate_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate evolved lore.
        
        Args:
            evolved_lore: Evolved lore state
            original_lore: Original lore state
            context: Evolution context
            
        Returns:
            Validated lore
        """
        # Mock implementation - would be replaced with actual logic
        return evolved_lore
    
    async def _enhance_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance evolved lore with additional information.
        
        Args:
            evolved_lore: Evolved lore state
            original_lore: Original lore state
            context: Evolution context
            
        Returns:
            Enhanced lore
        """
        # Mock implementation - would be replaced with actual logic
        return evolved_lore

class DynamicLoreGenerator(BaseGenerator):
    """
    Main lore generation coordinator.
    """
    
    @classmethod
    def get_instance(cls, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None) -> "DynamicLoreGenerator":
        """Get or create a singleton instance for the given user/conversation."""
        key = (user_id or 0, conversation_id or 0)
        
        if key not in _LORE_GENERATOR_INSTANCES:
            _LORE_GENERATOR_INSTANCES[key] = cls(user_id, conversation_id, governor)
        else:
            # Update governor if provided and not already set
            instance = _LORE_GENERATOR_INSTANCES[key]
            if governor and instance.governor is None:
                instance.governor = governor
                
        return _LORE_GENERATOR_INSTANCES[key]
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, governor=None):
        """Initialize the dynamic lore generator."""
        super().__init__(user_id, conversation_id, governor)
        self.world_builder = None
        self.faction_generator = None
        self.lore_evolution = None
        self.error_handler = None
    
    async def initialize(self) -> bool:
        """Initialize the dynamic lore generator."""
        if not await super().initialize():
            return False
            
        try:
            # Initialize specialized components with the same governor
            self.world_builder = WorldBuilder(self.user_id, self.conversation_id, self.governor)
            await self.world_builder.initialize()
            
            self.faction_generator = FactionGenerator(self.user_id, self.conversation_id, self.governor)
            await self.faction_generator.initialize()
            
            self.lore_evolution = LoreEvolution(self.user_id, self.conversation_id, self.governor)
            await self.lore_evolution.initialize()
            
            # Initialize error handler
            self.error_handler = ErrorHandler(self.user_id, self.conversation_id)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing DynamicLoreGenerator: {e}")
            return False
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore for a world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Dictionary containing foundation lore
        """
        return await self.world_builder.initialize_world_lore(environment_desc)
    
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete set of lore for a game world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Complete lore package
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_complete_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Complete lore generation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        try:
            # 1) Foundation lore
            foundation_data = await self.world_builder.initialize_world_lore(environment_desc)
            if isinstance(foundation_data, dict) and "error" in foundation_data:
                return foundation_data
    
            # 2) Factions referencing 'social_structure' from foundation_data
            factions_data = await self.faction_generator.generate_factions(environment_desc, foundation_data)
    
            # 3) Cultural elements referencing environment + factions
            cultural_data = await self.faction_generator.generate_cultural_elements(environment_desc, factions_data)
    
            # 4) Historical events referencing environment + foundation_data + factions
            historical_data = await self.faction_generator.generate_historical_events(environment_desc, foundation_data, factions_data)
    
            # 5) Locations referencing environment + factions
            locations_data = await self.faction_generator.generate_locations(environment_desc, factions_data)
    
            # 6) Quest hooks referencing factions + locations
            quests_data = await self.faction_generator.generate_quest_hooks(factions_data, locations_data)
    
            # Report complete action to governance
            await self.governor.process_agent_action_report(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_generator",
                action={
                    "type": "generate_complete_lore",
                    "description": f"Generated complete lore for environment: {environment_desc[:50]}"
                },
                result={
                    "world_lore_count": len(foundation_data) if isinstance(foundation_data, dict) else 0,
                    "factions_count": len(factions_data),
                    "cultural_elements_count": len(cultural_data),
                    "historical_events_count": len(historical_data),
                    "locations_count": len(locations_data),
                    "quests_count": len(quests_data),
                    "setting_name": await self.world_builder.get_setting_name()
                }
            )
    
            return {
                "world_lore": foundation_data,
                "factions": factions_data,
                "cultural_elements": cultural_data,
                "historical_events": historical_data,
                "locations": locations_data,
                "quests": quests_data
            }
        except Exception as e:
            error_msg = f"Error generating complete lore: {str(e)}"
            logger.error(error_msg)
            
            # Use error handler if available
            if self.error_handler:
                from .error_manager import LoreError, ErrorType
                error = LoreError(error_msg, ErrorType.UNKNOWN)
                await self.error_handler.handle_error(error)
            
            return {"error": error_msg}
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        return await self.lore_evolution.evolve_lore_with_event(event_description)
    
    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        
        if self.world_builder:
            await self.world_builder.cleanup()
            
        if self.faction_generator:
            await self.faction_generator.cleanup()
            
        if self.lore_evolution:
            await self.lore_evolution.cleanup()


# Agent getter functions - Updated with wrap_array_field parameter
# Agent getter functions - Updated to use the proper client
def get_foundation_lore_agent():
    """Get or create the foundation lore agent."""
    return Agent(
        name="FoundationLoreAgent",
        instructions=(
            "You produce foundational world lore for a fantasy environment. "
            "Return valid JSON that matches FoundationLoreOutput, which has keys: "
            "[cosmology, magic_system, world_history, calendar_system, social_structure]. "
            "Do NOT include any extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.4),
        output_type=FoundationLoreOutput,
    )

def get_factions_agent():
    """Get or create the factions agent."""
    return Agent(
        name="FactionsAgent",
        instructions=(
            "You generate 3-5 distinct factions for a given setting. "
            'Return valid JSON as an OBJECT: {"factions": [{...}, ...]}. '
            "Each faction object has: name, type, description, values, goals, "
            "headquarters, rivals, allies, hierarchy_type, etc. "
            "No extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.7),
        output_type=FactionsOutput,
    )

def get_cultural_agent():
    """Get or create the cultural agent."""
    return Agent(
        name="CulturalAgent",
        instructions=(
            "You create cultural elements like traditions, customs, rituals. "
            'Return JSON as an OBJECT: {"elements": [{...}, ...]}. '
            "Fields include: name, type, description, practiced_by, significance, "
            "historical_origin. No extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.5),
        output_type=CulturalElementsOutput,
    )

def get_history_agent():
    """Get or create the history agent."""
    return Agent(
        name="HistoryAgent",
        instructions=(
            "You create major historical events. Return JSON as "
            'an OBJECT: {"events": [{...}, ...]}. '
            "Fields: name, date_description, description, participating_factions, "
            "consequences, significance. No extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.6),
        output_type=HistoricalEventsOutput,
    )

def get_locations_agent():
    """Get or create the locations agent."""
    return Agent(
        name="LocationsAgent",
        instructions=(
            "You generate 5-8 significant locations. Return JSON as "
            'an OBJECT: {"locations": [{...}, ...]}. '
            "Fields: name, description, type, controlling_faction, notable_features, "
            "hidden_secrets, strategic_importance (INTEGER 1-10). \n"
            "IMPORTANT: strategic_importance must be a number between 1 and 10, NOT a text description.\n"
            "No extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.7),
        output_type=LocationsOutput,
    )

def get_quests_agent():
    """Get or create the quests agent."""
    return Agent(
        name="QuestsAgent",
        instructions=(
            "You create 5-7 quest hooks. Return JSON as "
            'an OBJECT: {"quests": [{...}, ...]}. '
            "Fields: quest_name, quest_giver, location, description, "
            "objectives, rewards, difficulty, lore_significance. "
            "No extra text outside the JSON.\n\n"
            "Always respect directives from the Nyx governance system and check permissions "
            "before performing any actions."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4.1-nano", 
            openai_client=get_async_openai_client()
        ),
        model_settings=ModelSettings(temperature=0.7),
        output_type=QuestsOutput,
    )
