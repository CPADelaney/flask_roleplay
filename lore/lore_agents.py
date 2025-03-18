# lore/lore_agents.py
"""
Refactored Lore Agents with full Nyx Governance integration.

Features:
1) Complete integration with Nyx central governance
2) Permission checking before all operations
3) Action reporting for monitoring and tracing
4) Directive handling for system control
5) Registration with proper agent types and constants
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner, trace
from agents.models.openai_responses import OpenAIResponsesModel

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import (
    AgentType, 
    DirectiveType, 
    DirectivePriority
)
from nyx.governance_helpers import (
    with_governance_permission,
    with_action_reporting, 
    with_governance
)
from nyx.directive_handler import DirectiveHandler

# Pydantic schemas for your outputs
from lore.unified_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput,
    IntegrationOutput,
    ConflictResolutionOutput,
    ValidationOutput,
    FixOutput
)

import logging
import json
import asyncio
from datetime import datetime
import psutil
import time

from .lore_system import LoreSystem
from .lore_validation import LoreValidator
from .error_handler import ErrorHandler
from .lore_cache_manager import LoreCacheManager
from .base_manager import BaseManager
from .resource_manager import resource_manager
from .dynamic_lore_generator import DynamicLoreGenerator
from .unified_validation import ValidationManager

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components
lore_system = DynamicLoreGenerator()
lore_validator = ValidationManager()
error_handler = ErrorHandler()

# -------------------------------------------------------------------------------
# Lore Agent Context and Directive Handler
# -------------------------------------------------------------------------------

class LoreAgentContext(BaseManager):
    """Context for lore agents with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.agent_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the agent context and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the agent context and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_agent_data(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent data: {e}")
            return None
    
    async def set_agent_data(
        self,
        agent_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent', agent_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting agent data: {e}")
            return False
    
    async def invalidate_agent_data(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent data cache."""
        try:
            await self.invalidate_cached_data('agent', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent data: {e}")
    
    async def get_agent_permissions(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent permissions from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent_permissions', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent permissions: {e}")
            return None
    
    async def set_agent_permissions(
        self,
        agent_id: str,
        permissions: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent permissions in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent_permissions', agent_id, permissions, tags)
        except Exception as e:
            logger.error(f"Error setting agent permissions: {e}")
            return False
    
    async def invalidate_agent_permissions(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent permissions cache."""
        try:
            await self.invalidate_cached_data('agent_permissions', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent permissions: {e}")
    
    async def get_action_history(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get action history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('action_history', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting action history: {e}")
            return None
    
    async def set_action_history(
        self,
        agent_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set action history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('action_history', agent_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting action history: {e}")
            return False
    
    async def invalidate_action_history(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate action history cache."""
        try:
            await self.invalidate_cached_data('action_history', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating action history: {e}")
    
    async def get_agent_state(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent state from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent_state', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent state: {e}")
            return None
    
    async def set_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent state in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent_state', agent_id, state, tags)
        except Exception as e:
            logger.error(f"Error setting agent state: {e}")
            return False
    
    async def invalidate_agent_state(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent state cache."""
        try:
            await self.invalidate_cached_data('agent_state', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent state: {e}")
    
    async def get_agent_relationships(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent relationships from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent_relationships', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent relationships: {e}")
            return None
    
    async def set_agent_relationships(
        self,
        agent_id: str,
        relationships: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent relationships in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent_relationships', agent_id, relationships, tags)
        except Exception as e:
            logger.error(f"Error setting agent relationships: {e}")
            return False
    
    async def invalidate_agent_relationships(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent relationships cache."""
        try:
            await self.invalidate_cached_data('agent_relationships', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent relationships: {e}")
    
    async def get_agent_metadata(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent_metadata', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent metadata: {e}")
            return None
    
    async def set_agent_metadata(
        self,
        agent_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent_metadata', agent_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting agent metadata: {e}")
            return False
    
    async def invalidate_agent_metadata(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent metadata cache."""
        try:
            await self.invalidate_cached_data('agent_metadata', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent metadata: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
agent_context = LoreAgentContext()

# -------------------------------------------------------------------------------
# Function Tools with Governance Integration
# -------------------------------------------------------------------------------

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_foundation_lore",
    action_description="Generating foundation lore for environment: {environment_desc}",
    id_from_context=lambda ctx: "foundation_lore"
)
async def generate_foundation_lore(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Generate foundation lore (cosmology, magic system, etc.) for a given environment
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    user_prompt = f"""
    Generate cohesive foundational world lore for this environment:
    {environment_desc}

    Return as JSON with keys:
    cosmology, magic_system, world_history, calendar_system, social_structure
    """
    result = await Runner.run(foundation_lore_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(FoundationLoreOutput)
    return final_output.dict()

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_factions",
    action_description="Generating factions for environment: {environment_desc}",
    id_from_context=lambda ctx: "factions"
)
async def generate_factions(ctx, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 distinct factions referencing environment_desc + social_structure
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        social_structure: Social structure description
    """
    user_prompt = f"""
    Generate 3-5 distinct factions for this environment:
    Environment: {environment_desc}
    Social Structure: {social_structure}
    
    Return JSON as an array of objects (matching FactionsOutput).
    """
    result = await Runner.run(factions_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(FactionsOutput)
    # __root__ is a list of FactionSchema objects
    return [f.dict() for f in final_output.__root__]

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_cultural_elements",
    action_description="Generating cultural elements for environment: {environment_desc}",
    id_from_context=lambda ctx: "cultural"
)
async def generate_cultural_elements(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate cultural elements (traditions, taboos, etc.) referencing environment + faction names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    user_prompt = f"""
    Generate 4-7 unique cultural elements for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching CulturalElementsOutput.
    """
    result = await Runner.run(cultural_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(CulturalElementsOutput)
    return [c.dict() for c in final_output.__root__]

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_historical_events",
    action_description="Generating historical events for environment: {environment_desc}",
    id_from_context=lambda ctx: "history"
)
async def generate_historical_events(ctx, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate historical events referencing environment, existing world_history, faction_names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        world_history: Existing world history
        faction_names: Comma-separated faction names
    """
    user_prompt = f"""
    Generate 5-7 significant historical events:
    Environment: {environment_desc}
    Existing World History: {world_history}
    Factions: {faction_names}

    Return JSON array matching HistoricalEventsOutput.
    """
    result = await Runner.run(history_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(HistoricalEventsOutput)
    return [h.dict() for h in final_output.__root__]

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_locations",
    action_description="Generating locations for environment: {environment_desc}",
    id_from_context=lambda ctx: "locations"
)
async def generate_locations(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-8 significant locations referencing environment_desc + faction names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    user_prompt = f"""
    Generate 5-8 significant locations for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching LocationsOutput.
    """
    result = await Runner.run(locations_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(LocationsOutput)
    return [l.dict() for l in final_output.__root__]

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_quest_hooks",
    action_description="Generating quest hooks for environment: {environment_desc}",
    id_from_context=lambda ctx: "quests"
)
async def generate_quest_hooks(ctx, environment_desc: str, faction_names: str, locations: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 quest hooks referencing environment, factions, and locations
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
        locations: Comma-separated location names
    """
    user_prompt = f"""
    Generate 3-5 quest hooks for:
    Environment: {environment_desc}
    Factions: {faction_names}
    Locations: {locations}

    Return JSON array matching QuestsOutput.
    """
    result = await Runner.run(quests_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(QuestsOutput)
    return [q.dict() for q in final_output.__root__]

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="analyze_setting",
    action_description="Analyzing setting data for environment: {environment_desc}",
    id_from_context=lambda ctx: "setting"
)
async def analyze_setting(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Analyze setting data to generate coherent organizations and relationships
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    user_prompt = f"""
    Analyze setting data for:
    Environment: {environment_desc}

    Return JSON object with:
    - organizations
    - relationships
    - power_structures
    - cultural_norms
    - economic_systems
    """
    result = await Runner.run(setting_analysis_agent, user_prompt, context=ctx.context)
    return result.final_output

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Foundation lore agent
foundation_lore_agent = Agent(
    name="FoundationLoreAgent",
    instructions=(
        "You produce foundational world lore for a fantasy environment. "
        "Return valid JSON that matches FoundationLoreOutput, which has keys: "
        "[cosmology, magic_system, world_history, calendar_system, social_structure]. "
        "Do NOT include any extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.4),
    output_type=FoundationLoreOutput,
)

# Factions agent
factions_agent = Agent(
    name="FactionsAgent",
    instructions=(
        "You generate 3-5 distinct factions for a given setting. "
        "Return valid JSON as an array of objects, matching FactionsOutput. "
        "Each faction object has: name, type, description, values, goals, "
        "headquarters, rivals, allies, hierarchy_type, etc. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=FactionsOutput,
)

# Cultural elements agent
cultural_agent = Agent(
    name="CulturalAgent",
    instructions=(
        "You create cultural elements like traditions, customs, rituals. "
        "Return JSON matching CulturalElementsOutput: an array of objects. "
        "Fields include: name, type, description, practiced_by, significance, "
        "historical_origin. No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.5),
    output_type=CulturalElementsOutput,
)

# Historical events agent
history_agent = Agent(
    name="HistoryAgent",
    instructions=(
        "You create major historical events. Return JSON matching "
        "HistoricalEventsOutput: an array with fields name, date_description, "
        "description, participating_factions, consequences, significance. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.6),
    output_type=HistoricalEventsOutput,
)

# Locations agent
locations_agent = Agent(
    name="LocationsAgent",
    instructions=(
        "You generate 5-8 significant locations. Return JSON matching "
        "LocationsOutput: an array of objects with fields name, description, "
        "type, controlling_faction, notable_features, hidden_secrets, "
        "strategic_importance. No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=LocationsOutput,
)

# Quests agent
quests_agent = Agent(
    name="QuestsAgent",
    instructions=(
        "You create 5-7 quest hooks. Return JSON matching QuestsOutput: an "
        "array of objects with quest_name, quest_giver, location, description, "
        "objectives, rewards, difficulty, lore_significance. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=QuestsOutput,
)

# Setting Analysis Agent
setting_analysis_agent = Agent(
    name="SettingAnalysisAgent",
    instructions=(
        "You analyze the game setting and NPC data to propose relevant "
        "organizations or factions. Return JSON with categories like "
        "academic, athletic, social, professional, cultural, political, other. "
        "Each category is an array of objects with fields like name, type, "
        "description, membership_basis, hierarchy, gathering_location, etc. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
)

# -------------------------------------------------------------------------------
# Main Functions for Lore Creation with Governance
# -------------------------------------------------------------------------------

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="create_complete_lore",
    action_description="Creating complete lore for environment: {environment_desc}",
    id_from_context=lambda ctx: "complete_lore"
)
async def create_complete_lore_with_governance(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Create complete lore for an environment with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    try:
        # Generate foundation lore
        foundation = await generate_foundation_lore(ctx, environment_desc)
        
        # Generate factions
        factions = await generate_factions(ctx, environment_desc, foundation.get("social_structure", ""))
        faction_names = ", ".join(f["name"] for f in factions)
        
        # Generate cultural elements
        cultural = await generate_cultural_elements(ctx, environment_desc, faction_names)
        
        # Generate historical events
        history = await generate_historical_events(
            ctx, 
            environment_desc,
            foundation.get("world_history", ""),
            faction_names
        )
        
        # Generate locations
        locations = await generate_locations(ctx, environment_desc, faction_names)
        location_names = ", ".join(l["name"] for l in locations)
        
        # Generate quest hooks
        quests = await generate_quest_hooks(ctx, environment_desc, faction_names, location_names)
        
        # Analyze setting
        setting = await analyze_setting(ctx, environment_desc)
        
        # Combine all lore
        complete_lore = {
            "foundation": foundation,
            "factions": factions,
            "cultural_elements": cultural,
            "history": history,
            "locations": locations,
            "quests": quests,
            "setting_analysis": setting,
            "generated_at": datetime.now().isoformat()
        }
        
        return complete_lore
        
    except Exception as e:
        logger.error(f"Error creating complete lore: {e}")
        return None

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="integrate_npc_lore",
    action_description="Integrating lore with NPCs: {npc_ids}",
    id_from_context=lambda ctx: "npc_lore"
)
async def integrate_lore_with_npcs_with_governance(ctx, npc_ids: List[int], lore_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate lore with NPCs with Nyx governance oversight.
    
    Args:
        npc_ids: List of NPC IDs
        lore_context: Lore context to integrate
    """
    try:
        # Get NPC data
        npc_data = await get_npc_data(npc_ids)
        
        # Integrate lore for each NPC
        integration_results = {}
        for npc_id, npc in npc_data.items():
            # Determine relevant lore
            relevant_lore = await determine_relevant_lore(npc_id, lore_context)
            
            # Integrate lore
            integration_result = await integrate_npc_lore(npc_id, relevant_lore)
            
            integration_results[npc_id] = integration_result
        
        return {
            "success": True,
            "npcs_processed": len(npc_ids),
            "results": integration_results
        }
        
    except Exception as e:
        logger.error(f"Error integrating lore with NPCs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_scene_description",
    action_description="Generating scene description for location: {location_name}",
    id_from_context=lambda ctx: "scene"
)
async def generate_scene_description_with_lore_and_governance(
    ctx,
    location_name: str,
    lore_context: Dict[str, Any],
    npc_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Generate scene description with integrated lore and NPCs with Nyx governance oversight.
    
    Args:
        location_name: Name of the location
        lore_context: Lore context to integrate
        npc_ids: Optional list of NPC IDs present
    """
    try:
        # Get location data
        location_data = await get_location_data(location_name)
        
        # Get NPC data if specified
        npc_data = {}
        if npc_ids:
            npc_data = await get_npc_data(npc_ids)
        
        # Generate scene description
        scene_description = await generate_scene_description(
            location_data,
            lore_context,
            npc_data
        )
        
        return {
            "success": True,
            "location": location_name,
            "description": scene_description,
            "npc_count": len(npc_data)
        }
        
    except Exception as e:
        logger.error(f"Error generating scene description: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def process_lore_directive(ctx, directive: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a lore-related directive with Nyx governance oversight.
    
    Args:
        ctx: Context object
        directive: Directive to process
    """
    try:
        directive_type = directive.get("type")
        directive_id = directive.get("id")
        
        # Process based on directive type
        if directive_type == DirectiveType.ACTION:
            if "generate_lore" in directive.get("instruction", "").lower():
                return await create_complete_lore_with_governance(ctx, directive.get("environment_desc", ""))
            elif "integrate_npc" in directive.get("instruction", "").lower():
                return await integrate_lore_with_npcs_with_governance(
                    ctx,
                    directive.get("npc_ids", []),
                    directive.get("lore_context", {})
                )
            elif "generate_scene" in directive.get("instruction", "").lower():
                return await generate_scene_description_with_lore_and_governance(
                    ctx,
                    directive.get("location_name", ""),
                    directive.get("lore_context", {}),
                    directive.get("npc_ids", [])
                )
        
        return {
            "status": "unknown_directive",
            "directive_id": directive_id,
            "type": directive_type
        }
        
    except Exception as e:
        logger.error(f"Error processing lore directive: {e}")
        return {
            "status": "error",
            "directive_id": directive_id,
            "error": str(e)
        }

async def register_with_governance(user_id: int, conversation_id: int) -> bool:
    """
    Register lore agents with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Get governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Register each agent type
        agent_types = [
            (AgentType.NARRATIVE_CRAFTER, "lore_generator"),
            (AgentType.NARRATIVE_CRAFTER, "lore_integrator"),
            (AgentType.NARRATIVE_CRAFTER, "setting_analyzer")
        ]
        
        for agent_type, agent_id in agent_types:
            await governance.register_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                capabilities={
                    "lore_generation": True,
                    "lore_integration": True,
                    "setting_analysis": True,
                    "npc_interaction": True
                }
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error registering with governance: {e}")
        return False

async def get_npc_data(npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Get NPC data for a list of NPC IDs.
    
    Args:
        npc_ids: List of NPC IDs
        
    Returns:
        Dictionary mapping NPC IDs to their data
    """
    try:
        npc_data = {}
        for npc_id in npc_ids:
            # Get basic NPC info
            query = """
                SELECT id, name, description, faction_id, location_id, traits, 
                       relationships, knowledge, status
                FROM NPCs
                WHERE id = $1
            """
            result = await lore_system.execute_query(query, npc_id)
            if result:
                npc_data[npc_id] = result[0]
                
                # Get NPC stats
                stats_query = """
                    SELECT health, energy, influence
                    FROM NPCStats
                    WHERE npc_id = $1
                """
                stats_result = await lore_system.execute_query(stats_query, npc_id)
                if stats_result:
                    npc_data[npc_id]["stats"] = stats_result[0]
                
                # Get relationships
                rel_query = """
                    SELECT related_npc_id, relationship_type, strength
                    FROM NPCRelationships
                    WHERE npc_id = $1
                """
                rel_result = await lore_system.execute_query(rel_query, npc_id)
                if rel_result:
                    npc_data[npc_id]["relationships"] = rel_result
                
                # Get schedule
                schedule_query = """
                    SELECT schedule_data
                    FROM NPCSchedules
                    WHERE npc_id = $1
                """
                schedule_result = await lore_system.execute_query(schedule_query, npc_id)
                if schedule_result:
                    npc_data[npc_id]["schedule"] = schedule_result[0]["schedule_data"]
        
        return npc_data
        
    except Exception as e:
        logger.error(f"Error getting NPC data: {e}")
        return {}

async def determine_relevant_lore(
    self,
    npc_id: int,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Determine which lore elements are relevant to a specific NPC based on their stats,
    relationships, and current context.
    
    Args:
        npc_id: The ID of the NPC to determine lore for
        context: Optional context dictionary containing current state
        
    Returns:
        Dict containing relevant lore elements and their relevance scores
    """
    try:
        # Get NPC stats and current state
        npc_stats = await get_npc_stats(self.ctx)
        current_location = await self._get_current_location()
        
        # Get NPC's relationships and beliefs
        relationships = await self.lore_system.get_npc_relationships(npc_id)
        beliefs = await self.lore_system.get_npc_beliefs(npc_id)
        
        # Get all available lore elements
        world_lore = await self.lore_system.get_world_lore()
        location_lore = await self.lore_system.get_location_lore(current_location)
        faction_lore = await self.lore_system.get_faction_lore()
        
        # Initialize relevance scores
        relevant_lore = {
            'world_lore': [],
            'location_lore': [],
            'faction_lore': [],
            'historical_events': [],
            'cultural_elements': []
        }
        
        # Score world lore based on NPC's personality and interests
        for lore in world_lore:
            relevance_score = 0.0
            
            # Check if lore aligns with NPC's personality traits
            for trait in npc_stats.personality_traits:
                if trait.lower() in lore.get('themes', []):
                    relevance_score += 0.3
            
            # Check if lore relates to NPC's interests
            for interest in npc_stats.hobbies + npc_stats.likes:
                if interest.lower() in lore.get('keywords', []):
                    relevance_score += 0.2
            
            # Check if lore relates to NPC's relationships
            for rel in relationships:
                if rel['npc_id'] in lore.get('related_npcs', []):
                    relevance_score += 0.3
            
            # Check if lore aligns with NPC's beliefs
            for belief in beliefs:
                if belief['theme'] in lore.get('themes', []):
                    relevance_score += 0.2
            
            if relevance_score > 0.5:  # Threshold for relevance
                relevant_lore['world_lore'].append({
                    'lore': lore,
                    'relevance_score': relevance_score
                })
        
        # Score location lore based on current location and NPC's schedule
        for lore in location_lore:
            relevance_score = 0.0
            
            # Check if lore is about current location
            if lore.get('location_id') == current_location:
                relevance_score += 0.4
            
            # Check if lore relates to NPC's schedule
            current_time = await self._get_current_time()
            if lore.get('time_period') in current_time.get('periods', []):
                relevance_score += 0.3
            
            # Check if lore relates to NPC's relationships in this location
            for rel in relationships:
                if rel['npc_id'] in lore.get('related_npcs', []):
                    relevance_score += 0.3
            
            if relevance_score > 0.5:
                relevant_lore['location_lore'].append({
                    'lore': lore,
                    'relevance_score': relevance_score
                })
        
        # Score faction lore based on NPC's relationships and beliefs
        for lore in faction_lore:
            relevance_score = 0.0
            
            # Check if NPC is part of the faction
            if lore.get('faction_id') in [rel.get('faction_id') for rel in relationships]:
                relevance_score += 0.4
            
            # Check if lore aligns with NPC's beliefs
            for belief in beliefs:
                if belief['faction_id'] == lore.get('faction_id'):
                    relevance_score += 0.3
            
            # Check if lore relates to NPC's relationships
            for rel in relationships:
                if rel.get('faction_id') == lore.get('faction_id'):
                    relevance_score += 0.3
            
            if relevance_score > 0.5:
                relevant_lore['faction_lore'].append({
                    'lore': lore,
                    'relevance_score': relevance_score
                })
        
        # Sort all lore by relevance score
        for category in relevant_lore:
            relevant_lore[category].sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_lore
        
    except Exception as e:
        logger.error(f"Error determining relevant lore for NPC {npc_id}: {e}")
        return {}

async def integrate_npc_lore(
    self,
    npc_id: int,
    relevant_lore: Dict[str, Any],
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Integrate relevant lore with an NPC, updating their knowledge, beliefs, and relationships.
    
    Args:
        npc_id: The ID of the NPC to integrate lore with
        relevant_lore: Dictionary of relevant lore elements and their relevance scores
        context: Optional context dictionary containing current state
        
    Returns:
        Dict containing the integration results and any updates made
    """
    try:
        # Get current NPC state
        npc_stats = await get_npc_stats(self.ctx)
        current_location = await self._get_current_location()
        
        # Initialize integration results
        integration_results = {
            'updated_knowledge': [],
            'updated_beliefs': [],
            'updated_relationships': [],
            'new_memories': [],
            'emotional_impacts': []
        }
        
        # Process world lore
        for lore_item in relevant_lore.get('world_lore', []):
            lore = lore_item['lore']
            relevance_score = lore_item['relevance_score']
            
            # Update knowledge based on relevance
            if relevance_score > 0.7:  # High relevance threshold
                knowledge_update = {
                    'type': 'world_knowledge',
                    'content': lore.get('content'),
                    'themes': lore.get('themes', []),
                    'relevance_score': relevance_score,
                    'timestamp': datetime.now().isoformat()
                }
                integration_results['updated_knowledge'].append(knowledge_update)
                
                # Create memory if highly relevant
                if relevance_score > 0.8:
                    memory = {
                        'type': 'lore_memory',
                        'content': lore.get('content'),
                        'emotional_impact': self._calculate_emotional_impact(lore, npc_stats),
                        'timestamp': datetime.now().isoformat()
                    }
                    integration_results['new_memories'].append(memory)
            
            # Update beliefs if lore aligns with existing beliefs
            for belief in npc_stats.beliefs:
                if any(theme in belief.get('themes', []) for theme in lore.get('themes', [])):
                    belief_update = {
                        'belief_id': belief.get('id'),
                        'strengthened_themes': lore.get('themes', []),
                        'new_evidence': lore.get('content'),
                        'timestamp': datetime.now().isoformat()
                    }
                    integration_results['updated_beliefs'].append(belief_update)
        
        # Process location lore
        for lore_item in relevant_lore.get('location_lore', []):
            lore = lore_item['lore']
            relevance_score = lore_item['relevance_score']
            
            # Update location-specific knowledge
            if relevance_score > 0.6:  # Medium relevance threshold
                location_knowledge = {
                    'type': 'location_knowledge',
                    'location_id': lore.get('location_id'),
                    'content': lore.get('content'),
                    'relevance_score': relevance_score,
                    'timestamp': datetime.now().isoformat()
                }
                integration_results['updated_knowledge'].append(location_knowledge)
                
                # Update relationships if lore mentions other NPCs
                for rel in lore.get('related_npcs', []):
                    if rel['npc_id'] not in [r['npc_id'] for r in npc_stats.relationships]:
                        relationship_update = {
                            'npc_id': rel['npc_id'],
                            'type': 'lore_connection',
                            'strength': relevance_score,
                            'context': lore.get('content'),
                            'timestamp': datetime.now().isoformat()
                        }
                        integration_results['updated_relationships'].append(relationship_update)
        
        # Process faction lore
        for lore_item in relevant_lore.get('faction_lore', []):
            lore = lore_item['lore']
            relevance_score = lore_item['relevance_score']
            
            # Update faction knowledge and relationships
            if relevance_score > 0.6:
                faction_knowledge = {
                    'type': 'faction_knowledge',
                    'faction_id': lore.get('faction_id'),
                    'content': lore.get('content'),
                    'relevance_score': relevance_score,
                    'timestamp': datetime.now().isoformat()
                }
                integration_results['updated_knowledge'].append(faction_knowledge)
                
                # Update faction relationships
                for faction_member in lore.get('faction_members', []):
                    if faction_member['npc_id'] not in [r['npc_id'] for r in npc_stats.relationships]:
                        faction_relationship = {
                            'npc_id': faction_member['npc_id'],
                            'type': 'faction_member',
                            'strength': relevance_score,
                            'faction_id': lore.get('faction_id'),
                            'timestamp': datetime.now().isoformat()
                        }
                        integration_results['updated_relationships'].append(faction_relationship)
        
        # Apply updates to NPC
        if integration_results['updated_knowledge']:
            await self.lore_system.update_npc_knowledge(npc_id, integration_results['updated_knowledge'])
        
        if integration_results['updated_beliefs']:
            await self.lore_system.update_npc_beliefs(npc_id, integration_results['updated_beliefs'])
        
        if integration_results['updated_relationships']:
            await self.lore_system.update_npc_relationships(npc_id, integration_results['updated_relationships'])
        
        if integration_results['new_memories']:
            await self.lore_system.add_npc_memories(npc_id, integration_results['new_memories'])
        
        return integration_results
        
    except Exception as e:
        logger.error(f"Error integrating lore for NPC {npc_id}: {e}")
        return {}

def _calculate_emotional_impact(self, lore: Dict[str, Any], npc_stats: NPCStats) -> int:
    """Calculate the emotional impact of lore on an NPC based on their personality."""
    impact = 0
    
    # Check if lore aligns with NPC's personality traits
    for trait in npc_stats.personality_traits:
        if trait.lower() in lore.get('themes', []):
            impact += 1
    
    # Check if lore relates to NPC's interests
    for interest in npc_stats.hobbies + npc_stats.likes:
        if interest.lower() in lore.get('keywords', []):
            impact += 1
    
    # Check if lore contradicts NPC's dislikes
    for dislike in npc_stats.dislikes:
        if dislike.lower() in lore.get('keywords', []):
            impact -= 1
    
    return max(min(impact, 5), -5)  # Clamp between -5 and 5

async def get_location_data(
    self,
    location_name: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Retrieve location-specific data including environment, NPCs, and lore.
    
    Args:
        location_name: Name of the location to get data for
        context: Optional context dictionary containing current state
        
    Returns:
        Dict containing location data including environment, NPCs, and lore
    """
    try:
        # Get basic location data
        location_data = await self.lore_system.get_location(location_name)
        if not location_data:
            logger.warning(f"Location {location_name} not found")
            return {}
        
        # Get current time and environment state
        current_time = await self._get_current_time()
        environment_state = await self.lore_system.get_environment_state(location_name)
        
        # Get NPCs in the location
        npcs_in_location = await self._get_nearby_npcs(location_name)
        
        # Get location-specific lore
        location_lore = await self.lore_system.get_location_lore(location_name)
        
        # Get faction presence in location
        faction_presence = await self.lore_system.get_location_factions(location_name)
        
        # Get active events or quests in location
        active_events = await self.lore_system.get_location_events(location_name)
        
        # Get environmental conditions
        environmental_conditions = await self.lore_system.get_environmental_conditions(location_name)
        
        # Compile location data
        location_info = {
            'basic_info': {
                'name': location_data.get('name'),
                'type': location_data.get('type'),
                'description': location_data.get('description'),
                'size': location_data.get('size'),
                'population': location_data.get('population'),
                'climate': location_data.get('climate'),
                'terrain': location_data.get('terrain')
            },
            'current_state': {
                'time': current_time,
                'environment': environment_state,
                'conditions': environmental_conditions
            },
            'npcs': {
                'total': len(npcs_in_location),
                'list': npcs_in_location,
                'faction_representation': self._calculate_faction_representation(npcs_in_location)
            },
            'lore': {
                'historical': [l for l in location_lore if l.get('type') == 'historical'],
                'cultural': [l for l in location_lore if l.get('type') == 'cultural'],
                'current': [l for l in location_lore if l.get('type') == 'current']
            },
            'factions': {
                'present': faction_presence,
                'influence': self._calculate_faction_influence(faction_presence)
            },
            'events': {
                'active': active_events,
                'upcoming': await self.lore_system.get_upcoming_events(location_name)
            },
            'resources': {
                'available': await self.lore_system.get_location_resources(location_name),
                'scarcity': await self.lore_system.get_resource_scarcity(location_name)
            },
            'connections': {
                'adjacent_locations': await self.lore_system.get_adjacent_locations(location_name),
                'travel_routes': await self.lore_system.get_travel_routes(location_name)
            }
        }
        
        # Add dynamic elements based on time and conditions
        location_info['dynamic_elements'] = await self._get_dynamic_elements(
            location_info,
            current_time,
            environmental_conditions
        )
        
        # Add interaction possibilities
        location_info['interaction_possibilities'] = await self._get_interaction_possibilities(
            location_info,
            npcs_in_location,
            active_events
        )
        
        return location_info
        
    except Exception as e:
        logger.error(f"Error getting location data for {location_name}: {e}")
        return {}

def _calculate_faction_representation(self, npcs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate faction representation among NPCs in a location."""
    faction_counts = {}
    total_npcs = len(npcs)
    
    for npc in npcs:
        faction_id = npc.get('faction_id')
        if faction_id:
            faction_counts[faction_id] = faction_counts.get(faction_id, 0) + 1
    
    return {
        faction_id: count / total_npcs
        for faction_id, count in faction_counts.items()
    }

def _calculate_faction_influence(self, faction_presence: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate faction influence in a location based on presence and resources."""
    influence = {}
    
    for faction in faction_presence:
        faction_id = faction.get('faction_id')
        if not faction_id:
            continue
            
        # Base influence from presence
        base_influence = faction.get('presence_strength', 0.5)
        
        # Modify influence based on resources
        resource_modifier = faction.get('resource_control', 0.5)
        
        # Modify influence based on relationships
        relationship_modifier = faction.get('relationship_strength', 0.5)
        
        # Calculate final influence
        influence[faction_id] = (base_influence + resource_modifier + relationship_modifier) / 3
    
    return influence

async def _get_dynamic_elements(
    self,
    location_info: Dict[str, Any],
    current_time: Dict[str, Any],
    environmental_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """Get dynamic elements of a location based on time and conditions."""
    dynamic_elements = {
        'time_based': [],
        'weather_based': [],
        'event_based': [],
        'npc_based': []
    }
    
    # Add time-based elements
    if current_time.get('is_night'):
        dynamic_elements['time_based'].append({
            'type': 'lighting',
            'description': 'Dim lighting throughout the area',
            'effect': 'reduced_visibility'
        })
    
    # Add weather-based elements
    if environmental_conditions.get('weather') == 'rain':
        dynamic_elements['weather_based'].append({
            'type': 'weather',
            'description': 'Rain creates slippery surfaces',
            'effect': 'reduced_traction'
        })
    
    # Add event-based elements
    for event in location_info['events']['active']:
        dynamic_elements['event_based'].append({
            'type': 'event',
            'description': event.get('description'),
            'effect': event.get('effect')
        })
    
    # Add NPC-based elements
    for npc in location_info['npcs']['list']:
        if npc.get('is_causing_disturbance'):
            dynamic_elements['npc_based'].append({
                'type': 'npc_activity',
                'npc_id': npc.get('id'),
                'description': npc.get('activity_description'),
                'effect': npc.get('activity_effect')
            })
    
    return dynamic_elements

async def _get_interaction_possibilities(
    self,
    location_info: Dict[str, Any],
    npcs: List[Dict[str, Any]],
    active_events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Get possible interactions in a location based on NPCs and events."""
    interactions = []
    
    # Add NPC-based interactions
    for npc in npcs:
        if npc.get('is_interactable'):
            interactions.append({
                'type': 'npc_interaction',
                'npc_id': npc.get('id'),
                'description': npc.get('interaction_description'),
                'requirements': npc.get('interaction_requirements', []),
                'possible_outcomes': npc.get('interaction_outcomes', [])
            })
    
    # Add event-based interactions
    for event in active_events:
        if event.get('has_interactions'):
            interactions.append({
                'type': 'event_interaction',
                'event_id': event.get('id'),
                'description': event.get('interaction_description'),
                'requirements': event.get('interaction_requirements', []),
                'possible_outcomes': event.get('interaction_outcomes', [])
            })
    
    # Add location-based interactions
    for resource in location_info['resources']['available']:
        if resource.get('is_interactable'):
            interactions.append({
                'type': 'resource_interaction',
                'resource_id': resource.get('id'),
                'description': resource.get('interaction_description'),
                'requirements': resource.get('interaction_requirements', []),
                'possible_outcomes': resource.get('interaction_outcomes', [])
            })
    
    return interactions

class QuestAgent:
    """Agent responsible for managing quest-related lore and progression."""
    
    def __init__(self, lore_manager: LoreManager):
        self.lore_manager = lore_manager
        self.state = {
            'active_quests': {},
            'quest_progress': {},
            'quest_relationships': {}
        }
    
    async def initialize(self):
        """Initialize the quest agent."""
        try:
            # Load active quests
            active_quests = await self.lore_manager.get_all_components('quest')
            self.state['active_quests'] = {
                quest['id']: quest for quest in active_quests
                if quest.get('status') == 'active'
            }
            
            # Load quest progress
            for quest_id in self.state['active_quests']:
                progress = await self.lore_manager.get_quest_progression(quest_id)
                if progress:
                    self.state['quest_progress'][quest_id] = progress
            
            return True
        except Exception as e:
            logger.error(f"Error initializing QuestAgent: {e}")
            return False
    
    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """Get comprehensive quest context including related lore and NPCs."""
        try:
            # Get quest data
            quest = self.state['active_quests'].get(quest_id)
            if not quest:
                return {}
            
            # Get quest lore
            quest_lore = await self.lore_manager.get_quest_lore(quest_id)
            
            # Get quest progression
            progression = self.state['quest_progress'].get(quest_id, {})
            
            # Get related NPCs
            related_npcs = []
            for lore in quest_lore:
                if lore.get('metadata', {}).get('npc_id'):
                    npc_id = lore['metadata']['npc_id']
                    npc_data = await self.lore_manager.get_component(f"npc_{npc_id}")
                    if npc_data:
                        related_npcs.append(npc_data)
            
            return {
                'quest': quest,
                'lore': quest_lore,
                'progression': progression,
                'related_npcs': related_npcs
            }
        except Exception as e:
            logger.error(f"Error getting quest context: {e}")
            return {}
    
    async def update_quest_stage(self, quest_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update quest stage with new data and handle related updates."""
        try:
            # Update quest progression
            success = await self.lore_manager.update_quest_progression(quest_id, stage, data)
            if not success:
                return False
            
            # Update quest state
            if quest_id in self.state['active_quests']:
                quest = self.state['active_quests'][quest_id]
                quest['current_stage'] = stage
                quest['last_updated'] = datetime.now().isoformat()
            
            # Update quest progress in state
            self.state['quest_progress'][quest_id] = {
                'stage': stage,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
        except Exception as e:
            logger.error(f"Error updating quest stage: {e}")
            return False

class NarrativeAgent:
    """Agent responsible for managing narrative progression and story elements."""
    
    def __init__(self, lore_manager: LoreManager):
        self.lore_manager = lore_manager
        self.state = {
            'active_narratives': {},
            'narrative_stages': {},
            'story_elements': {}
        }
    
    async def initialize(self):
        """Initialize the narrative agent."""
        try:
            # Load active narratives
            narratives = await self.lore_manager.get_all_components('narrative')
            self.state['active_narratives'] = {
                narrative['id']: narrative for narrative in narratives
                if narrative.get('status') == 'active'
            }
            
            # Load narrative stages
            for narrative_id in self.state['active_narratives']:
                data = await self.lore_manager.get_narrative_data(narrative_id)
                if data:
                    self.state['narrative_stages'][narrative_id] = data
            
            return True
        except Exception as e:
            logger.error(f"Error initializing NarrativeAgent: {e}")
            return False
    
    async def get_narrative_context(self, narrative_id: int) -> Dict[str, Any]:
        """Get comprehensive narrative context including related elements."""
        try:
            # Get narrative data
            narrative = self.state['active_narratives'].get(narrative_id)
            if not narrative:
                return {}
            
            # Get narrative stages
            stages = self.state['narrative_stages'].get(narrative_id, {})
            
            # Get related story elements
            story_elements = []
            for element_id in narrative.get('related_elements', []):
                element = await self.lore_manager.get_component(f"story_element_{element_id}")
                if element:
                    story_elements.append(element)
            
            return {
                'narrative': narrative,
                'stages': stages,
                'story_elements': story_elements
            }
        except Exception as e:
            logger.error(f"Error getting narrative context: {e}")
            return {}
    
    async def update_narrative_stage(self, narrative_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update narrative stage with new data and handle related updates."""
        try:
            # Update narrative progression
            success = await self.lore_manager.update_narrative_progression(narrative_id, stage, data)
            if not success:
                return False
            
            # Update narrative state
            if narrative_id in self.state['active_narratives']:
                narrative = self.state['active_narratives'][narrative_id]
                narrative['current_stage'] = stage
                narrative['last_updated'] = datetime.now().isoformat()
            
            # Update narrative stages in state
            if narrative_id not in self.state['narrative_stages']:
                self.state['narrative_stages'][narrative_id] = {}
            
            self.state['narrative_stages'][narrative_id][stage] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
        except Exception as e:
            logger.error(f"Error updating narrative stage: {e}")
            return False

class SocialAgent:
    """Agent responsible for managing social relationships and interactions."""
    
    def __init__(self, lore_manager: LoreManager):
        self.lore_manager = lore_manager
        self.state = {
            'social_networks': {},
            'relationship_strengths': {},
            'interaction_history': {}
        }
    
    async def initialize(self):
        """Initialize the social agent."""
        try:
            # Load social networks
            networks = await self.lore_manager.get_all_components('social_network')
            self.state['social_networks'] = {
                network['id']: network for network in networks
            }
            
            # Load relationship strengths
            for network_id in self.state['social_networks']:
                relationships = await self.lore_manager.get_social_links(network_id, 'network')
                self.state['relationship_strengths'][network_id] = {
                    rel['metadata']['entity_id']: rel['metadata']['strength']
                    for rel in relationships
                }
            
            return True
        except Exception as e:
            logger.error(f"Error initializing SocialAgent: {e}")
            return False
    
    async def get_social_context(self, entity_id: int, entity_type: str) -> Dict[str, Any]:
        """Get comprehensive social context including relationships and history."""
        try:
            # Get social links
            links = await self.lore_manager.get_social_links(entity_id, entity_type)
            
            # Get relationship strengths
            strengths = self.state['relationship_strengths'].get(entity_id, {})
            
            # Get interaction history
            history = self.state['interaction_history'].get(entity_id, [])
            
            return {
                'links': links,
                'strengths': strengths,
                'history': history
            }
        except Exception as e:
            logger.error(f"Error getting social context: {e}")
            return {}
    
    async def update_relationships(self, entity_id: int, entity_type: str, links: List[Dict[str, Any]]) -> bool:
        """Update social relationships and handle related updates."""
        try:
            # Update social links
            success = await self.lore_manager.update_social_links(entity_id, entity_type, links)
            if not success:
                return False
            
            # Update relationship strengths
            self.state['relationship_strengths'][entity_id] = {
                link['metadata']['entity_id']: link['metadata']['strength']
                for link in links
            }
            
            # Update interaction history
            if entity_id not in self.state['interaction_history']:
                self.state['interaction_history'][entity_id] = []
            
            self.state['interaction_history'][entity_id].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'relationship_update',
                'links': links
            })
            
            return True
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")
            return False

class EnvironmentAgent:
    """Agent responsible for managing environmental conditions and state."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._environmental_states = {}
        self._resource_levels = {}
        self._current_time = None
        
    async def initialize(self):
        """Initialize the environment agent."""
        try:
            # Load environmental states
            locations = await self.lore_system.get_all_components('location')
            for location in locations:
                location_id = location['id']
                state = await self.lore_system.get_environment_state(location_id)
                conditions = await self.lore_system.get_environmental_conditions(location_id)
                resources = await self.lore_system.get_location_resources(location_id)
                
                self._environmental_states[location_id] = {
                    'state': state,
                    'conditions': conditions,
                    'resources': resources
                }
            
            return True
        except Exception as e:
            logger.error(f"Error initializing EnvironmentAgent: {e}")
            return False
    
    async def get_environment_context(self, location_id: int) -> Dict[str, Any]:
        """Get comprehensive environment context including conditions and resources."""
        try:
            # Get environmental state
            state = await self.lore_system.get_environment_state(location_id)
            
            # Get environmental conditions
            conditions = await self.lore_system.get_environmental_conditions(location_id)
            
            # Get location resources
            resources = await self.lore_system.get_location_resources(location_id)
            
            # Get resource scarcity
            scarcity = await self.lore_system.get_resource_scarcity(location_id)
            
            # Get active events
            events = await self.lore_system.get_location_events(location_id)
            
            return {
                'state': state,
                'conditions': conditions,
                'resources': resources,
                'scarcity': scarcity,
                'events': events
            }
        except Exception as e:
            logger.error(f"Error getting environment context: {e}")
            return {}
    
    async def update_environment_state(self, location_id: int, updates: Dict[str, Any]) -> bool:
        """Update environment state and handle related updates."""
        try:
            # Update environmental state
            current_state = await self.lore_system.get_environment_state(location_id)
            updated_state = {**current_state, **updates}
            
            success = await self.lore_system.update_component(
                f"environment_state_{location_id}",
                updated_state
            )
            if not success:
                return False
            
            # Update state in memory
            if location_id in self._environmental_states:
                self._environmental_states[location_id]['state'] = updated_state
            
            return True
        except Exception as e:
            logger.error(f"Error updating environment state: {e}")
            return False

    async def update_game_time(self, time_data: Dict[str, Any]) -> bool:
        """Update game time and related events."""
        try:
            query = """
                INSERT INTO LoreComponents (
                    user_id, conversation_id, component_type,
                    content, metadata, created_at
                ) VALUES ($1, $2, 'game_time', $3, $4, NOW())
            """
            await self.lore_system.db.execute(
                query,
                self.lore_system.user_id,
                self.lore_system.conversation_id,
                json.dumps(time_data),
                json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'time_period': time_data.get('period'),
                    'season': time_data.get('season'),
                    'weather': time_data.get('weather')
                })
            )
            self._current_time = time_data
            return True
        except Exception as e:
            logger.error(f"Error updating game time: {e}")
            return False
            
    async def update_resource_levels(self, location_id: int, resources: Dict[str, float]) -> bool:
        """Update resource levels for a location."""
        try:
            query = """
                INSERT INTO LoreComponents (
                    user_id, conversation_id, component_type,
                    content, metadata, created_at
                ) VALUES ($1, $2, 'resource_levels', $3, $4, NOW())
            """
            await self.lore_system.db.execute(
                query,
                self.lore_system.user_id,
                self.lore_system.conversation_id,
                json.dumps(resources),
                json.dumps({
                    'location_id': location_id,
                    'timestamp': datetime.now().isoformat()
                })
            )
            self._resource_levels[location_id] = resources
            return True
        except Exception as e:
            logger.error(f"Error updating resource levels: {e}")
            return False

# Global agent instances
quest_agent = None
narrative_agent = None
social_agent = None
environment_agent = None
conflict_agent = None
artifact_agent = None
event_agent = None
foundation_agent = None
faction_agent = None
cultural_agent = None
historical_agent = None
location_agent = None
integration_agent = None
validation_agent = None

async def get_agents(lore_system):
    """Get or create global agent instances."""
    global quest_agent, narrative_agent, social_agent, environment_agent
    global conflict_agent, artifact_agent, event_agent, foundation_agent
    global faction_agent, cultural_agent, historical_agent, location_agent
    global integration_agent, validation_agent
    
    if not quest_agent:
        quest_agent = QuestAgent(lore_system)
    if not narrative_agent:
        narrative_agent = NarrativeAgent(lore_system)
    if not social_agent:
        social_agent = SocialAgent(lore_system)
    if not environment_agent:
        environment_agent = EnvironmentAgent(lore_system)
    if not conflict_agent:
        conflict_agent = ConflictAgent(lore_system)
    if not artifact_agent:
        artifact_agent = ArtifactAgent(lore_system)
    if not event_agent:
        event_agent = EventAgent(lore_system)
    if not foundation_agent:
        foundation_agent = FoundationAgent(lore_system)
    if not faction_agent:
        faction_agent = FactionAgent(lore_system)
    if not cultural_agent:
        cultural_agent = CulturalAgent(lore_system)
    if not historical_agent:
        historical_agent = HistoricalAgent(lore_system)
    if not location_agent:
        location_agent = LocationAgent(lore_system)
    if not integration_agent:
        integration_agent = IntegrationAgent(lore_system)
    if not validation_agent:
        validation_agent = ValidationAgent(lore_system)
        
    return (
        quest_agent, narrative_agent, social_agent, environment_agent,
        conflict_agent, artifact_agent, event_agent, foundation_agent,
        faction_agent, cultural_agent, historical_agent, location_agent,
        integration_agent, validation_agent
    )

# Initialize global agent instances
quest_agent = None
narrative_agent = None
social_agent = None
environment_agent = None
conflict_agent = None
artifact_agent = None
event_agent = None
foundation_agent = None
faction_agent = None
cultural_agent = None
historical_agent = None
location_agent = None
quest_agent = None
integration_agent = None
validation_agent = None

class FoundationAgent:
    """Agent responsible for managing world foundation lore."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self.initialized = False
        
    async def initialize(self):
        """Initialize the foundation agent."""
        if not self.initialized:
            self.initialized = True
            # Initialize any required resources
            
    async def generate_foundation(self, environment_desc: str) -> Dict[str, Any]:
        """Generate foundation lore for the world."""
        try:
            # Get NPC data for better context
            npc_data = await self.aggregate_npc_data()
            
            # Analyze demographics
            demographics = await self.analyze_setting_demographics(npc_data)
            
            # Generate organizations
            organizations = await self.generate_organizations(demographics)
            
            return {
                "environment": environment_desc,
                "demographics": demographics,
                "organizations": organizations
            }
        except Exception as e:
            logger.error(f"Error generating foundation: {e}")
            return {"error": str(e)}
            
    async def update_foundation(self, updates: Dict[str, Any]) -> bool:
        """Update foundation lore with new information."""
        try:
            # Update foundation data
            return True
        except Exception as e:
            logger.error(f"Error updating foundation: {e}")
            return False
            
    async def analyze_setting(self, environment_desc: str) -> Dict[str, Any]:
        """Analyze setting data to generate coherent organizations and relationships."""
        try:
            # Get NPC data
            npc_data = await self.aggregate_npc_data()
            
            # Analyze demographics
            demographics = await self.analyze_setting_demographics(npc_data)
            
            # Generate organizations
            organizations = await self.generate_organizations(demographics)
            
            return {
                "environment": environment_desc,
                "demographics": demographics,
                "organizations": organizations
            }
        except Exception as e:
            logger.error(f"Error analyzing setting: {e}")
            return {"error": str(e)}
            
    async def aggregate_npc_data(self) -> Dict[str, Any]:
        """Collect all NPC data into a unified format."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT npc_id, npc_name, archetypes, likes, dislikes, 
                       hobbies, affiliations, personality_traits, 
                       current_location, archetype_summary
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s
            """, (self.lore_system.user_id, self.lore_system.conversation_id))
            
            rows = cursor.fetchall()
            
            # Process rows into a structured dict
            all_npcs = []
            all_archetypes, all_likes, all_hobbies, all_affiliations, all_locations = (
                set(), set(), set(), set(), set()
            )

            for row in rows:
                npc_id, npc_name, archetypes_json, likes_json, dislikes_json, \
                hobbies_json, affiliations_json, personality_json, \
                current_location, archetype_summary = row

                # Safely load JSON fields
                def safe_load(s):
                    try:
                        return json.loads(s) if s else []
                    except:
                        return []

                archetypes = safe_load(archetypes_json)
                likes = safe_load(likes_json)
                dislikes = safe_load(dislikes_json)
                hobbies = safe_load(hobbies_json)
                affiliations = safe_load(affiliations_json)
                personality_traits = safe_load(personality_json)

                # Update sets
                all_archetypes.update(archetypes)
                all_likes.update(likes)
                all_hobbies.update(hobbies)
                all_affiliations.update(affiliations)
                if current_location:
                    all_locations.add(current_location)
                
                all_npcs.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "archetypes": archetypes,
                    "likes": likes,
                    "dislikes": dislikes,
                    "hobbies": hobbies,
                    "affiliations": affiliations,
                    "personality_traits": personality_traits,
                    "current_location": current_location,
                    "archetype_summary": archetype_summary
                })

            # Get the environment desc and setting name from DB
            setting_desc, setting_name = await self._get_current_setting_info()

            return {
                "setting_name": setting_name,
                "setting_description": setting_desc,
                "npcs": all_npcs,
                "aggregated": {
                    "archetypes": list(all_archetypes),
                    "likes": list(all_likes),
                    "hobbies": list(all_hobbies),
                    "affiliations": list(all_affiliations),
                    "locations": list(all_locations),
                }
            }
        except Exception as e:
            logger.error(f"Error aggregating NPC data: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    async def analyze_setting_demographics(self, npc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the demographics and social structure of the setting."""
        try:
            # Group NPCs by location
            locations = {}
            for npc in npc_data["npcs"]:
                location = npc.get("current_location")
                if location:
                    if location not in locations:
                        locations[location] = []
                    locations[location].append(npc)
            
            # Count archetypes
            archetype_counts = {}
            for npc in npc_data["npcs"]:
                for archetype in npc.get("archetypes", []):
                    if archetype not in archetype_counts:
                        archetype_counts[archetype] = 0
                    archetype_counts[archetype] += 1
            
            # Count affiliations
            affiliation_counts = {}
            for npc in npc_data["npcs"]:
                for affiliation in npc.get("affiliations", []):
                    if affiliation not in affiliation_counts:
                        affiliation_counts[affiliation] = 0
                    affiliation_counts[affiliation] += 1
            
            return {
                "setting_name": npc_data["setting_name"],
                "total_npcs": len(npc_data["npcs"]),
                "locations": {
                    location: len(npcs) for location, npcs in locations.items()
                },
                "archetype_distribution": archetype_counts,
                "affiliation_distribution": affiliation_counts,
                "setting_description": npc_data["setting_description"]
            }
        except Exception as e:
            logger.error(f"Error analyzing setting demographics: {e}")
            return {"error": str(e)}
            
    async def generate_organizations(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate organizations based on setting analysis."""
        try:
            # Use the setting analysis agent to generate organizations
            user_prompt = f"""
            Analyze setting data for:
            Setting: {demographics['setting_name']}
            Description: {demographics['setting_description']}
            Total NPCs: {demographics['total_npcs']}
            Locations: {demographics['locations']}
            Archetype Distribution: {demographics['archetype_distribution']}
            Affiliation Distribution: {demographics['affiliation_distribution']}

            Generate organizations that would make sense in this setting.
            Return JSON with categories like academic, athletic, social, professional, cultural, political, other.
            Each category should be an array of objects with fields like name, type, description, membership_basis, hierarchy, gathering_location, etc.
            """
            
            result = await Runner.run(setting_analysis_agent, user_prompt)
            return result.final_output
        except Exception as e:
            logger.error(f"Error generating organizations: {e}")
            return {"error": str(e)}
            
    async def _get_current_setting_info(self) -> Tuple[str, str]:
        """Helper to fetch the current setting name and environment desc from the DB."""
        setting_desc = "A setting with no description."
        setting_name = "The Setting"
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
            """, (self.lore_system.user_id, self.lore_system.conversation_id))
            row = cursor.fetchone()
            if row:
                setting_desc = row[0] or setting_desc

            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
            """, (self.lore_system.user_id, self.lore_system.conversation_id))
            row = cursor.fetchone()
            if row:
                setting_name = row[0] or setting_name
        finally:
            cursor.close()
            conn.close()

        return setting_desc, setting_name

class FactionAgent:
    """Agent responsible for managing factions and their relationships."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._faction_data = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the faction agent."""
        try:
            # Load existing faction data
            faction_data = await self.lore_system.get_component('factions')
            if faction_data:
                self._faction_data = faction_data
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing FactionAgent: {e}")
            return False
            
    async def generate_factions(self, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
        """Generate factions for the world."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_factions_agent
            result = await Runner.run(
                generate_factions_agent,
                json.dumps({
                    'environment_desc': environment_desc,
                    'social_structure': social_structure,
                    'existing_factions': self._faction_data
                }),
                context=run_ctx.context
            )
            
            factions = result.final_output_as(FactionsOutput).dict()
            
            # Update internal state
            self._faction_data = factions
            
            # Store in database
            await self.lore_system.save_component(
                'factions',
                factions
            )
            
            return factions
            
        except Exception as e:
            logger.error(f"Error generating factions: {e}")
            return []
            
    async def update_faction_relationships(self, faction_id: int, relationships: List[Dict[str, Any]]) -> bool:
        """Update relationships between factions."""
        try:
            if faction_id not in self._faction_data:
                return False
                
            self._faction_data[faction_id]['relationships'] = relationships
            
            # Store updated data
            await self.lore_system.save_component(
                'factions',
                self._faction_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating faction relationships: {e}")
            return False
            
    async def calculate_faction_influence(self, faction_id: int) -> float:
        """Calculate the influence level of a faction."""
        try:
            if faction_id not in self._faction_data:
                return 0.0
                
            faction = self._faction_data[faction_id]
            influence = 0.0
            
            # Calculate based on relationships
            for rel in faction.get('relationships', []):
                influence += rel.get('strength', 0) * rel.get('type', 0)
                
            # Normalize influence
            influence = max(0.0, min(1.0, influence / 100.0))
            
            return influence
            
        except Exception as e:
            logger.error(f"Error calculating faction influence: {e}")
            return 0.0

class CulturalAgent:
    """Agent responsible for managing cultural elements and traditions."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._cultural_data = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the cultural agent."""
        try:
            # Load existing cultural data
            cultural_data = await self.lore_system.get_component('cultural_elements')
            if cultural_data:
                self._cultural_data = cultural_data
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing CulturalAgent: {e}")
            return False
            
    async def generate_cultural_elements(self, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
        """Generate cultural elements for the world."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_cultural_elements_agent
            result = await Runner.run(
                generate_cultural_elements_agent,
                json.dumps({
                    'environment_desc': environment_desc,
                    'faction_names': faction_names,
                    'existing_culture': self._cultural_data
                }),
                context=run_ctx.context
            )
            
            cultural_elements = result.final_output_as(CulturalElementsOutput).dict()
            
            # Update internal state
            self._cultural_data = cultural_elements
            
            # Store in database
            await self.lore_system.save_component(
                'cultural_elements',
                cultural_elements
            )
            
            return cultural_elements
            
        except Exception as e:
            logger.error(f"Error generating cultural elements: {e}")
            return []
            
    async def update_cultural_traditions(self, updates: Dict[str, Any]) -> bool:
        """Update cultural traditions and practices."""
        try:
            # Merge updates with existing data
            self._cultural_data.update(updates)
            
            # Store updated data
            await self.lore_system.save_component(
                'cultural_elements',
                self._cultural_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating cultural traditions: {e}")
            return False
            
    async def get_cultural_context(self, location_id: int) -> Dict[str, Any]:
        """Get cultural context for a specific location."""
        try:
            location_data = await self.lore_system.get_component(f'location_{location_id}')
            if not location_data:
                return {}
                
            # Get relevant cultural elements
            relevant_culture = {
                k: v for k, v in self._cultural_data.items()
                if v.get('location_id') == location_id
            }
            
            return {
                'location': location_data,
                'cultural_elements': relevant_culture
            }
            
        except Exception as e:
            logger.error(f"Error getting cultural context: {e}")
            return {}

class HistoricalAgent:
    """Agent responsible for managing historical events and timelines."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._historical_data = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the historical agent."""
        try:
            # Load existing historical data
            historical_data = await self.lore_system.get_component('historical_events')
            if historical_data:
                self._historical_data = historical_data
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing HistoricalAgent: {e}")
            return False
            
    async def generate_historical_events(self, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
        """Generate historical events for the world."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_historical_events_agent
            result = await Runner.run(
                generate_historical_events_agent,
                json.dumps({
                    'environment_desc': environment_desc,
                    'world_history': world_history,
                    'faction_names': faction_names,
                    'existing_events': self._historical_data
                }),
                context=run_ctx.context
            )
            
            historical_events = result.final_output_as(HistoricalEventsOutput).dict()
            
            # Update internal state
            self._historical_data = historical_events
            
            # Store in database
            await self.lore_system.save_component(
                'historical_events',
                historical_events
            )
            
            return historical_events
            
        except Exception as e:
            logger.error(f"Error generating historical events: {e}")
            return []
            
    async def update_timeline(self, event_id: int, updates: Dict[str, Any]) -> bool:
        """Update a historical event in the timeline."""
        try:
            if event_id not in self._historical_data:
                return False
                
            self._historical_data[event_id].update(updates)
            
            # Store updated data
            await self.lore_system.save_component(
                'historical_events',
                self._historical_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating timeline: {e}")
            return False
            
    async def get_historical_context(self, time_period: str) -> Dict[str, Any]:
        """Get historical context for a specific time period."""
        try:
            # Filter events by time period
            relevant_events = [
                event for event in self._historical_data.values()
                if event.get('time_period') == time_period
            ]
            
            return {
                'time_period': time_period,
                'events': relevant_events
            }
            
        except Exception as e:
            logger.error(f"Error getting historical context: {e}")
            return {}

class LocationAgent:
    """Agent responsible for managing locations and their states."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._location_data = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the location agent."""
        try:
            # Load existing location data
            location_data = await self.lore_system.get_component('locations')
            if location_data:
                self._location_data = location_data
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing LocationAgent: {e}")
            return False
            
    async def generate_locations(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate locations for the world."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_locations_agent
            result = await Runner.run(
                generate_locations_agent,
                json.dumps({
                    'environment_desc': environment_desc,
                    'factions': factions,
                    'existing_locations': self._location_data
                }),
                context=run_ctx.context
            )
            
            locations = result.final_output_as(LocationsOutput).dict()
            
            # Update internal state
            self._location_data = locations
            
            # Store in database
            await self.lore_system.save_component(
                'locations',
                locations
            )
            
            return locations
            
        except Exception as e:
            logger.error(f"Error generating locations: {e}")
            return []
            
    async def update_location_state(self, location_id: int, updates: Dict[str, Any]) -> bool:
        """Update the state of a location."""
        try:
            if location_id not in self._location_data:
                return False
                
            self._location_data[location_id].update(updates)
            
            # Store updated data
            await self.lore_system.save_component(
                'locations',
                self._location_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating location state: {e}")
            return False
            
    async def get_location_context(self, location_id: int) -> Dict[str, Any]:
        """Get full context for a location."""
        try:
            if location_id not in self._location_data:
                return {}
                
            location = self._location_data[location_id]
            
            # Get related data
            cultural_context = await self.lore_system.get_component(f'cultural_{location_id}')
            historical_context = await self.lore_system.get_component(f'historical_{location_id}')
            
            return {
                'location': location,
                'cultural_context': cultural_context,
                'historical_context': historical_context
            }
            
        except Exception as e:
            logger.error(f"Error getting location context: {e}")
            return {}

class QuestAgent:
    """Agent responsible for managing quests and their progression."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._quest_data = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the quest agent."""
        try:
            # Load existing quest data
            quest_data = await self.lore_system.get_component('quests')
            if quest_data:
                self._quest_data = quest_data
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing QuestAgent: {e}")
            return False
            
    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate quest hooks for the world."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_quest_hooks_agent
            result = await Runner.run(
                generate_quest_hooks_agent,
                json.dumps({
                    'factions': factions,
                    'locations': locations,
                    'existing_quests': self._quest_data
                }),
                context=run_ctx.context
            )
            
            quest_hooks = result.final_output_as(QuestsOutput).dict()
            
            # Update internal state
            self._quest_data = quest_hooks
            
            # Store in database
            await self.lore_system.save_component(
                'quests',
                quest_hooks
            )
            
            return quest_hooks
            
        except Exception as e:
            logger.error(f"Error generating quest hooks: {e}")
            return []
            
    async def update_quest_progression(self, quest_id: int, updates: Dict[str, Any]) -> bool:
        """Update the progression of a quest."""
        try:
            if quest_id not in self._quest_data:
                return False
                
            self._quest_data[quest_id].update(updates)
            
            # Store updated data
            await self.lore_system.save_component(
                'quests',
                self._quest_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating quest progression: {e}")
            return False
            
    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """Get full context for a quest."""
        try:
            if quest_id not in self._quest_data:
                return {}
                
            quest = self._quest_data[quest_id]
            
            # Get related data
            faction_data = await self.lore_system.get_component(f'faction_{quest.get("faction_id")}')
            location_data = await self.lore_system.get_component(f'location_{quest.get("location_id")}')
            
            return {
                'quest': quest,
                'faction': faction_data,
                'location': location_data
            }
            
        except Exception as e:
            logger.error(f"Error getting quest context: {e}")
            return {}

class IntegrationAgent:
    """Agent responsible for integrating different types of lore."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._initialized = False
        
    async def initialize(self):
        """Initialize the integration agent."""
        try:
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing IntegrationAgent: {e}")
            return False
            
    async def integrate_lore(self, lore_parts: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate different parts of lore."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the integration_agent
            result = await Runner.run(
                integration_agent,
                json.dumps(lore_parts),
                context=run_ctx.context
            )
            
            integrated_lore = result.final_output_as(IntegrationOutput).dict()
            
            # Store integrated lore
            await self.lore_system.save_component(
                'integrated_lore',
                integrated_lore
            )
            
            return integrated_lore
            
        except Exception as e:
            logger.error(f"Error integrating lore: {e}")
            return {}
            
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> bool:
        """Resolve conflicts between different parts of lore."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the conflict_resolution_agent
            result = await Runner.run(
                conflict_resolution_agent,
                json.dumps(conflicts),
                context=run_ctx.context
            )
            
            resolutions = result.final_output_as(ConflictResolutionOutput).dict()
            
            # Apply resolutions
            for resolution in resolutions:
                await self.lore_system.save_component(
                    f'resolution_{resolution["id"]}',
                    resolution
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return False
            
    async def validate_integration(self, integrated_lore: Dict[str, Any]) -> bool:
        """Validate the integrated lore for consistency."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the validation_agent
            result = await Runner.run(
                validation_agent,
                json.dumps(integrated_lore),
                context=run_ctx.context
            )
            
            validation_result = result.final_output_as(ValidationOutput).dict()
            
            return validation_result.get('is_valid', False)
            
        except Exception as e:
            logger.error(f"Error validating integration: {e}")
            return False

class ValidationAgent:
    """Agent responsible for validating lore consistency and quality."""
    
    def __init__(self, lore_system):
        self.lore_system = lore_system
        self._initialized = False
        
    async def initialize(self):
        """Initialize the validation agent."""
        try:
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing ValidationAgent: {e}")
            return False
            
    async def validate_lore(self, lore: Dict[str, Any]) -> Dict[str, Any]:
        """Validate lore for consistency and quality."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the validation_agent
            result = await Runner.run(
                validation_agent,
                json.dumps(lore),
                context=run_ctx.context
            )
            
            validation_result = result.final_output_as(ValidationOutput).dict()
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating lore: {e}")
            return {'is_valid': False, 'issues': []}
            
    async def fix_inconsistencies(self, issues: List[Dict[str, Any]]) -> bool:
        """Fix inconsistencies in the lore."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the fix_agent
            result = await Runner.run(
                fix_agent,
                json.dumps(issues),
                context=run_ctx.context
            )
            
            fixes = result.final_output_as(FixOutput).dict()
            
            # Apply fixes
            for fix in fixes:
                await self.lore_system.save_component(
                    f'fix_{fix["id"]}',
                    fix
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing inconsistencies: {e}")
            return False
            
    async def validate_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """Validate relationships between different parts of lore."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the relationship_validation_agent
            result = await Runner.run(
                relationship_validation_agent,
                json.dumps(relationships),
                context=run_ctx.context
            )
            
            validation_result = result.final_output_as(ValidationOutput).dict()
            
            return validation_result.get('is_valid', False)
            
        except Exception as e:
            logger.error(f"Error validating relationships: {e}")
            return False
