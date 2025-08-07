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

from typing import Any, Dict, List, Optional, Union, Tuple, Set, Type, Callable, TypeVar
import logging
import json
import asyncio
from datetime import datetime
import psutil
import time
import functools

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

# Pydantic schemas for outputs
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

from .lore_system import LoreSystem
from .lore_validation import LoreValidator
from .error_handler import ErrorHandler
from .lore_cache_manager import LoreCacheManager
from lore.managers.base_manager import BaseLoreManager
from .resource_manager import resource_manager
from .dynamic_lore_generator import DynamicLoreGenerator
from .unified_validation import ValidationManager

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components
lore_system = DynamicLoreGenerator()
lore_validator = ValidationManager()
error_handler = ErrorHandler()

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# -------------------------------------------------------------------------------
# Lore Agent Context and Directive Handler
# -------------------------------------------------------------------------------

class LoreAgentContext(BaseLoreManager):
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
        self.registration_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the agent context and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the agent context and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def _get_cached_data_with_resource_check(
        self,
        data_type: str,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Generic method to get data from cache with resource availability check."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data(data_type, data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting {data_type} data: {e}")
            return None
    
    async def _set_cached_data_with_resource_check(
        self,
        data_type: str,
        data_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Generic method to set data in cache with resource availability check."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data(data_type, data_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting {data_type} data: {e}")
            return False
    
    async def _invalidate_cached_data_generic(
        self,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Generic method to invalidate cached data."""
        try:
            await self.invalidate_cached_data(data_type, data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating {data_type} data: {e}")
    
    # Generic data access methods that use the above helper methods
    
    async def get_data(
        self,
        data_type: str,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get generic data from cache or fetch if not available."""
        return await self._get_cached_data_with_resource_check(data_type, data_id, fetch_func)
    
    async def set_data(
        self,
        data_type: str,
        data_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set generic data in cache."""
        return await self._set_cached_data_with_resource_check(data_type, data_id, data, tags)
    
    async def invalidate_data(
        self,
        data_type: str,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate generic data cache."""
        await self._invalidate_cached_data_generic(data_type, data_id, recursive)
    
    # Specific data access methods using the generic ones
    
    # Agent data methods
    async def get_agent_data(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('agent', agent_id, fetch_func)
    
    async def set_agent_data(self, agent_id: str, data: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('agent', agent_id, data, tags)
    
    async def invalidate_agent_data(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('agent', agent_id, recursive)
    
    # Registration data methods
    async def get_registration_data(self, registration_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('registration', registration_id, fetch_func)
    
    async def set_registration_data(self, registration_id: str, data: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('registration', registration_id, data, tags)
    
    async def invalidate_registration_data(self, registration_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('registration', registration_id, recursive)
    
    async def get_registration_history(self, registration_id: str, fetch_func: Optional[callable] = None) -> Optional[List[Dict[str, Any]]]:
        return await self.get_data('registration_history', registration_id, fetch_func)
    
    async def set_registration_history(self, registration_id: str, history: List[Dict[str, Any]], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('registration_history', registration_id, history, tags)
    
    async def invalidate_registration_history(self, registration_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('registration_history', registration_id, recursive)
    
    async def get_registration_metadata(self, registration_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('registration_metadata', registration_id, fetch_func)
    
    async def set_registration_metadata(self, registration_id: str, metadata: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('registration_metadata', registration_id, metadata, tags)
    
    async def invalidate_registration_metadata(self, registration_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('registration_metadata', registration_id, recursive)
    
    # Agent permissions methods
    async def get_agent_permissions(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('agent_permissions', agent_id, fetch_func)
    
    async def set_agent_permissions(self, agent_id: str, permissions: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('agent_permissions', agent_id, permissions, tags)
    
    async def invalidate_agent_permissions(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('agent_permissions', agent_id, recursive)
    
    # Action history methods
    async def get_action_history(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[List[Dict[str, Any]]]:
        return await self.get_data('action_history', agent_id, fetch_func)
    
    async def set_action_history(self, agent_id: str, history: List[Dict[str, Any]], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('action_history', agent_id, history, tags)
    
    async def invalidate_action_history(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('action_history', agent_id, recursive)
    
    # Agent state methods
    async def get_agent_state(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('agent_state', agent_id, fetch_func)
    
    async def set_agent_state(self, agent_id: str, state: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('agent_state', agent_id, state, tags)
    
    async def invalidate_agent_state(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('agent_state', agent_id, recursive)
    
    # Agent relationships methods
    async def get_agent_relationships(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('agent_relationships', agent_id, fetch_func)
    
    async def set_agent_relationships(self, agent_id: str, relationships: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('agent_relationships', agent_id, relationships, tags)
    
    async def invalidate_agent_relationships(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('agent_relationships', agent_id, recursive)
    
    # Agent metadata methods
    async def get_agent_metadata(self, agent_id: str, fetch_func: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        return await self.get_data('agent_metadata', agent_id, fetch_func)
    
    async def set_agent_metadata(self, agent_id: str, metadata: Dict[str, Any], tags: Optional[Set[str]] = None) -> bool:
        return await self.set_data('agent_metadata', agent_id, metadata, tags)
    
    async def invalidate_agent_metadata(self, agent_id: Optional[str] = None, recursive: bool = True) -> None:
        await self.invalidate_data('agent_metadata', agent_id, recursive)
    
    # Resource management methods
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
agent_context = LoreAgentContext(0, 0)  # Will be properly initialized later

# -------------------------------------------------------------------------------
# Function Tools with Governance Integration
# -------------------------------------------------------------------------------

def create_governed_function_tool(
    agent_type: str,
    action_type: str,
    action_description_template: str,
    id_extractor: Callable = lambda ctx: action_type
):
    """Factory for creating governed function tools with consistent patterns."""
    def decorator(func):
        @function_tool
        @with_governance(
            agent_type=agent_type,
            action_type=action_type,
            action_description=action_description_template,
            id_from_context=id_extractor
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Define common governance parameters for lore functions
LORE_GOVERNANCE_PARAMS = {
    'agent_type': AgentType.NARRATIVE_CRAFTER,
}

# Create governed function tools using the factory
@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_foundation_lore",
    action_description_template="Generating foundation lore for environment: {environment_desc}",
    id_extractor=lambda ctx: "foundation_lore"
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

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_factions",
    action_description_template="Generating factions for environment: {environment_desc}",
    id_extractor=lambda ctx: "factions"
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
    return [f.dict() for f in final_output.factions]

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_cultural_elements",
    action_description_template="Generating cultural elements for environment: {environment_desc}",
    id_extractor=lambda ctx: "cultural"
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
    return [c.dict() for c in final_output.elements]

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_historical_events",
    action_description_template="Generating historical events for environment: {environment_desc}",
    id_extractor=lambda ctx: "history"
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
    return [h.dict() for h in final_output.events]

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_locations",
    action_description_template="Generating locations for environment: {environment_desc}",
    id_extractor=lambda ctx: "locations"
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
    return [l.dict() for l in final_output.locations]

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_quest_hooks",
    action_description_template="Generating quest hooks for environment: {environment_desc}",
    id_extractor=lambda ctx: "quests"
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
    return [q.dict() for q in final_output.quests]

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="analyze_setting",
    action_description_template="Analyzing setting data for environment: {environment_desc}",
    id_extractor=lambda ctx: "setting"
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
# Agent Definitions Factory
# -------------------------------------------------------------------------------

def create_lore_agent(
    name: str,
    instructions: str,
    output_type: Optional[Type] = None,
    temperature: float = 0.5
) -> Agent:
    """
    Factory function to create lore agents with consistent settings.
    
    Args:
        name: Agent name
        instructions: Agent instructions
        output_type: Expected output type
        temperature: Model temperature
        
    Returns:
        Configured Agent instance
    """
    base_instructions = (
        f"{instructions}\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    )
    
    return Agent(
        name=name,
        instructions=base_instructions,
        model=OpenAIResponsesModel(model="gpt-5-nano"),
        model_settings=ModelSettings(temperature=temperature),
        output_type=output_type,
    )

# Create agents using the factory
foundation_lore_agent = create_lore_agent(
    name="FoundationLoreAgent",
    instructions=(
        "You produce foundational world lore for a fantasy environment. "
        "Return valid JSON that matches FoundationLoreOutput, which has keys: "
        "[cosmology, magic_system, world_history, calendar_system, social_structure]. "
        "Do NOT include any extra text outside the JSON."
    ),
    output_type=FoundationLoreOutput,
    temperature=0.4
)

factions_agent = create_lore_agent(
    name="FactionsAgent",
    instructions=(
        "You generate 3-5 distinct factions for a given setting. "
        'Return valid JSON as an OBJECT with a "factions" field containing an array: {"factions": [{...}, ...]}. '
        "Each faction object has: name, type, description, values, goals, "
        "headquarters, rivals, allies, hierarchy_type, etc. "
        "Do NOT return just an array. Always wrap in an object with a 'factions' key. "
        "No extra text outside the JSON."
    ),
    output_type=FactionsOutput,
    temperature=0.7
)

cultural_agent = create_lore_agent(
    name="CulturalAgent",
    instructions=(
        "You create cultural elements like traditions, customs, rituals. "
        'Return JSON as an OBJECT with an "elements" field containing an array: {"elements": [{...}, ...]}. '
        "Fields include: name, type, description, practiced_by, significance, "
        "historical_origin. Do NOT return just an array. Always wrap in an object with an 'elements' key. "
        "No extra text outside the JSON."
    ),
    output_type=CulturalElementsOutput,
    temperature=0.5
)

history_agent = create_lore_agent(
    name="HistoryAgent",
    instructions=(
        "You create major historical events. Return JSON as "
        'an OBJECT with an "events" field containing an array: {"events": [{...}, ...]}. '
        "Fields: name, date_description, description, participating_factions, "
        "consequences, significance. Do NOT return just an array. Always wrap in an object with an 'events' key. "
        "No extra text outside the JSON."
    ),
    output_type=HistoricalEventsOutput,
    temperature=0.6
)

locations_agent = create_lore_agent(
    name="LocationsAgent",
    instructions=(
        "You generate 5-8 significant locations. Return JSON as "
        'an OBJECT with a "locations" field containing an array: {"locations": [{...}, ...]}. '
        "Fields: name, description, type, controlling_faction, notable_features, "
        "hidden_secrets, strategic_importance. Do NOT return just an array. Always wrap in an object with a 'locations' key. "
        "No extra text outside the JSON."
    ),
    output_type=LocationsOutput,
    temperature=0.7
)

quests_agent = create_lore_agent(
    name="QuestsAgent",
    instructions=(
        "You create 5-7 quest hooks. Return JSON as "
        'an OBJECT with a "quests" field containing an array: {"quests": [{...}, ...]}. '
        "Fields: quest_name, quest_giver, location, description, "
        "objectives, rewards, difficulty, lore_significance. "
        "Do NOT return just an array. Always wrap in an object with a 'quests' key. "
        "No extra text outside the JSON."
    ),
    output_type=QuestsOutput,
    temperature=0.7
)

setting_analysis_agent = create_lore_agent(
    name="SettingAnalysisAgent",
    instructions=(
        "You analyze the game setting and NPC data to propose relevant "
        "organizations or factions. Return JSON with categories like "
        "academic, athletic, social, professional, cultural, political, other. "
        "Each category is an array of objects with fields like name, type, "
        "description, membership_basis, hierarchy, gathering_location, etc. "
        "No extra text outside the JSON."
    ),
    temperature=0.7
)

integration_agent = create_lore_agent(
    name="IntegrationAgent",
    instructions=(
        "You integrate different parts of lore to ensure consistency. "
        "Return JSON matching IntegrationOutput with fields for any "
        "inconsistencies, resolutions, and integrated data. "
        "No extra text outside the JSON."
    ),
    output_type=IntegrationOutput,
    temperature=0.5
)

conflict_resolution_agent = create_lore_agent(
    name="ConflictResolutionAgent",
    instructions=(
        "You resolve conflicts between different parts of lore. "
        "Return JSON matching ConflictResolutionOutput with fields for "
        "conflict description, resolution approach, and updated data. "
        "No extra text outside the JSON."
    ),
    output_type=ConflictResolutionOutput,
    temperature=0.6
)

validation_agent = create_lore_agent(
    name="ValidationAgent",
    instructions=(
        "You validate lore for consistency and quality. "
        "Return JSON matching ValidationOutput with fields for "
        "is_valid (boolean), issues (array), and suggestions (array). "
        "No extra text outside the JSON."
    ),
    output_type=ValidationOutput,
    temperature=0.4
)

fix_agent = create_lore_agent(
    name="FixAgent",
    instructions=(
        "You fix inconsistencies in lore. "
        "Return JSON matching FixOutput with fields for issue_id, "
        "fix_description, updated_data, and affected_components. "
        "No extra text outside the JSON."
    ),
    output_type=FixOutput,
    temperature=0.5
)

relationship_validation_agent = create_lore_agent(
    name="RelationshipValidationAgent",
    instructions=(
        "You validate relationships between different parts of lore. "
        "Return JSON matching ValidationOutput with fields for "
        "is_valid (boolean), relationship_issues (array), and "
        "relationship_suggestions (array). "
        "No extra text outside the JSON."
    ),
    output_type=ValidationOutput, 
    temperature=0.4
)

# -------------------------------------------------------------------------------
# Main Functions for Lore Creation with Governance
# -------------------------------------------------------------------------------

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="create_complete_lore",
    action_description_template="Creating complete lore for environment: {environment_desc}",
    id_extractor=lambda ctx: "complete_lore"
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

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="integrate_npc_lore",
    action_description_template="Integrating lore with NPCs: {npc_ids}",
    id_extractor=lambda ctx: "npc_lore"
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

@create_governed_function_tool(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_scene_description",
    action_description_template="Generating scene description for location: {location_name}",
    id_extractor=lambda ctx: "scene"
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

# -------------------------------------------------------------------------------
# Database and NPC Utility Functions
# -------------------------------------------------------------------------------

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

async def determine_relevant_lore(npc_id: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
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
        npc_stats = await get_npc_stats()
        current_location = await get_current_location()
        
        # Get NPC's relationships and beliefs
        relationships = await lore_system.get_npc_relationships(npc_id)
        beliefs = await lore_system.get_npc_beliefs(npc_id)
        
        # Get all available lore elements
        world_lore = await lore_system.get_world_lore()
        location_lore = await lore_system.get_location_lore(current_location)
        faction_lore = await lore_system.get_faction_lore()
        
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
            current_time = await get_current_time()
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

async def integrate_npc_lore(npc_id: int, relevant_lore: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
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
        npc_stats = await get_npc_stats()
        current_location = await get_current_location()
        
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
                        'emotional_impact': calculate_emotional_impact(lore, npc_stats),
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
            await lore_system.update_npc_knowledge(npc_id, integration_results['updated_knowledge'])
        
        if integration_results['updated_beliefs']:
            await lore_system.update_npc_beliefs(npc_id, integration_results['updated_beliefs'])
        
        if integration_results['updated_relationships']:
            await lore_system.update_npc_relationships(npc_id, integration_results['updated_relationships'])
        
        if integration_results['new_memories']:
            await lore_system.add_npc_memories(npc_id, integration_results['new_memories'])
        
        return integration_results
        
    except Exception as e:
        logger.error(f"Error integrating lore for NPC {npc_id}: {e}")
        return {}

def calculate_emotional_impact(lore: Dict[str, Any], npc_stats) -> int:
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

async def get_location_data(location_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
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
        location_data = await lore_system.get_location(location_name)
        if not location_data:
            logger.warning(f"Location {location_name} not found")
            return {}
        
        # Get current time and environment state
        current_time = await get_current_time()
        environment_state = await lore_system.get_environment_state(location_name)
        
        # Get NPCs in the location
        npcs_in_location = await get_nearby_npcs(location_name)
        
        # Get location-specific lore
        location_lore = await lore_system.get_location_lore(location_name)
        
        # Get faction presence in location
        faction_presence = await lore_system.get_location_factions(location_name)
        
        # Get active events or quests in location
        active_events = await lore_system.get_location_events(location_name)
        
        # Get environmental conditions
        environmental_conditions = await lore_system.get_environmental_conditions(location_name)
        
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
                'faction_representation': calculate_faction_representation(npcs_in_location)
            },
            'lore': {
                'historical': [l for l in location_lore if l.get('type') == 'historical'],
                'cultural': [l for l in location_lore if l.get('type') == 'cultural'],
                'current': [l for l in location_lore if l.get('type') == 'current']
            },
            'factions': {
                'present': faction_presence,
                'influence': calculate_faction_influence(faction_presence)
            },
            'events': {
                'active': active_events,
                'upcoming': await lore_system.get_upcoming_events(location_name)
            },
            'resources': {
                'available': await lore_system.get_location_resources(location_name),
                'scarcity': await lore_system.get_resource_scarcity(location_name)
            },
            'connections': {
                'adjacent_locations': await lore_system.get_adjacent_locations(location_name),
                'travel_routes': await lore_system.get_travel_routes(location_name)
            }
        }
        
        # Add dynamic elements based on time and conditions
        location_info['dynamic_elements'] = await get_dynamic_elements(
            location_info,
            current_time,
            environmental_conditions
        )
        
        # Add interaction possibilities
        location_info['interaction_possibilities'] = await get_interaction_possibilities(
            location_info,
            npcs_in_location,
            active_events
        )
        
        return location_info
        
    except Exception as e:
        logger.error(f"Error getting location data for {location_name}: {e}")
        return {}

def calculate_faction_representation(npcs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate faction representation among NPCs in a location."""
    faction_counts = {}
    total_npcs = len(npcs)
    
    if total_npcs == 0:
        return {}
    
    for npc in npcs:
        faction_id = npc.get('faction_id')
        if faction_id:
            faction_counts[faction_id] = faction_counts.get(faction_id, 0) + 1
    
    return {
        faction_id: count / total_npcs
        for faction_id, count in faction_counts.items()
    }

def calculate_faction_influence(faction_presence: List[Dict[str, Any]]) -> Dict[str, float]:
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

async def get_dynamic_elements(
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

async def get_interaction_possibilities(
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

# -------------------------------------------------------------------------------
# Directive Handler
# -------------------------------------------------------------------------------

class LoreDirectiveHandler:
    """
    Standardized handler for processing lore-related directives from Nyx governance.
    
    This class provides a unified way for all lore agents to handle directives
    from the central Nyx governance system.
    """
    
    def __init__(self, user_id: int, conversation_id: int, agent_type: str = AgentType.NARRATIVE_CRAFTER, agent_id: str = "lore_generator"):
        """
        Initialize the lore directive handler.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            agent_type: Agent type (default: NARRATIVE_CRAFTER)
            agent_id: Agent ID (default: lore_generator)
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.governor = None
        self.directive_handler = None
        
        # Store prohibited actions from directives
        self.prohibited_actions = []
        
        # Store action modifications from directives
        self.action_modifications = {}
    
    async def initialize(self):
        """Initialize the handler with Nyx governance."""
        # Get governance system
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        
        # Initialize directive handler
        self.directive_handler = DirectiveHandler(
            self.user_id, 
            self.conversation_id, 
            self.agent_type,
            self.agent_id,
            governance=governance  # pass the object here
        )
        
        # Register handlers for different directive types
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        
        # Start background processing of directives
        self.directive_task = await self.directive_handler.start_background_processing(interval=60.0)
    
    async def _handle_action_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle action directives.
        
        Args:
            directive: The directive data
            
        Returns:
            Result of processing
        """
        directive_id = directive.get("id")
        instruction = directive.get("instruction", "")
        
        logger.info(f"Processing action directive {directive_id}: {instruction}")
        
        if "generate_lore" in instruction.lower():
            # Handle lore generation directive
            environment_desc = directive.get("environment_desc", "")
            if environment_desc:
                # Import here to avoid circular imports
                lore_generator = DynamicLoreGenerator(self.user_id, self.conversation_id)
                result = await lore_generator.generate_complete_lore(environment_desc)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "lore_generated": True,
                    "environment": environment_desc
                }
        
        elif "integrate_lore" in instruction.lower():
            # Handle lore integration directive
            npc_ids = directive.get("npc_ids", [])
            if npc_ids:
                # Import here to avoid circular imports
                from lore.lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                result = await integration_system.integrate_lore_with_npcs(npc_ids)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "npcs_integrated": len(npc_ids)
                }
        
        elif "update_lore" in instruction.lower():
            # Handle lore update directive
            event_description = directive.get("event_description", "")
            if event_description:
                # Import here to avoid circular imports
                from lore.lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                result = await integration_system.update_lore_after_narrative_event(event_description)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "lore_updated": True,
                    "event_description": event_description
                }
        
        elif "analyze_setting" in instruction.lower():
            # Handle setting analysis directive
            # Import here to avoid circular imports
            from lore.setting_analyzer import SettingAnalyzer
            analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
            await analyzer.initialize_governance()
            result = await analyzer.aggregate_npc_data(None)
            return {
                "status": "completed",
                "directive_id": directive_id,
                "setting_analyzed": True,
                "npc_count": len(result.get("npcs", []))
            }
        
        elif "modify_action" in instruction.lower():
            # Store action modifications for future use
            action_type = directive.get("action_type")
            modifications = directive.get("modifications", {})
            
            if action_type:
                self.action_modifications[action_type] = modifications
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "action_type": action_type,
                    "modifications_stored": True
                }
        
        # Default unknown directive case
        return {
            "status": "unknown_directive",
            "directive_id": directive_id,
            "instruction": instruction
        }
    
    async def _handle_prohibition_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prohibition directives.
        
        Args:
            directive: The directive data
            
        Returns:
            Result of processing
        """
        directive_id = directive.get("id")
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        logger.info(f"Processing prohibition directive {directive_id}: {prohibited_actions}")
        
        # Store prohibited actions
        self.prohibited_actions.extend(prohibited_actions)
        
        # Remove duplicates
        self.prohibited_actions = list(set(self.prohibited_actions))
        
        return {
            "status": "prohibition_registered",
            "directive_id": directive_id,
            "prohibited_actions": self.prohibited_actions,
            "reason": reason
        }
    
    async def check_permission(self, action_type: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action is permitted based on directives.
        
        Args:
            action_type: Type of action to check
            details: Optional action details
            
        Returns:
            Permission status dictionary
        """
        # Check if action is prohibited
        if action_type in self.prohibited_actions or "*" in self.prohibited_actions:
            return {
                "approved": False,
                "reasoning": f"Action {action_type} is prohibited by Nyx directive",
                "directive_applied": True
            }
        
        # Check if action has modifications
        if action_type in self.action_modifications:
            modifications = self.action_modifications[action_type]
            return {
                "approved": True,
                "reasoning": f"Action {action_type} is modified by Nyx directive",
                "directive_applied": True,
                "modifications": modifications
            }
        
        # Default approval
        return {
            "approved": True,
            "reasoning": "No directives prohibit this action",
            "directive_applied": False
        }
    
    async def process_directives(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Process all active directives for this lore agent.
        
        Args:
            force_check: Whether to force checking directives
            
        Returns:
            Processing results
        """
        return await self.directive_handler.process_directives(force_check)
    
    async def get_action_modifications(self, action_type: str) -> Dict[str, Any]:
        """
        Get any modifications for a specific action type.
        
        Args:
            action_type: Type of action
            
        Returns:
            Modifications dictionary (empty if none)
        """
        return self.action_modifications.get(action_type, {})
    
    async def is_action_prohibited(self, action_type: str) -> bool:
        """
        Check if an action is prohibited.
        
        Args:
            action_type: Type of action
            
        Returns:
            True if prohibited, False otherwise
        """
        return action_type in self.prohibited_actions or "*" in self.prohibited_actions
    
    async def apply_directive_to_response(self, response: Any, action_type: str) -> Any:
        """
        Apply any directive modifications to a response.
        
        Args:
            response: Original response
            action_type: Type of action
            
        Returns:
            Modified response if applicable, otherwise original
        """
        # If action is prohibited, return error response
        if await self.is_action_prohibited(action_type):
            if isinstance(response, dict):
                return {
                    "error": f"Action {action_type} is prohibited by Nyx directive",
                    "approved": False
                }
            return response
        
        # Apply modifications if any
        modifications = await self.get_action_modifications(action_type)
        if modifications and isinstance(response, dict):
            # Apply each modification
            for key, value in modifications.items():
                if key in response:
                    response[key] = value
        
        return response

# -------------------------------------------------------------------------------
# Base Agent Class for Lore Agents
# -------------------------------------------------------------------------------

class BaseLoreAgent:
    """Base class for all lore agent types with common functionality."""
    
    def __init__(self, lore_system, agent_type: str = AgentType.NARRATIVE_CRAFTER):
        self.lore_system = lore_system
        self.agent_type = agent_type
        self.initialized = False
        self.directive_handler = None
        self._cached_data = {}
    
    async def initialize(self):
        """Initialize the agent with governance and resources."""
        if not self.initialized:
            try:
                # Create directive handler
                self.directive_handler = LoreDirectiveHandler(
                    self.lore_system.user_id,
                    self.lore_system.conversation_id,
                    self.agent_type,
                    self.__class__.__name__
                )
                await self.directive_handler.initialize()
                
                # Additional initialization
                await self._load_cached_data()
                
                self.initialized = True
                return True
            except Exception as e:
                logger.error(f"Error initializing {self.__class__.__name__}: {e}")
                return False
        return True
    
    async def _load_cached_data(self):
        """Load cached data for this agent."""
        try:
            component_name = self._get_component_name()
            data = await self.lore_system.get_component(component_name)
            if data:
                self._cached_data = data
        except Exception as e:
            logger.error(f"Error loading cached data for {self.__class__.__name__}: {e}")
    
    async def _save_cached_data(self):
        """Save cached data for this agent."""
        try:
            component_name = self._get_component_name()
            await self.lore_system.save_component(component_name, self._cached_data)
        except Exception as e:
            logger.error(f"Error saving cached data for {self.__class__.__name__}: {e}")
    
    def _get_component_name(self):
        """Get standardized component name for this agent type."""
        return f"{self.__class__.__name__.lower()}_data"
    
    async def check_permission(self, action_type: str, details: Dict[str, Any] = None) -> bool:
        """Check if an action is permitted by governance."""
        if not self.initialized:
            await self.initialize()
            
        result = await self.directive_handler.check_permission(action_type, details)
        return result.get('approved', False)
    
    async def process_directives(self):
        """Process any pending directives for this agent."""
        if not self.initialized:
            await self.initialize()
            
        return await self.directive_handler.process_directives()

# -------------------------------------------------------------------------------
# Lore Agent Implementations Using the Base Class
# -------------------------------------------------------------------------------

class QuestAgent(BaseLoreAgent):
    """Agent responsible for managing quest-related lore and progression."""
    
    def __init__(self, lore_system):
        super().__init__(lore_system, AgentType.NARRATIVE_CRAFTER)
        self.state = {
            'active_quests': {},
            'quest_progress': {},
            'quest_relationships': {}
        }
    
    async def initialize(self):
        """Initialize the quest agent."""
        if await super().initialize():
            try:
                # Load active quests
                active_quests = await self.lore_system.get_all_components('quest')
                self.state['active_quests'] = {
                    quest['id']: quest for quest in active_quests
                    if quest.get('status') == 'active'
                }
                
                # Load quest progress
                for quest_id in self.state['active_quests']:
                    progress = await self.lore_system.get_quest_progression(quest_id)
                    if progress:
                        self.state['quest_progress'][quest_id] = progress
                
                return True
            except Exception as e:
                logger.error(f"Error initializing QuestAgent: {e}")
                return False
        return False
    
    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """Get comprehensive quest context including related lore and NPCs."""
        try:
            # Check permission
            if not await self.check_permission('get_quest_context'):
                return {}
                
            # Get quest data
            quest = self.state['active_quests'].get(quest_id)
            if not quest:
                return {}
            
            # Get quest lore
            quest_lore = await self.lore_system.get_quest_lore(quest_id)
            
            # Get quest progression
            progression = self.state['quest_progress'].get(quest_id, {})
            
            # Get related NPCs
            related_npcs = []
            for lore in quest_lore:
                if lore.get('metadata', {}).get('npc_id'):
                    npc_id = lore['metadata']['npc_id']
                    npc_data = await self.lore_system.get_component(f"npc_{npc_id}")
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
            # Check permission
            if not await self.check_permission('update_quest_stage'):
                return False
                
            # Update quest progression
            success = await self.lore_system.update_quest_progression(quest_id, stage, data)
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

class NarrativeAgent(BaseLoreAgent):
    """Agent responsible for managing narrative progression and story elements."""
    
    def __init__(self, lore_system):
        super().__init__(lore_system, AgentType.NARRATIVE_CRAFTER)
        self.state = {
            'active_narratives': {},
            'narrative_stages': {},
            'story_elements': {}
        }
    
    async def initialize(self):
        """Initialize the narrative agent."""
        if await super().initialize():
            try:
                # Load active narratives
                narratives = await self.lore_system.get_all_components('narrative')
                self.state['active_narratives'] = {
                    narrative['id']: narrative for narrative in narratives
                    if narrative.get('status') == 'active'
                }
                
                # Load narrative stages
                for narrative_id in self.state['active_narratives']:
                    data = await self.lore_system.get_narrative_data(narrative_id)
                    if data:
                        self.state['narrative_stages'][narrative_id] = data
                
                return True
            except Exception as e:
                logger.error(f"Error initializing NarrativeAgent: {e}")
                return False
        return False
    
    async def get_narrative_context(self, narrative_id: int) -> Dict[str, Any]:
        """Get comprehensive narrative context including related elements."""
        try:
            # Check permission
            if not await self.check_permission('get_narrative_context'):
                return {}
                
            # Get narrative data
            narrative = self.state['active_narratives'].get(narrative_id)
            if not narrative:
                return {}
            
            # Get narrative stages
            stages = self.state['narrative_stages'].get(narrative_id, {})
            
            # Get related story elements
            story_elements = []
            for element_id in narrative.get('related_elements', []):
                element = await self.lore_system.get_component(f"story_element_{element_id}")
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
            # Check permission
            if not await self.check_permission('update_narrative_stage'):
                return False
                
            # Update narrative progression
            success = await self.lore_system.update_narrative_progression(narrative_id, stage, data)
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

class EnvironmentAgent(BaseLoreAgent):
    """Agent responsible for managing environmental conditions and state."""
    
    def __init__(self, lore_system):
        super().__init__(lore_system, AgentType.NARRATIVE_CRAFTER)
        self._environmental_states = {}
        self._resource_levels = {}
        self._current_time = None
        
    async def initialize(self):
        """Initialize the environment agent."""
        if await super().initialize():
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
        return False
    
    async def get_environment_context(self, location_id: int) -> Dict[str, Any]:
        """Get comprehensive environment context including conditions and resources."""
        try:
            # Check permission
            if not await self.check_permission('get_environment_context'):
                return {}
                
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
            # Check permission
            if not await self.check_permission('update_environment_state'):
                return False
                
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
            # Check permission
            if not await self.check_permission('update_game_time'):
                return False
                
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

class FoundationAgent(BaseLoreAgent):
    """Agent responsible for managing world foundation lore."""
    
    def __init__(self, lore_system):
        super().__init__(lore_system, AgentType.NARRATIVE_CRAFTER)
        
    async def generate_foundation(self, environment_desc: str) -> Dict[str, Any]:
        """Generate foundation lore for the world."""
        try:
            # Check permission
            if not await self.check_permission('generate_foundation'):
                return {"error": "Operation not permitted by governance"}
                
            # Get NPC data for better context
            npc_data = await self.aggregate_npc_data()
            
            # Analyze demographics
            demographics = await self.analyze_setting_demographics(npc_data)
            
            # Generate organizations
            organizations = await self.generate_organizations(demographics)
            
            result = {
                "environment": environment_desc,
                "demographics": demographics,
                "organizations": organizations
            }
            
            # Store in cached data
            self._cached_data = result
            await self._save_cached_data()
            
            return result
        except Exception as e:
            logger.error(f"Error generating foundation: {e}")
            return {"error": str(e)}
            
    async def update_foundation(self, updates: Dict[str, Any]) -> bool:
        """Update foundation lore with new information."""
        try:
            # Check permission
            if not await self.check_permission('update_foundation'):
                return False
                
            # Update cached data
            self._cached_data.update(updates)
            await self._save_cached_data()
            
            return True
        except Exception as e:
            logger.error(f"Error updating foundation: {e}")
            return False
            
    async def analyze_setting(self, environment_desc: str) -> Dict[str, Any]:
        """Analyze setting data to generate coherent organizations and relationships."""
        try:
            # Check permission
            if not await self.check_permission('analyze_setting'):
                return {"error": "Operation not permitted by governance"}
                
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
            # Use async connection context manager
            async with get_db_connection_context() as conn:
                # Use async query execution
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, archetypes, likes, dislikes, 
                           hobbies, affiliations, personality_traits, 
                           current_location, archetype_summary
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """, self.lore_system.user_id, self.lore_system.conversation_id)
                
                # Process rows into a structured dict
                all_npcs = []
                all_archetypes, all_likes, all_hobbies, all_affiliations, all_locations = (
                    set(), set(), set(), set(), set()
                )
    
                for row in rows:
                    # asyncpg returns row as a Record object with named attributes
                    npc_id = row['npc_id']
                    npc_name = row['npc_name']
                    archetypes_json = row['archetypes']
                    likes_json = row['likes']
                    dislikes_json = row['dislikes']
                    hobbies_json = row['hobbies']
                    affiliations_json = row['affiliations']
                    personality_json = row['personality_traits']
                    current_location = row['current_location']
                    archetype_summary = row['archetype_summary']
    
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
            # No need for finally block with connection cleanup as the context manager handles it
            
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
            async with get_db_connection_context() as conn:
                # Get environment description
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
                """, self.lore_system.user_id, self.lore_system.conversation_id)
                
                if row:
                    setting_desc = row['value'] or setting_desc
    
                # Get setting name
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentSetting'
                """, self.lore_system.user_id, self.lore_system.conversation_id)
                
                if row:
                    setting_name = row['value'] or setting_name
                    
        except Exception as e:
            logger.error(f"Error fetching setting info: {e}")
            
        return setting_desc, setting_name

class FactionAgent(BaseLoreAgent):
    """Agent responsible for managing factions and their relationships."""
    
    def __init__(self, lore_system):
        super().__init__(lore_system, AgentType.NARRATIVE_CRAFTER)
        
    async def generate_factions(self, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
        """Generate factions for the world."""
        try:
            # Check permission
            if not await self.check_permission('generate_factions'):
                return []
                
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Use the generate_factions_agent
            result = await Runner.run(
                factions_agent,
                json.dumps({
                    'environment_desc': environment_desc,
                    'social_structure': social_structure,
                    'existing_factions': self._cached_data
                }),
                context=run_ctx.context
            )
            
            factions = result.final_output_as(FactionsOutput)
            faction_data = [f.dict() for f in factions.__root__]
            
            # Update internal state
            self._cached_data = faction_data
            await self._save_cached_data()
            
            return faction_data
            
        except Exception as e:
            logger.error(f"Error generating factions: {e}")
            return []
            
    async def update_faction_relationships(self, faction_id: int, relationships: List[Dict[str, Any]]) -> bool:
        """Update relationships between factions."""
        try:
            # Check permission
            if not await self.check_permission('update_faction_relationships'):
                return False
                
            # Find faction in cached data
            faction_found = False
            for faction in self._cached_data:
                if faction.get('id') == faction_id:
                    faction['relationships'] = relationships
                    faction_found = True
                    break
                    
            if not faction_found:
                return False
                
            # Store updated data
            await self._save_cached_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating faction relationships: {e}")
            return False
            
    async def calculate_faction_influence(self, faction_id: int) -> float:
        """Calculate the influence level of a faction."""
        try:
            # Find faction in cached data
            faction = None
            for f in self._cached_data:
                if f.get('id') == faction_id:
                    faction = f
                    break
                    
            if not faction:
                return 0.0
                
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
