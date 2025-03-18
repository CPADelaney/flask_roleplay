# lore/lore_tools.py
"""
Updated lore tools with Nyx integration.

This module contains function tools for lore generation that are 
properly integrated with the Nyx governance system.
"""

from typing import Dict, Any, List

from agents import function_tool, Runner
from agents.run_context import RunContextWrapper

# Import agent definitions from lore_agents
from lore.lore_agents import (
    foundation_lore_agent,
    factions_agent,
    cultural_agent,
    history_agent,
    locations_agent,
    quests_agent
)

# Import schemas
from lore.unified_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

# Import governance helpers
from nyx.governance_helpers import with_governance
from nyx.nyx_governance import AgentType

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate cohesive foundational world lore for this environment:
    {environment_desc}

    Return as JSON with keys:
    cosmology, magic_system, world_history, calendar_system, social_structure
    """
    
    result = await Runner.run(foundation_lore_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(FoundationLoreOutput)
    return final_output.dict()

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate 3-5 distinct factions for this environment:
    Environment: {environment_desc}
    Social Structure: {social_structure}
    
    Return JSON as an array of objects (matching FactionsOutput).
    """
    
    result = await Runner.run(factions_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(FactionsOutput)
    return [f.dict() for f in final_output.__root__]

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate 4-7 unique cultural elements for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching CulturalElementsOutput.
    """
    
    result = await Runner.run(cultural_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(CulturalElementsOutput)
    return [c.dict() for c in final_output.__root__]

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate 5-7 significant historical events:
    Environment: {environment_desc}
    Existing World History: {world_history}
    Factions: {faction_names}

    Return JSON array matching HistoricalEventsOutput.
    """
    
    result = await Runner.run(history_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(HistoricalEventsOutput)
    return [h.dict() for h in final_output.__root__]

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate 5-8 significant locations for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching LocationsOutput.
    """
    
    result = await Runner.run(locations_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(LocationsOutput)
    return [l.dict() for l in final_output.__root__]

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
    run_ctx = RunContextWrapper(context=ctx.context)
    
    user_prompt = f"""
    Generate 5-7 engaging quest hooks:
    Factions: {faction_names}
    Locations: {location_names}

    Return JSON array matching QuestsOutput.
    """
    
    result = await Runner.run(quests_agent, user_prompt, context=run_ctx.context)
    final_output = result.final_output_as(QuestsOutput)
    return [q.dict() for q in final_output.__root__]
