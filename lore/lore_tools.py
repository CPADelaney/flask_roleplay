# lore/lore_tools.py
from agents import function_tool, Runner, RunContextWrapper
from typing import Dict, Any, List

from lore.lore_agents import (
    foundation_lore_agent,
    factions_agent,
    cultural_agent,
    history_agent,
    locations_agent,
    quests_agent
)
from lore.lore_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

@function_tool
async def generate_foundation_lore(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Generate foundation lore (cosmology, magic system, etc.) for a given environment.
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
async def generate_factions(ctx, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 distinct factions referencing environment_desc + social_structure.
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
async def generate_cultural_elements(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate cultural elements (traditions, taboos, etc.) referencing environment + faction names.
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
async def generate_historical_events(ctx, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate historical events referencing environment, existing world_history, faction_names.
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
async def generate_locations(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-8 significant locations referencing environment_desc + faction names.
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
async def generate_quest_hooks(ctx, faction_names: str, location_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-7 quest hooks referencing existing factions, locations, etc.
    """
    user_prompt = f"""
    Generate 5-7 engaging quest hooks:
    Factions: {faction_names}
    Locations: {location_names}

    Return JSON array matching QuestsOutput.
    """
    result = await Runner.run(quests_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(QuestsOutput)
    return [q.dict() for q in final_output.__root__]
