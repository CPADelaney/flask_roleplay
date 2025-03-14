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

from typing import Any, Dict, List, Optional, Union

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
from lore.lore_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

import logging
import json
import asyncio
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Lore Agent Context and Directive Handler
# -------------------------------------------------------------------------------

class LoreAgentContext:
    """Context object for lore agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.directive_handler = None
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        
        # Initialize directive handler
        self.directive_handler = DirectiveHandler(
            self.user_id, 
            self.conversation_id, 
            AgentType.NARRATIVE_CRAFTER,
            "lore_generator"
        )
        
        # Register handlers for different directive types
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        
        # Start background processing of directives
        self.directive_task = await self.directive_handler.start_background_processing(interval=60.0)
            
    async def _handle_action_directive(self, directive):
        """Handle action directives from Nyx"""
        instruction = directive.get("instruction", "")
        
        if "generate lore" in instruction.lower():
            # Generate lore based on directive
            environment_desc = directive.get("environment_desc", "")
            if environment_desc:
                from lore.dynamic_lore_generator import DynamicLoreGenerator
                lore_generator = DynamicLoreGenerator(self.user_id, self.conversation_id)
                return await lore_generator.generate_complete_lore(environment_desc)
                
        elif "integrate lore" in instruction.lower():
            # Integrate lore with NPCs
            npc_ids = directive.get("npc_ids", [])
            if npc_ids:
                from lore.lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                return await integration_system.integrate_lore_with_npcs(npc_ids)
        
        return {"status": "unknown_directive", "instruction": instruction}
    
    async def _handle_prohibition_directive(self, directive):
        """Handle prohibition directives from Nyx"""
        # Mark certain lore generation activities as prohibited
        prohibited = directive.get("prohibited_actions", [])
        
        # Store these in context for later checking
        self.prohibited_lore_actions = prohibited
        
        return {"status": "prohibition_registered", "prohibited": prohibited}

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
    action_description="Generating quest hooks for factions and locations",
    id_from_context=lambda ctx: "quests"
)
async def generate_quest_hooks(ctx, faction_names: str, location_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-7 quest hooks referencing existing factions, locations, etc.
    with Nyx governance oversight.
    
    Args:
        faction_names: Comma-separated faction names
        location_names: Comma-separated location names
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

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="analyze_setting",
    action_description="Analyzing setting and NPC data",
    id_from_context=lambda ctx: "setting_analysis"
)
async def analyze_setting(ctx, environment_desc: str, npc_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the game setting and NPC data with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        npc_data: List of NPC data
    """
    user_prompt = f"""
    Analyze this game setting and NPC data:
    
    Environment: {environment_desc}
    
    NPCs: {json.dumps(npc_data)}
    
    Return JSON with categories like academic, athletic, social, professional,
    cultural, political, other. Each category is an array of organization objects.
    """
    result = await Runner.run(setting_analysis_agent, user_prompt, context=ctx.context)
    return json.loads(result.text)

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

async def create_complete_lore_with_governance(
    user_id: int,
    conversation_id: int,
    environment_desc: str
) -> Dict[str, Any]:
    """
    Create complete lore with full Nyx governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        environment_desc: Description of the environment
        
    Returns:
        Complete lore dictionary
    """
    # Create lore context
    lore_context = LoreAgentContext(user_id, conversation_id)
    await lore_context.initialize()
    
    # Check if lore generation is prohibited by governance
    if hasattr(lore_context, 'prohibited_lore_actions') and "generate_complete_lore" in lore_context.prohibited_lore_actions:
        return {
            "error": "Complete lore generation is prohibited by governance directive",
            "prohibited": True
        }
    
    # Get governor for permission check
    governor = lore_context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action_type="generate_lore",
        action_details={"environment_desc": environment_desc}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"], "approved": False}
    
    # Create trace for monitoring
    with trace(
        workflow_name="Lore Generation",
        trace_id=f"lore-gen-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # 1) Foundation lore
        foundation_data = await generate_foundation_lore(lore_context, environment_desc)
        
        # 2) Factions referencing 'social_structure' from foundation_data
        factions_data = await generate_factions(lore_context, environment_desc, foundation_data["social_structure"])
        
        # Get faction names for other generators
        faction_names = ", ".join([f.get("name", "Unknown") for f in factions_data])
        
        # 3) Cultural elements referencing environment + factions
        cultural_data = await generate_cultural_elements(lore_context, environment_desc, faction_names)
        
        # 4) Historical events referencing environment + foundation_data + factions
        historical_data = await generate_historical_events(
            lore_context, 
            environment_desc, 
            foundation_data["world_history"], 
            faction_names
        )
        
        # 5) Locations referencing environment + factions
        locations_data = await generate_locations(lore_context, environment_desc, faction_names)
        
        # Get location names for quest hooks
        location_names = ", ".join([l.get("name", "Unknown") for l in locations_data])
        
        # 6) Quest hooks referencing factions + locations
        quests_data = await generate_quest_hooks(lore_context, faction_names, location_names)
    
    # Assemble complete lore
    complete_lore = {
        "world_lore": foundation_data,
        "factions": factions_data,
        "cultural_elements": cultural_data,
        "historical_events": historical_data,
        "locations": locations_data,
        "quests": quests_data
    }
    
    # Report the action
    await governor.process_agent_action_report(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action={
            "type": "generate_lore",
            "description": f"Generated complete lore for environment: {environment_desc[:50]}"
        },
        result={
            "world_lore_count": len(foundation_data),
            "factions_count": len(factions_data),
            "cultural_elements_count": len(cultural_data),
            "historical_events_count": len(historical_data),
            "locations_count": len(locations_data),
            "quests_count": len(quests_data)
        }
    )
    
    return complete_lore

async def integrate_lore_with_npcs_with_governance(
    user_id: int,
    conversation_id: int,
    npc_ids: List[int]
) -> Dict[str, Any]:
    """
    Integrate lore with NPCs with full Nyx governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        npc_ids: List of NPC IDs to integrate with
        
    Returns:
        Integration results
    """
    # Create lore context
    lore_context = LoreAgentContext(user_id, conversation_id)
    await lore_context.initialize()
    
    # Check if lore integration is prohibited by governance
    if hasattr(lore_context, 'prohibited_lore_actions') and "integrate_lore_with_npcs" in lore_context.prohibited_lore_actions:
        return {
            "error": "Lore integration with NPCs is prohibited by governance directive",
            "prohibited": True
        }
    
    # Get governor for permission check
    governor = lore_context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action_type="integrate_lore_with_npcs",
        action_details={"npc_ids": npc_ids}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"], "approved": False}
    
    # Import LoreIntegrationSystem here to avoid circular imports
    from lore.lore_integration import LoreIntegrationSystem
    integration_system = LoreIntegrationSystem(user_id, conversation_id)
    
    # Create trace for monitoring
    with trace(
        workflow_name="Lore NPC Integration",
        trace_id=f"lore-npc-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Perform the integration
        results = await integration_system.integrate_lore_with_npcs(npc_ids)
    
    # Report the action
    await governor.process_agent_action_report(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action={
            "type": "integrate_lore_with_npcs",
            "description": f"Integrated lore with {len(npc_ids)} NPCs"
        },
        result={
            "npcs_integrated": len(results)
        }
    )
    
    return results

async def generate_scene_description_with_lore_and_governance(
    user_id: int,
    conversation_id: int,
    location: str
) -> Dict[str, Any]:
    """
    Generate a scene description enhanced with lore with full Nyx governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        location: Location name
        
    Returns:
        Enhanced scene description
    """
    # Create lore context
    lore_context = LoreAgentContext(user_id, conversation_id)
    await lore_context.initialize()
    
    # Get governor for permission check
    governor = lore_context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action_type="generate_scene_with_lore",
        action_details={"location": location}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"], "approved": False}
    
    # Import LoreIntegrationSystem here to avoid circular imports
    from lore.lore_integration import LoreIntegrationSystem
    integration_system = LoreIntegrationSystem(user_id, conversation_id)
    
    # Create trace for monitoring
    with trace(
        workflow_name="Lore Scene Generation",
        trace_id=f"lore-scene-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Generate the scene
        scene = await integration_system.generate_scene_description_with_lore(location)
    
    # Report the action
    await governor.process_agent_action_report(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="lore_generator",
        action={
            "type": "generate_scene_with_lore",
            "description": f"Generated lore-enhanced scene for location: {location}"
        },
        result={
            "location": location,
            "has_enhanced_description": "enhanced_description" in scene
        }
    )
    
    return scene

# -------------------------------------------------------------------------------
# Directive Handling
# -------------------------------------------------------------------------------

async def process_lore_directive(directive_data: Dict[str, Any], user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Process a directive from Nyx governance system.
    
    Args:
        directive_data: The directive data
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Result of processing the directive
    """
    # Create lore context
    lore_context = LoreAgentContext(user_id, conversation_id)
    await lore_context.initialize()
    
    # Initialize directive handler if needed
    if not lore_context.directive_handler:
        lore_context.directive_handler = DirectiveHandler(
            user_id, 
            conversation_id, 
            AgentType.NARRATIVE_CRAFTER,
            "lore_generator"
        )
        
    # Process the directive
    result = await lore_context.directive_handler._handle_action_directive(directive_data)
    
    return result

# -------------------------------------------------------------------------------
# Nyx Governance Registration
# -------------------------------------------------------------------------------

async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register lore agents with Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    governor = await get_central_governance(user_id, conversation_id)
    
    # Register all lore agents
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=foundation_lore_agent,
        agent_id="foundation_lore"
    )
    
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=factions_agent,
        agent_id="factions"
    )
    
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=cultural_agent,
        agent_id="cultural"
    )
    
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=history_agent,
        agent_id="history"
    )
    
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=locations_agent,
        agent_id="locations"
    )
    
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=quests_agent,
        agent_id="quests"
    )
    
    # Also register the setting analysis agent if available
    await governor.register_agent(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_instance=setting_analysis_agent,
        agent_id="setting_analysis"
    )
    
    # Issue a general directive for lore maintenance
    await governor.issue_directive(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="foundation_lore",  # Primary lore agent
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Maintain world lore consistency and integrate with other systems.",
            "scope": "global"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    logger.info(f"All lore agents registered with Nyx governance system for user {user_id}, conversation {conversation_id}")
