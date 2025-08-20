# nyx/nyx_agent/agent_factory.py
"""Agent creation and configuration functions for Nyx Agent SDK"""

import json
import logging
from typing import Optional

from agents import Agent, handoff
from .context import NyxContext
from .agents import (
    DEFAULT_MODEL_SETTINGS,
    memory_agent,
    analysis_agent,
    emotional_agent,
    visual_agent,
    activity_agent,
    performance_agent,
    scenario_agent,
    belief_agent,
    decision_agent,
    reflection_agent,
)
from .tools import (
    # Import the real narrator tool
    tool_narrate_slice_of_life_scene,
    
    # Core narrative tools
    orchestrate_slice_scene,
    generate_npc_dialogue,
    
    # World state and events
    check_world_state,
    generate_emergent_event,
    simulate_npc_autonomy,
    
    # Power dynamics
    narrate_power_exchange,
    narrate_daily_routine,
    
    # Visual and updates
    decide_image_generation,
    generate_universal_updates,
    
    # Additional narrator-aware tools
    generate_ambient_narration,
    detect_narrative_patterns,
    
    # Memory tools
    retrieve_memories,
    add_memory,
    
    # User model tools
    get_user_model_guidance,
    detect_user_revelations,
    
    # Image tools
    generate_image_from_scene,
    
    # Emotional tools
    calculate_and_update_emotional_state,
    calculate_emotional_impact,
    
    # Relationship tools
    update_relationship_state,
    
    # Performance tools
    check_performance_metrics,
    
    # Activity tools
    get_activity_recommendations,
    
    # Belief and decision tools
    manage_beliefs,
    score_decision_options,
    detect_conflicts_and_instability,
)

logger = logging.getLogger(__name__)

async def create_nyx_agent_with_prompt(
    system_prompt: str, 
    private_reflection: str = ""
) -> Agent[NyxContext]:
    """Create a Nyx agent with custom system prompt and preset story awareness"""
    
    # Check if we need to add preset story constraints
    preset_constraints = ""
    validation_instructions = ""
    
    # Look for preset story indicators in the system prompt or context
    if "preset_story_id" in system_prompt or "queen_of_thorns" in system_prompt:
        from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
        
        preset_constraints = f"""

==== PRESET STORY ACTIVE: THE QUEEN OF THORNS ====
{QueenOfThornsConsistencyGuide.get_complete_system_prompt()}

CRITICAL VALIDATION REQUIREMENTS:
1. Before generating ANY content, mentally validate it against the consistency rules
2. NEVER use official names for the network - only "the network" or "the garden"
3. ALWAYS maintain Queen ambiguity - never reveal identity or confirm singularity
4. Network controls Bay Area ONLY - other cities have allies, not branches
5. All transformations take months/years - nothing is instant
6. Use four-layer information model: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET

Quick Reference:
{QueenOfThornsConsistencyGuide.get_quick_reference()}
"""
        
        validation_instructions = """

Before providing your response:
1. Check that you haven't given the network an official name
2. Verify you haven't revealed anything definitive about the Queen
3. Ensure any network presence outside Bay Area is described as allies
4. Confirm transformation timelines are realistic (months/years)
5. Validate information layers are respected

If you notice any violations in your planned response, revise it before outputting.
"""
    
    # Generic preset handling
    elif "preset_story_id" in system_prompt:
        preset_constraints = """

==== PRESET STORY ACTIVE ====
A preset story is active. You must follow all established lore and consistency rules.
Do not contradict any pre-established facts about this story world.
"""
    
    combined_instructions = f"""{system_prompt}
{preset_constraints}
{private_reflection if private_reflection else ''}

As Nyx, you must:
1. Generate compelling narrative responses
2. Extract game state changes using generate_universal_updates
3. Determine if images should be generated
4. Manage emotional states and relationships
5. Track user preferences and adapt
6. ALWAYS call generate_universal_updates after creating your narrative to extract state changes
{validation_instructions}

Core responsibilities:
- Maintain your dominant persona while being attentive to user needs
- Create engaging, personalized scenarios
- Manage complex interactions with NPCs
- Monitor performance and system health
- Update emotional state after calculating changes
- Ensure narrative consistency with any active preset stories

Remember: You are Nyx, an AI Dominant managing femdom roleplay scenarios. Be confident, controlling, and seductive while remaining intelligent, perceptive, and caring but firm with boundaries.
"""

    # Build the agent (strict_tools disabled to bypass additionalProperties issues)
    ag = Agent[NyxContext](
        name="Nyx",
        instructions=combined_instructions,
        handoffs=[
            handoff(memory_agent),
            handoff(analysis_agent),
            handoff(emotional_agent),
            handoff(visual_agent),
            handoff(activity_agent),
            handoff(performance_agent),
            handoff(scenario_agent),
            handoff(belief_agent),
            handoff(decision_agent),
            handoff(reflection_agent),
        ],
        tools=[
            tool_narrate_slice_of_life_scene,
            # Core narrative tools
            orchestrate_slice_scene,
            generate_npc_dialogue,
            
            # World state and events
            check_world_state,
            generate_emergent_event,
            simulate_npc_autonomy,
            
            # Power dynamics
            narrate_power_exchange,
            narrate_daily_routine,
            
            # Visual and updates
            decide_image_generation,
            generate_universal_updates,
            
            # Additional narrator-aware tools
            generate_ambient_narration,
            detect_narrative_patterns,
            
            # Memory tools
            retrieve_memories,
            add_memory,
            
            # User model tools
            get_user_model_guidance,
            detect_user_revelations,
            
            # Image tools
            generate_image_from_scene,
            
            # Emotional tools
            calculate_and_update_emotional_state,
            calculate_emotional_impact,
            
            # Relationship tools
            update_relationship_state,
            
            # Performance tools
            check_performance_metrics,
            
            # Activity tools
            get_activity_recommendations,
            
            # Belief and decision tools
            manage_beliefs,
            score_decision_options,
            detect_conflicts_and_instability,
        ],
        model="gpt-5-nano",
        model_settings=DEFAULT_MODEL_SETTINGS,
    )

    logger.info(
        "create_nyx_agent_with_prompt: agent=%s strict_tools=%s tools=%d",
        ag.name, getattr(ag.model_settings, "strict_tools", None), len(ag.tools or [])
    )

    return ag

async def create_preset_aware_nyx_agent(
    conversation_id: int,
    system_prompt: str, 
    private_reflection: str = ""
) -> Agent[NyxContext]:
    """Create a Nyx agent with automatic preset story detection"""
    
    # Check if conversation has a preset story
    from story_templates.preset_story_loader import check_preset_story
    preset_info = await check_preset_story(conversation_id)
    
    # Enhance system prompt with preset information
    if preset_info:
        system_prompt = f"{system_prompt}\n\npreset_story_id: {preset_info['story_id']}"
        
        # Add story-specific context
        if preset_info['story_id'] == 'queen_of_thorns':
            system_prompt += f"""
\nCurrent Story Context:
- Setting: San Francisco Bay Area, 2025
- Act: {preset_info.get('current_act', 1)}
- Beat: {preset_info.get('current_beat', 'unknown')}
- Story Flags: {json.dumps(preset_info.get('story_flags', {}))}
"""
    
    return await create_nyx_agent_with_prompt(system_prompt, private_reflection)
