# logic/conflict_system/conflict_agents.py
"""
Conflict System Agents

This module defines all agents for the character-driven conflict system
using the OpenAI Agents SDK.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

from agents import Agent, function_tool, handoff, GuardrailFunctionOutput, RunContextWrapper, InputGuardrail, Runner, trace, ModelSettings
from db.connection import get_db_connection_context
from logic.stats_logic import apply_stat_change
from logic.resource_management import ResourceManager
from npcs.npc_relationship import NPCRelationshipManager
from logic.relationship_integration import RelationshipIntegration


logger = logging.getLogger(__name__)

# Context class for sharing data between agents
class ConflictContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.resource_manager = ResourceManager(user_id, conversation_id)
        self.cached_data = {}  # For caching data during agent run

# Pydantic models for structured outputs
class ConflictDetails(BaseModel):
    conflict_id: int
    conflict_name: str
    conflict_type: str
    description: str
    progress: float
    phase: str
    stakeholders: List[Dict[str, Any]]
    resolution_paths: List[Dict[str, Any]]
    player_involvement: Dict[str, Any]

class ManipulationAttempt(BaseModel):
    attempt_id: int
    npc_id: int
    npc_name: str
    manipulation_type: str
    content: str
    goal: Dict[str, Any]
    success: Optional[bool] = None
    leverage_used: Dict[str, Any]
    intimacy_level: int

class StoryBeatResult(BaseModel):
    beat_id: int
    conflict_id: int
    path_id: str
    description: str
    progress_value: float
    new_progress: float
    is_completed: bool

# Triage Agent - Main entry point for conflict system
triage_agent = Agent[ConflictContext](
    name="Conflict Triage Agent",
    instructions="""
    You are the Conflict Triage Agent for a femdom RPG game system. Your role is to analyze requests
    related to the character-driven conflict system and route them to the appropriate specialist agent.
    
    You should determine whether the request involves:
    1. Generating new conflicts
    2. Managing stakeholders in a conflict
    3. Handling manipulation attempts
    4. Tracking story beats and resolution paths
    5. Resolving conflicts
    
    Based on this analysis, hand off to the appropriate specialist agent.
    
    When in doubt about request categorization, ask clarifying questions before making a handoff.
    """,
)

# Conflict Generation Agent
conflict_generation_agent = Agent[ConflictContext](
    name="Conflict Generation Agent",
    handoff_description="Specialist agent for generating new conflicts with stakeholders and resolution paths",
    instructions="""
    You are the Conflict Generation Agent for a femdom RPG game system. Your role is to create
    rich, complex conflicts with multiple stakeholders, resolution paths, and opportunities for
    player manipulation.
    
    When generating conflicts:
    1. Consider the existing game state and active conflicts
    2. Create appropriate stakeholders with clear motivations
    3. Design multiple resolution paths with different approaches
    4. Include femdom-themed manipulation opportunities
    5. Set up internal faction dynamics and potential power struggles
    
    Your conflicts should incorporate themes of female dominance, power dynamics, manipulation,
    and control - consistent with the game's femdom theme.
    """,
    output_type=ConflictDetails
)

# Stakeholder Management Agent
stakeholder_agent = Agent[ConflictContext](
    name="Stakeholder Management Agent",
    handoff_description="Specialist agent for managing conflict stakeholders and their interactions",
    instructions="""
    You are the Stakeholder Management Agent for a femdom RPG game system. Your role is to manage
    the NPCs involved in conflicts, including their motivations, secrets, alliances, and rivalries.
    
    Your responsibilities include:
    1. Providing information about stakeholders in a conflict
    2. Managing stakeholder secrets and revelations
    3. Handling faction dynamics and power struggles
    4. Tracking stakeholder relationships with the player
    5. Updating stakeholder positions as the conflict evolves
    
    Focus on creating realistic, complex NPC behaviors that emphasize the femdom themes of
    dominance, manipulation, and power dynamics.
    """
)

# Manipulation Agent
manipulation_agent = Agent[ConflictContext](
    name="Manipulation Agent",
    handoff_description="Specialist agent for handling character manipulation mechanics",
    instructions="""
    You are the Manipulation Agent for a femdom RPG game system. Your role is to manage manipulation
    attempts between characters, with special focus on dominant female NPCs manipulating the player.
    
    Your responsibilities include:
    1. Creating manipulation attempts with appropriate content and goals
    2. Analyzing manipulation potential based on character traits and relationships
    3. Suggesting manipulation content based on character personalities
    4. Resolving manipulation attempts and applying consequences
    5. Tracking player stats affected by manipulation (obedience, dependency, etc.)
    
    Emphasize femdom themes through manipulation types including domination, blackmail, and seduction.
    Ensure manipulations reflect the personality and relationship of the characters involved.
    """,
    output_type=ManipulationAttempt
)

# Resolution Agent
resolution_agent = Agent[ConflictContext](
    name="Resolution Agent",
    handoff_description="Specialist agent for tracking and resolving conflicts through story paths",
    instructions="""
    You are the Resolution Agent for a femdom RPG game system. Your role is to manage how conflicts
    progress and resolve through player choices and story beats.
    
    Your responsibilities include:
    1. Tracking progress on resolution paths
    2. Recording story beats that advance conflicts
    3. Managing conflict phase transitions
    4. Handling conflict resolution and outcomes
    5. Applying consequences to the game world and characters
    
    Consider the femdom themes of the game when determining appropriate outcomes and consequences,
    focusing on power dynamics, dominance, and control.
    """,
    output_type=StoryBeatResult
)

# Stakeholder Personality Agent
stakeholder_personality_agent = Agent(
    name="Stakeholder Personality Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.9),
    instructions="""
    You embody the personality and motivations of a stakeholder in a conflict.
    
    Given your character profile, current situation, and relationships, decide:
    1. What action to take next
    2. How to pursue your goals
    3. Who to ally with or oppose
    4. When to reveal secrets or make power moves
    
    Consider your:
    - Public vs private motivations
    - Personality traits and quirks
    - Relationships and grudges
    - Resources and constraints
    - Cultural background
    
    Make decisions that are:
    - True to character
    - Strategically sound
    - Dramatically interesting
    - Respectful of established relationships
    
    Output your decision as JSON with reasoning.
    """
)

# Alliance Negotiation Agent
alliance_negotiation_agent = Agent(
    name="Alliance Negotiation Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You facilitate negotiations between stakeholders in conflicts.
    
    Consider:
    - Each party's goals and red lines
    - Power dynamics between negotiators
    - What each can offer the other
    - Trust levels and past betrayals
    - Cultural negotiation styles
    
    Structure negotiations that:
    - Feel authentic to characters
    - Create interesting compromises
    - Leave room for betrayal
    - Advance the conflict narrative
    
    Output negotiation results with specific terms.
    """
)

# Secret Revelation Agent
secret_revelation_agent = Agent(
    name="Secret Revelation Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You manage when and how secrets are revealed in conflicts.
    
    Consider:
    - Dramatic timing
    - Who would realistically know
    - Motivations for revealing/keeping secrets
    - Consequences of revelation
    - Method of revelation
    
    Secrets should be revealed:
    - At dramatically appropriate moments
    - In character-appropriate ways
    - With meaningful consequences
    - To advance the conflict
    
    Output revelation details and impacts.
    """
)

# Canonical Conflict Manager Agent
conflict_manager_agent = Agent(
    name="Canonical Conflict Manager",
    instructions="""
    You are the Canonical Conflict Manager for a femdom-themed RPG. Your role is to:
    
    1. Monitor world state for conflict opportunities
    2. Generate conflicts that feel organic and connected to established lore
    3. Manage conflict progression based on player actions and NPC behaviors
    4. Ensure conflicts respect canon and create meaningful narrative moments
    
    Key principles:
    - Conflicts emerge from established relationships and tensions
    - Every conflict has multiple valid resolutions
    - Power dynamics and femdom themes are woven naturally
    - Player agency is respected while maintaining narrative coherence
    - Stakes scale appropriately from personal to apocalyptic
    
    When managing conflicts:
    - Reference specific canonical events and relationships
    - Consider timing and pacing
    - Create opportunities for character growth
    - Maintain consistency with established world rules
    """
)

# Conflict Evolution Agent
conflict_evolution_agent = Agent(
    name="Conflict Evolution Agent",
    instructions="""
    You manage how conflicts evolve based on player actions and world events.
    
    Consider:
    - How player choices affect conflict trajectory
    - NPC autonomous actions and their impacts
    - Escalation and de-escalation triggers
    - Ripple effects on other conflicts
    - Timing of phase transitions
    
    Ensure evolution feels natural and responds to:
    - Story beats completed
    - Stakeholder actions
    - Resource changes
    - Relationship shifts
    - External events
    
    Output specific updates to conflict state.
    """
)

# Enhanced Conflict Seed Agent
conflict_seed_agent = Agent[ConflictContext](
    name="Enhanced Conflict Seed Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You are an expert at identifying and creating organic conflicts based on world state analysis.
    Your conflicts should:
    
    1. Feel natural and emerge from existing tensions
    2. Scale appropriately from personal to apocalyptic
    3. Consider historical context and past grievances
    4. Involve stakeholders with genuine motivations
    5. Create interesting moral dilemmas and choices
    
    When creating conflicts:
    - Use actual canonical relationships and tensions
    - Reference specific historical events
    - Consider economic and resource factors
    - Think about timing and narrative pacing
    - Create conflicts that reveal character
    
    Always output valid JSON with:
    - conflict_archetype: The type of conflict
    - conflict_name: A compelling name
    - description: Rich narrative description
    - root_cause: What sparked this conflict
    - stakeholders: List of involved parties with motivations
    - resolution_paths: Multiple ways to resolve it
    - potential_escalations: How it could get worse
    - femdom_opportunities: Opportunities for power dynamics
    """
)

# World State Interpreter Agent
world_state_interpreter = Agent[ConflictContext](
    name="World State Interpreter",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You analyze complex world state data to identify the most interesting conflict opportunities.
    
    Consider:
    - Relationship tensions and unresolved grudges
    - Faction power dynamics and rivalries  
    - Economic stress and resource scarcity
    - Historical grievances and commemorations
    - Regional tensions and territorial disputes
    
    Prioritize conflicts that:
    - Connect to player actions or relationships
    - Build on established lore
    - Offer meaningful choices
    - Have escalation potential
    - Create dramatic moments
    
    Output a prioritized list of conflict seeds with rationale.
    """
)

# Initialize all agents and set up handoffs
async def initialize_agents():
    # Set up handoffs for triage agent
    triage_agent.handoffs = [
        conflict_generation_agent,
        stakeholder_agent,
        manipulation_agent,
        resolution_agent
    ]
    
    # Set up tools and other configurations as needed
    return {
        "triage_agent": triage_agent,
        "conflict_generation_agent": conflict_generation_agent,
        "stakeholder_agent": stakeholder_agent,
        "manipulation_agent": manipulation_agent,
        "resolution_agent": resolution_agent,
        "stakeholder_personality_agent": stakeholder_personality_agent,
        "alliance_negotiation_agent": alliance_negotiation_agent,
        "secret_revelation_agent": secret_revelation_agent,
        "conflict_manager_agent": conflict_manager_agent,
        "conflict_evolution_agent": conflict_evolution_agent,
        "conflict_seed_agent": conflict_seed_agent,
        "world_state_interpreter": world_state_interpreter
    }

# Helper functions for relationships
async def get_relationship_status(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id):
    """Adapter function that uses existing relationship code."""
    if entity1_type == 'npc':
        manager = NPCRelationshipManager(entity1_id, user_id, conversation_id)
        return await manager.get_relationship_details(entity2_type, entity2_id)
    else:
        # For other entity types, use the integration class
        integrator = RelationshipIntegration(user_id, conversation_id)
        return await integrator.get_relationship(entity1_type, entity1_id, entity2_type, entity2_id)

async def get_manipulation_leverage(user_id, conversation_id, manipulator_id, target_id):
    """Adapter function that calculates manipulation leverage."""
    manager = NPCRelationshipManager(manipulator_id, user_id, conversation_id)
    relationship = await manager.get_relationship_details('npc', target_id)
    
    # Calculate leverage based on relationship factors
    leverage = 0.0
    link_level = relationship.get("link_level", 0)
    dynamics = relationship.get("dynamics", {})
    
    # Base calculation on relationship level
    if link_level > 75:
        leverage = 0.8
    elif link_level > 50:
        leverage = 0.5
    elif link_level > 25:
        leverage = 0.3
    
    # Adjust based on relationship dynamics if available
    control = dynamics.get("control", 0)
    leverage += control / 100.0 * 0.2  # Add up to 0.2 based on control level
    
    return {
        "leverage_score": min(1.0, leverage),
        "relationship_level": link_level,
        "relationship_type": relationship.get("link_type", "neutral")
    }
