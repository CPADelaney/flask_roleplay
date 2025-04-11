# logic/conflict_system/conflict_agents.py
"""
Conflict System Agents

This module defines the agent-based architecture for the character-driven conflict system
using the OpenAI Agents SDK.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

from agents import Agent, function_tool, handoff, GuardrailFunctionOutput, RunContextWrapper, InputGuardrail
from db.connection import get_db_connection_context
from logic.stats_logic import apply_stat_change
from logic.resource_management import ResourceManager
from npcs.npc_relationship import NPCRelationshipManager
from logic.relationship_integration import RelationshipIntegration


logger = logging.getLogger(__name__)

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
    # Handoffs will be defined later
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
        "resolution_agent": resolution_agent
    }
