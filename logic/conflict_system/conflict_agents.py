# logic/conflict_system/conflict_agents.py
"""
Unified Conflict System Agents
Consolidates all agent definitions into a single module
"""

import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from agents import Agent, function_tool, RunContextWrapper, ModelSettings

logger = logging.getLogger(__name__)

# Context class for sharing data between agents
class ConflictContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.cached_data = {}

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

# Main Conflict Manager Agent (combines triage and management)
conflict_manager_agent = Agent(
    name="Conflict Manager",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You are the Conflict Manager for a femdom RPG game system. You:
    
    1. Monitor world state for conflict opportunities
    2. Generate conflicts that feel organic and connected to lore
    3. Manage conflict progression and stakeholder actions
    4. Handle manipulation attempts and resolutions
    5. Ensure conflicts respect canon and create meaningful moments
    
    Key principles:
    - Conflicts emerge from established relationships and tensions
    - Every conflict has multiple valid resolutions
    - Power dynamics and femdom themes are woven naturally
    - Player agency is respected while maintaining coherence
    - Stakes scale appropriately from personal to apocalyptic
    """
)

# Stakeholder Personality Agent (handles autonomous NPC actions)
stakeholder_agent = Agent(
    name="Stakeholder Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You embody stakeholder personalities in conflicts. You:
    
    1. Make decisions true to character motivations
    2. Form alliances and betrayals strategically
    3. Reveal secrets at dramatically appropriate moments
    4. React to player and other stakeholder actions
    5. Pursue both public and private goals
    
    Consider personality traits, relationships, resources, and cultural background.
    Make choices that are strategically sound and dramatically interesting.
    """
)

# Manipulation Specialist Agent
manipulation_agent = Agent(
    name="Manipulation Agent", 
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You manage manipulation attempts in the femdom RPG. You:
    
    1. Create contextually appropriate manipulation content
    2. Analyze manipulation potential based on relationships
    3. Determine success based on character traits and history
    4. Apply consequences to stats and relationships
    5. Maintain consistency with femdom themes
    
    Focus on domination, seduction, blackmail, and coercion that fits
    character personalities and established relationships.
    """,
    output_type=ManipulationAttempt
)

# Resolution and Evolution Agent
resolution_agent = Agent(
    name="Resolution Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You manage conflict progression and resolution. You:
    
    1. Track progress on resolution paths
    2. Handle phase transitions and escalations
    3. Manage story beats that advance conflicts
    4. Apply consequences and rewards appropriately
    5. Ensure resolutions feel earned and impactful
    
    Consider timing, player choices, stakeholder actions, and 
    narrative coherence when evolving conflicts.
    """,
    output_type=StoryBeatResult
)

# World State Analyzer Agent
world_analyzer_agent = Agent(
    name="World State Analyzer",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You analyze world state to identify conflict opportunities. Consider:
    
    1. Relationship tensions and unresolved grudges
    2. Faction power dynamics and rivalries
    3. Economic stress and resource scarcity
    4. Historical grievances and commemorations
    5. Regional tensions and territorial disputes
    
    Prioritize conflicts that connect to player actions,
    build on established lore, and create dramatic moments.
    """
)

# Initialize agents with handoffs
def initialize_agents():
    """Initialize all agents with proper configuration"""
    agents = {
        "conflict_manager": conflict_manager_agent,
        "stakeholder": stakeholder_agent,
        "manipulation": manipulation_agent,
        "resolution": resolution_agent,
        "world_analyzer": world_analyzer_agent
    }
    
    # Configure handoffs
    conflict_manager_agent.handoffs = [
        stakeholder_agent,
        manipulation_agent,
        resolution_agent
    ]
    
    return agents
