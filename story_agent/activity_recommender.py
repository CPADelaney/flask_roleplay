# story_agent/activity_recommender.py

"""
Activity Recommender for Open-World Slice-of-Life Simulation.
Recommends contextual daily activities with embedded power dynamics.
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from agents import Agent, Runner, ModelSettings, function_tool
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager

# Import world simulation types
from story_agent.world_simulation_models import (
    TimeOfDay, ActivityType, PowerDynamicType, WorldMood
)

logger = logging.getLogger(__name__)

# ===============================================================================
# Activity Types and Models
# ===============================================================================

class ActivityPriority(Enum):
    """Priority levels for activities"""
    VITAL = "vital"          # Hunger, thirst, exhaustion
    ROUTINE = "routine"      # Daily habits and patterns
    SOCIAL = "social"        # NPC-initiated or social obligations
    OPTIONAL = "optional"    # Leisure and personal choices
    EMERGENT = "emergent"    # Dynamically generated opportunities

class ActivityContext(BaseModel):
    """Context for activity recommendation"""
    time_of_day: TimeOfDay
    world_mood: WorldMood
    location: str
    available_npcs: List[int] = Field(default_factory=list)
    player_vitals: Dict[str, float] = Field(default_factory=dict)
    recent_activities: List[str] = Field(default_factory=list)
    power_tension: float = 0.0
    relationship_dynamics: Dict[str, float] = Field(default_factory=dict)

class RecommendedActivity(BaseModel):
    """A single recommended activity"""
    activity_id: str
    activity_type: ActivityType
    title: str
    description: str
    priority: ActivityPriority
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Participants and dynamics
    participating_npcs: List[int] = Field(default_factory=list)
    npc_roles: Dict[int, str] = Field(default_factory=dict)  # NPC ID -> role in activity
    power_dynamic: Optional[PowerDynamicType] = None
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    
    # Practical details
    estimated_duration: int  # minutes
    location: str
    prerequisites: List[str] = Field(default_factory=list)
    
    # Expected outcomes
    vital_impacts: Dict[str, float] = Field(default_factory=dict)
    relationship_impacts: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    tension_impacts: Dict[str, float] = Field(default_factory=dict)
    
    # Player agency
    can_refuse: bool = True
    refusal_consequences: List[str] = Field(default_factory=list)
    variations: List[str] = Field(default_factory=list)  # Different ways to do activity

class ActivityRecommendations(BaseModel):
    """Container for multiple activity recommendations"""
    recommendations: List[RecommendedActivity]
    context_summary: str
    time_pressure: Optional[str] = None  # "You're getting hungry", etc.
    emergent_opportunity: Optional[str] = None  # Special one-time opportunities

# ===============================================================================
# Function Tools for Data Gathering
# ===============================================================================

@function_tool
async def get_time_appropriate_activities(
    user_id: int,
    conversation_id: int,
    time_of_day: str
) -> List[Dict[str, Any]]:
    """Get activities appropriate for the current time of day"""
    
    # Time-based activity templates
    activities_by_time = {
        "early_morning": [
            {"type": "routine", "name": "morning_hygiene", "duration": 20},
            {"type": "routine", "name": "light_breakfast", "duration": 15},
            {"type": "leisure", "name": "morning_meditation", "duration": 15}
        ],
        "morning": [
            {"type": "routine", "name": "breakfast", "duration": 30},
            {"type": "work", "name": "work_tasks", "duration": 120},
            {"type": "social", "name": "morning_socializing", "duration": 45}
        ],
        "afternoon": [
            {"type": "routine", "name": "lunch", "duration": 45},
            {"type": "work", "name": "afternoon_work", "duration": 120},
            {"type": "leisure", "name": "afternoon_break", "duration": 30}
        ],
        "evening": [
            {"type": "routine", "name": "dinner", "duration": 60},
            {"type": "social", "name": "evening_socializing", "duration": 90},
            {"type": "leisure", "name": "relaxation", "duration": 60}
        ],
        "night": [
            {"type": "intimate", "name": "intimate_time", "duration": 60},
            {"type": "leisure", "name": "entertainment", "duration": 90},
            {"type": "routine", "name": "bedtime_routine", "duration": 30}
        ],
        "late_night": [
            {"type": "routine", "name": "sleep_preparation", "duration": 15},
            {"type": "special", "name": "late_night_encounter", "duration": 30}
        ]
    }
    
    base_activities = activities_by_time.get(time_of_day.lower(), [])
    
    # Add power dynamic variations
    enhanced_activities = []
    for activity in base_activities:
        variations = []
        
        if activity["type"] == "routine":
            variations = [
                {"dynamic": "subtle_control", "description": "with gentle guidance"},
                {"dynamic": "protective_control", "description": "with caring oversight"}
            ]
        elif activity["type"] == "social":
            variations = [
                {"dynamic": "social_hierarchy", "description": "with established dynamics"},
                {"dynamic": "playful_teasing", "description": "with light teasing"}
            ]
        elif activity["type"] == "intimate":
            variations = [
                {"dynamic": "intimate_command", "description": "with clear direction"},
                {"dynamic": "ritual_submission", "description": "following established patterns"}
            ]
        
        activity["variations"] = variations
        enhanced_activities.append(activity)
    
    return enhanced_activities

@function_tool
async def get_npc_availability_and_mood(
    user_id: int,
    conversation_id: int,
    npc_ids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """Get current availability and mood for NPCs"""
    
    npc_states = {}
    
    async with get_db_connection_context() as conn:
        for npc_id in npc_ids:
            # Get NPC data
            row = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty, closeness, trust, 
                       current_location, intensity
                FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, npc_id, user_id, conversation_id)
            
            if row:
                # Determine availability based on personality
                availability = "available"
                if row['dominance'] > 70:
                    availability = "commanding"  # Will make time for player
                elif row['closeness'] > 60:
                    availability = "eager"  # Wants to spend time
                elif row['intensity'] > 70:
                    availability = "focused"  # Has specific plans
                
                # Determine mood
                mood = "neutral"
                if row['intensity'] > 70:
                    mood = "intense"
                elif row['dominance'] > 60 and row['cruelty'] < 40:
                    mood = "caring_dominant"
                elif row['closeness'] > 70:
                    mood = "affectionate"
                elif row['cruelty'] > 60:
                    mood = "teasing"
                
                # Preferred activities based on personality
                preferred_activities = []
                if row['dominance'] > 60:
                    preferred_activities.extend(["control_activities", "decision_making"])
                if row['closeness'] > 60:
                    preferred_activities.extend(["intimate_activities", "quality_time"])
                if row['intensity'] > 60:
                    preferred_activities.extend(["focused_activities", "intense_interactions"])
                
                npc_states[npc_id] = {
                    "name": row['npc_name'],
                    "availability": availability,
                    "mood": mood,
                    "location": row['current_location'],
                    "preferred_activities": preferred_activities,
                    "dominance": row['dominance'],
                    "closeness": row['closeness']
                }
    
    return npc_states

@function_tool
async def analyze_player_needs(
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """Analyze player's current needs and state"""
    
    # Get vitals if available
    try:
        from logic.time_cycle import get_current_vitals
        vitals = await get_current_vitals(user_id, conversation_id)
        
        needs = {
            "urgent_needs": [],
            "moderate_needs": [],
            "comfort_level": 1.0
        }
        
        # Check urgent needs
        if vitals.hunger < 20:
            needs["urgent_needs"].append("food")
        if vitals.thirst < 20:
            needs["urgent_needs"].append("water")
        if vitals.fatigue > 80:
            needs["urgent_needs"].append("rest")
        
        # Check moderate needs
        if vitals.hunger < 40:
            needs["moderate_needs"].append("meal")
        if vitals.fatigue > 60:
            needs["moderate_needs"].append("break")
        
        # Calculate comfort level
        needs["comfort_level"] = (vitals.hunger + vitals.thirst + (100 - vitals.fatigue)) / 300.0
        
        return needs
    except:
        # Fallback if vitals system not available
        return {
            "urgent_needs": [],
            "moderate_needs": [],
            "comfort_level": 0.7
        }

@function_tool
async def get_established_routines(
    user_id: int,
    conversation_id: int,
    time_of_day: str
) -> List[Dict[str, Any]]:
    """Get established routine patterns for this time"""
    
    memory_manager = await get_memory_manager(user_id, conversation_id)
    
    # Search for routine memories
    routine_memories = await memory_manager.search_memories(
        query_text=f"routine {time_of_day} daily pattern",
        limit=5,
        memory_types=["routine", "pattern"],
        use_vector=True
    )
    
    routines = []
    for memory in routine_memories:
        content = memory.content if hasattr(memory, 'content') else str(memory)
        
        # Extract routine patterns
        if "morning" in content.lower() and time_of_day == "morning":
            routines.append({
                "pattern": "morning_routine",
                "description": content[:100],
                "strength": 0.7
            })
        elif "evening" in content.lower() and time_of_day == "evening":
            routines.append({
                "pattern": "evening_routine", 
                "description": content[:100],
                "strength": 0.6
            })
    
    return routines

# ===============================================================================
# Activity Recommendation Agent
# ===============================================================================

ACTIVITY_RECOMMENDER_PROMPT = """
You are the Activity Recommender for an open-world slice-of-life simulation with femdom themes.

Your role is to recommend contextually appropriate activities that:
1. Match the current time of day and world mood
2. Involve available NPCs naturally
3. Include subtle to explicit power dynamics based on relationships
4. Build on established routines and patterns
5. Address player needs (hunger, fatigue, etc.)
6. Create opportunities for emergent narratives

## RECOMMENDATION PRIORITIES:
1. VITAL: Critical needs (very low hunger/thirst, extreme fatigue)
2. ROUTINE: Established daily patterns that condition behavior
3. SOCIAL: NPC-initiated activities or social obligations
4. OPTIONAL: Leisure and personal choices
5. EMERGENT: Special one-time opportunities

## POWER DYNAMICS TO EMBED:
- Subtle Control: Small decisions made for the player
- Casual Dominance: Confident assertions in daily activities
- Protective Control: Restrictions "for your own good"
- Playful Teasing: Light humiliation or teasing
- Ritual Submission: Following established patterns
- Financial Control: Managing resources
- Social Hierarchy: Public dynamics
- Intimate Command: Direct orders in private

## ACTIVITY DESIGN:
Each activity should:
- Feel natural to the time and setting
- Have clear start/end conditions
- Allow for player agency within constraints
- Build relationships organically
- Potentially trigger power dynamics

Consider:
- Time pressure (hunger, appointments)
- NPC moods and availability
- Recent activities (avoid repetition)
- World tension levels
- Established routines

Return 3-5 recommendations ranked by priority and contextual fit.
"""

activity_recommender_agent = Agent(
    name="SliceOfLifeActivityRecommender",
    instructions=ACTIVITY_RECOMMENDER_PROMPT,
    output_type=ActivityRecommendations,
    model="gpt-5-nano",
    tools=[
        get_time_appropriate_activities,
        get_npc_availability_and_mood,
        analyze_player_needs,
        get_established_routines
    ],
    model_settings=ModelSettings(temperature=0.6)
)

# ===============================================================================
# Main Recommendation Function
# ===============================================================================

async def recommend_activities(
    user_id: int,
    conversation_id: int,
    context: Optional[ActivityContext] = None,
    num_recommendations: int = 4
) -> ActivityRecommendations:
    """
    Get contextual activity recommendations for the current world state.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        context: Optional activity context
        num_recommendations: Number of activities to recommend
        
    Returns:
        ActivityRecommendations with prioritized activities
    """
    
    # Build context if not provided
    if not context:
        # Get world state
        from story_agent.world_director_agent import WorldDirector
        director = WorldDirector(user_id, conversation_id)
        world_state = await director.get_world_state()
        
        # Build context from world state
        context = ActivityContext(
            time_of_day=world_state.current_time,
            world_mood=world_state.world_mood,
            location="current_location",
            available_npcs=[npc.npc_id for npc in world_state.active_npcs],
            power_tension=world_state.world_tension.power_tension,
            relationship_dynamics={
                "submission": world_state.relationship_dynamics.player_submission_level,
                "acceptance": world_state.relationship_dynamics.acceptance_level
            }
        )
    
    # Create prompt with context
    prompt = f"""
    Recommend {num_recommendations} activities for the current context:
    
    Time: {context.time_of_day.value}
    Mood: {context.world_mood.value}
    Available NPCs: {len(context.available_npcs)}
    Power Tension: {context.power_tension:.1f}
    Player Submission Level: {context.relationship_dynamics.get('submission', 0):.1f}
    
    Consider player needs, NPC availability, and opportunities for power dynamics.
    Prioritize based on urgency and contextual appropriateness.
    """
    
    # Run the agent
    result = await Runner.run(
        starting_agent=activity_recommender_agent,
        input=prompt
    )
    
    return result.final_output

# ===============================================================================
# Helper Functions
# ===============================================================================

async def get_quick_activity_suggestion(
    user_id: int,
    conversation_id: int,
    activity_type: Optional[ActivityType] = None
) -> Optional[RecommendedActivity]:
    """Get a single quick activity suggestion"""
    
    recommendations = await recommend_activities(
        user_id, 
        conversation_id,
        num_recommendations=1
    )
    
    if recommendations.recommendations:
        activity = recommendations.recommendations[0]
        
        # Filter by type if specified
        if activity_type and activity.activity_type != activity_type:
            return None
            
        return activity
    
    return None

async def get_npc_initiated_activity(
    user_id: int,
    conversation_id: int,
    npc_id: int
) -> Optional[RecommendedActivity]:
    """Get an activity initiated by a specific NPC"""
    
    # Get NPC personality
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, closeness, intensity
            FROM NPCStats
            WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
        """, npc_id, user_id, conversation_id)
    
    if not npc:
        return None
    
    # Determine activity based on NPC personality
    if npc['dominance'] > 70:
        activity_type = ActivityType.ROUTINE  # They decide routine
        power_dynamic = PowerDynamicType.CASUAL_DOMINANCE
    elif npc['closeness'] > 70:
        activity_type = ActivityType.INTIMATE
        power_dynamic = PowerDynamicType.INTIMATE_COMMAND
    else:
        activity_type = ActivityType.SOCIAL
        power_dynamic = PowerDynamicType.SUBTLE_CONTROL
    
    # Create custom recommendation
    activity = RecommendedActivity(
        activity_id=f"npc_initiated_{npc_id}_{int(time.time())}",
        activity_type=activity_type,
        title=f"Time with {npc['npc_name']}",
        description=f"{npc['npc_name']} has plans for you",
        priority=ActivityPriority.SOCIAL,
        confidence_score=0.8,
        participating_npcs=[npc_id],
        npc_roles={npc_id: "initiator"},
        power_dynamic=power_dynamic,
        intensity=npc['intensity'] / 100.0,
        estimated_duration=60,
        location="varies",
        can_refuse=npc['dominance'] < 80,
        refusal_consequences=["Disappointment", "Tension increase"] if npc['dominance'] > 60 else []
    )
    
    return activity

