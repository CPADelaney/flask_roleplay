# story_agent/world_simulation_models.py

"""
Shared models and enums for the world simulation system.
This file contains common data structures used by both world_director_agent
and world_simulation_agents to avoid circular imports.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# ===============================================================================
# Enums
# ===============================================================================

class WorldMood(Enum):
    """Overall mood/atmosphere of the world"""
    RELAXED = "relaxed"
    TENSE = "tense"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"
    OPPRESSIVE = "oppressive"
    CHAOTIC = "chaotic"
    EXHAUSTED = "exhausted"
    DESPERATE = "desperate"
    CORRUPTED = "corrupted"
    DREAMLIKE = "dreamlike"
    CRAVING = "craving"

class TimeOfDay(Enum):
    """Time periods in the simulation"""
    EARLY_MORNING = "early_morning"  # 5-7 AM
    MORNING = "morning"  # 7-12 PM
    AFTERNOON = "afternoon"  # 12-5 PM
    EVENING = "evening"  # 5-9 PM
    NIGHT = "night"  # 9 PM-12 AM
    LATE_NIGHT = "late_night"  # 12 AM-5 AM

class ActivityType(Enum):
    """Types of slice-of-life activities"""
    WORK = "work"
    SOCIAL = "social"
    LEISURE = "leisure"
    INTIMATE = "intimate"
    ROUTINE = "routine"
    SPECIAL = "special"
    ADDICTION = "addiction"
    VITAL = "vital"
    DREAM = "dream"
    REVELATION = "revelation"

class PowerDynamicType(Enum):
    """Types of power dynamics in interactions"""
    SUBTLE_CONTROL = "subtle_control"
    GENTLE_GUIDANCE = "gentle_guidance"
    FIRM_DIRECTION = "firm_direction"
    CASUAL_DOMINANCE = "casual_dominance"
    PROTECTIVE_CONTROL = "protective_control"
    MANIPULATIVE = "manipulative"
    NURTURING_DEPENDENCY = "nurturing_dependency"
    COLLABORATIVE = "collaborative"
    RESISTANCE = "resistance"

# ===============================================================================
# Base Models
# ===============================================================================

class SliceOfLifeEvent(BaseModel):
    """A slice-of-life event in the simulation"""
    event_type: ActivityType
    title: str
    description: str
    location: str
    involved_npcs: List[int] = Field(default_factory=list)
    duration_minutes: int = 30
    power_dynamic: Optional[PowerDynamicType] = None
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    mood_impact: Optional[str] = None
    stat_impacts: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class PowerExchange(BaseModel):
    """A power exchange moment between entities"""
    initiator_type: str  # "npc" or "player"
    initiator_id: int
    recipient_type: str  # "npc" or "player"
    recipient_id: int
    exchange_type: PowerDynamicType
    intensity: float = Field(ge=0, le=1)
    description: str
    is_public: bool = False
    witnesses: List[int] = Field(default_factory=list)
    resistance_possible: bool = True
    consequences: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class WorldTension(BaseModel):
    """Current tension levels in the world"""
    overall_tension: float = Field(ge=0, le=1, default=0.0)
    social_tension: float = Field(ge=0, le=1, default=0.0)
    power_tension: float = Field(ge=0, le=1, default=0.0)
    emotional_tension: float = Field(ge=0, le=1, default=0.0)
    unresolved_conflicts: int = 0
    
    model_config = {"extra": "forbid"}

class RelationshipDynamics(BaseModel):
    """Overall relationship dynamics in the world"""
    player_submission_level: float = Field(ge=0, le=100, default=0.0)
    player_resistance_level: float = Field(ge=0, le=100, default=50.0)
    dominant_npc_ids: List[int] = Field(default_factory=list)
    supportive_npc_ids: List[int] = Field(default_factory=list)
    adversarial_npc_ids: List[int] = Field(default_factory=list)
    
    model_config = {"extra": "forbid"}

class WorldState(BaseModel):
    """Current state of the simulated world"""
    current_time: TimeOfDay
    world_mood: WorldMood
    world_tension: WorldTension
    relationship_dynamics: RelationshipDynamics
    
    # Active elements
    active_npcs: List[Dict[str, Any]] = Field(default_factory=list)
    ongoing_events: List[SliceOfLifeEvent] = Field(default_factory=list)
    available_activities: List[ActivityType] = Field(default_factory=list)
    recent_power_exchanges: List[PowerExchange] = Field(default_factory=list)
    
    # Environmental factors
    location: str = "apartment"
    weather: Optional[str] = None
    special_conditions: List[str] = Field(default_factory=list)
    
    # Metadata
    last_update: datetime = Field(default_factory=datetime.now)
    tick_count: int = 0
    
    model_config = {"extra": "forbid"}

class NPCRoutine(BaseModel):
    """NPC's daily routine"""
    npc_id: int
    npc_name: str
    schedule: List[Dict[str, Any]] = Field(default_factory=list)
    current_activity: Optional[str] = None
    current_location: Optional[str] = None
    availability: str = "available"  # "busy", "available", "interruptible"
    mood: Optional[str] = None
    
    model_config = {"extra": "forbid"}

# Export all models
__all__ = [
    'WorldMood',
    'TimeOfDay', 
    'ActivityType',
    'PowerDynamicType',
    'SliceOfLifeEvent',
    'PowerExchange',
    'WorldTension',
    'RelationshipDynamics',
    'WorldState',
    'NPCRoutine'
]
