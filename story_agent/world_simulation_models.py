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
    """Complete world state with ALL system integrations"""
    # Time and Calendar
    current_time: CurrentTimeData
    calendar_names: Dict[str, Any] = Field(default_factory=dict)
    calendar_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Vitals and Stats
    player_vitals: VitalsData
    visible_stats: Dict[str, Any] = Field(default_factory=dict)
    hidden_stats: Dict[str, Any] = Field(default_factory=dict)
    active_stat_combinations: List[Dict[str, Any]] = Field(default_factory=list)
    stat_thresholds_active: Dict[str, Any] = Field(default_factory=dict)
    
    # Memory and Context
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)
    active_flashbacks: List[Dict[str, Any]] = Field(default_factory=list)
    pending_reveals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Dreams and Revelations
    pending_dreams: List[Dict[str, Any]] = Field(default_factory=list)
    recent_revelations: List[Dict[str, Any]] = Field(default_factory=list)
    inner_monologues: List[str] = Field(default_factory=list)
    
    # Rules and Effects
    active_rules: List[Dict[str, Any]] = Field(default_factory=list)
    triggered_effects: List[Dict[str, Any]] = Field(default_factory=list)
    pending_effects: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Inventory
    player_inventory: List[Dict[str, Any]] = Field(default_factory=list)
    recent_item_changes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # NPCs and Relationships
    active_npcs: List[Dict[str, Any]] = Field(default_factory=list)
    npc_masks: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    npc_narrative_stages: Dict[int, str] = Field(default_factory=dict)
    relationship_states: Dict[str, Any] = Field(default_factory=dict)
    relationship_overview: Optional[Dict[str, Any]] = None
    pending_relationship_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Addictions
    addiction_status: Dict[str, Any] = Field(default_factory=dict)
    active_cravings: List[Dict[str, Any]] = Field(default_factory=list)
    addiction_contexts: Dict[str, Any] = Field(default_factory=dict)
    
    # Currency
    player_money: int = 0
    currency_system: Dict[str, Any] = Field(default_factory=dict)
    recent_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # World State
    world_mood: WorldMood
    tension_factors: Dict[str, float] = Field(default_factory=dict)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    location_data: str = ""
    
    # Events
    ongoing_events: List[Dict[str, Any]] = Field(default_factory=list)
    available_activities: List[Dict[str, Any]] = Field(default_factory=list)
    event_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Governance
    nyx_directives: List[Dict[str, Any]] = Field(default_factory=list)
    
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="forbid")

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
