# story_agent/world_simulation_models.py

"""
Shared models and enums for the world simulation system.
This file contains ALL data structures used by world simulation components.
Single source of truth to avoid circular imports and model duplication.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone

# ===============================================================================
# Core Time/Vitals Models (imported from time_cycle for type consistency)
# ===============================================================================

class TimeOfDay(Enum):
    """Time periods in the simulation"""
    EARLY_MORNING = "early_morning"  # 5-7 AM
    MORNING = "morning"  # 7-12 PM
    AFTERNOON = "afternoon"  # 12-5 PM
    EVENING = "evening"  # 5-9 PM
    NIGHT = "night"  # 9 PM-12 AM
    LATE_NIGHT = "late_night"  # 12 AM-5 AM

class CurrentTimeData(BaseModel):
    """Current time in the simulation"""
    year: int = 2025
    month: int = 1
    day: int = 1
    hour: int = 12
    minute: int = 0
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    
    model_config = ConfigDict(extra="forbid")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "time_of_day": self.time_of_day.value
        }

class VitalsData(BaseModel):
    """Player vital statistics"""
    hunger: float = Field(ge=0, le=100, default=50)
    thirst: float = Field(ge=0, le=100, default=50)
    fatigue: float = Field(ge=0, le=100, default=30)
    arousal: float = Field(ge=0, le=100, default=0)
    
    model_config = ConfigDict(extra="forbid")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "fatigue": self.fatigue,
            "arousal": self.arousal
        }

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
# Event Data Models (moved from world_director_agent)
# ===============================================================================

class AddictionCravingData(BaseModel):
    """Data describing an addiction craving"""
    addiction_type: Optional[str] = None
    intensity: float = Field(ge=0, le=1, default=1.0)
    time_since_last: Optional[float] = None
    withdrawal_stage: Optional[int] = None
    
    model_config = ConfigDict(extra="forbid")

class DreamData(BaseModel):
    """Data describing a dream trigger"""
    dream_type: Optional[str] = None
    emotional_tone: Optional[str] = None
    symbolism: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class RevelationData(BaseModel):
    """Data describing a personal revelation"""
    topic: Optional[str] = None
    trigger: Optional[str] = None
    intensity: float = Field(ge=0, le=1, default=0.5)
    
    model_config = ConfigDict(extra="forbid")

class ChoiceData(BaseModel):
    """Comprehensive player choice information"""
    text: Optional[str] = None
    stat_impacts: Optional[Dict[str, float]] = None
    addiction_impacts: Optional[Dict[str, float]] = None
    npc_id: Optional[int] = None
    relationship_impacts: Optional[Dict[str, Any]] = None
    activity_type: Optional[str] = None
    intensity: Optional[float] = Field(None, ge=0, le=1)
    inventory_changes: Optional[List[Dict[str, Any]]] = None
    currency_change: Optional[float] = None
    time_passed: Optional[float] = None
    emotional_valence: Optional[float] = Field(None, ge=-1, le=1)
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Processing Result Models
# ===============================================================================

class ChoiceProcessingResult(BaseModel):
    """Structured result for complete player choice processing"""
    success: bool
    effects: List[Dict[str, Any]] = Field(default_factory=list)
    narratives: List[str] = Field(default_factory=list)
    stat_changes: Optional[Dict[str, Any]] = None
    new_thresholds: Optional[Dict[str, Any]] = None
    npc_stage_change: Optional[Dict[str, Any]] = None
    activity_result: Optional[Dict[str, Any]] = None
    triggered_rules: Optional[List[Dict[str, Any]]] = None
    currency: Optional[Dict[str, Any]] = None
    hunger_update: Optional[Dict[str, Any]] = None
    preferences_detected: Optional[Dict[str, Any]] = None
    narrative: Optional[str] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Core Simulation Models
# ===============================================================================

class SliceOfLifeEvent(BaseModel):
    """A slice-of-life event in the simulation"""
    event_type: ActivityType
    title: str
    description: str
    location: str = "unknown"
    involved_npcs: List[int] = Field(default_factory=list)
    duration_minutes: int = 30
    power_dynamic: Optional[PowerDynamicType] = None
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    mood_impact: Optional[str] = None
    stat_impacts: Dict[str, float] = Field(default_factory=dict)
    addiction_triggers: List[str] = Field(default_factory=list)
    memory_tags: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

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
    
    model_config = ConfigDict(extra="forbid")

class WorldTension(BaseModel):
    """Current tension levels in the world"""
    overall_tension: float = Field(ge=0, le=1, default=0.0)
    social_tension: float = Field(ge=0, le=1, default=0.0)
    power_tension: float = Field(ge=0, le=1, default=0.0)
    emotional_tension: float = Field(ge=0, le=1, default=0.0)
    addiction_tension: float = Field(ge=0, le=1, default=0.0)
    vital_tension: float = Field(ge=0, le=1, default=0.0)
    unresolved_conflicts: int = 0
    tension_sources: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class RelationshipDynamics(BaseModel):
    """Overall relationship dynamics in the world"""
    player_submission_level: float = Field(ge=0, le=100, default=0.0)
    player_resistance_level: float = Field(ge=0, le=100, default=50.0)
    player_corruption_level: float = Field(ge=0, le=100, default=0.0)
    dominant_npc_ids: List[int] = Field(default_factory=list)
    supportive_npc_ids: List[int] = Field(default_factory=list)
    adversarial_npc_ids: List[int] = Field(default_factory=list)
    intimate_npc_ids: List[int] = Field(default_factory=list)
    
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
    relationship_state: Optional[Dict[str, Any]] = None
    narrative_stage: Optional[str] = None
    mask_integrity: float = Field(ge=0, le=100, default=100)
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Complete World State (the main comprehensive model)
# ===============================================================================

class CompleteWorldState(BaseModel):
    """Complete world state with ALL system integrations"""
    
    # Time and Calendar
    current_time: CurrentTimeData = Field(default_factory=CurrentTimeData)
    calendar_names: Dict[str, Any] = Field(default_factory=dict)
    calendar_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Vitals and Stats
    player_vitals: VitalsData = Field(default_factory=VitalsData)
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
    world_mood: WorldMood = WorldMood.RELAXED
    tension_factors: Dict[str, float] = Field(default_factory=dict)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    location_data: str = ""
    
    # Events
    ongoing_events: List[Dict[str, Any]] = Field(default_factory=list)
    available_activities: List[Dict[str, Any]] = Field(default_factory=list)
    event_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Governance
    nyx_directives: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(extra="forbid")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        return self.model_dump(mode='json')

# ===============================================================================
# Simplified World State (for backwards compatibility)
# ===============================================================================

# Alias for backwards compatibility
WorldState = CompleteWorldState

# ===============================================================================
# Pattern Detection Models
# ===============================================================================

class EmergentPattern(BaseModel):
    """An emergent pattern detected across systems"""
    pattern_type: str  # "memory", "relationship", "addiction", "stat", "rule"
    pattern_name: str
    confidence: float = Field(ge=0, le=1)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    narrative_implications: Optional[str] = None
    predicted_outcomes: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class NarrativeThread(BaseModel):
    """A narrative thread emerging from system interactions"""
    thread_id: str
    thread_type: str  # "corruption", "romance", "dependency", "resistance", etc.
    participants: List[Dict[str, Any]] = Field(default_factory=list)
    current_stage: str
    tension_level: float = Field(ge=0, le=1)
    key_events: List[str] = Field(default_factory=list)
    potential_climax: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Export all models
# ===============================================================================

__all__ = [
    # Time and Vitals
    'TimeOfDay',
    'CurrentTimeData',
    'VitalsData',
    
    # Enums
    'WorldMood',
    'ActivityType',
    'PowerDynamicType',
    
    # Event Data
    'AddictionCravingData',
    'DreamData',
    'RevelationData',
    'ChoiceData',
    'ChoiceProcessingResult',
    
    # Core Models
    'SliceOfLifeEvent',
    'PowerExchange',
    'WorldTension',
    'RelationshipDynamics',
    'NPCRoutine',
    
    # World State
    'CompleteWorldState',
    'WorldState',  # Alias for backwards compatibility
    
    # Pattern Detection
    'EmergentPattern',
    'NarrativeThread',
]
