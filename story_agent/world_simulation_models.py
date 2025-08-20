# story_agent/world_simulation_models.py

"""
Shared models and enums for the world simulation system.
This file contains ALL data structures used by world simulation components.
Single source of truth to avoid circular imports and model duplication.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel as _PydanticBaseModel, Field, ConfigDict
from datetime import datetime, timezone

# ===============================================================================
# SANITIZED BASE MODEL - Prevents additionalProperties in JSON schemas
# ===============================================================================

def _strip_ap(obj, path=""):
    """Recursively strip additionalProperties and fix required arrays"""
    if isinstance(obj, dict):
        # Remove problematic fields
        obj.pop('additionalProperties', None)
        obj.pop('unevaluatedProperties', None)
        
        # Fix 'required' to only include actual properties at THIS level
        props = obj.get("properties", {})
        req = obj.get("required", [])
        
        if isinstance(req, list) and isinstance(props, dict):
            # Filter out any required fields that don't exist in properties
            valid_required = []
            for k in req:
                if k in props:
                    valid_required.append(k)
                else:
                    # Log when we remove invalid required fields
                    if path:
                        logger.debug(f"Removing invalid required field '{k}' from {path}")
            obj["required"] = valid_required
        
        # Recurse into nested structures
        if isinstance(props, dict):
            for prop_name, prop_value in props.items():
                _strip_ap(prop_value, f"{path}.properties.{prop_name}")
        
        # Also check other schema keywords that might contain schemas
        for key in ['items', 'allOf', 'anyOf', 'oneOf']:
            if key in obj:
                _strip_ap(obj[key], f"{path}.{key}")
                
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _strip_ap(item, f"{path}[{i}]")
    
    return obj

class BaseModel(_PydanticBaseModel):
    """Base model that ensures no additionalProperties/unevaluatedProperties anywhere."""
    model_config = ConfigDict()  # don't set extra=...

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This is the path TypeAdapter() uses (what the Agents SDK relies on).
        schema = handler(core_schema)
        return _strip_ap(schema)

    @classmethod
    def model_json_schema(cls, **kwargs):
        # Kept for completeness; some tools call this directly.
        schema = super().model_json_schema(**kwargs)
        return _strip_ap(schema)

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
    year: int = 1
    month: int = 1
    day: int = 1
    hour: int = Field(12, ge=0, le=23)
    minute: int = Field(0, ge=0, le=59)
    time_of_day: Union[TimeOfDay, str] = "morning"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "time_of_day": getattr(self.time_of_day, "value", self.time_of_day)
        }

class VitalsData(BaseModel):
    """Player vital statistics"""
    energy: int = Field(100, ge=0, le=100)
    hunger: int = Field(100, ge=0, le=100)
    thirst: int = Field(100, ge=0, le=100)
    fatigue: int = Field(0, ge=0, le=100)
    arousal: float = Field(0, ge=0, le=100)
    
    # Optional fields to match logic/time_cycle.py
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    player_name: Optional[str] = None
    last_update: Optional[datetime] = None
    
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
    PLAYFUL_TEASING = "playful_teasing"
    INTIMATE_COMMAND = "intimate_command"

# ===============================================================================
# Event Data Models (moved from world_director_agent)
# ===============================================================================

class AddictionCravingData(BaseModel):
    """Data describing an addiction craving"""
    addiction_type: Optional[str] = None
    intensity: float = Field(ge=0, le=1, default=1.0)
    time_since_last: Optional[float] = None
    withdrawal_stage: Optional[int] = None

class DreamData(BaseModel):
    """Data describing a dream trigger"""
    dream_type: Optional[str] = None
    emotional_tone: Optional[str] = None
    symbolism: List[str] = Field(default_factory=list)

class RevelationData(BaseModel):
    """Data describing a personal revelation"""
    topic: Optional[str] = None
    trigger: Optional[str] = None
    intensity: float = Field(ge=0, le=1, default=0.5)

class KVItem(BaseModel):
    """Key-value pair for flexible data passing"""
    key: str
    value: Any

def kvlist_from_obj(obj: Any) -> List[KVItem]:
    """Convert various objects to a list of KVItems"""
    if isinstance(obj, dict):
        return [KVItem(key=str(k), value=v) for k, v in obj.items()]
    if isinstance(obj, list):
        return [KVItem(key=str(i), value=v) for i, v in enumerate(obj)]
    return [KVItem(key="value", value=obj)]

def kvdict(items: List[KVItem]) -> Dict[str, Any]:
    """Convert list of KVItems back to dictionary"""
    return {it.key: it.value for it in (items or [])}

class RelationshipImpact(BaseModel):
    """Impact on a relationship from a choice"""
    npc_name: str
    impacts: List[KVItem] = Field(default_factory=list)

class InventoryChange(BaseModel):
    """Change to inventory from a choice"""
    action: Literal["add", "remove"]
    item_name: str
    description: Optional[str] = None
    effect: Optional[str] = None

class ChoiceData(BaseModel):
    """Data representing a player choice and its impacts"""
    text: Optional[str] = None
    stat_impacts: List[KVItem] = Field(default_factory=list)
    addiction_impacts: List[KVItem] = Field(default_factory=list)
    npc_id: Optional[int] = None
    relationship_impacts: List[RelationshipImpact] = Field(default_factory=list)
    activity_type: Optional[str] = None
    intensity: Optional[float] = 1.0
    inventory_changes: List[InventoryChange] = Field(default_factory=list)
    currency_change: Optional[float] = None
    time_passed: Optional[float] = 0.0
    emotional_valence: Optional[float] = 0.0

# ===============================================================================
# Processing Result Models
# ===============================================================================

class ChoiceProcessingResult(BaseModel):
    """Result of processing a player choice"""
    success: bool
    effects: List[List[KVItem]] = Field(default_factory=list)
    narratives: List[str] = Field(default_factory=list)
    stat_changes: Optional[List[KVItem]] = None
    new_thresholds: Optional[List[KVItem]] = None
    npc_stage_change: Optional[List[KVItem]] = None
    activity_result: Optional[List[KVItem]] = None
    triggered_rules: List[List[KVItem]] = Field(default_factory=list)
    currency: Optional[List[KVItem]] = None
    hunger_update: Optional[List[KVItem]] = None
    preferences_detected: Optional[List[KVItem]] = None
    narrative: Optional[str] = None
    error: Optional[str] = None

# ===============================================================================
# Core Simulation Models
# ===============================================================================

class SliceOfLifeEvent(BaseModel):
    """A slice-of-life event in the simulation"""
    id: Optional[str] = None
    event_type: ActivityType
    title: str
    description: str
    location: str = "unknown"
    participants: List[int] = Field(default_factory=list)  # NPC IDs
    involved_npcs: List[int] = Field(default_factory=list)  # Alias for compatibility
    duration_minutes: int = 30
    priority: float = Field(ge=0, le=1, default=0.5)
    power_dynamic: Optional[PowerDynamicType] = None
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    mood_impact: Optional[str] = None
    stat_impacts: Dict[str, float] = Field(default_factory=dict)
    addiction_triggers: List[str] = Field(default_factory=list)
    memory_tags: List[str] = Field(default_factory=list)

class PowerExchange(BaseModel):
    """A power exchange moment between entities"""
    initiator_npc_id: int  # For compatibility
    initiator_type: str = "npc"  # "npc" or "player"
    initiator_id: int
    recipient_type: str = "player"  # "npc" or "player"
    recipient_id: int = 1  # Default to player
    exchange_type: PowerDynamicType
    intensity: float = Field(ge=0, le=1)
    description: str = ""
    is_public: bool = False
    witnesses: List[int] = Field(default_factory=list)
    resistance_possible: bool = True
    player_response_options: List[str] = Field(default_factory=list)
    consequences: Dict[str, Any] = Field(default_factory=dict)

class WorldTension(BaseModel):
    """Current tension levels in the world"""
    overall_tension: float = Field(ge=0, le=1, default=0.0)
    social_tension: float = Field(ge=0, le=1, default=0.0)
    power_tension: float = Field(ge=0, le=1, default=0.0)
    sexual_tension: float = Field(ge=0, le=1, default=0.0)
    emotional_tension: float = Field(ge=0, le=1, default=0.0)
    addiction_tension: float = Field(ge=0, le=1, default=0.0)
    vital_tension: float = Field(ge=0, le=1, default=0.0)
    unresolved_conflicts: int = 0
    tension_sources: List[str] = Field(default_factory=list)
    
    def get_dominant_tension(self) -> tuple[str, float]:
        """Get the dominant tension type and its level"""
        tensions = {
            "social": self.social_tension,
            "power": self.power_tension,
            "sexual": self.sexual_tension,
            "emotional": self.emotional_tension,
            "addiction": self.addiction_tension,
            "vital": self.vital_tension
        }
        dominant = max(tensions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]

class RelationshipDynamics(BaseModel):
    """Overall relationship dynamics in the world"""
    player_submission_level: float = Field(ge=0, le=100, default=0.0)
    player_resistance_level: float = Field(ge=0, le=100, default=50.0)
    player_corruption_level: float = Field(ge=0, le=100, default=0.0)
    acceptance_level: float = Field(ge=0, le=100, default=0.0)
    dominant_npc_ids: List[int] = Field(default_factory=list)
    supportive_npc_ids: List[int] = Field(default_factory=list)
    adversarial_npc_ids: List[int] = Field(default_factory=list)
    intimate_npc_ids: List[int] = Field(default_factory=list)

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
    npc_schedules: Optional[Dict[str, Any]] = None
    relationship_states: Dict[str, Any] = Field(default_factory=dict)
    relationship_dynamics: RelationshipDynamics = Field(default_factory=RelationshipDynamics)
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
    world_tension: WorldTension = Field(default_factory=WorldTension)
    tension_factors: Dict[str, float] = Field(default_factory=dict)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    location_data: str = ""
    
    # Events
    ongoing_events: List[SliceOfLifeEvent] = Field(default_factory=list)
    available_activities: List[Dict[str, Any]] = Field(default_factory=list)
    event_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Governance
    nyx_directives: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
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

class NarrativeThread(BaseModel):
    """A narrative thread emerging from system interactions"""
    thread_id: str
    thread_type: str  # "corruption", "romance", "dependency", "resistance", etc.
    participants: List[Dict[str, Any]] = Field(default_factory=list)
    current_stage: str
    tension_level: float = Field(ge=0, le=1)
    key_events: List[str] = Field(default_factory=list)
    potential_climax: Optional[str] = None

class MemorySimilarity(BaseModel):
    """Similarity between two memories"""
    m1_index: int
    m2_index: int
    m1_excerpt: str
    m2_excerpt: str
    similarity: float

class RelationshipPatternOut(BaseModel):
    """Detected relationship pattern output"""
    npc: str
    patterns: List[str] = Field(default_factory=list)
    archetype: str = "unknown"

class AddictionPatternOut(BaseModel):
    """Detected addiction pattern output"""
    type: str
    level: int
    trajectory: Literal["escalating", "stable"]

class StatPatternOut(BaseModel):
    """Detected stat combination pattern output"""
    combination: str
    behaviors: List[str] = Field(default_factory=list)

class RulePatternOut(BaseModel):
    """Detected rule trigger pattern output"""
    rule: str
    frequency: int

class EmergentPatternsResult(BaseModel):
    """Result of checking all emergent patterns"""
    memory_patterns: List[MemorySimilarity] = Field(default_factory=list)
    relationship_patterns: List[RelationshipPatternOut] = Field(default_factory=list)
    addiction_patterns: List[AddictionPatternOut] = Field(default_factory=list)
    stat_patterns: List[StatPatternOut] = Field(default_factory=list)
    rule_patterns: List[RulePatternOut] = Field(default_factory=list)
    narrative_analysis: Optional[str] = None

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

    # Helper key/value types
    'KVItem',
    'kvlist_from_obj',
    'kvdict',

    # Relationship / Inventory helpers
    'RelationshipImpact',
    'InventoryChange',

    # Choice and results
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
    'MemorySimilarity',
    'RelationshipPatternOut',
    'AddictionPatternOut',
    'StatPatternOut',
    'RulePatternOut',
    'EmergentPatternsResult',
]
