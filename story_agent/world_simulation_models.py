# story_agent/world_simulation_models.py

"""Pydantic models for Nyx Agent SDK - Single Source of Truth"""

from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel as _PydanticBaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

# ===== Agent-Safe Base Model with Complete Schema Sanitization =====

# ===== Canonical Agent-Safe Base =====

class AgentSafeModel(_PydanticBaseModel):
    """
    Canonical Pydantic v2 base model for Agents SDK strict mode:
    - extra="forbid" (reject undeclared fields)
    - arbitrary_types_allowed=True (lets us store SDK objects if needed)
    - removes additionalProperties/unevaluatedProperties in generated JSON schema
    - fixes 'required' to only include declared properties
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @classmethod
    def _sanitize_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        s = copy.deepcopy(schema)

        def strip(obj, path=""):
            if isinstance(obj, dict):
                obj.pop("additionalProperties", None)
                obj.pop("unevaluatedProperties", None)

                props = obj.get("properties")
                req = obj.get("required")
                if isinstance(props, dict) and isinstance(req, list):
                    obj["required"] = [k for k in req if k in props]
                elif req is not None and not isinstance(req, list):
                    obj.pop("required", None)

                for k, v in list(obj.items()):
                    strip(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    strip(item, f"{path}[{i}]")
            return obj

        return strip(s)

    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        return cls._sanitize_schema(schema)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        return cls._sanitize_schema(schema)

# Export canonical base for convenience/consistency
BaseModel = AgentSafeModel
StrictBaseModel = AgentSafeModel

# ===== Canonical Key-Value and helpers =====

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List[JsonScalar]]

class KeyValue(BaseModel):
    """Key-value pair used across tools for Agent compatibility"""
    key: str
    value: JsonValue

class KVPair(BaseModel):
    """Alias/compat wrapper for KeyValue when a different name is expected."""
    key: str
    value: JsonValue

class KVList(BaseModel):
    items: List[KVPair] = Field(default_factory=list)

def dict_to_kvlist(d: dict) -> KVList:
    return KVList(items=[KVPair(key=str(k), value=v) for k, v in d.items()])

def kvlist_to_dict(kv: KVList) -> dict:
    return {pair.key: pair.value for pair in kv.items}

def keyvalue_list_to_dict(kvs: List[KeyValue]) -> dict:
    return {kv.key: kv.value for kv in kvs}

def kvlist_from_obj(obj: Any) -> List[KVPair]:
    """Convert dict/list/scalar-like into List[KVPair] consistently."""
    if isinstance(obj, dict):
        return [KVPair(key=str(k), value=v) for k, v in obj.items()]
    if isinstance(obj, list):
        return [KVPair(key=str(i), value=v) for i, v in enumerate(obj)]
    return [KVPair(key="value", value=obj)]

def kvdict(items: List[KVPair]) -> Dict[str, Any]:
    """Convert List[KVPair] back to a dict."""
    return {it.key: it.value for it in (items or [])}

# ===== Core Enums =====

class TimeOfDay(Enum):
    """Time periods in the simulation"""
    EARLY_MORNING = "early_morning"  # 5-7 AM
    MORNING = "morning"               # 7-12 PM
    AFTERNOON = "afternoon"           # 12-5 PM
    EVENING = "evening"               # 5-9 PM
    NIGHT = "night"                   # 9 PM-12 AM
    LATE_NIGHT = "late_night"         # 12 AM-5 AM

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
    """
    Canonical power dynamics. Superset of all values used across modules
    (agent_interaction, preset_story_tracker, creative_task_agent, etc.).
    """
    SUBTLE_CONTROL = "subtle_control"
    GENTLE_GUIDANCE = "gentle_guidance"
    FIRM_DIRECTION = "firm_direction"
    CASUAL_DOMINANCE = "casual_dominance"
    PROTECTIVE_CONTROL = "protective_control"
    PLAYFUL_TEASING = "playful_teasing"
    INTIMATE_COMMAND = "intimate_command"
    SOCIAL_HIERARCHY = "social_hierarchy"
    RITUAL_SUBMISSION = "ritual_submission"
    FINANCIAL_CONTROL = "financial_control"
    PUBLIC_DISPLAY = "public_display"
    RESISTANCE = "resistance"
    NURTURING_DEPENDENCY = "nurturing_dependency"
    COLLABORATIVE = "collaborative"
    MANIPULATIVE = "manipulative"

class NarrativeTone(Enum):
    """Tone for slice-of-life narration"""
    CASUAL = "casual"
    INTIMATE = "intimate"
    OBSERVATIONAL = "observational"
    SENSUAL = "sensual"
    TEASING = "teasing"
    COMMANDING = "commanding"
    SUBTLE = "subtle"
    PSYCHOLOGICAL = "psychological"

class SceneFocus(Enum):
    """What to emphasize in scene narration"""
    ATMOSPHERE = "atmosphere"
    DIALOGUE = "dialogue"
    INTERNAL = "internal"
    DYNAMICS = "dynamics"
    ROUTINE = "routine"
    TENSION = "tension"

# ===== Time and Vitals Models =====

class CurrentTimeData(BaseModel):
    """Current time in the simulation"""
    year: int = 1
    month: int = 1
    day: int = 1
    hour: int = Field(12, ge=0, le=23)
    minute: int = Field(0, ge=0, le=59)
    time_of_day: Optional[Union[TimeOfDay, str]] = "morning"
    
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
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    player_name: Optional[str] = None
    last_update: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "energy": self.energy,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "fatigue": self.fatigue,
            "arousal": self.arousal
        }

# ===== Event Data Models =====

class AddictionCravingData(BaseModel):
    """Data describing an addiction craving"""
    addiction_type: Optional[str] = None
    intensity: float = Field(1.0, ge=0, le=1)
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
    intensity: float = Field(0.5, ge=0, le=1)

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

# ===== Processing Result Models =====

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

# ===== Core Simulation Models =====

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
    priority: float = Field(0.5, ge=0.0, le=1.0)
    power_dynamic: Optional[PowerDynamicType] = None
    choices: List[KeyValue] = Field(default_factory=list)  # Using KeyValue for Agent compatibility
    mood_impact: Optional[str] = None
    stat_impacts: List[KeyValue] = Field(default_factory=list)  # Using KeyValue for Agent compatibility
    addiction_triggers: List[str] = Field(default_factory=list)
    memory_tags: List[str] = Field(default_factory=list)

class PowerExchange(BaseModel):
    """A power exchange moment between entities"""
    initiator_npc_id: int
    initiator_type: str = "npc"
    initiator_id: int
    recipient_type: str = "player"
    recipient_id: int = 1
    exchange_type: PowerDynamicType
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    description: str = ""
    is_public: bool = False
    witnesses: List[int] = Field(default_factory=list)
    resistance_possible: bool = True
    player_response_options: List[str] = Field(default_factory=list)
    consequences: List[KeyValue] = Field(default_factory=list)

class WorldTension(BaseModel):
    """Current tension levels in the world"""
    overall_tension: float = Field(0.0, ge=0.0, le=1.0)
    social_tension: float = Field(0.0, ge=0.0, le=1.0)
    power_tension: float = Field(0.0, ge=0.0, le=1.0)
    sexual_tension: float = Field(0.0, ge=0.0, le=1.0)
    emotional_tension: float = Field(0.0, ge=0.0, le=1.0)
    addiction_tension: float = Field(0.0, ge=0.0, le=1.0)
    vital_tension: float = Field(0.0, ge=0.0, le=1.0)
    unresolved_conflicts: int = 0
    tension_sources: List[str] = Field(default_factory=list)
    
    def get_dominant_tension(self) -> Tuple[str, float]:
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
    player_submission_level: float = Field(0.0, ge=0.0, le=100.0)
    player_resistance_level: float = Field(50.0, ge=0.0, le=100.0)
    player_corruption_level: float = Field(0.0, ge=0.0, le=100.0)
    acceptance_level: float = Field(0.0, ge=0.0, le=100.0)
    dominant_npc_ids: List[int] = Field(default_factory=list)
    supportive_npc_ids: List[int] = Field(default_factory=list)
    adversarial_npc_ids: List[int] = Field(default_factory=list)
    intimate_npc_ids: List[int] = Field(default_factory=list)

class NPCRoutine(BaseModel):
    """NPC's daily routine"""
    npc_id: int
    npc_name: str
    schedule: List[KeyValue] = Field(default_factory=list)
    current_activity: Optional[str] = None
    current_location: Optional[str] = None
    location: str = "unknown"  # Alias for backwards compatibility
    availability: str = "available"  # "busy", "available", "interruptible"
    mood: Optional[str] = None
    power_tendency: Optional[str] = None  # From original models.py
    scheduled_events: List[KeyValue] = Field(default_factory=list)  # From original models.py
    relationship_state: Optional[List[KeyValue]] = None
    narrative_stage: Optional[str] = None
    mask_integrity: float = Field(100, ge=0, le=100)

# ===== Pattern Detection Models =====

class EmergentPattern(BaseModel):
    """An emergent pattern detected across systems"""
    pattern_type: str  # "memory", "relationship", "addiction", "stat", "rule"
    pattern_name: str
    confidence: float = Field(0.5, ge=0, le=1)
    evidence: List[KeyValue] = Field(default_factory=list)
    narrative_implications: Optional[str] = None
    predicted_outcomes: List[str] = Field(default_factory=list)

class NarrativeThread(BaseModel):
    """A narrative thread emerging from system interactions"""
    thread_id: str
    thread_type: str  # "corruption", "romance", "dependency", "resistance", etc.
    participants: List[KeyValue] = Field(default_factory=list)
    current_stage: str
    tension_level: float = Field(0.5, ge=0, le=1)
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

# ===== Complete World State =====

class CompleteWorldState(BaseModel):
    """Complete world state with ALL system integrations"""
    # Time and Calendar
    current_time: CurrentTimeData = Field(default_factory=CurrentTimeData)
    calendar_names: List[KeyValue] = Field(default_factory=list)
    calendar_events: List[KeyValue] = Field(default_factory=list)
    
    # Vitals and Stats
    player_vitals: VitalsData = Field(default_factory=VitalsData)
    visible_stats: List[KeyValue] = Field(default_factory=list)
    hidden_stats: List[KeyValue] = Field(default_factory=list)
    active_stat_combinations: List[KeyValue] = Field(default_factory=list)
    stat_thresholds_active: List[KeyValue] = Field(default_factory=list)
    
    # Memory and Context
    recent_memories: List[KeyValue] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)
    active_flashbacks: List[KeyValue] = Field(default_factory=list)
    pending_reveals: List[KeyValue] = Field(default_factory=list)
    pending_dreams: List[KeyValue] = Field(default_factory=list)
    recent_revelations: List[KeyValue] = Field(default_factory=list)
    inner_monologues: List[str] = Field(default_factory=list)
    
    # Rules and Effects
    active_rules: List[KeyValue] = Field(default_factory=list)
    triggered_effects: List[KeyValue] = Field(default_factory=list)
    pending_effects: List[KeyValue] = Field(default_factory=list)
    
    # Inventory
    player_inventory: List[KeyValue] = Field(default_factory=list)
    recent_item_changes: List[KeyValue] = Field(default_factory=list)
    
    # NPCs and Relationships
    active_npcs: List[KeyValue] = Field(default_factory=list)
    npc_masks: List[KeyValue] = Field(default_factory=list)
    npc_narrative_stages: List[KeyValue] = Field(default_factory=list)
    npc_schedules: Optional[List[KeyValue]] = None
    relationship_states: List[KeyValue] = Field(default_factory=list)
    relationship_dynamics: RelationshipDynamics = Field(default_factory=RelationshipDynamics)
    relationship_overview: Optional[List[KeyValue]] = None
    pending_relationship_events: List[KeyValue] = Field(default_factory=list)
    
    # Addictions
    addiction_status: List[KeyValue] = Field(default_factory=list)
    active_cravings: List[KeyValue] = Field(default_factory=list)
    addiction_contexts: List[KeyValue] = Field(default_factory=list)
    
    # Currency
    player_money: int = 0
    currency_system: List[KeyValue] = Field(default_factory=list)
    recent_transactions: List[KeyValue] = Field(default_factory=list)
    
    # World State
    world_mood: WorldMood = WorldMood.RELAXED
    world_tension: WorldTension = Field(default_factory=WorldTension)
    tension_factors: List[KeyValue] = Field(default_factory=list)
    environmental_factors: List[KeyValue] = Field(default_factory=list)
    location_data: str = ""
    
    # Events - Using List[KeyValue] to avoid nested model issues
    ongoing_events: List[KeyValue] = Field(default_factory=list)
    available_activities: List[KeyValue] = Field(default_factory=list)
    event_history: List[KeyValue] = Field(default_factory=list)
    
    # Governance
    nyx_directives: List[KeyValue] = Field(default_factory=list)
    
    # Metadata
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        return self.model_dump(mode='json')

# Alias for backwards compatibility
WorldState = CompleteWorldState

# ===== Slice-of-Life Narration Models =====

class SliceOfLifeNarration(BaseModel):
    """Complete narration for a slice-of-life scene"""
    scene_description: str
    atmosphere: str
    tone: NarrativeTone
    focus: SceneFocus
    power_dynamic_hints: List[str] = Field(default_factory=list)
    sensory_details: List[str] = Field(default_factory=list)
    npc_observations: List[str] = Field(default_factory=list)
    internal_monologue: Optional[str] = None
    governance_approved: bool = True
    governance_notes: Optional[str] = None
    emergent_elements: List[KeyValue] = Field(default_factory=list)
    system_triggers: List[str] = Field(default_factory=list)
    context_aware: bool = True
    relevant_memories: List[str] = Field(default_factory=list)

class NPCDialogue(BaseModel):
    """Dialogue from an NPC in daily life"""
    npc_id: int
    npc_name: str
    dialogue: str
    tone: str
    subtext: str
    body_language: str
    power_dynamic: Optional[PowerDynamicType] = None
    requires_response: bool = False
    hidden_triggers: List[str] = Field(default_factory=list)
    memory_influence: Optional[str] = None
    governance_approved: bool = True
    context_informed: bool = False

class AmbientNarration(BaseModel):
    """Ambient narration for world atmosphere"""
    description: str
    focus: str
    intensity: float = 0.5
    affects_mood: bool = False
    reflects_systems: List[str] = Field(default_factory=list)

class PowerMomentNarration(BaseModel):
    """Narration for a power exchange moment"""
    setup: str
    moment: str
    aftermath: str
    player_feelings: str
    options_presentation: List[str]
    potential_consequences: List[KeyValue] = Field(default_factory=list)
    governance_tracking: List[KeyValue] = Field(default_factory=list)

class DailyActivityNarration(BaseModel):
    """Narration for routine daily activities"""
    activity: str
    description: str
    routine_with_dynamics: str
    npc_involvement: List[str] = Field(default_factory=list)
    subtle_control_elements: List[str] = Field(default_factory=list)
    emergent_variations: Optional[List[str]] = None

# ===== Core Data Models =====

class MemoryItem(BaseModel):
    id: Optional[str] = Field(None, description="Memory ID if available")
    text: str = Field(..., description="Memory text")
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score 0-1")
    tags: List[str] = Field(default_factory=list, description="Memory tags")

class EmotionalChanges(BaseModel):
    valence_change: float
    arousal_change: float
    dominance_change: float

class ScoreComponents(BaseModel):
    context: float
    emotional: float
    pattern: float
    relationship: float

class PerformanceNumbers(BaseModel):
    memory_mb: float
    cpu_percent: float
    avg_response_time: float
    success_rate: float

class ConflictItem(BaseModel):
    type: str
    severity: float
    description: str
    entities: Optional[List[str]] = None
    blocked_objectives: Optional[List[str]] = None

class InstabilityItem(BaseModel):
    type: str
    severity: float
    description: str
    recommendation: Optional[str] = None

class ActivityRec(BaseModel):
    name: str
    description: str
    requirements: List[str]
    duration: str
    intensity: str
    partner_id: Optional[str] = None

class RelationshipStateOut(BaseModel):
    trust: float
    power_dynamic: float
    emotional_bond: float
    interaction_count: int
    last_interaction: float
    type: str

class RelationshipChanges(BaseModel):
    trust: float
    power: float
    bond: float

class DecisionMetadata(BaseModel):
    data: KVList = Field(default_factory=KVList, description="Additional metadata")

class DecisionOption(BaseModel):
    id: str = Field(..., description="Option ID")
    description: str = Field(..., description="Option description")
    metadata: DecisionMetadata = Field(default_factory=DecisionMetadata)

class ScoredOption(BaseModel):
    option: DecisionOption
    score: float
    components: ScoreComponents
    is_fallback: Optional[bool] = False

# ===== Structured Output Models =====

class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    universal_updates: Optional[KVList] = Field(None, description="Universal updates extracted from narrative")
    world_mood: Optional[str] = Field(None, description="Current world mood")
    ongoing_events: Optional[List[str]] = Field(None, description="Active slice-of-life events")
    available_activities: Optional[List[str]] = Field(None, description="Available player activities")
    npc_schedules: Optional[KVList] = None
    time_of_day: Optional[str] = Field(None, description="Current time period")
    emergent_opportunities: Optional[List[str]] = Field(None, description="Emergent narrative opportunities")

class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    topic: Optional[str] = Field(None, description="Topic of the reflection")

class ContentModeration(BaseModel):
    """Output for content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")
    
# ===== Input Models for Tools =====

class NarrateSliceOfLifeInput(BaseModel):
    """Input for narrating a slice-of-life scene"""
    scene_type: str = Field("routine", description="Type of scene to narrate")
    scene: Optional[SliceOfLifeEvent] = None
    world_state: Optional[WorldState] = None
    player_action: Optional[str] = None

class RetrieveMemoriesInput(BaseModel):
    query: str = Field(..., description="Search query to find memories")
    limit: int = Field(5, description="Maximum number of memories to return", ge=1, le=20)

class AddMemoryInput(BaseModel):
    memory_text: str = Field(..., description="The content of the memory")
    memory_type: str = Field("observation", description="Type of memory")
    significance: int = Field(5, description="Importance of memory (1-10)", ge=1, le=10)

class DetectUserRevelationsInput(BaseModel):
    user_message: str = Field(..., description="The user's message to analyze")

class GenerateImageFromSceneInput(BaseModel):
    scene_description: str = Field(..., description="Description of the scene")
    characters: List[str] = Field(..., description="List of characters in the scene")
    style: str = Field("realistic", description="Style for the image")

class CalculateEmotionalStateInput(BaseModel):
    context: KVList = Field(..., description="Current interaction context")

class UpdateRelationshipStateInput(BaseModel):
    entity_id: str = Field(..., description="ID of the entity (NPC or user)")
    trust_change: float = Field(0.0, description="Change in trust level", ge=-1.0, le=1.0)
    power_change: float = Field(0.0, description="Change in power dynamic", ge=-1.0, le=1.0)
    bond_change: float = Field(0.0, description="Change in emotional bond", ge=-1.0, le=1.0)

class GetActivityRecommendationsInput(BaseModel):
    scenario_type: str = Field(..., description="Type of current scenario")
    npc_ids: List[str] = Field(..., description="List of present NPC IDs")

class BeliefDataModel(BaseModel):
    entity_id: str = Field("nyx", description="Entity ID")
    type: str = Field("general", description="Belief type")
    content: KVList = Field(default_factory=KVList, description="Belief content")
    query: Optional[str] = Field(None, description="Query for belief search")

class ManageBeliefsInput(BaseModel):
    action: Literal["get", "update", "query"] = Field(..., description="Action to perform")
    belief_data: BeliefDataModel = Field(..., description="Data for the belief operation")

class ScoreDecisionOptionsInput(BaseModel):
    options: List[DecisionOption] = Field(..., description="List of possible decisions/actions")
    decision_context: KVList = Field(..., description="Context for making the decision")

class DetectConflictsAndInstabilityInput(BaseModel):
    scenario_state: KVList = Field(..., description="Current scenario state")

class GenerateUniversalUpdatesInput(BaseModel):
    narrative: str = Field(..., description="The narrative text to process")

class DecideImageInput(BaseModel):
    scene_text: str = Field(..., description="Scene description to evaluate")

class EmptyInput(BaseModel):
    """Empty input for functions that don't require parameters"""
    pass

# Open World / Slice-of-life Input Models
class NarrateSliceInput(BaseModel):
    scene_type: str = Field("routine", description="Slice-of-life scene type")

class EmergentEventInput(BaseModel):
    event_type: Optional[str] = Field(None, description="Optional event type hint")

class SimulateAutonomyInput(BaseModel):
    hours: int = Field(1, ge=1, le=24, description="Hours to advance")

# ===== Output Models for Tools =====

class MemorySearchResult(BaseModel):
    memories: List[MemoryItem] = Field(..., description="List of retrieved memories")
    formatted_text: str = Field(..., description="Formatted memory text")

class MemoryStorageResult(BaseModel):
    memory_id: str = Field(..., description="ID of stored memory")
    success: bool = Field(..., description="Whether memory was stored successfully")

class UserGuidanceResult(BaseModel):
    top_kinks: List[Tuple[str, float]] = Field(..., description="Top user preferences with levels")
    behavior_patterns: KVList = Field(..., description="Identified behavior patterns")
    suggested_intensity: float = Field(..., description="Suggested interaction intensity")
    reflections: List[str] = Field(..., description="User model reflections")

class RevelationDetectionResult(BaseModel):
    revelations: List[KVList] = Field(..., description="Detected revelations")
    has_revelations: bool = Field(..., description="Whether any revelations were found")

class ImageGenerationResult(BaseModel):
    success: bool = Field(..., description="Whether image was generated")
    image_url: Optional[str] = Field(None, description="URL of generated image")
    error: Optional[str] = Field(None, description="Error message if failed")

class EmotionalCalculationResult(BaseModel):
    valence: float = Field(..., description="New valence value")
    arousal: float = Field(..., description="New arousal value")
    dominance: float = Field(..., description="New dominance value")
    primary_emotion: str = Field(..., description="Primary emotion label")
    changes: EmotionalChanges = Field(..., description="Changes applied")
    state_updated: Optional[bool] = Field(None, description="Whether state was persisted")

class RelationshipUpdateResult(BaseModel):
    entity_id: str = Field(..., description="Entity ID")
    relationship: RelationshipStateOut = Field(..., description="Updated relationship state")
    changes: RelationshipChanges = Field(..., description="Changes applied")

class PerformanceMetricsResult(BaseModel):
    metrics: PerformanceNumbers = Field(..., description="Current performance metrics")
    suggestions: List[str] = Field(..., description="Performance improvement suggestions")
    actions_taken: List[str] = Field(..., description="Remediation actions taken")
    health: str = Field(..., description="Overall system health status")

class ActivityRecommendationsResult(BaseModel):
    recommendations: List[ActivityRec] = Field(..., description="Recommended activities")
    total_available: int = Field(..., description="Total number of available activities")

class BeliefManagementResult(BaseModel):
    result: Union[str, KVList] = Field(..., description="Operation result")
    error: Optional[str] = Field(None, description="Error message if failed")

class DecisionScoringResult(BaseModel):
    scored_options: List[ScoredOption] = Field(..., description="Options with scores")
    best_option: DecisionOption = Field(..., description="Highest scoring option")
    confidence: float = Field(..., description="Confidence in best option")

class ConflictDetectionResult(BaseModel):
    conflicts: List[ConflictItem] = Field(..., description="Detected conflicts")
    instabilities: List[InstabilityItem] = Field(..., description="Detected instabilities")
    overall_stability: float = Field(..., description="Overall stability score (0-1)")
    stability_note: str = Field(..., description="Explanation of stability score")
    requires_intervention: bool = Field(..., description="Whether intervention is needed")

class UniversalUpdateResult(BaseModel):
    success: bool = Field(..., description="Whether updates were generated")
    updates_generated: bool = Field(..., description="Whether any updates were found")
    error: Optional[str] = Field(None, description="Error message if failed")

# ===== State Models =====

class EmotionalState(BaseModel):
    valence: float = Field(0.0, description="Positive/negative emotion (-1 to 1)", ge=-1.0, le=1.0)
    arousal: float = Field(0.5, description="Emotional intensity (0 to 1)", ge=0.0, le=1.0)
    dominance: float = Field(0.7, description="Control level (0 to 1)", ge=0.0, le=1.0)

class RelationshipState(BaseModel):
    trust: float = Field(0.5, description="Trust level (0-1)", ge=0.0, le=1.0)
    power_dynamic: float = Field(0.5, description="Power dynamic (0-1)", ge=0.0, le=1.0)
    emotional_bond: float = Field(0.3, description="Emotional bond strength (0-1)", ge=0.0, le=1.0)
    interaction_count: int = Field(0, description="Number of interactions", ge=0)
    last_interaction: float = Field(..., description="Timestamp of last interaction")
    type: str = Field("neutral", description="Relationship type")

class EmotionalStateUpdate(BaseModel):
    """Structured output for emotional state changes"""
    valence: float = Field(..., description="Positive/negative emotion (-1 to 1)")
    arousal: float = Field(..., description="Emotional intensity (0 to 1)")
    dominance: float = Field(..., description="Control level (0 to 1)")
    primary_emotion: str = Field(..., description="Primary emotion label")
    reasoning: str = Field(..., description="Why the emotional state changed")

class ScenarioDecision(BaseModel):
    """Structured output for scenario management decisions"""
    action: str = Field(..., description="Action to take (advance, maintain, escalate, de-escalate)")
    next_phase: str = Field(..., description="Next scenario phase")
    tasks: List[KVList] = Field(default_factory=list, description="Tasks to execute")
    npc_actions: List[KVList] = Field(default_factory=list, description="NPC actions to take")
    time_advancement: bool = Field(False, description="Whether to advance time after this phase")

class RelationshipUpdate(BaseModel):
    """Structured output for relationship changes"""
    trust_change: float = Field(0.0, description="Change in trust level")
    power_dynamic_change: float = Field(0.0, description="Change in power dynamic")
    emotional_bond_change: float = Field(0.0, description="Change in emotional bond")
    relationship_type: str = Field(..., description="Type of relationship")

class ImageGenerationDecision(BaseModel):
    """Decision about whether to generate an image"""
    should_generate: bool = Field(..., description="Whether an image should be generated")
    score: float = Field(0.0, description="Confidence score for the decision")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    reasoning: str = Field(..., description="Reasoning for the decision")

class PerformanceMetrics(BaseModel):
    total_actions: int = Field(0, ge=0)
    successful_actions: int = Field(0, ge=0)
    failed_actions: int = Field(0, ge=0)
    response_times: List[float] = Field(default_factory=list)
    memory_usage: float = Field(0.0, ge=0.0)
    cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    error_rates: KVList = Field(default_factory=lambda: KVList(items=[
        KVPair(key="total", value=0),
        KVPair(key="recovered", value=0),
        KVPair(key="unrecovered", value=0)
    ]))

class LearningMetrics(BaseModel):
    pattern_recognition_rate: float = Field(0.0, ge=0.0, le=1.0)
    strategy_improvement_rate: float = Field(0.0, ge=0.0, le=1.0)
    adaptation_success_rate: float = Field(0.0, ge=0.0, le=1.0)

# ===== Composite Models =====

class ScenarioManagementRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    scenario_id: Optional[str] = Field(None, description="Scenario ID")
    type: str = Field("general", description="Scenario type")
    participants: List[KVList] = Field(default_factory=list, description="Scenario participants")
    objectives: List[KVList] = Field(default_factory=list, description="Scenario objectives")

class RelationshipInteractionData(BaseModel):
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    participants: List[KVList] = Field(..., description="Interaction participants")
    interaction_type: str = Field(..., description="Type of interaction")
    outcome: str = Field(..., description="Interaction outcome")
    emotional_impact: Optional[KVList] = Field(None, description="Emotional impact data")

# Rebuild forward references
for model in [KVPair, ScoredOption, DecisionOption]:
    model.model_rebuild()

# Export a function to validate schemas
def validate_model_schemas():
    """Validate that all models have Agent-safe schemas"""
    models_to_check = [
        SliceOfLifeEvent, PowerExchange, WorldTension, RelationshipDynamics, NPCRoutine,
        CompleteWorldState, SliceOfLifeNarration, NPCDialogue, AmbientNarration,
        PowerMomentNarration, DailyActivityNarration, NarrateSliceOfLifeInput,
        MemoryItem, NarrativeResponse, ImageGenerationDecision,
        EmergentPattern, NarrativeThread, MemorySimilarity, RelationshipPatternOut,
        AddictionPatternOut, StatPatternOut, RulePatternOut, EmergentPatternsResult,
        AddictionCravingData, DreamData, RevelationData, RelationshipImpact,
        InventoryChange, ChoiceData, ChoiceProcessingResult
    ]
    
    issues = []
    for model in models_to_check:
        try:
            schema = model.model_json_schema()
            
            def check_schema(obj, path=""):
                if isinstance(obj, dict):
                    if "additionalProperties" in obj or "unevaluatedProperties" in obj:
                        issues.append(f"{model.__name__}{path}: has additionalProperties/unevaluatedProperties")
                    
                    props = obj.get("properties", {})
                    req = obj.get("required", [])
                    if isinstance(req, list) and isinstance(props, dict):
                        invalid = [k for k in req if k not in props]
                        if invalid:
                            issues.append(f"{model.__name__}{path}: invalid required fields {invalid}")
                    
                    for k, v in obj.items():
                        check_schema(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_schema(item, f"{path}[{i}]")
            
            check_schema(schema)
        except Exception as e:
            issues.append(f"{model.__name__}: schema generation failed - {e}")
    
    if issues:
        for issue in issues:
            logger.error(f"Schema validation issue: {issue}")
        raise ValueError(f"Schema validation failed with {len(issues)} issues")
    
    logger.info(f"All {len(models_to_check)} models have valid Agent-safe schemas")

# Alias for compatibility
strict_output = lambda model_cls: model_cls

__all__ = [
    # Base classes
    'BaseModel', 'StrictBaseModel', 'AgentSafeModel',
    
    # Enums
    'TimeOfDay', 'WorldMood', 'ActivityType', 'PowerDynamicType',
    'NarrativeTone', 'SceneFocus',
    
    # Key-Value helpers
    'KeyValue', 'KVPair', 'KVList', 'KVItem',
    'dict_to_kvlist', 'kvlist_to_dict', 'keyvalue_list_to_dict',
    'kvlist_from_obj', 'kvdict',
    
    # Time and Vitals
    'CurrentTimeData', 'VitalsData',
    
    # Event Data Models
    'AddictionCravingData', 'DreamData', 'RevelationData',
    'RelationshipImpact', 'InventoryChange', 'ChoiceData',
    'ChoiceProcessingResult',
    
    # Core Simulation Models
    'SliceOfLifeEvent', 'PowerExchange', 'WorldTension',
    'RelationshipDynamics', 'NPCRoutine',
    
    # Pattern Detection
    'EmergentPattern', 'NarrativeThread', 'MemorySimilarity',
    'RelationshipPatternOut', 'AddictionPatternOut', 'StatPatternOut',
    'RulePatternOut', 'EmergentPatternsResult',
    
    # World State
    'CompleteWorldState', 'WorldState',
    
    # Narration Models
    'SliceOfLifeNarration', 'NPCDialogue', 'AmbientNarration',
    'PowerMomentNarration', 'DailyActivityNarration',
    
    # Core Data Models
    'MemoryItem', 'EmotionalChanges', 'ScoreComponents',
    'PerformanceNumbers', 'ConflictItem', 'InstabilityItem',
    'ActivityRec', 'RelationshipStateOut', 'RelationshipChanges',
    'DecisionMetadata', 'DecisionOption', 'ScoredOption',
    
    # Structured Output Models
    'NarrativeResponse', 'MemoryReflection', 'ContentModeration',
    'EmotionalStateUpdate', 'ScenarioDecision', 'RelationshipUpdate',
    'ImageGenerationDecision',
    
    # Input Models
    'NarrateSliceOfLifeInput', 'RetrieveMemoriesInput', 'AddMemoryInput',
    'DetectUserRevelationsInput', 'GenerateImageFromSceneInput',
    'CalculateEmotionalStateInput', 'UpdateRelationshipStateInput',
    'GetActivityRecommendationsInput', 'BeliefDataModel', 'ManageBeliefsInput',
    'ScoreDecisionOptionsInput', 'DetectConflictsAndInstabilityInput',
    'GenerateUniversalUpdatesInput', 'DecideImageInput', 'EmptyInput',
    'NarrateSliceInput', 'EmergentEventInput', 'SimulateAutonomyInput',
    
    # Output Models
    'MemorySearchResult', 'MemoryStorageResult', 'UserGuidanceResult',
    'RevelationDetectionResult', 'ImageGenerationResult',
    'EmotionalCalculationResult', 'RelationshipUpdateResult',
    'PerformanceMetricsResult', 'ActivityRecommendationsResult',
    'BeliefManagementResult', 'DecisionScoringResult',
    'ConflictDetectionResult', 'UniversalUpdateResult',
    
    # State Models
    'EmotionalState', 'RelationshipState', 'PerformanceMetrics', 'LearningMetrics',
    
    # Composite Models
    'ScenarioManagementRequest', 'RelationshipInteractionData',
    
    # Utility functions
    'strict_output', 'validate_model_schemas',
]

