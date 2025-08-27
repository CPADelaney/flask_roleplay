# nyx/nyx_agent/models.py
"""Pydantic models for Nyx Agent SDK"""

from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from pydantic import BaseModel as _PydanticBaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

# ===== Base Model with Schema Sanitization =====
def sanitize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove additionalProperties / unevaluatedProperties and normalize 'required'."""
    import copy
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            node.pop("unevaluatedProperties", None)

            props = node.get("properties")
            if isinstance(props, dict):
                req = node.get("required")
                if isinstance(req, list):
                    node["required"] = [k for k in req if k in props]
                elif req is not None:
                    node.pop("required", None)

            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s

class BaseModel(_PydanticBaseModel):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        try:
            schema = super().model_json_schema(*args, **kwargs)
            if 'additionalProperties' in str(schema):
                logger.warning(f"Model {cls.__name__} has additionalProperties in schema")
            return sanitize_json_schema(schema)
        except Exception as e:
            logger.error(f"Schema generation failed for {cls.__name__}: {e}")
            raise

StrictBaseModel = BaseModel  # Alias for compatibility

# ===== Utility Types =====
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List[JsonScalar]]

class KVPair(BaseModel):
    key: str
    value: JsonValue

class KVList(BaseModel):
    items: List[KVPair] = Field(default_factory=list)

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
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

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

# ===== Input Models for Tools =====
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

class DecideImageGenerationInput(BaseModel):
    """Input for deciding whether to generate an image"""
    scene_text: str
    context: Optional[str] = None
    user_preference: Optional[str] = None

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

# ===== Open World / Slice-of-life Models =====
class NarrateSliceInput(BaseModel):
    scene_type: str = Field("routine", description="Slice-of-life scene type")

class EmergentEventInput(BaseModel):
    event_type: Optional[str] = Field(None, description="Optional event type hint")

class SimulateAutonomyInput(BaseModel):
    hours: int = Field(1, ge=1, le=24, description="Hours to advance")

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
KVPair.model_rebuild()
ScoredOption.model_rebuild()
DecisionOption.model_rebuild()

# Helper functions for conversion
def dict_to_kvlist(d: dict) -> KVList:
    return KVList(items=[KVPair(key=k, value=v) for k, v in d.items()])

def kvlist_to_dict(kv: KVList) -> dict:
    return {pair.key: pair.value for pair in kv.items}

def strict_output(model_cls):
    """Return the Pydantic model class as-is."""
    import inspect
    if not inspect.isclass(model_cls):
        raise TypeError("strict_output expects a Pydantic model class")
    try:
        _ = model_cls.model_json_schema()
        logger.debug("strict_output: schema ready for %s", getattr(model_cls, "__name__", model_cls))
    except Exception:
        logger.debug("strict_output: schema build skipped for %r", model_cls, exc_info=True)
    return model_cls
