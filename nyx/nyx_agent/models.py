# nyx/nyx_agent/models.py
"""Unified Pydantic models for Nyx Agent SDK
- Harmonized to match tools.py (source of truth)
- Back-compat aliases preserved for older names used elsewhere
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple, Union, Literal
import logging

from pydantic import BaseModel as _PydanticBaseModel, Field

from story_agent.world_simulation_models import (
    AgentSafeModel,
    KVList,
    NarrativeResponse,          # we'll alias as NyxResponse
    WorldState as _WorldState,  # already an alias of CompleteWorldState
    MemoryItem as _MemoryItem,  # good stand-in for MemoryHighlight
    SliceOfLifeEvent as _SliceOfLifeEvent,  # good stand-in for EmergentEvent
    NPCDialogue as _NPCDialogue,
    ChoiceData as _ChoiceData,  # good stand-in for Choice
)

# Bring in the scene/context types you already defined
from nyx.nyx_agent.context import (
    ContextBundle as _ContextBundle,
    SceneScope as _SceneScope,
)

# ── Aliases expected by assembly.py ────────────────────────────────────────────
NyxResponse    = NarrativeResponse
WorldState     = _WorldState
MemoryHighlight = _MemoryItem
EmergentEvent  = _SliceOfLifeEvent
NPCDialogue    = _NPCDialogue
Choice         = _ChoiceData
ContextBundle  = _ContextBundle
SceneScope     = _SceneScope

# ── Minimal metadata container (assembly expects a symbol named BundleMetadata)
class BundleMetadata(AgentSafeModel):
    """Lightweight bundle metadata; stays agent-safe."""
    schema_version: Optional[int] = None
    fetch_time: Optional[float] = None
    link_hints: Dict[str, List[Union[int, str]]] = Field(default_factory=dict)
    expanded_sections: List[str] = Field(default_factory=list)
    extras: KVList = Field(default_factory=KVList)

logger = logging.getLogger(__name__)

# =========================
# Base & Schema Sanitization
# =========================

def _sanitize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
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
    """Pydantic base with JSON-schema cleanup to avoid 'additionalProperties' issues."""
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        try:
            schema = super().model_json_schema(*args, **kwargs)
            if 'additionalProperties' in str(schema):
                logger.debug("Model %s exported schema with additionalProperties", cls.__name__)
            return _sanitize_json_schema(schema)
        except Exception as e:
            logger.error("Schema generation failed for %s: %s", cls.__name__, e)
            raise


StrictBaseModel = BaseModel  # compatibility alias

# =============
# Utility Types
# =============

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List[JsonScalar]]

class KVPair(BaseModel):
    key: str
    value: JsonValue

class KVList(BaseModel):
    items: List[KVPair] = Field(default_factory=list)

def dict_to_kvlist(d: dict) -> KVList:
    return KVList(items=[KVPair(key=k, value=v) for k, v in d.items()])

def kvlist_to_dict(kv: KVList) -> dict:
    return {pair.key: pair.value for pair in kv.items}

def strict_output(model_cls):
    """Return the Pydantic model class as-is (hook for SDK strict outputs)."""
    import inspect
    if not inspect.isclass(model_cls):
        raise TypeError("strict_output expects a Pydantic model class")
    try:
        _ = model_cls.model_json_schema()
    except Exception:
        logger.debug("strict_output: schema build skipped for %r", model_cls, exc_info=True)
    return model_cls

# =====================
# Core domain structures
# =====================

class MemoryItem(BaseModel):
    id: Optional[str] = Field(None, description="Memory ID if available")
    text: str = Field(..., description="Memory text")
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score 0-1")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    entities: Optional[List[Union[str, int]]] = Field(None, description="Linked entity IDs (optional)")

class EmotionalState(BaseModel):
    """Simple VAD state used by tools."""
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Positive/negative emotion")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Emotional intensity")
    dominance: float = Field(0.7, ge=0.0, le=1.0, description="Control level")

# =================
# Inputs (tools.py)
# =================

class EmptyInput(BaseModel):
    """Empty input for functions that don't require parameters"""
    pass

class RetrieveMemoriesInput(BaseModel):
    query: str = Field(..., description="Search query to find memories")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of memories to return")

class AddMemoryInput(BaseModel):
    memory_text: str = Field(..., description="The content of the memory")
    memory_type: str = Field("observation", description="Type of memory")
    significance: int = Field(5, ge=1, le=10, description="Importance of memory (1-10)")
    entities: Optional[List[Union[str, int]]] = Field(None, description="Linked entities for graph context")

class DecideImageGenerationInput(BaseModel):
    """Input for deciding whether to generate an image"""
    scene_text: str
    context: Optional[str] = None
    user_preference: Optional[str] = None

# For back-compat with older references:
class DecideImageInput(DecideImageGenerationInput):
    pass

class UpdateEmotionalStateInput(BaseModel):
    """Input for updating emotional state based on events"""
    triggering_event: str
    valence_change: float = 0.0
    arousal_change: float = 0.0
    dominance_change: float = 0.0

class UpdateRelationshipStateInput(BaseModel):
    """Tools expect trust/attraction/respect deltas + optional event"""
    entity_id: Union[str, int]
    trust_change: float = Field(0.0, ge=-1.0, le=1.0)
    attraction_change: float = Field(0.0, ge=-1.0, le=1.0)
    respect_change: float = Field(0.0, ge=-1.0, le=1.0)
    triggering_event: Optional[str] = None

class DecisionOption(BaseModel):
    id: str = Field(..., description="Option ID")
    description: str = Field(..., description="Option description")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ScoreDecisionOptionsInput(BaseModel):
    options: List[str] | List[DecisionOption] = Field(..., description="Options to score")
    decision_context: KVList = Field(default_factory=KVList, description="Context for making the decision")

# Optional inputs referenced elsewhere (compat)
class DetectUserRevelationsInput(BaseModel):
    user_message: str = Field(..., description="The user's message to analyze")

class GetActivityRecommendationsInput(BaseModel):
    scenario_type: str = Field("general", description="Type of current scenario")
    npc_ids: List[Union[str, int]] = Field(default_factory=list, description="List of present NPC IDs")

# ==================
# Outputs (tools.py)
# ==================

class MemorySearchResult(BaseModel):
    memories: List[MemoryItem] = Field(..., description="List of retrieved memories")
    formatted_text: str = Field(..., description="Formatted memory text")
    graph_connections: int = Field(0, description="How many memories had graph links")

class MemoryStorageResult(BaseModel):
    memory_id: str = Field(..., description="ID of stored memory")
    success: bool = Field(..., description="Whether memory was stored successfully")
    linked_entities: List[Union[str, int]] = Field(default_factory=list, description="Entities linked to the memory")

class UserGuidanceResult(BaseModel):
    """What the user-analysis tool returns in tools.py"""
    preferred_approach: str = Field(..., description="High-level guidance on approach")
    tone_suggestions: List[str] = Field(default_factory=list)
    topics_to_explore: List[str] = Field(default_factory=list)
    boundaries_detected: List[str] = Field(default_factory=list)
    engagement_level: float = Field(0.5, ge=0.0, le=1.0)

class ImageGenerationDecision(BaseModel):
    """Decision about whether to generate an image (tools.py shape)"""
    should_generate: bool = Field(..., description="Whether an image should be generated")
    scene_score: float = Field(0.0, description="Heuristic score for the scene")
    prompt: Optional[str] = Field(None, description="Prompt to use if generating")
    reason: str = Field("", description="Reasoning")
    style_hints: List[str] = Field(default_factory=list)

    # Back-compat computed aliases (older code expected 'score'/'image_prompt'/'reasoning')
    @property
    def score(self) -> float:
        return self.scene_score

    @property
    def image_prompt(self) -> Optional[str]:
        return self.prompt

    @property
    def reasoning(self) -> str:
        return self.reason

class EmotionalStateResult(BaseModel):
    """tools.calculate_and_update_emotional_state expected shape"""
    valence: float
    arousal: float
    dominance: float
    emotional_label: str = Field("neutral")
    manifestation: str = Field("", description="How it shows up")

# Back-compat alias used by some older code
class EmotionalCalculationResult(EmotionalStateResult):
    pass

class RelationshipUpdate(BaseModel):
    """tools.update_relationship_state expected shape"""
    entity_id: Union[str, int]
    new_trust: float
    new_attraction: float
    new_respect: float
    relationship_level: str = Field("neutral")
    change_description: str = Field("")

# Compatibility wrapper used by some legacy paths
class RelationshipStateOut(BaseModel):
    trust: float
    power_dynamic: Optional[float] = None
    emotional_bond: Optional[float] = None
    interaction_count: Optional[int] = 0
    last_interaction: Optional[float] = 0.0
    type: str = "neutral"

class RelationshipChanges(BaseModel):
    trust: float = 0.0
    power: float = 0.0
    bond: float = 0.0

class RelationshipUpdateResult(BaseModel):
    entity_id: Union[str, int]
    relationship: RelationshipStateOut
    changes: RelationshipChanges

class PerformanceMetrics(BaseModel):
    """tools.check_performance_metrics expected shape"""
    response_time_ms: int = 0
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_fetches: int = 0
    bundle_size_kb: int = 0
    sections_loaded: List[str] = Field(default_factory=list)

class ActivityRec(BaseModel):
    name: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    duration: str = "unknown"
    intensity: str = "medium"
    partner_id: Optional[Union[str, int]] = None

class ActivityRecommendations(BaseModel):
    recommendations: List[ActivityRec] = Field(default_factory=list)
    total_available: int = 0
    scenario_context: Dict[str, Any] = Field(default_factory=dict)

# Back-compat name (earlier file used ...Result)
class ActivityRecommendationsResult(ActivityRecommendations):
    pass

class DecisionScores(BaseModel):
    """Return type used by tools.score_decision_options"""
    options: Dict[str, float] = Field(default_factory=dict, description="option -> score")
    recommended: Optional[str] = None
    reasoning: Dict[str, Any] = Field(default_factory=dict)

# Back-compat name
class DecisionScoringResult(DecisionScores):
    pass

class ConflictDetection(BaseModel):
    """Return type used by tools.detect_conflicts_and_instability"""
    active_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    potential_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    tension_level: float = 0.5
    hot_spots: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

# Back-compat name
class ConflictDetectionResult(ConflictDetection):
    pass

# =======================
# Narrative / Misc (used)
# =======================

class NarrativeResponse(BaseModel):
    """Structured output sometimes used by higher layers (or tests)"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = None
    time_advancement: bool = False
    universal_updates: Optional[KVList] = None
    world_mood: Optional[str] = None
    ongoing_events: Optional[List[str]] = None
    available_activities: Optional[List[str]] = None
    npc_schedules: Optional[KVList] = None
    time_of_day: Optional[str] = None
    emergent_opportunities: Optional[List[str]] = None

class MemoryReflection(BaseModel):
    reflection: str
    confidence: float
    topic: Optional[str] = None

class ContentModeration(BaseModel):
    is_appropriate: bool
    reasoning: str
    suggested_adjustment: Optional[str] = None

# Optional shapes some code references (rare / compat)
class ScenarioDecision(BaseModel):
    action: str
    next_phase: str
    tasks: List[KVList] = Field(default_factory=list)
    npc_actions: List[KVList] = Field(default_factory=list)
    time_advancement: bool = False

# ============================
# Legacy/alt names kept around
# ============================

# Older inputs which some paths still import:
class GenerateImageFromSceneInput(BaseModel):
    scene_description: str
    characters: List[str]
    style: str = "realistic"

class CalculateEmotionalStateInput(BaseModel):
    context: KVList

class ManageBeliefsInput(BaseModel):
    action: Literal["get", "update", "query"]
    belief_data: KVList = Field(default_factory=KVList)

class DetectConflictsAndInstabilityInput(BaseModel):
    scenario_state: KVList = Field(default_factory=KVList)

class GenerateUniversalUpdatesInput(BaseModel):
    narrative: str

# Legacy "Result" wrappers that some code may still expect:
class ImageGenerationResult(BaseModel):
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None

class PerformanceMetricsResult(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    health: str = "unknown"

class UniversalUpdateResult(BaseModel):
    success: bool
    updates_generated: bool
    error: Optional[str] = None


__all__ = [
    "NyxResponse",
    "WorldState",
    "MemoryHighlight",
    "EmergentEvent",
    "NPCDialogue",
    "Choice",
    "ContextBundle",
    "SceneScope",
    "BundleMetadata",
]
