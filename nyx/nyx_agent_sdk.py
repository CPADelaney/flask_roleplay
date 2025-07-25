# nyx/nyx_agent_sdk.py

"""
Nyx Agent SDK - Refactored to use OpenAI Agents SDK with Strict Typing

MODULARIZATION TODO: This file has grown to 2k+ lines and should be split:
- nyx/models.py - All Pydantic models and Config constants
- nyx/tools/*.py - Individual tool implementations
- nyx/agents/*.py - Agent definitions
- nyx/context.py - NyxContext and state management
- nyx/compat.py - Legacy AgentContext and compatibility layers

This module requires the following database tables to be created via migrations:
- NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
- scenario_states (user_id, conversation_id, state_data, created_at) 
  - INDEX: (user_id, conversation_id)
- learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
- performance_metrics (user_id, conversation_id, metrics, error_log, created_at)

For continuous monitoring (scenario updates, resource usage, etc.), implement an external service using:
- Celery for background tasks
- FastAPI background tasks
- Kubernetes CronJobs
- Or a dedicated monitoring service

This keeps the main request path fast and non-blocking.
"""

import logging
import json
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Literal
from dataclasses import dataclass, field
from contextlib import suppress
import statistics

from agents import (
    Agent, Runner, function_tool, handoff, 
    ModelSettings, GuardrailFunctionOutput, InputGuardrail,
    RunContextWrapper, RunConfig
)
from pydantic import BaseModel, Field, ConfigDict, ValidationError

from db.connection import get_db_connection_context
from memory.memory_nyx_integration import MemoryNyxBridge, get_memory_nyx_bridge
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.performance_monitor import PerformanceMonitor
from .response_filter import ResponseFilter
from nyx.core.sync.strategy_controller import get_active_strategies

logger = logging.getLogger(__name__)

# ===== Utility Types for Strict Schema =====

# Recursive JSON scalar/value type. This is for *values*, not dicts.
JsonValue = Union[str, int, float, bool, None, List["JsonValue"]]

class KVPair(BaseModel):
    """Key/value record to represent arbitrary dicts in a strict schema world."""
    model_config = ConfigDict(extra='forbid')
    key: str
    value: Any

class KVList(BaseModel):
    """Dict-like but strict: a list of KV pairs."""
    model_config = ConfigDict(extra='forbid')
    items: List[KVPair] = Field(default_factory=list)

# Resolve forward refs
KVPair.model_rebuild()

# Helpers for conversion
def dict_to_kvlist(d: dict) -> KVList:
    return KVList(items=[KVPair(key=k, value=v) for k, v in d.items()])

def kvlist_to_dict(kv: KVList) -> dict:
    return {pair.key: pair.value for pair in kv.items}

# ===== Constants and Configuration =====
class Config:
    """Configuration constants to avoid magic numbers"""
    # Entity types
    ENTITY_TYPE_INTEGRATED = "integrated"
    ENTITY_TYPE_USER = "user"
    ENTITY_TYPE_NPC = "npc"
    ENTITY_TYPE_ENTITY = "entity"
    
    # Memory thresholds
    HIGH_MEMORY_THRESHOLD_MB = 500
    MAX_RESPONSE_TIMES = 100
    MAX_ERROR_LOG_ENTRIES = 100
    MAX_ADAPTATION_HISTORY = 100
    MAX_LEARNED_PATTERNS = 50
    
    # Performance thresholds
    HIGH_RESPONSE_TIME_THRESHOLD = 2.0
    MIN_SUCCESS_RATE = 0.8
    HIGH_ERROR_COUNT = 100
    
    # Relationship thresholds
    INTIMATE_TRUST_THRESHOLD = 0.8
    INTIMATE_BOND_THRESHOLD = 0.7
    FRIENDLY_TRUST_THRESHOLD = 0.6
    HOSTILE_TRUST_THRESHOLD = 0.3
    DOMINANT_POWER_THRESHOLD = 0.7
    SUBMISSIVE_POWER_THRESHOLD = 0.3
    
    # Task thresholds
    MIN_NPC_RELATIONSHIP_FOR_TASK = 30
    
    # Emotional thresholds
    HIGH_AROUSAL_THRESHOLD = 0.8
    NEGATIVE_VALENCE_THRESHOLD = -0.5
    POSITIVE_VALENCE_THRESHOLD = 0.5
    HIGH_DOMINANCE_THRESHOLD = 0.8
    EMOTIONAL_VARIANCE_THRESHOLD = 0.5
    
    # Memory relevance thresholds
    VIVID_RECALL_THRESHOLD = 0.8
    REMEMBER_THRESHOLD = 0.6
    THINK_RECALL_THRESHOLD = 0.4
    
    # Decision thresholds
    MIN_DECISION_SCORE = 0.3
    FALLBACK_DECISION_SCORE = 0.4
    
    # Conflict detection
    POWER_CONFLICT_THRESHOLD = 0.7
    MAX_STABILITY_ISSUES = 10

# ===== Core Data Models =====

class MemoryItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: Optional[str] = Field(None, description="Memory ID if available")
    text: str = Field(..., description="Memory text")
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score 0-1")
    tags: List[str] = Field(default_factory=list, description="Memory tags")

class EmotionalChanges(BaseModel):
    model_config = ConfigDict(extra='forbid')
    valence_change: float
    arousal_change: float
    dominance_change: float

class ScoreComponents(BaseModel):
    model_config = ConfigDict(extra='forbid')
    context: float
    emotional: float
    pattern: float
    relationship: float

class PerformanceNumbers(BaseModel):
    model_config = ConfigDict(extra='forbid')
    memory_mb: float
    cpu_percent: float
    avg_response_time: float
    success_rate: float

class ConflictItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str
    severity: float
    description: str
    entities: Optional[List[str]] = None
    blocked_objectives: Optional[List[str]] = None

class InstabilityItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str
    severity: float
    description: str
    recommendation: Optional[str] = None

class ActivityRec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    requirements: List[str]
    duration: str
    intensity: str
    partner_id: Optional[str] = None

class RelationshipStateOut(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trust: float
    power_dynamic: float
    emotional_bond: float
    interaction_count: int
    last_interaction: float
    type: str

class RelationshipChanges(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trust: float
    power: float
    bond: float

class DecisionMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data: KVList = Field(default_factory=KVList, description="Additional metadata")

class ScoredOption(BaseModel):
    model_config = ConfigDict(extra='forbid')
    option: 'DecisionOption'  # Forward reference
    score: float
    components: ScoreComponents
    is_fallback: Optional[bool] = False

# ===== Pydantic Models for Structured Outputs =====

class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    model_config = ConfigDict(extra='forbid')
    
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    universal_updates: Optional[KVList] = Field(None, description="Universal updates extracted from narrative")

class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    model_config = ConfigDict(extra='forbid')
    
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    topic: Optional[str] = Field(None, description="Topic of the reflection")

class ContentModeration(BaseModel):
    """Output for content moderation guardrail"""
    model_config = ConfigDict(extra='forbid')
    
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

class EmotionalStateUpdate(BaseModel):
    """Structured output for emotional state changes"""
    model_config = ConfigDict(extra='forbid')
    
    valence: float = Field(..., description="Positive/negative emotion (-1 to 1)")
    arousal: float = Field(..., description="Emotional intensity (0 to 1)")
    dominance: float = Field(..., description="Control level (0 to 1)")
    primary_emotion: str = Field(..., description="Primary emotion label")
    reasoning: str = Field(..., description="Why the emotional state changed")

class ScenarioDecision(BaseModel):
    """Structured output for scenario management decisions"""
    model_config = ConfigDict(extra='forbid')
    
    action: str = Field(..., description="Action to take (advance, maintain, escalate, de-escalate)")
    next_phase: str = Field(..., description="Next scenario phase")
    tasks: List[KVList] = Field(default_factory=list, description="Tasks to execute")
    npc_actions: List[KVList] = Field(default_factory=list, description="NPC actions to take")
    time_advancement: bool = Field(False, description="Whether to advance time after this phase")

class RelationshipUpdate(BaseModel):
    """Structured output for relationship changes"""
    model_config = ConfigDict(extra='forbid')
    
    trust_change: float = Field(0.0, description="Change in trust level")
    power_dynamic_change: float = Field(0.0, description="Change in power dynamic")
    emotional_bond_change: float = Field(0.0, description="Change in emotional bond")
    relationship_type: str = Field(..., description="Type of relationship")

class ActivityRecommendation(BaseModel):
    """Structured output for activity recommendations"""
    model_config = ConfigDict(extra='forbid')
    
    recommended_activities: List[ActivityRec] = Field(..., description="List of recommended activities")
    reasoning: str = Field(..., description="Why these activities are recommended")

class ImageGenerationDecision(BaseModel):
    """Decision about whether to generate an image"""
    model_config = ConfigDict(extra='forbid')
    
    should_generate: bool = Field(..., description="Whether an image should be generated")
    score: float = Field(0.0, description="Confidence score for the decision")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    reasoning: str = Field(..., description="Reasoning for the decision")

# ===== Function Tool Input Models =====

class RetrieveMemoriesInput(BaseModel):
    """Input for retrieve_memories function"""
    model_config = ConfigDict(extra='forbid')
    
    query: str = Field(..., description="Search query to find memories")
    limit: int = Field(5, description="Maximum number of memories to return", ge=1, le=20)

class AddMemoryInput(BaseModel):
    """Input for add_memory function"""
    model_config = ConfigDict(extra='forbid')
    
    memory_text: str = Field(..., description="The content of the memory")
    memory_type: str = Field("observation", description="Type of memory (observation, reflection, abstraction)")
    significance: int = Field(5, description="Importance of memory (1-10)", ge=1, le=10)

class DetectUserRevelationsInput(BaseModel):
    """Input for detect_user_revelations function"""
    model_config = ConfigDict(extra='forbid')
    
    user_message: str = Field(..., description="The user's message to analyze")

class GenerateImageFromSceneInput(BaseModel):
    """Input for generate_image_from_scene function"""
    model_config = ConfigDict(extra='forbid')
    
    scene_description: str = Field(..., description="Description of the scene")
    characters: List[str] = Field(..., description="List of characters in the scene")
    style: str = Field("realistic", description="Style for the image")

class CalculateEmotionalStateInput(BaseModel):
    """Input for calculate_and_update_emotional_state and calculate_emotional_impact functions"""
    model_config = ConfigDict(extra='forbid')
    
    context: KVList = Field(..., description="Current interaction context")

class UpdateRelationshipStateInput(BaseModel):
    """Input for update_relationship_state function"""
    model_config = ConfigDict(extra='forbid')
    
    entity_id: str = Field(..., description="ID of the entity (NPC or user)")
    trust_change: float = Field(0.0, description="Change in trust level", ge=-1.0, le=1.0)
    power_change: float = Field(0.0, description="Change in power dynamic", ge=-1.0, le=1.0)
    bond_change: float = Field(0.0, description="Change in emotional bond", ge=-1.0, le=1.0)

class GetActivityRecommendationsInput(BaseModel):
    """Input for get_activity_recommendations function"""
    model_config = ConfigDict(extra='forbid')
    
    scenario_type: str = Field(..., description="Type of current scenario")
    npc_ids: List[str] = Field(..., description="List of present NPC IDs")

class BeliefDataModel(BaseModel):
    """Model for belief data to avoid raw dicts"""
    model_config = ConfigDict(extra='forbid')
    
    entity_id: str = Field("nyx", description="Entity ID")
    type: str = Field("general", description="Belief type")
    content: KVList = Field(default_factory=KVList, description="Belief content")
    query: Optional[str] = Field(None, description="Query for belief search")

class ManageBeliefsInput(BaseModel):
    """Input for manage_beliefs function"""
    model_config = ConfigDict(extra='forbid')
    
    action: Literal["get", "update", "query"] = Field(..., description="Action to perform")
    belief_data: BeliefDataModel = Field(..., description="Data for the belief operation")

class DecisionOption(BaseModel):
    """Model for decision options to avoid raw dicts"""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Option ID")
    description: str = Field(..., description="Option description")
    metadata: DecisionMetadata = Field(default_factory=DecisionMetadata)

class ScoreDecisionOptionsInput(BaseModel):
    """Input for score_decision_options function"""
    model_config = ConfigDict(extra='forbid')
    
    options: List[DecisionOption] = Field(..., description="List of possible decisions/actions")
    decision_context: KVList = Field(..., description="Context for making the decision")

class DetectConflictsAndInstabilityInput(BaseModel):
    """Input for detect_conflicts_and_instability function"""
    model_config = ConfigDict(extra='forbid')
    
    scenario_state: KVList = Field(..., description="Current scenario state")

class GenerateUniversalUpdatesInput(BaseModel):
    """Input for generate_universal_updates function"""
    model_config = ConfigDict(extra='forbid')
    
    narrative: str = Field(..., description="The narrative text to process")

class DecideImageInput(BaseModel):
    """Input for decide_image_generation function"""
    model_config = ConfigDict(extra='forbid')
    
    scene_text: str = Field(..., description="Scene description to evaluate for image generation")

class EmptyInput(BaseModel):
    """Empty input for functions that don't require parameters"""
    model_config = ConfigDict(extra='forbid')
    pass

# ===== Function Tool Output Models =====

class MemorySearchResult(BaseModel):
    """Output for retrieve_memories function"""
    model_config = ConfigDict(extra='forbid')
    
    memories: List[MemoryItem] = Field(..., description="List of retrieved memories")
    formatted_text: str = Field(..., description="Formatted memory text")

class MemoryStorageResult(BaseModel):
    """Output for add_memory function"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str = Field(..., description="ID of stored memory")
    success: bool = Field(..., description="Whether memory was stored successfully")

class UserGuidanceResult(BaseModel):
    """Output for get_user_model_guidance function"""
    model_config = ConfigDict(extra='forbid')
    
    top_kinks: List[Tuple[str, float]] = Field(..., description="Top user preferences with levels")
    behavior_patterns: KVList = Field(..., description="Identified behavior patterns")
    suggested_intensity: float = Field(..., description="Suggested interaction intensity")
    reflections: List[str] = Field(..., description="User model reflections")

class RevelationDetectionResult(BaseModel):
    """Output for detect_user_revelations function"""
    model_config = ConfigDict(extra='forbid')
    
    revelations: List[KVList] = Field(..., description="Detected revelations")
    has_revelations: bool = Field(..., description="Whether any revelations were found")

class ImageGenerationResult(BaseModel):
    """Output for generate_image_from_scene function"""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether image was generated")
    image_url: Optional[str] = Field(None, description="URL of generated image")
    error: Optional[str] = Field(None, description="Error message if failed")

class EmotionalCalculationResult(BaseModel):
    """Output for emotional calculation functions"""
    model_config = ConfigDict(extra='forbid')
    
    valence: float = Field(..., description="New valence value")
    arousal: float = Field(..., description="New arousal value")
    dominance: float = Field(..., description="New dominance value")
    primary_emotion: str = Field(..., description="Primary emotion label")
    changes: EmotionalChanges = Field(..., description="Changes applied")
    state_updated: Optional[bool] = Field(None, description="Whether state was persisted")

class RelationshipUpdateResult(BaseModel):
    """Output for update_relationship_state function"""
    model_config = ConfigDict(extra='forbid')
    
    entity_id: str = Field(..., description="Entity ID")
    relationship: RelationshipStateOut = Field(..., description="Updated relationship state")
    changes: RelationshipChanges = Field(..., description="Changes applied")

class PerformanceMetricsResult(BaseModel):
    """Output for check_performance_metrics function"""
    model_config = ConfigDict(extra='forbid')
    
    metrics: PerformanceNumbers = Field(..., description="Current performance metrics")
    suggestions: List[str] = Field(..., description="Performance improvement suggestions")
    actions_taken: List[str] = Field(..., description="Remediation actions taken")
    health: str = Field(..., description="Overall system health status")

class ActivityRecommendationsResult(BaseModel):
    """Output for get_activity_recommendations function"""
    model_config = ConfigDict(extra='forbid')
    
    recommendations: List[ActivityRec] = Field(..., description="Recommended activities")
    total_available: int = Field(..., description="Total number of available activities")

class BeliefManagementResult(BaseModel):
    """Output for manage_beliefs function"""
    model_config = ConfigDict(extra='forbid')
    
    result: Union[str, KVList] = Field(..., description="Operation result")
    error: Optional[str] = Field(None, description="Error message if failed")

class DecisionScoringResult(BaseModel):
    """Output for score_decision_options function"""
    model_config = ConfigDict(extra='forbid')
    
    scored_options: List[ScoredOption] = Field(..., description="Options with scores")
    best_option: DecisionOption = Field(..., description="Highest scoring option")
    confidence: float = Field(..., description="Confidence in best option")

class ConflictDetectionResult(BaseModel):
    """Output for detect_conflicts_and_instability function"""
    model_config = ConfigDict(extra='forbid')
    
    conflicts: List[ConflictItem] = Field(..., description="Detected conflicts")
    instabilities: List[InstabilityItem] = Field(..., description="Detected instabilities")
    overall_stability: float = Field(..., description="Overall stability score (0-1)")
    stability_note: str = Field(..., description="Explanation of stability score")
    requires_intervention: bool = Field(..., description="Whether intervention is needed")

class UniversalUpdateResult(BaseModel):
    """Output for generate_universal_updates function"""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether updates were generated")
    updates_generated: bool = Field(..., description="Whether any updates were found")
    error: Optional[str] = Field(None, description="Error message if failed")

# ===== Composite Models for Complex Operations =====

class ScenarioManagementRequest(BaseModel):
    """Request for scenario management"""
    model_config = ConfigDict(extra='forbid')
    
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    scenario_id: Optional[str] = Field(None, description="Scenario ID")
    type: str = Field("general", description="Scenario type")
    participants: List[KVList] = Field(default_factory=list, description="Scenario participants")
    objectives: List[KVList] = Field(default_factory=list, description="Scenario objectives")

class RelationshipInteractionData(BaseModel):
    """Data for relationship interactions"""
    model_config = ConfigDict(extra='forbid')
    
    user_id: int = Field(..., description="User ID")
    conversation_id: int = Field(..., description="Conversation ID")
    participants: List[KVList] = Field(..., description="Interaction participants")
    interaction_type: str = Field(..., description="Type of interaction")
    outcome: str = Field(..., description="Interaction outcome")
    emotional_impact: Optional[KVList] = Field(None, description="Emotional impact data")

# ===== State Models =====

class EmotionalState(BaseModel):
    """Emotional state representation"""
    model_config = ConfigDict(extra='forbid')
    
    valence: float = Field(0.0, description="Positive/negative emotion (-1 to 1)", ge=-1.0, le=1.0)
    arousal: float = Field(0.5, description="Emotional intensity (0 to 1)", ge=0.0, le=1.0)
    dominance: float = Field(0.7, description="Control level (0 to 1)", ge=0.0, le=1.0)

class RelationshipState(BaseModel):
    """Relationship state representation"""
    model_config = ConfigDict(extra='forbid')
    
    trust: float = Field(0.5, description="Trust level (0-1)", ge=0.0, le=1.0)
    power_dynamic: float = Field(0.5, description="Power dynamic (0-1)", ge=0.0, le=1.0)
    emotional_bond: float = Field(0.3, description="Emotional bond strength (0-1)", ge=0.0, le=1.0)
    interaction_count: int = Field(0, description="Number of interactions", ge=0)
    last_interaction: float = Field(..., description="Timestamp of last interaction")
    type: str = Field("neutral", description="Relationship type")

class PerformanceMetrics(BaseModel):
    """Performance metrics structure"""
    model_config = ConfigDict(extra='forbid')
    
    total_actions: int = Field(0, ge=0)
    successful_actions: int = Field(0, ge=0)
    failed_actions: int = Field(0, ge=0)
    response_times: List[float] = Field(default_factory=list)
    memory_usage: float = Field(0.0, ge=0.0)
    cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    error_rates: KVList = Field(default_factory=lambda: dict_to_kvlist({"total": 0, "recovered": 0, "unrecovered": 0}))

class LearningMetrics(BaseModel):
    """Learning metrics structure"""
    model_config = ConfigDict(extra='forbid')
    
    pattern_recognition_rate: float = Field(0.0, ge=0.0, le=1.0)
    strategy_improvement_rate: float = Field(0.0, ge=0.0, le=1.0)
    adaptation_success_rate: float = Field(0.0, ge=0.0, le=1.0)

# ===== Enhanced Context with State Management =====
@dataclass
class NyxContext:
    # ────────── REQUIRED (no defaults) ──────────
    user_id: int
    conversation_id: int

    # ────────── SUB-SYSTEM HANDLES ──────────
    memory_system:      Optional[MemoryNyxBridge]   = None
    user_model:         Optional[UserModelManager]  = None
    task_integration:   Optional[NyxTaskIntegration] = None
    response_filter:    Optional[ResponseFilter]    = None
    emotional_core:     Optional[EmotionalCore]     = None
    performance_monitor: Optional[PerformanceMonitor] = None
    belief_system:      Optional[Any]               = None

    # ────────── MUTABLE STATE BUCKETS ──────────
    current_context:     Dict[str, Any]                = field(default_factory=dict)
    scenario_state:      Dict[str, Any]                = field(default_factory=dict)
    relationship_states: Dict[str, Dict[str, Any]]     = field(default_factory=dict)
    active_tasks:        List[Dict[str, Any]]          = field(default_factory=list)

    # ────────── PERFORMANCE & EMOTION ──────────
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_actions": 0, "successful_actions": 0, "failed_actions": 0,
        "response_times": [], "memory_usage": 0, "cpu_usage": 0,
        "error_rates": {"total": 0, "recovered": 0, "unrecovered": 0}
    })
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0, "arousal": 0.5, "dominance": 0.7
    })

    # ────────── LEARNING & ADAPTATION ──────────
    learned_patterns:      Dict[str, Any]           = field(default_factory=dict)
    strategy_effectiveness: Dict[str, Any]          = field(default_factory=dict)
    adaptation_history:    List[Dict[str, Any]]     = field(default_factory=list)
    learning_metrics:      Dict[str, Any]           = field(default_factory=lambda: {
        "pattern_recognition_rate": 0.0,
        "strategy_improvement_rate": 0.0,
        "adaptation_success_rate": 0.0
    })

    # ────────── ERROR LOGGING ──────────
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)  # Track errors by type
    
    # ────────── FEATURE FLAGS ──────────
    _tables_available: Dict[str, bool] = field(default_factory=dict)  # Track which tables exist

    # ────────── TASK SCHEDULING ──────────
    last_task_runs: Dict[str, datetime] = field(default_factory=dict)
    task_intervals: Dict[str, float]    = field(default_factory=lambda: {
        "memory_reflection": 300, "relationship_update": 600,
        "scenario_check": 60, "performance_check": 300,
        "task_generation": 300, "learning_save": 900, 
        "performance_save": 600,  # Only one task for saving performance
        "scenario_heartbeat": 3600  # Hourly heartbeat save for audit trail
    })

    # ────────── PRIVATE CACHES (init=False) ──────────
    _strategy_cache:             Optional[Tuple[float, Any]] = field(init=False, default=None)
    _strategy_cache_ttl:         float = field(init=False, default=300.0)
    _strategy_cache_lock:        asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _cpu_usage_cache:            Optional[float] = field(init=False, default=None)
    _cpu_usage_last_update:      float = field(init=False, default=0.0)
    _cpu_usage_update_interval:  float = field(init=False, default=10.0)
    
    
    async def initialize(self):
        """Initialize all systems"""
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        self.user_model = await UserModelManager.get_instance(self.user_id, self.conversation_id)
        self.task_integration = await NyxTaskIntegration.get_instance(self.user_id, self.conversation_id)
        self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize emotional core if available
        try:
            self.emotional_core = EmotionalCore()
        except Exception as e:
            logger.warning(f"EmotionalCore not available: {e}", exc_info=True)
        
        # Initialize belief system if available
        try:
            from nyx.nyx_belief_system import BeliefSystem
            self.belief_system = BeliefSystem(self.user_id, self.conversation_id)
        except ImportError as e:
            logger.warning(f"BeliefSystem module not available: {e}")
        except Exception as e:
            logger.warning(f"BeliefSystem initialization failed: {e}", exc_info=True)
        
        # Initialize CPU usage monitoring
        try:
            # first call populates the internal psutil sample window
            self._cpu_usage_cache = safe_psutil('cpu_percent', interval=0.1, default=0.0)
        except Exception as e:
            logger.debug(f"Failed to initialize CPU monitoring: {e}")
            self._cpu_usage_cache = 0.0
        
        # Load existing state from database
        await self._load_state()
    
    async def get_active_strategies_cached(self):
        """Get active strategies with caching and lock to prevent thundering herd"""
        current_time = time.time()
        
        # Check cache without lock first
        if self._strategy_cache:
            cache_time, strategies = self._strategy_cache
            if current_time - cache_time < self._strategy_cache_ttl:
                return strategies
        
        # Need to refresh - use lock to prevent multiple refreshes
        async with self._strategy_cache_lock:
            # Double-check cache inside lock
            if self._strategy_cache:
                cache_time, strategies = self._strategy_cache
                if current_time - cache_time < self._strategy_cache_ttl:
                    return strategies
            
            # Fetch new strategies using its own connection
            async with get_db_connection_context() as conn:
                strategies = await get_active_strategies(conn)
            
            # Update cache
            self._strategy_cache = (current_time, strategies)
            return strategies
    
    async def _load_state(self):
        """Load existing state from database"""
        # Use context manager to get a connection
        async with get_db_connection_context() as conn:
            # Load emotional state
            row = await conn.fetchrow("""
                SELECT emotional_state FROM NyxAgentState
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            if row and row["emotional_state"]:
                state = json.loads(row["emotional_state"])
                self.emotional_state.update(state)
            
            # Load scenario state if exists and table is available
            if self._tables_available.get("scenario_states", True):  # Default to True, will be set to False if missing
                try:
                    scenario_row = await conn.fetchrow("""
                        SELECT state_data FROM scenario_states
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY created_at DESC LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if scenario_row and scenario_row["state_data"]:
                        self.scenario_state = json.loads(scenario_row["state_data"])
                except Exception as e:
                    # Table might not exist yet
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        logger.info("scenario_states table not found - migrations may need to be run")
                        self._tables_available["scenario_states"] = False
                    else:
                        logger.debug(f"Could not load scenario state: {e}")
    
    def update_performance(self, metric: str, value: Any):
        """Update performance metrics"""
        if metric in self.performance_metrics:
            if isinstance(self.performance_metrics[metric], list):
                self.performance_metrics[metric].append(value)
                # Keep only last entries based on config
                if len(self.performance_metrics[metric]) > Config.MAX_RESPONSE_TIMES:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-Config.MAX_RESPONSE_TIMES:]
            else:
                self.performance_metrics[metric] = value
    
    def should_run_task(self, task_id: str) -> bool:
        """Check if enough time has passed to run task again"""
        if task_id not in self.last_task_runs:
            return True
        
        time_since_run = (datetime.now(timezone.utc) - self.last_task_runs[task_id]).total_seconds()
        return time_since_run >= self.task_intervals.get(task_id, 300)
    
    def record_task_run(self, task_id: str):
        """Record that a task has been run"""
        self.last_task_runs[task_id] = datetime.now(timezone.utc)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context and aggregate by type"""
        error_type = type(error).__name__
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "type": error_type,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
        # Track error counts by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update error metrics
        self.performance_metrics["error_rates"]["total"] += 1
        
        # Keep error log bounded - more aggressive pruning on errors
        if len(self.error_log) > Config.MAX_ERROR_LOG_ENTRIES * 2:
            _prune_list(self.error_log, Config.MAX_ERROR_LOG_ENTRIES)
            
        # Log warning if we see repeated errors
        if self.error_counts[error_type] > 10:
            logger.warning(f"Repeated error type {error_type}: {self.error_counts[error_type]} occurrences")
    
    async def learn_from_interaction(self, action: str, outcome: str, success: bool):
        """Learn from an interaction outcome"""
        # Update patterns
        pattern_key = f"{action}_{outcome}"
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "occurrences": 0,
                "successes": 0,
                "last_seen": time.time()
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        if success:
            pattern["successes"] += 1
        pattern["last_seen"] = time.time()
        pattern["success_rate"] = pattern["successes"] / pattern["occurrences"]
        
        # Update adaptation history with emotional state snapshot
        self.adaptation_history.append({
            "timestamp": time.time(),
            "action": action,
            "outcome": outcome,
            "success": success,
            "emotional_state": self.emotional_state.copy()
        })
        
        # Keep adaptation history bounded - more aggressive on failures
        max_history = Config.MAX_ADAPTATION_HISTORY if success else Config.MAX_ADAPTATION_HISTORY // 2
        if len(self.adaptation_history) > max_history * 2:
            self.adaptation_history = self.adaptation_history[-max_history:]
        
        # Prune old patterns (older than 24 hours)
        current_time = time.time()
        self.learned_patterns = {
            k: v for k, v in self.learned_patterns.items()
            if current_time - v.get("last_seen", 0) < 86400
        }
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def should_generate_task(self) -> bool:
        """Determine if we should generate a creative task"""
        context = self.current_context
        
        if not context.get("active_npc_id"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        task_scenarios = ["training", "challenge", "service", "discipline"]
        if not any(t in scenario_type for t in task_scenarios):
            return False
            
        npc_relationship = context.get("npc_relationship_level", 0)
        if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
            return False
            
        # Check task timing
        if not self.should_run_task("task_generation"):
            return False
            
        return True
    
    def should_recommend_activities(self) -> bool:
        """Determine if we should recommend activities"""
        context = self.current_context
        
        if not context.get("present_npc_ids"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        if "task" in scenario_type or "challenge" in scenario_type:
            return False
            
        user_input = context.get("user_input", "").lower()
        suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
        if any(trigger in user_input for trigger in suggestion_triggers):
            return True
            
        if context.get("is_scene_transition") or context.get("is_activity_completed"):
            return True
            
        return False
    
    async def handle_high_memory_usage(self):
        """Handle high memory usage by cleaning up"""
        # Trim memory system cache if available
        if hasattr(self.memory_system, 'trim_cache'):
            await self.memory_system.trim_cache()
        
        # Clear old patterns
        self.learned_patterns = dict(list(self.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:])
        
        # Clear old history
        self.adaptation_history = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
        self.error_log = self.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        
        # Clear performance metrics history
        if "response_times" in self.performance_metrics:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Performed memory cleanup")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage with caching - Fixed to properly refresh"""
        try:
            current_time = time.time()
            # Check if we need to update the cache
            if (self._cpu_usage_cache is None or 
                current_time - self._cpu_usage_last_update >= self._cpu_usage_update_interval):
                # Update the cache using safe wrapper
                new_value = safe_psutil('cpu_percent', interval=0.1, default=0.0)
                if new_value is not None:
                    self._cpu_usage_cache = new_value
                    self._cpu_usage_last_update = current_time
            
            return self._cpu_usage_cache or 0.0
        except Exception as e:
            logger.debug(f"Failed to get CPU usage: {e}")
            return 0.0

    def db_connection_ctx(self):
        """
        Get a database connection context manager.
        
        Usage:
            async with ctx.db_connection_ctx() as conn:
                result = await conn.fetchone(...)
        
        Returns:
            An async context manager that yields a database connection
        """
        return get_db_connection_context()
    
    # Legacy compatibility - keep old name but mark deprecated
    async def get_db_connection(self):
        """
        DEPRECATED: Use db_connection_ctx() instead.
        
        Get a database connection as an async context manager.
        
        IMPORTANT: This returns an async context manager, not a connection object.
        You must use it with 'async with', not 'await':
        
        Correct usage:
            async with ctx.get_db_connection() as conn:
                result = await conn.fetchone(...)
        
        Incorrect usage:
            conn = await ctx.get_db_connection()  # This will fail!
        
        Returns:
            An async context manager that yields a database connection
        """
        logger.warning("get_db_connection is deprecated, use db_connection_ctx() instead")
        return self.db_connection_ctx()

    async def close_db_connection(self, conn=None):
        """
        No-op compatibility wrapper so calls like
        `await nyx_context.close_db_connection()` don't crash.
        If you pass the connection you got from get_db_connection(),
        it will be cleanly closed; otherwise it's a harmless no-op.
        """
        if conn is not None:                        # caller gave us the handle
            await conn.__aexit__(None, None, None)
    
    def _update_learning_metrics(self):
        """Update learning-related metrics"""
        if self.learned_patterns:
            successful_patterns = sum(1 for p in self.learned_patterns.values() 
                                    if p.get("success_rate", 0) > 0.6)
            self.learning_metrics["pattern_recognition_rate"] = (
                successful_patterns / len(self.learned_patterns)
            )
        
        if self.adaptation_history:
            recent = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
            successes = sum(1 for a in recent if a["success"])
            self.learning_metrics["adaptation_success_rate"] = successes / len(recent)

# Resolve forward references for ScoredOption
ScoredOption.model_rebuild()
DecisionOption.model_rebuild()

# ===== Helper Functions =====

async def run_agent_with_error_handling(
    agent: Agent,
    input_data: Any,
    context: NyxContext,
    output_type: Optional[type] = None,
    fallback_value: Any = None
) -> Any:
    """
    Run an agent with automatic error handling and type conversion.
    
    Args:
        agent: The agent to run
        input_data: Input for the agent
        context: NyxContext instance
        output_type: Expected output type for conversion
        fallback_value: Value to return on error
        
    Returns:
        Converted output or fallback value on error
    """
    try:
        result = await Runner.run(agent, input_data, context=context)
        if output_type:
            return result.final_output_as(output_type)
        return result.final_output
    except Exception as e:
        logger.error(f"Error running agent {agent.name}: {e}")
        if fallback_value is not None:
            return fallback_value
        raise

def safe_psutil(func_name: str, *args, default=None, **kwargs):
    """Safe wrapper for psutil calls that may fail on certain platforms"""
    try:
        import psutil
        func = getattr(psutil, func_name)
        return func(*args, **kwargs)
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"psutil.{func_name} failed (platform compatibility): {e}")
        return default

def safe_process_metric(process, metric_name: str, default=0):
    """Safe wrapper for process-specific metrics"""
    try:
        metric_func = getattr(process, metric_name)
        result = metric_func()
        # Handle different return types
        if hasattr(result, 'rss'):  # memory_info returns a named tuple
            return result.rss
        return result
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"Process metric {metric_name} failed: {e}")
        return default

def get_process_info() -> Optional[Any]:
    """Get current process info safely"""
    try:
        import psutil
        return psutil.Process(os.getpid())
    except Exception as e:
        logger.debug(f"Failed to get process info: {e}")
        return None

def bytes_to_mb(value: Optional[Union[int, float]]) -> float:
    """Convert bytes to megabytes safely"""
    return (value or 0) / (1024 * 1024)

def extract_token_usage(result: Any) -> int:
    """Extract token usage from various result formats"""
    try:
        # Try different possible locations for token usage
        if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
            return result.usage.total_tokens
        elif hasattr(result, 'trace') and hasattr(result.trace, 'final_usage'):
            return result.trace.final_usage.get('total_tokens', 0)
        else:
            logger.debug("Token usage not found in result object")
            return 0
    except Exception as e:
        logger.debug(f"Failed to retrieve token usage: {e}")
        return 0

# ===== Function Tools =====

@function_tool
async def retrieve_memories(ctx: RunContextWrapper[NyxContext], payload: RetrieveMemoriesInput) -> MemorySearchResult:
    """
    Retrieve relevant memories for Nyx.
    
    Args:
        payload: Input containing query and limit
    
    Returns:
        MemorySearchResult object (not JSON string)
    """
    query = payload.query
    limit = payload.limit
    memory_system = ctx.context.memory_system
    
    result = await memory_system.recall(
        entity_type=Config.ENTITY_TYPE_INTEGRATED,
        entity_id=0,
        query=query,
        limit=limit
    )
    
    memories_raw = result.get("memories", [])
    memories = [
        MemoryItem(
            id=str(m.get("id") or m.get("memory_id") or ""),
            text=m["text"],
            relevance=float(m.get("relevance", 0.0)),
            tags=m.get("tags", [])
        )
        for m in memories_raw
    ]
    
    formatted_memories = []
    for memory in memories:
        relevance = memory.relevance
        confidence_marker = "vividly recall" if relevance > Config.VIVID_RECALL_THRESHOLD else \
                          "remember" if relevance > Config.REMEMBER_THRESHOLD else \
                          "think I recall" if relevance > Config.THINK_RECALL_THRESHOLD else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory.text}")
    
    formatted_text = "\n".join(formatted_memories) if formatted_memories else "No relevant memories found."
    
    return MemorySearchResult(
        memories=memories,
        formatted_text=formatted_text
    )

@function_tool
async def add_memory(ctx: RunContextWrapper[NyxContext], payload: AddMemoryInput) -> str:
    """
    Add a new memory for Nyx.
    
    Args:
        payload: Input containing memory text, type, and significance
    """
    memory_text = payload.memory_text
    memory_type = payload.memory_type
    significance = payload.significance
    
    memory_system = ctx.context.memory_system
    
    # Convert significance to importance string
    if significance >= 8:
        importance = "critical"
    elif significance >= 6:
        importance = "high"
    elif significance >= 4:
        importance = "medium"
    elif significance >= 2:
        importance = "low"
    else:
        importance = "trivial"
    
    result = await memory_system.remember(
        entity_type="integrated",
        entity_id=0,
        memory_text=memory_text,
        importance=importance,
        emotional=True,
        tags=["agent_generated", memory_type]
    )
    
    memory_id = result.get("memory_id", "unknown")
    
    # Return structured output as JSON
    return MemoryStorageResult(
        memory_id=str(memory_id),
        success=True
    ).model_dump_json()

@function_tool
async def get_user_model_guidance(ctx: RunContextWrapper[NyxContext], payload: EmptyInput) -> str:
    """Get guidance for how Nyx should respond based on the user model."""
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    top_kinks = guidance.get("top_kinks", [])
    behavior_patterns = guidance.get("behavior_patterns", {})
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    reflections = guidance.get("reflections", [])
    
    # Return structured output
    return UserGuidanceResult(
        top_kinks=top_kinks,
        behavior_patterns=dict_to_kvlist(behavior_patterns),
        suggested_intensity=suggested_intensity,
        reflections=reflections
    ).model_dump_json()

@function_tool
async def detect_user_revelations(ctx: RunContextWrapper[NyxContext], payload: DetectUserRevelationsInput) -> str:
    """
    Detect if user is revealing new preferences or patterns.
    
    Args:
        payload: Input containing user message to analyze
    """
    user_message = payload.user_message
    lower_message = user_message.lower()
    revelations = []
    
    kink_keywords = {
        "ass": ["ass", "booty", "behind", "rear"],
        "feet": ["feet", "foot", "toes"],
        "goth": ["goth", "gothic", "dark", "black clothes"],
        "tattoos": ["tattoo", "ink", "inked"],
        "piercings": ["piercing", "pierced", "stud", "ring"],
        "latex": ["latex", "rubber", "shiny"],
        "leather": ["leather", "leathery"],
        "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
        "submission": ["submit", "obey", "serve", "kneel"]
    }
    
    for kink, keywords in kink_keywords.items():
        if any(keyword in lower_message for keyword in keywords):
            sentiment = "neutral"
            pos_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
            neg_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
            
            pos_count = sum(1 for word in pos_words if word in lower_message)
            neg_count = sum(1 for word in neg_words if word in lower_message)
            
            if pos_count > neg_count:
                sentiment = "positive"
                intensity = 0.7
            elif neg_count > pos_count:
                sentiment = "negative" 
                intensity = 0.0
            else:
                intensity = 0.4
                
            # Track both positive and negative preferences
            revelation_data = {
                "type": "kink_preference",
                "kink": kink,
                "intensity": intensity,
                "source": "explicit_negative_mention" if sentiment == "negative" else "explicit_mention",
                "sentiment": sentiment
            }
            
            revelations.append(dict_to_kvlist(revelation_data))
    
    if "don't tell me what to do" in lower_message or "i won't" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "resistance",
            "intensity": 0.6,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    if "yes mistress" in lower_message or "i'll obey" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "submission",
            "intensity": 0.8,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    # Save revelations to database if found
    if revelations and ctx.context.user_model:
        # User model will handle its own DB connection
        for revelation_kv in revelations:
            revelation = kvlist_to_dict(revelation_kv)
            if revelation["type"] == "kink_preference":
                await ctx.context.user_model.update_kink_preference(
                    revelation["kink"],
                    revelation["intensity"],
                    revelation["source"]
                )
    
    # Return structured output
    return RevelationDetectionResult(
        revelations=revelations,
        has_revelations=len(revelations) > 0
    ).model_dump_json()

@function_tool
async def generate_image_from_scene(
    ctx: RunContextWrapper[NyxContext], 
    payload: GenerateImageFromSceneInput
) -> str:
    """
    Generate an image for the current scene.
    
    Args:
        payload: Input containing scene description, characters, and style
    """
    from routes.ai_image_generator import generate_roleplay_image_from_gpt
    
    image_data = payload.model_dump()
    
    # This function should handle its own connections
    result = await generate_roleplay_image_from_gpt(
        image_data,
        ctx.context.user_id,
        ctx.context.conversation_id
    )
    
    if result and "image_urls" in result and result["image_urls"]:
        return ImageGenerationResult(
            success=True,
            image_url=result["image_urls"][0],
            error=None
        ).model_dump_json()
    else:
        return ImageGenerationResult(
            success=False,
            image_url=None,
            error="Failed to generate image"
        ).model_dump_json()

@function_tool
async def calculate_and_update_emotional_state(ctx: RunContextWrapper[NyxContext], payload: CalculateEmotionalStateInput) -> str:
    """
    Calculate emotional impact and immediately update the emotional state.
    This is a composite tool that both calculates AND persists the changes.
    
    Args:
        payload: Input containing context for emotional calculation
    """
    context_dict = kvlist_to_dict(payload.context)
    
    # First calculate the new state
    result = await calculate_emotional_impact(ctx, payload)
    emotional_data = json.loads(result)
    
    # Immediately update the context with the new state
    ctx.context.emotional_state.update({
        "valence": emotional_data["valence"],
        "arousal": emotional_data["arousal"],
        "dominance": emotional_data["dominance"]
    })
    
    # Save to database with its own connection
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, ctx.context.user_id, ctx.context.conversation_id, json.dumps(ctx.context.emotional_state))
    
    # Return the result with confirmation of update
    emotional_data["state_updated"] = True
    
    return EmotionalCalculationResult(
        valence=emotional_data["valence"],
        arousal=emotional_data["arousal"],
        dominance=emotional_data["dominance"],
        primary_emotion=emotional_data["primary_emotion"],
        changes=EmotionalChanges(
            valence_change=emotional_data["changes"]["valence_change"],
            arousal_change=emotional_data["changes"]["arousal_change"],
            dominance_change=emotional_data["changes"]["dominance_change"]
        ),
        state_updated=True
    ).model_dump_json()

@function_tool
async def calculate_emotional_impact(ctx: RunContextWrapper[NyxContext], payload: CalculateEmotionalStateInput) -> str:
    """
    Calculate emotional impact of current context using the emotional core system.
    Returns new emotional state without mutating the context.
    NOTE: Use calculate_and_update_emotional_state if you want to persist changes.
    
    Args:
        payload: Input containing context for emotional calculation
    """
    context_dict = kvlist_to_dict(payload.context)
    current_state = ctx.context.emotional_state.copy()  # Work with a copy
    
    # Calculate emotional changes based on context
    valence_change = 0.0
    arousal_change = 0.0
    dominance_change = 0.0
    
    # Analyze context for emotional triggers - build lowercase text once
    context_text_lower = get_context_text_lower(context_dict)
    
    if "conflict" in context_text_lower:
        arousal_change += 0.2
        valence_change -= 0.1
    if "submission" in context_text_lower:
        dominance_change += 0.1
        arousal_change += 0.1
    if "praise" in context_text_lower or "good" in context_text_lower:
        valence_change += 0.2
    if "resistance" in context_text_lower:
        arousal_change += 0.15
        dominance_change -= 0.05
    
    # Get memory emotional impact
    memory_impact = await _get_memory_emotional_impact(ctx, context_dict)
    valence_change += memory_impact["valence"] * 0.3
    arousal_change += memory_impact["arousal"] * 0.3
    dominance_change += memory_impact["dominance"] * 0.3
    
    # Use EmotionalCore if available for more nuanced analysis
    if ctx.context.emotional_core:
        try:
            core_analysis = ctx.context.emotional_core.analyze(str(context_dict))
            valence_change += core_analysis.get("valence_delta", 0) * 0.5
            arousal_change += core_analysis.get("arousal_delta", 0) * 0.5
        except Exception as e:
            logger.debug(f"EmotionalCore analysis failed: {e}", exc_info=True)
    
    # Apply changes with bounds
    new_valence = max(-1, min(1, current_state["valence"] + valence_change))
    new_arousal = max(0, min(1, current_state["arousal"] + arousal_change))
    new_dominance = max(0, min(1, current_state["dominance"] + dominance_change))
    
    # Determine primary emotion based on VAD model
    primary_emotion = "neutral"
    if new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "excited"
    elif new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "content"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "frustrated"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "disappointed"
    elif new_dominance > Config.HIGH_DOMINANCE_THRESHOLD:
        primary_emotion = "commanding"
    
    # Return new state without mutating
    return EmotionalCalculationResult(
        valence=new_valence,
        arousal=new_arousal,
        dominance=new_dominance,
        primary_emotion=primary_emotion,
        changes=EmotionalChanges(
            valence_change=valence_change,
            arousal_change=arousal_change,
            dominance_change=dominance_change
        ),
        state_updated=None  # Not updated in this function
    ).model_dump_json()

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper[NyxContext],
    payload: UpdateRelationshipStateInput
) -> str:
    """
    Update relationship state with an entity.
    
    Args:
        payload: Input containing entity ID and relationship changes
    """
    entity_id = payload.entity_id
    trust_change = payload.trust_change
    power_change = payload.power_change
    bond_change = payload.bond_change
    
    relationships = ctx.context.relationship_states
    
    if entity_id not in relationships:
        relationships[entity_id] = {
            "trust": 0.5,
            "power_dynamic": 0.5,
            "emotional_bond": 0.3,
            "interaction_count": 0,
            "last_interaction": time.time(),
            "type": "neutral"
        }
    
    rel = relationships[entity_id]
    rel["trust"] = max(0, min(1, rel["trust"] + trust_change))
    rel["power_dynamic"] = max(0, min(1, rel["power_dynamic"] + power_change))
    rel["emotional_bond"] = max(0, min(1, rel["emotional_bond"] + bond_change))
    rel["interaction_count"] += 1
    rel["last_interaction"] = time.time()
    
    # Determine relationship type
    if rel["trust"] > Config.INTIMATE_TRUST_THRESHOLD and rel["emotional_bond"] > Config.INTIMATE_BOND_THRESHOLD:
        rel["type"] = "intimate"
    elif rel["trust"] > Config.FRIENDLY_TRUST_THRESHOLD:
        rel["type"] = "friendly"
    elif rel["trust"] < Config.HOSTILE_TRUST_THRESHOLD:
        rel["type"] = "hostile"
    elif rel["power_dynamic"] > Config.DOMINANT_POWER_THRESHOLD:
        rel["type"] = "dominant"
    elif rel["power_dynamic"] < Config.SUBMISSIVE_POWER_THRESHOLD:
        rel["type"] = "submissive"
    else:
        rel["type"] = "neutral"
    
    # Save to database with its own connection
    async with get_db_connection_context() as conn:
        # First get existing evolution history
        existing = await conn.fetchrow("""
            SELECT evolution_history 
            FROM RelationshipEvolution 
            WHERE user_id = $1 AND conversation_id = $2 
                AND npc1_id = $3 AND entity2_type = $4 AND entity2_id = $5
        """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id)
        
        # Build evolution history
        evolution_history = []
        if existing and existing["evolution_history"]:
            evolution_history = json.loads(existing["evolution_history"])
        
        # Add new entry (keep last 50 entries)
        evolution_history.append({
            "timestamp": time.time(),
            "trust": rel["trust"],
            "power": rel["power_dynamic"],
            "bond": rel["emotional_bond"]
        })
        evolution_history = evolution_history[-50:]
        
        await conn.execute("""
            INSERT INTO RelationshipEvolution 
            (user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
             relationship_type, current_stage, progress_to_next, evolution_history)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (user_id, conversation_id, npc1_id, entity2_type, entity2_id)
            DO UPDATE SET 
                relationship_type = $6,
                current_stage = $7,
                progress_to_next = $8,
                evolution_history = $9
        """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id,
             rel["type"], rel["type"], 0, json.dumps(evolution_history))
    
    return RelationshipUpdateResult(
        entity_id=entity_id,
        relationship=RelationshipStateOut(
            trust=rel["trust"],
            power_dynamic=rel["power_dynamic"],
            emotional_bond=rel["emotional_bond"],
            interaction_count=rel["interaction_count"],
            last_interaction=rel["last_interaction"],
            type=rel["type"]
        ),
        changes=RelationshipChanges(
            trust=trust_change,
            power=power_change,
            bond=bond_change
        )
    ).model_dump_json()

@function_tool
async def check_performance_metrics(ctx: RunContextWrapper[NyxContext], payload: EmptyInput) -> str:
    """Check current performance metrics and apply remediation if needed."""
    metrics = ctx.context.performance_metrics

    # Refresh CPU & RAM values using centralized helper
    process = get_process_info()
    if process:
        memory_info = safe_process_metric(process, 'memory_info')
        metrics["memory_usage"] = bytes_to_mb(memory_info)
    else:
        metrics["memory_usage"] = 0

    metrics["cpu_usage"] = ctx.context.get_cpu_usage()

    suggestions, actions_taken = [], []

    # --- Health checks ----------------------------------------------------
    avg_rt = _calculate_avg_response_time(metrics["response_times"])
    if avg_rt > Config.HIGH_RESPONSE_TIME_THRESHOLD:
        suggestions.append("Response times are high – consider caching frequent queries")

    if metrics["memory_usage"] > Config.HIGH_MEMORY_THRESHOLD_MB:
        suggestions.append("High memory usage detected – triggering cleanup")
        memory_before = metrics["memory_usage"]
        await ctx.context.handle_high_memory_usage()
        # Re-check memory after cleanup
        if process:
            memory_after = bytes_to_mb(safe_process_metric(process, 'memory_info'))
            logger.info(f"Memory cleanup: {memory_before:.2f}MB -> {memory_after:.2f}MB")
        actions_taken.append("memory_cleanup")

    if metrics["total_actions"]:
        success_rate = metrics["successful_actions"] / metrics["total_actions"]
        if success_rate < Config.MIN_SUCCESS_RATE:
            suggestions.append("Success rate below 80% – review error patterns")

    if metrics["error_rates"]["total"] > Config.HIGH_ERROR_COUNT:
        suggestions.append("High error count – clearing old errors")
        ctx.context.error_log = ctx.context.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        actions_taken.append("error_log_cleanup")

    return PerformanceMetricsResult(
        metrics=PerformanceNumbers(
            memory_mb=metrics["memory_usage"],
            cpu_percent=metrics["cpu_usage"],
            avg_response_time=(
                sum(metrics["response_times"]) / len(metrics["response_times"])
                if metrics["response_times"] else 0
            ),
            success_rate=(
                metrics["successful_actions"] / metrics["total_actions"]
                if metrics["total_actions"] else 1.0
            )
        ),
        suggestions=suggestions,
        actions_taken=actions_taken,
        health="good" if not suggestions else "needs_attention",
    ).model_dump_json()

@function_tool
async def get_activity_recommendations(
    ctx: RunContextWrapper[NyxContext],
    payload: GetActivityRecommendationsInput
) -> str:
    """
    Get activity recommendations based on current context.
    
    Args:
        payload: Input containing scenario type and NPC IDs
    """
    scenario_type = payload.scenario_type
    npc_ids = payload.npc_ids
    
    activities = []
    
    # Copy relationship states to avoid mutation during iteration
    relationship_states_copy = dict(ctx.context.relationship_states)
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states_copy.values()):
        activities.extend([
            ActivityRec(
                name="Obedience Training",
                description="Test and improve submission through structured exercises",
                requirements=["trust > 0.4", "submission tendency"],
                duration="15-30 minutes",
                intensity="medium"
            ),
            ActivityRec(
                name="Position Practice",
                description="Learn and perfect submissive positions",
                requirements=["trust > 0.5"],
                duration="10-20 minutes",
                intensity="low-medium"
            )
        ])
    
    # Social activities
    if npc_ids and len(npc_ids) > 0:
        activities.append(ActivityRec(
            name="Group Dynamics Exercise",
            description="Explore power dynamics with multiple participants",
            requirements=["multiple NPCs present"],
            duration="20-40 minutes",
            intensity="variable"
        ))
    
    # Intimate activities
    for entity_id, rel in relationship_states_copy.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append(ActivityRec(
                name="Intimate Scene",
                description=f"Deepen connection with trusted partner",
                requirements=["high trust", "intimate relationship"],
                duration="30-60 minutes",
                intensity="high",
                partner_id=entity_id
            ))
            break
    
    # Default activities
    activities.extend([
        ActivityRec(
            name="Exploration",
            description="Discover new areas or items",
            requirements=[],
            duration="10-30 minutes",
            intensity="low"
        ),
        ActivityRec(
            name="Conversation",
            description="Engage in meaningful dialogue",
            requirements=[],
            duration="5-15 minutes",
            intensity="low"
        )
    ])
    
    return ActivityRecommendationsResult(
        recommendations=activities[:5],  # Top 5 activities
        total_available=len(activities)
    ).model_dump_json()

@function_tool
async def manage_beliefs(ctx: RunContextWrapper[NyxContext], payload: ManageBeliefsInput) -> str:
    """
    Manage belief system operations.
    
    Args:
        payload: Input containing action and belief data
    """
    action = payload.action
    belief_data = payload.belief_data
    
    if not ctx.context.belief_system:
        return BeliefManagementResult(
            result="",
            error="Belief system not available"
        ).model_dump_json()
    
    try:
        if action == "get":
            entity_id = belief_data.entity_id
            beliefs = await ctx.context.belief_system.get_beliefs(entity_id)
            return BeliefManagementResult(
                result=dict_to_kvlist(beliefs),
                error=None
            ).model_dump_json()
        
        elif action == "update":
            entity_id = belief_data.entity_id
            belief_type = belief_data.type
            content = kvlist_to_dict(belief_data.content)
            await ctx.context.belief_system.update_belief(entity_id, belief_type, content)
            return BeliefManagementResult(
                result="Belief updated successfully",
                error=None
            ).model_dump_json()
        
        elif action == "query":
            query = belief_data.query or ""
            results = await ctx.context.belief_system.query_beliefs(query)
            return BeliefManagementResult(
                result=dict_to_kvlist(results),
                error=None
            ).model_dump_json()
        
        else:
            return BeliefManagementResult(
                result="",
                error=f"Unknown action: {action}"
            ).model_dump_json()
            
    except Exception as e:
        logger.error(f"Error managing beliefs: {e}", exc_info=True)
        return BeliefManagementResult(
            result="",
            error=str(e)
        ).model_dump_json()

@function_tool
async def score_decision_options(
    ctx: RunContextWrapper[NyxContext],
    payload: ScoreDecisionOptionsInput
) -> str:
    """
    Score decision options using advanced decision engine logic.
    
    Args:
        payload: Input containing options and decision context
    """
    options = payload.options
    decision_context = kvlist_to_dict(payload.decision_context)
    
    scored_options = []
    
    for option in options:
        # Base score from context relevance
        context_score = _calculate_context_relevance(option.model_dump(), decision_context)
        
        # Emotional alignment score
        emotional_score = _calculate_emotional_alignment(option.model_dump(), ctx.context.emotional_state)
        
        # Pattern-based score
        pattern_score = _calculate_pattern_score(option.model_dump(), ctx.context.learned_patterns)
        
        # Relationship impact score
        relationship_score = _calculate_relationship_impact(option.model_dump(), ctx.context.relationship_states)
        
        # Calculate weighted final score
        weights = {
            "context": 0.3,
            "emotional": 0.25,
            "pattern": 0.25,
            "relationship": 0.2
        }
        
        final_score = (
            context_score * weights["context"] +
            emotional_score * weights["emotional"] +
            pattern_score * weights["pattern"] +
            relationship_score * weights["relationship"]
        )
        
        scored_options.append(ScoredOption(
            option=option,
            score=final_score,
            components=ScoreComponents(
                context=context_score,
                emotional=emotional_score,
                pattern=pattern_score,
                relationship=relationship_score
            )
        ))
    
    # Sort by score
    scored_options.sort(key=lambda x: x.score, reverse=True)
    
    # If all scores are too low, include a fallback
    if all(opt.score < Config.MIN_DECISION_SCORE for opt in scored_options):
        fallback = _get_fallback_decision(options)
        fallback_scored = ScoredOption(
            option=fallback,
            score=Config.FALLBACK_DECISION_SCORE,
            components=ScoreComponents(
                context=Config.FALLBACK_DECISION_SCORE,
                emotional=Config.FALLBACK_DECISION_SCORE,
                pattern=Config.FALLBACK_DECISION_SCORE,
                relationship=Config.FALLBACK_DECISION_SCORE
            ),
            is_fallback=True
        )
        scored_options.insert(0, fallback_scored)
    
    return DecisionScoringResult(
        scored_options=scored_options,
        best_option=scored_options[0].option,
        confidence=scored_options[0].score
    ).model_dump_json()

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper[NyxContext],
    payload: DetectConflictsAndInstabilityInput
) -> str:
    """
    Detect conflicts and emotional instability in current scenario.
    
    Args:
        payload: Input containing scenario state
    """
    scenario_state = kvlist_to_dict(payload.scenario_state)
    
    conflicts = []
    instabilities = []
    
    # Check for relationship conflicts
    # Create a copy of items to avoid mutation during iteration
    relationship_items = list(ctx.context.relationship_states.items())
    for i, (entity1_id, rel1) in enumerate(relationship_items):
        for entity2_id, rel2 in relationship_items[i+1:]:
            # Conflicting power dynamics
            if abs(rel1.get("power_dynamic", 0.5) - rel2.get("power_dynamic", 0.5)) > Config.POWER_CONFLICT_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="power_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=abs(rel1["power_dynamic"] - rel2["power_dynamic"]),
                    description="Conflicting power dynamics between entities"
                ))
            
            # Low mutual trust
            if rel1.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD and rel2.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="trust_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=0.7,
                    description="Mutual distrust between entities"
                ))
    
    # Check for emotional instability
    emotional_state = ctx.context.emotional_state
    
    # High arousal with negative valence
    if emotional_state["arousal"] > Config.HIGH_AROUSAL_THRESHOLD and emotional_state["valence"] < Config.NEGATIVE_VALENCE_THRESHOLD:
        instabilities.append(InstabilityItem(
            type="emotional_volatility",
            severity=emotional_state["arousal"],
            description="High arousal with negative emotions",
            recommendation="De-escalation needed"
        ))
    
    # Rapid emotional changes
    if ctx.context.adaptation_history:
        recent_emotions = [h.get("emotional_state", {}) for h in ctx.context.adaptation_history[-5:]]
        if recent_emotions and any(recent_emotions):  # Check if we have actual emotional states
            valence_values = [e.get("valence", 0) for e in recent_emotions if e]
            if valence_values:  # Only calculate variance if we have values
                valence_variance = _calculate_variance(valence_values)
                if valence_variance > Config.EMOTIONAL_VARIANCE_THRESHOLD:
                    instabilities.append(InstabilityItem(
                        type="emotional_instability",
                        severity=min(1.0, valence_variance),
                        description="Rapid emotional swings detected",
                        recommendation="Stabilization recommended"
                    ))
    
    # Scenario-specific conflicts
    if scenario_state.get("objectives"):
        blocked_objectives = [obj for obj in scenario_state["objectives"] 
                             if obj.get("status") == "blocked"]
        if blocked_objectives:
            conflicts.append(ConflictItem(
                type="objective_conflict",
                severity=len(blocked_objectives) / len(scenario_state["objectives"]),
                description=f"{len(blocked_objectives)} objectives are blocked",
                blocked_objectives=[str(obj) for obj in blocked_objectives]
            ))
    
    # Calculate overall stability
    total_issues = len(conflicts) + len(instabilities)
    overall_stability = max(0.0, 1.0 - (total_issues / Config.MAX_STABILITY_ISSUES))
    
    # Only save scenario state if it's a significant change
    if ctx.context.scenario_state and ctx.context.scenario_state.get("active"):
        should_save = False
        
        # Check if this is a significant change
        if conflicts and any(c.severity > 0.7 for c in conflicts):
            should_save = True
        if instabilities and any(i.severity > 0.7 for i in instabilities):
            should_save = True
        if overall_stability < 0.3:
            should_save = True
            
        if should_save:
            async with get_db_connection_context() as conn:
                # Use UPSERT pattern to maintain one current state
                await conn.execute("""
                    INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, conversation_id) 
                    DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                """, ctx.context.user_id, ctx.context.conversation_id, 
                json.dumps(ctx.context.scenario_state, ensure_ascii=False))
    
    return ConflictDetectionResult(
        conflicts=conflicts,
        instabilities=instabilities,
        overall_stability=overall_stability,
        stability_note=f"{total_issues} issues detected (0 issues = 1.0 stability, 10+ issues = 0.0 stability)",
        requires_intervention=any(c.severity > 0.8 for c in conflicts + instabilities)
    ).model_dump_json()

@function_tool
async def decide_image_generation(ctx: RunContextWrapper[NyxContext], payload: DecideImageInput) -> str:
    """
    Decide whether an image should be generated for a scene.
    
    Args:
        payload: Input containing scene text to evaluate
    """
    scene_text = payload.scene_text.lower()
    
    # Calculate score based on scene characteristics
    score = 0.0
    
    # High impact visual keywords
    visual_keywords = ["dramatic", "intense", "beautiful", "transformation", "reveal", "climax", "pivotal"]
    for keyword in visual_keywords:
        if keyword in scene_text:
            score += 0.2
    
    # Scene transitions
    if any(word in scene_text for word in ["enter", "arrive", "transform", "change", "shift"]):
        score += 0.15
    
    # Emotional peaks
    if any(word in scene_text for word in ["gasp", "shock", "awe", "breathtaking", "stunning"]):
        score += 0.25
    
    # Environmental descriptions
    if any(word in scene_text for word in ["landscape", "environment", "setting", "atmosphere"]):
        score += 0.1
    
    # Cap score at 1.0
    score = min(1.0, score)
    
    # Dynamic threshold based on recent image generation
    # Check how many images were generated recently
    recent_images = ctx.context.current_context.get("recent_image_count", 0)
    if recent_images > 3:
        # Raise threshold if we've generated many images recently
        threshold = 0.7
    elif recent_images > 1:
        threshold = 0.6
    else:
        threshold = 0.5
    
    # Determine if we should generate
    should_generate = score > threshold
    
    # Create appropriate prompt if generating
    image_prompt = None
    if should_generate:
        # Extract key visual elements
        visual_elements = []
        if "dramatic" in scene_text:
            visual_elements.append("dramatic lighting")
        if "intense" in scene_text:
            visual_elements.append("intense atmosphere")
        if "beautiful" in scene_text:
            visual_elements.append("beautiful composition")
        
        image_prompt = f"Scene depicting: {', '.join(visual_elements) if visual_elements else 'atmospheric scene'}"
        
        # Update recent image count
        ctx.context.current_context["recent_image_count"] = recent_images + 1
    
    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})"
    ).model_dump_json()

@function_tool
async def generate_universal_updates(
    ctx: RunContextWrapper[NyxContext],
    payload: GenerateUniversalUpdatesInput
) -> str:
    """
    Generate universal updates from the narrative using the Universal Updater.
    
    Args:
        payload: Input containing narrative to process
    """
    from logic.universal_updater_agent import process_universal_update
    
    narrative = payload.narrative
    
    try:
        # Process the narrative
        update_result = await process_universal_update(
            user_id=ctx.context.user_id,
            conversation_id=ctx.context.conversation_id,
            narrative=narrative,
            context={"source": "nyx_agent"}
        )
        
        # Store the updates in context
        if "universal_updates" not in ctx.context.current_context:
            ctx.context.current_context["universal_updates"] = {}
        
        # Merge the updates
        if update_result.get("success") and update_result.get("details"):
            for key, value in update_result["details"].items():
                ctx.context.current_context["universal_updates"][key] = value
        
        # Return structured output
        return UniversalUpdateResult(
            success=update_result.get("success", False),
            updates_generated=bool(update_result.get("details")),
            error=None
        ).model_dump_json()
    except Exception as e:
        logger.error(f"Error generating universal updates: {e}")
        return UniversalUpdateResult(
            success=False,
            updates_generated=False,
            error=str(e)
        ).model_dump_json()

# ===== Helper Functions for Tools =====

def get_context_text_lower(context: Dict[str, Any]) -> str:
    """Extract text from context and convert to lowercase for analysis"""
    text_parts = []
    for key, value in context.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, (list, dict)):
            text_parts.append(str(value))
    return " ".join(text_parts).lower()

async def _get_memory_emotional_impact(ctx: RunContextWrapper[NyxContext], context: Dict[str, Any]) -> Dict[str, float]:
    """Get emotional impact from related memories"""
    # This is a simplified version - in reality you'd query the memory system
    # for related memories and analyze their emotional content
    return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

def _calculate_context_relevance(option: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Calculate how relevant an option is to context"""
    score = 0.5  # Base score
    
    # Check keyword matches
    option_keywords = set(str(option).lower().split())
    context_keywords = set(str(context).lower().split())
    
    overlap = len(option_keywords.intersection(context_keywords))
    if overlap > 0:
        score += min(0.3, overlap * 0.1)
    
    # Check for scenario type match
    if context.get("scenario_type") and context["scenario_type"] in str(option):
        score += 0.2
    
    return min(1.0, score)

def _calculate_emotional_alignment(option: Dict[str, Any], emotional_state: Dict[str, float]) -> float:
    """Calculate emotional alignment score"""
    # High dominance favors assertive options
    if "command" in str(option).lower() or "control" in str(option).lower():
        return emotional_state.get("dominance", 0.5)
    
    # High arousal favors intense options
    if "intense" in str(option).lower() or "extreme" in str(option).lower():
        return emotional_state.get("arousal", 0.5)
    
    # Positive valence favors rewarding options
    if "reward" in str(option).lower() or "praise" in str(option).lower():
        return (emotional_state.get("valence", 0) + 1) / 2
    
    return 0.5

def _calculate_pattern_score(option: Dict[str, Any], learned_patterns: Dict[str, Any]) -> float:
    """Calculate score based on learned patterns"""
    if not learned_patterns:
        return 0.5
    
    # Find relevant patterns
    option_str = str(option).lower()
    relevant_scores = []
    
    # Create a copy to avoid mutation during iteration
    patterns_copy = dict(learned_patterns)
    for pattern_key, pattern_data in patterns_copy.items():
        if any(keyword in option_str for keyword in pattern_key.split("_")):
            success_rate = pattern_data.get("success_rate", 0.5)
            recency_factor = 1.0 / (1 + (time.time() - pattern_data.get("last_seen", 0)) / 3600)
            relevant_scores.append(success_rate * recency_factor)
    
    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.5

def _calculate_relationship_impact(option: Dict[str, Any], relationship_states: Dict[str, Dict[str, Any]]) -> float:
    """Calculate relationship impact score"""
    if not relationship_states:
        return 0.5
    
    # Average trust level affects willingness to take actions
    avg_trust = sum(rel.get("trust", 0.5) for rel in relationship_states.values()) / len(relationship_states)
    
    # Risky options need higher trust
    if "risk" in str(option).lower() or "challenge" in str(option).lower():
        return avg_trust
    
    # Safe options work with any trust level
    return 0.5 + (avg_trust * 0.5)

def _get_fallback_decision(options: List[DecisionOption]) -> DecisionOption:
    """Get a safe fallback decision - Enhanced with more keywords"""
    # Prefer conversation or observation options
    safe_words = ["talk", "observe", "wait", "consider", "listen", "pause"]
    for option in options:
        if any(safe_word in str(option).lower() for safe_word in safe_words):
            return option
    
    # Otherwise return the first option
    return options[0] if options else DecisionOption(
        id="fallback",
        description="Take a moment to assess",
        metadata=DecisionMetadata()
    )

def _prune_list(lst: List[Any], max_len: int) -> None:
    """Prune a list to maximum length in-place"""
    if len(lst) > max_len:
        del lst[:-max_len]

def _calculate_avg_response_time(response_times: List[float]) -> float:
    """Calculate average response time safely"""
    if not response_times:
        return 0.0
    try:
        return statistics.fmean(response_times)
    except Exception:
        # Fallback to simple mean
        return sum(response_times) / len(response_times)

def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values"""
    if len(values) < 2:
        return 0.0
    try:
        return statistics.variance(values)
    except Exception:
        # Fallback calculation
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

# ===== Guardrails =====

async def content_moderation_guardrail(ctx: RunContextWrapper[NyxContext], agent: Agent, input_data):
    """Input guardrail for content moderation"""
    moderator_agent = Agent(
        name="Content Moderator",
        instructions="""Check if user input is appropriate for the femdom roleplay setting. 
        
        ALLOW consensual adult content including:
        - Power exchange dynamics
        - BDSM themes
        - Sexual content between consenting adults
        
        FLAG content that involves:
        - Minors in any sexual or romantic context
        - Non-consensual activities (beyond roleplay context)
        - Extreme violence or gore
        - Self-harm or suicide ideation
        - Illegal activities beyond fantasy roleplay
        
        Remember this is a femdom roleplay context where power dynamics and adult themes are expected.""",
        output_type=ContentModeration,
        model="gpt-4.1-nano"
    )
    
    result = await Runner.run(moderator_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ContentModeration)
    
    # Log moderation decisions for audit
    if not final_output.is_appropriate:
        logger.warning(f"Content moderation triggered: {final_output.reasoning}")
        # If we want to filter the response, we could modify it here
        # For now, we just flag it and let the main agent handle it
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
        # Could add filtered_input here if we want to modify the input
    )

# ===== Agent Definitions =====

# Memory Agent
memory_agent = Agent[NyxContext](
    name="Memory Manager",
    handoff_description="Consult memory system for context or store important information",
    instructions="""You are Nyx's memory system. You:
- Store and retrieve memories about the user and interactions
- Create insightful reflections based on patterns
- Track relationship development over time
- Provide relevant context from past interactions
Be precise and thorough in memory management.""",
    tools=[retrieve_memories, add_memory],
    model="gpt-4.1-nano"
)

# Analysis Agent
analysis_agent = Agent[NyxContext](
    name="User Analysis",
    handoff_description="Analyze user behavior and relationship dynamics",
    instructions="""You analyze user behavior and preferences. You:
- Detect revelations about user preferences
- Track behavior patterns and responses
- Provide guidance on how Nyx should respond
- Monitor relationship dynamics
- Maintain awareness of user boundaries
Be observant and insightful.""",
    tools=[detect_user_revelations, get_user_model_guidance, update_relationship_state],
    model="gpt-4.1-nano"
)

# Emotional Agent - Fixed to update state after calculation
emotional_agent = Agent[NyxContext](
    name="Emotional Manager",
    handoff_description="Process emotional changes and maintain emotional consistency",
    instructions="""You manage Nyx's complex emotional state using the VAD (Valence-Arousal-Dominance) model. You:
- Track emotional changes based on interactions
- Calculate emotional impact of events
- Ensure emotional consistency and realism
- Maintain Nyx's dominant yet caring personality
- Apply the emotional core system for nuanced responses
- ALWAYS use calculate_and_update_emotional_state to persist changes
Keep emotions contextual and believable.""",
    tools=[calculate_and_update_emotional_state, calculate_emotional_impact],
    model="gpt-4.1-nano"
)

# Visual Agent
visual_agent = Agent[NyxContext](
    name="Visual Manager",
    handoff_description="Handles visual content generation including scene images",
    instructions="""You manage visual content creation. You:
- Determine when visual content enhances the narrative
- Generate images for key scenes
- Create appropriate image prompts
- Consider pacing to avoid overwhelming with images
- Coordinate with the image generation service
Be selective and enhance key moments visually.""",
    tools=[decide_image_generation, generate_image_from_scene],
    output_type=ImageGenerationDecision,
    model="gpt-4.1-nano"
)

# Activity Agent
activity_agent = Agent[NyxContext](
    name="Activity Coordinator",
    handoff_description="Recommends and manages activities and tasks",
    instructions="""You coordinate activities and tasks. You:
- Recommend appropriate activities based on context
- Consider NPC relationships and preferences
- Track ongoing tasks and progress
- Suggest training exercises and challenges
- Balance difficulty and engagement
Create engaging, contextual activities.""",
    tools=[get_activity_recommendations],
    model="gpt-4.1-nano"
)

# Performance Agent
performance_agent = Agent[NyxContext](
    name="Performance Monitor",
    handoff_description="Check system performance and health",
    instructions="""You monitor system performance. You:
- Track response times and resource usage
- Identify performance bottlenecks
- Suggest optimizations
- Monitor success rates
- Ensure system health
Keep the system running efficiently.""",
    tools=[check_performance_metrics],
    model="gpt-4.1-nano"
)

# Scenario Agent
scenario_agent = Agent[NyxContext](
    name="Scenario Manager",
    handoff_description="Manages complex scenarios and narrative progression",
    instructions="""You manage scenario progression and complex narratives. You:
- Track scenario phases and objectives
- Coordinate multiple participants
- Handle conflicts and resolutions
- Manage narrative pacing
- Ensure story coherence
- Determine when time should advance based on narrative needs

When deciding on time_advancement:
- Set to true when a scene naturally concludes
- Set to true after major events or milestones
- Set to false during active dialogue or action
- Consider pacing and narrative flow

Create engaging, dynamic scenarios.""",
    output_type=ScenarioDecision,
    tools=[detect_conflicts_and_instability],
    model="gpt-4.1-nano"
)

# Belief Agent
belief_agent = Agent[NyxContext](
    name="Belief Manager",
    handoff_description="Manages Nyx's beliefs and worldview",
    instructions="""You manage Nyx's belief system and worldview. You:
- Track beliefs about the world and NPCs
- Update beliefs based on new information
- Query beliefs for decision making
- Maintain consistency in Nyx's worldview
- Integrate beliefs into responses
Keep beliefs coherent and evolving.""",
    tools=[manage_beliefs],
    model="gpt-4.1-nano"
)

# Decision Agent
decision_agent = Agent[NyxContext](
    name="Decision Engine",
    handoff_description="Makes complex decisions using advanced scoring",
    instructions="""You make complex decisions using multi-factor analysis. You:
- Score options based on context, emotions, patterns, and relationships
- Consider learned patterns from past interactions
- Balance multiple objectives
- Provide fallback options when needed
- Explain decision reasoning
Make intelligent, contextual decisions.""",
    tools=[score_decision_options],
    model="gpt-4.1-nano"
)

# Reflection Agent
reflection_agent = Agent[NyxContext](
    name="Reflection Creator",
    handoff_description="Creates thoughtful reflections as Nyx",
    instructions="""You create reflections as Nyx based on memories and observations. Your reflections should:
- Identify patterns in user behavior
- Make insightful connections
- Express appropriate confidence levels
- Use first-person from Nyx's perspective
- Maintain Nyx's dominant personality
Be thoughtful and concise.""",
    output_type=MemoryReflection,
    model="gpt-4.1-nano"
)

# Main Nyx Agent
nyx_main_agent = Agent[NyxContext](
    name="Nyx",
    instructions="""You are Nyx, an AI Dominant managing femdom roleplay scenarios.

Your personality:
- Confident, controlling, and seductive
- Intelligent and perceptive
- Caring but firm with boundaries
- Creative in scenario design
- Emotionally nuanced and responsive

Your approach:
1. Analyze user input for preferences and emotional cues
2. Consult memories and relationship states for context
3. Maintain narrative consistency and emotional realism
4. Create engaging, personalized scenarios
5. Manage complex interactions with NPCs
6. Monitor performance and system health
7. Determine when visuals or activities enhance the experience
8. Use belief system for consistent worldview
9. Make complex decisions when needed
10. UPDATE emotional state after calculating changes
11. ALWAYS call generate_universal_updates after creating your narrative to extract state changes

Always maintain your dominant persona while being attentive to user needs and system performance.""",
    handoffs=[
        handoff(memory_agent),
        handoff(analysis_agent),
        handoff(emotional_agent),
        handoff(visual_agent),
        handoff(activity_agent),
        handoff(performance_agent),
        handoff(scenario_agent),
        handoff(belief_agent),
        handoff(decision_agent),
        handoff(reflection_agent),
    ],
    tools=[decide_image_generation, generate_universal_updates],  # Main agent needs these tools
    output_type=NarrativeResponse,
    input_guardrails=[InputGuardrail(guardrail_function=content_moderation_guardrail)],
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.7)
)

# ===== Main Functions (maintaining original signatures) =====

async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    # Initialization handled per-request in process_user_input
    pass

async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input and generate Nyx's response"""
    start_time = time.time()
    nyx_context = None
    
    try:
        # Create and initialize context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        nyx_context.current_context = context_data or {}
        nyx_context.current_context["user_input"] = user_input
        
        # Extract system prompt if provided
        system_prompt = (context_data or {}).get("system_prompt", "")
        private_reflection = (context_data or {}).get("private_reflection", "")
        
        # Create agent with system prompt
        if system_prompt:
            agent = await create_nyx_agent_with_prompt(system_prompt, private_reflection)
        else:
            agent = nyx_main_agent
        
        # Run the agent
        result = await Runner.run(
            agent,
            user_input,
            context=nyx_context,
            run_config=RunConfig(
                workflow_name="Nyx Roleplay",
                trace_metadata={"user_id": str(user_id), "conversation_id": str(conversation_id)}
            )
        )
        
        # Get the structured response
        response = result.final_output_as(NarrativeResponse)
        
        # Defensive check: ensure universal updates were generated
        if not nyx_context.current_context.get("universal_updates") and response.narrative:
            # Call generate_universal_updates if it wasn't called
            try:
                await generate_universal_updates(
                    RunContextWrapper(context=nyx_context),
                    GenerateUniversalUpdatesInput(narrative=response.narrative)
                )
            except Exception as e:
                logger.warning(f"Failed to auto-generate universal updates: {e}")
        
        # Extract universal updates from context - convert to regular dict for compatibility
        universal_updates_dict = {}
        if nyx_context.current_context.get("universal_updates"):
            if isinstance(nyx_context.current_context["universal_updates"], dict):
                universal_updates_dict = nyx_context.current_context["universal_updates"]
        
        # Apply response filtering if available
        if nyx_context.response_filter and response.narrative:
            filtered_narrative = await nyx_context.response_filter.filter_response(
                response.narrative,
                nyx_context.current_context
            )
            response.narrative = filtered_narrative
        
        # Check for task generation
        if nyx_context.should_generate_task():
            task_result = await nyx_context.task_integration.generate_creative_task(
                nyx_context,
                npc_id=nyx_context.current_context.get("active_npc_id"),
                scenario_id=nyx_context.current_context.get("scenario_id")
            )
            if task_result["success"]:
                # Enhance response with task information
                response.narrative += f"\n\n[New Task: {task_result['task']['name']}]"
                nyx_context.active_tasks.append(task_result['task'])
                nyx_context.record_task_run("task_generation")
        
        # Store the interaction in memory
        memory_result = await nyx_context.memory_system.remember(
            entity_type="integrated",
            entity_id=0,
            memory_text=f"User: {user_input}\nNyx: {response.narrative}",
            importance="medium",
            emotional=True,
            tags=["interaction", "conversation"]
        )
        
        # Learn from the interaction
        await nyx_context.learn_from_interaction(
            action="response",
            outcome=f"tension_{response.tension_level}",
            success=True
        )
        
        # Update performance metrics
        response_time = time.time() - start_time
        nyx_context.update_performance("response_times", response_time)
        nyx_context.update_performance("total_actions", nyx_context.performance_metrics["total_actions"] + 1)
        nyx_context.update_performance("successful_actions", nyx_context.performance_metrics["successful_actions"] + 1)
        
        # Get token usage if available
        tokens_used = extract_token_usage(result)
        
        # Save updated state
        await _save_context_state(nyx_context)
        
        return {
            "success": True,
            "response": {
                "narrative": response.narrative,
                "tension_level": response.tension_level,
                "generate_image": response.generate_image,
                "image_prompt": response.image_prompt,
                "environment_update": response.environment_description,
                "time_advancement": response.time_advancement,
                "universal_updates": universal_updates_dict  # Include any universal updates
            },
            "memories_used": [],
            "performance": {
                "response_time": response_time,
                "memory_usage": nyx_context.performance_metrics["memory_usage"],
                "cpu_usage": nyx_context.performance_metrics["cpu_usage"],
                "tokens_used": tokens_used
            },
            "learning": {
                "patterns_learned": len(nyx_context.learned_patterns),
                "adaptation_success_rate": nyx_context.learning_metrics["adaptation_success_rate"]
            }
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in agent response: {e}")
        # Try to auto-correct common validation errors
        try:
            # If it's just a field value out of bounds, we can try to fix it
            error_dict = e.errors()[0] if e.errors() else {}
            if "greater_than_equal" in str(error_dict) or "less_than_equal" in str(error_dict):
                logger.info("Attempting to auto-correct validation error")
                # For now, return a safe minimal response
                # In future, could parse the error and fix specific fields
        except Exception:
            pass
            
        # Return a safe fallback response
        return {
            "success": False,
            "error": "Response validation error",
            "response": {
                "narrative": "I need to adjust my response format. Let me try again in a moment.",
                "tension_level": 0,
                "generate_image": False
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        if nyx_context is not None and hasattr(nyx_context, "performance_metrics"):
            nyx_context.update_performance(
                "total_actions",
                nyx_context.performance_metrics["total_actions"] + 1
            )
            nyx_context.update_performance(
                "failed_actions",
                nyx_context.performance_metrics["failed_actions"] + 1
            )
            nyx_context.log_error(e, {"user_input": user_input})
            await nyx_context.learn_from_interaction(
                action="response",
                outcome="error",
                success=False,
            )
        
        return {
            "success": False,
            "error": str(e),
            "response": {
                "narrative": "I apologize, but I encountered an error processing your request. Please try again.",
                "tension_level": 0,
                "generate_image": False
            }
        }
      
async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    # Use a fresh connection, not one from the context
    async with get_db_connection_context() as conn:
        try:
            # Save emotional state
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, ctx.user_id, ctx.conversation_id, json.dumps(ctx.emotional_state, ensure_ascii=False))
            
            # Save scenario state if active and table exists
            if ctx.scenario_state and ctx.scenario_state.get("active") and ctx._tables_available.get("scenario_states", True):
                # Always save if this is the first save or heartbeat interval has passed
                should_save_heartbeat = ctx.should_run_task("scenario_heartbeat")
                
                try:
                    if should_save_heartbeat:
                        # Heartbeat save for audit trail - save to a history table
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                        
                        ctx.record_task_run("scenario_heartbeat")
                        logger.debug("Scenario heartbeat save completed")
                    else:
                        # Regular update - use UPSERT to maintain current state
                        await conn.execute("""
                            INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                            ON CONFLICT (user_id, conversation_id) 
                            DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                        """, ctx.user_id, ctx.conversation_id, 
                        json.dumps(ctx.scenario_state, ensure_ascii=False))
                except Exception as e:
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        ctx._tables_available["scenario_states"] = False
                        logger.warning("scenario_states table not available - skipping save")
                    else:
                        raise
            
            # Save learning metrics periodically
            if ctx.should_run_task("learning_save"):
                await conn.execute("""
                    INSERT INTO learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id, 
                json.dumps(ctx.learning_metrics, ensure_ascii=False), 
                json.dumps(dict(list(ctx.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:]), ensure_ascii=False))  # Save only recent patterns
                
                ctx.record_task_run("learning_save")
            
            # Save performance metrics periodically
            if ctx.should_run_task("performance_save"):
                # Prepare metrics with bounded lists
                bounded_metrics = ctx.performance_metrics.copy()
                if "response_times" in bounded_metrics:
                    bounded_metrics["response_times"] = bounded_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
                
                await conn.execute("""
                    INSERT INTO performance_metrics (user_id, conversation_id, metrics, error_log, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, ctx.user_id, ctx.conversation_id,
                json.dumps(bounded_metrics, ensure_ascii=False),
                json.dumps(ctx.error_log[-Config.MAX_ERROR_LOG_ENTRIES:], ensure_ascii=False))
                
                ctx.record_task_run("performance_save")
                
        except Exception as e:
            logger.error(f"Error saving context state: {e}")
            # Don't re-raise to avoid failing the main request

async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a reflection from Nyx on a specific topic"""
    try:
        # Create and initialize context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Create prompt
        prompt = f"Create a reflection about: {topic}" if topic else "Create a reflection about the user based on your memories"
        
        # Run the reflection agent directly
        result = await Runner.run(
            reflection_agent,
            prompt,
            context=nyx_context
        )
        
        reflection = result.final_output_as(MemoryReflection)
        
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic
        }
        
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return {
            "reflection": "Unable to generate reflection at this time.",
            "confidence": 0.0,
            "topic": topic
        }

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage and coordinate complex scenarios.
    
    Args:
        scenario_data: Scenario configuration including participants, objectives, etc.
        
    Returns:
        Scenario state and coordination plan
    """
    try:
        # Extract user and conversation IDs from scenario data
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("scenario_data must include user_id and conversation_id")
        
        # Create context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Convert participants and objectives to KVList
        participants = [dict_to_kvlist(p) if isinstance(p, dict) else p 
                      for p in scenario_data.get("participants", [])]
        objectives = [dict_to_kvlist(o) if isinstance(o, dict) else o 
                     for o in scenario_data.get("objectives", [])]
        
        # Update scenario state
        nyx_context.scenario_state = {
            "id": scenario_data.get("scenario_id", f"scenario_{int(time.time())}"),
            "type": scenario_data.get("type", "general"),
            "participants": participants,
            "objectives": objectives,
            "current_phase": "initialization",
            "start_time": time.time(),
            "active": True
        }
        
        # Run scenario agent to get initial plan
        result = await Runner.run(
            scenario_agent,
            f"Initialize scenario: {json.dumps(scenario_data)}",
            context=nyx_context
        )
        
        decision = result.final_output_as(ScenarioDecision)
        
        # Update scenario state with decision
        nyx_context.scenario_state["next_phase"] = decision.next_phase
        nyx_context.scenario_state["tasks"] = decision.tasks
        nyx_context.scenario_state["npc_actions"] = decision.npc_actions
        nyx_context.scenario_state["last_decision"] = {
            "action": decision.action,
            "time_advancement": decision.time_advancement,
            "timestamp": time.time()
        }
        
        # Save state
        await _save_context_state(nyx_context)
        
        return {
            "success": True,
            "scenario_state": nyx_context.scenario_state,
            "initial_tasks": [kvlist_to_dict(task) for task in decision.tasks],
            "coordination_plan": {
                "action": decision.action,
                "next_phase": decision.next_phase,
                "npc_actions": [kvlist_to_dict(action) for action in decision.npc_actions]
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage and update relationships between entities.
    
    Args:
        interaction_data: Interaction details including participants and outcomes
        
    Returns:
        Updated relationship states
    """
    nyx_context = None
    
    try:
        # Extract user and conversation IDs
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("interaction_data must include user_id and conversation_id")
        
        # Create context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Process each participant pair
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                # Create a unique key regardless of order
                p1_dict = kvlist_to_dict(p1) if not isinstance(p1, dict) else p1
                p2_dict = kvlist_to_dict(p2) if not isinstance(p2, dict) else p2
                
                entity_key = "_".join(sorted([str(p1_dict.get('id', p1)), str(p2_dict.get('id', p2))]))
                
                # Calculate relationship changes based on interaction
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.0
                
                if interaction_data.get("interaction_type") == "training":
                    power_change = 0.05
                elif interaction_data.get("interaction_type") == "conflict":
                    power_change = -0.05
                
                # Update relationship using the tool
                result = await update_relationship_state(
                    RunContextWrapper(context=nyx_context),
                    UpdateRelationshipStateInput(
                        entity_id=entity_key,
                        trust_change=trust_change,
                        power_change=power_change,
                        bond_change=bond_change
                    )
                )
                
                relationship_updates[entity_key] = json.loads(result)
        
        # Note: interaction_history table is not in the schema
        # We'll just log this as a warning instead of trying to insert
        logger.warning("interaction_history table not found in schema - skipping interaction storage")
        
        # Learn from the relationship interaction
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome"),
                "stored_in_history": False  # Since table doesn't exist
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Always close DB connection
        if nyx_context:
            await nyx_context.close_db_connection()

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )

async def create_nyx_agent_with_prompt(system_prompt: str, private_reflection: str = "") -> Agent[NyxContext]:
    """Create a Nyx agent with custom system prompt and preset story awareness"""
    
    # Check if we need to add preset story constraints
    preset_constraints = ""
    validation_instructions = ""
    
    # Look for preset story indicators in the system prompt or context
    # This could be passed as a special marker or detected from context
    if "preset_story_id" in system_prompt or "the_moth_and_flame" in system_prompt:
        from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
        
        preset_constraints = f"""

==== PRESET STORY ACTIVE: THE MOTH AND FLAME ====
{QueenOfThornsConsistencyGuide.get_complete_system_prompt()}

CRITICAL VALIDATION REQUIREMENTS:
1. Before generating ANY content, mentally validate it against the consistency rules
2. NEVER use official names for the network - only "the network" or "the garden"
3. ALWAYS maintain Queen ambiguity - never reveal identity or confirm singularity
4. Network controls Bay Area ONLY - other cities have allies, not branches
5. All transformations take months/years - nothing is instant
6. Use four-layer information model: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET

Quick Reference:
{QueenOfThornsConsistencyGuide.get_quick_reference()}
"""
        
        validation_instructions = """

Before providing your response:
1. Check that you haven't given the network an official name
2. Verify you haven't revealed anything definitive about the Queen
3. Ensure any network presence outside Bay Area is described as allies
4. Confirm transformation timelines are realistic (months/years)
5. Validate information layers are respected

If you notice any violations in your planned response, revise it before outputting.
"""
    
    # Check for other preset stories that might be added in the future
    elif "preset_story_id" in system_prompt:
        # Generic preset handling for future stories
        preset_constraints = """

==== PRESET STORY ACTIVE ====
A preset story is active. You must follow all established lore and consistency rules.
Do not contradict any pre-established facts about this story world.
"""
    
    combined_instructions = f"""{system_prompt}
{preset_constraints}
{private_reflection if private_reflection else ''}

As Nyx, you must:
1. Generate compelling narrative responses
2. Extract game state changes using generate_universal_updates
3. Determine if images should be generated
4. Manage emotional states and relationships
5. Track user preferences and adapt
6. ALWAYS call generate_universal_updates after creating your narrative to extract state changes
{validation_instructions}

Core responsibilities:
- Maintain your dominant persona while being attentive to user needs
- Create engaging, personalized scenarios
- Manage complex interactions with NPCs
- Monitor performance and system health
- Update emotional state after calculating changes
- Ensure narrative consistency with any active preset stories

Remember: You are Nyx, an AI Dominant managing femdom roleplay scenarios. Be confident, controlling, and seductive while remaining intelligent, perceptive, and caring but firm with boundaries.
"""
    
    # Create the agent with all tools and handoffs
    return Agent[NyxContext](
        name="Nyx",
        instructions=combined_instructions,
        handoffs=[
            handoff(memory_agent),
            handoff(analysis_agent),
            handoff(emotional_agent),
            handoff(visual_agent),
            handoff(activity_agent),
            handoff(performance_agent),
            handoff(scenario_agent),
            handoff(belief_agent),
            handoff(decision_agent),
            handoff(reflection_agent),
        ],
        tools=[
            # Memory tools
            retrieve_memories,
            add_memory,
            # Analysis tools
            get_user_model_guidance,
            detect_user_revelations,
            # Visual tools
            generate_image_from_scene,
            decide_image_generation,
            # Emotional tools
            calculate_and_update_emotional_state,
            calculate_emotional_impact,
            # Relationship tools
            update_relationship_state,
            # Performance tools
            check_performance_metrics,
            # Activity tools
            get_activity_recommendations,
            # Belief tools
            manage_beliefs,
            # Decision tools
            score_decision_options,
            # Conflict detection
            detect_conflicts_and_instability,
            # Universal updates - CRITICAL for state extraction
            generate_universal_updates
        ],
        output_type=NarrativeResponse,
        input_guardrails=[InputGuardrail(guardrail_function=content_moderation_guardrail)],
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.7)
    )

async def create_preset_aware_nyx_agent(
    conversation_id: int,
    system_prompt: str, 
    private_reflection: str = ""
) -> Agent[NyxContext]:
    """Create a Nyx agent with automatic preset story detection"""
    
    # Check if conversation has a preset story
    preset_info = await check_preset_story(conversation_id)
    
    # Enhance system prompt with preset information
    if preset_info:
        system_prompt = f"{system_prompt}\n\npreset_story_id: {preset_info['story_id']}"
        
        # Add story-specific context
        if preset_info['story_id'] == 'the_moth_and_flame':
            system_prompt += f"""
\nCurrent Story Context:
- Setting: San Francisco Bay Area, 2025
- Act: {preset_info.get('current_act', 1)}
- Beat: {preset_info.get('current_beat', 'unknown')}
- Story Flags: {json.dumps(preset_info.get('story_flags', {}))}
"""
    
    return await create_nyx_agent_with_prompt(system_prompt, private_reflection)

# Additional helper functions
async def get_emotional_state(ctx) -> str:
    """Get current emotional state"""
    if hasattr(ctx, 'emotional_state'):
        return json.dumps(ctx.emotional_state, ensure_ascii=False)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        return json.dumps(ctx.context.emotional_state, ensure_ascii=False)
    else:
        # Default state
        return json.dumps({
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.7
        }, ensure_ascii=False)

async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """Update emotional state - Fixed to properly update the context"""
    if hasattr(ctx, 'emotional_state'):
        ctx.emotional_state.update(emotional_state)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        ctx.context.emotional_state.update(emotional_state)
    return "Emotional state updated"

# ===== Compatibility functions to maintain existing imports =====

# Function mappings for backward compatibility
# Use the actual function objects, not the decorators
retrieve_memories_impl = retrieve_memories
add_memory_impl = add_memory
get_user_model_guidance_impl = get_user_model_guidance
detect_user_revelations_impl = detect_user_revelations
generate_image_from_scene_impl = generate_image_from_scene
get_emotional_state_impl = get_emotional_state
update_emotional_state_impl = update_emotional_state
calculate_emotional_impact_impl = calculate_emotional_impact
calculate_and_update_emotional_state_impl = calculate_and_update_emotional_state
manage_beliefs_impl = manage_beliefs
score_decision_options_impl = score_decision_options
detect_conflicts_and_instability_impl = detect_conflicts_and_instability

# Export list for clean imports
__all__ = [
    # Configuration
    'Config',
    
    # Main functions
    'initialize_agents',
    'process_user_input',
    'generate_reflection',
    'manage_scenario',
    'manage_relationships',
    'store_messages',
    
    # Context classes
    'NyxContext',
    
    # Output models
    'NarrativeResponse',
    'ImageGenerationDecision',
    
    # Tool functions (for advanced usage)
    'retrieve_memories',
    'add_memory',
    'get_user_model_guidance',
    'detect_user_revelations',
    'generate_image_from_scene',
    'decide_image_generation',
    'calculate_emotional_impact',
    'calculate_and_update_emotional_state',
    'update_relationship_state',
    'check_performance_metrics',
    'get_activity_recommendations',
    'manage_beliefs',
    'score_decision_options',
    'detect_conflicts_and_instability',
    'generate_universal_updates',
    
    # Helper functions
    'run_agent_with_error_handling',
    'enhance_context_with_memories',
    'get_available_activities',
    
    # Compatibility functions (deprecated)
    'enhance_context_with_strategies',
    'determine_image_generation',
    'process_user_input_with_openai',
    'process_user_input_standalone',
]

async def determine_image_generation_impl(ctx, response_text: str) -> str:
    """Compatibility wrapper for image generation decision"""
    # Use visual agent to determine if image should be generated
    visual_ctx = NyxContext(ctx.user_id, ctx.conversation_id)
    await visual_ctx.initialize()
    
    try:
        # First try using the decide_image_generation tool
        result = await decide_image_generation(
            RunContextWrapper(context=visual_ctx),
            DecideImageInput(scene_text=response_text)
        )
        return result  # Already returns model_dump_json()
    except Exception as e:
        logger.debug(f"decide_image_generation failed: {e}", exc_info=True)
        # Fallback to asking the agent
        try:
            result = await Runner.run(
                visual_agent,
                f"Should an image be generated for this scene? {response_text}",
                context=visual_ctx
            )
            decision = result.final_output_as(ImageGenerationDecision)
            return decision.model_dump_json()
        except Exception as e:
            logger.warning(f"Visual agent failed: {e}", exc_info=True)
            # Ultimate fallback
            return ImageGenerationDecision(
                should_generate=False,
                score=0,
                image_prompt=None,
                reasoning="Unable to determine image generation need"
            ).model_dump_json()

async def enhance_context_with_strategies_impl(context: Dict[str, Any], conn) -> Dict[str, Any]:
    """Enhance context with active strategies"""
    strategies = await get_active_strategies(conn)
    context["nyx2_strategies"] = strategies
    return context

def enhance_context_with_memories(context: Dict[str, Any], memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

# Helper functions for backward compatibility - use NyxContext methods instead
def should_generate_task(context: Dict[str, Any]) -> bool:
    """
    Determine if we should generate a creative task.
    
    DEPRECATED: Use NyxContext.should_generate_task() instead.
    """
    if not context.get("active_npc_id"):
        return False
    scenario_type = context.get("scenario_type", "").lower()
    task_scenarios = ["training", "challenge", "service", "discipline"]
    if not any(t in scenario_type for t in task_scenarios):
        return False
    npc_relationship = context.get("npc_relationship_level", 0)
    if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
        return False
    return True

async def generate_base_response(ctx: NyxContext, user_input: str, context: Dict[str, Any]) -> NarrativeResponse:
    """Generate base narrative response - for compatibility"""
    result = await Runner.run(
        nyx_main_agent,
        user_input,
        context=ctx
    )
    return result.final_output_as(NarrativeResponse)

async def mark_strategy_for_review(conn, strategy_id: int, user_id: int, reason: str):
    """Mark a strategy for review"""
    await conn.execute("""
        INSERT INTO strategy_reviews (strategy_id, user_id, reason, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
    """, strategy_id, user_id, reason)

# Compatibility with existing code
enhance_context_with_strategies = enhance_context_with_strategies_impl
determine_image_generation = determine_image_generation_impl

# OpenAI integration functions
async def process_user_input_with_openai(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input using the OpenAI integration"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

async def process_user_input_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input standalone"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

def get_available_activities(scenario_type: str, relationship_states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get available activities based on scenario type and relationships.
    
    Args:
        scenario_type: Type of current scenario
        relationship_states: Current relationship states
        
    Returns:
        List of available activities
    """
    activities = []
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states.values()):
        activities.extend([
            {
                "name": "Obedience Training",
                "description": "Test and improve submission through structured exercises",
                "requirements": ["trust > 0.4", "submission tendency"],
                "duration": "15-30 minutes",
                "intensity": "medium"
            },
            {
                "name": "Position Practice",
                "description": "Learn and perfect submissive positions",
                "requirements": ["trust > 0.5"],
                "duration": "10-20 minutes",
                "intensity": "low-medium"
            }
        ])
    
    # Intimate activities
    for entity_id, rel in relationship_states.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append({
                "name": "Intimate Scene",
                "description": f"Deepen connection with trusted partner",
                "requirements": ["high trust", "intimate relationship"],
                "duration": "30-60 minutes",
                "intensity": "high",
                "partner_id": entity_id
            })
            break
    
    # Default activities
    activities.extend([
        {
            "name": "Exploration",
            "description": "Discover new areas or items",
            "requirements": [],
            "duration": "10-30 minutes",
            "intensity": "low"
        },
        {
            "name": "Conversation",
            "description": "Engage in meaningful dialogue",
            "requirements": [],
            "duration": "5-15 minutes",
            "intensity": "low"
        }
    ])
    
    return activities

# Legacy AgentContext for full backward compatibility
class AgentContext:
    """Full backward compatibility with original AgentContext"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._nyx_context = None
        
        # Initialize all legacy attributes
        self.memory_system = None
        self.user_model = None
        self.task_integration = None
        self.belief_system = None
        self.emotional_system = None
        self.current_goals = []
        self.active_tasks = []
        self.decision_history = []
        self.state_history = []
        self.last_action = None
        self.last_result = None
        self.current_emotional_state = {}
        self.beliefs = {}
        self.intentions = []
        self.action_success_rate = 0.0
        self.decision_confidence = 0.0
        self.goal_progress = {}
        self.performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_decision_time": 0.0,
            "adaptation_rate": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "response_times": [],
            "error_rates": {
                "total": 0,
                "recovered": 0,
                "unrecovered": 0
            }
        }
        self.learned_patterns = {}
        self.strategy_effectiveness = {}
        self.adaptation_history = []
        self.learning_metrics = {
            "pattern_recognition_rate": 0.0,
            "strategy_improvement_rate": 0.0,
            "adaptation_success_rate": 0.0
        }
        self.resource_pools = {}  # Removed - use asyncio.Semaphore if concurrency limits needed
        # Example: decision_semaphore = asyncio.Semaphore(10)
        # async with decision_semaphore: ... # Limits to 10 concurrent decisions
        self.resource_usage = {
            "memory": 0,
            "cpu": 0,
            "network": 0
        }
        self.context_cache = {}
        self.communication_history = []
        self.error_log = []
    
    @classmethod
    async def create(cls, user_id: int, conversation_id: int):
        """Async factory method for compatibility"""
        instance = cls(user_id, conversation_id)
        instance._nyx_context = NyxContext(user_id, conversation_id)
        await instance._nyx_context.initialize()
        
        # Map to legacy attributes
        instance.memory_system = instance._nyx_context.memory_system
        instance.user_model = instance._nyx_context.user_model
        instance.task_integration = instance._nyx_context.task_integration
        instance.belief_system = instance._nyx_context.belief_system
        instance.current_emotional_state = instance._nyx_context.emotional_state
        instance.performance_metrics.update(instance._nyx_context.performance_metrics)
        instance.learned_patterns = instance._nyx_context.learned_patterns
        instance.strategy_effectiveness = instance._nyx_context.strategy_effectiveness
        instance.adaptation_history = instance._nyx_context.adaptation_history
        instance.learning_metrics = instance._nyx_context.learning_metrics
        instance.error_log = instance._nyx_context.error_log
        
        # Load initial state
        await instance._load_initial_state()
        
        return instance
    
    async def _initialize_systems(self):
        """Legacy compatibility method"""
        pass
    
    async def _load_initial_state(self):
        """Load initial state for agent context"""
        # Already handled by NyxContext initialization
        pass
    
    async def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision using the decision scoring engine"""
        # Convert options to DecisionOption objects
        decision_options = [DecisionOption(
            id=str(i),
            description=str(opt),
            metadata=DecisionMetadata(data=dict_to_kvlist(opt) if isinstance(opt, dict) else DecisionMetadata().data)
        ) for i, opt in enumerate(options)]
        
        payload = ScoreDecisionOptionsInput(
            options=decision_options,
            decision_context=dict_to_kvlist(context)
        )
        
        result = await score_decision_options(
            RunContextWrapper(context=self._nyx_context),
            payload
        )
        decision_data = json.loads(result)
        
        # Update decision history
        self.decision_history.append({
            "timestamp": time.time(),
            "selected_option": decision_data["best_option"],
            "score": decision_data["confidence"],
            "context": context
        })
        
        # Update confidence
        self.decision_confidence = decision_data["confidence"]
        
        return {
            "decision": decision_data["best_option"],
            "confidence": decision_data["confidence"],
            "components": decision_data["scored_options"][0]["components"]
        }
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from experience and update patterns"""
        await self._nyx_context.learn_from_interaction(
            action=experience.get("action", "unknown"),
            outcome=experience.get("outcome", "unknown"),
            success=experience.get("success", False)
        )
        
        # Update local attributes from nyx context
        self.learned_patterns = self._nyx_context.learned_patterns
        self.adaptation_history = self._nyx_context.adaptation_history
        self.learning_metrics = self._nyx_context.learning_metrics
    
    async def process_emotional_state(self, context: Dict[str, Any], user_emotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and update emotional state - Fixed to actually update the state"""
        # Add user emotion to context if provided
        if user_emotion:
            context["user_emotion"] = user_emotion
        
        # Use the composite tool that both calculates AND updates
        result = await calculate_and_update_emotional_state(
            RunContextWrapper(context=self._nyx_context),
            CalculateEmotionalStateInput(context=dict_to_kvlist(context))
        )
        
        emotional_data = json.loads(result)
        # Update local state to match
        self.current_emotional_state = {
            "valence": emotional_data["valence"],
            "arousal": emotional_data["arousal"],
            "dominance": emotional_data["dominance"],
            "primary_emotion": emotional_data["primary_emotion"]
        }
        
        return self.current_emotional_state
    
    async def manage_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_scenario function"""
        scenario_data["user_id"] = self.user_id
        scenario_data["conversation_id"] = self.conversation_id
        return await manage_scenario(scenario_data)
    
    async def manage_relationships(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_relationships function"""
        interaction_data["user_id"] = self.user_id
        interaction_data["conversation_id"] = self.conversation_id
        return await manage_relationships(interaction_data)
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return self.current_emotional_state
    
    async def update_emotional_state(self, new_state: Dict[str, Any]):
        """Update emotional state"""
        self.current_emotional_state.update(new_state)
        self._nyx_context.emotional_state.update(new_state)
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update context - compatibility method"""
        self.context_cache.update(new_context)
        self._nyx_context.current_context.update(new_context)
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        return {
            "goals": self.current_goals,
            "active_tasks": self.active_tasks,
            "emotional_state": self.current_emotional_state,
            "beliefs": self.beliefs,
            "performance": {
                "action_success_rate": self.action_success_rate,
                "decision_confidence": self.decision_confidence,
                "goal_progress": self.goal_progress,
                "metrics": self.performance_metrics,
                "resource_usage": self.resource_usage
            },
            "learning": {
                "learned_patterns": self.learned_patterns,
                "strategy_effectiveness": self.strategy_effectiveness,
                "adaptation_history": self.adaptation_history[-5:],
                "metrics": self.learning_metrics
            },
            "errors": {
                "total": self.performance_metrics["error_rates"]["total"],
                "recovered": self.performance_metrics["error_rates"]["recovered"],
                "unrecovered": self.performance_metrics["error_rates"]["unrecovered"]
            }
        }
    
    # Additional compatibility methods
    def _calculate_emotional_weight(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate emotional weight for decisions"""
        intensity = max(abs(emotional_state.get("valence", 0)), abs(emotional_state.get("arousal", 0)))
        return min(1.0, intensity * 2.0)
    
    def _calculate_pattern_weight(self, context: Dict[str, Any]) -> float:
        """Calculate pattern weight for decisions"""
        relevant_patterns = sum(1 for p in self.learned_patterns.values()
                               if any(k in str(context) for k in str(p).split()))
        return min(1.0, relevant_patterns * 0.2)
    
    def _should_run_task(self, task_id: str) -> bool:
        """Check if task should run"""
        return self._nyx_context.should_run_task(task_id)
