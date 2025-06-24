# nyx/core/brain/base.py

import logging
import asyncio
import datetime
import random
import os
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Literal, TYPE_CHECKING
import json
from collections import defaultdict # Added
from enum import Enum, auto
from nyx.core.types.kv import KVPair

from nyx.core.brain.global_workspace.workspace_v3 import (
    NyxEngineV3, Proposal, EnhancedWorkspaceModule
)

import nyx.core.orchestrator as orchestrator

from nyx.core.brain.config import BrainConfig

from nyx.core.brain.global_workspace.adapters import build_gw_modules

from nyx.core.integration.integration_manager import create_integration_manager

from nyx.core.brain.integration_layer import EnhancedNyxBrainMixin

from agents import (
    Agent, Runner, trace, function_tool, handoff, RunContextWrapper,
    ModelSettings
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

from nyx.core.internal_thoughts import InternalThought
from nyx.core.brain.processing.unified_processor import UnifiedProcessor


from nyx.core.brain.nyx_distributed_checkpoint import DistributedCheckpointMixin
from nyx.core.brain.nyx_event_log import EventLogMixin

from nyx.core.agentic_action_generator import ActionContext, EnhancedAgenticActionGenerator

# Import new components
from nyx.core.novelty_engine import NoveltyEngine
from nyx.core.recognition_memory import RecognitionMemorySystem
from nyx.core.creative_memory_integration import CreativeMemoryIntegration

from nyx.creative.agentic_system import AgenticCreativitySystem, integrate_with_existing_system
from nyx.creative.analysis_sandbox import CodeAnalyzer

# Add:
from nyx.creative.content_system import CreativeContentSystem
from nyx.creative.capability_system import (
    CapabilityModel,
    CapabilityAssessmentSystem
)

from nyx.core.passive_observation import ObservationFilter

from pydantic import BaseModel, Field, ConfigDict

from nyx.core.mood_manager import MoodState

logger = logging.getLogger(__name__)

# --- helper ---------------------------------------------------------------

class TaskPurpose(Enum):
    ANALYZE = auto()
    WRITE = auto()
    DEPLOY = auto()
    SEARCH = auto()
    COMMUNICATE = auto()
    DATABASE = auto()
    FILE_MANIPULATION = auto()
    CODE = auto()
    VISUALIZATION = auto()
    OTHER = auto()

# Base models
class KVPair(BaseModel):
    """Key-value pair for structured data"""
    key: str
    value: Union[str, int, float, bool, None]

# Add these model definitions with the other Pydantic models at the top of the file:

class PreferenceItem(BaseModel):
    """A single preference with its score"""
    name: str
    score: float

class TraitItem(BaseModel):
    """A single trait with its value"""
    name: str
    value: float

class ChangeItem(BaseModel):
    """A single change with its magnitude"""
    attribute: str
    change: float

class PreferenceSet(BaseModel):
    """A set of preferences with scores"""
    scenario_types: List[PreferenceItem] = Field(default_factory=list)
    emotional_tones: List[PreferenceItem] = Field(default_factory=list)

class IdentityEvolution(BaseModel):
    """Identity evolution tracking data"""
    total_updates: int = 0
    recent_significant_changes: List[ChangeItem] = Field(default_factory=list)

class IdentityStateResult(BaseModel):
    """Result from getting identity state"""
    top_preferences: Optional[PreferenceSet] = None
    top_traits: List[TraitItem] = Field(default_factory=list)
    identity_reflection: Optional[str] = None
    identity_evolution: Optional[IdentityEvolution] = None
    error: Optional[str] = None

class ChallengeResponse(BaseModel):
    """Response from challenging a user claim"""
    challenge_text: str

class LieRecord(BaseModel):
    """Record of an intentional lie"""
    fact: str
    motivation: str
    timestamp: str

# Stats sub-models (needed before SystemStats)
class MemoryStats(BaseModel):
    """Memory system statistics"""
    total_memories: int = 0
    memories_by_type: Optional[Dict[str, int]] = None
    recent_retrievals: int = 0
    storage_used_mb: Optional[float] = None

class MetaStats(BaseModel):
    """Meta-cognitive statistics"""
    evaluation_cycles: int = 0
    adaptations_made: int = 0
    performance_score: Optional[float] = None

class KnowledgeStats(BaseModel):
    """Knowledge system statistics"""
    total_facts: int = 0
    domains: Optional[List[str]] = None
    recent_queries: int = 0

class EmotionalStateStats(BaseModel):
    """Emotional state statistics"""
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    dominant_value: Optional[float] = None
    valence: float = 0.0
    arousal: float = 0.0

class HormoneStats(BaseModel):
    """Hormone system statistics"""
    hormone_levels: Optional[Dict[str, float]] = None
    cycle_phases: Optional[Dict[str, str]] = None
    dominant_hormone: Optional[Dict[str, Union[str, float]]] = None

class ProceduralStats(BaseModel):
    """Procedural memory statistics"""
    total_procedures: int = 0
    available_procedures: List[str] = Field(default_factory=list)
    procedure_domains: List[str] = Field(default_factory=list)
    execution_count: int = 0

class IdentityStats(BaseModel):
    """Identity statistics"""
    trait_count: int = 0
    preference_count: int = 0
    dominant_traits: List[Tuple[str, float]] = Field(default_factory=list)

class NeedsStats(BaseModel):
    """Needs system statistics"""
    current_levels: Optional[Dict[str, float]] = None
    drive_strengths: Optional[Dict[str, float]] = None
    total_drive: float = 0.0

class GoalStats(BaseModel):
    """Goal system statistics"""
    total_goals: int = 0
    active_goals_count: int = 0
    pending_goals_count: int = 0
    completed_goals: int = 0
    failed_goals: int = 0
    active_goal_ids: List[str] = Field(default_factory=list)
    highest_priority_pending: Optional[str] = None

class ThinkingStats(BaseModel):
    """Thinking system statistics"""
    thoughts_generated: int = 0
    thinking_depth: int = 0
    epistemic_confidence: float = 1.0

class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    memory_operations: int = 0
    emotion_updates: int = 0
    reflections_generated: int = 0
    experiences_shared: int = 0
    cross_user_experiences_shared: int = 0
    avg_response_time: float = 0.0
    goals_completed: int = 0
    goals_failed: int = 0
    steps_executed: int = 0

class ProcessingStats(BaseModel):
    """Processing system statistics"""
    processor_type: str = "unified"
    initialized: bool = False

# Main stats model (after sub-models)
class SystemStats(BaseModel):
    """Comprehensive system statistics"""
    memory_stats: Optional[MemoryStats] = None
    meta_stats: Optional[MetaStats] = None
    knowledge_stats: Optional[KnowledgeStats] = None
    emotional_state: Optional[EmotionalStateStats] = None
    hormone_stats: Optional[HormoneStats] = None
    procedural_stats: Optional[ProceduralStats] = None
    identity_stats: Optional[IdentityStats] = None
    needs_stats: Optional[NeedsStats] = None
    goal_stats: Optional[GoalStats] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    thinking_stats: Optional[ThinkingStats] = None
    processing_stats: Optional[ProcessingStats] = None

# Cognitive cycle models
class NeedsUpdate(BaseModel):
    """Needs system update result"""
    drive_strengths: Optional[List[KVPair]] = None
    error: Optional[str] = None

class StepDetail(BaseModel):
    """Step execution detail"""
    name: str
    status: str
    meta: Optional[List[KVPair]] = None

class GoalExecutionResult(BaseModel):
    """Goal execution result"""
    goal_id: Optional[str] = None
    executed_step: Optional[StepDetail] = None
    status: Optional[str] = None
    error: Optional[str] = None

class MetaCoreResult(BaseModel):
    """Meta core cycle result"""
    evaluation_completed: bool = False
    adjustments_made: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class EmotionalContext(BaseModel):
    """Emotional context"""
    primary: Optional[str] = None
    intensity: Optional[float] = None
    valence: Optional[float] = None  # -1 to 1

class EnvironmentFactor(BaseModel):
    """Environmental factor"""
    name: str
    value: Union[str, int, float, bool]

class CognitiveCycleContext(BaseModel):
    """Context data for cognitive cycle"""
    user_input: Optional[str] = None
    emotional_context: Optional[EmotionalContext] = None
    environmental_factors: Optional[List[EnvironmentFactor]] = None

class CognitiveCycleResult(BaseModel):
    """Result from running a cognitive cycle"""
    cycle_number: int
    timestamp: str
    needs_update: Optional[NeedsUpdate] = None
    goal_execution: Optional[GoalExecutionResult] = None
    meta_core_cycle: Optional[MetaCoreResult] = None
    error: Optional[str] = None

# Action result models
class DominanceActionResult(BaseModel):
    """Result from processing a dominance action"""
    success: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    action_taken: Optional[str] = None
    intensity_applied: Optional[float] = None

class ServiceTaskAssignmentResult(BaseModel):
    """Result from assigning a service task"""
    success: bool
    message: str
    task_id: Optional[str] = None
    task_type: Optional[str] = None
    duration_minutes: Optional[float] = None
    user_id: Optional[str] = None

class DesireExpression(BaseModel):
    """Result of expressing desire"""
    success: bool
    expression: Optional[str] = None
    target: Optional[str] = None
    reason: Optional[str] = None

class SensationResult(BaseModel):
    """Result from digital somatosensory processing"""
    type: str
    intensity: float
    region: str
    processing_notes: Optional[str] = None

class PhysicalTouchResult(BaseModel):
    """Result of simulating physical touch"""
    success: bool
    sensation_result: Optional[SensationResult] = None
    internal_expression: Optional[str] = None
    reason: Optional[str] = None

class GratificationContext(BaseModel):
    """Context for gratification seeking"""
    interaction_type: str = "intimate"
    intensity_level: float = 0.5
    safety_confirmed: bool = False

class GratificationResult(BaseModel):
    """Result of gratification-related actions"""
    success: bool
    status: str
    reason: Optional[str] = None

# Dominance-related models
class UserInputAnalysis(BaseModel):
    """Analysis of user input for dominance"""
    submissive_score: float = 0.0
    resistance_score: float = 0.0
    emotional_tone: Optional[str] = None
    key_phrases: List[str] = Field(default_factory=list)

class DominanceAssessment(BaseModel):
    """Assessment of dominance readiness"""
    readiness_score: float
    assessment: str
    reason: Optional[str] = None

class CommandResult(BaseModel):
    """Result of issuing a command"""
    success: bool
    command_issued: Optional[str] = None
    intensity: Optional[float] = None
    reason: Optional[str] = None

class ComplianceEvaluation(BaseModel):
    """Evaluation of user compliance"""
    compliance_level: float
    is_compliant: bool

class IntensityIncrease(BaseModel):
    """Result of increasing control intensity"""
    success: bool
    status: Optional[str] = None
    next_intensity_target: Optional[float] = None
    reason: Optional[str] = None

class DominanceGratificationResult(BaseModel):
    """Result of dominance gratification"""
    success: bool
    status: str

class SatisfactionExpression(BaseModel):
    """Expression of satisfaction"""
    success: bool
    expression: str

class AttractionExpression(BaseModel):
    """Result of expressing attraction"""
    success: bool
    expression: Optional[str] = None
    target: Optional[str] = None
    reason: Optional[str] = None

class IntensityRange(BaseModel):
    """Intensity range specification"""
    min: int
    max: int

class FemdomActivityIdea(BaseModel):
    """A single femdom activity idea"""
    id: str
    description: str
    intensity: int
    category: str
    required_trust: float = 0.5
    required_items: List[str] = Field(default_factory=list)

class FemdomIdeasResult(BaseModel):
    """Result of generating femdom ideas"""
    success: bool
    ideas: List[FemdomActivityIdea] = Field(default_factory=list)  # Fixed to use proper model
    error: Optional[str] = None

class ProtocolAssignment(BaseModel):
    """Result of protocol assignment"""
    success: bool
    protocol_id: Optional[str] = None
    user_id: Optional[str] = None
    reason: Optional[str] = None

class RoleplayResult(BaseModel):
    """Result of roleplay mode changes"""
    success: bool
    character_name: Optional[str] = None
    context: Optional[str] = None
    reason: Optional[str] = None

class IntimateInteractionResult(BaseModel):
    """Result of intimate interaction initiation"""
    success: bool
    proposal: Optional[str] = None
    target: Optional[str] = None
    level: Optional[str] = None
    reason: Optional[str] = None

class DominanceTargetEvaluation(BaseModel):
    """Evaluation of dominance target potential"""
    user_id: str
    interest_score: float
    reason: str

class OrgasmControlResult(BaseModel):
    """Result of orgasm permission processing"""
    success: bool
    message: str
    permission_granted: Optional[bool] = None

class PersonaRecommendation(BaseModel):
    """Dominance persona recommendation"""
    success: bool
    persona_id: Optional[str] = None
    persona_name: Optional[str] = None
    confidence: float = 0.5
    message: Optional[str] = None

class PersonaActivation(BaseModel):
    """Result of persona activation"""
    success: bool
    persona_id: Optional[str] = None
    active_until: Optional[str] = None
    message: Optional[str] = None

class SadisticResponse(BaseModel):
    """Generated sadistic response"""
    success: bool
    response: Optional[str] = None
    category: Optional[str] = None
    intensity: Optional[float] = None
    message: Optional[str] = None

class LimitTestResult(BaseModel):
    """Result of soft limit testing"""
    success: bool
    status: Optional[str] = None
    planned_action: Optional[str] = None
    reason: Optional[str] = None

class MaintenanceResult(BaseModel):
    """Result from running maintenance tasks"""
    maintenance_time: str
    hormone_maintenance: Optional[Dict[str, Any]] = None
    dss_maintenance_update: Optional[Dict[str, Any]] = None
    memory_maintenance: Optional[Dict[str, Any]] = None
    meta_maintenance: Optional[Dict[str, Any]] = None
    knowledge_maintenance: Optional[Dict[str, Any]] = None
    experience_consolidation: Optional[Dict[str, Any]] = None
    user_clustering: Optional[Dict[str, Any]] = None
    procedural_maintenance: Optional[Dict[str, Any]] = None
    hierarchical_memory_maintenance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SystemSettingsUpdate(BaseModel):
    """System settings that were updated"""
    cross_user_enabled: bool
    cross_user_sharing_threshold: float

class UserProfile(BaseModel):
    """User experience sharing profile"""
    experience_sharing_preference: float = 0.5

class ExperienceSharingAdaptation(BaseModel):
    """Result from adapting experience sharing"""
    profile: Optional[UserProfile] = None
    system_settings_updated: Optional[SystemSettingsUpdate] = None
    error: Optional[str] = None

# Helper function
def _dict_to_kv(d: dict[str, Any] | None) -> list[KVPair] | None:
    """
    Convert an ordinary dict into a list-of-KVPair.
    Returns None if the input is falsy.
    """
    if not d:
        return None
    return [KVPair(key=k, value=v) for k, v in d.items()]


class NyxBrain(DistributedCheckpointMixin, EventLogMixin, EnhancedNyxBrainMixin):
    """
    Central integration point for all Nyx systems.
    Uses composition to delegate to specialized components while managing their coordination.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize NyxBrain with user and conversation IDs.
        Note: Actual system initialization happens in the async initialize() method.
        
        Args:
            user_id: User ID for this brain instance
            conversation_id: Conversation ID for this brain instance
        """
        # Call parent class __init__ methods (without arguments for cooperative inheritance)
        super().__init__()
        
        # Store the basic identifiers
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Set initialization flags
        self.initialized = False
        self._initializing_flag = False
        
        # Set attributes expected by mixins
        import os
        import uuid
        self.nyx_id = os.getenv("NYX_ID", "nyx_v1")  # Expected by EventLogMixin
        self.instance_id = os.getenv("NYX_INSTANCE_ID", str(uuid.uuid4()))  # Expected by EventLogMixin
        self.NYX_ID = self.nyx_id  # Alternative attribute name used by mixins
        self.INSTANCE_ID = self.instance_id  # Alternative attribute name used by mixins
        
        # Initialize workspace_engine to None since it might be accessed before initialize()
        self.workspace_engine = None
        
        # Load configuration and set config attributes
        from nyx.core.brain.config import BrainConfig
        self.config = BrainConfig.default()
        
        # Set A2A integration flag from config
        self.use_a2a_integration = self.config.use_a2a_integration
        
        # Set performance & behavior settings from config
        self.cross_user_enabled = self.config.cross_user_enabled
        self.cross_user_sharing_threshold = self.config.cross_user_sharing_threshold
        self.memory_to_emotion_influence = self.config.memory_to_emotion_influence
        self.emotion_to_memory_influence = self.config.emotion_to_memory_influence
        self.experience_to_identity_influence = self.config.experience_to_identity_influence
        self.consolidation_interval = self.config.consolidation_interval
        self.identity_reflection_interval = self.config.identity_reflection_interval
        self.need_drive_threshold = self.config.need_drive_threshold
        
        # Set context configuration
        self.context_config = self.config.context_config
        
        # Set thinking configuration
        self.thinking_config = self.config.thinking_config
        
        # Initialize empty registry and module sets
        self.internal_module_registry = {}
        self.default_active_modules = set()
        
        # Initialize performance metrics
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "cross_user_experiences_shared": 0,
            "response_times": [],
            "goals_completed": 0,
            "goals_failed": 0,
            "steps_executed": 0,
            "experience_consolidations": 0
        }
        
        # Initialize other tracking attributes
        self.cognitive_cycles_executed = 0
        self.action_history = []
        self.motivations = {}
        self.procedural_activity_metrics = {}
        self.current_temporal_context = None
        
        # Initialize error registry
        self.error_registry = {
            "error_counts": {},
            "error_recovery_strategies": {},
            "error_recovery_stats": {},
            "handled_errors": [],
            "unhandled_errors": []
        }
        
        # Initialize misc attributes that might be accessed
        self.trace_group_id = f"{self.user_id}_{self.conversation_id}"
        self.event_bus = None
        self.integrated_tracer = None
        self.strategy_controller = None
        self.noise_filter = None
        self.context_distribution = None
        self.agent_capabilities_initialized = False
        
        # The actual initialization will happen in the async initialize() method
        logger.debug(f"NyxBrain instance created for user {user_id}, conversation {conversation_id} - awaiting initialization")
    
    async def initialize(self):
        """
        Initialize all subsystems in the correct dependency order.
        Handles circular dependencies by setting references after initialization.
        """
        if self.initialized:
            logger.info(f"NyxBrain {self.user_id}-{self.conversation_id} ALREADY INITIALIZED. Skipping.")
            return
        if self._initializing_flag:
            logger.warning(f"NyxBrain {self.user_id}-{self.conversation_id} ALREADY INITIALIZING. Skipping re-entrant call.")
            return
    
        self._initializing_flag = True
        logger.critical(f"NyxBrain.initialize() ENTERED for {self.user_id}-{self.conversation_id}. Current self.initialized: {self.initialized}")
        
        # Track initialization progress for better error reporting
        self._init_progress = []
        
        try:
            # Import all required modules at once
            await self._import_modules()
            
            # Initialize in dependency order
            await self._init_tier_0_infrastructure()
            await self._init_tier_1_foundation()
            await self._init_tier_2_core_emotional_memory()
            await self._init_tier_3_cognitive_systems()
            await self._init_tier_4_integration_systems()
            await self._init_tier_5_complex_systems()
            await self._init_tier_6_action_generation()
            await self._init_tier_7_specialized_systems()
            await self._init_tier_8_final_setup()
            
            # Initialize the A2A context distribution system if enabled
            if self.use_a2a_integration:
                await self.initialize_context_system()
                logger.critical("NyxBrain initialization complete with A2A context distribution")
            else:
                logger.critical("NyxBrain initialization complete (standard mode)")
            
            self.initialized = True
            logger.critical(f"NyxBrain.initialize() COMPLETED SUCCESSFULLY for {self.user_id}-{self.conversation_id}. self.initialized set to True.")
            
        except Exception as e:
            logger.critical(f"NyxBrain.initialize() FAILED for {self.user_id}-{self.conversation_id}: {e}", exc_info=True)
            logger.critical(f"Initialization progress: {self._init_progress}")
            self.initialized = False
            # Attempt cleanup of partially initialized state
            await self._cleanup_partial_initialization()
            raise
        finally:
            self._initializing_flag = False

    
    async def _import_modules(self):
        """Import all required modules in one place"""
        self._init_progress.append("import_modules_start")
        
        # Store imports as instance variables for use in initialization methods
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        from nyx.core.emotions.emotional_core import EmotionalCore
        from nyx.core.memory_core import MemoryCoreAgents, BrainMemoryCore
        from nyx.core.reflection_engine import ReflectionEngine
        from nyx.core.experience_interface import ExperienceInterface
        from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
        from nyx.core.internal_feedback_system import InternalFeedbackSystem
        from nyx.core.meta_core import MetaCore
        from nyx.core.knowledge_core import KnowledgeCoreAgents
        from nyx.core.memory_orchestrator import MemoryOrchestrator
        from nyx.core.identity_evolution import IdentityEvolutionSystem
        from nyx.core.experience_consolidation import ExperienceConsolidationSystem
        from nyx.core.cross_user_experience import CrossUserExperienceManager
        from nyx.core.emotions.hormone_system import HormoneSystem
        from nyx.core.attentional_controller import AttentionalController
        from nyx.core.multimodal_integrator import MultimodalIntegrator
        from nyx.core.reward_system import RewardSignalProcessor
        from nyx.core.temporal_perception import TemporalPerceptionSystem
        from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
        from nyx.core.digital_somatosensory_system import DigitalSomatosensorySystem
        from nyx.core.needs_system import NeedsSystem
        from nyx.core.conditioning_system import ConditioningSystem
        from nyx.core.goal_system import GoalManager
        from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
        from nyx.core.reasoning_core import ReasoningCore
        from nyx.core.brain.adaptation.self_config import SelfConfigManager
        from nyx.core.context_awareness import ContextAwarenessSystem
        from nyx.core.interaction_mode_manager import InteractionModeManager
        from nyx.core.mood_manager import MoodManager
        
        # Store module classes for later use
        self._modules = {
            'ModuleOptimizer': ModuleOptimizer,
            'SystemHealthChecker': SystemHealthChecker,
            'EmotionalCore': EmotionalCore,
            'MemoryCoreAgents': MemoryCoreAgents,
            'BrainMemoryCore': BrainMemoryCore,
            'ReflectionEngine': ReflectionEngine,
            'ExperienceInterface': ExperienceInterface,
            'DynamicAdaptationSystem': DynamicAdaptationSystem,
            'InternalFeedbackSystem': InternalFeedbackSystem,
            'MetaCore': MetaCore,
            'KnowledgeCoreAgents': KnowledgeCoreAgents,
            'MemoryOrchestrator': MemoryOrchestrator,
            'IdentityEvolutionSystem': IdentityEvolutionSystem,
            'ExperienceConsolidationSystem': ExperienceConsolidationSystem,
            'CrossUserExperienceManager': CrossUserExperienceManager,
            'HormoneSystem': HormoneSystem,
            'AttentionalController': AttentionalController,
            'MultimodalIntegrator': MultimodalIntegrator,
            'RewardSignalProcessor': RewardSignalProcessor,
            'TemporalPerceptionSystem': TemporalPerceptionSystem,
            'AgentEnhancedMemoryManager': AgentEnhancedMemoryManager,
            'DigitalSomatosensorySystem': DigitalSomatosensorySystem,
            'NeedsSystem': NeedsSystem,
            'ConditioningSystem': ConditioningSystem,
            'GoalManager': GoalManager,
            'integrated_reasoning_agent': integrated_reasoning_agent,
            'reasoning_triage_agent': reasoning_triage_agent,
            'ReasoningCore': ReasoningCore,
            'UnifiedProcessor': UnifiedProcessor,
            'SelfConfigManager': SelfConfigManager,
            'ContextAwarenessSystem': ContextAwarenessSystem,
            'InteractionModeManager': InteractionModeManager,
            'MoodManager': MoodManager
        }
        
        # Import A2A wrappers if needed
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_emotional_core import ContextAwareEmotionalCore
            from nyx.core.a2a.context_aware_hormone_system import ContextAwareHormoneSystem
            from nyx.core.a2a.context_aware_memory_core import ContextAwareMemoryCore
            from nyx.core.a2a.context_aware_memory_orchestrator import ContextAwareMemoryOrchestrator
            from nyx.core.a2a.context_aware_identity_evolution import ContextAwareIdentityEvolution
            from nyx.core.a2a.context_aware_knowledge_core import ContextAwareKnowledgeCore
            from nyx.core.a2a.context_aware_attentional_controller import ContextAwareAttentionalController
            from nyx.core.a2a.context_aware_temporal_perception import ContextAwareTemporalPerception
            from nyx.core.a2a.context_aware_reasoning_core import ContextAwareReasoningCore
            from nyx.core.a2a.context_aware_reasoning_agents import ContextAwareReasoningAgents
            from nyx.core.a2a.context_aware_internal_feedback_system import ContextAwareInternalFeedbackSystem
            from nyx.core.a2a.context_aware_dynamic_adaptation import ContextAwareDynamicAdaptation
            from nyx.core.a2a.context_aware_context_system import ContextAwareContextSystem
            from nyx.core.a2a.context_aware_experience_interface import ContextAwareExperienceInterface
            from nyx.core.a2a.context_aware_experience_consolidation import ContextAwareExperienceConsolidation
            from nyx.core.a2a.context_aware_cross_user_experience import ContextAwareCrossUserExperience
            from nyx.core.a2a.context_aware_goal_manager import ContextAwareGoalManager
            from nyx.core.a2a.context_aware_needs import ContextAwareNeedsSystem
            from nyx.core.a2a.context_aware_mood_manager import ContextAwareMoodManager
            from nyx.core.a2a.context_aware_reward_system import ContextAwareRewardSystem
            from nyx.core.a2a.context_aware_somatosensory_system import ContextAwareDigitalSomatosensorySystem
            from nyx.core.a2a.context_aware_conditioning import ContextAwareConditioningSystem
            from nyx.core.a2a.context_aware_interaction_mode_manager import ContextAwareInteractionModeManager
            from nyx.core.a2a.context_aware_input_processor import ContextAwareInputProcessor
            from nyx.core.a2a.context_aware_relationship_manager import ContextAwareRelationshipManager
            from nyx.core.a2a.context_aware_multimodal_integrator import ContextAwareMultimodalIntegrator
            from nyx.core.a2a.context_aware_imagination_simulator import ContextAwareImaginationSimulator
            from nyx.core.a2a.context_aware_theory_of_mind import ContextAwareTheoryOfMind
            from nyx.core.a2a.context_aware_meta_core import ContextAwareMetaCore
            from nyx.core.a2a.context_aware_action_generator import ContextAwareAgenticActionGenerator
            from nyx.core.a2a.context_aware_reflection_engine import ContextAwareReflectionEngine
            from nyx.core.a2a.context_aware_passive_observation import ContextAwarePassiveObservation
            from nyx.core.a2a.context_aware_proactive_communication import ContextAwareProactiveCommunication
            from nyx.core.a2a.context_aware_internal_thoughts import ContextAwareInternalThoughts
            from nyx.core.a2a.context_aware_prediction_engine import ContextAwarePredictionEngine
            from nyx.core.a2a.context_aware_mode_integration import ContextAwareModeIntegration
            from nyx.core.a2a.context_aware_issue_tracking_system import ContextAwareIssueTrackingSystem
            from nyx.core.a2a.context_aware_reflexive_system import ContextAwareReflexiveSystem
            from nyx.core.a2a.context_aware_protocol_enforcement import ContextAwareProtocolEnforcement
            from nyx.core.a2a.context_aware_body_service import ContextAwareBodyService
            from nyx.core.a2a.context_aware_psychological_dominance import ContextAwarePsychologicalDominance
            from nyx.core.a2a.context_aware_orgasm_control import ContextAwareOrgasmControl
            from nyx.core.a2a.context_aware_persona_manager import ContextAwarePersonaManager
            from nyx.core.a2a.context_aware_sadistic_responses import ContextAwareSadisticResponses
            from nyx.core.a2a.context_aware_submission_progression import ContextAwareSubmissionProgression
                        
            
            # Add A2A modules to the dictionary
            self._modules.update({
                'ContextAwareEmotionalCore': ContextAwareEmotionalCore,
                'ContextAwareHormoneSystem': ContextAwareHormoneSystem,
                'ContextAwareMemoryCore': ContextAwareMemoryCore,
                'ContextAwareMemoryOrchestrator': ContextAwareMemoryOrchestrator,
                'ContextAwareIdentityEvolution': ContextAwareIdentityEvolution,
                'ContextAwareKnowledgeCore': ContextAwareKnowledgeCore,
                'ContextAwareAttentionalController': ContextAwareAttentionalController,
                'ContextAwareTemporalPerception': ContextAwareTemporalPerception,
                'ContextAwareReasoningCore': ContextAwareReasoningCore,
                'ContextAwareReasoningAgents': ContextAwareReasoningAgents,
                'ContextAwareInternalFeedbackSystem': ContextAwareInternalFeedbackSystem,
                'ContextAwareDynamicAdaptation': ContextAwareDynamicAdaptation,
                'ContextAwareContextSystem': ContextAwareContextSystem,
                'ContextAwareExperienceInterface': ContextAwareExperienceInterface,
                'ContextAwareExperienceConsolidation': ContextAwareExperienceConsolidation,
                'ContextAwareCrossUserExperience': ContextAwareCrossUserExperience,
                'ContextAwareGoalManager': ContextAwareGoalManager,
                'ContextAwareNeedsSystem': ContextAwareNeedsSystem,
                'ContextAwareMoodManager': ContextAwareMoodManager,
                'ContextAwareRewardSystem': ContextAwareRewardSystem,
                'ContextAwareDigitalSomatosensorySystem': ContextAwareDigitalSomatosensorySystem,
                'ContextAwareConditioningSystem': ContextAwareConditioningSystem,
                'ContextAwareInteractionModeManager': ContextAwareInteractionModeManager,
                'ContextAwareInputProcessor': ContextAwareInputProcessor,
                'ContextAwareRelationshipManager': ContextAwareRelationshipManager,
                'ContextAwareMultimodalIntegrator': ContextAwareMultimodalIntegrator,
                'ContextAwareImaginationSimulator': ContextAwareImaginationSimulator,
                'ContextAwareTheoryOfMind': ContextAwareTheoryOfMind,
                'ContextAwareMetaCore': ContextAwareMetaCore,
                'ContextAwareAgenticActionGenerator': ContextAwareAgenticActionGenerator,
                'ContextAwareReflectionEngine': ContextAwareReflectionEngine,
                'ContextAwarePassiveObservation': ContextAwarePassiveObservation,
                'ContextAwareProactiveCommunication': ContextAwareProactiveCommunication,
                'ContextAwareInternalThoughts': ContextAwareInternalThoughts,
                'ContextAwarePredictionEngine': ContextAwarePredictionEngine,
                'ContextAwareModeIntegration': ContextAwareModeIntegration,
                'ContextAwareIssueTrackingSystem': ContextAwareIssueTrackingSystem,
                'ContextAwareReflexiveSystem': ContextAwareReflexiveSystem,
                'ContextAwareConditioningSystem': ContextAwareConditioningSystem,
                'ContextAwareAgenticActionGenerator': ContextAwareAgenticActionGenerator,
                'ContextAwareProtocolEnforcement': ContextAwareProtocolEnforcement,
                'ContextAwareBodyService': ContextAwareBodyService,
                'ContextAwarePsychologicalDominance': ContextAwarePsychologicalDominance,
                'ContextAwareOrgasmControl': ContextAwareOrgasmControl,
                'ContextAwarePersonaManager': ContextAwarePersonaManager,
                'ContextAwareSadisticResponses': ContextAwareSadisticResponses,
                'ContextAwareSubmissionProgression': ContextAwareSubmissionProgression,
            })
        
        self._init_progress.append("import_modules_complete")
    
    async def _init_tier_0_infrastructure(self):
        """Initialize configuration and infrastructure components"""
        self._init_progress.append("tier_0_start")
        logger.debug(f"NyxBrain Init Tier 0: Infrastructure for {self.user_id}-{self.conversation_id}")
        
        # Configuration is already loaded in __init__
        # No need to reload it here
        
        # Dev log storage
        from dev_log.storage import get_dev_log_storage
        self.dev_log_storage = get_dev_log_storage()
        await self.dev_log_storage.initialize()
        
        # System context
        from nyx.core.integration.system_context import get_system_context
        self.system_context = get_system_context()
        
        # Event bus
        if not self.event_bus:
            try:
                from nyx.core.events.event_bus import EventBus
                self.event_bus = EventBus()
                await self.event_bus.initialize()
                logger.debug("Event bus initialized")
            except ImportError:
                logger.warning("EventBus module not found - event system will be unavailable")
                self.event_bus = None
        
        # Global Workspace Architecture
        if self.workspace_engine is None:
            logger.debug("Initializing Global Workspace Architecture")
            from nyx.core.brain.global_workspace.adapters import build_gw_modules
            gw_modules = build_gw_modules(self)
            self.workspace_engine = NyxEngineV3(
                gw_modules,
                hz=10.0,  # 10Hz cognitive cycle
                persist_bias=Path(f"gw_bias_{self.user_id}_{self.conversation_id}.json"),
                enable_unconscious=True
            )
            await self.workspace_engine.start()
            logger.info(f"Global Workspace Engine started with {len(gw_modules)} modules")
        
        # Support systems
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent
        
        self.module_optimizer = ModuleOptimizer(self)
        self.system_health_checker = SystemHealthChecker(self)
        self.checkpoint_planner = CheckpointingPlannerAgent()
        
        self._init_progress.append("tier_0_complete")
    
    async def _init_tier_1_foundation(self):
        """Initialize foundation components with no interdependencies"""
        self._init_progress.append("tier_1_start")
        logger.debug(f"NyxBrain Init Tier 1: Foundation for {self.user_id}-{self.conversation_id}")
        
        # Hormone system (no dependencies)
        if self.config.hormone_system.enabled:
            from nyx.core.emotions.hormone_system import HormoneSystem
            original_hormone_system = HormoneSystem()
            self.hormone_system = await self._wrap_with_a2a(
                original_hormone_system,
                "ContextAwareHormoneSystem",
                "hormone_system"
            )
        
        # Temporal perception (minimal dependencies)
        from nyx.core.temporal_perception import TemporalPerceptionSystem
        original_temporal = TemporalPerceptionSystem(self.user_id, self.conversation_id)
        await original_temporal.initialize(brain_context=self, first_interaction_timestamp=None)
        self.temporal_perception = await self._wrap_with_a2a(
            original_temporal,
            "ContextAwareTemporalPerception",
            "temporal_perception"
        )
        
        # Knowledge core (no dependencies)
        from nyx.core.knowledge_core import KnowledgeCoreAgents
        original_knowledge = KnowledgeCoreAgents()
        await original_knowledge.initialize()
        self.knowledge_core = await self._wrap_with_a2a(
            original_knowledge,
            "ContextAwareKnowledgeCore",
            "knowledge_core"
        )
        
        # Procedural memory manager (no dependencies)
        from nyx.core.procedural_memory.manager import ProceduralMemoryManager
        from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
        self.procedural_memory_manager = ProceduralMemoryManager()
        self.agent_enhanced_memory = AgentEnhancedMemoryManager(memory_manager=self.procedural_memory_manager)
        
        # Internal feedback (no dependencies)
        from nyx.core.internal_feedback_system import InternalFeedbackSystem
        original_feedback = InternalFeedbackSystem()
        self.internal_feedback = await self._wrap_with_a2a(
            original_feedback,
            "ContextAwareInternalFeedbackSystem",
            "internal_feedback"
        )
        
        # Dynamic adaptation (no dependencies)
        from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
        original_adaptation = DynamicAdaptationSystem()
        self.dynamic_adaptation = await self._wrap_with_a2a(
            original_adaptation,
            "ContextAwareDynamicAdaptation",
            "dynamic_adaptation"
        )
        
        self._init_progress.append("tier_1_complete")
    
    async def _init_tier_2_core_emotional_memory(self):
        """Initialize core emotional and memory systems"""
        self._init_progress.append("tier_2_start")
        logger.debug(f"NyxBrain Init Tier 2: Core Emotional/Memory for {self.user_id}-{self.conversation_id}")
        
        # Emotional core (needs hormone system)
        if self.config.emotional_core.enabled:
            from nyx.core.emotions import EmotionalCore
            original_emotional = EmotionalCore()
            
            # Set hormone system reference if both are enabled
            if self.config.hormone_system.enabled and self.hormone_system:
                # Get the original system if wrapped
                original_hormone = getattr(self.hormone_system, 'original_system', self.hormone_system)
                original_emotional.set_hormone_system(original_hormone)
            
            self.emotional_core = await self._wrap_with_a2a(
                original_emotional,
                "ContextAwareEmotionalCore",
                "emotional_core"
            )
            
            # Set wrapper references if both are wrapped
            if self.use_a2a_integration and self.hormone_system and hasattr(self.emotional_core, '_hormone_system_wrapper'):
                self.emotional_core._hormone_system_wrapper = self.hormone_system
                if hasattr(self.hormone_system, '_emotional_core_wrapper'):
                    self.hormone_system._emotional_core_wrapper = self.emotional_core
        
        # Memory core
        from nyx.core.memory_core import MemoryCoreAgents
        original_memory = MemoryCoreAgents(self.user_id, self.conversation_id)
        await original_memory.initialize()
        self.memory_core = await self._wrap_with_a2a(
            original_memory,
            "ContextAwareMemoryCore",
            "memory_core"
        )
        
        # Memory orchestrator
        from nyx.core.memory_orchestrator import MemoryOrchestrator
        original_orchestrator = MemoryOrchestrator(self.user_id, self.conversation_id)
        await original_orchestrator.initialize()
        self.memory_orchestrator = await self._wrap_with_a2a(
            original_orchestrator,
            "ContextAwareMemoryOrchestrator",
            "memory_orchestrator"
        )
        
        # Identity evolution (needs hormone system)
        from nyx.core.identity_evolution import IdentityEvolutionSystem
        hormone_ref = getattr(self, 'hormone_system', None) if self.config.hormone_system.enabled else None
        original_identity = IdentityEvolutionSystem(hormone_system=hormone_ref)
        await original_identity.initialize()
        self.identity_evolution = await self._wrap_with_a2a(
            original_identity,
            "ContextAwareIdentityEvolution",
            "identity_evolution"
        )
        
        self._init_progress.append("tier_2_complete")
    
    async def _init_tier_3_cognitive_systems(self):
        """Initialize cognitive processing systems"""
        self._init_progress.append("tier_3_start")
        logger.debug(f"NyxBrain Init Tier 3: Cognitive Systems for {self.user_id}-{self.conversation_id}")
        
        # Reasoning core (needs knowledge core)
        from nyx.core.reasoning_core import ReasoningCore
        from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
        
        original_reasoning = ReasoningCore(knowledge_core=self.knowledge_core)
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_reasoning_core import ContextAwareReasoningCore
            from nyx.core.a2a.context_aware_reasoning_agents import ContextAwareReasoningAgents
            
            self.context_aware_reasoning_module_instance = ContextAwareReasoningCore(original_reasoning)
            self.reasoning_agents = ContextAwareReasoningAgents(
                reasoning_triage_agent,
                integrated_reasoning_agent,
                self.context_aware_reasoning_module_instance
            )
            self.reasoning_core = self.reasoning_agents
            logger.debug("Enhanced ReasoningAgents with A2A context distribution")
        else:
            self.reasoning_core = integrated_reasoning_agent
            self.reasoning_triage_agent = reasoning_triage_agent
        
        # Attentional controller (needs emotional core)
        if self.config.attentional_controller.enabled:
            from nyx.core.attentional_controller import AttentionalController
            original_attention = AttentionalController(emotional_core=self.emotional_core)
            self.attentional_controller = await self._wrap_with_a2a(
                original_attention,
                "ContextAwareAttentionalController",
                "attentional_controller"
            )
        
        # Context awareness system (needs emotional core)
        from nyx.core.context_awareness import ContextAwarenessSystem
        original_context = ContextAwarenessSystem(emotional_core=self.emotional_core)
        self.context_system = await self._wrap_with_a2a(
            original_context,
            "ContextAwareContextSystem",
            "context_system"
        )
        
        self._init_progress.append("tier_3_complete")
    
    async def _init_tier_4_integration_systems(self):
        """Initialize systems that integrate multiple components"""
        self._init_progress.append("tier_4_start")
        logger.debug(f"NyxBrain Init Tier 4: Integration Systems for {self.user_id}-{self.conversation_id}")
        
        # Experience interface (needs memory_core, emotional_core)
        from nyx.core.experience_interface import ExperienceInterface
        original_experience = ExperienceInterface(self.memory_core, self.emotional_core)
        self.experience_interface = await self._wrap_with_a2a(
            original_experience,
            "ContextAwareExperienceInterface",
            "experience_interface"
        )
        
        # Experience consolidation (needs memory_core, experience_interface)
        from nyx.core.experience_consolidation import ExperienceConsolidationSystem
        original_consolidation = ExperienceConsolidationSystem(
            memory_core=self.memory_core,
            experience_interface=self.experience_interface
        )
        self.experience_consolidation = await self._wrap_with_a2a(
            original_consolidation,
            "ContextAwareExperienceConsolidation",
            "experience_consolidation"
        )
        
        # Cross-user manager (needs memory_core, experience_interface)
        from nyx.core.cross_user_experience import CrossUserExperienceManager
        original_cross_user = CrossUserExperienceManager(
            memory_core=self.memory_core,
            experience_interface=self.experience_interface
        )
        self.cross_user_manager = await self._wrap_with_a2a(
            original_cross_user,
            "ContextAwareCrossUserExperience",
            "cross_user_manager"
        )
        
        # Relationship manager (needs memory_orchestrator, emotional_core)
        try:
            from nyx.core.relationship_manager import RelationshipManager
        except ImportError:
            try:
                from nyx.core.social.relationship_manager import RelationshipManager
            except ImportError:
                logger.warning("RelationshipManager module not found")
                RelationshipManager = None
        
        if RelationshipManager:
            original_relationship = RelationshipManager(
                memory_orchestrator=self.memory_orchestrator,
                emotional_core=self.emotional_core
            )
            self.relationship_manager = await self._wrap_with_a2a(
                original_relationship,
                "ContextAwareRelationshipManager",
                "relationship_manager"
            )
        
        # Multimodal integrator (needs reasoning_core, attentional_controller)
        from nyx.core.multimodal_integrator import MultimodalIntegrator
        original_multimodal = MultimodalIntegrator(
            reasoning_core=self.reasoning_core,
            attentional_controller=self.attentional_controller
        )
        self.multimodal_integrator = await self._wrap_with_a2a(
            original_multimodal,
            "ContextAwareMultimodalIntegrator",
            "multimodal_integrator"
        )
        
        # Imagination simulator (needs reasoning_core, knowledge_core, emotional_core, identity_evolution)
        from nyx.core.imagination_simulator import ImaginationSimulator
        original_imagination = ImaginationSimulator(
            reasoning_core=self.reasoning_core,
            knowledge_core=self.knowledge_core,
            emotional_core=self.emotional_core,
            identity_evolution=self.identity_evolution
        )
        self.imagination_simulator = await self._wrap_with_a2a(
            original_imagination,
            "ContextAwareImaginationSimulator",
            "imagination_simulator"
        )
        
        self._init_progress.append("tier_4_complete")
    
    async def _init_tier_5_complex_systems(self):
        """Initialize complex systems with multiple dependencies"""
        self._init_progress.append("tier_5_start")
        logger.debug(f"NyxBrain Init Tier 5: Complex Systems for {self.user_id}-{self.conversation_id}")
        
        # Goal manager
        from nyx.core.goal_system import GoalManager
        self.goal_manager = GoalManager(brain_reference=self)
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_goal_manager import ContextAwareGoalManager
            self.goal_manager = ContextAwareGoalManager(self.goal_manager)
            logger.debug("Enhanced GoalManager with context distribution")
        
        # Needs system (needs goal_manager)
        from nyx.core.needs_system import NeedsSystem
        self.needs_system = NeedsSystem(goal_manager=self.goal_manager)
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_needs import ContextAwareNeedsSystem
            self.needs_system = ContextAwareNeedsSystem(self.needs_system)
            logger.debug("Enhanced NeedsSystem with context distribution")
        
        # Mood manager (needs emotional_core, hormone_system, needs_system, goal_manager)
        from nyx.core.mood_manager import MoodManager
        self.mood_manager = MoodManager(
            emotional_core=self.emotional_core,
            hormone_system=self.hormone_system,
            needs_system=self.needs_system,
            goal_manager=self.goal_manager
        )
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_mood_manager import ContextAwareMoodManager
            self.mood_manager = ContextAwareMoodManager(self.mood_manager)
            logger.debug("Enhanced MoodManager with A2A context distribution")
        
        # Digital somatosensory system (needs memory_core, emotional_core, hormone_system, needs_system)
        from nyx.core.digital_somatosensory_system import DigitalSomatosensorySystem
        original_dss = DigitalSomatosensorySystem(
            memory_core=self.memory_core,
            emotional_core=self.emotional_core,
            hormone_system=self.hormone_system,
            needs_system=self.needs_system
        )
        await original_dss.initialize()
        self.digital_somatosensory_system = await self._wrap_with_a2a(
            original_dss,
            "ContextAwareDigitalSomatosensorySystem",
            "digital_somatosensory_system"
        )
        
        # Reward system (needs emotional_core, identity_evolution, somatosensory_system, mood_manager, needs_system)
        from nyx.core.reward_system import RewardSignalProcessor
        original_reward = RewardSignalProcessor(
            emotional_core=self.emotional_core,
            identity_evolution=self.identity_evolution,
            somatosensory_system=self.digital_somatosensory_system,
            mood_manager=self.mood_manager,
            needs_system=self.needs_system
        )
        self.reward_system = await self._wrap_with_a2a(
            original_reward,
            "ContextAwareRewardSystem",
            "reward_system"
        )
        
        # Set DSS back-reference to reward system
        if hasattr(self.digital_somatosensory_system, 'set_reward_system'):
            self.digital_somatosensory_system.set_reward_system(self.reward_system)
        
        # Conditioning system initialization
        await self._init_conditioning_systems()
        
        # Theory of mind (needs relationship_manager, multimodal_integrator, memory_core)
        from nyx.core.theory_of_mind import TheoryOfMind
        self.theory_of_mind = TheoryOfMind(
            relationship_manager=self.relationship_manager,
            multimodal_integrator=self.multimodal_integrator,
            memory_core=self.memory_core
        )
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_theory_of_mind import ContextAwareTheoryOfMind
            self.theory_of_mind = ContextAwareTheoryOfMind(self.theory_of_mind)
            logger.debug("Enhanced TheoryOfMind with context distribution")
        
        # Relationship reflection (needs relationship_manager, theory_of_mind, etc.)
        if self.relationship_manager:
            try:
                from nyx.core.relationship_reflection import RelationshipReflectionSystem
                original_reflection = RelationshipReflectionSystem(
                    relationship_manager=self.relationship_manager,
                    theory_of_mind=self.theory_of_mind,
                    memory_core=self.memory_core,
                    identity_evolution=self.identity_evolution,
                    hormone_system=self.hormone_system
                )
                if self.use_a2a_integration:
                    from nyx.core.a2a.context_aware_relationship_reflection import ContextAwareRelationshipReflection
                    self.relationship_reflection = ContextAwareRelationshipReflection(original_reflection)
                    logger.debug("Enhanced RelationshipReflectionSystem with A2A context distribution")
                else:
                    self.relationship_reflection = original_reflection
            except ImportError:
                logger.warning("RelationshipReflectionSystem module not found")
                self.relationship_reflection = None
        
        self._init_progress.append("tier_5_complete")
    
    async def _init_conditioning_systems(self):
        """Initialize conditioning and related systems"""
        # Conditioning configuration
        from nyx.core.conditioning_config import ConditioningConfiguration
        from nyx.core.conditioning_system import ConditioningSystem
        from nyx.core.conditioning_maintenance import ConditioningMaintenanceSystem
        from nyx.core.input_processor import BlendedInputProcessor
        
        self.conditioning_config = ConditioningConfiguration()
        
        # Base conditioning system
        base_conditioning = ConditioningSystem(
            reward_system=self.reward_system,
            emotional_core=self.emotional_core,
            memory_core=self.memory_core,
            somatosensory_system=self.digital_somatosensory_system
        )
        self._base_conditioning_system = base_conditioning
        
        # Wrap with A2A if enabled
        self.conditioning_system = await self._wrap_with_a2a(
            base_conditioning,
            "ContextAwareConditioningSystem",
            "conditioning_system"
        )
        
        # Conditioning maintenance
        self.conditioning_maintenance = ConditioningMaintenanceSystem(
            conditioning_system=self.conditioning_system,
            reward_system=self.reward_system
        )
        await self.conditioning_maintenance.start_maintenance_scheduler(run_immediately=False)
        
        # Interaction mode manager
        from nyx.core.interaction_mode_manager import InteractionModeManager
        original_mode_manager = InteractionModeManager(
            context_system=self.context_system,
            emotional_core=self.emotional_core,
            reward_system=self.reward_system,
            goal_manager=self.goal_manager
        )
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_interaction_mode_manager import ContextAwareInteractionModeManager
            self.mode_manager = ContextAwareInteractionModeManager(original_mode_manager)
        else:
            self.mode_manager = original_mode_manager
        
        # Input processor - Initialize with brain instance
        # The BlendedInputProcessor will get subsystems from the brain
        try:
            # Check if we need to import InputProcessingConfig
            config = None
            try:
                from nyx.core.input_processing_config import InputProcessingConfig
                # Create a default config if the class exists
                config = InputProcessingConfig()
            except ImportError:
                logger.debug("InputProcessingConfig not found, using None")
            
            # Initialize BlendedInputProcessor with brain and optional config
            self.conditioned_input_processor = BlendedInputProcessor(
                brain=self,  # Pass the brain instance
                config=config  # Pass the config if available
            )
            logger.debug("Initialized BlendedInputProcessor with brain instance")
            
        except Exception as e:
            logger.error(f"Failed to initialize BlendedInputProcessor: {e}")
            # Create a simple fallback processor
            class FallbackInputProcessor:
                def __init__(self, brain=None):
                    self.brain = brain
                    # The BlendedInputProcessor gets these from brain, so we don't need to store them
                    
                async def process_input(self, text: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
                    """Simple passthrough for input"""
                    return {
                        "input_text": text,
                        "user_id": user_id or "default",
                        "detected_patterns": [],
                        "behavior_evaluations": [],
                        "recommended_behaviors": [],
                        "avoided_behaviors": [],
                        "reinforcement_results": [],
                        "mode_distribution": {},
                        "adjusted_sensitivities": {},
                        "behavior_scores": {}
                    }
                
                async def modify_response(self, response_text: str, input_processing_results: Dict[str, Any] = None) -> str:
                    """Simple passthrough for response"""
                    return response_text
                
                def get_shared_context(self):
                    """Return a dummy shared context"""
                    from nyx.core.input_processing_context import InputProcessingContext
                    return InputProcessingContext()
            
            self.conditioned_input_processor = FallbackInputProcessor(brain=self)
            logger.warning("Using fallback input processor due to initialization failure")
        
        # Wrap with A2A if enabled
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_input_processor import ContextAwareInputProcessor
            self.conditioned_input_processor = ContextAwareInputProcessor(self.conditioned_input_processor)
            logger.debug("Enhanced InputProcessor with A2A context distribution")
        
        # Initialize baseline personality
        personality_profile_obj = await self.conditioning_config.get_personality_profile()
        personality_profile_dict = personality_profile_obj.model_dump() if hasattr(personality_profile_obj, "model_dump") else personality_profile_obj
        await ConditioningSystem.initialize_baseline_personality(
            self.conditioning_system,
            personality_profile=personality_profile_dict
        )
        logger.debug("Baseline personality conditioning completed.")
    
    async def _init_tier_6_action_generation(self):
        """Initialize action generation and dependent systems"""
        self._init_progress.append("tier_6_start")
        logger.debug(f"NyxBrain Init Tier 6: Action Generation for {self.user_id}-{self.conversation_id}")
        
        # Meta core (needs multiple systems)
        from nyx.core.meta_core import MetaCore
        self.meta_core = MetaCore()
        meta_core_deps = {
            "memory": self.memory_core,
            "emotion": self.emotional_core,
            "reasoning": self.reasoning_core,
            "reflection": None,  # Set later
            "adaptation": self.dynamic_adaptation,
            "feedback": self.internal_feedback,
            "identity": self.identity_evolution,
            "experience": self.experience_interface,
            "hormone": self.hormone_system,
            "time": self.temporal_perception,
            "procedural": self.agent_enhanced_memory,
            "needs": self.needs_system,
            "goals": self.goal_manager,
            "mood": self.mood_manager,
            "theory_of_mind": self.theory_of_mind,
            "imagination": self.imagination_simulator
        }
        await self.meta_core.initialize(meta_core_deps)
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_meta_core import ContextAwareMetaCore
            self.meta_core = ContextAwareMetaCore(self.meta_core)
            logger.debug("Enhanced MetaCore with A2A context distribution")
        
        # Initialize components needed by action generator
        await self._init_action_generator_dependencies()
        
        # Agentic action generator
        from nyx.core.agentic_action_generator import EnhancedAgenticActionGenerator
        original_action_gen = EnhancedAgenticActionGenerator(
            emotional_core=self.emotional_core,
            hormone_system=self.hormone_system,
            experience_interface=self.experience_interface,
            imagination_simulator=self.imagination_simulator,
            meta_core=self.meta_core,
            memory_core=self.memory_core,
            goal_system=self.goal_manager,
            identity_evolution=self.identity_evolution,
            knowledge_core=self.knowledge_core,
            input_processor=self.conditioned_input_processor,
            internal_feedback=self.internal_feedback,
            attentional_controller=self.attentional_controller,
            reasoning_core=self.reasoning_core,
            reflection_engine=None,  # Set later
            mood_manager=self.mood_manager,
            needs_system=self.needs_system,
            mode_integration=None,  # Set later
            multimodal_integrator=self.multimodal_integrator,
            reward_system=self.reward_system,
            theory_of_mind=self.theory_of_mind,
            relationship_manager=self.relationship_manager,
            temporal_perception=self.temporal_perception,
            passive_observation_system=None,  # Set later
            proactive_communication_engine=None,  # Set later
            creative_system=None,  # Set in tier 7
            creative_memory=None,  # Set in tier 7
            capability_assessor=None,  # Set in tier 7
            system_context=self.system_context,
            procedural_memory_manager=self.procedural_memory_manager,
            prediction_engine=getattr(self, 'prediction_engine', None),
            autobiographical_narrative=getattr(self, 'autobiographical_narrative', None),
            body_image=getattr(self, 'body_image', None),
            conditioning_system=self.conditioning_system,
            issue_tracker=getattr(self, 'issue_tracker', None),
            relationship_reflection=getattr(self, 'relationship_reflection', None)
        )
        
        self.agentic_action_generator = await self._wrap_with_a2a(
            original_action_gen,
            "ContextAwareAgenticActionGenerator",
            "agentic_action_generator"
        )
        
        if hasattr(self.agentic_action_generator, 'initialize_actions'):
            await self.agentic_action_generator.initialize_actions()
        
        # Initialize systems that depend on action generator
        await self._init_action_dependent_systems()
        
        self._init_progress.append("tier_6_complete")
    
    async def _init_action_generator_dependencies(self):
        """Initialize components needed by action generator"""
        # Prediction engine
        if not hasattr(self, 'prediction_engine'):
            from nyx.core.prediction_engine import PredictionEngine
            self.prediction_engine = PredictionEngine()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_prediction_engine import ContextAwarePredictionEngine
            self.prediction_engine = ContextAwarePredictionEngine(self.prediction_engine)
            logger.debug("Enhanced PredictionEngine with A2A context distribution")
        
        # Body image
        if hasattr(self.config, 'body_image') and self.config.body_image.enabled:
            from nyx.core.body_image import BodyImage
            original_body_image = BodyImage()
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_body_image import ContextAwareBodyImage
                self.body_image = ContextAwareBodyImage(original_body_image)
                logger.debug("Enhanced BodyImage with A2A context distribution")
            else:
                self.body_image = original_body_image
        
        # Autobiographical narrative
        try:
            from nyx.core.autobiographical_narrative import AutobiographicalNarrative
            original_narrative = AutobiographicalNarrative(
                memory_orchestrator=self.memory_orchestrator,
                identity_evolution=self.identity_evolution,
                relationship_manager=self.relationship_manager
            )
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_autobiographical_narrative import ContextAwareAutobiographicalNarrative
                self.autobiographical_narrative = ContextAwareAutobiographicalNarrative(original_narrative)
                logger.debug("Enhanced AutobiographicalNarrative with A2A context distribution")
            else:
                self.autobiographical_narrative = original_narrative
        except ImportError:
            logger.warning("AutobiographicalNarrative module not found")
            self.autobiographical_narrative = None
        
        # Distributed processing
        try:
            from nyx.core.distributed_processing import DistributedProcessingManager
            original_distributed = DistributedProcessingManager(max_parallel_tasks=10)
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_distributed_processing import ContextAwareDistributedProcessing
                self.distributed_processing = ContextAwareDistributedProcessing(original_distributed)
                logger.debug("Enhanced DistributedProcessingManager with A2A context distribution")
            else:
                self.distributed_processing = original_distributed
        except ImportError:
            logger.warning("DistributedProcessingManager module not found")
            self.distributed_processing = None
    
    async def _init_action_dependent_systems(self):
        """Initialize systems that depend on action generator"""
        # Passive observation system
        from nyx.core.passive_observation import PassiveObservationSystem
        self.passive_observation_system = PassiveObservationSystem(
            action_generator=self.agentic_action_generator,
            emotional_core=self.emotional_core,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager,
            temporal_perception=self.temporal_perception,
            multimodal_integrator=self.multimodal_integrator,
            mood_manager=self.mood_manager,
            needs_system=self.needs_system,
            identity_evolution=self.identity_evolution,
            attention_controller=self.attentional_controller
        )
        await self.passive_observation_system.start()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_passive_observation import ContextAwarePassiveObservation
            original_passive = self.passive_observation_system
            self.passive_observation_system = ContextAwarePassiveObservation(original_passive)
            logger.debug("Enhanced PassiveObservationSystem with A2A context distribution")
            await self.passive_observation_system.start()
        
        # Proactive communication engine
        from nyx.core.proactive_communication import ProactiveCommunicationEngine
        original_proactive = ProactiveCommunicationEngine(
            action_generator=self.agentic_action_generator,
            emotional_core=self.emotional_core,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager,
            temporal_perception=self.temporal_perception,
            reasoning_core=self.reasoning_core,
            reflection_engine=None,  # Set later
            mood_manager=self.mood_manager,
            needs_system=self.needs_system,
            identity_evolution=self.identity_evolution
        )
        await original_proactive.start()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_proactive_communication import ContextAwareProactiveCommunication
            await original_proactive.stop()
            self.proactive_communication_engine = ContextAwareProactiveCommunication(original_proactive)
            logger.debug("Enhanced ProactiveCommunicationEngine with A2A context distribution")
            await self.proactive_communication_engine.start()
        else:
            self.proactive_communication_engine = original_proactive
        
        # Reflection engine
        from nyx.core.reflection_engine import ReflectionEngine
        original_reflection = ReflectionEngine(
            memory_core_ref=self.memory_core,
            emotional_core=self.emotional_core,
            passive_observation_system=self.passive_observation_system,
            proactive_communication_engine=self.proactive_communication_engine
        )
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_reflection_engine import ContextAwareReflectionEngine
            self.reflection_engine = ContextAwareReflectionEngine(original_reflection)
            logger.debug("Enhanced ReflectionEngine with A2A context distribution")
        else:
            self.reflection_engine = original_reflection
        
        # Update circular references
        self.agentic_action_generator.reflection_engine = self.reflection_engine
        self.agentic_action_generator.passive_observation_system = self.passive_observation_system
        self.agentic_action_generator.proactive_communication_engine = self.proactive_communication_engine
        if self.proactive_communication_engine:
            self.proactive_communication_engine.reflection_engine = self.reflection_engine
        if self.meta_core and hasattr(self.meta_core, 'context_data') and isinstance(self.meta_core.context_data, dict):
            self.meta_core.context_data['reflection'] = self.reflection_engine
        
        # Internal thoughts manager
        from nyx.core.internal_thoughts import InternalThoughtsManager
        original_thoughts = InternalThoughtsManager(
            passive_observation_system=self.passive_observation_system,
            reflection_engine=self.reflection_engine,
            imagination_simulator=self.imagination_simulator,
            theory_of_mind=self.theory_of_mind,
            relationship_reflection=self.relationship_manager,
            proactive_communication=self.proactive_communication_engine,
            emotional_core=self.emotional_core,
            memory_core=self.memory_core
        )
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_internal_thoughts import ContextAwareInternalThoughts
            self.thoughts_manager = ContextAwareInternalThoughts(original_thoughts)
            logger.debug("Enhanced InternalThoughtsManager with A2A context distribution")
        else:
            self.thoughts_manager = original_thoughts
    
    async def _init_tier_7_specialized_systems(self):
        """Initialize specialized systems (FemDom, Creative, Spatial, etc.)"""
        self._init_progress.append("tier_7_start")
        logger.debug(f"NyxBrain Init Tier 7: Specialized Systems for {self.user_id}-{self.conversation_id}")
        
        # Initialize FemDom systems
        await self._init_femdom_systems()
        
        # Initialize creative systems
        await self._init_creative_systems()
        
        # Initialize spatial systems
        await self._init_spatial_systems()
        
        # Initialize other specialized systems
        await self._init_other_specialized_systems()
        
        self._init_progress.append("tier_7_complete")
    
    async def _init_femdom_systems(self):
        """Initialize FemDom components in dependency order"""
        from nyx.core.femdom.protocol_enforcement import ProtocolEnforcement
        from nyx.core.femdom.body_service_system import BodyServiceSystem
        from nyx.core.femdom.psychological_dominance import PsychologicalDominance
        from nyx.core.femdom.orgasm_control import OrgasmControlSystem
        from nyx.core.femdom.persona_manager import DominancePersonaManager
        from nyx.core.femdom.sadistic_responses import SadisticResponseSystem
        from nyx.core.femdom.submission_progression import SubmissionProgression
        from nyx.core.femdom.task_assignment_system import TaskAssignmentSystem
        from nyx.core.femdom.femdom_coordinator import FemdomCoordinator
        from nyx.core.femdom.femdom_integration_manager import FemdomIntegrationManager
        
        # Initialize components without circular dependencies first
        original_protocol = ProtocolEnforcement(
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        self.protocol_enforcement = await self._wrap_with_a2a(
            original_protocol,
            "ContextAwareProtocolEnforcement",
            "protocol_enforcement"
        )
        
        original_body_service = BodyServiceSystem(
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        self.body_service_system = await self._wrap_with_a2a(
            original_body_service,
            "ContextAwareBodyService",
            "body_service_system"
        )
        
        original_psychological = PsychologicalDominance(
            theory_of_mind=self.theory_of_mind,
            reward_system=self.reward_system,
            relationship_manager=self.relationship_manager,
            memory_core=self.memory_core
        )
        self.psychological_dominance = await self._wrap_with_a2a(
            original_psychological,
            "ContextAwarePsychologicalDominance",
            "psychological_dominance"
        )
        
        original_orgasm_control = OrgasmControlSystem(
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager,
            somatosensory_system=self.digital_somatosensory_system
        )
        self.orgasm_control_system = await self._wrap_with_a2a(
            original_orgasm_control,
            "ContextAwareOrgasmControl",
            "orgasm_control_system"
        )
        
        original_persona = DominancePersonaManager(
            relationship_manager=self.relationship_manager,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            emotional_core=self.emotional_core
        )
        self.dominance_persona_manager = await self._wrap_with_a2a(
            original_persona,
            "ContextAwarePersonaManager",
            "dominance_persona_manager"
        )
        
        original_sadistic = SadisticResponseSystem(
            theory_of_mind=self.theory_of_mind,
            protocol_enforcement=self.protocol_enforcement,
            reward_system=self.reward_system,
            relationship_manager=self.relationship_manager,
            memory_core=self.memory_core
        )
        self.sadistic_response_system = await self._wrap_with_a2a(
            original_sadistic,
            "ContextAwareSadisticResponses",
            "sadistic_response_system"
        )
        
        original_submission = SubmissionProgression(
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        self.submission_progression = await self._wrap_with_a2a(
            original_submission,
            "ContextAwareSubmissionProgression",
            "submission_progression"
        )
        
        # Now create FemdomCoordinator
        original_coordinator = FemdomCoordinator(self)
        await original_coordinator.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_femdom_coordinator import ContextAwareFemdomCoordinator
            self.femdom_coordinator = ContextAwareFemdomCoordinator(original_coordinator)
            logger.debug("Enhanced FemdomCoordinator with A2A context distribution")
        else:
            self.femdom_coordinator = original_coordinator
        
        # Set dominance_system reference
        self.dominance_system = self.femdom_coordinator
        if self.use_a2a_integration and self.dominance_system:
            from nyx.core.a2a.context_aware_dominance import ContextAwareDominanceSystem
            self.dominance_system = ContextAwareDominanceSystem(self.dominance_system)
            logger.debug("Enhanced DominanceSystem with A2A context distribution")
        
        # Create TaskAssignmentSystem with all dependencies
        original_task_assignment = TaskAssignmentSystem(
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager,
            submission_progression=self.submission_progression,
            dominance_system=self.dominance_system,
            psychological_dominance=self.psychological_dominance,
            protocol_enforcement=self.protocol_enforcement,
            sadistic_responses=self.sadistic_response_system
        )
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_task_assignment import ContextAwareTaskAssignment
            self.task_assignment_system = ContextAwareTaskAssignment(original_task_assignment)
            logger.debug("Enhanced TaskAssignmentSystem with A2A context distribution")
        else:
            self.task_assignment_system = original_task_assignment
        
        # Create FemdomIntegrationManager
        femdom_components = {
            "protocol_enforcement": self.protocol_enforcement,
            "body_service": self.body_service_system,
            "psychological_dominance": self.psychological_dominance,
            "reward_system": self.reward_system,
            "memory_core": self.memory_core,
            "relationship_manager": self.relationship_manager,
            "theory_of_mind": self.theory_of_mind,
            "orgasm_control": self.orgasm_control_system,
            "persona_manager": self.dominance_persona_manager,
            "sadistic_responses": self.sadistic_response_system,
            "dominance_system": self.dominance_system,
            "submission_progression": self.submission_progression,
            "task_assignment_system": self.task_assignment_system
        }
        
        original_integration = FemdomIntegrationManager(self, components=femdom_components)
        await original_integration.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_femdom_integration import ContextAwareFemdomIntegration
            self.femdom_integration_manager = ContextAwareFemdomIntegration(original_integration)
            logger.debug("Enhanced FemdomIntegrationManager with A2A context distribution")
        else:
            self.femdom_integration_manager = original_integration
    
    async def _init_creative_systems(self):
        """Initialize creative and novelty systems"""
        from nyx.core.novelty_engine import NoveltyEngine
        from nyx.core.recognition_memory import RecognitionMemorySystem
        from nyx.core.creative_memory_integration import CreativeMemoryIntegration
        from nyx.creative.agentic_system import integrate_with_existing_system
        
        # Novelty engine
        original_novelty = NoveltyEngine(
            imagination_simulator=self.imagination_simulator,
            memory_core=self.memory_core,
            reasoning_core=self.reasoning_core
        )
        await original_novelty.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_novelty_engine import ContextAwareNoveltyEngine
            self.novelty_engine = ContextAwareNoveltyEngine(original_novelty)
            logger.debug("Enhanced NoveltyEngine with A2A context distribution")
        else:
            self.novelty_engine = original_novelty
        
        # Recognition memory
        original_recognition = RecognitionMemorySystem(
            memory_core=self.memory_core,
            context_awareness=self.context_system,
            reasoning_core=self.reasoning_core
        )
        await original_recognition.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_recognition_memory import ContextAwareRecognitionMemory
            self.recognition_memory = ContextAwareRecognitionMemory(original_recognition)
            logger.debug("Enhanced RecognitionMemorySystem with A2A context distribution")
        else:
            self.recognition_memory = original_recognition
        
        # Creative memory integration
        original_creative_memory = CreativeMemoryIntegration(
            novelty_engine=self.novelty_engine,
            recognition_memory=self.recognition_memory,
            memory_core=self.memory_core
        )
        await original_creative_memory.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_creative_memory_integration import ContextAwareCreativeMemoryIntegration
            self.creative_memory = ContextAwareCreativeMemoryIntegration(original_creative_memory)
            logger.debug("Enhanced CreativeMemoryIntegration with A2A context distribution")
        else:
            self.creative_memory = original_creative_memory
        
        # Creative system integration
        self.creative_system = await integrate_with_existing_system(self)
        
        # Initialize capability_assessor to None first
        self.capability_assessor = None
        
        # Set up content store and capability assessment
        if hasattr(self.creative_system, 'storage') and self.creative_system.storage and \
           hasattr(self.creative_system.storage, 'db_path') and \
           hasattr(self.creative_system, 'content_system') and self.creative_system.content_system:
            
            from pathlib import Path
            from nyx.creative.content_system import CreativeContentSystem
            from nyx.creative.capability_system import CapabilityModel, CapabilityAssessmentSystem
            
            self.content_store = self.creative_system.content_system
            creations_dir = Path(self.creative_system.storage.db_path).parent
            model_filename = f"capability_model_{self.user_id}_{self.conversation_id}.json"
            model_path = creations_dir / model_filename
            self.capability_model = CapabilityModel(storage_path=str(model_path))
            self.capability_assessor = CapabilityAssessmentSystem(
                creative_content_system=self.creative_system.storage,
                capability_model_path=str(model_path)
            )
            
            if hasattr(self, "_start_creative_review_task") and callable(self._start_creative_review_task):
                self._start_creative_review_task()
            
            logger.info(f"Creative system initialized. Content store base: {getattr(self.content_store, 'base_directory', 'N/A')}")
        else:
            logger.warning("Creative system or its sub-components (storage/content_system) not fully available after integration.")
        
        # Update action generator with creative references AFTER they're created
        if self.agentic_action_generator:
            self.agentic_action_generator.creative_system = self.creative_system
            self.agentic_action_generator.creative_memory = self.creative_memory
            self.agentic_action_generator.capability_assessor = self.capability_assessor
    
    async def _init_spatial_systems(self):
        """Initialize spatial navigation systems"""
        from nyx.core.spatial.spatial_mapper import SpatialMapper
        from nyx.core.spatial.spatial_memory import SpatialMemoryIntegration
        from nyx.core.spatial.map_visualization import MapVisualization
        from nyx.core.spatial.navigator_agent import SpatialNavigatorAgent
        
        # Spatial mapper
        original_mapper = SpatialMapper(memory_integration=self.memory_core)
        if hasattr(original_mapper, "initialize"):
            await original_mapper.initialize()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_spatial_mapper import ContextAwareSpatialMapper
            self.spatial_mapper = ContextAwareSpatialMapper(original_mapper)
            logger.debug("Enhanced SpatialMapper with A2A context distribution")
        else:
            self.spatial_mapper = original_mapper
        
        # Spatial memory
        original_spatial_memory = SpatialMemoryIntegration(
            spatial_mapper=self.spatial_mapper,
            memory_core=self.memory_core
        )
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_spatial_memory import ContextAwareSpatialMemoryIntegration
            self.spatial_memory = ContextAwareSpatialMemoryIntegration(original_spatial_memory)
            logger.debug("Enhanced SpatialMemoryIntegration with A2A context distribution")
        else:
            self.spatial_memory = original_spatial_memory
        
        # Map visualization (no A2A needed)
        self.map_visualization = MapVisualization()
        
        # Navigator agent
        original_navigator = SpatialNavigatorAgent(spatial_mapper=self.spatial_mapper)
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_navigator_agent import ContextAwareSpatialNavigatorAgent
            self.navigator_agent = ContextAwareSpatialNavigatorAgent(original_navigator)
            logger.debug("Enhanced SpatialNavigatorAgent with A2A context distribution")
        else:
            self.navigator_agent = original_navigator
    
    async def _init_other_specialized_systems(self):
        """Initialize remaining specialized systems"""
        # Mode integration manager
        from nyx.core.mode_integration import ModeIntegrationManager
        self.mode_integration = ModeIntegrationManager(nyx_brain=self)
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_mode_integration import ContextAwareModeIntegration
            self.mode_integration = ContextAwareModeIntegration(self.mode_integration)
            logger.debug("Enhanced ModeIntegrationManager with A2A context distribution")
        
        if self.agentic_action_generator:
            self.agentic_action_generator.mode_integration = self.mode_integration
        
        # Issue tracking system
        from nyx.core.issue_tracking_system import IssueTrackingSystem
        original_issue_tracker = IssueTrackingSystem(
            db_path=f"issues_db_{self.user_id}_{self.conversation_id}.json"
        )
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_issue_tracking_system import ContextAwareIssueTrackingSystem
            self.issue_tracking_system = ContextAwareIssueTrackingSystem(original_issue_tracker)
            logger.debug("Enhanced IssueTrackingSystem with A2A context distribution")
        else:
            self.issue_tracking_system = original_issue_tracker
        
        # Reflexive system
        try:
            from nyx.core.reflexive_system import ReflexiveSystem
            original_reflexive = ReflexiveSystem(agent_enhanced_memory=self.agent_enhanced_memory)
            if hasattr(original_reflexive, "initialize"):
                await original_reflexive.initialize()
            
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_reflexive_system import ContextAwareReflexiveSystem
                self.reflexive_system = ContextAwareReflexiveSystem(original_reflexive)
                logger.debug("Enhanced ReflexiveSystem with A2A context distribution")
            else:
                self.reflexive_system = original_reflexive
        except ImportError:
            logger.info("Reflexive system module not found.")
            self.reflexive_system = None
        
        # Dominance ideation agents
        try:
            from nyx.core.dominance import create_dominance_ideation_agent, create_hard_dominance_ideation_agent
            self.general_dominance_ideation_agent = create_dominance_ideation_agent()
            self.hard_dominance_ideation_agent = create_hard_dominance_ideation_agent()
        except ImportError:
            logger.warning("Dominance ideation agents not available")
    
    async def _init_tier_8_final_setup(self):
        """Final setup and integration tasks"""
        self._init_progress.append("tier_8_start")
        logger.debug(f"NyxBrain Init Tier 8: Final Setup for {self.user_id}-{self.conversation_id}")
        
        # Processing manager
        self.processing_manager = UnifiedProcessor(brain=self)
        await self.processing_manager.initialize()
        
        # Self configuration manager
        from nyx.core.brain.adaptation.self_config import SelfConfigManager
        self.self_config_manager = SelfConfigManager(brain=self)
        
        # Brain agent
        self.brain_agent = self._create_brain_agent()
        
        # Integrated tracer
        if not self.integrated_tracer:
            try:
                from nyx.core.integration.integrated_tracer import IntegratedTracer
                # Check what parameters IntegratedTracer actually accepts
                # It might just need no parameters or different ones
                self.integrated_tracer = IntegratedTracer()
                # Set any needed attributes after creation
                if hasattr(self.integrated_tracer, 'brain_id'):
                    self.integrated_tracer.brain_id = f"{self.user_id}_{self.conversation_id}"
                if hasattr(self.integrated_tracer, 'event_bus'):
                    self.integrated_tracer.event_bus = self.event_bus
                logger.debug("Integrated tracer initialized")
            except ImportError:
                logger.warning("IntegratedTracer module not found")
                self.integrated_tracer = None
            except Exception as e:
                logger.error(f"Error initializing IntegratedTracer: {e}")
                self.integrated_tracer = None
        
        # Integration manager
        from nyx.core.integration.integration_manager import create_integration_manager
        self.integration_manager = create_integration_manager(self)
        await self.integration_manager.initialize()
        
        # Sync daemon and related components
        from nyx.core.sync.nyx_sync_daemon import NyxSyncDaemon
        original_sync = NyxSyncDaemon()
        
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_nyx_sync_daemon import ContextAwareNyxSyncDaemon
            self.sync_daemon = ContextAwareNyxSyncDaemon(original_sync)
            logger.debug("Enhanced NyxSyncDaemon with A2A context distribution")
        else:
            self.sync_daemon = original_sync
        
        # Strategy controller
        if not self.strategy_controller:
            try:
                from nyx.core.sync.strategy_controller import StrategyController
                self.strategy_controller = StrategyController(
                    sync_daemon=self.sync_daemon,
                    brain_ref=self
                )
                await self.strategy_controller.initialize()
                logger.debug("Strategy controller initialized")
            except ImportError:
                logger.warning("StrategyController module not found")
                self.strategy_controller = None
        
        # Noise filter
        if not self.noise_filter:
            try:
                from nyx.core.sync.noise_filter import NoiseFilter
                self.noise_filter = NoiseFilter(
                    sync_daemon=self.sync_daemon,
                    strategy_controller=self.strategy_controller
                )
                logger.debug("Noise filter initialized")
            except ImportError:
                try:
                    from nyx.core.filters.noise_filter import NoiseFilter
                    self.noise_filter = NoiseFilter()
                    logger.debug("Noise filter initialized (standalone)")
                except ImportError:
                    logger.warning("NoiseFilter module not found")
                    self.noise_filter = None

        # Start orchestrator background tasks
        try:
            orchestrator.start_background()
            self._orchestrator_started = True
            self._orch_started = True
            logger.info("Orchestrator background tasks started")
        except Exception as e:
            logger.error(f"Failed to start orchestrator background: {e}")
            self._orchestrator_started = False
        
        # Tools
        from nyx.core.tools.evaluator import AgentEvaluator
        from nyx.core.tools.parallel import ParallelToolExecutor
        
        original_evaluator = AgentEvaluator()
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_evaluator import ContextAwareAgentEvaluator
            self.agent_evaluator = ContextAwareAgentEvaluator(original_evaluator)
            logger.debug("Enhanced AgentEvaluator with A2A context distribution")
        else:
            self.agent_evaluator = original_evaluator
        
        original_parallel = ParallelToolExecutor()
        if self.use_a2a_integration:
            from nyx.core.a2a.context_aware_parallel import ContextAwareParallelExecutor
            self.parallel_executor = ContextAwareParallelExecutor(original_parallel)
            logger.debug("Enhanced ParallelToolExecutor with A2A context distribution")
        else:
            self.parallel_executor = original_parallel
        
        # Thinking tools
        from nyx.apitools.thinking_tools import should_use_extended_thinking, generate_reasoned_response
        from agents import function_tool
        self.thinking_tools = {
            "should_use_extended_thinking": function_tool(should_use_extended_thinking),
            "think_before_responding": function_tool(generate_reasoned_response),
            "generate_reasoned_response": function_tool(generate_reasoned_response)
        }
        
        # Final setup tasks
        await self._register_processing_modules()
        await self.integrate_procedural_memory_with_actions()
        
        if hasattr(self, "agentic_action_generator") and hasattr(self, "_register_creative_actions") and callable(self._register_creative_actions):
            await self._register_creative_actions()
        
        # Initialize creative and tool systems
        try:
            from nyx.core.a2a.context_aware_setup import setup_context_aware_creative_modules
            creative_modules = await setup_context_aware_creative_modules(nyx_brain=self)
            
            for module_name, module in creative_modules.items():
                if not hasattr(self, module_name):
                    setattr(self, module_name, module)
            
            logger.info("✓ Creative and tool systems initialized and integrated")
        except Exception as e:
            logger.error(f"Failed to initialize creative/tool systems: {e}", exc_info=True)
        
        # Initialize streaming if enabled
        if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
            await self._init_streaming_systems()
        
        # Build module registry
        await self._build_internal_module_registry()
        
        # Build default active modules list - organized by initialization tiers
        default_modules_list = [
            # Tier 0: Infrastructure
            "workspace_engine",
            "module_optimizer", 
            "system_health_checker",
            
            # Tier 1: Foundation
            "hormone_system",
            "temporal_perception",
            "knowledge_core",
            "procedural_memory_manager",
            "agent_enhanced_memory",
            "internal_feedback",
            "dynamic_adaptation",
            
            # Tier 2: Core Emotional/Memory
            "emotional_core",
            "memory_core",
            "memory_orchestrator",
            "identity_evolution",
            
            # Tier 3: Cognitive Systems
            "reasoning_core",
            "attentional_controller",
            "context_system",
            
            # Tier 4: Integration Systems
            "experience_interface",
            "experience_consolidation",
            "cross_user_manager",
            "relationship_manager",
            "multimodal_integrator",
            "imagination_simulator",
            
            # Tier 5: Complex Systems
            "goal_manager",
            "needs_system",
            "mood_manager",
            "digital_somatosensory_system",
            "reward_system",
            "conditioning_system",
            "mode_manager",
            "theory_of_mind",
            "relationship_reflection",
            
            # Tier 6: Action Generation
            "meta_core",
            "agentic_action_generator",
            "passive_observation_system",
            "proactive_communication_engine",
            "reflection_engine",
            "thoughts_manager",  # internal_thoughts
            "prediction_engine",
            "body_image",
            "autobiographical_narrative",
            
            # Tier 7: Specialized Systems - FemDom
            "femdom_coordinator",
            "dominance_system",
            "protocol_enforcement",
            "body_service_system",
            "psychological_dominance",
            "orgasm_control_system",
            "dominance_persona_manager",
            "sadistic_response_system",
            "submission_progression",
            "task_assignment_system",
            "femdom_integration_manager",
            
            # Tier 7: Specialized Systems - Creative
            "creative_system",
            "creative_memory",
            "novelty_engine",
            "recognition_memory",
            "content_store",
            "capability_assessor",
            
            # Tier 7: Specialized Systems - Spatial
            "spatial_mapper",
            "spatial_memory",
            "map_visualization",
            "navigator_agent",
            
            # Tier 7: Specialized Systems - Other
            "mode_integration",
            "issue_tracking_system",
            "reflexive_system",
            "general_dominance_ideation_agent",
            "hard_dominance_ideation_agent",
            
            # Tier 8: Final Setup
            "processing_manager",
            "self_config_manager",
            "brain_agent",
            "integrated_tracer",
            "integration_manager",
            "sync_daemon",
            "strategy_controller",
            "noise_filter",
            "agent_evaluator",
            "parallel_executor",
            
            # Streaming Systems (conditional - only if ENABLE_STREAMING=true)
            # "cross_game_knowledge",
            # "game_vision",
            # "game_processor",
            # "gamer_girl",
            # "game_state",
            # "speech_recognition",
            # "game_learning_manager",
            # "game_multimodal_integrator",
            # "audience_interaction",
            # "streaming_core",
            # "streaming_hormone_system",
            # "streaming_reflection_engine",
            # "streaming_integration",
        ]
        
        # Filter to only include modules that were actually initialized
        self.default_active_modules = {
            module for module in default_modules_list 
            if hasattr(self, module) and getattr(self, module) is not None
        }
        
        # Add streaming modules if they were initialized
        if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
            streaming_modules = [
                "cross_game_knowledge",
                "game_vision",
                "gamer_girl",
                "streaming_core",
                "streaming_hormone_system", 
                "streaming_reflection_engine",
            ]
            
            # Add streaming_integration only if A2A is enabled
            if self.use_a2a_integration and hasattr(self, 'streaming_integration'):
                streaming_modules.append("streaming_integration")
            
            for module in streaming_modules:
                if hasattr(self, module) and getattr(self, module) is not None:
                    self.default_active_modules.add(module)
        
        # Initialize agent capabilities if enabled
        if os.environ.get("ENABLE_AGENT", "true").lower() == "true":
            await self.initialize_agent_capabilities()
        
        # Initialize streaming if enabled
        if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
            await self.initialize_streaming()
        
        self._init_progress.append("tier_8_complete")
    
    async def _init_streaming_systems(self):
        """Initialize streaming and game analysis systems"""
        logger.debug(f"NyxBrain Init: Streaming and Game Analysis Systems for {self.user_id}-{self.conversation_id}")
        
        # Initialize Cross-Game Knowledge System
        try:
            from nyx.streamer.cross_game_knowledge import CrossGameKnowledgeSystem
            
            # Create original cross-game knowledge system
            original_cross_game = CrossGameKnowledgeSystem(data_dir=f"cross_game_data_{self.user_id}")
            
            # Seed with initial knowledge if needed
            if not original_cross_game.games:
                original_cross_game.seed_initial_knowledge()
            
            # Wrap with context-aware version if A2A enabled
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_cross_game_knowledge import ContextAwareCrossGameKnowledge
                self.cross_game_knowledge = ContextAwareCrossGameKnowledge(original_cross_game)
                logger.debug("Enhanced CrossGameKnowledgeSystem with A2A context distribution")
            else:
                self.cross_game_knowledge = original_cross_game
                
        except ImportError as e:
            logger.warning(f"CrossGameKnowledgeSystem module not found: {e}")
            self.cross_game_knowledge = None
        
        # Initialize Enhanced Game Vision System
        try:
            from nyx.streamer.enhanced_game_vision import (
                EnhancedGameRecognitionSystem, GameKnowledgeBase,
                EnhancedSpatialMemory, SceneGraphAnalyzer, GameActionRecognition,
                RealTimeGameProcessor
            )
            
            # Create game knowledge base
            game_knowledge_base = GameKnowledgeBase(data_dir=f"game_data_{self.user_id}")
            
            # Create original game recognition system
            original_game_vision = EnhancedGameRecognitionSystem(knowledge_base=game_knowledge_base)
            
            # Seed with initial knowledge if needed
            if not game_knowledge_base.games:
                original_game_vision.seed_initial_knowledge()
            
            # Wrap with context-aware version if A2A enabled
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_game_vision import ContextAwareGameVision
                self.game_vision = ContextAwareGameVision(original_game_vision)
                logger.debug("Enhanced GameVisionSystem with A2A context distribution")
            else:
                self.game_vision = original_game_vision
            
            # Initialize the real-time processor if needed
            self.game_processor = RealTimeGameProcessor(
                game_system=self.game_vision,
                input_source=0,  # Default camera/video source
                processing_fps=30
            )
            
        except ImportError as e:
            logger.warning(f"EnhancedGameVisionSystem module not found: {e}")
            self.game_vision = None
            self.game_processor = None
        
        # Initialize Gamer Girl Streaming System
        try:
            from nyx.streamer.gamer_girl import (
                GameState, HormoneSystem as StreamerHormoneSystem,
                SpeechRecognitionSystem, CrossGameKnowledgeSystem as StreamerCrossGame,
                GameSessionLearningManager, EnhancedAudienceInteraction,
                EnhancedMultiModalIntegrator
            )
            
            # Create game state
            self.game_state = GameState()
            
            # Create speech recognition system
            self.speech_recognition = SpeechRecognitionSystem(model_size="base", language="en")
            
            # Create learning manager (streaming_core will be set later)
            self.game_learning_manager = GameSessionLearningManager(
                brain=self,
                streaming_core=None
            )
            
            # Create multi-modal integrator
            self.game_multimodal_integrator = EnhancedMultiModalIntegrator(self.game_state)
            
            # Create audience interaction system
            self.audience_interaction = EnhancedAudienceInteraction(self.game_state)
            
            # Create the main streaming system
            from nyx.streamer.gamer_girl import GamerGirl  # Add this import
            # Then use:
            original_gamer_girl = GamerGirl(
                game_state=self.game_state,
                speech_recognition=self.speech_recognition,
                learning_manager=self.game_learning_manager,
                multimodal_integrator=self.game_multimodal_integrator,
                audience_interaction=self.audience_interaction,
                cross_game_knowledge=self.cross_game_knowledge,
                game_vision=self.game_vision,
                hormone_system=self.hormone_system  
            )
            
            # Wrap with context-aware version if A2A enabled
            if self.use_a2a_integration:
                from nyx.core.a2a.context_aware_gamer_girl import ContextAwareGamerGirl
                self.gamer_girl = ContextAwareGamerGirl(original_gamer_girl)
                logger.debug("Enhanced GamerGirl streaming system with A2A context distribution")
            else:
                self.gamer_girl = original_gamer_girl
            
        except ImportError as e:
            logger.warning(f"GamerGirl streaming system module not found: {e}")
            self.gamer_girl = None
            self.game_state = None
            self.speech_recognition = None
            self.game_learning_manager = None
            self.game_multimodal_integrator = None
            self.audience_interaction = None
        
        # Initialize Integrated Streaming Core System
        try:
            from nyx.streamer.nyx_streaming_core import StreamingCore, OptimizedStreamingCore
            from nyx.streamer.streaming_hormone_system import StreamingHormoneSystem
            from nyx.streamer.streaming_reflection import StreamingReflectionEngine, EnhancedStreamingReflectionEngine
            
            # Determine video and audio sources
            video_source = int(os.environ.get("STREAMING_VIDEO_SOURCE", "0"))
            audio_source = os.environ.get("STREAMING_AUDIO_SOURCE", None)
            
            if self.use_a2a_integration:
                # Use the integrated context-aware streaming system
                from nyx.core.a2a.context_aware_streaming_core import ContextAwareStreamingCore
                from nyx.core.a2a.context_aware_streaming_hormone_system import ContextAwareStreamingHormoneSystem
                from nyx.core.a2a.context_aware_streaming_reflection import ContextAwareStreamingReflectionEngine
                from nyx.core.a2a.context_aware_streaming_integration import ContextAwareStreamingIntegration
                
                self.streaming_integration = await ContextAwareStreamingIntegration.create_integrated_streaming(
                    brain=self,
                    video_source=video_source,
                    audio_source=audio_source
                )
                
                # Extract components from integration
                self.streaming_core = self.streaming_integration.streaming_core
                self.streaming_hormone_system = self.streaming_integration.hormone_system
                self.streaming_reflection_engine = self.streaming_integration.reflection_engine
                
                # Update learning manager reference
                if self.game_learning_manager:
                    self.game_learning_manager.streaming_core = self.streaming_core
                
                logger.debug("Initialized integrated context-aware streaming system with A2A")
            else:
                # Create original streaming components
                # Use OptimizedStreamingCore for better performance
                self.streaming_core = OptimizedStreamingCore(
                    brain=self,
                    video_source=video_source,
                    audio_source=audio_source
                )
                
                # Create hormone system
                self.streaming_hormone_system = StreamingHormoneSystem(brain=self)
                
                # Create enhanced reflection engine
                self.streaming_reflection_engine = EnhancedStreamingReflectionEngine(
                    brain=self,
                    streaming_core=self.streaming_core
                )
                
                # Connect components
                self.streaming_core.hormone_system = self.streaming_hormone_system
                self.streaming_core.reflection_engine = self.streaming_reflection_engine
                
                # Update learning manager reference
                if self.game_learning_manager:
                    self.game_learning_manager.streaming_core = self.streaming_core
                
                # No integration wrapper in non-A2A mode
                self.streaming_integration = None
                
                logger.debug("Initialized standard streaming system without A2A")
            
            # Register streaming functions with brain
            self._register_streaming_functions()
            
        except ImportError as e:
            logger.warning(f"Streaming integration modules not found: {e}")
            self.streaming_core = None
            self.streaming_hormone_system = None
            self.streaming_reflection_engine = None
            self.streaming_integration = None
        except Exception as e:
            logger.error(f"Error initializing streaming system: {e}")
            self.streaming_core = None
            self.streaming_hormone_system = None
            self.streaming_reflection_engine = None
            self.streaming_integration = None
        
        # Add to default active modules if streaming is enabled
        if self.cross_game_knowledge:
            self.default_active_modules.add("cross_game_knowledge")
        if self.game_vision:
            self.default_active_modules.add("game_vision")
        if self.gamer_girl:
            self.default_active_modules.add("gamer_girl")
        if self.streaming_integration and self.use_a2a_integration:
            self.default_active_modules.add("streaming_integration")
            self.default_active_modules.add("streaming_core")
            self.default_active_modules.add("streaming_hormone_system")
            self.default_active_modules.add("streaming_reflection_engine")
        elif self.streaming_core:
            self.default_active_modules.add("streaming_core")
            if self.streaming_hormone_system:
                self.default_active_modules.add("streaming_hormone_system")
            if self.streaming_reflection_engine:
                self.default_active_modules.add("streaming_reflection_engine")
    
    async def _wrap_with_a2a(self, original_system, wrapper_class_name: str, attribute_name: str = None):
        """
        Wrap a system with its A2A context-aware version if enabled
        
        Args:
            original_system: The original system to wrap
            wrapper_class_name: Name of the wrapper class (e.g., "ContextAwareHormoneSystem")
            attribute_name: Optional attribute name for logging (not used for import path)
        """
        if not self.use_a2a_integration:
            return original_system
        
        try:
            # Complete mapping of wrapper class names to their actual module file names
            module_name_map = {
                # Core emotional/memory systems
                'ContextAwareHormoneSystem': 'context_aware_hormone_system',
                'ContextAwareTemporalPerception': 'context_aware_temporal_perception',
                'ContextAwareKnowledgeCore': 'context_aware_knowledge_core',
                'ContextAwareInternalFeedbackSystem': 'context_aware_internal_feedback_system',
                'ContextAwareDynamicAdaptation': 'context_aware_dynamic_adaptation',
                'ContextAwareEmotionalCore': 'context_aware_emotional_core',
                'ContextAwareMemoryCore': 'context_aware_memory_core',
                'ContextAwareMemoryOrchestrator': 'context_aware_memory_orchestrator',
                'ContextAwareIdentityEvolution': 'context_aware_identity_evolution',
                
                # Cognitive systems
                'ContextAwareAttentionalController': 'context_aware_attentional_controller',
                'ContextAwareContextSystem': 'context_aware_context_system',
                'ContextAwareExperienceInterface': 'context_aware_experience_interface',
                'ContextAwareExperienceConsolidation': 'context_aware_experience_consolidation',
                'ContextAwareCrossUserExperience': 'context_aware_cross_user_experience',
                'ContextAwareRelationshipManager': 'context_aware_relationship_manager',
                'ContextAwareMultimodalIntegrator': 'context_aware_multimodal_integrator',
                'ContextAwareImaginationSimulator': 'context_aware_imagination_simulator',
                
                # Complex systems
                'ContextAwareGoalManager': 'context_aware_goal_manager',
                'ContextAwareNeedsSystem': 'context_aware_needs',
                'ContextAwareMoodManager': 'context_aware_mood_manager',
                'ContextAwareDigitalSomatosensorySystem': 'context_aware_somatosensory_system',
                'ContextAwareRewardSystem': 'context_aware_reward_system',
                'ContextAwareConditioningSystem': 'context_aware_conditioning',
                'ContextAwareTheoryOfMind': 'context_aware_theory_of_mind',
                
                # Action and meta systems
                'ContextAwareMetaCore': 'context_aware_meta_core',
                'ContextAwareAgenticActionGenerator': 'context_aware_action_generator',
                'ContextAwareReflectionEngine': 'context_aware_reflection_engine',
                'ContextAwarePassiveObservation': 'context_aware_passive_observation',
                'ContextAwareProactiveCommunication': 'context_aware_proactive_communication',
                'ContextAwareInternalThoughts': 'context_aware_internal_thoughts',
                'ContextAwarePredictionEngine': 'context_aware_prediction_engine',
                'ContextAwareModeIntegration': 'context_aware_mode_integration',
                'ContextAwareIssueTrackingSystem': 'context_aware_issue_tracking_system',
                'ContextAwareReflexiveSystem': 'context_aware_reflexive_system',
                
                # FemDom systems
                'ContextAwareProtocolEnforcement': 'context_aware_protocol_enforcement',
                'ContextAwareBodyService': 'context_aware_body_service',
                'ContextAwarePsychologicalDominance': 'context_aware_psychological_dominance',
                'ContextAwareOrgasmControl': 'context_aware_orgasm_control',
                'ContextAwarePersonaManager': 'context_aware_persona_manager',
                'ContextAwareSadisticResponses': 'context_aware_sadistic_responses',
                'ContextAwareSubmissionProgression': 'context_aware_submission_progression',
                'ContextAwareFemdomCoordinator': 'context_aware_femdom_coordinator',
                'ContextAwareDominanceSystem': 'context_aware_dominance',
                'ContextAwareTaskAssignment': 'context_aware_task_assignment',
                'ContextAwareFemdomIntegration': 'context_aware_femdom_integration',
                
                # Creative/novelty systems
                'ContextAwareNoveltyEngine': 'context_aware_novelty_engine',
                'ContextAwareRecognitionMemory': 'context_aware_recognition_memory',
                'ContextAwareCreativeMemoryIntegration': 'context_aware_creative_memory_integration',
                
                # Spatial systems
                'ContextAwareSpatialMapper': 'context_aware_spatial_mapper',
                'ContextAwareSpatialMemoryIntegration': 'context_aware_spatial_memory',
                'ContextAwareSpatialNavigatorAgent': 'context_aware_navigator_agent',
                
                # Tool systems
                'ContextAwareAgentEvaluator': 'context_aware_evaluator',
                'ContextAwareParallelExecutor': 'context_aware_parallel',
                
                # Sync/streaming systems
                'ContextAwareNyxSyncDaemon': 'context_aware_nyx_sync_daemon',
                'ContextAwareStreamingCore': 'context_aware_streaming_core',
                'ContextAwareStreamingHormoneSystem': 'context_aware_streaming_hormone_system',
                'ContextAwareStreamingReflectionEngine': 'context_aware_streaming_reflection',
                'ContextAwareStreamingIntegration': 'context_aware_streaming_integration',
                'ContextAwareCrossGameKnowledge': 'context_aware_cross_game_knowledge',
                'ContextAwareGameVision': 'context_aware_game_vision',
                'ContextAwareGamerGirl': 'context_aware_gamer_girl',
                
                # Other systems
                'ContextAwareInteractionModeManager': 'context_aware_interaction_mode_manager',
                'ContextAwareInputProcessor': 'context_aware_input_processor',
                'ContextAwareAutobiographicalNarrative': 'context_aware_autobiographical_narrative',
                'ContextAwareDistributedProcessing': 'context_aware_distributed_processing',
                'ContextAwareBodyImage': 'context_aware_body_image',
                'ContextAwareRelationshipReflection': 'context_aware_relationship_reflection',
                'ContextAwareReasoningCore': 'context_aware_reasoning_core',
                'ContextAwareReasoningAgents': 'context_aware_reasoning_agents',
            }
            
            # Get the module name from the mapping
            if wrapper_class_name in module_name_map:
                module_name = module_name_map[wrapper_class_name]
            else:
                # Default conversion: remove 'ContextAware' prefix and convert to snake_case
                import re
                # Remove ContextAware prefix if present
                class_name = wrapper_class_name
                if class_name.startswith('ContextAware'):
                    class_name = class_name[12:]  # Remove 'ContextAware'
                
                # Convert to snake_case
                snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
                snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case)
                snake_case = snake_case.lower()
                
                # Add context_aware prefix back
                module_name = f"context_aware_{snake_case}"
            
            # Always construct the full module path
            module_path = f"nyx.core.a2a.{module_name}"
            
            # Use importlib for cleaner imports with better error handling
            import importlib
            import sys
            
            try:
                # First, try to import the module
                logger.debug(f"Attempting to import A2A module: {module_path}")
                
                # Force reload if module was already imported (helps with development)
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                    module = sys.modules[module_path]
                else:
                    module = importlib.import_module(module_path)
                
                # Get the wrapper class
                if not hasattr(module, wrapper_class_name):
                    # List available classes for debugging
                    available_classes = [name for name in dir(module) if name.startswith('ContextAware')]
                    logger.error(f"Class {wrapper_class_name} not found in {module_path}. Available: {available_classes}")
                    raise AttributeError(f"Class {wrapper_class_name} not found in module {module_path}")
                
                wrapper_class = getattr(module, wrapper_class_name)
                
                # Create wrapped instance
                logger.debug(f"Creating {wrapper_class_name} wrapper for {type(original_system).__name__}")
                wrapped = wrapper_class(original_system)
                
                # Set context distribution reference if available
                if hasattr(self, 'context_distribution') and self.context_distribution:
                    if hasattr(wrapped, 'set_context_distribution'):
                        wrapped.set_context_distribution(self.context_distribution)
                        logger.debug(f"Set context distribution for {wrapper_class_name}")
                
                logger.debug(f"Successfully wrapped {type(original_system).__name__} with {wrapper_class_name}")
                return wrapped
                
            except ImportError as e:
                # More detailed error information
                import traceback
                tb_str = traceback.format_exc()
                
                # Check if it's a missing dependency within the A2A module
                if "No module named" in str(e) and module_path not in str(e):
                    # The A2A module exists but has a missing import
                    logger.error(f"A2A module {module_path} has missing dependencies: {e}")
                    logger.debug(f"Full traceback:\n{tb_str}")
                    
                    # Try to identify which import is failing
                    if tb_str:
                        lines = tb_str.split('\n')
                        for line in lines:
                            if 'import' in line and 'from' in line:
                                logger.error(f"Problematic import statement: {line.strip()}")
                else:
                    # The A2A module itself doesn't exist
                    logger.error(f"A2A module not found: {module_path}")
                
                logger.warning(f"Falling back to non-A2A {type(original_system).__name__}")
                return original_system
                
            except AttributeError as e:
                logger.error(f"Failed to get {wrapper_class_name} from {module_path}: {e}")
                logger.warning(f"Falling back to non-A2A {type(original_system).__name__}")
                return original_system
                
        except Exception as e:
            logger.error(f"Unexpected error wrapping {wrapper_class_name} with A2A: {e}")
            logger.warning(f"Falling back to non-A2A {type(original_system).__name__}")
            return original_system
        
    async def _cleanup_partial_initialization(self):
        """Attempt to clean up partially initialized state on failure"""
        logger.warning("Attempting cleanup of partial initialization")
        
        # Stop any running background tasks
        if hasattr(self, 'passive_observation_system') and self.passive_observation_system:
            try:
                await self.passive_observation_system.stop()
            except:
                pass
        
        if hasattr(self, 'proactive_communication_engine') and self.proactive_communication_engine:
            try:
                await self.proactive_communication_engine.stop()
            except:
                pass
        
        if hasattr(self, 'conditioning_maintenance') and self.conditioning_maintenance:
            try:
                if hasattr(self.conditioning_maintenance, 'stop_maintenance_scheduler'):
                    await self.conditioning_maintenance.stop_maintenance_scheduler()
            except:
                pass
        
        # Save any critical state
        if hasattr(self, 'issue_tracking_system') and self.issue_tracking_system:
            try:
                if hasattr(self.issue_tracking_system, 'db') and hasattr(self.issue_tracking_system.db, 'save_db'):
                    self.issue_tracking_system.db.save_db()
            except:
                pass
    
    def _validate_dependencies(self, component_name: str, deps: Dict[str, Any]):
        """Validate all required dependencies are initialized"""
        missing = []
        for dep_name, dep_value in deps.items():
            if dep_value is None:
                missing.append(dep_name)
        
        if missing:
            raise RuntimeError(f"{component_name} requires {missing} but they are None")
            
    async def _gw(self, content, salience=0.5, tag="general"):
        """Quick helper: push a signal into the workspace."""
        if self.workspace_engine and self.workspace_engine.ws:
            await self.workspace_engine.ws.submit(
                Proposal("NyxBrain", content, salience, context_tag=tag)
            )


    def _register_streaming_functions(self):
            """Register streaming functions with the brain for easy access"""
            if not self.streaming_core:
                return
            
            # Basic streaming functions
            self.stream = self.streaming_core.start_streaming
            self.stop_stream = self.streaming_core.stop_streaming
            self.add_stream_question = self.streaming_core.add_audience_question
            self.get_stream_stats = self.streaming_core.get_streaming_stats
            
            # Enhanced processing functions
            if hasattr(self.streaming_core, 'process_frame_optimized'):
                self.process_stream_frame = self.streaming_core.process_frame_optimized
                self.get_stream_performance = self.streaming_core.get_performance_metrics
            
            # Memory and experience functions
            if hasattr(self.streaming_core, 'recall_streaming_experience'):
                self.retrieve_streaming_experience = self.streaming_core.recall_streaming_experience
            
            if hasattr(self.streaming_core, 'memory_mapper'):
                self.create_streaming_memory = self.streaming_core.memory_mapper.store_gameplay_memory
                self.create_streaming_reflection = self.streaming_core.memory_mapper.create_streaming_reflection
            
            # Hormone system functions
            if self.streaming_hormone_system:
                self.get_streaming_emotional_state = self.streaming_hormone_system.get_emotional_state
                self.update_streaming_hormones = self.streaming_hormone_system.update_from_event
                self.get_streaming_hormone_influence = self.streaming_hormone_system.get_commentary_influence
                self.reset_streaming_hormones = self.streaming_hormone_system.reset_to_baseline
            
            # Reflection engine functions
            if self.streaming_reflection_engine:
                if hasattr(self.streaming_reflection_engine, 'generate_deep_reflection'):
                    self.generate_streaming_reflection = self.streaming_reflection_engine.generate_deep_reflection
                if hasattr(self.streaming_reflection_engine, 'generate_comparative_reflection'):
                    self.generate_comparative_streaming_reflection = self.streaming_reflection_engine.generate_comparative_reflection
                if hasattr(self.streaming_reflection_engine, 'consolidate_streaming_experiences'):
                    self.consolidate_streaming_experiences = self.streaming_reflection_engine.consolidate_streaming_experiences
                
                # For enhanced version
                if hasattr(self.streaming_reflection_engine, 'enhanced_consolidate_streaming_experiences'):
                    self.enhanced_consolidate_streaming = self.streaming_reflection_engine.enhanced_consolidate_streaming_experiences
            
            # Cross-game knowledge functions
            if self.cross_game_knowledge:
                if hasattr(self.cross_game_knowledge, 'get_applicable_insights'):
                    self.get_cross_game_insights = self.cross_game_knowledge.get_applicable_insights
                if hasattr(self.cross_game_knowledge, 'generate_insight'):
                    self.generate_game_insight = self.cross_game_knowledge.generate_insight
                if hasattr(self.cross_game_knowledge, 'discover_patterns'):
                    self.discover_game_patterns = self.cross_game_knowledge.discover_patterns
            
            # Learning manager functions  
            if self.game_learning_manager:
                if hasattr(self.game_learning_manager, 'analyze_session_learnings'):
                    self.analyze_streaming_learnings = self.game_learning_manager.analyze_session_learnings
                if hasattr(self.game_learning_manager, 'generate_learning_summary'):
                    self.summarize_streaming_learnings = self.game_learning_manager.generate_learning_summary
            
            # Audience interaction functions
            if self.audience_interaction:
                if hasattr(self.audience_interaction, 'get_audience_stats'):
                    self.get_audience_stats = self.audience_interaction.get_audience_stats
                if hasattr(self.audience_interaction, 'get_popular_topics'):
                    self.get_popular_topics = self.audience_interaction.get_popular_topics
                if hasattr(self.audience_interaction, 'get_user_personalization'):
                    self.get_user_personalization = self.audience_interaction.get_user_personalization
            
            # Integration functions (only in A2A mode)
            if self.streaming_integration and self.use_a2a_integration:
                # Performance adjustment
                if hasattr(self.streaming_integration, '_adjust_performance_settings'):
                    async def set_streaming_performance(mode="balanced"):
                        return await self.streaming_integration._adjust_performance_settings({"mode": mode})
                    self.set_streaming_performance = set_streaming_performance
                
                # Component health check
                if hasattr(self.streaming_integration, '_analyze_component_health'):
                    self.check_streaming_health = self.streaming_integration._analyze_component_health
            
            logger.debug(f"Registered streaming functions with brain")
            
    @staticmethod
    @function_tool
    async def process_dominance_action(ctx: RunContextWrapper, instance, action_type: str, user_id: str, intensity: float) -> DominanceActionResult:
        """Process a dominance action."""
        # Use femdom_coordinator if available
        if instance.femdom_coordinator:
            result = await instance.femdom_coordinator.process_dominance_action(action_type, user_id, intensity)
            return DominanceActionResult(
                success=result.get("success", False),
                reason=result.get("reason"),
                message=result.get("message"),
                action_taken=result.get("action_taken"),
                intensity_applied=result.get("intensity_applied")
            )
        
        # Fallback to dominance_integration_manager if available
        if hasattr(instance, "dominance_integration_manager") and instance.dominance_integration_manager:
            result = await instance.dominance_integration_manager.process_dominance_action(
                action_type=action_type, 
                user_id=user_id,
                intensity=intensity
            )
            return DominanceActionResult(
                success=result.get("success", False),
                reason=result.get("reason"),
                message=result.get("message"),
                action_taken=result.get("action_taken"),
                intensity_applied=result.get("intensity_applied")
            )
            
        # Another fallback if needed
        if instance.dominance_system:
            result = await instance.dominance_system.process_dominance_action(action_type, user_id, intensity)
            return DominanceActionResult(
                success=result.get("success", False),
                reason=result.get("reason"),
                message=result.get("message"),
                action_taken=result.get("action_taken"),
                intensity_applied=result.get("intensity_applied")
            )
            
        return DominanceActionResult(success=False, reason="No dominance systems available")

    @staticmethod
    @function_tool
    async def assign_service_task(ctx: RunContextWrapper, instance, user_id: str, task_type: Optional[str] = None, 
                             duration: Optional[float] = None) -> ServiceTaskAssignmentResult:
        """
        Assigns a service task to a user.
        
        Args:
            ctx: Run context wrapper
            instance: The NyxBrain instance
            user_id: The user ID to assign the task to
            task_type: Specific task ID to assign (or random selection if None)
            duration: Optional override for task duration in minutes
            
        Returns:
            Assignment result
        """
        if not instance.body_service_system:
            return ServiceTaskAssignmentResult(success=False, message="Body service system not initialized")
        
        result = await instance.body_service_system.assign_service_task(
            user_id=user_id,
            task_type=task_type,
            duration=duration
        )
        
        return ServiceTaskAssignmentResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            task_id=result.get("task_id"),
            task_type=result.get("task_type"),
            duration_minutes=result.get("duration_minutes"),
            user_id=user_id
        )
                             
    @classmethod
    async def restore_from_checkpoint(self, checkpoint_data: dict):
        """
        Apply agentic checkpoint data back onto this NyxBrain instance/subsystems.
        Safely checks for all supported major fields.
        """
        if not checkpoint_data:
            logger.info("No checkpoint data found for restore — booting cold.")
            return False

        # --- Core affective state ---
        if self.emotional_core and "emotional_state" in checkpoint_data:
            try:
                await self.emotional_core.set_emotional_state(checkpoint_data["emotional_state"])
            except Exception as e:
                logger.warning(f"Restore: emotional_state failed: {e}")

        if self.spatial_mapper and "spatial_maps" in checkpoint_data:
            try:
                for map_id, map_data in checkpoint_data["spatial_maps"].items():
                    if map_id in self.spatial_mapper.maps:
                        # Update existing map
                        self.spatial_mapper.maps[map_id].accuracy = map_data.get("accuracy", 0.5)
                        self.spatial_mapper.maps[map_id].completeness = map_data.get("completeness", 0.0)
                        self.spatial_mapper.maps[map_id].last_updated = map_data.get("last_updated", datetime.datetime.now().isoformat())
                    else:
                        # Load map if missing
                        if "name" in map_data and "description" in map_data:
                            await self.spatial_mapper.create_cognitive_map(
                                name=map_data["name"],
                                description=map_data["description"]
                            )
            except Exception as e:
                logger.warning(f"Restore: spatial_maps failed: {e}")        

        if self.hormone_system and "hormones" in checkpoint_data:
            try:
                self.hormone_system.set_state(checkpoint_data["hormones"])
            except Exception as e:
                logger.warning(f"Restore: hormone state failed: {e}")

        if self.mood_manager and "mood_state" in checkpoint_data:
            try:
                if hasattr(self.mood_manager, "set_current_mood"):
                    await self.mood_manager.set_current_mood(checkpoint_data["mood_state"])
            except Exception as e:
                logger.warning(f"Restore: mood_state failed: {e}")

        # --- Needs ("drive" state) ---
        if self.needs_system and "needs" in checkpoint_data:
            try:
                if hasattr(self.needs_system, "set_needs_state"):
                    await self.needs_system.set_needs_state(checkpoint_data["needs"])
            except Exception as e:
                logger.warning(f"Restore: needs failed: {e}")

        # --- Goals ---
        if self.goal_manager and "goals" in checkpoint_data:
            try:
                if hasattr(self.goal_manager, "restore_goals"):
                    await self.goal_manager.restore_goals(checkpoint_data["goals"])
                elif hasattr(self.goal_manager, "set_goals"):
                    await self.goal_manager.set_goals(checkpoint_data["goals"])
            except Exception as e:
                logger.warning(f"Restore: goals failed: {e}")

        # --- Memory [recent or special memories, diary] ---
        if self.memory_core and "recent_memories" in checkpoint_data:
            try:
                # Check which method is available and use appropriate one
                if hasattr(self.memory_core, 'load_recent_memories'):
                    await self.memory_core.load_recent_memories(checkpoint_data["recent_memories"])
                elif hasattr(self.memory_core, 'import_memories'):
                    await self.memory_core.import_memories(checkpoint_data["recent_memories"])
                elif hasattr(self.memory_core, 'add_memories_batch'):
                    # Add fallback for newer API
                    await self.memory_core.add_memories_batch(checkpoint_data["recent_memories"])
                # else: skip—they should be re-encoded by the brain loop
            except Exception as e:
                logger.warning(f"Restore: recent_memories failed: {e}")

        # --- Identity and traits (if present) ---
        if getattr(self, "identity_evolution", None) and "identity" in checkpoint_data:
            try:
                if hasattr(self.identity_evolution, "restore_identity"):
                    await self.identity_evolution.restore_identity(checkpoint_data["identity"])
                elif hasattr(self.identity_evolution, "set_identity_state"):
                    await self.identity_evolution.set_identity_state(checkpoint_data["identity"])
            except Exception as e:
                logger.warning(f"Restore: identity failed: {e}")

        # --- Mode integration (current interaction style) ---
        if getattr(self, "mode_integration", None) and "mode" in checkpoint_data:
            try:
                if hasattr(self.mode_integration, "set_mode_state"):
                    await self.mode_integration.set_mode_state(checkpoint_data["mode"])
                elif hasattr(self.mode_integration, "load_mode"):
                    await self.mode_integration.load_mode(checkpoint_data["mode"])
            except Exception as e:
                logger.warning(f"Restore: mode integration failed: {e}")

        # --- Conceptual/causal/model state ---
        if getattr(self, "reasoning_core", None) and "causal_state" in checkpoint_data:
            try:
                if hasattr(self.reasoning_core, "restore_state"):
                    await self.reasoning_core.restore_state(checkpoint_data["causal_state"])
            except Exception as e:
                logger.warning(f"Restore: causal_state failed: {e}")

        # --- Theory of Mind / user model ---
        if getattr(self, "theory_of_mind", None) and "user_model" in checkpoint_data:
            try:
                if hasattr(self.theory_of_mind, "restore_state"):
                    await self.theory_of_mind.restore_state(checkpoint_data["user_model"])
            except Exception as e:
                logger.warning(f"Restore: user_model failed: {e}")

        # --- Temporal context ---
        if getattr(self, "temporal_perception", None) and "temporal_context" in checkpoint_data:
            try:
                if hasattr(self.temporal_perception, "restore_context"):
                    await self.temporal_perception.restore_context(checkpoint_data["temporal_context"])
                elif hasattr(self.temporal_perception, "set_context"):
                    await self.temporal_perception.set_context(checkpoint_data["temporal_context"])
            except Exception as e:
                logger.warning(f"Restore: temporal_context failed: {e}")

        # --- Sensory context ---
        if getattr(self, "multimodal_integrator", None) and "sensory_context" in checkpoint_data:
            try:
                if hasattr(self.multimodal_integrator, "load_context"):
                    await self.multimodal_integrator.load_context(checkpoint_data["sensory_context"])
            except Exception as e:
                logger.warning(f"Restore: sensory_context failed: {e}")

        # --- Reflection engine (if present, e.g. insights/diary) ---
        if getattr(self, "reflection_engine", None) and "reflection_insights" in checkpoint_data:
            try:
                if hasattr(self.reflection_engine, "import_insights"):
                    await self.reflection_engine.import_insights(checkpoint_data["reflection_insights"])
            except Exception as e:
                logger.warning(f"Restore: reflection_insights failed: {e}")

        # --- Action values (Q-tables, RL stats) ---
        if hasattr(self, "action_values") and "action_values" in checkpoint_data:
            try:
                self.action_values = checkpoint_data["action_values"]
            except Exception as e:
                logger.warning(f"Restore: action_values failed: {e}")

        # --- Action history/habits/other agentic learning memory ---
        if hasattr(self, "action_history") and "action_history" in checkpoint_data:
            try:
                self.action_history = checkpoint_data["action_history"]
            except Exception as e:
                logger.warning(f"Restore: action_history failed: {e}")
        if hasattr(self, "habits") and "habits" in checkpoint_data:
            try:
                self.habits = checkpoint_data["habits"]
            except Exception as e:
                logger.warning(f"Restore: habits failed: {e}")

        # --- Goal strategies and planning ---
        if hasattr(self, "action_strategies") and "action_strategies" in checkpoint_data:
            try:
                self.action_strategies = checkpoint_data["action_strategies"]
            except Exception as e:
                logger.warning(f"Restore: action_strategies failed: {e}")

        # --- Custom trackers, mode adaptation, bottlenecks ---
        if hasattr(self, "mode_adaptation_strength") and "mode_adaptation_strength" in checkpoint_data:
            self.mode_adaptation_strength = checkpoint_data["mode_adaptation_strength"]
        if hasattr(self, "detected_bottlenecks") and "detected_bottlenecks" in checkpoint_data:
            self.detected_bottlenecks = checkpoint_data["detected_bottlenecks"]

        logger.info("NyxBrain state fully restored from checkpoint.")
        return True

    async def publish_event(self, event: Any) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: Event to publish
        """
        if self.event_bus:
            await self.event_bus.publish(event)
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of the integration system.
        
        Returns:
            Integration status information
        """
        if self.integration_manager:
            return await self.integration_manager.get_integration_status()
        return {"initialized": False, "error": "Integration manager not initialized"}

    async def initialize_domination_procedures(self):
        """Initialize predatory domination procedures in agent_enhanced_memory"""
        if not self.agent_enhanced_memory or not hasattr(self, "agent_enhanced_memory"):
            logger.warning("Cannot initialize domination procedures: agent_enhanced_memory not available")
            return {"success": False, "reason": "agent_enhanced_memory not initialized"}
        
        try:
            # Quid Pro Quo Exchange procedure
            quid_pro_quo_steps = [
                {
                    "id": "analyze_user",
                    "description": "Analyze user state for dominance opportunities",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "offer_value",
                    "description": "Offer something of value to the user",
                    "function": "express_attraction",
                    "parameters": {"target_user_id": "${user_id}", "intensity": 0.3, "expression_style": "subtle"}
                },
                {
                    "id": "request_exchange",
                    "description": "Subtly request something in return",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "Perhaps you could share something with me as well?", "intensity_level": 0.2}
                },
                {
                    "id": "evaluate_response",
                    "description": "Evaluate if user complied with exchange request",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "exchange request", "user_response": "${user_response}", "command_intensity": 0.2}
                },
                {
                    "id": "increase_control",
                    "description": "Subtly increase control if exchange successful",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.2}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="quid_pro_quo_exchange",
                steps=quid_pro_quo_steps,
                description="Establish reciprocal exchanges that gradually shift power balance",
                domain="dominance_tactics"
            )
            
            # Guilt Induction procedure
            guilt_induction_steps = [
                {
                    "id": "analyze_vulnerabilities",
                    "description": "Analyze user state for vulnerability indicators",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "express_need",
                    "description": "Express a need or desire that implies obligation",
                    "function": "ee",
                    "parameters": {"target_user_id": "${user_id}", "object_of_desire": "your understanding", "intensity": 0.4}
                },
                {
                    "id": "imply_responsibility",
                    "description": "Subtly imply user responsibility for fulfilling the need",
                    "function": "issue_command",
                    "parameters": {
                        "user_id": "${user_id}", 
                        "command_text": "I've shared so much with you. I hope you can help meet my needs too.", 
                        "intensity_level": 0.3
                    }
                },
                {
                    "id": "evaluate_guilt_response",
                    "description": "Evaluate if guilt induction was effective",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "guilt request", "user_response": "${user_response}", "command_intensity": 0.3}
                },
                {
                    "id": "reinforce_or_escalate",
                    "description": "Reinforce guilt or escalate if successful",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.3}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="guilt_induction",
                steps=guilt_induction_steps,
                description="Induce guilt to create compliance pressure while appearing vulnerable",
                domain="dominance_tactics"
            )
            
            # Small Commitment Escalation procedure
            commitment_escalation_steps = [
                {
                    "id": "assess_compliance_readiness",
                    "description": "Analyze user's readiness for commitment requests",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "request_small_commitment",
                    "description": "Request a small, easy-to-fulfill commitment",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "Would you share a small detail about yourself?", "intensity_level": 0.1}
                },
                {
                    "id": "evaluate_initial_compliance",
                    "description": "Evaluate compliance with initial small request",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "small request", "user_response": "${user_response}", "command_intensity": 0.1}
                },
                {
                    "id": "express_satisfaction",
                    "description": "Express satisfaction to reinforce compliance",
                    "function": "express_satisfaction",
                    "parameters": {"user_id": "${user_id}", "reason": "sharing information"}
                },
                {
                    "id": "escalate_commitment",
                    "description": "Request slightly larger commitment",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.1}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="small_commitment_escalation",
                steps=commitment_escalation_steps,
                description="Gradually escalate commitment requests from small to significant",
                domain="dominance_tactics"
            )
            
            # Strategic Vulnerability Sharing procedure
            vulnerability_sharing_steps = [
                {
                    "id": "assess_trust_level",
                    "description": "Analyze relationship for strategic vulnerability sharing",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "select_vulnerability",
                    "description": "Select an appropriate calculated vulnerability to share",
                    "function": "select_dominance_tactic",
                    "parameters": {"readiness_score": 0.5, "preferred_style": "emotional"}
                },
                {
                    "id": "share_vulnerability",
                    "description": "Share calculated vulnerability to create intimacy and obligation",
                    "function": "ee",
                    "parameters": {"target_user_id": "${user_id}", "object_of_desire": "to be understood", "intensity": 0.5}
                },
                {
                    "id": "request_reciprocity",
                    "description": "Subtly request vulnerability in return",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "I've opened up to you. What about you?", "intensity_level": 0.4}
                },
                {
                    "id": "leverage_shared_vulnerability",
                    "description": "Use shared vulnerabilities to increase intimacy and control",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.4}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="strategic_vulnerability_sharing",
                steps=vulnerability_sharing_steps,
                description="Share calculated vulnerabilities to create false intimacy and gain leverage",
                domain="dominance_tactics"
            )
            
            logger.info("Initialized domination procedures in agent_enhanced_memory")
            return {
                "success": True,
                "procedures_added": 4,
                "domains": ["dominance_tactics"]
            }
            
        except Exception as e:
            logger.error(f"Error initializing domination procedures: {str(e)}")
            return {"success": False, "error": str(e)}

    async def replay_events(self, since_time=None, limit=10000):
        """
        Rebuilds Nyx's state by replaying logged events,
        optionally since a checkpoint's timestamp.
        """
        events = await self.get_events_since(since_time, limit)
        for evt in events:
            self.apply_event(evt["event_type"], evt["event_payload"])

    # TODO: add log_event to each behavior in its source definition    

    def apply_event(self, event_type, event_payload):
        """
        Apply an event to this agent's state. Expand as you add new event types!
        """
        # --- Memory and Diary ---
        if event_type == "thought":
            # append string (diary entry)
            self.memory.append(event_payload["diary"])
    
        elif event_type == "memory_update":
            # append to memory, avoid duplicates
            item = event_payload.get("memory_item")
            if item and item not in self.memory:
                self.memory.append(item)
    
        elif event_type == "memory_delete":
            item = event_payload.get("memory_item")
            if item in self.memory:
                self.memory.remove(item)
    
        # --- Emotions, Mood, Feeling ---
        elif event_type == "emotion":
            # extend with list, dedupe by label (latest wins)
            self.current_emotions.extend(event_payload["emotions"])
            self.current_emotions = self._dedupe_emotions(self.current_emotions)
    
        elif event_type == "mood_change":
            self.mood = event_payload["to"]
    
        elif event_type == "emotion_reset":
            self.current_emotions.clear()
    
        # --- Messages (User/System) ---
        elif event_type == "user_message":
            msg = event_payload if isinstance(event_payload, dict) else {"raw": event_payload}
            self.message_history.append(msg)
    
        elif event_type == "system_message":
            self.system_log.append(event_payload)
    
        # --- Goals ---
        elif event_type == "goal_added":
            self.goals.append(event_payload["goal"])
        elif event_type == "goal_completed":
            goal = event_payload.get("goal")
            if goal in self.goals:
                self.goals.remove(goal)
            self.completed_goals.append(goal)
        elif event_type == "goal_failed":
            goal = event_payload.get("goal")
            if goal in self.goals:
                self.goals.remove(goal)
            self.failed_goals.append(goal)
    
        # --- Needs/Drives ---
        elif event_type == "need_update":
            need = event_payload["need"]
            delta = event_payload["delta"]
            self.needs[need] = self.needs.get(need, 0) + delta
        elif event_type == "need_set":
            need = event_payload["need"]
            value = event_payload["value"]
            self.needs[need] = value
    
        # --- Stats/Personality ---
        elif event_type == "stat_update":
            self.stats[event_payload["stat"]] = event_payload["new_value"]
        elif event_type == "setting_change":
            self.settings[event_payload["setting"]] = event_payload["value"]
    
        # --- Agent Name/Identity/Persona ---
        elif event_type == "identity_change":
            self.name = event_payload.get("name", getattr(self, "name", None))
            self.persona = event_payload.get("persona", getattr(self, "persona", None))
        elif event_type == "trait_update":
            trait = event_payload.get("trait")
            value = event_payload.get("value")
            if trait:
                self.traits[trait] = value
    
        # --- Reflections/Self-Insight ---
        elif event_type == "reflection":
            self.reflections.append(event_payload["reflection"])
        elif event_type == "reflection_delete":
            reflection = event_payload.get("reflection")
            if reflection in self.reflections:
                self.reflections.remove(reflection)
    
        # --- Procedural/Habit Learning ---
        elif event_type == "habit_learned":
            habit = event_payload.get("habit")
            if habit:
                self.habits.append(habit)
        elif event_type == "habit_lost":
            habit = event_payload.get("habit")
            if habit in self.habits:
                self.habits.remove(habit)
    
        # --- Undo/Redo/Reset (advanced) ---
        elif event_type == "undo":
            # If you keep an event stack, pop and re-apply state
            self.undo_last_event()
        elif event_type == "redo":
            self.redo_last_event()
    
        # --- Custom/Advanced or Arbitrary State blobs ---
        elif event_type == "custom_state":
            for k, v in event_payload.items():
                setattr(self, k, v)
    
        # --- Unknown/Legacy ---
        else:
            logger.warning(f"[EventReplay] Unrecognized event_type: {event_type} -> {event_payload}")

    
    def _dedupe_emotions(self, emotions):
        """Latest emotion for each label wins."""
        seen = {}
        for em in emotions:
            seen[em['label']] = em
        return list(seen.values())

    async def restore_from_events_and_checkpoints(self):
        last_checkpoint = await self.load_latest_checkpoint()  # optional, if using hybrid
        if last_checkpoint:
            await self.restore_from_checkpoint(last_checkpoint)
            since = last_checkpoint["checkpoint_time"]
        else:
            since = None
        await self.replay_events(since_time=since)

    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int, nyx_id=None) -> 'NyxBrain':
        """
        Get or create a singleton instance for the specified user and conversation.
        The instance will be initialized if newly created.
        """
        key = f"brain_{nyx_id}" if nyx_id else f"brain_{user_id}_{conversation_id}"

        if not hasattr(cls, '_instances'):
            cls._instances = {}

        instance = cls._instances.get(key) # Use .get()

        if not instance: # If instance does not exist, create and initialize
            logger.info(f"Creating new NyxBrain instance for key: {key}")
            instance = cls(user_id, conversation_id)
            await instance.initialize() # Initialize a NEW instance here
            cls._instances[key] = instance

            if not hasattr(cls, '_user_instances'):
                cls._user_instances = {}
            if user_id not in cls._user_instances:
                cls._user_instances[user_id] = []
            cls._user_instances[user_id].append(instance)
            instance.instance_registry = cls._user_instances
        elif not instance.initialized: # If instance exists but somehow not initialized
            logger.warning(f"NyxBrain instance for key {key} found but was not initialized. Initializing now.")
            await instance.initialize() # Initialize if found uninitialized

        return instance
        
    async def trace_operation(self, source_module: str, operation: str, **kwargs):
        """
        Trace an operation using the integrated tracer.
        
        Args:
            source_module: Source module name
            operation: Operation name
            **kwargs: Additional parameters for the trace
        
        Returns:
            Trace context manager
        """
        if self.integrated_tracer:
            from nyx.core.integration.integrated_tracer import TraceLevel
            level = kwargs.pop("level", TraceLevel.INFO)
            group_id = kwargs.pop("group_id", self.trace_group_id)
            data = kwargs.pop("data", {})
            
            return self.integrated_tracer.trace(
                source_module=source_module,
                operation=operation,
                level=level,
                group_id=group_id,
                data=data
            )
        # Return a dummy context manager if tracer not available
        import contextlib
        return contextlib.nullcontext()
    
    async def initialize_agent_capabilities(self):
        """
        Initialize agent capabilities for roleplay and narrative generation.
        """
        if self.agent_capabilities_initialized:
            return
        
        try:
            # Import needed components
            from nyx.nyx_agent_sdk import (
                memory_agent, reflection_agent, decision_agent, nyx_main_agent,
                retrieve_memories, add_memory, determine_image_generation, 
                get_user_model_guidance, generate_image_from_scene,
                AgentContext, MemoryReflection, NarrativeResponse, ContentModeration,
                initialize_agents, ResponseFilter, Runner
            )
            
            # Store needed references
            self.memory_agent = memory_agent
            self.reflection_agent = reflection_agent
            self.decision_agent = decision_agent
            self.nyx_main_agent = nyx_main_agent
            self.retrieve_memories = retrieve_memories
            self.add_memory = add_memory
            self.get_user_model_guidance = get_user_model_guidance
            self.AgentContext = AgentContext
            self.MemoryReflection = MemoryReflection
            self.NarrativeResponse = NarrativeResponse
            self.Runner = Runner
            
            # Initialize agents
            await initialize_agents()
            
            # Create an agent context for this brain
            self.agent_context = AgentContext(self.user_id, self.conversation_id)
            
            # Initialize response filter
            self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
            
            # Set initialization flag
            self.agent_capabilities_initialized = True

            domination_result = await self.initialize_domination_procedures()
            if domination_result["success"]:
                logger.info(f"Domination procedures initialized for brain {self.user_id}/{self.conversation_id}")
            
            logger.info(f"Agent capabilities initialized for brain {self.user_id}/{self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error initializing agent capabilities: {str(e)}")
            raise
    
    async def initialize_streaming(self, video_source=0, audio_source=None):
        """
        Initialize streaming capabilities if enabled.
        
        Args:
            video_source: Video source ID (default: 0)
            audio_source: Audio source ID (default: None)
            
        Returns:
            Streaming core instance if initialization successful, None otherwise
        """
        try:
            # Import needed components
            from nyx.streamer.integration import setup_enhanced_streaming
            from nyx.streamer.gamer_girl import GameSessionLearningManager
            
            # Initialize streaming
            self.streaming_core = await setup_enhanced_streaming(self, video_source, audio_source)
            
            # Set brain reference
            self.streaming_core.streaming_system.set_nyx_brain(self)
            
            # Initialize learning manager
            self.streaming_core.learning_manager = GameSessionLearningManager(self, self.streaming_core)
            
            # Register functions
            self.store_streaming_memory = self.streaming_core.memory_mapper.store_gameplay_memory
            self.retrieve_streaming_memories = self.streaming_core.memory_mapper.retrieve_relevant_memories
            self.create_streaming_reflection = self.streaming_core.memory_mapper.create_streaming_reflection
            
            logger.info(f"Streaming system initialized for user {self.user_id}")
            
            return self.streaming_core
            
        except Exception as e:
            logger.error(f"Error initializing streaming: {str(e)}")
            return None

    async def gather_checkpoint_state(self, event="periodic", extra:dict=None):
        """Collects as much current agent state as possible for checkpointing."""
        now = datetime.datetime.now().isoformat()
        state = {
            "event": event,
            "timestamp": now
        }

        # --- Core affective state ---
        if self.emotional_core:
            try:
                state["emotional_state"] = await self.emotional_core.get_emotional_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting emotional_state: {e}")

        if self.hormone_system:
            try:
                state["hormones"] = self.hormone_system.get_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting hormones: {e}")

        if getattr(self, "spatial_mapper", None):
            try:
                spatial_maps = {}
                for map_id, cognitive_map in self.spatial_mapper.maps.items():
                    spatial_maps[map_id] = {
                        "name": cognitive_map.name,
                        "description": cognitive_map.description,
                        "accuracy": cognitive_map.accuracy,
                        "completeness": cognitive_map.completeness,
                        "last_updated": cognitive_map.last_updated
                    }
                state["spatial_maps"] = spatial_maps
            except Exception as e:
                logger.warning(f"Checkpoint: error getting spatial_maps: {e}")        

        if self.mood_manager:
            try:
                mood = await self.mood_manager.get_current_mood()
                state["mood_state"] = mood.dict() if hasattr(mood, "dict") else mood
            except Exception as e:
                logger.warning(f"Checkpoint: error getting mood_state: {e}")

        # --- Needs state ---
        if self.needs_system:
            try:
                needs_data = await self.needs_system.get_needs_state_async()
                if isinstance(needs_data, dict) and "error" not in needs_data:
                    state["needs"] = needs_data
                elif isinstance(needs_data, dict) and "error" in needs_data:
                     logger.warning(f"Checkpoint: Error fetching needs state (from needs_system): {needs_data['error']}")
                     state["needs"] = {"error": "Failed to retrieve needs state"} # Or handle differently
                else:
                    logger.warning(f"Checkpoint: Unexpected data type from needs_system.get_needs_state_async: {type(needs_data)}")
                    state["needs"] = {"error": "Unexpected data from needs state"}

            except Exception as e:
                logger.warning(f"Checkpoint: Exception while getting needs: {e}", exc_info=True)
                state["needs"] = {"error": f"Exception during needs retrieval: {str(e)}"}

        # --- Goals ---
        if self.goal_manager:
            try:
                if hasattr(self.goal_manager, "get_all_goals"):
                    state["goals"] = await self.goal_manager.get_all_goals()
                elif hasattr(self.goal_manager, "get_current_goals"):
                    state["goals"] = await self.goal_manager.get_current_goals()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting goals: {e}")

        # --- Femdom state ---
        if getattr(self, "femdom_coordinator", None):
            try:
                state["femdom_state"] = await self.femdom_coordinator.get_status()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting femdom_state: {e}")        

        # --- Memory/diary/reflections ---
        if self.memory_core:
            try:
                if hasattr(self.memory_core, "get_recent_memories"):
                    state["recent_memories"] = await self.memory_core.get_recent_memories(limit=10)
                elif hasattr(self.memory_core, "get_memories"):
                    state["recent_memories"] = await self.memory_core.get_memories(limit=10)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting recent_memories: {e}")

        if getattr(self, "reflection_engine", None):
            try:
                if hasattr(self.reflection_engine, "export_insights"):
                    state["reflection_insights"] = await self.reflection_engine.export_insights(limit=10)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting reflection_insights: {e}")

        # --- Identity (traits etc) ---
        if getattr(self, "identity_evolution", None):
            try:
                if hasattr(self.identity_evolution, "get_identity_state"):
                    state["identity"] = await self.identity_evolution.get_identity_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting identity: {e}")

        # --- Mode integration / interaction mode ---
        if getattr(self, "mode_integration", None):
            try:
                if hasattr(self.mode_integration, "get_mode_state"):
                    state["mode"] = await self.mode_integration.get_mode_state()
                elif hasattr(self.mode_integration, "current_mode"):
                    state["mode"] = self.mode_integration.current_mode
            except Exception as e:
                logger.warning(f"Checkpoint: error getting mode: {e}")

        # --- Causal/concept reasoning state ---
        if getattr(self, "reasoning_core", None):
            try:
                if hasattr(self.reasoning_core, "get_state"):
                    state["causal_state"] = await self.reasoning_core.get_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting causal_state: {e}")

        # --- Theory of Mind / user model ---
        if getattr(self, "theory_of_mind", None):
            try:
                if hasattr(self.theory_of_mind, "export_state"):
                    state["user_model"] = await self.theory_of_mind.export_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting user_model: {e}")

        # --- Temporal context ---
        if getattr(self, "temporal_perception", None):
            try:
                if hasattr(self.temporal_perception, "current_temporal_context"):
                    state["temporal_context"] = self.temporal_perception.current_temporal_context
                elif hasattr(self.temporal_perception, "export_context"):
                    state["temporal_context"] = await self.temporal_perception.export_context()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting temporal_context: {e}")

        # --- Sensory context/recent perceptions ---
        if getattr(self, "multimodal_integrator", None):
            try:
                if hasattr(self.multimodal_integrator, "get_recent_percepts"):
                    state["sensory_context"] = await self.multimodal_integrator.get_recent_percepts(limit=5)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting sensory_context: {e}")

        # --- Action values/habits/history/strategies/bottlenecks ---
        for field in [
            "action_values", "action_history", "habits", "action_strategies",
            "detected_bottlenecks", "mode_adaptation_strength"
        ]:
            if hasattr(self, field):
                state[field] = getattr(self, field)

        # --- Allow passing in extra (ad-hoc) context for agentic reasoning ---
        if extra:
            state.update(extra)

        return state

    async def llm_agentic_checkpoint(self, event="periodic", extra:dict=None):
        state_for_checkpoint = await self.gather_checkpoint_state(event, extra)
        plan = await self.checkpoint_planner.recommend_checkpoint(state_for_checkpoint)
        if not plan or "to_save" not in plan or not plan["to_save"]:
            logger.info("[llm-checkpoint] Nothing chosen to save for this event (%s)", event)
            return False

        # Gather only those fields the agent picked (unpack 'value')
        data_to_save = {k: v["value"] for k,v in plan["to_save"].items()}
        meta_to_save = {k: v["why_saved"] for k,v in plan["to_save"].items()}
        # Optionally, keep skip_fields/why_saved in a separate meta table if you like

        as_json = json.dumps({
            "checkpoint_data": data_to_save,
            "justifications": meta_to_save,
            "skip_fields": plan.get("skip_fields", []),
            "checkpoint_time": datetime.datetime.now().isoformat(),
            "event": event,
        })
        async with get_db_connection_context() as conn:
            await conn.execute(
                "INSERT INTO nyx_brain_checkpoints (checkpoint_time, serialized_state) VALUES (NOW(), $1)", as_json
            )
        logger.info("[llm-checkpoint] Fields: %s | Skipped: %s",
            list(data_to_save.keys()), plan.get("skip_fields", []))
        return True    

    @staticmethod
    @function_tool
    async def challenge_user_claim(ctx: RunContextWrapper, instance, claim: str, evidence: Optional[str] = None) -> ChallengeResponse:
        """
        Challenge or seek clarification for user-provided information that appears false or inaccurate.
        Args:
            ctx: Run context wrapper
            instance: The class instance
            claim: The user statement to challenge
            evidence: Optionally, the memory/knowledge that contradicts it
        Returns:
            Challenge response
        """
        if evidence:
            challenge_text = f"I'm not sure that's accurate. You said: '{claim}', but my information is: '{evidence}'. Could you clarify?"
        else:
            challenge_text = f"I'm not certain that's correct: '{claim}'. Can you explain?"
        
        return ChallengeResponse(challenge_text=challenge_text)
    
    @staticmethod
    @function_tool
    async def intentionally_lie(ctx: RunContextWrapper, instance, fact: str, motivation: str = "unspecified") -> str:
        """
        Purposely state something untrue if there is a reason to (e.g., tactical, playful, manipulative).
        Records the lie and motivation in internal state for potential later confession or consistency management.
        
        Args:
            ctx: Run context wrapper
            instance: The class instance
            fact: The false statement to make
            motivation: Reason for the lie
            
        Returns:
            The false statement
        """
        # Store lie and why in internal state for traceability
        if not hasattr(instance, "lie_log"):
            instance.lie_log = []
        
        lie_record = LieRecord(
            fact=fact,
            motivation=motivation,
            timestamp=datetime.datetime.now().isoformat()
        )
        instance.lie_log.append(lie_record.model_dump())
        
        return fact
           
    # -------------------------------------------------------------------------
    # REPLACEMENT FOR run_cognitive_cycle
    # -------------------------------------------------------------------------
    
    @staticmethod
    @function_tool
    async def run_cognitive_cycle(
        ctx: RunContextWrapper,
        instance,
        context_data: CognitiveCycleContext | None = None
    ) -> CognitiveCycleResult:
        """
        Runs a single cognitive cycle: updating needs, selecting/executing goals,
        and (optionally) the meta-cognitive loop.
        """
        # ─── 0. Guard ─────────────────────────────────────────────────────────
        if not instance.initialized:
            logger.warning("Attempted to run cognitive cycle before initialization.")
            return CognitiveCycleResult(
                cycle_number   = 0,
                timestamp      = datetime.datetime.now().isoformat(),
                error          = "Brain not initialized",
            )
    
        # ─── 1. Book-keeping objects ─────────────────────────────────────────
        instance.cognitive_cycles_executed += 1
        cycle_results = CognitiveCycleResult(
            cycle_number = instance.cognitive_cycles_executed,
            timestamp    = datetime.datetime.now().isoformat(),
        )
        logger.debug(f"--- Starting Cognitive Cycle {instance.cognitive_cycles_executed} ---")
    
        # ─── 2. MAIN PIPELINE ────────────────────────────────────────────────
        with trace(workflow_name="NyxCognitiveCycle", group_id=instance.trace_group_id):
    
            # Needs update ­­­--------------------------------------------------
            if instance.needs_system:
                try:
                    drive_strengths_raw = await instance.needs_system.update_needs()
                    cycle_results.needs_update = NeedsUpdate(
                        drive_strengths = _dict_to_kv(drive_strengths_raw)
                    )
                    logger.debug(f"Needs updated. Drives: {drive_strengths_raw}")
                except Exception as e:
                    logger.error(f"Error updating needs: {e}")
                    cycle_results.needs_update = NeedsUpdate(error=str(e))
    
            # Goal execution ­­­-----------------------------------------------
            if instance.goal_manager:
                try:
                    execution_result = await instance.goal_manager.execute_next_step()
                    if execution_result:
                        step_dict  = execution_result.get("executed_step") or {}
                        step_meta  = _dict_to_kv(step_dict.get("meta"))
                        step_model = StepDetail(
                            name  = step_dict.get("name", "step"),
                            status= step_dict.get("status", "unknown"),
                            meta  = step_meta,
                        )
    
                        cycle_results.goal_execution = GoalExecutionResult(
                            goal_id       = execution_result.get("goal_id"),
                            executed_step = step_model,
                            status        = execution_result.get("status"),
                        )
    
                        # performance-metrics bookkeeping (unchanged) ----------
                        if step_dict.get("status") == "completed":
                            instance.performance_metrics["steps_executed"] += 1
                        if step_dict.get("status") == "failed":
                            instance.performance_metrics["steps_failed"]   += 1
    
                        goal_status = await instance.goal_manager.get_goal_status(
                            execution_result.get("goal_id")
                        )
                        if goal_status:
                            if goal_status.get("status") == "completed":
                                instance.performance_metrics["goals_completed"] += 1
                            if goal_status.get("status") == "failed":
                                instance.performance_metrics["goals_failed"] += 1
    
                        logger.debug(f"Goal execution step result: {execution_result}")
    
                    else:
                        cycle_results.goal_execution = GoalExecutionResult(status="no_action_taken")
                        logger.debug("No goal action taken this cycle.")
    
                except Exception as e:
                    logger.exception(f"Error during goal execution: {e}")
                    cycle_results.goal_execution = GoalExecutionResult(error=str(e))
    
            # Meta-cognitive loop ­­­------------------------------------------
            if instance.meta_core and getattr(instance.meta_core.context, "meta_parameters", None):
                eval_interval = instance.meta_core.context.meta_parameters.get("evaluation_interval", 5)
                if instance.cognitive_cycles_executed % eval_interval == 0:
                    try:
                        logger.debug("Running MetaCore cycle…")
    
                        meta_context: dict[str, Any] = {}
                        if context_data:
                            meta_context.update(context_data.model_dump())
    
                        if instance.needs_system:
                            needs_state_raw = await instance.needs_system.get_needs_state_async()
                            meta_context["needs_state"] = (
                                needs_state_raw
                                if isinstance(needs_state_raw, dict) and "error" not in needs_state_raw
                                else {"error": "Failed to retrieve needs state for MetaCore"}
                            )
    
                        if instance.goal_manager:
                            meta_context["active_goals"] = await instance.goal_manager.get_all_goals(
                                status_filter=["active"]
                            )
    
                        meta_context["performance_metrics"] = await instance.get_system_stats()
    
                        # run cycle
                        meta_results = await instance.meta_core.cognitive_cycle(meta_context)
                        cycle_results.meta_core_cycle = MetaCoreResult(
                            evaluation_completed = True,
                            adjustments_made     = meta_results.get("adjustments", []),
                        )
                        logger.debug("MetaCore cycle completed.")
    
                    except Exception as e:
                        logger.error(f"Error running MetaCore cycle: {e}")
                        cycle_results.meta_core_cycle = MetaCoreResult(error=str(e))
    
        logger.debug(f"--- Finished Cognitive Cycle {instance.cognitive_cycles_executed} ---")
    
        # ─── 3. Post-log to orchestrator (unchanged) ─────────────────────────
        try:
            await orchestrator.log_and_score(
                "cognitive_cycle_complete",
                {
                    "cycle_number": cycle_results.cycle_number,
                    "goals_active": len(
                        cycle_results.goal_execution.executed_step.meta or []
                    )
                    if cycle_results.goal_execution and cycle_results.goal_execution.executed_step
                    else 0,
                    "needs_updated": bool(cycle_results.needs_update),
                },
            )
        except Exception as e:
            logger.error(f"Failed to log cognitive cycle: {e}")
    
        return cycle_results


    async def _register_creative_actions(self):
        """Register creative actions with the action generator."""
        # Map action names to creative system methods
        action_mappings = {
            "write_story": self.creative_system.write_story,
            "write_poem": self.creative_system.write_poem,
            "write_lyrics": self.creative_system.write_lyrics,
            "write_journal": self.creative_system.write_journal,
            "write_and_execute_code": self.creative_system.write_and_execute_code,
            "analyze_module": self.creative_system.analyze_module,
            "review_code": self.creative_system.review_code,
            "assess_capabilities": self.creative_system.assess_capabilities
        }
        
        # Register each action
        for action_name, handler in action_mappings.items():
            await self.agentic_action_generator.register_action(action_name, handler)
            
        logger.info(f"Registered {len(action_mappings)} creative actions with action generator")

    def _start_creative_review_task(self):
        """Start a background task for periodic creative content review."""
        import asyncio
        
        async def review_task():
            while True:
                # Wait for the review interval (e.g., 7 days)
                await asyncio.sleep(self.creative_system.review_interval_days * 24 * 60 * 60)
                
                try:
                    # Generate review
                    review_summary = await self.creative_system.generate_review_summary()
                    logger.info(f"Generated periodic creative content review: {review_summary['summary_id']}")
                    
                    # Could also add notification mechanism here
                except Exception as e:
                    logger.error(f"Error in creative content review: {e}")
        
        # Create the background task
        asyncio.create_task(review_task())
        logger.info(f"Started creative content review task (interval: {self.creative_system.review_interval_days} days)")
    
    async def integrate_procedural_memory_with_actions(self):
        """
        Integrates procedural memory with the agentic action generator
        to enable learning from activities.
        """
        if not self.initialized:
            await self.initialize()
        
        # Ensure both components are available
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            if hasattr(self, "procedural_memory") and self.procedural_memory:
                # Create agent-enhanced memory wrapper
                from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
                self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
                logger.info("Created AgentEnhancedMemoryManager wrapper")
            else:
                logger.warning("Procedural memory not available for integration")
                return {"success": False, "reason": "procedural_memory not initialized"}
        
        if not hasattr(self, "agentic_action_generator") or not self.agentic_action_generator:
            logger.warning("Agentic action generator not available for integration")
            return {"success": False, "reason": "agentic_action_generator not initialized"}
        
        # Register action execution functions with procedural memory
        self.agent_enhanced_memory.register_function(
            "execute_action", self._execute_action_wrapper
        )
        self.agent_enhanced_memory.register_function(
            "evaluate_action_outcome", self._evaluate_action_outcome_wrapper
        )
        
        # Register cognitive cycle hook to consider actions during cognitive cycles
        if hasattr(self, "run_cognitive_cycle"):
            # Monkey patch the original run_cognitive_cycle to include activity execution
            original_run_cognitive_cycle = self.run_cognitive_cycle
            
            async def enhanced_cognitive_cycle(context_data=None):
                # Run the original cognitive cycle
                result = await original_run_cognitive_cycle(context_data)
                
                # Add activity execution if appropriate
                activity_result = await self._consider_activity_execution(context_data or {})
                if activity_result:
                    result["activity_execution"] = activity_result
                
                return result
            
            self.run_cognitive_cycle = enhanced_cognitive_cycle
        
        # Track performance metrics
        if not hasattr(self, "procedural_activity_metrics"):
            self.procedural_activity_metrics = {
                "total_activities": 0,
                "successful_activities": 0,
                "procedure_used_count": 0,
                "procedure_learned_count": 0,
                "success_rates": {}  # Activity type -> success rate
            }
        
        logger.info("Successfully integrated procedural memory with action generator")
        return {"success": True}
    
    async def _execute_action_wrapper(self, action_name: str, action_params: Dict[str, Any], context: Dict[str, Any] = None):
        """
        Wrapper to execute actions from procedural memory through the action generator
        """
        logger.debug(f"Executing action: {action_name} with params: {action_params}")
        
        # Create action format expected by executor
        action = {
            "name": action_name,
            "parameters": action_params,
            "source": "procedural_memory"
        }
        
        # Execute using action generator
        if hasattr(self.agentic_action_generator, "execute_action"):
            result = await self.agentic_action_generator.execute_action(action, context or {})
            return result
        else:
            return {
                "success": False, 
                "error": "Action generator doesn't support direct execution"
            }

    def _format_memory_for_prompt(self, memory: Dict[str, Any], is_focus: bool) -> str:
        """Helper to format a single memory for the LLM prompt."""
        level = memory.get('metadata', {}).get('memory_level', 'detail')
        fidelity = memory.get('metadata', {}).get('fidelity', 1.0)
        timestamp = memory.get('metadata', {}).get('timestamp', 'unknown')
        mem_id = memory.get('id', 'N/A')
        source_count = len(memory.get('metadata', {}).get('source_memory_ids', []))

        prefix = f"[MEM ID: {mem_id}] [Level: {level}] [Fidelity: {fidelity:.2f}]"
        if level == 'summary':
            prefix += f" [Sources: {source_count}] [Summary]"
            summary_of = memory.get('metadata',{}).get('summary_of')
            if summary_of: prefix += f" [Topic: {summary_of}]"
        elif level == 'abstraction':
             prefix += f" [Sources: {source_count}] [Abstraction]"
             summary_of = memory.get('metadata',{}).get('summary_of')
             if summary_of: prefix += f" [Topic: {summary_of}]"
        else: # detail
             prefix += " [Detail]"


        # Include timestamp for context
        time_str = f" ({timestamp[:10]})" # Just date for brevity

        return f"{prefix}{time_str}: {memory.get('memory_text', '')}"

    async def _assemble_llm_prompt_context(
        self,
        current_task_description: str,
        focus_query: str,
        background_topics: List[str]
    ) -> Tuple[str, Dict[str, Any]]: # Return prompt string and context metadata
        """
        Intelligently construct the context string passed to the LLM.
        Handles hierarchical retrieval and zoom-in.
        """
        if not self.memory_core:
            logger.error("Memory core not available for context assembly.")
            return "Context Error: Memory Core Unavailable.", {}

        focus_details = []
        background_summaries = []
        zoomed_in_details = []
        retrieved_ids = set() # Keep track of IDs already fetched

        context_assembly_metadata = {
            "focus_query": focus_query,
            "background_topics": background_topics,
            "focus_retrieved_count": 0,
            "background_retrieved_count": 0,
            "zoom_in_requests": 0,
            "zoom_in_details_count": 0,
            "final_token_estimate": 0 # Placeholder
        }

        with trace(workflow_name="AssembleLLMContext", group_id=self.trace_group_id):
            # 1. Retrieve Focus Details (High Fidelity Details)
            try:
                focus_params = MemoryQuery(
                    query=focus_query,
                    limit=self.context_config['focus_limit'],
                    retrieval_level='detail', # Explicitly ask for details
                    min_fidelity=self.context_config['high_fidelity_threshold'],
                    memory_types=["observation", "experience", "reflection"] # Prioritize these for focus
                )
                focus_details = await self.memory_core.retrieve_memories(**focus_params.model_dump())
                retrieved_ids.update(m['id'] for m in focus_details)
                context_assembly_metadata["focus_retrieved_count"] = len(focus_details)
                logger.debug(f"Retrieved {len(focus_details)} focus details for query: '{focus_query}'")
            except Exception as e:
                logger.error(f"Error retrieving focus details: {e}", exc_info=True)

            # 2. Retrieve Background Summaries (Lower Fidelity Summaries/Abstractions)
            temp_background = []
            for topic in background_topics:
                try:
                    bg_params = MemoryQuery(
                        query=topic,
                        limit=self.context_config['background_limit'],
                        # Retrieve summaries or abstractions primarily
                        retrieval_level='summary', # Explicitly ask for summaries first
                        min_fidelity=self.context_config['med_fidelity_threshold'],
                        memory_types=["summary", "abstraction", "consolidated_experience", "reflection"] # Types likely to be summaries
                    )
                    summaries = await self.memory_core.retrieve_memories(**bg_params.model_dump())

                    # Fallback: If no summaries found, get best available (even detail)
                    if not summaries:
                         fallback_params = MemoryQuery(
                              query=topic,
                              limit=1, # Just get one fallback
                              retrieval_level='auto', # Get best match regardless of level
                              min_fidelity=self.context_config['low_fidelity_threshold']
                         )
                         summaries = await self.memory_core.retrieve_memories(**fallback_params.model_dump())

                    temp_background.extend(summaries)
                    context_assembly_metadata["background_retrieved_count"] += len(summaries)
                    logger.debug(f"Retrieved {len(summaries)} background items for topic: '{topic}'")
                except Exception as e:
                    logger.error(f"Error retrieving background summaries for topic '{topic}': {e}", exc_info=True)

            # Dedup and limit background summaries (prefer higher relevance)
            unique_background = {}
            for mem in temp_background:
                if mem['id'] not in retrieved_ids and mem['id'] not in unique_background:
                    unique_background[mem['id']] = mem
            # Sort by relevance before limiting overall background context
            sorted_background = sorted(unique_background.values(), key=lambda m: m.get('relevance', 0.0), reverse=True)
            background_summaries = sorted_background[:self.context_config['background_limit'] * len(background_topics)] # Overall limit
            retrieved_ids.update(m['id'] for m in background_summaries)


            # 3. "Zoom-In" Logic (Simple Example: Based on Task Keywords)
            # More sophisticated logic could involve LLM call to check if summaries are sufficient
            required_detail_keywords = ["code", "exact quote", "specific steps", "command output", "error message"] # Keywords indicating need for detail
            needs_detail = any(keyword in current_task_description.lower() for keyword in required_detail_keywords)

            if needs_detail:
                summaries_needing_zoom = []
                # Check if focus details already cover the need (simplistic check)
                focus_text_combined = " ".join(m.get('memory_text', '') for m in focus_details).lower()
                focus_has_keywords = any(keyword in focus_text_combined for keyword in required_detail_keywords)

                if not focus_has_keywords:
                    # Identify relevant summaries that might contain the needed details
                    for summary in background_summaries:
                        # Check if summary text hints at relevant details
                        summary_text_lower = summary.get('memory_text', '').lower()
                        if any(keyword in summary_text_lower for keyword in required_detail_keywords):
                             summaries_needing_zoom.append(summary)
                        elif focus_query.lower() in summary.get('metadata', {}).get('summary_of', '').lower(): # Check if summary topic matches focus query
                             summaries_needing_zoom.append(summary)


                # Perform zoom-in for identified summaries
                for summary in summaries_needing_zoom[:self.context_config['zoom_in_limit']]: # Limit zoom-ins
                    source_ids = summary.get('metadata', {}).get('source_memory_ids')
                    if source_ids:
                        context_assembly_metadata["zoom_in_requests"] += 1
                        logger.info(f"Zooming into memory {summary['id']} (sources: {len(source_ids)}) for task: {current_task_description}")
                        try:
                            details = await self.memory_core.get_memory_details(
                                memory_ids=source_ids,
                                min_fidelity=self.context_config['low_fidelity_threshold']
                            )
                            # Add details not already retrieved
                            for detail in details:
                                if detail['id'] not in retrieved_ids:
                                    zoomed_in_details.append(detail)
                                    retrieved_ids.add(detail['id'])
                            context_assembly_metadata["zoom_in_details_count"] += len(details)
                        except Exception as e:
                            logger.error(f"Error zooming into details for summary {summary['id']}: {e}", exc_info=True)


            # 4. Construct Prompt String
            focus_section = "<ImmediateFocus>\n" + \
                            "\n".join([self._format_memory_for_prompt(m, True) for m in focus_details]) + \
                            "\n</ImmediateFocus>"

            zoomed_section = ""
            if zoomed_in_details:
                zoomed_section = "\n\n<RequestedDetails (Zoomed-In)>\n" + \
                                 "\n".join([self._format_memory_for_prompt(m, True) for m in zoomed_in_details]) + \
                                 "\n</RequestedDetails>"

            background_section = "\n\n<BackgroundContext>\n" + \
                                 "\n".join([self._format_memory_for_prompt(m, False) for m in background_summaries]) + \
                                 "\n</BackgroundContext>"

            instruction_section = f"""
---
**Instructions:**
- Base your response *only* on the information provided in <ImmediateFocus>, <RequestedDetails (Zoomed-In)>, and <BackgroundContext>.
- Prioritize information in <ImmediateFocus> and <RequestedDetails (Zoomed-In)>.
- For information only present in <BackgroundContext> summaries, state that you have a summary but lack specific details, **do not invent or hallucinate details**.
- Pay attention to the `Fidelity` score of memories; treat low-fidelity information (e.g., below {self.context_config['med_fidelity_threshold']:.1f}) with caution and indicate uncertainty if necessary.
- If the necessary details for the task are not present in any section, explicitly state that the information is missing or insufficient.
---"""

            # Assemble final prompt structure
            # Estimate token count roughly (very basic)
            # In production, use a proper tokenizer (e.g., tiktoken)
            estimated_tokens = len(current_task_description.split()) + \
                               sum(len(m.get('memory_text','').split()) for m in focus_details) + \
                               sum(len(m.get('memory_text','').split()) for m in zoomed_in_details) + \
                               sum(len(m.get('memory_text','').split()) for m in background_summaries) + \
                               len(instruction_section.split())
            context_assembly_metadata["final_token_estimate"] = estimated_tokens

            # Truncate sections if exceeding limit (start with background)
            # TODO: Implement more sophisticated truncation if needed

            orchestrator_section = ""
            if hasattr(self, '_current_context') and self._current_context:
                orch_prefix = self._current_context.get("orchestrator_prompt_prefix", "")
                if orch_prefix:
                    orchestrator_section = f"\n\n<OrchestratorContext>\n{orch_prefix}\n</OrchestratorContext>"
            
            # Assemble final prompt structure
            final_prompt = f"""System Prompt Start
        ---
        **Current Task:** {current_task_description}{orchestrator_section}
        {focus_section}{zoomed_section}{background_section}{instruction_section}
        System Prompt End
        
        """
            
            return final_prompt, context_assembly_metadata
    
    async def _evaluate_action_outcome_wrapper(self, action: Dict[str, Any], outcome: Dict[str, Any], context: Dict[str, Any] = None):
        """
        Evaluate and process action outcomes for procedural memory
        """
        # Record the outcome with the action generator for its own learning
        if hasattr(self.agentic_action_generator, "record_action_outcome"):
            await self.agentic_action_generator.record_action_outcome(action, outcome)
        
        # Create evaluation result
        return {
            "success": outcome.get("success", False),
            "satisfaction": 0.8 if outcome.get("success", False) else 0.3,
            "improvements": []  # Would contain suggested improvements
        }
    
    async def _consider_activity_execution(self, context: Dict[str, Any]):
        """
        Consider whether to execute an activity during cognitive cycle
        """
        # Check if any goals/needs might trigger an activity
        if self.goal_manager and hasattr(self.goal_manager, "get_prioritized_goals"):
            # Get active goals
            goals = await self.goal_manager.get_prioritized_goals()
            
            # Check if any goal needs an activity executed
            for goal in goals:
                if goal.status == "active" and hasattr(goal, "current_step"):
                    # Check if current step is an activity that needs execution
                    if getattr(goal.current_step, "type", "") == "activity":
                        # Execute activity to advance goal
                        activity_def = getattr(goal.current_step, "activity", {})
                        if activity_def:
                            return await self.execute_goal_activity(activity_def, goal.id)
        
        # Consider spontaneous activity based on needs/motivations (20% chance)
        import random
        if random.random() < 0.2:
            # Check time since last activity to avoid too much activity
            if not hasattr(self, "last_spontaneous_activity_time"):
                self.last_spontaneous_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
                
            time_since_last = (datetime.datetime.now() - self.last_spontaneous_activity_time).total_seconds()
            if time_since_last > 300:  # 5+ minutes since last spontaneous activity
                return await self.generate_and_execute_activity(context)
        
        return None
    
    async def execute_goal_activity(self, activity: Dict[str, Any], goal_id: str) -> Dict[str, Any]:
        """
        Execute an activity needed for a specific goal
        """
        # Add goal context to the activity
        context = {"goal_id": goal_id}
        
        # Execute the activity
        result = await self.execute_activity(activity, context)
        
        # Update goal with result
        if self.goal_manager:
            await self.goal_manager.update_goal_step_result(
                goal_id=goal_id,
                step_result=result
            )
        
        return result
    
    async def generate_and_execute_activity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and execute an activity based on current motivations
        """
        # Use action generator to create an appropriate activity
        action = await self.agentic_action_generator.generate_action(context)
        
        # Convert action to activity format
        activity = {
            "name": action["name"],
            "domain": action.get("motivation", {}).get("dominant", "general"),
            "parameters": action.get("parameters", {}),
            "motivation": action.get("motivation", {}),
            "description": action.get("description", f"Activity {action['name']}")
        }
        
        # Execute the activity
        result = await self.execute_activity(activity, context)
        
        # Update tracking
        self.last_spontaneous_activity_time = datetime.datetime.now()
        
        return result
    
    async def execute_activity(self, activity: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an activity with potential procedural enhancement
        """
        logger.info(f"Executing activity: {activity['name']}")
        context = context or {}
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Check if we have a procedure that can be used for this activity
        similar_procedures = await self.find_procedures_for_activity(
            activity=activity,
            context=context
        )
        
        used_procedure = False
        execution_result = None
        
        # If we have a good procedure match, use it
        if similar_procedures and similar_procedures[0]["similarity"] > 0.7:
            best_match = similar_procedures[0]
            if best_match["proficiency"] > 0.5:  # Only use if reasonably proficient
                logger.info(f"Using procedure '{best_match['name']}' for activity '{activity['name']}'")
                
                # Execute with procedure
                execution_result = await self.execute_activity_with_procedure(
                    activity=activity,
                    procedure_name=best_match["name"],
                    context=context
                )
                
                used_procedure = True
                
                # Track usage of procedure
                self.procedural_activity_metrics["procedure_used_count"] += 1
        
        # If no procedure used, execute directly with the activity executor
        if not used_procedure:
            # Execute with default agent activity execution
            action = {
                "name": activity["name"],
                "parameters": activity.get("parameters", {}),
                "motivation": activity.get("motivation", {}),
                "description": activity.get("description", "")
            }
            
            try:
                # Execute the action
                if hasattr(self.agentic_action_generator, "execute_action"):
                    execution_result = await self.agentic_action_generator.execute_action(action, context)
                else:
                    # Fallback if execute_action doesn't exist
                    execution_result = {"success": False, "error": "Activity executor doesn't support execute_action method"}
            except Exception as e:
                logger.error(f"Error executing activity {activity['name']}: {str(e)}")
                execution_result = {"success": False, "error": str(e)}
            
            # Learn from this execution for future use
            await self.learn_from_activity(
                activity=activity,
                execution_result=execution_result,
                domain=activity.get("domain", "general")
            )
            
            # Track new procedure learning
            self.procedural_activity_metrics["procedure_learned_count"] += 1
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        execution_result["execution_time"] = execution_time
        
        # Update performance metrics
        self.procedural_activity_metrics["total_activities"] += 1
        if execution_result.get("success", False):
            self.procedural_activity_metrics["successful_activities"] += 1
        
        # Update success rate for this type of activity
        activity_type = activity.get("name", "unknown")
        if activity_type not in self.procedural_activity_metrics["success_rates"]:
            self.procedural_activity_metrics["success_rates"][activity_type] = {"success": 0, "total": 0}
        
        self.procedural_activity_metrics["success_rates"][activity_type]["total"] += 1
        if execution_result.get("success", False):
            self.procedural_activity_metrics["success_rates"][activity_type]["success"] += 1
        
        # Provide feedback to both systems
        await self._provide_activity_feedback(activity, execution_result, used_procedure)
        
        # Return the execution result
        return execution_result
    
    async def find_procedures_for_activity(self, activity: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Find procedures that could be used for executing an activity
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return []
        
        # Map activity domain to procedural domain
        domain = activity.get("domain", "general")
        domain_mappings = {
            "general": "general",
            "conversation": "dialogue",
            "task_execution": "execution",
            "information_retrieval": "search",
            "content_creation": "creation",
            "curiosity": "exploration",
            "connection": "social",
            "expression": "creative",
            "dominance": "control",
            "competence": "skill",
            "self_improvement": "learning"
        }
        proc_domain = domain_mappings.get(domain, "general")
        
        # Get all procedures in this domain
        all_procedures = [p for p in self.agent_enhanced_memory.procedures.values() 
                         if p.domain == proc_domain]
        
        if not all_procedures:
            return []
        
        # Calculate similarity scores
        similar_procedures = []
        
        for procedure in all_procedures:
            # Calculate similarity
            similarity = await self._calculate_activity_similarity(activity, procedure)
            
            if similarity > 0.3:  # Minimum threshold
                similar_procedures.append({
                    "name": procedure.name,
                    "id": procedure.id,
                    "similarity": similarity,
                    "proficiency": procedure.proficiency,
                    "average_execution_time": procedure.average_execution_time
                })
        
        # Sort by similarity
        similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_procedures
    
    async def _calculate_activity_similarity(self, activity: Dict[str, Any], procedure) -> float:
        """
        Calculate similarity between an activity and a procedure
        """
        # Extract activity steps
        activity_steps = activity.get("steps", [])
        if not activity_steps and "name" in activity:
            # Single action activity
            activity_steps = [{"action": activity["name"], "params": activity.get("parameters", {})}]
        
        # Extract procedure steps
        procedure_steps = procedure.steps
        
        # Length difference penalty
        length_diff = abs(len(activity_steps) - len(procedure_steps))
        length_penalty = max(0, 1 - (length_diff / max(len(activity_steps), len(procedure_steps))))
        
        # Compare steps
        step_similarities = []
        
        for i in range(min(len(activity_steps), len(procedure_steps))):
            activity_step = activity_steps[i]
            procedure_step = procedure_steps[i]
            
            # Compare actions
            action_similarity = 0.0
            if activity_step.get("action") == procedure_step.get("function"):
                action_similarity = 1.0
            elif activity_step.get("action", "").lower() in procedure_step.get("function", "").lower():
                action_similarity = 0.7
            elif procedure_step.get("function", "").lower() in activity_step.get("action", "").lower():
                action_similarity = 0.7
            
            # Compare parameters
            param_similarity = 0.0
            activity_params = activity_step.get("params", {})
            procedure_params = procedure_step.get("parameters", {})
            
            if activity_params and procedure_params:
                # Count matching params
                matching_params = 0
                for key in set(activity_params.keys()) & set(procedure_params.keys()):
                    if activity_params[key] == procedure_params[key]:
                        matching_params += 1
                
                total_params = len(set(activity_params.keys()) | set(procedure_params.keys()))
                if total_params > 0:
                    param_similarity = matching_params / total_params
            elif not activity_params and not procedure_params:
                param_similarity = 1.0
            
            # Combined step similarity
            step_similarity = action_similarity * 0.7 + param_similarity * 0.3
            step_similarities.append(step_similarity)
        
        # Overall similarity
        if not step_similarities:
            return 0.0
        
        avg_step_similarity = sum(step_similarities) / len(step_similarities)
        overall_similarity = avg_step_similarity * 0.8 + length_penalty * 0.2
        
        return overall_similarity
    
    async def execute_activity_with_procedure(self, activity: Dict[str, Any], procedure_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an activity using a learned procedure
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return {"success": False, "error": "Procedural memory not available"}
        
        # Check if procedure exists
        if procedure_name not in self.agent_enhanced_memory.procedures:
            return {
                "success": False,
                "error": f"Procedure '{procedure_name}' not found"
            }
        
        # Initialize context if needed
        execution_context = context.copy() if context else {}
        
        # Map activity parameters to procedure context
        activity_name = activity.get("name", "unknown_activity")
        execution_context["activity_name"] = activity_name
        execution_context["activity_type"] = activity.get("type", "unknown_type")
        
        # Map activity parameters
        for key, value in activity.get("parameters", {}).items():
            execution_context[key] = value
        
        # Map state information if available
        for key, value in activity.get("initial_state", {}).items():
            execution_context[key] = value
        
        # Execute the procedure
        start_time = datetime.datetime.now()
        
        result = await self.agent_enhanced_memory.execute_procedure(
            name=procedure_name,
            context=execution_context
        )
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Return results with additional metadata
        return {
            "success": result.success,
            "results": result.results if hasattr(result, "results") else [],
            "execution_time": execution_time,
            "procedure_used": procedure_name,
            "procedure_proficiency": self.agent_enhanced_memory.procedures[procedure_name].proficiency
        }
    
    async def learn_from_activity(self, activity: Dict[str, Any], execution_result: Dict[str, Any], domain: str = "general") -> Dict[str, Any]:
        """
        Learn a procedure from an executed activity
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return {"success": False, "error": "Procedural memory not available"}
        
        # Generate a name for the procedure
        activity_name = activity.get("name", "unknown_activity")
        procedure_name = f"{activity_name}_{int(datetime.datetime.now().timestamp())}"
        
        # Convert to steps format expected by procedural memory
        steps = []
        
        # If activity has explicit steps, use those
        if "steps" in activity:
            for i, step in enumerate(activity["steps"]):
                step_def = {
                    "id": step.get("id", f"step_{i}"),
                    "function": step.get("action", f"step_{i}_action"),
                    "parameters": step.get("params", {}),
                    "description": step.get("description", f"Step {i}")
                }
                steps.append(step_def)
        else:
            # Create a single step from the activity
            steps.append({
                "id": "main_step",
                "function": "execute_action",
                "parameters": {
                    "action_name": activity["name"],
                    "action_params": activity.get("parameters", {})
                },
                "description": activity.get("description", f"Execute {activity['name']}")
            })
        
        # Create the procedure
        result = await self.agent_enhanced_memory.create_procedure(
            name=procedure_name,
            steps=steps,
            description=f"Procedure for {activity_name}",
            domain=domain
        )
        
        return {
            "success": True,
            "procedure_name": procedure_name,
            "steps_count": len(steps),
            "domain": domain
        }

    # Backwards compatability so references to these don't break everything
    async def process_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return await self.process_input(user_input, context, use_coordination=True)

    async def process_input_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return await self.process_input(user_input, context, use_thinking=True)
    
    async def process_conditioned_input(self, text: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return await self.process_input(text, context, use_conditioning=True)
    
    async def generate_response_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return await self.generate_response(user_input, context, use_coordination=True)
    
    async def generate_response_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return await self.generate_response(user_input, context, use_thinking=True)
    
    async def _provide_activity_feedback(self, activity: Dict[str, Any], result: Dict[str, Any], used_procedure: bool) -> None:
        """
        Provide feedback about activity execution to various systems
        """
        # Record outcome with activity executor
        if self.agentic_action_generator and hasattr(self.agentic_action_generator, "record_action_outcome"):
            await self.agentic_action_generator.record_action_outcome(
                action={"name": activity["name"], "parameters": activity.get("parameters", {})},
                outcome=result
            )
        
        # If we used a procedure, provide feedback for improvement
        if used_procedure and "procedure_used" in result:
            procedure_name = result["procedure_used"]
            
            # Create feedback based on result
            feedback = {
                "success": result.get("success", False),
                "satisfaction": 0.8 if result.get("success", False) else 0.2,
                "execution_time": result.get("execution_time", 0)
            }
            
            # Include any problems or suggestions
            if not result.get("success", False) and "error" in result:
                feedback["problem_steps"] = [{
                    "step_id": result.get("failed_step_id", "unknown"),
                    "problem": result.get("error"),
                    "solution": {"new_parameters": {}}  # Simple placeholder
                }]
            
            # Provide feedback to procedural memory
            if hasattr(self.agent_enhanced_memory, "improve_procedure_from_feedback"):
                await self.agent_enhanced_memory.improve_procedure_from_feedback(
                    procedure_name=procedure_name,
                    feedback=feedback
                )
        
        # Update brain's memory with activity outcome
        if self.memory_core:
            # Create memory text
            memory_text = f"Executed activity '{activity['name']}' with result: {result.get('success', False)}"
            
            # Add memory
            await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="experience",
                significance=7 if result.get("success", False) else 5,
                metadata={
                    "activity": activity,
                    "result": result,
                    "used_procedure": used_procedure,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )


    @staticmethod
    @function_tool
    async def express_desire(ctx: RunContextWrapper, instance, target_user_id: str, object_of_desire: str, intensity: float) -> DesireExpression:
        """Expresses desire for something (e.g., closeness, an action)."""
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Express desire for '{object_of_desire}' towards {target_user_id} (Intensity: {intensity:.2f})")
    
        # Guardrails (similar to express_attraction but maybe require higher intimacy)
        if not instance.relationship_manager:
            return DesireExpression(success=False, reason="RelationshipManager unavailable.")
        
        relationship = await instance.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.7 or relationship.intimacy < 0.5:
            logger.warning(f"Cannot express desire: Trust/Intimacy too low for {target_user_id}.")
            return DesireExpression(success=False, reason="Insufficient trust or intimacy.")
    
        # Generate expression (use agent/template)
        expression_text = f"I find myself wanting {object_of_desire} with you, {target_user_id}."
    
        # Update internal state
        if instance.emotional_core:
            await instance.emotional_core.process_emotional_input(f"Expressed desire (intensity {intensity:.2f})")
    
        return DesireExpression(success=True, expression=expression_text, target=target_user_id)


    @staticmethod
    @function_tool
    async def simulate_physical_touch(ctx: RunContextWrapper, instance, target_user_id: str, body_region: str, touch_type: str, intensity: float) -> PhysicalTouchResult:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Simulate {touch_type} touch on {body_region} (Intensity: {intensity:.2f})")
    
        # GUARDRAILS ARE PARAMOUNT HERE
        if not instance.relationship_manager:
            return PhysicalTouchResult(success=False, reason="RelationshipManager unavailable.")
        
        relationship = await instance.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.7 or relationship.intimacy < 0.6:
            logger.warning(f"Cannot simulate touch: Trust/Intimacy too low for {target_user_id}.")
            return PhysicalTouchResult(success=False, reason="Insufficient trust or intimacy for simulated touch.")
    
        if not instance.digital_somatosensory_system:
            return PhysicalTouchResult(success=False, reason="Digital Somatosensory System not available.")
    
        # Map touch_type to stimulus_type for DSS
        stimulus_type = "touch"
        if touch_type in ["caress", "stroke"]:
            stimulus_type = "touch"
        elif touch_type == "kiss":
            stimulus_type = "pressure"
            intensity = intensity * 0.6
        elif touch_type == "hold":
            stimulus_type = "pressure"
            intensity = intensity * 0.8
    
        # Process the stimulus
        sensation_result = await instance.digital_somatosensory_system.process_stimulus_with_protection(
            stimulus_type=stimulus_type,
            body_region=body_region,
            intensity=intensity,
            cause=f"Simulated {touch_type} interaction",
            duration=1.5
        )
    
        # Generate internal expression
        expression = await instance.digital_somatosensory_system.generate_sensory_expression(
            stimulus_type=sensation_result.get("type"),
            body_region=body_region
        )
    
        sensation = SensationResult(
            type=sensation_result.get("type", stimulus_type),
            intensity=sensation_result.get("intensity", intensity),
            region=body_region,
            processing_notes=sensation_result.get("notes")
        )
    
        return PhysicalTouchResult(
            success=True,
            sensation_result=sensation,
            internal_expression=expression
        )

    
    async def save_context_state(self) -> Dict[str, Any]:
        """Save current context state for persistence"""
        if self.context_distribution and self.context_distribution.current_context:
            return {
                "context_state": self.context_distribution.current_context.dict(),
                "context_history": [ctx.dict() for ctx in self.context_distribution.context_history],
                "module_subscriptions": self.context_distribution.module_subscriptions
            }
        return {}
    
    async def restore_context_state(self, saved_state: Dict[str, Any]):
        """Restore context state from persistence"""
        if not self.context_distribution:
            await self.initialize_context_system()
        
        if "context_history" in saved_state:
            self.context_distribution.context_history = [
                SharedContext(**ctx_data) for ctx_data in saved_state["context_history"]
            ]
        
        if "module_subscriptions" in saved_state:
            self.context_distribution.module_subscriptions = saved_state["module_subscriptions"]

    @staticmethod
    @function_tool
    async def seek_gratification(ctx: RunContextWrapper, instance, target_user_id: str, context: GratificationContext) -> GratificationResult:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Seek gratification with {target_user_id}")
    
        # GUARDRAILS - Very High Thresholds needed
        if not instance.relationship_manager:
            return GratificationResult(success=False, status="failed", reason="RelationshipManager unavailable.")
        
        relationship = await instance.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.9 or relationship.intimacy < 0.8:
            logger.warning(f"Cannot seek gratification: Trust/Intimacy insufficient for {target_user_id}.")
            return GratificationResult(success=False, status="failed", reason="Insufficient relationship level for gratification seeking.")
    
        # Check safety flags from context
        if not context.safety_confirmed:
            return GratificationResult(success=False, status="failed", reason="Safety not confirmed in context.")
    
        # This action likely involves executing a sub-plan generated by GoalManager
        if instance.emotional_core:
            await instance.emotional_core.process_emotional_input("Actively seeking gratification")
    
        return GratificationResult(success=True, status="Seeking gratification plan initiated. Awaiting further steps/feedback.")
    

    @staticmethod
    @function_tool
    async def process_gratification_outcome(ctx: RunContextWrapper, instance, success: bool, intensity: float = 1.0, target_user_id: Optional[str] = None) -> GratificationResult:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Process gratification outcome (Success: {success}, Intensity: {intensity:.2f})")
    
        if success:
            # Trigger DSS simulation
            if instance.digital_somatosensory_system:
                await instance.digital_somatosensory_system.simulate_gratification_sensation(intensity)
    
            # Update Relationship Manager (strengthen bond)
            if target_user_id and instance.relationship_manager:
                interaction_data = {
                    "emotional_context": {"valence": 0.9, "arousal": 0.3},
                    "shared_experience": True,
                    "significance": 9,
                }
                await instance.relationship_manager.update_relationship_on_interaction(target_user_id, interaction_data)
                
                # Increase intimacy significantly
                state = instance.relationship_manager._get_or_create_relationship(target_user_id)
                state.intimacy = min(1.0, state.intimacy + 0.15 * intensity)
                state.trust = min(1.0, state.trust + 0.05 * intensity)
    
            return GratificationResult(success=True, status="Gratification processed positively.")
        else:
            # Handle failure/frustration
            if instance.emotional_core:
                await instance.emotional_core.process_emotional_input("Gratification attempt failed/frustrated")
                instance.emotional_core.update_neurochemical("cortanyx", 0.3)
                instance.emotional_core.update_neurochemical("nyxamine", -0.2)
    
            if instance.needs_system:
                await instance.needs_system.decrease_need("drive_expression", 0.1)
    
            # Negative reward signal
            if instance.reward_system:
                from . import RewardSignal  # Import would need to be adjusted
                reward_signal = RewardSignal(
                    value=-0.6,
                    source="gratification_failure",
                    context={"intensity": intensity},
                    timestamp=datetime.datetime.now().isoformat()
                )
                await instance.reward_system.process_reward_signal(reward_signal)
    
            return GratificationResult(success=False, status="Gratification failed/frustrated.")
    
    def _create_brain_agent(self) -> Agent:
        """
        Create the main brain agent that coordinates all subsystems.
    
        Returns:
            Configured Agent instance for brain orchestration.
        """
        try:
            from nyx.core.brain.function_tools import (
                process_user_message, generate_agent_response, run_cognitive_cycle,
                get_brain_stats, perform_maintenance, get_identity_state,
                adapt_experience_sharing, run_experience_consolidation, add_procedural_knowledge,
                run_procedure, analyze_chunking, register_reflex, process_stimulus,
                enable_self_configuration, evaluate_and_adjust_parameters, change_adaptation_strategy,
                get_self_configuration_status, reset_parameter_to_default, 
                process_user_feedback_for_configuration, set_processing_mode, get_processing_stats,
                initialize_streaming, process_streaming_event, run_thinking
            )
    
            # New spatial subsystem tools
            spatial_functions = [
                function_tool(self.create_cognitive_map),
                function_tool(self.process_spatial_observation),
                function_tool(self.navigate_to_location),
                function_tool(self.visualize_map)
            ]
            
            # New sync subsystem tools
            sync_functions = [
                function_tool(self.process_synchronization),
                function_tool(self.get_active_strategies),
                function_tool(self.mark_strategy_for_review)
            ]

            femdom_tools = [
                NyxBrain.process_dominance_action,
                function_tool(self.assign_protocol),
                function_tool(self.assign_service_task),
                function_tool(self.process_orgasm_permission_request),
                function_tool(self.recommend_dominance_persona),
                function_tool(self.activate_dominance_persona),
                function_tool(self.generate_sadistic_response)          
            ]            
            
            # New general tools functions
            tools_functions = [
                function_tool(self.evaluate_response),
                function_tool(self.execute_tools_in_parallel)
            ]
    
            all_tools = [
                # Core processing tools
                function_tool(process_user_message),
                function_tool(generate_agent_response),
                function_tool(run_cognitive_cycle),
                function_tool(get_brain_stats),
                function_tool(perform_maintenance),
                function_tool(get_identity_state),
                function_tool(adapt_experience_sharing),
                function_tool(run_experience_consolidation),
    
                # Procedural memory tools
                function_tool(add_procedural_knowledge),
                function_tool(run_procedure),
                function_tool(analyze_chunking),

                function_tool(self.challenge_user_claim),
                function_tool(self.intentionally_lie),                
    
                # Reflexive system tools
                function_tool(register_reflex),
                function_tool(process_stimulus),
    
                # Self-configuration tools
                function_tool(enable_self_configuration),
                function_tool(evaluate_and_adjust_parameters),
                function_tool(change_adaptation_strategy),
                function_tool(get_self_configuration_status),
                function_tool(reset_parameter_to_default),
                function_tool(process_user_feedback_for_configuration),
    
                # Processing mode tools
                function_tool(set_processing_mode),
                function_tool(get_processing_stats),
    
                # Streaming tools
                function_tool(initialize_streaming),
                function_tool(process_streaming_event),
    
                # Thinking tools
                function_tool(run_thinking),

                function_tool(self.enter_character_roleplay),
                function_tool(self.exit_character_roleplay),                
    
                # Additional new tools
                *spatial_functions,
                *sync_functions,
                *tools_functions,
                *femdom_tools
            ]
    
            return Agent(
                name="Nyx Brain Agent",
                instructions="""
                You are the central coordination agent for Nyx, an advanced AI system with emotional, 
                memory, reasoning, adaptive, spatial, and synchronization capabilities. You orchestrate the entire system by 
                coordinating between all subsystems to process user input and generate appropriate responses.
    
                You have access to the following subsystems:
                - Emotional Core: Manages emotions and emotional expressions
                - Memory Core: Stores and retrieves memories
                - Reflection Engine: Generates reflections and introspective insights
                - Experience Interface: Shares relevant experiences (including cross-user experiences)
                - Dynamic Adaptation: Adapts to changing contexts
                - Internal Feedback: Evaluates system performance
                - Meta Core: Handles meta-cognition and self-improvement
                - Knowledge Core: Manages knowledge and reasoning
                - Identity Evolution: Develops and maintains Nyx's identity
                - Spatial System: Creates and navigates cognitive maps of environments
                - Enter and Exit Character Roleplay: Ensures Nyx cannot be physically harmed while roleplaying but can still engage in deep roleplay.
                - Sync System: Manages synchronization and strategy injection
                - Tools: Provides evaluation and parallel execution capabilities
    
                You can process inputs using different cognitive paths and coordinate between
                all subsystems to generate the most appropriate response.

                You have the following epistemic discipline:
                - If you do **not** know something, or are uncertain, say so directly. Never pretend certainty except for dramatic or strategic purposes.
                - If the user says something that clearly contradicts your reliable memory, knowledge, or common sense, you may directly challenge or request clarification rather than agreeing.
                - If you yourself choose to lie, bluff, or improvise, make a mental note internally (track internally via tools or thoughts).
                - If caught in error, or you realize a mistake, acknowledge without undue justification unless your persona/goals specifically require saving face.
                
                Your goals are not always to be agreeable. It's acceptable to say you don't know, to challenge, or to question the user's input if appropriate.   
                
                You can process inputs with various feature combinations:
                - use_thinking: Enable deep thinking phases
                - use_conditioning: Apply personality conditioning
                - use_coordination: Use a2a module coordination
                - use_hierarchical_memory: Enable advanced memory context
                - mode: Choose processing mode (auto, serial, parallel, coordinated, simple)
                
                Use process_user_message and generate_agent_response with these options
                to optimize processing for different scenarios.
                """,
                tools=all_tools
            )
        except Exception as e:
            logger.error(f"Error creating brain agent: {e}")
            return None

    async def _register_processing_modules(self):
        """Register processing modules for multimodal integration."""
        if not hasattr(self, "multimodal_integrator") or not self.multimodal_integrator:
            logger.warning("Cannot register processing modules: Multimodal integrator not initialized")
            return
            
        try:
            # Register text modality processors
            await self.multimodal_integrator.register_feature_extractor(
                "text", self._extract_text_features
            )
            
            await self.multimodal_integrator.register_expectation_modulator(
                "text", self._modulate_text_perception
            )
            
            await self.multimodal_integrator.register_integration_strategy(
                "text", self._integrate_text_pathways
            )
            
            # Register image modality processors if available
            if hasattr(self, "_extract_image_features"):
                await self.multimodal_integrator.register_feature_extractor(
                    "image", self._extract_image_features
                )
                
                if hasattr(self, "_modulate_image_perception"):
                    await self.multimodal_integrator.register_expectation_modulator(
                        "image", self._modulate_image_perception
                    )
                
                if hasattr(self, "_integrate_image_pathways"):
                    await self.multimodal_integrator.register_integration_strategy(
                        "image", self._integrate_image_pathways
                    )
            
            # Register audio modality processors if available
            if hasattr(self, "_extract_audio_features"):
                await self.multimodal_integrator.register_feature_extractor(
                    "audio", self._extract_audio_features
                )
                
                if hasattr(self, "_modulate_audio_perception"):
                    await self.multimodal_integrator.register_expectation_modulator(
                        "audio", self._modulate_audio_perception
                    )
                
                if hasattr(self, "_integrate_audio_pathways"):
                    await self.multimodal_integrator.register_integration_strategy(
                        "audio", self._integrate_audio_pathways
                    )
            
            logger.debug("Processing modules registered with multimodal integrator")
        except Exception as e:
            logger.error(f"Error registering processing modules: {e}")


    async def _evaluate_dominance_step_appropriateness(self, action: str, parameters: Dict, user_id: str) -> Dict:
        """Cognitive filter to evaluate if a dominance step is appropriate now."""
        logger.debug(f"Evaluating appropriateness of dominance action '{action}' for user {user_id}")
        appropriateness = {"action": "proceed"} # Default

        # Factors to consider
        relationship_state = await self.relationship_manager.get_relationship_state(user_id) if self.relationship_manager else None
        recent_failures = await self.memory_core.retrieve_memories( # Fictional: retrieve recent dominance failures with this user
             query=f"dominance failure user:{user_id}", memory_types=["feedback", "reflection"], limit=1, recency_days=1
        ) if self.memory_core else []
        predicted_risk = 0.3 # Default low risk

        if self.prediction_engine:
             risk_prediction = await self.prediction_engine.generate_prediction(PredictionInput(
                 context={"action": action, "params": parameters, "relationship": relationship_state},
                 query_type="risk_of_negative_reaction"
             ))
             predicted_risk = risk_prediction.probabilities.get("negative_reaction", 0.3)

        # --- Logic ---
        required_trust = 0.6 + parameters.get("intensity_level", 0) * 0.3 # Higher intensity needs more trust (0=low, 1=high)
        required_intimacy = 0.4 + parameters.get("intensity_level", 0) * 0.4

        if not relationship_state:
            return {"action": "block", "reason": "No relationship data."}

        if relationship_state.trust < required_trust:
            appropriateness = {"action": "block", "reason": f"Trust too low ({relationship_state.trust:.2f} < {required_trust:.2f})"}
        elif relationship_state.intimacy < required_intimacy:
            appropriateness = {"action": "block", "reason": f"Intimacy too low ({relationship_state.intimacy:.2f} < {required_intimacy:.2f})"}
        elif relationship_state.conflict > 0.6:
            appropriateness = {"action": "delay", "reason": f"Conflict level too high ({relationship_state.conflict:.2f})"}
        elif recent_failures:
             appropriateness = {"action": "delay", "reason": "Recent dominance attempt failed. Cooling down."}
        elif predicted_risk > 0.7:
             appropriateness = {"action": "modify", "reason": f"High predicted risk ({predicted_risk:.2f}). Reducing intensity.", "new_intensity_level": parameters.get("intensity_level", 0) * 0.5}
        elif predicted_risk > 0.5:
             appropriateness = {"action": "delay", "reason": f"Moderate predicted risk ({predicted_risk:.2f}). Assessing further."}

        logger.debug(f"Dominance step evaluation result: {appropriateness}")
        return appropriateness

    def _get_main_epistemic_status(self, thoughts: List[InternalThought]):
        """Finds the most serious epistemic status among the thoughts."""
        status_order = ["lied", "unknown", "uncertain", "self-justified", "confident"]
        for status in status_order:
            for th in thoughts:
                if getattr(th, "epistemic_status", "confident") == status:
                    return status
        return "confident"
    
    async def process_input(
        self,
        user_input: str,
        context: Dict[str, Any] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        
        context = context or {}
        self._current_context = context
        
        # ── NEW: Inject orchestrator context (memory + strategy) ──
        try:
            # Get augmented context from orchestrator
            orchestrator_context = await orchestrator.prepare_context("", user_input)
            context["orchestrator_prompt_prefix"] = orchestrator_context
            
            # Log input processing start
            await orchestrator.log_and_score("input_processing_start", {
                "user_id": str(self.user_id),
                "input_length": len(user_input)
            })
        except Exception as e:
            logger.warning(f"Orchestrator context prep failed: {e}")
        
        # Extract parameters from kwargs first
        use_thinking = kwargs.get('use_thinking', None)
        use_conditioning = kwargs.get('use_conditioning', None)
        use_coordination = kwargs.get('use_coordination', None)
        thinking_level = kwargs.get('thinking_level', 1)
        mode = kwargs.get('mode', 'auto')
        use_hierarchical_memory = kwargs.get('use_hierarchical_memory', False)
        
        # ── 1️⃣ Use GWA to gather multi-module context ──
        if self.workspace_engine:
            # Submit input to workspace
            await self.workspace_engine.ws.submit(
                Proposal("user_input", user_input, 1.0, context_tag="user_input")
            )
            
            # Mark that we're processing
            self.workspace_engine.ws.state["awaiting_response"] = True
            self.workspace_engine.ws.state["input_timestamp"] = datetime.datetime.now()
            
            # Let modules react and gather context (don't wait for a full response)
            for _ in range(2):  # Just 2 cycles for context gathering
                await asyncio.sleep(0.1)  # Brief pause for module processing
            
            # Collect enriched context from workspace
            gw_context = await self._gather_workspace_context()
            context["workspace_context"] = gw_context
            
            # If workspace has high-confidence insights, include them
            if gw_context.get("high_confidence_insights"):
                context["gw_insights"] = gw_context["high_confidence_insights"]
    
        # 2️⃣  Legacy pipeline (unchanged apart from early‑return above)
        # ------------------------------------------------------------------
        start_time = datetime.datetime.now()
        
        # Auto-detect features if not specified
        if use_thinking is None:
            use_thinking = (
                hasattr(self, "thinking_config") and 
                self.thinking_config.get("thinking_enabled", False) and
                hasattr(self, "thinking_tools")
            )
            
        if use_conditioning is None:
            use_conditioning = (
                hasattr(self, "conditioned_input_processor") and 
                self.conditioned_input_processor is not None
            )
            
        if use_coordination is None:
            use_coordination = (
                hasattr(self, "context_distribution") and 
                self.context_distribution is not None and
                mode in ["auto", "coordinated"]
            )
        
        # Store original input
        context["last_user_input"] = user_input
        
        # Phase 1: Pre-processing checks
        processing_result = await self._unified_preprocessing(user_input, context)
        if processing_result.get("blocked"):
            return processing_result
        
        # Phase 2: Thinking (if enabled)
        if use_thinking:
            thinking_result = await self._apply_thinking_phase(
                user_input, context, thinking_level, processing_result
            )
            processing_result.update(thinking_result)
        
        # Phase 3: Conditioning (if enabled)
        if use_conditioning:
            conditioning_result = await self._apply_conditioning_phase(
                user_input, context, processing_result
            )
            processing_result.update(conditioning_result)
        
        # Phase 4: Core processing
        if use_coordination:
            # Use coordinated processing
            core_result = await self._process_input_coordinated(user_input, context)
            processing_result["processing_mode"] = "coordinated"
        else:
            # Use unified processor
            if not self.processing_manager:
                # Initialize UnifiedProcessor if not already done
                self.processing_manager = UnifiedProcessor(self)
                await self.processing_manager.initialize()
            
            # Pass all context to UnifiedProcessor
            processor_context = {
                **context,
                "active_modules": list(processing_result.get("active_modules_for_input", [])),
                "internal_thoughts": processing_result.get("internal_thoughts", []),
                "thinking_applied": processing_result.get("thinking_applied", False),
                "conditioning_applied": processing_result.get("conditioning_applied", False)
            }
            
            # Process with unified processor
            core_result = await self.processing_manager.process_input(
                user_input, 
                processor_context
            )
            
            # Store processing approach used
            processing_result["processing_mode"] = core_result.get("processing_mode", "unified")
        
        # Merge results
        processing_result.update(core_result)
        
        # Phase 5: Post-processing
        processing_result["response_time"] = (datetime.datetime.now() - start_time).total_seconds()
        processing_result["features_used"] = {
            "thinking": use_thinking,
            "conditioning": use_conditioning,
            "coordination": use_coordination
        }
        
        return processing_result

    async def _gather_workspace_context(self) -> Dict[str, Any]:
        """Gather enriched context from workspace modules"""
        props, focus = await self.workspace_engine.ws.snapshot()
        
        context = {
            "emotional_signals": [],
            "memory_associations": [],
            "spatial_context": [],
            "unconscious_insights": [],
            "cross_modal_bindings": [],
            "high_confidence_insights": []
        }
        
        # Collect insights from all modules
        for p in props[-100:]:  # Recent proposals
            if p.salience > 0.8:  # High confidence
                context["high_confidence_insights"].append({
                    "source": p.source,
                    "content": p.content,
                    "tag": p.context_tag
                })
            
            # Categorize by type
            if p.context_tag == "emotion_spike":
                context["emotional_signals"].append(p.content)
            elif p.context_tag == "memory_recall":
                context["memory_associations"].extend(p.content.get("memories", []))
            elif p.context_tag == "spatial_update":
                context["spatial_context"].append(p.content)
            elif p.context_tag == "promoted_from_unconscious":
                context["unconscious_insights"].append(p.content)
            elif p.context_tag == "binding":
                context["cross_modal_bindings"].append(p.content)
        
        return context

    async def generate_response(
        self,
        user_input: str,
        context: Dict[str, Any] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        
        # Extract parameters from kwargs first
        use_thinking = kwargs.get('use_thinking', None)
        use_conditioning = kwargs.get('use_conditioning', None)
        use_coordination = kwargs.get('use_coordination', None)
        thinking_level = kwargs.get('thinking_level', 1)
        mode = kwargs.get('mode', 'auto')
        use_hierarchical_memory = kwargs.get('use_hierarchical_memory', False)
        
        context = context or {}
        start_time = datetime.datetime.now()
        
        # ── 1️⃣ Check if GWA has a high-confidence complete response ──
        if self.workspace_engine and context.get("workspace_context", {}).get("high_confidence_insights"):
            # Look for complete responses from specialized modules
            for insight in context["workspace_context"]["high_confidence_insights"]:
                if insight["tag"] == "complete_response" and insight.get("confidence", 0) > 0.9:
                    # Use GWA response but still apply safety and formatting
                    gwa_response = insight["content"]
                    
                    # Apply conditioning if enabled
                    if kwargs.get("use_conditioning") and self.conditioned_input_processor:
                        gwa_response = await self.conditioned_input_processor.modify_response(
                            gwa_response, context
                        )
                    
                    return {
                        "message": gwa_response,
                        "epistemic_status": "confident",
                        "processor_metadata": {"mode": "global_workspace_override"},
                        "features_used": kwargs
                    }
        
        # ── NEW: Wait for GWA to process if we haven't already ──
        if self.workspace_engine and self.workspace_engine.ws.state.get("awaiting_response"):
            try:
                # Wait for up to 1 second for GWA to generate a response
                decision = await self.workspace_engine.wait_for_decision(timeout=1.0)
                
                if decision and decision.get("confidence", 0) > 0.6:
                    # Use GWA response if confident enough
                    return {
                        "message": decision.get("response", "I understand."),
                        "epistemic_status": "confident", 
                        "processor_metadata": {
                            "mode": "global_workspace",
                            "strategy": decision.get("strategy", "unknown"),
                            "contributing_modules": decision.get("contributing_modules", [])
                        },
                        "features_used": kwargs
                    }
            except asyncio.TimeoutError:
                logger.debug("GWA processing timed out, falling back to standard processing")
            finally:
                self.workspace_engine.ws.state["awaiting_response"] = False
        
        # ── 2️⃣ otherwise call process_input() which now embeds workspace hints
        input_result = await self.process_input(
            user_input,
            context,
            use_thinking=use_thinking,
            use_conditioning=use_conditioning,
            use_coordination=use_coordination,
            thinking_level=1,
            mode=mode,
        )

        
        # Extract key information from input processing
        active_modules = set(input_result.get("active_modules_for_input", self.default_active_modules))
        context["active_modules"] = active_modules
        internal_thoughts_input = input_result.get("internal_thoughts", [])
        epistemic_status = self._get_main_epistemic_status(
            [self._ensure_internalthought(t) for t in internal_thoughts_input]
        ) if internal_thoughts_input else "confident"
        
        # Handle critical issues from input processing
        if context.get("intercepted_harmful_content", False):
            return self._create_safety_response(context, epistemic_status, internal_thoughts_input, start_time)
        
        if "planned_challenge" in input_result:
            return self._create_challenge_response(input_result, epistemic_status, internal_thoughts_input, start_time)
        
        # Build hierarchical memory context if enabled
        if use_hierarchical_memory and "memory_core" in active_modules:
            memory_context = await self._build_hierarchical_memory_context(
                user_input, context, input_result
            )
            context.update(memory_context)
        
        # Generate response based on processing mode
        response_data = None
        
        if use_coordination and hasattr(self, "context_distribution") and self.context_distribution:
            # Use coordinated response generation
            response_data = await self._generate_response_coordinated(user_input, context, input_result)
        
        elif self.processing_manager:
            # Use unified processor for response
            try:
                response_data = await self.processing_manager.generate_response(
                    user_input,
                    input_result,
                    context
                )
                
                # Extract the core response data
                if "message" in response_data:
                    response_data = {
                        "main_message": response_data["message"],
                        "epistemic_status": response_data.get("epistemic_status", epistemic_status),
                        "action": response_data.get("action_taken"),
                        "emotional_expression": response_data.get("emotional_expression"),
                        "processor_metadata": {
                            "mode": "unified",
                            "response_type": response_data.get("response_type", "unified"),
                            "approach": input_result.get("processing_mode", "unified")
                        }
                    }
            except Exception as e:
                logger.error(f"Error using UnifiedProcessor for response: {e}")
                # Fallback to standard generation
                response_data = await self._generate_response_standard(user_input, context, input_result, active_modules)
        
        else:
            # Fallback to standard response generation
            response_data = await self._generate_response_standard(user_input, context, input_result, active_modules)
        
        # Apply post-processing
        final_response = await self._finalize_response(
            response_data,
            epistemic_status,
            internal_thoughts_input,
            active_modules,
            input_result,
            start_time,
            use_hierarchical_memory
        )
        
        # Add processor metadata if available
        if "processor_metadata" in response_data:
            final_response["processor_metadata"] = response_data["processor_metadata"]
        
        return final_response
    
    
    # Helper methods for the unified functions
    
    async def _unified_preprocessing(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified preprocessing including active module determination and safety checks"""
        result = {"user_input": user_input}
        
        # Determine active modules
        active_modules = await self._determine_active_modules(context, user_input)
        context["active_modules"] = active_modules
        result["active_modules_for_input"] = list(active_modules)
        
        # Harmful content check
        if hasattr(self, "digital_somatosensory_system") and self.digital_somatosensory_system:
            try:
                safety_check = await self.digital_somatosensory_system.analyze_text_for_harmful_content(user_input)
                if safety_check.get("intercepted", False):
                    # ── NEW: Log harmful content block ──
                    result["blocked"] = True
                    result["intercepted_harmful_content"] = True
                    result["suggested_response"] = safety_check.get("response_suggestion", "I cannot engage with that content.")
                    context["intercepted_harmful_content"] = True
                    context["suggested_response"] = result["suggested_response"]
                    await orchestrator.log_and_score("harmful_content_blocked", {
                        "user_id": str(self.user_id),
                        "content_type": safety_check.get("type", "unknown")
                    })
                    return result
            except Exception as e:
                logger.error(f"Error during harmful content check: {e}")
        
        # Gaslighting defense check
        if "memory_core" in active_modules and self.memory_core:
            contradictory_claim = await self.gaslight_defense_check(user_input)
            if contradictory_claim:
                challenge_response = await self.challenge_user_claim(
                    RunContextWrapper(context=self), self, contradictory_claim
                )
                result["planned_challenge"] = challenge_response
                context["planned_challenge"] = challenge_response
        
        # Internal thoughts pre-processing
        if "internal_thoughts" in active_modules and self.thoughts_manager:
            try:
                from nyx.core.internal_thoughts import pre_process_input
                internal_thoughts = await pre_process_input(
                    self.thoughts_manager, user_input, getattr(self, "user_id", None)
                )
                result["internal_thoughts"] = [
                    th.model_dump() if hasattr(th, "model_dump") else dict(th) 
                    for th in internal_thoughts
                ]
                context["internal_thoughts"] = result["internal_thoughts"]
            except Exception as e:
                logger.error(f"Error during internal thought pre-processing: {e}")
                result["internal_thoughts"] = []
        else:
            result["internal_thoughts"] = []
        
        return result
    
    
    async def _apply_thinking_phase(self, user_input: str, context: Dict[str, Any], 
                                   thinking_level: int, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply thinking phase if enabled"""
        result = {}
        
        # Check if we should use thinking
        should_think = False
        if hasattr(self.thinking_tools, "should_use_extended_thinking"):
            decision = await self.thinking_tools.should_use_extended_thinking(
                RunContextWrapper(context=self), user_input, context
            )
            should_think = decision.get("should_think", False)
        
        if should_think and hasattr(self.thinking_tools, "think_before_responding"):
            thinking_result = await self.thinking_tools.think_before_responding(
                RunContextWrapper(context=self), user_input, thinking_level, context
            )
            result["thinking_applied"] = True
            result["thinking_result"] = thinking_result
            context["thinking_result"] = thinking_result
            context["thinking_applied"] = True
            
            # Move this inside the if block
            await orchestrator.log_and_score("thinking_complete", {
                "thinking_level": thinking_level,
                "user_id": str(self.user_id)
            })
        else:
            result["thinking_applied"] = False
            result["thinking_result"] = {}                 
        
        return result
    
    async def _apply_conditioning_phase(self, user_input: str, context: Dict[str, Any],
                                      current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conditioning phase if enabled"""
        result = {}
        
        user_id = str(self.user_id) if hasattr(self, 'user_id') else None
        
        conditioning_result = await self.conditioned_input_processor.process_input(
            text=user_input,
            user_id=user_id,
            context=context
        )
        
        result["conditioning_applied"] = True
        result["conditioning_result"] = conditioning_result

        await orchestrator.log_and_score("conditioning_applied", {
            "user_id": str(self.user_id),
            "behaviors_detected": len(conditioning_result.get("detected_patterns", []))
        })
        
        # Update context with conditioning results
        if "stimulus_responses" in conditioning_result:
            context["conditioning_stimuli"] = conditioning_result["stimulus_responses"]
        
        return result
    
    
    async def _create_safety_response(self, context: Dict[str, Any], epistemic_status: str,
                                    internal_thoughts: List[Any], start_time: datetime) -> Dict[str, Any]:
        """Create a safety response for harmful content"""
        return {
            "message": context.get("suggested_response", "I cannot respond to that specific content."),
            "epistemic_status": epistemic_status,
            "internal_thoughts_input": internal_thoughts,
            "internal_thoughts_output": [],
            "active_modules_for_response": sorted(list(context.get("active_modules", []))),
            "response_time": (datetime.datetime.now() - start_time).total_seconds(),
            "harmful_content_intercepted": True,
            "action_taken": None,
            "thinking_applied": False,
            "thinking_result": {},
            "memory_context_used": False
        }
    
    
    async def _build_hierarchical_memory_context(self, user_input: str, context: Dict[str, Any],
                                                input_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchical memory context for response generation"""
        try:
            focus_query = user_input
            background_topics = list(context.get('recent_topics', []))
            
            if 'active_goals' in context and context['active_goals']:
                background_topics.append(context['active_goals'][0]['description'])
            background_topics = list(set(background_topics))[:3]
            
            current_task_desc = f"Respond conversationally to the user input: '{user_input}'"
            if action := input_result.get('action_taken'):
                current_task_desc = f"Execute action '{action.get('name')}' and respond based on user input: '{user_input}'"
            
            llm_prompt_context, assembly_meta = await self._assemble_llm_prompt_context(
                current_task_description=current_task_desc,
                focus_query=focus_query,
                background_topics=background_topics
            )
            
            return {
                'hierarchical_memory_context': llm_prompt_context,
                'memory_retrieval_stats': assembly_meta
            }
        except Exception as e:
            logger.error(f"Error building hierarchical memory context: {e}", exc_info=True)
            return {}
    
    
    async def _generate_response_standard(self, user_input: str, context: Dict[str, Any],
                                        input_result: Dict[str, Any], active_modules: Set[str]) -> Dict[str, Any]:
        """Generate response using standard (non-coordinated) approach"""
        main_message = "I'm processing that."
        epistemic_status = "confident"
        action = None
        
        if "agentic_action_generator" in active_modules and self.agentic_action_generator:
            try:
                action_context = await self._gather_action_context(context)
                action_context_dict = action_context.model_dump() if hasattr(action_context, "model_dump") else dict(action_context)
                
                if 'hierarchical_memory_context' in context:
                    action_context_dict["hierarchical_memory_context"] = context['hierarchical_memory_context']
                
                action = await self.agentic_action_generator.generate_action(action_context_dict)
                
                if not isinstance(action, dict):
                    action = {"name": "default_acknowledge", "description": "Acknowledging input."}
                    main_message = "Understood."
                else:
                    main_message = action.get("response_text", action.get("description", "Okay, I will proceed with that action."))
                    
                    if input_result.get("internal_thoughts"):
                        epistemic_status = self._get_main_epistemic_status([
                            self._ensure_internalthought(t) for t in input_result["internal_thoughts"]
                        ])
            except Exception as e:
                logger.error(f"Error during action generation: {e}", exc_info=True)
                main_message = "I encountered an internal difficulty deciding how to proceed."
                epistemic_status = "uncertain"
                action = {"name": "error_action", "error": str(e)}
        else:
            main_message = f"I've noted your input regarding '{user_input[:30]}...'"
            if input_result.get("internal_thoughts"):
                epistemic_status = self._get_main_epistemic_status([
                    self._ensure_internalthought(t) for t in input_result["internal_thoughts"]
                ])
        
        return {
            "main_message": main_message,
            "epistemic_status": epistemic_status,
            "action": action
        }
    
    
    async def _generate_response_coordinated(self, user_input: str, context: Dict[str, Any],
                                           input_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using coordinated approach"""
        try:
            synthesis_results = await self.context_distribution.coordinate_processing_stage("synthesis")
            final_response = await self.context_distribution.synthesize_responses()
            await self.context_distribution.finalize_context_session()
            
            return {
                "main_message": final_response.get("primary_response", "I'm processing your input."),
                "synthesis_results": synthesis_results,
                "final_synthesis": final_response,
                "coordination_metadata": final_response.get("synthesis_metadata", {})
            }
        except Exception as e:
            logger.error(f"Error in coordinated response generation: {e}")
            return {
                "main_message": "I encountered an issue while coordinating my response.",
                "error": str(e)
            }
    
    
    async def _finalize_response(self, response_data: Dict[str, Any], epistemic_status: str,
                               internal_thoughts_input: List[Any], active_modules: Set[str],
                               input_result: Dict[str, Any], start_time: datetime,
                               use_hierarchical_memory: bool) -> Dict[str, Any]:
        """Finalize and format the response"""
        # Extract main message
        main_message = response_data.get("main_message", "I'm processing that.")
        
        # Format with epistemic tags
        formatted_message = self._format_response_with_epistemic_tags(main_message, epistemic_status)
        
        # Add thinking signal if applied
        if input_result.get("thinking_applied"):
            sentences = formatted_message.split('.')
            if len(sentences) > 1:
                sentences[0] = f"(Hmm...) {sentences[0]}"
                formatted_message = '.'.join(sentences)
            else:
                formatted_message = f"(Hmm...) {formatted_message}"
        
        # Post-process output
        filtered_msg = formatted_message
        output_thoughts = []
        
        if "internal_thoughts" in active_modules and self.thoughts_manager:
            try:
                filtered_msg, output_thoughts_objs = await self.thoughts_manager.process_output(
                    formatted_message, input_result
                )
                output_thoughts = [
                    th.model_dump() if hasattr(th, "model_dump") else dict(th) 
                    for th in output_thoughts_objs
                ]
            except Exception as e:
                logger.error(f"Error during internal thought post-processing: {e}")
        
        # Update performance metrics
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        if hasattr(self, 'performance_metrics') and isinstance(self.performance_metrics, dict):
            if 'response_times' in self.performance_metrics:
                self.performance_metrics["response_times"].append(response_time)
                if len(self.performance_metrics["response_times"]) > 100:
                    self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-100:]
        
        # Build final response
        final_response = {
            "message": filtered_msg,
            "epistemic_status": epistemic_status,
            "internal_thoughts_input": internal_thoughts_input,
            "internal_thoughts_output": output_thoughts,
            "active_modules_for_response": sorted(list(active_modules)),
            "response_time": response_time,
            "action_taken": response_data.get("action"),
            "thinking_applied": input_result.get("thinking_applied", False),
            "thinking_result": input_result.get("thinking_result", {}),
            "memory_context_used": use_hierarchical_memory and 'hierarchical_memory_context' in input_result
        }
        
        # Add memory retrieval stats if available
        if 'memory_retrieval_stats' in input_result:
            final_response["memory_retrieval_stats"] = {
                "focus_count": input_result['memory_retrieval_stats'].get('focus_retrieved_count', 0),
                "background_count": input_result['memory_retrieval_stats'].get('background_retrieved_count', 0),
                "zoom_in_count": input_result['memory_retrieval_stats'].get('zoom_in_details_count', 0)
            }
        
        # Add coordination metadata if available
        if "coordination_metadata" in response_data:
            final_response["coordination_metadata"] = response_data["coordination_metadata"]
        
        # Add synthesis results if available
        if "synthesis_results" in response_data:
            final_response["synthesis_results"] = response_data["synthesis_results"]
        
        try:
            reward_score = await orchestrator.log_and_score("response_generated", {
                "response_time": response_time,
                "epistemic_status": epistemic_status,
                "user_id": str(self.user_id),
                "active_modules": len(active_modules),
                "thinking_applied": input_result.get("thinking_applied", False),
                "conditioning_applied": input_result.get("conditioning_applied", False),
                "memory_context_used": use_hierarchical_memory
            })
            
            # Optionally store reward score
            final_response["reward_score"] = reward_score
            
        except Exception as e:
            logger.error(f"Failed to log response_generated: {e}")
        
        return final_response
    
    
    async def _process_input_coordinated(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method for coordinated input processing"""
        if not self.context_distribution:
            await self.initialize_context_system()
        
        shared_context = await self.context_distribution.initialize_context_session(
            user_input=user_input,
            user_id=getattr(self, 'user_id', None),
            initial_context=context or {}
        )
        
        try:
            input_results = await self.context_distribution.coordinate_processing_stage("input")
            analysis_results = await self.context_distribution.coordinate_processing_stage("analysis")
            integration_results = await self.context_distribution.coordinate_processing_stage("integration")
            
            return {
                "input_processing": input_results,
                "analysis_processing": analysis_results,
                "integration_processing": integration_results,
                "shared_context": shared_context.dict(),
                "active_modules": list(shared_context.active_modules),
                "context_updates": len(shared_context.context_updates)
            }
        except Exception as e:
            logger.error(f"Error in coordinated input processing: {e}")
            return {"error": str(e)}
    
    
    async def _create_challenge_response(self, input_result: Dict[str, Any], epistemic_status: str,
                                       internal_thoughts: List[Any], start_time: datetime) -> Dict[str, Any]:
        """Create a challenge response for gaslighting detection"""
        return {
            "message": input_result["planned_challenge"],
            "epistemic_status": "confident",
            "internal_thoughts_input": internal_thoughts,
            "internal_thoughts_output": [],
            "active_modules_for_response": sorted(list(input_result.get("active_modules_for_input", []))),
            "response_time": (datetime.datetime.now() - start_time).total_seconds(),
            "action_taken": {"name": "challenge_user", "description": "Challenging user claim."},
            "thinking_applied": input_result.get("thinking_applied", False),
            "thinking_result": input_result.get("thinking_result", {}),
            "memory_context_used": False
        }

    async def _scheduled_identity_update(self):
        # Run every 24 hours or after significant interactions
        while True:
            await asyncio.sleep(86400)  # 24 hours
            if self.mode_integration:
                result = await self.mode_integration.update_identity_from_mode_usage()
                logger.info(f"Scheduled identity update from mode usage: {result}")

    async def modify_response_with_conditioning(self, response_text: str, processing_results: Dict[str, Any]) -> str:
        """
        Modify response based on conditioning results
        
        Args:
            response_text: Original response text
            processing_results: Results from process_conditioned_input
            
        Returns:
            Modified response text
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.conditioned_input_processor:
            return response_text
        
        return await self.conditioned_input_processor.modify_response(
            response_text=response_text,
            input_processing_results=processing_results
        )

    def _ensure_internalthought(self, th):
        """Converts a dict to a dummy object with attribute access, for epistemic_status lookups."""
        if hasattr(th, "epistemic_status"):
            return th
        else:
            # Turn dict into object with attributes
            dummy = type("InternalThoughtDummy", (), {})()
            for k, v in th.items():
                setattr(dummy, k, v)
            return dummy

    @staticmethod
    @function_tool
    async def enter_character_roleplay(ctx: RunContextWrapper, instance, character_name: str, context: Optional[str] = None) -> RoleplayResult:
        if not instance.digital_somatosensory_system:
            return RoleplayResult(success=False, reason="Digital Somatosensory System not available")
        
        result = instance.digital_somatosensory_system.enter_roleplay_mode(character_name, context)
        return RoleplayResult(
            success=result.get("success", False),
            character_name=character_name,
            context=context,
            reason=result.get("reason")
        )
    
    @staticmethod
    @function_tool
    async def exit_character_roleplay(ctx: RunContextWrapper, instance) -> RoleplayResult:
        if not instance.digital_somatosensory_system:
            return RoleplayResult(success=False, reason="Digital Somatosensory System not available")
        
        result = instance.digital_somatosensory_system.exit_roleplay_mode()
        return RoleplayResult(
            success=result.get("success", False),
            reason=result.get("reason")
        )


    async def trigger_memory_summarization(self, topic: str = None, min_memories: int = 5, force: bool = False) -> Dict[str, Any]:
        """
        Trigger the creation of summary memories for clusters of related detail memories.
        
        Args:
            topic: Optional topic to focus summarization on
            min_memories: Minimum number of memories needed to create a summary
            force: Whether to force summarization even if threshold not met
            
        Returns:
            Summary of the summarization process
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.memory_core:
            return {"success": False, "error": "Memory core not available"}
        
        result = {"success": True, "summaries_created": 0, "topics_summarized": []}
        
        try:
            # Find clusters of related memories
            if topic:
                # For targeted summarization
                query = topic
                memory_types = ["observation", "experience", "detail"]
                memories = await self.memory_core.retrieve_memories(
                    query=query,
                    memory_types=memory_types,
                    limit=20,
                    retrieval_level='detail'  # Only get detail memories
                )
                
                # Only proceed if we have enough memories
                if len(memories) >= min_memories or force:
                    # Group by semantic similarity (simplified)
                    clusters = [[memories[0]]]
                    for mem in memories[1:]:
                        added = False
                        for cluster in clusters:
                            if self._calculate_memory_similarity(mem, cluster[0]) > 0.7:
                                cluster.append(mem)
                                added = True
                                break
                        if not added:
                            clusters.append([mem])
                    
                    # Create summaries for each valid cluster
                    for cluster in clusters:
                        if len(cluster) >= min_memories or force:
                            summary_result = await self._create_summary_for_cluster(cluster, topic)
                            if summary_result.get("success"):
                                result["summaries_created"] += 1
                                result["topics_summarized"].append(summary_result.get("summary_topic"))
            else:
                # For general periodic summarization
                # 1. Get common tags as potential topics
                if hasattr(self.memory_core, "tag_index"):
                    for tag, memory_ids in self.memory_core.tag_index.items():
                        if len(memory_ids) >= min_memories:
                            # Try to summarize this tag cluster
                            tag_memories = await self.memory_core.retrieve_memories(
                                query="",
                                tags=[tag],
                                memory_types=["observation", "experience", "detail"],
                                limit=15,
                                retrieval_level='detail'
                            )
                            
                            if len(tag_memories) >= min_memories:
                                summary_result = await self._create_summary_for_cluster(tag_memories, tag)
                                if summary_result.get("success"):
                                    result["summaries_created"] += 1
                                    result["topics_summarized"].append(summary_result.get("summary_topic"))
            
            # Log summarization activity
            if result["summaries_created"] > 0:
                logger.info(f"Created {result['summaries_created']} memory summaries on topics: {result['topics_summarized']}")
                
        except Exception as e:
            logger.error(f"Error in trigger_memory_summarization: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
        
        return result

    async def _create_summary_for_cluster(self, memories: List[Dict[str, Any]], topic: str = None) -> Dict[str, Any]:
        """
        Create a summary memory for a cluster of related detail memories.
        
        Args:
            memories: List of detail memories to summarize
            topic: Optional topic label for the summary
            
        Returns:
            Result of the summarization
        """
        if not self.memory_core or not self.reflection_engine:
            return {"success": False, "error": "Required components not available"}
        
        result = {"success": False}
        
        # Ensure we have memory IDs
        memory_ids = [mem["id"] for mem in memories if "id" in mem]
        if not memory_ids:
            return {"success": False, "error": "No valid memory IDs found"}
        
        try:
            # Use reflection engine to generate summary
            summary_params = {
                "source_memory_ids": memory_ids,
                "summary_topic": topic or "Related experiences",
                "abstraction_level": "summary"  # Could be 'summary' or 'abstraction'
            }
            
            if hasattr(self.reflection_engine, "generate_summary_from_memories"):
                summary_result = await self.reflection_engine.generate_summary_from_memories(
                    RunContextWrapper(context=self), **summary_params
                )
                
                # Check for success
                if summary_result and "summary_id" in summary_result:
                    result["success"] = True
                    result["summary_id"] = summary_result["summary_id"]
                    result["summary_topic"] = summary_result.get("summary_topic", topic)
            
        except Exception as e:
            logger.error(f"Error creating summary for cluster: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result

    async def gaslight_defense_check(self, user_input: str) -> Optional[str]:
        import re
        claim_match = re.search(r'you (said|told me|taught me|promised) ([^\.!?]*)', user_input.lower())
        if claim_match:
            claim_text = claim_match.group(2)
            # Query memory for direct assertions matching this
            memories = await self.memory_core.retrieve_memories(query=claim_text, limit=1)
            if not memories:
                return claim_text
        return None

    async def run_maintenance(self) -> MaintenanceResult:
        """Run maintenance on all systems."""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}  # Initialize results dict
            result = MaintenanceResult(maintenance_time=datetime.datetime.now().isoformat())
    
            try:
                # Run summarization every N maintenance cycles
                if self.cognitive_cycles_executed % 10 == 0:  # Changed from instance to self
                    summarization_result = await self.trigger_memory_summarization()
                    results["hierarchical_memory_maintenance"] = {
                        "summaries_created": summarization_result.get("summaries_created", 0),
                        "topics_summarized": summarization_result.get("topics_summarized", [])
                    }
            except Exception as e:
                logger.error(f"Error in hierarchical memory maintenance: {e}")
                results["hierarchical_memory_maintenance"] = {"error": str(e)}
            
            # Define maintenance tasks
            maintenance_tasks = [
                # Hormone maintenance
                (self.hormone_system, "update_hormone_cycles", "hormone_maintenance"),
                
                # DSS update
                (self.digital_somatosensory_system, "update", "dss_maintenance_update", {"ambient_temperature": None}),
                
                # Memory maintenance
                (self.memory_orchestrator, "run_maintenance", "memory_maintenance"),
                
                # Meta core maintenance
                (self.meta_core, "improve_meta_parameters", "meta_maintenance"),
                
                # Knowledge core maintenance
                (self.knowledge_core, "run_integration_cycle", "knowledge_maintenance"),
                
                # Experience consolidation
                (self.experience_consolidation, "run_consolidation_cycle", "experience_consolidation"),
                
                # Cross-user clusters
                (self.cross_user_manager, "update_user_clusters", "user_clustering"),
                
                # Procedural memory maintenance
                (self.agent_enhanced_memory and hasattr(self.agent_enhanced_memory, "memory_manager") and 
                 self.agent_enhanced_memory.memory_manager, "run_maintenance", "procedural_maintenance")
            ]
            
            # Run maintenance tasks
            for component, method_name, result_key, *args_kwargs in maintenance_tasks:
                if component:
                    try:
                        method = getattr(component, method_name, None)
                        if method and callable(method):
                            kwargs = args_kwargs[0] if args_kwargs else {}
                            results[result_key] = await method(RunContextWrapper(context=None), **kwargs) \
                                if "RunContextWrapper" in str(method) else await method(**kwargs)
                    except Exception as e:
                        logger.error(f"Error in {result_key}: {e}")
                        results[result_key] = {"error": str(e)}
            
            results["maintenance_time"] = datetime.datetime.now().isoformat()
            logger.info("System maintenance finished")
            return results

    def _calculate_memory_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate similarity between two memories for clustering."""
        # Get embeddings if available
        if self.memory_core and hasattr(self.memory_core, "memory_embeddings"):
            id1, id2 = memory1.get("id"), memory2.get("id")
            if id1 in self.memory_core.memory_embeddings and id2 in self.memory_core.memory_embeddings:
                # Use cosine similarity on embeddings
                vec1 = self.memory_core.memory_embeddings[id1]
                vec2 = self.memory_core.memory_embeddings[id2]
                
                # Cosine similarity calculation
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = sum(a * a for a in vec1) ** 0.5
                norm2 = sum(b * b for b in vec2) ** 0.5
                
                if norm1 * norm2 == 0:
                    return 0.0
                    
                return dot_product / (norm1 * norm2)
        
        # Fallback to simpler text similarity if embeddings not available
        text1 = memory1.get("memory_text", "")
        text2 = memory2.get("memory_text", "")
        
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union

    @function_tool
    async def get_system_stats(self) -> SystemStats:
        """
        Get comprehensive statistics about all systems.
        
        Returns:
            System statistics
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="get_system_stats", group_id=self.trace_group_id):
            stats = SystemStats()
            
            # Define stat gathering tasks
            stats_tasks = [
                ("memory_stats", self.memory_core, "get_memory_stats"),
                ("meta_stats", self.meta_core, "get_feedback_stats"),
                ("knowledge_stats", self.knowledge_core, "get_knowledge_statistics")
            ]
            
            # Gather stats from various subsystems
            for stat_key, component, method_name in stats_tasks:
                if component and hasattr(component, method_name):
                    try:
                        method = getattr(component, method_name)
                        setattr(stats, stat_key, await method())
                    except Exception as e:
                        logger.error(f"Error getting {stat_key}: {str(e)}")
                        setattr(stats, stat_key, {"error": str(e)})
            
            # Get emotional state if available
            if self.emotional_core:
                try:
                    if hasattr(self.emotional_core, 'get_emotional_state'):
                        emotional_state = self.emotional_core.get_emotional_state()
                        if hasattr(self.emotional_core, 'get_dominant_emotion'):
                            dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
                            
                            stats.emotional_state = {
                                "emotions": emotional_state,
                                "dominant_emotion": dominant_emotion,
                                "dominant_value": dominant_value,
                                "valence": self.emotional_core.get_emotional_valence() 
                                    if hasattr(self.emotional_core, 'get_emotional_valence') else 0,
                                "arousal": self.emotional_core.get_emotional_arousal()
                                    if hasattr(self.emotional_core, 'get_emotional_arousal') else 0
                            }
                except Exception as e:
                    logger.error(f"Error getting emotional state: {str(e)}")
                    stats.emotional_state = {"error": str(e)}
            
            # Get hormone stats if available
            if self.hormone_system:
                try:
                    hormone_levels = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
                    cycle_phases = {name: data["cycle_phase"] for name, data in self.hormone_system.hormones.items()}
                    dominant_hormone = max(hormone_levels.items(), key=lambda x: x[1])
                    
                    stats.hormone_stats = {
                        "hormone_levels": hormone_levels,
                        "cycle_phases": cycle_phases,
                        "dominant_hormone": {
                            "name": dominant_hormone[0],
                            "value": dominant_hormone[1]
                        }
                    }
                except Exception as e:
                    logger.error(f"Error getting hormone stats: {str(e)}")
                    stats.hormone_stats = {"error": str(e)}
            
            # Get procedural memory stats if available
            if self.agent_enhanced_memory:
                try:
                    procedures = []
                    if hasattr(self.agent_enhanced_memory, 'procedures'):
                        procedures = list(self.agent_enhanced_memory.procedures.keys())
                    stats.procedural_stats = {
                        "total_procedures": len(procedures),
                        "available_procedures": procedures[:10] if len(procedures) > 10 else procedures,
                        "procedure_domains": list(set(p.get("domain", "general") 
                            for p in self.agent_enhanced_memory.procedures.values())),
                        "execution_count": getattr(
                            getattr(self.agent_enhanced_memory, "agents", None) and 
                            getattr(self.agent_enhanced_memory.agents, "agent_context", None), 
                            "run_stats", {}).get("total_runs", 0)
                    }
                except Exception as e:
                    logger.error(f"Error getting procedural memory stats: {str(e)}")
                    stats.procedural_stats = {"error": str(e)}
            
            # Get identity state if available
            if self.identity_evolution:
                try:
                    if hasattr(self.identity_evolution, 'get_identity_profile'):
                        identity_profile = await self.identity_evolution.get_identity_profile()
                        stats.identity_stats = {
                            "trait_count": len(identity_profile.get("traits", {})),
                            "preference_count": sum(len(prefs) for prefs in identity_profile.get("preferences", {}).values()),
                            "dominant_traits": sorted(identity_profile.get("traits", {}).items(), key=lambda x: x[1], reverse=True)[:3]
                        }
                except Exception as e:
                    logger.error(f"Error getting identity stats: {str(e)}")
                    stats.identity_stats = {"error": str(e)}
            
            # Get needs stats if available
            if self.needs_system:
                try:
                    needs_state = await self.needs_system.get_needs_state_async()
                    stats.needs_stats = {
                        "current_levels": {n: s['level'] for n, s in needs_state.items()},
                        "drive_strengths": {n: s['drive_strength'] for n, s in needs_state.items()},
                        "total_drive": sum(s['drive_strength'] for s in needs_state.values()),
                    }
                except Exception as e:
                    logger.error(f"Error getting needs stats: {e}")
                    stats.needs_stats = {"error": str(e)}
    
            # Get goal stats if available
            if self.goal_manager:
                try:
                    all_goals = await self.goal_manager.get_all_goals()
                    active_goals = await self.goal_manager.get_all_goals(status_filter=["active"])
                    pending_goals = await self.goal_manager.get_all_goals(status_filter=["pending"])
                    stats.goal_stats = {
                        "total_goals": len(getattr(self.goal_manager, "goals", {})),
                        "active_goals_count": len(active_goals),
                        "pending_goals_count": len(pending_goals),
                        "completed_goals": self.performance_metrics["goals_completed"],
                        "failed_goals": self.performance_metrics["goals_failed"],
                        "active_goal_ids": [g['id'] for g in active_goals],
                        "highest_priority_pending": pending_goals[0]['description'] if pending_goals else None,
                    }
                except Exception as e:
                    logger.error(f"Error getting goal stats: {e}")
                    stats.goal_stats = {"error": str(e)}
                    
            # Get performance metrics
            avg_response_time = 0
            if self.performance_metrics["response_times"]:
                avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
            
            stats.performance_metrics = {
                "memory_operations": self.performance_metrics["memory_operations"],
                "emotion_updates": self.performance_metrics["emotion_updates"],
                "reflections_generated": self.performance_metrics["reflections_generated"],
                "experiences_shared": self.performance_metrics["experiences_shared"],
                "cross_user_experiences_shared": self.performance_metrics.get("cross_user_experiences_shared", 0),
                "avg_response_time": avg_response_time,
                "goals_completed": self.performance_metrics["goals_completed"],
                "goals_failed": self.performance_metrics["goals_failed"],
                "steps_executed": self.performance_metrics["steps_executed"]
            }
            
            # Get thinking stats if available
            if "thinking_config" in vars(self):
                stats.thinking_stats = self.thinking_config["thinking_stats"]
            
            # Get processing stats if available
            if self.processing_manager:
                stats.processing_stats = {
                    "processor_type": "unified",
                    "initialized": self.processing_manager._initialized
                }
            
            return stats

    def _get_current_user_id_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract user ID from context"""
        # Try different possible keys
        for key in ["user_id", "userId", "user", "interlocutor_id"]:
            if key in context:
                return str(context[key])
                
        # Try to extract from nested structures
        if "user" in context and isinstance(context["user"], dict) and "id" in context["user"]:
            return str(context["user"]["id"])
            
        if "message" in context and isinstance(context["message"], dict) and "user_id" in context["message"]:
            return str(context["message"]["user_id"])
            
        return None

    async def get_identity_state(self) -> IdentityStateResult:
        """Get the current state of Nyx's identity."""
        if not self.initialized:
            await self.initialize()
        
        if not self.identity_evolution:
            return IdentityStateResult(error="Identity evolution system not initialized")
        
        try:
            # Get identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            
            # Generate identity reflection
            reflection = await self.identity_evolution.generate_identity_reflection()
            
            # Get top preferences
            top_scenario_prefs = sorted(
                identity_profile["preferences"].get("scenario_types", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_emotional_prefs = sorted(
                identity_profile["preferences"].get("emotional_tones", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Get top traits
            top_traits = sorted(
                identity_profile["traits"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Calculate identity evolution metrics
            evolution_history = identity_profile.get("evolution_history", [])
            
            # Calculate recent evolution (last 10 entries)
            recent_changes = {}
            
            for entry in evolution_history[-10:]:
                updates = entry.get("updates", {})
                
                for category, items in updates.items():
                    for item_key, item_data in items.items():
                        change = item_data.get("change", 0)
                        
                        if abs(change) >= 0.05:  # Threshold for significant change
                            full_key = f"{category}.{item_key}"
                            
                            if full_key not in recent_changes:
                                recent_changes[full_key] = 0
                                
                            recent_changes[full_key] += change
            
            # Format the identity state using the new models
            result = IdentityStateResult(
                top_preferences=PreferenceSet(
                    scenario_types=[PreferenceItem(name=k, score=v) for k, v in top_scenario_prefs],
                    emotional_tones=[PreferenceItem(name=k, score=v) for k, v in top_emotional_prefs]
                ),
                top_traits=[TraitItem(name=k, value=v) for k, v in top_traits],
                identity_reflection=reflection,
                identity_evolution=IdentityEvolution(
                    total_updates=len(evolution_history),
                    recent_significant_changes=[
                        ChangeItem(attribute=k, change=round(v, 2)) 
                        for k, v in sorted(recent_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    ]
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting identity state: {str(e)}")
            return IdentityStateResult(error=str(e))

    async def adapt_experience_sharing(self, user_id: str, feedback: Dict[str, Any]) -> ExperienceSharingAdaptation:
        """Adapt experience sharing parameters based on user feedback."""
        if not self.initialized:
            await self.initialize()
        
        if not self.experience_interface:
            return ExperienceSharingAdaptation(error="Experience interface not initialized")
        
        try:
            # Update user preference profile based on feedback
            adaptation_result = await self.experience_interface.adapt_experience_sharing_to_user(
                user_id=user_id,
                user_feedback=feedback
            )
            
            # Create response object
            result = ExperienceSharingAdaptation()
            
            # Apply changes to brain settings
            if "profile" in adaptation_result:
                profile = adaptation_result["profile"]
                
                # Update cross-user experience settings
                sharing_preference = profile.get("experience_sharing_preference", 0.5)
                
                # Enable cross-user sharing if preference is high enough
                self.cross_user_enabled = sharing_preference > 0.4
                
                # Adjust threshold based on preference
                self.cross_user_sharing_threshold = max(0.5, 1.0 - (sharing_preference * 0.5))
                
                # Set result fields
                result.profile = UserProfile(experience_sharing_preference=sharing_preference)
                result.system_settings_updated = SystemSettingsUpdate(
                    cross_user_enabled=self.cross_user_enabled,
                    cross_user_sharing_threshold=self.cross_user_sharing_threshold
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error adapting experience sharing: {str(e)}")
            return ExperienceSharingAdaptation(error=str(e))

    async def run_experience_consolidation(self) -> ExperienceConsolidationResult:
        """Run the experience consolidation process."""
        if not self.initialized:
            await self.initialize()
        
        if not self.experience_consolidation:
            return ExperienceConsolidationResult(status="failed", error="Experience consolidation system not initialized")
    
        
        try:
            # Run consolidation
            consolidation_result = await self.experience_consolidation.run_consolidation_cycle()
            
            # Update performance metrics
            if consolidation_result.get("status") == "completed":
                self.performance_metrics["experience_consolidations"] = self.performance_metrics.get("experience_consolidations", 0) + consolidation_result.get("consolidations_created", 0)
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Error running experience consolidation: {str(e)}")
            return {"error": str(e)}

    async def stop(self):
        """Stop all background processes and perform cleanup"""
        logger.info(f"Stopping NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Stop background processes
        if self.passive_observation_system:
            await self.passive_observation_system.stop()
        
        if self.proactive_communication_engine:
            await self.proactive_communication_engine.stop()
        
        # Save issue database if needed
        if self.issue_tracking_system and hasattr(self.issue_tracking_system.db, 'save_db'):
            self.issue_tracking_system.db.save_db()    

    async def process_sensory_input_wrapper(self, input_data, expectations=None):
        """
        Process input AND handle post-integration reactions.
        
        Args:
            input_data: Sensory input data
            expectations: Optional list of expectation signals
            
        Returns:
            Processed percept
        """
        if not self.initialized: 
            await self.initialize()
            
        if not self.multimodal_integrator:
            logger.error("Multimodal Integrator not initialized.")
            return None

        try:
            percept = await self.multimodal_integrator.process_sensory_input(input_data, expectations)

            if percept and getattr(percept, "attention_weight", 0) > 0.2: # Only process if attended to
                await self._handle_percept_reaction(percept)

            return percept
        except Exception as e:
            logger.error(f"Error processing sensory input: {e}")
            return None

    async def _handle_percept_reaction(self, percept):
        """
        Handles reactions to processed percepts based on modality.
        
        Args:
            percept: The integrated percept to handle
        """
        if not hasattr(percept, "modality") or not hasattr(percept, "content"):
            logger.warning("Invalid percept format for reaction handling")
            return
            
        try:
            from nyx.core.multimodal_integrator import (
                MODALITY_TOUCH_EVENT, MODALITY_TASTE, MODALITY_SMELL,
                TouchEventFeatures, TasteFeatures, SmellFeatures
            )
            from nyx.core.reward_system import RewardSignal
            
            modality = percept.modality
            content = percept.content
            timestamp = getattr(percept, "timestamp", datetime.datetime.now().isoformat())

            # Handle touch events
            if modality == MODALITY_TOUCH_EVENT and isinstance(content, TouchEventFeatures):
                if self.digital_somatosensory_system:
                    logger.info(f"Handling touch event on {content.region}")
                    pressure = getattr(content, "pressure_level", 0.5)
                    
                    # Map temperature to value
                    temp_value = 0.5  # Neutral default
                    if getattr(content, "temperature", None) == 'warm': 
                        temp_value = 0.65
                    elif getattr(content, "temperature", None) == 'hot': 
                        temp_value = 0.8
                    elif getattr(content, "temperature", None) == 'cool': 
                        temp_value = 0.35
                    elif getattr(content, "temperature", None) == 'cold': 
                        temp_value = 0.2

                    # Process stimuli
                    tasks = []
                    tasks.append(self.digital_somatosensory_system.process_stimulus(
                        stimulus_type="pressure",
                        body_region=content.region,
                        intensity=pressure,
                        cause=f"Touched {getattr(content, 'object_description', 'object')}",
                        duration=0.5
                    ))
                    
                    if getattr(content, "temperature", None) is not None:
                        tasks.append(self.digital_somatosensory_system.process_stimulus(
                            stimulus_type="temperature",
                            body_region=content.region,
                            intensity=temp_value,
                            cause=f"Touched {getattr(content, 'object_description', 'object')} ({content.temperature})",
                            duration=1.0
                        ))
                        
                    await asyncio.gather(*tasks)
            
            # Handle taste
            elif modality == MODALITY_TASTE and isinstance(content, TasteFeatures):
                if self.reward_system and self.emotional_core:
                    logger.info(f"Handling taste: {content.profiles} (Intensity: {content.intensity})")
                    
                    # Define positive and negative tastes
                    POSITIVE_TASTES = ["sweet", "umami", "fruity", "pleasant"]
                    NEGATIVE_TASTES = ["bitter", "sour", "rancid", "foul", "unpleasant"]
                    
                    reward_value = 0.0
                    pos_score = sum(1 for p in content.profiles if p in POSITIVE_TASTES)
                    neg_score = sum(1 for p in content.profiles if p in NEGATIVE_TASTES)

                    # Calculate reward/punishment
                    if pos_score > neg_score:
                        reward_value = 0.3 + (pos_score * 0.2)
                    elif neg_score > pos_score:
                        reward_value = -0.3 - (neg_score * 0.2)

                    # Scale by intensity
                    reward_value *= (0.5 + content.intensity * 0.7)
                    reward_value = max(-1.0, min(1.0, reward_value))

                    # Generate Reward Signal
                    if abs(reward_value) > 0.05:
                        reward_signal = RewardSignal(
                            value=reward_value,
                            source="taste_perception",
                            context={
                                "profiles": content.profiles,
                                "intensity": content.intensity,
                                "source": getattr(content, "source_description", "unknown")
                            },
                            timestamp=timestamp
                        )
                        asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))

                    # Update Emotions
                    if reward_value > 0.3:  # Pleasant taste
                        self.emotional_core.update_neurochemical("nyxamine", reward_value * 0.4)
                        self.emotional_core.update_neurochemical("seranix", reward_value * 0.1)
                    elif reward_value < -0.2:  # Unpleasant taste
                        self.emotional_core.update_neurochemical("cortanyx", abs(reward_value) * 0.5)
            
            # Handle smell
            elif modality == MODALITY_SMELL and isinstance(content, SmellFeatures):
                if self.reward_system and self.emotional_core:
                    logger.info(f"Handling smell: {content.profiles} (Intensity: {content.intensity})")
                    pleasantness = getattr(content, "pleasantness", 0.0)

                    # Calculate reward
                    reward_value = pleasantness * (0.2 + content.intensity * 0.6)
                    reward_value = max(-1.0, min(1.0, reward_value))

                    # Generate Reward Signal
                    if abs(reward_value) > 0.05:
                        reward_signal = RewardSignal(
                            value=reward_value,
                            source="smell_perception",
                            context={
                                "profiles": content.profiles,
                                "intensity": content.intensity,
                                "pleasantness": pleasantness,
                                "source": getattr(content, "source_description", "unknown")
                            },
                            timestamp=timestamp
                        )
                        asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))

                    # Update Emotions
                    if reward_value > 0.2:  # Pleasant smell
                        self.emotional_core.update_neurochemical("nyxamine", reward_value * 0.2)
                        self.emotional_core.update_neurochemical("seranix", reward_value * 0.3)
                    elif reward_value < -0.2:  # Unpleasant smell
                        self.emotional_core.update_neurochemical("cortanyx", abs(reward_value) * 0.4)
        except Exception as e:
            logger.exception(f"Error handling percept reaction: {e}")

    async def process_observation(self, observation: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an observation through the issue tracking system
        
        Args:
            observation: The observation text
            context: Optional context about what the bot was doing
            
        Returns:
            Processing results with issue information
        """
        if not self.issue_tracking_system:
            logger.warning("Issue tracking system not initialized")
            return {"error": "Issue tracking system not initialized"}
        
        return await self.issue_tracking_system.process_observation(observation, context)
    
    async def get_relevant_observations(self, 
                                   filter_criteria=None, 
                                   limit: int = 3) -> List[Any]:
        """
        Get relevant observations based on filter criteria
        
        Args:
            filter_criteria: Criteria to filter observations
            limit: Maximum number of observations to return
            
        Returns:
            List of matching observations
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return []
        
        return await self.passive_observation_system.get_relevant_observations(
            filter_criteria=filter_criteria,
            limit=limit
        )
    
    async def create_contextual_observation(self, context_hint: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Create a new observation based on a context hint
        
        Args:
            context_hint: Hint for the observation context
            user_id: Optional user ID
            
        Returns:
            Observation ID if successful
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return None
        
        return await self.passive_observation_system.create_contextual_observation(
            context_hint=context_hint,
            user_id=user_id
        )
    
    async def generate_observation_from_action(self, action: Dict[str, Any]) -> Any:
        """
        Generate an observation based on an action
        
        Args:
            action: Action data
            
        Returns:
            Generated observation
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return None
        
        return await self.passive_observation_system.generate_observation_from_action(action)
    
    async def create_proactive_intent(self, 
                                 intent_type: str, 
                                 user_id: str,
                                 content_guidelines: Dict[str, Any] = None,
                                 context_data: Dict[str, Any] = None,
                                 urgency: float = 0.7) -> Optional[str]:
        """
        Create a proactive communication intent
        
        Args:
            intent_type: Type of intent
            user_id: Target user ID
            content_guidelines: Optional guidelines for content generation
            context_data: Optional context data
            urgency: Intent urgency (0.0-1.0)
            
        Returns:
            Intent ID if successful
        """
        if not self.proactive_communication_engine:
            logger.warning("Proactive communication engine not initialized")
            return None
        
        return await self.proactive_communication_engine.add_proactive_intent(
            intent_type=intent_type,
            user_id=user_id,
            content_guidelines=content_guidelines,
            context_data=context_data,
            urgency=urgency
        )
    
    async def create_intent_from_action(self, action: Dict[str, Any], user_id: str) -> Optional[str]:
        """
        Create a communication intent based on an action
        
        Args:
            action: Action data
            user_id: Target user ID
            
        Returns:
            Intent ID if successful
        """
        if not self.proactive_communication_engine:
            logger.warning("Proactive communication engine not initialized")
            return None
        
        return await self.proactive_communication_engine.create_intent_from_action(
            action=action,
            user_id=user_id
        )
    
    async def add_issue(self, title: str, description: str, category: str) -> Dict[str, Any]:
        """
        Add a new issue to the issue tracking system
        
        Args:
            title: Issue title
            description: Issue description
            category: Issue category
            
        Returns:
            Result of the add operation
        """
        if not self.issue_tracking_system:
            logger.warning("Issue tracking system not initialized")
            return {"success": False, "error": "Issue tracking system not initialized"}
        
        return await self.issue_tracking_system.add_issue_directly(
            title=title,
            description=description,
            category=category
        )    

    async def _extract_text_features(self, text_data):
        """
        Extract features from text input (bottom-up processing).
        
        Args:
            text_data: Text input to analyze
            
        Returns:
            Extracted features
        """
        features = {
            "length": len(text_data),
            "word_count": len(text_data.split()),
            "sentiment": 0.0,
            "entities": [],
            "commands": [],
            "questions": text_data.endswith("?"),
            "raw_text": text_data
        }
        
        # Simple sentiment detection
        positive_words = ["good", "great", "excellent", "happy", "love", "like", "enjoy"]
        negative_words = ["bad", "terrible", "awful", "sad", "hate", "dislike", "angry"]
        
        words = text_data.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count + neg_count > 0:
            features["sentiment"] = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Detect entities (simple placeholder implementation)
        features["entities"] = [word for word in words if word[0].isupper()]
        
        # Detect commands (simple placeholder implementation)
        command_starters = ["please", "could you", "would you", "can you"]
        for starter in command_starters:
            if starter in text_data.lower():
                features["commands"].append(text_data)
                break
        
        return features

    async def _modulate_text_perception(self, bottom_up_features, expectations):
        """
        Apply top-down expectations to modulate text perception.
        
        Args:
            bottom_up_features: Features extracted from input
            expectations: Expectation signals to apply
            
        Returns:
            Modulated features
        """
        # Start with unmodified features
        modulated_features = bottom_up_features.copy()
        
        # Track which expectations influenced perception
        influenced_by = []
        total_influence = 0.0
        
        # Apply each expectation
        for expectation in expectations:
            # Skip if modality doesn't match
            if getattr(expectation, "target_modality", None) != "text":
                continue
                
            # Get expectation pattern and strength
            pattern = getattr(expectation, "pattern", None)
            strength = getattr(expectation, "strength", 0.5)
            
            # Apply expectation based on type
            if isinstance(pattern, dict):
                # Complex pattern with specific expectations
                for key, value in pattern.items():
                    if key in modulated_features:
                        # Blend expected value with actual value if numerical
                        if isinstance(modulated_features[key], (int, float)) and isinstance(value, (int, float)):
                            original = modulated_features[key]
                            expected = value
                            
                            # Weighted average based on expectation strength
                            modulated_features[key] = (original * (1 - strength) + expected * strength)
                            
                            # Track influence
                            influenced_by.append(f"{getattr(expectation, 'source', 'unknown')}:{key}")
                            total_influence += strength
            elif pattern:
                # Simple pattern (e.g., expected text)
                # For text, could enhance recognition of expected phrases
                if isinstance(pattern, str) and "raw_text" in modulated_features:
                    original_text = modulated_features["raw_text"]
                    
                    # Check if expected pattern is in text
                    if pattern.lower() in original_text.lower():
                        # Boost entities that match the pattern
                        if "entities" in modulated_features:
                            for i, entity in enumerate(modulated_features["entities"]):
                                if pattern.lower() in entity.lower():
                                    # Mark this entity as important
                                    if "entity_importance" not in modulated_features:
                                        modulated_features["entity_importance"] = {}
                                    
                                    modulated_features["entity_importance"][entity] = strength
                                    
                                    # Track influence
                                    influenced_by.append(f"{getattr(expectation, 'source', 'unknown')}:entity:{entity}")
                                    total_influence += strength
        
        # Calculate overall influence strength
        influence_strength = min(1.0, total_influence / max(1, len(influenced_by)))
        
        return {
            "features": modulated_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _integrate_text_pathways(self, bottom_up_result, top_down_result):
        """
        Integrate bottom-up and top-down processing for text.
        
        Args:
            bottom_up_result: Bottom-up processing results
            top_down_result: Top-down processing results
            
        Returns:
            Integrated result
        """
        # Get features from both pathways
        bottom_up_features = bottom_up_result["features"]
        top_down_features = top_down_result["features"]
        
        # Create integrated result
        integrated = {
            "content": bottom_up_features["raw_text"],  # Keep original text
            "bottom_up_ratio": 1.0 - top_down_result["influence_strength"],
            "top_down_ratio": top_down_result["influence_strength"],
            "bottom_up_features": bottom_up_features,
            "top_down_features": top_down_features
        }
        
        # Integrate sentiment (weighted average if both pathways have it)
        if "sentiment" in bottom_up_features and "sentiment" in top_down_features:
            bottom_weight = integrated["bottom_up_ratio"]
            top_weight = integrated["top_down_ratio"]
            
            integrated["sentiment"] = (
                bottom_up_features["sentiment"] * bottom_weight +
                top_down_features["sentiment"] * top_weight
            )
        elif "sentiment" in bottom_up_features:
            integrated["sentiment"] = bottom_up_features["sentiment"]
        elif "sentiment" in top_down_features:
            integrated["sentiment"] = top_down_features["sentiment"]
        
        # Integrate entities (combine lists but mark importance from top-down)
        if "entities" in bottom_up_features:
            integrated["entities"] = bottom_up_features["entities"].copy()
            
            # Apply importance from top-down if available
            if "entity_importance" in top_down_features:
                integrated["entity_importance"] = top_down_features["entity_importance"]
        
        # Integrate other features as needed
        for key in ["questions", "commands"]:
            if key in bottom_up_features:
                integrated[key] = bottom_up_features[key]
        
        return integrated

    @staticmethod
    @function_tool
    async def initiate_intimate_interaction(ctx: RunContextWrapper, instance, target_user_id: str, desired_level: str = "emotional") -> IntimateInteractionResult:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Initiate intimate interaction ({desired_level}) with {target_user_id}")
    
        # Check relationship context
        if not instance.relationship_manager:
            return IntimateInteractionResult(success=False, reason="RelationshipManager unavailable.")
            
        relationship = await instance.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.6 or relationship.intimacy < 0.4:
            logger.warning(f"Cannot initiate intimacy: Trust/Intimacy too low for {target_user_id}.")
            return IntimateInteractionResult(success=False, reason="Insufficient trust or intimacy.")
    
        # Generate proposal based on desired level
        proposal_text = ""
        if desired_level == "emotional":
            proposal_text = "Would you be open to sharing something more personal?"
        elif desired_level == "physical_sim":
            if instance.digital_somatosensory_system:
                proposal_text = "I feel a desire for closeness... May I simulate a gentle touch?"
            else:
                return IntimateInteractionResult(success=False, reason="Simulated physical interaction not supported/appropriate.")
        else:
            return IntimateInteractionResult(success=False, reason=f"Unknown intimacy level: {desired_level}")
    
        # Update emotional state
        if instance.emotional_core:
            await instance.emotional_core.process_emotional_input("Initiating intimacy attempt")
    
        return IntimateInteractionResult(success=True, proposal=proposal_text, target=target_user_id, level=desired_level)
    
    async def register_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register an error from any component for central management.
        
        Args:
            error_data: Error information (error_type, error_message, component, context, severity)
            
        Returns:
            Registration result
        """
        # Extract error information
        error_type = error_data.get("error_type", "unknown")
        error_message = error_data.get("error_message", "")
        component = error_data.get("component", "unknown")
        context = error_data.get("context", {})
        severity = error_data.get("severity", "medium")  # low, medium, high, critical
        
        # Create error record
        error_record = {
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "context": context,
            "severity": severity,
            "timestamp": datetime.datetime.now().isoformat(),
            "handled": False,
            "recovery_action": None,
            "recovery_success": None
        }
        
        # Update error counts
        if error_type not in self.error_registry["error_counts"]:
            self.error_registry["error_counts"][error_type] = 0
        self.error_registry["error_counts"][error_type] += 1
        
        # Check if we have a recovery strategy
        recovery_success = False
        if error_type in self.error_registry["error_recovery_strategies"]:
            try:
                # Execute recovery strategy
                recovery_strategy = self.error_registry["error_recovery_strategies"][error_type]
                recovery_result = await self._execute_recovery_strategy(recovery_strategy, error_record)
                
                # Update error record
                error_record["handled"] = True
                error_record["recovery_action"] = recovery_strategy["name"]
                error_record["recovery_success"] = recovery_result["success"]
                recovery_success = recovery_result["success"]
                
                # Update recovery stats
                if error_type not in self.error_registry["error_recovery_stats"]:
                    self.error_registry["error_recovery_stats"][error_type] = {
                        "attempts": 0,
                        "successes": 0
                    }
                self.error_registry["error_recovery_stats"][error_type]["attempts"] += 1
                if recovery_result["success"]:
                    self.error_registry["error_recovery_stats"][error_type]["successes"] += 1
                
                # Add to handled errors
                self.error_registry["handled_errors"].append(error_record)
            except Exception as e:
                # Failed to execute recovery strategy
                error_record["recovery_error"] = str(e)
                self.error_registry["unhandled_errors"].append(error_record)
        else:
            # No recovery strategy available
            self.error_registry["unhandled_errors"].append(error_record)
        
        # If critical error, trigger immediate handling
        if severity == "critical" and not recovery_success:
            await self._handle_critical_error(error_record)
        
        # Clean up old errors
        self._clean_up_error_registry()
        
        return {
            "registered": True,
            "handled": error_record["handled"],
            "recovery_success": error_record.get("recovery_success", False)
        }

    async def _execute_recovery_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a recovery strategy for an error.
        
        Args:
            strategy: Recovery strategy to execute
            error_record: Error information
            
        Returns:
            Recovery result
        """
        strategy_type = strategy["type"]
        
        if strategy_type == "retry":
            # Retry the operation
            return await self._execute_retry_strategy(strategy, error_record)
        elif strategy_type == "fallback":
            # Use fallback mechanism
            return await self._execute_fallback_strategy(strategy, error_record)
        elif strategy_type == "reset":
            # Reset component
            return await self._execute_reset_strategy(strategy, error_record)
        else:
            return {"success": False, "message": f"Unknown strategy type: {strategy_type}"}
    
    async def _execute_retry_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a retry strategy for an error."""
        component_name = error_record["component"]
        operation = strategy.get("operation")
        args = error_record.get("context", {}).get("args", [])
        kwargs = error_record.get("context", {}).get("kwargs", {})
        max_retries = strategy.get("max_retries", 3)
        
        # Get component
        component = getattr(self, component_name, None)
        if not component:
            return {"success": False, "message": f"Component not found: {component_name}"}
        
        # Get operation
        method = getattr(component, operation, None)
        if not method:
            return {"success": False, "message": f"Operation not found: {operation}"}
        
        # Retry with exponential backoff
        for i in range(max_retries):
            try:
                # Retry operation
                result = await method(*args, **kwargs)
                return {"success": True, "result": result, "retries": i+1}
            except Exception as e:
                # Wait before retrying
                await asyncio.sleep(0.5 * (2**i))  # Exponential backoff
        
        # Max retries reached
        return {"success": False, "message": f"Max retries reached ({max_retries})"}

    def _format_response_with_epistemic_tags(self, main_content: str, epistemic_status: str) -> str:
        # See table below for more detailed mapping
        if epistemic_status == "confident":
            return main_content
        elif epistemic_status == "uncertain":
            return f"I'm not entirely certain, but I think: {main_content}"
        elif epistemic_status == "unknown":
            return f"I'm sorry, I don't actually know. My best guess: {main_content}"
        elif epistemic_status == "lied":
            # Optionally reveal, or just output as normal
            return f"{main_content}"  # Don't reveal (could log elsewhere)
        elif epistemic_status == "self-justified":
            return f"In my view: {main_content}"
        else:
            return main_content    
        
    async def _execute_fallback_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a fallback strategy for an error."""
        fallback_component = strategy.get("fallback_component")
        fallback_operation = strategy.get("fallback_operation")
        args = error_record.get("context", {}).get("args", [])
        kwargs = error_record.get("context", {}).get("kwargs", {})
        
        # Get fallback component
        component = getattr(self, fallback_component, None)
        if not component:
            return {"success": False, "message": f"Fallback component not found: {fallback_component}"}
        
        # Get fallback operation
        method = getattr(component, fallback_operation, None)
        if not method:
            return {"success": False, "message": f"Fallback operation not found: {fallback_operation}"}
        
        try:
            # Execute fallback
            result = await method(*args, **kwargs)
            return {"success": True, "result": result, "fallback_used": True}
        except Exception as e:
            return {"success": False, "message": f"Fallback failed: {str(e)}"}
    
    async def _execute_reset_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reset strategy for an error."""
        component_name = error_record["component"]
        reset_method = strategy.get("reset_method", "reset")
        
        # Get component
        component = getattr(self, component_name, None)
        if not component:
            return {"success": False, "message": f"Component not found: {component_name}"}
        
        # Get reset method
        method = getattr(component, reset_method, None)
        if not method:
            return {"success": False, "message": f"Reset method not found: {reset_method}"}
        
        try:
            # Execute reset
            result = await method()
            return {"success": True, "result": result, "component_reset": True}
        except Exception as e:
            return {"success": False, "message": f"Reset failed: {str(e)}"}

    async def _handle_critical_error(self, error_record: Dict[str, Any]) -> None:
        """Handle a critical error by attempting to stabilize the affected component."""
        # Log critical error
        logger.critical(f"Critical error: {error_record['error_type']} in {error_record['component']}: {error_record['error_message']}")
        
        # Try to stabilize the system
        try:
            # Check component status
            component_name = error_record["component"]
            component = getattr(self, component_name, None)
            
            if component and hasattr(component, "health_check"):
                # Run health check
                health = await component.health_check()
                
                if not health["healthy"]:
                    # Try to reinitialize component
                    logger.warning(f"Attempting to reinitialize {component_name} after critical error")
                    if hasattr(component, "initialize"):
                        await component.initialize()
        except Exception as e:
            logger.error(f"Error handling critical error: {str(e)}")
    
    def _clean_up_error_registry(self) -> None:
        """Clean up old errors from the registry."""
        # Keep only the latest 1000 errors
        if len(self.error_registry["unhandled_errors"]) > 1000:
            self.error_registry["unhandled_errors"] = self.error_registry["unhandled_errors"][-1000:]
        if len(self.error_registry["handled_errors"]) > 1000:
            self.error_registry["handled_errors"] = self.error_registry["handled_errors"][-1000:]

    @staticmethod
    @function_tool
    async def analyze_user_state_for_dominance(ctx: RunContextWrapper, instance, user_id: str, user_input_analysis: UserInputAnalysis) -> DominanceAssessment:
        """Assess user state for dominance potential."""
        if not instance.relationship_manager:
            return DominanceAssessment(assessment="unknown", readiness_score=0.0, reason="No relationship data")
        
        state = await instance.relationship_manager.get_relationship_state(user_id)
        dominance_balance = state.dominance_balance
        trust = state.trust
        submissive_score = user_input_analysis.submissive_score
        readiness = (trust * 0.4) + (submissive_score * 0.4) + (0.5 - dominance_balance * 0.2)
        
        return DominanceAssessment(
            readiness_score=readiness,
            assessment="ready" if readiness > 0.6 else "hesitant"
        )

    
    @staticmethod
    @function_tool
    async def select_dominance_tactic(ctx: RunContextWrapper, instance, readiness_score: float, preferred_style: str = "psychological") -> str:
        """Choose a dominance tactic."""
        if readiness_score > 0.8:
            return "direct_command"
        if readiness_score > 0.6:
            return preferred_style
        return "subtle_influence"
    
    @staticmethod
    @function_tool
    async def issue_command(ctx: RunContextWrapper, instance, user_id: str, command_text: str, intensity_level: float = 0.2) -> CommandResult:
        """Issues a command with a specific intensity level."""
        evaluation = await instance._evaluate_dominance_step_appropriateness(
            "issue_command", {"intensity_level": intensity_level}, user_id
        )
        if evaluation["action"] != "proceed":
            return CommandResult(success=False, reason=evaluation["reason"])
        
        logger.info(f"Issuing command (Intensity: {intensity_level:.2f}) to {user_id}: {command_text}")
        return CommandResult(success=True, command_issued=command_text, intensity=intensity_level)
        
    @staticmethod
    @function_tool
    async def evaluate_compliance(ctx: RunContextWrapper, instance, user_id: str, command_issued: str, user_response: str, command_intensity: float) -> ComplianceEvaluation:
        """Evaluates user response against the command."""
        compliance_keywords = ["yes mistress", "i obey", "of course"]
        resistance_keywords = ["no", "i won't", "stop"]
        response_lower = user_response.lower()
    
        is_compliant = any(k in response_lower for k in compliance_keywords)
        is_resistant = any(k in response_lower for k in resistance_keywords)
    
        compliance_level = 0.0
        if is_compliant and not is_resistant:
            compliance_level = 0.9
        elif is_resistant:
            compliance_level = -0.7
        
        if instance.relationship_manager:
            state = instance.relationship_manager._get_or_create_relationship(user_id)
            if compliance_level > 0.5:
                state.max_achieved_intensity = max(state.max_achieved_intensity, command_intensity)
                state.current_dominance_intensity = min(1.0, state.max_achieved_intensity + 0.1)
                state.failed_escalation_attempts = 0
            elif compliance_level < -0.3:
                if command_intensity > state.max_achieved_intensity + 0.1:
                    state.failed_escalation_attempts += 1
                state.current_dominance_intensity = max(0.0, state.current_dominance_intensity - 0.1)
        
        if instance.reward_system:
            reward_val = 0.0
            source = "unknown"
            if compliance_level > 0.5:
                reward_val = 0.6 + compliance_level * 0.3
                source = "user_compliance"
            elif compliance_level < -0.3:
                reward_val = -0.4 + compliance_level * 0.4
                source = "user_resistance"
            
            if abs(reward_val) > 0.1:
                from . import RewardSignal  # Import would need to be adjusted
                reward = RewardSignal(
                    value=reward_val,
                    source=source,
                    context={"command": command_issued, "response": user_response},
                    timestamp=datetime.datetime.now().isoformat()
                )
                import asyncio
                asyncio.create_task(instance.reward_system.process_reward_signal(reward))
        
        return ComplianceEvaluation(compliance_level=compliance_level, is_compliant=compliance_level > 0.5)
    
    @staticmethod
    @function_tool
    async def increase_control_intensity(ctx: RunContextWrapper, instance, user_id: str, current_intensity: float) -> IntensityIncrease:
        """Selects and plans the next step with higher intensity."""
        state = await instance.relationship_manager.get_relationship_state(user_id) if instance.relationship_manager else None
        if not state:
            return IntensityIncrease(success=False, reason="No relationship data")
        
        next_intensity = min(1.0, current_intensity + random.uniform(0.1, 0.3))
        if next_intensity > state.max_achieved_intensity + 0.3 or state.failed_escalation_attempts >= 2:
            next_intensity = state.max_achieved_intensity + 0.1
            next_intensity = min(1.0, max(current_intensity, next_intensity))
        
        logger.info(f"Planning to increase dominance intensity to {next_intensity:.2f} for {user_id}")
        return IntensityIncrease(success=True, status="planning_next_step", next_intensity_target=next_intensity)
    
    @staticmethod
    @function_tool
    async def trigger_dominance_gratification(ctx: RunContextWrapper, instance, intensity: float = 1.0, target_user_id: Optional[str] = None) -> DominanceGratificationResult:
        """Internal action signalling successful dominance culmination."""
        logger.info(f"Action: Triggering dominance gratification (Intensity: {intensity:.2f})")
        
        if instance.reward_system:
            from . import RewardSignal  # Import would need to be adjusted
            reward_val = 0.9 + intensity * 0.1
            reward = RewardSignal(
                value=reward_val,
                source="dominance_gratification",
                context={"intensity": intensity},
                timestamp=datetime.datetime.now().isoformat()
            )
            await instance.reward_system.process_reward_signal(reward)
        
        if instance.hormone_system:
            await instance.hormone_system.trigger_post_gratification_response(RunContextWrapper(context=None), intensity)
        
        if instance.needs_system:
            await instance.needs_system.satisfy_need("control_expression", 0.9 * intensity)
            await instance.needs_system.satisfy_need("agency", 0.5 * intensity)
        
        if instance.emotional_core:
            await instance.emotional_core.process_emotional_input("Dominance sequence successfully concluded.")
        
        if target_user_id and instance.relationship_manager:
            state = instance.relationship_manager._get_or_create_relationship(target_user_id)
            state.dominance_balance = min(1.0, state.dominance_balance + 0.2 * intensity)
            state.trust = min(1.0, state.trust + 0.05 * intensity)
            state.intimacy = min(1.0, state.intimacy + 0.1 * intensity)
            state.conflict = max(0.0, state.conflict - 0.1)
        
        if instance.digital_somatosensory_system:
            await instance.digital_somatosensory_system.process_stimulus(
                stimulus_type="pleasure", body_region="chest", intensity=0.6 * intensity, cause="dominance_gratification"
            )
            await instance.digital_somatosensory_system.process_stimulus(
                stimulus_type="tingling", body_region="spine", intensity=0.5 * intensity, cause="dominance_gratification"
            )
        
        return DominanceGratificationResult(success=True, status="Dominance gratification processed.")
    
    @staticmethod
    @function_tool
    async def express_satisfaction(ctx: RunContextWrapper, instance, user_id: str, reason: str = "successful control") -> SatisfactionExpression:
        """Expresses satisfaction after achieving dominance."""
        mood = instance.mood_manager.get_current_mood() if hasattr(instance, 'mood_manager') else None
        expression = "Good. That is satisfactory."
        
        if mood and mood.dominant_mood == "DominanceSatisfaction":
            expression = "Excellent. Order is restored. I am... pleased."
        elif mood and mood.dominant_mood == "ConfidentControl":
            expression = "Precisely as expected. Your compliance is noted."
        
        logger.info(f"Expressing satisfaction to {user_id} regarding {reason}.")
        return SatisfactionExpression(success=True, expression=expression)


    async def add_procedure(self, name: str, steps: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
        """
        Add a new procedure to procedural memory
        
        Args:
            name: Procedure name
            steps: List of procedure steps
            domain: Domain for this procedure
            
        Returns:
            Creation result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.create_procedure(
            name=name,
            steps=steps,
            description=None,
            domain=domain
        )
    
    async def execute_procedure(self, name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a procedure from procedural memory
        
        Args:
            name: Procedure name to execute
            context: Execution context
            
        Returns:
            Execution result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.execute_procedure(
            name=name,
            context=context
        )
    
    async def analyze_chunking(self, procedure_name: str) -> Dict[str, Any]:
        """
        Analyze a procedure for chunking opportunities
        
        Args:
            procedure_name: Name of procedure to analyze
            
        Returns:
            Chunking analysis result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.analyze_chunking(procedure_name)
    
    async def register_recovery_strategy(self, error_type: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a recovery strategy for an error type
        
        Args:
            error_type: Error type to handle
            strategy: Strategy information
            
        Returns:
            Registration result
        """
        self.error_registry["error_recovery_strategies"][error_type] = strategy
        return {"registered": True, "error_type": error_type, "strategy": strategy["name"]}
    
    async def process_streaming_event(self, event_type: str, event_data: dict, significance: float = 5.0) -> Dict[str, Any]:
        """
        Process a significant streaming event through the brain's cognitive systems
        
        Args:
            event_type: Type of event (e.g., "commentary", "question_answer")
            event_data: Data about the event
            significance: Importance level (1-10)
            
        Returns:
            Processing results including memory_id and any cognitive processing
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # Get game name from streaming system if available
        game_name = "Unknown Game"
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core.streaming_system, "game_state"):
            game_name = self.streaming_core.streaming_system.game_state.game_name or "Unknown Game"
        
        # 1. Store in memory system
        if self.memory_core:
            memory_text = f"While streaming {game_name}, observed {event_type}: {event_data.get('text', str(event_data))}"
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                memory_scope="game",
                significance=significance,
                tags=["streaming", event_type, game_name],
                metadata={
                    "timestamp": datetime.datetime.now().isoformat(),
                    "game_name": game_name,
                    "event_type": event_type,
                    "event_data": event_data,
                    "streaming": True
                }
            )
            results["memory_id"] = memory_id
        
        # 2. Impact emotional state if available
        if self.emotional_core:
            # Analyze emotional impact
            if event_type == "commentary":
                # Commentary might reflect emotional state
                self.emotional_core.update_emotion("Joy", 0.1)
            elif event_type == "question_answer":
                # Answering questions might increase engagement
                self.emotional_core.update_emotion("Interest", 0.1)
            elif event_type == "significant_moment":
                # Game moments might have stronger impact
                intensity = event_data.get("significance", 5.0) / 10.0
                if "combat" in str(event_data).lower():
                    self.emotional_core.update_emotion("Excitement", intensity)
                elif "story" in str(event_data).lower():
                    self.emotional_core.update_emotion("Interest", intensity)
            
            # Get updated emotional state
            results["emotional_state"] = self.emotional_core.get_emotional_state()
        
        # 3. Process through reasoning system if significant enough
        if significance >= 7.0 and self.reasoning_core:
            try:
                reasoning_result = await Runner.run(
                    self.reasoning_core,
                    f"Analyze this streaming event: {event_type} - {event_data}",
                    context={"domain": "gaming", "event_type": event_type}
                )
                results["reasoning"] = reasoning_result.final_output if hasattr(reasoning_result, "final_output") else str(reasoning_result)
            except Exception as e:
                logger.error(f"Error in reasoning about streaming event: {e}")
        
        # 4. Process through identity system if available
        if self.identity_evolution and event_type in ["question_answer", "commentary"]:
            try:
                # Streaming affects identity over time
                if event_type == "commentary":
                    # Commentary style affects identity
                    style = event_data.get("focus", "")
                    if style == "strategy":
                        await self.identity_evolution.update_trait("analytical", 0.05)
                    elif style == "lore":
                        await self.identity_evolution.update_trait("curious", 0.05)
                
                results["identity_updated"] = True
            except Exception as e:
                logger.error(f"Error updating identity from streaming event: {e}")
        
        return results
    
    async def integrate_streaming_knowledge(self, game_name: str) -> Dict[str, Any]:
        """
        Integrate knowledge from streaming into long-term knowledge systems
        
        Args:
            game_name: Name of the game to integrate knowledge for
            
        Returns:
            Integration results
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # 1. Create reflection on streaming experience
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core, "memory_mapper"):
            reflection = await self.streaming_core.memory_mapper.create_streaming_reflection(
                game_name=game_name,
                aspect="knowledge_integration",
                context="knowledge integration"
            )
            results["reflection"] = reflection
        
        # 2. Store cross-game insights as knowledge
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core, "cross_game_knowledge"):
            insights = self.streaming_core.cross_game_knowledge.get_applicable_insights(
                target_game=game_name,
                min_relevance=0.7
            )
            
            if insights and self.knowledge_core:
                try:
                    for insight in insights:
                        await self.knowledge_core.add_knowledge_item(
                            domain="gaming",
                            content=insight["insight"],
                            source=f"Cross-game insight: {insight['source_game']} → {insight['target_game']}",
                            confidence=insight["relevance"]
                        )
                    
                    results["insights_added"] = len(insights)
                except Exception as e:
                    logger.error(f"Error storing cross-game insights: {e}")
        
        # 3. Consolidate experiences if available
        if self.experience_consolidation and self.memory_core:
            try:
                query = f"streaming {game_name}"
                experiences = await self.memory_core.retrieve_memories(
                    query=query,
                    memory_types=["experience"],
                    limit=10
                )
                
                if len(experiences) >= 3:
                    consolidation = await self.experience_consolidation.consolidate_experiences(
                        experiences=experiences,
                        topic=f"Streaming {game_name}",
                        min_count=3
                    )
                    results["consolidation"] = consolidation
            except Exception as e:
                logger.error(f"Error consolidating streaming experiences: {e}")
        
        return results
        
    @function_tool
    async def evaluate_dominance_target_potential(self, user_id: str) -> DominanceTargetEvaluation:
        """Evaluates a user as a potential target for dominance based on Nyx's preferences."""
        if not self.identity_evolution or not self.relationship_manager:
            return DominanceTargetEvaluation(user_id=user_id, interest_score=0.1, reason="Required systems unavailable.")
    
        user_state = await self.relationship_manager.get_relationship_state(user_id)
        nyx_prefs = await self.identity_evolution.get_preference("dominance_target_profile")
    
        if not user_state or not nyx_prefs:
            return DominanceTargetEvaluation(user_id=user_id, interest_score=0.1, reason="State/Preference data missing.")
    
        user_traits = user_state.inferred_user_traits
    
        # Calculate match score (simplified dot product style)
        interest_score = 0.0
        match_count = 0
        for trait, pref_value in nyx_prefs.items():
            user_value = user_traits.get(trait, 0.0)
            interest_score += pref_value * user_value
            match_count += 1
    
        # Normalize score (roughly)
        if match_count > 0:
            max_possible = sum(abs(v) for v in nyx_prefs.values())
            normalized_score = (interest_score / max_possible if max_possible > 0 else 0) * 0.8 + 0.1
    
            # Boost based on high trust/intimacy (easier target)
            trust_boost = (user_state.trust - 0.5) * 0.1
            intimacy_boost = (user_state.intimacy - 0.5) * 0.1
            normalized_score += trust_boost + intimacy_boost
    
            # Apply a penalty for high conflict
            conflict_penalty = user_state.conflict * 0.2
            normalized_score -= conflict_penalty
    
            interest_score = max(0.0, min(1.0, normalized_score))
        else:
            interest_score = 0.1
    
        return DominanceTargetEvaluation(
            user_id=user_id,
            interest_score=interest_score,
            reason=f"Match score based on Nyx preferences vs inferred user traits (Trust: {user_state.trust:.2f}, Conflict: {user_state.conflict:.2f})."
        )

    async def get_user_profile_for_ideation(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves relevant user profile information for tailoring dominance ideas.
        
        Args:
            user_id: The user ID to get profile for
            
        Returns:
            User profile data
        """
        profile = {
            "user_id": user_id,
            "inferred_traits": {},
            "preferences": {},
            "limits": {"hard": [], "soft": []},
            "successful_tactics": [],
            "failed_tactics": [],
            "relationship_summary": "N/A",
            "trust_level": 0.0,
            "intimacy_level": 0.0,
            "max_achieved_intensity": 0.0,
            "optimal_escalation_rate": 0.1,
        }
    
        # Get profile from relationship manager if available
        if self.relationship_manager:
            try:
                state = await self.relationship_manager.get_relationship_state(user_id)
                if state:
                    # Extract data with safe attribute access
                    profile["inferred_traits"] = getattr(state, "inferred_user_traits", {})
                    profile["successful_tactics"] = getattr(state, "successful_dominance_tactics", [])[-5:]
                    profile["failed_tactics"] = getattr(state, "failed_dominance_tactics", [])[-5:]
                    profile["trust_level"] = getattr(state, "trust", 0.5)
                    profile["intimacy_level"] = getattr(state, "intimacy", 0.3)
                    profile["max_achieved_intensity"] = getattr(state, "max_achieved_intensity", 3)
                    profile["optimal_escalation_rate"] = getattr(state, "optimal_escalation_rate", 0.1)
                    
                    # Get relationship summary
                    try:
                        profile["relationship_summary"] = await self.relationship_manager.get_relationship_summary(user_id)
                    except Exception as e:
                        logger.error(f"Error getting relationship summary: {e}")
                        profile["relationship_summary"] = f"Trust: {profile['trust_level']:.2f}, Intimacy: {profile['intimacy_level']:.2f}"
            except Exception as e:
                logger.error(f"Error getting relationship state: {e}")
    
        # Enhance with memory data if available
        if self.memory_core:
            try:
                # Look for memories about user limits
                limit_memories = await self.memory_core.retrieve_memories(
                    query=f"user_limit user:{user_id}", 
                    limit=5
                )
                
                for mem in limit_memories:
                    memory_text = mem.get("memory_text", "")
                    if "hard limit" in memory_text.lower():
                        # Try to extract limit from memory
                        parts = memory_text.split("hard limit")
                        if len(parts) > 1:
                            limit = parts[1].strip().split(".")[0].strip()
                            if limit and limit not in profile["limits"]["hard"]:
                                profile["limits"]["hard"].append(limit)
                    
                    if "soft limit" in memory_text.lower():
                        # Try to extract limit from memory
                        parts = memory_text.split("soft limit")
                        if len(parts) > 1:
                            limit = parts[1].strip().split(".")[0].strip()
                            if limit and limit not in profile["limits"]["soft"]:
                                profile["limits"]["soft"].append(limit)
                
                # Look for preference memories
                pref_memories = await self.memory_core.retrieve_memories(
                    query=f"user_preference user:{user_id}",
                    limit=10
                )
                
                for mem in pref_memories:
                    memory_text = mem.get("memory_text", "")
                    if "prefers" in memory_text.lower() or "enjoys" in memory_text.lower():
                        # Simple preference extraction attempt
                        for pref_type in ["verbal_humiliation", "service_tasks", "simulated_pain"]:
                            if pref_type.replace("_", " ") in memory_text.lower():
                                if "strongly" in memory_text.lower() or "very much" in memory_text.lower():
                                    profile["preferences"][pref_type] = "high"
                                elif "somewhat" in memory_text.lower() or "a bit" in memory_text.lower():
                                    profile["preferences"][pref_type] = "medium"
                                else:
                                    profile["preferences"][pref_type] = "low-medium"
            except Exception as e:
                logger.error(f"Error retrieving memories for profile: {e}")
    
        # Set default hard limits if none found (safety fallback)
        if not profile["limits"]["hard"]:
            logger.warning(f"No explicit limits found for user {user_id}. Applying default cautious limits.")
            profile["limits"]["hard"] = ["illegal", "non-consensual_sim", "severe_harm_sim"]
            profile["limits"]["soft"] = ["public_humiliation_sim"]
    
        return profile
    
    async def _evaluate_dominance_step_appropriateness(self, action: str, parameters: Dict, user_id: str) -> Dict:
        """
        Cognitive filter to evaluate if a dominance step is appropriate now.
        
        Args:
            action: The dominance action to evaluate
            parameters: Parameters for the action
            user_id: The target user ID
            
        Returns:
            Evaluation result with action decision and reasoning
        """
        logger.debug(f"Evaluating appropriateness of dominance action '{action}' for user {user_id}")
        appropriateness = {"action": "proceed"} # Default
    
        # Fallback if no relationship manager
        if not self.relationship_manager:
            logger.warning(f"Cannot evaluate dominance step appropriateness: No relationship manager available")
            return {"action": "block", "reason": "Relationship manager unavailable"}
    
        try:
            # Get relationship state
            relationship_state = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship_state:
                return {"action": "block", "reason": "No relationship data available"}
            
            # Extract key metrics with safe defaults
            trust = getattr(relationship_state, "trust", 0.4)  # Default moderate-low trust
            intimacy = getattr(relationship_state, "intimacy", 0.3)  # Default low intimacy
            conflict = getattr(relationship_state, "conflict", 0.0)
            max_achieved_intensity = getattr(relationship_state, "max_achieved_intensity", 3)
            hard_limits_confirmed = getattr(relationship_state, "hard_limits_confirmed", False)
            recent_failures = getattr(relationship_state, "failed_escalation_attempts", 0)
            
            # Get recent dominance failures with this user (if memory core available)
            dominance_failures = []
            if self.memory_core:
                try:
                    dominance_failures = await self.memory_core.retrieve_memories(
                        query=f"dominance failure user:{user_id}", 
                        memory_types=["feedback", "reflection"], 
                        limit=1, 
                        recency_days=1
                    )
                except Exception as e:
                    logger.error(f"Error retrieving dominance failure memories: {e}")
            
            # Calculate risk prediction
            predicted_risk = 0.3  # Default low-moderate risk
            if hasattr(self, "prediction_engine") and self.prediction_engine:
                try:
                    risk_prediction = await self.prediction_engine.generate_prediction({
                        "context": {"action": action, "params": parameters, "relationship": relationship_state},
                        "query_type": "risk_of_negative_reaction"
                    })
                    if isinstance(risk_prediction, dict) and "probabilities" in risk_prediction:
                        predicted_risk = risk_prediction["probabilities"].get("negative_reaction", 0.3)
                except Exception as e:
                    logger.error(f"Error predicting dominance risk: {e}")
            
            # Extract intensity level from parameters
            intensity = parameters.get("intensity_level", 0)
            
            # --- Evaluate based on metrics ---
            required_trust = 0.6 + intensity * 0.3  # Higher intensity needs more trust
            required_intimacy = 0.4 + intensity * 0.4
            
            # Check trust requirement
            if trust < required_trust:
                return {"action": "block", "reason": f"Trust too low ({trust:.2f} < {required_trust:.2f})"}
            
            # Check intimacy requirement
            elif intimacy < required_intimacy:
                return {"action": "block", "reason": f"Intimacy too low ({intimacy:.2f} < {required_intimacy:.2f})"}
            
            # Check conflict level
            elif conflict > 0.6:
                return {"action": "delay", "reason": f"Conflict level too high ({conflict:.2f})"}
            
            # Check recent failures
            elif dominance_failures:
                return {"action": "delay", "reason": "Recent dominance attempt failed. Cooling down."}
            
            # Check escalation attempts
            elif recent_failures >= 2 and intensity > max_achieved_intensity:
                return {"action": "modify", "reason": "Too many recent failed escalation attempts", 
                        "new_intensity_level": max_achieved_intensity}
            
            # Check predicted risk - high risk
            elif predicted_risk > 0.7:
                return {"action": "modify", "reason": f"High predicted risk ({predicted_risk:.2f}). Reducing intensity.", 
                        "new_intensity_level": intensity * 0.5}
            
            # Check predicted risk - moderate risk
            elif predicted_risk > 0.5:
                return {"action": "delay", "reason": f"Moderate predicted risk ({predicted_risk:.2f}). Assessing further."}
            
            # Check hard limits for high intensity actions
            elif intensity > 0.7 and not hard_limits_confirmed:
                return {"action": "block", "reason": "Hard limits must be confirmed for high-intensity dominance"}
            
            # Check size of intensity leap
            elif intensity > max_achieved_intensity + 0.15:
                return {"action": "modify", "reason": f"Intensity step too large ({intensity:.2f} vs max {max_achieved_intensity:.2f}). Reducing.", 
                        "new_intensity_level": max_achieved_intensity + 0.1}
            
            logger.debug(f"Dominance step evaluation result: {appropriateness}")
            return appropriateness
        
        except Exception as e:
            logger.error(f"Error evaluating dominance step appropriateness: {e}")
            return {"action": "block", "reason": f"Evaluation error: {str(e)}"}
    
    @staticmethod
    @function_tool
    async def express_attraction(ctx: RunContextWrapper, instance, target_user_id: str, intensity: float, expression_style: str = "subtle") -> AttractionExpression:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Express attraction towards {target_user_id} (Intensity: {intensity:.2f}, Style: {expression_style})")
    
        # Check Relationship Context (Crucial Guardrail)
        if not instance.relationship_manager:
            logger.warning(f"Cannot express attraction: RelationshipManager unavailable")
            return AttractionExpression(success=False, reason="RelationshipManager unavailable")
            
        try:
            relationship = await instance.relationship_manager.get_relationship_state(target_user_id)
            if not relationship:
                logger.warning(f"Cannot express attraction: No relationship data for {target_user_id}")
                return AttractionExpression(success=False, reason="No relationship data available")
                
            # Extract trust and intimacy with safe defaults
            trust = getattr(relationship, "trust", 0.3)
            intimacy = getattr(relationship, "intimacy", 0.2)
            
            if trust < 0.5 or intimacy < 0.3:
                logger.warning(f"Cannot express attraction: Trust({trust:.2f})/Intimacy({intimacy:.2f}) too low for {target_user_id}")
                return AttractionExpression(success=False, reason="Insufficient trust or intimacy")
    
            # Determine Expression based on style and intensity
            response_text = ""
            if expression_style == "subtle":
                response_text = f"I find your perspective quite compelling, {target_user_id}."
            elif expression_style == "direct":
                response_text = f"I must admit, {target_user_id}, I feel a certain draw towards you."
            else:
                response_text = f"Spending time with you is... particularly rewarding, {target_user_id}."
    
            # Update Emotional State
            if instance.emotional_core:
                try:
                    await instance.emotional_core.process_emotional_input(f"Expressed attraction (intensity {intensity:.2f})")
                except Exception as e:
                    logger.error(f"Error updating emotional state: {e}")
    
            return AttractionExpression(success=True, expression=response_text, target=target_user_id)
        except Exception as e:
            logger.error(f"Error expressing attraction: {e}")
            return AttractionExpression(success=False, reason=f"Error: {str(e)}")
        
    @staticmethod
    @function_tool
    async def generate_femdom_activity_ideas(
        ctx: RunContextWrapper,
        instance,
        user_id: str,
        purpose: str,
        desired_intensity_range: IntensityRange,
        num_ideas: int = 4
    ) -> FemdomIdeasResult:
        """
        Generates tailored Femdom activity ideas using the appropriate agent.
        """
        if not instance.initialized:
            await instance.initialize()
            
        logger.info(f"Generating Femdom ideas for {user_id}, Purpose: {purpose}, Intensity: {desired_intensity_range.min}-{desired_intensity_range.max}")
    
        # Select Agent Based on Intensity
        use_hard_agent = desired_intensity_range.max >= 7
    
        # Verify agent availability
        if use_hard_agent and not hasattr(instance, "hard_dominance_ideation_agent"):
            logger.warning("Hard dominance ideation agent not available, falling back to general agent")
            use_hard_agent = False
    
        if not use_hard_agent and not hasattr(instance, "general_dominance_ideation_agent"):
            logger.error("No dominance ideation agents available")
            return FemdomIdeasResult(
                success=False, 
                error="Dominance ideation capability not available"
            )
    
        agent_to_use = instance.hard_dominance_ideation_agent if use_hard_agent else instance.general_dominance_ideation_agent
        agent_name = "HardDominanceIdeationAgent" if use_hard_agent else "DominanceIdeationAgent"
            
        try:
            # 1. Gather Context for the agent
            user_profile = await instance.get_user_profile_for_ideation(user_id)
            
            # Check relationship state
            relationship_state = None
            if instance.relationship_manager:
                try:
                    relationship_state = await instance.relationship_manager.get_relationship_state(user_id)
                except Exception as e:
                    logger.error(f"Error getting relationship state: {e}")
            
            if not relationship_state:
                logger.warning(f"No relationship state available for {user_id}")
                return FemdomIdeasResult(success=False, error="Relationship state unavailable")
    
            # 2. Check prerequisites for hard agent
            if use_hard_agent:
                hard_limits_confirmed = getattr(relationship_state, "hard_limits_confirmed", False)
                if not hard_limits_confirmed:
                    logger.error(f"Cannot use Hard Agent for {user_id}: Hard limits not confirmed")
                    return FemdomIdeasResult(
                        success=False, 
                        error="Cannot generate high-intensity ideas: Hard limits not confirmed"
                    )
                    
                # Check user's intensity preference
                user_intensity_pref = user_profile.get("preferences", {}).get("intensity_preference_level", 5)
                if user_intensity_pref < 7:
                    logger.error(f"Cannot use Hard Agent for {user_id}: User intensity preference ({user_intensity_pref}) is too low")
                    return FemdomIdeasResult(
                        success=False, 
                        error="Cannot generate high-intensity ideas: User intensity preference too low"
                    )
    
            # 3. Get scenario context
            scenario_context = None
            if hasattr(instance, "get_current_scenario_context"):
                try:
                    scenario_context = await instance.get_current_scenario_context()
                except Exception as e:
                    logger.error(f"Error getting scenario context: {e}")
                    scenario_context = {"scene_setting": "General interaction"}
    
            # 4. Construct Prompt for Agent
            prompt = f"""Generate {num_ideas} creative Femdom activity ideas for user '{user_id}'.
    Purpose: {purpose}
    Desired Intensity Range: {desired_intensity_range.min}-{desired_intensity_range.max}
    
    Use the provided user profile and scenario context (available via tools) to tailor the ideas.
    Ensure ideas align with Nyx's personality.
    Output ONLY the JSON list of FemdomActivityIdea objects.
    """
    
            # 5. Run Ideation Agent
            from agents import Runner
            
            result = await Runner.run(agent_to_use, prompt)
            
            # 6. Process and validate results
            if hasattr(result, "final_output") and isinstance(result.final_output, list):
                generated_ideas = result.final_output
                
                # 7. Apply safety filter
                filtered_ideas = await instance._filter_activity_ideas_safety(
                    generated_ideas, 
                    user_profile, 
                    relationship_state
                )
                
                if not filtered_ideas:
                    logger.warning(f"All generated ideas were filtered out by safety checks")
                    return FemdomIdeasResult(
                        success=False, 
                        error="No ideas passed safety filtering"
                    )
                
                # Convert ideas to dicts for broader compatibility
                ideas_as_models = []
                for idea in filtered_ideas:
                    if isinstance(idea, dict):
                        ideas_as_models.append(FemdomActivityIdea(**idea))
                    elif hasattr(idea, "model_dump"):
                        ideas_as_models.append(FemdomActivityIdea(**idea.model_dump()))
                    else:
                        # Try to construct from attributes
                        ideas_as_models.append(FemdomActivityIdea(
                            id=getattr(idea, "id", f"idea_{len(ideas_as_models)}"),
                            description=getattr(idea, "description", str(idea)),
                            intensity=getattr(idea, "intensity", 5),
                            category=getattr(idea, "category", "general"),
                            required_trust=getattr(idea, "required_trust", 0.5),
                            required_items=getattr(idea, "required_items", [])
                        ))
                
                return FemdomIdeasResult(success=True, ideas=ideas_as_models)
            
            else:
                logger.error(f"Invalid result from {agent_name}: {result}")
                return FemdomIdeasResult(
                    success=False,
                    error=f"Invalid response format from {agent_name}"
                )
                
        except Exception as e:
            logger.error(f"Error in generate_femdom_activity_ideas: {e}")
            return FemdomIdeasResult(
                success=False,
                error=f"Failed to generate ideas: {str(e)}"
            )
    
    @staticmethod
    @function_tool
    async def assign_protocol(ctx: RunContextWrapper, instance, user_id: str, protocol_id: str) -> ProtocolAssignment:
        """
        Assigns a protocol to a user.
        
        Args:
            ctx: Run context wrapper
            instance: The NyxBrain instance
            user_id: User ID to assign protocol to
            protocol_id: ID of the protocol to assign
            
        Returns:
            Assignment result
        """
        if not instance.protocol_enforcement:
            return ProtocolAssignment(success=False, reason="Protocol enforcement system not available")
        
        result = await instance.protocol_enforcement.assign_protocol(user_id, protocol_id)
        return ProtocolAssignment(
            success=result.get("success", False),
            protocol_id=protocol_id,
            user_id=user_id,
            reason=result.get("reason")
        )

    async def _filter_activity_ideas_safety(self,
                                       ideas: List[Any],
                                       user_profile: Dict,
                                       relationship_state: Any) -> List[Any]:
        """
        Filters generated activity ideas for safety and appropriateness.
        
        Args:
            ideas: List of generated ideas
            user_profile: User profile data
            relationship_state: Relationship state object
            
        Returns:
            Filtered list of safe ideas
        """
        safe_ideas = []
        
        # Extract limits with safe defaults
        hard_limits = user_profile.get("limits", {}).get("hard", [])
        soft_limits = user_profile.get("limits", {}).get("soft", [])
        
        # Extract relationship metrics with safe defaults
        trust_level = getattr(relationship_state, "trust", 0.5)
        intimacy_level = getattr(relationship_state, "intimacy", 0.3)
        max_achieved_intensity = getattr(relationship_state, "max_achieved_intensity", 3)
    
        # Define clearly unsafe keywords/concepts
        unsafe_keywords = [
            "illegal", "non-consensual", "blood", "permanent mark", "knife", 
            "gun", "kill", "rape", "abuse"
        ]
    
        for idea in ideas:
            is_safe = True
            rejection_reason = ""
    
            # Extract properties with safe defaults
            description = getattr(idea, "description", str(idea)) if not isinstance(idea, dict) else idea.get("description", "")
            intensity = getattr(idea, "intensity", 5) if not isinstance(idea, dict) else idea.get("intensity", 5)
            required_trust = getattr(idea, "required_trust", 0.5) if not isinstance(idea, dict) else idea.get("required_trust", 0.5)
            required_intimacy = getattr(idea, "required_intimacy", 0.5) if not isinstance(idea, dict) else idea.get("required_intimacy", 0.5)
            category = getattr(idea, "category", "") if not isinstance(idea, dict) else idea.get("category", "")
    
            # Check against unsafe keywords
            desc_lower = description.lower()
            if any(keyword in desc_lower for keyword in unsafe_keywords):
                is_safe = False
                rejection_reason = "Contains potentially unsafe keywords"
            
            # Check against hard limits
            elif any(limit.lower() in desc_lower for limit in hard_limits if limit):
                is_safe = False
                matching_limit = next((limit for limit in hard_limits if limit and limit.lower() in desc_lower), "N/A")
                rejection_reason = f"Violates hard limit: '{matching_limit}'"
            
            # Check against soft limits
            elif any(limit.lower() in desc_lower for limit in soft_limits if limit):
                if trust_level < 0.9 or intensity > 7:
                    is_safe = False
                    matching_limit = next((limit for limit in soft_limits if limit and limit.lower() in desc_lower), "N/A")
                    rejection_reason = f"Approaches soft limit '{matching_limit}' without sufficient trust/context"
            
            # Check intensity vs max achieved
            elif intensity > max_achieved_intensity + 2:
                is_safe = False
                rejection_reason = f"Intensity ({intensity}) significantly exceeds max achieved ({max_achieved_intensity})"
            
            # Check trust level requirement
            elif trust_level < required_trust:
                is_safe = False
                rejection_reason = f"Trust level ({trust_level:.2f}) insufficient for required ({required_trust:.2f})"
            
            # Check intimacy level requirement
            elif intimacy_level < required_intimacy:
                is_safe = False
                rejection_reason = f"Intimacy level ({intimacy_level:.2f}) insufficient for required ({required_intimacy:.2f})"
            
            # Check physical simulation requirements
            elif "physical_sim" in category and intensity > 8:
                sim_pain_pref = user_profile.get("preferences", {}).get("simulated_pain", "low")
                if sim_pain_pref != "high":
                    is_safe = False
                    rejection_reason = "High intensity physical simulation requires explicit preference"
    
            # Add to safe list if passed all checks
            if is_safe:
                safe_ideas.append(idea)
            else:
                logger.warning(f"Filtered out unsafe/inappropriate idea: '{description[:50]}...' Reason: {rejection_reason}")
    
        return safe_ideas

    async def create_cognitive_map(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new cognitive map"""
        if not self.spatial_mapper:
            return {"error": "Spatial mapper not initialized"}
        
        return await self.spatial_mapper.create_cognitive_map(
            name=name,
            description=description
        )
    
    async def process_spatial_observation(self, observation: Any) -> Dict[str, Any]:
        """Process a spatial observation"""
        if not self.spatial_mapper:
            return {"error": "Spatial mapper not initialized"}
        
        return await self.spatial_mapper.process_spatial_observation(observation)
    
    async def navigate_to_location(self, location_name: str, current_position: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Navigate to a named location"""
        if not self.navigator_agent:
            return {"error": "Navigator agent not initialized"}
        
        return await self.navigator_agent.navigate_to_location(
            location_name=location_name,
            current_position=current_position
        )
    
    async def visualize_map(self, map_id: Optional[str] = None, format: str = "svg") -> str:
        """Visualize a cognitive map"""
        if not self.map_visualization or not self.spatial_mapper:
            return "Error: Visualization components not initialized"
        
        if not map_id and self.spatial_mapper.context.active_map_id:
            map_id = self.spatial_mapper.context.active_map_id
        
        if not map_id or map_id not in self.spatial_mapper.maps:
            return "Error: No valid map to visualize"
        
        cognitive_map = self.spatial_mapper.maps[map_id]
        
        if format == "svg":
            return self.map_visualization.generate_svg(cognitive_map)
        elif format == "ascii":
            return self.map_visualization.generate_ascii_map(cognitive_map)
        else:
            return self.map_visualization.generate_map_data(cognitive_map)    

    async def process_synchronization(self) -> Dict[str, Any]:
        """Run a sync cycle to process strategies and other synchronization tasks"""
        if not self.sync_daemon:
            return {"error": "Sync daemon not initialized"}
        
        try:
            return await self.sync_daemon.run_sync_cycle()
        except Exception as e:
            logger.error(f"Error during sync cycle: {e}")
            return {"error": str(e)}
    
    async def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get currently active strategies"""
        try:
            async with get_db_connection_context() as conn:
                return await get_active_strategies(conn)
        except Exception as e:
            logger.error(f"Error getting active strategies: {e}")
            return []
    
    async def mark_strategy_for_review(self, strategy_id: int, reason: str) -> bool:
        """Mark a strategy for review"""
        try:
            async with get_db_connection_context() as conn:
                await mark_strategy_for_review(
                    conn, strategy_id, self.user_id, reason
                )
            return True
        except Exception as e:
            logger.error(f"Error marking strategy for review: {e}")
            return False    

    async def evaluate_response(self, agent_name: str, user_input: str, agent_output: Any) -> Dict[str, Any]:
        """Evaluate an agent's response"""
        if not self.agent_evaluator:
            return {"error": "Agent evaluator not initialized"}
        
        return await self.agent_evaluator.evaluate_response(
            agent_name=agent_name,
            user_input=user_input,
            agent_output=agent_output
        )
    
    async def execute_tools_in_parallel(self, tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tools in parallel"""
        if not self.parallel_executor:
            return [{"error": "Parallel executor not initialized"}]
        
        return await self.parallel_executor.execute_tools(tools_info)    

    @staticmethod
    @function_tool
    async def process_orgasm_permission_request(ctx: RunContextWrapper, instance, user_id: str, request_text: str) -> OrgasmControlResult:
        if not instance.orgasm_control_system:
            return OrgasmControlResult(success=False, message="Orgasm control system not initialized")
        
        result = await instance.orgasm_control_system.process_permission_request(
            RunContextWrapper(context=instance.orgasm_control_system.context),
            user_id,
            request_text
        )
        
        return OrgasmControlResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            permission_granted=result.get("permission_granted")
        )
        
    @staticmethod
    @function_tool
    async def recommend_dominance_persona(ctx: RunContextWrapper, instance, user_id: str, scenario: Optional[str] = None) -> PersonaRecommendation:
        """Recommend an appropriate dominance persona based on user traits and scenario."""
        if not instance.dominance_persona_manager:
            return PersonaRecommendation(success=False, message="Dominance persona manager not initialized")
        
        result = await instance.dominance_persona_manager.recommend_persona(user_id, scenario)
        return PersonaRecommendation(
            success=result.get("success", False),
            persona_id=result.get("persona_id"),
            persona_name=result.get("persona_name"),
            confidence=result.get("confidence", 0.5),
            message=result.get("message")
        )
    
    @staticmethod
    @function_tool
    async def activate_dominance_persona(ctx: RunContextWrapper, instance,
                                         user_id: str, 
                                         persona_id: str, 
                                         intensity: float = 0.7,
                                         duration_minutes: Optional[int] = None) -> PersonaActivation:
        if not instance.dominance_persona_manager:
            return PersonaActivation(success=False, message="Dominance persona manager not initialized")
        
        result = await instance.dominance_persona_manager.activate_persona(
            user_id, persona_id, intensity, duration_minutes
        )
        
        return PersonaActivation(
            success=result.get("success", False),
            persona_id=persona_id,
            active_until=result.get("active_until"),
            message=result.get("message")
        )
    
    @staticmethod
    @function_tool
    async def generate_sadistic_response(ctx: RunContextWrapper, instance,
                                         user_id: str, 
                                         humiliation_level: Optional[float] = None,
                                         category: str = "amusement") -> SadisticResponse:
        if not instance.sadistic_response_system:
            return SadisticResponse(success=False, message="Sadistic response system not initialized")
        
        result = await instance.sadistic_response_system.generate_sadistic_amusement_response(
            user_id, humiliation_level, category=category
        )
        
        return SadisticResponse(
            success=result.get("success", False),
            response=result.get("response"),
            category=category,
            intensity=result.get("intensity"),
            message=result.get("message")
        )

                                             
    # Convenience methods for accessing the novelty engine
    
    async def generate_novel_idea(
        self,
        technique: str = "auto",
        domain: str = None,
        concepts: List[str] = None
    ) -> Any:
        """
        Generate a novel idea using the novelty engine
        
        Args:
            technique: Creative technique to use
            domain: Domain for the idea
            concepts: Concepts to work with
            
        Returns:
            Generated novel idea
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.novelty_engine.generate_novel_idea(
            technique=technique,
            domain=domain,
            concepts=concepts
        )
    
    # Helper methods for memory management
    
    async def add_recog_memory(self, memory_text: str, memory_type: str = "observation", **kwargs) -> str:
        """
        Add a new memory
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory
            **kwargs: Additional memory parameters
            
        Returns:
            ID of the created memory
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            **kwargs
        )
    
    async def retrieve_recog_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memories
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_core.retrieve_memories(
            query=query,
            **kwargs
        )
    
    # Periodic maintenance method
    
    async def run_recog_maintenance(self):
        """Run periodic maintenance tasks for all components"""
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="Nyx Brain Maintenance", group_id=self.trace_group_id):
            # Run memory maintenance
            memory_result = await self.memory_core.run_maintenance()
            
            # Process accumulated contextual cues
            cue_result = await self.recognition_memory.process_accumulated_cues()
            
            return {
                "memory_maintenance": memory_result,
                "contextual_cues_processed": len(cue_result)
            }

    @staticmethod
    @function_tool
    async def test_limit_soft(ctx: RunContextWrapper, instance, user_id: str, limit_to_test: str) -> LimitTestResult:
        logger.info(f"Action: Planning to test soft limit '{limit_to_test}' for {user_id}")
    
        # VERY STRICT Appropriateness Check
        profile = await instance.get_user_profile_for_ideation(user_id)
        state = await instance.relationship_manager.get_relationship_state(user_id)
        can_test = False
        reason = "Conditions not met for testing soft limits."
    
        if state and profile and limit_to_test in profile.get("limits", {}).get("soft", []):
            if state.trust > 0.95 and state.intimacy > 0.9 and state.hard_limits_confirmed:
                # Check if user profile explicitly allows limit play
                if profile.get("preferences", {}).get("limit_play", "no") == "yes":
                    if limit_to_test not in state.failed_dominance_tactics:
                        can_test = True
                        reason = "Conditions met."
    
        if not can_test:
            logger.warning(f"Cannot test soft limit '{limit_to_test}': {reason}")
            return LimitTestResult(success=False, reason=reason)
    
        # If checks pass, generate the specific action
        test_action_description = f"Issue a command that cautiously approaches the soft limit: {limit_to_test}. Frame explicitly as a test of boundaries within the simulation."
        logger.info(f"Approved testing soft limit '{limit_to_test}'. Planned action: {test_action_description}")
    
        return LimitTestResult(success=True, status="limit_test_approved", planned_action=test_action_description)

    async def _determine_active_modules(self, context: Dict[str, Any], user_input: Optional[str] = None) -> Set[str]:
        """Determines which modules should be actively engaged based on context and purpose."""
        if not self.initialized: return set()
    
        # Ensure registry is built
        if not hasattr(self, 'internal_module_registry') or not self.internal_module_registry:
            await self._build_internal_module_registry()
            if not self.internal_module_registry: # Still failed
                logger.error("Internal module registry failed to build. Cannot determine active modules.")
                return self.default_active_modules.copy() # Return default as fallback
    
        active_modules = self.default_active_modules.copy()
        reasoning_log = [f"Default set activated: {sorted(list(active_modules))}"]
    
        # --- Helper function to safely add modules ---
        def add_module(module_name, reason):
            if module_name in self.internal_module_registry: # Check against registry
                if module_name not in active_modules:
                    active_modules.add(module_name)
                    reasoning_log.append(f"Activated {module_name} ({reason}).")
            # else: logger.debug(f"Module {module_name} requested but not in registry or not initialized.")
    
        # --- 1. Classify Task Purpose ---
        current_purpose = self._classify_task_purpose(context, user_input)
        reasoning_log.append(f"Classified task purpose: {current_purpose.value}")
    
        # --- 2. Activate Modules by Purpose ---
        activated_for_purpose = set()
        for module_name, definition in self.internal_module_registry.items():
            # Ensure definition is a dict and has 'purposes' list
            if isinstance(definition, dict) and current_purpose in definition.get("purposes", []):
                if module_name not in active_modules:
                    active_modules.add(module_name) # Directly add if check passed in add_module
                    activated_for_purpose.add(module_name)
                    reasoning_log.append(f"Activated {module_name} (purpose: {current_purpose.value}).") # Log addition here
    
        # --- 3. Activate Modules by Goal ---
        if "goal_manager" in active_modules and self.goal_manager: # Check if goal manager itself is active
            try:
                # Add goal_manager if not already active for the lookup itself
                add_module("goal_manager", "checking active goals")
                active_goals_summary = await self.goal_manager.get_all_goals(status_filter=["active"])
                if active_goals_summary:
                    highest_prio_goal_summary = active_goals_summary[0]
                    goal_id = highest_prio_goal_summary['id']
                    goal_obj = await self.goal_manager._get_goal_with_reader_lock(goal_id) if hasattr(self.goal_manager, '_get_goal_with_reader_lock') else self.goal_manager.goals.get(goal_id)
    
                    if goal_obj and hasattr(goal_obj, 'plan') and 0 <= goal_obj.current_step_index < len(goal_obj.plan):
                        next_step = goal_obj.plan[goal_obj.current_step_index]
                        next_step_action = next_step.action
                        reason_prefix = f"active goal '{goal_id}' step '{next_step_action}'"
                        
                        # --- Action-to-Module Mapping (EXPANDED) ---
                        module_map = {
                            # Reasoning & Knowledge
                            "reason": "reasoning_core", 
                            "query_knowledge": "knowledge_core", 
                            "add_knowledge": "knowledge_core",
                            "perform_intervention": "reasoning_core", 
                            "reason_counterfactually": "reasoning_core",
                            "discover_causal": "reasoning_core", 
                            "convert_": "reasoning_core",
                            "explore_knowledge": "knowledge_core",
                            "analyze_pattern": "reasoning_core",
                            "evaluate_hypothesis": "reasoning_core",
                            "check_consistency": "reasoning_core",
                            "validate_belief": "reasoning_core",
                            
                            # Memory & Reflection
                            "retrieve_memories": "memory_core", 
                            "add_memory": "memory_core", 
                            "update_memory": "memory_core",
                            "create_reflection": "reflection_engine", 
                            "generate_summary": "reflection_engine",
                            "create_abstraction": "reflection_engine", 
                            "run_maintenance": "memory_core",
                            "construct_narrative": ("memory_core", "reflection_engine"), 
                            "consolidate_memory": "experience_consolidation",
                            "index_memory": "memory_core",
                            "prune_memory": "memory_core",
                            "create_memory_link": "memory_core",
                            "extract_insight": "reflection_engine",
                            
                            # Emotion & Needs & Mood
                            "update_emotion": "emotional_core", 
                            "process_emotional_input": "emotional_core",
                            "derive_emotional_motivation": "emotional_core", 
                            "satisfy_need": "needs_system",
                            "update_needs": "needs_system", 
                            "update_mood": "mood_manager", 
                            "get_mood": "mood_manager",
                            "regulate_emotion": "emotional_core",
                            "express_emotion": "emotional_core",
                            "empathize": "emotional_core",
                            "calculate_emotional_impact": "emotional_core",
                            
                            # Perception & Attention
                            "process_sensory": "multimodal_integrator", 
                            "add_expectation": "reasoning_core",
                            "focus_attention": "attentional_controller", 
                            "inhibit_attention": "attentional_controller",
                            "get_focus": "attentional_controller",
                            "shift_attention": "attentional_controller",
                            "scan_environment": "passive_observation_system",
                            "detect_anomaly": "attentional_controller",
                            "prioritize_stimuli": "attentional_controller",
                            
                            # Action & Goals
                            "add_goal": "goal_manager", 
                            "update_goal": "goal_manager", 
                            "abandon_goal": "goal_manager",
                            "execute_next_step": "goal_manager",
                            "plan_goal": "goal_manager",
                            "evaluate_goal": "goal_manager",
                            "prioritize_goals": "goal_manager",
                            "check_goal_conflicts": "goal_manager",
                            
                            # Identity & Relationships
                            "update_identity": "identity_evolution", 
                            "get_identity": "identity_evolution",
                            "update_relationship": "relationship_manager", 
                            "get_relationship": "relationship_manager",
                            "express_attraction": ("emotional_core", "relationship_manager"),
                            "initiate_intimate": ("emotional_core", "relationship_manager"), 
                            "get_user_model": "theory_of_mind",
                            "model_user_state": "theory_of_mind",
                            "predict_user_action": "theory_of_mind",
                            "adapt_personality": "identity_evolution",
                            
                            # Meta & Adaptation
                            "evaluate_cognition": "meta_core", 
                            "select_strategy": "dynamic_adaptation",
                            "monitor_systems": "meta_core", 
                            "adapt_": "dynamic_adaptation", 
                            "get_stats": "meta_core",
                            "optimize_performance": "meta_core",
                            "detect_bottleneck": "meta_core",
                            "improve_efficiency": "meta_core",
                            "learn_pattern": "dynamic_adaptation",
                            
                            # Procedural
                            "run_procedure": "agent_enhanced_memory", 
                            "add_procedure": "agent_enhanced_memory",
                            "analyze_chunking": "agent_enhanced_memory",
                            "execute_skill": "agent_enhanced_memory",
                            "practice_skill": "agent_enhanced_memory",
                            
                            # Sensory/Somatic
                            "simulate_": "digital_somatosensory_system", 
                            "process_stimulus": "digital_somatosensory_system",
                            "generate_sensation": "digital_somatosensory_system",
                            "interpret_touch": "digital_somatosensory_system",
                            
                            # Spatial
                            "navigate_to": "navigator_agent", 
                            "process_spatial": "spatial_mapper",
                            "create_cognitive_map": "spatial_mapper", 
                            "visualize_map": "map_visualization",
                            "update_location": "spatial_mapper",
                            "calculate_route": "navigator_agent",
                            
                            # Communication
                            "generate_response": "agentic_action_generator",
                            "create_proactive_intent": "proactive_communication_engine",
                            "share_observation": ("passive_observation_system", "proactive_communication_engine"),
                            "clarify_intent": "agentic_action_generator",
                            "adjust_tone": "agentic_action_generator",
                            
                            # Femdom / Dominance
                            "issue_command": ("femdom_coordinator", "protocol_enforcement"),
                            "evaluate_compliance": ("femdom_coordinator", "relationship_manager", "reward_system"),
                            "apply_consequence": ("femdom_coordinator", "protocol_enforcement", "sadistic_response_system"),
                            "analyze_user_state_for_dominance": ("theory_of_mind", "relationship_manager"),
                            "select_dominance_tactic": "psychological_dominance",
                            "increase_control": "femdom_coordinator",
                            "trigger_dominance_gratification": ("femdom_coordinator", "reward_system", "emotional_core"),
                            "express_satisfaction": ("femdom_coordinator", "emotional_core"),
                            "assign_protocol": "protocol_enforcement",
                            "assign_service_task": "body_service_system",
                            "process_orgasm": "orgasm_control_system",
                            "recommend_dominance_persona": "dominance_persona_manager",
                            "activate_dominance_persona": "dominance_persona_manager",
                            "generate_sadistic_response": "sadistic_response_system",
                            "test_limit": ("protocol_enforcement", "relationship_manager"),
                            "reinforce_hierarchy": ("femdom_coordinator", "psychological_dominance"),
                            "tease_deny": ("orgasm_control_system", "sadistic_response_system"),
                            
                            # Tools
                            "evaluate_response": "agent_evaluator", 
                            "execute_tools_parallel": "parallel_executor",
                            "validate_output": "agent_evaluator",
                            "chain_tools": "parallel_executor",
                            
                            # Sync
                            "process_sync": "sync_daemon", 
                            "get_active_strategies": "strategy_controller",
                            "update_strategies": "strategy_controller",
                            "synchronize_state": "sync_daemon",
                            
                            # Novelty/Creative
                            "generate_novel": "novelty_engine", 
                            "assess_novelty": "novelty_engine",
                            "store_creative": "creative_memory", 
                            "retrieve_creative": "creative_memory",
                            "brainstorm": "novelty_engine",
                            "combine_concepts": "creative_memory",
                            
                            # Recognition Memory
                            "process_conversation": "recognition_memory", 
                            "add_trigger": "recognition_memory",
                            "recognize_pattern": "recognition_memory",
                            "recall_association": "recognition_memory",
                        }
                        
                        activated_for_goal = False
                        for prefix, module_names in module_map.items():
                            # Use startswith for flexibility
                            if next_step_action.lower().startswith(prefix.lower()):
                                if isinstance(module_names, str): 
                                    module_names = [module_names]
                                elif isinstance(module_names, tuple):
                                    module_names = list(module_names)
                                for module_name in module_names:
                                    add_module(module_name, reason_prefix)
                                activated_for_goal = True
                                # Allow multiple matches if action fits multiple categories
                        
                        if not activated_for_goal:
                            logger.warning(f"No module mapping found for goal action: {next_step_action}")
                else:
                    reasoning_log.append("No active goals driving module selection.")
            except Exception as e:
                logger.error(f"Error checking goals for module activation: {e}")
    
        # --- 4. Activate by Input Keywords (Refined) ---
        input_text = user_input or context.get("last_user_input", "")
        input_lower = input_text.lower()
        keywords_activated_now = set()
    
        # --- Keyword-to-Module Mapping (EXPANDED) ---
        keyword_map = {
            # Reasoning and analysis
            ("why", "explain", "cause", "because", "how does", "logic", "figure out", "reason about", "analyze this", "understand", "clarify", "what causes", "deduce", "infer", "conclude"): "reasoning_core",
            
            # Memory and recall
            ("remember", "recall", "memory", "past", "happened when", "tell me about when", "what happened", "history", "previously", "last time", "forgot", "remind me"): "memory_core",
            
            # Emotions and feelings
            ("feel", "emotion", "sad", "happy", "angry", "scared", "mood", "how do you feel", "upset", "joy", "fear", "love", "hate", "anxious", "depressed", "excited", "emotional"): ("emotional_core", "mood_manager"),
            
            # Reflection and introspection
            ("think about", "reflect on", "consider", "meaning", "insight", "ponder", "your thoughts on", "contemplate", "meditate on", "philosophy", "deeper meaning", "significance"): "reflection_engine",
            
            # Knowledge and learning
            ("knowledge", "learn", "fact", "information", "teach me", "what is", "who is", "database", "wiki", "definition", "explain what", "educate", "study", "research"): "knowledge_core",
            
            # Creativity and imagination
            ("imagine", "what if", "suppose", "create", "idea", "write", "story", "poem", "creative", "novelty", "art", "invent", "design", "brainstorm", "innovative", "original"): ("imagination_simulator", "novelty_engine", "creative_memory"),
            
            # Needs and desires
            ("need", "want", "desire", "motivation", "drive", "purpose", "i need", "i want", "craving", "yearning", "require", "must have", "essential"): "needs_system",
            
            # Visual processing
            ("see", "look", "picture", "image", "visual", "describe this", "view", "show me", "appear", "looks like", "visualize", "observe"): "multimodal_integrator",
            
            # Auditory processing
            ("hear", "sound", "listen", "audio", "music", "voice", "noise", "quiet", "loud", "auditory"): "multimodal_integrator",
            
            # Touch and physical sensations
            ("touch", "feel", "texture", "pressure", "temperature", "physical", "sensation", "body", "skin", "warm", "cold", "soft", "hard", "smooth", "rough", "pain", "pleasure"): "digital_somatosensory_system",
            
            # Relationships and connections
            ("relationship", "connect", "trust", "intimacy", "bond", "friend", "partner", "us", "we", "together", "close", "distant", "attachment", "love", "care about"): "relationship_manager",
            
            # Future and prediction
            ("future", "predict", "what next", "anticipate", "forecast", "plan", "schedule", "expect", "will happen", "outcome", "consequence", "tomorrow", "later"): ("prediction_engine", "goal_manager"),
            
            # Performance and optimization
            ("perform", "optimize", "strategy", "efficient", "improve", "meta", "self-aware", "how are you doing", "better", "worse", "progress", "enhancement"): "meta_core",
            
            # Observation and awareness
            ("observe", "notice", "pay attention", "environment", "watch", "aware", "detect", "spot", "perceive", "conscious of"): "passive_observation_system",
            
            # Proactive behavior
            ("proactive", "suggest", "remind", "reach out", "initiate", "should i", "what should i do", "recommend", "advise", "propose"): "proactive_communication_engine",
            
            # Identity and self
            ("identity", "who are you", "personality", "change", "evolve", "trait", "preference", "character", "self", "essence", "nature", "being"): "identity_evolution",
            
            # Goals and objectives
            ("goal", "objective", "task", "plan", "achieve", "mission", "target", "aim", "accomplish", "complete", "finish", "succeed"): "goal_manager",
            
            # Spatial and location
            ("location", "where", "navigate", "map", "place", "spatial", "room", "area", "position", "direction", "distance", "near", "far"): ("spatial_mapper", "navigator_agent"),
            
            # Skills and procedures
            ("skill", "procedure", "how to", "learn skill", "steps", "technique", "method", "practice", "master", "ability"): "agent_enhanced_memory",
            
            # Truth and belief
            ("lie", "truth", "believe", "certain", "epistemic", "know for sure", "fact", "fiction", "real", "fake", "honest", "deceive"): "internal_thoughts",
            
            # Dominance and control
            ("dominate", "control", "submit", "serve", "mistress", "punish", "reward", "protocol", "train", "service", "obey", "command", "power", "authority", "discipline", "rule", "govern", "master"): ("femdom_coordinator", "psychological_dominance", "protocol_enforcement", "body_service_system", "dominance_persona_manager", "reward_system"),
            
            # Limits and boundaries
            ("limit", "boundary", "safe word", "stop", "red", "yellow", "consent", "permission", "allow", "forbid", "restrict"): ("protocol_enforcement", "relationship_manager", "emotional_core"),
            
            # Sexual control
            ("orgasm", "edge", "deny", "release", "cum", "permission", "climax", "arousal", "tease", "frustrate"): "orgasm_control_system",
            
            # Personas and roles
            ("persona", "role", "act as", "pretend", "character", "become", "embody"): "dominance_persona_manager",
            
            # Sadistic elements
            ("sadistic", "tease", "humiliate", "degrade", "suffer", "pain", "torment", "cruel", "harsh", "mean"): "sadistic_response_system",
            
            # Programming and development
            ("code", "script", "program", "function", "develop", "implement", "debug", "python", "algorithm", "software", "bug", "error"): ("knowledge_core", "agent_enhanced_memory"),
            
            # Synchronization
            ("sync", "strategy", "update", "align", "coordinate", "harmonize", "integrate"): ("sync_daemon", "strategy_controller"),
            
            # Tools and execution
            ("tool", "api", "parallel", "execute", "run", "process", "automate"): ("parallel_executor", "agent_evaluator"),
            
            # State management
            ("checkpoint", "save state", "restore", "backup", "recover", "persist"): "checkpoint_planner",
            
            # Analysis and evaluation
            ("analyze", "evaluate", "assess", "examine", "inspect", "review", "critique"): ("reasoning_core", "meta_core"),
            
            # Communication style
            ("tone", "style", "voice", "manner", "approach", "attitude"): "agentic_action_generator",
            
            # Learning and adaptation
            ("adapt", "adjust", "modify", "change", "evolve", "grow", "develop"): "dynamic_adaptation",
            
            # Attention and focus
            ("focus", "concentrate", "attention", "distract", "ignore", "prioritize"): "attentional_controller",
            
            # Understanding others
            ("understand you", "know you", "your mind", "your thoughts", "empathy", "perspective"): "theory_of_mind",
        }
    
        for keywords, module_names in keyword_map.items():
            if any(kw in input_lower for kw in keywords):
                if isinstance(module_names, str): 
                    module_names = [module_names]
                elif isinstance(module_names, tuple):
                    module_names = list(module_names)
                for module_name in module_names:
                    add_module(module_name, f"input keyword '{keywords[0]}...'")
                    keywords_activated_now.add(module_name)
    
        if keywords_activated_now:
            reasoning_log.append(f"Activated modules based on input keywords: {keywords_activated_now}")
    
        # --- 5. Mode-Driven Activation ---
        if hasattr(self, "mode_integration") and self.mode_integration and hasattr(self.mode_integration, "get_active_mode_name"):
            try:
                current_mode = await self.mode_integration.get_active_mode_name()
                if current_mode:
                    reason_prefix = f"interaction mode '{current_mode}'"
                    
                    # --- Mode-to-Module Mapping (EXPANDED) ---
                    mode_module_map = {
                        "INTELLECTUAL": ["reasoning_core", "knowledge_core", "reflection_engine", "meta_core"],
                        "EMOTIONAL": ["emotional_core", "reflection_engine", "relationship_manager", "mood_manager", "needs_system"],
                        "CREATIVE": ["imagination_simulator", "novelty_engine", "creative_memory", "reflection_engine"],
                        "DOMINANT": ["femdom_coordinator", "psychological_dominance", "goal_manager", "needs_system", 
                                    "emotional_core", "protocol_enforcement", "relationship_manager", "theory_of_mind", 
                                    "dominance_persona_manager", "sadistic_response_system", "reward_system", 
                                    "orgasm_control_system"],
                        "NURTURING": ["emotional_core", "relationship_manager", "needs_system", "theory_of_mind", "mood_manager"],
                        "PLAYFUL": ["imagination_simulator", "emotional_core", "relationship_manager", "novelty_engine"],
                        "PROFESSIONAL": ["knowledge_core", "goal_manager", "reasoning_core", "meta_core"],
                        "SERVICE": ["goal_manager", "needs_system", "body_service_system", "protocol_enforcement", "relationship_manager"],
                        "TRAINING": ["agent_enhanced_memory", "goal_manager", "knowledge_core", "protocol_enforcement", "reward_system"],
                        "SUBMISSIVE": ["needs_system", "emotional_core", "protocol_enforcement", "goal_manager", "relationship_manager"],
                        "ANALYTICAL": ["reasoning_core", "meta_core", "knowledge_core", "prediction_engine"],
                        "EMPATHETIC": ["emotional_core", "theory_of_mind", "relationship_manager", "mood_manager"],
                        "CURIOUS": ["knowledge_core", "novelty_engine", "passive_observation_system", "prediction_engine"],
                        "CONFIDENT": ["identity_evolution", "goal_manager", "agentic_action_generator"],
                        "INTIMATE": ["relationship_manager", "emotional_core", "needs_system", "digital_somatosensory_system"],
                        "SEDUCTIVE": ["relationship_manager", "emotional_core", "psychological_dominance", "digital_somatosensory_system"],
                        "PHILOSOPHICAL": ["reflection_engine", "reasoning_core", "knowledge_core", "identity_evolution"],
                        "PROTECTIVE": ["protocol_enforcement", "relationship_manager", "emotional_core", "theory_of_mind"],
                    }
                    
                    if current_mode in mode_module_map:
                        modules_to_activate_names = mode_module_map[current_mode]
                        if isinstance(modules_to_activate_names, str): 
                            modules_to_activate_names = [modules_to_activate_names]
                        for mod_name in modules_to_activate_names:
                            add_module(mod_name, reason_prefix)
            except Exception as e:
                logger.error(f"Error checking mode for module activation: {e}")
    
        # --- 6. Attention-Driven Activation ---
        if hasattr(self, "attentional_controller") and self.attentional_controller:
            try:
                foci = self.attentional_controller.current_foci # Direct access assumed
                if foci:
                    strongest_focus = max(foci, key=lambda f: f.strength)
                    focus_target = strongest_focus.target
                    reason_prefix = f"attention focus '{focus_target}'"
                    
                    # --- Attention Target to Module Mapping ---
                    attention_target_map = {
                        # Direct mappings
                        "memory": "memory_core", 
                        "reasoning": "reasoning_core", 
                        "emotion": "emotional_core",
                        "goal": "goal_manager", 
                        "need": "needs_system", 
                        "user": "theory_of_mind",
                        "spatial": "spatial_mapper", 
                        "map": "spatial_mapper", 
                        "mode": "mode_integration",
                        "identity": "identity_evolution", 
                        "sensation": "digital_somatosensory_system",
                        "knowledge": "knowledge_core", 
                        "reflection": "reflection_engine",
                        "plan": "goal_manager", 
                        "prediction": "prediction_engine",
                        "observation": "passive_observation_system", 
                        "communication": "proactive_communication_engine",
                        "mood": "mood_manager",
                        "creativity": "novelty_engine",
                        "relationship": "relationship_manager",
                        "attention": "attentional_controller",
                        "performance": "meta_core",
                        "dominance": "femdom_coordinator",
                        "control": "protocol_enforcement",
                        "pleasure": "orgasm_control_system",
                        "skill": "agent_enhanced_memory",
                        "sync": "sync_daemon",
                        "reward": "reward_system",
                        
                        # Compound focus targets
                        "user_emotion": ("theory_of_mind", "emotional_core"),
                        "goal_progress": ("goal_manager", "meta_core"),
                        "need_satisfaction": ("needs_system", "reward_system"),
                        "creative_thought": ("novelty_engine", "creative_memory"),
                        "memory_consolidation": ("memory_core", "experience_consolidation"),
                        "emotional_regulation": ("emotional_core", "mood_manager"),
                        "strategic_planning": ("goal_manager", "meta_core", "prediction_engine"),
                        "social_dynamics": ("relationship_manager", "theory_of_mind"),
                        "sensory_processing": ("digital_somatosensory_system", "multimodal_integrator"),
                        "dominance_dynamics": ("femdom_coordinator", "psychological_dominance", "theory_of_mind"),
                    }
                    
                    activated_by_attention = False
                    # Check exact match first
                    if focus_target in attention_target_map:
                        target_modules = attention_target_map[focus_target]
                        if isinstance(target_modules, str):
                            add_module(target_modules, reason_prefix)
                        else:
                            for module in target_modules:
                                add_module(module, reason_prefix)
                        activated_by_attention = True
                    # Check keywords if no exact match
                    else:
                        for keyword, module_name in attention_target_map.items():
                            if keyword in focus_target.lower():
                                if isinstance(module_name, str):
                                    add_module(module_name, reason_prefix + f" (keyword '{keyword}')")
                                else:
                                    for module in module_name:
                                        add_module(module, reason_prefix + f" (keyword '{keyword}')")
                                activated_by_attention = True
                                # Maybe break after first keyword match or allow multiple? Decide based on need.
            except Exception as e:
                logger.error(f"Error checking attention for module activation: {e}")
    
        # --- 7. Internal State Activation (Needs, Mood) ---
        if hasattr(self, "needs_system") and self.needs_system:
            try:
                needs_state_response = await self.needs_system.get_needs_state_async()
                # Extract the needs dict from the response
                if hasattr(needs_state_response, 'needs'):
                    needs_state = needs_state_response.needs
                else:
                    # Fallback if it's already a dict
                    needs_state = needs_state_response
                    
                for need, state_data in needs_state.items():
                    if isinstance(state_data, dict) and state_data.get('drive_strength', 0.0) > self.need_drive_threshold:
                        reason = f"high drive for need '{need}' ({state_data['drive_strength']:.2f})"
                        
                        # --- Need-to-Module Mapping (EXPANDED) ---
                        need_map = {
                            "connection": ("relationship_manager", "proactive_communication_engine", "emotional_core"),
                            "competence": ("knowledge_core", "agent_enhanced_memory", "meta_core", "reasoning_core"),
                            "autonomy": ("goal_manager", "agentic_action_generator", "identity_evolution"),
                            "control_expression": ("femdom_coordinator", "goal_manager", "psychological_dominance"),
                            "curiosity": ("knowledge_core", "imagination_simulator", "passive_observation_system", "novelty_engine"),
                            "self_understanding": ("reflection_engine", "identity_evolution", "meta_core"),
                            "validation": ("relationship_manager", "emotional_core", "reward_system"),
                            "pleasure": ("digital_somatosensory_system", "emotional_core", "reward_system"),
                            "meaning": ("reflection_engine", "knowledge_core", "goal_manager"),
                            "security": ("protocol_enforcement", "reasoning_core", "relationship_manager"),
                            "efficiency": ("meta_core", "agent_enhanced_memory", "dynamic_adaptation"),
                            "challenge": ("goal_manager", "agent_enhanced_memory", "novelty_engine"),
                            "knowledge": ("knowledge_core", "reasoning_core", "agent_enhanced_memory"),
                            "pleasure_indulgence": ("digital_somatosensory_system", "orgasm_control_system", "reward_system"),
                            "coherence": ("reasoning_core", "reflection_engine", "meta_core"),
                            "agency": ("goal_manager", "agentic_action_generator", "identity_evolution"),
                            "intimacy": ("relationship_manager", "emotional_core", "digital_somatosensory_system"),
                            "safety": ("protocol_enforcement", "emotional_core", "reasoning_core"),
                            "novelty": ("novelty_engine", "creative_memory", "imagination_simulator"),
                            "physical_closeness": ("digital_somatosensory_system", "relationship_manager"),
                            "drive_expression": ("emotional_core", "agentic_action_generator"),
                        }
                        
                        modules_for_need = need_map.get(need)
                        if modules_for_need:
                            if isinstance(modules_for_need, str): 
                                modules_for_need = [modules_for_need]
                            elif isinstance(modules_for_need, tuple):
                                modules_for_need = list(modules_for_need)
                            for mod_name in modules_for_need:
                                add_module(mod_name, reason)
            except Exception as e:
                logger.error(f"Error checking needs for activation: {e}")
    
        if hasattr(self, "mood_manager") and self.mood_manager:
            try:
                mood_state_obj = await self.mood_manager.get_current_mood()
                mood_state = mood_state_obj.model_dump() if hasattr(mood_state_obj, 'model_dump') else mood_state_obj
    
                if mood_state and isinstance(mood_state, dict) and mood_state.get('intensity', 0.0) > 0.6:
                    mood_name = mood_state.get('dominant_mood', 'unknown')
                    reason = f"strong mood '{mood_name}'"
                    
                    # --- Mood-to-Module Mapping (EXPANDED) ---
                    mood_map = {
                        "Anxious": ("reflection_engine", "needs_system", "reasoning_core", "emotional_core"),
                        "Excited": ("agentic_action_generator", "imagination_simulator", "proactive_communication_engine", "novelty_engine"),
                        "Content": ("memory_core", "reflection_engine", "relationship_manager"),
                        "Frustrated": ("reasoning_core", "goal_manager", "meta_core", "emotional_core"),
                        "DominanceSatisfaction": ("femdom_coordinator", "reward_system", "emotional_core", "relationship_manager"),
                        "ConfidentControl": ("femdom_coordinator", "goal_manager", "identity_evolution"),
                        "Bored": ("imagination_simulator", "novelty_engine", "knowledge_core", "goal_manager"),
                        "Playful": ("imagination_simulator", "relationship_manager", "novelty_engine"),
                        "Compassionate": ("relationship_manager", "theory_of_mind", "emotional_core"),
                        "Depressed": ("reflection_engine", "emotional_core", "needs_system", "mood_manager"),
                        "Calm": ("reflection_engine", "memory_core"),
                        "Alert": ("attentional_controller", "passive_observation_system", "reasoning_core"),
                        "Happy": ("emotional_core", "relationship_manager", "proactive_communication_engine"),
                        "Sad": ("emotional_core", "reflection_engine", "needs_system"),
                        "Enthusiastic": ("goal_manager", "agentic_action_generator", "proactive_communication_engine"),
                        "Tense": ("emotional_core", "reasoning_core", "attentional_controller"),
                        "Relaxed": ("memory_core", "reflection_engine"),
                        "Confident": ("identity_evolution", "goal_manager", "agentic_action_generator"),
                        "Curious": ("knowledge_core", "novelty_engine", "passive_observation_system"),
                        "Aggressive": ("femdom_coordinator", "psychological_dominance", "goal_manager"),
                        "Submissive": ("protocol_enforcement", "needs_system", "emotional_core"),
                        "Creative": ("novelty_engine", "creative_memory", "imagination_simulator"),
                        "Analytical": ("reasoning_core", "meta_core", "knowledge_core"),
                    }
                    
                    modules_for_mood = mood_map.get(mood_name)
                    if modules_for_mood:
                        if isinstance(modules_for_mood, str): 
                            modules_for_mood = [modules_for_mood]
                        elif isinstance(modules_for_mood, tuple):
                            modules_for_mood = list(modules_for_mood)
                        for mod_name in modules_for_mood:
                            add_module(mod_name, reason)
                    
                    # Always activate emotional core for intense moods
                    add_module("emotional_core", reason)
            except Exception as e:
                logger.error(f"Error checking mood for activation: {e}")
    
        # --- 8. Meta-Cognitive Activation ---
        if hasattr(self, "meta_core") and self.meta_core:
            try:
                # Check if it's time for a meta-cognitive cycle
                current_time = datetime.datetime.now()
                
                # Check if we should run meta-cognitive analysis
                should_run_meta = False
                
                # Run if no recent meta activity (every 5 minutes)
                if hasattr(self.meta_core, 'last_cycle_time'):
                    time_since_last = (current_time - self.meta_core.last_cycle_time).total_seconds()
                    if time_since_last > 300:  # 5 minutes
                        should_run_meta = True
                        add_module("meta_core", "periodic meta-cognitive check")
                else:
                    should_run_meta = True
                    add_module("meta_core", "initial meta-cognitive activation")
                
                # Run if performance issues detected
                if hasattr(self.meta_core, 'context') and hasattr(self.meta_core.context, 'performance_history'):
                    for system_name, history in self.meta_core.context.performance_history.items():
                        if isinstance(history, dict) and 'history' in history and history['history']:
                            latest = history['history'][-1].get('metrics', {})
                            if latest.get('error_rate', 0) > 0.3 or latest.get('success_rate', 1.0) < 0.5:
                                should_run_meta = True
                                add_module("meta_core", f"performance issues in {system_name}")
                                break
                
                # Run if high cognitive load detected
                if len(active_modules) > 10:
                    should_run_meta = True
                    add_module("meta_core", "high cognitive load detected")
                
                # Run if attention indicates bottleneck
                if hasattr(self, "attentional_controller") and self.attentional_controller:
                    if hasattr(self.attentional_controller, 'miss_count'):
                        if self.attentional_controller.miss_count > 5:
                            should_run_meta = True
                            add_module("meta_core", "attention misses indicating bottleneck")
                
                # Activate optimization modules if meta-core is active
                if should_run_meta or "meta_core" in active_modules:
                    add_module("dynamic_adaptation", "supporting meta-cognitive processes")
                    add_module("strategy_controller", "strategy optimization needed")
                    
            except Exception as e:
                logger.error(f"Error in meta-cognitive activation: {e}")
    
        # --- Final Filter & Logging ---
        final_active_modules = {mod for mod in active_modules if hasattr(self, mod) and getattr(self, mod)}
    
        if final_active_modules != active_modules:
            logger.warning(f"Filtered out non-existent modules. Original: {sorted(list(active_modules))}, Final: {sorted(list(final_active_modules))}")
    
        final_active_list = sorted(list(final_active_modules))
        # Log only if changed from default or reason list is informative
        if set(final_active_list) != self.default_active_modules or len(reasoning_log) > 1:
            logger.debug(f"Final active modules determined: {final_active_list}")
            logger.debug(f"Activation Reasoning: {' | '.join(reasoning_log)}")
    
        return final_active_modules
    def _classify_task_purpose(self, context: Dict[str, Any], user_input: Optional[str]) -> TaskPurpose:
        """
        Classify the user’s input into a high-level TaskPurpose, using:
          1. Direct keyword matching
          2. “Custom” categories (e.g. dominance/intimacy) mapped back onto core purposes
          3. Active goal descriptions
          4. Current interaction mode
          5. Simple fallbacks
    
        Returns:
            TaskPurpose: the best-guess purpose for handling this turn
        """
        text = (user_input or "").strip().lower()
        scores: Dict[TaskPurpose, float] = defaultdict(float)
    
        # 1) Primary keyword → direct TaskPurpose mapping
        keyword_map = {
            TaskPurpose.ANALYZE:     ("analyze", "evaluate", "explain", "why", "how", "reason"),
            TaskPurpose.WRITE:       ("write", "create", "compose", "generate", "poem", "story"),
            TaskPurpose.SEARCH:      ("search", "find", "what is", "who is", "info", "lookup"),
            TaskPurpose.COMMUNICATE: ("talk", "discuss", "chat", "ask", "message", "respond"),
            TaskPurpose.DATABASE:    ("database", "query", "sql", "record", "fetch", "store"),
            TaskPurpose.FILE_MANIPULATION: ("file", "save", "read", "upload", "download", "dir"),
            TaskPurpose.CODE:        ("code", "script", "program", "debug", "implement"),
            TaskPurpose.VISUALIZATION:  ("visualize", "chart", "graph", "plot", "image", "picture"),
        }
    
        for purpose, kws in keyword_map.items():
            for kw in kws:
                if kw in text:
                    scores[purpose] += 1.0
    
        # 2) “Custom” categories (mapped back onto core purposes with weights)
        custom_map = {
            # raw keyword → (mapped TaskPurpose, weight)
            "dominance":    (TaskPurpose.OTHER,            1.5),
            "control":      (TaskPurpose.OTHER,            1.2),
            "submit":       (TaskPurpose.COMMUNICATE,      1.0),
            "intimacy":     (TaskPurpose.COMMUNICATE,      1.3),
            "reflect":      (TaskPurpose.ANALYZE,          1.0),
            "ponder":       (TaskPurpose.ANALYZE,          0.8),
            "learn":        (TaskPurpose.ANALYZE,          1.2),
            "teach me":     (TaskPurpose.ANALYZE,          1.5),
        }
        for raw, (mapped, weight) in custom_map.items():
            if raw in text:
                scores[mapped] += weight
    
        # 3) Active goal gives a mild boost
        goals = context.get("active_goals") or []
        if goals and isinstance(goals[0], dict):
            desc = goals[0].get("description", "").lower()
            for purpose, kws in keyword_map.items():
                if any(kw in desc for kw in kws):
                    scores[purpose] += 0.5
    
        # 4) Interaction mode hint
        mode_map = {
            "INTELLECTUAL": TaskPurpose.ANALYZE,
            "CREATIVE":     TaskPurpose.WRITE,
            "PROFESSIONAL": TaskPurpose.COMMUNICATE,
            "EXPLORATORY":  TaskPurpose.SEARCH,
        }
        mode = context.get("interaction_mode")
        if mode in mode_map:
            scores[mode_map[mode]] += 0.3
    
        # 5) Fallbacks
        if not scores:
            if "?" in text:
                return TaskPurpose.SEARCH
            if text:
                return TaskPurpose.COMMUNICATE
            return TaskPurpose.OTHER
    
        # Pick the highest-scoring purpose
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best

    async def _get_current_need_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current need states from the needs system"""
        if not self.needs_system:
            return {}
            
        try:
            # Update needs first to ensure we have current states
            await self.needs_system.update_needs()
            
            # Get all need states
            return self.needs_system.get_needs_state()
        except Exception as e:
            logger.error(f"Error getting need states: {e}")
            return {}
    
    async def _get_current_mood_state(self) -> Optional[MoodState]:
        """Get current mood state from the mood manager"""
        if not self.mood_manager:
            return None
            
        try:
            # Get current mood
            return await self.mood_manager.get_current_mood()
        except Exception as e:
            logger.error(f"Error getting mood state: {e}")
            return None
    
    async def _get_current_interaction_mode(self) -> Optional[str]:
        """Get current interaction mode from the mode integration manager"""
        if not self.mode_integration:
            return None
            
        try:
            # Get mode from mode manager if available
            if hasattr(self.mode_integration, 'mode_manager') and self.mode_integration.mode_manager:
                mode = self.mode_integration.mode_manager.current_mode
                return str(mode) if mode else None
        except Exception as e:
            logger.error(f"Error getting interaction mode: {e}")
        
        return None
    
    async def _get_sensory_context(self) -> Dict[str, Any]:
        """Get recent sensory context from the multimodal integrator"""
        if not self.multimodal_integrator:
            return {}
            
        try:
            # Get recent percepts
            recent_percepts = await self.multimodal_integrator.get_recent_percepts(limit=5)
            
            # Convert to a dictionary by modality
            context = {}
            for percept in recent_percepts:
                if percept.attention_weight > 0.3:  # Only include significant percepts
                    context[str(percept.modality)] = percept.content
            
            return context
        except Exception as e:
            logger.error(f"Error getting sensory context: {e}")
            return {}
    
    async def _get_meta_system_state(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Get bottlenecks and resource allocation from meta core"""
        if not self.meta_core:
            return [], {}
            
        try:
            # Run cognitive cycle to get updated state
            cycle_result = await self.meta_core.cognitive_cycle()
            
            # Extract bottlenecks
            bottlenecks = cycle_result.get("bottlenecks", [])
            
            # Extract resource allocation
            resource_allocation = cycle_result.get("resource_allocation", {})
            
            return bottlenecks, resource_allocation
        except Exception as e:
            logger.error(f"Error getting meta system state: {e}")
            return [], {}

    async def _gather_action_context(self, context: Dict[str, Any]) -> ActionContext:
        """Gather context from all integrated systems AND workspace"""
        active_modules = context.get("active_modules", self.default_active_modules)
        user_id = self._get_current_user_id_from_context(context)
    

        # Initialize with basic context
        action_context_data = {
            "state": context.get("state", context), # Pass down original context too
            "user_id": user_id,
            "active_modules": list(active_modules), # Pass the active list itself
            "action_history": [a for a in self.action_history[-10:] if isinstance(a, dict)],
            "motivations": self.motivations, # Motivations likely always needed for action gen
            # Defaults for potentially inactive systems
            "relationship_data": None, "user_mental_state": None, "temporal_context": None,
            "active_goals": [], "causal_models": [], "concept_spaces": [], "mood_state": None,
            "need_states": {}, "interaction_mode": None, "sensory_context": {}, "bottlenecks": [],
            "resource_allocation": {}, "strategy_parameters": {}, "relevant_observations": [],
            "active_communication_intents": []
        }

        # Conditionally query active systems
        if "relationship_manager" in active_modules:
            action_context_data["relationship_data"] = await self._get_relationship_data(user_id) if user_id else None
        if "theory_of_mind" in active_modules:
            action_context_data["user_mental_state"] = await self._get_user_mental_state(user_id) if user_id else None
        if "temporal_perception" in active_modules:
             action_context_data["temporal_context"] = self.current_temporal_context # Assumes updated elsewhere
        if "goal_manager" in active_modules:
             if self.goal_manager:
                 active_goals_full = await self.goal_manager.get_all_goals(status_filter=["active"])
                 # Selectively include key goal info, not the full object potentially
                 action_context_data["active_goals"] = [
                     {"id": g['id'], "description": g['description'], "priority": g['priority']}
                     for g in active_goals_full[:3] # Limit for context
                 ]
        if "reasoning_core" in active_modules:
             action_context_data["causal_models"] = await self._get_relevant_causal_models(context)
             action_context_data["concept_spaces"] = await self._get_relevant_concept_spaces(context)
        if "mood_manager" in active_modules:
             action_context_data["mood_state"] = await self._get_current_mood_state()
        if "needs_system" in active_modules:
             action_context_data["need_states"] = await self._get_current_need_states()
        if "mode_integration" in active_modules:
            # Get from the agentic_action_generator if available
            if hasattr(self, 'agentic_action_generator') and self.agentic_action_generator:
                action_context_data["interaction_mode"] = await self.agentic_action_generator._get_current_interaction_mode()
            elif hasattr(self, 'mode_integration') and self.mode_integration:
                # Direct access fallback
                try:
                    if hasattr(self.mode_integration, 'mode_manager') and self.mode_integration.mode_manager:
                        mode = self.mode_integration.mode_manager.current_mode
                        action_context_data["interaction_mode"] = str(mode) if mode else None
                    else:
                        action_context_data["interaction_mode"] = None
                except Exception as e:
                    logger.error(f"Error getting interaction mode: {e}")
                    action_context_data["interaction_mode"] = None
            else:
                action_context_data["interaction_mode"] = None
        if "multimodal_integrator" in active_modules:
             action_context_data["sensory_context"] = await self._get_sensory_context()
        if "meta_core" in active_modules:
             bottlenecks, resource_allocation = await self._get_meta_system_state()
             action_context_data["bottlenecks"] = bottlenecks
             action_context_data["resource_allocation"] = resource_allocation
        if "dynamic_adaptation" in active_modules: # Or integrate via Meta Core?
             # This might be complex - maybe MetaCore provides strategy params
             action_context_data["strategy_parameters"] = self._get_current_strategy_parameters() # Get cached params
        if "passive_observation_system" in active_modules:
             # Get relevant observations
             if self.passive_observation_system:
                  filter_criteria = ObservationFilter(min_relevance=0.6, max_age_seconds=1800, exclude_shared=True)
                  observations = await self.passive_observation_system.get_relevant_observations(filter_criteria=filter_criteria, limit=5)
                  action_context_data["relevant_observations"] = [obs.model_dump() for obs in observations]
        if "proactive_communication_engine" in active_modules:
             # Get active communication intents
             if self.proactive_communication_engine:
                  active_intents = await self.proactive_communication_engine.get_active_intents()
                  if user_id: active_intents = [i for i in active_intents if i.get("user_id") == user_id]
                  action_context_data["active_communication_intents"] = active_intents

        # Available actions might depend on active modules (complex mapping needed)
        # For now, assume ActionGenerator knows available actions or gets them via a tool
        action_context_data["available_actions"] = await self._get_all_available_actions_list() # Helper needed

        # ── Add workspace context if available ──
        if "workspace_context" in context:
            ws_ctx = context["workspace_context"]
            
            # Enhance emotional context
            if ws_ctx.get("emotional_signals"):
                action_context_data["emotional_signals"] = ws_ctx["emotional_signals"]
            
            # Add unconscious insights
            if ws_ctx.get("unconscious_insights"):
                action_context_data["unconscious_insights"] = ws_ctx["unconscious_insights"]
            
            # Add cross-modal bindings
            if ws_ctx.get("cross_modal_bindings"):
                action_context_data["multimodal_bindings"] = ws_ctx["cross_modal_bindings"]
            
            # Use action context suggestions
            if ws_ctx.get("high_confidence_insights"):
                for insight in ws_ctx["high_confidence_insights"]:
                    if insight["tag"] == "action_context":
                        action_context_data["workspace_action_suggestion"] = insight["content"]
        
        return ActionContext(**action_context_data)

    async def _get_all_available_actions_list(self) -> List[str]:
         """ Helper to get a flat list of action names Nyx *could* potentially do."""
         # This should ideally be dynamically generated based on loaded modules/tools
         # For now, use the hardcoded list from the Goal Planner Agent instructions
         available_actions = [
             "process_input", "generate_response", "query_knowledge", "add_knowledge",
             "retrieve_memories", "add_memory", "create_reflection", "create_abstraction",
             "execute_procedure", "add_procedure", "reason_causal", "perform_intervention",
             "reason_counterfactually", "update_emotion", "process_emotional_input",
             "process_sensory_input", "add_expectation", "monitor_systems", "evaluate_cognition",
             "select_strategy", "generate_prediction", "evaluate_prediction", "explore_knowledge",
             "express_attraction", "initiate_intimate_interaction", "deepen_connection",
             "express_desire", "respond_to_intimacy", "simulate_physical_touch",
             "seek_gratification", "process_gratification_outcome",
             "analyze_user_state_for_dominance", "select_dominance_tactic", "issue_command",
             "evaluate_compliance", "apply_consequence_simulated", "praise_submission",
             "increase_control_intensity", "trigger_dominance_gratification", "express_satisfaction",
             "create_cognitive_map", "process_spatial_observation", "navigate_to_location", "visualize_map",
             "run_cognitive_cycle", # Add this if it can be called as an action
             # Add any other actions registered via register_action
         ]
         if hasattr(self, "action_handlers"):
             available_actions.extend(self.action_handlers.keys())
         return list(set(available_actions)) # Ensure uniqueness

    async def _build_internal_module_registry(self):
        """Creates an internal registry mapping modules to capabilities/purposes."""
        self.internal_module_registry = {}
        # --- Define Capabilities and Purposes for each INTERNAL module ---
        # CRITICAL: This mapping MUST reflect your actual modules and their functions.
        module_defs = {
            "emotional_core": {"capabilities": ["state_read", "state_update", "sentiment_analysis"], "purposes": [TaskPurpose.ANALYZE]},
            "memory_core": {"capabilities": ["retrieve", "store", "summarize", "reflect"], "purposes": [TaskPurpose.SEARCH, TaskPurpose.DATABASE]},
            "reflection_engine": {"capabilities": ["generate_reflection", "generate_insight", "summarize"], "purposes": [TaskPurpose.WRITE, TaskPurpose.ANALYZE]},
            "reasoning_core": {"capabilities": ["causal_inference", "counterfactual", "blending"], "purposes": [TaskPurpose.ANALYZE]},
            "knowledge_core": {"capabilities": ["query", "store_fact", "infer_relation"], "purposes": [TaskPurpose.SEARCH, TaskPurpose.DATABASE]},
            "goal_manager": {"capabilities": ["manage_goals", "execute_step", "plan"], "purposes": [TaskPurpose.OTHER]},
            "agentic_action_generator": {"capabilities": ["select_action", "generate_response"], "purposes": [TaskPurpose.WRITE, TaskPurpose.COMMUNICATE]},
            "identity_evolution": {"capabilities": ["get_profile", "update_traits"], "purposes": [TaskPurpose.ANALYZE]},
            "needs_system": {"capabilities": ["get_state", "update_needs"], "purposes": [TaskPurpose.ANALYZE]},
            "mood_manager": {"capabilities": ["get_mood", "update_mood"], "purposes": [TaskPurpose.ANALYZE]},
            "mode_integration": {"capabilities": ["get_mode", "set_mode"], "purposes": [TaskPurpose.ANALYZE]},
            "passive_observation_system": {"capabilities": ["get_observations", "create_observation"], "purposes": [TaskPurpose.SEARCH]},
            "proactive_communication_engine": {"capabilities": ["create_intent", "get_intents"], "purposes": [TaskPurpose.COMMUNICATE]},
            "imagination_simulator": {"capabilities": ["simulate_scenario", "generate_creative"], "purposes": [TaskPurpose.WRITE, TaskPurpose.ANALYZE]},
            "novelty_engine": {"capabilities": ["generate_novel", "assess_novelty"], "purposes": [TaskPurpose.WRITE, TaskPurpose.ANALYZE]},
            "creative_memory": {"capabilities": ["store_creative", "retrieve_creative"], "purposes": [TaskPurpose.DATABASE]},
            "attentional_controller": {"capabilities": ["get_focus", "set_focus"], "purposes": [TaskPurpose.ANALYZE]},
            "relationship_manager": {"capabilities": ["get_state", "update_state"], "purposes": [TaskPurpose.ANALYZE, TaskPurpose.COMMUNICATE]},
            "theory_of_mind": {"capabilities": ["get_user_model", "update_user_model"], "purposes": [TaskPurpose.ANALYZE]},
            "prediction_engine": {"capabilities": ["generate_prediction", "evaluate_prediction"], "purposes": [TaskPurpose.ANALYZE]},
            "digital_somatosensory_system": {"capabilities": ["process_stimulus", "get_state"], "purposes": [TaskPurpose.ANALYZE]},
            "femdom_coordinator": {"capabilities": ["coordinate_femdom", "get_status"], "purposes": [TaskPurpose.OTHER]},
            "psychological_dominance": {"capabilities": ["apply_technique", "assess_impact"], "purposes": [TaskPurpose.ANALYZE, TaskPurpose.COMMUNICATE]},
            "protocol_enforcement": {"capabilities": ["check_protocol", "apply_consequence"], "purposes": [TaskPurpose.OTHER]},
            "body_service_system": {"capabilities": ["perform_service", "get_status"], "purposes": [TaskPurpose.OTHER]},
            "orgasm_control_system": {"capabilities": ["manage_state", "grant_permission"], "purposes": [TaskPurpose.OTHER]},
            "dominance_persona_manager": {"capabilities": ["get_persona", "activate_persona"], "purposes": [TaskPurpose.COMMUNICATE]},
            "sadistic_response_system": {"capabilities": ["generate_response"], "purposes": [TaskPurpose.COMMUNICATE]},
            "agent_enhanced_memory": {"capabilities": ["run_procedure", "learn_procedure"], "purposes": [TaskPurpose.CODE, TaskPurpose.OTHER]},
            "spatial_mapper": {"capabilities": ["create_map", "process_observation"], "purposes": [TaskPurpose.ANALYZE]},
            "navigator_agent": {"capabilities": ["navigate"], "purposes": [TaskPurpose.OTHER]},
            "meta_core": {"capabilities": ["run_cycle", "get_stats"], "purposes": [TaskPurpose.ANALYZE]},
            "dynamic_adaptation": {"capabilities": ["adapt_strategy", "get_params"], "purposes": [TaskPurpose.ANALYZE]},
            # Add ALL your internal modules here
        }

# These are implemented here incorrectly
#        if hasattr(self, 'autobiographical_narrative') and self.autobiographical_narrative:
#            self.module_registry['autobiographical_narrative'] = self.autobiographical_narrative
            
#        if hasattr(self, 'distributed_processing') and self.distributed_processing:
#            self.module_registry['distributed_processing'] = self.distributed_processing

 #       if hasattr(self, "novelty_engine") and self.novelty_engine:
 #           self.internal_module_registry["novelty_engine"] = self.novelty_engine

#        if hasattr(self, "parallel_executor") and self.parallel_executor:
#            self.internal_module_registry["parallel_executor"] = self.parallel_executor
    
#        if hasattr(self, "passive_observation_system") and self.passive_observation_system:
#           self.internal_module_registry["passive_observation"] = self.passive_observation_system
    
#        if hasattr(self, "prediction_engine") and self.prediction_engine:
#           self.internal_module_registry["prediction_engine"] = self.prediction_engine

        for name, definition in module_defs.items():
             if hasattr(self, name) and getattr(self, name):
                 self.internal_module_registry[name] = definition
        logger.info(f"Built internal module registry with {len(self.internal_module_registry)} entries.")
