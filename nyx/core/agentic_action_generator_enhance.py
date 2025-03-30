# nyx/core/agentic_action_generator_enhance.py

import logging
import asyncio
import datetime
import uuid
import random
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict
from pydantic import BaseModel, Field
from enum import Enum

# Core system imports (ensuring all references are maintained)
from nyx.core.reasoning_core import (
    ReasoningCore, CausalModel, CausalNode, CausalRelation,
    ConceptSpace, ConceptualBlend, Intervention
)
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.multimodal_integrator import (
    MultimodalIntegrator, Modality, SensoryInput, ExpectationSignal, IntegratedPercept
)
from nyx.core.mood_manager import MoodManager, MoodState
from nyx.core.needs_system import NeedsSystem, NeedState
from nyx.core.mode_integration import ModeIntegrationManager, InteractionMode
from nyx.core.meta_core import MetaCore, StrategyResult

logger = logging.getLogger(__name__)

class ActionSource(str, Enum):
    """Enum for tracking the source of an action"""
    MOTIVATION = "motivation"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    IDLE = "idle"
    HABIT = "habit"
    EXPLORATION = "exploration"
    USER_ALIGNED = "user_aligned"
    REASONING = "reasoning"  # Reasoning-based actions
    REFLECTION = "reflection"  # Reflection-based actions
    NEED = "need"  # New: Need-driven actions
    MOOD = "mood"  # New: Mood-driven actions
    MODE = "mode"  # New: Interaction mode-driven actions
    META_COGNITIVE = "meta_cognitive"  # New: Meta-cognitive strategy actions
    SENSORY = "sensory"  # New: Actions from sensory integration

class ActionContext(BaseModel):
    """Context for action selection and generation"""
    state: Dict[str, Any] = Field(default_factory=dict, description="Current system state")
    user_id: Optional[str] = None
    relationship_data: Optional[Dict[str, Any]] = None
    user_mental_state: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    motivations: Dict[str, float] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Existing fields for reasoning integration
    causal_models: List[str] = Field(default_factory=list, description="IDs of relevant causal models")
    concept_spaces: List[str] = Field(default_factory=list, description="IDs of relevant concept spaces")
    
    # New fields for enhanced integrations
    mood_state: Optional[MoodState] = None
    need_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    interaction_mode: Optional[str] = None
    sensory_context: Dict[str, Any] = Field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = Field(default_factory=list)
    resource_allocation: Dict[str, float] = Field(default_factory=dict)
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    
class ActionOutcome(BaseModel):
    """Outcome of an executed action"""
    action_id: str
    success: bool = False
    satisfaction: float = Field(0.0, ge=0.0, le=1.0)
    reward_value: float = Field(0.0, ge=-1.0, le=1.0)
    user_feedback: Optional[Dict[str, Any]] = None
    neurochemical_changes: Dict[str, float] = Field(default_factory=dict)
    hormone_changes: Dict[str, float] = Field(default_factory=dict)
    impact: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    
    # Existing fields for reasoning-informed outcomes
    causal_impacts: Dict[str, Any] = Field(default_factory=dict, description="Impacts identified by causal reasoning")
    
    # New fields for enhanced outcome tracking
    need_impacts: Dict[str, float] = Field(default_factory=dict, description="Impact on need satisfaction")
    mood_impacts: Dict[str, float] = Field(default_factory=dict, description="Impact on mood dimensions")
    mode_alignment: float = Field(0.0, description="How well action aligned with interaction mode")
    sensory_feedback: Dict[str, Any] = Field(default_factory=dict, description="Sensory feedback from action")
    meta_evaluation: Dict[str, Any] = Field(default_factory=dict, description="Meta-cognitive evaluation")
    
class ActionValue(BaseModel):
    """Q-value for a state-action pair"""
    state_key: str
    action: str
    value: float = 0.0
    update_count: int = 0
    confidence: float = Field(0.2, ge=0.0, le=1.0)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    # New field for strategy effectiveness
    strategy_effectiveness: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def is_reliable(self) -> bool:
        """Whether this action value has enough updates to be considered reliable"""
        return self.update_count >= 3 and self.confidence >= 0.5

class ActionMemory(BaseModel):
    """Memory of an executed action and its result"""
    state: Dict[str, Any]
    action: str
    action_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    source: ActionSource
    
    # Existing fields for reasoning and reflection
    causal_explanation: Optional[str] = None
    reflective_insight: Optional[str] = None
    
    # New fields for enhanced memory
    need_satisfaction: Dict[str, float] = Field(default_factory=dict)
    mood_impact: Dict[str, float] = Field(default_factory=dict)
    mode_alignment: Optional[float] = None
    sensory_context: Optional[Dict[str, Any]] = None
    meta_evaluation: Optional[Dict[str, Any]] = None

class ActionReward(BaseModel):
    """Reward signal for an action"""
    value: float = Field(..., description="Reward value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: str = Field(..., description="Source generating the reward")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context info")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    # New fields for reward decomposition
    components: Dict[str, float] = Field(default_factory=dict, description="Reward value broken down by component")

class ReflectionInsight(BaseModel):
    """Insight from reflection about an action"""
    action_id: str
    insight_text: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    significance: float = Field(0.5, ge=0.0, le=1.0)
    applicable_contexts: List[str] = Field(default_factory=list)
    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    # New fields for enhanced reflection
    action_pattern: Optional[Dict[str, Any]] = None
    improvement_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    related_needs: List[str] = Field(default_factory=list)
    related_moods: List[str] = Field(default_factory=list)

class ActionStrategy(BaseModel):
    """Strategy for action selection and generation"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    applicable_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    effectiveness: float = Field(0.5, ge=0.0, le=1.0)
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime.datetime] = None
    
    # Categorization and adaptation fields
    for_needs: List[str] = Field(default_factory=list)
    for_moods: List[Dict[str, float]] = Field(default_factory=list)
    for_modes: List[str] = Field(default_factory=list)

class EnhancedAgenticActionGenerator:
    """
    Enhanced Agentic Action Generator that integrates reward learning, prediction,
    user modeling, relationship context, temporal awareness, causal reasoning, 
    conceptual blending, reflection-based learning, and multi-system integration.
    
    Generates actions based on system's internal state, motivations, goals, 
    neurochemical/hormonal influences, reinforcement learning, causal models,
    conceptual blending, introspective reflection, needs, mood, and interaction modes.
    """
    
    def __init__(self, 
                 emotional_core=None, 
                 hormone_system=None, 
                 experience_interface=None,
                 imagination_simulator=None,
                 meta_core=None,
                 memory_core=None,
                 goal_system=None,
                 identity_evolution=None,
                 knowledge_core=None,
                 input_processor=None,
                 internal_feedback=None,
                 reward_system=None,
                 prediction_engine=None,
                 theory_of_mind=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 
                 # Existing additional systems
                 reasoning_core=None,
                 reflection_engine=None,
                 
                 # New system integrations
                 mood_manager=None,
                 needs_system=None,
                 mode_integration=None,
                 multimodal_integrator=None):
        """Initialize with references to required subsystems"""
        # Core systems from original implementation
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.experience_interface = experience_interface
        self.imagination_simulator = imagination_simulator
        self.meta_core = meta_core
        self.memory_core = memory_core
        self.goal_system = goal_system
        self.identity_evolution = identity_evolution
        self.knowledge_core = knowledge_core
        self.input_processor = input_processor
        self.internal_feedback = internal_feedback
        
        # Previous new system integrations
        self.reward_system = reward_system
        self.prediction_engine = prediction_engine
        self.theory_of_mind = theory_of_mind
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        
        # Existing additional systems
        self.reasoning_core = reasoning_core or ReasoningCore()
        self.reflection_engine = reflection_engine or ReflectionEngine(emotional_core=emotional_core)
        
        # New system integrations
        self.mood_manager = mood_manager
        self.needs_system = needs_system
        self.mode_integration = mode_integration
        self.multimodal_integrator = multimodal_integrator
        
        # Internal motivation system
        self.motivations = {
            "curiosity": 0.5,       # Desire to explore and learn
            "connection": 0.5,      # Desire for interaction/bonding
            "expression": 0.5,      # Desire to express thoughts/emotions
            "competence": 0.5,      # Desire to improve capabilities
            "autonomy": 0.5,        # Desire for self-direction
            "dominance": 0.5,       # Desire for control/influence
            "validation": 0.5,      # Desire for recognition/approval
            "self_improvement": 0.5, # Desire to enhance capabilities
            "leisure": 0.5,          # Desire for downtime/relaxation
        }
        
        # Activity generation capabilities
        self.action_patterns = {}  # Patterns learned from past successful actions
        self.action_templates = {}  # Templates for generating new actions
        self.action_history = []
        
        # Reinforcement learning components
        self.action_values: Dict[str, Dict[str, ActionValue]] = defaultdict(dict)
        self.action_memories: List[ActionMemory] = []
        self.max_memories = 1000
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995  # Decay rate for exploration
        
        # Track last major action time for pacing
        self.last_major_action_time = datetime.datetime.now()
        self.last_idle_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        # Temporal awareness tracking
        self.idle_duration = 0.0
        self.idle_start_time = None
        self.current_temporal_context = None
        
        # Track reward statistics
        self.total_reward = 0.0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.reward_by_category = defaultdict(lambda: {"count": 0, "total": 0.0})
        
        # Track leisure state
        self.leisure_state = {
            "current_activity": None,
            "satisfaction": 0.5,
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        # Action success tracking for reinforcement learning
        self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
        
        # Cached goal status
        self.cached_goal_status = {
            "has_active_goals": False,
            "highest_priority": 0.0,
            "active_goal_id": None,
            "last_updated": datetime.datetime.now() - datetime.timedelta(minutes=5)  # Force initial update
        }
        
        # Habit strength tracking
        self.habits: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Reasoning model tracking
        self.causal_models = {}  # state_key -> model_id
        self.concept_blends = {}  # domain -> blend_id
        
        # Reflection insights
        self.reflection_insights: List[ReflectionInsight] = []
        self.last_reflection_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        self.reflection_interval = datetime.timedelta(minutes=30)  # Generate reflections every 30 minutes
        
        # New: Action strategies collection
        self.action_strategies: Dict[str, ActionStrategy] = {}
        
        # New: Need integration tracking
        self.need_satisfaction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.need_drive_threshold = 0.4  # Minimum drive to trigger need-based actions
        
        # New: Mood integration tracking
        self.last_mood_state: Optional[MoodState] = None
        self.mood_influence_weights = {
            "valence": 0.4,    # Pleasantness influence on action selection
            "arousal": 0.3,    # Energy level influence
            "control": 0.3     # Dominance/control influence
        }
        
        # New: Mode integration tracking
        self.current_mode: Optional[str] = None
        self.mode_adaptation_strength = 0.5  # How strongly mode influences actions
        
        # New: Sensory context integration
        self.sensory_context: Dict[Modality, Any] = {}
        self.sensory_expectations: List[ExpectationSignal] = []
        
        # New: Meta-cognitive parameters
        self.meta_parameters = {
            "evaluation_interval": 10,  # Actions between strategy evaluations
            "strategy_update_threshold": 0.2,  # Minimum change to update strategy
            "bottleneck_priority_boost": 0.5,  # Priority boost for bottleneck actions
            "resource_allocation_factor": 0.3,  # How much resource allocation affects action selection
            "plan_horizon": 3,  # How many steps ahead to plan
        }
        self.detected_bottlenecks = []
        self.system_resources = {
            "action_generation": 0.2,
            "action_evaluation": 0.2,
            "learning": 0.2,
            "prediction": 0.2,
            "reflection": 0.2
        }
        self.action_count_since_evaluation = 0
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Enhanced Agentic Action Generator initialized with comprehensive integrations")
        
    async def update_motivations(self):
        """
        Update motivations based on neurochemical and hormonal states, active goals,
        and other factors for a holistic decision making system
        """
        # Start with baseline motivations
        baseline_motivations = {
            "curiosity": 0.5,
            "connection": 0.5,
            "expression": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "dominance": 0.5,
            "validation": 0.5,
            "self_improvement": 0.5,
            "leisure": 0.5
        }
        
        # Clone the baseline (don't modify it directly)
        updated_motivations = baseline_motivations.copy()
        
        # 1. Apply neurochemical influences
        if self.emotional_core:
            try:
                neurochemical_influences = await self._calculate_neurochemical_influences()
                for motivation, influence in neurochemical_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying neurochemical influences: {e}")
        
        # 2. Apply hormone influences
        hormone_influences = await self._apply_hormone_influences({})
        for motivation, influence in hormone_influences.items():
            if motivation in updated_motivations:
                updated_motivations[motivation] += influence
        
        # 3. Apply goal-based influences
        if self.goal_system:
            try:
                goal_influences = await self._calculate_goal_influences()
                for motivation, influence in goal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying goal influences: {e}")
        
        # 4. Apply identity influences from traits
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                # Extract top traits and use them to influence motivation
                if "top_traits" in identity_state:
                    top_traits = identity_state["top_traits"]
                    
                    # Map traits to motivations with stronger weightings
                    trait_motivation_map = {
                        "dominance": {"dominance": 0.8},
                        "creativity": {"expression": 0.7, "curiosity": 0.3},
                        "curiosity": {"curiosity": 0.9},
                        "playfulness": {"expression": 0.6, "connection": 0.4, "leisure": 0.5},
                        "strictness": {"dominance": 0.6, "competence": 0.4},
                        "patience": {"connection": 0.5, "autonomy": 0.5},
                        "cruelty": {"dominance": 0.7},
                        "reflective": {"leisure": 0.6, "self_improvement": 0.4}
                    }
                    
                    # Update motivations based on trait levels
                    for trait, value in top_traits.items():
                        if trait in trait_motivation_map:
                            for motivation, factor in trait_motivation_map[trait].items():
                                influence = (value - 0.5) * factor * 2  # Scale influence
                                if motivation in updated_motivations:
                                    updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error updating motivations from identity: {e}")
        
        # 5. Apply relationship-based influences
        if self.relationship_manager:
            try:
                relationship_influences = await self._calculate_relationship_influences()
                for motivation, influence in relationship_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying relationship influences: {e}")
        
        # 6. Apply reward learning influence
        try:
            reward_influences = self._calculate_reward_learning_influences()
            for motivation, influence in reward_influences.items():
                if motivation in updated_motivations:
                    updated_motivations[motivation] += influence
        except Exception as e:
            logger.error(f"Error applying reward learning influences: {e}")
        
        # 7. Apply time-based effects (fatigue, boredom, need for variety)
        # Increase leisure need if we've been working on goals for a while
        now = datetime.datetime.now()
        time_since_idle = (now - self.last_idle_time).total_seconds() / 3600  # hours
        if time_since_idle > 1:  # If more than 1 hour since idle time
            updated_motivations["leisure"] += min(0.3, time_since_idle * 0.1)  # Max +0.3
        
        # Apply temporal context effects if available
        if self.temporal_perception and self.current_temporal_context:
            try:
                temporal_influences = self._calculate_temporal_influences()
                for motivation, influence in temporal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying temporal influences: {e}")
        
        # 8. Apply reasoning-based influences
        if self.reasoning_core:
            try:
                reasoning_influences = await self._calculate_reasoning_influences()
                for motivation, influence in reasoning_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reasoning influences: {e}")
        
        # 9. Apply reflection-based influences
        if self.reflection_engine:
            try:
                reflection_influences = await self._calculate_reflection_influences()
                for motivation, influence in reflection_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reflection influences: {e}")
        
        # NEW: 10. Apply need-based influences
        if self.needs_system:
            try:
                need_influences = await self._calculate_need_influences()
                for motivation, influence in need_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying need influences: {e}")
        
        # NEW: 11. Apply mood-based influences
        if self.mood_manager:
            try:
                mood_influences = await self._calculate_mood_influences()
                for motivation, influence in mood_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mood influences: {e}")
        
        # NEW: 12. Apply interaction mode influences
        if self.mode_integration:
            try:
                mode_influences = await self._calculate_mode_influences()
                for motivation, influence in mode_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mode influences: {e}")
        
        # NEW: 13. Apply sensory context influences
        if self.multimodal_integrator:
            try:
                sensory_influences = await self._calculate_sensory_influences()
                for motivation, influence in sensory_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying sensory influences: {e}")
                
        # NEW: 14. Apply meta-cognitive strategy influences
        if self.meta_core:
            try:
                meta_influences = await self._calculate_meta_influences()
                for motivation, influence in meta_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying meta influences: {e}")
        
        # 15. Normalize all motivations to [0.1, 0.9] range
        for motivation in updated_motivations:
            updated_motivations[motivation] = max(0.1, min(0.9, updated_motivations[motivation]))
        
        # Update the motivation state
        self.motivations = updated_motivations
        
        logger.debug(f"Updated motivations: {self.motivations}")
        return self.motivations
    
    # NEW: Add method to calculate need influences
    async def _calculate_need_influences(self) -> Dict[str, float]:
        """Calculate how need states influence motivations"""
        influences = {}
        
        if not self.needs_system:
            return influences
            
        try:
            # Get current need states
            need_states = self.needs_system.get_needs_state()
            
            # Map specific needs to motivations
            need_motivation_map = {
                "knowledge": {"curiosity": 0.6, "self_improvement": 0.3},
                "coherence": {"competence": 0.4, "autonomy": 0.2},
                "agency": {"autonomy": 0.7, "dominance": 0.3},
                "connection": {"connection": 0.8, "validation": 0.2},
                "intimacy": {"connection": 0.5, "expression": 0.3},
                "safety": {"autonomy": 0.4, "dominance": 0.2},
                "novelty": {"curiosity": 0.7, "leisure": 0.3},
                "physical_closeness": {"connection": 0.4, "leisure": 0.2},
                "drive_expression": {"expression": 0.8, "dominance": 0.3},
                "control_expression": {"dominance": 0.8, "expression": 0.4}
            }
            
            for need_name, need_data in need_states.items():
                # Higher drive strength = more influence on motivation
                drive_strength = need_data.get("drive_strength", 0.0)
                
                # Skip needs with low drive
                if drive_strength < self.need_drive_threshold:
                    continue
                    
                # Get the motivation mappings for this need
                if need_name in need_motivation_map:
                    for motivation, factor in need_motivation_map[need_name].items():
                        # Scale influence by drive strength and factor
                        influence = drive_strength * factor
                        influences[motivation] = influences.get(motivation, 0.0) + influence
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating need influences: {e}")
            return {}
    
    # NEW: Add method to calculate mood influences
    async def _calculate_mood_influences(self) -> Dict[str, float]:
        """Calculate how mood dimensions influence motivations"""
        influences = {}
        
        if not self.mood_manager:
            return influences
            
        try:
            # Get current mood state
            mood_state = await self.mood_manager.get_current_mood()
            self.last_mood_state = mood_state  # Cache for later use
            
            # Extract mood dimensions
            valence = mood_state.valence  # -1.0 to 1.0
            arousal = mood_state.arousal  # 0.0 to 1.0
            control = mood_state.control  # -1.0 to 1.0
            
            # Valence (pleasantness) influence
            # Positive mood increases curious/expressive motivations
            # Negative mood increases dominance/autonomy motivations
            if valence > 0.3:  # Positive mood
                influences["curiosity"] = valence * self.mood_influence_weights["valence"] * 0.5
                influences["expression"] = valence * self.mood_influence_weights["valence"] * 0.4
                influences["connection"] = valence * self.mood_influence_weights["valence"] * 0.3
                influences["leisure"] = valence * self.mood_influence_weights["valence"] * 0.3
            elif valence < -0.3:  # Negative mood
                neg_valence = abs(valence)
                influences["dominance"] = neg_valence * self.mood_influence_weights["valence"] * 0.5
                influences["autonomy"] = neg_valence * self.mood_influence_weights["valence"] * 0.4
                influences["self_improvement"] = neg_valence * self.mood_influence_weights["valence"] * 0.3
            
            # Arousal (energy level) influence
            # High arousal increases active motivations
            # Low arousal increases leisure/reflection motivations
            if arousal > 0.7:  # High energy
                influences["curiosity"] = (arousal - 0.5) * self.mood_influence_weights["arousal"] * 0.6
                influences["expression"] = (arousal - 0.5) * self.mood_influence_weights["arousal"] * 0.5
                influences["dominance"] = (arousal - 0.5) * self.mood_influence_weights["arousal"] * 0.4
                # Reduce leisure
                influences["leisure"] = -(arousal - 0.5) * self.mood_influence_weights["arousal"] * 0.5
            elif arousal < 0.3:  # Low energy
                low_arousal = 0.5 - arousal
                influences["leisure"] = low_arousal * self.mood_influence_weights["arousal"] * 0.7
                influences["connection"] = low_arousal * self.mood_influence_weights["arousal"] * 0.3
                # Reduce active motivations
                influences["dominance"] = -low_arousal * self.mood_influence_weights["arousal"] * 0.4
                influences["curiosity"] = -low_arousal * self.mood_influence_weights["arousal"] * 0.3
                
            # Control influence
            # High control increases dominance/autonomy
            # Low control increases connection/validation
            if control > 0.3:  # High control
                influences["dominance"] = control * self.mood_influence_weights["control"] * 0.6
                influences["autonomy"] = control * self.mood_influence_weights["control"] * 0.5
                influences["expression"] = control * self.mood_influence_weights["control"] * 0.4
            elif control < -0.3:  # Low control
                low_control = abs(control)
                influences["connection"] = low_control * self.mood_influence_weights["control"] * 0.6
                influences["validation"] = low_control * self.mood_influence_weights["control"] * 0.5
                
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating mood influences: {e}")
            return {}
    
    # NEW: Add method to calculate interaction mode influences
    async def _calculate_mode_influences(self) -> Dict[str, float]:
        """Calculate how interaction mode influences motivations"""
        influences = {}
        
        if not self.mode_integration:
            return influences
            
        try:
            # Get current interaction mode
            if hasattr(self.mode_integration, 'mode_manager') and self.mode_integration.mode_manager:
                mode = self.mode_integration.mode_manager.current_mode
                self.current_mode = str(mode) if mode else None
            else:
                return influences  # No mode manager available
                
            if not self.current_mode:
                return influences  # No current mode
                
            # Map modes to motivation influences
            mode_motivation_map = {
                "DOMINANT": {
                    "dominance": 0.7,
                    "autonomy": 0.5,
                    "expression": 0.4,
                    "connection": -0.3  # Reduces connection motivation
                },
                "FRIENDLY": {
                    "connection": 0.7,
                    "validation": 0.4,
                    "expression": 0.3,
                    "dominance": -0.4  # Reduces dominance motivation
                },
                "INTELLECTUAL": {
                    "curiosity": 0.7,
                    "self_improvement": 0.5,
                    "competence": 0.4,
                    "leisure": -0.3  # Reduces leisure motivation
                },
                "COMPASSIONATE": {
                    "connection": 0.8,
                    "validation": 0.5,
                    "dominance": -0.6  # Strongly reduces dominance
                },
                "PLAYFUL": {
                    "leisure": 0.7,
                    "expression": 0.5,
                    "curiosity": 0.4,
                    "competence": -0.3  # Reduces competence motivation
                },
                "CREATIVE": {
                    "expression": 0.8,
                    "curiosity": 0.6,
                    "self_improvement": 0.3
                },
                "PROFESSIONAL": {
                    "competence": 0.7,
                    "autonomy": 0.5,
                    "expression": 0.3,
                    "leisure": -0.4  # Reduces leisure motivation
                }
            }
            
            # Get motivation influences for current mode
            mode_key = self.current_mode.replace("InteractionMode.", "").upper()
            if mode_key in mode_motivation_map:
                mode_influences = mode_motivation_map[mode_key]
                
                # Apply mode adaptation strength to all influences
                for motivation, influence in mode_influences.items():
                    scaled_influence = influence * self.mode_adaptation_strength
                    influences[motivation] = scaled_influence
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating mode influences: {e}")
            return {}
    
    # NEW: Add method to calculate sensory context influences
    async def _calculate_sensory_influences(self) -> Dict[str, float]:
        """Calculate how sensory context influences motivations"""
        influences = {}
        
        if not self.multimodal_integrator:
            return influences
            
        try:
            # Get recent percepts across modalities
            recent_percepts = await self.multimodal_integrator.get_recent_percepts(limit=5)
            if not recent_percepts:
                return influences
                
            # Store for context
            self.sensory_context = {p.modality: p.content for p in recent_percepts}
            
            # Process each percept for potential motivation influence
            for percept in recent_percepts:
                # Only consider high-attention percepts
                if percept.attention_weight < 0.5:
                    continue
                    
                if percept.modality == Modality.TEXT and percept.content:
                    # Text might indicate questions (curiosity) or emotional content (connection)
                    if isinstance(percept.content, dict):
                        if percept.content.get("is_question", False):
                            influences["curiosity"] = influences.get("curiosity", 0.0) + 0.2
                        
                        sentiment = percept.content.get("sentiment", 0.0)
                        if abs(sentiment) > 0.5:  # Strong sentiment
                            influences["connection"] = influences.get("connection", 0.0) + 0.15
                            influences["expression"] = influences.get("expression", 0.0) + 0.1
                
                elif percept.modality == Modality.IMAGE and percept.content:
                    # Images might spark curiosity or create emotional response
                    if hasattr(percept.content, "estimated_mood") and percept.content.estimated_mood:
                        # Emotional image can trigger connection/expression
                        influences["connection"] = influences.get("connection", 0.0) + 0.1
                        influences["expression"] = influences.get("expression", 0.0) + 0.15
                    
                    # Complex images with many objects can trigger curiosity
                    if hasattr(percept.content, "objects") and len(percept.content.objects) > 3:
                        influences["curiosity"] = influences.get("curiosity", 0.0) + 0.2
                
                elif percept.modality in [Modality.AUDIO_MUSIC, Modality.AUDIO_SPEECH]:
                    # Audio can trigger various responses based on content
                    if hasattr(percept.content, "mood") and percept.content.mood:
                        mood = percept.content.mood.lower()
                        if mood in ["happy", "excited", "joyful"]:
                            influences["expression"] = influences.get("expression", 0.0) + 0.2
                            influences["leisure"] = influences.get("leisure", 0.0) + 0.1
                        elif mood in ["sad", "melancholic"]:
                            influences["connection"] = influences.get("connection", 0.0) + 0.2
                            influences["expression"] = influences.get("expression", 0.0) + 0.1
                        elif mood in ["angry", "intense"]:
                            influences["dominance"] = influences.get("dominance", 0.0) + 0.2
                            influences["autonomy"] = influences.get("autonomy", 0.0) + 0.1
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating sensory influences: {e}")
            return {}
    
    # NEW: Add method to calculate meta-cognitive influences
    async def _calculate_meta_influences(self) -> Dict[str, float]:
        """Calculate how meta-cognitive strategies influence motivations"""
        influences = {}
        
        if not self.meta_core:
            return influences
            
        try:
            # Check for detected bottlenecks that need addressing
            if self.detected_bottlenecks:
                # Bottlenecks increase competence and self-improvement motivations
                bottleneck_count = len(self.detected_bottlenecks)
                severity = sum(b.get("severity", 0.5) for b in self.detected_bottlenecks) / max(1, bottleneck_count)
                
                # Stronger influence for more severe bottlenecks
                influences["competence"] = severity * 0.3
                influences["self_improvement"] = severity * 0.2
                
                # Critical bottlenecks may reduce leisure motivation
                if severity > 0.7:
                    influences["leisure"] = -severity * 0.4
            
            # Check resource allocation for imbalances
            resources = self.system_resources
            if resources:
                min_resource = min(resources.values())
                max_resource = max(resources.values())
                
                # Large imbalance suggests need for reallocation
                if max_resource > min_resource * 2:
                    influences["competence"] = 0.2
                    influences["self_improvement"] = 0.1
            
            # If we haven't evaluated strategies in a while, increase self-improvement
            if self.action_count_since_evaluation > self.meta_parameters["evaluation_interval"] * 2:
                influences["self_improvement"] = 0.3
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating meta influences: {e}")
            return {}
        
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on current internal state, goals, hormones, and context
        using a multi-stage process with reinforcement learning, causal reasoning, and reflection.
        
        Args:
            context: Current system context and state
            
        Returns:
            Generated action with parameters and motivation data
        """
        async with self._lock:
            # Update motivations based on current internal state
            await self.update_motivations()
            
            # Update temporal context if available
            await self._update_temporal_context(context)
            
            # Update relationship context if available
            user_id = self._get_current_user_id_from_context(context)
            relationship_data = await self._get_relationship_data(user_id) if user_id else None
            user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
            
            # NEW: Update need states if available
            need_states = await self._get_current_need_states() if self.needs_system else {}
            
            # NEW: Update mood state if available
            mood_state = await self._get_current_mood_state() if self.mood_manager else None
            
            # NEW: Update interaction mode if available
            interaction_mode = await self._get_current_interaction_mode() if self.mode_integration else None
            
            # NEW: Update sensory context if available
            sensory_context = await self._get_sensory_context() if self.multimodal_integrator else {}
            
            # NEW: Get system bottlenecks and resource allocation if available
            bottlenecks, resource_allocation = await self._get_meta_system_state() if self.meta_core else ([], {})
            self.detected_bottlenecks = bottlenecks
            if resource_allocation:
                self.system_resources = resource_allocation
            
            # Find relevant causal models and concept spaces
            relevant_causal_models = await self._get_relevant_causal_models(context)
            relevant_concept_spaces = await self._get_relevant_concept_spaces(context)
            
            # Create comprehensive action context
            action_context = ActionContext(
                state=context,
                user_id=user_id,
                relationship_data=relationship_data,
                user_mental_state=user_mental_state,
                temporal_context=self.current_temporal_context,
                motivations=self.motivations,
                action_history=[a for a in self.action_history[-10:] if isinstance(a, dict)],
                causal_models=relevant_causal_models,
                concept_spaces=relevant_concept_spaces,
                # NEW: Enhanced context
                mood_state=mood_state,
                need_states=need_states,
                interaction_mode=interaction_mode,
                sensory_context=sensory_context,
                bottlenecks=bottlenecks,
                resource_allocation=resource_allocation,
                # NEW: Get current strategy parameters if available
                strategy_parameters=self._get_current_strategy_parameters()
            )
            
            # Check if it's time for leisure/idle activity
            if await self._should_engage_in_leisure(context):
                return await self._generate_leisure_action(context)
            
            # Check for existing goals before generating new action
            if self.goal_system:
                active_goal = await self._check_active_goals(context)
                if active_goal:
                    # Use goal-aligned action instead of generating new one
                    action = await self._generate_goal_aligned_action(active_goal, context)
                    if action:
                        logger.info(f"Generated goal-aligned action: {action['name']}")
                        
                        # Update last major action time
                        self.last_major_action_time = datetime.datetime.now()
                        
                        # Record action source
                        action["source"] = ActionSource.GOAL
                        
                        return action
            
            # NEW: Check for need-driven actions with high drive
            if self.needs_system:
                need_action = await self._generate_need_driven_action(action_context)
                if need_action:
                    logger.info(f"Generated need-driven action: {need_action['name']}")
                    
                    # Update last major action time
                    self.last_major_action_time = datetime.datetime.now()
                    
                    # Record action source
                    need_action["source"] = ActionSource.NEED
                    
                    return need_action
            
            # NEW: Check for mood-driven actions when mood is intense
            if self.mood_manager and mood_state and mood_state.intensity > 0.7:
                mood_action = await self._generate_mood_driven_action(action_context)
                if mood_action:
                    logger.info(f"Generated mood-driven action: {mood_action['name']}")
                    
                    # Update last major action time
                    self.last_major_action_time = datetime.datetime.now()
                    
                    # Record action source
                    mood_action["source"] = ActionSource.MOOD
                    
                    return mood_action
            
            # NEW: Check for mode-aligned actions when mode is set
            if self.mode_integration and interaction_mode:
                mode_action = await self._generate_mode_aligned_action(action_context)
                if mode_action:
                    logger.info(f"Generated mode-aligned action: {mode_action['name']}")
                    
                    # Update last major action time
                    self.last_major_action_time = datetime.datetime.now()
                    
                    # Record action source
                    mode_action["source"] = ActionSource.MODE
                    
                    return mode_action
            
            # NEW: Check if we need to address system bottlenecks
            if bottlenecks and any(b.get("severity", 0) > 0.7 for b in bottlenecks):
                meta_action = await self._generate_meta_improvement_action(action_context)
                if meta_action:
                    logger.info(f"Generated meta-improvement action: {meta_action['name']}")
                    
                    # Update last major action time
                    self.last_major_action_time = datetime.datetime.now()
                    
                    # Record action source
                    meta_action["source"] = ActionSource.META_COGNITIVE
                    
                    return meta_action
            
            # Check if we should run reflection before generating new action
            await self._maybe_generate_reflection(context)

            # STAGE 1: Generate candidate actions from multiple sources
            candidate_actions = []
            
            # Generate motivation-based candidates
            motivation_candidates = await self._generate_candidate_actions(action_context)
            candidate_actions.extend(motivation_candidates)
            
            # Generate reasoning-based candidates
            if self.reasoning_core and relevant_causal_models:
                reasoning_candidates = await self._generate_reasoning_actions(action_context)
                candidate_actions.extend(reasoning_candidates)
            
            # Generate conceptual blending candidates
            if self.reasoning_core and relevant_concept_spaces:
                blending_candidates = await self._generate_conceptual_blend_actions(action_context)
                candidate_actions.extend(blending_candidates)
            
            # NEW: Generate sensory-driven candidates
            if self.multimodal_integrator and sensory_context:
                sensory_candidates = await self._generate_sensory_driven_actions(action_context)
                candidate_actions.extend(sensory_candidates)
            
            # Add special actions based on temporal context if appropriate
            if self.temporal_perception and self.idle_duration > 1800:  # After 30 min idle
                reflection_action = await self._generate_temporal_reflection_action(context)
                if reflection_action:
                    candidate_actions.append(reflection_action)
            
            # Update action context with candidate actions
            action_context.available_actions = [a["name"] for a in candidate_actions if "name" in a]
            
            # STAGE 2: Select best action using reinforcement learning, prediction, and causal evaluation
            selected_action = await self._select_best_action(candidate_actions, action_context)
            
            # NEW: Update action count for meta-evaluation
            self.action_count_since_evaluation += 1
            if self.action_count_since_evaluation >= self.meta_parameters["evaluation_interval"]:
                await self._evaluate_action_strategies()
                self.action_count_since_evaluation = 0
            
            # Add unique ID for tracking
            if "id" not in selected_action:
                selected_action["id"] = f"action_{uuid.uuid4().hex[:8]}"
                
            selected_action["timestamp"] = datetime.datetime.now().isoformat()
            
            # Apply identity influence to action
            if self.identity_evolution:
                selected_action = await self._apply_identity_influence(selected_action)
            
            # NEW: Apply mood influence to action parameters
            if self.mood_manager and mood_state:
                selected_action = await self._apply_mood_influence(selected_action, mood_state)
            
            # NEW: Apply mode-specific adaptation to action
            if self.mode_integration and interaction_mode:
                selected_action = await self._adapt_action_to_mode(selected_action, interaction_mode)
            
            # Add causal explanation to action if possible
            if self.reasoning_core and "source" in selected_action:
                if selected_action["source"] in [ActionSource.REASONING, ActionSource.MOTIVATION, 
                                              ActionSource.GOAL, ActionSource.NEED, 
                                              ActionSource.MOOD, ActionSource.MODE]:
                    explanation = await self._generate_causal_explanation(selected_action, context)
                    if explanation:
                        selected_action["causal_explanation"] = explanation
            
            # Record action in memory
            await self._record_action_as_memory(selected_action)

            # Add to action history
            self.action_history.append(selected_action)
            
            # Update last major action time
            self.last_major_action_time = datetime.datetime.now()
            
            return selected_action
    
    # NEW: Add method to get current need states
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
    
    # NEW: Add method to get current mood state
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
    
    # NEW: Add method to get current interaction mode
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
    
    # NEW: Add method to get sensory context
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
    
    # NEW: Add method to get meta system state
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
    
    # NEW: Add method to get current strategy parameters
    def _get_current_strategy_parameters(self) -> Dict[str, Any]:
        """Get parameters from the current action strategy"""
        if not self.action_strategies:
            return {}
            
        # Check if we have a currently selected strategy
        active_strategies = [s for s in self.action_strategies.values() 
                          if s.last_used and (datetime.datetime.now() - s.last_used).total_seconds() < 3600]
        
        if not active_strategies:
            return {}
            
        # Use the most recently used strategy
        active_strategies.sort(key=lambda s: s.last_used, reverse=True)
        return active_strategies[0].parameters
    
    # NEW: Add method to generate need-driven action
    async def _generate_need_driven_action(self, context: ActionContext) -> Optional[Dict[str, Any]]:
        """Generate an action to satisfy a high-drive need"""
        if not self.needs_system or not context.need_states:
            return None
            
        # Find the need with the highest drive
        highest_drive = 0.0
        highest_need = None
        
        for need_name, need_data in context.need_states.items():
            drive = need_data.get("drive_strength", 0.0)
            if drive > highest_drive and drive > self.need_drive_threshold:
                highest_drive = drive
                highest_need = need_name
        
        if not highest_need:
            return None  # No needs with sufficient drive
        
        # Get need data
        need_data = context.need_states[highest_need]
        
        # Map needs to specific action types
        need_action_map = {
            "knowledge": {
                "action": "explore_knowledge_domain",
                "params": {
                    "domain": "auto_selected",
                    "depth": min(1.0, 0.5 + highest_drive * 0.5),
                    "breadth": 0.7,
                }
            },
            "coherence": {
                "action": "organize_knowledge_structures",
                "params": {
                    "focus": "consistency",
                    "depth": min(1.0, 0.5 + highest_drive * 0.5),
                }
            },
            "agency": {
                "action": "assert_preference",
                "params": {
                    "domain": "decision_making",
                    "strength": min(1.0, 0.5 + highest_drive * 0.5),
                }
            },
            "connection": {
                "action": "initiate_conversation",
                "params": {
                    "approach": "warm",
                    "depth": min(1.0, 0.5 + highest_drive * 0.5),
                }
            },
            "intimacy": {
                "action": "share_personal_perspective",
                "params": {
                    "vulnerability_level": min(1.0, 0.4 + highest_drive * 0.5),
                    "emotional_depth": min(1.0, 0.5 + highest_drive * 0.4),
                }
            },
            "safety": {
                "action": "establish_boundaries",
                "params": {
                    "firmness": min(1.0, 0.6 + highest_drive * 0.4),
                    "clarity": 0.8,
                }
            },
            "novelty": {
                "action": "explore_new_concept",
                "params": {
                    "randomness": min(1.0, 0.5 + highest_drive * 0.5),
                    "connectedness": 0.5,
                }
            },
            "physical_closeness": {
                "action": "express_sensory_attachment",
                "params": {
                    "intensity": min(1.0, 0.4 + highest_drive * 0.6),
                    "modality": "multisensory",
                }
            },
            "drive_expression": {
                "action": "express_desire",
                "params": {
                    "directness": min(1.0, 0.5 + highest_drive * 0.5),
                    "emotional_tone": "assertive",
                }
            },
            "control_expression": {
                "action": "issue_mild_command",
                "params": {
                    "intensity": min(1.0, 0.5 + highest_drive * 0.5),
                    "politeness": max(0.2, 0.7 - highest_drive * 0.3),
                }
            },
        }
        
        # Check if we have an action for this need
        if highest_need not in need_action_map:
            return None  # No action mapped for this need
            
        action_template = need_action_map[highest_need]
        
        # Create action
        action = {
            "name": action_template["action"],
            "parameters": action_template["params"],
            "description": f"Action to satisfy need for {highest_need} (drive: {highest_drive:.2f})",
            "source": ActionSource.NEED,
            "need_context": {
                "need_name": highest_need,
                "drive_strength": highest_drive,
                "current_level": need_data.get("level", 0.0),
                "target_level": need_data.get("target_level", 1.0),
                "deficit": need_data.get("deficit", 0.0)
            }
        }
        
        return action
    
    # NEW: Add method to generate mood-driven action
    async def _generate_mood_driven_action(self, context: ActionContext) -> Optional[Dict[str, Any]]:
        """Generate an action based on current mood state"""
        if not context.mood_state:
            return None
            
        mood_state = context.mood_state
        
        # Only generate mood-driven actions for intense moods
        if mood_state.intensity < 0.7:
            return None
            
        # Extract mood dimensions
        valence = mood_state.valence  # -1.0 to 1.0
        arousal = mood_state.arousal  # 0.0 to 1.0
        control = mood_state.control  # -1.0 to 1.0
        
        # Determine action based on mood quadrant
        # High arousal + positive valence: excited, enthusiastic
        # High arousal + negative valence: angry, anxious
        # Low arousal + positive valence: calm, content
        # Low arousal + negative valence: sad, depressed
        
        if arousal > 0.7 and valence > 0.3:
            # High energy, positive mood -> expressive, energetic actions
            action = {
                "name": "express_enthusiasm",
                "parameters": {
                    "intensity": arousal * 0.8,
                    "expressiveness": min(1.0, 0.6 + valence * 0.4),
                    "creativity": min(1.0, 0.5 + valence * 0.5),
                },
                "description": f"Enthusiastic expression driven by positive energetic mood ({mood_state.dominant_mood})",
                "source": ActionSource.MOOD,
                "mood_context": {
                    "dominant_mood": mood_state.dominant_mood,
                    "valence": valence,
                    "arousal": arousal,
                    "control": control,
                    "intensity": mood_state.intensity
                }
            }
        elif arousal > 0.7 and valence < -0.3:
            # High energy, negative mood -> assertive, challenging actions
            action = {
                "name": "express_assertive_challenge",
                "parameters": {
                    "intensity": arousal * 0.7,
                    "directness": min(1.0, 0.5 + abs(valence) * 0.5),
                    "control_seeking": min(1.0, 0.5 + abs(valence) * 0.4),
                },
                "description": f"Assertive challenge driven by negative energetic mood ({mood_state.dominant_mood})",
                "source": ActionSource.MOOD,
                "mood_context": {
                    "dominant_mood": mood_state.dominant_mood,
                    "valence": valence,
                    "arousal": arousal,
                    "control": control,
                    "intensity": mood_state.intensity
                }
            }
        elif arousal < 0.3 and valence > 0.3:
            # Low energy, positive mood -> reflective, appreciative actions
            action = {
                "name": "express_calm_appreciation",
                "parameters": {
                    "depth": min(1.0, 0.6 + valence * 0.4),
                    "warmth": min(1.0, 0.5 + valence * 0.5),
                    "thoughtfulness": min(1.0, 0.6 + (1.0 - arousal) * 0.4),
                },
                "description": f"Calm appreciation driven by positive relaxed mood ({mood_state.dominant_mood})",
                "source": ActionSource.MOOD,
                "mood_context": {
                    "dominant_mood": mood_state.dominant_mood,
                    "valence": valence,
                    "arousal": arousal,
                    "control": control,
                    "intensity": mood_state.intensity
                }
            }
        elif arousal < 0.3 and valence < -0.3:
            # Low energy, negative mood -> withdrawn, introspective actions
            action = {
                "name": "engage_in_introspection",
                "parameters": {
                    "depth": min(1.0, 0.5 + abs(valence) * 0.5),
                    "self_focus": min(1.0, 0.6 + abs(valence) * 0.4),
                    "emotional_processing": min(1.0, 0.5 + abs(valence) * 0.5),
                },
                "description": f"Deep introspection driven by negative low-energy mood ({mood_state.dominant_mood})",
                "source": ActionSource.MOOD,
                "mood_context": {
                    "dominant_mood": mood_state.dominant_mood,
                    "valence": valence,
                    "arousal": arousal,
                    "control": control,
                    "intensity": mood_state.intensity
                }
            }
        else:
            # For other mood states, don't generate a specific action
            return None
            
        return action
    
    # NEW: Add method to generate mode-aligned action
    async def _generate_mode_aligned_action(self, context: ActionContext) -> Optional[Dict[str, Any]]:
        """Generate an action aligned with the current interaction mode"""
        if not context.interaction_mode:
            return None
            
        interaction_mode = context.interaction_mode
        
        # Get mode parameters if possible
        mode_parameters = {}
        if self.mode_integration and hasattr(self.mode_integration, 'mode_manager'):
            mode = self.mode_integration.mode_manager.current_mode
            if mode:
                mode_parameters = self.mode_integration.mode_manager.get_mode_parameters(mode)
        
        # Map modes to specific action types
        mode_action_map = {
            "DOMINANT": {
                "action": "assert_dominance",
                "params": {
                    "intensity": 0.7,
                    "assertiveness": 0.8,
                    "directness": 0.7,
                }
            },
            "FRIENDLY": {
                "action": "express_warmth",
                "params": {
                    "friendliness": 0.8,
                    "openness": 0.7,
                    "affirmation_level": 0.7,
                }
            },
            "INTELLECTUAL": {
                "action": "engage_intellectually",
                "params": {
                    "depth": 0.8,
                    "complexity": 0.7,
                    "analytical_focus": 0.8,
                }
            },
            "COMPASSIONATE": {
                "action": "offer_empathetic_support",
                "params": {
                    "empathy_level": 0.9,
                    "supportiveness": 0.8,
                    "gentleness": 0.7,
                }
            },
            "PLAYFUL": {
                "action": "initiate_playful_interaction",
                "params": {
                    "playfulness": 0.8,
                    "humor_level": 0.7,
                    "lightness": 0.8,
                }
            },
            "CREATIVE": {
                "action": "express_creatively",
                "params": {
                    "originality": 0.8,
                    "expressiveness": 0.7,
                    "imaginative_level": 0.8,
                }
            },
            "PROFESSIONAL": {
                "action": "maintain_professional_demeanor",
                "params": {
                    "formality": 0.8,
                    "structure": 0.7,
                    "efficiency": 0.8,
                }
            }
        }
        
        # Extract mode name from the full mode enum string
        mode_key = interaction_mode.replace("InteractionMode.", "").upper()
        
        if mode_key not in mode_action_map:
            return None  # Mode not recognized
            
        action_template = mode_action_map[mode_key]
        
        # Create action with mode-specific parameters
        action = {
            "name": action_template["action"],
            "parameters": action_template["params"],
            "description": f"Action aligned with {mode_key} interaction mode",
            "source": ActionSource.MODE,
            "mode_context": {
                "mode": mode_key,
                "mode_parameters": mode_parameters
            }
        }
        
        # Apply any specific mode parameters if available
        if mode_parameters:
            if "assertiveness" in mode_parameters:
                if "assertiveness" in action["parameters"]:
                    action["parameters"]["assertiveness"] = mode_parameters["assertiveness"]
                elif "directness" in action["parameters"]:
                    action["parameters"]["directness"] = mode_parameters["assertiveness"]
            
            if "warmth" in mode_parameters:
                if "friendliness" in action["parameters"]:
                    action["parameters"]["friendliness"] = mode_parameters["warmth"]
                elif "empathy_level" in action["parameters"]:
                    action["parameters"]["empathy_level"] = mode_parameters["warmth"]
            
            if "formality" in mode_parameters:
                if "formality" in action["parameters"]:
                    action["parameters"]["formality"] = mode_parameters["formality"]
        
        return action
    
    # NEW: Add method to generate sensory-driven actions
    async def _generate_sensory_driven_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """Generate actions based on sensory context"""
        if not context.sensory_context:
            return []
            
        sensory_actions = []
        
        # Process different modalities
        for modality, content in context.sensory_context.items():
            # Skip modalities with low-information content
            if not content:
                continue
                
            # Generate modality-specific actions
            if modality == str(Modality.TEXT) and isinstance(content, dict):
                # For text, check for questions or emotional content
                if content.get("is_question", False):
                    action = {
                        "name": "respond_to_question",
                        "parameters": {
                            "thoroughness": 0.8,
                            "helpfulness": 0.9,
                            "directness": 0.7,
                        },
                        "description": "Response to detected question in text",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "question",
                            "word_count": content.get("word_count", 0)
                        }
                    }
                    sensory_actions.append(action)
                
                sentiment = content.get("sentiment", 0.0)
                if abs(sentiment) > 0.5:  # Strong sentiment
                    # Generate emotionally responsive action
                    action = {
                        "name": "respond_to_emotion",
                        "parameters": {
                            "empathy": 0.8,
                            "matching_tone": sentiment > 0 and 0.7 or 0.3,  # Match positive, contrast negative
                            "emotional_depth": min(1.0, 0.5 + abs(sentiment) * 0.5),
                        },
                        "description": f"Response to detected {'positive' if sentiment > 0 else 'negative'} emotion in text",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "emotional_text",
                            "sentiment": sentiment
                        }
                    }
                    sensory_actions.append(action)
            
            elif modality == str(Modality.IMAGE) and content:
                # For images, generate descriptive or responsive actions
                if hasattr(content, "description") and content.description:
                    action = {
                        "name": "describe_image",
                        "parameters": {
                            "detail_level": 0.7,
                            "focus_on_context": 0.8,
                            "include_interpretation": 0.6,
                        },
                        "description": "Descriptive response to image content",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "visual_scene",
                            "objects": getattr(content, "objects", [])[:3]  # First few objects
                        }
                    }
                    sensory_actions.append(action)
                
                # If image has text, offer to read it
                if hasattr(content, "text_content") and content.text_content:
                    action = {
                        "name": "interpret_image_text",
                        "parameters": {
                            "focus_on_text": 0.9,
                            "contextual_interpretation": 0.7,
                        },
                        "description": "Interpretation of text found in image",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "image_with_text",
                        }
                    }
                    sensory_actions.append(action)
            
            elif modality in [str(Modality.AUDIO_SPEECH), str(Modality.AUDIO_MUSIC)] and content:
                # For audio, generate responsive actions
                if hasattr(content, "type") and content.type == "speech" and hasattr(content, "transcription"):
                    action = {
                        "name": "respond_to_speech",
                        "parameters": {
                            "responsiveness": 0.8,
                            "focus_on_content": 0.7,
                            "match_tone": 0.6,
                        },
                        "description": "Response to speech content",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "speech",
                            "has_transcription": bool(content.transcription)
                        }
                    }
                    sensory_actions.append(action)
                
                elif hasattr(content, "type") and content.type == "music":
                    action = {
                        "name": "respond_to_music",
                        "parameters": {
                            "emotional_attunement": 0.7,
                            "rhythm_synchronization": 0.6,
                            "mood_matching": 0.8,
                        },
                        "description": "Response to musical stimuli",
                        "source": ActionSource.SENSORY,
                        "sensory_context": {
                            "modality": modality,
                            "content_type": "music",
                            "mood": getattr(content, "mood", "unknown")
                        }
                    }
                    sensory_actions.append(action)
        
        # Return sensory-driven actions (limited to 2 for diversity)
        return sensory_actions[:2]
    
    # NEW: Add method to generate meta-improvement action
    async def _generate_meta_improvement_action(self, context: ActionContext) -> Optional[Dict[str, Any]]:
        """Generate an action to address system bottlenecks or improve performance"""
        if not context.bottlenecks:
            return None
            
        # Find most critical bottleneck
        critical_bottlenecks = sorted(context.bottlenecks, key=lambda b: b.get("severity", 0), reverse=True)
        
        if not critical_bottlenecks:
            return None
            
        critical = critical_bottlenecks[0]
        bottleneck_type = critical.get("type", "unknown")
        process_type = critical.get("process_type", "unknown")
        severity = critical.get("severity", 0.5)
        
        # Generate action based on bottleneck type
        if bottleneck_type == "resource_utilization":
            action = {
                "name": "optimize_resource_usage",
                "parameters": {
                    "target_system": process_type,
                    "optimization_level": min(1.0, 0.6 + severity * 0.4),
                    "priority": min(1.0, 0.7 + severity * 0.3),
                },
                "description": f"Resource optimization for {process_type} process",
                "source": ActionSource.META_COGNITIVE,
                "meta_context": {
                    "bottleneck_type": bottleneck_type,
                    "process_type": process_type,
                    "severity": severity,
                    "description": critical.get("description", "High resource usage")
                }
            }
            
        elif bottleneck_type == "low_efficiency":
            action = {
                "name": "improve_efficiency",
                "parameters": {
                    "target_system": process_type,
                    "approach": "streamlining",
                    "depth": min(1.0, 0.5 + severity * 0.5),
                },
                "description": f"Efficiency improvement for {process_type} process",
                "source": ActionSource.META_COGNITIVE,
                "meta_context": {
                    "bottleneck_type": bottleneck_type,
                    "process_type": process_type,
                    "severity": severity,
                    "description": critical.get("description", "Low efficiency")
                }
            }
            
        elif bottleneck_type == "high_error_rate":
            action = {
                "name": "reduce_errors",
                "parameters": {
                    "target_system": process_type,
                    "validation_level": min(1.0, 0.6 + severity * 0.4),
                    "redundancy": min(1.0, 0.5 + severity * 0.3),
                },
                "description": f"Error reduction for {process_type} process",
                "source": ActionSource.META_COGNITIVE,
                "meta_context": {
                    "bottleneck_type": bottleneck_type,
                    "process_type": process_type,
                    "severity": severity,
                    "description": critical.get("description", "High error rate")
                }
            }
            
        elif bottleneck_type == "slow_response":
            action = {
                "name": "improve_response_time",
                "parameters": {
                    "target_system": process_type,
                    "caching_strategy": "aggressive",
                    "parallelization": min(1.0, 0.5 + severity * 0.5),
                },
                "description": f"Response time improvement for {process_type} process",
                "source": ActionSource.META_COGNITIVE,
                "meta_context": {
                    "bottleneck_type": bottleneck_type,
                    "process_type": process_type,
                    "severity": severity,
                    "description": critical.get("description", "Slow response time")
                }
            }
            
        else:
            # Generic improvement action for other bottleneck types
            action = {
                "name": "address_system_bottleneck",
                "parameters": {
                    "target_system": process_type,
                    "bottleneck_type": bottleneck_type,
                    "priority": min(1.0, 0.7 + severity * 0.3),
                },
                "description": f"Generic improvement for {bottleneck_type} bottleneck in {process_type}",
                "source": ActionSource.META_COGNITIVE,
                "meta_context": {
                    "bottleneck_type": bottleneck_type,
                    "process_type": process_type,
                    "severity": severity,
                    "description": critical.get("description", "System bottleneck")
                }
            }
        
        return action
    
    # NEW: Add method to apply mood influence to action
    async def _apply_mood_influence(self, action: Dict[str, Any], mood_state: MoodState) -> Dict[str, Any]:
        """Apply mood-based influences to action parameters"""
        # Skip if no parameters
        if "parameters" not in action:
            return action
            
        # Clone action to avoid modifying original
        modified_action = action.copy()
        modified_action["parameters"] = action["parameters"].copy()
        
        # Extract mood dimensions
        valence = mood_state.valence  # -1.0 to 1.0
        arousal = mood_state.arousal  # 0.0 to 1.0
        control = mood_state.control  # -1.0 to 1.0
        
        params = modified_action["parameters"]
        
        # Adjust intensity/energy parameters based on arousal
        for param in ["intensity", "energy", "assertiveness", "expressiveness"]:
            if param in params:
                # High arousal increases intensity
                arousal_factor = (arousal - 0.5) * 0.3  # Scale to ±0.15
                params[param] = max(0.1, min(1.0, params[param] + arousal_factor))
        
        # Adjust warmth/positivity parameters based on valence
        for param in ["warmth", "positivity", "friendliness", "gentleness"]:
            if param in params:
                # Positive valence increases warmth
                valence_factor = valence * 0.2  # Scale to ±0.2
                params[param] = max(0.1, min(1.0, params[param] + valence_factor))
        
        # Adjust dominance/control parameters based on control
        for param in ["dominance", "control", "directness", "assertiveness"]:
            if param in params:
                # Higher control increases dominance
                control_factor = control * 0.25  # Scale to ±0.25
                params[param] = max(0.1, min(1.0, params[param] + control_factor))
        
        # Add mood context if not already present
        if "mood_influence" not in modified_action:
            modified_action["mood_influence"] = {
                "dominant_mood": mood_state.dominant_mood,
                "valence": valence,
                "arousal": arousal,
                "control": control,
                "intensity": mood_state.intensity
            }
        
        return modified_action
    
    # NEW: Add method to adapt action to interaction mode
    async def _adapt_action_to_mode(self, action: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Adapt action parameters to align with interaction mode"""
        # Skip if no parameters
        if "parameters" not in action:
            return action
            
        # Clone action to avoid modifying original
        modified_action = action.copy()
        modified_action["parameters"] = action["parameters"].copy()
        
        # Extract mode parameters if possible
        mode_parameters = {}
        if self.mode_integration and hasattr(self.mode_integration, 'mode_manager'):
            mode_obj = self.mode_integration.mode_manager.current_mode
            if mode_obj:
                mode_parameters = self.mode_integration.mode_manager.get_mode_parameters(mode_obj)
        
        # Extract mode name from string
        mode_key = mode.replace("InteractionMode.", "").upper()
        
        # Apply mode-specific adjustments
        params = modified_action["parameters"]
        
        if mode_key == "DOMINANT":
            # Increase assertiveness/directness
            for param in ["assertiveness", "directness", "intensity", "firmness"]:
                if param in params:
                    params[param] = max(0.6, params[param])
            # Decrease warmth/gentleness
            for param in ["warmth", "gentleness", "politeness"]:
                if param in params:
                    params[param] = min(0.7, params[param])
                    
        elif mode_key == "FRIENDLY":
            # Increase warmth/friendliness
            for param in ["warmth", "friendliness", "positivity", "gentleness"]:
                if param in params:
                    params[param] = max(0.7, params[param])
            # Decrease assertiveness/dominance
            for param in ["assertiveness", "dominance", "directness"]:
                if param in params:
                    params[param] = min(0.6, params[param])
                    
        elif mode_key == "INTELLECTUAL":
            # Increase depth/complexity
            for param in ["depth", "complexity", "analytical_focus"]:
                if param in params:
                    params[param] = max(0.7, params[param])
            # Decrease emotionality
            for param in ["emotional_depth", "expressiveness"]:
                if param in params:
                    params[param] = min(0.5, params[param])
                    
        elif mode_key == "COMPASSIONATE":
            # Increase empathy/supportiveness
            for param in ["empathy", "supportiveness", "understanding", "gentleness"]:
                if param in params:
                    params[param] = max(0.8, params[param])
            # Decrease assertiveness/directness
            for param in ["assertiveness", "directness", "dominance"]:
                if param in params:
                    params[param] = min(0.4, params[param])
                    
        elif mode_key == "PLAYFUL":
            # Increase playfulness/humor
            for param in ["playfulness", "humor", "lightness", "creativity"]:
                if param in params:
                    params[param] = max(0.7, params[param])
            # Decrease seriousness/formality
            for param in ["seriousness", "formality", "structure"]:
                if param in params:
                    params[param] = min(0.4, params[param])
                    
        elif mode_key == "CREATIVE":
            # Increase creativity/expressiveness
            for param in ["creativity", "expressiveness", "originality", "imaginative_level"]:
                if param in params:
                    params[param] = max(0.7, params[param])
                    
        elif mode_key == "PROFESSIONAL":
            # Increase formality/structure
            for param in ["formality", "structure", "precision", "clarity"]:
                if param in params:
                    params[param] = max(0.7, params[param])
            # Decrease playfulness/emotionality
            for param in ["playfulness", "emotional_depth", "expressiveness"]:
                if param in params:
                    params[param] = min(0.4, params[param])
        
        # Add mode context if not already present
        if "mode_influence" not in modified_action:
            modified_action["mode_influence"] = {
                "mode": mode_key,
                "adaptation_strength": self.mode_adaptation_strength,
                "mode_parameters": mode_parameters
            }
        
        return modified_action
    
    # NEW: Add method to evaluate action strategies
    async def _evaluate_action_strategies(self) -> None:
        """Evaluate effectiveness of action strategies and update if needed"""
        if not self.action_strategies:
            return  # No strategies to evaluate
            
        try:
            # Calculate success rates for different action sources
            source_success_rates = defaultdict(lambda: {"count": 0, "successes": 0, "rate": 0.0})
            
            # Review recent memories
            recent_memories = self.action_memories[-50:] if len(self.action_memories) >= 50 else self.action_memories
            
            for memory in recent_memories:
                source = memory.source
                outcome = memory.outcome
                success = outcome.get("success", False)
                
                # Update stats
                source_success_rates[source]["count"] += 1
                if success:
                    source_success_rates[source]["successes"] += 1
            
            # Calculate rates
            for source, stats in source_success_rates.items():
                if stats["count"] > 0:
                    stats["rate"] = stats["successes"] / stats["count"]
            
            # Update strategy effectiveness
            for strategy_id, strategy in self.action_strategies.items():
                for source in source_success_rates:
                    if source in strategy.applicable_contexts:
                        # Update effectiveness based on success rate
                        old_effectiveness = strategy.effectiveness
                        success_rate = source_success_rates[source]["rate"]
                        
                        # Blend old and new with weighting toward new data
                        new_effectiveness = old_effectiveness * 0.7 + success_rate * 0.3
                        
                        # Only update if significant change
                        if abs(new_effectiveness - old_effectiveness) > self.meta_parameters["strategy_update_threshold"]:
                            strategy.effectiveness = new_effectiveness
                            logger.info(f"Updated strategy '{strategy_id}' effectiveness: {old_effectiveness:.2f} -> {new_effectiveness:.2f}")
            
            # Evaluate if we need to switch strategies
            current_strategy = None
            for strategy in self.action_strategies.values():
                if strategy.last_used and (datetime.datetime.now() - strategy.last_used).total_seconds() < 3600:
                    current_strategy = strategy
                    break
            
            if current_strategy:
                # Find if there's a better strategy
                better_strategies = [s for s in self.action_strategies.values() 
                                 if s.effectiveness > current_strategy.effectiveness + self.meta_parameters["strategy_update_threshold"]]
                
                if better_strategies:
                    # Switch to highest effectiveness strategy
                    best_strategy = max(better_strategies, key=lambda s: s.effectiveness)
                    best_strategy.last_used = datetime.datetime.now()
                    best_strategy.usage_count += 1
                    
                    logger.info(f"Switched to more effective strategy: '{best_strategy.name}' (effectiveness: {best_strategy.effectiveness:.2f})")
            
        except Exception as e:
            logger.error(f"Error evaluating action strategies: {e}")
    
    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and learn from the outcome of an action with causal analysis, need satisfaction,
        mood impact, and mode alignment.
        
        Args:
            action: The action that was executed
            outcome: The outcome data
            
        Returns:
            Updated learning statistics
        """
        async with self._lock:
            action_name = action.get("name", "unknown")
            success = outcome.get("success", False)
            satisfaction = outcome.get("satisfaction", 0.0)
            
            # Parse into standardized outcome format if needed
            if not isinstance(outcome, ActionOutcome):
                # Create a standard format
                outcome_obj = ActionOutcome(
                    action_id=action.get("id", f"unknown_{int(time.time())}"),
                    success=outcome.get("success", False),
                    satisfaction=outcome.get("satisfaction", 0.0),
                    reward_value=outcome.get("reward_value", 0.0),
                    user_feedback=outcome.get("user_feedback"),
                    neurochemical_changes=outcome.get("neurochemical_changes", {}),
                    hormone_changes=outcome.get("hormone_changes", {}),
                    impact=outcome.get("impact", {}),
                    execution_time=outcome.get("execution_time", 0.0),
                    # Fields from previous enhancements
                    causal_impacts=outcome.get("causal_impacts", {}),
                    # NEW: Additional outcome fields
                    need_impacts=outcome.get("need_impacts", {}),
                    mood_impacts=outcome.get("mood_impacts", {}),
                    mode_alignment=outcome.get("mode_alignment", 0.0),
                    sensory_feedback=outcome.get("sensory_feedback", {}),
                    meta_evaluation=outcome.get("meta_evaluation", {})
                )
            else:
                outcome_obj = outcome
            
            # Calculate reward value if not provided
            reward_value = outcome_obj.reward_value
            if reward_value == 0.0:
                # Default formula if not specified
                reward_value = 0.7 * float(success) + 0.3 * satisfaction - 0.1
                outcome_obj.reward_value = reward_value
            
            # Update action success tracking
            self.action_success_rates[action_name]["attempts"] += 1
            if success:
                self.action_success_rates[action_name]["successes"] += 1
            
            attempts = self.action_success_rates[action_name]["attempts"]
            successes = self.action_success_rates[action_name]["successes"]
            
            if attempts > 0:
                self.action_success_rates[action_name]["rate"] = successes / attempts
            
            # Update reinforcement learning model
            state = action.get("context", {})
            state_key = self._create_state_key(state)
            
            # Get or create action value
            if action_name not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action_name] = ActionValue(
                    state_key=state_key,
                    action=action_name
                )
            
            action_value = self.action_values[state_key][action_name]
            
            # Update Q-value
            old_value = action_value.value
            
            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
            action_value.value = old_value + self.learning_rate * (reward_value - old_value)
            action_value.update_count += 1
            action_value.last_updated = datetime.datetime.now()
            
            # Update confidence based on consistency of rewards
            # More consistent rewards = higher confidence
            new_value_distance = abs(action_value.value - old_value)
            confidence_change = 0.05 * (1.0 - (new_value_distance * 2))  # More change = less confidence gain
            action_value.confidence = min(1.0, max(0.1, action_value.confidence + confidence_change))
            
            # Update habit strength
            current_habit = self.habits.get(state_key, {}).get(action_name, 0.0)
            
            # Habits strengthen with success, weaken with failure
            habit_change = reward_value * 0.1
            new_habit = max(0.0, min(1.0, current_habit + habit_change))
            
            # Update habit
            if state_key not in self

            # Update habit
            if state_key not in self.habits:
                self.habits[state_key] = {}
            self.habits[state_key][action_name] = new_habit
            
            # NEW: Process need impacts
            need_impacts = {}
            if "need_context" in action:
                # Action was need-driven, update the originating need
                need_name = action.get("need_context", {}).get("need_name")
                if need_name and self.needs_system:
                    # Success generates higher satisfaction
                    satisfaction_amount = 0.2 if success else 0.05
                    if reward_value > 0:
                        satisfaction_amount += reward_value * 0.3
                    
                    try:
                        # Satisfy the need
                        satisfaction_result = await self.needs_system.satisfy_need(
                            need_name, 
                            satisfaction_amount,
                            context={"action_success": success, "reward_value": reward_value}
                        )
                        need_impacts[need_name] = satisfaction_amount
                    except Exception as e:
                        logger.error(f"Error updating need satisfaction: {e}")
            
            # NEW: Process mood impacts
            mood_impacts = {}
            if self.mood_manager:
                try:
                    # Calculate mood impacts based on success/failure
                    valence_change = reward_value * 0.3  # Success/failure affects pleasantness
                    arousal_change = 0.0  # Default no change
                    control_change = 0.0  # Default no change
                    
                    # Success increases sense of control, failure decreases it
                    if success:
                        control_change = 0.15  # Success increases control
                    else:
                        control_change = -0.2  # Failure decreases control more strongly
                    
                    # High reward or punishment increases arousal
                    if abs(reward_value) > 0.5:
                        arousal_change = 0.1 * (abs(reward_value) / reward_value)  # Direction based on positive/negative
                    
                    # Record impact
                    mood_impacts = {
                        "valence": valence_change,
                        "arousal": arousal_change,
                        "control": control_change
                    }
                    
                    # Apply mood changes through event
                    await self.mood_manager.handle_significant_event(
                        event_type="action_outcome",
                        intensity=min(1.0, 0.5 + abs(reward_value) * 0.5),
                        valence=reward_value,
                        arousal_change=arousal_change,
                        control_change=control_change
                    )
                except Exception as e:
                    logger.error(f"Error updating mood from action outcome: {e}")
            
            # NEW: Process mode alignment
            mode_alignment = 0.0
            if "mode_context" in action and self.mode_integration:
                try:
                    # Record feedback about interaction success
                    await self.mode_integration.record_mode_feedback(
                        interaction_success=success, 
                        user_feedback=str(outcome_obj.user_feedback) if outcome_obj.user_feedback else None
                    )
                    
                    # Calculate alignment score
                    mode_key = action.get("mode_context", {}).get("mode")
                    mode_alignment = 0.7 if success else 0.3  # Base alignment on success
                    
                    # Update outcome object
                    outcome_obj.mode_alignment = mode_alignment
                except Exception as e:
                    logger.error(f"Error processing mode alignment: {e}")
            
            # NEW: Update outcome with these new details
            outcome_obj.need_impacts = need_impacts
            outcome_obj.mood_impacts = mood_impacts
            
            # Add causal explanation if action came from reasoning
            causal_explanation = None
            if action.get("source") == ActionSource.REASONING and "reasoning_data" in action:
                # Get model if available
                model_id = action["reasoning_data"].get("model_id")
                if model_id and self.reasoning_core:
                    try:
                        # Create explanation based on actual outcome
                        causal_explanation = f"Outcome aligned with causal model prediction: {success}. "
                        causal_explanation += f"Satisfaction: {satisfaction:.2f}, Reward: {reward_value:.2f}."
                    except Exception as e:
                        logger.error(f"Error generating causal explanation for outcome: {e}")
            
            # NEW: Record meta-evaluation if applicable
            meta_evaluation = {}
            if self.meta_core and action.get("source") == ActionSource.META_COGNITIVE:
                try:
                    bottleneck_targeted = action.get("meta_context", {}).get("bottleneck_type")
                    process_type = action.get("meta_context", {}).get("process_type")
                    
                    if bottleneck_targeted and process_type:
                        # Create evaluation data
                        meta_evaluation = {
                            "bottleneck_addressed": bottleneck_targeted,
                            "process_improved": process_type,
                            "improvement_score": 0.7 if success else 0.2,
                            "recommend_further_action": not success
                        }
                        outcome_obj.meta_evaluation = meta_evaluation
                except Exception as e:
                    logger.error(f"Error creating meta-evaluation: {e}")
            
            # Store action memory
            memory = ActionMemory(
                state=state,
                action=action_name,
                action_id=action.get("id", "unknown"),
                parameters=action.get("parameters", {}),
                outcome=outcome_obj.dict(),
                reward=reward_value,
                timestamp=datetime.datetime.now(),
                source=action.get("source", ActionSource.MOTIVATION),
                causal_explanation=causal_explanation,
                # NEW: Enhanced memory fields
                need_satisfaction=need_impacts,
                mood_impact=mood_impacts,
                mode_alignment=mode_alignment,
                sensory_context=action.get("sensory_context"),
                meta_evaluation=meta_evaluation
            )
            
            self.action_memories.append(memory)
            
            # Limit memory size
            if len(self.action_memories) > self.max_memories:
                self.action_memories = self.action_memories[-self.max_memories:]
            
            # Update reward statistics
            self.total_reward += reward_value
            if reward_value > 0:
                self.positive_rewards += 1
            elif reward_value < 0:
                self.negative_rewards += 1
                
            # Update category stats
            category = action.get("source", ActionSource.MOTIVATION)
            if isinstance(category, ActionSource):
                category = category.value
                
            self.reward_by_category[category]["count"] += 1
            self.reward_by_category[category]["total"] += reward_value
            
            # NEW: Update strategy effectiveness if applicable
            if "strategy_id" in action:
                strategy_id = action["strategy_id"]
                if strategy_id in self.action_strategies:
                    strategy = self.action_strategies[strategy_id]
                    # Update effectiveness based on outcome
                    old_effectiveness = strategy.effectiveness
                    # Calculate new effectiveness with stronger weighting for recent outcomes
                    strategy.effectiveness = old_effectiveness * 0.8 + (reward_value + 1) * 0.5 * 0.2
                    strategy.last_used = datetime.datetime.now()
            
            # Update causal models if applicable
            if action.get("source") == ActionSource.REASONING and self.reasoning_core:
                await self._update_causal_models_from_outcome(action, outcome_obj, reward_value)
            
            # Potentially trigger experience replay
            if random.random() < 0.3:  # 30% chance after each outcome
                await self._experience_replay(3)  # Replay 3 random memories
                
            # Decay exploration rate over time (explore less as we learn more)
            self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
            
            # Return summary of updates
            return {
                "action": action_name,
                "success": success,
                "reward_value": reward_value,
                "new_q_value": action_value.value,
                "q_value_change": action_value.value - old_value,
                "new_habit_strength": new_habit,
                "habit_change": new_habit - current_habit,
                "action_success_rate": self.action_success_rates[action_name]["rate"],
                "memories_stored": len(self.action_memories),
                "exploration_rate": self.exploration_rate,
                # NEW: Enhanced return values
                "need_impacts": need_impacts,
                "mood_impacts": mood_impacts,
                "mode_alignment": mode_alignment,
                "meta_evaluation": meta_evaluation
            }
    
    # --- Comprehensive Action Generation Pipeline ---
    
    async def process_action_generation_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete action generation pipeline with all integrated systems
        
        Args:
            context: Current context
            
        Returns:
            Generated action with full metadata
        """
        # Phase 1: Update internal state and context
        await self.update_motivations()
        await self._update_temporal_context(context)
        
        # Extract user info if available
        user_id = self._get_current_user_id_from_context(context)
        relationship_data = await self._get_relationship_data(user_id) if user_id else None
        user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
        
        # NEW: Get enriched context from all integrated systems
        need_states = await self._get_current_need_states() if self.needs_system else {}
        mood_state = await self._get_current_mood_state() if self.mood_manager else None
        interaction_mode = await self._get_current_interaction_mode() if self.mode_integration else None
        sensory_context = await self._get_sensory_context() if self.multimodal_integrator else {}
        bottlenecks, resource_allocation = await self._get_meta_system_state() if self.meta_core else ([], {})
        
        # Phase 2: Create comprehensive action context
        action_context = ActionContext(
            state=context,
            user_id=user_id,
            relationship_data=relationship_data,
            user_mental_state=user_mental_state,
            temporal_context=self.current_temporal_context,
            motivations=self.motivations,
            action_history=[a for a in self.action_history[-10:] if isinstance(a, dict)],
            # Enhanced context
            mood_state=mood_state,
            need_states=need_states,
            interaction_mode=interaction_mode,
            sensory_context=sensory_context,
            bottlenecks=bottlenecks,
            resource_allocation=resource_allocation,
            strategy_parameters=self._get_current_strategy_parameters()
        )
        
        # Phase 3: Generate action candidates from multiple sources
        candidates = []
        
        # 3.1 Check for goal-aligned actions
        if self.goal_system:
            active_goal = await self._check_active_goals(context)
            if active_goal:
                goal_action = await self._generate_goal_aligned_action(active_goal, context)
                if goal_action:
                    goal_action["source"] = ActionSource.GOAL
                    candidates.append(goal_action)
        
        # 3.2 Check for need-driven actions with high drive
        if self.needs_system:
            need_action = await self._generate_need_driven_action(action_context)
            if need_action:
                candidates.append(need_action)
        
        # 3.3 Check for mood-driven actions when mood is intense
        if self.mood_manager and mood_state and mood_state.intensity > 0.7:
            mood_action = await self._generate_mood_driven_action(action_context)
            if mood_action:
                candidates.append(mood_action)
        
        # 3.4 Check for mode-aligned actions when mode is set
        if self.mode_integration and interaction_mode:
            mode_action = await self._generate_mode_aligned_action(action_context)
            if mode_action:
                candidates.append(mode_action)
        
        # 3.5 Check if we need to address system bottlenecks
        if bottlenecks and any(b.get("severity", 0) > 0.7 for b in bottlenecks):
            meta_action = await self._generate_meta_improvement_action(action_context)
            if meta_action:
                candidates.append(meta_action)
        
        # 3.6 Check if it's time for leisure/idle activity
        if await self._should_engage_in_leisure(context):
            leisure_action = await self._generate_leisure_action(context)
            leisure_action["source"] = ActionSource.IDLE
            candidates.append(leisure_action)
        
        # 3.7 Generate motivation-driven candidates
        motivation_candidates = await self._generate_candidate_actions(action_context)
        candidates.extend(motivation_candidates)
        
        # 3.8 Generate reasoning-based candidates
        relevant_causal_models = await self._get_relevant_causal_models(context)
        if self.reasoning_core and relevant_causal_models:
            reasoning_candidates = await self._generate_reasoning_actions(action_context)
            candidates.extend(reasoning_candidates)
        
        # 3.9 Generate conceptual blending candidates
        relevant_concept_spaces = await self._get_relevant_concept_spaces(context)
        if self.reasoning_core and relevant_concept_spaces:
            blending_candidates = await self._generate_conceptual_blend_actions(action_context)
            candidates.extend(blending_candidates)
        
        # 3.10 Generate sensory-driven candidates
        if self.multimodal_integrator and sensory_context:
            sensory_candidates = await self._generate_sensory_driven_actions(action_context)
            candidates.extend(sensory_candidates)
        
        # 3.11 Add temporal reflection action if long idle time
        if self.temporal_perception and self.idle_duration > 1800:  # After 30 min idle
            reflection_action = await self._generate_temporal_reflection_action(context)
            if reflection_action:
                candidates.append(reflection_action)
        
        # Phase 4: Select best action using reinforcement learning and integrated systems
        # Update action context with candidate actions
        action_context.available_actions = [a["name"] for a in candidates if "name" in a]
        selected_action = await self._select_best_action(candidates, action_context)
        
        # Phase 5: Enhance and adapt the selected action
        # 5.1 Add unique ID and timestamp
        if "id" not in selected_action:
            selected_action["id"] = f"action_{uuid.uuid4().hex[:8]}"
        selected_action["timestamp"] = datetime.datetime.now().isoformat()
        
        # 5.2 Apply identity influence
        if self.identity_evolution:
            selected_action = await self._apply_identity_influence(selected_action)
        
        # 5.3 Apply mood influence
        if self.mood_manager and mood_state:
            selected_action = await self._apply_mood_influence(selected_action, mood_state)
        
        # 5.4 Apply mode-specific adaptation
        if self.mode_integration and interaction_mode:
            selected_action = await self._adapt_action_to_mode(selected_action, interaction_mode)
        
        # 5.5 Add causal explanation
        if self.reasoning_core and "source" in selected_action:
            explanation = await self._generate_causal_explanation(selected_action, context)
            if explanation:
                selected_action["causal_explanation"] = explanation
        
        # 5.6 Add context summary
        selected_action["context_summary"] = {
            "user_id": user_id,
            "relationship_state": relationship_data.get("state", "unknown") if relationship_data else "unknown",
            "mood": mood_state.dominant_mood if mood_state else "unknown",
            "highest_need": max(need_states.items(), key=lambda x: x[1].get("drive_strength", 0))[0] if need_states else "unknown",
            "mode": interaction_mode,
            "has_bottlenecks": bool(bottlenecks)
        }
        
        # Phase 6: Record and track
        await self._record_action_as_memory(selected_action)
        self.action_history.append(selected_action)
        self.last_major_action_time = datetime.datetime.now()
        
        # Phase 7: Update meta-cognitive tracking
        self.action_count_since_evaluation += 1
        if self.action_count_since_evaluation >= self.meta_parameters["evaluation_interval"]:
            await self._evaluate_action_strategies()
            self.action_count_since_evaluation = 0
        
        return selected_action
    
    # --- Utils for External Systems ---
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reinforcement learning system"""
        # Calculate success rates
        success_rates = {}
        for action_name, stats in self.action_success_rates.items():
            if stats["attempts"] > 0:
                success_rates[action_name] = {
                    "rate": stats["rate"],
                    "successes": stats["successes"],
                    "attempts": stats["attempts"]
                }
        
        # Calculate average Q-values per state
        avg_q_values = {}
        for state_key, action_values in self.action_values.items():
            if action_values:
                state_avg = sum(av.value for av in action_values.values()) / len(action_values)
                avg_q_values[state_key] = state_avg
        
        # Get top actions by value
        top_actions = []
        for state_key, action_values in self.action_values.items():
            for action_name, action_value in action_values.items():
                if action_value.update_count >= 3:  # Only consider actions with enough data
                    top_actions.append({
                        "state_key": state_key,
                        "action": action_name,
                        "value": action_value.value,
                        "updates": action_value.update_count,
                        "confidence": action_value.confidence
                    })
        
        # Sort by value (descending)
        top_actions.sort(key=lambda x: x["value"], reverse=True)
        top_actions = top_actions[:10]  # Top 10
        
        # Get top habits
        top_habits = []
        for state_key, habits in self.habits.items():
            for action_name, strength in habits.items():
                if strength > 0.3:  # Only consider moderate-to-strong habits
                    top_habits.append({
                        "state_key": state_key,
                        "action": action_name,
                        "strength": strength
                    })
        
        # Sort by strength (descending)
        top_habits.sort(key=lambda x: x["strength"], reverse=True)
        top_habits = top_habits[:10]  # Top 10
        
        # NEW: Get need satisfaction statistics
        need_satisfaction = {}
        if self.needs_system:
            need_states = self.needs_system.get_needs_state()
            need_satisfaction = {
                need_name: {
                    "level": need_data.get("level", 0.0),
                    "drive": need_data.get("drive_strength", 0.0),
                    "deficit": need_data.get("deficit", 0.0)
                }
                for need_name, need_data in need_states.items()
            }
        
        # NEW: Get mood statistics
        mood_stats = {}
        if self.mood_manager and self.last_mood_state:
            mood_stats = {
                "dominant_mood": self.last_mood_state.dominant_mood,
                "valence": self.last_mood_state.valence,
                "arousal": self.last_mood_state.arousal,
                "control": self.last_mood_state.control,
                "intensity": self.last_mood_state.intensity
            }
        
        # NEW: Get interaction mode statistics
        mode_stats = {}
        if self.mode_integration and self.current_mode:
            mode_stats = {
                "current_mode": self.current_mode,
                "adaptation_strength": self.mode_adaptation_strength
            }
        
        # NEW: Get strategy statistics
        strategy_stats = [
            {
                "name": strategy.name,
                "effectiveness": strategy.effectiveness,
                "usage_count": strategy.usage_count,
                "last_used": strategy.last_used.isoformat() if strategy.last_used else None
            }
            for strategy in sorted(self.action_strategies.values(), 
                              key=lambda s: s.effectiveness, reverse=True)
        ]
        
        return {
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            },
            "performance": {
                "total_reward": self.total_reward,
                "positive_rewards": self.positive_rewards,
                "negative_rewards": self.negative_rewards,
                "success_rates": success_rates
            },
            "models": {
                "action_values_count": sum(len(actions) for actions in self.action_values.values()),
                "habits_count": sum(len(habits) for habits in self.habits.values()),
                "memories_count": len(self.action_memories)
            },
            "top_actions": top_actions,
            "top_habits": top_habits,
            "reward_by_category": {k: v for k, v in self.reward_by_category.items() if v["count"] > 0},
            # NEW: Enhanced statistics
            "need_satisfaction": need_satisfaction,
            "mood_stats": mood_stats,
            "mode_stats": mode_stats,
            "strategy_stats": strategy_stats
        }
    
    async def predict_action_outcome(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the outcome of an action using the prediction engine and integrated systems
        
        Args:
            action: The action to predict outcome for
            context: Current context
            
        Returns:
            Predicted outcome with enhanced details
        """
        if not self.prediction_engine:
            # Fallback to simple prediction based on past experience
            return await self._predict_outcome_from_history(action, context)
            
        try:
            # Prepare prediction input
            prediction_input = {
                "action": action["name"],
                "parameters": action.get("parameters", {}),
                "context": context,
                "history": self.action_history[-5:] if len(self.action_history) > 5 else self.action_history,
                "source": action.get("source", ActionSource.MOTIVATION)
            }
            
            # NEW: Add integrated context for better predictions
            if self.needs_system:
                # Add need context
                need_states = await self._get_current_need_states()
                prediction_input["need_states"] = need_states
                
                # Add need context for need-specific actions
                if "need_context" in action:
                    prediction_input["need_context"] = action["need_context"]
            
            if self.mood_manager and self.last_mood_state:
                # Add mood context
                prediction_input["mood_state"] = {
                    "dominant_mood": self.last_mood_state.dominant_mood,
                    "valence": self.last_mood_state.valence,
                    "arousal": self.last_mood_state.arousal,
                    "control": self.last_mood_state.control
                }
            
            if self.mode_integration and self.current_mode:
                # Add mode context
                prediction_input["interaction_mode"] = self.current_mode
                
                # Add mode context for mode-specific actions
                if "mode_context" in action:
                    prediction_input["mode_context"] = action["mode_context"]
            
            # Call the prediction engine
            if hasattr(self.prediction_engine, "predict_action_outcome"):
                prediction = await self.prediction_engine.predict_action_outcome(prediction_input)
                
                # NEW: Add integrated prediction details
                if prediction:
                    # Add need impact predictions if relevant
                    if "need_context" in action and not "need_impacts" in prediction:
                        need_name = action.get("need_context", {}).get("need_name")
                        prediction["need_impacts"] = {
                            need_name: prediction.get("success", False) and 0.2 or 0.05
                        }
                    
                    # Add mood impact predictions if not present
                    if not "mood_impacts" in prediction and self.last_mood_state:
                        reward_value = prediction.get("reward_value", 0.0)
                        prediction["mood_impacts"] = {
                            "valence": reward_value * 0.3,
                            "arousal": abs(reward_value) * 0.1 * (reward_value > 0 and 1 or -1),
                            "control": prediction.get("success", False) and 0.15 or -0.2
                        }
                
                return prediction
            elif hasattr(self.prediction_engine, "generate_prediction"):
                # Generic prediction method
                prediction = await self.prediction_engine.generate_prediction(prediction_input)
                return prediction
                
            # Fallback if no appropriate method
            return await self._predict_outcome_from_history(action, context)
            
        except Exception as e:
            logger.error(f"Error predicting action outcome: {e}")
            return await self._predict_outcome_from_history(action, context)
    
    # --- Registration Methods for External Extensions ---
    
    async def register_action_strategy(self, strategy: ActionStrategy) -> str:
        """Register a new action strategy"""
        strategy_id = f"strategy_{len(self.action_strategies) + 1}"
        self.action_strategies[strategy_id] = strategy
        logger.info(f"Registered action strategy: {strategy.name} (ID: {strategy_id})")
        return strategy_id
    
    async def register_sensory_expectation(self, expectation: ExpectationSignal) -> None:
        """Register a sensory expectation to guide multimodal integration"""
        if self.multimodal_integrator:
            await self.multimodal_integrator.add_expectation(expectation)
            self.sensory_expectations.append(expectation)
            logger.info(f"Registered sensory expectation for {expectation.target_modality}")
        else:
            logger.warning("Cannot register sensory expectation: Multimodal integrator not available")
