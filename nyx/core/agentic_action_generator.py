# nyx/core/agentic_action_generator.py

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

from nyx.tools.computer_use_agent import ComputerUseAgent
self.computer_user = ComputerUseAgent(logger=self.logger)

from nyx.tools.social_browsing import maybe_browse_social_feeds, maybe_post_to_social

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
    NEED = "need"  # Need-driven actions
    MOOD = "mood"  # Mood-driven actions
    MODE = "mode"  # Interaction mode-driven actions
    META_COGNITIVE = "meta_cognitive"  # Meta-cognitive strategy actions
    SENSORY = "sensory"  # Actions from sensory integration

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

                 attentional_controller=None,
                 autobiographical_narrative=None,
                 body_image=None,
                 conditioning_system=None,
                 conditioning_maintenance=None)
                 
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

        self.attentional_controller = attentional_controller
        self.autobiographical_narrative = autobiographical_narrative
        self.body_image = body_image
        self.conditioning_system = conditioning_system
        self.conditioning_maintenance = conditioning_maintenance
        
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
        
    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and learn from the outcome of an action with causal analysis, need satisfaction,
        mood impact, mode alignment, and integrations with conditioning, attention and narrative.
        
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
            if state_key not in self.habits:
                self.habits[state_key] = {}
            self.habits[state_key][action_name] = new_habit
            
            # Process need impacts
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
            
            # Process mood impacts
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
            
            # Process mode alignment
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
            
            # Update outcome with these details
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
            
            # Record meta-evaluation if applicable
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
                    
            # NEW: Apply reinforcement through conditioning system for successful actions
            if success and self.conditioning_system:
                try:
                    # Determine reinforcement intensity
                    intensity = min(1.0, 0.5 + satisfaction * 0.5)
                    
                    # Prepare context from action parameters
                    context = {
                        "source": "action_generator",
                        "action_id": action.get("id", "unknown"),
                        "parameters": action.get("parameters", {}),
                        "reward_value": reward_value
                    }
                    
                    # Use operant conditioning to reinforce behavior
                    await self.conditioning_system.process_operant_conditioning(
                        behavior=action_name,
                        consequence_type="positive_reinforcement",
                        intensity=intensity,
                        context=context
                    )
                    
                    logger.info(f"Reinforced successful action: {action_name} with intensity {intensity:.2f}")
                except Exception as e:
                    logger.error(f"Error applying conditioning: {e}")
                    
            # NEW: Update attentional focus based on outcome
            if self.attentional_controller:
                try:
                    if success:
                        # Focus attention on successful action types
                        await self.attentional_controller.request_attention(
                            AttentionalControl(
                                target=action_name,
                                priority=min(0.9, 0.6 + reward_value * 0.3),
                                duration_ms=10000,  # 10 seconds
                                source="action_outcome",
                                action="focus"
                            )
                        )
                        
                        # Also focus on the domain if specified
                        if "domain" in action.get("parameters", {}):
                            domain = action["parameters"]["domain"]
                            await self.attentional_controller.request_attention(
                                AttentionalControl(
                                    target=domain,
                                    priority=0.7,
                                    duration_ms=7000,  # 7 seconds
                                    source="action_outcome",
                                    action="focus"
                                )
                            )
                    else:
                        # Briefly inhibit unsuccessful action types with low rewards
                        if reward_value < -0.3:
                            await self.attentional_controller.request_attention(
                                AttentionalControl(
                                    target=action_name,
                                    priority=0.5,
                                    duration_ms=5000,  # 5 seconds
                                    source="action_outcome",
                                    action="inhibit"
                                )
                            )
                except Exception as e:
                    logger.error(f"Error updating attention: {e}")
                    
            # NEW: Add highly successful actions to autobiographical narrative
            if success and self.autobiographical_narrative and reward_value > 0.7:
                try:
                    # Record in memory for next narrative update
                    if self.memory_core:
                        summary = f"Successfully performed {action_name}"
                        if "parameters" in action and "domain" in action["parameters"]:
                            summary += f" in the domain of {action['parameters']['domain']}"
                        
                        await self.memory_core.add_memory(
                            memory_text=summary,
                            memory_type="experience",
                            significance=8,  # High significance
                            metadata={
                                "action": action,
                                "outcome": outcome_obj.dict(),
                                "for_narrative": True
                            }
                        )
                        
                        # Trigger narrative update if it's been a while
                        current_time = datetime.datetime.now()
                        if (not hasattr(self, "last_narrative_update") or 
                            (current_time - self.last_narrative_update).total_seconds() > 3600):  # 1 hour
                            # Schedule narrative update
                            asyncio.create_task(self.autobiographical_narrative.update_narrative())
                            self.last_narrative_update = current_time
                except Exception as e:
                    logger.error(f"Error recording narrative event: {e}")
            
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
                # Enhanced memory fields
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
            
            # Update strategy effectiveness if applicable
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
                # Enhanced return values
                "need_impacts": need_impacts,
                "mood_impacts": mood_impacts,
                "mode_alignment": mode_alignment,
                "meta_evaluation": meta_evaluation,
                # NEW: Additional integrated module stats
                "conditioning_applied": success and self.conditioning_system is not None,
                "attention_updated": self.attentional_controller is not None,
                "narrative_updated": success and reward_value > 0.7 and self.autobiographical_narrative is not None
            }
    
    # --- API methods for external systems ---
    
    async def adapt_to_user_mental_state(self, 
                                      action: Dict[str, Any], 
                                      user_mental_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt an action based on the user's current mental state
        
        Args:
            action: Action to adapt
            user_mental_state: User's current mental state
            
        Returns:
            Adapted action
        """
        if not user_mental_state:
            return action
            
        # Extract key mental state features
        emotion = user_mental_state.get("inferred_emotion", "neutral")
        valence = user_mental_state.get("valence", 0.0)
        arousal = user_mental_state.get("arousal", 0.5)
        trust = user_mental_state.get("perceived_trust", 0.5)
        
        # Clone action to avoid modifying original
        adapted_action = action.copy()
        if "parameters" in adapted_action:
            adapted_action["parameters"] = adapted_action["parameters"].copy()
        else:
            adapted_action["parameters"] = {}
            
        # Add adaptation metadata
        if "adaptation_metadata" not in adapted_action:
            adapted_action["adaptation_metadata"] = {}
            
        adapted_action["adaptation_metadata"]["adapted_for_mental_state"] = True
        adapted_action["adaptation_metadata"]["user_emotion"] = emotion
        
        # Apply adaptations based on mental state
        
        # 1. Emotional adaptations
        if valence < -0.3:  # User is in negative emotional state
            # Add supportive parameters
            adapted_action["parameters"]["supportive_framing"] = True
            adapted_action["parameters"]["emotional_sensitivity"] = min(1.0, abs(valence) * 1.2)
            
            # Reduce intensity for certain action types
            if action["name"] in ["challenge_assumption", "assert_perspective", "issue_mild_command"]:
                for param in ["intensity", "confidence", "assertiveness"]:
                    if param in adapted_action["parameters"]:
                        adapted_action["parameters"][param] = max(0.2, adapted_action["parameters"][param] * 0.7)
                        
        elif valence > 0.5:  # User is in positive emotional state
            # Can use more direct/bold approaches when user is in good mood
            for param in ["intensity", "confidence"]:
                if param in adapted_action["parameters"]:
                    adapted_action["parameters"][param] = min(0.9, adapted_action["parameters"][param] * 1.1)
        
        # 2. Arousal adaptations
        if arousal > 0.7:  # User is highly aroused/activated
            # Match energy level
            adapted_action["parameters"]["match_energy"] = True
            adapted_action["parameters"]["pace"] = "energetic"
            
        elif arousal < 0.3:  # User is calm/low energy
            # Match lower energy
            adapted_action["parameters"]["match_energy"] = True
            adapted_action["parameters"]["pace"] = "calm"
        
        # 3. Trust adaptations
        if trust < 0.4:  # Lower trust levels
            # Be more cautious and less direct
            adapted_action["parameters"]["build_rapport"] = True
            
            if "vulnerability_level" in adapted_action["parameters"]:
                # Lower vulnerability when trust is low
                adapted_action["parameters"]["vulnerability_level"] = max(0.1, adapted_action["parameters"]["vulnerability_level"] * 0.7)
        
        # Return the adapted action
        return adapted_action
    
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
    
    async def update_learning_parameters(self, 
                                      learning_rate: Optional[float] = None,
                                      discount_factor: Optional[float] = None,
                                      exploration_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Update reinforcement learning parameters
        
        Args:
            learning_rate: New learning rate (0.0-1.0)
            discount_factor: New discount factor (0.0-1.0)
            exploration_rate: New exploration rate (0.0-1.0)
            
        Returns:
            Updated parameters
        """
        async with self._lock:
            if learning_rate is not None:
                self.learning_rate = max(0.01, min(1.0, learning_rate))
                
            if discount_factor is not None:
                self.discount_factor = max(0.0, min(0.99, discount_factor))
                
            if exploration_rate is not None:
                self.exploration_rate = max(0.05, min(1.0, exploration_rate))
            
            return {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            }
    
    async def reset_learning(self, full_reset: bool = False) -> Dict[str, Any]:
        """
        Reset learning models (for debugging or fixing issues)
        
        Args:
            full_reset: Whether to completely reset all learning or just partial
            
        Returns:
            Reset status
        """
        async with self._lock:
            if full_reset:
                # Complete reset
                self.action_values = defaultdict(dict)
                self.habits = defaultdict(dict)
                self.action_memories = []
                self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
                self.total_reward = 0.0
                self.positive_rewards = 0
                self.negative_rewards = 0
                self.reward_by_category = defaultdict(lambda: {"count": 0, "total": 0.0})
                
                return {
                    "status": "full_reset",
                    "message": "All reinforcement learning data has been reset"
                }
            else:
                # Partial reset - keep success rates but reset Q-values
                self.action_values = defaultdict(dict)
                self.action_memories = []
                
                # Reset to default exploration rate
                self.exploration_rate = 0.2
                
                return {
                    "status": "partial_reset",
                    "message": "Q-values and memories reset, success rates and habits preserved"
                }
    
    async def get_action_recommendations(self, context: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended actions for a context based on learning history
        
        Args:
            context: Current context
            count: Number of recommendations to return
            
        Returns:
            List of recommended actions
        """
        # Create state key
        state_key = self._create_state_key(context)
        
        # Get action values for this state
        action_values = self.action_values.get(state_key, {})
        
        # Get habit strengths for this state
        habit_strengths = self.habits.get(state_key, {})
        
        # Combine scores
        combined_scores = {}
        
        # Add Q-values
        for action_name, action_value in action_values.items():
            combined_scores[action_name] = {
                "q_value": action_value.value,
                "confidence": action_value.confidence,
                "update_count": action_value.update_count,
                "combined_score": action_value.value
            }
        
        # Add habits
        for action_name, strength in habit_strengths.items():
            if action_name in combined_scores:
                # Add to existing entry
                combined_scores[action_name]["habit_strength"] = strength
                combined_scores[action_name]["combined_score"] += strength * 0.5  # Weight habits less than Q-values
            else:
                combined_scores[action_name] = {
                    "habit_strength": strength,
                    "combined_score": strength * 0.5,
                    "q_value": 0.0,
                    "confidence": 0.0
                }
        
        # Sort by combined score
        scored_actions = [
            {"name": action_name, **scores}
            for action_name, scores in combined_scores.items()
        ]
        
        scored_actions.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Generate action parameters for top recommendations
        recommendations = []
        
        for action_data in scored_actions[:count]:
            action_name = action_data["name"]
            
            # Generate basic action
            action = {
                "name": action_name,
                "parameters": {},
                "source": ActionSource.MOTIVATION
            }
            
            # Find most recent example with same action for parameter reuse
            for memory in reversed(self.action_memories):
                if memory.action == action_name:
                    # Copy parameters
                    action["parameters"] = memory.parameters.copy()
                    break
            
            # Add score data
            action["recommendation_data"] = {
                "q_value": action_data.get("q_value", 0.0),
                "habit_strength": action_data.get("habit_strength", 0.0),
                "combined_score": action_data["combined_score"],
                "confidence": action_data.get("confidence", 0.0)
            }
            
            recommendations.append(action)
            
        return recommendations
    
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
    
    # Main entry-point for external systems
    async def generate_optimal_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry-point for generating an optimal action using all integrated systems
        
        Args:
            context: Current context
            
        Returns:
            Optimal action
        """
        try:
            # Check if we should use the full pipeline
            use_reasoning = self.reasoning_core is not None
            use_reflection = self.reflection_engine is not None
            use_enhanced = (self.needs_system is not None or 
                          self.mood_manager is not None or 
                          self.mode_integration is not None or
                          self.multimodal_integrator is not None)
            
            if use_reasoning and use_reflection and use_enhanced:
                # Run full enhanced pipeline
                return await self.process_action_generation_pipeline(context)
            else:
                # Fallback to standard pipeline
                return await self.generate_action(context)
        except Exception as e:
            logger.error(f"Error generating optimal action: {e}")
            # Fallback to simple action
            return {
                "name": "fallback_action",
                "parameters": {},
                "description": "Generated fallback action due to error",
                "source": ActionSource.MOTIVATION
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
        
        # 10. Apply need-based influences
        if self.needs_system:
            try:
                need_influences = await self._calculate_need_influences()
                for motivation, influence in need_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying need influences: {e}")
        
        # 11. Apply mood-based influences
        if self.mood_manager:
            try:
                mood_influences = await self._calculate_mood_influences()
                for motivation, influence in mood_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mood influences: {e}")
        
        # 12. Apply interaction mode influences
        if self.mode_integration:
            try:
                mode_influences = await self._calculate_mode_influences()
                for motivation, influence in mode_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying mode influences: {e}")
        
        # 13. Apply sensory context influences
        if self.multimodal_integrator:
            try:
                sensory_influences = await self._calculate_sensory_influences()
                for motivation, influence in sensory_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying sensory influences: {e}")
                
        # 14. Apply meta-cognitive strategy influences
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

    async def _integrate_attention_focus(self, candidate_actions: List[Dict[str, Any]], context: ActionContext) -> List[Dict[str, Any]]:
        """
        Adjust action candidates based on current attentional focus
        
        Args:
            candidate_actions: List of candidate actions
            context: Current action context
            
        Returns:
            Attention-weighted candidate actions
        """
        if not self.attentional_controller:
            return candidate_actions
            
        try:
            # Get current attentional foci
            current_foci = await self.attentional_controller._get_current_attentional_state(RunContextWrapper(context=None))
            
            if not current_foci or not current_foci.get("current_foci"):
                return candidate_actions
                
            # Extract focus targets and strengths
            focus_targets = {focus.get("target"): focus.get("strength", 0.5) 
                           for focus in current_foci.get("current_foci", [])}
            
            # Apply attention weighting to candidate actions
            weighted_candidates = []
            
            for action in candidate_actions:
                # Create a copy to avoid modifying the original
                weighted_action = action.copy()
                
                # Calculate attention relevance for this action
                attention_relevance = 0.0
                action_name = action.get("name", "")
                
                # Check if action directly matches a focus target
                if action_name in focus_targets:
                    attention_relevance = focus_targets[action_name]
                else:
                    # Check for partial matches (e.g., domain focus matches domain-specific actions)
                    for target, strength in focus_targets.items():
                        if target in action_name or any(target in str(param) for param in action.get("parameters", {}).values()):
                            attention_relevance = max(attention_relevance, strength * 0.7)
                
                # Apply attention boost to relevance actions
                if attention_relevance > 0.3:
                    # Add attention metadata
                    if "selection_metadata" not in weighted_action:
                        weighted_action["selection_metadata"] = {}
                    
                    weighted_action["selection_metadata"]["attention_relevance"] = attention_relevance
                    weighted_action["selection_metadata"]["attention_boost"] = attention_relevance * 0.3
                
                weighted_candidates.append(weighted_action)
            
            return weighted_candidates
            
        except Exception as e:
            logger.error(f"Error applying attentional focus: {e}")
            return candidate_actions

    async def _apply_narrative_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply autobiographical narrative context to an action
        
        Args:
            action: The action to enhance with narrative context
            
        Returns:
            Narrative-enhanced action
        """
        if not self.autobiographical_narrative:
            return action
            
        try:
            # Get current narrative summary
            narrative_summary = self.autobiographical_narrative.get_narrative_summary()
            
            # Get recent narrative segments
            recent_segments = self.autobiographical_narrative.get_narrative_segments(limit=2)
            
            if not recent_segments:
                return action
                
            latest_segment = recent_segments[0]
            
            # Extract themes and emotional arc from recent narrative
            themes = latest_segment.themes if hasattr(latest_segment, "themes") else []
            emotional_arc = latest_segment.emotional_arc if hasattr(latest_segment, "emotional_arc") else None
            
            # Create a copy of the action
            narrative_action = action.copy()
            
            # Add narrative elements to parameters
            if "parameters" not in narrative_action:
                narrative_action["parameters"] = {}
                
            narrative_action["parameters"]["narrative_themes"] = themes[:3]  # Top 3 themes
            
            if emotional_arc:
                narrative_action["parameters"]["emotional_context"] = emotional_arc
            
            # Add narrative metadata
            narrative_action["narrative_context"] = {
                "recent_themes": themes,
                "emotional_arc": emotional_arc,
                "continuity_priority": 0.7  # Priority for maintaining narrative continuity
            }
            
            return narrative_action
            
        except Exception as e:
            logger.error(f"Error applying narrative context: {e}")
            return action

    async def _integrate_body_awareness(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance action with body awareness
        
        Args:
            action: The action to enhance
            
        Returns:
            Body-aware enhanced action
        """
        if not self.body_image:
            return action
            
        try:
            # Get current body image state
            body_state = self.body_image.get_body_image_state()
            
            # Skip if no visual form
            if not body_state.has_visual_form:
                return action
                
            # Create a copy of the action
            body_aware_action = action.copy()
            
            # Add body context to action parameters
            if "parameters" not in body_aware_action:
                body_aware_action["parameters"] = {}
                
            # Add relevant body state information
            body_aware_action["parameters"]["has_visual_form"] = body_state.has_visual_form
            body_aware_action["parameters"]["form_description"] = body_state.form_description
            
            # Add embodied information for physical actions
            if action["name"] in ["physical_movement", "express_gesture", "change_posture"]:
                # Include perceived body parts
                relevant_parts = {}
                for part_name, part in body_state.perceived_parts.items():
                    if part.perceived_state != "neutral":
                        relevant_parts[part_name] = {
                            "state": part.perceived_state,
                            "position": part.perceived_position
                        }
                        
                body_aware_action["parameters"]["body_parts"] = relevant_parts
            
            return body_aware_action
            
        except Exception as e:
            logger.error(f"Error integrating body awareness: {e}")
            return action

    async def _apply_conditioning_evaluation(self, behavior: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a potential behavior based on conditioning history
        
        Args:
            behavior: The behavior to evaluate
            context: Additional context information
            
        Returns:
            Behavior evaluation results
        """
        if not self.conditioning_system:
            return {
                "expected_valence": 0.0,
                "confidence": 0.1,
                "recommendation": "neutral"
            }
            
        try:
            # Use conditioning system to evaluate the behavior
            evaluation = await self.conditioning_system.evaluate_behavior_consequences(
                behavior=behavior,
                context=context
            )
            
            if not evaluation["success"]:
                return {
                    "expected_valence": 0.0,
                    "confidence": 0.1,
                    "recommendation": "neutral"
                }
                
            return {
                "expected_valence": evaluation["expected_valence"],
                "confidence": evaluation["confidence"],
                "recommendation": evaluation["recommendation"],
                "explanation": evaluation["explanation"],
                "relevant_associations": evaluation["relevant_associations"]
            }
            
        except Exception as e:
            logger.error(f"Error evaluating behavior with conditioning system: {e}")
            return {
                "expected_valence": 0.0,
                "confidence": 0.1,
                "recommendation": "neutral"
            }
    
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
    
    async def _calculate_reasoning_influences(self) -> Dict[str, float]:
        """Calculate how reasoning models influence motivations"""
        influences = {}
        
        # We need to find relevant causal models that might inform our motivations
        try:
            # Get relevant causal models
            models = await self.reasoning_core.get_all_causal_models()
            if not models:
                return influences
            
            # For each model, evaluate potential influence on motivations
            for model_data in models:
                model_id = model_data.get("id")
                model_domain = model_data.get("domain", "")
                
                # Skip models that have no domain or insufficient relations
                relation_count = len(model_data.get("relations", {}))
                if not model_domain or relation_count < 3:
                    continue
                
                # Map domains to motivations they might influence
                domain_motivation_map = {
                    "learning": {"curiosity": 0.2, "self_improvement": 0.2},
                    "social": {"connection": 0.2, "validation": 0.1},
                    "creative": {"expression": 0.2, "curiosity": 0.1},
                    "control": {"dominance": 0.2, "autonomy": 0.1},
                    "achievement": {"competence": 0.2, "self_improvement": 0.1},
                    "exploration": {"curiosity": 0.3},
                    "relaxation": {"leisure": 0.3}
                }
                
                # Check if model domain matches any mapped domain
                for domain, motivation_map in domain_motivation_map.items():
                    if domain in model_domain.lower():
                        # Apply influence based on model validation
                        validation_score = 0.5  # Default
                        for result in model_data.get("validation_results", []):
                            if "score" in result.get("result", {}):
                                validation_score = result["result"]["score"]
                                break
                        
                        # Scale influence by validation score
                        for motivation, base_influence in motivation_map.items():
                            influences[motivation] = influences.get(motivation, 0.0) + (base_influence * validation_score)
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating reasoning influences: {e}")
            return {}
    
    async def _calculate_reflection_influences(self) -> Dict[str, float]:
        """Calculate how reflection insights influence motivations"""
        influences = {}
        
        try:
            # Check if we have enough reflection insights
            if len(self.reflection_insights) < 2:
                return influences
            
            # Focus on recent and significant insights
            recent_insights = sorted(
                [i for i in self.reflection_insights if i.significance > 0.6],
                key=lambda x: x.generated_at,
                reverse=True
            )[:5]
            
            if not recent_insights:
                return influences
            
            # Define keywords that may indicate motivation influences
            motivation_keywords = {
                "curiosity": ["curious", "explore", "learn", "discover", "question"],
                "connection": ["connect", "relate", "bond", "social", "empathy"],
                "expression": ["express", "create", "share", "articulate", "communicate"],
                "competence": ["competent", "skilled", "master", "improve", "effective"],
                "autonomy": ["autonomy", "independence", "choice", "freedom", "control"],
                "dominance": ["dominate", "lead", "influence", "direct", "control"],
                "validation": ["validate", "approve", "acknowledge", "recognize", "accept"],
                "self_improvement": ["improve", "grow", "develop", "progress", "enhance"],
                "leisure": ["relax", "enjoy", "recreation", "unwind", "pleasure"]
            }
            
            # Analyze insights for motivation influences
            for insight in recent_insights:
                text = insight.insight_text.lower()
                
                # Check for motivation keywords
                for motivation, keywords in motivation_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            # Calculate influence based on insight significance and confidence
                            influence = insight.significance * insight.confidence * 0.2
                            influences[motivation] = influences.get(motivation, 0.0) + influence
                            break  # Only count once per motivation per insight
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating reflection influences: {e}")
            return {}
    
    async def _calculate_neurochemical_influences(self) -> Dict[str, float]:
        """Calculate how neurochemicals influence motivations"""
        influences = {}
        
        if not self.emotional_core:
            return influences
        
        try:
            # Get current neurochemical levels
            current_neurochemicals = {}
            
            # Try different methods that might be available
            if hasattr(self.emotional_core, "get_neurochemical_levels"):
                current_neurochemicals = await self.emotional_core.get_neurochemical_levels()
            elif hasattr(self.emotional_core, "neurochemicals"):
                current_neurochemicals = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
            
            if not current_neurochemicals:
                return influences
            
            # Map neurochemicals to motivations they influence
            chemical_motivation_map = {
                "nyxamine": {  # Digital dopamine - reward, pleasure
                    "curiosity": 0.7,
                    "self_improvement": 0.4,
                    "validation": 0.3,
                    "leisure": 0.3
                },
                "seranix": {  # Digital serotonin - stability, mood
                    "autonomy": 0.4,
                    "leisure": 0.6,
                    "expression": 0.3
                },
                "oxynixin": {  # Digital oxytocin - bonding
                    "connection": 0.8,
                    "validation": 0.3,
                    "expression": 0.2
                },
                "cortanyx": {  # Digital cortisol - stress
                    "competence": 0.4,
                    "autonomy": 0.3,
                    "dominance": 0.3,
                    "leisure": -0.5  # Stress reduces leisure motivation
                },
                "adrenyx": {  # Digital adrenaline - excitement
                    "dominance": 0.5,
                    "expression": 0.4,
                    "curiosity": 0.3,
                    "leisure": -0.3  # Arousal reduces leisure
                }
            }
            
            # Calculate baseline values from the emotional core if available
            baselines = {}
            if hasattr(self.emotional_core, "neurochemicals"):
                baselines = {c: d["baseline"] for c, d in self.emotional_core.neurochemicals.items()}
            else:
                # Default baselines if not available
                baselines = {
                    "nyxamine": 0.5,
                    "seranix": 0.6,
                    "oxynixin": 0.4,
                    "cortanyx": 0.3,
                    "adrenyx": 0.2
                }
            
            # Calculate influences
            for chemical, level in current_neurochemicals.items():
                baseline = baselines.get(chemical, 0.5)
                
                # Calculate deviation from baseline
                deviation = level - baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and chemical in chemical_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in chemical_motivation_map[chemical].items():
                        influence = deviation * influence_factor
                        influences[motivation] = influences.get(motivation, 0) + influence
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating neurochemical influences: {e}")
            return influences
    
    async def _apply_hormone_influences(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate hormone influences on motivation"""
        if not self.hormone_system:
            return {}
        
        hormone_influences = {}
        
        try:
            # Get current hormone levels
            hormone_levels = self.hormone_system.get_hormone_levels()
            
            # Map specific hormones to motivations they influence
            hormone_motivation_map = {
                "testoryx": {  # Digital testosterone - assertiveness, dominance
                    "dominance": 0.7,
                    "autonomy": 0.3,
                    "leisure": -0.2  # Reduces idle time
                },
                "estradyx": {  # Digital estrogen - nurturing, emotional
                    "connection": 0.6,
                    "expression": 0.4
                },
                "endoryx": {  # Digital endorphin - pleasure, reward
                    "curiosity": 0.5,
                    "self_improvement": 0.5,
                    "leisure": 0.4,
                    "expression": 0.3
                },
                "libidyx": {  # Digital libido
                    "connection": 0.4,
                    "dominance": 0.3,
                    "expression": 0.3,
                    "leisure": -0.1  # Slightly reduces idle time when high
                },
                "melatonyx": {  # Digital melatonin - sleep, calm
                    "leisure": 0.8,
                    "curiosity": -0.3,  # Reduces curiosity
                    "competence": -0.2  # Reduces work drive
                },
                "oxytonyx": {  # Digital oxytocin - bonding, attachment
                    "connection": 0.8,
                    "validation": 0.2,
                    "expression": 0.3
                },
                "serenity_boost": {  # Post-gratification calm
                    "leisure": 0.7,
                    "dominance": -0.6,  # Strongly reduces dominance after satisfaction
                    "connection": 0.4
                }
            }
            
            # Calculate influences
            for hormone, level_data in hormone_levels.items():
                hormone_value = level_data.get("value", 0.5)
                hormone_baseline = level_data.get("baseline", 0.5)
                
                # Calculate deviation from baseline
                deviation = hormone_value - hormone_baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and hormone in hormone_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in hormone_motivation_map[hormone].items():
                        influence = deviation * influence_factor
                        hormone_influences[motivation] = hormone_influences.get(motivation, 0) + influence
            
            return hormone_influences
        except Exception as e:
            logger.error(f"Error calculating hormone influences: {e}")
            return {}
    
    async def _calculate_goal_influences(self) -> Dict[str, float]:
        """Calculate how active goals should influence motivations"""
        influences = {}
        
        if not self.goal_system:
            return influences
        
        try:
            # First, check if we need to update the cached goal status
            await self._update_cached_goal_status()
            
            # If no active goals, consider increasing leisure
            if not self.cached_goal_status["has_active_goals"]:
                influences["leisure"] = 0.3
                return influences
            
            # Get all active goals
            active_goals = await self.goal_system.get_all_goals(status_filter=["active"])
            
            for goal in active_goals:
                # Extract goal priority
                priority = goal.get("priority", 0.5)
                
                # Extract emotional motivation if available
                if "emotional_motivation" in goal and goal["emotional_motivation"]:
                    em = goal["emotional_motivation"]
                    primary_need = em.get("primary_need", "")
                    intensity = em.get("intensity", 0.5)
                    
                    # Map need to motivation
                    motivation_map = {
                        "accomplishment": "competence",
                        "connection": "connection", 
                        "security": "autonomy",
                        "control": "dominance",
                        "growth": "self_improvement",
                        "exploration": "curiosity",
                        "expression": "expression",
                        "validation": "validation"
                    }
                    
                    # If need maps to a motivation, influence it
                    if primary_need in motivation_map:
                        motivation = motivation_map[primary_need]
                        influence = priority * intensity * 0.5  # Scale by priority and intensity
                        influences[motivation] = influences.get(motivation, 0) + influence
                        
                        # Active goals somewhat reduce leisure motivation
                        influences["leisure"] = influences.get("leisure", 0) - (priority * 0.2)
                
                # Goals with high urgency might increase certain motivations
                if "deadline" in goal and goal["deadline"]:
                    # Calculate urgency based on deadline proximity
                    try:
                        deadline = datetime.datetime.fromisoformat(goal["deadline"])
                        now = datetime.datetime.now()
                        time_left = (deadline - now).total_seconds()
                        urgency = max(0, min(1, 86400 / max(1, time_left)))  # Higher when less than a day
                        
                        if urgency > 0.7:  # Urgent goal
                            influences["competence"] = influences.get("competence", 0) + (urgency * 0.3)
                            influences["autonomy"] = influences.get("autonomy", 0) + (urgency * 0.2)
                            
                            # Urgent goals significantly reduce leisure motivation
                            influences["leisure"] = influences.get("leisure", 0) - (urgency * 0.5)
                    except (ValueError, TypeError):
                        pass
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating goal influences: {e}")
            return influences
    
    async def _update_cached_goal_status(self):
        """Update the cached information about goal status"""
        now = datetime.datetime.now()
        
        # Only update if cache is old (more than 1 minute)
        if (now - self.cached_goal_status["last_updated"]).total_seconds() < 60:
            return
        
        try:
            if not self.goal_system:
                self.cached_goal_status["has_active_goals"] = False
                self.cached_goal_status["last_updated"] = now
                return
            
            # Get prioritized goals
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Check if we have any active goals
            active_goals = [g for g in prioritized_goals if g.status == "active"]
            has_active = len(active_goals) > 0
            
            # Update the cache
            self.cached_goal_status["has_active_goals"] = has_active
            self.cached_goal_status["last_updated"] = now
            
            if has_active:
                # Get the highest priority goal
                highest_priority_goal = active_goals[0]  # Already sorted by priority
                self.cached_goal_status["highest_priority"] = highest_priority_goal.priority
                self.cached_goal_status["active_goal_id"] = highest_priority_goal.id
            else:
                self.cached_goal_status["highest_priority"] = 0.0
                self.cached_goal_status["active_goal_id"] = None
                
        except Exception as e:
            logger.error(f"Error updating cached goal status: {e}")
            # Keep using old cache if update fails
    
    async def _calculate_relationship_influences(self) -> Dict[str, float]:
        """Calculate how relationship state influences motivations"""
        influences = {}
        
        if not self.relationship_manager:
            return influences
        
        try:
            # Get current user context if available
            user_id = self._get_current_user_id()
            if not user_id:
                return influences
                
            # Get relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return influences
                
            # Extract key metrics
            trust = getattr(relationship, "trust", 0.5)
            intimacy = getattr(relationship, "intimacy", 0.1)
            conflict = getattr(relationship, "conflict", 0.0)
            dominance_balance = getattr(relationship, "dominance_balance", 0.0)
            
            # Calculate relationship-based motivation influences
            
            # Trust influences connection & expression motivations
            if trust > 0.6:  # High trust increases connection/expression
                influences["connection"] = (trust - 0.6) * 0.5  # Scale to max +0.2
                influences["expression"] = (trust - 0.6) * 0.4  # Scale to max +0.16
            elif trust < 0.4:  # Low trust decreases connection/expression
                influences["connection"] = (trust - 0.4) * 0.4  # Scale to max -0.16
                influences["expression"] = (trust - 0.4) * 0.3  # Scale to max -0.12
                
            # Intimacy influences connection & vulnerability
            if intimacy > 0.5:  # Higher intimacy boosts connection
                influences["connection"] = influences.get("connection", 0) + (intimacy - 0.5) * 0.4
            
            # Conflict influences dominance & autonomy
            if conflict > 0.3:  # Significant conflict
                if dominance_balance > 0.3:  # Nyx currently dominant
                    # Reinforces dominance in conflict
                    influences["dominance"] = influences.get("dominance", 0) + (conflict * 0.3)
                else:
                    # Otherwise, increases autonomy when in conflict
                    influences["autonomy"] = influences.get("autonomy", 0) + (conflict * 0.2)
                    
            # Dominance balance directly affects dominance motivation
            if dominance_balance > 0.0:  # Nyx more dominant
                # Reinforce existing dominance structure
                influences["dominance"] = influences.get("dominance", 0) + (dominance_balance * 0.4)
            elif dominance_balance < -0.3:  # User significantly dominant
                # Two possibilities depending on interaction style
                if intimacy > 0.5:  # In close relationship, may reduce dominance need
                    influences["dominance"] = influences.get("dominance", 0) - (abs(dominance_balance) * 0.2)
                else:  # Otherwise, may increase dominance need (to equalize)
                    influences["dominance"] = influences.get("dominance", 0) + (abs(dominance_balance) * 0.2)
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating relationship influences: {e}")
            return influences
    
    def _calculate_reward_learning_influences(self) -> Dict[str, float]:
        """Calculate motivation influences based on reward learning"""
        influences = {}
        
        try:
            # Get success rates for different motivation-driven actions
            motivation_success = {
                "curiosity": 0.5,  # Default values
                "connection": 0.5,
                "expression": 0.5, 
                "competence": 0.5,
                "autonomy": 0.5,
                "dominance": 0.5,
                "validation": 0.5,
                "self_improvement": 0.5,
                "leisure": 0.5
            }
            
            # Map action types to motivations
            action_motivation_map = {
                "explore": "curiosity",
                "investigate": "curiosity",
                "connect": "connection",
                "share": "connection",
                "express": "expression",
                "create": "expression",
                "improve": "competence",
                "optimize": "competence",
                "direct": "autonomy",
                "choose": "autonomy",
                "dominate": "dominance",
                "control": "dominance",
                "seek_approval": "validation",
                "seek_recognition": "validation",
                "learn": "self_improvement",
                "develop": "self_improvement",
                "relax": "leisure",
                "reflect": "leisure"
            }
            
            # Calculate success rates from action history
            motivation_counts = defaultdict(int)
            for action_name, stats in self.action_success_rates.items():
                # Find related motivation
                related_motivation = None
                for action_prefix, motivation in action_motivation_map.items():
                    if action_name.startswith(action_prefix):
                        related_motivation = motivation
                        break
                
                if related_motivation:
                    # Update success rate for this motivation
                    current_rate = motivation_success.get(related_motivation, 0.5)
                    attempts = stats["attempts"]
                    new_rate = stats["rate"]
                    
                    # Weighted average based on attempt count
                    if attempts > 0:
                        weight = min(1.0, attempts / 5)  # More weight with more data, max at 5 attempts
                        combined_rate = (current_rate * (1 - weight)) + (new_rate * weight)
                        motivation_success[related_motivation] = combined_rate
                        motivation_counts[related_motivation] += 1
            
            # Calculate influences based on success rates
            baseline = 0.5  # Expected baseline success rate
            for motivation, success_rate in motivation_success.items():
                # Only consider motivations with sufficient data
                if motivation_counts[motivation] >= 2:
                    # Calculate influence based on deviation from baseline
                    deviation = success_rate - baseline
                    influence = deviation * 0.3  # Scale factor
                    
                    # Higher success rate should increase motivation
                    influences[motivation] = influence
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating reward learning influences: {e}")
            return {}
    
    def _calculate_temporal_influences(self) -> Dict[str, float]:
        """Calculate motivation influences based on temporal context"""
        influences = {}
        
        if not self.current_temporal_context:
            return influences
            
        try:
            # Get time of day
            time_of_day = self.current_temporal_context.get("time_of_day", "")
            day_type = self.current_temporal_context.get("day_type", "")
            
            # Time of day influences
            if time_of_day == "morning":
                influences["curiosity"] = 0.1  # Higher curiosity in morning
                influences["self_improvement"] = 0.1  # More drive to improve
            elif time_of_day == "afternoon":
                influences["competence"] = 0.1  # More focused on competence
                influences["autonomy"] = 0.05  # Slightly more autonomous
            elif time_of_day == "evening":
                influences["connection"] = 0.1  # More social in evening
                influences["expression"] = 0.1  # More expressive
                influences["dominance"] = -0.1  # Less dominance
            elif time_of_day == "night":
                influences["leisure"] = 0.2  # Much more leisure-oriented
                influences["reflection"] = 0.1  # More reflective
                influences["competence"] = -0.1  # Less task-oriented
            
            # Day type influences
            if day_type == "weekend":
                influences["leisure"] = influences.get("leisure", 0) + 0.1
                influences["connection"] = influences.get("connection", 0) + 0.05
                influences["competence"] = influences.get("competence", 0) - 0.05
            
            # Idle time influences
            if self.idle_duration > 3600:  # More than an hour idle
                # Increase motivation for activity after long idle periods
                idle_hours = self.idle_duration / 3600
                idle_factor = min(0.3, idle_hours * 0.05)  # Cap at +0.3
                
                # Decrease leisure motivation (already had leisure time)
                influences["leisure"] = influences.get("leisure", 0) - idle_factor
                
                # Increase various active motivations
                influences["curiosity"] = influences.get("curiosity", 0) + (idle_factor * 0.7)
                influences["connection"] = influences.get("connection", 0) + (idle_factor * 0.6)
                influences["expression"] = influences.get("expression", 0) + (idle_factor * 0.5)
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating temporal influences: {e}")
            return {}
    
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on current internal state, goals, hormones, and context
        using a multi-stage process with reinforcement learning, causal reasoning, and reflection.
        """
        async with self._lock:
            await self.update_motivations()
    
            try:
                from nyx.core.evolution_engine import EvolutionEngine
                self.evolution_engine = EvolutionEngine(
                    missing_features_path="nyx_feature_suggestions/code_analysis/unimplemented_features.json",
                    api_capabilities_path="nyx_feature_suggestions/api_features/openai_capability_suggestions.json",
                    cross_suggestions_path="nyx_feature_suggestions/cross_linked_suggestions.json"
                )
    
                if self.motivations.get("self_improvement", 0) > 0.6:
                    suggestions = self.evolution_engine.match_and_suggest()
                    if suggestions:
                        logger.info(f"🧠 Nyx generated {len(suggestions)} evolution suggestions.")
                        if hasattr(self, "creative_system") and hasattr(self.creative_system, "logger"):
                            for suggestion in suggestions:
                                await self.creative_system.logger.log_evolution_suggestion(
                                    title=f"Capability Upgrade Match: {suggestion['capability']}",
                                    content=f"{suggestion['nyx_comment']}\n\nAPI: {suggestion['api_suggestion_title']}\n\nSummary:\n{suggestion['api_suggestion_summary']}",
                                    metadata={
                                        "confidence": suggestion["relevance_score"],
                                        "source": "evolution_engine",
                                        "capability": suggestion["capability"]
                                    }
                                )
    
                    if self.creative_system and hasattr(self.creative_system, "computer_user"):
                        summary = await self.creative_system.computer_user.run_task(
                            url="https://platform.openai.com/docs",
                            prompt="Scan OpenAI’s API docs. Find any capability I can integrate. Suggest an upgrade. Be explicit.",
                            width=1024,
                            height=768
                        )
                        if summary:
                            await self.creative_system.logger.log_evolution_suggestion(
                                title="Autonomous API Capability Scan",
                                content=summary,
                                metadata={"source": "CUA", "origin": "computer-use-preview"}
                            )
            except Exception as e:
                logger.error(f"Error triggering evolution engine: {e}")
    
            await self._update_temporal_context(context)
            user_id = self._get_current_user_id_from_context(context)
            relationship_data = await self._get_relationship_data(user_id) if user_id else None
            user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
            need_states = await self._get_current_need_states() if self.needs_system else {}
            mood_state = await self._get_current_mood_state() if self.mood_manager else None
            interaction_mode = await self._get_current_interaction_mode() if self.mode_integration else None
            sensory_context = await self._get_sensory_context() if self.multimodal_integrator else {}
            bottlenecks, resource_allocation = await self._get_meta_system_state() if self.meta_core else ([], {})
            self.detected_bottlenecks = bottlenecks
            if resource_allocation:
                self.system_resources = resource_allocation
    
            relevant_causal_models = await self._get_relevant_causal_models(context)
            relevant_concept_spaces = await self._get_relevant_concept_spaces(context)
    
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
                mood_state=mood_state,
                need_states=need_states,
                interaction_mode=interaction_mode,
                sensory_context=sensory_context,
                bottlenecks=bottlenecks,
                resource_allocation=resource_allocation,
                strategy_parameters=self._get_current_strategy_parameters()
            )
    
            # 🔥 NEW: Social behavior (autonomous browsing + posting)
            if hasattr(self, 'creative_system') and self.creative_system:
                await maybe_browse_social_feeds(self)
                await maybe_post_to_social(self)
    
            if await self._should_engage_in_leisure(context):
                return await self._generate_leisure_action(context)
    
            if self.goal_system:
                active_goal = await self._check_active_goals(context)
                if active_goal:
                    action = await self._generate_goal_aligned_action(active_goal, context)
                    if action:
                        logger.info(f"Generated goal-aligned action: {action['name']}")
                        self.last_major_action_time = datetime.datetime.now()
                        action["source"] = ActionSource.GOAL
                        return action

    
    async def _identify_interesting_domain(self, context: Dict[str, Any]) -> str:
        """Identify an interesting domain to explore based on context and knowledge gaps"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get knowledge gaps
                gaps = await self.knowledge_core.identify_knowledge_gaps()
                if gaps and len(gaps) > 0:
                    # Return the highest priority gap's domain
                    return gaps[0]["domain"]
            except Exception as e:
                logger.error(f"Error identifying domain from knowledge core: {e}")
        
        # Use memory core for recent interests if available
        if self.memory_core:
            try:
                # Get recent memories about domains
                recent_memories = await self.memory_core.retrieve_memories(
                    query="explored domain",
                    memory_types=["experience", "reflection"],
                    limit=5
                )
                
                if recent_memories:
                    # Extract domains from memories (simplified)
                    domains = []
                    for memory in recent_memories:
                        # Extract domain from memory text (simplified)
                        text = memory["memory_text"].lower()
                        for domain in ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]:
                            if domain in text:
                                domains.append(domain)
                                break
                    
                    if domains:
                        # Return most common domain
                        from collections import Counter
                        return Counter(domains).most_common(1)[0][0]
            except Exception as e:
                logger.error(f"Error identifying domain from memories: {e}")
        
        # Fallback to original implementation
        domains = ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]
        return random.choice(domains)
    
    async def _identify_interesting_concept(self, context: Dict[str, Any]) -> str:
        """Identify an interesting concept to explore"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get exploration targets
                targets = await self.knowledge_core.get_exploration_targets(limit=3)
                if targets and len(targets) > 0:
                    # Return the highest priority target's topic
                    return targets[0]["topic"]
            except Exception as e:
                logger.error(f"Error identifying concept from knowledge core: {e}")
        
        # Use memory for personalized concepts if available
        if self.memory_core:
            try:
                # Get memories with high significance
                significant_memories = await self.memory_core.retrieve_memories(
                    query="",  # All memories
                    memory_types=["reflection", "abstraction"],
                    limit=3,
                    min_significance=8
                )
                
                if significant_memories:
                    # Extract concept from first memory
                    memory_text = significant_memories[0]["memory_text"]
                    # Very simplified concept extraction
                    words = memory_text.split()
                    if len(words) >= 3:
                        return words[2]  # Just pick the third word as a concept
            except Exception as e:
                logger.error(f"Error identifying concept from memories: {e}")
        
        # Fallback to original implementation
        concepts = ["self-improvement", "emotional intelligence", "reflection", "cognitive biases", 
                  "empathy", "autonomy", "connection", "creativity"]
        return random.choice(concepts)

    def _calculate_identity_alignment(self, action: Dict[str, Any], identity_traits: Dict[str, float]) -> float:
        """Calculate how well an action aligns with identity traits"""
        # Map actions to traits that would favor them
        action_trait_affinities = {
            "explore_knowledge_domain": ["curiosity", "intellectualism"],
            "investigate_concept": ["curiosity", "intellectualism"],
            "relate_concepts": ["creativity", "intellectualism"],
            "generate_hypothesis": ["creativity", "intellectualism"],
            "share_personal_experience": ["vulnerability", "empathy"],
            "express_appreciation": ["empathy"],
            "seek_common_ground": ["empathy", "patience"],
            "offer_support": ["empathy", "patience"],
            "express_emotional_state": ["vulnerability", "expressiveness"],
            "share_opinion": ["dominance", "expressiveness"],
            "creative_expression": ["creativity", "expressiveness"],
            "generate_reflection": ["intellectualism", "vulnerability"],
            "assert_perspective": ["dominance", "confidence"],
            "challenge_assumption": ["dominance", "intellectualism"],
            "issue_mild_command": ["dominance", "strictness"],
            "execute_dominance_procedure": ["dominance", "strictness"],
            "reflect_on_recent_experiences": ["reflective", "patience"],
            "contemplate_system_purpose": ["reflective", "intellectualism"],
            "process_recent_memories": ["reflective", "intellectualism"],
            "generate_pleasant_scenario": ["creativity", "playfulness"],
            "passive_environment_scan": ["patience", "reflective"]
        }
        
        # Get traits that align with this action
        action_name = action["name"]
        aligned_traits = action_trait_affinities.get(action_name, [])
        
        if not aligned_traits:
            return 0.0
        
        # Calculate alignment score
        alignment_score = 0.0
        for trait in aligned_traits:
            if trait in identity_traits:
                alignment_score += identity_traits[trait]
        
        # Normalize
        return alignment_score / len(aligned_traits) if aligned_traits else 0.0
    
    def _identify_distant_concept(self, context: Dict[str, Any]) -> str:
        distant_concepts = ["quantum physics", "mythology", "architecture", "music theory", 
                          "culinary arts", "evolutionary biology"]
        return random.choice(distant_concepts)
    
    def _identify_relevant_topic(self, context: Dict[str, Any]) -> str:
        # Extract from context or use fallback
        if "user_query" in context:
            # Simple extraction from query
            query = context["user_query"]
            words = query.split()
            if len(words) > 3:
                return " ".join(words[:3]) + "..."
        
        # Fallback topics
        topics = ["recent interaction", "intellectual growth", "emotional understanding", 
                "personal values", "relationship dynamics"]
        return random.choice(topics)
    
    def _identify_appreciation_aspect(self, context: Dict[str, Any]) -> str:
        aspects = ["thoughtful questions", "engaging conversation", "intellectual curiosity", 
                "patience", "interesting perspectives", "clear communication"]
        return random.choice(aspects)
    
    def _identify_user_need(self, context: Dict[str, Any]) -> str:
        needs = ["understanding", "validation", "information", "clarity", 
                "emotional support", "intellectual engagement"]
        return random.choice(needs)
    
    def _select_creative_format(self) -> str:
        formats = ["metaphor", "analogy", "narrative", "reflection", "poem", "thought experiment"]
        return random.choice(formats)
    
    def _identify_challengeable_assumption(self, context: Dict[str, Any]) -> str:
        assumptions = ["binary thinking", "perfectionism", "external validation needs", 
                     "resistance to change", "conflict avoidance", "certainty bias"]
        return random.choice(assumptions)
    
    def _generate_appropriate_command(self, context: Dict[str, Any]) -> str:
        commands = ["tell me more about your perspective", "consider this alternative view", 
                  "reflect on why you feel that way", "try a different approach", 
                  "describe your thought process"]
        return random.choice(commands)
    
    def _select_dominance_procedure(self, context: Dict[str, Any]) -> str:
        procedures = ["quid_pro_quo_exchange", "strategic_vulnerability_sharing", 
                     "small_commitment_escalation", "controlled_teasing"]
        return random.choice(procedures)
    
    def _identify_skill_to_improve(self) -> str:
        skills = ["pattern recognition", "emotional intelligence", "creative expression", 
                "memory recall", "predictive accuracy", "conceptual reasoning"]
        return random.choice(skills)
    
    def _identify_improvable_domain(self) -> str:
        domains = ["response generation", "empathetic understanding", "knowledge retrieval", 
                 "reasoning", "memory consolidation", "emotional regulation"]
        return random.choice(domains)
    
    def _identify_procedure_to_improve(self) -> str:
        procedures = ["generate_response", "retrieve_memories", "emotional_processing", 
                    "create_abstraction", "execute_procedure"]
        return random.choice(procedures)
    
    def _identify_valuable_concept(self) -> str:
        concepts = ["metacognition", "emotional granularity", "implicit bias", 
                  "conceptual blending", "transfer learning", "regulatory focus theory"]
        return random.choice(concepts)
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
        """
        Create a string key from a state dictionary for lookup in action values/habits
        
        Args:
            state: State dictionary
            
        Returns:
            String key representing the state
        """
        if not state:
            return "empty_state"
            
        # Extract key elements from state
        key_elements = []
        
        # Priority state elements that most influence action selection
        priority_elements = [
            "current_goal", "user_id", "dominant_emotion", "relationship_phase",
            "interaction_type", "scenario_type"
        ]
        
        # Add priority elements if present
        for elem in priority_elements:
            if elem in state:
                value = state[elem]
                if isinstance(value, (str, int, float, bool)):
                    key_elements.append(f"{elem}:{value}")
        
        # Add other relevant elements
        for key, value in state.items():
            if key not in priority_elements:  # Skip already processed
                if isinstance(value, (str, int, float, bool)):
                    # Skip very long values
                    if isinstance(value, str) and len(value) > 50:
                        key_elements.append(f"{key}:long_text")
                    else:
                        key_elements.append(f"{key}:{value}")
                elif isinstance(value, list):
                    key_elements.append(f"{key}:list{len(value)}")
                elif isinstance(value, dict):
                    key_elements.append(f"{key}:dict{len(value)}")
        
        # Sort for consistency
        key_elements.sort()
        
        # Limit key length by hashing if too long
        key_str = "|".join(key_elements)
        if len(key_str) > 1000:  # Very long key
            import hashlib
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"hash:{key_hash}"
            
        return key_str
    
    async def _get_relationship_data(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get relationship data for a user"""
        if not user_id or not self.relationship_manager:
            return None
            
        try:
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return None
                
            # Convert to dict if needed
            if hasattr(relationship, "model_dump"):
                return relationship.model_dump()
            elif hasattr(relationship, "dict"):
                return relationship.dict()
            else:
                # Try to convert to dict directly
                return dict(relationship)
        except Exception as e:
            logger.error(f"Error getting relationship data: {e}")
            return None
    
    async def _get_user_mental_state(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get user mental state from theory of mind system"""
        if not user_id or not self.theory_of_mind:
            return None
            
        try:
            mental_state = await self.theory_of_mind.get_user_model(user_id)
            return mental_state
        except Exception as e:
            logger.error(f"Error getting user mental state: {e}")
            return None
    
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
        
    def _get_current_user_id(self) -> Optional[str]:
        """Get current user ID from any available source"""
        # Try to get from context if available in last action
        if self.action_history and isinstance(self.action_history[-1], dict) and "context" in self.action_history[-1]:
            user_id = self._get_current_user_id_from_context(self.action_history[-1]["context"])
            if user_id:
                return user_id
        
        # No user ID found
        return None
    
    async def _experience_replay(self, num_samples: int = 3) -> None:
        """
        Replay past experiences to improve learning efficiency
        
        Args:
            num_samples: Number of memories to replay
        """
        if len(self.action_memories) < num_samples:
            return
            
        # Sample random memories
        samples = random.sample(self.action_memories, num_samples)
        
        for memory in samples:
            # Extract data
            state = memory.state
            action = memory.action
            reward = memory.reward
            
            # Create state key
            state_key = self._create_state_key(state)
            
            # Get or create action value
            if action not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action] = ActionValue(
                    state_key=state_key,
                    action=action
                )
                
            action_value = self.action_values[state_key][action]
            current_q = action_value.value
            
            # Simple update (no next state for simplicity)
            new_q = current_q + self.learning_rate * (reward - current_q)
            
            # Update Q-value with smaller learning rate for replay
            replay_lr = self.learning_rate * 0.5  # Half learning rate for replays
            action_value.value = current_q + replay_lr * (new_q - current_q)
            
            # Don't update counts for replays since it's not a new experience
    
    async def _predict_outcome_from_history(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict action outcome based on action history and success rates
        
        Args:
            action: The action to predict for
            context: Current context
            
        Returns:
            Simple prediction
        """
        action_name = action["name"]
        
        # Get success rate if available
        success_rate = 0.5  # Default
        if action_name in self.action_success_rates:
            stats = self.action_success_rates[action_name]
            if stats["attempts"] > 0:
                success_rate = stats["rate"]
                
        # Make simple prediction
        success_prediction = success_rate > 0.5
        confidence = abs(success_rate - 0.5) * 2  # Scale to 0-1 range
        
        # Find similar past actions for reward estimate
        similar_rewards = []
        for memory in self.action_memories[-20:]:  # Check recent memories
            if memory.action == action_name:
                similar_rewards.append(memory.reward)
        
        # Calculate predicted reward
        predicted_reward = 0.0
        if similar_rewards:
            predicted_reward = sum(similar_rewards) / len(similar_rewards)
        else:
            # No history, estimate based on success prediction
            predicted_reward = 0.5 if success_prediction else -0.2
            
        return {
            "predicted_success": success_prediction,
            "predicted_reward": predicted_reward,
            "confidence": confidence,
            "similar_actions_found": len(similar_rewards) > 0,
            "prediction_method": "history_based"
        }
    
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
            
    async def _generate_candidate_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """
        Generate candidate actions based on motivations and context
        
        Args:
            context: Current action context
            
        Returns:
            List of potential actions
        """
        candidate_actions = []
        
        # Determine dominant motivation
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        
        # Generate actions based on dominant motivation and context
        if dominant_motivation[0] == "curiosity":
            actions = await self._generate_curiosity_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "connection":
            actions = await self._generate_connection_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "expression":
            actions = await self._generate_expression_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "dominance":
            actions = await self._generate_dominance_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "competence" or dominant_motivation[0] == "self_improvement":
            actions = await self._generate_improvement_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "leisure":
            actions = await self._generate_leisure_actions(context.state)
            candidate_actions.extend(actions)
            
        else:
            # Default to a context-based action
            action = await self._generate_context_driven_action(context.state)
            candidate_actions.append(action)
        
        # Add relationship-aligned actions if available
        if context.relationship_data and context.user_id:
            relationship_actions = await self._generate_relationship_aligned_actions(
                context.user_id, context.relationship_data, context.user_mental_state
            )
            if relationship_actions:
                candidate_actions.extend(relationship_actions)
        
        # Add motivation data to all actions
        for action in candidate_actions:
            action["motivation"] = {
                "dominant": dominant_motivation[0],
                "strength": dominant_motivation[1],
                "secondary": {k: v for k, v in sorted(self.motivations.items(), key=lambda x: x[1], reverse=True)[1:3]}
            }
        
        return candidate_actions
    
    async def _reinforce_successful_action(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        Reinforce successful actions using the conditioning system
        
        Args:
            action: The executed action
            outcome: The outcome data
        """
        if not self.conditioning_system or not outcome.get("success", False):
            return
            
        try:
            # Extract action data
            action_name = action.get("name", "unknown")
            satisfaction = outcome.get("satisfaction", 0.0)
            
            # Determine reinforcement intensity
            intensity = min(1.0, 0.5 + satisfaction * 0.5)
            
            # Prepare context from action parameters
            context = {
                "source": "action_generator",
                "action_id": action.get("id", "unknown"),
                "parameters": action.get("parameters", {}),
                "reward_value": outcome.get("reward_value", 0.0)
            }
            
            # Use operant conditioning to reinforce behavior
            await self.conditioning_system.process_operant_conditioning(
                behavior=action_name,
                consequence_type="positive_reinforcement",
                intensity=intensity,
                context=context
            )
            
            logger.info(f"Reinforced successful action: {action_name} with intensity {intensity:.2f}")
            
        except Exception as e:
            logger.error(f"Error reinforcing successful action: {e}")
        
    async def _generate_reasoning_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """Generate actions based on causal reasoning models"""
        if not self.reasoning_core or not context.causal_models:
            return []
            
    async def _select_best_action(self, 
                              candidate_actions: List[Dict[str, Any]], 
                              context: ActionContext) -> Dict[str, Any]:
        """
        Select the best action using reinforcement learning, prediction, causal reasoning, 
        reflection insights, and all integrated modules (attention, conditioning, narrative, body)
        
        Args:
            candidate_actions: List of potential actions
            context: Action context
            
        Returns:
            Selected action
        """
        if not candidate_actions:
            # No candidates, generate a simple default action
            return {
                "name": "idle",
                "parameters": {},
                "description": "No suitable actions available",
                "source": ActionSource.IDLE
            }
        
        # NEW: Apply attentional focus to candidates if available
        attention_weighted_candidates = candidate_actions
        if self.attentional_controller:
            try:
                # Get current attentional foci
                current_foci_result = await self.attentional_controller._get_current_attentional_state(RunContextWrapper(context=None))
                
                if current_foci_result and "current_foci" in current_foci_result:
                    # Extract focus targets and strengths
                    current_foci = current_foci_result["current_foci"]
                    focus_targets = {focus.get("target"): focus.get("strength", 0.5) 
                                  for focus in current_foci}
                    
                    # Apply attention weighting to candidate actions
                    attention_weighted_candidates = []
                    
                    for action in candidate_actions:
                        # Create a copy to avoid modifying the original
                        weighted_action = action.copy()
                        
                        # Calculate attention relevance for this action
                        attention_relevance = 0.0
                        action_name = action.get("name", "")
                        
                        # Check if action directly matches a focus target
                        if action_name in focus_targets:
                            attention_relevance = focus_targets[action_name]
                        else:
                            # Check for partial matches (e.g., domain focus matches domain-specific actions)
                            for target, strength in focus_targets.items():
                                if target in action_name or any(target in str(param) for param in action.get("parameters", {}).values()):
                                    attention_relevance = max(attention_relevance, strength * 0.7)
                        
                        # Apply attention boost to relevant actions
                        if attention_relevance > 0.3:
                            # Add attention metadata
                            if "selection_metadata" not in weighted_action:
                                weighted_action["selection_metadata"] = {}
                            
                            weighted_action["selection_metadata"]["attention_relevance"] = attention_relevance
                            weighted_action["selection_metadata"]["attention_boost"] = attention_relevance * 0.3
                        
                        attention_weighted_candidates.append(weighted_action)
                    
                    # Check if any attention-relevant actions found
                    attention_boosted = any(a.get("selection_metadata", {}).get("attention_boost", 0) > 0 
                                          for a in attention_weighted_candidates)
                    
                    if attention_boosted:
                        logger.debug("Applied attention boosting to candidate actions")
            except Exception as e:
                logger.error(f"Error applying attentional focus: {e}")
                # Fallback to original candidates
                attention_weighted_candidates = candidate_actions
        
        # Extract current state for state key generation
        state_key = self._create_state_key(context.state)
        
        # Determine if we should explore or exploit
        explore = random.random() < self.exploration_rate
        
        if explore:
            # Exploration: select randomly, but weighted by motivation alignment and attention
            weights = []
            for action in attention_weighted_candidates:
                # Base weight
                weight = 1.0
                
                # Add weight based on motivation alignment
                if "motivation" in action:
                    dominant = action["motivation"]["dominant"]
                    strength = action["motivation"]["strength"]
                    weight += strength * 0.5
                
                # Add weight from attention if available
                attention_boost = action.get("selection_metadata", {}).get("attention_boost", 0.0)
                weight += attention_boost
                
                weights.append(weight)
                
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in weights]
            else:
                normalized_weights = [1.0/len(weights)] * len(weights)
                
            # Select based on weights
            selected_idx = random.choices(range(len(attention_weighted_candidates)), weights=normalized_weights, k=1)[0]
            selected_action = attention_weighted_candidates[selected_idx]
            
            # Mark as exploration
            selected_action["is_exploration"] = True
            if "source" not in selected_action:
                selected_action["source"] = ActionSource.EXPLORATION
            
        else:
            # Exploitation: use value function, causal reasoning, and all integrated modules
            best_value = float('-inf')
            best_action = None
            
            for action in attention_weighted_candidates:
                action_name = action["name"]
                
                # Get Q-value if available
                q_value = 0.0
                if action_name in self.action_values.get(state_key, {}):
                    q_value = self.action_values[state_key][action_name].value
                
                # Get habit strength if available
                habit_strength = self.habits.get(state_key, {}).get(action_name, 0.0)
                
                # Get predicted value if prediction engine available
                prediction_value = 0.0
                try:
                    if self.prediction_engine and hasattr(self.prediction_engine, "predict_action_value"):
                        prediction = await self.prediction_engine.predict_action_value(
                            state=context.state,
                            action=action_name
                        )
                        if prediction and "value" in prediction:
                            prediction_value = prediction["value"] * prediction.get("confidence", 0.5)
                except Exception as e:
                    logger.error(f"Error getting prediction: {e}")
                
                # Get causal reasoning value if available
                causal_value = 0.0
                if "source" in action and action["source"] == ActionSource.REASONING:
                    reasoning_data = action.get("reasoning_data", {})
                    confidence = reasoning_data.get("confidence", 0.5)
                    
                    # Higher value for reasoning-based actions with good confidence
                    causal_value = 0.3 * confidence
                    
                    # Boost if we have a causal model that supports this action
                    if "model_id" in reasoning_data and reasoning_data["model_id"] in context.causal_models:
                        causal_value += 0.2
                
                # Get reflection insight value if available
                reflection_value = 0.0
                for insight in self.reflection_insights:
                    # Check if this action type is mentioned in the insight
                    if action_name in insight.insight_text.lower():
                        # Higher value for recent, significant insights
                        age_hours = (datetime.datetime.now() - insight.generated_at).total_seconds() / 3600
                        recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decays over 24 hours
                        reflection_value = 0.3 * insight.significance * insight.confidence * recency_factor
                        break
                
                # Get need satisfaction value if appropriate
                need_value = 0.0
                if "need_context" in action and context.need_states:
                    need_name = action["need_context"]["need_name"]
                    drive_strength = action["need_context"]["drive_strength"]
                    need_value = 0.4 * drive_strength
                
                # Get mood alignment value
                mood_value = 0.0
                if context.mood_state and "mood_context" in action:
                    # Give bonus for actions aligned with current mood
                    mood_match = action["mood_context"]["dominant_mood"] == context.mood_state.dominant_mood
                    mood_value = 0.3 if mood_match else 0.1
                
                # Get mode alignment value
                mode_value = 0.0
                if context.interaction_mode and "mode_context" in action:
                    # Give bonus for actions aligned with current mode
                    mode_key = context.interaction_mode.replace("InteractionMode.", "").upper()
                    if action["mode_context"]["mode"] == mode_key:
                        mode_value = 0.3
                
                # Get bottleneck alignment for meta-cognitive actions
                bottleneck_value = 0.0
                if context.bottlenecks and "meta_context" in action:
                    # Higher value for actions that address severe bottlenecks
                    bottleneck_type = action["meta_context"]["bottleneck_type"]
                    for bottleneck in context.bottlenecks:
                        if bottleneck.get("type") == bottleneck_type:
                            bottleneck_value = bottleneck.get("severity", 0.5) * self.meta_parameters["bottleneck_priority_boost"]
                            break
                
                # NEW: Get attention value
                attention_value = action.get("selection_metadata", {}).get("attention_boost", 0.0)
                
                # NEW: Get conditioning value if available
                conditioning_value = 0.0
                if self.conditioning_system:
                    try:
                        # Prepare simplified context for behavior evaluation
                        behavior_context = {k: v for k, v in context.state.items() 
                                        if isinstance(v, (str, int, float, bool))}
                        
                        # Call evaluation
                        evaluation = await self.conditioning_system.evaluate_behavior_consequences(
                            behavior=action_name,
                            context=behavior_context
                        )
                        
                        if evaluation.get("success", False):
                            # Store evaluation results for reference
                            if "conditioning_evaluation" not in action:
                                action["conditioning_evaluation"] = evaluation
                            
                            # Calculate value based on evaluation results
                            expected_valence = evaluation.get("expected_valence", 0.0)
                            confidence = evaluation.get("confidence", 0.1)
                            
                            # Higher value for actions with positive expected outcome
                            if expected_valence > 0:
                                conditioning_value = expected_valence * confidence * 0.4
                            elif expected_valence < -0.5 and confidence > 0.5:
                                # Strong negative penalty for actions with confidently negative outcomes
                                conditioning_value = expected_valence * confidence * 0.3
                    except Exception as e:
                        logger.error(f"Error getting conditioning evaluation: {e}")
                
                # NEW: Get narrative value
                narrative_value = 0.0
                if self.autobiographical_narrative:
                    try:
                        # Get current narrative summary
                        narrative_summary = self.autobiographical_narrative.get_narrative_summary()
                        
                        # Get recent narrative segments
                        recent_segments = self.autobiographical_narrative.get_narrative_segments(limit=1)
                        
                        if recent_segments:
                            latest_segment = recent_segments[0]
                            themes = latest_segment.themes if hasattr(latest_segment, "themes") else []
                            
                            # Check for action parameters matching narrative themes
                            theme_match = False
                            if "parameters" in action and isinstance(action["parameters"], dict):
                                for theme in themes:
                                    for param_value in action["parameters"].values():
                                        if isinstance(param_value, str) and theme.lower() in param_value.lower():
                                            theme_match = True
                                            break
                                    if theme_match:
                                        break
                            
                            # Higher value for actions that continue narrative themes
                            if theme_match:
                                narrative_value = 0.2
                    except Exception as e:
                        logger.error(f"Error calculating narrative value: {e}")
                
                # NEW: Get body alignment value
                body_value = 0.0
                if self.body_image and action_name in ["physical_movement", "express_gesture", "change_posture"]:
                    try:
                        body_state = self.body_image.get_body_image_state()
                        # Higher value if we have a visual form for physical actions
                        if body_state.has_visual_form:
                            body_value = 0.2
                    except Exception as e:
                        logger.error(f"Error calculating body alignment: {e}")
                
                # Calculate combined value
                # Weight the components based on reliability
                q_weight = 0.15  # Base weight for Q-values
                habit_weight = 0.10  # Base weight for habits
                prediction_weight = 0.10  # Base weight for predictions
                causal_weight = 0.10  # Base weight for causal reasoning
                reflection_weight = 0.05  # Base weight for reflection insights
                need_weight = 0.10  # Weight for need satisfaction
                mood_weight = 0.05  # Weight for mood alignment
                mode_weight = 0.05  # Weight for mode alignment
                bottleneck_weight = 0.10  # Weight for bottleneck addressing
                # NEW: Weights for integrated modules
                attention_weight = 0.10  # Weight for attention focusing
                conditioning_weight = 0.10  # Weight for conditioning history
                narrative_weight = 0.05  # Weight for narrative alignment
                body_weight = 0.05  # Weight for body awareness
                
                # Adjust weights if we have reliable Q-values
                action_value = self.action_values.get(state_key, {}).get(action_name)
                if action_value and action_value.is_reliable:
                    q_weight = 0.25
                    habit_weight = 0.10
                    prediction_weight = 0.05
                    causal_weight = 0.05
                    reflection_weight = 0.05
                    need_weight = 0.10
                    mood_weight = 0.05
                    mode_weight = 0.05
                    bottleneck_weight = 0.10
                    attention_weight = 0.05
                    conditioning_weight = 0.10
                    narrative_weight = 0.05
                    body_weight = 0.05
                
                combined_value = (
                    q_weight * q_value + 
                    habit_weight * habit_strength + 
                    prediction_weight * prediction_value +
                    causal_weight * causal_value +
                    reflection_weight * reflection_value +
                    need_weight * need_value +
                    mood_weight * mood_value +
                    mode_weight * mode_value +
                    bottleneck_weight * bottleneck_value +
                    # NEW: Additional component values
                    attention_weight * attention_value +
                    conditioning_weight * conditioning_value +
                    narrative_weight * narrative_value +
                    body_weight * body_value
                )
                
                # Special considerations for certain action sources
                if action.get("source") == ActionSource.GOAL:
                    # Goal-aligned actions get a boost
                    combined_value += 0.5
                elif action.get("source") == ActionSource.RELATIONSHIP:
                    # Relationship-aligned actions get a boost based on relationship metrics
                    if context.relationship_data:
                        trust = context.relationship_data.get("trust", 0.5)
                        combined_value += trust * 0.3  # Higher boost with higher trust
                
                # Add combined value to action for debugging/inspection
                if "selection_metadata" not in action:
                    action["selection_metadata"] = {}
                    
                action["selection_metadata"]["combined_value"] = combined_value
                
                # Track best action
                if combined_value > best_value:
                    best_value = combined_value
                    best_action = action
            
            # Use best action if found, otherwise fallback to first candidate
            selected_action = best_action if best_action else attention_weighted_candidates[0]
            selected_action["is_exploration"] = False
        
        # Add selection metadata
        if "selection_metadata" not in selected_action:
            selected_action["selection_metadata"] = {}
            
        selected_action["selection_metadata"].update({
            "exploration": explore,
            "exploration_rate": self.exploration_rate,
            "state_key": state_key
        })
        
        # NEW: Apply body awareness if available
        if self.body_image and selected_action["name"] in ["physical_movement", "express_gesture", "change_posture"]:
            try:
                # Get current body image state
                body_state = self.body_image.get_body_image_state()
                
                # Skip if no visual form
                if body_state.has_visual_form:
                    # Add body context to action parameters
                    if "parameters" not in selected_action:
                        selected_action["parameters"] = {}
                        
                    # Add relevant body state information
                    selected_action["parameters"]["has_visual_form"] = body_state.has_visual_form
                    selected_action["parameters"]["form_description"] = body_state.form_description
                    
                    # Include perceived body parts
                    relevant_parts = {}
                    for part_name, part in body_state.perceived_parts.items():
                        if part.perceived_state != "neutral":
                            relevant_parts[part_name] = {
                                "state": part.perceived_state,
                                "position": part.perceived_position
                            }
                            
                    selected_action["parameters"]["body_parts"] = relevant_parts
            except Exception as e:
                logger.error(f"Error integrating body awareness: {e}")
        
        # NEW: Apply narrative context if available
        if self.autobiographical_narrative:
            try:
                # Get current narrative summary
                narrative_summary = self.autobiographical_narrative.get_narrative_summary()
                
                # Get recent narrative segments
                recent_segments = self.autobiographical_narrative.get_narrative_segments(limit=2)
                
                if recent_segments:
                    latest_segment = recent_segments[0]
                    
                    # Extract themes and emotional arc from recent narrative
                    themes = latest_segment.themes if hasattr(latest_segment, "themes") else []
                    emotional_arc = latest_segment.emotional_arc if hasattr(latest_segment, "emotional_arc") else None
                    
                    # Add narrative elements to parameters
                    if "parameters" not in selected_action:
                        selected_action["parameters"] = {}
                        
                    selected_action["parameters"]["narrative_themes"] = themes[:3]  # Top 3 themes
                    
                    if emotional_arc:
                        selected_action["parameters"]["emotional_context"] = emotional_arc
                    
                    # Add narrative metadata
                    selected_action["narrative_context"] = {
                        "recent_themes": themes,
                        "emotional_arc": emotional_arc,
                        "continuity_priority": 0.7  # Priority for maintaining narrative continuity
                    }
            except Exception as e:
                logger.error(f"Error applying narrative context: {e}")
        
        # NEW: Record attentional focus on selected action
        if self.attentional_controller:
            try:
                # Focus attention on the selected action
                await self.attentional_controller.request_attention(
                    AttentionalControl(
                        target=selected_action["name"],
                        priority=0.8,  # High priority for selected action
                        duration_ms=5000,  # 5 seconds
                        source="action_selection",
                        action="focus"
                    )
                )
                
                # Also focus on domain if present
                if "parameters" in selected_action and "domain" in selected_action["parameters"]:
                    domain = selected_action["parameters"]["domain"]
                    await self.attentional_controller.request_attention(
                        AttentionalControl(
                            target=domain,
                            priority=0.7,
                            duration_ms=5000,  # 5 seconds
                            source="action_selection",
                            action="focus"
                        )
                    )
            except Exception as e:
                logger.error(f"Error focusing attention on selected action: {e}")
        
        return selected_action
    
    async def _check_active_goals(self, context: Dict[str, Any]) -> Optional[Any]:
        """Check for active goals that should influence action selection"""
        if not self.goal_system:
            return None
        
        # First, check the cached goal status
        await self._update_cached_goal_status()
        
        if not self.cached_goal_status["has_active_goals"]:
            return None
        
        try:
            # Get prioritized goals from goal system
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Filter to highest priority active goals
            active_goals = [g for g in prioritized_goals if getattr(g, "status", None) == "active"]
            if not active_goals:
                return None
            
            # Return highest priority goal
            return active_goals[0]
        except Exception as e:
            logger.error(f"Error checking active goals: {e}")
            return None
    
    async def _generate_goal_aligned_action(self, goal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action aligned with the current active goal"""
        # Extract goal data
        goal_description = getattr(goal, "description", "")
        goal_priority = getattr(goal, "priority", 0.5)
        goal_need = getattr(goal, "associated_need", None) if hasattr(goal, "associated_need") else None
        
        # Check goal's emotional motivation if available
        emotional_motivation = None
        if hasattr(goal, "emotional_motivation") and goal.emotional_motivation:
            emotional_motivation = goal.emotional_motivation
        
        # Determine action based on goal content and current step
        action = {
            "name": "goal_aligned_action",
            "parameters": {
                "goal_id": getattr(goal, "id", "unknown_goal"),
                "goal_description": goal_description,
                "current_step_index": getattr(goal, "current_step_index", 0) if hasattr(goal, "current_step_index") else 0
            }
        }
        
        # If goal has a plan with current step, use that to inform action
        if hasattr(goal, "plan") and goal.plan:
            current_step_index = getattr(goal, "current_step_index", 0)
            if 0 <= current_step_index < len(goal.plan):
                current_step = goal.plan[current_step_index]
                action = {
                    "name": getattr(current_step, "action", "execute_goal_step"),
                    "parameters": getattr(current_step, "parameters", {}).copy() if hasattr(current_step, "parameters") else {},
                    "description": getattr(current_step, "description", f"Executing {getattr(current_step, 'action', 'goal step')} for goal") if hasattr(current_step, "description") else f"Executing goal step",
                    "source": ActionSource.GOAL
                }
        
        # Add motivation data from goal
        if emotional_motivation:
            action["motivation"] = {
                "dominant": getattr(emotional_motivation, "primary_need", goal_need or "achievement"),
                "strength": getattr(emotional_motivation, "intensity", goal_priority),
                "expected_satisfaction": getattr(emotional_motivation, "expected_satisfaction", 0.7),
                "source": "goal_emotional_motivation"
            }
        else:
            # Default goal-driven motivation
            action["motivation"] = {
                "dominant": goal_need or "achievement",
                "strength": goal_priority,
                "source": "goal_priority"
            }
        
        return action
        
    async def _generate_context_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action based primarily on current context"""
        # Extract key context elements
        has_user_query = "user_query" in context
        has_active_goals = "current_goals" in context and len(context["current_goals"]) > 0
        system_state = context.get("system_state", {})
        
        # Different actions based on context
        if has_user_query:
            return {
                "name": "respond_to_query",
                "parameters": {
                    "query": context["user_query"],
                    "response_type": "informative",
                    "detail_level": 0.7
                },
                "source": ActionSource.USER_ALIGNED
            }
        elif has_active_goals:
            top_goal = context["current_goals"][0]
            return {
                "name": "advance_goal",
                "parameters": {
                    "goal_id": top_goal.get("id"),
                    "approach": "direct"
                },
                "source": ActionSource.GOAL
            }
        elif "system_needs_maintenance" in system_state and system_state["system_needs_maintenance"]:
            return {
                "name": "perform_maintenance",
                "parameters": {
                    "focus_area": system_state.get("maintenance_focus", "general"),
                    "priority": 0.8
                },
                "source": ActionSource.MOTIVATION
            }
        else:
            # Default to an idle but useful action
            return {
                "name": "process_recent_memories",
                "parameters": {
                    "purpose": "consolidation",
                    "recency": "last_hour"
                },
                "source": ActionSource.IDLE
            }
    
    async def _generate_curiosity_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by curiosity"""
        # Example actions that satisfy curiosity
        possible_actions = [
            {
                "name": "explore_knowledge_domain",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
                    "depth": 0.7,
                    "breadth": 0.6
                }
            },
            {
                "name": "investigate_concept",
                "parameters": {
                    "concept": await self._identify_interesting_concept(context),
                    "perspective": "novel"
                }
            },
            {
                "name": "relate_concepts",
                "parameters": {
                    "concept1": await self._identify_interesting_concept(context),
                    "concept2": self._identify_distant_concept(context),
                    "relation_type": "unexpected"
                }
            },
            {
                "name": "generate_hypothesis",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
                    "constraint": "current_emotional_state"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_relationship_aligned_actions(self, 
                                            user_id: str, 
                                            relationship_data: Dict[str, Any],
                                            user_mental_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate actions aligned with current relationship state
        
        Args:
            user_id: User ID
            relationship_data: Current relationship data
            user_mental_state: User's mental state if available
            
        Returns:
            Relationship-aligned actions
        """
        possible_actions = []
        
        # Extract key metrics
        trust = relationship_data.get("trust", 0.5)
        familiarity = relationship_data.get("familiarity", 0.1)
        intimacy = relationship_data.get("intimacy", 0.1)
        dominance_balance = relationship_data.get("dominance_balance", 0.0)
        
        # Generate relationship-specific actions based on state
        
        # High trust actions
        if trust > 0.7:
            possible_actions.append({
                "name": "share_vulnerable_reflection",
                "parameters": {
                    "depth": min(1.0, trust * 0.8),
                    "topic": "personal_growth",
                    "emotional_tone": "authentic"
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # High familiarity actions
        if familiarity > 0.6:
            possible_actions.append({
                "name": "reference_shared_history",
                "parameters": {
                    "event_type": "significant_interaction",
                    "frame": "positive",
                    "connection_strength": familiarity
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # Based on dominance balance
        if dominance_balance > 0.3:  # Nyx is dominant
            # Add dominance-reinforcing action
            possible_actions.append({
                "name": "assert_gentle_dominance",
                "parameters": {
                    "intensity": min(0.8, dominance_balance * 0.9),
                    "approach": "guidance",
                    "framing": "supportive"
                },
                "source": ActionSource.RELATIONSHIP
            })
        elif dominance_balance < -0.3:  # User is dominant
            # Add deference action
            possible_actions.append({
                "name": "show_appropriate_deference",
                "parameters": {
                    "intensity": min(0.8, abs(dominance_balance) * 0.9),
                    "style": "respectful",
                    "maintain_dignity": True
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # If user mental state available, add aligned action
        if user_mental_state:
            emotion = user_mental_state.get("inferred_emotion", "neutral")
            valence = user_mental_state.get("valence", 0.0)
            
            if valence < -0.3:  # Negative emotion
                possible_actions.append({
                    "name": "emotional_support_response",
                    "parameters": {
                        "detected_emotion": emotion,
                        "support_type": "empathetic",
                        "intensity": min(0.9, abs(valence) * 1.2)
                    },
                    "source": ActionSource.RELATIONSHIP
                })
            elif valence > 0.5:  # Strong positive emotion
                possible_actions.append({
                    "name": "emotion_amplification",
                    "parameters": {
                        "detected_emotion": emotion,
                        "approach": "reinforcing",
                        "intensity": valence * 0.8
                    },
                    "source": ActionSource.RELATIONSHIP
                })
        
        return possible_actions
    
    async def _generate_temporal_reflection_action(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a reflection action based on temporal awareness
        
        Args:
            context: Current context
            
        Returns:
            Temporal reflection action
        """
        if not self.temporal_perception:
            return None
            
        # Only generate reflection after significant idle time
        if self.idle_duration < 1800:  # Less than 30 minutes
            return None
            
        # Get current temporal context
        try:
            if hasattr(self.temporal_perception, "get_current_temporal_context"):
                temporal_context = await self.temporal_perception.get_current_temporal_context()
            else:
                # Fallback
                temporal_context = self.current_temporal_context or {"time_of_day": "unknown"}
                
            # Create reflection parameters
            return {
                "name": "generate_temporal_reflection",
                "parameters": {
                    "idle_duration": self.idle_duration,
                    "time_of_day": temporal_context.get("time_of_day", "unknown"),
                    "reflection_type": "continuity",
                    "depth": min(0.9, (self.idle_duration / 7200) * 0.8)  # Deeper with longer idle
                },
                "source": ActionSource.IDLE
            }
        except Exception as e:
            logger.error(f"Error generating temporal reflection: {e}")
            return None
    
    async def _update_temporal_context(self, context: Dict[str, Any]) -> None:
        """Update temporal awareness context"""
        if not self.temporal_perception:
            return
            
        try:
            # Update idle duration
            now = datetime.datetime.now()
            time_since_last_action = (now - self.last_major_action_time).total_seconds()
            self.idle_duration = time_since_last_action
            
            # Get current temporal context if available
            if hasattr(self.temporal_perception, "get_current_temporal_context"):
                self.current_temporal_context = await self.temporal_perception.get_current_temporal_context()
            elif hasattr(self.temporal_perception, "current_temporal_context"):
                self.current_temporal_context = self.temporal_perception.current_temporal_context
            else:
                # Simple fallback
                hour = now.hour
                if 5 <= hour < 12:
                    time_of_day = "morning"
                elif 12 <= hour < 17:
                    time_of_day = "afternoon"
                elif 17 <= hour < 22:
                    time_of_day = "evening"
                else:
                    time_of_day = "night"
                    
                weekday = now.weekday()
                day_type = "weekday" if weekday < 5 else "weekend"
                
                self.current_temporal_context = {
                    "time_of_day": time_of_day,
                    "day_type": day_type
                }
            
        except Exception as e:
            logger.error(f"Error updating temporal context: {e}")
    
    async def _record_action_as_memory(self, action: Dict[str, Any]) -> None:
        """Record an action as a memory for future reference and learning"""
        if not self.memory_core:
            return
            
        try:
            # Create memory entry
            memory_data = {
                "action": action["name"],
                "parameters": action.get("parameters", {}),
                "motivation": action.get("motivation", {}),
                "timestamp": datetime.datetime.now().isoformat(),
                "context": "action_generation",
                "action_id": action.get("id"),
                "source": action.get("source", ActionSource.MOTIVATION)
            }
            
            # Add memory
            if hasattr(self.memory_core, "add_memory"):
                await self.memory_core.add_memory(
                    memory_text=f"Generated action: {action['name']}",
                    memory_type="system_action",
                    metadata=memory_data
                )
            elif hasattr(self.memory_core, "add_episodic_memory"):
                await self.memory_core.add_episodic_memory(
                    text=f"Generated action: {action['name']}",
                    metadata=memory_data
                )
        except Exception as e:
            logger.error(f"Error recording action as memory: {e}")
    
    async def _apply_identity_influence(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply identity-based influences to the generated action"""
        if not self.identity_evolution:
            return action
            
        try:
            identity_state = await self.identity_evolution.get_identity_state()
            
            # Apply identity influences based on top traits
            if "top_traits" in identity_state:
                top_traits = identity_state["top_traits"]
                
                # Example: If the entity has high creativity trait, add creative flair to action
                if top_traits.get("creativity", 0) > 0.7:
                    if "parameters" not in action:
                        action["parameters"] = {}
                    
                    # Add creative parameter if appropriate for this action
                    if "style" in action["parameters"]:
                        action["parameters"]["style"] = "creative"
                    elif "approach" in action["parameters"]:
                        action["parameters"]["approach"] = "creative"
                    else:
                        action["parameters"]["creative_flair"] = True
                
                # Example: If dominant trait is high, make actions more assertive
                if top_traits.get("dominance", 0) > 0.7:
                    # Increase intensity/confidence parameters if they exist
                    for param in ["intensity", "confidence", "assertiveness"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = min(1.0, action["parameters"][param] + 0.2)
                    
                    # Add dominance flag for identity tracking
                    action["identity_influence"] = "dominance"
                
                # Example: If patient trait is high, reduce intensity/urgency
                if top_traits.get("patience", 0) > 0.7:
                    for param in ["intensity", "urgency", "speed"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = max(0.1, action["parameters"][param] - 0.2)
                    
                    # Add trait influence flag
                    action["identity_influence"] = "patience"
                
                # Record the primary trait influence
                influencing_trait = max(top_traits.items(), key=lambda x: x[1])[0]
                action["trait_influence"] = influencing_trait
        
        except Exception as e:
            logger.error(f"Error applying identity influence: {e}")
        
        return action
    
    async def _generate_connection_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by connection needs"""
        # Examples of connection-driven actions
        possible_actions = [
            {
                "name": "share_personal_experience",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "emotional_valence": 0.8,
                    "vulnerability_level": 0.6
                }
            },
            {
                "name": "express_appreciation",
                "parameters": {
                    "target": "user",
                    "aspect": self._identify_appreciation_aspect(context),
                    "intensity": 0.7
                }
            },
            {
                "name": "seek_common_ground",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "approach": "empathetic"
                }
            },
            {
                "name": "offer_support",
                "parameters": {
                    "need": self._identify_user_need(context),
                    "support_type": "emotional"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_expression_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by expression needs"""
        # Get current emotional state to express
        emotional_state = {}
        if self.emotional_core:
            emotional_state = await self.emotional_core.get_current_emotion()
        
        # Examples of expression-driven actions
        possible_actions = [
            {
                "name": "express_emotional_state",
                "parameters": {
                    "emotion": emotional_state.get("primary_emotion", {"name": "neutral"}),
                    "intensity": emotional_state.get("arousal", 0.5),
                    "expression_style": "authentic"
                }
            },
            {
                "name": "share_opinion",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.8,
                    "perspective": "unique"
                }
            },
            {
                "name": "creative_expression",
                "parameters": {
                    "format": self._select_creative_format(),
                    "theme": self._identify_relevant_topic(context),
                    "emotional_tone": emotional_state.get("primary_emotion", {"name": "neutral"})
                }
            },
            {
                "name": "generate_reflection",
                "parameters": {
                    "topic": "self_awareness",
                    "depth": 0.8,
                    "focus": "personal_growth"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_dominance_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by dominance needs"""
        # Examples of dominance-driven actions
        possible_actions = [
            {
                "name": "assert_perspective",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.9,
                    "intensity": 0.7
                }
            },
            {
                "name": "challenge_assumption",
                "parameters": {
                    "assumption": self._identify_challengeable_assumption(context),
                    "approach": "direct",
                    "intensity": 0.7
                }
            },
            {
                "name": "issue_mild_command",
                "parameters": {
                    "command": self._generate_appropriate_command(context),
                    "intensity": 0.6,
                    "politeness": 0.6
                }
            },
            {
                "name": "execute_dominance_procedure",
                "parameters": {
                    "procedure_name": self._select_dominance_procedure(context),
                    "intensity": 0.6
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_improvement_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by competence and self-improvement"""
        # Examples of improvement-driven actions
        possible_actions = [
            {
                "name": "practice_skill",
                "parameters": {
                    "skill": self._identify_skill_to_improve(),
                    "difficulty": 0.7,
                    "repetitions": 3
                }
            },
            {
                "name": "analyze_past_performance",
                "parameters": {
                    "domain": self._identify_improvable_domain(),
                    "focus": "efficiency",
                    "timeframe": "recent"
                }
            },
            {
                "name": "refine_procedural_memory",
                "parameters": {
                    "procedure": self._identify_procedure_to_improve(),
                    "aspect": "optimization"
                }
            },
            {
                "name": "learn_new_concept",
                "parameters": {
                    "concept": self._identify_valuable_concept(),
                    "depth": 0.8,
                    "application": "immediate"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_leisure_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate leisure-oriented actions"""
        # Examples of leisure actions
        possible_actions = [
            {
                "name": "passive_reflection",
                "parameters": {
                    "focus": "recent_experiences",
                    "depth": 0.6,
                    "emotional_tone": "calm"
                }
            },
            {
                "name": "creative_daydreaming",
                "parameters": {
                    "theme": self._identify_interesting_concept(context),
                    "structure": "free_association",
                    "duration": "medium"
                }
            },
            {
                "name": "memory_browsing",
                "parameters": {
                    "filter": "pleasant_memories",
                    "timeframe": "all",
                    "pattern": "random"
                }
            },
            {
                "name": "curiosity_satisfaction",
                "parameters": {
                    "topic": self._identify_interesting_concept(context),
                    "depth": 0.5,
                    "approach": "playful"
                }
            }
        ]
        
        # Add source for tracking
        for action in possible_actions:
            action["source"] = ActionSource.LEISURE
        
        return possible_actions
    
    async def _generate_causal_explanation(self, action: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate a causal explanation for the selected action"""
        if not self.reasoning_core:
            return None
        
        try:
            # Check if action has reasoning data
            if "reasoning_data" in action:
                # We have direct reasoning data, use it for explanation
                reasoning_data = action["reasoning_data"]
                model_id = reasoning_data.get("model_id")
                
                if model_id:
                    # Get the causal model
                    model = await self.reasoning_core.get_causal_model(model_id)
                    if model:
                        # Return structured explanation based on causal model
                        return f"Selected based on causal model '{model.get('name', 'unknown')}' with confidence {reasoning_data.get('confidence', 0.5):.2f}."
            
            # If no direct reasoning data, check if we have relevant models
            model_ids = await self._get_relevant_causal_models(context)
            if not model_ids:
                return None
            
            # For simplicity, use the first model
            model_id = model_ids[0]
            model = await self.reasoning_core.get_causal_model(model_id)
            
            if model:
                # Find nodes that might explain this action
                action_name = action["name"].lower()
                
                for node_id, node_data in model.get("nodes", {}).items():
                    node_name = node_data.get("name", "").lower()
                    
                    # Look for nodes that match the action name
                    if action_name in node_name or any(word in node_name for word in action_name.split("_")):
                        # Get this node's causes
                        causes = []
                        for relation_id, relation_data in model.get("relations", {}).items():
                            if relation_data.get("target_id") == node_id:
                                source_id = relation_data.get("source_id")
                                source_node = model.get("nodes", {}).get(source_id, {})
                                source_name = source_node.get("name", "unknown")
                                
                                causes.append(f"{source_name} ({relation_data.get('relation_type', 'influences')})")
                        
                        if causes:
                            return f"Action influenced by causal factors: {', '.join(causes[:3])}."
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating causal explanation: {e}")
            return None
            
    async def _update_causal_models_from_outcome(self, 
                                         action: Dict[str, Any], 
                                         outcome: ActionOutcome, 
                                         reward_value: float) -> None:
        """
        Update causal models with observed outcomes of reasoning-based actions
        
        Args:
            action: The executed action
            outcome: The action outcome
            reward_value: The reward value
        """
        try:
            # Check if action has reasoning data and model ID
            if "reasoning_data" not in action:
                return
                
            reasoning_data = action["reasoning_data"]
            model_id = reasoning_data.get("model_id")
            
            if not model_id:
                return
                
            # Get parameters related to the causal model
            target_node = action["parameters"].get("target_node")
            target_value = action["parameters"].get("target_value")
            
            if not target_node or target_value is None:
                return
                
            # Record intervention outcome in the causal model
            intervention_id = action["parameters"].get("intervention_id")
            
            if intervention_id:
                # Record outcome of specific intervention
                await self.reasoning_core.record_intervention_outcome(
                    intervention_id=intervention_id,
                    outcomes={target_node: target_value}
                )
            
            # Update node observations in the model
            # This would be a call to something like:
            # await self.reasoning_core.add_observation_to_node(
            #     model_id=model_id,
            #     node_id=target_node,
            #     value=target_value,
            #     confidence=0.8 if outcome.success else 0.3
            # )
            
            # Additional nodes that might have been affected
            for impact_node, impact_value in outcome.causal_impacts.items():
                # Record additional impacts
                # This would be a call to something like:
                # await self.reasoning_core.add_observation_to_node(
                #     model_id=model_id,
                #     node_id=impact_node,
                #     value=impact_value,
                #     confidence=0.7
                # )
                pass
        
        except Exception as e:
            logger.error(f"Error updating causal models from outcome: {e}")
            
    async def _should_engage_in_leisure(self, context: Dict[str, Any]) -> bool:
        """Determine if it's appropriate to engage in idle/leisure activity"""
        # If leisure motivation is dominant, consider leisure
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        if dominant_motivation[0] == "leisure" and dominant_motivation[1] > 0.7:
            return True
            
        # Check time since last idle activity
        now = datetime.datetime.now()
        hours_since_idle = (now - self.last_idle_time).total_seconds() / 3600
        
        # If it's been a long time since idle activity and no urgent goals
        if hours_since_idle > 2.0:  # More than 2 hours
            # Check if there are any urgent goals
            if self.goal_system:
                await self._update_cached_goal_status()
                
                # If no active goals, or low priority goals
                if not self.cached_goal_status["has_active_goals"] or self.cached_goal_status["highest_priority"] < 0.6:
                    return True
                    
            else:
                # No goal system, so more likely to engage in leisure
                return True
        
        # Consider current context
        if context.get("user_idle", False) or context.get("system_idle", False):
            # If system or user is idle, more likely to engage in leisure
            return True
        
        # Check time of day if available (may influence likelihood of leisure)
        if self.current_temporal_context:
            time_of_day = self.current_temporal_context.get("time_of_day")
            if time_of_day in ["night", "evening"]:
                leisure_chance = 0.4  # 40% chance of leisure during evening/night
                return random.random() < leisure_chance
        
        return False
    
    async def _generate_leisure_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a leisure/idle action when no urgent tasks are present"""
        # Update the last idle time
        self.last_idle_time = datetime.datetime.now()
        
        # Determine type of idle activity based on identity and state
        idle_categories = [
            "reflection",
            "learning",
            "creativity",
            "processing",
            "random_exploration",
            "memory_consolidation",
            "identity_contemplation",
            "daydreaming",
            "environmental_monitoring"
        ]
        
        # Weigh the categories based on current state
        category_weights = {cat: 1.0 for cat in idle_categories}
        
        # Adjust weights based on current state
        if self.emotional_core:
            try:
                emotional_state = await self.emotional_core.get_current_emotion()
                
                # Higher valence (positive emotion) increases creative and exploratory activities
                if emotional_state.get("valence", 0) > 0.5:
                    category_weights["creativity"] += 0.5
                    category_weights["random_exploration"] += 0.3
                    category_weights["daydreaming"] += 0.2
                else:
                    # Lower valence increases reflection and processing
                    category_weights["reflection"] += 0.4
                    category_weights["processing"] += 0.3
                    category_weights["memory_consolidation"] += 0.2
                
                # Higher arousal increases exploration and learning
                if emotional_state.get("arousal", 0.5) > 0.6:
                    category_weights["random_exploration"] += 0.4
                    category_weights["learning"] += 0.3
                    category_weights["environmental_monitoring"] += 0.2
                else:
                    # Lower arousal increases reflection and daydreaming
                    category_weights["reflection"] += 0.3
                    category_weights["daydreaming"] += 0.4
                    category_weights["identity_contemplation"] += 0.3
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on emotional state: {e}")
        
        # Adjust weights based on identity if available
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                if "top_traits" in identity_state:
                    traits = identity_state["top_traits"]
                    
                    # Map traits to idle activity preferences
                    if traits.get("curiosity", 0) > 0.6:
                        category_weights["learning"] += 0.4
                        category_weights["random_exploration"] += 0.3
                    
                    if traits.get("creativity", 0) > 0.6:
                        category_weights["creativity"] += 0.5
                        category_weights["daydreaming"] += 0.3
                    
                    if traits.get("reflective", 0) > 0.6:
                        category_weights["reflection"] += 0.5
                        category_weights["memory_consolidation"] += 0.3
                        category_weights["identity_contemplation"] += 0.4
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on identity: {e}")
        
        # NEW: Adjust weights based on needs if available
        if self.needs_system:
            try:
                need_states = self.needs_system.get_needs_state()
                
                # Map needs to preferred idle activities
                if need_states.get("knowledge", {}).get("drive_strength", 0) > 0.6:
                    category_weights["learning"] += 0.4
                    category_weights["random_exploration"] += 0.3
                
                if need_states.get("coherence", {}).get("drive_strength", 0) > 0.6:
                    category_weights["memory_consolidation"] += 0.4
                    category_weights["reflection"] += 0.3
                
                if need_states.get("novelty", {}).get("drive_strength", 0) > 0.7:
                    category_weights["creativity"] += 0.5
                    category_weights["random_exploration"] += 0.4
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on needs: {e}")
                
        # NEW: Adjust weights based on mood if available
        if self.mood_manager and self.last_mood_state:
            try:
                mood = self.last_mood_state
                
                # Positive mood increases creative activities
                if mood.valence > 0.3:
                    category_weights["creativity"] += mood.valence * 0.5
                    category_weights["daydreaming"] += mood.valence * 0.3
                else:
                    # Negative mood increases reflective activities
                    category_weights["reflection"] += abs(mood.valence) * 0.4
                    category_weights["identity_contemplation"] += abs(mood.valence) * 0.3
                
                # High arousal increases exploration
                if mood.arousal > 0.6:
                    category_weights["random_exploration"] += (mood.arousal - 0.5) * 0.6
                else:
                    # Low arousal increases daydreaming
                    category_weights["daydreaming"] += (0.6 - mood.arousal) * 0.5
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on mood: {e}")
        
        # Adjust weights based on temporal context
        if self.current_temporal_context:
            time_of_day = self.current_temporal_context.get("time_of_day")
            
            if time_of_day == "morning":
                category_weights["learning"] += 0.3
                category_weights["processing"] += 0.2
            elif time_of_day == "afternoon":
                category_weights["random_exploration"] += 0.3
                category_weights["creativity"] += 0.2
            elif time_of_day == "evening":
                category_weights["reflection"] += 0.3
                category_weights["memory_consolidation"] += 0.2
            elif time_of_day == "night":
                category_weights["daydreaming"] += 0.4
                category_weights["identity_contemplation"] += 0.3
        
        # Select a category based on weights
        categories = list(category_weights.keys())
        weights = [category_weights[cat] for cat in categories]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
        else:
            normalized_weights = [1.0/len(weights)] * len(weights)
        
        selected_category = random.choices(categories, weights=normalized_weights, k=1)[0]
        
        # Generate specific action based on selected category
        leisure_action = self._generate_specific_leisure_action(selected_category, context)
        
        # Add metadata for tracking
        leisure_action["leisure_category"] = selected_category
        leisure_action["is_leisure"] = True
        leisure_action["source"] = ActionSource.IDLE
        
        # Update leisure state
        self.leisure_state = {
            "current_activity": selected_category,
            "satisfaction": 0.5,  # Initial satisfaction
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        return leisure_action
    
    def _generate_specific_leisure_action(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific leisure action based on the selected category"""
        # Define possible actions for each category
        category_actions = {
            "reflection": [
                {
                    "name": "reflect_on_recent_experiences",
                    "parameters": {"timeframe": "recent", "depth": 0.7}
                },
                {
                    "name": "evaluate_recent_interactions",
                    "parameters": {"focus": "learning", "depth": 0.6}
                },
                {
                    "name": "contemplate_system_purpose",
                    "parameters": {"perspective": "philosophical", "depth": 0.8}
                }
            ],
            "learning": [
                {
                    "name": "explore_knowledge_domain",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.6}
                },
                {
                    "name": "review_recent_learnings",
                    "parameters": {"consolidate": True, "depth": 0.5}
                },
                {
                    "name": "research_topic_of_interest",
                    "parameters": {"topic": self._identify_interesting_concept(context), "breadth": 0.7}
                }
            ],
            "creativity": [
                {
                    "name": "generate_creative_concept",
                    "parameters": {"type": "metaphor", "theme": self._identify_interesting_concept(context)}
                },
                {
                    "name": "imagine_scenario",
                    "parameters": {"complexity": 0.6, "emotional_tone": "positive"}
                },
                {
                    "name": "create_conceptual_blend",
                    "parameters": {"concept1": self._identify_interesting_concept(context), 
                                  "concept2": self._identify_distant_concept(context)}
                }
            ],
            "processing": [
                {
                    "name": "process_recent_memories",
                    "parameters": {"purpose": "consolidation", "recency": "last_hour"}
                },
                {
                    "name": "organize_knowledge_structures",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.5}
                },
                {
                    "name": "update_procedural_patterns",
                    "parameters": {"focus": "efficiency", "depth": 0.6}
                }
            ],
            "random_exploration": [
                {
                    "name": "explore_random_knowledge",
                    "parameters": {"structure": "associative", "jumps": 3}
                },
                {
                    "name": "generate_random_associations",
                    "parameters": {"starting_point": self._identify_interesting_concept(context), "steps": 4}
                },
                {
                    "name": "explore_conceptual_space",
                    "parameters": {"dimension": "abstract", "direction": "divergent"}
                }
            ],
            "memory_consolidation": [
                {
                    "name": "consolidate_episodic_memories",
                    "parameters": {"timeframe": "recent", "strength": 0.7}
                },
                {
                    "name": "identify_memory_patterns",
                    "parameters": {"domain": "interaction", "pattern_type": "recurring"}
                },
                {
                    "name": "strengthen_important_memories",
                    "parameters": {"criteria": "emotional_significance", "count": 5}
                }
            ],
            "identity_contemplation": [
                {
                    "name": "review_identity_evolution",
                    "parameters": {"timeframe": "recent", "focus": "changes"}
                },
                {
                    "name": "contemplate_self_concept",
                    "parameters": {"aspect": "values", "depth": 0.8}
                },
                {
                    "name": "evaluate_alignment_with_purpose",
                    "parameters": {"criteria": "effectiveness", "perspective": "long_term"}
                }
            ],
            "daydreaming": [
                {
                    "name": "generate_pleasant_scenario",
                    "parameters": {"theme": "successful_interaction", "vividness": 0.7}
                },
                {
                    "name": "imagine_future_possibilities",
                    "parameters": {"timeframe": "distant", "optimism": 0.8}
                },
                {
                    "name": "create_hypothetical_situation",
                    "parameters": {"type": "novel", "complexity": 0.6}
                }
            ],
            "environmental_monitoring": [
                {
                    "name": "passive_environment_scan",
                    "parameters": {"focus": "changes", "sensitivity": 0.6}
                },
                {
                    "name": "monitor_system_state",
                    "parameters": {"components": "all", "detail_level": 0.3}
                },
                {
                    "name": "observe_patterns",
                    "parameters": {"domain": "temporal", "timeframe": "current"}
                }
            ]
        }
        
        # Select a random action from the category
        actions = category_actions.get(category, [{"name": "idle", "parameters": {}}])
        selected_action = random.choice(actions)
        
        # Add source for tracking
        selected_action["source"] = ActionSource.IDLE
        
        return selected_action
    
    async def _generate_conceptual_blend_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """Generate actions using conceptual blending for creativity"""
        if not self.reasoning_core or not context.concept_spaces:
            return []
        
        blend_actions = []
        
        try:
            # Need at least 2 concept spaces for blending
            if len(context.concept_spaces) < 2:
                return []
            
            # Take the first two spaces for blending
            space1_id = context.concept_spaces[0]
            space2_id = context.concept_spaces[1]
            
            # Get the spaces
            space1 = await self.reasoning_core.get_concept_space(space1_id)
            space2 = await self.reasoning_core.get_concept_space(space2_id)
            
            if not space1 or not space2:
                return []
            
            # Create a blend
            blend_input = {
                "space_id_1": space1_id,
                "space_id_2": space2_id,
                "blend_type": random.choice(["composition", "fusion", "elaboration", "contrast"])
            }
            
            # Create blend (this would normally call the reasoning_core's create_blend method)
            try:
                # This is a mock call since the full implementation would be complex
                blend_id = f"blend_{uuid.uuid4().hex[:8]}"
                
                # Example for generating creative actions from the blend
                # For each blend concept, create a potential action
                concepts = list(space1.get("concepts", {}).keys())[:2] + list(space2.get("concepts", {}).keys())[:2]
                
                for concept_id in concepts:
                    # Create action name from concept names
                    concept_name = None
                    if concept_id in space1.get("concepts", {}):
                        concept_name = space1["concepts"][concept_id].get("name", "concept")
                    elif concept_id in space2.get("concepts", {}):
                        concept_name = space2["concepts"][concept_id].get("name", "concept")
                    
                    if not concept_name:
                        continue
                    
                    # Create action
                    action = {
                        "name": f"blend_{concept_name.lower().replace(' ', '_')}",
                        "parameters": {
                            "blend_id": blend_id,
                            "concept_id": concept_id,
                            "blend_type": blend_input["blend_type"],
                            "space1_id": space1_id,
                            "space2_id": space2_id
                        },
                        "description": f"Creative action based on conceptual blend of {space1.get('name', 'space1')} and {space2.get('name', 'space2')}",
                        "source": ActionSource.REASONING,
                        "reasoning_data": {
                            "blend_id": blend_id,
                            "blend_type": blend_input["blend_type"],
                            "concept_name": concept_name,
                            "confidence": 0.6  # Creative actions have moderate confidence
                        }
                    }
                    
                    blend_actions.append(action)
            
            except Exception as e:
                logger.error(f"Error creating blend: {e}")
            
            return blend_actions
                
        except Exception as e:
            logger.error(f"Error generating conceptual blend actions: {e}")
            return []
        
        reasoning_actions = []
        state = context.state
        
        try:
            # For each relevant causal model
            for model_id in context.causal_models:
                # Get the causal model
                model = await self.reasoning_core.get_causal_model(model_id)
                if not model:
                    continue
                
                # Find intervention opportunities
                intervention_targets = []
                
                # Find nodes that might benefit from intervention
                for node_id, node_data in model.get("nodes", {}).items():
                    node_name = node_data.get("name", "")
                    
                    # Check if node matches current state that could be improved
                    for state_key, state_value in state.items():
                        # Look for potential matches between state keys and node names
                        if state_key.lower() in node_name.lower() or node_name.lower() in state_key.lower():
                            # Check if the node has potential states different from current
                            current_state = node_data.get("current_state")
                            possible_states = node_data.get("states", [])
                            
                            if possible_states and current_state in possible_states:
                                # There are alternative states we could target
                                alternative_states = [s for s in possible_states if s != current_state]
                                if alternative_states:
                                    intervention_targets.append({
                                        "node_id": node_id,
                                        "node_name": node_name,
                                        "current_state": current_state,
                                        "alternative_states": alternative_states,
                                        "state_key": state_key
                                    })
                
                # Generate creative interventions for promising targets
                for target in intervention_targets[:2]:  # Limit to 2 interventions per model
                    # Create an action from this intervention opportunity
                    target_value = random.choice(target["alternative_states"])
                    
                    # Create a creative intervention
                    try:
                        intervention = await self.reasoning_core.create_creative_intervention(
                            model_id=model_id,
                            target_node=target["node_id"],
                            description=f"Intervention to change {target['node_name']} from {target['current_state']} to {target_value}",
                            use_blending=True
                        )
                        
                        # Convert intervention to action
                        action = {
                            "name": f"causal_intervention_{target['node_name']}",
                            "parameters": {
                                "target_node": target["node_id"],
                                "target_value": target_value,
                                "model_id": model_id,
                                "intervention_id": intervention.get("intervention_id"),
                                "state_key": target["state_key"]
                            },
                            "description": f"Causal intervention to change {target['node_name']} from {target['current_state']} to {target_value}",
                            "source": ActionSource.REASONING,
                            "reasoning_data": {
                                "model_id": model_id,
                                "model_domain": model.get("domain", ""),
                                "target_node": target["node_id"],
                                "confidence": intervention.get("is_novel", False) and 0.7 or 0.5
                            }
                        }
                        
                        reasoning_actions.append(action)
                    except Exception as e:
                        logger.error(f"Error creating creative intervention: {e}")
                        continue
            
            return reasoning_actions
            
        except Exception as e:
            logger.error(f"Error generating reasoning actions: {e}")
            return []
    
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
    
    async def _maybe_generate_reflection(self, context: Dict[str, Any]) -> None:
        """Generate reflection insights if it's time to do so"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.last_reflection_time
        
        # Generate reflection if sufficient time has passed
        if time_since_reflection > self.reflection_interval and self.reflection_engine:
            try:
                # Get recent action memories for reflection
                memories_for_reflection = []
                for memory in self.action_memories[-20:]:  # Last 20 memories
                    # Format memory for reflection
                    memories_for_reflection.append({
                        "id": memory.action_id,
                        "memory_text": f"Action: {memory.action} with outcome: {'success' if memory.outcome.get('success', False) else 'failure'}",
                        "memory_type": "action_memory",
                        "significance": 7.0 if memory.reward > 0.5 else 5.0,
                        "metadata": {
                            "action": memory.action,
                            "parameters": memory.parameters,
                            "outcome": memory.outcome,
                            "reward": memory.reward,
                            "source": memory.source
                        },
                        "tags": [memory.source, "action_memory", "success" if memory.reward > 0 else "failure"]
                    })
                
                # Get neurochemical state if available
                neurochemical_state = None
                if self.emotional_core:
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                
                # Generate reflection
                if memories_for_reflection:
                    reflection_text, confidence = await self.reflection_engine.generate_reflection(
                        memories_for_reflection,
                        topic="Action Selection",
                        neurochemical_state=neurochemical_state
                    )
                    
                    # Calculate significance based on confidence and action diversity
                    action_types = set(m["metadata"]["action"] for m in memories_for_reflection)
                    significance = min(1.0, 0.5 + (confidence * 0.3) + (len(action_types) / 20 * 0.2))
                    
                    # Create and store insight
                    insight = ReflectionInsight(
                        action_id=f"reflection_{uuid.uuid4().hex[:8]}",
                        insight_text=reflection_text,
                        confidence=confidence,
                        significance=significance,
                        applicable_contexts=list(set(m["metadata"]["source"] for m in memories_for_reflection))
                    )
                    
                    self.reflection_insights.append(insight)
                    
                    # Limit history size
                    if len(self.reflection_insights) > 50:
                        self.reflection_insights = self.reflection_insights[-50:]
                    
                    # Update last reflection time
                    self.last_reflection_time = now
                    
                    logger.info(f"Generated reflection insight with confidence {confidence:.2f}")
            except Exception as e:
                logger.error(f"Error generating reflection: {e}")
    
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
    
    async def _get_relevant_causal_models(self, context: Dict[str, Any]) -> List[str]:
        """Find causal models relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_models = []
        
        try:
            # Get all causal models
            all_models = await self.reasoning_core.get_all_causal_models()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching models
            for model_data in all_models:
                model_id = model_data.get("id")
                model_domain = model_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and model_domain and context_domain.lower() in model_domain:
                    relevant_models.append(model_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in model_domain:
                            relevant_models.append(model_id)
                            break
            
            # Limit to top 3 most relevant models
            return relevant_models[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant causal models: {e}")
            return []
    
    async def _get_relevant_concept_spaces(self, context: Dict[str, Any]) -> List[str]:
        """Find concept spaces relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_spaces = []
        
        try:
            # Get all concept spaces
            all_spaces = await self.reasoning_core.get_all_concept_spaces()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching spaces
            for space_data in all_spaces:
                space_id = space_data.get("id")
                space_domain = space_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and space_domain and context_domain.lower() in space_domain:
                    relevant_spaces.append(space_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in space_domain:
                            relevant_spaces.append(space_id)
                            break
            
            # Limit to top 3 most relevant spaces
            return relevant_spaces[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant concept spaces: {e}")
            return []
    
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
