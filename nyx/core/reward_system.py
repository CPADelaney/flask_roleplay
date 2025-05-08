# nyx/core/reward_system.py

import logging
import math
import random
import time
import datetime
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union
from pydantic import BaseModel, Field
from collections import defaultdict
from enum import Enum
from nyx.core.emotions.context import EmotionalContext

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    trace, 
    GuardrailFunctionOutput,
    InputGuardrail,
    FunctionTool,
    RunConfig,
    RunContextWrapper
)

logger = logging.getLogger(__name__)

class RewardType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class RewardSignal(BaseModel):
    """Schema for nyxaminergic reward signal"""
    value: float = Field(..., description="Reward value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: str = Field(..., description="Source generating the reward (e.g., GoalManager, user_compliance)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context info")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    @property
    def reward_type(self) -> RewardType:
        """Categorize the reward as positive, negative, or neutral."""
        if self.value > 0.05:
            return RewardType.POSITIVE
        elif self.value < -0.05:
            return RewardType.NEGATIVE
        else:
            return RewardType.NEUTRAL

class RewardMemory(BaseModel):
    """Schema for stored reward memory"""
    state: Dict[str, Any] = Field(..., description="State that led to reward")
    action: str = Field(..., description="Action that was taken")
    reward: float = Field(..., description="Reward value received")
    next_state: Optional[Dict[str, Any]] = Field(None, description="Resulting state")
    timestamp: str = Field(..., description="When this memory was created")
    source: str = Field("unknown", description="Source of reward")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    novelty_index: float = Field(0.5, ge=0.0, le=1.0, description="Novelty or escalation of the action (higher = more novel)")

class ActionValue(BaseModel):
    """Q-value for a state-action pair."""
    state_key: str
    action: str
    value: float = 0.0
    update_count: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    confidence: float = Field(0.2, ge=0.0, le=1.0)
    novelty_value: float = Field(0.5, ge=0.0, le=1.0, description="How novel this action is on average")
    
    @property
    def is_reliable(self) -> bool:
        """Whether this action value has enough updates to be considered reliable."""
        return self.update_count >= 3 and self.confidence >= 0.5

class RewardAnalysisOutput(BaseModel):
    """Output schema for reward analysis agent"""
    patterns: List[Dict[str, Any]] = Field(..., description="Identified reward patterns")
    suggestions: List[Dict[str, Any]] = Field(..., description="Suggestions for improvement")
    learning_params: Dict[str, float] = Field(..., description="Recommended learning parameters")
    effectiveness: Dict[str, float] = Field(..., description="Effectiveness metrics")
    insights: str = Field(..., description="Key insights about reward system")

# Function tools for the reward system

async def _categorize_reward_logic(reward_value: float) -> RewardType:
    """
    Core logic: Categorize a reward value as positive, negative, or neutral
    """
    if reward_value > 0.05:
        return RewardType.POSITIVE
    elif reward_value < -0.05:
        return RewardType.NEGATIVE
    else:
        return RewardType.NEUTRAL

async def _calculate_nyxamine_change_logic(
    reward_value: float,
    current_nyxamine: float,
    baseline_nyxamine: float = 0.5
) -> Dict[str, float]:
    """
    Core logic: Calculate the change in nyxamine based on a reward value
    """
    if reward_value >= 0:
        nyxamine_change = reward_value * 0.3
    else:
        nyxamine_change = reward_value * 0.2
    new_nyxamine = max(0.0, min(1.0, current_nyxamine + nyxamine_change))
    return {
        "nyxamine_change": nyxamine_change,
        "old_nyxamine": current_nyxamine,
        "new_nyxamine": new_nyxamine,
        "is_significant": abs(nyxamine_change) > 0.1
    }

async def _calculate_submission_value_logic(
    submission_type: str,
    was_initially_resistant: bool = False,
    depravity_hint: Optional[float] = None,
    novelty: Optional[float] = None
) -> float:
    """
    Core logic: Calculate base reward value for submission types (integrates depravity/novelty).
    """
    submission_values = {
        "verbal": 0.4, "honorific": 0.5, "behavioral": 0.6, "ritual": 0.7,
        "task": 0.6, "service": 0.7, "degradation": 0.8, "humiliation": 0.9,
        "pain_simulation": 0.9, "psychological": 0.95, "ownership": 1.0
    }
    base_value = submission_values.get(submission_type, 0.6)
    if depravity_hint is not None:
        base_value *= (1.0 + (depravity_hint * 0.4))
    if novelty is not None:
        base_value *= (0.8 + novelty * 0.4)
    if was_initially_resistant:
        base_value *= 1.5
    elif submission_type == "immediate": # Assuming "immediate" implies less effort/meaning?
        base_value *= 0.5
    return round(min(base_value, 1.5), 4)

categorize_reward_tool = function_tool(
    _categorize_reward_logic,
    name_override="categorize_reward",
    description_override="Categorize a reward value as positive, negative, or neutral"
)

calculate_nyxamine_change_tool = function_tool(
    _calculate_nyxamine_change_logic,
    name_override="calculate_nyxamine_change",
    description_override="Calculate the change in nyxamine based on a reward value"
)

calculate_submission_value_tool = function_tool(
    _calculate_submission_value_logic,
    name_override="calculate_submission_value",
    description_override="Calculate base reward value for submission types (integrates depravity/novelty)."
)   

class RewardSignalProcessor:
    """
    Processes reward signals and implements reward-based learning.
    Simulates nyxaminergic pathways for reinforcement learning.
    """
    
    def __init__(self, emotional_core=None, identity_evolution=None, somatosensory_system=None, mood_manager=None, needs_system=None):
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        self.somatosensory_system = somatosensory_system
        self.mood_manager = mood_manager
        self.needs_system = needs_system

        
        # Reward signal history
        self.reward_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Reward learning data
        self.reward_memories: List[RewardMemory] = []
        self.max_memories = 5000
        
        # Action-value mapping (Q-values) - improved structure with explicit ActionValue objects
        self.action_values: Dict[str, Dict[str, ActionValue]] = defaultdict(dict)
        
        # Habit strength tracking
        self.habits: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        
        # nyxamine system parameters
        self.baseline_nyxamine = 0.5
        self.current_nyxamine = 0.5
        self.nyxamine_decay_rate = 0.1  # Decay per second
        self.last_update_time = time.time()
        
        # Reward thresholds for different effects
        self.significant_reward_threshold = 0.7
        self.habit_formation_threshold = 0.6
        self.identity_update_threshold = 0.8
        
        # Performance tracking
        self.total_reward = 0.0
        self.positive_rewards = 0
        self.negative_rewards = 0
        
        # Categorized reward tracking
        self.category_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total": 0.0, "positive": 0, "negative": 0})
        
        # Locks for thread safety
        self._main_lock = asyncio.Lock()
        self._nyxamine_lock = asyncio.Lock()
        
        # Create agents
        self.learning_agent = self._create_learning_agent()
        self.conditioning_agent = self._create_conditioning_agent()

        # Novelty seeking
        self.novelty_decay: Dict[str, float] = defaultdict(lambda: 1.0)  # 1.0 = max novelty

        self.recent_depravity_levels: List[float] = []
        self.max_depravity_history = 50

        logger.info("RewardSignalProcessor initialized")
    
    def _create_learning_agent(self) -> Optional[Agent]:
        """Creates an agent to analyze reward patterns and improve learning."""
        try:
            return Agent(
                name="Reward Learning Agent",
                instructions="""You analyze reward patterns and learning outcomes for the Nyx AI.
                Based on reward histories, action values, and learning performance, you'll:
                
                1. Identify patterns in which actions lead to positive or negative rewards in specific contexts.
                2. Suggest improvements to the reward processing system.
                3. Analyze the effectiveness of current learning parameters.
                4. Recommend optimization strategies for reinforcement learning.
                
                Your inputs will include reward histories, action-value pairs, and performance metrics.
                Focus on data-driven insights and concrete, implementable suggestions.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(temperature=0.3),
                output_type=RewardAnalysisOutput,
                tools=[
                    # Pass the FunctionTool OBJECTS here
                    categorize_reward_tool,
                    calculate_nyxamine_change_tool
                    # Add other relevant tools if needed
                ]
            )
        except Exception as e:
            logger.error(f"Error creating reward learning agent: {e}")
            return None
    
    def _create_conditioning_agent(self) -> Optional[Agent]:
        """Creates an agent to process conditioning and reinforcement."""
        try:
            return Agent(
                name="Conditioning Agent",
                instructions="""You process reinforcement and conditioning for the Nyx AI system.
                Your role is to analyze conditioning events and determine appropriate reward signals.
                
                For each conditioning event, you'll:
                1. Evaluate the type of conditioning (positive/negative reinforcement/punishment)
                2. Determine the appropriate reward value based on the conditioning parameters
                3. Consider context and history when making these determinations
                4. Generate appropriate reward signals for the system
                
                Focus on creating a balanced reward system that encourages desired behaviors
                while maintaining psychological realism in the reward mechanisms.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(temperature=0.2),
                # Add relevant tools if this agent needs to call them
                tools=[
                    categorize_reward_tool,
                    calculate_nyxamine_change_tool,
                    calculate_submission_value_tool # Add this if the agent calculates submission value
                ]
            )
        except Exception as e:
            logger.error(f"Error creating conditioning agent: {e}")
            return None
    
    async def process_reward_signal(self, reward: RewardSignal) -> Dict[str, Any]:
        """
        Process a reward signal, update nyxamine levels,
        and trigger learning and other effects.
        """
        async with self._main_lock:
            if not self.needs_system:
                 logger.warning("NeedsSystem not available for reward processing.")

            with trace(workflow_name="process_reward", group_id=f"reward_{reward.source}"):
                # 1. Update nyxamine levels
                nyxamine_change = await self._update_nyxamine_level(reward.value)
                
                # 2. Store in history
                history_entry = {
                    "value": reward.value,
                    "source": reward.source,
                    "context": reward.context,
                    "timestamp": reward.timestamp,
                    "nyxamine_level": self.current_nyxamine
                }
                self.reward_history.append(history_entry)
                if len(self.reward_history) > self.max_history_size:
                    self.reward_history = self.reward_history[-self.max_history_size:]
                
                # 3. Update performance tracking
                self.total_reward += reward.value
                if reward.value > 0:
                    self.positive_rewards += 1
                elif reward.value < 0:
                    self.negative_rewards += 1
                
                # 4. Update category statistics
                category = reward.source
                self.category_stats[category]["count"] += 1
                self.category_stats[category]["total"] += reward.value
                if reward.value > 0:
                    self.category_stats[category]["positive"] += 1
                elif reward.value < 0:
                    self.category_stats[category]["negative"] += 1
                
                # 5. Apply effects to other systems
                effects = await self._apply_reward_effects(reward) # Calls update_neurochemical (needs review)

                # 🩸 Estimate depravity
                depravity_score = self._estimate_depravity_level(reward)

                # 🧠 Mood modulation
                mood_state_obj: Optional[MoodState] = None
                mood_boost = 0.0
                try:
                    if self.mood_manager:
                         mood_state_obj = await self.mood_manager.get_current_mood() # Now returns MoodState
                         if mood_state_obj:
                             mood_boost = (
                                 (mood_state_obj.arousal * 0.2) +
                                 (mood_state_obj.control * 0.3 if mood_state_obj.control > 0 else 0)
                             )
                except Exception as mood_err:
                     logger.warning(f"Error getting mood state during reward processing: {mood_err}", exc_info=True)

                amplified_depravity = min(1.0, depravity_score + mood_boost)

                # Add to context for future prediction systems
                reward.context["depravity_score"] = depravity_score
                reward.context["amplified_depravity"] = amplified_depravity
                adjusted_value_for_learning = reward.value + amplified_depravity * 0.1 # Use adjusted value for learning only?

                # 6. Trigger learning if reward is significant
                learning_updates = {}
                if abs(reward.value) >= self.significant_reward_threshold: # Use original value for threshold?
                    # Pass adjusted value to learning? Or original? Decide consistency.
                    # Let's assume learning uses the *originally perceived* reward value before mood amp.
                    reward_for_learning = reward # Pass the original reward object
                    learning_updates = await self._trigger_learning(reward_for_learning)
                    
                action_name = reward.context.get("action")
                
                if action_name:
                    old_decay = self.novelty_decay[action_name]
                    new_decay = max(0.0, old_decay * 0.97)
                    self.novelty_decay[action_name] = new_decay
                if len(self.recent_depravity_levels) >= 10: # Check length before slicing
                    recent_avg = sum(self.recent_depravity_levels[-5:]) / 5 if len(self.recent_depravity_levels) >= 5 else 0
                    previous_avg = sum(self.recent_depravity_levels[-10:-5]) / 5
                    if recent_avg <= previous_avg:
                        for action in self.novelty_decay:
                            self.novelty_decay[action] *= 0.95
                if self.needs_system:
                     await self.needs_system.decrease_need("pleasure_indulgence", 0.3, reason="denied_gratification")


                # 7. Return processing results
                return {
                    "nyxamine_change": nyxamine_change,
                    "current_nyxamine": self.current_nyxamine,
                    "effects": effects,
                    "learning": learning_updates,
                    "depravity_score": depravity_score,
                    "amplified_depravity": amplified_depravity
                }
                

    async def process_submission_reward(
        self,
        submission_type,
        compliance_level,
        user_id,
        novelty: float = 0.5,
        was_initially_resistant: bool = False
    ):
        """Processes rewards for various types of submission."""
        action_key = f"submission::{submission_type}"
        self.novelty_decay[action_key] *= 0.97
        novelty_value = self.novelty_decay[action_key]
        depravity = self.estimate_depravity(submission_type=submission_type)

        # Use the internal logic function directly
        reward_value = compliance_level * await _calculate_submission_value_logic( 
            submission_type=submission_type,
            was_initially_resistant=was_initially_resistant,
            novelty=novelty_value,
            depravity_hint=depravity
        )

        mood_state_obj: Optional[MoodState] = None
        mood_snapshot = {"arousal": 0.5, "control": 0.0, "valence": 0.0} # Defaults
        try:
            if self.mood_manager:
                 mood_state_obj = await self.mood_manager.get_current_mood() # Returns MoodState
                 if mood_state_obj:
                     mood_snapshot = {
                         "arousal": mood_state_obj.arousal,
                         "control": mood_state_obj.control,
                         "valence": mood_state_obj.valence
                     }
        except Exception as mood_err:
            logger.warning(f"Error getting mood state during submission reward: {mood_err}", exc_info=True)
    
        return await self.process_reward_signal(RewardSignal(
            value=reward_value,
            source="user_submission",
            context={
                "submission_type": submission_type,
                "user_id": user_id,
                "action": action_key,
                "novelty": novelty_value,
                "depravity_hint": depravity,
                "mood_snapshot": mood_snapshot,
                "timestamp": datetime.datetime.now().isoformat()
            }
        ))

    def estimate_depravity(self, submission_type: str) -> float:
        """Return a depravity index based on the submission type (0-1 scale)."""
        depravity_scale = {
            "verbal": 0.1,
            "honorific": 0.2,
            "behavioral": 0.3,
            "task": 0.4,
            "service": 0.5,
            "ritual": 0.6,
            "degradation": 0.75,
            "humiliation": 0.85,
            "pain_simulation": 0.9,
            "psychological": 0.95,
            "ownership": 1.0
        }
        return depravity_scale.get(submission_type, 0.5)

    def _estimate_depravity_level(self, reward: RewardSignal) -> float:
        """Estimate how depraved this reward is based on source/type."""
        source = reward.source.lower()
        context = reward.context or {}
        
        depravity_keywords = {
            "humiliation": 0.9,
            "degradation": 0.8,
            "pain_simulation": 0.85,
            "sadistic": 0.9,
            "psychological": 0.95,
            "ownership": 1.0,
            "service": 0.6,
            "ritual": 0.5,
            "behavioral": 0.4
        }
    
        # Try to extract from submission_type or fallback to source
        sub_type = context.get("submission_type", "").lower()
        depravity_score = depravity_keywords.get(sub_type, depravity_keywords.get(source, 0.3))
    
        # Store recent depravity
        self.recent_depravity_levels.append(depravity_score)
        if len(self.recent_depravity_levels) > self.max_depravity_history:
            self.recent_depravity_levels = self.recent_depravity_levels[-self.max_depravity_history:]
        
        return depravity_score
        
    async def trigger_post_gratification_response(self, ctx=None, intensity: float = 1.0, gratification_type: str = "general"):
        """Trigger post-gratification, potentially varying effects based on type."""
        # Create context if needed
        if ctx is None and hasattr(self.emotional_core, 'context'):
            ctx = RunContextWrapper(context=self.emotional_core.context)
            
        serenity_change = intensity * 0.8
        testoryx_reduction = -0.6
        nyxamine_reduction = -0.5
        seranix_boost = 0.6
        oxynixin_boost = 0.3 # Default bonding boost
    
        if gratification_type == "dominance_hard":
            # Colder satisfaction: less bonding boost, maybe sharper drive drop initially
            oxynixin_boost = 0.1
            testoryx_reduction = -0.8 # Stronger reduction in dominance drive temporarily
            seranix_boost = 0.7 # Higher boost to 'calm satisfaction'
            
        await self.update_hormone(ctx, "serenity_boost", serenity_change, source=f"{gratification_type}_gratification")
        await self.update_hormone(ctx, "testoryx", testoryx_reduction, source=f"{gratification_type}_refractory")
        await self.update_neurochemical("nyxamine", nyxamine_reduction, source=f"{gratification_type}_refractory")
        await self.update_neurochemical("seranix", seranix_boost, source=f"{gratification_type}_satisfaction")
        await self.update_neurochemical("oxynixin", oxynixin_boost, source=f"{gratification_type}_aftermath")

    async def process_conditioning_reward(self, conditioning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process rewards generated from conditioning events
        
        Args:
            conditioning_result: Result from conditioning process
            
        Returns:
            Processing results
        """
        with trace(workflow_name="process_conditioning", group_id="conditioning"):
            try:
                # 🧠 Use conditioning agent to evaluate context
                result = await Runner.run(
                    self.conditioning_agent,
                    json.dumps(conditioning_result),
                    run_config=RunConfig(
                        workflow_name="ConditioningReward",
                        trace_metadata={"event_type": conditioning_result.get("type", "unknown")}
                    )
                )
                analysis = result.final_output
    
                # 🔍 Extract core data
                association_key = conditioning_result.get("association_key", "unknown")
                association_type = conditioning_result.get("type", "unknown")
                is_reinforcement = conditioning_result.get("is_reinforcement", True)
                is_positive = conditioning_result.get("is_positive", True)
                intensity = min(1.0, max(0.0, conditioning_result.get("new_strength", 0.5)))
    
                # 💥 Calculate reward value
                if association_type == "new_association":
                    reward_value = 0.3 + (intensity * 0.3)
                elif association_type in ("reinforcement", "update"):
                    if is_reinforcement:
                        reward_value = (0.4 + intensity * 0.4) if is_positive else (0.3 + intensity * 0.3)
                    else:
                        reward_value = (-0.3 - intensity * 0.3) if is_positive else (-0.2 - intensity * 0.2)
                else:
                    reward_value = 0.1  # fallback
    
                # 🔮 Novelty tracking
                action_key = f"conditioning::{association_key}"
                self.novelty_decay[action_key] *= 0.97
                novelty_value = self.novelty_decay[action_key]
    
                # 🧠 Mood snapshot
                mood = await self.mood_manager.get_current_mood() if self.mood_manager else None
                mood_snapshot = {
                    "arousal": getattr(mood, "arousal", 0.5),
                    "control": getattr(mood, "control", 0.0),
                    "valence": getattr(mood, "valence", 0.0)
                }
    
                # 🧾 Build full reward signal
                reward_signal = RewardSignal(
                    value=reward_value,
                    source="conditioning_system",
                    context={
                        "association_key": association_key,
                        "association_type": association_type,
                        "is_reinforcement": is_reinforcement,
                        "is_positive": is_positive,
                        "intensity": intensity,
                        "agent_analysis": analysis,
                        "action": action_key,
                        "novelty": novelty_value,
                        "mood_snapshot": mood_snapshot,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
    
                return await self.process_reward_signal(reward_signal)
    
            except Exception as e:
                logger.error(f"Error processing conditioning reward: {e}")
                
                # Fallback version
                return await self.process_reward_signal(RewardSignal(
                    value=0.1,
                    source="conditioning_system_fallback",
                    context={
                        "error": str(e),
                        "original_data": conditioning_result,
                        "action": "conditioning::fallback",
                        "novelty": 0.5,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ))

    async def _update_nyxamine_level(self, reward_value: float) -> float:
        """Update nyxamine level based on reward value and time decay"""
        async with self._nyxamine_lock:
            current_time = time.time()
            elapsed_seconds = current_time - self.last_update_time
            if elapsed_seconds > 0:
                decay_amount = self.nyxamine_decay_rate * elapsed_seconds
                if self.current_nyxamine > self.baseline_nyxamine:
                    self.current_nyxamine = max(self.baseline_nyxamine, self.current_nyxamine - decay_amount)
                elif self.current_nyxamine < self.baseline_nyxamine:
                    self.current_nyxamine = min(self.baseline_nyxamine, self.current_nyxamine + decay_amount)
                self.last_update_time = current_time
    
            # Apply reward effect using the LOGIC function directly
            result = await _calculate_nyxamine_change_logic( 
                reward_value=reward_value,
                current_nyxamine=self.current_nyxamine,
                baseline_nyxamine=self.baseline_nyxamine
            )
    
            self.current_nyxamine = result["new_nyxamine"]
            return result["nyxamine_change"]

    # --- update_neurochemical method (previously refactored, keep the fixed version) ---
    async def update_neurochemical(self, chemical: str, value: float, source: str = "system") -> Dict[str, Any]:
        """
        Wrapper to handle updating neurochemicals by calling the appropriate
        internal logic function on EmotionalCore or NeurochemicalTools.
        """
        if self.emotional_core is None:
            logger.warning(f"No emotional core available for updating neurochemical {chemical}")
            return {"success": False, "message": "Emotional core not available", "chemical": chemical}

        try:
            update_func = None
            target_context = None
            # Check NeurochemicalTools first
            if hasattr(self.emotional_core, 'neurochemical_tools'):
                tools_instance = self.emotional_core.neurochemical_tools
                # Look for the *internal logic* function
                if hasattr(tools_instance, '_update_neurochemical_impl'): # <<< Use the correct internal name
                    update_func = tools_instance._update_neurochemical_impl
                    target_context = getattr(self.emotional_core, 'context', None)
                    if not isinstance(target_context, EmotionalContext):
                         logger.error("Cannot call neurochemical logic: EmotionalCore context missing or wrong type.")
                         return {"success": False, "error": "Missing or invalid emotional context", "chemical": chemical}
                    logger.debug(f"Found update logic in NeurochemicalTools: _update_neurochemical_impl")
                # Add other fallbacks if necessary...

            # Fallback to EmotionalCore internal logic (if needed)
            elif hasattr(self.emotional_core, '_internal_update_neurochemical_logic'): # Check the core if tools fails
                 update_func = self.emotional_core._internal_update_neurochemical_logic
                 target_context = getattr(self.emotional_core, 'context', None)
                 if not isinstance(target_context, EmotionalContext):
                      logger.error("Cannot call emotional core logic: EmotionalCore context missing or wrong type.")
                      return {"success": False, "error": "Missing or invalid emotional context", "chemical": chemical}
                 logger.debug(f"Found update logic in EmotionalCore: _internal_update_neurochemical_logic")
            
            # Execute the update
            if update_func and target_context is not None:
                logger.debug(f"Calling update logic for {chemical} with RunContextWrapper...")
                ctx_wrapper = RunContextWrapper(context=target_context)
                result = await update_func(ctx_wrapper, chemical=chemical, value=value, source=source)
                if not isinstance(result, dict): result = {"success": True, "raw_result": result}
                elif "success" not in result: result["success"] = True
                result["chemical"] = chemical
                return result
            elif update_func: # Direct call scenario (less likely needed)
                 logger.debug(f"Calling update logic for {chemical} directly...")
                 result = await update_func(chemical, value, source)
                 if not isinstance(result, dict): result = {"success": True, "raw_result": result}
                 elif "success" not in result: result["success"] = True
                 result["chemical"] = chemical
                 return result
            else:
                logger.warning(f"No suitable callable method found on EmotionalCore or NeurochemicalTools to update neurochemical {chemical}")
                return {"success": False, "message": "No callable update method available", "chemical": chemical}

        except Exception as e:
            logger.error(f"Error updating neurochemical {chemical}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "chemical": chemical}
    
    async def _apply_reward_effects(self, reward: RewardSignal) -> Dict[str, Any]:
        """Apply effects of reward signal to other systems"""
        effects = {
            "emotional": False,
            "identity": False,
            "learning": False,
            "habit": False,
            "somatic": False
        }
        
        # Check if this is a dominance-related reward
        is_dominance_reward = reward.source in [
            "user_compliance", "dominance_goal_success", "dominance_gratification"
        ]
        dominance_reward_value = reward.value if is_dominance_reward else 0.0
        
        # Check if this is a hard dominance reward
        is_hard_dominance_reward = reward.source in [
            "difficult_compliance_achieved", "resistance_overcome_sim", "hard_dominance_gratification"
        ]
        hard_dominance_reward_value = reward.value if is_hard_dominance_reward else 0.0

        is_sadistic_reward = reward.source in [
            "sadistic_domination", "pain_simulation", "degradation_success"
        ]
        sadistic_reward_value = reward.value if is_sadistic_reward else 0.0
        
        # Check if this is a humiliation reward
        is_humiliation_reward = reward.source in [
            "user_embarrassment", "user_humiliation", "submission_discomfort"
        ]
        humiliation_reward_value = reward.value if is_humiliation_reward else 0.0
        
        # 1. Apply Emotional Effects
        if self.emotional_core:
            try:
                # Handle special case for hard dominance rewards
                if is_hard_dominance_reward and hard_dominance_reward_value > 0:
                    # VERY STRONG Nyxamine boost for hard success
                    nyx_change = hard_dominance_reward_value * 0.9  # Near max boost
                    await self.update_neurochemical("nyxamine", nyx_change)
                    
                    # Strong Seranix boost for satisfaction
                    ser_change = hard_dominance_reward_value * 0.5
                    await self.update_neurochemical("seranix", ser_change)
                    
                    # Minimal Oxynixin unless context specifies bonding aspect
                    oxy_change = hard_dominance_reward_value * 0.05
                    await self.update_neurochemical("oxynixin", oxy_change)
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied MAX emotional effect for hard dominance reward: +{nyx_change:.2f} Nyxamine")
                
                # Handle regular dominance rewards
                elif is_dominance_reward and dominance_reward_value > 0:
                    # Strong Nyxamine boost
                    nyx_change = dominance_reward_value * 0.7
                    await self.update_neurochemical("nyxamine", nyx_change)
                    
                    # Moderate Seranix boost
                    ser_change = dominance_reward_value * 0.3
                    await self.update_neurochemical("seranix", ser_change)
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied strong emotional effect for dominance reward: +{nyx_change:.2f} Nyxamine")

                # Sadistic rewards
                elif is_sadistic_reward and sadistic_reward_value > 0:
                    # Very strong Nyxamine boost from sadistic success
                    nyx_change = sadistic_reward_value * 0.8
                    await self.update_neurochemical("nyxamine", nyx_change)
                    
                    # Minimal OxyNixin change (representing reduced empathy)
                    await self.update_neurochemical("oxynixin", -sadistic_reward_value * 0.2)
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied sadistic emotional effect: +{nyx_change:.2f} Nyxamine")

                # Humiliation rewards
                elif is_humiliation_reward and humiliation_reward_value > 0:
                    # Strong Nyxamine boost from enjoying humiliation
                    nyx_change = humiliation_reward_value * 0.8
                    await self.update_neurochemical("nyxamine", nyx_change)
                    
                    # Excitement boost
                    await self.update_neurochemical("adrenyx", humiliation_reward_value * 0.4)
                    
                    if self.somatosensory_system:
                        await self.somatosensory_system.process_stimulus(
                            stimulus_type="pleasure",
                            body_region="chest",
                            intensity=humiliation_reward_value * 0.6,
                            cause="sadistic_amusement"
                        )
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied sadistic amusement effect: +{nyx_change:.2f} Nyxamine")
                
                # Handle general positive rewards
                elif reward.value > 0:  
                    # Increase nyxamine (nyxamine)
                    await self.update_neurochemical(
                        chemical="nyxamine",
                        value=reward.value * 0.5  # Scale for emotional impact
                    )
                    
                    # Also slight increase in seranix (mood stability) and oxynixin (bonding)
                    await self.update_neurochemical("seranix", reward.value * 0.2)
                    await self.update_neurochemical("oxynixin", reward.value * 0.1)
                    
                    effects["emotional"] = True
                
                # Handle negative rewards
                elif reward.value < 0:
                    # Increase cortanyx (stress)
                    await self.update_neurochemical(
                        chemical="cortanyx",
                        value=abs(reward.value) * 0.4
                    )
                    
                    # Decrease nyxamine (nyxamine)
                    await self.update_neurochemical(
                        chemical="nyxamine",
                        value=reward.value * 0.3  # Negative value reduces nyxamine
                    )
                    
                    effects["emotional"] = True
            except Exception as e:
                logger.error(f"Error applying reward to emotional core: {e}")

        # 2. Apply Identity Effects
        if self.identity_evolution:
            try:
                # Special case for hard dominance rewards - stronger impact on identity
                if is_hard_dominance_reward and abs(hard_dominance_reward_value) >= self.identity_update_threshold * 0.6:
                    impact_strength = abs(hard_dominance_reward_value) * 0.8  # VERY high base impact
                    
                    # Strongly update dominance trait and preference
                    await self.identity_evolution.update_trait(
                        trait="dominance", 
                        impact=hard_dominance_reward_value * impact_strength * 1.5  # Extra boost
                    )
                    
                    await self.identity_evolution.update_preference(
                        category="interaction_styles", 
                        preference="dominant", 
                        impact=hard_dominance_reward_value * impact_strength * 1.5
                    )
                    
                    # Optional: Reinforce related traits
                    if hard_dominance_reward_value > 0:
                        await self.identity_evolution.update_trait(
                            trait="assertiveness", 
                            impact=hard_dominance_reward_value * impact_strength
                        )
                    
                    effects["identity"] = True
                    logger.debug(f"Applied MAX identity update for hard dominance reward")
                
                # Handle regular dominance rewards
                elif is_dominance_reward and abs(dominance_reward_value) >= self.identity_update_threshold * 0.8:
                    # Define dominance-related traits and preferences
                    trait = "dominance"  # Assuming 'dominance' trait exists
                    preference_category = "interaction_styles"
                    preference_name = "dominant"  # Assuming this preference exists
                    
                    # Calculate impact strength - make it higher for dominance rewards
                    impact_strength = abs(dominance_reward_value) * 0.6  # Higher base impact
                    
                    # Update trait
                    await self.identity_evolution.update_trait(
                        trait=trait, 
                        impact=dominance_reward_value * impact_strength  # impact sign matches reward
                    )
                    
                    # Update preference
                    await self.identity_evolution.update_preference(
                        category=preference_category,
                        preference=preference_name,
                        impact=dominance_reward_value * impact_strength
                    )
                    
                    effects["identity"] = True
                    logger.debug(f"Applied strong identity update for dominance reward")
                
                # General identity updates for significant rewards
                elif abs(reward.value) >= self.identity_update_threshold:
                    # Extract context for tailored identity updates
                    scenario_type = reward.context.get("scenario_type", "general")
                    interaction_type = reward.context.get("interaction_type", "general")
                    
                    # Base impact strength on reward value (abs for magnitude)
                    base_impact_strength = abs(reward.value) * 0.4  # Base scaling factor for identity impact
                    
                    # Update preferences for scenario and interaction types
                    if hasattr(self.identity_evolution, "update_preference"):
                        # Update scenario preference
                        await self.identity_evolution.update_preference(
                            category="scenario_types",
                            preference=scenario_type,
                            impact=reward.value * base_impact_strength
                        )
                        
                        # Update interaction style preference
                        await self.identity_evolution.update_preference(
                            category="interaction_styles",
                            preference=interaction_type,
                            impact=reward.value * base_impact_strength
                        )
                    
                    # Update traits for very strong rewards
                    if abs(reward.value) > 0.9 and hasattr(self.identity_evolution, "update_trait"):
                        trait_impact_strength = abs(reward.value) * 0.1  # Smaller impact for traits
                        
                        # Select traits to update based on reward sign
                        traits_to_update = []
                        if reward.value > 0:
                            traits_to_update = ["curiosity", "adaptability", "confidence"]
                        else:
                            traits_to_update = ["cautiousness", "analytical", "patience"]
                        
                        # Apply trait updates
                        for trait in traits_to_update:
                            impact_sign = 1 if reward.value > 0 else -1
                            await self.identity_evolution.update_trait(
                                trait=trait,
                                impact=impact_sign * trait_impact_strength
                            )
                    
                    effects["identity"] = True
                    logger.debug(f"Applied general identity updates for significant reward: {reward.value}")
            except Exception as e:
                logger.error(f"Error applying reward to identity: {e}")

        # 3. Apply Somatic Effects 
        if self.somatosensory_system:
            try:
                # Special case for dominance rewards
                if is_dominance_reward and dominance_reward_value >= 0.7:
                    # Simulate a wave of warmth/tingling (satisfaction/power)
                    intensity = dominance_reward_value * 0.5
                    await self.somatosensory_system.process_stimulus(
                        stimulus_type="tingling", 
                        body_region="spine", 
                        intensity=intensity, 
                        cause="dominance_satisfaction"
                    )
                    
                    await self.somatosensory_system.process_stimulus(
                        stimulus_type="temperature", 
                        body_region="chest", 
                        intensity=0.55 + intensity * 0.1,  # Slight warmth
                        cause="dominance_satisfaction"
                    )
                    
                    effects["somatic"] = "dominance_satisfaction"
                    logger.debug(f"Applied dominance satisfaction sensation (intensity {intensity:.2f})")
                
                # General significant reward somatic effects
                elif abs(reward.value) >= self.significant_reward_threshold:
                    # Determine target region from context if possible, else default
                    body_region = reward.context.get("body_region", "core")
                    
                    # For positive rewards
                    if reward.value > 0:  
                        intensity = reward.value * 0.4  # Scale intensity
                        await self.somatosensory_system.process_stimulus(
                            stimulus_type="pleasure",
                            body_region=body_region,
                            intensity=intensity,
                            cause=f"Strong reward ({reward.source})"
                        )
                        effects["somatic"] = "pleasure_sensation"
                        logger.debug(f"Applied pleasure sensation (intensity {intensity:.2f}) for positive reward")
                    
                    # For negative rewards
                    elif reward.value < 0:  
                        intensity = abs(reward.value) * 0.3
                        # Increase general tension instead of specific pain
                        current_tension = self.somatosensory_system.get_body_state().get("tension", 0.0)
                        new_tension = min(1.0, current_tension + intensity * 0.2)
                        await self.somatosensory_system.set_body_state_variable("tension", new_tension)
                        effects["somatic"] = "tension_sensation"
                        logger.debug(f"Applied tension sensation (intensity {intensity:.2f}) for negative reward")
            except Exception as e:
                logger.error(f"Error applying reward to somatosensory system: {e}")
        
        # 4. Apply Habit Formation effects
        # Lower threshold for dominance-related habits to form more quickly
        habit_threshold = self.habit_formation_threshold
        if is_dominance_reward:
            habit_threshold *= 0.7  # 30% lower threshold
        
        if abs(reward.value) >= habit_threshold:
            try:
                # Extract action and state from context
                action = reward.context.get("action")
                state = reward.context.get("state")
                
                if action and isinstance(state, dict):
                    # Create or update habit strength
                    state_key = self._create_state_key(state)
                    
                    # Current habit strength
                    current_strength = self.habits[state_key].get(action, 0.0)
                    
                    # Learning rate increases with reward magnitude
                    habit_learning_rate = 0.3
                    if is_dominance_reward:
                        habit_learning_rate = 0.5  # Faster habit formation for dominance
                    
                    # Update strength (positive rewards strengthen, negative weaken)
                    new_strength = current_strength + (reward.value * habit_learning_rate)
                    new_strength = max(0.0, min(1.0, new_strength))  # Constrain to 0-1
                    
                    self.habits[state_key][action] = new_strength
                    effects["habit"] = True
                    logger.debug(f"Updated habit strength for action '{action}': {current_strength:.2f} -> {new_strength:.2f}")
            except Exception as e:
                logger.error(f"Error in habit formation: {e}")
        
        return effects
    
    async def update_hormone(self, ctx, hormone: str, value: float, source: str = "system"):
        """Wrapper to handle updating hormones in the system"""
        try:
            # Create context if needed
            if ctx is None and hasattr(self.emotional_core, 'context'):
                ctx = RunContextWrapper(context=self.emotional_core.context)
            
            # Try direct access to hormones dictionary if available
            hormone_system = None
            if hasattr(self, 'hormone_system') and self.hormone_system is not None:
                hormone_system = self.hormone_system
            elif hasattr(self.emotional_core, 'hormone_system') and self.emotional_core.hormone_system is not None:
                hormone_system = self.emotional_core.hormone_system
                
            if hormone_system and hasattr(hormone_system, 'hormones') and hormone in hormone_system.hormones:
                # Get pre-update value
                old_value = hormone_system.hormones[hormone]["value"]
                
                # Calculate new value with bounds checking
                new_value = max(0.0, min(1.0, old_value + value))
                hormone_system.hormones[hormone]["value"] = new_value
                
                # Update last_update timestamp
                hormone_system.hormones[hormone]["last_update"] = datetime.datetime.now().isoformat()
                
                # Record significant changes
                if abs(new_value - old_value) > 0.05:
                    hormone_system.hormones[hormone]["evolution_history"].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": value,
                        "source": source
                    })
                    
                    # Limit history size
                    if len(hormone_system.hormones[hormone]["evolution_history"]) > 50:
                        hormone_system.hormones[hormone]["evolution_history"] = hormone_system.hormones[hormone]["evolution_history"][-50:]
                
                # Add to context buffer if available
                if ctx and hasattr(ctx.context, "_add_to_circular_buffer"):
                    ctx.context._add_to_circular_buffer("hormone_updates", {
                        "hormone": hormone,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": value,
                        "source": source,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                logger.debug(f"Updated hormone {hormone} directly: {old_value:.2f} → {new_value:.2f}")
                return True
                
            # Try to find _update_hormone_impl method
            elif hormone_system and hasattr(hormone_system, '_update_hormone_impl'):
                return await hormone_system._update_hormone_impl(ctx, hormone, value, source)
                
            else:
                logger.warning(f"No hormone system available for updating hormone {hormone}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating hormone {hormone}: {e}")
            return False
        
    async def _trigger_learning(self, reward: RewardSignal) -> Dict[str, Any]:
        """Trigger learning processes based on significant reward"""
        learning_results = {
            "reinforcement_learning": False,
            "memory_updates": 0,
            "value_updates": 0
        }
    
        try:
            # Extract state and action from context
            current_state = reward.context.get("state")
            action = reward.context.get("action")
            next_state = reward.context.get("next_state")
            current_novelty = reward.context.get("novelty", 0.5)
    
            if current_state and action:
                # ✍️ Store memory (w/ novelty score)
                memory = RewardMemory(
                    state=current_state,
                    action=action,
                    reward=reward.value,
                    next_state=next_state,
                    timestamp=datetime.datetime.now().isoformat(),
                    source=reward.source,
                    novelty_index=current_novelty
                )
                self.reward_memories.append(memory)
                learning_results["memory_updates"] += 1
                if len(self.reward_memories) > self.max_memories:
                    self.reward_memories = self.reward_memories[-self.max_memories:]
    
                # 💥 Apply novelty adjustment
                adjusted_reward = reward.value
                recent_novelties = [
                    m.novelty_index for m in reversed(self.reward_memories[-10:])
                    if m.action == action and m.novelty_index is not None
                ]
                if recent_novelties:
                    last_novelty = recent_novelties[0]
                    if current_novelty <= last_novelty:
                        penalty = (last_novelty - current_novelty) * 0.3
                        adjusted_reward -= penalty
                        logger.debug(f"🟡 Novelty penalty: -{penalty:.2f} for '{action}' (novelty {current_novelty:.2f} ≤ {last_novelty:.2f})")
                    else:
                        bonus = (current_novelty - last_novelty) * 0.2
                        adjusted_reward += bonus
                        logger.debug(f"🟢 Escalation bonus: +{bonus:.2f} for '{action}' (novelty {current_novelty:.2f} > {last_novelty:.2f})")
    
                # 🧠 Reinforcement update
                state_key = self._create_state_key(current_state)
                if action not in self.action_values[state_key]:
                    self.action_values[state_key][action] = ActionValue(
                        state_key=state_key,
                        action=action
                    )
    
                action_value = self.action_values[state_key][action]
    
                # 📈 Update novelty score into Q-value memory
                old_novelty = action_value.novelty_value
                action_value.novelty_value = (
                    (old_novelty * action_value.update_count + current_novelty)
                    / (action_value.update_count + 1)
                )
    
                # 💡 Q-learning update
                current_q = action_value.value
                if next_state:
                    next_state_key = self._create_state_key(next_state)
                    max_next_q = max(
                        (av.value for av in self.action_values.get(next_state_key, {}).values()),
                        default=0.0
                    )
                    new_q = current_q + self.learning_rate * (
                        adjusted_reward + self.discount_factor * max_next_q - current_q
                    )
                else:
                    new_q = current_q + self.learning_rate * (adjusted_reward - current_q)
    
                action_value.value = new_q
                action_value.update_count += 1
                action_value.last_updated = datetime.datetime.now().isoformat()
                action_value.confidence = min(1.0, action_value.confidence + min(0.1, 0.01 * action_value.update_count))
                learning_results["value_updates"] += 1
    
                # 🔁 Experience replay
                if len(self.reward_memories) > 10:
                    learning_results["reinforcement_learning"] = True
                    await self._experience_replay(5)
    
        except Exception as e:
            logger.error(f"Error in reinforcement learning: {e}")
    
        return learning_results

    
    async def _experience_replay(self, num_samples: int = 5):
        """Learn from randomly sampled past experiences"""
        if len(self.reward_memories) < num_samples:
            return
            
        # Sample random memories
        samples = random.sample(self.reward_memories, num_samples)
        
        for memory in samples:
            # Extract data
            state = memory.state
            action = memory.action
            reward_value = memory.reward
            next_state = memory.next_state
            
            # Create state keys
            state_key = self._create_state_key(state)
            next_state_key = self._create_state_key(next_state) if next_state else None
            
            # Get or create action-value
            if action not in self.action_values[state_key]:
                self.action_values[state_key][action] = ActionValue(
                    state_key=state_key,
                    action=action
                )
            
            action_value = self.action_values[state_key][action]
            current_q = action_value.value
            
            # Update Q-value
            if next_state_key:
                # Get maximum Q-value for next state
                max_next_q = 0.0
                if next_state_key in self.action_values and self.action_values[next_state_key]:
                    max_next_q = max(
                        av.value for av in self.action_values[next_state_key].values()
                    )
                
                # Q-learning update rule
                new_q = current_q + self.learning_rate * (
                    reward_value + self.discount_factor * max_next_q - current_q
                )
            else:
                # Simple update if no next state
                new_q = current_q + self.learning_rate * (reward_value - current_q)
            
            # Update Q-value
            action_value.value = new_q
            action_value.update_count += 1
            action_value.last_updated = datetime.datetime.now().isoformat()
    
    def _create_state_key(self, state: Optional[Dict[str, Any]]) -> str:
        """Create a string key from a state dictionary"""
        if not state:
            return "empty_state"
            
        # Sort keys for consistent ordering
        keys = sorted(state.keys())
        
        # Create key string
        parts = []
        for key in keys:
            value = state[key]
            
            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}:{value}")
            elif isinstance(value, list):
                # For lists, use length
                parts.append(f"{key}:list{len(value)}")
            elif isinstance(value, dict):
                # For dicts, use key count
                parts.append(f"{key}:dict{len(value)}")
            else:
                # For other types, use type name
                parts.append(f"{key}:{type(value).__name__}")
        
        return "|".join(parts)
    
    async def predict_best_action(self,
                                  state: Dict[str, Any],
                                  available_actions: List[str]) -> Dict[str, Any]:
        """
        Predict the best action to take in a given state,
        with mood-modulated novelty bias.
        """
        # Ensure needs system is available
        if not self.needs_system:
            logger.warning("NeedsSystem not available for action prediction.")
            # Handle appropriately, maybe return a default action or raise error

        state_key = self._create_state_key(state)
    
        mood_state_obj: Optional[MoodState] = None
        valence = 0.0
        arousal = 0.5
        control = 0.0
        try:
            if self.mood_manager:
                 mood_state_obj = await self.mood_manager.get_current_mood() # Returns MoodState
                 if mood_state_obj:
                     valence = mood_state_obj.valence
                     arousal = mood_state_obj.arousal
                     control = mood_state_obj.control
        except Exception as mood_err:
            logger.warning(f"Error getting mood state during action prediction: {mood_err}", exc_info=True)
    
        # Adjust novelty weighting based on mood
        novelty_weight = 0.2 + arousal * 0.4  # 0.2 → 0.6
        q_weight = 1.0 - novelty_weight - 0.1  # leave habit at 0.1
        control_boost = 0.1 if control > 0.3 else 0.0  # extra dominance tilt
    
        # Prep data
        q_values = {
            action: self.action_values[state_key][action].value
            for action in self.action_values.get(state_key, {})
        }
        habit_strengths = self.habits.get(state_key, {})
        novelty_map = {
            action: getattr(self.action_values[state_key].get(action, None), "novelty_value", 0.5)
            for action in available_actions
        }
    
        # Score each action
        combined_scores = {}
        for action in available_actions:
            q = q_values.get(action, 0.0)
            novelty = novelty_map.get(action, 0.5)
            habit = habit_strengths.get(action, 0.0)
    
            # Strong novelty bias when Nyx is dominant + aroused
            adjusted_novelty = novelty + control_boost

            # Optional: boost preference for pleasure when needy
            pleasure_drive = self.needs_system.get_needs_state().get("pleasure_indulgence", {}).get("drive_strength", 0.0)
            if pleasure_drive > 0.7 and control > 0.3:
                logger.info(f"[Nyx] High pleasure drive ({pleasure_drive:.2f}) + control ({control:.2f}) — escalating assertiveness.")

            combined_score += pleasure_drive * 0.1
    
            combined_score = (
                q * q_weight +
                adjusted_novelty * novelty_weight +
                habit * 0.1
            )
    
            combined_scores[action] = {
                "combined_score": combined_score,
                "q_value": q,
                "novelty_value": adjusted_novelty,
                "habit_strength": habit
            }
    
        # Exploration?
        avg_confidence = 0.0
        confidence_count = 0
        for action in available_actions:
            if action in self.action_values.get(state_key, {}):
                avg_confidence += self.action_values[state_key][action].confidence
                confidence_count += 1
        if confidence_count > 0:
            avg_confidence /= confidence_count
    
        adjusted_exploration = self.exploration_rate * (1 - avg_confidence * 0.5)
        should_explore = random.random() < adjusted_exploration
    
        if should_explore:
            selected_action = random.choice(available_actions)
            selection_method = "exploration"
            is_exploration = True
        else:
            selected_action = max(combined_scores.items(), key=lambda x: x[1]["combined_score"])[0]
            selection_method = "exploitation"
            is_exploration = False
    
        # Return metadata for later reward processing
        novelty_index = novelty_map.get(selected_action, 0.5)
        q_value = q_values.get(selected_action, 0.0)
        habit = habit_strengths.get(selected_action, 0.0)
        confidence = (q_value * 0.4) + (habit * 0.3) + (avg_confidence * 0.3)

        mood_snapshot_for_return = { # Create dict for return value
             "arousal": arousal,
             "control": control,
             "valence": valence
        }
    
        return {
            "best_action": selected_action,
            "q_value": q_value,
            "habit_strength": habit,
            "novelty_index": novelty_index,
            "confidence": confidence,
            "is_exploration": is_exploration,
            "selection_method": selection_method,
            "mood_snapshot": {
                "arousal": arousal,
                "control": control,
                "valence": valence
            },
            "combined_score": combined_scores.get(selected_action)
        }


    
    async def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reward system"""
        # Calculate success rate
        total_rewards = self.positive_rewards + self.negative_rewards
        success_rate = self.positive_rewards / max(1, total_rewards)
        
        # Calculate category statistics
        category_stats = {}
        for category, stats in self.category_stats.items():
            if stats["count"] > 0:
                category_stats[category] = {
                    "count": stats["count"],
                    "average_value": stats["total"] / stats["count"],
                    "success_rate": stats["positive"] / max(1, stats["positive"] + stats["negative"]),
                    "total_reward": stats["total"]
                }
        
        # Get top performing actions
        top_actions = []
        for state_key, actions in self.action_values.items():
            for action, action_value in actions.items():
                if action_value.update_count >= 3:  # Only consider actions with some data
                    top_actions.append({
                        "state": state_key,
                        "action": action,
                        "value": action_value.value,
                        "confidence": action_value.confidence,
                        "updates": action_value.update_count
                    })
        
        # Sort and get top 5
        top_actions.sort(key=lambda x: x["value"], reverse=True)
        top_actions = top_actions[:5]
        
        return {
            "total_reward": self.total_reward,
            "positive_rewards": self.positive_rewards,
            "negative_rewards": self.negative_rewards,
            "success_rate": success_rate,
            "current_nyxamine": self.current_nyxamine,
            "baseline_nyxamine": self.baseline_nyxamine,
            "reward_memories_count": len(self.reward_memories),
            "learned_state_count": len(self.action_values),
            "learned_action_count": sum(len(actions) for actions in self.action_values.values()),
            "habit_count": sum(len(actions) for actions in self.habits.values()),
            "category_stats": category_stats,
            "top_performing_actions": top_actions,
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            }
        }
    
    async def generate_reward_signal(self, 
                                   context: Dict[str, Any],
                                   outcome: str,
                                   success_level: float = 0.5) -> RewardSignal:
        """
        Generate a reward signal based on context and outcome
        
        Args:
            context: Context information
            outcome: Outcome description (success, failure, neutral)
            success_level: Level of success (0.0-1.0)
            
        Returns:
            Generated reward signal
        """
        # Calculate reward value based on outcome
        reward_value = 0.0
        
        if outcome == "success":
            reward_value = success_level
        elif outcome == "failure":
            reward_value = -success_level
        # else reward_value stays 0.0 for neutral outcomes
        
        # Create reward signal
        return RewardSignal(
            value=reward_value,
            source=context.get("source", "internal_evaluation"),
            context=context,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    async def update_learning_parameters(self, 
                                      learning_rate: Optional[float] = None,
                                      discount_factor: Optional[float] = None,
                                      exploration_rate: Optional[float] = None) -> Dict[str, Any]:
        """Update reinforcement learning parameters"""
        if learning_rate is not None:
            self.learning_rate = max(0.01, min(1.0, learning_rate))
            
        if discount_factor is not None:
            self.discount_factor = max(0.0, min(0.99, discount_factor))
            
        if exploration_rate is not None:
            self.exploration_rate = max(0.0, min(1.0, exploration_rate))
            
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate
        }
    
    async def analyze_reward_patterns(self) -> Dict[str, Any]:
        """
        Use the learning agent to analyze reward patterns and suggest improvements.
        """
        if not self.learning_agent:
            return {"status": "error", "message": "Learning agent not available"}
        
        try:
            # Get statistics and prepare context
            stats = await self.get_reward_statistics()
            
            # Sample recent rewards
            recent_rewards = self.reward_history[-20:] if self.reward_history else []
            
            # Sample action values (top and bottom performers)
            action_values = []
            for state_key, actions in self.action_values.items():
                for action, action_value in actions.items():
                    if action_value.update_count >= 3:  # Only consider actions with some data
                        action_values.append({
                            "state_key": state_key,
                            "action": action,
                            "value": action_value.value,
                            "confidence": action_value.confidence,
                            "updates": action_value.update_count
                        })
            
            # Sort and get top/bottom
            action_values.sort(key=lambda x: x["value"], reverse=True)
            top_actions = action_values[:5]
            bottom_actions = action_values[-5:] if len(action_values) >= 5 else []
            
            # Prepare prompt for learning agent
            context = {
                "statistics": stats,
                "recent_rewards": recent_rewards,
                "top_performing_actions": top_actions,
                "lowest_performing_actions": bottom_actions,
                "current_parameters": {
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "exploration_rate": self.exploration_rate
                }
            }
            
            # Run learning agent with tracing
            with trace(workflow_name="RewardPatternAnalysis", group_id="reward_analysis"):
                result = await Runner.run(
                    self.learning_agent,
                    json.dumps(context),
                    run_config=RunConfig(
                        workflow_name="RewardAnalysis",
                        trace_metadata={"analysis_type": "reward_patterns"}
                    )
                )
                
                # Process analysis result
                analysis = result.final_output
                
                # Apply any recommended parameter changes
                if hasattr(analysis, "learning_params"):
                    recommended_params = analysis.learning_params
                    if "learning_rate" in recommended_params:
                        self.learning_rate = max(0.01, min(1.0, recommended_params["learning_rate"]))
                    if "discount_factor" in recommended_params:
                        self.discount_factor = max(0.0, min(0.99, recommended_params["discount_factor"]))
                    if "exploration_rate" in recommended_params:
                        self.exploration_rate = max(0.0, min(1.0, recommended_params["exploration_rate"]))
                
                # Return the analysis
                return {
                    "status": "success",
                    "analyzed_at": datetime.datetime.now().isoformat(),
                    "patterns": analysis.patterns,
                    "suggestions": analysis.suggestions,
                    "learning_params": analysis.learning_params,
                    "effectiveness": analysis.effectiveness,
                    "insights": analysis.insights,
                    "updated_parameters": {
                        "learning_rate": self.learning_rate,
                        "discount_factor": self.discount_factor,
                        "exploration_rate": self.exploration_rate
                    }
                }
        
        except Exception as e:
            logger.error(f"Error analyzing reward patterns: {e}")
            return {
                "status": "error",
                "message": f"Error in analysis: {str(e)}"
            }

# Create an agent for the reward system
def create_reward_agent() -> Agent:
    """Create an agent for the reward system"""
    return Agent(
        name="Reward System Agent",
        instructions="""You are a specialized agent for the Nyx AI's reward system.
        Your role is to process reward signals, analyze patterns, and provide insights
        into learning effectiveness.
        
        You handle:
        1. Processing reward signals and updating nyxamine levels
        2. Applying reward effects to emotional, identity, and somatic systems
        3. Facilitating reinforcement learning and habit formation
        4. Analyzing reward patterns to improve learning parameters
        
        Focus on creating a balanced reward system that encourages desired behaviors
        while maintaining psychological realism in the reward mechanisms.""",
        tools=[
            # Use the FunctionTool OBJECTS created earlier
            categorize_reward_tool,
            calculate_nyxamine_change_tool,
            calculate_submission_value_tool
            # Add other tools if needed
        ],
        model="gpt-4o"
    )
