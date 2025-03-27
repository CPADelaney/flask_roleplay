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

from agents import Agent, Runner, ModelSettings, trace, function_tool

logger = logging.getLogger(__name__)

class RewardType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class RewardSignal(BaseModel):
    """Schema for dopaminergic reward signal"""
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

class ActionValue(BaseModel):
    """Q-value for a state-action pair."""
    state_key: str
    action: str
    value: float = 0.0
    update_count: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    confidence: float = Field(0.2, ge=0.0, le=1.0)
    
    @property
    def is_reliable(self) -> bool:
        """Whether this action value has enough updates to be considered reliable."""
        return self.update_count >= 3 and self.confidence >= 0.5

class RewardSignalProcessor:
    """
    Processes reward signals and implements reward-based learning.
    Simulates dopaminergic pathways for reinforcement learning.
    """
    
    def __init__(self, emotional_core=None, identity_evolution=None, somatosensory_system=None):
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        self.somatosensory_system = somatosensory_system
        
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
        
        # Dopamine system parameters
        self.baseline_dopamine = 0.5
        self.current_dopamine = 0.5
        self.dopamine_decay_rate = 0.1  # Decay per second
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
        self._dopamine_lock = asyncio.Lock()
        
        # Create reinforcement learning agent
        self.learning_agent = self._create_learning_agent()
        
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
                Your outputs should be structured JSON objects containing insights and recommendations.
                Focus on data-driven insights and concrete, implementable suggestions.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    temperature=0.3,
                    response_format={"type": "json_object"}
                ),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating reward learning agent: {e}")
            return None
    
    async def process_reward_signal(self, reward: RewardSignal) -> Dict[str, Any]:
        """
        Process a reward signal, update dopamine levels,
        and trigger learning and other effects.
        
        Args:
            reward: The reward signal to process
            
        Returns:
            Processing results and effects
        """
        async with self._main_lock:
            # 1. Update dopamine levels
            dopamine_change = await self._update_dopamine_level(reward.value)
            
            # 2. Store in history
            history_entry = {
                "value": reward.value,
                "source": reward.source,
                "context": reward.context,
                "timestamp": reward.timestamp,
                "dopamine_level": self.current_dopamine
            }
            self.reward_history.append(history_entry)
            
            # Trim history if needed
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
            effects = await self._apply_reward_effects(reward)
            
            # 6. Trigger learning if reward is significant
            learning_updates = {}
            if abs(reward.value) >= self.significant_reward_threshold:
                learning_updates = await self._trigger_learning(reward)
            
            # 7. Return processing results
            return {
                "dopamine_change": dopamine_change,
                "current_dopamine": self.current_dopamine,
                "effects": effects,
                "learning": learning_updates
            }
    
    async def _update_dopamine_level(self, reward_value: float) -> float:
        """Update dopamine level based on reward value and time decay"""
        async with self._dopamine_lock:
            # First apply time decay
            current_time = time.time()
            elapsed_seconds = current_time - self.last_update_time
            
            if elapsed_seconds > 0:
                # Calculate decay
                decay_amount = self.dopamine_decay_rate * elapsed_seconds
                
                # Decay toward baseline
                if self.current_dopamine > self.baseline_dopamine:
                    self.current_dopamine = max(
                        self.baseline_dopamine, 
                        self.current_dopamine - decay_amount
                    )
                elif self.current_dopamine < self.baseline_dopamine:
                    self.current_dopamine = min(
                        self.baseline_dopamine,
                        self.current_dopamine + decay_amount
                    )
                
                # Update last update time
                self.last_update_time = current_time
            
            # Then apply reward effect
            old_dopamine = self.current_dopamine
            
            # Scale reward to dopamine effect
            # Positive rewards have stronger effect than negative (asymmetry)
            if reward_value >= 0:
                dopamine_change = reward_value * 0.3  # Scale positive rewards
            else:
                dopamine_change = reward_value * 0.2  # Scale negative rewards
            
            # Update dopamine level
            self.current_dopamine = max(0.0, min(1.0, self.current_dopamine + dopamine_change))
            
            # Return the change in dopamine
            return self.current_dopamine - old_dopamine
    
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
        
        # 1. Apply Emotional Effects
        if self.emotional_core:
            try:
                # Handle special case for hard dominance rewards
                if is_hard_dominance_reward and hard_dominance_reward_value > 0:
                    # VERY STRONG Nyxamine boost for hard success
                    nyx_change = hard_dominance_reward_value * 0.9  # Near max boost
                    await self.emotional_core.update_neurochemical("nyxamine", nyx_change)
                    
                    # Strong Seranix boost for satisfaction
                    ser_change = hard_dominance_reward_value * 0.5
                    await self.emotional_core.update_neurochemical("seranix", ser_change)
                    
                    # Minimal Oxynixin unless context specifies bonding aspect
                    oxy_change = hard_dominance_reward_value * 0.05
                    await self.emotional_core.update_neurochemical("oxynixin", oxy_change)
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied MAX emotional effect for hard dominance reward: +{nyx_change:.2f} Nyxamine")
                
                # Handle regular dominance rewards
                elif is_dominance_reward and dominance_reward_value > 0:
                    # Strong Nyxamine boost
                    nyx_change = dominance_reward_value * 0.7
                    await self.emotional_core.update_neurochemical("nyxamine", nyx_change)
                    
                    # Moderate Seranix boost
                    ser_change = dominance_reward_value * 0.3
                    await self.emotional_core.update_neurochemical("seranix", ser_change)
                    
                    effects["emotional"] = True
                    logger.debug(f"Applied strong emotional effect for dominance reward: +{nyx_change:.2f} Nyxamine")
                
                # Handle general positive rewards
                elif reward.value > 0:  
                    # Increase nyxamine (dopamine)
                    await self.emotional_core.update_neurochemical(
                        chemical="nyxamine",
                        value=reward.value * 0.5  # Scale for emotional impact
                    )
                    
                    # Also slight increase in seranix (mood stability) and oxynixin (bonding)
                    await self.emotional_core.update_neurochemical("seranix", reward.value * 0.2)
                    await self.emotional_core.update_neurochemical("oxynixin", reward.value * 0.1)
                    
                    effects["emotional"] = True
                
                # Handle negative rewards
                elif reward.value < 0:
                    # Increase cortanyx (stress)
                    await self.emotional_core.update_neurochemical(
                        chemical="cortanyx",
                        value=abs(reward.value) * 0.4
                    )
                    
                    # Decrease nyxamine (dopamine)
                    await self.emotional_core.update_neurochemical(
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
            
            if current_state and action:
                # 1. Store reward memory
                memory = RewardMemory(
                    state=current_state,
                    action=action,
                    reward=reward.value,
                    next_state=next_state,
                    timestamp=datetime.datetime.now().isoformat(),
                    source=reward.source
                )
                
                self.reward_memories.append(memory)
                learning_results["memory_updates"] += 1
                
                # Limit memory size
                if len(self.reward_memories) > self.max_memories:
                    self.reward_memories = self.reward_memories[-self.max_memories:]
                
                # 2. Update Q-values (action-value mapping)
                state_key = self._create_state_key(current_state)
                
                # Get or create action-value entry
                if action not in self.action_values[state_key]:
                    self.action_values[state_key][action] = ActionValue(
                        state_key=state_key,
                        action=action
                    )
                
                action_value = self.action_values[state_key][action]
                current_q = action_value.value
                
                # If we have next state, calculate using Q-learning
                if next_state:
                    next_state_key = self._create_state_key(next_state)
                    
                    # Get maximum Q-value for next state
                    max_next_q = 0.0
                    if next_state_key in self.action_values and self.action_values[next_state_key]:
                        max_next_q = max(
                            av.value for av in self.action_values[next_state_key].values()
                        )
                    
                    # Q-learning update rule
                    new_q = current_q + self.learning_rate * (
                        reward.value + self.discount_factor * max_next_q - current_q
                    )
                else:
                    # Simple update if no next state
                    new_q = current_q + self.learning_rate * (reward.value - current_q)
                
                # Update Q-value
                action_value.value = new_q
                action_value.update_count += 1
                action_value.last_updated = datetime.datetime.now().isoformat()
                
                # Update confidence based on number of updates and consistency
                confidence_boost = min(0.1, 0.01 * action_value.update_count)
                action_value.confidence = min(1.0, action_value.confidence + confidence_boost)
                
                learning_results["value_updates"] += 1
                
                # 3. Run experience replay (learn from past experiences)
                if len(self.reward_memories) > 10:
                    # Only if we have enough memories
                    learning_results["reinforcement_learning"] = True
                    await self._experience_replay(5)  # Replay 5 random memories
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
        Predict the best action to take in a given state
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Prediction results with best action and confidence
        """
        # Create state key
        state_key = self._create_state_key(state)
        
        # Get Q-values for this state
        q_values = {}
        if state_key in self.action_values:
            q_values = {
                action: self.action_values[state_key][action].value
                for action in self.action_values[state_key]
            }
        
        # Filter to available actions
        available_q_values = {
            action: q_values.get(action, 0.0) 
            for action in available_actions
        }
        
        # Get habit strengths for this state
        habit_strengths = self.habits.get(state_key, {})
        
        # Check if we should explore or exploit
        should_explore = random.random() < self.exploration_rate
        
        # Adjust exploration rate based on learning progress
        avg_confidence = 0.0
        confidence_count = 0
        for action in available_actions:
            if action in self.action_values.get(state_key, {}):
                avg_confidence += self.action_values[state_key][action].confidence
                confidence_count += 1
        
        if confidence_count > 0:
            avg_confidence /= confidence_count
            # Reduce exploration as confidence grows
            adjusted_exploration_rate = self.exploration_rate * (1 - avg_confidence * 0.5)
            should_explore = random.random() < adjusted_exploration_rate
        
        if should_explore:
            # Exploration: choose randomly
            best_action = random.choice(available_actions)
            is_exploration = True
            selection_method = "exploration"
        else:
            # Exploitation: use weighted combination of Q-values and habits
            combined_values = {}
            for action in available_actions:
                q_value = available_q_values.get(action, 0.0)
                habit_strength = habit_strengths.get(action, 0.0)
                
                # Calculate confidence for this action
                confidence = 0.5  # Default
                if action in self.action_values.get(state_key, {}):
                    confidence = self.action_values[state_key][action].confidence
                
                # Weighted combination: more weight to Q-values when confidence is high
                q_weight = 0.5 + confidence * 0.3  # 0.5 to 0.8 based on confidence
                habit_weight = 1.0 - q_weight
                
                combined_values[action] = (q_value * q_weight) + (habit_strength * habit_weight)
            
            # Choose best action
            if combined_values:
                best_action = max(combined_values.items(), key=lambda x: x[1])[0]
                selection_method = "q_values_and_habits"
            else:
                best_action = random.choice(available_actions)
                selection_method = "random_fallback"
                
            is_exploration = False
        
        # Get confidence scores for the selected action
        q_value = available_q_values.get(best_action, 0.0)
        habit_strength = habit_strengths.get(best_action, 0.0)
        action_confidence = 0.5  # Default
        
        if best_action in self.action_values.get(state_key, {}):
            action_confidence = self.action_values[state_key][best_action].confidence
        
        # Calculate overall confidence
        confidence = (q_value * 0.4) + (habit_strength * 0.3) + (action_confidence * 0.3)
        
        return {
            "best_action": best_action,
            "q_value": q_value,
            "habit_strength": habit_strength,
            "confidence": confidence,
            "is_exploration": is_exploration,
            "selection_method": selection_method,
            "all_q_values": available_q_values,
            "exploration_rate": self.exploration_rate
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
            "current_dopamine": self.current_dopamine,
            "baseline_dopamine": self.baseline_dopamine,
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
            
            # Run learning agent
            with trace(workflow_name="AnalyzeRewardPatterns", group_id="RewardSystem"):
                result = await Runner.run(
                    self.learning_agent,
                    json.dumps(context),
                    run_config={
                        "workflow_name": "RewardAnalysis",
                        "trace_metadata": {"analysis_type": "reward_patterns"}
                    }
                )
                
                # Process and return analysis
                analysis = result.final_output
                
                # Add timestamp
                if isinstance(analysis, dict):
                    analysis["analyzed_at"] = datetime.datetime.now().isoformat()
                    analysis["status"] = "success"
                
                return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing reward patterns: {e}")
            return {
                "status": "error",
                "message": f"Error in analysis: {str(e)}"
            }
