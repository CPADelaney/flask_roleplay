# nyx/core/reward_system.py

import logging
import math
import random
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict

class RewardSignal(BaseModel):
    """Schema for dopaminergic reward signal"""
    value: float = Field(..., description="Reward value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: str = Field(..., description="Source generating the reward (e.g., GoalManager, user_compliance, dominance_goal_success)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context info")
    timestamp: str = Field(..., description="When the reward was generated")

class RewardMemory(BaseModel):
    """Schema for stored reward memory"""
    state: Dict[str, Any] = Field(..., description="State that led to reward")
    action: str = Field(..., description="Action that was taken")
    reward: float = Field(..., description="Reward value received")
    next_state: Optional[Dict[str, Any]] = Field(None, description="Resulting state")
    timestamp: str = Field(..., description="When this memory was created")

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
        self.reward_history = []
        self.max_history_size = 1000
        
        # Reward learning data
        self.reward_memories = []
        self.max_memories = 5000
        
        # Action-value mapping (Q-values)
        self.action_values = defaultdict(lambda: defaultdict(float))  # state -> action -> value
        
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
        
        self.logger = logging.getLogger(__name__)
    
    async def process_reward_signal(self, reward: RewardSignal) -> Dict[str, Any]:
        """
        Process a reward signal, update dopamine levels,
        and trigger learning and other effects.
        
        Args:
            reward: The reward signal to process
            
        Returns:
            Processing results and effects
        """
        # 1. Update dopamine levels
        dopamine_change = await self._update_dopamine_level(reward.value)
        
        # 2. Store in history
        self.reward_history.append({
            "value": reward.value,
            "source": reward.source,
            "context": reward.context,
            "timestamp": reward.timestamp,
            "dopamine_level": self.current_dopamine
        })
        
        # Trim history if needed
        if len(self.reward_history) > self.max_history_size:
            self.reward_history = self.reward_history[-self.max_history_size:]
        
        # 3. Update performance tracking
        self.total_reward += reward.value
        if reward.value > 0:
            self.positive_rewards += 1
        elif reward.value < 0:
            self.negative_rewards += 1
        
        # 4. Apply effects to other systems
        effects = await self._apply_reward_effects(reward)
        
        # 5. Trigger learning if reward is significant
        learning_updates = {}
        if abs(reward.value) >= self.significant_reward_threshold:
            learning_updates = await self._trigger_learning(reward)
        
        # 6. Return processing results
        return {
            "dopamine_change": dopamine_change,
            "current_dopamine": self.current_dopamine,
            "effects": effects,
            "learning": learning_updates
        }
    
    async def _update_dopamine_level(self, reward_value: float) -> float:
        """Update dopamine level based on reward value and time decay"""
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
        is_dominance_reward = reward.source in ["user_compliance", "dominance_goal_success", "dominance_gratification"]
        dominance_reward_value = reward.value if is_dominance_reward else 0.0        

        is_hard_dominance_reward = reward.source in ["difficult_compliance_achieved", "resistance_overcome_sim", "hard_dominance_gratification"]
        hard_dominance_reward_value = reward.value if is_hard_dominance_reward else 0.0
        
        # 1. Emotional Effects
        if self.emotional_core:
            try:
                if is_hard_dominance_reward and hard_dominance_reward_value > 0:
                    # VERY STRONG Nyxamine boost for hard success
                    nyx_change = hard_dominance_reward_value * 0.9 # Near max boost
                    self.emotional_core.update_neurochemical("nyxamine", nyx_change)
                    # Strong Seranix boost for satisfaction
                    ser_change = hard_dominance_reward_value * 0.5
                    self.emotional_core.update_neurochemical("seranix", ser_change)
                    # Minimal Oxynixin unless context specifies bonding aspect
                    oxy_change = hard_dominance_reward_value * 0.05
                    self.emotional_core.update_neurochemical("oxynixin", oxy_change)
                    effects["emotional"] = True
                    logger.debug(f"Applied MAX emotional effect for hard dominance reward: +{nyx_change:.2f} Nyxamine")
    
                    effects["emotional"] = True
                    logger.debug(f"Applied strong emotional effect for dominance reward: +{nyx_change:.2f} Nyxamine")
    
                elif reward.value > 0: # Non-dominance positive reward
                    # Positive reward - increase nyxamine (dopamine)
                    self.emotional_core.update_neurochemical(
                        chemical="nyxamine",
                        value=reward.value * 0.5  # Scale for emotional impact
                    )
                    
                    # Also slight increase in seranix (mood stability) and oxynixin (bonding)
                    self.emotional_core.update_neurochemical("seranix", reward.value * 0.2)
                    self.emotional_core.update_neurochemical("oxynixin", reward.value * 0.1)
                    
                    effects["emotional"] = True
                elif reward.value < 0:
                    # Negative reward - increase cortanyx (stress)
                    self.emotional_core.update_neurochemical(
                        chemical="cortanyx",
                        value=abs(reward.value) * 0.4
                    )
                    
                    # Decrease nyxamine (dopamine)
                    self.emotional_core.update_neurochemical(
                        chemical="nyxamine",
                        value=reward.value * 0.3  # Negative value reduces nyxamine
                    )
                    
                    effects["emotional"] = True
            except Exception as e:
                self.logger.error(f"Error applying reward to emotional core: {e}")

        # 2. Identity Effects (Strong reinforcement of dominance trait/preference)
        if self.identity_evolution and is_hard_dominance_reward and abs(hard_dominance_reward_value) >= self.identity_update_threshold * 0.6: # Lower threshold, bigger impact
            try:
                impact_strength = abs(hard_dominance_reward_value) * 0.8 # VERY high base impact
                # Strongly update dominance trait and preference
                await self.identity_evolution.update_trait(trait="dominance", impact=hard_dominance_reward_value * impact_strength * 1.5) # Extra boost
                await self.identity_evolution.update_preference(category="interaction_styles", preference="dominant", impact=hard_dominance_reward_value * impact_strength * 1.5)
                # Potentially reinforce related traits like 'ruthlessness' or 'intensity' if they exist
                effects["identity"] = True
                logger.debug(f"Applied MAX identity update for hard dominance reward.")
            except Exception as e:
                # Define dominance-related traits and preferences
                trait = "dominance" # Assuming 'dominance' trait exists
                preference_category = "interaction_styles"
                preference_name = "dominant" # Assuming this preference exists
    
                # Calculate impact strength - make it higher for dominance rewards
                impact_strength = abs(dominance_reward_value) * 0.6 # Higher base impact
    
                # Update trait
                await self.identity_evolution.update_trait(trait=trait, impact=dominance_reward_value * impact_strength) # impact sign matches reward
    
                # Update preference
                await self.identity_evolution.update_preference(
                    category=preference_category,
                    preference=preference_name,
                    impact=dominance_reward_value * impact_strength # impact sign matches reward
                )
                effects["identity"] = True
                logger.debug(f"Applied strong identity update for dominance reward.")
            except Exception as e:
                self.logger.error(f"Error applying dominance reward to identity: {e}")
                scenario_type = reward.context.get("scenario_type", "general")
                interaction_type = reward.context.get("interaction_type", "general")
                somatic_type = reward.context.get("stimulus_type") # Check if reward came from sensation

                # Base impact strength on reward value (abs for magnitude)
                base_impact_strength = abs(reward.value) * 0.4 # Base scaling factor for identity impact

                # Get adaptability (requires methods in IdentityEvolutionSystem)
                scenario_adaptability = await self.identity_evolution.get_preference_adaptability(
                    "scenario_types", scenario_type
                )
                interaction_adaptability = await self.identity_evolution.get_preference_adaptability(
                    "interaction_styles", interaction_type
                )

                # Calculate final impact (sign depends on reward.value)
                scenario_impact = reward.value * base_impact_strength * scenario_adaptability
                interaction_impact = reward.value * base_impact_strength * interaction_adaptability

                # Update preferences using 'impact' which considers adaptability internally
                await self.identity_evolution.update_preference(
                    category="scenario_types",
                    preference=scenario_type, # Use 'preference' kwarg
                    impact=scenario_impact
                )
                await self.identity_evolution.update_preference(
                    category="interaction_styles",
                    preference=interaction_type, # Use 'preference' kwarg
                    impact=interaction_impact
                )

                # Update somatic preference if reward originated from sensation
                if somatic_type and somatic_type in ["pain", "pleasure", "temperature", "pressure"]:
                     somatic_adaptability = await self.identity_evolution.get_preference_adaptability(
                         "somatic_preferences", somatic_type # Assumes this category exists
                     )
                     somatic_impact = reward.value * base_impact_strength * somatic_adaptability
                     await self.identity_evolution.update_preference(
                         category="somatic_preferences",
                         preference=somatic_type,
                         impact=somatic_impact
                     )

                # Update traits for very strong rewards (consider trait stability)
                if abs(reward.value) > 0.9:
                    trait_impact_strength = abs(reward.value) * 0.1 # Smaller impact for traits

                    traits_to_update = []
                    if reward.value > 0:
                        traits_to_update = ["curiosity", "adaptability", "confidence"]
                    else:
                        traits_to_update = ["cautiousness", "analytical", "patience"]

                    for trait in traits_to_update:
                         trait_stability = await self.identity_evolution.get_trait_stability(trait)
                         # Apply impact considering stability (handled within update_trait)
                         await self.identity_evolution.update_trait(
                             trait=trait,
                             impact=(reward.value / abs(reward.value)) * trait_impact_strength # Keep sign, use strength
                         )

                effects["identity"] = True
            except Exception as e:
                self.logger.error(f"Error applying reward to identity: {e}")

        # 3. Effect on Somatic System (Trigger 'thrill' or 'satisfaction' sensation)
        if self.somatosensory_system and is_dominance_reward and dominance_reward_value >= 0.7:
             try:
                 if dominance_reward_value > 0: # Successful dominance
                     # Simulate a wave of warmth/tingling (satisfaction/power)
                     intensity = dominance_reward_value * 0.5
                     await self.somatosensory_system.process_stimulus(
                         stimulus_type="tingling", body_region="spine", intensity=intensity, cause="dominance_satisfaction"
                     )
                     await self.somatosensory_system.process_stimulus(
                         stimulus_type="temperature", body_region="chest", intensity=0.55 + intensity * 0.1, cause="dominance_satisfaction" # Slight warmth
                     )
                     effects["somatic"] = "satisfaction_sensation"
             except Exception as e:
                 self.logger.error(f"Error applying dominance reward to somatosensory system: {e}")
                # Determine target region from context if possible, else default
                body_region = reward.context.get("body_region", "skin")
                if body_region not in self.somatosensory_system.body_regions:
                    body_region = "skin" # Fallback to skin

                if reward.value > 0: # Strong positive reward -> induce pleasure
                    intensity = reward.value * 0.4 # Scale intensity
                    await self.somatosensory_system.process_stimulus(
                        stimulus_type="pleasure",
                        body_region=body_region,
                        intensity=intensity,
                        cause=f"Strong reward ({reward.source})"
                    )
                    effects["somatic"] = "pleasure_induced"
                    logger.debug(f"Induced pleasure (intensity {intensity:.2f}) in {body_region} due to strong reward.")
               #  Optional: Induce discomfort/tension for strong negative rewards?
                elif reward.value < 0: # Strong negative reward -> induce discomfort
                    intensity = abs(reward.value) * 0.3
                    # Maybe increase general tension instead of specific pain?
                    current_tension = self.somatosensory_system.body_state.get("tension", 0.0)
                    new_tension = min(1.0, current_tension + intensity * 0.2)
                    self.somatosensory_system.body_state["tension"] = new_tension
                    effects["somatic"] = "tension_induced"
                    logger.debug(f"Induced tension ({new_tension:.2f}) due to strong negative reward.")

            except Exception as e:
                self.logger.error(f"Error applying reward to somatosensory system: {e}")
        
        # 4. Habit formation (Dominant actions get strongly reinforced)
        if is_dominance_reward and abs(reward.value) >= self.habit_formation_threshold * 0.7: # Lower threshold for habit
            habit_learning_rate = 0.5 # Faster habit formation for dominance
            effects["habit"] = True
        if abs(reward.value) >= self.habit_formation_threshold:
            try:
                # Extract action and state from context
                action = reward.context.get("action")
                state = reward.context.get("state")
                
                if action and state:
                    # Create or update habit strength
                    state_key = self._create_state_key(state)
                    
                    # Update using simple reinforcement
                    if "habits" not in self.__dict__:
                        self.habits = defaultdict(lambda: defaultdict(float))
                    
                    # Current habit strength
                    current_strength = self.habits[state_key][action]
                    
                    # Update strength (positive rewards strengthen, negative weaken)
                    new_strength = current_strength + (reward.value * 0.3)
                    new_strength = max(0.0, min(1.0, new_strength))  # Constrain to 0-1
                    
                    self.habits[state_key][action] = new_strength
                    
                    effects["habit"] = True
            except Exception as e:
                self.logger.error(f"Error in habit formation: {e}")
        
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
                    timestamp=datetime.datetime.now().isoformat()
                )
                
                self.reward_memories.append(memory)
                learning_results["memory_updates"] += 1
                
                # Limit memory size
                if len(self.reward_memories) > self.max_memories:
                    self.reward_memories = self.reward_memories[-self.max_memories:]
                
                # 2. Update Q-values (action-value mapping)
                state_key = self._create_state_key(current_state)
                
                # Current Q-value for this state-action pair
                current_q = self.action_values[state_key][action]
                
                # If we have next state, calculate using Q-learning
                if next_state:
                    next_state_key = self._create_state_key(next_state)
                    
                    # Get maximum Q-value for next state
                    if self.action_values[next_state_key]:
                        max_next_q = max(self.action_values[next_state_key].values())
                    else:
                        max_next_q = 0
                    
                    # Q-learning update rule
                    new_q = current_q + self.learning_rate * (
                        reward.value + self.discount_factor * max_next_q - current_q
                    )
                else:
                    # Simple update if no next state
                    new_q = current_q + self.learning_rate * (reward.value - current_q)
                
                # Update Q-value
                self.action_values[state_key][action] = new_q
                learning_results["value_updates"] += 1
                
                # 3. Run experience replay (learn from past experiences)
                if len(self.reward_memories) > 10:
                    # Only if we have enough memories
                    learning_results["reinforcement_learning"] = True
                    await self._experience_replay(5)  # Replay 5 random memories
        except Exception as e:
            self.logger.error(f"Error in reinforcement learning: {e}")
        
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
            
            # Current Q-value
            current_q = self.action_values[state_key][action]
            
            # Update Q-value
            if next_state_key:
                # Get maximum Q-value for next state
                if self.action_values[next_state_key]:
                    max_next_q = max(self.action_values[next_state_key].values())
                else:
                    max_next_q = 0
                
                # Q-learning update rule
                new_q = current_q + self.learning_rate * (
                    reward_value + self.discount_factor * max_next_q - current_q
                )
            else:
                # Simple update if no next state
                new_q = current_q + self.learning_rate * (reward_value - current_q)
            
            # Update Q-value
            self.action_values[state_key][action] = new_q
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
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
        q_values = self.action_values[state_key]
        
        # Filter to available actions
        available_q_values = {action: q_values.get(action, 0.0) 
                           for action in available_actions}
        
        # Check if we should explore or exploit
        if random.random() < self.exploration_rate:
            # Exploration: choose randomly
            best_action = random.choice(available_actions)
            is_exploration = True
        else:
            # Exploitation: choose best Q-value
            if available_q_values:
                best_action = max(available_q_values.items(), key=lambda x: x[1])[0]
            else:
                best_action = random.choice(available_actions)
            is_exploration = False
        
        # Also check habits for this state
        habit_strengths = {}
        if hasattr(self, "habits"):
            habit_strengths = self.habits[state_key]
        
        # Calculate confidence based on Q-value and habituation
        q_value = available_q_values.get(best_action, 0.0)
        habit_strength = habit_strengths.get(best_action, 0.0)
        
        # Weight Q-value and habit strength
        confidence = (q_value * 0.7) + (habit_strength * 0.3)
        
        return {
            "best_action": best_action,
            "q_value": q_value,
            "habit_strength": habit_strength,
            "confidence": confidence,
            "is_exploration": is_exploration,
            "all_q_values": available_q_values
        }
    
    async def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reward system"""
        return {
            "total_reward": self.total_reward,
            "positive_rewards": self.positive_rewards,
            "negative_rewards": self.negative_rewards,
            "current_dopamine": self.current_dopamine,
            "baseline_dopamine": self.baseline_dopamine,
            "reward_memories_count": len(self.reward_memories),
            "learned_state_count": len(self.action_values),
            "learned_action_count": sum(len(actions) for actions in self.action_values.values()),
            "habit_count": sum(len(actions) for actions in self.habits.values()) if hasattr(self, "habits") else 0
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
            source="internal_evaluation",
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
