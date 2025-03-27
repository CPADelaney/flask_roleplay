# nyx/core/integration/reward_learning_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method
from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

class RewardLearningBridge:
    """
    Integrates reward system with learning, memory, and emotional systems.
    Coordinates reward distribution across modules and ensures 
    reward-based learning drives behavior adaptation.
    """
    
    def __init__(self, 
                reward_system=None,
                goal_manager=None,
                memory_orchestrator=None,
                emotional_core=None,
                action_selector=None):
        """Initialize the reward-learning bridge."""
        self.reward_system = reward_system
        self.goal_manager = goal_manager
        self.memory_orchestrator = memory_orchestrator
        self.emotional_core = emotional_core
        self.action_selector = action_selector
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.memory_significance_multiplier = 5.0  # Convert reward value to memory significance (0-10)
        self.significant_reward_threshold = 0.6  # Threshold for storing memory
        
        # Integration state tracking
        self.reward_source_mappings = {
            "goal_completion": {"system": "goal_manager", "importance": 0.8},
            "user_compliance": {"system": "dominance_system", "importance": 0.9},
            "knowledge_acquisition": {"system": "knowledge_system", "importance": 0.7},
            "user_feedback": {"system": "user_interaction", "importance": 1.0}
        }
        self._subscribed = False
        
        logger.info("RewardLearningBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("goal_completed", self._handle_goal_completed)
                self.event_bus.subscribe("action_completed", self._handle_action_completed)
                self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
                self._subscribed = True
            
            logger.info("RewardLearningBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing RewardLearningBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="RewardLearning")
    async def process_achievement(self, 
                              achievement_type: str,
                              achievement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an achievement and generate appropriate rewards.
        
        Args:
            achievement_type: Type of achievement (goal, interaction, learning, etc.)
            achievement_data: Data about the achievement
            
        Returns:
            Processing results with generated rewards
        """
        if not self.reward_system:
            return {"status": "error", "message": "Reward system not available"}
        
        try:
            # 1. Calculate reward value based on achievement type and data
            reward_value = 0.0
            reward_source = "generic_achievement"
            reward_context = achievement_data.copy()
            
            # Determine value based on achievement type
            if achievement_type == "goal_completion":
                priority = achievement_data.get("priority", 0.5)
                difficulty = achievement_data.get("difficulty", 0.5)
                reward_value = 0.4 + (priority * 0.3) + (difficulty * 0.3)  # 0.4 - 1.0
                reward_source = "goal_completion"
                
            elif achievement_type == "user_compliance":
                intensity = achievement_data.get("intensity", 0.5) 
                resistance = achievement_data.get("resistance_overcome", False)
                reward_value = 0.5 + (intensity * 0.3)
                if resistance:
                    reward_value += 0.2  # Bonus for overcoming resistance
                reward_source = "user_compliance"
                
            elif achievement_type == "knowledge_acquisition":
                significance = achievement_data.get("significance", 0.5)
                novelty = achievement_data.get("novelty", 0.5)
                reward_value = 0.3 + (significance * 0.4) + (novelty * 0.3)
                reward_source = "knowledge_acquisition"
                
            elif achievement_type == "user_feedback":
                rating = achievement_data.get("rating", 0.0)
                explicit = achievement_data.get("explicit", False)
                
                # Map rating to reward (-1.0 to 1.0)
                if isinstance(rating, (int, float)):
                    # Direct value
                    reward_value = max(-1.0, min(1.0, rating))
                elif isinstance(rating, str):
                    # String rating
                    rating_mapping = {
                        "excellent": 1.0, "good": 0.7, "positive": 0.5,
                        "neutral": 0.0,
                        "negative": -0.5, "bad": -0.7, "terrible": -1.0
                    }
                    reward_value = rating_mapping.get(rating.lower(), 0.0)
                
                # Increase magnitude for explicit feedback
                if explicit:
                    reward_value *= 1.5
                    reward_value = max(-1.0, min(1.0, reward_value))
                    
                reward_source = "user_feedback"
            
            else:
                # Generic achievement
                significance = achievement_data.get("significance", 0.5)
                reward_value = significance * 0.7  # Scale to reasonable reward
            
            # 2. Create reward signal
            reward_signal = RewardSignal(
                value=reward_value,
                source=reward_source,
                context=reward_context
            )
            
            # 3. Process the reward
            reward_result = await self.reward_system.process_reward_signal(reward_signal)
            
            # 4. Store memory of significant achievements
            memory_id = None
            if self.memory_orchestrator and abs(reward_value) >= self.significant_reward_threshold:
                # Create memory text
                memory_text = self._create_memory_text(achievement_type, achievement_data, reward_value)
                
                # Calculate significance (0-10 scale)
                significance = min(10, abs(reward_value) * self.memory_significance_multiplier)
                
                # Create tags
                tags = ["reward", achievement_type]
                if reward_value > 0:
                    tags.append("positive_reward")
                elif reward_value < 0:
                    tags.append("negative_reward")
                
                # Add any additional tags from achievement
                if "tags" in achievement_data and isinstance(achievement_data["tags"], list):
                    tags.extend(achievement_data["tags"])
                
                # Add memory
                memory_id = await self.memory_orchestrator.add_memory(
                    memory_text=memory_text,
                    memory_type="experience",
                    significance=int(significance),
                    tags=tags,
                    metadata={
                        "achievement_type": achievement_type,
                        "reward_value": reward_value,
                        "reward_source": reward_source,
                        "timestamp": datetime.datetime.now().isoformat(),
                        **{k: v for k, v in achievement_data.items() if isinstance(v, (str, int, float, bool))}
                    }
                )
            
            return {
                "status": "success",
                "achievement_type": achievement_type,
                "reward_value": reward_value,
                "reward_source": reward_source,
                "memory_created": memory_id is not None,
                "memory_id": memory_id,
                "reward_effects": reward_result.get("effects", {})
            }
        except Exception as e:
            logger.error(f"Error processing achievement: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="RewardLearning")
    async def train_action_preferences(self, 
                                  session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train action preferences based on session outcomes.
        
        Args:
            session_data: Data about interaction session
            
        Returns:
            Training results
        """
        if not self.reward_system or not self.action_selector:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Extract relevant information from session
            actions_taken = session_data.get("actions_taken", [])
            session_outcome = session_data.get("outcome", {})
            session_context = session_data.get("context", {})
            
            # Skip if no actions
            if not actions_taken:
                return {"status": "skipped", "reason": "No actions to train on"}
            
            # 2. Calculate session reward
            session_reward = session_outcome.get("reward_value", 0.0)
            
            if not session_reward and "success" in session_outcome:
                # Derive reward from success boolean
                success = session_outcome["success"]
                session_reward = 0.7 if success else -0.3
            
            # 3. Train each action with reward
            training_results = []
            
            for action_info in actions_taken:
                # Extract action data
                action_id = action_info.get("action_id")
                action_type = action_info.get("action_type")
                action_context = action_info.get("context", {})
                
                if not action_id or not action_type:
                    continue
                
                # Adjust reward for this action based on its contribution
                action_reward = session_reward
                if "reward_weight" in action_info:
                    action_reward *= action_info["reward_weight"]
                
                # Combine session context with action context
                combined_context = {**session_context, **action_context}
                
                # Create state representation for learning
                state = {
                    key: value for key, value in combined_context.items() 
                    if isinstance(value, (str, int, float, bool))
                }
                
                # Create reward signal for learning
                reward_signal = RewardSignal(
                    value=action_reward,
                    source="action_training",
                    context={
                        "action": action_type,
                        "action_id": action_id,
                        "state": state,
                        "session_id": session_data.get("session_id")
                    }
                )
                
                # Process reward for learning
                reward_result = await self.reward_system.process_reward_signal(reward_signal)
                
                training_results.append({
                    "action_id": action_id,
                    "action_type": action_type,
                    "reward": action_reward,
                    "learning_updates": reward_result.get("learning", {})
                })
            
            return {
                "status": "success",
                "session_reward": session_reward,
                "actions_trained": len(training_results),
                "training_results": training_results
            }
        except Exception as e:
            logger.error(f"Error training action preferences: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="RewardLearning")
    async def predict_action_rewards(self, 
                                 state: Dict[str, Any],
                                 available_actions: List[str]) -> Dict[str, Any]:
        """
        Predict rewards for available actions in a state.
        
        Args:
            state: Current state
            available_actions: Available actions
            
        Returns:
            Predicted rewards for actions
        """
        if not self.reward_system:
            return {"status": "error", "message": "Reward system not available"}
        
        try:
            # Use reward system to predict best action
            prediction = await self.reward_system.predict_best_action(
                state=state,
                available_actions=available_actions
            )
            
            return {
                "status": "success",
                "best_action": prediction.get("best_action"),
                "predicted_rewards": prediction.get("all_q_values", {}),
                "confidence": prediction.get("confidence", 0.5),
                "is_exploration": prediction.get("is_exploration", False)
            }
        except Exception as e:
            logger.error(f"Error predicting action rewards: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_memory_text(self, achievement_type: str, achievement_data: Dict[str, Any], reward_value: float) -> str:
        """Create memory text for an achievement."""
        # Generic start
        memory = f"Received {'positive' if reward_value > 0 else 'negative'} reward ({reward_value:.2f}) for "
        
        # Specific text based on achievement type
        if achievement_type == "goal_completion":
            goal_desc = achievement_data.get("description", "completing a goal")
            memory += f"completing goal: {goal_desc}"
            
        elif achievement_type == "user_compliance":
            action = achievement_data.get("action", "a user compliance action")
            user_id = achievement_data.get("user_id", "a user")
            memory += f"successful {action} with user {user_id}"
            
        elif achievement_type == "knowledge_acquisition":
            topic = achievement_data.get("topic", "new information")
            memory += f"acquiring knowledge about {topic}"
            
        elif achievement_type == "user_feedback":
            feedback = achievement_data.get("feedback", "feedback")
            user_id = achievement_data.get("user_id", "a user")
            memory += f"receiving {feedback} feedback from user {user_id}"
            
        else:
            # Generic achievement
            memory += f"achievement of type {achievement_type}"
        
        return memory
    
    async def _handle_goal_completed(self, event: Event) -> None:
        """
        Handle goal completed events.
        
        Args:
            event: Goal completed event
        """
        try:
            # Extract data
            goal_id = event.data.get("goal_id")
            
            if not goal_id or not self.goal_manager:
                return
            
            # Get goal details
            goal_status = await self.goal_manager.get_goal_status(goal_id)
            
            if not goal_status:
                return
            
            # Create achievement data
            achievement_data = {
                "goal_id": goal_id,
                "description": goal_status.get("description", "Unknown goal"),
                "priority": goal_status.get("priority", 0.5),
                "difficulty": goal_status.get("difficulty", 0.5),
                "steps_completed": goal_status.get("steps_completed", 0),
                "tags": goal_status.get("tags", [])
            }
            
            # Process achievement
            asyncio.create_task(
                self.process_achievement("goal_completion", achievement_data)
            )
        except Exception as e:
            logger.error(f"Error handling goal completed: {e}")
    
    async def _handle_action_completed(self, event: Event) -> None:
        """
        Handle action completed events.
        
        Args:
            event: Action completed event
        """
        try:
            # Extract data
            action_id = event.data.get("action_id")
            action_type = event.data.get("action_type")
            success = event.data.get("success", False)
            
            if not action_id or not action_type:
                return
            
            # Special handling for dominance-related actions
            if "dominance" in action_type.lower() or "command" in action_type.lower():
                # Extract additional data if available
                user_id = event.data.get("user_id", "unknown")
                intensity = event.data.get("intensity", 0.5)
                resistance = event.data.get("resistance_overcome", False)
                
                # Create achievement data
                achievement_data = {
                    "action_id": action_id,
                    "action_type": action_type,
                    "user_id": user_id,
                    "intensity": intensity,
                    "resistance_overcome": resistance,
                    "success": success
                }
                
                if success:
                    # Process as user compliance achievement
                    asyncio.create_task(
                        self.process_achievement("user_compliance", achievement_data)
                    )
        except Exception as e:
            logger.error(f"Error handling action completed: {e}")
    
    async def _handle_user_feedback(self, event: Event) -> None:
        """
        Handle user feedback events.
        
        Args:
            event: User feedback event
        """
        try:
            # Extract data
            user_id = event.data.get("user_id")
            feedback_type = event.data.get("type")
            rating = event.data.get("rating")
            explicit = event.data.get("explicit", False)
            
            if not user_id or not feedback_type:
                return
            
            # Create achievement data
            achievement_data = {
                "user_id": user_id,
                "type": feedback_type,
                "rating": rating,
                "explicit": explicit,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Process achievement
            asyncio.create_task(
                self.process_achievement("user_feedback", achievement_data)
            )
        except Exception as e:
            logger.error(f"Error handling user feedback: {e}")

# Function to create the bridge
def create_reward_learning_bridge(nyx_brain):
    """Create a reward-learning bridge for the given brain."""
    return RewardLearningBridge(
        reward_system=nyx_brain.reward_system if hasattr(nyx_brain, "reward_system") else None,
        goal_manager=nyx_brain.goal_manager if hasattr(nyx_brain, "goal_manager") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        action_selector=nyx_brain.action_selector if hasattr(nyx_brain, "action_selector") else None
    )
