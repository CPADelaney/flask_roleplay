# nyx/core/mode_integration.py

"""
Integration module for connecting the new interaction mode system
with existing Nyx architecture components.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

from nyx.core.context_awareness import ContextAwarenessSystem, InteractionContext
from nyx.core.interaction_mode_manager import InteractionModeManager, InteractionMode
from nyx.core.interaction_goals import get_goals_for_mode

logger = logging.getLogger(__name__)

class ModeIntegrationManager:
    """
    Manages the integration of the interaction mode system with
    other Nyx components. Serves as a central hub for mode-related
    functionality and coordinates between systems.
    """
    
    def __init__(self, nyx_brain=None):
        """
        Initialize the integration manager
        
        Args:
            nyx_brain: Reference to the main NyxBrain instance
        """
        self.brain = nyx_brain
        
        # Core mode components
        self.context_system = None
        self.mode_manager = None
        
        # Connected Nyx components
        self.emotional_core = None
        self.identity_evolution = None
        self.goal_manager = None
        self.reward_system = None
        self.autobiographical_narrative = None
        
        # Initialize if brain reference provided
        if self.brain:
            self.initialize_from_brain()
        
        logger.info("ModeIntegrationManager initialized")
    
    def initialize_from_brain(self) -> bool:
        """
        Initialize components from the brain reference
        
        Returns:
            Success status
        """
        try:
            # Get references to existing components
            self.emotional_core = getattr(self.brain, 'emotional_core', None)
            self.identity_evolution = getattr(self.brain, 'identity_evolution', None)
            self.goal_manager = getattr(self.brain, 'goal_manager', None)
            self.reward_system = getattr(self.brain, 'reward_system', None)
            self.autobiographical_narrative = getattr(self.brain, 'autobiographical_narrative', None)
            
            # Initialize mode components
            self.context_system = ContextAwarenessSystem(emotional_core=self.emotional_core)
            
            self.mode_manager = InteractionModeManager(
                context_system=self.context_system,
                emotional_core=self.emotional_core,
                reward_system=self.reward_system,
                goal_manager=self.goal_manager
            )
            
            # Add references to the brain
            if self.brain:
                setattr(self.brain, 'context_system', self.context_system)
                setattr(self.brain, 'mode_manager', self.mode_manager)
                setattr(self.brain, 'mode_integration', self)
            
            logger.info("ModeIntegrationManager successfully initialized from brain")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing mode integration: {e}")
            return False
    
    async def process_input(self, message: str) -> Dict[str, Any]:
        """
        Process user input through the mode system
        
        Args:
            message: User message
            
        Returns:
            Processing results
        """
        results = {
            "context_processed": False,
            "mode_updated": False,
            "goals_added": False
        }
        
        try:
            # 1. Process through context system
            if self.context_system:
                context_result = await self.context_system.process_message(message)
                results["context_result"] = context_result
                results["context_processed"] = True
                
                # 2. Update interaction mode based on context
                if self.mode_manager:
                    mode_result = await self.mode_manager.update_interaction_mode(context_result)
                    results["mode_result"] = mode_result
                    results["mode_updated"] = True
                    
                    # 3. Add appropriate goals if mode changed
                    if mode_result.get("mode_changed", False) and self.goal_manager:
                        await self._add_mode_specific_goals(mode_result["current_mode"])
                        results["goals_added"] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Error in mode integration processing: {e}")
            return {
                "error": str(e),
                **results
            }
    
    async def _add_mode_specific_goals(self, mode: str) -> List[str]:
        """
        Add goals specific to the current interaction mode
        
        Args:
            mode: Current interaction mode
            
        Returns:
            List of added goal IDs
        """
        if not self.goal_manager:
            return []
            
        try:
            # Get goals for this mode
            mode_goals = get_goals_for_mode(mode)
            
            # Add goals to manager
            added_goal_ids = []
            for goal_template in mode_goals:
                goal_id = await self.goal_manager.add_goal(
                    description=goal_template["description"],
                    priority=goal_template.get("priority", 0.5),
                    source=goal_template.get("source", "mode_integration"),
                    plan=goal_template.get("plan", [])
                )
                
                if goal_id:
                    added_goal_ids.append(goal_id)
            
            logger.info(f"Added {len(added_goal_ids)} goals for mode: {mode}")
            return added_goal_ids
            
        except Exception as e:
            logger.error(f"Error adding mode-specific goals: {e}")
            return []
    
    def get_response_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive guidance for response generation
        
        Returns:
            Guidance parameters for current mode
        """
        if not self.mode_manager:
            return {}
            
        # Get detailed guidance from mode manager
        mode_guidance = self.mode_manager.get_current_mode_guidance()
        
        # Add any additional contextual guidance
        guidance = {
            "mode_guidance": mode_guidance,
            "current_context": self.context_system.get_current_context() if self.context_system else {}
        }
        
        return guidance
    
    async def modify_response_for_mode(self, response_text: str) -> str:
        """
        Modify a response to better fit the current interaction mode
        
        Args:
            response_text: Original response text
            
        Returns:
            Modified response better suited to current mode
        """
        if not self.mode_manager:
            return response_text
            
        # Get current mode
        mode = self.mode_manager.current_mode
        
        # Get mode parameters
        parameters = self.mode_manager.get_mode_parameters(mode)
        conversation_style = self.mode_manager.mode_conversation_styles.get(mode, {})
        vocalization = self.mode_manager.mode_vocalization_patterns.get(mode, {})
        
        # For now, just add mode-appropriate phrases
        # In a real implementation, this would use an LLM to transform the response
        
        try:
            # Simple enhancement with key phrases
            key_phrases = vocalization.get("key_phrases", [])
            if key_phrases and parameters.get("assertiveness", 0.5) > 0.6:
                # Add a mode-specific phrase to the beginning for high-assertiveness modes
                if response_text and not response_text.startswith(tuple(key_phrases)):
                    selected_phrase = key_phrases[0]  # Just use first phrase for simplicity
                    response_text = f"{selected_phrase}. {response_text}"
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error modifying response for mode: {e}")
            return response_text  # Return original if error
    
    async def record_mode_feedback(self, interaction_success: bool, user_feedback: Optional[str] = None) -> None:
        """
        Record feedback about interaction success for learning
        
        Args:
            interaction_success: Whether the interaction was successful
            user_feedback: Optional explicit user feedback
        """
        if not self.mode_manager or not self.reward_system:
            return
            
        # Current mode information
        current_mode = self.mode_manager.current_mode
        
        # Create reward context
        context = {
            "interaction_mode": current_mode.value,
            "user_feedback": user_feedback,
            "interaction_success": interaction_success,
            "mode_parameters": self.mode_manager.get_mode_parameters(current_mode)
        }
        
        # Generate reward value based on success
        reward_value = 0.3 if interaction_success else -0.2
        
        # If explicit feedback provided, adjust reward
        if user_feedback:
            # This would ideally use sentiment analysis
            if "good" in user_feedback.lower() or "like" in user_feedback.lower():
                reward_value = 0.5
            elif "bad" in user_feedback.lower() or "don't like" in user_feedback.lower():
                reward_value = -0.3
        
        # Create and process reward signal
        if self.reward_system and hasattr(self.reward_system, 'process_reward_signal'):
            from nyx.core.reward_system import RewardSignal
            
            reward_signal = RewardSignal(
                value=reward_value,
                source="interaction_mode_feedback",
                context=context
            )
            
            await self.reward_system.process_reward_signal(reward_signal)
    
    async def update_identity_from_mode_usage(self) -> Dict[str, Any]:
        """
        Update identity based on mode usage patterns
        
        Returns:
            Identity update results
        """
        if not self.identity_evolution or not self.mode_manager:
            return {"success": False, "reason": "Required components missing"}
            
        try:
            # Analyze mode history to find patterns
            mode_counts = {}
            for entry in self.mode_manager.mode_switch_history:
                mode = entry.get("new_mode")
                if mode:
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # Find most common mode
            if not mode_counts:
                return {"success": False, "reason": "No mode history available"}
                
            most_common_mode = max(mode_counts.items(), key=lambda x: x[1])
            mode_name, count = most_common_mode
            
            # Calculate proportion of this mode
            total_switches = sum(mode_counts.values())
            proportion = count / total_switches if total_switches > 0 else 0
            
            # Only update if there's a clear preference (>30%)
            if proportion > 0.3:
                # Map mode to trait updates
                trait_updates = {
                    InteractionMode.DOMINANT.value: {
                        "dominance": 0.1,
                        "assertiveness": 0.1
                    },
                    InteractionMode.FRIENDLY.value: {
                        "empathy": 0.1,
                        "humor": 0.1,
                        "warmth": 0.1
                    },
                    InteractionMode.INTELLECTUAL.value: {
                        "intellectualism": 0.1,
                        "analytical": 0.1
                    },
                    InteractionMode.COMPASSIONATE.value: {
                        "empathy": 0.2,
                        "patience": 0.1,
                        "vulnerability": 0.1
                    },
                    InteractionMode.PLAYFUL.value: {
                        "playfulness": 0.15,
                        "humor": 0.15
                    },
                    InteractionMode.CREATIVE.value: {
                        "creativity": 0.15,
                        "openness": 0.1
                    },
                    InteractionMode.PROFESSIONAL.value: {
                        "conscientiousness": 0.1,
                        "analytical": 0.1
                    }
                }
                
                # Get traits to update for this mode
                mode_trait_updates = trait_updates.get(mode_name, {})
                
                # Apply trait updates
                identity_updates = {}
                for trait, impact in mode_trait_updates.items():
                    # Scale impact by proportion
                    scaled_impact = impact * proportion
                    
                    # Update trait if method exists
                    if hasattr(self.identity_evolution, 'update_trait'):
                        update_result = await self.identity_evolution.update_trait(
                            trait=trait,
                            impact=scaled_impact
                        )
                        
                        identity_updates[trait] = update_result
                
                return {
                    "success": True,
                    "most_common_mode": mode_name,
                    "proportion": proportion,
                    "updates": identity_updates
                }
            else:
                return {
                    "success": False,
                    "reason": "No dominant mode preference detected",
                    "mode_counts": mode_counts
                }
                
        except Exception as e:
            logger.error(f"Error updating identity from mode usage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
