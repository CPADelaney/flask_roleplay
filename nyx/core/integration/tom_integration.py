# nyx/core/integration/tom_integration.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, UserInteractionEvent, EmotionalEvent, get_event_bus
from nyx.core.integration.system_context import get_system_context, UserModel

logger = logging.getLogger(__name__)

class UserStateUpdater:
    """
    Integrates Theory of Mind inferences with other modules.
    Provides continuous updates to user state models and broadcasts changes to other modules.
    """
    def __init__(self, theory_of_mind, dominance_system=None, goal_manager=None, imagination_simulator=None):
        self.theory_of_mind = theory_of_mind
        self.dominance_system = dominance_system
        self.goal_manager = goal_manager
        self.imagination_simulator = imagination_simulator
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        
        # Module integration flags
        self._dominance_integration_enabled = dominance_system is not None
        self._goal_integration_enabled = goal_manager is not None
        self._imagination_integration_enabled = imagination_simulator is not None
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.5
        
        # Subscription registrations
        self._subscribed = False
        
        logger.info(f"UserStateUpdater initialized (Dominance: {self._dominance_integration_enabled}, "
                   f"Goal: {self._goal_integration_enabled}, Imagination: {self._imagination_integration_enabled})")
    
    async def start(self):
        """Start the updater by subscribing to events."""
        if not self._subscribed:
            self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
            self.event_bus.subscribe("emotional_state_change", self._handle_emotional_event)
            self._subscribed = True
            logger.info("UserStateUpdater: Subscribed to events")
    
    async def stop(self):
        """Stop the updater by unsubscribing from events."""
        if self._subscribed:
            self.event_bus.unsubscribe("user_interaction", self._handle_user_interaction)
            self.event_bus.unsubscribe("emotional_state_change", self._handle_emotional_event)
            self._subscribed = False
            logger.info("UserStateUpdater: Unsubscribed from events")
    
    async def _handle_user_interaction(self, event: UserInteractionEvent) -> None:
        """
        Handle user interaction events by updating the user model.
        
        Args:
            event: The user interaction event
        """
        user_id = event.data["user_id"]
        content = event.data["content"]
        input_type = event.data["input_type"]
        emotional_analysis = event.data["emotional_analysis"]
        
        logger.debug(f"UserStateUpdater: Processing interaction for user {user_id}")
        
        # Prepare interaction data for ToM update
        interaction_data = {
            "user_input": content,
            "input_type": input_type,
            "emotional_analysis": emotional_analysis
        }
        
        # Update the user model through ToM
        update_result = await self.theory_of_mind.update_user_model(user_id, interaction_data)
        
        if update_result.get("status") == "success":
            # Update the system context's user model
            user_model = self.system_context.get_or_create_user_model(user_id)
            
            # Prepare update fields
            update_fields = {
                "inferred_emotion": update_result.get("inferred_emotion", user_model.inferred_emotion),
                "emotion_confidence": update_result.get("emotion_confidence", user_model.emotion_confidence),
                "valence": update_result.get("valence", user_model.valence),
                "arousal": update_result.get("arousal", user_model.arousal)
            }
            
            # Get inferred goals if available
            if "inferred_goals" in update_result:
                update_fields["inferred_goals"] = update_result["inferred_goals"]
            
            # Update user model
            await user_model.update_state(update_fields)
            
            # Now, integrate with other modules if their integration is enabled
            await self._integrate_with_other_modules(user_id, user_model)
            
            logger.info(f"UserStateUpdater: Updated model for user {user_id} (Emotion: {update_fields['inferred_emotion']})")
    
    async def _handle_emotional_event(self, event: EmotionalEvent) -> None:
        """
        Handle emotional events by adjusting the user model.
        For example, if Nyx expresses a strong emotion, this might influence the user's perception.
        
        Args:
            event: The emotional event
        """
        # Only process if we have an active conversation and a user model
        if not self.system_context.conversation_id or not self.system_context.user_models:
            return
        
        # For simplicity, use the first (or only) user in the context
        user_id = next(iter(self.system_context.user_models.keys()))
        user_model = self.system_context.user_models[user_id]
        
        # Update the user model based on Nyx's emotional state
        # This is a simplification - real implementation would be more nuanced
        emotion = event.data.get("emotion")
        valence = event.data.get("valence")
        
        if emotion and abs(valence) > 0.7:  # Strong emotion
            # Prepare simulation to predict user's reaction
            if self.imagination_simulator and self._imagination_integration_enabled:
                sim_input = {
                    "description": f"How would user {user_id} react to Nyx expressing {emotion}?",
                    "initial_state": {
                        "user_emotion": user_model.inferred_emotion,
                        "user_valence": user_model.valence,
                        "nyx_emotion": emotion,
                        "nyx_valence": valence,
                        "relationship_trust": user_model.perceived_trust
                    },
                    "focus_variables": ["user_reaction", "user_new_emotion", "trust_change"]
                }
                
                # Execute simulation to predict user reaction
                sim_result = await self.imagination_simulator.run_simulation(sim_input)
                
                # Update user model based on prediction if confident
                if sim_result and sim_result.confidence > 0.5:
                    predicted_reaction = sim_result.predicted_outcome
                    logger.info(f"UserStateUpdater: Predicted user reaction: {predicted_reaction}")
                    
                    # This would be more sophisticated in a real implementation
                    # Just a simple example of how imagination simulation could inform ToM
    
    async def _integrate_with_other_modules(self, user_id: str, user_model: UserModel) -> None:
        """
        Integrate user model information with other modules.
        
        Args:
            user_id: User ID
            user_model: User model to integrate
        """
        # 1. Dominance Integration
        if self._dominance_integration_enabled and self.dominance_system:
            await self._update_dominance_approach(user_id, user_model)
        
        # 2. Goal Integration
        if self._goal_integration_enabled and self.goal_manager:
            await self._update_goal_priorities(user_id, user_model)
        
        # 3. Imagination Integration
        if self._imagination_integration_enabled and self.imagination_simulator:
            await self._run_predictive_simulation(user_id, user_model)
    
    async def _update_dominance_approach(self, user_id: str, user_model: UserModel) -> None:
        """
        Update dominance approach based on user model.
        
        Args:
            user_id: User ID
            user_model: User model for the user
        """
        # Only proceed if we have high confidence in our user model
        if user_model.overall_confidence < self.medium_confidence_threshold:
            return
        
        # Evaluate dominance potential
        if hasattr(self.dominance_system, 'evaluate_dominance_target_potential'):
            potential = await self.dominance_system.evaluate_dominance_target_potential(user_id)
            
            # If there's high potential, run a more detailed analysis
            if potential.get("interest_score", 0) > 0.7:
                # Prepare new analysis with ToM insights
                tom_enhanced_analysis = {
                    "submissive_score": max(0, 1.0 - user_model.perceived_dominance),
                    "receptivity": user_model.perceived_receptivity,
                    "current_emotion": user_model.inferred_emotion,
                    "current_valence": user_model.valence,
                    "trust_level": user_model.perceived_trust
                }
                
                # Send to dominance system for strategy selection
                if hasattr(self.dominance_system, 'analyze_user_state_for_dominance'):
                    analysis = await self.dominance_system.analyze_user_state_for_dominance(
                        user_id, tom_enhanced_analysis
                    )
                    
                    logger.info(f"UserStateUpdater: Dominance potential analysis: {analysis}")
                    
                    # If user is ready, select approach
                    if analysis.get("assessment") == "ready" and hasattr(self.dominance_system, 'select_dominance_tactic'):
                        # Use readiness score to select tactic
                        readiness = analysis.get("readiness_score", 0.5)
                        
                        # Determine preferred style based on user model
                        if "verbal_humiliation" in user_model.inferred_beliefs:
                            preferred_style = "psychological"
                        elif user_model.valence < -0.3:  # Negative valence
                            preferred_style = "comfort_control"  # Softer approach if user is negative
                        else:
                            preferred_style = "balanced"
                        
                        tactic = await self.dominance_system.select_dominance_tactic(
                            readiness, preferred_style
                        )
                        
                        logger.info(f"UserStateUpdater: Selected dominance tactic: {tactic} for user {user_id}")
    
    async def _update_goal_priorities(self, user_id: str, user_model: UserModel) -> None:
        """
        Update goal priorities based on user model.
        
        Args:
            user_id: User ID
            user_model: User model for the user
        """
        if not hasattr(self.goal_manager, 'get_all_goals') or not hasattr(self.goal_manager, 'update_goal_status'):
            return
        
        # Get active goals
        all_goals = await self.goal_manager.get_all_goals(status_filter=["active", "pending"])
        
        for goal in all_goals:
            goal_id = goal.get("id")
            description = goal.get("description", "").lower()
            priority = goal.get("priority", 0.5)
            
            # Adjust goal priorities based on user emotional state
            new_priority = priority
            
            # If user is in negative emotional state, prioritize connection/comfort goals
            if user_model.valence < -0.3 and ("comfort" in description or "connection" in description):
                new_priority = min(1.0, priority + 0.2)
            
            # If user seems curious (high arousal, positive valence), prioritize knowledge/information goals
            if user_model.arousal > 0.7 and user_model.valence > 0.3 and ("knowledge" in description or "information" in description):
                new_priority = min(1.0, priority + 0.15)
            
            # If priority changed, update the goal
            if abs(new_priority - priority) > 0.05:
                logger.info(f"UserStateUpdater: Adjusting goal {goal_id} priority from {priority:.2f} to {new_priority:.2f} based on user state")
                
                # This assumes the goal_manager has a method to update priority
                # The actual implementation may vary
                if hasattr(self.goal_manager, 'update_goal_priority'):
                    await self.goal_manager.update_goal_priority(goal_id, new_priority, "tom_adjustment")
    
    async def _run_predictive_simulation(self, user_id: str, user_model: UserModel) -> None:
        """
        Run a predictive simulation to anticipate user reactions.
        
        Args:
            user_id: User ID
            user_model: User model for the user
        """
        if not self.imagination_simulator or not hasattr(self.imagination_simulator, 'run_simulation'):
            return
        
        # Only run simulations periodically or when emotions change significantly
        should_run = False
        
        # Check if we have a stored previous emotion
        prev_emotion = self.system_context.get_value(f"prev_emotion_{user_id}")
        if prev_emotion != user_model.inferred_emotion:
            should_run = True
            # Store new emotion
            self.system_context.set_value(f"prev_emotion_{user_id}", user_model.inferred_emotion)
        
        if not should_run:
            return
        
        # Prepare simulation input
        current_brain_state = {
            "user_emotion": user_model.inferred_emotion,
            "user_valence": user_model.valence,
            "user_arousal": user_model.arousal,
            "trust_level": user_model.perceived_trust,
            "familiarity": user_model.perceived_familiarity,
            "interaction_history": await user_model.get_update_history(5)
        }
        
        # Run hypothetical simulation based on current state
        hypotheticals = [
            "What if we express empathy about their current emotional state?",
            "What if we suggest a change of topic?",
            "What if we ask a probing personal question?",
            "What if we express a complementary emotion?"
        ]
        
        # Choose one hypothetical based on user state
        if user_model.valence < -0.5:  # Negative state
            hypothetical = hypotheticals[0]  # Express empathy
        elif user_model.arousal < 0.3:  # Low arousal
            hypothetical = hypotheticals[1]  # Change topic
        elif user_model.perceived_trust > 0.7:  # High trust
            hypothetical = hypotheticals[2]  # Personal question
        else:
            hypothetical = hypotheticals[3]  # Complementary emotion
        
        try:
            sim_input = await self.imagination_simulator.setup_simulation(
                description=hypothetical,
                current_brain_state=current_brain_state
            )
            
            if sim_input:
                sim_result = await self.imagination_simulator.run_simulation(sim_input)
                
                if sim_result and sim_result.success:
                    logger.info(f"UserStateUpdater: Simulation results for '{hypothetical}': {sim_result.predicted_outcome}")
                    
                    # Store simulation results for action selection
                    self.system_context.set_value(f"sim_result_{user_id}", {
                        "hypothetical": hypothetical,
                        "prediction": sim_result.predicted_outcome,
                        "confidence": sim_result.confidence,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"UserStateUpdater: Error running prediction simulation: {e}")

class TheoryOfMindIntegrator:
    """
    Main entry point for Theory of Mind integration. Sets up connections
    between ToM and other modules and manages the integration lifecycle.
    """
    def __init__(self, nyx_brain):
        self.brain = nyx_brain
        self.user_state_updater = None
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        
        # Integration config
        self.enabled = True
        self.dominance_integration = True
        self.goal_integration = True
        self.imagination_integration = True
        
        logger.info("TheoryOfMindIntegrator initialized")
    
    async def initialize(self):
        """Initialize the integration components."""
        try:
            # Create user state updater
            self.user_state_updater = UserStateUpdater(
                theory_of_mind=self.brain.theory_of_mind,
                dominance_system=self.brain.dominance_system if self.dominance_integration else None,
                goal_manager=self.brain.goal_manager if self.goal_integration else None,
                imagination_simulator=self.brain.imagination_simulator if self.imagination_integration else None
            )
            
            # Start updater
            await self.user_state_updater.start()
            
            logger.info("TheoryOfMindIntegrator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing TheoryOfMindIntegrator: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the integration cleanly."""
        if self.user_state_updater:
            await self.user_state_updater.stop()
        logger.info("TheoryOfMindIntegrator shut down")
    
    def enable_integration(self, module_name: str, enabled: bool = True) -> bool:
        """
        Enable or disable integration with a specific module.
        
        Args:
            module_name: Module to enable/disable ("dominance", "goal", "imagination")
            enabled: Whether to enable or disable
            
        Returns:
            True if successful, False otherwise
        """
        if module_name == "dominance":
            self.dominance_integration = enabled
        elif module_name == "goal":
            self.goal_integration = enabled
        elif module_name == "imagination":
            self.imagination_integration = enabled
        else:
            logger.warning(f"Unknown module name for ToM integration: {module_name}")
            return False
        
        logger.info(f"TheoryOfMindIntegrator: {module_name} integration {'enabled' if enabled else 'disabled'}")
        
        # Reinitialize with new settings
        asyncio.create_task(self.initialize())
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current integration status."""
        return {
            "enabled": self.enabled,
            "dominance_integration": self.dominance_integration,
            "goal_integration": self.goal_integration,
            "imagination_integration": self.imagination_integration,
            "updater_running": self.user_state_updater is not None,
            "timestamp": datetime.datetime.now().isoformat()
        }

# Function to create integrator
def create_tom_integrator(nyx_brain):
    """Create a Theory of Mind integrator for the given brain."""
    return TheoryOfMindIntegrator(nyx_brain)
