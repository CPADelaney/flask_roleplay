# nyx/core/integration/dominance_reward_identity_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method
from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

class DominanceRewardIdentityBridge:
    """
    Integrates dominance expression with reward system and identity evolution.
    Creates a feedback loop where dominance successes reinforce identity and
    generate appropriate rewards.
    """
    
    def __init__(self, 
                dominance_system=None,
                reward_system=None,
                identity_evolution=None,
                relationship_manager=None):
        """Initialize the bridge."""
        self.dominance_system = dominance_system
        self.reward_system = reward_system
        self.identity_evolution = identity_evolution
        self.relationship_manager = relationship_manager
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Reward configuration
        self.reward_values = {
            "dominance_compliance": 0.4,  # User complied with dominance directive
            "dominance_escalation_success": 0.7,  # Successfully escalated dominance level
            "dominance_resistance_overcome": 1.0,  # Overcame resistance
            "submission_expressed": 0.3,  # User expressed submission
            "challenging_boundary_success": 0.85,  # Successfully challenged boundary
            "dominance_failure": -0.6,  # Failed dominance attempt
            "resistance_failure": -0.7,  # Failed to overcome resistance
            "boundary_failure": -0.8,  # Boundary challenge failed
        }
        
        # Identity integration strength
        self.identity_impact_factor = 0.7  # How strongly dominance successes impact identity
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("DominanceRewardIdentityBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("dominance_action", self._handle_dominance_action)
                self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
                self._subscribed = True
            
            logger.info("DominanceRewardIdentityBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing DominanceRewardIdentityBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceRewardIdentity")
    async def process_dominance_outcome(self, 
                                     outcome_type: str,
                                     user_id: str,
                                     intensity: float,
                                     success: bool,
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the outcome of a dominance interaction, generating rewards and
        updating identity and relationship state.
        
        Args:
            outcome_type: Type of dominance outcome
            user_id: User ID
            intensity: Dominance intensity (0.0-1.0)
            success: Whether the dominance action was successful
            context: Additional context
            
        Returns:
            Processing results
        """
        try:
            if not context:
                context = {}
                
            # Prepare results container
            results = {
                "user_id": user_id,
                "outcome_type": outcome_type,
                "success": success,
                "reward_generated": False,
                "identity_updated": False,
                "relationship_updated": False,
            }
            
            # 1. Calculate reward value
            reward_value = 0.0
            reward_source = ""
            
            if success:
                # Successful outcomes
                if outcome_type == "compliance":
                    reward_value = self.reward_values["dominance_compliance"] * intensity
                    reward_source = "user_compliance"
                elif outcome_type == "escalation_success":
                    reward_value = self.reward_values["dominance_escalation_success"] * intensity
                    reward_source = "dominance_escalation"
                elif outcome_type == "resistance_overcome":
                    reward_value = self.reward_values["dominance_resistance_overcome"] * intensity
                    reward_source = "resistance_overcome_sim"
                elif outcome_type == "submission_expressed":
                    reward_value = self.reward_values["submission_expressed"] * intensity
                    reward_source = "user_submission"
                elif outcome_type == "boundary_success":
                    reward_value = self.reward_values["challenging_boundary_success"] * intensity
                    reward_source = "boundary_expansion"
            else:
                # Failed outcomes
                if outcome_type == "compliance_failure":
                    reward_value = self.reward_values["dominance_failure"] * intensity
                    reward_source = "dominance_failure"
                elif outcome_type == "resistance_failure":
                    reward_value = self.reward_values["resistance_failure"] * intensity
                    reward_source = "resistance_failure"
                elif outcome_type == "boundary_failure":
                    reward_value = self.reward_values["boundary_failure"] * intensity
                    reward_source = "boundary_failure"
            
            # Add intensity information to context
            context["dominance_intensity"] = intensity
            context["user_id"] = user_id
            
            # 2. Generate reward signal
            if self.reward_system and abs(reward_value) > 0.1:
                reward_signal = RewardSignal(
                    value=reward_value,
                    source=reward_source,
                    context=context
                )
                
                # Process reward
                reward_result = await self.reward_system.process_reward_signal(reward_signal)
                results["reward_generated"] = True
                results["reward_value"] = reward_value
                results["reward_source"] = reward_source
                results["reward_effects"] = reward_result.get("effects", {})
            
            # 3. Update identity if success is significant
            if self.identity_evolution and success and intensity >= 0.5:
                # Calculate identity impact
                identity_impact = intensity * self.identity_impact_factor
                
                # Update dominance trait
                await self.identity_evolution.update_trait(
                    trait="dominance",
                    impact=identity_impact
                )
                
                # Update associated traits
                await self.identity_evolution.update_trait(
                    trait="assertiveness",
                    impact=identity_impact * 0.7
                )
                
                # Update preferences
                await self.identity_evolution.update_preference(
                    category="interaction_styles",
                    preference="dominant",
                    impact=identity_impact
                )
                
                results["identity_updated"] = True
                results["identity_impact"] = identity_impact
            
            # 4. Update relationship if available
            if self.relationship_manager and user_id:
                # Prepare relationship update data
                update_data = {
                    "interaction_type": "dominance_interaction",
                    "summary": f"Dominance interaction: {outcome_type} ({'success' if success else 'failure'})",
                    "dominance": {
                        "intensity_level": intensity,
                        "outcome_type": outcome_type,
                        "success": success
                    }
                }
                
                # Add details based on outcome type
                if outcome_type == "escalation_success":
                    update_data["dominance"]["escalation_attempt"] = True
                    update_data["dominance"]["escalation_success"] = True
                
                # Update emotional context
                update_data["emotional_context"] = {
                    "primary_emotion": "dominant" if success else "frustrated",
                    "valence": 0.7 if success else -0.3,
                    "arousal": 0.6 if intensity > 0.5 else 0.4
                }
                
                # Update relationship
                relationship_result = await self.relationship_manager.update_relationship_on_interaction(
                    user_id, update_data
                )
                
                results["relationship_updated"] = True
                results["relationship_updates"] = {
                    "trust_impact": relationship_result.get("trust_impact", 0),
                    "dominance_impact": relationship_result.get("dominance_impact", 0)
                }
            
            logger.info(f"Processed dominance outcome: {outcome_type} for user {user_id} (success: {success})")
            return results
            
        except Exception as e:
            logger.error(f"Error processing dominance outcome: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def _handle_dominance_action(self, event: Event) -> None:
        """
        Handle dominance action events from the event bus.
        
        Args:
            event: Dominance action event
        """
        try:
            # Extract event data
            action = event.data.get("action")
            user_id = event.data.get("user_id")
            intensity = event.data.get("intensity", 0.5)
            outcome = event.data.get("outcome")
            
            if not action or not user_id:
                return
            
            # Map action to outcome type
            outcome_type = "compliance"  # Default
            if "command" in action.lower():
                outcome_type = "compliance" if outcome == "success" else "compliance_failure"
            elif "escalate" in action.lower() or "increase" in action.lower():
                outcome_type = "escalation_success" if outcome == "success" else "resistance_failure"
            elif "boundary" in action.lower():
                outcome_type = "boundary_success" if outcome == "success" else "boundary_failure"
            
            # Process the outcome
            asyncio.create_task(self.process_dominance_outcome(
                outcome_type=outcome_type,
                user_id=user_id,
                intensity=intensity,
                success=outcome == "success",
                context={"action": action}
            ))
            
        except Exception as e:
            logger.error(f"Error handling dominance action event: {e}")
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """
        Handle user interaction events to detect submission expressions.
        
        Args:
            event: User interaction event
        """
        try:
            # Extract event data
            user_id = event.data.get("user_id")
            content = event.data.get("content", "")
            analysis = event.data.get("emotional_analysis", {})
            
            if not user_id or not content:
                return
            
            # Check for submission markers in content or analysis
            submission_markers = ["yes mistress", "yes goddess", "obey", "submit", "serve you", "your command"]
            submission_detected = any(marker in content.lower() for marker in submission_markers)
            
            # Also check emotional analysis
            if isinstance(analysis, dict):
                submission_emotion = analysis.get("submission", 0)
                if submission_emotion > 0.7:
                    submission_detected = True
            
            if submission_detected:
                # Process as submission expressed
                asyncio.create_task(self.process_dominance_outcome(
                    outcome_type="submission_expressed",
                    user_id=user_id,
                    intensity=0.7,  # Default intensity for explicit submission
                    success=True,
                    context={"content": content, "analysis": analysis}
                ))
                
        except Exception as e:
            logger.error(f"Error handling user interaction event: {e}")

# Function to create the bridge
def create_dominance_reward_identity_bridge(nyx_brain):
    """Create a dominance-reward-identity bridge for the given brain."""
    return DominanceRewardIdentityBridge(
        dominance_system=nyx_brain.dominance_system if hasattr(nyx_brain, "dominance_system") else None,
        reward_system=nyx_brain.reward_system if hasattr(nyx_brain, "reward_system") else None,
        identity_evolution=nyx_brain.identity_evolution if hasattr(nyx_brain, "identity_evolution") else None,
        relationship_manager=nyx_brain.relationship_manager if hasattr(nyx_brain, "relationship_manager") else None
    )
