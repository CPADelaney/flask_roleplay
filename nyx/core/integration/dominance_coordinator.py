# nyx/core/femdom/bridges/dominance_coordinator.py

import logging
import asyncio
from typing import Dict, Any, Optional

from nyx.core.integration.event_bus import Event, get_event_bus

logger = logging.getLogger(__name__)

class DominanceCoordinatorBridge:
    """Coordinates dominance-related activities across systems."""
    
    def __init__(self, brain, dominance_system=None, sadistic_responses=None, psychological_dominance=None):
        self.brain = brain
        self.dominance_system = dominance_system
        self.sadistic_responses = sadistic_responses
        self.psychological_dominance = psychological_dominance
        self.event_bus = get_event_bus()
        self.initialized = False
        
        # Active dominance sessions per user
        self.active_sessions = {}
    
    async def initialize(self):
        """Initialize the bridge."""
        try:
            # Subscribe to relevant events
            self.event_bus.subscribe("user_interaction", self._on_user_interaction)
            self.event_bus.subscribe("emotional_state_change", self._on_emotional_change)
            
            self.initialized = True
            logger.info("DominanceCoordinatorBridge initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing DominanceCoordinatorBridge: {e}")
            return False
    
    async def _on_user_interaction(self, event):
        """Handle user interactions."""
        # Implementation for processing user interactions
        pass
    
    async def _on_emotional_change(self, event):
        """Handle emotional state changes."""
        # Implementation for responding to emotional changes
        pass
    
    async def process_dominance_action(self, action: str, user_id: str, intensity: float = 0.5) -> Dict[str, Any]:
        """Process a dominance action."""
        try:
            logger.info(f"Processing dominance action: {action} for user {user_id} at intensity {intensity}")
            
            # Record session activity
            self._update_session_activity(user_id, action)
            
            # Route to appropriate system based on action type
            if action.startswith("psychological_"):
                # Psychological dominance action
                if self.psychological_dominance:
                    # Extract action subtype
                    action_subtype = action.replace("psychological_", "")
                    
                    if action_subtype == "mindfuck":
                        # Generate mind game
                        user_state = await self._get_user_state(user_id)
                        result = await self.psychological_dominance.generate_mindfuck(
                            user_id, user_state, intensity
                        )
                        return result
                    
                    elif action_subtype == "gaslighting":
                        # Apply gaslighting strategy
                        result = await self.psychological_dominance.apply_gaslighting(
                            user_id, None, intensity
                        )
                        return result
                        
            elif action.startswith("sadistic_"):
                # Sadistic response action
                if self.sadistic_responses:
                    # Extract action subtype
                    action_subtype = action.replace("sadistic_", "")
                    
                    if action_subtype == "mockery":
                        # Generate mockery response
                        result = await self.sadistic_responses.generate_sadistic_amusement_response(
                            user_id, intensity, category="mockery"
                        )
                        return result
                    
                    elif action_subtype == "humiliation":
                        # Generate humiliation response
                        result = await self.sadistic_responses.generate_sadistic_amusement_response(
                            user_id, intensity, category="humiliation"
                        )
                        return result
            
            elif action.startswith("dominance_idea"):
                # Generate dominance ideas
                if self.dominance_system:
                    intensity_range = f"{int(intensity * 10) - 2}-{int(intensity * 10)}"
                    result = await self.dominance_system.generate_dominance_ideas(
                        user_id, "general", intensity_range, intensity > 0.7
                    )
                    return result
            
            # Default fallback
            return {
                "success": False,
                "message": f"Unsupported dominance action: {action}",
                "action": action,
                "user_id": user_id
            }
                
        except Exception as e:
            logger.error(f"Error processing dominance action: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action,
                "user_id": user_id
            }
    
    async def respond_to_violation(self, user_id: str, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a dominance response to a protocol violation."""
        try:
            protocol_name = violation.get("protocol_name", "protocol")
            severity = violation.get("severity", 0.5)
            
            # Generate appropriate response based on violation severity
            if severity > 0.7:
                # High severity - use sadistic mockery
                if self.sadistic_responses:
                    result = await self.sadistic_responses.generate_sadistic_amusement_response(
                        user_id, severity, category="mockery"
                    )
                    return {
                        "success": True,
                        "response_type": "sadistic_mockery",
                        "response": result.get("response", "Your failure to follow protocol is noted."),
                        "severity": severity
                    }
            else:
                # Lower severity - use dominance system
                if self.dominance_system:
                    # Evaluate appropriateness first
                    evaluation = await self.dominance_system.evaluate_dominance_step_appropriateness(
                        "protocol_correction", {"intensity": severity}, user_id
                    )
                    
                    if evaluation.get("action") == "proceed":
                        return {
                            "success": True,
                            "response_type": "protocol_correction",
                            "response": f"You've failed to follow the {protocol_name} protocol correctly. Correct this immediately.",
                            "severity": severity
                        }
                    else:
                        # Modified response based on evaluation
                        return {
                            "success": True,
                            "response_type": "modified_correction",
                            "response": f"I notice you didn't follow {protocol_name} protocol. Be more careful in the future.",
                            "severity": evaluation.get("new_intensity_level", severity * 0.7)  # Reduced intensity
                        }
            
            # Default fallback response
            return {
                "success": True,
                "response_type": "default_correction",
                "response": f"Your compliance with {protocol_name} protocol needs improvement.",
                "severity": severity
            }
            
        except Exception as e:
            logger.error(f"Error generating violation response: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    def _update_session_activity(self, user_id: str, action: str) -> None:
        """Update dominance session activity for a user."""
        # Create session if it doesn't exist
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = {
                "started_at": asyncio.get_event_loop().time(),
                "actions": [],
                "intensity_progression": []
            }
        
        # Add action to history
        session = self.active_sessions[user_id]
        session["actions"].append({
            "action": action,
            "timestamp": asyncio.get_event_loop().time() - session["started_at"]
        })
        
        # Limit history size
        if len(session["actions"]) > 50:
            session["actions"] = session["actions"][-50:]
    
    async def _get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get current user state for context."""
        user_state = {"user_id": user_id}
        
        # Get theory of mind data if available
        if hasattr(self.brain, "theory_of_mind") and self.brain.theory_of_mind:
            try:
                mental_state = await self.brain.theory_of_mind.get_user_model(user_id)
                if mental_state:
                    user_state.update(mental_state)
            except Exception as e:
                logger.error(f"Error getting mental state: {e}")
        
        # Add dominance-specific state
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            user_state["dominance_session_duration"] = asyncio.get_event_loop().time() - session["started_at"]
            user_state["dominance_action_count"] = len(session["actions"])
        
        return user_state
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            "initialized": self.initialized,
            "active_sessions": len(self.active_sessions),
            "systems_available": {
                "dominance_system": self.dominance_system is not None,
                "sadistic_responses": self.sadistic_responses is not None,
                "psychological_dominance": self.psychological_dominance is not None
            }
        }
