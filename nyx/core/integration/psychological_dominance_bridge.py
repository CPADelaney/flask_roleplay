# nyx/core/femdom/bridges/psychological_dominance_bridge.py

import logging
from typing import Dict, Any, Optional

from nyx.core.integration.event_bus import Event, get_event_bus

logger = logging.getLogger(__name__)

class PsychologicalDominanceBridge:
    """Bridge for psychological dominance capabilities."""
    
    def __init__(self, brain, psychological_dominance=None, theory_of_mind=None):
        self.brain = brain
        self.psychological_dominance = psychological_dominance
        self.theory_of_mind = theory_of_mind
        self.event_bus = get_event_bus()
        self.initialized = False
        
        # Track active mind games
        self.active_mind_games = {}
    
    async def initialize(self):
        """Initialize the bridge."""
        try:
            # Subscribe to relevant events
            self.event_bus.subscribe("user_interaction", self._on_user_interaction)
            
            # Initialize subcomponents if needed
            if hasattr(self.psychological_dominance, "initialize_event_subscriptions"):
                await self.psychological_dominance.initialize_event_subscriptions(self.event_bus)
            
            self.initialized = True
            logger.info("PsychologicalDominanceBridge initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing PsychologicalDominanceBridge: {e}")
            return False
    
    async def _on_user_interaction(self, event):
        """Handle user interactions."""
        user_id = event.data.get("user_id")
        content = event.data.get("content", "")
        
        # Check for active mind games
        if user_id in self.active_mind_games:
            game_id = self.active_mind_games[user_id]
            
            # Update mind game with user reaction
            if self.psychological_dominance:
                try:
                    # Analyze reaction type and intensity
                    reaction_type = "general"
                    intensity = 0.5
                    
                    # Use theory of mind for better reaction analysis if available
                    if self.theory_of_mind:
                        mental_state = await self.theory_of_mind.get_user_model(user_id)
                        if mental_state:
                            # Extract emotional reaction
                            emotion = mental_state.get("inferred_emotion", "neutral")
                            valence = mental_state.get("valence", 0.0)
                            arousal = mental_state.get("arousal", 0.5)
                            
                            # Map to reaction types
                            if emotion in ["confused", "uncertain", "puzzled"]:
                                reaction_type = "confusion"
                                intensity = arousal
                            elif emotion in ["anxious", "nervous", "worried"]:
                                reaction_type = "anxiety"
                                intensity = arousal
                            elif emotion in ["angry", "frustrated", "irritated"]:
                                reaction_type = "frustration"
                                intensity = abs(valence) * arousal
                            elif valence < -0.3:
                                reaction_type = "distress"
                                intensity = abs(valence)
                            elif valence > 0.3:
                                reaction_type = "pleasure"
                                intensity = valence
                    
                    # Record reaction in the mind game
                    await self.psychological_dominance.record_user_reaction(
                        user_id, game_id, reaction_type, intensity
                    )
                    
                    # Publish reaction event
                    await self.event_bus.publish(Event(
                        event_type="mind_game_reaction",
                        source="psychological_dominance_bridge",
                        data={
                            "user_id": user_id,
                            "game_id": game_id,
                            "reaction_type": reaction_type,
                            "intensity": intensity
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing mind game reaction: {e}")
        
        # Check for humiliation signals
        if self.theory_of_mind and hasattr(self.theory_of_mind, "detect_humiliation_signals"):
            try:
                humiliation_signals = await self.theory_of_mind.detect_humiliation_signals(content)
                
                if humiliation_signals.get("humiliation_detected", False):
                    intensity = humiliation_signals.get("intensity", 0.0)
                    
                    # Publish humiliation detected event
                    await self.event_bus.publish(Event(
                        event_type="humiliation_detected",
                        source="psychological_dominance_bridge",
                        data={
                            "user_id": user_id,
                            "intensity": intensity,
                            "markers": humiliation_signals.get("marker_count", 0)
                        }
                    ))
                    
                    # Update psychological dominance system if available
                    if self.psychological_dominance:
                        await self.psychological_dominance.update_humiliation_level(
                            user_id, humiliation_signals
                        )
                        
            except Exception as e:
                logger.error(f"Error detecting humiliation signals: {e}")
    
    async def amplify_dominance(self, intensity: float) -> Dict[str, Any]:
        """Amplify psychological dominance based on system state."""
        results = {
            "amplification_applied": False,
            "actions": []
        }
        
        try:
            # Only apply if psychological dominance is available
            if not self.psychological_dominance:
                return results
            
            # Apply gaslighting boost
            if hasattr(self.psychological_dominance, "_apply_gaslighting_boost"):
                boost_result = await self.psychological_dominance._apply_gaslighting_boost(intensity)
                if boost_result:
                    results["amplification_applied"] = True
                    results["actions"].append({
                        "type": "gaslighting_boost",
                        "intensity": intensity,
                        "result": boost_result
                    })
            
            # Boost mind game effectiveness
            if hasattr(self.psychological_dominance, "_boost_mind_game_effectiveness"):
                games_result = await self.psychological_dominance._boost_mind_game_effectiveness(intensity)
                if games_result:
                    results["actions"].append({
                        "type": "mind_game_boost",
                        "intensity": intensity,
                        "affected_games": games_result.get("affected_games", 0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error amplifying dominance: {e}")
            results["error"] = str(e)
            return results
    
    async def check_subspace(self, user_id: str) -> Dict[str, Any]:
        """Check if user is in subspace."""
        try:
            # First try the dedicated subspace detection
            if hasattr(self.psychological_dominance, "subspace_detection"):
                detection = self.psychological_dominance.subspace_detection
                if hasattr(detection, "get_subspace_guidance"):
                    guidance = await detection.get_subspace_guidance(user_id)
                    return {
                        "in_subspace": guidance.get("in_subspace", False),
                        "depth": guidance.get("depth_value", 0.0),
                        "depth_category": guidance.get("depth", "none"),
                        "guidance": guidance.get("guidance", ""),
                        "recommendations": guidance.get("recommendations", []),
                        "detection_method": "dedicated"
                    }
            
            # Fallback to theory of mind if available
            if self.theory_of_mind:
                mental_state = await self.theory_of_mind.get_user_model(user_id)
                if mental_state:
                    # Look for subspace indicators in mental state
                    arousal = mental_state.get("arousal", 0.0)
                    valence = mental_state.get("valence", 0.0)
                    emotion = mental_state.get("inferred_emotion", "")
                    
                    # Simple heuristic for subspace detection
                    subspace_score = 0.0
                    if arousal > 0.7 and valence > 0.3:
                        subspace_score += 0.5
                    
                    if emotion in ["euphoric", "hazy", "floating", "surrendered", "submissive"]:
                        subspace_score += 0.5
                    
                    return {
                        "in_subspace": subspace_score > 0.6,
                        "depth": subspace_score,
                        "depth_category": "light" if subspace_score > 0.6 else "none",
                        "detection_method": "inference"
                    }
            
            # Default response if no detection methods available
            return {
                "in_subspace": False,
                "depth": 0.0,
                "depth_category": "none",
                "detection_method": "default"
            }
                
        except Exception as e:
            logger.error(f"Error checking subspace: {e}")
            return {
                "in_subspace": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "initialized": self.initialized,
            "active_mind_games": len(self.active_mind_games),
            "systems_available": {
                "psychological_dominance": self.psychological_dominance is not None,
                "theory_of_mind": self.theory_of_mind is not None
            }
        }
