# nyx/core/brain/adaptation/context_detection.py
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ContextChangeDetector:
    """Detects changes in interaction context for adaptation"""
    
    def __init__(self, brain):
        self.brain = brain
        self.context_history = []
        self.last_emotional_state = {}
        self.change_thresholds = {
            "emotional_valence": 0.4,  # Threshold for significant emotional valence change
            "emotional_arousal": 0.3,  # Threshold for significant emotional arousal change
            "topic_shift": 0.6,        # Threshold for topic shift detection
            "scenario_change": 0.7,    # Threshold for scenario type change
            "cross_user_transition": 0.8  # Threshold for transition to/from cross-user experiences
        }
    
    async def detect_context_change(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect significant changes in context
        
        Args:
            current_context: Current interaction context
            
        Returns:
            Context change detection results
        """
        # If no history, establish baseline
        if not self.context_history:
            self.context_history.append(current_context)
            if "emotional_state" in current_context:
                self.last_emotional_state = current_context["emotional_state"]
            return {"significant_change": False, "change_type": "initial_context"}
        
        # Get previous context
        previous_context = self.context_history[-1]
        
        # Check for emotional state changes
        emotional_change = self._detect_emotional_change(
            previous_context.get("emotional_state", {}),
            current_context.get("emotional_state", {})
        )
        
        # Check for scenario type changes
        scenario_change = self._detect_scenario_change(
            previous_context.get("scenario_type", ""),
            current_context.get("scenario_type", "")
        )
        
        # Check for cross-user experience transitions
        cross_user_change = self._detect_cross_user_transition(
            previous_context.get("cross_user_experience", False),
            current_context.get("cross_user_experience", False)
        )
        
        # Determine if a significant change has occurred
        significant_changes = []
        
        if emotional_change["significant"]:
            significant_changes.append(emotional_change)
        
        if scenario_change["significant"]:
            significant_changes.append(scenario_change)
        
        if cross_user_change["significant"]:
            significant_changes.append(cross_user_change)
        
        # Add this context to history
        self.context_history.append(current_context)
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]  # Keep only the last 10 entries
        
        # Update last emotional state
        if "emotional_state" in current_context:
            self.last_emotional_state = current_context["emotional_state"]
        
        # Return change detection results
        if significant_changes:
            # Determine primary change type
            primary_change = max(significant_changes, key=lambda x: x["confidence"])
            
            result = {
                "significant_change": True,
                "change_type": primary_change["type"],
                "confidence": primary_change["confidence"],
                "detected_changes": significant_changes,
                "prior_context": previous_context,
                "current_context": current_context
            }
        else:
            result = {
                "significant_change": False,
                "prior_context": previous_context,
                "current_context": current_context
            }
        
        return result
    
    def _detect_emotional_change(self, 
                              previous_state: Dict[str, float], 
                              current_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect significant changes in emotional state
        
        Args:
            previous_state: Previous emotional state
            current_state: Current emotional state
            
        Returns:
            Emotional change detection results
        """
        # If emotional states not available, no change
        if not previous_state or not current_state:
            return {"significant": False, "type": "emotional_change", "confidence": 0.0}
        
        # Calculate emotional valence change if available
        valence_change = 0.0
        if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_emotional_valence"):
            previous_valence = self.brain.emotional_core.get_emotional_valence(previous_state)
            current_valence = self.brain.emotional_core.get_emotional_valence(current_state)
            valence_change = abs(current_valence - previous_valence)
        
        # Calculate emotional arousal change if available
        arousal_change = 0.0
        if hasattr(self.brain, "emotional_core") and hasattr(self.brain.emotional_core, "get_emotional_arousal"):
            previous_arousal = self.brain.emotional_core.get_emotional_arousal(previous_state)
            current_arousal = self.brain.emotional_core.get_emotional_arousal(current_state)
            arousal_change = abs(current_arousal - previous_arousal)
        
        # Check if dominant emotion has changed
        dominant_emotion_change = False
        previous_dominant = max(previous_state.items(), key=lambda x: x[1])[0] if previous_state else ""
        current_dominant = max(current_state.items(), key=lambda x: x[1])[0] if current_state else ""
        
        if previous_dominant != current_dominant:
            dominant_emotion_change = True
        
        # Calculate overall confidence in emotional change
        valence_confidence = valence_change / self.change_thresholds["emotional_valence"] if valence_change > 0 else 0
        arousal_confidence = arousal_change / self.change_thresholds["emotional_arousal"] if arousal_change > 0 else 0
        dominant_confidence = 0.7 if dominant_emotion_change else 0
        
        overall_confidence = max(valence_confidence, arousal_confidence, dominant_confidence)
        
        # Determine if the change is significant
        significant = overall_confidence >= 1.0
        
        return {
            "significant": significant,
            "type": "emotional_change",
            "confidence": overall_confidence,
            "valence_change": valence_change,
            "arousal_change": arousal_change,
            "dominant_emotion_change": dominant_emotion_change,
            "previous_dominant": previous_dominant,
            "current_dominant": current_dominant
        }
    
    def _detect_scenario_change(self, previous_scenario: str, current_scenario: str) -> Dict[str, Any]:
        """
        Detect changes in scenario type
        
        Args:
            previous_scenario: Previous scenario type
            current_scenario: Current scenario type
            
        Returns:
            Scenario change detection results
        """
        # If scenarios not specified, no change
        if not previous_scenario or not current_scenario:
            return {"significant": False, "type": "scenario_change", "confidence": 0.0}
        
        # Direct comparison
        if previous_scenario != current_scenario:
            return {
                "significant": True,
                "type": "scenario_change",
                "confidence": 1.0,
                "previous_scenario": previous_scenario,
                "current_scenario": current_scenario
            }
        
        return {"significant": False, "type": "scenario_change", "confidence": 0.0}
    
    def _detect_cross_user_transition(self, previous_cross_user: bool, current_cross_user: bool) -> Dict[str, Any]:
        """
        Detect transitions to/from cross-user experiences
        
        Args:
            previous_cross_user: Whether previous interaction involved cross-user experience
            current_cross_user: Whether current interaction involves cross-user experience
            
        Returns:
            Cross-user transition detection results
        """
        # If no change, return immediately
        if previous_cross_user == current_cross_user:
            return {"significant": False, "type": "cross_user_transition", "confidence": 0.0}
        
        # Transition detected
        direction = "to_cross_user" if current_cross_user else "from_cross_user"
        
        return {
            "significant": True,
            "type": "cross_user_transition",
            "confidence": 1.0,
            "direction": direction,
            "previous_cross_user": previous_cross_user,
            "current_cross_user": current_cross_user
        }
