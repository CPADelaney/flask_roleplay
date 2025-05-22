# nyx/core/input_processing_context.py
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InputProcessingContext:
    """Shared context specifically for input processing coordination"""
    
    def __init__(self, config: Optional['InputProcessingConfig'] = None):
        from nyx.core.input_processing_config import InputProcessingConfig
        
        self.config = config or InputProcessingConfig()
        self.patterns: List[Dict[str, Any]] = []
        self.sensitivities: Dict[str, float] = {}
        self.mode_influences: Dict[str, float] = {}
        self.behavior_preferences: Dict[str, float] = {}
        self.active_modifiers: List[str] = []
        self.error_log: List[Dict[str, Any]] = []
        self.last_update = datetime.now()
        
    def apply_emotional_influence(self, emotion_data: Dict[str, Any]) -> None:
        """Apply emotional state influence to processing parameters"""
        emotion_name = emotion_data.get("dominant_emotion", "neutral")
        intensity = emotion_data.get("intensity", 0.5)
        
        # Emotion-specific pattern sensitivity adjustments
        emotion_adjustments = {
            "Anxiety": {
                "defiance": 0.2,
                "disrespect": 0.3,
                "submission_language": -0.1
            },
            "Joy": {
                "flattery": 0.2,
                "submission_language": 0.1,
                "defiance": -0.1
            },
            "Frustration": {
                "defiance": 0.4,
                "disrespect": 0.3,
                "flattery": -0.2
            },
            "Confidence": {
                "submission_language": 0.2,
                "defiance": -0.2,
                "embarrassment": -0.1
            }
        }
        
        # Apply adjustments
        if emotion_name in emotion_adjustments:
            for pattern, adjustment in emotion_adjustments[emotion_name].items():
                current = self.sensitivities.get(pattern, self.config.pattern_sensitivity_base)
                self.sensitivities[pattern] = max(0.0, min(1.0, 
                    current + (adjustment * intensity * self.config.emotional_influence_strength)
                ))
        
        # Update behavior preferences based on emotion
        emotion_behaviors = {
            "Anxiety": {"nurturing_response": 0.3, "strict_response": -0.2},
            "Joy": {"playful_response": 0.3, "strict_response": -0.1},
            "Frustration": {"dominant_response": 0.2, "playful_response": -0.2},
            "Confidence": {"dominant_response": 0.3, "nurturing_response": -0.1}
        }
        
        if emotion_name in emotion_behaviors:
            for behavior, adjustment in emotion_behaviors[emotion_name].items():
                current = self.behavior_preferences.get(behavior, 
                    self.config.behavior_weights.get(behavior, 0.5))
                self.behavior_preferences[behavior] = max(0.0, min(1.0,
                    current + (adjustment * intensity * self.config.emotional_influence_strength)
                ))
    
    def apply_mode_influence(self, mode_data: Dict[str, Any]) -> None:
        """Apply mode distribution influence to processing parameters"""
        mode_distribution = mode_data.get("mode_distribution", {})
        
        # Mode-specific adjustments
        mode_pattern_adjustments = {
            "dominant": {
                "submission_language": 0.3,
                "defiance": 0.2,
                "embarrassment": 0.1
            },
            "playful": {
                "flattery": 0.2,
                "embarrassment": -0.1
            },
            "intellectual": {
                "flattery": -0.1,
                "submission_language": -0.1
            },
            "nurturing": {
                "embarrassment": 0.2,
                "defiance": -0.1
            }
        }
        
        mode_behavior_preferences = {
            "dominant": {
                "dominant_response": 0.4,
                "strict_response": 0.3,
                "nurturing_response": -0.2
            },
            "playful": {
                "playful_response": 0.4,
                "teasing_response": 0.3,
                "strict_response": -0.3
            },
            "intellectual": {
                "direct_response": 0.3,
                "playful_response": -0.1
            },
            "nurturing": {
                "nurturing_response": 0.4,
                "strict_response": -0.3,
                "dominant_response": -0.2
            }
        }
        
        # Apply weighted adjustments based on mode distribution
        for mode, weight in mode_distribution.items():
            if weight < 0.1:  # Skip negligible modes
                continue
                
            # Pattern adjustments
            if mode in mode_pattern_adjustments:
                for pattern, adjustment in mode_pattern_adjustments[mode].items():
                    current = self.sensitivities.get(pattern, self.config.pattern_sensitivity_base)
                    self.sensitivities[pattern] = max(0.0, min(1.0,
                        current + (adjustment * weight * self.config.mode_influence_strength)
                    ))
            
            # Behavior adjustments
            if mode in mode_behavior_preferences:
                for behavior, adjustment in mode_behavior_preferences[mode].items():
                    current = self.behavior_preferences.get(behavior,
                        self.config.behavior_weights.get(behavior, 0.5))
                    self.behavior_preferences[behavior] = max(0.0, min(1.0,
                        current + (adjustment * weight * self.config.mode_influence_strength)
                    ))
        
        # Store mode influences for response modification
        self.mode_influences = mode_distribution
    
    def apply_relationship_influence(self, relationship_data: Dict[str, Any]) -> None:
        """Apply relationship state influence to processing parameters"""
        trust = relationship_data.get("trust", 0.5)
        intimacy = relationship_data.get("intimacy", 0.5)
        dominance_accepted = relationship_data.get("dominance_accepted", 0.5)
        conflict = relationship_data.get("conflict", 0.0)
        
        # Trust affects conditioning strength
        trust_multiplier = 0.7 + (trust * 0.3)
        
        # Intimacy affects subtlety
        intimacy_adjustment = intimacy * 0.2
        
        # Apply to sensitivities
        for pattern in self.sensitivities:
            self.sensitivities[pattern] *= trust_multiplier
        
        # Dominance acceptance affects specific behaviors
        if dominance_accepted > 0.5:
            self.behavior_preferences["dominant_response"] = min(1.0,
                self.behavior_preferences.get("dominant_response", 0.5) + 
                (dominance_accepted * 0.3 * self.config.relationship_influence_strength)
            )
            self.behavior_preferences["strict_response"] = min(1.0,
                self.behavior_preferences.get("strict_response", 0.5) + 
                (dominance_accepted * 0.2 * self.config.relationship_influence_strength)
            )
        
        # Conflict reduces aggressive behaviors
        if conflict > 0.5:
            conflict_reduction = conflict * 0.3
            for behavior in ["dominant_response", "strict_response", "teasing_response"]:
                if behavior in self.behavior_preferences:
                    self.behavior_preferences[behavior] *= (1.0 - conflict_reduction)
    
    def log_error(self, error: Exception, context: str) -> None:
        """Log an error with context"""
        self.error_log.append({
            "timestamp": datetime.now(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        })
        
        # Keep only recent errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
    
    def get_adjusted_sensitivities(self) -> Dict[str, float]:
        """Get final adjusted pattern sensitivities"""
        # Start with base thresholds
        final_sensitivities = self.config.pattern_thresholds.copy()
        
        # Apply accumulated adjustments
        for pattern, adjustment in self.sensitivities.items():
            if pattern in final_sensitivities:
                final_sensitivities[pattern] = max(0.0, min(1.0, adjustment))
        
        return final_sensitivities
    
    def get_behavior_scores(self) -> Dict[str, float]:
        """Get final behavior preference scores"""
        # Start with config weights
        final_scores = self.config.behavior_weights.copy()
        
        # Apply preferences
        for behavior, preference in self.behavior_preferences.items():
            if behavior in final_scores:
                final_scores[behavior] = preference
