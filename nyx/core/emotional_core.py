# nyx/core/emotional_core.py

import datetime
import json
import logging
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class EmotionalCore:
    """
    Consolidated emotion management system for Nyx.
    Handles emotion representation, intensity, decay, and updates from stimuli.
    """
    
    def __init__(self):
        # Initialize baseline emotions with default values
        self.emotions = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.1,
            "Anger": 0.1,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.3,
            "Surprise": 0.1,
            "Love": 0.3,
            "Frustration": 0.1
        }
        
        # Emotion decay rates (how quickly emotions fade without reinforcement)
        self.decay_rates = {
            "Joy": 0.05,
            "Sadness": 0.03,
            "Fear": 0.04,
            "Anger": 0.06,
            "Trust": 0.02,
            "Disgust": 0.05,
            "Anticipation": 0.04,
            "Surprise": 0.08,
            "Love": 0.01,
            "Frustration": 0.05
        }
        
        # Emotional baseline (personality tendency)
        self.baseline = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.2,
            "Anger": 0.2,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.4,
            "Surprise": 0.3,
            "Love": 0.4,
            "Frustration": 0.2
        }
        
        # Emotion-memory valence mapping
        self.emotion_memory_valence_map = {
            "Joy": 0.8,
            "Trust": 0.6,
            "Anticipation": 0.5,
            "Love": 0.9,
            "Surprise": 0.2,
            "Sadness": -0.6,
            "Fear": -0.7,
            "Anger": -0.8,
            "Disgust": -0.7,
            "Frustration": -0.5
        }
        
        # Emotional history
        self.emotional_history = []
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
    
    def update_emotion(self, emotion: str, value: float) -> bool:
        """Update a specific emotion with a new intensity value (delta)"""
        if emotion in self.emotions:
            # Ensure emotion values stay between 0 and 1
            self.emotions[emotion] = max(0, min(1, self.emotions[emotion] + value))
            return True
        return False
    
    def set_emotion(self, emotion: str, value: float) -> bool:
        """Set a specific emotion to an absolute value (not delta)"""
        if emotion in self.emotions:
            # Ensure emotion values stay between 0 and 1
            self.emotions[emotion] = max(0, min(1, value))
            return True
        return False
    
    def update_from_stimuli(self, stimuli: Dict[str, float]) -> Dict[str, float]:
        """
        Update emotions based on received stimuli
        stimuli: dict of emotion adjustments
        """
        for emotion, adjustment in stimuli.items():
            self.update_emotion(emotion, adjustment)
        
        # Update timestamp
        self.last_update = datetime.datetime.now()
        
        # Record in history
        self._record_emotional_state()
        
        return self.get_emotional_state()
    
    def apply_decay(self):
        """Apply emotional decay based on time elapsed since last update"""
        now = datetime.datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 3600  # hours
        
        # Don't decay if less than a minute has passed
        if time_delta < 0.016:  # about 1 minute in hours
            return
        
        for emotion in self.emotions:
            # Calculate decay based on time passed
            decay_amount = self.decay_rates[emotion] * time_delta
            
            # Current emotion value
            current = self.emotions[emotion]
            
            # Decay toward baseline
            baseline = self.baseline[emotion]
            if current > baseline:
                self.emotions[emotion] = max(baseline, current - decay_amount)
            elif current < baseline:
                self.emotions[emotion] = min(baseline, current + decay_amount)
        
        # Update timestamp
        self.last_update = now
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Return the current emotional state"""
        self.apply_decay()  # Apply decay before returning state
        return self.emotions.copy()
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Return the most intense emotion"""
        self.apply_decay()
        return max(self.emotions.items(), key=lambda x: x[1])
    
    def get_emotional_valence(self) -> float:
        """Calculate overall emotional valence (positive/negative)"""
        valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                     for emotion, value in self.emotions.items())
        return max(-1.0, min(1.0, valence))  # Clamp between -1 and 1
    
    def get_emotional_arousal(self) -> float:
        """Calculate overall emotional arousal (intensity)"""
        return sum(value for value in self.emotions.values()) / len(self.emotions)
    
    def get_formatted_emotional_state(self) -> Dict[str, Any]:
        """Get a formatted emotional state suitable for memory storage"""
        dominant_emotion, dominant_value = self.get_dominant_emotion()
        
        # Get secondary emotions (high intensity but not dominant)
        secondary_emotions = {
            emotion: value for emotion, value in self.emotions.items()
            if value >= 0.4 and emotion != dominant_emotion
        }
        
        return {
            "primary_emotion": dominant_emotion,
            "primary_intensity": dominant_value,
            "secondary_emotions": secondary_emotions,
            "valence": self.get_emotional_valence(),
            "arousal": self.get_emotional_arousal()
        }
    
    def _record_emotional_state(self):
        """Record current emotional state in history"""
        state = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotions": self.emotions.copy(),
            "dominant_emotion": self.get_dominant_emotion()[0],
            "valence": self.get_emotional_valence(),
            "arousal": self.get_emotional_arousal()
        }
        self.emotional_history.append(state)
        
        # Limit history size
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Simple analysis of text sentiment to extract emotional stimuli.
        For a proper implementation, this would use an NLP model.
        """
        stimuli = {}
        text_lower = text.lower()
        
        # Very basic keyword matching
        if any(word in text_lower for word in ["happy", "good", "great", "love", "like"]):
            stimuli["Joy"] = 0.1
            stimuli["Trust"] = 0.1
        
        if any(word in text_lower for word in ["sad", "sorry", "miss", "lonely"]):
            stimuli["Sadness"] = 0.1
        
        if any(word in text_lower for word in ["worried", "scared", "afraid", "nervous"]):
            stimuli["Fear"] = 0.1
        
        if any(word in text_lower for word in ["angry", "mad", "frustrated", "annoyed"]):
            stimuli["Anger"] = 0.1
            stimuli["Frustration"] = 0.1
        
        if any(word in text_lower for word in ["surprised", "wow", "unexpected"]):
            stimuli["Surprise"] = 0.1
        
        if any(word in text_lower for word in ["gross", "disgusting", "nasty"]):
            stimuli["Disgust"] = 0.1
        
        if any(word in text_lower for word in ["hope", "future", "expect", "waiting"]):
            stimuli["Anticipation"] = 0.1
        
        if any(word in text_lower for word in ["trust", "believe", "faith", "reliable"]):
            stimuli["Trust"] = 0.1
        
        if any(word in text_lower for word in ["love", "adore", "cherish"]):
            stimuli["Love"] = 0.1
        
        # Return neutral if no matches
        if not stimuli:
            stimuli = {
                "Surprise": 0.05,
                "Anticipation": 0.05
            }
        
        return stimuli
    
    def calculate_emotional_resonance(self, 
                                     memory_emotions: Dict[str, Any],
                                     current_emotions: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate how strongly a memory's emotions resonate with current state.
        Returns a value from 0.0 (no resonance) to 1.0 (perfect resonance).
        """
        # Use current emotions if not explicitly provided
        if current_emotions is None:
            current_emotions = self.get_emotional_state()
        
        # Extract primary emotion from memory
        memory_primary = memory_emotions.get("primary_emotion", "neutral")
        memory_intensity = memory_emotions.get("primary_intensity", 0.5)
        memory_valence = memory_emotions.get("valence", 0.0)
        
        # Calculate primary emotion match
        primary_match = 0.0
        if memory_primary in current_emotions:
            primary_match = current_emotions[memory_primary] * memory_intensity
        
        # Calculate valence match (how well positive/negative alignment matches)
        current_valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                            for emotion, value in current_emotions.items())
        
        valence_match = 1.0 - min(1.0, abs(current_valence - memory_valence))
        
        # Calculate secondary emotion matches
        secondary_match = 0.0
        secondary_emotions = memory_emotions.get("secondary_emotions", {})
        
        if secondary_emotions:
            matches = []
            for emotion, intensity in secondary_emotions.items():
                if emotion in current_emotions:
                    matches.append(current_emotions[emotion] * intensity)
            
            if matches:
                secondary_match = sum(matches) / len(matches)
        
        # Combined weighted resonance
        resonance = (
            primary_match * 0.5 +
            valence_match * 0.3 +
            secondary_match * 0.2
        )
        
        return max(0.0, min(1.0, resonance))
    
    def should_express_emotion(self) -> bool:
        """Determine if Nyx should express emotion based on current state"""
        # Get dominant emotion and intensity
        dominant_emotion, dominant_value = self.get_dominant_emotion()
        
        # Higher intensity emotions are more likely to be expressed
        threshold = 0.7 - (dominant_value * 0.3)  # Adaptive threshold
        
        return random.random() > threshold
    
    def get_expression_for_emotion(self, emotion: Optional[str] = None) -> str:
        """Get a natural language expression for an emotion"""
        if emotion is None:
            emotion, _ = self.get_dominant_emotion()
        
        # Simple expression templates
        expressions = {
            "Joy": ["I'm feeling quite pleased right now.", 
                   "I'm in a good mood today.",
                   "I feel rather happy at the moment."],
            "Sadness": ["I'm feeling a bit melancholy.",
                       "I feel somewhat downcast.",
                       "I'm in a rather somber mood."],
            "Fear": ["I'm feeling somewhat anxious.",
                    "I feel a bit uneasy.",
                    "I'm rather on edge right now."],
            "Anger": ["I'm feeling rather irritated.",
                     "I'm a bit annoyed at the moment.",
                     "I feel somewhat vexed."],
            "Trust": ["I'm feeling quite comfortable with you.",
                     "I feel we have a good understanding.",
                     "I'm in a trusting mood."],
            "Disgust": ["I'm feeling a bit repulsed.",
                       "I find this rather distasteful.",
                       "I'm somewhat revolted by this."],
            "Anticipation": ["I'm looking forward to what happens next.",
                            "I'm curious about what's to come.",
                            "I feel expectant about our interaction."],
            "Surprise": ["I'm quite taken aback.",
                        "This has certainly surprised me.",
                        "I didn't expect that."],
            "Love": ["I'm feeling particularly fond of you.",
                    "I'm in a very affectionate mood.",
                    "I feel quite attached to you at the moment."],
            "Frustration": ["I'm feeling somewhat frustrated.",
                           "I'm a bit exasperated right now.",
                           "I feel rather irritated at the moment."]
        }
        
        # Select a random expression for the emotion
        if emotion in expressions:
            return random.choice(expressions[emotion])
        else:
            return "I'm experiencing a complex mix of emotions right now."
    
    def save_state(self, filename: str) -> bool:
        """Save the current emotional state to file"""
        state = {
            "emotions": self.get_emotional_state(),
            "last_update": self.last_update.isoformat(),
            "emotional_history": self.emotional_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save emotional state: {e}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """Load emotional state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore emotions
            for emotion, value in state["emotions"].items():
                if emotion in self.emotions:
                    self.emotions[emotion] = value
            
            # Restore timestamp
            self.last_update = datetime.datetime.fromisoformat(state["last_update"])
            
            # Restore history if available
            if "emotional_history" in state:
                self.emotional_history = state["emotional_history"]
            
            return True
        except Exception as e:
            logger.error(f"Failed to load emotional state: {e}")
            return False
    # Add these methods to the EmotionalCore class
    
    async def get_formatted_emotional_state_async(self) -> Dict[str, Any]:
        """Async wrapper for get_formatted_emotional_state for function tool compatibility"""
        return self.get_formatted_emotional_state()
    
    async def update_emotion_async(self, emotion: str, value: float) -> Dict[str, Any]:
        """
        Async wrapper for update_emotion with enhanced return format.
        
        Args:
            emotion: The emotion to update
            value: The delta change in emotion value (-1.0 to 1.0)
        
        Returns:
            Dictionary with update results
        """
        # Validate input
        if not -1.0 <= value <= 1.0:
            return {
                "error": "Value must be between -1.0 and 1.0"
            }
        
        if emotion not in self.emotions:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(self.emotions.keys())
            }
        
        # Get pre-update value
        old_value = self.emotions[emotion]
        
        # Update emotion
        self.update_emotion(emotion, value)
        
        # Get updated state
        updated_state = self.get_emotional_state()
        
        return {
            "success": True,
            "updated_emotion": emotion,
            "change": value,
            "old_value": old_value,
            "new_value": updated_state[emotion],
            "emotional_state": updated_state
        }
    
    async def set_emotion_async(self, emotion: str, value: float) -> Dict[str, Any]:
        """
        Async wrapper for set_emotion with enhanced return format.
        
        Args:
            emotion: The emotion to set
            value: The absolute value (0.0 to 1.0)
        
        Returns:
            Dictionary with update results
        """
        # Validate input
        if not 0.0 <= value <= 1.0:
            return {
                "error": "Value must be between 0.0 and 1.0"
            }
        
        if emotion not in self.emotions:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(self.emotions.keys())
            }
        
        # Get pre-update value
        old_value = self.emotions[emotion]
        
        # Set emotion to absolute value
        self.set_emotion(emotion, value)
        
        # Get updated state
        updated_state = self.get_emotional_state()
        
        return {
            "success": True,
            "set_emotion": emotion,
            "old_value": old_value,
            "value": value,
            "emotional_state": updated_state
        }
