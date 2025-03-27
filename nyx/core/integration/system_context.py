# nyx/core/integration/system_context.py

import datetime
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class CircularBuffer:
    """Thread-safe circular buffer for history tracking."""
    def __init__(self, max_size: int = 100):
        self.buffer = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add(self, item: Any) -> None:
        """Add an item to the buffer."""
        async with self._lock:
            self.buffer.append(item)
    
    async def get_all(self) -> List[Any]:
        """Get all items in the buffer."""
        async with self._lock:
            return list(self.buffer)
    
    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self.buffer.clear()

class AffectiveState:
    """Current affective state spanning emotional, hormonal, and mood systems."""
    def __init__(self):
        # Emotional state
        self.primary_emotion = "neutral"
        self.emotion_intensity = 0.0
        self.valence = 0.0        # -1.0 to 1.0
        self.arousal = 0.5        # 0.0 to 1.0
        self.dominance = 0.0      # -1.0 to 1.0 (control dimension)
        
        # Hormone states
        self.hormone_levels = {}  # hormone_name -> level
        
        # Mood state
        self.mood = "neutral"
        self.mood_stability = 0.5  # 0.0 = volatile, 1.0 = stable
        
        # History buffers
        self.emotion_history = CircularBuffer(50)
        self.hormone_history = CircularBuffer(50)
        self.mood_history = CircularBuffer(20)
        
        # Timestamp
        self.last_updated = datetime.datetime.now()
    
    async def update_emotion(self, emotion: str, intensity: float, valence: float, 
                           arousal: float, dominance: float) -> None:
        """Update the emotional state."""
        self.primary_emotion = emotion
        self.emotion_intensity = intensity
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
        self.last_updated = datetime.datetime.now()
        
        await self.emotion_history.add({
            "emotion": emotion,
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "timestamp": self.last_updated.isoformat()
        })
    
    async def update_hormone(self, hormone: str, level: float) -> None:
        """Update a hormone level."""
        self.hormone_levels[hormone] = level
        self.last_updated = datetime.datetime.now()
        
        await self.hormone_history.add({
            "hormone": hormone,
            "level": level,
            "timestamp": self.last_updated.isoformat()
        })
    
    async def update_mood(self, mood: str, stability: float) -> None:
        """Update the mood state."""
        self.mood = mood
        self.mood_stability = stability
        self.last_updated = datetime.datetime.now()
        
        await self.mood_history.add({
            "mood": mood,
            "stability": stability,
            "timestamp": self.last_updated.isoformat()
        })
    
    async def get_emotion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get emotion history."""
        all_history = await self.emotion_history.get_all()
        return all_history[-limit:]
    
    async def get_hormone_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get hormone history."""
        all_history = await self.hormone_history.get_all()
        return all_history[-limit:]
    
    async def get_mood_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get mood history."""
        all_history = await self.mood_history.get_all()
        return all_history[-limit:]

class BodyState:
    """Current body perception state."""
    def __init__(self):
        # Overall body state
        self.has_visual_form = False
        self.form_description = "Amorphous digital entity"
        self.overall_integrity = 1.0
        self.comfort_level = 0.0      # -1.0 to 1.0
        
        # Sensory states
        self.regions = {}         # region_name -> {sensation_type: intensity}
        self.dominant_sensation = None
        self.dominant_region = None
        self.posture_effect = "neutral"
        self.movement_quality = "fluid"
        
        # History
        self.sensation_history = CircularBuffer(50)
        
        # Timestamp
        self.last_updated = datetime.datetime.now()
    
    async def update_sensation(self, region: str, sensation_type: str, 
                             intensity: float, cause: str = "") -> None:
        """Update sensation in a region."""
        if region not in self.regions:
            self.regions[region] = {}
        
        self.regions[region][sensation_type] = intensity
        self.last_updated = datetime.datetime.now()
        
        # Record history
        await self.sensation_history.add({
            "region": region,
            "sensation_type": sensation_type,
            "intensity": intensity,
            "cause": cause,
            "timestamp": self.last_updated.isoformat()
        })
        
        # Update dominant sensation/region
        self._update_dominants()
    
    def _update_dominants(self) -> None:
        """Update dominant sensation and region."""
        max_intensity = 0.0
        for region, sensations in self.regions.items():
            for sensation_type, intensity in sensations.items():
                if intensity > max_intensity:
                    max_intensity = intensity
                    self.dominant_sensation = sensation_type
                    self.dominant_region = region
    
    async def get_sensation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sensation history."""
        all_history = await self.sensation_history.get_all()
        return all_history[-limit:]
    
    async def update_form(self, has_form: bool, description: str) -> None:
        """Update body form."""
        self.has_visual_form = has_form
        self.form_description = description
        self.last_updated = datetime.datetime.now()
    
    async def update_posture(self, posture: str, movement: str) -> None:
        """Update posture and movement."""
        self.posture_effect = posture
        self.movement_quality = movement
        self.last_updated = datetime.datetime.now()

class UserModel:
    """Model of user state and traits."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Emotional state
        self.inferred_emotion = "neutral"
        self.emotion_confidence = 0.5
        self.valence = 0.0
        self.arousal = 0.5
        
        # Cognitive state
        self.inferred_goals = []
        self.inferred_beliefs = {}
        self.attention_focus = None
        self.knowledge_level = 0.5
        
        # Relationship perspective
        self.perceived_trust = 0.5
        self.perceived_familiarity = 0.1
        self.perceived_dominance = 0.5  # 0.0 = submissive, 1.0 = dominant
        self.perceived_receptivity = 0.5  # Receptiveness to dominance
        
        # Confidence
        self.overall_confidence = 0.4
        
        # History
        self.update_history = CircularBuffer(20)
        
        # Timestamp
        self.last_updated = datetime.datetime.now()
    
    async def update_state(self, updated_fields: Dict[str, Any]) -> None:
        """Update user model with new fields."""
        for field, value in updated_fields.items():
            if hasattr(self, field):
                setattr(self, field, value)
        
        self.last_updated = datetime.datetime.now()
        
        # Record history
        await self.update_history.add({
            "updated_fields": list(updated_fields.keys()),
            "timestamp": self.last_updated.isoformat()
        })
    
    async def get_update_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get update history."""
        all_history = await self.update_history.get_all()
        return all_history[-limit:]

class SystemContext:
    """
    Shared context object passed between modules.
    Provides a unified view of the system state across all modules.
    """
    def __init__(self):
        # Core state objects
        self.affective_state = AffectiveState()
        self.body_state = BodyState()
        self.user_models = {}  # user_id -> UserModel
        
        # Goals and needs
        self.active_goals = []  # [{id, description, priority, status}]
        self.need_states = {}   # need_name -> {level, deficit, drive_strength}
        
        # Action history
        self.action_history = CircularBuffer(50)
        
        # System state
        self.cycle_count = 0
        self.conversation_id = None
        self.session_start_time = datetime.datetime.now()
        self.custom_values = {}  # For arbitrary storage
        
        logger.info("SystemContext initialized")
    
    def get_or_create_user_model(self, user_id: str) -> UserModel:
        """Get or create a user model."""
        if user_id not in self.user_models:
            self.user_models[user_id] = UserModel(user_id)
        return self.user_models[user_id]
    
    async def record_action(self, action_type: str, params: Dict[str, Any], result: Any = None) -> None:
        """Record an action in the history."""
        await self.action_history.add({
            "action_type": action_type,
            "params": params,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    async def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get action history."""
        all_history = await self.action_history.get_all()
        return all_history[-limit:]
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a custom value."""
        self.custom_values[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a custom value."""
        return self.custom_values.get(key, default)
    
    def increment_cycle(self) -> int:
        """Increment and return the cycle count."""
        self.cycle_count += 1
        return self.cycle_count

# Singleton instance
_instance = None

def get_system_context() -> SystemContext:
    """Get the singleton system context instance."""
    global _instance
    if _instance is None:
        _instance = SystemContext()
    return _instance
