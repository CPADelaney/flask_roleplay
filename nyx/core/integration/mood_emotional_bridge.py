# nyx/core/integration/mood_emotional_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class MoodEmotionalBridge:
    """
    Integrates mood with emotional core, identity, and needs systems.
    Ensures mood influences emotional responses and conversely that
    emotional events appropriately update mood state.
    """
    
    def __init__(self, 
                mood_manager=None,
                emotional_core=None, 
                identity_evolution=None,
                needs_system=None):
        """Initialize the mood-emotional bridge."""
        self.mood_manager = mood_manager
        self.emotional_core = emotional_core
        self.identity_evolution = identity_evolution
        self.needs_system = needs_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.mood_to_emotion_influence = 0.4  # How strongly mood affects emotion
        self.emotion_to_mood_influence = 0.15  # How strongly emotions affect mood
        self.update_interval_seconds = 60  # How often to sync
        
        # Integration state tracking
        self.last_sync_time = datetime.datetime.now()
        self._subscribed = False
        
        logger.info("MoodEmotionalBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("significant_event", self._handle_significant_event)
                self.event_bus.subscribe("need_state_change", self._handle_need_change)
                self._subscribed = True
            
            logger.info("MoodEmotionalBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing MoodEmotionalBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="MoodEmotional")
    async def sync_mood_emotional_state(self) -> Dict[str, Any]:
        """
        Synchronize mood and emotional state bidirectionally.
        """
        if not self.mood_manager or not self.emotional_core:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get current mood state
            mood_state = await self.mood_manager.get_current_mood()
            
            # 2. Get current emotional state
            if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                emotional_state = await self.emotional_core.get_emotional_state_matrix()
            else:
                # Fallback to simpler method if available
                emotional_state = self.emotional_core.get_emotional_state()
            
            # 3. Transfer mood information to system context
            self.system_context.affective_state.valence = mood_state.valence
            self.system_context.affective_state.arousal = mood_state.arousal
            self.system_context.affective_state.dominance = mood_state.control
            self.system_context.affective_state.mood = mood_state.dominant_mood
            
            # 4. Update emotional system based on mood (if significant time passed)
            if (datetime.datetime.now() - self.last_sync_time).total_seconds() > self.update_interval_seconds:
                # Only influence long-term neurochemicals, not immediate emotions
                if hasattr(self.emotional_core, 'update_neurochemical'):
                    # Mood influences baseline neurochemical levels
                    influence = self.mood_to_emotion_influence
                    
                    # Update seranix (serotonin) based on mood valence
                    seranix_change = mood_state.valence * influence * 0.5
                    await self.emotional_core.update_neurochemical("seranix", seranix_change)
                    
                    # Update adrenyx (adrenaline) based on mood arousal
                    adrenyx_change = (mood_state.arousal - 0.5) * influence
                    await self.emotional_core.update_neurochemical("adrenyx", adrenyx_change)
                    
                # Update identity if available
                if self.identity_evolution and mood_state.intensity > 0.7:
                    # Strong moods influence identity
                    influence = mood_state.intensity * 0.05
                    
                    # Update traits based on mood
                    if mood_state.valence > 0.4 and mood_state.arousal > 0.6:
                        # Positive, high energy mood
                        await self.identity_evolution.update_trait("enthusiasm", influence)
                    elif mood_state.valence > 0.4 and mood_state.arousal < 0.4:
                        # Positive, low energy mood
                        await self.identity_evolution.update_trait("serenity", influence)
                    elif mood_state.valence < -0.4 and mood_state.arousal > 0.6:
                        # Negative, high energy mood
                        await self.identity_evolution.update_trait("vigilance", influence)
                    elif mood_state.valence < -0.4 and mood_state.arousal < 0.4:
                        # Negative, low energy mood
                        await self.identity_evolution.update_trait("reflection", influence)
                
                self.last_sync_time = datetime.datetime.now()
            
            return {
                "status": "success",
                "mood_state": {
                    "dominant_mood": mood_state.dominant_mood,
                    "valence": mood_state.valence,
                    "arousal": mood_state.arousal,
                    "control": mood_state.control
                },
                "emotional_state": {
                    "primary_emotion": emotional_state.get("primary_emotion", "unknown"),
                    "valence": emotional_state.get("valence", 0),
                    "arousal": emotional_state.get("arousal", 0.5)
                }
            }
        except Exception as e:
            logger.error(f"Error synchronizing mood and emotional state: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        if not self.mood_manager:
            return
            
        try:
            # Extract data from event
            emotion = event.data.get("emotion")
            valence = event.data.get("valence", 0.0)
            arousal = event.data.get("arousal", 0.5)
            intensity = event.data.get("intensity", 0.5)
            
            # Only significant emotions affect mood
            if intensity > 0.5:
                # Apply changes proportional to intensity and influence factor
                valence_change = valence * intensity * self.emotion_to_mood_influence
                arousal_change = (arousal - 0.5) * intensity * self.emotion_to_mood_influence
                
                # For control/dominance, only certain emotions affect it
                control_change = 0.0
                dominance_emotions = ["dominant", "confident", "proud", "powerful"]
                submission_emotions = ["submissive", "intimidated", "overwhelmed", "weak"]
                
                if emotion in dominance_emotions:
                    control_change = 0.1 * intensity * self.emotion_to_mood_influence
                elif emotion in submission_emotions:
                    control_change = -0.1 * intensity * self.emotion_to_mood_influence
                
                # Update mood
                await self.mood_manager.modify_mood(
                    valence_change=valence_change,
                    arousal_change=arousal_change, 
                    control_change=control_change,
                    reason=f"emotional_update:{emotion}"
                )
        except Exception as e:
            logger.error(f"Error handling emotional change: {e}")
    
    async def _handle_significant_event(self, event: Event) -> None:
        """
        Handle significant events that should impact mood.
        
        Args:
            event: Significant event
        """
        if not self.mood_manager:
            return
            
        try:
            # Extract data from event
            event_type = event.data.get("event_type")
            valence = event.data.get("valence", 0.0)
            intensity = event.data.get("intensity", 0.5)
            
            # Pass to mood manager
            await self.mood_manager.handle_significant_event(
                event_type=event_type,
                intensity=intensity,
                valence=valence
            )
        except Exception as e:
            logger.error(f"Error handling significant event: {e}")
    
    async def _handle_need_change(self, event: Event) -> None:
        """
        Handle need state changes that should affect mood.
        
        Args:
            event: Need state change event
        """
        if not self.mood_manager:
            return
            
        try:
            # Extract data from event
            need_name = event.data.get("need_name")
            drive_strength = event.data.get("drive_strength", 0.0)
            
            # Only significant need deficits affect mood
            if drive_strength > 0.7:
                # High drive strength = significant deficit = negative valence
                valence_change = -0.1 * (drive_strength - 0.5)
                
                # Specific needs affect different aspects of mood
                if need_name in ["knowledge", "novelty"]:
                    # Cognitive needs increase arousal when unfulfilled
                    arousal_change = 0.1
                elif need_name in ["connection", "intimacy"]:
                    # Social needs decrease arousal when unfulfilled (withdrawal)
                    arousal_change = -0.1
                elif need_name in ["agency", "control_expression"]:
                    # Agency needs decrease control when unfulfilled
                    control_change = -0.1
                    await self.mood_manager.modify_mood(
                        valence_change=valence_change,
                        control_change=control_change,
                        reason=f"need_deficit:{need_name}"
                    )
                    return
                else:
                    # Default
                    arousal_change = 0.0
                
                # Update mood
                await self.mood_manager.modify_mood(
                    valence_change=valence_change,
                    arousal_change=arousal_change,
                    reason=f"need_deficit:{need_name}"
                )
        except Exception as e:
            logger.error(f"Error handling need change: {e}")

# Function to create the bridge
def create_mood_emotional_bridge(nyx_brain):
    """Create a mood-emotional bridge for the given brain."""
    return MoodEmotionalBridge(
        mood_manager=nyx_brain.mood_manager if hasattr(nyx_brain, "mood_manager") else None,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        identity_evolution=nyx_brain.identity_evolution if hasattr(nyx_brain, "identity_evolution") else None,
        needs_system=nyx_brain.needs_system if hasattr(nyx_brain, "needs_system") else None
    )
