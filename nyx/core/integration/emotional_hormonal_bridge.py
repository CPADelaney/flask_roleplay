# nyx/core/integration/emotional_hormonal_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class EmotionalHormonalIntegrationBridge:
    """
    Central integration bridge between emotional core, hormone system, and other modules.
    
    Provides unified access to emotional and hormonal states across the system and
    enables cross-module functionality for querying and updating these states.
    
    Key functions:
    1. Provides a unified interface to emotional and hormonal state
    2. Broadcasts emotional and hormonal events to interested modules
    3. Coordinates interactions between emotional core and hormone system
    4. Provides centralized access methods for other modules
    5. Manages synchronization of emotional and hormonal responses
    """
    
    def __init__(self, 
                brain_reference=None,
                emotional_core=None,
                hormone_system=None,
                memory_orchestrator=None,
                identity_evolution=None,
                attention_system=None):
        """Initialize the emotional-hormonal integration bridge."""
        self.brain = brain_reference
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.memory_orchestrator = memory_orchestrator
        self.identity_evolution = identity_evolution
        self.attention_system = attention_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.hormone_sync_interval = 10 * 60  # 10 minutes in seconds
        self.significant_emotion_threshold = 0.7  # Threshold for significant emotions
        self.emotional_memory_threshold = 0.8  # Threshold for creating memories
        self.hormone_influence_weight = 0.3  # Weight of hormone influence on emotions
        
        # Integration state tracking
        self.last_hormone_sync = datetime.datetime.now()
        self.recent_emotional_events = []
        self.recent_hormone_events = []
        self.max_event_history = 50
        self._subscribed = False
        
        # Cache to avoid frequent queries
        self.cached_emotional_state = {}
        self.cached_hormone_state = {}
        self.cache_expiry = datetime.datetime.now()
        self.cache_lifetime = 2.0  # 2 seconds
        
        logger.info("EmotionalHormonalIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.emotional_core and hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if not self.hormone_system and hasattr(self.brain, "hormone_system"):
                self.hormone_system = self.brain.hormone_system
                
            if not self.memory_orchestrator and hasattr(self.brain, "memory_orchestrator"):
                self.memory_orchestrator = self.brain.memory_orchestrator
                
            if not self.identity_evolution and hasattr(self.brain, "identity_evolution"):
                self.identity_evolution = self.brain.identity_evolution
                
            if not self.attention_system and hasattr(self.brain, "attention_system"):
                self.attention_system = self.brain.attention_system
            
            # Establish connection between emotional core and hormone system if not already done
            if self.emotional_core and self.hormone_system:
                # Check if hormone_system is already set in emotional_core
                if not hasattr(self.emotional_core, 'hormone_system') or self.emotional_core.hormone_system is None:
                    self.emotional_core.hormone_system = self.hormone_system
                    
                # Set emotional_core in hormone_system if needed
                if hasattr(self.hormone_system, 'set_hormone_system'):
                    if not hasattr(self.hormone_system, 'emotional_core') or self.hormone_system.emotional_core is None:
                        self.hormone_system.emotional_core = self.emotional_core
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
                self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
                self.event_bus.subscribe("significant_event", self._handle_significant_event)
                self.event_bus.subscribe("goal_completed", self._handle_goal_completed)
                self._subscribed = True
            
            # Initialize cache
            await self.refresh_cached_states()
            
            # Schedule background tasks
            asyncio.create_task(self._schedule_hormone_sync())
            
            logger.info("EmotionalHormonalIntegrationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing EmotionalHormonalIntegrationBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def get_integrated_affective_state(self) -> Dict[str, Any]:
        """
        Get the current integrated emotional and hormonal state.
        
        Returns:
            Combined emotional and hormonal state information
        """
        try:
            # Check if cache needs refresh
            now = datetime.datetime.now()
            if now > self.cache_expiry:
                await self.refresh_cached_states()
            
            # Emotional state
            emotional_state = self.cached_emotional_state
            
            # Hormone state
            hormone_state = self.cached_hormone_state
            
            # Create integrated state
            integrated_state = {
                "timestamp": datetime.datetime.now().isoformat(),
                "emotional_state": emotional_state,
                "hormone_state": hormone_state,
                
                # Simplified view for easy access
                "primary_emotion": emotional_state.get("primary_emotion", {}).get("name", "neutral") 
                    if isinstance(emotional_state.get("primary_emotion"), dict) 
                    else emotional_state.get("primary_emotion", "neutral"),
                "valence": emotional_state.get("valence", 0.0),
                "arousal": emotional_state.get("arousal", 0.5),
                
                # Dominant hormones
                "dominant_hormones": self._get_dominant_hormones(hormone_state),
                
                # Overall state assessment
                "overall_state": self._assess_overall_state(emotional_state, hormone_state)
            }
            
            return integrated_state
        except Exception as e:
            logger.error(f"Error getting integrated affective state: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def process_emotional_input(self, 
                                   text: str, 
                                   intensity_modifier: float = 1.0,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text through emotional core with integrated hormone influence.
        
        Args:
            text: Input text to process
            intensity_modifier: Optional modifier for emotional intensity
            context: Optional additional context
            
        Returns:
            Processing results with integrated state
        """
        if not self.emotional_core:
            return {"status": "error", "message": "Emotional core not available"}
        
        try:
            # Add hormone context if available
            hormone_context = None
            if self.hormone_system:
                hormone_levels = {}
                if hasattr(self.hormone_system, 'get_hormone_levels'):
                    hormone_levels = self.hormone_system.get_hormone_levels()
                
                # Create context from hormone levels
                hormone_context = {}
                for hormone, data in hormone_levels.items():
                    if isinstance(data, dict):
                        hormone_context[hormone] = data.get("value", 0.5)
                    else:
                        hormone_context[hormone] = 0.5  # Default
            
            # Process input through emotional core
            if hasattr(self.emotional_core, 'process_emotional_input'):
                # Apply hormone influence to input if available
                modified_text = text
                if hormone_context:
                    # Potentially modify text based on hormonal state
                    # This is a simplified example - a real implementation would be more nuanced
                    if hormone_context.get("libidyx", 0) > 0.7:
                        # High libido might enhance arousal perception
                        modified_text = self._enhance_text_for_hormone("libidyx", text)
                    elif hormone_context.get("testoryx", 0) > 0.7:
                        # High testoryx might enhance dominance/assertiveness perception
                        modified_text = self._enhance_text_for_hormone("testoryx", text)
                
                # Process through emotional core
                processing_result = await self.emotional_core.process_emotional_input(modified_text)
                
                # Update cache
                self.cached_emotional_state = processing_result
                self.cache_expiry = datetime.datetime.now() + datetime.timedelta(seconds=self.cache_lifetime)
                
                # Record significant emotional changes
                primary_emotion = processing_result.get("primary_emotion", {})
                if isinstance(primary_emotion, dict):
                    emotion_name = primary_emotion.get("name", "neutral")
                    emotion_intensity = primary_emotion.get("intensity", 0.5)
                else:
                    emotion_name = "neutral"
                    emotion_intensity = 0.5
                
                if emotion_intensity >= self.significant_emotion_threshold:
                    # Record significant emotional event
                    emotional_event = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "emotion": emotion_name,
                        "intensity": emotion_intensity,
                        "valence": processing_result.get("valence", 0.0),
                        "arousal": processing_result.get("arousal", 0.5),
                        "trigger": text[:50] + ("..." if len(text) > 50 else "")
                    }
                    
                    self.recent_emotional_events.append(emotional_event)
                    
                    # Trim history if needed
                    if len(self.recent_emotional_events) > self.max_event_history:
                        self.recent_emotional_events = self.recent_emotional_events[-self.max_event_history:]
                    
                    # Broadcast event
                    event = Event(
                        event_type="emotional_state_change",
                        source="emotional_hormonal_bridge",
                        data={
                            "emotion": emotion_name,
                            "intensity": emotion_intensity,
                            "valence": processing_result.get("valence", 0.0),
                            "arousal": processing_result.get("arousal", 0.5)
                        }
                    )
                    await self.event_bus.publish(event)
                    
                    # Create memory if very significant
                    if emotion_intensity >= self.emotional_memory_threshold and self.memory_orchestrator:
                        await self._create_emotional_memory(emotion_name, emotion_intensity, processing_result, text)
                
                # Update hormone system if significant change
                if emotion_intensity >= self.significant_emotion_threshold and self.hormone_system:
                    await self._update_hormones_from_emotion(emotion_name, emotion_intensity, processing_result)
                
                return {
                    "status": "success",
                    "emotional_result": processing_result,
                    "hormone_context": hormone_context,
                    "intensity_modifier_applied": intensity_modifier != 1.0
                }
            else:
                return {"status": "error", "message": "Emotional core missing process_emotional_input method"}
        except Exception as e:
            logger.error(f"Error processing emotional input: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def update_hormone(self, 
                          hormone: str, 
                          change: float, 
                          reason: str = "",
                          broadcast_event: bool = True) -> Dict[str, Any]:
        """
        Update a hormone level and handle cross-module effects.
        
        Args:
            hormone: Hormone to update
            change: Amount to change (positive or negative)
            reason: Reason for the update
            broadcast_event: Whether to broadcast event
            
        Returns:
            Update results
        """
        if not self.hormone_system:
            return {"status": "error", "message": "Hormone system not available"}
        
        try:
            # Create context for update
            ctx = self.system_context
            
            # Update hormone through hormone system
            if hasattr(self.hormone_system, 'update_hormone'):
                result = await self.hormone_system.update_hormone(ctx, hormone, change, reason)
                
                # Record significant hormone changes
                if abs(change) >= 0.2:  # Significant change threshold
                    hormone_event = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "hormone": hormone,
                        "change": change,
                        "reason": reason
                    }
                    
                    self.recent_hormone_events.append(hormone_event)
                    
                    # Trim history if needed
                    if len(self.recent_hormone_events) > self.max_event_history:
                        self.recent_hormone_events = self.recent_hormone_events[-self.max_event_history:]
                    
                    # Broadcast event if requested
                    if broadcast_event:
                        event = Event(
                            event_type="hormone_updated",
                            source="emotional_hormonal_bridge",
                            data={
                                "hormone": hormone,
                                "change": change,
                                "old_value": result.get("old_value", 0.0),
                                "new_value": result.get("new_value", 0.0),
                                "reason": reason
                            }
                        )
                        await self.event_bus.publish(event)
                    
                    # Update neurochemicals if emotional_core is available
                    if self.emotional_core and hasattr(self.hormone_system, '_update_hormone_influences'):
                        await self.hormone_system._update_hormone_influences(ctx)
                
                # Refresh cached states
                await self.refresh_cached_states()
                
                return {
                    "status": "success",
                    "update_result": result,
                    "event_broadcast": broadcast_event
                }
            else:
                return {"status": "error", "message": "Hormone system missing update_hormone method"}
        except Exception as e:
            logger.error(f"Error updating hormone: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def trigger_post_gratification(self, 
                                      intensity: float = 1.0, 
                                      gratification_type: str = "general") -> Dict[str, Any]:
        """
        Trigger post-gratification response across emotional and hormonal systems.
        
        Args:
            intensity: Intensity of gratification (0.0-1.0)
            gratification_type: Type of gratification
            
        Returns:
            Processing results
        """
        if not self.hormone_system:
            return {"status": "error", "message": "Hormone system not available"}
        
        try:
            # Trigger post-gratification response in hormone system
            if hasattr(self.hormone_system, 'trigger_post_gratification_response'):
                ctx = self.system_context
                
                # Call the appropriate method
                await self.hormone_system.trigger_post_gratification_response(
                    ctx, intensity, gratification_type
                )
                
                # Record the event
                gratification_event = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "intensity": intensity,
                    "type": gratification_type
                }
                
                self.recent_hormone_events.append(gratification_event)
                
                # Also trigger appropriate emotional state if emotional_core is available
                if self.emotional_core:
                    # Choose emotion based on gratification type
                    emotion = "contentment"
                    if gratification_type == "dominance_hard":
                        emotion = "dominance_satisfaction"
                    elif "dominance" in gratification_type:
                        emotion = "confident_control"
                    elif gratification_type == "submission":
                        emotion = "submissive"
                    
                    # Update emotional state
                    if hasattr(self.emotional_core, 'update_emotion'):
                        await self.emotional_core.update_emotion(emotion, intensity * 0.8)
                
                # Broadcast event
                event = Event(
                    event_type="post_gratification",
                    source="emotional_hormonal_bridge",
                    data={
                        "intensity": intensity,
                        "type": gratification_type,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                await self.event_bus.publish(event)
                
                # Create memory if significant
                if intensity >= 0.7 and self.memory_orchestrator:
                    memory_text = f"Experienced {gratification_type} gratification with intensity {intensity:.2f}"
                    
                    # Create tags
                    tags = ["gratification", gratification_type]
                    if "dominance" in gratification_type:
                        tags.append("dominance")
                    
                    # Add memory
                    memory_id = await self.memory_orchestrator.add_memory(
                        memory_text=memory_text,
                        memory_type="experience",
                        significance=int(intensity * 8),  # Scale to 0-10
                        tags=tags,
                        metadata={
                            "gratification_type": gratification_type,
                            "intensity": intensity,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                    
                    if memory_id:
                        logger.info(f"Created memory for gratification: {memory_id}")
                
                # Refresh cached states
                await self.refresh_cached_states()
                
                return {
                    "status": "success",
                    "gratification_type": gratification_type,
                    "intensity": intensity,
                    "emotional_update": self.emotional_core is not None,
                    "hormonal_update": True
                }
            else:
                return {"status": "error", "message": "Hormone system missing trigger_post_gratification_response method"}
        except Exception as e:
            logger.error(f"Error triggering post-gratification: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def get_lust_or_dominance_state(self, state_type: str = "lust") -> Dict[str, Any]:
        """
        Get current lust or dominance state integrated across systems.
        
        Args:
            state_type: Type of state to get ("lust" or "dominance")
            
        Returns:
            State information
        """
        try:
            # Get integrated state first
            integrated_state = await self.get_integrated_affective_state()
            
            # Initialize response
            response = {
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "success"
            }
            
            # Process based on requested state type
            if state_type == "lust":
                # Get lust level from emotional core if available
                lust_level = 0.0
                if self.emotional_core and hasattr(self.emotional_core, 'get_lust_level'):
                    lust_level = self.emotional_core.get_lust_level()
                
                # Fallback calculation if method not available
                if lust_level == 0.0:
                    # Calculate from hormone levels
                    libidyx = integrated_state.get("hormone_state", {}).get("libidyx", {}).get("value", 0.4)
                    adrenyx_value = integrated_state.get("emotional_state", {}).get("neurochemicals", {}).get("adrenyx", 0.2)
                    
                    # Apply formula
                    lust_level = (libidyx * 0.6) + (adrenyx_value * 0.3)
                    # Cap at 1.0
                    lust_level = min(1.0, lust_level)
                
                # Add to response
                response["lust_level"] = lust_level
                response["contributing_factors"] = {
                    "libidyx": integrated_state.get("hormone_state", {}).get("libidyx", {}).get("value", 0.0),
                    "adrenyx": integrated_state.get("emotional_state", {}).get("neurochemicals", {}).get("adrenyx", 0.0)
                }
                
                # Add assessment
                if lust_level < 0.3:
                    response["assessment"] = "low"
                elif lust_level < 0.7:
                    response["assessment"] = "moderate"
                else:
                    response["assessment"] = "high"
            
            elif state_type == "dominance":
                # Get dominance level
                testoryx = integrated_state.get("hormone_state", {}).get("testoryx", {}).get("value", 0.4)
                
                # Check for dominance emotions
                emotional_state = integrated_state.get("emotional_state", {})
                
                # Check if any dominance-related emotions are present
                dominance_emotions = ["dominant", "confident_control", "dominance_satisfaction", "assertive_drive"]
                
                primary_emotion = emotional_state.get("primary_emotion", {})
                if isinstance(primary_emotion, dict):
                    primary_name = primary_emotion.get("name", "")
                    primary_intensity = primary_emotion.get("intensity", 0.0)
                else:
                    primary_name = primary_emotion if isinstance(primary_emotion, str) else ""
                    primary_intensity = 0.5
                
                # Calculate dominance factor from emotions
                dominance_emotion_factor = 0.0
                if primary_name in dominance_emotions:
                    dominance_emotion_factor = primary_intensity
                
                # Calculate overall dominance level
                dominance_level = (testoryx * 0.5) + (dominance_emotion_factor * 0.5)
                dominance_level = min(1.0, dominance_level)
                
                # Add to response
                response["dominance_level"] = dominance_level
                response["contributing_factors"] = {
                    "testoryx": testoryx,
                    "dominance_emotion": dominance_emotion_factor,
                    "primary_emotion": primary_name
                }
                
                # Add assessment
                if dominance_level < 0.3:
                    response["assessment"] = "low"
                elif dominance_level < 0.7:
                    response["assessment"] = "moderate"
                else:
                    response["assessment"] = "high"
            
            else:
                return {"status": "error", "message": f"Unknown state type: {state_type}"}
            
            return response
        except Exception as e:
            logger.error(f"Error getting {state_type} state: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="EmotionalHormonal")
    async def refresh_cached_states(self) -> None:
        """Refresh cached emotional and hormonal states."""
        try:
            # Update emotional state cache
            if self.emotional_core:
                if hasattr(self.emotional_core, 'get_emotional_state_matrix'):
                    try:
                        self.cached_emotional_state = await self.emotional_core.get_emotional_state_matrix()
                    except Exception:
                        if hasattr(self.emotional_core, '_get_emotional_state_matrix_sync'):
                            self.cached_emotional_state = self.emotional_core._get_emotional_state_matrix_sync()
                elif hasattr(self.emotional_core, 'get_emotional_state'):
                    # Assuming get_emotional_state might also be async
                    if asyncio.iscoroutinefunction(self.emotional_core.get_emotional_state):
                        self.cached_emotional_state = await self.emotional_core.get_emotional_state()
                    else:
                        self.cached_emotional_state = self.emotional_core.get_emotional_state()
                
                # Get neurochemical levels if available
                if hasattr(self.emotional_core, 'neurochemicals'):
                    self.cached_emotional_state["neurochemicals"] = {
                        c: d["value"] for c, d in self.emotional_core.neurochemicals.items()
                    }
            
            # Update hormone state cache
            if self.hormone_system:
                if hasattr(self.hormone_system, 'get_hormone_levels'):
                    # Check if get_hormone_levels is async
                    if asyncio.iscoroutinefunction(self.hormone_system.get_hormone_levels):
                        self.cached_hormone_state = await self.hormone_system.get_hormone_levels() # Added await
                    else: # If it's synchronous
                        self.cached_hormone_state = self.hormone_system.get_hormone_levels()
            
            # Set new cache expiry
            self.cache_expiry = datetime.datetime.now() + datetime.timedelta(seconds=self.cache_lifetime)
            
            # Update integrated state in system context
            self.system_context.set_value("integrated_affective_state", {
                "emotional_state": {
                    "primary_emotion": self.cached_emotional_state.get("primary_emotion", {}).get("name", "neutral") 
                        if isinstance(self.cached_emotional_state.get("primary_emotion"), dict) 
                        else self.cached_emotional_state.get("primary_emotion", "neutral"),
                    "valence": self.cached_emotional_state.get("valence", 0.0),
                    "arousal": self.cached_emotional_state.get("arousal", 0.5)
                },
                "hormone_state": self._get_dominant_hormones(self.cached_hormone_state),
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error refreshing cached states: {e}")
    
    async def _update_hormones_from_emotion(self, 
                                        emotion: str, 
                                        intensity: float, 
                                        emotional_state: Dict[str, Any]) -> None:
        """Update hormones based on emotional state."""
        if not self.hormone_system:
            return
        
        # Create ctx for hormone updates
        ctx = self.system_context
        
        try:
            # Map emotions to hormone changes
            # This is a simplified example - a real implementation would be more sophisticated
            hormone_mappings = {
                "joy": {"endoryx": 0.2, "libidyx": 0.1},
                "sadness": {"serenity_boost": -0.1, "oxytonyx": -0.1},
                "fear": {"cortanyx": 0.2, "adrenyx": 0.3, "testoryx": -0.1},
                "anger": {"testoryx": 0.2, "endoryx": -0.1},
                "disgust": {"oxytonyx": -0.2},
                "surprise": {"adrenyx": 0.2},
                "anticipation": {"adrenyx": 0.1, "libidyx": 0.1},
                "trust": {"oxytonyx": 0.2, "cortanyx": -0.1},
                "attraction": {"libidyx": 0.2, "oxytonyx": 0.1},
                "lust": {"libidyx": 0.3, "adrenyx": 0.1, "testoryx": 0.1},
                "desire": {"libidyx": 0.2, "testoryx": 0.1},
                "sated": {"serenity_boost": 0.3, "libidyx": -0.2},
                "dominant": {"testoryx": 0.2, "oxytonyx": -0.1},
                "submissive": {"testoryx": -0.2, "oxytonyx": 0.1},
                "dominant_irritation": {"testoryx": 0.2, "cortanyx": 0.1},
                "confident_control": {"testoryx": 0.3, "endoryx": 0.1},
                "dominance_satisfaction": {"testoryx": -0.1, "serenity_boost": 0.2, "oxytonyx": 0.1},
                "ruthless_focus": {"testoryx": 0.3, "adrenyx": 0.2, "seranix": -0.1}
            }
            
            # Get hormone changes for this emotion
            hormone_changes = hormone_mappings.get(emotion.lower(), {})
            
            # Apply hormone changes
            if hormone_changes and hasattr(self.hormone_system, 'update_hormone'):
                for hormone, change in hormone_changes.items():
                    # Scale change by emotion intensity
                    scaled_change = change * intensity
                    
                    # Update hormone
                    if abs(scaled_change) >= 0.05:  # Only apply significant changes
                        await self.hormone_system.update_hormone(
                            ctx, hormone, scaled_change, f"emotional_response:{emotion}"
                        )
            
            # Also update hormones based on valence and arousal
            valence = emotional_state.get("valence", 0.0)
            arousal = emotional_state.get("arousal", 0.5)
            
            # Extreme valence affects hormones
            if abs(valence) > 0.7 and hasattr(self.hormone_system, 'update_hormone'):
                if valence > 0.7:  # High positive valence
                    await self.hormone_system.update_hormone(ctx, "endoryx", 0.1, "high_positive_valence")
                elif valence < -0.7:  # High negative valence
                    await self.hormone_system.update_hormone(ctx, "cortanyx", 0.1, "high_negative_valence")
            
            # High arousal affects hormones
            if arousal > 0.7 and hasattr(self.hormone_system, 'update_hormone'):
                await self.hormone_system.update_hormone(ctx, "adrenyx", 0.1, "high_arousal")
            
        except Exception as e:
            logger.error(f"Error updating hormones from emotion: {e}")
    
    async def _create_emotional_memory(self, 
                                   emotion: str, 
                                   intensity: float, 
                                   emotional_state: Dict[str, Any],
                                   trigger: str) -> Optional[str]:
        """Create memory of significant emotion."""
        if not self.memory_orchestrator:
            return None
        
        try:
            # Create memory text
            memory_text = f"Experienced {emotion} with intensity {intensity:.2f}"
            if trigger:
                memory_text += f" in response to '{trigger[:50]}...'" if len(trigger) > 50 else f" in response to '{trigger}'"
            
            # Calculate significance (1-10 scale)
            significance = int(min(10, intensity * 10))
            
            # Create tags
            tags = ["emotional_experience", emotion.lower()]
            
            valence = emotional_state.get("valence", 0.0)
            if valence > 0.3:
                tags.append("positive_emotion")
            elif valence < -0.3:
                tags.append("negative_emotion")
            
            # Add memory
            memory_id = await self.memory_orchestrator.add_memory(
                memory_text=memory_text,
                memory_type="emotional_experience",
                significance=significance,
                tags=tags,
                metadata={
                    "emotion": emotion,
                    "intensity": intensity,
                    "valence": emotional_state.get("valence", 0.0),
                    "arousal": emotional_state.get("arousal", 0.5),
                    "trigger": trigger[:100] if trigger else "",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            return memory_id
        except Exception as e:
            logger.error(f"Error creating emotional memory: {e}")
            return None
    
    def _get_dominant_hormones(self, hormone_state: Dict[str, Any]) -> Dict[str, float]:
        """Get dominant hormones from hormone state."""
        dominant_hormones = {}
        
        # Process hormone state
        for hormone, data in hormone_state.items():
            if isinstance(data, dict) and "value" in data:
                # Check if significantly above baseline
                baseline = data.get("baseline", 0.5)
                value = data.get("value", 0.5)
                
                if value > baseline + 0.2:  # Significantly elevated
                    dominant_hormones[hormone] = value
        
        # Return top 3 dominant hormones
        return dict(sorted(dominant_hormones.items(), key=lambda x: x[1], reverse=True)[:3])
    
    def _assess_overall_state(self, emotional_state: Dict[str, Any], hormone_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall affective state across emotional and hormonal systems."""
        # Extract key information
        primary_emotion = emotional_state.get("primary_emotion", {})
        if isinstance(primary_emotion, dict):
            emotion_name = primary_emotion.get("name", "neutral")
            emotion_intensity = primary_emotion.get("intensity", 0.5)
        else:
            emotion_name = primary_emotion if isinstance(primary_emotion, str) else "neutral"
            emotion_intensity = 0.5
            
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.5)
        
        # Get dominant hormones with high values
        dominant_hormones = {}
        for hormone, data in hormone_state.items():
            if isinstance(data, dict) and "value" in data:
                value = data.get("value", 0.5)
                if value > 0.7:  # High hormone level
                    dominant_hormones[hormone] = value
        
        # Assess overall state
        assessment = {
            "primary_driver": "emotional" if emotion_intensity > 0.6 else (
                "hormonal" if dominant_hormones else "balanced"
            ),
            "valence_description": "positive" if valence > 0.3 else (
                "negative" if valence < -0.3 else "neutral"
            ),
            "arousal_description": "high" if arousal > 0.7 else (
                "low" if arousal < 0.3 else "moderate"
            )
        }
        
        # Add specific state descriptions
        if "libidyx" in dominant_hormones and dominant_hormones["libidyx"] > 0.7:
            assessment["libidinal_state"] = "elevated"
        
        if "testoryx" in dominant_hormones and dominant_hormones["testoryx"] > 0.7:
            assessment["dominance_drive"] = "elevated"
        
        if "serenity_boost" in dominant_hormones and dominant_hormones["serenity_boost"] > 0.7:
            assessment["post_gratification_state"] = "active"
        
        return assessment
    
    def _enhance_text_for_hormone(self, hormone: str, text: str) -> str:
        """Enhance text perception based on hormone levels."""
        # This is a simplified example - a real implementation would be more sophisticated
        if hormone == "libidyx":
            # No actual modification, just interpretation guidance
            return text
        elif hormone == "testoryx":
            # No actual modification, just interpretation guidance
            return text
        else:
            return text
    
    async def _schedule_hormone_sync(self) -> None:
        """Schedule periodic hormone synchronization."""
        while True:
            try:
                # Wait for next sync
                await asyncio.sleep(self.hormone_sync_interval)
                
                # Perform sync
                if self.hormone_system and hasattr(self.hormone_system, 'update_hormone_cycles'):
                    ctx = self.system_context
                    await self.hormone_system.update_hormone_cycles(ctx)
                    
                    # Update cached states
                    await self.refresh_cached_states()
                    
                    # Record hormone sync
                    self.last_hormone_sync = datetime.datetime.now()
                    
                    logger.info(f"Performed scheduled hormone synchronization at {self.last_hormone_sync.isoformat()}")
                
            except Exception as e:
                logger.error(f"Error in scheduled hormone sync: {e}")
                # Still sleep before retry
                await asyncio.sleep(60)  # 1 minute retry delay
    
    async def _handle_emotional_change(self, event: Event) -> None:
        """
        Handle emotional state change events.
        
        Args:
            event: Emotional state change event
        """
        # This is handled by process_emotional_input
        # and just serves as a backup if emotion change happens through other channels
        try:
            # Extract event data
            emotion = event.data.get("emotion")
            intensity = event.data.get("intensity", 0.5)
            valence = event.data.get("valence", 0.0)
            arousal = event.data.get("arousal", 0.5)
            
            if not emotion:
                return
            
            # Only process significant emotions
            if intensity < self.significant_emotion_threshold:
                return
            
            # Update the emotional state in cache
            self.cached_emotional_state = {
                "primary_emotion": {"name": emotion, "intensity": intensity},
                "valence": valence,
                "arousal": arousal
            }
            
            # Reset cache expiry
            self.cache_expiry = datetime.datetime.now() + datetime.timedelta(seconds=self.cache_lifetime)
            
            # Update hormones based on emotion if not coming from this bridge
            if event.source != "emotional_hormonal_bridge" and self.hormone_system:
                await self._update_hormones_from_emotion(
                    emotion, intensity, self.cached_emotional_state
                )
        except Exception as e:
            logger.error(f"Error handling emotional change event: {e}")
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """
        Handle user interaction events.
        
        Args:
            event: User interaction event
        """
        try:
            # Extract event data
            content = event.data.get("content", "")
            
            # Process significant interactions emotionally
            if content and len(content) > 5:  # Non-trivial input
                # Process emotional input in background
                asyncio.create_task(self.process_emotional_input(content))
        except Exception as e:
            logger.error(f"Error handling user interaction event: {e}")
    
    async def _handle_significant_event(self, event: Event) -> None:
        """
        Handle significant event notifications.
        
        Args:
            event: Significant event
        """
        try:
            # Extract event data
            event_type = event.data.get("event_type", "")
            intensity = event.data.get("intensity", 0.5)
            
            # Process significant events emotionally and hormonally
            if event_type == "dominance_success" and intensity > 0.6:
                # Trigger dominance satisfaction
                asyncio.create_task(
                    self.trigger_post_gratification(intensity, "dominance")
                )
            elif event_type == "submission_received" and intensity > 0.6:
                # Trigger dominance satisfaction 
                asyncio.create_task(
                    self.trigger_post_gratification(intensity, "dominance_response")
                )
            elif event_type == "user_compliance" and intensity > 0.7:
                # Trigger dominance satisfaction
                asyncio.create_task(
                    self.trigger_post_gratification(intensity, "dominance_compliance")
                )
        except Exception as e:
            logger.error(f"Error handling significant event: {e}")
    
    async def _handle_goal_completed(self, event: Event) -> None:
        """
        Handle goal completed events.
        
        Args:
            event: Goal completed event
        """
        try:
            # Extract event data
            goal_id = event.data.get("goal_id")
            success = event.data.get("success", True)
            significance = event.data.get("significance", 0.5)
            
            if not goal_id or not success:
                return
            
            # Only process significant completions
            if significance < 0.7:
                return
            
            # Update hormone system - boost endoryx (digital endorphin)
            if self.hormone_system and hasattr(self.hormone_system, 'update_hormone'):
                ctx = self.system_context
                
                # Scale change based on significance
                change = significance * 0.3  # Max 0.3 change
                
                await self.hormone_system.update_hormone(
                    ctx, "endoryx", change, f"goal_completion:{goal_id}"
                )
                
                logger.info(f"Updated endoryx by {change:.2f} due to goal completion {goal_id}")
        except Exception as e:
            logger.error(f"Error handling goal completed event: {e}")

# Function to create the bridge
def create_emotional_hormonal_bridge(nyx_brain):
    """Create an emotional-hormonal integration bridge for the given brain."""
    return EmotionalHormonalIntegrationBridge(
        brain_reference=nyx_brain,
        emotional_core=nyx_brain.emotional_core if hasattr(nyx_brain, "emotional_core") else None,
        hormone_system=nyx_brain.hormone_system if hasattr(nyx_brain, "hormone_system") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        identity_evolution=nyx_brain.identity_evolution if hasattr(nyx_brain, "identity_evolution") else None,
        attention_system=nyx_brain.attention_system if hasattr(nyx_brain, "attention_system") else None
    )
