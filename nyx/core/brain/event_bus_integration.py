# nyx/core/brain/event_bus_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from functools import wraps

from nyx.core.integration.event_bus import (
    get_event_bus, Event, EmotionalEvent, PhysicalSensationEvent,
    GoalEvent, NeedEvent, UserInteractionEvent, DominanceEvent,
    NarrativeEvent, ReasoningEvent, DecisionEvent, IntegrationStatusEvent,
    PredictionEvent, SimulationEvent, AttentionEvent, UserModelEvent,
    SpatialEvent, NavigationEvent, ConditioningEvent, ConditionedResponseEvent
)
from nyx.core.integration.system_context import get_system_context

logger = logging.getLogger(__name__)

class NyxEventBusIntegration:
    """
    Comprehensive event bus integration for NyxBrain.
    Connects all subsystems through event-driven communication.
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.subscriptions = []
        self._event_handlers = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize event bus subscriptions for all brain modules."""
        if self._initialized:
            logger.warning("Event bus integration already initialized")
            return
            
        logger.info("Initializing comprehensive event bus integration for NyxBrain")
        
        # Subscribe to events for each module
        await self._setup_emotional_subscriptions()
        await self._setup_memory_subscriptions()
        await self._setup_goal_subscriptions()
        await self._setup_need_subscriptions()
        await self._setup_reasoning_subscriptions()
        await self._setup_relationship_subscriptions()
        await self._setup_spatial_subscriptions()
        await self._setup_dominance_subscriptions()
        await self._setup_sensory_subscriptions()
        await self._setup_conditioning_subscriptions()
        await self._setup_identity_subscriptions()
        await self._setup_attention_subscriptions()
        await self._setup_prediction_subscriptions()
        await self._setup_creative_subscriptions()
        await self._setup_communication_subscriptions()
        await self._setup_metacognitive_subscriptions()
        
        # Setup module-specific publishers
        await self._setup_module_publishers()
        
        self._initialized = True
        logger.info("Event bus integration initialized successfully")
        
        # Publish initialization status
        await self.publish_integration_status("event_bus", "initialized")
    
    # ========== SUBSCRIPTION SETUP METHODS ==========
    
    async def _setup_emotional_subscriptions(self):
        """Setup subscriptions for emotional and mood systems."""
        
        # Emotional Core subscriptions
        if self.brain.emotional_core:
            # Subscribe to events that should trigger emotional changes
            self.event_bus.subscribe("user_interaction", 
                self._wrap_handler(self._handle_user_interaction_for_emotion), 
                "emotional_core")
            
            self.event_bus.subscribe("goal_status_change", 
                self._wrap_handler(self._handle_goal_change_for_emotion),
                "emotional_core")
            
            self.event_bus.subscribe("dominance_action",
                self._wrap_handler(self._handle_dominance_for_emotion),
                "emotional_core")
            
            self.event_bus.subscribe("physical_sensation",
                self._wrap_handler(self._handle_sensation_for_emotion),
                "emotional_core")
        
        # Mood Manager subscriptions
        if self.brain.mood_manager:
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._handle_emotion_for_mood),
                "mood_manager")
            
            self.event_bus.subscribe("need_state_change",
                self._wrap_handler(self._handle_need_for_mood),
                "mood_manager")
    
    async def _setup_memory_subscriptions(self):
        """Setup subscriptions for memory systems."""
        
        if self.brain.memory_core:
            # Subscribe to events that should be stored as memories
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._store_interaction_memory),
                "memory_core")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._store_emotional_memory),
                "memory_core")
            
            self.event_bus.subscribe("goal_status_change",
                self._wrap_handler(self._store_goal_memory),
                "memory_core")
            
            self.event_bus.subscribe("dominance_action",
                self._wrap_handler(self._store_dominance_memory),
                "memory_core")
            
            self.event_bus.subscribe("decision_made",
                self._wrap_handler(self._store_decision_memory),
                "memory_core")
    
    async def _setup_goal_subscriptions(self):
        """Setup subscriptions for goal management."""
        
        if self.brain.goal_manager:
            self.event_bus.subscribe("need_state_change",
                self._wrap_handler(self._handle_need_for_goals),
                "goal_manager")
            
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._check_goal_triggers),
                "goal_manager")
    
    async def _setup_need_subscriptions(self):
        """Setup subscriptions for needs system."""
        
        if self.brain.needs_system:
            self.event_bus.subscribe("goal_status_change",
                self._wrap_handler(self._handle_goal_for_needs),
                "needs_system")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._handle_emotion_for_needs),
                "needs_system")
            
            self.event_bus.subscribe("physical_sensation",
                self._wrap_handler(self._handle_sensation_for_needs),
                "needs_system")
    
    async def _setup_reasoning_subscriptions(self):
        """Setup subscriptions for reasoning systems."""
        
        if self.brain.reasoning_core:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._analyze_reasoning_need),
                "reasoning_core")
            
            self.event_bus.subscribe("decision_needed",
                self._wrap_handler(self._provide_reasoning_support),
                "reasoning_core")
    
    async def _setup_relationship_subscriptions(self):
        """Setup subscriptions for relationship management."""
        
        if self.brain.relationship_manager:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._update_relationship_from_interaction),
                "relationship_manager")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._update_relationship_from_emotion),
                "relationship_manager")
            
            self.event_bus.subscribe("dominance_action",
                self._wrap_handler(self._update_relationship_from_dominance),
                "relationship_manager")
    
    async def _setup_spatial_subscriptions(self):
        """Setup subscriptions for spatial systems."""
        
        if self.brain.spatial_mapper:
            self.event_bus.subscribe("location_changed",
                self._wrap_handler(self._update_spatial_location),
                "spatial_mapper")
            
            self.event_bus.subscribe("navigation_requested",
                self._wrap_handler(self._handle_navigation_request),
                "navigator_agent")
    
    async def _setup_dominance_subscriptions(self):
        """Setup subscriptions for dominance systems."""
        
        if self.brain.femdom_coordinator:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._analyze_dominance_opportunity),
                "femdom_coordinator")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._adjust_dominance_from_emotion),
                "femdom_coordinator")
    
    async def _setup_sensory_subscriptions(self):
        """Setup subscriptions for sensory systems."""
        
        if self.brain.digital_somatosensory_system:
            self.event_bus.subscribe("physical_touch",
                self._wrap_handler(self._process_touch_sensation),
                "digital_somatosensory_system")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._adjust_sensitivity_from_emotion),
                "digital_somatosensory_system")
    
    async def _setup_conditioning_subscriptions(self):
        """Setup subscriptions for conditioning systems."""
        
        if self.brain.conditioning_system:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._check_conditioning_triggers),
                "conditioning_system")
            
            self.event_bus.subscribe("reward_signal",
                self._wrap_handler(self._update_conditioning_from_reward),
                "conditioning_system")
    
    async def _setup_identity_subscriptions(self):
        """Setup subscriptions for identity evolution."""
        
        if self.brain.identity_evolution:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._evolve_identity_from_interaction),
                "identity_evolution")
            
            self.event_bus.subscribe("goal_status_change",
                self._wrap_handler(self._evolve_identity_from_goals),
                "identity_evolution")
    
    async def _setup_attention_subscriptions(self):
        """Setup subscriptions for attention systems."""
        
        if self.brain.attentional_controller:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._focus_attention_on_interaction),
                "attentional_controller")
            
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._adjust_attention_from_emotion),
                "attentional_controller")
    
    async def _setup_prediction_subscriptions(self):
        """Setup subscriptions for prediction systems."""
        
        if hasattr(self.brain, 'prediction_engine') and self.brain.prediction_engine:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._predict_user_intent),
                "prediction_engine")
            
            self.event_bus.subscribe("decision_needed",
                self._wrap_handler(self._predict_outcomes),
                "prediction_engine")
    
    async def _setup_creative_subscriptions(self):
        """Setup subscriptions for creative systems."""
        
        if self.brain.novelty_engine:
            self.event_bus.subscribe("user_interaction",
                self._wrap_handler(self._generate_creative_response),
                "novelty_engine")
    
    async def _setup_communication_subscriptions(self):
        """Setup subscriptions for communication systems."""
        
        if self.brain.proactive_communication_engine:
            self.event_bus.subscribe("emotional_state_change",
                self._wrap_handler(self._check_proactive_communication),
                "proactive_communication_engine")
            
            self.event_bus.subscribe("need_state_change",
                self._wrap_handler(self._communicate_needs),
                "proactive_communication_engine")
    
    async def _setup_metacognitive_subscriptions(self):
        """Setup subscriptions for metacognitive systems."""
        
        if self.brain.meta_core:
            self.event_bus.subscribe("*",  # Subscribe to all events
                self._wrap_handler(self._monitor_system_performance),
                "meta_core")
    
    # ========== EVENT HANDLERS ==========
    
    async def _handle_user_interaction_for_emotion(self, event: UserInteractionEvent):
        """Process user interaction for emotional impact."""
        try:
            if event.data.get("emotional_analysis"):
                valence = event.data["emotional_analysis"].get("valence", 0)
                arousal = event.data["emotional_analysis"].get("arousal", 0.5)
                
                # Update emotional state based on interaction
                await self.brain.emotional_core.process_emotional_input(
                    f"User interaction with valence {valence}"
                )
                
                # Publish emotional state change
                await self._publish_emotional_state()
                
        except Exception as e:
            logger.error(f"Error handling user interaction for emotion: {e}")
    
    async def _handle_goal_change_for_emotion(self, event: GoalEvent):
        """Process goal changes for emotional impact."""
        try:
            status = event.data.get("status")
            
            if status == "completed":
                # Positive emotion for goal completion
                self.brain.emotional_core.update_emotion("Joy", 0.7)
            elif status == "failed":
                # Negative emotion for goal failure
                self.brain.emotional_core.update_emotion("Frustration", 0.6)
            
            await self._publish_emotional_state()
            
        except Exception as e:
            logger.error(f"Error handling goal change for emotion: {e}")
    
    async def _handle_dominance_for_emotion(self, event: DominanceEvent):
        """Process dominance actions for emotional impact."""
        try:
            outcome = event.data.get("outcome")
            
            if outcome == "success":
                self.brain.emotional_core.update_emotion("Satisfaction", 0.8)
            elif outcome == "resistance":
                self.brain.emotional_core.update_emotion("Determination", 0.7)
            
            await self._publish_emotional_state()
            
        except Exception as e:
            logger.error(f"Error handling dominance for emotion: {e}")
    
    async def _handle_sensation_for_emotion(self, event: PhysicalSensationEvent):
        """Process physical sensations for emotional impact."""
        try:
            intensity = event.data.get("intensity", 0)
            sensation_type = event.data.get("sensation_type")
            
            if sensation_type == "pleasure" and intensity > 0.5:
                self.brain.emotional_core.update_emotion("Pleasure", intensity)
            elif sensation_type == "pain" and intensity > 0.3:
                self.brain.emotional_core.update_emotion("Discomfort", intensity * 0.7)
            
            await self._publish_emotional_state()
            
        except Exception as e:
            logger.error(f"Error handling sensation for emotion: {e}")
    
    async def _handle_emotion_for_mood(self, event: EmotionalEvent):
        """Update mood based on emotional changes."""
        try:
            # Mood manager will process the emotion change
            await self.brain.mood_manager.process_emotion_change(
                event.data.get("emotion"),
                event.data.get("intensity", 0.5)
            )
            
            # Update system context
            mood = self.brain.mood_manager.get_current_mood()
            await self.system_context.affective_state.update_mood(
                mood.dominant_mood,
                mood.stability
            )
            
        except Exception as e:
            logger.error(f"Error handling emotion for mood: {e}")
    
    async def _handle_need_for_mood(self, event: NeedEvent):
        """Update mood based on need satisfaction."""
        try:
            drive_strength = event.data.get("drive_strength", 0)
            
            if drive_strength > 0.7:
                # High unmet need affects mood negatively
                await self.brain.mood_manager.shift_mood_toward("Frustrated", 0.3)
            elif drive_strength < 0.2:
                # Well-satisfied needs improve mood
                await self.brain.mood_manager.shift_mood_toward("Content", 0.2)
                
        except Exception as e:
            logger.error(f"Error handling need for mood: {e}")
    
    # Memory handlers
    async def _store_interaction_memory(self, event: UserInteractionEvent):
        """Store user interaction as memory."""
        try:
            user_id = event.data.get("user_id")
            content = event.data.get("content")
            
            memory_text = f"User {user_id} said: {content}"
            
            await self.brain.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="interaction",
                significance=5,
                tags=["conversation", f"user:{user_id}"],
                metadata={
                    "user_id": user_id,
                    "input_type": event.data.get("input_type"),
                    "timestamp": event.timestamp.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")
    
    async def _store_emotional_memory(self, event: EmotionalEvent):
        """Store significant emotional states as memories."""
        try:
            intensity = event.data.get("intensity", 0)
            
            if intensity > 0.7:  # Only store significant emotions
                emotion = event.data.get("emotion")
                memory_text = f"Experienced strong {emotion} (intensity: {intensity:.2f})"
                
                await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="emotional",
                    significance=int(intensity * 10),
                    tags=["emotion", emotion.lower()],
                    metadata={
                        "emotion": emotion,
                        "valence": event.data.get("valence"),
                        "arousal": event.data.get("arousal"),
                        "timestamp": event.timestamp.isoformat()
                    }
                )
                
        except Exception as e:
            logger.error(f"Error storing emotional memory: {e}")
    
    # ========== PUBLISHER METHODS ==========
    
    async def _setup_module_publishers(self):
        """Setup event publishers for each module."""
        
        # Wrap module methods to publish events on state changes
        if self.brain.emotional_core:
            self._wrap_emotional_publishers()
        
        if self.brain.goal_manager:
            self._wrap_goal_publishers()
        
        if self.brain.needs_system:
            self._wrap_needs_publishers()
        
        # Add more module publishers as needed
    
    def _wrap_emotional_publishers(self):
        """Wrap emotional core methods to publish events."""
        original_update = self.brain.emotional_core.update_emotion
        
        async def wrapped_update_emotion(emotion: str, intensity: float):
            # Call original method
            result = await original_update(emotion, intensity) if asyncio.iscoroutinefunction(original_update) else original_update(emotion, intensity)
            
            # Publish event
            await self._publish_emotional_state()
            
            return result
        
        self.brain.emotional_core.update_emotion = wrapped_update_emotion
    
    def _wrap_goal_publishers(self):
        """Wrap goal manager methods to publish events."""
        if hasattr(self.brain.goal_manager, 'update_goal_status'):
            original_update = self.brain.goal_manager.update_goal_status
            
            async def wrapped_update_status(goal_id: str, status: str):
                # Call original method
                result = await original_update(goal_id, status) if asyncio.iscoroutinefunction(original_update) else original_update(goal_id, status)
                
                # Publish event
                goal = self.brain.goal_manager.goals.get(goal_id)
                if goal:
                    await self.event_bus.publish(GoalEvent(
                        source="goal_manager",
                        goal_id=goal_id,
                        status=status,
                        priority=goal.priority
                    ))
                
                return result
            
            self.brain.goal_manager.update_goal_status = wrapped_update_status
    
    def _wrap_needs_publishers(self):
        """Wrap needs system methods to publish events."""
        if hasattr(self.brain.needs_system, 'update_need'):
            original_update = self.brain.needs_system.update_need
            
            async def wrapped_update_need(need_name: str, change: float):
                # Call original method
                result = await original_update(need_name, change) if asyncio.iscoroutinefunction(original_update) else original_update(need_name, change)
                
                # Get updated state and publish
                need_state = self.brain.needs_system.needs.get(need_name)
                if need_state:
                    await self.event_bus.publish(NeedEvent(
                        source="needs_system",
                        need_name=need_name,
                        level=need_state['level'],
                        change=change,
                        drive_strength=need_state['drive_strength']
                    ))
                
                return result
            
            self.brain.needs_system.update_need = wrapped_update_need
    
    # ========== HELPER METHODS ==========
    
    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap an async handler to handle exceptions."""
        @wraps(handler)
        async def wrapped(event):
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
        return wrapped
    
    async def _publish_emotional_state(self):
        """Publish current emotional state."""
        if self.brain.emotional_core:
            state = self.brain.emotional_core.get_emotional_state()
            dominant_emotion, intensity = self.brain.emotional_core.get_dominant_emotion()
            
            await self.event_bus.publish(EmotionalEvent(
                source="emotional_core",
                emotion=dominant_emotion,
                valence=self.brain.emotional_core.get_emotional_valence(),
                arousal=self.brain.emotional_core.get_emotional_arousal(),
                intensity=intensity
            ))
            
            # Update system context
            await self.system_context.affective_state.update_emotion(
                dominant_emotion,
                intensity,
                self.brain.emotional_core.get_emotional_valence(),
                self.brain.emotional_core.get_emotional_arousal(),
                0.0  # dominance - could be calculated from emotional state
            )
    
    async def publish_integration_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """Publish integration status update."""
        await self.event_bus.publish(IntegrationStatusEvent(
            source="brain_integration",
            component=component,
            status=status,
            details=details or {}
        ))
    
    # ========== PUBLIC METHODS ==========
    
    async def publish_user_interaction(self, user_id: str, input_type: str, 
                                      content: str, emotional_analysis: Optional[Dict] = None):
        """Publish a user interaction event."""
        await self.event_bus.publish(UserInteractionEvent(
            source="brain",
            user_id=user_id,
            input_type=input_type,
            content=content,
            emotional_analysis=emotional_analysis
        ))
    
    async def publish_decision(self, decision_type: str, options: List[Dict[str, Any]], 
                             selected_option: Dict[str, Any], confidence: float):
        """Publish a decision event."""
        await self.event_bus.publish(DecisionEvent(
            source="brain",
            decision_type=decision_type,
            options=options,
            selected_option=selected_option,
            confidence=confidence
        ))
    
    async def request_reasoning(self, query: str, context: Dict[str, Any], 
                              timeout: float = 5.0) -> Optional[ReasoningEvent]:
        """Request reasoning from the reasoning core."""
        event = Event("reasoning_request", "brain", {
            "query": query,
            "context": context
        })
        
        response = await self.event_bus.request(event, "reasoning_core", timeout)
        return response
    
    async def shutdown(self):
        """Shutdown event bus integration."""
        logger.info("Shutting down event bus integration")
        
        # Unsubscribe all handlers
        for event_type, handler, subscriber_id in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler, subscriber_id)
        
        self.subscriptions.clear()
        self._initialized = False
        
        await self.publish_integration_status("event_bus", "shutdown")


# Integration method to add to NyxBrain class
async def initialize_event_bus_integration(brain):
    """
    Initialize event bus integration for the brain.
    This should be called during brain initialization.
    """
    if not hasattr(brain, 'event_bus_integration'):
        brain.event_bus_integration = NyxEventBusIntegration(brain)
        await brain.event_bus_integration.initialize()
        
        # Add convenience methods to brain
        brain.publish_event = brain.event_bus_integration.event_bus.publish
        brain.publish_user_interaction = brain.event_bus_integration.publish_user_interaction
        brain.publish_decision = brain.event_bus_integration.publish_decision
        brain.request_reasoning = brain.event_bus_integration.request_reasoning
        
        logger.info("Event bus integration added to NyxBrain")
    else:
        logger.warning("Event bus integration already exists")
