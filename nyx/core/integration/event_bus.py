# nyx/core/integration/event_bus.py

import asyncio
import logging
import datetime
import uuid
from collections import defaultdict
from typing import Dict, List, Any, Callable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class Event:
    """Base event class for the Nyx event system."""
    def __init__(self, event_type: str, source: str, data: Any = None):
        self.id = f"evt_{uuid.uuid4().hex[:8]}"
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = datetime.datetime.now()
    
    def __str__(self):
        return f"Event(type={self.event_type}, source={self.source}, time={self.timestamp.isoformat()})"

class ConditioningEvent(Event):
    """Conditioning-related event."""
    def __init__(self, source: str, association_type: str, association_key: str, 
                 strength: float, user_id: Optional[str] = None):
        super().__init__("conditioning_update", source, {
            "association_type": association_type,  # "classical" or "operant"
            "association_key": association_key,
            "strength": strength,
            "user_id": user_id
        })

class ConditionedResponseEvent(Event):
    """Triggered conditioned response event."""
    def __init__(self, source: str, stimulus: str, responses: List[Dict[str, Any]], 
                 user_id: Optional[str] = None):
        super().__init__("conditioned_response", source, {
            "stimulus": stimulus,
            "triggered_responses": responses,
            "user_id": user_id
        })

class EmotionalEvent(Event):
    """Emotional state change event."""
    def __init__(self, source: str, emotion: str, valence: float, arousal: float, intensity: float = 0.5):
        super().__init__("emotional_state_change", source, {
            "emotion": emotion,
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity
        })

class PhysicalSensationEvent(Event):
    """Physical sensation event from the DSS."""
    def __init__(self, source: str, region: str, sensation_type: str, intensity: float, cause: str = ""):
        super().__init__("physical_sensation", source, {
            "region": region,
            "sensation_type": sensation_type,
            "intensity": intensity,
            "cause": cause
        })

class GoalEvent(Event):
    """Goal-related event."""
    def __init__(self, source: str, goal_id: str, status: str, priority: float = 0.5):
        super().__init__("goal_status_change", source, {
            "goal_id": goal_id,
            "status": status,
            "priority": priority
        })

class NeedEvent(Event):
    """Need state change event."""
    def __init__(self, source: str, need_name: str, level: float, change: float, drive_strength: float):
        super().__init__("need_state_change", source, {
            "need_name": need_name,
            "level": level,
            "change": change,
            "drive_strength": drive_strength
        })

class UserInteractionEvent(Event):
    """User interaction event."""
    def __init__(self, source: str, user_id: str, input_type: str, content: str, emotional_analysis: Optional[Dict] = None):
        super().__init__("user_interaction", source, {
            "user_id": user_id,
            "input_type": input_type,
            "content": content,
            "emotional_analysis": emotional_analysis
        })

class DominanceEvent(Event):
    """Dominance-related event."""
    def __init__(self, source: str, action: str, user_id: str, intensity: float, outcome: Optional[str] = None):
        super().__init__("dominance_action", source, {
            "action": action,
            "user_id": user_id,
            "intensity": intensity,
            "outcome": outcome
        })

class NarrativeEvent(Event):
    """Narrative formation event."""
    def __init__(self, source: str, segment_id: str, title: str, summary: str):
        super().__init__("narrative_update", source, {
            "segment_id": segment_id,
            "title": title,
            "summary": summary
        })

class ReasoningEvent(Event):
    """Reasoning result event."""
    def __init__(self, source: str, query: str, result: Dict[str, Any], confidence: float):
        super().__init__("reasoning_result", source, {
            "query": query,
            "result": result,
            "confidence": confidence
        })

class DecisionEvent(Event):
    """Decision event."""
    def __init__(self, source: str, decision_type: str, options: List[Dict[str, Any]], 
                selected_option: Dict[str, Any], confidence: float):
        super().__init__("decision_made", source, {
            "decision_type": decision_type,
            "options": options,
            "selected_option": selected_option,
            "confidence": confidence
        })

class IntegrationStatusEvent(Event):
    """Integration status update event."""
    def __init__(self, source: str, component: str, status: str, details: Dict[str, Any] = None):
        super().__init__("integration_status", source, {
            "component": component,
            "status": status,
            "details": details or {}
        })

# Add additional event types needed by new bridges
class PredictionEvent(Event):
    """Prediction event."""
    def __init__(self, source: str, prediction_type: str, prediction: Any, confidence: float):
        super().__init__("prediction_completed", source, {
            "prediction_type": prediction_type,
            "prediction": prediction,
            "confidence": confidence
        })

class SimulationEvent(Event):
    """Simulation event."""
    def __init__(self, source: str, simulation_id: str, outcome: Any, confidence: float):
        super().__init__("simulation_completed", source, {
            "simulation_id": simulation_id,
            "outcome": outcome,
            "confidence": confidence
        })

class AttentionEvent(Event):
    """Attention focus change event."""
    def __init__(self, source: str, focus_target: str, focus_type: str, attention_level: float):
        super().__init__("attention_focus_changed", source, {
            "focus_target": focus_target,
            "focus_type": focus_type,
            "attention_level": attention_level
        })

class UserModelEvent(Event):
    """User model update event."""
    def __init__(self, source: str, user_id: str, updates: Dict[str, Any]):
        super().__init__("user_model_updated", source, {
            "user_id": user_id,
            "updates": updates
        })

class EventBus:
    """
    Central event distribution system for inter-module communication.
    Provides pub-sub mechanism for modules to communicate without direct dependencies.
    """
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = []
        self.max_history = 1000
        self._lock = asyncio.Lock()  # For thread safety
        self.event_stats = defaultdict(int)
        logger.info("EventBus initialized")
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        async with self._lock:
            # Log the event
            logger.debug(f"Publishing event: {event}")
            
            # Add to history
            self.event_history.append(event)
            self.event_stats[event.event_type] += 1
            
            # Trim history if needed
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        for callback in self.subscribers[event.event_type]:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber callback: {e}")
        
        # Also notify wildcard subscribers
        for callback in self.subscribers["*"]:  # Wildcard subscribers get all events
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in wildcard event subscriber callback: {e}")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to, or "*" for all events
            callback: Async function to call when event occurs
        """
        self.subscribers[event_type].append(callback)
        logger.debug(f"Added subscriber for event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
            
        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Removed subscriber for event type: {event_type}")
            return True
        return False
    
    async def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Event]:
        """
        Get recent events, optionally filtered by type.
        
        Args:
            event_type: Optional type of events to return
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        async with self._lock:
            if event_type:
                filtered = [e for e in self.event_history if e.event_type == event_type]
                return filtered[-limit:]
            else:
                return self.event_history[-limit:]
    
    def get_event_stats(self) -> Dict[str, int]:
        """
        Get statistics on events processed.
        
        Returns:
            Dictionary of event counts by type
        """
        return dict(self.event_stats)
    
    async def clear_history(self) -> None:
        """Clear event history."""
        async with self._lock:
            self.event_history = []
            logger.info("Event history cleared")

# Singleton instance
_instance = None

def get_event_bus() -> EventBus:
    """Get the singleton event bus instance."""
    global _instance
    if _instance is None:
        _instance = EventBus()
    return _instance
