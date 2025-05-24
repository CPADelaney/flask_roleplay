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

class SpatialEvent(Event):
    """Spatial event for location changes or observations."""
    def __init__(self, source: str, location_id: str, coordinates: Dict[str, float], 
                 event_type: str = "location_changed"):
        super().__init__(event_type, source, {
            "location_id": location_id,
            "coordinates": coordinates
        })

class NavigationEvent(Event):
    """Navigation event for path planning and movement."""
    def __init__(self, source: str, start_location: str, destination: str, 
                 path: List[Dict[str, float]] = None):
        super().__init__("navigation_requested", source, {
            "start_location": start_location,
            "destination": destination,
            "path": path
        })

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
    Advanced event bus for Nyx—supports targeted delivery, async request/response,
    and dynamic subscriber management, with backward compatibility.
    """
    def __init__(self):
        self.subscribers = defaultdict(list)  # event_type -> list of (callback, subscriber_id)
        self.event_history = []
        self.max_history = 1000
        self._lock = asyncio.Lock()
        self.event_stats = defaultdict(int)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        logger.info("EventBus initialized")

    def subscribe(self, event_type: str, callback: Callable, subscriber_id: Optional[str] = None) -> None:
        """
        Subscribe a callback to an event type, optionally with a subscriber_id (for targeted delivery).
        """
        self.subscribers[event_type].append((callback, subscriber_id))
        logger.debug(f"Added subscriber ({subscriber_id or callback}) for event type: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable, subscriber_id: Optional[str] = None) -> bool:
        """
        Unsubscribe a callback (and optional id) from an event type.
        """
        before = len(self.subscribers[event_type])
        self.subscribers[event_type] = [
            (cb, sid) for cb, sid in self.subscribers[event_type]
            if cb != callback or (subscriber_id and sid != subscriber_id)
        ]
        logger.debug(f"Unsubscribed ({subscriber_id or callback}) from event type: {event_type}")
        return len(self.subscribers[event_type]) < before

    async def publish(
        self,
        event: Event,
        target: Optional[str] = None,
        target_group: Optional[List[str]] = None,
        respond_to: Optional[str] = None,
    ) -> None:
        """
        Publish an event to all or a targeted set of subscribers.
        - target: single subscriber_id (module name)
        - target_group: list of subscriber_ids
        - respond_to: used for request/response, to signal a specific return event_type
        """
        async with self._lock:
            # Store event
            self.event_history.append(event)
            self.event_stats[event.event_type] += 1
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # Build the target filter
        callbacks = self.subscribers[event.event_type] + self.subscribers["*"]
        for callback, subscriber_id in callbacks:
            deliver = False
            if target:
                deliver = (subscriber_id == target)
            elif target_group:
                deliver = (subscriber_id in target_group)
            else:
                deliver = True  # global/wildcard
            if deliver:
                try:
                    await callback(event)
                    logger.debug(
                        f"Delivered {event.event_type} to {subscriber_id or callback} at {event.timestamp.isoformat()}"
                    )
                except Exception as e:
                    logger.error(f"Error in event subscriber callback: {e}")

    # --------- Request/Response Pattern ---------
    async def request(
        self,
        event: Event,
        target: str,
        timeout: float = 5.0,
    ) -> Any:
        """
        Publish a request event, await a response from the target subscriber_id.
        The target should reply via respond_to/correlation_id.
        """
        correlation_id = event.id
        response_event_type = f"{event.event_type}_response_{correlation_id}"
        fut = asyncio.get_event_loop().create_future()
        self.pending_responses[response_event_type] = fut

        # Setup a temporary responder on the response event
        async def _on_response(resp_event):
            if not fut.done():
                fut.set_result(resp_event)
            self.unsubscribe(response_event_type, _on_response)

        self.subscribe(response_event_type, _on_response, subscriber_id="__requester__")

        # Attach response event type/correlation to event data for the responder
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            event.data = {}
        event.data["respond_to"] = response_event_type
        event.data["correlation_id"] = correlation_id

        await self.publish(event, target=target)
        try:
            resp_event = await asyncio.wait_for(fut, timeout)
            return resp_event
        except asyncio.TimeoutError:
            logger.warning(f"Request for {event.event_type} timed out (target={target})")
            return None
        finally:
            self.unsubscribe(response_event_type, _on_response)
            self.pending_responses.pop(response_event_type, None)

    async def respond(self, event: Event, response_data: Any) -> None:
        """
        Send a response to a request (using event's respond_to and correlation_id fields).
        """
        if not hasattr(event, "data") or "respond_to" not in event.data:
            raise ValueError("Event missing 'respond_to' for response pattern.")
        response_event_type = event.data["respond_to"]
        correlation_id = event.data.get("correlation_id")
        response_event = Event(
            event_type=response_event_type,
            source="event_bus",
            data={"correlation_id": correlation_id, "response": response_data},
        )
        await self.publish(response_event, target=None)

    # --------- History/Stats Intact ---------
    async def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Event]:
        async with self._lock:
            if event_type:
                filtered = [e for e in self.event_history if e.event_type == event_type]
                return filtered[-limit:]
            else:
                return self.event_history[-limit:]

    def get_event_stats(self) -> Dict[str, int]:
        return dict(self.event_stats)

    async def clear_history(self) -> None:
        async with self._lock:
            self.event_history = []
            logger.info("Event history cleared")

# Singleton instance remains the same
_instance = None

def get_event_bus() -> EventBus:
    global _instance
    if _instance is None:
        _instance = EventBus()
    return _instance
