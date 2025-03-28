# nyx/core/integration/conditioning_integration_bridge.py

import logging
import asyncio
from typing import Dict, List, Any, Optional

from nyx.core.integration.event_bus import Event, ConditioningEvent, ConditionedResponseEvent, get_event_bus

logger = logging.getLogger(__name__)

class ConditioningIntegrationBridge:
    """
    Bridge to integrate the conditioning system with other modules.
    Connects the conditioning system with the experience interface,
    dominance system, and reward system.
    """
    
    def __init__(self, nyx_brain):
        self.brain = nyx_brain
        self.event_bus = get_event_bus()
        self.conditioning_system = getattr(nyx_brain, "conditioning_system", None)
        self.experience_interface = getattr(nyx_brain, "experience_interface", None)
        self.dominance_system = getattr(nyx_brain, "dominance_system", None)
        self.reward_system = getattr(nyx_brain, "reward_system", None)
        self.initialized = False
        
        logger.info("ConditioningIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and set up subscriptions."""
        if not self.conditioning_system:
            logger.warning("Conditioning system not available, bridge initialization failed")
            return False
        
        # Subscribe to relevant events
        self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
        self.event_bus.subscribe("dominance_action", self._handle_dominance_action)
        
        # Add publish methods to conditioning system
        self._extend_conditioning_system()
        
        self.initialized = True
        logger.info("ConditioningIntegrationBridge successfully initialized")
        return True
    
    def _extend_conditioning_system(self) -> None:
        """Extend the conditioning system with event publishing methods."""
        conditioning_system = self.conditioning_system
        
        # Store original methods
        original_classical = conditioning_system.process_classical_conditioning
        original_operant = conditioning_system.process_operant_conditioning
        original_trigger = conditioning_system.trigger_conditioned_response
        
        # Replace with wrapped methods
        async def wrapped_classical_conditioning(*args, **kwargs):
            result = await original_classical(*args, **kwargs)
            await self._publish_conditioning_update("classical", result, kwargs.get("context", {}))
            return result
        
        async def wrapped_operant_conditioning(*args, **kwargs):
            result = await original_operant(*args, **kwargs)
            await self._publish_conditioning_update("operant", result, kwargs.get("context", {}))
            return result
        
        async def wrapped_trigger_conditioned_response(*args, **kwargs):
            result = await original_trigger(*args, **kwargs)
            if result:
                await self._publish_conditioned_response(args[0], result, kwargs.get("context", {}))
            return result
        
        # Replace methods
        conditioning_system.process_classical_conditioning = wrapped_classical_conditioning
        conditioning_system.process_operant_conditioning = wrapped_operant_conditioning
        conditioning_system.trigger_conditioned_response = wrapped_trigger_conditioned_response
    
    async def _publish_conditioning_update(self, conditioning_type: str, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Publish a conditioning update event."""
        if not hasattr(self, "event_bus"):
            return
        
        association_key = result.get("association_key", "unknown")
        association_type = result.get("type", "unknown")
        strength = result.get("new_strength", result.get("strength", 0.0))
        user_id = context.get("user_id", None)
        
        event = ConditioningEvent(
            source="conditioning_system",
            association_type=conditioning_type,
            association_key=association_key,
            strength=strength,
            user_id=user_id
        )
        
        await self.event_bus.publish(event)
    
    async def _publish_conditioned_response(self, stimulus: str, response: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Publish a conditioned response event."""
        if not hasattr(self, "event_bus"):
            return
        
        triggered_responses = response.get("triggered_responses", [])
        user_id = context.get("user_id", None)
        
        event = ConditionedResponseEvent(
            source="conditioning_system",
            stimulus=stimulus,
            responses=triggered_responses,
            user_id=user_id
        )
        
        await self.event_bus.publish(event)
    
    async def _handle_user_interaction(self, event: Event) -> None:
        """Handle user interaction events for conditioning."""
        if not self.conditioning_system:
            return
        
        data = event.data
        user_id = data.get("user_id", "default")
        content = data.get("content", "")
        
        if content and user_id:
            # Use the input processor if available, or basic pattern detection
            if hasattr(self.brain, "input_processor"):
                await self.brain.input_processor.process_input(content, user_id)
            else:
                # Basic pattern detection
                patterns = self._detect_patterns(content)
                for pattern in patterns:
                    await self.conditioning_system.trigger_conditioned_response(
                        stimulus=pattern,
                        context={"user_id": user_id, "source": "user_interaction"}
                    )
    
    async def _handle_dominance_action(self, event: Event) -> None:
        """Handle dominance action events for conditioning."""
        if not self.conditioning_system:
            return
        
        data = event.data
        action = data.get("action", "")
        outcome = data.get("outcome", "")
        intensity = data.get("intensity", 0.5)
        user_id = data.get("user_id", "default")
        
        # Only process successful dominance actions
        if outcome != "success":
            return
        
        # Reinforce dominance behaviors
        await self.conditioning_system.process_operant_conditioning(
            behavior=f"dominance_{action}",
            consequence_type="positive_reinforcement",
            intensity=intensity,
            context={"user_id": user_id, "source": "dominance_system"}
        )
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Basic pattern detection for conditioning triggers."""
        patterns = []
        
        # Very simple pattern detection
        if any(s in text.lower() for s in ["yes mistress", "i obey", "as you command"]):
            patterns.append("submission_language")
        
        if any(s in text.lower() for s in ["no i won't", "you can't make me", "i refuse"]):
            patterns.append("defiance")
        
        return patterns
    
    async def get_bridge_state(self) -> Dict[str, Any]:
        """Get the current state of the bridge."""
        state = {
            "initialized": self.initialized,
            "has_conditioning_system": self.conditioning_system is not None,
            "has_experience_interface": self.experience_interface is not None,
            "has_dominance_system": self.dominance_system is not None,
            "has_reward_system": self.reward_system is not None
        }
        
        return state

# Function to create the conditioning bridge
def create_conditioning_integration_bridge(nyx_brain):
    """Create a conditioning integration bridge for the given brain."""
    return ConditioningIntegrationBridge(nyx_brain)
