# nyx/core/integration/dominance_integration_manager.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class DominanceIntegrationManager:
    """
    Master integration manager for dominance subsystems.
    
    This class coordinates all dominance-related integration bridges
    and provides a unified interface for dominance functionality.
    """
    
    def __init__(self, nyx_brain):
        """Initialize the dominance integration manager."""
        self.brain = nyx_brain
        
        # Initialize bridges
        self.reward_identity_bridge = self._create_reward_identity_bridge()
        self.imagination_decision_bridge = self._create_imagination_decision_bridge()
        self.memory_reflection_bridge = self._create_memory_reflection_bridge()
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration status tracking
        self.initialized_bridges = []
        self.initialization_complete = False
        
        logger.info("DominanceIntegrationManager initialized")
    
    def _create_reward_identity_bridge(self):
        """Create the dominance-reward-identity bridge."""
        try:
            from nyx.core.integration.dominance_reward_identity_bridge import create_dominance_reward_identity_bridge
            return create_dominance_reward_identity_bridge(self.brain)
        except Exception as e:
            logger.error(f"Error creating dominance-reward-identity bridge: {e}")
            return None
    
    def _create_imagination_decision_bridge(self):
        """Create the dominance-imagination-decision bridge."""
        try:
            from nyx.core.integration.dominance_imagination_decision_bridge import create_dominance_imagination_decision_bridge
            return create_dominance_imagination_decision_bridge(self.brain)
        except Exception as e:
            logger.error(f"Error creating dominance-imagination-decision bridge: {e}")
            return None
    
    def _create_memory_reflection_bridge(self):
        """Create the dominance-memory-reflection bridge."""
        try:
            from nyx.core.integration.dominance_memory_reflection_bridge import create_dominance_memory_reflection_bridge
            return create_dominance_memory_reflection_bridge(self.brain)
        except Exception as e:
            logger.error(f"Error creating dominance-memory-reflection bridge: {e}")
            return None
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all bridges."""
        results = {
            "reward_identity_initialized": False,
            "imagination_decision_initialized": False,
            "memory_reflection_initialized": False,
        }
        
        # Initialize reward-identity bridge
        if self.reward_identity_bridge:
            results["reward_identity_initialized"] = await self.reward_identity_bridge.initialize()
            if results["reward_identity_initialized"]:
                self.initialized_bridges.append("reward_identity")
        
        # Initialize imagination-decision bridge
        if self.imagination_decision_bridge:
            results["imagination_decision_initialized"] = await self.imagination_decision_bridge.initialize()
            if results["imagination_decision_initialized"]:
                self.initialized_bridges.append("imagination_decision")
        
        # Initialize memory-reflection bridge
        if self.memory_reflection_bridge:
            results["memory_reflection_initialized"] = await self.memory_reflection_bridge.initialize()
            if results["memory_reflection_initialized"]:
                self.initialized_bridges.append("memory_reflection")
        
        # Set initialization status
        self.initialization_complete = all([
            results["reward_identity_initialized"],
            results["imagination_decision_initialized"],
            results["memory_reflection_initialized"]
        ])
        
        results["status"] = "success" if self.initialization_complete else "partial"
        results["initialized_bridges"] = self.initialized_bridges
        
        logger.info(f"DominanceIntegrationManager initialization: {results['status']}")
        return results
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceIntegration")
    async def process_dominance_action(self,
                                    action_type: str,
                                    user_id: str,
                                    intensity: float,
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a dominance action through all integration bridges.
        
        Args:
            action_type: Type of dominance action
            user_id: User ID
            intensity: Dominance intensity (0.0-1.0)
            context: Additional context
            
        Returns:
            Processing results
        """
        results = {
            "action_type": action_type,
            "user_id": user_id,
            "intensity": intensity,
            "bridges_processed": [],
            "decision": None,
            "outcome": None,
            "memory_stored": False,
            "reward_processed": False
        }
        
        try:
            # 1. Check if action should proceed using imagination-decision bridge
            if self.imagination_decision_bridge:
                simulation = await self.imagination_decision_bridge.simulate_dominance_outcome(
                    user_id=user_id,
                    dominance_action=action_type,
                    intensity=intensity,
                    relationship_context=context
                )
                
                results["decision"] = simulation.get("decision", {})
                results["bridges_processed"].append("imagination_decision")
                
                # If decision is not to proceed, return early
                if not simulation.get("decision", {}).get("proceed", True):
                    results["outcome"] = "cancelled"
                    results["reason"] = simulation.get("decision", {}).get("reasoning", "Decision to not proceed")
                    return results
            
            # 2. Execute the dominance action
            action_result = None
            if hasattr(self.brain, "dominance_system") and hasattr(self.brain.dominance_system, "execute_dominance_action"):
                action_result = await self.brain.dominance_system.execute_dominance_action(
                    action_type=action_type,
                    user_id=user_id,
                    intensity=intensity,
                    context=context
                )
                
                results["action_result"] = action_result
                results["outcome"] = action_result.get("outcome", "unknown")
            else:
                # No dominance system available, simulate outcome
                # This branch should rarely happen in a properly configured system
                import random
                success_chance = 0.8 - (intensity * 0.3)  # Higher intensity = lower success chance
                success = random.random() < success_chance
                results["outcome"] = "success" if success else "failure"
            
            # 3. Store memory using memory-reflection bridge
            if self.memory_reflection_bridge:
                # Prepare metadata
                metadata = {
                    "action_type": action_type,
                    "intensity": intensity,
                    "outcome": results["outcome"],
                    "user_id": user_id
                }
                if context:
                    metadata.update(context)
                
                # Store memory
                memory_result = await self.memory_reflection_bridge.store_dominance_memory(
                    user_id=user_id,
                    memory_text=f"Executed dominance action '{action_type}' on user {user_id} with intensity {intensity:.2f}. Outcome: {results['outcome']}",
                    significance=int(5 + (intensity * 5)),  # Scale intensity to significance (1-10)
                    metadata=metadata
                )
                
                results["memory_stored"] = memory_result.get("status") == "success"
                results["memory_id"] = memory_result.get("memory_id")
                results["bridges_processed"].append("memory_reflection")
            
            # 4. Process reward using reward-identity bridge
            if self.reward_identity_bridge:
                # Map outcome type
                outcome_type = "compliance"
                if "command" in action_type.lower():
                    outcome_type = "compliance" if results["outcome"] == "success" else "compliance_failure"
                elif "escalate" in action_type.lower() or "increase" in action_type.lower():
                    outcome_type = "escalation_success" if results["outcome"] == "success" else "resistance_failure"
                
                # Process reward
                reward_result = await self.reward_identity_bridge.process_dominance_outcome(
                    outcome_type=outcome_type,
                    user_id=user_id,
                    intensity=intensity,
                    success=results["outcome"] == "success",
                    context={"action": action_type}
                )
                
                results["reward_processed"] = reward_result.get("reward_generated", False)
                results["identity_updated"] = reward_result.get("identity_updated", False)
                results["bridges_processed"].append("reward_identity")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing dominance action: {e}")
            return {
                "error": str(e),
                "action_type": action_type,
                "user_id": user_id,
                "outcome": "error"
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceIntegration")
    async def get_dominance_recommendation(self,
                                        user_id: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a recommendation for dominance action.
        
        Args:
            user_id: User ID
            context: Current interaction context
            
        Returns:
            Recommendation for dominance action
        """
        if self.imagination_decision_bridge:
            return await self.imagination_decision_bridge.get_dominance_recommendation(
                user_id=user_id,
                context=context
            )
        else:
            return {
                "status": "error",
                "message": "Imagination-decision bridge not available",
                "has_recommendation": False
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceIntegration")
    async def reflect_on_dominance_patterns(self,
                                        user_id: str) -> Dict[str, Any]:
        """
        Generate reflection on dominance patterns for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Reflection results
        """
        if self.memory_reflection_bridge:
            return await self.memory_reflection_bridge._trigger_dominance_reflection(user_id)
        else:
            return {
                "status": "error",
                "message": "Memory-reflection bridge not available"
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="DominanceIntegration")
    async def get_dominance_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of dominance integration.
        
        Returns:
            Integration status
        """
        return {
            "status": "success",
            "initialization_complete": self.initialization_complete,
            "initialized_bridges": self.initialized_bridges,
            "reward_identity_bridge_available": self.reward_identity_bridge is not None,
            "imagination_decision_bridge_available": self.imagination_decision_bridge is not None,
            "memory_reflection_bridge_available": self.memory_reflection_bridge is not None
        }

# Function to create the integration manager
def create_dominance_integration_manager(nyx_brain):
    """Create a dominance integration manager for the given brain."""
    return DominanceIntegrationManager(nyx_brain)
