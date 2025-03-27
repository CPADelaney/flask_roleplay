# nyx/core/integration/integration_manager.py

import logging
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class IntegrationManager:
    """
    Master integration manager for Nyx.
    
    This class coordinates all integration bridges and provides a
    unified interface for initializing and monitoring the integration layer.
    """
    
    def __init__(self, nyx_brain):
        self.brain = nyx_brain
        
        # Initialize integration bridges
        self.bridges = {}
        
        # Setup core integration bridges
        self._setup_bridges()
        
        logger.info("IntegrationManager initialized")
    
    def _setup_bridges(self):
        """Set up all integration bridges."""
        try:
            # Import bridge creation functions
            from nyx.core.integration.mood_emotional_bridge import create_mood_emotional_bridge
            from nyx.core.integration.multimodal_attention_bridge import create_multimodal_attention_bridge
            from nyx.core.integration.prediction_imagination_bridge import create_prediction_imagination_bridge
            from nyx.core.integration.relationship_tom_bridge import create_relationship_tom_bridge
            from nyx.core.integration.reward_learning_bridge import create_reward_learning_bridge
            from nyx.core.integration.dominance_integration_manager import create_dominance_integration_manager
            from nyx.core.integration.tom_integration import create_tom_integrator
            from nyx.core.integration.need_goal_action_pipeline import create_need_goal_action_pipeline
            from nyx.core.integration.narrative_memory_identity_nexus import create_narrative_memory_identity_nexus
            
            # Create bridges
            self.bridges["mood_emotional"] = create_mood_emotional_bridge(self.brain)
            self.bridges["multimodal_attention"] = create_multimodal_attention_bridge(self.brain)
            self.bridges["prediction_imagination"] = create_prediction_imagination_bridge(self.brain)
            self.bridges["relationship_tom"] = create_relationship_tom_bridge(self.brain)
            self.bridges["reward_learning"] = create_reward_learning_bridge(self.brain)
            self.bridges["dominance"] = create_dominance_integration_manager(self.brain)
            self.bridges["tom"] = create_tom_integrator(self.brain)
            self.bridges["need_goal_action"] = create_need_goal_action_pipeline(self.brain)
            self.bridges["narrative_memory_identity"] = create_narrative_memory_identity_nexus(self.brain)
            
            logger.info(f"Set up {len(self.bridges)} integration bridges")
        except Exception as e:
            logger.error(f"Error setting up integration bridges: {e}")
    
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all integration bridges."""
        results = {}
        
        for name, bridge in self.bridges.items():
            try:
                logger.info(f"Initializing {name} bridge...")
                success = await bridge.initialize()
                results[name] = success
                logger.info(f"Initialized {name} bridge: {'success' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Error initializing {name} bridge: {e}")
                results[name] = False
        
        return {
            "status": "completed",
            "results": results,
            "success_count": sum(1 for status in results.values() if status),
            "failed_count": sum(1 for status in results.values() if not status)
        }
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integration bridges."""
        status = {}
        
        for name, bridge in self.bridges.items():
            try:
                # Check if bridge has get_status method
                if hasattr(bridge, "get_status"):
                    bridge_status = await bridge.get_status()
                elif hasattr(bridge, "get_bridge_state"):
                    bridge_status = await bridge.get_bridge_state()
                elif hasattr(bridge, "get_queue_status"):
                    bridge_status = await bridge.get_queue_status()
                else:
                    bridge_status = {"available": True}
                
                status[name] = bridge_status
            except Exception as e:
                logger.error(f"Error getting status for {name} bridge: {e}")
                status[name] = {"error": str(e)}
        
        return {
            "status": "success",
            "bridges": status,
            "bridge_count": len(self.bridges)
        }

# Function to create the integration manager
def create_integration_manager(nyx_brain):
    """Create an integration manager for the given brain."""
    return IntegrationManager(nyx_brain)
