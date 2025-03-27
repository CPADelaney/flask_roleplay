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

# nyx/core/integration/integration_manager.py

import logging
import asyncio
from typing import Dict, List, Any, Optional

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class IntegrationManager:
    """
    Master integration manager for the Nyx system.
    
    Manages all bridges and ensures system-wide coordination and consistency.
    Provides centralized monitoring, configuration, and diagnostics for all
    integration components.
    """
    
    def __init__(self, nyx_brain):
        """Initialize the integration manager."""
        self.brain = nyx_brain
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Track all integration bridges
        self.bridges = {}
        self.bridge_statuses = {}
        self.bridge_dependencies = {}
        
        # Integration state
        self.initialization_sequence = []
        self.initialized = False
        
        logger.info("IntegrationManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize all integration bridges in correct order."""
        try:
            # Build dependency graph
            self._build_dependency_graph()
            
            # Determine initialization sequence
            self._determine_initialization_sequence()
            
            # Initialize bridges in sequence
            for bridge_name in self.initialization_sequence:
                if bridge_name in self.bridges:
                    bridge = self.bridges[bridge_name]
                    success = await bridge.initialize()
                    self.bridge_statuses[bridge_name] = success
                    
                    if not success:
                        logger.warning(f"Bridge {bridge_name} initialization failed")
            
            # Check overall initialization status
            failed_bridges = [name for name, status in self.bridge_statuses.items() if not status]
            
            if failed_bridges:
                logger.error(f"Failed to initialize bridges: {failed_bridges}")
                self.initialized = False
                return False
            
            self.initialized = True
            logger.info("IntegrationManager successfully initialized all bridges")
            return True
        except Exception as e:
            logger.error(f"Error initializing IntegrationManager: {e}")
            self.initialized = False
            return False
    
    def register_bridge(self, name: str, bridge: Any, dependencies: List[str] = None) -> None:
        """
        Register an integration bridge.
        
        Args:
            name: Name of the bridge
            bridge: Bridge instance
            dependencies: List of bridge names this bridge depends on
        """
        self.bridges[name] = bridge
        self.bridge_statuses[name] = False
        self.bridge_dependencies[name] = dependencies or []
        
        logger.info(f"Registered bridge: {name}")
    
    @trace_method(level=TraceLevel.INFO, group_id="IntegrationManager")
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of all integration components.
        
        Returns:
            Status information for all bridges
        """
        status = {
            "initialized": self.initialized,
            "bridge_count": len(self.bridges),
            "bridges_initialized": sum(1 for status in self.bridge_statuses.values() if status),
            "initialization_sequence": self.initialization_sequence,
            "bridge_statuses": self.bridge_statuses.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph for bridges."""
        # Implementation would build a graph of bridge dependencies
        pass
    
    def _determine_initialization_sequence(self) -> None:
        """Determine correct initialization sequence based on dependencies."""
        # Implementation would use topological sort to determine sequence
        # For bridges with no dependencies or circular dependencies
        pass

# Function to create the integration manager
def create_integration_manager(nyx_brain):
    """Create an integration manager for the given brain."""
    return IntegrationManager(nyx_brain)
