# nyx/core/integration/integration_manager.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set

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
    
    async def _setup_bridges(self):
        """Set up all integration bridges with appropriate dependencies."""
        try:
            # Import bridge creation functions
            from nyx.core.integration.action_selector import create_action_selector
            from nyx.core.integration.adaptation_goal_bridge import create_core_systems_integration_bridge
            from nyx.core.integration.autonomous_cognitive_bridge import create_autonomous_cognitive_bridge
            from nyx.core.integration.decision_action_coordinator import create_decision_action_coordinator
            from nyx.core.integration.dynamic_attention_system import create_dynamic_attention_system
            from nyx.core.integration.emotional_cognitive_bridge import create_emotional_cognitive_bridge
            from nyx.core.integration.emotional_hormonal_bridge import create_emotional_hormonal_bridge
            from nyx.core.integration.identity_imagination_emotional_bridge import create_identity_imagination_emotional_bridge
            from nyx.core.integration.knowledge_curiosity_exploration_bridge import create_knowledge_curiosity_exploration_bridge
            from nyx.core.integration.knowledge_memory_reasoning_bridge import create_knowledge_memory_reasoning_bridge
            from nyx.core.integration.memory_integration_bridge import create_memory_integration_bridge
            from nyx.core.integration.mood_emotional_bridge import create_mood_emotional_bridge
            from nyx.core.integration.multimodal_attention_bridge import create_multimodal_attention_bridge
            from nyx.core.integration.narrative_memory_identity_nexus import create_narrative_memory_identity_nexus
            from nyx.core.integration.need_goal_action_pipeline import create_need_goal_action_pipeline
            from nyx.core.integration.perceptual_integration_layer import create_perceptual_integration_layer
            from nyx.core.integration.prediction_imagination_bridge import create_prediction_imagination_bridge
            from nyx.core.integration.procedural_memory_integration_bridge import create_procedural_memory_integration_bridge
            from nyx.core.integration.reasoning_cognitive_bridge import create_reasoning_cognitive_bridge
            from nyx.core.integration.relationship_tom_bridge import create_relationship_tom_bridge
            from nyx.core.integration.reward_learning_bridge import create_reward_learning_bridge
            from nyx.core.integration.somatic_perception_bridge import create_somatic_perception_bridge
            from nyx.core.integration.tom_integration import create_tom_integrator
            from nyx.core.integration.synergy_optimizer import create_synergy_optimizer
            from nyx.core.relationship_reflection import RelationshipReflectionSystem
            from nyx.core.integration.spatial_integration_bridge import create_spatial_integration_bridge
            from nyx.core.integration.conditioning_integration_bridge import create_conditioning_integration_bridge
            
            # Dominance-related bridges
            from nyx.core.integration.dominance_integration_manager import create_dominance_integration_manager
            from nyx.core.integration.dominance_imagination_decision_bridge import create_dominance_imagination_decision_bridge
            from nyx.core.integration.dominance_memory_reflection_bridge import create_dominance_memory_reflection_bridge
            from nyx.core.integration.dominance_reward_identity_bridge import create_dominance_reward_identity_bridge
            
            # Register core system bridges first (no dependencies)
            await self.register_bridge("action_selector", create_action_selector(self.brain), [])
            await self.register_bridge("dynamic_attention", create_dynamic_attention_system(self.brain), [])
            await self.register_bridge("perceptual_integration", create_perceptual_integration_layer(self.brain), [])
            await self.register_bridge("somatic_perception", create_somatic_perception_bridge(self.brain), [])
            
            # Register secondary bridges with dependencies on core bridges
            await self.register_bridge("emotional_cognitive", create_emotional_cognitive_bridge(self.brain), 
                               ["dynamic_attention"])
            await self.register_bridge("emotional_hormonal", create_emotional_hormonal_bridge(self.brain), 
                               ["emotional_cognitive"])
            await self.register_bridge("mood_emotional", create_mood_emotional_bridge(self.brain), 
                               ["emotional_cognitive", "emotional_hormonal"])
            await self.register_bridge("multimodal_attention", create_multimodal_attention_bridge(self.brain), 
                               ["dynamic_attention", "perceptual_integration"])
            await self.register_bridge("memory_integration", create_memory_integration_bridge(self.brain), 
                               ["dynamic_attention"])
            await self.register_bridge("prediction_imagination", create_prediction_imagination_bridge(self.brain), 
                               [])
            await self.register_bridge("identity_imagination_emotional", create_identity_imagination_emotional_bridge(self.brain), 
                               ["emotional_cognitive", "prediction_imagination"])
            await self.register_bridge("narrative_memory_identity", create_narrative_memory_identity_nexus(self.brain), 
                               ["memory_integration", "identity_imagination_emotional"])
            await self.register_bridge("procedural_memory", create_procedural_memory_integration_bridge(self.brain), 
                               ["memory_integration", "emotional_cognitive"])
            
            # Register bridges that depend on secondary bridges
            await self.register_bridge("knowledge_memory_reasoning", create_knowledge_memory_reasoning_bridge(self.brain), 
                               ["memory_integration"])
            await self.register_bridge("knowledge_curiosity", create_knowledge_curiosity_exploration_bridge(self.brain), 
                               ["knowledge_memory_reasoning"])
            await self.register_bridge("reasoning_cognitive", create_reasoning_cognitive_bridge(self.brain), 
                               ["emotional_cognitive", "memory_integration"])
            await self.register_bridge("tom_integrator", create_tom_integrator(self.brain), 
                               ["emotional_cognitive"])
            await self.register_bridge("relationship_tom", create_relationship_tom_bridge(self.brain), 
                               ["tom_integrator", "memory_integration"])
            await self.register_bridge("reward_learning", create_reward_learning_bridge(self.brain), 
                               ["action_selector", "memory_integration"])
            await self.register_bridge("core_systems_integration", create_core_systems_integration_bridge(self.brain), 
                               ["action_selector"])
            await self.register_bridge("autonomous_cognitive", create_autonomous_cognitive_bridge(self.brain), 
                               ["emotional_cognitive", "memory_integration", "reasoning_cognitive"])
    
            await self.register_bridge("conditioning_integration", create_conditioning_integration_bridge(self.brain), 
                               ["reward_learning", "memory_integration"])
            
            # Register dominance-related bridges
            await self.register_bridge("dominance_reward_identity", create_dominance_reward_identity_bridge(self.brain), 
                               ["reward_learning", "identity_imagination_emotional"])
            await self.register_bridge("dominance_memory_reflection", create_dominance_memory_reflection_bridge(self.brain), 
                               ["memory_integration"])
            await self.register_bridge("dominance_imagination_decision", create_dominance_imagination_decision_bridge(self.brain), 
                               ["prediction_imagination", "relationship_tom"])
            
            # Register main dominance integration manager
            await self.register_bridge("dominance_integration", create_dominance_integration_manager(self.brain), 
                               ["dominance_reward_identity", "dominance_memory_reflection", "dominance_imagination_decision"])
    
            await self.register_bridge("spatial_integration", create_spatial_integration_bridge(self.brain), 
                                ["memory_integration", "dynamic_attention"])
            
            # Register high-level pipeline bridges
            await self.register_bridge("need_goal_action", create_need_goal_action_pipeline(self.brain), 
                               ["action_selector", "reward_learning"])
            await self.register_bridge("decision_action_coordinator", create_decision_action_coordinator(self.brain), 
                               ["action_selector", "need_goal_action", "prediction_imagination"])
    
            await self.register_bridge("synergy_optimizer", create_synergy_optimizer(self.brain), 
                         ["event_bus", "memory_integration", "dynamic_attention"])
    
            self.brain.relationship_reflection_system = RelationshipReflectionSystem(
                relationship_manager=self.brain.relationship_manager,
                theory_of_mind=self.brain.theory_of_mind,
                memory_core=self.brain.memory_core,
                identity_evolution=self.brain.identity_evolution,
                hormone_system=self.brain.hormone_system if hasattr(self.brain, "hormone_system") else None
            )
            
            logger.info(f"Set up {len(self.bridges)} integration bridges")
            
        except Exception as e:
            logger.error(f"Error setting up integration bridges asynchronously: {e}", exc_info=True)
            raise # Re-raise to indicate setup failure
    
    async def register_bridge(self, 
                              bridge_name: str, 
                              bridge_instance: Any, 
                              dependencies: List[str] = None) -> None:
        if bridge_name in self.bridges:
            logger.warning(f"Bridge '{bridge_name}' is already registered. Overwriting.")
        
        self.bridges[bridge_name] = bridge_instance
        self.bridge_dependencies[bridge_name] = dependencies or []
        logger.info(f"Bridge '{bridge_name}' registered with dependencies: {self.bridge_dependencies[bridge_name]}.")
        return
    
    @trace_method(level=TraceLevel.INFO, group_id="IntegrationManager")
    async def initialize(self) -> bool:
        """Initialize all integration bridges in correct order."""
        try:
            # Call async setup first
            await self._setup_bridges() # <<< CALL IT HERE

            # Build dependency graph (can stay sync if it only reads self.bridges/self.bridge_dependencies)
            self._build_dependency_graph() 
            
            # Determine initialization sequence (can stay sync)
            self._determine_initialization_sequence()
            
            # Initialize bridges in sequence
            for bridge_name in self.initialization_sequence:
                if bridge_name in self.bridges:
                    bridge = self.bridges[bridge_name]
                    if bridge is None:
                        logger.warning(f"Bridge {bridge_name} not available, skipping")
                        self.bridge_statuses[bridge_name] = False # Mark as failed if None
                        continue
                    
                    logger.info(f"Initializing bridge: {bridge_name}")
                    if hasattr(bridge, 'initialize') and callable(getattr(bridge, 'initialize')):
                        success = await bridge.initialize() # This bridge.initialize() MUST be async
                        self.bridge_statuses[bridge_name] = success
                        if not success:
                            logger.warning(f"Bridge {bridge_name} initialization failed")
                    else:
                        logger.warning(f"Bridge {bridge_name} has no initialize method")
                        self.bridge_statuses[bridge_name] = True 
            
            failed_bridges = [name for name, status in self.bridge_statuses.items() if not status]
            if failed_bridges:
                logger.error(f"Failed to initialize bridges: {failed_bridges}")
                self.initialized = False
                return False
            
            self.initialized = True
            logger.info("IntegrationManager successfully initialized all bridges")
            return True
        except Exception as e:
            logger.error(f"Error initializing IntegrationManager: {e}", exc_info=True)
            self.initialized = False
            return False
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph for bridges."""
        # This is a simple implementation to ensure all dependencies are registered
        all_bridge_names = set(self.bridges.keys())
        
        # Check for unregistered dependencies
        for bridge_name, dependencies in self.bridge_dependencies.items():
            for dependency in dependencies:
                if dependency not in all_bridge_names:
                    logger.warning(f"Bridge {bridge_name} depends on unregistered bridge {dependency}")
    
    def _determine_initialization_sequence(self) -> None:
        """Determine correct initialization sequence based on dependencies."""
        # Simple topological sort implementation
        # Initialize structures
        remaining_bridges = set(self.bridges.keys())
        visited = set()
        temp_visited = set()
        order = []
        
        # Recursive DFS function
        def visit(bridge: str) -> None:
            if bridge in temp_visited:
                # Circular dependency detected
                logger.warning(f"Circular dependency detected involving {bridge}")
                return
            
            if bridge in visited:
                return
            
            temp_visited.add(bridge)
            
            # Visit all dependencies first
            for dependency in self.bridge_dependencies.get(bridge, []):
                # Skip non-registered dependencies
                if dependency in remaining_bridges:
                    visit(dependency)
            
            temp_visited.remove(bridge)
            visited.add(bridge)
            order.append(bridge)
        
        # Visit all bridges
        while remaining_bridges:
            bridge = next(iter(remaining_bridges))
            visit(bridge)
            remaining_bridges -= visited
        
        # Reverse the order to get the initialization sequence
        self.initialization_sequence = list(reversed(order))
        logger.info(f"Initialization sequence: {self.initialization_sequence}")
    
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
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Get detailed status for each bridge
        bridge_details = {}
        for name, bridge in self.bridges.items():
            if bridge is None:
                bridge_details[name] = {"available": False}
                continue
                
            # Get bridge status if it has a status method
            if hasattr(bridge, 'get_status'):
                try:
                    bridge_status = await bridge.get_status()
                    bridge_details[name] = bridge_status
                except Exception as e:
                    bridge_details[name] = {"error": str(e)}
            elif hasattr(bridge, 'get_bridge_state'):
                try:
                    bridge_status = await bridge.get_bridge_state()
                    bridge_details[name] = bridge_status
                except Exception as e:
                    bridge_details[name] = {"error": str(e)}
            elif hasattr(bridge, 'get_integration_status'):
                try:
                    bridge_status = await bridge.get_integration_status()
                    bridge_details[name] = bridge_status
                except Exception as e:
                    bridge_details[name] = {"error": str(e)}
            else:
                bridge_details[name] = {"available": True}
        
        status["bridge_details"] = bridge_details
        
        return status

# Function to create the integration manager
def create_integration_manager(nyx_brain):
    """Create an integration manager for the given brain."""
    return IntegrationManager(nyx_brain)
