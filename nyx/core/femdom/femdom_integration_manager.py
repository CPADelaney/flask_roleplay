# nyx/core/femdom/femdom_integration_manager.py

import logging
import asyncio
from typing import Dict, List, Any, Optional

from nyx.core.integration.event_bus import Event, get_event_bus, DominanceEvent
from nyx.core.integration.integrated_tracer import trace_method, TraceLevel

logger = logging.getLogger(__name__)

class FemdomIntegrationManager:
    """Manages integration of all femdom-related systems."""
    
    def __init__(self, nyx_brain, components=None):
        """Initialize the femdom integration manager."""
        self.brain = nyx_brain
        self.event_bus = get_event_bus()
        self.bridges = {}
        self.initialized = False
        self.components = components or {}
        
        # Register for relevant events
        self._setup_event_handlers()
        
        # Register all bridges
        self._setup_bridges()
        
        logger.info("FemdomIntegrationManager initialized")
    
    def _setup_event_handlers(self):
        """Set up handlers for relevant events."""
        self.event_bus.subscribe("dominance_action", self._handle_dominance_event)
        self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
        
        # Additional subscriptions as needed
        self.event_bus.subscribe("emotional_state_change", self._handle_emotional_change)
        self.event_bus.subscribe("physical_sensation", self._handle_physical_sensation)
    
    async def _handle_dominance_event(self, event):
        """Handle dominance-related events."""
        try:
            action = event.data.get("action")
            user_id = event.data.get("user_id")
            intensity = event.data.get("intensity", 0.5)
            
            # Log the event
            logger.info(f"Processing dominance action: {action} for user {user_id} (intensity: {intensity})")
            
            # Trigger appropriate responses in related systems
            if "dominance_coordinator" in self.bridges:
                await self.bridges["dominance_coordinator"].process_dominance_action(action, user_id, intensity)
                
        except Exception as e:
            logger.error(f"Error handling dominance event: {e}")
    
    async def _handle_user_interaction(self, event):
        """Handle user interaction events."""
        try:
            user_id = event.data.get("user_id")
            content = event.data.get("content", "")
            
            # Check for protocol violations
            if "protocol_enforcement_bridge" in self.bridges:
                protocol_results = await self.bridges["protocol_enforcement_bridge"].check_protocols(user_id, content)
                
                # If violations found, process them
                if protocol_results and not protocol_results.get("compliant", True):
                    await self._process_protocol_violations(user_id, protocol_results)
            
            # Check for submission signals
            if "submission_progression_bridge" in self.bridges:
                submission_results = await self.bridges["submission_progression_bridge"].detect_submission(user_id, content)
                
                # If submission detected, process it
                if submission_results and submission_results.get("submission_detected", False):
                    await self._process_submission_signals(user_id, submission_results)
        
        except Exception as e:
            logger.error(f"Error handling user interaction: {e}")
    
    async def _handle_emotional_change(self, event):
        """Handle emotional state changes."""
        try:
            emotion = event.data.get("emotion")
            intensity = event.data.get("intensity", 0.5)
            
            # Propagate emotional changes to relevant dominance systems
            if emotion in ["dominant", "sadistic", "powerful", "cruel"] and intensity > 0.6:
                # Intensity boost for dominance systems
                if "psychological_dominance_bridge" in self.bridges:
                    await self.bridges["psychological_dominance_bridge"].amplify_dominance(intensity)
        
        except Exception as e:
            logger.error(f"Error handling emotional change: {e}")
    
    async def _handle_physical_sensation(self, event):
        """Handle physical sensation events."""
        # Implementation for handling physical sensations
        pass
    
    async def _process_protocol_violations(self, user_id, violation_results):
        """Process protocol violations."""
        try:
            violations = violation_results.get("violations", [])
            
            for violation in violations:
                # Record the violation
                if "protocol_enforcement_bridge" in self.bridges:
                    await self.bridges["protocol_enforcement_bridge"].record_violation(
                        user_id, 
                        violation.get("protocol_id"),
                        violation.get("description", "Protocol violation")
                    )
                
                # Generate appropriate dominance response
                if "dominance_coordinator" in self.bridges:
                    await self.bridges["dominance_coordinator"].respond_to_violation(
                        user_id, violation
                    )
                    
                # Update submission metrics
                if "submission_progression_bridge" in self.bridges:
                    await self.bridges["submission_progression_bridge"].update_compliance(
                        user_id, False, 0.5, "Protocol violation"
                    )
        
        except Exception as e:
            logger.error(f"Error processing protocol violations: {e}")
    
    async def _process_submission_signals(self, user_id, submission_results):
        """Process detected submission signals."""
        try:
            submission_level = submission_results.get("submission_level", 0.0)
            submission_type = submission_results.get("submission_type", "general")
            
            # Process submission with reward system
            if "reward_integration_bridge" in self.bridges:
                await self.bridges["reward_integration_bridge"].process_submission_reward(
                    user_id, submission_type, submission_level
                )
                
            # Update submission progression
            if "submission_progression_bridge" in self.bridges:
                await self.bridges["submission_progression_bridge"].update_submission_metric(
                    user_id, "depth", submission_level * 0.1, f"Detected {submission_type} submission"
                )
                
            # Check for subspace indications
            if submission_level > 0.7 and "psychological_dominance_bridge" in self.bridges:
                subspace_check = await self.bridges["psychological_dominance_bridge"].check_subspace(user_id)
                
                if subspace_check.get("in_subspace", False):
                    # Publish subspace event
                    await self.event_bus.publish(Event(
                        event_type="subspace_detected",
                        source="femdom_integration_manager",
                        data={
                            "user_id": user_id,
                            "depth": subspace_check.get("depth", 0.0),
                            "duration": subspace_check.get("duration", 0)
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error processing submission signals: {e}")
    
    def _setup_bridges(self):
        """Set up all femdom integration bridges."""
        try:
            # Create core bridges
            self.bridges["dominance_coordinator"] = DominanceCoordinatorBridge(
                self.brain, 
                dominance_system=self.components.get("dominance_system"),
                sadistic_responses=self.components.get("sadistic_responses"),
                psychological_dominance=self.components.get("psychological_dominance")
            )
            
            self.bridges["protocol_enforcement_bridge"] = ProtocolEnforcementBridge(
                self.brain,
                protocol_enforcement=self.components.get("protocol_enforcement"),
                reward_system=self.components.get("reward_system")
            )
            
            self.bridges["orgasm_control_bridge"] = OrgasmControlBridge(
                self.brain,
                orgasm_control=self.components.get("orgasm_control"),
                reward_system=self.components.get("reward_system")
            )
            
            self.bridges["submission_progression_bridge"] = SubmissionProgressionBridge(
                self.brain,
                submission_progression=self.components.get("submission_progression"),
                theory_of_mind=self.components.get("theory_of_mind")
            )
            
            self.bridges["persona_management_bridge"] = PersonaManagementBridge(
                self.brain,
                persona_manager=self.components.get("persona_manager")
            )
            
            self.bridges["psychological_dominance_bridge"] = PsychologicalDominanceBridge(
                self.brain,
                psychological_dominance=self.components.get("psychological_dominance"),
                theory_of_mind=self.components.get("theory_of_mind")
            )
            
            self.bridges["reward_integration_bridge"] = RewardIntegrationBridge(
                self.brain,
                reward_system=self.components.get("reward_system")
            )
            
            self.bridges["body_service_bridge"] = BodyServiceBridge(
                self.brain,
                body_service=self.components.get("body_service")
            )
            
            logger.info(f"Set up {len(self.bridges)} femdom integration bridges")
            
        except Exception as e:
            logger.error(f"Error setting up femdom integration bridges: {e}", exc_info=True)
    
    @trace_method(level=TraceLevel.INFO, group_id="FemdomIntegration")
    async def initialize(self):
        """Initialize all femdom integration bridges."""
        try:
            # Initialize all bridges
            for name, bridge in self.bridges.items():
                logger.info(f"Initializing femdom bridge: {name}")
                
                if hasattr(bridge, 'initialize') and callable(getattr(bridge, 'initialize')):
                    await bridge.initialize()
            
            self.initialized = True
            logger.info("Femdom integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing femdom integration: {e}", exc_info=True)
            return False
    
    async def get_status(self):
        """Get current integration status."""
        status = {
            "initialized": self.initialized,
            "active_bridges": list(self.bridges.keys()),
            "bridge_status": {}
        }
        
        # Collect status from all bridges
        for name, bridge in self.bridges.items():
            if hasattr(bridge, 'get_status'):
                try:
                    bridge_status = await bridge.get_status()
                    status["bridge_status"][name] = bridge_status
                except Exception as e:
                    status["bridge_status"][name] = {"error": str(e)}
            else:
                status["bridge_status"][name] = {"status": "unknown"}
        
        return status
