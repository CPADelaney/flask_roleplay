# nyx/core/femdom/femdom_coordinator.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
import datetime

from nyx.core.integration.event_bus import Event, get_event_bus, DominanceEvent
from nyx.core.femdom.femdom_integration_manager import FemdomIntegrationManager

logger = logging.getLogger(__name__)

class FemdomCoordinator:
    """
    Central coordination system for all femdom capabilities.
    
    Manages the integration of all femdom components and provides
    high-level APIs for femdom interactions.
    """
    
    def __init__(self, nyx_brain):
        """Initialize the femdom coordinator."""
        self.brain = nyx_brain
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        
        # Reference core femdom components
        self.dominance_system = getattr(self.brain, "dominance_system", None)
        self.body_service = getattr(self.brain, "body_service_system", None)
        self.orgasm_control = getattr(self.brain, "orgasm_control", None)
        self.persona_manager = getattr(self.brain, "dominance_persona_manager", None)
        self.protocol_enforcement = getattr(self.brain, "protocol_enforcement", None)
        self.psychological_dominance = getattr(self.brain, "psychological_dominance", None)
        self.sadistic_responses = getattr(self.brain, "sadistic_responses", None)
        self.submission_progression = getattr(self.brain, "submission_progression", None)
        self.reward_system = getattr(self.brain, "reward_system", None)
        self.theory_of_mind = getattr(self.brain, "theory_of_mind", None)
        
        # Collect components for integration
        self.components = {
            "dominance_system": self.dominance_system,
            "body_service": self.body_service,
            "orgasm_control": self.orgasm_control,
            "persona_manager": self.persona_manager,
            "protocol_enforcement": self.protocol_enforcement,
            "psychological_dominance": self.psychological_dominance,
            "sadistic_responses": self.sadistic_responses,
            "submission_progression": self.submission_progression,
            "reward_system": self.reward_system,
            "theory_of_mind": self.theory_of_mind
        }
        
        # Create integration manager
        self.integration_manager = FemdomIntegrationManager(nyx_brain, self.components)
        
        # Active femdom sessions per user
        self.active_sessions = {}
        
        # Initialize system status
        self.initialized = False
        
        logger.info("FemdomCoordinator created")
    
    async def initialize(self):
        """Initialize the femdom coordinator and all components."""
        try:
            # Initialize integration manager
            await self.integration_manager.initialize()
            
            # Subscribe to essential events
            self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
            
            self.initialized = True
            logger.info("FemdomCoordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FemdomCoordinator: {e}")
            return False
    
    async def _handle_user_interaction(self, event):
        """Handle user interaction events."""
        user_id = event.data.get("user_id")
        content = event.data.get("content", "")
        
        # Update active session for this user
        await self._ensure_active_session(user_id)
        
        # Record interaction in session
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            session["interactions"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "content": content[:100]  # Store truncated content
            })
    
    async def _ensure_active_session(self, user_id: str) -> Dict[str, Any]:
        """Ensure an active femdom session exists for the user."""
        if user_id not in self.active_sessions:
            # Create new session
            session = {
                "user_id": user_id,
                "started_at": datetime.datetime.now().isoformat(),
                "active_persona": None,
                "dominance_level": 0.5,  # Default level
                "interactions": [],
                "active_protocols": [],
                "training_program": None
            }
            
            # Try to get existing dominance level from relationship
            if hasattr(self.brain, "relationship_manager") and self.brain.relationship_manager:
                try:
                    relationship = await self.brain.relationship_manager.get_relationship_state(user_id)
                    if relationship and hasattr(relationship, "dominance_level"):
                        session["dominance_level"] = relationship.dominance_level
                except Exception as e:
                    logger.error(f"Error getting relationship data: {e}")
            
            # Recommend and set persona
            if self.persona_manager:
                try:
                    recommendation = await self.persona_manager.recommend_persona(user_id)
                    if recommendation and recommendation.get("primary_recommendation"):
                        persona_id = recommendation["primary_recommendation"]["id"]
                        await self.persona_manager.activate_persona(
                            user_id, persona_id, session["dominance_level"]
                        )
                        session["active_persona"] = persona_id
                except Exception as e:
                    logger.error(f"Error setting persona: {e}")
            
            # Get active protocols
            if self.protocol_enforcement:
                try:
                    protocols = await self.protocol_enforcement.get_active_protocols(user_id)
                    if protocols:
                        session["active_protocols"] = protocols
                except Exception as e:
                    logger.error(f"Error getting protocols: {e}")
            
            # Get submission level
            if self.submission_progression:
                try:
                    submission_data = await self.submission_progression.get_user_submission_data(user_id)
                    if submission_data:
                        session["submission_level"] = submission_data.get("submission_level", {}).get("id", 1)
                except Exception as e:
                    logger.error(f"Error getting submission data: {e}")
            
            # Store session
            self.active_sessions[user_id] = session
            
            # Publish session start event
            await self.event_bus.publish(Event(
                event_type="femdom_session_started",
                source="femdom_coordinator",
                data=session
            ))
            
        return self.active_sessions[user_id]
    
    async def process_user_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Process a user message through all femdom systems.
        
        This is a high-level method for external usage that coordinates
        all relevant femdom processing for a user message.
        
        Args:
            user_id: The user ID
            message: The user's message
            
        Returns:
            Comprehensive processing results from all systems
        """
        try:
            # Ensure active session
            session = await self._ensure_active_session(user_id)
            
            # Prepare results structure
            results = {
                "user_id": user_id,
                "message_processed": True,
                "protocol_compliance": None,
                "submission_signals": None,
                "mental_state": None,
                "dominance_response": None,
                "recommended_actions": []
            }
            
            # 1. Check protocol compliance
            if self.protocol_enforcement:
                try:
                    protocol_check = await self.protocol_enforcement.check_protocol_compliance(
                        user_id, message
                    )
                    results["protocol_compliance"] = protocol_check
                    
                    # If protocol violation, recommend appropriate response
                    if not protocol_check.get("compliant", True):
                        dominance_bridge = self.integration_manager.bridges.get("dominance_coordinator")
                        if dominance_bridge:
                            violation_response = await dominance_bridge.respond_to_violation(
                                user_id, protocol_check.get("violations", [{}])[0]
                            )
                            
                            if violation_response and violation_response.get("success"):
                                results["recommended_actions"].append({
                                    "action_type": "protocol_violation_response",
                                    "response": violation_response.get("response"),
                                    "priority": 0.9  # High priority
                                })
                except Exception as e:
                    logger.error(f"Error checking protocol compliance: {e}")
            
            # 2. Detect submission signals
            submission_bridge = self.integration_manager.bridges.get("submission_progression_bridge")
            if submission_bridge:
                try:
                    submission_signals = await submission_bridge.detect_submission(user_id, message)
                    results["submission_signals"] = submission_signals
                    
                    # If high submission detected, recommend dominance reinforcement
                    if submission_signals and submission_signals.get("submission_level", 0.0) > 0.7:
                        results["recommended_actions"].append({
                            "action_type": "submission_reinforcement",
                            "submission_type": submission_signals.get("submission_type", "general"),
                            "priority": 0.8
                        })
                except Exception as e:
                    logger.error(f"Error detecting submission signals: {e}")
            
            # 3. Update mental state model
            if self.theory_of_mind:
                try:
                    mental_state = await self.theory_of_mind.update_user_model(
                        user_id, {"user_input": message}
                    )
                    results["mental_state"] = mental_state
                except Exception as e:
                    logger.error(f"Error updating mental state: {e}")
            
            # 4. Check for subspace
            psychological_bridge = self.integration_manager.bridges.get("psychological_dominance_bridge")
            if psychological_bridge:
                try:
                    subspace_check = await psychological_bridge.check_subspace(user_id)
                    results["subspace"] = subspace_check
                    
                    # If in subspace, adjust recommended actions
                    if subspace_check.get("in_subspace", False):
                        results["recommended_actions"].append({
                            "action_type": "subspace_response",
                            "depth": subspace_check.get("depth", 0),
                            "guidance": subspace_check.get("guidance", ""),
                            "priority": 1.0  # Highest priority
                        })
                except Exception as e:
                    logger.error(f"Error checking subspace: {e}")
            
            # 5. Generate recommended dominance action based on context
            dominance_bridge = self.integration_manager.bridges.get("dominance_coordinator")
            if dominance_bridge:
                try:
                    # Use current session dominance level
                    dominance_level = session.get("dominance_level", 0.5)
                    
                    # Generate dominance ideas if appropriate
                    submission_level = results.get("submission_signals", {}).get("submission_level", 0.0)
                    mental_valence = results.get("mental_state", {}).get("valence", 0.0)
                    
                    # Only recommend dominance action if user is receptive
                    if submission_level > 0.4 or mental_valence > 0.2:
                        dominance_action = "dominance_idea"
                        dominance_intensity = dominance_level
                        
                        # Adjust based on submission level
                        dominance_intensity = min(0.9, dominance_intensity + (submission_level * 0.2))
                        
                        # Check for subspace - reduce intensity if deep
                        if results.get("subspace", {}).get("in_subspace", False) and \
                           results.get("subspace", {}).get("depth_category") == "deep":
                            dominance_intensity = max(0.3, dominance_intensity * 0.7)
                            dominance_action = "psychological_mindfuck"  # Better for subspace
                        
                        # Process dominance action
                        action_result = await dominance_bridge.process_dominance_action(
                            dominance_action, user_id, dominance_intensity
                        )
                        
                        if action_result:
                            results["dominance_response"] = action_result
                except Exception as e:
                    logger.error(f"Error generating dominance action: {e}")
            
            # Return comprehensive results
            return results
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_dominance_response(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a dominance response for a specific user.
        
        This is a simplified API for getting a dominance response without
        the full message processing pipeline.
        
        Args:
            user_id: The user ID
            context: Optional context information
            
        Returns:
            Generated dominance response
        """
        try:
            # Ensure active session
            session = await self._ensure_active_session(user_id)
            context = context or {}
            
            # Get dominance level from session
            dominance_level = session.get("dominance_level", 0.5)
            
            # Get requested response type from context or use default
            response_type = context.get("response_type", "general")
            intensity_override = context.get("intensity")
            intensity = intensity_override if intensity_override is not None else dominance_level
            
            # Route to appropriate system based on requested type
            if response_type.startswith("psychological_"):
                # Psychological dominance response
                if self.psychological_dominance:
                    subtype = response_type.replace("psychological_", "")
                    
                    if subtype == "mindfuck":
                        user_state = await self._get_user_state(user_id)
                        return await self.psychological_dominance.generate_mindfuck(
                            user_id, user_state, intensity
                        )
                    elif subtype == "gaslighting":
                        return await self.psychological_dominance.apply_gaslighting(
                            user_id, None, intensity
                        )
            
            elif response_type.startswith("sadistic_"):
                # Sadistic response
                if self.sadistic_responses:
                    subtype = response_type.replace("sadistic_", "")
                    category = subtype if subtype in ["mockery", "amusement", "degradation"] else "amusement"
                    
                    return await self.sadistic_responses.generate_sadistic_amusement_response(
                        user_id, intensity, category=category
                    )
            
            elif response_type == "dominance_idea" or response_type == "general":
                # Generate dominance ideas
                if self.dominance_system:
                    purpose = context.get("purpose", "general")
                    intensity_range = f"{int(intensity * 10) - 2}-{int(intensity * 10)}"
                    
                    return await self.dominance_system.generate_dominance_ideas(
                        user_id, purpose, intensity_range, intensity > 0.7
                    )
            
            elif response_type == "orgasm_control":
                # Orgasm control response
                if self.orgasm_control:
                    action = context.get("action", "process_permission_request")
                    
                    if action == "process_permission_request":
                        request_text = context.get("request_text", "May I orgasm?")
                        return await self.orgasm_control.process_permission_request(
                            user_id, request_text, context
                        )
                    elif action == "start_denial":
                        duration_hours = context.get("duration_hours", 24)
                        level = min(5, int(intensity * 6))  # Scale 0-5
                        
                        return await self.orgasm_control.start_denial_period(
                            user_id, duration_hours, level, context.get("begging_allowed", True)
                        )
            
            elif response_type == "protocol_violation":
                # Protocol violation response
                dominance_bridge = self.integration_manager.bridges.get("dominance_coordinator")
                if dominance_bridge:
                    violation = context.get("violation", {})
                    return await dominance_bridge.respond_to_violation(user_id, violation)
            
            # Default fallback - generate generic dominance idea
            if self.dominance_system:
                intensity_range = f"{int(intensity * 10) - 2}-{int(intensity * 10)}"
                return await self.dominance_system.generate_dominance_ideas(
                    user_id, "general", intensity_range, intensity > 0.7
                )
            
            # Ultimate fallback if no systems available
            return {
                "success": False,
                "message": "No dominance systems available",
                "response": "I'm unable to generate a dominance response at this time."
            }
            
        except Exception as e:
            logger.error(f"Error generating dominance response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "An error occurred while generating a dominance response."
            }
    
    async def start_training_program(self, 
                                 user_id: str, 
                                 focus_area: Optional[str] = None,
                                 duration_days: int = 7) -> Dict[str, Any]:
        """
        Start a structured training program for a user.
        
        Args:
            user_id: The user ID
            focus_area: Optional specific focus area, or auto-recommended if None
            duration_days: Duration of the training program in days
            
        Returns:
            Created training program details
        """
        try:
            # Ensure active session
            session = await self._ensure_active_session(user_id)
            
            # Check if submission progression is available
            if not self.submission_progression:
                return {
                    "success": False,
                    "message": "Submission progression system not available"
                }
            
            # Get current submission level
            submission_data = await self.submission_progression.get_user_submission_data(user_id)
            current_level = submission_data["submission_level"]["id"]
            
            # Get training focus if not specified
            if not focus_area:
                # Use recommended focus from current level
                level_info = self.submission_progression.submission_levels[current_level]
                if level_info and level_info.training_focus:
                    focus_area = level_info.training_focus[0]  # Use first recommended focus
                else:
                    focus_area = "general"  # Default
            
            # Create training program
            program = {
                "user_id": user_id,
                "focus_area": focus_area,
                "submission_level": current_level,
                "duration_days": duration_days,
                "start_date": datetime.datetime.now().isoformat(),
                "tasks": [],
                "protocols": [],
                "rituals": [],
                "milestones": []
            }
            
            # Generate training content based on focus area
            if focus_area == "protocol_adherence":
                # Add protocol training
                if self.protocol_enforcement:
                    protocols = ["address_protocol", "permission_protocol"]
                    for protocol_id in protocols:
                        result = await self.protocol_enforcement.assign_protocol(user_id, protocol_id)
                        if result.get("success", False):
                            program["protocols"].append(result)
                
                # Add tasks focused on following protocols
                if self.body_service:
                    task_result = await self.body_service.assign_service_task(user_id, "recite_rules")
                    if task_result.get("success", False):
                        program["tasks"].append({
                            "type": "body_service",
                            "task": task_result
                        })
            
            elif focus_area == "service":
                # Add service training
                if self.body_service:
                    service_tasks = ["serve_beverage", "extended_kneeling", "verbal_worship"]
                    for task_id in service_tasks:
                        task_result = await self.body_service.assign_service_task(user_id, task_id)
                        if task_result.get("success", False):
                            program["tasks"].append({
                                "type": "body_service",
                                "task": task_result
                            })
            
            elif focus_area == "psychological":
                # Add psychological submission training
                if self.psychological_dominance:
                    # Add mind games appropriate for level
                    user_state = await self._get_user_state(user_id)
                    game_result = await self.psychological_dominance.generate_mindfuck(
                        user_id, 
                        user_state,
                        0.3 + (current_level * 0.1)  # Scale intensity with level
                    )
                    
                    if game_result.get("success", False):
                        program["tasks"].append({
                            "type": "psychological",
                            "task": game_result
                        })
            
            elif focus_area == "orgasm_control":
                # Add orgasm control training
                if self.orgasm_control:
                    # Start with appropriate level denial
                    denial_level = min(3, current_level)  # Scale 1-3 for early levels
                    duration = 12 * current_level  # Hours based on level
                    
                    result = await self.orgasm_control.start_denial_period(
                        user_id, duration, denial_level
                    )
                    
                    if result:
                        program["tasks"].append({
                            "type": "orgasm_control",
                            "task": result
                        })
            
            # Add milestones for completion
            milestone_results = await self.submission_progression.check_milestone_progress(user_id)
            if milestone_results.get("success", False):
                program["milestones"] = milestone_results.get("upcoming_milestones", [])
            
            # Store in session
            session["training_program"] = program
            
            # Publish training program started event
            await self.event_bus.publish(Event(
                event_type="training_program_started",
                source="femdom_coordinator",
                data={
                    "user_id": user_id,
                    "focus_area": focus_area,
                    "level": current_level,
                    "duration_days": duration_days
                }
            ))
            
            return {
                "success": True,
                "program": program
            }
            
        except Exception as e:
            logger.error(f"Error starting training program: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get current user state for context."""
        user_state = {"user_id": user_id}
        
        # Get theory of mind data if available
        if self.theory_of_mind:
            try:
                mental_state = await self.theory_of_mind.get_user_model(user_id)
                if mental_state:
                    user_state.update(mental_state)
            except Exception as e:
                logger.error(f"Error getting mental state: {e}")
        
        # Add session state
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            user_state["dominance_level"] = session.get("dominance_level", 0.5)
            user_state["active_persona"] = session.get("active_persona")
            user_state["submission_level"] = session.get("submission_level", 1)
        
        return user_state
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current femdom coordinator status."""
        return {
            "initialized": self.initialized,
            "active_sessions": len(self.active_sessions),
            "components": {
                "dominance_system": self.dominance_system is not None,
                "body_service": self.body_service is not None,
                "orgasm_control": self.orgasm_control is not None,
                "persona_manager": self.persona_manager is not None,
                "protocol_enforcement": self.protocol_enforcement is not None,
                "psychological_dominance": self.psychological_dominance is not None,
                "sadistic_responses": self.sadistic_responses is not None,
                "submission_progression": self.submission_progression is not None
            },
            "integration_manager": await self.integration_manager.get_status()
        }
