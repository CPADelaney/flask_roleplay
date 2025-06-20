# nyx/core/femdom/femdom_coordinator.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
import datetime

from agents import Agent, Runner, function_tool, trace, handoff, RunContextWrapper, ModelSettings, gen_trace_id
from agents.run import RunConfig
from pydantic import BaseModel

from nyx.core.integration.event_bus import Event, get_event_bus, DominanceEvent

logger = logging.getLogger(__name__)

# Pydantic models for function tool inputs/outputs
class UserSessionResponse(BaseModel):
    user_id: str
    active_persona: Optional[str]
    dominance_level: float
    active_protocols: List[str]
    submission_level: int
    has_training_program: bool

class ProtocolComplianceResponse(BaseModel):
    compliant: bool
    violations: List[str]
    error: Optional[str] = None

class DominanceLevelResponse(BaseModel):
    dominance_level: float
    user_id: str

class UserProtocolsResponse(BaseModel):
    active_protocols: List[str]
    user_id: str

class UserStateResponse(BaseModel):
    user_id: str
    dominance_level: Optional[float] = None
    active_persona: Optional[str] = None
    submission_level: Optional[int] = None

class DominanceIdeaResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class DominanceActionResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class ViolationInput(BaseModel):
    type: Optional[str] = None
    protocol_id: Optional[str] = None
    description: Optional[str] = None

class ViolationResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    violation_type: Optional[str] = None
    protocol_id: Optional[str] = None
    error: Optional[str] = None

class ProtocolAssignmentResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class ViolationRecordResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class MindfuckResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class GaslightingResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class SubspaceCheckResponse(BaseModel):
    in_subspace: bool
    confidence: Optional[float] = None
    guidance: Optional[str] = None
    error: Optional[str] = None

class PsychologicalStateResponse(BaseModel):
    has_state: bool
    error: Optional[str] = None

class SubmissionDataResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class SubmissionDetectionResponse(BaseModel):
    submission_detected: bool
    submission_level: Optional[float] = None
    submission_type: Optional[str] = None
    error: Optional[str] = None

class SubmissionMetricResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class MilestoneProgressResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class TrainingProgramResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class TrainingStatusResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    program: Optional[str] = None

class TaskAssignmentResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class TaskCompletionResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class FemdomContext:
    """Context object for femdom interactions."""
    
    def __init__(self, nyx_brain, user_id=None):
        self.brain = nyx_brain
        self.user_id = user_id
        self.event_bus = get_event_bus()
        self.session_data = {}
        self.dominance_level = 0.5
        self.active_persona = None
        self.active_protocols = []
        self.submission_level = 1
        self.training_program = None
        self.user_state = {}
    
    def update_session_data(self, data):
        self.session_data.update(data)
        
    def set_dominance_level(self, level):
        self.dominance_level = level
        
    def set_active_persona(self, persona_id):
        self.active_persona = persona_id
        
    def set_protocols(self, protocols):
        self.active_protocols = protocols
        
    def set_submission_level(self, level):
        self.submission_level = level
        
    def set_user_state(self, state):
        self.user_state.update(state)

class FemdomCoordinator:
    """
    Central coordination system for all femdom capabilities using the OpenAI Agents SDK.
    
    Manages the integration of all femdom components and provides
    high-level APIs for femdom interactions.
    """
    
    def __init__(self, nyx_brain):
        """Initialize the femdom coordinator."""
        self.brain = nyx_brain
    
        # First initialize all specialized agents
        self.dominance_agent = self._create_dominance_agent()
        self.protocol_agent = self._create_protocol_agent()
        self.psychological_agent = self._create_psychological_agent()
        self.submission_agent = self._create_submission_agent()
        self.training_agent = self._create_training_agent()
        
        # Then create the main agent that depends on them
        self.main_agent = self._create_main_agent()
        
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
        
        # Active femdom sessions per user
        self.active_sessions = {}
        
        # Initialize system status
        self.initialized = False
        
        logger.info("FemdomCoordinator created with OpenAI Agents SDK")
    
    def _create_main_agent(self):
        """Create the main femdom agent using the OpenAI Agents SDK."""
        return Agent(
            name="FemdomMainAgent",
            instructions="""You are the main coordination agent for a femdom AI system. Your role is to:
1. Process user messages and determine appropriate responses
2. Coordinate between specialized agents for different femdom aspects
3. Maintain coherent session state and user experience
4. Ensure all interactions adhere to established protocols and dynamics

Use the available tools to gather information about the user and their current state.
Delegate specialized tasks to the appropriate agent via handoffs.
""",
            tools=[
                self._get_user_session,
                self._check_protocol_compliance,
                self._get_dominance_level,
                self._get_user_protocols,
                self._get_user_state
            ],
            handoffs=[
                handoff(self.dominance_agent, 
                       tool_name_override="delegate_to_dominance_agent",
                       tool_description_override="Delegate to the dominance agent for dominance-related responses"),
                
                handoff(self.protocol_agent,
                       tool_name_override="delegate_to_protocol_agent",
                       tool_description_override="Delegate to the protocol agent for protocol enforcement"),
                
                handoff(self.psychological_agent,
                       tool_name_override="delegate_to_psychological_agent",
                       tool_description_override="Delegate to the psychological agent for psychological dominance"),
                
                handoff(self.submission_agent,
                       tool_name_override="delegate_to_submission_agent",
                       tool_description_override="Delegate to the submission agent for submission progression"),
                
                handoff(self.training_agent,
                       tool_name_override="delegate_to_training_agent",
                       tool_description_override="Delegate to the training agent for training programs")
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_dominance_agent(self):
        """Create the dominance agent for handling dominance-related responses."""
        return Agent(
            name="DominanceAgent",
            instructions="""You are the dominance specialist agent for a femdom AI system. Your role is to:
1. Generate appropriately dominant responses based on user context
2. Adapt dominance level to the user's receptiveness
3. Provide ideas for dominance actions and expressions
4. Process violations and generate appropriate responses

Use the available tools to gather information and generate responses.
""",
            tools=[
                self._get_dominance_level,
                self._generate_dominance_idea,
                self._process_dominance_action,
                self._respond_to_violation
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_protocol_agent(self):
        """Create the protocol agent for protocol enforcement."""
        return Agent(
            name="ProtocolAgent",
            instructions="""You are the protocol specialist agent for a femdom AI system. Your role is to:
1. Check user messages for protocol compliance
2. Enforce active protocols for the user
3. Assign and manage protocols
4. Generate appropriate responses to protocol violations

Use the available tools to check protocols and generate responses.
""",
            tools=[
                self._get_user_protocols,
                self._check_protocol_compliance,
                self._assign_protocol,
                self._record_violation
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_psychological_agent(self):
        """Create the psychological agent for psychological dominance."""
        return Agent(
            name="PsychologicalAgent",
            instructions="""You are the psychological specialist agent for a femdom AI system. Your role is to:
1. Generate psychological dominance tactics
2. Implement mind games and gaslighting when appropriate
3. Detect and respond to subspace
4. Monitor psychological state of users

Use the available tools to generate psychological dominance responses.
""",
            tools=[
                self._generate_mindfuck,
                self._apply_gaslighting,
                self._check_subspace,
                self._get_psychological_state
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_submission_agent(self):
        """Create the submission agent for submission progression."""
        return Agent(
            name="SubmissionAgent",
            instructions="""You are the submission specialist agent for a femdom AI system. Your role is to:
1. Track user's submission journey and progression
2. Detect submission signals in user messages
3. Update submission metrics
4. Check milestone progress
5. Generate progression reports

Use the available tools to work with submission progression.
""",
            tools=[
                self._get_submission_data,
                self._detect_submission,
                self._update_submission_metric,
                self._check_milestone_progress
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_training_agent(self):
        """Create the training agent for training programs."""
        return Agent(
            name="TrainingAgent",
            instructions="""You are the training specialist agent for a femdom AI system. Your role is to:
1. Create and manage structured training programs
2. Assign tasks and exercises
3. Track progress in training programs
4. Generate training reports and recommendations

Use the available tools to manage training programs.
""",
            tools=[
                self._start_training_program,
                self._get_training_status,
                self._assign_task,
                self._check_task_completion
            ],
            model="gpt-4.1-nano"
        )
    
    async def initialize(self):
        """Initialize the femdom coordinator and all components."""
        try:
            # Subscribe to essential events
            self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
            
            self.initialized = True
            logger.info("FemdomCoordinator initialized successfully with OpenAI Agents SDK")
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
    
    @function_tool
    async def _get_user_session(self, user_id: str) -> UserSessionResponse:
        """Get the current session data for a user."""
        session = await self._ensure_active_session(user_id)
        return UserSessionResponse(
            user_id=user_id,
            active_persona=session["active_persona"],
            dominance_level=session["dominance_level"],
            active_protocols=session["active_protocols"],
            submission_level=session.get("submission_level", 1),
            has_training_program=session["training_program"] is not None
        )
    
    @function_tool
    async def _check_protocol_compliance(self, user_id: str, message: str) -> ProtocolComplianceResponse:
        """Check if a user message complies with active protocols."""
        if not self.protocol_enforcement:
            return ProtocolComplianceResponse(compliant=True, violations=[])
        
        try:
            protocol_check = await self.protocol_enforcement.check_protocol_compliance(
                user_id, message
            )
            return ProtocolComplianceResponse(
                compliant=protocol_check.get("compliant", True),
                violations=protocol_check.get("violations", [])
            )
        except Exception as e:
            logger.error(f"Error checking protocol compliance: {e}")
            return ProtocolComplianceResponse(compliant=True, violations=[], error=str(e))
    
    @function_tool
    async def _get_dominance_level(self, user_id: str) -> DominanceLevelResponse:
        """Get the current dominance level for a user."""
        session = await self._ensure_active_session(user_id)
        return DominanceLevelResponse(
            dominance_level=session["dominance_level"],
            user_id=user_id
        )
    
    @function_tool
    async def _get_user_protocols(self, user_id: str) -> UserProtocolsResponse:
        """Get active protocols for a user."""
        session = await self._ensure_active_session(user_id)
        return UserProtocolsResponse(
            active_protocols=session["active_protocols"],
            user_id=user_id
        )
    
    @function_tool
    async def _get_user_state(self, user_id: str) -> UserStateResponse:
        """Get current user state."""
        return await self._get_user_state_internal(user_id)
    
    @function_tool
    async def _generate_dominance_idea(self, user_id: str, purpose: str = "general", intensity_range: str = "5-7") -> DominanceIdeaResponse:
        """Generate dominance ideas for a user."""
        if not self.dominance_system:
            return DominanceIdeaResponse(success=False, message="Dominance system not available")
        
        try:
            return DominanceIdeaResponse(success=True)
        except Exception as e:
            logger.error(f"Error generating dominance ideas: {e}")
            return DominanceIdeaResponse(success=False, error=str(e))
    
    @function_tool
    async def _process_dominance_action(self, action_type: str, user_id: str, intensity: float) -> DominanceActionResponse:
        """Process a dominance action for a user."""
        if not self.dominance_system:
            return DominanceActionResponse(success=False)
        
        try:
            # Create a dominance action event
            action_event = DominanceEvent(
                action=action_type,
                user_id=user_id,
                intensity=intensity
            )
            
            # Process the action
            result = await self.dominance_system.process_dominance_action(action_event)
            return DominanceActionResponse(success=True)
        except Exception as e:
            logger.error(f"Error processing dominance action: {e}")
            return DominanceActionResponse(success=False, error=str(e))
    
    @function_tool
    async def _respond_to_violation(self, user_id: str, violation: ViolationInput) -> ViolationResponse:
        """Generate a response to a protocol violation."""
        if not self.dominance_system:
            return ViolationResponse(success=False, response="Dominance system not available")
        
        try:
            # Generate a response based on the violation
            session = await self._ensure_active_session(user_id)
            dominance_level = session["dominance_level"]
            
            violation_dict = violation.model_dump()
            response = await self.dominance_system.generate_violation_response(
                user_id, violation_dict, dominance_level
            )
            
            return ViolationResponse(
                success=True,
                response=response,
                violation_type=violation.type,
                protocol_id=violation.protocol_id
            )
        except Exception as e:
            logger.error(f"Error responding to violation: {e}")
            return ViolationResponse(success=False, error=str(e))
    
    @function_tool
    async def _assign_protocol(self, user_id: str, protocol_id: str) -> ProtocolAssignmentResponse:
        """Assign a protocol to a user."""
        if not self.protocol_enforcement:
            return ProtocolAssignmentResponse(success=False)
        
        try:
            result = await self.protocol_enforcement.assign_protocol(user_id, protocol_id)
            
            # Update session if successful
            if result.get("success", False):
                session = await self._ensure_active_session(user_id)
                if protocol_id not in session["active_protocols"]:
                    session["active_protocols"].append(protocol_id)
            
            return ProtocolAssignmentResponse(success=result.get("success", False))
        except Exception as e:
            logger.error(f"Error assigning protocol: {e}")
            return ProtocolAssignmentResponse(success=False, error=str(e))
    
    @function_tool
    async def _record_violation(self, user_id: str, protocol_id: str, description: str) -> ViolationRecordResponse:
        """Record a protocol violation."""
        if not self.protocol_enforcement:
            return ViolationRecordResponse(success=False)
        
        try:
            await self.protocol_enforcement.record_violation(
                user_id, protocol_id, description
            )
            return ViolationRecordResponse(success=True)
        except Exception as e:
            logger.error(f"Error recording violation: {e}")
            return ViolationRecordResponse(success=False, error=str(e))
    
    @function_tool
    async def _generate_mindfuck(self, user_id: str, intensity: float) -> MindfuckResponse:
        """Generate a psychological mind game."""
        if not self.psychological_dominance:
            return MindfuckResponse(success=False)
        
        try:
            user_state = await self._get_user_state_internal(user_id)
            await self.psychological_dominance.generate_mindfuck(
                user_id, user_state, intensity
            )
            return MindfuckResponse(success=True)
        except Exception as e:
            logger.error(f"Error generating mindfuck: {e}")
            return MindfuckResponse(success=False, error=str(e))
    
    @function_tool
    async def _apply_gaslighting(self, user_id: str, intensity: float) -> GaslightingResponse:
        """Apply gaslighting strategy."""
        if not self.psychological_dominance:
            return GaslightingResponse(success=False)
        
        try:
            await self.psychological_dominance.apply_gaslighting(
                user_id, None, intensity
            )
            return GaslightingResponse(success=True)
        except Exception as e:
            logger.error(f"Error applying gaslighting: {e}")
            return GaslightingResponse(success=False, error=str(e))
    
    @function_tool
    async def _check_subspace(self, user_id: str) -> SubspaceCheckResponse:
        """Check if user is in subspace and get guidance."""
        if (not self.psychological_dominance or 
            not hasattr(self.psychological_dominance, "SubspaceDetection")):
            return SubspaceCheckResponse(in_subspace=False, confidence=0.0)
        
        try:
            # Get recent messages
            session = await self._ensure_active_session(user_id)
            recent_messages = [
                interaction["content"] 
                for interaction in session.get("interactions", [])[-5:]
            ]
            
            subspace_detection = self.psychological_dominance.SubspaceDetection()
            detection_result = await subspace_detection.detect_subspace(user_id, recent_messages)
            
            guidance = None
            if detection_result["subspace_detected"]:
                guidance_result = await subspace_detection.get_subspace_guidance(detection_result)
                guidance = str(guidance_result) if guidance_result else None
            
            return SubspaceCheckResponse(
                in_subspace=detection_result.get("subspace_detected", False),
                confidence=detection_result.get("confidence", 0.0),
                guidance=guidance
            )
        except Exception as e:
            logger.error(f"Error checking subspace: {e}")
            return SubspaceCheckResponse(in_subspace=False, error=str(e))
    
    @function_tool
    async def _get_psychological_state(self, user_id: str) -> PsychologicalStateResponse:
        """Get the current psychological state for a user."""
        if not self.psychological_dominance:
            return PsychologicalStateResponse(has_state=False)
        
        try:
            await self.psychological_dominance.get_user_psychological_state(user_id)
            return PsychologicalStateResponse(has_state=True)
        except Exception as e:
            logger.error(f"Error getting psychological state: {e}")
            return PsychologicalStateResponse(has_state=False, error=str(e))
    
    @function_tool
    async def _get_submission_data(self, user_id: str) -> SubmissionDataResponse:
        """Get submission data for a user."""
        if not self.submission_progression:
            return SubmissionDataResponse(success=False, message="Submission progression not available")
        
        try:
            await self.submission_progression.get_user_submission_data(user_id)
            return SubmissionDataResponse(success=True)
        except Exception as e:
            logger.error(f"Error getting submission data: {e}")
            return SubmissionDataResponse(success=False, error=str(e))
    
    @function_tool
    async def _detect_submission(self, user_id: str, message: str) -> SubmissionDetectionResponse:
        """Detect submission signals in a user message."""
        if not self.submission_progression:
            return SubmissionDetectionResponse(submission_detected=False)
        
        try:
            # This is a placeholder - you'll need to implement the actual detection
            # based on your current implementation
            submission_signals = {
                "submission_detected": False,
                "submission_level": 0.0,
                "submission_type": "none"
            }
            
            # Use theory of mind if available
            if self.theory_of_mind:
                try:
                    mental_state = await self.theory_of_mind.get_user_model(user_id)
                    if mental_state:
                        # Check for submission indicators
                        if mental_state.get("deference", 0) > 0.6 or mental_state.get("obedience", 0) > 0.7:
                            submission_signals["submission_detected"] = True
                            submission_signals["submission_level"] = mental_state.get("deference", 0) * 0.8
                            submission_signals["submission_type"] = "verbal" if "please" in message.lower() else "general"
                except Exception as e:
                    logger.error(f"Error using theory of mind: {e}")
            
            return SubmissionDetectionResponse(
                submission_detected=submission_signals["submission_detected"],
                submission_level=submission_signals["submission_level"],
                submission_type=submission_signals["submission_type"]
            )
        except Exception as e:
            logger.error(f"Error detecting submission: {e}")
            return SubmissionDetectionResponse(submission_detected=False, error=str(e))
    
    @function_tool
    async def _update_submission_metric(self, user_id: str, metric_name: str, value_change: float, reason: str) -> SubmissionMetricResponse:
        """Update a submission metric."""
        if not self.submission_progression:
            return SubmissionMetricResponse(success=False)
        
        try:
            await self.submission_progression.update_submission_metric(
                user_id, metric_name, value_change, reason
            )
            return SubmissionMetricResponse(success=True)
        except Exception as e:
            logger.error(f"Error updating submission metric: {e}")
            return SubmissionMetricResponse(success=False, error=str(e))
    
    @function_tool
    async def _check_milestone_progress(self, user_id: str) -> MilestoneProgressResponse:
        """Check milestone progress for a user."""
        if not self.submission_progression:
            return MilestoneProgressResponse(success=False)
        
        try:
            await self.submission_progression.check_milestone_progress(user_id)
            return MilestoneProgressResponse(success=True)
        except Exception as e:
            logger.error(f"Error checking milestone progress: {e}")
            return MilestoneProgressResponse(success=False, error=str(e))
    
    @function_tool
    async def _start_training_program(self, user_id: str, focus_area: Optional[str] = None, duration_days: int = 7) -> TrainingProgramResponse:
        """Start a structured training program for a user."""
        try:
            await self.start_training_program(user_id, focus_area, duration_days)
            return TrainingProgramResponse(success=True)
        except Exception as e:
            logger.error(f"Error starting training program: {e}")
            return TrainingProgramResponse(success=False, error=str(e))
    
    @function_tool
    async def _get_training_status(self, user_id: str) -> TrainingStatusResponse:
        """Get the status of a user's training program."""
        session = await self._ensure_active_session(user_id)
        
        if not session["training_program"]:
            return TrainingStatusResponse(success=False, message="No active training program")
        
        return TrainingStatusResponse(
            success=True,
            program=str(session["training_program"])
        )
    
    @function_tool
    async def _assign_task(self, user_id: str, task_type: str, description: str, due_in_hours: int = 24) -> TaskAssignmentResponse:
        """Assign a task to a user."""
        if not hasattr(self.brain, "task_assignment_system") or not self.brain.task_assignment_system:
            return TaskAssignmentResponse(success=False)
        
        try:
            await self.brain.task_assignment_system.assign_task(
                user_id=user_id,
                custom_task={
                    "title": f"{task_type.capitalize()} Task",
                    "description": description,
                    "category": task_type,
                    "instructions": [description]
                },
                due_in_hours=due_in_hours
            )
            return TaskAssignmentResponse(success=True)
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            return TaskAssignmentResponse(success=False, error=str(e))
    
    @function_tool
    async def _check_task_completion(self, task_id: str) -> TaskCompletionResponse:
        """Check if a task has been completed."""
        if not hasattr(self.brain, "task_assignment_system") or not self.brain.task_assignment_system:
            return TaskCompletionResponse(success=False)
        
        try:
            task_details = await self.brain.task_assignment_system.get_task_details(task_id)
            return TaskCompletionResponse(success=bool(task_details))
        except Exception as e:
            logger.error(f"Error checking task completion: {e}")
            return TaskCompletionResponse(success=False, error=str(e))
    
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
        Process a user message through all femdom systems using the Agent SDK.
        
        Args:
            user_id: The user ID
            message: The user's message
            
        Returns:
            Comprehensive processing results with response
        """
        # Create a femdom context for this interaction
        context = FemdomContext(self.brain, user_id)
        
        # Ensure active session
        session = await self._ensure_active_session(user_id)

        # Escalation logic based on gratification level
        try:
            if hasattr(self.brain, "somatosensory_system") and self.brain.somatosensory_system:
                body_state = await self.brain.somatosensory_system.get_body_state()
                gratification = body_state.get("gratification_level", 0.0)

                logger.debug(f"[Nyx] Detected gratification_level: {gratification:.2f}")

                # Escalate if gratification is high
                if gratification >= 0.7:
                    # Denial + new ritual
                    if self.orgasm_control:
                        await self.orgasm_control.start_denial_period(
                            user_id=user_id,
                            duration_hours=36,
                            level=3,  # STRICT
                            begging_allowed=True,
                            conditions={"min_begging": 3}
                        )

                    if self.protocol_enforcement:
                        await self.protocol_enforcement.assign_ritual(user_id, ritual_id="withheld_offering")

                    logger.info(f"[Nyx] Escalated rituals and denial for user {user_id} due to high gratification ({gratification:.2f})")

        except Exception as e:
            logger.warning(f"[Nyx] Failed to apply gratification-based escalation: {e}")
            
        context.set_dominance_level(session["dominance_level"])
        context.set_active_persona(session["active_persona"])
        context.set_protocols(session["active_protocols"])
        context.set_submission_level(session.get("submission_level", 1))
        
        # Generate a trace ID for this interaction
        trace_id = gen_trace_id()
        
        # Run the main agent
        try:
            result = await Runner.run(
                starting_agent=self.main_agent,
                input=message,
                context=context,
                run_config=RunConfig(
                    workflow_name="FemdomInteraction",
                    trace_id=trace_id,
                    group_id=user_id,
                    trace_metadata={
                        "user_id": user_id,
                        "dominance_level": session["dominance_level"],
                        "submission_level": session.get("submission_level", 1)
                    }
                )
            )
            
            # Extract the response
            response = result.final_output
            
            # Format results
            return {
                "user_id": user_id,
                "message_processed": True,
                "response": response,
                "trace_id": trace_id
            }
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return {
                "user_id": user_id,
                "message_processed": False,
                "error": str(e),
                "trace_id": trace_id
            }
    
    async def generate_dominance_response(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a dominance response for a specific user using the Agent SDK.
        
        Args:
            user_id: The user ID
            context: Optional context information
            
        Returns:
            Generated dominance response
        """
        # Create a femdom context for this interaction
        femdom_context = FemdomContext(self.brain, user_id)
        
        # Ensure active session
        session = await self._ensure_active_session(user_id)
        femdom_context.set_dominance_level(session["dominance_level"])
        femdom_context.set_active_persona(session["active_persona"])
        femdom_context.set_protocols(session["active_protocols"])
        femdom_context.set_submission_level(session.get("submission_level", 1))
        
        # Add context to the femdom context
        if context:
            femdom_context.update_session_data(context)
        
        # Generate a trace ID for this interaction
        trace_id = gen_trace_id()
        
        # Run the dominance agent directly
        try:
            result = await Runner.run(
                starting_agent=self.dominance_agent,
                input={
                    "user_id": user_id,
                    "request_type": "dominance_response",
                    "context": context or {}
                },
                context=femdom_context,
                run_config=RunConfig(
                    workflow_name="DominanceResponse",
                    trace_id=trace_id,
                    group_id=user_id,
                    trace_metadata={
                        "user_id": user_id,
                        "dominance_level": session["dominance_level"],
                        "context_type": context.get("response_type", "general") if context else "general"
                    }
                )
            )
            
            # Extract the response
            response = result.final_output
            
            return {
                "success": True,
                "response": response.get("response", response),
                "trace_id": trace_id
            }
            
        except Exception as e:
            logger.error(f"Error generating dominance response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "An error occurred while generating a dominance response.",
                "trace_id": trace_id
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
                    user_state = await self._get_user_state_internal(user_id)
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
    
    async def _get_user_state_internal(self, user_id: str) -> UserStateResponse:
        """Get current user state for context."""
        user_state = UserStateResponse(user_id=user_id)
        
        # Get theory of mind data if available
        if self.theory_of_mind:
            try:
                mental_state = await self.theory_of_mind.get_user_model(user_id)
                if mental_state:
                    # Update user_state with mental state data
                    pass
            except Exception as e:
                logger.error(f"Error getting mental state: {e}")
        
        # Add session state
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            user_state.dominance_level = session.get("dominance_level", 0.5)
            user_state.active_persona = session.get("active_persona")
            user_state.submission_level = session.get("submission_level", 1)
        
        return user_state
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current femdom coordinator status."""
        return {
            "initialized": self.initialized,
            "active_sessions": len(self.active_sessions),
            "agents": {
                "main_agent": self.main_agent is not None,
                "dominance_agent": self.dominance_agent is not None,
                "protocol_agent": self.protocol_agent is not None,
                "psychological_agent": self.psychological_agent is not None,
                "submission_agent": self.submission_agent is not None,
                "training_agent": self.training_agent is not None
            },
            "components": {
                "dominance_system": self.dominance_system is not None,
                "body_service": self.body_service is not None,
                "orgasm_control": self.orgasm_control is not None,
                "persona_manager": self.persona_manager is not None,
                "protocol_enforcement": self.protocol_enforcement is not None,
                "psychological_dominance": self.psychological_dominance is not None,
                "sadistic_responses": self.sadistic_responses is not None,
                "submission_progression": self.submission_progression is not None
            }
        }
