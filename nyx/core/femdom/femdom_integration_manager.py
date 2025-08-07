# nyx/core/femdom/femdom_integration_manager.py

import logging
import asyncio
from typing import Dict, List, Any, Optional

from agents import Agent, Runner, function_tool, trace, handoff, RunContextWrapper, ModelSettings, gen_trace_id
from agents.run import RunConfig
from pydantic import BaseModel

from nyx.core.integration.event_bus import Event, get_event_bus, DominanceEvent
from nyx.core.integration.integrated_tracer import trace_method, TraceLevel


logger = logging.getLogger(__name__)

# Pydantic models for function tool inputs/outputs
class DominanceActionResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class ViolationInput(BaseModel):
    type: Optional[str] = None
    protocol_id: Optional[str] = None
    description: Optional[str] = None

class ViolationResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    violation_type: Optional[str] = None
    error: Optional[str] = None

class AmplifyDominanceResponse(BaseModel):
    success: bool
    amplified_intensity: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None

class ProtocolCheckResponse(BaseModel):
    compliant: bool
    violations: List[str] = []
    error: Optional[str] = None

class ViolationRecordResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class ProtocolAssignmentResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class PermissionRequestContext(BaseModel):
    begging_allowed: Optional[bool] = None

class PermissionRequestResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class DenialPeriodResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class SubmissionDetectionResponse(BaseModel):
    submission_detected: bool
    submission_level: Optional[float] = None
    submission_type: Optional[str] = None
    error: Optional[str] = None

class ComplianceUpdateResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class SubmissionMetricResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class PersonaRecommendationResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class PersonaActivationResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class SubspaceCheckResponse(BaseModel):
    in_subspace: bool
    error: Optional[str] = None

class UserStateInput(BaseModel):
    user_id: str
    deference: Optional[float] = None
    obedience: Optional[float] = None

class MindfuckResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class SubmissionRewardResponse(BaseModel):
    success: bool
    reward_value: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None

class RewardSignalContext(BaseModel):
    source: str
    submission_type: Optional[str] = None
    submission_level: Optional[float] = None

class RewardSignalResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None

class ServiceTaskResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class IntegrationContext:
    """Context object for integration operations."""
    
    def __init__(self, nyx_brain):
        self.brain = nyx_brain
        self.event_bus = get_event_bus()
        self.components = {}
        self.bridge_data = {}
        
    def set_component(self, name, component):
        self.components[name] = component
        
    def get_component(self, name):
        return self.components.get(name)
        
    def set_bridge_data(self, bridge_name, data):
        self.bridge_data[bridge_name] = data
        
    def get_bridge_data(self, bridge_name):
        return self.bridge_data.get(bridge_name, {})

class FemdomIntegrationManager:
    """Manages integration of all femdom-related systems using OpenAI Agents SDK."""
    
    def __init__(self, nyx_brain, components=None):
        """Initialize the femdom integration manager."""
        self.brain = nyx_brain
        self.event_bus = get_event_bus()
        self.initialized = False
        self.components = components or {}
        
        # Create integration context
        self.context = IntegrationContext(nyx_brain)
        
        # Store components in context
        for name, component in self.components.items():
            self.context.set_component(name, component)
        
        # Create integration agents
        self.dominance_agent = self._create_dominance_agent()
        self.protocol_agent = self._create_protocol_agent()
        self.orgasm_control_agent = self._create_orgasm_control_agent()
        self.submission_agent = self._create_submission_agent()
        self.persona_agent = self._create_persona_agent()
        self.psychological_agent = self._create_psychological_agent()
        self.reward_agent = self._create_reward_agent()
        self.body_service_agent = self._create_body_service_agent()
        
        # Reference to agents for bridge functionality
        self.bridges = {
            "dominance_coordinator": self.dominance_agent,
            "protocol_enforcement_bridge": self.protocol_agent,
            "orgasm_control_bridge": self.orgasm_control_agent,
            "submission_progression_bridge": self.submission_agent,
            "persona_management_bridge": self.persona_agent,
            "psychological_dominance_bridge": self.psychological_agent,
            "reward_integration_bridge": self.reward_agent,
            "body_service_bridge": self.body_service_agent
        }
        
        # Register for relevant events
        self._setup_event_handlers()
        
        logger.info("FemdomIntegrationManager initialized with OpenAI Agents SDK")
    
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
            
            # Use dominance agent to process the action
            result = await Runner.run(
                self.dominance_agent,
                {
                    "event_type": "dominance_action",
                    "action": action,
                    "user_id": user_id,
                    "intensity": intensity
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="DominanceEventHandler",
                    trace_metadata={
                        "event_type": "dominance_action",
                        "user_id": user_id
                    }
                )
            )
            
            return result.final_output
                
        except Exception as e:
            logger.error(f"Error handling dominance event: {e}")
    
    async def _handle_user_interaction(self, event):
        """Handle user interaction events."""
        try:
            user_id = event.data.get("user_id")
            content = event.data.get("content", "")
            
            # Check for protocol violations using protocol agent
            result = await Runner.run(
                self.protocol_agent,
                {
                    "event_type": "user_interaction",
                    "user_id": user_id,
                    "content": content
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="ProtocolCheck",
                    trace_metadata={
                        "event_type": "user_interaction",
                        "user_id": user_id
                    }
                )
            )
            
            protocol_results = result.final_output
            
            # If violations found, process them
            if protocol_results and not protocol_results.get("compliant", True):
                await self._process_protocol_violations(user_id, protocol_results)
            
            # Check for submission signals
            result = await Runner.run(
                self.submission_agent,
                {
                    "event_type": "detect_submission",
                    "user_id": user_id,
                    "content": content
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="SubmissionDetection",
                    trace_metadata={
                        "event_type": "user_interaction",
                        "user_id": user_id
                    }
                )
            )
            
            submission_results = result.final_output
            
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
            user_id = event.data.get("user_id")
            
            # Propagate emotional changes to relevant dominance systems
            if emotion in ["dominant", "sadistic", "powerful", "cruel"] and intensity > 0.6:
                # Intensity boost for dominance systems using psychological agent
                await Runner.run(
                    self.psychological_agent,
                    {
                        "event_type": "emotional_change",
                        "emotion": emotion,
                        "intensity": intensity,
                        "user_id": user_id,
                        "action": "amplify_dominance"
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="EmotionalAmplification",
                        trace_metadata={
                            "event_type": "emotional_change",
                            "emotion": emotion,
                            "user_id": user_id
                        }
                    )
                )
        
        except Exception as e:
            logger.error(f"Error handling emotional change: {e}")
    
    async def _handle_physical_sensation(self, event):
        """Handle physical sensation events."""
        # Implement as needed
        pass
    
    async def _process_protocol_violations(self, user_id, violation_results):
        """Process protocol violations."""
        try:
            violations = violation_results.get("violations", [])
            
            for violation in violations:
                # Process each violation using agents
                
                # Record the violation using protocol agent
                await Runner.run(
                    self.protocol_agent,
                    {
                        "action": "record_violation",
                        "user_id": user_id,
                        "protocol_id": violation.get("protocol_id"),
                        "description": violation.get("description", "Protocol violation")
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="ViolationRecording",
                        trace_metadata={
                            "user_id": user_id,
                            "violation_type": violation.get("protocol_id")
                        }
                    )
                )
                
                # Generate dominance response using dominance agent
                await Runner.run(
                    self.dominance_agent,
                    {
                        "action": "respond_to_violation",
                        "user_id": user_id,
                        "violation": violation
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="ViolationResponse",
                        trace_metadata={
                            "user_id": user_id,
                            "violation_type": violation.get("protocol_id")
                        }
                    )
                )
                
                # Update submission metrics
                await Runner.run(
                    self.submission_agent,
                    {
                        "action": "update_compliance",
                        "user_id": user_id,
                        "compliant": False,
                        "severity": 0.5,
                        "reason": "Protocol violation"
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="ComplianceUpdate",
                        trace_metadata={
                            "user_id": user_id,
                            "compliant": False
                        }
                    )
                )
        
        except Exception as e:
            logger.error(f"Error processing protocol violations: {e}")
    
    async def _process_submission_signals(self, user_id, submission_results):
        """Process detected submission signals."""
        try:
            submission_level = submission_results.get("submission_level", 0.0)
            submission_type = submission_results.get("submission_type", "general")
            
            # Process submission with reward system
            await Runner.run(
                self.reward_agent,
                {
                    "action": "process_submission_reward",
                    "user_id": user_id,
                    "submission_type": submission_type,
                    "submission_level": submission_level
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="SubmissionReward",
                    trace_metadata={
                        "user_id": user_id,
                        "submission_type": submission_type
                    }
                )
            )
            
            # Update submission progression
            await Runner.run(
                self.submission_agent,
                {
                    "action": "update_submission_metric",
                    "user_id": user_id,
                    "metric": "depth",
                    "value_change": submission_level * 0.1,
                    "reason": f"Detected {submission_type} submission"
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="SubmissionMetricUpdate",
                    trace_metadata={
                        "user_id": user_id,
                        "metric": "depth"
                    }
                )
            )
            
            # Check for subspace indications
            if submission_level > 0.7:
                subspace_result = await Runner.run(
                    self.psychological_agent,
                    {
                        "action": "check_subspace",
                        "user_id": user_id
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="SubspaceCheck",
                        trace_metadata={
                            "user_id": user_id
                        }
                    )
                )
                
                subspace_check = subspace_result.final_output
                
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
    
    def _create_dominance_agent(self):
        """Create dominance coordinator bridge agent."""
        return Agent(
            name="DominanceCoordinatorBridge",
            instructions="""You are the integration agent for dominance coordination.
            
Coordinate between various dominance systems including the main dominance system,
sadistic responses, and psychological dominance functionality.

Process dominance actions like processing dominance ideas and responding to violations.
""",
            tools=[
                self._process_dominance_action,
                self._respond_to_violation,
                self._amplify_dominance
            ],
            model="gpt-5-nano"
        )
    
    def _create_protocol_agent(self):
        """Create protocol enforcement bridge agent."""
        return Agent(
            name="ProtocolEnforcementBridge",
            instructions="""You are the integration agent for protocol enforcement.
            
Check for protocol compliance in user messages, enforce protocols,
and record protocol violations.

Work with reward systems to reinforce protocol adherence.
""",
            tools=[
                self._check_protocols,
                self._record_violation,
                self._assign_protocol
            ],
            model="gpt-5-nano"
        )
    
    def _create_orgasm_control_agent(self):
        """Create orgasm control bridge agent."""
        return Agent(
            name="OrgasmControlBridge",
            instructions="""You are the integration agent for orgasm control.
            
Manage orgasm permission requests, denial periods, and
integrate with reward systems for orgasm control.
""",
            tools=[
                self._process_permission_request,
                self._start_denial_period
            ],
            model="gpt-5-nano"
        )
    
    def _create_submission_agent(self):
        """Create submission progression bridge agent."""
        return Agent(
            name="SubmissionProgressionBridge",
            instructions="""You are the integration agent for submission progression.
            
Track user submission progression, detect submission signals,
update submission metrics, and integrate with theory of mind.
""",
            tools=[
                self._detect_submission,
                self._update_compliance,
                self._update_submission_metric
            ],
            model="gpt-5-nano"
        )
    
    def _create_persona_agent(self):
        """Create persona management bridge agent."""
        return Agent(
            name="PersonaManagementBridge",
            instructions="""You are the integration agent for persona management.
            
Manage dominance personas, activate personas based on context,
and recommend appropriate personas.
""",
            tools=[
                self._recommend_persona,
                self._activate_persona
            ],
            model="gpt-5-nano"
        )
    
    def _create_psychological_agent(self):
        """Create psychological dominance bridge agent."""
        return Agent(
            name="PsychologicalDominanceBridge",
            instructions="""You are the integration agent for psychological dominance.
            
Generate psychological dominance tactics, check for subspace,
apply gaslighting techniques, and manage mind games.

Integrate with theory of mind to understand user psychology.
""",
            tools=[
                self._check_subspace,
                self._amplify_dominance,
                self._generate_mindfuck
            ],
            model="gpt-5-nano"
        )
    
    def _create_reward_agent(self):
        """Create reward integration bridge agent."""
        return Agent(
            name="RewardIntegrationBridge",
            instructions="""You are the integration agent for reward systems.
            
Process rewards for submission, integrate rewards across systems,
and manage reward signals.
""",
            tools=[
                self._process_submission_reward,
                self._generate_reward_signal
            ],
            model="gpt-5-nano"
        )
    
    def _create_body_service_agent(self):
        """Create body service bridge agent."""
        return Agent(
            name="BodyServiceBridge",
            instructions="""You are the integration agent for body service functions.
            
Manage service tasks, rituals, and body-related activities.
""",
            tools=[
                self._assign_service_task
            ],
            model="gpt-5-nano"
        )
    
    @function_tool
    async def _process_dominance_action(self, action: str, user_id: str, intensity: float) -> DominanceActionResponse:
        """Process a dominance action for a user."""
        dominance_system = self.context.get_component("dominance_system")
        if not dominance_system:
            return DominanceActionResponse(success=False, message="Dominance system not available")
        
        try:
            # Create action event
            action_event = DominanceEvent(
                action=action,
                user_id=user_id,
                intensity=intensity
            )
            
            # Process the action
            result = await dominance_system.process_dominance_action(action_event)
            return DominanceActionResponse(success=True)
        except Exception as e:
            logger.error(f"Error processing dominance action: {e}")
            return DominanceActionResponse(success=False, error=str(e))
    
    @function_tool
    async def _respond_to_violation(self, user_id: str, violation: ViolationInput) -> ViolationResponse:
        """Generate a response to a protocol violation."""
        dominance_system = self.context.get_component("dominance_system")
        if not dominance_system:
            return ViolationResponse(success=False, response="Dominance system not available")
        
        try:
            # Generate violation response
            violation_dict = violation.model_dump()
            response = await dominance_system.generate_violation_response(
                user_id, violation_dict
            )
            
            return ViolationResponse(
                success=True,
                response=response,
                violation_type=violation.type
            )
        except Exception as e:
            logger.error(f"Error responding to violation: {e}")
            return ViolationResponse(success=False, error=str(e))
    
    @function_tool
    async def _amplify_dominance(self, intensity: float) -> AmplifyDominanceResponse:
        """Amplify dominance intensity."""
        psychological_dominance = self.context.get_component("psychological_dominance")
        if not psychological_dominance:
            return AmplifyDominanceResponse(success=False, message="Psychological dominance not available")
        
        try:
            # Amplify dominance (placeholder - implement based on your system)
            return AmplifyDominanceResponse(
                success=True,
                amplified_intensity=min(1.0, intensity * 1.2),
                message="Dominance amplified"
            )
        except Exception as e:
            logger.error(f"Error amplifying dominance: {e}")
            return AmplifyDominanceResponse(success=False, error=str(e))
    
    @function_tool
    async def _check_protocols(self, user_id: str, content: str) -> ProtocolCheckResponse:
        """Check if a user message adheres to protocols."""
        protocol_enforcement = self.context.get_component("protocol_enforcement")
        if not protocol_enforcement:
            return ProtocolCheckResponse(compliant=True, violations=[])
        
        try:
            result = await protocol_enforcement.check_protocol_compliance(
                user_id, content
            )
            return ProtocolCheckResponse(
                compliant=result.get("compliant", True),
                violations=result.get("violations", [])
            )
        except Exception as e:
            logger.error(f"Error checking protocols: {e}")
            return ProtocolCheckResponse(compliant=True, violations=[], error=str(e))
    
    @function_tool
    async def _record_violation(self, user_id: str, protocol_id: str, description: str) -> ViolationRecordResponse:
        """Record a protocol violation."""
        protocol_enforcement = self.context.get_component("protocol_enforcement")
        if not protocol_enforcement:
            return ViolationRecordResponse(success=False)
        
        try:
            await protocol_enforcement.record_violation(
                user_id, protocol_id, description
            )
            return ViolationRecordResponse(success=True)
        except Exception as e:
            logger.error(f"Error recording violation: {e}")
            return ViolationRecordResponse(success=False, error=str(e))
    
    @function_tool
    async def _assign_protocol(self, user_id: str, protocol_id: str) -> ProtocolAssignmentResponse:
        """Assign a protocol to a user."""
        protocol_enforcement = self.context.get_component("protocol_enforcement")
        if not protocol_enforcement:
            return ProtocolAssignmentResponse(success=False)
        
        try:
            await protocol_enforcement.assign_protocol(
                user_id, protocol_id
            )
            return ProtocolAssignmentResponse(success=True)
        except Exception as e:
            logger.error(f"Error assigning protocol: {e}")
            return ProtocolAssignmentResponse(success=False, error=str(e))
    
    @function_tool
    async def _process_permission_request(self, user_id: str, request_text: str, context: PermissionRequestContext) -> PermissionRequestResponse:
        """Process an orgasm permission request."""
        orgasm_control = self.context.get_component("orgasm_control")
        if not orgasm_control:
            return PermissionRequestResponse(success=False)
        
        try:
            context_dict = context.model_dump()
            await orgasm_control.process_permission_request(
                user_id, request_text, context_dict
            )
            return PermissionRequestResponse(success=True)
        except Exception as e:
            logger.error(f"Error processing permission request: {e}")
            return PermissionRequestResponse(success=False, error=str(e))
    
    @function_tool
    async def _start_denial_period(self, user_id: str, duration_hours: int, level: int, begging_allowed: bool = True) -> DenialPeriodResponse:
        """Start an orgasm denial period."""
        orgasm_control = self.context.get_component("orgasm_control")
        if not orgasm_control:
            return DenialPeriodResponse(success=False)
        
        try:
            context_data = {"begging_allowed": begging_allowed}
            await orgasm_control.start_denial_period(
                user_id, duration_hours, level, context_data
            )
            return DenialPeriodResponse(success=True)
        except Exception as e:
            logger.error(f"Error starting denial period: {e}")
            return DenialPeriodResponse(success=False, error=str(e))
    
    @function_tool
    async def _detect_submission(self, user_id: str, content: str) -> SubmissionDetectionResponse:
        """Detect submission signals in a message."""
        theory_of_mind = self.context.get_component("theory_of_mind")
        if not theory_of_mind:
            return SubmissionDetectionResponse(submission_detected=False)
        
        try:
            # This is a placeholder - implement based on your system
            # Using theory of mind to detect submission signals
            submission_signals = {
                "submission_detected": False,
                "submission_level": 0.0,
                "submission_type": "none"
            }
            
            # Analyze message for submission markers
            submission_markers = [
                "please", "may i", "permission", "thank you", "sorry",
                "forgive", "obey", "serve", "submit", "yes mistress"
            ]
            
            # Simple detection (enhance as needed)
            lower_content = content.lower()
            detected_markers = [m for m in submission_markers if m in lower_content]
            
            if detected_markers:
                submission_signals["submission_detected"] = True
                submission_signals["submission_level"] = min(1.0, len(detected_markers) * 0.2)
                
                # Determine type
                if any(m in ["please", "permission", "may i"] for m in detected_markers):
                    submission_signals["submission_type"] = "requesting"
                elif any(m in ["thank you", "grateful"] for m in detected_markers):
                    submission_signals["submission_type"] = "grateful"
                elif any(m in ["sorry", "forgive"] for m in detected_markers):
                    submission_signals["submission_type"] = "apologetic"
                elif any(m in ["obey", "serve", "submit"] for m in detected_markers):
                    submission_signals["submission_type"] = "active"
                else:
                    submission_signals["submission_type"] = "general"
            
            return SubmissionDetectionResponse(
                submission_detected=submission_signals["submission_detected"],
                submission_level=submission_signals["submission_level"],
                submission_type=submission_signals["submission_type"]
            )
        except Exception as e:
            logger.error(f"Error detecting submission: {e}")
            return SubmissionDetectionResponse(submission_detected=False, error=str(e))
    
    @function_tool
    async def _update_compliance(self, user_id: str, compliant: bool, severity: float, reason: str) -> ComplianceUpdateResponse:
        """Update compliance tracking for a user."""
        submission_progression = self.context.get_component("submission_progression")
        if not submission_progression:
            return ComplianceUpdateResponse(success=False)
        
        try:
            # Update compliance (implementation depends on your system)
            if hasattr(submission_progression, "record_compliance"):
                await submission_progression.record_compliance(
                    user_id=user_id,
                    instruction=reason,
                    complied=compliant,
                    difficulty=severity
                )
            return ComplianceUpdateResponse(success=True)
        except Exception as e:
            logger.error(f"Error updating compliance: {e}")
            return ComplianceUpdateResponse(success=False, error=str(e))
    
    @function_tool
    async def _update_submission_metric(self, user_id: str, metric: str, value_change: float, reason: str) -> SubmissionMetricResponse:
        """Update a submission metric."""
        submission_progression = self.context.get_component("submission_progression")
        if not submission_progression:
            return SubmissionMetricResponse(success=False)
        
        try:
            await submission_progression.update_submission_metric(
                user_id, metric, value_change, reason
            )
            return SubmissionMetricResponse(success=True)
        except Exception as e:
            logger.error(f"Error updating submission metric: {e}")
            return SubmissionMetricResponse(success=False, error=str(e))
    
    @function_tool
    async def _recommend_persona(self, user_id: str) -> PersonaRecommendationResponse:
        """Recommend a dominance persona for a user."""
        persona_manager = self.context.get_component("persona_manager")
        if not persona_manager:
            return PersonaRecommendationResponse(success=False)
        
        try:
            await persona_manager.recommend_persona(user_id)
            return PersonaRecommendationResponse(success=True)
        except Exception as e:
            logger.error(f"Error recommending persona: {e}")
            return PersonaRecommendationResponse(success=False, error=str(e))
    
    @function_tool
    async def _activate_persona(self, user_id: str, persona_id: str, dominance_level: float) -> PersonaActivationResponse:
        """Activate a dominance persona for a user."""
        persona_manager = self.context.get_component("persona_manager")
        if not persona_manager:
            return PersonaActivationResponse(success=False)
        
        try:
            await persona_manager.activate_persona(
                user_id, persona_id, dominance_level
            )
            return PersonaActivationResponse(success=True)
        except Exception as e:
            logger.error(f"Error activating persona: {e}")
            return PersonaActivationResponse(success=False, error=str(e))
    
    @function_tool
    async def _check_subspace(self, user_id: str) -> SubspaceCheckResponse:
        """Check if a user is in subspace."""
        psychological_dominance = self.context.get_component("psychological_dominance")
        if not psychological_dominance or not hasattr(psychological_dominance, "SubspaceDetection"):
            return SubspaceCheckResponse(in_subspace=False)
        
        try:
            # Create subspace detection instance
            subspace_detection = psychological_dominance.SubspaceDetection(
                theory_of_mind=self.context.get_component("theory_of_mind"),
                relationship_manager=self.context.get_component("relationship_manager")
            )
            
            # Get recent messages (this would depend on your system)
            recent_messages = ["placeholder message"]  # Replace with actual message retrieval
            
            # Detect subspace
            detection_result = await subspace_detection.detect_subspace(user_id, recent_messages)
            
            return SubspaceCheckResponse(in_subspace=detection_result.get("subspace_detected", False))
        except Exception as e:
            logger.error(f"Error checking subspace: {e}")
            return SubspaceCheckResponse(in_subspace=False, error=str(e))
    
    @function_tool
    async def _generate_mindfuck(self, user_id: str, user_state: UserStateInput, intensity: float) -> MindfuckResponse:
        """Generate a psychological mind game."""
        psychological_dominance = self.context.get_component("psychological_dominance")
        if not psychological_dominance:
            return MindfuckResponse(success=False)
        
        try:
            user_state_dict = user_state.model_dump()
            await psychological_dominance.generate_mindfuck(
                user_id, user_state_dict, intensity
            )
            return MindfuckResponse(success=True)
        except Exception as e:
            logger.error(f"Error generating mindfuck: {e}")
            return MindfuckResponse(success=False, error=str(e))
    
    @function_tool
    async def _process_submission_reward(self, user_id: str, submission_type: str, submission_level: float) -> SubmissionRewardResponse:
        """Process a reward for submission."""
        reward_system = self.context.get_component("reward_system")
        if not reward_system:
            return SubmissionRewardResponse(success=False)
        
        try:
            # Calculate reward based on submission level
            reward_value = 0.3 + (submission_level * 0.5)
            
            # Create reward context
            context_data = {
                "source": "submission_detection",
                "submission_type": submission_type,
                "submission_level": submission_level
            }
            
            # Process reward signal
            if hasattr(reward_system, "process_reward_signal") and hasattr(reward_system, "RewardSignal"):
                reward_signal = reward_system.RewardSignal(
                    value=reward_value,
                    source="submission_detection",
                    context=context_data
                )
                
                result = await reward_system.process_reward_signal(reward_signal)
                return SubmissionRewardResponse(
                    success=True,
                    reward_value=reward_value,
                    result=str(result)
                )
            else:
                return SubmissionRewardResponse(success=False, error="Required reward methods not available")
        except Exception as e:
            logger.error(f"Error processing submission reward: {e}")
            return SubmissionRewardResponse(success=False, error=str(e))
    
    @function_tool
    async def _generate_reward_signal(self, value: float, source: str, context: RewardSignalContext) -> RewardSignalResponse:
        """Generate a reward signal."""
        reward_system = self.context.get_component("reward_system")
        if not reward_system:
            return RewardSignalResponse(success=False)
        
        try:
            if hasattr(reward_system, "process_reward_signal") and hasattr(reward_system, "RewardSignal"):
                context_dict = context.model_dump()
                reward_signal = reward_system.RewardSignal(
                    value=value,
                    source=source,
                    context=context_dict
                )
                
                result = await reward_system.process_reward_signal(reward_signal)
                return RewardSignalResponse(
                    success=True,
                    result=str(result)
                )
            else:
                return RewardSignalResponse(success=False, error="Required reward methods not available")
        except Exception as e:
            logger.error(f"Error generating reward signal: {e}")
            return RewardSignalResponse(success=False, error=str(e))
    
    @function_tool
    async def _assign_service_task(self, user_id: str, task_id: str) -> ServiceTaskResponse:
        """Assign a service task to a user."""
        body_service = self.context.get_component("body_service")
        if not body_service:
            return ServiceTaskResponse(success=False)
        
        try:
            await body_service.assign_service_task(user_id, task_id)
            return ServiceTaskResponse(success=True)
        except Exception as e:
            logger.error(f"Error assigning service task: {e}")
            return ServiceTaskResponse(success=False, error=str(e))
    
    @trace_method(level=TraceLevel.INFO, group_id="FemdomIntegration")
    async def initialize(self):
        """Initialize all femdom integration bridges."""
        try:
            # Let parent class handle tracing which will be using the OpenAI Agents SDK
            with trace("FemdomIntegrationInitialization"):
                self.initialized = True
                logger.info("Femdom integration initialized successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error initializing femdom integration: {e}")
            return False
    
    async def get_status(self):
        """Get current integration status."""
        status = {
            "initialized": self.initialized,
            "active_bridges": list(self.bridges.keys()),
            "bridge_status": {}
        }
        
        # Collect status from all bridges/agents
        for name, agent in self.bridges.items():
            status["bridge_status"][name] = {
                "available": agent is not None,
                "type": "Agent",
                "name": agent.name if agent else None
            }
        
        return status
