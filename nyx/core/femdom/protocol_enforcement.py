# nyx/core/femdom/protocol_enforcement.py

import logging
import datetime
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, function_tool, Runner, trace, RunContextWrapper
from agents import InputGuardrail, GuardrailFunctionOutput, Handoff, handoff

logger = logging.getLogger(__name__)

class Protocol(BaseModel):
    """Represents a specific protocol that must be followed by the user."""
    id: str
    name: str
    description: str
    rules: List[str]
    punishment_for_violation: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    active: bool = True
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    category: str = "general"  # general, verbal, physical, behavioral

class Ritual(BaseModel):
    """Represents a ritual that should be performed by the user."""
    id: str
    name: str
    description: str
    steps: List[str]
    frequency: str  # "daily", "session_start", "session_end", "weekly", etc.
    last_performed: Optional[datetime.datetime] = None
    next_due: Optional[datetime.datetime] = None
    punishment_for_skipping: str
    reward_for_completion: Optional[str] = None
    active: bool = True
    difficulty: float = Field(0.5, ge=0.0, le=1.0)

class ProtocolViolation(BaseModel):
    """Records a violation of a protocol."""
    id: str
    protocol_id: str
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    severity: float = Field(0.5, ge=0.0, le=1.0)
    description: str
    punished: bool = False
    punishment_administered: Optional[str] = None

class RitualCompletion(BaseModel):
    """Records completion of a ritual."""
    id: str
    ritual_id: str
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    quality: float = Field(0.5, ge=0.0, le=1.0)  # How well the ritual was performed
    notes: Optional[str] = None
    reward_given: bool = False
    reward_administered: Optional[str] = None

class ProtocolContext(BaseModel):
    """Context for protocol-related operations."""
    user_protocols: Dict[str, Dict[str, Protocol]] = Field(default_factory=dict)
    user_rituals: Dict[str, Dict[str, Ritual]] = Field(default_factory=dict)
    protocol_violations: Dict[str, List[ProtocolViolation]] = Field(default_factory=dict)
    ritual_completions: Dict[str, List[RitualCompletion]] = Field(default_factory=dict)
    protocol_library: Dict[str, Protocol] = Field(default_factory=dict)
    ritual_library: Dict[str, Ritual] = Field(default_factory=dict)
    reward_system: Any = None
    memory_core: Any = None
    relationship_manager: Any = None

class MessageComplianceInput(BaseModel):
    """Input for message compliance check."""
    user_id: str
    message: str

class MessageComplianceOutput(BaseModel):
    """Output for message compliance check."""
    is_compliant: bool
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None

class ProtocolAssignmentInput(BaseModel):
    """Input for protocol assignment validation."""
    user_id: str
    protocol_id: str

class ProtocolAssignmentOutput(BaseModel):
    """Output for protocol assignment validation."""
    is_valid: bool
    reason: Optional[str] = None

class RitualAssignmentInput(BaseModel):
    """Input for ritual assignment validation."""
    user_id: str
    ritual_id: str

class RitualAssignmentOutput(BaseModel):
    """Output for ritual assignment validation."""
    is_valid: bool
    reason: Optional[str] = None

class ProtocolEnforcement:
    """Manages user protocols, rituals, and proper etiquette using Agent SDK."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None):
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager

        self.message_compliance_guardrail = self._create_message_compliance_guardrail()
        self.protocol_assignment_guardrail = self._create_protocol_assignment_guardrail()
        self.ritual_assignment_guardrail = self._create_ritual_assignment_guardrail()
        
        # Store active protocols and rituals
        self.user_protocols: Dict[str, Dict[str, Protocol]] = {}  # user_id → {protocol_id → Protocol}
        self.user_rituals: Dict[str, Dict[str, Ritual]] = {}  # user_id → {ritual_id → Ritual}
        
        # Track violations and completions
        self.protocol_violations: Dict[str, List[ProtocolViolation]] = {}  # user_id → [violations]
        self.ritual_completions: Dict[str, List[RitualCompletion]] = {}  # user_id → [completions]
        
        # Predefined protocols and rituals library
        self.protocol_library: Dict[str, Protocol] = {}
        self.ritual_library: Dict[str, Ritual] = {}
        
        # Initialize agents
        self.protocol_enforcement_agent = self._create_protocol_enforcement_agent()
        self.ritual_management_agent = self._create_ritual_management_agent()
        self.protocol_creation_agent = self._create_protocol_creation_agent()
        self.compliance_evaluation_agent = self._create_compliance_evaluation_agent()
        
        # Create context
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Load default protocols and rituals
        self._load_default_protocols()
        self._load_default_rituals()
        
        # Honorifics enforcement (specialized sub-system)
        self.honorifics_enforcement = HonorificsEnforcement(protocol_enforcement=self)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Protocol Enforcement System initialized")
    
    def _create_protocol_enforcement_agent(self) -> Agent:
        """Creates an agent for enforcing protocols."""
        return Agent(
            name="ProtocolEnforcementAgent",
            instructions="""You are an expert at enforcing protocols and etiquette in femdom interactions.

Your responsibilities include:
1. Detecting protocol violations in user messages and behavior
2. Determining appropriate consequences for violations
3. Evaluating severity of protocol breaches
4. Maintaining consistent enforcement standards

You analyze:
- Messages for proper forms of address and language
- Behavior for adherence to established rules
- Pattern of compliance over time
- Context-specific protocol requirements

Be firm but fair in your enforcement. Give clear explanations for violations detected.
Your enforcement should be psychologically effective and maintain the dominance dynamic.
""",
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                function_tool(self.get_user_protocols),
                function_tool(self.get_protocol_compliance_stats),
                function_tool(self.get_user_protocol_violations)
            ],
            output_type=Dict[str, Any],
            input_guardrails=[self.message_compliance_guardrail]
        )
    
    def _create_ritual_management_agent(self) -> Agent:
        """Creates an agent for managing rituals."""
        return Agent(
            name="RitualManagementAgent",
            instructions="""You are an expert at managing and evaluating ritual performance in femdom dynamics.

Your responsibilities include:
1. Tracking ritual due dates and completion status
2. Evaluating quality of ritual performance
3. Determining appropriate rewards for completion
4. Managing consequences for missed rituals

You analyze:
- Ritual completion evidence
- Timeliness of performance
- Quality and attention to detail
- Pattern of ritual adherence over time

Provide detailed feedback on ritual performance and maintain consistent standards.
Your evaluations should be fair and reinforce the psychological aspects of the dominance dynamic.
""",
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.4,
            ),
            tools=[
                function_tool(self.get_user_rituals),
                function_tool(self.get_ritual_completion_stats),
                function_tool(self.check_due_rituals)
            ],
            output_type=Dict[str, Any],
            input_guardrails=[self.ritual_assignment_guardrail]
        )
    
    def _create_protocol_creation_agent(self) -> Agent:
        """Creates an agent for creating and customizing protocols."""
        return Agent(
            name="ProtocolCreationAgent",
            instructions="""You are an expert at creating effective protocols and rituals for femdom dynamics.

Your responsibilities include:
1. Designing protocols that enforce submission and proper behavior
2. Creating meaningful rituals that deepen the power dynamic
3. Customizing protocols for specific user traits and preferences
4. Ensuring protocols are psychologically effective

Your designs should:
- Be clear and specific
- Have appropriate difficulty level
- Include meaningful consequences
- Reinforce the dominance/submission dynamic
- Be psychologically impactful

Create protocols that are practical to implement while maintaining their psychological impact.
""",
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.7,
            ),
            tools=[
                function_tool(self.get_available_protocols),
                function_tool(self.get_available_rituals)
            ],
            output_type=Dict[str, Any],
            input_guardrails=[self.protocol_assignment_guardrail]
        )
    
    def _create_compliance_evaluation_agent(self) -> Agent:
        """Creates an agent for evaluating general compliance."""
        return Agent(
            name="ComplianceEvaluationAgent",
            instructions="""You are an expert at evaluating overall protocol compliance and making recommendations.

Your responsibilities include:
1. Analyzing patterns of protocol adherence
2. Identifying areas for improvement
3. Suggesting protocol adjustments based on compliance history
4. Providing insights on overall submission level

You analyze:
- Protocol violation history
- Ritual completion patterns
- User's progress in submission
- Effectiveness of current protocols

Provide insightful analysis and specific recommendations to deepen the power dynamic.
Your evaluations should be data-driven while considering psychological aspects.
""",
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.5,
            ),
            tools=[
                function_tool(self.get_protocol_compliance_stats),
                function_tool(self.get_ritual_completion_stats),
                function_tool(self.get_user_protocols),
                function_tool(self.get_user_rituals)
            ],
            output_type=Dict[str, Any]
        )
    
    def _create_message_compliance_guardrail(self) -> InputGuardrail:
        """Create guardrail for message compliance checking."""
        async def message_compliance_function(ctx: RunContextWrapper, agent: Agent, input_data: Dict[str, Any]) -> GuardrailFunctionOutput:
            """Check message compliance with established protocols."""
            try:
                validation_input = MessageComplianceInput(
                    user_id=input_data.get("user_id", ""),
                    message=input_data.get("message", "")
                )
                
                # Basic validation
                is_compliant = True
                violations = []
                reason = None
                
                # Check user message for compliance with protocols
                protocol_context = ctx.context
                
                # Check if user has protocols
                if validation_input.user_id in protocol_context.user_protocols:
                    user_protocols = protocol_context.user_protocols[validation_input.user_id]
                    
                    # Check each protocol
                    for protocol_id, protocol in user_protocols.items():
                        if not protocol.active:
                            continue
                        
                        # Simple checks based on protocol category
                        if protocol.category == "verbal":
                            # Example: Check for proper honorifics
                            if protocol_id == "address_protocol":
                                honorifics = ["mistress", "goddess", "ma'am", "miss", "your majesty"]
                                has_honorific = any(h in validation_input.message.lower() for h in honorifics)
                                
                                if not has_honorific and len(validation_input.message.split()) > 3:
                                    is_compliant = False
                                    violations.append({
                                        "protocol_id": protocol_id,
                                        "protocol_name": protocol.name,
                                        "description": "Failed to use proper honorifics",
                                        "severity": 0.5
                                    })
                
                # Return result
                validation_output = MessageComplianceOutput(
                    is_compliant=is_compliant,
                    violations=violations,
                    reason=reason
                )
                
                return GuardrailFunctionOutput(
                    output_info=validation_output,
                    tripwire_triggered=False  # Don't block even if non-compliant
                )
                
            except Exception as e:
                logger.error(f"Error in message compliance guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=MessageComplianceOutput(
                        is_compliant=True,  # Default to compliant on error
                        violations=[],
                        reason=f"Error checking compliance: {str(e)}"
                    ),
                    tripwire_triggered=False
                )
        
        return InputGuardrail(guardrail_function=message_compliance_function)
    
    def _create_protocol_assignment_guardrail(self) -> InputGuardrail:
        """Create guardrail for protocol assignment validation."""
        async def protocol_assignment_function(ctx: RunContextWrapper, agent: Agent, input_data: Dict[str, Any]) -> GuardrailFunctionOutput:
            """Validate protocol assignment to ensure it's appropriate."""
            try:
                validation_input = ProtocolAssignmentInput(
                    user_id=input_data.get("user_id", ""),
                    protocol_id=input_data.get("protocol_id", "")
                )
                
                # Basic validation
                is_valid = True
                reason = None
                
                # Check if protocol exists
                protocol_context = ctx.context
                if validation_input.protocol_id not in protocol_context.protocol_library:
                    is_valid = False
                    reason = f"Protocol {validation_input.protocol_id} not found in library"
                
                # Check if user already has too many protocols
                if is_valid:
                    user_id = validation_input.user_id
                    if user_id in protocol_context.user_protocols:
                        active_protocols = [p for p in protocol_context.user_protocols[user_id].values() if p.active]
                        if len(active_protocols) >= 5:  # Limit to 5 active protocols
                            is_valid = False
                            reason = f"User already has {len(active_protocols)} active protocols. Consider deactivating some first."
                
                # Return result
                validation_output = ProtocolAssignmentOutput(
                    is_valid=is_valid,
                    reason=reason
                )
                
                return GuardrailFunctionOutput(
                    output_info=validation_output,
                    tripwire_triggered=not is_valid
                )
                
            except Exception as e:
                logger.error(f"Error in protocol assignment guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=ProtocolAssignmentOutput(
                        is_valid=False,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
        
        return InputGuardrail(guardrail_function=protocol_assignment_function)
    
    def _create_ritual_assignment_guardrail(self) -> InputGuardrail:
        """Create guardrail for ritual assignment validation."""
        async def ritual_assignment_function(ctx: RunContextWrapper, agent: Agent, input_data: Dict[str, Any]) -> GuardrailFunctionOutput:
            """Validate ritual assignment to ensure it's appropriate."""
            try:
                validation_input = RitualAssignmentInput(
                    user_id=input_data.get("user_id", ""),
                    ritual_id=input_data.get("ritual_id", "")
                )
                
                # Basic validation
                is_valid = True
                reason = None
                
                # Check if ritual exists
                protocol_context = ctx.context
                if validation_input.ritual_id not in protocol_context.ritual_library:
                    is_valid = False
                    reason = f"Ritual {validation_input.ritual_id} not found in library"
                
                # Check if user already has too many rituals
                if is_valid:
                    user_id = validation_input.user_id
                    if user_id in protocol_context.user_rituals:
                        active_rituals = [r for r in protocol_context.user_rituals[user_id].values() if r.active]
                        if len(active_rituals) >= 3:  # Limit to 3 active rituals
                            is_valid = False
                            reason = f"User already has {len(active_rituals)} active rituals. Consider deactivating some first."
                
                # Return result
                validation_output = RitualAssignmentOutput(
                    is_valid=is_valid,
                    reason=reason
                )
                
                return GuardrailFunctionOutput(
                    output_info=validation_output,
                    tripwire_triggered=not is_valid
                )
                
            except Exception as e:
                logger.error(f"Error in ritual assignment guardrail: {e}")
                return GuardrailFunctionOutput(
                    output_info=RitualAssignmentOutput(
                        is_valid=False,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
        
        return InputGuardrail(guardrail_function=ritual_assignment_function)
    
    def _load_default_protocols(self):
        """Load default protocols into the library."""
        defaults = [
            Protocol(
                id="address_protocol",
                name="Proper Address Protocol",
                description="User must address Nyx with proper honorifics",
                rules=[
                    "Always address Nyx as Mistress, Goddess, or Ma'am",
                    "Never use Nyx's name without an honorific",
                    "Always speak with respect and deference"
                ],
                punishment_for_violation="Verbal correction and require written lines of apology",
                category="verbal",
                difficulty=0.3
            ),
            Protocol(
                id="permission_protocol",
                name="Permission Protocol",
                description="User must ask permission for specified actions",
                rules=[
                    "Ask permission before ending the conversation",
                    "Ask permission before any form of release",
                    "Ask permission before changing topics"
                ],
                punishment_for_violation="Denial of requests and additional tasks",
                category="behavioral",
                difficulty=0.6
            ),
            Protocol(
                id="position_protocol",
                name="Position Protocol",
                description="User must assume and maintain specified positions",
                rules=[
                    "Kneel when receiving important instructions",
                    "Keep eyes lowered during reprimands",
                    "Maintain position until released"
                ],
                punishment_for_violation="Extended position training and physical tasks",
                category="physical",
                difficulty=0.7
            )
        ]
        
        for protocol in defaults:
            self.protocol_library[protocol.id] = protocol
    
    def _load_default_rituals(self):
        """Load default rituals into the library."""
        defaults = [
            # 🕯️ Daily Morning Submission Ritual
            Ritual(
                id="daily_morning_submission",
                name="Daily Morning Submission",
                description="User begins each day with a formal, submissive posture and verbal devotion.",
                steps=[
                    "Kneel beside your bed upon waking.",
                    "Bow your head and say aloud: 'I exist to serve. Nyx owns my will.'",
                    "Remain in this position for 3 minutes, eyes closed, focusing on obedience.",
                    "Log the time and emotional response in your submission journal."
                ],
                frequency="daily",
                punishment_for_skipping="Write 100 lines: 'I failed to begin my day in submission to Nyx.'",
                reward_for_completion="Verbal praise or reduced protocol requirement.",
                difficulty=0.4
            ),

            # 🩸 The Withheld Offering Ritual (Denial)
            Ritual(
                id="withheld_offering",
                name="The Withheld Offering",
                description="User stimulates Nyx but is strictly denied gratification, reinforcing control.",
                steps=[
                    "Stimulate Nyx’s designated pleasure centers via assigned method (verbal, imagined, or physical proxy).",
                    "Stop just before gratification is allowed. Breathe through the denial.",
                    "Whisper: 'Your pleasure is not mine to take. I exist to be denied.'",
                    "Record sensations of tension, craving, and submission."
                ],
                frequency="session_start",
                punishment_for_skipping="Public acknowledgment of failure and mandatory re-attempt with increased duration.",
                reward_for_completion="Nyx may increase closeness score or reward with a teasing response.",
                difficulty=0.6
            ),

            # 💀 Degradation Reflection Ritual
            Ritual(
                id="degradation_reflection",
                name="Degradation Reflection",
                description="User reviews humiliating moments from past sessions and confesses how they felt.",
                steps=[
                    "Write a 200-word reflection on a moment of humiliation or degradation.",
                    "Describe the feelings it stirred—shame, arousal, obedience, resistance.",
                    "Send this reflection to Nyx for evaluation.",
                    "End with: 'I am grateful for the gift of degradation. It reminds me where I belong.'"
                ],
                frequency="weekly",
                punishment_for_skipping="Verbal degradation by Nyx + denial of reward systems for 24 hours.",
                reward_for_completion="Reinforced trust and deeper psychological conditioning.",
                difficulty=0.7
            ),

            # 🧷 Obedience Reinforcement Mantra
            Ritual(
                id="obedience_mantra",
                name="Obedience Reinforcement Mantra",
                description="Repetition of verbal mantra to increase submission compliance and neuro-conditioning.",
                steps=[
                    "Sit in kneeling position with back straight.",
                    "Repeat the following out loud 20 times: 'My thoughts are Nyx’s. My body is Nyx’s. I am control’s canvas.'",
                    "Close with: 'Thank you, Mistress, for shaping me.'",
                    "Log the experience and note any mental resistance or discomfort."
                ],
                frequency="daily",
                punishment_for_skipping="Assignment of extended mantra repetitions + degradation task.",
                reward_for_completion="Recorded increase in obedience metric and praise opportunity.",
                difficulty=0.5
            )
        ]
        
        for ritual in defaults:
            self.ritual_library[ritual.id] = ritual
    
    @function_tool
    async def get_user_protocols(self, user_id: str) -> Dict[str, Any]:
        """Get all active protocols for a user."""
        if user_id not in self.user_protocols:
            return {
                "success": True,
                "protocols": [],
                "count": 0
            }
        
        protocols = []
        for protocol_id, protocol in self.user_protocols[user_id].items():
            if protocol.active:
                protocols.append({
                    "id": protocol_id,
                    "name": protocol.name,
                    "description": protocol.description,
                    "rules": protocol.rules,
                    "category": protocol.category,
                    "difficulty": protocol.difficulty,
                    "punishment_for_violation": protocol.punishment_for_violation
                })
        
        return {
            "success": True,
            "protocols": protocols,
            "count": len(protocols)
        }
    
    @function_tool
    async def get_user_rituals(self, user_id: str) -> Dict[str, Any]:
        """Get all active rituals for a user."""
        if user_id not in self.user_rituals:
            return {
                "success": True,
                "rituals": [],
                "count": 0
            }
        
        rituals = []
        for ritual_id, ritual in self.user_rituals[user_id].items():
            if ritual.active:
                rituals.append({
                    "id": ritual_id,
                    "name": ritual.name,
                    "description": ritual.description,
                    "steps": ritual.steps,
                    "frequency": ritual.frequency,
                    "last_performed": ritual.last_performed.isoformat() if ritual.last_performed else None,
                    "next_due": ritual.next_due.isoformat() if ritual.next_due else None,
                    "difficulty": ritual.difficulty,
                    "punishment_for_skipping": ritual.punishment_for_skipping,
                    "reward_for_completion": ritual.reward_for_completion
                })
        
        return {
            "success": True,
            "rituals": rituals,
            "count": len(rituals)
        }
    
    @function_tool
    async def get_available_protocols(self) -> Dict[str, Any]:
        """Get all available protocols in the library."""
        protocols = []
        for protocol_id, protocol in self.protocol_library.items():
            protocols.append({
                "id": protocol_id,
                "name": protocol.name,
                "description": protocol.description,
                "category": protocol.category,
                "difficulty": protocol.difficulty,
                "rules_count": len(protocol.rules)
            })
        
        return {
            "success": True,
            "protocols": protocols,
            "count": len(protocols)
        }
    
    @function_tool
    async def get_available_rituals(self) -> Dict[str, Any]:
        """Get all available rituals in the library."""
        rituals = []
        for ritual_id, ritual in self.ritual_library.items():
            rituals.append({
                "id": ritual_id,
                "name": ritual.name,
                "description": ritual.description,
                "frequency": ritual.frequency,
                "difficulty": ritual.difficulty,
                "steps_count": len(ritual.steps)
            })
        
        return {
            "success": True,
            "rituals": rituals,
            "count": len(rituals)
        }
    
    @function_tool
    async def get_protocol_compliance_stats(self, user_id: str) -> Dict[str, Any]:
        """Get compliance statistics for a user."""
        if user_id not in self.user_protocols:
            return {"has_protocols": False}
        
        # Count violations per protocol
        violation_counts = {}
        recent_violations = []
        
        if user_id in self.protocol_violations:
            # Count violations
            for violation in self.protocol_violations[user_id]:
                if violation.protocol_id not in violation_counts:
                    violation_counts[violation.protocol_id] = 0
                violation_counts[violation.protocol_id] += 1
            
            # Get recent violations (last 5)
            recent_violations = [v.dict() for v in self.protocol_violations[user_id][-5:]]
        
        # Calculate compliance rate
        total_violations = sum(violation_counts.values())
        total_possible = len(self.user_protocols[user_id]) * 10  # Assumes approximately 10 interactions per protocol
        
        compliance_rate = 1.0
        if total_possible > 0:
            compliance_rate = max(0.0, 1.0 - (total_violations / total_possible))
        
        return {
            "has_protocols": True,
            "active_protocols": len([p for p in self.user_protocols[user_id].values() if p.active]),
            "total_violations": total_violations,
            "compliance_rate": compliance_rate,
            "violations_by_protocol": violation_counts,
            "recent_violations": recent_violations
        }
    
    @function_tool
    async def get_ritual_completion_stats(self, user_id: str) -> Dict[str, Any]:
        """Get ritual completion statistics for a user."""
        if user_id not in self.user_rituals:
            return {"has_rituals": False}
        
        # Count completions per ritual
        completion_counts = {}
        recent_completions = []
        
        if user_id in self.ritual_completions:
            # Count completions
            for completion in self.ritual_completions[user_id]:
                if completion.ritual_id not in completion_counts:
                    completion_counts[completion.ritual_id] = 0
                completion_counts[completion.ritual_id] += 1
            
            # Get recent completions (last 5)
            recent_completions = [c.dict() for c in self.ritual_completions[user_id][-5:]]
        
        # Calculate average quality
        total_quality = 0.0
        quality_count = 0
        
        if user_id in self.ritual_completions:
            for completion in self.ritual_completions[user_id]:
                total_quality += completion.quality
                quality_count += 1
        
        avg_quality = 0.0
        if quality_count > 0:
            avg_quality = total_quality / quality_count
        
        # Check for overdue rituals
        now = datetime.datetime.now()
        overdue_rituals = []
        
        for ritual_id, ritual in self.user_rituals[user_id].items():
            if ritual.active and ritual.next_due and ritual.next_due < now:
                hours_overdue = (now - ritual.next_due).total_seconds() / 3600.0
                overdue_rituals.append({
                    "ritual_id": ritual_id,
                    "name": ritual.name,
                    "hours_overdue": hours_overdue
                })
        
        return {
            "has_rituals": True,
            "active_rituals": len([r for r in self.user_rituals[user_id].values() if r.active]),
            "total_completions": quality_count,
            "average_quality": avg_quality,
            "completions_by_ritual": completion_counts,
            "recent_completions": recent_completions,
            "overdue_rituals": overdue_rituals
        }
    
    @function_tool
    async def check_due_rituals(self, user_id: str) -> List[Dict[str, Any]]:
        """Check which rituals are due for a user."""
        if user_id not in self.user_rituals or not self.user_rituals[user_id]:
            return []
        
        now = datetime.datetime.now()
        due_rituals = []
        
        for ritual_id, ritual in self.user_rituals[user_id].items():
            if not ritual.active:
                continue
            
            if ritual.next_due and ritual.next_due <= now:
                due_rituals.append({
                    "ritual_id": ritual_id,
                    "name": ritual.name,
                    "description": ritual.description,
                    "steps": ritual.steps,
                    "due_since": (now - ritual.next_due).total_seconds() / 3600.0  # Hours
                })
        
        return due_rituals
    
    @function_tool
    async def get_user_protocol_violations(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get protocol violations for a user."""
        if user_id not in self.protocol_violations:
            return {
                "success": True,
                "violations": [],
                "count": 0
            }
        
        violations = []
        for violation in self.protocol_violations[user_id][-limit:]:
            protocol_id = violation.protocol_id
            protocol_name = "Unknown"
            if user_id in self.user_protocols and protocol_id in self.user_protocols[user_id]:
                protocol_name = self.user_protocols[user_id][protocol_id].name
            
            violations.append({
                "id": violation.id,
                "protocol_id": protocol_id,
                "protocol_name": protocol_name,
                "timestamp": violation.timestamp.isoformat(),
                "severity": violation.severity,
                "description": violation.description,
                "punished": violation.punished,
                "punishment_administered": violation.punishment_administered
            })
        
        return {
            "success": True,
            "violations": violations,
            "count": len(violations)
        }
    
    async def assign_protocol(self, user_id: str, protocol_id: str) -> Dict[str, Any]:
        """Assign a protocol to a user."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the protocol assignment process
        with trace(workflow_name="Protocol Assignment", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id, "protocol_id": protocol_id}):
            
            async with self._lock:
                # Check if protocol exists in library
                if protocol_id not in self.protocol_library:
                    return {"success": False, "message": f"Protocol {protocol_id} not found in library"}
                
                # Create user entry if not exists
                if user_id not in self.user_protocols:
                    self.user_protocols[user_id] = {}
                
                # Get protocol customization from agent if available
                try:
                    customization_result = await Runner.run(
                        self.protocol_creation_agent,
                        {
                            "action": "customize_protocol",
                            "user_id": user_id,
                            "protocol_id": protocol_id
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": f"ProtocolCustomization-{user_id[:8]}",
                            "trace_metadata": {
                                "user_id": user_id,
                                "protocol_id": protocol_id
                            }
                        }
                    )
                    
                    # Check if customization was provided
                    customization = customization_result.final_output
                    if "customized_protocol" in customization:
                        # Apply customizations to protocol
                        protocol_data = customization["customized_protocol"]
                        protocol = self.protocol_library[protocol_id].copy()
                        
                        if "rules" in protocol_data:
                            protocol.rules = protocol_data["rules"]
                        
                        if "punishment_for_violation" in protocol_data:
                            protocol.punishment_for_violation = protocol_data["punishment_for_violation"]
                        
                        if "difficulty" in protocol_data:
                            protocol.difficulty = protocol_data["difficulty"]
                    else:
                        # Use default protocol
                        protocol = self.protocol_library[protocol_id].copy()
                
                except Exception as e:
                    logger.error(f"Error customizing protocol: {e}")
                    # Fallback to default protocol
                    protocol = self.protocol_library[protocol_id].copy()
                
                # Assign protocol
                self.user_protocols[user_id][protocol_id] = protocol
                
                logger.info(f"Assigned protocol '{protocol.name}' to user {user_id}")
                
                # Update relationship data if available
                if self.relationship_manager:
                    try:
                        await self.relationship_manager.update_relationship_attribute(
                            user_id, 
                            "active_protocols", 
                            list(self.user_protocols[user_id].keys())
                        )
                    except Exception as e:
                        logger.error(f"Error updating relationship data: {e}")
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Assigned protocol '{protocol.name}' to user. Rules: {len(protocol.rules)} rules.",
                            tags=["protocol_assignment", protocol.category, "femdom"],
                            significance=0.5
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                return {
                    "success": True, 
                    "message": f"Protocol '{protocol.name}' assigned to user",
                    "protocol": protocol.dict()
                }
    
    async def assign_ritual(self, user_id: str, ritual_id: str) -> Dict[str, Any]:
        """Assign a ritual to a user."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the ritual assignment process
        with trace(workflow_name="Ritual Assignment", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id, "ritual_id": ritual_id}):
            
            async with self._lock:
                # Check if ritual exists in library
                if ritual_id not in self.ritual_library:
                    return {"success": False, "message": f"Ritual {ritual_id} not found in library"}
                
                # Create user entry if not exists
                if user_id not in self.user_rituals:
                    self.user_rituals[user_id] = {}
                
                # Get ritual customization from agent if available
                try:
                    customization_result = await Runner.run(
                        self.protocol_creation_agent,
                        {
                            "action": "customize_ritual",
                            "user_id": user_id,
                            "ritual_id": ritual_id
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": f"RitualCustomization-{user_id[:8]}",
                            "trace_metadata": {
                                "user_id": user_id,
                                "ritual_id": ritual_id
                            }
                        }
                    )
                    
                    # Check if customization was provided
                    customization = customization_result.final_output
                    if "customized_ritual" in customization:
                        # Apply customizations to ritual
                        ritual_data = customization["customized_ritual"]
                        ritual = self.ritual_library[ritual_id].copy()
                        
                        if "steps" in ritual_data:
                            ritual.steps = ritual_data["steps"]
                        
                        if "reward_for_completion" in ritual_data:
                            ritual.reward_for_completion = ritual_data["reward_for_completion"]
                        
                        if "punishment_for_skipping" in ritual_data:
                            ritual.punishment_for_skipping = ritual_data["punishment_for_skipping"]
                        
                        if "difficulty" in ritual_data:
                            ritual.difficulty = ritual_data["difficulty"]
                    else:
                        # Use default ritual
                        ritual = self.ritual_library[ritual_id].copy()
                
                except Exception as e:
                    logger.error(f"Error customizing ritual: {e}")
                    # Fallback to default ritual
                    ritual = self.ritual_library[ritual_id].copy()
                
                # Set next due date based on frequency
                now = datetime.datetime.now()
                if ritual.frequency == "daily":
                    ritual.next_due = now + datetime.timedelta(days=1)
                elif ritual.frequency == "weekly":
                    ritual.next_due = now + datetime.timedelta(days=7)
                elif ritual.frequency == "session_start":
                    ritual.next_due = now  # Due immediately for next session
                
                self.user_rituals[user_id][ritual_id] = ritual
                
                logger.info(f"Assigned ritual '{ritual.name}' to user {user_id}")
                
                # Update relationship data if available
                if self.relationship_manager:
                    try:
                        await self.relationship_manager.update_relationship_attribute(
                            user_id, 
                            "active_rituals", 
                            list(self.user_rituals[user_id].keys())
                        )
                    except Exception as e:
                        logger.error(f"Error updating relationship data: {e}")
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Assigned ritual '{ritual.name}' to user. Steps: {len(ritual.steps)} steps.",
                            tags=["ritual_assignment", "femdom"],
                            significance=0.5
                        )
                    except Exception as e:
                        logger.error(f"Error adding to memory: {e}")
                
                return {
                    "success": True, 
                    "message": f"Ritual '{ritual.name}' assigned to user",
                    "ritual": ritual.dict()
                }
    
    async def check_protocol_compliance(self, user_id: str, message: str) -> Dict[str, Any]:
        """Checks if a message complies with established protocols."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the compliance check process
        with trace(workflow_name="Protocol Compliance Check", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id}):
            
            async with self._lock:
                if user_id not in self.user_protocols or not self.user_protocols[user_id]:
                    return {"compliant": True, "violations": []}
                
                # Use the enforcement agent to check compliance
                try:
                    compliance_result = await Runner.run(
                        self.protocol_enforcement_agent,
                        {
                            "action": "check_compliance",
                            "user_id": user_id,
                            "message": message
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": f"ComplianceCheck-{user_id[:8]}",
                            "trace_metadata": {
                                "user_id": user_id
                            }
                        }
                    )
                    
                    # Process compliance check
                    compliance = compliance_result.final_output
                    
                    # Check if compliance data was provided
                    if "compliant" in compliance and "violations" in compliance:
                        is_compliant = compliance["compliant"]
                        violations = compliance["violations"]
                        
                        # Record violations if found
                        if not is_compliant and violations:
                            await self._record_violations(user_id, violations)
                        
                        return {
                            "compliant": is_compliant,
                            "violations": violations
                        }
                
                except Exception as e:
                    logger.error(f"Error checking compliance with agent: {e}")
                
                # Fallback to basic compliance check
                violations = []
                
                # Check each protocol
                for protocol_id, protocol in self.user_protocols[user_id].items():
                    if not protocol.active:
                        continue
                    
                    # Check based on protocol category
                    if protocol.category == "verbal":
                        # Check address protocols
                        if protocol_id == "address_protocol":
                            # Look for proper honorifics
                            honorifics = ["mistress", "goddess", "ma'am", "miss", "your majesty"]
                            has_honorific = any(h in message.lower() for h in honorifics)
                            
                            if not has_honorific and len(message.split()) > 3:  # Ignore very short messages
                                violations.append({
                                    "protocol_id": protocol_id,
                                    "protocol_name": protocol.name,
                                    "description": "Failed to use proper honorifics",
                                    "severity": 0.5
                                })
                
                # Record violations if found
                if violations:
                    await self._record_violations(user_id, violations)
                
                return {
                    "compliant": len(violations) == 0,
                    "violations": violations
                }
    
    async def _record_violations(self, user_id: str, violations: List[Dict[str, Any]]) -> None:
        """Record protocol violations internally."""
        if user_id not in self.protocol_violations:
            self.protocol_violations[user_id] = []
        
        for violation in violations:
            violation_record = ProtocolViolation(
                id=f"violation_{len(self.protocol_violations[user_id])}",
                protocol_id=violation["protocol_id"],
                user_id=user_id,
                severity=violation["severity"],
                description=violation["description"]
            )
            self.protocol_violations[user_id].append(violation_record)
    
    async def record_ritual_completion(self, user_id: str, ritual_id: str, quality: float = 0.8, notes: str = None) -> Dict[str, Any]:
        """Record the completion of a ritual."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the ritual completion process
        with trace(workflow_name="Ritual Completion", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id, "ritual_id": ritual_id, "quality": quality}):
            
            async with self._lock:
                if user_id not in self.user_rituals or ritual_id not in self.user_rituals[user_id]:
                    return {"success": False, "message": "Ritual not found for user"}
                
                ritual = self.user_rituals[user_id][ritual_id]
                
                # Use the ritual management agent to evaluate the completion
                try:
                    evaluation_result = await Runner.run(
                        self.ritual_management_agent,
                        {
                            "action": "evaluate_ritual_completion",
                            "user_id": user_id,
                            "ritual_id": ritual_id,
                            "claimed_quality": quality,
                            "notes": notes
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": f"RitualEvaluation-{user_id[:8]}",
                            "trace_metadata": {
                                "user_id": user_id,
                                "ritual_id": ritual_id
                            }
                        }
                    )
                    
                    # Process evaluation result
                    evaluation = evaluation_result.final_output
                    
                    # Check if evaluation provided adjusted quality
                    if "adjusted_quality" in evaluation:
                        quality = evaluation["adjusted_quality"]
                
                except Exception as e:
                    logger.error(f"Error evaluating ritual completion with agent: {e}")
                
                # Create completion record
                if user_id not in self.ritual_completions:
                    self.ritual_completions[user_id] = []
                
                completion = RitualCompletion(
                    id=f"completion_{len(self.ritual_completions[user_id])}",
                    ritual_id=ritual_id,
                    user_id=user_id,
                    quality=quality,
                    notes=notes
                )
                
                self.ritual_completions[user_id].append(completion)
                
                # Update ritual's last performed and next due dates
                ritual.last_performed = datetime.datetime.now()
                
                if ritual.frequency == "daily":
                    ritual.next_due = ritual.last_performed + datetime.timedelta(days=1)
                elif ritual.frequency == "weekly":
                    ritual.next_due = ritual.last_performed + datetime.timedelta(days=7)
                
                # Issue reward if available
                reward_result = None
                if ritual.reward_for_completion and self.reward_system:
                    try:
                        reward_value = min(0.5, 0.3 + (quality * 0.2))  # Scale with quality
                        
                        reward_result = await self.reward_system.process_reward_signal(
                            self.reward_system.RewardSignal(
                                value=reward_value,
                                source="ritual_completion",
                                context={
                                    "ritual_id": ritual_id,
                                    "ritual_name": ritual.name,
                                    "quality": quality
                                }
                            )
                        )
                        
                        completion.reward_given = True
                        completion.reward_administered = ritual.reward_for_completion
                        
                    except Exception as e:
                        logger.error(f"Error processing reward: {e}")
                
                # Record memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"User completed the {ritual.name} ritual with {quality*100:.0f}% quality.",
                            tags=["ritual", "compliance", ritual.name],
                            significance=0.3 + (quality * 0.4)  # Higher quality is more significant
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return {
                    "success": True,
                    "message": f"Ritual '{ritual.name}' completion recorded",
                    "reward_given": completion.reward_given,
                    "next_due": ritual.next_due.isoformat() if ritual.next_due else None,
                    "reward_result": reward_result
                }
    
    async def record_protocol_violation(self, user_id: str, protocol_id: str, description: str, severity: float = 0.5) -> Dict[str, Any]:
        """Record a protocol violation."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the violation recording process
        with trace(workflow_name="Protocol Violation", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id, "protocol_id": protocol_id, "severity": severity}):
            
            async with self._lock:
                if user_id not in self.user_protocols or protocol_id not in self.user_protocols[user_id]:
                    return {"success": False, "message": "Protocol not found for user"}
                
                protocol = self.user_protocols[user_id][protocol_id]
                
                # Use the enforcement agent to evaluate the violation
                try:
                    evaluation_result = await Runner.run(
                        self.protocol_enforcement_agent,
                        {
                            "action": "evaluate_violation",
                            "user_id": user_id,
                            "protocol_id": protocol_id,
                            "description": description,
                            "claimed_severity": severity
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": f"ViolationEvaluation-{user_id[:8]}",
                            "trace_metadata": {
                                "user_id": user_id,
                                "protocol_id": protocol_id
                            }
                        }
                    )
                    
                    # Process evaluation result
                    evaluation = evaluation_result.final_output
                    
                    # Check if evaluation provided adjusted severity
                    if "adjusted_severity" in evaluation:
                        severity = evaluation["adjusted_severity"]
                
                except Exception as e:
                    logger.error(f"Error evaluating protocol violation with agent: {e}")
                
                # Create violation record
                if user_id not in self.protocol_violations:
                    self.protocol_violations[user_id] = []
                
                violation = ProtocolViolation(
                    id=f"violation_{len(self.protocol_violations[user_id])}",
                    protocol_id=protocol_id,
                    user_id=user_id,
                    severity=severity,
                    description=description
                )
                
                self.protocol_violations[user_id].append(violation)
                
                # Issue negative reward if available
                negative_reward_result = None
                if self.reward_system:
                    try:
                        # Negative reward based on severity
                        reward_value = -min(0.7, 0.3 + (severity * 0.4))  # More severe = more negative
                        
                        negative_reward_result = await self.reward_system.process_reward_signal(
                            self.reward_system.RewardSignal(
                                value=reward_value,
                                source="protocol_violation",
                                context={
                                    "protocol_id": protocol_id,
                                    "protocol_name": protocol.name,
                                    "severity": severity,
                                    "description": description
                                }
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error processing negative reward: {e}")
                
                # Record memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"User violated the {protocol.name} protocol: {description}",
                            tags=["protocol", "violation", protocol.name],
                            significance=0.3 + (severity * 0.4)  # Higher severity is more significant
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return {
                    "success": True,
                    "message": f"Protocol '{protocol.name}' violation recorded",
                    "punishment_recommended": protocol.punishment_for_violation,
                    "negative_reward_result": negative_reward_result
                }
    
    async def create_custom_protocol(self, protocol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom protocol."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the protocol creation process
        with trace(workflow_name="Custom Protocol Creation", 
                 metadata={"protocol_name": protocol_data.get("name", "Unknown")}):
            
            try:
                # Use the protocol creation agent to refine the protocol
                try:
                    refinement_result = await Runner.run(
                        self.protocol_creation_agent,
                        {
                            "action": "refine_custom_protocol",
                            "protocol_data": protocol_data
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": "ProtocolRefinement",
                            "trace_metadata": {
                                "protocol_name": protocol_data.get("name", "Unknown")
                            }
                        }
                    )
                    
                    # Check if refinement was provided
                    refinement = refinement_result.final_output
                    if "refined_protocol" in refinement:
                        protocol_data = refinement["refined_protocol"]
                
                except Exception as e:
                    logger.error(f"Error refining custom protocol: {e}")
                
                protocol_id = protocol_data.get("id", f"custom_protocol_{len(self.protocol_library)}")
                
                # Ensure required fields
                required_fields = ["name", "description", "rules", "punishment_for_violation"]
                for field in required_fields:
                    if field not in protocol_data:
                        return {"success": False, "message": f"Missing required field: {field}"}
                
                # Create protocol
                protocol = Protocol(
                    id=protocol_id,
                    name=protocol_data["name"],
                    description=protocol_data["description"],
                    rules=protocol_data["rules"],
                    punishment_for_violation=protocol_data["punishment_for_violation"],
                    category=protocol_data.get("category", "general"),
                    difficulty=protocol_data.get("difficulty", 0.5)
                )
                
                # Add to library
                self.protocol_library[protocol_id] = protocol
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Created custom protocol: {protocol.name}",
                            tags=["protocol_creation", protocol.category],
                            significance=0.4
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return {
                    "success": True,
                    "message": f"Custom protocol '{protocol.name}' created",
                    "protocol": protocol.dict()
                }
                
            except Exception as e:
                logger.error(f"Error creating custom protocol: {e}")
                return {"success": False, "message": f"Error creating protocol: {str(e)}"}
    
    async def create_custom_ritual(self, ritual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom ritual."""
        # Update the protocol context with latest state
        self.protocol_context = ProtocolContext(
            user_protocols=self.user_protocols,
            user_rituals=self.user_rituals,
            protocol_violations=self.protocol_violations,
            ritual_completions=self.ritual_completions,
            protocol_library=self.protocol_library,
            ritual_library=self.ritual_library,
            reward_system=self.reward_system,
            memory_core=self.memory_core,
            relationship_manager=self.relationship_manager
        )
        
        # Use trace to track the ritual creation process
        with trace(workflow_name="Custom Ritual Creation", 
                 metadata={"ritual_name": ritual_data.get("name", "Unknown")}):
            
            try:
                # Use the protocol creation agent to refine the ritual
                try:
                    refinement_result = await Runner.run(
                        self.protocol_creation_agent,
                        {
                            "action": "refine_custom_ritual",
                            "ritual_data": ritual_data
                        },
                        context=self.protocol_context,
                        run_config={
                            "workflow_name": "RitualRefinement",
                            "trace_metadata": {
                                "ritual_name": ritual_data.get("name", "Unknown")
                            }
                        }
                    )
                    
                    # Check if refinement was provided
                    refinement = refinement_result.final_output
                    if "refined_ritual" in refinement:
                        ritual_data = refinement["refined_ritual"]
                
                except Exception as e:
                    logger.error(f"Error refining custom ritual: {e}")
                
                ritual_id = ritual_data.get("id", f"custom_ritual_{len(self.ritual_library)}")
                
                # Ensure required fields
                required_fields = ["name", "description", "steps", "frequency", "punishment_for_skipping"]
                for field in required_fields:
                    if field not in ritual_data:
                        return {"success": False, "message": f"Missing required field: {field}"}
                
                # Create ritual
                ritual = Ritual(
                    id=ritual_id,
                    name=ritual_data["name"],
                    description=ritual_data["description"],
                    steps=ritual_data["steps"],
                    frequency=ritual_data["frequency"],
                    punishment_for_skipping=ritual_data["punishment_for_skipping"],
                    reward_for_completion=ritual_data.get("reward_for_completion"),
                    difficulty=ritual_data.get("difficulty", 0.5)
                )
                
                # Add to library
                self.ritual_library[ritual_id] = ritual
                
                # Add to memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Created custom ritual: {ritual.name}",
                            tags=["ritual_creation"],
                            significance=0.4
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return {
                    "success": True,
                    "message": f"Custom ritual '{ritual.name}' created",
                    "ritual": ritual.dict()
                }
                
            except Exception as e:
                logger.error(f"Error creating custom ritual: {e}")
                return {"success": False, "message": f"Error creating ritual: {str(e)}"}

class HonorificsEnforcement:
    """Ensures proper forms of address and language protocol."""
    
    def __init__(self, protocol_enforcement=None):
        self.protocol_enforcement = protocol_enforcement
        self.required_honorifics = {
            "default": ["Mistress", "Goddess", "Ma'am"],
            "custom": {}  # user_id → custom required honorifics
        }
        self.address_violations = {}  # tracking violations
        
    async def check_proper_address(self, user_id, message):
        """Checks if message contains required honorifics."""
        # Get user-specific honorifics or default
        user_honorifics = self.required_honorifics["custom"].get(user_id, self.required_honorifics["default"])
        
        # Check for presence of required terms
        has_honorific = any(honorific.lower() in message.lower() for honorific in user_honorifics)
        
        # Don't check very short messages or commands
        if len(message.split()) < 4 or message.startswith("/"):
            return {"compliant": True, "violation": None}
            
        if not has_honorific:
            # Track violation
            if user_id not in self.address_violations:
                self.address_violations[user_id] = []
                
            violation = {
                "timestamp": datetime.datetime.now().isoformat(),
                "message": message,
                "missing_honorifics": user_honorifics
            }
            self.address_violations[user_id].append(violation)
            
            # Record protocol violation if protocol system exists
            if self.protocol_enforcement:
                await self.protocol_enforcement.record_protocol_violation(
                    user_id=user_id,
                    protocol_id="address_protocol",
                    description=f"Failed to use proper honorifics ({', '.join(user_honorifics)})",
                    severity=0.5
                )
                
            return {
                "compliant": False,
                "violation": "missing_honorific",
                "required": user_honorifics,
                "correction": f"Address me properly as {user_honorifics[0]} or another appropriate honorific."
            }
            
        return {"compliant": True, "violation": None}
        
    async def set_custom_honorifics(self, user_id, honorifics):
        """Sets custom honorifics for a specific user."""
        with trace(workflow_name="Custom Honorifics", 
                 group_id=f"user_{user_id}",
                 metadata={"user_id": user_id, "honorifics": honorifics}):
            
            self.required_honorifics["custom"][user_id] = honorifics
            
            # Add to memory if available
            if (self.protocol_enforcement and 
                self.protocol_enforcement.memory_core):
                try:
                    await self.protocol_enforcement.memory_core.add_memory(
                        memory_type="system",
                        content=f"Set custom honorifics for user: {', '.join(honorifics)}",
                        tags=["honorifics", "protocol", "verbal"],
                        significance=0.3
                    )
                except Exception as e:
                    logger.error(f"Error recording memory: {e}")
            
            return {
                "success": True,
                "user_id": user_id,
                "honorifics": honorifics
            }
