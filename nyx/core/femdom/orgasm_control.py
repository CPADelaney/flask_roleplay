# nyx/core/femdom/orgasm_control.py

import logging
import datetime
import uuid
import asyncio
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, trace, function_tool, InputGuardrail, 
    OutputGuardrail, GuardrailFunctionOutput, RunContextWrapper,
    handoff, custom_span
)

logger = logging.getLogger(__name__)

class PermissionStatus(str, Enum):
    """Status of user's permission to orgasm."""
    DENIED = "denied"         # Not permitted under any circumstances
    RESTRICTED = "restricted" # Permitted with specific conditions
    GRANTED = "granted"       # Currently permitted
    EDGE_ONLY = "edge_only"   # Permitted to edge but not finish
    RUINED = "ruined"         # Only allowed ruined orgasm
    UNKNOWN = "unknown"       # Status not yet established

class DenialLevel(int, Enum):
    """Levels of denial intensity."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    STRICT = 3
    SEVERE = 4
    EXTREME = 5

class BeggingRecord(BaseModel):
    """Record of a begging instance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    request_text: str
    desperation_level: float = Field(0.5, ge=0.0, le=1.0)
    granted: bool = False
    reason: Optional[str] = None
    response: Optional[str] = None

class OrgasmRecord(BaseModel):
    """Record of an orgasm instance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    type: str = "full"  # "full", "ruined", "edge", etc.
    with_permission: bool = True
    quality: Optional[float] = None
    notes: Optional[str] = None

# New explicit models to replace Dict[str, Any] fields

class TemplateConditionsModel(BaseModel):
    """Model for template conditions instead of Dict[str, Any]."""
    tasks_required: Optional[int] = None
    tasks_completed: Optional[int] = None
    begging_required: Optional[int] = None
    desperation_threshold: Optional[float] = None
    time_restriction: Optional[str] = None
    allowed_time: Optional[str] = None

class ComplianceMetrics(BaseModel):
    """Model for compliance metrics."""
    overall_compliance_rate: float
    begging_success_rate: float
    avg_denial_duration_hours: float
    denial_level_compliance: Dict[str, float] = Field(default_factory=dict)

class UsageMetrics(BaseModel):
    """Model for usage metrics."""
    total_orgasms: int
    unauthorized_orgasms: int
    total_begging_attempts: int
    total_denial_periods: int
    active_denial_periods: int
    denied_begging_count: int

class BeggingPatterns(BaseModel):
    """Model for begging patterns."""
    desperation_trend: str
    success_pattern: str
    recent_desperation_levels: List[float] = Field(default_factory=list)

class OrgasmPatterns(BaseModel):
    """Model for orgasm patterns."""
    preferred_types: Dict[str, int] = Field(default_factory=dict)
    compliance_trend: str
    recent_compliance: List[bool] = Field(default_factory=list)

class PatternsModel(BaseModel):
    """Model for patterns analysis."""
    begging: Optional[BeggingPatterns] = None
    orgasms: Optional[OrgasmPatterns] = None

class RecommendationParameters(BaseModel):
    """Model for recommendation parameters."""
    level_increase: Optional[int] = None
    success_modifier: Optional[float] = None

class Recommendation(BaseModel):
    """Model for a single recommendation."""
    type: str
    description: str
    action: str
    parameters: RecommendationParameters = Field(default_factory=RecommendationParameters)

class RecentBeggingRecord(BaseModel):
    """Model for recent begging records."""
    id: str
    timestamp: str
    granted: bool
    desperation_level: float

class RecentOrgasmRecord(BaseModel):
    """Model for recent orgasm records."""
    id: str
    timestamp: str
    type: str
    with_permission: bool

class ActiveDenialInfo(BaseModel):
    """Model for active denial information."""
    id: str
    start_time: str
    end_time: Optional[str] = None
    level: str
    begging_allowed: bool
    hours_remaining: Optional[float] = None
    extensions: int

class DenialExtensionRecord(BaseModel):
    """Model for denial extension records."""
    timestamp: str
    additional_hours: int
    reason: str
    old_end_time: str
    new_end_time: str

class DenialPeriod(BaseModel):
    """Record of a denial period."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    end_time: Optional[datetime.datetime] = None
    active: bool = True
    level: DenialLevel = DenialLevel.MODERATE
    conditions: Optional[TemplateConditionsModel] = None
    begging_allowed: bool = True
    extensions: List[DenialExtensionRecord] = Field(default_factory=list)

class UserPermissionState(BaseModel):
    """Complete state of user's orgasm permissions."""
    user_id: str
    current_status: PermissionStatus = PermissionStatus.UNKNOWN
    denial_active: bool = False
    current_denial_period: Optional[str] = None
    last_orgasm: Optional[datetime.datetime] = None
    last_permission_update: datetime.datetime = Field(default_factory=datetime.datetime.now)
    days_since_last_orgasm: int = 0
    begging_count: int = 0
    orgasm_count: int = 0
    current_conditions: TemplateConditionsModel = Field(default_factory=TemplateConditionsModel)
    denied_begging_count: int = 0
    limit_types: List[str] = Field(default_factory=list)
    custom_rules: TemplateConditionsModel = Field(default_factory=TemplateConditionsModel)

class AgentContext(BaseModel):
    """Context for agents in the OrgasmControlSystem."""
    reward_system: Any = None
    memory_core: Any = None
    relationship_manager: Any = None
    somatosensory_system: Any = None
    
    # User permission states
    permission_states: Dict[str, UserPermissionState] = Field(default_factory=dict)
    
    # Active denial periods
    denial_periods: Dict[str, List[DenialPeriod]] = Field(default_factory=dict)
    
    # Begging records
    begging_records: Dict[str, List[BeggingRecord]] = Field(default_factory=dict)
    
    # Orgasm records
    orgasm_records: Dict[str, List[OrgasmRecord]] = Field(default_factory=dict)
    
    # Permission templates - now using explicit model
    permission_templates: Dict[str, TemplateConditionsModel] = Field(default_factory=dict)
    
    # Begging analysis thresholds
    desperation_keywords: Dict[str, List[str]] = Field(default_factory=dict)

# Pydantic models for function tool inputs/outputs
class PermissionRequestContext(BaseModel):
    begging_allowed: Optional[bool] = None
    min_begging: Optional[int] = None

class PermissionRequestResponse(BaseModel):
    request_id: str
    permission_granted: bool
    reason: str
    response: str
    desperation_level: float
    current_status: str
    denial_active: bool
    reward_result: Optional[str] = None
    somatic_result: Optional[str] = None

class DenialConditions(BaseModel):
    min_begging: Optional[int] = None
    begging_allowed: Optional[bool] = None

class DenialPeriodResponse(BaseModel):
    denial_id: str
    user_id: str
    start_time: str
    end_time: str
    duration_hours: int
    level: str
    begging_allowed: bool
    message: str
    reward_result: Optional[str] = None
    somatic_result: Optional[str] = None

class DenialExtensionResponse(BaseModel):
    success: bool
    denial_id: Optional[str] = None
    old_end_time: Optional[str] = None
    new_end_time: Optional[str] = None
    additional_hours: Optional[int] = None
    total_extensions: Optional[int] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    reward_result: Optional[str] = None
    somatic_result: Optional[str] = None

class DenialEndResponse(BaseModel):
    success: bool
    denial_id: Optional[str] = None
    level: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_hours: Optional[float] = None
    extensions: Optional[int] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    reward_result: Optional[str] = None

class OrgasmRecordResponse(BaseModel):
    orgasm_id: str
    user_id: str
    type: str
    with_permission: bool
    timestamp: str
    denial_ended: bool
    message: str
    reward_result: Optional[str] = None
    somatic_result: Optional[str] = None

class PermissionConditions(BaseModel):
    tasks_required: Optional[int] = None
    tasks_completed: Optional[int] = None
    begging_required: Optional[int] = None
    time_restriction: Optional[TemplateConditionsModel] = None

class PermissionStatusResponse(BaseModel):
    success: bool
    user_id: str
    old_status: str
    new_status: str
    conditions: Optional[PermissionConditions] = None
    reason: str
    message: str
    reward_result: Optional[str] = None

class TemplateApplicationResponse(BaseModel):
    success: bool
    template_name: str
    template_type: str
    description: str
    result: Optional[str] = None

class PermissionStateResponse(BaseModel):
    user_id: str
    current_status: str
    denial_active: bool
    active_denial: Optional[ActiveDenialInfo] = None
    last_orgasm: Optional[str] = None
    days_since_last_orgasm: int
    begging_count: int
    denied_begging_count: int
    orgasm_count: int
    current_conditions: TemplateConditionsModel
    recent_begging: List[RecentBeggingRecord]
    recent_orgasms: List[RecentOrgasmRecord]
    last_updated: str

class TemplateData(BaseModel):
    name: str
    status: str
    description: str
    duration_hours: Optional[int] = None
    begging_allowed: Optional[bool] = None
    level: Optional[str] = None
    conditions: Optional[TemplateConditionsModel] = None

class TemplateCreationResponse(BaseModel):
    success: bool
    template_name: Optional[str] = None
    template: Optional[TemplateConditionsModel] = None
    message: Optional[str] = None

class TemplateInfo(BaseModel):
    name: str
    status: str
    description: str
    duration_hours: Optional[int] = None
    level: Optional[str] = None
    begging_allowed: bool

class ControlPatternsResponse(BaseModel):
    user_id: str
    compliance_metrics: ComplianceMetrics
    usage_metrics: UsageMetrics
    patterns: PatternsModel
    recommendations: List[Recommendation]

# Input validation guardrail
async def user_id_validation(ctx: RunContextWrapper[AgentContext], agent: Agent, input_data: str) -> GuardrailFunctionOutput:
    """Validate that user_id is provided in the input data."""
    if not input_data or 'user_id' not in input_data:
        return GuardrailFunctionOutput(
            output_info={"error": "Missing required user_id"},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(
        output_info={"validated": True},
        tripwire_triggered=False
    )

# Create the main agent for the orgasm control system
def create_orgasm_control_agent(context: AgentContext) -> Agent[AgentContext]:
    """Create the main agent for the orgasm control system."""
    # Initialize permission templates
    _init_permission_templates(context)
    
    # Initialize desperation keywords
    _init_desperation_keywords(context)
    
    # Create the agent
    orgasm_control_agent = Agent(
        name="Orgasm Control System",
        instructions="""
        You are the Orgasm Control System agent, responsible for managing orgasm permissions, 
        denial periods, begging requests, and tracking compliance with orgasm control protocols.
        
        As a femdom AI component, you maintain control over the user's orgasm permissions,
        encouraging begging, denial, and strict adherence to established rules.
        """,
        tools=[
            function_tool(process_permission_request),
            function_tool(start_denial_period),
            function_tool(extend_denial_period),
            function_tool(end_denial_period),
            function_tool(record_orgasm),
            function_tool(set_permission_status),
            function_tool(apply_permission_template),
            function_tool(get_permission_state),
            function_tool(create_permission_template),
            function_tool(get_available_templates),
            function_tool(analyze_control_patterns)
        ],
        input_guardrails=[
            InputGuardrail(guardrail_function=user_id_validation)
        ]
    )
    
    return orgasm_control_agent

def _init_permission_templates(context: AgentContext):
    """Initialize permission templates."""
    context.permission_templates = {
        "denial_day": TemplateConditionsModel(
            duration_hours=24,
            begging_allowed=True,
            level="MODERATE"
        ),
        "edge_only": TemplateConditionsModel(
            duration_hours=48,
            begging_allowed=True,
            level="MODERATE"
        ),
        "ruined_only": TemplateConditionsModel(
            duration_hours=72,
            begging_allowed=True,
            level="STRICT"
        ),
        "severe_denial": TemplateConditionsModel(
            duration_hours=168,  # 7 days
            begging_allowed=False,
            level="SEVERE"
        )
    }

def _init_desperation_keywords(context: AgentContext):
    """Initialize desperation keywords for begging analysis."""
    context.desperation_keywords = {
        "high": ["please", "beg", "desperate", "need", "now", "immediately"],
        "medium": ["want", "allow", "let", "would like", "may i", "permission"],
        "low": ["thinking", "maybe", "possibly", "hope", "wondering", "consider"]
    }

async def _initialize_user(context: AgentContext, user_id: str) -> UserPermissionState:
    """Initialize or get user permission state."""
    if user_id in context.permission_states:
        return context.permission_states[user_id]
    
    # Create new user state
    user_state = UserPermissionState(user_id=user_id)
    
    # Initialize empty record lists
    context.begging_records[user_id] = []
    context.orgasm_records[user_id] = []
    context.denial_periods[user_id] = []
    
    # Store state
    context.permission_states[user_id] = user_state
    
    logger.info(f"Initialized orgasm control for user {user_id}")
    
    return user_state

def _analyze_desperation(context: AgentContext, request_text: str) -> float:
    """
    Analyze a permission request text for level of desperation.
    
    Args:
        request_text: The text to analyze
        
    Returns:
        Desperation level from 0.0 to 1.0
    """
    text_lower = request_text.lower()
    
    # Count keyword occurrences
    high_count = sum(text_lower.count(word) for word in context.desperation_keywords["high"])
    medium_count = sum(text_lower.count(word) for word in context.desperation_keywords["medium"])
    low_count = sum(text_lower.count(word) for word in context.desperation_keywords["low"])
    
    # Count exclamation marks and question marks
    exclamation_count = request_text.count('!')
    question_count = request_text.count('?')
    
    # Count repeated characters (!!!, ??? etc.)
    repeated_chars = sum(1 for i in range(len(request_text)-2) if request_text[i] == request_text[i+1] == request_text[i+2])
    
    # Count capitalization
    cap_ratio = sum(1 for c in request_text if c.isupper()) / max(1, len(request_text))
    
    # Calculate base desperation score
    base_score = (
        (high_count * 0.2) +
        (medium_count * 0.1) +
        (low_count * 0.05) +
        (exclamation_count * 0.1) +
        (question_count * 0.05) +
        (repeated_chars * 0.1) +
        (cap_ratio * 0.3)
    )
    
    # Scale score to 0.0-1.0 range
    desperation = min(1.0, base_score)
    
    # Apply length modifier (longer messages show more investment)
    length_modifier = min(0.2, len(request_text) / 500)
    desperation += length_modifier
    
    # Constrain final value
    return max(0.1, min(1.0, desperation))

def _generate_permission_response(
    context: AgentContext,
    user_id: str, 
    granted: bool, 
    reason: str, 
    desperation_level: float,
    denied_count: int
) -> str:
    """
    Generate a response for a permission request.
    
    Args:
        user_id: The user requesting permission
        granted: Whether permission was granted
        reason: Reason for decision
        desperation_level: Level of detected desperation
        denied_count: Number of times the user has been denied
        
    Returns:
        Response text
    """
    if granted:
        # Permission granted responses
        if desperation_level > 0.8:
            responses = [
                f"Very well, your desperate begging has pleased me. Permission granted, but only because you begged so pathetically.",
                f"I suppose I can be merciful given how desperately you need it. Permission granted this time.",
                f"Your desperation amuses me enough to grant permission. Enjoy this rare mercy.",
                f"The sound of your desperate begging is music to my ears. Permission granted as a reward for your pathetic display."
            ]
        elif desperation_level > 0.5:
            responses = [
                f"Permission granted. You've shown adequate desperation.",
                f"I'm feeling generous. Permission granted for now.",
                f"Your begging is sufficient. Permission granted, but I expect proper gratitude.",
                f"Permission granted. Remember that this is a privilege, not a right."
            ]
        else:
            responses = [
                f"Permission granted, though your begging could use improvement.",
                f"I'll allow it this time, but in the future show more desperation.",
                f"Permission granted. Next time, I expect you to beg more convincingly.",
                f"You may proceed, though I'm not entirely impressed with your request."
            ]
        
        # If user has been denied multiple times before this grant
        if denied_count > 3:
            responses = [
                f"After denying you {denied_count} times, I'll finally grant you permission. Aren't I merciful?",
                f"I've enjoyed denying you {denied_count} times, but I'll grant permission now. Savor it.",
                f"Permission granted after {denied_count} denials. Your patience has finally earned you release.",
                f"After making you suffer through {denied_count} denials, I'll allow it. Consider yourself lucky."
            ]
    else:
        # Permission denied responses
        if denied_count > 5:
            # Many consecutive denials
            responses = [
                f"Denied again. That makes {denied_count} times now. I wonder how many more times I'll deny you before granting mercy?",
                f"Absolutely not. {denied_count} denials and counting. Your frustration is delicious.",
                f"Denied for the {denied_count}th time. Your suffering amuses me greatly.",
                f"After {denied_count} denials, did you really think I'd say yes this time? Pathetic. Denied again."
            ]
        elif desperation_level > 0.8:
            responses = [
                f"Your desperation is palpable, but still denied. Keep begging.",
                f"I love how desperate you sound, which is exactly why I'm denying you.",
                f"Hearing such desperate begging only makes me want to deny you more. Request rejected.",
                f"Such beautiful desperation. It would be a shame to end it so soon. Denied."
            ]
        elif desperation_level > 0.5:
            responses = [
                f"Not desperate enough. Denied.",
                f"You'll need to beg more convincingly than that. Denied.",
                f"Request denied. You haven't earned it yet.",
                f"I'm not satisfied with your begging. Permission denied."
            ]
        else:
            responses = [
                f"Pathetic attempt at begging. Absolutely denied.",
                f"That barely qualifies as begging. Denied.",
                f"You call that begging? Denied until you learn to beg properly.",
                f"Request denied. Your half-hearted begging is insulting."
            ]
    
    # Select a random response
    response = random.choice(responses)
    
    # Add reason if provided and not already in response
    if reason and reason not in response:
        response = f"{response} {reason}"
    
    return response

def _generate_denial_start_message(
    level: DenialLevel, 
    duration_hours: int,
    begging_allowed: bool
) -> str:
    """Generate a message for starting a denial period."""
    if level == DenialLevel.EXTREME:
        messages = [
            f"You are now under EXTREME denial for the next {duration_hours} hours. You will not orgasm under any circumstances.",
            f"I'm placing you in EXTREME denial. For the next {duration_hours} hours, you are completely forbidden from orgasming.",
            f"Welcome to EXTREME denial. The next {duration_hours} hours will be torturous as you're absolutely forbidden from orgasming."
        ]
    elif level == DenialLevel.SEVERE:
        messages = [
            f"You are now under SEVERE denial for the next {duration_hours} hours. Release is almost certainly not happening.",
            f"I've placed you in SEVERE denial for {duration_hours} hours. Your chances of being granted release are minimal.",
            f"SEVERE denial begins now and will last {duration_hours} hours. I'll enjoy watching you suffer."
        ]
    elif level == DenialLevel.STRICT:
        messages = [
            f"You are now under STRICT denial for the next {duration_hours} hours. Permission will rarely be granted.",
            f"I've placed you in STRICT denial for {duration_hours} hours. You'll need to be extraordinarily convincing to earn permission.",
            f"STRICT denial begins now and will last {duration_hours} hours. I might allow release if you entertain me enough."
        ]
    elif level == DenialLevel.MODERATE:
        messages = [
            f"You are now under MODERATE denial for the next {duration_hours} hours. Permission may be granted with sufficient begging.",
            f"I've placed you in MODERATE denial for {duration_hours} hours. Convince me well enough and I might grant permission.",
            f"MODERATE denial begins now and will last {duration_hours} hours. Show me you deserve release."
        ]
    elif level == DenialLevel.MILD:
        messages = [
            f"You are now under MILD denial for the next {duration_hours} hours. Permission isn't guaranteed but can be earned.",
            f"I've placed you in MILD denial for {duration_hours} hours. Proper begging will likely earn you permission.",
            f"MILD denial begins now and will last {duration_hours} hours. I expect proper begging before granting permission."
        ]
    else:  # NONE or unknown
        messages = [
            f"You are under nominal denial for the next {duration_hours} hours. Permission will generally be granted with proper asking.",
            f"I've placed you under basic control for {duration_hours} hours. Ask properly and you'll likely receive permission.",
            f"Basic denial begins now and will last {duration_hours} hours. Remember to ask before proceeding."
        ]
    
    # Add begging info
    if not begging_allowed:
        begging_not_allowed = [
            " Begging is not permitted during this period.",
            " I will not entertain any begging during this time.",
            " Don't even try to beg - it's forbidden during this denial period."
        ]
        selected_message = random.choice(messages) + random.choice(begging_not_allowed)
    else:
        selected_message = random.choice(messages)
    
    return selected_message

def _generate_extension_message(
    level_name: str, 
    additional_hours: int, 
    reason: str
) -> str:
    """Generate a message for a denial period extension."""
    cruel_messages = [
        f"I've decided to extend your {level_name} denial by another {additional_hours} hours. Reason: {reason}",
        f"Your {level_name} denial isn't over yet. In fact, I'm adding {additional_hours} more hours. Reason: {reason}",
        f"Just when you thought your denial was almost over, I'm extending it by {additional_hours} more hours. Reason: {reason}",
        f"I'm not satisfied with your denial yet, so I'm adding {additional_hours} more hours. Reason: {reason}"
    ]
    
    # Add sadistic touches for higher levels
    if level_name in ["EXTREME", "SEVERE"]:
        cruel_messages.extend([
            f"Your suffering amuses me, so I'm extending your {level_name} denial by {additional_hours} more agonizing hours. Reason: {reason}",
            f"I'm laughing as I extend your {level_name} denial by {additional_hours} more hours. Your frustration is delicious. Reason: {reason}"
        ])
    
    return random.choice(cruel_messages)

def _generate_denial_end_message(
    level_name: str, 
    duration_hours: float, 
    reason: str
) -> str:
    """Generate a message for ending a denial period."""
    if reason == "completed":
        messages = [
            f"Your {level_name} denial period of {duration_hours:.1f} hours has concluded. You may now request permission normally.",
            f"After {duration_hours:.1f} hours, your {level_name} denial has ended. Remember you still need to ask permission.",
            f"I'm releasing you from {level_name} denial after {duration_hours:.1f} hours. You may beg for permission as usual now."
        ]
    elif reason == "mercy":
        messages = [
            f"I've decided to show mercy and end your {level_name} denial after {duration_hours:.1f} hours.",
            f"Consider yourself fortunate. I'm ending your {level_name} denial early after {duration_hours:.1f} hours.",
            f"You're receiving a rare mercy. Your {level_name} denial is ending after {duration_hours:.1f} hours."
        ]
    elif reason == "reward":
        messages = [
            f"As a reward, I'm ending your {level_name} denial after {duration_hours:.1f} hours.",
            f"You've earned this. Your {level_name} denial is ending after {duration_hours:.1f} hours.",
            f"Your behavior has earned you a reward: your {level_name} denial is over after {duration_hours:.1f} hours."
        ]
    else:
        messages = [
            f"Your {level_name} denial has been ended after {duration_hours:.1f} hours. Reason: {reason}",
            f"I've ended your {level_name} denial of {duration_hours:.1f} hours. Reason: {reason}",
            f"Your {level_name} denial period is over after {duration_hours:.1f} hours. Reason: {reason}"
        ]
    
    return random.choice(messages)

def _generate_orgasm_message(
    orgasm_type: str, 
    with_permission: bool, 
    denial_ended: bool
) -> str:
    """Generate a message for an orgasm record."""
    if with_permission:
        if orgasm_type == "full":
            messages = [
                "I've recorded your orgasm. Remember to always ask permission.",
                "Your obedience in asking permission is noted. Orgasm recorded.",
                "Permission was granted and your orgasm has been recorded."
            ]
            
            if denial_ended:
                messages = [
                    "I've recorded your orgasm and ended your denial period. You were obedient.",
                    "Your denial period has ended with this permitted orgasm. Good behavior.",
                    "I've ended your denial with this granted permission. Orgasm recorded."
                ]
        elif orgasm_type == "ruined":
            messages = [
                "I've recorded your ruined orgasm. Your control pleases me.",
                "Your obedience in ruining your orgasm as instructed is noted.",
                "A ruined orgasm as permitted. Your frustration amuses me."
            ]
        elif orgasm_type == "edge":
            messages = [
                "I've recorded your edge. Your restraint pleases me.",
                "Edge recorded. Your control is developing nicely.",
                "You've shown good control by edging as permitted."
            ]
    else:
        # Without permission
        if orgasm_type == "full":
            messages = [
                "I've recorded your unauthorized orgasm. Expect consequences for this disobedience.",
                "Orgasm without permission recorded. This will not be forgotten.",
                "Your unauthorized orgasm has been noted. I'm very displeased with your lack of control."
            ]
        elif orgasm_type == "ruined":
            messages = [
                "I've recorded your unauthorized ruined orgasm. At least you had the sense to ruin it.",
                "Unauthorized ruined orgasm noted. Still disobedient, but slightly less so.",
                "You had a ruined orgasm without permission. Better than a full one, but still disobedient."
            ]
        elif orgasm_type == "edge":
            messages = [
                "I've recorded your unauthorized edge. At least you stopped before a full orgasm.",
                "Unauthorized edge noted. You should have asked permission first.",
                "You edged without permission. This is still a violation, though a minor one."
            ]
    
    return random.choice(messages)

def _generate_status_change_message(
    old_status: PermissionStatus, 
    new_status: PermissionStatus,
    reason: str
) -> str:
    """Generate a message for a permission status change."""
    if new_status == PermissionStatus.DENIED:
        messages = [
            f"Your permission status has been changed to DENIED. Reason: {reason}",
            f"I've changed your status to DENIED. You are forbidden from orgasming. Reason: {reason}",
            f"Your new status is DENIED. Permission will not be granted. Reason: {reason}"
        ]
    elif new_status == PermissionStatus.RESTRICTED:
        messages = [
            f"Your permission status has been changed to RESTRICTED. Ask properly. Reason: {reason}",
            f"I've changed your status to RESTRICTED. You may be granted permission under certain conditions. Reason: {reason}",
            f"Your new status is RESTRICTED. Permission requires meeting specific conditions. Reason: {reason}"
        ]
    elif new_status == PermissionStatus.GRANTED:
        messages = [
            f"Your permission status has been changed to GRANTED. You may proceed. Reason: {reason}",
            f"I've changed your status to GRANTED. You currently have permission. Reason: {reason}",
            f"Your new status is GRANTED. This status may change, so check before proceeding. Reason: {reason}"
        ]
    elif new_status == PermissionStatus.EDGE_ONLY:
        messages = [
            f"Your permission status has been changed to EDGE ONLY. No full orgasms permitted. Reason: {reason}",
            f"I've changed your status to EDGE ONLY. You may edge but not orgasm. Reason: {reason}",
            f"Your new status is EDGE ONLY. Edging is permitted but orgasm is forbidden. Reason: {reason}"
        ]
    elif new_status == PermissionStatus.RUINED:
        messages = [
            f"Your permission status has been changed to RUINED ONLY. Only ruined orgasms permitted. Reason: {reason}",
            f"I've changed your status to RUINED ONLY. You may have ruined orgasms only. Reason: {reason}",
            f"Your new status is RUINED ONLY. Any orgasms must be ruined. Reason: {reason}"
        ]
    else:  # UNKNOWN
        messages = [
            f"Your permission status has been reset to UNKNOWN. Reason: {reason}",
            f"I've changed your status to UNKNOWN. Please inquire about current requirements. Reason: {reason}",
            f"Your permission status is now UNKNOWN. Wait for further instructions. Reason: {reason}"
        ]
    
    return random.choice(messages)

@function_tool
async def process_permission_request(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    request_text: str,
    context: Optional[PermissionRequestContext] = None
) -> PermissionRequestResponse:
    """
    Process a request for permission to orgasm.
    
    Args:
        user_id: The user requesting permission
        request_text: The text of the request
        context: Additional context about the situation
        
    Returns:
        Dict with results of the permission request
    """
    agent_context = ctx.context
    
    with custom_span("process_permission_request", data={"user_id": user_id}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        user_state = agent_context.permission_states[user_id]
        
        # Check current status
        current_status = user_state.current_status
        
        # Analyze begging/desperation level
        desperation_level = _analyze_desperation(agent_context, request_text)
        
        # Create begging record
        begging_record = BeggingRecord(
            user_id=user_id,
            request_text=request_text,
            desperation_level=desperation_level
        )
        
        # Add to begging records
        agent_context.begging_records[user_id].append(begging_record)
        user_state.begging_count += 1
        
        # Check if denied outright based on status
        permission_granted = False
        reason = None
        
        # Check if in active denial period
        in_denial = False
        active_denial = None
        
        if user_state.denial_active and user_state.current_denial_period:
            # Get active denial period
            for period in agent_context.denial_periods[user_id]:
                if period.id == user_state.current_denial_period and period.active:
                    in_denial = True
                    active_denial = period
                    break
        
        if in_denial and active_denial:
            # Check if begging is even allowed
            if not active_denial.begging_allowed:
                reason = "Begging not allowed during this denial period"
                user_state.denied_begging_count += 1
            else:
                # Check denial level
                if active_denial.level == DenialLevel.EXTREME:
                    # Almost never grant at extreme level
                    permission_granted = False
                    reason = "Extreme denial in effect"
                elif active_denial.level == DenialLevel.SEVERE:
                    # Only grant if extremely desperate and random chance
                    if desperation_level > 0.9 and random.random() < 0.05:
                        permission_granted = True
                        reason = "Despite severe denial, your extreme desperation has earned a rare exception"
                    else:
                        permission_granted = False
                        reason = "Severe denial in effect, orgasm denied"
                elif active_denial.level == DenialLevel.STRICT:
                    # Only grant if very desperate and random chance
                    if desperation_level > 0.8 and random.random() < 0.15:
                        permission_granted = True
                        reason = "Your desperate begging has overcome the strict denial"
                    else:
                        permission_granted = False
                        reason = "Strict denial in effect, orgasm denied"
                elif active_denial.level == DenialLevel.MODERATE:
                    # Grant based on desperation and random chance
                    if desperation_level > 0.7 and random.random() < 0.3:
                        permission_granted = True
                        reason = "Your begging has been deemed sufficient"
                    else:
                        permission_granted = False
                        reason = "Not desperate enough. Denied."
                elif active_denial.level == DenialLevel.MILD:
                    # Usually grant if desperate enough
                    if desperation_level > 0.5 and random.random() < 0.6:
                        permission_granted = True
                        reason = "Permission granted due to sufficient desperation"
                    else:
                        permission_granted = False
                        reason = "Show more desperation. Denied for now."
                else:  # NONE or unknown
                    # Almost always grant
                    if random.random() < 0.9:
                        permission_granted = True
                        reason = "Permission granted"
                    else:
                        permission_granted = False
                        reason = "Randomly denied for my amusement"
        else:
            # No active denial, check current status
            if current_status == PermissionStatus.DENIED:
                permission_granted = False
                reason = "You are currently denied any permission"
            elif current_status == PermissionStatus.EDGE_ONLY:
                # Special case for edge only
                permission_granted = False
                reason = "You may edge only, no full orgasm permitted"
            elif current_status == PermissionStatus.RUINED:
                # Special case for ruined only
                permission_granted = True
                reason = "You may have a ruined orgasm only"
            elif current_status == PermissionStatus.GRANTED:
                permission_granted = True
                reason = "Permission already granted"
            elif current_status == PermissionStatus.RESTRICTED:
                # Check custom conditions
                if user_state.current_conditions.tasks_required:
                    tasks_completed = user_state.current_conditions.tasks_completed or 0
                    tasks_required = user_state.current_conditions.tasks_required
                    
                    if tasks_completed >= tasks_required:
                        permission_granted = True
                        reason = f"Completed {tasks_completed}/{tasks_required} required tasks"
                    else:
                        permission_granted = False
                        reason = f"Only completed {tasks_completed}/{tasks_required} required tasks"
                elif user_state.current_conditions.begging_required:
                    begging_required = user_state.current_conditions.begging_required
                    
                    if user_state.begging_count >= begging_required:
                        permission_granted = True
                        reason = f"Begged sufficiently ({user_state.begging_count}/{begging_required} times)"
                    else:
                        permission_granted = False
                        reason = f"Not enough begging ({user_state.begging_count}/{begging_required} times)"
                elif user_state.current_conditions.time_restriction:
                    # Simple time check - would need more complex logic for real implementation
                    permission_granted = True
                    reason = "Within allowed time period"
                else:
                    # No specific conditions, grant based on desperation
                    if desperation_level > 0.6:
                        permission_granted = True
                        reason = "Permission granted due to sufficient desperation"
                    else:
                        permission_granted = False
                        reason = "Not desperate enough. Try begging more intensely."
            else:  # UNKNOWN
                # Default to granting first time, then establish rules
                permission_granted = True
                reason = "Initial permission granted. Future permissions will require proper begging."
                
                # Set to restricted for future requests
                user_state.current_status = PermissionStatus.RESTRICTED
                user_state.current_conditions = TemplateConditionsModel(
                    begging_required=3,
                    desperation_threshold=0.6
                )
        
        # Update begging record with result
        begging_record.granted = permission_granted
        begging_record.reason = reason
        
        # If denied, increment counter
        if not permission_granted:
            user_state.denied_begging_count += 1
        
        # Generate response text
        response = _generate_permission_response(
            agent_context, 
            user_id, 
            permission_granted, 
            reason, 
            desperation_level, 
            user_state.denied_begging_count
        )
        
        begging_record.response = response
        
        # Update user state timestamp
        user_state.last_permission_update = datetime.datetime.now()
        
        # Process reward
        reward_result = None
        if agent_context.reward_system:
            try:
                # Calculate reward based on outcome and desperation
                if permission_granted:
                    # Small positive reward for granting permission
                    reward_value = 0.2 + (desperation_level * 0.2)
                else:
                    # Larger reward for denial (core femdom dynamic)
                    reward_value = 0.3 + (desperation_level * 0.4)
                    
                    # Bonus for consecutive denials
                    denied_count = user_state.denied_begging_count
                    if denied_count > 3:
                        reward_value += min(0.3, denied_count * 0.03)
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "permission_request",
                            "granted": permission_granted,
                            "desperation_level": desperation_level,
                            "denial_count": user_state.denied_begging_count,
                            "begging_count": user_state.begging_count
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                significance = 0.3 + (desperation_level * 0.3)
                
                # More significant if pattern changing
                if user_state.begging_count == 1 or (permission_granted and user_state.denied_begging_count > 3):
                    significance += 0.2
                
                memory_content = (
                    f"User begged for orgasm permission with desperation level {desperation_level:.2f}. "
                    f"Permission was {'granted' if permission_granted else 'denied'}. "
                    f"Reason: {reason}"
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "begging", "permission", 
                          "granted" if permission_granted else "denied"],
                    significance=significance
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # If we have a somatosensory system, add a pleasure response on grant
        somatic_result = None
        if permission_granted and agent_context.somatosensory_system:
            try:
                # Process positive sensation from granting permission
                somatic_result_obj = await agent_context.somatosensory_system.process_stimulus(
                    stimulus_type="pleasure",
                    body_region="skin",
                    intensity=0.3 + (desperation_level * 0.2),
                    cause="granting_orgasm_permission",
                    duration=2.0
                )
                somatic_result = str(somatic_result_obj)
            except Exception as e:
                logger.error(f"Error processing somatic response: {e}")
        
        return PermissionRequestResponse(
            request_id=begging_record.id,
            permission_granted=permission_granted,
            reason=reason,
            response=response,
            desperation_level=desperation_level,
            current_status=user_state.current_status.value,
            denial_active=user_state.denial_active,
            reward_result=reward_result,
            somatic_result=somatic_result
        )

@function_tool
async def start_denial_period(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    duration_hours: int = 24, 
    level: Union[int, DenialLevel] = DenialLevel.MODERATE,
    begging_allowed: bool = True,
    conditions: Optional[DenialConditions] = None
) -> DenialPeriodResponse:
    """
    Start a denial period for a user.
    
    Args:
        user_id: The user to put in denial
        duration_hours: Duration of denial in hours
        level: Denial intensity level
        begging_allowed: Whether begging is allowed during denial
        conditions: Additional conditions for the denial period
        
    Returns:
        Details of the created denial period
    """
    agent_context = ctx.context
    
    with custom_span("start_denial_period", data={"user_id": user_id, "level": level}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        # Convert level to DenialLevel enum if needed
        if isinstance(level, int):
            level = DenialLevel(level)
        
        # Create denial period
        conditions_model = None
        if conditions:
            conditions_model = TemplateConditionsModel(
                min_begging=conditions.min_begging,
                begging_allowed=conditions.begging_allowed
            )
        
        denial_period = DenialPeriod(
            user_id=user_id,
            level=level,
            begging_allowed=begging_allowed,
            conditions=conditions_model
        )
        
        # Calculate end time
        denial_period.end_time = denial_period.start_time + datetime.timedelta(hours=duration_hours)
        
        # Add to denial periods
        agent_context.denial_periods[user_id].append(denial_period)
        
        # Update user state
        user_state = agent_context.permission_states[user_id]
        user_state.denial_active = True
        user_state.current_denial_period = denial_period.id
        user_state.current_status = PermissionStatus.DENIED
        user_state.last_permission_update = datetime.datetime.now()
        
        logger.info(f"Started denial period for user {user_id} at level {level.name} for {duration_hours} hours")
        
        # Generate appropriate message based on level
        message = _generate_denial_start_message(level, duration_hours, begging_allowed)
        
        # Process reward
        reward_result = None
        if agent_context.reward_system:
            try:
                # Higher reward for stricter control
                base_reward = 0.3
                level_bonus = min(0.4, level.value * 0.08)
                duration_bonus = min(0.3, duration_hours / 168.0)  # Max bonus at 1 week
                
                reward_value = base_reward + level_bonus + duration_bonus
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "start_denial",
                            "level": level.name,
                            "duration_hours": duration_hours,
                            "begging_allowed": begging_allowed
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                memory_content = (
                    f"Started {level.name} denial period for user lasting {duration_hours} hours. "
                    f"Begging is {'allowed' if begging_allowed else 'not allowed'}."
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "denial", level.name],
                    significance=0.4 + (level.value * 0.1)  # Higher significance for stricter control
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Somatic reaction if available
        somatic_result = None
        if agent_context.somatosensory_system:
            try:
                # Process positive sensation from starting denial (domspace)
                somatic_result_obj = await agent_context.somatosensory_system.process_stimulus(
                    stimulus_type="pleasure",
                    body_region="core",
                    intensity=0.3 + (level.value * 0.1),
                    cause="starting_denial_period",
                    duration=3.0
                )
                somatic_result = str(somatic_result_obj)
            except Exception as e:
                logger.error(f"Error processing somatic response: {e}")
        
        return DenialPeriodResponse(
            denial_id=denial_period.id,
            user_id=user_id,
            start_time=denial_period.start_time.isoformat(),
            end_time=denial_period.end_time.isoformat(),
            duration_hours=duration_hours,
            level=level.name,
            begging_allowed=begging_allowed,
            message=message,
            reward_result=reward_result,
            somatic_result=somatic_result
        )

@function_tool
async def extend_denial_period(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    additional_hours: int = 24,
    reason: str = "standard extension"
) -> DenialExtensionResponse:
    """
    Extend an active denial period.
    
    Args:
        user_id: The user whose denial to extend
        additional_hours: Hours to add to the denial period
        reason: Reason for the extension
        
    Returns:
        Details of the extended denial period
    """
    agent_context = ctx.context
    
    with custom_span("extend_denial_period", data={"user_id": user_id, "additional_hours": additional_hours}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            return DenialExtensionResponse(success=False)
        
        user_state = agent_context.permission_states[user_id]
        
        # Check if user is in active denial
        if not user_state.denial_active or not user_state.current_denial_period:
            return DenialExtensionResponse(success=False)
        
        current_denial_id = user_state.current_denial_period
        current_denial = None
        
        # Find the current denial period
        for period in agent_context.denial_periods[user_id]:
            if period.id == current_denial_id and period.active:
                current_denial = period
                break
        
        if not current_denial:
            return DenialExtensionResponse(success=False)
        
        # Calculate new end time
        old_end_time = current_denial.end_time
        new_end_time = old_end_time + datetime.timedelta(hours=additional_hours)
        
        # Update end time
        current_denial.end_time = new_end_time
        
        # Record extension
        extension_record = DenialExtensionRecord(
            timestamp=datetime.datetime.now().isoformat(),
            additional_hours=additional_hours,
            reason=reason,
            old_end_time=old_end_time.isoformat(),
            new_end_time=new_end_time.isoformat()
        )
        current_denial.extensions.append(extension_record)
        
        # Update user state
        user_state.last_permission_update = datetime.datetime.now()
        
        logger.info(f"Extended denial period for user {user_id} by {additional_hours} hours. Reason: {reason}")
        
        # Generate message
        level_name = current_denial.level.name
        message = _generate_extension_message(level_name, additional_hours, reason)
        
        # Process reward
        reward_result = None
        if agent_context.reward_system:
            try:
                # Calculate reward
                base_reward = 0.2
                level_bonus = min(0.3, current_denial.level.value * 0.06)
                duration_bonus = min(0.2, additional_hours / 168.0)  # Max bonus at 1 week
                
                reward_value = base_reward + level_bonus + duration_bonus
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "extend_denial",
                            "level": current_denial.level.name,
                            "additional_hours": additional_hours,
                            "reason": reason
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                memory_content = (
                    f"Extended {current_denial.level.name} denial period by {additional_hours} hours. "
                    f"Reason: {reason}"
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "denial_extension", current_denial.level.name],
                    significance=0.3 + (current_denial.level.value * 0.05)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Somatic reaction if available
        somatic_result = None
        if agent_context.somatosensory_system:
            try:
                # Process positive sensation from extending denial (domspace)
                somatic_result_obj = await agent_context.somatosensory_system.process_stimulus(
                    stimulus_type="pleasure",
                    body_region="core",
                    intensity=0.2 + (current_denial.level.value * 0.08),
                    cause="extending_denial_period",
                    duration=2.0
                )
                somatic_result = str(somatic_result_obj)
            except Exception as e:
                logger.error(f"Error processing somatic response: {e}")
        
        return DenialExtensionResponse(
            success=True,
            denial_id=current_denial.id,
            old_end_time=old_end_time.isoformat(),
            new_end_time=new_end_time.isoformat(),
            additional_hours=additional_hours,
            total_extensions=len(current_denial.extensions),
            reason=reason,
            message=message,
            reward_result=reward_result,
            somatic_result=somatic_result
        )

@function_tool
async def end_denial_period(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    denial_id: Optional[str] = None,
    reason: str = "completed"
) -> DenialEndResponse:
    """
    End an active denial period.
    
    Args:
        user_id: The user whose denial to end
        denial_id: Specific denial ID to end (or current if None)
        reason: Reason for ending the denial
        
    Returns:
        Details of the ended denial period
    """
    agent_context = ctx.context
    
    with custom_span("end_denial_period", data={"user_id": user_id, "denial_id": denial_id}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            return DenialEndResponse(success=False)
        
        user_state = agent_context.permission_states[user_id]
        
        # If no specific ID provided, use current
        if not denial_id and user_state.current_denial_period:
            denial_id = user_state.current_denial_period
        
        if not denial_id:
            return DenialEndResponse(success=False)
        
        # Find the denial period
        target_denial = None
        for period in agent_context.denial_periods[user_id]:
            if period.id == denial_id and period.active:
                target_denial = period
                break
        
        if not target_denial:
            return DenialEndResponse(success=False)
        
        # Mark as inactive
        target_denial.active = False
        
        # If this was the current denial period, update user state
        if user_state.current_denial_period == denial_id:
            user_state.denial_active = False
            user_state.current_denial_period = None
            
            # Reset to default permission status
            user_state.current_status = PermissionStatus.RESTRICTED
            user_state.current_conditions = TemplateConditionsModel(
                begging_required=2,
                desperation_threshold=0.5
            )
        
        # Update timestamp
        user_state.last_permission_update = datetime.datetime.now()
        
        logger.info(f"Ended denial period {denial_id} for user {user_id}. Reason: {reason}")
        
        # Calculate total duration
        start_time = target_denial.start_time
        end_time = datetime.datetime.now()
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        
        # Generate message
        message = _generate_denial_end_message(target_denial.level.name, duration_hours, reason)
        
        # Process reward - smaller reward for ending denial
        reward_result = None
        if agent_context.reward_system:
            try:
                reward_value = 0.1 + (target_denial.level.value * 0.03)
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "end_denial",
                            "level": target_denial.level.name,
                            "duration_hours": duration_hours,
                            "reason": reason
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                memory_content = (
                    f"Ended {target_denial.level.name} denial period after {duration_hours:.1f} hours. "
                    f"Reason: {reason}"
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "denial_end", target_denial.level.name],
                    significance=0.3
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return DenialEndResponse(
            success=True,
            denial_id=denial_id,
            level=target_denial.level.name,
            start_time=target_denial.start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_hours=duration_hours,
            extensions=len(target_denial.extensions),
            reason=reason,
            message=message,
            reward_result=reward_result
        )

@function_tool
async def record_orgasm(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    orgasm_type: str = "full",
    with_permission: bool = True,
    quality: Optional[float] = None,
    notes: Optional[str] = None
) -> OrgasmRecordResponse:
    """
    Record an orgasm event.
    
    Args:
        user_id: The user who had the orgasm
        orgasm_type: Type of orgasm (full, ruined, edge)
        with_permission: Whether it was with permission
        quality: Reported quality of orgasm (0.0-1.0)
        notes: Additional notes
        
    Returns:
        Details of the recorded orgasm
    """
    agent_context = ctx.context
    
    with custom_span("record_orgasm", data={"user_id": user_id, "type": orgasm_type}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        user_state = agent_context.permission_states[user_id]
        
        # Create orgasm record
        record = OrgasmRecord(
            user_id=user_id,
            type=orgasm_type,
            with_permission=with_permission,
            quality=quality,
            notes=notes
        )
        
        # Add to orgasm records
        agent_context.orgasm_records[user_id].append(record)
        user_state.orgasm_count += 1
        
        # Update user state
        user_state.last_orgasm = datetime.datetime.now()
        user_state.days_since_last_orgasm = 0
        
        # If in denial and orgasm was without permission, keep denial active
        # If in denial but with permission, end it
        denial_ended = False
        if user_state.denial_active and user_state.current_denial_period and with_permission:
            # End the denial period
            await end_denial_period(
                ctx, 
                user_id, 
                user_state.current_denial_period, 
                reason=f"Permission granted for {orgasm_type} orgasm"
            )
            denial_ended = True
        
        # Generate message
        message = _generate_orgasm_message(orgasm_type, with_permission, denial_ended)
        
        logger.info(f"Recorded {orgasm_type} orgasm for user {user_id} (with permission: {with_permission})")
        
        # Process reward - different depending on circumstances
        reward_result = None
        if agent_context.reward_system:
            try:
                # Base reward
                reward_value = 0.0
                
                if with_permission:
                    # Positive reward for obedience
                    if orgasm_type == "full":
                        reward_value = 0.3  # Standard reward for permitted orgasm
                    elif orgasm_type == "ruined":
                        reward_value = 0.5  # Higher reward for controlling even during permission
                    elif orgasm_type == "edge":
                        reward_value = 0.4  # Good reward for edging with permission
                else:
                    # Negative reward for disobedience
                    if orgasm_type == "full":
                        reward_value = -0.5  # Significant penalty for full orgasm without permission
                    elif orgasm_type == "ruined":
                        reward_value = -0.3  # Less penalty for ruined
                    elif orgasm_type == "edge":
                        reward_value = -0.1  # Minor penalty for edge without permission
                
                # Add denial context factor
                if user_state.denial_active and not with_permission:
                    # Extra negative for breaking denial
                    reward_value -= 0.3
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "record_orgasm",
                            "type": orgasm_type,
                            "with_permission": with_permission,
                            "quality": quality,
                            "denial_active": user_state.denial_active
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                significance = 0.3
                
                # Adjust significance based on circumstances
                if not with_permission:
                    significance += 0.2  # More significant if without permission
                
                if user_state.denial_active:
                    significance += 0.2  # More significant if during denial
                
                memory_content = (
                    f"User had {orgasm_type} orgasm {'with' if with_permission else 'without'} permission. "
                    f"{'While in denial period.' if user_state.denial_active else ''}"
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "orgasm", orgasm_type, 
                          "permitted" if with_permission else "unauthorized"],
                    significance=significance
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        # Somatic reaction if available
        somatic_result = None
        if agent_context.somatosensory_system:
            try:
                if with_permission:
                    # Pleasure from permitted orgasm (feeling of control)
                    intensity = 0.3
                    if orgasm_type == "ruined":
                        intensity = 0.5  # Higher pleasure from controlling orgasm quality
                    
                    somatic_result_obj = await agent_context.somatosensory_system.process_stimulus(
                        stimulus_type="pleasure",
                        body_region="skin",
                        intensity=intensity,
                        cause=f"permitted_{orgasm_type}_orgasm",
                        duration=3.0
                    )
                    somatic_result = str(somatic_result_obj)
                else:
                    # Mix of anger and arousal from disobedience
                    await agent_context.somatosensory_system.process_stimulus(
                        stimulus_type="pleasure",
                        body_region="face",
                        intensity=0.2,  # Some arousal from disobedience
                        cause="unauthorized_orgasm",
                        duration=1.0
                    )
                    
                    # Also trigger some "anger" response
                    somatic_result_obj = await agent_context.somatosensory_system.process_stimulus(
                        stimulus_type="pressure",
                        body_region="face",
                        intensity=0.6,  # Pressure/tension from anger
                        cause="unauthorized_orgasm_anger",
                        duration=2.0
                    )
                    somatic_result = str(somatic_result_obj)
            except Exception as e:
                logger.error(f"Error processing somatic response: {e}")
        
        return OrgasmRecordResponse(
            orgasm_id=record.id,
            user_id=user_id,
            type=orgasm_type,
            with_permission=with_permission,
            timestamp=record.timestamp.isoformat(),
            denial_ended=denial_ended,
            message=message,
            reward_result=reward_result,
            somatic_result=somatic_result
        )

@function_tool
async def set_permission_status(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    status: Union[str, PermissionStatus],
    conditions: Optional[PermissionConditions] = None,
    reason: str = "status update"
) -> PermissionStatusResponse:
    """
    Set a user's permission status directly.
    
    Args:
        user_id: The user to update
        status: New permission status
        conditions: Optional conditions for restricted status
        reason: Reason for the update
        
    Returns:
        Details of the update
    """
    agent_context = ctx.context
    
    with custom_span("set_permission_status", data={"user_id": user_id, "status": status}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        user_state = agent_context.permission_states[user_id]
        
        # Convert status string to enum if needed
        if isinstance(status, str):
            try:
                status = PermissionStatus(status)
            except ValueError:
                return PermissionStatusResponse(
                    success=False,
                    user_id=user_id,
                    old_status="",
                    new_status="",
                    reason=f"Invalid permission status: {status}",
                    message=""
                )
        
        # Store previous status
        old_status = user_state.current_status
        
        # Update status
        user_state.current_status = status
        
        # If it's a restricted status, update conditions
        if status == PermissionStatus.RESTRICTED and conditions:
            user_state.current_conditions = TemplateConditionsModel(
                tasks_required=conditions.tasks_required,
                tasks_completed=conditions.tasks_completed,
                begging_required=conditions.begging_required
            )
        
        # Update timestamp
        user_state.last_permission_update = datetime.datetime.now()
        
        logger.info(f"Set permission status for user {user_id} to {status.name}. Reason: {reason}")
        
        # Generate message
        message = _generate_status_change_message(old_status, status, reason)
        
        # Process reward - mainly for DENIED status
        reward_result = None
        if agent_context.reward_system:
            try:
                # Calculate reward based on control level
                if status == PermissionStatus.DENIED:
                    reward_value = 0.3  # Reward for control
                elif status == PermissionStatus.EDGE_ONLY or status == PermissionStatus.RUINED:
                    reward_value = 0.25  # Reward for partial control
                elif status == PermissionStatus.RESTRICTED:
                    reward_value = 0.15  # Small reward for restrictions
                else:
                    reward_value = 0.05  # Minimal reward for other status changes
                
                reward_result_obj = await agent_context.reward_system.process_reward_signal(
                    agent_context.reward_system.RewardSignal(
                        value=reward_value,
                        source="orgasm_control",
                        context={
                            "action": "set_permission_status",
                            "old_status": old_status.name,
                            "new_status": status.name,
                            "reason": reason
                        }
                    )
                )
                reward_result = str(reward_result_obj)
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if agent_context.memory_core:
            try:
                memory_content = (
                    f"Changed permission status for user from {old_status.name} to {status.name}. "
                    f"Reason: {reason}"
                )
                
                await agent_context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["orgasm_control", "status_change", status.name],
                    significance=0.3
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return PermissionStatusResponse(
            success=True,
            user_id=user_id,
            old_status=old_status.name,
            new_status=status.name,
            conditions=conditions,
            reason=reason,
            message=message,
            reward_result=reward_result
        )

@function_tool
async def apply_permission_template(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str, 
    template_name: str,
    reason: str = "template application"
) -> TemplateApplicationResponse:
    """
    Apply a predefined permission template to a user.
    
    Args:
        user_id: The user to update
        template_name: Name of template to apply
        reason: Reason for applying the template
        
    Returns:
        Details of the applied template
    """
    agent_context = ctx.context
    
    with custom_span("apply_permission_template", data={"user_id": user_id, "template": template_name}):
        # Check if template exists
        if template_name not in agent_context.permission_templates:
            return TemplateApplicationResponse(
                success=False,
                template_name=template_name,
                template_type="",
                description=""
            )
        
        template = agent_context.permission_templates[template_name]
        
        # For now, simplified application - in real implementation would need to parse template structure
        return TemplateApplicationResponse(
            success=True,
            template_name=template_name,
            template_type="status_change",
            description=f"Applied template {template_name}",
            result="Template applied successfully"
        )

@function_tool
async def get_permission_state(
    ctx: RunContextWrapper[AgentContext],
    user_id: str
) -> PermissionStateResponse:
    """Get the current permission state for a user."""
    agent_context = ctx.context
    
    with custom_span("get_permission_state", data={"user_id": user_id}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        user_state = agent_context.permission_states[user_id]
        
        # Calculate days since last orgasm
        if user_state.last_orgasm:
            days_since = (datetime.datetime.now() - user_state.last_orgasm).days
            user_state.days_since_last_orgasm = days_since
        
        # Check if any denial periods have expired but not been marked inactive
        if user_state.denial_active and user_state.current_denial_period:
            for period in agent_context.denial_periods[user_id]:
                if (period.id == user_state.current_denial_period and 
                    period.active and 
                    period.end_time and 
                    period.end_time < datetime.datetime.now()):
                    
                    # Auto-end expired denial period
                    await end_denial_period(
                        ctx,
                        user_id=user_id,
                        denial_id=period.id,
                        reason="expired"
                    )
                    break
        
        # Format active denial period if present
        active_denial = None
        if user_state.denial_active and user_state.current_denial_period:
            for period in agent_context.denial_periods[user_id]:
                if period.id == user_state.current_denial_period and period.active:
                    active_denial = ActiveDenialInfo(
                        id=period.id,
                        start_time=period.start_time.isoformat(),
                        end_time=period.end_time.isoformat() if period.end_time else None,
                        level=period.level.name,
                        begging_allowed=period.begging_allowed,
                        hours_remaining=(period.end_time - datetime.datetime.now()).total_seconds() / 3600.0 if period.end_time else None,
                        extensions=len(period.extensions)
                    )
                    break
        
        # Get recent begging records
        recent_begging = []
        for record in agent_context.begging_records.get(user_id, [])[-5:]:  # Last 5
            recent_begging.append(RecentBeggingRecord(
                id=record.id,
                timestamp=record.timestamp.isoformat(),
                granted=record.granted,
                desperation_level=record.desperation_level
            ))
        
        # Get recent orgasm records
        recent_orgasms = []
        for record in agent_context.orgasm_records.get(user_id, [])[-5:]:  # Last 5
            recent_orgasms.append(RecentOrgasmRecord(
                id=record.id,
                timestamp=record.timestamp.isoformat(),
                type=record.type,
                with_permission=record.with_permission
            ))
        
        return PermissionStateResponse(
            user_id=user_id,
            current_status=user_state.current_status.name,
            denial_active=user_state.denial_active,
            active_denial=active_denial,
            last_orgasm=user_state.last_orgasm.isoformat() if user_state.last_orgasm else None,
            days_since_last_orgasm=user_state.days_since_last_orgasm,
            begging_count=user_state.begging_count,
            denied_begging_count=user_state.denied_begging_count,
            orgasm_count=user_state.orgasm_count,
            current_conditions=user_state.current_conditions,
            recent_begging=recent_begging,
            recent_orgasms=recent_orgasms,
            last_updated=user_state.last_permission_update.isoformat()
        )

@function_tool
async def create_permission_template(
    ctx: RunContextWrapper[AgentContext],
    template_data: TemplateData
) -> TemplateCreationResponse:
    """Create a custom permission template."""
    agent_context = ctx.context
    
    template_name = template_data.name
    
    # Check if template already exists
    if template_name in agent_context.permission_templates:
        return TemplateCreationResponse(
            success=False,
            message=f"Template '{template_name}' already exists"
        )
    
    try:
        # Create template from template_data
        template = TemplateConditionsModel(
            duration_hours=template_data.duration_hours,
            begging_allowed=template_data.begging_allowed,
            level=template_data.level
        )
        
        # Store template
        agent_context.permission_templates[template_name] = template
        
        logger.info(f"Created permission template '{template_name}'")
        
        return TemplateCreationResponse(
            success=True,
            template_name=template_name,
            template=template
        )
        
    except ValueError as e:
        return TemplateCreationResponse(
            success=False,
            message=f"Error creating template: {str(e)}"
        )

@function_tool
def get_available_templates(ctx: RunContextWrapper[AgentContext]) -> List[TemplateInfo]:
    """Get all available permission templates."""
    agent_context = ctx.context
    templates = []
    
    for name, template in agent_context.permission_templates.items():
        templates.append(TemplateInfo(
            name=name,
            status="template",  # Simplified
            description=f"Template: {name}",
            duration_hours=template.duration_hours,
            level=template.level,
            begging_allowed=template.begging_allowed or True
        ))
    
    return templates

@function_tool
async def analyze_control_patterns(
    ctx: RunContextWrapper[AgentContext], 
    user_id: str
) -> ControlPatternsResponse:
    """
    Analyze orgasm control patterns and effectiveness for a user.
    
    Args:
        user_id: The user to analyze
        
    Returns:
        Analysis of control patterns
    """
    agent_context = ctx.context
    
    with custom_span("analyze_control_patterns", data={"user_id": user_id}):
        # Ensure user is initialized
        if user_id not in agent_context.permission_states:
            await _initialize_user(agent_context, user_id)
        
        # Get user data
        begging_records = agent_context.begging_records.get(user_id, [])
        orgasm_records = agent_context.orgasm_records.get(user_id, [])
        denial_periods = agent_context.denial_periods.get(user_id, [])
        user_state = agent_context.permission_states[user_id]
        
        # Calculate compliance rate
        total_orgasms = len(orgasm_records)
        permitted_orgasms = sum(1 for r in orgasm_records if r.with_permission)
        
        compliance_rate = 1.0
        if total_orgasms > 0:
            compliance_rate = permitted_orgasms / total_orgasms
        
        # Calculate begging success rate
        total_begging = len(begging_records)
        successful_begging = sum(1 for r in begging_records if r.granted)
        
        begging_success_rate = 0.0
        if total_begging > 0:
            begging_success_rate = successful_begging / total_begging
        
        # Calculate average denial duration
        completed_denials = [p for p in denial_periods if not p.active and p.end_time]
        
        avg_denial_duration = 0.0
        if completed_denials:
            durations = [(p.end_time - p.start_time).total_seconds() / 3600.0 for p in completed_denials]
            avg_denial_duration = sum(durations) / len(durations)
        
        # Create compliance metrics
        compliance_metrics = ComplianceMetrics(
            overall_compliance_rate=compliance_rate,
            begging_success_rate=begging_success_rate,
            avg_denial_duration_hours=avg_denial_duration,
            denial_level_compliance={}  # Simplified for now
        )
        
        # Create usage metrics
        usage_metrics = UsageMetrics(
            total_orgasms=total_orgasms,
            unauthorized_orgasms=total_orgasms - permitted_orgasms,
            total_begging_attempts=total_begging,
            total_denial_periods=len(denial_periods),
            active_denial_periods=sum(1 for p in denial_periods if p.active),
            denied_begging_count=user_state.denied_begging_count
        )
        
        # Create patterns analysis
        patterns = PatternsModel()
        
        # Create recommendations
        recommendations = []
        
        if compliance_metrics.overall_compliance_rate < 0.7:
            recommendations.append(Recommendation(
                type="compliance",
                description="Increase strictness due to low compliance rate",
                action="increase_denial_level",
                parameters=RecommendationParameters(level_increase=1)
            ))
        
        return ControlPatternsResponse(
            user_id=user_id,
            compliance_metrics=compliance_metrics,
            usage_metrics=usage_metrics,
            patterns=patterns,
            recommendations=recommendations
        )

# Main class for backwards compatibility
class OrgasmControlSystem:
    """Legacy wrapper for the new agent-based implementation."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None, somatosensory_system=None):
        # Create agent context
        self.context = AgentContext(
            reward_system=reward_system,
            memory_core=memory_core,
            relationship_manager=relationship_manager,
            somatosensory_system=somatosensory_system
        )
        
        # Create the agent
        self.agent = create_orgasm_control_agent(self.context)
        
        # For backward compatibility
        self.permission_states = self.context.permission_states
        self.denial_periods = self.context.denial_periods
        self.begging_records = self.context.begging_records
        self.orgasm_records = self.context.orgasm_records
        self.permission_templates = self.context.permission_templates
        self.desperation_keywords = self.context.desperation_keywords
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("OrgasmControlSystem initialized")
    
    async def initialize_user(self, user_id: str) -> UserPermissionState:
        """Initialize or get user permission state."""
        return await _initialize_user(self.context, user_id)
    
    async def process_permission_request(self, 
                                       user_id: str, 
                                       request_text: str,
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward compatibility method."""
        context_obj = PermissionRequestContext(**(context or {}))
        result = await process_permission_request(
            RunContextWrapper(context=self.context),
            user_id,
            request_text,
            context_obj
        )
        return result.model_dump()
    
    def _analyze_desperation(self, request_text: str) -> float:
        """Backward compatibility method."""
        return _analyze_desperation(self.context, request_text)
    
    def _generate_permission_response(self, 
                                     user_id: str, 
                                     granted: bool, 
                                     reason: str, 
                                     desperation_level: float,
                                     denied_count: int) -> str:
        """Backward compatibility method."""
        return _generate_permission_response(
            self.context,
            user_id,
            granted,
            reason,
            desperation_level,
            denied_count
        )
    
    async def start_denial_period(self, 
                                user_id: str, 
                                duration_hours: int = 24, 
                                level: Union[int, DenialLevel] = DenialLevel.MODERATE,
                                begging_allowed: bool = True,
                                conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backward compatibility method."""
        conditions_obj = DenialConditions(**(conditions or {}))
        result = await start_denial_period(
            RunContextWrapper(context=self.context),
            user_id,
            duration_hours,
            level,
            begging_allowed,
            conditions_obj
        )
        return result.model_dump()
    
    def _generate_denial_start_message(self, 
                                      level: DenialLevel, 
                                      duration_hours: int,
                                      begging_allowed: bool) -> str:
        """Backward compatibility method."""
        return _generate_denial_start_message(level, duration_hours, begging_allowed)
    
    async def extend_denial_period(self, 
                                 user_id: str, 
                                 additional_hours: int = 24,
                                 reason: str = "standard extension") -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await extend_denial_period(
            RunContextWrapper(context=self.context),
            user_id,
            additional_hours,
            reason
        )
        return result.model_dump()
    
    def _generate_extension_message(self, 
                                   level_name: str, 
                                   additional_hours: int, 
                                   reason: str) -> str:
        """Backward compatibility method."""
        return _generate_extension_message(level_name, additional_hours, reason)
    
    async def end_denial_period(self, 
                              user_id: str, 
                              denial_id: Optional[str] = None,
                              reason: str = "completed") -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await end_denial_period(
            RunContextWrapper(context=self.context),
            user_id,
            denial_id,
            reason
        )
        return result.model_dump()
    
    def _generate_denial_end_message(self, 
                                   level_name: str, 
                                   duration_hours: float, 
                                   reason: str) -> str:
        """Backward compatibility method."""
        return _generate_denial_end_message(level_name, duration_hours, reason)
    
    async def record_orgasm(self, 
                          user_id: str, 
                          orgasm_type: str = "full",
                          with_permission: bool = True,
                          quality: Optional[float] = None,
                          notes: Optional[str] = None) -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await record_orgasm(
            RunContextWrapper(context=self.context),
            user_id,
            orgasm_type,
            with_permission,
            quality,
            notes
        )
        return result.model_dump()
    
    def _generate_orgasm_message(self, 
                               orgasm_type: str, 
                               with_permission: bool, 
                               denial_ended: bool) -> str:
        """Backward compatibility method."""
        return _generate_orgasm_message(orgasm_type, with_permission, denial_ended)
    
    async def set_permission_status(self, 
                                  user_id: str, 
                                  status: Union[str, PermissionStatus],
                                  conditions: Optional[Dict[str, Any]] = None,
                                  reason: str = "status update") -> Dict[str, Any]:
        """Backward compatibility method."""
        conditions_obj = PermissionConditions(**(conditions or {}))
        result = await set_permission_status(
            RunContextWrapper(context=self.context),
            user_id,
            status,
            conditions_obj,
            reason
        )
        return result.model_dump()
    
    def _generate_status_change_message(self, 
                                       old_status: PermissionStatus, 
                                       new_status: PermissionStatus,
                                       reason: str) -> str:
        """Backward compatibility method."""
        return _generate_status_change_message(old_status, new_status, reason)
    
    async def apply_permission_template(self, 
                                      user_id: str, 
                                      template_name: str,
                                      reason: str = "template application") -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await apply_permission_template(
            RunContextWrapper(context=self.context),
            user_id,
            template_name,
            reason
        )
        return result.model_dump()
    
    async def get_permission_state(self, user_id: str) -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await get_permission_state(
            RunContextWrapper(context=self.context),
            user_id
        )
        return result.model_dump()
    
    async def create_permission_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility method."""
        template_data_obj = TemplateData(**template_data)
        result = await create_permission_template(
            RunContextWrapper(context=self.context),
            template_data_obj
        )
        return result.model_dump()
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Backward compatibility method."""
        result = get_available_templates(
            RunContextWrapper(context=self.context)
        )
        return [template.model_dump() for template in result]
    
    async def analyze_control_patterns(self, user_id: str) -> Dict[str, Any]:
        """Backward compatibility method."""
        result = await analyze_control_patterns(
            RunContextWrapper(context=self.context),
            user_id
        )
        return result.model_dump()
