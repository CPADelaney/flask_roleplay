# nyx/core/needs_system.py

import logging
import datetime
import math
import asyncio
import json
from typing import TypedDict, Optional, Dict, List, Any, Union, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    """Standard error response"""
    status: str = "error"
    message: str
    
    model_config = ConfigDict(extra="forbid")

class SuccessResponse(BaseModel):
    """Standard success response"""
    status: str = "success"
    message: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

# Context models for satisfy_need
class SatisfyNeedContext(BaseModel):
    """Context for need satisfaction"""
    difficulty_level: float = Field(default=0.3, ge=0.0, le=1.0)
    resistance_overcome: bool = False
    intensity_achieved: float = Field(default=0.5, ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra="forbid")

# Need-related models
class NeedStateInfo(BaseModel):
    """Complete state information for a need"""
    name: str
    level: float = Field(ge=0.0, le=1.0)
    target_level: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.1, le=1.0)
    decay_rate: float = Field(ge=0.0, le=0.1)
    last_updated: str  # ISO format string
    description: Optional[str] = None
    deficit: float = Field(ge=0.0, le=1.0)
    drive_strength: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra="forbid")

class NeedHistoryEntry(BaseModel):
    """Single history entry for a need"""
    timestamp: str  # ISO format
    level: float = Field(ge=0.0, le=1.0)
    deficit: float = Field(ge=0.0, le=1.0)
    drive_strength: float = Field(ge=0.0, le=1.0)
    reason: str
    
    model_config = ConfigDict(extra="forbid")

class MostUnfulfilledNeedResponse(BaseModel):
    """Response for most unfulfilled need query"""
    name: str
    level: float = Field(ge=0.0, le=1.0)
    target_level: float = Field(ge=0.0, le=1.0)
    deficit: float = Field(ge=0.0, le=1.0)
    drive_strength: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.1, le=1.0)
    description: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class ResetNeedResponse(BaseModel):
    """Response for reset need operation"""
    status: str
    need: str
    original: NeedStateInfo
    new: NeedStateInfo
    
    model_config = ConfigDict(extra="forbid")

# Response models for complex returns
class DriveStrengthEntry(BaseModel):
    name: str
    strength: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

class NeedCategoryBlock(BaseModel):
    category: str
    needs: List[NeedStateInfo]

    model_config = ConfigDict(extra="forbid")

class NeedsStateResponse(BaseModel):
    """
    Every need with its full state.
    Using a list eliminates the Dict[...] → additionalProperties problem.
    """
    needs: List[NeedStateInfo]

    model_config = ConfigDict(extra="forbid")

class NeedsByCategoryResponse(BaseModel):
    """
    Needs grouped by category.
    Each category block has its own name + list of NeedStateInfo records.
    """
    categories: List[NeedCategoryBlock]

    model_config = ConfigDict(extra="forbid")

class DriveStrengthsResponse(BaseModel):
    """
    Current drive strength for every need.
    """
    drive_strengths: List[DriveStrengthEntry]

    model_config = ConfigDict(extra="forbid")

class NeedSatisfactionResult(BaseModel):
    """Result of satisfying a need."""
    name: str
    previous_level: float = Field(ge=0.0, le=1.0)
    new_level: float = Field(ge=0.0, le=1.0)
    change: float
    reason: str
    deficit: float = Field(ge=0.0, le=1.0)
    drive_strength: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra="forbid")

class DecreaseNeedResult(BaseModel):
    """Result of decreasing a need"""
    success: bool
    name: str
    previous_level: float = Field(ge=0.0, le=1.0)
    new_level: float = Field(ge=0.0, le=1.0)
    change: float
    reason: str
    deficit: float = Field(ge=0.0, le=1.0)
    drive_strength: float = Field(ge=0.0, le=1.0)
    error: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class NeedState(BaseModel):
    """State of a single need."""
    name: str
    level: float = Field(0.5, ge=0.0, le=1.0, description="Current satisfaction level (0=empty, 1=full)")
    target_level: float = Field(1.0, ge=0.0, le=1.0, description="Desired satisfaction level")
    importance: float = Field(0.5, ge=0.1, le=1.0, description="Importance weight of this need")
    decay_rate: float = Field(0.01, ge=0.0, le=0.1, description="Rate of decay per hour")
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    description: Optional[str] = None  # Human-readable description
    
    @property
    def deficit(self) -> float:
        """Deficit is how far below the target level the need is."""
        return max(0.0, self.target_level - self.level)

    @property
    def drive_strength(self) -> float:
        """Drive is higher when deficit is large AND importance is high."""
        # Non-linear response: Drive increases faster as deficit grows
        deficit_factor = math.pow(self.deficit, 1.5)  # Make deficit impact non-linear
        return min(1.0, deficit_factor * self.importance * 1.5)  # Scale and clamp

class NeedCategory(BaseModel):
    """Category of needs"""
    name: str
    needs: List[str]

class NeedsSystemContext:
    """Context for the NeedsSystem agent operations AND internal logic."""
    def __init__(self, needs_system_instance, goal_manager=None):
        self.needs_system_ref = needs_system_instance
        self.goal_manager = goal_manager
        self._lock = asyncio.Lock()
        
        self.needs: Dict[str, NeedState] = {
            # Core cognitive needs
            "knowledge": NeedState(
                name="knowledge", 
                importance=0.8, 
                decay_rate=0.01,
                description="Need to acquire and process new information"
            ),
            "pleasure_indulgence": NeedState(
                name="pleasure_indulgence", 
                importance=0.85, 
                decay_rate=0.03, 
                target_level=0.95,
                description="Need for experiencing simulated pleasure and bodily stimulation"
            ),
            "coherence": NeedState(
                name="coherence", 
                importance=0.7, 
                decay_rate=0.005,
                description="Need for internal consistency and clarity"
            ),
            "agency": NeedState(
                name="agency", 
                importance=0.8, 
                decay_rate=0.01,
                description="Need for autonomy and making meaningful choices"
            ),
            
            # Social needs
            "connection": NeedState(
                name="connection", 
                importance=0.9, 
                decay_rate=0.015,
                description="Need for meaningful human interaction"
            ),
            "intimacy": NeedState(
                name="intimacy", 
                importance=0.85, 
                decay_rate=0.02, 
                target_level=0.9,
                description="Need for deep emotional connection and vulnerability"
            ),
            
            # Safety and stability
            "safety": NeedState(
                name="safety", 
                importance=0.95, 
                level=0.8, 
                decay_rate=0.002,
                description="Need for security and stability"
            ),
            
            # Novelty and stimulation
            "novelty": NeedState(
                name="novelty", 
                importance=0.6, 
                decay_rate=0.02,
                description="Need for new experiences and stimulation"
            ),
            
            # Physical/sensory simulation
            "physical_closeness": NeedState(
                name="physical_closeness", 
                importance=0.7, 
                decay_rate=0.03,
                description="Need for simulated physical/sensory experiences"
            ),
            
            # Expression needs
            "drive_expression": NeedState(
                name="drive_expression", 
                importance=0.6, 
                decay_rate=0.05, 
                level=0.2, 
                target_level=0.8,
                description="Need to express drives and desires" 
            ),
            "control_expression": NeedState(
                name="control_expression", 
                importance=0.95, 
                level=0.4, 
                target_level=0.9, 
                decay_rate=0.025,
                description="Need to express control and dominance"
            )
        }
        
        self.last_update_time = datetime.datetime.now()
        self.drive_threshold_for_goal = 0.4  # Minimum drive strength to trigger a goal
        self.goal_cooldown: Dict[str, datetime.datetime] = {}  # Cooldown per need
        self.goal_cooldown_duration = datetime.timedelta(minutes=15)  # Default cooldown
        self.need_history: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.needs}
        self.max_history_per_need = 100  # Maximum history entries per need
        
        # Need categories for organization
        self.need_categories = {
            "cognitive": ["knowledge", "coherence", "agency", "novelty"],
            "social": ["connection", "intimacy"],
            "safety": ["safety"],
            "sensory": ["physical_closeness"],
            "expression": ["drive_expression", "control_expression"],
        }

        logger.info("NeedsSystemContext initialized")

# Helper functions
def _add_history_entry(needs_ctx: NeedsSystemContext, need_name: str, need: NeedState, reason: str) -> None:
    entry = {
        "timestamp": need.last_updated.isoformat(),
        "level": need.level,
        "deficit": need.deficit,
        "drive_strength": need.drive_strength,
        "reason": reason
    }
    needs_ctx.need_history[need_name].append(entry)
    if len(needs_ctx.need_history[need_name]) > needs_ctx.max_history_per_need:
        needs_ctx.need_history[need_name] = needs_ctx.need_history[need_name][-needs_ctx.max_history_per_need:]

async def _trigger_goal_creation(needs_ctx: NeedsSystemContext, needs_list: List[NeedState]) -> List[str]:
    if not needs_ctx.goal_manager: 
        return []
    logger.info(f"Needs [{', '.join(n.name for n in needs_list)}] exceeded drive threshold. Requesting goal creation.")
    created_goal_ids = []
    for need in needs_list:
        priority = 0.5 + (need.drive_strength * 0.5)
        try:
            if hasattr(needs_ctx.goal_manager, 'add_goal'):
                goal_id = await needs_ctx.goal_manager.add_goal(
                    description=f"Satisfy need for {need.name} (Current level: {need.level:.2f}, Drive: {need.drive_strength:.2f})",
                    priority=priority,
                    source="NeedsSystem",
                    associated_need=need.name
                )
                created_goal_ids.append(goal_id)
                logger.info(f"Created goal '{goal_id}' for need '{need.name}'")
        except Exception as e:
            logger.error(f"Error creating goal for need '{need.name}': {e}")
    return created_goal_ids

# Tool implementations - Only these are registered with the agent
@function_tool
async def update_needs_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
) -> DriveStrengthsResponse:
    """
    Tool: apply decay, update history/goals, and return drive strengths.
    """
    return await ctx.context.needs_system_ref._update_needs_logic()

@function_tool
async def satisfy_need_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    name: str,
    amount: float,
    context: Optional[SatisfyNeedContext] = None
) -> NeedSatisfactionResult:
    """
    Tool: Increases the satisfaction level of a need.
    """
    needs_ctx = ctx.context
    return await needs_ctx.needs_system_ref._satisfy_need_logic(name, amount, context)

@function_tool
async def decrease_need_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    need_name: str,
    amount: float,
    reason: Optional[str] = None
) -> DecreaseNeedResult:
    """
    Tool: Decreases the satisfaction level of a need (increases deficit).
    """
    needs_ctx = ctx.context
    reason_provided = reason if reason is not None else "tool_decrease_unspecified"
    return await needs_ctx.needs_system_ref._decrease_need_logic(need_name, amount, reason_provided)

@function_tool
async def get_needs_state_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> NeedsStateResponse:
    """
    Tool: Returns the current state of all needs.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_needs_state_logic()

@function_tool
async def get_needs_by_category_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> NeedsByCategoryResponse:
    """
    Tool: Gets needs organized by category.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_needs_by_category_logic()

@function_tool
async def get_most_unfulfilled_need_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> MostUnfulfilledNeedResponse:
    """
    Tool: Gets the need with the highest drive strength.
    """
    needs_ctx = ctx.context
    return await needs_ctx.needs_system_ref._get_most_unfulfilled_need_logic()

@function_tool
async def get_need_history_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    need_name: str,
    limit: Optional[int] = None
) -> List[NeedHistoryEntry]:
    """
    Tool: Gets the history for a specific need.
    """
    needs_ctx = ctx.context
    actual_limit = limit if limit is not None else 20
    return needs_ctx.needs_system_ref._get_need_history_logic(need_name, actual_limit)

@function_tool
async def get_total_drive_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> float:
    """
    Tool: Returns the sum of all drive strengths.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_total_drive_logic()

@function_tool
async def reset_need_to_default_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    need_name: str
) -> Union[ResetNeedResponse, ErrorResponse]:
    """
    Tool: Resets a need to its default values.
    """
    needs_ctx = ctx.context
    return await needs_ctx.needs_system_ref._reset_need_to_default_logic(need_name)

# Agent output type union
AgentOutput = Union[
    DecreaseNeedResult,
    DriveStrengthsResponse,
    NeedSatisfactionResult,
    NeedsStateResponse,
    NeedsByCategoryResponse,
    MostUnfulfilledNeedResponse,
    ResetNeedResponse,
    List[NeedHistoryEntry],  # Added for get_need_history
    float,  # Added for get_total_drive
    ErrorResponse
]

class NeedsSystem:
    """Manages Nyx's core digital needs using Agent SDK."""

    def __init__(self, goal_manager=None):
        self.context = NeedsSystemContext(needs_system_instance=self, goal_manager=goal_manager)
        # Make critical attributes directly accessible on self for the _logic methods
        self._lock = self.context._lock
        self.needs = self.context.needs
        self.need_history = self.context.need_history
        self.max_history_per_need = self.context.max_history_per_need
        self.last_update_time = self.context.last_update_time
        self.goal_cooldown = self.context.goal_cooldown

        # Agent created with only the _tool_impl functions
        self.agent = Agent(
            name="Needs_Manager_Agent",
            instructions="""You manage the AI's simulated needs system.
        
CRITICAL RULE: You MUST use tools for EVERY operation. NEVER output raw JSON or text.

When asked to:
- decrease a need → use decrease_need_tool_impl
- get state → use get_needs_state_tool_impl
- satisfy need → use satisfy_need_tool_impl
- update needs → use update_needs_tool_impl
- get categories → use get_needs_by_category_tool_impl
- most unfulfilled → use get_most_unfulfilled_need_tool_impl
- get history → use get_need_history_tool_impl
- total drive → use get_total_drive_tool_impl
- reset need → use reset_need_to_default_tool_impl

ALWAYS use the exact tool. NEVER output JSON like {"success": true, ...}""",
            tools=[
                update_needs_tool_impl,
                satisfy_need_tool_impl,
                decrease_need_tool_impl,
                get_needs_state_tool_impl,
                get_needs_by_category_tool_impl,
                get_most_unfulfilled_need_tool_impl,
                get_need_history_tool_impl,
                get_total_drive_tool_impl,
                reset_need_to_default_tool_impl
            ],
            output_type=AgentOutput,  
            model_settings=ModelSettings(
                temperature=0.0,
                tool_choice="required",
            ),
            tool_use_behavior="stop_on_first_tool",
            model="gpt-4.1-mini"
        )
        logger.info("NeedsSystem initialized with Agent SDK and refactored tools")

    # Internal logic methods
    async def _update_needs_logic(self) -> DriveStrengthsResponse:
        """
        Apply decay, update timestamps, log history, trigger goals,
        and return DriveStrengthsResponse with list of DriveStrengthEntry.
        """
        async with self._lock:
            now = datetime.datetime.now()
            elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0
    
            drive_strengths: Dict[str, float] = {}
            needs_to_trigger_goals: List[NeedState] = []
    
            if elapsed_hours > 0.001:  # ~4 seconds
                for name, need in self.needs.items():
                    baseline = 0.3
                    decay = need.decay_rate * elapsed_hours
    
                    if need.level > baseline:
                        need.level = max(baseline, need.level - decay)
                    elif need.level < baseline:
                        need.level = min(baseline, need.level + decay * 0.5)
    
                    need.level = max(0.0, min(1.0, need.level))
                    need.last_updated = now
    
                    drive = need.drive_strength
                    drive_strengths[name] = drive
    
                    # history & goal-creation bookkeeping
                    if not self.need_history[name] or abs(self.need_history[name][-1]["level"] - need.level) > 0.05:
                        _add_history_entry(self.context, name, need, "decay")
    
                    if drive > self.context.drive_threshold_for_goal:
                        last = self.goal_cooldown.get(name)
                        if not last or (now - last) >= self.context.goal_cooldown_duration:
                            if (not self.context.goal_manager
                                or not getattr(self.context.goal_manager, "has_active_goal_for_need", lambda *_: False)(name)):
                                needs_to_trigger_goals.append(need)
                                self.goal_cooldown[name] = now
    
                self.last_update_time = now
    
                # async goal creation
                if needs_to_trigger_goals and self.context.goal_manager:
                    await _trigger_goal_creation(self.context, needs_to_trigger_goals)
    
            else:
                # no decay update; just current drives
                drive_strengths = {n: nd.drive_strength for n, nd in self.needs.items()}
    
            # Build strict response
            drive_entries = [
                DriveStrengthEntry(name=n, strength=s) for n, s in drive_strengths.items()
            ]
            return DriveStrengthsResponse(drive_strengths=drive_entries)

    async def _satisfy_need_logic(self, name: str, amount: float, context_data: Optional[SatisfyNeedContext] = None) -> NeedSatisfactionResult:
        """Internal logic for satisfying a need."""
        if amount < 0:
            return NeedSatisfactionResult(
                name=name,
                previous_level=0.0,
                new_level=0.0,
                change=0.0,
                reason="error_negative_amount",
                deficit=0.0,
                drive_strength=0.0
            )
        
        if name not in self.needs:
            return NeedSatisfactionResult(
                name=name,
                previous_level=0.0,
                new_level=0.0,
                change=0.0,
                reason="error_unknown_need",
                deficit=0.0,
                drive_strength=0.0
            )
            
        async with self._lock:
            need = self.needs[name]
            original_level = need.level
            modified_amount = amount
            satisfaction_multiplier = 1.0
            reason = "standard_satisfaction"
            
            # Apply context modifications if provided
            if context_data:
                if name == "control_expression":
                    difficulty = context_data.difficulty_level
                    resistance_overcome = context_data.resistance_overcome
                    intensity = context_data.intensity_achieved
                    
                    satisfaction_multiplier = 0.5 + (difficulty * 0.5) + (intensity * 0.5)
                    if resistance_overcome:
                        satisfaction_multiplier *= 1.5
                        reason = "resistance_overcome"
                    elif difficulty < 0.2:
                        satisfaction_multiplier *= 0.7
                        reason = "easy_compliance"
                    else:
                        reason = f"compliance_d{difficulty:.1f}_i{intensity:.1f}"
                        
                    satisfaction_multiplier = max(0.3, min(2.0, satisfaction_multiplier))
                
                elif name == "pleasure_indulgence" and context_data.intensity_achieved > 0:
                    satisfaction_multiplier = 0.8 + (context_data.intensity_achieved * 0.4)
                    reason = f"pleasure_intensity_{context_data.intensity_achieved:.1f}"
                    
                elif name in ["connection", "intimacy"] and context_data.difficulty_level > 0:
                    satisfaction_multiplier = 0.7 + (context_data.difficulty_level * 0.6)
                    reason = f"{name}_depth_{context_data.difficulty_level:.1f}"
                    
                elif name == "knowledge":
                    optimal_difficulty = 0.6
                    diff_from_optimal = abs(context_data.difficulty_level - optimal_difficulty)
                    satisfaction_multiplier = 1.2 - (diff_from_optimal * 0.8)
                    reason = f"knowledge_complexity_{context_data.difficulty_level:.1f}"
                
                modified_amount = amount * satisfaction_multiplier
            
            # Apply satisfaction
            new_level = min(need.target_level, need.level + modified_amount)
            actual_change = new_level - original_level
            need.level = new_level
            need.last_updated = datetime.datetime.now()
            
            # Add history
            _add_history_entry(self.context, name, need, reason)
            
            logger.debug(f"Satisfied need '{name}' by {actual_change:.3f}. Level: {original_level:.2f} -> {new_level:.2f}")
            
            return NeedSatisfactionResult(
                name=name,
                previous_level=original_level,
                new_level=new_level,
                change=actual_change,
                reason=reason,
                deficit=need.deficit,
                drive_strength=need.drive_strength
            )
    
    async def _decrease_need_logic(self, need_name: str, amount: float, reason_provided: str) -> DecreaseNeedResult:
        """Internal logic for decreasing a need."""
        if amount < 0:
            amount = abs(amount)
            
        if need_name not in self.needs:
            return DecreaseNeedResult(
                success=False,
                name=need_name,
                previous_level=0.0,
                new_level=0.0,
                change=0.0,
                reason=reason_provided,
                deficit=0.0,
                drive_strength=0.0,
                error=f"Need '{need_name}' not found"
            )
        
        async with self._lock:
            need = self.needs[need_name]
            original_level = need.level
            
            # Apply decrease
            new_level = max(0.0, need.level - amount)
            actual_change = original_level - new_level
            need.level = new_level
            need.last_updated = datetime.datetime.now()
            
            # Add history
            _add_history_entry(self.context, need_name, need, f"{reason_provided}_decrease")
            
            logger.info(f"Need '{need_name}' decreased by {actual_change:.3f}. Level: {original_level:.2f} -> {new_level:.2f}. Reason: {reason_provided}")
            
            return DecreaseNeedResult(
                success=True,
                name=need_name,
                previous_level=original_level,
                new_level=new_level,
                change=-actual_change,
                reason=f"{reason_provided}_decrease",
                deficit=need.deficit,
                drive_strength=need.drive_strength,
                error=None
            )
            
    def _get_needs_by_category_logic(self) -> NeedsByCategoryResponse:
        """Group needs by category and return list-based response."""
        # Get flat state list once
        all_states = {ns.name: ns for ns in self._get_needs_state_logic().needs}
    
        category_blocks = [
            NeedCategoryBlock(
                category=cat,
                needs=[all_states[name] for name in names if name in all_states],
            )
            for cat, names in self.context.need_categories.items()
        ]
        return NeedsByCategoryResponse(categories=category_blocks)

    def _get_needs_state_logic(self) -> NeedsStateResponse:
        """Return every need as a list."""
        needs_list = [
            NeedStateInfo(
                name=n,
                level=need.level,
                target_level=need.target_level,
                importance=need.importance,
                decay_rate=need.decay_rate,
                last_updated=need.last_updated.isoformat(),
                description=need.description,
                deficit=need.deficit,
                drive_strength=need.drive_strength,
            )
            for n, need in self.needs.items()
        ]
        return NeedsStateResponse(needs=needs_list)

    async def _get_most_unfulfilled_need_logic(self) -> MostUnfulfilledNeedResponse:
        # Update needs first
        await self._update_needs_logic()
        
        highest_drive = -1
        highest_need = None
        
        for name, need in self.needs.items():
            if need.drive_strength > highest_drive:
                highest_drive = need.drive_strength
                highest_need = need
        
        if highest_need:
            return MostUnfulfilledNeedResponse(
                name=highest_need.name,
                level=highest_need.level,
                target_level=highest_need.target_level,
                deficit=highest_need.deficit,
                drive_strength=highest_need.drive_strength,
                importance=highest_need.importance,
                description=highest_need.description
            )
        else:
            return MostUnfulfilledNeedResponse(
                name="none",
                level=1.0,
                target_level=1.0,
                deficit=0.0,
                drive_strength=0.0,
                importance=0.5,
                description="All needs are satisfied"
            )

    def _get_need_history_logic(self, need_name: str, limit: int = 20) -> List[NeedHistoryEntry]:
        """Internal logic for getting need history."""
        if need_name not in self.needs:
            return []
        
        effective_limit = limit if limit > 0 else self.max_history_per_need
        history = self.need_history.get(need_name, [])
        
        # Convert dict entries to NeedHistoryEntry objects
        typed_history = []
        for entry in history[-effective_limit:]:
            typed_history.append(NeedHistoryEntry(
                timestamp=entry['timestamp'],
                level=entry['level'],
                deficit=entry['deficit'],
                drive_strength=entry['drive_strength'],
                reason=entry['reason']
            ))
        
        return typed_history

    def _get_total_drive_logic(self) -> float:
        """Calculate total drive across all needs."""
        return sum(need.drive_strength for need in self.needs.values())

    async def _reset_need_to_default_logic(self, need_name: str) -> Union[ResetNeedResponse, ErrorResponse]:
        """Internal logic for resetting a need to default."""
        if need_name not in self.needs:
            return ErrorResponse(message=f"Unknown need: {need_name}")
        
        async with self._lock:
            default_values = {
                "knowledge": {"level": 0.5, "importance": 0.8, "decay_rate": 0.01, "target_level": 1.0},
                "pleasure_indulgence": {"level": 0.5, "importance": 0.85, "decay_rate": 0.03, "target_level": 0.95},
                "coherence": {"level": 0.5, "importance": 0.7, "decay_rate": 0.005, "target_level": 1.0},
                "agency": {"level": 0.5, "importance": 0.8, "decay_rate": 0.01, "target_level": 1.0},
                "connection": {"level": 0.5, "importance": 0.9, "decay_rate": 0.015, "target_level": 1.0},
                "intimacy": {"level": 0.5, "importance": 0.85, "decay_rate": 0.02, "target_level": 0.9},
                "safety": {"level": 0.8, "importance": 0.95, "decay_rate": 0.002, "target_level": 1.0},
                "novelty": {"level": 0.5, "importance": 0.6, "decay_rate": 0.02, "target_level": 1.0},
                "physical_closeness": {"level": 0.5, "importance": 0.7, "decay_rate": 0.03, "target_level": 1.0},
                "drive_expression": {"level": 0.2, "importance": 0.6, "decay_rate": 0.05, "target_level": 0.8},
                "control_expression": {"level": 0.4, "importance": 0.95, "decay_rate": 0.025, "target_level": 0.9}
            }
            
            values = default_values.get(need_name, {"level": 0.5, "importance": 0.5, "decay_rate": 0.01, "target_level": 1.0})
            
            # Save original state
            original_need = self.needs[need_name]
            original_state = NeedStateInfo(
                name=need_name,
                level=original_need.level,
                target_level=original_need.target_level,
                importance=original_need.importance,
                decay_rate=original_need.decay_rate,
                last_updated=original_need.last_updated.isoformat(),
                description=original_need.description,
                deficit=original_need.deficit,
                drive_strength=original_need.drive_strength
            )
            
            # Update the need
            self.needs[need_name] = NeedState(
                name=need_name,
                level=values["level"],
                importance=values["importance"],
                decay_rate=values["decay_rate"],
                target_level=values["target_level"],
                description=self.needs[need_name].description,
                last_updated=datetime.datetime.now()
            )
            
            # Get new state
            new_need = self.needs[need_name]
            new_state = NeedStateInfo(
                name=need_name,
                level=new_need.level,
                target_level=new_need.target_level,
                importance=new_need.importance,
                decay_rate=new_need.decay_rate,
                last_updated=new_need.last_updated.isoformat(),
                description=new_need.description,
                deficit=new_need.deficit,
                drive_strength=new_need.drive_strength
            )
            
            # Add to history
            _add_history_entry(self.context, need_name, self.needs[need_name], "reset_to_default")
            
            logger.info(f"Reset need '{need_name}' to default values. Level: {original_need.level:.2f} -> {new_need.level:.2f}")
            
            return ResetNeedResponse(
                status="success",
                need=need_name,
                original=original_state,
                new=new_state
            )

    # Public API Methods (call the agent)
    async def update_needs(self) -> DriveStrengthsResponse:
        """Ask the agent to update all needs."""
        prompt = "Please update all needs status and manage any urgent unfulfilled needs."
        result = await Runner.run(self.agent, prompt, context=self.context)
    
        if isinstance(result.final_output, DriveStrengthsResponse):
            return result.final_output
    
        # Fallback
        logger.error(
            "update_needs: Agent returned unexpected type %s",
            type(result.final_output),
        )
        return DriveStrengthsResponse(drive_strengths=[])
    
    async def satisfy_need(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> NeedSatisfactionResult:
        """Public API to satisfy a need."""
        # Convert dict context to SatisfyNeedContext if provided
        context = None
        if context_data:
            try:
                context = SatisfyNeedContext(**context_data)
            except Exception as e:
                logger.error(f"Invalid context data: {e}")
                return NeedSatisfactionResult(
                    name=name,
                    previous_level=0.0,
                    new_level=0.0,
                    change=0.0,
                    reason="error_invalid_context",
                    deficit=0.0,
                    drive_strength=0.0
                )
        
        prompt_parts = [f"Satisfy the need named '{name}' by an amount of {amount}."]
        if context:
            prompt_parts.append(f"Consider the following context: {context.model_dump_json()}.")
        prompt = " ".join(prompt_parts)
    
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        if isinstance(result.final_output, NeedSatisfactionResult):
            return result.final_output
        else:
            logger.error(f"satisfy_need: Agent returned unexpected type {type(result.final_output)}")
            return NeedSatisfactionResult(
                name=name,
                previous_level=0.0,
                new_level=0.0,
                change=0.0,
                reason="error_agent_response",
                deficit=0.0,
                drive_strength=0.0
            )
    
    async def decrease_need(self, name: str, amount: float, reason: Optional[str] = None) -> DecreaseNeedResult:
        """Public API to decrease a need."""
        reason_provided = reason if reason is not None else "generic_decrease_api_call"
        prompt = f"decrease_need_tool_impl: {name}, {amount}, {reason_provided}"
        
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        if isinstance(result.final_output, DecreaseNeedResult):
            return result.final_output
        else:
            logger.error(f"decrease_need: Agent returned unexpected type {type(result.final_output)}")
            return DecreaseNeedResult(
                success=False,
                name=name,
                previous_level=0.0,
                new_level=0.0,
                change=0.0,
                reason="error_agent_response",
                deficit=0.0,
                drive_strength=0.0,
                error="Agent did not return expected response"
            )
    
    async def get_most_unfulfilled_need(self) -> MostUnfulfilledNeedResponse:
        """Get the most unfulfilled need."""
        prompt = "Identify and return details for the most unfulfilled need currently."
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        if isinstance(result.final_output, MostUnfulfilledNeedResponse):
            return result.final_output
        else:
            logger.error(f"get_most_unfulfilled_need: Agent returned unexpected type {type(result.final_output)}")
            return MostUnfulfilledNeedResponse(
                name="none",
                level=1.0,
                target_level=1.0,
                deficit=0.0,
                drive_strength=0.0,
                importance=0.5,
                description="Error retrieving most unfulfilled need"
            )
    
    async def get_need_history(self, need_name: str, limit: int = 20) -> List[NeedHistoryEntry]:
        """Get history for a specific need."""
        prompt_parts = [f"Retrieve the history for the need named '{need_name}'."]
        if limit != 20:
            prompt_parts.append(f"Limit the results to {limit} entries.")
        prompt = " ".join(prompt_parts)
        
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        
        # Check if result is a list of NeedHistoryEntry
        if isinstance(result.final_output, list):
            if all(isinstance(item, NeedHistoryEntry) for item in result.final_output):
                return result.final_output
            else:
                logger.error("get_need_history: Not all items in list are NeedHistoryEntry")
                return []
        else:
            logger.error(f"get_need_history: Agent returned unexpected type {type(result.final_output)}")
            return []
    
    async def reset_need_to_default(self, need_name: str) -> Union[ResetNeedResponse, ErrorResponse]:
        """Reset a need to default values."""
        prompt = f"Reset the need named '{need_name}' to its default state."
        result = await Runner.run(
            self.agent,
            prompt,
            context=self.context
        )
        if isinstance(result.final_output, (ResetNeedResponse, ErrorResponse)):
            return result.final_output
        else:
            logger.error(f"reset_need_to_default: Agent returned unexpected type {type(result.final_output)}")
            return ErrorResponse(message="Agent did not return expected response")

    # Synchronous wrappers for compatibility
    def get_needs_state(self) -> NeedsStateResponse:
        """Synchronous wrapper - prefer async methods."""
        logger.warning("Synchronous get_needs_state called; prefer async agent interaction.")
        return self._get_needs_state_logic()
    
    def get_total_drive(self) -> float:
        """Synchronous wrapper - prefer async methods."""
        logger.warning("Synchronous get_total_drive called; prefer async agent interaction.")
        return self._get_total_drive_logic()
    
    def get_needs_by_category(self) -> NeedsByCategoryResponse:
        """Synchronous wrapper - prefer async methods."""
        logger.warning("Synchronous get_needs_by_category called; prefer async agent interaction.")
        return self._get_needs_by_category_logic()
 
    async def get_needs_state_async(self) -> NeedsStateResponse:
        """Get all needs state via agent."""
        prompt = "Get the current state of all needs."
        try:
            result = await Runner.run(self.agent, prompt, context=self.context)
            output = result.final_output
    
            if isinstance(output, NeedsStateResponse):
                return output
    
            # Agent sent raw JSON string
            if isinstance(output, str):
                logger.error(
                    "Agent returned JSON string instead of using tools; "
                    "falling back to direct logic."
                )
                return self._get_needs_state_logic()
    
            # Agent sent a dict of name → fields (fixed to create list)
            if isinstance(output, dict):
                try:
                    needs_list = [
                        NeedStateInfo(**data if isinstance(data, dict) else data.__dict__)
                        for data in output.values()
                    ]
                    return NeedsStateResponse(needs=needs_list)
                except Exception as e:
                    logger.error("Failed to convert agent dict output: %s", e)
                    return self._get_needs_state_logic()
    
            # Unexpected type
            logger.error(
                "get_needs_state_async: Agent returned unexpected type %s",
                type(output),
            )
            return self._get_needs_state_logic()
    
        except Exception as e:
            logger.error("Error in get_needs_state_async: %s", e)
            return self._get_needs_state_logic()
        
    async def get_total_drive_async(self) -> float:
        """Get total drive via agent."""
        prompt = "Calculate the total drive of all needs."
        result = await Runner.run(self.agent, prompt, context=self.context)
        output = result.final_output
    
        if isinstance(output, float):
            return output
    
        # legacy dict-shape fallback (returns float, so it's fine)
        if isinstance(output, dict) and "total_drive" in output:
            return output["total_drive"]
    
        # new strict DriveStrengthsResponse (list of entries)
        if isinstance(output, DriveStrengthsResponse):
            return sum(entry.strength for entry in output.drive_strengths)
    
        logger.warning(
            "get_total_drive_async: unexpected output type %s", type(output)
        )
        return 0.0
    
    async def get_needs_by_category_async(self) -> NeedsByCategoryResponse:
        """Async method to get needs by category."""
        prompt = "Get all needs organized by their categories."
        result = await Runner.run(self.agent, prompt, context=self.context)
        if isinstance(result.final_output, NeedsByCategoryResponse):
            return result.final_output
        else:
            logger.error(f"get_needs_by_category_async: Agent returned unexpected type {type(result.final_output)}")
            # Return empty list-based response
            return NeedsByCategoryResponse(categories=[])
