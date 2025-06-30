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

# New Rate Limiter Class
class NeedsSystemRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, default_interval: float = 60):
        self.default_interval = default_interval
        self.operation_intervals = {
            'update_needs': 300,  # 5 minutes
            'satisfy_need': 10,   # 10 seconds
            'decrease_need': 10,  # 10 seconds
            'get_needs_state': 30,  # 30 seconds
            'get_total_drive': 60,  # 1 minute
            'get_most_unfulfilled': 120,  # 2 minutes
        }
        self.last_calls = {}
        self.call_counts = {}
        self.burst_limits = {
            'satisfy_need': 5,    # Max 5 calls in burst window
            'decrease_need': 5,
            'get_needs_state': 3,
        }
        self.burst_windows = {}  # Track burst windows
        self.burst_window_duration = 60  # 1 minute burst windows
        
    async def should_allow_call(self, operation: str, bypass_for_critical: bool = False) -> Tuple[bool, str]:
        """
        Check if operation should be allowed.
        Returns (allowed, reason_if_denied)
        """
        now = datetime.datetime.now()
        
        # Allow critical operations to bypass
        if bypass_for_critical and operation in ['satisfy_need', 'decrease_need']:
            return True, "bypassed_for_critical"
        
        # Check rate limit
        interval = self.operation_intervals.get(operation, self.default_interval)
        if operation in self.last_calls:
            elapsed = (now - self.last_calls[operation]).total_seconds()
            if elapsed < interval:
                remaining = interval - elapsed
                return False, f"rate_limited_{remaining:.1f}s_remaining"
        
        # Check burst limit if applicable
        if operation in self.burst_limits:
            allowed, reason = self._check_burst_limit(operation, now)
            if not allowed:
                return False, reason
        
        # Update tracking
        self.last_calls[operation] = now
        self._update_burst_tracking(operation, now)
        
        return True, "allowed"
    
    def _check_burst_limit(self, operation: str, now: datetime.datetime) -> Tuple[bool, str]:
        """Check if operation exceeds burst limit"""
        if operation not in self.burst_windows:
            self.burst_windows[operation] = []
        
        # Clean old entries
        cutoff = now - datetime.timedelta(seconds=self.burst_window_duration)
        self.burst_windows[operation] = [t for t in self.burst_windows[operation] if t > cutoff]
        
        # Check limit
        current_count = len(self.burst_windows[operation])
        limit = self.burst_limits[operation]
        
        if current_count >= limit:
            return False, f"burst_limit_exceeded_{current_count}/{limit}"
        
        return True, "within_burst_limit"
    
    def _update_burst_tracking(self, operation: str, now: datetime.datetime):
        """Update burst tracking for operation"""
        if operation in self.burst_limits:
            if operation not in self.burst_windows:
                self.burst_windows[operation] = []
            self.burst_windows[operation].append(now)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status"""
        now = datetime.datetime.now()
        status = {}
        
        for operation, last_time in self.last_calls.items():
            interval = self.operation_intervals.get(operation, self.default_interval)
            elapsed = (now - last_time).total_seconds()
            remaining = max(0, interval - elapsed)
            
            status[operation] = {
                'last_call': last_time.isoformat(),
                'cooldown_remaining': remaining,
                'interval': interval
            }
            
            # Add burst info if applicable
            if operation in self.burst_limits:
                current_count = len([t for t in self.burst_windows.get(operation, []) 
                                   if (now - t).total_seconds() < self.burst_window_duration])
                status[operation]['burst_count'] = current_count
                status[operation]['burst_limit'] = self.burst_limits[operation]
        
        return status
    
    def reset_limits(self, operation: Optional[str] = None):
        """Reset rate limits for specific operation or all"""
        if operation:
            self.last_calls.pop(operation, None)
            self.burst_windows.pop(operation, None)
        else:
            self.last_calls.clear()
            self.burst_windows.clear()

class NeedsSystemContext:
    """Context for the NeedsSystem agent operations AND internal logic."""
    def __init__(self, needs_system_instance, goal_manager=None):
        self.needs_system_ref = needs_system_instance
        self.goal_manager = goal_manager
        self._lock = asyncio.Lock()
        
        # Add decay control parameters (Solution 5)
        self.min_decay_interval = 60  # Minimum seconds between decay updates
        self.max_decay_interval = 3600  # Maximum seconds between decay updates
        self.adaptive_decay = True  # Enable adaptive decay intervals
        self.high_drive_threshold = 0.7  # Drive level that triggers faster updates
        self.low_activity_threshold = 0.2  # Activity level below which we slow updates
        self.last_significant_change = datetime.datetime.now()
        self.activity_score = 0.5  # Track system activity level
        
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
        
        # Add batch processing attributes (Solution 4)
        self._pending_satisfactions = []  # Queue satisfaction operations
        self._pending_decreases = []      # Queue decrease operations
        self._last_batch_update = datetime.datetime.now()
        self._batch_interval = 60  # Process batch every minute
        self._batch_lock = asyncio.Lock()
        
        # Add rate limiter (Solution 6)
        self.rate_limiter = NeedsSystemRateLimiter()
        self._cached_responses = {}  # Cache for rate-limited responses
        self._cache_expiry = {}  # Expiry times for cached responses

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
            model="gpt-4.1-nano"
        )
        logger.info("NeedsSystem initialized with Agent SDK and performance optimizations")

    # Solution 5: Adaptive decay calculation
    def _calculate_adaptive_decay_interval(self) -> float:
        """
        Calculate decay interval based on current system state.
        Returns interval in seconds.
        """
        base_interval = 300  # 5 minutes base
        
        # Check for high-priority situations
        high_drive_needs = sum(1 for need in self.needs.values() if need.drive_strength > self.context.high_drive_threshold)
        if high_drive_needs > 0:
            # Urgent needs - check more frequently
            return max(self.context.min_decay_interval, base_interval / (1 + high_drive_needs))
        
        # Check activity level
        if self.context.activity_score < self.context.low_activity_threshold:
            # Low activity - check less frequently
            return min(self.context.max_decay_interval, base_interval * 2)
        
        # Check time since last significant change
        time_since_change = (datetime.datetime.now() - self.context.last_significant_change).total_seconds()
        if time_since_change > 1800:  # 30 minutes
            # Nothing happening - slow down
            return min(self.context.max_decay_interval, base_interval * 1.5)
        
        return base_interval
    
    def _update_activity_score(self, change_magnitude: float):
        """Update activity score based on changes"""
        # Exponential moving average
        alpha = 0.3
        self.context.activity_score = (alpha * change_magnitude + 
                                       (1 - alpha) * self.context.activity_score)
        self.context.activity_score = max(0.0, min(1.0, self.context.activity_score))

    # Internal logic methods - Modified with rate limiting
    async def _update_needs_logic(self) -> DriveStrengthsResponse:
        """
        Apply decay with rate limiting and enhanced frequency control.
        """
        async with self._lock:
            now = datetime.datetime.now()
            elapsed_seconds = (now - self.last_update_time).total_seconds()
            
            # Calculate appropriate interval
            if self.context.adaptive_decay:
                required_interval = self._calculate_adaptive_decay_interval()
            else:
                required_interval = self.context.min_decay_interval
            
            # Skip if too soon (respecting 1 minute minimum)
            min_interval = max(60, required_interval)  # At least 1 minute
            if elapsed_seconds < min_interval:
                logger.debug(
                    f"Skipping needs decay (only {elapsed_seconds:.1f}s elapsed, "
                    f"need {min_interval:.1f}s)"
                )
                # Return current drives without update
                drive_entries = [
                    DriveStrengthEntry(name=n, strength=nd.drive_strength) 
                    for n, nd in self.needs.items()
                ]
                return DriveStrengthsResponse(drive_strengths=drive_entries)
            
            # Proceed with decay
            elapsed_hours = elapsed_seconds / 3600.0
            drive_strengths: Dict[str, float] = {}
            needs_to_trigger_goals: List[NeedState] = []
            total_change = 0.0
            significant_changes = 0

            for name, need in self.needs.items():
                baseline = 0.3
                decay = need.decay_rate * elapsed_hours
                old_level = need.level

                if need.level > baseline:
                    need.level = max(baseline, need.level - decay)
                elif need.level < baseline:
                    need.level = min(baseline, need.level + decay * 0.5)

                need.level = max(0.0, min(1.0, need.level))
                need.last_updated = now

                # Track changes
                change = abs(old_level - need.level)
                total_change += change
                if change > 0.05:
                    significant_changes += 1

                drive = need.drive_strength
                drive_strengths[name] = drive

                # History & goal-creation bookkeeping
                if not self.need_history[name] or change > 0.05:
                    _add_history_entry(self.context, name, need, "decay")

                if drive > self.context.drive_threshold_for_goal:
                    last = self.goal_cooldown.get(name)
                    if not last or (now - last) >= self.context.goal_cooldown_duration:
                        if (not self.context.goal_manager
                            or not getattr(self.context.goal_manager, "has_active_goal_for_need", lambda *_: False)(name)):
                            needs_to_trigger_goals.append(need)
                            self.goal_cooldown[name] = now

            # Update tracking
            self.last_update_time = now
            if significant_changes > 0:
                self.context.last_significant_change = now
            
            # Update activity score
            avg_change = total_change / len(self.needs) if self.needs else 0
            self._update_activity_score(avg_change * 10)  # Scale for sensitivity
            
            logger.debug(
                f"Needs decay applied after {elapsed_seconds:.1f}s. "
                f"Total change: {total_change:.3f}, Activity: {self.context.activity_score:.2f}"
            )

            # Async goal creation
            if needs_to_trigger_goals and self.context.goal_manager:
                await _trigger_goal_creation(self.context, needs_to_trigger_goals)

            # Build response
            drive_entries = [
                DriveStrengthEntry(name=n, strength=s) for n, s in drive_strengths.items()
            ]
            return DriveStrengthsResponse(drive_strengths=drive_entries)

    # Solution 4: Batch processing methods
    async def satisfy_need_batched(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Queue a need satisfaction for batch processing.
        Returns immediately with a pending status.
        """
        async with self._batch_lock:
            self._pending_satisfactions.append({
                'name': name,
                'amount': amount,
                'context': context_data,
                'timestamp': datetime.datetime.now(),
                'id': f"{name}_{datetime.datetime.now().timestamp()}"
            })
            
            # Check if we should process batch
            if await self._should_process_batch():
                return await self._process_satisfaction_batch()
            
            return {
                'status': 'queued',
                'name': name,
                'amount': amount,
                'batch_size': len(self._pending_satisfactions),
                'next_batch_in': self._batch_interval - (datetime.datetime.now() - self._last_batch_update).total_seconds()
            }
    
    async def decrease_need_batched(self, name: str, amount: float, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Queue a need decrease for batch processing.
        Returns immediately with a pending status.
        """
        async with self._batch_lock:
            self._pending_decreases.append({
                'name': name,
                'amount': amount,
                'reason': reason or "batched_decrease",
                'timestamp': datetime.datetime.now(),
                'id': f"{name}_{datetime.datetime.now().timestamp()}"
            })
            
            # Check if we should process batch
            if await self._should_process_batch():
                return await self._process_all_batches()
            
            return {
                'status': 'queued',
                'name': name,
                'amount': amount,
                'batch_size': len(self._pending_decreases),
                'next_batch_in': self._batch_interval - (datetime.datetime.now() - self._last_batch_update).total_seconds()
            }
    
    async def _should_process_batch(self) -> bool:
        """Check if we should process the batch now"""
        time_elapsed = (datetime.datetime.now() - self._last_batch_update).total_seconds()
        
        # Process if time interval exceeded
        if time_elapsed >= self._batch_interval:
            return True
        
        # Process if batch is getting large
        total_pending = len(self._pending_satisfactions) + len(self._pending_decreases)
        if total_pending >= 20:  # Max batch size
            return True
            
        # Process if we have critical needs
        for item in self._pending_satisfactions:
            if item['name'] in ['safety', 'coherence'] and item['amount'] > 0.5:
                return True
                
        return False
    
    async def _process_satisfaction_batch(self) -> Dict[str, Any]:
        """Process all pending satisfactions at once"""
        if not self._pending_satisfactions:
            return {'status': 'no_pending_satisfactions'}
            
        results = []
        
        # Group by need name and aggregate
        grouped = {}
        contexts_by_need = {}
        
        for item in self._pending_satisfactions:
            name = item['name']
            if name not in grouped:
                grouped[name] = 0
                contexts_by_need[name] = []
            grouped[name] += item['amount']
            if item['context']:
                contexts_by_need[name].append(item['context'])
        
        # Apply satisfactions
        for name, total_amount in grouped.items():
            # Use the most recent context if multiple
            context = contexts_by_need[name][-1] if contexts_by_need[name] else None
            
            # Apply satisfaction with aggregated amount
            result = await self._satisfy_need_immediate(name, total_amount, context)
            results.append({
                'need': name,
                'total_amount': total_amount,
                'final_level': result.new_level,
                'context_count': len(contexts_by_need[name])
            })
            
            logger.info(f"Batch processed {len(contexts_by_need[name])} satisfactions for '{name}', total amount: {total_amount:.3f}")
        
        self._pending_satisfactions.clear()
        self._last_batch_update = datetime.datetime.now()
        
        return {
            'status': 'batch_processed',
            'results': results,
            'total_processed': len(results)
        }
    
    async def _process_decrease_batch(self) -> Dict[str, Any]:
        """Process all pending decreases at once"""
        if not self._pending_decreases:
            return {'status': 'no_pending_decreases'}
            
        results = []
        
        # Group by need name and aggregate
        grouped = {}
        reasons_by_need = {}
        
        for item in self._pending_decreases:
            name = item['name']
            if name not in grouped:
                grouped[name] = 0
                reasons_by_need[name] = []
            grouped[name] += item['amount']
            reasons_by_need[name].append(item['reason'])
        
        # Apply decreases
        for name, total_amount in grouped.items():
            # Combine reasons
            combined_reason = f"batch_decrease: {', '.join(set(reasons_by_need[name]))}"
            
            # Apply decrease with aggregated amount
            result = await self._decrease_need_logic(name, total_amount, combined_reason)
            results.append({
                'need': name,
                'total_decrease': total_amount,
                'final_level': result.new_level,
                'reason_count': len(reasons_by_need[name])
            })
            
            logger.info(f"Batch processed {len(reasons_by_need[name])} decreases for '{name}', total amount: {total_amount:.3f}")
        
        self._pending_decreases.clear()
        
        return {
            'status': 'batch_processed',
            'results': results,
            'total_processed': len(results)
        }
    
    async def _process_all_batches(self) -> Dict[str, Any]:
        """Process both satisfaction and decrease batches"""
        satisfaction_results = await self._process_satisfaction_batch()
        decrease_results = await self._process_decrease_batch()
        
        self._last_batch_update = datetime.datetime.now()
        
        return {
            'status': 'all_batches_processed',
            'satisfactions': satisfaction_results,
            'decreases': decrease_results,
            'timestamp': self._last_batch_update.isoformat()
        }
    
    async def force_process_batches(self) -> Dict[str, Any]:
        """Force immediate processing of all pending batches"""
        async with self._batch_lock:
            return await self._process_all_batches()
    
    async def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch processing status"""
        time_until_next = self._batch_interval - (datetime.datetime.now() - self._last_batch_update).total_seconds()
        
        return {
            'pending_satisfactions': len(self._pending_satisfactions),
            'pending_decreases': len(self._pending_decreases),
            'time_until_next_batch': max(0, time_until_next),
            'batch_interval': self._batch_interval,
            'last_batch_processed': self._last_batch_update.isoformat()
        }

    # Solution 6: Rate limiting integration
    async def _get_cached_or_execute(self, operation: str, cache_duration: int, 
                                   execute_func, *args, **kwargs):
        """Get cached response or execute function with rate limiting"""
        # Check cache first
        if operation in self._cached_responses:
            if operation in self._cache_expiry:
                if datetime.datetime.now() < self._cache_expiry[operation]:
                    logger.debug(f"Returning cached response for {operation}")
                    return self._cached_responses[operation]
        
        # Check rate limit
        allowed, reason = await self.rate_limiter.should_allow_call(operation)
        if not allowed:
            logger.warning(f"Rate limited: {operation} - {reason}")
            # Return cached if available, even if expired
            if operation in self._cached_responses:
                return self._cached_responses[operation]
            # Return default based on operation type
            if operation == 'update_needs':
                return self._get_current_drives_without_update()
            elif operation == 'get_needs_state':
                return self._get_needs_state_logic()
            else:
                raise RuntimeError(f"Rate limited and no cache available: {operation}")
        
        # Execute function
        result = await execute_func(*args, **kwargs)
        
        # Cache result
        self._cached_responses[operation] = result
        self._cache_expiry[operation] = datetime.datetime.now() + datetime.timedelta(seconds=cache_duration)
        
        return result
    
    def _get_current_drives_without_update(self) -> DriveStrengthsResponse:
        """Get current drives without applying decay"""
        drive_entries = [
            DriveStrengthEntry(name=n, strength=nd.drive_strength) 
            for n, nd in self.needs.items()
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

    # Internal update method for rate limiting
    async def _update_needs_internal(self) -> DriveStrengthsResponse:
        """Internal update needs implementation"""
        prompt = "Please update all needs status and manage any urgent unfulfilled needs."
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        if isinstance(result.final_output, DriveStrengthsResponse):
            return result.final_output
        
        logger.error("update_needs: Agent returned unexpected type %s", type(result.final_output))
        return DriveStrengthsResponse(drive_strengths=[])

    async def _get_needs_state_internal(self) -> NeedsStateResponse:
        """Internal get needs state implementation"""
        prompt = "Get the current state of all needs."
        try:
            result = await Runner.run(self.agent, prompt, context=self.context)
            output = result.final_output
            
            if isinstance(output, NeedsStateResponse):
                return output
            
            # Fallback handling...
            return self._get_needs_state_logic()
            
        except Exception as e:
            logger.error("Error in get_needs_state_async: %s", e)
            return self._get_needs_state_logic()

    async def _satisfy_need_immediate(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> NeedSatisfactionResult:
        """Execute satisfy need immediately"""
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
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
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

    # Public API Methods with rate limiting
    async def update_needs(self) -> DriveStrengthsResponse:
        """Ask the agent to update all needs with rate limiting."""
        return await self._get_cached_or_execute(
            'update_needs',
            300,  # Cache for 5 minutes
            self._update_needs_internal
        )
    
    async def satisfy_need(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> NeedSatisfactionResult:
        """Satisfy need with rate limiting and optional batching."""
        # Check if this is a critical need
        is_critical = name in ['safety', 'coherence'] and amount > 0.3
        
        allowed, reason = await self.rate_limiter.should_allow_call('satisfy_need', bypass_for_critical=is_critical)
        
        if not allowed and not is_critical:
            # Queue for batching instead
            logger.info(f"Rate limited satisfy_need for '{name}', queuing for batch")
            batch_result = await self.satisfy_need_batched(name, amount, context_data)
            
            # Return a placeholder result
            current_need = self.needs.get(name)
            if current_need:
                return NeedSatisfactionResult(
                    name=name,
                    previous_level=current_need.level,
                    new_level=current_need.level,
                    change=0.0,
                    reason=f"queued_for_batch_{reason}",
                    deficit=current_need.deficit,
                    drive_strength=current_need.drive_strength
                )
            else:
                return NeedSatisfactionResult(
                    name=name,
                    previous_level=0.5,
                    new_level=0.5,
                    change=0.0,
                    reason=f"queued_for_batch_{reason}",
                    deficit=0.5,
                    drive_strength=0.5
                )
        
        # Proceed with immediate execution
        return await self._satisfy_need_immediate(name, amount, context_data)
    
    async def decrease_need(self, name: str, amount: float, reason: Optional[str] = None) -> DecreaseNeedResult:
        """Public API to decrease a need with rate limiting."""
        # Check if this is a critical operation
        is_critical = name in ['safety', 'coherence'] and amount > 0.3
        
        allowed, rate_reason = await self.rate_limiter.should_allow_call('decrease_need', bypass_for_critical=is_critical)
        
        if not allowed and not is_critical:
            # Queue for batching
            logger.info(f"Rate limited decrease_need for '{name}', queuing for batch")
            batch_result = await self.decrease_need_batched(name, amount, reason)
            
            # Return a placeholder result
            current_need = self.needs.get(name)
            if current_need:
                return DecreaseNeedResult(
                    success=True,
                    name=name,
                    previous_level=current_need.level,
                    new_level=current_need.level,
                    change=0.0,
                    reason=f"queued_for_batch_{rate_reason}",
                    deficit=current_need.deficit,
                    drive_strength=current_need.drive_strength,
                    error=None
                )
            else:
                return DecreaseNeedResult(
                    success=False,
                    name=name,
                    previous_level=0.5,
                    new_level=0.5,
                    change=0.0,
                    reason=f"queued_for_batch_{rate_reason}",
                    deficit=0.5,
                    drive_strength=0.5,
                    error=f"Need '{name}' not found"
                )
        
        # Proceed with immediate execution
        reason_provided = reason if reason is not None else "generic_decrease_api_call"
        prompt = f"decrease_need_tool_impl: {name}, {amount}, {reason_provided}"
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
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
        """Get all needs state with rate limiting."""
        return await self._get_cached_or_execute(
            'get_needs_state',
            120,  # Cache for 2 minutes
            self._get_needs_state_internal
        )
        
    async def get_total_drive_async(self) -> float:
        """Get total drive with rate limiting."""
        return await self._get_cached_or_execute(
            'get_total_drive',
            60,  # Cache for 1 minute
            self._get_total_drive_internal
        )
    
    async def _get_total_drive_internal(self) -> float:
        """Internal get total drive implementation"""
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

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting and batch status"""
        return {
            'rate_limits': self.rate_limiter.get_rate_limit_status(),
            'cached_responses': list(self._cached_responses.keys()),
            'batch_status': await self.get_batch_status(),
            'adaptive_decay': {
                'enabled': self.context.adaptive_decay,
                'activity_score': self.context.activity_score,
                'last_significant_change': self.context.last_significant_change.isoformat(),
                'current_interval': self._calculate_adaptive_decay_interval()
            }
        }
