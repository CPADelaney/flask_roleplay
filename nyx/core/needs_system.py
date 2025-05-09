# nyx/core/needs_system.py

import logging
import datetime
import math
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

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

class NeedSatisfactionResult(BaseModel):
    """Result of satisfying a need."""
    name: str
    previous_level: float
    new_level: float
    change: float
    reason: str
    deficit: float
    drive_strength: float

class NeedCategory(BaseModel):
    """Category of needs"""
    name: str
    needs: List[str]

class NeedsSystemContext:
    """Context for the NeedsSystem agent operations AND internal logic."""
    def __init__(self, needs_system_instance, goal_manager=None): # Add needs_system_instance
        self.needs_system_ref = needs_system_instance # Store reference to parent NeedsSystem
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

# Function tools for needs operations
@function_tool
async def update_needs(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, float]:
    """
    Updates all needs by applying decay and returns current drive strengths, potentially triggering goals.
    
    Returns:
        Dictionary mapping need names to their drive strengths
    """
    needs_ctx = ctx.context
    
    async with needs_ctx._lock:
        now = datetime.datetime.now()
        elapsed_hours = (now - needs_ctx.last_update_time).total_seconds() / 3600.0

        drive_strengths = {}
        needs_to_trigger_goals = []

        if elapsed_hours > 0.001:  # Only update if ~4 seconds passed
            for name, need in needs_ctx.needs.items():
                # Apply decay towards a baseline (e.g., 0.3)
                baseline_satisfaction = 0.3
                decay_amount = need.decay_rate * elapsed_hours
                
                if need.level > baseline_satisfaction:
                    need.level = max(baseline_satisfaction, need.level - decay_amount)
                elif need.level < baseline_satisfaction:
                    # Slowly drifts up towards baseline if very low
                    need.level = min(baseline_satisfaction, need.level + (decay_amount * 0.5))

                need.level = max(0.0, min(1.0, need.level))  # Clamp
                need.last_updated = now
                drive = need.drive_strength
                drive_strengths[name] = drive
                
                # Add history entry if significant change or first entry
                if (len(needs_ctx.need_history[name]) == 0 or 
                    abs(needs_ctx.need_history[name][-1]['level'] - need.level) > 0.05):
                    _add_history_entry(needs_ctx, name, need, "decay")

                # Check if need deficit triggers a goal
                if drive > needs_ctx.drive_threshold_for_goal:
                    # Check cooldown
                    last_triggered = needs_ctx.goal_cooldown.get(name)
                    if last_triggered and (now - last_triggered) < needs_ctx.goal_cooldown_duration:
                        continue  # Still in cooldown

                    # Check if a similar goal already exists and is active/pending
                    if not needs_ctx.goal_manager or not hasattr(needs_ctx.goal_manager, 'has_active_goal_for_need') or not needs_ctx.goal_manager.has_active_goal_for_need(name):
                        needs_to_trigger_goals.append(need)
                        needs_ctx.goal_cooldown[name] = now  # Set cooldown

            needs_ctx.last_update_time = now

            # Trigger goal creation asynchronously for high-drive needs
            if needs_to_trigger_goals and needs_ctx.goal_manager:
                 await _trigger_goal_creation(needs_ctx, needs_to_trigger_goals)
        else:
            # Just get current drive strengths without updating
            drive_strengths = {name: need.drive_strength for name, need in needs_ctx.needs.items()}

        return drive_strengths

@function_tool
async def satisfy_need(
    ctx: RunContextWrapper[NeedsSystemContext], 
    name: str, 
    amount: float,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Increases the satisfaction level of a need, potentially modified by context.
    
    Args:
        name: Name of the need
        amount: Base amount to satisfy by (0.0-1.0)
        context: Optional context affecting satisfaction (e.g., difficulty_level, intensity)
        
    Returns:
        Dictionary with details about the need satisfaction
    """
    needs_ctx = ctx.context
    
    if name not in needs_ctx.needs:
        logger.warning(f"Attempted to satisfy unknown need: {name}")
        return {"status": "error", "message": f"Unknown need: {name}"}
        
    async with needs_ctx._lock:
        need = needs_ctx.needs[name]
        original_level = need.level
        modified_amount = amount
        satisfaction_multiplier = 1.0
        reason = "standard_satisfaction"

        # --- Context-specific modifications ---
        if name == "control_expression" and context:
            difficulty = context.get("difficulty_level", 0.3)  # Default moderate difficulty
            resistance_overcome = context.get("resistance_overcome", False)
            intensity = context.get("intensity_achieved", 0.5)

            # Base multiplier on difficulty/intensity
            satisfaction_multiplier = 0.5 + (difficulty * 0.5) + (intensity * 0.5)  # Range 0.5 - 1.5

            # Big boost for overcoming resistance
            if resistance_overcome:
                satisfaction_multiplier *= 1.5  # Overcoming resistance is highly satisfying
                reason = "resistance_overcome"
            elif difficulty < 0.2:  # Very easy compliance
                satisfaction_multiplier *= 0.7  # Less satisfying
                reason = "easy_compliance"
            else:
                 reason = f"compliance_d{difficulty:.1f}_i{intensity:.1f}"

            # Clamp multiplier
            satisfaction_multiplier = max(0.3, min(2.0, satisfaction_multiplier))
            modified_amount = amount * satisfaction_multiplier
        # Other context-specific modifications could be added here
        # ---

        # Apply the satisfaction
        need.level = min(need.target_level, need.level + modified_amount)
        need.last_updated = datetime.datetime.now()
        
        # Add to history
        _add_history_entry(needs_ctx, name, need, reason)
        
        logger.debug(f"Satisfied need '{name}' by {modified_amount:.3f} (Base: {amount:.3f}, Multiplier: {satisfaction_multiplier:.2f}, Reason: {reason}). Level: {original_level:.2f} -> {need.level:.2f}")
        
        result = NeedSatisfactionResult(
            name=name,
            previous_level=original_level,
            new_level=need.level,
            change=modified_amount,
            reason=reason,
            deficit=need.deficit,
            drive_strength=need.drive_strength
        )
        
        return result.dict()

@function_tool
async def decrease_need(self,
                       need_name: str,
                       amount: float,
                       reason: Optional[str] = None # <--- CORRECT: Use Optional with default None
                       ) -> Dict[str, Any]:
    """
    Decreases the satisfaction level of a need.
    
    Args:
        name: Name of the need
        amount: Amount to decrease by (0.0-1.0)
        reason: Reason for the decrease
        
    Returns:
        Dictionary with details about the need change
    """
    # --- Handle None case internally ---
    reason_provided = reason if reason is not None else "unspecified"
    logger.info(f"Decreasing need '{need_name}' by {amount}. Reason: {reason_provided}")
    # --- End internal handling ---

    # Check if this method uses Runner.run internally as suggested by traceback
    # If so, the agent it calls needs access to the *logic* of decreasing a need,
    # NOT the decrease_need method itself if that method is also the tool.
    # This implies a potential structure issue similar to the MoodManager problem.

    # --- Potential Internal Agent Call (as suggested by traceback) ---
    if hasattr(self, '_decrease_need_agent'): # Assuming an agent handles the logic
         agent_input = {
             "need_name": need_name,
             "amount": amount,
             "reason": reason_provided # Pass the handled reason
         }
         # It's this Runner.run call that fails if the *agent's tools* have bad schemas
         result = await Runner.run(
             starting_agent=self._decrease_need_agent, # The agent doing the work
             input=json.dumps(agent_input),
             context=self.context # Assuming NeedsSystem has a context
         )
         # The error happens *before* the agent returns, during the API call setup.
         return result.final_output # Or process result as needed

    # --- Alternative: Direct Logic (If no internal agent) ---
    else:
         async with self._lock: # Assuming a lock exists
             if need_name not in self.needs:
                 logger.warning(f"Attempted to decrease non-existent need: {need_name}")
                 return {"success": False, "error": f"Need '{need_name}' not found."}

             # Ensure amount is positive for decreasing deficit
             if amount < 0:
                  logger.warning(f"Decrease amount must be positive, got: {amount}")
                  amount = abs(amount) # Or return error

             need_state = self.needs[need_name]
             old_deficit = need_state['deficit']
             # Decrease deficit (increase satisfaction), clamp at 0
             need_state['deficit'] = max(0.0, old_deficit - amount)
             need_state['last_satisfied'] = datetime.datetime.now().isoformat()

             logger.info(f"Need '{need_name}' deficit decreased from {old_deficit:.2f} to {need_state['deficit']:.2f}. Reason: {reason_provided}")

             # Add history logic if needed...

             return {
                 "success": True,
                 "need_name": need_name,
                 "old_deficit": old_deficit,
                 "new_deficit": need_state['deficit'],
                 "amount": amount
             }

@function_tool
async def get_needs_state(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Dict[str, Any]]:
    """
    Returns the current state of all needs, including deficit and drive.
    
    Returns:
        Dictionary mapping need names to their complete state
    """
    needs_ctx = ctx.context
    
    return {name: {**need.dict(exclude={'last_updated'}),
                  'last_updated': need.last_updated.isoformat(),
                  'deficit': need.deficit,
                  'drive_strength': need.drive_strength
                 }
            for name, need in needs_ctx.needs.items()}

@function_tool
async def get_needs_by_category(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Gets needs organized by category for better organization and understanding.
    
    Returns:
        Dictionary of categories mapping to their contained needs
    """
    needs_ctx = ctx.context
    result = {}
    all_states = await get_needs_state(ctx)
    
    for category, need_names in needs_ctx.need_categories.items():
        result[category] = {
            name: all_states[name] for name in need_names 
            if name in all_states
        }
    
    return result

@function_tool
async def get_most_unfulfilled_need(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Any]:
    """
    Gets the need with the highest drive strength (most urgent).
    
    Returns:
        Details about the most unfulfilled need
    """
    needs_ctx = ctx.context
    
    await update_needs(ctx)  # Ensure needs are current
    
    highest_drive = -1
    highest_need = None
    
    for name, need in needs_ctx.needs.items():
        if need.drive_strength > highest_drive:
            highest_drive = need.drive_strength
            highest_need = need
    
    if highest_need:
        return {
            "name": highest_need.name,
            "level": highest_need.level,
            "target_level": highest_need.target_level,
            "deficit": highest_need.deficit,
            "drive_strength": highest_need.drive_strength,
            "importance": highest_need.importance,
            "description": highest_need.description
        }
    else:
        return {"name": "none", "drive_strength": 0}

@function_tool
async def get_need_history(
    ctx: RunContextWrapper[NeedsSystemContext], 
    need_name: str, 
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Gets the history for a specific need.
    
    Args:
        need_name: Name of the need to get history for
        limit: Maximum number of history entries to return
        
    Returns:
        List of history entries for the need
    """
    needs_ctx = ctx.context
    
    if need_name not in needs_ctx.needs:
        return []
    
    async with needs_ctx._lock:
        history = needs_ctx.need_history.get(need_name, [])
        return history[-limit:] if limit > 0 else history

@function_tool
async def get_total_drive(ctx: RunContextWrapper[NeedsSystemContext]) -> float:
    """
    Returns the sum of all drive strengths across all needs.
    
    Returns:
        Total drive strength as a float (higher values indicate more unfulfilled needs)
    """
    needs_ctx = ctx.context
    return sum(need.drive_strength for need in needs_ctx.needs.values())

@function_tool
async def reset_need_to_default(
    ctx: RunContextWrapper[NeedsSystemContext], 
    need_name: str
) -> Dict[str, Any]:
    """
    Resets a need to its default values.
    
    Args:
        need_name: Name of the need to reset
        
    Returns:
        Dictionary with details about the reset operation
    """
    needs_ctx = ctx.context
    
    if need_name not in needs_ctx.needs:
        return {"status": "error", "message": f"Unknown need: {need_name}"}
    
    async with needs_ctx._lock:
        # Create a new need state with default values
        default_values = {
            "knowledge": {"level": 0.5, "importance": 0.8, "decay_rate": 0.01, "target_level": 1.0},
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
        
        # Save original for reporting
        original = needs_ctx.needs[need_name].dict()
        
        # Update the need
        needs_ctx.needs[need_name] = NeedState(
            name=need_name,
            level=values["level"],
            importance=values["importance"],
            decay_rate=values["decay_rate"],
            target_level=values["target_level"],
            description=needs_ctx.needs[need_name].description,
            last_updated=datetime.datetime.now()
        )
        
        # Add to history
        _add_history_entry(needs_ctx, need_name, needs_ctx.needs[need_name], "reset_to_default")
        
        return {
            "status": "success",
            "need": need_name,
            "original": original,
            "new": needs_ctx.needs[need_name].dict()
        }

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
    if not needs_ctx.goal_manager: return []
    logger.info(f"Needs [{', '.join(n.name for n in needs_list)}] exceeded drive threshold. Requesting goal creation.")
    created_goal_ids = []
    for need in needs_list:
        priority = 0.5 + (need.drive_strength * 0.5)
        try:
            if hasattr(needs_ctx.goal_manager, 'add_goal'): # Corrected from needs_ctx.goal_manager to ctx.goal_manager
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


@function_tool
async def update_needs_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, float]:
    """
    Tool: Updates all needs by applying decay and returns current drive strengths, potentially triggering goals.
    """
    needs_ctx = ctx.context # This is NeedsSystemContext
    # Call the *actual logic* which now resides in NeedsSystem or a helper
    return await needs_ctx.needs_system_ref._update_needs_logic()


@function_tool
async def satisfy_need_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    name: str,
    amount: float,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tool: Increases the satisfaction level of a need.
    """
    needs_ctx = ctx.context
    # Call the *actual logic*
    return await needs_ctx.needs_system_ref._satisfy_need_logic(name, amount, context)

@function_tool
async def decrease_need_tool_impl( # Renamed from decrease_need to avoid clash
    ctx: RunContextWrapper[NeedsSystemContext], # Takes RunContextWrapper
    need_name: str,
    amount: float,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tool: Decreases the satisfaction level of a need (increases deficit).
    """
    needs_ctx = ctx.context # This is NeedsSystemContext
    reason_provided = reason if reason is not None else "tool_decrease_unspecified"
    # Call the actual logic method on the NeedsSystem instance via the context reference
    return await needs_ctx.needs_system_ref._decrease_need_logic(need_name, amount, reason_provided)


@function_tool
async def get_needs_state_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Dict[str, Any]]:
    """
    Tool: Returns the current state of all needs.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_needs_state_logic() # Call logic method


@function_tool
async def get_needs_by_category_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Tool: Gets needs organized by category.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_needs_by_category_logic() # Call logic method

@function_tool
async def get_most_unfulfilled_need_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> Dict[str, Any]:
    """
    Tool: Gets the need with the highest drive strength.
    """
    needs_ctx = ctx.context
    return await needs_ctx.needs_system_ref._get_most_unfulfilled_need_logic() # Call logic method

@function_tool
async def get_need_history_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    need_name: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Tool: Gets the history for a specific need.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_need_history_logic(need_name, limit) # Call logic method

@function_tool
async def get_total_drive_tool_impl(ctx: RunContextWrapper[NeedsSystemContext]) -> float:
    """
    Tool: Returns the sum of all drive strengths.
    """
    needs_ctx = ctx.context
    return needs_ctx.needs_system_ref._get_total_drive_logic() # Call logic method

@function_tool
async def reset_need_to_default_tool_impl(
    ctx: RunContextWrapper[NeedsSystemContext],
    need_name: str
) -> Dict[str, Any]:
    """
    Tool: Resets a need to its default values.
    """
    needs_ctx = ctx.context
    return await needs_ctx.needs_system_ref._reset_need_to_default_logic(need_name) # Call logic method

class NeedsSystem:
    """Manages Nyx's core digital needs using Agent SDK."""

    def __init__(self, goal_manager=None):
        self.context = NeedsSystemContext(needs_system_instance=self, goal_manager=goal_manager)
        # Make critical attributes directly accessible on self for the _logic methods
        self._lock = self.context._lock
        self.needs = self.context.needs
        self.need_history = self.context.need_history
        self.max_history_per_need = self.context.max_history_per_need
        self.last_update_time = self.context.last_update_time # Initialized by NeedsSystemContext
        self.goal_cooldown = self.context.goal_cooldown # Initialized by NeedsSystemContext

        # *** Agent is now created here, using the _tool_impl FunctionTool objects ***
        self.agent = Agent(
            name="Needs_Manager_Agent", # Give it a unique name if you have multiple agents
            instructions="""You manage the AI's simulated needs, responsible for:
1. Updating needs based on time decay and AI activities
2. Satisfying needs when relevant interactions occur
3. Triggering goals when needs are significantly unfulfilled
4. Providing insights into current need states

Use the appropriate tools to perform these tasks and maintain the AI's underlying need system.
Your available tools include: update_needs_tool_impl, satisfy_need_tool_impl, decrease_need_tool_impl, etc.
When a user asks to "decrease need 'X'", you should use the 'decrease_need_tool_impl' tool with the appropriate arguments.
""",
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
            model_settings=ModelSettings(temperature=0.1)
        )
        logger.info("NeedsSystem initialized with Agent SDK and refactored tools")


    async def _update_needs_logic(self) -> Dict[str, float]:
        async with self._lock:
            now = datetime.datetime.now()
            elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0
            drive_strengths = {}
            needs_to_trigger_goals = []

            if elapsed_hours > 0.001:
                for name, need in self.needs.items():
                    baseline_satisfaction = 0.3
                    decay_amount = need.decay_rate * elapsed_hours
                    if need.level > baseline_satisfaction:
                        need.level = max(baseline_satisfaction, need.level - decay_amount)
                    elif need.level < baseline_satisfaction:
                        need.level = min(baseline_satisfaction, need.level + (decay_amount * 0.5))
                    need.level = max(0.0, min(1.0, need.level))
                    need.last_updated = now
                    drive = need.drive_strength
                    drive_strengths[name] = drive
                    if (len(self.need_history[name]) == 0 or
                        abs(self.need_history[name][-1]['level'] - need.level) > 0.05):
                        _add_history_entry(self.context, name, need, "decay")
                    if drive > self.context.drive_threshold_for_goal:
                        last_triggered = self.goal_cooldown.get(name)
                        if last_triggered and (now - last_triggered) < self.context.goal_cooldown_duration:
                            continue
                        if not self.context.goal_manager or not hasattr(self.context.goal_manager, 'has_active_goal_for_need') or not self.context.goal_manager.has_active_goal_for_need(name):
                            needs_to_trigger_goals.append(need)
                            self.goal_cooldown[name] = now
                self.last_update_time = now
                if needs_to_trigger_goals and self.context.goal_manager:
                    await _trigger_goal_creation(self.context, needs_to_trigger_goals)
            else:
                drive_strengths = {name: need.drive_strength for name, need in self.needs.items()}
            return drive_strengths

    async def _satisfy_need_logic(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self.needs:
            return {"status": "error", "message": f"Unknown need: {name}"}
        async with self._lock:
            need = self.needs[name]
            original_level = need.level
            modified_amount = amount
            satisfaction_multiplier = 1.0
            reason = "standard_satisfaction"
            if name == "control_expression" and context_data:
                difficulty = context_data.get("difficulty_level", 0.3)
                resistance_overcome = context_data.get("resistance_overcome", False)
                intensity = context_data.get("intensity_achieved", 0.5)
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
                modified_amount = amount * satisfaction_multiplier
            need.level = min(need.target_level, need.level + modified_amount)
            need.last_updated = datetime.datetime.now()
            _add_history_entry(self.context, name, need, reason)
            logger.debug(f"Satisfied need '{name}' by {modified_amount:.3f} (Base: {amount:.3f}, Multiplier: {satisfaction_multiplier:.2f}, Reason: {reason}). Level: {original_level:.2f} -> {need.level:.2f}")
            result = NeedSatisfactionResult(name=name, previous_level=original_level, new_level=need.level, change=modified_amount, reason=reason, deficit=need.deficit, drive_strength=need.drive_strength)
            return result.dict()

    async def _decrease_need_logic(self, need_name: str, amount: float, reason_provided: str) -> Dict[str, Any]:
        async with self._lock:
            if need_name not in self.needs:
                logger.warning(f"Attempted to decrease non-existent need: {need_name}")
                return {"success": False, "error": f"Need '{need_name}' not found."}
            if amount < 0:
                logger.warning(f"Decrease amount for need '{need_name}' should be positive, got: {amount}. Using absolute value.")
                amount = abs(amount)

            need = self.needs[need_name]
            original_level = need.level
            # Decrease level (increase deficit), clamp at 0
            need.level = max(0.0, need.level - amount)
            need.last_updated = datetime.datetime.now()
            _add_history_entry(self.context, need_name, need, reason_provided)
            logger.info(f"Need '{need_name}' level decreased from {original_level:.2f} to {need.level:.2f}. Reason: {reason_provided}")
            return {
                "success": True, "name": need_name, "previous_level": original_level,
                "new_level": need.level, "change": -amount, "reason": reason_provided, # Change is negative
                "deficit": need.deficit, "drive_strength": need.drive_strength
            }

    def _get_needs_state_logic(self) -> Dict[str, Dict[str, Any]]:
        return {name: {**need.dict(exclude={'last_updated'}), 'last_updated': need.last_updated.isoformat(), 'deficit': need.deficit, 'drive_strength': need.drive_strength} for name, need in self.needs.items()}

    def _get_needs_by_category_logic(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result = {}
        all_states = self._get_needs_state_logic()
        for category, need_names in self.context.need_categories.items():
            result[category] = {name: all_states[name] for name in need_names if name in all_states}
        return result

    async def _get_most_unfulfilled_need_logic(self) -> Dict[str, Any]:
        await self._update_needs_logic() # Ensure needs are current
        highest_drive = -1
        highest_need = None
        for name, need in self.needs.items():
            if need.drive_strength > highest_drive:
                highest_drive = need.drive_strength
                highest_need = need
        if highest_need:
            return {
                "name": highest_need.name, "level": highest_need.level, "target_level": highest_need.target_level,
                "deficit": highest_need.deficit, "drive_strength": highest_need.drive_strength,
                "importance": highest_need.importance, "description": highest_need.description
            }
        return {"name": "none", "drive_strength": 0}

    def _get_need_history_logic(self, need_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        if need_name not in self.needs: return []
        history = self.need_history.get(need_name, [])
        return history[-limit:] if limit > 0 else history

    def _get_total_drive_logic(self) -> float:
        return sum(need.drive_strength for need in self.needs.values())

    async def _reset_need_to_default_logic(self, need_name: str) -> Dict[str, Any]:
        if need_name not in self.needs:
            return {"status": "error", "message": f"Unknown need: {need_name}"}
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
            original = self.needs[need_name].dict()
            self.needs[need_name] = NeedState(
                name=need_name, level=values["level"], importance=values["importance"],
                decay_rate=values["decay_rate"], target_level=values["target_level"],
                description=self.needs[need_name].description, last_updated=datetime.datetime.now()
            )
            _add_history_entry(self.context, need_name, self.needs[need_name], "reset_to_default")
            return {"status": "success", "need": need_name, "original": original, "new": self.needs[need_name].dict()}

    # --- Public API Methods (now call the agent) ---
    async def update_needs(self) -> Dict[str, float]:
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": "Update all needs and trigger goals if necessary."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_update", "type": "function", "function": {
                     "name": "update_needs_tool_impl",
                     "arguments": json.dumps({})
                 }}
             ]}],
            context=self.context
        )
        final_output = result.final_output
        return final_output if isinstance(final_output, dict) else {}


    async def satisfy_need(self, name: str, amount: float, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool_args = {"name": name, "amount": amount, "context": context_data or {}}
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": f"Satisfy need '{name}'."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_satisfy", "type": "function", "function": {
                     "name": "satisfy_need_tool_impl",
                     "arguments": json.dumps(tool_args)
                 }}
             ]}],
            context=self.context
        )
        final_output = result.final_output
        return final_output if isinstance(final_output, dict) else {"status": "error", "message": "Agent did not return expected dict for satisfy_need."}

    async def decrease_need(self, name: str, amount: float, reason: Optional[str] = None) -> Dict[str, Any]:
        reason_provided = reason if reason is not None else "generic_decrease_api_call"
        tool_args = {"need_name": name, "amount": amount, "reason": reason_provided}
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": f"Decrease need '{name}'."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_decrease", "type": "function", "function": {
                     "name": "decrease_need_tool_impl",
                     "arguments": json.dumps(tool_args)
                 }}
             ]}],
            context=self.context
        )
        final_output = result.final_output
        return final_output if isinstance(final_output, dict) else {"status": "error", "message": "Agent did not return expected dict for decrease_need."}

    async def get_most_unfulfilled_need(self) -> Dict[str, Any]:
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": "Find the most unfulfilled need."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_get_most", "type": "function", "function": {
                     "name": "get_most_unfulfilled_need_tool_impl",
                     "arguments": json.dumps({})
                 }}
             ]}],
            context=self.context)
        final_output = result.final_output
        return final_output if isinstance(final_output, dict) else {}

    async def get_need_history(self, need_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        tool_args = {"need_name": need_name, "limit": limit}
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": f"Get history for need '{need_name}'."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_history", "type": "function", "function": {
                     "name": "get_need_history_tool_impl",
                     "arguments": json.dumps(tool_args)
                 }}
             ]}],
            context=self.context)
        final_output = result.final_output
        return final_output if isinstance(final_output, list) else []

    async def reset_need_to_default(self, need_name: str) -> Dict[str, Any]:
        tool_args = {"need_name": need_name}
        result = await Runner.run(
            self.agent,
             [{"role": "user", "content": f"Reset need '{need_name}'."},
             {"role": "assistant", 
              "content": "",  # <--- CHANGE: None to ""
              "tool_calls": [
                 {"id": "call_reset", "type": "function", "function": {
                     "name": "reset_need_to_default_tool_impl",
                     "arguments": json.dumps(tool_args)
                 }}
             ]}],
            context=self.context)
        final_output = result.final_output
        return final_output if isinstance(final_output, dict) else {}


    # Synchronous wrappers for compatibility, if strictly needed, but discourage for agent interactions
    def get_needs_state(self) -> Dict[str, Dict[str, Any]]:
        logger.warning("Synchronous get_needs_state called; prefer async agent interaction or direct logic call if within NeedsSystem.")
        return self._get_needs_state_logic() # Calls internal logic directly for sync access

    def get_total_drive(self) -> float:
        logger.warning("Synchronous get_total_drive called; prefer async agent interaction or direct logic call if within NeedsSystem.")
        return self._get_total_drive_logic()

    def get_needs_by_category(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        logger.warning("Synchronous get_needs_by_category called; prefer async agent interaction or direct logic call if within NeedsSystem.")
        return self._get_needs_by_category_logic()

 
