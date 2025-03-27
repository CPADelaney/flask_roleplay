# nyx/core/goal_system.py

import logging
import datetime
import uuid
import asyncio
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator

# Assuming Agent SDK components are available
try:
    from agents import Agent, Runner, ModelSettings, trace
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    # Define dummy Agent/Runner if SDK not present, for basic structure
    class Agent: pass
    class Runner: pass
    class ModelSettings: pass
    def trace(workflow_name, group_id):
        class DummyTrace:
            def __enter__(self): pass
            def __aenter__(self): pass
            def __exit__(self, *args): pass
            def __aexit__(self, *args): pass
        return DummyTrace()

logger = logging.getLogger(__name__)

# --- Data Models ---

class GoalStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    description: str
    action: str # Name of a method callable on NyxBrain (e.g., 'query_knowledge', 'generate_response')
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field("pending", description="pending, active, completed, failed, skipped")
    result: Optional[Any] = Field(None, description="Result from action execution")
    error: Optional[str] = Field(None, description="Error message if step failed")
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

    @field_validator('action')
    @classmethod
    def action_must_be_valid(cls, v):
        # Basic validation - could be expanded to check against known brain methods
        if not v or not isinstance(v, str):
            raise ValueError('Action must be a non-empty string')
        return v

class Goal(BaseModel):
    id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    description: str
    status: str = Field("pending", description="pending, active, completed, failed, abandoned")
    priority: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field("unknown", description="Originator (NeedsSystem, User, MetaCore, etc.)")
    associated_need: Optional[str] = None # Link to a need in NeedsSystem
    creation_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completion_time: Optional[datetime.datetime] = None
    deadline: Optional[datetime.datetime] = None
    plan: List[GoalStep] = Field(default_factory=list)
    current_step_index: int = Field(0, description="Index of the next step to execute")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Log of step execution attempts")
    last_error: Optional[str] = None

# --- Goal Manager Class ---

class GoalManager:
    """Manages goals, planning, and execution oversight for Nyx."""

    def __init__(self, brain_reference=None):
        """
        Args:
            brain_reference: Reference to the main NyxBrain instance for action execution.
        """
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set() # IDs of goals currently being executed
        self.brain = brain_reference # Set via set_brain_reference if needed later
        self.max_concurrent_goals = 1 # Limit concurrency for simplicity initially
        self.planning_agent = self._create_planning_agent()
        self.trace_group_id = "NyxGoalManagement"
        self._lock = asyncio.Lock() # Lock for managing goals dict safely

        logger.info("GoalManager initialized.")

    def set_brain_reference(self, brain):
        """Set the reference to the main NyxBrain after initialization."""
        self.brain = brain
        logger.info("NyxBrain reference set for GoalManager.")

    def _create_planning_agent(self) -> Optional[Agent]:
        """Creates the agent responsible for generating plans for goals."""
        if not AGENT_SDK_AVAILABLE:
            logger.warning("Agent SDK not available. Planning agent cannot be created. Plans must be provided manually.")
            return None
        try:
            # Define tools available to the planner (methods on NyxBrain)
            # This list needs to be maintained based on NyxBrain capabilities
            available_actions = [
                "process_input", "generate_response", # Basic interaction
                "query_knowledge", "add_knowledge", # Knowledge Core
                "retrieve_memories", "add_memory", "create_reflection", "create_abstraction", # Memory Core
                "execute_procedure", "add_procedure", # Procedural Memory
                "reason_causal", "perform_intervention", "reason_counterfactually", # Reasoning Core
                "update_emotion", "process_emotional_input", # Emotional Core
                "process_sensory_input", "add_expectation", # Multimodal/Somatic
                "monitor_systems", "evaluate_cognition", "select_strategy", # Meta Core
                "generate_prediction", "evaluate_prediction", # Prediction Engine
                "explore_knowledge", # Curiosity/Knowledge
                "express_attraction",        # Express positive affinity towards user/entity
                "initiate_intimate_interaction", # Propose or start more intimate dialogue/action
                "deepen_connection",       # Actions focused on building trust/familiarity/intimacy
                "express_desire",          # Communicate wanting for closeness, intimacy, or specific action
                "respond_to_intimacy",     # Process and react to user's intimate actions/words
                "simulate_physical_touch", # Trigger DSS for simulated touch (requires consent/context checks)
                "seek_gratification",        # Goal-driven action towards achieving satisfaction (needs careful planning)
                "process_gratification_outcome", # Handle the state changes after gratification event
                # Dominance Actions
                "analyze_user_state_for_dominance", # Assess vulnerability, willingness, resistance
                "select_dominance_tactic",       # Choose psych, emotional, physical (sim) approach
                "issue_command",                 # Give a directive
                "evaluate_compliance",           # Check if user response complies
                "apply_consequence_simulated",   # Describe a consequence (reward/punishment in context)
                "praise_submission",             # Reinforce compliant behavior
                "increase_control_intensity",    # Escalate dominance
                "trigger_dominance_gratification", # Internal action marking successful dominance sequence
                "express_satisfaction"          # Express satisfaction after successful dominance
                # Add more core NyxBrain methods here...
            ]
            # Create better descriptions based on function docstrings if possible
            tool_descriptions = "\n".join([f"- {action}" for action in available_actions])

            return Agent(
                name="Goal Planner Agent",
                instructions=f"""You are a planner agent for the Nyx AI. Your task is to break down a high-level goal description into a sequence of concrete, actionable steps using Nyx's available actions.

                Available Actions Nyx can perform (these are methods on the main system):
                {tool_descriptions}

                For a given goal, create a JSON list of steps. Each step MUST be a JSON object with keys:
                - "description": (string) A human-readable description of what the step does.
                - "action": (string) The name of the action to perform (MUST be one from the available list).
                - "parameters": (object) A dictionary of parameters required by the action. Use placeholders like "$step_N.result.key" to refer to outputs from previous steps (where N is the 1-based index of the step).

                Example Output Format:
                ```json
                [
                  {{"description": "Retrieve relevant memories about topic X", "action": "retrieve_memories", "parameters": {{"query": "topic X", "limit": 5}}}},
                  {{"description": "Analyze the memories using causal reasoning", "action": "reason_causal", "parameters": {{"input_data": "$step_1.result"}}}},
                  {{"description": "Generate a response based on the analysis", "action": "generate_response", "parameters": {{"prompt": "Based on causal analysis ($step_2.result.summary), respond to user.", "context": "..."}}}}
                ]
                ```

                Ensure the plan is logical, sequential, and likely to achieve the goal. Be precise with action names and expected parameters. If an action needs data from a previous step, use the '$step_N.result...' format correctly. The output MUST be only the valid JSON list of steps, nothing else.
                """,
                model="gpt-4o", # Or another capable model like Claude 3.5 Sonnet
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.1), # Enforce JSON, low temp for planning
                output_type=List[Dict] # Expecting a list of step dicts
            )
        except Exception as e:
            logger.error(f"Error creating planning agent: {e}")
            return None

    async def add_goal(self, description: str, priority: float = 0.5, source: str = "unknown",
                     associated_need: Optional[str] = None, plan: Optional[List[Dict]] = None) -> str:
        """Adds a new goal, optionally generating a plan if none is provided."""
        if not description:
            raise ValueError("Goal description cannot be empty.")

        async with self._lock:
            goal = Goal(
                description=description,
                priority=priority,
                source=source,
                associated_need=associated_need,
                plan=[] # Start with empty plan, generate/add later
            )
            self.goals[goal.id] = goal

        logger.info(f"Added goal '{goal.id}': {description} (Priority: {priority:.2f}, Source: {source})")

        plan_steps = None
        if plan: # If plan is directly provided
            try:
                plan_steps = [GoalStep(**step_data) for step_data in plan]
            except Exception as e:
                 logger.error(f"Invalid plan structure provided for goal '{goal.id}': {e}. Plan generation required.")
                 plan_steps = None # Fallback to generation

        elif self.planning_agent: # If no plan provided and agent exists, generate it
            # Trigger plan generation outside the lock to avoid holding it during LLM call
            asyncio.create_task(self.generate_plan_for_goal(goal.id))
        else:
            logger.warning(f"Goal '{goal.id}' added without a plan and no planning agent available.")
            # Goal remains pending until a plan is added manually or agent becomes available

        # If plan was provided and valid, update the goal
        if plan_steps:
            async with self._lock:
                if goal.id in self.goals: # Check if goal still exists
                    self.goals[goal.id].plan = plan_steps
                    self.goals[goal.id].status = "pending" # Ready to be activated

        return goal.id

    async def generate_plan_for_goal(self, goal_id: str) -> bool:
        """Generates and assigns a plan for a goal using the planning agent."""
        async with self._lock:
            if goal_id not in self.goals:
                logger.error(f"Cannot generate plan: Goal {goal_id} not found.")
                return False
            goal = self.goals[goal_id]
            if goal.plan: # Don't overwrite existing plan
                 logger.info(f"Goal '{goal_id}' already has a plan.")
                 return True

        if not self.planning_agent:
            logger.warning(f"Cannot generate plan for goal '{goal_id}': Planning agent not available.")
            return False

        logger.info(f"Generating plan for goal '{goal.id}': {goal.description}")

        try:
            with trace(workflow_name="GenerateGoalPlan", group_id=self.trace_group_id):
                prompt = f"Generate a plan as a JSON list of steps to achieve the following goal for an AI named Nyx: {goal.description}"
                # Add more context? Like available systems? Let agent instructions handle this for now.

                result = await Runner.run(
                    self.planning_agent,
                    prompt
                )

                # Agent SDK v3 should parse the JSON automatically if response_format and output_type are set
                if isinstance(result.final_output, list):
                    plan_data = result.final_output
                    plan_steps = [GoalStep(**step_data) for step_data in plan_data]
                    async with self._lock:
                        if goal_id in self.goals: # Re-check existence
                            self.goals[goal_id].plan = plan_steps
                            self.goals[goal_id].current_step_index = 0 # Reset index
                            if self.goals[goal_id].status != "failed": # Don't reactivate failed goals
                                 self.goals[goal_id].status = "pending"
                            logger.info(f"Generated plan with {len(plan_steps)} steps for goal '{goal.id}'.")
                    return True
                else:
                    # Handle unexpected output format
                    error_msg = f"Planning agent returned invalid plan format: {type(result.final_output)}. Output: {str(result.final_output)[:500]}"
                    logger.error(error_msg)
                    await self.update_goal_status(goal_id, "failed", error=error_msg)
                    return False

        except Exception as e:
            error_msg = f"Error generating plan for goal '{goal.id}': {e}"
            logger.exception(error_msg) # Log full traceback
            await self.update_goal_status(goal_id, "failed", error=error_msg)
            return False

    def get_prioritized_goals(self) -> List[Goal]:
        """Returns active and pending goals sorted by priority."""
        asyncio.run(self._lock.acquire()) # Use run for sync context
        try:
            active_goals = [g for g in self.goals.values() if g.status in ["pending", "active"]]
        finally:
            self._lock.release()

        # Add urgency factor? Higher priority for older pending goals?
        now = datetime.datetime.now()
        def sort_key(g: Goal) -> float:
            age_penalty = (now - g.creation_time).total_seconds() / (3600 * 24) # Age in days
            # Increase priority slightly for older pending goals, decrease for older active ones
            status_boost = 0.05 if g.status == "pending" else -0.05
            return g.priority + (age_penalty * 0.01 * status_boost) # Small age effect

        active_goals.sort(key=sort_key, reverse=True)
        return active_goals

    def select_active_goal(self) -> Optional[str]:
        """Selects the highest priority goal to work on, respecting concurrency limits."""
        # This method needs to acquire the lock internally or be called within a lock
        # Let's make it acquire the lock.
        # Note: This sync lock acquisition might block the event loop briefly.
        # Consider redesign if this becomes a bottleneck.
        asyncio.run(self._lock.acquire())
        try:
            prioritized = self.get_prioritized_goals() # This gets called within the lock
            active_count = len(self.active_goals)
            selected_goal_id = None

            if not prioritized:
                self.active_goals.clear() # No goals left to be active
                return None

            # Check if any currently active goals are finished or failed
            finished_active = {gid for gid in self.active_goals if gid not in self.goals or self.goals[gid].status not in ["active", "pending"]}
            self.active_goals -= finished_active
            active_count = len(self.active_goals)

            # Find the highest priority goal that can be activated
            for goal in prioritized:
                if goal.id in self.active_goals:
                    # If it's already active, it's a candidate to continue
                    selected_goal_id = goal.id
                    break # Keep executing the highest priority *active* goal first
                elif goal.status == "pending" and active_count < self.max_concurrent_goals and goal.plan:
                    # Activate this pending goal if concurrency allows and it has a plan
                    selected_goal_id = goal.id
                    goal.status = "active"
                    goal.current_step_index = 0 # Ensure starting from step 0
                    self.active_goals.add(goal.id)
                    logger.info(f"Activated goal '{goal.id}' (Priority: {goal.priority:.2f})")
                    break

            return selected_goal_id
        finally:
            self._lock.release()

    async def execute_next_step(self) -> Optional[Dict[str, Any]]:
        """Selects and executes the next step of the highest priority active goal."""
        goal_id = self.select_active_goal() # Selection handles concurrency & prioritization

        if goal_id is None:
            #logger.debug("No suitable active goal found to execute.")
            return None

        async with self._lock: # Lock for modifying goal state
            if goal_id not in self.goals:
                logger.warning(f"Selected goal '{goal_id}' disappeared before execution.")
                self.active_goals.discard(goal_id)
                return None
            goal = self.goals[goal_id]

            if goal.status != "active" or not goal.plan:
                logger.warning(f"Goal '{goal_id}' is not ready for execution (Status: {goal.status}, Has Plan: {bool(goal.plan)}).")
                self.active_goals.discard(goal_id)
                return None

            step_index = goal.current_step_index
            if not (0 <= step_index < len(goal.plan)):
                logger.error(f"Goal '{goal.id}' has invalid step index {step_index}. Failing goal.")
                await self.update_goal_status(goal_id, "failed", error="Invalid plan state")
                self.active_goals.discard(goal_id)
                return None

            step = goal.plan[step_index]

            if step.status != "pending":
                logger.warning(f"Step '{step.step_id}' for goal '{goal.id}' is not pending (Status: {step.status}). Skipping.")
                goal.current_step_index += 1
                if goal.current_step_index >= len(goal.plan):
                    await self.update_goal_status(goal.id, "completed", result="Plan finished after skipping steps.")
                    self.active_goals.discard(goal.id)
                return {"skipped_step": step.model_dump(), "goal_id": goal_id} # Indicate skip

        # --- Execute Step (Outside Lock) ---
        if not self.brain:
            error_msg = "NyxBrain reference not set in GoalManager. Cannot execute actions."
            logger.error(error_msg)
            await self.update_goal_status(goal_id, "failed", error=error_msg) # Update status needs lock
            return None

        logger.info(f"Executing step '{step.step_id}' for goal '{goal.id}': Action={step.action}")
        step_result = None
        step_error = None
        step_start_time = datetime.datetime.now()

        try:
            with trace(workflow_name="ExecuteGoalStep", group_id=self.trace_group_id):
                action_method = getattr(self.brain, step.action, None)
                if not (action_method and callable(action_method)):
                    raise ValueError(f"Action '{step.action}' not found or not callable on NyxBrain.")

                resolved_params = await self._resolve_step_parameters(goal_id, step.parameters) # Pass goal_id for context
                logger.debug(f"Executing {step.action} with params: {resolved_params}")

                step_result = await action_method(**resolved_params)
                step_status = "completed"
                logger.info(f"Step '{step.step_id}' completed successfully.")

        except Exception as e:
            step_error = str(e)
            step_status = "failed"
            logger.exception(f"Error executing step '{step.step_id}' for goal '{goal.id}': {e}")

        step_end_time = datetime.datetime.now()

        # --- Update Goal State (With Lock) ---
        async with self._lock:
            # Re-fetch goal in case it was modified
            if goal_id not in self.goals: return None
            goal = self.goals[goal_id]
            # Ensure we are updating the correct step
            if goal.current_step_index == step_index and step.step_id == goal.plan[step_index].step_id:
                step = goal.plan[step_index] # Get the official step object again
                step.status = step_status
                step.result = step_result
                step.error = step_error
                step.start_time = step_start_time
                step.end_time = step_end_time
                goal.execution_history.append({
                    "step_id": step.step_id,
                    "action": step.action,
                    "status": step_status,
                    "timestamp": step_end_time.isoformat(),
                    "duration": (step_end_time - step_start_time).total_seconds(),
                    "error": step_error
                })

                if step_status == "completed":
                    goal.current_step_index += 1
                    if goal.current_step_index >= len(goal.plan):
                        await self.update_goal_status(goal.id, "completed", result=step_result)
                        self.active_goals.discard(goal.id)
                    # else: goal remains active for next step
                elif step_status == "failed":
                    await self.update_goal_status(goal.id, "failed", error=f"Step '{step.step_id}' failed: {step_error}")
                    self.active_goals.discard(goal.id)

            else:
                 logger.warning(f"State mismatch while updating step result for goal {goal_id}. Step index may have changed.")

        return {"executed_step": step.model_dump(), "goal_id": goal_id} # Return executed step info


    async def _resolve_step_parameters(self, goal_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
         """Resolves parameter placeholders like '$step_1.result' within the context of a specific goal."""
         # Needs read-only access to the goal state, potentially under lock if strict consistency needed
         # For simplicity, let's assume read access is safe enough or caller handles locking if needed.
         # Fetching goal state without lock might lead to using slightly stale data for resolution.
         goal = self.goals.get(goal_id)
         if not goal:
              logger.error(f"Cannot resolve parameters: Goal {goal_id} not found.")
              return parameters # Return original if goal gone

         resolved_params = {}
         for key, value in parameters.items():
             if isinstance(value, str) and value.startswith("$step_"):
                 parts = value[1:].split('.')
                 source_step_index_str = parts[0][4:] # Get the number after 'step_'
                 path = parts[1:]

                 try:
                     # Step indices in plans are 0-based, but placeholders likely use 1-based. Adjust.
                     source_step_index = int(source_step_index_str) - 1
                 except ValueError:
                      logger.warning(f"Invalid step index '{source_step_index_str}' in parameter placeholder '{value}' for goal '{goal.id}'.")
                      resolved_params[key] = None
                      continue

                 if 0 <= source_step_index < len(goal.plan):
                     source_step = goal.plan[source_step_index]
                     if source_step.status == "completed" and source_step.result is not None:
                         try:
                             current_value = source_step.result
                             for part in path:
                                 if isinstance(current_value, dict):
                                     current_value = current_value.get(part)
                                 elif isinstance(current_value, BaseModel): # Check for Pydantic models
                                      current_value = getattr(current_value, part, None)
                                 elif hasattr(current_value, part):
                                     current_value = getattr(current_value, part)
                                 else:
                                     try: # Try index access for lists/tuples
                                         idx = int(part)
                                         if isinstance(current_value, (list, tuple)) and -len(current_value) <= idx < len(current_value):
                                             current_value = current_value[idx]
                                         else:
                                             current_value = None; break
                                     except (ValueError, TypeError, IndexError):
                                         current_value = None; break
                                 if current_value is None: break

                             resolved_params[key] = current_value
                             if current_value is None:
                                 logger.warning(f"Path '{'.'.join(path)}' resolved to None for parameter placeholder '{value}' in goal '{goal.id}'.")

                         except Exception as e:
                             logger.error(f"Error resolving path '{'.'.join(path)}' for parameter '{value}': {e}")
                             resolved_params[key] = None
                     else:
                         logger.warning(f"Could not resolve parameter '{value}' for goal '{goal.id}'. Source step '{source_step.step_id}' not completed or has no result.")
                         resolved_params[key] = None
                 else:
                     logger.warning(f"Invalid source step index {source_step_index + 1} for parameter placeholder '{value}' in goal '{goal.id}'.")
                     resolved_params[key] = None
             else:
                 resolved_params[key] = value
         return resolved_params


    async def update_goal_status(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None):
        """Updates the status of a goal and notifies relevant systems."""
        async with self._lock: # Ensure atomic update of goal state
            if goal_id not in self.goals:
                logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
                return

            goal = self.goals[goal_id]
            old_status = goal.status
            if old_status == status: return # No change

            goal.status = status
            goal.last_error = error

            if status in ["completed", "failed", "abandoned"]:
                goal.completion_time = datetime.datetime.now()
                self.active_goals.discard(goal_id) # Remove from active set

            logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")

            # --- Notifications (Keep outside lock if they involve awaits/long ops) ---
            need_name = goal.associated_need
            source = goal.source
            priority = goal.priority

        # Notify NeedsSystem
        if need_name and self.brain and self.brain.needs_system:
            try:
                if status == "completed":
                    satisfaction_amount = priority * 0.3 + 0.1 # Base + priority bonus
                    await self.brain.needs_system.satisfy_need(need_name, satisfaction_amount)
                elif status == "failed":
                    decrease_amount = priority * 0.1 # Small decrease for failure
                    await self.brain.needs_system.decrease_need(need_name, decrease_amount)
            except Exception as e:
                logger.error(f"Error notifying NeedsSystem about goal {goal_id}: {e}")

        # Notify RewardSystem
        if self.brain and self.brain.reward_system:
            try:
                reward_value = 0.0
                if status == "completed": reward_value = priority * 0.6 # Higher reward for completion
                elif status == "failed": reward_value = -priority * 0.4 # Punish failure
                elif status == "abandoned": reward_value = -0.1 # Small punishment for abandoning

                if abs(reward_value) > 0.05:
                    from nyx.core.reward_system import RewardSignal # Local import
                    reward_signal = RewardSignal(
                        value=reward_value, source="GoalManager",
                        context={"goal_id": goal_id, "goal_description": goal.description, "outcome": status, "associated_need": need_name},
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    # Use create_task for non-blocking reward processing
                    asyncio.create_task(self.brain.reward_system.process_reward_signal(reward_signal))
            except Exception as e:
                logger.error(f"Error notifying RewardSystem about goal {goal_id}: {e}")

        # Notify MetaCore? (e.g., for learning about goal success/failure rates)
        if self.brain and self.brain.meta_core:
             try:
                 # meta_core might have a method like 'record_goal_outcome'
                 if hasattr(self.brain.meta_core, 'record_goal_outcome'):
                     asyncio.create_task(self.brain.meta_core.record_goal_outcome(goal.model_dump()))
             except Exception as e:
                  logger.error(f"Error notifying MetaCore about goal {goal_id}: {e}")


    async def abandon_goal(self, goal_id: str, reason: str):
        """Abandons an active or pending goal."""
        logger.info(f"Abandoning goal '{goal_id}': {reason}")
        await self.update_goal_status(goal_id, "abandoned", error=reason)

    def has_active_goal_for_need(self, need_name: str) -> bool:
        """Checks if there's an active goal associated with a specific need."""
        # Needs lock for safe iteration
        asyncio.run(self._lock.acquire())
        try:
            return any(g.status == "active" and g.associated_need == need_name for g in self.goals.values())
        finally:
            self._lock.release()

    async def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status and plan of a specific goal."""
        async with self._lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                # Return a copy, exclude potentially large history for status checks
                return goal.model_dump(exclude={'execution_history', 'plan'}) | {'plan_step_count': len(goal.plan)}
            return None

    async def get_all_goals(self, status_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
         """Gets all goals, optionally filtered by status."""
         async with self._lock:
             filtered_goals = []
             for goal in self.goals.values():
                  if status_filter is None or goal.status in status_filter:
                       # Return summaries, exclude large fields
                       summary = goal.model_dump(exclude={'execution_history', 'plan'})
                       summary['plan_step_count'] = len(goal.plan)
                       summary['current_step_description'] = goal.plan[goal.current_step_index].description if 0 <= goal.current_step_index < len(goal.plan) else "N/A"
                       filtered_goals.append(summary)
             # Optionally sort results here if needed
             filtered_goals.sort(key=lambda g: g['priority'], reverse=True)
         return filtered_goals
