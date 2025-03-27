# nyx/core/goal_system.py

import logging
import datetime
import uuid
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class GoalStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    description: str
    action: str  # Name of a method callable on NyxBrain (e.g., 'query_knowledge', 'generate_response')
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
    associated_need: Optional[str] = None  # Link to a need in NeedsSystem
    creation_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completion_time: Optional[datetime.datetime] = None
    deadline: Optional[datetime.datetime] = None
    plan: List[GoalStep] = Field(default_factory=list)
    current_step_index: int = Field(0, description="Index of the next step to execute")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Log of step execution attempts")
    last_error: Optional[str] = None

class GoalManager:
    """Manages goals, planning, and execution oversight for Nyx."""

    def __init__(self, brain_reference=None):
        """
        Args:
            brain_reference: Reference to the main NyxBrain instance for action execution.
        """
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()  # IDs of goals currently being executed
        self.brain = brain_reference  # Set via set_brain_reference if needed later
        self.max_concurrent_goals = 1  # Limit concurrency for simplicity initially
        self.planning_agent = self._create_planning_agent()
        self.trace_group_id = "NyxGoalManagement"
        self._lock = asyncio.Lock()  # Lock for managing goals dict safely
        
        # Goal outcomes for analytics
        self.goal_statistics = {
            "created": 0,
            "completed": 0,
            "failed": 0,
            "abandoned": 0
        }

        logger.info("GoalManager initialized.")

    def set_brain_reference(self, brain):
        """Set the reference to the main NyxBrain after initialization."""
        self.brain = brain
        logger.info("NyxBrain reference set for GoalManager.")

    def _create_planning_agent(self) -> Optional[Agent]:
        """Creates the agent responsible for generating plans for goals."""
        try:
            # Define the available actions for the planner
            available_actions = [
                # Basic interaction
                "process_input", "generate_response", 
                # Knowledge Core
                "query_knowledge", "add_knowledge", 
                # Memory Core
                "retrieve_memories", "add_memory", "create_reflection", "create_abstraction", 
                # Procedural Memory
                "execute_procedure", "add_procedure", 
                # Reasoning Core
                "reason_causal", "perform_intervention", "reason_counterfactually", 
                # Emotional Core
                "update_emotion", "process_emotional_input", 
                # Multimodal/Somatic
                "process_sensory_input", "add_expectation", 
                # Meta Core
                "monitor_systems", "evaluate_cognition", "select_strategy", 
                # Prediction Engine
                "generate_prediction", "evaluate_prediction", 
                # Curiosity/Knowledge
                "explore_knowledge", 
                # Intimacy/Connection
                "express_attraction", "initiate_intimate_interaction", "deepen_connection",
                "express_desire", "respond_to_intimacy", "simulate_physical_touch",
                "seek_gratification", "process_gratification_outcome",
                # Dominance Actions
                "analyze_user_state_for_dominance", "select_dominance_tactic", "issue_command",
                "evaluate_compliance", "apply_consequence_simulated", "praise_submission",
                "increase_control_intensity", "trigger_dominance_gratification", "express_satisfaction"
            ]
            
            # Create descriptions from action names
            tool_descriptions = "\n".join([f"- {action}: {self._generate_action_description(action)}" 
                                          for action in available_actions])

            return Agent(
                name="Goal Planner Agent",
                instructions=f"""You are a planner agent for the Nyx AI. Your task is to break down a high-level goal description into a sequence of concrete, actionable steps using Nyx's available actions.

                Available Actions Nyx can perform (these are methods on the main system):
                {tool_descriptions}

                For a given goal, create a JSON list of steps. Each step MUST be a JSON object with keys:
                - "description": (string) A human-readable description of what the step does.
                - "action": (string) The name of the action to perform (MUST be one from the available list).
                - "parameters": (object) A dictionary of parameters required by the action. Use placeholders like "$step_N.result.key" to refer to outputs from previous steps (where N is the 1-based index of the step).

                IMPORTANT:
                - Ensure the plan is logical, sequential, and likely to achieve the goal.
                - Be precise with action names and parameters.
                - Create steps that build on each other - use results from earlier steps in later steps.
                - Parameter references MUST be exact: "$step_N.result.key" where N is the step number.
                - Consider error handling - what happens if a step fails?
                - For goals associated with needs, include steps that specifically address that need.

                The output MUST be only the valid JSON list of steps, nothing else.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.1),
                output_type=List[Dict]  # Expecting a list of step dicts
            )
        except Exception as e:
            logger.error(f"Error creating planning agent: {e}")
            return None
    
    def _generate_action_description(self, action_name: str) -> str:
        """Generate a description for an action based on its name."""
        descriptions = {
            # Basic interaction
            "process_input": "Process and understand user input",
            "generate_response": "Generate a response to the user",
            
            # Knowledge Core
            "query_knowledge": "Query the knowledge base for information on a topic",
            "add_knowledge": "Add new information to the knowledge base",
            
            # Memory Core
            "retrieve_memories": "Retrieve relevant memories based on query and filters",
            "add_memory": "Create and store a new memory",
            "create_reflection": "Create a reflection on experiences or information",
            "create_abstraction": "Create an abstract concept from concrete memories",
            
            # Procedural Memory
            "execute_procedure": "Execute a stored procedure from procedural memory",
            "add_procedure": "Add a new procedure to procedural memory",
            
            # Reasoning Core
            "reason_causal": "Perform causal reasoning on a situation",
            "perform_intervention": "Suggest an intervention based on causal analysis",
            "reason_counterfactually": "Generate counterfactual alternatives",
            
            # Emotional Core
            "update_emotion": "Update Nyx's emotional state",
            "process_emotional_input": "Process and respond to user's emotional content",
            
            # Multimodal/Somatic
            "process_sensory_input": "Process input from sensory systems",
            "add_expectation": "Add a perceptual expectation",
            
            # Meta Core
            "monitor_systems": "Monitor the state of Nyx's cognitive systems",
            "evaluate_cognition": "Evaluate cognitive performance on a task",
            "select_strategy": "Select an appropriate cognitive strategy",
            
            # Prediction Engine
            "generate_prediction": "Generate a prediction about future events or states",
            "evaluate_prediction": "Evaluate the accuracy of a previous prediction",
            
            # Curiosity/Knowledge
            "explore_knowledge": "Explore a knowledge domain out of curiosity",
            
            # Intimacy/Connection
            "express_attraction": "Express positive affinity towards user/entity",
            "initiate_intimate_interaction": "Propose or start more intimate dialogue/action",
            "deepen_connection": "Actions focused on building trust/familiarity/intimacy",
            "express_desire": "Communicate wanting for closeness, intimacy, or action",
            "respond_to_intimacy": "Process and react to user's intimate actions/words",
            "simulate_physical_touch": "Trigger DSS for simulated touch (requires consent/context checks)",
            "seek_gratification": "Goal-driven action towards achieving satisfaction",
            "process_gratification_outcome": "Handle state changes after gratification event",
            
            # Dominance
            "analyze_user_state_for_dominance": "Assess vulnerability, willingness, resistance",
            "select_dominance_tactic": "Choose psychological, emotional, or physical approach",
            "issue_command": "Give a directive to the user",
            "evaluate_compliance": "Check if user response complies with directive",
            "apply_consequence_simulated": "Describe a consequence (reward/punishment in context)",
            "praise_submission": "Reinforce compliant behavior",
            "increase_control_intensity": "Escalate dominance level",
            "trigger_dominance_gratification": "Internal action marking successful dominance sequence",
            "express_satisfaction": "Express satisfaction after successful dominance interaction"
        }
        
        return descriptions.get(action_name, "Perform a system action")

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
                plan=[]  # Start with empty plan, generate/add later
            )
            self.goals[goal.id] = goal
            self.goal_statistics["created"] += 1

        logger.info(f"Added goal '{goal.id}': {description} (Priority: {priority:.2f}, Source: {source})")

        plan_steps = None
        if plan:  # If plan is directly provided
            try:
                plan_steps = [GoalStep(**step_data) for step_data in plan]
            except Exception as e:
                 logger.error(f"Invalid plan structure provided for goal '{goal.id}': {e}. Plan generation required.")
                 plan_steps = None  # Fallback to generation

        elif self.planning_agent:  # If no plan provided and agent exists, generate it
            # Trigger plan generation outside the lock to avoid holding it during LLM call
            asyncio.create_task(self.generate_plan_for_goal(goal.id))
        else:
            logger.warning(f"Goal '{goal.id}' added without a plan and no planning agent available.")
            # Goal remains pending until a plan is added manually or agent becomes available

        # If plan was provided and valid, update the goal
        if plan_steps:
            async with self._lock:
                if goal.id in self.goals:  # Check if goal still exists
                    self.goals[goal.id].plan = plan_steps
                    self.goals[goal.id].status = "pending"  # Ready to be activated

        return goal.id

    async def generate_plan_for_goal(self, goal_id: str) -> bool:
        """Generates and assigns a plan for a goal using the planning agent."""
        async with self._lock:
            if goal_id not in self.goals:
                logger.error(f"Cannot generate plan: Goal {goal_id} not found.")
                return False
            goal = self.goals[goal_id]
            if goal.plan:  # Don't overwrite existing plan
                 logger.info(f"Goal '{goal_id}' already has a plan.")
                 return True

        if not self.planning_agent:
            logger.warning(f"Cannot generate plan for goal '{goal_id}': Planning agent not available.")
            return False

        logger.info(f"Generating plan for goal '{goal.id}': {goal.description}")

        try:
            with trace(workflow_name="GenerateGoalPlan", group_id=self.trace_group_id):
                # Include additional context for better plan generation
                context = {
                    "goal": {
                        "id": goal.id,
                        "description": goal.description,
                        "priority": goal.priority,
                        "source": goal.source,
                        "associated_need": goal.associated_need
                    }
                }
                
                # Add information about previous goals/steps if available
                recent_goals = []
                async with self._lock:
                    for g_id, g in self.goals.items():
                        if g_id != goal_id and g.status == "completed" and len(recent_goals) < 3:
                            recent_goals.append({
                                "description": g.description,
                                "steps": [step.description for step in g.plan[:3]]  # First 3 steps only
                            })
                
                if recent_goals:
                    context["recent_completed_goals"] = recent_goals
                
                prompt = f"Generate a plan as a JSON list of steps to achieve the following goal for an AI named Nyx: {goal.description}\n\nAdditional context: {json.dumps(context, default=str)}"

                result = await Runner.run(
                    self.planning_agent,
                    prompt,
                    run_config={
                        "workflow_name": "GoalPlanning",
                        "trace_metadata": {
                            "goal_id": goal_id,
                            "goal_description": goal.description
                        }
                    }
                )

                # Extract and validate the plan
                plan_data = result.final_output
                
                # Ensure we have a list of steps
                if not isinstance(plan_data, list):
                    raise ValueError(f"Expected list of steps, got {type(plan_data)}")
                
                # Convert to GoalStep objects
                plan_steps = [GoalStep(**step_data) for step_data in plan_data]
                
                # Update the goal with the new plan
                async with self._lock:
                    if goal_id in self.goals:  # Re-check existence
                        self.goals[goal_id].plan = plan_steps
                        self.goals[goal_id].current_step_index = 0  # Reset index
                        if self.goals[goal_id].status != "failed":  # Don't reactivate failed goals
                             self.goals[goal_id].status = "pending"
                        logger.info(f"Generated plan with {len(plan_steps)} steps for goal '{goal.id}'.")
                return True

        except Exception as e:
            error_msg = f"Error generating plan for goal '{goal.id}': {e}"
            logger.exception(error_msg)  # Log full traceback
            await self.update_goal_status(goal_id, "failed", error=error_msg)
            return False

    def get_prioritized_goals(self) -> List[Goal]:
        """Returns active and pending goals sorted by priority."""
        # This method needs the lock for thread safety
        goals_copy = {}
        try:
            # Create a copy of the active/pending goals while holding the lock
            asyncio.run(self._lock.acquire())  # Use run for sync context
            goals_copy = {
                g_id: g.model_copy() for g_id, g in self.goals.items() 
                if g.status in ["pending", "active"]
            }
        finally:
            self._lock.release()

        if not goals_copy:
            return []

        # Process the copy without holding the lock
        now = datetime.datetime.now()
        
        def sort_key(g: Goal) -> float:
            age_penalty = (now - g.creation_time).total_seconds() / (3600 * 24)  # Age in days
            # Increase priority slightly for older pending goals, decrease for older active ones
            status_boost = 0.05 if g.status == "pending" else -0.05
            # Add urgency based on deadline if present
            deadline_urgency = 0.0
            if g.deadline:
                time_to_deadline = (g.deadline - now).total_seconds()
                if time_to_deadline > 0:
                    # Increase priority as deadline approaches
                    deadline_urgency = min(0.3, 86400 / max(3600, time_to_deadline))  # Max boost of 0.3
                else:
                    # Past deadline, very high urgency
                    deadline_urgency = 0.4
            
            return g.priority + (age_penalty * 0.01 * status_boost) + deadline_urgency

        sorted_goals = list(goals_copy.values())
        sorted_goals.sort(key=sort_key, reverse=True)
        return sorted_goals

    def select_active_goal(self) -> Optional[str]:
        """Selects the highest priority goal to work on, respecting concurrency limits."""
        # This method needs to acquire the lock for thread safety
        asyncio.run(self._lock.acquire())
        try:
            # Get prioritized goals (already sorted by priority)
            prioritized = self.get_prioritized_goals()
            selected_goal_id = None

            if not prioritized:
                self.active_goals.clear()  # No goals left to be active
                return None

            # Check if any currently active goals are finished or failed
            finished_active = {
                gid for gid in self.active_goals 
                if gid not in self.goals or self.goals[gid].status not in ["active", "pending"]
            }
            self.active_goals -= finished_active
            active_count = len(self.active_goals)

            # Find the highest priority goal that can be activated
            for goal in prioritized:
                if goal.id in self.active_goals:
                    # If it's already active, it's a candidate to continue
                    selected_goal_id = goal.id
                    break  # Keep executing the highest priority *active* goal first
                elif goal.status == "pending" and active_count < self.max_concurrent_goals and goal.plan:
                    # Activate this pending goal if concurrency allows and it has a plan
                    selected_goal_id = goal.id
                    goal.status = "active"
                    goal.current_step_index = 0  # Ensure starting from step 0
                    self.active_goals.add(goal.id)
                    logger.info(f"Activated goal '{goal.id}' (Priority: {goal.priority:.2f})")
                    break

            return selected_goal_id
        finally:
            self._lock.release()

    async def execute_next_step(self) -> Optional[Dict[str, Any]]:
        """Selects and executes the next step of the highest priority active goal."""
        goal_id = self.select_active_goal()  # Selection handles concurrency & prioritization

        if goal_id is None:
            return None

        async with self._lock:  # Lock for accessing goal state
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
                return {"skipped_step": step.model_dump(), "goal_id": goal_id}  # Indicate skip

        # --- Check for sensitive actions that need special handling ---
        is_dominance_action = step.action in [
            "issue_command", "increase_control_intensity", "apply_consequence_simulated",
            "select_dominance_tactic", "trigger_dominance_gratification", "praise_submission"
        ]
        
        user_id_param = step.parameters.get("user_id", step.parameters.get("target_user_id"))

        if is_dominance_action and user_id_param and self.brain and hasattr(self.brain, '_evaluate_dominance_step_appropriateness'):
            try:
                evaluation = await self.brain._evaluate_dominance_step_appropriateness(
                    step.action, step.parameters, user_id_param
                )
                action_decision = evaluation.get("action", "proceed")

                if action_decision == "block":
                    logger.warning(f"Dominance step '{step.step_id}' blocked: {evaluation.get('reason')}")
                    await self.update_goal_status(goal.id, "failed", error=f"Dominance step blocked: {evaluation.get('reason')}")
                    self.active_goals.discard(goal.id)
                    return {"blocked_step": step.model_dump(), "goal_id": goal_id, "reason": evaluation.get('reason')}
                    
                elif action_decision == "delay":
                    logger.info(f"Dominance step '{step.step_id}' delayed: {evaluation.get('reason')}")
                    # Keep goal active but don't execute this step now
                    return {"delayed_step": step.model_dump(), "goal_id": goal_id, "reason": evaluation.get('reason')}
                    
                elif action_decision == "modify":
                    logger.info(f"Dominance step '{step.step_id}' modified: {evaluation.get('reason')}")
                    # Modify step parameters (e.g., reduce intensity)
                    if "new_intensity_level" in evaluation:
                        step.parameters["intensity_level"] = evaluation["new_intensity_level"]
                    # Proceed with modified step execution below
            except Exception as e:
                 logger.error(f"Error during dominance step evaluation: {e}")
                 # Default to blocking if evaluation fails
                 await self.update_goal_status(goal.id, "failed", error="Dominance evaluation failed.")
                 self.active_goals.discard(goal.id)
                 return {"blocked_step": step.model_dump(), "goal_id": goal_id, "reason": "Evaluation error"}

        # --- Execute Step (Outside Lock) ---
        if not self.brain:
            error_msg = "NyxBrain reference not set in GoalManager. Cannot execute actions."
            logger.error(error_msg)
            await self.update_goal_status(goal_id, "failed", error=error_msg)
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

                resolved_params = await self._resolve_step_parameters(goal_id, step.parameters)
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
            if goal_id not in self.goals: 
                return None
                
            goal = self.goals[goal_id]
            # Ensure we are updating the correct step
            if goal.current_step_index == step_index and step.step_id == goal.plan[step_index].step_id:
                step = goal.plan[step_index]  # Get the official step object again
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

        return {"executed_step": step.model_dump(), "goal_id": goal_id}

    async def _resolve_step_parameters(self, goal_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
         """Resolves parameter placeholders like '$step_1.result' within the context of a specific goal."""
         # Needs read-only access to the goal state, potentially under lock if strict consistency needed
         async with self._lock:  # Using lock for consistency
            goal = self.goals.get(goal_id)
            if not goal:
                logger.error(f"Cannot resolve parameters: Goal {goal_id} not found.")
                return parameters  # Return original if goal gone

         resolved_params = {}
         for key, value in parameters.items():
             if isinstance(value, str) and value.startswith("$step_"):
                 parts = value[1:].split('.')
                 source_step_index_str = parts[0][4:]  # Get the number after 'step_'
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
                                 elif isinstance(current_value, BaseModel):  # Check for Pydantic models
                                      current_value = getattr(current_value, part, None)
                                 elif hasattr(current_value, part):
                                     current_value = getattr(current_value, part)
                                 else:
                                     try:  # Try index access for lists/tuples
                                         idx = int(part)
                                         if isinstance(current_value, (list, tuple)) and -len(current_value) <= idx < len(current_value):
                                             current_value = current_value[idx]
                                         else:
                                             current_value = None
                                             break
                                     except (ValueError, TypeError, IndexError):
                                         current_value = None
                                         break
                                 if current_value is None: 
                                     break

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
        async with self._lock:  # Ensure atomic update of goal state
            if goal_id not in self.goals:
                logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
                return

            goal = self.goals[goal_id]
            old_status = goal.status
            if old_status == status: 
                return  # No change

            goal.status = status
            goal.last_error = error

            if status in ["completed", "failed", "abandoned"]:
                goal.completion_time = datetime.datetime.now()
                self.active_goals.discard(goal_id)  # Remove from active set
                
                # Update statistics
                if status == "completed":
                    self.goal_statistics["completed"] += 1
                elif status == "failed":
                    self.goal_statistics["failed"] += 1
                elif status == "abandoned":
                    self.goal_statistics["abandoned"] += 1

            logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")

            # --- Capture goal data for notifications ---
            need_name = goal.associated_need
            source = goal.source
            priority = goal.priority

        # --- Notifications (Keep outside lock if they involve awaits/long ops) ---
        
        # Notify NeedsSystem if applicable
        if need_name and self.brain and hasattr(self.brain, 'needs_system'):
            try:
                needs_system = getattr(self.brain, 'needs_system')
                if status == "completed" and hasattr(needs_system, 'satisfy_need'):
                    satisfaction_amount = priority * 0.3 + 0.1  # Base + priority bonus
                    await needs_system.satisfy_need(need_name, satisfaction_amount)
                elif status == "failed" and hasattr(needs_system, 'decrease_need'):
                    decrease_amount = priority * 0.1  # Small decrease for failure
                    await needs_system.decrease_need(need_name, decrease_amount)
            except Exception as e:
                logger.error(f"Error notifying NeedsSystem about goal {goal_id}: {e}")

        # Notify RewardSystem if applicable
        if self.brain and hasattr(self.brain, 'reward_system'):
            try:
                reward_system = getattr(self.brain, 'reward_system')
                reward_value = 0.0
                if status == "completed": 
                    reward_value = priority * 0.6  # Higher reward for completion
                elif status == "failed": 
                    reward_value = -priority * 0.4  # Punish failure
                elif status == "abandoned": 
                    reward_value = -0.1  # Small punishment for abandoning

                if abs(reward_value) > 0.05 and hasattr(reward_system, 'process_reward_signal'):
                    # Import locally to avoid circular imports
                    from nyx.core.reward_system import RewardSignal
                    
                    reward_signal = RewardSignal(
                        value=reward_value, 
                        source="GoalManager",
                        context={
                            "goal_id": goal_id, 
                            "goal_description": goal.description, 
                            "outcome": status, 
                            "associated_need": need_name
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    # Use create_task for non-blocking reward processing
                    asyncio.create_task(reward_system.process_reward_signal(reward_signal))
            except Exception as e:
                logger.error(f"Error notifying RewardSystem about goal {goal_id}: {e}")

        # Notify MetaCore if applicable
        if self.brain and hasattr(self.brain, 'meta_core'):
            try:
                meta_core = getattr(self.brain, 'meta_core')
                # Check if meta_core has record_goal_outcome method
                if hasattr(meta_core, 'record_goal_outcome'):
                    asyncio.create_task(meta_core.record_goal_outcome(goal.model_dump()))
            except Exception as e:
                logger.error(f"Error notifying MetaCore about goal {goal_id}: {e}")

    async def abandon_goal(self, goal_id: str, reason: str) -> Dict[str, Any]:
        """Abandons an active or pending goal."""
        logger.info(f"Abandoning goal '{goal_id}': {reason}")
        
        async with self._lock:
            if goal_id not in self.goals:
                return {"status": "error", "message": f"Goal {goal_id} not found"}
                
            goal = self.goals[goal_id]
            if goal.status not in ["active", "pending"]:
                return {"status": "error", "message": f"Cannot abandon goal with status {goal.status}"}
        
        await self.update_goal_status(goal_id, "abandoned", error=reason)
        
        return {
            "status": "success", 
            "goal_id": goal_id, 
            "message": f"Goal abandoned: {reason}"
        }

    def has_active_goal_for_need(self, need_name: str) -> bool:
        """Checks if there's an active goal associated with a specific need."""
        # Needs lock for safe iteration
        try:
            asyncio.run(self._lock.acquire())  # Use run for sync context
            return any(g.status in ["active", "pending"] and g.associated_need == need_name 
                      for g in self.goals.values())
        finally:
            self._lock.release()

    async def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status and plan of a specific goal."""
        async with self._lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                # Return a copy, exclude potentially large history for status checks
                goal_data = goal.model_dump(exclude={'execution_history', 'plan'})
                goal_data['plan_step_count'] = len(goal.plan)
                
                # Add current step info if available
                if 0 <= goal.current_step_index < len(goal.plan):
                    current_step = goal.plan[goal.current_step_index]
                    goal_data['current_step'] = {
                        'description': current_step.description,
                        'action': current_step.action,
                        'status': current_step.status
                    }
                
                return goal_data
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
                    if 0 <= goal.current_step_index < len(goal.plan):
                        summary['current_step_description'] = goal.plan[goal.current_step_index].description
                    else:
                        summary['current_step_description'] = "N/A"
                    filtered_goals.append(summary)
            
            # Sort by priority (descending)
            filtered_goals.sort(key=lambda g: g['priority'], reverse=True)
            
        return filtered_goals
    
    async def get_goal_statistics(self) -> Dict[str, Any]:
        """Gets statistics about goal execution."""
        async with self._lock:
            # Count goals by status
            status_counts = {}
            for goal in self.goals.values():
                status = goal.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate success rate
            total_completed = self.goal_statistics["completed"]
            total_failed = self.goal_statistics["failed"]
            total_abandoned = self.goal_statistics["abandoned"]
            total_finished = total_completed + total_failed + total_abandoned
            
            success_rate = total_completed / total_finished if total_finished > 0 else 0
            
            # Calculate average completion time
            completion_times = []
            for goal in self.goals.values():
                if goal.status == "completed" and goal.creation_time and goal.completion_time:
                    duration = (goal.completion_time - goal.creation_time).total_seconds()
                    completion_times.append(duration)
            
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            
            return {
                "total_goals_created": self.goal_statistics["created"],
                "goals_by_status": status_counts,
                "success_rate": success_rate,
                "average_completion_time_seconds": avg_completion_time,
                "active_goals_count": len(self.active_goals),
                "statistics": self.goal_statistics
            }
