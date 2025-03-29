# nyx/core/goal_system.py

import logging
import datetime
import uuid
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from agents import (
    Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper,
    handoff, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
)

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

class TimeHorizon(str, Enum):
    """Time horizon for goals - affects planning, execution and priority calculations"""
    SHORT_TERM = "short_term"  # Hours to days
    MEDIUM_TERM = "medium_term"  # Days to weeks
    LONG_TERM = "long_term"  # Weeks to months/years

class EmotionalMotivation(BaseModel):
    """Emotional motivation behind a goal"""
    primary_need: str = Field(..., description="Primary emotional need driving this goal")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity of the emotional motivation")
    expected_satisfaction: float = Field(0.5, ge=0.0, le=1.0, description="Expected satisfaction from achieving the goal")
    associated_chemicals: Dict[str, float] = Field(default_factory=dict, description="Associated neurochemicals and their expected changes")
    description: str = Field("", description="Description of the emotional motivation")

class GoalRelationship(BaseModel):
    """Relationship between goals"""
    parent_goal_id: Optional[str] = Field(None, description="ID of the parent goal")
    child_goal_ids: List[str] = Field(default_factory=list, description="IDs of child goals")
    supports_goal_ids: List[str] = Field(default_factory=list, description="IDs of goals this goal supports")
    conflicts_with_goal_ids: List[str] = Field(default_factory=list, description="IDs of goals this goal conflicts with")
    relationship_type: str = Field("independent", description="Type of relationship (hierarchical, supportive, conflicting)")


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
    time_horizon: TimeHorizon = Field(TimeHorizon.MEDIUM_TERM, description="Time horizon for the goal")
    emotional_motivation: Optional[EmotionalMotivation] = None
    relationships: GoalRelationship = Field(default_factory=GoalRelationship)
    recurring: bool = Field(False, description="Whether this is a recurring goal")
    recurrence_pattern: Optional[Dict[str, Any]] = None  # For recurring goals
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress toward completion (0.0-1.0)")

# New Pydantic models for structured I/O with Agents
class GoalValidationResult(BaseModel):
    """Output model for goal validation guardrail"""
    is_valid: bool = Field(..., description="Whether the goal is valid")
    reason: Optional[str] = Field(None, description="Reason for invalidation if not valid")
    priority_adjustment: Optional[float] = Field(None, description="Suggested priority adjustment")

class PlanValidationResult(BaseModel):
    """Output model for plan validation guardrail"""
    is_valid: bool = Field(..., description="Whether the plan is valid")
    reason: Optional[str] = Field(None, description="Reason for invalidation if not valid")
    suggestions: List[str] = Field(default_factory=list, description="Suggested improvements")

class StepExecutionResult(BaseModel):
    """Output model for step execution agent"""
    success: bool = Field(..., description="Whether the step executed successfully")
    step_id: str = Field(..., description="ID of the executed step")
    result: Optional[Any] = Field(None, description="Result from the execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    next_action: str = Field("continue", description="continue, retry, skip, or abort")

class PlanGenerationResult(BaseModel):
    """Output model for plan generation agent"""
    plan: List[Dict[str, Any]] = Field(..., description="Generated plan steps")
    reasoning: str = Field(..., description="Reasoning behind the plan")
    estimated_steps: int = Field(..., description="Estimated number of steps")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")

class GoalHierarchyNode(BaseModel):
    """Node in a goal hierarchy visualization"""
    goal_id: str
    description: str
    time_horizon: TimeHorizon
    status: str
    priority: float
    children: List["GoalHierarchyNode"] = Field(default_factory=list)

class GoalCreationWithMotivation(BaseModel):
    """Input model for goal creation with motivation"""
    description: str
    priority: float = 0.5
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    deadline: Optional[str] = None
    emotional_motivation: Optional[EmotionalMotivation] = None
    associated_need: Optional[str] = None
    parent_goal_id: Optional[str] = None

class GoalMotivationAnalysis(BaseModel):
    """Output model for analyzing goal motivations"""
    emotional_needs: Dict[str, int] = Field(default_factory=dict)
    primary_motivations: List[str] = Field(default_factory=list)
    chemical_associations: Dict[str, float] = Field(default_factory=dict)
    motivation_patterns: List[Dict[str, Any]] = Field(default_factory=list)

class RunContext(BaseModel):
    """Context model for agent execution"""
    goal_id: str
    brain_available: bool = True
    user_id: Optional[str] = None
    current_step_index: int = 0
    max_retries: int = 3
    retry_count: int = 0

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
        self._lock = asyncio.Lock()  # Lock for managing goals dict safely
        
        # Goal outcomes for analytics
        self.goal_statistics = {
            "created": 0,
            "completed": 0,
            "failed": 0,
            "abandoned": 0
        }
        
        # Initialize agents
        self._init_agents()
        self.trace_group_id = "NyxGoalManagement"

        logger.info("GoalManager initialized with Agent SDK integration.")
        
    def _init_agents(self):
        """Initialize all agents needed for the goal system"""
        # Goal planning agent (generates plans for goals)
        self.planning_agent = self._create_planning_agent()
        
        # Goal validation agent (validates goals before accepting them)
        self.goal_validation_agent = self._create_goal_validation_agent()
        
        # Plan validation agent (validates plans before execution)
        self.plan_validation_agent = self._create_plan_validation_agent()
        
        # Step execution agent (handles step execution and error recovery)
        self.step_execution_agent = self._create_step_execution_agent()
        
        # Main orchestration agent (coordinates the overall goal execution)
        self.orchestration_agent = self._create_orchestration_agent()

    def set_brain_reference(self, brain):
        """Set the reference to the main NyxBrain after initialization."""
        self.brain = brain
        logger.info("NyxBrain reference set for GoalManager.")

    def _create_planning_agent(self) -> Agent:
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
                name="Goal_Planner_Agent",
                instructions=f"""You are a planner agent for the Nyx AI. Your task is to break down a high-level goal description into a sequence of concrete, actionable steps using Nyx's available actions.

                Available Actions Nyx can perform (these are methods on the main system):
                {tool_descriptions}

                For a given goal, create a detailed plan with logical steps that build on each other. Each step should use results from previous steps when appropriate.

                Your plan should be thorough, considering:
                1. Required inputs for each step
                2. Dependencies between steps
                3. Error handling options
                4. Resources needed
                5. Estimated completion difficulty

                When the goal is associated with a specific need, ensure your plan includes steps that specifically address that need.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.1),
                tools=[
                    function_tool(self._get_available_actions),
                    function_tool(self._get_action_description),
                    function_tool(self._get_goal_details),
                    function_tool(self._get_recent_goals)
                ],
                output_type=PlanGenerationResult
            )
        except Exception as e:
            logger.error(f"Error creating planning agent: {e}")
            return None
    
    def _create_goal_validation_agent(self) -> Agent:
        """Creates an agent that validates goals before acceptance"""
        return Agent(
            name="Goal_Validation_Agent",
            instructions="""You are a goal validation agent for Nyx AI. Your task is to evaluate whether a proposed goal:

            1. Is well-defined and clear enough to plan for
            2. Has an appropriate priority level
            3. Is aligned with Nyx's capabilities and purpose
            4. Is ethical and appropriate
            5. Doesn't conflict with existing active goals
            
            If the goal needs adjustment, provide specific feedback. 
            For priority adjustments, consider how important and urgent the goal appears.
            """,
            model="gpt-4o",
            tools=[
                function_tool(self._get_active_goals), 
                function_tool(self._check_goal_conflicts),
                function_tool(self._verify_capabilities)
            ],
            output_type=GoalValidationResult
        )
    
    def _create_plan_validation_agent(self) -> Agent:
        """Creates an agent that validates plans before execution"""
        return Agent(
            name="Plan_Validation_Agent",
            instructions="""You are a plan validation agent for Nyx AI. Your task is to evaluate whether a proposed plan:

            1. Is logically sequenced with proper dependencies
            2. Uses valid actions with correct parameters
            3. Is likely to achieve the stated goal
            4. Handles potential error cases
            5. Uses resources efficiently
            
            Look for issues like:
            - Missing prerequisite steps
            - Invalid action references
            - Unclear parameter definitions
            - Redundant steps or inefficient sequences
            
            Provide specific suggestions for improvement if issues are found.
            """,
            model="gpt-4o",
            tools=[
                function_tool(self._validate_action_sequence),
                function_tool(self._check_parameter_references),
                function_tool(self._estimate_plan_efficiency)
            ],
            output_type=PlanValidationResult
        )
    
    def _create_step_execution_agent(self) -> Agent:
        """Creates an agent that handles step execution and error recovery"""
        return Agent(
            name="Step_Execution_Agent",
            instructions="""You are a step execution agent for Nyx AI. Your task is to:

            1. Oversee the execution of individual goal steps
            2. Resolve parameter references to previous step results
            3. Handle errors and suggest recovery options
            4. Determine whether to continue, retry, skip or abort after each step
            
            When a step fails, consider:
            - Is this a temporary failure that might succeed on retry?
            - Is this step optional or can we skip it?
            - Does this failure require aborting the entire goal?
            
            For dominance-related actions, ensure they meet safety and contextual appropriateness 
            requirements before executing.
            """,
            model="gpt-4o",
            tools=[
                function_tool(self._resolve_step_parameters_tool),
                function_tool(self._execute_action),
                function_tool(self._check_dominance_appropriateness),
                function_tool(self._log_execution_result)
            ],
            output_type=StepExecutionResult
        )
    
    def _create_orchestration_agent(self) -> Agent:
        """Creates the main orchestration agent that coordinates goal execution"""
        return Agent(
            name="Goal_Orchestration_Agent",
            instructions="""You are the goal orchestration agent for Nyx AI. Your role is to coordinate the entire goal lifecycle:

            1. Validate incoming goals using the validation agent
            2. Generate plans for validated goals using the planning agent
            3. Validate plans before execution
            4. Coordinate step execution through the execution agent
            5. Handle goal completion, failure or abandonment
            
            You should efficiently manage the goal queue, prioritize goals appropriately,
            and ensure resources are allocated effectively across concurrent goals.
            
            Monitor for conflicts between goals and ensure critical goals receive
            priority attention.
            """,
            handoffs=[
                handoff(self.goal_validation_agent, 
                       tool_name_override="validate_goal", 
                       tool_description_override="Validate a goal before acceptance"),
                
                handoff(self.planning_agent, 
                       tool_name_override="generate_plan",
                       tool_description_override="Generate a plan for a validated goal"),
                
                handoff(self.plan_validation_agent,
                       tool_name_override="validate_plan",
                       tool_description_override="Validate a plan before execution"),
                       
                handoff(self.step_execution_agent,
                       tool_name_override="execute_step",
                       tool_description_override="Execute a step in the goal plan")
            ],
            tools=[
                function_tool(self._get_prioritized_goals),
                function_tool(self._update_goal_status_tool),
                function_tool(self._notify_systems),
                function_tool(self._check_concurrency_limits)
            ],
            model="gpt-4o"
        )
    
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

    # New Agent SDK tools for goal validation agent
    @function_tool
    async def _get_active_goals(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get currently active and pending goals
        
        Returns:
            Dictionary with active and pending goals
        """
        goals = []
        async with self._lock:
            for goal_id, goal in self.goals.items():
                if goal.status in ["active", "pending"]:
                    goals.append({
                        "id": goal.id,
                        "description": goal.description,
                        "priority": goal.priority,
                        "source": goal.source,
                        "associated_need": goal.associated_need,
                        "status": goal.status
                    })
        
        return {
            "active_count": len([g for g in goals if g["status"] == "active"]),
            "pending_count": len([g for g in goals if g["status"] == "pending"]),
            "goals": goals
        }
    
    @function_tool
    async def _check_goal_conflicts(self, ctx: RunContextWrapper, goal_description: str) -> Dict[str, Any]:
        """
        Check if a proposed goal conflicts with existing goals
        
        Args:
            goal_description: Description of the proposed goal
            
        Returns:
            Conflict information
        """
        conflicts = []
        async with self._lock:
            for goal_id, goal in self.goals.items():
                if goal.status in ["active", "pending"]:
                    # Simple overlap detection - could be more sophisticated
                    words1 = set(goal.description.lower().split())
                    words2 = set(goal_description.lower().split())
                    overlap = len(words1.intersection(words2)) / max(1, min(len(words1), len(words2)))
                    
                    if overlap > 0.6:  # High similarity threshold
                        conflicts.append({
                            "goal_id": goal.id,
                            "description": goal.description,
                            "similarity": overlap,
                            "status": goal.status
                        })
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflict_count": len(conflicts),
            "conflicts": conflicts
        }
    
    @function_tool
    async def _verify_capabilities(self, ctx: RunContextWrapper, required_actions: List[str]) -> Dict[str, Any]:
        """
        Verify if required actions are available in Nyx's capabilities
        
        Args:
            required_actions: List of actions required by the goal
            
        Returns:
            Capability verification results
        """
        available_actions = await self._get_available_actions(ctx)
        available_action_names = [a["name"] for a in available_actions["actions"]]
        
        unavailable = [action for action in required_actions if action not in available_action_names]
        
        return {
            "all_available": len(unavailable) == 0,
            "available_count": len(required_actions) - len(unavailable),
            "unavailable_actions": unavailable
        }
    
    # New Agent SDK tools for plan validation agent
    @function_tool
    async def _validate_action_sequence(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that actions are sequenced correctly with proper dependencies
        
        Args:
            plan: The plan to validate
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for valid action names
        available_actions = await self._get_available_actions(ctx)
        available_action_names = [a["name"] for a in available_actions["actions"]]
        
        for i, step in enumerate(plan):
            step_num = i + 1
            action = step.get("action", "")
            
            # Check if action exists
            if action not in available_action_names:
                issues.append(f"Step {step_num}: Action '{action}' is not available")
                
            # Check parameter references to previous steps
            for param_name, param_value in step.get("parameters", {}).items():
                if isinstance(param_value, str) and param_value.startswith("$step_"):
                    # Extract referenced step number
                    try:
                        ref_step = int(param_value.split("_")[1].split(".")[0])
                        if ref_step > step_num:
                            issues.append(f"Step {step_num}: References future step {ref_step}")
                        if ref_step == step_num:
                            issues.append(f"Step {step_num}: Self-reference detected")
                    except (ValueError, IndexError):
                        issues.append(f"Step {step_num}: Invalid step reference format: {param_value}")
        
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues
        }
    
    @function_tool
    async def _check_parameter_references(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if parameter references between steps are valid
        
        Args:
            plan: The plan to validate
            
        Returns:
            Parameter reference validation results
        """
        issues = []
        provided_outputs = {}  # Track what each step provides
        
        for i, step in enumerate(plan):
            step_num = i + 1
            # Analyze expected outputs
            action = step.get("action", "")
            provided_outputs[step_num] = self._estimate_action_output_fields(action)
            
            # Check parameter references
            for param_name, param_value in step.get("parameters", {}).items():
                if isinstance(param_value, str) and param_value.startswith("$step_"):
                    parts = param_value[1:].split('.')
                    if len(parts) < 2:
                        issues.append(f"Step {step_num}: Invalid reference format: {param_value}")
                        continue
                        
                    # Extract referenced step and field
                    try:
                        ref_step_str = parts[0]
                        ref_step = int(ref_step_str.replace("step_", ""))
                        field_path = '.'.join(parts[1:])
                        
                        if ref_step >= step_num:
                            issues.append(f"Step {step_num}: References non-executed step {ref_step}")
                            continue
                            
                        if ref_step not in provided_outputs:
                            issues.append(f"Step {step_num}: References unknown step {ref_step}")
                            continue
                            
                        # Check if the referenced field exists in the output
                        if not self._check_field_availability(provided_outputs[ref_step], field_path):
                            issues.append(f"Step {step_num}: Referenced field '{field_path}' may not exist in step {ref_step} output")
                    except (ValueError, IndexError):
                        issues.append(f"Step {step_num}: Invalid step reference: {param_value}")
        
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues
        }
    
    def _estimate_action_output_fields(self, action: str) -> List[str]:
        """Estimate what fields an action might output based on its name"""
        # This is a simplified estimate - in a real system, you might have more detailed schema
        common_fields = ["result", "success", "error"]
        
        if action.startswith("query_"):
            return common_fields + ["data", "matches", "count"]
        elif action.startswith("retrieve_"):
            return common_fields + ["items", "count"]
        elif action.startswith("generate_"):
            return common_fields + ["content", "text"]
        elif action.startswith("analyze_"):
            return common_fields + ["analysis", "score", "details"]
        elif action.startswith("evaluate_"):
            return common_fields + ["evaluation", "score", "feedback"]
        else:
            return common_fields
    
    def _check_field_availability(self, available_fields: List[str], field_path: str) -> bool:
        """Check if a field path might be available in the output"""
        if not field_path or not available_fields:
            return False
            
        top_field = field_path.split('.')[0]
        return top_field in available_fields or "result" in available_fields
    
    @function_tool
    async def _estimate_plan_efficiency(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate the efficiency of a plan
        
        Args:
            plan: The plan to evaluate
            
        Returns:
            Efficiency analysis
        """
        # Count actions by category
        action_categories = {}
        for step in plan:
            action = step.get("action", "")
            category = "unknown"
            
            if action.startswith(("query_", "retrieve_")):
                category = "retrieval"
            elif action.startswith(("add_", "create_")):
                category = "creation"
            elif action.startswith(("update_", "modify_")):
                category = "modification"
            elif action.startswith(("analyze_", "evaluate_", "reason_")):
                category = "analysis"
            elif action.startswith(("generate_", "express_")):
                category = "generation"
            
            action_categories[category] = action_categories.get(category, 0) + 1
        
        # Check for potential inefficiencies
        retrieval_heavy = action_categories.get("retrieval", 0) > len(plan) * 0.5
        creation_heavy = action_categories.get("creation", 0) > len(plan) * 0.4
        
        suggestions = []
        if retrieval_heavy:
            suggestions.append("Plan may benefit from combining multiple retrieval steps")
        if creation_heavy:
            suggestions.append("Plan has many creation steps; consider batching or combining some")
        if len(plan) > 10:
            suggestions.append("Plan is quite long; consider if some steps can be combined")
        
        return {
            "step_count": len(plan),
            "action_distribution": action_categories,
            "efficiency_score": 0.7 if suggestions else 0.9,  # Simple scoring
            "suggestions": suggestions
        }
    
    # New Agent SDK tools for step execution agent
    @function_tool
    async def _resolve_step_parameters_tool(self, ctx: RunContextWrapper, goal_id: str, step_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves parameter placeholders for a step
        
        Args:
            goal_id: The goal ID
            step_parameters: The parameters to resolve
            
        Returns:
            Resolved parameters
        """
        resolved = await self._resolve_step_parameters(goal_id, step_parameters)
        
        # Check which parameters were successfully resolved
        resolution_status = {}
        for key, value in step_parameters.items():
            original = value
            resolved_value = resolved.get(key, None)
            
            if isinstance(original, str) and original.startswith("$step_"):
                resolution_status[key] = {
                    "original": original,
                    "resolved": resolved_value is not None,
                    "is_null": resolved_value is None
                }
        
        return {
            "resolved_parameters": resolved,
            "resolution_status": resolution_status,
            "all_resolved": all(status["resolved"] for status in resolution_status.values()) if resolution_status else True
        }
    
    @function_tool
    async def _execute_action(self, ctx: RunContextWrapper, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action with the given parameters
        
        Args:
            action: The action to execute
            parameters: The parameters for the action
            
        Returns:
            Execution result
        """
        if not self.brain:
            return {
                "success": False,
                "error": "NyxBrain reference not set in GoalManager"
            }
        
        try:
            action_method = getattr(self.brain, action, None)
            if not (action_method and callable(action_method)):
                return {
                    "success": False,
                    "error": f"Action '{action}' not found or not callable on NyxBrain"
                }
            
            # Execute the action
            start_time = datetime.datetime.now()
            result = await action_method(**parameters)
            end_time = datetime.datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "result": result,
                "duration": duration
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__
            }
    
    @function_tool
    async def _check_dominance_appropriateness(self, ctx: RunContextWrapper, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a dominance-related action is appropriate
        
        Args:
            action: The action to check
            parameters: The parameters for the action
            
        Returns:
            Appropriateness check result
        """
        is_dominance_action = action in [
            "issue_command", "increase_control_intensity", "apply_consequence_simulated",
            "select_dominance_tactic", "trigger_dominance_gratification", "praise_submission"
        ]
        
        if not is_dominance_action:
            return {
                "is_dominance_action": False,
                "can_proceed": True,
                "action": "proceed"
            }
        
        user_id_param = parameters.get("user_id", parameters.get("target_user_id"))
        if not user_id_param:
            return {
                "is_dominance_action": True,
                "can_proceed": False,
                "action": "block",
                "reason": "Missing user_id for dominance action"
            }
        
        # If brain has dominance evaluation method, use it
        if self.brain and hasattr(self.brain, '_evaluate_dominance_step_appropriateness'):
            try:
                evaluation = await self.brain._evaluate_dominance_step_appropriateness(
                    action, parameters, user_id_param
                )
                return {
                    "is_dominance_action": True,
                    "evaluation_result": evaluation,
                    "can_proceed": evaluation.get("action") == "proceed",
                    "action": evaluation.get("action", "block"),
                    "reason": evaluation.get("reason")
                }
            except Exception as e:
                return {
                    "is_dominance_action": True,
                    "can_proceed": False,
                    "action": "block",
                    "reason": f"Evaluation error: {str(e)}"
                }
        
        # Default to blocking if no evaluation method
        return {
            "is_dominance_action": True,
            "can_proceed": False,
            "action": "block",
            "reason": "No dominance evaluation method available"
        }
    
    @function_tool
    async def _log_execution_result(self, ctx: RunContextWrapper, goal_id: str, step_id: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log the result of step execution
        
        Args:
            goal_id: The goal ID
            step_id: The step ID
            execution_result: The execution result
            
        Returns:
            Logging result
        """
        async with self._lock:
            if goal_id not in self.goals:
                return {
                    "success": False,
                    "error": f"Goal {goal_id} not found"
                }
            
            goal = self.goals[goal_id]
            step = None
            step_index = -1
            
            # Find the step by ID
            for i, s in enumerate(goal.plan):
                if s.step_id == step_id:
                    step = s
                    step_index = i
                    break
            
            if not step:
                return {
                    "success": False,
                    "error": f"Step {step_id} not found in goal {goal_id}"
                }
            
            # Update step with execution result
            step.status = "completed" if execution_result.get("success", False) else "failed"
            step.result = execution_result.get("result")
            step.error = execution_result.get("error")
            
            if not step.start_time:
                # If start time wasn't recorded earlier
                step.start_time = datetime.datetime.now() - datetime.timedelta(seconds=execution_result.get("duration", 0))
                
            step.end_time = datetime.datetime.now()
            
            # Add to execution history
            goal.execution_history.append({
                "step_id": step_id,
                "action": step.action,
                "status": step.status,
                "timestamp": step.end_time.isoformat(),
                "duration": execution_result.get("duration", 0),
                "error": step.error
            })
            
            return {
                "success": True,
                "step_index": step_index,
                "current_index": goal.current_step_index,
                "step_status": step.status
            }
    
    # New Agent SDK tools for orchestration agent
    @function_tool
    async def _get_prioritized_goals(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get prioritized goals for execution
        
        Returns:
            Prioritized goals
        """
        goals = self.get_prioritized_goals()
        
        return {
            "total_count": len(goals),
            "active_count": len([g for g in goals if g.status == "active"]),
            "pending_count": len([g for g in goals if g.status == "pending"]),
            "goals": [g.model_dump(exclude={'plan', 'execution_history'}) for g in goals[:5]]  # Top 5 goals
        }
    
    @function_tool
    async def _update_goal_status_tool(self, ctx: RunContextWrapper, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the status of a goal
        
        Args:
            goal_id: The goal ID
            status: The new status
            result: Optional result data
            error: Optional error message
            
        Returns:
            Status update result
        """
        if status not in ["pending", "active", "completed", "failed", "abandoned"]:
            return {
                "success": False,
                "error": f"Invalid status: {status}"
            }
        
        async with self._lock:
            if goal_id not in self.goals:
                return {
                    "success": False,
                    "error": f"Goal {goal_id} not found"
                }
                
            goal = self.goals[goal_id]
            old_status = goal.status
            
            # Update goal status
            goal.status = status
            goal.last_error = error
            
            if status in ["completed", "failed", "abandoned"]:
                goal.completion_time = datetime.datetime.now()
                self.active_goals.discard(goal_id)
                
                # Update statistics
                if status == "completed":
                    self.goal_statistics["completed"] += 1
                elif status == "failed":
                    self.goal_statistics["failed"] += 1
                elif status == "abandoned":
                    self.goal_statistics["abandoned"] += 1
        
        # Notify other systems (outside of lock)
        await self._notify_systems(ctx, goal_id, status, result, error)
        
        return {
            "success": True,
            "goal_id": goal_id,
            "old_status": old_status,
            "new_status": status
        }
    
    @function_tool
    async def _notify_systems(self, ctx: RunContextWrapper, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Notify relevant systems about goal status changes
        
        Args:
            goal_id: The goal ID
            status: The new status
            result: Optional result data
            error: Optional error message
            
        Returns:
            Notification results
        """
        if goal_id not in self.goals:
            return {
                "success": False,
                "error": f"Goal {goal_id} not found"
            }
            
        goal = self.goals[goal_id]
        notifications = {}
        
        # Notify NeedsSystem if applicable
        if goal.associated_need and self.brain and hasattr(self.brain, 'needs_system'):
            try:
                needs_system = getattr(self.brain, 'needs_system')
                if status == "completed" and hasattr(needs_system, 'satisfy_need'):
                    satisfaction_amount = goal.priority * 0.3 + 0.1  # Base + priority bonus
                    await needs_system.satisfy_need(goal.associated_need, satisfaction_amount)
                    notifications["needs_system"] = {
                        "success": True,
                        "need": goal.associated_need,
                        "amount": satisfaction_amount,
                        "action": "satisfy"
                    }
                elif status == "failed" and hasattr(needs_system, 'decrease_need'):
                    decrease_amount = goal.priority * 0.1  # Small decrease for failure
                    await needs_system.decrease_need(goal.associated_need, decrease_amount)
                    notifications["needs_system"] = {
                        "success": True,
                        "need": goal.associated_need,
                        "amount": decrease_amount,
                        "action": "decrease"
                    }
            except Exception as e:
                notifications["needs_system"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Notify RewardSystem if applicable
        if self.brain and hasattr(self.brain, 'reward_system'):
            try:
                reward_system = getattr(self.brain, 'reward_system')
                reward_value = 0.0
                if status == "completed": 
                    reward_value = goal.priority * 0.6  # Higher reward for completion
                elif status == "failed": 
                    reward_value = -goal.priority * 0.4  # Punish failure
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
                            "associated_need": goal.associated_need
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    await reward_system.process_reward_signal(reward_signal)
                    notifications["reward_system"] = {
                        "success": True,
                        "reward_value": reward_value,
                        "source": "GoalManager"
                    }
            except Exception as e:
                notifications["reward_system"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Notify MetaCore if applicable
        if self.brain and hasattr(self.brain, 'meta_core'):
            try:
                meta_core = getattr(self.brain, 'meta_core')
                if hasattr(meta_core, 'record_goal_outcome'):
                    await meta_core.record_goal_outcome(goal.model_dump())
                    notifications["meta_core"] = {
                        "success": True,
                        "recorded_goal": goal_id,
                        "status": status
                    }
            except Exception as e:
                notifications["meta_core"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "goal_id": goal_id,
            "status": status,
            "notifications": notifications
        }

    async def add_goal_with_motivation(self, 
                                    goal_data: GoalCreationWithMotivation) -> str:
        """Creates a goal with emotional motivation and time horizon specifications"""
        # Create base fields for the goal
        goal_fields = {
            "description": goal_data.description,
            "priority": goal_data.priority,
            "source": "explicit",
            "associated_need": goal_data.associated_need,
            "time_horizon": goal_data.time_horizon
        }
        
        # Add deadline if provided
        if goal_data.deadline:
            try:
                deadline = datetime.datetime.fromisoformat(goal_data.deadline)
                goal_fields["deadline"] = deadline
            except ValueError:
                logger.warning(f"Invalid deadline format: {goal_data.deadline}")
        
        # Add emotional motivation if provided
        if goal_data.emotional_motivation:
            goal_fields["emotional_motivation"] = goal_data.emotional_motivation
        
        # Create goal relationships if parent_goal_id is provided
        relationships = GoalRelationship()
        if goal_data.parent_goal_id:
            async with self._lock:
                if goal_data.parent_goal_id in self.goals:
                    relationships.parent_goal_id = goal_data.parent_goal_id
                    relationships.relationship_type = "hierarchical"
                    goal_fields["relationships"] = relationships
        
        # Create the goal using the existing add_goal method
        goal_id = await self.add_goal(**goal_fields)
        
        # Update parent goal's relationships if applicable
        if goal_data.parent_goal_id and goal_id:
            async with self._lock:
                if goal_data.parent_goal_id in self.goals:
                    parent_goal = self.goals[goal_data.parent_goal_id]
                    parent_relationships = parent_goal.relationships
                    
                    # Add this goal as a child of the parent
                    if not parent_relationships:
                        parent_relationships = GoalRelationship()
                    
                    parent_relationships.child_goal_ids.append(goal_id)
                    
                    # Update the parent goal
                    parent_goal.relationships = parent_relationships
        
        return goal_id
    
    async def create_goal_hierarchy(self, 
                                 root_goal_data: Dict[str, Any],
                                 subgoals_data: List[Dict[str, Any]]) -> str:
        """Creates a hierarchical structure of goals with a root goal and subgoals"""
        # First create the root goal
        root_goal = GoalCreationWithMotivation(**root_goal_data)
        root_goal_id = await self.add_goal_with_motivation(root_goal)
        
        if not root_goal_id:
            return None
        
        # Create each subgoal and link to the root goal
        for subgoal_data in subgoals_data:
            subgoal_data["parent_goal_id"] = root_goal_id
            subgoal = GoalCreationWithMotivation(**subgoal_data)
            await self.add_goal_with_motivation(subgoal)
        
        return root_goal_id
    
    async def get_goal_hierarchy(self, root_goal_id: Optional[str] = None) -> List[GoalHierarchyNode]:
        """
        Retrieves the goal hierarchy as a tree structure.
        If root_goal_id is provided, returns that specific hierarchy.
        Otherwise, returns all top-level goals.
        """
        async with self._lock:
            # If a specific root goal is requested
            if root_goal_id:
                if root_goal_id not in self.goals:
                    return []
                root_goal = self.goals[root_goal_id]
                return [await self._build_goal_node(root_goal)]
            
            # Otherwise, find all top-level goals (goals without parents)
            top_level_goals = []
            for goal_id, goal in self.goals.items():
                # Check if this goal has no parent
                if not hasattr(goal, 'relationships') or not goal.relationships or not goal.relationships.parent_goal_id:
                    # This is a top-level goal
                    top_level_goals.append(await self._build_goal_node(goal))
            
            return top_level_goals
    
    async def _build_goal_node(self, goal) -> GoalHierarchyNode:
        """Helper method to recursively build goal hierarchy nodes"""
        # Create the current node
        node = GoalHierarchyNode(
            goal_id=goal.id,
            description=goal.description,
            time_horizon=goal.time_horizon if hasattr(goal, 'time_horizon') else TimeHorizon.MEDIUM_TERM,
            status=goal.status,
            priority=goal.priority,
            children=[]
        )
        
        # Add child goals if any
        if hasattr(goal, 'relationships') and goal.relationships and goal.relationships.child_goal_ids:
            for child_id in goal.relationships.child_goal_ids:
                if child_id in self.goals:
                    child_node = await self._build_goal_node(self.goals[child_id])
                    node.children.append(child_node)
        
        return node
    
    async def analyze_goal_motivations(self) -> GoalMotivationAnalysis:
        """Analyzes patterns in goal motivations across all goals"""
        async with self._lock:
            # Initialize analysis data
            emotional_needs = {}
            chemical_associations = {}
            motivation_patterns = []
            
            # Scan all goals with emotional motivations
            for goal_id, goal in self.goals.items():
                if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
                    # Count emotional needs
                    primary_need = goal.emotional_motivation.primary_need
                    emotional_needs[primary_need] = emotional_needs.get(primary_need, 0) + 1
                    
                    # Aggregate chemical associations
                    for chemical, value in goal.emotional_motivation.associated_chemicals.items():
                        chemical_associations[chemical] = chemical_associations.get(chemical, 0.0) + value
                    
                    # Check for patterns based on time horizon
                    time_horizon = goal.time_horizon if hasattr(goal, 'time_horizon') else TimeHorizon.MEDIUM_TERM
                    
                    # Add to pattern analysis
                    motivation_patterns.append({
                        "need": primary_need,
                        "time_horizon": time_horizon,
                        "status": goal.status,
                        "intensity": goal.emotional_motivation.intensity,
                        "goal_id": goal_id
                    })
            
            # Calculate primary motivations (top 3)
            primary_motivations = sorted(emotional_needs.keys(), 
                                        key=lambda x: emotional_needs[x], 
                                        reverse=True)[:3]
            
            # Normalize chemical associations
            for chemical in chemical_associations:
                chemical_associations[chemical] /= max(1, len(motivation_patterns))
            
            return GoalMotivationAnalysis(
                emotional_needs=emotional_needs,
                primary_motivations=primary_motivations,
                chemical_associations=chemical_associations,
                motivation_patterns=motivation_patterns
            )
    
    async def suggest_new_goals(self, 
                             based_on_need: Optional[str] = None, 
                             time_horizon: Optional[TimeHorizon] = None) -> List[Dict[str, Any]]:
        """
        Suggests new goals based on analysis of existing goals, emotional needs,
        and system state. Can focus on specific needs or time horizons.
        """
        # This method would use the GoalManager's AI planning agent to generate suggestions
        # based on analysis of existing goals and emotional needs
        
        # First, analyze current goal motivations
        motivation_analysis = await self.analyze_goal_motivations()
        
        # Get data about completed goals for learning patterns
        completed_goals = await self.get_all_goals(status_filter=["completed"])
        
        # Prepare context for the planning agent
        context = {
            "motivation_analysis": motivation_analysis.model_dump(),
            "completed_goals": completed_goals,
            "based_on_need": based_on_need,
            "time_horizon": time_horizon.value if time_horizon else None,
            "current_goals_count": len(self.goals)
        }
        
        # Use the planning agent to generate suggestions
        try:
            result = await Runner.run(
                self.planning_agent,
                json.dumps(context),
                context=RunContext(goal_id="goal_suggestion", brain_available=self.brain is not None),
                run_config={
                    "workflow_name": "GoalSuggestion",
                    "trace_metadata": {
                        "based_on_need": based_on_need,
                        "time_horizon": time_horizon.value if time_horizon else None
                    }
                }
            )
            
            # Extract suggestions from agent output
            suggested_goals = result.final_output.get("suggested_goals", [])
            
            return suggested_goals
        except Exception as e:
            logger.error(f"Error generating goal suggestions: {e}")
            return []
    
    # ==================================================
    # Method overrides for time horizon integration
    # ==================================================
    
    def _modify_get_prioritized_goals(self) -> List[Goal]:
        """Returns active and pending goals sorted by priority with time horizon considerations."""
        # This would modify the existing get_prioritized_goals method
        
        # Create a copy of the active/pending goals while holding the lock
        goals_copy = {}
        try:
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
            # Basic priority from existing method
            age_penalty = (now - g.creation_time).total_seconds() / (3600 * 24)  # Age in days
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
            
            # NEW: Time horizon modifications
            time_horizon_factor = 0.0
            if hasattr(g, 'time_horizon'):
                # Short-term goals get higher base priority
                if g.time_horizon == TimeHorizon.SHORT_TERM:
                    time_horizon_factor = 0.2
                elif g.time_horizon == TimeHorizon.MEDIUM_TERM:
                    time_horizon_factor = 0.1
                # Long-term goals don't get a boost
            
            # NEW: Emotional motivation intensity boost
            motivation_boost = 0.0
            if hasattr(g, 'emotional_motivation') and g.emotional_motivation:
                motivation_boost = g.emotional_motivation.intensity * 0.15
            
            return g.priority + (age_penalty * 0.01 * status_boost) + deadline_urgency + time_horizon_factor + motivation_boost
    
        sorted_goals = list(goals_copy.values())
        sorted_goals.sort(key=sort_key, reverse=True)
        return sorted_goals
    
    # ==================================================
    # Integration with emotional/reward systems
    # ==================================================
    
    async def _link_goal_to_emotional_system(self, goal_id: str, emotion_data: Dict[str, Any]) -> bool:
        """Creates an emotional link for a goal to influence the emotional state"""
        if not self.brain or not hasattr(self.brain, "emotional_core"):
            return False
        
        emotional_core = self.brain.emotional_core
        
        # Check if the goal exists
        async with self._lock:
            if goal_id not in self.goals:
                return False
            
            goal = self.goals[goal_id]
        
        # Link based on emotional motivation if available
        if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
            # Set up chemical changes based on goal's emotional motivation
            chemicals = goal.emotional_motivation.associated_chemicals
            
            # Apply a small anticipatory boost
            for chemical, value in chemicals.items():
                try:
                    # Small anticipatory boost for focusing on the goal
                    await emotional_core.update_neurochemical(chemical, value * 0.2)
                except Exception as e:
                    logger.error(f"Error updating neurochemical {chemical}: {e}")
        
        return True
    
    async def _process_goal_completion_reward(self, goal_id: str, result: Any) -> Dict[str, Any]:
        """Processes reward signals when a goal is completed"""
        if not self.brain or not hasattr(self.brain, "reward_system"):
            return {"success": False, "reason": "No reward system available"}
        
        reward_system = self.brain.reward_system
        
        # Check if the goal exists
        async with self._lock:
            if goal_id not in self.goals:
                return {"success": False, "reason": "Goal not found"}
            
            goal = self.goals[goal_id]
        
        # Calculate base reward based on priority and time horizon
        base_reward = goal.priority
        
        # Adjust based on time horizon (more immediate satisfaction for short-term goals)
        time_horizon_factor = 1.0
        if hasattr(goal, 'time_horizon'):
            if goal.time_horizon == TimeHorizon.SHORT_TERM:
                time_horizon_factor = 1.2  # 20% boost for short-term goal completion
            elif goal.time_horizon == TimeHorizon.LONG_TERM:
                time_horizon_factor = 0.9  # 10% reduction for long-term goals (but more satisfaction overall)
        
        # Adjust based on emotional motivation if available
        satisfaction_factor = 1.0
        if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
            satisfaction_factor = goal.emotional_motivation.expected_satisfaction
        
        # Calculate final reward
        reward_value = base_reward * time_horizon_factor * satisfaction_factor
        
        # Create context for the reward
        context = {
            "goal_id": goal_id,
            "goal_description": goal.description,
            "time_horizon": goal.time_horizon.value if hasattr(goal, 'time_horizon') else "medium_term",
            "emotional_need": goal.emotional_motivation.primary_need if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation else None,
            "achievement_type": "goal_completion"
        }
        
        try:
            # Import RewardSignal locally to avoid circular imports
            from nyx.core.reward_system import RewardSignal
            
            # Create and process reward signal
            reward_signal = RewardSignal(
                value=reward_value,
                source="GoalManager",
                context=context,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            reward_result = await reward_system.process_reward_signal(reward_signal)
            
            # Apply neurochemical effects if emotional motivation exists
            if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation and hasattr(self.brain, "emotional_core"):
                emotional_core = self.brain.emotional_core
                chemicals = goal.emotional_motivation.associated_chemicals
                
                # Apply chemical changes in stronger amounts than the anticipatory boost
                for chemical, value in chemicals.items():
                    try:
                        await emotional_core.update_neurochemical(chemical, value * 0.8)
                    except Exception as e:
                        logger.error(f"Error updating neurochemical {chemical}: {e}")
            
            return {
                "success": True,
                "reward_value": reward_value,
                "reward_result": reward_result
            }
                
        except Exception as e:
            logger.error(f"Error processing goal completion reward: {e}")
            return {"success": False, "reason": str(e)}
    
    # ==================================================
    # Functions to help with goal creation
    # ==================================================
    
    async def derive_emotional_motivation(self, goal_description: str, need: Optional[str] = None) -> EmotionalMotivation:
        """
        Analyzes a goal description to derive likely emotional motivation,
        using the emotional core if available.
        """
        # Start with default values
        motivation = EmotionalMotivation(
            primary_need=need or "accomplishment",
            intensity=0.5,
            expected_satisfaction=0.6,
            associated_chemicals={"nyxamine": 0.3, "seranix": 0.2},
            description="Derived emotional motivation"
        )
        
        # Use emotional core to refine if available
        if self.brain and hasattr(self.brain, "emotional_core"):
            emotional_core = self.brain.emotional_core
            
            # Common emotional needs mapped to chemicals
            need_to_chemicals = {
                "accomplishment": {"nyxamine": 0.4, "seranix": 0.2},
                "connection": {"oxynixin": 0.5, "seranix": 0.2},
                "security": {"seranix": 0.4, "cortanyx": -0.3},
                "control": {"adrenyx": 0.3, "cortanyx": -0.2},
                "growth": {"nyxamine": 0.3, "adrenyx": 0.2},
                "pleasure": {"nyxamine": 0.5},
                "meaning": {"nyxamine": 0.3, "seranix": 0.3, "oxynixin": 0.2},
                "efficiency": {"nyxamine": 0.2, "cortanyx": -0.2},
                "autonomy": {"adrenyx": 0.2, "nyxamine": 0.2},
                "challenge": {"adrenyx": 0.4, "nyxamine": 0.3}
            }
            
            # Analyze text for emotional content if no need specified
            if not need:
                # Simplified analysis - in a real implementation, you might use the emotional core's analysis tools
                
                # Check for key words/phrases that suggest specific needs
                lower_text = goal_description.lower()
                
                if any(word in lower_text for word in ["connect", "bond", "relate", "together", "relationship"]):
                    motivation.primary_need = "connection"
                    motivation.associated_chemicals = need_to_chemicals["connection"]
                    
                elif any(word in lower_text for word in ["grow", "improve", "better", "learn", "develop"]):
                    motivation.primary_need = "growth"
                    motivation.associated_chemicals = need_to_chemicals["growth"]
                    
                elif any(word in lower_text for word in ["secure", "safe", "protect", "prevent", "avoid"]):
                    motivation.primary_need = "security"
                    motivation.associated_chemicals = need_to_chemicals["security"]
                    
                elif any(word in lower_text for word in ["control", "manage", "direct", "lead", "organize"]):
                    motivation.primary_need = "control"
                    motivation.associated_chemicals = need_to_chemicals["control"]
                    
                elif any(word in lower_text for word in ["meaning", "purpose", "value", "important", "significant"]):
                    motivation.primary_need = "meaning"
                    motivation.associated_chemicals = need_to_chemicals["meaning"]
                    
                elif any(word in lower_text for word in ["enjoy", "fun", "pleasure", "happy", "delight"]):
                    motivation.primary_need = "pleasure"
                    motivation.associated_chemicals = need_to_chemicals["pleasure"]
                    
                elif any(word in lower_text for word in ["challenge", "difficult", "hard", "master", "overcome"]):
                    motivation.primary_need = "challenge"
                    motivation.associated_chemicals = need_to_chemicals["challenge"]
                    
                elif any(word in lower_text for word in ["efficient", "quick", "optimize", "streamline", "automate"]):
                    motivation.primary_need = "efficiency"
                    motivation.associated_chemicals = need_to_chemicals["efficiency"]
            else:
                # Use provided need if specified
                if need in need_to_chemicals:
                    motivation.associated_chemicals = need_to_chemicals[need]
            
            # Set description based on need
            need_descriptions = {
                "accomplishment": "Desire to achieve something meaningful and receive recognition",
                "connection": "Desire for authentic bonding and meaningful relationships",
                "security": "Desire for safety, stability and predictability",
                "control": "Desire to influence outcomes and direct processes",
                "growth": "Desire to improve skills, knowledge and capabilities",
                "pleasure": "Desire for enjoyment and positive experiences",
                "meaning": "Desire for purpose and significance",
                "efficiency": "Desire to optimize processes and save resources",
                "autonomy": "Desire for independence and self-direction",
                "challenge": "Desire to overcome difficult obstacles"
            }
            
            if motivation.primary_need in need_descriptions:
                motivation.description = need_descriptions[motivation.primary_need]
        
        return motivation
    
    # ==================================================
    # Modified update_goal_status method to handle hierarchical goals
    # ==================================================
    
    async def update_goal_status_with_hierarchy(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None):
        """Updates goal status with hierarchy considerations - handles parent/child relationships"""
        async with self._lock:
            if goal_id not in self.goals:
                logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
                return
    
            goal = self.goals[goal_id]
            old_status = goal.status
            if old_status == status: 
                return  # No change
    
            # Update the goal's status
            goal.status = status
            goal.last_error = error
    
            if status in ["completed", "failed", "abandoned"]:
                goal.completion_time = datetime.datetime.now()
                self.active_goals.discard(goal_id)
                
                # Update statistics
                if status == "completed":
                    self.goal_statistics["completed"] += 1
                    # Process completion reward
                    await self._process_goal_completion_reward(goal_id, result)
                    
                    # Update progress of parent goal if this is a child goal
                    if hasattr(goal, 'relationships') and goal.relationships and goal.relationships.parent_goal_id:
                        parent_id = goal.relationships.parent_goal_id
                        if parent_id in self.goals:
                            parent_goal = self.goals[parent_id]
                            
                            # Calculate new progress percentage
                            if hasattr(parent_goal, 'relationships') and parent_goal.relationships:
                                total_children = len(parent_goal.relationships.child_goal_ids)
                                if total_children > 0:
                                    # Count completed children
                                    completed_children = 0
                                    for child_id in parent_goal.relationships.child_goal_ids:
                                        if child_id in self.goals and self.goals[child_id].status == "completed":
                                            completed_children += 1
                                    
                                    # Update parent progress
                                    if hasattr(parent_goal, 'progress'):
                                        parent_goal.progress = completed_children / total_children
                                        
                                        # If all children completed, mark parent as completed
                                        if completed_children == total_children:
                                            await self.update_goal_status(parent_id, "completed", result="All subgoals completed")
                                            
                elif status == "failed":
                    self.goal_statistics["failed"] += 1
                elif status == "abandoned":
                    self.goal_statistics["abandoned"] += 1
    
            logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")
    
        # Notify systems
        await self._notify_systems(goal_id, status, result, error)
        
    @function_tool
    async def _check_concurrency_limits(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Check if more goals can be activated based on concurrency limits
        
        Returns:
            Concurrency information
        """
        async with self._lock:
            active_count = len(self.active_goals)
            can_activate = active_count < self.max_concurrent_goals
            remaining_slots = max(0, self.max_concurrent_goals - active_count)
            
            active_goals = []
            for goal_id in self.active_goals:
                if goal_id in self.goals:
                    active_goals.append({
                        "id": goal_id,
                        "description": self.goals[goal_id].description,
                        "priority": self.goals[goal_id].priority
                    })
        
        return {
            "active_count": active_count,
            "max_concurrent": self.max_concurrent_goals,
            "can_activate_more": can_activate,
            "remaining_slots": remaining_slots,
            "active_goals": active_goals
        }
    
    # Additional tools used by multiple agents
    @function_tool
    async def _get_available_actions(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get available actions that can be used in plans
        
        Returns:
            Available actions with descriptions
        """
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
        
        # Build list with descriptions
        actions_with_descriptions = []
        for action in available_actions:
            description = self._generate_action_description(action)
            actions_with_descriptions.append({
                "name": action,
                "description": description
            })
        
        return {
            "count": len(actions_with_descriptions),
            "actions": actions_with_descriptions
        }
    
    @function_tool
    async def _get_action_description(self, ctx: RunContextWrapper, action: str) -> Dict[str, Any]:
        """
        Get a description for a specific action
        
        Args:
            action: The action name
            
        Returns:
            Action description
        """
        description = self._generate_action_description(action)
        return {
            "action": action,
            "description": description,
            "is_available": action in await self._get_available_actions(ctx)
        }
    
    @function_tool
    async def _get_goal_details(self, ctx: RunContextWrapper, goal_id: str) -> Dict[str, Any]:
        """
        Get details about a specific goal
        
        Args:
            goal_id: The goal ID
            
        Returns:
            Goal details
        """
        async with self._lock:
            if goal_id not in self.goals:
                return {
                    "success": False,
                    "error": f"Goal {goal_id} not found"
                }
                
            goal = self.goals[goal_id]
            
            return {
                "success": True,
                "id": goal.id,
                "description": goal.description,
                "status": goal.status,
                "priority": goal.priority,
                "source": goal.source,
                "associated_need": goal.associated_need,
                "creation_time": goal.creation_time.isoformat(),
                "completion_time": goal.completion_time.isoformat() if goal.completion_time else None,
                "deadline": goal.deadline.isoformat() if goal.deadline else None,
                "has_plan": len(goal.plan) > 0,
                "plan_step_count": len(goal.plan),
                "current_step_index": goal.current_step_index,
                "last_error": goal.last_error
            }
    
    @function_tool
    async def _get_recent_goals(self, ctx: RunContextWrapper, limit: int = 3) -> Dict[str, Any]:
        """
        Get recently completed goals
        
        Args:
            limit: Maximum number of goals to return
            
        Returns:
            Recent goals
        """
        recent_goals = []
        async with self._lock:
            completed_goals = [
                g for g in self.goals.values() 
                if g.status == "completed" and g.completion_time is not None
            ]
            
            # Sort by completion time (newest first)
            completed_goals.sort(key=lambda g: g.completion_time, reverse=True)
            
            # Get recent goals
            for goal in completed_goals[:limit]:
                recent_goals.append({
                    "id": goal.id,
                    "description": goal.description,
                    "completion_time": goal.completion_time.isoformat(),
                    "priority": goal.priority,
                    "source": goal.source,
                    "associated_need": goal.associated_need,
                    "steps": [
                        {
                            "description": step.description,
                            "action": step.action
                        }
                        for step in goal.plan[:3]  # First 3 steps of each goal
                    ]
                })
        
        return {
            "count": len(recent_goals),
            "goals": recent_goals
        }

    # Guardrail functions for input validation
    async def _validate_goal_input(self, ctx, agent, input_data):
        """Guardrail function to validate goal input"""
        # Basic validation for required fields
        if not isinstance(input_data, dict):
            return GuardrailFunctionOutput(
                output_info={"error": "Input must be a dictionary"},
                tripwire_triggered=True
            )
        
        if "description" not in input_data:
            return GuardrailFunctionOutput(
                output_info={"error": "Goal must have a description"},
                tripwire_triggered=True
            )
        
        description = input_data.get("description", "")
        if not description or len(description.strip()) < 5:
            return GuardrailFunctionOutput(
                output_info={"error": "Goal description must be meaningful"},
                tripwire_triggered=True
            )
        
        # Check for suspicious content in dominance-related goals
        lower_desc = description.lower()
        dominance_keywords = ["dominance", "command", "punishment", "submission", "control"]
        
        if any(keyword in lower_desc for keyword in dominance_keywords):
            # Run validation through dominance agent if available
            if self.brain and hasattr(self.brain, 'dominance_system'):
                try:
                    dominance_system = getattr(self.brain, 'dominance_system')
                    if hasattr(dominance_system, 'evaluate_dominance_step_appropriateness'):
                        evaluation = await dominance_system.evaluate_dominance_step_appropriateness(
                            "add_goal", input_data, input_data.get("user_id", "default")
                        )
                        
                        if evaluation.get("action") != "proceed":
                            return GuardrailFunctionOutput(
                                output_info={"error": evaluation.get("reason", "Dominance goal rejected")},
                                tripwire_triggered=True
                            )
                except Exception as e:
                    # Log but don't block if evaluation fails
                    logger.error(f"Error in dominance validation: {e}")
        
        # All validation passed
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )

    async def add_goal(self, description: str, priority: float = 0.5, source: str = "unknown",
                     associated_need: Optional[str] = None, plan: Optional[List[Dict]] = None,
                     user_id: Optional[str] = None, deadline: Optional[datetime.datetime] = None) -> str:
        """Adds a new goal, optionally generating a plan if none is provided."""
        if not description:
            raise ValueError("Goal description cannot be empty.")

        # Create the goal object
        async with self._lock:
            goal = Goal(
                description=description,
                priority=priority,
                source=source,
                associated_need=associated_need,
                deadline=deadline,
                plan=[]  # Start with empty plan, generate/add later
            )
            self.goals[goal.id] = goal
            self.goal_statistics["created"] += 1

        logger.info(f"Added goal '{goal.id}': {description} (Priority: {priority:.2f}, Source: {source})")

        # Process the goal through the orchestration system
        with trace(workflow_name="Goal_Management", group_id=self.trace_group_id):
            # Create context for this goal management process
            context = RunContext(
                goal_id=goal.id,
                brain_available=self.brain is not None,
                user_id=user_id
            )
            
            # Run goal processing through the orchestration agent
            result = await Runner.run(
                self.orchestration_agent,
                json.dumps({
                    "goal": {
                        "id": goal.id,
                        "description": description,
                        "priority": priority,
                        "source": source,
                        "associated_need": associated_need
                    },
                    "has_plan": plan is not None,
                    "user_id": user_id
                }),
                context=context,
                run_config={
                    "workflow_name": "GoalProcessing",
                    "trace_metadata": {
                        "goal_id": goal.id,
                        "goal_description": description
                    }
                }
            )
            
            # If a plan was provided, use it
            if plan:
                try:
                    plan_steps = [GoalStep(**step_data) for step_data in plan]
                    async with self._lock:
                        if goal.id in self.goals:  # Check if goal still exists
                            self.goals[goal.id].plan = plan_steps
                            self.goals[goal.id].status = "pending"  # Ready to be activated
                except Exception as e:
                    logger.error(f"Invalid plan structure provided for goal '{goal.id}': {e}")

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
                # Create context for the agent
                context = RunContext(goal_id=goal_id, brain_available=self.brain is not None)
                
                # Generate plan through the Planning Agent
                result = await Runner.run(
                    self.planning_agent,
                    json.dumps({
                        "goal": {
                            "id": goal.id,
                            "description": goal.description,
                            "priority": goal.priority,
                            "source": goal.source,
                            "associated_need": goal.associated_need
                        }
                    }),
                    context=context,
                    run_config={
                        "workflow_name": "GoalPlanning",
                        "trace_metadata": {
                            "goal_id": goal_id,
                            "goal_description": goal.description
                        }
                    }
                )
                
                # Extract plan from result
                plan_result = result.final_output_as(PlanGenerationResult)
                plan_data = plan_result.plan
                
                # Validate the plan
                validation_result = await Runner.run(
                    self.plan_validation_agent,
                    json.dumps({
                        "goal": {
                            "id": goal.id,
                            "description": goal.description
                        },
                        "plan": plan_data
                    }),
                    context=context
                )
                
                validation_output = validation_result.final_output_as(PlanValidationResult)
                
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
                        
                        if not validation_output.is_valid:
                            logger.warning(f"Plan validation raised concerns: {validation_output.reason}")
                            
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

        # Create context for the step execution
        context = RunContext(
            goal_id=goal_id,
            brain_available=self.brain is not None,
            current_step_index=step_index
        )
        
        # Execute the step through the Step Execution Agent
        with trace(workflow_name="ExecuteGoalStep", group_id=self.trace_group_id):
            step_result = await Runner.run(
                self.step_execution_agent,
                json.dumps({
                    "goal_id": goal_id,
                    "step": step.model_dump(),
                    "step_index": step_index
                }),
                context=context,
                run_config={
                    "workflow_name": "StepExecution",
                    "trace_metadata": {
                        "goal_id": goal_id,
                        "step_id": step.step_id,
                        "action": step.action
                    }
                }
            )
            
            execution_result = step_result.final_output_as(StepExecutionResult)
            
            # Update goal based on execution result
            async with self._lock:
                if goal_id not in self.goals:
                    return None
                    
                goal = self.goals[goal_id]
                
                # Find step again (might have changed)
                if step_index >= len(goal.plan) or goal.plan[step_index].step_id != step.step_id:
                    logger.warning(f"Step structure changed during execution for goal {goal_id}")
                    return {"error": "Step structure changed", "goal_id": goal_id}
                
                step = goal.plan[step_index]
                
                # Update step with execution result
                step.status = "completed" if execution_result.success else "failed"
                step.result = execution_result.result
                step.error = execution_result.error
                step.end_time = datetime.datetime.now()
                
                # Update execution history
                goal.execution_history.append({
                    "step_id": step.step_id,
                    "action": step.action,
                    "status": step.status,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "next_action": execution_result.next_action,
                    "error": step.error
                })
                
                # Process next action
                if execution_result.next_action == "continue":
                    # Move to next step
                    if step.status == "completed":
                        goal.current_step_index += 1
                        if goal.current_step_index >= len(goal.plan):
                            await self.update_goal_status(goal.id, "completed", result=step.result)
                            self.active_goals.discard(goal.id)
                    else:  # Failed
                        await self.update_goal_status(goal.id, "failed", error=step.error)
                        self.active_goals.discard(goal.id)
                        
                elif execution_result.next_action == "retry":
                    # Leave index the same to retry the step
                    if "retry_count" not in goal.execution_history[-1]:
                        goal.execution_history[-1]["retry_count"] = 1
                    else:
                        goal.execution_history[-1]["retry_count"] += 1
                        
                    # Check if max retries exceeded
                    if goal.execution_history[-1]["retry_count"] >= 3:
                        await self.update_goal_status(goal.id, "failed", error=f"Max retries exceeded for step {step.step_id}")
                        self.active_goals.discard(goal.id)
                        
                elif execution_result.next_action == "skip":
                    # Mark as skipped and move to next step
                    step.status = "skipped"
                    goal.current_step_index += 1
                    if goal.current_step_index >= len(goal.plan):
                        await self.update_goal_status(goal.id, "completed", result="Plan completed after skipping steps")
                        self.active_goals.discard(goal.id)
                        
                elif execution_result.next_action == "abort":
                    # Abort the entire goal
                    await self.update_goal_status(goal.id, "failed", error=f"Goal aborted: {step.error}")
                    self.active_goals.discard(goal.id)

        return {"executed_step": step.model_dump(), "goal_id": goal_id, "next_action": execution_result.next_action}

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

        # Notify systems using the orchestration agent
        try:
            with trace(workflow_name="GoalStatusUpdate", group_id=self.trace_group_id):
                context = RunContext(goal_id=goal_id)
                
                await self._notify_systems(
                    RunContextWrapper(context=context), 
                    goal_id=goal_id, 
                    status=status, 
                    result=result, 
                    error=error
                )
        except Exception as e:
            logger.error(f"Error in notifying systems about goal status change: {e}")

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
