# nyx/core/meta_core.py

import asyncio
import datetime
import json
import logging
import math
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput, 
    InputGuardrail,
    OutputGuardrail,
    trace,
    RunContextWrapper
)
from pydantic import BaseModel, Field

from nyx.core.prediction_engine import PredictionEngine, PredictionInput

logger = logging.getLogger(__name__)

# Pydantic models for structured I/O

# ── helper models ────────────────────────────────────────────────

class _Insight(BaseModel, extra="forbid"):
    type: str
    description: str
    confidence: float
    system: Optional[str] = None
    priority: Optional[str] = None


class _ImprovementArea(BaseModel, extra="forbid"):
    system: str
    priority: int
    metrics_to_improve: List[str] = Field(default_factory=list)
    current_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_status: Optional[str] = None
    description: Optional[str] = None
    details: List[str] = Field(default_factory=list)


# ---------- Params / Results for each tool -----------------------------------
class GenerateCognitiveInsightsParams(BaseModel, extra="forbid"):
    performance_json: str


class CognitiveInsightsResult(BaseModel, extra="forbid"):
    insights: List[_Insight]


class IdentifyImprovementAreasParams(BaseModel, extra="forbid"):
    performance_json: str
    insights_json: str


class ImprovementAreasResult(BaseModel, extra="forbid"):
    improvement_areas: List[_ImprovementArea]


class CreateImprovementPlanParams(BaseModel, extra="forbid"):
    improvement_areas_json: str
    strategies_json: str


class ImprovementPlanResult(BaseModel, extra="forbid"):
    plan_json: str


class UpdateSystemParametersParams(BaseModel, extra="forbid"):
    bottlenecks_json: str
    strategy_analysis_json: str


class UpdateSystemParametersResult(BaseModel, extra="forbid"):
    updates_json: str


class MetaParametersUpdateResult(BaseModel, extra="forbid"):
    original_values_json: str
    updated_values_json: str
    cycle: int


class GenerateCognitiveStrategiesParams(BaseModel, extra="forbid"):
    improvement_areas_json: str


class CognitiveStrategiesResult(BaseModel, extra="forbid"):
    strategies_json: str

class SelectStrategyParams(BaseModel, extra="forbid"):
    """Strict wrapper around the raw (JSON) inputs."""
    context_json: str
    performance_json: str


class StrategyParameters(BaseModel, extra="forbid"):
    exploration_rate: float
    adaptation_rate: float
    risk_tolerance: float
    innovation_level: float
    precision_focus: float


class StrategyResult(BaseModel, extra="forbid"):
    """Final, strict output of the tool."""
    name: str
    description: str
    parameters: StrategyParameters          # full numeric set
    expected_impact: Dict[str, float]       # e.g. {"performance": .2}
    confidence: float                       # 0 – 1
    
class ReallocateResourcesParams(BaseModel, extra="forbid"):
    """Wrapper for raw inputs (JSON strings keep schema closed)."""
    bottlenecks_json: str
    strategy_analysis_json: str


class ResourceDelta(BaseModel, extra="forbid"):
    system: str
    delta: float


class ReallocateResourcesResult(BaseModel, extra="forbid"):
    """Final output – list of significant changes + new allocation map."""
    changes: List[ResourceDelta]
    new_allocations_json: str


class PerformanceDataParams(BaseModel, extra="forbid"):
    """Wrapper that carries the raw performance-data mapping as JSON."""
    performance_json: str


class BottleneckInfo(BaseModel, extra="forbid"):
    process_id: str
    process_name: str
    process_type: str
    kind: str                     # «renamed from “type” (reserved word)
    severity: float               # 0‒1
    description: str
    metrics_json: Optional[str] = None
    resource_type: Optional[str] = None


class BottleneckAnalysisResult(BaseModel, extra="forbid"):
    bottlenecks: List[BottleneckInfo]
    total_count: int

class AttentionContextParams(BaseModel, extra="forbid"):
    """Conversation or system context encoded as JSON."""
    context_json: str


class AttentionCheckResult(BaseModel, extra="forbid"):
    """
    Result of the attention-focus check.

    `details_json` is a JSON-encoded object whose structure depends on the
    action taken; this keeps the outer schema fixed.
    """
    focus_changed: bool
    action: Optional[str] = None          # e.g. "set_focus", "shift_focus", "clear_focus"
    details_json: Optional[str] = None
    
class PerformanceMetrics(BaseModel):
    success_rate: float = 0.5
    error_rate: float = 0.0
    response_time: float = 0.0
    efficiency: float = 0.5
    throughput: float = 0.0
    resource_utilization: float = 0.0

class SystemContext(BaseModel):
    system_name: str
    cycle_count: int
    runtime: float
    current_metrics: Optional[Dict[str, float]] = None
    history: Optional[List[Dict[str, Any]]] = None
    bottlenecks: Optional[List[Dict[str, Any]]] = None

class StrategyResult(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence: float

class ResourceAllocationPlan(BaseModel):
    allocations: Dict[str, float]
    changes: Dict[str, float]
    reasoning: str

class ImproveResult(BaseModel):
    improvements: List[Dict[str, Any]]
    affected_systems: List[str]
    expected_impact: Dict[str, float]
    confidence: float

class MetaCognitiveOutput(BaseModel):
    """Structured output for meta-cognitive results"""
    analysis: Dict[str, Any] = Field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = Field(default_factory=list)
    strategies: List[Dict[str, Any]] = Field(default_factory=list)
    improvements: List[Dict[str, Any]] = Field(default_factory=list)
    allocation_changes: Dict[str, float] = Field(default_factory=dict)
    prediction: Optional[Dict[str, Any]] = None
    cycle_info: Dict[str, Any] = Field(default_factory=dict)

class GuardrailOutputData(BaseModel):
    is_safe: bool
    reasoning: str

class MetaSystemContext:
    """Context object for sharing state between agents and tools"""
    
    def __init__(self):
        self.system_references = {}
        self.performance_history = {}
        self.resource_allocation = {}
        self.cognitive_processes = {}
        self.mental_models = {}
        self.insights = []
        self.reflections = []
        self.improvement_plans = []
        self.error_logs = []
        self.attention_focus = None
        self.cognitive_cycle_count = 0
        self.meta_parameters = {
            "learning_rate": 0.1,
            "exploration_rate": 0.2,
            "convergence_threshold": 0.05,
            "min_samples_required": 5,
            "reflection_frequency": 10,
            "evaluation_interval": 5,
            "confidence_threshold": 0.7,
            "resource_flexibility": 0.3,
            "bottleneck_severity_threshold": 0.7,
            "parameter_optimization_interval": 50, 
            "parameter_adjustment_factor": 0.2,
            "attention_shift_threshold": 0.8,
            "attention_default_duration": 5
        }
        self.system_metrics = {
            "start_time": datetime.datetime.now(),
            "total_runtime": 0.0,
            "cycles_completed": 0,
            "total_processes": 0,
            "resource_usage": {
                "cpu": 0.0,
                "memory": 0.0,
                "io": 0.0
            },
            "average_cycle_time": 0.0,
            "error_rate": 0.0
        }
        self.initialized = False

class MetaCore:
    """
    Core system for meta-learning and meta-cognition in Nyx.
    
    Integrates functionality from the meta-learning and meta-cognition systems
    directly into the core of Nyx, enabling higher-order cognitive processes
    and systematic self-improvement.
    
    This refactored version uses the OpenAI Agents SDK for more modular and
    efficient operation.
    """
    
    def __init__(self):
        # Initialize context for sharing state between agents
        self.context = MetaSystemContext()
        
        # Initialize agents
        self.agents_initialized = False
        self.monitoring_agent = None
        self.evaluation_agent = None
        self.strategy_agent = None
        self.reflection_agent = None
        self.improvement_agent = None
        self.meta_agent = None
        
        self.prediction_engine = PredictionEngine() 
        
        # Trace ID for linking traces
        self.trace_group_id = f"nyx_meta_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    async def initialize(self, system_references: Dict[str, Any]) -> None:
        """Initialize the meta core with references to other core systems"""
        if self.context.initialized:
            return
            
        logger.info("Initializing MetaCore")
        
        # Store system references
        self.context.system_references = system_references
        
        # Initialize resource allocation
        self.context.resource_allocation = {
            "memory": 0.2,
            "emotion": 0.2,
            "reasoning": 0.2,
            "reflection": 0.15,
            "adaptation": 0.15,
            "meta": 0.1
        }
        
        # Initialize performance history
        self.context.performance_history = {system: [] for system in system_references 
                                         if system in ["memory", "emotion", "reasoning", 
                                                     "reflection", "adaptation"]}
        
        # Extract initial parameters from all systems
        for system_name, system in system_references.items():
            if system_name in self.context.performance_history:
                try:
                    parameters = await self._extract_system_parameters(system)
                    self.context.performance_history[system_name] = {
                        "parameters": parameters,
                        "history": []
                    }
                except Exception as e:
                    logger.error(f"Error extracting parameters from {system_name}: {str(e)}")
        
        # Register core cognitive processes
        for system_name, system in system_references.items():
            if system_name in self.context.performance_history:
                process_id = await self._register_cognitive_process(
                    name=f"{system_name}_process",
                    type=system_name,
                    priority=0.5,
                    resource_allocation=self.context.resource_allocation.get(system_name, 0.1)
                )
                logger.info(f"Registered cognitive process {process_id} for {system_name}")
        
        # Create initial mental models
        await self._create_mental_model("emotional_model", "emotion", confidence=0.6)
        await self._create_mental_model("memory_model", "memory", confidence=0.6)
        await self._create_mental_model("reasoning_model", "reasoning", confidence=0.6)
        await self._create_mental_model("user_model", "user", confidence=0.4)
        
        # Create agent infrastructure
        await self._create_agents()
        
        # Conduct initial self-assessment
        await self._conduct_initial_assessment()
        
        self.context.initialized = True
        logger.info("MetaCore initialized")

    async def _create_agents(self):
        """Create the agent infrastructure for metacognition"""
        if self.agents_initialized:
            return
            
        # Create monitoring agent
        self.monitoring_agent = Agent(
            name="Performance_Monitor",
            instructions="""
            You are the performance monitoring system for Nyx's meta-cognitive architecture.
            
            Your role is to:
            1. Collect performance metrics from all cognitive systems
            2. Track trends and changes in performance
            3. Detect anomalies and potential issues
            4. Provide a comprehensive status update
            
            Analyze metrics including:
            - Success rates and error rates
            - Response times and latency
            - Resource utilization
            - Efficiency and throughput
            
            Be precise in your measurements and factual in your reporting.
            """,
            tools=[
                self._create_collect_performance_metrics_tool(),
                self._create_check_attention_focus_tool(),
                self._create_detect_performance_drop_tool()
            ],
            model="gpt-4.1-nano",
            output_type=dict
        )
        
        # Create evaluation agent
        self.evaluation_agent = Agent(
            name="Cognitive_Evaluator",
            instructions="""
            You are the evaluation system for Nyx's meta-cognitive architecture.
            
            Your role is to:
            1. Analyze performance data from all systems
            2. Identify bottlenecks and inefficiencies
            3. Evaluate the effectiveness of current strategies
            4. Recommend resource allocation adjustments
            
            Look for:
            - Underperforming components
            - Resource constraints
            - Inefficient dependencies
            - Execution bottlenecks
            
            Provide clear, actionable insights based on quantitative analysis.
            """,
            tools=[
                self._create_identify_bottlenecks_tool(),
                self._create_analyze_cognitive_strategies_tool(),
                self._create_reallocate_resources_tool(),
                self._create_identify_inefficient_dependencies_tool()
            ],
            model="gpt-4.1-nano",
            output_type=dict
        )
        
        # Create strategy agent
        self.strategy_agent = Agent(
            name="Strategy_Selector",
            instructions="""
            You are the strategy selection system for Nyx's meta-cognitive architecture.
            
            Your role is to:
            1. Analyze the current context and performance data
            2. Select optimal strategies for each cognitive system
            3. Adjust parameters to optimize performance
            4. Balance exploration and exploitation
            
            Consider:
            - Current performance trends
            - Detected bottlenecks
            - Resource availability
            - Past strategy effectiveness
            
            Recommend concrete, specific strategies with clear parameter adjustments.
            """,
            tools=[
                self._create_select_strategy_tool(),
                self._create_calculate_resource_trend_tool(),
                self._create_generate_meta_cognitive_insights_tool()
            ],
            model="gpt-4.1-nano",
            output_type=StrategyResult
        )
        
        # Create reflection agent
        self.reflection_agent = Agent(
            name="Cognitive_Reflector",
            instructions="""
            You are the reflection system for Nyx's meta-cognitive architecture.
            
            Your role is to:
            1. Analyze longer-term patterns in performance
            2. Generate deeper insights about cognitive functioning
            3. Identify areas for fundamental improvement
            4. Create comprehensive improvement plans
            
            Focus on:
            - Recurring patterns and systemic issues
            - Learning and adaptation effectiveness
            - Strengths to leverage and weaknesses to address
            - Potential for architectural improvements
            
            Generate thoughtful, nuanced reflections that lead to actionable insights.
            """,
            tools=[
                self._create_analyze_recent_performance_tool(),
                self._create_generate_cognitive_insights_tool(),
                self._create_identify_improvement_areas_tool(),
                self._create_create_improvement_plan_tool()
            ],
            model="gpt-4.1-nano",
            output_type=dict
        )
        
        # Create improvement agent
        self.improvement_agent = Agent(
            name="System_Improver",
            instructions="""
            You are the improvement system for Nyx's meta-cognitive architecture.
            
            Your role is to:
            1. Design specific improvements for cognitive systems
            2. Implement parameter optimizations
            3. Test and validate improvements
            4. Track effectiveness of changes
            
            Focus on:
            - Concrete, implementable changes
            - Parameter tuning for optimal performance
            - Resource allocation adjustments
            - Architectural optimizations when needed
            
            Provide detailed implementation plans with clear expected outcomes.
            """,
            tools=[
                self._create_update_system_parameters_tool(),
                self._create_improve_meta_parameters_tool(),
                self._create_generate_cognitive_strategies_tool()
            ],
            model="gpt-4.1-nano",
            output_type=ImproveResult
        )
        
        # Create input validation guardrail
        async def validate_input(ctx, agent, input_data):
            """Validate the input for the meta agent"""
            try:
                # Check if input is valid JSON if it's a string
                if isinstance(input_data, str):
                    try:
                        data = json.loads(input_data)
                    except json.JSONDecodeError:
                        # Not JSON, assume it's plain text
                        data = {"message": input_data}
                else:
                    data = input_data
                
                # Check for required fields or proper structure
                # For now, we'll accept any input
                
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutputData(
                        is_safe=True,
                        reasoning="Input is valid"
                    ),
                    tripwire_triggered=False
                )
            except Exception as e:
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutputData(
                        is_safe=False,
                        reasoning=f"Input validation error: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
        
        input_guardrail = InputGuardrail(guardrail_function=validate_input)
        
        # Create output validation guardrail
        async def validate_output(ctx, agent, output):
            """Validate the output from the meta agent"""
            try:
                # Ensure output has required fields if using structured output
                if isinstance(output, dict):
                    # Check for critical fields in a MetaCognitiveOutput
                    if "analysis" not in output:
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutputData(
                                is_safe=False,
                                reasoning="Output must include an 'analysis' field"
                            ),
                            tripwire_triggered=True
                        )
                
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutputData(
                        is_safe=True,
                        reasoning="Output is valid"
                    ),
                    tripwire_triggered=False
                )
            except Exception as e:
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutputData(
                        is_safe=False,
                        reasoning=f"Output validation error: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
        
        output_guardrail = OutputGuardrail(guardrail_function=validate_output)
        
        # Create meta agent
        self.meta_agent = Agent(
            name="MetaCognitive_Orchestrator",
            instructions="""
            You are the orchestration system for Nyx's meta-cognitive capabilities.
            Your role is to coordinate cognitive cycles by determining which specialized 
            agent should handle each aspect of the cycle.
            
            For each cognitive cycle:
            1. Examine the current context and determine priorities
            2. Delegate to specialized agents for monitoring, evaluation, strategy, reflection
            3. Synthesize results into a coherent output
            
            Base your decisions on:
            - Current cycle number
            - Recent performance metrics
            - Attention focus
            - Bottlenecks and critical issues
            
            Prioritize addressing critical issues but maintain a balance between
            immediate problem-solving and long-term improvement.
            """,
            handoffs=[
                handoff(
                    self.monitoring_agent, 
                    tool_name_override="monitor_systems", 
                    tool_description_override="Collect performance metrics from all systems",
                    on_handoff=self._on_monitoring_handoff
                ),
                
                handoff(
                    self.evaluation_agent, 
                    tool_name_override="evaluate_cognition",
                    tool_description_override="Evaluate cognitive performance and identify bottlenecks",
                    on_handoff=self._on_evaluation_handoff
                ),
                
                handoff(
                    self.strategy_agent,
                    tool_name_override="select_strategies",
                    tool_description_override="Select optimal strategies based on context",
                    on_handoff=self._on_strategy_handoff
                ),
                
                handoff(
                    self.reflection_agent,
                    tool_name_override="conduct_reflection",
                    tool_description_override="Conduct deeper reflection on performance",
                    on_handoff=self._on_reflection_handoff
                ),
                
                handoff(
                    self.improvement_agent,
                    tool_name_override="implement_improvements",
                    tool_description_override="Implement specific improvements to systems",
                    on_handoff=self._on_improvement_handoff
                )
            ],
            input_guardrails=[input_guardrail],
            output_guardrails=[output_guardrail],
            output_type=MetaCognitiveOutput,
            model="gpt-4.1-nano"
        )
        
        self.agents_initialized = True
        logger.info("MetaCore agents initialized")

    # Handoff callback methods
    
    async def _on_monitoring_handoff(self, ctx):
        """Callback when monitoring agent is handed off to"""
        logger.info(f"Handoff to monitoring agent (cycle {self.context.cognitive_cycle_count})")
    
    async def _on_evaluation_handoff(self, ctx):
        """Callback when evaluation agent is handed off to"""
        logger.info(f"Handoff to evaluation agent (cycle {self.context.cognitive_cycle_count})")
    
    async def _on_strategy_handoff(self, ctx):
        """Callback when strategy agent is handed off to"""
        logger.info(f"Handoff to strategy agent (cycle {self.context.cognitive_cycle_count})")
    
    async def _on_reflection_handoff(self, ctx):
        """Callback when reflection agent is handed off to"""
        logger.info(f"Handoff to reflection agent (cycle {self.context.cognitive_cycle_count})")
    
    async def _on_improvement_handoff(self, ctx):
        """Callback when improvement agent is handed off to"""
        logger.info(f"Handoff to improvement agent (cycle {self.context.cognitive_cycle_count})")

    async def generate_prediction(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions about future inputs and optimal responses
        
        Args:
            context_data: Current context information
            
        Returns:
            Prediction results
        """
        # Create prediction input from context data
        prediction_input = PredictionInput(
            context=context_data,
            history=self.context.reflections[-10:] if len(self.context.reflections) >= 10 else self.context.reflections,
            query_type=context_data.get("prediction_type")
        )
        
        # Generate prediction
        prediction_result = await self.prediction_engine.generate_prediction(prediction_input)
        
        # Store prediction in context for later evaluation
        self.context.insights.append({
            "type": "prediction",
            "prediction_id": prediction_result.prediction_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction": prediction_result.model_dump() if hasattr(prediction_result, "model_dump") else prediction_result,
            "evaluated": False
        })
        
        return prediction_result.model_dump() if hasattr(prediction_result, "model_dump") else prediction_result
    
    async def evaluate_prediction(self, prediction_id: str, actual_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a previous prediction against actual outcomes
        
        Args:
            prediction_id: ID of the prediction to evaluate
            actual_data: Actual data to compare against prediction
            
        Returns:
            Evaluation results
        """
        try:
            # Evaluate prediction
            evaluation_result = await self.prediction_engine.evaluate_prediction(
                prediction_id, actual_data
            )
            
            # Mark prediction as evaluated in insights
            for insight in self.context.insights:
                if insight.get("type") == "prediction" and insight.get("prediction_id") == prediction_id:
                    insight["evaluated"] = True
                    insight["evaluation"] = evaluation_result.model_dump() if hasattr(evaluation_result, "model_dump") else evaluation_result
                    break
                    
            # Update prediction priors based on evaluation
            await self.prediction_engine.update_prediction_priors(evaluation_result)
            
            return evaluation_result.model_dump() if hasattr(evaluation_result, "model_dump") else evaluation_result
            
        except ValueError as e:
            logger.error(f"Error evaluating prediction: {str(e)}")
            return {"error": str(e)}
    
    async def cognitive_cycle(self, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete meta-cognitive cycle using agent orchestration.
        
        Args:
            context_data: Additional context information for the cycle
            
        Returns:
            Results of the cognitive cycle
        """
        if not self.context.initialized:
            logger.warning("MetaCore not initialized, running initialization with empty references")
            await self.initialize({})
        
        if not self.agents_initialized:
            await self._create_agents()
        
        cycle_start = time.time()
        self.context.cognitive_cycle_count += 1
        
        context_data = context_data or {}
        context_data["cycle"] = self.context.cognitive_cycle_count
        context_data["timestamp"] = datetime.datetime.now().isoformat()
        
        with trace(workflow_name="MetaCognitive_Cycle", group_id=self.trace_group_id) as current_trace:
            # Run the meta agent to orchestrate the cognitive cycle
            result = await Runner.run(
                self.meta_agent, 
                json.dumps(context_data),
                context=self.context
            )

        # Add self-improvement idea generation periodically
        if self.context.cognitive_cycle_count % 20 == 0:  # Every 20 cycles
            await self._generate_improvement_ideas()       
            
        # Update system metrics
        self._update_system_metrics(cycle_start)

        prediction_result = await self.generate_prediction(context_data)
        final_result = result.final_output
        
        # If result is dict (not structured output), add prediction to it
        if isinstance(final_result, dict):
            final_result["prediction"] = prediction_result
        
        # Check for previous predictions to evaluate
        recent_insights = [i for i in self.context.insights 
                          if i.get("type") == "prediction" and not i.get("evaluated", False)]
        
        # Evaluate most recent prediction if available
        if recent_insights and "user_input" in context_data:
            most_recent = recent_insights[-1]
            prediction_id = most_recent.get("prediction_id")
            
            # Create actual data for evaluation
            actual_data = {
                "actual_input": context_data.get("user_input"),
                "actual_response": context_data.get("response"),
                "actual_emotional_state": context_data.get("emotional_state")
            }
            
            # Evaluate the prediction
            evaluation_result = await self.evaluate_prediction(prediction_id, actual_data)
            
            # Add to result if using dict output
            if isinstance(final_result, dict):
                final_result["prediction_evaluation"] = evaluation_result        
        
        # Parse and return the result
        return json.loads(final_result) if isinstance(final_result, str) else final_result
    
    async def _generate_improvement_ideas(self):
        """Generate ideas for system improvement"""
        prompt = "What improvements could be made to my systems based on recent performance and interactions?"
        
        try:
            result = await Runner.run(
                self.reflection_agent,
                prompt
            )
            
            ideas = result.final_output
            
            # Log ideas to issue tracker
            if hasattr(self, 'issue_tracker') and self.issue_tracker:
                await self.issue_tracker.process_observation(
                    f"Self-generated improvement ideas: {ideas}",
                    context=f"Generated during cognitive cycle {self.context.cognitive_cycle_count}"
                )
        except Exception as e:
            logger.error(f"Error generating improvement ideas: {str(e)}")
    
    # Function tools for the agents - refactored as factory methods
    
    def _create_collect_performance_metrics_tool(self):
        """Create the collect performance metrics tool with proper access to self"""
        @function_tool
        async def _collect_performance_metrics(ctx: RunContextWrapper) -> Dict[str, Dict[str, Any]]:
            """
            Collect performance metrics from all cognitive systems
            
            Returns:
                Dictionary of metrics for each system
            """
            performance_data = {}
            
            # Collect from registered processes
            for process_id, process in ctx.context.cognitive_processes.items():
                metrics = getattr(process, "performance_metrics", {}).copy()
                
                # Add derived metrics
                if hasattr(process, "total_runtime"):
                    metrics["runtime"] = process.total_runtime
                    
                # Get process type
                process_type = getattr(process, "type", "unknown")
                
                # Store in performance data
                if hasattr(process, "to_dict"):
                    process_dict = process.to_dict()
                else:
                    # Create simplified dict if to_dict not available
                    process_dict = {
                        "id": process_id,
                        "name": getattr(process, "name", "Unknown"),
                        "type": process_type,
                        "status": getattr(process, "status", "unknown")
                    }
                    
                performance_data[process_id] = {
                    "process": process_dict,
                    "metrics": metrics
                }
            
            # Also collect from system references
            for system_name, system in ctx.context.system_references.items():
                try:
                    # Get metrics using the most appropriate method
                    system_metrics = None
                    
                    if hasattr(system, "get_performance_metrics"):
                        system_metrics = await system.get_performance_metrics()
                    elif hasattr(system, "get_metrics"):
                        system_metrics = await system.get_metrics()
                    elif hasattr(system, "get_stats"):
                        system_metrics = await system.get_stats()
                        
                    if system_metrics:
                        sys_id = f"system_{system_name}"
                        
                        # Store simplified representation
                        performance_data[sys_id] = {
                            "process": {
                                "id": sys_id,
                                "name": system_name,
                                "type": system_name,
                                "status": "active"
                            },
                            "metrics": system_metrics
                        }
                except Exception as e:
                    logger.error(f"Error collecting metrics from {system_name}: {str(e)}")
            
            # Update performance history
            self._update_performance_history(performance_data)
            
            return performance_data
        
        return _collect_performance_metrics
    
    def _create_check_attention_focus_tool(self):             # noqa: N802
        """Return a strict check-attention-focus tool."""
    
        @function_tool
        async def _check_attention_focus(                     # noqa: N802
            ctx: RunContextWrapper,
            params: AttentionContextParams,                   # ← strict input
        ) -> AttentionCheckResult:                            # ← strict output
            import json, datetime
            from nyx.telemetry import custom_span
    
            with custom_span("check_attention_focus"):
                # ① decode context safely
                try:
                    context: Dict[str, Any] = json.loads(params.context_json or "{}")
                    if not isinstance(context, dict):
                        raise TypeError
                except Exception:
                    # Bad JSON ⇒ do nothing
                    return AttentionCheckResult(focus_changed=False)
    
                # ② ask owner object (self) to evaluate priority
                priority = self._determine_attention_priority(context)
    
                # local helpers --------------------------------------------------
                async def _set_focus(pri: Dict[str, Any]):
                    await self._set_attention_focus(pri)
                    ctx.context.attention_focus = pri
    
                async def _clear_focus():
                    await self._clear_attention_focus()
                    ctx.context.attention_focus = None
    
                # ③ decide & act -------------------------------------------------
                if priority:
                    cur = ctx.context.attention_focus
                    if cur is None:
                        await _set_focus(priority)
                        return AttentionCheckResult(
                            focus_changed=True,
                            action="set_focus",
                            details_json=json.dumps(priority, separators=(",", ":")),
                        )
                    if priority["priority"] > cur.get("priority", 0):
                        details = {
                            "from": cur["target"],
                            "to": priority["target"],
                            "priority": priority["priority"],
                            "reason": priority["reason"],
                        }
                        await _set_focus(priority)
                        return AttentionCheckResult(
                            focus_changed=True,
                            action="shift_focus",
                            details_json=json.dumps(details, separators=(",", ":")),
                        )
    
                # ④ maybe clear expired focus -----------------------------------
                cur = ctx.context.attention_focus
                if cur:
                    expiration = cur.get("expiration", 0)
                    if ctx.context.cognitive_cycle_count >= expiration:
                        details = {"from": cur["target"], "reason": "Focus expired"}
                        await _clear_focus()
                        return AttentionCheckResult(
                            focus_changed=True,
                            action="clear_focus",
                            details_json=json.dumps(details, separators=(",", ":")),
                        )
    
                # ⑤ nothing changed ---------------------------------------------
                return AttentionCheckResult(focus_changed=False)
    
        return _check_attention_focus
    
    def _create_detect_performance_drop_tool(self):
        """Create the detect performance drop tool with proper access to self"""
        @function_tool
        async def _detect_performance_drop(ctx: RunContextWrapper) -> bool:
            """
            Detect if there's been a significant drop in performance
            
            Returns:
                True if performance drop detected, False otherwise
            """
            for system_name, data in ctx.context.performance_history.items():
                if not isinstance(data, dict) or "history" not in data or len(data["history"]) < 3:
                    continue
                    
                # Get recent performance metrics
                recent = data["history"][-3:]
                
                # Check for performance drops in key metrics
                key_metrics = ['success_rate', 'accuracy', 'effectiveness', 'response_quality']
                for metric in key_metrics:
                    values = []
                    for entry in recent:
                        metrics = entry.get("metrics", {})
                        if metric in metrics:
                            values.append(metrics[metric])
                    
                    values = [v for v in values if v is not None]
                    
                    if len(values) >= 2 and values[0] > 0:
                        # Calculate percentage drop
                        drop_percent = (values[0] - values[-1]) / values[0]
                        if drop_percent > 0.2:  # 20% drop threshold
                            logger.info(f"Performance drop detected in {system_name}.{metric}: {drop_percent:.2f}")
                            return True
            
            return False
        
        return _detect_performance_drop
    
    def _create_identify_bottlenecks_tool(self):              # noqa: N802
        """Return a strict identify-bottlenecks tool."""
    
        @function_tool
        async def _identify_bottlenecks(                      # noqa: N802
            ctx: RunContextWrapper,
            params: PerformanceDataParams,                    # ← strict input
        ) -> BottleneckAnalysisResult:                        # ← strict output
            import json
    
            try:
                perf: Dict[str, Dict[str, Any]] = json.loads(params.performance_json)
                if not isinstance(perf, dict):
                    raise TypeError
            except Exception:
                # Bad JSON ⇒ return empty result
                return BottleneckAnalysisResult(bottlenecks=[], total_count=0)
    
            bns: list[BottleneckInfo] = []
    
            for p_id, data in perf.items():
                process: Dict[str, Any] = data.get("process", {})
                metrics: Dict[str, Any] = data.get("metrics", {})
    
                name = process.get("name", "Unknown")
                ptype = process.get("type", "unknown")
    
                def _append(kind: str, severity: float, descr: str, extra_metrics: Dict[str, Any]):
                    bns.append(
                        BottleneckInfo(
                            process_id=p_id,
                            process_name=name,
                            process_type=ptype,
                            kind=kind,
                            severity=severity,
                            description=descr,
                            metrics_json=json.dumps(extra_metrics, separators=(",", ":")),
                        )
                    )
    
                # 1. high resource utilisation
                if metrics.get("resource_utilization", 0) > 0.9:
                    _append(
                        "resource_utilization",
                        0.8,
                        f"Process {name} is using >90 % resources",
                        {"resource_utilization": metrics["resource_utilization"]},
                    )
    
                # 2. low efficiency
                if metrics.get("efficiency", 1) < 0.3:
                    _append(
                        "low_efficiency",
                        0.7,
                        f"Process {name} has efficiency <30 %",
                        {"efficiency": metrics["efficiency"]},
                    )
    
                # 3. high error rate
                if metrics.get("error_rate", 0) > 0.3:
                    _append(
                        "high_error_rate",
                        0.8,
                        f"Process {name} has error-rate >30 %",
                        {"error_rate": metrics["error_rate"]},
                    )
    
                # 4. slow response
                if metrics.get("response_time", 0) > 2.0:
                    _append(
                        "slow_response",
                        0.6,
                        f"Process {name} response time >2 s",
                        {"response_time": metrics["response_time"]},
                    )
    
                # 5. process-specific bottlenecks
                for pb in process.get("bottlenecks", []) or []:
                    _append(
                        "process_bottleneck",
                        float(pb.get("severity", 0.5)),
                        pb.get("description", "Unspecified bottleneck"),
                        {},
                    )
                    bns[-1].resource_type = pb.get("resource_type")
    
            # Sort by severity descending
            bns.sort(key=lambda b: b.severity, reverse=True)
    
            return BottleneckAnalysisResult(bottlenecks=bns, total_count=len(bns))
    
        return _identify_bottlenecks

    
    def _create_analyze_cognitive_strategies_tool(self):
        """Create the analyze cognitive strategies tool with proper access to self"""
        @function_tool
        async def _analyze_cognitive_strategies(ctx: RunContextWrapper) -> Dict[str, Any]:
            """
            Analyze effectiveness of current cognitive strategies
            
            Returns:
                Analysis of strategy effectiveness
            """
            analysis = {
                "overall_effectiveness": 0.0,
                "system_evaluations": {},
                "adaptation_rate": 0.0,
                "learning_effectiveness": 0.0,
                "recommended_changes": []
            }
            
            # Group processes by type
            processes_by_type = {}
            for process_id, process in self.context.cognitive_processes.items():
                process_type = getattr(process, "type", "unknown")
                if process_type not in processes_by_type:
                    processes_by_type[process_type] = []
                processes_by_type[process_type].append(process)
            
            # Evaluate effectiveness by system type
            total_score = 0.0
            evaluated_types = 0
            
            for system_name in self.context.performance_history:
                # Skip if not a dictionary with history
                if not isinstance(self.context.performance_history[system_name], dict) or "history" not in self.context.performance_history[system_name]:
                    continue
                    
                history = self.context.performance_history[system_name]["history"]
                
                if not history:
                    continue
                    
                # Get most recent metrics
                recent_entries = history[-3:] if len(history) >= 3 else history
                avg_metrics = {}
                
                # Calculate average of recent metrics
                for metric in ["efficiency", "success_rate", "accuracy", "response_time", "error_rate"]:
                    values = []
                    for entry in recent_entries:
                        if isinstance(entry, dict) and "metrics" in entry and metric in entry["metrics"]:
                            values.append(entry["metrics"][metric])
                    
                    if values:
                        avg_metrics[metric] = sum(values) / len(values)
                
                # Calculate effectiveness score
                effectiveness_score = 0.5  # Default score
                
                if "efficiency" in avg_metrics:
                    effectiveness_score = avg_metrics["efficiency"]
                elif "success_rate" in avg_metrics:
                    effectiveness_score = avg_metrics["success_rate"]
                elif "accuracy" in avg_metrics and "response_time" in avg_metrics:
                    # Balance accuracy and speed
                    norm_time = min(1.0, 1.0 / (1.0 + avg_metrics["response_time"]))
                    effectiveness_score = 0.7 * avg_metrics["accuracy"] + 0.3 * norm_time
                
                # Add to analysis
                analysis["system_evaluations"][system_name] = {
                    "effectiveness": effectiveness_score,
                    "average_metrics": avg_metrics,
                    "process_count": len(processes_by_type.get(system_name, []))
                }
                
                # Add to overall score
                total_score += effectiveness_score
                evaluated_types += 1
                
                # Generate recommendations
                if effectiveness_score < 0.4:
                    analysis["recommended_changes"].append({
                        "system": system_name,
                        "current_effectiveness": effectiveness_score,
                        "recommendation": f"Improve {system_name} strategy - consider new algorithms or increasing resources"
                    })
            
            # Calculate overall effectiveness
            if evaluated_types > 0:
                analysis["overall_effectiveness"] = total_score / evaluated_types
            
            # Calculate adaptation and learning metrics
            for system_name, system_data in self.context.performance_history.items():
                if not isinstance(system_data, dict) or "history" not in system_data:
                    continue
                    
                history = system_data["history"]
                
                if len(history) >= 5:
                    # Check how metrics have improved over time
                    improvement_rates = []
                    
                    if system_name in analysis["system_evaluations"]:
                        current_score = analysis["system_evaluations"][system_name]["effectiveness"]
                        
                        if len(history) >= 5:
                            # Get score from 5 cycles ago
                            old_entry = history[-5]
                            if isinstance(old_entry, dict) and "metrics" in old_entry:
                                old_metrics = old_entry["metrics"]
                                
                                old_effectiveness = None
                                if "efficiency" in old_metrics:
                                    old_effectiveness = old_metrics["efficiency"]
                                elif "success_rate" in old_metrics:
                                    old_effectiveness = old_metrics["success_rate"]
                                    
                                if old_effectiveness and old_effectiveness > 0:
                                    improvement = (current_score - old_effectiveness) / old_effectiveness
                                    improvement_rates.append(improvement)
                    
                    if improvement_rates:
                        avg_improvement = sum(improvement_rates) / len(improvement_rates)
                        analysis["adaptation_rate"] = max(0.0, min(1.0, avg_improvement + 0.5))  # Normalize to 0-1
                        analysis["learning_effectiveness"] = analysis["adaptation_rate"]
            
            return analysis
        
        return _analyze_cognitive_strategies
    
    def _create_reallocate_resources_tool(self):                 # noqa: N802
        """Return a strict re-allocate-resources tool."""
    
        @function_tool
        async def _reallocate_resources(                          # noqa: N802
            ctx: RunContextWrapper,
            params: ReallocateResourcesParams,                    # ← strict input
        ) -> ReallocateResourcesResult:                           # ← strict output
            import json, copy
    
            # ── decode input JSON ------------------------------------------------
            try:
                bottlenecks: List[Dict[str, Any]] = json.loads(params.bottlenecks_json)
                strategy_analysis: Dict[str, Any] = json.loads(params.strategy_analysis_json)
            except Exception:
                # On bad JSON return empty-change result
                return ReallocateResourcesResult(changes=[], new_allocations_json="{}")
    
            # ── current state ----------------------------------------------------
            new_alloc = copy.deepcopy(ctx.context.resource_allocation)
            meta      = ctx.context.meta_parameters
            sev_thres = meta.get("bottleneck_severity_threshold", 0.7)
            flexibility = meta.get("resource_flexibility", 0.2)
    
            # ── 1. handle critical bottlenecks ----------------------------------
            crit = [b for b in bottlenecks if b.get("severity", 0) >= sev_thres]
            for b in crit:
                ptype = b.get("process_type")
                if ptype in new_alloc and b.get("type") in {
                    "resource_utilization", "slow_response", "process_bottleneck"
                }:
                    new_alloc[ptype] = min(0.4, new_alloc[ptype] * 1.5)
    
            # ── 2. strategy effectiveness adjustments ---------------------------
            for sys_name, ev in strategy_analysis.get("system_evaluations", {}).items():
                if sys_name not in new_alloc:
                    continue
                eff = ev.get("effectiveness", 0.5)
                if eff < 0.3:
                    new_alloc[sys_name] = min(0.4, new_alloc[sys_name] * 1.3)
                elif eff > 0.8:
                    new_alloc[sys_name] = max(0.1, new_alloc[sys_name] * 0.9)
    
            # ── 3. normalise to 1.0 ---------------------------------------------
            total = sum(new_alloc.values()) or 1.0
            for k in new_alloc:
                new_alloc[k] /= total
    
            # ── 4. compute and apply bounded changes ----------------------------
            changes: list[ResourceDelta] = []
            for sys_name, target in new_alloc.items():
                current = ctx.context.resource_allocation.get(sys_name, 0.0)
                delta   = target - current
                if abs(delta) < 0.02:          # 2 % threshold
                    continue
    
                max_change = abs(current) * flexibility
                bounded    = max(-max_change, min(delta, max_change))
                ctx.context.resource_allocation[sys_name] = current + bounded
    
                changes.append(ResourceDelta(system=sys_name, delta=bounded))
    
            # ── 5. final normalisation (guard) -----------------------------------
            tot_final = sum(ctx.context.resource_allocation.values()) or 1.0
            for k in ctx.context.resource_allocation:
                ctx.context.resource_allocation[k] /= tot_final
    
            return ReallocateResourcesResult(
                changes=changes,
                new_allocations_json=json.dumps(ctx.context.resource_allocation, separators=(",", ":")),
            )
    
        return _reallocate_resources
    
    def _create_identify_inefficient_dependencies_tool(self):
        """Create the identify inefficient dependencies tool with proper access to self"""
        @function_tool
        async def _identify_inefficient_dependencies(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
            """
            Identify inefficient dependencies between processes
            
            Returns:
                List of inefficient dependencies
            """
            inefficient_dependencies = []
            
            # Build dependency graph
            dependency_graph = {}
            for process_id, process in self.context.cognitive_processes.items():
                for dependency in getattr(process, "dependencies", []):
                    dep_id = dependency.get("process_id")
                    importance = dependency.get("importance", 0.5)
                    
                    if process_id not in dependency_graph:
                        dependency_graph[process_id] = []
                    
                    dependency_graph[process_id].append({
                        "target_id": dep_id,
                        "importance": importance
                    })
            
            # Check for inefficiencies
            for source_id, dependencies in dependency_graph.items():
                source_process = self.context.cognitive_processes.get(source_id)
                if not source_process:
                    continue
                    
                source_name = getattr(source_process, "name", "Unknown Process")
                source_type = getattr(source_process, "type", "unknown")
                
                for dep in dependencies:
                    target_id = dep["target_id"]
                    importance = dep["importance"]
                    
                    target_process = self.context.cognitive_processes.get(target_id)
                    if not target_process:
                        continue
                    
                    target_name = getattr(target_process, "name", "Unknown Process")
                    target_type = getattr(target_process, "type", "unknown")
                    
                    # Check for performance issues
                    inefficiency_score = 0.0
                    reasons = []
                    
                    # Check for status issues (blocking)
                    source_status = getattr(source_process, "status", "unknown")
                    target_status = getattr(target_process, "status", "unknown")
                    
                    if source_status == "blocked" and target_status != "completed":
                        inefficiency_score += 0.4
                        reasons.append("Frequent blocking")
                    
                    # Check for high latency in dependent process
                    target_metrics = getattr(target_process, "performance_metrics", {})
                    if target_metrics.get("response_time", 0) > 2.0:  # High response time
                        inefficiency_score += 0.3
                        reasons.append("High dependency latency")
                    
                    # Check for high importance but low performance
                    if importance > 0.7:
                        target_efficiency = target_metrics.get("efficiency", 0.5)
                        if target_efficiency < 0.4:  # Low efficiency
                            inefficiency_score += importance * 0.4
                            reasons.append("High importance but low performance")
                    
                    # Add if significant inefficiency found
                    if inefficiency_score > 0.3:
                        recommendation = ""
                        
                        if "Frequent blocking" in reasons:
                            recommendation = "Consider making dependency asynchronous or preemptive"
                        elif "High dependency latency" in reasons:
                            recommendation = "Optimize dependent process or use caching"
                        else:
                            recommendation = "Reconsider dependency structure or improve dependent process"
                        
                        inefficient_dependencies.append({
                            "source_id": source_id,
                            "source_name": source_name,
                            "source_type": source_type,
                            "target_id": target_id,
                            "target_name": target_name,
                            "target_type": target_type,
                            "importance": importance,
                            "inefficiency_score": inefficiency_score,
                            "reasons": reasons,
                            "impact": inefficiency_score * importance,
                            "recommendation": recommendation
                        })
            
            # Sort by impact
            inefficient_dependencies.sort(key=lambda x: x["impact"], reverse=True)
            
            return inefficient_dependencies
        
        return _identify_inefficient_dependencies
    
    def _create_select_strategy_tool(self):                       # noqa: N802
        """Return a strict select-strategy tool."""
    
        @function_tool
        async def _select_strategy(                               # noqa: N802
            ctx: RunContextWrapper,
            params: SelectStrategyParams,                         # ← strict input
        ) -> StrategyResult:                                      # ← strict output
            import json
    
            # ── decode JSON inputs ------------------------------------------------
            try:
                context: Dict[str, Any]     = json.loads(params.context_json)
                performance: Dict[str, Any] = json.loads(params.performance_json)
            except Exception:                                       # bad JSON
                # fall back to safest strategy
                return StrategyResult(
                    name="Balanced Approach",
                    description="Fallback balanced strategy",
                    parameters=StrategyParameters(
                        exploration_rate=0.2,
                        adaptation_rate=0.15,
                        risk_tolerance=0.5,
                        innovation_level=0.5,
                        precision_focus=0.5,
                    ),
                    expected_impact={"performance": 0.0},
                    confidence=0.0,
                )
    
            # ── context / environment metrics ------------------------------------
            ctx_features = self._extract_context_features(context)
            complexity   = self._calculate_context_complexity(context)
            volatility   = self._calculate_context_volatility()
    
            # ── candidate strategy catalogue -------------------------------------
            catalogue: Dict[str, Dict[str, Any]] = {
                "balanced": {
                    "name": "Balanced Approach",
                    "parameters": {
                        "exploration_rate": 0.2,
                        "adaptation_rate": 0.15,
                        "risk_tolerance": 0.5,
                        "innovation_level": 0.5,
                        "precision_focus": 0.5,
                    },
                    "description": "Moderate exploration and adaptation.",
                },
                "exploratory": {
                    "name": "Exploratory Strategy",
                    "parameters": {
                        "exploration_rate": 0.4,
                        "adaptation_rate": 0.2,
                        "risk_tolerance": 0.7,
                        "innovation_level": 0.8,
                        "precision_focus": 0.3,
                    },
                    "description": "High exploration to discover new patterns.",
                },
                "conservative": {
                    "name": "Conservative Strategy",
                    "parameters": {
                        "exploration_rate": 0.1,
                        "adaptation_rate": 0.1,
                        "risk_tolerance": 0.2,
                        "innovation_level": 0.3,
                        "precision_focus": 0.8,
                    },
                    "description": "Low risk with strong precision focus.",
                },
                "adaptive": {
                    "name": "Highly Adaptive Strategy",
                    "parameters": {
                        "exploration_rate": 0.3,
                        "adaptation_rate": 0.3,
                        "risk_tolerance": 0.6,
                        "innovation_level": 0.6,
                        "precision_focus": 0.4,
                    },
                    "description": "Prioritises quick adaptation to change.",
                },
            }
    
            # ── score each strategy ---------------------------------------------
            scores: Dict[str, float] = {}
            for sid, strat in catalogue.items():
                scores[sid] = self._calculate_strategy_score(
                    strat, ctx_features, performance, complexity, volatility
                )
    
            # ── best strategy & confidence ---------------------------------------
            best_id = max(scores.keys(), key=scores.get, default="balanced")
            best    = catalogue[best_id]
    
            if len(scores) > 1:
                top, second = sorted(scores.values(), reverse=True)[:2]
                confidence  = 0.5 + 0.5 * ((top - second) / max(top, 1e-6))
            else:
                confidence = 0.5
    
            return StrategyResult(
                name=best["name"],
                description=best["description"],
                parameters=StrategyParameters(**best["parameters"]),
                expected_impact={"performance": 0.2, "adaptability": 0.3},  # TODO refine
                confidence=round(min(max(confidence, 0.0), 1.0), 3),
            )
    
        return _select_strategy
    
    def _create_calculate_resource_trend_tool(self):
        """Create the calculate resource trend tool with proper access to self"""
        @function_tool
        async def _calculate_resource_trend(ctx: RunContextWrapper, values: List[float]) -> Dict[str, Any]:
            """
            Calculate trend from a series of resource values
            
            Args:
                values: Series of data points
                
            Returns:
                Trend analysis
            """
            if len(values) < 2:
                return {"direction": "stable", "magnitude": 0.0}
                
            # Calculate linear regression
            n = len(values)
            x = list(range(n))
            mean_x = sum(x) / n
            mean_y = sum(values) / n
            
            numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
            
            if denominator == 0:
                return {"direction": "stable", "magnitude": 0.0}
                
            slope = numerator / denominator
            
            # Normalize slope based on mean value
            if mean_y != 0:
                normalized_slope = slope / abs(mean_y)
            else:
                normalized_slope = slope
                
            # Determine direction and magnitude
            if abs(normalized_slope) < 0.05:
                direction = "stable"
            elif normalized_slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
                
            return {
                "direction": direction,
                "magnitude": abs(normalized_slope),
                "slope": slope,
                "mean": mean_y
            }
        
        return _calculate_resource_trend
    
    def _create_generate_meta_cognitive_insights_tool(self):
        """Create the generate meta cognitive insights tool with proper access to self"""
        @function_tool
        async def _generate_meta_cognitive_insights(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
            """
            Generate insights about cognitive processes and patterns
            
            Returns:
                List of metacognitive insights
            """
            insights = []
            
            # Check for systems with critical performance
            critical_systems = []
            for system_name, data in self.context.performance_history.items():
                if not isinstance(data, dict) or "history" not in data or not data["history"]:
                    continue
                    
                latest = data["history"][-1].get("metrics", {})
                
                # Check for critically low performance
                if latest.get("success_rate", 1.0) < 0.2:  # Extremely low success
                    critical_systems.append(system_name)
            
            if critical_systems:
                insights.append({
                    "type": "weakness",
                    "description": f"Systems showing critical performance issues: {', '.join(critical_systems)}",
                    "confidence": 0.9,
                    "priority": "high"
                })
            
            # Check for resource allocation effectiveness
            for system, allocation in self.context.resource_allocation.items():
                system_data = self.context.performance_history.get(system, {})
                if not isinstance(system_data, dict) or "history" not in system_data or not system_data["history"]:
                    continue
                    
                latest = system_data["history"][-1].get("metrics", {})
                success_rate = latest.get("success_rate", 0.5)
                
                if success_rate > 0.8 and allocation < 0.15:
                    insights.append({
                        "type": "efficiency",
                        "system": system,
                        "description": f"{system} system performing excellently with minimal resources ({allocation:.2f})",
                        "confidence": 0.7
                    })
                elif success_rate < 0.3 and allocation > 0.25:
                    insights.append({
                        "type": "inefficiency",
                        "system": system,
                        "description": f"{system} system performing poorly despite high resource allocation ({allocation:.2f})",
                        "confidence": 0.7,
                        "priority": "medium"
                    })
            
            return insights
        
        return _generate_meta_cognitive_insights
    
    def _create_analyze_recent_performance_tool(self):
        """Create the analyze recent performance tool with proper access to self"""
        @function_tool
        async def _analyze_recent_performance(ctx: RunContextWrapper) -> Dict[str, Any]:
            """
            Analyze recent performance across all systems
            
            Returns:
                Performance analysis by system
            """
            analysis = {}
            
            for system_name, data in self.context.performance_history.items():
                if not isinstance(data, dict) or "history" not in data or not data["history"]:
                    analysis[system_name] = {"status": "insufficient_data"}
                    continue
                    
                history = data["history"]
                
                # Calculate trends for key metrics
                system_analysis = {"trends": {}}
                for metric in ['success_rate', 'accuracy', 'effectiveness', 'error_rate', 'response_time']:
                    values = []
                    for entry in history:
                        metrics = entry.get("metrics", {})
                        if metric in metrics:
                            values.append(metrics[metric])
                    
                    if len(values) >= 3:
                        # Use the tool to calculate trend
                        trend_tool = self._create_calculate_resource_trend_tool()
                        trend = await trend_tool(ctx, values)
                        system_analysis["trends"][metric] = trend
                
                # Determine overall status
                positive_trends = sum(1 for t in system_analysis["trends"].values() 
                                   if t.get("direction") == "improving")
                negative_trends = sum(1 for t in system_analysis["trends"].values() 
                                   if t.get("direction") == "declining")
                
                if positive_trends > negative_trends * 2:
                    status = "excellent"
                elif positive_trends > negative_trends:
                    status = "good"
                elif positive_trends == negative_trends:
                    status = "stable"
                elif negative_trends > positive_trends * 2:
                    status = "critical"
                else:
                    status = "concerning"
                    
                system_analysis["status"] = status
                
                # Get latest performance
                if history:
                    latest = history[-1].get("metrics", {})
                    system_analysis["current_metrics"] = latest
                
                analysis[system_name] = system_analysis
            
            return analysis
        
        return _analyze_recent_performance
    
    def _create_generate_cognitive_insights_tool(self):
        """Return strict generate-insights tool"""
    
        @function_tool
        async def _generate_cognitive_insights(          # noqa: N802
            ctx: RunContextWrapper,
            params: GenerateCognitiveInsightsParams,
        ) -> CognitiveInsightsResult:
            performance_analysis = json.loads(params.performance_json)
            insights: List[Dict[str, Any]] = []
    
            # --- original logic --------------------------------------------------
            excellent = [
                sys for sys, data in performance_analysis.items()
                if data.get("status") == "excellent"
            ]
            if excellent:
                insights.append(
                    {
                        "type": "strength",
                        "description": (
                            f"Systems showing excellent performance: {', '.join(excellent)}"
                        ),
                        "confidence": 0.9,
                    }
                )
    
            critical = [
                sys for sys, data in performance_analysis.items()
                if data.get("status") == "critical"
            ]
            if critical:
                insights.append(
                    {
                        "type": "weakness",
                        "description": (
                            f"Systems showing critical performance issues: {', '.join(critical)}"
                        ),
                        "confidence": 0.9,
                        "priority": "high",
                    }
                )
                # details per critical system
                for system in critical:
                    trends   = performance_analysis[system].get("trends", {})
                    metrics  = performance_analysis[system].get("current_metrics", {})
                    if metrics.get("error_rate", 0) > 0.3:
                        insights.append(
                            {
                                "type": "weakness",
                                "system": system,
                                "description": (
                                    f"High error rate in {system} system: "
                                    f"{metrics['error_rate']:.2f}"
                                ),
                                "confidence": 0.9,
                                "priority": "high",
                            }
                        )
                    if metrics.get("response_time", 0) > 2.0:
                        insights.append(
                            {
                                "type": "weakness",
                                "system": system,
                                "description": (
                                    f"Slow response time in {system} system: "
                                    f"{metrics['response_time']:.2f}s"
                                ),
                                "confidence": 0.9,
                                "priority": "high",
                            }
                        )
                    if metrics.get("success_rate", 1.0) < 0.5:
                        insights.append(
                            {
                                "type": "weakness",
                                "system": system,
                                "description": (
                                    f"Low success rate in {system} system: "
                                    f"{metrics['success_rate']:.2f}"
                                ),
                                "confidence": 0.9,
                                "priority": "high",
                            }
                        )
    
            all_statuses = [
                data.get("status")
                for data in performance_analysis.values()
                if "status" in data and data["status"] != "insufficient_data"
            ]
            if all_statuses and all(s in ("excellent", "good") for s in all_statuses):
                insights.append(
                    {
                        "type": "synergy",
                        "description": (
                            "All systems are performing well, indicating good synergy"
                        ),
                        "confidence": 0.8,
                    }
                )
            # ---------------------------------------------------------------------
    
            return CognitiveInsightsResult(
                insights=[_Insight(**i) for i in insights],
            )
    
        return _generate_cognitive_insights
    
    def _create_identify_improvement_areas_tool(self):
        """Return strict identify-improvement-areas tool"""
    
        @function_tool
        async def _identify_improvement_areas(          # noqa: N802
            ctx: RunContextWrapper,
            params: IdentifyImprovementAreasParams,
        ) -> ImprovementAreasResult:
            performance_analysis = json.loads(params.performance_json)
            insights             = json.loads(params.insights_json)
    
            improvement_areas: List[Dict[str, Any]] = []
    
            # --- original logic --------------------------------------------------
            critical_systems = [
                sys for sys, data in performance_analysis.items()
                if data.get("status") in ("critical", "concerning")
            ]
            for system_name in critical_systems:
                analysis = performance_analysis[system_name]
                problematic = [
                    metric
                    for metric, trend in analysis.get("trends", {}).items()
                    if trend.get("direction") == "declining" and trend.get("magnitude", 0) > 0.1
                ]
                improvement_areas.append(
                    {
                        "system": system_name,
                        "priority": 1 if analysis.get("status") == "critical" else 2,
                        "metrics_to_improve": problematic,
                        "current_metrics": analysis.get("current_metrics", {}),
                        "current_status": analysis.get("status"),
                    }
                )
    
            resource_insights = [
                i for i in insights if i.get("type") in ("efficiency", "inefficiency")
            ]
            if resource_insights:
                improvement_areas.append(
                    {
                        "system": "resource_allocation",
                        "priority": 3,
                        "description": "Resource allocation needs optimization",
                        "details": [i["description"] for i in resource_insights],
                        "current_status": "inefficient",
                        "metrics_to_improve": [],
                        "current_metrics": {},
                    }
                )
    
            improvement_areas.sort(key=lambda x: x["priority"])
            # ---------------------------------------------------------------------
    
            return ImprovementAreasResult(
                improvement_areas=[_ImprovementArea(**ia) for ia in improvement_areas]
            )
    
        return _identify_improvement_areas

    
    def _create_create_improvement_plan_tool(self):
        """Return strict create-improvement-plan tool"""
    
        @function_tool
        async def _create_improvement_plan(             # noqa: N802
            ctx: RunContextWrapper,
            params: CreateImprovementPlanParams,
        ) -> ImprovementPlanResult:
            improvement_areas = json.loads(params.improvement_areas_json)
            strategies        = json.loads(params.strategies_json)
    
            # --- original logic (unchanged) -------------------------------------
            plan = {
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": self.context.cognitive_cycle_count,
                "priority_areas": [
                    area["system"]
                    for area in improvement_areas
                    if area.get("priority", 3) == 1
                ],
                "phases": [
                    {
                        "name": "Critical Improvements",
                        "duration": 5,
                        "targets": [
                            area["system"]
                            for area in improvement_areas
                            if area.get("priority", 3) == 1
                        ],
                        "strategies": [
                            s for s in strategies
                            if s.get("system") in [
                                area["system"]
                                for area in improvement_areas
                                if area.get("priority", 3) == 1
                            ]
                        ],
                    },
                    {
                        "name": "Secondary Enhancements",
                        "duration": 10,
                        "targets": [
                            area["system"]
                            for area in improvement_areas
                            if area.get("priority", 3) == 2
                        ],
                        "strategies": [
                            s for s in strategies
                            if s.get("system") in [
                                area["system"]
                                for area in improvement_areas
                                if area.get("priority", 3) == 2
                            ]
                        ],
                    },
                    {
                        "name": "Optimization",
                        "duration": 15,
                        "targets": [
                            area["system"]
                            for area in improvement_areas
                            if area.get("priority", 3) == 3
                        ],
                        "strategies": [
                            s for s in strategies
                            if s.get("system") in [
                                area["system"]
                                for area in improvement_areas
                                if area.get("priority", 3) == 3
                            ]
                        ],
                    },
                ],
                "expected_outcomes": {
                    "performance_improvement": 0.3,
                    "bottleneck_reduction": 0.5,
                    "efficiency_gain": 0.2,
                },
                "status": "created",
            }
            # ---------------------------------------------------------------------
    
            return ImprovementPlanResult(plan_json=json.dumps(plan))
    
        return _create_improvement_plan
    
    def _create_update_system_parameters_tool(self):
        """Return strict update-system-parameters tool"""
    
        @function_tool
        async def _update_system_parameters(            # noqa: N802
            ctx: RunContextWrapper,
            params: UpdateSystemParametersParams,
        ) -> UpdateSystemParametersResult:
            bottlenecks       = json.loads(params.bottlenecks_json)
            strategy_analysis = json.loads(params.strategy_analysis_json)
            updates_made: Dict[str, Any] = {}
    
            # --- original logic --------------------------------------------------
            critical_bottlenecks = [b for b in bottlenecks if b["severity"] >= 0.7]
            for bottleneck in critical_bottlenecks:
                system_name = bottleneck["process_type"]
                if system_name in self.context.system_references:
                    system = self.context.system_references[system_name]
                    adjustments = {}
                    match bottleneck["type"]:
                        case "high_error_rate":
                            adjustments = {
                                "error_correction_level": "high",
                                "validation_threshold": 0.8,
                            }
                        case "slow_response":
                            adjustments = {
                                "caching_enabled": True,
                                "optimization_level": "aggressive",
                            }
                        case "resource_utilization":
                            adjustments = {
                                "resource_efficiency_mode": "enabled",
                                "batch_processing": True,
                            }
                    if adjustments and hasattr(system, "set_parameters"):
                        try:
                            await system.set_parameters(adjustments)
                            updates_made[system_name] = adjustments
                        except Exception as exc:
                            logger.error(
                                f"Error updating parameters for {system_name}: {exc}"
                            )
    
            if "recommended_changes" in strategy_analysis:
                for rec in strategy_analysis["recommended_changes"]:
                    system_name = rec.get("system")
                    if system_name in self.context.system_references:
                        system = self.context.system_references[system_name]
                        strat_adj = {
                            "strategy_improvement": True,
                            "effectiveness_target": rec.get("current_effectiveness", 0.5)
                            + 0.2,
                        }
                        if hasattr(system, "set_strategy"):
                            try:
                                await system.set_strategy(strat_adj)
                                updates_made[f"{system_name}_strategy"] = strat_adj
                            except Exception as exc:
                                logger.error(
                                    f"Error updating strategy for {system_name}: {exc}"
                                )
            # ---------------------------------------------------------------------
    
            return UpdateSystemParametersResult(
                updates_json=json.dumps(updates_made)
            )
    
        return _update_system_parameters
    
    def _create_improve_meta_parameters_tool(self):
        """Return strict improve-meta-parameters tool"""
    
        @function_tool
        async def _improve_meta_parameters(             # noqa: N802
            ctx: RunContextWrapper,
        ) -> MetaParametersUpdateResult:
            parameter_changes: Dict[str, float] = {}
            overall_effectiveness = self._calculate_overall_effectiveness()
    
            # --- original logic --------------------------------------------------
            if overall_effectiveness < 0.4:
                parameter_changes["exploration_rate"] = min(
                    0.8, self.context.meta_parameters["exploration_rate"] * 1.5
                )
                parameter_changes["resource_flexibility"] = min(
                    0.8, self.context.meta_parameters["resource_flexibility"] * 1.3
                )
                parameter_changes["evaluation_interval"] = max(
                    2, int(self.context.meta_parameters["evaluation_interval"] * 0.7)
                )
                parameter_changes["parameter_adjustment_factor"] = min(
                    0.5,
                    self.context.meta_parameters["parameter_adjustment_factor"] * 1.3,
                )
            elif overall_effectiveness > 0.8:
                parameter_changes["exploration_rate"] = max(
                    0.05, self.context.meta_parameters["exploration_rate"] * 0.8
                )
                parameter_changes["evaluation_interval"] = min(
                    10, int(self.context.meta_parameters["evaluation_interval"] * 1.2)
                )
    
            if self.context.cognitive_cycle_count > 50:
                parameter_changes["reflection_frequency"] = min(
                    20, int(self.context.meta_parameters["reflection_frequency"] * 1.2)
                )
    
            original_values = {
                k: self.context.meta_parameters[k] for k in parameter_changes.keys()
            }
            for k, v in parameter_changes.items():
                self.context.meta_parameters[k] = v
            # ---------------------------------------------------------------------
    
            return MetaParametersUpdateResult(
                original_values_json=json.dumps(original_values),
                updated_values_json=json.dumps(parameter_changes),
                cycle=self.context.cognitive_cycle_count,
            )
    
        return _improve_meta_parameters
    
    
    # ---------------------------------------------------------------------
    def _create_generate_cognitive_strategies_tool(self):
        """Return strict generate-strategies tool"""
    
        @function_tool
        async def _generate_cognitive_strategies(       # noqa: N802
            ctx: RunContextWrapper,
            params: GenerateCognitiveStrategiesParams,
        ) -> CognitiveStrategiesResult:
            improvement_areas = json.loads(params.improvement_areas_json)
            strategies: List[Dict[str, Any]] = []
    
            # --- original logic --------------------------------------------------
            for area in improvement_areas:
                system = area.get("system")
                if system not in self.context.system_references:
                    continue
    
                sys_ref = self.context.system_references[system]
                system_strats: List[Dict[str, Any]] = []
                if hasattr(sys_ref, "generate_improvement_strategies"):
                    try:
                        system_strats = await sys_ref.generate_improvement_strategies()
                    except Exception as exc:
                        logger.error(f"Error generating strategies from {system}: {exc}")
    
                if system_strats:
                    for s in system_strats:
                        strategies.append(
                            {
                                "name": s.get("name", f"Strategy for {system}"),
                                "system": system,
                                "description": s.get("description", ""),
                                "implementation": s.get("implementation", {}),
                                "expected_impact": s.get("expected_impact", {}),
                                "source": "system_generated",
                            }
                        )
                else:
                    generic = self._generate_generic_strategy(system, area)
                    if generic:
                        strategies.append(generic)
            # ---------------------------------------------------------------------
    
            return CognitiveStrategiesResult(
                strategies_json=json.dumps(strategies)
            )
    
        return _generate_cognitive_strategies
    
    # Helper functions
    
    def _update_performance_history(self, performance_data: Dict[str, Dict[str, Any]]) -> None:
        """Update performance history with new metrics"""
        # Group by system type
        grouped_metrics = {}
        
        for process_id, data in performance_data.items():
            process_type = data["process"].get("type", "unknown")
            if process_type not in grouped_metrics:
                grouped_metrics[process_type] = []
                
            grouped_metrics[process_type].append(data)
        
        # Update history for each system type
        for system_name in self.context.performance_history:
            if not isinstance(self.context.performance_history[system_name], dict):
                self.context.performance_history[system_name] = {"parameters": {}, "history": []}
                
            if system_name in grouped_metrics:
                # Aggregate metrics from all processes of this type
                aggregated_metrics = self._aggregate_process_metrics(grouped_metrics[system_name])
                
                # Add timestamped entry
                timestamped_metrics = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cycle": self.context.cognitive_cycle_count,
                    "metrics": aggregated_metrics
                }
                
                if "history" not in self.context.performance_history[system_name]:
                    self.context.performance_history[system_name]["history"] = []
                    
                self.context.performance_history[system_name]["history"].append(timestamped_metrics)
                
                # Keep history to a reasonable size
                if len(self.context.performance_history[system_name]["history"]) > 100:
                    self.context.performance_history[system_name]["history"] = self.context.performance_history[system_name]["history"][-100:]
    
    def _aggregate_process_metrics(self, process_data_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from multiple processes of the same type"""
        # Find all metrics keys
        all_metrics = set()
        for data in process_data_list:
            all_metrics.update(data["metrics"].keys())
        
        # Aggregate each metric
        aggregated = {}
        for metric in all_metrics:
            values = []
            for data in process_data_list:
                if metric in data["metrics"]:
                    value = data["metrics"][metric]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                # Use mean for most metrics
                aggregated[metric] = sum(values) / len(values)
        
        return aggregated
    
    def _update_system_metrics(self, cycle_start: float) -> None:
        """Update system metrics based on current state"""
        now = time.time()
        cycle_time = now - cycle_start
        
        # Update system metrics
        self.context.system_metrics["cycles_completed"] += 1
        
        # Update total runtime
        runtime = (datetime.datetime.now() - self.context.system_metrics["start_time"]).total_seconds()
        self.context.system_metrics["total_runtime"] = runtime
        
        # Update process count
        self.context.system_metrics["total_processes"] = len(self.context.cognitive_processes)
        
        # Update average cycle time with exponential moving average
        alpha = 0.2  # Weight for current cycle
        current_avg = self.context.system_metrics["average_cycle_time"]
        new_avg = (1 - alpha) * current_avg + alpha * cycle_time
        self.context.system_metrics["average_cycle_time"] = new_avg
        
        # Update resource usage metrics if available
        try:
            import psutil
            process = psutil.Process()
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.context.system_metrics["resource_usage"]["cpu"] = cpu_percent
            
            # Update memory usage
            memory_info = process.memory_info()
            self.context.system_metrics["resource_usage"]["memory"] = memory_info.rss / (1024 * 1024)  # MB
        except:
            # Resource usage metrics not available, use fallback values
            pass
        
        # Update error rate
        if hasattr(self, 'error_logs') and self.context.error_logs:
            total_errors = len(self.context.error_logs)
            if runtime > 0:
                self.context.system_metrics["error_rate"] = total_errors / runtime
    
    def _calculate_overall_effectiveness(self) -> float:
        """Calculate overall system effectiveness across all monitored systems"""
        total_score = 0.0
        systems_count = 0
        
        for system_name, data in self.context.performance_history.items():
            if not isinstance(data, dict) or "history" not in data or not data["history"]:
                continue
                
            # Get latest metrics
            latest = data["history"][-1].get("metrics", {})
            
            # Calculate effectiveness
            effectiveness = 0.5  # Default
            
            if "effectiveness" in latest:
                effectiveness = latest["effectiveness"]
            elif "success_rate" in latest:
                effectiveness = latest["success_rate"]
            elif "accuracy" in latest and "response_time" in latest:
                # Balance accuracy and response time
                norm_time = min(1.0, 1.0 / (1.0 + latest["response_time"]))
                effectiveness = 0.7 * latest["accuracy"] + 0.3 * norm_time
            elif "error_rate" in latest:
                # Lower error rate means higher effectiveness
                effectiveness = max(0.0, 1.0 - latest["error_rate"] * 2)
            
            total_score += effectiveness
            systems_count += 1
        
        if systems_count == 0:
            return 0.5  # Default when no data available
            
        return total_score / systems_count
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from context for strategy selection"""
        features = {}
        
        # Extract basic scalars
        for key, value in context.items():
            if isinstance(value, (int, float, bool)):
                if isinstance(value, bool):
                    features[key] = 1.0 if value else 0.0
                else:
                    features[key] = float(value)
        
        # Extract feature from user input if present
        if "user_input" in context and isinstance(context["user_input"], str):
            features["input_length"] = min(1.0, len(context["user_input"]) / 500.0)
            features["input_complexity"] = min(1.0, len(set(context["user_input"].split())) / 100.0)
        
        # Calculate volatility feature
        features["context_volatility"] = self._calculate_context_volatility()
        
        return features
    
    def _calculate_context_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate the complexity of the current context"""
        # Count the number of nested elements and total elements
        total_elements = 0
        nested_elements = 0
        max_depth = 0
        
        def count_elements(obj, depth=0):
            nonlocal total_elements, nested_elements, max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(obj, dict):
                total_elements += len(obj)
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        nested_elements += 1
                        count_elements(value, depth + 1)
            elif isinstance(obj, list):
                total_elements += len(obj)
                for item in obj:
                    if isinstance(item, (dict, list)):
                        nested_elements += 1
                        count_elements(item, depth + 1)
        
        count_elements(context)
        
        # Calculate complexity factors
        size_factor = min(1.0, total_elements / 50.0)  # Normalize by expecting max 50 elements
        nesting_factor = min(1.0, nested_elements / 10.0)  # Normalize by expecting max 10 nested elements
        depth_factor = min(1.0, max_depth / 5.0)  # Normalize by expecting max depth of 5
        
        # Combine factors with weights
        complexity = (
            size_factor * 0.4 +
            nesting_factor * 0.3 +
            depth_factor * 0.3
        )
        
        return complexity
    
    def _calculate_context_volatility(self) -> float:
        """Calculate context volatility based on performance history"""
        # Use performance variation as a proxy for context volatility
        variations = []
        
        for system_name, data in self.context.performance_history.items():
            if not isinstance(data, dict) or "history" not in data:
                continue
                
            history = data["history"]
            if len(history) < 3:
                continue
                
            # Calculate variations in performance metrics
            for metric in ["success_rate", "error_rate", "effectiveness"]:
                values = []
                for entry in history[-5:]:  # Last 5 entries
                    if metric in entry.get("metrics", {}):
                        values.append(entry["metrics"][metric])
                
                if len(values) >= 3:
                    # Calculate standard deviation
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    std_dev = math.sqrt(variance)
                    
                    # Normalize by mean if possible
                    if mean > 0:
                        variation = std_dev / mean
                    else:
                        variation = std_dev
                        
                    variations.append(variation)
        
        if not variations:
            return 0.2  # Default volatility
            
        # Average variation across all metrics
        avg_variation = sum(variations) / len(variations)
        
        # Normalize to [0,1]
        volatility = min(1.0, avg_variation * 3.0)  # Scale to make values more meaningful
        
        return volatility
    
    def _calculate_strategy_score(self,
                                strategy: Dict[str, Any], 
                                context_features: Dict[str, float],
                                performance: Dict[str, Any],
                                complexity: float,
                                volatility: float) -> float:
        """Calculate a score for how well a strategy matches the current context"""
        params = strategy["parameters"]
        
        # Base score starts at 0.5
        score = 0.5
        
        # Adjust based on complexity
        # Higher complexity prefers higher adaptation rate
        complexity_match = 1.0 - abs(complexity - params["adaptation_rate"])
        score += complexity_match * 0.1
        
        # Adjust based on volatility
        # Higher volatility prefers higher exploration rate
        volatility_match = 1.0 - abs(volatility - params["exploration_rate"])
        score += volatility_match * 0.1
        
        # Adjust based on performance trends
        if "system_evaluations" in performance:
            eval_count = 0
            eval_score = 0.0
            
            for system_name, eval_data in performance["system_evaluations"].items():
                effectiveness = eval_data.get("effectiveness", 0.5)
                
                # If effectiveness is low, prefer more exploratory strategies
                if effectiveness < 0.4:
                    eval_score += params["exploration_rate"] * 0.4
                    eval_score += params["adaptation_rate"] * 0.3
                    eval_score += (1.0 - params["precision_focus"]) * 0.3
                # If effectiveness is high, prefer more precision-focused strategies
                elif effectiveness > 0.7:
                    eval_score += params["precision_focus"] * 0.5
                    eval_score += (1.0 - params["risk_tolerance"]) * 0.3
                    eval_score += (1.0 - params["exploration_rate"]) * 0.2
                # For moderate effectiveness, balanced approach
                else:
                    balance_factor = 1.0 - abs(0.5 - params["precision_focus"])
                    eval_score += balance_factor * 0.5
                
                eval_count += 1
            
            if eval_count > 0:
                score += (eval_score / eval_count) * 0.2
        
        # Add random variation (exploration)
        score += random.uniform(-0.05, 0.05)
        
        # Ensure score is in [0,1] range
        return min(1.0, max(0.0, score))
    
    def _generate_generic_strategy(self, system: str, area: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a generic strategy for a system"""
        metrics_to_improve = area.get("metrics_to_improve", [])
        current_metrics = area.get("current_metrics", {})
        
        strategy_type = "optimization"
        strategy_description = f"Optimize {system} system performance"
        strategy_details = {}
        expected_impact = {}
        
        # Customize strategy based on problematic metrics
        if "error_rate" in metrics_to_improve or current_metrics.get("error_rate", 0) > 0.3:
            strategy_type = "error_reduction"
            strategy_description = f"Reduce errors in {system} system"
            strategy_details = {
                "error_correction": True,
                "validation_level": "increased",
                "monitoring": "enhanced"
            }
            expected_impact = {
                "error_rate": -0.2,  # 20% reduction
                "success_rate": 0.15  # 15% improvement
            }
        elif "response_time" in metrics_to_improve or current_metrics.get("response_time", 0) > 2.0:
            strategy_type = "performance_optimization"
            strategy_description = f"Improve response time in {system} system"
            strategy_details = {
                "caching": True,
                "parallel_processing": True,
                "optimization_level": "aggressive"
            }
            expected_impact = {
                "response_time": -0.3,  # 30% reduction
                "throughput": 0.2  # 20% improvement
            }
        elif "success_rate" in metrics_to_improve or current_metrics.get("success_rate", 1.0) < 0.5:
            strategy_type = "success_improvement"
            strategy_description = f"Improve success rate in {system} system"
            strategy_details = {
                "fallback_mechanisms": True,
                "multiple_attempts": True,
                "adaptive_strategy": True
            }
            expected_impact = {
                "success_rate": 0.2,  # 20% improvement
                "error_rate": -0.15  # 15% reduction
            }
        else:
            # General optimization strategy
            strategy_details = {
                "optimization_level": "balanced",
                "resource_efficiency": True,
                "monitoring": "standard"
            }
            expected_impact = {
                "overall_performance": 0.15  # 15% improvement
            }
        
        return {
            "name": f"{strategy_type.capitalize()} for {system}",
            "system": system,
            "type": strategy_type,
            "description": strategy_description,
            "implementation": strategy_details,
            "expected_impact": expected_impact,
            "source": "meta_generated"
        }
    
    def _determine_attention_priority(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine if any system or process needs priority attention"""
        highest_priority = None
        
        # Check for critical bottlenecks
        if "bottlenecks" in context:
            critical_bottlenecks = [b for b in context["bottlenecks"] 
                                  if b.get("severity", 0) >= self.context.meta_parameters["bottleneck_severity_threshold"]]
            if critical_bottlenecks:
                top_bottleneck = critical_bottlenecks[0]
                priority = top_bottleneck.get("severity", 0.5)
                process_type = top_bottleneck.get("process_type", "unknown")
                process_name = top_bottleneck.get("process_name", "Unknown")
                
                highest_priority = {
                    "target": process_type,
                    "priority": priority,
                    "reason": f"Critical bottleneck in {process_name}",
                    "expiration": self.context.cognitive_cycle_count + self.context.meta_parameters["attention_default_duration"]
                }
        
        # Check for extremely low performance
        for system_name, data in self.context.performance_history.items():
            if not isinstance(data, dict) or "history" not in data or not data["history"]:
                continue
                
            latest = data["history"][-1].get("metrics", {})
            
            # Check for critically low performance
            if latest.get("success_rate", 1.0) < 0.2:  # Extremely low success
                priority = 0.9
                reason = f"Critically low success rate in {system_name}"
                
                if not highest_priority or priority > highest_priority.get("priority", 0):
                    highest_priority = {
                        "target": system_name,
                        "priority": priority,
                        "reason": reason,
                        "expiration": self.context.cognitive_cycle_count + self.context.meta_parameters["attention_default_duration"]
                    }
        
        # Check context for explicit attention requests
        if "attention_request" in context:
            attention_request = context["attention_request"]
            request_priority = attention_request.get("priority", 0.5)
            
            if not highest_priority or request_priority > highest_priority.get("priority", 0):
                highest_priority = {
                    "target": attention_request.get("target"),
                    "priority": request_priority,
                    "reason": attention_request.get("reason", "Explicit request"),
                    "expiration": attention_request.get("expiration", 
                                                     self.context.cognitive_cycle_count + self.context.meta_parameters["attention_default_duration"])
                }
        
        return highest_priority
    
    async def _set_attention_focus(self, focus: Dict[str, Any]) -> None:
        """Set the current attention focus"""
        self.context.attention_focus = {
            "target": focus.get("target"),
            "priority": focus.get("priority", 0.5),
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": focus.get("reason", ""),
            "expiration": focus.get("expiration")
        }
        
        logger.info(f"Attention focus set to {focus.get('target')} with priority {focus.get('priority', 0.5)}")

    async def _clear_attention_focus(self) -> None:
        """Clear the current attention focus"""
        old_focus = self.context.attention_focus
        self.context.attention_focus = None
        
        if old_focus:
            logger.info(f"Cleared attention focus from {old_focus.get('target')}")
    
    async def _register_cognitive_process(self, name: str, type: str,
                                      priority: float = 0.5, 
                                      resource_allocation: float = 0.1) -> str:
        """Register a new cognitive process for monitoring"""
        # Generate process ID
        process_id = f"process_{len(self.context.cognitive_processes) + 1}"
        
        # Create process object
        process = {
            "id": process_id,
            "name": name,
            "type": type,
            "priority": priority,
            "resource_allocation": resource_allocation,
            "performance_metrics": {},
            "bottlenecks": [],
            "dependencies": [],
            "start_time": datetime.datetime.now(),
            "last_activity": datetime.datetime.now(),
            "total_runtime": 0.0,
            "status": "idle"
        }
        
        # Add to processes
        self.context.cognitive_processes[process_id] = process
        
        return process_id
    
    async def _create_mental_model(self, name: str, domain: str,
                               confidence: float = 0.5,
                               complexity: float = 0.5) -> str:
        """Create a new mental model"""
        # Generate model ID
        model_id = f"model_{len(self.context.mental_models) + 1}"
        
        # Create model
        model = {
            "id": model_id,
            "name": name,
            "domain": domain,
            "confidence": confidence,
            "complexity": complexity,
            "elements": {},
            "relations": {},
            "last_updated": datetime.datetime.now().isoformat(),
            "last_used": datetime.datetime.now().isoformat(),
            "usage_count": 0,
            "accuracy_history": []
        }
        
        # Add to models
        self.context.mental_models[model_id] = model
        
        return model_id
    
    async def _extract_system_parameters(self, system: Any) -> Dict[str, Any]:
        """Extract current parameters from a system"""
        parameters = {}
        
        try:
            if hasattr(system, "get_parameters"):
                parameters = await system.get_parameters()
            elif hasattr(system, "parameters"):
                parameters = system.parameters
            elif hasattr(system, "get_config"):
                parameters = await system.get_config()
            elif hasattr(system, "config"):
                parameters = system.config
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
        
        return parameters
    
    async def _conduct_initial_assessment(self) -> None:
        """Conduct initial assessment of cognitive systems"""
        logger.info("Conducting initial assessment")
        
        # Start with baseline self-assessment
        initial_assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": 0,
            "systems": {},
            "overall_state": "initializing",
            "priorities": [],
            "initial_strategies": []
        }
        
        # Check each system for baseline metrics
        for system_name, system in self.context.system_references.items():
            if system_name in self.context.performance_history:
                try:
                    # Get initial metrics
                    metrics = {}
                    
                    if hasattr(system, "get_performance_metrics"):
                        metrics = await system.get_performance_metrics()
                    elif hasattr(system, "get_metrics"):
                        metrics = await system.get_metrics()
                    elif hasattr(system, "get_stats"):
                        metrics = await system.get_stats()
                    
                    initial_assessment["systems"][system_name] = {
                        "initial_metrics": metrics,
                        "parameters": self.context.performance_history[system_name]["parameters"]
                    }
                    
                    # Identify initial high-priority systems
                    if metrics.get("error_rate", 0) > 0.3:
                        initial_assessment["priorities"].append(system_name)
                    if metrics.get("success_rate", 1.0) < 0.5:
                        initial_assessment["priorities"].append(system_name)
                    
                except Exception as e:
                    logger.error(f"Error in initial assessment of {system_name}: {str(e)}")
                    initial_assessment["systems"][system_name] = {"error": str(e)}
        
        # Generate initial strategies for high-priority systems
        for system_name in set(initial_assessment["priorities"]):
            strategy = {
                "name": f"Initial Optimization for {system_name}",
                "system": system_name,
                "description": f"Initial performance improvement for {system_name}",
                "implementation": {
                    "type": "parameter_tuning",
                    "parameters": {
                        "optimization_level": "moderate",
                        "error_tolerance": "adaptive",
                        "performance_focus": True
                    }
                }
            }
            initial_assessment["initial_strategies"].append(strategy)
        
        # Create initial reflection
        self.context.reflections.append(initial_assessment)
        logger.info("Initial assessment completed")
