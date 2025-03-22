# nyx/core/dynamic_adaptation_system.py

import asyncio
import json
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    InputGuardrail,
    trace
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models for structured I/O
class StrategyParameters(BaseModel):
    exploration_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Rate of exploration vs exploitation")
    adaptation_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Rate of adaptation to changes")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Tolerance for risk in decisions")
    innovation_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Level of innovation in strategies")
    precision_focus: float = Field(default=0.5, ge=0.0, le=1.0, description="Focus on precision vs speed")

class Strategy(BaseModel):
    id: str
    name: str
    description: str
    parameters: StrategyParameters

class ContextFeatures(BaseModel):
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Complexity of the context")
    volatility: float = Field(default=0.2, ge=0.0, le=1.0, description="Volatility of the context")
    user_complexity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Complexity of user inputs")
    input_length: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized length of user input")
    task_familiarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Familiarity with the task")
    emotional_intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Emotional intensity in context")
    
class PerformanceMetrics(BaseModel):
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0) 
    response_time: Optional[float] = Field(default=None, ge=0.0)
    efficiency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class ContextAnalysisResult(BaseModel):
    significant_change: bool
    change_magnitude: float
    description: str
    features: ContextFeatures
    recommended_strategy_id: Optional[str] = None

class StrategySelectionResult(BaseModel):
    selected_strategy: Strategy
    confidence: float
    reasoning: str
    alternatives: List[str]

class MonitoringResult(BaseModel):
    trends: Dict[str, Dict[str, Any]]
    insights: List[str]
    performance_changes: Dict[str, float]
    bottlenecks: List[Dict[str, Any]]

class DynamicAdaptationContext:
    """Context object for sharing state between agents and tools"""
    
    def __init__(self):
        # Strategy registry
        self.strategies = {}
        
        # History and monitoring
        self.context_history = []
        self.strategy_history = []
        self.performance_history = []
        
        # Configuration
        self.max_history_size = 20
        self.context_change_threshold = 0.3
        
        # Current state
        self.current_strategy_id = "balanced"
        self.current_context = {}
        self.cycle_count = 0
        
        # Initialize flag
        self.initialized = False

class DynamicAdaptationSystem:
    """
    System for dynamically adapting to changing contexts and selecting optimal strategies.
    Refactored to use the OpenAI Agents SDK for improved modularity and functionality.
    """
    
    def __init__(self):
        # Initialize context for sharing state between agents
        self.context = DynamicAdaptationContext()
        
        # Initialize agents
        self.context_analyzer_agent = self._create_context_analyzer_agent()
        self.strategy_selector_agent = self._create_strategy_selector_agent()
        self.performance_monitor_agent = self._create_performance_monitor_agent()
        
        # Create main orchestration agent
        self.orchestrator_agent = self._create_orchestrator_agent()
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Trace ID for linking traces
        self.trace_group_id = f"nyx_adaptation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _initialize_default_strategies(self):
        """Initialize system with default strategies"""
        self.register_strategy({
            "id": "balanced",
            "name": "Balanced Approach",
            "description": "A balanced approach with moderate exploration and adaptation",
            "parameters": {
                "exploration_rate": 0.2,
                "adaptation_rate": 0.15,
                "risk_tolerance": 0.5,
                "innovation_level": 0.5,
                "precision_focus": 0.5
            }
        })
        
        self.register_strategy({
            "id": "exploratory",
            "name": "Exploratory Strategy",
            "description": "High exploration rate with focus on discovering new patterns",
            "parameters": {
                "exploration_rate": 0.4,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.7,
                "innovation_level": 0.8,
                "precision_focus": 0.3
            }
        })
        
        self.register_strategy({
            "id": "conservative",
            "name": "Conservative Strategy",
            "description": "Low risk with high precision focus",
            "parameters": {
                "exploration_rate": 0.1,
                "adaptation_rate": 0.1,
                "risk_tolerance": 0.2,
                "innovation_level": 0.3,
                "precision_focus": 0.8
            }
        })
        
        self.register_strategy({
            "id": "adaptive",
            "name": "Highly Adaptive Strategy",
            "description": "Focuses on quick adaptation to changes",
            "parameters": {
                "exploration_rate": 0.3,
                "adaptation_rate": 0.3,
                "risk_tolerance": 0.6,
                "innovation_level": 0.6,
                "precision_focus": 0.4
            }
        })
    
    def register_strategy(self, strategy: Dict[str, Any]) -> None:
        """
        Register a new strategy in the system.
        
        Args:
            strategy: Strategy definition to register
        """
        if "id" not in strategy:
            strategy["id"] = f"strategy_{len(self.context.strategies) + 1}"
            
        self.context.strategies[strategy["id"]] = strategy
    
    async def detect_context_change(self, current_context: Dict[str, Any]) -> ContextAnalysisResult:
        """
        Detect if there has been a significant change in context and analyze it
        
        Args:
            current_context: Current context information
            
        Returns:
            Analysis result with change detection information
        """
        with trace(workflow_name="Context_Analysis", group_id=self.trace_group_id):
            # Run the context analyzer agent
            result = await Runner.run(
                self.context_analyzer_agent,
                json.dumps({
                    "current_context": current_context,
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
            
        # Parse and return the result
        return ContextAnalysisResult.model_validate_json(result.final_output) if isinstance(result.final_output, str) else result.final_output
    
    async def select_strategy(self, context: Dict[str, Any], performance: Dict[str, Any]) -> StrategySelectionResult:
        """
        Select the optimal strategy for the current context.
        
        Args:
            context: Current context information
            performance: Current performance metrics
            
        Returns:
            Selected strategy and reasoning
        """
        with trace(workflow_name="Strategy_Selection", group_id=self.trace_group_id):
            # Run the strategy selector agent
            result = await Runner.run(
                self.strategy_selector_agent,
                json.dumps({
                    "context": context,
                    "performance": performance,
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
            
        # Parse and return the result
        return StrategySelectionResult.model_validate_json(result.final_output) if isinstance(result.final_output, str) else result.final_output
    
    async def monitor_performance(self, metrics: Dict[str, float]) -> MonitoringResult:
        """
        Monitor performance metrics and detect trends.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance analysis
        """
        with trace(workflow_name="Performance_Monitoring", group_id=self.trace_group_id):
            # Run the performance monitor agent
            result = await Runner.run(
                self.performance_monitor_agent,
                json.dumps({
                    "metrics": metrics,
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
            
        # Parse and return the result
        return MonitoringResult.model_validate_json(result.final_output) if isinstance(result.final_output, str) else result.final_output
    
    async def adaptation_cycle(self, context: Dict[str, Any], performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a complete adaptation cycle to analyze context and select a strategy.
        
        Args:
            context: Current context information
            performance_metrics: Current performance metrics
            
        Returns:
            Results of the adaptation cycle
        """
        self.context.cycle_count += 1
        self.context.current_context = context
        
        with trace(workflow_name="Adaptation_Cycle", group_id=self.trace_group_id):
            # Run the orchestrator agent
            result = await Runner.run(
                self.orchestrator_agent,
                json.dumps({
                    "context": context,
                    "performance_metrics": performance_metrics,
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
        
        # Parse and return the result
        return json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
        
    def _create_orchestrator_agent(self) -> Agent:
        """Create the main orchestration agent"""
        return Agent(
            name="Adaptation_Orchestrator",
            instructions="""
            You are the orchestration system for Nyx's dynamic adaptation capabilities.
            Your role is to coordinate adaptation cycles by:
            
            1. Analyzing the current context to detect significant changes
            2. Evaluating current performance metrics
            3. Selecting the optimal strategy based on context and performance
            4. Monitoring for trends and generating insights
            
            Your goal is to ensure that Nyx adapts appropriately to changing contexts
            and maintains optimal performance through strategy selection.
            
            Base your decisions on:
            - Significance of context changes
            - Current performance metrics and trends
            - Available strategies and their characteristics
            - Historical performance of different strategies
            
            Provide a clear summary of your decision-making process and recommendations.
            """,
            handoffs=[
                handoff(self.context_analyzer_agent, 
                       tool_name_override="analyze_context", 
                       tool_description_override="Analyze context and detect significant changes"),
                
                handoff(self.strategy_selector_agent, 
                       tool_name_override="select_strategy",
                       tool_description_override="Select optimal strategy based on context and performance"),
                
                handoff(self.performance_monitor_agent,
                       tool_name_override="monitor_performance",
                       tool_description_override="Monitor performance metrics and detect trends")
            ],
            tools=[
                function_tool(self._get_strategy),
                function_tool(self._update_strategy_history),
                function_tool(self._extract_context_features)
            ],
            model="gpt-4o",
            output_type=dict
        )
    
    def _create_context_analyzer_agent(self) -> Agent:
        """Create the context analyzer agent"""
        return Agent(
            name="Context_Analyzer",
            instructions="""
            You are the context analysis system for Nyx's dynamic adaptation architecture.
            
            Your role is to:
            1. Analyze the current context and compare it to historical context
            2. Detect significant changes in the context
            3. Extract relevant features from the context
            4. Recommend initial strategy based on context characteristics
            
            Focus on identifying meaningful changes while ignoring noise.
            Consider changes in:
            - User behavior and input patterns
            - Task complexity and requirements
            - Environmental factors
            - Emotional context
            
            Provide clear descriptions of detected changes and their potential impact.
            """,
            tools=[
                function_tool(self._calculate_context_difference),
                function_tool(self._generate_change_description),
                function_tool(self._extract_context_features),
                function_tool(self._calculate_context_complexity),
                function_tool(self._add_to_context_history)
            ],
            model="gpt-4o",
            output_type=ContextAnalysisResult
        )
    
    def _create_strategy_selector_agent(self) -> Agent:
        """Create the strategy selector agent"""
        return Agent(
            name="Strategy_Selector",
            instructions="""
            You are the strategy selection system for Nyx's dynamic adaptation architecture.
            
            Your role is to:
            1. Evaluate the current context and performance metrics
            2. Consider available strategies and their characteristics
            3. Select the optimal strategy for the current situation
            4. Provide reasoning for your selection
            
            Consider factors such as:
            - Context complexity and volatility
            - Current performance trends
            - Historical performance of strategies
            - Risk-reward tradeoffs
            
            Select a strategy that balances immediate performance with long-term adaptation.
            Provide clear reasoning for your selection and alternatives considered.
            """,
            tools=[
                function_tool(self._get_available_strategies),
                function_tool(self._calculate_strategy_score),
                function_tool(self._get_strategy),
                function_tool(self._calculate_context_volatility),
                function_tool(self._update_strategy_history)
            ],
            model="gpt-4o",
            output_type=StrategySelectionResult
        )
    
    def _create_performance_monitor_agent(self) -> Agent:
        """Create the performance monitoring agent"""
        return Agent(
            name="Performance_Monitor",
            instructions="""
            You are the performance monitoring system for Nyx's dynamic adaptation architecture.
            
            Your role is to:
            1. Track performance metrics over time
            2. Detect trends and anomalies in performance
            3. Identify potential bottlenecks and issues
            4. Generate insights about performance patterns
            
            Analyze metrics including:
            - Success rates and accuracy
            - Error rates and failure patterns
            - Response times and efficiency
            - Resource utilization
            
            Focus on identifying actionable insights that can drive strategy adjustments.
            """,
            tools=[
                function_tool(self._calculate_performance_trends),
                function_tool(self._generate_performance_insights),
                function_tool(self._calculate_resource_trend),
                function_tool(self._update_performance_history)
            ],
            model="gpt-4o",
            output_type=MonitoringResult
        )
    
    # Function tools for the agents
    
    @function_tool
    async def _calculate_context_difference(self, 
                                        current: Dict[str, Any], 
                                        previous: Dict[str, Any]) -> float:
        """
        Calculate the difference between two contexts
        
        Args:
            current: Current context
            previous: Previous context
            
        Returns:
            Difference magnitude (0.0-1.0)
        """
        # Focus on key elements common to both contexts
        common_keys = set(current.keys()) & set(previous.keys())
        if not common_keys:
            return 1.0  # Maximum difference if no common keys
        
        differences = []
        for key in common_keys:
            # Skip complex nested structures, consider only scalar values
            if isinstance(current[key], (str, int, float, bool)) and isinstance(previous[key], (str, int, float, bool)):
                if isinstance(current[key], bool) or isinstance(previous[key], bool):
                    # For boolean values, difference is either 0 or 1
                    diff = 0.0 if current[key] == previous[key] else 1.0
                elif isinstance(current[key], str) or isinstance(previous[key], str):
                    # For string values, difference is either 0 or 1
                    diff = 0.0 if str(current[key]) == str(previous[key]) else 1.0
                else:
                    # For numeric values, calculate normalized difference
                    max_val = max(abs(float(current[key])), abs(float(previous[key])))
                    if max_val > 0:
                        diff = abs(float(current[key]) - float(previous[key])) / max_val
                    else:
                        diff = 0.0
                differences.append(diff)
        
        if not differences:
            return 0.5  # Middle value if no comparable elements
            
        # Return average difference
        return sum(differences) / len(differences)
    
    @function_tool
    async def _generate_change_description(self, 
                                       current: Dict[str, Any], 
                                       previous: Dict[str, Any], 
                                       magnitude: float) -> str:
        """
        Generate a description of the context change
        
        Args:
            current: Current context
            previous: Previous context
            magnitude: Change magnitude
            
        Returns:
            Change description
        """
        changes = []
        
        # Check for new or modified keys
        for key in current:
            if key in previous:
                if current[key] != previous[key] and isinstance(current[key], (str, int, float, bool)):
                    changes.append(f"{key} changed from {previous[key]} to {current[key]}")
            else:
                changes.append(f"New element: {key}")
        
        # Check for removed keys
        for key in previous:
            if key not in current:
                changes.append(f"Removed element: {key}")
        
        if not changes:
            return f"Context changed with magnitude {magnitude:.2f}"
            
        change_desc = ", ".join(changes[:3])  # Limit to first 3 changes
        if len(changes) > 3:
            change_desc += f", and {len(changes) - 3} more changes"
            
        return f"Context changed ({magnitude:.2f}): {change_desc}"
    
    @function_tool
    async def _extract_context_features(self, context: Dict[str, Any]) -> ContextFeatures:
        """
        Extract numerical features from context for analysis
        
        Args:
            context: Context to analyze
            
        Returns:
            Context features
        """
        features = {}
        
        # Calculate complexity
        complexity = await self._calculate_context_complexity(context)
        features["complexity"] = complexity
        
        # Calculate volatility if we have history
        volatility = self._calculate_context_volatility()
        features["volatility"] = volatility
        
        # Extract feature from user input if present
        if "user_input" in context and isinstance(context["user_input"], str):
            features["input_length"] = min(1.0, len(context["user_input"]) / 500.0)
            features["user_complexity"] = min(1.0, len(set(context["user_input"].split())) / 100.0)
        
        # Extract emotional intensity if present
        if "emotional_state" in context:
            emotional_state = context["emotional_state"]
            if isinstance(emotional_state, dict) and "arousal" in emotional_state:
                features["emotional_intensity"] = emotional_state["arousal"]
        
        # Return as pydantic model
        return ContextFeatures(**features)
    
    @function_tool
    async def _calculate_context_complexity(self, context: Dict[str, Any]) -> float:
        """
        Calculate the complexity of the current context
        
        Args:
            context: Context to analyze
            
        Returns:
            Complexity score (0.0-1.0)
        """
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
    
    @function_tool
    async def _add_to_context_history(self, context: Dict[str, Any]) -> bool:
        """
        Add current context to the history
        
        Args:
            context: Context to add
        
        Returns:
            Success status
        """
        # Add context to history
        self.context.context_history.append(context)
        
        # Trim history if needed
        if len(self.context.context_history) > self.context.max_history_size:
            self.context.context_history.pop(0)
        
        return True
    
    @function_tool
    async def _get_available_strategies(self) -> List[Strategy]:
        """
        Get list of available strategies
        
        Returns:
            List of strategies
        """
        strategies = []
        for strategy_id, strategy_data in self.context.strategies.items():
            strategy = Strategy(
                id=strategy_id,
                name=strategy_data["name"],
                description=strategy_data["description"],
                parameters=StrategyParameters(**strategy_data["parameters"])
            )
            strategies.append(strategy)
        
        return strategies
    
    @function_tool
    async def _calculate_strategy_score(self,
                                    strategy: Strategy, 
                                    context_features: ContextFeatures,
                                    performance_metrics: Dict[str, Any]) -> float:
        """
        Calculate how well a strategy matches the current context
        
        Args:
            strategy: Strategy to evaluate
            context_features: Features of the current context
            performance_metrics: Current performance metrics
            
        Returns:
            Strategy score (0.0-1.0)
        """
        params = strategy.parameters.model_dump()
        
        # Base score starts at 0.5
        score = 0.5
        
        # Adjust based on complexity
        # Higher complexity prefers higher adaptation rate
        complexity_match = 1.0 - abs(context_features.complexity - params["adaptation_rate"])
        score += complexity_match * 0.1
        
        # Adjust based on volatility
        # Higher volatility prefers higher exploration rate
        volatility_match = 1.0 - abs(context_features.volatility - params["exploration_rate"])
        score += volatility_match * 0.1
        
        # Adjust based on performance metrics
        success_rate = performance_metrics.get("success_rate", 0.5)
        error_rate = performance_metrics.get("error_rate", 0.2)
        
        # If success is low, favor exploration
        if success_rate < 0.4:
            score += params["exploration_rate"] * 0.1
        # If error rate is high, favor precision
        if error_rate > 0.3:
            score += params["precision_focus"] * 0.1
        
        # Adjust for emotional intensity if available
        if context_features.emotional_intensity is not None:
            emotional_intensity = context_features.emotional_intensity
            # High emotional intensity might benefit from conservative approach
            if emotional_intensity > 0.7:
                score += (1.0 - params["risk_tolerance"]) * 0.1
        
        # Adjust for recency bias from strategy history
        recency_penalty = 0.0
        for i, history_item in enumerate(reversed(self.context.strategy_history[-5:])):
            if history_item["strategy_id"] == strategy.id:
                recency_penalty += 0.05 * (0.8 ** i)  # Exponential decay with distance
        
        score -= min(0.2, recency_penalty)  # Cap penalty
        
        # Add a small random factor for exploration
        score += random.uniform(-0.05, 0.05)
        
        # Ensure score is in [0,1] range
        return min(1.0, max(0.0, score))
    
    @function_tool
    async def _get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a specific strategy by ID
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            The strategy or None if not found
        """
        if strategy_id in self.context.strategies:
            strategy_data = self.context.strategies[strategy_id]
            return Strategy(
                id=strategy_id,
                name=strategy_data["name"],
                description=strategy_data["description"],
                parameters=StrategyParameters(**strategy_data["parameters"])
            )
        return None
    
    def _calculate_context_volatility(self) -> float:
        """
        Calculate the volatility of the context over time
        
        Returns:
            Volatility score (0.0-1.0)
        """
        if len(self.context.context_history) < 3:
            return 0.0  # Not enough history to calculate volatility
        
        # Calculate pairwise differences between consecutive contexts
        differences = []
        for i in range(1, len(self.context.context_history)):
            diff = asyncio.run(self._calculate_context_difference(
                self.context.context_history[i], 
                self.context.context_history[i-1]
            ))
            differences.append(diff)
        
        # Calculate variance of differences
        mean_diff = sum(differences) / len(differences)
        variance = sum((diff - mean_diff) ** 2 for diff in differences) / len(differences)
        
        # Normalize to [0,1]
        volatility = min(1.0, math.sqrt(variance) * 3.0)  # Scale to make values more meaningful
        
        return volatility
    
    @function_tool
    async def _update_strategy_history(self, strategy_id: str, context_summary: Dict[str, Any]) -> bool:
        """
        Update strategy history with newly selected strategy
        
        Args:
            strategy_id: ID of selected strategy
            context_summary: Summary of the context
            
        Returns:
            Success status
        """
        # Record strategy selection
        self.context.strategy_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "context_summary": context_summary,
            "cycle": self.context.cycle_count
        })
        
        # Update current strategy
        self.context.current_strategy_id = strategy_id
        
        # Trim history if needed
        if len(self.context.strategy_history) > self.context.max_history_size:
            self.context.strategy_history.pop(0)
        
        return True
    
    @function_tool
    async def _calculate_performance_trends(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate trends in performance metrics
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            Trends for each metric
        """
        trends = {}
        
        # Add metrics to history
        await self._update_performance_history(metrics)
        
        if len(self.context.performance_history) < 2:
            # Not enough history for trends
            for metric, value in metrics.items():
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
            return trends
        
        # Calculate trends for each metric
        for metric, current_value in metrics.items():
            # Find previous values for this metric
            previous_values = []
            for history_point in self.context.performance_history[:-1]:  # Skip current point
                if metric in history_point["metrics"]:
                    previous_values.append(history_point["metrics"][metric])
            
            if not previous_values:
                trends[metric] = {
                    "direction": "stable",
                    "magnitude": 0.0
                }
                continue
                
            # Calculate average of previous values
            avg_previous = sum(previous_values) / len(previous_values)
            
            # Calculate difference
            diff = current_value - avg_previous
            
            # Determine direction and magnitude
            if abs(diff) < 0.05:  # Small threshold for stability
                direction = "stable"
                magnitude = 0.0
            else:
                direction = "improving" if diff > 0 else "declining"
                magnitude = min(1.0, abs(diff))
                
            trends[metric] = {
                "direction": direction,
                "magnitude": magnitude,
                "diff_from_avg": diff
            }
        
        return trends
    
    @function_tool
    async def _generate_performance_insights(self, 
                                         metrics: Dict[str, float], 
                                         trends: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate insights based on performance metrics and trends
        
        Args:
            metrics: Performance metrics
            trends: Calculated trends
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Check for significant improvements
        improvements = [metric for metric, trend in trends.items() 
                       if trend["direction"] == "improving" and trend["magnitude"] > 0.1]
        if improvements:
            metrics_list = ", ".join(improvements)
            insights.append(f"Significant improvement in {metrics_list}")
        
        # Check for significant declines
        declines = [metric for metric, trend in trends.items() 
                   if trend["direction"] == "declining" and trend["magnitude"] > 0.1]
        if declines:
            metrics_list = ", ".join(declines)
            insights.append(f"Significant decline in {metrics_list}")
        
        # Check for overall performance
        avg_performance = sum(metrics.values()) / len(metrics) if metrics else 0.5
        if avg_performance > 0.8:
            insights.append("Overall performance is excellent")
        elif avg_performance < 0.4:
            insights.append("Overall performance is concerning")
        
        # Check for volatility
        volatility = self._calculate_performance_volatility()
        if volatility > 0.2:
            insights.append("Performance metrics show high volatility")
        
        return insights
    
    @function_tool
    async def _calculate_resource_trend(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate trend from a series of resource values
        
        Args:
            values: Series of values to analyze
            
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
    
    @function_tool
    async def _update_performance_history(self, metrics: Dict[str, float]) -> bool:
        """
        Update performance history with current metrics
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            Success status
        """
        # Add metrics to history
        self.context.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "strategy_id": self.context.current_strategy_id,
            "cycle": self.context.cycle_count
        })
        
        # Trim history if needed
        if len(self.context.performance_history) > self.context.max_history_size:
            self.context.performance_history.pop(0)
        
        return True
    
    def _calculate_performance_volatility(self) -> float:
        """
        Calculate the volatility of performance metrics over time
        
        Returns:
            Volatility score (0.0-1.0)
        """
        if len(self.context.performance_history) < 3:
            return 0.0  # Not enough history
        
        # Extract all metric values
        metric_values = {}
        
        for history_point in self.context.performance_history:
            for metric, value in history_point["metrics"].items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
        
        # Calculate standard deviation for each metric
        std_devs = []
        for values in metric_values.values():
            if len(values) >= 3:  # Need at least 3 points
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std_devs.append(math.sqrt(variance))
        
        if not std_devs:
            return 0.0
            
        # Average standard deviation across metrics
        avg_std_dev = sum(std_devs) / len(std_devs)
        
        # Normalize to [0,1] with reasonable scaling
        volatility = min(1.0, avg_std_dev * 3.0)
        
        return volatility
