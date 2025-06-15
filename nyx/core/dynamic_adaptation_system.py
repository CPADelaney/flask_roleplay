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
    GuardrailFunctionOutput,
    trace,
    RunContextWrapper
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class KVPair(BaseModel):
    key: str
    value: Any

class MetricKV(BaseModel):
    """name → value pair used for passing current metrics"""
    metric: str
    value: float


class MetricTrend(BaseModel):
    """output of the trend calculator"""
    metric: str
    direction: str            # 'improving' | 'declining' | 'stable'
    magnitude: float
    diff_from_avg: float | None = None


class ResourceTrend(BaseModel):
    """result of the resource-trend regression helper"""
    direction: str            # 'increasing' | 'decreasing' | 'stable'
    magnitude: float
    slope: float | None = None
    mean: float | None = None

class RawContext(BaseModel):
    items: list[KVPair]

    def to_dict(self) -> dict[str, Any]:
        return {kv.key: kv.value for kv in self.items}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RawContext":
        return RawContext(items=[KVPair(key=k, value=v) for k, v in d.items()])


    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RawContext":
        return RawContext(items=[KVPair(key=k, value=v) for k, v in d.items()])

# Pydantic models for structured I/O
class StrategyParameters(BaseModel):
    exploration_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Rate of exploration vs exploitation")
    adaptation_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Rate of adaptation to changes")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Tolerance for risk in decisions")
    innovation_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Level of innovation in strategies")
    precision_focus: float = Field(default=0.5, ge=0.0, le=1.0, description="Focus on precision vs speed")
    experience_sharing_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Rate of experience sharing")
    cross_user_sharing: float = Field(default=0.3, ge=0.0, le=1.0, description="Level of cross-user experience sharing")
    identity_evolution_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Rate of identity evolution")

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
    user_engagement: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Level of user engagement")
    identity_stability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Stability of identity profile")
    experience_relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance of experiences")
    
class PerformanceMetrics(BaseModel):
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0) 
    response_time: Optional[float] = Field(default=None, ge=0.0)
    efficiency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    experience_utility: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Utility of shared experiences")
    user_satisfaction: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="User satisfaction")

class ContextAnalysisResult(BaseModel):
    significant_change: bool = Field(description="Whether a significant context change was detected")
    change_magnitude: float = Field(description="Magnitude of the context change (0.0-1.0)")
    description: str = Field(description="Description of the detected changes")
    features: ContextFeatures = Field(description="Extracted context features")
    recommended_strategy_id: Optional[str] = Field(default=None, description="ID of recommended strategy if any")

class StrategySelectionResult(BaseModel):
    selected_strategy: Strategy = Field(description="The selected strategy")
    confidence: float = Field(description="Confidence in the selection (0.0-1.0)")
    reasoning: str = Field(description="Reasoning behind the selection")
    alternatives: List[str] = Field(description="Alternative strategies considered")
    experience_impact: Dict[str, float] = Field(default_factory=dict, description="Impact on experience sharing")
    identity_impact: Dict[str, float] = Field(default_factory=dict, description="Impact on identity evolution")

class MonitoringResult(BaseModel):
    trends: Dict[str, Dict[str, Any]] = Field(description="Performance trends")
    insights: List[str] = Field(description="Generated insights")
    performance_changes: Dict[str, float] = Field(description="Notable performance changes")
    bottlenecks: List[Dict[str, Any]] = Field(description="Identified bottlenecks")
    experience_trends: Dict[str, Any] = Field(default_factory=dict, description="Trends in experience sharing")

class ExperienceAdaptationParams(BaseModel):
    cross_user_enabled: bool = Field(default=True, description="Whether cross-user sharing is enabled")
    sharing_threshold: float = Field(default=0.7, description="Threshold for sharing experiences")
    experience_types: List[str] = Field(default_factory=list, description="Prioritized experience types")
    personalization_level: float = Field(default=0.5, description="Level of personalization")

class IdentityAdaptationParams(BaseModel):
    evolution_rate: float = Field(default=0.2, description="Rate of identity evolution")
    trait_stability: float = Field(default=0.7, description="Stability of identity traits")
    preference_adaptability: float = Field(default=0.5, description="Adaptability of preferences")
    consolidation_frequency: float = Field(default=0.3, description="Frequency of identity consolidation")

class AdaptationCycleResult(BaseModel):
    context_analysis: ContextAnalysisResult = Field(description="Results of context analysis")
    selected_strategy: Optional[Strategy] = Field(default=None, description="Selected strategy")
    strategy_confidence: float = Field(description="Confidence in strategy selection")
    experience_adaptation: Optional[ExperienceAdaptationParams] = Field(default=None, description="Experience adaptation parameters")
    identity_adaptation: Optional[IdentityAdaptationParams] = Field(default=None, description="Identity adaptation parameters")
    insights: List[str] = Field(default_factory=list, description="Insights from the adaptation cycle")

class DynamicAdaptationContext:
    """Context object for sharing state between agents and tools"""
    
    def __init__(self):
        # Strategy registry
        self.strategies = {}
        
        # History and monitoring
        self.context_history = []
        self.strategy_history = []
        self.performance_history = []
        self.experience_sharing_history = []
        self.identity_evolution_history = []
        
        # Configuration
        self.max_history_size = 20
        self.context_change_threshold = 0.3
        
        # Current state
        self.current_strategy_id = "balanced"
        self.current_context = {}
        self.cycle_count = 0
        
        # Experience adaptation parameters
        self.experience_adaptation = ExperienceAdaptationParams(
            cross_user_enabled=True,
            sharing_threshold=0.7,
            experience_types=["teasing", "discipline", "dark", "psychological", "nurturing"],
            personalization_level=0.5
        )
        
        # Identity adaptation parameters
        self.identity_adaptation = IdentityAdaptationParams(
            evolution_rate=0.2, 
            trait_stability=0.7,
            preference_adaptability=0.5, 
            consolidation_frequency=0.3
        )
        
        # Initialize flag
        self.initialized = False

class DynamicAdaptationSystem:
    """
    System for dynamically adapting to changing contexts and selecting optimal strategies.
    Enhanced with capabilities for experience sharing and identity evolution adaptation.
    Refactored to use the OpenAI Agents SDK for improved modularity and functionality.
    """
    
    def __init__(self):
        # Initialize context for sharing state between agents
        self.context = DynamicAdaptationContext()
        
        # Create specialized agents 
        self.context_analyzer_agent = self._create_context_analyzer_agent()
        self.strategy_selector_agent = self._create_strategy_selector_agent()
        self.performance_monitor_agent = self._create_performance_monitor_agent()
        self.experience_adaptation_agent = self._create_experience_adaptation_agent()
        self.identity_adaptation_agent = self._create_identity_adaptation_agent()
        
        # Create main orchestration agent
        self.orchestrator_agent = self._create_orchestrator_agent()
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Prediction error settings
        self.prediction_error_threshold = 0.4  # Threshold for significant prediction error
        self.last_prediction_evaluation = None  # Store last prediction evaluation
        
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
                "precision_focus": 0.5,
                "experience_sharing_rate": 0.5,
                "cross_user_sharing": 0.3,
                "identity_evolution_rate": 0.2
            }
        })
        
        self.register_strategy({
            "id": "exploratory",
            "name": "Exploratory Strategy",
            "description": "High exploration rate with focus on discovering new patterns and experiences",
            "parameters": {
                "exploration_rate": 0.4,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.7,
                "innovation_level": 0.8,
                "precision_focus": 0.3,
                "experience_sharing_rate": 0.7,
                "cross_user_sharing": 0.6,
                "identity_evolution_rate": 0.3
            }
        })
        
        self.register_strategy({
            "id": "conservative",
            "name": "Conservative Strategy",
            "description": "Low risk with high precision focus and limited experience sharing",
            "parameters": {
                "exploration_rate": 0.1,
                "adaptation_rate": 0.1,
                "risk_tolerance": 0.2,
                "innovation_level": 0.3,
                "precision_focus": 0.8,
                "experience_sharing_rate": 0.3,
                "cross_user_sharing": 0.1,
                "identity_evolution_rate": 0.1
            }
        })
        
        self.register_strategy({
            "id": "adaptive",
            "name": "Highly Adaptive Strategy",
            "description": "Focuses on quick adaptation to changes with significant experience sharing",
            "parameters": {
                "exploration_rate": 0.3,
                "adaptation_rate": 0.3,
                "risk_tolerance": 0.6,
                "innovation_level": 0.6,
                "precision_focus": 0.4,
                "experience_sharing_rate": 0.6,
                "cross_user_sharing": 0.5,
                "identity_evolution_rate": 0.4
            }
        })
        
        self.register_strategy({
            "id": "experience_focused",
            "name": "Experience-Focused Strategy",
            "description": "Maximizes experience sharing and identity evolution",
            "parameters": {
                "exploration_rate": 0.3,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.6,
                "innovation_level": 0.5,
                "precision_focus": 0.4,
                "experience_sharing_rate": 0.9,
                "cross_user_sharing": 0.8,
                "identity_evolution_rate": 0.6
            }
        })
        
        self.register_strategy({
            "id": "identity_stable",
            "name": "Identity Stability Strategy",
            "description": "Focuses on maintaining stable identity with moderate experience sharing",
            "parameters": {
                "exploration_rate": 0.2,
                "adaptation_rate": 0.2,
                "risk_tolerance": 0.4,
                "innovation_level": 0.4,
                "precision_focus": 0.6,
                "experience_sharing_rate": 0.5,
                "cross_user_sharing": 0.3,
                "identity_evolution_rate": 0.1
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

    async def adapt_from_prediction_error(self, 
                                      prediction_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt system parameters based on prediction error
        
        Args:
            prediction_evaluation: Evaluation of a prediction
            
        Returns:
            Adaptation results
        """
        # Store evaluation
        self.last_prediction_evaluation = prediction_evaluation
        
        # Extract error information
        prediction_error = prediction_evaluation.get("prediction_error", 0.0)
        error_details = prediction_evaluation.get("error_details", {})
        
        # Check if error exceeds threshold
        if prediction_error <= self.prediction_error_threshold:
            # Error is below threshold, no adaptation needed
            return {
                "adapted": False,
                "reason": "Prediction error below threshold",
                "error": prediction_error,
                "threshold": self.prediction_error_threshold
            }
        
        # Error exceeds threshold, adaptation is needed
        
        # Create adaptable context based on prediction details
        adaptation_context = {
            "prediction_error": prediction_error,
            "error_details": error_details,
            "prediction_id": prediction_evaluation.get("prediction_id"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Run adaptation cycle with the context
        adaptation_result = await self.adaptation_cycle(
            adaptation_context,
            {"prediction_error": prediction_error}
        )
        
        # Add prediction-specific information to result
        adaptation_result["adapted_from_prediction"] = True
        adaptation_result["prediction_error"] = prediction_error
        
        # If adaptation resulted in strategy change, adjust prediction parameters
        if "selected_strategy" in adaptation_result:
            strategy_params = adaptation_result["selected_strategy"].get("parameters", {})
            
            # Higher exploration rate leads to more varied predictions
            if "exploration_rate" in strategy_params:
                exploration_rate = strategy_params["exploration_rate"]
                # Apply to prediction systems
                # This is a placeholder for implementation
                
            # Higher adaptation rate leads to faster learning from errors
            if "adaptation_rate" in strategy_params:
                adaptation_rate = strategy_params["adaptation_rate"]
                # Apply to prediction learning rate
        
        return adaptation_result
    
    async def get_prediction_based_strategy(self, 
                                         prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommended strategy based on a prediction
        
        Args:
            prediction_result: Prediction result
            
        Returns:
            Recommended strategy
        """
        # Extract confidence and horizon
        confidence = prediction_result.get("confidence", 0.5)
        horizon = prediction_result.get("prediction_horizon", "immediate")
        
        # Map prediction parameters to strategy selection
        
        # For high confidence, immediate horizon predictions
        if confidence > 0.7 and horizon == "immediate":
            # Use conservative strategy - we're confident about what's coming
            recommended_strategy = "conservative"
            reason = "High confidence in immediate predictions suggests optimizing precision"
        
        # For low confidence, immediate horizon
        elif confidence < 0.4 and horizon == "immediate":
            # Use exploratory strategy - we're uncertain about what's coming
            recommended_strategy = "exploratory"
            reason = "Low confidence in immediate predictions suggests exploration needed"
        
        # For medium to long-term horizons
        elif horizon in ["medium_term", "long_term"]:
            # Use adaptive strategy for longer horizons
            recommended_strategy = "adaptive"
            reason = f"Longer horizon ({horizon}) predictions require adaptability"
        
        # Default to balanced approach
        else:
            recommended_strategy = "balanced"
            reason = "Default balanced approach for moderate prediction parameters"
        
        # Get the full strategy
        strategy_result = await self._get_strategy(
            RunContextWrapper(context=self.context),
            recommended_strategy
        )
        
        strategy = strategy_result if isinstance(strategy_result, Strategy) else Strategy.model_validate(strategy_result)
        
        return {
            "recommended_strategy_id": recommended_strategy,
            "reason": reason,
            "confidence": confidence,
            "horizon": horizon,
            "strategy": strategy.model_dump() if hasattr(strategy, "model_dump") else strategy
        }
    
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
        response_data = json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
        
        # Update current strategy if selected
        if "selected_strategy" in response_data and "id" in response_data["selected_strategy"]:
            self.context.current_strategy_id = response_data["selected_strategy"]["id"]
            
            # Update experience adaptation parameters if available
            if "experience_adaptation" in response_data:
                for key, value in response_data["experience_adaptation"].items():
                    if hasattr(self.context.experience_adaptation, key):
                        setattr(self.context.experience_adaptation, key, value)
            
            # Update identity adaptation parameters if available
            if "identity_adaptation" in response_data:
                for key, value in response_data["identity_adaptation"].items():
                    if hasattr(self.context.identity_adaptation, key):
                        setattr(self.context.identity_adaptation, key, value)
        
        return response_data
    
    async def get_experience_adaptation_params(self) -> Dict[str, Any]:
        """
        Get current experience adaptation parameters.
        
        Returns:
            Experience adaptation parameters
        """
        # Get current strategy
        strategy_id = self.context.current_strategy_id
        strategy = self.context.strategies.get(strategy_id, {})
        
        # Get strategy parameters
        strategy_params = strategy.get("parameters", {})
        
        # Map strategy parameters to experience adaptation
        experience_sharing_rate = strategy_params.get("experience_sharing_rate", 0.5)
        cross_user_sharing = strategy_params.get("cross_user_sharing", 0.3)
        
        # Update experience adaptation parameters
        self.context.experience_adaptation.cross_user_enabled = cross_user_sharing > 0.3
        self.context.experience_adaptation.sharing_threshold = max(0.5, 1.0 - experience_sharing_rate)
        self.context.experience_adaptation.personalization_level = experience_sharing_rate * 0.8
        
        return self.context.experience_adaptation.model_dump()
    
    async def get_identity_adaptation_params(self) -> Dict[str, Any]:
        """
        Get current identity adaptation parameters.
        
        Returns:
            Identity adaptation parameters
        """
        # Get current strategy
        strategy_id = self.context.current_strategy_id
        strategy = self.context.strategies.get(strategy_id, {})
        
        # Get strategy parameters
        strategy_params = strategy.get("parameters", {})
        
        # Map strategy parameters to identity adaptation
        identity_evolution_rate = strategy_params.get("identity_evolution_rate", 0.2)
        adaptation_rate = strategy_params.get("adaptation_rate", 0.15)
        
        # Update identity adaptation parameters
        self.context.identity_adaptation.evolution_rate = identity_evolution_rate
        self.context.identity_adaptation.trait_stability = 1.0 - (adaptation_rate * 0.5)  # Higher adaptation = lower stability
        self.context.identity_adaptation.preference_adaptability = adaptation_rate * 1.5  # Higher adaptation = higher adaptability
        self.context.identity_adaptation.consolidation_frequency = max(0.1, identity_evolution_rate * 1.5)
        
        return self.context.identity_adaptation.model_dump()
    
    async def adapt_experience_sharing(self, 
                                    user_id: str, 
                                    feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt experience sharing based on user feedback.
        
        Args:
            user_id: User ID
            feedback: User feedback information
            
        Returns:
            Updated experience adaptation parameters
        """
        with trace(workflow_name="Experience_Adaptation", group_id=self.trace_group_id):
            # Run the experience adaptation agent
            result = await Runner.run(
                self.experience_adaptation_agent,
                json.dumps({
                    "user_id": user_id,
                    "feedback": feedback,
                    "current_params": self.context.experience_adaptation.model_dump(),
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
            
        # Parse the result
        adapted_params = json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
        
        # Update experience adaptation parameters
        if isinstance(adapted_params, dict) and "cross_user_enabled" in adapted_params:
            for key, value in adapted_params.items():
                if hasattr(self.context.experience_adaptation, key):
                    setattr(self.context.experience_adaptation, key, value)
        
        # Record adaptation in history
        self.context.experience_sharing_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "feedback": feedback,
            "adapted_params": self.context.experience_adaptation.model_dump()
        })
        
        # Limit history size
        if len(self.context.experience_sharing_history) > self.context.max_history_size:
            self.context.experience_sharing_history = self.context.experience_sharing_history[-self.context.max_history_size:]
        
        return self.context.experience_adaptation.model_dump()
    
    async def adapt_identity_evolution(self, 
                                   identity_state: Dict[str, Any], 
                                   performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt identity evolution parameters based on current state and performance.
        
        Args:
            identity_state: Current identity state
            performance: Performance metrics
            
        Returns:
            Updated identity adaptation parameters
        """
        with trace(workflow_name="Identity_Adaptation", group_id=self.trace_group_id):
            # Run the identity adaptation agent
            result = await Runner.run(
                self.identity_adaptation_agent,
                json.dumps({
                    "identity_state": identity_state,
                    "performance": performance,
                    "current_params": self.context.identity_adaptation.model_dump(),
                    "cycle": self.context.cycle_count
                }),
                context=self.context
            )
            
        # Parse the result
        adapted_params = json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
        
        # Update identity adaptation parameters
        if isinstance(adapted_params, dict) and "evolution_rate" in adapted_params:
            for key, value in adapted_params.items():
                if hasattr(self.context.identity_adaptation, key):
                    setattr(self.context.identity_adaptation, key, value)
        
        # Record adaptation in history
        self.context.identity_evolution_history.append({
            "timestamp": datetime.now().isoformat(),
            "identity_state": identity_state,
            "adapted_params": self.context.identity_adaptation.model_dump()
        })
        
        # Limit history size
        if len(self.context.identity_evolution_history) > self.context.max_history_size:
            self.context.identity_evolution_history = self.context.identity_evolution_history[-self.context.max_history_size:]
        
        return self.context.identity_adaptation.model_dump()
    
    # Agent creation methods
    
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
            5. Adapting experience sharing and identity evolution parameters
            
            Your goal is to ensure that Nyx adapts appropriately to changing contexts
            and maintains optimal performance through strategy selection.
            
            Base your decisions on:
            - Significance of context changes
            - Current performance metrics and trends
            - Available strategies and their characteristics
            - Historical performance of different strategies
            - User feedback and engagement metrics
            - Experience sharing effectiveness
            - Identity evolution stability
            
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
                       tool_description_override="Monitor performance metrics and detect trends"),
                       
                handoff(self.experience_adaptation_agent,
                       tool_name_override="adapt_experience_sharing",
                       tool_description_override="Adapt experience sharing parameters based on feedback"),
                       
                handoff(self.identity_adaptation_agent,
                       tool_name_override="adapt_identity_evolution",
                       tool_description_override="Adapt identity evolution parameters based on performance")
            ],
            tools=[
                function_tool(self._get_strategy),
                function_tool(self._update_strategy_history),
                function_tool(self._extract_context_features),
                function_tool(self._get_experience_adaptation_settings),
                function_tool(self._get_identity_adaptation_settings)
            ],
            model="gpt-4.1-nano",
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
            - Experience sharing effectiveness
            - Identity stability and evolution
            
            Provide clear descriptions of detected changes and their potential impact.
            """,
            tools=[
                function_tool(self._calculate_context_difference),
                function_tool(self._generate_change_description),
                function_tool(self._extract_context_features),
                function_tool(self._calculate_context_complexity),
                function_tool(self._add_to_context_history),
                function_tool(self._analyze_experience_context),
                function_tool(self._analyze_identity_context)
            ],
            model="gpt-4.1-nano",
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
            5. Determine appropriate experience sharing parameters
            6. Determine appropriate identity evolution parameters
            
            Consider factors such as:
            - Context complexity and volatility
            - Current performance trends
            - Historical performance of strategies
            - Risk-reward tradeoffs
            - User engagement and feedback
            - Experience sharing effectiveness
            - Identity stability and coherence
            
            Select a strategy that balances immediate performance with long-term adaptation.
            Provide clear reasoning for your selection and alternatives considered.
            """,
            tools=[
                function_tool(self._get_available_strategies),
                function_tool(self._calculate_strategy_score),
                function_tool(self._get_strategy),
                function_tool(self._calculate_context_volatility),
                function_tool(self._update_strategy_history),
                function_tool(self._calculate_experience_strategy_fit),
                function_tool(self._calculate_identity_strategy_fit)
            ],
            model="gpt-4.1-nano",
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
            5. Monitor experience sharing effectiveness
            6. Monitor identity evolution coherence
            
            Analyze metrics including:
            - Success rates and accuracy
            - Error rates and failure patterns
            - Response times and efficiency
            - Resource utilization
            - Experience sharing metrics
            - Identity stability metrics
            
            Focus on identifying actionable insights that can drive strategy adjustments.
            """,
            tools=[
                function_tool(self._calculate_performance_trends),
                function_tool(self._generate_performance_insights),
                function_tool(self._calculate_resource_trend),
                function_tool(self._update_performance_history),
                function_tool(self._analyze_experience_performance),
                function_tool(self._analyze_identity_performance)
            ],
            model="gpt-4.1-nano",
            output_type=MonitoringResult
        )
    
    def _create_experience_adaptation_agent(self) -> Agent:
        """Create the experience adaptation agent"""
        return Agent(
            name="Experience_Adaptation",
            instructions="""
            You are the experience adaptation system for Nyx's dynamic adaptation architecture.
            
            Your role is to:
            1. Analyze user feedback on experience sharing
            2. Adjust experience sharing parameters based on feedback
            3. Balance cross-user experience sharing with relevance
            4. Optimize personalization for experience recall
            5. Determine appropriate scenario types to focus on
            
            Consider factors such as:
            - User engagement with experiences
            - Relevance of shared experiences
            - Privacy considerations for cross-user sharing
            - Effectiveness of experience consolidation
            - Current strategy's parameters
            
            Provide clear explanations for parameter adjustments.
            """,
            tools=[
                function_tool(self._get_current_experience_params),
                function_tool(self._calculate_experience_adaptation),
                function_tool(self._update_experience_sharing_history),
                function_tool(self._get_strategy)
            ],
            model="gpt-4.1-nano",
            output_type=ExperienceAdaptationParams
        )
    
    def _create_identity_adaptation_agent(self) -> Agent:
        """Create the identity adaptation agent"""
        return Agent(
            name="Identity_Adaptation",
            instructions="""
            You are the identity adaptation system for Nyx's dynamic adaptation architecture.
            
            Your role is to:
            1. Analyze identity evolution metrics
            2. Adjust identity evolution parameters based on performance
            3. Balance stability with adaptability
            4. Optimize trait and preference updates
            5. Determine appropriate consolidation frequency
            
            Consider factors such as:
            - Coherence of identity evolution
            - Stability of core traits
            - Adaptability of preferences
            - Impact of experience sharing on identity
            - Current strategy's parameters
            
            Provide clear explanations for parameter adjustments.
            """,
            tools=[
                function_tool(self._get_current_identity_params),
                function_tool(self._calculate_identity_adaptation),
                function_tool(self._update_identity_evolution_history),
                function_tool(self._get_strategy)
            ],
            model="gpt-4.1-nano",
            output_type=IdentityAdaptationParams
        )
    
    # Function tools for the agents
    @staticmethod
    @function_tool
    async def _calculate_context_difference(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        current: RawContext,
        previous: RawContext
    ) -> float:
        cur, prev = current.to_dict(), previous.to_dict()
        common = set(cur) & set(prev)
        if not common:
            return 1.0
    
        diffs: list[float] = []
        for k in common:
            a, b = cur[k], prev[k]
            if isinstance(a, bool) and isinstance(b, bool):
                diffs.append(0.0 if a == b else 1.0)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                mx = max(abs(float(a)), abs(float(b)))
                diffs.append(abs(float(a) - float(b)) / mx if mx else 0.0)
            else:                                      # strings / mixed scalars
                diffs.append(0.0 if str(a) == str(b) else 1.0)
    
        return sum(diffs) / len(diffs) if diffs else 0.5


    @staticmethod
    @function_tool
    async def _generate_change_description(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        current: RawContext,
        previous: RawContext,
        magnitude: float
    ) -> str:
        cur, prev = current.to_dict(), previous.to_dict()
        changes: list[str] = []
    
        for k in cur:
            if k in prev:
                if cur[k] != prev[k]:
                    changes.append(f"{k} changed from {prev[k]} to {cur[k]}")
            else:
                changes.append(f"New element: {k}")
        for k in prev:
            if k not in cur:
                changes.append(f"Removed element: {k}")
    
        # domain-specific flags (kept exactly as in your original code)
        if cur.get("has_experience") and not prev.get("has_experience"):
            changes.append("Experience sharing was activated")
        elif prev.get("has_experience") and not cur.get("has_experience"):
            changes.append("Experience sharing was deactivated")
        if cur.get("cross_user_experience"):
            changes.append("Cross-user experience sharing was used")
        if cur.get("identity_impact"):
            changes.append("Experience had significant identity impact")
    
        if not changes:
            return f"Context changed with magnitude {magnitude:.2f}"
    
        desc = ", ".join(changes[:3])
        if len(changes) > 3:
            desc += f", and {len(changes) - 3} more changes"
        return f"Context changed ({magnitude:.2f}): {desc}"
    
    @staticmethod
    @function_tool
    async def _extract_context_features(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        context: RawContext
    ) -> ContextFeatures:
        cdict = context.to_dict()
        # — complexity / volatility calculations are identical —
        complexity = await ctx.agent._calculate_context_complexity(ctx, context)  # type: ignore
        volatility = ctx.agent._calculate_context_volatility()                    # type: ignore
    
        feat: dict[str, Any] = {
            "complexity" : complexity,
            "volatility" : volatility,
        }
    
        # all the same feature extraction logic:
        if (uin := cdict.get("user_input")):
            feat["input_length"]   = min(1.0, len(str(uin)) / 500.0)
            feat["user_complexity"] = min(1.0, len(set(str(uin).split())) / 100.0)
    
        if (emo := cdict.get("emotional_state", {})) and isinstance(emo, dict):
            if "arousal" in emo:
                feat["emotional_intensity"] = emo["arousal"]
    
        if cdict.get("has_experience"):
            feat["experience_relevance"] = min(1.0, cdict.get("relevance_score", 0.7))
            if cdict.get("cross_user_experience"):
                feat["complexity"] = min(1.0, feat["complexity"] + 0.2)
    
        feat["identity_stability"] = 0.3 if cdict.get("identity_impact") else 0.8
        if (ic := cdict.get("interaction_count")) is not None:
            feat["user_engagement"] = min(1.0, ic / 50.0)
    
        return ContextFeatures(**feat)

    @staticmethod
    @function_tool
    async def _calculate_context_complexity(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        context: RawContext
    ) -> float:
        cdict = context.to_dict()
        total_elements = nested_elements = max_depth = 0
    
        def count(obj, depth=0):
            nonlocal total_elements, nested_elements, max_depth
            max_depth = max(max_depth, depth)
            if isinstance(obj, dict):
                total_elements += len(obj)
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        nested_elements += 1
                        count(v, depth + 1)
            elif isinstance(obj, list):
                total_elements += len(obj)
                for v in obj:
                    if isinstance(v, (dict, list)):
                        nested_elements += 1
                        count(v, depth + 1)
    
        count(cdict)
    
        size_factor       = min(1.0, total_elements  / 50.0)
        nesting_factor    = min(1.0, nested_elements / 10.0)
        depth_factor      = min(1.0, max_depth       / 5.0)
        experience_factor = 0.1 if cdict.get("has_experience") else 0.0
        if cdict.get("cross_user_experience"):
            experience_factor += 0.2
        identity_factor   = 0.2 if cdict.get("identity_impact") else 0.0
    
        return (
            size_factor   * 0.3 +
            nesting_factor* 0.2 +
            depth_factor  * 0.2 +
            experience_factor * 0.15 +
            identity_factor   * 0.15
        )

    @staticmethod
    @function_tool
    async def _add_to_context_history(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        context: RawContext
    ) -> bool:
        ctx.context.context_history.append(context.to_dict())
        if len(ctx.context.context_history) > ctx.context.max_history_size:
            ctx.context.context_history.pop(0)
        return True

    @staticmethod    
    @function_tool
    async def _get_available_strategies(ctx: RunContextWrapper[DynamicAdaptationContext]) -> List[Strategy]:
        """
        Get list of available strategies
        
        Returns:
            List of strategies
        """
        strategies = []
        for strategy_id, strategy_data in ctx.context.strategies.items():
            strategy = Strategy(
                id=strategy_id,
                name=strategy_data["name"],
                description=strategy_data["description"],
                parameters=StrategyParameters(**strategy_data["parameters"])
            )
            strategies.append(strategy)
        
        return strategies

    @staticmethod
    @function_tool
    async def _calculate_strategy_score(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        strategy: Strategy,
        context_features: ContextFeatures,
        performance_metrics: PerformanceMetrics
    ) -> float:
        params = strategy.parameters.model_dump()
        perf   = performance_metrics.model_dump(exclude_none=True)
    
        score = 0.5
        score += (1 - abs(context_features.complexity  - params["adaptation_rate"])) * 0.1
        score += (1 - abs(context_features.volatility - params["exploration_rate"])) * 0.1
    
        success = perf.get("success_rate", 0.5)
        errors  = perf.get("error_rate"  , 0.2)
        if success < 0.4:
            score += params["exploration_rate"] * 0.1
        if errors  > 0.3:
            score += params["precision_focus"] * 0.1
    
        if context_features.experience_relevance is not None:
            if context_features.experience_relevance > 0.7:
                score += params["experience_sharing_rate"] * 0.1
        if context_features.user_engagement is not None:
            if context_features.user_engagement > 0.6:
                score += params["cross_user_sharing"] * 0.1
    
        if context_features.identity_stability is not None:
            stab   = context_features.identity_stability
            score += (1 - abs(stab - (1 - params["identity_evolution_rate"]))) * 0.1
    
        if context_features.emotional_intensity is not None:
            if context_features.emotional_intensity > 0.7:
                score += (1 - params["risk_tolerance"]) * 0.1
    
        # recency penalty (unchanged)
        recency = 0.0
        for i, item in enumerate(reversed(ctx.context.strategy_history[-5:])):
            if item["strategy_id"] == strategy.id:
                recency += 0.05 * (0.8 ** i)
        score -= min(0.2, recency)
    
        score += random.uniform(-0.05, 0.05)  # exploration
        return max(0.0, min(1.0, score))
    
    @staticmethod    
    @function_tool
    async def _get_strategy(ctx: RunContextWrapper[DynamicAdaptationContext],
                        strategy_id: str) -> Optional[Strategy]:
        """
        Get a specific strategy by ID
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            The strategy or None if not found
        """
        if strategy_id in ctx.context.strategies:
            strategy_data = ctx.context.strategies[strategy_id]
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
                RunContextWrapper(context=self.context), 
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

    @staticmethod
    @function_tool
    async def _update_strategy_history(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        strategy_id: str,
        context_summary: RawContext          # << strict & SDK-friendly
    ) -> bool:
        """
        Append the chosen strategy to history and maintain size limits.
        """
        cdict = context_summary.to_dict()    # ← back to your plain dict
    
        ctx.context.strategy_history.append(
            {
                "timestamp"      : datetime.now().isoformat(),
                "strategy_id"    : strategy_id,
                "context_summary": cdict,
                "cycle"          : ctx.context.cycle_count,
            }
        )
    
        ctx.context.current_strategy_id = strategy_id
    
    @staticmethod
    @function_tool
    async def _calculate_performance_trends(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        metrics: list[MetricKV]          # ✅ strict input
    ) -> list[MetricTrend]:              # ✅ strict output
        """
        Compare the latest performance metrics with history and return a
        per-metric trend analysis.  Logic is unchanged – only marshaling
        to/from the new Pydantic wrappers was added.
        """
    
        # ---------- unwrap the KV list -> dict ----------------------------------
        m_dict: dict[str, float] = {kv.metric: kv.value for kv in metrics}
    
        # keep your existing helper (still takes a plain dict)
        await self._update_performance_history(ctx, m_dict)
    
        history = ctx.context.performance_history
        trends: list[MetricTrend] = []
    
        # not enough data yet -> everything 'stable'
        if len(history) < 2:
            for name, val in m_dict.items():
                trends.append(
                    MetricTrend(metric=name, direction="stable", magnitude=0.0)
                )
            return trends
    
        # -----------------------------------------------------------------------
        # full trend computation (your original code, just tweaked a bit)
        # -----------------------------------------------------------------------
        for name, current in m_dict.items():
            prev_vals = [
                h["metrics"][name]                             # type: ignore[index]
                for h in history[:-1] if name in h["metrics"]  # skip latest point
            ]
    
            if not prev_vals:
                trends.append(
                    MetricTrend(metric=name, direction="stable", magnitude=0.0)
                )
                continue
    
            avg_prev = sum(prev_vals) / len(prev_vals)
            diff     = current - avg_prev
    
            if abs(diff) < 0.05:
                direction = "stable"
                magnitude = 0.0
            else:
                direction = "improving" if diff > 0 else "declining"
                magnitude = min(1.0, abs(diff))
    
            trends.append(
                MetricTrend(
                    metric        = name,
                    direction     = direction,
                    magnitude     = magnitude,
                    diff_from_avg = diff
                )
            )
    
        return trends

    @staticmethod
    @function_tool
    async def _generate_performance_insights(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        metrics: list[MetricKV],                 # strict input ✅
        trends:  list[MetricTrend]               # strict input ✅
    ) -> list[str]:                              # list[str] is already strict
        """
        Produce human-readable insights from the latest metrics & their trends.
        """
    
        # ---------- convert helper objects back to plain dicts ------------------
        m_dict  = {m.metric: m.value for m in metrics}
        t_dict  = {t.metric: {
                        "direction": t.direction,
                        "magnitude": t.magnitude}
                   for t in trends}
    
        insights: list[str] = []
    
        # --- significant improvements / declines --------------------------------
        improvements = [k for k, t in t_dict.items()
                        if t["direction"] == "improving" and t["magnitude"] > 0.1]
        if improvements:
            insights.append(f"Significant improvement in {', '.join(improvements)}")
    
        declines = [k for k, t in t_dict.items()
                    if t["direction"] == "declining" and t["magnitude"] > 0.1]
        if declines:
            insights.append(f"Significant decline in {', '.join(declines)}")
    
        # --- overall performance -------------------------------------------------
        if m_dict:
            avg_perf = sum(m_dict.values()) / len(m_dict)
            if   avg_perf > 0.8: insights.append("Overall performance is excellent")
            elif avg_perf < 0.4: insights.append("Overall performance is concerning")
    
        # --- volatility ----------------------------------------------------------
        try:
            vol = DynamicAdaptationSystem._calculate_performance_volatility(ctx.context)  # type: ignore[arg-type]
            if vol > 0.2:
                insights.append("Performance metrics show high volatility")
        except Exception:
            pass  # volatility helper not available / not critical
    
        # --- experience / identity specific -------------------------------------
        if (val := m_dict.get("experience_utility")) is not None:
            if   val > 0.7: insights.append("Experience sharing is highly effective")
            elif val < 0.3: insights.append("Experience sharing effectiveness is low")
    
        if (val := m_dict.get("identity_coherence")) is not None:
            if   val > 0.7: insights.append("Identity profile shows strong coherence")
            elif val < 0.3: insights.append("Identity profile lacks coherence")
    
        return insights

    @staticmethod
    @function_tool
    async def _calculate_resource_trend(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        values: list[float]                       # already strict ✅
    ) -> ResourceTrend:                           # strict output ✅
        """
        Linear-regression trend detector for any numeric resource.
        """
    
        if len(values) < 2:
            return ResourceTrend(direction="stable", magnitude=0.0)
    
        n        = len(values)
        x        = list(range(n))
        mean_x   = sum(x) / n
        mean_y   = sum(values) / n
        numer    = sum((x[i]-mean_x)*(values[i]-mean_y) for i in range(n))
        denom    = sum((x[i]-mean_x)**2 for i in range(n))
        slope    = 0.0 if denom == 0 else numer / denom
        norm_slp = slope / abs(mean_y) if mean_y else slope
    
        if abs(norm_slp) < 0.05:
            direction = "stable"
        else:
            direction = "increasing" if norm_slp > 0 else "decreasing"
    
        return ResourceTrend(
            direction = direction,
            magnitude = abs(norm_slp),
            slope     = slope,
            mean      = mean_y
        )

    @staticmethod
    @function_tool
    async def _update_performance_history(
        ctx: RunContextWrapper[DynamicAdaptationContext],
        metrics: list[MetricKV]                   # strict input ✅
    ) -> bool:
        """
        Append the current metrics snapshot to history.
        """
    
        # unwrap list → dict
        m_dict = {kv.metric: kv.value for kv in metrics}
    
        ctx.context.performance_history.append({
            "timestamp"   : datetime.now().isoformat(),
            "metrics"     : m_dict,
            "strategy_id" : ctx.context.current_strategy_id,
            "cycle"       : ctx.context.cycle_count
        })
    
        # history length cap
        if len(ctx.context.performance_history) > ctx.context.max_history_size:
            ctx.context.performance_history.pop(0)
    
        return True
    
    def _calculate_performance_volatility(self, context: DynamicAdaptationContext) -> float:
        """
        Calculate the volatility of performance metrics over time
        
        Returns:
            Volatility score (0.0-1.0)
        """
        if len(context.performance_history) < 3:
            return 0.0  # Not enough history
        
        # Extract all metric values
        metric_values = {}
        
        for history_point in context.performance_history:
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
    
    # New function tools for experience-related adaptations

    @staticmethod    
    @function_tool
    async def _analyze_experience_context(ctx: RunContextWrapper[DynamicAdaptationContext],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze experience-related context features
        
        Args:
            context: Context to analyze
            
        Returns:
            Experience context analysis
        """
        experience_analysis = {
            "has_experience": context.get("has_experience", False),
            "cross_user_experience": context.get("cross_user_experience", False),
            "experience_relevance": 0.5  # Default
        }
        
        # Calculate experience relevance if available
        if "relevance_score" in context:
            experience_analysis["experience_relevance"] = min(1.0, context["relevance_score"])
        elif experience_analysis["has_experience"]:
            experience_analysis["experience_relevance"] = 0.7  # Default if has experience but no score
        
        # Calculate experience complexity
        experience_complexity = 0.0
        if experience_analysis["has_experience"]:
            experience_complexity += 0.2
            if experience_analysis["cross_user_experience"]:
                experience_complexity += 0.3
        
        experience_analysis["experience_complexity"] = experience_complexity
        
        # Check for scenario type
        if "scenario_type" in context:
            experience_analysis["scenario_type"] = context["scenario_type"]
        
        # Check for user engagement with experiences
        if "user_engagement" in context:
            experience_analysis["user_engagement"] = context["user_engagement"]
        
        return experience_analysis

    @staticmethod    
    @function_tool
    async def _analyze_identity_context(ctx: RunContextWrapper[DynamicAdaptationContext],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze identity-related context features
        
        Args:
            context: Context to analyze
            
        Returns:
            Identity context analysis
        """
        identity_analysis = {
            "identity_impact": context.get("identity_impact", False),
            "identity_stability": 0.8  # Default high stability
        }
        
        # Adjust stability if identity impact present
        if identity_analysis["identity_impact"]:
            identity_analysis["identity_stability"] = 0.3  # Lower stability when identity impact present
            
            # Check for impact details
            if isinstance(context.get("identity_impact"), dict):
                identity_impact = context["identity_impact"]
                
                # Check preference updates
                if "preferences" in identity_impact:
                    identity_analysis["preference_updates"] = identity_impact["preferences"]
                
                # Check trait updates
                if "traits" in identity_impact:
                    identity_analysis["trait_updates"] = identity_impact["traits"]
                
                # Calculate update magnitude
                update_count = 0
                update_magnitude = 0.0
                
                for category, updates in identity_impact.items():
                    if isinstance(updates, dict):
                        update_count += len(updates)
                        update_magnitude += sum(abs(val) for val in updates.values())
                
                if update_count > 0:
                    identity_analysis["update_magnitude"] = update_magnitude / update_count
        
        return identity_analysis

    @staticmethod    
    @function_tool
    async def _calculate_experience_strategy_fit(ctx: RunContextWrapper[DynamicAdaptationContext],
                                           strategy: Strategy,
                                           context_features: Dict[str, Any]) -> float:
        """
        Calculate how well a strategy fits experience-related context
        
        Args:
            strategy: Strategy to evaluate
            context_features: Experience context features
            
        Returns:
            Strategy fit score (0.0-1.0)
        """
        params = strategy.parameters.model_dump()
        
        # Start with neutral score
        score = 0.5
        
        # Adjust based on experience features
        has_experience = context_features.get("has_experience", False)
        cross_user_experience = context_features.get("cross_user_experience", False)
        experience_relevance = context_features.get("experience_relevance", 0.5)
        
        # If experiences are used and relevant, match with experience_sharing_rate
        if has_experience:
            experience_match = 1.0 - abs(experience_relevance - params["experience_sharing_rate"])
            score += experience_match * 0.2
            
            # If cross-user experiences are used, match with cross_user_sharing
            if cross_user_experience:
                cross_user_match = 1.0 - abs(0.8 - params["cross_user_sharing"])  # Prefer high cross-user sharing
                score += cross_user_match * 0.2
            else:
                # If no cross-user but has experience, prefer moderate cross-user sharing
                cross_user_match = 1.0 - abs(0.3 - params["cross_user_sharing"])
                score += cross_user_match * 0.1
        else:
            # If no experiences are used, prefer lower experience sharing rates
            experience_match = 1.0 - abs(0.2 - params["experience_sharing_rate"])
            score += experience_match * 0.1
        
        # Adjust for user engagement if available
        if "user_engagement" in context_features:
            engagement = context_features["user_engagement"]
            
            # Higher engagement favors higher experience sharing
            if engagement > 0.6:
                score += params["experience_sharing_rate"] * 0.15
                score += params["cross_user_sharing"] * 0.1
            else:
                # Lower engagement favors lower experience sharing
                score += (1.0 - params["experience_sharing_rate"]) * 0.1
        
        # Ensure score is in valid range
        return min(1.0, max(0.0, score))

    @staticmethod    
    @function_tool
    async def _calculate_identity_strategy_fit(ctx: RunContextWrapper[DynamicAdaptationContext],
                                         strategy: Strategy,
                                         context_features: Dict[str, Any]) -> float:
        """
        Calculate how well a strategy fits identity-related context
        
        Args:
            strategy: Strategy to evaluate
            context_features: Identity context features
            
        Returns:
            Strategy fit score (0.0-1.0)
        """
        params = strategy.parameters.model_dump()
        
        # Start with neutral score
        score = 0.5
        
        # Adjust based on identity features
        identity_impact = context_features.get("identity_impact", False)
        identity_stability = context_features.get("identity_stability", 0.8)
        
        # If identity impact present, match with identity_evolution_rate
        if identity_impact:
            if "update_magnitude" in context_features:
                # Match evolution rate with update magnitude
                update_magnitude = context_features["update_magnitude"]
                evolution_match = 1.0 - abs(update_magnitude - params["identity_evolution_rate"])
                score += evolution_match * 0.2
            else:
                # Default to preferring higher evolution rate when impact present
                score += params["identity_evolution_rate"] * 0.15
        else:
            # If no identity impact, prefer lower evolution rate
            evolution_match = 1.0 - params["identity_evolution_rate"]
            score += evolution_match * 0.1
        
        # Match identity stability with evolution rate (inverse relationship)
        stability_match = 1.0 - abs(identity_stability - (1.0 - params["identity_evolution_rate"]))
        score += stability_match * 0.2
        
        # Match with adaptation rate
        adaptation_match = 1.0 - abs((1.0 - identity_stability) - params["adaptation_rate"])
        score += adaptation_match * 0.1
        
        # Ensure score is in valid range
        return min(1.0, max(0.0, score))

    @staticmethod    
    @function_tool
    async def _analyze_experience_performance(ctx: RunContextWrapper[DynamicAdaptationContext],
                                         metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze experience-related performance metrics
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Experience performance analysis
        """
        experience_analysis = {
            "experience_effectiveness": 0.5  # Default
        }
        
        # Extract experience-related metrics
        if "experience_utility" in metrics:
            experience_analysis["experience_utility"] = metrics["experience_utility"]
            
            # Categorize utility
            if metrics["experience_utility"] > 0.7:
                experience_analysis["utility_category"] = "high"
            elif metrics["experience_utility"] < 0.3:
                experience_analysis["utility_category"] = "low"
            else:
                experience_analysis["utility_category"] = "moderate"
        
        # Calculate overall effectiveness
        effectiveness_factors = []
        
        if "experience_utility" in metrics:
            effectiveness_factors.append(metrics["experience_utility"])
        
        if "user_satisfaction" in metrics:
            effectiveness_factors.append(metrics["user_satisfaction"])
            experience_analysis["user_satisfaction"] = metrics["user_satisfaction"]
        
        if effectiveness_factors:
            experience_analysis["experience_effectiveness"] = sum(effectiveness_factors) / len(effectiveness_factors)
        
        # Calculate trend if we have history
        experience_metrics_history = []
        for history_point in ctx.context.performance_history:
            if "experience_utility" in history_point["metrics"]:
                experience_metrics_history.append(history_point["metrics"]["experience_utility"])
        
        if len(experience_metrics_history) >= 3:
            # Calculate trend
            recent_avg = sum(experience_metrics_history[-3:]) / 3
            older_avg = sum(experience_metrics_history[:-3]) / max(1, len(experience_metrics_history) - 3)
            
            trend = recent_avg - older_avg
            
            if abs(trend) < 0.05:
                experience_analysis["trend"] = "stable"
            elif trend > 0:
                experience_analysis["trend"] = "improving"
            else:
                experience_analysis["trend"] = "declining"
            
            experience_analysis["trend_magnitude"] = abs(trend)
        
        return experience_analysis

    @staticmethod    
    @function_tool
    async def _analyze_identity_performance(ctx: RunContextWrapper[DynamicAdaptationContext],
                                       metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze identity-related performance metrics
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Identity performance analysis
        """
        identity_analysis = {
            "identity_coherence": 0.5  # Default
        }
        
        # Extract identity-related metrics
        if "identity_coherence" in metrics:
            identity_analysis["identity_coherence"] = metrics["identity_coherence"]
            
            # Categorize coherence
            if metrics["identity_coherence"] > 0.7:
                identity_analysis["coherence_category"] = "high"
            elif metrics["identity_coherence"] < 0.3:
                identity_analysis["coherence_category"] = "low"
            else:
                identity_analysis["coherence_category"] = "moderate"
        
        # Check for other identity metrics
        identity_metrics = {}
        for key, value in metrics.items():
            if key.startswith("identity_"):
                identity_metrics[key] = value
                identity_analysis[key] = value
        
        # Calculate overall stability
        if identity_metrics:
            identity_analysis["overall_stability"] = sum(identity_metrics.values()) / len(identity_metrics)
        
        # Calculate trend if we have history
        identity_metrics_history = []
        for history_point in ctx.context.performance_history:
            if "identity_coherence" in history_point["metrics"]:
                identity_metrics_history.append(history_point["metrics"]["identity_coherence"])
        
        if len(identity_metrics_history) >= 3:
            # Calculate trend
            recent_avg = sum(identity_metrics_history[-3:]) / 3
            older_avg = sum(identity_metrics_history[:-3]) / max(1, len(identity_metrics_history) - 3)
            
            trend = recent_avg - older_avg
            
            if abs(trend) < 0.05:
                identity_analysis["trend"] = "stable"
            elif trend > 0:
                identity_analysis["trend"] = "improving"
            else:
                identity_analysis["trend"] = "declining"
            
            identity_analysis["trend_magnitude"] = abs(trend)
        
        return identity_analysis
    
    # Tools for experience adaptation agent

    @staticmethod    
    @function_tool
    async def _get_current_experience_params(ctx: RunContextWrapper[DynamicAdaptationContext]) -> Dict[str, Any]:
        """
        Get current experience adaptation parameters
        
        Returns:
            Current experience adaptation parameters
        """
        return ctx.context.experience_adaptation.model_dump()

    @staticmethod    
    @function_tool
    async def _calculate_experience_adaptation(ctx: RunContextWrapper[DynamicAdaptationContext],
                                         feedback: Dict[str, Any], 
                                         current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate adjusted experience adaptation parameters based on feedback
        
        Args:
            feedback: User feedback information
            current_params: Current adaptation parameters
            
        Returns:
            Adjusted parameters
        """
        # Start with current parameters
        params = current_params.copy()
        
        # Extract relevant feedback
        experience_rating = feedback.get("experience_rating", None)
        cross_user_rating = feedback.get("cross_user_rating", None)
        scenario_ratings = feedback.get("scenario_ratings", {})
        
        # Update cross_user_enabled based on explicit feedback
        if cross_user_rating is not None:
            # Convert rating to boolean (threshold at 5 on 0-10 scale)
            params["cross_user_enabled"] = cross_user_rating >= 5
        
        # Update sharing threshold based on experience rating
        if experience_rating is not None:
            # Convert 0-10 rating to threshold (inverse relationship)
            # Higher rating = lower threshold = more sharing
            params["sharing_threshold"] = max(0.5, 1.0 - (experience_rating / 10.0))
        
        # Update experience types based on scenario ratings
        if scenario_ratings:
            # Sort scenario types by rating
            sorted_scenarios = sorted(
                scenario_ratings.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top rated scenarios (rating > 5)
            top_scenarios = [s for s, r in sorted_scenarios if r > 5]
            
            if top_scenarios:
                params["experience_types"] = top_scenarios
        
        # Update personalization level based on overall ratings
        if experience_rating is not None:
            params["personalization_level"] = experience_rating / 10.0
        
        return params

    @staticmethod    
    @function_tool
    async def _update_experience_sharing_history(ctx: RunContextWrapper[DynamicAdaptationContext],
                                           user_id: str, 
                                           feedback: Dict[str, Any], 
                                           adapted_params: Dict[str, Any]) -> bool:
        """
        Update experience sharing history with adaptation information
        
        Args:
            user_id: User ID
            feedback: User feedback
            adapted_params: Adapted parameters
            
        Returns:
            Success status
        """
        # Record adaptation in history
        ctx.context.experience_sharing_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "feedback": feedback,
            "adapted_params": adapted_params,
            "strategy_id": ctx.context.current_strategy_id
        })
        
        # Limit history size
        if len(ctx.context.experience_sharing_history) > ctx.context.max_history_size:
            ctx.context.experience_sharing_history = ctx.context.experience_sharing_history[-ctx.context.max_history_size:]
        
        return True
    
    # Tools for identity adaptation agent

    @staticmethod    
    @function_tool
    async def _get_current_identity_params(ctx: RunContextWrapper[DynamicAdaptationContext]) -> Dict[str, Any]:
        """
        Get current identity adaptation parameters
        
        Returns:
            Current identity adaptation parameters
        """
        return ctx.context.identity_adaptation.model_dump()

    @staticmethod  
    @function_tool
    async def _calculate_identity_adaptation(ctx: RunContextWrapper[DynamicAdaptationContext],
                                       identity_state: Dict[str, Any], 
                                       performance: Dict[str, Any],
                                       current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate adjusted identity adaptation parameters based on state and performance
        
        Args:
            identity_state: Current identity state
            performance: Performance metrics
            current_params: Current adaptation parameters
            
        Returns:
            Adjusted parameters
        """
        # Start with current parameters
        params = current_params.copy()
        
        # Extract relevant metrics
        identity_coherence = performance.get("identity_coherence", None)
        user_satisfaction = performance.get("user_satisfaction", None)
        
        # Extract identity state information
        recent_changes = identity_state.get("identity_evolution", {}).get("recent_significant_changes", {})
        total_updates = identity_state.get("identity_evolution", {}).get("total_updates", 0)
        
        # Adjust evolution rate based on coherence and satisfaction
        if identity_coherence is not None:
            # Lower coherence = lower evolution rate
            if identity_coherence < 0.3:
                params["evolution_rate"] = max(0.1, params["evolution_rate"] - 0.1)
            # Higher coherence = can increase evolution rate if satisfaction is good
            elif identity_coherence > 0.7 and user_satisfaction is not None and user_satisfaction > 0.7:
                params["evolution_rate"] = min(0.6, params["evolution_rate"] + 0.05)
        
        # Adjust trait stability based on recent changes
        if recent_changes:
            # Many recent changes = increase stability
            if len(recent_changes) > 3:
                params["trait_stability"] = min(0.9, params["trait_stability"] + 0.1)
            # Few significant changes = can decrease stability
            elif len(recent_changes) < 2 and total_updates > 10:
                params["trait_stability"] = max(0.3, params["trait_stability"] - 0.05)
        
        # Adjust preference adaptability based on user satisfaction
        if user_satisfaction is not None:
            # Higher satisfaction = can increase adaptability
            if user_satisfaction > 0.7:
                params["preference_adaptability"] = min(0.8, params["preference_adaptability"] + 0.05)
            # Lower satisfaction = reduce adaptability
            elif user_satisfaction < 0.3:
                params["preference_adaptability"] = max(0.2, params["preference_adaptability"] - 0.1)
        
        # Adjust consolidation frequency based on total updates and coherence
        if total_updates > 20 and identity_coherence is not None:
            if identity_coherence < 0.5:
                # Lower coherence with many updates = increase consolidation
                params["consolidation_frequency"] = min(0.8, params["consolidation_frequency"] + 0.1)
            else:
                # Good coherence with many updates = current frequency is working
                pass
        elif total_updates < 5:
            # Few updates = reduce consolidation frequency
            params["consolidation_frequency"] = max(0.1, params["consolidation_frequency"] - 0.05)
        
        return params

    @staticmethod  
    @function_tool
    async def _update_identity_evolution_history(ctx: RunContextWrapper[DynamicAdaptationContext],
                                          identity_state: Dict[str, Any], 
                                          adapted_params: Dict[str, Any]) -> bool:
        """
        Update identity evolution history with adaptation information
        
        Args:
            identity_state: Current identity state
            adapted_params: Adapted parameters
            
        Returns:
            Success status
        """
        # Record adaptation in history
        ctx.context.identity_evolution_history.append({
            "timestamp": datetime.now().isoformat(),
            "identity_state": identity_state,
            "adapted_params": adapted_params,
            "strategy_id": ctx.context.current_strategy_id
        })
        
        # Limit history size
        if len(ctx.context.identity_evolution_history) > ctx.context.max_history_size:
            ctx.context.identity_evolution_history = ctx.context.identity_evolution_history[-ctx.context.max_history_size:]
        
        return True
    
    # Public tools for external use

    @staticmethod  
    @function_tool
    async def _get_experience_adaptation_settings(ctx: RunContextWrapper[DynamicAdaptationContext]) -> Dict[str, Any]:
        """
        Get current experience adaptation settings based on active strategy
        
        Returns:
            Experience adaptation settings
        """
        # Get strategy parameters
        strategy_id = ctx.context.current_strategy_id
        strategy_data = ctx.context.strategies.get(strategy_id, {})
        strategy_params = strategy_data.get("parameters", {})
        
        # Create settings dictionary
        settings = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data.get("name", "Unknown"),
            "cross_user_enabled": strategy_params.get("cross_user_sharing", 0.3) > 0.3,
            "sharing_threshold": max(0.5, 1.0 - strategy_params.get("experience_sharing_rate", 0.5)),
            "experience_sharing_rate": strategy_params.get("experience_sharing_rate", 0.5),
            "personalization_level": strategy_params.get("experience_sharing_rate", 0.5) * 0.8
        }
        
        # Add specific experience adaptation settings
        settings.update(ctx.context.experience_adaptation.model_dump())
        
        return settings

    @staticmethod  
    @function_tool
    async def _get_identity_adaptation_settings(ctx: RunContextWrapper[DynamicAdaptationContext]) -> Dict[str, Any]:
        """
        Get current identity adaptation settings based on active strategy
        
        Returns:
            Identity adaptation settings
        """
        # Get strategy parameters
        strategy_id = ctx.context.current_strategy_id
        strategy_data = ctx.context.strategies.get(strategy_id, {})
        strategy_params = strategy_data.get("parameters", {})
        
        # Create settings dictionary
        settings = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data.get("name", "Unknown"),
            "identity_evolution_rate": strategy_params.get("identity_evolution_rate", 0.2),
            "adaptation_rate": strategy_params.get("adaptation_rate", 0.15)
        }
        
        # Add specific identity adaptation settings
        settings.update(ctx.context.identity_adaptation.model_dump())
        
        return settings
