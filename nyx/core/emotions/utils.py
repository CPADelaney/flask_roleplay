# nyx/core/emotions/utils.py

"""
Enhanced utility functions for the Nyx emotional system.
Provides improved error handling, tracing, and run configuration utilities.
"""

import functools
import logging
import datetime
import json
from typing import Callable, Any, Dict, Optional, TypeVar, List, Awaitable, Type, Union, Literal, cast

from agents import (
    RunConfig, ModelSettings, InputGuardrail, OutputGuardrail, 
    trace, custom_span, gen_trace_id, function_span, Agent
)
from agents.exceptions import UserError, AgentsException, ModelBehaviorError
from agents.tracing import Trace, Span

logger = logging.getLogger(__name__)

# Define type variables for better typing
T = TypeVar('T')
TFunc = TypeVar('TFunc', bound=Callable[..., Any])
TContext = TypeVar('TContext')
TReturn = TypeVar('TReturn')

def handle_errors(logger_message: str = "An error occurred", 
                 log_level: Literal["debug", "info", "warning", "error"] = "error") -> Callable[[TFunc], TFunc]:
    """
    Enhanced decorator for consistent error handling across emotional system functions
    
    Args:
        logger_message: Custom error message prefix for logging
        log_level: Logging level to use for errors
        
    Returns:
        Decorated function with consistent error handling
    """
    def decorator(func: TFunc) -> TFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except UserError as e:
                # Handle user errors (invalid inputs, etc.)
                getattr(logger, log_level)(f"{logger_message}: {e}")
                return {
                    "success": False, 
                    "error": str(e), 
                    "error_type": "user_error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except ModelBehaviorError as e:
                # Handle model-specific errors
                getattr(logger, "warning")(f"{logger_message} (Model behavior): {e}")
                return {
                    "success": False, 
                    "error": str(e), 
                    "error_type": "model_behavior_error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except AgentsException as e:
                # Handle Agent SDK errors
                getattr(logger, "error")(f"{logger_message} (Agent SDK): {e}")
                return {
                    "success": False, 
                    "error": str(e), 
                    "error_type": "agent_error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except Exception as e:
                # Handle unexpected errors
                getattr(logger, "error")(f"{logger_message} (Unexpected): {e}", exc_info=True)
                return {
                    "success": False, 
                    "error": "An unexpected error occurred", 
                    "error_type": "system_error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        return cast(TFunc, wrapper)
    return decorator

def create_run_config(
    workflow_name: Optional[str] = None,
    trace_id: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 300,
    include_sensitive_data: bool = True,
    cycle_count: int = 0,
    model: str = "o3-mini",
    input_guardrails: Optional[List[InputGuardrail[Any]]] = None,
    output_guardrails: Optional[List[OutputGuardrail[Any]]] = None,
    context_data: Optional[Dict[str, Any]] = None
) -> RunConfig:
    """
    Enhanced factory function for creating standardized run configurations
    
    Args:
        workflow_name: Name of the workflow
        trace_id: Unique trace identifier
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        include_sensitive_data: Whether to include sensitive data in traces
        cycle_count: Current processing cycle count
        model: Model to use for inference
        input_guardrails: Optional list of input guardrails
        output_guardrails: Optional list of output guardrails
        context_data: Additional context metadata
        
    Returns:
        Run configuration for the Agent SDK
    """
    from agents.extensions import handoff_filters
    
    # Generate a proper trace ID if not provided
    if not trace_id:
        trace_id = gen_trace_id()
    elif not trace_id.startswith("trace_"):
        trace_id = f"trace_{trace_id}"
    
    metadata = {
        "system": "nyx_emotional_core",
        "version": "1.0",
        "cycle": cycle_count,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add any additional context data if provided
    if context_data:
        metadata.update(context_data)
    
    return RunConfig(
        workflow_name=workflow_name or f"Emotion_{cycle_count}",
        trace_id=trace_id,
        model=model,
        model_settings=ModelSettings(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        ),
        handoff_input_filter=handoff_filters.keep_relevant_history,
        trace_include_sensitive_data=include_sensitive_data,
        trace_metadata=metadata,
        input_guardrails=input_guardrails or [],
        output_guardrails=output_guardrails or []
    )

def with_emotion_trace(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Enhanced decorator to add tracing to emotional methods with improved metadata
    
    Args:
        func: The function to wrap with tracing
        
    Returns:
        Wrapped function with tracing
    """
    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        # Create a workflow name from the function name
        workflow_name = f"Emotion_{func.__name__}"
        
        # Generate a unique trace ID
        trace_id = gen_trace_id()
        
        # Add useful metadata
        metadata = {
            "function": func.__name__,
            "cycle_count": getattr(self, "context", {}).cycle_count if hasattr(self, "context") else 0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add dominant emotion information if available
        if hasattr(self, "get_dominant_emotion"):
            try:
                dominant_emotion, intensity = self.get_dominant_emotion()
                metadata["current_emotion"] = dominant_emotion
                metadata["intensity"] = intensity
            except:
                pass
        
        # Add performance metrics if available
        if hasattr(self, "performance_metrics"):
            try:
                metrics = {
                    k: v for k, v in self.performance_metrics.items() 
                    if k in ["api_calls", "average_response_time"]
                }
                metadata["performance_metrics"] = metrics
            except:
                pass
        
        # Get conversation ID from context if available
        conversation_id = "default"
        if hasattr(self, "context") and hasattr(self.context, "temp_data"):
            conversation_id = self.context.temp_data.get("conversation_id", "default")
        
        # Create the trace
        with trace(
            workflow_name=workflow_name, 
            trace_id=trace_id,
            group_id=f"conversation_{conversation_id}",
            metadata=metadata
        ):
            # Create a span specifically for this function
            with function_span(func.__name__, input=str(args)[:100]):
                return await func(self, *args, **kwargs)
    
    return wrapper

class EmotionalToolUtils:
    """Static utility functions for emotional tools"""
    
    @staticmethod
    def normalize_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a value to the given range
        
        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Normalized value
        """
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
        """
        Calculate weighted average
        
        Args:
            values: List of values
            weights: List of weights
            
        Returns:
            Weighted average
        """
        if not values or not weights or len(values) != len(weights):
            return 0.0
            
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight
    
    @staticmethod
    def format_emotional_data(data: Dict[str, Any]) -> str:
        """
        Format emotional data for display
        
        Args:
            data: Emotional data to format
            
        Returns:
            Formatted string
        """
        formatted = []
        
        if "primary_emotion" in data:
            primary = data["primary_emotion"]
            if isinstance(primary, dict):
                formatted.append(f"Primary: {primary.get('name', 'Unknown')} ({primary.get('intensity', 0):.2f})")
            else:
                formatted.append(f"Primary: {primary}")
        
        if "valence" in data:
            formatted.append(f"Valence: {data['valence']:.2f}")
            
        if "arousal" in data:
            formatted.append(f"Arousal: {data['arousal']:.2f}")
            
        if "chemicals" in data:
            chemicals = data["chemicals"]
            if isinstance(chemicals, dict):
                chem_str = ", ".join(f"{c}: {v:.2f}" for c, v in chemicals.items())
                formatted.append(f"Chemicals: {chem_str}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def log_and_create_span(name: str, data: Dict[str, Any]) -> Span:
        """
        Log an event and create a span for it
        
        Args:
            name: Span name
            data: Span data
            
        Returns:
            Created span
        """
        logger.debug(f"{name}: {json.dumps(data)}")
        return custom_span(name, data=data)
    
    @staticmethod
    def agent_config_from_state(
        agent_type: str, 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create agent configuration parameters from emotional state
        
        Args:
            agent_type: Type of agent to configure
            state: Current emotional state
            
        Returns:
            Agent configuration dictionary
        """
        config = {
            "temperature": 0.4,
            "context": {}
        }
        
        # Add primary emotion to context if available
        if "primary_emotion" in state:
            if isinstance(state["primary_emotion"], dict):
                primary = state["primary_emotion"]
                config["context"]["primary_emotion"] = primary.get("name", "Neutral")
                config["context"]["intensity"] = primary.get("intensity", 0.5)
                
                # Adjust temperature based on emotion
                if primary.get("name") in ["Joy", "Excitement", "Anticipation"]:
                    config["temperature"] = 0.6  # More creative for positive emotions
                elif primary.get("name") in ["Sadness", "Fear", "Anger"]:
                    config["temperature"] = 0.3  # More conservative for negative emotions
            else:
                config["context"]["primary_emotion"] = state["primary_emotion"]
        
        # Add valence and arousal
        if "valence" in state:
            config["context"]["valence"] = state["valence"]
        
        if "arousal" in state:
            config["context"]["arousal"] = state["arousal"]
            
            # Adjust max tokens based on arousal
            arousal = state.get("arousal", 0.5)
            if arousal > 0.7:
                config["max_tokens"] = 400  # More verbose when aroused
            elif arousal < 0.3:
                config["max_tokens"] = 200  # More concise when calm
        
        return config
