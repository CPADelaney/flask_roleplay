# nyx/core/emotions/utils.py

"""
Enhanced utility functions for the Nyx emotional system.
Provides improved error handling, tracing, and run configuration utilities
with better OpenAI Agents SDK integration.
"""

import functools
import logging
import datetime
import json
import inspect
from typing import (
    Callable, Any, Dict, Optional, TypeVar, List, Awaitable, Type, 
    Union, Literal, cast, Tuple, Set, Generic, overload
)

from agents import (
    RunConfig, ModelSettings, InputGuardrail, OutputGuardrail, 
    trace, custom_span, agent_span, gen_span_id, gen_trace_id, function_span, Agent, 
    Model, ModelProvider, OpenAIProvider, RunContextWrapper
)
from agents.exceptions import UserError, AgentsException, ModelBehaviorError
from agents.tracing import Trace, Span, gen_span_id, add_trace_processor
from agents.extensions.handoff_filters import remove_all_tools

from nyx.core.emotions.context import EmotionalContext

logger = logging.getLogger(__name__)

# Define type variables for better typing
T = TypeVar('T')
TFunc = TypeVar('TFunc', bound=Callable[..., Any])
TContext = TypeVar('TContext', bound=EmotionalContext)
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
                # Get function signature for better error context
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Extract context from parameters if available (for better error tracking)
                ctx = None
                if len(args) > 0 and param_names and param_names[0] == "ctx":
                    ctx = args[0]
                
                # Track start time for performance monitoring
                start_time = datetime.datetime.now()
                
                # Execute the function with tracing
                with function_span(func.__name__, input=str(args)[:100] if args else "{}"):
                    result = func(*args, **kwargs)
                    # Ensure result is awaitable
                    result = ensure_awaitable(result)
                    result = await result
                
                # Track execution time
                duration = (datetime.datetime.now() - start_time).total_seconds()
                
                # Log success if needed
                if log_level == "debug":
                    getattr(logger, "debug")(f"{func.__name__} completed in {duration:.3f}s")
                
                # Record timing information in context if available
                if ctx and hasattr(ctx, "context") and hasattr(ctx.context, "get_value"):
                    # Store function timing in context
                    function_timing = ctx.context.get_value("function_timing", {})
                    
                    if func.__name__ not in function_timing:
                        function_timing[func.__name__] = {"count": 0, "total_time": 0}
                    
                    function_timing[func.__name__]["count"] += 1
                    function_timing[func.__name__]["total_time"] += duration
                    function_timing[func.__name__]["avg_time"] = (
                        function_timing[func.__name__]["total_time"] / 
                        function_timing[func.__name__]["count"]
                    )
                    
                    ctx.context.set_value("function_timing", function_timing)
                
                return result
                
            except UserError as e:
                # Handle user errors (invalid inputs, etc.)
                getattr(logger, log_level)(f"{logger_message}: {e}")
                
                # Create a custom error span for better tracing
                with custom_span(
                    "error", 
                    data={
                        "type": "user_error",
                        "message": str(e),
                        "function": func.__name__,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    # Return structured error response
                    return {
                        "success": False, 
                        "error": str(e), 
                        "error_type": "user_error",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
            except ModelBehaviorError as e:
                # Handle model-specific errors
                getattr(logger, "warning")(f"{logger_message} (Model behavior): {e}")
                
                # Create a custom error span for better tracing
                with custom_span(
                    "error", 
                    data={
                        "type": "model_behavior_error",
                        "message": str(e),
                        "function": func.__name__,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    # Return structured error response
                    return {
                        "success": False, 
                        "error": str(e), 
                        "error_type": "model_behavior_error",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
            except AgentsException as e:
                # Handle Agent SDK errors
                getattr(logger, "error")(f"{logger_message} (Agent SDK): {e}")
                
                # Create a custom error span for better tracing
                with custom_span(
                    "error", 
                    data={
                        "type": "agent_error",
                        "message": str(e),
                        "function": func.__name__,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    # Return structured error response
                    return {
                        "success": False, 
                        "error": str(e), 
                        "error_type": "agent_error",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
            except Exception as e:
                # Handle unexpected errors
                getattr(logger, "error")(f"{logger_message} (Unexpected): {e}", exc_info=True)
                
                # Create a custom error span for better tracing
                with custom_span(
                    "error", 
                    data={
                        "type": "system_error",
                        "message": str(e),
                        "function": func.__name__,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                ):
                    # Return structured error response for unexpected errors
                    return {
                        "success": False, 
                        "error": "An unexpected error occurred", 
                        "error_type": "system_error",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "function": func.__name__
                    }
                    
        return cast(TFunc, wrapper)
    return decorator

@overload
def create_run_config(
    workflow_name: Optional[str] = None,
    trace_id: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 300,
    include_sensitive_data: bool = True,
    cycle_count: int = 0,
    model: str = "gpt-5-nano",
    model_provider: Optional[ModelProvider] = None,
    input_guardrails: Optional[List[InputGuardrail[Any]]] = None,
    output_guardrails: Optional[List[OutputGuardrail[Any]]] = None,
    context_data: Optional[Dict[str, Any]] = None,
    handoff_input_filter: Optional[Callable] = None,
    group_id: Optional[str] = None
) -> RunConfig:
    ...

def create_run_config(
    workflow_name: str,
    cycle_count: int,
    conversation_id: Optional[str] = None,
    input_text_length: Optional[int] = None,
    pattern_analysis: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 300,
    trace_id: Optional[str] = None,
) -> RunConfig:
    """
    SDK-optimized factory for creating standardized run configurations
    
    Args:
        workflow_name: Name of the workflow
        cycle_count: Current processing cycle count
        conversation_id: Conversation identifier for grouping traces
        input_text_length: Length of input text if available
        pattern_analysis: Pattern analysis result if available
        model: Model to use for inference (optional override)
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        trace_id: Optional custom trace ID
        
    Returns:
        Optimized run configuration
    """
    # Generate a proper trace ID using SDK function if not provided
    if not trace_id:
        trace_id = gen_trace_id()
    elif not trace_id.startswith("trace_"):
        trace_id = f"trace_{trace_id}"
    
    # Build metadata dictionary with standard fields
    metadata = {
        "system": "nyx_emotional_core",
        "version": "1.0",
        "cycle": cycle_count,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add optional metadata
    if input_text_length is not None:
        metadata["input_text_length"] = input_text_length
    
    if pattern_analysis:
        metadata["pattern_analysis"] = pattern_analysis
        
    if conversation_id:
        metadata["conversation_id"] = conversation_id
    
    # Create run configuration with SDK features
    return RunConfig(
        workflow_name=workflow_name,
        trace_id=trace_id,
        group_id=conversation_id,
        model=model,
        model_provider=OpenAIProvider(),
        model_settings=ModelSettings(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        ),
        handoff_input_filter=remove_all_tools,
        trace_include_sensitive_data=True,
        trace_metadata=metadata
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
        
        # Extract conversation ID from context or kwargs if available
        context = getattr(self, "context", None)
        conversation_id = kwargs.get("conversation_id")
        
        if not conversation_id and context:
            # Try to get conversation ID from context
            if hasattr(context, "get_value"):
                conversation_id = context.get_value("conversation_id")
            elif hasattr(context, "temp_data"):
                conversation_id = context.temp_data.get("conversation_id")
        
        if not conversation_id:
            # Generate a new conversation ID if none exists
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            
            # Store in context if possible
            if context:
                if hasattr(context, "set_value"):
                    context.set_value("conversation_id", conversation_id)
                elif hasattr(context, "temp_data"):
                    context.temp_data["conversation_id"] = conversation_id
        
        # Add useful metadata
        metadata = {
            "function": func.__name__,
            "cycle_count": getattr(self, "context", {}).cycle_count if hasattr(self, "context") else 0,
            "timestamp": datetime.datetime.now().isoformat(),
            "conversation_id": conversation_id
        }
        
        # Add dominant emotion information if available
        if hasattr(self, "get_dominant_emotion"):
            try:
                dominant_emotion, intensity = self.get_dominant_emotion()
                metadata["current_emotion"] = dominant_emotion
                metadata["intensity"] = intensity
            except:
                pass
        
        # Create the trace with SDK features
        with trace(
            workflow_name=workflow_name, 
            trace_id=trace_id,
            group_id=conversation_id,
            metadata=metadata
        ):
            # Create a span specifically for this function
            function_id = gen_span_id()
            with function_span(
                func.__name__, 
                input=str(args)[:100] if args else "{}", 
                span_id=function_id
            ):
                # Execute the function and return its result
                result = await func(self, *args, **kwargs)
                
                # Add result metadata to the function span
                if isinstance(result, dict) and "success" in result:
                    # For error results or status reports
                    with custom_span(
                        "function_result",
                        data={
                            "function": func.__name__,
                            "success": result.get("success", False),
                            "duration": self.context.get_elapsed_time(
                                f"start_{func.__name__}", 
                                f"end_{func.__name__}"
                            ) if hasattr(self, "context") else 0
                        }
                    ):
                        pass
                
                return result
    
    return wrapper

class EmotionalToolUtils:
    """Static utility functions for emotional tools with improved SDK integration"""
    
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

    @staticmethod
    def create_trace_metadata(context: Any) -> Dict[str, Any]:
        """
        Create metadata dictionary for traces from context
        
        Args:
            context: Context object
            
        Returns:
            Metadata dictionary for tracing
        """
        metadata = {
            "system": "nyx_emotional_core",
            "version": "1.0",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add context data if available
        if hasattr(context, "cycle_count"):
            metadata["cycle"] = context.cycle_count
            
        if hasattr(context, "create_trace_metadata"):
            # Use context's built-in method if available
            return context.create_trace_metadata()
            
        # Try to extract information from context manually
        if hasattr(context, "last_emotions") and context.last_emotions:
            # Find dominant emotion
            try:
                dominant = max(context.last_emotions.items(), key=lambda x: x[1])
                metadata["dominant_emotion"] = dominant[0]
                metadata["intensity"] = dominant[1]
            except:
                pass
        
        # Add conversation ID if available
        if hasattr(context, "get_value"):
            conversation_id = context.get_value("conversation_id")
            if conversation_id:
                metadata["conversation_id"] = conversation_id
        elif hasattr(context, "temp_data"):
            conversation_id = context.temp_data.get("conversation_id")
            if conversation_id:
                metadata["conversation_id"] = conversation_id
                
        return metadata
    
    @staticmethod
    def configure_trace_processor():
        """
        Configure a custom trace processor for emotional analysis
        """
        from agents.tracing import add_trace_processor, BatchTraceProcessor
        from agents.tracing.processors import BackendSpanExporter
        from collections import defaultdict
        
        # Define a custom trace processor that can generate emotional analytics
        class EmotionalAnalyticsProcessor(BatchTraceProcessor):
            """Custom processor that analyzes emotional patterns in traces"""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.emotion_transitions = defaultdict(int)
                self.chemical_patterns = defaultdict(list)
            
            def on_span_end(self, span):
                """Process span data for emotional analytics"""
                super().on_span_end(span)
                
                # Track emotion transitions
                if hasattr(span, "data") and span.data.get("type") == "emotion_transition":
                    from_emotion = span.data.get("from_emotion", "unknown")
                    to_emotion = span.data.get("to_emotion", "unknown")
                    self.emotion_transitions[(from_emotion, to_emotion)] += 1
                
                # Track chemical patterns
                if hasattr(span, "data") and span.data.get("type") == "chemical_update":
                    chemical = span.data.get("chemical")
                    value = span.data.get("value")
                    if chemical and value is not None:
                        self.chemical_patterns[chemical].append(value)
        
        # Create and add the custom processor
        emotion_trace_processor = EmotionalAnalyticsProcessor(
            exporter=BackendSpanExporter(project="nyx_emotional_system"),
            max_batch_size=100,
            schedule_delay=3.0
        )
        add_trace_processor(emotion_trace_processor)
    # New utility functions to add to utils.py
    
    def create_standard_span(
        span_type: str,
        data: Dict[str, Any],
        ctx: Optional[RunContextWrapper[EmotionalContext]] = None
    ) -> Span:
        """
        Create a standardized span with consistent metadata
        
        Args:
            span_type: Type of span to create
            data: Span data
            ctx: Optional context wrapper
            
        Returns:
            Created span
        """
        # Add standard metadata
        full_data = {
            "type": span_type,
            "timestamp": datetime.datetime.now().isoformat(),
            **data
        }
        
        # Add context data if available
        if ctx is not None:
            full_data["cycle"] = ctx.context.cycle_count if hasattr(ctx.context, "cycle_count") else 0
            
            # Add metadata from context
            if hasattr(ctx.context, "create_trace_metadata"):
                trace_metadata = ctx.context.create_trace_metadata()
                if "dominant_emotion" in trace_metadata:
                    full_data["dominant_emotion"] = trace_metadata["dominant_emotion"]
        
        # Create and return span
        return custom_span(span_type, data=full_data)
    
    def check_sdk_compatibility() -> bool:
        """
        Check compatibility with the OpenAI Agents SDK version
        
        Returns:
            True if compatible, False otherwise
        """
        try:
            # The SDK doesn't currently provide a version, so we check for required components
            from agents import (
                Agent, Runner, function_tool, handoff, trace,
                ModelSettings, RunConfig
            )
            
            # Check for specific newer SDK features
            features_present = {
                "agent": hasattr(Agent, "clone"),
                "runner": hasattr(Runner, "run_streamed"),
                "function_tool": callable(function_tool),
                "handoff": callable(handoff),
                "trace": callable(trace)
            }
            
            # Consider compatible if all required features are present
            return all(features_present.values())
        except ImportError:
            logger.warning("OpenAI Agents SDK not found")
            return False

    def create_emotion_trace(
        workflow_name: str,
        ctx: RunContextWrapper[EmotionalContext],
        **additional_metadata
    ) -> Callable:
        """
        Create a standardized emotion trace with consistent metadata structure.
        
        Args:
            workflow_name: Name of the workflow (e.g., "Emotional_Processing")
            ctx: Context wrapper containing emotional state
            additional_metadata: Any additional metadata to include
            
        Returns:
            Trace context manager
        """
        # Get conversation ID for grouping from context
        conversation_id = ctx.context.get_value("conversation_id")
        if not conversation_id:
            conversation_id = f"conversation_{datetime.datetime.now().timestamp()}"
            ctx.context.set_value("conversation_id", conversation_id)
        
        # Create consistent base metadata
        metadata = {
            "system": "nyx_emotional_core",
            "version": "1.0",
            "cycle": str(ctx.context.cycle_count),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add emotional state if available
        if ctx.context.last_emotions:
            # Get primary emotion (with highest intensity)
            primary_emotion = max(ctx.context.last_emotions.items(), key=lambda x: x[1])
            metadata["primary_emotion"] = primary_emotion[0]
            metadata["intensity"] = primary_emotion[1]
            
            # Add significant secondary emotions for better emotional context
            secondary_emotions = {
                emotion: intensity 
                for emotion, intensity in ctx.context.last_emotions.items()
                if emotion != primary_emotion[0] and intensity > 0.3
            }
            if secondary_emotions:
                metadata["secondary_emotions"] = secondary_emotions
        
        # Add active agent information
        if ctx.context.active_agent:
            metadata["active_agent"] = ctx.context.active_agent
        
        # Add additional metadata
        metadata.update(additional_metadata)
        
        # Create and return the trace
        return trace(
            workflow_name=workflow_name,
            trace_id=gen_trace_id(),
            group_id=conversation_id,
            metadata=metadata
        )
    
    def create_enhanced_run_config(
        workflow_name: str,
        conversation_id: str,
        cycle_count: int,
        context_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 300,
        trace_id: Optional[str] = None
    ) -> RunConfig:
        """
        Create a standardized run configuration with SDK enhancements
        
        Args:
            workflow_name: Name of the workflow
            conversation_id: Conversation identifier
            cycle_count: Current processing cycle count
            context_data: Additional context metadata
            model: Optional model override
            temperature: Model temperature
            max_tokens: Maximum tokens for response
            trace_id: Optional trace identifier
            
        Returns:
            Enhanced run configuration
        """
        # Generate trace ID with SDK function if not provided
        if not trace_id:
            trace_id = gen_trace_id()
        
        # Create metadata with standard fields
        metadata = {
            "system": "nyx_emotional_core",
            "version": "1.0",
            "cycle": cycle_count,
            "conversation_id": conversation_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add custom context data
        if context_data:
            metadata.update(context_data)
        
        # Create run configuration with SDK features
        return RunConfig(
            workflow_name=workflow_name,
            trace_id=trace_id,
            group_id=conversation_id,
            model=model,
            model_settings=ModelSettings(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens
            ),
            handoff_input_filter=remove_all_tools,
            trace_include_sensitive_data=True,
            trace_metadata=metadata
        )
    def create_emotional_run_config(
        workflow_name: str,
        cycle_count: int,
        conversation_id: Optional[str] = None,
        input_text_length: Optional[int] = None,
        pattern_analysis: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 300,
        trace_id: Optional[str] = None,
    ) -> RunConfig:
        """
        SDK-optimized factory for creating standardized run configurations
        for emotional processing
        
        Args:
            workflow_name: Name of the workflow
            cycle_count: Current processing cycle count
            conversation_id: Conversation identifier for grouping traces
            input_text_length: Length of input text if available
            pattern_analysis: Pattern analysis result if available
            model: Model to use for inference (optional override)
            temperature: Model temperature setting
            max_tokens: Maximum tokens for model response
            trace_id: Optional custom trace ID
            
        Returns:
            Optimized run configuration
        """
        # Generate a proper trace ID using SDK function if not provided
        if not trace_id:
            trace_id = gen_trace_id()
        elif not trace_id.startswith("trace_"):
            trace_id = f"trace_{trace_id}"
        
        # Build metadata dictionary with standard fields
        metadata = {
            "system": "nyx_emotional_core",
            "version": "1.0",
            "cycle": cycle_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add optional metadata
        if input_text_length is not None:
            metadata["input_text_length"] = input_text_length
        
        if pattern_analysis:
            metadata["pattern_analysis"] = pattern_analysis
            
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        
        # Create run configuration with SDK features
        return RunConfig(
            workflow_name=workflow_name,
            trace_id=trace_id,
            group_id=conversation_id,
            model=model,
            model_settings=ModelSettings(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens
            ),
            handoff_input_filter=remove_all_tools,
            trace_include_sensitive_data=True,
            trace_metadata=metadata
        )
        
def ensure_awaitable(value):
    """
    Ensures a value is awaitable, converting non-awaitable results to coroutines.
    
    Args:
        value: Any value that might or might not be awaitable
        
    Returns:
        Coroutine that will resolve to the value
    """
    if inspect.isawaitable(value):
        return value
    
    async def wrapped_value():
        return value
    
    return wrapped_value()
