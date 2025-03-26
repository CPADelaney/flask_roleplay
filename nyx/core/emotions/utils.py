# nyx/core/emotions/utils.py

"""
Utility functions for the Nyx emotional system.
"""

import functools
import logging
from typing import Callable, Any, Dict, Optional

from agents.exceptions import UserError, AgentsException

logger = logging.getLogger(__name__)

def handle_errors(logger_message: str = "An error occurred"):
    """
    Decorator for consistent error handling across emotional system functions
    
    Args:
        logger_message: Custom error message prefix for logging
        
    Returns:
        Decorated function with consistent error handling
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except UserError as e:
                # Handle user errors (invalid inputs, etc.)
                logger.warning(f"{logger_message}: {e}")
                return {
                    "success": False, 
                    "error": str(e), 
                    "error_type": "user_error"
                }
            except AgentsException as e:
                # Handle Agent SDK errors
                logger.error(f"{logger_message} (Agent SDK): {e}")
                return {
                    "success": False, 
                    "error": str(e), 
                    "error_type": "agent_error"
                }
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"{logger_message} (Unexpected): {e}", exc_info=True)
                return {
                    "success": False, 
                    "error": "An unexpected error occurred", 
                    "error_type": "system_error"
                }
        return wrapper
    return decorator

def create_run_config(
    workflow_name: Optional[str] = None,
    trace_id: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 300,
    include_sensitive_data: bool = True,
    cycle_count: int = 0,
    context_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Factory function for creating standardized run configurations
    
    Args:
        workflow_name: Name of the workflow
        trace_id: Unique trace identifier
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        include_sensitive_data: Whether to include sensitive data in traces
        cycle_count: Current processing cycle count
        context_data: Additional context metadata
        
    Returns:
        Run configuration dictionary for the Agent SDK
    """
    from agents import RunConfig, ModelSettings
    from agents.extensions import handoff_filters
    import datetime
    
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
        trace_id=trace_id or f"emotion_{datetime.datetime.now().timestamp()}",
        model="o3-mini",
        model_settings=ModelSettings(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        ),
        handoff_input_filter=handoff_filters.keep_relevant_history,
        trace_include_sensitive_data=include_sensitive_data,
        trace_metadata=metadata
    )
def with_emotion_trace(func: Callable):
    """
    Decorator to add tracing to emotional methods with improved metadata
    
    Args:
        func: The function to wrap with tracing
        
    Returns:
        Wrapped function with tracing
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        workflow_name = f"Emotion_{func.__name__}"
        trace_id = f"emotion_{func.__name__}_{datetime.datetime.now().timestamp()}"
        
        # Add useful metadata and group traces by conversation
        metadata = {
            "function": func.__name__,
            "cycle_count": self.context.cycle_count,
            "current_emotion": self.get_dominant_emotion()[0] if hasattr(self, "get_dominant_emotion") else "Unknown",
            "performance_metrics": {
                k: v for k, v in self.performance_metrics.items() 
                if hasattr(self, "performance_metrics") and k in ["api_calls", "average_response_time"]
            }
        }
        
        # Get conversation ID from context if available
        conversation_id = "default"
        if hasattr(self, "context") and hasattr(self.context, "temp_data"):
            conversation_id = self.context.temp_data.get("conversation_id", "default")
        
        with trace(
            workflow_name=workflow_name, 
            trace_id=trace_id,
            group_id=f"conversation_{conversation_id}",
            metadata=metadata
        ):
            return await func(self, *args, **kwargs)
    return wrapper
