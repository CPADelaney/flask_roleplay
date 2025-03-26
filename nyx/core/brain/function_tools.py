# nyx/core/brain/function_tools.py
import logging
import asyncio
from typing import Dict, List, Any, Optional
from agents import function_tool

logger = logging.getLogger(__name__)

# =============== Brain Function Tools ===============

@function_tool
async def process_user_message(ctx, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user message with all cognitive systems
    
    Args:
        user_input: User's message text
        context: Optional additional context
    
    Returns:
        Processing results with emotional state, memories, etc.
    """
    brain = ctx.context
    
    # Process through the full system
    result = await brain.process_input(user_input, context)
    return result

@function_tool
async def generate_agent_response(ctx, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a complete response to the user
    
    Args:
        user_input: User's message text
        context: Optional additional context
    
    Returns:
        Complete response with message, emotional expression, etc.
    """
    brain = ctx.context
    
    # Generate a full response
    response = await brain.generate_response(user_input, context)
    return response

@function_tool
async def run_cognitive_cycle(ctx, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a meta-cognitive cycle
    
    Args:
        context_data: Additional context for the cognitive cycle
    
    Returns:
        Results of the cognitive cycle
    """
    brain = ctx.context
    
    # Run meta-cognitive cycle
    if brain.meta_core:
        result = await brain.meta_core.cognitive_cycle(context_data)
        return result
    
    return {"error": "Meta core not initialized"}

@function_tool
async def get_brain_stats(ctx) -> Dict[str, Any]:
    """
    Get comprehensive statistics about all systems
    
    Returns:
        Statistics for all systems
    """
    brain = ctx.context
    
    # Get comprehensive stats
    stats = await brain.get_system_stats()
    return stats

@function_tool
async def perform_maintenance(ctx) -> Dict[str, Any]:
    """
    Run maintenance on all systems
    
    Returns:
        Maintenance results
    """
    brain = ctx.context
    
    # Run maintenance on all systems
    result = await brain.run_maintenance()
    return result

@function_tool
async def get_identity_state(ctx) -> Dict[str, Any]:
    """
    Get current state of Nyx's identity
    
    Returns:
        Identity state information
    """
    brain = ctx.context
    
    # Get identity state
    result = await brain.get_identity_state()
    return result

@function_tool
async def adapt_experience_sharing(ctx, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt experience sharing based on user feedback
    
    Args:
        user_id: User ID
        feedback: User feedback data
    
    Returns:
        Adaptation results
    """
    brain = ctx.context
    
    # Adapt experience sharing
    result = await brain.adapt_experience_sharing(user_id, feedback)
    return result

@function_tool
async def run_experience_consolidation(ctx) -> Dict[str, Any]:
    """
    Run experience consolidation process
    
    Returns:
        Consolidation results
    """
    brain = ctx.context
    
    # Run consolidation
    result = await brain.run_experience_consolidation()
    return result

@function_tool
async def add_procedural_knowledge(ctx, name: str, steps: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
    """
    Add procedural knowledge to the system
    
    Args:
        name: Name of the procedure
        steps: List of procedure steps
        domain: Knowledge domain
    
    Returns:
        Procedure creation result
    """
    brain = ctx.context
    
    # Add procedure
    result = await brain.add_procedure(name, steps, domain=domain)
    return result

@function_tool
async def run_procedure(ctx, name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a stored procedure
    
    Args:
        name: Name of the procedure to run
        context: Optional execution context
    
    Returns:
        Execution result
    """
    brain = ctx.context
    
    # Execute procedure
    result = await brain.execute_procedure(name, context)
    return result

@function_tool
async def analyze_chunking(ctx, procedure_name: str) -> Dict[str, Any]:
    """
    Analyze a procedure for chunking opportunities
    
    Args:
        procedure_name: Name of procedure to analyze
        
    Returns:
        Chunking analysis result
    """
    brain = ctx.context
    
    # Analyze chunking opportunities
    result = await brain.analyze_chunking(procedure_name)
    return result

@function_tool
async def register_reflex(ctx, reflex_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a new reflex pattern
    
    Args:
        reflex_data: Reflex pattern data
    
    Returns:
        Registration result
    """
    brain = ctx.context
    
    # Register reflex
    if hasattr(brain, "reflexive_system") and brain.reflexive_system:
        result = await brain.reflexive_system.register_reflex(reflex_data)
        return result
    
    return {"error": "Reflexive system not initialized"}

@function_tool
async def process_stimulus(ctx, stimulus: Dict[str, Any], domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a stimulus with reflexes
    
    Args:
        stimulus: Stimulus data
        domain: Optional domain to limit pattern matching
    
    Returns:
        Processing result
    """
    brain = ctx.context
    
    # Process stimulus
    if hasattr(brain, "reflexive_system") and brain.reflexive_system:
        result = await brain.reflexive_system.process_stimulus_fast(stimulus, domain)
        return result
    
    return {"error": "Reflexive system not initialized"}

@function_tool
async def enable_self_configuration(ctx) -> Dict[str, Any]:
    """
    Enable the self-configuration system
    
    Returns:
        Enablement result
    """
    brain = ctx.context
    
    # Enable self-configuration
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.enable()
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def evaluate_and_adjust_parameters(ctx) -> Dict[str, Any]:
    """
    Evaluate current parameters and adjust if needed
    
    Returns:
        Adjustment results
    """
    brain = ctx.context
    
    # Evaluate parameters
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.evaluate_and_adjust_parameters()
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def change_adaptation_strategy(ctx, strategy: str) -> Dict[str, Any]:
    """
    Change the adaptation strategy
    
    Args:
        strategy: Strategy name to use
    
    Returns:
        Change result
    """
    brain = ctx.context
    
    # Change strategy
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.change_adaptation_strategy(strategy)
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def get_self_configuration_status(ctx) -> Dict[str, Any]:
    """
    Get status of the self-configuration system
    
    Returns:
        Self-configuration status
    """
    brain = ctx.context
    
    # Get status
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.get_status()
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def reset_parameter_to_default(ctx, param_name: str) -> Dict[str, Any]:
    """
    Reset a parameter to its default value
    
    Args:
        param_name: Name of parameter to reset
    
    Returns:
        Reset result
    """
    brain = ctx.context
    
    # Reset parameter
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.reset_parameter_to_default(param_name)
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def process_user_feedback_for_configuration(ctx, 
                                             feedback_type: str, 
                                             feedback_text: str,
                                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process user feedback to influence configuration
    
    Args:
        feedback_type: Type of feedback ("positive", "negative", "specific")
        feedback_text: Text of user feedback
        context: Additional context information
    
    Returns:
        Processing results
    """
    brain = ctx.context
    
    # Process feedback
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.process_user_feedback(feedback_type, feedback_text, context)
        return result
    
    return {"error": "Self-configuration manager not initialized"}

@function_tool
async def set_processing_mode(ctx, mode: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Set the processing mode for the brain
    
    Args:
        mode: Processing mode ("serial", "parallel", "distributed", "reflexive", "agent", "integrated", "auto")
        reason: Optional reason for the mode change
    
    Returns:
        Mode change results
    """
    brain = ctx.context
    
    # Set processing mode
    if hasattr(brain, "processing_manager") and brain.processing_manager:
        result = await brain.processing_manager.set_processing_mode(mode, reason)
        return result
    
    return {"error": "Processing manager not initialized"}

@function_tool
async def get_processing_stats(ctx) -> Dict[str, Any]:
    """
    Get processing statistics
    
    Returns:
        Processing statistics
    """
    brain = ctx.context
    
    # Get processing stats
    if hasattr(brain, "processing_manager") and brain.processing_manager:
        result = {
            "current_mode": brain.processing_manager.current_mode,
            "mode_switch_history": brain.processing_manager.mode_switch_history[-5:] if brain.processing_manager.mode_switch_history else [],
            "processors": list(brain.processing_manager.processors.keys())
        }
        return result
    
    return {"error": "Processing manager not initialized"}

@function_tool
async def initialize_streaming(ctx, video_source=0, audio_source=None) -> Dict[str, Any]:
    """
    Initialize streaming capabilities
    
    Args:
        video_source: Video source
        audio_source: Audio source
    
    Returns:
        Initialization results
    """
    brain = ctx.context
    
    # Initialize streaming
    if hasattr(brain, "initialize_streaming"):
        result = await brain.initialize_streaming(video_source, audio_source)
        return {"success": True, "streaming_initialized": True}
    
    return {"error": "Streaming initialization not available"}

@function_tool
async def process_streaming_event(ctx, event_type: str, event_data: Dict[str, Any], significance: float = 5.0) -> Dict[str, Any]:
    """
    Process a significant streaming event
    
    Args:
        event_type: Type of event
        event_data: Event data
        significance: Event significance
    
    Returns:
        Processing results
    """
    brain = ctx.context
    
    # Process streaming event
    if hasattr(brain, "process_streaming_event"):
        result = await brain.process_streaming_event(event_type, event_data, significance)
        return result
    
    return {"error": "Streaming event processing not available"}

@function_tool
async def run_thinking(ctx, user_input: str, thinking_level: int = 1, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run thinking process on input
    
    Args:
        user_input: User's input text
        thinking_level: Thinking depth level (1-3)
        context: Additional context
    
    Returns:
        Thinking results
    """
    brain = ctx.context
    
    # Run thinking
    if hasattr(brain.thinking_tools, "think_before_responding"):
        result = await brain.thinking_tools.think_before_responding(brain, user_input, thinking_level, context)
        return result
    
    return {"error": "Thinking tools not available"}
