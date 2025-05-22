# nyx/core/brain/function_tools.py
import logging
import asyncio
from typing import Dict, List, Any, Optional
from agents import function_tool

logger = logging.getLogger(__name__)

# =============== Brain Function Tools ===============

@function_tool
async def process_user_message(ctx, 
                              user_input: str, 
                              context: Optional[Dict[str, Any]] = None,
                              use_thinking: Optional[bool] = None,
                              use_conditioning: Optional[bool] = None,
                              use_coordination: Optional[bool] = None,
                              mode: str = "auto") -> Dict[str, Any]:
    """
    Process a user message with all cognitive systems
    
    Args:
        user_input: User's message text
        context: Optional additional context
        use_thinking: Enable explicit thinking phase (None=auto-detect)
        use_conditioning: Enable conditioning system (None=auto-detect)
        use_coordination: Enable context distribution coordination (None=auto-detect)
        mode: Processing mode ("auto", "serial", "parallel", "coordinated", "simple")
    
    Returns:
        Processing results with emotional state, memories, etc.
    """
    brain = ctx.context
    
    # Process through the full system with feature control
    result = await brain.process_input(
        user_input, 
        context,
        use_thinking=use_thinking,
        use_conditioning=use_conditioning,
        use_coordination=use_coordination,
        mode=mode
    )
    return result

@function_tool
async def generate_agent_response(ctx, 
                                user_input: str, 
                                context: Optional[Dict[str, Any]] = None,
                                use_thinking: Optional[bool] = None,
                                use_conditioning: Optional[bool] = None,
                                use_coordination: Optional[bool] = None,
                                use_hierarchical_memory: Optional[bool] = None,
                                mode: str = "auto") -> Dict[str, Any]:
    """
    Generate a complete response to the user
    
    Args:
        user_input: User's message text
        context: Optional additional context
        use_thinking: Enable explicit thinking phase (None=auto-detect)
        use_conditioning: Enable conditioning system (None=auto-detect)
        use_coordination: Enable context distribution coordination (None=auto-detect)
        use_hierarchical_memory: Enable hierarchical memory context (None=auto-detect)
        mode: Processing mode ("auto", "serial", "parallel", "coordinated", "simple")
    
    Returns:
        Complete response with message, emotional expression, etc.
    """
    brain = ctx.context
    
    # Generate a full response with feature control
    response = await brain.generate_response(
        user_input, 
        context,
        use_thinking=use_thinking,
        use_conditioning=use_conditioning,
        use_coordination=use_coordination,
        use_hierarchical_memory=use_hierarchical_memory,
        mode=mode
    )
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
        result = await brain.processing_manager.get_processing_stats()
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
async def run_thinking(ctx, 
                      user_input: str, 
                      thinking_level: int = 1, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run thinking process on input using the unified processing system
    
    Args:
        user_input: User's input text
        thinking_level: Thinking depth level (1-3)
        context: Additional context
    
    Returns:
        Thinking results
    """
    brain = ctx.context
    
    # Use unified processing with thinking explicitly enabled
    result = await brain.process_input(
        user_input,
        context,
        use_thinking=True,
        thinking_level=thinking_level
    )
    
    # Extract just the thinking results if available
    if result.get("thinking_applied") and "thinking_result" in result:
        return {
            "thinking_applied": True,
            "thinking_result": result["thinking_result"],
            "thinking_level": thinking_level
        }
    
    return {"error": "Thinking was not applied or thinking tools not available"}

@function_tool
async def process_with_coordination(ctx, 
                                  user_input: str, 
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process input using the coordinated processing system (a2a)
    
    Args:
        user_input: User's input text
        context: Optional additional context
    
    Returns:
        Coordinated processing results
    """
    brain = ctx.context
    
    # Force coordination mode
    result = await brain.process_input(
        user_input,
        context,
        use_coordination=True,
        mode="coordinated"
    )
    return result


@function_tool
async def process_simple(ctx, 
                       user_input: str, 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process input using minimal features for fast response
    
    Args:
        user_input: User's input text
        context: Optional additional context
    
    Returns:
        Simple processing results
    """
    brain = ctx.context
    
    # Disable all optional features for speed
    result = await brain.process_input(
        user_input,
        context,
        use_thinking=False,
        use_conditioning=False,
        use_coordination=False,
        mode="serial"
    )
    return result


@function_tool
async def process_with_all_features(ctx, 
                                  user_input: str, 
                                  context: Optional[Dict[str, Any]] = None,
                                  thinking_level: int = 2) -> Dict[str, Any]:
    """
    Process input with all available cognitive features enabled
    
    Args:
        user_input: User's input text
        context: Optional additional context
        thinking_level: Thinking depth level (1-3)
    
    Returns:
        Full processing results with all features
    """
    brain = ctx.context
    
    # Enable all features
    result = await brain.generate_response(
        user_input,
        context,
        use_thinking=True,
        use_conditioning=True,
        use_coordination=True,
        use_hierarchical_memory=True,
        thinking_level=thinking_level,
        mode="auto"
    )
    return result


@function_tool
async def get_processing_capabilities(ctx) -> Dict[str, Any]:
    """
    Get information about available processing capabilities
    
    Returns:
        Available processing features and their status
    """
    brain = ctx.context
    
    capabilities = {
        "thinking": {
            "available": hasattr(brain, "thinking_config") and brain.thinking_config.get("thinking_enabled", False),
            "config": getattr(brain, "thinking_config", {})
        },
        "conditioning": {
            "available": hasattr(brain, "conditioned_input_processor") and brain.conditioned_input_processor is not None
        },
        "coordination": {
            "available": hasattr(brain, "context_distribution") and brain.context_distribution is not None,
            "status": brain.get_context_distribution_status() if hasattr(brain, "get_context_distribution_status") else {}
        },
        "hierarchical_memory": {
            "available": hasattr(brain, "memory_core") and brain.memory_core is not None
        },
        "processing_modes": ["auto", "serial", "parallel", "coordinated", "simple"],
        "current_processing_manager": hasattr(brain, "processing_manager") and brain.processing_manager is not None
    }
    
    return capabilities


@function_tool
async def set_thinking_enabled(ctx, enabled: bool) -> Dict[str, Any]:
    """
    Enable or disable thinking for processing
    
    Args:
        enabled: Whether to enable thinking
        
    Returns:
        Configuration update result
    """
    brain = ctx.context
    
    if hasattr(brain, "thinking_config"):
        brain.thinking_config["thinking_enabled"] = enabled
        return {
            "success": True,
            "thinking_enabled": enabled,
            "config": brain.thinking_config
        }
    
    return {"error": "Thinking configuration not available"}


@function_tool
async def compare_processing_modes(ctx, 
                                 user_input: str, 
                                 modes: List[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compare different processing modes on the same input
    
    Args:
        user_input: User's input text
        modes: List of modes to compare (default: ["simple", "auto", "coordinated"])
        context: Optional additional context
        
    Returns:
        Comparison results with timing and features used
    """
    brain = ctx.context
    
    if modes is None:
        modes = ["simple", "auto", "coordinated"]
    
    results = {}
    
    for mode in modes:
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Configure based on mode
            if mode == "simple":
                result = await brain.process_input(
                    user_input, context,
                    use_thinking=False,
                    use_conditioning=False,
                    use_coordination=False,
                    mode="serial"
                )
            elif mode == "coordinated":
                result = await brain.process_input(
                    user_input, context,
                    use_coordination=True,
                    mode="coordinated"
                )
            else:
                result = await brain.process_input(
                    user_input, context,
                    mode=mode
                )
            
            end_time = asyncio.get_event_loop().time()
            
            results[mode] = {
                "success": True,
                "processing_time": end_time - start_time,
                "features_used": result.get("features_used", {}),
                "active_modules": len(result.get("active_modules_for_input", [])),
                "response_quality_indicators": {
                    "has_memories": bool(result.get("memories", [])),
                    "has_emotional_state": bool(result.get("emotional_state", {})),
                    "has_thinking": result.get("thinking_applied", False),
                    "has_conditioning": result.get("conditioning_applied", False)
                }
            }
            
        except Exception as e:
            results[mode] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "comparison_results": results,
        "fastest_mode": min(
            [(mode, data["processing_time"]) for mode, data in results.items() if data.get("success", False)],
            key=lambda x: x[1]
        )[0] if any(data.get("success", False) for data in results.values()) else None
    }

@function_tool
async def register_issue(ctx, 
                      issue_data: Dict[str, Any], 
                      auto_attempt_resolution: bool = True) -> Dict[str, Any]:
    """
    Register an issue with the issue tracking system
    
    Args:
        issue_data: Issue information
        auto_attempt_resolution: Whether to attempt automatic resolution
        
    Returns:
        Registration result
    """
    brain = ctx.context
    
    # Register issue
    if hasattr(brain, "issue_tracker") and brain.issue_tracker:
        result = await brain.issue_tracker.register_issue(issue_data, auto_attempt_resolution)
        return result
    
    return {"error": "Issue tracker not initialized"}

@function_tool
async def set_meta_tone(ctx, tone_name: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Set the meta-tone for agent responses
    
    Args:
        tone_name: Name of the meta-tone to use
        reason: Optional reason for the tone change
        
    Returns:
        Tone change status
    """
    brain = ctx.context
    
    # Set meta-tone
    if hasattr(brain, "agent_integration") and brain.agent_integration:
        result = await brain.agent_integration.set_meta_tone(tone_name, reason)
        return result
    
    return {"error": "Agent integration not initialized"}

@function_tool
async def get_agent_stats(ctx) -> Dict[str, Any]:
    """
    Get agent system statistics
    
    Returns:
        Agent statistics
    """
    brain = ctx.context
    
    # Get agent stats
    if hasattr(brain, "agent_integration") and brain.agent_integration:
        result = await brain.agent_integration.get_system_statistics()
        return result
    
    return {"error": "Agent integration not initialized"}
# Add these to nyx/core/brain/function_tools.py

@function_tool
async def analyze_module(ctx, module_path: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze a module for potential improvements
    
    Args:
        module_path: Path to the module (e.g., 'nyx.core.brain.base')
        detailed: Whether to provide detailed analysis
        
    Returns:
        Analysis results
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return await brain.module_optimizer.analyze_module(module_path, detailed)

@function_tool
async def optimize_module(ctx, module_path: str, optimization_goal: str = "general", ensure_backwards_compatible: bool = True) -> Dict[str, Any]:
    """
    Create an optimized version of an existing module
    
    Args:
        module_path: Path to the module to optimize
        optimization_goal: Specific goal for optimization (e.g., "performance", "error_handling")
        ensure_backwards_compatible: Whether the optimized module should be backwards compatible
        
    Returns:
        Optimization results
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return await brain.module_optimizer.optimize_module(module_path, optimization_goal, ensure_backwards_compatible)

@function_tool
async def create_new_module(ctx, module_name: str, description: str, requirements: List[str] = None, integration_points: List[str] = None) -> Dict[str, Any]:
    """
    Create a new module from scratch
    
    Args:
        module_name: Name for the new module
        description: Description of the module's purpose
        requirements: List of specific requirements for the module
        integration_points: List of modules this should integrate with
        
    Returns:
        Creation results
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return await brain.module_optimizer.create_new_module(module_name, description, requirements, integration_points)

@function_tool
async def edit_optimized_module(ctx, module_name: str, version: int, edits_description: str) -> Dict[str, Any]:
    """
    Edit an optimized module
    
    Args:
        module_name: Name of the module to edit
        version: Version of the optimized module
        edits_description: Description of the edits to make
        
    Returns:
        Editing results
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return await brain.module_optimizer.edit_optimized_module(module_name, version, edits_description)

@function_tool
async def get_optimized_module(ctx, module_name: str, version: Optional[int] = None) -> Dict[str, Any]:
    """
    Get an optimized module
    
    Args:
        module_name: Name of the module
        version: Specific version to get, or latest if None
        
    Returns:
        Module data
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return brain.module_optimizer.get_optimized_module(module_name, version)

@function_tool
async def list_optimized_modules(ctx) -> Dict[str, Any]:
    """
    List all optimized modules
    
    Returns:
        List of optimized modules
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return brain.module_optimizer.list_optimized_modules()

@function_tool
async def import_and_test_module(ctx, module_name: str, version: int) -> Dict[str, Any]:
    """
    Import and run basic tests on an optimized module
    
    Args:
        module_name: Name of the module
        version: Version to test
        
    Returns:
        Test results
    """
    brain = ctx.context
    
    # Initialize module optimizer if not available
    if not hasattr(brain, "module_optimizer"):
        from nyx.core.brain.module_optimizer import ModuleOptimizer
        brain.module_optimizer = ModuleOptimizer(brain)
    
    return await brain.module_optimizer.import_and_test_module(module_name, version)
# Add to nyx/core/brain/function_tools.py

@function_tool
async def check_system_health(ctx, detailed: bool = True) -> Dict[str, Any]:
    """
    Run a comprehensive health check on all Nyx components, functions, and agents
    
    Args:
        detailed: Whether to include detailed information in the results
        
    Returns:
        Health check results with component status and issues
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return await brain.system_health_checker.check_system_health(detailed)

@function_tool
async def get_system_overview(ctx) -> Dict[str, Any]:
    """
    Get a high-level overview of the system's operational status
    
    Returns:
        System overview data including component health statistics
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return await brain.system_health_checker.get_system_overview()

@function_tool
async def test_function_tool(ctx, tool_name: str) -> Dict[str, Any]:
    """
    Test a specific function tool to verify it's accessible and properly structured
    
    Args:
        tool_name: Name of the function tool to test
        
    Returns:
        Test results for the function tool
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return await brain.system_health_checker.test_function_tool(tool_name)

@function_tool
async def check_agent_capabilities(ctx, agent_name: str) -> Dict[str, Any]:
    """
    Check the capabilities of a specific agent
    
    Args:
        agent_name: Name of the agent to check
        
    Returns:
        Agent capabilities assessment
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return await brain.system_health_checker.check_agent_capabilities(agent_name)

@function_tool
async def verify_module_imports(ctx, module_names: List[str] = None) -> Dict[str, Any]:
    """
    Verify that specified modules can be imported
    
    Args:
        module_names: List of module names to check, or None for default core modules
        
    Returns:
        Import verification results
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return await brain.system_health_checker.verify_module_imports(module_names)

@function_tool
async def get_component_docs(ctx, component_name: str) -> Dict[str, Any]:
    """
    Get documentation for a component
    
    Args:
        component_name: Name of the component
        
    Returns:
        Component documentation including methods and docstrings
    """
    brain = ctx.context
    
    # Initialize system health checker if not available
    if not hasattr(brain, "system_health_checker"):
        from nyx.core.brain.system_health_checker import SystemHealthChecker
        brain.system_health_checker = SystemHealthChecker(brain)
    
    return brain.system_health_checker.get_component_docs(component_name)
@function_tool
async def get_integration_status(ctx) -> Dict[str, Any]:
    """
    Get the status of integration components
    
    Returns:
        Integration system status
    """
    brain = ctx.context
    
    if hasattr(brain, "integration_manager") and brain.integration_manager:
        return await brain.integration_manager.get_integration_status()
    
    return {"error": "Integration manager not initialized"}

@function_tool
async def publish_event(ctx, event_type: str, data: Dict[str, Any], source: str = "brain_agent") -> Dict[str, Any]:
    """
    Publish an event to the event bus
    
    Args:
        event_type: Type of event
        data: Event data
        source: Event source
        
    Returns:
        Publication result
    """
    brain = ctx.context
    
    if hasattr(brain, "event_bus") and brain.event_bus:
        from nyx.core.integration.event_bus import Event
        event = Event(event_type=event_type, source=source, data=data)
        await brain.event_bus.publish(event)
        return {"success": True, "event_type": event_type}
    
    return {"success": False, "error": "Event bus not initialized"}

@function_tool
async def get_system_context(ctx) -> Dict[str, Any]:
    """
    Get the current system context state
    
    Returns:
        System context state
    """
    brain = ctx.context
    
    if hasattr(brain, "system_context") and brain.system_context:
        # Return a simplified view of the system context
        context = brain.system_context
        return {
            "cycle_count": context.cycle_count,
            "active_goals_count": len(context.active_goals),
            "user_models_count": len(context.user_models),
            "affective_state": {
                "primary_emotion": context.affective_state.primary_emotion,
                "valence": context.affective_state.valence,
                "arousal": context.affective_state.arousal
            },
            "body_state": {
                "has_visual_form": context.body_state.has_visual_form,
                "form_description": context.body_state.form_description,
                "dominant_sensation": context.body_state.dominant_sensation,
                "dominant_region": context.body_state.dominant_region
            }
        }
    
    return {"error": "System context not initialized"}
@function_tool
async def get_dev_logs(ctx, 
                     log_type: Optional[str] = None,
                     limit: int = 10,
                     tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get developer logs with optional filtering
    
    Args:
        log_type: Type of logs to retrieve
        limit: Maximum number of logs to return
        tags: Filter by tags
        
    Returns:
        Filtered logs
    """
    brain = ctx.context
    
    if not hasattr(brain, "dev_log_storage") or not brain.dev_log_storage:
        from nyx.dev_log.storage import get_dev_log_storage
        brain.dev_log_storage = get_dev_log_storage()
        await brain.dev_log_storage.initialize()
    
    logs = await brain.dev_log_storage.get_logs(
        log_type=log_type,
        limit=limit,
        tags=tags
    )
    
    return {
        "logs": logs,
        "count": len(logs),
        "has_more": len(logs) == limit
    }

@function_tool
async def get_synergy_stats(ctx) -> Dict[str, Any]:
    """
    Get statistics about synergy recommendations
    
    Returns:
        Synergy recommendation statistics
    """
    brain = ctx.context
    
    if not hasattr(brain, "dev_log_storage") or not brain.dev_log_storage:
        from nyx.dev_log.storage import get_dev_log_storage
        brain.dev_log_storage = get_dev_log_storage()
        await brain.dev_log_storage.initialize()
    
    return await brain.dev_log_storage.get_recommendation_stats()
