# nyx/core/brain/function_tools.py
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from agents import function_tool

logger = logging.getLogger(__name__)

# =============== Pydantic Models for Function Tools ===============

class GenericContext(BaseModel):
    """Generic context model for passing arbitrary context data"""
    pass  # Allow any fields

class GenericResult(BaseModel):
    """Generic result model for function returns"""
    success: Optional[bool] = None
    error: Optional[str] = None
    # Allow any additional fields

class EmotionalState(BaseModel):
    """Emotional state information"""
    primary_emotion: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    intensity: Optional[float] = None

class Memory(BaseModel):
    """Memory item"""
    content: Optional[str] = None
    timestamp: Optional[str] = None
    relevance: Optional[float] = None
    memory_type: Optional[str] = None

class FeaturesUsed(BaseModel):
    """Features used in processing"""
    thinking: Optional[bool] = None
    conditioning: Optional[bool] = None
    coordination: Optional[bool] = None
    hierarchical_memory: Optional[bool] = None

class ProcessingResult(BaseModel):
    """Result from processing operations"""
    success: Optional[bool] = None
    error: Optional[str] = None
    thinking_applied: Optional[bool] = None
    thinking_result: Optional[Any] = None
    thinking_level: Optional[int] = None
    conditioning_applied: Optional[bool] = None
    features_used: Optional[FeaturesUsed] = None
    active_modules_for_input: Optional[List[str]] = None
    memories: Optional[List[Memory]] = None
    emotional_state: Optional[EmotionalState] = None
    message: Optional[str] = None

class FeedbackData(BaseModel):
    """User feedback data"""
    rating: Optional[float] = None
    comment: Optional[str] = None
    timestamp: Optional[str] = None
    context: Optional[GenericContext] = None

class ProcedureParameters(BaseModel):
    """Parameters for a procedure step"""
    pass  # Allow any fields

class ProcedureConditions(BaseModel):
    """Conditions for a procedure step"""
    pass  # Allow any fields

class ProcedureStep(BaseModel):
    """Single step in a procedure"""
    name: str
    action: str
    parameters: Optional[ProcedureParameters] = None
    conditions: Optional[ProcedureConditions] = None

class StimulusMetadata(BaseModel):
    """Metadata for stimulus"""
    pass  # Allow any fields

class StimulusData(BaseModel):
    """Stimulus data for reflexive processing"""
    type: str
    content: Any
    metadata: Optional[StimulusMetadata] = None

class EventMetadata(BaseModel):
    """Metadata for events"""
    pass  # Allow any fields

class EventData(BaseModel):
    """Event data for streaming events"""
    timestamp: Optional[str] = None
    payload: Optional[Any] = None
    metadata: Optional[EventMetadata] = None

class IssueMetadata(BaseModel):
    """Metadata for issues"""
    pass  # Allow any fields

class IssueData(BaseModel):
    """Issue tracking data"""
    type: str
    description: str
    severity: Optional[str] = None
    component: Optional[str] = None
    metadata: Optional[IssueMetadata] = None

class SystemStats(BaseModel):
    """System statistics"""
    pass  # Allow any fields

class StatsResult(BaseModel):
    """System statistics result"""
    stats: Optional[SystemStats] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None

class IdentityTraits(BaseModel):
    """Identity traits"""
    pass  # Allow any fields

class CurrentState(BaseModel):
    """Current identity state"""
    pass  # Allow any fields

class IdentityState(BaseModel):
    """Identity state information"""
    name: Optional[str] = None
    current_state: Optional[CurrentState] = None
    traits: Optional[List[str]] = None
    metadata: Optional[GenericContext] = None

class MaintenanceIssue(BaseModel):
    """Maintenance issue details"""
    component: Optional[str] = None
    issue_type: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None

class MaintenanceResult(BaseModel):
    """Maintenance operation result"""
    operations_performed: Optional[List[str]] = None
    issues_found: Optional[List[MaintenanceIssue]] = None
    issues_resolved: Optional[List[MaintenanceIssue]] = None
    success: bool = True

class ThinkingCapability(BaseModel):
    """Thinking capability info"""
    available: bool
    config: Optional[GenericContext] = None

class ConditioningCapability(BaseModel):
    """Conditioning capability info"""
    available: bool

class CoordinationStatus(BaseModel):
    """Coordination status details"""
    pass  # Allow any fields

class CoordinationCapability(BaseModel):
    """Coordination capability info"""
    available: bool
    status: Optional[CoordinationStatus] = None

class HierarchicalMemoryCapability(BaseModel):
    """Hierarchical memory capability info"""
    available: bool

class CapabilitiesResult(BaseModel):
    """Processing capabilities information"""
    thinking: Optional[ThinkingCapability] = None
    conditioning: Optional[ConditioningCapability] = None
    coordination: Optional[CoordinationCapability] = None
    hierarchical_memory: Optional[HierarchicalMemoryCapability] = None
    processing_modes: Optional[List[str]] = None
    current_processing_manager: Optional[bool] = None

class ResponseQualityIndicators(BaseModel):
    """Response quality indicators"""
    has_memories: bool
    has_emotional_state: bool
    has_thinking: bool
    has_conditioning: bool

class ModeResult(BaseModel):
    """Result for a single processing mode"""
    success: bool
    error: Optional[str] = None
    processing_time: Optional[float] = None
    features_used: Optional[FeaturesUsed] = None
    active_modules: Optional[int] = None
    response_quality_indicators: Optional[ResponseQualityIndicators] = None

class ComparisonResults(BaseModel):
    """Results from mode comparison"""
    pass  # Allow any mode names as fields

class ComparisonResult(BaseModel):
    """Mode comparison results"""
    comparison_results: ComparisonResults
    fastest_mode: Optional[str] = None

class ComponentInfo(BaseModel):
    """Component health info"""
    healthy: bool
    status: Optional[str] = None
    error: Optional[str] = None

class ComponentsHealth(BaseModel):
    """Health status of all components"""
    pass  # Allow any component names as fields

class HealthIssue(BaseModel):
    """Health check issue"""
    component: str
    issue_type: str
    description: str
    severity: str

class HealthCheckResult(BaseModel):
    """Health check results"""
    healthy: bool
    components: Optional[ComponentsHealth] = None
    issues: Optional[List[HealthIssue]] = None
    warnings: Optional[List[str]] = None

class AffectiveStateInfo(BaseModel):
    """Affective state information"""
    primary_emotion: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None

class BodyStateInfo(BaseModel):
    """Body state information"""
    has_visual_form: Optional[bool] = None
    form_description: Optional[str] = None
    dominant_sensation: Optional[str] = None
    dominant_region: Optional[str] = None

class SystemContextResult(BaseModel):
    """System context state"""
    cycle_count: Optional[int] = None
    active_goals_count: Optional[int] = None
    user_models_count: Optional[int] = None
    affective_state: Optional[AffectiveStateInfo] = None
    body_state: Optional[BodyStateInfo] = None

class LogEntry(BaseModel):
    """Single log entry"""
    timestamp: str
    log_type: str
    message: str
    tags: Optional[List[str]] = None
    metadata: Optional[GenericContext] = None

class LogsResult(BaseModel):
    """Developer logs result"""
    logs: List[LogEntry]
    count: int
    has_more: bool

class ModuleAnalysis(BaseModel):
    """Module analysis details"""
    pass  # Allow any fields

class ModuleIssue(BaseModel):
    """Module issue details"""
    type: str
    description: str
    line: Optional[int] = None
    severity: Optional[str] = None

class ModuleAnalysisResult(BaseModel):
    """Module analysis result"""
    module_path: str
    analysis: ModuleAnalysis
    suggestions: Optional[List[str]] = None
    issues: Optional[List[ModuleIssue]] = None

class ModuleOptimizationResult(BaseModel):
    """Module optimization result"""
    success: bool
    module_name: str
    version: Optional[int] = None
    optimized_code: Optional[str] = None
    changes_made: Optional[List[str]] = None
    error: Optional[str] = None

class MethodInfo(BaseModel):
    """Method documentation info"""
    name: str
    docstring: Optional[str] = None
    parameters: Optional[List[str]] = None

class ComponentDocsResult(BaseModel):
    """Component documentation result"""
    component_name: str
    documentation: Optional[str] = None
    methods: Optional[List[MethodInfo]] = None
    error: Optional[str] = None

# =============== Brain Function Tools ===============

@function_tool
async def process_user_message(ctx, 
                              user_input: str, 
                              context: Optional[GenericContext] = None,
                              use_thinking: Optional[bool] = None,
                              use_conditioning: Optional[bool] = None,
                              use_coordination: Optional[bool] = None,
                              mode: str = "auto") -> ProcessingResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Process through the full system with feature control
    result = await brain.process_input(
        user_input, 
        context_dict,
        use_thinking=use_thinking,
        use_conditioning=use_conditioning,
        use_coordination=use_coordination,
        mode=mode
    )
    return ProcessingResult(**result)

@function_tool
async def generate_agent_response(ctx, 
                                user_input: str, 
                                context: Optional[GenericContext] = None,
                                use_thinking: Optional[bool] = None,
                                use_conditioning: Optional[bool] = None,
                                use_coordination: Optional[bool] = None,
                                use_hierarchical_memory: Optional[bool] = None,
                                mode: str = "auto") -> ProcessingResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Generate a full response with feature control
    response = await brain.generate_response(
        user_input, 
        context_dict,
        use_thinking=use_thinking,
        use_conditioning=use_conditioning,
        use_coordination=use_coordination,
        use_hierarchical_memory=use_hierarchical_memory,
        mode=mode
    )
    return ProcessingResult(**response)

@function_tool
async def run_cognitive_cycle(ctx, context_data: Optional[GenericContext] = None) -> GenericResult:
    """
    Run a meta-cognitive cycle
    
    Args:
        context_data: Additional context for the cognitive cycle
    
    Returns:
        Results of the cognitive cycle
    """
    brain = ctx.context
    
    # Convert context to dict if provided
    context_dict = context_data.model_dump() if context else None
    
    # Run meta-cognitive cycle
    if brain.meta_core:
        result = await brain.meta_core.cognitive_cycle(context_dict)
        return GenericResult(**result)
    
    return GenericResult(error="Meta core not initialized")

@function_tool
async def get_brain_stats(ctx) -> StatsResult:
    """
    Get comprehensive statistics about all systems
    
    Returns:
        Statistics for all systems
    """
    brain = ctx.context
    
    # Get comprehensive stats
    stats = await brain.get_system_stats()
    return StatsResult(stats=SystemStats(**stats))

@function_tool
async def perform_maintenance(ctx) -> MaintenanceResult:
    """
    Run maintenance on all systems
    
    Returns:
        Maintenance results
    """
    brain = ctx.context
    
    # Run maintenance on all systems
    result = await brain.run_maintenance()
    return MaintenanceResult(**result)

@function_tool
async def get_identity_state(ctx) -> IdentityState:
    """
    Get current state of Nyx's identity
    
    Returns:
        Identity state information
    """
    brain = ctx.context
    
    # Get identity state
    result = await brain.get_identity_state()
    return IdentityState(**result)

@function_tool
async def adapt_experience_sharing(ctx, user_id: str, feedback: FeedbackData) -> GenericResult:
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
    result = await brain.adapt_experience_sharing(user_id, feedback.model_dump())
    return GenericResult(**result)

@function_tool
async def run_experience_consolidation(ctx) -> GenericResult:
    """
    Run experience consolidation process
    
    Returns:
        Consolidation results
    """
    brain = ctx.context
    
    # Run consolidation
    result = await brain.run_experience_consolidation()
    return GenericResult(**result)

@function_tool
async def add_procedural_knowledge(ctx, name: str, steps: List[ProcedureStep], domain: str = "general") -> GenericResult:
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
    
    # Convert steps to dicts
    steps_data = [step.model_dump() for step in steps]
    
    # Add procedure
    result = await brain.add_procedure(name, steps_data, domain=domain)
    return GenericResult(**result)

@function_tool
async def run_procedure(ctx, name: str, context: Optional[GenericContext] = None) -> GenericResult:
    """
    Execute a stored procedure
    
    Args:
        name: Name of the procedure to run
        context: Optional execution context
    
    Returns:
        Execution result
    """
    brain = ctx.context
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Execute procedure
    result = await brain.execute_procedure(name, context_dict)
    return GenericResult(**result)

@function_tool
async def analyze_chunking(ctx, procedure_name: str) -> GenericResult:
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
    return GenericResult(**result)

@function_tool
async def register_reflex(ctx, reflex_data: GenericContext) -> GenericResult:
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
        result = await brain.reflexive_system.register_reflex(reflex_data.model_dump())
        return GenericResult(**result)
    
    return GenericResult(error="Reflexive system not initialized")

@function_tool
async def process_stimulus(ctx, stimulus: StimulusData, domain: Optional[str] = None) -> GenericResult:
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
        result = await brain.reflexive_system.process_stimulus_fast(stimulus.model_dump(), domain)
        return GenericResult(**result)
    
    return GenericResult(error="Reflexive system not initialized")

@function_tool
async def enable_self_configuration(ctx) -> GenericResult:
    """
    Enable the self-configuration system
    
    Returns:
        Enablement result
    """
    brain = ctx.context
    
    # Enable self-configuration
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.enable()
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def evaluate_and_adjust_parameters(ctx) -> GenericResult:
    """
    Evaluate current parameters and adjust if needed
    
    Returns:
        Adjustment results
    """
    brain = ctx.context
    
    # Evaluate parameters
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.evaluate_and_adjust_parameters()
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def change_adaptation_strategy(ctx, strategy: str) -> GenericResult:
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
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def get_self_configuration_status(ctx) -> GenericResult:
    """
    Get status of the self-configuration system
    
    Returns:
        Self-configuration status
    """
    brain = ctx.context
    
    # Get status
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.get_status()
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def reset_parameter_to_default(ctx, param_name: str) -> GenericResult:
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
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def process_user_feedback_for_configuration(ctx, 
                                             feedback_type: str, 
                                             feedback_text: str,
                                             context: Optional[GenericContext] = None) -> GenericResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Process feedback
    if hasattr(brain, "self_config_manager") and brain.self_config_manager:
        result = await brain.self_config_manager.process_user_feedback(feedback_type, feedback_text, context_dict)
        return GenericResult(**result)
    
    return GenericResult(error="Self-configuration manager not initialized")

@function_tool
async def set_processing_mode(ctx, mode: str, reason: Optional[str] = None) -> GenericResult:
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
        return GenericResult(**result)
    
    return GenericResult(error="Processing manager not initialized")

@function_tool
async def get_processing_stats(ctx) -> StatsResult:
    """
    Get processing statistics
    
    Returns:
        Processing statistics
    """
    brain = ctx.context
    
    # Get processing stats
    if hasattr(brain, "processing_manager") and brain.processing_manager:
        result = await brain.processing_manager.get_processing_stats()
        return StatsResult(stats=SystemStats(**result))
    
    return StatsResult(error="Processing manager not initialized")

@function_tool
async def initialize_streaming(ctx, video_source=0, audio_source=None) -> GenericResult:
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
        return GenericResult(success=True, streaming_initialized=True)
    
    return GenericResult(error="Streaming initialization not available")

@function_tool
async def process_streaming_event(ctx, event_type: str, event_data: EventData, significance: float = 5.0) -> GenericResult:
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
        result = await brain.process_streaming_event(event_type, event_data.model_dump(), significance)
        return GenericResult(**result)
    
    return GenericResult(error="Streaming event processing not available")

@function_tool
async def run_thinking(ctx, 
                      user_input: str, 
                      thinking_level: int = 1, 
                      context: Optional[GenericContext] = None) -> ProcessingResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Use unified processing with thinking explicitly enabled
    result = await brain.process_input(
        user_input,
        context_dict,
        use_thinking=True,
        thinking_level=thinking_level
    )
    
    # Extract just the thinking results if available
    if result.get("thinking_applied") and "thinking_result" in result:
        return ProcessingResult(
            thinking_applied=True,
            thinking_result=result["thinking_result"],
            thinking_level=thinking_level
        )
    
    return ProcessingResult(error="Thinking was not applied or thinking tools not available")

@function_tool
async def process_with_coordination(ctx, 
                                  user_input: str, 
                                  context: Optional[GenericContext] = None) -> ProcessingResult:
    """
    Process input using the coordinated processing system (a2a)
    
    Args:
        user_input: User's input text
        context: Optional additional context
    
    Returns:
        Coordinated processing results
    """
    brain = ctx.context
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Force coordination mode
    result = await brain.process_input(
        user_input,
        context_dict,
        use_coordination=True,
        mode="coordinated"
    )
    return ProcessingResult(**result)


@function_tool
async def process_simple(ctx, 
                       user_input: str, 
                       context: Optional[GenericContext] = None) -> ProcessingResult:
    """
    Process input using minimal features for fast response
    
    Args:
        user_input: User's input text
        context: Optional additional context
    
    Returns:
        Simple processing results
    """
    brain = ctx.context
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Disable all optional features for speed
    result = await brain.process_input(
        user_input,
        context_dict,
        use_thinking=False,
        use_conditioning=False,
        use_coordination=False,
        mode="serial"
    )
    return ProcessingResult(**result)


@function_tool
async def process_with_all_features(ctx, 
                                  user_input: str, 
                                  context: Optional[GenericContext] = None,
                                  thinking_level: int = 2) -> ProcessingResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    # Enable all features
    result = await brain.generate_response(
        user_input,
        context_dict,
        use_thinking=True,
        use_conditioning=True,
        use_coordination=True,
        use_hierarchical_memory=True,
        thinking_level=thinking_level,
        mode="auto"
    )
    return ProcessingResult(**result)


@function_tool
async def get_processing_capabilities(ctx) -> CapabilitiesResult:
    """
    Get information about available processing capabilities
    
    Returns:
        Available processing features and their status
    """
    brain = ctx.context
    
    capabilities = {
        "thinking": ThinkingCapability(
            available=hasattr(brain, "thinking_config") and brain.thinking_config.get("thinking_enabled", False),
            config=GenericContext(**getattr(brain, "thinking_config", {}))
        ),
        "conditioning": ConditioningCapability(
            available=hasattr(brain, "conditioned_input_processor") and brain.conditioned_input_processor is not None
        ),
        "coordination": CoordinationCapability(
            available=hasattr(brain, "context_distribution") and brain.context_distribution is not None,
            status=CoordinationStatus(**brain.get_context_distribution_status()) if hasattr(brain, "get_context_distribution_status") else None
        ),
        "hierarchical_memory": HierarchicalMemoryCapability(
            available=hasattr(brain, "memory_core") and brain.memory_core is not None
        ),
        "processing_modes": ["auto", "serial", "parallel", "coordinated", "simple"],
        "current_processing_manager": hasattr(brain, "processing_manager") and brain.processing_manager is not None
    }
    
    return CapabilitiesResult(**capabilities)


@function_tool
async def set_thinking_enabled(ctx, enabled: bool) -> GenericResult:
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
        return GenericResult(
            success=True,
            thinking_enabled=enabled,
            config=brain.thinking_config
        )
    
    return GenericResult(error="Thinking configuration not available")


@function_tool
async def compare_processing_modes(ctx, 
                                 user_input: str, 
                                 modes: List[str] = None,
                                 context: Optional[GenericContext] = None) -> ComparisonResult:
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
    
    # Convert context to dict if provided
    context_dict = context.model_dump() if context else None
    
    results_dict = {}
    
    for mode in modes:
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Configure based on mode
            if mode == "simple":
                result = await brain.process_input(
                    user_input, context_dict,
                    use_thinking=False,
                    use_conditioning=False,
                    use_coordination=False,
                    mode="serial"
                )
            elif mode == "coordinated":
                result = await brain.process_input(
                    user_input, context_dict,
                    use_coordination=True,
                    mode="coordinated"
                )
            else:
                result = await brain.process_input(
                    user_input, context_dict,
                    mode=mode
                )
            
            end_time = asyncio.get_event_loop().time()
            
            results_dict[mode] = ModeResult(
                success=True,
                processing_time=end_time - start_time,
                features_used=FeaturesUsed(**result.get("features_used", {})),
                active_modules=len(result.get("active_modules_for_input", [])),
                response_quality_indicators=ResponseQualityIndicators(
                    has_memories=bool(result.get("memories", [])),
                    has_emotional_state=bool(result.get("emotional_state", {})),
                    has_thinking=result.get("thinking_applied", False),
                    has_conditioning=result.get("conditioning_applied", False)
                )
            )
            
        except Exception as e:
            results_dict[mode] = ModeResult(
                success=False,
                error=str(e)
            )
    
    fastest_mode = None
    if any(data.success for data in results_dict.values()):
        fastest_mode = min(
            [(mode, data.processing_time) for mode, data in results_dict.items() if data.success and data.processing_time is not None],
            key=lambda x: x[1]
        )[0]
    
    # Create ComparisonResults dynamically
    comparison_results = ComparisonResults(**results_dict)
    
    return ComparisonResult(
        comparison_results=comparison_results,
        fastest_mode=fastest_mode
    )

@function_tool
async def register_issue(ctx, 
                      issue_data: IssueData, 
                      auto_attempt_resolution: bool = True) -> GenericResult:
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
        result = await brain.issue_tracker.register_issue(issue_data.model_dump(), auto_attempt_resolution)
        return GenericResult(**result)
    
    return GenericResult(error="Issue tracker not initialized")

@function_tool
async def set_meta_tone(ctx, tone_name: str, reason: Optional[str] = None) -> GenericResult:
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
        return GenericResult(**result)
    
    return GenericResult(error="Agent integration not initialized")

@function_tool
async def get_agent_stats(ctx) -> StatsResult:
    """
    Get agent system statistics
    
    Returns:
        Agent statistics
    """
    brain = ctx.context
    
    # Get agent stats
    if hasattr(brain, "agent_integration") and brain.agent_integration:
        result = await brain.agent_integration.get_system_statistics()
        return StatsResult(stats=SystemStats(**result))
    
    return StatsResult(error="Agent integration not initialized")

# Add these to nyx/core/brain/function_tools.py

@function_tool
async def analyze_module(ctx, module_path: str, detailed: bool = False) -> ModuleAnalysisResult:
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
    
    result = await brain.module_optimizer.analyze_module(module_path, detailed)
    return ModuleAnalysisResult(**result)

@function_tool
async def optimize_module(ctx, module_path: str, optimization_goal: str = "general", ensure_backwards_compatible: bool = True) -> ModuleOptimizationResult:
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
    
    result = await brain.module_optimizer.optimize_module(module_path, optimization_goal, ensure_backwards_compatible)
    return ModuleOptimizationResult(**result)

@function_tool
async def create_new_module(ctx, module_name: str, description: str, requirements: List[str] = None, integration_points: List[str] = None) -> ModuleOptimizationResult:
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
    
    result = await brain.module_optimizer.create_new_module(module_name, description, requirements, integration_points)
    return ModuleOptimizationResult(**result)

@function_tool
async def edit_optimized_module(ctx, module_name: str, version: int, edits_description: str) -> ModuleOptimizationResult:
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
    
    result = await brain.module_optimizer.edit_optimized_module(module_name, version, edits_description)
    return ModuleOptimizationResult(**result)

@function_tool
async def get_optimized_module(ctx, module_name: str, version: Optional[int] = None) -> ModuleOptimizationResult:
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
    
    result = brain.module_optimizer.get_optimized_module(module_name, version)
    return ModuleOptimizationResult(**result)

@function_tool
async def list_optimized_modules(ctx) -> GenericResult:
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
    
    result = brain.module_optimizer.list_optimized_modules()
    return GenericResult(**result)

@function_tool
async def import_and_test_module(ctx, module_name: str, version: int) -> GenericResult:
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
    
    result = await brain.module_optimizer.import_and_test_module(module_name, version)
    return GenericResult(**result)

# Add to nyx/core/brain/function_tools.py

@function_tool
async def check_system_health(ctx, detailed: bool = True) -> HealthCheckResult:
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
    
    result = await brain.system_health_checker.check_system_health(detailed)
    return HealthCheckResult(**result)

@function_tool
async def get_system_overview(ctx) -> GenericResult:
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
    
    result = await brain.system_health_checker.get_system_overview()
    return GenericResult(**result)

@function_tool
async def test_function_tool(ctx, tool_name: str) -> GenericResult:
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
    
    result = await brain.system_health_checker.test_function_tool(tool_name)
    return GenericResult(**result)

@function_tool
async def check_agent_capabilities(ctx, agent_name: str) -> GenericResult:
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
    
    result = await brain.system_health_checker.check_agent_capabilities(agent_name)
    return GenericResult(**result)

@function_tool
async def verify_module_imports(ctx, module_names: List[str] = None) -> GenericResult:
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
    
    result = await brain.system_health_checker.verify_module_imports(module_names)
    return GenericResult(**result)

@function_tool
async def get_component_docs(ctx, component_name: str) -> ComponentDocsResult:
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
    
    result = brain.system_health_checker.get_component_docs(component_name)
    return ComponentDocsResult(**result)

@function_tool
async def get_integration_status(ctx) -> GenericResult:
    """
    Get the status of integration components
    
    Returns:
        Integration system status
    """
    brain = ctx.context
    
    if hasattr(brain, "integration_manager") and brain.integration_manager:
        result = await brain.integration_manager.get_integration_status()
        return GenericResult(**result)
    
    return GenericResult(error="Integration manager not initialized")

class EventPublishResult(BaseModel):
    """Result from publishing an event"""
    success: bool
    event_type: Optional[str] = None
    error: Optional[str] = None

@function_tool
async def publish_event(ctx, event_type: str, data: GenericContext, source: str = "brain_agent") -> EventPublishResult:
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
        event = Event(event_type=event_type, source=source, data=data.model_dump())
        await brain.event_bus.publish(event)
        return EventPublishResult(success=True, event_type=event_type)
    
    return EventPublishResult(success=False, error="Event bus not initialized")

@function_tool
async def get_system_context(ctx) -> SystemContextResult:
    """
    Get the current system context state
    
    Returns:
        System context state
    """
    brain = ctx.context
    
    if hasattr(brain, "system_context") and brain.system_context:
        # Return a simplified view of the system context
        context = brain.system_context
        return SystemContextResult(
            cycle_count=context.cycle_count,
            active_goals_count=len(context.active_goals),
            user_models_count=len(context.user_models),
            affective_state=AffectiveStateInfo(
                primary_emotion=context.affective_state.primary_emotion,
                valence=context.affective_state.valence,
                arousal=context.affective_state.arousal
            ),
            body_state=BodyStateInfo(
                has_visual_form=context.body_state.has_visual_form,
                form_description=context.body_state.form_description,
                dominant_sensation=context.body_state.dominant_sensation,
                dominant_region=context.body_state.dominant_region
            )
        )
    
    return SystemContextResult()

@function_tool
async def get_dev_logs(ctx, 
                     log_type: Optional[str] = None,
                     limit: int = 10,
                     tags: Optional[List[str]] = None) -> LogsResult:
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
    
    # Convert logs to LogEntry models
    log_entries = [LogEntry(**log) for log in logs]
    
    return LogsResult(
        logs=log_entries,
        count=len(log_entries),
        has_more=len(log_entries) == limit
    )

@function_tool
async def get_synergy_stats(ctx) -> StatsResult:
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
    
    stats = await brain.dev_log_storage.get_recommendation_stats()
    return StatsResult(stats=SystemStats(**stats))
