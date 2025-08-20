# nyx/nyx_agent/__init__.py

"""Nyx Agent SDK Package

This package contains the modularized components of the Nyx Agent system.
"""

# Core configuration
from .config import Config

# Models
from .models import (
    # Base
    BaseModel, StrictBaseModel,
    KVPair, KVList,
    dict_to_kvlist, kvlist_to_dict,
    strict_output,
    
    # Core Data Models
    MemoryItem, EmotionalChanges, ScoreComponents,
    PerformanceNumbers, ConflictItem, InstabilityItem,
    ActivityRec, RelationshipStateOut, RelationshipChanges,
    DecisionMetadata, DecisionOption, ScoredOption,
    
    # Structured Output Models
    NarrativeResponse, MemoryReflection, ContentModeration,
    EmotionalStateUpdate, ScenarioDecision, RelationshipUpdate,
    ImageGenerationDecision,
    
    # Input Models
    RetrieveMemoriesInput, AddMemoryInput, DetectUserRevelationsInput,
    GenerateImageFromSceneInput, CalculateEmotionalStateInput,
    UpdateRelationshipStateInput, GetActivityRecommendationsInput,
    BeliefDataModel, ManageBeliefsInput, ScoreDecisionOptionsInput,
    DetectConflictsAndInstabilityInput, GenerateUniversalUpdatesInput,
    DecideImageInput, EmptyInput,
    
    # Output Models
    MemorySearchResult, MemoryStorageResult, UserGuidanceResult,
    RevelationDetectionResult, ImageGenerationResult,
    EmotionalCalculationResult, RelationshipUpdateResult,
    PerformanceMetricsResult, ActivityRecommendationsResult,
    BeliefManagementResult, DecisionScoringResult,
    ConflictDetectionResult, UniversalUpdateResult,
    
    # State Models
    EmotionalState, RelationshipState, PerformanceMetrics, LearningMetrics,
    
    # Open World Models
    NarrateSliceInput, EmergentEventInput, SimulateAutonomyInput,
    
    # Composite Models
    ScenarioManagementRequest, RelationshipInteractionData,
)

# Context
from .context import NyxContext

try:
    from .tools import (
        # Memory Tools
        retrieve_memories,
        add_memory,
        
        # User Model Tools
        get_user_model_guidance,
        detect_user_revelations,
        
        # Emotional Tools
        calculate_and_update_emotional_state,
        calculate_emotional_impact,
        
        # Relationship Tools
        update_relationship_state,
        
        # Performance and Activity Tools
        check_performance_metrics,
        get_activity_recommendations,
        
        # Belief and Decision Tools
        manage_beliefs,
        score_decision_options,
        detect_conflicts_and_instability,
        
        # Image and Updates Tools
        decide_image_generation,  # <-- Changed from decide_image_generation_standalone
        generate_image_from_scene,
        generate_universal_updates,
        generate_universal_updates_impl,
        
        # Open World / Slice-of-life Tools
        tool_narrate_slice_of_life_scene,
        orchestrate_slice_scene,
        check_world_state,
        generate_emergent_event,
        simulate_npc_autonomy,
        generate_npc_dialogue,
        narrate_power_exchange,
        narrate_daily_routine,
        generate_ambient_narration,
        detect_narrative_patterns,
    )
except ImportError as e:
    import logging
    logging.warning(f"Some tools could not be imported: {e}")

# Agents
from .agents import (
    DEFAULT_MODEL_SETTINGS,
    memory_agent,
    analysis_agent,
    emotional_agent,
    visual_agent,
    activity_agent,
    performance_agent,
    scenario_agent,
    belief_agent,
    decision_agent,
    reflection_agent,
    nyx_main_agent,
)

# Assembly
from .assembly import (
    assemble_nyx_response,
    resolve_scene_requests,
)

# Utils
from .utils import (
    # Process safety
    safe_psutil,
    safe_process_metric,
    get_process_info,
    bytes_to_mb,
    
    # Token and response extraction
    extract_token_usage,
    extract_runner_response,
    
    # Context helpers
    get_context_text_lower,
    get_canonical_context,
    
    # Tool schema helpers
    sanitize_agent_tools_in_place,
    debug_strict_schema_for_agent,
    log_strict_hits,
    force_fix_tool_parameters,
    assert_no_required_leaks,
    
    # Response helpers
    run_compat,
    
    # Decision scoring helpers
    should_generate_task,
    enhance_context_with_memories,
    get_available_activities,
    
    # World state helpers
    add_nyx_hosting_style,
    calculate_world_tension,
    should_generate_image_for_scene,
    detect_emergent_opportunities,
)

# Orchestrator - Main runtime functions
from .orchestrator import (
    process_user_input,
    generate_reflection,
    manage_scenario,
    manage_relationships,
    store_messages,
    run_agent_safely,
    run_agent_with_error_handling,
    decide_image_generation_standalone,  # <-- Import this from orchestrator
)

# Agent Factory - Agent creation utilities
from .agent_factory import (
    create_nyx_agent_with_prompt,
    create_preset_aware_nyx_agent,
)

# Guardrails
from .guardrails import (
    content_moderation_guardrail,
)

# Compatibility layer
from .compatibility import (
    AgentContext,
    
    # Legacy function mappings
    retrieve_memories_impl,
    add_memory_impl,
    get_user_model_guidance_impl,
    detect_user_revelations_impl,
    generate_image_from_scene_impl,
    calculate_emotional_impact_impl,
    calculate_and_update_emotional_state_impl,
    manage_beliefs_impl,
    score_decision_options_impl,
    detect_conflicts_and_instability_impl,
    get_emotional_state,
    update_emotional_state,
    enhance_context_with_strategies,
    determine_image_generation,
    generate_base_response,
    mark_strategy_for_review,
    process_user_input_with_openai,
    process_user_input_standalone,
)

__all__ = [
    # Configuration
    'Config',
    
    # Context classes
    'NyxContext',
    'AgentContext',  # Legacy compatibility
    
    # Main agent
    'nyx_main_agent',
    'DEFAULT_MODEL_SETTINGS',
    
    # Sub-agents
    'memory_agent',
    'analysis_agent',
    'emotional_agent',
    'visual_agent',
    'activity_agent',
    'performance_agent',
    'scenario_agent',
    'belief_agent',
    'decision_agent',
    'reflection_agent',
    
    # Assembly functions
    'assemble_nyx_response',
    'resolve_scene_requests',
    
    # Main orchestration functions
    'process_user_input',
    'generate_reflection',
    'manage_scenario',
    'manage_relationships',
    'store_messages',
    
    # Agent creation functions
    'create_nyx_agent_with_prompt',
    'create_preset_aware_nyx_agent',
    
    # Error handling
    'run_agent_safely',
    'run_agent_with_error_handling',
    
    # Guardrails
    'content_moderation_guardrail',
    
    # Output models
    'NarrativeResponse',
    'ImageGenerationDecision',
    
    # Tool functions (for advanced usage)
    'retrieve_memories',
    'add_memory',
    'get_user_model_guidance',
    'detect_user_revelations',
    'generate_image_from_scene',
    'decide_image_generation',
    'decide_image_generation_standalone', 
    'calculate_emotional_impact',
    'calculate_and_update_emotional_state',
    'update_relationship_state',
    'check_performance_metrics',
    'get_activity_recommendations',
    'manage_beliefs',
    'score_decision_options',
    'detect_conflicts_and_instability',
    'generate_universal_updates',
    'orchestrate_slice_scene',
    'check_world_state',
    'generate_emergent_event',
    'simulate_npc_autonomy',
    
    # Helper functions
    'run_compat',
    'enhance_context_with_memories',
    'get_available_activities',
    'sanitize_agent_tools_in_place',
    'log_strict_hits',
    
    # Compatibility functions (deprecated)
    'enhance_context_with_strategies',
    'determine_image_generation',
    'process_user_input_with_openai',
    'process_user_input_standalone',
]
