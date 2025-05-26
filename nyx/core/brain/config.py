# nyx/core/brain/config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class ComponentConfig:
    """Base configuration for a component"""
    enabled: bool = True
    initialization_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BrainConfig:
    """Complete configuration for NyxBrain with all components"""
    
    # === Core Emotional & Hormonal Systems ===
    emotional_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    hormone_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    mood_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Memory Systems ===
    memory_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    memory_orchestrator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    procedural_memory_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    agent_enhanced_memory: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    recognition_memory: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    creative_memory: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Cognitive & Reasoning Systems ===
    reasoning_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    knowledge_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    meta_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    reflection_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    prediction_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    imagination_simulator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Attention & Perception ===
    attentional_controller: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    multimodal_integrator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    context_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    temporal_perception: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    passive_observation_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Identity & Self-Model ===
    identity_evolution: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    body_image: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))  # Often disabled
    autobiographical_narrative: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Needs & Goals ===
    needs_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    goal_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    reward_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Social & Relationship Systems ===
    relationship_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    theory_of_mind: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    cross_user_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Experience & Adaptation ===
    experience_interface: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    experience_consolidation: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    dynamic_adaptation: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    internal_feedback: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Somatic & Sensory ===
    digital_somatosensory_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    conditioning_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    conditioning_maintenance: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    conditioned_input_processor: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Spatial Systems ===
    spatial_mapper: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    spatial_memory: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    map_visualization: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    navigator_agent: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Communication & Interaction ===
    proactive_communication_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    thoughts_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    mode_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    mode_integration: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    agentic_action_generator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Femdom & Dominance Systems ===
    femdom_coordinator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    protocol_enforcement: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    body_service_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    psychological_dominance: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    orgasm_control_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    dominance_persona_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    sadistic_response_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    submission_progression: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    task_assignment_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    femdom_integration_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    dominance_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    general_dominance_ideation_agent: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    hard_dominance_ideation_agent: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Creative Systems ===
    novelty_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    creative_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    content_store: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    capability_model: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    capability_assessor: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Processing & Orchestration ===
    processing_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    brain_agent: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    agent_evaluator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    parallel_executor: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    reflexive_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    distributed_processing: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === System Management ===
    integration_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    sync_daemon: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    module_optimizer: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    system_health_checker: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    self_config_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    checkpoint_planner: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    issue_tracking_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === Global Workspace ===
    workspace_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        enabled=True,
        initialization_params={
            "hz": 10.0,
            "enable_unconscious": True
        }
    ))
    
    # === Thinking & Analysis Tools ===
    thinking_tools: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    thinking_config: Dict[str, Any] = field(default_factory=lambda: {
        "thinking_enabled": True,
        "last_thinking_interaction": 0,
        "thinking_stats": {
            "total_thinking_used": 0,
            "basic_thinking_used": 0,
            "moderate_thinking_used": 0,
            "deep_thinking_used": 0,
            "thinking_time_avg": 0.0
        }
    })
    
    # === Streaming & Gaming Systems ===
    streaming_core: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        enabled=False,  # Disabled by default, enabled via ENABLE_STREAMING env var
        initialization_params={
            "video_source": 0,
            "audio_source": None
        }
    ))
    streaming_hormone_system: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    streaming_reflection_engine: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    streaming_integration: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    cross_game_knowledge: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    game_vision: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    gamer_girl: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    game_state: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    speech_recognition: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    game_learning_manager: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    game_multimodal_integrator: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    audience_interaction: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=False))
    
    # === Storage & Logging ===
    dev_log_storage: ComponentConfig = field(default_factory=lambda: ComponentConfig(enabled=True))
    
    # === A2A Integration ===
    use_a2a_integration: bool = True  # Enable context-aware wrappers
    
    # === Performance & Behavior Settings ===
    cross_user_enabled: bool = True
    cross_user_sharing_threshold: float = 0.7
    memory_to_emotion_influence: float = 0.3
    emotion_to_memory_influence: float = 0.4
    experience_to_identity_influence: float = 0.2
    consolidation_interval: int = 24  # Hours
    identity_reflection_interval: int = 10  # Interactions
    need_drive_threshold: float = 0.4
    
    # === Context Configuration ===
    context_config: Dict[str, Any] = field(default_factory=lambda: {
        "focus_limit": 4,
        "background_limit": 3,
        "zoom_in_limit": 2,
        "high_fidelity_threshold": 0.7,
        "med_fidelity_threshold": 0.5,
        "low_fidelity_threshold": 0.3,
        "max_context_tokens": 3500
    })
    
    @classmethod
    def default(cls):
        """Returns default configuration with all systems enabled"""
        return cls()
    
    @classmethod
    def minimal(cls):
        """Returns minimal configuration with only essential systems"""
        config = cls()
        # Disable non-essential systems
        config.body_image.enabled = False
        config.streaming_core.enabled = False
        config.streaming_hormone_system.enabled = False
        config.streaming_reflection_engine.enabled = False
        config.streaming_integration.enabled = False
        config.cross_game_knowledge.enabled = False
        config.game_vision.enabled = False
        config.gamer_girl.enabled = False
        config.game_state.enabled = False
        config.speech_recognition.enabled = False
        config.game_learning_manager.enabled = False
        config.game_multimodal_integrator.enabled = False
        config.audience_interaction.enabled = False
        config.distributed_processing.enabled = False
        return config
    
    @classmethod
    def streaming_enabled(cls):
        """Returns configuration with streaming systems enabled"""
        config = cls.default()
        config.streaming_core.enabled = True
        config.streaming_hormone_system.enabled = True
        config.streaming_reflection_engine.enabled = True
        config.streaming_integration.enabled = True
        config.cross_game_knowledge.enabled = True
        config.game_vision.enabled = True
        config.gamer_girl.enabled = True
        config.game_state.enabled = True
        config.speech_recognition.enabled = True
        config.game_learning_manager.enabled = True
        config.game_multimodal_integrator.enabled = True
        config.audience_interaction.enabled = True
        return config
    
    def get_component_config(self, component_name: str) -> Optional[ComponentConfig]:
        """Get configuration for a specific component"""
        return getattr(self, component_name, None)
    
    def is_enabled(self, component_name: str) -> bool:
        """Check if a component is enabled"""
        config = self.get_component_config(component_name)
        return config.enabled if config else False
