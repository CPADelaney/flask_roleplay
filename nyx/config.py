# nyx/config.py

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # Memory System Configuration
    "memory": {
        "cache_ttl": 300,                 # Cache TTL in seconds (5 minutes)
        "max_memories_per_query": 10,     # Maximum memories to return per query
        "default_significance": 5,        # Default significance for new memories (1-10)
        "memory_decay_rate": 0.1,         # Rate at which memory significance decays
        "maintenance_interval": 86400,    # Memory maintenance interval in seconds (24 hours)
        "minimal_similarity_threshold": 0.65,  # Threshold for memory similarity (0.0-1.0)
        "embedding_model": "text-embedding-ada-002"  # Model to use for embeddings
    },
    
    # Decision Engine Configuration
    "decision_engine": {
        "default_temperature": 0.7,       # Default temperature for response generation
        "max_tokens": 1000,               # Maximum tokens for generated responses
        "model_name": "gpt-4.1-nano",            # Model to use for decision engine
        "instruction_depth": "detailed",  # Instruction depth (minimal, moderate, detailed)
        "response_guidance_weight": 0.8,  # Weight given to user model guidance (0.0-1.0)
        "memory_weight": 0.7              # Weight given to memories in decision-making (0.0-1.0)
    },
    
    # User Model Configuration
    "user_model": {
        "preference_detection_threshold": 0.6,  # Threshold for detecting preferences
        "preference_level_thresholds": [1, 3, 5, 10],  # Thresholds for preference levels
        "behavior_pattern_thresholds": [2, 5, 8, 12],  # Thresholds for behavior patterns
        "model_update_frequency": "every_interaction",  # How often to update model
        "confidence_scaling": 0.8         # Scale factor for confidence in user preferences
    },
    
    # Narrative Configuration
    "narrative": {
        "arc_progression_step": 5,        # Percentage to advance narrative arcs per relevant action
        "min_arc_completion_threshold": 85,  # Threshold for considering an arc complete
        "crossroads_check_interval": 3,   # How often to check for crossroads (interactions)
        "ritual_check_interval": 5        # How often to check for rituals (interactions)
    },
    
    # System Configuration
    "system": {
        "log_level": "INFO",              # Logging level
        "enable_performance_tracking": True,  # Track performance metrics
        "enable_caching": True,           # Enable caching
        "maintenance_threads": 2,         # Number of threads for maintenance tasks
        "max_parallel_requests": 5,       # Maximum parallel LLM requests
        "timeout": 30                     # Request timeout in seconds
    }
}

# Environment variable prefix for configuration
ENV_PREFIX = "NYX_"

def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment variable overrides.
    
    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for section_key, section_values in config.items():
        for key, value in section_values.items():
            env_var = f"{ENV_PREFIX}{section_key.upper()}_{key.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                # Convert environment variable to appropriate type
                if isinstance(value, bool):
                    config[section_key][key] = env_value.lower() in ("true", "yes", "1")
                elif isinstance(value, int):
                    config[section_key][key] = int(env_value)
                elif isinstance(value, float):
                    config[section_key][key] = float(env_value)
                else:
                    config[section_key][key] = env_value
    
    return config

# Global configuration instance
CONFIG = get_config()

def get_memory_config() -> Dict[str, Any]:
    """Get memory-specific configuration."""
    return CONFIG["memory"]

def get_decision_engine_config() -> Dict[str, Any]:
    """Get decision engine-specific configuration."""
    return CONFIG["decision_engine"]

def get_user_model_config() -> Dict[str, Any]:
    """Get user model-specific configuration."""
    return CONFIG["user_model"]

def get_narrative_config() -> Dict[str, Any]:
    """Get narrative-specific configuration."""
    return CONFIG["narrative"]

def get_system_config() -> Dict[str, Any]:
    """Get system-specific configuration."""
    return CONFIG["system"]

def update_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        new_config: New configuration values
        
    Returns:
        Updated configuration
    """
    global CONFIG
    
    # Update configuration recursively
    for section_key, section_values in new_config.items():
        if section_key in CONFIG:
            if isinstance(section_values, dict):
                for key, value in section_values.items():
                    if key in CONFIG[section_key]:
                        CONFIG[section_key][key] = value
            else:
                CONFIG[section_key] = section_values
    
    return CONFIG
