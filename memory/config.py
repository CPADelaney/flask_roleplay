# memory/config.py

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("memory_config")

# Default configuration values
DEFAULT_CONFIG = {
    "database": {
        "dsn": os.getenv("DB_DSN") or os.getenv("DATABASE_URL"),
        "min_connections": 5,
        "max_connections": 20,
        "command_timeout": 60,
        "statement_cache_size": 100,
        "max_inactive_connection_lifetime": 300
    },
    "embedding": {
        "model": "text-embedding-ada-002", 
        "batch_size": 5,
        "fallback_model": "all-MiniLM-L6-v2"
    },
    "memory": {
        "cache_ttl": 300,  # 5 minutes
        "default_significance": 3,  # Medium
        "default_emotional_intensity": 30,
        "consolidation_threshold_days": 7,
        "decay_rate": 0.2,
        "archive_threshold_days": 60
    },
    "emotional": {
        "default_intensity": 0.5,
        "emotional_decay_rate": 0.1,
        "trauma_threshold": 0.7
    },
    "mask": {
        "default_integrity": 100,
        "reveal_probability": 0.15
    },
    "openai": {
        "model": "gpt-4.1-nano",
        "temperature": 0.4,
        "max_tokens": 200
    },
    "performance": {
        "telemetry_enabled": True,
        "telemetry_retention_days": 30,
        "background_maintenance_interval": 3600,  # 1 hour
        "log_slow_operations": True,
        "slow_operation_threshold_ms": 500
    }
}

# Global configuration object
config = DEFAULT_CONFIG.copy()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, environment variables, or use defaults.
    
    Args:
        config_path: Path to JSON configuration file (optional)
        
    Returns:
        Configuration dictionary
    """
    global config
    
    # Start with default config
    new_config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Deep merge file config with defaults
            _deep_merge(new_config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Override with environment variables
    # Format: MEMORY_SECTION_KEY (e.g., MEMORY_DATABASE_DSN)
    for env_var, value in os.environ.items():
        if env_var.startswith('MEMORY_'):
            parts = env_var.lower().split('_')
            if len(parts) >= 3:
                section = parts[1]
                key = '_'.join(parts[2:])
                
                if section in new_config and key in new_config[section]:
                    # Convert value to appropriate type based on default
                    default_value = new_config[section][key]
                    if isinstance(default_value, bool):
                        new_config[section][key] = value.lower() in ('true', 'yes', '1')
                    elif isinstance(default_value, int):
                        new_config[section][key] = int(value)
                    elif isinstance(default_value, float):
                        new_config[section][key] = float(value)
                    else:
                        new_config[section][key] = value
                    
                    logger.info(f"Configuration overridden by environment variable {env_var}")
    
    # Update global config
    config = new_config
    
    # Create convenience shortcuts for commonly used sections
    global DB_CONFIG, EMBEDDING_CONFIG, MEMORY_CONFIG, OPENAI_CONFIG
    DB_CONFIG = config["database"]
    EMBEDDING_CONFIG = config["embedding"]
    MEMORY_CONFIG = config["memory"]
    OPENAI_CONFIG = config["openai"]
    
    return config

def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Deep merge source dictionary into target.
    Only merges keys that already exist in target.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        elif key in target:
            target[key] = value

# Convenience shortcuts for commonly used sections
DB_CONFIG = config["database"]
EMBEDDING_CONFIG = config["embedding"]
MEMORY_CONFIG = config["memory"]
OPENAI_CONFIG = config["openai"]
