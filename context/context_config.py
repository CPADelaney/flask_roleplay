# context/context_config.py

"""
Configuration settings for the context optimization system.

This module provides a centralized place to configure the behavior
of the context optimization system.
"""

import os
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # Cache settings
    "cache": {
        "enabled": True,
        "l1_ttl_seconds": 60,       # 1 minute
        "l2_ttl_seconds": 300,      # 5 minutes
        "l3_ttl_seconds": 86400,    # 24 hours
        "l1_max_size": 100,
        "l2_max_size": 500,
        "l3_max_size": 2000
    },
    
    # Vector database settings
    "vector_db": {
        "enabled": True,
        "db_type": "in_memory",     # Options: "in_memory", "qdrant", "pinecone", "milvus"
        "url": "http://localhost:6333",
        "api_key": None,
        "environment": None,
        "dimension": 384
    },
    
    # Token budget settings
    "token_budget": {
        "default_budget": 4000,     # Default token budget
        "npcs_percent": 30,         # Percentage of budget for NPCs
        "memories_percent": 20,     # Percentage of budget for memories
        "conflicts_percent": 15,    # Percentage of budget for conflicts
        "quests_percent": 15,       # Percentage of budget for quests
        "base_percent": 20          # Percentage of budget for base context
    },
    
    # Memory consolidation settings
    "memory_consolidation": {
        "enabled": True,
        "days_threshold": 7,        # Consolidate memories older than this many days
        "min_memories_to_consolidate": 5,
        "consolidation_interval_hours": 24
    },
    
    # Predictive preloading settings
    "preloading": {
        "enabled": True,
        "max_locations": 3,         # Maximum number of locations to preload
        "prediction_threshold": 0.3 # Minimum prediction score to trigger preloading
    },
    
    # Performance settings
    "performance": {
        "background_processing": True,
        "retry_attempts": 3,
        "retry_delay_seconds": 1,
        "async_preloading": True
    },
    
    # Integration settings
    "integration": {
        "patch_existing_functions": False,
        "add_to_maintenance": True,
        "maintenance_schedule": "daily"
    },
    
    # Feature flags
    "features": {
        "use_incremental_context": True,
        "use_vector_search": True,
        "use_temporal_decay": True,
        "use_attention_tracking": True,
        "track_access_patterns": True
    }
}

class ContextConfig:
    """
    Configuration manager for the context optimization system.
    
    This class manages loading, accessing, and updating configuration
    settings for the context optimization system.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement as singleton"""
        if cls._instance is None:
            cls._instance = super(ContextConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize with default config if not already initialized"""
        if self._initialized:
            return
            
        self.config = DEFAULT_CONFIG.copy()
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment or config file"""
        # Check for config file
        config_path = os.environ.get("CONTEXT_CONFIG_PATH", "config/context_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                
                # Update config with file values
                self._update_nested_dict(self.config, file_config)
                logger.info(f"Loaded context configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        # Check for environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Check for flat environment variables like CONTEXT_CACHE_ENABLED
        prefix = "CONTEXT_"
        
        for key in os.environ:
            if key.startswith(prefix):
                try:
                    # Split the key into parts (CONTEXT_CACHE_ENABLED -> ["CACHE", "ENABLED"])
                    parts = key[len(prefix):].lower().split("_")
                    
                    if len(parts) >= 2:
                        section = parts[0]
                        setting = "_".join(parts[1:])
                        
                        # Convert value to appropriate type
                        value = os.environ[key]
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            value = float(value)
                        
                        # Update config
                        if section in self.config and setting in self.config[section]:
                            self.config[section][setting] = value
                            logger.debug(f"Updated config from env: {section}.{setting} = {value}")
                except Exception as e:
                    logger.warning(f"Error processing environment variable {key}: {e}")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary recursively"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get(self, section: str, setting: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section (e.g., "cache", "vector_db")
            setting: Setting name within the section
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if section in self.config and setting in self.config[section]:
            return self.config[section][setting]
        return default
    
    def set(self, section: str, setting: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section (e.g., "cache", "vector_db")
            setting: Setting name within the section
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][setting] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Configuration section (e.g., "cache", "vector_db")
            
        Returns:
            Dictionary with section settings
        """
        return self.config.get(section, {}).copy()
    
    def is_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name (e.g., "use_vector_search")
            
        Returns:
            True if enabled, False otherwise
        """
        return self.get("features", feature, False)
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """
        Get vector database configuration in the format expected by RPGEntityManager.
        
        Returns:
            Dictionary with vector database configuration
        """
        vector_section = self.get_section("vector_db")
        
        return {
            "db_type": vector_section.get("db_type", "in_memory"),
            "url": vector_section.get("url"),
            "api_key": vector_section.get("api_key"),
            "environment": vector_section.get("environment")
        }
    
    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration (default from environment or "config/context_config.json")
            
        Returns:
            True if saved successfully, False otherwise
        """
        if config_path is None:
            config_path = os.environ.get("CONTEXT_CONFIG_PATH", "config/context_config.json")
        
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved context configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False

# Singleton instance
config = ContextConfig()

def get_config() -> ContextConfig:
    """
    Get the singleton configuration instance.
    
    Returns:
        ContextConfig instance
    """
    return config
