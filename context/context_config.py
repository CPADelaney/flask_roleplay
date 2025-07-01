# context/context_config.py

"""
Configuration settings for the context optimization system.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional
from agents import ModelSettings

logger = logging.getLogger(__name__)

# Default configuration values focused on essential settings
DEFAULT_CONFIG = {
    # Cache settings
    "cache": {
        "enabled": True,
        "ttl_seconds": {
            "l1": 60,   # 1 minute
            "l2": 300,  # 5 minutes
            "l3": 1800  # 30 minutes
        },
        "max_size": {
            "l1": 100,
            "l2": 500,
            "l3": 2000
        }
    },
    
    # Vector database settings
    "vector_db": {
        "enabled": True,
        "db_type": "in_memory",  # Options: "in_memory", "qdrant"
        "url": "http://localhost:6333",
        "api_key": None,
        "dimension": 384
    },
    
    # Token budget settings
    "token_budget": {
        "default": 4000,  # Default token budget
        "allocation": {
            "npcs": 30,         # Percentage for NPCs
            "memories": 20,     # Percentage for memories
            "location": 15,     # Percentage for location
            "quests": 15,       # Percentage for quests
            "base": 20          # Percentage for base context
        }
    },
    
    # Feature flags
    "features": {
        "use_vector_search": True,
        "use_memory_system": True,
        "use_delta_updates": True,
        "track_performance": True
    },
    
    # Agent SDK settings
    "agent_sdk": {
        "default_model": "gpt-4.1-nano",
        "model_settings": {
            "temperature": 0.1,
            "top_p": 0.9
        },
        "workflow_name": "context_optimization",
        "trace_include_sensitive_data": False
    }
}

class ContextConfig:
    """
    Configuration manager for the context optimization system.
    
    Provides a simple interface for accessing configuration settings.
    """
    
    _instance = None
    _initialized = False
    _init_lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls):
        """Get the singleton instance asynchronously"""
        if cls._instance is None:
            async with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        elif not cls._instance._initialized:
            await cls._instance.initialize()
        return cls._instance
    
    def __init__(self):
        """Initialize with default configuration"""
        self.config = DEFAULT_CONFIG.copy()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the configuration asynchronously"""
        if not self._initialized:
            await self._load_config()
            self._initialized = True
    
    async def _load_config(self):
        """Load configuration from environment or config file asynchronously"""
        # Check for config file
        config_path = os.environ.get("CONTEXT_CONFIG_PATH", "config/context_config.json")
        
        if os.path.exists(config_path):
            try:
                # Use asyncio.to_thread for file I/O to avoid blocking event loop
                loop = asyncio.get_event_loop()
                file_content = await loop.run_in_executor(None, self._read_config_file, config_path)
                
                if file_content:
                    file_config = json.loads(file_content)
                    # Update config with file values
                    self._update_nested_dict(self.config, file_config)
                    logger.info(f"Loaded context configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        # Check for environment variables
        await self._load_from_env()
    
    def _read_config_file(self, config_path):
        """Read the config file synchronously (called via run_in_executor)"""
        try:
            with open(config_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading config file {config_path}: {e}")
            return None
    
    async def _load_from_env(self):
        """Load configuration from environment variables asynchronously"""
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
                        if section in self.config:
                            if setting in self.config[section]:
                                self.config[section][setting] = value
                            elif isinstance(self.config[section], dict):
                                # Try to handle nested settings
                                for subsection, subsettings in self.config[section].items():
                                    if isinstance(subsettings, dict) and setting in subsettings:
                                        self.config[section][subsection][setting] = value
                                        break
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
        if section in self.config:
            if setting in self.config[section]:
                return self.config[section][setting]
            
            # Try to handle nested settings
            if isinstance(self.config[section], dict):
                for subsection, subsettings in self.config[section].items():
                    if isinstance(subsettings, dict) and setting in subsettings:
                        return subsettings[setting]
        
        return default
    
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
        Get vector database configuration.
        
        Returns:
            Dictionary with vector database configuration
        """
        vector_section = self.get_section("vector_db")
        
        return {
            "db_type": vector_section.get("db_type", "in_memory"),
            "url": vector_section.get("url"),
            "api_key": vector_section.get("api_key"),
            "dimension": vector_section.get("dimension", 384)
        }
    
    def get_token_budget(self, content_type: str = None) -> int:
        """
        Get the token budget for a specific content type.
        
        Args:
            content_type: Type of content (e.g., "npcs", "memories")
            
        Returns:
            Token budget
        """
        total_budget = self.get("token_budget", "default", 4000)
        
        if content_type is None:
            return total_budget
        
        # Get percentage allocation for the content type
        allocations = self.config.get("token_budget", {}).get("allocation", {})
        percentage = allocations.get(content_type, 0) / 100.0
        
        # Calculate budget
        return int(total_budget * percentage)
    
    def get_model_settings(self) -> ModelSettings:
        """
        Get model settings for Agents SDK.
        
        Returns:
            ModelSettings object for the OpenAI Agents SDK
        """
        settings = self.get_section("agent_sdk").get("model_settings", {})
        return ModelSettings(
            temperature=settings.get("temperature", 0.1),
            top_p=settings.get("top_p", 0.9),
            max_tokens=settings.get("max_tokens", None),
            presence_penalty=settings.get("presence_penalty", None),
            frequency_penalty=settings.get("frequency_penalty", None),
        )
    
    def get_default_model(self) -> str:
        """
        Get default model for Agents SDK.
        
        Returns:
            Model name string
        """
        return self.get_section("agent_sdk").get("default_model", "gpt-4.1-nano")

    def get_render_config() -> Dict[str, Any]:
        """
        Get Render-specific configuration settings.
        Automatically detects Render environment and configures accordingly.
        """
        is_render = os.environ.get('RENDER', '').lower() == 'true'
        
        # Base configuration
        config = {
            "is_render": is_render,
            "environment": os.environ.get('RENDER_ENV', 'production'),
            
            # Vector database configuration
            "vector_db": {
                "enabled": True,
                "db_type": "qdrant",  # Using Qdrant as the vector database
                "url": os.environ.get('QDRANT_URL', 'http://localhost:6333'),
                "api_key": os.environ.get('QDRANT_API_KEY'),
                "dimension": 384,  # Consistent with existing configuration
                
                # Render-specific settings
                "connection_timeout": 30,  # seconds
                "operation_timeout": 15,  # seconds
                "max_retries": 3,
                "retry_delay": 1,  # seconds
                
                # Connection pool settings
                "pool_size": int(os.environ.get('QDRANT_POOL_SIZE', '10')),
                "max_connections": int(os.environ.get('QDRANT_MAX_CONNECTIONS', '20')),
                
                # Performance optimization for Render
                "prefer_grpc": True,  # Use gRPC when available
                "optimize_for_render": True
            },
            
            # Cache settings optimized for Render
            "cache": {
                "enabled": True,
                "ttl_seconds": {
                    "l1": 300,    # 5 minutes
                    "l2": 1800,   # 30 minutes
                    "l3": 7200    # 2 hours
                },
                "max_size": {
                    "l1": 1000,   # Increased for better performance
                    "l2": 5000,
                    "l3": 10000
                }
            }
        }
        
        # Render-specific optimizations
        if is_render:
            # Adjust vector database settings based on Render environment
            if os.environ.get('RENDER_ENV') == 'production':
                config['vector_db'].update({
                    "pool_size": int(os.environ.get('QDRANT_POOL_SIZE', '20')),
                    "max_connections": int(os.environ.get('QDRANT_MAX_CONNECTIONS', '40')),
                    "connection_timeout": 45,  # Increased for production
                    "operation_timeout": 30
                })
            
            # Configure automatic cleanup
            config['vector_db']['cleanup'] = {
                "enabled": True,
                "interval": 3600,  # 1 hour
                "max_age": 86400 * 30  # 30 days
            }
        
        return config
    
    def configure_render_vector_db():
        """
        Configure vector database specifically for Render deployment.
        Sets up necessary environment variables and optimizations.
        """
        config = get_render_config()
        
        if not config['is_render']:
            return config
        
        # Set up environment variables if not already set
        if 'QDRANT_URL' not in os.environ:
            os.environ['QDRANT_URL'] = config['vector_db']['url']
        
        if 'QDRANT_API_KEY' not in os.environ:
            os.environ['QDRANT_API_KEY'] = config['vector_db'].get('api_key', '')
        
        return config


# Async singleton access function
async def get_config() -> ContextConfig:
    """
    Get the singleton configuration instance asynchronously.
    
    Returns:
        ContextConfig instance
    """
    return await ContextConfig.get_instance()
