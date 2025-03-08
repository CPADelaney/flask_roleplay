# config.py

"""
Centralized configuration system for NPC agents and related systems.
Uses environment variables with sensible defaults.
"""

import os
import logging
from typing import Dict, Any

class Config:
    """Configuration management with environment and file-based configuration."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # NPC Configuration
        self.NPC_BATCH_SIZE = int(os.getenv("NPC_BATCH_SIZE", "3"))
        self.NPC_DAILY_ACTIVITY_LIMIT = int(os.getenv("NPC_DAILY_ACTIVITY_LIMIT", "5"))
        self.NPC_MEMORY_LIMIT = int(os.getenv("NPC_MEMORY_LIMIT", "50"))
        self.NPC_MASK_SLIP_THRESHOLD = float(os.getenv("NPC_MASK_SLIP_THRESHOLD", "0.15"))
        
        # Memory Configuration
        self.MEMORY_LIFECYCLE_DAYS = int(os.getenv("MEMORY_LIFECYCLE_DAYS", "14"))
        self.MEMORY_SIGNIFICANCE_THRESHOLD = int(os.getenv("MEMORY_SIGNIFICANCE_THRESHOLD", "3"))
        self.MEMORY_DECAY_RATE = float(os.getenv("MEMORY_DECAY_RATE", "0.2"))
        self.MEMORY_CONSOLIDATION_THRESHOLD = int(os.getenv("MEMORY_CONSOLIDATION_THRESHOLD", "3"))
        self.MEMORY_CONSOLIDATION_DAYS = int(os.getenv("MEMORY_CONSOLIDATION_DAYS", "7"))
        self.MEMORY_ARCHIVE_DAYS = int(os.getenv("MEMORY_ARCHIVE_DAYS", "60"))
        
        # Relationship Configuration
        self.RELATIONSHIP_CROSSROADS_THRESHOLD = int(os.getenv("RELATIONSHIP_CROSSROADS_THRESHOLD", "40"))
        self.RELATIONSHIP_RITUAL_THRESHOLD = int(os.getenv("RELATIONSHIP_RITUAL_THRESHOLD", "60"))
        
        # Cache Configuration
        self.NPC_CACHE_SIZE = int(os.getenv("NPC_CACHE_SIZE", "50"))
        self.NPC_CACHE_TTL = int(os.getenv("NPC_CACHE_TTL", "30"))
        self.LOCATION_CACHE_SIZE = int(os.getenv("LOCATION_CACHE_SIZE", "20"))
        self.LOCATION_CACHE_TTL = int(os.getenv("LOCATION_CACHE_TTL", "120"))
        self.AGGREGATOR_CACHE_SIZE = int(os.getenv("AGGREGATOR_CACHE_SIZE", "10"))
        self.AGGREGATOR_CACHE_TTL = int(os.getenv("AGGREGATOR_CACHE_TTL", "15"))
        self.TIME_CACHE_SIZE = int(os.getenv("TIME_CACHE_SIZE", "5"))
        self.TIME_CACHE_TTL = int(os.getenv("TIME_CACHE_TTL", "10"))
        
        # Database Configuration
        self.DB_MIN_CONN = int(os.getenv("DB_MIN_CONN", "5"))
        self.DB_MAX_CONN = int(os.getenv("DB_MAX_CONN", "20"))
        self.DB_COMMAND_TIMEOUT = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        self.DB_STATEMENT_TIMEOUT = int(os.getenv("DB_STATEMENT_TIMEOUT", "60"))
        self.DB_INACTIVE_CONN_LIFETIME = int(os.getenv("DB_INACTIVE_CONN_LIFETIME", "300"))
        
        # Performance Configuration
        self.PERF_SLOW_QUERY_THRESHOLD_MS = int(os.getenv("PERF_SLOW_QUERY_THRESHOLD_MS", "500"))
        self.PERF_MEMORY_CHECK_INTERVAL = int(os.getenv("PERF_MEMORY_CHECK_INTERVAL", "60"))
        self.PERF_HIGH_MEMORY_THRESHOLD = float(os.getenv("PERF_HIGH_MEMORY_THRESHOLD", "70.0"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Initialize logging configuration
        self._configure_logging()
        
        # Log config initialization
        logging.info("Configuration initialized")
    
    def _configure_logging(self):
        """Configure logging based on settings."""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        log_level = log_levels.get(self.LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return str(self.to_dict())
    
    def get(self, key, default=None):
        """Get configuration value with optional default."""
        return getattr(self, key, default)

# Create global instance
CONFIG = Config()

# Helper functions
def get_config():
    """Get the global configuration instance."""
    return CONFIG

def get(key, default=None):
    """Get configuration value with fallback."""
    return CONFIG.get(key, default)
