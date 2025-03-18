"""
Unified Configuration System

Manages all configuration settings for the lore system.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import os
from functools import lru_cache
import logging
from enum import Enum
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class CacheConfig:
    """Cache configuration"""
    name: str
    max_size: int
    ttl: int
    eviction_policy: str = "lru"
    max_memory_usage: float = 0.8
    monitoring_enabled: bool = True

@dataclass
class ResourceConfig:
    """Resource management configuration"""
    caches: Dict[str, CacheConfig]
    cleanup_interval: int = 300
    validation_batch_size: int = 50
    performance_monitoring: bool = True
    rate_limit_requests: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    monitoring_interval: int = 60

@dataclass
class ValidationConfig:
    """Validation configuration"""
    schema_path: str = "schemas"
    validators_path: str = "validators"
    max_workers: int = 4
    max_parallel_validations: int = 10
    validation_mode: str = "strict"
    cache_ttl: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class ErrorHandlerConfig:
    """Error handling configuration"""
    max_error_history: int = 1000
    recovery_timeout: float = 30.0
    log_level: str = "INFO"
    error_reporting_enabled: bool = True
    recovery_strategies: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    user: str
    password: str
    database: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    sampling_rate: float = 1.0
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    token_expiry: int = 3600
    rate_limit_enabled: bool = True
    cors_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class LoreConfig:
    """Main configuration class"""
    environment: Environment
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    cache: ResourceConfig = field(default_factory=ResourceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    error_handler: ErrorHandlerConfig = field(default_factory=ErrorHandlerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """
    Unified configuration management system.
    Handles loading and accessing configuration from environment variables and config files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('LORE_CONFIG_PATH', 'config')
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from environment variables and config files."""
        # Load environment variables
        load_dotenv()
        
        # Load base config file
        base_config = self._load_json_file('config.json')
        if base_config:
            self.config.update(base_config)
            
        # Load environment-specific config
        env = os.getenv('LORE_ENV', 'development')
        env_config = self._load_json_file(f'config.{env}.json')
        if env_config:
            self.config.update(env_config)
            
        # Override with environment variables
        self._load_env_vars()
        
    def _load_json_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load configuration from a JSON file."""
        try:
            file_path = Path(self.config_path) / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {filename}: {str(e)}")
        return None
        
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith('LORE_'):
                # Convert LORE_DATABASE_HOST to database.host
                config_key = key[5:].lower().replace('_', '.')
                self._set_nested_value(self.config, config_key, value)
                
    def _set_nested_value(self, d: Dict[str, Any], key: str, value: Any):
        """Set a nested dictionary value using dot notation."""
        keys = key.split('.')
        current = d
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'host': self.get('database.host', 'localhost'),
            'port': int(self.get('database.port', 5432)),
            'database': self.get('database.name', 'lore_db'),
            'user': self.get('database.user', 'lore_user'),
            'password': self.get('database.password', ''),
            'min_size': int(self.get('database.pool.min_size', 1)),
            'max_size': int(self.get('database.pool.max_size', 10))
        }
        
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            'type': self.get('cache.type', 'memory'),
            'host': self.get('cache.host', 'localhost'),
            'port': int(self.get('cache.port', 6379)),
            'db': int(self.get('cache.db', 0)),
            'ttl': int(self.get('cache.ttl', 3600))
        }
        
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.get('logging.level', 'INFO'),
            'format': self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'file': self.get('logging.file', 'lore.log'),
            'max_size': int(self.get('logging.max_size', 10485760)),  # 10MB
            'backup_count': int(self.get('logging.backup_count', 5))
        }
        
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            'host': self.get('api.host', '0.0.0.0'),
            'port': int(self.get('api.port', 8000)),
            'debug': self.get('api.debug', False),
            'rate_limit': int(self.get('api.rate_limit', 100))
        }
        
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration."""
        return {
            'enabled': self.get('metrics.enabled', True),
            'host': self.get('metrics.host', '0.0.0.0'),
            'port': int(self.get('metrics.port', 9090)),
            'path': self.get('metrics.path', '/metrics')
        }
        
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            'secret_key': self.get('security.secret_key', ''),
            'token_expiry': int(self.get('security.token_expiry', 3600)),
            'allowed_origins': self.get('security.allowed_origins', ['*'])
        }
        
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return {
            'enabled': self.get('integration.enabled', True),
            'timeout': int(self.get('integration.timeout', 30)),
            'retry_count': int(self.get('integration.retry_count', 3))
        }
        
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            'strict_mode': self.get('validation.strict_mode', False),
            'max_depth': int(self.get('validation.max_depth', 10)),
            'max_length': int(self.get('validation.max_length', 1000))
        }
        
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return {
            'cache_ttl': int(self.get('performance.cache_ttl', 3600)),
            'batch_size': int(self.get('performance.batch_size', 100)),
            'max_connections': int(self.get('performance.max_connections', 100))
        }
        
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values."""
        for key, value in updates.items():
            self._set_nested_value(self.config, key, value)
            
    def save_config(self):
        """Save current configuration to file."""
        try:
            file_path = Path(self.config_path) / 'config.json'
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")

# Create global config instance
config = ConfigManager()

@lru_cache()
def get_config() -> LoreConfig:
    """Get the global configuration instance"""
    return config.config 