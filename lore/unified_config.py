"""
Unified Configuration System

This module provides centralized configuration management for the lore system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, Type, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigSource(str, Enum):
    """Configuration source types"""
    ENV = "environment"
    FILE = "file"
    DEFAULT = "default"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "lore_db"
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    charset: str = 'utf8mb4'
    use_unicode: bool = True
    autocommit: bool = True

@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    max_size: int = 1000
    ttl: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    strategy: str = "lru"
    compression: bool = False
    persistence: bool = False
    persistence_path: Optional[str] = None

@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enabled: bool = True
    collection_interval: int = 60
    retention_period: int = 86400
    max_samples: int = 1000
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    export_format: str = "json"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/lore.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_enabled: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    token_expiration: int = 3600
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    max_login_attempts: int = 3
    lockout_duration: int = 300
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=list)

@dataclass
class SystemConfig:
    """System configuration"""
    environment: str = "development"
    debug: bool = False
    testing: bool = False
    max_workers: int = 4
    request_timeout: int = 30
    max_request_size: int = 10485760  # 10MB
    temp_dir: str = "/tmp/lore"
    file_upload_dir: str = "uploads"
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".json", ".yaml", ".md"
    ])

@dataclass
class LoreConfig:
    """Main configuration class"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

class ConfigurationManager:
    """Manages system configuration"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "LORE_"
    ):
        self.config_path = config_path or "config/config.yaml"
        self.env_prefix = env_prefix
        self.config = LoreConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from all sources"""
        # Load default config
        config_dict = asdict(self.config)
        
        # Load from file if exists
        file_config = self._load_from_file()
        if file_config:
            self._deep_update(config_dict, file_config)
        
        # Load from environment variables
        env_config = self._load_from_env()
        if env_config:
            self._deep_update(config_dict, env_config)
        
        # Create new config instance
        self.config = self._dict_to_config(config_dict)
        
        # Setup logging
        self._setup_logging()
    
    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        try:
            if not os.path.exists(self.config_path):
                return None
            
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    return json.load(f)
                elif self.config_path.endswith('.yaml'):
                    return yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
                    return None
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return None
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue
            
            # Remove prefix and split into parts
            key = key[len(self.env_prefix):].lower()
            parts = key.split('_')
            
            # Build nested dictionary
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value with appropriate type conversion
            current[parts[-1]] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, list]:
        """Convert environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try list (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]
        
        return value
    
    def _deep_update(self, d: dict, u: dict):
        """Recursively update dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> LoreConfig:
        """Convert dictionary to config object"""
        return LoreConfig(
            database=DatabaseConfig(**config_dict['database']),
            cache=CacheConfig(**config_dict['cache']),
            metrics=MetricsConfig(**config_dict['metrics']),
            logging=LoggingConfig(**config_dict['logging']),
            security=SecurityConfig(**config_dict['security']),
            system=SystemConfig(**config_dict['system'])
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.logging
        
        # Create logs directory if needed
        if log_config.file_enabled:
            os.makedirs(os.path.dirname(log_config.file_path), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format=log_config.format,
            handlers=[
                logging.StreamHandler() if log_config.console_enabled else None,
                logging.FileHandler(log_config.file_path) if log_config.file_enabled else None
            ]
        )
    
    def get_config(self) -> LoreConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        config_dict = asdict(self.config)
        self._deep_update(config_dict, updates)
        self.config = self._dict_to_config(config_dict)
        
        # Reload logging if logging config changed
        if 'logging' in updates:
            self._setup_logging()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            # Create config directory if needed
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                elif self.config_path.endswith('.yaml'):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def get_database_url(self) -> str:
        """Get database URL from config"""
        db = self.config.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"

# Global configuration manager instance
config_manager = ConfigurationManager()

def get_config() -> LoreConfig:
    """Get the current configuration"""
    return config_manager.get_config()

def update_config(updates: Dict[str, Any]):
    """Update the current configuration"""
    config_manager.update_config(updates) 