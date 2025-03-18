"""Tests for the configuration management implementation."""

import pytest
import os
import json
import yaml
from pathlib import Path
import shutil

from lore.config import (
    ConfigManager, LoreConfig, DatabaseConfig,
    LoggingConfig, MonitoringConfig
)

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for configuration files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yield str(config_dir)
    shutil.rmtree(config_dir)

@pytest.fixture
def config_manager(temp_config_dir):
    """Create a configuration manager instance."""
    return ConfigManager(config_dir=temp_config_dir)

def test_config_manager_initialization(temp_config_dir):
    """Test configuration manager initialization."""
    manager = ConfigManager(config_dir=temp_config_dir)
    assert manager.config_dir == temp_config_dir
    assert manager.config is None
    assert os.path.exists(temp_config_dir)

def test_create_default_config(config_manager):
    """Test creation of default configuration."""
    config = config_manager._create_default_config()
    assert isinstance(config, LoreConfig)
    assert isinstance(config.database, DatabaseConfig)
    assert isinstance(config.logging, LoggingConfig)
    assert isinstance(config.monitoring, MonitoringConfig)
    
    # Verify default values
    assert config.debug is False
    assert config.environment == "development"
    assert config.timezone == "UTC"
    assert config.database.host == "localhost"
    assert config.database.port == 5432
    assert config.logging.log_dir == "logs"
    assert config.monitoring.enabled is True

def test_load_config_from_yaml(config_manager, temp_config_dir):
    """Test loading configuration from YAML file."""
    # Create test YAML config
    test_config = {
        'debug': True,
        'environment': 'test',
        'database': {
            'host': 'test_host',
            'port': 5433
        },
        'logging': {
            'log_dir': 'test_logs',
            'log_level': 'DEBUG'
        }
    }
    
    config_file = os.path.join(temp_config_dir, 'test.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    # Load configuration
    config = config_manager.load_config(config_file)
    
    # Verify loaded values
    assert config.debug is True
    assert config.environment == 'test'
    assert config.database.host == 'test_host'
    assert config.database.port == 5433
    assert config.logging.log_dir == 'test_logs'
    assert config.logging.log_level == 'DEBUG'

def test_load_config_from_json(config_manager, temp_config_dir):
    """Test loading configuration from JSON file."""
    # Create test JSON config
    test_config = {
        'debug': True,
        'environment': 'test',
        'database': {
            'host': 'test_host',
            'port': 5433
        },
        'logging': {
            'log_dir': 'test_logs',
            'log_level': 'DEBUG'
        }
    }
    
    config_file = os.path.join(temp_config_dir, 'test.json')
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Load configuration
    config = config_manager.load_config(config_file)
    
    # Verify loaded values
    assert config.debug is True
    assert config.environment == 'test'
    assert config.database.host == 'test_host'
    assert config.database.port == 5433
    assert config.logging.log_dir == 'test_logs'
    assert config.logging.log_level == 'DEBUG'

def test_save_config_to_yaml(config_manager, temp_config_dir):
    """Test saving configuration to YAML file."""
    # Create test configuration
    config = LoreConfig(
        debug=True,
        environment='test',
        database=DatabaseConfig(host='test_host', port=5433),
        logging=LoggingConfig(log_dir='test_logs', log_level='DEBUG')
    )
    config_manager.config = config
    
    # Save configuration
    config_file = os.path.join(temp_config_dir, 'test.yaml')
    config_manager.save_config(config_file)
    
    # Verify saved file
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        saved_config = yaml.safe_load(f)
        assert saved_config['debug'] is True
        assert saved_config['environment'] == 'test'
        assert saved_config['database']['host'] == 'test_host'
        assert saved_config['database']['port'] == 5433
        assert saved_config['logging']['log_dir'] == 'test_logs'
        assert saved_config['logging']['log_level'] == 'DEBUG'

def test_save_config_to_json(config_manager, temp_config_dir):
    """Test saving configuration to JSON file."""
    # Create test configuration
    config = LoreConfig(
        debug=True,
        environment='test',
        database=DatabaseConfig(host='test_host', port=5433),
        logging=LoggingConfig(log_dir='test_logs', log_level='DEBUG')
    )
    config_manager.config = config
    
    # Save configuration
    config_file = os.path.join(temp_config_dir, 'test.json')
    config_manager.save_config(config_file)
    
    # Verify saved file
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        saved_config = json.load(f)
        assert saved_config['debug'] is True
        assert saved_config['environment'] == 'test'
        assert saved_config['database']['host'] == 'test_host'
        assert saved_config['database']['port'] == 5433
        assert saved_config['logging']['log_dir'] == 'test_logs'
        assert saved_config['logging']['log_level'] == 'DEBUG'

def test_update_config(config_manager):
    """Test updating configuration."""
    # Load default configuration
    config_manager.load_config()
    
    # Update configuration
    updates = {
        'debug': True,
        'database': {'host': 'test_host'},
        'logging': {'log_dir': 'test_logs'}
    }
    config_manager.update_config(updates)
    
    # Verify updates
    config = config_manager.get_config()
    assert config.debug is True
    assert config.database.host == 'test_host'
    assert config.logging.log_dir == 'test_logs'

def test_validate_config(config_manager):
    """Test configuration validation."""
    # Test valid configuration
    config = LoreConfig()
    config_manager.config = config
    assert config_manager.validate_config() is True
    
    # Test invalid database configuration
    config.database.host = ""
    assert config_manager.validate_config() is False
    
    # Test invalid logging configuration
    config.database.host = "localhost"
    config.logging.log_dir = ""
    assert config_manager.validate_config() is False
    
    # Test invalid monitoring configuration
    config.logging.log_dir = "logs"
    config.monitoring.collection_interval = 0
    assert config_manager.validate_config() is False
    
    # Test invalid lore-specific settings
    config.monitoring.collection_interval = 60.0
    config.max_npc_count = 0
    assert config_manager.validate_config() is False
    
    # Test invalid API settings
    config.max_npc_count = 1000
    config.api_port = 0
    assert config_manager.validate_config() is False

def test_get_environment_config(config_manager, temp_config_dir):
    """Test getting environment-specific configuration."""
    # Create test environment config
    test_config = {
        'debug': True,
        'environment': 'test'
    }
    
    config_file = os.path.join(temp_config_dir, 'test.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    # Set environment variable
    os.environ['LORE_ENV'] = 'test'
    
    # Get environment config
    env_config = config_manager.get_environment_config()
    assert env_config['debug'] is True
    assert env_config['environment'] == 'test'

def test_get_secret_config(config_manager, temp_config_dir):
    """Test getting secret configuration."""
    # Create test secrets config
    test_secrets = {
        'database_password': 'test_password',
        'api_key': 'test_key'
    }
    
    secret_file = os.path.join(temp_config_dir, 'secrets.yaml')
    with open(secret_file, 'w') as f:
        yaml.dump(test_secrets, f)
    
    # Get secret config
    secret_config = config_manager.get_secret_config()
    assert secret_config['database_password'] == 'test_password'
    assert secret_config['api_key'] == 'test_key'

def test_dict_to_config_conversion(config_manager):
    """Test dictionary to configuration object conversion."""
    test_dict = {
        'debug': True,
        'database': {
            'host': 'test_host',
            'port': 5433
        },
        'logging': {
            'log_dir': 'test_logs',
            'log_level': 'DEBUG'
        }
    }
    
    config = config_manager._dict_to_config(test_dict)
    assert isinstance(config, LoreConfig)
    assert config.debug is True
    assert isinstance(config.database, DatabaseConfig)
    assert config.database.host == 'test_host'
    assert config.database.port == 5433
    assert isinstance(config.logging, LoggingConfig)
    assert config.logging.log_dir == 'test_logs'
    assert config.logging.log_level == 'DEBUG'

def test_config_to_dict_conversion(config_manager):
    """Test configuration object to dictionary conversion."""
    config = LoreConfig(
        debug=True,
        database=DatabaseConfig(host='test_host', port=5433),
        logging=LoggingConfig(log_dir='test_logs', log_level='DEBUG')
    )
    
    config_dict = config_manager._config_to_dict(config)
    assert config_dict['debug'] is True
    assert config_dict['database']['host'] == 'test_host'
    assert config_dict['database']['port'] == 5433
    assert config_dict['logging']['log_dir'] == 'test_logs'
    assert config_dict['logging']['log_level'] == 'DEBUG' 
