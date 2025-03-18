"""Tests for the logging configuration implementation."""

import pytest
import os
import logging
import json
from datetime import datetime
from pathlib import Path
import shutil

from lore.logging_config import LogConfig, LogManager, JsonFormatter

@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    yield str(log_dir)
    shutil.rmtree(log_dir)

@pytest.fixture
def log_config(temp_log_dir):
    """Create a log configuration instance."""
    config = LogConfig(log_dir=temp_log_dir)
    config.configure_logging()
    return config

@pytest.fixture
def log_manager(temp_log_dir):
    """Create a log manager instance."""
    return LogManager(log_dir=temp_log_dir)

def test_log_config_initialization(temp_log_dir):
    """Test log configuration initialization."""
    config = LogConfig(log_dir=temp_log_dir)
    assert config.log_dir == temp_log_dir
    assert config.log_level == logging.INFO
    assert config.max_bytes == 10 * 1024 * 1024
    assert config.backup_count == 5

def test_log_config_directory_creation(temp_log_dir):
    """Test log directory creation."""
    # Directory should be created during initialization
    assert os.path.exists(temp_log_dir)
    assert os.path.isdir(temp_log_dir)

def test_log_configuration(log_config, temp_log_dir):
    """Test logging configuration."""
    # Verify log files are created
    assert os.path.exists(os.path.join(temp_log_dir, 'all.log'))
    assert os.path.exists(os.path.join(temp_log_dir, 'error.log'))
    assert os.path.exists(os.path.join(temp_log_dir, 'structured.log'))
    
    # Test logging
    logger = log_config.get_logger('test')
    logger.info('Test message')
    logger.error('Test error')
    
    # Verify messages are written to appropriate files
    with open(os.path.join(temp_log_dir, 'all.log'), 'r') as f:
        content = f.read()
        assert 'Test message' in content
        assert 'Test error' in content
    
    with open(os.path.join(temp_log_dir, 'error.log'), 'r') as f:
        content = f.read()
        assert 'Test error' in content
        assert 'Test message' not in content

def test_json_formatter():
    """Test JSON formatter."""
    formatter = JsonFormatter()
    
    # Create a log record
    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    # Format the record
    formatted = formatter.format(record)
    
    # Parse the JSON
    log_entry = json.loads(formatted)
    
    # Verify fields
    assert 'timestamp' in log_entry
    assert log_entry['level'] == 'INFO'
    assert log_entry['logger'] == 'test'
    assert log_entry['message'] == 'Test message'
    assert log_entry['module'] == 'test'
    assert log_entry['function'] == '<module>'
    assert log_entry['line'] == 1

def test_json_formatter_with_exception():
    """Test JSON formatter with exception information."""
    formatter = JsonFormatter()
    
    try:
        raise ValueError('Test error')
    except ValueError as e:
        # Create a log record with exception
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=1,
            msg='Test error',
            args=(),
            exc_info=(ValueError, e, None)
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON
        log_entry = json.loads(formatted)
        
        # Verify exception information
        assert 'exception' in log_entry
        assert log_entry['exception']['type'] == 'ValueError'
        assert log_entry['exception']['message'] == 'Test error'

def test_log_manager_initialization(temp_log_dir):
    """Test log manager initialization."""
    manager = LogManager(log_dir=temp_log_dir)
    assert manager.log_dir == temp_log_dir
    assert os.path.exists(temp_log_dir)

def test_get_log_files(log_manager, temp_log_dir):
    """Test getting log files."""
    # Create some test log files
    test_files = ['test1.log', 'test2.log']
    for file in test_files:
        with open(os.path.join(temp_log_dir, file), 'w') as f:
            f.write('Test content')
    
    # Get log files
    log_files = log_manager.get_log_files()
    
    # Verify files are found
    assert len(log_files) == len(test_files)
    for file in test_files:
        assert file in log_files
        assert log_files[file] == os.path.join(temp_log_dir, file)

def test_get_log_content(log_manager, temp_log_dir):
    """Test getting log content."""
    # Create a test log file with content
    test_file = 'test.log'
    test_content = ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5']
    with open(os.path.join(temp_log_dir, test_file), 'w') as f:
        f.write('\n'.join(test_content))
    
    # Test getting all lines
    content = log_manager.get_log_content(test_file)
    assert content == test_content
    
    # Test getting last N lines
    content = log_manager.get_log_content(test_file, lines=2)
    assert content == test_content[-2:]
    
    # Test searching content
    content = log_manager.get_log_content(test_file, search='Line 2')
    assert content == ['Line 2']

def test_clear_logs(log_manager, temp_log_dir):
    """Test clearing logs."""
    # Create test log files
    test_files = ['test1.log', 'test2.log']
    for file in test_files:
        with open(os.path.join(temp_log_dir, file), 'w') as f:
            f.write('Test content')
    
    # Clear specific log file
    log_manager.clear_logs('test1.log')
    assert os.path.exists(os.path.join(temp_log_dir, 'test1.log'))
    assert os.path.getsize(os.path.join(temp_log_dir, 'test1.log')) == 0
    assert os.path.getsize(os.path.join(temp_log_dir, 'test2.log')) > 0
    
    # Clear all logs
    log_manager.clear_logs()
    for file in test_files:
        assert os.path.getsize(os.path.join(temp_log_dir, file)) == 0

def test_archive_logs(log_manager, temp_log_dir):
    """Test archiving logs."""
    # Create test log files
    test_files = ['test1.log', 'test2.log']
    for file in test_files:
        with open(os.path.join(temp_log_dir, file), 'w') as f:
            f.write('Test content')
    
    # Archive logs
    log_manager.archive_logs()
    
    # Verify archive directory is created
    archive_dir = os.path.join(temp_log_dir, 'archives')
    assert os.path.exists(archive_dir)
    
    # Verify archive contains log files
    archive_contents = os.listdir(archive_dir)
    assert len(archive_contents) == 1  # One timestamped directory
    
    archive_path = os.path.join(archive_dir, archive_contents[0])
    assert os.path.isdir(archive_path)
    
    archived_files = os.listdir(archive_path)
    assert len(archived_files) == len(test_files)
    for file in test_files:
        assert file in archived_files

def test_get_log_statistics(log_manager, temp_log_dir):
    """Test getting log statistics."""
    # Create a test log file with various log levels
    test_file = 'test.log'
    test_content = [
        '2024-01-01 00:00:00 - test - INFO - Info message',
        '2024-01-01 00:00:01 - test - WARNING - Warning message',
        '2024-01-01 00:00:02 - test - ERROR - Error message',
        '2024-01-01 00:00:03 - test - INFO - Another info message'
    ]
    with open(os.path.join(temp_log_dir, test_file), 'w') as f:
        f.write('\n'.join(test_content))
    
    # Get statistics
    stats = log_manager.get_log_statistics(test_file)
    
    # Verify statistics
    assert stats['line_count'] == len(test_content)
    assert stats['level_counts']['INFO'] == 2
    assert stats['level_counts']['WARNING'] == 1
    assert stats['level_counts']['ERROR'] == 1
    assert 'size' in stats
    assert 'created' in stats
    assert 'modified' in stats 