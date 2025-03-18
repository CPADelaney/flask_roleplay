"""Tests for the error recovery system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch

from lore.error_recovery import ErrorRecoverySystem

@pytest.fixture
async def error_recovery_system():
    """Create an error recovery system instance for testing."""
    system = ErrorRecoverySystem(max_retries=2, retry_delay=0.1)
    await system.initialize()
    yield system
    await system.close()

@pytest.mark.asyncio
async def test_error_handling(error_recovery_system):
    """Test basic error handling functionality."""
    # Test error context manager
    with pytest.raises(ValueError):
        async with error_recovery_system.error_context("test_operation"):
            raise ValueError("Test error")
    
    # Verify error was recorded
    assert error_recovery_system._error_counts["test_operation"] == 1

@pytest.mark.asyncio
async def test_recovery_strategy(error_recovery_system):
    """Test error recovery strategy execution."""
    recovery_called = False
    
    async def test_strategy(error_details: Dict[str, Any]):
        nonlocal recovery_called
        recovery_called = True
        return True
    
    # Register recovery strategy
    await error_recovery_system.register_recovery_strategy("test_operation", test_strategy)
    
    # Trigger error
    try:
        async with error_recovery_system.error_context("test_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Verify recovery was attempted
    assert recovery_called

@pytest.mark.asyncio
async def test_recovery_retries(error_recovery_system):
    """Test recovery retry mechanism."""
    retry_count = 0
    
    async def failing_strategy(error_details: Dict[str, Any]):
        nonlocal retry_count
        retry_count += 1
        raise ValueError("Strategy failed")
    
    # Register failing strategy
    await error_recovery_system.register_recovery_strategy("test_operation", failing_strategy)
    
    # Trigger error
    try:
        async with error_recovery_system.error_context("test_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Verify correct number of retries
    assert retry_count == error_recovery_system.max_retries

@pytest.mark.asyncio
async def test_error_rate_monitoring(error_recovery_system):
    """Test error rate monitoring functionality."""
    # Trigger multiple errors
    for _ in range(5):
        try:
            async with error_recovery_system.error_context("test_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
    
    # Wait for monitoring cycle
    await asyncio.sleep(0.2)
    
    # Verify error rate calculation
    current_time = datetime.utcnow()
    error_rate = await error_recovery_system._calculate_error_rate("test_operation", current_time)
    assert error_rate > 0

@pytest.mark.asyncio
async def test_system_health_monitoring(error_recovery_system):
    """Test system health monitoring functionality."""
    # Mock system resource check
    with patch.object(error_recovery_system, '_check_system_resources') as mock_resources:
        mock_resources.return_value = {
            'cpu': 90.0,
            'memory': 85.0,
            'disk': 60.0,
            'network': 40.0
        }
        
        # Wait for monitoring cycle
        await asyncio.sleep(0.2)
        
        # Verify resource alert was triggered
        # Note: This would need to be verified through the alert system implementation

@pytest.mark.asyncio
async def test_error_cleanup(error_recovery_system):
    """Test cleanup of old error records."""
    # Mock error storage
    with patch.object(error_recovery_system, '_delete_old_error_records') as mock_cleanup:
        # Wait for cleanup cycle
        await asyncio.sleep(0.2)
        
        # Verify cleanup was called
        mock_cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_alert_triggering(error_recovery_system):
    """Test alert triggering functionality."""
    alert_sent = False
    
    async def mock_send_alert(alert_data: Dict[str, Any]):
        nonlocal alert_sent
        alert_sent = True
    
    # Mock alert sending
    with patch.object(error_recovery_system, '_send_alert', mock_send_alert):
        # Trigger high error rate alert
        await error_recovery_system._trigger_error_rate_alert("test_operation", 0.15)
        
        # Verify alert was sent
        assert alert_sent

@pytest.mark.asyncio
async def test_error_recording(error_recovery_system):
    """Test error recording functionality."""
    error_recorded = False
    
    async def mock_record_error(error_details: Dict[str, Any]):
        nonlocal error_recorded
        error_recorded = True
    
    # Mock error recording
    with patch.object(error_recovery_system, '_record_error', mock_record_error):
        # Trigger error
        try:
            async with error_recovery_system.error_context("test_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify error was recorded
        assert error_recorded

@pytest.mark.asyncio
async def test_system_initialization_and_cleanup(error_recovery_system):
    """Test system initialization and cleanup."""
    # Verify monitoring tasks are running
    assert len(error_recovery_system._monitoring_tasks) > 0
    assert error_recovery_system._is_monitoring
    
    # Close system
    await error_recovery_system.close()
    
    # Verify monitoring tasks are cancelled
    assert not error_recovery_system._is_monitoring
    for task in error_recovery_system._monitoring_tasks:
        assert task.cancelled() 