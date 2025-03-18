"""Tests for the system monitoring implementation."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import psutil

from lore.monitoring import SystemMonitor, SystemMetrics, PerformanceMetrics

@pytest.fixture
async def system_monitor():
    """Create a system monitor instance for testing."""
    monitor = SystemMonitor(collection_interval=0.1)
    await monitor.initialize()
    yield monitor
    await monitor.close()

@pytest.mark.asyncio
async def test_performance_metric_recording(system_monitor):
    """Test recording of performance metrics."""
    # Record a successful operation
    await system_monitor.record_performance_metric(
        operation="test_operation",
        duration=0.5,
        success=True
    )
    
    # Record a failed operation
    await system_monitor.record_performance_metric(
        operation="test_operation",
        duration=0.3,
        success=False,
        error=ValueError("Test error")
    )
    
    # Get metrics
    metrics = await system_monitor.get_performance_metrics("test_operation")
    assert len(metrics) == 2
    
    # Verify metrics
    assert metrics[0].operation == "test_operation"
    assert metrics[0].duration == 0.5
    assert metrics[0].success is True
    assert metrics[0].error_type is None
    
    assert metrics[1].operation == "test_operation"
    assert metrics[1].duration == 0.3
    assert metrics[1].success is False
    assert metrics[1].error_type == "ValueError"
    assert metrics[1].error_message == "Test error"

@pytest.mark.asyncio
async def test_system_metrics_collection(system_monitor):
    """Test collection of system metrics."""
    # Wait for metrics collection
    await asyncio.sleep(0.2)
    
    # Get metrics
    metrics = await system_monitor.get_system_metrics()
    assert len(metrics) > 0
    
    # Verify latest metrics
    latest = metrics[-1]
    assert isinstance(latest, SystemMetrics)
    assert isinstance(latest.timestamp, datetime)
    assert 0 <= latest.cpu_percent <= 100
    assert 0 <= latest.memory_percent <= 100
    assert 0 <= latest.disk_percent <= 100
    assert isinstance(latest.network_io, dict)
    assert latest.process_count > 0
    assert latest.thread_count > 0

@pytest.mark.asyncio
async def test_metrics_time_range_filtering(system_monitor):
    """Test filtering metrics by time range."""
    # Record metrics
    await system_monitor.record_performance_metric(
        operation="test_operation",
        duration=0.5,
        success=True
    )
    
    # Wait a bit
    await asyncio.sleep(0.2)
    
    # Get metrics with time range
    time_range = timedelta(seconds=0.1)
    metrics = await system_monitor.get_performance_metrics(
        operation="test_operation",
        time_range=time_range
    )
    assert len(metrics) == 0  # Should be empty due to time range

@pytest.mark.asyncio
async def test_operation_statistics(system_monitor):
    """Test calculation of operation statistics."""
    # Record multiple operations
    for i in range(5):
        await system_monitor.record_performance_metric(
            operation="test_operation",
            duration=0.5,
            success=True
        )
    
    # Record some failures
    for i in range(3):
        await system_monitor.record_performance_metric(
            operation="test_operation",
            duration=0.3,
            success=False,
            error=ValueError("Test error")
        )
    
    # Get statistics
    stats = await system_monitor.get_operation_statistics("test_operation")
    
    # Verify statistics
    assert stats['count'] == 8
    assert stats['success_rate'] == 5/8
    assert stats['error_rate'] == 3/8
    assert 0.3 <= stats['avg_duration'] <= 0.5
    assert stats['error_types']['ValueError'] == 3

@pytest.mark.asyncio
async def test_high_resource_usage_detection(system_monitor):
    """Test detection of high resource usage."""
    # Mock psutil to simulate high resource usage
    with patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_usage') as mock_disk:
        
        # Set high resource usage
        mock_cpu.return_value = 90.0
        mock_memory.return_value = Mock(percent=85.0)
        mock_disk.return_value = Mock(percent=75.0)
        
        # Wait for metrics collection
        await asyncio.sleep(0.2)
        
        # Get metrics
        metrics = await system_monitor.get_system_metrics()
        assert len(metrics) > 0
        
        # Verify high resource usage is reflected
        latest = metrics[-1]
        assert latest.cpu_percent == 90.0
        assert latest.memory_percent == 85.0
        assert latest.disk_percent == 75.0

@pytest.mark.asyncio
async def test_metrics_cleanup(system_monitor):
    """Test cleanup of old metrics."""
    # Record metrics
    await system_monitor.record_performance_metric(
        operation="test_operation",
        duration=0.5,
        success=True
    )
    
    # Wait for cleanup cycle
    await asyncio.sleep(0.2)
    
    # Get metrics with time range
    time_range = timedelta(hours=24)
    metrics = await system_monitor.get_performance_metrics(
        operation="test_operation",
        time_range=time_range
    )
    assert len(metrics) > 0  # Should still have metrics within 24 hours

@pytest.mark.asyncio
async def test_system_monitor_initialization_and_cleanup(system_monitor):
    """Test system monitor initialization and cleanup."""
    # Verify monitoring tasks are running
    assert len(system_monitor._monitoring_tasks) > 0
    assert system_monitor._is_monitoring
    
    # Close monitor
    await system_monitor.close()
    
    # Verify monitoring tasks are cancelled
    assert not system_monitor._is_monitoring
    for task in system_monitor._monitoring_tasks:
        assert task.cancelled()

@pytest.mark.asyncio
async def test_metrics_history_size_limit(system_monitor):
    """Test that metrics history size is limited."""
    # Record more metrics than the limit
    for i in range(system_monitor._max_history_size + 10):
        await system_monitor.record_performance_metric(
            operation="test_operation",
            duration=0.5,
            success=True
        )
    
    # Get metrics
    metrics = await system_monitor.get_performance_metrics()
    assert len(metrics) == system_monitor._max_history_size 