"""Error recovery and monitoring system for the lore system."""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import traceback
from contextlib import asynccontextmanager
import psutil
from prometheus_client import Gauge, Counter
import json
import aiohttp
from dataclasses import asdict

from .lore_cache_manager import LoreCacheManager
from .resource_manager import resource_manager
from .governance_registration import GovernanceRegistration
from .error_handler import ErrorHandler, LoreError, ErrorType, ErrorContext

logger = logging.getLogger(__name__)

# Prometheus metrics
SYSTEM_HEALTH = Gauge('lore_system_health', 'System health status', ['component'])
ERROR_RECOVERY_COUNT = Counter('lore_error_recovery_total', 'Total number of error recoveries', ['type'])
RESOURCE_USAGE = Gauge('lore_resource_usage', 'Resource usage metrics', ['resource_type'])

# Configuration
ERROR_RATE_THRESHOLD = 0.1  # 10% error rate threshold
RESOURCE_USAGE_THRESHOLD = 80  # 80% resource usage threshold
MONITORING_INTERVALS = {
    'error_rates': 60,  # Check every minute
    'system_health': 300,  # Check every 5 minutes
    'resource_usage': 60,  # Check every minute
    'cleanup': 3600  # Run every hour
}
ALERT_ENDPOINTS = {
    'error_rate': 'http://alerting-service/api/v1/alerts/error-rate',
    'resource': 'http://alerting-service/api/v1/alerts/resource',
    'system': 'http://alerting-service/api/v1/alerts/system'
}

class ErrorRecoverySystem:
    """Error recovery and monitoring system."""
    
    def __init__(self, user_id: int, conversation_id: int, max_retries: int = 3, retry_delay: float = 1.0):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._error_counts = {}
        self._recovery_strategies = {}
        self._monitoring_tasks = []
        self._is_monitoring = False
        self._operation_counts = {}
        self._operation_timestamps = {}
        
        # Initialize components
        self.cache_manager = LoreCacheManager(user_id, conversation_id)
        self.governance = GovernanceRegistration(user_id, conversation_id)
        self.error_handler = ErrorHandler(user_id, conversation_id)
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        # Initialize alert client
        self.alert_client = aiohttp.ClientSession()
    
    def _register_default_strategies(self):
        """Register default recovery strategies for common operations."""
        self._recovery_strategies.update({
            'cache_operation': self._recover_cache_operation,
            'resource_operation': self._recover_resource_operation,
            'governance_operation': self._recover_governance_operation,
            'database_operation': self._recover_database_operation
        })
    
    async def initialize(self):
        """Initialize the error recovery system."""
        try:
            # Initialize components
            await self.cache_manager.initialize()
            await self.governance.initialize()
            
            # Start monitoring tasks
            self._is_monitoring = True
            self._monitoring_tasks = [
                asyncio.create_task(self._monitor_error_rates()),
                asyncio.create_task(self._monitor_system_health()),
                asyncio.create_task(self._cleanup_old_errors()),
                asyncio.create_task(self._monitor_resource_usage())
            ]
            logger.info("Error recovery system initialized")
        except Exception as e:
            logger.error(f"Error initializing error recovery system: {str(e)}")
            raise
    
    async def close(self):
        """Close the error recovery system."""
        try:
            # Stop monitoring tasks
            self._is_monitoring = False
            for task in self._monitoring_tasks:
                task.cancel()
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            # Cleanup components
            await self.cache_manager.close()
            await self.governance.close()
            await self.alert_client.close()
            
            logger.info("Error recovery system closed")
        except Exception as e:
            logger.error(f"Error closing error recovery system: {str(e)}")
    
    @asynccontextmanager
    async def error_context(self, operation: str):
        """Context manager for handling errors in operations."""
        try:
            # Record operation start
            await self._record_operation_start(operation)
            yield
        except Exception as e:
            await self.handle_error(operation, e)
            raise
        finally:
            # Record operation end
            await self._record_operation_end(operation)
    
    async def _record_operation_start(self, operation: str):
        """Record the start of an operation."""
        self._operation_timestamps[operation] = datetime.utcnow()
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
        
        # Store operation record in cache
        await self.cache_manager.set_lore(
            'operation_record',
            f"{operation}_{self._operation_timestamps[operation].isoformat()}",
            {
                'operation': operation,
                'start_time': self._operation_timestamps[operation].isoformat(),
                'status': 'started'
            }
        )
    
    async def _record_operation_end(self, operation: str):
        """Record the end of an operation."""
        if operation in self._operation_timestamps:
            end_time = datetime.utcnow()
            duration = (end_time - self._operation_timestamps[operation]).total_seconds()
            
            # Update operation record in cache
            await self.cache_manager.set_lore(
                'operation_record',
                f"{operation}_{self._operation_timestamps[operation].isoformat()}",
                {
                    'operation': operation,
                    'start_time': self._operation_timestamps[operation].isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration': duration,
                    'status': 'completed'
                }
            )
            
            del self._operation_timestamps[operation]
    
    async def handle_error(self, operation: str, error: Exception):
        """Handle an error that occurred during an operation."""
        try:
            # Create error context
            error_context = ErrorContext(
                timestamp=datetime.utcnow(),
                severity=self._determine_severity(error),
                category=self._determine_category(error),
                source=operation,
                details={
                    'operation': operation,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'traceback': traceback.format_exc()
                }
            )
            
            # Update context with system state
            await self._update_error_context(error_context)
            
            # Log error
            logger.error(f"Error in {operation}: {str(error)}")
            
            # Update error counts
            self._error_counts[operation] = self._error_counts.get(operation, 0) + 1
            
            # Attempt recovery if strategy exists
            if operation in self._recovery_strategies:
                await self._attempt_recovery(operation, error_context)
            
            # Record error for monitoring
            await self._record_error(error_context)
            
            # Update metrics
            ERROR_RECOVERY_COUNT.labels(type=type(error).__name__).inc()
            
        except Exception as e:
            logger.error(f"Error handling error: {str(e)}")
    
    async def _update_error_context(self, context: ErrorContext):
        """Update error context with current system state."""
        try:
            # Get resource usage
            context.resource_usage = await resource_manager.get_resource_usage()
            
            # Get cache state
            context.cache_state = await self.cache_manager.get_cache_stats()
            
            # Get governance state
            context.governance_state = await self.governance.get_state()
            
        except Exception as e:
            logger.error(f"Error updating error context: {str(e)}")
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, (DatabaseError, GovernanceError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ResourceError, CacheError)):
            return ErrorSeverity.ERROR
        elif isinstance(error, ValidationError):
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.INFO
    
    def _determine_category(self, error: Exception) -> ErrorCategory:
        """Determine error category based on error type."""
        if isinstance(error, DatabaseError):
            return ErrorCategory.DATABASE
        elif isinstance(error, ResourceError):
            return ErrorCategory.RESOURCE
        elif isinstance(error, CacheError):
            return ErrorCategory.SYSTEM
        elif isinstance(error, GovernanceError):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.CUSTOM
    
    async def _attempt_recovery(self, operation: str, error_context: ErrorContext):
        """Attempt to recover from an error using the registered strategy."""
        try:
            strategy = self._recovery_strategies[operation]
            for attempt in range(self.max_retries):
                try:
                    await strategy(error_context)
                    error_context.recovered = True
                    error_context.recovery_time = datetime.utcnow()
                    logger.info(f"Successfully recovered from error in {operation}")
                    return True
                except Exception as e:
                    logger.warning(f"Recovery attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
            
            logger.error(f"All recovery attempts failed for {operation}")
            return False
        except Exception as e:
            logger.error(f"Error during recovery attempt: {str(e)}")
            return False
    
    async def _recover_cache_operation(self, context: ErrorContext):
        """Recover from cache operation errors."""
        try:
            await self.cache_manager.clear_cache()
            await self.cache_manager.initialize()
            return True
        except Exception as e:
            logger.error(f"Cache recovery failed: {str(e)}")
            return False
    
    async def _recover_resource_operation(self, context: ErrorContext):
        """Recover from resource operation errors."""
        try:
            await resource_manager.optimize_resources()
            return True
        except Exception as e:
            logger.error(f"Resource recovery failed: {str(e)}")
            return False
    
    async def _recover_governance_operation(self, context: ErrorContext):
        """Recover from governance operation errors."""
        try:
            await self.governance.recover_state()
            return True
        except Exception as e:
            logger.error(f"Governance recovery failed: {str(e)}")
            return False
    
    async def _recover_database_operation(self, context: ErrorContext):
        """Recover from database operation errors."""
        try:
            # Implement database recovery logic
            return True
        except Exception as e:
            logger.error(f"Database recovery failed: {str(e)}")
            return False
    
    async def _monitor_error_rates(self):
        """Monitor error rates across operations."""
        while self._is_monitoring:
            try:
                # Calculate error rates
                current_time = datetime.utcnow()
                for operation, count in self._error_counts.items():
                    error_rate = await self._calculate_error_rate(operation, current_time)
                    if error_rate > ERROR_RATE_THRESHOLD:
                        logger.warning(f"High error rate detected for {operation}: {error_rate:.2%}")
                        await self._trigger_error_rate_alert(operation, error_rate)
                
                await asyncio.sleep(MONITORING_INTERVALS['error_rates'])
            except Exception as e:
                logger.error(f"Error monitoring error rates: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['error_rates'])
    
    async def _monitor_system_health(self):
        """Monitor overall system health."""
        while self._is_monitoring:
            try:
                # Check system resources
                resource_usage = await self._check_system_resources()
                for resource, usage in resource_usage.items():
                    RESOURCE_USAGE.labels(resource_type=resource).set(usage)
                    if usage > RESOURCE_USAGE_THRESHOLD:
                        logger.warning(f"High {resource} usage detected: {usage}%")
                        await self._trigger_resource_alert(resource_usage)
                
                # Check system state
                system_state = await self._check_system_state()
                for component, health in system_state['components'].items():
                    SYSTEM_HEALTH.labels(component=component).set(1 if health else 0)
                    if not health:
                        logger.warning(f"Unhealthy component detected: {component}")
                        await self._trigger_state_alert(system_state)
                
                await asyncio.sleep(MONITORING_INTERVALS['system_health'])
            except Exception as e:
                logger.error(f"Error monitoring system health: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['system_health'])
    
    async def _monitor_resource_usage(self):
        """Monitor resource usage and trigger alerts if needed."""
        while self._is_monitoring:
            try:
                usage = await resource_manager.get_resource_usage()
                for resource, value in usage.items():
                    if value > RESOURCE_USAGE_THRESHOLD:
                        await self._trigger_resource_alert({resource: value})
                await asyncio.sleep(MONITORING_INTERVALS['resource_usage'])
            except Exception as e:
                logger.error(f"Error monitoring resource usage: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['resource_usage'])
    
    async def _cleanup_old_errors(self):
        """Clean up old error records."""
        while self._is_monitoring:
            try:
                # Delete error records older than 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                await self._delete_old_error_records(cutoff_time)
                
                await asyncio.sleep(MONITORING_INTERVALS['cleanup'])
            except Exception as e:
                logger.error(f"Error cleaning up old errors: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['cleanup'])
    
    async def _calculate_error_rate(self, operation: str, current_time: datetime) -> float:
        """Calculate the error rate for an operation."""
        try:
            # Get error count for the last hour
            recent_errors = await self._get_recent_errors(operation, current_time - timedelta(hours=1))
            error_count = len(recent_errors)
            
            # Get total operation count for the last hour
            total_operations = await self._get_operation_count(operation, current_time - timedelta(hours=1))
            
            if total_operations == 0:
                return 0.0
            
            return error_count / total_operations
        except Exception as e:
            logger.error(f"Error calculating error rate: {str(e)}")
            return 0.0
    
    async def _get_operation_count(self, operation: str, since: datetime) -> int:
        """Get total operation count for a time period."""
        try:
            # Get operation records from cache
            records = await self.cache_manager.get_lore(
                'operation_record',
                f"{operation}_{since.isoformat()}"
            )
            
            if not records:
                return 0
            
            # Count completed operations
            completed_operations = sum(
                1 for record in records
                if record.get('status') == 'completed' and
                datetime.fromisoformat(record['start_time']) >= since
            )
            
            return completed_operations
        except Exception as e:
            logger.error(f"Error getting operation count: {str(e)}")
            return 0
    
    async def _check_system_resources(self) -> Dict[str, float]:
        """Check current system resource usage."""
        try:
            return {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent,
                'network': await self._get_network_usage()
            }
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return {}
    
    async def _get_network_usage(self) -> float:
        """Get current network usage percentage."""
        try:
            net_io = psutil.net_io_counters()
            return (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024 * 1024)  # Convert to GB
        except Exception:
            return 0.0
    
    async def _check_system_state(self) -> Dict[str, Any]:
        """Check the current state of the system."""
        try:
            return {
                'healthy': True,
                'components': {
                    'cache': await self.cache_manager.is_healthy(),
                    'governance': await self.governance.is_healthy(),
                    'resources': await resource_manager.is_healthy()
                },
                'issues': []
            }
        except Exception as e:
            logger.error(f"Error checking system state: {str(e)}")
            return {'healthy': False, 'error': str(e)}
    
    async def _trigger_error_rate_alert(self, operation: str, error_rate: float):
        """Trigger an alert for high error rates."""
        try:
            alert_data = {
                'type': 'error_rate',
                'operation': operation,
                'error_rate': error_rate,
                'timestamp': datetime.utcnow().isoformat(),
                'threshold': ERROR_RATE_THRESHOLD
            }
            await self._send_alert(ALERT_ENDPOINTS['error_rate'], alert_data)
        except Exception as e:
            logger.error(f"Error triggering error rate alert: {str(e)}")
    
    async def _trigger_resource_alert(self, resource_usage: Dict[str, float]):
        """Trigger an alert for high resource usage."""
        try:
            alert_data = {
                'type': 'resource_usage',
                'usage': resource_usage,
                'timestamp': datetime.utcnow().isoformat(),
                'threshold': RESOURCE_USAGE_THRESHOLD
            }
            await self._send_alert(ALERT_ENDPOINTS['resource'], alert_data)
        except Exception as e:
            logger.error(f"Error triggering resource alert: {str(e)}")
    
    async def _trigger_state_alert(self, system_state: Dict[str, Any]):
        """Trigger an alert for unhealthy system state."""
        try:
            alert_data = {
                'type': 'system_state',
                'state': system_state,
                'timestamp': datetime.utcnow().isoformat()
            }
            await self._send_alert(ALERT_ENDPOINTS['system'], alert_data)
        except Exception as e:
            logger.error(f"Error triggering state alert: {str(e)}")
    
    async def _send_alert(self, endpoint: str, alert_data: Dict[str, Any]):
        """Send an alert to the monitoring system."""
        try:
            # Prepare alert payload
            payload = {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'alert': alert_data
            }
            
            # Send alert to monitoring service
            async with self.alert_client.post(endpoint, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send alert: {await response.text()}")
                else:
                    logger.info(f"Alert sent successfully: {alert_data['type']}")
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    async def _record_error(self, error_context: ErrorContext):
        """Record error details for monitoring."""
        try:
            # Convert error context to dict, handling datetime serialization
            error_dict = asdict(error_context)
            error_dict['timestamp'] = error_dict['timestamp'].isoformat()
            if error_dict['recovery_time']:
                error_dict['recovery_time'] = error_dict['recovery_time'].isoformat()
            
            # Store error record in cache
            await self.cache_manager.set_lore(
                'error_record',
                f"{error_context.timestamp.isoformat()}_{error_context.source}",
                error_dict
            )
        except Exception as e:
            logger.error(f"Error recording error: {str(e)}")
    
    async def _get_recent_errors(self, operation: str, since: datetime) -> List[Dict[str, Any]]:
        """Get recent error records for an operation."""
        try:
            # Get error records from cache
            records = await self.cache_manager.get_lore(
                'error_record',
                f"{since.isoformat()}_{operation}"
            )
            return records or []
        except Exception as e:
            logger.error(f"Error getting recent errors: {str(e)}")
            return []
    
    async def _delete_old_error_records(self, cutoff_time: datetime):
        """Delete error records older than the cutoff time."""
        try:
            # Delete old records from cache
            await self.cache_manager.delete_old_records('error_record', cutoff_time)
        except Exception as e:
            logger.error(f"Error deleting old error records: {str(e)}") 