# lore/error_manager.py

"""
Unified Error Handling and Recovery System

Provides consistent error handling, logging, recovery mechanisms,
and monitoring across the lore system.
"""

import logging
import traceback
import asyncio
import json
from typing import Dict, Any, Optional, List, Type, Callable, Union, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import psutil
from functools import wraps
import structlog
from prometheus_client import Counter, Histogram, Gauge
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

# Prometheus metrics
ERROR_COUNT = Counter('lore_errors_total', 'Total number of errors', ['error_type', 'recovered'])
ERROR_RECOVERY_TIME = Histogram('lore_error_recovery_seconds', 'Error recovery time in seconds')
RESOURCE_USAGE = Gauge('lore_resource_usage', 'Resource usage metrics', ['resource_type'])
SYSTEM_HEALTH = Gauge('lore_system_health', 'System health status', ['component'])

# Configuration
DB_RECOVERY_MAX_RETRIES = 3
DB_RECOVERY_INITIAL_WAIT = 1.0
INTEGRATION_RECOVERY_MAX_RETRIES = 3
INTEGRATION_RECOVERY_INITIAL_WAIT = 1.0
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

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors"""
    VALIDATION = "validation"
    RESOURCE = "resource"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    CUSTOM = "custom"

class ErrorType(Enum):
    """Types of errors that can occur in the lore system."""
    DATABASE = "database"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    CACHE = "cache"
    PERMISSION = "permission"
    RESOURCE = "resource"
    GOVERNANCE = "governance"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    source: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    recovered: bool = False
    recovery_time: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None
    cache_state: Optional[Dict[str, Any]] = None
    governance_state: Optional[Dict[str, Any]] = None

class LoreError(Exception):
    """Base exception for all lore-related errors."""
    def __init__(self, message: str, error_type: Optional[ErrorType] = None, 
                 category: Optional[ErrorCategory] = None, 
                 severity: Optional[ErrorSeverity] = None,
                 details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type or ErrorType.UNKNOWN
        self.category = category or ErrorCategory.CUSTOM
        self.severity = severity or ErrorSeverity.ERROR
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.context = ErrorContext(
            timestamp=self.timestamp,
            severity=self.severity,
            category=self.category,
            source="lore",
            details=self.details,
            stack_trace=traceback.format_exc()
        )
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "error_type": self.error_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.context.stack_trace
        }

# Define specific error types
class DatabaseError(LoreError):
    """Exception for database-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.DATABASE, ErrorCategory.DATABASE, ErrorSeverity.ERROR, details)

class ValidationError(LoreError):
    """Exception for validation-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.VALIDATION, ErrorCategory.VALIDATION, ErrorSeverity.ERROR, details)

class IntegrationError(LoreError):
    """Exception for integration-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.INTEGRATION, ErrorCategory.NETWORK, ErrorSeverity.ERROR, details)

class CacheError(LoreError):
    """Exception for cache-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.CACHE, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, details)

class PermissionError(LoreError):
    """Exception for permission-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.PERMISSION, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, details)

class ResourceError(LoreError):
    """Exception for resource-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.RESOURCE, ErrorCategory.RESOURCE, ErrorSeverity.ERROR, details)

class GovernanceError(LoreError):
    """Exception for governance-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.GOVERNANCE, ErrorCategory.SYSTEM, ErrorSeverity.ERROR, details)

class ErrorHandler:
    """
    Unified error handling system for the lore system.
    Provides logging, recovery strategies, and error reporting.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, 
                 config: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = config or {}
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        self.cache_manager = None  # Set externally if needed
        self.governance = None  # Set externally if needed
        self.recovery_strategies = {
            ErrorType.DATABASE: self._handle_database_error,
            ErrorType.CACHE: self._handle_cache_error,
            ErrorType.INTEGRATION: self._handle_integration_error,
            ErrorType.RESOURCE: self._handle_resource_error,
            ErrorType.GOVERNANCE: self._handle_governance_error
        }
        self.resource_monitor = ResourceMonitor(user_id, conversation_id)
        self.alert_client = None  # Initialize when needed
        self._monitoring_tasks = []
        self._is_monitoring = False
        self._operation_counts = {}
        self._operation_timestamps = {}
        self._error_history = []
        self._max_error_history = self.config.get('max_error_history', 1000)
    
    async def start_monitoring(self):
        """Start monitoring tasks."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        
        # Initialize alert client if needed
        if not self.alert_client:
            import aiohttp
            self.alert_client = aiohttp.ClientSession()
        
        # Start monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._monitor_error_rates()),
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_resource_usage()),
            asyncio.create_task(self._cleanup_old_errors())
        ]
        
        logger.info("Error monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        # Close alert client
        if self.alert_client:
            await self.alert_client.close()
            self.alert_client = None
        
        logger.info("Error monitoring stopped")
    
    async def handle_error(self, error: LoreError) -> Dict[str, Any]:
        """
        Handle a lore error with appropriate logging and recovery.
        
        Args:
            error: The LoreError instance to handle
            
        Returns:
            Dict containing error handling results
        """
        self.error_counts[error.error_type] += 1
        
        # Add to error history
        self._add_to_error_history(error)
        
        # Update error context with current system state
        await self._update_error_context(error)
        
        # Log the error
        self._log_error(error)
        
        # Update metrics
        ERROR_COUNT.labels(error_type=error.error_type.value, recovered="false").inc()
        
        # Attempt recovery if strategy exists
        recovery_result = None
        if error.error_type in self.recovery_strategies:
            try:
                recovery_result = await self.recovery_strategies[error.error_type](error)
                if recovery_result and recovery_result.get('status') == 'recovered':
                    ERROR_COUNT.labels(error_type=error.error_type.value, recovered="true").inc()
                    error.context.recovered = True
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {str(recovery_error)}")
        
        # Generate error report
        report = self._generate_error_report(error, recovery_result)
        
        # Cache the error report if cache manager is available
        if self.cache_manager:
            await self.cache_manager.set('error_report', f"{error.timestamp.isoformat()}_{error.error_type.value}", report)
        
        return report
    
    def _add_to_error_history(self, error: LoreError):
        """Add error to history with limits."""
        self._error_history.append(error)
        
        # Trim history if needed
        if len(self._error_history) > self._max_error_history:
            self._error_history = self._error_history[-self._max_error_history:]
    
    async def _update_error_context(self, error: LoreError):
        """Update error context with current system state."""
        try:
            # Get resource usage
            error.context.resource_usage = await self.resource_monitor.get_resource_usage()
            
            # Get cache state if cache manager is available
            if self.cache_manager:
                error.context.cache_state = await self.cache_manager.get_cache_stats()
            
            # Get governance state if governance is available
            if self.governance:
                error.context.governance_state = await self.governance.get_state()
            
        except Exception as e:
            logger.error(f"Error updating error context: {str(e)}")
    
    def _log_error(self, error: LoreError):
        """Log error details with appropriate severity."""
        log_message = f"Lore Error: {error.message} (Type: {error.error_type.value})"
        if error.details:
            log_message += f"\nDetails: {json.dumps(error.details)}"
            
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        else:  # DEBUG
            logger.debug(log_message)
    
    @retry(
        stop=stop_after_attempt(DB_RECOVERY_MAX_RETRIES),
        wait=wait_exponential(multiplier=DB_RECOVERY_INITIAL_WAIT)
    )
    async def _handle_database_error(self, error: DatabaseError) -> Optional[Dict[str, Any]]:
        """Handle database errors with retry logic and connection recovery."""
        try:
            # Get connection details from error context
            connection_details = error.details.get('connection_details', {})
            
            # Attempt to establish new connection
            from db.connection import get_db_connection
            connection = await get_db_connection(**connection_details)
            
            # Verify connection is working
            await connection.execute('SELECT 1')
            
            # Log successful recovery
            logger.info(f"Database connection recovered after {error.context.recovery_attempts} attempts")
            
            return {
                "status": "recovered",
                "action": "connection_reset",
                "attempts": error.context.recovery_attempts,
                "connection_id": id(connection)
            }
            
        except Exception as e:
            logger.error(f"Database recovery failed: {str(e)}")
            error.context.recovery_attempts += 1
            raise
    
    @retry(
        stop=stop_after_attempt(INTEGRATION_RECOVERY_MAX_RETRIES),
        wait=wait_exponential(multiplier=INTEGRATION_RECOVERY_INITIAL_WAIT)
    )
    async def _handle_integration_error(self, error: IntegrationError) -> Optional[Dict[str, Any]]:
        """Handle integration errors with retry logic and fallback options."""
        try:
            # Get integration details from error context
            integration_type = error.details.get('integration_type')
            integration_config = error.details.get('integration_config', {})
            
            if not integration_type:
                raise ValueError("Integration type not specified in error details")
            
            # Attempt to recover based on integration type
            if integration_type == 'api':
                # Handle API integration recovery
                result = await self._recover_api_integration(integration_config)
            elif integration_type == 'service':
                # Handle service integration recovery
                result = await self._recover_service_integration(integration_config)
            elif integration_type == 'event':
                # Handle event integration recovery
                result = await self._recover_event_integration(integration_config)
            else:
                raise ValueError(f"Unknown integration type: {integration_type}")
            
            # Log successful recovery
            logger.info(f"Integration {integration_type} recovered after {error.context.recovery_attempts} attempts")
            
            return {
                "status": "recovered",
                "action": "integration_retry",
                "attempts": error.context.recovery_attempts,
                "integration_type": integration_type,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Integration recovery failed: {str(e)}")
            error.context.recovery_attempts += 1
            raise
    
    async def _recover_api_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recover API integration."""
        # Mock implementation - replace with actual logic
        return {
            "status": "healthy",
            "endpoints_tested": True,
            "connection_established": True
        }
    
    async def _recover_service_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recover service integration."""
        # Mock implementation - replace with actual logic
        return {
            "status": "healthy",
            "service_available": True,
            "dependencies_verified": True
        }
    
    async def _recover_event_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recover event integration."""
        # Mock implementation - replace with actual logic
        return {
            "status": "healthy",
            "listeners_active": True,
            "channels_verified": True
        }
    
    async def _handle_cache_error(self, error: CacheError) -> Optional[Dict[str, Any]]:
        """Handle cache errors with cache clearing and reinitialization."""
        try:
            # Clear and reinitialize cache if cache manager is available
            if self.cache_manager:
                await self.cache_manager.clear_cache()
                await self.cache_manager.initialize()
            return {"status": "recovered", "action": "cache_reset"}
        except Exception as e:
            logger.error(f"Cache recovery failed: {str(e)}")
            return None
    
    async def _handle_resource_error(self, error: ResourceError) -> Optional[Dict[str, Any]]:
        """Handle resource errors with resource optimization."""
        try:
            # Attempt to optimize resource usage
            await self.resource_monitor.optimize_resources()
            return {"status": "recovered", "action": "resource_optimization"}
        except Exception as e:
            logger.error(f"Resource recovery failed: {str(e)}")
            return None
    
    async def _handle_governance_error(self, error: GovernanceError) -> Optional[Dict[str, Any]]:
        """Handle governance errors with state recovery."""
        try:
            # Attempt to recover governance state if governance is available
            if self.governance:
                await self.governance.recover_state()
            return {"status": "recovered", "action": "governance_recovery"}
        except Exception as e:
            logger.error(f"Governance recovery failed: {str(e)}")
            return None
    
    def _generate_error_report(self, error: LoreError, recovery_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive error report."""
        return {
            "error_type": error.error_type.value,
            "message": error.message,
            "timestamp": error.timestamp.isoformat(),
            "details": error.details,
            "context": {
                "resource_usage": error.context.resource_usage,
                "cache_state": error.context.cache_state,
                "governance_state": error.context.governance_state
            },
            "recovery_attempted": recovery_result is not None,
            "recovery_result": recovery_result,
            "error_count": self.error_counts[error.error_type]
        }
    
    async def _monitor_error_rates(self):
        """Monitor error rates and trigger alerts if needed."""
        while self._is_monitoring:
            try:
                # Calculate error rates
                current_time = datetime.utcnow()
                total_operations = sum(self._operation_counts.values())
                
                if total_operations > 0:
                    total_errors = sum(self.error_counts.values())
                    error_rate = total_errors / total_operations
                    
                    # Trigger alert if error rate is too high
                    if error_rate > ERROR_RATE_THRESHOLD:
                        logger.warning(f"High error rate detected: {error_rate:.2%}")
                        await self._trigger_error_rate_alert({
                            "error_rate": error_rate,
                            "threshold": ERROR_RATE_THRESHOLD,
                            "total_errors": total_errors,
                            "total_operations": total_operations
                        })
                
                await asyncio.sleep(MONITORING_INTERVALS['error_rates'])
            except Exception as e:
                logger.error(f"Error monitoring error rates: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['error_rates'])
    
    async def _monitor_system_health(self):
        """Monitor overall system health."""
        while self._is_monitoring:
            try:
                # Check system resources
                resource_usage = await self.resource_monitor.get_resource_usage()
                
                # Check for high resource usage
                for resource, usage in resource_usage.items():
                    RESOURCE_USAGE.labels(resource_type=resource).set(usage)
                    if usage > RESOURCE_USAGE_THRESHOLD:
                        logger.warning(f"High {resource} usage detected: {usage}%")
                        await self._trigger_resource_alert({resource: usage})
                
                # Check system components
                components = {
                    "database": await self._check_database_health(),
                    "cache": await self._check_cache_health(),
                    "governance": await self._check_governance_health()
                }
                
                # Update component health metrics
                for component, health in components.items():
                    SYSTEM_HEALTH.labels(component=component).set(1 if health else 0)
                    if not health:
                        logger.warning(f"Unhealthy component detected: {component}")
                        await self._trigger_system_alert({"component": component, "healthy": False})
                
                await asyncio.sleep(MONITORING_INTERVALS['system_health'])
            except Exception as e:
                logger.error(f"Error monitoring system health: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['system_health'])
    
    async def _monitor_resource_usage(self):
        """Monitor resource usage and trigger alerts if needed."""
        while self._is_monitoring:
            try:
                usage = await self.resource_monitor.get_resource_usage()
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
                self._error_history = [
                    error for error in self._error_history 
                    if error.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(MONITORING_INTERVALS['cleanup'])
            except Exception as e:
                logger.error(f"Error cleaning up old errors: {str(e)}")
                await asyncio.sleep(MONITORING_INTERVALS['cleanup'])
    
    async def _check_database_health(self) -> bool:
        """Check database health."""
        # Mock implementation - replace with actual check
        return True
    
    async def _check_cache_health(self) -> bool:
        """Check cache health."""
        # Mock implementation - replace with actual check
        return True
    
    async def _check_governance_health(self) -> bool:
        """Check governance health."""
        # Mock implementation - replace with actual check
        return True
    
    async def _trigger_error_rate_alert(self, data: Dict[str, Any]):
        """Trigger an alert for high error rates."""
        try:
            if not self.alert_client:
                return
                
            alert_data = {
                'type': 'error_rate',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'data': data
            }
            
            async with self.alert_client.post(ALERT_ENDPOINTS['error_rate'], json=alert_data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send error rate alert: {await response.text()}")
        except Exception as e:
            logger.error(f"Error triggering error rate alert: {str(e)}")
    
    async def _trigger_resource_alert(self, resource_usage: Dict[str, float]):
        """Trigger an alert for high resource usage."""
        try:
            if not self.alert_client:
                return
                
            alert_data = {
                'type': 'resource_usage',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'data': {
                    'usage': resource_usage,
                    'threshold': RESOURCE_USAGE_THRESHOLD
                }
            }
            
            async with self.alert_client.post(ALERT_ENDPOINTS['resource'], json=alert_data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send resource alert: {await response.text()}")
        except Exception as e:
            logger.error(f"Error triggering resource alert: {str(e)}")
    
    async def _trigger_system_alert(self, system_state: Dict[str, Any]):
        """Trigger an alert for unhealthy system state."""
        try:
            if not self.alert_client:
                return
                
            alert_data = {
                'type': 'system_state',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'data': system_state
            }
            
            async with self.alert_client.post(ALERT_ENDPOINTS['system'], json=alert_data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send system alert: {await response.text()}")
        except Exception as e:
            logger.error(f"Error triggering system alert: {str(e)}")
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors handled by the system."""
        try:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_counts": {k.value: v for k, v in self.error_counts.items()},
                "timestamp": datetime.utcnow().isoformat(),
                "resource_usage": await self.resource_monitor.get_resource_usage(),
                "error_distribution": self._get_error_distribution()
            }
        except Exception as e:
            logger.error(f"Error getting error statistics: {str(e)}")
            return {
                "total_errors": 0,
                "error": str(e)
            }
    
    def _get_error_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by category."""
        distribution = {}
        for error in self._error_history:
            category = error.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def reset_error_counts(self):
        """Reset error count statistics."""
        self.error_counts = {error_type: 0 for error_type in ErrorType}
    
    def add_recovery_strategy(self, error_type: ErrorType, strategy: Callable):
        """Add a custom recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy

class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            process = psutil.Process()
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Get disk usage
            disk_usage = psutil.disk_usage('/')
            
            return {
                'memory': memory_percent,
                'cpu': cpu_percent,
                'disk': disk_usage.percent,
                'memory_rss': memory_info.rss / (1024 * 1024),  # MB
                'memory_vms': memory_info.vms / (1024 * 1024)   # MB
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {str(e)}")
            return {
                'memory': 0,
                'cpu': 0,
                'disk': 0
            }
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            # Release memory
            import gc
            gc.collect()
            
            # Close any unnecessary file handles
            for proc in psutil.process_iter(['pid', 'open_files']):
                if proc.info['pid'] == os.getpid():
                    for file in proc.info['open_files'] or []:
                        try:
                            os.close(file.fd)
                        except:
                            pass
            
            logger.info("Resources optimized")
        except Exception as e:
            logger.error(f"Error optimizing resources: {str(e)}")

def handle_errors(
    error_types: Optional[List[Type[Exception]]] = None,
    max_retries: int = 3,
    delay: float = 1.0
):
    """Decorator for handling errors in async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                    continue
            
            if error_types and not any(isinstance(last_error, t) for t in error_types):
                raise last_error
            
            return last_error
        return wrapper
    return decorator
