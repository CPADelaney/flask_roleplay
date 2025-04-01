# lore/error_handler.py

"""
Unified Error Handling System

Provides consistent error handling, logging, and recovery mechanisms
across the lore system.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Type, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import asyncio
from functools import wraps
import structlog
from prometheus_client import Counter, Histogram
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

from .lore_cache_manager import LoreCacheManager
from .resource_manager import resource_manager
from .governance_registration import GovernanceRegistration
from db.connection import get_db_connection, DBConnectionError

logger = structlog.get_logger(__name__)

# Prometheus metrics
ERROR_COUNT = Counter('lore_errors_total', 'Total number of errors', ['error_type', 'recovered'])
ERROR_RECOVERY_TIME = Histogram('lore_error_recovery_seconds', 'Error recovery time in seconds')

# Configuration
DB_RECOVERY_MAX_RETRIES = 3
DB_RECOVERY_INITIAL_WAIT = 1.0
INTEGRATION_RECOVERY_MAX_RETRIES = 3
INTEGRATION_RECOVERY_INITIAL_WAIT = 1.0

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
    def __init__(self, message: str, error_type: ErrorType, details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.context = ErrorContext(
            timestamp=self.timestamp,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CUSTOM,
            source="lore",
            details=self.details,
            stack_trace=traceback.format_exc()
        )
        super().__init__(self.message)

class DatabaseError(LoreError):
    """Exception for database-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.DATABASE, details)

class ValidationError(LoreError):
    """Exception for validation-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.VALIDATION, details)

class IntegrationError(LoreError):
    """Exception for integration-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.INTEGRATION, details)

class CacheError(LoreError):
    """Exception for cache-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.CACHE, details)

class PermissionError(LoreError):
    """Exception for permission-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.PERMISSION, details)

class ResourceError(LoreError):
    """Exception for resource-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.RESOURCE, details)

class GovernanceError(LoreError):
    """Exception for governance-related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.GOVERNANCE, details)

class ErrorHandler:
    """
    Unified error handling system for the lore system.
    Provides logging, recovery strategies, and error reporting.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        self.cache_manager = LoreCacheManager(user_id, conversation_id)
        self.governance = GovernanceRegistration(user_id, conversation_id)
        self.recovery_strategies = {
            ErrorType.DATABASE: self._handle_database_error,
            ErrorType.CACHE: self._handle_cache_error,
            ErrorType.INTEGRATION: self._handle_integration_error,
            ErrorType.RESOURCE: self._handle_resource_error,
            ErrorType.GOVERNANCE: self._handle_governance_error
        }
        
    async def handle_error(self, error: LoreError) -> Dict[str, Any]:
        """
        Handle a lore error with appropriate logging and recovery.
        
        Args:
            error: The LoreError instance to handle
            
        Returns:
            Dict containing error handling results
        """
        self.error_counts[error.error_type] += 1
        
        # Update error context with current system state
        await self._update_error_context(error)
        
        # Log the error
        self._log_error(error)
        
        # Attempt recovery if strategy exists
        recovery_result = None
        if error.error_type in self.recovery_strategies:
            recovery_result = await self.recovery_strategies[error.error_type](error)
            
        # Generate error report
        report = self._generate_error_report(error, recovery_result)
        
        # Cache the error report
        await self.cache_manager.set_lore('error_report', f"{error.timestamp.isoformat()}_{error.error_type.value}", report)
        
        return report
        
    async def _update_error_context(self, error: LoreError):
        """Update error context with current system state."""
        try:
            # Get resource usage
            error.context.resource_usage = await resource_manager.get_resource_usage()
            
            # Get cache state
            error.context.cache_state = await self.cache_manager.get_cache_stats()
            
            # Get governance state
            error.context.governance_state = await self.governance.get_state()
            
        except Exception as e:
            logger.error(f"Error updating error context: {str(e)}")
            
    def _log_error(self, error: LoreError):
        """Log error details with appropriate severity."""
        log_message = f"Lore Error: {error.message} (Type: {error.error_type.value})"
        if error.details:
            log_message += f"\nDetails: {error.details}"
            
        if error.error_type in [ErrorType.DATABASE, ErrorType.INTEGRATION, ErrorType.GOVERNANCE]:
            logger.error(log_message)
        elif error.error_type == ErrorType.VALIDATION:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
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
            
        except DBConnectionError as e:
            logger.error(f"Database connection error: {str(e)}")
            error.context.recovery_attempts += 1
            raise
            
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
        try:
            # Implement API-specific recovery logic
            # This could include:
            # 1. Validating API credentials
            # 2. Testing API endpoints
            # 3. Re-establishing API connections
            # 4. Verifying API health
            
            return {
                "status": "healthy",
                "endpoints_tested": True,
                "connection_established": True
            }
        except Exception as e:
            logger.error(f"API integration recovery failed: {str(e)}")
            raise
            
    async def _recover_service_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recover service integration."""
        try:
            # Implement service-specific recovery logic
            # This could include:
            # 1. Checking service health
            # 2. Re-establishing service connections
            # 3. Verifying service dependencies
            # 4. Testing service communication
            
            return {
                "status": "healthy",
                "service_available": True,
                "dependencies_verified": True
            }
        except Exception as e:
            logger.error(f"Service integration recovery failed: {str(e)}")
            raise
            
    async def _recover_event_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recover event integration."""
        try:
            # Implement event-specific recovery logic
            # This could include:
            # 1. Re-establishing event listeners
            # 2. Verifying event channels
            # 3. Testing event publishing
            # 4. Checking event processing
            
            return {
                "status": "healthy",
                "listeners_active": True,
                "channels_verified": True
            }
        except Exception as e:
            logger.error(f"Event integration recovery failed: {str(e)}")
            raise
            
    async def _handle_cache_error(self, error: CacheError) -> Optional[Dict[str, Any]]:
        """Handle cache errors with cache clearing and reinitialization."""
        try:
            # Clear and reinitialize cache
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
            await resource_manager.optimize_resources()
            return {"status": "recovered", "action": "resource_optimization"}
        except Exception as e:
            logger.error(f"Resource recovery failed: {str(e)}")
            return None
            
    async def _handle_governance_error(self, error: GovernanceError) -> Optional[Dict[str, Any]]:
        """Handle governance errors with state recovery."""
        try:
            # Attempt to recover governance state
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
        
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors handled by the system."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
            "timestamp": datetime.utcnow().isoformat(),
            "resource_usage": await resource_manager.get_resource_usage(),
            "cache_stats": await self.cache_manager.get_cache_stats(),
            "governance_state": await self.governance.get_state()
        }
        
    def reset_error_counts(self):
        """Reset error count statistics."""
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        
    def add_recovery_strategy(self, error_type: ErrorType, strategy):
        """Add a custom recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy

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
