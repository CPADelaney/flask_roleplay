"""
Unified Validation System

This module provides centralized validation and error handling for the lore system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Union
from datetime import datetime
from functools import wraps
from pydantic import BaseModel, ValidationError
from enum import Enum
import traceback

from .unified_schemas import (
    BaseEntity,
    World,
    NPC,
    Location,
    Culture,
    Religion,
    Faction,
    Conflict,
    HistoricalEvent,
    Artifact,
    Quest,
    LoreType
)

logger = logging.getLogger(__name__)

class ErrorType(str, Enum):
    """Types of errors in the system"""
    VALIDATION = "validation_error"
    DATABASE = "database_error"
    BUSINESS_LOGIC = "business_logic_error"
    SYSTEM = "system_error"
    RESOURCE = "resource_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict_error"

class LoreError(Exception):
    """Base exception class for lore system errors"""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

class ValidationManager:
    """Manages validation logic and error handling"""
    
    def __init__(self):
        self._model_map = {
            LoreType.WORLD: World,
            LoreType.NPC: NPC,
            LoreType.LOCATION: Location,
            LoreType.CULTURE: Culture,
            LoreType.RELIGION: Religion,
            LoreType.FACTION: Faction,
            LoreType.CONFLICT: Conflict,
            LoreType.ARTIFACT: Artifact,
            LoreType.QUEST: Quest
        }
        self._error_handlers = {}
        self._validation_rules = {}
        self._error_stats = {
            error_type: 0 for error_type in ErrorType
        }
    
    def validate_entity(self, entity_type: LoreType, data: Dict[str, Any]) -> BaseEntity:
        """Validate an entity against its schema"""
        try:
            model_class = self._model_map.get(entity_type)
            if not model_class:
                raise LoreError(
                    f"Unknown entity type: {entity_type}",
                    ErrorType.VALIDATION,
                    {"entity_type": entity_type}
                )
            
            # Apply model validation
            entity = model_class(**data)
            
            # Apply custom validation rules
            if rules := self._validation_rules.get(entity_type):
                for rule in rules:
                    rule(entity)
            
            return entity
        except ValidationError as e:
            self._error_stats[ErrorType.VALIDATION] += 1
            raise LoreError(
                "Validation failed",
                ErrorType.VALIDATION,
                {"errors": e.errors()}
            )
        except Exception as e:
            self._error_stats[ErrorType.SYSTEM] += 1
            raise LoreError(
                str(e),
                ErrorType.SYSTEM,
                {"traceback": traceback.format_exc()}
            )
    
    def add_validation_rule(
        self,
        entity_type: LoreType,
        rule: Callable[[BaseEntity], None]
    ):
        """Add a custom validation rule for an entity type"""
        if entity_type not in self._validation_rules:
            self._validation_rules[entity_type] = []
        self._validation_rules[entity_type].append(rule)
    
    def register_error_handler(
        self,
        error_type: ErrorType,
        handler: Callable[[Exception], Any]
    ):
        """Register an error handler for a specific error type"""
        self._error_handlers[error_type] = handler
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle an error with appropriate error handler"""
        if isinstance(error, LoreError):
            error_type = error.error_type
            self._error_stats[error_type] += 1
            
            if handler := self._error_handlers.get(error_type):
                return handler(error)
            
            return {
                "error": error.message,
                "type": error_type,
                "details": error.details,
                "status_code": error.status_code,
                "timestamp": error.timestamp.isoformat()
            }
        
        # Handle unexpected errors
        self._error_stats[ErrorType.SYSTEM] += 1
        return {
            "error": str(error),
            "type": ErrorType.SYSTEM,
            "details": {
                "traceback": traceback.format_exc(),
                "context": context
            },
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self._error_stats.copy()
    
    def clear_error_stats(self):
        """Clear error statistics"""
        self._error_stats = {
            error_type: 0 for error_type in ErrorType
        }

def validate_request(func: Callable):
    """Decorator for validating request data"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            validation_manager = ValidationManager()
            return validation_manager.handle_error(e)
    return wrapper

# Common validation functions
def validate_coordinates(coords: Dict[str, float]) -> bool:
    """Validate location coordinates"""
    required = {'latitude', 'longitude'}
    if not all(k in coords for k in required):
        return False
    return -90 <= coords['latitude'] <= 90 and -180 <= coords['longitude'] <= 180

def validate_relationship_strength(strength: float) -> bool:
    """Validate relationship strength"""
    return 0 <= strength <= 1

def validate_date_range(start: datetime, end: Optional[datetime]) -> bool:
    """Validate a date range"""
    if end is None:
        return True
    return start <= end 

"""
Unified Validation with Resource Management

This module provides unified validation capabilities with integrated resource management.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from .base_manager import BaseManager
from .resource_manager import resource_manager

logger = logging.getLogger(__name__)

class UnifiedValidation(BaseManager):
    """Manager for unified validation with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.validation_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the unified validation manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the unified validation manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_validation_data(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get validation data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('validation', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting validation data: {e}")
            return None
    
    async def set_validation_data(
        self,
        data_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set validation data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('validation', data_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting validation data: {e}")
            return False
    
    async def invalidate_validation_data(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate validation data cache."""
        try:
            await self.invalidate_cached_data('validation', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating validation data: {e}")
    
    async def get_validation_history(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get validation history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('validation_history', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting validation history: {e}")
            return None
    
    async def set_validation_history(
        self,
        data_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set validation history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('validation_history', data_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting validation history: {e}")
            return False
    
    async def invalidate_validation_history(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate validation history cache."""
        try:
            await self.invalidate_cached_data('validation_history', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating validation history: {e}")
    
    async def get_validation_metadata(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get validation metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('validation_metadata', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting validation metadata: {e}")
            return None
    
    async def set_validation_metadata(
        self,
        data_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set validation metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('validation_metadata', data_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting validation metadata: {e}")
            return False
    
    async def invalidate_validation_metadata(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate validation metadata cache."""
        try:
            await self.invalidate_cached_data('validation_metadata', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating validation metadata: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
unified_validation = UnifiedValidation() 