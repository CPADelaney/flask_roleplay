# lore/validation.py

"""
Unified Validation System

This module provides centralized validation and error handling for the lore system,
integrating schema definitions, validation logic, and resource management.
"""

import logging
import json
import re
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Set, Type, Callable
from datetime import datetime
from enum import Enum
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, lru_cache
import hashlib
from pydantic import BaseModel, ValidationError, Field
from logic.game_time_helper import get_game_timestamp, get_game_datetime

logger = logging.getLogger(__name__)

#---------------------------
# Schema Definitions
#---------------------------

class LoreType(str, Enum):
    """Types of lore content"""
    WORLD = "world"
    NPC = "npc"
    LOCATION = "location"
    EVENT = "event"
    CULTURE = "culture"
    RELIGION = "religion"
    FACTION = "faction"
    CONFLICT = "conflict"
    ARTIFACT = "artifact"
    QUEST = "quest"

class RelationType(str, Enum):
    """Types of relationships between entities"""
    ALLY = "ally"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    FAMILY = "family"
    FRIEND = "friend"
    RIVAL = "rival"
    MENTOR = "mentor"
    STUDENT = "student"
    BUSINESS = "business"
    ROMANTIC = "romantic"

class LocationType(str, Enum):
    """Types of locations"""
    CITY = "city"
    TOWN = "town"
    VILLAGE = "village"
    DUNGEON = "dungeon"
    WILDERNESS = "wilderness"
    LANDMARK = "landmark"
    BUILDING = "building"
    REGION = "region"
    REALM = "realm"

class ConflictType(str, Enum):
    """Types of conflicts"""
    WAR = "war"
    SKIRMISH = "skirmish"
    POLITICAL = "political"
    ECONOMIC = "economic"
    SOCIAL = "social"
    RELIGIOUS = "religious"
    PERSONAL = "personal"

class ConflictStatus(str, Enum):
    """Status of conflicts"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    DORMANT = "dormant"
    ESCALATING = "escalating"
    DEESCALATING = "deescalating"

class BaseEntity(BaseModel):
    """Base model for all lore entities"""
    id: int
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

class World(BaseEntity):
    """World/Setting model"""
    settings: Dict[str, Any]
    rules: Dict[str, Any]
    themes: List[str]
    cultures: List[int] = Field(default_factory=list)  # Culture IDs
    religions: List[int] = Field(default_factory=list)  # Religion IDs
    factions: List[int] = Field(default_factory=list)  # Faction IDs

class NPC(BaseEntity):
    """NPC model"""
    faction_id: Optional[int]
    location_id: Optional[int]
    traits: List[str]
    relationships: Dict[int, Dict[str, Any]]  # NPC ID -> Relationship details
    knowledge: Dict[str, Any]
    status: str = "active"

class Location(BaseEntity):
    """Location model"""
    region_id: Optional[int]
    type: LocationType
    coordinates: Optional[Dict[str, float]]
    properties: Dict[str, Any]
    climate: Optional[str]
    terrain: Optional[str]
    resources: List[str] = Field(default_factory=list)
    points_of_interest: List[int] = Field(default_factory=list)  # POI IDs

class Faction(BaseEntity):
    """Faction model"""
    alignment: str
    influence: float = Field(ge=0, le=1)
    relationships: Dict[int, Dict[str, Any]]  # Faction ID -> Relationship details
    resources: Dict[str, Any]
    goals: List[str]

class Conflict(BaseEntity):
    """Conflict model"""
    type: ConflictType
    status: ConflictStatus
    start_date: datetime
    end_date: Optional[datetime]
    participants: List[Dict[str, Any]]
    causes: List[str]
    effects: List[str]

class HistoricalEvent(BaseEntity):
    """Historical Event model"""
    date: datetime
    significance: str
    participants: List[Dict[str, Any]]
    causes: List[str]
    effects: List[str]
    related_events: List[int] = Field(default_factory=list)  # Event IDs

#---------------------------
# Error Handling
#---------------------------

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

class ErrorCategory(str, Enum):
    """Categories of errors"""
    VALIDATION = "validation"
    RESOURCE = "resource"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    CUSTOM = "custom"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LoreError(Exception):
    """Base exception class for lore system errors"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ):
        self.message = message
        self.error_type = error_type
        self.category = category or ErrorCategory.VALIDATION
        self.severity = severity or ErrorSeverity.ERROR
        self.details = details or {}
        self.status_code = status_code
        self.user_id = user_id
        self.conversation_id = conversation_id
        # Initialized later by initialize_timestamps(); may remain None.
        self.timestamp = None
        self.timestamp_float = None
        super().__init__(self.message)

    async def initialize_timestamps(self):
        """Asynchronously initialize timestamp fields using game time."""
        if self.user_id is None or self.conversation_id is None:
            utc_now = datetime.utcnow()
            self.timestamp = utc_now
            self.timestamp_float = utc_now.timestamp()
            return
        self.timestamp = await get_game_datetime(self.user_id, self.conversation_id)
        self.timestamp_float = await get_game_timestamp(self.user_id, self.conversation_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "error_type": self.error_type,
            "category": self.category,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_float": self.timestamp_float,
            "status_code": self.status_code
        }

#---------------------------
# Validation Context and Results
#---------------------------

@dataclass
class ValidationContext:
    """Context for validation operations"""
    schema_version: str
    validation_rules: Dict[str, Any]
    custom_validators: Dict[str, callable]
    reference_cache: Dict[str, Any]
    validation_mode: str = "strict"  # strict, lenient, or custom
    max_parallel_validations: int = 10
    cache_ttl: int = 3600  # 1 hour in seconds

class ValidationResult:
    """Result of a validation operation"""
    def __init__(self):
        self.is_valid = True
        self.errors: List[LoreError] = []
        self.warnings: List[str] = []
        self.validation_time = 0.0
        self.recovered = False
        self.recovery_attempts = 0
        self.cache_hit = False
    
    def add_error(self, error: LoreError):
        """Add a validation error"""
        self.is_valid = False
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a validation warning"""
        self.warnings.append(warning)
    
    def set_validation_time(self, time: float):
        """Set the validation time"""
        self.validation_time = time

#---------------------------
# Resource Cache
#---------------------------

class ResourceCache:
    """Cache for validation resources with expiration"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.access_times = {}
    
    async def get(self, key: str) -> Any:
        """Get an item from the cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry > datetime.utcnow().timestamp():
                # Update access time for LRU
                self.access_times[key] = datetime.utcnow().timestamp()
                return value
            # Remove expired item
            self._remove_key(key)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set an item in the cache"""
        expiry = datetime.utcnow().timestamp() + (ttl or self.default_ttl)
        
        # Manage cache size - use LRU strategy
        if len(self.cache) >= self.max_size:
            # Find oldest accessed item
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_key(oldest_key)
        
        self.cache[key] = (value, expiry)
        self.access_times[key] = datetime.utcnow().timestamp()
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a specific key"""
        self._remove_key(key)
    
    async def clear(self) -> None:
        """Clear all items from the cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def _remove_key(self, key: str) -> None:
        """Remove a key from both cache and access times"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]

#---------------------------
# Validation Manager
#---------------------------

class ValidationManager:
    """
    Unified validation manager with integrated resource management and caching.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize validation resources
        self._model_map = {
            LoreType.WORLD: World,
            LoreType.NPC: NPC,
            LoreType.LOCATION: Location,
            LoreType.FACTION: Faction,
            LoreType.CONFLICT: Conflict,
            LoreType.EVENT: HistoricalEvent,
        }
        self._custom_validators: Dict[str, List[callable]] = {}
        self._validation_semaphore = None  # Will be initialized as asyncio.Semaphore
        self._active_validations: Set[str] = set()
        self._validation_stats = {
            error_type.value: 0 for error_type in ErrorType
        }
        self._validation_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4) if config else 4)
        self._resource_cache = ResourceCache(
            max_size=config.get('cache_max_size', 1000) if config else 1000,
            ttl=config.get('cache_ttl', 3600) if config else 3600
        )
    
    async def initialize(self):
        """Initialize the validation manager"""
        max_parallel = self.config.get('max_parallel_validations', 10) if self.config else 10
        self._validation_semaphore = asyncio.Semaphore(max_parallel)
        await self._load_schemas()
        await self._load_custom_validators()
    
    async def cleanup(self):
        """Cleanup validation resources"""
        self._validation_pool.shutdown(wait=True)
        await self._resource_cache.clear()
    
    async def _load_schemas(self):
        """Load validation schemas"""
        # Load schema paths and versions from config
        schema_path = self.config.get('schema_path', 'schemas') if self.config else 'schemas'
        # Actual implementation would load schemas from files or database
    
    async def _load_custom_validators(self):
        """Load custom validation functions"""
        # Load validators from config or registry
        validators_path = self.config.get('validators_path', 'validators') if self.config else 'validators'
        # Actual implementation would load validators from files or registry
    
    async def validate(
        self,
        entity_type: Union[LoreType, str],
        data: Dict[str, Any],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate data against schemas and rules with caching and resource management.
        
        Args:
            entity_type: Type of entity to validate
            data: Data to validate
            context: Optional validation context
            
        Returns:
            ValidationResult with validation results
        """
        # Convert string entity_type to enum if needed
        if isinstance(entity_type, str):
            try:
                entity_type = LoreType(entity_type)
            except ValueError:
                raise LoreError(
                    f"Unknown entity type: {entity_type}",
                    ErrorType.VALIDATION,
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"entity_type": entity_type}
                )
        
        # Create default context if not provided
        if context is None:
            context = ValidationContext(
                schema_version="latest",
                validation_rules={},
                custom_validators={},
                reference_cache={}
            )
        
        result = ValidationResult()
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(entity_type, data, context)
            cached_result = await self._resource_cache.get(cache_key)
            if cached_result:
                result = cached_result
                result.cache_hit = True
                return result
            
            # Check for concurrent validation of the same data
            data_id = self._get_data_id(data)
            if data_id in self._active_validations:
                result.add_warning(f"Concurrent validation detected for {data_id}")
            
            self._active_validations.add(data_id)
            
            try:
                # Use semaphore to limit parallel validations
                async with self._validation_semaphore:
                    # Perform schema validation
                    model_class = self._model_map.get(entity_type)
                    if not model_class:
                        raise LoreError(
                            f"Unknown entity type: {entity_type}",
                            ErrorType.VALIDATION,
                            ErrorCategory.VALIDATION,
                            ErrorSeverity.ERROR,
                            {"entity_type": str(entity_type)}
                        )
                    
                    # Apply model validation
                    try:
                        entity = model_class(**data)
                    except ValidationError as e:
                        self._validation_stats[ErrorType.VALIDATION.value] += 1
                        raise LoreError(
                            "Validation failed",
                            ErrorType.VALIDATION,
                            ErrorCategory.VALIDATION,
                            ErrorSeverity.ERROR,
                            {"errors": [{"loc": err["loc"], "msg": err["msg"]} for err in e.errors()]}
                        )
                    
                    # Run validation tasks in parallel
                    validation_tasks = [
                        self._validate_model(entity, model_class),
                        self._validate_references(entity, entity_type, context),
                        self._validate_custom_rules(entity, entity_type, context)
                    ]
                    
                    validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                    
                    # Process validation results
                    for task_result in validation_results:
                        if isinstance(task_result, Exception):
                            if isinstance(task_result, LoreError):
                                result.add_error(task_result)
                            else:
                                result.add_error(LoreError(
                                    str(task_result),
                                    ErrorType.VALIDATION,
                                    ErrorCategory.VALIDATION,
                                    ErrorSeverity.ERROR,
                                    {"task": task_result.__class__.__name__}
                                ))
            finally:
                self._active_validations.discard(data_id)
            
            # Set validation time
            validation_time = time.time() - start_time
            result.set_validation_time(validation_time)
            
            # Cache the result
            await self._resource_cache.set(cache_key, result)
            
            return result
        except LoreError as e:
            # Handle expected errors
            self._validation_stats[e.error_type.value] += 1
            result.add_error(e)
            result.set_validation_time(time.time() - start_time)
            return result
        except Exception as e:
            # Handle unexpected errors
            self._validation_stats[ErrorType.SYSTEM.value] += 1
            error = LoreError(
                str(e),
                ErrorType.SYSTEM,
                ErrorCategory.SYSTEM,
                ErrorSeverity.ERROR,
                {"traceback": traceback.format_exc()}
            )
            result.add_error(error)
            result.set_validation_time(time.time() - start_time)
            return result
    
    async def _validate_model(self, entity: BaseEntity, model_class: Type[BaseEntity]) -> None:
        """Validate model against schema"""
        # This validates that the entity conforms to the model structure
        # The pydantic validation should have already caught basic issues
        pass
    
    async def _validate_references(self, entity: BaseEntity, entity_type: LoreType, context: ValidationContext) -> None:
        """Validate references in the entity"""
        # Check that any referenced IDs exist in the system
        pass
    
    async def _validate_custom_rules(self, entity: BaseEntity, entity_type: LoreType, context: ValidationContext) -> None:
        """Validate entity against custom rules"""
        validators = context.custom_validators.get(str(entity_type), [])
        if not validators:
            validators = self._custom_validators.get(str(entity_type), [])
        
        for validator in validators:
            try:
                validator(entity)
            except Exception as e:
                raise LoreError(
                    str(e),
                    ErrorType.VALIDATION,
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"validator": validator.__name__}
                )
    
    def register_validator(
        self,
        entity_type: Union[LoreType, str],
        validator: Callable[[BaseEntity], None]
    ) -> None:
        """
        Register a custom validator for an entity type.
        
        Args:
            entity_type: Type of entity to validate
            validator: Validation function that takes an entity and raises an exception if invalid
        """
        if isinstance(entity_type, LoreType):
            entity_type = entity_type.value
        
        if entity_type not in self._custom_validators:
            self._custom_validators[entity_type] = []
        
        self._custom_validators[entity_type].append(validator)
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return self._validation_stats.copy()
    
    def clear_validation_stats(self) -> None:
        """Clear validation statistics"""
        self._validation_stats = {
            error_type.value: 0 for error_type in ErrorType
        }
    
    def _get_cache_key(
        self,
        entity_type: LoreType,
        data: Dict[str, Any],
        context: ValidationContext
    ) -> str:
        """Generate a cache key for the validation result"""
        data_str = json.dumps(data, sort_keys=True)
        context_str = json.dumps({
            'schema_version': context.schema_version,
            'validation_mode': context.validation_mode
        }, sort_keys=True)
        key_input = f"{entity_type}:{data_str}:{context_str}"
        return hashlib.sha256(key_input.encode()).hexdigest()
    
    def _get_data_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for the data"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

#---------------------------
# Validation Helper Functions
#---------------------------

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

def validate_request(func: Callable):
    """Decorator for validating request data"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_type = getattr(e, "error_type", ErrorType.SYSTEM)
            error_category = getattr(e, "category", ErrorCategory.SYSTEM)
            error_severity = getattr(e, "severity", ErrorSeverity.ERROR)
            
            return {
                "error": str(e),
                "type": error_type,
                "category": error_category,
                "severity": error_severity,
                "details": getattr(e, "details", {"traceback": traceback.format_exc()}),
                "status_code": getattr(e, "status_code", 500),
                "timestamp": datetime.utcnow().isoformat()
            }
    return wrapper

# Create global validation manager instance
validation_manager = ValidationManager()
