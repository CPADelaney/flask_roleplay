# lore/validation.py

"""
Unified Validation System

This module provides centralized validation and error handling for the lore system,
integrating schema definitions and validation logic.
"""

import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Type, Callable
from datetime import datetime
from enum import Enum
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, lru_cache
import hashlib
from pydantic import BaseModel, ValidationError, Field

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
        orm_mode = True

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

#---------------------------
# Validation Context
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
# Validation Manager
#---------------------------

class ValidationManager:
    """
    Manages validation operations with integrated resource management.
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
        }
        self._custom_validators: Dict[str, callable] = {}
        self._validation_semaphore = None  # Will be initialized as asyncio.Semaphore
        self._active_validations: Set[str] = set()
        self._validation_stats = {
            error_type: 0 for error_type in ErrorType
        }
        self._validation_cache = {}
        self._validation_history = {}
    
    async def initialize(self):
        """Initialize the validation manager"""
        self._validation_semaphore = asyncio.Semaphore(
            self.config.get('max_parallel_validations', 10)
        )
        await self._load_schemas()
        await self._load_custom_validators()
    
    async def _load_schemas(self):
        """Load validation schemas"""
        # Implementation would depend on your schema storage mechanism
        pass
    
    async def _load_custom_validators(self):
        """Load custom validation functions"""
        # Implementation would depend on your validator registration mechanism
        pass
    
    async def validate(
        self,
        entity_type: LoreType,
        data: Dict[str, Any],
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Validate data against schemas and rules with caching and resource management.
        """
        # Create default context if not provided
        if context is None:
            context = ValidationContext(
                schema_version="latest",
                validation_rules={},
                custom_validators={},
                reference_cache={}
            )
        
        result = ValidationResult()
        start_time = datetime.utcnow().timestamp()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(entity_type, data, context)
            cached_result = self._validation_cache.get(cache_key)
            if cached_result:
                # Check if cached result is still valid
                cache_time, cached_validation = cached_result
                if datetime.utcnow().timestamp() - cache_time < context.cache_ttl:
                    result = cached_validation
                    result.cache_hit = True
                    return result
            
            # Check for concurrent validation of the same data
            data_id = self._get_data_id(data)
            if data_id in self._active_validations:
                raise LoreError(
                    "Concurrent validation detected",
                    ErrorType.VALIDATION,
                    {"data_id": data_id}
                )
            
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
                            {"entity_type": entity_type}
                        )
                    
                    # Apply model validation
                    try:
                        entity = model_class(**data)
                    except ValidationError as e:
                        self._validation_stats[ErrorType.VALIDATION] += 1
                        raise LoreError(
                            "Validation failed",
                            ErrorType.VALIDATION,
                            {"errors": [{"loc": err["loc"], "msg": err["msg"]} for err in e.errors()]}
                        )
                    
                    # Apply custom validation rules
                    custom_validators = context.custom_validators or self._custom_validators
                    if validators := custom_validators.get(entity_type):
                        for validator in validators:
                            try:
                                validator(entity)
                            except Exception as e:
                                result.add_error(LoreError(
                                    str(e),
                                    ErrorType.VALIDATION,
                                    {"validator": validator.__name__}
                                ))
            finally:
                self._active_validations.discard(data_id)
            
            # Update validation statistics
            if not result.is_valid:
                self._validation_stats[ErrorType.VALIDATION] += 1
            
            # Set validation time
            validation_time = datetime.utcnow().timestamp() - start_time
            result.set_validation_time(validation_time)
            
            # Cache the result
            self._validation_cache[cache_key] = (datetime.utcnow().timestamp(), result)
            
            return result
        except LoreError as e:
            # Handle expected errors
            self._validation_stats[e.error_type] += 1
            result.add_error(e)
            return result
        except Exception as e:
            # Handle unexpected errors
            self._validation_stats[ErrorType.SYSTEM] += 1
            result.add_error(LoreError(
                str(e),
                ErrorType.SYSTEM,
                {"traceback": traceback.format_exc()}
            ))
            return result
    
    def add_validation_rule(
        self,
        entity_type: LoreType,
        rule: Callable[[BaseEntity], None]
    ):
        """Add a custom validation rule for an entity type"""
        if entity_type not in self._custom_validators:
            self._custom_validators[entity_type] = []
        self._custom_validators[entity_type].append(rule)
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self._validation_stats.copy()
    
    def clear_validation_stats(self):
        """Clear error statistics"""
        self._validation_stats = {
            error_type: 0 for error_type in ErrorType
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

def validate_request(func: Callable):
    """Decorator for validating request data"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            validation_manager = ValidationManager()
            return {
                "error": str(e),
                "type": getattr(e, "error_type", ErrorType.SYSTEM),
                "details": getattr(e, "details", {"traceback": traceback.format_exc()}),
                "status_code": getattr(e, "status_code", 500),
                "timestamp": datetime.utcnow().isoformat()
            }
    return wrapper

# Create global validation manager
validation_manager = ValidationManager()
