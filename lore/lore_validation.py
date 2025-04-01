# lore/lore_validation.py

"""
Lore Validation Module

This module handles validation of lore content, ensuring consistency
and adherence to rules and constraints.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import json
import re
import asyncio
from enum import Enum
import traceback
from functools import wraps, lru_cache
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import jsonschema
from jsonschema import validate, ValidationError as JSONSchemaError
import aiohttp
import backoff
import hashlib
from prometheus_client import Counter, Histogram

from .resource_manager import ResourceManager, ResourceConfig, CacheConfig
from .error_handler import (
    LoreError, ErrorHandler, ErrorCategory, ErrorSeverity,
    handle_errors, ErrorContext
)

logger = logging.getLogger(__name__)

# Prometheus metrics
VALIDATION_COUNT = Counter('lore_validations_total', 'Total number of validations', ['status'])
VALIDATION_TIME = Histogram('lore_validation_duration_seconds', 'Validation duration in seconds')

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

class ValidationManager:
    """Manages validation operations and resources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize resource manager with cache configurations
        cache_configs = {
            'validation': CacheConfig(
                name='validation',
                max_size=1000,
                ttl=3600,
                eviction_policy='lru'
            ),
            'schema': CacheConfig(
                name='schema',
                max_size=100,
                ttl=3600,
                eviction_policy='ttl'
            )
        }
        
        resource_config = ResourceConfig(
            caches=cache_configs,
            cleanup_interval=300,
            validation_batch_size=50,
            performance_monitoring=True,
            rate_limit_requests=100,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60,
            monitoring_interval=60
        )
        
        self._resource_manager = ResourceManager(resource_config)
        self._error_handler = ErrorHandler(config.get('error_handler', {}))
        self._custom_validators: Dict[str, callable] = {}
        self._validation_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self._active_validations: Set[str] = set()
        self._validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0
        }
        self._validation_semaphore = asyncio.Semaphore(config.get('max_parallel_validations', 10))
        
        # Register recovery strategies
        self._error_handler.register_recovery_strategy(
            ErrorCategory.VALIDATION,
            self._recover_validation_error
        )
    
    async def initialize(self):
        """Initialize the validation manager"""
        await self._resource_manager.initialize()
        await self._load_schemas()
        await self._load_custom_validators()
    
    async def cleanup(self):
        """Cleanup validation resources"""
        await self._resource_manager.cleanup()
        self._validation_pool.shutdown(wait=True)
    
    @lru_cache(maxsize=100)
    async def _load_schemas(self):
        """Load validation schemas with caching"""
        try:
            schema_path = self.config.get('schema_path', 'schemas')
            # Implement schema loading logic here
            pass
        except Exception as e:
            error = LoreError(
                message=f"Failed to load schemas: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'schema_path': schema_path}
            )
            await self._error_handler.handle_error(error)
            raise
    
    async def _load_custom_validators(self):
        """Load custom validation functions"""
        try:
            validators_path = self.config.get('validators_path', 'validators')
            # Implement custom validator loading logic here
            pass
        except Exception as e:
            error = LoreError(
                message=f"Failed to load custom validators: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'validators_path': validators_path}
            )
            await self._error_handler.handle_error(error)
            raise
    
    @handle_errors(error_types=[LoreError], max_retries=3)
    async def validate(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Validate data against schemas and rules with performance optimizations"""
        result = ValidationResult()
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(data, context)
            if cached_result := await self._resource_manager.get('validation', cache_key):
                result = cached_result
                result.cache_hit = True
                self._validation_stats['total_validations'] += 1
                VALIDATION_COUNT.labels(status='cache_hit').inc()
                return result
            
            self._validation_stats['total_validations'] += 1
            VALIDATION_COUNT.labels(status='cache_miss').inc()
            
            # Check for concurrent validation
            data_id = self._get_data_id(data)
            if data_id in self._active_validations:
                raise LoreError(
                    message="Concurrent validation detected",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.WARNING,
                    details={'data_id': data_id}
                )
            
            self._active_validations.add(data_id)
            
            # Use semaphore to limit parallel validations
            async with self._validation_semaphore:
                # Perform validation tasks in parallel
                validation_tasks = [
                    self._validate_schema(data, context),
                    self._validate_references(data, context),
                    self._validate_custom_rules(data, context)
                ]
                
                validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                
                # Process validation results
                for task_result in validation_results:
                    if isinstance(task_result, Exception):
                        if isinstance(task_result, LoreError):
                            result.add_error(task_result)
                        else:
                            result.add_error(LoreError(
                                message=str(task_result),
                                category=ErrorCategory.VALIDATION,
                                severity=ErrorSeverity.ERROR,
                                details={'task': task_result.__class__.__name__}
                            ))
                    elif isinstance(task_result, ValidationResult):
                        if not task_result.is_valid:
                            result.errors.extend(task_result.errors)
                        result.warnings.extend(task_result.warnings)
            
            # Update validation statistics
            validation_time = time.time() - start_time
            result.set_validation_time(validation_time)
            self._update_validation_stats(result.is_valid, validation_time)
            
            # Cache the result
            await self._resource_manager.set('validation', cache_key, result, context.cache_ttl)
            
        except Exception as e:
            error = LoreError(
                message=str(e),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'traceback': traceback.format_exc()}
            )
            await self._error_handler.handle_error(error)
            result.add_error(error)
        finally:
            self._active_validations.discard(data_id)
            VALIDATION_TIME.observe(time.time() - start_time)
        
        return result
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, data: Dict[str, Any], context: ValidationContext) -> str:
        """Generate a cache key for the validation result"""
        data_str = json.dumps(data, sort_keys=True)
        context_str = json.dumps({
            'schema_version': context.schema_version,
            'validation_mode': context.validation_mode
        }, sort_keys=True)
        return hashlib.sha256(f"{data_str}:{context_str}".encode()).hexdigest()
    
    async def _validate_schema(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Validate data against JSON schema with caching"""
        result = ValidationResult()
        
        try:
            schema = await self._get_schema(context.schema_version)
            validate(instance=data, schema=schema)
        except JSONSchemaError as e:
            result.add_error(LoreError(
                message=str(e),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'path': list(e.path), 'validator': e.validator}
            ))
        
        return result
    
    async def _validate_references(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Validate references in the data"""
        result = ValidationResult()
        
        try:
            # Implement reference validation logic here
            pass
        except Exception as e:
            result.add_error(LoreError(
                message=str(e),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'type': 'reference_validation'}
            ))
        
        return result
    
    async def _validate_custom_rules(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Validate data against custom rules"""
        result = ValidationResult()
        
        try:
            for rule_name, validator in context.custom_validators.items():
                try:
                    await validator(data)
                except Exception as e:
                    result.add_error(LoreError(
                        message=str(e),
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.ERROR,
                        details={'rule': rule_name}
                    ))
        except Exception as e:
            result.add_error(LoreError(
                message=str(e),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'type': 'custom_validation'}
            ))
        
        return result
    
    @lru_cache(maxsize=100)
    async def _get_schema(self, version: str) -> Dict[str, Any]:
        """Get schema for the specified version with caching"""
        if schema := await self._resource_manager.get('schema', version):
            return schema
        
        # Load schema if not in cache
        schema = await self._load_schema(version)
        if schema:
            await self._resource_manager.set('schema', version, schema)
        return schema
    
    async def _load_schema(self, version: str) -> Optional[Dict[str, Any]]:
        """Load a schema from storage"""
        try:
            # Implement schema loading logic here
            pass
        except Exception as e:
            error = LoreError(
                message=f"Failed to load schema version {version}: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                details={'version': version}
            )
            await self._error_handler.handle_error(error)
            return None
    
    def _get_data_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for the data"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def _update_validation_stats(self, success: bool, validation_time: float):
        """Update validation statistics"""
        if success:
            self._validation_stats['successful_validations'] += 1
            VALIDATION_COUNT.labels(status='success').inc()
        else:
            self._validation_stats['failed_validations'] += 1
            VALIDATION_COUNT.labels(status='failure').inc()
        
        # Update average validation time
        current_avg = self._validation_stats['average_validation_time']
        total_validations = self._validation_stats['total_validations']
        self._validation_stats['average_validation_time'] = (
            (current_avg * (total_validations - 1) + validation_time) / total_validations
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        return self._validation_stats.copy()
    
    async def _recover_validation_error(self, error: LoreError) -> bool:
        """Attempt to recover from a validation error"""
        try:
            # Implement recovery logic here
            # For example, try alternative validation strategies
            # or fall back to lenient validation mode
            return True
        except Exception as e:
            logger.error(
                "Recovery attempt failed",
                error=error.to_dict(),
                recovery_error=str(e)
            )
            return False
