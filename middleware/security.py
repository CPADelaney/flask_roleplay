# middleware/security.py

"""
Security middleware for input validation and protection.
"""

import re
import json
import logging
import bleach
from functools import wraps
from quart import request, abort, current_app
from marshmallow import ValidationError
from typing import Dict, Any, Callable, Optional
from middleware.validation import sanitize_json as custom_sanitize_json

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Security middleware for request validation and sanitization."""
    
    def __init__(self):
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ul', 'ol', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]
        self.allowed_attributes = {
            '*': ['class', 'id', 'style']
        }
        
        # Common regex patterns
        self.patterns = {
            'username': r'^[a-zA-Z0-9_-]{3,32}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'npc_id': r'^npc_[0-9a-f]{8}$'
        }
    
    async def sanitize_html(self, content: str) -> str:
        """Sanitize HTML content."""
        return bleach.clean(
            content,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
    
    async def validate_pattern(self, value: str, pattern_name: str) -> bool:
        """Validate string against predefined pattern."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        return bool(re.match(self.patterns[pattern_name], value))
    
    async def validate_json_structure(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate JSON structure against schema."""
        try:
            from marshmallow import Schema, fields
            schema_obj = Schema.from_dict(schema)()
            schema_obj.load(data)
            return True
        except ValidationError as e:
            logger.warning(f"JSON validation error: {e.messages}")
            return False

def validate_input(schema: Optional[Dict[str, Any]] = None, patterns: Optional[Dict[str, str]] = None):
    """Decorator for input validation."""
    def decorator(f: Callable):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            security = SecurityMiddleware() # Still useful for URL param validation or HTML sanitization
            
            # Validate URL parameters (using SecurityMiddleware's patterns)
            for param_name, pattern_name in (patterns or {}).items():
                if param_name in kwargs:
                    if not await security.validate_pattern(kwargs[param_name], pattern_name):
                        logger.warning(f"Invalid {param_name} format: {kwargs[param_name]}")
                        abort(400, f"Invalid {param_name} format")
            
            # Validate JSON body USING YOUR CUSTOM SANITIZER
            if schema and request.is_json:
                data_to_validate = {}
                try:
                    data_to_validate = await request.get_json()
                    if data_to_validate is None: data_to_validate = {}
                except Exception as json_err:
                    logger.warning(f"Invalid JSON body for validation in 'validate_input': {json_err}")
                    abort(400, "Invalid JSON request body")
                
                # Use your custom sanitize_json from middleware.validation.py
                # This function expects your custom schema format.
                sanitized, errors = await custom_sanitize_json(data_to_validate, schema) 
                
                if errors:
                    logger.warning(f"Custom JSON validation/sanitization error: {errors}")
                    # You might want a more specific error response creator here
                    return jsonify({
                        "error": "Validation failed",
                        "validation_errors": errors, # Errors from your custom_sanitize_json
                        "status_code": 400
                    }), 400
                
                # Add sanitized data to request object for the route to use
                request.sanitized_data = sanitized # Store the result from custom_sanitize_json
            
            return await f(*args, **kwargs)
        return decorated_function
    return decorator

def sanitize_output(f: Callable):
    """Decorator for output sanitization."""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        security = SecurityMiddleware()
        response = await f(*args, **kwargs)
        
        # If response is JSON
        if isinstance(response, dict):
            async def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
                for key, value in d.items():
                    if isinstance(value, str):
                        d[key] = await security.sanitize_html(value)
                    elif isinstance(value, dict):
                        d[key] = await sanitize_dict(value)
                    elif isinstance(value, list):
                        d[key] = [
                            await security.sanitize_html(item) if isinstance(item, str)
                            else await sanitize_dict(item) if isinstance(item, dict)
                            else item
                            for item in value
                        ]
                return d
            
            return await sanitize_dict(response)
        
        return response
    return decorated_function
