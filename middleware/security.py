"""
Security middleware for input validation and protection.
"""

import re
import json
import logging
import bleach
from functools import wraps
from flask import request, abort, current_app
from marshmallow import ValidationError
from typing import Dict, Any, Callable, Optional

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
    
    def sanitize_html(self, content: str) -> str:
        """Sanitize HTML content."""
        return bleach.clean(
            content,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
    
    def validate_pattern(self, value: str, pattern_name: str) -> bool:
        """Validate string against predefined pattern."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        return bool(re.match(self.patterns[pattern_name], value))
    
    def validate_json_structure(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
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
        def decorated_function(*args, **kwargs):
            security = SecurityMiddleware()
            
            # Validate URL parameters
            for param_name, pattern_name in (patterns or {}).items():
                if param_name in kwargs:
                    if not security.validate_pattern(kwargs[param_name], pattern_name):
                        logger.warning(f"Invalid {param_name} format: {kwargs[param_name]}")
                        abort(400, f"Invalid {param_name} format")
            
            # Validate JSON body
            if schema and request.is_json:
                data = request.get_json()
                if not security.validate_json_structure(data, schema):
                    abort(400, "Invalid request body")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def sanitize_output(f: Callable):
    """Decorator for output sanitization."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        security = SecurityMiddleware()
        response = f(*args, **kwargs)
        
        # If response is JSON
        if isinstance(response, dict):
            def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
                for key, value in d.items():
                    if isinstance(value, str):
                        d[key] = security.sanitize_html(value)
                    elif isinstance(value, dict):
                        d[key] = sanitize_dict(value)
                    elif isinstance(value, list):
                        d[key] = [
                            security.sanitize_html(item) if isinstance(item, str)
                            else sanitize_dict(item) if isinstance(item, dict)
                            else item
                            for item in value
                        ]
                return d
            
            return sanitize_dict(response)
        
        return response
    return decorated_function

# Example usage:
"""
@validate_input(
    schema={
        'name': {'type': 'string', 'required': True},
        'personality_traits': {'type': 'list', 'schema': {'type': 'string'}},
        'stats': {
            'type': 'dict',
            'schema': {
                'intensity': {'type': 'integer', 'min': 0, 'max': 100},
                'corruption': {'type': 'integer', 'min': 0, 'max': 100}
            }
        }
    },
    patterns={'npc_id': 'npc_id'}
)
@sanitize_output
def create_npc():
    # Your route logic here
    pass
""" 
