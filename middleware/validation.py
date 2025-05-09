"""
Input validation and sanitization utilities.
"""

import re
import html
import json
from functools import wraps
from quart import request, jsonify
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Common validation patterns
PATTERNS = {
    'username': re.compile(r'^[a-zA-Z0-9_-]{3,30}$'),
    'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    'no_html': re.compile(r'<[^>]*>'),
    'no_script': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
    'integer': re.compile(r'^-?\d+$'),
    'float': re.compile(r'^-?\d+(\.\d+)?$'),
}

async def sanitize_string(text, max_length=None, allowed_tags=None):
    """
    Sanitize a string input.
    
    Args:
        text: The string to sanitize
        max_length: Optional maximum length
        allowed_tags: Optional list of allowed HTML tags
        
    Returns:
        Sanitized string
    """
    if text is None:
        return None
    
    # Convert to string if needed
    if not isinstance(text, str):
        text = str(text)
    
    # Trim to max length if specified
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    if allowed_tags:
        # If specific tags are allowed, we use a more complex logic
        # This is a simplified approach; for production use, consider a library like bleach
        for tag in allowed_tags:
            # Create a pattern that matches this specific tag
            tag_pattern = re.compile(f'</?{tag}( [^>]*)?>', re.IGNORECASE)
            # Temporarily replace allowed tags with a marker
            text = tag_pattern.sub(lambda m: f'__ALLOWED_TAG_{hash(m.group(0))}__', text)
        
        # Escape all remaining HTML
        text = html.escape(text)
        
        # Restore allowed tags
        for tag in allowed_tags:
            tag_pattern = re.compile(f'__ALLOWED_TAG_-?\d+__')
            # Replace markers with original tags
            text = tag_pattern.sub(lambda m: m.group(0).replace('__ALLOWED_TAG_', '<').replace('__', '>'), text)
    else:
        # No tags allowed, escape all HTML
        text = html.escape(text)
    
    return text

async def is_valid_pattern(text, pattern_name):
    """
    Check if text matches a named pattern.
    
    Args:
        text: The text to validate
        pattern_name: The name of the pattern to use
        
    Returns:
        bool: Whether the text matches the pattern
    """
    if text is None:
        return False
    
    pattern = PATTERNS.get(pattern_name)
    if not pattern:
        logger.warning(f"Unknown validation pattern: {pattern_name}")
        return False
    
    return bool(pattern.match(str(text)))

async def sanitize_json(json_data, schema):
    """
    Sanitize a JSON object according to a schema.
    
    Args:
        json_data: The JSON data to sanitize
        schema: A dictionary mapping field names to validation rules
               Example: {'username': {'type': 'string', 'pattern': 'username', 'max_length': 30}}
        
    Returns:
        tuple: (sanitized_data, errors)
    """
    if not json_data:
        return {}, {"general": "No data provided"}
    
    sanitized = {}
    errors = {}
    
    for field, rules in schema.items():
        # Check if field is required
        is_required = rules.get('required', False)
        if is_required and field not in json_data:
            errors[field] = "This field is required"
            continue
        
        # Skip if field is not present and not required
        if field not in json_data:
            continue
        
        value = json_data[field]
        field_type = rules.get('type', 'string')
        
        # Special handling for IDs that can be either string or integer
        if field_type == 'id':
            # Convert to string for consistent handling in the application
            try:
                if value is None:
                    if is_required:
                        errors[field] = "This field is required"
                    continue
                sanitized[field] = str(value)
            except (ValueError, TypeError):
                errors[field] = "Must be a valid ID"
            continue
        
        # Type validation
        if field_type == 'string':
            if not isinstance(value, str):
                errors[field] = "Must be a string"
                continue
            
            # Apply pattern validation if specified
            if 'pattern' in rules and not await is_valid_pattern(value, rules['pattern']):
                errors[field] = f"Invalid format for {field}"
                continue
            
            # Apply sanitization
            max_length = rules.get('max_length')
            allowed_tags = rules.get('allowed_tags')
            sanitized[field] = await sanitize_string(value, max_length, allowed_tags)
            
        elif field_type == 'integer':
            try:
                sanitized[field] = int(value)
            except (ValueError, TypeError):
                errors[field] = "Must be an integer"
                continue
                
            # Range validation
            if 'min' in rules and sanitized[field] < rules['min']:
                errors[field] = f"Must be at least {rules['min']}"
            if 'max' in rules and sanitized[field] > rules['max']:
                errors[field] = f"Must be at most {rules['max']}"
                
        elif field_type == 'float':
            try:
                sanitized[field] = float(value)
            except (ValueError, TypeError):
                errors[field] = "Must be a number"
                continue
                
            # Range validation
            if 'min' in rules and sanitized[field] < rules['min']:
                errors[field] = f"Must be at least {rules['min']}"
            if 'max' in rules and sanitized[field] > rules['max']:
                errors[field] = f"Must be at most {rules['max']}"
                
        elif field_type == 'boolean':
            if isinstance(value, bool):
                sanitized[field] = value
            else:
                # Try to convert to boolean
                if value in (1, '1', 'true', 'True', 'yes', 'Yes'):
                    sanitized[field] = True
                elif value in (0, '0', 'false', 'False', 'no', 'No'):
                    sanitized[field] = False
                else:
                    errors[field] = "Must be a boolean value"
                    
        elif field_type == 'array':
            if not isinstance(value, list):
                errors[field] = "Must be an array"
                continue
                
            # Length validation
            if 'min_length' in rules and len(value) < rules['min_length']:
                errors[field] = f"Must have at least {rules['min_length']} items"
                continue
            if 'max_length' in rules and len(value) > rules['max_length']:
                errors[field] = f"Must have at most {rules['max_length']} items"
                continue
                
            # Item validation if specified
            if 'items' in rules:
                sanitized_items = []
                item_errors = []
                
                for i, item in enumerate(value):
                    if isinstance(rules['items'], dict):
                        # Recursive sanitization for objects
                        item_sanitized, item_error = await sanitize_json({0: item}, {0: rules['items']})
                        if item_error:
                            item_errors.append((i, item_error))
                        else:
                            sanitized_items.append(item_sanitized[0])
                    else:
                        # Simple type validation
                        item_type = rules['items']
                        if item_type == 'string' and isinstance(item, str):
                            sanitized_items.append(await sanitize_string(item, rules.get('item_max_length')))
                        elif item_type == 'integer' and isinstance(item, int):
                            sanitized_items.append(item)
                        elif item_type == 'float' and isinstance(item, (int, float)):
                            sanitized_items.append(float(item))
                        else:
                            item_errors.append((i, f"Item must be of type {item_type}"))
                
                if item_errors:
                    errors[field] = {"items": item_errors}
                sanitized[field] = sanitized_items
            else:
                # No item validation, just pass through
                sanitized[field] = value
        
        elif field_type == 'object':
            if not isinstance(value, dict):
                errors[field] = "Must be an object"
                continue
                
            # Recursive sanitization for nested objects
            if 'properties' in rules:
                obj_sanitized, obj_errors = await sanitize_json(value, rules['properties'])
                if obj_errors:
                    errors[field] = obj_errors
                sanitized[field] = obj_sanitized
            else:
                # No property validation, just pass through
                sanitized[field] = value
    
    return sanitized, errors if errors else None
    
def validate_request(schema):
    """
    Decorator to validate and sanitize request data.
    
    Args:
        schema: Schema definition for validation
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            data_to_validate = {} # Initialize with a default

            # Get request data based on content type
            if request.is_json:
                try:
                    # AWAIT the coroutine to get the actual JSON data
                    data_to_validate = await request.get_json()
                    if data_to_validate is None: # Handle empty JSON body properly
                        data_to_validate = {}
                except Exception as e: # Catch errors if body is not valid JSON
                    logger.warning(f"Request to {request.path} with Content-Type application/json had invalid JSON body: {e}")
                    return jsonify({"error": "Invalid JSON in request body", "status_code": 400}), 400
            elif request.form:
                data_to_validate = request.form.to_dict()
            else: # Fallback or for GET requests if schema is used
                data_to_validate = request.args.to_dict()
            
            # Sanitize and validate
            # Now 'data_to_validate' will be an actual dictionary (or whatever request.form/args return)
            sanitized, errors = await sanitize_json(data_to_validate, schema) 
            
            if errors:
                return jsonify({
                    "error": "Validation failed",
                    "validation_errors": errors,
                    "status_code": 400
                }), 400
            
            # Add sanitized data to request object for the route to use
            request.sanitized_data = sanitized
            
            return await f(*args, **kwargs)
        return decorated_function
    return decorator
