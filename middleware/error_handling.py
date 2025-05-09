# middleware/error_handling.py

"""
Error handling utilities for consistent error responses and logging.
"""

import time
import random
import logging
from quart import jsonify

# Configure logging
logger = logging.getLogger(__name__)

async def generate_error_id():
    """
    Generate a unique error ID for tracking errors.
    
    Returns:
        str: A unique error ID in format 'err_timestamp_random'
    """
    timestamp = int(time.time())
    random_suffix = random.randint(1000, 9999)
    return f"err_{timestamp}_{random_suffix}"

async def create_error_response(error, message=None, status_code=500):
    """
    Create a standardized error response.
    
    Args:
        error: The error object or message
        message: Optional user-friendly message (defaults to a generic message)
        status_code: HTTP status code (defaults to 500)
        
    Returns:
        tuple: A Flask response tuple (jsonify(data), status_code)
    """
    error_id = await generate_error_id()
    error_message = str(error)
    user_message = message or "An error occurred. Please try again later."
    
    # Log the error with error_id for tracking
    logger.error(f"Error ID {error_id}: {error_message}", exc_info=True)
    
    return jsonify({
        "success": False,
        "error": error_message,
        "error_id": error_id,
        "message": user_message,
        "status_code": status_code
    }), status_code

async def handle_validation_errors(errors):
    """
    Handle validation errors and create a standardized response.
    
    Args:
        errors: Validation error dictionary
        
    Returns:
        tuple: A Flask response tuple
    """
    return jsonify({
        "success": False,
        "error": "Validation failed",
        "validation_errors": errors,
        "message": "Please check your input and try again.",
        "status_code": 400
    }), 400
