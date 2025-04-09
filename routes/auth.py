# routes/auth.py

"""
Authentication utilities for the application.
Provides login decorator and compatibility layer with main app auth.
"""

import logging
from functools import wraps
from quart import Blueprint, request, jsonify, session, redirect, url_for

# Create logger
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

#---------------------------
# Authentication Decorators
#---------------------------

def require_login(route_handler):
    """
    Decorator for routes that require authentication.
    If user is not logged in, returns a 401 Unauthorized response.
    Passes user_id to the route handler as first argument.
    """
    @wraps(route_handler)
    async def wrapper(*args, **kwargs):
        # Check if user is logged in
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        # If logged in, call the original route handler with user_id
        user_id = session.get('user_id')
        return await route_handler(user_id, *args, **kwargs)
    
    return wrapper

#---------------------------
# Blueprint Registration
#---------------------------

def register_auth_routes(app):
    """Register auth routes with the Quart app."""
    app.register_blueprint(auth_bp)
    logger.info("Auth utilities registered")
    
    # Note: The actual auth routes are defined in main.py
    # This function mainly registers the blueprint for utilities like require_login
