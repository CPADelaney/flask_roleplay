# routes/auth.py

"""
Authentication routes and utilities for the application.
Provides login, registration, and session management functionality.
"""

import logging
from functools import wraps
from quart import Blueprint, request, jsonify, session, redirect, url_for
import hashlib
import os
import asyncio
from typing import Dict, Any, Callable, Awaitable

# Import database connection
from db.connection import get_db_connection_context

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
# Authentication Routes
#---------------------------

@auth_bp.route("/register", methods=["POST"])
async def register():
    """Register a new user."""
    data = await request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    # Hash the password with a random salt
    salt = os.urandom(32)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt, 
        100000
    )
    # Store salt and password hash together
    stored_password = salt + password_hash
    
    try:
        async with get_db_connection_context() as conn:
            # Check if username already exists
            existing_user = await conn.fetchrow(
                "SELECT id FROM users WHERE username = $1", 
                username
            )
            
            if existing_user:
                return jsonify({"error": "Username already exists"}), 409
                
            # Insert new user
            user_id = await conn.fetchval(
                "INSERT INTO users (username, password_hash) VALUES ($1, $2) RETURNING id",
                username, 
                stored_password.hex()  # Convert bytes to hex string for storage
            )
            
            # Set session
            session['user_id'] = user_id
            session['username'] = username
            
            return jsonify({"user_id": user_id, "message": "Registration successful"}), 201
            
    except Exception as e:
        logger.error(f"Error registering user: {e}", exc_info=True)
        return jsonify({"error": "Failed to register user"}), 500

@auth_bp.route("/login", methods=["POST"])
async def login():
    """Login a user."""
    data = await request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
        
    try:
        async with get_db_connection_context() as conn:
            # Get user with matching username
            user_row = await conn.fetchrow(
                "SELECT id, password_hash FROM users WHERE username = $1", 
                username
            )
            
            if not user_row:
                return jsonify({"error": "Invalid username or password"}), 401
                
            user_id = user_row['id']
            stored_password_hex = user_row['password_hash']
            
            # Convert hex string back to bytes
            stored_password = bytes.fromhex(stored_password_hex)
            
            # Extract salt (first 32 bytes) and hash
            salt = stored_password[:32]
            stored_hash = stored_password[32:]
            
            # Hash the provided password with the same salt
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', 
                password.encode('utf-8'), 
                salt, 
                100000
            )
            
            # Compare hashes
            if password_hash != stored_hash:
                return jsonify({"error": "Invalid username or password"}), 401
                
            # Set session
            session['user_id'] = user_id
            session['username'] = username
            
            return jsonify({"user_id": user_id, "message": "Login successful"}), 200
            
    except Exception as e:
        logger.error(f"Error logging in user: {e}", exc_info=True)
        return jsonify({"error": "Failed to login"}), 500

@auth_bp.route("/logout", methods=["POST"])
async def logout():
    """Logout a user."""
    # Clear session
    session.clear()
    return jsonify({"message": "Logout successful"}), 200

@auth_bp.route("/whoami", methods=["GET"])
async def whoami():
    """Return current user info."""
    if 'user_id' not in session:
        return jsonify({"logged_in": False}), 200
        
    return jsonify({
        "logged_in": True,
        "user_id": session.get('user_id'),
        "username": session.get('username')
    }), 200

#---------------------------
# Blueprint Registration
#---------------------------

def register_auth_routes(app):
    """Register auth routes with the Quart app."""
    app.register_blueprint(auth_bp)
    logger.info("Auth routes registered")
    
    # Set up session secret key
    app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    logger.info("Session configuration complete")
