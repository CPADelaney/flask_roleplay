# story_agent/conflict_narrative_api.py

"""
API Routes for the World Director Agent with improved production readiness.

This module defines the Flask routes for integrating the open-world World
Director with the game API, including proper error handling, rate limiting,
and caching.
"""

import logging
import json
from datetime import datetime, timedelta
import time
from functools import wraps
from quart import Blueprint, request, jsonify, session, current_app, g
from werkzeug.exceptions import TooManyRequests, InternalServerError
import redis
from utils.performance import timed_function
from cachetools import TTLCache, cached

from agents.exceptions import AgentsException, ModelBehaviorError

# Updated to use CompleteWorldDirector for open-world narrative management
from story_agent.world_director_agent import CompleteWorldDirector

world_director_bp = Blueprint("world_director_bp", __name__)
logger = logging.getLogger(__name__)

# In-memory caches
# This is for simple use cases; consider using Redis for production
WORLD_STATE_CACHE = TTLCache(maxsize=1000, ttl=60)  # 60 second TTL
WORLD_METRICS_CACHE = TTLCache(maxsize=100, ttl=300)  # 5 minute TTL

# ----- Middleware & Decorators -----

def get_rate_limit_redis():
    """Get or create Redis connection for rate limiting"""
    if not hasattr(g, 'rate_limit_redis'):
        redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
        g.rate_limit_redis = redis.from_url(redis_url)
    return g.rate_limit_redis

def rate_limit(limit=10, period=60):
    """
    Rate limiting decorator.
    
    Args:
        limit: Maximum number of requests per period
        period: Time period in seconds
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"error": "Not logged in"}), 401
            
            # Create a rate limit key specific to this user and endpoint
            key = f"rate_limit:{user_id}:{request.endpoint}"
            
            # Get Redis connection
            try:
                r = get_rate_limit_redis()
                
                # Increment the counter
                current = r.incr(key)
                
                # Set expiry on first request
                if current == 1:
                    r.expire(key, period)
                
                if current > limit:
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "retry_after": r.ttl(key)
                    }), 429
                
                # Add rate limit headers
                response = f(*args, **kwargs)
                
                # Check if response is a tuple (response, status_code)
                if isinstance(response, tuple):
                    resp, code = response
                    resp.headers["X-RateLimit-Limit"] = str(limit)
                    resp.headers["X-RateLimit-Remaining"] = str(limit - current)
                    resp.headers["X-RateLimit-Reset"] = str(r.ttl(key))
                    return resp, code
                else:
                    response.headers["X-RateLimit-Limit"] = str(limit)
                    response.headers["X-RateLimit-Remaining"] = str(limit - current)
                    response.headers["X-RateLimit-Reset"] = str(r.ttl(key))
                    return response
                
            except redis.RedisError:
                # If Redis fails, log but don't block the request
                logger.error("Rate limiting failed - Redis error")
                return f(*args, **kwargs)
                
        return wrapped
    return decorator

def error_handler(f):
    """Error handling decorator for story agent endpoints"""
    @wraps(f)
    async def wrapped(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except AgentsException as e:
            logger.error(f"Agent error: {str(e)}")
            return jsonify({
                "error": "Agent error",
                "message": str(e),
                "type": type(e).__name__
            }), 500
        except ModelBehaviorError as e:
            logger.error(f"Model behavior error: {str(e)}")
            return jsonify({
                "error": "Model behavior error",
                "message": str(e),
                "type": "ModelBehaviorError"
            }), 500
        except TooManyRequests as e:
            return jsonify({
                "error": "Rate limit exceeded",
                "message": str(e)
            }), 429
        except Exception as e:
            logger.exception("Unhandled error in story director endpoint")
            return jsonify({
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }), 500
    return wrapped

# ----- API Routes -----

@world_director_bp.route("/world/state", methods=["GET"])
@timed_function(name="get_world_state")
@rate_limit(limit=10, period=60)  # 10 requests per minute
@error_handler
async def get_world_state_api():
    """
    Get the current state of the world, including narrative tensions
    and potential events.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    # Check cache first
    cache_key = f"world_state:{user_id}:{conversation_id}"
    cached_state = WORLD_STATE_CACHE.get(cache_key)
    if cached_state:
        logger.info(f"Returning cached story state for user {user_id}, conversation {conversation_id}")
        cached_state["cache_hit"] = True
        return jsonify(cached_state)
    
    # Initialize the World Director Agent
    try:
        director = CompleteWorldDirector(user_id, int(conversation_id))
        await director.initialize()
    except Exception as e:
        logger.error(f"Error initializing world director: {str(e)}")
        return jsonify({"error": f"Failed to initialize world director: {str(e)}"}), 500

    # Get current world state
    try:
        start_time = time.time()
        result = await director.generate_next_moment()
        execution_time = time.time() - start_time

        state_data = result.get("world_state", {})
        response = {
            "world_state": state_data,
            "moment": result.get("moment"),
            "patterns": result.get("patterns", {}),
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }

        # Cache the result
        WORLD_STATE_CACHE[cache_key] = response

        return jsonify(response)

    except Exception as e:
        logger.exception(f"Error getting world state: {str(e)}")
        raise

@world_director_bp.route("/world/process", methods=["POST"])
@timed_function(name="process_world_input")
@rate_limit(limit=20, period=60)  # 20 requests per minute
@error_handler
async def process_world_input_api():
    """
    Process narrative text to determine if it should trigger conflicts or narrative events.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    narrative_text = data.get("narrative_text")
    if not narrative_text:
        return jsonify({"error": "Missing narrative_text parameter"}), 400
    
    # Check if text exceeds reasonable limits
    if len(narrative_text) > 4000:
        return jsonify({"error": "Narrative text exceeds maximum length of 4000 characters"}), 400
    
    # Initialize the World Director Agent
    try:
        director = CompleteWorldDirector(user_id, int(conversation_id))
        await director.initialize()
    except Exception as e:
        logger.error(f"Error initializing world director: {str(e)}")
        return jsonify({"error": f"Failed to initialize world director: {str(e)}"}), 500

    # Process player action
    try:
        start_time = time.time()
        result = await director.process_player_action(narrative_text)
        execution_time = time.time() - start_time

        response_text = result.get("response") if isinstance(result, dict) else None
        conflict_data = result.get("patterns", {}) if isinstance(result, dict) else {}

        # Invalidate world state cache
        cache_key = f"world_state:{user_id}:{conversation_id}"
        if cache_key in WORLD_STATE_CACHE:
            del WORLD_STATE_CACHE[cache_key]

        return jsonify({
            "analysis": response_text,
            "conflict_generated": bool(conflict_data),
            "conflict_data": conflict_data,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error processing narrative: {str(e)}")
        raise

@world_director_bp.route("/world/advance", methods=["POST"])
@timed_function(name="advance_world")
@rate_limit(limit=15, period=60)  # 15 requests per minute
@error_handler
async def advance_world_api():
    """
    Advance the world based on player actions.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    player_actions = data.get("player_actions")
    if not player_actions:
        return jsonify({"error": "Missing player_actions parameter"}), 400
    
    # Check if text exceeds reasonable limits
    if len(player_actions) > 4000:
        return jsonify({"error": "Player actions text exceeds maximum length of 4000 characters"}), 400
    
    # Initialize the World Director Agent
    try:
        director = CompleteWorldDirector(user_id, int(conversation_id))
        await director.initialize()
    except Exception as e:
        logger.error(f"Error initializing world director: {str(e)}")
        return jsonify({"error": f"Failed to initialize world director: {str(e)}"}), 500

    # Advance the world
    try:
        start_time = time.time()
        result = await director.process_player_action(player_actions)
        execution_time = time.time() - start_time

        response_text = result.get("response") if isinstance(result, dict) else None

        return jsonify({
            "world_advancement": response_text,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error advancing story: {str(e)}")
        raise

@world_director_bp.route("/world/metrics", methods=["GET"])
@rate_limit(limit=10, period=60)
@error_handler
async def get_world_metrics_api():
    """
    Get metrics for the World Director agent.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    # Check cache first
    cache_key = f"metrics:{user_id}:{conversation_id}"
    cached_metrics = WORLD_METRICS_CACHE.get(cache_key)
    if cached_metrics:
        cached_metrics["cache_hit"] = True
        return jsonify(cached_metrics)
    
    # Get fresh metrics
    try:
        director = CompleteWorldDirector(user_id, int(conversation_id))
        await director.initialize()

        metrics = {}
        if director.context and getattr(director.context, 'performance_monitor', None):
            metrics = director.context.performance_monitor.get_metrics()
        
        response = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }
        
        # Cache the metrics
        WORLD_METRICS_CACHE[cache_key] = response
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Error getting metrics: {str(e)}")
        raise

@world_director_bp.route("/world/reset", methods=["POST"])
@rate_limit(limit=5, period=300)  # 5 requests per 5 minutes
@error_handler
async def reset_world_director_api():
    """
    Reset the World Director's state.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400

    try:
        director = CompleteWorldDirector(user_id, int(conversation_id))
        await director.initialize()

        # Clear caches for this conversation
        cache_key = f"world_state:{user_id}:{conversation_id}"
        if cache_key in WORLD_STATE_CACHE:
            del WORLD_STATE_CACHE[cache_key]

        metrics_key = f"metrics:{user_id}:{conversation_id}"
        if metrics_key in WORLD_METRICS_CACHE:
            del WORLD_METRICS_CACHE[metrics_key]

        return jsonify({
            "status": "success",
            "message": "World director reset successfully",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.exception(f"Error resetting world director: {str(e)}")
        raise
