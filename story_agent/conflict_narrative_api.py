# story_agent/conflict_narrative_api.py

"""
API Routes for the Story Director Agent with improved production readiness.

This module defines the Flask routes for integrating the Story Director Agent
with your existing game API, including proper error handling, rate limiting,
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

from story_director_agent import (
    initialize_story_director,
    get_current_story_state,
    process_narrative_input,
    advance_story,
    get_story_director_metrics,
    reset_story_director
)

story_director_bp = Blueprint("story_director_bp", __name__)
logger = logging.getLogger(__name__)

# In-memory caches
# This is for simple use cases; consider using Redis for production
STORY_STATE_CACHE = TTLCache(maxsize=1000, ttl=60)  # 60 second TTL
METRICS_CACHE = TTLCache(maxsize=100, ttl=300)  # 5 minute TTL

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

@story_director_bp.route("/story/state", methods=["GET"])
@timed_function(name="get_story_state")
@rate_limit(limit=10, period=60)  # 10 requests per minute
@error_handler
async def get_story_state_api():
    """
    Get the current state of the story, including narrative stage, 
    active conflicts, and potential narrative events.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    # Check cache first
    cache_key = f"story_state:{user_id}:{conversation_id}"
    cached_state = STORY_STATE_CACHE.get(cache_key)
    if cached_state:
        logger.info(f"Returning cached story state for user {user_id}, conversation {conversation_id}")
        cached_state["cache_hit"] = True
        return jsonify(cached_state)
    
    # Initialize the Story Director Agent
    try:
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
    except Exception as e:
        logger.error(f"Error initializing story director: {str(e)}")
        return jsonify({"error": f"Failed to initialize story director: {str(e)}"}), 500
    
    # Get current story state
    try:
        start_time = time.time()
        result = await get_current_story_state(agent, context)
        execution_time = time.time() - start_time
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract JSON if present
        try:
            # Look for JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                state_data = json.loads(json_str)
            else:
                state_data = {"state_description": response_text}
        except json.JSONDecodeError:
            # If JSON extraction fails, use the full text
            state_data = {"state_description": response_text}
        
        response = {
            "story_state": state_data,
            "narrative_analysis": response_text,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }
        
        # Cache the result
        STORY_STATE_CACHE[cache_key] = response
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Error getting story state: {str(e)}")
        raise

@story_director_bp.route("/story/process", methods=["POST"])
@timed_function(name="process_narrative")
@rate_limit(limit=20, period=60)  # 20 requests per minute
@error_handler
async def process_narrative_api():
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
    
    # Initialize the Story Director Agent
    try:
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
    except Exception as e:
        logger.error(f"Error initializing story director: {str(e)}")
        return jsonify({"error": f"Failed to initialize story director: {str(e)}"}), 500
    
    # Process narrative input
    try:
        start_time = time.time()
        result = await process_narrative_input(agent, context, narrative_text)
        execution_time = time.time() - start_time
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract any structured data about generated conflicts
        conflict_data = {}
        try:
            # Look for conflict/event info in the response
            if "conflict generated" in response_text.lower():
                # Try to find any JSON data
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    conflict_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON extraction fails, continue with empty data
            pass
        
        # Invalidate story state cache
        cache_key = f"story_state:{user_id}:{conversation_id}"
        if cache_key in STORY_STATE_CACHE:
            del STORY_STATE_CACHE[cache_key]
        
        return jsonify({
            "analysis": response_text,
            "conflict_generated": "conflict generated" in response_text.lower(),
            "narrative_event_generated": "narrative event" in response_text.lower(),
            "conflict_data": conflict_data,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error processing narrative: {str(e)}")
        raise

@story_director_bp.route("/story/advance", methods=["POST"])
@timed_function(name="advance_story")
@rate_limit(limit=15, period=60)  # 15 requests per minute
@error_handler
async def advance_story_api():
    """
    Advance the story based on player actions.
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
    
    # Initialize the Story Director Agent
    try:
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
    except Exception as e:
        logger.error(f"Error initializing story director: {str(e)}")
        return jsonify({"error": f"Failed to initialize story director: {str(e)}"}), 500
    
    # Advance the story
    try:
        start_time = time.time()
        result = await advance_story(agent, context, player_actions)
        execution_time = time.time() - start_time
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract structured data about story advancement
        advancement_data = {}
        try:
            # Look for JSON data in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                advancement_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON extraction fails, continue with empty data
            pass
        
        # Determine what occurred during story advancement
        conflicts_advanced = "conflict" in response_text.lower() and (
            "progress" in response_text.lower() or 
            "advance" in response_text.lower() or
            "update" in response_text.lower()
        )
        
        conflicts_resolved = "conflict" in response_text.lower() and (
            "resolve" in response_text.lower() or
            "resolution" in response_text.lower() or
            "concluded" in response_text.lower()
        )
        
        narrative_events = (
            "revelation" in response_text.lower() or
            "narrative moment" in response_text.lower() or
            "dream sequence" in response_text.lower() or
            "moment of clarity" in response_text.lower()
        )
        
        # Invalidate story state cache
        cache_key = f"story_state:{user_id}:{conversation_id}"
        if cache_key in STORY_STATE_CACHE:
            del STORY_STATE_CACHE[cache_key]
        
        return jsonify({
            "story_advancement": response_text,
            "conflicts_advanced": conflicts_advanced,
            "conflicts_resolved": conflicts_resolved,
            "narrative_events": narrative_events,
            "advancement_data": advancement_data,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error advancing story: {str(e)}")
        raise

@story_director_bp.route("/story/metrics", methods=["GET"])
@rate_limit(limit=10, period=60)
@error_handler
async def get_metrics_api():
    """
    Get metrics for the Story Director agent.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    # Check cache first
    cache_key = f"metrics:{user_id}:{conversation_id}"
    cached_metrics = METRICS_CACHE.get(cache_key)
    if cached_metrics:
        cached_metrics["cache_hit"] = True
        return jsonify(cached_metrics)
    
    # Get fresh metrics
    try:
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        metrics = get_story_director_metrics(context)
        
        response = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }
        
        # Cache the metrics
        METRICS_CACHE[cache_key] = response
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Error getting metrics: {str(e)}")
        raise

@story_director_bp.route("/story/reset", methods=["POST"])
@rate_limit(limit=5, period=300)  # 5 requests per 5 minutes
@error_handler
async def reset_story_director_api():
    """
    Reset the Story Director's state.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        await reset_story_director(context)
        
        # Clear caches for this conversation
        cache_key = f"story_state:{user_id}:{conversation_id}"
        if cache_key in STORY_STATE_CACHE:
            del STORY_STATE_CACHE[cache_key]
            
        metrics_key = f"metrics:{user_id}:{conversation_id}"
        if metrics_key in METRICS_CACHE:
            del METRICS_CACHE[metrics_key]
        
        return jsonify({
            "status": "success",
            "message": "Story director reset successfully",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.exception(f"Error resetting story director: {str(e)}")
        raise 
