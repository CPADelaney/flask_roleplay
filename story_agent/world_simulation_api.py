# story_agent/world_simulation_api.py

"""
API Routes for the World Director Agent - Open-world slice-of-life simulation.

This module provides API endpoints for the World Director Agent,
focusing on world state management, slice-of-life events, and emergent gameplay.
"""

import logging
import json
from datetime import datetime, timedelta
import time
from functools import wraps
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from quart import Blueprint, request, jsonify, session, current_app, g
from werkzeug.exceptions import TooManyRequests
import redis
from utils.performance import timed_function
from cachetools import TTLCache

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from story_agent.world_director_agent import (
        WorldDirector,
        WorldState,
        SliceOfLifeEvent,
        PowerExchange,
        WorldMood,
        TimeOfDay,
        ActivityType,
        PowerDynamicType
    )

world_simulation_bp = Blueprint("world_simulation_bp", __name__)
logger = logging.getLogger(__name__)

# Caches for world state
WORLD_STATE_CACHE = TTLCache(maxsize=1000, ttl=30)  # 30 second TTL for dynamic world
EVENT_CACHE = TTLCache(maxsize=500, ttl=60)  # 60 second TTL for events

# Lazy loader for WorldDirector
def _get_world_director():
    """Lazy load WorldDirector to avoid circular imports"""
    from story_agent.world_director_agent import WorldDirector
    return WorldDirector

# Rate limiting decorator (reused from original)
def rate_limit(limit=10, period=60):
    def decorator(f):
        @wraps(f)
        async def wrapped(*args, **kwargs):
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"error": "Not logged in"}), 401
            
            key = f"rate_limit:{user_id}:{request.endpoint}"
            
            try:
                r = redis.from_url(current_app.config.get('REDIS_URL', 'redis://localhost:6379/0'))
                current = r.incr(key)
                
                if current == 1:
                    r.expire(key, period)
                
                if current > limit:
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "retry_after": r.ttl(key)
                    }), 429
                
                return await f(*args, **kwargs)
            except redis.RedisError:
                logger.error("Rate limiting failed - Redis error")
                return await f(*args, **kwargs)
        return wrapped
    return decorator

# ----- World State Endpoints -----

@world_simulation_bp.route("/world/state", methods=["GET"])
@timed_function(name="get_world_state")
@rate_limit(limit=20, period=60)  # More frequent for real-time world
async def get_world_state_api():
    """Get the current state of the simulated world."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    # Check cache
    cache_key = f"world_state:{user_id}:{conversation_id}"
    if cache_key in WORLD_STATE_CACHE:
        cached = WORLD_STATE_CACHE[cache_key]
        cached["from_cache"] = True
        return jsonify(cached)
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        world_state = await director.get_world_state()
        
        # Handle different response types
        if hasattr(world_state, 'model_dump'):
            state_dict = world_state.model_dump()
        elif hasattr(world_state, 'dict'):
            state_dict = world_state.dict()
        else:
            state_dict = dict(world_state) if world_state else {}
        
        response = {
            "world_state": state_dict,
            "timestamp": datetime.now().isoformat(),
            "from_cache": False
        }
        
        WORLD_STATE_CACHE[cache_key] = response
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting world state: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@world_simulation_bp.route("/world/tick", methods=["POST"])
@timed_function(name="world_tick")
@rate_limit(limit=30, period=60)
async def simulate_world_tick_api():
    """Advance the world simulation by one tick."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        result = await director.simulate_world_tick()
        
        # Invalidate cache
        cache_key = f"world_state:{user_id}:{conversation_id}"
        WORLD_STATE_CACHE.pop(cache_key, None)
        
        return jsonify({
            "tick_result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error simulating world tick: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Event Generation Endpoints -----

@world_simulation_bp.route("/world/events/generate", methods=["POST"])
@timed_function(name="generate_event")
@rate_limit(limit=20, period=60)
async def generate_event_api():
    """Generate a slice-of-life event."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    event_type = data.get("event_type")  # Optional
    involved_npcs = data.get("involved_npcs", [])
    preferred_mood = data.get("preferred_mood")
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        await director.initialize()
        
        from agents import RunContextWrapper
        event = await director.context.generate_slice_of_life_event(
            RunContextWrapper(director.context),
            event_type=event_type,
            involved_npcs=involved_npcs,
            preferred_mood=preferred_mood
        )
        
        # Handle different response types
        if hasattr(event, 'model_dump'):
            event_dict = event.model_dump()
        elif hasattr(event, 'dict'):
            event_dict = event.dict()
        else:
            event_dict = dict(event) if event else {}
        
        return jsonify({
            "event": event_dict,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating event: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@world_simulation_bp.route("/world/events/active", methods=["GET"])
@rate_limit(limit=30, period=60)
async def get_active_events_api():
    """Get all currently active events."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        world_state = await director.get_world_state()
        
        # Handle different attribute names
        ongoing_events = []
        available_activities = []
        
        if hasattr(world_state, 'ongoing_events'):
            ongoing_events = world_state.ongoing_events
        elif hasattr(world_state, 'events'):
            ongoing_events = world_state.events
        
        if hasattr(world_state, 'available_activities'):
            available_activities = world_state.available_activities
        elif hasattr(world_state, 'activities'):
            available_activities = world_state.activities
        
        # Convert to dict format
        def to_dict(obj):
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            else:
                return dict(obj) if obj else {}
        
        return jsonify({
            "active_events": [to_dict(e) for e in ongoing_events],
            "available_activities": [to_dict(a) for a in available_activities],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting active events: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Power Dynamics Endpoints -----

@world_simulation_bp.route("/world/power/exchange", methods=["POST"])
@timed_function(name="power_exchange")
@rate_limit(limit=15, period=60)
async def trigger_power_exchange_api():
    """Trigger a power exchange moment."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_id = data.get("npc_id")
    exchange_type = data.get("exchange_type", "subtle_control")
    intensity = data.get("intensity", 0.5)
    is_public = data.get("is_public", False)
    
    if not conversation_id or not npc_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        await director.initialize()
        
        from agents import RunContextWrapper
        exchange = await director.context.trigger_power_exchange(
            RunContextWrapper(director.context),
            npc_id=int(npc_id),
            exchange_type=exchange_type,
            intensity=float(intensity),
            is_public=bool(is_public)
        )
        
        # Handle different response types
        if hasattr(exchange, 'model_dump'):
            exchange_dict = exchange.model_dump()
        elif hasattr(exchange, 'dict'):
            exchange_dict = exchange.dict()
        else:
            exchange_dict = dict(exchange) if exchange else {}
        
        return jsonify({
            "exchange": exchange_dict,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error triggering power exchange: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@world_simulation_bp.route("/world/power/recent", methods=["GET"])
@rate_limit(limit=30, period=60)
async def get_recent_power_exchanges_api():
    """Get recent power exchanges."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        world_state = await director.get_world_state()
        
        # Safely extract values with fallbacks
        recent_exchanges = []
        power_tension = 0.0
        submission_level = 0.0
        
        if hasattr(world_state, 'recent_power_exchanges'):
            recent_exchanges = world_state.recent_power_exchanges
        
        if hasattr(world_state, 'world_tension'):
            tension_obj = world_state.world_tension
            if hasattr(tension_obj, 'power_tension'):
                power_tension = tension_obj.power_tension
            elif isinstance(tension_obj, dict):
                power_tension = tension_obj.get('power_tension', 0.0)
        
        if hasattr(world_state, 'relationship_dynamics'):
            dynamics = world_state.relationship_dynamics
            if hasattr(dynamics, 'player_submission_level'):
                submission_level = dynamics.player_submission_level
            elif isinstance(dynamics, dict):
                submission_level = dynamics.get('player_submission_level', 0.0)
        
        return jsonify({
            "recent_exchanges": recent_exchanges,
            "power_tension": power_tension,
            "submission_level": submission_level,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting power exchanges: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- World Management Endpoints -----

@world_simulation_bp.route("/world/time/advance", methods=["POST"])
@timed_function(name="advance_time")
@rate_limit(limit=10, period=60)
async def advance_time_api():
    """Advance the world time period."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    skip_to = data.get("skip_to")  # Optional specific time
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        await director.initialize()
        
        from agents import RunContextWrapper
        result = await director.context.advance_time_period(
            RunContextWrapper(director.context),
            skip_to=skip_to
        )
        
        # Invalidate cache
        cache_key = f"world_state:{user_id}:{conversation_id}"
        WORLD_STATE_CACHE.pop(cache_key, None)
        
        return jsonify({
            "time_result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error advancing time: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@world_simulation_bp.route("/world/mood/adjust", methods=["POST"])
@rate_limit(limit=10, period=60)
async def adjust_world_mood_api():
    """Adjust the world mood."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    target_mood = data.get("target_mood", "relaxed")
    intensity = data.get("intensity", 0.5)
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        await director.initialize()
        
        from agents import RunContextWrapper
        result = await director.context.adjust_world_mood(
            RunContextWrapper(director.context),
            target_mood=target_mood,
            intensity=float(intensity)
        )
        
        return jsonify({
            "mood_result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error adjusting mood: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Player Action Endpoint -----

@world_simulation_bp.route("/world/action", methods=["POST"])
@timed_function(name="process_action")
@rate_limit(limit=30, period=60)
async def process_player_action_api():
    """Process a player action in the world."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    action = data.get("action")
    
    if not conversation_id or not action:
        return jsonify({"error": "Missing required parameters"}), 400
    
    if len(action) > 1000:
        return jsonify({"error": "Action text too long"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        result = await director.process_player_action(action)
        
        # Invalidate cache
        cache_key = f"world_state:{user_id}:{conversation_id}"
        WORLD_STATE_CACHE.pop(cache_key, None)
        
        return jsonify({
            "action_result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error processing action: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- NPC Autonomy Endpoint -----

@world_simulation_bp.route("/world/npc/simulate", methods=["POST"])
@rate_limit(limit=20, period=60)
async def simulate_npc_autonomy_api():
    """Simulate autonomous NPC behavior."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_id = data.get("npc_id")
    hours = data.get("hours_to_simulate", 1)
    
    if not conversation_id or not npc_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        WorldDirector = _get_world_director()
        director = WorldDirector(user_id, int(conversation_id))
        await director.initialize()
        
        from agents import RunContextWrapper
        result = await director.context.simulate_npc_autonomy(
            RunContextWrapper(director.context),
            npc_id=int(npc_id),
            hours_to_simulate=int(hours)
        )
        
        return jsonify({
            "npc_simulation": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error simulating NPC: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
