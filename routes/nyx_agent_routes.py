# routes/nyx_agent_routes.py

import logging
import json
import time
import asyncio
from flask import Blueprint, request, jsonify, session

from nyx.nyx_agent import NyxAgent
from db.connection import get_db_connection
from logic.nyx_enhancements_integration import update_narrative_arcs_for_interaction
from logic.time_cycle import advance_time_with_events, should_advance_time
from logic.social_links import check_for_relationship_crossroads, check_for_relationship_ritual
from logic.addiction_system import process_addiction_effects, get_addiction_status
from logic.rule_enforcement import enforce_all_rules_on_player
from utils.performance import PerformanceTracker, timed_function
from utils.caching import NPC_CACHE, MEMORY_CACHE

nyx_agent_bp = Blueprint("nyx_agent_bp", __name__)

@nyx_agent_bp.route("/nyx_response", methods=["POST"])
@timed_function(name="nyx_response")
async def nyx_response():
    """
    Enhanced endpoint that processes user input through Nyx agent
    and returns a complete response.
    """
    tracker = PerformanceTracker("nyx_response")
    tracker.start_phase("initialization")
    
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conversation_id = data.get("conversation_id")
        
        # Validate required params
        if not user_input or not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize Nyx agent
        agent = NyxAgent(user_id, conversation_id)
        tracker.end_phase()
        
        # Process context information
        tracker.start_phase("context_building")
        context = {
            "location": data.get("location", "Unknown"),
            "time_of_day": data.get("time_of_day", "Morning"),
            "npc_present": data.get("npc_present", []),
            "player_name": data.get("player_name", "Chase"),
            "timestamp": time.time()
        }
        
        # Add any additional context from request
        if data.get("additional_context"):
            context.update(data["additional_context"])
        tracker.end_phase()
        
        # Process input through agent
        tracker.start_phase("agent_processing")
        system_directives = data.get("system_directives", {})
        response = await agent.process_input(user_input, context, system_directives)
        tracker.end_phase()
        
        # Post-processing steps
        tracker.start_phase("post_processing")
        
        # 1. Check for time advancement
        activity_type = data.get("activity_type", "conversation")
        time_result = await process_time_advancement(user_id, conversation_id, activity_type, data)
        
        # 2. Check for relationship events
        relationship_events = await process_relationship_events(user_id, conversation_id, data)
        
        # 3. Process rule enforcement
        rule_results = process_rule_enforcement(user_id, conversation_id)
        
        # 4. Check addiction system
        addiction_status = await get_addiction_status(user_id, conversation_id, context["player_name"])
        addiction_effects = None
        if addiction_status and addiction_status.get("has_addictions"):
            addiction_effects = await process_addiction_effects(
                user_id, conversation_id, context["player_name"], addiction_status
            )
        
        # 5. Update narrative arcs
        async with (await get_db_connection()) as conn:
            await update_narrative_arcs_for_interaction(
                user_id, conversation_id, user_input, response["text"], conn
            )
        tracker.end_phase()
        
        # Store user and Nyx messages in database
        tracker.start_phase("storage")
        await store_messages(user_id, conversation_id, user_input, response["text"])
        tracker.end_phase()
        
        # Build complete response
        tracker.start_phase("response_building")
        api_response = {
            "message": response["text"],
            "time_result": time_result,
            "confirm_needed": time_result.get("would_advance", False) and not data.get("confirm_time_advance", False),
            "relationship_events": relationship_events,
            "rule_effects": rule_results,
            "performance_metrics": tracker.get_metrics(),
            "processing_time": response.get("processing_time", 0)
        }
        
        # Add optional elements
        if addiction_effects:
            api_response["addiction_effects"] = addiction_effects
            
        if response.get("generate_image", False):
            api_response["should_generate_image"] = True
            
        if response.get("revelations_processed", 0) > 0:
            api_response["revelations_processed"] = response["revelations_processed"]
        tracker.end_phase()
        
        return jsonify(api_response)
        
    except Exception as e:
        # End current phase if one is active
        if tracker.current_phase:
            tracker.end_phase()
            
        logging.exception("[nyx_response] Error")
        return jsonify({
            "error": str(e),
            "performance": tracker.get_metrics()
        }), 500

@nyx_agent_bp.route("/nyx_reflection", methods=["POST"])
@timed_function(name="nyx_reflection")
async def nyx_reflection():
    """
    Generate a reflection from Nyx on a specific topic.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        topic = data.get("topic")
        
        # Initialize Nyx agent
        agent = NyxAgent(user_id, conversation_id)
        
        # Generate reflection
        reflection = await agent.generate_reflection(topic)
        
        return jsonify(reflection)
        
    except Exception as e:
        logging.exception("[nyx_reflection] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_introspection", methods=["GET"])
@timed_function(name="nyx_introspection")
async def nyx_introspection():
    """
    Get Nyx's introspection about her memories and understanding.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        # Check cache
        cache_key = f"introspection:{user_id}:{conversation_id}"
        cached_result = MEMORY_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Initialize Nyx agent
        agent = NyxAgent(user_id, int(conversation_id))
        
        # Get introspection
        introspection = await agent.get_introspection()
        
        # Cache result
        MEMORY_CACHE.set(cache_key, introspection, 300)  # 5 minute TTL
        
        return jsonify(introspection)
        
    except Exception as e:
        logging.exception("[nyx_introspection] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_memory_maintenance", methods=["POST"])
@timed_function(name="nyx_memory_maintenance")
async def nyx_memory_maintenance():
    """
    Manually trigger memory maintenance for Nyx.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        # Initialize Nyx agent
        agent = NyxAgent(user_id, conversation_id)
        
        # Run maintenance
        await agent._run_maintenance()
        
        # Clear caches
        MEMORY_CACHE.remove_pattern(f"introspection:{user_id}:{conversation_id}")
        
        return jsonify({"status": "Memory maintenance completed"})
        
    except Exception as e:
        logging.exception("[nyx_memory_maintenance] Error")
        return jsonify({"error": str(e)}), 500

# Helper functions
async def process_time_advancement(user_id, conversation_id, activity_type, data):
    """Process time advancement with proper verification."""
    # Check if activity should advance time
    advance_info = should_advance_time(activity_type)
    
    # Default time result
    time_result = {
        "time_advanced": False,
        "would_advance": False,
        "periods": 0,
        "confirm_needed": False
    }
    
    # Handle direct confirmation
    if data.get("confirm_time_advance", False) and advance_info["should_advance"]:
        # Actually perform time advance
        time_result = await advance_time_with_events(
            user_id, conversation_id, activity_type
        )
    elif advance_info["should_advance"]:
        time_result = {
            "time_advanced": False,
            "would_advance": True,
            "periods": advance_info["periods"],
            "confirm_needed": True
        }
    
    return time_result

async def process_relationship_events(user_id, conversation_id, data):
    """Process relationship events and crossroads with enhanced dynamics."""
    result = {
        "event": None,
        "result": None
    }
    
    # Check for crossroads events (significant relationship moments)
    if data.get("check_crossroads", False):
        event = await check_for_relationship_crossroads(user_id, conversation_id)
        if event:
            result["event"] = event
    
    # Check for relationship rituals (shared activities that boost bonds)
    if data.get("check_rituals", False):
        ritual = await check_for_relationship_ritual(user_id, conversation_id)
        if ritual:
            result["ritual"] = ritual
    
    # Process crossroads choice if provided
    if data.get("crossroads_choice") is not None and data.get("crossroads_name") and data.get("link_id"):
        choice_result = await apply_crossroads_choice(
            user_id,
            conversation_id,
            int(data["link_id"]),
            data["crossroads_name"],
            int(data["crossroads_choice"])
        )
        
        result["result"] = choice_result
    
    return result

def process_rule_enforcement(user_id, conversation_id):
    """Process game rule enforcement."""
    # Enforce all rules
    triggered_effects = enforce_all_rules_on_player()
    
    return triggered_effects

async def store_messages(user_id, conversation_id, user_input, nyx_response):
    """Store user and Nyx messages in database."""
    async with (await get_db_connection()) as conn:
        # Store user message
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """, conversation_id, "user", user_input)
        
        # Store Nyx message
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """, conversation_id, "Nyx", nyx_response)
