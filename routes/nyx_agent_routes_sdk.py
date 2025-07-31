# routes/nyx_agent_routes_sdk.py

import logging
import json
import time
import contextlib
from quart import Blueprint, request, jsonify, session
from typing import Dict, Any, Optional

from nyx.nyx_agent_sdk import process_user_input, generate_reflection, initialize_agents
from logic.time_cycle import should_advance_time, advance_time_with_events
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    event_generator,
    process_relationship_interaction_tool,
    get_relationship_summary_tool,
    poll_relationship_events_tool,
    drain_relationship_events_tool
)
from logic.addiction_system_sdk import process_addiction_effects, get_addiction_status
from logic.rule_enforcement import enforce_all_rules_on_player
from utils.performance import PerformanceTracker, timed_function
from utils.caching import NPC_CACHE, MEMORY_CACHE
from db.connection import get_db_connection_context
from agents import RunContextWrapper

nyx_agent_bp = Blueprint("nyx_agent_bp", __name__)

# Initialize the agent system when the Flask app starts
_nyx_initialized = False

@nyx_agent_bp.before_app_request
async def initialize_once():
    """
    Initialize the Nyx agent system exactly once.
    """
    global _nyx_initialized
    if not _nyx_initialized:
        await initialize_agents()
        logging.info("Nyx agent system initialized")
        _nyx_initialized = True

@nyx_agent_bp.route("/nyx_response", methods=["POST"])
@timed_function(name="nyx_response")
async def nyx_response():
    """
    Enhanced endpoint that processes user input through unified pipeline
    and returns a complete response using the OpenAI Agents SDK.
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
        
        # Process input through unified pipeline
        tracker.start_phase("agent_processing")
        from logic.chatgpt_integration import get_chatgpt_response
        
        response_data = await get_chatgpt_response(
            conversation_id=conversation_id,
            aggregator_text="",
            user_input=user_input,
            reflection_enabled=data.get('reflection_enabled', False),
            use_nyx_integration=True,
            context=context  # Pass context if get_chatgpt_response accepts it
        )
        
        # Transform to expected format
        if response_data['type'] == 'function_call':
            response_text = response_data['function_args'].get('narrative', '')
            generate_image = response_data['function_args'].get('image_generation', {}).get('generate', False)
            image_prompt = response_data['function_args'].get('image_generation', {}).get('reason', '')
            time_advancement = response_data['function_args'].get('time_advancement', False)
            environment_update = response_data['function_args'].get('environment_update')
            tension_level = response_data['function_args'].get('tension_level')
        else:
            response_text = response_data.get('response', '')
            generate_image = False
            image_prompt = None
            time_advancement = False
            environment_update = None
            tension_level = None
        
        # Build response object to match existing interface
        response = {
            "message": response_text,
            "generate_image": generate_image,
            "image_prompt": image_prompt,
            "time_advancement": time_advancement,
            "environment_update": environment_update,
            "tension_level": tension_level
        }
        tracker.end_phase()
        
        # Post-processing steps
        tracker.start_phase("post_processing")
        
        # 1. Check for time advancement
        activity_type = data.get("activity_type", "conversation")
        time_result = await process_time_advancement(
            user_id, 
            conversation_id, 
            activity_type, 
            data,
            response.get("time_advancement", False)
        )
        
        # 2. Check for relationship events using new dynamic system
        relationship_events = await process_relationship_events(user_id, conversation_id, data)
        
        # 3. Process rule enforcement
        rule_results = enforce_all_rules_on_player()
        
        # 4. Check addiction system
        addiction_status = await get_addiction_status(user_id, conversation_id, context["player_name"])
        addiction_effects = None
        if addiction_status and addiction_status.get("has_addictions"):
            addiction_effects = await process_addiction_effects(
                user_id, conversation_id, context["player_name"], addiction_status
            )
        tracker.end_phase()

        # 5. Update narrative arcs   
        try:
            from nyx.scene_manager_sdk import update_narrative_arcs_for_interaction
            await update_narrative_arcs_for_interaction(
                user_id, conversation_id, user_input, response["message"]
            )
        except Exception as e:
            logging.warning(f"Error updating narrative arcs: {e}")    
        
        # Build complete response
        tracker.start_phase("response_building")
        api_response = {
            "message": response["message"],
            "time_result": time_result,
            "confirm_needed": time_result.get("would_advance", False) and not data.get("confirm_time_advance", False),
            "relationship_events": relationship_events,
            "rule_effects": rule_results,
            "performance_metrics": tracker.get_metrics()
        }
        
        # Add optional elements
        if addiction_effects:
            api_response["addiction_effects"] = addiction_effects
            
        if response.get("generate_image", False):
            api_response["should_generate_image"] = True
            api_response["image_prompt"] = response.get("image_prompt")
            
        if response.get("environment_update"):
            api_response["environment_update"] = response.get("environment_update")
            
        if response.get("tension_level") is not None:
            api_response["tension_level"] = response.get("tension_level")
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
        
        # Generate reflection
        reflection = await generate_reflection(user_id, conversation_id, topic)
        
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
        
        # Generate reflection with introspection focus
        introspection = await generate_reflection(user_id, int(conversation_id), "self_understanding")
        
        # Cache result
        MEMORY_CACHE.set(cache_key, introspection, 300)  # 5 minute TTL
        
        return jsonify(introspection)
        
    except Exception as e:
        logging.exception("[nyx_introspection] Error")
        return jsonify({"error": str(e)}), 500

# Helper functions
async def process_time_advancement(
    user_id: int, 
    conversation_id: int, 
    activity_type: str, 
    data: Dict[str, Any],
    agent_requested_advancement: bool = False
) -> Dict[str, Any]:
    """Process time advancement with proper verification."""
    # Check if activity should advance time or if the agent requested advancement
    advance_info = await should_advance_time(activity_type)
    should_advance = advance_info["should_advance"] or agent_requested_advancement
    
    # Default time result
    time_result = {
        "time_advanced": False,
        "would_advance": False,
        "periods": 0,
        "confirm_needed": False
    }
    
    # Handle direct confirmation
    if data.get("confirm_time_advance", False) and should_advance:
        # Actually perform time advance
        time_result = await advance_time_with_events(
            user_id, conversation_id, activity_type
        )
    elif should_advance:
        time_result = {
            "time_advanced": False,
            "would_advance": True,
            "periods": advance_info["periods"],
            "confirm_needed": True
        }
    
    return time_result

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
        
        # Call the memory maintenance function from SDK
        from nyx.memory_integration_sdk import perform_memory_maintenance
        
        # Run maintenance
        result = await perform_memory_maintenance(user_id, conversation_id)
        
        # Clear caches
        MEMORY_CACHE.remove_pattern(f"introspection:{user_id}:{conversation_id}")
        
        return jsonify({"status": "Memory maintenance completed", "result": result})
        
    except Exception as e:
        logging.exception("[nyx_memory_maintenance] Error")
        return jsonify({"error": str(e)}), 500

async def process_relationship_events(
    user_id: int, 
    conversation_id: int, 
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Process relationship events using the new dynamic relationships system."""
    result = {
        "events": [],
        "interaction_results": None
    }
    
    # Create context wrapper for tools
    ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    # Process any relationship interactions from the current activity
    if data.get("relationship_interaction"):
        interaction = data["relationship_interaction"]
        
        # Use the new process_relationship_interaction_tool
        interaction_result = await process_relationship_interaction_tool(
            ctx=ctx,
            entity1_type=interaction.get("entity1_type", "player"),
            entity1_id=interaction.get("entity1_id", 1),  # Assuming player ID is 1
            entity2_type=interaction.get("entity2_type", "npc"),
            entity2_id=interaction.get("entity2_id"),
            interaction_type=interaction.get("interaction_type", "conversation"),
            context=interaction.get("context", "casual"),
            check_for_event=True
        )
        
        result["interaction_results"] = interaction_result
        
        # Check if an event was generated
        if interaction_result.get("event"):
            result["events"].append(interaction_result["event"])
    
    # Poll for any pending relationship events
    poll_result = await poll_relationship_events_tool(ctx=ctx, timeout=0.1)
    if poll_result.get("has_event"):
        result["events"].append(poll_result["event"])
    
    # If we need to drain multiple events (batch processing)
    if data.get("drain_all_events", False):
        drain_result = await drain_relationship_events_tool(ctx=ctx, max_events=10)
        if drain_result.get("events"):
            result["events"].extend([e["event"] for e in drain_result["events"]])
    
    # Process any event choices
    if data.get("event_choice") and data.get("event_id"):
        # Handle event choice application
        # This would need to be implemented based on your event handling logic
        choice_result = await apply_relationship_event_choice(
            user_id,
            conversation_id,
            data["event_id"],
            data["event_choice"]
        )
        result["choice_result"] = choice_result
    
    return result

async def apply_relationship_event_choice(
    user_id: int,
    conversation_id: int,
    event_id: str,
    choice_id: str
) -> Dict[str, Any]:
    """Apply the effects of a relationship event choice."""
    # This is a placeholder - you'll need to implement based on your event structure
    # The new system generates events with choices that have potential_impacts
    
    # For now, return a basic structure
    return {
        "success": True,
        "event_id": event_id,
        "choice_id": choice_id,
        "message": "Choice applied successfully"
    }

@contextlib.asynccontextmanager
async def get_db_transaction(user_id, conv_id):
    """Context manager for database transactions."""
    async with get_db_connection_context() as conn:
        try:
            yield conn
            await conn.commit()
        except Exception as e:
            logging.error(f"Transaction error: {e}")
            raise

async def store_message(user_id, conv_id, sender, content, structured_content=None):
    """Store a message in the database."""
    async with get_db_transaction(user_id, conv_id) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO messages (conversation_id, sender, content, structured_content)
                VALUES (%s, %s, %s, %s)
            """, (
                conv_id,
                sender,
                content,
                json.dumps(structured_content) if structured_content else None
            ))

# New route for getting relationship summaries
@nyx_agent_bp.route("/relationship_summary", methods=["GET"])
@timed_function(name="relationship_summary")
async def get_relationship_summary():
    """Get a summary of a specific relationship."""
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        # Get query parameters
        conversation_id = request.args.get("conversation_id", type=int)
        entity1_type = request.args.get("entity1_type", "player")
        entity1_id = request.args.get("entity1_id", 1, type=int)
        entity2_type = request.args.get("entity2_type", "npc")
        entity2_id = request.args.get("entity2_id", type=int)
        
        if not all([conversation_id, entity2_id]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Create context wrapper
        ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })
        
        # Get relationship summary
        summary = await get_relationship_summary_tool(
            ctx=ctx,
            entity1_type=entity1_type,
            entity1_id=entity1_id,
            entity2_type=entity2_type,
            entity2_id=entity2_id
        )
        
        return jsonify(summary)
        
    except Exception as e:
        logging.exception("[get_relationship_summary] Error")
        return jsonify({"error": str(e)}), 500
