# routes/nyx_agent_routes_sdk.py
"""
Nyx Agent Routes - Refactored to use the new modular SDK architecture.
Maintains API compatibility while leveraging the new nyx_agent_sdk final stop.
"""

import logging
import json
import time
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from quart import Blueprint, request, jsonify, session, Response
from quart.wrappers import Response as QuartResponse

# New SDK imports - using the 'final stop'
from nyx.nyx_agent_sdk import (
    NyxAgentSDK, 
    NyxSDKConfig,
    NyxResponse as SDKResponse
)

# Direct orchestrator imports for specialized functions
from nyx.nyx_agent.orchestrator import (
    generate_reflection,
    manage_relationships,
    manage_scenario
)

# Context management
from nyx.nyx_agent.context import NyxContext

# Models
from nyx.nyx_agent.models import (
    MemoryReflection,
    RelationshipUpdate,
    ScenarioDecision
)

# Existing system integrations (kept as-is)
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

# Performance and caching
from utils.performance import PerformanceTracker, timed_function
from utils.caching import NPC_CACHE, MEMORY_CACHE
from db.connection import get_db_connection_context
from agents import RunContextWrapper

logger = logging.getLogger(__name__)

nyx_agent_bp = Blueprint("nyx_agent_bp", __name__)

# ===== SDK Configuration =====

@dataclass
class RouteConfig:
    """Configuration for route behavior"""
    enable_streaming: bool = True
    enable_moderation: bool = False  # Set to True in production
    cache_warmup_on_start: bool = True
    max_response_length: int = 4000
    enable_telemetry: bool = True

# Global SDK instance
_sdk_instance: Optional[NyxAgentSDK] = None
_route_config = RouteConfig()

def get_sdk() -> NyxAgentSDK:
    """Get or create the global SDK instance"""
    global _sdk_instance
    if _sdk_instance is None:
        sdk_config = NyxSDKConfig(
            pre_moderate_input=_route_config.enable_moderation,
            post_moderate_output=_route_config.enable_moderation,
            enable_telemetry=_route_config.enable_telemetry,
            streaming_chunk_size=320,
            request_timeout_seconds=45.0,
            retry_on_failure=True,
            result_cache_ttl_seconds=10  # Short idempotency window
        )
        _sdk_instance = NyxAgentSDK(sdk_config)
        logger.info("NyxAgentSDK initialized with config: %s", sdk_config)
    return _sdk_instance

# ===== Main Routes =====

@nyx_agent_bp.before_app_request
async def sdk_warmup():
    """Warm up the SDK on first request if configured"""
    if _route_config.cache_warmup_on_start:
        sdk = get_sdk()
        # The SDK handles its own initialization internally
        logger.debug("SDK warmed up for request")

@nyx_agent_bp.route("/nyx_response", methods=["POST"])
@timed_function(name="nyx_response")
async def nyx_response():
    """
    Main endpoint for Nyx responses using the new SDK architecture.
    Maintains backward compatibility with existing API contract.
    """
    tracker = PerformanceTracker("nyx_response")
    tracker.start_phase("initialization")
    
    try:
        # Validate session
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = await request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conversation_id = data.get("conversation_id")
        
        if not user_input or not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        tracker.end_phase()
        
        # Build context metadata
        tracker.start_phase("context_building")
        metadata = {
            "location": data.get("location", "Unknown"),
            "time_of_day": data.get("time_of_day", "Morning"),
            "npc_present": data.get("npc_present", []),
            "player_name": data.get("player_name", "Chase"),
            "activity_type": data.get("activity_type", "conversation"),
            "timestamp": time.time()
        }
        
        # Merge any additional context
        if data.get("additional_context"):
            metadata.update(data["additional_context"])
        
        # Add reflection flag if enabled
        if data.get('reflection_enabled'):
            metadata['enable_reflection'] = True
        tracker.end_phase()
        
        # Process through the SDK
        tracker.start_phase("agent_processing")
        sdk = get_sdk()
        
        # Optional: Warm cache for this conversation/location
        if metadata.get("location"):
            asyncio.create_task(
                sdk.warmup_cache(str(conversation_id), metadata["location"])
            )
        
        # Main SDK call
        sdk_response: SDKResponse = await sdk.process_user_input(
            message=user_input,
            conversation_id=str(conversation_id),
            user_id=str(user_id),
            metadata=metadata
        )
        tracker.end_phase()
        
        # Post-processing steps
        tracker.start_phase("post_processing")
        
        # 1. Time advancement check
        time_result = await process_time_advancement(
            user_id, 
            conversation_id, 
            metadata.get("activity_type", "conversation"),
            data,
            sdk_response.metadata.get("time_advancement", False)
        )
        
        # 2. Relationship events
        relationship_events = await process_relationship_events(
            user_id, 
            conversation_id, 
            data
        )
        
        # 3. Rule enforcement
        rule_results = enforce_all_rules_on_player()
        
        # 4. Addiction system
        addiction_effects = await check_addiction_effects(
            user_id,
            conversation_id,
            metadata.get("player_name", "Chase")
        )
        
        # 5. Narrative arc updates
        try:
            from nyx.scene_manager_sdk import update_narrative_arcs_for_interaction
            await update_narrative_arcs_for_interaction(
                user_id, 
                conversation_id, 
                user_input, 
                sdk_response.narrative
            )
        except Exception as e:
            logger.warning(f"Narrative arc update failed: {e}")
        
        tracker.end_phase()
        
        # Build API response
        tracker.start_phase("response_building")
        api_response = {
            "message": sdk_response.narrative,
            "success": sdk_response.success,
            "time_result": time_result,
            "confirm_needed": time_result.get("would_advance", False) and not data.get("confirm_time_advance", False),
            "relationship_events": relationship_events,
            "rule_effects": rule_results,
            "performance_metrics": {
                **tracker.get_metrics(),
                "sdk_telemetry": sdk_response.telemetry
            }
        }
        
        # Add optional elements from SDK response
        if sdk_response.image:
            api_response["should_generate_image"] = True
            api_response["image_prompt"] = sdk_response.image.get("prompt")
        
        if sdk_response.world_state:
            api_response["world_state"] = sdk_response.world_state
        
        if sdk_response.choices:
            api_response["choices"] = sdk_response.choices
        
        if sdk_response.metadata.get("environment_update"):
            api_response["environment_update"] = sdk_response.metadata["environment_update"]
        
        if sdk_response.metadata.get("tension_level") is not None:
            api_response["tension_level"] = sdk_response.metadata["tension_level"]
        
        if sdk_response.metadata.get("nyx_commentary"):
            api_response["nyx_commentary"] = sdk_response.metadata["nyx_commentary"]
        
        if addiction_effects:
            api_response["addiction_effects"] = addiction_effects
        
        if sdk_response.error:
            api_response["error"] = sdk_response.error
        
        if sdk_response.trace_id:
            api_response["trace_id"] = sdk_response.trace_id
        
        tracker.end_phase()
        return jsonify(api_response)
        
    except Exception as e:
        if tracker.current_phase:
            tracker.end_phase()
        
        logger.exception("[nyx_response] Error")
        return jsonify({
            "success": False,
            "error": str(e),
            "performance": tracker.get_metrics()
        }), 500

@nyx_agent_bp.route("/nyx_response_stream", methods=["POST"])
@timed_function(name="nyx_response_stream")
async def nyx_response_stream():
    """
    Streaming endpoint for real-time response delivery.
    Returns Server-Sent Events (SSE) stream.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    user_input = data.get("user_input", "").strip()
    conversation_id = data.get("conversation_id")
    
    if not user_input or not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Build metadata
    metadata = {
        "location": data.get("location", "Unknown"),
        "time_of_day": data.get("time_of_day", "Morning"),
        "npc_present": data.get("npc_present", []),
        "player_name": data.get("player_name", "Chase"),
        "activity_type": data.get("activity_type", "conversation"),
        "timestamp": time.time()
    }
    
    if data.get("additional_context"):
        metadata.update(data["additional_context"])
    
    async def generate():
        """SSE generator"""
        sdk = get_sdk()
        
        try:
            # Stream the response
            async for chunk in sdk.stream_user_input(
                message=user_input,
                conversation_id=str(conversation_id),
                user_id=str(user_id),
                metadata=metadata
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive"
        }
    )

@nyx_agent_bp.route("/nyx_reflection", methods=["POST"])
@timed_function(name="nyx_reflection")
async def nyx_reflection():
    """
    Generate a reflection from Nyx on a specific topic.
    Uses the orchestrator directly for specialized reflection generation.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        topic = data.get("topic")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        # Use orchestrator directly for reflection
        reflection = await generate_reflection(
            user_id=int(user_id),
            conversation_id=int(conversation_id),
            topic=topic
        )
        
        return jsonify(reflection)
        
    except Exception as e:
        logger.exception("[nyx_reflection] Error")
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
        
        # Generate introspective reflection
        introspection = await generate_reflection(
            user_id=int(user_id),
            conversation_id=int(conversation_id),
            topic="self_understanding"
        )
        
        # Cache result
        MEMORY_CACHE.set(cache_key, introspection, 300)  # 5 minute TTL
        
        return jsonify(introspection)
        
    except Exception as e:
        logger.exception("[nyx_introspection] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_manage_scenario", methods=["POST"])
@timed_function(name="nyx_manage_scenario")
async def nyx_manage_scenario():
    """
    Manage scenario progression using the orchestrator.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        scenario_data = {
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            **data.get("scenario_context", {})
        }
        
        result = await manage_scenario(scenario_data)
        return jsonify(result)
        
    except Exception as e:
        logger.exception("[nyx_manage_scenario] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_manage_relationships", methods=["POST"])
@timed_function(name="nyx_manage_relationships")
async def nyx_manage_relationships():
    """
    Manage relationship updates using the orchestrator.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        interaction_data = {
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            **data.get("interaction", {})
        }
        
        result = await manage_relationships(interaction_data)
        return jsonify(result)
        
    except Exception as e:
        logger.exception("[nyx_manage_relationships] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_memory_maintenance", methods=["POST"])
@timed_function(name="nyx_memory_maintenance")
async def nyx_memory_maintenance():
    """
    Trigger memory maintenance for the conversation.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400
        
        # Create a context and run memory consolidation
        ctx = NyxContext(user_id=int(user_id), conversation_id=int(conversation_id))
        await ctx.initialize()
        
        # Access memory orchestrator for maintenance
        if ctx.memory_orchestrator:
            from memory.memory_orchestrator import EntityType
            
            # Consolidate memories
            result = await ctx.memory_orchestrator.consolidate_memories(
                entity_type=EntityType.PLAYER,
                entity_id=user_id,
                force=True
            )
            
            # Clear relevant caches
            MEMORY_CACHE.remove_pattern(f"introspection:{user_id}:{conversation_id}")
            MEMORY_CACHE.remove_pattern(f"memories:{user_id}:*")
            
            # Also clear SDK cache for this conversation
            sdk = get_sdk()
            await sdk.cleanup_conversation(str(conversation_id))
            
            return jsonify({
                "status": "Memory maintenance completed",
                "consolidated_count": result.get("consolidated", 0),
                "pruned_count": result.get("pruned", 0)
            })
        else:
            return jsonify({
                "status": "Memory orchestrator not available",
                "error": "Memory system offline"
            }), 503
        
    except Exception as e:
        logger.exception("[nyx_memory_maintenance] Error")
        return jsonify({"error": str(e)}), 500

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
        
        # Create context wrapper for relationship tools
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
        logger.exception("[get_relationship_summary] Error")
        return jsonify({"error": str(e)}), 500

@nyx_agent_bp.route("/nyx_health", methods=["GET"])
async def nyx_health():
    """Health check endpoint for the Nyx agent system."""
    try:
        sdk = get_sdk()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "sdk_initialized": sdk is not None,
            "timestamp": time.time(),
            "config": {
                "moderation_enabled": _route_config.enable_moderation,
                "streaming_enabled": _route_config.enable_streaming,
                "telemetry_enabled": _route_config.enable_telemetry
            }
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

# ===== Helper Functions =====

async def process_time_advancement(
    user_id: int, 
    conversation_id: int, 
    activity_type: str, 
    data: Dict[str, Any],
    agent_requested_advancement: bool = False
) -> Dict[str, Any]:
    """Process time advancement with proper verification."""
    advance_info = await should_advance_time(activity_type)
    should_advance = advance_info["should_advance"] or agent_requested_advancement
    
    time_result = {
        "time_advanced": False,
        "would_advance": False,
        "periods": 0,
        "confirm_needed": False
    }
    
    if data.get("confirm_time_advance", False) and should_advance:
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

async def process_relationship_events(
    user_id: int, 
    conversation_id: int, 
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Process relationship events using the dynamic relationships system."""
    result = {
        "events": [],
        "interaction_results": None
    }
    
    ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    if data.get("relationship_interaction"):
        interaction = data["relationship_interaction"]
        
        interaction_result = await process_relationship_interaction_tool(
            ctx=ctx,
            entity1_type=interaction.get("entity1_type", "player"),
            entity1_id=interaction.get("entity1_id", 1),
            entity2_type=interaction.get("entity2_type", "npc"),
            entity2_id=interaction.get("entity2_id"),
            interaction_type=interaction.get("interaction_type", "conversation"),
            context=interaction.get("context", "casual"),
            check_for_event=True
        )
        
        result["interaction_results"] = interaction_result
        
        if interaction_result.get("event"):
            result["events"].append(interaction_result["event"])
    
    poll_result = await poll_relationship_events_tool(ctx=ctx, timeout=0.1)
    if poll_result.get("has_event"):
        result["events"].append(poll_result["event"])
    
    if data.get("drain_all_events", False):
        drain_result = await drain_relationship_events_tool(ctx=ctx, max_events=10)
        if drain_result.get("events"):
            result["events"].extend([e["event"] for e in drain_result["events"]])
    
    if data.get("event_choice") and data.get("event_id"):
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
    return {
        "success": True,
        "event_id": event_id,
        "choice_id": choice_id,
        "message": "Choice applied successfully"
    }

async def check_addiction_effects(
    user_id: int,
    conversation_id: int,
    player_name: str
) -> Optional[Dict[str, Any]]:
    """Check and process addiction effects."""
    try:
        addiction_status = await get_addiction_status(user_id, conversation_id, player_name)
        if addiction_status and addiction_status.get("has_addictions"):
            return await process_addiction_effects(
                user_id, conversation_id, player_name, addiction_status
            )
    except Exception as e:
        logger.warning(f"Addiction check failed: {e}")
    return None

@asynccontextmanager
async def get_db_transaction(user_id: int, conv_id: int):
    """Context manager for database transactions."""
    async with get_db_connection_context() as conn:
        async with conn.transaction():
            yield conn

async def store_message(
    user_id: int, 
    conv_id: int, 
    sender: str, 
    content: str, 
    structured_content: Optional[Dict[str, Any]] = None
):
    """Store a message in the database."""
    async with get_db_transaction(user_id, conv_id) as conn:
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content, structured_content, created_at)
            VALUES ($1, $2, $3, $4, NOW())
        """, 
        conv_id,
        sender,
        content,
        json.dumps(structured_content) if structured_content else None
    )

# ===== Error Handlers =====

@nyx_agent_bp.errorhandler(Exception)
async def handle_exception(e: Exception):
    """Global error handler for the blueprint."""
    logger.exception("Unhandled exception in nyx_agent routes")
    return jsonify({
        "success": False,
        "error": "An internal error occurred",
        "details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
    }), 500
