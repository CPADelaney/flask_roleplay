# routes/conflict_routes.py

"""
Refactored routes for the modularized conflict system.
All operations now go through the central ConflictSynthesizer.
Integrated with NYX governance for permission checking and action reporting.
"""

import logging
import json
import random
import os
import asyncio
from functools import wraps
from typing import Dict, Any, Optional, List
from datetime import datetime
from quart import Blueprint, request, jsonify, session

# Import the new synthesizer API
from logic.conflict_system.conflict_synthesizer import (
    get_synthesizer,
    release_synthesizer,
    # Import constants for metrics
    MAX_BUNDLE_CACHE,
    MAX_EVENT_QUEUE,
    MAX_EVENT_HISTORY
)

# Import governance integration
from nyx.governance import AgentType
from nyx.governance_helpers import (
    check_permission,
    report_action,
    propose_canonical_change,
)

logger = logging.getLogger(__name__)

# Create blueprint
conflict_bp = Blueprint("conflict", __name__)

# ===============================================================================
# MIDDLEWARE & HELPERS
# ===============================================================================

def require_login(f):
    """Decorator to require login."""
    @wraps(f)
    async def decorated(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        return await f(user_id, *args, **kwargs)
    return decorated

async def check_conflict_permission(
    user_id: int,
    conversation_id: int,
    action_type: str,
    conflict_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Check with governance before performing conflict operations."""
    try:
        permission = await check_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action_type=action_type,
            action_details=conflict_data
        )
        return permission
    except Exception as e:
        logger.warning(f"Governance check failed, allowing by default: {e}")
        return {"approved": True, "reasoning": "Governance check failed, allowing by default"}

async def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> Optional[Dict[str, Any]]:
    """Validate that required fields are present in request data."""
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        return {
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "status": 400
        }
    
    # Convert conversation_id to int if it's a string
    if "conversation_id" in data and isinstance(data["conversation_id"], str):
        try:
            data["conversation_id"] = int(data["conversation_id"])
        except ValueError:
            return {
                "error": "conversation_id must be a valid integer",
                "status": 400
            }
    
    return None

# ===============================================================================
# SYSTEM STATE & STATUS ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/status", methods=["GET"])
@require_login
async def get_system_status(user_id):
    """Get the overall conflict system status."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Get synthesizer instance
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get comprehensive system state
        state = await synthesizer.get_system_state()
        
        return jsonify({
            "success": True,
            "system_state": state
        })
    except Exception as e:
        logger.error(f"Error getting system state: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/health", methods=["GET"])
@require_login
async def get_system_health(user_id):
    """Get health status of all subsystems."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        health = await synthesizer.health_check()
        
        return jsonify({
            "success": True,
            "health": health
        })
    except Exception as e:
        logger.error(f"Error checking system health: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# CONFLICT MANAGEMENT ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/active", methods=["GET"])
@require_login
async def get_active_conflicts(user_id):
    """Get all active conflicts."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get system state which includes active conflicts
        state = await synthesizer.get_system_state()
        
        # Extract and enrich active conflicts
        active_conflicts = []
        for conflict_id in state.get("active_conflicts", []):
            conflict_state = await synthesizer.get_conflict_state(conflict_id)
            if conflict_state:
                active_conflicts.append({
                    "conflict_id": conflict_id,
                    "type": conflict_state.get("conflict_type", "unknown"),
                    "status": conflict_state.get("status", "active"),
                    "participants": conflict_state.get("participants", []),
                    "tension_level": conflict_state.get("tension", {}).get("current_level", 0),
                    "phase": conflict_state.get("flow", {}).get("current_phase", "unknown"),
                    "is_multiparty": conflict_state.get("is_multiparty", False),
                    "party_count": conflict_state.get("party_count", 2)
                })
        
        return jsonify({
            "success": True,
            "conflicts": active_conflicts,
            "count": len(active_conflicts),
            "metrics": state.get("metrics", {})
        })
    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/details/<int:conflict_id>", methods=["GET"])
@require_login
async def get_conflict_details(user_id, conflict_id):
    """Get detailed information about a specific conflict."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get detailed conflict state
        conflict_state = await synthesizer.get_conflict_state(conflict_id)
        
        if not conflict_state:
            return jsonify({"error": "Conflict not found"}), 404
        
        return jsonify({
            "success": True,
            "conflict": conflict_state
        })
    except Exception as e:
        logger.error(f"Error getting conflict details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# CONFLICT CREATION ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/create", methods=["POST"])
@require_login
async def create_conflict(user_id):
    """Create a new conflict through the synthesizer with governance approval."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Get conflict parameters
        conflict_type = data.get("conflict_type", "slice")
        
        # Validate conflict type
        valid_types = ["social", "slice", "background", "political", "economic", "resource", "ideological"]
        if conflict_type not in valid_types:
            return jsonify({
                "error": f"Invalid conflict type. Must be one of: {', '.join(valid_types)}"
            }), 400
        
        # Build context
        context = data.get("context", {})
        
        # Auto-detect multiparty status
        participants = context.get("participants", [])
        is_multiparty = context.get("is_multiparty", len(participants) > 2)
        
        if is_multiparty:
            context["is_multiparty"] = True
            context["party_count"] = len(participants) if participants else context.get("party_count", 3)
            context["multiparty_dynamics"] = context.get("multiparty_dynamics", {
                "alliance_potential": True,
                "shifting_sides": True,
                "faction_formation": len(participants) > 4 if participants else False,
                "betrayal_likelihood": "medium"
            })
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="create_conflict",
            conflict_data={
                "conflict_type": conflict_type,
                "context": context,
                "is_multiparty": is_multiparty
            }
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied conflict creation",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Apply governance overrides if present
        if permission.get("override_action"):
            if "conflict_type" in permission["override_action"]:
                conflict_type = permission["override_action"]["conflict_type"]
            if "context" in permission["override_action"]:
                context.update(permission["override_action"]["context"])
        
        # Create conflict through synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        result = await synthesizer.create_conflict(conflict_type, context)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "create_conflict",
                "conflict_type": conflict_type,
                "is_multiparty": is_multiparty
            },
            result={
                "success": True,
                "conflict_id": result.get("conflict_id", -1)
            }
        )
        
        # Propose canonical change if significant
        if result.get("conflict_id") and conflict_type in ["political", "background"]:
            await propose_canonical_change(
                user_id=user_id,
                conversation_id=conversation_id,
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_id=f"conflict_system_{conversation_id}",
                change_type="conflict_creation",
                change_data={
                    "conflict_id": result["conflict_id"],
                    "conflict_type": conflict_type,
                    "name": result.get("conflict_name", "Unnamed Conflict"),
                    "impact": "medium"
                }
            )
        
        return jsonify({
            "success": True,
            "conflict": result,
            "is_multiparty": is_multiparty,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "override_applied": bool(permission.get("override_action"))
            }
        })
    except Exception as e:
        logger.error(f"Error creating conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/emergent", methods=["POST"])
@require_login
async def generate_emergent_conflict(user_id):
    """Generate an emergent conflict based on current world state."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="generate_emergent_conflict",
            conflict_data=data
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied emergent conflict generation",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Determine conflict type based on triggers and world state
        base_conflict_types = ["social", "slice", "background"]
        weights = [0.4, 0.4, 0.2]  # Default weights
        
        # Adjust weights based on context
        trigger = data.get("trigger", "")
        if trigger == "relationship_tension":
            weights = [0.7, 0.2, 0.1]
        elif trigger == "world_event":
            weights = [0.1, 0.2, 0.7]
        elif trigger == "daily_routine":
            weights = [0.2, 0.7, 0.1]
        
        conflict_type = random.choices(base_conflict_types, weights=weights)[0]
        
        # Apply governance override if present
        if permission.get("override_action", {}).get("conflict_type"):
            conflict_type = permission["override_action"]["conflict_type"]
        
        # Randomly determine if multiparty (30% chance)
        is_multiparty = random.random() < 0.3
        
        # Build emergent context
        context = {
            "emergent": True,
            "trigger": trigger,
            "is_multiparty": is_multiparty,
            "integration_mode": data.get("integration_mode", "emergent")
        }
        
        # Add type-specific emergent context
        if conflict_type == "social":
            context.update({
                "relationship_factor": random.choice(["trust", "respect", "affection", "loyalty"]),
                "tension_source": random.choice(["betrayal", "miscommunication", "competition", "past_grievance"])
            })
        elif conflict_type == "slice":
            context.update({
                "daily_issue": random.choice(["resources", "scheduling", "boundaries", "responsibilities"]),
                "urgency": random.choice(["low", "medium", "high"])
            })
        elif conflict_type == "background":
            context.update({
                "scope": random.choice(["local", "regional", "global"]),
                "visibility": random.choice(["hidden", "rumored", "public"])
            })
        
        if is_multiparty:
            context["party_count"] = random.randint(3, 5)
            context["multiparty_dynamics"] = {
                "alliance_potential": True,
                "shifting_sides": random.random() < 0.5,
                "faction_formation": context.get("party_count", 3) > 3,
                "betrayal_likelihood": random.choice(["low", "medium", "high"])
            }
        
        # Generate through synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        result = await synthesizer.create_conflict(conflict_type, context)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "emergent_conflict_generated",
                "conflict_type": conflict_type,
                "trigger": trigger,
                "is_multiparty": is_multiparty
            },
            result={
                "success": True,
                "conflict_id": result.get("conflict_id", -1)
            }
        )
        
        logger.info(f"Generated emergent {conflict_type} conflict (multiparty: {is_multiparty}) for user {user_id}")
        
        return jsonify({
            "success": True,
            "conflict": result,
            "emergent": True,
            "is_multiparty": is_multiparty,
            "trigger": trigger
        })
        
    except Exception as e:
        logger.error(f"Error generating emergent conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# CONFLICT UPDATE & RESOLUTION ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/update", methods=["POST"])
@require_login
async def update_conflict(user_id):
    """Update an existing conflict."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id", "conflict_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    conflict_id = data["conflict_id"]
    update_context = data.get("context", {})
    
    try:
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="update_conflict",
            conflict_data={
                "conflict_id": conflict_id,
                "update_context": update_context
            }
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied conflict update",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Apply governance overrides
        if permission.get("override_action", {}).get("context"):
            update_context.update(permission["override_action"]["context"])
        
        # Update through synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        result = await synthesizer.update_conflict(conflict_id, update_context)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "update_conflict",
                "conflict_id": conflict_id
            },
            result={
                "success": True,
                "phase": result.get("phase")
            }
        )
        
        return jsonify({
            "success": True,
            "update_result": result,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "override_applied": bool(permission.get("override_action"))
            }
        })
    except Exception as e:
        logger.error(f"Error updating conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/resolve", methods=["POST"])
@require_login
async def resolve_conflict(user_id):
    """Resolve a conflict through the synthesizer with governance approval."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id", "conflict_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    conflict_id = data["conflict_id"]
    
    try:
        # Get resolution parameters
        resolution_type = data.get("resolution_type", "negotiated")
        context = data.get("context", {})
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="resolve_conflict",
            conflict_data={
                "conflict_id": conflict_id,
                "resolution_type": resolution_type,
                "context": context
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Conflict resolution denied by governance: {permission.get('reasoning')}")
            return jsonify({
                "success": False,
                "error": "Governance denied conflict resolution",
                "reasoning": permission.get("reasoning"),
                "governance_blocked": True
            }), 403
        
        # Apply governance overrides
        if permission.get("override_action"):
            if "resolution_type" in permission["override_action"]:
                resolution_type = permission["override_action"]["resolution_type"]
            if "context" in permission["override_action"]:
                context.update(permission["override_action"]["context"])
        
        # Resolve through synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        result = await synthesizer.resolve_conflict(conflict_id, resolution_type, context)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "resolve_conflict",
                "conflict_id": conflict_id,
                "resolution_type": resolution_type
            },
            result={
                "success": result.get("resolved", False),
                "resolution_details": result.get("resolution_details")
            }
        )
        
        # Propose canonical change for significant resolutions
        if result.get("resolved") and resolution_type in ["victory", "defeat", "treaty"]:
            await propose_canonical_change(
                user_id=user_id,
                conversation_id=conversation_id,
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_id=f"conflict_system_{conversation_id}",
                change_type="conflict_resolution",
                change_data={
                    "conflict_id": conflict_id,
                    "resolution_type": resolution_type,
                    "impact": "high",
                    "resolution_details": result.get("resolution_details", {})
                }
            )
        
        return jsonify({
            "success": True,
            "resolution": result,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "override_applied": bool(permission.get("override_action"))
            }
        })
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/scene-transition", methods=["POST"])
@require_login
async def handle_scene_transition(user_id):
    """Handle transition between scenes for conflict context."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    old_scene = data.get("old_scene", {})
    new_scene = data.get("new_scene", {})
    
    try:
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="scene_transition",
            conflict_data={
                "old_scene": old_scene,
                "new_scene": new_scene
            }
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied scene transition",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Use ConflictEventHooks for the transition
        from logic.conflict_system.integration_hooks import ConflictEventHooks
        
        context = await ConflictEventHooks.on_scene_transition(
            user_id, conversation_id, old_scene, new_scene
        )
        
        # Get synthesizer and emit transition event
        from logic.conflict_system.conflict_synthesizer import SystemEvent, EventType, SubsystemType
        
        synthesizer = await get_synthesizer(user_id, conversation_id)
        
        event = SystemEvent(
            event_id=f"scene_transition_{datetime.now().timestamp()}",
            event_type=EventType.SCENE_ENTER,
            source_subsystem=SubsystemType.ORCHESTRATOR,
            payload={
                'old_scene': old_scene, 
                'new_scene': new_scene, 
                'context': context
            },
            requires_response=False,
            priority=6
        )
        
        # Emit the event
        await synthesizer.emit_event(event)
        
        # Clear scene-specific caches
        await synthesizer._invalidate_caches_for_scene(new_scene.get("location_id"))
        
        return jsonify({
            "success": True,
            "transition_context": context,
            "old_scene": old_scene,
            "new_scene": new_scene
        })
    except Exception as e:
        logger.error(f"Error handling scene transition: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# SCENE PROCESSING ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/process-scene", methods=["POST"])
@require_login
async def process_scene(user_id):
    """Process a scene through the conflict system."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Build scene context
        scene_context = {
            "scene_type": data.get("scene_type", "dialogue"),
            "scene_description": data.get("scene_description", ""),
            "activity": data.get("activity", "conversation"),
            "activity_type": data.get("activity_type", "social"),
            "location": data.get("location", ""),
            "location_id": data.get("location_id"),
            "participants": data.get("participants", []),
            "present_npcs": data.get("present_npcs", []),
            "recent_events": data.get("recent_events", []),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "integration_mode": data.get("integration_mode", "emergent"),
            "boost_engagement": data.get("boost_engagement", False)
        }
        
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="process_scene",
            conflict_data={"scene_context": scene_context}
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied scene processing",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Process through synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)
        scene_result = await synthesizer.process_scene(scene_context)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "process_scene",
                "scene_type": scene_context.get("scene_type"),
                "location": scene_context.get("location")
            },
            result={
                "success": True,
                "conflicts_detected": scene_result.get("conflicts_detected", []),
                "events_triggered": scene_result.get("events_triggered", [])
            }
        )
        
        return jsonify({
            "success": True,
            "scene_result": scene_result,
            "governance_metadata": {
                "permission_tracking_id": permission.get("tracking_id", -1),
                "directive_applied": permission.get("directive_applied", False)
            }
        })
    except Exception as e:
        logger.error(f"Error processing scene: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# SCENE BUNDLE ROUTES (for fast context assembly)
# ===============================================================================

@conflict_bp.route("/api/conflict/scene-bundle", methods=["GET"])
@require_login
async def get_scene_bundle(user_id):
    """Get scene-scoped conflict bundle for fast context assembly."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Build scene scope from query params
        scope = {
            "location_id": request.args.get("location_id", type=int),
            "npc_ids": request.args.getlist("npc_ids", type=int),
            "topics": request.args.getlist("topics"),
            "lore_tags": request.args.getlist("lore_tags")
        }
        
        # Remove None values
        scope = {k: v for k, v in scope.items() if v}
        
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        bundle = await synthesizer.get_scene_bundle(scope)
        
        return jsonify({
            "success": True,
            "bundle": bundle,
            "cache_hit": bundle.get("from_cache", False)
        })
    except Exception as e:
        logger.error(f"Error getting scene bundle: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/invalidate-cache", methods=["POST"])
@require_login
async def invalidate_cache(user_id):
    """Invalidate scene bundle cache for a conversation."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        synthesizer = await get_synthesizer(user_id, conversation_id)
        
        # Clear the bundle cache
        await synthesizer._bundle_lock.acquire()
        try:
            synthesizer._bundle_cache.clear()
            synthesizer._cache_hits = 0
            synthesizer._cache_misses = 0
        finally:
            synthesizer._bundle_lock.release()
        
        return jsonify({
            "success": True,
            "message": "Cache invalidated successfully"
        })
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# BACKGROUND PROCESSING ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/background/process-queue", methods=["POST"])
@require_login
async def process_background_queue(user_id):
    """Process items from the background conflict queue."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    max_items = data.get("max_items", 5)
    
    try:
        synthesizer = await get_synthesizer(user_id, conversation_id)
        
        # Process background queue
        results = await synthesizer.process_background_queue(max_items)
        
        return jsonify({
            "success": True,
            "processed_count": len(results),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error processing background queue: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/background/check-limits", methods=["GET"])
@require_login
async def check_content_limits(user_id):
    """Check if content generation is allowed for a conflict."""
    conversation_id = request.args.get("conversation_id")
    conflict_id = request.args.get("conflict_id", type=int)
    content_type = request.args.get("content_type", "news")
    
    if not conversation_id or not conflict_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Check generation limits
        allowed, reason = await synthesizer.should_generate_content(content_type, conflict_id)
        
        return jsonify({
            "success": True,
            "allowed": allowed,
            "reason": reason,
            "content_type": content_type,
            "conflict_id": conflict_id
        })
    except Exception as e:
        logger.error(f"Error checking content limits: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/background/initialize", methods=["POST"])
@require_login
async def initialize_background_world(user_id):
    """Initialize the background world with grand conflicts."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        # Check governance permission
        permission = await check_conflict_permission(
            user_id=user_id,
            conversation_id=conversation_id,
            action_type="initialize_background",
            conflict_data={}
        )
        
        if not permission["approved"]:
            return jsonify({
                "success": False,
                "error": "Governance denied background initialization",
                "reasoning": permission.get("reasoning")
            }), 403
        
        # Initialize through background processor
        from logic.conflict_system.background_grand_conflicts import BackgroundConflictSubsystem
        subsystem = BackgroundConflictSubsystem(user_id, conversation_id)
        
        # Queue initialization
        subsystem.processor._processing_queue.append({
            'type': 'generate_initial_conflicts',
            'priority': 1  # High priority
        })
        
        # Process immediately
        synthesizer = await get_synthesizer(user_id, conversation_id)
        results = await synthesizer.process_background_queue(max_items=10)
        
        return jsonify({
            "success": True,
            "message": "Background world initialization queued",
            "initial_results": results
        })
    except Exception as e:
        logger.error(f"Error initializing background world: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/background/daily-flavor", methods=["GET"])
@require_login
async def get_daily_background_flavor(user_id):
    """Get daily background conflict flavor text."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        from logic.conflict_system.background_grand_conflicts import BackgroundConflictSubsystem
        subsystem = BackgroundConflictSubsystem(user_id, int(conversation_id))
        
        # Get daily update without generating new content
        update = await subsystem.daily_background_update(generate_new=False)
        
        return jsonify({
            "success": True,
            "world_tension": update.get("world_tension", 0.0),
            "background_news": update.get("news", []),
            "ambient_effects": update.get("ambient_effects", []),
            "overheard": update.get("overheard_conversation", ""),
            "optional_hook": update.get("optional_hook", "")
        })
    except Exception as e:
        logger.error(f"Error getting daily flavor: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# DAY TRANSITION & EVENT ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/day-transition", methods=["POST"])
@require_login
async def handle_day_transition(user_id):
    """Handle game day transition for conflict system."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id", "new_day"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    new_day = data["new_day"]
    
    try:
        synthesizer = await get_synthesizer(user_id, conversation_id)
        
        # Handle day transition
        result = await synthesizer.handle_day_transition(new_day)
        
        # Report to governance
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=f"conflict_system_{conversation_id}",
            action={
                "type": "day_transition",
                "new_day": new_day
            },
            result={
                "success": True,
                "updates_processed": result.get("updates_processed", 0)
            }
        )
        
        return jsonify({
            "success": True,
            "new_day": new_day,
            "transition_result": result
        })
    except Exception as e:
        logger.error(f"Error handling day transition: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/emit-event", methods=["POST"])
@require_login
async def emit_custom_event(user_id):
    """Emit a custom event to the conflict system."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id", "event_type", "payload"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        from logic.conflict_system.conflict_synthesizer import SystemEvent, EventType, SubsystemType
        
        # Map event type string to enum
        try:
            event_type = EventType[data["event_type"].upper()]
        except KeyError:
            return jsonify({"error": f"Invalid event type: {data['event_type']}"}), 400
        
        # Create event
        event = SystemEvent(
            event_id=f"custom_{datetime.now().timestamp()}",
            event_type=event_type,
            source_subsystem=SubsystemType.ORCHESTRATOR,
            payload=data["payload"],
            target_subsystems=set(SubsystemType[s.upper()] for s in data.get("target_subsystems", [])) if data.get("target_subsystems") else None,
            requires_response=data.get("requires_response", False),
            priority=data.get("priority", 5)
        )
        
        # Emit event
        synthesizer = await get_synthesizer(user_id, conversation_id)
        responses = await synthesizer.emit_event(event)
        
        # Format responses
        formatted_responses = []
        if responses:
            for response in responses:
                formatted_responses.append({
                    "subsystem": response.subsystem.value,
                    "success": response.success,
                    "data": response.data
                })
        
        return jsonify({
            "success": True,
            "event_id": event.event_id,
            "responses": formatted_responses
        })
    except Exception as e:
        logger.error(f"Error emitting custom event: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/event-history", methods=["GET"])
@require_login
async def get_event_history(user_id):
    """Get recent event history from the conflict system."""
    conversation_id = request.args.get("conversation_id")
    limit = request.args.get("limit", 50, type=int)
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get event history (last N events)
        history = synthesizer._event_history[-limit:]
        
        # Format events
        formatted_history = []
        for event in history:
            formatted_history.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "source_subsystem": event.source_subsystem.value if event.source_subsystem else None,
                "timestamp": event.timestamp.isoformat(),
                "priority": event.priority,
                "payload_summary": {k: type(v).__name__ for k, v in event.payload.items()} if event.payload else {}
            })
        
        return jsonify({
            "success": True,
            "event_count": len(formatted_history),
            "events": formatted_history
        })
    except Exception as e:
        logger.error(f"Error getting event history: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# CLEANUP ROUTE
# ===============================================================================

@conflict_bp.route("/api/conflict/cleanup", methods=["POST"])
@require_login
async def cleanup_synthesizer(user_id):
    """Release and cleanup synthesizer resources for a conversation."""
    data = await request.get_json() or {}
    
    validation_error = await validate_request_data(data, ["conversation_id"])
    if validation_error:
        return jsonify(validation_error), validation_error["status"]
    
    conversation_id = data["conversation_id"]
    
    try:
        await release_synthesizer(user_id, conversation_id)
        
        return jsonify({
            "success": True,
            "message": f"Synthesizer released for conversation {conversation_id}"
        })
    except Exception as e:
        logger.error(f"Error cleaning up synthesizer: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# MONITORING & METRICS ROUTES
# ===============================================================================

@conflict_bp.route("/api/conflict/metrics/performance", methods=["GET"])
@require_login
async def get_performance_metrics(user_id):
    """Get detailed performance metrics including P95."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Calculate P95 for various operations
        def calculate_p95(times):
            if not times:
                return 0
            sorted_times = sorted(times)
            index = int(len(sorted_times) * 0.95)
            return sorted_times[min(index, len(sorted_times) - 1)] if sorted_times else 0
        
        metrics = {
            "cache_performance": {
                "hits": synthesizer._cache_hits,
                "misses": synthesizer._cache_misses,
                "hit_ratio": synthesizer._cache_hits / max(1, synthesizer._cache_hits + synthesizer._cache_misses),
                "bundle_cache_size": len(synthesizer._bundle_cache),
                "max_cache_size": MAX_BUNDLE_CACHE
            },
            "processing_performance": {
                "bundle_fetch_p95": calculate_p95(synthesizer._performance_metrics.get('bundle_fetch_times', [])),
                "parallel_process_p95": calculate_p95(synthesizer._performance_metrics.get('parallel_process_times', [])),
                "cache_operations_total": synthesizer._performance_metrics.get('cache_operations', 0),
                "events_processed": synthesizer._performance_metrics.get('events_processed', 0),
                "timeouts": synthesizer._performance_metrics.get('timeouts_count', 0),
                "failures": synthesizer._performance_metrics.get('failures_count', 0)
            },
            "subsystem_health": {
                "timeouts_by_subsystem": dict(synthesizer._performance_metrics.get('subsystem_timeouts', {})),
                "failures_by_subsystem": dict(synthesizer._performance_metrics.get('subsystem_failures', {}))
            },
            "system_state": synthesizer._global_metrics,
            "queue_status": {
                "event_queue_size": synthesizer._event_queue.qsize(),
                "event_queue_max": MAX_EVENT_QUEUE,
                "event_history_size": len(synthesizer._event_history),
                "event_history_max": MAX_EVENT_HISTORY
            }
        }
        
        return jsonify({
            "success": True,
            "metrics": metrics
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/api/conflict/metrics/background-queue", methods=["GET"])
@require_login
async def get_background_queue_status(user_id):
    """Get status of the background processing queue."""
    conversation_id = request.args.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        synthesizer = await get_synthesizer(user_id, int(conversation_id))
        
        # Get queue status from processor
        queue_items = synthesizer.processor._processing_queue if hasattr(synthesizer, 'processor') else []
        
        # Group by type and priority
        queue_summary = {}
        for item in queue_items:
            item_type = item.get('type', 'unknown')
            priority = item.get('priority', 3)
            
            if item_type not in queue_summary:
                queue_summary[item_type] = {"count": 0, "priorities": {}}
            
            queue_summary[item_type]["count"] += 1
            queue_summary[item_type]["priorities"][priority] = \
                queue_summary[item_type]["priorities"].get(priority, 0) + 1
        
        return jsonify({
            "success": True,
            "queue_length": len(queue_items),
            "queue_summary": queue_summary,
            "last_daily_process": getattr(synthesizer.processor, '_last_daily_process', None) if hasattr(synthesizer, 'processor') else None
        })
    except Exception as e:
        logger.error(f"Error getting background queue status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===============================================================================
# DEBUG ROUTES (only in development)
# ===============================================================================

import os
if os.getenv("ENV", "development") == "development":
    
    @conflict_bp.route("/api/conflict/debug/metrics", methods=["GET"])
    @require_login
    async def get_debug_metrics(user_id):
        """Get detailed performance metrics (debug only)."""
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400

        try:
            synthesizer = await get_synthesizer(user_id, int(conversation_id))

            cache_hits = getattr(synthesizer, "_cache_hits", 0)
            cache_misses = getattr(synthesizer, "_cache_misses", 0)
            metrics = {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio": cache_hits / max(1, cache_hits + cache_misses),
                "bundle_cache_size": len(getattr(synthesizer, "_bundle_cache", {})),
                "active_conflicts": len(getattr(synthesizer, "_conflict_states", {})),
                "performance_metrics": getattr(synthesizer, "_performance_metrics", {}),
                "global_metrics": getattr(synthesizer, "_global_metrics", {}),
            }

            return jsonify({"success": True, "metrics": metrics})
        except Exception as e:
            logger.error(f"Error getting debug metrics: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @conflict_bp.route("/api/conflict/debug/subsystems", methods=["GET"])
    @require_login
    async def get_subsystem_info(user_id):
        """Get information about registered subsystems (debug only)."""
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400

        try:
            synthesizer = await get_synthesizer(user_id, int(conversation_id))

            subsystem_info = {}
            for subsystem_type, subsystem in getattr(synthesizer, "_subsystems", {}).items():
                subsystem_info[getattr(subsystem_type, "value", str(subsystem_type))] = {
                    "capabilities": list(getattr(subsystem, "capabilities", [])),
                    "dependencies": [getattr(dep, "value", str(dep)) for dep in getattr(subsystem, "dependencies", [])],
                    "event_subscriptions": [getattr(evt, "value", str(evt)) for evt in getattr(subsystem, "event_subscriptions", [])],
                }

            return jsonify({"success": True, "subsystems": subsystem_info})
        except Exception as e:
            logger.error(f"Error getting subsystem info: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @conflict_bp.route("/api/conflict/debug/subsystem/<string:subsystem_type>", methods=["GET"])
    @require_login
    async def get_specific_subsystem_status(user_id, subsystem_type):
        """Get detailed status for a specific subsystem (debug only)."""
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400

        try:
            from logic.conflict_system.conflict_synthesizer import SubsystemType

            # Validate subsystem type
            try:
                subsystem_enum = SubsystemType[subsystem_type.upper()]
            except KeyError:
                return jsonify({"error": f"Invalid subsystem type: {subsystem_type}"}), 400

            synthesizer = await get_synthesizer(user_id, int(conversation_id))

            subsystem = getattr(synthesizer, "_subsystems", {}).get(subsystem_enum)
            if not subsystem:
                return jsonify({"error": f"Subsystem {subsystem_type} not registered"}), 404

            # Health check can be async
            health = await subsystem.health_check()

            info = {
                "type": subsystem_type,
                "healthy": health.get("healthy", False),
                "capabilities": list(getattr(subsystem, "capabilities", [])),
                "dependencies": [getattr(dep, "value", str(dep)) for dep in getattr(subsystem, "dependencies", [])],
                "event_subscriptions": [getattr(evt, "value", str(evt)) for evt in getattr(subsystem, "event_subscriptions", [])],
                "health_details": health,
            }

            return jsonify({"success": True, "subsystem": info})
        except Exception as e:
            logger.error(f"Error getting subsystem status: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @conflict_bp.route("/api/conflict/debug/force-event-processing", methods=["POST"])
    @require_login
    async def force_event_processing(user_id):
        """Force immediate processing of all queued events (debug only)."""
        data = await request.get_json() or {}

        validation_error = await validate_request_data(data, ["conversation_id"])
        if validation_error:
            return jsonify(validation_error), validation_error["status"]

        conversation_id = int(data["conversation_id"])
        try:
            synthesizer = await get_synthesizer(user_id, conversation_id)

            processed = []
            # Drain the queue
            while not synthesizer._event_queue.empty():
                try:
                    event = synthesizer._event_queue.get_nowait()
                    responses = await synthesizer.emit_event(event)
                    processed.append({
                        "event_id": getattr(event, "event_id", None),
                        "event_type": getattr(getattr(event, "event_type", None), "value", None),
                        "responses_count": len(responses) if responses else 0,
                    })
                except asyncio.QueueEmpty:
                    break

            return jsonify({"success": True, "events_processed": len(processed), "events": processed})
        except Exception as e:
            logger.error(f"Error forcing event processing: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
