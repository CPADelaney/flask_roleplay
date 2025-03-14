# nyx/integration_api.py

"""
API module for the central governance system.

This module provides functions to access and use the central governance system
from other parts of the application, including Flask routes.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from flask import Blueprint, request, jsonify, session

from nyx.integration import (
    get_central_governance,
    reset_governance,
    process_story_beat_with_governance,
    create_new_game_with_governance,
    orchestrate_scene_with_governance,
    broadcast_event_with_governance,
    add_joint_memory_with_governance
)

from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

# Create Flask blueprint
nyx_governance_bp = Blueprint("nyx_governance", __name__)

# ----- Middleware -----

def require_login(f):
    """Decorator to require login."""
    async def decorated(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        return await f(user_id, *args, **kwargs)
    return decorated

# ----- API Routes -----

@nyx_governance_bp.route("/nyx/governance/status", methods=["GET"])
@require_login
async def get_governance_status(user_id):
    """Get the current status of the governance system."""
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        governance = await get_central_governance(user_id, int(conversation_id))
        
        narrative_status = await governance.get_narrative_status()
        
        # Count active directives for each agent type
        directive_counts = {}
        async with governance.governor.governor.lock:
            for agent_type in [AgentType.NPC, AgentType.STORY_DIRECTOR, AgentType.CONFLICT_ANALYST,
                             AgentType.NARRATIVE_CRAFTER, AgentType.RESOURCE_OPTIMIZER,
                             AgentType.RELATIONSHIP_MANAGER, AgentType.UNIVERSAL_UPDATER]:
                # Get all active directives for this agent type
                directives = await governance.governor.get_agent_directives(agent_type, "all")
                directive_counts[agent_type] = len(directives)
        
        return jsonify({
            "status": "active",
            "user_id": user_id,
            "conversation_id": conversation_id,
            "narrative_status": narrative_status,
            "directive_counts": directive_counts,
            "registered_agents": list(governance.registered_agents.keys())
        })
    except Exception as e:
        logger.error(f"Error getting governance status: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/reset", methods=["POST"])
@require_login
async def reset_governance_api(user_id):
    """Reset the governance system."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        result = await reset_governance(user_id, int(conversation_id))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error resetting governance: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/story-beat", methods=["POST"])
@require_login
async def process_story_beat(user_id):
    """Process a story beat with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    context_data = data.get("context_data", {})
    
    try:
        result = await process_story_beat_with_governance(
            user_id, int(conversation_id), context_data
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing story beat: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/new-game", methods=["POST"])
@require_login
async def create_new_game(user_id):
    """Create a new game with governance oversight."""
    data = request.get_json() or {}
    
    try:
        result = await create_new_game_with_governance(user_id, data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error creating new game: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/scene", methods=["POST"])
@require_login
async def orchestrate_scene(user_id):
    """Orchestrate a scene with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    location = data.get("location")
    if not location:
        return jsonify({"error": "Missing location parameter"}), 400
    
    player_action = data.get("player_action")
    involved_npcs = data.get("involved_npcs")
    
    try:
        result = await orchestrate_scene_with_governance(
            user_id, int(conversation_id), location, player_action, involved_npcs
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error orchestrating scene: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/event", methods=["POST"])
@require_login
async def broadcast_event(user_id):
    """Broadcast an event with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    event_type = data.get("event_type")
    if not event_type:
        return jsonify({"error": "Missing event_type parameter"}), 400
    
    event_data = data.get("event_data", {})
    
    try:
        result = await broadcast_event_with_governance(
            user_id, int(conversation_id), event_type, event_data
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error broadcasting event: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/memory", methods=["POST"])
@require_login
async def add_memory(user_id):
    """Add a joint memory with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    memory_text = data.get("memory_text")
    if not memory_text:
        return jsonify({"error": "Missing memory_text parameter"}), 400
    
    source_type = data.get("source_type", "system")
    source_id = data.get("source_id", 0)
    shared_with = data.get("shared_with", [])
    significance = data.get("significance", 5)
    tags = data.get("tags", [])
    metadata = data.get("metadata", {})
    
    try:
        memory_id = await add_joint_memory_with_governance(
            user_id, int(conversation_id), memory_text, source_type, source_id,
            shared_with, significance, tags, metadata
        )
        
        return jsonify({
            "memory_id": memory_id,
            "success": memory_id > 0
        })
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/directive", methods=["POST"])
@require_login
async def issue_directive(user_id):
    """Issue a directive with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    agent_type = data.get("agent_type")
    if not agent_type:
        return jsonify({"error": "Missing agent_type parameter"}), 400
    
    agent_id = data.get("agent_id")
    if agent_id is None:  # Allow 0 as a valid ID
        return jsonify({"error": "Missing agent_id parameter"}), 400
    
    directive_type = data.get("directive_type")
    if not directive_type:
        return jsonify({"error": "Missing directive_type parameter"}), 400
    
    directive_data = data.get("directive_data", {})
    priority = data.get("priority", DirectivePriority.MEDIUM)
    duration_minutes = data.get("duration_minutes", 30)
    scene_id = data.get("scene_id")
    
    try:
        governance = await get_central_governance(user_id, int(conversation_id))
        
        directive_id = await governance.issue_directive(
            agent_type=agent_type,
            agent_id=agent_id,
            directive_type=directive_type,
            directive_data=directive_data,
            priority=priority,
            duration_minutes=duration_minutes,
            scene_id=scene_id
        )
        
        return jsonify({
            "directive_id": directive_id,
            "success": directive_id > 0
        })
    except Exception as e:
        logger.error(f"Error issuing directive: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/directives", methods=["GET"])
@require_login
async def get_directives(user_id):
    """Get directives for an agent."""
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    agent_type = request.args.get("agent_type")
    if not agent_type:
        return jsonify({"error": "Missing agent_type parameter"}), 400
    
    agent_id = request.args.get("agent_id", "all")
    
    try:
        governance = await get_central_governance(user_id, int(conversation_id))
        
        if agent_type == AgentType.NPC:
            directives = await governance.governor.get_npc_directives(int(agent_id) if agent_id != "all" else "all")
        else:
            directives = await governance.governor.get_agent_directives(agent_type, agent_id)
        
        return jsonify({
            "directives": directives,
            "count": len(directives)
        })
    except Exception as e:
        logger.error(f"Error getting directives: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/lore/generate", methods=["POST"])
@require_login
async def generate_lore_api(user_id):
    """Generate comprehensive lore with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    environment_desc = data.get("environment_desc")
    if not environment_desc:
        return jsonify({"error": "Missing environment_desc parameter"}), 400
    
    try:
        result = await generate_lore_with_governance(
            user_id, int(conversation_id), environment_desc
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error generating lore: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/lore/integrate_npcs", methods=["POST"])
@require_login
async def integrate_lore_with_npcs_api(user_id):
    """Integrate lore with NPCs with governance oversight."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    npc_ids = data.get("npc_ids", [])
    if not npc_ids:
        return jsonify({"error": "Missing npc_ids parameter"}), 400
    
    try:
        result = await integrate_lore_with_npcs(
            user_id, int(conversation_id), npc_ids
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error integrating lore with NPCs: {e}")
        return jsonify({"error": str(e)}), 500

@nyx_governance_bp.route("/nyx/governance/lore/scene", methods=["POST"])
@require_login
async def generate_scene_with_lore_api(user_id):
    """Generate a scene description enhanced with lore."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    location = data.get("location")
    if not location:
        return jsonify({"error": "Missing location parameter"}), 400
    
    try:
        result = await generate_scene_with_lore(
            user_id, int(conversation_id), location
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error generating scene with lore: {e}")
        return jsonify({"error": str(e)}), 500
