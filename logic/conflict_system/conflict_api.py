# logic/conflict_system/conflict_api.py
"""
Conflict System API Routes

This module defines the Flask routes for the conflict system API.
It integrates with the conflict system integration module to provide
a complete API for the frontend to interact with the conflict system.
"""

import logging
import json
from flask import Blueprint, request, jsonify, session
from utils.performance import timed_function
from db.connection import get_db_connection
from logic.conflict_system.conflict_integration import ConflictSystemIntegration

conflict_bp = Blueprint("conflict_bp", __name__)
logger = logging.getLogger(__name__)

@conflict_bp.route("/conflicts/active", methods=["GET"])
@timed_function(name="get_active_conflicts")
async def get_active_conflicts():
    """
    Get all active conflicts for the current user and conversation.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        active_conflicts = await conflict_integration.get_active_conflicts()
        
        return jsonify({"active_conflicts": active_conflicts})
    except Exception as e:
        logger.exception("Error getting active conflicts")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>", methods=["GET"])
@timed_function(name="get_conflict_details")
async def get_conflict_details(conflict_id):
    """
    Get detailed information about a specific conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        conflict = await conflict_integration.get_conflict_details(conflict_id)
        
        if not conflict:
            return jsonify({"error": "Conflict not found"}), 404
        
        return jsonify({"conflict": conflict})
    except Exception as e:
        logger.exception(f"Error getting conflict details for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/generate", methods=["POST"])
@timed_function(name="generate_conflict")
async def generate_conflict():
    """
    Generate a new conflict, optionally of a specific type.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_type = data.get("conflict_type")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        conflict = await conflict_integration.generate_new_conflict(conflict_type)
        
        if isinstance(conflict, dict) and "error" in conflict:
            return jsonify(conflict), 400
        
        return jsonify({"conflict": conflict, "message": "Conflict generated successfully"})
    except Exception as e:
        logger.exception("Error generating conflict")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/involvement", methods=["POST"])
@timed_function(name="set_involvement")
async def set_involvement(conflict_id):
    """
    Set the player's involvement in a conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        involvement_level = data.get("involvement_level")
        if not involvement_level:
            return jsonify({"error": "Missing involvement_level parameter"}), 400
        
        faction = data.get("faction", "neutral")
        money = data.get("money", 0)
        supplies = data.get("supplies", 0)
        influence = data.get("influence", 0)
        action = data.get("action")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.set_involvement(
            conflict_id, involvement_level, faction, money, supplies, influence, action
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Involvement set to {involvement_level} for conflict {conflict_id}",
            "conflict": result
        })
    except Exception as e:
        logger.exception(f"Error setting involvement for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/recruit", methods=["POST"])
@timed_function(name="recruit_npc")
async def recruit_npc(conflict_id):
    """
    Recruit an NPC to help with a conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        npc_id = data.get("npc_id")
        if not npc_id:
            return jsonify({"error": "Missing npc_id parameter"}), 400
        
        faction = data.get("faction")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.recruit_npc(conflict_id, npc_id, faction)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"NPC {npc_id} recruited for conflict {conflict_id}",
            "recruitment_result": result
        })
    except Exception as e:
        logger.exception(f"Error recruiting NPC for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/progress", methods=["POST"])
@timed_function(name="update_progress")
async def update_progress(conflict_id):
    """
    Update the progress of a conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        progress_increment = data.get("progress_increment")
        if progress_increment is None:
            return jsonify({"error": "Missing progress_increment parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.update_progress(conflict_id, float(progress_increment))
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        # Check if the conflict phase changed
        phase_changed = data.get("previous_phase") != result.get("phase")
        
        return jsonify({
            "message": f"Progress updated for conflict {conflict_id}",
            "conflict": result,
            "phase_changed": phase_changed
        })
    except Exception as e:
        logger.exception(f"Error updating progress for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/resolve", methods=["POST"])
@timed_function(name="resolve_conflict")
async def resolve_conflict(conflict_id):
    """
    Resolve a conflict and apply consequences.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.resolve_conflict(conflict_id)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Conflict {conflict_id} resolved with outcome: {result.get('outcome', 'unknown')}",
            "conflict": result
        })
    except Exception as e:
        logger.exception(f"Error resolving conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/daily-update", methods=["POST"])
@timed_function(name="daily_update")
async def daily_update():
    """
    Run the daily conflict update.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.run_daily_update()
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": "Daily conflict update completed",
            "update_results": result
        })
    except Exception as e:
        logger.exception("Error running daily conflict update")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/state", methods=["GET"])
@timed_function(name="get_conflict_state")
async def get_conflict_state():
    """
    Get the comprehensive state of the conflict system.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        state = await conflict_integration.get_current_state()
        
        if isinstance(state, dict) and "error" in state:
            return jsonify(state), 400
        
        return jsonify(state)
    except Exception as e:
        logger.exception("Error getting conflict system state")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/player/vitals", methods=["GET"])
@timed_function(name="get_player_vitals")
async def get_player_vitals():
    """
    Get the current player vitals.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        vitals = await conflict_integration.get_player_vitals()
        
        return jsonify(vitals)
    except Exception as e:
        logger.exception("Error getting player vitals")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/player/vitals/update", methods=["POST"])
@timed_function(name="update_player_vitals")
async def update_player_vitals():
    """
    Update player vitals based on activity type.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        activity_type = data.get("activity_type", "standard")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.update_player_vitals(activity_type)
        
        return jsonify({
            "message": f"Vitals updated for activity type: {activity_type}",
            "vitals": result
        })
    except Exception as e:
        logger.exception("Error updating player vitals")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/narrative/add-conflict", methods=["POST"])
@timed_function(name="add_conflict_to_narrative")
async def add_conflict_to_narrative():
    """
    Analyze a narrative text to determine if it should trigger a conflict.
    """
    try:
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
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.add_conflict_to_narrative(narrative_text)
        
        return jsonify(result)
    except Exception as e:
        logger.exception("Error adding conflict to narrative")
        return jsonify({"error": str(e)}), 500
