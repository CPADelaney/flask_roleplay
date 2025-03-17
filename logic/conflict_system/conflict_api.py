# logic/conflict_system/conflict_api.py
"""
Conflict System API Routes

This module defines the Flask routes for the character-driven conflict system API.
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

@conflict_bp.route("/conflicts/<int:conflict_id>/track_story_beat", methods=["POST"])
@timed_function(name="track_story_beat")
async def track_story_beat(conflict_id):
    """
    Track a story beat for a resolution path, advancing progress.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        path_id = data.get("path_id")
        if not path_id:
            return jsonify({"error": "Missing path_id parameter"}), 400
        
        beat_description = data.get("beat_description")
        if not beat_description:
            return jsonify({"error": "Missing beat_description parameter"}), 400
        
        involved_npcs = data.get("involved_npcs", [])
        progress_value = data.get("progress_value", 5.0)
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.track_story_beat(
            conflict_id, path_id, beat_description, involved_npcs, progress_value
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Story beat tracked for conflict {conflict_id}, path {path_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error tracking story beat for conflict_id={conflict_id}")
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
        result = await conflict_integration.set_player_involvement(
            conflict_id, involvement_level, faction, money, supplies, influence, action
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Involvement set to {involvement_level} for conflict {conflict_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error setting involvement for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/manipulation_attempts", methods=["GET"])
@timed_function(name="get_manipulation_attempts")
async def get_manipulation_attempts(conflict_id):
    """
    Get manipulation attempts targeted at the player for a specific conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        attempts = await conflict_integration.get_player_manipulation_attempts(conflict_id)
        
        return jsonify({
            "conflict_id": conflict_id,
            "manipulation_attempts": attempts
        })
    except Exception as e:
        logger.exception(f"Error getting manipulation attempts for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/manipulation_attempts/<int:attempt_id>/resolve", methods=["POST"])
@timed_function(name="resolve_manipulation_attempt")
async def resolve_manipulation_attempt(attempt_id):
    """
    Resolve a manipulation attempt by the player.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        success = data.get("success")
        if success is None:
            return jsonify({"error": "Missing success parameter"}), 400
        
        player_response = data.get("player_response", "")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.resolve_manipulation_attempt(
            attempt_id, success, player_response
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Manipulation attempt {attempt_id} resolved",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error resolving manipulation attempt {attempt_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/create_manipulation", methods=["POST"])
@timed_function(name="create_manipulation_attempt")
async def create_manipulation_attempt(conflict_id):
    """
    Create a manipulation attempt by an NPC targeted at the player.
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
        
        manipulation_type = data.get("manipulation_type")
        if not manipulation_type:
            return jsonify({"error": "Missing manipulation_type parameter"}), 400
        
        content = data.get("content")
        if not content:
            return jsonify({"error": "Missing content parameter"}), 400
        
        goal = data.get("goal", {})
        leverage_used = data.get("leverage_used", {})
        intimacy_level = data.get("intimacy_level", 0)
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.create_manipulation_attempt(
            conflict_id, npc_id, manipulation_type, content, 
            goal, leverage_used, intimacy_level
        )
        
        return jsonify({
            "message": f"Manipulation attempt created for NPC {npc_id} in conflict {conflict_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error creating manipulation attempt for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/analyze_manipulation_potential", methods=["GET"])
@timed_function(name="analyze_manipulation_potential")
async def analyze_manipulation_potential(conflict_id):
    """
    Analyze manipulation potential for NPCs in a conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        npc_id = request.args.get("npc_id")
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        
        if npc_id:
            # Analyze specific NPC
            potential = await conflict_integration.analyze_manipulation_potential(int(npc_id))
            return jsonify({
                "conflict_id": conflict_id,
                "npc_id": npc_id,
                "manipulation_potential": potential
            })
        else:
            # Analyze all stakeholders
            stakeholders = await conflict_integration.get_conflict_stakeholders(conflict_id)
            
            results = []
            for stakeholder in stakeholders:
                potential = await conflict_integration.analyze_manipulation_potential(stakeholder["npc_id"])
                results.append(potential)
            
            return jsonify({
                "conflict_id": conflict_id,
                "manipulation_potentials": results
            })
    except Exception as e:
        logger.exception(f"Error analyzing manipulation potential for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/suggest_manipulation", methods=["POST"])
@timed_function(name="suggest_manipulation")
async def suggest_manipulation(conflict_id):
    """
    Suggest manipulation content for an NPC.
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
        
        manipulation_type = data.get("manipulation_type")
        if not manipulation_type:
            return jsonify({"error": "Missing manipulation_type parameter"}), 400
        
        goal = data.get("goal", {})
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.suggest_manipulation_content(
            npc_id, conflict_id, manipulation_type, goal
        )
        
        return jsonify({
            "message": f"Manipulation content suggested for NPC {npc_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error suggesting manipulation for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/<int:conflict_id>/initiate_faction_struggle", methods=["POST"])
@timed_function(name="initiate_faction_struggle")
async def initiate_faction_struggle(conflict_id):
    """
    Initiate a power struggle within a faction.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        faction_id = data.get("faction_id")
        if not faction_id:
            return jsonify({"error": "Missing faction_id parameter"}), 400
        
        challenger_npc_id = data.get("challenger_npc_id")
        if not challenger_npc_id:
            return jsonify({"error": "Missing challenger_npc_id parameter"}), 400
        
        target_npc_id = data.get("target_npc_id")
        if not target_npc_id:
            return jsonify({"error": "Missing target_npc_id parameter"}), 400
        
        prize = data.get("prize", "leadership")
        approach = data.get("approach", "subtle")
        is_public = data.get("is_public", False)
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.initiate_faction_power_struggle(
            conflict_id, faction_id, challenger_npc_id, target_npc_id,
            prize, approach, is_public
        )
        
        return jsonify({
            "message": f"Faction power struggle initiated in conflict {conflict_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error initiating faction struggle for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/faction_struggles/<int:struggle_id>/coup", methods=["POST"])
@timed_function(name="attempt_faction_coup")
async def attempt_faction_coup(struggle_id):
    """
    Attempt a coup within a faction struggle.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        approach = data.get("approach", "direct")
        supporting_npcs = data.get("supporting_npcs", [])
        resources_committed = data.get("resources_committed", {})
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        result = await conflict_integration.attempt_faction_coup(
            struggle_id, approach, supporting_npcs, resources_committed
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "message": f"Faction coup attempted for struggle {struggle_id}",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error attempting faction coup for struggle_id={struggle_id}")
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
            "message": f"Conflict {conflict_id} resolved",
            "result": result
        })
    except Exception as e:
        logger.exception(f"Error resolving conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/stakeholders/<int:conflict_id>", methods=["GET"])
@timed_function(name="get_conflict_stakeholders")
async def get_conflict_stakeholders(conflict_id):
    """
    Get all stakeholders in a specific conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        stakeholders = await conflict_integration.get_conflict_stakeholders(conflict_id)
        
        return jsonify({
            "conflict_id": conflict_id,
            "stakeholders": stakeholders
        })
    except Exception as e:
        logger.exception(f"Error getting stakeholders for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500

@conflict_bp.route("/conflicts/resolution_paths/<int:conflict_id>", methods=["GET"])
@timed_function(name="get_resolution_paths")
async def get_resolution_paths(conflict_id):
    """
    Get all resolution paths for a specific conflict.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_integration = ConflictSystemIntegration(user_id, int(conversation_id))
        paths = await conflict_integration.get_resolution_paths(conflict_id)
        
        return jsonify({
            "conflict_id": conflict_id,
            "resolution_paths": paths
        })
    except Exception as e:
        logger.exception(f"Error getting resolution paths for conflict_id={conflict_id}")
        return jsonify({"error": str(e)}), 500
