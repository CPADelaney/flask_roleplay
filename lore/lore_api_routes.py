# lore/lore_api_routes.py

"""
Lore API Routes

This module provides standardized API routes for the lore system,
consolidating all lore-related API functionality in one place.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from flask import Blueprint, request, jsonify, current_app
from routes.auth import require_login
from lore.lore_system import LoreSystem

logger = logging.getLogger(__name__)

lore_api_bp = Blueprint('lore_api', __name__)

# ----- World Lore Generation Routes -----

@lore_api_bp.route("/api/lore/generate", methods=["POST"])
@require_login
async def generate_world_lore(user_id):
    """Generate complete lore for a world/environment."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    environment_desc = data.get("environment_desc")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not environment_desc:
        return jsonify({"error": "Missing environment_desc parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Generate lore
        result = await lore_system.generate_world_lore(environment_desc)
        
        return jsonify({
            "success": True,
            "lore": result
        })
    except Exception as e:
        logger.error(f"Error generating world lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Location Lore Routes -----

@lore_api_bp.route("/api/lore/location", methods=["GET"])
@require_login
async def get_location_lore(user_id):
    """Get lore for a specific location."""
    conversation_id = request.args.get("conversation_id")
    location_name = request.args.get("location_name")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not location_name:
        return jsonify({"error": "Missing location_name parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Get location lore
        result = await lore_system.get_location_lore(location_name)
        
        return jsonify({
            "success": True,
            "location_name": location_name,
            "lore": result
        })
    except Exception as e:
        logger.error(f"Error getting location lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_api_bp.route("/api/lore/scene", methods=["GET"])
@require_login
async def get_scene_with_lore(user_id):
    """Get a scene description with integrated lore for a location."""
    conversation_id = request.args.get("conversation_id")
    location_name = request.args.get("location_name")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not location_name:
        return jsonify({"error": "Missing location_name parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Get scene with lore
        result = await lore_system.create_scene_with_lore(location_name)
        
        return jsonify({
            "success": True,
            "location_name": location_name,
            "scene": result
        })
    except Exception as e:
        logger.error(f"Error getting scene with lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- NPC Lore Integration Routes -----

@lore_api_bp.route("/api/lore/npc/knowledge", methods=["GET"])
@require_login
async def get_npc_lore_knowledge(user_id):
    """Get an NPC's knowledge of lore."""
    conversation_id = request.args.get("conversation_id")
    npc_id = request.args.get("npc_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not npc_id:
        return jsonify({"error": "Missing npc_id parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Get NPC lore knowledge
        result = await lore_system.get_npc_knowledge(int(npc_id))
        
        return jsonify({
            "success": True,
            "npc_id": npc_id,
            "knowledge": result
        })
    except Exception as e:
        logger.error(f"Error getting NPC lore knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_api_bp.route("/api/lore/npcs/integrate", methods=["POST"])
@require_login
async def integrate_lore_with_npcs(user_id):
    """Integrate lore with NPCs."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_ids = data.get("npc_ids", [])
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not npc_ids:
        return jsonify({"error": "Missing npc_ids parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Integrate lore with NPCs
        result = await lore_system.integrate_lore_with_npcs(npc_ids)
        
        return jsonify({
            "success": True,
            "integration_results": result
        })
    except Exception as e:
        logger.error(f"Error integrating lore with NPCs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Lore Update Routes -----

@lore_api_bp.route("/api/lore/update", methods=["POST"])
@require_login
async def update_lore_after_event(user_id):
    """Update lore based on a narrative event."""
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    event_description = data.get("event_description")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not event_description:
        return jsonify({"error": "Missing event_description parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Update lore after event
        result = await lore_system.update_lore_after_event(event_description)
        
        return jsonify({
            "success": True,
            "updated_lore": result
        })
    except Exception as e:
        logger.error(f"Error updating lore after event: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Lore Search Routes -----

@lore_api_bp.route("/api/lore/search", methods=["GET"])
@require_login
async def search_lore(user_id):
    """Search for lore matching a query."""
    conversation_id = request.args.get("conversation_id")
    query = request.args.get("query")
    limit = request.args.get("limit", 5)
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
        await lore_system.initialize()
        
        # Search lore
        result = await lore_system.search_lore(query, int(limit))
        
        return jsonify({
            "success": True,
            "query": query,
            "results": result,
            "count": len(result)
        })
    except Exception as e:
        logger.error(f"Error searching lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ----- Register Blueprint -----

def register_lore_api_routes(app):
    """Register the lore API routes with the Flask app."""
    app.register_blueprint(lore_api_bp)
    logger.info("Lore API routes registered") 
