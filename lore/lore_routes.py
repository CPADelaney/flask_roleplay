# lore/lore_routes.py

"""
Lore API Routes

This module provides API routes for the lore system.
"""

import logging
from quart import Blueprint, request, jsonify, session
from typing import Dict, Any

# Import core lore system
from lore.core.lore_system import LoreSystem

# Import route utilities - these would need to be updated for Quart
from routes.auth import require_login

# Import monitoring
from lore.metrics import track_request

logger = logging.getLogger(__name__)

# Create blueprint
lore_bp = Blueprint('lore', __name__)

#---------------------------
# Helper Functions
#---------------------------

async def _get_lore_system(user_id, conversation_id):
    """Get an initialized lore system instance"""
    lore_system = LoreSystem.get_instance(user_id, int(conversation_id))
    await lore_system.initialize()
    return lore_system

#---------------------------
# Lore Generation Routes
#---------------------------

@lore_bp.route("/api/lore/generate", methods=["POST"])
@require_login
@track_request("lore_generate", "POST")
async def generate_world_lore(user_id):
    """Generate complete lore for a world/environment."""
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    environment_desc = data.get("environment_desc")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not environment_desc:
        return jsonify({"error": "Missing environment_desc parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Generate lore
        result = await lore_system.generate_complete_lore(None, environment_desc)
        
        return jsonify({
            "success": True,
            "lore": result
        })
    except Exception as e:
        logger.error(f"Error generating world lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# Location Lore Routes
#---------------------------

@lore_bp.route("/api/lore/location", methods=["GET"])
@require_login
@track_request("lore_location", "GET")
async def get_location_lore(user_id):
    """Get lore for a specific location."""
    args = request.args
    conversation_id = args.get("conversation_id")
    location_name = args.get("location_name")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not location_name:
        return jsonify({"error": "Missing location_name parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Get comprehensive location context
        result = await lore_system.get_comprehensive_location_context(None, location_name)
        
        return jsonify({
            "success": True,
            "location_name": location_name,
            "lore": result
        })
    except Exception as e:
        logger.error(f"Error getting location lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_bp.route("/api/lore/scene", methods=["GET"])
@require_login
@track_request("lore_scene", "GET")
async def get_scene_with_lore(user_id):
    """Get a scene description with integrated lore for a location."""
    args = request.args
    conversation_id = args.get("conversation_id")
    location_name = args.get("location_name")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not location_name:
        return jsonify({"error": "Missing location_name parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Generate scene description
        scene = await lore_system.generate_scene_description_with_lore(None, location_name)
        
        return jsonify(scene)
    except Exception as e:
        logger.error(f"Error getting scene with lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# NPC Lore Integration Routes
#---------------------------

@lore_bp.route("/api/lore/npc/knowledge", methods=["GET"])
@require_login
@track_request("lore_npc_knowledge", "GET")
async def get_npc_lore_knowledge(user_id):
    """Get an NPC's knowledge of lore."""
    args = request.args
    conversation_id = args.get("conversation_id")
    npc_id = args.get("npc_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not npc_id:
        return jsonify({"error": "Missing npc_id parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Get NPC lore knowledge
        result = await lore_system.get_npc_lore_knowledge(int(npc_id))
        
        return jsonify({
            "success": True,
            "npc_id": npc_id,
            "knowledge": result
        })
    except Exception as e:
        logger.error(f"Error getting NPC lore knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_bp.route("/api/lore/npcs/integrate", methods=["POST"])
@require_login
@track_request("lore_npcs_integrate", "POST")
async def integrate_lore_with_npcs_route(user_id):
    """Integrate lore with NPCs."""
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_ids = data.get("npc_ids", [])
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not npc_ids:
        return jsonify({"error": "Missing npc_ids parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Integrate lore with NPCs
        results = {}
        for npc_id in npc_ids:
            # Get NPC details (name, background, etc.)
            # This would use your NPC data access layer
            npc_details = {"cultural_background": "default", "faction_affiliations": []}
            
            # Initialize NPC knowledge
            result = await lore_system.initialize_npc_lore_knowledge(
                None,
                npc_id,
                npc_details.get("cultural_background", "unknown"),
                npc_details.get("faction_affiliations", [])
            )
            
            results[npc_id] = result
        
        return jsonify({
            "status": "success",
            "message": f"Lore integrated with {len(results)} NPCs",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error integrating lore with NPCs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# Lore Update Routes
#---------------------------

@lore_bp.route("/api/lore/update", methods=["POST"])
@require_login
@track_request("lore_update", "POST")
async def update_lore_after_event(user_id):
    """Update lore based on a narrative event."""
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    event_description = data.get("event_description")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not event_description:
        return jsonify({"error": "Missing event_description parameter"}), 400
    
    try:
        # Initialize lore system
        lore_system = await _get_lore_system(user_id, conversation_id)
        
        # Update lore
        lore_updates = await lore_system.evolve_lore_with_event(None, event_description)
        
        return jsonify({
            "status": "success",
            "message": "Lore updated successfully",
            "updates": lore_updates
        })
    except Exception as e:
        logger.error(f"Error updating lore after event: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# Blueprint Registration
#---------------------------

def register_lore_routes(app):
    """Register lore routes with the Quart app."""
    app.register_blueprint(lore_bp)
    logger.info("Lore API routes registered")
