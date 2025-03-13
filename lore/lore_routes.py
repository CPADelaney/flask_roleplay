# lore/lore_routes.py

import logging
from flask import Blueprint, request, jsonify, session
from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.lore_integration import LoreIntegrationSystem
from lore.lore_manager import LoreManager

lore_bp = Blueprint('lore_bp', __name__)

@lore_bp.route('/generate', methods=['POST'])
async def generate_lore():
    """Generate complete lore for a game."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        environment_desc = data.get('environment_desc')
        
        if not conversation_id or not environment_desc:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Generate complete lore
        lore = await lore_system.initialize_game_lore(environment_desc)
        
        return jsonify({
            "status": "success",
            "message": "Lore generated successfully",
            "lore_overview": {
                "world_lore_count": len(lore.get("world_lore", {})),
                "factions_count": len(lore.get("factions", [])),
                "cultural_elements_count": len(lore.get("cultural_elements", [])),
                "historical_events_count": len(lore.get("historical_events", [])),
                "locations_count": len(lore.get("locations", [])),
                "quests_count": len(lore.get("quests", []))
            }
        })
        
    except Exception as e:
        logging.exception("Error generating lore")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/integrate_npcs', methods=['POST'])
async def integrate_npcs():
    """Integrate lore with NPCs by giving them knowledge."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        npc_ids = data.get('npc_ids', [])
        
        if not conversation_id or not npc_ids:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Integrate lore with NPCs
        results = await lore_system.integrate_lore_with_npcs(npc_ids)
        
        return jsonify({
            "status": "success",
            "message": f"Lore integrated with {len(results)} NPCs",
            "results": results
        })
        
    except Exception as e:
        logging.exception("Error integrating lore with NPCs")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/npc_response', methods=['POST'])
async def npc_lore_response():
    """Get a lore-based response from an NPC."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        npc_id = data.get('npc_id')
        player_input = data.get('player_input')
        
        if not all([conversation_id, npc_id, player_input]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Get NPC response
        response = await lore_system.generate_npc_lore_response(int(npc_id), player_input)
        
        return jsonify(response)
        
    except Exception as e:
        logging.exception("Error getting NPC lore response")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/enhance_scene', methods=['POST'])
async def enhance_scene():
    """Enhance a scene description with lore."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        location = data.get('location')
        
        if not conversation_id or not location:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Generate enhanced scene description
        scene = await lore_system.generate_scene_description_with_lore(location)
        
        return jsonify(scene)
        
    except Exception as e:
        logging.exception("Error enhancing scene")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/update_lore', methods=['POST'])
async def update_lore():
    """Update lore after a significant narrative event."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        event_description = data.get('event_description')
        
        if not conversation_id or not event_description:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Update lore
        updates = await lore_system.update_lore_after_narrative_event(event_description)
        
        return jsonify({
            "status": "success",
            "message": "Lore updated successfully",
            "updates": updates
        })
        
    except Exception as e:
        logging.exception("Error updating lore")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/quest_context/<int:quest_id>', methods=['GET'])
async def quest_context(quest_id):
    """Get lore context for a specific quest."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Initialize lore system
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Get quest context
        context = await lore_system.get_quest_lore_context(quest_id)
        
        return jsonify(context)
        
    except Exception as e:
        logging.exception("Error getting quest context")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/search', methods=['GET'])
async def search_lore():
    """Search for lore by keyword."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        query = request.args.get('query')
        conversation_id = request.args.get('conversation_id')
        
        if not query or not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize lore manager
        lore_manager = LoreManager(user_id, int(conversation_id))
        
        # Search for lore
        results = await lore_manager.get_relevant_lore(
            query,
            min_relevance=0.5,
            limit=20
        )
        
        return jsonify({
            "query": query,
            "results": results
        })
        
    except Exception as e:
        logging.exception("Error searching lore")
        return jsonify({"error": str(e)}), 500

def init_app(app):
    """Initialize the lore blueprint with the Flask app."""
    app.register_blueprint(lore_bp, url_prefix='/api/lore')
