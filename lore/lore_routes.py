# lore/lore_routes.py

import logging
from flask import Blueprint, request, jsonify, session

# Import Nyx governance integration functions
from nyx.integrate import (
    get_central_governance,
    generate_lore_with_governance,
    integrate_lore_with_npcs,
    enhance_context_with_lore,
    generate_scene_with_lore,
    process_universal_update_with_governance
)

lore_bp = Blueprint('lore_bp', __name__)

@lore_bp.route('/generate', methods=['POST'])
async def generate_lore():
    """Generate complete lore for a game with Nyx governance oversight."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        environment_desc = data.get('environment_desc')
        
        if not conversation_id or not environment_desc:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use Nyx governance integration
        result = await generate_lore_with_governance(
            user_id, 
            int(conversation_id), 
            environment_desc
        )
        
        if "error" in result:
            return jsonify(result), 400
        
        lore = result.get("lore", {})
        
        return jsonify({
            "status": "success",
            "message": "Lore generated successfully with Nyx governance oversight",
            "governance_approved": result.get("governance_approved", False),
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
    """Integrate lore with NPCs by giving them knowledge with Nyx governance oversight."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        npc_ids = data.get('npc_ids', [])
        
        if not conversation_id or not npc_ids:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use Nyx governance integration
        result = await integrate_lore_with_npcs(
            user_id, 
            int(conversation_id), 
            npc_ids
        )
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify({
            "status": "success",
            "message": f"Lore integrated with {len(result.get('results', {}))} NPCs",
            "governance_approved": result.get("governance_approved", False),
            "results": result.get("results", {})
        })
        
    except Exception as e:
        logging.exception("Error integrating lore with NPCs")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/npc_response', methods=['POST'])
async def npc_lore_response():
    """Get a lore-based response from an NPC with Nyx governance oversight."""
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
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission with governance system
        permission = await governance.check_action_permission(
            agent_type="npc",
            agent_id=npc_id,
            action_type="generate_lore_response",
            action_details={
                "player_input": player_input
            }
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
            
        # Use the LoreIntegrationSystem but with governance oversight
        from lore.lore_integration import LoreIntegrationSystem
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        response = await lore_system.generate_npc_lore_response(int(npc_id), player_input)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="npc",
            agent_id=npc_id,
            action={
                "type": "generate_lore_response",
                "description": f"Generated lore response to: {player_input[:50]}..."
            },
            result={
                "lore_shared": response.get("lore_shared", False),
                "knowledge_level": response.get("knowledge_level", 0)
            }
        )
        
        return jsonify(response)
        
    except Exception as e:
        logging.exception("Error getting NPC lore response")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/enhance_scene', methods=['POST'])
async def enhance_scene():
    """Enhance a scene description with lore using Nyx governance."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        location = data.get('location')
        
        if not conversation_id or not location:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use Nyx governance integration
        scene = await generate_scene_with_lore(
            user_id, 
            int(conversation_id), 
            location
        )
        
        return jsonify(scene)
        
    except Exception as e:
        logging.exception("Error enhancing scene")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/update_lore', methods=['POST'])
async def update_lore():
    """Update lore after a significant narrative event with Nyx governance oversight."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        event_description = data.get('event_description')
        
        if not conversation_id or not event_description:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use Nyx governance integration for universal updates
        updates = await process_universal_update_with_governance(
            user_id,
            int(conversation_id),
            f"Lore update triggered by: {event_description}",
            {"event_description": event_description, "type": "lore_update"}
        )
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Use the LoreIntegrationSystem with governance oversight
        from lore.lore_integration import LoreIntegrationSystem
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_integration",
            action_type="update_lore_after_narrative_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Update lore
        lore_updates = await lore_system.update_lore_after_narrative_event(event_description)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_integration",
            action={
                "type": "update_lore_after_narrative_event",
                "description": f"Updated lore after event: {event_description[:50]}..."
            },
            result=lore_updates
        )
        
        return jsonify({
            "status": "success",
            "message": "Lore updated successfully with Nyx governance oversight",
            "updates": lore_updates,
            "universal_updates": updates
        })
        
    except Exception as e:
        logging.exception("Error updating lore")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/quest_context/<int:quest_id>', methods=['GET'])
async def quest_context(quest_id):
    """Get lore context for a specific quest with Nyx governance oversight."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_integration",
            action_type="get_quest_lore_context",
            action_details={"quest_id": quest_id}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
            
        # Use LoreIntegrationSystem with governance oversight
        from lore.lore_integration import LoreIntegrationSystem
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        context = await lore_system.get_quest_lore_context(quest_id)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_integration",
            action={
                "type": "get_quest_lore_context",
                "description": f"Retrieved lore context for quest {quest_id}"
            },
            result={
                "quest_id": quest_id,
                "lore_elements": len(context.get("relevant_lore", []))
            }
        )
        
        return jsonify(context)
        
    except Exception as e:
        logging.exception("Error getting quest context")
        return jsonify({"error": str(e)}), 500

@lore_bp.route('/search', methods=['GET'])
async def search_lore():
    """Search for lore by keyword with Nyx governance oversight."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        query = request.args.get('query')
        conversation_id = request.args.get('conversation_id')
        
        if not query or not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_manager",
            action_type="search_lore",
            action_details={"query": query}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
            
        # Use LoreManager with governance oversight
        from lore.lore_manager import LoreManager
        lore_manager = LoreManager(user_id, int(conversation_id))
        
        # Initialize governance for the manager
        await lore_manager.initialize_governance()
        
        # Search for lore
        results = await lore_manager.get_relevant_lore(
            query,
            min_relevance=0.5,
            limit=20
        )
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_manager",
            action={
                "type": "search_lore",
                "description": f"Searched lore with query: {query}"
            },
            result={
                "result_count": len(results)
            }
        )
        
        return jsonify({
            "query": query,
            "results": results,
            "result_count": len(results)
        })
        
    except Exception as e:
        logging.exception("Error searching lore")
        return jsonify({"error": str(e)}), 500

def init_app(app):
    """Initialize the lore blueprint with the Flask app."""
    app.register_blueprint(lore_bp, url_prefix='/api/lore')
