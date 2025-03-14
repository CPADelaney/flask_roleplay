# lore/enhanced_lore_routes.py

import logging
from flask import Blueprint, request, jsonify, session

# Import Nyx governance integration functions
from nyx.integrate import get_central_governance

# Import enhanced lore systems
from lore.enhanced_lore import (
    FaithSystem,
    LoreEvolutionSystem,
    EmergentLoreSystem,
    LoreExpansionSystem,
    register_all_enhanced_lore_systems,
    initialize_enhanced_lore_tables,
    generate_initial_enhanced_lore,
    evolve_world_over_time
)

enhanced_lore_bp = Blueprint('enhanced_lore_bp', __name__)

@enhanced_lore_bp.route('/initialize', methods=['POST'])
async def initialize_enhanced_lore():
    """Initialize all enhanced lore systems."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize database tables
        tables_initialized = await initialize_enhanced_lore_tables(user_id, int(conversation_id))
        
        # Register with governance
        registration_results = await register_all_enhanced_lore_systems(user_id, int(conversation_id))
        
        return jsonify({
            "status": "success",
            "message": "Enhanced lore systems initialized",
            "tables_initialized": tables_initialized,
            "registration_results": registration_results
        })
        
    except Exception as e:
        logging.exception("Error initializing enhanced lore")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/generate', methods=['POST'])
async def generate_enhanced_lore():
    """Generate initial enhanced lore for a new game."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Generate initial enhanced lore
        lore_results = await generate_initial_enhanced_lore(user_id, int(conversation_id))
        
        return jsonify({
            "status": "success",
            "message": "Enhanced lore generated",
            "lore_results": lore_results
        })
        
    except Exception as e:
        logging.exception("Error generating enhanced lore")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/faith/generate_pantheon', methods=['POST'])
async def generate_pantheon():
    """Generate a pantheon for the world."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="faith_system",
            action_type="generate_pantheon",
            action_details={}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Create faith system and generate pantheon
        faith_system = FaithSystem(user_id, int(conversation_id))
        await faith_system.initialize_governance()
        
        # Create run context
        from agents.run_context import RunContextWrapper
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": int(conversation_id)
        })
        
        pantheon_data = await faith_system.generate_pantheon(run_ctx)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="faith_system",
            action={
                "type": "generate_pantheon",
                "description": "Generated a complete pantheon"
            },
            result={
                "pantheon_name": pantheon_data.get("pantheon", {}).get("name", "Unknown"),
                "deity_count": len(pantheon_data.get("deities", []))
            }
        )
        
        return jsonify(pantheon_data)
        
    except Exception as e:
        logging.exception("Error generating pantheon")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/evolve', methods=['POST'])
async def evolve_world():
    """Evolve the world over time."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        days_passed = data.get('days_passed', 30)
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_evolution",
            action_type="mature_lore_over_time",
            action_details={"days_passed": days_passed}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Evolve the world
        evolution_results = await evolve_world_over_time(
            user_id, 
            int(conversation_id), 
            days_passed
        )
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_evolution",
            action={
                "type": "evolve_world",
                "description": f"Evolved world over {days_passed} days"
            },
            result={
                "days_passed": days_passed,
                "events_generated": len(evolution_results.get("emergent_events", []))
            }
        )
        
        return jsonify({
            "status": "success",
            "message": f"World evolved over {days_passed} days",
            "evolution_results": evolution_results
        })
        
    except Exception as e:
        logging.exception("Error evolving world")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/emergent/generate_event', methods=['POST'])
async def generate_emergent_event():
    """Generate an emergent world event."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="emergent_lore",
            action_type="generate_emergent_event",
            action_details={}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Generate emergent event
        emergent_lore = EmergentLoreSystem(user_id, int(conversation_id))
        await emergent_lore.initialize_governance()
        
        # Create run context
        from agents.run_context import RunContextWrapper
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": int(conversation_id)
        })
        
        event_data = await emergent_lore.generate_emergent_event(run_ctx)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="emergent_lore",
            action={
                "type": "generate_emergent_event",
                "description": "Generated emergent world event"
            },
            result={
                "event_name": event_data.get("event_name", "Unknown Event"),
                "event_type": event_data.get("event_type", "unknown")
            }
        )
        
        return jsonify(event_data)
        
    except Exception as e:
        logging.exception("Error generating emergent event")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/expand/generate_faction', methods=['POST'])
async def generate_additional_faction():
    """Generate an additional faction."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        faction_type = data.get('faction_type')
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_expansion",
            action_type="generate_additional_faction",
            action_details={"faction_type": faction_type}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Generate additional faction
        lore_expansion = LoreExpansionSystem(user_id, int(conversation_id))
        await lore_expansion.initialize_governance()
        
        # Create run context
        from agents.run_context import RunContextWrapper
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": int(conversation_id)
        })
        
        faction_data = await lore_expansion.generate_additional_faction(run_ctx, faction_type)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_expansion",
            action={
                "type": "generate_additional_faction",
                "description": f"Generated additional faction of type: {faction_type}"
            },
            result={
                "faction_name": faction_data.get("name", "Unknown Faction"),
                "faction_type": faction_data.get("type", "unknown")
            }
        )
        
        return jsonify(faction_data)
        
    except Exception as e:
        logging.exception("Error generating additional faction")
        return jsonify({"error": str(e)}), 500

@enhanced_lore_bp.route('/expand/generate_locations', methods=['POST'])
async def generate_additional_locations():
    """Generate additional locations."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        location_types = data.get('location_types')
        count = data.get('count', 3)
        
        if not conversation_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get governance
        governance = await get_central_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await governance.check_action_permission(
            agent_type="narrative_crafter",
            agent_id="lore_expansion",
            action_type="generate_additional_locations",
            action_details={"location_types": location_types, "count": count}
        )
        
        if not permission["approved"]:
            return jsonify({
                "error": permission.get("reasoning", "Not approved by governance"),
                "approved": False
            }), 403
        
        # Generate additional locations
        lore_expansion = LoreExpansionSystem(user_id, int(conversation_id))
        await lore_expansion.initialize_governance()
        
        # Create run context
        from agents.run_context import RunContextWrapper
        run_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": int(conversation_id)
        })
        
        locations_data = await lore_expansion.generate_additional_locations(run_ctx, location_types, count)
        
        # Report the action
        await governance.process_agent_action_report(
            agent_type="narrative_crafter",
            agent_id="lore_expansion",
            action={
                "type": "generate_additional_locations",
                "description": f"Generated {len(locations_data)} additional locations"
            },
            result={
                "location_count": len(locations_data),
                "location_types": [loc.get("type", "unknown") for loc in locations_data]
            }
        )
        
        return jsonify({
            "status": "success",
            "locations": locations_data
        })
        
    except Exception as e:
        logging.exception("Error generating additional locations")
        return jsonify({"error": str(e)}), 500

def init_app(app):
    """Initialize the enhanced lore blueprint with the Flask app."""
    app.register_blueprint(enhanced_lore_bp, url_prefix='/api/enhanced_lore')

# Add this to main app.py:
# from lore.enhanced_lore_routes import init_app as init_enhanced_lore
# init_enhanced_lore(app)
