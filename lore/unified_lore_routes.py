"""
Unified Lore API Routes

This module provides a comprehensive set of API routes for the lore system,
consolidating all lore-related API functionality in one place.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, session, current_app

# Import route utilities
from routes.auth import require_login

# Import core lore system
from lore.lore_system import LoreSystem
from .data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess

# Import Nyx governance integration functions
from nyx.integrate import (
    get_central_governance,
    generate_lore_with_governance,
    integrate_lore_with_npcs,
    enhance_context_with_lore,
    generate_scene_with_lore,
    process_universal_update_with_governance
)

# Import for enhanced lore
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

# Import for integration
from lore.lore_integration import LoreIntegrationSystem
from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.setting_analyzer import SettingAnalyzer
from lore.lore_manager import LoreManager

# Import monitoring
from lore.unified_monitoring import track_request, metrics_manager

logger = logging.getLogger(__name__)

# Create unified blueprint
lore_unified_bp = Blueprint('lore_unified', __name__)

#---------------------------
# Helper Functions
#---------------------------

async def _log_route_call(route_name: str, user_id: int, conversation_id: int, data: Dict = None):
    """Log route call with contextual information"""
    logger.info(f"Route call: {route_name} - User: {user_id}, Conversation: {conversation_id}")
    metrics_manager.record_request("POST", route_name, 200, 0)  # We'll update duration later

async def _get_governance(user_id: int, conversation_id: int):
    """Get governance instance with error handling"""
    try:
        return await get_central_governance(user_id, conversation_id)
    except Exception as e:
        logger.error(f"Error getting governance: {e}")
        raise

async def _check_permission(governance, agent_type: str, agent_id: str, action_type: str, action_details: Dict):
    """Check permission with governance system"""
    try:
        return await governance.check_action_permission(
            agent_type=agent_type,
            agent_id=agent_id,
            action_type=action_type,
            action_details=action_details
        )
    except Exception as e:
        logger.error(f"Error checking permission: {e}")
        return {"approved": False, "reasoning": f"Error checking permission: {str(e)}"}

async def _report_action(governance, agent_type: str, agent_id: str, action: Dict, result: Dict):
    """Report action to governance system"""
    try:
        return await governance.process_agent_action_report(
            agent_type=agent_type,
            agent_id=agent_id,
            action=action,
            result=result
        )
    except Exception as e:
        logger.error(f"Error reporting action: {e}")
        return None

#---------------------------
# Standard Lore Generation Routes
#---------------------------

@lore_unified_bp.route("/api/lore/generate", methods=["POST"])
@require_login
@track_request("lore_generate", "POST")
async def generate_world_lore(user_id):
    """Generate complete lore for a world/environment."""
    await _log_route_call("generate_world_lore", user_id, request.json.get("conversation_id"))
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

@lore_unified_bp.route("/api/lore/generate_with_governance", methods=["POST"])
@require_login
@track_request("lore_generate_with_governance", "POST")
async def generate_lore_with_gov(user_id):
    """Generate complete lore for a game with Nyx governance oversight."""
    await _log_route_call("generate_lore_with_gov", user_id, request.json.get("conversation_id"))
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

#---------------------------
# Location Lore Routes
#---------------------------

@lore_unified_bp.route("/api/lore/location", methods=["GET"])
@require_login
@track_request("lore_location", "GET")
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

@lore_unified_bp.route("/api/lore/scene", methods=["GET"])
@require_login
@track_request("lore_scene", "GET")
async def get_scene_with_lore(user_id):
    """Get a scene description with integrated lore for a location."""
    conversation_id = request.args.get("conversation_id")
    location_name = request.args.get("location_name")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not location_name:
        return jsonify({"error": "Missing location_name parameter"}), 400
    
    try:
        # Use Nyx governance integration
        scene = await generate_scene_with_lore(
            user_id, 
            int(conversation_id), 
            location_name
        )
        
        return jsonify(scene)
    except Exception as e:
        logger.error(f"Error getting scene with lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# NPC Lore Integration Routes
#---------------------------

@lore_unified_bp.route("/api/lore/npc/knowledge", methods=["GET"])
@require_login
@track_request("lore_npc_knowledge", "GET")
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

@lore_unified_bp.route("/api/lore/npcs/integrate", methods=["POST"])
@require_login
@track_request("lore_npcs_integrate", "POST")
async def integrate_lore_with_npcs_route(user_id):
    """Integrate lore with NPCs."""
    await _log_route_call("integrate_lore_with_npcs", user_id, request.json.get("conversation_id"))
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_ids = data.get("npc_ids", [])
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not npc_ids:
        return jsonify({"error": "Missing npc_ids parameter"}), 400
    
    try:
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
        logger.error(f"Error integrating lore with NPCs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_unified_bp.route("/api/lore/npc_response", methods=["POST"])
@require_login
@track_request("lore_npc_response", "POST")
async def npc_lore_response(user_id):
    """Get a lore-based response from an NPC with Nyx governance oversight."""
    await _log_route_call("npc_lore_response", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    npc_id = data.get('npc_id')
    player_input = data.get('player_input')
    
    if not all([conversation_id, npc_id, player_input]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Get governance
    governance = await _get_governance(user_id, int(conversation_id))
    
    # Check permission
    permission = await _check_permission(
        governance,
        agent_type="npc",
        agent_id=npc_id,
        action_type="generate_lore_response",
        action_details={"player_input": player_input}
    )
    
    if not permission["approved"]:
        return jsonify({
            "error": permission.get("reasoning", "Not approved by governance"),
            "approved": False
        }), 403
        
    # Use the LoreIntegrationSystem but with governance oversight
    lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
    response = await lore_system.generate_npc_lore_response(int(npc_id), player_input)
    
    # Report the action
    await _report_action(
        governance,
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

#---------------------------
# Lore Update Routes
#---------------------------

@lore_unified_bp.route("/api/lore/update", methods=["POST"])
@require_login
@track_request("lore_update", "POST")
async def update_lore_after_event(user_id):
    """Update lore based on a narrative event."""
    await _log_route_call("update_lore_after_event", user_id, request.json.get("conversation_id"))
    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    event_description = data.get("event_description")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
        
    if not event_description:
        return jsonify({"error": "Missing event_description parameter"}), 400
    
    try:
        # Use Nyx governance integration for universal updates
        updates = await process_universal_update_with_governance(
            user_id,
            int(conversation_id),
            f"Lore update triggered by: {event_description}",
            {"event_description": event_description, "type": "lore_update"}
        )
        
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Use the LoreIntegrationSystem with governance oversight
        lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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
        logger.error(f"Error updating lore after event: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

#---------------------------
# Lore Search Routes
#---------------------------

@lore_unified_bp.route("/api/lore/search", methods=["GET"])
@require_login
@track_request("lore_search", "GET")
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
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        lore_manager = LoreManager(user_id, int(conversation_id))
        
        # Initialize governance for the manager
        await lore_manager.initialize_governance()
        
        # Search for lore
        results = await lore_manager.get_relevant_lore(
            query,
            min_relevance=0.5,
            limit=int(limit)
        )
        
        # Report the action
        await _report_action(
            governance,
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
        logger.error(f"Error searching lore: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@lore_unified_bp.route("/api/lore/quest_context/<int:quest_id>", methods=["GET"])
@require_login
@track_request("lore_quest_context", "GET")
async def quest_context(user_id, quest_id):
    """Get lore context for a specific quest with Nyx governance oversight."""
    conversation_id = request.args.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    # Get governance
    governance = await _get_governance(user_id, int(conversation_id))
    
    # Check permission
    permission = await _check_permission(
        governance,
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
    lore_system = LoreIntegrationSystem(user_id, int(conversation_id))
    context = await lore_system.get_quest_lore_context(quest_id)
    
    # Report the action
    await _report_action(
        governance,
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

#---------------------------
# Enhanced Lore Routes
#---------------------------

@lore_unified_bp.route('/api/enhanced_lore/initialize', methods=['POST'])
@require_login
@track_request("enhanced_lore_initialize", "POST")
async def initialize_enhanced_lore(user_id):
    """Initialize all enhanced lore systems."""
    await _log_route_call("initialize_enhanced_lore", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
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

@lore_unified_bp.route('/api/enhanced_lore/generate', methods=['POST'])
@require_login
@track_request("enhanced_lore_generate", "POST")
async def generate_enhanced_lore(user_id):
    """Generate initial enhanced lore for a new game."""
    await _log_route_call("generate_enhanced_lore", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
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

@lore_unified_bp.route('/api/enhanced_lore/faith/generate_pantheon', methods=['POST'])
@require_login
@track_request("enhanced_lore_generate_pantheon", "POST")
async def generate_pantheon(user_id):
    """Generate a pantheon for the world."""
    await _log_route_call("generate_pantheon", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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

@lore_unified_bp.route('/api/enhanced_lore/evolve', methods=['POST'])
@require_login
@track_request("enhanced_lore_evolve", "POST")
async def evolve_world(user_id):
    """Evolve the world over time."""
    await _log_route_call("evolve_world", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    days_passed = data.get('days_passed', 30)
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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

@lore_unified_bp.route('/api/enhanced_lore/emergent/generate_event', methods=['POST'])
@require_login
@track_request("enhanced_lore_generate_event", "POST")
async def generate_emergent_event(user_id):
    """Generate an emergent world event."""
    await _log_route_call("generate_emergent_event", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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

@lore_unified_bp.route('/api/enhanced_lore/expand/generate_faction', methods=['POST'])
@require_login
@track_request("enhanced_lore_generate_faction", "POST")
async def generate_additional_faction(user_id):
    """Generate an additional faction."""
    await _log_route_call("generate_additional_faction", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    faction_type = data.get('faction_type')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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

@lore_unified_bp.route('/api/enhanced_lore/expand/generate_locations', methods=['POST'])
@require_login
@track_request("enhanced_lore_generate_locations", "POST")
async def generate_additional_locations(user_id):
    """Generate additional locations."""
    await _log_route_call("generate_additional_locations", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    location_types = data.get('location_types')
    count = data.get('count', 3)
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Get governance
        governance = await _get_governance(user_id, int(conversation_id))
        
        # Check permission
        permission = await _check_permission(
            governance,
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
        await _report_action(
            governance,
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

#---------------------------
# Setting Analysis Routes
#---------------------------

@lore_unified_bp.route('/api/lore/setting/analyze', methods=['POST'])
@require_login
@track_request("lore_setting_analyze", "POST")
async def analyze_setting(user_id):
    """Analyze setting data to generate insights."""
    await _log_route_call("analyze_setting", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Create setting analyzer
        analyzer = SettingAnalyzer(user_id, int(conversation_id))
        await analyzer.initialize_governance()
        
        # Analyze setting
        result = await analyzer.analyze_setting_demographics(None)
        
        return jsonify({
            "status": "success",
            "setting": result
        })
        
    except Exception as e:
        logging.exception("Error analyzing setting")
        return jsonify({"error": str(e)}), 500

@lore_unified_bp.route('/api/lore/setting/generate_organizations', methods=['POST'])
@require_login
@track_request("lore_setting_generate_organizations", "POST")
async def generate_organizations(user_id):
    """Generate organizations based on setting analysis."""
    await _log_route_call("generate_organizations", user_id, request.json.get("conversation_id"))
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Create setting analyzer
        analyzer = SettingAnalyzer(user_id, int(conversation_id))
        await analyzer.initialize_governance()
        
        # Generate organizations
        result = await analyzer.generate_organizations(None)
        
        return jsonify({
            "status": "success",
            "organizations": result.get("organizations", {}),
            "organization_count": result.get("organization_count", 0)
        })
        
    except Exception as e:
        logging.exception("Error generating organizations")
        return jsonify({"error": str(e)}), 500

#---------------------------
# Blueprint Registration
#---------------------------

def init_app(app):
    """Initialize all lore blueprints with the Flask app."""
    app.register_blueprint(lore_unified_bp)
    logger.info("Unified Lore API routes registered")
