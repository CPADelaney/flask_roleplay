"""
Unified Routes for Lore System

This module provides all HTTP routes for the lore system, organized by functionality.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .dynamic_lore_generator import DynamicLoreGenerator
from .unified_data_access import (
    NPCDataAccess,
    LocationDataAccess,
    LoreDataAccess,
    ConflictDataAccess
)
from .error_handler import handle_error, LoreError
from .monitoring import track_request
from .lore_validation import validate_request

logger = logging.getLogger(__name__)

# Create blueprints for different route categories
lore_bp = Blueprint('lore', __name__, url_prefix='/lore')
npc_bp = Blueprint('npcs', __name__, url_prefix='/npcs')
location_bp = Blueprint('locations', __name__, url_prefix='/locations')
conflict_bp = Blueprint('conflicts', __name__, url_prefix='/conflicts')
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# World and General Lore Routes
@lore_bp.route('/worlds/<int:world_id>', methods=['GET'])
@track_request
async def get_world_details(world_id: int):
    """Get details for a specific world."""
    try:
        lore_system = DynamicLoreGenerator.get_instance()
        world = await lore_system.get_world_details(world_id)
        return jsonify(world)
    except Exception as e:
        return handle_error(e)

@lore_bp.route('/worlds/<int:world_id>/cultures', methods=['GET'])
@track_request
async def get_world_cultures(world_id: int):
    """Get all cultures in a world."""
    try:
        lore_system = DynamicLoreGenerator.get_instance()
        cultures = await lore_system.get_cultures(world_id)
        return jsonify(cultures)
    except Exception as e:
        return handle_error(e)

@lore_bp.route('/worlds/<int:world_id>/events', methods=['GET'])
@track_request
async def get_world_events(world_id: int):
    """Get historical events for a world."""
    time_period = request.args.get('time_period')
    try:
        lore_system = DynamicLoreGenerator.get_instance()
        events = await lore_system.get_historical_events(world_id, time_period)
        return jsonify(events)
    except Exception as e:
        return handle_error(e)

# NPC Routes
@npc_bp.route('/<int:npc_id>', methods=['GET'])
@track_request
async def get_npc_details(npc_id: int):
    """Get details for a specific NPC."""
    try:
        npc_data = NPCDataAccess()
        npc = await npc_data.get_npc_by_id(npc_id)
        if not npc:
            raise LoreError(f"NPC {npc_id} not found", status_code=404)
        return jsonify(npc)
    except Exception as e:
        return handle_error(e)

@npc_bp.route('/<int:npc_id>/relationships', methods=['GET'])
@track_request
async def get_npc_relationships(npc_id: int):
    """Get relationships for a specific NPC."""
    try:
        npc_data = NPCDataAccess()
        relationships = await npc_data.get_npc_relationships(npc_id)
        return jsonify(relationships)
    except Exception as e:
        return handle_error(e)

@npc_bp.route('/faction/<int:faction_id>', methods=['GET'])
@track_request
async def get_faction_npcs(faction_id: int):
    """Get all NPCs in a faction."""
    try:
        npc_data = NPCDataAccess()
        npcs = await npc_data.get_npcs_by_faction(faction_id)
        return jsonify(npcs)
    except Exception as e:
        return handle_error(e)

# Location Routes
@location_bp.route('/<int:location_id>', methods=['GET'])
@track_request
async def get_location_details(location_id: int):
    """Get details for a specific location."""
    try:
        location_data = LocationDataAccess()
        location = await location_data.get_location_by_id(location_id)
        if not location:
            raise LoreError(f"Location {location_id} not found", status_code=404)
        return jsonify(location)
    except Exception as e:
        return handle_error(e)

@location_bp.route('/<int:location_id>/poi', methods=['GET'])
@track_request
async def get_location_points_of_interest(location_id: int):
    """Get points of interest at a location."""
    try:
        location_data = LocationDataAccess()
        pois = await location_data.get_points_of_interest(location_id)
        return jsonify(pois)
    except Exception as e:
        return handle_error(e)

# Conflict Routes
@conflict_bp.route('/<int:conflict_id>', methods=['GET'])
@track_request
async def get_conflict_details(conflict_id: int):
    """Get details for a specific conflict."""
    try:
        conflict_data = ConflictDataAccess()
        conflict = await conflict_data.get_conflict_by_id(conflict_id)
        if not conflict:
            raise LoreError(f"Conflict {conflict_id} not found", status_code=404)
        return jsonify(conflict)
    except Exception as e:
        return handle_error(e)

@conflict_bp.route('/faction/<int:faction_id>', methods=['GET'])
@track_request
async def get_faction_conflicts(faction_id: int):
    """Get all conflicts involving a faction."""
    try:
        conflict_data = ConflictDataAccess()
        conflicts = await conflict_data.get_faction_conflicts(faction_id)
        return jsonify(conflicts)
    except Exception as e:
        return handle_error(e)

# API Routes for External Integration
@api_bp.route('/query', methods=['POST'])
@track_request
@validate_request
async def query_lore():
    """Query the lore system."""
    try:
        data = request.get_json()
        query_type = data.get('type')
        query_params = data.get('params', {})
        
        lore_system = DynamicLoreGenerator.get_instance()
        result = await lore_system.query_lore(query_type, query_params)
        return jsonify(result)
    except Exception as e:
        return handle_error(e)

@api_bp.route('/generate', methods=['POST'])
@track_request
@validate_request
async def generate_lore():
    """Generate new lore content."""
    try:
        data = request.get_json()
        generation_type = data.get('type')
        parameters = data.get('parameters', {})
        
        lore_system = DynamicLoreGenerator.get_instance()
        result = await lore_system.generate_lore(generation_type, parameters)
        return jsonify(result)
    except Exception as e:
        return handle_error(e)

@api_bp.route('/integrate', methods=['POST'])
@track_request
@validate_request
async def integrate_lore():
    """Integrate new lore with existing content."""
    try:
        data = request.get_json()
        integration_type = data.get('type')
        content = data.get('content')
        
        lore_system = DynamicLoreGenerator.get_instance()
        result = await lore_system.integrate_lore(integration_type, content)
        return jsonify(result)
    except Exception as e:
        return handle_error(e)

def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(lore_bp)
    app.register_blueprint(npc_bp)
    app.register_blueprint(location_bp)
    app.register_blueprint(conflict_bp)
    app.register_blueprint(api_bp) 