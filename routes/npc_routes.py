# routes/npc_routes.py

from quart import Blueprint, request, jsonify
from flasgger import swag_from
from npcs.npc_creation import NPCCreationHandler
from middleware.error_handling import create_error_response
from db.connection import get_db_connection_context

npc_bp = Blueprint('npc_bp', __name__)

@npc_bp.route('/create', methods=['POST'])
@swag_from({
    'tags': ['NPCs'],
    'summary': 'Create a new NPC',
    'description': 'Create a new NPC with specified attributes and personality',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Name of the NPC'},
                    'personality_traits': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of personality traits'
                    },
                    'background': {'type': 'string', 'description': 'NPC background story'},
                    'goals': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of NPC goals'
                    },
                    'stats': {
                        'type': 'object',
                        'properties': {
                            'intensity': {'type': 'integer', 'minimum': 0, 'maximum': 100},
                            'corruption': {'type': 'integer', 'minimum': 0, 'maximum': 100},
                            'dependency': {'type': 'integer', 'minimum': 0, 'maximum': 100}
                        }
                    }
                },
                'required': ['name', 'personality_traits', 'background']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'NPC created successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'npc_id': {'type': 'string'},
                    'message': {'type': 'string'}
                }
            }
        },
        '400': {
            'description': 'Invalid request parameters',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'error': {'type': 'string'},
                    'message': {'type': 'string'}
                }
            }
        }
    }
})
async def create_npc():
    """Create a new NPC."""
    try:
        data = request.get_json()
        handler = NPCCreationHandler()
        npc = await handler.create_npc(data)
        
        return jsonify({
            'success': True,
            'npc_id': npc.id,
            'message': f'NPC {npc.name} created successfully'
        })
    except ValueError as e:
        return create_error_response(e, status_code=400)
    except Exception as e:
        return create_error_response(e)

@npc_bp.route('/<npc_id>', methods=['GET'])
@swag_from({
    'tags': ['NPCs'],
    'summary': 'Get NPC details',
    'description': 'Retrieve detailed information about a specific NPC',
    'parameters': [
        {
            'name': 'npc_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Unique identifier of the NPC'
        }
    ],
    'responses': {
        '200': {
            'description': 'NPC details retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'npc': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'personality_traits': {'type': 'array', 'items': {'type': 'string'}},
                            'background': {'type': 'string'},
                            'goals': {'type': 'array', 'items': {'type': 'string'}},
                            'stats': {
                                'type': 'object',
                                'properties': {
                                    'intensity': {'type': 'integer'},
                                    'corruption': {'type': 'integer'},
                                    'dependency': {'type': 'integer'}
                                }
                            }
                        }
                    }
                }
            }
        },
        '404': {
            'description': 'NPC not found',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'error': {'type': 'string'},
                    'message': {'type': 'string'}
                }
            }
        }
    }
})
async def get_npc(npc_id):
    """Get NPC details by ID."""
    try:
        handler = NPCCreationHandler()
        npc = await handler.get_npc(npc_id)
        
        if not npc:
            return create_error_response("NPC not found", status_code=404)
        
        return jsonify({
            'success': True,
            'npc': npc.to_dict()
        })
    except Exception as e:
        return create_error_response(e)
