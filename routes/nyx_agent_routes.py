from flask import Blueprint, request, jsonify
from nyx.nyx_agent import NyxAgent
from middleware.validation import validate_request
from middleware.rate_limiting import rate_limit
from middleware.error_handling import create_error_response
from logic.nyx_enhancements_integration import initialize_nyx_memory_system
from logic.nyx_memory import NyxMemoryManager

nyx_agent_bp = Blueprint('nyx_agent', __name__)

@nyx_agent_bp.route("/api/nyx/introspection", methods=["GET"])
@rate_limit(limit=10, period=60)  # 10 requests per minute
@validate_request({
    'user_id': {
        'type': 'id',
        'required': True
    },
    'conversation_id': {
        'type': 'id',
        'required': True
    }
})
async def get_nyx_introspection():
    """
    Get Nyx's introspection about its own memory and understanding.
    
    This endpoint provides insight into how Nyx perceives the conversation,
    what memories it finds significant, and its confidence in its understanding.
    
    Query Parameters:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        JSON response with Nyx's introspection
    """
    try:
        # Get sanitized data
        data = request.sanitized_data
        user_id = data["user_id"]
        conversation_id = data["conversation_id"]
        
        # Get introspection from memory system
        nyx_memory = NyxMemoryManager(user_id, conversation_id)
        introspection = await nyx_memory.generate_introspection()
        
        return jsonify({
            "success": True,
            "introspection": introspection.get("introspection", ""),
            "memory_stats": introspection.get("memory_stats", {}),
            "confidence": introspection.get("confidence", 0)
        })
    except Exception as e:
        return create_error_response(
            e,
            "Failed to retrieve Nyx introspection."
        )

@nyx_agent_bp.route("/api/nyx/generate_task", methods=["POST"])
@rate_limit(limit=10, period=60)  # 10 requests per minute
@validate_request({
    'user_id': {
        'type': 'id',
        'required': True
    },
    'conversation_id': {
        'type': 'id',
        'required': True
    },
    'npc_id': {
        'type': 'id',
        'required': True
    },
    'scenario_id': {
        'type': 'id',
        'required': True
    },
    'intensity_level': {
        'type': 'integer',
        'required': False,
        'min': 1,
        'max': 5
    }
})
async def generate_creative_task():
    """
    Generate a creative task for the current scenario.
    
    Request Body:
        user_id: User ID
        conversation_id: Conversation ID
        npc_id: ID of the NPC giving the task
        scenario_id: Current scenario ID
        intensity_level: Optional task intensity (1-5)
    """
    try:
        # Get sanitized data
        data = request.sanitized_data
        
        # Initialize context
        ctx = AgentContext(data["user_id"], data["conversation_id"])
        
        # Generate task
        task_result = await ctx.task_integration.generate_creative_task(
            ctx,
            npc_id=data["npc_id"],
            scenario_id=data["scenario_id"],
            intensity_level=data.get("intensity_level")
        )
        
        if task_result["success"]:
            return jsonify({
                "success": True,
                "task": task_result["task"]
            })
        else:
            return create_error_response(
                Exception(task_result["error"]),
                "Failed to generate creative task."
            )
            
    except Exception as e:
        return create_error_response(
            e,
            "Failed to generate creative task."
        )

@nyx_agent_bp.route("/api/nyx/recommend_activities", methods=["POST"])
@rate_limit(limit=10, period=60)  # 10 requests per minute
@validate_request({
    'user_id': {
        'type': 'id',
        'required': True
    },
    'conversation_id': {
        'type': 'id',
        'required': True
    },
    'scenario_id': {
        'type': 'id',
        'required': True
    },
    'npc_ids': {
        'type': 'list',
        'required': True,
        'schema': {
            'type': 'id'
        }
    },
    'num_recommendations': {
        'type': 'integer',
        'required': False,
        'min': 1,
        'max': 5,
        'default': 2
    }
})
async def recommend_activities():
    """
    Get activity recommendations for the current scene.
    
    Request Body:
        user_id: User ID
        conversation_id: Conversation ID
        scenario_id: Current scenario ID
        npc_ids: List of present NPC IDs
        num_recommendations: Number of recommendations to return (default 2)
    """
    try:
        # Get sanitized data
        data = request.sanitized_data
        
        # Initialize context
        ctx = AgentContext(data["user_id"], data["conversation_id"])
        
        # Get available activities
        available_activities = get_available_activities()
        
        # Get recommendations
        activity_result = await ctx.task_integration.recommend_activities(
            ctx,
            scenario_id=data["scenario_id"],
            npc_ids=data["npc_ids"],
            available_activities=available_activities,
            num_recommendations=data.get("num_recommendations", 2)
        )
        
        if activity_result["success"]:
            return jsonify({
                "success": True,
                "recommendations": activity_result["recommendations"]
            })
        else:
            return create_error_response(
                Exception(activity_result["error"]),
                "Failed to get activity recommendations."
            )
            
    except Exception as e:
        return create_error_response(
            e,
            "Failed to get activity recommendations."
        ) 