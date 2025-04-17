# routes/npc_learning_routes.py

"""
NPC Learning Routes

This module provides API endpoints for interacting with the NPC learning and adaptation system.
It allows monitoring and managing how NPCs learn from player interactions and adapt their behavior.
"""

import logging
from quart import Blueprint, request, jsonify, render_template
import json
import asyncio
import time
import random

from npcs.npc_learning_adaptation import NPCLearningManager, NPCLearningAdaptation
from npcs.npc_handler import NPCHandler
from db.connection import get_db_connection_context
from middleware.validation import validate_request
from middleware.rate_limiting import rate_limit
from middleware.error_handling import create_error_response, generate_error_id

# Create blueprint
npc_learning_bp = Blueprint("npc_learning_bp", __name__)

@npc_learning_bp.route("/npc/learning/monitor")
async def npc_learning_monitor():
    """
    Render the NPC learning monitoring UI.
    """
    return render_template("npc_learning_monitor.html")

@npc_learning_bp.route("/api/npc/learning/status/<string:npc_id>", methods=["GET"])
@rate_limit(limit=30, period=60)  # 30 requests per minute
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
async def get_npc_learning_status(npc_id):
    """
    Get the current learning status for an NPC.
    This includes intensity level and other adaptable stats.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    
    try:
        # Get NPC details
        async with get_db_connection_context() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT npc_name, dominance, cruelty, intensity, aggression, manipulativeness
                    FROM NPCStats 
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, user_id, conversation_id))
                row = await cursor.fetchone()
        
        if not row:
            return create_error_response(
                f"NPC {npc_id} not found", 
                "The requested NPC was not found.", 
                404
            )
        
        npc_name, dominance, cruelty, intensity, aggression, manipulativeness = row
        
        # Get learning system
        learning_system = NPCLearningAdaptation(user_id, conversation_id, npc_id)
        await learning_system.initialize()
        
        # Get recent memories that influenced learning
        memory_system = await learning_system.memory_system
        memories = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            tags=["learning", "adaptation"],
            limit=10
        )
        
        learning_memories = []
        if memories and "memories" in memories:
            learning_memories = memories["memories"]
        
        return jsonify({
            "success": True,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "stats": {
                "intensity": intensity,
                "dominance": dominance,
                "cruelty": cruelty,
                "aggression": aggression,
                "manipulativeness": manipulativeness
            },
            "learning_memories": learning_memories,
            "status": "active"
        })
    
    except Exception as e:
        return create_error_response(
            e, 
            "An error occurred while retrieving NPC learning status."
        )

@npc_learning_bp.route("/api/npc/learning/process/<string:npc_id>", methods=["POST"])
@rate_limit(limit=5, period=60)  # 5 requests per minute
@validate_request({
    'user_id': {
        'type': 'id',
        'required': True
    },
    'conversation_id': {
        'type': 'id',
        'required': True
    },
    'force_update': {
        'type': 'boolean',
        'required': False
    }
})
async def process_npc_learning(npc_id):
    """
    Manually trigger learning cycle for an NPC.
    This processes recent memories and relationship changes.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    force_update = data.get("force_update", False)
    
    try:
        # Get the NPC handler
        handler = NPCHandler(user_id, conversation_id)
        
        # Process the learning cycle
        results = await handler.process_npc_learning_cycle(npc_id, force_update=force_update)
        
        if not results.get("success", False):
            return create_error_response(
                results.get("error", "Unknown error"),
                "Failed to process learning cycle",
                500
            )
        
        return jsonify({
            "success": True,
            "npc_id": npc_id,
            "process_results": results,
            "message": "Learning cycle completed successfully"
        })
    
    except Exception as e:
        return create_error_response(
            e,
            "An error occurred while processing NPC learning."
        )

@npc_learning_bp.route("/api/npc/learning/trigger/<string:npc_id>", methods=["POST"])
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
    'trigger_type': {
        'type': 'string',
        'required': True,
        'max_length': 50
    },
    'trigger_details': {
        'type': 'object',
        'required': False
    }
})
async def trigger_learning_event(npc_id):
    """
    Trigger a specific learning event for an NPC.
    This allows manual creation of learning experiences.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    trigger_type = data["trigger_type"]
    trigger_details = data.get("trigger_details", {})
    
    try:
        # Create learning system for this NPC
        learning_system = NPCLearningAdaptation(user_id, conversation_id, npc_id)
        await learning_system.initialize()
        
        # Trigger the learning event
        result = await learning_system.respond_to_trigger(
            trigger_type=trigger_type,
            trigger_details=trigger_details
        )
        
        return jsonify({
            "success": True,
            "npc_id": npc_id,
            "trigger_type": trigger_type,
            "result": result,
            "message": "Learning trigger processed successfully"
        })
    
    except Exception as e:
        return create_error_response(
            e,
            "An error occurred while triggering the learning event."
        )

@npc_learning_bp.route("/api/npc/learning/batch-process", methods=["POST"])
@rate_limit(limit=3, period=60)  # 3 requests per minute
@validate_request({
    'user_id': {
        'type': 'id',
        'required': True
    },
    'conversation_id': {
        'type': 'id',
        'required': True
    },
    'npc_ids': {
        'type': 'array',
        'items': {
            'type': 'id'
        },
        'required': True
    },
    'force_update': {
        'type': 'boolean',
        'required': False
    }
})
async def batch_process_npc_learning():
    """
    Process learning for multiple NPCs at once.
    Supports detailed error reporting and validation.
    
    Request JSON:
        user_id (string): Required - The user ID
        conversation_id (string): Required - The conversation ID
        npc_ids (list): Required - List of NPC IDs to process
        force_update (bool): Optional - Force update even if recent processing occurred (default: false)
    
    Returns:
        A JSON object with processing results and error details if any
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    npc_ids = data["npc_ids"]
    force_update = data.get("force_update", False)
    
    # Check for empty NPC list
    if not npc_ids:
        return create_error_response(
            "No NPCs provided for processing",
            "Please provide a list of NPC IDs to process.",
            400
        )
    
    # Limit number of NPCs to process at once
    if len(npc_ids) > 50:
        return create_error_response(
            "Too many NPCs requested for batch processing",
            "Maximum of 50 NPCs can be processed in a single batch request",
            400
        )
    
    try:
        # Verify NPCs exist and belong to the user/conversation
        valid_npc_ids = []
        
        async with get_db_connection_context() as conn:
            # Get valid NPCs
            async with conn.cursor() as cursor:
                # Create placeholders for the IN clause
                placeholder = ', '.join(['%s'] * len(npc_ids))
                query = f"""
                    SELECT npc_id 
                    FROM NPCStats 
                    WHERE user_id = %s 
                    AND conversation_id = %s 
                    AND npc_id IN ({placeholder})
                """
                params = [user_id, conversation_id] + npc_ids
                await cursor.execute(query, tuple(params))
                
                rows = await cursor.fetchall()
                valid_npc_ids = [row[0] for row in rows]
        
        # Check if any NPCs were invalid
        invalid_npc_ids = set(npc_ids) - set(valid_npc_ids)
        if invalid_npc_ids:
            return create_error_response(
                "Some NPC IDs are invalid or do not belong to this user/conversation",
                f"The following NPC IDs are invalid: {', '.join(map(str, invalid_npc_ids))}",
                400
            )
        
        # Create learning manager with retry logic
        max_retries = 3
        retry_count = 0
        learning_manager = None
        
        while retry_count < max_retries:
            try:
                learning_manager = NPCLearningManager(user_id, conversation_id)
                await learning_manager.initialize()
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to initialize learning manager after {max_retries} attempts: {str(e)}")
                logging.warning(f"Retry {retry_count}/{max_retries} initializing learning manager: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
        
        # Run adaptation cycle for all requested NPCs with performance tracking
        start_time = time.time()
        results = await learning_manager.run_regular_adaptation_cycle(
            valid_npc_ids, 
            force_update=force_update
        )
        end_time = time.time()
        
        # Extract success and failure counts
        success_count = sum(1 for result in results if result.get("success", False))
        failed_count = len(results) - success_count
        processing_time = round(end_time - start_time, 2)
        
        # Prepare detailed response
        response = {
            "success": True,
            "batch_processed": True,
            "npc_count": len(valid_npc_ids),
            "success_count": success_count,
            "failed_count": failed_count,
            "processing_time_seconds": processing_time,
            "results": results,
            "message": f"Batch processing completed: {success_count} succeeded, {failed_count} failed"
        }
        
        # Log the results
        logging.info(f"Batch NPC learning processed: {success_count}/{len(valid_npc_ids)} NPCs successful in {processing_time}s")
        
        return jsonify(response)
    
    except Exception as e:
        return create_error_response(
            e,
            "An internal error occurred during batch processing. Check server logs."
        )

@npc_learning_bp.route("/npcs/api/get-all", methods=["GET"])
@rate_limit(limit=20, period=60)  # 20 requests per minute
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
async def get_all_npcs():
    """
    Get all NPCs for a specific user and conversation.
    
    Returns:
        JSON response with NPC data.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    
    try:
        npcs = []
        
        async with get_db_connection_context() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT n.npc_id, n.npc_name, s.dominance, s.cruelty, s.intensity
                    FROM npcs n
                    JOIN npc_stats s ON n.npc_id = s.npc_id
                    WHERE n.user_id = %s AND n.conversation_id = %s
                    ORDER BY n.npc_name
                    """,
                    (user_id, conversation_id)
                )
                
                rows = await cursor.fetchall()
                
                for row in rows:
                    npcs.append({
                        "npc_id": row[0],
                        "npc_name": row[1],
                        "dominance": row[2],
                        "cruelty": row[3],
                        "intensity": row[4]
                    })
        
        return jsonify({"success": True, "npcs": npcs})
    except Exception as e:
        return create_error_response(e, "Failed to retrieve NPCs")

@npc_learning_bp.route("/api/npc/beliefs/process-event", methods=["POST"])
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
    'event_text': {
        'type': 'string',
        'required': True,
        'max_length': 1000
    },
    'event_type': {
        'type': 'string',
        'required': True,
        'max_length': 50
    },
    'npc_ids': {
        'type': 'array',
        'items': {
            'type': 'id'
        },
        'required': True
    },
    'factuality': {
        'type': 'number',
        'min': 0.0,
        'max': 1.0,
        'required': False
    },
    'importance': {
        'type': 'number',
        'min': 0.0,
        'max': 1.0,
        'required': False
    }
})
async def process_event_for_beliefs_api():
    """
    Process a game event to generate beliefs for multiple NPCs.
    
    This endpoint allows NPCs to form subjective interpretations of the same event,
    potentially leading to different beliefs based on their personalities and backgrounds.
    
    Returns:
        JSON response with the beliefs formed for each NPC.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    event_text = data["event_text"]
    event_type = data["event_type"]
    npc_ids = data["npc_ids"]
    factuality = data.get("factuality", 1.0)
    importance = data.get("importance", 0.5)
    
    # Limit number of NPCs to process at once
    if len(npc_ids) > 10:
        return create_error_response(
            "Too many NPCs requested",
            "Maximum of 10 NPCs can be processed in a single request",
            400
        )
    
    try:
        # Initialize belief system integration
        from npcs.belief_system_integration import NPCBeliefSystemIntegration
        belief_integration = NPCBeliefSystemIntegration(user_id, conversation_id)
        await belief_integration.initialize()
        
        # Process the event for beliefs
        results = await belief_integration.process_event_for_beliefs(
            event_text=event_text,
            event_type=event_type,
            npc_ids=npc_ids,
            factuality=factuality,
            importance=importance
        )
        
        return jsonify({
            "success": True,
            "results": results,
            "message": f"Event processed for {len(npc_ids)} NPCs"
        })
    except Exception as e:
        return create_error_response(
            e,
            "An error occurred while processing the event for beliefs."
        )

@npc_learning_bp.route("/api/npc/beliefs/process-conversation", methods=["POST"])
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
    'conversation_text': {
        'type': 'string',
        'required': True,
        'max_length': 2000
    },
    'speaker_id': {
        'type': 'id',
        'required': True
    },
    'listener_id': {
        'type': 'id',
        'required': True
    },
    'speaker_credibility': {
        'type': 'number',
        'min': 0.0,
        'max': 1.0,
        'required': False
    },
    'topic': {
        'type': 'string',
        'required': False,
        'max_length': 100
    }
})
async def process_conversation_for_beliefs_api():
    """
    Process a conversation to generate beliefs for an NPC listener.
    
    This endpoint allows NPCs to form beliefs based on conversations with other characters,
    taking into account the speaker's credibility and the listener's relationship with the speaker.
    
    Returns:
        JSON response with the beliefs formed for the listener NPC.
    """
    # Get sanitized data
    data = request.sanitized_data
    user_id = data["user_id"]
    conversation_id = data["conversation_id"]
    conversation_text = data["conversation_text"]
    speaker_id = data["speaker_id"]
    listener_id = data["listener_id"]
    speaker_credibility = data.get("speaker_credibility", 0.7)
    topic = data.get("topic", "general")
    
    try:
        # Initialize belief system integration
        from npcs.belief_system_integration import NPCBeliefSystemIntegration
        belief_integration = NPCBeliefSystemIntegration(user_id, conversation_id)
        await belief_integration.initialize()
        
        # Process the conversation for beliefs
        results = await belief_integration.process_conversation_for_beliefs(
            conversation_text=conversation_text,
            speaker_id=speaker_id,
            listener_id=listener_id,
            speaker_credibility=speaker_credibility,
            topic=topic
        )
        
        return jsonify({
            "success": True,
            "results": results,
            "message": f"Conversation processed for NPC {listener_id}"
        })
    except Exception as e:
        return create_error_response(
            e,
            "An error occurred while processing the conversation for beliefs."
        )
