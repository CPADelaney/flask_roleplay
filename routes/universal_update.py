# routes/universal_update.py

from quart import Blueprint, request, jsonify, session
import asyncio
import asyncpg
import os
import logging
from db.connection import get_db_connection_context
from logic.universal_updater_agent import apply_universal_updates_async 
from logic.aggregator_sdk import get_aggregated_roleplay_context

universal_bp = Blueprint("universal_bp", __name__)
logger = logging.getLogger(__name__)

@universal_bp.route("/universal_update", methods=["POST"])
async def universal_update():
    # 1) Get user_id from session
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    # 2) Get JSON payload and conversation_id
    data = await request.get_json()  # Note: added await here
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400
    
    # Convert conversation_id to int if it's a string
    try:
        conversation_id = int(conversation_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid conversation_id format"}), 400

    # 3) Use the async context manager
    async with get_db_connection_context() as conn:
        try:
            # 4) Call the updater with all required parameters
            result = await apply_universal_updates_async(user_id, conversation_id, data, conn)
            if "error" in result:
                return jsonify(result), 500
            else:
                return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@universal_bp.route("/get_roleplay_value", methods=["GET"])
async def get_roleplay_value():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
        
    conversation_id = request.args.get("conversation_id")
    key = request.args.get("key")
    
    if not conversation_id or not key:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Convert conversation_id to integer
    try:
        conversation_id = int(conversation_id)
    except ValueError:
        return jsonify({"error": "Invalid conversation_id format"}), 400
    
    async with get_db_connection_context() as conn:
        # Change from cursor pattern to direct query
        row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key=$3
        """, user_id, conversation_id, key)
    
    if row:
        return jsonify({"value": row['value']})
    else:
        return jsonify({"value": None})

# ADD THIS NEW ENDPOINT - This is what's missing and causing the 404
@universal_bp.route('/get_aggregated_roleplay_context', methods=['GET'])
async def get_aggregated_roleplay_context_endpoint():
    """
    Endpoint to get aggregated roleplay context for the game UI.
    """
    # Check authentication
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    # Get parameters
    conversation_id = request.args.get("conversation_id")
    player_name = request.args.get("player_name", "Chase")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id parameter"}), 400
    
    try:
        # Convert to int
        conversation_id = int(conversation_id)
        
        # Get the aggregated context
        context_data = await get_aggregated_roleplay_context(
            user_id=user_id,
            conversation_id=conversation_id,
            player_name=player_name
        )
        
        return jsonify(context_data), 200
        
    except ValueError:
        return jsonify({"error": "Invalid conversation_id"}), 400
    except SyntaxError as e:
        # Log the actual syntax error details
        logger.error(f"SyntaxError in get_aggregated_roleplay_context: {e}")
        logger.error(f"Error details: {e.filename}:{e.lineno} - {e.text}")
        return jsonify({"error": f"Syntax error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error getting aggregated context: {e}", exc_info=True)
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Full traceback:\n{tb}")
        return jsonify({"error": str(e)}), 500

# OPTIONALLY ADD THIS TOO - for player resources
@universal_bp.route('/player/resources', methods=['GET'])
async def get_player_resources():
    """
    Get player resources (money, supplies, influence) and vitals (energy, hunger).
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    conversation_id = request.args.get("conversation_id")
    player_name = request.args.get("player_name", "Chase")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    try:
        conversation_id = int(conversation_id)
        
        async with get_db_connection_context() as conn:
            # Get resources
            resources_row = await conn.fetchrow("""
                SELECT money, supplies, influence
                FROM PlayerResources
                WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
            """, user_id, conversation_id, player_name)
            
            # Get vitals
            vitals_row = await conn.fetchrow("""
                SELECT energy, hunger
                FROM PlayerVitals
                WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
            """, user_id, conversation_id, player_name)
            
            resources = {}
            vitals = {}
            
            if resources_row:
                resources = {
                    "money": resources_row["money"],
                    "supplies": resources_row["supplies"],
                    "influence": resources_row["influence"]
                }
            
            if vitals_row:
                vitals = {
                    "energy": vitals_row["energy"],
                    "hunger": vitals_row["hunger"]
                }
            
            return jsonify({
                "resources": resources,
                "vitals": vitals
            }), 200
            
    except ValueError:
        return jsonify({"error": "Invalid conversation_id"}), 400
    except Exception as e:
        logger.error(f"Error getting player resources: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
