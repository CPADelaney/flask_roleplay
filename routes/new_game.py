# routes/new_game.py

import logging
import json
import random
import time
import os
import asyncio
from quart import Blueprint, request, jsonify, session
import asyncpg

from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator_sdk import get_aggregated_roleplay_context
from db.connection import get_db_connection_context
from routes.story_routes import build_aggregator_text

# Use your Railway DSN (public URL for local development)
DB_DSN = os.getenv("DB_DSN") 
logging.info(f"[new_game] Using DB_DSN={DB_DSN}")  # Add right after retrieving

new_game_bp = Blueprint('new_game_bp', __name__)


async def create_conversation_async(user_id):
    async with get_db_connection_context() as conn:
        preliminary_name = "New Game"
        status = "processing"
        
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name, status)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (user_id, preliminary_name, status))
            
            row = await cursor.fetchone()
            conversation_id = row[0]
            
        logging.info(f"Created conversation_id: {conversation_id} for user_id: {user_id}")
        return conversation_id


# --- Helper: a spaced-out GPT call (used by the background task if needed) ---
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    """
    Waits for a short delay and then calls get_chatgpt_response in a thread.
    Adjust delay (in seconds) as needed to spread out API calls.
    """
    await asyncio.sleep(delay)
    return await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)

@new_game_bp.route('/start_new_game', methods=['POST'])
async def start_new_game():
    from tasks import process_new_game_task  # Import task that uses NewGameAgent and Nyx governance
    logging.info("=== /start_new_game endpoint called (offloading to background task using NewGameAgent) ===")
    
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conversation_data = request.get_json() or {}

    if not conversation_data.get("conversation_id"):
        conversation_id = await create_conversation_async(user_id)
        conversation_data["conversation_id"] = conversation_id
    else:
        conversation_id = conversation_data["conversation_id"]

    task = process_new_game_task.delay(user_id, conversation_data)
    logging.info(f"Enqueued process_new_game_task with NewGameAgent for user_id={user_id}, conversation_id={conversation_id}, task id: {task.id}")

    return jsonify({"job_id": task.id, "conversation_id": conversation_id}), 202

@new_game_bp.route('/conversation_status', methods=['GET'])
async def conversation_status():
    conversation_id_str = request.args.get("conversation_id")
    user_id = session.get("user_id")
    
    # Ensure user_id is int
    if user_id is not None:
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid user session"}), 400
    
    if not user_id or not conversation_id_str:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        conversation_id = int(conversation_id_str)
    except ValueError:
        return jsonify({"error": "Invalid conversation_id parameter"}), 400

    async with get_db_connection_context() as conn:
        # Get conversation status
        row = await conn.fetchrow(
            """
            SELECT conversation_name, status 
            FROM conversations 
            WHERE id=$1 AND user_id=$2
            """,
            conversation_id, user_id
        )
        
        if not row:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Also fetch the opening narrative
        opening_msg_row = await conn.fetchrow(
            """
            SELECT content 
            FROM messages 
            WHERE conversation_id=$1 AND sender='Nyx'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            conversation_id
        )
        
        return jsonify({
            "conversation_name": row['conversation_name'], 
            "status": row['status'],
            "opening_narrative": opening_msg_row['content'] if opening_msg_row else None
        })

@new_game_bp.route('/api/preset-stories', methods=['GET'])
async def list_preset_stories():
    """Get all available preset stories"""
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT 
                story_id,
                story_data->>'name' as name,
                story_data->>'theme' as theme,
                story_data->>'synopsis' as synopsis,
                story_data->'acts' as acts
            FROM PresetStories
            ORDER BY created_at DESC
        """)
        
        stories = []
        for row in rows:
            stories.append({
                "id": row["story_id"],
                "name": row["name"],
                "theme": row["theme"], 
                "synopsis": row["synopsis"],
                "num_acts": len(json.loads(row["acts"])) if row["acts"] else 0
            })
        
        return jsonify({"stories": stories})

@new_game_bp.route('/api/new-game/preset', methods=['POST'])
async def start_preset_game():
    """Start a new game with a preset story"""
    from tasks import process_new_game_preset_task
    
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = await request.get_json()
    story_id = data.get('story_id')
    
    if not story_id:
        return jsonify({"error": "story_id required"}), 400
    
    # Verify story exists
    async with get_db_connection_context() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM PresetStories WHERE story_id = $1)",
            story_id
        )
        
        if not exists:
            return jsonify({"error": f"Preset story '{story_id}' not found"}), 404
    
    # Queue the task with correct data structure
    task = process_new_game_preset_task.delay(user_id, {
        "preset_story_id": story_id  # Make sure this key matches
    })
    
    return jsonify({
        "status": "processing",
        "task_id": task.id,
        "story_id": story_id
    })
