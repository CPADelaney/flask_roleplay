import logging
import json
import random
import time
import asyncio
from flask import Blueprint, request, jsonify, session
import asyncpg
import openai

from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text
from db.connection import get_db_connection  # Not used anymore if using asyncpg
from tasks import process_new_game_task, create_npcs_task

# Use your Railway DSN (public URL for local development)
DB_DSN = "postgresql://postgres:gUAfzAPnULbYOAvZeaOiwuKLLebutXEY@monorail.proxy.rlwy.net:24727/railway"

new_game_bp = Blueprint('new_game_bp', __name__)

def create_conversation_sync(user_id):
    async def _create_conversation():
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            preliminary_name = "New Game"
            status = "processing"
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name, status)
                VALUES ($1, $2, $3)
                RETURNING id
            """, user_id, preliminary_name, status)
            conversation_id = row["id"]
            logging.info(f"Created conversation_id: {conversation_id} for user_id: {user_id}")
            return conversation_id
        finally:
            await conn.close()
    return asyncio.run(_create_conversation())


# --- Helper: a spaced-out GPT call (used by the background task if needed) ---
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    """
    Waits for a short delay and then calls get_chatgpt_response in a thread.
    Adjust delay (in seconds) as needed to spread out API calls.
    """
    await asyncio.sleep(delay)
    return await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    logging.info("=== /start_new_game endpoint called (offloading to background task) ===")
    
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conversation_data = request.get_json() or {}

    if not conversation_data.get("conversation_id"):
        conversation_id = create_conversation_sync(user_id)
        conversation_data["conversation_id"] = conversation_id
    else:
        conversation_id = conversation_data["conversation_id"]

    task = process_new_game_task.delay(user_id, conversation_data)
    logging.info(f"Enqueued process_new_game_task for user_id={user_id}, conversation_id={conversation_id}, task id: {task.id}")

    return jsonify({"job_id": task.id, "conversation_id": conversation_id}), 202

@new_game_bp.route('/conversation_status', methods=['GET'])
def conversation_status():
    conversation_id = request.args.get("conversation_id")
    user_id = session.get("user_id")
    if not user_id or not conversation_id:
        return jsonify({"error": "Missing parameters"}), 400

    async def _get_status():
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            row = await conn.fetchrow("""
                SELECT conversation_name, status 
                FROM conversations 
                WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if row:
                return {"conversation_name": row["conversation_name"], "status": row["status"]}
            else:
                return {"error": "Conversation not found"}
        finally:
            await conn.close()
    return jsonify(asyncio.run(_get_status()))


@new_game_bp.route('/spawn_npcs', methods=['POST'])
async def spawn_npcs():
    """
    Spawns NPCs for a given conversation.
    Expects JSON with a 'conversation_id'.
    Spawns 5 NPCs concurrently using asyncio.gather.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400

    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        row = await conn.fetchrow("""
            SELECT id FROM conversations WHERE id=$1 AND user_id=$2
        """, conversation_id, user_id)
        if not row:
            return jsonify({"error": "Conversation not found or unauthorized"}), 403

        spawn_tasks = [
            asyncio.to_thread(create_npc, user_id=user_id, conversation_id=conversation_id, introduced=False)
            for _ in range(5)
        ]
        spawned_npc_ids = await asyncio.gather(*spawn_tasks)
        logging.info(f"Spawned NPCs concurrently: {spawned_npc_ids}")
        return jsonify({"message": "NPCs spawned", "npc_ids": spawned_npc_ids}), 200
    except Exception as e:
        logging.exception("Error in /spawn_npcs:")
        return jsonify({"error": str(e)}), 500
    finally:
        await conn.close()
