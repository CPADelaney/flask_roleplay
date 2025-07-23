# routes/multiuser_routes.py

from quart import Blueprint, request, jsonify, session
from db.connection import get_db_connection_context
import logging
import asyncpg
from db.connection import get_db_dsn, get_db_connection_context

multiuser_bp = Blueprint("multiuser_bp", __name__)

@multiuser_bp.route("/folders", methods=["POST"])
async def create_folder():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    async with get_db_connection_context() as conn:
        # Insert new folder
        row = await conn.fetchrow("""
            INSERT INTO folders (user_id, folder_name)
            VALUES ($1, $2)
            RETURNING id
        """, user_id, folder_name)
        folder_id = row['id']

    return jsonify({"folder_id": folder_id, "folder_name": folder_name})

@multiuser_bp.route("/folders", methods=["GET"])
async def list_folders():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT id, folder_name, created_at
            FROM folders
            WHERE user_id = $1
            ORDER BY created_at
        """, user_id)

    results = []
    for r in rows:
        results.append({
            "folder_id": r['id'],
            "folder_name": r['folder_name'],
            "created_at": r['created_at'].isoformat() if r['created_at'] else None
        })
    return jsonify({"folders": results})

@multiuser_bp.route("/folders/<int:folder_id>", methods=["PUT"])
async def rename_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json() or {}
    new_name = data.get("folder_name", "").strip()
    if not new_name:
        return jsonify({"error": "No folder name"}), 400

    async with get_db_connection_context() as conn:
        # check folder ownership
        row = await conn.fetchrow("SELECT user_id FROM folders WHERE id=$1", folder_id)
        if not row:
            return jsonify({"error":"Folder not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error":"Unauthorized"}), 403

        # rename
        await conn.execute("UPDATE folders SET folder_name=$1 WHERE id=$2", new_name, folder_id)

    return jsonify({"message":"Folder renamed"})

@multiuser_bp.route("/conversations/<int:conv_id>/move_folder", methods=["POST"])
async def move_folder_auto_create(conv_id):
    """
    Creates folder if needed, then moves conversation to folder_id
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    async with get_db_connection_context() as conn:
        # Check conversation ownership
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id=$1", conv_id)
        if not row:
            return jsonify({"error":"Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error":"Unauthorized"}), 403

        # Check if folder already exists for this user
        frow = await conn.fetchrow("""
            SELECT id FROM folders
            WHERE user_id=$1 AND folder_name ILIKE $2
        """, user_id, folder_name)

        if frow:
            # folder already exists
            folder_id = frow['id']
        else:
            # Create a new folder
            row = await conn.fetchrow("""
                INSERT INTO folders (user_id, folder_name)
                VALUES ($1, $2) RETURNING id
            """, user_id, folder_name)
            folder_id = row['id']

        # Now update the conversation's folder_id
        await conn.execute("""
            UPDATE conversations
            SET folder_id=$1
            WHERE id=$2
        """, folder_id, conv_id)

    return jsonify({"message": f"Conversation moved to folder '{folder_name}' (ID={folder_id})"})

@multiuser_bp.route("/folders/<int:folder_id>", methods=["DELETE"])
async def delete_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        # check ownership
        row = await conn.fetchrow("SELECT user_id FROM folders WHERE id=$1", folder_id)
        if not row:
            return jsonify({"error":"Folder not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error":"Unauthorized"}), 403

        # Delete the folder
        await conn.execute("DELETE FROM folders WHERE id=$1", folder_id)

    return jsonify({"message":"Folder deleted"})

@multiuser_bp.route("/conversations", methods=["GET"])
async def list_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    # FIX: Ensure user_id is int
    try:
        user_id = int(user_id) if isinstance(user_id, str) else user_id
    except (ValueError, TypeError):
        logging.error(f"Invalid user_id in session: {user_id}")
        return jsonify({"error": "Invalid user session"}), 400
    
    try:
        # Use a fresh connection with statement_cache_size=0 for pgbouncer compatibility
        dsn = get_db_dsn()
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            # Correct way to fetch conversations with asyncpg
            rows = await conn.fetch("""
                SELECT c.id, c.conversation_name as name, c.created_at,
                       f.folder_name
                FROM conversations c
                LEFT JOIN folders f ON c.folder_id = f.id
                WHERE c.user_id = $1
                ORDER BY c.created_at DESC
            """, user_id)
            
            conversations = [dict(id=row['id'], 
                                 name=row['name'], 
                                 created_at=row['created_at'].isoformat() if row['created_at'] else None,
                                 folder=row['folder_name'])
                            for row in rows]
            
            return jsonify(conversations)
        finally:
            await conn.close()
    except Exception as e:
        logging.error(f"Error listing conversations for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@multiuser_bp.route("/conversations", methods=["POST"])
async def create_conversation():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json()
    name = data.get("conversation_name", "Untitled Session")

    async with get_db_connection_context() as conn:
        row = await conn.fetchrow("""
            INSERT INTO conversations (user_id, conversation_name)
            VALUES ($1, $2) RETURNING id
        """, user_id, name)
        new_id = row['id']

    return jsonify({"conversation_id": new_id, "conversation_name": name})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["GET"])
async def get_messages(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        # Check conversation ownership
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id = $1", conv_id)
        if not row:
            return jsonify({"error": "Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        # Fetch messages
        rows = await conn.fetch("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY id ASC
        """, conv_id)

    messages = [
        {"sender": r['sender'], "content": r['content'], "created_at": r['created_at'].isoformat()}
        for r in rows
    ]
    return jsonify({"messages": messages})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["POST"])
async def add_message(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        # Check ownership again
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id = $1", conv_id)
        if not row:
            return jsonify({"error": "Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        # Insert the message
        data = await request.get_json()
        sender = data.get("sender", "user")
        content = data.get("content", "")

        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """, conv_id, sender, content)

    return jsonify({"status": "ok"})

# RENAME
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["PUT"])
async def rename_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json()
    new_name = data.get("conversation_name", "New Chat")

    async with get_db_connection_context() as conn:
        # Check ownership
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id=$1", conv_id)
        if not row:
            return jsonify({"error": "Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        # Rename
        await conn.execute("""
            UPDATE conversations
            SET conversation_name = $1
            WHERE id = $2
        """, new_name, conv_id)

    return jsonify({"message": "Renamed"})

# Move
@multiuser_bp.route("/conversations/<int:conv_id>/folder", methods=["POST"])
async def move_conversation_to_folder(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = await request.get_json() or {}
    new_folder_id = data.get("folder_id")

    # For 'inbox' or 'no folder', you can pass folder_id=None
    # or pass 0, etc.

    async with get_db_connection_context() as conn:
        # check conversation ownership
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id=$1", conv_id)
        if not row:
            return jsonify({"error":"Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error":"Unauthorized"}), 403

        # If new_folder_id is not None, verify that folder belongs to user
        if new_folder_id:
            frow = await conn.fetchrow("SELECT user_id FROM folders WHERE id=$1", new_folder_id)
            if not frow:
                return jsonify({"error":"Folder not found"}), 404
            if frow['user_id'] != user_id:
                return jsonify({"error":"Unauthorized folder"}), 403

        # set folder_id
        await conn.execute("""
            UPDATE conversations
            SET folder_id=$1
            WHERE id=$2
        """, new_folder_id, conv_id)

    return jsonify({"message":"Conversation moved"})

# DELETE
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["DELETE"])
async def delete_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error":"Not logged in"}), 401

    async with get_db_connection_context() as conn:
        row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id=$1", conv_id)
        if not row:
            return jsonify({"error":"Conversation not found"}), 404
        if row['user_id'] != user_id:
            return jsonify({"error":"Unauthorized"}), 403

        # This will also delete all messages referencing conversation_id
        await conn.execute("DELETE FROM conversations WHERE id=$1", conv_id)

    return jsonify({"message":"Deleted"})
