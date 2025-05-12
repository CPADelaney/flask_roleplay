# routes/multiuser_routes.py

from quart import Blueprint, request, jsonify, session
from db.connection import get_db_connection_context  # Updated import
import logger

multiuser_bp = Blueprint("multiuser_bp", __name__)

@multiuser_bp.route("/folders", methods=["POST"])
async def create_folder():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    async with get_db_connection_context() as conn:
        # Insert new folder
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO folders (user_id, folder_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, folder_name))
            row = await cur.fetchone()
            folder_id = row[0]
        await conn.commit()

    return jsonify({"folder_id": folder_id, "folder_name": folder_name})

@multiuser_bp.route("/folders", methods=["GET"])
async def list_folders():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT id, folder_name, created_at
                FROM folders
                WHERE user_id = %s
                ORDER BY created_at
            """, (user_id,))
            rows = await cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "folder_id": r[0],
            "folder_name": r[1],
            "created_at": r[2].isoformat() if r[2] else None
        })
    return jsonify({"folders": results})

@multiuser_bp.route("/folders/<int:folder_id>", methods=["PUT"])
async def rename_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    new_name = data.get("folder_name", "").strip()
    if not new_name:
        return jsonify({"error": "No folder name"}), 400

    async with get_db_connection_context() as conn:
        # check folder ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM folders WHERE id=%s", (folder_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error":"Folder not found"}), 404
            if row[0] != user_id:
                return jsonify({"error":"Unauthorized"}), 403

            # rename
            await cur.execute("UPDATE folders SET folder_name=%s WHERE id=%s", (new_name, folder_id))
        await conn.commit()

    return jsonify({"message":"Folder renamed"})

@multiuser_bp.route("/conversations/<int:conv_id>/move_folder", methods=["POST"])
async def move_folder_auto_create(conv_id):
    """
    Creates folder if needed, then moves conversation to folder_id
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    folder_name = data.get("folder_name", "").strip()
    if not folder_name:
        return jsonify({"error": "No folder name provided"}), 400

    async with get_db_connection_context() as conn:
        # Check conversation ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error":"Conversation not found"}),404
            if row[0] != user_id:
                return jsonify({"error":"Unauthorized"}),403

            # Check if folder already exists for this user
            await cur.execute("""
                SELECT id FROM folders
                WHERE user_id=%s AND folder_name ILIKE %s
            """, (user_id, folder_name))
            frow = await cur.fetchone()

            if frow:
                # folder already exists
                folder_id = frow[0]
            else:
                # Create a new folder
                await cur.execute("""
                    INSERT INTO folders (user_id, folder_name)
                    VALUES (%s, %s) RETURNING id
                """, (user_id, folder_name))
                row = await cur.fetchone()
                folder_id = row[0]

            # Now update the conversation's folder_id
            await cur.execute("""
                UPDATE conversations
                SET folder_id=%s
                WHERE id=%s
            """, (folder_id, conv_id))
        await conn.commit()

    return jsonify({"message": f"Conversation moved to folder '{folder_name}' (ID={folder_id})"})


@multiuser_bp.route("/folders/<int:folder_id>", methods=["DELETE"])
async def delete_folder(folder_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        # check ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM folders WHERE id=%s", (folder_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error":"Folder not found"}), 404
            if row[0] != user_id:
                return jsonify({"error":"Unauthorized"}),403

            # If using ON DELETE SET NULL, removing folder won't remove convos.
            # If you used ON DELETE CASCADE, removing folder would also remove convos.
            await cur.execute("DELETE FROM folders WHERE id=%s", (folder_id,))
        await conn.commit()

    return jsonify({"message":"Folder deleted"})

@multiuser_bp.route("/conversations", methods=["GET"])
async def list_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
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
        logger.error(f"Error listing conversations for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@multiuser_bp.route("/conversations", methods=["POST"])
async def create_conversation():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    name = data.get("conversation_name", "Untitled Session")

    async with get_db_connection_context() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s) RETURNING id
            """, (user_id, name))
            row = await cur.fetchone()
            new_id = row[0]
        await conn.commit()

    return jsonify({"conversation_id": new_id, "conversation_name": name})

@multiuser_bp.route("/conversations/<int:conv_id>/messages", methods=["GET"])
async def get_messages(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    async with get_db_connection_context() as conn:
        # Check conversation ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error": "Conversation not found"}), 404
            if row[0] != user_id:
                return jsonify({"error": "Unauthorized"}), 403

            # Fetch messages
            await cur.execute("""
                SELECT sender, content, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id ASC
            """, (conv_id,))
            rows = await cur.fetchall()

    messages = [
        {"sender": r[0], "content": r[1], "created_at": r[2].isoformat()}
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
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error": "Conversation not found"}), 404
            if row[0] != user_id:
                return jsonify({"error": "Unauthorized"}), 403

            # Insert the message
            data = request.get_json()
            sender = data.get("sender", "user")
            content = data.get("content", "")

            await cur.execute("""
                INSERT INTO messages (conversation_id, sender, content)
                VALUES (%s, %s, %s)
            """, (conv_id, sender, content))
        await conn.commit()

    return jsonify({"status": "ok"})

# RENAME
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["PUT"])
async def rename_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    new_name = data.get("conversation_name", "New Chat")

    async with get_db_connection_context() as conn:
        # Check ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error": "Conversation not found"}), 404
            if row[0] != user_id:
                return jsonify({"error": "Unauthorized"}), 403

            # Rename
            await cur.execute("""
                UPDATE conversations
                SET conversation_name = %s
                WHERE id = %s
            """, (new_name, conv_id))
        await conn.commit()

    return jsonify({"message": "Renamed"})

# Move
@multiuser_bp.route("/conversations/<int:conv_id>/folder", methods=["POST"])
async def move_conversation_to_folder(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    new_folder_id = data.get("folder_id")

    # For 'inbox' or 'no folder', you can pass folder_id=None
    # or pass 0, etc.

    async with get_db_connection_context() as conn:
        # check conversation ownership
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error":"Conversation not found"}), 404
            if row[0] != user_id:
                return jsonify({"error":"Unauthorized"}), 403

            # If new_folder_id is not None, verify that folder belongs to user
            if new_folder_id:
                await cur.execute("SELECT user_id FROM folders WHERE id=%s",(new_folder_id,))
                frow = await cur.fetchone()
                if not frow:
                    return jsonify({"error":"Folder not found"}),404
                if frow[0] != user_id:
                    return jsonify({"error":"Unauthorized folder"}),403

            # set folder_id
            await cur.execute("""
                UPDATE conversations
                SET folder_id=%s
                WHERE id=%s
            """, (new_folder_id, conv_id))
        await conn.commit()
    return jsonify({"message":"Conversation moved"})


# DELETE
@multiuser_bp.route("/conversations/<int:conv_id>", methods=["DELETE"])
async def delete_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error":"Not logged in"}),401

    async with get_db_connection_context() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT user_id FROM conversations WHERE id=%s",(conv_id,))
            row = await cur.fetchone()
            if not row:
                return jsonify({"error":"Conversation not found"}),404
            if row[0]!=user_id:
                return jsonify({"error":"Unauthorized"}),403

            # This will also delete all messages referencing conversation_id
            await cur.execute("DELETE FROM conversations WHERE id=%s",(conv_id,))
        await conn.commit()

    return jsonify({"message":"Deleted"})
