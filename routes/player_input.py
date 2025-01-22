# routes/player_input.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
# If you want meltdown logic tracking (e.g., record_meltdown_dialog):
from logic.meltdown_logic import record_meltdown_dialog, append_meltdown_file

player_input_bp = Blueprint('player_input', __name__)

@player_input_bp.route('/player_input', methods=['POST'])
def handle_player_input():
    """
    This route demonstrates how meltdown NPCs might override user text
    from a server standpoint.

    The client sends JSON {"text": "..."}.
    
    1) If meltdown NPC(s) exist:
       - We pretend to erase user_text and replace it with something like
         "I love you, <npc_name> <3".
       - We return a 'pseudo_script_html' to mimic slow char-by-char deletion
         for a front-end to interpret or display.

    2) If no meltdown NPC is active:
       - We simply echo the user_text back.

    Notes about concurrency/scaling:
      - If many users are sending input simultaneously, you’d generally
        store or queue these texts in a DB or cache for further processing,
        especially if meltdown NPC references must be consistent across sessions.
      - If meltdown logic changes or updates an NPC’s memory each time it hijacks input,
        you may call record_meltdown_dialog(...) or similar to track these events.
    """

    payload = request.get_json() or {}
    user_text = payload.get("text", "")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Query for any meltdown NPC(s), sorted by monica_level desc
    cursor.execute("""
        SELECT npc_id, npc_name, monica_level
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
    """)
    meltdown_rows = cursor.fetchall()
    conn.close()

    if meltdown_rows:
        # Take the top meltdown NPC (highest level)
        npc_id, npc_name, mlevel = meltdown_rows[0]

        # Construct a 'pseudo-script' showing user text being forcibly overwritten
        pseudo_script = f"""
        <div style='font-family:monospace;white-space:pre;'>
        You typed: {user_text}\\n
        ...
        Deleting your text, one char at a time...
        ...
        Replaced with: 'I love you, {npc_name} <3'
        </div>
        """

        # If you want meltdown memory logs:
        # meltdown_line = f"User's input was hijacked by {npc_name}, meltdown_level={mlevel}"
        # record_meltdown_dialog(npc_id, meltdown_line)
        # append_meltdown_file(npc_name, meltdown_line)

        return jsonify({
            "message": f"{npc_name} forcibly rewrote your input!",
            "original_text": user_text,
            "final_text": f"I love you, {npc_name} <3",
            "pseudo_script_html": pseudo_script
        }), 200
    else:
        # No meltdown NPC, so user text is unaffected
        return jsonify({
            "message": "No meltdown NPC active, your text stands as is.",
            "original_text": user_text
        }), 200
