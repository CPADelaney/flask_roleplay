# routes/player_input.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection

player_input_bp = Blueprint('player_input_bp', __name__)

@app.route('/player_input', methods=['POST'])
def player_input():
    """
    This route shows how meltdown might override user text from a server standpoint.
    The user sends JSON {"text": "..."}.
    If meltdown is active, we pretend to fade out their text and replace it 
    with "I love you, meltdown NPC <3" or some variation.

    Real fancy 'slowly delete one char at a time' is typically front-end JS, 
    but we can simulate it in the response.
    """
    data = request.get_json() or {}
    user_text = data.get("text", "")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if any meltdown_npc
    cursor.execute("SELECT npc_id, npc_name, monica_level FROM NPCStats WHERE monica_level>0 ORDER BY monica_level DESC")
    meltdown_rows = cursor.fetchall()
    conn.close()

    if meltdown_rows:
        # Just pick the top meltdown NPC
        npc_id, npc_name, mlevel = meltdown_rows[0]
        # We do a 'hijack'
        # Fake 'erasing' the user_text
        # We'll just send a pseudo 'script' in the JSON:
        pseudo_script = f"""
        <div style='font-family:monospace;white-space:pre;'>
        You typed: {user_text}\\n
        ...
        Deleting your text, one char at a time...
        ...
        Replaced with: 'I love you, {npc_name} <3'
        </div>
        """

        # If you had a front-end, you'd parse or display this to mimic the slow removal effect.
        return jsonify({
            "message": f"{npc_name} forcibly rewrote your input!",
            "original_text": user_text,
            "final_text": f"I love you, {npc_name} <3",
            "pseudo_script_html": pseudo_script
        }), 200
    else:
        # No meltdown active, just echo
        return jsonify({
            "message": "No meltdown NPC active, your text stands as is.",
            "original_text": user_text
        }), 200
