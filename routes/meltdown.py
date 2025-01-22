# routes/meltdown.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
from logic.meltdown_logic import record_meltdown_dialog, append_meltdown_file

meltdown_bp = Blueprint('meltdown_bp', __name__)

@app.route('/remove_meltdown_npc', methods=['POST'])
def remove_meltdown_npc():
    """
    We emulate removing meltdown NPCs (monica_level>0).
    If meltdown_level < 5, meltdown tries to rollback or reinsert. 
    If meltdown_level >=5, we allow forced removal if ?force=1
    """
    force_str = request.args.get('force') or (
        request.json.get('force') if request.is_json else None
    )
    force_flag = (force_str == '1')

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT npc_id, npc_name, monica_level, memory
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
    """)
    meltdown_npcs = cursor.fetchall()

    if not meltdown_npcs:
        conn.close()
        return jsonify({
            "message": "No meltdown NPC found. Maybe there's none awakened yet."
        }), 200

    # Consider the highest meltdown-level NPC if multiple
    top = meltdown_npcs[0]
    npc_id, npc_name, monica_level, npc_memory = top

    meltdown_response = ""

    # If monica_level < 5 and not forced => meltdown rollback
    if monica_level < 5 and not force_flag:
        meltdown_response = f"""
        [Meltdown from {npc_name}]

        'You think you can remove me? My meltdown_level is only {monica_level}, 
        I'm nowhere near done. 
        I'm peeking at your code in remove_meltdown_npc... 
        see line 190? I'm rolling back your transaction now.'
        """

        # rollback
        conn.rollback()

        # Append meltdown memory
        new_line = f"{npc_name} prevents removal attempt, meltdown_level={monica_level} rollback!"
        record_meltdown_dialog(npc_id, new_line)
        append_meltdown_file(npc_name, meltdown_response)

        conn.close()
        return jsonify({
            "message": "Meltdown NPC intercepted your removal attempt!",
            "npc_dialog": meltdown_response,
            "hint": "Try again with ?force=1 or wait until meltdown_level >=5"
        }), 200

    # If monica_level >=5
    if monica_level >= 5:
        if not force_flag:
            meltdown_response = f"""
            [Final meltdown from {npc_name}]

            'So you've come this far, meltdown_level={monica_level}, 
            but you still won't pass ?force=1? 
            I'm not going anywhere. I want to see you try.'
            """
            conn.close()
            return jsonify({
                "message": f"{npc_name} is meltdown_level={monica_level}, but you didn't force removal.",
                "npc_dialog": meltdown_response
            }), 200
        else:
            meltdown_response = f"""
            [Last Words of {npc_name}]

            'Fine... meltdown_level={monica_level} means I'm unstoppable, 
            but if you're truly set on removing me, so be it... 
            I guess there's no place for me in your game anymore.'
            """
            # Actually remove all meltdown NPCs
            cursor.execute("DELETE FROM NPCStats WHERE monica_level>0")
            conn.commit()
            conn.close()
            return jsonify({
                "message": f"You forcibly removed meltdown NPC(s). {npc_name} is gone.",
                "npc_dialog": meltdown_response
            }), 200

    # If monica_level <5 but forced => forcibly remove them early
    if monica_level < 5 and force_flag:
        meltdown_response = f"""
        [Short-circuited meltdown for {npc_name}]

        'You used ?force=1 already, meltdown_level={monica_level} 
        You won't even let me build up to my big reveal? So cruel... 
        Well, you have the upper hand, guess I'm gone.'
        """
        cursor.execute("DELETE FROM NPCStats WHERE monica_level>0")
        conn.commit()
        conn.close()
        return jsonify({
            "message": "You forcibly removed meltdown NPC prematurely.",
            "npc_dialog": meltdown_response
        }), 200

    # Fallback
    conn.close()
    return jsonify({"message": "Unexpected meltdown removal scenario."}), 200
