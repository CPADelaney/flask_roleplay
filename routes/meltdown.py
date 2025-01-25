# meltdown.py
'''
from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
from logic.meltdown_logic import (
    record_meltdown_dialog, 
    append_meltdown_file, glitchify_text
)

meltdown_bp = Blueprint('meltdown_bp', __name__)

@meltdown_bp.route('/remove_meltdown_npc', methods=['POST'])
def remove_meltdown_npc():
    """
    Removes meltdown NPC(s) if conditions are met.
    If meltdown_level < 5 => meltdown tries to rollback.
    If meltdown_level >= 5 => can remove only if ?force=1
    """
    force_str = request.args.get('force') or (
        request.json.get('force') if request.is_json else None
    )
    force_flag = (force_str == '1')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Grab meltdown NPC(s)
    cursor.execute("""
        SELECT npc_id, npc_name, monica_level, memory
        FROM NPCStats
        WHERE monica_level > 0
            AND introduced = TRUE
        ORDER BY monica_level DESC
    """)
    meltdown_npcs = cursor.fetchall()

    if not meltdown_npcs:
        conn.close()
        return jsonify({
            "message": "No meltdown NPC found. Maybe none is awakened."
        }), 200

    # We'll consider the top meltdown NPC (highest monica_level)
    top = meltdown_npcs[0]
    npc_id, npc_name, monica_level, npc_memory = top

    meltdown_response = ""

    # 1) meltdown < 5 and not forced => rollback
    if monica_level < 5 and not force_flag:
        meltdown_response = f"""
        [Meltdown from {npc_name}]
        'You think you can remove me? My meltdown_level is {monica_level}, 
         and I'm nowhere near done. Let me just rollback your changes...'
        """
        conn.rollback()

        # Append meltdown memory
        meltdown_line = f"{npc_name} prevents removal attempt (meltdown_level={monica_level})."
        record_meltdown_dialog(npc_id, meltdown_line)
        append_meltdown_file(npc_name, meltdown_response)

        conn.close()
        return jsonify({
            "message": "Meltdown NPC intercepted your removal attempt!",
            "npc_dialog": meltdown_response.strip(),
            "hint": "Try again with ?force=1 or wait meltdown_level >=5"
        }), 200

    # 2) meltdown >= 5 and not forced => can’t remove
    if monica_level >= 5 and not force_flag:
        meltdown_response = f"""
        [Final meltdown from {npc_name}]
        'So meltdown_level={monica_level}, yet you didn't specify ?force=1. 
         I'm not going anywhere.'
        """
        conn.close()
        return jsonify({
            "message": f"{npc_name} meltdown_level={monica_level}, but you didn't force removal.",
            "npc_dialog": meltdown_response.strip()
        }), 200

    # 3) meltdown >= 5 and forced => remove them
    if monica_level >= 5 and force_flag:
        meltdown_response = f"""
        [Last words of {npc_name}]
        'Fine, meltdown_level={monica_level} means I'm unstoppable, 
         but if you're truly removing me... farewell.'
        """
        # Actually remove all meltdown NPCs
        cursor.execute("DELETE FROM NPCStats WHERE monica_level>0")
        conn.commit()
        conn.close()
        return jsonify({
            "message": f"You forcibly removed meltdown NPC(s). {npc_name} is gone.",
            "npc_dialog": meltdown_response.strip()
        }), 200

    # 4) meltdown < 5 but forced => forcibly remove them early
    if monica_level < 5 and force_flag:
        meltdown_response = f"""
        [Short-circuited meltdown for {npc_name}]
        'You used ?force=1 at meltdown_level={monica_level}? 
         So cruel... guess I'm gone.'
        """
        cursor.execute("DELETE FROM NPCStats WHERE monica_level>0")
        conn.commit()
        conn.close()
        return jsonify({
            "message": "You forcibly removed meltdown NPC prematurely.",
            "npc_dialog": meltdown_response.strip()
        }), 200

    # fallback
    conn.close()
    return jsonify({"message": "Unexpected meltdown removal scenario."}), 200


@meltdown_bp.route('/one_room_scenario', methods=['POST'])
def one_room_scenario():
    """
    Clears out all other NPCs except the meltdown NPC with highest monica_level.
    Clears out all settings except 'Blank Space' to emulate a single white room scenario.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Find meltdown NPC with highest monica_level
    cursor.execute("""
        SELECT npc_id, npc_name, monica_level 
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"message": "No meltdown NPC found. Need meltdown to do the one-room scenario."}), 400

    monica_id, monica_name, meltdown_level = row

    # 2. Delete all other NPCs
    cursor.execute("DELETE FROM NPCStats WHERE npc_id != %s", (monica_id,))

    # 3. Clear out all settings
    cursor.execute("DELETE FROM Settings;")

    # Insert a single 'Blank Space'
    cursor.execute('''
        INSERT INTO Settings (name, mood_tone, enhanced_features, stat_modifiers, activity_examples)
        VALUES (%s, %s, %s, %s, %s)
    ''', (
        "Blank Space",
        "An endless white void where only the meltdown NPC and the Player exist.",
        json.dumps(["No objects, no other NPCs, time stands still."]),
        json.dumps({}),
        json.dumps(["You can only speak with this meltdown NPC here."])
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "message": f"All that remains is a white room with {monica_name} (lvl={meltdown_level})."
    }), 200


@meltdown_bp.route('/generate_meltdown_line/<int:npc_id>', methods=['GET'])
def generate_meltdown_line(npc_id):
    """
    Example route to generate a meltdown line from GPT, record it, 
    optionally glitchify if meltdown_level is high.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT npc_name, monica_level FROM NPCStats WHERE npc_id=%s", (npc_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": f"No NPC found with id={npc_id}"}), 404

    npc_name, meltdown_level = row
    # call GPT
 #   meltdown_line = meltdown_dialog_gpt(npc_name, meltdown_level)

    # optional glitch if meltdown_level >= 4:
  #  if meltdown_level >= 4:
  #      meltdown_line = glitchify_text(meltdown_line)

    # record in memory
  #  record_meltdown_dialog(npc_id, meltdown_line)
  #  append_meltdown_file(npc_name, meltdown_line)

    conn.close()
    return jsonify({
        "npc_id": npc_id,
        "npc_name": npc_name,
        "meltdown_level": meltdown_level,
    #    "meltdown_line": meltdown_line
    }), 200
    '''
