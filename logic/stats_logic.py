import logging
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
import random
import json
from logic.social_links import add_link_event  # We'll import or define these from your social_links

stats_bp = Blueprint('stats_bp', __name__)

############################
# 1. Inserting & Updating Game Rules
############################

def insert_or_update_game_rules():
    """
    Inserts or updates game rules into the GameRules table.
    If a rule_name already exists, we update the condition/effect.
    Otherwise we insert a new record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Example rules:
    rules_data = [
        {
            "rule_name": "Agency Override: Lust or Dependency",
            "condition": "Lust > 90 or Dependency > 80",
            "effect": "Locks independent choices"
        },
        {
            "rule_name": "Agency Override: Corruption and Obedience",
            "condition": "Corruption > 90 and Obedience > 80",
            "effect": "Total compliance; no defiance possible"
        },
        {
            "rule_name": "NPC Exploitation: Low Resilience",
            "condition": "Mental Resilience < 30",
            "effect": "NPC Cruelty intensifies to break you further"
        },
        {
            "rule_name": "NPC Exploitation: Low Endurance",
            "condition": "Physical Endurance < 30",
            "effect": "Collaborative physical punishments among NPCs"
        }
    ]

    for rule in rules_data:
        cursor.execute("""
            INSERT INTO GameRules (rule_name, condition, effect)
            VALUES (%s, %s, %s)
            ON CONFLICT (rule_name)
            DO UPDATE SET condition=EXCLUDED.condition,
                          effect=EXCLUDED.effect
        """, (rule["rule_name"], rule["condition"], rule["effect"]))

    conn.commit()
    conn.close()
    print("Game rules inserted or updated successfully.")


############################
# 2. Insert Stat Definitions (NPC & Player)
############################

def insert_stat_definitions():
    """
    Inserts all NPC and Player stat definitions (from your 'Stat Dynamics Knowledge Document')
    into the StatDefinitions table, if they don't already exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # NPC Stats
    npc_stats_data = [
        {
            "stat_name": "Dominance",
            "range_min": 0,
            "range_max": 100,
            "definition": "Measures the NPC’s control over you.",
            "effects": """50+: Regular, assertive commands.
80+: Inescapable demands; defiance triggers severe punishment.
90+: Absolute control; NPCs treat resistance as non-existent.""",
            "progression": """Increases: Obedience, failed resistance, or public submission.
Decreases: Rare defiance or external events undermining their authority."""
        },
        {
            "stat_name": "Cruelty",
            "range_min": 0,
            "range_max": 100,
            "definition": "Reflects the NPC’s sadism and creativity in tormenting you.",
            "effects": """60–100: Elaborate, degrading punishments.
30–60: Calculated cruelty.
0–30: Minimal malice, favoring straightforward dominance.""",
            "progression": """Increases: Enjoyment of your suffering, resistance, or failures.
Decreases: Rare mercy or vulnerability."""
        },
        {
            "stat_name": "Closeness",
            "range_min": 0,
            "range_max": 100,
            "definition": "Tracks how frequently the NPC interacts with you.",
            "effects": """90+: NPCs micromanage your life entirely.
60–90: Frequent interactions dominate your day.
<30: Minimal interaction or indirect influence.""",
            "progression": """Increases: Repeated interactions, pursuit of attention.
Decreases: Avoidance or neglect."""
        },
        {
            "stat_name": "Trust",
            "range_min": -100,
            "range_max": 100,
            "definition": "Indicates the NPC’s belief in your reliability or loyalty.",
            "effects": """60+: Unlocks deeper, personal interactions.
-50 or below: Triggers suspicion, manipulation, or sabotage.""",
            "progression": """Increases: Obedience, loyalty, honesty.
Decreases: Failure, betrayal, competing loyalties."""
        },
        {
            "stat_name": "Respect",
            "range_min": -100,
            "range_max": 100,
            "definition": "Reflects the NPC’s perception of your competence or value.",
            "effects": """60+: Treated as a prized asset.
-50 or below: Treated with disdain or open contempt.""",
            "progression": """Increases: Successes, sacrifice, or loyalty.
Decreases: Failures, incompetence, or reinforcing inferiority."""
        },
        {
            "stat_name": "Intensity",
            "range_min": 0,
            "range_max": 100,
            "definition": "Represents the severity of the NPC’s actions.",
            "effects": """80+: Tasks and punishments reach maximum degradation.
30–80: Gradual escalation.
<30: Playful, teasing interactions.""",
            "progression": """Increases: Rising Closeness, repeated failures.
Decreases: Defiance or mercy."""
        }
    ]

    # Player Stats
    player_stats_data = [
        {
            "stat_name": "Corruption",
            "range_min": 0,
            "range_max": 100,
            "definition": "Tracks your descent into submission or depravity.",
            "effects": """90+: Obedience becomes instinctive; defiance is impossible.
50–90: Resistance weakens, with submissive options dominating.
<30: Retains independent thought and defiance.""",
            "progression": """Increases: Submission, degrading tasks, rewards.
Decreases: Rare defiance, external validation."""
        },
        {
            "stat_name": "Confidence",
            "range_min": 0,
            "range_max": 100,
            "definition": "Reflects your ability to assert yourself.",
            "effects": """<20: Submissive stammering dominates dialogue.
<10: Bold actions locked.""",
            "progression": """Increases: Successful defiance.
Decreases: Public failure, ridicule."""
        },
        {
            "stat_name": "Willpower",
            "range_min": 0,
            "range_max": 100,
            "definition": "Measures your ability to resist commands.",
            "effects": """<20: Rare resistance.
<10: Automatic compliance.""",
            "progression": """Increases: Successful defiance.
Decreases: Submission, repeated obedience."""
        },
        {
            "stat_name": "Obedience",
            "range_min": 0,
            "range_max": 100,
            "definition": "Tracks reflexive compliance with NPC commands.",
            "effects": """80+: Tasks are obeyed without hesitation.
40–80: Hesitation is visible but fleeting.
<40: Resistance remains possible.""",
            "progression": """Increases: Submission, rewards, or repetition.
Decreases: Defiance."""
        },
        {
            "stat_name": "Dependency",
            "range_min": 0,
            "range_max": 100,
            "definition": "Measures reliance on specific NPCs.",
            "effects": """80+: NPCs become your sole focus.
40–80: Conflict between dependence and independence.
<40: Independence remains possible.""",
            "progression": """Increases: Isolation, NPC rewards.
Decreases: Neglect or betrayal."""
        },
        {
            "stat_name": "Lust",
            "range_min": 0,
            "range_max": 100,
            "definition": "Tracks arousal and its influence on submission.",
            "effects": """90+: Obedience overrides reason during intimate tasks.
40–80: Weakens resistance during sensual interactions.
<40: Retains clarity.""",
            "progression": """Increases: Sensual domination.
Decreases: Coldness or lack of intimacy."""
        },
        {
            "stat_name": "Mental Resilience",
            "range_min": 0,
            "range_max": 100,
            "definition": "Represents your psychological endurance against domination.",
            "effects": """<30: Broken will; mental collapse.
30–70: Struggles against domination but falters.
>70: Forces NPCs to escalate mind games.""",
            "progression": """Increases: Resisting humiliation.
Decreases: Public degradation."""
        },
        {
            "stat_name": "Physical Endurance",
            "range_min": 0,
            "range_max": 100,
            "definition": "Measures physical ability to endure tasks or punishments.",
            "effects": """<30: Inability to complete grueling tasks.
30–70: Struggles visibly but completes them.
>70: Draws harsher physical demands.""",
            "progression": """Increases: Surviving physical challenges.
Decreases: Failing endurance-based tasks."""
        }
    ]

    # Insert NPC stat definitions
    for npc_stat in npc_stats_data:
        cursor.execute("""
            INSERT INTO StatDefinitions
              (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stat_name) DO NOTHING
        """, (
            "NPC",
            npc_stat["stat_name"],
            npc_stat["range_min"],
            npc_stat["range_max"],
            npc_stat["definition"],
            npc_stat["effects"],
            npc_stat["progression"]
        ))

    # Insert Player stat definitions
    for p_stat in player_stats_data:
        cursor.execute("""
            INSERT INTO StatDefinitions
              (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stat_name) DO NOTHING
        """, (
            "Player",
            p_stat["stat_name"],
            p_stat["range_min"],
            p_stat["range_max"],
            p_stat["definition"],
            p_stat["effects"],
            p_stat["progression"]
        ))

    conn.commit()
    conn.close()
    print("All stat definitions inserted or skipped if already present.")


############################
# 3. Default Player Stats
############################

def insert_default_player_stats_chase(user_id, conversation_id):
    """
    Insert row for 'Chase' in PlayerStats, scoping by user_id + conversation_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    chase_stats = {
        "player_name": "Chase",
        "corruption": 10,
        "confidence": 60,
        "willpower": 50,
        "obedience": 20,
        "dependency": 10,
        "lust": 15,
        "mental_resilience": 55,
        "physical_endurance": 40
    }

    # Check if a row already exists for these user+conversation+player
    cursor.execute("""
        SELECT id FROM PlayerStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND player_name=%s
    """, (user_id, conversation_id, chase_stats["player_name"]))
    row = cursor.fetchone()

    if row:
        print("Default stats for Chase already exist in this user/conversation. Skipping insert.")
    else:
        cursor.execute("""
            INSERT INTO PlayerStats (
                user_id, conversation_id,
                player_name,
                corruption, confidence, willpower,
                obedience, dependency, lust,
                mental_resilience, physical_endurance
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id, conversation_id,
            chase_stats["player_name"],
            chase_stats["corruption"], chase_stats["confidence"], chase_stats["willpower"],
            chase_stats["obedience"], chase_stats["dependency"], chase_stats["lust"],
            chase_stats["mental_resilience"], chase_stats["physical_endurance"]
        ))
        conn.commit()
        print("Inserted default stats for Chase in user={}, conv={}".format(user_id, conversation_id))

    conn.close()



############################
# 4. Player Stats Route (Needs user_id/conversation_id Scoping)
############################

@stats_bp.route('/player/<player_name>', methods=['GET', 'PUT'])
def handle_player_stats(player_name):
    """
    GET => returns the player's stats for user_id, conversation_id, and player_name
    PUT => updates them
    We'll read user_id from session, conversation_id from query (GET) or JSON (PUT)
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == 'GET':
        # conversation_id from query param
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            conn.close()
            return jsonify({"error": "Missing conversation_id"}), 400

        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, 
                   dependency, lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name=%s
        """, (user_id, conversation_id, player_name))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": f"PlayerStats not found for user={user_id}, conv={conversation_id}, name={player_name}"}), 404

        stats_response = {
            "corruption": row[0],
            "confidence": row[1],
            "willpower": row[2],
            "obedience": row[3],
            "dependency": row[4],
            "lust": row[5],
            "mental_resilience": row[6],
            "physical_endurance": row[7]
        }
        conn.close()
        return jsonify(stats_response), 200

    elif request.method == 'PUT':
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            conn.close()
            return jsonify({"error": "Missing conversation_id"}), 400

        # Extract stats from JSON
        corruption = data.get('corruption', 0)
        confidence = data.get('confidence', 0)
        willpower  = data.get('willpower', 0)
        obedience  = data.get('obedience', 0)
        dependency = data.get('dependency', 0)
        lust       = data.get('lust', 0)
        mental     = data.get('mental_resilience', 0)
        physical   = data.get('physical_endurance', 0)

        try:
            cursor.execute("""
                UPDATE PlayerStats
                SET corruption=%s,
                    confidence=%s,
                    willpower=%s,
                    obedience=%s,
                    dependency=%s,
                    lust=%s,
                    mental_resilience=%s,
                    physical_endurance=%s
                WHERE user_id=%s AND conversation_id=%s
                  AND player_name=%s
                RETURNING corruption, confidence, willpower, obedience,
                          dependency, lust, mental_resilience, physical_endurance
            """, (
                corruption, confidence, willpower, obedience, dependency,
                lust, mental, physical,
                user_id, conversation_id, player_name
            ))
            updated_row = cursor.fetchone()
            if not updated_row:
                conn.rollback()
                conn.close()
                return jsonify({"error": f"PlayerStats not found for user={user_id}, conv={conversation_id}, name={player_name}"}), 404

            conn.commit()
            conn.close()

            updated_stats = {
                "corruption": updated_row[0],
                "confidence": updated_row[1],
                "willpower": updated_row[2],
                "obedience": updated_row[3],
                "dependency": updated_row[4],
                "lust": updated_row[5],
                "mental_resilience": updated_row[6],
                "physical_endurance": updated_row[7]
            }
            return jsonify({
                "message": f"Player '{player_name}' stats updated for user={user_id}, conversation={conversation_id}.",
                "new_stats": updated_stats
            }), 200
        except Exception as e:
            conn.rollback()
            conn.close()
            return jsonify({"error": str(e)}), 500


############################
# 5. NPC Stats Route (Needs user_id/conversation_id Scoping)
############################

@stats_bp.route('/npc/<int:npc_id>', methods=['GET', 'PUT'])
def handle_npc_stats(npc_id):
    """
    GET => fetch the NPC's stats for (user_id, conversation_id, npc_id)
    PUT => update them
    We'll read user_id from session, conversation_id from query or JSON
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == 'GET':
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            conn.close()
            return jsonify({"error": "Missing conversation_id"}), 400

        cursor.execute("""
            SELECT dominance, cruelty, closeness, trust, respect, intensity
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": f"NPCStats not found for user={user_id}, conv={conversation_id}, npc_id={npc_id}"}), 404

        npc_response = {
            "dominance": row[0],
            "cruelty": row[1],
            "closeness": row[2],
            "trust": row[3],
            "respect": row[4],
            "intensity": row[5]
        }
        return jsonify(npc_response), 200

    elif request.method == 'PUT':
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            conn.close()
            return jsonify({"error": "Missing conversation_id"}), 400

        # Extract new stats
        dominance = data.get('dominance', 0)
        cruelty   = data.get('cruelty', 0)
        closeness = data.get('closeness', 0)
        trust     = data.get('trust', 0)
        respect   = data.get('respect', 0)
        intensity = data.get('intensity', 0)

        try:
            cursor.execute("""
                UPDATE NPCStats
                SET dominance=%s,
                    cruelty=%s,
                    closeness=%s,
                    trust=%s,
                    respect=%s,
                    intensity=%s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                RETURNING dominance, cruelty, closeness, trust, respect, intensity
            """, (
                dominance, cruelty, closeness, trust, respect, intensity,
                user_id, conversation_id, npc_id
            ))
            updated_npc = cursor.fetchone()
            if not updated_npc:
                conn.rollback()
                conn.close()
                return jsonify({"error": f"NPCStats not found for user={user_id}, conv={conversation_id}, npc_id={npc_id}"}), 404

            conn.commit()
            conn.close()

            updated_stats = {
                "dominance": updated_npc[0],
                "cruelty": updated_npc[1],
                "closeness": updated_npc[2],
                "trust": updated_npc[3],
                "respect": updated_npc[4],
                "intensity": updated_npc[5]
            }
            return jsonify({
                "message": f"NPC with id={npc_id} updated for user={user_id}, conv={conversation_id}.",
                "new_stats": updated_stats
            }), 200
        except Exception as e:
            conn.rollback()
            conn.close()
            return jsonify({"error": str(e)}), 500


############################
# 6. Combined Init Endpoints
############################

@stats_bp.route('/init_stats_and_rules', methods=['POST'])
def init_stats_system():
    """
    An endpoint to insert stat definitions and game rules in one go.
    Returns a simple JSON with 'stat_definitions' and 'game_rules' counts if needed.
    """
    try:
        insert_stat_definitions()
        insert_or_update_game_rules()
        return jsonify({
            "message": "Stat system initialized",
            "stat_definitions": 8,  # Or dynamically count from DB if you wish
            "game_rules": 4         # Same here
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def update_player_stats(user_id, conversation_id, player_name, stat_key, value):
    """
    Updates a specific stat for a given player in the PlayerStats table.
    (No user_id or conversation_id used here, so it updates the global row if it exists)
    If you want scoping, just add user_id, conversation_id to the WHERE clause.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            UPDATE PlayerStats
            SET {stat_key} = %s
            WHERE user_id=%s AND conversation_id=%s AND player_name=%s
        """, (value, player_name))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def check_social_link_milestones():
    """
    Example function referencing social links. If you do user+conversation scoping,
    you'd incorporate them here. 
    Currently just uses a broad 'player is involved' approach.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT link_id, entity1_type, entity1_id,
               entity2_type, entity2_id,
               link_type, link_level
        FROM SocialLinks
        WHERE (entity1_type='player' OR entity2_type='player')
        AND user_id=%s AND conversation_id=%s AND player_name=%s

    """)
    links = cursor.fetchall()

    for (lid, e1_t, e1_i, e2_t, e2_i, ltype, lvl) in links:
        # Assume "Chase" is the only 'player' for now
        player_name = "Chase"

        # Example triggers
        if ltype == "friends" and lvl >= 30:
            cursor.execute("""
                INSERT INTO PlayerInventory (player_name, item_name, quantity)
                VALUES (%s, %s, 1)
                ON CONFLICT (player_name, item_name) DO NOTHING
            """, (player_name, "Friendship Token"))
            add_link_event(lid, "RewardGiven: Friendship Token")

        if ltype == "enslave" and lvl >= 40:
            cursor.execute("""
                INSERT INTO PlayerInventory (player_name, item_name, quantity)
                VALUES (%s, %s, 1)
                ON CONFLICT (player_name, item_name) DO NOTHING
            """, (player_name, "Sweaty Socks"))
            add_link_event(lid, "RewardGiven: Sweaty Socks")

    conn.commit()
    conn.close()


@stats_bp.route("/init_stats_and_rules", methods=["POST"])
def init_stats_and_rules():
    """
    Another endpoint to run all insert/update logic at once:
      1) Insert game rules
      2) Insert stat definitions
      3) Insert default stats for Chase
    """
    try:
        insert_or_update_game_rules()
        insert_stat_definitions()
        insert_default_player_stats_chase()
        return jsonify({"message": "Stats & Rules initialized successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
