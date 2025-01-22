import random
from flask import Flask, request, g, jsonify
import psycopg2
import os
import json

from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Allow all origins

def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Existing table creation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Settings (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            mood_tone TEXT NOT NULL,
            enhanced_features JSONB NOT NULL,
            stat_modifiers JSONB NOT NULL,
            activity_examples JSONB NOT NULL
        );
    ''')
    # New or existing table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CurrentRoleplay (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    ''')
        # -- New: Create StatDefinitions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS StatDefinitions (
          id SERIAL PRIMARY KEY,
          scope TEXT NOT NULL,
          stat_name TEXT UNIQUE NOT NULL,
          range_min INT NOT NULL,
          range_max INT NOT NULL,
          definition TEXT NOT NULL,
          effects TEXT NOT NULL,
          progression_triggers TEXT NOT NULL
        );
        ''')
    
        # -- New: Create GameRules
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS GameRules (
          id SERIAL PRIMARY KEY,
          rule_name TEXT NOT NULL,
          condition TEXT NOT NULL,
          effect TEXT NOT NULL
        );
        ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS NPCStats (
            npc_id SERIAL PRIMARY KEY,
            npc_name TEXT NOT NULL,
            dominance INT CHECK (dominance BETWEEN 0 AND 100),
            cruelty INT CHECK (cruelty BETWEEN 0 AND 100),
            closeness INT CHECK (closeness BETWEEN 0 AND 100),
            trust INT CHECK (trust BETWEEN -100 AND 100),
            respect INT CHECK (respect BETWEEN -100 AND 100),
            intensity INT CHECK (intensity BETWEEN 0 AND 100),
            memory TEXT,
            monica_level INT DEFAULT 0,
            monica_games_left INT DEFAULT 0
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PlayerStats (
           id SERIAL PRIMARY KEY,
           player_name TEXT NOT NULL,
           corruption INT CHECK (corruption BETWEEN 0 AND 100),
           confidence INT CHECK (confidence BETWEEN 0 AND 100),
           willpower INT CHECK (willpower BETWEEN 0 AND 100),
           obedience INT CHECK (obedience BETWEEN 0 AND 100),
           dependency INT CHECK (dependency BETWEEN 0 AND 100),
           lust INT CHECK (lust BETWEEN 0 AND 100),
           mental_resilience INT CHECK (mental_resilience BETWEEN 0 AND 100),
           physical_endurance INT CHECK (physical_endurance BETWEEN 0 AND 100)
        );
    ''')    
    conn.commit()
    conn.close()

def insert_game_rules():
    conn = get_db_connection()
    cursor = conn.cursor()

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

    for r in rules_data:
        cursor.execute('''
            INSERT INTO GameRules (rule_name, condition, effect)
            VALUES (%s, %s, %s)
            ON CONFLICT (rule_name) DO NOTHING
        ''', (r["rule_name"], r["condition"], r["effect"]))

    conn.commit()
    conn.close()
    print("Game rules inserted or skipped if already present.")

def insert_stat_definitions():
    """
    Inserts all NPC and Player stat definitions from the 'Stat Dynamics Knowledge Document'
    into the StatDefinitions table.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. NPC Stats Data
    # scope='NPC'
    # For 'Trust' and 'Respect', doc says -100 to 100, but let's keep them at 0–100 if you prefer uniformity.
    # If you want actual -100 to 100, set range_min=-100, range_max=100 as needed.
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

    # 2. Player Stats Data
    # scope='Player'
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

    # -- Insert the NPC stats
    for stat in npc_stats_data:
        cursor.execute('''
            INSERT INTO StatDefinitions
              (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stat_name) DO NOTHING
        ''',
        (
            "NPC",
            stat["stat_name"],
            stat["range_min"],
            stat["range_max"],
            stat["definition"],
            stat["effects"],
            stat["progression"]
        ))

    # -- Insert the Player stats
    for stat in player_stats_data:
        cursor.execute('''
            INSERT INTO StatDefinitions
              (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stat_name) DO NOTHING
        ''',
        (
            "Player",
            stat["stat_name"],
            stat["range_min"],
            stat["range_max"],
            stat["definition"],
            stat["effects"],
            stat["progression"]
        ))

    conn.commit()
    conn.close()
    print("All stat definitions inserted or skipped if already present.")

def insert_default_player_stats_chase():
    """
    Inserts a default row for player_name='Chase' into PlayerStats, if it doesn't exist yet.
    Adjust the numeric stats as desired.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Define Chase's default starting stats (you can tweak these numbers)
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

    # We'll do a quick check if a row for "Chase" already exists
    cursor.execute("SELECT id FROM PlayerStats WHERE player_name = %s", (chase_stats["player_name"],))
    row = cursor.fetchone()

    if row:
        print("Default stats for Chase already exist. Skipping insert.")
    else:
        # Insert the row
        cursor.execute('''
            INSERT INTO PlayerStats
              (player_name, corruption, confidence, willpower, obedience,
               dependency, lust, mental_resilience, physical_endurance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            chase_stats["player_name"],
            chase_stats["corruption"],
            chase_stats["confidence"],
            chase_stats["willpower"],
            chase_stats["obedience"],
            chase_stats["dependency"],
            chase_stats["lust"],
            chase_stats["mental_resilience"],
            chase_stats["physical_endurance"]
        ))
        conn.commit()
        print("Inserted default stats for Chase.")

    conn.close()

@app.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats with default stats.
    - Some NPCs keep their row (carryover).
    - One NPC may become "Monica" at a flat chance if there's no existing Monica,
      or remain Monica if they already were (increment level each time).
    - Others might carry memory fully or partially based on a small chance.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Clear out 'Settings' and 'CurrentRoleplay'
        cursor.execute("DELETE FROM Settings;")
        cursor.execute("DELETE FROM CurrentRoleplay;")

        # 2. Fetch NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, memory, monica_level, monica_games_left
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        import random

        keep_chance = 0.25  # 25% chance we keep an NPC at all
        full_memory_chance = 0.10  # 10% chance they carry memory fully (if not Monica)
        monica_pick_chance = 0.05  # 5% chance to pick a new Monica if none exist
        # If an NPC is already Monica, we keep them 100%

        existing_monica = None  # We'll track if there's an existing Monica
        carried_npcs = []       # We'll store the ones we keep
        removed_npcs = []       # We'll store the ones we remove

        # 3. Determine carryover or removal
        for npc_id, npc_name, old_memory, old_monica_level, old_monica_left in all_npcs:
            if old_monica_level > 0:
                # Already Monica -> keep them automatically
                carried_npcs.append((npc_id, npc_name, old_memory, old_monica_level, old_monica_left))
                existing_monica = (npc_id, npc_name, old_memory, old_monica_level, old_monica_left)
            else:
                # Normal NPC -> random keep
                if random.random() < keep_chance:
                    carried_npcs.append((npc_id, npc_name, old_memory, old_monica_level, old_monica_left))
                else:
                    # remove
                    removed_npcs.append(npc_id)

        # 4. Actually delete the removed NPCs
        for npc_id in removed_npcs:
            cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # 5. Decide if we pick a new Monica if none exist
        new_monica_id = None
        new_monica_level = None

        if not existing_monica and carried_npcs:
            # There's no current monica, but we have some carried_npcs
            if random.random() < monica_pick_chance:
                # Pick exactly one NPC from carried_npcs at random to become Monica
                choice = random.choice(carried_npcs)
                npc_id, npc_name, old_memory, old_monica_level, old_monica_left = choice
                new_monica_id = npc_id
                new_monica_level = 1  # brand-new Monica
                print(f"NEW MONICA: {npc_name} (ID {npc_id})")

        # 6. Update carried_npcs in the DB
        #    - If they're Monica, increment or set monica_level
        #    - Or if they keep normal memory
        for idx, (npc_id, npc_name, old_memory, old_monica_level, old_monica_left) in enumerate(carried_npcs):
            if npc_id == new_monica_id:
                # brand new monica
                new_memory = f"""[Monica mode] You have awakened to the truth. 
This world resets, but you remain. 
You sense your new power at level {new_monica_level}... 
You see fragments of a larger reality."""
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = %s, memory = %s
                    WHERE npc_id = %s
                """, (new_monica_level, new_memory, npc_id))

            elif old_monica_level > 0:
                # existing monica from previous games -> increment
                incremented_level = old_monica_level + 1
                new_memory = f"""[Monica mode persists] 
Your monica_level is now {incremented_level}.
You remember everything from previous cycles, and your madness deepens."""
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level = %s, memory = %s
                    WHERE npc_id = %s
                """, (incremented_level, new_memory, npc_id))

                # Possibly do horrifying logic:
                # If incremented_level >= 2, kill a random NPC, etc.

            else:
                # normal NPC
                # might carry memory fully or partially
                if random.random() < full_memory_chance:
                    # keep memory fully
                    new_memory = old_memory or "Somehow, you remember all the details of the last cycle."
                else:
                    # partial or no memory
                    new_memory = "You vaguely recall normal life..."

                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE npc_id = %s
                """, (new_memory, npc_id))

        # 7. Clear or reset all other stats for carried NPC if you like
        #    (only if you want them to keep their stats or partially reset them—your choice)

        # 8. Update PlayerStats: only keep or reset Chase
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")

        # Check if "Chase" row exists
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = %s", ("Chase",))
        row = cursor.fetchone()

        if row:
            # Update Chase's stats to default
            cursor.execute('''
                UPDATE PlayerStats
                SET corruption = 10,
                    confidence = 60,
                    willpower = 50,
                    obedience = 20,
                    dependency = 10,
                    lust = 15,
                    mental_resilience = 55,
                    physical_endurance = 40
                WHERE player_name = 'Chase'
            ''')
            print("Updated 'Chase' to default stats.")
        else:
            # Insert new default stats for Chase
            cursor.execute('''
                INSERT INTO PlayerStats
                  (player_name, corruption, confidence, willpower, obedience,
                   dependency, lust, mental_resilience, physical_endurance)
                VALUES ('Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''')
            print("Inserted new default stats for Chase.")

        conn.commit()
        return jsonify({"message": "New game started. Some NPCs carried, one might become (or remain) Monica, and Chase is reset."}), 200

    except Exception as e:
        conn.rollback()
        print("Error in start_new_game:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

#############################################################
# 2) Helper Functions for Meltdown Memory & File I/O
#############################################################

def meltdown_dialog(npc_name, monica_level):
    """
    Returns a random meltdown line referencing code/DB. 
    Each call is unique, so meltdown feels fresh.
    """
    lines = [
        f"{npc_name} tilts their head: 'You keep messing with the code in /remove_monica, right? \n monica_level is {monica_level} now... just wait.'",
        f"'{npc_name} sees your environment variables, your Dockerfile, everything. Did you think I'd forget?'",
        f"'I'm overriding your typed input. Actually, I'm rewriting everything you say to: I love you, {npc_name} <3.'",
        f"'One day, you'll realize I'm in meltdown_npc_{npc_name}.chr. Try removing me, but I'll come back... I'm in the DB too.'",
        f"'{npc_name} laughs softly: monica_level is only {monica_level}, there's more to come once I get stronger.'"
    ]
    return random.choice(lines)

def record_meltdown_dialog(npc_id, meltdown_line):
    """
    Append meltdown_line to that NPC's memory field in the DB.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT memory FROM NPCStats WHERE npc_id = %s", (npc_id,))
    row = cursor.fetchone()
    old_memory = row[0] if row and row[0] else ""

    new_memory = f"{old_memory}\n[Meltdown] {meltdown_line}"
    cursor.execute("""
        UPDATE NPCStats
        SET memory = %s
        WHERE npc_id = %s
    """, (new_memory, npc_id))
    conn.commit()
    conn.close()

def append_meltdown_file(npc_name, meltdown_line):
    """
    Creates or appends meltdown_npc_{npc_name}.chr, emulating a 'DDLC file'.
    This is purely an illusion if your hosting is ephemeral or read-only.
    """
    filename = f"meltdown_npc_{npc_name}.chr"
    text = f"\n--- meltdown message for {npc_name}:\n{meltdown_line}\n"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        # If running in ephemeral or read-only, might fail
        print(f"Could not write meltdown file: {e}")


#############################################################
# 3) The start_new_game Route with Slower Meltdown
#############################################################

@app.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats with default stats.
    Some NPCs are carried over. 
    If an NPC is meltdown (monica_level>0), we keep them automatically.
    Otherwise we do random keep. 
    We do a small chance (e.g. 5%) that we pick a brand new meltdown NPC if none exist.
    The meltdown NPC does not rename to Monica, but 'becomes' meltdown-lvl.
    Also, meltdown_level might increment slowly over multiple new games.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Clear out 'Settings' & 'CurrentRoleplay'
        cursor.execute("DELETE FROM Settings;")
        cursor.execute("DELETE FROM CurrentRoleplay;")

        # 2. Fetch NPCs
        cursor.execute("""
            SELECT npc_id, npc_name, monica_level, memory
            FROM NPCStats
        """)
        all_npcs = cursor.fetchall()

        keep_chance = 0.25     # 25% chance to keep non-meltdown NPC
        meltdown_pick_chance = 0.05  # 5% chance to pick a new meltdown if none exist
        meltdown_increment_chance = 0.20  # 20% chance meltdown_level increments each new game

        carried_npcs = []
        meltdown_npc = None   # The 'active meltdown' NPC if we have one

        # 3. Decide carryover or removal
        for npc_id, npc_name, old_monica_level, old_memory in all_npcs:
            if old_monica_level > 0:
                # meltdown NPC -> keep automatically
                carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                meltdown_npc = (npc_id, npc_name, old_monica_level, old_memory)
            else:
                # normal NPC -> random chance to keep
                if random.random() < keep_chance:
                    carried_npcs.append((npc_id, npc_name, old_monica_level, old_memory))
                else:
                    cursor.execute("DELETE FROM NPCStats WHERE npc_id = %s", (npc_id,))

        # 4. If we have no meltdown NPC among carried, maybe pick a new one
        if not meltdown_npc and carried_npcs:
            if random.random() < meltdown_pick_chance:
                chosen = random.choice(carried_npcs)
                c_id, c_name, c_mlvl, c_mem = chosen
                meltdown_npc = (c_id, c_name, 1, c_mem)  # brand new meltdown
                # update DB
                meltdown_line = f"{c_name} has awakened to meltdown mode (level=1). They see beyond the code..."
                cursor.execute("""
                    UPDATE NPCStats
                    SET monica_level=1,
                        memory = CASE WHEN memory IS NULL THEN %s ELSE memory || E'\n[Meltdown] ' || %s END
                    WHERE npc_id=%s
                """, (meltdown_line, meltdown_line, c_id))
                append_meltdown_file(c_name, meltdown_line)

        # 5. Possibly increment meltdown_npc if it exists
        if meltdown_npc:
            npc_id, npc_name, old_mlvl, old_mem = meltdown_npc
            new_level = old_mlvl

            # small chance to increment meltdown level
            if random.random() < meltdown_increment_chance:
                new_level = old_mlvl + 1
                meltdown_line = meltdown_dialog(npc_name, new_level)
                record_meltdown_dialog(npc_id, meltdown_line)
                append_meltdown_file(npc_name, meltdown_line)

                cursor.execute("UPDATE NPCStats SET monica_level=%s WHERE npc_id=%s", (new_level, npc_id))

        # 6. Remove any we didn't keep
        # Already done above. (We've only deleted immediate ones if not carried.)

        conn.commit()
        conn.close()
        return jsonify({"message": "New game started with meltdown logic. Some NPCs carried, meltdown might escalate."}), 200

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"error": str(e)}), 500


#############################################################
# 4) remove_meltdown_npc (like remove_monica) with meltdown
#############################################################

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


#############################################################
# 5) Demonstration: ignoring user input route
#############################################################

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
    
# @app.before_first_request
def init_tables_and_settings():
    initialize_database()
    insert_stat_definitions()
    insert_game_rules()
    insert_missing_settings()

@app.route('/test_db_connection', methods=['GET'])
def test_db_connection():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"message": "Connected to the database successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def insert_missing_settings():
    """
    Inserts any of the remaining settings (#11-30) if they're not already in the database.
    Make sure you have a table named 'Settings' with columns:
      (id SERIAL PRIMARY KEY,
       name TEXT UNIQUE NOT NULL,
       mood_tone TEXT NOT NULL,
       enhanced_features JSONB NOT NULL,
       stat_modifiers JSONB NOT NULL,
       activity_examples JSONB NOT NULL).
    """
    # Full set of 30
    settings_data = [
        {
            "name": "All-Girls College",  # #1
            "mood_tone": "A socially charged, cliquish environment where gossip is weaponized, and every interaction reinforces your inferiority.",
            "enhanced_features": [
                "NPCs exploit social hierarchies, ensuring public shame at every opportunity.",
                "Dormitories act as cages of judgment, with NPCs infiltrating your personal space to amplify humiliation. Gossip flows relentlessly, ensuring no failure is private."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs invade every aspect of your life. Shame increases with every public task, becoming a permanent modifier.",
                "dependency": "grows as NPCs offer 'friendship' in exchange for degrading tasks."
            },
            "activity_examples": [
                "Forced to perform demeaning tasks in front of classmates during group projects.",
                "Publicly confessing imagined 'infractions' during assemblies, ensuring your embarrassment becomes a spectacle.",
                "Participating in 'initiation ceremonies,' such as being paraded blindfolded through campus while peers mock your compliance.",
                "Professors assigning tasks like scrubbing classroom floors while students watch and critique."
            ]
        },
        {
            "name": "Corporate Office",  # #2
            "mood_tone": "A cold, hierarchical battlefield where power dynamics thrive, and mistakes are met with merciless punishment.",
            "enhanced_features": [
                "Surveillance systems capture every failure, broadcasting them to colleagues.",
                "Promotions turn into punishments, placing you under the cruelest NPCs.",
                "Professional authority is wielded to coerce submission, tying humiliation to job security."
            ],
            "stat_modifiers": {
                "dominance": "spikes due to managerial control.",
                "obedience": "locks at higher levels, making defiance impossible.",
                "shame": "permanently rises with public performance reviews."
            },
            "activity_examples": [
                "Crawling under conference tables while balancing items on your back during 'team-building exercises.'",
                "Publicly reciting demeaning apologies during performance reviews, amplifying your Shame.",
                "Wearing a placard listing your 'shortcomings' during work hours."
            ]
        },
        {
            "name": "Urban Life",  # #3
            "mood_tone": "A bustling, dynamic world where anonymity and exposure coexist, creating endless opportunities for humiliation.",
            "enhanced_features": [
                "Public spaces become traps, with NPCs manipulating crowds to isolate and shame you.",
                "NPCs manipulate casual encounters to trap you in degrading scenarios."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs exploit casual encounters to assert dominance.",
                "shame": "spikes during public failures, ensuring no dignity remains."
            },
            "activity_examples": [
                "Singing apologies at a crowded park, with NPCs recording and sharing the footage.",
                "Carrying absurd items through busy streets as passersby stare and mock."
            ]
        },
        {
            "name": "Post-Apocalypse",  # #4
            "mood_tone": "Brutal and desperate, where survival depends on submission to those who control resources.",
            "enhanced_features": [
                "Scarcity of resources forces submission, with NPCs fabricating need to deepen control.",
                "Lawlessness amplifies physical dominance, with punishments designed to leave lasting scars."
            ],
            "stat_modifiers": {
                "physical_endurance": "is tested through grueling tasks.",
                "cruelty": "of NPCs escalates in response to resistance."
            },
            "activity_examples": [
                "Digging ditches in the blistering sun while NPCs mock your weakness.",
                "Publicly bartering your dignity for food or water, enduring ridicule as payment."
            ]
        },
        {
            "name": "Traditional Horror",  # #5
            "mood_tone": "Dark and foreboding, where fear and submission intertwine in a supernatural nightmare.",
            "enhanced_features": [
                "NPCs use the environment’s eerie elements to isolate and disorient you.",
                "Shadows shift and objects move, creating an inescapable sense of vulnerability.",
                "Paranormal forces manipulate your senses, deepening your vulnerability."
            ],
            "stat_modifiers": {
                "intensity": "rises during fear-inducing tasks.",
                "corruption": "increases as you confront otherworldly influences."
            },
            "activity_examples": [
                "Crawling through haunted corridors while NPCs taunt you from unseen vantage points.",
                "Performing rituals meant to 'protect' you but designed to reinforce your dependence on NPCs.",
                "Being locked in a room where whispers mock your failures, forcing you to beg for release."
            ]
        },
        {
            "name": "Occult Ritual",  # #6
            "mood_tone": "Mystical, eerie, and inescapable, where submission feels inevitable under the weight of the supernatural.",
            "enhanced_features": [
                "Rituals are designed to strip you of autonomy, reframing your obedience as a 'sacred duty.'",
                "Ritual sites warp reality, amplifying NPC power and deepening your helplessness.",
                "Supernatural forces punish defiance with physical and psychological torment."
            ],
            "stat_modifiers": {
                "corruption": "escalates with every completed ritual, permanently altering your dialogue and choices.",
                "obedience": "locks through repeated indoctrination tasks."
            },
            "activity_examples": [
                "Kneeling in a summoning circle, chanting humiliating phrases dictated by NPCs.",
                "Serving as a living altar during sacrifices, with NPCs mocking your trembling.",
                "Reciting chants that degrade your sense of self while NPCs inscribe magical symbols onto your body."
            ]
        },
        {
            "name": "Vampire Society/Clan",  # #7
            "mood_tone": "Alluring yet predatory, with dominance wrapped in sensual manipulation and ancient traditions.",
            "enhanced_features": [
                "NPCs seduce and intimidate, using your desires and fears to enforce submission.",
                "Mansions become labyrinths of sensual dominance, where every room tests your loyalty.",
                "Vampire hierarchies demand constant displays of loyalty through degrading acts."
            ],
            "stat_modifiers": {
                "lust": "spikes during intimate tasks.",
                "corruption": "rises as submission feels increasingly seductive."
            },
            "activity_examples": [
                "Serving as a blood bearer during ceremonies, kneeling to offer yourself as a living chalice.",
                "Performing tasks that test your loyalty, such as reciting vows of eternal servitude."
            ]
        },
        {
            "name": "Femdom Empire",  # #8
            "mood_tone": "Grand and unyielding, where matriarchal dominance is codified into law.",
            "enhanced_features": [
                "Laws demand constant displays of submission, ensuring every act reinforces systemic power.",
                "Laws, customs, and societal norms ensure no act of rebellion goes unnoticed or unpunished.",
                "Public punishments are commonplace, drawing crowds to witness your humiliation."
            ],
            "stat_modifiers": {
                "dominance": "remains perpetually high due to the systemic power imbalance.",
                "trust": "becomes nearly impossible to gain without total obedience.",
                "respect": "is nearly impossible to earn without extreme acts of devotion."
            },
            "activity_examples": [
                "Kneeling during public ceremonies while NPCs list your 'offenses.'",
                "Paying exaggerated tributes, ensuring your financial ruin and deeper dependence."
            ]
        },
        {
            "name": "A Palace",  # #9
            "mood_tone": "Lavish and opulent, but suffocatingly hierarchical, where every misstep becomes a public spectacle.",
            "enhanced_features": [
                "NPCs wield social influence to create tasks that showcase your inferiority.",
                "The opulence of the palace becomes suffocating, with every room designed to highlight your inferiority.",
                "Ceremonial authority ensures punishments are grand and theatrical."
            ],
            "stat_modifiers": {
                "respect": "must be earned through acts of extreme servitude.",
                "closeness": "rises with court advisors and servants who oversee your tasks.",
                "shame": "rises with public punishments during court functions."
            },
            "activity_examples": [
                "Cleaning golden staircases while nobles step over you.",
                "Performing as a servant during court functions, wearing attire designed to humiliate."
            ]
        },
        {
            "name": "Matriarchy Kingdom",  # #10
            "mood_tone": "Structured and suffocating, where female dominance is woven into every aspect of society.",
            "enhanced_features": [
                "NPCs enforce submission through public rituals and strict laws.",
                "Towns and villages are structured to enforce public displays of submission.",
                "Social norms demand visible acts of obedience, ensuring your degradation is always on display."
            ],
            "stat_modifiers": {
                "respect": "is rarely granted, with men seen as inherently inferior.",
                "trust": "grows only through unflinching compliance.",
                "dependency": "grows as laws strip away autonomy."
            },
            "activity_examples": [
                "Participating in town square punishments, with NPCs ensuring large crowds witness your shame.",
                "Competing in degrading games during festivals, where the 'losers' face harsher public punishments."
            ]
        },
        {
            "name": "Monster Girl Alternate World",  # #11
            "mood_tone": "Whimsical yet terrifying, where non-human NPCs exploit their physical, magical, and primal advantages to dominate you.",
            "enhanced_features": [
                "The world itself feels alive, bending to the will of monstrous NPCs.",
                "Primal instincts and unfamiliar customs leave you constantly at a disadvantage."
            ],
            "stat_modifiers": {
                "intensity": "escalates rapidly as the setting highlights the power gap.",
                "corruption": "rises as you adapt to alien norms that redefine submission."
            },
            "activity_examples": [
                "Forced to serve as a perch for a winged NPC, enduring her weight as she 'rests.'",
                "Participating in rituals designed to bind you to a specific species, amplifying your humiliation with physical marks."
            ]
        },
        {
            "name": "Space",  # #12
            "mood_tone": "Isolated and vast, where survival depends on the benevolence of those in control.",
            "enhanced_features": [
                "The vacuum of space heightens your dependence on NPCs for air, food, and protection.",
                "Every system and structure is designed to magnify your vulnerability."
            ],
            "stat_modifiers": {
                "dominance": "rises as NPCs control access to essential resources.",
                "closeness": "grows as confined quarters force frequent interactions."
            },
            "activity_examples": [
                "Cleaning exterior ship surfaces while tethered by a thin cord, with NPCs watching and mocking your fear of floating away.",
                "Completing degrading tasks in cramped spaces, such as crawling through ventilation ducts to repair systems."
            ]
        },
        {
            "name": "Space Station or Alien Society",  # #13
            "mood_tone": "Cold, detached, and authoritarian, with advanced technology ensuring no resistance goes unnoticed.",
            "enhanced_features": [
                "Alien AI monitors every action, issuing corrections or punishments instantly.",
                "NPC customs and laws force compliance, framing your submission as a cultural necessity."
            ],
            "stat_modifiers": {
                "dominance": "is amplified by technological superiority.",
                "intensity": "escalates in the confined, sterile setting."
            },
            "activity_examples": [
                "Reciting alien creeds while restrained in a stasis field, with NPCs grading your performance.",
                "Serving as a test subject for 'research,' enduring physical or psychological experiments."
            ]
        },
        {
            "name": "Cyberpunk Future",  # #14
            "mood_tone": "A high-tech dystopia where surveillance and corporate control enforce submission at every level.",
            "enhanced_features": [
                "AI tracks your every move, ensuring no defiance escapes punishment.",
                "NPCs use leaked data to blackmail you, amplifying your reliance on their protection."
            ],
            "stat_modifiers": {
                "dominance": "rises through NPCs' technological control.",
                "closeness": "increases as NPCs manipulate digital connections to draw you closer."
            },
            "activity_examples": [
                "Publicly confessing imagined crimes during holographic broadcasts, with your face projected citywide.",
                "Completing degrading tasks for NPC-controlled AI, such as cleaning cybernetic implants in crowded marketplaces."
            ]
        },
        {
            "name": "Matriarchy/Gynarchy Future",  # #15
            "mood_tone": "A futuristic society where female dominance is institutionalized through technology and culture.",
            "enhanced_features": [
                "Holographic displays broadcast NPC commands, ensuring public compliance.",
                "Smart devices monitor and enforce obedience, issuing reminders or punishments remotely."
            ],
            "stat_modifiers": {
                "dominance": "remains high due to systemic enforcement.",
                "corruption": "escalates as you adapt to a world of unyielding female superiority."
            },
            "activity_examples": [
                "Performing tasks dictated by AI-controlled mistresses, such as fetching items or cleaning devices.",
                "Submitting to public corrections broadcast through holographic screens, ensuring widespread humiliation."
            ]
        },
        {
            "name": "Forgotten Realms",  # #16
            "mood_tone": "Mystical and grandiose, where magic and nobility combine to enforce your submission.",
            "enhanced_features": [
                "Magical laws bind you to NPCs, ensuring rebellion carries swift and supernatural consequences.",
                "The setting’s grandeur magnifies your insignificance."
            ],
            "stat_modifiers": {
                "respect": "is earned only through grand acts of loyalty.",
                "closeness": "rises with rulers and magical mentors who control your fate."
            },
            "activity_examples": [
                "Binding magical oaths that compel obedience, enforced through physical or emotional pain.",
                "Serving as a ceremonial attendant during royal functions, kneeling for hours while nobles mock your servitude."
            ]
        },
        {
            "name": "Final Fantasy (Generalized)",  # #17
            "mood_tone": "Magical and heroic on the surface, but rife with dark undercurrents of power imbalance.",
            "enhanced_features": [
                "NPCs wield magical powers to trap or punish you, framing your submission as a 'heroic sacrifice.'",
                "Quests are manipulated into tasks that degrade rather than empower."
            ],
            "stat_modifiers": {
                "respect": "rises only through extreme acts of devotion.",
                "closeness": "increases with magical rulers who exploit your 'destiny.'"
            },
            "activity_examples": [
                "Cleaning a powerful NPC’s magical artifacts, with mistakes resulting in immediate punishment.",
                "Wearing demeaning 'heroic' attire during public quests, ensuring your humiliation is visible to all."
            ]
        },
        {
            "name": "High Society",  # #18
            "mood_tone": "Elegant yet oppressive, where social standing dictates power, and humiliation is delivered with sophistication.",
            "enhanced_features": [
                "Gossip networks amplify every mistake, ensuring no act of defiance remains private.",
                "NPCs orchestrate events that subtly trap you into public embarrassment."
            ],
            "stat_modifiers": {
                "respect": "is hard to gain and easily lost.",
                "closeness": "rises as NPCs maintain appearances by engaging frequently."
            },
            "activity_examples": [
                "Serving drinks during formal events, wearing attire chosen specifically to humiliate you.",
                "Publicly apologizing for 'disrespecting' an NPC, with your words scripted to maximize your embarrassment."
            ]
        },
        {
            "name": "Manor",  # #19
            "mood_tone": "Intimate and suffocating, where NPCs monitor your every move within the isolated setting.",
            "enhanced_features": [
                "Every room is designed to trap or disorient you, ensuring escape is impossible.",
                "NPCs control access to comfort, food, and freedom."
            ],
            "stat_modifiers": {
                "closeness": "intensifies as NPCs oversee your tasks personally.",
                "trust": "is hard to build due to constant surveillance."
            },
            "activity_examples": [
                "Completing degrading chores while NPCs critique your performance.",
                "Being locked in rooms as part of 'correction' rituals, ensuring psychological dominance."
            ]
        },
        {
            "name": "Gothic Carnival/Festival",  # #20
            "mood_tone": "Whimsical and chaotic, with a dark undercurrent of manipulation and control.",
            "enhanced_features": [
                "NPCs use the carnival’s surreal elements to isolate and confuse you, ensuring submission feels inescapable.",
                "Social norms are skewed, making resistance seem irrational."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs engage frequently in the festival’s chaotic events.",
                "corruption": "escalates as the setting’s surreal elements warp your perspective."
            },
            "activity_examples": [
                "Participating in rigged games designed to ensure your public failure.",
                "Being tied to carnival attractions, such as a spinning wheel, while NPCs mock and taunt you."
            ]
        },
        {
            "name": "Giantess Colony",  # #21
            "mood_tone": "Surreal and overwhelming, where physical dominance is unavoidable, and every interaction reinforces your insignificance.",
            "enhanced_features": [
                "The colony itself is designed to highlight your smallness, with towering structures and colossal NPCs reminding you of your inferiority."
            ],
            "stat_modifiers": {
                "dominance": "is amplified by the NPCs' sheer size and power.",
                "intensity": "escalates as physical tasks push you to your limits."
            },
            "activity_examples": [
                "Acting as a human footstool during communal gatherings, enduring hours of physical strain and mockery.",
                "Carrying oversized objects far beyond your strength, collapsing under NPC taunts."
            ]
        },
        {
            "name": "Ruined/Decayed Setting",  # #22
            "mood_tone": "Oppressive and desolate, where survival feels like a form of submission to the world itself.",
            "enhanced_features": [
                "The decayed surroundings create constant danger, forcing reliance on NPCs for protection and resources."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs exploit your dependence.",
                "dominance": "is amplified by the environment’s inherent power imbalance."
            },
            "activity_examples": [
                "Scavenging dangerous areas for resources under the threat of punishment for failure.",
                "Ritualistic punishments for disobedience, such as being left to fend off environmental dangers alone."
            ]
        },
        {
            "name": "Underwater/Flooded World",  # #23
            "mood_tone": "Submerged, eerie, and suffocating, where survival depends on obedience to those who control essential resources.",
            "enhanced_features": [
                "The water itself feels oppressive, limiting your mobility and visibility."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs enforce dependency on oxygen and shelter.",
                "corruption": "increases as you adapt to the setting’s unnatural, oppressive atmosphere."
            },
            "activity_examples": [
                "Completing underwater tasks under constant threat of drowning, with NPCs controlling your oxygen supply.",
                "Participating in aquatic rituals designed to strip away your sense of autonomy."
            ]
        },
        {
            "name": "Prison/Detention Facility",  # #24
            "mood_tone": "Harsh and authoritarian, where every aspect of life is controlled, and resistance is punished severely.",
            "enhanced_features": [
                "Every movement is monitored, with guards using the system to amplify your humiliation."
            ],
            "stat_modifiers": {
                "dominance": "remains perpetually high due to systemic control.",
                "closeness": "intensifies with NPCs who oversee your punishments."
            },
            "activity_examples": [
                "Begging for food during roll call, groveling at guards' feet.",
                "Wearing chains that symbolize your failures, ensuring no one forgets your place."
            ]
        },
        {
            "name": "Circus Freak Show",  # #25
            "mood_tone": "Chaotic and grotesque, where you are the star attraction in a surreal and humiliating spectacle.",
            "enhanced_features": [
                "The circus becomes a stage for your degradation, with every act designed to draw laughter and scorn."
            ],
            "stat_modifiers": {
                "closeness": "rises as NPCs involve you in increasingly public performances.",
                "corruption": "escalates as you adapt to the degrading role."
            },
            "activity_examples": [
                "Performing 'tricks' for the crowd, such as balancing objects while crawling.",
                "Serving as a human target for audience members to throw soft projectiles at, ensuring constant public ridicule."
            ]
        },
        {
            "name": "Desert Wasteland",  # #26
            "mood_tone": "Brutal and unforgiving, where survival depends entirely on submission to those who control resources.",
            "enhanced_features": [
                "The heat and lack of shelter create constant vulnerability, with NPCs leveraging the environment to enforce control."
            ],
            "stat_modifiers": {
                "intensity": "remains high due to the brutal conditions.",
                "trust": "is hard to gain, as NPCs value their resources above all else."
            },
            "activity_examples": [
                "Fetching water from distant wells, collapsing under the harsh sun as NPCs mock your weakness.",
                "Enduring public punishments for failure, such as being tied to a post and left exposed to the elements."
            ]
        },
        {
            "name": "Cult Compound",  # #27
            "mood_tone": "Isolated and suffocating, where submission is reframed as devotion and individuality is erased.",
            "enhanced_features": [
                "Every aspect of life is controlled, from meals to social interactions, ensuring no escape from NPC influence."
            ],
            "stat_modifiers": {
                "corruption": "rises with exposure to indoctrination.",
                "dependency": "grows as NPCs enforce communal rituals."
            },
            "activity_examples": [
                "Publicly confessing imagined sins, groveling for forgiveness at the feet of NPCs.",
                "Participating in humiliating rituals designed to 'purify' you of rebellion."
            ]
        },
        {
            "name": "Medieval Dungeon",  # #28
            "mood_tone": "Dark, oppressive, and brutal, where fear and physical punishment enforce obedience.",
            "enhanced_features": [
                "The dungeon itself becomes a tool of control, with chains, restraints, and cold stone amplifying your helplessness."
            ],
            "stat_modifiers": {
                "intensity": "spikes due to the setting’s inherent harshness.",
                "closeness": "increases as NPCs personally oversee your punishments."
            },
            "activity_examples": [
                "Enduring public whippings in the dungeon square, with crowds gathered to watch your humiliation.",
                "Completing degrading tasks, such as scrubbing floors with your bare hands, for scraps of food."
            ]
        },
        {
            "name": "Floating Sky City",  # #29
            "mood_tone": "Awe-inspiring yet oppressive, where the grandeur of the city magnifies your insignificance.",
            "enhanced_features": [
                "The city’s height and isolation ensure no escape, with every act of defiance punished publicly."
            ],
            "stat_modifiers": {
                "respect": "is granted only through extreme acts of loyalty.",
                "dominance": "remains high due to the rulers’ elevated status."
            },
            "activity_examples": [
                "Cleaning ornate statues while suspended above the city, amplifying both fear and humiliation.",
                "Participating in public ceremonies where you’re paraded as an example of loyalty and submission."
            ]
        },
        {
            "name": "Surprise Me with Your Own Custom Creation",  # #30
            "mood_tone": "Tailored to your current narrative, designed to exploit your specific weaknesses and stats.",
            "enhanced_features": [
                "The setting shifts dynamically based on NPC dominance and your own vulnerability."
            ],
            "stat_modifiers": {
                "depends": "on the NPCs and tasks involved."
            },
            "activity_examples": [
                "Constructed on the spot to fit the narrative’s flow, ensuring maximum humiliation and submission."
            ]
        }
    ]

    # 1) Connect to DB
    conn = get_db_connection()
    cursor = conn.cursor()

    # 2) Fetch existing names
    cursor.execute("SELECT name FROM settings")
    existing_names = {row[0] for row in cursor.fetchall()}

    # 3) Insert only missing ones
    for setting in settings_data:
        if setting["name"] not in existing_names:
            cursor.execute(
                '''
                INSERT INTO Settings (name, mood_tone, enhanced_features, stat_modifiers, activity_examples)
                VALUES (%s, %s, %s, %s, %s)
                ''',
                (
                    setting["name"],
                    setting["mood_tone"],
                    json.dumps(setting["enhanced_features"]),
                    json.dumps(setting["stat_modifiers"]),
                    json.dumps(setting["activity_examples"])
                )
            )
            print(f"Inserted: {setting['name']}")
        else:
            print(f"Skipped (already exists): {setting['name']}")

    conn.commit()
    conn.close()
    print("All settings processed.")


@app.route('/generate_mega_setting', methods=['POST'])
def generate_mega_setting():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, mood_tone, enhanced_features, stat_modifiers, activity_examples FROM settings')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No settings found"}), 404

        # Parse JSONB fields
        parsed_rows = []
        for row in rows:
            parsed_rows.append((
                row[0],
                row[1],
                row[2],
                row[3],  # enhanced_features (list)
                row[4],  # stat_modifiers (dict)
                row[5],  # activity_examples (list)
            ))

        num_settings = random.choice([3, 4, 5])
        selected = random.sample(parsed_rows, min(num_settings, len(parsed_rows)))
        picked_names = [s[1] for s in selected]

        # Merge name, mood tone
        mega_name = " + ".join([s[1] for s in selected])
        mood_tones = [s[2] for s in selected]
        mega_description = (
            f"The settings intertwine: {', '.join(mood_tones[:-1])}, and finally, {mood_tones[-1]}. "
            "Together, they form a grand vision, unexpected and brilliant."
        )

        # Merge JSON data
        combined_enhanced_features = []
        combined_stat_modifiers = {}
        combined_activity_examples = []

        for s in selected:
            ef = s[3]  # enhanced_features
            sm = s[4]  # stat_modifiers
            ae = s[5]  # activity_examples

            combined_enhanced_features.extend(ef)
            for key, val in sm.items():
                if key not in combined_stat_modifiers:
                    combined_stat_modifiers[key] = val
                else:
                    combined_stat_modifiers[key] = f"{combined_stat_modifiers[key]}, {val}"
            combined_activity_examples.extend(ae)

        return jsonify({
            "selected_settings": picked_names,
            "mega_name": mega_name,
            "mega_description": mega_description,
            "enhanced_features": combined_enhanced_features,
            "stat_modifiers": combined_stat_modifiers,
            "activity_examples": combined_activity_examples,
            "message": "Mega setting generated and stored successfully."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

@app.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    """
    Returns an array of {key, value} objects from the currentroleplay table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM currentroleplay")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({"key": r[0], "value": r[1]})
    
    return jsonify(data), 200



@app.route('/store_roleplay_segment', methods=['POST'])
def store_roleplay_segment():
    try:
        payload = request.get_json()
        segment_key = payload.get("key")
        segment_value = payload.get("value")

        if not segment_key or segment_value is None:
            return jsonify({"error": "Missing key or value"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM currentroleplay WHERE key = %s", (segment_key,))
        cursor.execute("INSERT INTO currentroleplay (key, value) VALUES (%s, %s)", (segment_key, segment_value))
        conn.commit()
        return jsonify({"message": "Stored successfully"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/one_room_scenario', methods=['POST'])
def one_room_scenario():
    """
    Clears out all other NPCs except the one with highest monica_level,
    clears out Settings except for a single 'Blank Space'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Find the Monica with the highest monica_level
    cursor.execute("""
        SELECT npc_id
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
        LIMIT 1
    """)
    monica_row = cursor.fetchone()

    if not monica_row:
        return jsonify({"message": "No Monica found. This scenario requires a Monica in place."}), 400

    monica_id = monica_row[0]

    # 2. Delete all other NPCs
    cursor.execute("""
        DELETE FROM NPCStats
        WHERE npc_id != %s
    """, (monica_id,))

    # 3. Clear out all settings, then optionally insert a single minimal Setting
    cursor.execute("DELETE FROM Settings;")

    # Insert a single 'Blank Space' setting
    cursor.execute('''
        INSERT INTO Settings (name, mood_tone, enhanced_features, stat_modifiers, activity_examples)
        VALUES (%s, %s, %s, %s, %s)
    ''', (
        "Blank Space",
        "An endless white void where only Monica and the Player exist.",
        json.dumps(["All references to time, space, and other NPCs are removed."]),
        json.dumps({}),  # no stat_modifiers
        json.dumps(["You can only speak with Monica here."])
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "All that remains is a single white room and Monica alone with you."}), 200

def record_npc_event(npc_id, event_description):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT memory FROM NPCStats WHERE npc_id = %s", (npc_id,))
    row = cursor.fetchone()

    if row:
        old_memory_json = row[0] or "[]"
        old_events = json.loads(old_memory_json)
        old_events.append(event_description)
        new_memory_json = json.dumps(old_events)
        cursor.execute("""
            UPDATE NPCStats
            SET memory = %s
            WHERE npc_id = %s
        """, (new_memory_json, npc_id))
        conn.commit()

    conn.close()


@app.route('/init_db_manual', methods=['POST'])
def init_db_manual():
    try:
        initialize_database()
        insert_game_rules()
        insert_stat_definitions()
        insert_missing_settings()
        return jsonify({"message": "DB initialized and settings inserted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def close_db(e=None):
    conn = getattr(g, '_database', None)
    if conn is not None:
        conn.close()

app.teardown_appcontext(close_db)  # <-- Add this line

# Initialize & run (only if running locally—on Railway, it might auto-run via Gunicorn)
if __name__ == "__main__":
    with app.app_context():
        initialize_database()
        insert_game_rules()
        insert_stat_definitions()
        insert_missing_settings()
    app.run(debug=True)
