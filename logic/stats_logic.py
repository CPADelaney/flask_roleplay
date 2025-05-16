# logic/stats_logic.py

import logging
import asyncio
import asyncpg
import json
import random
from typing import Dict, Any, List
from db.connection import get_db_connection_context
from quart import Blueprint, request, jsonify, session, redirect, url_for  

stats_bp = Blueprint('stats_bp', __name__)

############################
# 1. Inserting & Updating Game Rules
############################

async def insert_or_update_game_rules():
    """
    Inserts or updates game rules into the GameRules table.
    If a rule_name already exists, we update the condition/effect.
    Otherwise we insert a new record.
    """
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

    async with get_db_connection_context() as conn:
        for rule in rules_data:
            await conn.execute("""
                INSERT INTO GameRules (rule_name, condition, effect)
                VALUES ($1, $2, $3)
                ON CONFLICT (rule_name)
                DO UPDATE SET condition=EXCLUDED.condition,
                          effect=EXCLUDED.effect
            """, rule["rule_name"], rule["condition"], rule["effect"])

    print("Game rules inserted or updated successfully.")


############################
# 2. Insert Stat Definitions (NPC & Player)
############################

async def insert_stat_definitions():
    """
    Inserts all NPC and Player stat definitions (from your 'Stat Dynamics Knowledge Document')
    into the StatDefinitions table, if they don't already exist.
    """
    # NPC Stats
    npc_stats_data = [
        {
            "stat_name": "Dominance",
            "range_min": 0,
            "range_max": 100,
            "definition": "Measures the NPC's control over you.",
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
            "definition": "Reflects the NPC's sadism and creativity in tormenting you.",
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
            "definition": "Indicates the NPC's belief in your reliability or loyalty.",
            "effects": """60+: Unlocks deeper, personal interactions.
-50 or below: Triggers suspicion, manipulation, or sabotage.""",
            "progression": """Increases: Obedience, loyalty, honesty.
Decreases: Failure, betrayal, competing loyalties."""
        },
        {
            "stat_name": "Respect",
            "range_min": -100,
            "range_max": 100,
            "definition": "Reflects the NPC's perception of your competence or value.",
            "effects": """60+: Treated as a prized asset.
-50 or below: Treated with disdain or open contempt.""",
            "progression": """Increases: Successes, sacrifice, or loyalty.
Decreases: Failures, incompetence, or reinforcing inferiority."""
        },
        {
            "stat_name": "Intensity",
            "range_min": 0,
            "range_max": 100,
            "definition": "Represents the severity of the NPC's actions.",
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

    async with get_db_connection_context() as conn:
        # Insert NPC stat definitions
        for npc_stat in npc_stats_data:
            await conn.execute("""
                INSERT INTO StatDefinitions
                  (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (stat_name) DO NOTHING
            """, 
                "NPC",
                npc_stat["stat_name"],
                npc_stat["range_min"],
                npc_stat["range_max"],
                npc_stat["definition"],
                npc_stat["effects"],
                npc_stat["progression"]
            )

        # Insert Player stat definitions
        for p_stat in player_stats_data:
            await conn.execute("""
                INSERT INTO StatDefinitions
                  (scope, stat_name, range_min, range_max, definition, effects, progression_triggers)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (stat_name) DO NOTHING
            """, 
                "Player",
                p_stat["stat_name"],
                p_stat["range_min"],
                p_stat["range_max"],
                p_stat["definition"],
                p_stat["effects"],
                p_stat["progression"]
            )

    print("All stat definitions inserted or skipped if already present.")


############################
# 3. Default Player Stats
############################

async def insert_default_player_stats_chase(user_id, conversation_id, provided_conn=None):
    """
    Insert row for 'Chase' in PlayerStats, scoping by user_id + conversation_id.
    """
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

    # Use the provided connection if available, otherwise get a new one
    if provided_conn:
        conn = provided_conn
        # Use the existing connection/transaction
        row = await conn.fetchrow("""
            SELECT id FROM PlayerStats
            WHERE user_id=$1
              AND conversation_id=$2
              AND player_name=$3
        """, user_id, conversation_id, chase_stats["player_name"])

        if row:
            print("Default stats for Chase already exist in this user/conversation. Skipping insert.")
        else:
            await conn.execute("""
                INSERT INTO PlayerStats (
                    user_id, conversation_id,
                    player_name,
                    corruption, confidence, willpower,
                    obedience, dependency, lust,
                    mental_resilience, physical_endurance
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                user_id, conversation_id,
                chase_stats["player_name"],
                chase_stats["corruption"], chase_stats["confidence"], chase_stats["willpower"],
                chase_stats["obedience"], chase_stats["dependency"], chase_stats["lust"],
                chase_stats["mental_resilience"], chase_stats["physical_endurance"]
            )
            print(f"Inserted default stats for Chase in user={user_id}, conv={conversation_id}")
    else:
        # Create our own connection context if none was provided
        async with get_db_connection_context() as conn:
            # Check if a row already exists for these user+conversation+player
            row = await conn.fetchrow("""
                SELECT id FROM PlayerStats
                WHERE user_id=$1
                  AND conversation_id=$2
                  AND player_name=$3
            """, user_id, conversation_id, chase_stats["player_name"])

            if row:
                print("Default stats for Chase already exist in this user/conversation. Skipping insert.")
            else:
                await conn.execute("""
                    INSERT INTO PlayerStats (
                        user_id, conversation_id,
                        player_name,
                        corruption, confidence, willpower,
                        obedience, dependency, lust,
                        mental_resilience, physical_endurance
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    user_id, conversation_id,
                    chase_stats["player_name"],
                    chase_stats["corruption"], chase_stats["confidence"], chase_stats["willpower"],
                    chase_stats["obedience"], chase_stats["dependency"], chase_stats["lust"],
                    chase_stats["mental_resilience"], chase_stats["physical_endurance"]
                )
                print(f"Inserted default stats for Chase in user={user_id}, conv={conversation_id}")


############################
# 4. Player Stats Route (Needs user_id/conversation_id Scoping)
############################

@stats_bp.route('/player/<player_name>', methods=['GET', 'PUT'])
async def handle_player_stats(player_name):
    """
    GET => returns the player's stats for user_id, conversation_id, and player_name
    PUT => updates them
    We'll read user_id from session, conversation_id from query (GET) or JSON (PUT)
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if request.method == 'GET':
        # conversation_id from query param
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, 
                       dependency, lust, mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
            """, user_id, conversation_id, player_name)

            if not row:
                return jsonify({"error": f"PlayerStats not found for user={user_id}, conv={conversation_id}, name={player_name}"}), 404

            stats_response = {
                "corruption": row['corruption'],
                "confidence": row['confidence'],
                "willpower": row['willpower'],
                "obedience": row['obedience'],
                "dependency": row['dependency'],
                "lust": row['lust'],
                "mental_resilience": row['mental_resilience'],
                "physical_endurance": row['physical_endurance']
            }
            return jsonify(stats_response), 200

    elif request.method == 'PUT':
        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
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
            async with get_db_connection_context() as conn:
                result = await conn.execute("""
                    UPDATE PlayerStats
                    SET corruption=$1,
                        confidence=$2,
                        willpower=$3,
                        obedience=$4,
                        dependency=$5,
                        lust=$6,
                        mental_resilience=$7,
                        physical_endurance=$8
                    WHERE user_id=$9 AND conversation_id=$10
                      AND player_name=$11
                    RETURNING corruption, confidence, willpower, obedience,
                              dependency, lust, mental_resilience, physical_endurance
                """, 
                    corruption, confidence, willpower, obedience, dependency,
                    lust, mental, physical,
                    user_id, conversation_id, player_name
                )
                
                # Check if any rows were affected
                if result == "UPDATE 0":
                    return jsonify({"error": f"PlayerStats not found for user={user_id}, conv={conversation_id}, name={player_name}"}), 404
                
                # Get the updated row
                updated_row = await conn.fetchrow("""
                    SELECT corruption, confidence, willpower, obedience, 
                           dependency, lust, mental_resilience, physical_endurance
                    FROM PlayerStats
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
                """, user_id, conversation_id, player_name)
                
                updated_stats = {
                    "corruption": updated_row['corruption'],
                    "confidence": updated_row['confidence'],
                    "willpower": updated_row['willpower'],
                    "obedience": updated_row['obedience'],
                    "dependency": updated_row['dependency'],
                    "lust": updated_row['lust'],
                    "mental_resilience": updated_row['mental_resilience'],
                    "physical_endurance": updated_row['physical_endurance']
                }
                
                return jsonify({
                    "message": f"Player '{player_name}' stats updated for user={user_id}, conversation={conversation_id}.",
                    "new_stats": updated_stats
                }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


############################
# 5. NPC Stats Route (Needs user_id/conversation_id Scoping)
############################

@stats_bp.route('/npc/<int:npc_id>', methods=['GET', 'PUT'])
async def handle_npc_stats(npc_id):
    """
    GET => fetch the NPC's stats for (user_id, conversation_id, npc_id)
    PUT => update them
    We'll read user_id from session, conversation_id from query or JSON
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if request.method == 'GET':
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT dominance, cruelty, closeness, trust, respect, intensity
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
            """, user_id, conversation_id, npc_id)

            if not row:
                return jsonify({"error": f"NPCStats not found for user={user_id}, conv={conversation_id}, npc_id={npc_id}"}), 404

            npc_response = {
                "dominance": row['dominance'],
                "cruelty": row['cruelty'],
                "closeness": row['closeness'],
                "trust": row['trust'],
                "respect": row['respect'],
                "intensity": row['intensity']
            }
            return jsonify(npc_response), 200

    elif request.method == 'PUT':
        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        # Extract new stats
        dominance = data.get('dominance', 0)
        cruelty   = data.get('cruelty', 0)
        closeness = data.get('closeness', 0)
        trust     = data.get('trust', 0)
        respect   = data.get('respect', 0)
        intensity = data.get('intensity', 0)

        try:
            async with get_db_connection_context() as conn:
                result = await conn.execute("""
                    UPDATE NPCStats
                    SET dominance=$1,
                        cruelty=$2,
                        closeness=$3,
                        trust=$4,
                        respect=$5,
                        intensity=$6
                    WHERE user_id=$7 AND conversation_id=$8 AND npc_id=$9
                """, 
                    dominance, cruelty, closeness, trust, respect, intensity,
                    user_id, conversation_id, npc_id
                )
                
                # Check if any rows were affected
                if result == "UPDATE 0":
                    return jsonify({"error": f"NPCStats not found for user={user_id}, conv={conversation_id}, npc_id={npc_id}"}), 404
                
                # Get the updated row
                updated_npc = await conn.fetchrow("""
                    SELECT dominance, cruelty, closeness, trust, respect, intensity
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """, user_id, conversation_id, npc_id)

                updated_stats = {
                    "dominance": updated_npc['dominance'],
                    "cruelty": updated_npc['cruelty'],
                    "closeness": updated_npc['closeness'],
                    "trust": updated_npc['trust'],
                    "respect": updated_npc['respect'],
                    "intensity": updated_npc['intensity']
                }
                
                return jsonify({
                    "message": f"NPC with id={npc_id} updated for user={user_id}, conv={conversation_id}.",
                    "new_stats": updated_stats
                }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


############################
# 6. Combined Init Endpoints
############################

@stats_bp.route('/init_stats_and_rules', methods=['POST'])
async def init_stats_system():
    """
    An endpoint to insert stat definitions and game rules in one go.
    Returns a simple JSON with 'stat_definitions' and 'game_rules' counts if needed.
    """
    try:
        await insert_stat_definitions()
        await insert_or_update_game_rules()
        return jsonify({
            "message": "Stat system initialized",
            "stat_definitions": 8,  # Or dynamically count from DB if you wish
            "game_rules": 4         # Same here
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def update_player_stats(user_id, conversation_id, player_name, stat_key, value):
    """
    Updates a specific stat for a given player in the PlayerStats table.
    """
    try:
        async with get_db_connection_context() as conn:
            # Note: Using parameterized SQL dynamically with column names requires special handling
            # This is a simplistic approach - ideally validate stat_key against allowed column names
            query = f"UPDATE PlayerStats SET {stat_key} = $1 WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4"
            result = await conn.execute(query, value, user_id, conversation_id, player_name)
            
            # Parse the result string (e.g., "UPDATE 1")
            affected = 0
            if result and result.startswith("UPDATE"):
                try:
                    affected = int(result.split()[1])
                except (IndexError, ValueError):
                    pass
                    
            return {"success": affected > 0, "affected_rows": affected}
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error updating player stats: {db_err}", exc_info=True)
        return {"success": False, "error": str(db_err)}
    except Exception as e:
        logging.error(f"Error updating player stats: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def check_social_link_milestones(user_id, conversation_id, player_name="Chase"):
    """
    Example function referencing social links. If you do user+conversation scoping,
    you'd incorporate them here. 
    Currently just uses a broad 'player is involved' approach.
    """
    from logic.social_links import add_link_event
    
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT link_id, entity1_type, entity1_id,
                   entity2_type, entity2_id,
                   link_type, link_level
            FROM SocialLinks
            WHERE (entity1_type='player' OR entity2_type='player')
            AND user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)

        for row in rows:
            link_id = row['link_id']
            link_type = row['link_type']
            link_level = row['link_level']
            
            # Example triggers
            if link_type == "friends" and link_level >= 30:
                await conn.execute("""
                    INSERT INTO PlayerInventory (player_name, item_name, quantity)
                    VALUES ($1, $2, 1)
                    ON CONFLICT (player_name, item_name) DO NOTHING
                """, player_name, "Friendship Token")
                await add_link_event(link_id, "RewardGiven: Friendship Token")

            if link_type == "enslave" and link_level >= 40:
                await conn.execute("""
                    INSERT INTO PlayerInventory (player_name, item_name, quantity)
                    VALUES ($1, $2, 1)
                    ON CONFLICT (player_name, item_name) DO NOTHING
                """, player_name, "Sweaty Socks")
                await add_link_event(link_id, "RewardGiven: Sweaty Socks")

@stats_bp.route("/init_stats_and_rules", methods=["POST"])
async def init_stats_and_rules():
    """
    Another endpoint to run all insert/update logic at once:
      1) Insert game rules
      2) Insert stat definitions
      3) Insert default stats for Chase
    """
    try:
        await insert_or_update_game_rules()
        await insert_stat_definitions()
        
        # Get user_id and conversation_id from request data
        data = await request.get_json() or {}
        user_id = data.get("user_id", session.get("user_id"))
        conversation_id = data.get("conversation_id")
        
        if user_id and conversation_id:
            await insert_default_player_stats_chase(user_id, conversation_id)
            return jsonify({"message": "Stats & Rules initialized successfully."}), 200
        else:
            return jsonify({"message": "Game rules and stat definitions initialized. No player stats created due to missing user_id or conversation_id."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define stat threshold tiers that trigger different game behaviors
STAT_THRESHOLDS = {
    "corruption": [
        {"level": 30, "name": "Curious", "behaviors": ["occasional_submission", "light_teasing"]},
        {"level": 50, "name": "Accepting", "behaviors": ["regular_submission", "moderate_humiliation", "voluntary_service"]},
        {"level": 70, "name": "Craving", "behaviors": ["eager_submission", "heavy_humiliation", "ritual_service"]},
        {"level": 90, "name": "Devoted", "behaviors": ["complete_submission", "public_humiliation", "permanent_servitude"]}
    ],
    "obedience": [
        {"level": 30, "name": "Compliant", "behaviors": ["follows_direct_orders", "minimal_resistance"]},
        {"level": 50, "name": "Dutiful", "behaviors": ["anticipates_orders", "apologizes_for_hesitation"]},
        {"level": 70, "name": "Conditioned", "behaviors": ["automatic_obedience", "discomfort_when_disobeying"]},
        {"level": 90, "name": "Programmed", "behaviors": ["cannot_disobey", "distress_without_orders"]}
    ],
    "dependency": [
        {"level": 30, "name": "Attached", "behaviors": ["seeks_approval", "mild_separation_anxiety"]},
        {"level": 50, "name": "Reliant", "behaviors": ["frequent_check-ins", "moderate_separation_anxiety"]},
        {"level": 70, "name": "Dependent", "behaviors": ["constant_presence", "severe_separation_anxiety"]},
        {"level": 90, "name": "Addicted", "behaviors": ["panic_when_alone", "physical_symptoms_when_separated"]}
    ],
    # Add similar tiers for other stats
}

# Define combinations of stats that trigger special states
STAT_COMBINATIONS = [
    {
        "name": "Willing Prisoner",
        "description": "Player actively seeks confinement and restriction",
        "conditions": {"corruption": 60, "obedience": 70, "dependency": 50},
        "behaviors": ["requests_restraints", "thanks_for_control", "anxiety_without_boundaries"]
    },
    {
        "name": "Breaking Point",
        "description": "Player exhibits signs of psychological fracturing",
        "conditions": {"mental_resilience": -10, "confidence": -20, "willpower": -30},
        "behaviors": ["disassociation", "fugue_states", "confused_identity"]
    },
    {
        "name": "Stockholm Syndrome",
        "description": "Player has formed unhealthy attachments to controlling NPCs",
        "conditions": {"dependency": 80, "corruption": 60, "mental_resilience": -40},
        "behaviors": ["defends_abusers", "reinterprets_cruelty_as_love", "fears_rescue"]
    },
    # Add more complex combinations
]

async def get_player_current_tier(user_id, conversation_id, stat_name):
    """
    Determine which tier a player is in for a given stat
    Returns the threshold dict with level, name and behaviors
    """
    async with get_db_connection_context() as conn:
        # Use parametrized query safely with column name validation
        valid_stat_columns = ["corruption", "confidence", "willpower", "obedience", 
                              "dependency", "lust", "mental_resilience", "physical_endurance"]
        
        if stat_name not in valid_stat_columns:
            return None
            
        query = f"SELECT {stat_name} FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'"
        row = await conn.fetchrow(query, user_id, conversation_id)
        
        if not row:
            return None
        
        stat_value = row[stat_name]
        thresholds = STAT_THRESHOLDS.get(stat_name, [])
        
        # Find the highest threshold the player meets
        current_tier = None
        for tier in thresholds:
            if stat_value >= tier["level"]:
                current_tier = tier
            else:
                break
                
        return current_tier

async def check_for_combination_triggers(user_id, conversation_id):
    """
    Check if player stats trigger any special combination states
    Returns a list of triggered combinations
    """
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow("""
            SELECT corruption, confidence, willpower, obedience, dependency, 
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
        
        if not row:
            return []
            
        player_stats = {
            "corruption": row['corruption'],
            "confidence": row['confidence'],
            "willpower": row['willpower'],
            "obedience": row['obedience'],
            "dependency": row['dependency'],
            "lust": row['lust'],
            "mental_resilience": row['mental_resilience'],
            "physical_endurance": row['physical_endurance']
        }
        
        triggered_combinations = []
        
        for combo in STAT_COMBINATIONS:
            meets_all_conditions = True
            for stat_name, required_value in combo["conditions"].items():
                if player_stats.get(stat_name, 0) < required_value:
                    meets_all_conditions = False
                    break
                    
            if meets_all_conditions:
                triggered_combinations.append(combo)
                
        return triggered_combinations

async def record_stat_change_event(user_id, conversation_id, stat_name, old_value, new_value, cause=""):
    """
    Record significant stat changes in a new StatsHistory table
    """
    # Only record significant changes (more than 5 points)
    if abs(new_value - old_value) < 5:
        return {"success": True, "recorded": False, "reason": "Change too small"}
        
    try:
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO StatsHistory 
                (user_id, conversation_id, player_name, stat_name, old_value, new_value, cause, timestamp)
                VALUES ($1, $2, 'Chase', $3, $4, $5, $6, CURRENT_TIMESTAMP)
            """, user_id, conversation_id, stat_name, old_value, new_value, cause)
            
            return {"success": True, "recorded": True}
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error recording stat change: {db_err}", exc_info=True)
        return {"success": False, "error": str(db_err)}
    except Exception as e:
        logging.error(f"Error recording stat change: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
        
async def apply_stat_change(user_id, conversation_id, changes, cause=""):
    """
    Apply multiple stat changes at once and record the history
    
    Example:
    changes = {
        "corruption": +5,
        "willpower": -3,
        "confidence": -2
    }
    """
    try:
        async with get_db_connection_context() as conn:
            # First get current values
            row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, dependency, 
                       lust, mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)
            
            if not row:
                return {"success": False, "error": "Player stats not found"}
                
            current_stats = {
                "corruption": row['corruption'],
                "confidence": row['confidence'],
                "willpower": row['willpower'],
                "obedience": row['obedience'],
                "dependency": row['dependency'],
                "lust": row['lust'],
                "mental_resilience": row['mental_resilience'],
                "physical_endurance": row['physical_endurance']
            }
            
            # Process each stat change
            for stat_name, change in changes.items():
                if stat_name not in current_stats:
                    continue
                    
                old_value = current_stats[stat_name]
                new_value = max(0, min(100, old_value + change))  # Clamp between 0-100
                
                # Update the stat
                # Note: Using parameterized SQL dynamically with column names requires special handling
                # This is a simplistic approach - ideally validate stat_name against allowed column names
                valid_stat_columns = ["corruption", "confidence", "willpower", "obedience", 
                                     "dependency", "lust", "mental_resilience", "physical_endurance"]
                                     
                if stat_name not in valid_stat_columns:
                    continue
                    
                query = f"UPDATE PlayerStats SET {stat_name} = $1 WHERE user_id=$2 AND conversation_id=$3 AND player_name='Chase'"
                await conn.execute(query, new_value, user_id, conversation_id)
                
                # Record the change
                await record_stat_change_event(user_id, conversation_id, stat_name, old_value, new_value, cause)
            
            return {"success": True, "changes_applied": len(changes)}
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error applying stat changes: {db_err}", exc_info=True)
        return {"success": False, "error": str(db_err)}
    except Exception as e:
        logging.error(f"Error applying stat changes: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Define specific activities that modify stats
ACTIVITY_EFFECTS = {
    "public_humiliation": {
        "corruption": +3,
        "confidence": -5,
        "obedience": +7,
        "mental_resilience": -3
    },
    "praise_and_reward": {
        "dependency": +5,
        "corruption": +2,
        "confidence": +3
    },
    "punishment": {
        "obedience": +8,
        "willpower": -4,
        "physical_endurance": +2
    },
    "isolation": {
        "dependency": +10,
        "mental_resilience": -6,
        "willpower": -3
    },
    "training_session": {
        "obedience": +6,
        "corruption": +4,
        "physical_endurance": +5
    },
    "service_task": {
        "obedience": +3,
        "corruption": +2,
        "dependency": +3
    },
    "resistance_attempt": {
        "willpower": +5,
        "confidence": +3,
        "obedience": -4,
        "corruption": -2
    },
    # Add more activities with effects
}

async def apply_activity_effects(user_id, conversation_id, activity_name, intensity=1.0):
    """
    Apply stat changes based on a specific activity
    Intensity multiplier can increase or decrease the effect
    """
    if activity_name not in ACTIVITY_EFFECTS:
        return {"success": False, "error": "Unknown activity"}
        
    # Get the base effects and scale by intensity
    base_effects = ACTIVITY_EFFECTS[activity_name]
    scaled_effects = {stat: change * intensity for stat, change in base_effects.items()}
    
    # Apply the changes
    return await apply_stat_change(
        user_id, conversation_id, 
        scaled_effects, 
        f"Activity: {activity_name} (intensity: {intensity})"
    )
