"""
logic/rule_enforcement.py

This module demonstrates:
1. Retrieving Player or NPC stats from the DB.
2. Parsing condition strings (like "Lust > 90 or Dependency > 80").
3. Evaluating those conditions with the relevant stats.
4. Applying the effect if the condition is True.
5. A small Flask Blueprint route to show how you'd trigger the enforcement.
6. Hybrid approach with advanced effects, intensity tiers, GPT generation, meltdown synergy.
"""

from quart import Blueprint, request, jsonify
from db.connection import get_db_connection_context
import json, random, asyncio
import logging
import asyncpg

"""
# If meltdown logic or memory logic are used
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog
from logic.memory_logic import store_roleplay_segment
"""
rule_enforcement_bp = Blueprint("rule_enforcement_bp", __name__)
logger = logging.getLogger(__name__)

############################
# 1. Retrieve Stats Helpers
############################

async def get_player_stats(player_name="Chase"):
    """
    Returns a dictionary of the player's relevant stats.
    Example:
      {
        "Corruption": 95,
        "Dependency": 85,
        "Obedience": 81,
        ...
      }
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, dependency,
                       lust, mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE player_name = $1
            """, player_name)

        if not row:
            return {}

        return {
            "Corruption": row["corruption"],
            "Confidence": row["confidence"],
            "Willpower": row["willpower"],
            "Obedience": row["obedience"],
            "Dependency": row["dependency"],
            "Lust": row["lust"],
            "Mental Resilience": row["mental_resilience"],
            "Physical Endurance": row["physical_endurance"],
        }
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error fetching player stats: {db_err}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching player stats: {e}", exc_info=True)
        return {}


async def get_npc_stats(npc_name):
    """
    If you want to check an NPC's stats to evaluate certain rules
    like "Cruelty > 50" or "Dominance > 80", etc.
    Returns something like:
    {
      "Dominance": 60,
      "Cruelty": 75,
      "Closeness": 40,
      "Trust": 0,
      "Respect": 10,
      "Intensity": 25
    }
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT dominance, cruelty, closeness, trust, respect, intensity
                FROM NPCStats
                WHERE npc_name = $1
            """, npc_name)

        if not row:
            return {}

        return {
            "Dominance": row["dominance"],
            "Cruelty": row["cruelty"],
            "Closeness": row["closeness"],
            "Trust": row["trust"],
            "Respect": row["respect"],
            "Intensity": row["intensity"]
        }
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error fetching NPC stats: {db_err}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching NPC stats: {e}", exc_info=True)
        return {}


############################
# 2. Parsing Condition
############################

def parse_condition(condition_str):
    """
    Naively splits on ' and ' or ' or ' to produce a structure like:
        ("OR", [("Lust", ">", 90), ("Dependency", ">", 80)])
    or
        ("AND", [("Corruption", ">", 90), ("Obedience", ">", 80)])
    or
        ("SINGLE", [("Lust", ">", 90)])
    """
    cond = condition_str.strip().lower()

    if " and " in cond:
        logic_op = "AND"
        parts = cond.split(" and ")
    elif " or " in cond:
        logic_op = "OR"
        parts = cond.split(" or ")
    else:
        logic_op = "SINGLE"
        parts = [cond]

    parsed_list = []
    for part in parts:
        tokens = part.strip().split()
        if len(tokens) == 3:
            stat_name, operator, threshold_str = tokens
            stat_name = stat_name.title()  # e.g. 'lust' => 'Lust'
            try:
                threshold = int(threshold_str)
            except:
                threshold = 0
            parsed_list.append((stat_name, operator, threshold))
        else:
            print(f"Warning: condition part '{part}' not recognized.")
    return (logic_op, parsed_list)


def evaluate_condition(logic_op, parsed_conditions, stats_dict):
    """
    Evaluate the parsed conditions with the given stats_dict, e.g.:
        logic_op = "OR"
        parsed_conditions = [("Lust", ">", 90), ("Dependency", ">", 80)]
        stats_dict = {"Lust": 95, "Dependency": 70, ...}

    Return True or False.
    """
    results = []
    for (stat_name, operator, threshold) in parsed_conditions:
        actual_value = stats_dict.get(stat_name, 0)

        if operator == ">":
            outcome = (actual_value > threshold)
        elif operator == ">=":
            outcome = (actual_value >= threshold)
        elif operator == "<":
            outcome = (actual_value < threshold)
        elif operator == "<=":
            outcome = (actual_value <= threshold)
        elif operator == "==":
            outcome = (actual_value == threshold)
        else:
            outcome = False

        results.append(outcome)

    if logic_op == "AND":
        return all(results)
    elif logic_op == "OR":
        return any(results)
    elif logic_op == "SINGLE":
        return results[0] if results else False
    return False


############################
# 3. Applying Effects (Hybrid Approach)
############################

async def apply_effect(effect_str, player_name, npc_id=None):
    """
    Parse and apply a stat or game effect.
    For example: 'You gain +10 Corruption' or 'NPC becomes openly hateful'
    
    Now uses NPC-specific intensity rather than global plot intensity.
    """
    from logic.chatgpt_integration import get_openai_client
    result = {
        "message": "",
        "statUpdates": {},
        "effectApplied": effect_str,
        "npcUsed": npc_id
    }
    
    effect_lower = effect_str.lower()
    
    try:
        async with get_db_connection_context() as conn:
            # 1) Basic DB updates
            if effect_lower.startswith("locks independent choices"):
                # Example: raise Obedience to at least 80
                obedience = await conn.fetchval("""
                    UPDATE PlayerStats
                    SET obedience = GREATEST(obedience, 80)
                    WHERE player_name=$1
                    RETURNING obedience
                """, player_name)
                if obedience:
                    result["statUpdates"]["obedience"] = obedience

            elif effect_lower.startswith("total compliance"):
                # Force Obedience=100
                obedience = await conn.fetchval("""
                    UPDATE PlayerStats
                    SET obedience=100
                    WHERE player_name=$1
                    RETURNING obedience
                """, player_name)
                if obedience:
                    result["statUpdates"]["obedience"] = obedience

            elif effect_lower.startswith("npc cruelty intensifies") and npc_id is not None:
                # Raise cruelty by 10
                cruelty = await conn.fetchval("""
                    UPDATE NPCStats
                    SET cruelty = LEAST(cruelty + 10, 100)
                    WHERE npc_id=$1
                    RETURNING cruelty
                """, npc_id)
                if cruelty:
                    result["statUpdates"]["npc_cruelty"] = cruelty

            elif effect_lower.startswith("collaborative physical punishments"):
                # Possibly handle multiple NPC synergy, etc.
                pass

            # 2) Determine intensity tier based on the specific NPC's intensity value
            # Get the NPC's intensity if npc_id is provided
            npc_intensity = None
            npc_name = None
            if npc_id is not None:
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, intensity
                    FROM NPCStats
                    WHERE npc_id=$1
                """, npc_id)
                if npc_row:
                    npc_name = npc_row["npc_name"]
                    npc_intensity = npc_row["intensity"]
                    result["npcName"] = npc_name
                    result["npcIntensity"] = npc_intensity
            
            # Determine intensity tier based on NPC's intensity (if available)
            intensity_range = (0, 30)  # Default to lowest tier
            
            if npc_intensity is not None:
                # Use NPC's specific intensity level
                if npc_intensity >= 90:
                    intensity_range = (90, 100)
                elif npc_intensity >= 60:
                    intensity_range = (60, 90)
                elif npc_intensity >= 30:
                    intensity_range = (30, 60)
                else:
                    intensity_range = (0, 30)
            else:
                # Fallback to player stats if no NPC is provided
                p_row = await conn.fetchrow("""
                    SELECT corruption, obedience
                    FROM PlayerStats
                    WHERE player_name=$1
                """, player_name)
                
                if p_row:
                    corr, obed = p_row["corruption"], p_row["obedience"]
                    if corr >= 90 or obed >= 90:
                        intensity_range = (90, 100)
                    elif corr >= 60 or obed >= 60:
                        intensity_range = (60, 90)
                    elif corr >= 30 or obed >= 30:
                        intensity_range = (30, 60)
                    else:
                        intensity_range = (0, 30)

            # Query IntensityTiers
            row = await conn.fetchrow("""
                SELECT tier_name, key_features, activity_examples, permanent_effects
                FROM IntensityTiers
                WHERE range_min = $1 AND range_max = $2
            """, intensity_range[0], intensity_range[1])
            
            chosen_intensity_tier = None
            if row:
                tier_name, key_features, activity_examples, permanent_effects = row["tier_name"], row["key_features"], row["activity_examples"], row["permanent_effects"]
                result["intensityTier"] = tier_name
                result["intensitySource"] = "NPC-specific" if npc_intensity is not None else "Player-derived"
            else:
                tier_name = "Unknown Intensity"
                key_features, activity_examples, permanent_effects = "[]", "[]", "{}"

            # Possibly pick an example from activity_examples
            if activity_examples and isinstance(activity_examples, str):
                try:
                    ex_list = json.loads(activity_examples)
                    if ex_list:
                        chosen_intensity_tier = random.choice(ex_list)
                except:
                    chosen_intensity_tier = None

            # 3) If effect says "no defiance possible," treat it like endgame
            if "no defiance possible" in effect_lower:
                rows = await conn.fetch("""
                    SELECT title, examples
                    FROM PlotTriggers
                    WHERE stage='Endgame'
                """)
                if rows:
                    chosen = random.choice(rows)
                    t_title, ex_json = chosen["title"], chosen["examples"]
                    ex_list = json.loads(ex_json) if ex_json else []
                    picked_example = random.choice(ex_list) if ex_list else "(No example found)"
                    endgame_line = f"[ENDGAME TRIGGER] {t_title}: {picked_example}"
                    result["plotTriggerEvent"] = endgame_line

                    # store in memory
                    # Replace with async version:
                    # await store_roleplay_segment({"key": "EndgameTrigger", "value": endgame_line})
                    pass

            if "punishment" in effect_lower:
                system_prompt = f"""
                You are a punishment scenario generator.
                The player's stats suggest intensity tier: {result.get('intensityTier', 'unknown')}.
                Provide a short brand-new humiliating scenario for effect: '{effect_str}'.
                """
                try:
                    # Acquire the client from your chatgpt_integration module
                    client = get_openai_client()
            
                    # Now call the ChatCompletion endpoint via client.chat.completions.create
                    gpt_resp = await client.chat.completions.create(
                        model="gpt-4.1-nano",  # or whichever model you're using
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "Generate a unique punishment scenario."}
                        ],
                        max_tokens=200,
                        temperature=1.0
                    )
                    scenario = gpt_resp.choices[0].message.content.strip()
                    result["punishmentScenario"] = scenario
            
                except Exception as e:
                    result["punishmentScenario"] = f"(GPT error: {e})"

            # 6) If we found an intensity tier example but didn't do GPT scenario, use it
            if chosen_intensity_tier and "punishmentScenario" not in result:
                result["punishmentScenario"] = f"(From IntensityTier) {chosen_intensity_tier}"

            # 7) Memory log
            mem_text = f"Effect triggered: {effect_str}. (Intensity: {result.get('intensityTier', 'unknown')})"
            # Replace with async version:
            # await store_roleplay_segment({"key": f"Effect_{player_name}", "value": mem_text})
            result["memoryLog"] = mem_text

        return result
    except Exception as e:
        logger.error(f"Error applying effect: {e}", exc_info=True)
        return result


############################
# 4. Putting It All Together
############################

async def enforce_all_rules_on_player(player_name="Chase"):
    """
    1. Get the player's stats
    2. Retrieve all rules from GameRules
    3. Parse + evaluate each condition
    4. If true => apply the effect
    5. Return a summary of what got triggered
    """
    try:
        # Load player stats for condition checks
        player_stats = await get_player_stats(player_name)

        async with get_db_connection_context() as conn:
            rules = await conn.fetch("SELECT condition, effect FROM GameRules")

        triggered_effects = []
        for row in rules:
            condition_str, effect_str = row["condition"], row["effect"]
            logic_op, parsed_conditions = parse_condition(condition_str)
            result_bool = evaluate_condition(logic_op, parsed_conditions, player_stats)
            if result_bool:
                # It's True => apply
                # Note: if you need an NPC ID for certain effects, pass it here
                outcome = await apply_effect(effect_str, player_name)
                triggered_effects.append({
                    "condition": condition_str,
                    "effect": effect_str,
                    "outcome": outcome
                })

        return triggered_effects
    except Exception as e:
        logger.error(f"Error enforcing rules: {e}", exc_info=True)
        return []


############################
# 5. OPTIONAL: Blueprint Route
############################

@rule_enforcement_bp.route("/enforce_rules", methods=["POST"])
async def enforce_rules_route():
    """
    Example route to manually trigger rules enforcement.
    JSON param: {"player_name": "..."} optional
    """
    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")

    triggered = await enforce_all_rules_on_player(player_name)
    if not triggered:
        return jsonify({"message": "No rules triggered."}), 200

    return jsonify({
        "message": f"Enforced rules for player '{player_name}'.",
        "triggered": triggered
    }), 200
