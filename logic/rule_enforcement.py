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

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
import json, random
import openai  # Only needed if you're calling GPT from here
"""
# If meltdown logic or memory logic are used
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog
from logic.memory_logic import store_roleplay_segment
"""
rule_enforcement_bp = Blueprint("rule_enforcement_bp", __name__)

############################
# 1. Retrieve Stats Helpers
############################

def get_player_stats(player_name="Chase"):
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT corruption, confidence, willpower, obedience, dependency,
               lust, mental_resilience, physical_endurance
        FROM PlayerStats
        WHERE player_name = %s
    """, (player_name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    return {
        "Corruption": row[0],
        "Confidence": row[1],
        "Willpower": row[2],
        "Obedience": row[3],
        "Dependency": row[4],
        "Lust": row[5],
        "Mental Resilience": row[6],
        "Physical Endurance": row[7],
    }


def get_npc_stats(npc_name):
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dominance, cruelty, closeness, trust, respect, intensity
        FROM NPCStats
        WHERE npc_name = %s
    """, (npc_name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    return {
        "Dominance": row[0],
        "Cruelty": row[1],
        "Closeness": row[2],
        "Trust": row[3],
        "Respect": row[4],
        "Intensity": row[5]
    }


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

def apply_effect(effect_str, player_name, npc_id=None):
    """
    Applies a rule effect or punishment scenario, integrating:
      - Stats DB updates (Obedience, Cruelty, etc.)
      - Intensity tier lookups (based on relevant stats)
      - Optionally a brand-new GPT-generated punishment scenario
      - PlotTriggers references if we detect "endgame" or major conditions
      - Light meltdown synergy if meltdown is triggered
      - Logging in CurrentRoleplay or NPC memory

    Returns a dict describing what happened, e.g.:
        {
          "appliedEffect": "...",
          "statUpdates": {...},
          "intensityTier": "High Intensity (60-90)",
          "punishmentScenario": "...some text...",
          "plotTriggerEvent": "...some endgame scenario text...",
          "meltdownLine": "... meltdown text ...",
          "memoryLog": "...",
        }
    """
    result = {
        "appliedEffect": effect_str,
        "statUpdates": {},
        "intensityTier": None,
        "punishmentScenario": None,
        "plotTriggerEvent": None,
     #   "meltdownLine": None,
        "memoryLog": None,
    }

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        effect_lower = effect_str.lower()

        # 1) Basic DB updates
        if effect_lower.startswith("locks independent choices"):
            # Example: raise Obedience to at least 80
            cursor.execute("""
                UPDATE PlayerStats
                SET obedience = GREATEST(obedience, 80)
                WHERE player_name=%s
                RETURNING obedience
            """, (player_name,))
            row = cursor.fetchone()
            if row:
                result["statUpdates"]["obedience"] = row[0]

        elif effect_lower.startswith("total compliance"):
            # Force Obedience=100
            cursor.execute("""
                UPDATE PlayerStats
                SET obedience=100
                WHERE player_name=%s
                RETURNING obedience
            """, (player_name,))
            row = cursor.fetchone()
            if row:
                result["statUpdates"]["obedience"] = row[0]

        elif effect_lower.startswith("npc cruelty intensifies") and npc_id is not None:
            # Raise cruelty by 10
            cursor.execute("""
                UPDATE NPCStats
                SET cruelty = LEAST(cruelty + 10, 100)
                WHERE npc_id=%s
                RETURNING cruelty
            """, (npc_id,))
            row = cursor.fetchone()
            if row:
                result["statUpdates"]["npc_cruelty"] = row[0]

        elif effect_lower.startswith("collaborative physical punishments"):
            # Possibly handle multiple NPC synergy, etc.
            pass

        # 2) Determine an intensity tier from player's stats
        cursor.execute("""
            SELECT corruption, obedience, willpower, confidence, dependency, lust
            FROM PlayerStats
            WHERE player_name=%s
        """, (player_name,))
        p_row = cursor.fetchone()

        chosen_intensity_tier = None
        if p_row:
            (corr, obed, willp, conf, dep, lust) = p_row
            # A naive approach:
            if corr >= 90 or obed >= 90:
                intensity_range = (90, 100)
            elif corr >= 60 or obed >= 60:
                intensity_range = (60, 90)
            elif corr >= 30 or obed >= 30:
                intensity_range = (30, 60)
            else:
                intensity_range = (0, 30)

            # Query IntensityTiers
            cursor.execute("""
                SELECT tier_name, key_features, activity_examples, permanent_effects
                FROM IntensityTiers
                WHERE range_min = %s AND range_max = %s
            """, (intensity_range[0], intensity_range[1]))
            row = cursor.fetchone()
            if row:
                tier_name, key_features, activity_examples, permanent_effects = row
                result["intensityTier"] = tier_name
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
            cursor.execute("""
                SELECT title, examples
                FROM PlotTriggers
                WHERE stage='Endgame'
            """)
            rows = cursor.fetchall()
            if rows:
                chosen = random.choice(rows)
                t_title, ex_json = chosen
                ex_list = json.loads(ex_json) if ex_json else []
                picked_example = random.choice(ex_list) if ex_list else "(No example found)"
                endgame_line = f"[ENDGAME TRIGGER] {t_title}: {picked_example}"
                result["plotTriggerEvent"] = endgame_line

                # store in memory
                store_roleplay_segment({"key": "EndgameTrigger", "value": endgame_line})

        # 4) GPT-based scenario if "punishment" is in effect text
        if "punishment" in effect_lower:
            system_prompt = f"""
            You are a punishment scenario generator. 
            The player's stats suggest intensity tier: {result['intensityTier']}.
            Provide a short brand-new humiliating scenario for effect: '{effect_str}'.
            """
            try:
                gpt_resp = openai.ChatCompletion.create(
                    model="gpt-4",
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
"""
        # 5) meltdown synergy if meltdown triggered
        if "meltdown" in effect_lower:
            meltdown_line = meltdown_dialog_gpt("EasterEggNPC", 2)
            record_meltdown_dialog(npc_id or 999, meltdown_line)
            result["meltdownLine"] = meltdown_line
"""
        # 6) If we found an intensity tier example but didn't do GPT scenario, use it
        if chosen_intensity_tier and not result["punishmentScenario"]:
            result["punishmentScenario"] = f"(From IntensityTier) {chosen_intensity_tier}"

        # 7) Memory log
        mem_text = f"Effect triggered: {effect_str}. (Intensity: {result['intensityTier']})"
        store_roleplay_segment({"key": f"Effect_{player_name}", "value": mem_text})
        result["memoryLog"] = mem_text

        conn.commit()

    except Exception as e:
        conn.rollback()
        result["error"] = str(e)
    finally:
        conn.close()

    return result


############################
# 4. Putting It All Together
############################

def enforce_all_rules_on_player(player_name="Chase"):
    """
    1. Get the player's stats
    2. Retrieve all rules from GameRules
    3. Parse + evaluate each condition
    4. If true => apply the effect
    5. Return a summary of what got triggered
    """
    # Load player stats for condition checks
    player_stats = get_player_stats(player_name)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT condition, effect FROM GameRules")
    rules = cursor.fetchall()
    conn.close()

    triggered_effects = []
    for (condition_str, effect_str) in rules:
        logic_op, parsed_conditions = parse_condition(condition_str)
        result_bool = evaluate_condition(logic_op, parsed_conditions, player_stats)
        if result_bool:
            # It's True => apply
            # Note: if you need an NPC ID for certain effects, pass it here
            outcome = apply_effect(effect_str, player_name)
            triggered_effects.append({
                "condition": condition_str,
                "effect": effect_str,
                "outcome": outcome
            })

    return triggered_effects


############################
# 5. OPTIONAL: Blueprint Route
############################

@rule_enforcement_bp.route("/enforce_rules", methods=["POST"])
def enforce_rules_route():
    """
    Example route to manually trigger rules enforcement.
    JSON param: {"player_name": "..."} optional
    """
    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")

    triggered = enforce_all_rules_on_player(player_name)
    if not triggered:
        return jsonify({"message": "No rules triggered."}), 200

    return jsonify({
        "message": f"Enforced rules for player '{player_name}'.",
        "triggered": triggered
    }), 200
