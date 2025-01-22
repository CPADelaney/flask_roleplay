"""
logic/rule_enforcement.py

This module demonstrates:
1. Retrieving Player or NPC stats from the DB.
2. Parsing condition strings (like "Lust > 90 or Dependency > 80").
3. Evaluating those conditions with the relevant stats.
4. Applying the effect if the condition is True.
5. A small Flask Blueprint route to show how you'd trigger the enforcement.

You can register this Blueprint in your main app.py (or main.py) to test
the logic. Alternatively, you can remove the Blueprint and just call
`enforce_all_rules_on_player(...)` from your code whenever needed.
"""

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection

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
    # Standardize spacing/casing
    cond = condition_str.strip()

    # Detect if we have ' and ' or ' or '
    if " and " in cond.lower():
        logic_op = "AND"
        parts = cond.lower().split(" and ")
    elif " or " in cond.lower():
        logic_op = "OR"
        parts = cond.lower().split(" or ")
    else:
        logic_op = "SINGLE"
        parts = [cond.lower()]

    parsed_list = []
    for part in parts:
        # e.g. part = "lust > 90"
        tokens = part.strip().split()
        if len(tokens) == 3:
            stat_name, operator, threshold_str = tokens
            # Convert to Title case so it matches our stats dictionary keys
            stat_name = stat_name.title()
            # Try to parse as int
            # (If you need float, do float(threshold_str))
            try:
                threshold = int(threshold_str)
            except:
                threshold = 0  # fallback

            parsed_list.append((stat_name, operator, threshold))
        else:
            # Possibly handle errors or ignore
            print(f"Warning: condition part '{part}' not recognized.")
            # We skip this one or you can throw an exception

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
# 3. Applying Effects
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

    Parameters:
      effect_str (str): The textual effect from a triggered rule (e.g., "Total compliance; no defiance possible").
      player_name (str): Which player is affected.
      npc_id (int or None): If an NPC is specifically involved, pass their ID.

    Returns:
      dict: Contains keys describing what happened, e.g.:
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
        "meltdownLine": None,
        "memoryLog": None,
    }

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Basic checks on effect_str → do DB updates
        effect_lower = effect_str.lower()

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
            # Force Obedience=100, remove defiance
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
            # Possibly do closeness or intensity bumps on multiple NPCs
            # We'll keep it minimal here
            pass

        # 2) Check relevant stats to determine an "intensity" for this punishment
        #    We'll default to using either the player's stats or the NPC's intensity.
        #    Let's pick the player's stats for an overall read.
        cursor.execute("""
            SELECT corruption, obedience, willpower, confidence, dependency, lust
            FROM PlayerStats
            WHERE player_name = %s
        """, (player_name,))
        player_row = cursor.fetchone()

        chosen_intensity_tier = None
        if player_row:
            (corr, obed, willp, conf, dep, lust) = player_row
            # A naive approach: if Obedience > 80 or Corruption > 70 => high intensity
            # If either is near 100 => max intensity
            # else moderate/low
            # Real logic can be more advanced.
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
                # fallback if not found
                tier_name = "Unknown Intensity"
                key_features, activity_examples, permanent_effects = "[]", "[]", "{}"

            # Possibly pick an example from activity_examples
            if activity_examples and isinstance(activity_examples, str):
                # It's probably a JSON string
                try:
                    ex_list = json.loads(activity_examples)
                    if ex_list:
                        chosen_intensity_tier = random.choice(ex_list)
                except:
                    chosen_intensity_tier = None

        # 3) If effect_str implies "no defiance possible," treat it like endgame
        if "no defiance possible" in effect_lower:
            cursor.execute("""
                SELECT title, examples
                FROM PlotTriggers
                WHERE stage='Endgame'
            """)
            rows = cursor.fetchall()
            if rows:
                chosen = random.choice(rows)
                title, ex_json = chosen
                ex_list = json.loads(ex_json) if ex_json else []
                picked_example = random.choice(ex_list) if ex_list else "No example found"
                endgame_line = f"[ENDGAME TRIGGER] {title}: {picked_example}"
                result["plotTriggerEvent"] = endgame_line

                # store in memory
                store_roleplay_segment({
                    "key": "EndgameTrigger",
                    "value": endgame_line
                })

        # 4) Optionally call GPT for a brand-new scenario
        #    We'll only do this if "punishment" is in the effect text, just as an example
        punishment_scenario = None
        if "punishment" in effect_lower:
            # A quick GPT call
            # Provide context about intensity or stats so GPT can generate something new
            # (Replace with your actual GPT config)
            system_prompt = f"""
            You are a punishment scenario generator. 
            The player's stats suggest intensity tier: {result['intensityTier']}.
            Provide a short brand-new humiliating scenario for effect: '{effect_str}'.
            """
            try:
                gpt_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate a unique punishment scenario."}
                    ],
                    max_tokens=200,
                    temperature=1.0
                )
                punishment_scenario = gpt_response.choices[0].message.content.strip()
            except Exception as e:
                punishment_scenario = f"(GPT error: {e})"

            result["punishmentScenario"] = punishment_scenario

        # 5) If meltdown is triggered as an easter egg
        if "meltdown" in effect_lower:
            meltdown_line = meltdown_dialog_gpt("EasterEggNPC", 2)
            record_meltdown_dialog(npc_id or 999, meltdown_line)  # or a dummy ID
            result["meltdownLine"] = meltdown_line

        # 6) Combine chosen_intensity_tier (if we got one from the DB) with the new scenario
        #    If not overshadowed by GPT scenario, we might incorporate it
        if chosen_intensity_tier and not punishment_scenario:
            result["punishmentScenario"] = f"(From IntensityTier) {chosen_intensity_tier}"

        # 7) Write a final memory log
        memory_text = f"Effect triggered: {effect_str}. (Intensity: {result['intensityTier']})"
        store_roleplay_segment({
            "key": f"Effect_{player_name}",
            "value": memory_text
        })
        result["memoryLog"] = memory_text

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
    player_stats = get_player_stats(player_name)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT condition, effect FROM GameRules")
    rules = cursor.fetchall()
    conn.close()

    triggered_effects = []
    for (condition_str, effect_str) in rules:
        logic_op, parsed_conditions = parse_condition(condition_str)
        result = evaluate_condition(logic_op, parsed_conditions, player_stats)
        if result:
            # It's True => apply
            outcome = apply_effect(effect_str, player_stats, {})
            triggered_effects.append((condition_str, effect_str, outcome))

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

    # Build a readable response
    results = []
    for (cond, eff, outcome) in triggered:
        results.append({
            "condition": cond,
            "effect": eff,
            "appliedOutcome": outcome
        })

    return jsonify({
        "message": f"Enforced rules for player '{player_name}'.",
        "triggered": results
    }), 200
