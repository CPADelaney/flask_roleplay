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

def apply_effect(effect_str, player_stats, npc_stats):
    """
    This is a placeholder function that does something
    if a rule's condition is True. You can expand this logic:
      - Lock the player's choices
      - Update stats in DB
      - Return an event describing the punishment, etc.
    """
    # Some examples:
    if effect_str.lower().startswith("locks independent choices"):
        # Potentially set a session flag or mark the player's
        # "can_defy" as false in DB
        print("Effect: Locking player's independent choices...")

    elif effect_str.lower().startswith("total compliance"):
        # Could forcibly set Obedience=100 or something
        print("Effect: Player forced to total compliance (Obedience=100).")

    elif effect_str.lower().startswith("npc cruelty intensifies"):
        # If we had an NPC in question, we might raise its cruelty stat
        # or trigger a punishment scene
        print("Effect: NPC cruelty intensifies, might escalate punishments...")

    elif effect_str.lower().startswith("collaborative physical punishments"):
        # Another special effect
        print("Effect: multiple NPCs collaborating to punish the player...")

    # This is just textual. If you want to do an actual DB update, do it here:
    # e.g. "Obedience=100" code snippet:
    #   conn = get_db_connection()
    #   c = conn.cursor()
    #   c.execute("UPDATE PlayerStats SET obedience=100 WHERE player_name=%s", ("Chase",))
    #   conn.commit()
    #   conn.close()

    # etc.
    return f"Applied effect: {effect_str}"


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
