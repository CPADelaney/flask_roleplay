# logic/stats_logic.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
import random, json

stats_bp = Blueprint('stats_bp', __name__)

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
