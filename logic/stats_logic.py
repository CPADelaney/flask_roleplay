# logic/stats_logic.py

import logging
import asyncio
import asyncpg
import json
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from db.connection import get_db_connection_context
from quart import Blueprint, request, jsonify, session, redirect, url_for
from lore.core import canon

logger = logging.getLogger(__name__)

stats_bp = Blueprint('stats_bp', __name__)

# ============================
# STAT SYSTEM CONFIGURATION
# ============================

# Visible stats configuration
VISIBLE_STATS = {
    'hp': {'min': 0, 'max': 999, 'default': 100},
    'max_hp': {'min': 1, 'max': 999, 'default': 100},
    'strength': {'min': 1, 'max': 100, 'default': 10},
    'endurance': {'min': 1, 'max': 100, 'default': 10},
    'agility': {'min': 1, 'max': 100, 'default': 10},
    'empathy': {'min': 1, 'max': 100, 'default': 10},
    'intelligence': {'min': 1, 'max': 100, 'default': 10}
}

# Hidden stats configuration
HIDDEN_STATS = {
    'corruption': {'min': 0, 'max': 100, 'default': 10},
    'confidence': {'min': 0, 'max': 100, 'default': 60},
    'willpower': {'min': 0, 'max': 100, 'default': 50},
    'obedience': {'min': 0, 'max': 100, 'default': 20},
    'dependency': {'min': 0, 'max': 100, 'default': 10},
    'lust': {'min': 0, 'max': 100, 'default': 15},
    'mental_resilience': {'min': 0, 'max': 100, 'default': 55}
}

# Character class templates for initialization
CHARACTER_CLASSES = {
    "default": {
        "hp": 100, "max_hp": 100, "strength": 10, 
        "endurance": 10, "agility": 10, "empathy": 10, "intelligence": 10
    },
    "fighter": {
        "hp": 120, "max_hp": 120, "strength": 14, 
        "endurance": 12, "agility": 8, "empathy": 8, "intelligence": 8
    },
    "scholar": {
        "hp": 80, "max_hp": 80, "strength": 6, 
        "endurance": 8, "agility": 10, "empathy": 12, "intelligence": 14
    },
    "rogue": {
        "hp": 90, "max_hp": 90, "strength": 8, 
        "endurance": 9, "agility": 14, "empathy": 11, "intelligence": 10
    },
    "empath": {
        "hp": 85, "max_hp": 85, "strength": 7,
        "endurance": 9, "agility": 11, "empathy": 15, "intelligence": 11
    }
}

# Hunger thresholds
HUNGER_THRESHOLDS = {
    'full': 80,
    'satisfied': 60,
    'hungry': 40,
    'very_hungry': 20,
    'starving': 0
}

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

# ============================
# GAME RULES & STAT DEFINITIONS
# ============================

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

    # Player Stats (both visible and hidden)
    player_stats_data = [
        # Visible stats
        {
            "stat_name": "HP",
            "range_min": 0,
            "range_max": 999,
            "definition": "Current health points",
            "effects": "0: Unconscious/defeated",
            "progression": "Decreases: Damage. Increases: Healing, rest"
        },
        {
            "stat_name": "Strength",
            "range_min": 1,
            "range_max": 100,
            "definition": "Physical power for attacks and carrying capacity",
            "effects": "Higher values increase damage output",
            "progression": "Increases: Training, combat. Decreases: Injury, exhaustion"
        },
        {
            "stat_name": "Endurance",
            "range_min": 1,
            "range_max": 100,
            "definition": "Stamina, pain tolerance, and defense. Higher values require more food",
            "effects": "Affects defense and hunger rate",
            "progression": "Increases: Exercise, survival. Decreases: Starvation, injury"
        },
        {
            "stat_name": "Agility",
            "range_min": 1,
            "range_max": 100,
            "definition": "Speed, reflexes, and dexterity",
            "effects": "Affects initiative and dodge chance",
            "progression": "Increases: Practice, successful dodges. Decreases: Injury, restraint"
        },
        {
            "stat_name": "Empathy",
            "range_min": 1,
            "range_max": 100,
            "definition": "Social intuition - reading people and situations",
            "effects": "Allows detection of deception and hidden emotions",
            "progression": "Increases: Social interaction, observation. Decreases: Isolation, trauma"
        },
        {
            "stat_name": "Intelligence",
            "range_min": 1,
            "range_max": 100,
            "definition": "Learning ability and problem solving",
            "effects": "Future: skill learning and puzzle solving",
            "progression": "Increases: Study, problem solving. Decreases: Mind-affecting trauma"
        },
        # Hidden stats
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
            "definition": "Legacy stat - now replaced by Endurance",
            "effects": "No longer used directly",
            "progression": "Migrated to new Endurance stat"
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

async def insert_default_player_stats_chase(user_id, conversation_id, provided_conn=None):
    """
    Insert row for 'Chase' in PlayerStats, scoping by user_id + conversation_id.
    Legacy function - now calls initialize_player_stats
    """
    return await initialize_player_stats(user_id, conversation_id, "Chase", "default", provided_conn)

# ============================
# CORE STAT FUNCTIONS
# ============================

async def get_player_visible_stats(user_id: int, conversation_id: int, player_name: str = "Chase") -> Dict[str, Any]:
    """
    Get only the visible stats for a player.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get stat values
            row = await conn.fetchrow("""
                SELECT hp, max_hp, strength, endurance, agility, empathy, intelligence
                FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not row:
                return {}
            
            # Get hunger from vitals
            vitals_row = await conn.fetchrow("""
                SELECT hunger, thirst, fatigue FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            stats = dict(row)
            if vitals_row:
                stats['hunger'] = vitals_row['hunger']
                stats['thirst'] = vitals_row.get('thirst', 100)
                stats['fatigue'] = vitals_row.get('fatigue', 0)
            else:
                stats['hunger'] = 100  # Default to full
                stats['thirst'] = 100
                stats['fatigue'] = 0
                
            # Add hunger status
            stats['hunger_status'] = get_hunger_status(stats['hunger'])
                
            return stats
    except Exception as e:
        logger.error(f"Error getting visible stats: {e}", exc_info=True)
        return {}

async def get_player_hidden_stats(user_id: int, conversation_id: int, player_name: str = "Chase") -> Dict[str, Any]:
    """
    Get the hidden background stats (for game logic use only).
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, 
                       dependency, lust, mental_resilience
                FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not row:
                return {}
                
            return dict(row)
    except Exception as e:
        logger.error(f"Error getting hidden stats: {e}", exc_info=True)
        return {}

async def get_all_player_stats(user_id: int, conversation_id: int, player_name: str = "Chase") -> Dict[str, Any]:
    """
    Get all stats (visible and hidden) with metadata about visibility.
    Used by AI and admin interfaces only.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not row:
                return {}
            
            # Get vitals
            vitals_row = await conn.fetchrow("""
                SELECT hunger, thirst, fatigue FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            # Organize stats by visibility
            all_stats = dict(row)
            stats = {
                "visible": {},
                "hidden": {},
                "vitals": {}
            }
            
            # Categorize stats
            for stat in VISIBLE_STATS:
                if stat in all_stats:
                    stats["visible"][stat] = all_stats[stat]
                    
            for stat in HIDDEN_STATS:
                if stat in all_stats:
                    stats["hidden"][stat] = all_stats[stat]
            
            # Add legacy physical_endurance to hidden
            if 'physical_endurance' in all_stats:
                stats["hidden"]['physical_endurance'] = all_stats['physical_endurance']
            
            # Add vitals
            if vitals_row:
                stats["vitals"] = dict(vitals_row)
            else:
                stats["vitals"] = {"hunger": 100, "thirst": 100, "fatigue": 0}
            
            return stats
    except Exception as e:
        logger.error(f"Error getting all stats: {e}", exc_info=True)
        return {}

# ============================
# STAT INITIALIZATION
# ============================

async def initialize_player_stats(user_id: int, conversation_id: int, player_name: str = "Chase", 
                                character_class: str = "default", provided_conn=None):
    """
    Initialize both visible and hidden stats for a new player.
    """
    try:
        # Get base stats for class
        visible_stats = CHARACTER_CLASSES.get(character_class, CHARACTER_CLASSES["default"]).copy()
        
        # Initialize hidden stats with defaults
        hidden_stats = {stat: config['default'] for stat, config in HIDDEN_STATS.items()}
        
        # Add legacy physical_endurance
        hidden_stats['physical_endurance'] = visible_stats['endurance'] * 4  # Convert back
        
        # Combine all stats
        all_stats = {**visible_stats, **hidden_stats}
        
        if provided_conn:
            conn = provided_conn
            await _insert_player_stats(conn, user_id, conversation_id, player_name, all_stats)
            await _initialize_player_vitals(conn, user_id, conversation_id, player_name)
        else:
            async with get_db_connection_context() as conn:
                await _insert_player_stats(conn, user_id, conversation_id, player_name, all_stats)
                await _initialize_player_vitals(conn, user_id, conversation_id, player_name)
        
        logger.info(f"Initialized stats for {player_name} as {character_class}")
        return True
    except Exception as e:
        logger.error(f"Error initializing player stats: {e}", exc_info=True)
        return False

async def _insert_player_stats(conn, user_id: int, conversation_id: int, player_name: str, stats: Dict[str, int]):
    """Helper to insert player stats."""
    await conn.execute("""
        INSERT INTO PlayerStats (
            user_id, conversation_id, player_name,
            hp, max_hp, strength, endurance, agility, empathy, intelligence,
            corruption, confidence, willpower, obedience, dependency, lust, 
            mental_resilience, physical_endurance
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        ON CONFLICT (user_id, conversation_id, player_name) 
        DO UPDATE SET
            hp = EXCLUDED.hp,
            max_hp = EXCLUDED.max_hp,
            strength = EXCLUDED.strength,
            endurance = EXCLUDED.endurance,
            agility = EXCLUDED.agility,
            empathy = EXCLUDED.empathy,
            intelligence = EXCLUDED.intelligence
    """, 
        user_id, conversation_id, player_name,
        stats.get('hp', 100), stats.get('max_hp', 100),
        stats.get('strength', 10), stats.get('endurance', 10),
        stats.get('agility', 10), stats.get('empathy', 10),
        stats.get('intelligence', 10),
        stats.get('corruption', 10), stats.get('confidence', 60),
        stats.get('willpower', 50), stats.get('obedience', 20),
        stats.get('dependency', 10), stats.get('lust', 15),
        stats.get('mental_resilience', 55), stats.get('physical_endurance', 40)
    )

async def _initialize_player_vitals(conn, user_id: int, conversation_id: int, player_name: str):
    """Helper to initialize player vitals."""
    await conn.execute("""
        INSERT INTO PlayerVitals (user_id, conversation_id, player_name, hunger, thirst, fatigue)
        VALUES ($1, $2, $3, 100, 100, 0)
        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
    """, user_id, conversation_id, player_name)

# ============================
# STAT UPDATE FUNCTIONS
# ============================

async def update_player_stats(user_id, conversation_id, player_name, stat_key, value):
    """
    Updates a specific stat for a given player in the PlayerStats table.
    Legacy function - determines if stat is visible or hidden and routes appropriately.
    """
    if stat_key in VISIBLE_STATS:
        return await update_visible_stat(user_id, conversation_id, player_name, stat_key, value)
    elif stat_key in HIDDEN_STATS or stat_key == 'physical_endurance':
        return await update_hidden_stat(user_id, conversation_id, player_name, stat_key, value)
    else:
        return {"success": False, "error": f"Unknown stat: {stat_key}"}

async def update_visible_stat(user_id: int, conversation_id: int, player_name: str, 
                            stat_name: str, new_value: int, reason: str = "direct_update") -> Dict[str, Any]:
    """
    Update a visible stat with bounds checking.
    """
    if stat_name not in VISIBLE_STATS:
        return {"success": False, "error": f"Invalid visible stat: {stat_name}"}
    
    config = VISIBLE_STATS[stat_name]
    clamped_value = max(config['min'], min(config['max'], new_value))
    
    try:
        async with get_db_connection_context() as conn:
            # Get old value
            old_value = await conn.fetchval(
                f"SELECT {stat_name} FROM PlayerStats "
                "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
                user_id, conversation_id, player_name
            )
            
            if old_value is None:
                return {"success": False, "error": "Player not found"}
            
            # Update stat
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            await canon.update_player_stat_canonically(
                canon_ctx, conn, player_name, stat_name, clamped_value, reason
            )
            
            # Trigger side effects
            await _handle_stat_change_side_effects(
                conn, user_id, conversation_id, player_name,
                stat_name, old_value, clamped_value
            )
            
            return {
                "success": True,
                "stat": stat_name,
                "old_value": old_value,
                "new_value": clamped_value,
                "clamped": new_value != clamped_value,
                "affected_rows": 1
            }
    except Exception as e:
        logger.error(f"Error updating visible stat: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def update_hidden_stat(user_id: int, conversation_id: int, player_name: str,
                           stat_name: str, new_value: int, reason: str = "game_logic") -> Dict[str, Any]:
    """
    Update a hidden stat (for internal game logic only).
    """
    if stat_name not in HIDDEN_STATS and stat_name != 'physical_endurance':
        return {"success": False, "error": f"Invalid hidden stat: {stat_name}"}
    
    # Handle physical_endurance specially
    if stat_name == 'physical_endurance':
        config = {'min': 0, 'max': 100}
    else:
        config = HIDDEN_STATS[stat_name]
        
    clamped_value = max(config['min'], min(config['max'], new_value))
    
    try:
        async with get_db_connection_context() as conn:
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            await canon.update_player_stat_canonically(
                canon_ctx, conn, player_name, stat_name, clamped_value, reason
            )
            
            return {
                "success": True,
                "stat": stat_name,
                "new_value": clamped_value,
                "affected_rows": 1
            }
    except Exception as e:
        logger.error(f"Error updating hidden stat: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def apply_stat_change(user_id: int, conversation_id: int, changes: Dict[str, int], 
                          cause: str = "") -> Dict[str, Any]:
    """
    Apply multiple stat changes at once and record the history.
    Legacy function that routes to apply_stat_changes.
    """
    return await apply_stat_changes(user_id, conversation_id, "Chase", changes, cause)

async def apply_stat_changes(user_id: int, conversation_id: int, player_name: str,
                           changes: Dict[str, int], reason: str = "") -> Dict[str, Any]:
    """
    Apply multiple stat changes at once (both visible and hidden).
    """
    results = {
        "success": True,
        "changes_applied": {},
        "errors": []
    }
    
    try:
        async with get_db_connection_context() as conn:
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            
            for stat_name, change in changes.items():
                # Determine if stat is visible or hidden
                is_visible = stat_name in VISIBLE_STATS
                is_hidden = stat_name in HIDDEN_STATS or stat_name == 'physical_endurance'
                
                if not is_visible and not is_hidden:
                    results["errors"].append(f"Unknown stat: {stat_name}")
                    continue
                
                # Get current value
                current_value = await conn.fetchval(
                    f"SELECT {stat_name} FROM PlayerStats "
                    "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
                    user_id, conversation_id, player_name
                )
                
                if current_value is None:
                    results["errors"].append(f"Player not found for stat: {stat_name}")
                    continue
                
                # Calculate new value
                if stat_name == 'physical_endurance':
                    config = {'min': 0, 'max': 100}
                else:
                    config = VISIBLE_STATS.get(stat_name, HIDDEN_STATS.get(stat_name))
                new_value = max(config['min'], min(config['max'], current_value + change))
                
                # Apply change
                await canon.update_player_stat_canonically(
                    canon_ctx, conn, player_name, stat_name, new_value, reason
                )
                
                results["changes_applied"][stat_name] = {
                    "old": current_value,
                    "new": new_value,
                    "change": new_value - current_value
                }
                
                # Handle side effects for visible stats
                if is_visible:
                    await _handle_stat_change_side_effects(
                        conn, user_id, conversation_id, player_name,
                        stat_name, current_value, new_value
                    )
    except Exception as e:
        logger.error(f"Error applying stat changes: {e}", exc_info=True)
        results["success"] = False
        results["errors"].append(str(e))
    
    return results

# ============================
# STAT HISTORY & TRACKING
# ============================

async def record_stat_change_event(user_id: int, conversation_id: int, stat_name: str, 
                                 old_value: int, new_value: int, cause: str = "") -> Dict[str, Any]:
    """
    Record significant stat changes in StatsHistory table.
    """
    # Only record significant changes (more than 5 points)
    if abs(new_value - old_value) < 5:
        return {"success": True, "recorded": False, "reason": "Change too small"}
        
    try:
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO StatsHistory 
                (user_id, conversation_id, player_name, stat_name,
                 old_value, new_value, cause, timestamp)
                VALUES ($1, $2, 'Chase', $3, $4, $5, $6, CURRENT_TIMESTAMP)
            """, user_id, conversation_id, stat_name, old_value, new_value, cause)
            
            return {"success": True, "recorded": True}
            
    except Exception as e:
        logger.error(f"Error recording stat change: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================
# STAT THRESHOLDS & TIERS
# ============================

async def get_player_current_tier(user_id: int, conversation_id: int, stat_name: str) -> Optional[Dict[str, Any]]:
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

async def check_for_combination_triggers(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
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

# ============================
# ACTIVITY EFFECTS
# ============================

async def apply_activity_effects(user_id: int, conversation_id: int, activity_name: str, 
                               intensity: float = 1.0) -> Dict[str, Any]:
    """
    Apply stat changes based on a specific activity
    Intensity multiplier can increase or decrease the effect
    """
    if activity_name not in ACTIVITY_EFFECTS:
        return {"success": False, "error": "Unknown activity"}
        
    # Get the base effects and scale by intensity
    base_effects = ACTIVITY_EFFECTS[activity_name]
    scaled_effects = {stat: int(change * intensity) for stat, change in base_effects.items()}
    
    # Apply the changes
    return await apply_stat_change(
        user_id, conversation_id, 
        scaled_effects, 
        f"Activity: {activity_name} (intensity: {intensity})"
    )

# ============================
# SOCIAL LINKS
# ============================

async def check_relationship_milestones(user_id: int, conversation_id: int, player_name: str = "Chase"):
    """Check relationship milestones using the new system."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    
    async with get_db_connection_context() as conn:
        # Get all relationships for the player
        rows = await conn.fetch("""
            SELECT link_id, entity1_type, entity1_id,
                   entity2_type, entity2_id, canonical_key
            FROM SocialLinks
            WHERE (entity1_type='player' OR entity2_type='player')
            AND user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)

        for row in rows:
            # Get full relationship state
            if row['entity1_type'] == 'player':
                entity_type = row['entity2_type']
                entity_id = row['entity2_id']
            else:
                entity_type = row['entity1_type']
                entity_id = row['entity1_id']
            
            state = await manager.get_relationship_state(
                entity1_type='player',
                entity1_id=1,
                entity2_type=entity_type,
                entity2_id=entity_id
            )
            
            # Check milestones based on dimensions
            if state.dimensions.trust >= 80 and state.dimensions.affection >= 70:
                # High trust and affection milestone
                await conn.execute("""
                    INSERT INTO PlayerInventory (player_name, item_name, quantity)
                    VALUES ($1, $2, 1)
                    ON CONFLICT (player_name, item_name) DO NOTHING
                """, player_name, "Bond Token")
                
                # Log in relationship history
                await state.history.record_interaction(
                    user_id,
                    conversation_id,
                    "milestone",
                    "high_trust_affection",
                )
# ============================
# HUNGER & VITALS SYSTEM
# ============================

def get_hunger_status(hunger_value: int) -> str:
    """Get descriptive hunger status."""
    if hunger_value >= HUNGER_THRESHOLDS['full']:
        return "Full"
    elif hunger_value >= HUNGER_THRESHOLDS['satisfied']:
        return "Satisfied"
    elif hunger_value >= HUNGER_THRESHOLDS['hungry']:
        return "Hungry"
    elif hunger_value >= HUNGER_THRESHOLDS['very_hungry']:
        return "Very Hungry"
    else:
        return "Starving"

async def update_hunger_from_time(user_id: int, conversation_id: int, player_name: str = "Chase", 
                                hours_passed: int = 1) -> Dict[str, Any]:
    """
    Update hunger based on time passed and endurance level.
    Higher endurance = faster hunger drain.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current endurance and hunger
            stats = await conn.fetchrow("""
                SELECT ps.endurance, pv.hunger
                FROM PlayerStats ps
                LEFT JOIN PlayerVitals pv ON ps.user_id = pv.user_id 
                    AND ps.conversation_id = pv.conversation_id 
                    AND ps.player_name = pv.player_name
                WHERE ps.user_id = $1 AND ps.conversation_id = $2 AND ps.player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not stats:
                return {"success": False, "error": "Player not found"}
            
            endurance = stats['endurance'] or 10
            current_hunger = stats['hunger'] or 100
            
            # Calculate hunger drain
            base_drain_per_hour = 2.5
            endurance_modifier = 1 + (endurance - 10) / 20  # +5% per point above 10
            total_drain = int(base_drain_per_hour * endurance_modifier * hours_passed)
            
            new_hunger = max(0, current_hunger - total_drain)
            
            # Update hunger
            await conn.execute("""
                INSERT INTO PlayerVitals (user_id, conversation_id, player_name, hunger)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, player_name)
                DO UPDATE SET hunger = $4, last_update = CURRENT_TIMESTAMP
            """, user_id, conversation_id, player_name, new_hunger)
            
            # Apply hunger effects
            old_status = get_hunger_status(current_hunger)
            new_status = get_hunger_status(new_hunger)
            status_changed = old_status != new_status
            
            effects = {}
            if new_hunger < HUNGER_THRESHOLDS['very_hungry']:
                effects = await apply_hunger_effects(user_id, conversation_id, player_name, new_hunger)
            
            return {
                "success": True,
                "old_hunger": current_hunger,
                "new_hunger": new_hunger,
                "drain": total_drain,
                "status": new_status,
                "status_changed": status_changed,
                "effects_applied": effects
            }
    except Exception as e:
        logger.error(f"Error updating hunger: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def apply_hunger_effects(user_id: int, conversation_id: int, player_name: str, 
                             hunger_level: int) -> Dict[str, Any]:
    """
    Apply stat modifiers based on hunger level.
    These affect both visible and hidden stats.
    """
    effects = {}
    
    if hunger_level < HUNGER_THRESHOLDS['very_hungry']:
        # Very hungry effects
        effects = {
            "strength": -2,
            "agility": -1,
            "confidence": -3,
            "willpower": -2
        }
    
    if hunger_level < HUNGER_THRESHOLDS['starving'] + 10:
        # Starving effects (more severe)
        effects = {
            "strength": -5,
            "agility": -3,
            "empathy": -2,
            "intelligence": -1,
            "confidence": -5,
            "willpower": -4,
            "mental_resilience": -3
        }
    
    if effects:
        # Apply as temporary effects (you might want to track these separately)
        # For now, we'll just return what would be applied
        return effects
    
    return {}

async def consume_food(user_id: int, conversation_id: int, player_name: str,
                      food_value: int, food_name: str = "food") -> Dict[str, Any]:
    """
    Consume food to restore hunger.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current hunger
            current_hunger = await conn.fetchval("""
                SELECT hunger FROM PlayerVitals
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if current_hunger is None:
                current_hunger = 100
            
            new_hunger = min(100, current_hunger + food_value)
            
            # Update hunger
            await conn.execute("""
                INSERT INTO PlayerVitals (user_id, conversation_id, player_name, hunger)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, player_name)
                DO UPDATE SET hunger = $4, last_update = CURRENT_TIMESTAMP
            """, user_id, conversation_id, player_name, new_hunger)
            
            # Log consumption in stats history
            await conn.execute("""
                INSERT INTO StatsHistory (user_id, conversation_id, player_name, 
                                        stat_name, old_value, new_value, cause)
                VALUES ($1, $2, $3, 'hunger', $4, $5, $6)
            """, user_id, conversation_id, player_name, current_hunger, new_hunger,
                f"Consumed {food_name}")
            
            return {
                "success": True,
                "food_consumed": food_name,
                "hunger_restored": new_hunger - current_hunger,
                "old_hunger": current_hunger,
                "new_hunger": new_hunger,
                "status": get_hunger_status(new_hunger)
            }
    except Exception as e:
        logger.error(f"Error consuming food: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================
# COMBAT SYSTEM
# ============================

def calculate_damage(strength: int, weapon_bonus: int = 0) -> int:
    """Calculate damage based on strength."""
    base_damage = (strength // 2) + 3
    return base_damage + weapon_bonus + random.randint(-2, 2)

def calculate_defense(endurance: int, armor_bonus: int = 0) -> int:
    """Calculate defense/damage reduction based on endurance."""
    base_defense = endurance // 4
    return base_defense + armor_bonus

def calculate_initiative(agility: int) -> int:
    """Calculate initiative/turn order based on agility."""
    return agility + random.randint(1, 20)

def calculate_hit_chance(attacker_agility: int, defender_agility: int) -> int:
    """Calculate chance to hit as percentage."""
    base_chance = 75
    agility_diff = attacker_agility - defender_agility
    return max(25, min(95, base_chance + (agility_diff * 2)))

async def apply_damage(user_id: int, conversation_id: int, player_name: str, 
                      damage: int, source: str = "combat") -> Dict[str, Any]:
    """
    Apply damage to a player's HP.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current HP
            current_hp = await conn.fetchval("""
                SELECT hp FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if current_hp is None:
                return {"success": False, "error": "Player not found"}
            
            new_hp = max(0, current_hp - damage)
            
            # Update HP
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            await canon.update_player_stat_canonically(
                canon_ctx, conn, player_name, "hp", new_hp, f"Damage from {source}"
            )
            
            # Check for defeat
            defeated = new_hp == 0
            if defeated:
                await _handle_defeat(conn, user_id, conversation_id, player_name, source)
            
            return {
                "success": True,
                "damage_taken": damage,
                "old_hp": current_hp,
                "new_hp": new_hp,
                "defeated": defeated
            }
    except Exception as e:
        logger.error(f"Error applying damage: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def heal_player(user_id: int, conversation_id: int, player_name: str,
                     heal_amount: int, source: str = "healing") -> Dict[str, Any]:
    """
    Heal a player's HP.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get current and max HP
            stats = await conn.fetchrow("""
                SELECT hp, max_hp FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            if not stats:
                return {"success": False, "error": "Player not found"}
            
            current_hp = stats['hp']
            max_hp = stats['max_hp']
            new_hp = min(max_hp, current_hp + heal_amount)
            actual_heal = new_hp - current_hp
            
            # Update HP
            canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
            await canon.update_player_stat_canonically(
                canon_ctx, conn, player_name, "hp", new_hp, f"Healing from {source}"
            )
            
            return {
                "success": True,
                "heal_amount": actual_heal,
                "old_hp": current_hp,
                "new_hp": new_hp,
                "at_max": new_hp == max_hp
            }
    except Exception as e:
        logger.error(f"Error healing player: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================
# SOCIAL SYSTEM
# ============================

def calculate_social_insight(empathy: int, difficulty: int = 10) -> Tuple[bool, int]:
    """
    Calculate if character notices social cues.
    Returns (success, roll_total)
    """
    roll = random.randint(1, 20)
    empathy_bonus = empathy // 5
    total = roll + empathy_bonus
    return (total >= difficulty, total)

async def detect_deception(user_id: int, conversation_id: int, player_name: str,
                          npc_id: int, deception_type: str) -> Dict[str, Any]:
    """
    Use empathy to detect NPC deception or hidden emotions.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get player empathy
            empathy = await conn.fetchval("""
                SELECT empathy FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
            
            # Get NPC stats
            npc_stats = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty, monica_level
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, user_id, conversation_id)
            
            if not empathy or not npc_stats:
                return {"success": False, "error": "Missing data"}
            
            # Calculate difficulty based on NPC stats
            base_difficulty = 12
            difficulty_modifiers = {
                'dominance': npc_stats['dominance'] // 20,  # +0-5
                'monica_level': npc_stats.get('monica_level', 0) // 2 if npc_stats.get('monica_level') else 0
            }
            total_difficulty = base_difficulty + sum(difficulty_modifiers.values())
            
            # Make insight check
            success, roll_total = calculate_social_insight(empathy, total_difficulty)
            
            insights_map = {
                "lying": f"{npc_stats['npc_name']} is not being truthful",
                "hidden_anger": f"{npc_stats['npc_name']} is suppressing anger",
                "false_kindness": f"{npc_stats['npc_name']}'s friendliness seems forced",
                "nervousness": f"{npc_stats['npc_name']} is more nervous than they appear",
                "hidden_motive": f"{npc_stats['npc_name']} has ulterior motives",
                "attraction": f"{npc_stats['npc_name']} is trying to hide their interest",
                "fear": f"Despite their demeanor, {npc_stats['npc_name']} is afraid"
            }
            
            result = {
                "success": success,
                "roll_total": roll_total,
                "difficulty": total_difficulty,
                "empathy_used": empathy
            }
            
            if success:
                result["insight"] = insights_map.get(deception_type, "Something seems off")
                
                # Reward successful insight with hidden stat changes
                await apply_stat_changes(
                    user_id, conversation_id, player_name,
                    {"confidence": 1, "mental_resilience": 1},
                    f"Successfully read {npc_stats['npc_name']}"
                )
            else:
                # Failed checks might have consequences
                margin = total_difficulty - roll_total
                if margin > 10:  # Critical failure
                    result["critical_failure"] = True
                    await apply_stat_changes(
                        user_id, conversation_id, player_name,
                        {"confidence": -1},
                        f"Completely misread {npc_stats['npc_name']}"
                    )
            
            return result
    except Exception as e:
        logger.error(f"Error detecting deception: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================
# SIDE EFFECTS & INTERACTIONS
# ============================

async def _handle_stat_change_side_effects(conn, user_id: int, conversation_id: int,
                                         player_name: str, stat_name: str,
                                         old_value: int, new_value: int):
    """
    Handle side effects when visible stats change.
    This is where visible stat changes can trigger hidden stat changes.
    """
    change = new_value - old_value
    if change == 0:
        return
    
    # Define side effect mappings
    side_effects = {}
    
    if stat_name == "hp":
        if new_value < old_value:  # Took damage
            damage_ratio = (old_value - new_value) / old_value
            if damage_ratio > 0.5:  # Lost more than 50% HP
                side_effects.update({
                    "confidence": -3,
                    "willpower": -2
                })
            elif damage_ratio > 0.25:  # Lost more than 25% HP
                side_effects.update({
                    "confidence": -1
                })
    
    elif stat_name == "strength":
        if change < -5:  # Major strength loss
            side_effects.update({
                "confidence": -2,
                "mental_resilience": -1
            })
    
    elif stat_name == "empathy":
        if change > 5:  # Major empathy gain
            side_effects.update({
                "mental_resilience": 1
            })
    
    elif stat_name == "endurance":
        if change < -5:  # Major endurance loss
            side_effects.update({
                "willpower": -2,
                "physical_endurance": change * 4  # Legacy stat
            })
    
    # Apply side effects to hidden stats
    if side_effects:
        canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
        for hidden_stat, change_value in side_effects.items():
            current = await conn.fetchval(
                f"SELECT {hidden_stat} FROM PlayerStats "
                "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
                user_id, conversation_id, player_name
            )
            if current is not None:
                if hidden_stat == 'physical_endurance':
                    config = {'min': 0, 'max': 100}
                else:
                    config = HIDDEN_STATS.get(hidden_stat, {'min': 0, 'max': 100})
                new_hidden_value = max(config['min'], min(config['max'], current + change_value))
                await canon.update_player_stat_canonically(
                    canon_ctx, conn, player_name, hidden_stat, new_hidden_value,
                    f"Side effect from {stat_name} change"
                )

async def _handle_defeat(conn, user_id: int, conversation_id: int, player_name: str, source: str):
    """
    Handle player defeat consequences.
    """
    # Apply hidden stat changes for defeat
    defeat_changes = {
        "confidence": -5,
        "willpower": -3,
        "obedience": 5,
        "corruption": 2,
        "mental_resilience": -2
    }
    
    canon_ctx = type("ctx", (), {"user_id": user_id, "conversation_id": conversation_id})()
    for stat, change in defeat_changes.items():
        current = await conn.fetchval(
            f"SELECT {stat} FROM PlayerStats "
            "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
            user_id, conversation_id, player_name
        )
        if current is not None:
            config = HIDDEN_STATS[stat]
            new_value = max(config['min'], min(config['max'], current + change))
            await canon.update_player_stat_canonically(
                canon_ctx, conn, player_name, stat, new_value,
                f"Defeated by {source}"
            )

# ============================
# MIGRATION FUNCTIONS
# ============================

async def migrate_to_new_stat_system(user_id: int, conversation_id: int):
    """
    Migrate existing characters to the new stat system.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get all players for this conversation
            players = await conn.fetch("""
                SELECT player_name, physical_endurance, confidence
                FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            for player in players:
                # Calculate new visible stats based on old stats
                old_phys_endurance = player['physical_endurance'] or 40
                old_confidence = player['confidence'] or 60
                
                # Convert physical_endurance to new endurance (divide by 4)
                new_endurance = max(10, old_phys_endurance // 4)
                
                # Set other stats based on confidence level
                if old_confidence > 70:
                    stat_modifier = 2
                elif old_confidence > 40:
                    stat_modifier = 0
                else:
                    stat_modifier = -2
                
                # Update visible stats
                await conn.execute("""
                    UPDATE PlayerStats
                    SET hp = COALESCE(hp, $1),
                        max_hp = COALESCE(max_hp, $1),
                        strength = COALESCE(strength, $2),
                        endurance = COALESCE(endurance, $3),
                        agility = COALESCE(agility, $4),
                        empathy = COALESCE(empathy, $5),
                        intelligence = COALESCE(intelligence, $6)
                    WHERE user_id = $7 AND conversation_id = $8 AND player_name = $9
                """, 
                    100,  # HP
                    10 + stat_modifier,  # Strength
                    new_endurance,  # Endurance
                    10 + stat_modifier,  # Agility
                    10,  # Empathy
                    10,  # Intelligence
                    user_id, conversation_id, player['player_name']
                )
                
                # Initialize vitals
                await _initialize_player_vitals(conn, user_id, conversation_id, player['player_name'])
                
            return {"success": True, "players_migrated": len(players)}
    except Exception as e:
        logger.error(f"Error migrating stats: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================
# API ROUTES
# ============================

@stats_bp.route('/player/<player_name>/visible', methods=['GET'])
async def get_visible_stats_route(player_name):
    """Get only visible stats for display to player."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    stats = await get_player_visible_stats(user_id, int(conversation_id), player_name)
    return jsonify(stats), 200

@stats_bp.route('/player/<player_name>/all', methods=['GET'])
async def get_all_stats_route(player_name):
    """Get all stats (admin/debug endpoint)."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    stats = await get_all_player_stats(user_id, int(conversation_id), player_name)
    return jsonify(stats), 200

@stats_bp.route('/player/<player_name>', methods=['GET', 'PUT'])
async def handle_player_stats(player_name):
    """
    Legacy route - maintained for backward compatibility.
    GET => returns visible stats only
    PUT => updates stats
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if request.method == 'GET':
        # conversation_id from query param
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        # Return visible stats for backward compatibility
        stats = await get_player_visible_stats(user_id, int(conversation_id), player_name)
        return jsonify(stats), 200

    elif request.method == 'PUT':
        data = await request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        # Extract stats from JSON and apply changes
        changes = {}
        all_stats = {**VISIBLE_STATS, **HIDDEN_STATS, 'physical_endurance': {'min': 0, 'max': 100}}
        
        for stat_name in all_stats:
            if stat_name in data:
                # This is setting absolute values, not changes
                # Get current value first
                async with get_db_connection_context() as conn:
                    current = await conn.fetchval(
                        f"SELECT {stat_name} FROM PlayerStats "
                        "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
                        user_id, int(conversation_id), player_name
                    )
                    if current is not None:
                        changes[stat_name] = data[stat_name] - current

        if not changes:
            return jsonify({"error": "No valid stats to update"}), 400

        result = await apply_stat_changes(user_id, int(conversation_id), player_name, changes, "API update")
        
        if result["success"]:
            # Get updated stats to return
            updated_stats = await get_player_visible_stats(user_id, int(conversation_id), player_name)
            return jsonify({
                "message": f"Player '{player_name}' stats updated",
                "new_stats": updated_stats
            }), 200
        else:
            return jsonify({"error": result.get("errors", ["Update failed"])[0]}), 500

@stats_bp.route('/npc/<int:npc_id>', methods=['GET', 'PUT'])
async def handle_npc_stats(npc_id):
    """
    GET => fetch the NPC's stats for (user_id, conversation_id, npc_id)
    PUT => update them
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
            """, user_id, int(conversation_id), npc_id)

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
                    user_id, int(conversation_id), npc_id
                )
                
                # Check if any rows were affected
                if result == "UPDATE 0":
                    return jsonify({"error": f"NPCStats not found for user={user_id}, conv={conversation_id}, npc_id={npc_id}"}), 404
                
                # Get the updated row
                updated_npc = await conn.fetchrow("""
                    SELECT dominance, cruelty, closeness, trust, respect, intensity
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """, user_id, int(conversation_id), npc_id)

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

@stats_bp.route('/init_stats_and_rules', methods=['POST'])
async def init_stats_and_rules():
    """
    Endpoint to run all insert/update logic at once:
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
            await insert_default_player_stats_chase(user_id, int(conversation_id))
            return jsonify({"message": "Stats & Rules initialized successfully."}), 200
        else:
            return jsonify({"message": "Game rules and stat definitions initialized. No player stats created due to missing user_id or conversation_id."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@stats_bp.route('/player/<player_name>/heal', methods=['POST'])
async def heal_player_route(player_name):
    """Heal a player."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    heal_amount = data.get("heal_amount", 10)
    source = data.get("source", "healing")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    result = await heal_player(user_id, int(conversation_id), player_name, heal_amount, source)
    return jsonify(result), 200 if result["success"] else 400

@stats_bp.route('/player/<player_name>/damage', methods=['POST'])
async def damage_player_route(player_name):
    """Apply damage to a player."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    damage = data.get("damage", 10)
    source = data.get("source", "combat")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    result = await apply_damage(user_id, int(conversation_id), player_name, damage, source)
    return jsonify(result), 200 if result["success"] else 400

@stats_bp.route('/player/<player_name>/eat', methods=['POST'])
async def eat_food_route(player_name):
    """Consume food to restore hunger."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    food_value = data.get("food_value", 20)
    food_name = data.get("food_name", "food")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    result = await consume_food(user_id, int(conversation_id), player_name, food_value, food_name)
    return jsonify(result), 200 if result["success"] else 400

@stats_bp.route('/player/<player_name>/detect', methods=['POST'])
async def detect_deception_route(player_name):
    """Attempt to detect NPC deception."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    npc_id = data.get("npc_id")
    deception_type = data.get("deception_type", "hidden_motive")
    
    if not conversation_id or not npc_id:
        return jsonify({"error": "Missing required parameters"}), 400
    
    result = await detect_deception(user_id, int(conversation_id), player_name, 
                                  int(npc_id), deception_type)
    return jsonify(result), 200

@stats_bp.route('/migrate', methods=['POST'])
async def migrate_stats_route():
    """Migrate stats to new system."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = await request.get_json() or {}
    conversation_id = data.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400
    
    result = await migrate_to_new_stat_system(user_id, int(conversation_id))
    return jsonify(result), 200 if result["success"] else 500
