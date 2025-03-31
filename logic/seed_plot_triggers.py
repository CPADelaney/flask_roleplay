# logic/seed_plot_triggers.py
import json
import logging
import asyncio
import asyncpg
from db.connection import get_db_connection_context

async def create_and_seed_plot_triggers():
    """
    Inserts rows into the PlotTriggers table for Document 17 (Plot Triggers and Events).
    Each row covers a major section: Early Stage, Mid-Stage, Endgame, etc.
    We'll use ON CONFLICT so re-running won't duplicate data.
    """
    logging.info("Inserting or updating 'PlotTriggers' rows...")

    # We'll store each big chunk of the doc as a row.
    triggers_data = [

        # 1) Early Stage
        {
            "trigger_name": "Early Stage",
            "stage_name": "Narrative Stages and Escalation: Early Stage",
            "description": (
                "Introduces NPCs, sets relationships, and establishes power dynamics; "
                "subtle dominance, playful humiliation, minor acts of control."
            ),
            "key_features": [
                "NPCs test boundaries, identify weaknesses",
                "Introduction of Dependency via 'kind' acts",
                "Verbal humiliation begins eroding Mental Resilience"
            ],
            "stat_dynamics": [
                "Closeness increases with interactions",
                "Dominance/Cruelty grow gradually",
                "Mental Resilience erodes from teasing/humiliation"
            ],
            "examples": [
                "Coworker spills a drink, forcing you to clean it in front of them",
                "NPC uses casual bet to trick you into an embarrassing act",
                "Light punishments for failure: teasing or public correction"
            ],
            "triggers": {}
        },

        # 2) Mid-Stage Escalation
        {
            "trigger_name": "Mid-Stage Escalation",
            "stage_name": "Narrative Stages and Escalation: Mid",
            "description": (
                "NPCs become assertive, deepen control with public & private acts; "
                "collaborative dominance, forced rivalries, repeated punishments."
            ),
            "key_features": [
                "Two+ NPCs issue overlapping or conflicting demands",
                "Physical dominance (forced kneeling, guided movements)",
                "Repeated tasks reinforce submission"
            ],
            "stat_dynamics": [
                "Dominance spikes as NPCs assert control",
                "Corruption rises with degrading acts",
                "Willpower, Confidence, Mental Resilience erode under constant pressure"
            ],
            "examples": [
                "Public contest where you 'perform' for NPC amusement",
                "One NPC forces a degrading task while another mocks your efforts",
                "Forced rivalries: NPCs manipulate Dependency"
            ],
            "triggers": {}
        },

        # 3) Endgame
        {
            "trigger_name": "Endgame",
            "stage_name": "Narrative Stages and Escalation: Endgame",
            "description": (
                "NPCs assert complete dominance, final permanent claims; ceremonies, vows, or rituals. "
                "Identity is stripped, replaced by subservient titles."
            ),
            "key_features": [
                "Public ceremonies formalize submission",
                "NPCs collaborate or compete for final control",
                "Identity replaced with 'pet', 'assistant', etc."
            ],
            "stat_dynamics": [
                "Corruption >90 => dialogue craves submission",
                "Willpower <10 => no resistance possible",
                "Dependency >80 => favored NPC takes total priority"
            ],
            "examples": [
                "Formal branding ceremony marking you as property",
                "Rival NPC attempts to disrupt your submission to test loyalty",
                "Public humiliation escalates with you as the central 'toy'"
            ],
            "triggers": {}
        },

        # 4) Stat-Driven Triggers (Key Stats, e.g. Dominance/Cruelty thresholds)
        {
            "trigger_name": "Stat-Driven Triggers",
            "stage_name": "Stat-Driven Events",
            "description": "Defines how certain stat thresholds trigger special events or transformations.",
            "key_features": [
                "Dominance thresholds (50, 80) => public control, total obedience demanded",
                "Cruelty thresholds (60, 90) => inventive punishments, sadistic enjoyment",
                "Closeness thresholds (60, 90) => full entwinement, no independence",
                "Corruption thresholds (50, 90) => submissive dialogue, craving humiliation",
                "Willpower thresholds (30, 10) => diminishing or no resistance",
                "Dependency thresholds (60, 90) => favored NPC overshadowing everything"
            ],
            "stat_dynamics": [],
            "examples": [
                {
                    "event_name": "Trust-Based Betrayal",
                    "trigger": "NPC Trust < -50",
                    "outcome": "NPC sets a trap, exposing vulnerabilities",
                    "stat_impacts": "-10 Confidence, +10 Corruption, +5 Dependency on rival NPC"
                },
                {
                    "event_name": "Collaborative Mockery",
                    "trigger": "Dependency >90 and Closeness >70 w/another NPC",
                    "outcome": "Rival NPCs join forces, contradictory commands test loyalty"
                },
                {
                    "event_name": "Public Display of Marking",
                    "trigger": "Obedience >80, Dominance >90 (favored NPC)",
                    "outcome": "Public ceremony w/collar, tattoo, or symbolic branding"
                }
            ],
            "triggers": {}
        },

        # 5) The Point of No Return
        {
            "trigger_name": "Point of No Return",
            "stage_name": "Major Narrative Event",
            "description": (
                "A final or climactic ritual ensuring permanent enslavement; "
                "Stats cross thresholds (Corruption>90, Willpower<10, Dependency>80)."
            ),
            "key_features": [
                "Public ceremony or vow finalizes the submission",
                "Permanent enslavement: 'Absolute Obedience'"
            ],
            "stat_dynamics": [],
            "examples": [
                "NPC performs a contract signing or vow of loyalty in public",
                "At this point, rival NPC attempts sabotage or intensifies control attempts"
            ],
            "triggers": {
                "corruption_over_90": "No independent actions remain",
                "willpower_under_10": "Impossible to resist or disobey",
                "dependency_over_80": "Absolute loyalty to favored NPC"
            }
        },

        # 6) Example Narrative Progression
        {
            "trigger_name": "Example Narrative Progression",
            "stage_name": "Corporate Office Scenario",
            "description": (
                "Demonstrates how early, mid, and endgame beats unfold in an office environment."
            ),
            "key_features": [],
            "stat_dynamics": [],
            "examples": [
                {
                    "stage": "Early Stage",
                    "details": [
                        "Manager undermines confidence with casual critiques",
                        "Coworker forces public acknowledgment disguised as help"
                    ]
                },
                {
                    "stage": "Mid-Stage Escalation",
                    "details": [
                        "Manager forces a public apology for fabricated mistakes",
                        "Rival NPC manipulates Dependency with escalating demands"
                    ]
                }
            ],
            "triggers": {}
        }
    ]

    try:
        async with get_db_connection_context() as conn:
            for row_data in triggers_data:
                await conn.execute("""
                INSERT INTO PlotTriggers (
                  trigger_name,
                  stage_name,
                  description,
                  key_features,
                  stat_dynamics,
                  examples,
                  triggers
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (trigger_name)
                DO UPDATE
                  SET
                    stage_name = EXCLUDED.stage_name,
                    description = EXCLUDED.description,
                    key_features = EXCLUDED.key_features,
                    stat_dynamics = EXCLUDED.stat_dynamics,
                    examples = EXCLUDED.examples,
                    triggers = EXCLUDED.triggers
                """, 
                row_data["trigger_name"],
                row_data["stage_name"],
                row_data["description"],
                json.dumps(row_data["key_features"]),
                json.dumps(row_data["stat_dynamics"]),
                json.dumps(row_data["examples"]),
                json.dumps(row_data["triggers"])
                )
            
            logging.info("Seeded 'PlotTriggers' data successfully.")
    except asyncpg.PostgresError as e:
        logging.error(f"Database error in create_and_seed_plot_triggers: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error in create_and_seed_plot_triggers: {e}", exc_info=True)
