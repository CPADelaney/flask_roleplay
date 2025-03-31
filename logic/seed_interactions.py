# logic/seed_interactions.py
import json
import logging
import asyncio
import asyncpg
from db.connection import get_db_connection_context

async def create_and_seed_interactions():
    """
    Creates the 'Interactions' table rows based on Document 15.
    If you want them in multiple rows, each with:
      - interaction_name
      - detailed_rules (a JSON representation of your doc text)
      - task_examples (for the examples from the doc)
      - agency_overrides (for the stat thresholds)
    """
    logging.info("Inserting or updating 'Interactions' rows...")

    # You can define a list of sections, each becoming a row in Interactions
    interactions_data = [
        {
            "interaction_name": "Weighted Success/Failure Rules",
            "detailed_rules": {
                "purpose": "Defines success/failure logic with randomization and stat interplay.",
                "base_success_rates": {
                    "your_stats": [
                        "Confidence => social challenges",
                        "Willpower => resisting commands",
                        "Shame => emotional response to humiliation",
                        "Corruption => blocks resistance at 90+",
                        "Obedience => reflexive compliance above 80",
                        "Physical Endurance => handle grueling tasks",
                        "Mental Resilience => endure psychological torment",
                        "Lust => locks out defiance above 90",
                        "Dependency => prioritizes favored NPC"
                    ],
                    "task_difficulty": "Easy, Moderate, Hard, or Impossible baseline success rates",
                    "npc_stats": [
                        "Dominance, Intensity => amplify difficulty",
                        "Cruelty => elaborate punishments for failures"
                    ]
                },
                "random_outcomes": {
                    "example": "Confidence 30 vs. NPC Dominance 80 => base ~10%, random 5–15%",
                    "critical_success": "Perfect execution => rare reward or partial stat recovery",
                    "critical_failure": "Disastrous => unique punishments or compounded penalties"
                },
                "failure_consequences": {
                    "stat_penalties": "e.g. public humiliation => -10 Confidence, +10 Corruption, +5 Shame",
                    "npc_reactions": "increased Dominance or collaborative punishments",
                    "narrative_impact": "failures permanently alter progression (NPC trust, sabotage, etc.)"
                }
            },
            "task_examples": {
                # You might leave these empty if row #2 covers tasks
            },
            "agency_overrides": {
                "obedience_over_80": "Reflexive task completion",
                "corruption_over_90": "No resistance possible",
                "willpower_below_20": "Defiance extremely rare",
                "lust_over_90": "Intimate tasks obeyed w/o hesitation",
                "dependency_over_80": "Favored NPC demands override everything"
            }
        },
        {
            "interaction_name": "Examples of Tasks",
            "detailed_rules": {
                "non_npc_challenges": [
                    {
                        "task_name": "Sabotaged Research",
                        "description": "NPC interference => need Shame & Mental Resilience to ask for help",
                        "failure_consequence": "Increased Dependency on a rival NPC offering assistance"
                    },
                    {
                        "task_name": "Physical Challenges",
                        "description": "Prolonged kneeling => tests Physical Endurance",
                        "critical_failure": "Collapse => NPC ridicule & punishment"
                    }
                ],
                "npc_driven_tasks": [
                    {
                        "task_name": "Collaborative Punishments",
                        "example": "Two NPCs => overlapping commands => test Endurance & Resilience"
                    },
                    {
                        "task_name": "Progressive Degradation",
                        "example": [
                            "Low: cleaning shoes",
                            "Moderate: kissing them",
                            "Max: licking them in public"
                        ]
                    }
                ]
            },
            "task_examples": {},  # or left empty
            "agency_overrides": {}
        },
        {
            "interaction_name": "Advanced Mechanics",
            "detailed_rules": {
                "multi_layered_tasks": {
                    "example": "Sabotaged Scavenger Task => Stage 1 (Confidence/Willpower), Stage 2 (Shame/MentalResilience), Stage 3 (Corruption/PhysicalEndurance)."
                },
                "cumulative_effects": {
                    "explanation": "Failing tasks compounds penalties => higher difficulty next time."
                },
                "npc_exploitation": {
                    "low_endurance": "NPCs escalate physical tasks to ensure collapse",
                    "high_lust": "NPCs introduce sensual rewards to manipulate focus",
                    "high_dependency": "Rival NPCs exploit favoritism to sow conflict"
                }
            },
            "task_examples": {},
            "agency_overrides": {}
        },
        {
            "interaction_name": "Example Narrative Use Case",
            "detailed_rules": {
                "scenario": "Public Submission at a Corporate Event",
                "context": "Boss & HR Manager collaborate to humiliate you in a meeting",
                "modifiers": {
                    "Confidence": 20,
                    "Lust": 85,
                    "DominanceNPC1": 80,
                    "DominanceNPC2": 70,
                    "PhysicalEndurance": 25
                },
                "stages": [
                    "Stage 1: humiliating apology (Willpower/Shame)",
                    "Stage 2: Clean Boss's shoes (MentalResilience vs. taunts)",
                    "Stage 3: Accept public punishment (Corruption & PhysicalEndurance)"
                ],
                "outcomes": {
                    "success": "slight stat recovery, increased closeness",
                    "failure": "loss of Confidence, +Corruption, permanent Respect loss"
                }
            },
            "task_examples": {},
            "agency_overrides": {}
        }
    ]

    try:
        async with get_db_connection_context() as conn:
            # Insert or update logic
            for row_data in interactions_data:
                await conn.execute("""
                INSERT INTO Interactions (interaction_name, detailed_rules, task_examples, agency_overrides)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (interaction_name)
                DO UPDATE
                SET
                    detailed_rules=EXCLUDED.detailed_rules,
                    task_examples=EXCLUDED.task_examples,
                    agency_overrides=EXCLUDED.agency_overrides
                """, 
                row_data["interaction_name"],
                json.dumps(row_data["detailed_rules"]),
                json.dumps(row_data["task_examples"]),
                json.dumps(row_data["agency_overrides"])
                )
            
            logging.info("Seeded 'Interactions' data successfully.")
    except asyncpg.PostgresError as e:
        logging.error(f"Database error in create_and_seed_interactions: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error in create_and_seed_interactions: {e}", exc_info=True)
