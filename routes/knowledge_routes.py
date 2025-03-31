# routes/knowledge_routes.py

from flask import Blueprint, jsonify
from db.connection import get_db_connection_context
import json

knowledge_bp = Blueprint('knowledge_bp', __name__)

@knowledge_bp.route('/init_knowledge', methods=['POST'])
async def init_knowledge():
    """
    Creates the knowledge tables (PlotTriggers, IntensityTiers, Interactions)
    if they don't exist. Then inserts the detailed data from your doc files.
    """
    try:
        await create_knowledge_tables()
        await insert_plot_triggers()
        await insert_intensity_tiers()
        await insert_interactions_data()
        return jsonify({"message": "Knowledge data initialized successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def create_knowledge_tables():
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                CREATE TABLE IF NOT EXISTS PlotTriggers (
                    id SERIAL PRIMARY KEY,
                    stage TEXT NOT NULL,            -- e.g. "Early Stage", "Mid-Stage Escalation", "Endgame"
                    title TEXT NOT NULL,            -- e.g. "Collaborative Mockery", "Public Display of Marking"
                    stat_requirements JSONB,        -- any numeric or threshold-based conditions
                    description TEXT,               -- main narrative chunk
                    examples JSONB                  -- list of example micro-narratives
                );
            ''')

            await cursor.execute('''
                CREATE TABLE IF NOT EXISTS IntensityTiers (
                    id SERIAL PRIMARY KEY,
                    tier_name TEXT NOT NULL,        -- e.g. "Low Intensity (0–30)"
                    range_min INT NOT NULL,
                    range_max INT NOT NULL,
                    key_features JSONB,            -- bullet points from doc
                    activity_examples JSONB,       -- short examples
                    permanent_effects JSONB        -- for maximum intensity or major changes
                );
            ''')

            await cursor.execute('''
                CREATE TABLE IF NOT EXISTS Interactions (
                    id SERIAL PRIMARY KEY,
                    interaction_name TEXT NOT NULL,   -- e.g. "Weighted Success/Failure Rules"
                    detailed_rules JSONB,            -- big chunk describing how you handle success/failure
                    task_examples JSONB,             -- e.g. "non_npc_challenges", "npc_driven_tasks"
                    agency_overrides JSONB           -- e.g. thresholds for Obedience, Corruption, etc.
                );
            ''')

        await conn.commit()


async def insert_plot_triggers():
    """
    Insert your Plot Triggers & Events Knowledge Document data.
    We break them up by stage (Early, Mid, Endgame, etc.)
    plus some stat-driven triggers like 'Trust-Based Betrayal', etc.
    """
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            # Example: Early Stage
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Early Stage",
                "Subtle Dominance",
                json.dumps({}),  # No strong stat requirement, or empty
                "Introduces NPCs with playful humiliation, minor acts of control. Tests boundaries.",
                json.dumps([
                    "Coworker spills a drink, forcing you to clean it up under watchful eyes.",
                    "An NPC uses a bet to trick you into a small embarrassing act."
                ])
            ))

            # Another early stage event
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Early Stage",
                "Light Punishments",
                json.dumps({}),
                "Teasing or public corrections for minor failures.",
                json.dumps([
                    "Verbal teasing in front of colleagues.",
                    "Forced apologies that highlight your inferiority."
                ])
            ))

            # Mid-stage
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Mid-Stage Escalation",
                "Collaborative Dominance",
                json.dumps({"Dominance": ">50"}),  # example threshold
                "NPCs become more assertive, physical tasks, public humiliations, forced rivalries.",
                json.dumps([
                    "NPC A and NPC B coordinate conflicting demands to overwhelm you.",
                    "Public contests where you're the main object of ridicule."
                ])
            ))

            # Endgame
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Endgame",
                "Absolute Submission Ceremony",
                json.dumps({"Corruption": ">90", "Dependency": ">80"}),
                "NPCs claim you fully with public ceremonies, branding, or vows of obedience.",
                json.dumps([
                    "A formal branding in front of an audience.",
                    "Final vow of loyalty to a favored NPC, overshadowing all else."
                ])
            ))

            # Additional: Trust-Based Betrayal, Collaborative Mockery, etc.
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Stat-Driven",
                "Trust-Based Betrayal",
                json.dumps({"Trust": "<-50"}),
                "NPC sets a trap, exposing vulnerabilities, leading to greater corruption and dependency.",
                json.dumps([
                    "NPC frames you for failure, causing you to rely on a rival NPC.",
                    "Confidence drops as betrayal shakes your resolve."
                ])
            ))

        await conn.commit()


async def insert_intensity_tiers():
    """
    Insert your IntensityTiers.doc data (0–30 = Low, 30–60 = Moderate, etc.).
    """
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            # Low Intensity
            await cursor.execute('''
                INSERT INTO IntensityTiers (tier_name, range_min, range_max, key_features, activity_examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Low Intensity (0–30)",
                0, 30,
                json.dumps([
                    "Activities appear benign or playful.",
                    "Humiliation feels harmless, illusions of choice remain."
                ]),
                json.dumps([
                    "Minor teasing, whispered commands.",
                    "Light tasks with minimal risk of punishment."
                ])
            ))

            # Moderate Intensity
            await cursor.execute('''
                INSERT INTO IntensityTiers (tier_name, range_min, range_max, key_features, activity_examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "Moderate Intensity (30–60)",
                30, 60,
                json.dumps([
                    "Overtly submissive tasks, public elements introduced.",
                    "Consequences for failure escalate (verbal or mild physical)."
                ]),
                json.dumps([
                    "Forced apologies in front of onlookers.",
                    "Prolonged kneeling or physically tiring tasks."
                ])
            ))

            # High Intensity
            await cursor.execute('''
                INSERT INTO IntensityTiers (tier_name, range_min, range_max, key_features, activity_examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                "High Intensity (60–90)",
                60, 90,
                json.dumps([
                    "Dominance becomes relentless; multiple forms of degradation combined.",
                    "Public mockery or group punishments are common."
                ]),
                json.dumps([
                    "Extended tasks leaving you visibly exhausted.",
                    "Group humiliations with multiple NPCs."
                ])
            ))

            # Maximum Intensity
            await cursor.execute('''
                INSERT INTO IntensityTiers (tier_name, range_min, range_max, key_features, activity_examples, permanent_effects)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                "Maximum Intensity (90–100)",
                90, 100,
                json.dumps([
                    "Every task is an ordeal, pushing you to physical/mental limits.",
                    "NPCs treat your submission as a foregone conclusion."
                ]),
                json.dumps([
                    "Grueling ordeals designed for collapse.",
                    "Public events revolve around your total degradation, with audience involvement."
                ]),
                json.dumps({
                    "stat_alterations": [
                        "Shame >90 => permanently apologetic dialogue.",
                        "Corruption >90 => no independent actions remain."
                    ],
                    "physical_marking": "Branding, tattoos, or collars ensure visible, permanent submission."
                })
            ))

        await conn.commit()


async def insert_interactions_data():
    """
    Insert data from Interactions.doc describing success/failure rules,
    examples of tasks, and agency overrides.
    """
    async with get_db_connection_context() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                INSERT INTO Interactions (interaction_name, detailed_rules, task_examples, agency_overrides)
                VALUES (%s, %s, %s, %s)
            ''', (
                "Weighted Success/Failure",
                json.dumps({
                    "base_success": "Calculated from Confidence, Willpower, etc. vs. NPC Dominance & Intensity.",
                    "randomization": "5-15% variation from the base rate, allowing critical outcomes.",
                    "failure_consequences": "Stat penalties, punishments, or narrative changes."
                }),
                json.dumps({
                    "non_npc": [
                        "Sabotaged research requiring Shame & Mental Resilience to recover.",
                        "Physical challenges testing Endurance, risking collapse."
                    ],
                    "npc_driven": [
                        "Collaborative punishments (two NPCs giving overlapping commands).",
                        "Progressive degradation (clean shoes → kiss them → lick them publicly)."
                    ]
                }),
                json.dumps({
                    "obedience_override": ">80 => tasks completed reflexively",
                    "corruption_override": ">90 => no defiance possible",
                    "willpower_override": "<20 => defiance extremely rare"
                })
            ))

        await conn.commit()
