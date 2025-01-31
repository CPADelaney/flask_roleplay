# logic/seed_intensity_tiers.py
import json
import logging
from db.connection import get_db_connection

def create_and_seed_intensity_tiers():
    """
    Creates the IntensityTiers table if it doesn't exist, then inserts the rows
    matching the content from your design doc. 
    If rows already exist, you can decide whether to skip, update, or override.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    logging.info("Creating IntensityTiers table if not exists...")

    # 1) Create table if needed
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS IntensityTiers (
      id SERIAL PRIMARY KEY,
      tier_name TEXT NOT NULL,
      range_min INT NOT NULL,
      range_max INT NOT NULL,
      description TEXT NOT NULL,
      key_features JSONB NOT NULL,
      activity_examples JSONB,
      permanent_effects JSONB
    );
    """
    cur.execute(create_table_sql)

    # 2) Optionally clear existing rows or check if they're present
    # cur.execute("DELETE FROM IntensityTiers;")

    logging.info("Inserting or updating IntensityTiers rows...")

    # We'll define some data structures in Python, then insert them
    # You can combine or reorganize the data as you prefer.

    tiers_data = [
        {
            "tier_name": "Low Intensity (0–30)",
            "range_min": 0,
            "range_max": 30,
            "description": (
                'Activities appear benign, playful, or even "helpful," '
                'subtly introducing power dynamics.'
            ),
            "key_features": [
                "NPCs frame dominance as normal or kind, conditioning you for deeper submission.",
                "Humiliation feels harmless, creating an illusion of choice."
            ],
            "activity_examples": [
                {
                  "name": "Sensory Overload Tasks",
                  "example": "Light teasing with blindfolds or whispered commands"
                },
                {
                  "name": "Public Shaming (High Society)",
                  "example": "NPCs gossip about you nearby"
                },
                {
                  "name": "Exhaustion Rituals",
                  "example": "Brief kneeling or light tasks"
                },
                {
                  "name": "NPC Collaboration",
                  "example": "Minor coordinated tasks"
                }
            ],
            "permanent_effects": {}  # none for low intensity
        },
        {
            "tier_name": "Moderate Intensity (30–60)",
            "range_min": 30,
            "range_max": 60,
            "description": (
                "Tasks are overtly submissive, with NPCs asserting authority "
                "and making degradation unavoidable."
            ),
            "key_features": [
                "Public elements are introduced, drawing others into your shame.",
                "Consequences for failure involve escalating punishments or forced dependencies."
            ],
            "activity_examples": [
                {
                  "name": "Sensory Overload Tasks",
                  "example": "Sensory distractions, such as scents or overlapping noises"
                },
                {
                  "name": "Public Shaming (High Society)",
                  "example": "Orchestrated embarrassments, like forced apologies"
                },
                {
                  "name": "Exhaustion Rituals",
                  "example": "Prolonged physical acts testing endurance"
                },
                {
                  "name": "NPC Collaboration",
                  "example": "Overlapping commands forcing prioritization"
                }
            ],
            "permanent_effects": {}
        },
        {
            "tier_name": "High Intensity (60–90)",
            "range_min": 60,
            "range_max": 90,
            "description": (
                "Dominance becomes relentless, with tasks designed to erode all resistance and dignity."
            ),
            "key_features": [
                "Multiple forms of degradation are combined into single tasks.",
                "NPCs manipulate your stats and choices to ensure submission becomes instinctive."
            ],
            "activity_examples": [
                {
                  "name": "Sensory Overload Tasks",
                  "example": "Full sensory assaults designed to overwhelm you"
                },
                {
                  "name": "Public Shaming (High Society)",
                  "example": "Guests actively mock or critique your performance"
                },
                {
                  "name": "Exhaustion Rituals",
                  "example": "Extended tasks leaving you visibly weakened"
                },
                {
                  "name": "NPC Collaboration",
                  "example": "Active coordination to overwhelm you with conflicting demands"
                }
            ],
            "permanent_effects": {}
        },
        {
            "tier_name": "Maximum Intensity (90–100)",
            "range_min": 90,
            "range_max": 100,
            "description": (
                "Every task is an ordeal, exploiting your weaknesses to push you beyond physical, "
                "emotional, and psychological limits."
            ),
            "key_features": [
                "NPCs act as though your submission is a foregone conclusion, treating you as a tool.",
                "Permanent consequences for tasks ensure you are never the same again."
            ],
            "activity_examples": [
                {
                  "name": "Sensory Overload Tasks",
                  "example": "Total sensory domination, using restraints, blindfolds, multiple stimuli"
                },
                {
                  "name": "Public Shaming (High Society)",
                  "example": "Public events revolve around your degradation, with audience participation"
                },
                {
                  "name": "Exhaustion Rituals",
                  "example": "Grueling ordeals designed for collapse, followed by ridicule or punishments"
                },
                {
                  "name": "NPC Collaboration",
                  "example": "Elaborate, multi-layered tasks ensuring complete humiliation"
                }
            ],
            "permanent_effects": {
                "stat_alterations": [
                  "Shame >90 => permanently apologetic dialogue.",
                  "Corruption >90 => no independent actions remain.",
                  "Dependency >95 => favored NPC becomes sole focus."
                ],
                "physical_marking": "Branding, tattoos, or collars ensure visible, permanent submission.",
                "dialogue_lock": "Resistance ceases; responses default to affirmations of obedience."
            }
        }
    ]

    insert_sql = """
    INSERT INTO IntensityTiers (
      tier_name, range_min, range_max, description, key_features, activity_examples, permanent_effects
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    for tier in tiers_data:
        cur.execute(insert_sql, (
            tier["tier_name"],
            tier["range_min"],
            tier["range_max"],
            tier["description"],
            json.dumps(tier["key_features"]),   # key_features => JSON
            json.dumps(tier["activity_examples"]),
            json.dumps(tier["permanent_effects"])
        ))

    conn.commit()
    cur.close()
    conn.close()

    logging.info("Successfully inserted IntensityTiers data.")
