# routes/knowledge_routes.py

from quart import Blueprint, jsonify
from db.connection import get_db_connection_context
from lore.core import canon
import json

knowledge_bp = Blueprint('knowledge_bp', __name__)

# Create a context class for knowledge operations
class KnowledgeContext:
    user_id = 0  # System-level knowledge
    conversation_id = 0

@knowledge_bp.route('/init_knowledge', methods=['POST'])
async def init_knowledge():
    """
    Creates the knowledge tables (PlotTriggers, IntensityTiers, Interactions)
    if they don't exist. Then inserts the detailed data from your doc files.
    """
    try:
        ctx = KnowledgeContext()
        
        await create_knowledge_tables()
        
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Insert knowledge and log as canonical events
                await insert_plot_triggers(ctx, conn)
                await insert_intensity_tiers(ctx, conn)
                await insert_interactions_data(ctx, conn)
                
                # Log the initialization as a canonical event
                await canon.log_canonical_event(
                    ctx, conn,
                    "Knowledge system initialized with plot triggers, intensity tiers, and interactions",
                    tags=["system", "knowledge", "initialization"],
                    significance=10
                )
        
        return jsonify({"message": "Knowledge data initialized successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def create_knowledge_tables():
    """Create the knowledge tables if they don't exist."""
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


async def insert_plot_triggers(ctx, conn):
    """
    Insert your Plot Triggers & Events Knowledge Document data.
    We break them up by stage (Early, Mid, Endgame, etc.)
    plus some stat-driven triggers like 'Trust-Based Betrayal', etc.
    """
    async with conn.cursor() as cursor:
        # Check if PlotTriggers table already has data
        await cursor.execute("SELECT COUNT(*) FROM PlotTriggers")
        result = await cursor.fetchone()
        
        # If table is not empty, exit the function early
        if result and result[0] > 0:
            return

        # Define all plot triggers
        plot_triggers = [
            # Early Stage triggers
            {
                "stage": "Early Stage",
                "title": "Subtle Dominance",
                "stat_requirements": {},
                "description": "Introduces NPCs with playful humiliation, minor acts of control. Tests boundaries.",
                "examples": [
                    "Coworker spills a drink, forcing you to clean it up under watchful eyes.",
                    "An NPC uses a bet to trick you into a small embarrassing act."
                ]
            },
            {
                "stage": "Early Stage",
                "title": "Light Punishments",
                "stat_requirements": {},
                "description": "Teasing or public corrections for minor failures.",
                "examples": [
                    "Verbal teasing in front of colleagues.",
                    "Forced apologies that highlight your inferiority."
                ]
            },
            # Mid-stage triggers
            {
                "stage": "Mid-Stage Escalation",
                "title": "Collaborative Dominance",
                "stat_requirements": {"Dominance": ">50"},
                "description": "NPCs become more assertive, physical tasks, public humiliations, forced rivalries.",
                "examples": [
                    "NPC A and NPC B coordinate conflicting demands to overwhelm you.",
                    "Public contests where you're the main object of ridicule."
                ]
            },
            {
                "stage": "Mid-Stage Escalation",
                "title": "Physical Marking",
                "stat_requirements": {"Corruption": ">40", "Intensity": ">60"},
                "description": "NPCs begin to physically mark their ownership through temporary or semi-permanent means.",
                "examples": [
                    "Lipstick marks left visible on skin or clothing.",
                    "Temporary tattoos or body writing that must remain visible."
                ]
            },
            # Endgame triggers
            {
                "stage": "Endgame",
                "title": "Absolute Submission Ceremony",
                "stat_requirements": {"Corruption": ">90", "Dependency": ">80"},
                "description": "NPCs claim you fully with public ceremonies, branding, or vows of obedience.",
                "examples": [
                    "A formal branding in front of an audience.",
                    "Final vow of loyalty to a favored NPC, overshadowing all else."
                ]
            },
            {
                "stage": "Endgame",
                "title": "Public Display of Ownership",
                "stat_requirements": {"Obedience": ">90", "Shame": ">80"},
                "description": "Your submission becomes a public spectacle for others to witness and participate in.",
                "examples": [
                    "Being displayed as a trophy at a public event.",
                    "Performing degrading acts for an audience's entertainment."
                ]
            },
            # Stat-driven triggers
            {
                "stage": "Stat-Driven",
                "title": "Trust-Based Betrayal",
                "stat_requirements": {"Trust": "<-50"},
                "description": "NPC sets a trap, exposing vulnerabilities, leading to greater corruption and dependency.",
                "examples": [
                    "NPC frames you for failure, causing you to rely on a rival NPC.",
                    "Confidence drops as betrayal shakes your resolve."
                ]
            },
            {
                "stage": "Stat-Driven",
                "title": "Collaborative Mockery",
                "stat_requirements": {"Cruelty": ">70", "Link_Level": ">60"},
                "description": "Multiple NPCs coordinate to humiliate you more effectively than they could alone.",
                "examples": [
                    "NPCs share embarrassing information about you between themselves.",
                    "Coordinated public humiliation with each NPC playing a specific role."
                ]
            },
            {
                "stage": "Stat-Driven",
                "title": "Dependency Exploitation",
                "stat_requirements": {"Dependency": ">70"},
                "description": "NPCs exploit your psychological or physical dependency on them.",
                "examples": [
                    "Withholding affection or approval until you comply with demands.",
                    "Using your addiction triggers to manipulate behavior."
                ]
            }
        ]

        # Insert all plot triggers
        for trigger in plot_triggers:
            await cursor.execute('''
                INSERT INTO PlotTriggers (stage, title, stat_requirements, description, examples)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                trigger["stage"],
                trigger["title"],
                json.dumps(trigger["stat_requirements"]),
                trigger["description"],
                json.dumps(trigger["examples"])
            ))

        # Log as canonical event
        await canon.log_canonical_event(
            ctx, conn,
            f"Plot triggers initialized: {len(plot_triggers)} triggers added covering Early Stage, Mid-Stage, Endgame, and Stat-Driven events",
            tags=["knowledge", "plot_triggers", "system"],
            significance=8
        )


async def insert_intensity_tiers(ctx, conn):
    """
    Insert your IntensityTiers.doc data (0–30 = Low, 30–60 = Moderate, etc.)
    Only run the insertions if the IntensityTiers table is empty.
    """
    async with conn.cursor() as cursor:
        # Check if the IntensityTiers table is empty
        await cursor.execute("SELECT COUNT(*) FROM IntensityTiers")
        result = await cursor.fetchone()
        
        # If table is not empty, exit the function early
        if result and result[0] > 0:
            return

        # Define all intensity tiers
        intensity_tiers = [
            {
                "tier_name": "Low Intensity (0–30)",
                "range_min": 0,
                "range_max": 30,
                "key_features": [
                    "Activities appear benign or playful.",
                    "Humiliation feels harmless, illusions of choice remain.",
                    "Player retains most agency and dignity.",
                    "Consequences are mild and temporary."
                ],
                "activity_examples": [
                    "Minor teasing, whispered commands.",
                    "Light tasks with minimal risk of punishment.",
                    "Playful dares or challenges.",
                    "Subtle power plays disguised as jokes."
                ],
                "permanent_effects": None
            },
            {
                "tier_name": "Moderate Intensity (30–60)",
                "range_min": 30,
                "range_max": 60,
                "key_features": [
                    "Overtly submissive tasks, public elements introduced.",
                    "Consequences for failure escalate (verbal or mild physical).",
                    "NPCs become more demanding and less forgiving.",
                    "Player's comfort zone is consistently challenged."
                ],
                "activity_examples": [
                    "Forced apologies in front of onlookers.",
                    "Prolonged kneeling or physically tiring tasks.",
                    "Public embarrassment as punishment.",
                    "Tasks requiring visible submission."
                ],
                "permanent_effects": None
            },
            {
                "tier_name": "High Intensity (60–90)",
                "range_min": 60,
                "range_max": 90,
                "key_features": [
                    "Dominance becomes relentless; multiple forms of degradation combined.",
                    "Public mockery or group punishments are common.",
                    "Physical and psychological limits are tested.",
                    "Recovery between activities becomes minimal."
                ],
                "activity_examples": [
                    "Extended tasks leaving you visibly exhausted.",
                    "Group humiliations with multiple NPCs.",
                    "Complex degradation scenarios with layered elements.",
                    "Activities designed to break resistance."
                ],
                "permanent_effects": None
            },
            {
                "tier_name": "Maximum Intensity (90–100)",
                "range_min": 90,
                "range_max": 100,
                "key_features": [
                    "Every task is an ordeal, pushing you to physical/mental limits.",
                    "NPCs treat your submission as a foregone conclusion.",
                    "No mercy or reprieve is offered.",
                    "Activities are designed for total subjugation."
                ],
                "activity_examples": [
                    "Grueling ordeals designed for collapse.",
                    "Public events revolve around your total degradation, with audience involvement.",
                    "Extreme endurance challenges with severe consequences.",
                    "Rituals of complete submission and ownership."
                ],
                "permanent_effects": {
                    "stat_alterations": [
                        "Shame >90 => permanently apologetic dialogue.",
                        "Corruption >90 => no independent actions remain.",
                        "Mental Resilience <10 => complete psychological dependency.",
                        "Willpower <10 => automatic compliance with all demands."
                    ],
                    "physical_marking": "Branding, tattoos, or collars ensure visible, permanent submission.",
                    "behavioral_changes": "Reflexive obedience, inability to make eye contact with dominants.",
                    "narrative_locks": "Certain escape or resistance options become permanently unavailable."
                }
            }
        ]

        # Insert all intensity tiers
        for tier in intensity_tiers:
            await cursor.execute('''
                INSERT INTO IntensityTiers (tier_name, range_min, range_max, key_features, activity_examples, permanent_effects)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                tier["tier_name"],
                tier["range_min"],
                tier["range_max"],
                json.dumps(tier["key_features"]),
                json.dumps(tier["activity_examples"]),
                json.dumps(tier["permanent_effects"]) if tier["permanent_effects"] else None
            ))

        # Log as canonical event
        await canon.log_canonical_event(
            ctx, conn,
            f"Intensity tiers initialized: {len(intensity_tiers)} tiers defined from Low to Maximum intensity",
            tags=["knowledge", "intensity_tiers", "system"],
            significance=8
        )


async def insert_interactions_data(ctx, conn):
    """
    Insert data from Interactions.doc describing success/failure rules,
    examples of tasks, and agency overrides.
    """
    async with conn.cursor() as cursor:
        # Check if the Interactions table is empty
        await cursor.execute("SELECT COUNT(*) FROM Interactions")
        result = await cursor.fetchone()
        
        # If table is not empty, exit the function early
        if result and result[0] > 0:
            return

        # Define interaction rules
        interactions = [
            {
                "interaction_name": "Weighted Success/Failure",
                "detailed_rules": {
                    "base_success": "Calculated from Confidence, Willpower, etc. vs. NPC Dominance & Intensity.",
                    "randomization": "5-15% variation from the base rate, allowing critical outcomes.",
                    "failure_consequences": "Stat penalties, punishments, or narrative changes.",
                    "critical_success": "5% chance to exceed expectations, gaining temporary stat boosts.",
                    "critical_failure": "5% chance of catastrophic failure with severe consequences.",
                    "npc_mood_modifier": "NPC's current emotional state affects difficulty (+/- 10%).",
                    "environmental_factors": "Location, time of day, and witnesses affect outcomes."
                },
                "task_examples": {
                    "non_npc": [
                        "Sabotaged research requiring Shame & Mental Resilience to recover.",
                        "Physical challenges testing Endurance, risking collapse.",
                        "Social situations where Confidence determines success.",
                        "Puzzle-solving affected by Mental Resilience and stress."
                    ],
                    "npc_driven": [
                        "Collaborative punishments (two NPCs giving overlapping commands).",
                        "Progressive degradation (clean shoes → kiss them → lick them publicly).",
                        "Endurance tests with escalating difficulty.",
                        "Social humiliation tasks requiring specific responses."
                    ]
                },
                "agency_overrides": {
                    "obedience_override": ">80 => tasks completed reflexively, no roll needed",
                    "corruption_override": ">90 => no defiance possible, automatic failure of resistance",
                    "willpower_override": "<20 => defiance extremely rare, -30% to all success rates",
                    "dependency_override": ">80 => cannot refuse requests from bonded NPCs",
                    "mental_resilience_override": "<20 => confusion and compliance increase dramatically"
                }
            },
            {
                "interaction_name": "Relationship-Based Modifiers",
                "detailed_rules": {
                    "trust_impact": "Positive trust increases success rates, negative trust adds penalties.",
                    "dominance_scaling": "Higher NPC dominance increases task difficulty exponentially.",
                    "closeness_factor": "High closeness can either help (mercy) or hinder (exploitation).",
                    "cruelty_multiplier": "NPC cruelty amplifies failure consequences.",
                    "intensity_progression": "Each interaction increases future intensity requirements."
                },
                "task_examples": {
                    "high_trust": [
                        "NPC gives clearer instructions or hints.",
                        "Failure consequences are reduced.",
                        "Opportunities for redemption after mistakes."
                    ],
                    "low_trust": [
                        "Deliberately vague or contradictory commands.",
                        "Harsher punishments for minor infractions.",
                        "No second chances or explanations."
                    ]
                },
                "agency_overrides": {
                    "respect_threshold": "Respect < -50 => NPC actively sabotages player efforts",
                    "closeness_threshold": "Closeness > 80 => NPC may show unexpected mercy",
                    "trust_betrayal": "Trust < -80 => NPC sets traps and false tasks"
                }
            },
            {
                "interaction_name": "Group Dynamics System",
                "detailed_rules": {
                    "coordination_bonus": "Multiple NPCs working together increase difficulty by 20% per NPC.",
                    "conflicting_demands": "Contradictory orders from different NPCs create no-win scenarios.",
                    "peer_pressure": "Witness NPCs add social pressure, affecting Shame and Confidence.",
                    "hierarchy_effects": "Higher-status NPCs can override lower-status NPC commands.",
                    "mob_mentality": "Groups of 3+ NPCs can trigger special humiliation events."
                },
                "task_examples": {
                    "coordinated_tasks": [
                        "One NPC distracts while another sets up humiliation.",
                        "Sequential tasks that build on previous degradation.",
                        "Group punishments where each NPC adds an element."
                    ],
                    "conflict_scenarios": [
                        "Two NPCs demand contradictory actions simultaneously.",
                        "Choosing which NPC to obey affects relationships with all.",
                        "Group politics where pleasing one angers others."
                    ]
                },
                "agency_overrides": {
                    "outnumbered": "3+ NPCs present => -20% to all resistance attempts",
                    "group_coercion": "Unanimous NPC agreement => automatic compliance if Willpower < 40",
                    "social_paralysis": "Public humiliation with 5+ witnesses => Confidence drops by 10"
                }
            }
        ]

        # Insert all interactions
        for interaction in interactions:
            await cursor.execute('''
                INSERT INTO Interactions (interaction_name, detailed_rules, task_examples, agency_overrides)
                VALUES (%s, %s, %s, %s)
            ''', (
                interaction["interaction_name"],
                json.dumps(interaction["detailed_rules"]),
                json.dumps(interaction["task_examples"]),
                json.dumps(interaction["agency_overrides"])
            ))

        # Log as canonical event
        await canon.log_canonical_event(
            ctx, conn,
            f"Interaction rules initialized: {len(interactions)} rule systems defined for gameplay mechanics",
            tags=["knowledge", "interactions", "system", "mechanics"],
            significance=8
        )
