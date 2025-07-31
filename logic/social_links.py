# logic/social_links_agentic.py
"""
Comprehensive End-to-End Social Links System with an Agentic approach using OpenAI's Agents SDK.
Refactored to use the new architecture with canon and LoreSystem.

Features:
1) Core CRUD and advanced relationship logic for SocialLinks.
2) Relationship dynamics, Crossroads, Ritual checking, multi-dimensional relationships.
3) Function tools bridging your existing Python logic so the LLM-based agent can invoke them.
4) A "SocialLinksAgent" that uses these tools.
5) Example usage demonstrating how to run the agent in code.
"""

import json
import logging
import random
import asyncio
from datetime import datetime, timedelta

from logic.stats_logic import calculate_social_insight, get_all_player_stats
from logic.relationship_integration import RelationshipIntegration
from typing import Dict, Any, Optional, List, Union, Tuple

import asyncpg

# Import canon and LoreSystem
from lore.core import canon
from lore.core.lore_system import LoreSystem

# ~~~~~~~~~ Agents SDK imports ~~~~~~~~~
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    AsyncOpenAI,
    OpenAIResponsesModel
)

# ~~~~~~~~~ DB imports & any other placeholders ~~~~~~~~~
from db.connection import get_db_connection_context

# ~~~~~~~~~ Logging Configuration ~~~~~~~~~
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Simple Core CRUD for SocialLinks Table
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_social_link(
    ctx,
    conn,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch an existing social link row using provided connection.
    Returns a dict with link details or None if not found.
    """
    try:
        row = await conn.fetchrow(
            """
            SELECT link_id, link_type, link_level, link_history, dynamics,
                   experienced_crossroads, experienced_rituals
            FROM SocialLinks
            WHERE user_id = $1 AND conversation_id = $2
              AND entity1_type = $3 AND entity1_id = $4
              AND entity2_type = $5 AND entity2_id = $6
            """,
            ctx.user_id, ctx.conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
        )
        if row:
            history = row['link_history'] if isinstance(row['link_history'], list) else (json.loads(row['link_history']) if isinstance(row['link_history'], str) else [])
            dynamics = row['dynamics'] if isinstance(row['dynamics'], dict) else (json.loads(row['dynamics']) if isinstance(row['dynamics'], str) else {})
            crossroads = row['experienced_crossroads'] if isinstance(row['experienced_crossroads'], list) else (json.loads(row['experienced_crossroads']) if isinstance(row['experienced_crossroads'], str) else [])
            rituals = row['experienced_rituals'] if isinstance(row['experienced_rituals'], list) else (json.loads(row['experienced_rituals']) if isinstance(row['experienced_rituals'], str) else [])

            return {
                "link_id": row['link_id'],
                "link_type": row['link_type'],
                "link_level": row['link_level'],
                "link_history": history,
                "dynamics": dynamics,
                "experienced_crossroads": crossroads,
                "experienced_rituals": rituals,
            }
        return None
    except Exception as e:
        logger.error(f"Error getting social link: {e}", exc_info=True)
        return None


async def create_social_link(
    ctx,
    conn,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    link_type: str = "neutral",
    link_level: int = 0,
    initial_dynamics: Optional[Dict] = None
) -> Optional[int]:
    """
    Create a new SocialLinks row using canon.
    Returns the link_id (new or existing).
    """
    # Prepare data package for canon
    link_data = {
        "user_id": ctx.user_id,
        "conversation_id": ctx.conversation_id,
        "entity1_type": entity1_type,
        "entity1_id": entity1_id,
        "entity2_type": entity2_type,
        "entity2_id": entity2_id,
        "link_type": link_type,
        "link_level": link_level,
        "link_history": [],
        "dynamics": initial_dynamics or {},
        "experienced_crossroads": [],
        "experienced_rituals": []
    }
    
    # Call canon function to create social link
    link_id = await canon.find_or_create_social_link(ctx, conn, **link_data)
    return link_id


async def update_link_type_and_level(
    ctx,
    link_id: int,
    new_type: Optional[str] = None,
    level_change: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Adjust an existing link's type or level using LoreSystem.
    Returns updated info or None if not found.
    """
    # Get LoreSystem instance
    lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
    
    # Prepare updates
    updates = {}
    if new_type is not None:
        updates["link_type"] = new_type
    if level_change != 0:
        # We need to fetch current level first
        async with get_db_connection_context() as conn:
            current = await conn.fetchrow(
                "SELECT link_level FROM SocialLinks WHERE link_id = $1",
                link_id
            )
            if current:
                updates["link_level"] = current['link_level'] + level_change
            else:
                return None
    
    if not updates:
        return None
    
    # Use LoreSystem to update
    result = await lore_system.propose_and_enact_change(
        ctx=ctx,
        entity_type="SocialLinks",
        entity_identifier={"link_id": link_id},
        updates=updates,
        reason=f"Updating link type/level: {updates}"
    )
    
    if result["status"] == "committed":
        return {
            "link_id": link_id,
            "new_type": updates.get("link_type"),
            "new_level": updates.get("link_level"),
        }
    else:
        logger.warning(f"Failed to update link {link_id}: {result}")
        return None


async def add_link_event(
    ctx,
    link_id: int,
    event_text: str
) -> bool:
    """
    Append an event string to link_history using LoreSystem.
    """
    # Get current link history
    async with get_db_connection_context() as conn:
        current = await conn.fetchrow(
            "SELECT link_history FROM SocialLinks WHERE link_id = $1",
            link_id
        )
        if not current:
            return False
        
        history = current['link_history'] if isinstance(current['link_history'], list) else json.loads(current['link_history'] or '[]')
        history.append(event_text)
    
    # Update using LoreSystem
    lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
    result = await lore_system.propose_and_enact_change(
        ctx=ctx,
        entity_type="SocialLinks",
        entity_identifier={"link_id": link_id},
        updates={"link_history": json.dumps(history)},
        reason=f"Adding event to link history: {event_text}"
    )
    
    return result["status"] == "committed"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Relationship Dynamics, Crossroads, Rituals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RELATIONSHIP_DYNAMICS = [
    {
        "name": "control",
        "description": "One entity exerts control over the other",
        "levels": [
            {"level": 10, "name": "Subtle Influence", "description": "Occasional suggestions that subtly guide behavior"},
            {"level": 30, "name": "Regular Direction", "description": "Frequent guidance that shapes decisions"},
            {"level": 50, "name": "Strong Authority", "description": "Explicit direction with expectation of compliance"},
            {"level": 70, "name": "Dominant Control", "description": "Commands given with assumption of obedience"},
            {"level": 90, "name": "Complete Dominance", "description": "Total control with no expectation of resistance"},
        ],
    },
    {
        "name": "manipulation",
        "description": "One entity manipulates the other through indirect means",
        "levels": [
            {"level": 10, "name": "Minor Misdirection", "description": "Occasional white lies to achieve small goals"},
            {"level": 30, "name": "Regular Deception", "description": "Consistent pattern of misleading to shape behavior"},
            {"level": 50, "name": "Calculated Manipulation", "description": "Strategic dishonesty to achieve control"},
            {"level": 70, "name": "Psychological Conditioning", "description": "Systematic reshaping of target's reality"},
            {"level": 90, "name": "Complete Gaslighting", "description": "Target's entire perception is controlled"},
        ],
    },
    {
        "name": "dependency",
        "description": "One entity becomes dependent on the other",
        "levels": [
            {"level": 10, "name": "Mild Attachment", "description": "Enjoys presence but functions fine independently"},
            {"level": 30, "name": "Regular Reliance", "description": "Seeks input for decisions and emotional support"},
            {"level": 50, "name": "Strong Dependency", "description": "Difficulty making decisions without approval"},
            {"level": 70, "name": "Heavy Dependency", "description": "Significant anxiety when separated"},
            {"level": 90, "name": "Complete Dependence", "description": "Unable to function independently"},
        ],
    },
    {
        "name": "alliance",
        "description": "Entities work together for mutual benefit",
        "levels": [
            {"level": 10, "name": "Casual Cooperation", "description": "Occasional collaboration when convenient"},
            {"level": 30, "name": "Regular Partnership", "description": "Consistent working relationship"},
            {"level": 50, "name": "Strong Alliance", "description": "Prioritize mutual support in most situations"},
            {"level": 70, "name": "Deep Partnership", "description": "Significant investment in each other's success"},
            {"level": 90, "name": "Unbreakable Alliance", "description": "Complete mutual commitment"},
        ],
    },
    {
        "name": "rivalry",
        "description": "Entities compete with each other",
        "levels": [
            {"level": 10, "name": "Mild Competition", "description": "Occasional friendly contests"},
            {"level": 30, "name": "Regular Rivalry", "description": "Consistent pattern of trying to outdo each other"},
            {"level": 50, "name": "Strong Competition", "description": "Active attempts to undermine each other"},
            {"level": 70, "name": "Intense Rivalry", "description": "Significant efforts to dominate the other"},
            {"level": 90, "name": "Bitter Enemies", "description": "Complete dedication to defeating the other"},
        ],
    },
]

RELATIONSHIP_CROSSROADS = [
    {
        "name": "Trust Test",
        "description": "A moment where trust is either solidified or broken",
        "trigger_level": 40,
        "dynamic": "control",
        "choices": [
            {
                "text": "Allow increased control",
                "effects": {"control": +15, "dependency": +10, "player_willpower": -5},
                "outcome": "You accept {npc_name}'s explanation and allow her greater involvement in your decisions. The relief in surrendering control is both unexpected and concerning.",
            },
            {
                "text": "Establish boundary",
                "effects": {"control": -10, "rivalry": +15, "player_willpower": +5},
                "outcome": "You firmly establish a boundary with {npc_name}. Her expression flickers between surprise and something colder before she composes herself.",
            },
        ],
    },
    {
        "name": "Public Display",
        "description": "A test of compliance in a public setting",
        "trigger_level": 60,
        "dynamic": "control",
        "choices": [
            {
                "text": "Comply despite discomfort",
                "effects": {"control": +20, "manipulation": +10, "player_confidence": -10},
                "outcome": "You follow {npc_name}'s instructions despite your discomfort. The approval in her eyes provides a confusing sense of validation.",
            },
            {
                "text": "Refuse publicly",
                "effects": {"control": -15, "rivalry": +20, "player_confidence": +5},
                "outcome": "You refuse {npc_name}'s request, causing a momentary tension. Later, she approaches you with a different demeanor, reassessing her approach.",
            },
        ],
    },
    {
        "name": "Manipulation Revealed",
        "description": "Player discovers evidence of manipulation",
        "trigger_level": 50,
        "dynamic": "manipulation",
        "choices": [
            {
                "text": "Confront directly",
                "effects": {"manipulation": -10, "rivalry": +15, "player_mental_resilience": +10},
                "outcome": "You confront {npc_name} about her deception. She seems genuinely caught off-guard by your assertion, quickly adapting with a new approach.",
            },
            {
                "text": "Pretend not to notice",
                "effects": {"manipulation": +15, "dependency": +5, "player_mental_resilience": -10},
                "outcome": "You keep your discovery to yourself, watching as {npc_name} continues her manipulations with growing confidence, unaware that you see through them.",
            },
        ],
    },
    {
        "name": "Support Need",
        "description": "NPC appears to need emotional support",
        "trigger_level": 30,
        "dynamic": "dependency",
        "choices": [
            {
                "text": "Provide unconditional support",
                "effects": {"dependency": +20, "manipulation": +15, "player_corruption": +10},
                "outcome": "You offer complete support to {npc_name}, prioritizing her needs above your own. Her vulnerability feels oddly calculated, but the bond strengthens.",
            },
            {
                "text": "Offer limited support",
                "effects": {"dependency": -5, "rivalry": +5, "player_corruption": -5},
                "outcome": "You offer support while maintaining some distance. {npc_name} seems disappointed but respects your boundaries, adjusting her approach accordingly.",
            },
        ],
    },
    {
        "name": "Alliance Opportunity",
        "description": "Chance to deepen alliance with significant commitment",
        "trigger_level": 40,
        "dynamic": "alliance",
        "choices": [
            {
                "text": "Commit fully to alliance",
                "effects": {"alliance": +25, "dependency": +10, "player_corruption": +5},
                "outcome": "You fully commit to your partnership with {npc_name}, integrating your goals with hers. The efficiency of your collaboration masks the gradual shift in power.",
            },
            {
                "text": "Maintain independence",
                "effects": {"alliance": -10, "manipulation": +5, "player_corruption": -5},
                "outcome": "You maintain some independence from {npc_name}'s influence. She seems to accept this with grace, though you notice new, more subtle approaches to integration.",
            },
        ],
    },
]

RELATIONSHIP_RITUALS = [
    {
        "name": "Formal Agreement",
        "description": "A formalized arrangement that defines relationship expectations",
        "trigger_level": 60,
        "dynamics": ["control", "alliance"],
        "ritual_text": (
            "{npc_name} presents you with an arrangement that feels strangely formal. "
            "'Let's be clear about our expectations,' she says with an intensity that makes it more than casual. "
            "The terms feel reasonable, almost beneficial, yet something about the ritual makes you acutely aware of a threshold being crossed."
        ),
    },
    {
        "name": "Symbolic Gift",
        "description": "A gift with deeper symbolic meaning that represents the relationship dynamic",
        "trigger_level": 50,
        "dynamics": ["control", "dependency"],
        "ritual_text": (
            "{npc_name} presents you with a gift - {gift_item}. "
            "'A small token,' she says, though her expression suggests deeper significance. "
            "As you accept it, the weight feels heavier than the object itself, as if you're accepting something beyond the physical item."
        ),
    },
    {
        "name": "Private Ceremony",
        "description": "A private ritual that solidifies the relationship's nature",
        "trigger_level": 75,
        "dynamics": ["control", "dependency", "manipulation"],
        "ritual_text": (
            "{npc_name} leads you through what she describes as 'a small tradition' - a sequence of actions and words "
            "that feels choreographed for effect. The intimacy of the moment creates a strange blend of comfort and vulnerability, "
            "like something is being sealed between you."
        ),
    },
    {
        "name": "Public Declaration",
        "description": "A public acknowledgment of the relationship's significance",
        "trigger_level": 70,
        "dynamics": ["alliance", "control"],
        "ritual_text": (
            "At the gathering, {npc_name} makes a point of publicly acknowledging your relationship in front of others. "
            "The words seem innocuous, even complimentary, but the subtext feels laden with meaning. "
            "You notice others' reactions - knowing glances, subtle nods - as if your position has been formally established."
        ),
    },
    {
        "name": "Shared Secret",
        "description": "Disclosure of sensitive information that creates mutual vulnerability",
        "trigger_level": 55,
        "dynamics": ["alliance", "manipulation"],
        "ritual_text": (
            "{npc_name} shares information with you that feels dangerously private. 'I don't tell this to just anyone,' she says "
            "with meaningful eye contact. The knowledge creates an intimacy that comes with implicit responsibility - "
            "you're now a keeper of her secrets, for better or worse."
        ),
    },
]

SYMBOLIC_GIFTS = [
    "a delicate bracelet that feels oddly like a subtle marker of ownership",
    "a key to her home that represents more than just physical access",
    "a personalized item that shows how closely she's been observing your habits",
    "a journal with the first few pages already filled in her handwriting, guiding your thoughts",
    "a piece of jewelry that others in her circle would recognize as significant",
    "a custom phone with 'helpful' modifications already installed",
    "a clothing item that subtly alters how others perceive you in her presence",
]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3) Support Functions for Relationship Dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_primary_dynamic(dynamics: Dict[str, int]) -> str:
    """
    Determine the primary relationship dynamic based on highest numeric level in 'dynamics'.
    """
    if not dynamics:
        return "neutral"
    primary_dynamic = "neutral"
    max_level = -float('inf')
    for dname, lvl in dynamics.items():
        if lvl > max_level:
            max_level = lvl
            primary_dynamic = dname
    if max_level <= 0 and dynamics:
         primary_dynamic = max(dynamics, key=dynamics.get)

    return primary_dynamic


def get_dynamic_description(dynamic_name: str, level: int) -> str:
    """
    Get the appropriate textual description for a dynamic at a specific level.
    """
    for dyn in RELATIONSHIP_DYNAMICS:
        if dyn["name"] == dynamic_name:
            matched_level = None
            for level_info in sorted(dyn["levels"], key=lambda x: x["level"]):
                if level <= level_info["level"]:
                    matched_level = level_info
                    break
            if not matched_level:
                matched_level = dyn["levels"][-1]
            return f"{matched_level['name']}: {matched_level['description']}"
    return "Unknown dynamic"

async def get_relationship_depth_multiplier(
    conn: asyncpg.Connection,
    user_id: int,
    conversation_id: int,
    npc_id: int
) -> float:
    """
    Calculate relationship depth multiplier based on empathy and link level.
    Higher empathy + stronger relationship = better understanding.
    """
    # Get player empathy
    player_stats = await conn.fetchrow("""
        SELECT empathy, intelligence FROM PlayerStats
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, user_id, conversation_id)
    
    empathy = player_stats['empathy'] if player_stats else 10
    intelligence = player_stats['intelligence'] if player_stats else 10
    
    # Get relationship link
    link = await conn.fetchrow("""
        SELECT link_level, dynamics FROM SocialLinks
        WHERE user_id=$1 AND conversation_id=$2
        AND ((entity1_type='player' AND entity2_type='npc' AND entity2_id=$3)
          OR (entity1_type='npc' AND entity1_id=$3 AND entity2_type='player'))
    """, user_id, conversation_id, npc_id)
    
    if not link:
        return 1.0
    
    link_level = link['link_level'] or 0
    dynamics = link['dynamics'] if isinstance(link['dynamics'], dict) else json.loads(link['dynamics'] or '{}')
    
    # Base multiplier from empathy (0.5 to 2.0)
    empathy_mult = 0.5 + (empathy / 100) * 1.5
    
    # Link level bonus (0 to 1.0)
    link_bonus = min(link_level / 100, 1.0)
    
    # Intelligence helps understand complex relationships
    int_bonus = intelligence / 200  # 0 to 0.5
    
    # Special bonuses for specific dynamics
    trust_bonus = dynamics.get('trust', 0) / 200  # 0 to 0.5
    intimacy_bonus = dynamics.get('intimacy', 0) / 200  # 0 to 0.5
    
    total_multiplier = empathy_mult + link_bonus + int_bonus + trust_bonus + intimacy_bonus
    
    return min(total_multiplier, 3.0)  # Cap at 3x

async def check_relationship_insights(
    ctx,
    conn: asyncpg.Connection,
    npc_id: int,
    situation: str = "general"
) -> Dict[str, Any]:
    """
    Use empathy and relationship depth to gain insights about an NPC.
    Different situations require different stat thresholds.
    """
    # Get player stats
    player_stats = await conn.fetchrow("""
        SELECT empathy, intelligence, confidence, mental_resilience
        FROM PlayerStats
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, ctx.user_id, ctx.conversation_id)
    
    if not player_stats:
        return {"insights": [], "success": False}
    
    empathy = player_stats['empathy']
    intelligence = player_stats['intelligence']
    
    # Get NPC data
    npc_data = await conn.fetchrow("""
        SELECT npc_name, dominance, cruelty, trust, respect, 
               personality_traits, current_mood, hidden_agenda
        FROM NPCStats
        WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
    """, npc_id, ctx.user_id, ctx.conversation_id)
    
    if not npc_data:
        return {"insights": [], "success": False}
    
    # Get relationship multiplier
    depth_mult = await get_relationship_depth_multiplier(conn, ctx.user_id, ctx.conversation_id, npc_id)
    
    # Calculate insight power
    base_insight = empathy + (intelligence // 2)
    total_insight = int(base_insight * depth_mult)
    
    insights = []
    hidden_info = {}
    
    # Different thresholds for different insights
    insight_thresholds = {
        "surface_mood": 20,
        "true_feelings": 40,
        "hidden_motives": 60,
        "deepest_secrets": 80,
        "manipulation_detection": 50,
        "weakness_detection": 70,
        "prediction": 90
    }
    
    # Surface mood (easy)
    if total_insight >= insight_thresholds["surface_mood"]:
        mood = npc_data.get('current_mood', 'neutral')
        insights.append({
            "type": "mood",
            "text": f"{npc_data['npc_name']} seems {mood}",
            "confidence": min(100, total_insight)
        })
    
    # True feelings (moderate)
    if total_insight >= insight_thresholds["true_feelings"]:
        true_feeling = _determine_true_feeling(npc_data, situation)
        insights.append({
            "type": "true_feeling",
            "text": f"Beneath the surface, {npc_data['npc_name']} feels {true_feeling}",
            "confidence": min(100, total_insight - 20)
        })
        hidden_info["true_feeling"] = true_feeling
    
    # Hidden motives (hard)
    if total_insight >= insight_thresholds["hidden_motives"]:
        if npc_data.get('hidden_agenda'):
            insights.append({
                "type": "hidden_motive",
                "text": f"{npc_data['npc_name']} has ulterior motives related to {npc_data['hidden_agenda']}",
                "confidence": min(100, total_insight - 40)
            })
            hidden_info["hidden_agenda"] = npc_data['hidden_agenda']
    
    # Manipulation detection
    if total_insight >= insight_thresholds["manipulation_detection"]:
        manipulation_level = _calculate_manipulation_level(npc_data)
        if manipulation_level > 50:
            insights.append({
                "type": "manipulation",
                "text": f"{npc_data['npc_name']} is trying to manipulate you",
                "severity": manipulation_level,
                "confidence": min(100, total_insight - 30)
            })
    
    # Weakness detection (very hard)
    if total_insight >= insight_thresholds["weakness_detection"]:
        weakness = _find_npc_weakness(npc_data)
        if weakness:
            insights.append({
                "type": "weakness",
                "text": f"{npc_data['npc_name']}'s weakness: {weakness}",
                "confidence": min(100, total_insight - 50),
                "exploitable": True
            })
            hidden_info["weakness"] = weakness
    
    # Prediction (extremely hard)
    if total_insight >= insight_thresholds["prediction"]:
        prediction = _predict_npc_action(npc_data, situation)
        insights.append({
            "type": "prediction",
            "text": f"{npc_data['npc_name']} is likely to {prediction}",
            "confidence": min(100, total_insight - 70)
        })
    
    # Store insights in the relationship for future reference
    if insights:
        await _store_relationship_insights(ctx, conn, npc_id, insights, hidden_info)
    
    return {
        "insights": insights,
        "success": len(insights) > 0,
        "total_insight_power": total_insight,
        "depth_multiplier": depth_mult,
        "hidden_info": hidden_info
    }

async def perform_social_interaction(
    ctx,
    conn: asyncpg.Connection,
    npc_id: int,
    interaction_type: str,
    use_stat: str = "empathy"
) -> Dict[str, Any]:
    """
    Perform a stat-based social interaction with an NPC.
    Different interactions use different primary stats.
    """
    # Get player stats
    player_stats = await conn.fetchrow("""
        SELECT empathy, intelligence, confidence, strength, agility,
               corruption, obedience, willpower, mental_resilience
        FROM PlayerStats
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, ctx.user_id, ctx.conversation_id)
    
    # Get NPC stats
    npc_stats = await conn.fetchrow("""
        SELECT npc_name, dominance, cruelty, trust, respect, intensity,
               personality_traits, current_location
        FROM NPCStats
        WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
    """, npc_id, ctx.user_id, ctx.conversation_id)
    
    # Get vitals for energy/fatigue effects
    vitals = await conn.fetchrow("""
        SELECT energy, hunger, fatigue FROM PlayerVitals
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, ctx.user_id, ctx.conversation_id)
    
    # Define interaction types and their stat requirements
    interaction_configs = {
        "persuade": {
            "primary_stat": "empathy",
            "secondary_stat": "intelligence",
            "opposed_by": "dominance",
            "success_effects": {"trust": 5, "respect": 3},
            "failure_effects": {"trust": -2, "player_confidence": -2}
        },
        "intimidate": {
            "primary_stat": "strength",
            "secondary_stat": "confidence",
            "opposed_by": "dominance",
            "success_effects": {"dominance": -5, "fear": 10},
            "failure_effects": {"dominance": 5, "player_confidence": -5}
        },
        "seduce": {
            "primary_stat": "empathy",
            "secondary_stat": "confidence",
            "opposed_by": "willpower",
            "success_effects": {"intimacy": 10, "trust": 5},
            "failure_effects": {"respect": -5, "player_lust": 5}
        },
        "deceive": {
            "primary_stat": "intelligence",
            "secondary_stat": "agility",
            "opposed_by": "trust",
            "success_effects": {"manipulation": 10},
            "failure_effects": {"trust": -10, "respect": -5}
        },
        "empathize": {
            "primary_stat": "empathy",
            "secondary_stat": "mental_resilience",
            "opposed_by": "cruelty",
            "success_effects": {"trust": 8, "intimacy": 5},
            "failure_effects": {"player_mental_resilience": -3}
        },
        "challenge": {
            "primary_stat": "confidence",
            "secondary_stat": "willpower",
            "opposed_by": "dominance",
            "success_effects": {"respect": 10, "dominance": -3},
            "failure_effects": {"obedience": 5, "player_willpower": -3}
        }
    }
    
    config = interaction_configs.get(interaction_type, interaction_configs["persuade"])
    
    # Calculate success chance
    primary_value = player_stats[config["primary_stat"]]
    secondary_value = player_stats[config["secondary_stat"]]
    
    # Fatigue penalty
    fatigue_penalty = (vitals['fatigue'] if vitals else 0) // 4
    
    # Relationship bonus
    depth_mult = await get_relationship_depth_multiplier(conn, ctx.user_id, ctx.conversation_id, npc_id)
    relationship_bonus = int((depth_mult - 1) * 20)
    
    # Calculate roll
    player_power = primary_value + (secondary_value // 2) + relationship_bonus - fatigue_penalty
    npc_resistance = npc_stats[config["opposed_by"]]
    
    # Special modifiers based on circumstances
    if interaction_type == "intimidate" and player_stats["strength"] < npc_stats["dominance"]:
        player_power -= 20  # Hard to intimidate someone stronger
    
    if interaction_type == "seduce" and player_stats["corruption"] > 70:
        player_power += 15  # Corrupted characters are more seductive
    
    success_chance = 50 + (player_power - npc_resistance)
    roll = random.randint(1, 100)
    success = roll <= success_chance
    
    # Apply effects
    changes = {"player_stats": {}, "npc_stats": {}, "dynamics": {}}
    
    if success:
        # Apply success effects
        for key, value in config["success_effects"].items():
            if key.startswith("player_"):
                stat_name = key.replace("player_", "")
                changes["player_stats"][stat_name] = value
            elif key in ["trust", "respect", "dominance", "cruelty", "intensity"]:
                changes["npc_stats"][key] = value
            else:
                # Dynamic changes (intimacy, fear, manipulation, etc)
                changes["dynamics"][key] = value
        
        # Bonus effects for critical success
        if roll <= success_chance - 50:
            changes["player_stats"]["confidence"] = 3
            changes["dynamics"]["breakthrough"] = True
    else:
        # Apply failure effects
        for key, value in config["failure_effects"].items():
            if key.startswith("player_"):
                stat_name = key.replace("player_", "")
                changes["player_stats"][stat_name] = value
            elif key in ["trust", "respect", "dominance", "cruelty", "intensity"]:
                changes["npc_stats"][key] = value
            else:
                changes["dynamics"][key] = value
        
        # Critical failure
        if roll >= success_chance + 50:
            changes["player_stats"]["confidence"] = -5
            changes["dynamics"]["humiliation"] = True
    
    # Apply the changes
    await _apply_interaction_changes(ctx, conn, npc_id, changes)
    
    # Create interaction memory
    memory_text = f"{interaction_type.capitalize()} attempt with {npc_stats['npc_name']}: "
    memory_text += "Success" if success else "Failure"
    
    await add_link_event(ctx, await _get_link_id(conn, ctx, npc_id), memory_text)
    
    return {
        "success": success,
        "roll": roll,
        "chance": success_chance,
        "changes": changes,
        "critical": abs(roll - success_chance) > 50,
        "player_power": player_power,
        "npc_resistance": npc_resistance
    }

async def check_for_stat_based_crossroads(
    ctx,
    conn: asyncpg.Connection
) -> Optional[Dict[str, Any]]:
    """
    Enhanced crossroads check that considers player stats for additional options.
    """
    # Get standard crossroads first
    standard_crossroads = await check_for_relationship_crossroads(ctx, conn)
    
    if not standard_crossroads:
        return None
    
    # Get player stats
    player_stats = await conn.fetchrow("""
        SELECT empathy, intelligence, confidence, willpower, corruption, obedience
        FROM PlayerStats
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, ctx.user_id, ctx.conversation_id)
    
    if not player_stats:
        return standard_crossroads
    
    # Add stat-gated choices
    additional_choices = []
    
    # High empathy option
    if player_stats['empathy'] > 60:
        additional_choices.append({
            "text": "[Empathy] Understand their deeper needs",
            "requirements": {"empathy": 60},
            "effects": {
                "trust": 15,
                "intimacy": 10,
                "player_empathy": 2,
                "player_corruption": 3
            },
            "outcome": "Your deep understanding of {npc_name} creates an intimate moment of connection."
        })
    
    # High intelligence option
    if player_stats['intelligence'] > 60:
        additional_choices.append({
            "text": "[Intelligence] Analyze their behavioral patterns",
            "requirements": {"intelligence": 60},
            "effects": {
                "manipulation": 10,
                "control": 5,
                "player_intelligence": 1,
                "player_mental_resilience": -2
            },
            "outcome": "You see through {npc_name}'s facade, understanding how to influence them."
        })
    
    # High confidence + low corruption option
    if player_stats['confidence'] > 70 and player_stats['corruption'] < 30:
        additional_choices.append({
            "text": "[Confident] Assert your independence",
            "requirements": {"confidence": 70, "corruption_max": 30},
            "effects": {
                "respect": 20,
                "dominance": -15,
                "player_confidence": 5,
                "player_willpower": 5
            },
            "outcome": "Your unwavering confidence forces {npc_name} to respect your boundaries."
        })
    
    # High corruption + obedience option
    if player_stats['corruption'] > 70 and player_stats['obedience'] > 60:
        additional_choices.append({
            "text": "[Corrupted] Embrace your role completely",
            "requirements": {"corruption": 70, "obedience": 60},
            "effects": {
                "submission": 20,
                "dependency": 15,
                "player_corruption": 10,
                "player_obedience": 10,
                "player_willpower": -10
            },
            "outcome": "You surrender completely to {npc_name}'s will, finding dark pleasure in submission."
        })
    
    # Add the additional choices
    if additional_choices:
        standard_crossroads["choices"].extend(additional_choices)
        standard_crossroads["has_stat_options"] = True
    
    return standard_crossroads

async def process_relationship_activity(
    ctx,
    conn: asyncpg.Connection,
    npc_id: int,
    activity_type: str,
    duration: int = 1  # Time periods
) -> Dict[str, Any]:
    """
    Process activities done with NPCs that affect both stats and relationships.
    """
    activity_configs = {
        "intimate_conversation": {
            "stat_requirements": {"empathy": 30},
            "stat_effects": {"empathy": 1, "mental_resilience": 1},
            "vital_effects": {"thirst": -10},
            "relationship_effects": {"trust": 5, "intimacy": 5},
            "dynamics": {"emotional_connection": 5}
        },
        "training_together": {
            "stat_requirements": {"endurance": 20},
            "stat_effects": {"strength": 1, "endurance": 1},
            "vital_effects": {"fatigue": 15, "thirst": -20},
            "relationship_effects": {"respect": 5},
            "dynamics": {"shared_growth": 5}
        },
        "intellectual_debate": {
            "stat_requirements": {"intelligence": 40},
            "stat_effects": {"intelligence": 1},
            "vital_effects": {"fatigue": 5},
            "relationship_effects": {"respect": 8},
            "dynamics": {"intellectual_rivalry": 5}
        },
        "submission_training": {
            "stat_requirements": {"obedience": 50},
            "stat_effects": {"obedience": 3, "corruption": 2, "willpower": -2},
            "vital_effects": {"fatigue": 10},
            "relationship_effects": {"dominance": 5},
            "dynamics": {"control": 10, "submission": 10}
        },
        "service_tasks": {
            "stat_requirements": {"obedience": 30},
            "stat_effects": {"obedience": 2, "dependency": 1},
            "vital_effects": {"fatigue": 20, "hunger": -10},
            "relationship_effects": {"trust": 3},
            "dynamics": {"servitude": 5}
        },
        "resistance_training": {
            "stat_requirements": {"willpower": 40, "confidence": 40},
            "stat_effects": {"willpower": 2, "confidence": 2, "obedience": -3},
            "vital_effects": {"fatigue": 15},
            "relationship_effects": {"dominance": -5, "respect": 10},
            "dynamics": {"rebellion": 10}
        }
    }
    
    config = activity_configs.get(activity_type)
    if not config:
        return {"error": "Unknown activity type"}
    
    # Check stat requirements
    player_stats = await conn.fetchrow("""
        SELECT * FROM PlayerStats
        WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
    """, ctx.user_id, ctx.conversation_id)
    
    for stat, required in config["stat_requirements"].items():
        if player_stats[stat] < required:
            return {
                "error": f"Insufficient {stat}",
                "required": required,
                "current": player_stats[stat]
            }
    
    # Apply effects
    changes = {
        "stat_changes": {},
        "vital_changes": {},
        "relationship_changes": {},
        "dynamic_changes": {}
    }
    
    # Scale effects by duration
    for stat, change in config["stat_effects"].items():
        changes["stat_changes"][stat] = change * duration
    
    for vital, change in config["vital_effects"].items():
        changes["vital_changes"][vital] = change * duration
    
    # Relationship effects scale differently
    for rel, change in config["relationship_effects"].items():
        changes["relationship_changes"][rel] = change * (1 + duration * 0.5)
    
    for dyn, change in config["dynamics"].items():
        changes["dynamic_changes"][dyn] = change * duration
    
    # Apply all changes
    await _apply_activity_changes(ctx, conn, npc_id, changes)
    
    # Create activity memory
    await add_link_event(
        ctx, 
        await _get_link_id(conn, ctx, npc_id),
        f"Engaged in {activity_type.replace('_', ' ')} for {duration} periods"
    )
    
    return {
        "success": True,
        "activity": activity_type,
        "duration": duration,
        "changes": changes
    }

# ========== HELPER FUNCTIONS ==========

def _determine_true_feeling(npc_data: Dict, situation: str) -> str:
    """Determine NPC's true feelings based on their stats and situation."""
    dominance = npc_data.get('dominance', 50)
    trust = npc_data.get('trust', 0)
    respect = npc_data.get('respect', 0)
    
    if trust > 70:
        return "genuine affection"
    elif respect > 70 and dominance > 70:
        return "possessive desire"
    elif trust < -50:
        return "deep suspicion"
    elif dominance > 80 and respect < 20:
        return "contemptuous amusement"
    else:
        return "calculated interest"

def _calculate_manipulation_level(npc_data: Dict) -> int:
    """Calculate how manipulative an NPC is being."""
    dominance = npc_data.get('dominance', 50)
    cruelty = npc_data.get('cruelty', 50)
    trust = npc_data.get('trust', 0)
    
    # High dominance + cruelty + low trust = manipulation
    manipulation = dominance + cruelty - trust
    return max(0, min(100, manipulation // 2))

def _find_npc_weakness(npc_data: Dict) -> Optional[str]:
    """Find potential weaknesses in an NPC."""
    traits = npc_data.get('personality_traits', [])
    if isinstance(traits, str):
        traits = json.loads(traits)
    
    # Check for exploitable traits
    weakness_map = {
        "arrogant": "flattery and ego-stroking",
        "lonely": "emotional vulnerability",
        "ambitious": "promises of power",
        "hedonistic": "sensual temptations",
        "paranoid": "playing on their fears",
        "perfectionist": "highlighting their failures"
    }
    
    for trait in traits:
        for weakness_trait, exploit in weakness_map.items():
            if weakness_trait in trait.lower():
                return exploit
    
    # Check stats for weaknesses
    if npc_data.get('trust', 0) > 80:
        return "excessive trust can be exploited"
    elif npc_data.get('cruelty', 50) < 20:
        return "compassion can be manipulated"
    
    return None

def _predict_npc_action(npc_data: Dict, situation: str) -> str:
    """Predict likely NPC behavior."""
    dominance = npc_data.get('dominance', 50)
    cruelty = npc_data.get('cruelty', 50)
    
    if situation == "confrontation":
        if dominance > 70:
            return "escalate and assert control"
        else:
            return "deflect or negotiate"
    elif situation == "vulnerability":
        if cruelty > 70:
            return "exploit your weakness"
        else:
            return "show unexpected kindness"
    else:
        return "maintain current dynamic"

async def _store_relationship_insights(
    ctx, conn: asyncpg.Connection, npc_id: int, 
    insights: List[Dict], hidden_info: Dict
):
    """Store discovered insights in the relationship data."""
    link_id = await _get_link_id(conn, ctx, npc_id)
    if not link_id:
        return
    
    # Get current context
    current_context = await conn.fetchval(
        "SELECT context FROM SocialLinks WHERE link_id=$1",
        link_id
    )
    
    context = current_context if isinstance(current_context, dict) else {}
    
    # Add insights
    if "insights_discovered" not in context:
        context["insights_discovered"] = []
    
    for insight in insights:
        context["insights_discovered"].append({
            "type": insight["type"],
            "text": insight["text"],
            "discovered_at": datetime.now().isoformat()
        })
    
    # Store hidden info
    context["hidden_info"] = hidden_info
    
    # Update using LoreSystem
    lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
    await lore_system.propose_and_enact_change(
        ctx=ctx,
        entity_type="SocialLinks",
        entity_identifier={"link_id": link_id},
        updates={"context": json.dumps(context)},
        reason="Stored relationship insights"
    )

async def _get_link_id(conn: asyncpg.Connection, ctx, npc_id: int) -> Optional[int]:
    """Get the link_id for a player-NPC relationship."""
    result = await conn.fetchval("""
        SELECT link_id FROM SocialLinks
        WHERE user_id=$1 AND conversation_id=$2
        AND ((entity1_type='player' AND entity2_type='npc' AND entity2_id=$3)
          OR (entity1_type='npc' AND entity1_id=$3 AND entity2_type='player'))
    """, ctx.user_id, ctx.conversation_id, npc_id)
    return result

async def _apply_interaction_changes(
    ctx, conn: asyncpg.Connection, npc_id: int, changes: Dict
):
    """Apply all changes from a social interaction."""
    # Apply player stat changes
    if changes["player_stats"]:
        from logic.stats_logic import apply_stat_changes
        await apply_stat_changes(
            ctx.user_id, ctx.conversation_id, "Chase",
            changes["player_stats"], "Social interaction"
        )
    
    # Apply NPC stat changes
    if changes["npc_stats"]:
        updates = []
        values = []
        param_idx = 1
        
        for stat, change in changes["npc_stats"].items():
            updates.append(f"{stat} = LEAST(100, GREATEST(-100, {stat} + ${param_idx}))")
            values.append(change)
            param_idx += 1
        
        values.extend([ctx.user_id, ctx.conversation_id, npc_id])
        
        await conn.execute(f"""
            UPDATE NPCStats
            SET {", ".join(updates)}
            WHERE user_id=${param_idx} AND conversation_id=${param_idx+1} AND npc_id=${param_idx+2}
        """, *values)
    
    # Apply dynamic changes
    if changes["dynamics"]:
        link_id = await _get_link_id(conn, ctx, npc_id)
        if link_id:
            # Get current dynamics
            current = await conn.fetchrow(
                "SELECT dynamics FROM SocialLinks WHERE link_id=$1",
                link_id
            )
            
            dynamics = current['dynamics'] if isinstance(current['dynamics'], dict) else json.loads(current['dynamics'] or '{}')
            
            # Apply changes
            for key, change in changes["dynamics"].items():
                if key == "breakthrough" or key == "humiliation":
                    # Special flags
                    dynamics[key] = True
                else:
                    current_val = dynamics.get(key, 0)
                    dynamics[key] = max(0, min(100, current_val + change))
            
            # Update
            lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
            await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="SocialLinks",
                entity_identifier={"link_id": link_id},
                updates={"dynamics": json.dumps(dynamics)},
                reason="Social interaction effects"
            )

async def _apply_activity_changes(
    ctx, conn: asyncpg.Connection, npc_id: int, changes: Dict
):
    """Apply changes from relationship activities."""
    # Apply stat changes
    if changes["stat_changes"]:
        from logic.stats_logic import apply_stat_changes
        await apply_stat_changes(
            ctx.user_id, ctx.conversation_id, "Chase",
            changes["stat_changes"], "Relationship activity"
        )
    
    # Apply vital changes
    if changes["vital_changes"]:
        for vital, change in changes["vital_changes"].items():
            current = await conn.fetchval(f"""
                SELECT {vital} FROM PlayerVitals
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, ctx.user_id, ctx.conversation_id)
            
            if current is not None:
                new_value = max(0, min(100, current + change))
                await conn.execute(f"""
                    UPDATE PlayerVitals
                    SET {vital} = $1
                    WHERE user_id=$2 AND conversation_id=$3 AND player_name='Chase'
                """, new_value, ctx.user_id, ctx.conversation_id)
    
    # Apply relationship changes (to NPCStats)
    if changes["relationship_changes"]:
        await _apply_interaction_changes(
            ctx, conn, npc_id, 
            {"player_stats": {}, "npc_stats": changes["relationship_changes"], "dynamics": {}}
        )
    
    # Apply dynamic changes
    if changes["dynamic_changes"]:
        await _apply_interaction_changes(
            ctx, conn, npc_id,
            {"player_stats": {}, "npc_stats": {}, "dynamics": changes["dynamic_changes"]}
        )



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4) Crossroad Checking + Ritual Checking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def check_for_relationship_crossroads(ctx, conn) -> Optional[Dict[str, Any]]:
    """
    Check if any NPC relationship triggers a Crossroads event.
    """
    try:
        # Gather all player-related links
        links = await conn.fetch(
            """
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                   dynamics, experienced_crossroads
            FROM SocialLinks
            WHERE user_id = $1 AND conversation_id = $2
            AND (
                (entity1_type='player' AND entity1_id = $1)
                OR (entity2_type='player' AND entity2_id = $1)
            )
            """,
            ctx.user_id, ctx.conversation_id
        )

        for link_record in links:
            link_id = link_record['link_id']
            e1t = link_record['entity1_type']
            e1id = link_record['entity1_id']
            e2t = link_record['entity2_type']
            e2id = link_record['entity2_id']
            dynamics_data = link_record['dynamics']
            crossroads_data = link_record['experienced_crossroads']

            dynamics = dynamics_data if isinstance(dynamics_data, dict) else (json.loads(dynamics_data) if isinstance(dynamics_data, str) else {})
            experienced = crossroads_data if isinstance(crossroads_data, list) else (json.loads(crossroads_data) if isinstance(crossroads_data, str) else [])

            # Determine NPC side
            npc_id = None
            if e1t == "npc" and e2t == "player":
                npc_id = e1id
            elif e2t == "npc" and e1t == "player":
                npc_id = e2id

            if npc_id is None:
                continue

            # Get NPC name
            npc_name = await conn.fetchval(
                "SELECT npc_name FROM NPCStats WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3",
                ctx.user_id, ctx.conversation_id, npc_id
            )
            if not npc_name:
                logger.warning(f"NPC {npc_id} not found for crossroads check.")
                continue

            # Check each Crossroads definition
            for crossroads_def in RELATIONSHIP_CROSSROADS:
                if crossroads_def["name"] in experienced:
                    continue
                dynamic_needed = crossroads_def["dynamic"]
                trigger_level = crossroads_def["trigger_level"]
                current_level = dynamics.get(dynamic_needed, 0)

                if current_level >= trigger_level:
                    # Trigger this crossroads
                    formatted_choices = []
                    for ch in crossroads_def["choices"]:
                        fc = {
                            "text": ch["text"],
                            "effects": ch["effects"],
                            "outcome": ch["outcome"].format(npc_name=npc_name),
                        }
                        formatted_choices.append(fc)
                    return {
                        "type": "relationship_crossroads",
                        "name": crossroads_def["name"],
                        "description": crossroads_def["description"],
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "choices": formatted_choices,
                        "link_id": link_id,
                    }
        return None

    except Exception as e:
        logger.error(f"Error checking for relationship crossroads: {e}", exc_info=True)
        return None


async def get_relationship_dynamic_level(user_id: int, entity_id: int, dynamic_name: str = "trust") -> int:
    """
    Get the level of a specific relationship dynamic between player and an entity.
    """
    integration = RelationshipIntegration(user_id, user_id)
    return await integration.get_dynamic_level("player", user_id, "npc", entity_id, dynamic_name)

async def update_relationship_dynamic(user_id: int, entity_id: int, dynamic_name: str, change: int) -> int:
    """
    Update a specific relationship dynamic between player and an entity.
    """
    integration = RelationshipIntegration(user_id, user_id)
    return await integration.update_dynamic("player", user_id, "npc", entity_id, dynamic_name, change)

async def apply_crossroads_choice(
    ctx,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Apply the chosen effect from a Crossroads event using LoreSystem.
    """
    cr_def = next((c for c in RELATIONSHIP_CROSSROADS if c["name"] == crossroads_name), None)
    if not cr_def:
        return {"error": f"Crossroads '{crossroads_name}' definition not found"}
    if not 0 <= choice_index < len(cr_def["choices"]):
        return {"error": "Invalid choice index"}
    choice = cr_def["choices"][choice_index]

    try:
        # Get LoreSystem instance
        lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
        
        # Get current link data
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_crossroads
                FROM SocialLinks
                WHERE link_id = $1 AND user_id = $2 AND conversation_id = $3
                """,
                link_id, ctx.user_id, ctx.conversation_id
            )
            if not row:
                return {"error": "Social link not found"}

            e1t, e1id, e2t, e2id, dyn_data, crossroads_data = row['entity1_type'], row['entity1_id'], row['entity2_type'], row['entity2_id'], row['dynamics'], row['experienced_crossroads']

            # Identify NPC
            npc_id = e1id if e1t == "npc" else (e2id if e2t == "npc" else None)
            if npc_id is None:
                 return {"error": "No NPC found in relationship"}

            # Get NPC name
            npc_name = await conn.fetchval(
                "SELECT npc_name FROM NPCStats WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3",
                ctx.user_id, ctx.conversation_id, npc_id
            )
            if not npc_name:
                 return {"error": f"NPC {npc_id} not found"}

            # Parse dynamics and experienced lists
            dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
            experienced = crossroads_data if isinstance(crossroads_data, list) else (json.loads(crossroads_data) if isinstance(crossroads_data, str) else [])

            # Apply Effects
            player_stat_updates = {}
            for dynamic_name, delta in choice["effects"].items():
                if dynamic_name.startswith("player_"):
                    player_stat = dynamic_name[7:]
                    player_stat_updates[player_stat] = delta
                else:
                    current_val = dynamics.get(dynamic_name, 0)
                    new_val = max(0, min(100, current_val + delta))
                    dynamics[dynamic_name] = new_val

            # Mark crossroads as experienced
            if crossroads_name not in experienced:
                experienced.append(crossroads_name)

            # Recompute primary link type/level based on new dynamics
            primary_type = get_primary_dynamic(dynamics)
            primary_level = dynamics.get(primary_type, 0)

            # Update the SocialLinks row using LoreSystem
            link_updates = {
                "dynamics": json.dumps(dynamics),
                "experienced_crossroads": json.dumps(experienced),
                "link_type": primary_type,
                "link_level": primary_level
            }
            
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="SocialLinks",
                entity_identifier={"link_id": link_id},
                updates=link_updates,
                reason=f"Applied crossroads choice '{crossroads_name}': {choice['text']}"
            )

            if result["status"] != "committed":
                return {"error": f"Failed to update link: {result}"}

            # Add event to link history
            event_text = (
                f"Crossroads '{crossroads_name}' chosen: {choice['text']}. "
                f"Outcome: {choice['outcome'].format(npc_name=npc_name)}"
            )
            await add_link_event(ctx, link_id, event_text)

            # Update player stats if any
            if player_stat_updates:
                # Get current player stats
                player_stats = await conn.fetchrow(
                    """
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                    """,
                    ctx.user_id, ctx.conversation_id
                )
                
                if player_stats:
                    # Calculate new values
                    stat_updates = {}
                    valid_player_stats = ["corruption", "confidence", "willpower", "obedience", "dependency", "lust", "mental_resilience", "physical_endurance"]
                    
                    for stat, delta in player_stat_updates.items():
                        if stat in valid_player_stats and stat in player_stats:
                            current_value = player_stats[stat]
                            new_value = max(0, min(100, current_value + delta))
                            stat_updates[stat] = new_value
                    
                    if stat_updates:
                        # Use LoreSystem to update player stats
                        player_result = await lore_system.propose_and_enact_change(
                            ctx=ctx,
                            entity_type="PlayerStats",
                            entity_identifier={"user_id": ctx.user_id, "conversation_id": ctx.conversation_id, "player_name": "Chase"},
                            updates=stat_updates,
                            reason=f"Crossroads choice effect: {choice['text']}"
                        )

            # Add Journal Entry
            journal_entry = (
                f"Crossroads: {crossroads_name} with {npc_name}. "
                f"Choice: {choice['text']} => {choice['outcome'].format(npc_name=npc_name)}"
            )
            
            # Create journal entry through canon
            await canon.create_journal_entry(
                ctx, conn,
                entry_type='relationship_crossroads',
                entry_text=journal_entry
            )

            return {"success": True, "outcome_text": choice["outcome"].format(npc_name=npc_name)}

    except Exception as e:
        logger.error(f"Error applying crossroads choice for link {link_id}: {e}", exc_info=True)
        return {"error": "An unexpected error occurred."}


async def check_for_relationship_ritual(ctx, conn) -> Optional[Dict[str, Any]]:
    """
    Check if any relationship triggers a Ritual event.
    """
    try:
        # Gather all player-related links
        links = await conn.fetch(
             """
             SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                    dynamics, experienced_rituals
             FROM SocialLinks
             WHERE user_id = $1 AND conversation_id = $2
             AND (
                 (entity1_type='player' AND entity1_id = $1)
                 OR (entity2_type='player' AND entity2_id = $1)
             )
             """,
             ctx.user_id, ctx.conversation_id
         )

        for link_record in links:
            link_id = link_record['link_id']
            e1t, e1id, e2t, e2id = link_record['entity1_type'], link_record['entity1_id'], link_record['entity2_type'], link_record['entity2_id']
            dyn_data = link_record['dynamics']
            rit_data = link_record['experienced_rituals']

            dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
            experienced = rit_data if isinstance(rit_data, list) else (json.loads(rit_data) if isinstance(rit_data, str) else [])

            # Identify NPC
            npc_id = e1id if e1t == "npc" else (e2id if e2t == "npc" else None)
            if npc_id is None: continue

            # Check NPC dominance
            npc_info = await conn.fetchrow(
                 "SELECT npc_name, dominance FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                 ctx.user_id, ctx.conversation_id, npc_id
            )
            if not npc_info or npc_info['dominance'] < 50: continue
            npc_name = npc_info['npc_name']

            # Check possible rituals
            possible_rituals = []
            for rit_def in RELATIONSHIP_RITUALS:
                 if rit_def["name"] in experienced: continue
                 triggered = False
                 for dyn_name in rit_def["dynamics"]:
                      if dynamics.get(dyn_name, 0) >= rit_def["trigger_level"]:
                           triggered = True
                           break
                 if triggered:
                      possible_rituals.append(rit_def)

            if possible_rituals:
                # Choose one and apply it
                chosen_ritual = random.choice(possible_rituals)

                ritual_txt = chosen_ritual["ritual_text"]
                if "{gift_item}" in ritual_txt:
                     gift_item = random.choice(SYMBOLIC_GIFTS)
                     ritual_txt = ritual_txt.format(npc_name=npc_name, gift_item=gift_item)
                else:
                     ritual_txt = ritual_txt.format(npc_name=npc_name)

                # Mark as experienced
                experienced.append(chosen_ritual["name"])
                
                # Update using LoreSystem
                lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
                result = await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="SocialLinks",
                    entity_identifier={"link_id": link_id},
                    updates={"experienced_rituals": json.dumps(experienced)},
                    reason=f"Ritual '{chosen_ritual['name']}' triggered"
                )

                # Add history event
                event_text = f"Ritual '{chosen_ritual['name']}': {ritual_txt}"
                await add_link_event(ctx, link_id, event_text)

                # Journal entry
                journal_text = f"Ritual with {npc_name}: {chosen_ritual['name']}. {ritual_txt}"
                await canon.create_journal_entry(
                    ctx, conn,
                    entry_type='relationship_ritual',
                    entry_text=journal_text
                )

                # Increase relevant dynamics by +10
                dynamics_update = {}
                for dyn_name in chosen_ritual["dynamics"]:
                     old_val = dynamics.get(dyn_name, 0)
                     new_val = min(100, old_val + 10)
                     if new_val != old_val:
                         dynamics_update[dyn_name] = new_val

                if dynamics_update:
                    # Update dynamics
                    for dyn_name, new_val in dynamics_update.items():
                        dynamics[dyn_name] = new_val
                    
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="SocialLinks",
                        entity_identifier={"link_id": link_id},
                        updates={"dynamics": json.dumps(dynamics)},
                        reason=f"Ritual '{chosen_ritual['name']}' increased dynamics"
                    )

                # Update PlayerStats
                player_updates = {
                    "corruption": min(100, npc_info.get('corruption', 0) + 5),
                    "dependency": min(100, npc_info.get('dependency', 0) + 5)
                }
                
                await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="PlayerStats",
                    entity_identifier={"user_id": ctx.user_id, "conversation_id": ctx.conversation_id, "player_name": "Chase"},
                    updates=player_updates,
                    reason=f"Ritual '{chosen_ritual['name']}' effect"
                )

                return {
                     "type": "relationship_ritual",
                     "name": chosen_ritual["name"],
                     "description": chosen_ritual["description"],
                     "npc_id": npc_id,
                     "npc_name": npc_name,
                     "ritual_text": ritual_txt,
                     "link_id": link_id,
                }
        return None

    except Exception as e:
        logger.error(f"Error checking for relationship ritual: {e}", exc_info=True)
        return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5) Summaries & Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_entity_name(
    conn: asyncpg.Connection,
    entity_type: str,
    entity_id: int,
    user_id: int,
    conversation_id: int
) -> str:
    """
    Get the name of an entity (NPC or player) using the provided connection.
    """
    if entity_type == "player" and (entity_id == 0 or entity_id == user_id):
        player_name = await conn.fetchval(
             "SELECT player_name FROM PlayerStats WHERE user_id = $1 AND conversation_id = $2 LIMIT 1",
             user_id, conversation_id
        )
        return player_name or "Player"
    elif entity_type == "npc":
        npc_name = await conn.fetchval(
             """
             SELECT npc_name FROM NPCStats
             WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
             """,
             user_id, conversation_id, entity_id
        )
        return npc_name or f"Unknown NPC {entity_id}"
    else:
        return f"Unknown Entity {entity_id}"

async def detect_deception(user_id: int, conversation_id: int, player_name: str, 
                          npc_id: int, deception_type: str) -> Dict[str, Any]:
    """
    Use empathy to detect NPC deception or hidden emotions.
    """
    async with get_db_connection_context() as conn:
        # Get player empathy
        empathy = await conn.fetchval("""
            SELECT empathy FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
        """, user_id, conversation_id, player_name)
        
        # Get NPC stats for difficulty
        npc_stats = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
        
        if not empathy or not npc_stats:
            return {"success": False, "error": "Missing data"}
        
        # Higher dominance = better at hiding intentions
        deception_skill = npc_stats['dominance'] // 10
        
        # Make insight check
        success = calculate_social_insight(empathy, deception_skill)
        
        insight = None
        if success:
            insights_map = {
                "lying": f"{npc_stats['npc_name']} is not being truthful",
                "hidden_anger": f"{npc_stats['npc_name']} is suppressing anger",
                "false_kindness": f"{npc_stats['npc_name']}'s friendliness seems forced",
                "nervousness": f"{npc_stats['npc_name']} is more nervous than they appear",
                "hidden_motive": f"{npc_stats['npc_name']} has ulterior motives"
            }
            insight = insights_map.get(deception_type, "Something seems off")
            
            # Trigger hidden stat changes for successful insight
            await conn.execute("""
                UPDATE PlayerStats
                SET confidence = LEAST(100, confidence + 1),
                    mental_resilience = LEAST(100, mental_resilience + 1)
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, user_id, conversation_id, player_name)
        
        return {
            "success": success,
            "insight": insight,
            "empathy_used": empathy,
            "difficulty": deception_skill * 10
        }



async def get_relationship_summary(
    ctx,
    conn,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get a summary of the relationship.
    """
    try:
        row = await conn.fetchrow(
             """
             SELECT link_id, link_type, link_level, dynamics, link_history,
                    experienced_crossroads, experienced_rituals
             FROM SocialLinks
             WHERE user_id = $1 AND conversation_id = $2
               AND entity1_type = $3 AND entity1_id = $4
               AND entity2_type = $5 AND entity2_id = $6
             """,
             ctx.user_id, ctx.conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
        )
        if not row:
            return None

        link_id, link_type, link_level, dyn_data, hist_data, cr_data, rit_data = \
            row['link_id'], row['link_type'], row['link_level'], row['dynamics'], \
            row['link_history'], row['experienced_crossroads'], row['experienced_rituals']

        dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
        history = hist_data if isinstance(hist_data, list) else (json.loads(hist_data) if isinstance(hist_data, str) else [])
        cr_list = cr_data if isinstance(cr_data, list) else (json.loads(cr_data) if isinstance(cr_data, str) else [])
        rit_list = rit_data if isinstance(rit_data, list) else (json.loads(rit_data) if isinstance(rit_data, str) else [])

        e1_name = await get_entity_name(conn, entity1_type, entity1_id, ctx.user_id, ctx.conversation_id)
        e2_name = await get_entity_name(conn, entity2_type, entity2_id, ctx.user_id, ctx.conversation_id)

        dynamic_descriptions = [
             f"{dnm.capitalize()} {lvl}/100 => {get_dynamic_description(dnm, lvl)}"
             for dnm, lvl in dynamics.items()
        ]

        return {
            "entity1_name": e1_name,
            "entity2_name": e2_name,
            "primary_type": link_type,
            "primary_level": link_level,
            "dynamics": dynamics,
            "dynamic_descriptions": dynamic_descriptions,
            "history": history[-5:],
            "experienced_crossroads": cr_list,
            "experienced_rituals": rit_list,
        }
    except Exception as e:
        logger.error(f"Error getting relationship summary: {e}", exc_info=True)
        return None



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6) Example Enhanced Classes (Optional)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RelationshipDimension:
    """
    A specific dimension/aspect of a relationship between entities
    """
    def __init__(self, name: str, description: str, min_value: int = -100, max_value: int = 100):
        self.name = name
        self.description = description
        self.min_value = min_value
        self.max_value = max_value

    def get_level_description(self, value: int) -> str:
        rng = self.max_value - self.min_value
        pct = (value - self.min_value) / float(rng)
        if pct < 0.2:
            return f"Very Low {self.name}"
        elif pct < 0.4:
            return f"Low {self.name}"
        elif pct < 0.6:
            return f"Moderate {self.name}"
        elif pct < 0.8:
            return f"High {self.name}"
        return f"Very High {self.name}"


class EnhancedRelationshipManager:
    """
    Manages more complex relationships: multiple dimensions,
    potential transitions, tension, etc.
    """
    RELATIONSHIP_DIMENSIONS = {
        "control": RelationshipDimension("Control", "How much one exerts control", 0, 100),
        "trust": RelationshipDimension("Trust", "Level of trust between entities", -100, 100),
        "intimacy": RelationshipDimension("Intimacy", "Emotional/physical closeness", 0, 100),
        "respect": RelationshipDimension("Respect", "Respect between entities", -100, 100),
        "dependency": RelationshipDimension("Dependency", "How dependent one is", 0, 100),
        "fear": RelationshipDimension("Fear", "How much one fears the other", 0, 100),
        "tension": RelationshipDimension("Tension", "Current tension level", 0, 100),
        "obsession": RelationshipDimension("Obsession", "How obsessed one is", 0, 100),
        "resentment": RelationshipDimension("Resentment", "Resentment level", 0, 100),
        "manipulation": RelationshipDimension("Manipulation", "Degree of manipulation", 0, 100),
    }

    RELATIONSHIP_TYPES = {
        "dominant": {"primary_dimensions": ["control", "fear", "tension", "manipulation"]},
        "submission": {"primary_dimensions": ["control", "dependency", "fear", "respect"]},
        "rivalry": {"primary_dimensions": ["tension", "resentment", "respect", "manipulation"]},
        "alliance": {"primary_dimensions": ["trust", "respect", "manipulation"]},
        "intimate": {"primary_dimensions": ["intimacy", "dependency", "obsession", "manipulation"]},
        "familial": {"primary_dimensions": ["control", "dependency", "respect", "resentment"]},
        "mentor": {"primary_dimensions": ["control", "respect", "dependency"]},
        "enmity": {"primary_dimensions": ["fear", "resentment", "tension"]},
        "neutral": {"primary_dimensions": []},
    }

    RELATIONSHIP_TRANSITIONS = [
        {
            "name": "Submission Acceptance",
            "from_type": "dominant",
            "to_type": "submission",
            "required_dimensions": {"dependency": 70, "fear": 50, "respect": 40},
        },
        {
            "name": "Rivalry to Alliance",
            "from_type": "rivalry",
            "to_type": "alliance",
            "required_dimensions": {"trust": 50, "respect": 60, "tension": -40},
        },
        {
            "name": "Alliance to Betrayal",
            "from_type": "alliance",
            "to_type": "enmity",
            "required_dimensions": {"trust": -60, "resentment": 70, "manipulation": 80},
        },
        {
            "name": "Mentor to Intimate",
            "from_type": "mentor",
            "to_type": "intimate",
            "required_dimensions": {"intimacy": 70, "obsession": 60, "dependency": 50},
        },
        {
            "name": "Enmity to Submission",
            "from_type": "enmity",
            "to_type": "submission",
            "required_dimensions": {"fear": 80, "control": 70, "dependency": 60},
        },
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 7) NPCGroup & MultiNPCInteractionManager
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NPCGroup:
    """
    Represents a group of NPCs with shared dynamics.
    """
    def __init__(self, name: str, description: str, members=None, dynamics=None, group_id=None):
        self.group_id = group_id
        self.name = name
        self.description = description
        self.members = members or []
        self.dynamics = dynamics or {}
        self.creation_date = datetime.now().isoformat()
        self.last_activity = None
        self.shared_history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "members": self.members,
            "dynamics": self.dynamics,
            "creation_date": self.creation_date,
            "last_activity": self.last_activity,
            "shared_history": self.shared_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], group_id=None):
        grp = cls(data["name"], data["description"], data.get("members", []), data.get("dynamics", {}), group_id)
        grp.creation_date = data.get("creation_date", datetime.now().isoformat())
        grp.last_activity = data.get("last_activity", None)
        grp.shared_history = data.get("shared_history", [])
        return grp


class MultiNPCInteractionManager:
    """
    Manages interactions between multiple NPCs, including group dynamics, factional behavior,
    and coordinated activities (e.g., multi-NPC scenes).
    """

    GROUP_DYNAMICS = {
        "hierarchy": {
            "description": "Formalized power structure in the group",
            "effects": "Chain of command and override authority"
        },
        "cohesion": {
            "description": "How unified the group is in goals/behavior",
            "effects": "Synergy vs. friction; influences group stability"
        },
        "secrecy": {
            "description": "How much the group hides from outsiders",
            "effects": "Information control and shared secrecy"
        },
        "territoriality": {
            "description": "Protectiveness over members/resources",
            "effects": "Reactions to perceived threats or intrusion"
        },
        "exclusivity": {
            "description": "Difficulty to join / acceptance threshold",
            "effects": "Initiation tests, membership gating"
        },
    }

    INTERACTION_STYLES = {
        "coordinated": {
            "description": "NPCs act in a coordinated, deliberate manner",
            "requirements": {"cohesion": 70},
            "dialogue_style": "NPCs build on each other's statements smoothly."
        },
        "hierarchical": {
            "description": "NPCs follow a clear status hierarchy",
            "requirements": {"hierarchy": 70},
            "dialogue_style": "Lower-status NPCs defer to higher-status NPCs."
        },
        "competitive": {
            "description": "NPCs compete for dominance or attention",
            "requirements": {"cohesion": -40, "hierarchy": -30},
            "dialogue_style": "NPCs interrupt or attempt to outdo each other."
        },
        "consensus": {
            "description": "NPCs seek group agreement before acting",
            "requirements": {"cohesion": 60, "hierarchy": -40},
            "dialogue_style": "NPCs exchange opinions politely, aiming for unity."
        },
        "protective": {
            "description": "NPCs protect and defend one target or idea",
            "requirements": {"territoriality": 70},
            "dialogue_style": "NPCs focus on ensuring safety or enforcing boundaries."
        },
        "exclusionary": {
            "description": "NPCs deliberately exclude someone (the player or another NPC)",
            "requirements": {"exclusivity": 70},
            "dialogue_style": "NPCs speak in code or inside references, ignoring outsiders."
        },
        "manipulative": {
            "description": "NPCs coordinate to manipulate a target",
            "requirements": {"cohesion": 60, "secrecy": 70},
            "dialogue_style": "NPCs set conversational traps, do good-cop/bad-cop routines."
        },
    }


    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def create_npc_group(
        self, ctx, conn, name: str, description: str, member_ids: List[int]
    ) -> Dict[str, Any]:
        """Create a new NPC group using canon."""
        members_data = []
        
        # Validate NPCs and gather data
        for npc_id in member_ids:
            row = await conn.fetchrow(
                """
                SELECT npc_id, npc_name, dominance, cruelty
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                ctx.user_id, ctx.conversation_id, npc_id
            )
            if not row:
                return {"error": f"NPC with ID {npc_id} not found."}
            members_data.append({
                "npc_id": row['npc_id'], "npc_name": row['npc_name'],
                "dominance": row['dominance'], "cruelty": row['cruelty'],
                "role": "member", "status": "active", "joined_date": datetime.now().isoformat()
            })

        # Generate initial dynamics and roles
        dynamics = {key: random.randint(20, 80) for key in self.GROUP_DYNAMICS.keys()}
        if len(members_data) > 1 and dynamics.get("hierarchy", 0) > 60:
            sorted_mem = sorted(members_data, key=lambda x: x.get("dominance", 0), reverse=True)
            sorted_mem[0]["role"] = "leader"
            members_data = sorted_mem

        group_obj = NPCGroup(name, description, members_data, dynamics)
        
        # Create through canon
        group_id = await canon.find_or_create_npc_group(ctx, conn, group_obj.to_dict())
        
        return {"success": True, "group_id": group_id, "message": f"Group '{name}' created."}

    async def get_npc_group(
        self, ctx, conn, group_id: Optional[int] = None, group_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve an NPC group by ID or name."""
        if not group_id and not group_name:
            return {"error": "Must provide group_id or group_name."}
        try:
            query = """SELECT group_id, group_name, group_data FROM NPCGroups
                       WHERE user_id = $1 AND conversation_id = $2"""
            params = [ctx.user_id, ctx.conversation_id]
            if group_id:
                query += " AND group_id = $3"
                params.append(group_id)
            else:
                query += " AND group_name = $3"
                params.append(group_name)

            row = await conn.fetchrow(query, *params)
            if not row:
                lookup = f"ID {group_id}" if group_id else f"name '{group_name}'"
                return {"error": f"Group with {lookup} not found."}

            real_group_id, real_group_name, group_data = row['group_id'], row['group_name'], row['group_data']
            group_dict = group_data if isinstance(group_data, dict) else (json.loads(group_data) if isinstance(group_data, str) else {})

            group_obj = NPCGroup.from_dict(group_dict, group_id=real_group_id)
            return {
                "success": True,
                "group_id": real_group_id,
                "group_name": real_group_name,
                "group_data": group_obj.to_dict(),
                "group_object": group_obj
            }
        except Exception as e:
            logger.error(f"Error retrieving group: {e}", exc_info=True)
            return {"error": str(e)}

    async def update_group_dynamics(
        self, ctx, group_id: int, changes: Dict[str, int]
    ) -> Dict[str, Any]:
        """Apply increments/decrements to group dynamics using LoreSystem."""
        try:
            # Get LoreSystem instance
            lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
            
            # Fetch current group data
            async with get_db_connection_context() as conn:
                group_data = await conn.fetchval(
                    """
                    SELECT group_data FROM NPCGroups
                    WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3
                    """,
                    ctx.user_id, ctx.conversation_id, group_id
                )
                if group_data is None:
                    return {"error": "Group not found"}

                group_dict = group_data if isinstance(group_data, dict) else json.loads(group_data)
                group_obj = NPCGroup.from_dict(group_dict, group_id=group_id)

                # Apply changes
                dynamics_updated = False
                for dyn_key, delta in changes.items():
                    if dyn_key in self.GROUP_DYNAMICS:
                        current = group_obj.dynamics.get(dyn_key, 0)
                        new_val = max(0, min(100, current + delta))
                        if new_val != current:
                            group_obj.dynamics[dyn_key] = new_val
                            dynamics_updated = True

                if not dynamics_updated:
                    return {"success": True, "message": "No dynamics changed."}

                # Record history and update
                group_obj.shared_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "dynamics_update",
                    "details": changes,
                    "new_dynamics": group_obj.dynamics
                })
                group_obj.last_activity = datetime.now().isoformat()

                # Update using LoreSystem
                result = await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCGroups",
                    entity_identifier={"group_id": group_id},
                    updates={"group_data": json.dumps(group_obj.to_dict())},
                    reason=f"Updated group dynamics: {changes}"
                )

                if result["status"] == "committed":
                    return {"success": True, "message": "Group dynamics updated.", "updated_dynamics": group_obj.dynamics}
                else:
                    return {"error": f"Failed to update: {result}"}

        except Exception as e:
            logger.error(f"Error updating dynamics for group {group_id}: {e}", exc_info=True)
            return {"error": str(e)}


    def determine_interaction_style(self, group_obj: NPCGroup) -> str:
        """
        Based on the group's dynamics, pick an appropriate style.
        """
        candidates = []
        for style_name, style_def in self.INTERACTION_STYLES.items():
            reqs = style_def["requirements"]
            meets_all = True
            for dyn_key, threshold in reqs.items():
                current_val = group_obj.dynamics.get(dyn_key, 0)
                if threshold >= 0 and current_val < threshold:
                    meets_all = False
                    break
                elif threshold < 0 and current_val > abs(threshold):
                    meets_all = False
                    break
            if meets_all:
                candidates.append(style_name)

        if not candidates:
            return "neutral"
        return random.choice(candidates)

    async def produce_multi_npc_scene(
        self,
        ctx,
        conn,
        group_id: int,
        topic: str = "General conversation",
        extra_context: str = ""
    ) -> Dict[str, Any]:
        """Create scene snippet and update group history."""
        group_info = await self.get_npc_group(ctx, conn, group_id=group_id)
        if "error" in group_info or "group_object" not in group_info:
             return {"error": group_info.get("error", "Failed to retrieve group object.")}

        group_obj: NPCGroup = group_info["group_object"]

        style = self.determine_interaction_style(group_obj)
        style_def = self.INTERACTION_STYLES.get(style, {})
        dialogue_style_desc = style_def.get("dialogue_style", "Plain interaction.")
        desc = style_def.get("description", "")

        # Build demo lines
        members_to_include = random.sample(group_obj.members, k=min(len(group_obj.members), 3))
        lines = [f"{mem['npc_name']} ({mem.get('role', 'member')}): [Responding to {topic} with style '{style}']" for mem in members_to_include]
        scene_text = "\n".join(lines)

        # Update history and timestamp
        group_obj.shared_history.append({
            "timestamp": datetime.now().isoformat(), "type": "scene_example",
            "style": style, "topic": topic, "lines": lines
        })
        group_obj.last_activity = datetime.now().isoformat()

        # Persist changes using LoreSystem
        lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
        result = await lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCGroups",
            entity_identifier={"group_id": group_id},
            updates={"group_data": json.dumps(group_obj.to_dict())},
            reason=f"Produced scene for topic: {topic}"
        )
        
        return {
            "success": result["status"] == "committed",
            "interaction_style": style,
            "style_description": desc,
            "dialogue_style": dialogue_style_desc,
            "scene_preview": scene_text
        }

    async def list_all_groups(self, ctx, conn) -> Dict[str, Any]:
        """Return a list of all NPC groups."""
        results = []
        try:
            rows = await conn.fetch(
                """
                SELECT group_id, group_name, group_data FROM NPCGroups
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY group_name
                """,
                ctx.user_id, ctx.conversation_id
            )
            for row in rows:
                g_data = row['group_data']
                g_dict = g_data if isinstance(g_data, dict) else (json.loads(g_data) if isinstance(g_data, str) else {})
                results.append({
                    "group_id": row['group_id'],
                    "group_name": row['group_name'],
                    "data": g_dict
                })
            return {"groups": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Error listing groups: {e}", exc_info=True)
            return {"error": str(e)}

    async def delete_group(self, ctx, group_id: int) -> Dict[str, Any]:
        """Delete an NPC group using LoreSystem."""
        # LoreSystem doesn't have a delete method, so we'd need to add one
        # For now, we'll use direct database access for deletion
        try:
            async with get_db_connection_context() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM NPCGroups
                    WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3
                    """,
                    ctx.user_id, ctx.conversation_id, group_id
                )
                if result == "DELETE 1":
                    # Log the deletion as a canonical event
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"NPC group {group_id} was deleted",
                        tags=['npc_group', 'deletion'],
                        significance=5
                    )
                    return {"success": True, "message": f"Group {group_id} deleted."}
                else:
                    return {"error": f"Group {group_id} not found or not deleted."}
        except Exception as e:
            logger.error(f"Error deleting group {group_id}: {e}", exc_info=True)
            return {"error": str(e)}
         
    async def update_npc_group_dynamics(
        self,
        ctx,
        group_id: int,
        dynamics_data: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Update NPC group dynamics and potentially related relationships using LoreSystem.
        """
        updates_applied = {"group_dynamics": [], "member_relationships": []}
        try:
            # Get LoreSystem instance
            lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
            
            async with get_db_connection_context() as conn:
                # Get current group data
                group_data = await conn.fetchval(
                    """SELECT group_data FROM NPCGroups
                       WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3""",
                    ctx.user_id, ctx.conversation_id, group_id
                )
                if group_data is None:
                    return {"success": False, "error": "Group not found."}

                group_dict = group_data if isinstance(group_data, dict) else json.loads(group_data)
                group_obj = NPCGroup.from_dict(group_dict, group_id=group_id)

                # Apply dynamics changes
                changed_dynamics = {}
                for dynamic_key, new_value in dynamics_data.items():
                    if dynamic_key in self.GROUP_DYNAMICS:
                        clamped_value = max(0, min(100, new_value))
                        if group_obj.dynamics.get(dynamic_key) != clamped_value:
                            group_obj.dynamics[dynamic_key] = clamped_value
                            changed_dynamics[dynamic_key] = clamped_value

                if not changed_dynamics:
                    return {"success": True, "message": "No changes applied.", "updates_applied": updates_applied}

                # Update history
                group_obj.shared_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "dynamics_set",
                    "details": changed_dynamics
                })
                group_obj.last_activity = datetime.now().isoformat()

                # Update group using LoreSystem
                result = await lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCGroups",
                    entity_identifier={"group_id": group_id},
                    updates={"group_data": json.dumps(group_obj.to_dict())},
                    reason=f"Updated group dynamics: {changed_dynamics}"
                )

                if result["status"] != "committed":
                    return {"success": False, "error": f"Failed to update group: {result}"}

                updates_applied["group_dynamics"] = changed_dynamics

                # Update member relationships based on cohesion
                if "cohesion" in changed_dynamics and group_obj.members and len(group_obj.members) > 1:
                    cohesion_level = changed_dynamics["cohesion"]
                    level_change = 0
                    if cohesion_level > 70: level_change = 3
                    elif cohesion_level < 30: level_change = -3

                    if level_change != 0:
                        for i in range(len(group_obj.members)):
                            for j in range(i + 1, len(group_obj.members)):
                                m1_id = group_obj.members[i]["npc_id"]
                                m2_id = group_obj.members[j]["npc_id"]

                                # Find link
                                link_id = await conn.fetchval(
                                    """SELECT link_id FROM SocialLinks
                                       WHERE user_id=$1 AND conversation_id=$2
                                       AND ((entity1_type='npc' AND entity1_id=$3 AND entity2_type='npc' AND entity2_id=$4)
                                         OR (entity1_type='npc' AND entity1_id=$4 AND entity2_type='npc' AND entity2_id=$3))
                                    """, ctx.user_id, ctx.conversation_id, m1_id, m2_id
                                )

                                if link_id:
                                    # Get current dynamics
                                    current_link_dynamics = await conn.fetchval(
                                        "SELECT dynamics FROM SocialLinks WHERE link_id=$1", link_id
                                    )
                                    link_dynamics = current_link_dynamics if isinstance(current_link_dynamics, dict) else json.loads(current_link_dynamics or '{}')

                                    trust_level = link_dynamics.get('trust', 0)
                                    new_trust = max(-100, min(100, trust_level + level_change))

                                    if new_trust != trust_level:
                                        link_dynamics['trust'] = new_trust
                                        
                                        # Update using LoreSystem
                                        link_result = await lore_system.propose_and_enact_change(
                                            ctx=ctx,
                                            entity_type="SocialLinks",
                                            entity_identifier={"link_id": link_id},
                                            updates={"dynamics": json.dumps(link_dynamics)},
                                            reason=f"Group cohesion change affected member relationships"
                                        )
                                        
                                        if link_result["status"] == "committed":
                                            updates_applied["member_relationships"].append({
                                                "member1_id": m1_id, "member2_id": m2_id,
                                                "change": {"trust": level_change}
                                            })

            return {"success": True, "updates_applied": updates_applied}

        except Exception as e:
            logger.error(f"Error updating group dynamics for {group_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 8) Tools (function_tool) that the agent can call
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@function_tool
async def get_relationship_insights_tool(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str = "general"
) -> dict:
    """Get insights about an NPC based on empathy and relationship depth."""
    async with get_db_connection_context() as conn:
        return await check_relationship_insights(ctx, conn, npc_id, situation)

@function_tool
async def perform_social_interaction_tool(
    ctx: RunContextWrapper,
    npc_id: int,
    interaction_type: str
) -> dict:
    """Perform a stat-based social interaction."""
    async with get_db_connection_context() as conn:
        return await perform_social_interaction(ctx, conn, npc_id, interaction_type)

@function_tool
async def process_relationship_activity_tool(
    ctx: RunContextWrapper,
    npc_id: int,
    activity_type: str,
    duration: int = 1
) -> dict:
    """Process a relationship activity that affects stats and bonds."""
    async with get_db_connection_context() as conn:
        return await process_relationship_activity(ctx, conn, npc_id, activity_type, duration)


@function_tool
async def get_social_link_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """
    Get an existing social link's details if it exists.
    """
    async with get_db_connection_context() as conn:
        link = await get_social_link(
            ctx, conn, entity1_type, entity1_id, entity2_type, entity2_id
        )
    if link is None:
        return {"error": "No link found"}
    return link


@function_tool
async def create_social_link_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    link_type: str = "neutral",
    link_level: int = 0
) -> dict:
    """
    Create a new social link, or return the existing link_id if it already exists.
    """
    async with get_db_connection_context() as conn:
        link_id = await create_social_link(
            ctx, conn, entity1_type, entity1_id,
            entity2_type, entity2_id, link_type, link_level
        )
    return {"link_id": link_id, "message": "Link created or fetched."}


@function_tool
async def update_link_type_and_level_tool(
    ctx: RunContextWrapper,
    link_id: int,
    new_type: str = None,
    level_change: int = 0
) -> dict:
    """
    Update an existing link's type and/or level.
    """
    result = await update_link_type_and_level(ctx, link_id, new_type, level_change)
    if result is None:
        return {"error": "Link not found or update failed"}
    return result


@function_tool
async def add_link_event_tool(
    ctx: RunContextWrapper,
    link_id: int,
    event_text: str
) -> dict:
    """
    Append an event string to a link's link_history.
    """
    success = await add_link_event(ctx, link_id, event_text)
    return {"success": success, "message": "Event added to link_history" if success else "Failed to add event"}


@function_tool
async def check_for_crossroads_tool(ctx: RunContextWrapper) -> dict:
    """
    Check if there's a relationship crossroads event triggered.
    """
    async with get_db_connection_context() as conn:
        result = await check_for_relationship_crossroads(ctx, conn)
    if not result:
        return {"message": "No crossroads triggered"}
    return result


@function_tool
async def apply_crossroads_choice_tool(
    ctx: RunContextWrapper,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> dict:
    """
    Apply a chosen effect from a triggered crossroads.
    """
    return await apply_crossroads_choice(
        ctx, link_id, crossroads_name, choice_index
    )


@function_tool
async def check_for_ritual_tool(ctx: RunContextWrapper) -> dict:
    """
    Check if there's a relationship ritual event triggered.
    """
    async with get_db_connection_context() as conn:
        result = await check_for_relationship_ritual(ctx, conn)
    if not result:
        return {"message": "No ritual triggered"}
    return result


@function_tool
async def get_relationship_summary_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """
    Get a summary of the relationship.
    """
    async with get_db_connection_context() as conn:
        summary = await get_relationship_summary(
            ctx, conn, entity1_type, entity1_id, entity2_type, entity2_id
        )
    if not summary:
        return {"error": "No relationship found"}
    return summary


@function_tool(strict_mode=False)
async def update_relationships_from_conflict(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_data: Dict[str, Any]
) -> dict:
    """
    Update relationships based on conflict resolution outcomes.
    """
    updates_applied = []
    try:
        # Get LoreSystem instance
        lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
        
        async with get_db_connection_context() as conn:
            # Get stakeholders from conflict
            stakeholders = await conn.fetch(
                """
                SELECT npc_id, faction_position
                FROM ConflictStakeholders
                WHERE conflict_id = $1
                """,
                conflict_id
            )
            
            if not stakeholders:
                return {"success": True, "message": "No stakeholders found for conflict.", "updates_applied": []}

            for stakeholder in stakeholders:
                npc_id = stakeholder['npc_id']
                if not npc_id: continue

                # Get player's link with this NPC
                player_link_id = await conn.fetchval(
                    """SELECT link_id FROM SocialLinks
                       WHERE user_id=$1 AND conversation_id=$2
                       AND ((entity1_type='player' AND entity1_id=$1 AND entity2_type='npc' AND entity2_id=$3)
                         OR (entity1_type='npc' AND entity1_id=$3 AND entity2_type='player' AND entity2_id=$1))
                    """, ctx.user_id, ctx.conversation_id, npc_id
                )

                if not player_link_id: continue

                # Calculate changes based on resolution_data
                level_change = 0
                resolution_success = resolution_data.get("success", False)
                winning_faction = resolution_data.get("winning_faction")
                npc_faction = stakeholder.get("faction_position")

                if winning_faction:
                    if npc_faction == winning_faction:
                        level_change = 10 if resolution_success else -10
                    else:
                        level_change = -5 if resolution_success else 5
                else:
                    level_change = 3 if resolution_success else -3

                # Apply changes
                update_result = await update_link_type_and_level(
                    ctx, player_link_id, None, level_change
                )

                if update_result:
                    updates_applied.append({"npc_id": npc_id, "changes": update_result})
                    # Add history event
                    event_text = f"Relationship changed due to conflict {conflict_id} resolution: {resolution_data.get('outcome', 'unknown')}"
                    await add_link_event(ctx, player_link_id, event_text)

        return {"success": True, "updates_applied": updates_applied}

    except Exception as e:
        logger.error(f"Error updating relationships from conflict {conflict_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@function_tool(strict_mode=False)
async def update_relationship_context(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update relationship context (merges JSONB data).
    """
    if not context_data:
        return {"success": False, "error": "No context_data provided."}

    try:
        async with get_db_connection_context() as conn:
            # Find link_id
            link_id = await conn.fetchval(
                """SELECT link_id FROM SocialLinks
                   WHERE user_id=$1 AND conversation_id=$2
                   AND ((entity1_type=$3 AND entity1_id=$4 AND entity2_type=$5 AND entity2_id=$6)
                     OR (entity1_type=$5 AND entity1_id=$6 AND entity2_type=$3 AND entity2_id=$4))
                """, ctx.user_id, ctx.conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
            )

            if not link_id:
                # Create link first if it doesn't exist
                link_id = await create_social_link(
                    ctx, conn, entity1_type, entity1_id, entity2_type, entity2_id
                )
                if not link_id:
                    return {"success": False, "error": "Relationship not found and could not be created."}

            # Get current context
            current_context = await conn.fetchval(
                "SELECT context FROM SocialLinks WHERE link_id = $1",
                link_id
            )
            
            if current_context:
                merged_context = {**current_context, **context_data}
            else:
                merged_context = context_data

            # Update using LoreSystem
            lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="SocialLinks",
                entity_identifier={"link_id": link_id},
                updates={"context": json.dumps(merged_context)},
                reason=f"Updated relationship context: {json.dumps(context_data)}"
            )

            if result["status"] == "committed":
                # Add event to relationship history
                event_text = f"Relationship context updated: {json.dumps(context_data)}"
                await add_link_event(ctx, link_id, event_text)
                
                return {"success": True, "link_id": link_id, "context_updated": context_data}
            else:
                return {"success": False, "error": f"Failed to update: {result}"}

    except Exception as e:
        logger.error(f"Error updating relationship context: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
     
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9) The Agent: "SocialLinksAgent"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SocialLinksAgent = Agent(
    name="SocialLinksAgent",
    instructions=(
        "You are a specialized 'Social Links Agent' for a persona-like system. "
        "You have function tools that let you manage relationships:\n"
        " - get_social_link_tool(...)\n"
        " - create_social_link_tool(...)\n"
        " - update_link_type_and_level_tool(...)\n"
        " - add_link_event_tool(...)\n"
        " - check_for_crossroads_tool(...)\n"
        " - apply_crossroads_choice_tool(...)\n"
        " - check_for_ritual_tool(...)\n"
        " - get_relationship_summary_tool(...)\n"
        " - update_relationships_from_conflict(...)\n"
        " - get_relationship_insights_tool(...) - NEW: Use empathy to understand NPCs\n"
        " - perform_social_interaction_tool(...) - NEW: Stat-based social actions\n"
        " - process_relationship_activity_tool(...) - NEW: Activities with NPCs\n\n"
        "Use these tools to retrieve or update relationship data, trigger or apply crossroads, "
        "check for rituals, and perform stat-based social interactions. "
        "Return helpful final text or JSON summarizing your result."
    ),
    model = "gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.5),
    tools=[
        get_social_link_tool,
        create_social_link_tool,
        update_link_type_and_level_tool,
        add_link_event_tool,
        check_for_crossroads_tool,
        apply_crossroads_choice_tool,
        check_for_ritual_tool,
        get_relationship_summary_tool,
        update_relationships_from_conflict,
        update_relationship_context,
        get_relationship_insights_tool,
        perform_social_interaction_tool,
        process_relationship_activity_tool
    ],
)
