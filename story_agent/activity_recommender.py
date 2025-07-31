# story_agent/activity_recommender.py

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass

#####################################################
# 1) OPENAI AGENTS IMPORTS
#####################################################
from agents import Agent, Runner, ModelSettings, function_tool
from pydantic import BaseModel, Field

#####################################################
# 2) IMPORTS FROM YOUR APP'S CODE
#####################################################
from db.connection import get_db_connection_context
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from logic.aggregator_sdk import get_aggregated_roleplay_context

logger = logging.getLogger(__name__)

#####################################################
# 3) DEFINE OUTPUT SCHEMA
#####################################################

class ActivityRecommendation(BaseModel):
    """
    Represents a single recommended activity.
    """
    activity_name: str
    confidence_score: float  # 0-1
    reasoning: str
    participating_npcs: List[str]
    estimated_duration: str
    prerequisites: List[str]
    expected_outcomes: List[str]

class ActivityRecommendations(BaseModel):
    """
    A container for multiple recommended activities.
    The LLM should return one of these as the final output.
    """
    recommendations: List[ActivityRecommendation] = Field(
        ..., description="List of recommended activities."
    )


#####################################################
# 4) CREATE FUNCTION TOOLS FOR DB / AGGREGATOR CALLS
#####################################################

@function_tool
async def fetch_npc_personality_traits(user_id: int, conversation_id: int, npc_id: int) -> Dict[str, Any]:
    """
    Fetch NPC personality data (stats, mood, role, relationship_level, etc.) from DB.
    Returns a dict with keys like:
      "id", "name", "traits", "relationship_level", "current_mood", "role", ...
    """
    async with get_db_connection_context() as conn:
        query = """
            SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                   personality_traits, hobbies, likes, dislikes, current_location
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """
        row = await conn.fetchrow(query, user_id, conversation_id, npc_id)
        if not row:
            return {
                "id": npc_id,
                "name": "Unknown",
                "traits": [],
                "relationship_level": 0,
                "current_mood": "neutral",
                "role": "unknown"
            }
        
        # Parse JSON fields safely
        def safe_json_load(value):
            try:
                if isinstance(value, str):
                    return json.loads(value)
                return value or []
            except:
                return []
        
        personality_traits = safe_json_load(row["personality_traits"])
        hobbies = safe_json_load(row["hobbies"])
        likes = safe_json_load(row["likes"])
        dislikes = safe_json_load(row["dislikes"])

        # Simple mood logic (example)
        mood = "neutral"
        if row["intensity"] > 75:
            mood = "focused"
        elif row["dominance"] > 75:
            mood = "strict"
        elif row["cruelty"] > 75:
            mood = "stressed"
        elif row["trust"] > 75 and row["respect"] > 75:
            mood = "happy"

        # Simple role logic
        role = "neutral"
        if row["dominance"] > 70:
            role = "dominant"
        elif row["closeness"] > 70:
            role = "close"

        # Relationship level
        rel_level = int((row["closeness"] + row["trust"] + row["respect"]) / 3)

        return {
            "id": row["npc_id"],
            "name": row["npc_name"],
            "traits": personality_traits,
            "hobbies": hobbies,
            "likes": likes,
            "dislikes": dislikes,
            "stats": {
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"]
            },
            "relationship_level": rel_level,
            "current_mood": mood,
            "role": role,
            "current_location": row["current_location"]
        }

@function_tool
async def fetch_current_scenario_context(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Retrieves the scenario context (time_of_day, location, scenario_type, etc.)
    from context service or aggregator fallback, plus last 5 user messages for recent activities.
    """
    try:
        context_service = await get_context_service(user_id, conversation_id)
        context_data = await context_service.get_context()

        scenario_type = "default"
        if context_data.get("current_conflict"):
            scenario_type = "conflict"
        elif context_data.get("current_event"):
            scenario_type = "event"

        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT content 
                FROM messages
                WHERE conversation_id = $1 AND sender = 'user'
                ORDER BY created_at DESC
                LIMIT 5
            """, conversation_id)
            recent_activities = [r["content"] for r in rows]

        return {
            "scenario_type": scenario_type,
            "location": context_data.get("current_location", "Unknown"),
            "time_of_day": context_data.get("time_of_day", "Morning"),
            "mood": "neutral",
            "player_status": context_data.get("player_stats", {}),
            "recent_activities": recent_activities
        }

    except Exception as e:
        logger.warning(f"Error from context service: {e}, using aggregator fallback...")
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id)

        scenario_type = "default"
        if aggregator_data.get("current_roleplay", {}).get("CurrentConflict"):
            scenario_type = "conflict"
        elif aggregator_data.get("current_roleplay", {}).get("CurrentEvent"):
            scenario_type = "event"

        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT content 
                FROM messages
                WHERE conversation_id = $1 AND sender = 'user'
                ORDER BY created_at DESC
                LIMIT 5
            """, conversation_id)
            recent_activities = [r["content"] for r in rows]

        return {
            "scenario_type": scenario_type,
            "location": aggregator_data.get("current_location", "Unknown"),
            "time_of_day": aggregator_data.get("time_of_day", "Morning"),
            "mood": "neutral",
            "player_status": aggregator_data.get("player_stats", {}),
            "recent_activities": recent_activities
        }

@function_tool
async def fetch_npc_relationship_state(user_id: int, conversation_id: int, npc_id: int) -> Dict[str, Any]:
    """Fetch relationship state between player and NPC."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    state = await manager.get_relationship_state(
        "player", 0,
        "npc", npc_id
    )
    
    return {
        "dimensions": state.dimensions.to_dict(),
        "patterns": list(state.history.active_patterns),
        "archetypes": list(state.active_archetypes),
        "duration_days": state.get_duration_days()
    }

@function_tool
async def fetch_relevant_memories(user_id: int, conversation_id: int, npc_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves relevant memories for a given NPC from memory manager or fallback DB queries.
    Each memory is a dict with { content, type, timestamp }.
    """
    try:
        memory_manager = await get_memory_manager(user_id, conversation_id)
        # Get NPC name from DB
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
            npc_name = row["npc_name"] if row else f"NPC-{npc_id}"

        memories = await memory_manager.search_memories(
            query_text=f"activities with {npc_name}",
            limit=limit,
            tags=[npc_name.lower().replace(" ", "_")],
            use_vector=True
        )
        # Convert to dict if necessary
        out = []
        for m in memories:
            if hasattr(m, "to_dict"):
                out.append(m.to_dict())
            else:
                out.append(m)
        return out

    except Exception as e:
        logger.warning(f"Memory manager error: {e}, fallback to direct DB query...")

        results = []
        async with get_db_connection_context() as conn:
            # Try unified_memories table
            try:
                rows = await conn.fetch("""
                    SELECT memory_text, memory_type, timestamp
                    FROM unified_memories
                    WHERE user_id = $1 AND conversation_id = $2
                      AND entity_type = 'npc' AND entity_id = $3
                    ORDER BY timestamp DESC
                    LIMIT $4
                """, user_id, conversation_id, npc_id, limit)
                for row in rows:
                    results.append({
                        "content": row["memory_text"],
                        "type": row["memory_type"],
                        "timestamp": row["timestamp"].isoformat()
                    })
            except Exception as e2:
                logger.warning(f"Error querying unified_memories: {e2}, trying NPCMemories")
                # Try NPCMemories
                try:
                    rows = await conn.fetch("""
                        SELECT memory_text, memory_type, timestamp
                        FROM NPCMemories
                        WHERE npc_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, npc_id, limit)
                    for row in rows:
                        results.append({
                            "content": row["memory_text"],
                            "type": row["memory_type"],
                            "timestamp": row["timestamp"].isoformat()
                        })
                except Exception as e3:
                    logger.error(f"Error querying NPCMemories: {e3}")
        return results

@function_tool
async def fetch_available_activities(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """
    Fetch possible activities from the Activities table. Return a list of dicts, each with keys like:
        "name", "category", "preferred_traits", "avoided_traits", "preferred_times", "prerequisites", "outcomes"
    """
    async with get_db_connection_context() as conn:
        try:
            rows = await conn.fetch("""
                SELECT name, purpose, stat_integration, intensity_tiers, setting_variants, fantasy_level
                FROM Activities
            """)
            results = []
            for r in rows:
                # Convert these fields from JSON, if present
                def safe_json_load(value, default={}):
                    try:
                        if isinstance(value, str):
                            return json.loads(value)
                        return value or default
                    except:
                        return default

                purpose = safe_json_load(r["purpose"], {})
                stat_integration = safe_json_load(r["stat_integration"], {})
                intensity_tiers = safe_json_load(r["intensity_tiers"], {})
                setting_variants = safe_json_load(r["setting_variants"], {})

                # Identify category heuristically
                category = "social"
                if "training" in purpose or "skill" in purpose:
                    category = "training"
                elif "service" in purpose or "help" in purpose:
                    category = "service"
                elif "relaxation" in purpose or "leisure" in purpose:
                    category = "relaxation"
                elif "challenge" in purpose or "test" in purpose:
                    category = "challenge"

                preferred_traits = purpose.get("compatibility", {}).get("preferred", [])
                avoided_traits = purpose.get("compatibility", {}).get("avoided", [])
                preferred_times = purpose.get("timing", {}).get("preferred_times", [])

                results.append({
                    "name": r["name"],
                    "category": category,
                    "purpose": purpose,
                    "stat_integration": stat_integration,
                    "intensity_tiers": intensity_tiers,
                    "setting_variants": setting_variants,
                    "fantasy_level": r["fantasy_level"],
                    "preferred_traits": preferred_traits,
                    "avoided_traits": avoided_traits,
                    "preferred_times": preferred_times,
                    "prerequisites": purpose.get("prerequisites", []),
                    "outcomes": stat_integration.get("outcomes", []),
                })
            return results

        except Exception as e:
            logger.error(f"Error fetching Activities from DB: {e}")
            # Fallback: return default “canned” activities
            return [
                {
                    "name": "Training Session",
                    "category": "training",
                    "preferred_traits": ["disciplined", "focused"],
                    "avoided_traits": ["lazy"],
                    "preferred_times": ["morning", "afternoon"],
                    "prerequisites": ["training equipment"],
                    "outcomes": ["skill improvement", "increased discipline"]
                },
                {
                    "name": "Relaxation Time",
                    "category": "relaxation",
                    "preferred_traits": ["calm", "patient"],
                    "avoided_traits": ["anxious"],
                    "preferred_times": ["evening", "night"],
                    "prerequisites": [],
                    "outcomes": ["reduced stress", "improved mood"]
                },
                {
                    "name": "Social Gathering",
                    "category": "social",
                    "preferred_traits": ["outgoing", "friendly"],
                    "avoided_traits": ["antisocial"],
                    "preferred_times": ["afternoon", "evening"],
                    "prerequisites": [],
                    "outcomes": ["improved relationships", "social info"]
                }
            ]

#####################################################
# 5) DEFINE THE AGENT INSTRUCTIONS & CREATE THE AGENT
#####################################################

# This prompt explains to the LLM how to combine all the data from the function tools
# (scenario context, NPC stats, available activities, memories) to produce final recommendations.
ACTIVITY_RECOMMENDER_SYSTEM_PROMPT = """
You are the 'Activity Recommender Agent'. The user wants suggestions for activities,
based on:
 - scenario context (time_of_day, location, scenario_type, recent_activities),
 - any number of NPCs, each with personality traits, relationship_level, mood, etc.,
 - a database of possible activities, each with a category (social, training, etc.) and prerequisites,
 - relevant memories for each NPC if needed.

## GOAL
Generate a set of recommended activities that best fit the current context. 
You may call your function tools in any order to gather data. 
Once you have enough info, produce a final JSON object with the schema:
{
  "recommendations": [
    {
      "activity_name": str,
      "confidence_score": float,
      "reasoning": str,
      "participating_npcs": [str],
      "estimated_duration": str,
      "prerequisites": [str],
      "expected_outcomes": [str]
    },
    ...
  ]
}

## RELATIONSHIP CONSIDERATIONS
- Intensity should escalate as relationships develop further
- Activities should align with relationship states
- High trust enables intimate activities
- Low respect limits certain interactions
- Active patterns (push_pull, slow_burn) suggest specific activities
- Archetypes (mentor_student, rivals) open unique options
- Unresolved conflicts might block some activities

For each NPC, also call fetch_npc_relationship_state to understand the relationship dynamics.

## LOGIC HINTS
1) You can retrieve scenario context by calling `fetch_current_scenario_context(...)`.
2) For each NPC, call `fetch_npc_personality_traits(...)` and optionally `fetch_relevant_memories(...)`.
3) Then call `fetch_available_activities(...)`.
4) Evaluate how each activity aligns with:
   - timeOfDay preferences,
   - NPC moods & traits,
   - location constraints,
   - variety (avoid repeating something found in recent_activities).
5) Return 2-3 best matches as final. Possibly add a "None" or fallback.

Return ONLY valid JSON for the final answer. Do not wrap in Markdown or add extra commentary.
"""

activity_recommender_agent = Agent(
    name="ActivityRecommenderAgent",
    instructions=ACTIVITY_RECOMMENDER_SYSTEM_PROMPT,
    output_type=ActivityRecommendations,
    model="gpt-4.1-nano",
    tools=[
        fetch_npc_personality_traits,
        fetch_current_scenario_context,
        fetch_relevant_memories,
        fetch_available_activities,
        fetch_npc_relationship_state
    ],
    model_settings=ModelSettings(
        temperature=0.7,
        # top_p=1.0, etc. — configure as you wish
    )
)

#####################################################
# 6) A CONVENIENCE FUNCTION FOR USERS TO CALL
#####################################################

async def recommend_activities(
    user_id: int,
    conversation_id: int,
    scenario_id: str,
    npc_ids: List[int],
    num_recommendations: int = 2
) -> ActivityRecommendations:
    """
    Orchestrates a conversation with the ActivityRecommenderAgent to produce
    `num_recommendations` recommended activities.

    If npc_ids is empty, the LLM can figure out how to handle that
    (possibly by pulling from the DB or returning "None").
    """
    # We pass a single user-style input describing what we need
    # The agent has instructions telling it how to gather data.
    # The agent will call the function tools, gather context, and produce a structured output.
    user_prompt = (
        f"Please recommend up to {num_recommendations} activities. \n"
        f"user_id={user_id}, conversation_id={conversation_id}, scenario_id={scenario_id}, npc_ids={npc_ids}"
    )

    # Run the agent in a single turn
    result = await Runner.run(
        starting_agent=activity_recommender_agent,
        input=user_prompt
    )

    # result.final_output is automatically cast to ActivityRecommendations, 
    # because the agent was created with output_type=ActivityRecommendations
    return result.final_output

