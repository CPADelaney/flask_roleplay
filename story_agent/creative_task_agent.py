# story_agent/creative_task_agent.py
"""
A production-ready Agentic module that generates tasks for a Femdom scenario.

Features:
  - Defines a Pydantic output model (CreativeTask).
  - Wraps your data-fetching logic (NPC info, scenario context, memories) as function tools.
  - Provides a single "build_femdom_task" tool to finalize a custom creative femdom challenge.
  - Defines a FemdomTaskAgent with system instructions, enabling agentic orchestration via the
    OpenAI Agents SDK.

Usage:
  1) Install dependencies:
       pip install openai-agents
  2) Configure environment variables:
       export OPENAI_API_KEY=<your key>
  3) Run this script:
       python femdom_task_agent.py
     This will demonstrate calling the agent with a sample prompt and printing the final output.
"""

import json
import random
import logging
from typing import List, Dict, Any

# The Agents SDK
from agents import Agent, Runner, function_tool
from agents.exceptions import AgentsException

# Pydantic for structured outputs
from pydantic import BaseModel

# ------------- LOGGING SETUP -------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------
# 1) DEFINE THE Pydantic MODEL for FINAL OUTPUT
# --------------------------------------------------------------------------
class CreativeTask(BaseModel):
    """
    Represents a creative and contextually appropriate Femdom task or challenge.

    Fields:
      - title: Name/title of the challenge.
      - description: Explanation or prompt describing what's required.
      - duration: Time estimate or limit.
      - difficulty: 1-5 scale representing how difficult/intense the task is.
      - required_items: A list of any items/tools the user must have.
      - success_criteria: How success is measured or recognized.
      - reward_type: What the user will gain if completed satisfactorily.
      - npc_involvement: Description of how the dominant NPC is involved/overseeing.
      - task_type: E.g., "skill_challenge", "service", "performance", etc.
    """
    title: str
    description: str
    duration: str
    difficulty: int
    required_items: List[str]
    success_criteria: str
    reward_type: str
    npc_involvement: str
    task_type: str


# --------------------------------------------------------------------------
# 2) WRAP YOUR DATA-FETCHING/CONTEXT METHODS AS FUNCTION TOOLS
#    These let the Agent dynamically call them. Each function can be
#    called by name with the arguments, returning structured data.
# --------------------------------------------------------------------------

@function_tool
async def get_relationship_context(user_id: int, conversation_id: int, npc_id: int) -> Dict[str, Any]:
    """Get relationship context for task generation."""
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    state = await manager.get_relationship_state("player", 0, "npc", npc_id)
    
    # Determine appropriate task intensity based on relationship
    if "toxic_bond" in state.active_archetypes:
        intensity_modifier = 1.5
    elif "mentor_student" in state.active_archetypes:
        intensity_modifier = 0.8
    else:
        intensity_modifier = 1.0
    
    # Check for relationship events
    from logic.dynamic_relationships import poll_relationship_events_tool
    ctx = type('obj', (object,), {'context': {'user_id': user_id, 'conversation_id': conversation_id}})
    events = await poll_relationship_events_tool(ctx)
    
    return {
        "relationship_quality": state.dimensions.trust + state.dimensions.affection,
        "power_dynamic": state.dimensions.influence,
        "intensity_modifier": intensity_modifier,
        "active_patterns": list(state.history.active_patterns),
        "pending_events": events.get("has_event", False)
    }

@function_tool
async def get_npc_data(user_id: int, conversation_id: int, npc_id: int) -> Dict[str, Any]:
    """
    Retrieve core NPC data from the database (dominance, cruelty, closeness, etc.).

    Args:
      user_id: The user ID.
      conversation_id: The conversation ID.
      npc_id: The NPC ID.

    Returns:
      A dictionary with NPC stats and parsed fields:
        {
          "id": int,
          "name": str,
          "traits": [str, ...],
          "role": str,
          "hobbies": [str, ...],
          "likes": [str, ...],
          "dislikes": [str, ...],
          "stats": {
             "dominance": int,
             "cruelty": int,
             "closeness": int,
             "trust": int,
             "respect": int,
             "intensity": int
          },
          "description": str,
          "current_location": str
        }
    """
    # Example import based on your existing code.
    # Adjust paths to match your actual project structure:
    from db.connection import get_db_connection_context

    async with get_db_connection_context() as conn:
        query = """
            SELECT npc_id, npc_name, dominance, cruelty, closeness,
                   trust, respect, intensity, physical_description,
                   personality_traits, hobbies, likes, dislikes,
                   current_location
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """
        row = await conn.fetchrow(query, user_id, conversation_id, npc_id)

        if not row:
            return {
                "id": npc_id,
                "name": f"NPC-{npc_id}",
                "traits": [],
                "role": "unknown",
                "hobbies": [],
                "likes": [],
                "dislikes": [],
                "stats": {},
                "description": "",
                "current_location": "Unknown"
            }

        def parse_json(field):
            if not field:
                return []
            try:
                return json.loads(field) if isinstance(field, str) else (field or [])
            except (json.JSONDecodeError, TypeError):
                return []

        personality_traits = parse_json(row["personality_traits"])
        hobbies = parse_json(row["hobbies"])
        likes = parse_json(row["likes"])
        dislikes = parse_json(row["dislikes"])

        # Determine NPC role
        role = "neutral"
        if row["dominance"] > 70:
            role = "dominant"
        elif row["closeness"] > 70:
            role = "mentor"
        elif row["intensity"] > 70:
            role = "disciplinarian"

        return {
            "id": row["npc_id"],
            "name": row["npc_name"],
            "traits": personality_traits,
            "role": role,
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
            "description": row["physical_description"],
            "current_location": row["current_location"]
        }


@function_tool
async def get_scenario_context(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Retrieve the current scenario context (location, intensity level, scenario type, etc.).

    Args:
      user_id: The user ID.
      conversation_id: The conversation ID.

    Returns:
      A dictionary with scenario info:
        {
          "type": str,  # e.g. "conflict", "event", "introduction", ...
          "location": str,
          "intensity": int,  # 1-5
          "player_status": Dict[str, Any]
        }
    """
    from context.context_service import get_context_service
    from logic.aggregator_sdk import get_aggregated_roleplay_context

    try:
        context_service = await get_context_service(user_id, conversation_id)
        context_data = await context_service.get_context()

        scenario_type = "default"
        if context_data.get("current_conflict"):
            scenario_type = "conflict"
        elif context_data.get("current_event"):
            scenario_type = "event"
        elif "narrative_stage" in context_data:
            stage = (context_data["narrative_stage"].get("name") or "").lower()
            if "beginning" in stage:
                scenario_type = "introduction"
            elif "revelation" in stage:
                scenario_type = "revelation"

        return {
            "type": scenario_type,
            "location": context_data.get("current_location", "Unknown"),
            "intensity": context_data.get("intensity_level", 3),
            "player_status": context_data.get("player_stats", {})
        }

    except Exception as e:
        logger.warning(f"Error retrieving scenario context: {e}. Falling back to aggregator.")
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id)

        scenario_type = "default"
        crp = aggregator_data.get("current_roleplay", {})
        if crp.get("CurrentConflict"):
            scenario_type = "conflict"
        elif crp.get("CurrentEvent"):
            scenario_type = "event"

        # Simple intensity derivation
        player_stats = aggregator_data.get("player_stats", {})
        stat_sum = sum([
            abs(player_stats.get("corruption", 50) - 50),
            abs(player_stats.get("obedience", 50) - 50),
            abs(player_stats.get("dependency", 50) - 50),
        ])
        intensity = max(1, min(5, 1 + stat_sum // 30))

        return {
            "type": scenario_type,
            "location": aggregator_data.get("current_location", "Unknown"),
            "intensity": intensity,
            "player_status": player_stats
        }


@function_tool
async def get_relevant_memories(user_id: int, conversation_id: int, npc_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch relevant memories related to this NPC from memory systems or fallback DB queries.

    Args:
      user_id: The user ID.
      conversation_id: The conversation ID.
      npc_id: The NPC ID.
      limit: Max number of memories to retrieve.

    Returns:
      A list of memory objects (dicts).
    """
    from db.connection import get_db_connection_context
    from context.memory_manager import get_memory_manager

    try:
        memory_manager = await get_memory_manager(user_id, conversation_id)

        # get NPC name for better search tagging
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
            """, user_id, conversation_id, npc_id)
            npc_name = row["npc_name"] if row else f"NPC-{npc_id}"

        results = await memory_manager.search_memories(
            query_text=f"tasks with {npc_name}",
            limit=limit,
            tags=[npc_name.lower().replace(" ", "_")],
            use_vector=True
        )
        memories = []
        for r in results:
            if hasattr(r, 'to_dict'):
                memories.append(r.to_dict())
            else:
                memories.append(r)
        return memories

    except Exception as e:
        logger.warning(f"Memory manager error: {e}, using fallback DB queries.")
        memories = []
        async with get_db_connection_context() as conn:
            # Try unified_memories
            try:
                rows = await conn.fetch("""
                    SELECT memory_text, memory_type, timestamp
                    FROM unified_memories
                    WHERE user_id=$1 AND conversation_id=$2
                      AND entity_type='npc' AND entity_id=$3
                    ORDER BY timestamp DESC
                    LIMIT $4
                """, user_id, conversation_id, npc_id, limit)
                for row in rows:
                    memories.append({
                        "content": row["memory_text"],
                        "type": row["memory_type"],
                        "timestamp": row["timestamp"].isoformat()
                    })

            except Exception as e2:
                logger.warning(f"Error in unified_memories: {e2}; Trying NPCMemories.")
                try:
                    rows = await conn.fetch("""
                        SELECT memory_text, memory_type, timestamp
                        FROM NPCMemories
                        WHERE npc_id=$1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, npc_id, limit)
                    for row in rows:
                        memories.append({
                            "content": row["memory_text"],
                            "type": row["memory_type"],
                            "timestamp": row["timestamp"].isoformat()
                        })
                except Exception as e3:
                    logger.error(f"NPCMemories fallback error: {e3}")

        return memories


# --------------------------------------------------------------------------
# 3) SINGLE “SUPER” TOOL: build_femdom_task
#    This merges your existing random generation logic into one function.
#    The agent can choose to call it, or call your other tools individually.
# --------------------------------------------------------------------------
@function_tool
async def build_femdom_task(user_id: int, conversation_id: int, npc_id: int) -> CreativeTask:
    """
    Assemble a fully-formed Femdom creative task by retrieving:
      - NPC data
      - Scenario context
      - Relevant memories

    Then apply your random generation logic to produce a final CreativeTask 
    with a dominantly flavored scenario.

    Returns:
      A CreativeTask (as Pydantic model).
    """
    # We can call the other function tools directly here in Python.
    npc_data = await get_npc_data(user_id, conversation_id, npc_id)
    scenario = await get_scenario_context(user_id, conversation_id)
    memories = await get_relevant_memories(user_id, conversation_id, npc_id, limit=5)

    # EXTRACT minimal fields
    npc_name = npc_data.get("name", "Unknown NPC")
    scenario_type = scenario.get("type", "default")
    location = scenario.get("location", "Unknown")
    intensity = scenario.get("intensity", 3)
    # random selection for demonstration
    possible_task_types = ["skill_challenge", "service", "performance", "personal_growth", "leadership"]
    chosen_task_type = random.choice(possible_task_types)

    # Build out final fields
    # You can incorporate all the detailed random logic from your original code here
    # (some placeholders used below)
    title = f"{npc_name}'s Femdom {chosen_task_type.capitalize()} Challenge"
    description = (
        f"You must undertake a special {chosen_task_type} to impress {npc_name}, who is known "
        f"for unyielding standards. Scenario type: {scenario_type}, setting: {location}.\n"
        f"Memories hint at: {memories[:1] or 'No relevant memory found'}\n\n"
        f"Keep your posture perfect and your mind submissive. {npc_name} will not tolerate failure."
    )
    duration = f"{15 + (5 * intensity)} minutes"
    difficulty = intensity
    required_items = [
        "Utmost obedience", 
        "Concentrated focus"
    ]
    success_criteria = "Flawless adherence to the demands; no sign of disrespect or disobedience."
    reward_type = "Elevated standing in the eyes of your Mistress"
    npc_involvement = (
        f"{npc_name} personally supervises each step, ensuring discipline "
        f"and unwavering compliance throughout."
    )

    return CreativeTask(
        title=title,
        description=description,
        duration=duration,
        difficulty=difficulty,
        required_items=required_items,
        success_criteria=success_criteria,
        reward_type=reward_type,
        npc_involvement=npc_involvement,
        task_type=chosen_task_type
    )


# --------------------------------------------------------------------------
# 4) DEFINE THE FEMDOM AGENT
#    The agent sees these tools + instructions about how to produce final JSON.
# --------------------------------------------------------------------------
femdom_task_agent = Agent(
    name="FemdomTaskAgent",
    instructions="""
You are a Femdom Scenario Orchestrator. The user wants a creative, 
contextually appropriate femdom challenge. 

Your job:
  1. If you need NPC data, scenario details, or memories, call:
       - get_npc_data
       - get_scenario_context
       - get_relevant_memories
       - get_relationship_context
  2. You can also call build_femdom_task if you want a single function 
     that does everything in Python.
  3. Ultimately, produce the final answer as valid JSON that 
     matches the CreativeTask schema:
       {
         "title": ...
         "description": ...
         "duration": ...
         "difficulty": ...
         "required_items": [...],
         "success_criteria": ...
         "reward_type": ...
         "npc_involvement": ...
         "task_type": ...
       }
   4. Intensity should positively correlate with how much a relationship has progressed.

Keep your responses in a strictly femdom context. 
Ensure it remains in JSON format, with no extra commentary outside JSON.
""",
    model="gpt-4.1-nano"
    tools=[
        get_npc_data,
        get_scenario_context,
        get_relevant_memories,
        build_femdom_task,
        get_relationship_context
    ],
    output_type=CreativeTask  # Tells the model to produce a CreativeTask as final output
)
