# npcs/npc_decisions.py

"""
Decision-making engine for NPCs with integrated behavior evolution.
Refactored from logic/npc_agents/decision_engine.py and behavior_evolution.py.
"""

import logging
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.tracing import custom_span, function_span, generation_span
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from memory.core import MemoryType, MemorySignificance

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Pydantic models for tool inputs/outputs
# -------------------------------------------------------

class NPCAction(BaseModel):
    type: str
    description: str
    target: str
    stats_influenced: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    decision_metadata: Optional[Dict[str, Any]] = None

class NPCStats(BaseModel):
    npc_id: int
    npc_name: str
    dominance: float = 50.0
    cruelty: float = 50.0
    closeness: float = 50.0
    trust: float = 50.0
    respect: float = 50.0
    intensity: float = 50.0
    hobbies: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    schedule: Dict[str, Any] = Field(default_factory=dict)
    current_location: Optional[str] = None
    sex: Optional[str] = None

class NPCPerception(BaseModel):
    environment: Dict[str, Any] = Field(default_factory=dict)
    relevant_memories: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    flashback: Optional[Dict[str, Any]] = None
    traumatic_trigger: Optional[Dict[str, Any]] = None
    mask: Dict[str, Any] = Field(default_factory=dict)
    beliefs: List[Dict[str, Any]] = Field(default_factory=list)
    time_context: Dict[str, Any] = Field(default_factory=dict)
    narrative_context: Dict[str, Any] = Field(default_factory=dict)

class ActionOptions(BaseModel):
    available_actions: List[NPCAction] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None

class ScoredAction(BaseModel):
    action: NPCAction
    score: float
    reasoning: Dict[str, Any] = Field(default_factory=dict)

class BehaviorEvolFactors(BaseModel):
    """Factors affecting NPC behavior from evolutionary processes."""
    scheme_level: int = 0  # How scheming the NPC is (0-10)
    trust_modifiers: Dict[str, float] = Field(default_factory=dict)
    loyalty_tests: int = 0  # How many loyalty tests performed
    betrayal_planning: bool = False
    targeting_player: bool = False
    npc_recruits: List[int] = Field(default_factory=list)
    paranoia_level: int = 0  # How paranoid the NPC is (0-10)
    adaptation_score: float = 0.0  # How well the NPC adapts (0.0-1.0)

# -------------------------------------------------------
# Decision Context class
# -------------------------------------------------------

class DecisionContext:
    """Context to be passed between tools and agents in the decision engine."""
    
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.decision_log = []
        self.long_term_goals = []
        self.npc_stats = None
        self.perception = None
        self.behavior_evolution_factors = BehaviorEvolFactors()
    
    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(
                self.user_id, 
                self.conversation_id
            )
        return self.memory_system
    
    def add_to_decision_log(self, decision: Dict[str, Any]) -> None:
        """Add a decision to the log."""
        now = datetime.now()
        
        log_entry = {
            "timestamp": now.isoformat(),
            "decision": decision
        }
        
        self.decision_log.append(log_entry)
        
        # Limit size to prevent memory growth
        if len(self.decision_log) > 20:
            self.decision_log = self.decision_log[-20:]

# -------------------------------------------------------
# Behavior Evolution System
# -------------------------------------------------------

class BehaviorEvolution:
    """
    Evolves NPC behavior over time, modifying their tactics based on past events.
    NPCs will develop hidden agendas, adjust their manipulation strategies, 
    and attempt to control the world around them.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None

    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self.memory_system

    async def evaluate_npc_scheming(self, npc_id: int) -> Dict[str, Any]:
        """
        Periodically evaluate if an NPC should adjust their behavior.
        Returns:
            Dict containing the NPC's updated behavior factors.
        """
        try:
            memory_system = await self.get_memory_system()
            npc_data = await self._get_npc_data(npc_id) # Now async
            if not npc_data:
                logger.warning(f"NPC data not found for {npc_id} during scheming eval.")
                return {"error": "NPC data not found"}

            name = npc_data["npc_name"]
            dominance = npc_data["dominance"]
            cruelty = npc_data["cruelty"]
            paranoia = "paranoid" in npc_data.get("personality_traits", [])
            deceptive = "manipulative" in npc_data.get("personality_traits", [])

            # Retrieve past manipulations & betrayals
            betrayals = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="betrayal",
                limit=5
            )
            successful_lies = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="deception success",
                limit=5
            )
            failed_lies = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="deception failure",
                limit=3
            )
            loyalty_tests = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="tested loyalty",
                limit=3
            )

            # Define NPC evolution behavior
            adjustments = {
                "scheme_level": 0,  # Determines how aggressive their plotting is
                "trust_modifiers": {},
                "loyalty_tests": 0,
                "betrayal_planning": False,
                "targeting_player": False,
                "npc_recruits": [],
                "paranoia_level": 5 if paranoia else 2
            }

            # Adjust scheming level based on success rate
            if "memories" in successful_lies:
                adjustments["scheme_level"] += len(successful_lies["memories"])
            if "memories" in failed_lies:
                adjustments["scheme_level"] -= len(failed_lies["memories"])  # Punishes failures
            if "memories" in betrayals:
                adjustments["scheme_level"] += len(betrayals["memories"]) * 2  # Increases scheming if they've been betrayed

            # If their deception is failing often, they become either cautious or reckless
            if "memories" in failed_lies and paranoia:
                adjustments["scheme_level"] += 3  # Paranoia increases scheming
                adjustments["paranoia_level"] += 2  # Increase paranoia level too

            # If an NPC has tested loyalty and found weak targets, they begin manipulating more
            if "memories" in loyalty_tests:
                adjustments["loyalty_tests"] += len(loyalty_tests["memories"])
                
                # Extract NPCs who failed loyalty tests
                weak_targets = []
                for memory in loyalty_tests.get("memories", []):
                    memory_text = memory.get("text", "").lower()
                    # Extract NPC IDs from text (requires consistent memory format)
                    # This is a simplification - actual implementation would need better parsing
                    if "npc_" in memory_text and "failed" in memory_text:
                        for word in memory_text.split():
                            if word.startswith("npc_") and word[4:].isdigit():
                                weak_targets.append(int(word[4:]))

                adjustments["npc_recruits"].extend(weak_targets)

            # Dominant NPCs escalate manipulation if they see success
            if dominance > 70 and successful_lies.get("memories", []):
                adjustments["scheme_level"] += 2

            # Cruel NPCs escalate based on betrayals
            if cruelty > 70 and betrayals.get("memories", []):
                adjustments["betrayal_planning"] = True

            # Paranoid NPCs will target anyone they suspect of deception
            if paranoia and failed_lies.get("memories", []):
                adjustments["targeting_player"] = True

            # Final checks: If the NPC is in full scheming mode, they begin long-term plans
            if adjustments["scheme_level"] >= 5:
                logger.info(f"{name} is entering full scheming mode.")

                # Set a secret goal
                secret_goal = f"{name} is planning to manipulate the world around them."
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=secret_goal,
                    importance="high",
                    emotional=True
                )

                # If deceptive, they will now actively deceive the player
                if deceptive:
                    adjustments["targeting_player"] = True

                # NPC starts actively recruiting allies if they aren't already doing so
                if not adjustments["npc_recruits"]:
                    all_npcs = await self._get_all_npcs()
                    potential_recruits = [n["npc_id"] for n in all_npcs if n.get("dominance", 50) < 50]
                    adjustments["npc_recruits"].extend(potential_recruits[:2])

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating NPC scheming: {e}")
            return {"error": str(e)}

    async def _get_npc_data(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve NPC data from database asynchronously."""
        query = """
            SELECT npc_id, npc_name, dominance, cruelty, personality_traits
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """
        try:
            async with get_db_connection_context() as conn:
                row: Optional[asyncpg.Record] = await conn.fetchrow(
                    query, npc_id, self.user_id, self.conversation_id
                )

            if row:
                # Parse personality_traits safely
                traits = []
                raw_traits = row['personality_traits']
                if raw_traits:
                    try:
                        # asyncpg might already parse JSONB/JSON, check type
                        if isinstance(raw_traits, list):
                            traits = raw_traits
                        elif isinstance(raw_traits, str):
                             traits = json.loads(raw_traits)
                        # Add handling for dict if needed, though list seems expected
                    except (json.JSONDecodeError, TypeError) as parse_err:
                        logger.warning(f"Failed to parse personality_traits for NPC {npc_id}: {parse_err}. Data: {raw_traits}")
                        traits = []

                return {
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty'],
                    "personality_traits": traits
                }
            else:
                return None # NPC not found
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching NPC data for {npc_id}: {db_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching NPC data for {npc_id}: {e}", exc_info=True)
            return None

    # --- Updated to use asyncpg ---
    async def _get_all_npcs(self) -> List[Dict[str, Any]]:
        """Get all NPCs for this user/conversation asynchronously."""
        npcs = []
        query = """
            SELECT npc_id, npc_name, dominance, cruelty
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2
        """
        try:
            async with get_db_connection_context() as conn:
                rows: List[asyncpg.Record] = await conn.fetch(
                    query, self.user_id, self.conversation_id
                )

            for row in rows:
                npcs.append({
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty']
                })
            return npcs
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching all NPCs for user {self.user_id}, convo {self.conversation_id}: {db_err}", exc_info=True)
            return [] # Return empty list on error
        except Exception as e:
            logger.error(f"Unexpected error fetching all NPCs: {e}", exc_info=True)
            return []

    # --- Updated to use asyncpg ---
    async def apply_scheming_adjustments(self, npc_id: int, adjustments: Dict[str, Any]) -> None:
        """
        Apply scheming adjustments to the NPC in the database using LoreSystem.
        """
        try:
            # Get LoreSystem instance
            from lore.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            
            # Create context for governance
            ctx = type('obj', (object,), {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'npc_id': npc_id
            })
            
            # Prepare updates
            updates = {}
            
            new_level = adjustments.get("scheme_level", 0)
            updates["scheming_level"] = new_level
            
            betrayal_planning = adjustments.get("betrayal_planning", False)
            updates["betrayal_planning"] = betrayal_planning
            
            # Use LoreSystem to update
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="NPCStats",
                entity_identifier={"npc_id": npc_id},
                updates=updates,
                reason=f"Scheming adjustments applied: level={new_level}, betrayal={betrayal_planning}"
            )
            
            if result.get("status") == "committed":
                # Optional: Log successful memory update after DB commit
                if adjustments.get("targeting_player"):
                    memory_system = await self.get_memory_system()
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text="I decided to target the player, suspecting them of deception.",
                        significance=MemorySignificance.HIGH,
                        tags=["scheming", "targeting_player"]
                    )
                
                logger.info(f"Applied scheming adjustments for NPC {npc_id}: level={new_level}, betrayal={betrayal_planning}")
            else:
                logger.error(f"Failed to apply scheming adjustments via LoreSystem: {result}")
                
        except Exception as e:
            logger.error(f"Database error applying scheming adjustments for NPC {npc_id}: {e}", exc_info=True)


# -------------------------------------------------------
# GPT-based Action Generation
# -------------------------------------------------------

gpt_action_agent = Agent(
    name="NPC GPT Action Generator",
    instructions="""
    You are a creative system that, given an NPC's personality (dominance, cruelty, etc.), 
    emotional state, memories, and environment context, proposes a list of 3-6 possible 
    actions the NPC might take. Each action should be realistic, psychologically consistent, 
    and reflect the NPC's character.

    Output format (JSON):
    {
      "actions": [
        {
          "type": "string",
          "description": "string - short textual description",
          "target": "string - e.g. 'player', 'environment', or NPC ID',
          "stats_influenced": {...}  # optional
        },
        ...
      ]
    }
    """,
    # Could specify model="gpt-4" or "gpt-3.5-turbo" as desired
)

async def generate_dynamic_actions_with_gpt(
    npc_data: Dict[str, Any], 
    perception: Dict[str, Any],
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Calls GPT to generate candidate actions for the NPC, 
    based on their stats, emotions, relationships, and environment.
    
    Args:
        npc_data: NPCStats dict (dominance, cruelty, npc_name, etc.)
        perception: The dictionary from perceive_environment 
                    (with emotional_state, relevant_memories, relationships, etc.)
        top_n: Max number of actions to keep from GPT output

    Returns:
        A list of action dicts
    """
    with function_span("generate_dynamic_actions_with_gpt"):
        # Prepare prompt
        prompt_context = {
            "npc_data": npc_data,
            "perception": perception
        }
        prompt_str = json.dumps(prompt_context, indent=2)

        # Run the GPT Action agent
        try:
            result = await Runner.run(gpt_action_agent, prompt_str)
            raw_output = result.final_output

            # Attempt to parse as JSON
            parsed = {}
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                # If GPT doesn't return valid JSON, fallback
                parsed = {"actions": []}

            actions = parsed.get("actions", [])
            if not isinstance(actions, list):
                actions = []
            
            # Limit to top_n
            actions = actions[:top_n]

            return actions

        except Exception as e:
            logger.error(f"Error generating GPT actions: {e}")
            return []

# -------------------------------------------------------
# Tool Functions for Decision Making
# -------------------------------------------------------

@function_tool(strict_mode=False)
async def get_npc_data(ctx: RunContextWrapper[DecisionContext]) -> NPCStats:
    """
    Get the NPC's stats and traits from the database asynchronously. Caches result in context.
    """
    with function_span("get_npc_data"):
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id

        # Check cache first
        if ctx.context.npc_stats:
            logger.debug(f"Using cached NPC stats for {npc_id}")
            # Ensure it's the correct Pydantic type for return consistency
            if isinstance(ctx.context.npc_stats, dict):
                 try:
                    return NPCStats(**ctx.context.npc_stats)
                 except Exception: # Catch potential validation errors if cache is stale/bad
                    logger.warning(f"Cached NPC stats for {npc_id} failed validation, fetching fresh.")
                    ctx.context.npc_stats = None # Clear bad cache
            elif isinstance(ctx.context.npc_stats, NPCStats):
                 return ctx.context.npc_stats


        logger.debug(f"Fetching NPC stats from DB for {npc_id}")
        query = """
            SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                   hobbies, personality_traits, likes, dislikes, schedule, current_location, sex,
                   scheming_level, betrayal_planning
            FROM NPCStats
            WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
        """
        row: Optional[asyncpg.Record] = None
        try:
            # Use the async context manager directly
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, npc_id, user_id, conversation_id)

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error in get_npc_data tool for {npc_id}: {db_err}", exc_info=True)
            # Fallback to default stats on DB error
            default_stats = NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}_DB_Error")
            ctx.context.npc_stats = default_stats.model_dump()
            return default_stats
        except Exception as e:
             logger.error(f"Unexpected error in get_npc_data tool for {npc_id}: {e}", exc_info=True)
             default_stats = NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}_Error")
             ctx.context.npc_stats = default_stats.model_dump()
             return default_stats


        if not row:
            logger.warning(f"NPC {npc_id} not found in DB. Returning default stats.")
            default_stats = NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}_NotFound")
            ctx.context.npc_stats = default_stats.model_dump()
            return default_stats

        # Helper to safely parse JSON-like fields (list or string)
        def _parse_json_field(field_data: Any) -> Union[List, Dict]:
            if field_data is None:
                return [] # Default to list if usually a list
            if isinstance(field_data, (list, dict)): # Already parsed by asyncpg?
                return field_data
            if isinstance(field_data, str):
                try:
                    return json.loads(field_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON field: {field_data}")
                    return [] # Default to list on error
            logger.warning(f"Unexpected type for JSON field: {type(field_data)}")
            return [] # Default fallback

        # Safely access columns by name, providing defaults
        stats = NPCStats(
            npc_id=npc_id,
            npc_name=row['npc_name'] or f"NPC_{npc_id}_NoName",
            dominance=row['dominance'] or 50.0,
            cruelty=row['cruelty'] or 50.0,
            closeness=row['closeness'] or 50.0,
            trust=row['trust'] or 50.0,
            respect=row['respect'] or 50.0,
            intensity=row['intensity'] or 50.0,
            hobbies=_parse_json_field(row['hobbies']),
            personality_traits=_parse_json_field(row['personality_traits']),
            likes=_parse_json_field(row['likes']),
            dislikes=_parse_json_field(row['dislikes']),
            schedule=_parse_json_field(row['schedule']) or {}, # Default schedule to dict
            current_location=row['current_location'],
            sex=row['sex'],
            scheming_level=row['scheming_level'] or 0,
            betrayal_planning=row['betrayal_planning'] or False
        )

        # Cache the stats as a dict
        ctx.context.npc_stats = stats.model_dump()
        logger.debug(f"Fetched and cached NPC stats for {npc_id}")
        return stats

@function_tool(strict_mode=False)
async def get_default_actions(
    ctx: RunContextWrapper[DecisionContext],
    npc_data: NPCStats,
    perception: NPCPerception
) -> List[NPCAction]:
    """
    Generate a base set of actions for the NPC to consider,
    merging hard-coded actions and GPT-based dynamic suggestions.
    
    Args:
        npc_data: The NPC's stats and traits
        perception: The NPC's current perception
    """
    with function_span("get_default_actions"):
        # 1. Hard-coded baseline actions
        actions = [
            NPCAction(
                type="talk",
                description="Engage in friendly conversation",
                target="player",
                stats_influenced={"closeness": 2, "trust": 1}
            ),
            NPCAction(
                type="observe",
                description="Observe quietly",
                target="environment",
                stats_influenced={}
            ),
            NPCAction(
                type="leave",
                description="Exit the current location",
                target="location",
                stats_influenced={}
            )
        ]
        
        # 2. Additional logic for dominance, cruelty, etc.
        dominance = npc_data.dominance
        cruelty = npc_data.cruelty
        mask = perception.mask
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})
        mask_integrity = mask.get("integrity", 100)
        
        # Example: submissive-presented but hidden-dominant
        submissive_presented = "submissive" in presented_traits or "gentle" in presented_traits
        dominant_hidden = "dominant" in hidden_traits or "controlling" in hidden_traits
        
        if submissive_presented and dominant_hidden and mask_integrity < 70:
            actions.append(NPCAction(
                type="assertive",
                description="Show an unexpected hint of assertiveness",
                target="player",
                stats_influenced={"dominance": 2, "respect": -1}
            ))
        
        if dominance > 60 or "dominant" in presented_traits:
            actions.append(NPCAction(
                type="command",
                description="Give an authoritative command",
                target="player",
                stats_influenced={"dominance": 1, "trust": -1}
            ))
            actions.append(NPCAction(
                type="test",
                description="Test player's obedience",
                target="player",
                stats_influenced={"dominance": 2, "respect": -1}
            ))
            
            if dominance > 75:
                actions.append(NPCAction(
                    type="dominate",
                    description="Assert dominance forcefully",
                    target="player",
                    stats_influenced={"dominance": 3, "fear": 2}
                ))
                actions.append(NPCAction(
                    type="punish",
                    description="Punish disobedience",
                    target="player",
                    stats_influenced={"fear": 3, "obedience": 2}
                ))
        
        if cruelty > 60 or "cruel" in presented_traits:
            actions.append(NPCAction(
                type="mock",
                description="Mock or belittle the player",
                target="player",
                stats_influenced={"cruelty": 1, "closeness": -2}
            ))
            if cruelty > 70:
                actions.append(NPCAction(
                    type="humiliate",
                    description="Deliberately humiliate the player",
                    target="player",
                    stats_influenced={"cruelty": 2, "fear": 2}
                ))
        
        trust = npc_data.trust
        if trust > 60:
            actions.append(NPCAction(
                type="confide",
                description="Share a personal secret",
                target="player",
                stats_influenced={"trust": 3, "closeness": 2}
            ))
        
        respect = npc_data.respect
        if respect > 60:
            actions.append(NPCAction(
                type="praise",
                description="Praise the player's submission",
                target="player",
                stats_influenced={"respect": 2, "closeness": 1}
            ))
        
        # Emotional-state-based examples
        current_emotion = perception.emotional_state.get("current_emotion", {})
        if current_emotion:
            primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
            intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
            if intensity > 0.7:
                if primary_emotion == "anger":
                    actions.append(NPCAction(
                        type="express_anger",
                        description="Express anger forcefully",
                        target="player",
                        stats_influenced={"dominance": 2, "closeness": -3}
                    ))
                elif primary_emotion == "fear":
                    actions.append(NPCAction(
                        type="act_defensive",
                        description="Act defensively and guarded",
                        target="environment",
                        stats_influenced={"trust": -2}
                    ))
                elif primary_emotion == "joy":
                    actions.append(NPCAction(
                        type="celebrate",
                        description="Share happiness enthusiastically",
                        target="player",
                        stats_influenced={"closeness": 3}
                    ))
                elif primary_emotion in ["arousal", "desire"]:
                    actions.append(NPCAction(
                        type="seduce",
                        description="Make seductive advances",
                        target="player",
                        stats_influenced={"closeness": 2, "fear": 1}
                    ))
        
        # Context-based expansions
        environment_data = perception.environment
        loc_str = environment_data.get("location", "").lower()
        if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append(NPCAction(
                type="socialize",
                description="Engage in group conversation",
                target="group",
                stats_influenced={"closeness": 1}
            ))
        
        # Check for other NPCs in location
        for entity in environment_data.get("entities_present", []):
            if entity.get("type") == "npc":
                t_id = entity.get("id")
                t_name = entity.get("name", f"NPC_{t_id}")
                actions.append(NPCAction(
                    type="talk_to",
                    description=f"Talk to {t_name}",
                    target=str(t_id),
                    stats_influenced={"closeness": 1}
                ))
                if dominance > 70:
                    actions.append(NPCAction(
                        type="direct",
                        description=f"Direct {t_name} to do something",
                        target=str(t_id),
                        stats_influenced={"dominance": 1}
                    ))
        
        # Memory-based expansions
        memory_based_actions = await generate_memory_based_actions(ctx, perception)
        actions.extend(memory_based_actions)

        # 3. Use GPT to generate a few dynamic candidate actions
        npc_data_dict = npc_data.model_dump()
        perception_dict = perception.model_dump()
        gpt_candidates = await generate_dynamic_actions_with_gpt(npc_data_dict, perception_dict, top_n=6)
        for candidate in gpt_candidates:
            # Convert GPT's dict into NPCAction
            try:
                actions.append(NPCAction(**candidate))
            except:
                # If GPT gave partial data, skip
                pass
        
        # 4. Integrate behavior evolution factors
        behavior_factors = ctx.context.behavior_evolution_factors
        scheme_level = behavior_factors.scheme_level
        paranoia_level = behavior_factors.paranoia_level
        
        if scheme_level >= 5:
            actions.append(NPCAction(
                type="gather_info",
                description="Gather information for future manipulation",
                target="player",
                stats_influenced={"trust": -1},
                weight=1.5
            ))
            
            if scheme_level >= 7:
                actions.append(NPCAction(
                    type="manipulate",
                    description="Subtly manipulate the conversation",
                    target="player",
                    stats_influenced={"trust": -2},
                    weight=1.8
                ))
        
        if paranoia_level >= 6:
            actions.append(NPCAction(
                type="test_loyalty",
                description="Test the player's loyalty with a subtle challenge",
                target="player",
                stats_influenced={"trust": -1},
                weight=1.3
            ))
        
        if behavior_factors.targeting_player:
            actions.append(NPCAction(
                type="probe_weaknesses",
                description="Probe for weaknesses to exploit later",
                target="player",
                stats_influenced={"trust": -1},
                weight=1.4
            ))
        
        return actions

@function_tool(strict_mode=False)
async def generate_memory_based_actions(
    ctx: RunContextWrapper[DecisionContext],
    perception: NPCPerception
) -> List[NPCAction]:
    """
    Generate actions based on relevant memories.
    
    Args:
        perception: The NPC's current perception
    """
    with function_span("generate_memory_based_actions"):
        actions = []
        memories = perception.relevant_memories
        
        memory_topics = set()
        for memory in memories:
            memory_text = memory.get("text", "")
            # Simple topic extraction
            for topic_indicator in ["about", "mentioned", "discussed", "talked about", "interested in"]:
                if topic_indicator in memory_text.lower():
                    parts = memory_text.lower().split(topic_indicator, 1)
                    if len(parts) > 1:
                        topic_part = parts[1].strip()
                        words = topic_part.split()
                        if words:
                            topic = " ".join(words[:3])
                            topic = topic.rstrip(".,:;!?")
                            if len(topic) > 3 and topic not in memory_topics:
                                memory_topics.add(topic)
                                actions.append(NPCAction(
                                    type="discuss_topic",
                                    description=f"Discuss the topic of {topic}",
                                    target="player",
                                    stats_influenced={"closeness": 1}
                                ))
            
            # references to past interactions
            if "last time" in memory_text.lower() or "previously" in memory_text.lower():
                actions.append(NPCAction(
                    type="reference_past",
                    description="Reference a past interaction",
                    target="player",
                    stats_influenced={"trust": 1}
                ))
        
        # Look for patterns in memories that might suggest specific actions
        submission_pattern = any("submit" in m.get("text", "").lower() for m in memories)
        resistance_pattern = any("resist" in m.get("text", "").lower() for m in memories)
        
        if submission_pattern:
            actions.append(NPCAction(
                type="reward_submission",
                description="Reward the player's previous submission",
                target="player",
                stats_influenced={"closeness": 2, "respect": 1}
            ))
        if resistance_pattern:
            actions.append(NPCAction(
                type="address_resistance",
                description="Address the player's previous resistance",
                target="player",
                stats_influenced={"dominance": 2, "fear": 1}
            ))
        
        return actions

@function_tool(strict_mode=False)
async def score_actions(
    ctx: RunContextWrapper[DecisionContext],
    npc_data: NPCStats,
    perception: NPCPerception,
    actions: List[NPCAction]
) -> List[ScoredAction]:
    """
    Score each action based on personality, memories, emotional state, etc.
    
    Args:
        npc_data: The NPC's stats and traits
        perception: The NPC's current perception
        actions: List of possible actions
    """
    with function_span("score_actions"):
        scored_actions = []
        
        # Possibly apply memory biases
        memories = perception.relevant_memories
        memories = await apply_memory_biases(ctx, memories)
        
        # Extract key data from perception
        emotional_state = perception.emotional_state
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.5)
        
        # Mask data
        mask = perception.mask
        mask_integrity = mask.get("integrity", 100)
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})
        
        # Calculate player knowledge level
        player_knowledge = calculate_player_knowledge(perception, mask_integrity)
        
        # Check for flashback or trauma triggers
        flashback = perception.flashback
        traumatic_trigger = perception.traumatic_trigger
        
        # Get behavior evolution factors
        behavior_factors = ctx.context.behavior_evolution_factors
        
        # Score each action
        for action in actions:
            score = 0.0
            scoring_factors = {}
            
            # 1. Personality alignment
            personality_score = score_personality_alignment(
                npc_data, action, mask_integrity, hidden_traits, presented_traits
            )
            score += personality_score
            scoring_factors["personality_alignment"] = personality_score
            
            # 2. Memory influence
            memory_score = await score_memory_influence(memories, action)
            score += memory_score
            scoring_factors["memory_influence"] = memory_score
            
            # 3. Relationship influence
            relationship_score = score_relationship_influence(
                perception.relationships, action
            )
            score += relationship_score
            scoring_factors["relationship_influence"] = relationship_score
            
            # 4. Environmental context
            environment_score = score_environmental_context(
                perception.environment, action
            )
            score += environment_score
            scoring_factors["environmental_context"] = environment_score
            
            # 5. Emotional state influence
            emotional_score = score_emotional_influence(
                primary_emotion, emotion_intensity, action
            )
            score += emotional_score
            scoring_factors["emotional_influence"] = emotional_score
            
            # 6. Mask influence
            mask_score = score_mask_influence(
                mask_integrity, npc_data, action, hidden_traits, presented_traits
            )
            score += mask_score
            scoring_factors["mask_influence"] = mask_score
            
            # 7. Trauma influence
            trauma_score = 0.0
            if flashback or traumatic_trigger:
                trauma_score = score_trauma_influence(action, flashback, traumatic_trigger)
                score += trauma_score
            scoring_factors["trauma_influence"] = trauma_score
            
            # 8. Belief influence
            belief_score = score_belief_influence(perception.beliefs, action)
            score += belief_score
            scoring_factors["belief_influence"] = belief_score
            
            # 9. Decision history influence
            if ctx.context.decision_log:
                history_score = score_decision_history(ctx, action)
                score += history_score
                scoring_factors["decision_history"] = history_score
            else:
                scoring_factors["decision_history"] = 0.0
            
            # 10. Player knowledge influence
            player_knowledge_score = score_player_knowledge_influence(
                action, player_knowledge, hidden_traits
            )
            score += player_knowledge_score
            scoring_factors["player_knowledge"] = player_knowledge_score
            
            # 11. Time context
            time_context = perception.time_context
            if time_context:
                time_context_score = score_time_context_influence(time_context, action)
                score += time_context_score
                scoring_factors["time_context_influence"] = time_context_score
            
            # 12. Behavior evolution influence
            behavior_score = score_behavior_evolution_influence(behavior_factors, action)
            score += behavior_score
            scoring_factors["behavior_evolution"] = behavior_score
            
            scored_actions.append(ScoredAction(
                action=action,
                score=score,
                reasoning=scoring_factors
            ))
        
        # Sort by score descending
        scored_actions.sort(key=lambda x: x.score, reverse=True)
        
        # Log top 3 for debugging
        log_decision_reasoning(ctx, scored_actions[:3])
        
        return scored_actions

@function_tool(strict_mode=False)
async def select_action(
    ctx: RunContextWrapper[DecisionContext],
    scored_actions: List[ScoredAction],
    randomness: float = 0.2
) -> NPCAction:
    """
    Select an action from scored actions, with some randomness and pattern breaking.
    
    Args:
        scored_actions: List of actions with scores
        randomness: Randomness factor (0.0-1.0)
    """
    with function_span("select_action"):
        if not scored_actions:
            return NPCAction(
                type="idle",
                description="Do nothing",
                target="self"
            )
        
        # Check if we want to prioritize any action with a high "weight"
        weight_based_selection = any(
            sa.action.weight > 2.0 for sa in scored_actions
        )
        
        if weight_based_selection:
            weighted_actions = [(sa.action, sa.action.weight) for sa in scored_actions]
            weighted_actions.sort(key=lambda x: x[1], reverse=True)
            selected_action = weighted_actions[0][0]
        else:
            # Normal score-based selection with randomness
            for sa in scored_actions:
                sa.score += random.uniform(0, randomness * 10)
            scored_actions.sort(key=lambda x: x.score, reverse=True)
            selected_action = scored_actions[0].action
        
        # Add decision metadata
        if len(scored_actions) > 1:
            if selected_action.decision_metadata is None:
                selected_action.decision_metadata = {}
            selected_action.decision_metadata["alternative_actions"] = [
                {"type": sa.action.type, "score": sa.score}
                for sa in scored_actions[1:3]
            ]
        
        if scored_actions[0].reasoning:
            if selected_action.decision_metadata is None:
                selected_action.decision_metadata = {}
            selected_action.decision_metadata["reasoning"] = scored_actions[0].reasoning
        
        # Add the action to the decision log
        ctx.context.add_to_decision_log({
            "action": selected_action.model_dump(),
            "score": scored_actions[0].score,
            "reasoning": scored_actions[0].reasoning
        })
        
        return selected_action

@function_tool(strict_mode=False)
async def generate_flashback_action(
    ctx: RunContextWrapper[DecisionContext],
    flashback: Dict[str, Any],
    npc_data: NPCStats
) -> Optional[NPCAction]:
    """
    Generate an action in response to a flashback.
    
    Args:
        flashback: The flashback data
        npc_data: The NPC's stats and traits
    """
    with function_span("generate_flashback_action"):
        if not flashback:
            return None
            
        flashback_text = flashback.get("text", "")
        emotion = "neutral"
        intensity = 0.5
        
        if any(word in flashback_text.lower() for word in ["anger", "furious"]):
            emotion = "anger"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["scared", "fear"]):
            emotion = "fear"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["happy", "joy"]):
            emotion = "joy"
            intensity = 0.6
        elif any(word in flashback_text.lower() for word in ["submission", "obedient"]):
            emotion = "trust"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["dominant", "control"]):
            emotion = "anticipation"
            intensity = 0.8
        
        if emotion == "anger":
            return NPCAction(
                type="express_anger",
                description="Express anger triggered by a flashback",
                target="player",
                stats_influenced={"dominance": 2, "fear": 1},
                weight=1.8,
                decision_metadata={"flashback_source": True}
            )
        elif emotion == "fear":
            return NPCAction(
                type="act_defensive",
                description="Act defensively due to a flashback",
                target="environment",
                stats_influenced={"trust": -1},
                weight=1.7,
                decision_metadata={"flashback_source": True}
            )
        elif emotion == "joy":
            return NPCAction(
                type="reminisce",
                description="Reminisce about a positive memory",
                target="player",
                stats_influenced={"closeness": 2},
                weight=1.5,
                decision_metadata={"flashback_source": True}
            )
        elif emotion == "trust" and npc_data.dominance > 60:
            return NPCAction(
                type="expect_submission",
                description="Expect submission based on past experiences",
                target="player",
                stats_influenced={"dominance": 2, "fear": 1},
                weight=1.6,
                decision_metadata={"flashback_source": True}
            )
        elif emotion == "anticipation" and npc_data.dominance > 60:
            return NPCAction(
                type="dominate",
                description="Assert dominance triggered by a flashback",
                target="player",
                stats_influenced={"dominance": 3, "fear": 2},
                weight=1.8,
                decision_metadata={"flashback_source": True}
            )
        
        # default
        return NPCAction(
            type="reveal_flashback",
            description="Reveal being affected by a flashback",
            target="player",
            stats_influenced={"closeness": 1},
            weight=1.5,
            decision_metadata={"flashback_source": True}
        )

@function_tool(strict_mode=False)
async def enhance_dominance_context(
    ctx: RunContextWrapper[DecisionContext],
    action: NPCAction,
    npc_data: NPCStats,
    mask: Dict[str, Any]
) -> NPCAction:
    """
    Enhance actions with dominance context.
    
    Args:
        action: The chosen action
        npc_data: The NPC's stats and traits
        mask: The NPC's mask data
    """
    with function_span("enhance_dominance_context"):
        dominance = npc_data.dominance
        if dominance > 80 and action.type in ["command", "dominate", "punish"]:
            # Add more flavor
            action.description = "With overwhelming confidence, " + action.description
            action.stats_influenced["dominance"] = action.stats_influenced.get("dominance", 0) + 1
        
        return action

@function_tool(strict_mode=False)
async def update_behavior_evolution(
    ctx: RunContextWrapper[DecisionContext]
) -> Dict[str, Any]:
    """
    Update the NPC's behavior evolution factors.
    
    Returns:
        Updated behavior evolution factors
    """
    with function_span("update_behavior_evolution"):
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        # Create BehaviorEvolution instance
        behavior_evolution = BehaviorEvolution(user_id, conversation_id)
        
        # Evaluate NPC scheming
        adjustments = await behavior_evolution.evaluate_npc_scheming(npc_id)
        
        # Apply adjustments to NPC
        await behavior_evolution.apply_scheming_adjustments(npc_id, adjustments)
        
        # Update context
        ctx.context.behavior_evolution_factors = BehaviorEvolFactors(
            scheme_level=adjustments.get("scheme_level", 0),
            trust_modifiers=adjustments.get("trust_modifiers", {}),
            loyalty_tests=adjustments.get("loyalty_tests", 0),
            betrayal_planning=adjustments.get("betrayal_planning", False),
            targeting_player=adjustments.get("targeting_player", False),
            npc_recruits=adjustments.get("npc_recruits", []),
            paranoia_level=adjustments.get("paranoia_level", 0)
        )
        
        return adjustments

# -------------------------------------------------------
# Helper Functions for Memory Bias & Scoring
# -------------------------------------------------------

async def apply_memory_biases(
    ctx: RunContextWrapper[DecisionContext],
    memories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply a chain of biases to memories based on personality.
    """
    if not memories:
        return memories
    
    memories = await apply_recency_bias(memories)
    memories = await apply_emotional_bias(memories)
    memories = await apply_personality_bias(ctx, memories)
    
    # Sort by adjusted relevance
    memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return memories

async def apply_recency_bias(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply recency bias to memories.
    """
    now = datetime.now()
    
    for mem in memories:
        ts = mem.get("timestamp")
        
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                days_ago = (now - dt).days
            except ValueError:
                days_ago = 30
        elif isinstance(ts, datetime):
            days_ago = (now - ts).days
        else:
            days_ago = 30
        
        # 0..1 recency factor
        recency_factor = max(0, 30 - days_ago) / 30.0
        mem["relevance_score"] = mem.get("relevance_score", 0) + (recency_factor * 5.0)
    
    return memories

async def apply_emotional_bias(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply emotional bias to memories.
    """
    for mem in memories:
        ei = mem.get("emotional_intensity", 0) / 100.0
        significance = mem.get("significance", 0) / 10.0
        base_score = ei * 3.0 + significance * 2.0
        
        mem["relevance_score"] = mem.get("relevance_score", 0) + base_score
        
        # Example: Additional boost for power-related tags
        tags = mem.get("tags", [])
        power_tags = [
            "dominance_dynamic", "power_exchange", "discipline",
            "service", "submission", "humiliation", "ownership"
        ]
        if any(t in power_tags for t in tags):
            mem["relevance_score"] += 2.0
    
    return memories

async def apply_personality_bias(
    ctx: RunContextWrapper[DecisionContext],
    memories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply personality bias to memories.
    """
    npc_data = await get_npc_data(ctx)
    personality_type = "neutral"
    
    if npc_data.dominance > 70:
        personality_type = "dominant"
    elif npc_data.dominance < 30:
        personality_type = "submissive"
    
    if npc_data.cruelty > 70:
        personality_type = "cruel"
    
    for mem in memories:
        text = mem.get("text", "").lower()
        tags = mem.get("tags", [])
        
        if personality_type == "dominant":
            if any(t in tags for t in ["dominance_dynamic", "control", "discipline"]):
                mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
            if "power_exchange" in tags:
                mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
        
        elif personality_type == "submissive":
            if any(t in tags for t in ["submission", "service", "obedience"]):
                mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
            if "ownership" in tags:
                mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
        
        elif personality_type == "cruel":
            if any(k in text for k in ["hurt", "pain", "suffer", "punish"]):
                mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
        
        # Weighted by memory confidence
        conf = mem.get("confidence", 1.0)
        mem["relevance_score"] = mem.get("relevance_score", 0) * conf
    
    return memories

def score_personality_alignment(
    npc_data: NPCStats,
    action: NPCAction,
    mask_integrity: float,
    hidden_traits: Dict[str, Any],
    presented_traits: Dict[str, Any]
) -> float:
    """
    Score how well an action aligns with the NPC's personality, accounting for mask.
    """
    score = 0.0
    action_type = action.type
    
    true_nature_weight = (100 - mask_integrity) / 100  # 0-1 range
    presented_weight = mask_integrity / 100  # 0-1 range
    
    dominance = npc_data.dominance
    cruelty = npc_data.cruelty
    
    # Hidden traits alignment
    if "dominant" in hidden_traits and action_type in ["command", "dominate", "test", "punish"]:
        score += 3.0 * true_nature_weight
    if "cruel" in hidden_traits and action_type in ["mock", "humiliate", "punish"]:
        score += 3.0 * true_nature_weight
    if "sadistic" in hidden_traits and action_type in ["punish", "humiliate", "mock"]:
        score += 3.0 * true_nature_weight
    
    # Presented traits alignment
    if "kind" in presented_traits and action_type in ["praise", "support", "help"]:
        score += 2.0 * presented_weight
    if "gentle" in presented_traits and action_type in ["talk", "observe", "support"]:
        score += 2.0 * presented_weight
    
    # Base stats
    if dominance > 70 and action_type in ["command", "dominate", "test"]:
        score += 2.0
    if dominance < 30 and action_type in ["observe", "wait", "leave"]:
        score += 1.5
    if cruelty > 70 and action_type in ["mock", "humiliate", "punish"]:
        score += 2.0
    
    return score

async def score_memory_influence(
    memories: List[Dict[str, Any]],
    action: NPCAction
) -> float:
    """
    Score how memories influence action preference with weighting.
    """
    score = 0.0
    if not memories:
        return score
    
    affected_by_memories = []
    action_type = action.type
    
    for i, memory in enumerate(memories):
        memory_text = memory.get("text", "").lower()
        memory_id = memory.get("id")
        emotional_intensity = memory.get("emotional_intensity", 0) / 100.0
        relevance = memory.get("relevance_score", 1.0)
        
        # If action references a memory by ID
        if action.decision_metadata and action.decision_metadata.get("memory_id") == memory_id:
            memory_score = 5 * relevance
            score += memory_score
            affected_by_memories.append({"id": memory_id, "influence": memory_score})
        
        # Check content for direct mention
        if action_type in memory_text:
            memory_score = 2 * relevance
            score += memory_score
            affected_by_memories.append({"id": memory_id, "influence": memory_score})
        
        if action.target:
            target_str = action.target
            if target_str in memory_text:
                memory_score = 3 * relevance
                score += memory_score
                affected_by_memories.append({"id": memory_id, "influence": memory_score})
        
        # Check memory emotion alignment
        memory_emotion = memory.get("primary_emotion", "").lower()
        emotion_aligned_actions = {
            "anger": ["express_anger", "mock", "attack", "challenge"],
            "fear": ["leave", "observe", "act_defensive"],
            "joy": ["celebrate", "praise", "talk", "confide"],
            "sadness": ["observe", "leave"],
            "disgust": ["mock", "leave", "observe"],
        }
        if memory_emotion in emotion_aligned_actions:
            if action_type in emotion_aligned_actions[memory_emotion]:
                emotion_score = 2 * relevance * emotional_intensity
                score += emotion_score
                affected_by_memories.append({
                    "id": memory_id, 
                    "influence_type": "emotion", 
                    "influence": emotion_score
                })
    
    if affected_by_memories:
        decision_metadata = action.decision_metadata or {}
        decision_metadata["memory_influences"] = affected_by_memories
        action.decision_metadata = decision_metadata
    
    return score

def score_relationship_influence(
    relationships: Dict[str, Any],
    action: NPCAction
) -> float:
    """
    Score how relationships influence action preference.
    """
    score = 0.0
    action_type = action.type
    
    # Example: if relationship with player is high trust,
    # penalize or reward certain behaviors
    player_rel = relationships.get("player", {})
    link_level = player_rel.get("link_level", 0)
    
    if action_type in ["punish", "mock", "humiliate"]:
        if link_level > 60:
            score -= 3.0
    elif action_type in ["confide", "talk", "praise"]:
        if link_level > 60:
            score += 2.0
    
    return score

def score_environmental_context(
    environment: Dict[str, Any],
    action: NPCAction
) -> float:
    """
    Score how well the action fits the environment.
    """
    score = 0.0
    loc_str = environment.get("location", "").lower()
    action_type = action.type
    
    # Example: library/church => discourage loud or violent
    if "library" in loc_str or "church" in loc_str:
        if action_type in ["express_anger", "mock", "shout", "celebrate", "dominate"]:
            score -= 2.0
    
    if "bar" in loc_str or "party" in loc_str:
        if action_type in ["socialize", "talk", "celebrate"]:
            score += 2.0
    
    return score

def score_emotional_influence(
    emotion: str,
    intensity: float,
    action: NPCAction
) -> float:
    """
    Score based on NPC's current emotional state with psychological realism.
    """
    score = 0.0
    if intensity < 0.4:
        return 0.0
    
    emotion_action_affinities = {
        "anger": {
            "express_anger": 4, "command": 2, "mock": 3, "test": 2, "leave": 1,
            "punish": 4, "humiliate": 3, "dominate": 3,
            "praise": -3, "confide": -2, "socialize": -1
        },
        "fear": {
            "act_defensive": 4, "observe": 3, "leave": 2,
            "command": -2, "confide": -3, "socialize": -2, "dominate": -3
        },
        "joy": {
            "celebrate": 4, "talk": 3, "praise": 3, "socialize": 3, "confide": 2,
            "reward_submission": 3,
            "mock": -3, "leave": -2, "act_defensive": -1, "punish": -2
        },
        "sadness": {
            "observe": 3, "leave": 2, "confide": 1,
            "celebrate": -3, "socialize": -2, "talk": -1, "dominate": -2
        },
        "disgust": {
            "mock": 3, "leave": 2, "act_defensive": 1, "humiliate": 3,
            "praise": -3, "confide": -2, "talk": -1
        },
        "trust": {
            "confide": 3, "praise": 2, "talk": 2,
            "reward_submission": 2,
            "mock": -3, "act_defensive": -2
        }
    }
    
    action_type = action.type
    action_affinities = emotion_action_affinities.get(emotion, {})
    if action_type in action_affinities:
        affinity = action_affinities[action_type]
        score += affinity * intensity
    
    decision_metadata = action.decision_metadata or {}
    decision_metadata["emotional_influence"] = {
        "emotion": emotion,
        "intensity": intensity,
        "affinity": action_affinities.get(action_type, 0),
        "score": score
    }
    action.decision_metadata = decision_metadata
    
    return score

def score_mask_influence(
    mask_integrity: float,
    npc_data: NPCStats,
    action: NPCAction,
    hidden_traits: Dict[str, Any],
    presented_traits: Dict[str, Any]
) -> float:
    """
    Score based on mask integrity - as mask deteriorates, true nature shows more.
    """
    score = 0.0
    if mask_integrity >= 95:
        return 0.0
    
    true_nature_factor = (100 - mask_integrity) / 100
    mask_influences = []
    action_type = action.type
    
    # Hidden traits
    # Convert hidden_traits to a list if it's a dict
    hidden_trait_names = []
    if isinstance(hidden_traits, dict):
        hidden_trait_names = list(hidden_traits.keys())
    elif isinstance(hidden_traits, list):
        hidden_trait_names = hidden_traits
    
    for trait in hidden_trait_names:
        trait_weight = true_nature_factor
        trait_score = 0
        
        if trait == "dominant" and action_type in ["command", "test", "dominate", "punish"]:
            trait_score = 5 * trait_weight
        elif trait == "cruel" and action_type in ["mock", "humiliate", "punish"]:
            trait_score = 5 * trait_weight
        elif trait == "sadistic" and action_type in ["punish", "humiliate", "mock"]:
            trait_score = 5 * trait_weight
        
        if trait_score != 0:
            score += trait_score
            mask_influences.append({"trait": trait, "type": "hidden", "score": trait_score})
    
    # Presented traits conflicts - same conversion logic
    presented_trait_names = []
    if isinstance(presented_traits, dict):
        presented_trait_names = list(presented_traits.keys())
    elif isinstance(presented_traits, list):
        presented_trait_names = presented_traits
    
    for trait in presented_trait_names:
        mask_factor = 1.0 - true_nature_factor
        trait_score = 0
        
        if trait == "kind" and action_type in ["mock", "humiliate", "punish"]:
            trait_score = -3 * mask_factor
        elif trait == "gentle" and action_type in ["dominate", "express_anger", "punish"]:
            trait_score = -3 * mask_factor
        elif trait == "submissive" and action_type in ["command", "dominate", "direct"]:
            trait_score = -4 * mask_factor
        elif trait == "honest" and action_type in ["deceive", "manipulate", "lie"]:
            trait_score = -4 * mask_factor
        
        if trait_score != 0:
            score += trait_score
            mask_influences.append({"trait": trait, "type": "presented", "score": trait_score})
    
    if mask_influences:
        decision_metadata = action.decision_metadata or {}
        decision_metadata["mask_influence"] = {
            "integrity": mask_integrity,
            "true_nature_factor": true_nature_factor,
            "trait_influences": mask_influences
        }
        action.decision_metadata = decision_metadata
    
    return score

def score_trauma_influence(
    action: NPCAction,
    flashback: Optional[Dict[str, Any]],
    traumatic_trigger: Optional[Dict[str, Any]]
) -> float:
    """
    Score actions based on flashbacks or traumatic triggers.
    """
    score = 0.0
    if not flashback and not traumatic_trigger:
        return score
    
    action_type = action.type
    trauma_action_map = {
        "traumatic_response": 5.0,
        "act_defensive": 4.0,
        "leave": 3.5,
        "express_anger": 3.0,
        "observe": 2.5,
    }
    
    if action_type in trauma_action_map:
        base_score = trauma_action_map[action_type]
        if flashback and not traumatic_trigger:
            score += base_score * 0.7
        elif traumatic_trigger:
            score += base_score
            response_type = traumatic_trigger.get("response_type")
            if response_type == "fight" and action_type in ["express_anger", "challenge"]:
                score += 2.0
            elif response_type == "flight" and action_type == "leave":
                score += 2.0
            elif response_type == "freeze" and action_type == "observe":
                score += 2.0
    
    # Penalize vulnerability actions
    vulnerability_actions = ["confide", "praise", "talk", "socialize"]
    if action_type in vulnerability_actions:
        if traumatic_trigger:
            score -= 3.0
        elif flashback:
            score -= 2.0
    
    if flashback or traumatic_trigger:
        decision_metadata = action.decision_metadata or {}
        decision_metadata["trauma_influence"] = {
            "has_flashback": bool(flashback),
            "has_trigger": bool(traumatic_trigger),
            "score": score
        }
        action.decision_metadata = decision_metadata
    
    return score

def score_belief_influence(
    beliefs: List[Dict[str, Any]],
    action: NPCAction
) -> float:
    """
    Score actions based on how well they align with NPC's beliefs.
    """
    score = 0.0
    if not beliefs:
        return score
    
    action_type = action.type
    target = action.target
    belief_influences = []
    
    for belief in beliefs:
        belief_text = belief.get("belief", "").lower()
        confidence = belief.get("confidence", 0.5)
        if confidence < 0.3:
            continue
        
        relevance = 0.0
        align_score = 0.0
        
        if (target == "player" or target == "group"):
            if any(word in belief_text for word in ["trust", "friend", "ally", "like"]):
                if action_type in ["talk", "praise", "confide", "support"]:
                    relevance = 0.8
                    align_score = 3.0
                elif action_type in ["mock", "challenge", "leave", "punish"]:
                    relevance = 0.7
                    align_score = -3.0
            elif any(word in belief_text for word in ["threat", "danger", "distrust", "wary"]):
                if action_type in ["observe", "act_defensive", "leave"]:
                    relevance = 0.9
                    align_score = 4.0
                elif action_type in ["confide", "praise", "support"]:
                    relevance = 0.8
                    align_score = -4.0
            elif any(word in belief_text for word in ["submit", "obey", "follow"]):
                if action_type in ["command", "test", "dominate"]:
                    relevance = 0.9
                    align_score = 3.5
                elif action_type in ["observe", "act_defensive"]:
                    relevance = 0.5
                    align_score = -2.0
            elif any(word in belief_text for word in ["rebel", "defy", "disobey"]):
                if action_type in ["punish", "test", "command"]:
                    relevance = 0.8
                    align_score = 3.0
                elif action_type in ["praise", "reward"]:
                    relevance = 0.6
                    align_score = -2.5
            
            if action_type in belief_text:
                relevance = 0.9
                align_score = 4.0
            
            if relevance > 0:
                belief_score = align_score * confidence * relevance
                score += belief_score
                belief_influences.append({
                    "text": belief_text[:50] + "..." if len(belief_text) > 50 else belief_text,
                    "confidence": confidence,
                    "relevance": relevance,
                    "align_score": align_score,
                    "final_score": belief_score
                })
    
    if belief_influences:
        decision_metadata = action.decision_metadata or {}
        decision_metadata["belief_influences"] = belief_influences
        action.decision_metadata = decision_metadata
    
    return score

def score_decision_history(
    ctx: RunContextWrapper[DecisionContext],
    action: NPCAction
) -> float:
    """
    Score actions based on decision history for psychological continuity.
    """
    score = 0.0
    if not ctx.context.decision_log:
        return score
    
    action_type = action.type
    
    # Get recent actions from the decision log
    recent_actions = []
    for decision in ctx.context.decision_log[-3:]:
        if "action" in decision:
            recent_action = decision["action"]
            if isinstance(recent_action, dict) and "type" in recent_action:
                recent_actions.append(recent_action["type"])
    
    if not recent_actions:
        return score
    
    action_counts = {}
    for i, a_type in enumerate(recent_actions):
        weight = 1.0 - (i * 0.2)
        action_counts[a_type] = action_counts.get(a_type, 0) + weight
    
    if action_type in action_counts:
        consistency_score = action_counts[action_type] * 1.5
        # Penalize repetitiveness if last two actions are the same
        if len(recent_actions) >= 2 and recent_actions[0] == recent_actions[1] == action_type:
            consistency_score -= 3.0
        score += consistency_score
    
    # Encourage variety
    if len(recent_actions) >= 2:
        if len(set(recent_actions)) == len(recent_actions) and action_type not in recent_actions:
            score -= 1.0
    
    decision_metadata = action.decision_metadata or {}
    decision_metadata["history_influence"] = {
        "recent_actions": recent_actions,
        "action_counts": action_counts,
        "score": score
    }
    action.decision_metadata = decision_metadata
    
    return score

def score_player_knowledge_influence(
    action: NPCAction,
    player_knowledge: float,
    hidden_traits: Dict[str, Any]
) -> float:
    """
    Score actions based on how much the player knows about the NPC's true nature.
    """
    score = 0.0
    if player_knowledge < 0.3:
        return score
    
    action_type = action.type
    
    # Convert hidden_traits to a list if it's a dict
    hidden_trait_names = []
    if isinstance(hidden_traits, dict):
        hidden_trait_names = list(hidden_traits.keys())
    elif isinstance(hidden_traits, list):
        hidden_trait_names = hidden_traits
    
    if player_knowledge > 0.7:
        if "dominant" in hidden_trait_names and action_type in ["command", "dominate", "punish"]:
            score += 2.0
        elif "cruel" in hidden_trait_names and action_type in ["mock", "humiliate"]:
            score += 2.0
        elif "submissive" in hidden_trait_names and action_type in ["observe", "act_defensive"]:
            score += 2.0
    elif player_knowledge > 0.4:
        # A bit of randomness
        if random.random() < 0.5:
            if "dominant" in hidden_trait_names and action_type in ["command", "direct"]:
                score += 1.5
            elif "cruel" in hidden_trait_names and action_type == "mock":
                score += 1.5
        else:
            if "dominant" in hidden_trait_names and action_type in ["talk", "observe"]:
                score += 1.0
    
    decision_metadata = action.decision_metadata or {}
    decision_metadata["player_knowledge_influence"] = {
        "knowledge_level": player_knowledge,
        "score": score
    }
    action.decision_metadata = decision_metadata
    
    return score

def calculate_player_knowledge(
    perception: NPCPerception,
    mask_integrity: float
) -> float:
    """
    Calculate how much the player knows about the NPC's true nature.
    Based on relationship level, memory count, and mask integrity.
    """
    player_knowledge = 0.0
    if mask_integrity < 50:
        player_knowledge += 0.4
    elif mask_integrity < 75:
        player_knowledge += 0.2
    
    memories = perception.relevant_memories
    if len(memories) > 7:
        player_knowledge += 0.3
    elif len(memories) > 3:
        player_knowledge += 0.15
    
    relationships = perception.relationships
    player_rel = relationships.get("player", {})
    link_level = player_rel.get("link_level", 0)
    if link_level > 70:
        player_knowledge += 0.3
    elif link_level > 40:
        player_knowledge += 0.15
    
    return min(1.0, max(0.0, player_knowledge))

def score_time_context_influence(
    time_context: Dict[str, Any],
    action: NPCAction
) -> float:
    """
    Score actions based on time-of-day appropriateness.
    """
    score = 0.0
    time_of_day = time_context.get("time_of_day", "").lower()
    action_type = action.type
    
    if time_of_day == "morning":
        if action_type in ["talk", "observe", "socialize"]:
            score += 1.0
        elif action_type in ["sleep", "seduce", "dominate"]:
            score -= 1.0
    elif time_of_day == "afternoon":
        if action_type in ["talk", "socialize", "command", "test"]:
            score += 1.0
    elif time_of_day == "evening":
        if action_type in ["talk", "socialize", "seduce", "flirt"]:
            score += 1.5
    elif time_of_day == "night":
        if action_type in ["seduce", "dominate", "sleep"]:
            score += 2.0
        elif action_type in ["talk", "socialize"]:
            score -= 0.5
    
    return score

def score_behavior_evolution_influence(
    behavior_factors: BehaviorEvolFactors,
    action: NPCAction
) -> float:
    """
    Score actions based on behavior evolution factors.
    """
    score = 0.0
    
    # Scheming influence
    if behavior_factors.scheme_level >= 5:
        scheming_scores = {
            "gather_info": 3.0,
            "manipulate": 4.0,
            "test_loyalty": 2.5,
            "probe_weaknesses": 3.0,
            "observe": 1.5,
            "confide": -1.5,  # Less likely to confide if scheming
            "praise": -1.0    # Less likely to praise genuinely
        }
        
        if action.type in scheming_scores:
            factor = min(1.0, behavior_factors.scheme_level / 10.0)  # 0.0-1.0 scaling
            score += scheming_scores[action.type] * factor
    
    # Paranoia influence
    if behavior_factors.paranoia_level >= 4:
        paranoia_scores = {
            "observe": 2.0,
            "act_defensive": 2.5,
            "test_loyalty": 2.0,
            "leave": 1.5,
            "confide": -2.0,
            "praise": -1.0
        }
        
        if action.type in paranoia_scores:
            factor = min(1.0, behavior_factors.paranoia_level / 10.0)  # 0.0-1.0 scaling
            score += paranoia_scores[action.type] * factor
    
    # Targeting player influence
    if behavior_factors.targeting_player and action.target == "player":
        targeting_scores = {
            "probe_weaknesses": 3.0,
            "manipulate": 3.0,
            "gather_info": 2.0,
            "confide": -2.0
        }
        
        if action.type in targeting_scores:
            score += targeting_scores[action.type]
    
    # Betrayal planning influence
    if behavior_factors.betrayal_planning:
        betrayal_scores = {
            "deceive": 3.0,
            "manipulate": 2.5,
            "lie": 2.0,
            "talk": 1.0,  # Can be used to set up betrayal
            "confide": -3.0,
            "reveal_secrets": -2.0
        }
        
        if action.type in betrayal_scores:
            score += betrayal_scores[action.type]
    
    return score

def log_decision_reasoning(
    ctx: RunContextWrapper[DecisionContext],
    top_actions: List[ScoredAction]
) -> None:
    """
    Log decision reasoning for debugging.
    """
    if not top_actions:
        return
    
    now = datetime.now()
    reasoning_entry = {
        "timestamp": now.isoformat(),
        "top_actions": [],
        "chosen_action": top_actions[0].action.type
    }
    
    for i, action_data in enumerate(top_actions[:3]):
        action = action_data.action
        score = action_data.score
        reasoning = action_data.reasoning
        
        action_entry = {
            "rank": i + 1,
            "type": action.type,
            "description": action.description,
            "score": score,
            "reasoning_factors": reasoning
        }
        reasoning_entry["top_actions"].append(action_entry)
    
    logger.debug(f"NPC {ctx.context.npc_id} decision reasoning recorded")

# -------------------------------------------------------
# Main Decision Engine Agent with Integrated Behavior Evolution
# -------------------------------------------------------

decision_engine_agent = Agent(
    name="NPCDecisionEngine",
    instructions="""
    You are a decision engine for non-player characters (NPCs) in an interactive narrative simulation.
    
    Your job is to select the most appropriate action for an NPC based on:
    1. The NPC's personality traits, stats, and long-term goals
    2. The NPC's current perception (environment, memories, relationships, emotions)
    3. The NPC's mask system (presented traits vs. hidden traits)
    4. The NPC's behavioral evolution factors (scheming, paranoia, etc.)
    5. Psychological realism and consistency
    
    For each decision, follow this process:
    1. Generate possible actions, considering both basic actions and complex ones
    2. Score each action based on multiple factors:
       - Personality alignment
       - Memory influence
       - Relationship context
       - Environmental factors
       - Emotional state
       - Mask influence
       - Behavior evolution influence
    3. Select the best action, incorporating some randomness for realism
    
    Consider these key psychological elements:
    - The NPC's dominance and cruelty levels
    - Current emotional state and its influence on decision-making
    - Memories that might trigger specific behaviors
    - The integrity of their "mask" (how well they're hiding their true nature)
    - Consistency with previous decisions for psychological continuity
    - The NPC's evolving behavior patterns and schemes
    
    Use the available tools to generate actions, score them, and select the most appropriate one.
    Provide the chosen action in a structured format with explanation of the reasoning.
    """,
    tools=[
        get_npc_data,
        get_default_actions,
        generate_memory_based_actions,
        score_actions,
        select_action,
        generate_flashback_action,
        enhance_dominance_context,
        update_behavior_evolution
    ],
    output_type=NPCAction
)

# -------------------------------------------------------
# Decision Engine class
# -------------------------------------------------------

class NPCDecisionEngine:
    """
    Decision-making engine for NPCs with integrated behavior evolution.
    """

    @classmethod
    async def create(cls, npc_id: int, user_id: int, conversation_id: int) -> "NPCDecisionEngine":
        """Async factory method to create an NPCDecisionEngine instance."""
        self = cls(npc_id, user_id, conversation_id)
        await self.initialize() # Call the async initializer
        return self

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = DecisionContext(npc_id, user_id, conversation_id)
        # BehaviorEvolution instance is created, but DB calls happen in its methods
        self.behavior_evolution = BehaviorEvolution(user_id, conversation_id)

    # --- Updated to be async ---
    async def initialize(self) -> None:
        """Async initialization: Fetches initial NPC data and evolution factors."""
        logger.info(f"Initializing NPCDecisionEngine for NPC {self.npc_id}")
        try:
            # Use a wrapper for the context needed by the tool
            ctx_wrapper = RunContextWrapper(self.context)
            npc_data_model = await get_npc_data(ctx_wrapper) # Fetch initial data (hits DB)

            # Initialize long-term goals based on fetched data
            # Ensure npc_data_model is converted to dict if needed by initialize_long_term_goals
            await self.initialize_long_term_goals(npc_data_model.model_dump())

            # Initialize behavior evolution factors (hits DB)
            await self.update_behavior_evolution_context() # Renamed method for clarity

            logger.info(f"NPCDecisionEngine initialized successfully for NPC {self.npc_id}")

        except Exception as e:
            logger.critical(f"FATAL: Error initializing NPCDecisionEngine for NPC {self.npc_id}: {e}", exc_info=True)
            # Depending on requirements, maybe raise this error or handle gracefully
            # For now, logging critical and continuing might leave engine in bad state.

    # --- Updated to be async ---
    async def decide(
        self,
        perception: Dict[str, Any], # Expects a dict, ideally matching NPCPerception schema
        available_actions: Optional[List[dict]] = None # Optional input
    ) -> Dict[str, Any]:
        """
        Asynchronously decide the NPC's next action.
        """
        with custom_span("NPCDecisionEngine.decide", attributes={"npc_id": self.npc_id}):
            logger.debug(f"Starting decision cycle for NPC {self.npc_id}")
            # Store perception in context (ensure it's a dict)
            try:
                # Validate perception if possible, or just store
                 self.context.perception = NPCPerception(**perception).model_dump() # Validate & store dict
            except Exception as p_err:
                 logger.error(f"Invalid perception data for NPC {self.npc_id}: {p_err}. Using raw dict.", exc_info=True)
                 self.context.perception = perception # Store raw dict if validation fails

            # Optional: Update behavior factors every decision cycle? Or less frequently?
            # If less frequently, move this call elsewhere (e.g., periodic background task)
            # await self.update_behavior_evolution_context()

            # --- Agent Runner Call ---
            # The agent runner executes the tools (get_npc_data, score_actions etc.) which handle DB access
            try:
                 result = await Runner.run(
                     decision_engine_agent,
                     f"Make a decision for NPC {self.npc_id} based on perception and context.",
                     context=self.context # Pass the DecisionContext object
                 )

                 chosen_action = result.final_output

                 # Convert back to dict for external usage
                 if isinstance(chosen_action, NPCAction):
                     logger.info(f"NPC {self.npc_id} chose action: {chosen_action.type}")
                     return chosen_action.model_dump()
                 elif isinstance(chosen_action, dict):
                      logger.info(f"NPC {self.npc_id} chose action (dict): {chosen_action.get('type', 'Unknown')}")
                      return chosen_action # Assume it's already a dict
                 else:
                      logger.error(f"Decision engine for NPC {self.npc_id} returned unexpected type: {type(chosen_action)}. Returning default idle action.")
                      # Fallback action
                      return NPCAction(type="idle", description="Engine error resulted in idle", target="self").model_dump()

            except Exception as agent_err:
                logger.error(f"Error running decision engine agent for NPC {self.npc_id}: {agent_err}", exc_info=True)
                # Fallback action on agent error
                return NPCAction(type="idle", description="Agent execution error", target="self").model_dump()


    # --- Updated to be async, takes dict ---
    async def initialize_long_term_goals(self, npc_data_dict: Dict[str, Any]) -> None:
        """
        Initialize NPC's long-term goals based on personality dict. (No DB calls here)
        """
        with function_span("initialize_long_term_goals", attributes={"npc_id": self.npc_id}):
            self.context.long_term_goals = [] # Reset goals

            # Use .get() for safe access
            dominance = npc_data_dict.get("dominance", 50.0)
            cruelty = npc_data_dict.get("cruelty", 50.0)
            traits = npc_data_dict.get("personality_traits", [])
            # Assuming archetypes might be added later or come from elsewhere
            archetypes = npc_data_dict.get("archetypes", [])

            # --- Goal definition logic (remains the same) ---
            if dominance > 75:
                self.context.long_term_goals.append({
                    "type": "dominance",
                    "description": "Assert complete control over submissives",
                    "importance": 0.9,
                    "progress": 0,
                    "target_entity": "player"
                })
            elif dominance > 60:
                self.context.long_term_goals.append({
                    "type": "dominance",
                    "description": "Establish authority in social hierarchy",
                    "importance": 0.8,
                    "progress": 0,
                    "target_entity": None
                })
            elif dominance < 30:
                self.context.long_term_goals.append({
                    "type": "submission",
                    "description": "Find strong dominant to serve",
                    "importance": 0.8,
                    "progress": 0,
                    "target_entity": None
                })
            
            # Create goals based on cruelty
            if cruelty > 70:
                self.context.long_term_goals.append({
                    "type": "sadism",
                    "description": "Break down resistances through humiliation",
                    "importance": 0.85,
                    "progress": 0,
                    "target_entity": "player"
                })
            elif cruelty < 30 and dominance > 60:
                self.context.long_term_goals.append({
                    "type": "guidance",
                    "description": "Guide submissives to growth through guidance",
                    "importance": 0.75,
                    "progress": 0,
                    "target_entity": "player"
                })
            
            # Personality traits
            if "ambitious" in traits:
                self.context.long_term_goals.append({
                    "type": "power",
                    "description": "Increase social influence and control",
                    "importance": 0.85,
                    "progress": 0,
                    "target_entity": None
                })
            if "protective" in traits:
                self.context.long_term_goals.append({
                    "type": "protection",
                    "description": "Ensure the safety and well-being of those in care",
                    "importance": 0.8,
                    "progress": 0,
                    "target_entity": "player" if dominance > 50 else None
                })
            
            # Archetypes
            if isinstance(archetypes, list):
                for arch in archetypes:
                    arch_name = arch if isinstance(arch, str) else arch.get("name", "") if isinstance(arch, dict) else ""
                    
                    if "mentor" in arch_name.lower():
                        self.context.long_term_goals.append({
                            "type": "development",
                            "description": "Guide the development of the player",
                            "importance": 0.9,
                            "progress": 0,
                            "target_entity": "player"
                        })
                    elif "seductress" in arch_name.lower():
                        self.context.long_term_goals.append({
                            "type": "seduction",
                            "description": "Gradually increase player's dependency and devotion",
                            "importance": 0.9,
                            "progress": 0,
                            "target_entity": "player"
                        })

            logger.debug(f"Initialized {len(self.context.long_term_goals)} long-term goals for NPC {self.npc_id}")


    # --- Renamed and uses BehaviorEvolution methods ---
    async def update_behavior_evolution_context(self) -> None:
        """
        Update the NPC's behavior evolution factors in the context by calling BehaviorEvolution.
        This method handles the DB interactions via the BehaviorEvolution class.
        """
        with function_span("update_behavior_evolution_context", attributes={"npc_id": self.npc_id}):
            logger.debug(f"Updating behavior evolution factors for NPC {self.npc_id}")
            # evaluate_npc_scheming fetches data and calculates adjustments
            adjustments = await self.behavior_evolution.evaluate_npc_scheming(self.npc_id)

            if "error" not in adjustments:
                # apply_scheming_adjustments updates the database
                await self.behavior_evolution.apply_scheming_adjustments(self.npc_id, adjustments)

                # Update the context object with the results
                self.context.behavior_evolution_factors = BehaviorEvolFactors(
                    scheme_level=adjustments.get("scheme_level", 0),
                    trust_modifiers=adjustments.get("trust_modifiers", {}),
                    loyalty_tests=adjustments.get("loyalty_tests", 0),
                    betrayal_planning=adjustments.get("betrayal_planning", False),
                    targeting_player=adjustments.get("targeting_player", False),
                    npc_recruits=adjustments.get("npc_recruits", []),
                    paranoia_level=adjustments.get("paranoia_level", 0),
                    adaptation_score=adjustments.get("adaptation_score", 0.0)
                )
                logger.debug(f"Behavior evolution context updated for NPC {self.npc_id}")
            else:
                logger.error(f"Failed to update behavior evolution context for NPC {self.npc_id} due to evaluation error: {adjustments['error']}")
                # Optionally clear or set default factors in context on error?
                # self.context.behavior_evolution_factors = BehaviorEvolFactors() # Reset on error?
