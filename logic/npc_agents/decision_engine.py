# npc_agents/decision_engine.py

"""
Decision-making engine for NPCs using OpenAI Agents SDK.
Replaces the original decision_engine.py with the Agent SDK architecture.
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.tracing import custom_span, function_span
from db.connection import get_db_connection

# Import memory subsystem components
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

# -------------------------------------------------------
# Context class for the decision engine
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
        npc_data: E.g. your NPCStats dict (dominance, cruelty, npc_name, etc.).
        perception: The dictionary from perceive_environment or your existing logic 
                    (with emotional_state, relevant_memories, relationships, etc.)
        top_n: Max number of actions to keep from GPT output

    Returns:
        A list of action dicts in your standard format:
        [
          {
            "type": "mock",
            "description": "Mock the player harshly",
            "target": "player",
            "stats_influenced": {"cruelty": 2}
          },
          ...
        ]
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
# Tool Functions
# -------------------------------------------------------

@function_tool
async def get_npc_data(ctx: RunContextWrapper[DecisionContext]) -> NPCStats:
    """
    Get the NPC's stats and traits from the database.
    """
    with function_span("get_npc_data"):
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        # If cached stats available, return them
        if ctx.context.npc_stats:
            return NPCStats(**ctx.context.npc_stats)
        
        def _fetch():
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                           hobbies, personality_traits, likes, dislikes, schedule, current_location, sex
                    FROM NPCStats
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """,
                    (npc_id, user_id, conversation_id),
                )
                return cursor.fetchone()

        row = await asyncio.to_thread(_fetch)
        if not row:
            default_stats = NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}")
            ctx.context.npc_stats = default_stats.model_dump()
            return default_stats

        # Parse JSON fields
        def _parse_json_field(field):
            if field is None:
                return []
            if isinstance(field, str):
                try:
                    return json.loads(field)
                except json.JSONDecodeError:
                    return []
            if isinstance(field, list):
                return field
            return []

        hobbies = _parse_json_field(row[7])
        personality_traits = _parse_json_field(row[8])
        likes = _parse_json_field(row[9])
        dislikes = _parse_json_field(row[10])
        schedule = _parse_json_field(row[11])

        stats = NPCStats(
            npc_id=npc_id,
            npc_name=row[0],
            dominance=row[1],
            cruelty=row[2],
            closeness=row[3],
            trust=row[4],
            respect=row[5],
            intensity=row[6],
            hobbies=hobbies,
            personality_traits=personality_traits,
            likes=likes,
            dislikes=dislikes,
            schedule=schedule,
            current_location=row[12],
            sex=row[13]
        )
        
        # Cache the stats
        ctx.context.npc_stats = stats.model_dump()
        return stats

@function_tool
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
        
        return actions

@function_tool
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

@function_tool
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

@function_tool
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

@function_tool
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

@function_tool
async def enhance_dominance_context(
    ctx: RunContextWrapper[DecisionContext],
    action: NPCAction,
    npc_data: NPCStats,
    mask: Dict[str, Any]
) -> NPCAction:
    """
    Enhance actions with dominance context for femdom gameplay.
    
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

@function_tool
async def maybe_create_belief(
    ctx: RunContextWrapper[DecisionContext],
    perception: NPCPerception,
    action: NPCAction,
    npc_data: NPCStats
) -> Dict[str, Any]:
    """
    Potentially create a belief based on a decision.
    
    Args:
        perception: The NPC's current perception
        action: The chosen action
        npc_data: The NPC's stats and traits
    """
    with function_span("maybe_create_belief"):
        result = {"belief_created": False}
        
        if random.random() > 0.05:
            return result
        
        memory_system = await ctx.context.get_memory_system()
        memories = perception.relevant_memories
        supporting_memory_ids = [m.get("id") for m in memories if m.get("id")]
        
        potential_beliefs = []
        # 1) Beliefs about player submission/resistance
        if "resistance" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player tends to resist my commands",
                "confidence": 0.7 if len(supporting_memory_ids) > 1 else 0.5
            })
        elif "submission" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player is submissive to my authority",
                "confidence": 0.7 if len(supporting_memory_ids) > 1 else 0.5
            })
        
        # 2) Based on chosen action
        if action.type in ["dominate", "punish", "command"] and npc_data.dominance > 70:
            potential_beliefs.append({
                "text": "I need to maintain strict control over the player",
                "confidence": 0.8
            })
        if action.type in ["reward_submission", "praise"] and "submission" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player responds well to praise for their submission",
                "confidence": 0.7
            })
        
        # 3) Based on emotional state
        emotional_state = perception.emotional_state
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        if primary_emotion == "anger" and action.type in ["punish", "express_anger"]:
            potential_beliefs.append({
                "text": "When I show my anger, the player becomes more compliant",
                "confidence": 0.6
            })
        
        if potential_beliefs:
            belief = random.choice(potential_beliefs)
            try:
                belief_result = await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    belief_text=belief["text"],
                    confidence=belief["confidence"]
                )
                result = {
                    "belief_created": True,
                    "belief_text": belief["text"],
                    "confidence": belief["confidence"],
                    "belief_id": belief_result.get("belief_id")
                }
            except Exception as e:
                logger.error(f"Error creating belief: {e}")
        
        return result

@function_tool
async def update_goal_progress(
    ctx: RunContextWrapper[DecisionContext],
    action: NPCAction,
    outcome: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update progress on long-term goals based on action outcomes.
    
    Args:
        action: The executed action
        outcome: The action outcome
    """
    with function_span("update_goal_progress"):
        result = {"goals_updated": 0}
        
        if not ctx.context.long_term_goals:
            return result
        
        success = "success" in outcome.get("result", "").lower()
        emotional_impact = outcome.get("emotional_impact", 0)
        goals_updated = 0
        
        for i, goal in enumerate(ctx.context.long_term_goals):
            goal_type = goal.get("type", "")
            target_entity = goal.get("target_entity")
            current_progress = goal.get("progress", 0)
            
            # Skip if target mismatch
            if (target_entity
                and action.target != target_entity
                and not (target_entity == "player" and action.target == "group")):
                continue
            
            progress_update = 0
            if goal_type == "dominance":
                if action.type in ["command", "dominate", "test"]:
                    if success:
                        progress_update = 5
                    else:
                        progress_update = -2
            elif goal_type == "submission":
                if action.type in ["observe", "obey", "assist"]:
                    progress_update = 3 if success else 1
            elif goal_type == "sadism":
                if action.type in ["punish", "humiliate", "mock"]:
                    progress_update = 2 if success else 0
            elif goal_type == "seduction":
                if action.type in ["seduce", "flirt"]:
                    if emotional_impact > 0:
                        progress_update = emotional_impact
            
            if progress_update != 0:
                ctx.context.long_term_goals[i]["progress"] = max(0, min(100, current_progress + progress_update))
                goals_updated += 1
        
        result["goals_updated"] = goals_updated
        return result

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
        
        # Example: Additional boost for femdom-related tags
        tags = mem.get("tags", [])
        femdom_tags = [
            "dominance_dynamic", "power_exchange", "discipline",
            "service", "submission", "humiliation", "ownership"
        ]
        if any(t in femdom_tags for t in tags):
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
    for trait, trait_data in hidden_traits.items():
        trait_intensity = trait_data.get("intensity", 50) / 100
        trait_weight = trait_intensity * true_nature_factor
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
    
    # Presented traits conflicts
    for trait, trait_data in presented_traits.items():
        trait_confidence = trait_data.get("confidence", 50) / 100
        mask_factor = 1.0 - true_nature_factor
        trait_score = 0
        
        if trait == "kind" and action_type in ["mock", "humiliate", "punish"]:
            trait_score = -3 * trait_confidence * mask_factor
        elif trait == "gentle" and action_type in ["dominate", "express_anger", "punish"]:
            trait_score = -3 * trait_confidence * mask_factor
        elif trait == "submissive" and action_type in ["command", "dominate", "direct"]:
            trait_score = -4 * trait_confidence * mask_factor
        elif trait == "honest" and action_type in ["deceive", "manipulate", "lie"]:
            trait_score = -4 * trait_confidence * mask_factor
        
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
    
    if player_knowledge > 0.7:
        if "dominant" in hidden_traits and action_type in ["command", "dominate", "punish"]:
            score += 2.0
        elif "cruel" in hidden_traits and action_type in ["mock", "humiliate"]:
            score += 2.0
        elif "submissive" in hidden_traits and action_type in ["observe", "act_defensive"]:
            score += 2.0
    elif player_knowledge > 0.4:
        # A bit of randomness
        if random.random() < 0.5:
            if "dominant" in hidden_traits and action_type in ["command", "direct"]:
                score += 1.5
            elif "cruel" in hidden_traits and action_type == "mock":
                score += 1.5
        else:
            if "dominant" in hidden_traits and action_type in ["talk", "observe"]:
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
# Decision Engine Agent
# -------------------------------------------------------

decision_agent = Agent(
    name="NPC Decision Engine",
    instructions="""
    You are a decision engine for non-player characters (NPCs) in an interactive narrative simulation.
    
    Your job is to select the most appropriate action for an NPC based on:
    1. The NPC's personality traits, stats, and long-term goals
    2. The NPC's current perception (environment, memories, relationships, emotions)
    3. The NPC's mask system (presented traits vs. hidden traits)
    4. Psychological realism and consistency
    
    For each decision, follow this process:
    1. Analyze the NPC's stats and current state
    2. Generate possible actions (both hard-coded logic and GPT-based suggestions)
    3. Score each action based on multiple factors
    4. Select the best action, incorporating some randomness for realism
    5. Enhance the chosen action with additional context if needed
    
    Consider these key psychological elements:
    - The NPC's dominance and cruelty levels
    - Current emotional state and its influence on decision-making
    - Memories that might trigger specific behaviors
    - The integrity of their "mask" (how well they're hiding their true nature)
    - Consistency with previous decisions for psychological continuity
    
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
        maybe_create_belief,
        update_goal_progress
    ],
    output_type=NPCAction
)

# -------------------------------------------------------
# Main Decision Engine class using Agents SDK
# -------------------------------------------------------

class DecisionEngineSDK:
    """
    Decision-making engine for NPCs using OpenAI Agents SDK.
    Replaces the original NPCDecisionEngine class.
    """
    
    @classmethod
    async def create(cls, npc_id: int, user_id: int, conversation_id: int) -> "DecisionEngineSDK":
        """
        Recommended async factory method to ensure goals are initialized before usage.
        
        Args:
            npc_id: ID of the NPC
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            An initialized DecisionEngineSDK instance
        """
        self = cls(npc_id, user_id, conversation_id, initialize_goals=False)
        await self._initialize_goals()  # ensure we initialize before returning
        return self
    
    def __init__(
        self, 
        npc_id: int, 
        user_id: int, 
        conversation_id: int, 
        initialize_goals: bool = True
    ):
        """
        Initialize a DecisionEngineSDK instance.
        
        Args:
            npc_id: ID of the NPC
            user_id: User ID
            conversation_id: Conversation ID
            initialize_goals: Whether to initialize goals immediately
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Create context
        self.context = DecisionContext(npc_id, user_id, conversation_id)
        
        # If someone constructs this without the factory, they can still get goals in the background
        if initialize_goals:
            asyncio.create_task(self._initialize_goals())
    
    async def _initialize_goals(self) -> None:
        """Initialize goals after getting NPC data."""
        try:
            ctx_wrapper = RunContextWrapper(self.context)
            npc_data = await get_npc_data(ctx_wrapper)
            
            if npc_data:
                await self.initialize_long_term_goals(npc_data.model_dump())
        except Exception as e:
            logger.error(f"Error initializing goals: {e}")
    
    async def decide(
        self, 
        perception: Dict[str, Any], 
        available_actions: Optional[List[dict]] = None
    ) -> dict:
        """
        Evaluate the NPC's current state, context, personality, memories, and emotional state
        to pick an action.
        
        Args:
            perception: NPC's perception of the environment
            available_actions: Optional list of action dicts the NPC could take;
                               if None, the system will generate them dynamically.
            
        Returns:
            The chosen action (dict format)
        """
        # Store perception
        self.context.perception = perception
        
        # Convert available_actions to NPCAction objects if provided
        if available_actions:
            action_options = [NPCAction(**act) for act in available_actions]
        else:
            action_options = None
        
        # Run the decision agent
        result = await Runner.run(
            decision_agent,
            f"Make a decision for NPC {self.npc_id}",
            context=self.context
        )
        
        # Grab the final action
        chosen_action = result.final_output
        
        # Convert back to dict for external usage
        if isinstance(chosen_action, NPCAction):
            return chosen_action.model_dump()
        else:
            return chosen_action
    
    async def initialize_long_term_goals(self, npc_data: Dict[str, Any]) -> None:
        """
        Initialize NPC's long-term goals based on personality and archetype.
        """
        self.context.long_term_goals = []
        
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        traits = npc_data.get("personality_traits", [])
        archetypes = npc_data.get("archetypes", [])
        
        # Create goals based on dominance
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
                "description": "Guide submissives to growth through gentle dominance",
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
        if "mentor" in archetypes:
            self.context.long_term_goals.append({
                "type": "development",
                "description": "Guide the development of the player's submissive nature",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"
            })
        elif "seductress" in archetypes:
            self.context.long_term_goals.append({
                "type": "seduction",
                "description": "Gradually increase player's dependency and devotion",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"
            })
    
    async def store_decision(self, action: Dict[str, Any], context: Dict[str, Any]):
        """
        Store the final decision in NPCAgentState.
        """
        def _store():
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM NPCAgentState
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id),
                )
                exists = cursor.fetchone() is not None
                
                action_copy = action.copy()
                # Clean up ephemeral keys
                for ephemeral_key in ["decision_factors", "mask_slippage"]:
                    if ephemeral_key in action_copy:
                        del action_copy[ephemeral_key]
                
                if exists:
                    cursor.execute(
                        """
                        UPDATE NPCAgentState
                        SET last_decision=%s, last_updated=NOW()
                        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                        """,
                        (json.dumps(action_copy), self.npc_id, self.user_id, self.conversation_id),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO NPCAgentState 
                        (npc_id, user_id, conversation_id, last_decision, last_updated)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (self.npc_id, self.user_id, self.conversation_id, json.dumps(action_copy)),
                    )
                conn.commit()
        
        await asyncio.to_thread(_store)
