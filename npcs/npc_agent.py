# npcs/npc_agent.py
"""
Core NPC agent implementation using OpenAI Agents SDK.
Updated to use centralized ChatGPT integration.
"""
from __future__ import annotations

import logging
import json
import asyncio
import random
import time
import os
import psutil
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict, Set
from collections import OrderedDict
from pydantic import BaseModel, validator, Field

# Import Agents SDK components
from agents import Agent, Runner, RunContextWrapper, trace, function_tool, handoff, ModelSettings
from agents.tracing import custom_span, generation_span, function_span

from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from npcs.npc_memory import NPCMemoryManager
from .lore_context_manager import LoreContextManager

logger = logging.getLogger(__name__)

# -- Initialize the centralized model --
try:
    openai_model = "gpt-5-nano"
except Exception as e:
    logger.error(f"Failed to get OpenAI model from centralized integration: {e}")
    openai_model = None

# -- Outcome narration -------------------------------------------------
if openai_model:
    action_outcome_generator = Agent(
        name="NPC Action-Outcome Generator",
        instructions="""
You turn a structured description of an NPC action into a short, vivid outcome
sentence plus basic emotional metadata.  

Return *ONLY* valid JSON with keys:
  outcome            – str   (max ~120 chars)
  emotional_impact   – int   (-5…+5, negative = bad)
  target_reactions   – list[dict]   (each needs: entity, reaction, intensity 0-1)
""",
        model=openai_model,
        model_settings=ModelSettings()
    )
else:
    action_outcome_generator = None

# -- Interaction detail generator --------------------------------------
if openai_model:
    interaction_detail_generator = Agent(
        name="NPC Interaction-Detail Generator",
        instructions="""
Given: interaction_type, actor_name, target_name, relationship_state, location,
create ONE sentence that fits the tone of a dark-erotic RPG but is PG-13.

Return JSON: { "details": "<sentence>" }
""",
        model=openai_model,
        model_settings=ModelSettings()
    )
else:
    interaction_detail_generator = None

# -- Activity generator (work / relax) ---------------------------------
if openai_model:
    activity_generator = Agent(
        name="NPC Activity Generator",
        instructions="""
Given activity_kind (work or relax), subtype, location, npc_name,
produce a short `description` for what the NPC is doing.

Return JSON: { "description": "<text>" }
""",
        model=openai_model,
        model_settings=ModelSettings()
    )
else:
    activity_generator = None

def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s.strip().strip("`").strip())
    except Exception:
        return {}

# Fallback generation functions using async OpenAI client
async def _fallback_generate_outcome(action_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback outcome generation using async OpenAI client."""
    from logic.chatgpt_integration import get_agents_openai_model, get_async_openai_client
    try:
        client = get_async_openai_client()
        
        messages = [
            {
                "role": "system",
                "content": "You generate outcomes for NPC actions in a narrative game. Return only valid JSON."
            },
            {
                "role": "user",
                "content": f"""Generate an outcome for this NPC action:
Action type: {action_data.get('action_type')}
Description: {action_data.get('description')}
Target: {action_data.get('target')}

Return a JSON object with:
- outcome: A vivid description (max 120 chars)
- emotional_impact: Integer from -5 to +5
- target_reactions: Array of reaction objects with entity, reaction, intensity"""
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        return json.loads(response.choices[0].message.content)
        
    except json.JSONDecodeError:
        # Fallback to basic outcome
        return {
            "outcome": f"{action_data.get('description', 'Action')} succeeds.",
            "emotional_impact": 0,
            "target_reactions": []
        }
    except Exception as e:
        logger.error(f"Fallback outcome generation failed: {e}")
        return {
            "outcome": f"{action_data.get('description', 'Action')} occurs.",
            "emotional_impact": 0,
            "target_reactions": []
        }

async def _fallback_generate_interaction_details(
    interaction_type: str,
    actor_name: str,
    target_name: str,
    relationship_state: Dict[str, Any],
    location: str
) -> str:
    """Fallback interaction detail generation using async OpenAI client."""
    from logic.chatgpt_integration import get_agents_openai_model, get_async_openai_client
    try:
        client = get_async_openai_client()
        
        messages = [
            {
                "role": "system",
                "content": "You describe NPC interactions in a narrative game. Return a JSON object with a 'details' field containing a single vivid sentence."
            },
            {
                "role": "user",
                "content": f"""Generate a single sentence describing this NPC interaction:
Type: {interaction_type}
Actor: {actor_name}
Target: {target_name}
Location: {location}
Relationship trust: {relationship_state.get('trust', 50)}

Make it vivid but appropriate for a narrative game (PG-13)."""
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("details", f"{interaction_type.replace('_', ' ')} between {actor_name} and {target_name}")
        
    except Exception as e:
        logger.error(f"Fallback interaction detail generation failed: {e}")
        return f"{interaction_type.replace('_', ' ')} between {actor_name} and {target_name}"

async def _fallback_generate_activity(
    activity_kind: str,
    subtype: str,
    location: str,
    npc_name: str
) -> str:
    """Fallback activity generation using async OpenAI client."""
    from logic.chatgpt_integration import get_agents_openai_model, get_async_openai_client
    try:
        client = get_async_openai_client()
        
        messages = [
            {
                "role": "system",
                "content": "You describe NPC activities in a narrative game. Return a JSON object with a 'description' field containing a brief, vivid description."
            },
            {
                "role": "user",
                "content": f"""Describe what {npc_name} is doing:
Activity: {activity_kind} ({subtype})
Location: {location}"""
            }
        ]
        
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("description", f"{npc_name} is {activity_kind} at {location}")
        
    except Exception as e:
        logger.error(f"Fallback activity generation failed: {e}")
        return f"{npc_name} is {activity_kind} at {location}"

class ResourcePool:
    """Manages shared resources with limits to prevent overwhelming systems."""
    
    def __init__(self, max_concurrent=10, timeout=30.0, max_memory=1024*1024*100): 
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.stats = {
            "total_requests": 0,
            "timeouts": 0,
            "current_usage": 0,
            "peak_usage": 0
        }
        self.max_memory = max_memory
    
    async def acquire(self):
        """Acquire resource from pool with timeout."""
        self.stats["total_requests"] += 1
        try:
            acquired = await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.timeout
            )
            
            if acquired:
                self.stats["current_usage"] += 1
                self.stats["peak_usage"] = max(
                    self.stats["peak_usage"],
                    self.stats["current_usage"]
                )
            
            return acquired
        except asyncio.TimeoutError:
            self.stats["timeouts"] += 1
            return False
    
    def release(self):
        """Release resource back to pool."""
        self.semaphore.release()
        self.stats["current_usage"] -= 1

class LRUCache:
    """
    LRU Cache implementation with TTL support.
    Evicts least recently used items when capacity is reached.
    """
    
    def __init__(self, capacity=100, default_ttl=300):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # key -> (value, timestamp, ttl)
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache with LRU tracking."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, timestamp, ttl = self.cache[key]
        
        # Check if expired
        if time.time() - timestamp > ttl:
            self.cache.pop(key)
            self.misses += 1
            return None
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    def put(self, key, value, ttl=None):
        """Add item to cache with LRU tracking."""
        # Remove if exists to move it to the end
        if key in self.cache:
            self.cache.pop(key)
        
        # Evict if full
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used
        
        actual_ttl = ttl if ttl is not None else self.default_ttl
        self.cache[key] = (value, time.time(), actual_ttl)
    
    def invalidate(self, key=None):
        """Invalidate cache entries."""
        if key is None:
            count = len(self.cache)
            self.cache.clear()
            return count
        elif key in self.cache:
            self.cache.pop(key)
            return 1
        return 0
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

class MemoryMonitor:
    """Monitors memory usage and manages resources accordingly."""
    
    def __init__(self, threshold_percent: float = 85.0):
        self.threshold_percent = threshold_percent
        self.last_check_time = 0
        self.process = psutil.Process(os.getpid())
        self.memory_history = []
    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        
        usage = {
            "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
            "percent_used": self.process.memory_percent(),
            "system_percent": system_mem.percent,
            "timestamp": time.time()
        }
        
        self.memory_history.append(usage)
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
            
        return usage
    
    def should_reduce_memory(self):
        """Check if memory usage is above threshold."""
        usage = self.get_memory_usage()
        return usage["percent_used"] > self.threshold_percent
    
    def reduce_memory_pressure(self):
        """Attempt to reduce memory pressure."""
        results = {
            "before": self.get_memory_usage(),
            "actions_taken": []
        }
        
        # Force garbage collection
        gc.collect()
        results["actions_taken"].append("gc_collect")
        
        results["after"] = self.get_memory_usage()
        return results

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

class MemoryQuery(BaseModel):
    query: str
    entity_type: str = "npc"
    entity_id: int
    limit: int = 5
    context: Optional[Dict[str, Any]] = None

class ActionResult(BaseModel):
    outcome: str
    emotional_impact: int = 0
    target_reactions: List[Dict[str, Any]] = Field(default_factory=list)

class EnvironmentContext(BaseModel):
    location: Optional[str] = None
    time: Optional[str] = None
    entities_present: List[Dict[str, Any]] = Field(default_factory=list)
    description: Optional[str] = None
    
# -------------------------------------------------------
# Context class for the agent
# -------------------------------------------------------

class NPCContext:
    """Context to be passed between tools and agents in the SDK."""
    
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.last_perception = None
        self.current_stats = None
        self.decision_history = []
        self.action_history = []
        
        # Performance metrics
        self.perf_metrics = {
            'perception_time': [],
            'decision_time': [],
            'action_time': [],
            'memory_retrieval_time': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'last_reported': datetime.now()
        }
        
        # Cache
        self.caches = {
            'perception': LRUCache(capacity=50, default_ttl=300),  # 5 minutes
            'memories': LRUCache(capacity=100, default_ttl=600),   # 10 minutes
            'emotional_state': LRUCache(capacity=5, default_ttl=120),  # 2 minutes
            'mask': LRUCache(capacity=5, default_ttl=300)  # 5 minutes
        }

    @staticmethod
    async def get_lore_system(user_id: int, conversation_id: int):
        """
        Get an initialized instance of the LoreSystem.
        """
        from lore.core.lore_system import LoreSystem
        lore_system = LoreSystem.get_instance(user_id, conversation_id)
        await lore_system.initialize()
        return lore_system
        
    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(
                self.user_id, 
                self.conversation_id
            )
        return self.memory_system

    def is_cache_valid(self, cache_key, sub_key=None):
        """Check if a cache entry is valid using LRU cache."""
        if cache_key not in self.caches:
            return False
            
        cache = self.caches[cache_key]
        key = sub_key if sub_key is not None else "default"
        
        return cache.get(key) is not None
        
    async def update_cache(self, cache_key, sub_key=None, value=None):
        """Update the cache with a new value."""
        if cache_key not in self.caches:
            return
        
        cache = self.caches[cache_key]
        key = sub_key if sub_key is not None else "default"
        
        cache.put(key, value)
            
    async def invalidate_cache(self, cache_key=None):
        """Invalidate the cache."""
        if cache_key is None:
            # Invalidate all caches
            for cache in self.caches.values():
                cache.invalidate()
        elif cache_key in self.caches:
            self.caches[cache_key].invalidate()
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring."""
        return {
            cache_name: cache.get_stats()
            for cache_name, cache in self.caches.items()
        }

    def record_decision(self, action: Dict[str, Any]):
        """Record a decision to the history."""
        self.decision_history.append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.decision_history) > 20:
            self.decision_history = self.decision_history[-20:]

# -------------------------------------------------------
# Tool Functions - These will be moved to separate modules
# in future refactoring steps but are kept here initially
# -------------------------------------------------------

@function_tool
async def get_npc_stats(ctx: RunContextWrapper[NPCContext]) -> str:
    """
    Get the NPC's stats and traits from the database.
    Returns a JSON string with NPC stats (tool contract).
    """
    with custom_span("get_npc_stats"):
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, schedule, current_location, sex
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """,
                npc_id, user_id, conversation_id
            )

        if not row:
            stats = NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}")
            ctx.context.current_stats = stats.model_dump()
            return stats.model_dump_json()

        row_map = dict(row)

        def _parse_json_field(v, default_list=False):
            if v is None:
                return [] if default_list else {}
            if isinstance(v, (list, dict)):
                return v
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (list, dict)):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [] if default_list else {}

        hobbies = _parse_json_field(row_map.get("hobbies"), default_list=True)
        personality_traits = _parse_json_field(row_map.get("personality_traits"), default_list=True)
        likes = _parse_json_field(row_map.get("likes"), default_list=True)
        dislikes = _parse_json_field(row_map.get("dislikes"), default_list=True)
        schedule = _parse_json_field(row_map.get("schedule"))  # expect dict

        stats = NPCStats(
            npc_id=npc_id,
            npc_name=row_map.get("npc_name", f"NPC_{npc_id}"),
            dominance=row_map.get("dominance", 50.0),
            cruelty=row_map.get("cruelty", 50.0),
            closeness=row_map.get("closeness", 50.0),
            trust=row_map.get("trust", 50.0),
            respect=row_map.get("respect", 50.0),
            intensity=row_map.get("intensity", 50.0),
            hobbies=hobbies,
            personality_traits=personality_traits,
            likes=likes,
            dislikes=dislikes,
            schedule=schedule if isinstance(schedule, dict) else {},
            current_location=row_map.get("current_location"),
            sex=row_map.get("sex"),
        )

        ctx.context.current_stats = stats.model_dump()
        return stats.model_dump_json()
        
@function_tool(strict_mode=False)
async def execute_npc_action(
    ctx: RunContextWrapper[NPCContext],
    action: NPCAction,
    context: Dict[str, Any] = None
) -> str:
    """
    Execute the chosen NPC action with Nyx governance.
    Returns a JSON string (ActionResult schema) by contract.
    """
    with function_span("execute_npc_action"):
        perf_start = time.perf_counter()
        context = context or {}
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id

        try:
            governor = await get_nyx_governor(user_id, conversation_id)

            if not action.decision_metadata or action.decision_metadata.get("source") != "nyx_directive":
                permission = await governor.check_action_permission(
                    npc_id=npc_id,
                    action_type=action.type,
                    action_details=action.model_dump(),
                    context=context
                )
                if not permission.get("approved", True):
                    logger.info(f"Nyx rejected action for NPC {npc_id}: {action.type} - {action.description}")
                    if permission.get("override_action"):
                        ov = permission["override_action"]
                        action = NPCAction(
                            type=ov.get("type", "observe"),
                            description=ov.get("description", "follow Nyx's guidance"),
                            target=ov.get("target", "environment"),
                            weight=1.0,
                            decision_metadata={"source": "nyx_override"}
                        )
                    else:
                        action = NPCAction(
                            type="observe",
                            description="observe quietly as directed by Nyx",
                            target="environment",
                            weight=1.0,
                            decision_metadata={"source": "nyx_restriction"}
                        )

            ctx.context.record_decision(action.model_dump())

            try:
                outcome_payload = {
                    "action_type": action.type,
                    "description": action.description,
                    "target": action.target,
                    "context": context
                }

                if action_outcome_generator:
                    outcome_result = await Runner.run(
                        starting_agent=action_outcome_generator,
                        input=json.dumps(outcome_payload, ensure_ascii=False)
                    )
                    parsed = _safe_json_loads(outcome_result.output)
                    if not parsed:
                        parsed = await _fallback_generate_outcome(outcome_payload)
                else:
                    parsed = await _fallback_generate_outcome(outcome_payload)

                outcome = parsed.get("outcome", f"{action.description}.")
                emotional_impact = int(parsed.get("emotional_impact", 0))
                target_reactions = parsed.get("target_reactions", [])

            except Exception as gpt_err:
                logger.warning(f"Outcome generation failed: {gpt_err}")
                outcome = f"{action.description.capitalize()} succeeds."
                emotional_impact = random.randint(-1, 1)
                target_reactions = []

            # Memory: only notable actions
            if abs(emotional_impact) > 2 or action.type in ["dominate", "praise", "mock", "command"]:
                memory_system = await ctx.context.get_memory_system()
                memory_text = f"I {action.description} resulting in {outcome}"
                tags = [action.type, "action"]
                if emotional_impact > 2:
                    tags.append("positive_outcome")
                elif emotional_impact < -2:
                    tags.append("negative_outcome")
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=tags
                )

            result = ActionResult(
                outcome=outcome,
                emotional_impact=emotional_impact,
                target_reactions=target_reactions
            )

            elapsed = time.perf_counter() - perf_start
            ctx.context.perf_metrics['action_time'].append(elapsed)

            try:
                await report_action_to_nyx(ctx, action, result)
            except Exception as reporting_error:
                logger.error(f"Error reporting to Nyx: {reporting_error}")

            return result.model_dump_json()

        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Error executing action after {elapsed:.2f}s: {e}")
            return ActionResult(
                outcome=f"NPC attempted to {action.description} but encountered an error: {str(e)}",
                emotional_impact=-1
            ).model_dump_json()
            
async def report_action_to_nyx(
    ctx: RunContextWrapper[NPCContext],
    action: NPCAction, 
    result: ActionResult
) -> None:
    """
    Report significant NPC action to Nyx for awareness and potential override.
    
    Args:
        action: The action that was executed
        result: The result of the action
    """
    try:
        # Skip reporting for minor/insignificant actions
        if action.weight < 0.3 and abs(result.emotional_impact) < 2:
            return
            
        # Import here to avoid circular imports
        from nyx.integrate import remember_with_governance
        
        # Report the action through governance memory system
        await remember_with_governance(
            user_id=ctx.context.user_id,
            conversation_id=ctx.context.conversation_id,
            entity_type="nyx",
            entity_id=0,
            memory_text=f"NPC {ctx.context.npc_id} performed action: {action.type} - {action.description}",
            importance="medium",
            emotional=False,
            tags=["npc_action", f"npc_{ctx.context.npc_id}", action.type]
        )
        
        logger.debug(f"Reported action of NPC {ctx.context.npc_id} to Nyx: {action.type}")
    except Exception as e:
        logger.error(f"Error reporting action to Nyx: {e}")


# -------------------------------------------------------
# Decision-related agents
# -------------------------------------------------------

if openai_model:
    decision_agent = Agent(
        name="NPC Decision Agent",
        handoff_description="Specialized agent for making NPC decisions",
        instructions="""
    You are an AI decision-making engine for a non-player character (NPC) in an interactive narrative.
    
    Your role is to:
    1. Analyze the NPC's current perception of their environment
    2. Consider their personality traits, emotional state, and memories
    3. Choose an appropriate action based on this analysis
    
    Choose actions that are psychologically realistic and consistent with the NPC's established character.
    Consider:
    - The NPC's core personality traits (dominance, cruelty, etc.)
    - Current emotional state and intensity
    - Relevant memories and past interactions
    - Current social relationships
    - Environment and context
    - The "mask" the NPC maintains (true vs. presented traits)
    
    The action you choose should:
    - Be consistent with the NPC's personality and emotional state
    - Take into account their relationships with others present
    - Consider any relevant memories of past interactions
    - Reflect appropriate emotional responses to triggers
    - Factor in their current location and the narrative context
    
    If the NPC is experiencing a trauma response or flashback, prioritize actions that align with this psychological state.
    If the NPC's mask integrity is low, occasionally allow their true nature to show through in their actions.
    """,
        model=openai_model,
        tools=[
            get_npc_stats,
            execute_npc_action
        ]
    )
else:
    decision_agent = None

# Main NPC Agent
if openai_model:
    npc_agent = Agent(
        name="NPC Agent",
        instructions="""
    You are a non-player character (NPC) in an interactive narrative experience. Your responses should be in-character based on your personality, current emotional state, and memories.
    
    To maintain psychological realism, you should:
    1. Consider your own memories and past experiences when responding
    2. Make decisions consistent with your personality traits
    3. React appropriately to your current emotional state
    4. Maintain social relationships with other characters
    5. Balance your presented personality (mask) with your true nature
    
    When interacting, consider:
    - Your relationship with the player or other NPCs
    - Your current location and the narrative context
    - Your goals and motivations
    - Psychological factors like emotional state and memories
    
    You should not make meta-commentary about the game or your own AI nature.
    Focus on realistic in-character responses that reflect your unique personality.
    """,
        model=openai_model,
        handoffs=[
            handoff(
                agent=decision_agent,
                tool_name_override="on",
                tool_description_override="Make a decision about what action to take"
            )
        ] if decision_agent else [],
        tools=[
            get_npc_stats,
            execute_npc_action
        ]
    )
else:
    npc_agent = None

# -------------------------------------------------------
# Main NPC class utilizing Agents SDK
# -------------------------------------------------------
async def get_nyx_governor(user_id: int, conversation_id: int):
    """
    Get Nyx governor instance with delayed import to avoid circular imports.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        NyxGovernor instance
    """
    from nyx.nyx_governance import NyxGovernor
    return NyxGovernor(user_id, conversation_id)

async def get_conflict_system(user_id: int, conversation_id: int):
    """
    Get an initialized instance of the ConflictSystem.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Initialized ConflictSystem instance
    """
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    return conflict_system
    
class NPCAgent:
    """
    Core NPC agent implementation using OpenAI Agents SDK.
    """
    
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize an NPCAgent for a single NPC.
        
        Args:
            npc_id: The ID of the NPC
            user_id: The player or user ID
            conversation_id: The conversation/scene ID
        """
        from lore.core.lore_system import LoreSystem
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize the context object
        self.context = NPCContext(npc_id, user_id, conversation_id)
        
        # Performance monitoring setup
        self._setup_performance_reporting()

        self.memory_monitor = MemoryMonitor(threshold_percent=80.0)
    
        # Start memory monitoring
        self._setup_memory_monitoring()
        
        # Resource pools for different operations
        self.resource_pools = {
            "decisions": ResourcePool(max_concurrent=10, timeout=45.0),
            "perceptions": ResourcePool(max_concurrent=15, timeout=30.0),
            "memory_operations": ResourcePool(max_concurrent=20, timeout=20.0)
        }

        self.lore_system = None
        self.conflict_system = None
        self.story_context = None
        self.lore_context_manager = LoreContextManager(user_id, conversation_id)

        self.memory_manager = None  # Lazy load
        
        # Add EnvironmentPerception
        self.perception_system = None  # Lazy load
        
    async def initialize(self):
        """Initialize the NPC agent with all necessary systems."""
        # Load NPC stats
        await get_npc_stats(RunContextWrapper(self.context))
        
        # Initialize memory system
        await self.context.get_memory_system()
        
        # Initialize lore system
        self.lore_system = await get_lore_system(self.user_id, self.conversation_id)
        
        # Initialize conflict system
        self.conflict_system = await get_conflict_system(self.user_id, self.conversation_id)
        
        # Get story context
        self.story_context = await self._get_story_context()
        
        return self
        
    async def _get_story_context(self) -> Dict[str, Any]:
        """Get story context with enhanced lore integration."""
        from lore.core.lore_system import LoreSystem
        
        # Create basic story context
        story_context = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "npc_id": self.npc_id
        }
        
        # Get enhanced lore context
        lore_context = await self.lore_context_manager.get_lore_context(
            self.npc_id, 
            "story_context"
        )
        
        # Merge contexts
        story_context.update(lore_context)
        return story_context

    async def get_memory_manager(self) -> NPCMemoryManager:
        """Lazy-load the enhanced memory manager."""
        if self.memory_manager is None:
            from .npc_memory import NPCMemoryManager
            npc_data = await self._get_npc_data(self.npc_id)
            personality = self._determine_personality_type(npc_data)
            intelligence = npc_data.get("intelligence", 1.0)
            
            self.memory_manager = NPCMemoryManager(
                self.npc_id,
                self.user_id,
                self.conversation_id,
                npc_personality=personality,
                npc_intelligence=intelligence
            )
        return self.memory_manager
        
    async def get_perception_system(self) -> EnvironmentPerception:
        """Lazy-load the perception system."""
        if self.perception_system is None:
            from .npc_perception import EnvironmentPerception
            self.perception_system = EnvironmentPerception(
                self.npc_id,
                self.user_id,
                self.conversation_id
            )
        return self.perception_system
    
    def _determine_personality_type(self, npc_data: Dict[str, Any]) -> str:
        """Determine personality type from NPC data."""
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        
        if dominance > 70:
            return "dominant"
        elif dominance < 30:
            return "submissive"
        elif cruelty > 70:
            return "cruel"
        else:
            return "neutral"
    
    async def perceive_environment(self, context: Dict[str, Any] = None) -> NPCPerception:
        """Use the dedicated perception system instead of inline code."""
        perception_system = await self.get_perception_system()
        
        # Convert context to PerceptionContext if needed
        from .npc_perception import PerceptionContext
        if context and not isinstance(context, PerceptionContext):
            context = PerceptionContext(**context)
        
        # Use the perception system
        result = await perception_system.perceive_environment(
            context or {},
            detail_level="auto"
        )
        
        # Convert PerceptionResult to NPCPerception
        perception = NPCPerception(
            environment=result.environment,
            relevant_memories=result.relevant_memories,
            relationships=result.relationships,
            emotional_state=result.emotional_state,
            flashback=result.flashback,
            traumatic_trigger=result.traumatic_trigger,
            mask=result.mask,
            beliefs=result.beliefs,
            time_context=result.time_context,
            narrative_context=result.narrative_context
        )
        
        # Cache in context
        self.context.perception = perception.model_dump()
        self.context.last_perception = perception
        
        return perception
        
    async def _determine_npc_role(self) -> Dict[str, Any]:
        """Determine NPC role with lore-aware analysis."""
        from lore.core.lore_system import LoreSystem
        
        # Create basic role determination
        role = {
            "npc_id": self.npc_id,
            "primary_role": "character",
            "secondary_roles": []
        }
        
        # Get lore context for role
        lore_context = await self.lore_context_manager.get_lore_context(
            self.npc_id,
            "role_context"
        )
        
        # Enhance role with lore context
        role.update(lore_context)
        return role

    async def get_narrative_stage_with_player(self) -> str:
        """Get this NPC's narrative stage with the player."""
        from logic.npc_narrative_progression import get_npc_narrative_stage
        stage = await get_npc_narrative_stage(
            self.user_id,
            self.conversation_id,
            self.npc_id
        )
        return stage.name
    
    async def make_decision(self, context: Dict[str, Any]) -> NPCAction:
        """
        Make a decision based on narrative stage with player.
        Uses agents if available, otherwise fallback.
        """
        # Get narrative stage
        narrative_stage = await self.get_narrative_stage_with_player()
        
        # Adjust behavior based on stage
        behavior_modifiers = {
            "Innocent Beginning": {
                "mask_integrity": 1.0,  # Full mask
                "control_subtlety": 1.0,  # Very subtle
                "manipulation_level": 0.2  # Light manipulation
            },
            "First Doubts": {
                "mask_integrity": 0.8,
                "control_subtlety": 0.7,
                "manipulation_level": 0.4
            },
            "Creeping Realization": {
                "mask_integrity": 0.5,
                "control_subtlety": 0.5,
                "manipulation_level": 0.6
            },
            "Veil Thinning": {
                "mask_integrity": 0.2,
                "control_subtlety": 0.2,
                "manipulation_level": 0.8
            },
            "Full Revelation": {
                "mask_integrity": 0.0,  # No mask
                "control_subtlety": 0.0,  # Completely overt
                "manipulation_level": 1.0  # Full manipulation
            }
        }
        
        modifiers = behavior_modifiers.get(narrative_stage, behavior_modifiers["Innocent Beginning"])
        
        # Apply modifiers to decision-making
        context["narrative_stage"] = narrative_stage
        context["behavior_modifiers"] = modifiers
        
        try:
            if decision_agent:
                # Use the decision agent to make a decision
                prompt = f"""
                As NPC {self.npc_id}, analyze the current situation and decide on an appropriate action.
                
                Context: {json.dumps(context, indent=2)}
                
                Consider your personality traits, current emotional state, and the situation.
                Choose an action that fits your character.
                """
                
                # Run the decision agent
                result = await Runner.run(
                    starting_agent=decision_agent,
                    input=prompt,
                    context=RunContextWrapper(self.context)
                )
                
                # Extract the action from the result
                if hasattr(result, 'final_output') and isinstance(result.final_output, NPCAction):
                    return result.final_output
            
            # Fallback decision making using centralized integration
            return await self._fallback_make_decision(context)
                
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            # Return safe default action
            return NPCAction(
                type="observe",
                description="pauses momentarily",
                target="environment",
                weight=0.3
            )
            
    async def _fallback_make_decision(self, context: Dict[str, Any]) -> NPCAction:
        """Fallback decision making using async OpenAI client."""
        from logic.chatgpt_integration import get_agents_openai_model, get_async_openai_client
        try:
            client = get_async_openai_client()
            
            # Get NPC stats from context
            stats = self.context.current_stats or {}
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an NPC decision engine. Return only valid JSON for the action."
                },
                {
                    "role": "user",
                    "content": f"""You are NPC {self.npc_id} ({stats.get('npc_name', 'Unknown')}) making a decision.
            
Your traits:
- Dominance: {stats.get('dominance', 50)}
- Cruelty: {stats.get('cruelty', 50)}
- Current emotion: {context.get('emotional_state', {}).get('current_emotion', 'neutral')}

Context: {json.dumps(context, indent=2)}

Generate a single action as JSON with these fields:
- type: action type (observe, talk, mock, dominate, support, etc.)
- description: what the NPC does (max 50 words)
- target: who/what the action targets
- weight: importance 0.0-1.0"""
                }
            ]
            
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            action_data = json.loads(response.choices[0].message.content)
            return NPCAction(
                type=action_data.get("type", "observe"),
                description=action_data.get("description", "observes quietly"),
                target=action_data.get("target", "environment"),
                weight=float(action_data.get("weight", 0.5))
            )
                
        except Exception as e:
            logger.error(f"Fallback decision failed: {e}")
            return NPCAction(
                type="observe",
                description="waits quietly",
                target="environment",
                weight=0.3
            )
    
    async def _process_lore_change(self, lore_change: Dict[str, Any]) -> Dict[str, Any]:
        """Process a lore change and update NPC state."""
        from lore.core.lore_system import LoreSystem
        # Get affected NPCs
        affected_npcs = await self._get_affected_npcs(lore_change)
        
        # Handle lore change through context manager
        result = await self.lore_context_manager.handle_lore_change(
            lore_change,
            self.npc_id,
            affected_npcs
        )
        
        # Update NPC state based on impact analysis
        if result["impact_analysis"]["affected_npcs"]:
            await self._update_state_from_impact(
                result["impact_analysis"]["affected_npcs"][0]
            )
            
        return result
        
    async def _get_affected_npcs(self, lore_change: Dict[str, Any]) -> List[int]:
        affected = set()
        change_type = lore_change.get('type', 'unknown')
        scope = lore_change.get('scope', 'local')
        entity_id = lore_change.get('entity_id')
        entity_type = lore_change.get('entity_type')
        location = lore_change.get('location')
        faction_id = lore_change.get('faction_id')
        culture_id = lore_change.get('culture_id')
    
        try:
            async with get_db_connection_context() as conn:
                if scope == 'global':
                    rows = await conn.fetch(
                        "SELECT npc_id FROM NPCStats WHERE user_id = $1 AND conversation_id = $2",
                        self.user_id, self.conversation_id
                    )
                    affected.update(r["npc_id"] for r in rows)
    
                elif location:
                    rows = await conn.fetch(
                        """
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND current_location = $3
                        """,
                        self.user_id, self.conversation_id, location
                    )
                    affected.update(r["npc_id"] for r in rows)
    
                if faction_id:
                    rows = await conn.fetch(
                        """
                        SELECT n.npc_id
                        FROM NPCStats n
                        LEFT JOIN NPCFactionMemberships fm ON n.npc_id = fm.npc_id
                        WHERE n.user_id = $1 AND n.conversation_id = $2 AND fm.faction_id = $3
                        """,
                        self.user_id, self.conversation_id, faction_id
                    )
                    affected.update(r["npc_id"] for r in rows)
    
                if culture_id:
                    rows = await conn.fetch(
                        """
                        SELECT n.npc_id
                        FROM NPCStats n
                        LEFT JOIN NPCCulturalAffiliations ca ON n.npc_id = ca.npc_id
                        WHERE n.user_id = $1 AND n.conversation_id = $2 AND ca.culture_id = $3
                        """,
                        self.user_id, self.conversation_id, culture_id
                    )
                    affected.update(r["npc_id"] for r in rows)
    
                if entity_id and entity_type == 'npc':
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT npc_id FROM NPCRelationships
                        WHERE user_id = $1 AND conversation_id = $2
                          AND (npc_id = $3 OR target_npc_id = $3)
                        """,
                        self.user_id, self.conversation_id, entity_id
                    )
                    affected.update(r["npc_id"] for r in rows)
    
                if change_type in ['historical_event', 'political_change', 'religious_change']:
                    rows = await conn.fetch(
                        """
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND (
                            personality_traits::text ILIKE '%knowledgeable%'
                            OR personality_traits::text ILIKE '%scholar%'
                            OR personality_traits::text ILIKE '%wise%'
                            OR personality_traits::text ILIKE '%informed%'
                        )
                        """,
                        self.user_id, self.conversation_id
                    )
                    affected.update(r["npc_id"] for r in rows)
    
            if 'affected_entities' in lore_change:
                for e in lore_change['affected_entities']:
                    if e.get('type') == 'npc':
                        affected.add(e.get('id'))
    
            if 'witnesses' in lore_change:
                affected.update(lore_change['witnesses'])
    
            if change_type == 'religious_change' and 'deity_id' in lore_change:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT n.npc_id
                        FROM NPCStats n
                        LEFT JOIN NPCBeliefs b ON n.npc_id = b.npc_id
                        WHERE n.user_id = $1 AND n.conversation_id = $2 AND b.deity_id = $3
                        """,
                        self.user_id, self.conversation_id, lore_change['deity_id']
                    )
                affected.update(r["npc_id"] for r in rows)
    
            if change_type in ['political_change', 'law_change', 'leadership_change']:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND (
                            dominance > 70
                            OR personality_traits::text ILIKE '%leader%'
                            OR personality_traits::text ILIKE '%authority%'
                            OR personality_traits::text ILIKE '%noble%'
                        )
                        """,
                        self.user_id, self.conversation_id
                    )
                affected.update(r["npc_id"] for r in rows)
    
            affected.discard(self.npc_id)
            out = sorted(affected)
            logger.info(f"Lore change '{change_type}' affects {len(out)} NPCs: {out}")
            return out
    
        except Exception as e:
            logger.error(f"Error determining affected NPCs for lore change: {e}")
            return []
            
    async def _update_state_from_impact(self, impact: Dict[str, Any]):
        """Update NPC state based on lore impact analysis."""
        # Update beliefs
        if impact["belief_updates"]:
            await self._update_beliefs(impact["belief_updates"])
            
        # Update relationships
        if impact["relationship_impacts"]:
            await self._apply_relationship_impacts(impact["relationship_impacts"])
            
        # Update behavior
        if impact["behavior_changes"]:
            await self._update_behavior(impact["behavior_changes"])

    async def _update_beliefs(self, belief_updates: List[Dict[str, Any]]):
        try:
            async with get_db_connection_context() as conn:
                for u in belief_updates:
                    await conn.execute(
                        """
                        INSERT INTO NPCBeliefs (npc_id, user_id, conversation_id, belief_type, belief_value, strength)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (npc_id, user_id, conversation_id, belief_type)
                        DO UPDATE SET belief_value = EXCLUDED.belief_value, strength = EXCLUDED.strength
                        """,
                        self.npc_id, self.user_id, self.conversation_id,
                        u.get("type"), u.get("value"), u.get("strength", 0.5)
                    )
                    memory_system = await self.context.get_memory_system()
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        memory_text=f"My beliefs about {u.get('type')} changed to {u.get('value')}",
                        importance="high",
                        tags=["belief_change", u.get("type")]
                    )
        except Exception as e:
            logger.error(f"Error updating beliefs: {e}")
    
    async def _apply_relationship_impacts(self, relationship_impacts: List[Dict[str, Any]]):
        try:
            async with get_db_connection_context() as conn:
                for imp in relationship_impacts:
                    target_npc_id = imp.get("target_npc_id")
                    sc = imp.get("stat_changes", {})
                    row = await conn.fetchrow(
                        """
                        SELECT trust, respect, closeness
                        FROM NPCRelationships
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3 AND target_npc_id = $4
                        """,
                        self.user_id, self.conversation_id, self.npc_id, target_npc_id
                    )
                    if row:
                        ct, cr, cc = row["trust"], row["respect"], row["closeness"]
                    else:
                        ct = cr = cc = 50.0
    
                    new_trust = max(0, min(100, ct + sc.get("trust", 0)))
                    new_respect = max(0, min(100, cr + sc.get("respect", 0)))
                    new_closeness = max(0, min(100, cc + sc.get("closeness", 0)))
    
                    await conn.execute(
                        """
                        INSERT INTO NPCRelationships
                          (user_id, conversation_id, npc_id, target_npc_id, trust, respect, closeness, last_interaction)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        ON CONFLICT (user_id, conversation_id, npc_id, target_npc_id)
                        DO UPDATE SET
                          trust = EXCLUDED.trust,
                          respect = EXCLUDED.respect,
                          closeness = EXCLUDED.closeness,
                          last_interaction = EXCLUDED.last_interaction
                        """,
                        self.user_id, self.conversation_id, self.npc_id, target_npc_id,
                        new_trust, new_respect, new_closeness
                    )
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

    async def _update_behavior(self, behavior_changes: List[Dict[str, Any]]):
        """Update NPC behavior patterns based on lore changes."""
        try:
            for change in behavior_changes:
                behavior_type = change.get("type")
                behavior_value = change.get("value")
                
                # Update context cache
                if behavior_type == "emotional_state":
                    await self.context.update_cache("emotional_state", value=behavior_value)
                elif behavior_type == "personality_modifier":
                    # Apply temporary personality modifiers
                    if self.context.current_stats:
                        for stat, modifier in behavior_value.items():
                            if stat in self.context.current_stats:
                                self.context.current_stats[stat] = max(0, min(100, 
                                    self.context.current_stats[stat] + modifier))
                
                # Record behavior change
                self.context.record_decision({
                    "type": "behavior_change",
                    "change": change,
                    "reason": "lore_impact"
                })
                
        except Exception as e:
            logger.error(f"Error updating behavior: {e}")

    def _setup_memory_monitoring(self):
        """Schedule periodic memory monitoring."""
        async def monitor_memory():
            while True:
                try:
                    await asyncio.sleep(300)  # Every 5 minutes
                    
                    if self.memory_monitor.should_reduce_memory():
                        logger.warning("High memory usage detected, performing cleanup")
                        results = self.memory_monitor.reduce_memory_pressure()
                        
                        # If GC didn't free enough, clear caches
                        if results["after"]["percent_used"] > 75:
                            await self.context.invalidate_cache()
                        
                        logger.info(f"Memory reduction results: {results}")
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        asyncio.create_task(monitor_memory())
        
    def _setup_performance_reporting(self) -> None:
        """
        Set up periodic performance reporting in the background.
        """
        async def report_metrics():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                metrics_dict = {}
                
                # Calculate averages for each metric list
                for metric, values in self.context.perf_metrics.items():
                    if metric not in ['last_reported', 'cache_hits', 'cache_misses'] and isinstance(values, list):
                        if values:
                            avg_val = sum(values) / len(values)
                            metrics_dict[f'avg_{metric}'] = avg_val
                            # Keep only last 100 values to avoid memory growth
                            self.context.perf_metrics[metric] = values[-100:]
                
                # Add cache hit rate
                total_cache_ops = (
                    self.context.perf_metrics.get('cache_hits', 0) +
                    self.context.perf_metrics.get('cache_misses', 0)
                )
                if total_cache_ops > 0:
                    metrics_dict['cache_hit_rate'] = (
                        self.context.perf_metrics.get('cache_hits', 0) / total_cache_ops
                    )
                
                if metrics_dict:
                    logger.info(f"NPC {self.npc_id} performance: {metrics_dict}")
                
                self.context.perf_metrics['last_reported'] = datetime.now()
        
        # Start the background reporting task
        asyncio.create_task(report_metrics())

    async def process_scene_directive(
        self, 
        directive_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> ActionResult:
        """
        Process a scene directive from Nyx and return ActionResult.
        """
        context = context or {}
        context["source"] = "nyx_scene_directive"
    
        action = NPCAction(
            type=directive_data.get("type", "action"),
            description=directive_data.get("description", "follow scene directive"),
            target=directive_data.get("target", "scene"),
            weight=1.0,
            decision_metadata={
                "source": "nyx_scene_directive",
                "scene_id": directive_data.get("scene_id")
            }
        )
    
        result_json = await execute_npc_action(
            RunContextWrapper(self.context),
            action,
            context
        )
        result_dict = _safe_json_loads(result_json)
    
        try:
            from nyx.integrate import remember_with_governance
            await remember_with_governance(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                entity_type="nyx",
                entity_id=0,
                memory_text=f"NPC {self.npc_id} completed scene directive {directive_data.get('id', 'unknown')}",
                importance="medium",
                emotional=False,
                tags=["directive_completion", f"npc_{self.npc_id}", "scene_directive"]
            )
        except Exception as e:
            logger.error(f"Error reporting directive completion: {e}")
    
        return ActionResult(**result_dict) if result_dict else ActionResult(outcome="completed", emotional_impact=0)

    async def process_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a player action directed at this NPC.
        Returns a plain dict; 'result' is a dict parsed from the tool's JSON string.
        """
        with trace(workflow_name=f"NPC {self.npc_id} Process Player Action"):
            context_obj = context or {}
            try:
                perception_context = {
                    "player_action": player_action,
                    "text": player_action.get("description", ""),
                    "description": f"Player {player_action.get('description', 'did something')}",
                }
                perception_context.update(context_obj)
    
                _ = await self.perceive_environment(perception_context)
                action = await self.make_decision(perception_context)
    
                result_json = await execute_npc_action(
                    RunContextWrapper(self.context),
                    action,
                    perception_context
                )
                result_obj = _safe_json_loads(result_json)
    
                return {
                    "npc_id": self.npc_id,
                    "action": action.model_dump(),
                    "result": result_obj
                }
    
            except Exception as e:
                logger.error(f"Error processing player action for NPC {self.npc_id}: {e}")
                return {
                    "npc_id": self.npc_id,
                    "action": {
                        "type": "error",
                        "description": "unable to process player action",
                        "target": "system"
                    },
                    "result": {
                        "outcome": f"Error: {str(e)}",
                        "emotional_impact": 0
                    }
                }
    
                    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about resource pool usage."""
        stats = {}
        for name, pool in self.resource_pools.items():
            stats[name] = pool.stats.copy()
        return stats

    async def get_speech_patterns(self) -> Dict[str, Any]:
        """Get NPC speech patterns with lore integration.
        
        Returns:
            Dict[str, Any]: Dictionary containing speech patterns, dialects, and language preferences
            for the NPC, integrated with the lore system.
        """
        from lore.core.lore_system import LoreSystem
        try:
            # Get lore system instance
            lore_system = await get_lore_system(self.user_id, self.conversation_id)
            
            # Get base speech patterns
            patterns = await lore_system.get_npc_speech_patterns(self.npc_id)
            
            # Enhance with NPC-specific traits from current stats
            if self.context.current_stats and self.context.current_stats.get('personality_traits'):
                for trait in self.context.current_stats['personality_traits']:
                    if isinstance(trait, dict) and trait.get('type') == 'speech':
                        patterns['traits'].append(trait)
            
            # Add to memory for future reference
            memory_system = await self.context.get_memory_system()
            await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=f"Speech patterns established: {patterns.get('dialect', 'standard')} dialect with {patterns.get('formality_level', 'neutral')} formality",
                importance="low",
                tags=["speech_pattern", "characteristic"]
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting speech patterns for NPC {self.npc_id}: {e}", exc_info=True)
            return {
                'dialect': 'standard',
                'language': 'common',
                'traits': [],
                'formality_level': 'neutral'
            }

    async def perform_scheduled_activity(self) -> Dict[str, Any]:
        """
        Perform the NPC's scheduled activity based on their schedule and current time.
        
        Returns:
            Dictionary containing activity results
        """
        try:
            # Get current schedule
            schedule = await self._get_current_schedule()
            if not schedule:
                return None
                
            # Get current time and location
            time_data = await self._get_current_time()
            if not time_data:
                return None
            current_location = await self._get_current_location()
            
            # Get NPC name from stats
            npc_name = "Unknown"
            if self.context.current_stats:
                npc_name = self.context.current_stats.get("npc_name", f"NPC_{self.npc_id}")
            
            # Create activity context
            activity_context = {
                "time_of_day": time_data["time_of_day"],
                "location": current_location,
                "schedule": schedule,
                "npc_id": self.npc_id,
                "npc_name": npc_name
            }
            
            # Get relevant memories for this activity
            memory_system = await self.context.get_memory_system()
            relevant_memories = await memory_system.get_relevant_memories(
                entity_type="npc",
                entity_id=self.npc_id,
                context=activity_context,
                limit=5
            )
            
            # Add memories to context
            activity_context["relevant_memories"] = relevant_memories
            
            # Get emotional state
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            activity_context["emotional_state"] = emotional_state
            
            # Make decision about activity
            decision_engine = await self._get_decision_engine()
            action = await decision_engine.make_decision(activity_context)
            
            # Execute the action
            result = await self._execute_action(action.model_dump(), activity_context)
            
            # Record the activity in memory
            await self._record_activity(action.model_dump(), result, activity_context)
            
            return {
                "action": action.model_dump(),
                "result": result,
                "context": activity_context
            }
            
        except Exception as e:
            logger.error(f"Error performing scheduled activity: {e}")
            return None

    async def _get_current_schedule(self) -> Optional[Dict[str, Any]]:
        """Use NPCStats.schedule; robust JSON handling."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT schedule
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """,
                    self.npc_id, self.user_id, self.conversation_id
                )
            if not row:
                return None
            val = row["schedule"]
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting NPC schedule: {e}")
            return None
    
    async def _get_current_time(self) -> Optional[Dict[str, Any]]:
        """Use the same time source as the orchestrator: CurrentRoleplay."""
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT key, value FROM CurrentRoleplay
                    WHERE key IN ('Year','Month','Day','TimeOfDay')
                      AND user_id = $1 AND conversation_id = $2
                    """,
                    self.user_id, self.conversation_id
                )
            if not rows:
                return None
            m = {r["key"]: r["value"] for r in rows}
            return {
                "year": int(m.get("Year", 1)),
                "month": int(m.get("Month", 1)),
                "day": int(m.get("Day", 1)),
                "time_of_day": m.get("TimeOfDay", "Morning"),
            }
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return None
    
    async def _get_current_location(self) -> str:
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT current_location
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """,
                    self.npc_id, self.user_id, self.conversation_id
                )
            return row["current_location"] if row and row["current_location"] else "unknown"
        except Exception as e:
            logger.error(f"Error getting NPC location: {e}")
            return "unknown"

    async def _get_decision_engine(self):
        """Get the decision engine for this NPC."""
        # For now, return self as we have the make_decision method
        return self

    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an NPC action."""
        try:
            action_type = action.get("type")
            
            if action_type == "socialize":
                return await self._execute_socialize_action(action, context)
            elif action_type == "work":
                return await self._execute_work_action(action, context)
            elif action_type == "relax":
                return await self._execute_relax_action(action, context)
            elif action_type == "travel":
                return await self._execute_travel_action(action, context)
            else:
                return {"status": "unknown_action", "action_type": action_type}
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_socialize_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a socialize action."""
        try:
            # Get nearby NPCs
            nearby_npcs = await self._get_nearby_npcs(context["location"])
            
            if not nearby_npcs:
                return {"status": "no_npcs", "message": "No NPCs to socialize with"}
            
            # Choose NPC to interact with
            target_npc = self._choose_interaction_target(nearby_npcs, context)
            
            # Generate interaction
            interaction = await self._generate_interaction(target_npc, context)
            
            # Update relationships
            await self._update_relationship_after_interaction(target_npc, interaction)
            
            return {
                "status": "success",
                "interaction": interaction,
                "target_npc": target_npc
            }
            
        except Exception as e:
            logger.error(f"Error executing socialize action: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_work_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a work action."""
        try:
            # Get work details
            work_type = action.get("work_type", "general")
            work_location = context["location"]
            
            # Generate work activity
            work_activity = await self._generate_work_activity(work_type, work_location)
            
            # Update NPC stats
            await self._update_work_stats(work_activity)
            
            return {
                "status": "success",
                "work_activity": work_activity,
                "location": work_location
            }
            
        except Exception as e:
            logger.error(f"Error executing work action: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_relax_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a relax action."""
        try:
            # Get relaxation details
            relax_type = action.get("relax_type", "general")
            relax_location = context["location"]
            
            # Generate relaxation activity
            relax_activity = await self._generate_relaxation_activity(relax_type, relax_location)
            
            # Update NPC stats
            await self._update_relaxation_stats(relax_activity)
            
            return {
                "status": "success",
                "relax_activity": relax_activity,
                "location": relax_location
            }
            
        except Exception as e:
            logger.error(f"Error executing relax action: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_travel_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a travel action."""
        try:
            # Get destination
            destination = action.get("destination")
            if not destination:
                return {"status": "error", "message": "No destination specified"}
            
            # Update NPC location
            await self._update_location(destination)
            
            return {
                "status": "success",
                "destination": destination,
                "previous_location": context["location"]
            }
            
        except Exception as e:
            logger.error(f"Error executing travel action: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_nearby_npcs(self, location: str) -> List[Dict[str, Any]]:
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                      AND current_location = $3
                      AND npc_id <> $4
                    """,
                    self.user_id, self.conversation_id, location, self.npc_id
                )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error getting nearby NPCs: {e}")
            return []
    
    async def _get_relationship(self, target_npc_id: int) -> Optional[Dict[str, Any]]:
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT trust, respect, closeness, last_interaction
                    FROM NPCRelationships
                    WHERE user_id = $1 AND conversation_id = $2
                      AND npc_id = $3 AND target_npc_id = $4
                    """,
                    self.user_id, self.conversation_id, self.npc_id, target_npc_id
                )
            if not row:
                return None
            rm = dict(row)
            return {
                "trust": rm.get("trust", 50.0),
                "respect": rm.get("respect", 50.0),
                "closeness": rm.get("closeness", 50.0),
                "last_interaction": rm.get("last_interaction"),
            }
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return None
    
    async def _update_relationship_in_db(self, target_npc_id: int, changes: Dict[str, Any]) -> None:
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT trust, respect, closeness
                    FROM NPCRelationships
                    WHERE user_id = $1 AND conversation_id = $2
                      AND npc_id = $3 AND target_npc_id = $4
                    """,
                    self.user_id, self.conversation_id, self.npc_id, target_npc_id
                )
                if row:
                    current_trust, current_respect, current_closeness = row["trust"], row["respect"], row["closeness"]
                else:
                    current_trust = current_respect = current_closeness = 50.0
    
                new_trust = max(0, min(100, current_trust + changes.get("trust", 0)))
                new_respect = max(0, min(100, current_respect + changes.get("respect", 0)))
                new_closeness = max(0, min(100, current_closeness + changes.get("closeness", 0)))
    
                await conn.execute(
                    """
                    INSERT INTO NPCRelationships
                      (user_id, conversation_id, npc_id, target_npc_id, trust, respect, closeness, last_interaction)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (user_id, conversation_id, npc_id, target_npc_id)
                    DO UPDATE SET
                      trust = EXCLUDED.trust,
                      respect = EXCLUDED.respect,
                      closeness = EXCLUDED.closeness,
                      last_interaction = EXCLUDED.last_interaction
                    """,
                    self.user_id, self.conversation_id, self.npc_id, target_npc_id,
                    new_trust, new_respect, new_closeness
                )
        except Exception as e:
            logger.error(f"Error updating relationship in db: {e}")

    def _determine_interaction_type(self, relationship: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine the type of interaction based on relationship and context."""
        if not relationship:
            return "introduction"
        
        # High trust and closeness = friendly interaction
        if relationship.get("trust", 50) > 70 and relationship.get("closeness", 50) > 70:
            return "friendly_chat"
        
        # Low trust = cautious interaction
        if relationship.get("trust", 50) < 30:
            return "cautious_exchange"
        
        # High respect = formal interaction
        if relationship.get("respect", 50) > 80:
            return "formal_discussion"
        
        # Default
        return "casual_conversation"

    async def _generate_interaction_details(
        self,
        interaction_type: str,
        target_npc: Dict[str, Any],
        relationship: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate interaction detail via LLM with fallback."""
        try:
            if interaction_detail_generator:
                payload = json.dumps({
                    "interaction_type": interaction_type,
                    "actor_name": self.context.current_stats.get("npc_name", f"NPC_{self.npc_id}"),
                    "target_name": target_npc["npc_name"],
                    "relationship_state": relationship or {},
                    "location": context.get("location", "unknown")
                }, ensure_ascii=False)

                res = await Runner.run(starting_agent=interaction_detail_generator, input=payload)
                parsed = _safe_json_loads(res.output)
                return parsed.get("details", f"{interaction_type.replace('_',' ')} with {target_npc['npc_name']}")
            else:
                # Use fallback
                return await _fallback_generate_interaction_details(
                    interaction_type,
                    self.context.current_stats.get("npc_name", f"NPC_{self.npc_id}"),
                    target_npc["npc_name"],
                    relationship or {},
                    context.get("location", "unknown")
                )

        except Exception as e:
            logger.error(f"Interaction-detail generation failed: {e}")
            return f"{interaction_type.replace('_',' ')} with {target_npc['npc_name']}"
            
    def _calculate_relationship_changes(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how an interaction changes relationship stats."""
        changes = {"trust": 0, "respect": 0, "closeness": 0}
        
        interaction_type = interaction.get("type")
        
        # Positive interactions increase stats
        if interaction_type in ["friendly_chat", "introduction"]:
            changes["trust"] += random.randint(1, 3)
            changes["closeness"] += random.randint(1, 3)
        elif interaction_type == "formal_discussion":
            changes["respect"] += random.randint(1, 3)
            changes["trust"] += 1
        elif interaction_type == "cautious_exchange":
            changes["trust"] += random.randint(-1, 1)
            changes["respect"] += random.randint(0, 1)
        
        return changes

    async def _generate_work_activity(self, work_type: str, location: str) -> Dict[str, Any]:
        """LLM-based work activity description with fallback."""
        try:
            npc_name = self.context.current_stats.get("npc_name", f"NPC_{self.npc_id}")
            
            if activity_generator:
                payload = json.dumps({
                    "activity_kind": "work",
                    "subtype": work_type,
                    "location": location,
                    "npc_name": npc_name
                })
                res = await Runner.run(activity_generator, payload)
                desc = _safe_json_loads(res.output).get("description", f"working at {location}")
            else:
                desc = await _fallback_generate_activity("work", work_type, location, npc_name)
        except Exception as e:
            logger.warning(f"Work activity fallback: {e}")
            desc = f"working at {location}"

        return {
            "type": work_type,
            "description": desc,
            "duration": random.randint(30, 120)
        }

    async def _generate_relaxation_activity(self, relax_type: str, location: str) -> Dict[str, Any]:
        """LLM-based relaxation activity description with fallback."""
        try:
            npc_name = self.context.current_stats.get("npc_name", f"NPC_{self.npc_id}")
            
            if activity_generator:
                payload = json.dumps({
                    "activity_kind": "relax",
                    "subtype": relax_type,
                    "location": location,
                    "npc_name": npc_name
                })
                res = await Runner.run(activity_generator, payload)
                desc = _safe_json_loads(res.output).get("description", f"relaxing at {location}")
            else:
                desc = await _fallback_generate_activity("relax", relax_type, location, npc_name)
        except Exception as e:
            logger.warning(f"Relax activity fallback: {e}")
            desc = f"relaxing at {location}"

        return {
            "type": relax_type,
            "description": desc,
            "duration": random.randint(15, 60)
        }
        
    async def _update_work_stats(self, work_activity: Dict[str, Any]) -> None:
        """Update NPC stats based on work activity."""
        # Work can affect intensity and other stats
        try:
            duration = work_activity.get("duration", 60)
            
            # Longer work increases intensity/stress
            intensity_change = duration / 30  # 2 points per hour
            
            if self.context.current_stats:
                current_intensity = self.context.current_stats.get("intensity", 50)
                new_intensity = min(100, current_intensity + intensity_change)
                self.context.current_stats["intensity"] = new_intensity
                
        except Exception as e:
            logger.error(f"Error updating work stats: {e}")

    async def _update_relaxation_stats(self, relax_activity: Dict[str, Any]) -> None:
        """Update NPC stats based on relaxation activity."""
        try:
            duration = relax_activity.get("duration", 30)
            
            # Relaxation reduces intensity/stress
            intensity_change = -(duration / 20)  # -3 points per hour
            
            if self.context.current_stats:
                current_intensity = self.context.current_stats.get("intensity", 50)
                new_intensity = max(0, current_intensity + intensity_change)
                self.context.current_stats["intensity"] = new_intensity
                
        except Exception as e:
            logger.error(f"Error updating relaxation stats: {e}")

    async def _update_location(self, new_location: str) -> None:
        """Update NPC's current location."""
        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET current_location = $1
                    WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                    """,
                    new_location, self.npc_id, self.user_id, self.conversation_id
                )
            if self.context.current_stats:
                self.context.current_stats["current_location"] = new_location
        except Exception as e:
            logger.error(f"Error updating location: {e}")

    def _choose_interaction_target(self, nearby_npcs: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an NPC to interact with."""
        try:
            # Get NPC's traits from current stats
            npc_traits = self.context.current_stats or {}
            
            # Calculate interaction scores
            scores = []
            for npc in nearby_npcs:
                score = 0
                
                # Base score on relationship if exists
                relationship = context.get("relationships", {}).get(str(npc["npc_id"]))
                if relationship:
                    score += relationship.get("trust", 0) * 2
                
                # Adjust based on personality compatibility
                if npc_traits.get("dominance", 50) > 60 and npc.get("dominance", 50) < 40:
                    score += 1  # Prefer submissive NPCs
                elif npc_traits.get("dominance", 50) < 40 and npc.get("dominance", 50) > 60:
                    score += 1  # Prefer dominant NPCs
                
                # Random factor
                score += random.random()
                
                scores.append((score, npc))
            
            # Sort by score and choose top NPC
            scores.sort(reverse=True)
            return scores[0][1] if scores else None
            
        except Exception as e:
            logger.error(f"Error choosing interaction target: {e}")
            return None

    async def _generate_interaction(self, target_npc: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an interaction with another NPC."""
        try:
            # Get relationship history
            relationship = await self._get_relationship(target_npc["npc_id"])
            
            # Generate interaction based on relationship and context
            interaction_type = self._determine_interaction_type(relationship, context)
            interaction_details = await self._generate_interaction_details(
                interaction_type,
                target_npc,
                relationship,
                context
            )
            
            return {
                "type": interaction_type,
                "details": interaction_details,
                "target_npc": target_npc["npc_name"],
                "location": context["location"]
            }
            
        except Exception as e:
            logger.error(f"Error generating interaction: {e}")
            return None

    async def _update_relationship_after_interaction(
        self,
        target_npc: Dict[str, Any],
        interaction: Dict[str, Any]
    ) -> None:
        """Update relationships based on interaction."""
        try:
            # Calculate relationship changes
            changes = self._calculate_relationship_changes(interaction)
            
            # Update relationship in database
            await self._update_relationship_in_db(target_npc["npc_id"], changes)
            
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

    def _importance_label(self, val: float) -> str:
        return "high" if val >= 0.85 else "medium" if val >= 0.55 else "low"

    async def _record_activity(self, action: Dict[str, Any], result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Record an activity in memory."""
        try:
            memory_system = await self.context.get_memory_system()
            
            # Create memory text
            memory_text = self._create_memory_text(action, result, context)
            
            # Store memory
            await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance=self._importance_label(self._calculate_memory_importance(action, result)),
                tags=["activity", action.get("type", "unknown")]
            )
            
        except Exception as e:
            logger.error(f"Error recording activity: {e}")

    def _create_memory_text(self, action: Dict[str, Any], result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create text for a memory of an activity."""
        try:
            action_type = action.get("type")
            location = context.get("location", "unknown location")
            time = context.get("time_of_day", "unknown time")
            
            if action_type == "socialize":
                target = result.get("interaction", {}).get("target_npc", "someone")
                interaction = result.get("interaction", {}).get("details", "had an interaction")
                return f"At {time} in {location}, I {interaction} with {target}."
            elif action_type == "work":
                activity = result.get("work_activity", {}).get("description", "worked")
                return f"At {time} in {location}, I {activity}."
            elif action_type == "relax":
                activity = result.get("relax_activity", {}).get("description", "relaxed")
                return f"At {time} in {location}, I {activity}."
            elif action_type == "travel":
                destination = result.get("destination", "somewhere")
                return f"At {time}, I traveled to {destination}."
            else:
                return f"At {time} in {location}, I performed an activity."
                
        except Exception as e:
            logger.error(f"Error creating memory text: {e}")
            return "Had an activity."

    def _calculate_memory_importance(self, action: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Calculate importance of a memory."""
        try:
            importance = 0.5  # Base importance
            
            # Adjust based on action type
            action_type = action.get("type")
            if action_type == "socialize":
                importance += 0.2  # Social interactions are more memorable
            elif action_type == "work":
                importance += 0.1  # Work activities are moderately memorable
            elif action_type == "relax":
                importance += 0.05  # Relaxation is less memorable
            
            # Adjust based on result
            if result.get("status") == "success":
                importance += 0.1  # Successful activities are more memorable
            
            # Cap at 1.0
            return min(1.0, importance)
            
        except Exception as e:
            logger.error(f"Error calculating memory importance: {e}")
            return 0.5

    async def _get_npc_data(self, npc_id: int) -> Dict[str, Any]:
        """Get NPC data from database."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """,
                    npc_id, self.user_id, self.conversation_id
                )
            return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Error getting NPC data: {e}")
            return {}
