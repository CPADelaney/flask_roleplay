# npcs/npc_agent.py

"""
Core NPC agent implementation using OpenAI Agents SDK.
"""

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
from pydantic import BaseModel, Field, validator
from collections import OrderedDict

from agents import Agent, Runner, RunContextWrapper, trace, function_tool, handoff
from agents.tracing import custom_span, generation_span, function_span
from db.connection import get_db_connection_context  # Updated import
from memory.wrapper import MemorySystem
from .lore_context_manager import LoreContextManager

logger = logging.getLogger(__name__)

class ResourcePool:
    """Manages shared resources with limits to prevent overwhelming systems."""
    
    def __init__(self, max_concurrent=10, timeout=30.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.stats = {
            "total_requests": 0,
            "timeouts": 0,
            "current_usage": 0,
            "peak_usage": 0
        }
    
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
async def get_npc_stats(ctx: RunContextWrapper[NPCContext]) -> NPCStats:
    """
    Get the NPC's stats and traits from the database.
    """
    with custom_span("get_npc_stats"):
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        # Updated to use async DB connection
        async with get_db_connection_context() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                           hobbies, personality_traits, likes, dislikes, schedule, current_location, sex
                    FROM NPCStats
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """,
                    (npc_id, user_id, conversation_id),
                )
                row = await cursor.fetchone()
                if not row:
                    return NPCStats(npc_id=npc_id, npc_name=f"NPC_{npc_id}")

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
        ctx.context.current_stats = stats.model_dump()
        return stats

@function_tool
async def execute_npc_action(
    ctx: RunContextWrapper[NPCContext],
    action: NPCAction,
    context: Dict[str, Any] = None
) -> ActionResult:
    """
    Execute the chosen NPC action with Nyx governance.
    
    Args:
        action: The action to execute
        context: Additional context information
    """
    with function_span("execute_npc_action"):
        perf_start = time.perf_counter()
        context = context or {}
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        try:
            # Check with Nyx governor before executing
            governor = await get_nyx_governor(user_id, conversation_id)
            
            # Skip permission check for directive-sourced actions
            if not action.decision_metadata or action.decision_metadata.get("source") != "nyx_directive":
                permission = await governor.check_action_permission(
                    npc_id=npc_id,
                    action_type=action.type,
                    action_details=action.model_dump(),
                    context=context
                )
                
                # If not approved, replace with override or default action
                if not permission.get("approved", True):
                    logger.info(f"Nyx rejected action for NPC {npc_id}: {action.type} - {action.description}")
                    
                    if permission.get("override_action"):
                        # Use Nyx's override
                        override = permission["override_action"]
                        action = NPCAction(
                            type=override.get("type", "observe"),
                            description=override.get("description", "follow Nyx's guidance"),
                            target=override.get("target", "environment"),
                            weight=1.0,
                            decision_metadata={"source": "nyx_override"}
                        )
                    else:
                        # Default safe action if no override provided
                        action = NPCAction(
                            type="observe",
                            description="observe quietly as directed by Nyx",
                            target="environment",
                            weight=1.0,
                            decision_metadata={"source": "nyx_restriction"}
                        )
            
            # Record action
            ctx.context.record_decision(action.model_dump())
            
            # For now, simulate an action result
            # In a real implementation, this would interact with game state
            outcome_templates = {
                "talk": [
                    "NPC engages in conversation about {topic}",
                    "NPC discusses {topic} with {target}",
                    "NPC shares thoughts on {topic}"
                ],
                "observe": [
                    "NPC quietly observes the situation",
                    "NPC watches carefully, taking mental notes",
                    "NPC studies {target} with interest"
                ],
                "leave": [
                    "NPC exits the location",
                    "NPC walks away from {target}",
                    "NPC decides to depart"
                ],
                "command": [
                    "NPC firmly orders {target} to {action}",
                    "NPC commands {target} with authority",
                    "NPC gives a direct order to {target}"
                ],
                "dominate": [
                    "NPC asserts complete dominance over {target}",
                    "NPC takes control of the situation forcefully",
                    "NPC demonstrates overwhelming authority"
                ],
                "mock": [
                    "NPC mocks {target} cruelly",
                    "NPC makes cutting remarks about {target}",
                    "NPC belittles {target} with harsh words"
                ],
                "emotional_outburst": [
                    "NPC has an emotional outburst about {topic}",
                    "NPC expresses powerful emotions",
                    "NPC loses emotional control momentarily"
                ]
            }
            
            # Select template for the action type
            action_type = action.type
            templates = outcome_templates.get(action_type, ["NPC performs {action}"])
            template = random.choice(templates)
            
            # Prepare format variables
            format_vars = {
                "action": action.description,
                "target": action.target,
                "topic": context.get("topic", "the current situation")
            }
            
            # Format the outcome
            outcome = template.format(**format_vars)
            
            # Determine emotional impact
            base_emotional_impact = random.randint(-3, 3)
            emotion_modifiers = {
                "praise": 2,
                "mock": -2,
                "talk": 1,
                "dominate": -1,
                "support": 2,
                "humiliate": -3
            }
            modifier = emotion_modifiers.get(action_type, 0)
            emotional_impact = base_emotional_impact + modifier
            
            # Simulate target reactions
            target_reactions = []
            
            # If action is directed at player or NPCs, simulate reaction
            if action.target in ["player", "group"] or action.target.isdigit():
                reaction_text = f"Reacts to {action.description}"
                if emotional_impact > 2:
                    reaction_text = f"Responds positively to {action.description}"
                elif emotional_impact < -2:
                    reaction_text = f"Responds negatively to {action.description}"
                    
                target_reactions.append({
                    "entity": action.target,
                    "reaction": reaction_text,
                    "intensity": abs(emotional_impact) / 3.0  # 0.0-1.0 scale
                })
            
            # Create memory of the action if significant
            if abs(emotional_impact) > 2 or action_type in ["dominate", "praise", "mock", "command"]:
                memory_system = await ctx.context.get_memory_system()
                
                memory_text = f"I {action.description} resulting in {outcome}"
                
                # Determine tags
                tags = [action_type, "action"]
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
            
            # Create the result
            result = ActionResult(
                outcome=outcome,
                emotional_impact=emotional_impact,
                target_reactions=target_reactions
            )
            
            # Record performance
            elapsed = time.perf_counter() - perf_start
            ctx.context.perf_metrics['action_time'].append(elapsed)
            
            # Report to Nyx - critical for governance
            try:
                await report_action_to_nyx(ctx, action, result)
            except Exception as reporting_error:
                logger.error(f"Error reporting to Nyx: {reporting_error}")
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Error executing action after {elapsed:.2f}s: {e}")
            
            return ActionResult(
                outcome=f"NPC attempted to {action.description} but encountered an error: {str(e)}",
                emotional_impact=-1
            )

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
        from nyx.integrate import NyxNPCIntegrationManager
        
        nyx_manager = NyxNPCIntegrationManager(
            ctx.context.user_id, 
            ctx.context.conversation_id
        )
        
        # Report the action
        await nyx_manager.process_npc_action_report({
            "npc_id": ctx.context.npc_id,
            "action": action.model_dump(),
            "result": result.model_dump(),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.debug(f"Reported action of NPC {ctx.context.npc_id} to Nyx: {action.type}")
    except Exception as e:
        logger.error(f"Error reporting action to Nyx: {e}")


# -------------------------------------------------------
# Decision-related agents
# -------------------------------------------------------

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
    tools=[
        get_npc_stats,
        execute_npc_action
    ]
)

# Main NPC Agent
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
    handoffs=[
        handoff(
            agent=decision_agent,
            tool_name_override="on",
            tool_description_override="Make a decision about what action to take"
        )
    ],
    tools=[
        get_npc_stats,
        execute_npc_action
    ]
)

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
        # Get basic story context
        story_context = await super()._get_story_context()
        
        # Get enhanced lore context
        lore_context = await self.lore_context_manager.get_lore_context(
            self.npc_id, 
            "story_context"
        )
        
        # Merge contexts
        story_context.update(lore_context)
        return story_context
        
    async def _determine_npc_role(self) -> Dict[str, Any]:
        """Determine NPC role with lore-aware analysis."""
        # Get basic role determination
        role = await super()._determine_npc_role()
        
        # Get lore context for role
        lore_context = await self.lore_context_manager.get_lore_context(
            self.npc_id,
            "role_context"
        )
        
        # Enhance role with lore context
        role.update(lore_context)
        return role
        
    async def _process_lore_change(self, lore_change: Dict[str, Any]) -> Dict[str, Any]:
        """Process a lore change and update NPC state."""
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
        """Get list of NPCs affected by a lore change."""
        # Implementation would determine affected NPCs based on:
        # - NPC relationships
        # - NPC knowledge levels
        # - Lore change scope
        pass
        
    async def _update_state_from_impact(self, impact: Dict[str, Any]):
        """Update NPC state based on lore impact analysis."""
        # Update beliefs
        if impact["belief_updates"]:
            await self._update_beliefs(impact["belief_updates"])
            
        # Update relationships
        if impact["relationship_impacts"]:
            await self._update_relationships(impact["relationship_impacts"])
            
        # Update behavior
        if impact["behavior_changes"]:
            await self._update_behavior(impact["behavior_changes"])

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
        Process a scene directive from Nyx.
        
        Args:
            directive_data: The directive data
            context: Additional context
            
        Returns:
            Result of the directive execution
        """
        context = context or {}
        context["source"] = "nyx_scene_directive"
        
        # Convert directive to action
        action_type = directive_data.get("type", "action")
        description = directive_data.get("description", "follow scene directive")
        target = directive_data.get("target", "scene")
        
        action = NPCAction(
            type=action_type,
            description=description,
            target=target,
            weight=1.0,
            decision_metadata={
                "source": "nyx_scene_directive",
                "scene_id": directive_data.get("scene_id")
            }
        )
        
        # Execute the action
        result = await execute_npc_action(
            RunContextWrapper(self.context),
            action,
            context
        )
        
        # Provide feedback to Nyx about directive completion
        try:
            # Report completion to Nyx
            from nyx.integrate import NyxNPCIntegrationManager
            
            nyx_manager = NyxNPCIntegrationManager(
                self.user_id, 
                self.conversation_id
            )
            
            await nyx_manager.report_directive_completion(
                self.npc_id,
                directive_data.get("id"),
                {
                    "action": action.model_dump(),
                    "result": result.model_dump(),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error reporting directive completion: {e}")
        
        return result

    
    async def perceive_environment(self, context: Dict[str, Any] = None) -> NPCPerception:
        """
        Perceive the environment and retrieve relevant memories.
        
        Args:
            context: Optional context information
            
        Returns:
            NPCPerception object with environment and memory data
        """
        # In the future, this will call into perception.py module
        # For now, return a minimal perception
        environment = context or {}
        
        memory_system = await self.context.get_memory_system()
        
        # Get emotional state
        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
        
        # Create basic perception
        perception = NPCPerception(
            environment=environment,
            emotional_state=emotional_state or {},
            mask={"integrity": 100},  # Default mask integrity
            time_context={"time_of_day": environment.get("time_of_day", "unknown")}
        )
        
        return perception
    
    async def process_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a player action directed at this NPC.
        
        Args:
            player_action: The player's action
            context: Additional context information
            
        Returns:
            NPC's response to the action
        """
        with trace(workflow_name=f"NPC {self.npc_id} Process Player Action"):
            context_obj = context or {}
            
            try:
                # Create perception context
                perception_context = {
                    "player_action": player_action,
                    "text": player_action.get("description", ""),
                    "description": f"Player {player_action.get('description', 'did something')}"
                }
                perception_context.update(context_obj)
                
                # Perceive environment
                perception = await self.perceive_environment(perception_context)
                
                # Make a decision
                action = await self.make_decision(perception_context)
                
                # Execute the action
                result = await execute_npc_action(
                    RunContextWrapper(self.context),
                    action,
                    perception_context
                )
                
                # Create and return response
                return {
                    "npc_id": self.npc_id,
                    "action": action.model_dump(),
                    "result": result.model_dump()
                }
                
            except Exception as e:
                logger.error(f"Error processing player action for NPC {self.npc_id}: {e}")
                
                # Return basic error response
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
        try:
            # Get lore system instance
            lore_system = await get_lore_system(self.user_id, self.conversation_id)
            
            # Get base speech patterns
            patterns = await lore_system.get_npc_speech_patterns(self.npc_id)
            
            # Enhance with NPC-specific traits
            if self.traits:
                for trait in self.traits:
                    if trait.get('type') == 'speech':
                        patterns['traits'].append(trait)
            
            # Add to memory for future reference
            await self.memory_system.add_memory(
                self.run_ctx,
                f"Speech patterns for NPC {self.npc_id}",
                patterns,
                memory_type="npc_speech"
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
            current_location = await self._get_current_location()
            
            # Create activity context
            activity_context = {
                "time_of_day": time_data["time_of_day"],
                "location": current_location,
                "schedule": schedule,
                "npc_id": self.npc_id,
                "npc_name": self.context.npc_name
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
            action = await decision_engine.decide(activity_context)
            
            # Execute the action
            result = await self._execute_action(action, activity_context)
            
            # Record the activity in memory
            await self._record_activity(action, result, activity_context)
            
            return {
                "action": action,
                "result": result,
                "context": activity_context
            }
            
        except Exception as e:
            logger.error(f"Error performing scheduled activity: {e}")
            return None

    async def _get_current_schedule(self) -> Dict[str, Any]:
        """Get the NPC's current schedule."""
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT schedule
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (self.npc_id, self.user_id, self.conversation_id))
                    
                    row = await cursor.fetchone()
                    
                    if row and row[0]:
                        return row[0]
                    return None
        except Exception as e:
            logger.error(f"Error getting NPC schedule: {e}")
            return None

    async def _get_current_time(self) -> Dict[str, Any]:
        """Get current game time information."""
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT year, month, day, time_of_day
                        FROM GameTime
                        WHERE user_id = %s AND conversation_id = %s
                    """, (self.user_id, self.conversation_id))
                    
                    row = await cursor.fetchone()
                    
                    if row:
                        return {
                            "year": row[0],
                            "month": row[1],
                            "day": row[2],
                            "time_of_day": row[3]
                        }
                    return None
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return None

    async def _get_current_location(self) -> str:
        """Get NPC's current location."""
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT current_location
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (self.npc_id, self.user_id, self.conversation_id))
                    
                    row = await cursor.fetchone()
                    
                    if row:
                        return row[0]
                    return "unknown"
        except Exception as e:
            logger.error(f"Error getting NPC location: {e}")
            return "unknown"


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
            await self._update_relationships(target_npc, interaction)
            
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
        """Get NPCs in the same location."""
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT npc_id, npc_name, dominance, cruelty 
                        FROM NPCStats
                        WHERE user_id = %s AND conversation_id = %s
                        AND current_location = %s
                        AND npc_id != %s
                    """, (self.user_id, self.conversation_id, location, self.npc_id))
                    
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting nearby NPCs: {e}")
            return []


    def _choose_interaction_target(self, nearby_npcs: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an NPC to interact with."""
        try:
            # Get NPC's traits
            npc_traits = self.context.npc_traits
            
            # Calculate interaction scores
            scores = []
            for npc in nearby_npcs:
                score = 0
                
                # Base score on relationship if exists
                relationship = context.get("relationships", {}).get(str(npc["npc_id"]))
                if relationship:
                    score += relationship.get("trust", 0) * 2
                
                # Adjust based on personality compatibility
                if npc_traits.get("dominance") > 60 and npc["dominance"] < 40:
                    score += 1  # Prefer submissive NPCs
                elif npc_traits.get("dominance") < 40 and npc["dominance"] > 60:
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

    async def _update_relationships(self, target_npc: Dict[str, Any], interaction: Dict[str, Any]) -> None:
        """Update relationships based on interaction."""
        try:
            # Calculate relationship changes
            changes = self._calculate_relationship_changes(interaction)
            
            # Update relationship in database
            await self._update_relationship_in_db(target_npc["npc_id"], changes)
            
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

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
                importance=self._calculate_memory_importance(action, result),
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
                target = result.get("target_npc", "someone")
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
