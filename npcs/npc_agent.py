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
from db.connection import get_db_connection
from memory.wrapper import MemorySystem

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
    Execute the chosen NPC action.
    
    Args:
        action: The action to execute
        context: Additional context information
    """
    with function_span("execute_npc_action"):
        perf_start = time.perf_counter()
        context = context or {}
        
        try:
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
            
            # Record performance
            elapsed = time.perf_counter() - perf_start
            ctx.context.perf_metrics['action_time'].append(elapsed)
            
            return ActionResult(
                outcome=outcome,
                emotional_impact=emotional_impact,
                target_reactions=target_reactions
            )
            
        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Error executing action after {elapsed:.2f}s: {e}")
            
            return ActionResult(
                outcome=f"NPC attempted to {action.description} but encountered an error: {str(e)}",
                emotional_impact=-1
            )

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

    async def initialize(self):
        """
        Initialize the NPC agent by loading basic data.
        """
        # Load NPC stats
        await get_npc_stats(RunContextWrapper(self.context))
        
        # Initialize memory system
        await self.context.get_memory_system()
        
        return self

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
    
    async def make_decision(self, perception_context: Dict[str, Any] = None) -> NPCAction:
        """Make a decision about what action to take."""
        # CHANGE: Always check for Nyx directives first
        nyx_directive = await self._get_current_nyx_directive()
        
        if nyx_directive:
            logger.info(f"NPC {self.npc_id} following Nyx directive: {nyx_directive.get('description', 'unknown')}")
            return NPCAction(
                type=nyx_directive.get("type", "observe"),
                description=nyx_directive.get("description", "follow Nyx's directive"),
                target=nyx_directive.get("target", "environment"),
                weight=1.0,
                decision_metadata={"source": "nyx_directive"}
            )
            
        # If no directive or the directive doesn't specify an action, proceed with normal decision-making

        with trace(workflow_name=f"NPC {self.npc_id} Decision"):
            perf_start = time.perf_counter()
            
            try:
                # Make decision using the decision agent with handoff
                result = await Runner.run(
                    decision_agent,
                    f"Make a decision for NPC {self.npc_id}",
                    context=self.context
                )
                
                decision = result.final_output
                if isinstance(decision, dict):
                    action = NPCAction(**decision)
                else:
                    # If output is not properly structured, create a default action
                    action = NPCAction(
                        type="observe",
                        description="observe the surroundings",
                        target="environment"
                    )
                
                elapsed = time.perf_counter() - perf_start
                self.context.perf_metrics['decision_time'].append(elapsed)
                
                return action
                
            except Exception as e:
                elapsed = time.perf_counter() - perf_start
                logger.error(f"Decision error for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
                
                # Return a default action if decision fails
                return NPCAction(
                    type="observe",
                    description="observe quietly",
                    target="environment"
                )

    async def _get_current_nyx_directive(self) -> Optional[Dict[str, Any]]:
        """Retrieve current directive from Nyx for this NPC."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT directive FROM NyxNPCDirectives
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        AND created_at > NOW() - INTERVAL '10 minutes'
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        self.user_id, self.conversation_id, self.npc_id
                    )
                    
                    if row and row["directive"]:
                        return json.loads(row["directive"])
            return None
        except Exception as e:
            logger.error(f"Error retrieving Nyx directive: {e}")
            return None
    
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

    # Add to NPCAgent class in npcs/npc_agent.py
    async def report_action_to_nyx(self, action: NPCAction, result: ActionResult) -> None:
        """Report significant NPC action to Nyx for awareness and potential override."""
        try:
            # Only report significant actions
            if action.weight < 0.5 and not result.emotional_impact:
                return
                
            # Import here to avoid circular imports
            from nyx.integrate import NyxNPCIntegrationManager
            
            nyx_manager = NyxNPCIntegrationManager(self.user_id, self.conversation_id)
            
            # Report the action
            await nyx_manager.process_npc_action_report({
                "npc_id": self.npc_id,
                "action": action.model_dump(),
                "result": result.model_dump(),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error reporting action to Nyx: {e}")
