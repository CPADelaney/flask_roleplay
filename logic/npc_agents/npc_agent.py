# npc_agents/npc_agent_sdk.py

"""
Core NPC agent implementation using OpenAI Agents SDK.
Replaces the original npc_agent.py with the Agent SDK architecture.
"""

import logging
import json
import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict
from pydantic import BaseModel, Field
import os
import psutil
import gc
from collections import OrderedDict

from agents import Agent, Runner, RunContextWrapper, trace, function_tool, handoff
from agents.tracing import custom_span, generation_span, function_span
from db.connection import get_db_connection

# Import memory subsystem components
from memory.core import Memory, MemoryType, MemorySignificance
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
        self.cache_timestamps = {
            'perception': {},
            'memories': {},
            'emotional_state': None,
            'mask': None
        }
        self.cache_ttls = {
            'perception': timedelta(minutes=5),
            'memories': timedelta(minutes=10),
            'emotional_state': timedelta(minutes=2),
            'mask': timedelta(minutes=5)
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
# Tool Functions
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
async def perceive_environment(
    ctx: RunContextWrapper[NPCContext], 
    current_context: Dict[str, Any],
    detail_level: str = "auto"  # "low", "standard", "high", "auto"
) -> NPCPerception:
    """Perceive environment with adaptive detail levels based on system load."""
    
    # If auto, determine detail level based on system load
    if detail_level == "auto":
        system_load = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        if system_load > 80 or mem_usage > 85:
            detail_level = "low"
        elif system_load > 60 or mem_usage > 70:
            detail_level = "standard"
        else:
            detail_level = "high"
    
    # Process based on detail level
    if detail_level == "low":
        # Minimal perception - fewer memories, no flashbacks
        # Limit memory retrieval to 3 items
        memory_limit = 3
        include_flashbacks = False
    elif detail_level == "high":
        # Detailed perception - more memories, flashbacks, detailed relationships
        memory_limit = 8
        include_flashbacks = True
    else:
        # Standard processing
        memory_limit = 5
        include_flashbacks = random.random() < 0.3
    with function_span("perceive_environment"):
        perf_start = time.perf_counter()
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        try:
            # Create cache key based on context
            context_key = str(hash(json.dumps(current_context, sort_keys=True, default=str)))
            
            # Check cache
            if ctx.context.is_cache_valid('perception', context_key):
                ctx.context.perf_metrics['cache_hits'] += 1
                perception = ctx.context.cache['perception'][context_key]
                ctx.context.last_perception = perception
                return NPCPerception(**perception)
            
            ctx.context.perf_metrics['cache_misses'] += 1
            
            # Get environment data
            environment_data = await fetch_environment_data(user_id, conversation_id, current_context)
            
            # Get memory system for retrieving memories
            memory_system = await ctx.context.get_memory_system()
            
            # Construct text description for memory retrieval
            context_description = current_context.get("description", "")
            if "text" in current_context:
                context_description += " " + current_context["text"]
                
            # Create context for memory recall
            context_for_recall = {
                "text": context_description,
                "location": environment_data.get("location", "Unknown"),
                "time_of_day": environment_data.get("time_of_day", "Unknown")
            }
            
            # Add entities present
            if "entities_present" in environment_data:
                context_for_recall["entities_present"] = [
                    e.get("name", "") for e in environment_data.get("entities_present", [])
                ]
            
            # Get emotional state
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Include emotional state in context if strong
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                emotion_name = current_emotion.get("primary", {}).get("name", "neutral")
                intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
                
                if intensity > 0.6:
                    context_for_recall["emotional_state"] = {
                        "primary_emotion": emotion_name,
                        "intensity": intensity
                    }
            
            # Determine adaptive memory limit
            base_limit = 5
            adaptive_limit = base_limit
            context_importance = 0
            
            keywords_high = ["critical", "emergency", "dangerous", "threat", "crucial", "sex", "intimate"]
            keywords_medium = ["important", "significant", "unusual", "strange", "unexpected"]
            
            lowered_desc = context_description.lower()
            for w in keywords_high:
                if w in lowered_desc:
                    context_importance += 2
            for w in keywords_medium:
                if w in lowered_desc:
                    context_importance += 1
                    
            if context_importance >= 3:
                adaptive_limit = base_limit + 5
            elif context_importance >= 1:
                adaptive_limit = base_limit + 2
                
            # Retrieve memories
            with generation_span(model="recall_memory"):
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=npc_id,
                    context=context_for_recall,
                    limit=adaptive_limit
                )
                relevant_memories = memory_result.get("memories", [])
            
            # Check for traumatic triggers
            traumatic_trigger = None
            if context_description:
                trigger_result = await memory_system.emotional_manager.process_traumatic_triggers(
                    entity_type="npc",
                    entity_id=npc_id,
                    text=context_description
                )
                if trigger_result and trigger_result.get("triggered", False):
                    traumatic_trigger = trigger_result
                    # Update emotional state if triggered
                    if "emotional_response" in trigger_result:
                        response = trigger_result["emotional_response"]
                        await memory_system.update_npc_emotion(
                            npc_id=npc_id,
                            emotion=response.get("primary_emotion", "fear"),
                            intensity=response.get("intensity", 0.7)
                        )
                        emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Check for flashback
            flashback = None
            flashback_chance = 0.15
            if traumatic_trigger:
                flashback_chance = 0.5
                
            if random.random() < flashback_chance:
                flashback = await memory_system.npc_flashback(
                    npc_id=npc_id,
                    context=context_description
                )
                
            # Get mask info
            mask_info = await memory_system.get_npc_mask(npc_id)
            if mask_info and "integrity" not in mask_info:
                mask_info["integrity"] = 100
                
            # Get relationships
            relationship_data = await fetch_relationships_with_memory(ctx)
            
            # Get beliefs
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic="player"
            )
            if beliefs:
                beliefs = sorted(beliefs, key=lambda x: x.get("confidence", 0), reverse=True)
                
            # Get time context
            time_context = await fetch_time_context(ctx)
            
            # Get narrative context
            narrative_context = await fetch_narrative_context(ctx)
            
            # Build final perception
            perception = {
                "environment": environment_data,
                "relevant_memories": relevant_memories,
                "flashback": flashback,
                "relationships": relationship_data,
                "emotional_state": emotional_state,
                "traumatic_trigger": traumatic_trigger,
                "mask": mask_info,
                "beliefs": beliefs,
                "time_context": time_context,
                "narrative_context": narrative_context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the perception
            await ctx.context.update_cache('perception', context_key, perception)
            ctx.context.last_perception = perception
            
            # Record performance
            elapsed = time.perf_counter() - perf_start
            ctx.context.perf_metrics['perception_time'].append(elapsed)
            
            # Warn if slow
            if elapsed > 0.5:
                logger.warning(f"Slow perception for NPC {npc_id}: {elapsed:.2f}s")
                
            return NPCPerception(**perception)
            
        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Perception error for NPC {npc_id} after {elapsed:.2f}s: {e}")
            
            # Return minimal perception
            return NPCPerception(
                environment=environment_data if 'environment_data' in locals() else {},
                emotional_state={
                    "current_emotion": {
                        "primary": {"name": "neutral", "intensity": 0.0}
                    }
                },
                mask={"integrity": 100}
            )

@function_tool
async def fetch_environment_data(
    user_id: int, 
    conversation_id: int, 
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fetch environment data from the database based on context.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        context: Current context information
    """
    with function_span("fetch_environment_data", parent=None):
        try:
            environment_data = {
                "location": "Unknown",
                "time_of_day": "Unknown",
                "entities_present": []
            }
            
            # Extract location from context if available
            if "location" in context:
                environment_data["location"] = context["location"]
                
            # Fetch location from database if not in context
            if environment_data["location"] == "Unknown":
                def fetch_location():
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT value FROM CurrentRoleplay 
                            WHERE key = 'CurrentLocation' AND user_id = %s AND conversation_id = %s
                            """,
                            (user_id, conversation_id)
                        )
                        row = cursor.fetchone()
                        return row[0] if row else "Unknown"
                
                location = await asyncio.to_thread(fetch_location)
                environment_data["location"] = location
                
            # Fetch time of day
            def fetch_time():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT value FROM CurrentRoleplay 
                        WHERE key = 'TimeOfDay' AND user_id = %s AND conversation_id = %s
                        """,
                        (user_id, conversation_id)
                    )
                    row = cursor.fetchone()
                    return row[0] if row else "Unknown"
            
            time_of_day = await asyncio.to_thread(fetch_time)
            environment_data["time_of_day"] = time_of_day
            
            # Fetch entities present
            def fetch_entities():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    # Fetch NPCs
                    cursor.execute(
                        """
                        SELECT npc_id, npc_name FROM NPCStats
                        WHERE current_location = %s AND user_id = %s AND conversation_id = %s
                        """,
                        (environment_data["location"], user_id, conversation_id)
                    )
                    entities = []
                    for npc_id, npc_name in cursor.fetchall():
                        entities.append({
                            "type": "npc",
                            "id": npc_id,
                            "name": npc_name
                        })
                    
                    # Add player entity
                    entities.append({
                        "type": "player",
                        "id": user_id,
                        "name": "Player"
                    })
                    
                    return entities
            
            entities = await asyncio.to_thread(fetch_entities)
            environment_data["entities_present"] = entities
            
            return environment_data
            
        except Exception as e:
            logger.error(f"Error fetching environment data: {e}")
            return {
                "location": "Unknown",
                "time_of_day": "Unknown",
                "entities_present": [],
                "error": str(e)
            }

@function_tool
async def fetch_time_context(ctx: RunContextWrapper[NPCContext]) -> Dict[str, Any]:
    """
    Fetch the current time context (year, month, day, time of day) from the database.
    """
    with function_span("fetch_time_context"):
        time_context = {
            "year": None,
            "month": None,
            "day": None,
            "time_of_day": None
        }
        
        try:
            def fetch_context():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT key, value 
                        FROM CurrentRoleplay 
                        WHERE key IN ('CurrentYear', 'CurrentMonth', 'CurrentDay', 'TimeOfDay')
                          AND user_id = %s 
                          AND conversation_id = %s
                        """,
                        (ctx.context.user_id, ctx.context.conversation_id),
                    )
                    return cursor.fetchall()
                    
            rows = await asyncio.to_thread(fetch_context)
            for key, value in rows:
                if key == "CurrentYear":
                    time_context["year"] = value
                elif key == "CurrentMonth":
                    time_context["month"] = value
                elif key == "CurrentDay":
                    time_context["day"] = value
                elif key == "TimeOfDay":
                    time_context["time_of_day"] = value
            
            return time_context
            
        except Exception as e:
            logger.error(f"Error fetching time context: {e}")
            return time_context

@function_tool
async def fetch_narrative_context(ctx: RunContextWrapper[NPCContext]) -> Dict[str, Any]:
    """
    Fetch the current narrative context (plot stage, tension) from the database.
    """
    with function_span("fetch_narrative_context"):
        narrative_context = {}
        
        try:
            def fetch_context():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT key, value 
                        FROM CurrentRoleplay 
                        WHERE key IN ('CurrentPlotStage', 'CurrentTension')
                          AND user_id = %s 
                          AND conversation_id = %s
                        """,
                        (ctx.context.user_id, ctx.context.conversation_id),
                    )
                    return cursor.fetchall()
                    
            rows = await asyncio.to_thread(fetch_context)
            for key, value in rows:
                narrative_context[key] = value
            
            return narrative_context
            
        except Exception as e:
            logger.error(f"Error fetching narrative context: {e}")
            return narrative_context

@function_tool
async def fetch_relationships_with_memory(ctx: RunContextWrapper[NPCContext]) -> Dict[str, Any]:
    """
    Fetch relationships enriched with memory references.
    """
    with function_span("fetch_relationships"):
        relationships = {}
        npc_id = ctx.context.npc_id
        user_id = ctx.context.user_id
        conversation_id = ctx.context.conversation_id
        
        try:
            # Fetch relationship data
            def fetch_rels():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT entity2_type, entity2_id, link_type, link_level
                        FROM SocialLinks
                        WHERE entity1_type = 'npc'
                          AND entity1_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                        """,
                        (npc_id, user_id, conversation_id)
                    )
                    rows = cursor.fetchall()
                    npc_entity_ids = []

                    for entity_type, entity_id, _, _ in rows:
                        if entity_type == "npc":
                            npc_entity_ids.append(entity_id)

                    npc_names = {}
                    if npc_entity_ids:
                        cursor.execute(
                            """
                            SELECT npc_id, npc_name
                            FROM NPCStats
                            WHERE npc_id = ANY(%s)
                              AND user_id = %s
                              AND conversation_id = %s
                            """,
                            (npc_entity_ids, user_id, conversation_id)
                        )
                        for npc_id, npc_name in cursor.fetchall():
                            npc_names[npc_id] = npc_name

                    for entity_type, entity_id, link_type, link_level in rows:
                        entity_name = "Unknown"
                        if entity_type == "npc":
                            entity_name = npc_names.get(entity_id, f"NPC_{entity_id}")
                        elif entity_type == "player":
                            entity_name = "Player"

                        relationships[entity_type] = {
                            "entity_id": entity_id,
                            "entity_name": entity_name,
                            "link_type": link_type,
                            "link_level": link_level
                        }
                    
                    return relationships
                    
            relationships = await asyncio.to_thread(fetch_rels)
            
            # Enrich with memory context
            memory_system = await ctx.context.get_memory_system()
            
            for etype, rel_data in relationships.items():
                e_id = rel_data["entity_id"]
                e_name = rel_data["entity_name"]
                
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=npc_id,
                    query=e_name,
                    limit=3
                )
                rel_data["memory_context"] = memory_result.get("memories", [])
                
            return relationships
            
        except Exception as e:
            logger.error(f"Error fetching relationships: {e}")
            return {}

@function_tool
async def recall_memories(
    ctx: RunContextWrapper[NPCContext],
    query: MemoryQuery
) -> Dict[str, Any]:
    """
    Retrieve memories matching the query.
    
    Args:
        query: Parameters for memory retrieval
    """
    with function_span("recall_memories"):
        perf_start = time.perf_counter()
        
        try:
            # Create cache key
            cache_key = (
                f"{query.query}_{query.entity_type}_{query.entity_id}_"
                f"{query.limit}_{str(query.context)}"
            )
            
            # Check cache
            if ctx.context.is_cache_valid('memories', cache_key):
                ctx.context.perf_metrics['cache_hits'] += 1
                return ctx.context.cache['memories'][cache_key]
                
            ctx.context.perf_metrics['cache_misses'] += 1
            
            # Get memory system
            memory_system = await ctx.context.get_memory_system()
            
            # Retrieve memories
            memory_result = await memory_system.recall(
                entity_type=query.entity_type,
                entity_id=query.entity_id,
                query=query.query,
                context=query.context.model_dump() if query.context else {},
                limit=query.limit
            )
            
            # Cache result
            await ctx.context.update_cache('memories', cache_key, memory_result)
            
            # Record performance
            elapsed = time.perf_counter() - perf_start
            ctx.context.perf_metrics['memory_retrieval_time'].append(elapsed)
            
            return memory_result
            
        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Memory retrieval error after {elapsed:.2f}s: {e}")
            return {
                "memories": [],
                "error": str(e)
            }

@function_tool
async def update_emotional_state(
    ctx: RunContextWrapper[NPCContext],
    emotion: str,
    intensity: float,
    trigger: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the NPC's emotional state.
    
    Args:
        emotion: Primary emotion name
        intensity: Emotion intensity (0.0-1.0)
        trigger: Optional trigger description
    """
    with function_span("update_emotional_state"):
        try:
            memory_system = await ctx.context.get_memory_system()
            
            result = await memory_system.update_npc_emotion(
                npc_id=ctx.context.npc_id,
                emotion=emotion,
                intensity=intensity,
                trigger=trigger
            )
            
            # If strong emotion, store a memory
            if intensity > 0.7:
                memory_text = f"I felt strong {emotion}"
                if trigger:
                    memory_text += f" due to {trigger}"
                    
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["emotional_state", emotion]
                )
                
            # Invalidate cache
            await ctx.context.invalidate_cache("emotional_state")
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}")
            return {"error": str(e)}

@function_tool
async def add_memory(
    ctx: RunContextWrapper[NPCContext],
    memory_text: str,
    memory_type: str = "observation",
    significance: int = 3,
    emotional_valence: int = 0,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Add a new memory for the NPC.
    
    Args:
        memory_text: The memory text
        memory_type: Type of memory
        significance: Importance (1-10)
        emotional_valence: Emotional valence (-10 to 10)
        tags: Optional memory tags
    """
    with function_span("add_memory"):
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # Determine importance
            if significance >= 7:
                importance = "high"
            elif significance <= 2:
                importance = "low"
            else:
                importance = "medium"
                
            # Check if emotional
            is_emotional = abs(emotional_valence) > 5 or (tags and "emotional" in tags)
            
            # Create the memory
            memory_result = await memory_system.remember(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                memory_text=memory_text,
                importance=importance,
                emotional=is_emotional,
                tags=tags or []
            )
            
            memory_id = memory_result.get("memory_id")
            
            # Apply schema if memory was created
            if memory_id:
                try:
                    await memory_system.schema_manager.apply_schema_to_memory(
                        memory_id=memory_id,
                        entity_type="npc",
                        entity_id=ctx.context.npc_id,
                        auto_detect=True
                    )
                except Exception as schema_err:
                    logger.error(f"Error applying schema: {schema_err}")
            
            # Invalidate memory cache
            await ctx.context.invalidate_cache("memories")
            
            return memory_result
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {"error": str(e)}

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
# Agent definitions
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
        perceive_environment,
        get_npc_stats,
        execute_npc_action
    ]
)

memory_agent = Agent(
    name="NPC Memory Agent",
    handoff_description="Specialized agent for managing NPC memories and emotion",
    instructions="""
    You are an AI memory management system for a non-player character (NPC) in an interactive narrative.
    
    Your role is to:
    1. Store and retrieve memories based on context and relevance
    2. Track and update the NPC's emotional state
    3. Manage the NPC's social relationships
    4. Create and update beliefs based on experiences
    
    When retrieving memories, consider:
    - Relevance to the current context
    - Emotional significance of memories
    - Recency of memories
    - How memories align with the NPC's personality
    
    When managing emotions:
    - Update emotional states based on events and interactions
    - Consider emotional reactions that align with the NPC's personality
    - Track emotional intensity and how it decays over time
    
    For beliefs:
    - Create beliefs that are consistent with the NPC's experiences
    - Update confidence in beliefs based on supporting or contradicting evidence
    - Recognize and reconcile contradictory beliefs
    
    For relationship management:
    - Update relationships based on interactions
    - Consider how relationships influence memory recall and emotional responses
    """,
    tools=[
        recall_memories,
        add_memory,
        update_emotional_state
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
            tool_name_override="make_decision",
            tool_description_override="Make a decision about what action to take"
        ),
        handoff(
            agent=memory_agent,
            tool_name_override="manage_memory",
            tool_description_override="Manage memories, emotions, or beliefs"
        )
    ],
    tools=[
        get_npc_stats,
        perceive_environment,
        execute_npc_action
    ]
)

# -------------------------------------------------------
# Main NPC class utilizing Agents SDK
# -------------------------------------------------------

class NPCAgentSDK:
    """
    Core NPC agent implementation using OpenAI Agents SDK.
    Replaces the original NPCAgent class with the Agent SDK architecture.
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
                            await self._clear_nonessential_caches()
                            
                            # If still high, unload inactive agents
                            if self.memory_monitor.should_reduce_memory():
                                await self._unload_inactive_agents()
                        
                        logger.info(f"Memory reduction results: {results}")
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        asyncio.create_task(monitor_memory())

    async def _clear_nonessential_caches(self):
        """Clear caches that aren't essential right now."""
        cleared_count = 0
        
        # Clear coordinator caches
        self.coordinator.invalidate_cache("memory")
        self.coordinator.invalidate_cache("mask")
        
        # Clear agent caches
        for npc_id, agent in self.npc_agents.items():
            await agent.context.invalidate_cache()
            cleared_count += 1
        
        return cleared_count
    
    async def _unload_inactive_agents(self):
        """Unload agents that haven't been active recently."""
        # Get inactive agents (no activity in 30 min)
        inactive_threshold = datetime.now() - timedelta(minutes=30)
        inactive_agents = []
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT npc_id FROM NPCAgentState
                WHERE user_id = $1 AND conversation_id = $2
                  AND last_updated < $3
            """, self.user_id, self.conversation_id, inactive_threshold)
            
            for row in rows:
                inactive_agents.append(row["npc_id"])
        
        # Unload inactive agents
        for npc_id in inactive_agents:
            if npc_id in self.npc_agents:
                del self.npc_agents[npc_id]
        
        return len(inactive_agents)
        
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
        """
        Make a decision about what action to take.
        
        Args:
            perception_context: Optional context for perception
            
        Returns:
            The chosen action
        """
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
    
    async def process_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs.
        """
        # Acquire decision resource
        decision_resource = await self.resource_pools["decisions"].acquire()
        with trace(workflow_name=f"NPC {self.npc_id} Process Player Action"):
            perf_start = time.perf_counter()
            context_obj = context or {}
            
            try:
                # Create perception context
                perception_context = {
                    "player_action": player_action,
                    "text": player_action.get("description", ""),
                    "description": f"Player {player_action.get('description', 'did something')}"
                }
                perception_context.update(context_obj)
                
                # Perceive environment with this context
                result = await Runner.run(
                    npc_agent,
                    f"Process player action: {player_action.get('description', '')}",
                    context=self.context
                )
                npc_responses = []
                batch_size = 3
                
                # Adjust batch size based on resource availability
                if not decision_resource:
                    batch_size = 2
                    logger.warning("Decision resources constrained, processing smaller batches")
                    
                for i in range(0, len(npc_ids), batch_size):                
                # Process the response
                response = result.final_output
                if isinstance(response, dict):
                    # Return as is if already formatted correctly
                    return response
                else:
                    # Format string response into a structured result
                    return {
                        "npc_id": self.npc_id,
                        "action": {
                            "type": "response",
                            "description": response
                        },
                        "result": {
                            "outcome": response
                        }
                    }
                    
            except Exception as e:
                elapsed = time.perf_counter() - perf_start
                logger.error(f"Error processing player action for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
                
                return {"npc_responses": npc_responses}
            finally:
                # Always release the resource
                if decision_resource:
                    self.resource_pools["decisions"].release()
    
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance tasks on the NPC's memory system.
        
        Returns:
            Results of the maintenance operations
        """
        with trace(workflow_name=f"NPC {self.npc_id} Memory Maintenance"):
            try:
                # Get memory system
                memory_system = await self.context.get_memory_system()
                
                # Run maintenance operations
                maintenance_result = await memory_system.integrated.run_memory_maintenance(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    maintenance_options={
                        "core_maintenance": True,
                        "schema_maintenance": True,
                        "emotional_decay": True,
                        "memory_consolidation": True,
                        "background_reconsolidation": True,
                        "interference_processing": True,
                        "belief_consistency": True,
                        "mask_checks": True
                    }
                )
                
                # Invalidate all caches after maintenance
                await self.context.invalidate_cache()
                
                return maintenance_result
                
            except Exception as e:
                logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
                return {"error": str(e)}

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

async def create_npc_agent(npc_id: int, user_id: int, conversation_id: int) -> NPCAgentSDK:
    """
    Factory function to create an NPCAgentSDK instance.
    
    Args:
        npc_id: The ID of the NPC
        user_id: The player or user ID
        conversation_id: The conversation/scene ID
        
    Returns:
        An initialized NPCAgentSDK instance
    """
    agent = NPCAgentSDK(npc_id, user_id, conversation_id)
    
    # Initialize by fetching stats
    await get_npc_stats(RunContextWrapper(agent.context))
    
    return agent
