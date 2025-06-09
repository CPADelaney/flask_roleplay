# npcs/npc_memory.py

"""
Enhanced memory manager for NPCs using OpenAI Agents SDK.
Refactored from the original memory_manager.py.
"""

import logging
import json
import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, TypedDict, Set
from pydantic import BaseModel, Field

from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.tracing import custom_span, function_span
# Update the import to use the new async connection context
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Pydantic models for tool inputs/outputs
# -------------------------------------------------------

class MemoryInput(BaseModel):
    memory_text: str
    memory_type: str = "observation"
    significance: int = 3
    emotional_valence: int = 0
    emotional_intensity: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    status: str = "active"
    confidence: float = 1.0
    feminine_context: bool = False

class MemoryQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    limit: int = 5
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "semantic", "secondhand"])
    include_archived: bool = False
    femdom_focus: bool = False

class MemoryResult(BaseModel):
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None

class EmotionalStateUpdate(BaseModel):
    primary_emotion: str
    intensity: float
    trigger: Optional[str] = None
    secondary_emotions: Optional[Dict[str, float]] = None

class MaskSlippageInput(BaseModel):
    trigger: str
    severity: Optional[int] = None
    femdom_context: bool = False

class BeliefInput(BaseModel):
    belief_text: str
    confidence: float = 0.7
    topic: Optional[str] = None
    femdom_context: bool = False

class MaintenanceOptions(BaseModel):
    include_femdom_maintenance: bool = True
    consolidate_memories: bool = True
    archive_old_memories: bool = True
    update_beliefs: bool = True
    check_mask: bool = True


# -------------------------------------------------------
# Memory Manager Context
# -------------------------------------------------------

class MemoryContext:
    """Context to be passed between tools and agents in the memory manager."""
    
    def __init__(
        self, 
        npc_id: int, 
        user_id: int, 
        conversation_id: int,
        npc_personality: str = "neutral",
        npc_intelligence: float = 1.0
    ):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_personality = npc_personality
        self.npc_intelligence = npc_intelligence
        self.memory_system = None
        
        # Performance tracking
        self.performance = {
            'operation_times': {
                'add_memory': [],
                'retrieve_memories': [],
                'update_emotion': [],
                'mask_operations': [],
                'belief_operations': []
            },
            'slow_operations': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'last_reported': datetime.now()
        }
        
        # Cache management
        self.cache = {
            'memory': {},
            'emotion': {},
            'mask': {},
            'belief': {}
        }
        self.cache_timestamps = {
            'memory': {},
            'emotion': {},
            'mask': {},
            'belief': {}
        }
        self.cache_ttl = {
            'memory': timedelta(minutes=5),
            'emotion': timedelta(minutes=2),
            'mask': timedelta(minutes=5),
            'belief': timedelta(minutes=10)
        }
    
    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(
                self.user_id, 
                self.conversation_id
            )
        return self.memory_system

    async def retrieve_nyx_memories(self, query, limit=5):
        """Access Nyx's memories for NPC awareness"""
        memory_system = await self.get_memory_system()
        return await memory_system.recall(
            entity_type="nyx",
            entity_id=0,
            query=query,
            limit=limit
        )
    
    def record_operation(self, operation_type: str, duration: float) -> None:
        """Record an operation's duration and track slow operations."""
        if operation_type in self.performance['operation_times']:
            self.performance['operation_times'][operation_type].append(duration)
            # Keep only last 100 durations for each operation type
            if len(self.performance['operation_times'][operation_type]) > 100:
                self.performance['operation_times'][operation_type] = self.performance['operation_times'][operation_type][-100:]
            
            # Slow if > 0.5s
            if duration > 0.5:
                self.performance['slow_operations'].append({
                    "type": operation_type,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                })
                # Keep only last 50 slow operations
                if len(self.performance['slow_operations']) > 50:
                    self.performance['slow_operations'] = self.performance['slow_operations'][-50:]
    
    def record_cache_hit(self) -> None:
        """Record a cache hit event."""
        self.performance['cache_hits'] += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss event."""
        self.performance['cache_misses'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Returns an aggregated performance report."""
        report = {
            "averages": {},
            "cache_hit_rate": 0,
            "slow_operation_count": len(self.performance['slow_operations']),
            "timestamp": datetime.now().isoformat()
        }
        
        # Averages for each operation type
        for op_type, times in self.performance['operation_times'].items():
            if times:
                report["averages"][op_type] = sum(times) / len(times)
            else:
                report["averages"][op_type] = 0
        
        # Cache hit rate
        total_cache_ops = self.performance['cache_hits'] + self.performance['cache_misses']
        if total_cache_ops > 0:
            report["cache_hit_rate"] = self.performance['cache_hits'] / total_cache_ops
        
        return report
    
    def is_cache_valid(self, cache_type: str, key: str) -> bool:
        """Check if a cache entry is valid."""
        if cache_type not in self.cache or key not in self.cache[cache_type]:
            return False
        
        timestamp = self.cache_timestamps[cache_type].get(key)
        if timestamp is None:
            return False
        
        ttl = self.cache_ttl.get(cache_type)
        if ttl is None:
            return False
        
        return datetime.now() - timestamp < ttl
    
    def update_cache(self, cache_type: str, key: str, value: Any) -> None:
        """Update a cache entry."""
        if cache_type not in self.cache:
            return
        
        self.cache[cache_type][key] = value
        self.cache_timestamps[cache_type][key] = datetime.now()
    
    def invalidate_cache(self, cache_type: Optional[str] = None, key: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if cache_type is None:
            # Invalidate all caches
            for c_type in self.cache:
                self.cache[c_type] = {}
                self.cache_timestamps[c_type] = {}
        elif cache_type in self.cache:
            if key is None:
                # Invalidate all entries for this cache type
                self.cache[cache_type] = {}
                self.cache_timestamps[cache_type] = {}
            elif key in self.cache[cache_type]:
                # Invalidate specific entry
                del self.cache[cache_type][key]
                if key in self.cache_timestamps[cache_type]:
                    del self.cache_timestamps[cache_type][key]

# -------------------------------------------------------
# Tool Functions
# -------------------------------------------------------

@function_tool(strict_mode=False)
async def add_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_input: MemoryInput
) -> Dict[str, Any]:
    """
    Add a new memory for the NPC with enhanced femdom (feminine dominance) context handling.
    
    Args:
        memory_input: The memory data to add
    """
    with function_span("add_memory"):
        start_time = time.time()
        
        try:
            # Auto-extract content tags
            content_tags = await analyze_memory_content(ctx, memory_input.memory_text)
            updated_tags = list(memory_input.tags)
            updated_tags.extend(content_tags)
            
            # Femdom context: add relevant tags and boost significance
            if memory_input.feminine_context:
                femdom_tags = extract_femdom_tags(memory_input.memory_text)
                updated_tags.extend(femdom_tags)
                if femdom_tags and memory_input.significance < 5:
                    memory_input.significance += 1
            
            # If emotional_intensity is not provided, try to derive it
            emotional_intensity = memory_input.emotional_intensity
            if emotional_intensity is None:
                try:
                    memory_system = await ctx.context.get_memory_system()
                    emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                        memory_input.memory_text
                    )
                    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                    analyzed_intensity = emotion_analysis.get("intensity", 0.5)
                    emotional_intensity = int(analyzed_intensity * 100)
                except Exception as e:
                    logger.error(f"Error analyzing emotional content: {e}")
                    # Fallback
                    emotional_intensity = await calculate_emotional_intensity(
                        ctx, 
                        memory_input.memory_text, 
                        memory_input.emotional_valence
                    )
            
            # Create memory with memory system
            memory_system = await ctx.context.get_memory_system()
            
            # Determine importance
            if memory_input.significance >= 7:
                importance = "high"
            elif memory_input.significance <= 2:
                importance = "low"
            else:
                importance = "medium"
            
            # Check if it's an emotional memory
            is_emotional = (emotional_intensity > 50) or ("emotional" in updated_tags)
            
            # If femdom context, possibly boost importance
            if memory_input.feminine_context and importance != "high":
                if importance == "low":
                    importance = "medium"
                elif importance == "medium":
                    importance = "high"
            
            # Create the memory
            memory_result = await memory_system.remember(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                memory_text=memory_input.memory_text,
                importance=importance,
                emotional=is_emotional,
                tags=updated_tags
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
                except Exception as e:
                    logger.error(f"Error applying schemas to memory {memory_id}: {e}")
            
            # If high significance or femdom context, propagate
            if memory_input.significance >= 4 or memory_input.feminine_context:
                try:
                    await propagate_memory(
                        ctx,
                        memory_input.memory_text,
                        updated_tags,
                        memory_input.significance,
                        emotional_intensity
                    )
                except Exception as e:
                    logger.error(f"Error propagating memory: {e}")
            
            # Record performance
            elapsed = time.time() - start_time
            ctx.context.record_operation("add_memory", elapsed)
            
            # Invalidate memory cache
            ctx.context.invalidate_cache("memory")
            
            return memory_result
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("add_memory", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def analyze_memory_content(
    ctx: RunContextWrapper[MemoryContext],
    memory_text: str
) -> List[str]:
    """
    Basic textual analysis to assign tags (including femdom themes).
    
    Args:
        memory_text: The memory text to analyze
    """
    with function_span("analyze_memory_content"):
        tags = []
        lower_text = memory_text.lower()
        
        # Emotional content
        if any(w in lower_text for w in ["angry", "upset", "mad", "furious", "betrayed"]):
            tags.append("negative_emotion")
        if any(w in lower_text for w in ["happy", "pleased", "joy", "delighted", "thrilled"]):
            tags.append("positive_emotion")
        if any(w in lower_text for w in ["afraid", "scared", "fearful", "terrified"]):
            tags.append("fear")
        if any(w in lower_text for w in ["aroused", "excited", "turned on", "desire"]):
            tags.append("arousal")
        
        # Player or rumor indicators
        if "player" in lower_text or "user" in lower_text or "chase" in lower_text:
            tags.append("player_related")
        if any(w in lower_text for w in ["heard", "told me", "said that"]):
            tags.append("rumor")
        
        # Social interactions
        if any(w in lower_text for w in ["helped", "assisted", "supported", "saved"]):
            tags.append("positive_interaction")
        if any(w in lower_text for w in ["betrayed", "attacked", "deceived", "tricked"]):
            tags.append("negative_interaction")
        
        # Femdom-specific keywords
        if any(word in lower_text for word in ["command", "ordered", "instructed", "demanded"]):
            tags.append("dominance_dynamic")
        if any(word in lower_text for word in ["obey", "submit", "comply", "kneel", "bow"]):
            tags.append("submission")
        if any(word in lower_text for word in ["punish", "discipline", "correct", "consequences"]):
            tags.append("discipline")
        if any(word in lower_text for word in ["spank", "whip", "paddle", "impact"]):
            tags.append("physical_discipline")
        if any(word in lower_text for word in ["humiliate", "embarrass", "shame", "mock"]):
            tags.append("humiliation")
        if any(word in lower_text for word in ["train", "condition", "learn", "lesson", "teach"]):
            tags.append("training")
        if any(word in lower_text for word in ["control", "restrict", "limit", "permission"]):
            tags.append("control")
        if any(word in lower_text for word in ["worship", "serve", "service", "please", "satisfy"]):
            tags.append("service")
        if any(word in lower_text for word in ["devoted", "loyal", "faithful", "belonging"]):
            tags.append("devotion")
        if any(word in lower_text for word in ["resist", "disobey", "refuse", "defy"]):
            tags.append("resistance")
        if any(word in lower_text for word in ["power", "exchange", "protocol", "dynamic", "role"]):
            tags.append("power_exchange")
        if any(word in lower_text for word in ["praise", "good", "well done", "proud"]):
            tags.append("praise")
        if any(word in lower_text for word in ["ritual", "ceremony", "protocol", "procedure"]):
            tags.append("ritual")
        if any(word in lower_text for word in ["collar", "own", "belong", "possess", "property"]):
            tags.append("ownership")
        if any(word in lower_text for word in ["mind", "psyche", "thoughts", "mental", "mindset"]):
            tags.append("psychological")
        
        return tags

def extract_femdom_tags(memory_text: str) -> List[str]:
    """
    Extract femdom-specific tags from memory text.
    
    Args:
        memory_text: The memory text to analyze
    """
    femdom_tags = []
    lower_text = memory_text.lower()
    
    # Typical femdom-themed keywords
    if any(word in lower_text for word in ["command", "control", "obey", "order", "dominated"]):
        femdom_tags.append("dominance_dynamic")
    if any(word in lower_text for word in ["power", "exchange", "protocol", "dynamic", "role"]):
        femdom_tags.append("power_exchange")
    if any(word in lower_text for word in ["punish", "discipline", "correct", "consequence"]):
        femdom_tags.append("discipline")
    if any(word in lower_text for word in ["serve", "service", "please", "worship"]):
        femdom_tags.append("service")
    if any(word in lower_text for word in ["submit", "obey", "comply", "kneel", "bow"]):
        femdom_tags.append("submission")
    if any(word in lower_text for word in ["bind", "restrain", "restrict", "tied"]):
        femdom_tags.append("bondage")
    if any(word in lower_text for word in ["humiliate", "embarrass", "shame", "mock"]):
        femdom_tags.append("humiliation")
    if any(word in lower_text for word in ["own", "belong", "property", "possession"]):
        femdom_tags.append("ownership")
    
    return femdom_tags

@function_tool(strict_mode=False)
async def calculate_emotional_intensity(
    ctx: RunContextWrapper[MemoryContext],
    memory_text: str,
    base_valence: float
) -> float:
    """
    Compute an emotional intensity from textual signals plus a base valence offset.
    
    Args:
        memory_text: The memory text
        base_valence: Base emotional valence (-10 to 10)
    """
    with function_span("calculate_emotional_intensity"):
        # Convert base valence [-10..10] to [0..100]
        intensity = (base_valence + 10) * 5
        
        # Standard emotional words
        emotion_words = {
            "furious": 20, "ecstatic": 20, "devastated": 20, "thrilled": 20,
            "angry": 15, "delighted": 15, "sad": 15, "happy": 15,
            "annoyed": 10, "pleased": 10, "upset": 10, "glad": 10,
            "concerned": 5, "fine": 5, "worried": 5, "okay": 5
        }
        
        # Femdom-specific intensifiers
        femdom_emotion_words = {
            "humiliated": 25, "dominated": 22, "controlled": 20, 
            "obedient": 18, "submissive": 18, "powerful": 20, 
            "superior": 15, "worshipped": 22
        }
        
        lower_text = memory_text.lower()
        
        # Check standard words
        for w, boost in emotion_words.items():
            if w in lower_text:
                intensity += boost
                break
        
        # Check femdom words
        for w, boost in femdom_emotion_words.items():
            if w in lower_text:
                intensity += boost
                break
        
        # Clamp [0..100]
        return float(min(100, max(0, intensity)))

@function_tool(strict_mode=False)
async def retrieve_memories(
    ctx: RunContextWrapper[MemoryContext],
    query: MemoryQuery
) -> MemoryResult:
    """
    Retrieve memories matching certain criteria, with optional femdom focus.
    
    Args:
        query: Memory retrieval query
    """
    with function_span("retrieve_memories"):
        start_time = time.time()
        
        try:
            # Create cache key
            cache_key = (
                f"{query.query}_{query.limit}_{query.include_archived}_"
                f"{query.femdom_focus}_{str(hash(str(query.context)))}"
            )
            
            # Check cache
            if ctx.context.is_cache_valid("memory", cache_key):
                ctx.context.record_cache_hit()
                cached_result = ctx.context.cache["memory"][cache_key]
                return MemoryResult(**cached_result)
            
            ctx.context.record_cache_miss()
            
            # Get memory system
            memory_system = await ctx.context.get_memory_system()
            
            # Attempt retrieval with subsystems
            if query.femdom_focus:
                result = await retrieve_memories_with_femdom_focus(
                    ctx, 
                    query.query, 
                    query.context or {}, 
                    query.limit, 
                    query.memory_types
                )
            else:
                # Standard recall
                recall_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    query=query.query,
                    context=query.context,
                    limit=query.limit
                )
                result = {
                    "memories": recall_result.get("memories", []),
                    "count": len(recall_result.get("memories", []))
                }
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("retrieve_memories", elapsed)
            
            # Cache the result
            ctx.context.update_cache("memory", cache_key, result)
            
            return MemoryResult(**result)
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("retrieve_memories", elapsed)
            return MemoryResult(memories=[], error=str(e))

async def retrieve_memories_with_femdom_focus(
    ctx: RunContextWrapper[MemoryContext],
    query: str,
    context: Dict[str, Any],
    limit: int,
    memory_types: List[str]
) -> Dict[str, Any]:
    """
    Retrieve memories with a focus on femdom content.
    
    Args:
        query: The query string
        context: Additional context
        limit: Maximum number of memories to retrieve
        memory_types: Types of memories to retrieve
    """
    memory_system = await ctx.context.get_memory_system()
    
    enh_context = dict(context)
    enh_context["priority_tags"] = [
        "dominance_dynamic", "power_exchange", "discipline",
        "service", "submission", "humiliation", "ownership"
    ]
    
    enhanced_limit = min(20, limit * 2)  # get a bigger set to filter
    result = await memory_system.recall(
        entity_type="npc",
        entity_id=ctx.context.npc_id,
        query=query,
        context=enh_context,
        limit=enhanced_limit
    )
    
    # Separate femdom vs non-femdom
    memories = result.get("memories", [])
    femdom_memories = []
    other_memories = []
    
    for m in memories:
        t = m.get("tags", [])
        if any(tag in enh_context["priority_tags"] for tag in t):
            femdom_memories.append(m)
        else:
            other_memories.append(m)
    
    final_memories = femdom_memories + other_memories
    final_memories = final_memories[:limit]
    return {"memories": final_memories, "count": len(final_memories)}

@function_tool(strict_mode=False)
async def search_memories(
    ctx: RunContextWrapper[MemoryContext],
    entity_type: str,
    entity_id: int,
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search memories with specific criteria beyond standard recall.
    
    Args:
        entity_type: 'npc' or 'player'
        entity_id: ID of the entity
        query: text search or tag-based search
        limit: max results
    """
    with function_span("search_memories"):
        memories = []
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # Split query parts to handle tags
            search_parts = query.split()
            tag_filters = []
            standard_terms = []
            
            for part in search_parts:
                if ":" in part:
                    t, val = part.split(":", 1)
                    tag_filters.append((t, val))
                else:
                    standard_terms.append(part)
            
            standard_query = " ".join(standard_terms)
            
            # First recall from memory system
            result = await memory_system.recall(
                entity_type=entity_type,
                entity_id=entity_id,
                query=standard_query,
                limit=limit * 2
            )
            
            memories = result.get("memories", [])
            
            # If tag filters exist, do a second pass
            if tag_filters:
                filtered = []
                for mem in memories:
                    mem_tags = mem.get("tags", [])
                    # Must satisfy all tag filters
                    pass_filters = True
                    for (tag, val) in tag_filters:
                        if tag == "text":
                            if val.lower() not in mem.get("text", "").lower():
                                pass_filters = False
                                break
                        else:
                            if tag not in mem_tags and val not in mem_tags:
                                pass_filters = False
                                break
                    if pass_filters:
                        filtered.append(mem)
                memories = filtered
            
            # Limit final
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

@function_tool(strict_mode=False)
async def update_emotional_state(
    ctx: RunContextWrapper[MemoryContext],
    update: EmotionalStateUpdate
) -> Dict[str, Any]:
    """
    Update the NPC's emotional state with optional trigger and secondary emotions.
    
    Args:
        update: The emotional state update data
    """
    with function_span("update_emotional_state"):
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # Construct emotional update
            current_emotion = {
                "primary": {"name": update.primary_emotion, "intensity": update.intensity},
                "secondary": {}
            }
            if update.secondary_emotions:
                for emo, val in update.secondary_emotions.items():
                    current_emotion["secondary"][emo] = val
            if update.trigger:
                current_emotion["trigger"] = update.trigger
            
            # Update in memory subsystem
            result = await memory_system.update_npc_emotion(
                npc_id=ctx.context.npc_id,
                emotion=update.primary_emotion,
                intensity=update.intensity,
                trigger=update.trigger
            )
            
            # If intensity is high, store a memory
            if update.intensity > 0.7:
                memory_text = f"I felt strong {update.primary_emotion}"
                if update.trigger:
                    memory_text += f" due to {update.trigger}"
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["emotional_state", update.primary_emotion]
                )
            
            ctx.context.invalidate_cache("emotion")
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("update_emotion", elapsed)
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("update_emotion", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def get_emotional_state(
    ctx: RunContextWrapper[MemoryContext]
) -> Dict[str, Any]:
    """
    Get the NPC's current emotional state, using a short-lived cache.
    """
    with function_span("get_emotional_state"):
        # Check cache first
        if ctx.context.is_cache_valid("emotion", "current"):
            ctx.context.record_cache_hit()
            return ctx.context.cache["emotion"]["current"]
        
        ctx.context.record_cache_miss()
        
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            state = await memory_system.get_npc_emotion(ctx.context.npc_id)
            
            ctx.context.update_cache("emotion", "current", state)
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("update_emotion", elapsed)
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("update_emotion", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def generate_mask_slippage(
    ctx: RunContextWrapper[MemoryContext],
    input_data: MaskSlippageInput
) -> Dict[str, Any]:
    """
    Trigger a mask slippage event, possibly more severe in femdom contexts.
    
    Args:
        input_data: Data for generating mask slippage
    """
    with function_span("generate_mask_slippage"):
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # Check mask integrity
            mask_info = await memory_system.get_npc_mask(ctx.context.npc_id)
            
            severity = input_data.severity
            if input_data.femdom_context and severity is None:
                integrity = mask_info.get("integrity", 100)
                base_severity = max(1, min(5, int((100 - integrity) / 20)))
                severity = base_severity + 1 if random.random() < 0.7 else base_severity
            
            # Reveal trait
            slip_result = await memory_system.reveal_npc_trait(
                npc_id=ctx.context.npc_id,
                trigger=input_data.trigger,
                severity=severity
            )
            
            # Create memory
            memory_text = f"My mask slipped when {input_data.trigger}, revealing a glimpse of my true nature"
            tags = ["mask_slip", "self_awareness"]
            
            if input_data.femdom_context:
                hidden_traits = mask_info.get("hidden_traits", {})
                if "dominant" in hidden_traits:
                    memory_text += ", showing my underlying dominance"
                elif "submissive" in hidden_traits:
                    memory_text += ", exposing my natural submission"
                elif "sadistic" in hidden_traits:
                    memory_text += ", revealing my cruel tendencies"
                
                tags.append("power_dynamic")
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                memory_text=memory_text,
                importance="high" if input_data.femdom_context else "medium",
                tags=tags
            )
            
            ctx.context.invalidate_cache("mask")
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("mask_operations", elapsed)
            
            return slip_result
            
        except Exception as e:
            logger.error(f"Error generating mask slippage: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("mask_operations", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def get_npc_mask(
    ctx: RunContextWrapper[MemoryContext]
) -> Dict[str, Any]:
    """
    Get NPC mask info, with caching.
    """
    with function_span("get_npc_mask"):
        if ctx.context.is_cache_valid("mask", "current"):
            ctx.context.record_cache_hit()
            return ctx.context.cache["mask"]["current"]
        
        ctx.context.record_cache_miss()
        
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            mask_info = await memory_system.get_npc_mask(ctx.context.npc_id)
            
            ctx.context.update_cache("mask", "current", mask_info)
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("mask_operations", elapsed)
            
            return mask_info
            
        except Exception as e:
            logger.error(f"Error getting NPC mask: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("mask_operations", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def create_belief(
    ctx: RunContextWrapper[MemoryContext],
    input_data: BeliefInput
) -> Dict[str, Any]:
    """
    Create a new belief for this NPC. If in femdom context, optionally record reflection memories.
    
    Args:
        input_data: The belief data to create
    """
    with function_span("create_belief"):
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            topic = input_data.topic
            if not topic:
                # Simple guess at a topic from the first word
                words = input_data.belief_text.lower().split()
                topic = words[0] if words else "general"
                
                if input_data.femdom_context:
                    # Attempt to refine topic
                    femdom_topics = [
                        "dominance", "submission", "control", "obedience",
                        "discipline", "service", "power", "humiliation"
                    ]
                    for ft in femdom_topics:
                        if ft in input_data.belief_text.lower():
                            topic = ft
                            break
            
            result = await memory_system.create_belief(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                belief_text=input_data.belief_text,
                confidence=input_data.confidence,
                topic=topic
            )
            
            # Optionally store reflection memory if femdom context
            if input_data.femdom_context and random.random() < 0.7:
                femdom_tags = ["belief", "reflection", topic]
                if topic in ["dominance", "submission", "control", "obedience"]:
                    femdom_tags.append("power_dynamic")
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    memory_text=f"I reflected on my belief that {input_data.belief_text}",
                    importance="medium",
                    tags=femdom_tags
                )
            
            ctx.context.invalidate_cache("belief", topic)
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating belief: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def get_beliefs(
    ctx: RunContextWrapper[MemoryContext],
    topic: Optional[str] = None,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Retrieve beliefs for this NPC, optionally filtered by topic and confidence threshold.
    
    Args:
        topic: Optional topic filter
        min_confidence: Minimum confidence threshold
    """
    with function_span("get_beliefs"):
        cache_key = f"beliefs_{topic or 'all'}_{min_confidence}"
        
        if ctx.context.is_cache_valid("belief", cache_key):
            ctx.context.record_cache_hit()
            return ctx.context.cache["belief"][cache_key]
        
        ctx.context.record_cache_miss()
        
        start_time = time.time()
        
        try:
            memory_system = await ctx.context.get_memory_system()
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                topic=topic
            )
            
            # Filter by confidence
            if min_confidence > 0:
                beliefs = [b for b in beliefs if b.get("confidence", 0) >= min_confidence]
            
            ctx.context.update_cache("belief", cache_key, beliefs)
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            
            return beliefs
            
        except Exception as e:
            logger.error(f"Error getting beliefs: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            return []

@function_tool(strict_mode=False)
async def get_femdom_beliefs(
    ctx: RunContextWrapper[MemoryContext],
    min_confidence: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Return beliefs relevant to femdom (power dynamics, submission, etc.).
    
    Args:
        min_confidence: Minimum confidence threshold
    """
    with function_span("get_femdom_beliefs"):
        cache_key = f"femdom_beliefs_{min_confidence}"
        
        if ctx.context.is_cache_valid("belief", cache_key):
            ctx.context.record_cache_hit()
            return ctx.context.cache["belief"][cache_key]
        
        ctx.context.record_cache_miss()
        
        start_time = time.time()
        
        try:
            all_beliefs = await get_beliefs(ctx, None, 0.0)
            femdom_keywords = [
                "dominance", "submission", "control", "obedience",
                "discipline", "service", "power", "humiliation",
                "command", "order", "punishment", "reward", "train",
                "serve", "worship", "respect", "protocol", "rule"
            ]
            
            filtered = []
            for b in all_beliefs:
                belief_txt = b.get("belief", "").lower()
                conf = b.get("confidence", 0)
                if any(k in belief_txt for k in femdom_keywords) and conf >= min_confidence:
                    filtered.append(b)
            
            filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            ctx.context.update_cache("belief", cache_key, filtered)
            
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error getting femdom beliefs: {e}")
            elapsed = time.time() - start_time
            ctx.context.record_operation("belief_operations", elapsed)
            return []

@function_tool(strict_mode=False)
async def run_femdom_maintenance(
    ctx: RunContextWrapper[MemoryContext]
) -> Dict[str, Any]:
    """
    Perform extra maintenance logic relevant to femdom (power dynamics) memories/beliefs.
    """
    with function_span("run_femdom_maintenance"):
        results = {
            "power_dynamic_memories_processed": 0,
            "dominance_memories_consolidated": 0,
            "submission_memories_consolidated": 0,
            "power_beliefs_reinforced": 0
        }
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # 1) Consolidate repetitive power dynamic memories
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            
            for tag in femdom_tags:
                memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    query=f"tags:{tag}",
                    limit=20
                )
                results["power_dynamic_memories_processed"] += len(memories)
                
                # Group similar memories
                memory_groups = {}
                for mem in memories:
                    text = mem.get("text", "").lower()
                    
                    key_elements = []
                    if tag == "dominance_dynamic":
                        for w in ["command", "control", "dominate", "authority"]:
                            if w in text:
                                key_elements.append(w)
                    elif tag == "submission":
                        for w in ["obey", "submit", "follow", "comply"]:
                            if w in text:
                                key_elements.append(w)
                    
                    if key_elements:
                        group_key = " ".join(sorted(key_elements))
                    else:
                        group_key = text[:20]  # fallback grouping
                    
                    if group_key not in memory_groups:
                        memory_groups[group_key] = []
                    memory_groups[group_key].append(mem)
                
                # Consolidate groups with >=3 memories
                for group_key, group_memories in memory_groups.items():
                    if len(group_memories) >= 3:
                        mem_texts = [m.get("text", "") for m in group_memories]
                        ids = [m.get("id") for m in group_memories]
                        consolidated_text = (
                            f"I have {len(mem_texts)} similar experiences involving {tag}: "
                            f"'{mem_texts[0]}' and similar events."
                        )
                        
                        await memory_system.consolidate_specific_memories(
                            entity_type="npc",
                            entity_id=ctx.context.npc_id,
                            memory_ids=ids,
                            consolidated_text=consolidated_text,
                            tags=[tag, "consolidated", "power_dynamic"]
                        )
                        
                        if tag in ["dominance_dynamic", "control"]:
                            results["dominance_memories_consolidated"] += 1
                        elif tag in ["submission", "service"]:
                            results["submission_memories_consolidated"] += 1
            
            # 2) Reinforce power-related beliefs if enough supporting memories
            femdom_beliefs = await get_femdom_beliefs(ctx)
            for belief in femdom_beliefs:
                b_text = belief.get("belief", "")
                conf = belief.get("confidence", 0.5)
                
                # Gather supporting memories
                supporting_memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    query=b_text,
                    limit=5
                )
                significant_support = sum(
                    1 for m in supporting_memories if m.get("significance", 0) >= 4
                )
                
                if significant_support >= 3:
                    new_conf = min(0.95, conf + 0.1)
                    await memory_system.update_belief_confidence(
                        entity_type="npc",
                        entity_id=ctx.context.npc_id,
                        belief_id=belief.get("id"),
                        new_confidence=new_conf,
                        reason=f"Reinforced by {significant_support} significant memories"
                    )
                    results["power_beliefs_reinforced"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in femdom maintenance: {e}")
            return {"error": str(e)}

@function_tool(strict_mode=False)
async def propagate_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_text: str,
    tags: List[str],
    significance: int,
    emotional_intensity: float
) -> Dict[str, Any]:
    """
    Propagate an important memory to related NPCs as secondhand info, with distortions.
    
    Args:
        memory_text: The memory text
        tags: Memory tags
        significance: Memory significance (1-10)
        emotional_intensity: Emotional intensity (0-100)
    """
    with function_span("propagate_memory"):
        result = {"propagated_to": 0}
        
        try:
            # Get related NPCs - rewritten to use async connection
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE user_id=$1
                      AND conversation_id=$2
                      AND entity1_type='npc'
                      AND entity1_id=$3
                      AND entity2_type='npc'
                    """,
                    ctx.context.user_id, ctx.context.conversation_id, ctx.context.npc_id
                )
                related_npcs = [(row["entity2_id"], row["link_type"], row["link_level"]) for row in rows]
            
            # Get this NPC name - rewritten to use async connection
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT npc_name
                    FROM NPCStats
                    WHERE user_id=$1
                      AND conversation_id=$2
                      AND npc_id=$3
                    """,
                    ctx.context.user_id, ctx.context.conversation_id, ctx.context.npc_id
                )
                npc_name = row["npc_name"] if row else f"NPC_{ctx.context.npc_id}"
            
            # Check if has femdom context
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            has_femdom_context = any(tag in femdom_tags for tag in tags)
            
            memory_system = await ctx.context.get_memory_system()
            
            # For each related NPC, create a secondhand memory
            for rid, link_type, link_level in related_npcs:
                # Distort the memory text according to link_level
                distortion_severity = 0.3
                if link_level > 75:
                    distortion_severity = 0.1
                elif link_level > 50:
                    distortion_severity = 0.2
                elif link_level < 25:
                    distortion_severity = 0.5
                
                distorted_text = distort_text(memory_text, severity=distortion_severity)
                secondhand_text = f"I heard that {npc_name} {distorted_text}"
                
                secondhand_significance = max(1, significance - 2)
                secondhand_intensity = max(0, emotional_intensity - 20)
                
                secondhand_tags = tags + ["secondhand", "rumor"]
                
                if has_femdom_context:
                    # Relationship-based nuance
                    if link_type == "submissive":
                        if any(t in tags for t in ["dominance_dynamic", "control"]):
                            secondhand_text = f"I heard that {npc_name} was extremely dominant when {distorted_text}"
                            secondhand_tags.append("exaggerated")
                    elif link_type == "dominant":
                        if any(t in tags for t in ["dominance_dynamic", "control"]):
                            secondhand_text = f"I heard that {npc_name} tried to act dominant by {distorted_text}"
                            secondhand_tags.append("diminished")
                
                # Convert significance (1-10) to importance level
                if secondhand_significance >= 7:
                    importance = "high"
                elif secondhand_significance <= 2:
                    importance = "low"
                else:
                    importance = "medium"
                
                # Create secondhand memory using memory system
                mem_result = await memory_system.remember(
                    entity_type="npc",
                    entity_id=rid,
                    memory_text=secondhand_text,
                    importance=importance,
                    emotional=secondhand_intensity > 50,
                    tags=secondhand_tags
                )
                
                if "memory_id" in mem_result:
                    result["propagated_to"] += 1
            
            logger.debug(f"Propagated memory to {result['propagated_to']} related NPCs")
            
            return result
            
        except Exception as e:
            logger.error(f"Error propagating memory: {e}")
            return {"error": str(e)}

def distort_text(original_text: str, severity=0.3) -> str:
    """
    Distort or partially rewrite the text at the word level, simulating rumor drift.
    
    Args:
        original_text: Original text to distort
        severity: Distortion severity (0.0-1.0)
    """
    synonyms_map = {
        "attacked": ["assaulted", "ambushed", "jumped"],
        "betrayed": ["backstabbed", "double-crossed", "deceived"],
        "stole": ["looted", "swiped", "snatched", "took"],
        "helped": ["assisted", "saved", "aided", "supported"],
        "rescued": ["freed", "saved", "liberated", "pulled out"],
        "said": ["mentioned", "claimed", "stated", "told me"],
        "saw": ["noticed", "spotted", "observed", "glimpsed"],
        "went": ["traveled", "journeyed", "ventured", "headed"],
        "found": ["discovered", "located", "uncovered", "came across"]
    }
    
    # Femdom synonyms
    femdom_synonyms = {
        "dominated": ["controlled completely", "took full control of", "overpowered"],
        "commanded": ["ordered", "instructed strictly", "demanded"],
        "punished": ["disciplined", "corrected", "taught a lesson to"],
        "submitted": ["obeyed", "yielded", "surrendered"],
        "praised": ["rewarded", "showed approval to", "acknowledged"],
        "humiliated": ["embarrassed", "shamed", "put in their place"]
    }
    
    # Combine dictionaries
    all_synonyms = {**synonyms_map, **femdom_synonyms}
    
    words = original_text.split()
    for i in range(len(words)):
        if random.random() < severity:
            w_lower = words[i].lower()
            if w_lower in all_synonyms:
                words[i] = random.choice(all_synonyms[w_lower])
            elif random.random() < 0.2:
                # Chance to remove word entirely
                words[i] = ""
    
    # Re-join, removing empties
    return " ".join([w for w in words if w])

@function_tool(strict_mode=False)
async def run_memory_maintenance(
    ctx: RunContextWrapper[MemoryContext],
    options: MaintenanceOptions
) -> Dict[str, Any]:
    """
    Run maintenance tasks on this NPC's memory system (consolidation, decay, etc.).
    Optimized with batched DB operations.
    
    Args:
        options: Options for what maintenance to perform
    """
    with function_span("run_memory_maintenance"):
        start_time = time.time()
        
        results = {
            "memories_processed": 0,
            "memories_archived": 0,
            "memories_updated": 0,
            "batch_operations": 0
        }
        
        try:
            memory_system = await ctx.context.get_memory_system()
            
            # Collect IDs for memory operations
            memory_ids_to_archive = []
            memory_ids_to_consolidate = []
            memory_ids_to_decay = []
            
            # Get candidate memories
            old_memories = await memory_system.search_memories(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                query="age_days:>90",  # Memories older than 90 days
                limit=100
            )
            
            low_importance_memories = await memory_system.search_memories(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                query="importance:low age_days:>30",  # Low importance memories older than 30 days
                limit=100
            )
            
            # Process memory sets
            for memory in old_memories:
                results["memories_processed"] += 1
                significance = memory.get("significance", 0)
                if significance < 4:  # Low significance threshold
                    memory_ids_to_archive.append(memory.get("id"))
            
            for memory in low_importance_memories:
                if memory.get("id") not in memory_ids_to_archive:  # Avoid duplicates
                    memory_ids_to_archive.append(memory.get("id"))
                    results["memories_processed"] += 1
            
            # Find candidates for consolidation (similar memories)
            duplicate_candidates = await memory_system.search_memories(
                entity_type="npc",
                entity_id=ctx.context.npc_id,
                query="duplicate_score:>0.7",  # Memories with high similarity
                limit=50
            )
            
            # Group by similarity for consolidation
            similarity_groups = {}
            for memory in duplicate_candidates:
                topic = memory.get("topic", "general")
                if topic not in similarity_groups:
                    similarity_groups[topic] = []
                similarity_groups[topic].append(memory)
            
            # Collect IDs for each group with more than 2 similar memories
            for topic, memories in similarity_groups.items():
                if len(memories) >= 3:  # Need at least 3 to consolidate
                    memory_ids = [m.get("id") for m in memories]
                    memory_ids_to_consolidate.extend(memory_ids)
            
            # Execute batch operations
            if memory_ids_to_archive and options.archive_old_memories:
                await memory_system.update_memory_status_batch(
                    entity_type="npc",
                    entity_id=ctx.context.npc_id,
                    memory_ids=memory_ids_to_archive,
                    new_status="archived"
                )
                results["memories_archived"] = len(memory_ids_to_archive)
                results["batch_operations"] += 1
            
            if memory_ids_to_consolidate and options.consolidate_memories:
                # Group by topic for meaningful consolidation
                for topic, mems in similarity_groups.items():
                    if len(mems) >= 3:
                        ids = [m.get("id") for m in mems]
                        await memory_system.consolidate_memories_batch(
                            entity_type="npc",
                            entity_id=ctx.context.npc_id,
                            memory_ids=ids,
                            consolidated_text=f"I have several similar memories about {topic}"
                        )
                        results["batch_operations"] += 1
            
            if options.include_femdom_maintenance:
                femdom_results = await run_femdom_maintenance(ctx)
                results["femdom_maintenance"] = femdom_results
            
            # Clear caches after maintenance
            ctx.context.invalidate_cache()
            
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance completed in {elapsed:.2f}s")
            results["execution_time"] = elapsed
            
            return results
            
        except Exception as e:
            logger.error(f"Error running memory maintenance: {e}")
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance failed after {elapsed:.2f}s")
            return {"error": str(e)}

# -------------------------------------------------------
# Memory Manager Agent
# -------------------------------------------------------

memory_agent = Agent(
    name="NPC Memory Manager",
    instructions="""
    You are an AI memory management system for non-player characters (NPCs) in an interactive narrative simulation.
    
    Your responsibilities include:
    1. Storing and retrieving memories based on context and relevance
    2. Managing the NPC's emotional state
    3. Creating and updating beliefs based on experiences
    4. Maintaining the NPC's "mask" system (presented vs. true personality)
    5. Propagating important memories to other NPCs
    6. Running regular maintenance on the memory system
    
    Focus on realistic memory retrieval with appropriate biases:
    - More recent memories are generally more accessible
    - Emotionally significant memories are more prominent
    - Memories aligned with the NPC's personality are emphasized
    - Femdom-related memories (power dynamics, dominance, submission) may have special significance
    
    When managing emotions, consider:
    - Current emotional state influences memory recall
    - Emotional reactions should be consistent with personality and past experiences
    - Strong emotions should be stored as memories themselves
    
    For beliefs:
    - Create beliefs that reflect the NPC's experiences and personality
    - Update confidence levels based on supporting evidence
    - Pay special attention to beliefs related to power dynamics and relationships
    
    Maintain psychological realism:
    - Consider how memories decay over time
    - Model how true personality can "leak" through a presented mask
    - Provide contextually appropriate emotional responses
    
    Use the available tools to perform memory operations efficiently.
    """,
    tools=[
        add_memory,
        analyze_memory_content,
        retrieve_memories,
        search_memories,
        update_emotional_state,
        get_emotional_state,
        generate_mask_slippage,
        get_npc_mask,
        create_belief,
        get_beliefs,
        get_femdom_beliefs,
        run_memory_maintenance,
        propagate_memory
    ]
)

# -------------------------------------------------------
# Main Memory Manager class
# -------------------------------------------------------

class NPCMemoryManager:
    """
    Enhanced memory manager for NPCs using OpenAI Agents SDK.
    Refactored from original EnhancedMemoryManager class.
    """
    
    def __init__(
        self, 
        npc_id: int, 
        user_id: int, 
        conversation_id: int,
        npc_personality: str = "neutral",
        npc_intelligence: float = 1.0
    ):
        """
        Initialize the memory manager for a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            user_id: ID of the user/player
            conversation_id: ID of the current conversation
            npc_personality: Personality type affecting memory biases
            npc_intelligence: Factor affecting memory decay rate (0.5-2.0)
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Create context
        self.context = MemoryContext(
            npc_id, 
            user_id, 
            conversation_id,
            npc_personality,
            npc_intelligence
        )
        
        # Initialize performance reporting
        self._setup_performance_reporting()
    
    def _setup_performance_reporting(self) -> None:
        """
        Set up periodic performance reporting in the background.
        """
        async def report_metrics():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                
                report = self.context.get_performance_report()
                
                logger.info(f"NPC {self.npc_id} memory performance: {report}")
                
                # Reset slow operations after reporting
                self.context.performance['slow_operations'] = []
        
        # Start the background reporting task
        asyncio.create_task(report_metrics())
    
    async def add_memory(
        self,
        memory_text: str,
        memory_type: str = "observation",
        significance: int = 3,
        emotional_valence: int = 0,
        emotional_intensity: Optional[int] = None,
        tags: Optional[List[str]] = None,
        status: str = "active",
        confidence: float = 1.0,
        feminine_context: bool = False
    ) -> Optional[int]:
        """
        Add a new memory for the NPC with enhanced femdom (feminine dominance) context handling.
        
        Args:
            memory_text: The memory text
            memory_type: Type of memory (e.g., 'observation', 'reflection', etc.)
            significance: Numeric indicator of importance (1-10)
            emotional_valence: Base emotional valence (-10 to 10)
            emotional_intensity: Direct override for emotional intensity (0-100)
            tags: Additional tags to store
            status: 'active', 'summarized', or 'archived'
            confidence: Confidence in this memory (0.0-1.0)
            feminine_context: Whether this memory has femdom context
            
        Returns:
            The created memory's ID, or None on error.
        """
        memory_input = MemoryInput(
            memory_text=memory_text,
            memory_type=memory_type,
            significance=significance,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            tags=tags or [],
            status=status,
            confidence=confidence,
            feminine_context=feminine_context
        )
        
        result = await add_memory(
            RunContextWrapper(self.context),
            memory_input
        )
        
        return result.get("memory_id")
    
    async def retrieve_memories(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        include_archived: bool = False,
        femdom_focus: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve memories matching certain criteria, with optional femdom focus.
        
        Args:
            query: Query string
            context: Additional context
            limit: Maximum number of memories to retrieve
            memory_types: Types of memories to retrieve
            include_archived: Whether to include archived memories
            femdom_focus: Whether to focus on femdom-related memories
        """
        memory_query = MemoryQuery(
            query=query,
            context=context,
            limit=limit,
            memory_types=memory_types or ["observation", "reflection", "semantic", "secondhand"],
            include_archived=include_archived,
            femdom_focus=femdom_focus
        )
        
        result = await retrieve_memories(
            RunContextWrapper(self.context),
            memory_query
        )
        
        return result.model_dump()
    
    async def update_emotional_state(
        self,
        primary_emotion: str,
        intensity: float,
        trigger: Optional[str] = None,
        secondary_emotions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Update the NPC's emotional state.
        
        Args:
            primary_emotion: Primary emotion name
            intensity: Intensity (0.0-1.0)
            trigger: Optional trigger description
            secondary_emotions: Optional secondary emotions
        """
        update = EmotionalStateUpdate(
            primary_emotion=primary_emotion,
            intensity=intensity,
            trigger=trigger,
            secondary_emotions=secondary_emotions
        )
        
        result = await update_emotional_state(
            RunContextWrapper(self.context),
            update
        )
        
        return result
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get the NPC's current emotional state.
        """
        return await get_emotional_state(RunContextWrapper(self.context))
    
    async def generate_mask_slippage(
        self,
        trigger: str,
        severity: Optional[int] = None,
        femdom_context: bool = False
    ) -> Dict[str, Any]:
        """
        Trigger a mask slippage event.
        
        Args:
            trigger: What triggered the slippage
            severity: Optional severity override
            femdom_context: Whether this is in a femdom context
        """
        input_data = MaskSlippageInput(
            trigger=trigger,
            severity=severity,
            femdom_context=femdom_context
        )
        
        result = await generate_mask_slippage(
            RunContextWrapper(self.context),
            input_data
        )
        
        return result
    
    async def get_npc_mask(self) -> Dict[str, Any]:
        """
        Get NPC mask info.
        """
        return await get_npc_mask(RunContextWrapper(self.context))
    
    async def create_belief(
        self,
        belief_text: str,
        confidence: float = 0.7,
        topic: Optional[str] = None,
        femdom_context: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new belief for this NPC.
        
        Args:
            belief_text: The belief text
            confidence: Confidence level (0.0-1.0)
            topic: Optional topic
            femdom_context: Whether this is in a femdom context
        """
        input_data = BeliefInput(
            belief_text=belief_text,
            confidence=confidence,
            topic=topic,
            femdom_context=femdom_context
        )
        
        result = await create_belief(
            RunContextWrapper(self.context),
            input_data
        )
        
        return result
    
    async def get_beliefs(
        self,
        topic: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get beliefs for this NPC.
        
        Args:
            topic: Optional topic filter
            min_confidence: Minimum confidence threshold
        """
        return await get_beliefs(
            RunContextWrapper(self.context),
            topic,
            min_confidence
        )
    
    async def get_femdom_beliefs(
        self,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get femdom-related beliefs.
        
        Args:
            min_confidence: Minimum confidence threshold
        """
        return await get_femdom_beliefs(
            RunContextWrapper(self.context),
            min_confidence
        )
    
    async def run_memory_maintenance(
        self,
        include_femdom_maintenance: bool = True
    ) -> Dict[str, Any]:
        """
        Run maintenance tasks on this NPC's memory system.
        
        Args:
            include_femdom_maintenance: Whether to include femdom-specific maintenance
        """
        options = MaintenanceOptions(
            include_femdom_maintenance=include_femdom_maintenance
        )
        
        result = await run_memory_maintenance(
            RunContextWrapper(self.context),
            options
        )
        
        return result
    
    async def search_memories(
        self,
        entity_type: str,
        entity_id: int,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories with specific criteria.
        
        Args:
            entity_type: 'npc' or 'player'
            entity_id: ID of the entity
            query: Search query
            limit: Maximum number of results
        """
        return await search_memories(
            RunContextWrapper(self.context),
            entity_type,
            entity_id,
            query,
            limit
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance metrics for the memory manager.
        """
        return self.context.get_performance_report()
