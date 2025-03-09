# logic/npc_agents/memory_manager.py

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

import asyncpg

# Import memory subsystem components
from memory.core import Memory, MemoryType, MemorySignificance
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)


class MemoryPerformanceTracker:
    """Tracks memory operation performance for optimization."""
    
    def __init__(self):
        self.operation_times = {
            "add_memory": [],
            "retrieve_memories": [],
            "update_emotion": [],
            "mask_operations": [],
            "belief_operations": []
        }
        self.slow_operations = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_reported = datetime.now()
    
    def record_operation(self, operation_type: str, duration: float) -> None:
        """
        Record an operation's duration and track slow operations.
        """
        if operation_type in self.operation_times:
            self.operation_times[operation_type].append(duration)
            # Keep only last 100 durations for each operation type
            if len(self.operation_times[operation_type]) > 100:
                self.operation_times[operation_type] = self.operation_times[operation_type][-100:]
            
            # Slow if > 0.5s
            if duration > 0.5:
                self.slow_operations.append({
                    "type": operation_type,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                })
                # Keep only last 50 slow operations
                if len(self.slow_operations) > 50:
                    self.slow_operations = self.slow_operations[-50:]
    
    def record_cache_hit(self) -> None:
        """Record a cache hit event."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss event."""
        self.cache_misses += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Returns an aggregated performance report containing average operation times,
        cache hit rate, slow operation count, and a timestamp.
        """
        report = {
            "averages": {},
            "cache_hit_rate": 0,
            "slow_operation_count": len(self.slow_operations),
            "timestamp": datetime.now().isoformat()
        }
        
        # Averages for each operation type
        for op_type, times in self.operation_times.items():
            if times:
                report["averages"][op_type] = sum(times) / len(times)
            else:
                report["averages"][op_type] = 0
        
        # Cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops > 0:
            report["cache_hit_rate"] = self.cache_hits / total_cache_ops
        
        return report


class EnhancedMemoryManager:
    """
    Enhanced memory manager for NPCs with improved performance, caching, and
    femdom-specific memory handling.
    """

    def __init__(
        self, 
        npc_id: int, 
        user_id: int, 
        conversation_id: int,
        db_pool: Optional[asyncpg.Pool] = None,
        npc_personality: str = "neutral",
        npc_intelligence: float = 1.0,
        use_subsystems: bool = True
    ):
        """
        Initialize the enhanced memory manager for a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            user_id: ID of the user/player
            conversation_id: ID of the current conversation
            db_pool: Asyncpg connection pool
            npc_personality: Personality type affecting memory biases
            npc_intelligence: Factor affecting memory decay rate (0.5-2.0)
            use_subsystems: Whether to use the memory subsystem managers
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db_pool = db_pool
        self.npc_personality = npc_personality
        self.npc_intelligence = npc_intelligence
        self.use_subsystems = use_subsystems
        
        # Track performance stats
        self.performance = MemoryPerformanceTracker()
        
        # Caches
        self._memory_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self._emotion_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self._mask_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self._belief_cache: Dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}
        
        # TTL for each cache type
        self._cache_ttl = {
            "memory": timedelta(minutes=5),
            "emotion": timedelta(minutes=2),
            "mask": timedelta(minutes=5),
            "belief": timedelta(minutes=10)
        }
        
        # MemorySystem reference
        self._memory_system: Optional[MemorySystem] = None
        
        # If using subsystems, initialize them asynchronously
        if use_subsystems:
            self._init_task = asyncio.create_task(self._initialize_memory_system())

    async def _initialize_memory_system(self):
        """
        Initialize the memory subsystem and ensure a mask is in place for the NPC.
        """
        try:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            await self._ensure_mask_initialized()
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")

    async def _get_memory_system(self) -> MemorySystem:
        """
        Retrieve (or lazily load) the MemorySystem instance.
        """
        if not self._memory_system:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

    async def _ensure_mask_initialized(self):
        """
        Ensure that the NPC has a mask entry in the memory system.
        """
        if not self.use_subsystems:
            return
        try:
            memory_system = await self._get_memory_system()
            await memory_system.mask_manager.initialize_npc_mask(self.npc_id)
        except Exception as e:
            logger.warning(f"Could not initialize mask for NPC {self.npc_id}: {e}")

    # --------------------------------------------------------------------------
    # Add Memory
    # --------------------------------------------------------------------------
    
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
        start_time = time.time()
        if tags is None:
            tags = []
        
        try:
            # Auto-extract content tags
            content_tags = await self.analyze_memory_content(memory_text)
            tags.extend(content_tags)
            
            # Femdom context: add relevant tags and boost significance
            if feminine_context:
                femdom_tags = self._extract_femdom_tags(memory_text)
                tags.extend(femdom_tags)
                if femdom_tags and significance < 5:
                    significance += 1
            
            # If emotional_intensity is not provided, try to derive it
            if emotional_intensity is None:
                if self.use_subsystems:
                    try:
                        memory_system = await self._get_memory_system()
                        emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(memory_text)
                        primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                        analyzed_intensity = emotion_analysis.get("intensity", 0.5)
                        emotional_intensity = int(analyzed_intensity * 100)
                    except Exception as e:
                        logger.error(f"Error analyzing emotional content: {e}")
                        # Fallback
                        emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
                else:
                    emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
            
            memory_id: Optional[int] = None
            
            # Try subsystem-based creation, else direct DB
            if self.use_subsystems:
                try:
                    memory_id = await self._add_memory_with_subsystems(
                        memory_text,
                        memory_type,
                        significance,
                        emotional_intensity,
                        tags,
                        feminine_context
                    )
                except Exception as e:
                    logger.error(f"Error adding memory with subsystems: {e}")
                    memory_id = await self._add_memory_with_db(
                        memory_text,
                        memory_type,
                        significance,
                        emotional_intensity,
                        tags,
                        status,
                        confidence
                    )
            else:
                memory_id = await self._add_memory_with_db(
                    memory_text,
                    memory_type,
                    significance,
                    emotional_intensity,
                    tags,
                    status,
                    confidence
                )
            
            # Record performance
            elapsed = time.time() - start_time
            self.performance.record_operation("add_memory", elapsed)
            
            # Invalidate memory cache
            self._invalidate_memory_cache()
            
            return memory_id
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("add_memory", elapsed)
            return None

    def _extract_femdom_tags(self, memory_text: str) -> List[str]:
        """
        Extract femdom-specific tags from memory text.
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

    async def _add_memory_with_subsystems(
        self,
        memory_text: str,
        memory_type: str,
        significance: int,
        emotional_intensity: int,
        tags: List[str],
        femdom_context: bool = False
    ) -> int:
        """
        Add memory via the memory subsystem managers, with optional femdom context enhancements.
        """
        memory_system = await self._get_memory_system()
        
        # Determine importance
        if significance >= 7:
            importance = "high"
        elif significance <= 2:
            importance = "low"
        else:
            importance = "medium"
        
        # Check if it's an emotional memory
        is_emotional = (emotional_intensity > 50) or ("emotional" in tags)
        
        # If femdom context, possibly boost importance to "high"
        if femdom_context and importance != "high":
            if importance == "low":
                importance = "medium"
            elif importance == "medium":
                importance = "high"
        
        # Create the memory in the subsystem
        memory_result = await memory_system.remember(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=is_emotional,
            tags=tags
        )
        memory_id = memory_result.get("memory_id")
        
        # Attempt schema auto-application
        try:
            await memory_system.schema_manager.apply_schema_to_memory(
                memory_id=memory_id,
                entity_type="npc",
                entity_id=self.npc_id,
                auto_detect=True
            )
        except Exception as e:
            logger.error(f"Error applying schemas to memory {memory_id}: {e}")
        
        # If high significance or femdom context, propagate
        if significance >= 4 or femdom_context:
            try:
                await self._propagate_memory_subsystems(memory_text, tags, significance, emotional_intensity)
            except Exception as e:
                logger.error(f"Error propagating memory: {e}")
        
        return memory_id

    async def _add_memory_with_db(
        self,
        memory_text: str,
        memory_type: str,
        significance: int,
        emotional_intensity: int,
        tags: List[str],
        status: str,
        confidence: float
    ) -> Optional[int]:
        """
        Add a memory directly to the database (fallback if subsystem not available or fails).
        """
        if not self.db_pool:
            logger.error("No asyncpg Pool available for direct DB insert.")
            return None
        
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO NPCMemories (
                            npc_id, memory_text, memory_type, tags,
                            emotional_intensity, significance,
                            associated_entities, is_consolidated, status, confidence
                        )
                        VALUES (
                            $1, $2, $3, $4,
                            $5, $6, $7, 
                            $8, $9, $10
                        )
                        RETURNING id
                        """,
                        self.npc_id,
                        memory_text,
                        memory_type,
                        tags,
                        emotional_intensity,
                        significance,
                        json.dumps({}),
                        False,   # is_consolidated
                        status,
                        confidence
                    )
                    memory_id = row["id"]

                    # Apply personality bias to memory confidence
                    await self.apply_npc_memory_bias(conn, memory_id)

                    # If significance is high, propagate memory to related NPCs
                    if significance >= 4:
                        await self.propagate_memory(
                            conn,
                            memory_text,
                            tags,
                            significance,
                            emotional_intensity
                        )
                    
                    return memory_id
        
        except Exception as e:
            logger.error(f"Error adding memory with DB: {e}")
            return None

    # --------------------------------------------------------------------------
    # Retrieve Memories
    # --------------------------------------------------------------------------
    
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
        """
        start_time = time.time()
        
        try:
            if context is None:
                context = {}
            cache_key = f"{query}_{limit}_{include_archived}_{femdom_focus}"
            
            # Check cache
            if cache_key in self._memory_cache:
                cache_time, cached_result = self._memory_cache[cache_key]
                if datetime.now() - cache_time < self._cache_ttl["memory"]:
                    self.performance.record_cache_hit()
                    return cached_result
            self.performance.record_cache_miss()
            
            # Default memory types if none provided
            if memory_types is None:
                memory_types = ["observation", "reflection", "semantic", "secondhand"]
            
            # Attempt subsystem retrieval; if it fails, fallback to direct DB
            if self.use_subsystems:
                try:
                    result = await self._retrieve_memories_subsystems(
                        query, context, limit, memory_types, femdom_focus
                    )
                except Exception as e:
                    logger.error(f"Error retrieving with subsystems: {e}")
                    result = await self._retrieve_memories_db(
                        query, context, limit, memory_types, include_archived, femdom_focus
                    )
            else:
                result = await self._retrieve_memories_db(
                    query, context, limit, memory_types, include_archived, femdom_focus
                )
            
            elapsed = time.time() - start_time
            self.performance.record_operation("retrieve_memories", elapsed)
            
            # Cache the result
            self._memory_cache[cache_key] = (datetime.now(), result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("retrieve_memories", elapsed)
            return {"memories": [], "error": str(e)}

    async def _retrieve_memories_subsystems(
        self,
        query: str,
        context: Dict[str, Any],
        limit: int,
        memory_types: List[str],
        femdom_focus: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve memories via memory subsystem recall, with optional femdom focus logic.
        """
        memory_system = await self._get_memory_system()
        
        if femdom_focus:
            enh_context = dict(context) if context else {}
            enh_context["priority_tags"] = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            
            enhanced_limit = min(20, limit * 2)  # get a bigger set to filter
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
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
        else:
            # Standard recall
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=query,
                context=context,
                limit=limit
            )
            return result

    async def _retrieve_memories_db(
        self,
        query: str,
        context: Dict[str, Any],
        limit: int,
        memory_types: List[str],
        include_archived: bool,
        femdom_focus: bool
    ) -> Dict[str, Any]:
        """
        Direct database retrieval of memories if subsystem approach is unavailable.
        """
        if not self.db_pool:
            logger.error("No asyncpg Pool available for direct DB fetch.")
            return {"memories": [], "error": "No DB pool"}
        
        memories = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Build status filter
                status_filter = "'active','summarized'"
                if include_archived:
                    status_filter += ",'archived'"
                
                femdom_condition = ""
                if femdom_focus:
                    # Condition for any femdom tag
                    femdom_tags = [
                        "dominance_dynamic", "power_exchange", "discipline",
                        "service", "submission", "humiliation", "ownership"
                    ]
                    femdom_condition = f"AND (tags && ARRAY{femdom_tags}::text[])"
                
                words = query.lower().split()
                
                if words:
                    conditions = []
                    params = [self.npc_id, memory_types]
                    
                    # Build search conditions for each word
                    for w in words:
                        conditions.append(f"LOWER(memory_text) LIKE ${len(params) + 1}")
                        params.append(f"%{w}%")
                    
                    condition_str = " OR ".join(conditions)
                    params.append(limit)
                    
                    q = f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id=$1
                          AND status IN ({status_filter})
                          AND memory_type = ANY($2)
                          AND ({condition_str})
                          {femdom_condition}
                        ORDER BY significance DESC, timestamp DESC
                        LIMIT ${len(params)}
                    """
                    rows = await conn.fetch(q, *params)
                else:
                    # No query words => get recent memories
                    rows = await conn.fetch(
                        f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id=$1
                          AND status IN ({status_filter})
                          AND memory_type = ANY($2)
                          {femdom_condition}
                        ORDER BY timestamp DESC
                        LIMIT $3
                        """,
                        self.npc_id, memory_types, limit
                    )
                
                for row in rows:
                    mem_dict = {
                        "id": row["id"],
                        "text": row["memory_text"],
                        "type": row["memory_type"],
                        "tags": row["tags"] or [],
                        "emotional_intensity": row["emotional_intensity"],
                        "significance": row["significance"],
                        "times_recalled": row["times_recalled"],
                        "timestamp": (
                            row["timestamp"].isoformat() 
                            if isinstance(row["timestamp"], datetime) 
                            else row["timestamp"]
                        ),
                        "status": row["status"],
                        "confidence": row["confidence"],
                        "relevance_score": 0.0
                    }
                    memories.append(mem_dict)
                
                # Update retrieval stats (times_recalled, last_recalled)
                mem_ids = [m["id"] for m in memories]
                if mem_ids:
                    await conn.execute(
                        """
                        UPDATE NPCMemories
                        SET times_recalled = times_recalled + 1,
                            last_recalled = CURRENT_TIMESTAMP
                        WHERE id = ANY($1)
                        """,
                        mem_ids
                    )
            
            # Apply biases
            memories = await self._apply_memory_biases(memories)
            return {"memories": memories, "count": len(memories)}
        
        except Exception as e:
            logger.error(f"Error in _retrieve_memories_db: {e}")
            return {"memories": [], "error": str(e)}

    async def _apply_memory_biases(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply a chain of biases to a list of memory dictionaries, then sort by relevance.
        """
        try:
            # Recency, emotional, and personality biases
            memories = await self.apply_recency_bias(memories)
            memories = await self.apply_emotional_bias(memories)
            memories = await self.apply_personality_bias(memories)
            # Highest scoring first
            memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            return memories
        except Exception as e:
            logger.error(f"Error applying memory biases: {e}")
            return memories

    # --------------------------------------------------------------------------
    # Search Memories (advanced queries)
    # --------------------------------------------------------------------------
    
    async def search_memories(
        self, 
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
        memories: List[Dict[str, Any]] = []
        try:
            memory_system = await self._get_memory_system()
            
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

    # --------------------------------------------------------------------------
    # Schemas
    # --------------------------------------------------------------------------
    
    async def generate_schemas(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """
        Group related memories into schemas automatically.
        """
        result = {
            "schemas_generated": 0,
            "memories_categorized": 0,
            "detected_patterns": []
        }
        
        try:
            memory_system = await self._get_memory_system()
            # Retrieve recent memories
            memory_results = await memory_system.recall(
                entity_type=entity_type,
                entity_id=entity_id,
                query="",
                limit=50,
                context={"max_age_days": 30}
            )
            
            memories = memory_results.get("memories", [])
            if not memories:
                return result
            
            # Group memories by patterns
            pattern_groups = {}
            
            for mem in memories:
                text = mem.get("text", "").lower()
                mem_tags = mem.get("tags", [])
                if "schema_applied" in mem_tags:
                    # Already has a schema
                    continue
                
                # Look for relationship patterns
                entity_mentions = []
                if "player" in text or "chase" in text:
                    entity_mentions.append("player")
                if "npc" in text:
                    entity_mentions.append("npc")
                
                # Common action keywords
                actions = []
                action_keywords = {
                    "conflict": ["argue", "fight", "disagree", "conflict", "dispute"],
                    "helping": ["help", "assist", "support", "aid"],
                    "teaching": ["teach", "learn", "train", "mentor"],
                    "intimacy": ["intimate", "close", "personal", "private"],
                    "submission": ["submit", "obey", "yield", "surrender"],
                    "dominance": ["dominate", "command", "control", "rule"]
                }
                
                for action_type, keywords in action_keywords.items():
                    if any(kw in text for kw in keywords):
                        actions.append(action_type)
                
                for ent in entity_mentions:
                    for act in actions:
                        pattern_key = f"{ent}_{act}"
                        if pattern_key not in pattern_groups:
                            pattern_groups[pattern_key] = []
                        pattern_groups[pattern_key].append(mem)
            
            # Create schemas for groups
            for pattern_key, group_memories in pattern_groups.items():
                if len(group_memories) < 2:
                    continue
                
                parts = pattern_key.split("_")
                if len(parts) != 2:
                    continue
                ent, act = parts
                schema_id = await memory_system.schema_manager.create_schema(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    name=f"{act.capitalize()} with {ent}",
                    description=f"Pattern of {act} interactions with {ent}"
                )
                
                mem_ids = [m.get("id") for m in group_memories]
                await memory_system.schema_manager.apply_schema_to_memories(
                    schema_id=schema_id,
                    memory_ids=mem_ids
                )
                
                result["schemas_generated"] += 1
                result["memories_categorized"] += len(mem_ids)
                result["detected_patterns"].append({
                    "pattern": pattern_key,
                    "memory_count": len(mem_ids),
                    "schema_id": schema_id
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating schemas: {e}")
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    # Emotion & Mask
    # --------------------------------------------------------------------------
    
    async def update_emotional_state(
        self,
        primary_emotion: str,
        intensity: float,
        trigger: Optional[str] = None,
        secondary_emotions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Update the NPC's emotional state with optional trigger and secondary emotions.
        """
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            
            # Construct emotional update
            current_emotion = {
                "primary": {"name": primary_emotion, "intensity": intensity},
                "secondary": {}
            }
            if secondary_emotions:
                for emo, val in secondary_emotions.items():
                    current_emotion["secondary"][emo] = val
            if trigger:
                current_emotion["trigger"] = trigger
            
            # Update in memory subsystem
            result = await memory_system.update_npc_emotion(
                npc_id=self.npc_id,
                emotion=primary_emotion,
                intensity=intensity,
                trigger=trigger
            )
            
            # If intensity is high, store a memory
            if intensity > 0.7:
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=(
                        f"I felt strong {primary_emotion}"
                        + (f" due to {trigger}" if trigger else "")
                    ),
                    importance="medium",
                    tags=["emotional_state", primary_emotion]
                )
            
            self._invalidate_emotion_cache()
            
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            return result
        
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            return {"error": str(e)}

    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get the NPC's current emotional state, using a short-lived cache.
        """
        # Check cache first
        if "current" in self._emotion_cache:
            cache_time, cached_state = self._emotion_cache["current"]
            if datetime.now() - cache_time < self._cache_ttl["emotion"]:
                self.performance.record_cache_hit()
                return cached_state
        self.performance.record_cache_miss()
        
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            state = await memory_system.get_npc_emotion(self.npc_id)
            self._emotion_cache["current"] = (datetime.now(), state)
            
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            return state
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            return {"error": str(e)}

    async def generate_mask_slippage(
        self,
        trigger: str,
        severity: Optional[int] = None,
        femdom_context: bool = False
    ) -> Dict[str, Any]:
        """
        Trigger a mask slippage event, possibly more severe in femdom contexts.
        """
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            
            # Check mask integrity
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            
            if femdom_context and severity is None:
                integrity = mask_info.get("integrity", 100)
                base_severity = max(1, min(5, int((100 - integrity) / 20)))
                severity = base_severity + 1 if random.random() < 0.7 else base_severity
            
            # Reveal trait
            slip_result = await memory_system.reveal_npc_trait(
                npc_id=self.npc_id,
                trigger=trigger,
                severity=severity
            )
            
            # Create memory
            memory_text = f"My mask slipped when {trigger}, revealing a glimpse of my true nature"
            tags = ["mask_slip", "self_awareness"]
            
            if femdom_context:
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
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance="high" if femdom_context else "medium",
                tags=tags
            )
            
            self._invalidate_mask_cache()
            
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            return slip_result
        
        except Exception as e:
            logger.error(f"Error generating mask slippage: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            return {"error": str(e)}

    async def get_npc_mask(self) -> Dict[str, Any]:
        """
        Get NPC mask info, with caching.
        """
        if "current" in self._mask_cache:
            cache_time, cached_mask = self._mask_cache["current"]
            if datetime.now() - cache_time < self._cache_ttl["mask"]:
                self.performance.record_cache_hit()
                return cached_mask
        self.performance.record_cache_miss()
        
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            self._mask_cache["current"] = (datetime.now(), mask_info)
            
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            return mask_info
        except Exception as e:
            logger.error(f"Error getting NPC mask: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    # Beliefs
    # --------------------------------------------------------------------------
    
    async def create_belief(
        self,
        belief_text: str,
        confidence: float = 0.7,
        topic: Optional[str] = None,
        femdom_context: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new belief for this NPC. If in femdom context, optionally record reflection memories.
        """
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            
            if not topic:
                # Simple guess at a topic from the first word
                words = belief_text.lower().split()
                topic = words[0] if words else "general"
                
                if femdom_context:
                    # Attempt to refine topic
                    femdom_topics = [
                        "dominance", "submission", "control", "obedience",
                        "discipline", "service", "power", "humiliation"
                    ]
                    for ft in femdom_topics:
                        if ft in belief_text.lower():
                            topic = ft
                            break
            
            result = await memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=confidence,
                topic=topic
            )
            
            # Optionally store reflection memory if femdom context
            if femdom_context and random.random() < 0.7:
                femdom_tags = ["belief", "reflection", topic]
                if topic in ["dominance", "submission", "control", "obedience"]:
                    femdom_tags.append("power_dynamic")
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"I reflected on my belief that {belief_text}",
                    importance="medium",
                    tags=femdom_tags
                )
            
            self._invalidate_belief_cache(topic)
            
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return result
        except Exception as e:
            logger.error(f"Error creating belief: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return {"error": str(e)}

    async def get_beliefs(self, topic: Optional[str] = None, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve beliefs for this NPC, optionally filtered by topic and confidence threshold.
        """
        cache_key = f"beliefs_{topic or 'all'}_{min_confidence}"
        if cache_key in self._belief_cache:
            cache_time, cached_beliefs = self._belief_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl["belief"]:
                self.performance.record_cache_hit()
                return cached_beliefs
        self.performance.record_cache_miss()
        
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            if min_confidence > 0:
                beliefs = [b for b in beliefs if b.get("confidence", 0) >= min_confidence]
            
            self._belief_cache[cache_key] = (datetime.now(), beliefs)
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return beliefs
        except Exception as e:
            logger.error(f"Error getting beliefs: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return []

    async def get_femdom_beliefs(self, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Return beliefs relevant to femdom (power dynamics, submission, etc.).
        """
        start_time = time.time()
        try:
            all_beliefs = await self.get_beliefs()
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
            
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return filtered
        except Exception as e:
            logger.error(f"Error getting femdom beliefs: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return []

    # --------------------------------------------------------------------------
    # Memory Maintenance
    # --------------------------------------------------------------------------
    
    async def run_memory_maintenance(self, include_femdom_maintenance: bool = True) -> Dict[str, Any]:
        """
        Run maintenance tasks on this NPC's memory system (consolidation, decay, etc.).
        """
        start_time = time.time()
        try:
            memory_system = await self._get_memory_system()
            results = await memory_system.maintain(
                entity_type="npc",
                entity_id=self.npc_id
            )
            
            if include_femdom_maintenance:
                try:
                    femdom_results = await self._run_femdom_maintenance()
                    results["femdom_maintenance"] = femdom_results
                except Exception as e:
                    logger.error(f"Error in femdom maintenance: {e}")
                    results["femdom_maintenance"] = {"error": str(e)}
            
            # Clear caches after maintenance
            self._invalidate_memory_cache()
            self._invalidate_emotion_cache()
            self._invalidate_mask_cache()
            self._invalidate_belief_cache()
            
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance completed in {elapsed:.2f}s")
            return results
        except Exception as e:
            logger.error(f"Error running memory maintenance: {e}")
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance failed after {elapsed:.2f}s")
            return {"error": str(e)}

    async def _run_femdom_maintenance(self) -> Dict[str, Any]:
        """
        Perform extra maintenance logic relevant to femdom (power dynamics) memories/beliefs.
        """
        results = {
            "power_dynamic_memories_processed": 0,
            "dominance_memories_consolidated": 0,
            "submission_memories_consolidated": 0,
            "power_beliefs_reinforced": 0
        }
        
        try:
            memory_system = await self._get_memory_system()
            
            # 1) Consolidate repetitive power dynamic memories
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            
            for tag in femdom_tags:
                memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    query=f"tags:{tag}",
                    limit=20
                )
                results["power_dynamic_memories_processed"] += len(memories)
                
                # Group similar memories
                memory_groups = {}
                for mem in memories:
                    text = mem.get("text", "").lower()
                    
                    key_elements: List[str] = []
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
                            entity_id=self.npc_id,
                            memory_ids=ids,
                            consolidated_text=consolidated_text,
                            tags=[tag, "consolidated", "power_dynamic"]
                        )
                        
                        if tag in ["dominance_dynamic", "control"]:
                            results["dominance_memories_consolidated"] += 1
                        elif tag in ["submission", "service"]:
                            results["submission_memories_consolidated"] += 1
            
            # 2) Reinforce power-related beliefs if enough supporting memories
            femdom_beliefs = await self.get_femdom_beliefs()
            for belief in femdom_beliefs:
                b_text = belief.get("belief", "")
                conf = belief.get("confidence", 0.5)
                
                # Gather supporting memories
                supporting_memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=self.npc_id,
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
                        entity_id=self.npc_id,
                        belief_id=belief.get("id"),
                        new_confidence=new_conf,
                        reason=f"Reinforced by {significant_support} significant memories"
                    )
                    results["power_beliefs_reinforced"] += 1
            
            return results
        except Exception as e:
            logger.error(f"Error in femdom maintenance: {e}")
            return {"error": str(e)}
    
    # --------------------------------------------------------------------------
    # Helper / Utility Methods
    # --------------------------------------------------------------------------
    
    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        """
        Basic textual analysis to assign tags (including femdom themes).
        """
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

    async def calculate_emotional_intensity(self, memory_text: str, base_valence: float) -> float:
        """
        Compute an emotional intensity from textual signals plus a base valence offset.
        """
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

    async def apply_npc_memory_bias(self, conn: asyncpg.Connection, memory_id: int):
        """
        Adjust memory confidence after insertion based on NPC personality.
        E.g. paranoid, gullible, dominant, etc.
        """
        personality_factors = {
            "gullible": 1.2,
            "skeptical": 0.8,
            "paranoid": 1.5,
            "neutral": 1.0,
            "dominant": 1.1,
            "submissive": 0.9
        }
        factor = personality_factors.get(self.npc_personality, 1.0)
        
        try:
            # Special checks for some personalities
            if self.npc_personality == "paranoid":
                row = await conn.fetchrow(
                    "SELECT memory_text FROM NPCMemories WHERE id=$1", 
                    memory_id
                )
                if row:
                    memory_text = row["memory_text"].lower()
                    # Highly suspicious of negative events
                    if any(word in memory_text for word in ["betray", "trick", "lie", "deceive", "attack"]):
                        factor = 1.5
                    else:
                        factor = 0.9
            
            elif self.npc_personality == "dominant":
                row = await conn.fetchrow(
                    "SELECT memory_text, tags FROM NPCMemories WHERE id=$1",
                    memory_id
                )
                if row:
                    tags = row["tags"] or []
                    # If the memory is about dominance
                    if any(tag in tags for tag in ["dominance_dynamic", "control", "discipline"]):
                        factor = 1.3
            
            await conn.execute(
                """
                UPDATE NPCMemories
                SET confidence = LEAST(confidence * $1, 1.0)
                WHERE id = $2
                """,
                factor,
                memory_id
            )
        
        except Exception as e:
            logger.error(f"Error applying personality bias: {e}")

    async def apply_recency_bias(self, memories: List[dict]) -> List[dict]:
        """
        Increase relevance_score for more recent memories (based on their timestamp).
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

    async def apply_emotional_bias(self, memories: List[dict]) -> List[dict]:
        """
        Increase relevance_score for higher emotional intensity and significance.
        Also boost for certain femdom tags.
        """
        for mem in memories:
            ei = mem.get("emotional_intensity", 0) / 100.0
            significance = mem.get("significance", 0) / 10.0
            base_score = ei * 3.0 + significance * 2.0
            
            mem["relevance_score"] = mem.get("relevance_score", 0) + base_score
            
            # Femdom tags
            tags = mem.get("tags", [])
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            if any(t in femdom_tags for t in tags):
                mem["relevance_score"] += 2.0
        
        return memories

    async def apply_personality_bias(self, memories: List[dict]) -> List[dict]:
        """
        Adjust final relevance scores based on the NPC's personality type.
        """
        for mem in memories:
            text = mem.get("text", "").lower()
            tags = mem.get("tags", [])
            
            if self.npc_personality == "paranoid":
                if any(k in text for k in ["threat", "danger", "betray", "attack"]):
                    mem["relevance_score"] += 3.0
                if "negative_emotion" in tags or "negative_interaction" in tags:
                    mem["relevance_score"] += 2.0
            
            elif self.npc_personality == "gullible":
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] += 2.0
            
            elif self.npc_personality == "skeptical":
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] -= 1.5
            
            elif self.npc_personality == "dominant":
                if any(t in tags for t in ["dominance_dynamic", "control", "discipline"]):
                    mem["relevance_score"] += 3.0
                if "power_exchange" in tags:
                    mem["relevance_score"] += 2.0
            
            elif self.npc_personality == "submissive":
                if any(t in tags for t in ["submission", "service", "obedience"]):
                    mem["relevance_score"] += 3.0
                if "ownership" in tags:
                    mem["relevance_score"] += 2.0
            
            # Weighted by memory confidence
            conf = mem.get("confidence", 1.0)
            mem["relevance_score"] *= conf
        
        return memories
    
    # --------------------------------------------------------------------------
    # Memory Propagation
    # --------------------------------------------------------------------------
    
    async def propagate_memory(
        self,
        conn: asyncpg.Connection,
        memory_text: str,
        tags: List[str],
        significance: int,
        emotional_intensity: float
    ):
        """
        Propagate an important memory to related NPCs as secondhand info, with distortions.
        """
        try:
            # Get related NPCs
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
                self.user_id,
                self.conversation_id,
                self.npc_id
            )
            related_npcs = [(r["entity2_id"], r["link_type"], r["link_level"]) for r in rows]
            
            # Get this NPC name
            row_npc = await conn.fetchrow(
                """
                SELECT npc_name
                FROM NPCStats
                WHERE user_id=$1
                  AND conversation_id=$2
                  AND npc_id=$3
                """,
                self.user_id,
                self.conversation_id,
                self.npc_id
            )
            npc_name = row_npc["npc_name"] if row_npc else f"NPC_{self.npc_id}"
            
            # Check if has femdom context
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline",
                "service", "submission", "humiliation", "ownership"
            ]
            has_femdom_context = any(tag in femdom_tags for tag in tags)
            
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
                
                distorted_text = self.distort_text(memory_text, severity=distortion_severity)
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
                
                # Insert secondhand memory
                await conn.execute(
                    """
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type, tags,
                        emotional_intensity, significance, status,
                        confidence, is_consolidated
                    )
                    VALUES (
                        $1, $2, 'secondhand', $3,
                        $4, $5, 'active',
                        0.7, FALSE
                    )
                    """,
                    rid,
                    secondhand_text,
                    secondhand_tags,
                    secondhand_intensity,
                    secondhand_significance
                )
            
            logger.debug(f"Propagated memory to {len(related_npcs)} related NPCs")
        
        except Exception as e:
            logger.error(f"Error propagating memory: {e}")

    async def _propagate_memory_subsystems(
        self,
        memory_text: str,
        tags: List[str],
        significance: int,
        emotional_intensity: float
    ):
        """
        Example hook if you want to propagate memory entirely through subsystem calls
        instead of direct DB calls. This method references the memory subsystem's
        built-in "propagate" or "broadcast" logic if you have one.
        (Stubbed out here; fill in as needed.)
        """
        # You could implement a specialized memory_system.propagate(...) call, or
        # you can simply replicate the logic from propagate_memory() in an async manner.
        pass

    def distort_text(self, original_text: str, severity=0.3) -> str:
        """
        Distort or partially rewrite the text at the word level, simulating rumor drift.
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

    # --------------------------------------------------------------------------
    # Cache Invalidations
    # --------------------------------------------------------------------------
    
    def _invalidate_memory_cache(self, query: Optional[str] = None) -> None:
        """
        Invalidate memory cache entries.
        If `query` is provided, remove only entries starting with that query.
        Otherwise, clear the entire memory cache.
        """
        if query:
            to_remove = []
            for key in self._memory_cache:
                if key.startswith(f"{query}_"):
                    to_remove.append(key)
            for r in to_remove:
                del self._memory_cache[r]
        else:
            self._memory_cache = {}

    def _invalidate_emotion_cache(self) -> None:
        """Clear the emotion cache."""
        self._emotion_cache = {}

    def _invalidate_mask_cache(self) -> None:
        """Clear the mask cache."""
        self._mask_cache = {}

    def _invalidate_belief_cache(self, topic: Optional[str] = None) -> None:
        """
        Invalidate belief cache entries. If topic is provided, remove only that topic's entries.
        """
        if topic:
            to_remove = []
            for key in self._belief_cache:
                if key.endswith(f"_{topic}"):
                    to_remove.append(key)
            for r in to_remove:
                del self._belief_cache[r]
        else:
            self._belief_cache = {}
