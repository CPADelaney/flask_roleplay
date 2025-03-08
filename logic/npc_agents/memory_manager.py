# logic/npc_agents/revised_memory_manager.py

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Set

# Import necessary dependencies
try:
    import asyncpg
except ImportError:
    logging.warning("asyncpg may not be installed.")

# Import memory subsystem components
from memory.core import Memory, MemoryType, MemorySignificance
from memory.wrapper import MemorySystem
from db.connection import get_db_connection

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
        """Record an operation's duration."""
        if operation_type in self.operation_times:
            self.operation_times[operation_type].append(duration)
            # Keep only the last 100 operations
            if len(self.operation_times[operation_type]) > 100:
                self.operation_times[operation_type] = self.operation_times[operation_type][-100:]
            
            # Record slow operations
            if duration > 0.5:  # Over 500ms is slow
                self.slow_operations.append({
                    "type": operation_type,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                })
                # Keep only the last 50 slow operations
                if len(self.slow_operations) > 50:
                    self.slow_operations = self.slow_operations[-50:]
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a performance report."""
        report = {
            "averages": {},
            "cache_hit_rate": 0,
            "slow_operation_count": len(self.slow_operations),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate averages
        for op_type, times in self.operation_times.items():
            if times:
                report["averages"][op_type] = sum(times) / len(times)
            else:
                report["averages"][op_type] = 0
        
        # Calculate cache hit rate
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
        db_pool=None,
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
            db_pool: Optional asyncpg connection pool (will create connections if None)
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
        
        # Performance tracking
        self.performance = MemoryPerformanceTracker()
        
        # Improved caching system
        self._memory_cache = {}
        self._emotion_cache = {}
        self._mask_cache = {}
        self._belief_cache = {}
        self._cache_ttl = {
            "memory": timedelta(minutes=5),
            "emotion": timedelta(minutes=2),
            "mask": timedelta(minutes=5),
            "belief": timedelta(minutes=10)
        }
        
        # Initialize subsystem managers if enabled
        if use_subsystems:
            self._memory_system = None
            # Create task for lazy initialization
            self._init_task = asyncio.create_task(self._initialize_memory_system())
    
    async def _initialize_memory_system(self):
        """Initialize the memory system."""
        try:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            
            # Initialize mask if not already done
            await self._ensure_mask_initialized()
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
    
    async def _get_memory_system(self):
        """Get the memory system, ensuring it's initialized."""
        if not hasattr(self, '_memory_system') or self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system
    
    async def _ensure_mask_initialized(self):
        """Ensure the NPC has a mask initialized."""
        if not self.use_subsystems:
            return
            
        try:
            memory_system = await self._get_memory_system()
            await memory_system.mask_manager.initialize_npc_mask(self.npc_id)
        except Exception as e:
            logger.warning(f"Could not initialize mask for NPC {self.npc_id}: {e}")
    
    async def add_memory(
        self, 
        memory_text: str, 
        memory_type: str = "observation", 
        significance: int = 3,
        emotional_valence: int = 0,
        emotional_intensity: Optional[int] = None,
        tags: List[str] = None,
        status: str = "active",
        confidence: float = 1.0,
        feminine_context: bool = False  # New parameter for femdom context
    ) -> Optional[int]:
        """
        Add a new memory for the NPC with enhanced femdom context handling.
        
        Args:
            memory_text: The memory text to store
            memory_type: Type of memory (observation, reflection, etc.)
            significance: Importance of the memory (1-10)
            emotional_valence: Emotion direction (-10 to 10)
            emotional_intensity: Optional direct intensity value (0-100)
            tags: Optional tags for the memory
            status: Memory status ('active', 'summarized', 'archived')
            confidence: Confidence in this memory (0.0-1.0)
            feminine_context: Whether this memory has femdom context
            
        Returns:
            ID of the created memory or None if failed
        """
        start_time = time.time()
        
        try:
            tags = tags or []
            
            # Auto-detect content tags
            content_tags = await self.analyze_memory_content(memory_text)
            tags.extend(content_tags)
            
            # Add femdom-specific tags if relevant context
            if feminine_context:
                femdom_tags = self._extract_femdom_tags(memory_text)
                tags.extend(femdom_tags)
                # Increase significance for femdom-related memories
                if femdom_tags and significance < 5:
                    significance += 1
            
            # If emotional_intensity not provided, calculate from valence or analyze text
            if emotional_intensity is None:
                if self.use_subsystems:
                    try:
                        memory_system = await self._get_memory_system()
                        emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(memory_text)
                        primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                        emotional_intensity = int(emotion_analysis.get("intensity", 0.5) * 100)
                    except Exception as e:
                        logger.error(f"Error analyzing emotional content: {e}")
                        emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
                else:
                    emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
            
            memory_id = None
            
            if self.use_subsystems:
                try:
                    memory_id = await self._add_memory_with_subsystems(
                        memory_text, memory_type, significance, 
                        emotional_intensity, tags, feminine_context
                    )
                except Exception as e:
                    logger.error(f"Error adding memory with subsystems: {e}")
                    # Fall back to direct DB access
                    memory_id = await self._add_memory_with_db(
                        memory_text, memory_type, significance,
                        emotional_intensity, tags, status, confidence
                    )
            else:
                memory_id = await self._add_memory_with_db(
                    memory_text, memory_type, significance,
                    emotional_intensity, tags, status, confidence
                )
            
            # Record performance data
            elapsed = time.time() - start_time
            self.performance.record_operation("add_memory", elapsed)
            
            # Invalidate relevant cache entries
            self._invalidate_memory_cache()
            
            return memory_id
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            # Record failed operation
            elapsed = time.time() - start_time
            self.performance.record_operation("add_memory", elapsed)
            return None
    
    def _extract_femdom_tags(self, memory_text: str) -> List[str]:
        """
        Extract femdom-specific tags from memory text.
        
        Args:
            memory_text: The memory text to analyze
            
        Returns:
            List of femdom-related tags
        """
        femdom_tags = []
        lower_text = memory_text.lower()
        
        # Dominance dynamics
        if any(word in lower_text for word in ["command", "control", "obey", "order", "dominated"]):
            femdom_tags.append("dominance_dynamic")
        
        # Power exchange
        if any(word in lower_text for word in ["power", "exchange", "protocol", "dynamic", "role"]):
            femdom_tags.append("power_exchange")
        
        # Discipline
        if any(word in lower_text for word in ["punish", "discipline", "correct", "consequence"]):
            femdom_tags.append("discipline")
        
        # Service
        if any(word in lower_text for word in ["serve", "service", "please", "worship"]):
            femdom_tags.append("service")
        
        # Submission
        if any(word in lower_text for word in ["submit", "obey", "comply", "kneel", "bow"]):
            femdom_tags.append("submission")
        
        # Bondage
        if any(word in lower_text for word in ["bind", "restrain", "restrict", "tied"]):
            femdom_tags.append("bondage")
        
        # Humiliation
        if any(word in lower_text for word in ["humiliate", "embarrass", "shame", "mock"]):
            femdom_tags.append("humiliation")
        
        # Ownership
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
        """Add memory using memory subsystem managers with femdom enhancements."""
        try:
            memory_system = await self._get_memory_system()
            
            # Try to add with emotional context
            importance = "medium"
            if significance >= 7:
                importance = "high"
            elif significance <= 2:
                importance = "low"
            
            # Determine if this is an emotional memory
            is_emotional = emotional_intensity > 50 or "emotional" in tags
            
            # Add femdom importance boost
            if femdom_context and importance != "high":
                # Promote importance level for femdom memories
                if importance == "low":
                    importance = "medium"
                elif importance == "medium":
                    importance = "high"
            
            # Create the memory
            memory_result = await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance=importance,
                emotional=is_emotional,
                tags=tags
            )
            
            memory_id = memory_result.get("memory_id")
            
            # Auto-detect and apply schemas
            try:
                await memory_system.schema_manager.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type="npc",
                    entity_id=self.npc_id,
                    auto_detect=True
                )
            except Exception as e:
                logger.error(f"Error applying schemas to memory {memory_id}: {e}")
            
            # If significance is high or femdom context, propagate to connected NPCs
            if significance >= 4 or femdom_context:
                try:
                    await self._propagate_memory_subsystems(memory_text, tags, significance, emotional_intensity)
                except Exception as e:
                    logger.error(f"Error propagating memory: {e}")
            
            return memory_id
        except Exception as e:
            logger.error(f"Error in _add_memory_with_subsystems: {e}")
            raise
    
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
        """Add memory using direct database access."""
        conn = None
        try:
            # Get a connection
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert row with the memory data
                cursor.execute("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type, tags,
                        emotional_intensity, significance, 
                        associated_entities, is_consolidated, status, confidence
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    RETURNING id
                """, (
                    self.npc_id,
                    memory_text,
                    memory_type,
                    tags,
                    emotional_intensity,
                    significance,
                    json.dumps({}),
                    False,
                    status,
                    confidence
                ))
                
                memory_id = cursor.fetchone()[0]
                
                # Apply personality-based bias to memory confidence
                await self.apply_npc_memory_bias(conn, memory_id)
                
                # If significance is high, propagate the memory to connected NPCs
                if significance >= 4:
                    await self.propagate_memory(conn, memory_text, tags, significance, emotional_intensity)
                
                return memory_id
                
        except Exception as e:
            logger.error(f"Error adding memory with DB: {e}")
            return None
    
    async def retrieve_memories(
        self, 
        query: str,
        context: Dict[str, Any] = None,
        limit: int = 5,
        memory_types: List[str] = None,
        include_archived: bool = False,
        femdom_focus: bool = False  # New parameter for femdom focus
    ) -> Dict[str, Any]:
        """
        Retrieve memories relevant to a query with enhanced femdom focus option.
        
        Args:
            query: Search query string
            context: Additional context dictionary
            limit: Maximum memories to return
            memory_types: Types of memories to include
            include_archived: Whether to include archived memories
            femdom_focus: Whether to prioritize femdom-related memories
            
        Returns:
            Dictionary with memories and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{limit}_{include_archived}_{femdom_focus}"
            if cache_key in self._memory_cache:
                cache_time, cached_result = self._memory_cache[cache_key]
                # Only use cache if it's recent
                if datetime.now() - cache_time < self._cache_ttl["memory"]:
                    self.performance.record_cache_hit()
                    return cached_result
            
            self.performance.record_cache_miss()
            
            # Set default memory types if not provided
            memory_types = memory_types or ["observation", "reflection", "semantic", "secondhand"]
            
            # Process the query
            if self.use_subsystems:
                try:
                    result = await self._retrieve_memories_subsystems(query, context, limit, memory_types, femdom_focus)
                except Exception as e:
                    logger.error(f"Error retrieving memories with subsystems: {e}")
                    # Fall back to direct DB access
                    result = await self._retrieve_memories_db(query, context, limit, memory_types, include_archived, femdom_focus)
            else:
                result = await self._retrieve_memories_db(query, context, limit, memory_types, include_archived, femdom_focus)
            
            # Record performance data
            elapsed = time.time() - start_time
            self.performance.record_operation("retrieve_memories", elapsed)
            
            # Cache the result
            self._memory_cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            # Return empty result
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
        """Retrieve memories using memory subsystem managers."""
        memory_system = await self._get_memory_system()
        
        # Enhance query for femdom focus if requested
        if femdom_focus:
            enhaned_context = context.copy() if context else {}
            # Prioritize power dynamic memories
            enhaned_context["priority_tags"] = [
                "dominance_dynamic", "power_exchange", "discipline", 
                "service", "submission", "humiliation", "ownership"
            ]
            # Increase limit for better selection
            enhanced_limit = min(20, limit * 2)
            
            # Get a larger set of memories
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=query,
                context=enhaned_context,
                limit=enhanced_limit
            )
            
            # Filter and prioritize femdom-related memories
            memories = result.get("memories", [])
            
            # Separate femdom from non-femdom memories
            femdom_memories = []
            other_memories = []
            
            for memory in memories:
                tags = memory.get("tags", [])
                if any(tag in enhaned_context["priority_tags"] for tag in tags):
                    femdom_memories.append(memory)
                else:
                    other_memories.append(memory)
            
            # Take prioritized memories up to limit
            prioritized_memories = femdom_memories + other_memories
            final_memories = prioritized_memories[:limit]
            
            return {"memories": final_memories, "count": len(final_memories)}
        else:
            # Standard memory retrieval
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
        femdom_focus: bool = False
    ) -> Dict[str, Any]:
        """Retrieve memories using direct database access."""
        try:
            conn = None
            memories = []
            
            # Get connection
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Determine status filter
                status_filter = "'active','summarized'"
                if include_archived:
                    status_filter += ",'archived'"
                
                # Add femdom focus if requested
                femdom_condition = ""
                if femdom_focus:
                    femdom_tags = [
                        "dominance_dynamic", "power_exchange", "discipline", 
                        "service", "submission", "humiliation", "ownership"
                    ]
                    # Create condition for ANY of these tags
                    femdom_condition = f"AND (tags && ARRAY{femdom_tags}::text[])"
                
                # Perform keyword-based search
                words = query.lower().split()
                if words:
                    conditions = []
                    params = [self.npc_id, memory_types]
                    
                    # Add word conditions
                    for i, word in enumerate(words):
                        conditions.append(f"LOWER(memory_text) LIKE %s")
                        params.append(f"%{word}%")
                    
                    condition_str = " OR ".join(conditions)
                    
                    # Add limit to params
                    params.append(limit)
                    
                    # Build the query
                    q = f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id = %s
                          AND status IN ({status_filter})
                          AND memory_type = ANY(%s)
                          AND ({condition_str})
                          {femdom_condition}
                        ORDER BY 
                          {f"CASE WHEN tags && ARRAY{femdom_tags}::text[] THEN 0 ELSE 1 END," if femdom_focus else ""}
                          significance DESC, 
                          timestamp DESC
                        LIMIT %s
                    """
                    
                    cursor.execute(q, params)
                    
                else:
                    # No query words, just get recent memories
                    cursor.execute(f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id = %s
                          AND status IN ({status_filter})
                          AND memory_type = ANY(%s)
                          {femdom_condition}
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (self.npc_id, memory_types, limit))
                
                # Process the results
                rows = cursor.fetchall()
                for row in rows:
                    memory_id, text, mem_type, tags, intensity, significance, times_recalled, timestamp, status, confidence = row
                    
                    memories.append({
                        "id": memory_id,
                        "text": text,
                        "type": mem_type,
                        "tags": tags or [],
                        "emotional_intensity": intensity,
                        "significance": significance,
                        "times_recalled": times_recalled,
                        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                        "status": status,
                        "confidence": confidence,
                        "relevance_score": 0.0
                    })
                
                # Update retrieval stats
                self._update_memory_retrieval_stats(conn, [m["id"] for m in memories])
                
            # Apply biases to the memories
            memories = await self._apply_memory_biases(memories)
            
            return {"memories": memories, "count": len(memories)}
            
        except Exception as e:
            logger.error(f"Error in _retrieve_memories_db: {e}")
            return {"memories": [], "error": str(e)}
    
    async def _apply_memory_biases(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply various biases to retrieved memories."""
        try:
            # Apply recency bias
            memories = await self.apply_recency_bias(memories)
            
            # Apply emotional bias
            memories = await self.apply_emotional_bias(memories)
            
            # Apply personality bias
            memories = await self.apply_personality_bias(memories)
            
            # Sort by relevance score
            memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return memories
        except Exception as e:
            logger.error(f"Error applying memory biases: {e}")
            return memories
    
    async def _update_memory_retrieval_stats(self, conn, memory_ids: List[int]):
        """Update statistics for retrieved memories."""
        if not memory_ids:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = CURRENT_TIMESTAMP
                WHERE id = ANY(%s)
            """, (memory_ids,))
        except Exception as e:
            logger.error(f"Error updating memory retrieval stats: {e}")
    
    def _invalidate_memory_cache(self, query: str = None):
        """Invalidate memory cache entries."""
        if query:
            # Invalidate specific query
            keys_to_remove = []
            for key in self._memory_cache:
                if key.startswith(f"{query}_"):
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self._memory_cache[key]
        else:
            # Invalidate all memory cache
            self._memory_cache = {}
    
    def _invalidate_emotion_cache(self):
        """Invalidate emotion cache."""
        self._emotion_cache = {}
    
    def _invalidate_mask_cache(self):
        """Invalidate mask cache."""
        self._mask_cache = {}
    
    def _invalidate_belief_cache(self, topic: str = None):
        """Invalidate belief cache entries."""
        if topic:
            # Invalidate specific topic
            keys_to_remove = []
            for key in self._belief_cache:
                if key.endswith(f"_{topic}"):
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self._belief_cache[key]
        else:
            # Invalidate all belief cache
            self._belief_cache = {}
    
    async def update_emotional_state(self, 
                                   primary_emotion: str, 
                                   intensity: float, 
                                   trigger: str = None,
                                   secondary_emotions: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Update the NPC's emotional state with enhanced monitoring.
        
        Args:
            primary_emotion: The primary emotion
            intensity: Intensity of the emotion (0.0-1.0)
            trigger: What triggered this emotion
            secondary_emotions: Optional additional emotions
            
        Returns:
            Updated emotional state
        """
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # Create emotional state update
            current_emotion = {
                "primary": {
                    "name": primary_emotion,
                    "intensity": intensity
                },
                "secondary": {}
            }
            
            # Add secondary emotions if provided
            if secondary_emotions:
                for emotion, value in secondary_emotions.items():
                    current_emotion["secondary"][emotion] = value
            
            # Add trigger if provided
            if trigger:
                current_emotion["trigger"] = trigger
            
            # Update emotional state
            result = await memory_system.update_npc_emotion(
                npc_id=self.npc_id,
                emotion=primary_emotion,
                intensity=intensity,
                trigger=trigger
            )
            
            # Create memory of significant emotional changes
            if intensity > 0.7:
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"I felt strong {primary_emotion}" + (f" due to {trigger}" if trigger else ""),
                    importance="medium",
                    tags=["emotional_state", primary_emotion]
                )
            
            # Invalidate emotion cache
            self._invalidate_emotion_cache()
            
            # Record performance
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
        Get the NPC's current emotional state with caching.
        
        Returns:
            Current emotional state
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
            
            # Get emotional state
            state = await memory_system.get_npc_emotion(self.npc_id)
            
            # Cache the result
            self._emotion_cache["current"] = (datetime.now(), state)
            
            # Record performance
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("update_emotion", elapsed)
            return {"error": str(e)}
    
    async def generate_mask_slippage(self, 
                                   trigger: str, 
                                   severity: int = None,
                                   femdom_context: bool = False) -> Dict[str, Any]:
        """
        Generate a mask slippage event where the NPC's true nature shows through.
        Enhanced with femdom context awareness.
        
        Args:
            trigger: What triggered the mask slippage
            severity: How severe the slippage is (1-5)
            femdom_context: Whether the slippage occurs in a femdom context
            
        Returns:
            Mask slippage details
        """
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # Get mask info for integrity checks
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            
            # In femdom context, slippages can be more severe
            if femdom_context and severity is None:
                integrity = mask_info.get("integrity", 100)
                base_severity = max(1, min(5, int((100 - integrity) / 20)))
                severity = base_severity + 1 if random.random() < 0.7 else base_severity
            
            # Generate slippage
            slip_result = await memory_system.reveal_npc_trait(
                npc_id=self.npc_id,
                trigger=trigger,
                severity=severity
            )
            
            # Create memory of the slippage with femdom context if applicable
            memory_text = f"My mask slipped when {trigger}, revealing a glimpse of my true nature"
            
            if femdom_context:
                hidden_traits = mask_info.get("hidden_traits", {})
                
                # Add femdom-specific context to memory
                if "dominant" in hidden_traits:
                    memory_text += ", showing my underlying dominance"
                elif "submissive" in hidden_traits:
                    memory_text += ", exposing my natural submission"
                elif "sadistic" in hidden_traits:
                    memory_text += ", revealing my cruel tendencies"
                
                # Add femdom tags
                tags = ["mask_slip", "self_awareness", "power_dynamic"]
                
                if "dominant" in hidden_traits:
                    tags.append("dominance_dynamic")
                elif "submissive" in hidden_traits:
                    tags.append("submission")
                elif "sadistic" in hidden_traits:
                    tags.append("discipline")
            else:
                tags = ["mask_slip", "self_awareness"]
            
            # Create memory
            await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance="high" if femdom_context else "medium",
                tags=tags
            )
            
            # Invalidate mask cache
            self._invalidate_mask_cache()
            
            # Record performance
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
        Get information about the NPC's mask with caching.
        
        Returns:
            Mask information (integrity, presented traits, etc.)
        """
        # Check cache first
        if "current" in self._mask_cache:
            cache_time, cached_mask = self._mask_cache["current"]
            if datetime.now() - cache_time < self._cache_ttl["mask"]:
                self.performance.record_cache_hit()
                return cached_mask
        
        self.performance.record_cache_miss()
        
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # Get mask info
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            
            # Cache the result
            self._mask_cache["current"] = (datetime.now(), mask_info)
            
            # Record performance
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            
            return mask_info
            
        except Exception as e:
            logger.error(f"Error getting NPC mask: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("mask_operations", elapsed)
            return {"error": str(e)}
    
    async def create_belief(self, 
                          belief_text: str, 
                          confidence: float = 0.7,
                          topic: str = None,
                          femdom_context: bool = False) -> Dict[str, Any]:
        """
        Create a belief for the NPC with femdom context awareness.
        
        Args:
            belief_text: The belief statement
            confidence: Confidence level (0.0-1.0)
            topic: Optional topic categorization
            femdom_context: Whether this belief has femdom context
            
        Returns:
            Created belief information
        """
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # If no topic provided, try to extract one
            if not topic:
                # Default topic extraction
                words = belief_text.lower().split()
                topic = words[0] if words else "general"
                
                # Check for femdom-specific topics
                if femdom_context:
                    # Extract femdom topics
                    femdom_topics = [
                        "dominance", "submission", "control", "obedience",
                        "discipline", "service", "power", "humiliation"
                    ]
                    
                    for ft in femdom_topics:
                        if ft in belief_text.lower():
                            topic = ft
                            break
            
            # Create the belief
            result = await memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=confidence,
                topic=topic
            )
            
            # In femdom context, create a supporting memory
            if femdom_context and random.random() < 0.7:  # 70% chance
                # Create a memory supporting this belief
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"I reflected on my belief that {belief_text}",
                    importance="medium",
                    tags=["belief", "reflection", topic] + 
                         (["power_dynamic"] if topic in ["dominance", "submission", "control", "obedience"] else [])
                )
            
            # Invalidate belief cache
            self._invalidate_belief_cache(topic)
            
            # Record performance
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating belief: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return {"error": str(e)}
    
    async def get_beliefs(self, topic: str = None, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get the NPC's beliefs with caching and confidence filtering.
        
        Args:
            topic: Optional topic filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of beliefs
        """
        # Create cache key
        cache_key = f"beliefs_{topic or 'all'}_{min_confidence}"
        
        # Check cache first
        if cache_key in self._belief_cache:
            cache_time, cached_beliefs = self._belief_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl["belief"]:
                self.performance.record_cache_hit()
                return cached_beliefs
        
        self.performance.record_cache_miss()
        
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # Get beliefs
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            
            # Filter by confidence if needed
            if min_confidence > 0:
                beliefs = [b for b in beliefs if b.get("confidence", 0) >= min_confidence]
            
            # Cache the result
            self._belief_cache[cache_key] = (datetime.now(), beliefs)
            
            # Record performance
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
        Get femdom-related beliefs for the NPC.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of femdom-related beliefs
        """
        start_time = time.time()
        
        try:
            # Get all beliefs
            all_beliefs = await self.get_beliefs()
            
            # Femdom-related keywords
            femdom_keywords = [
                "dominance", "submission", "control", "obedience",
                "discipline", "service", "power", "humiliation",
                "command", "order", "punishment", "reward", "train",
                "serve", "worship", "respect", "protocol", "rule"
            ]
            
            # Filter beliefs
            femdom_beliefs = []
            for belief in all_beliefs:
                text = belief.get("belief", "").lower()
                confidence = belief.get("confidence", 0)
                
                # Check if belief contains femdom keywords
                if any(keyword in text for keyword in femdom_keywords) and confidence >= min_confidence:
                    femdom_beliefs.append(belief)
            
            # Sort by confidence
            femdom_beliefs.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Record performance
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            
            return femdom_beliefs
            
        except Exception as e:
            logger.error(f"Error getting femdom beliefs: {e}")
            elapsed = time.time() - start_time
            self.performance.record_operation("belief_operations", elapsed)
            return []
    
    async def run_memory_maintenance(self, include_femdom_maintenance: bool = True) -> Dict[str, Any]:
        """
        Run maintenance tasks on the NPC's memory system with femdom-specific optimizations.
        
        Args:
            include_femdom_maintenance: Whether to run femdom-specific maintenance
            
        Returns:
            Results of maintenance operations
        """
        start_time = time.time()
        
        try:
            memory_system = await self._get_memory_system()
            
            # Run standard maintenance
            results = await memory_system.maintain(
                entity_type="npc",
                entity_id=self.npc_id
            )
            
            # Run femdom-specific maintenance if requested
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
            
            # Record performance
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance completed in {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running memory maintenance: {e}")
            elapsed = time.time() - start_time
            logger.info(f"Memory maintenance failed after {elapsed:.2f}s")
            return {"error": str(e)}
    
    async def _run_femdom_maintenance(self) -> Dict[str, Any]:
        """Run femdom-specific memory maintenance tasks."""
        results = {
            "power_dynamic_memories_processed": 0,
            "dominance_memories_consolidated": 0,
            "submission_memories_consolidated": 0,
            "power_beliefs_reinforced": 0
        }
        
        try:
            memory_system = await self._get_memory_system()
            
            # 1. Consolidate repetitive power dynamic memories
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline", 
                "service", "submission", "humiliation", "ownership"
            ]
            
            for tag in femdom_tags:
                # Get memories with this tag
                memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    query=f"tags:{tag}",
                    limit=20
                )
                
                results["power_dynamic_memories_processed"] += len(memories)
                
                # Group similar memories
                memory_groups = {}
                for memory in memories:
                    # Create a simplified version of the text for grouping
                    text = memory.get("text", "").lower()
                    
                    # Extract key elements based on tag
                    key_elements = []
                    if tag == "dominance_dynamic":
                        for word in ["command", "control", "dominate", "authority"]:
                            if word in text:
                                key_elements.append(word)
                    elif tag == "submission":
                        for word in ["obey", "submit", "follow", "comply"]:
                            if word in text:
                                key_elements.append(word)
                    
                    # Create group key
                    group_key = " ".join(sorted(key_elements)) if key_elements else text[:20]
                    
                    if group_key not in memory_groups:
                        memory_groups[group_key] = []
                    
                    memory_groups[group_key].append(memory)
                
                # Consolidate groups with multiple memories
                for group_key, group_memories in memory_groups.items():
                    if len(group_memories) >= 3:
                        # Create consolidated memory
                        memory_texts = [m.get("text", "") for m in group_memories]
                        ids = [m.get("id") for m in group_memories]
                        
                        consolidated_text = f"I have {len(memory_texts)} similar experiences involving {tag}: " + \
                                           f"'{memory_texts[0]}' and similar events."
                        
                        # Create consolidated memory
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
            
            # 2. Reinforce power-related beliefs based on memory evidence
            # Get femdom beliefs
            femdom_beliefs = await self.get_femdom_beliefs()
            
            for belief in femdom_beliefs:
                belief_text = belief.get("belief", "")
                confidence = belief.get("confidence", 0.5)
                
                # Search for supporting memories
                supporting_memories = await memory_system.search_memories(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    query=belief_text,
                    limit=5
                )
                
                # Count strong supporting memories
                significant_support = sum(1 for m in supporting_memories 
                                        if m.get("significance", 0) >= 4)
                
                # Update confidence based on memory support
                if significant_support >= 3:
                    # Strong support - increase confidence
                    new_confidence = min(0.95, confidence + 0.1)
                    
                    await memory_system.update_belief_confidence(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        belief_id=belief.get("id"),
                        new_confidence=new_confidence,
                        reason=f"Reinforced by {significant_support} significant memories"
                    )
                    
                    results["power_beliefs_reinforced"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in femdom maintenance: {e}")
            return {"error": str(e)}
    
    # Other methods from original class...
    # These methods remain unchanged from the original implementation:
    # - analyze_memory_content(self, memory_text: str)
    # - calculate_emotional_intensity(self, memory_text: str, base_valence: float)
    # - apply_npc_memory_bias(self, conn, memory_id: int)
    # - propagate_memory(self, conn, memory_text: str, tags, significance, emotional_intensity)
    # - distort_text(self, original_text: str, severity=0.3)
    # - categorize_memories_by_significance(self, retention_threshold=5)
    # - apply_recency_bias(self, memories: List[dict])
    # - apply_emotional_bias(self, memories: List[dict])
    # - apply_personality_bias(self, memories: List[dict])
    # They would be included here in a complete implementation
    
    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        """
        Analyze memory text content to extract relevant tags.
        Enhanced with femdom-specific categories and themes.
        """
        tags = []
        lower_text = memory_text.lower()
        
        # Emotional content
        if any(word in lower_text for word in ["angry", "upset", "mad", "furious", "betrayed"]):
            tags.append("negative_emotion")
        if any(word in lower_text for word in ["happy", "pleased", "joy", "delighted", "thrilled"]):
            tags.append("positive_emotion")
        if any(word in lower_text for word in ["afraid", "scared", "fearful", "terrified"]):
            tags.append("fear")
        if any(word in lower_text for word in ["aroused", "excited", "turned on", "desire"]):
            tags.append("arousal")
        
        # Player-related
        if "player" in lower_text or "user" in lower_text or "chase" in lower_text:
            tags.append("player_related")
        
        # Rumor or secondhand
        if "heard" in lower_text or "told me" in lower_text or "said that" in lower_text:
            tags.append("rumor")
        
        # Social interactions
        if any(word in lower_text for word in ["helped", "assisted", "supported", "saved"]):
            tags.append("positive_interaction")
        if any(word in lower_text for word in ["betrayed", "attacked", "deceived", "tricked"]):
            tags.append("negative_interaction")
        
        # FEMDOM SPECIFIC TAGS
        
        # Dominance dynamics
        if any(word in lower_text for word in ["command", "ordered", "instructed", "demanded"]):
            tags.append("dominance_dynamic")
        if any(word in lower_text for word in ["obey", "comply", "submit", "kneel", "bow"]):
            tags.append("submission")
        
        # Punishment/discipline
        if any(word in lower_text for word in ["punish", "discipline", "correct", "consequences"]):
            tags.append("discipline")
        if any(word in lower_text for word in ["spank", "whip", "paddle", "impact"]):
            tags.append("physical_discipline")
        
        # Humiliation
        if any(word in lower_text for word in ["humiliate", "embarrass", "shame", "mock"]):
            tags.append("humiliation")
        
        # Training/conditioning
        if any(word in lower_text for word in ["train", "condition", "learn", "lesson", "teach"]):
            tags.append("training")
        
        # Control
        if any(word in lower_text for word in ["control", "restrict", "limit", "permission"]):
            tags.append("control")
        
        # Worship/service
        if any(word in lower_text for word in ["worship", "serve", "service", "please", "satisfy"]):
            tags.append("service")
        
        # Devotion/loyalty
        if any(word in lower_text for word in ["devoted", "loyal", "faithful", "belonging"]):
            tags.append("devotion")
        
        # Resistance
        if any(word in lower_text for word in ["resist", "disobey", "refuse", "defy"]):
            tags.append("resistance")
        
        # Power exchange
        if any(word in lower_text for word in ["power", "exchange", "protocol", "dynamic", "role"]):
            tags.append("power_exchange")
        
        # Praise
        if any(word in lower_text for word in ["praise", "good", "well done", "proud"]):
            tags.append("praise")
        
        # Rituals
        if any(word in lower_text for word in ["ritual", "ceremony", "protocol", "procedure"]):
            tags.append("ritual")
        
        # Collaring and ownership
        if any(word in lower_text for word in ["collar", "own", "belong", "possess", "property"]):
            tags.append("ownership")
        
        # Psychological
        if any(word in lower_text for word in ["mind", "psyche", "thoughts", "mental", "mindset"]):
            tags.append("psychological")
        
        return tags
    
    async def calculate_emotional_intensity(self, memory_text: str, base_valence: float) -> float:
        """
        Calculate emotional intensity from text content and base valence.
        
        Args:
            memory_text: The memory text 
            base_valence: Base emotional valence (-10 to 10)
            
        Returns:
            Emotional intensity (0-100)
        """
        # Convert valence to base intensity
        intensity = (base_valence + 10) * 5  # Map [-10,10] to [0,100]
        
        # Emotional words and their intensity boost values
        emotion_words = {
            "furious": 20, "ecstatic": 20, "devastated": 20, "thrilled": 20,
            "angry": 15, "delighted": 15, "sad": 15, "happy": 15,
            "annoyed": 10, "pleased": 10, "upset": 10, "glad": 10,
            "concerned": 5, "fine": 5, "worried": 5, "okay": 5
        }
        
        # Femdom-specific emotional words
        femdom_emotion_words = {
            "humiliated": 25, "dominated": 22, "controlled": 20, "obedient": 18,
            "submissive": 18, "powerful": 20, "superior": 15, "worshipped": 22
        }
        
        # Scan text for emotional keywords
        lower_text = memory_text.lower()
        
        # Check standard emotional words
        for word, boost in emotion_words.items():
            if word in lower_text:
                intensity += boost
                break
                
        # Check femdom-specific words
        for word, boost in femdom_emotion_words.items():
            if word in lower_text:
                intensity += boost
                break
        
        return float(min(100, max(0, intensity)))
    
    async def apply_npc_memory_bias(self, conn, memory_id: int):
        """
        Adjust memory confidence based on NPC personality type.
        
        Args:
            conn: Database connection
            memory_id: ID of the memory to adjust
        """
        personality_factors = {
            "gullible": 1.2,   # More likely to believe (higher confidence)
            "skeptical": 0.8,  # Less likely to believe (lower confidence)
            "paranoid": 1.5,   # Much more likely to believe negative things
            "neutral": 1.0,    # No bias
            "dominant": 1.1,   # Slight bias toward confidence (femdom)
            "submissive": 0.9  # Slight bias against confidence (femdom)
        }
        
        factor = personality_factors.get(self.npc_personality, 1.0)
        
        try:
            cursor = conn.cursor()
            
            # For paranoid NPCs, check if the memory has negative connotations
            if self.npc_personality == "paranoid":
                cursor.execute("""
                    SELECT memory_text FROM NPCMemories WHERE id = %s
                """, (memory_id,))
                
                row = cursor.fetchone()
                if row:
                    memory_text = row[0]
                    lower_text = memory_text.lower()
                    
                    if any(word in lower_text for word in ["betray", "trick", "lie", "deceive", "attack"]):
                        factor = 1.5  # Higher confidence in negative memories
                    else:
                        factor = 0.9  # Lower confidence in positive/neutral memories
            
            # For dominant NPCs, boost confidence in dominance-related memories
            elif self.npc_personality == "dominant":
                cursor.execute("""
                    SELECT memory_text, tags FROM NPCMemories WHERE id = %s
                """, (memory_id,))
                
                row = cursor.fetchone()
                if row and len(row) >= 2:
                    memory_text, tags = row
                    
                    # Check if memory has dominance tags
                    if tags and any(tag in tags for tag in ["dominance_dynamic", "control", "discipline"]):
                        factor = 1.3  # Higher confidence in dominance memories
            
            # Apply the confidence adjustment
            cursor.execute("""
                UPDATE NPCMemories
                SET confidence = LEAST(confidence * %s, 1.0)
                WHERE id = %s
            """, (factor, memory_id))
            
        except Exception as e:
            logger.error(f"Error applying personality bias: {e}")
    
    async def apply_recency_bias(self, memories: List[dict]) -> List[dict]:
        """
        Boost recent memories' relevance_score.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        now = datetime.now()
        for mem in memories:
            ts = mem.get("timestamp")
            
            # Parse timestamp if needed
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    days_ago = (now - dt).days
                except ValueError:
                    days_ago = 30  # Default if parsing fails
            elif isinstance(ts, datetime):
                days_ago = (now - ts).days
            else:
                days_ago = 30  # Default if no timestamp
            
            # Calculate recency factor (1.0 for very recent, decreasing over time)
            recency_factor = max(0, 30 - days_ago) / 30.0
            
            # Add to relevance score
            mem["relevance_score"] = mem.get("relevance_score", 0) + recency_factor * 5.0
        
        return memories
    
    async def apply_emotional_bias(self, memories: List[dict]) -> List[dict]:
        """
        Adjust memory relevance based on emotional intensity and significance.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        for mem in memories:
            # Normalize values to 0-1 range
            emotional_intensity = mem.get("emotional_intensity", 0) / 100.0
            significance = mem.get("significance", 0) / 10.0
            
            # Add to relevance score
            mem["relevance_score"] = mem.get("relevance_score", 0) + (emotional_intensity * 3.0 + significance * 2.0)
            
            # Check for femdom tags for additional boost
            tags = mem.get("tags", [])
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline", 
                "service", "submission", "humiliation", "ownership"
            ]
            
            if any(tag in femdom_tags for tag in tags):
                # Boost femdom-related memories
                mem["relevance_score"] += 2.0
        
        return memories
    
    async def apply_personality_bias(self, memories: List[dict]) -> List[dict]:
        """
        Apply personality-specific biases to memory relevance.
        Enhanced for femdom contexts.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        for mem in memories:
            text = mem.get("text", "").lower()
            tags = mem.get("tags", [])
            
            # Personality-specific biases
            if self.npc_personality == "paranoid":
                # Paranoid NPCs prioritize threatening or negative memories
                if any(word in text for word in ["threat", "danger", "betray", "attack"]):
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
                if "negative_emotion" in tags or "negative_interaction" in tags:
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
            
            elif self.npc_personality == "gullible":
                # Gullible NPCs prioritize secondhand information
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
            
            elif self.npc_personality == "skeptical":
                # Skeptical NPCs deprioritize rumors and secondhand info
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] = mem.get("relevance_score", 0) - 1.5
            
            # Femdom-specific personality biases
            elif self.npc_personality == "dominant":
                # Dominant NPCs prioritize dominance, control, and discipline memories
                if any(tag in tags for tag in ["dominance_dynamic", "control", "discipline"]):
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
                if "power_exchange" in tags:
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
            
            elif self.npc_personality == "submissive":
                # Submissive NPCs prioritize service, submission, and obedience memories
                if any(tag in tags for tag in ["submission", "service", "obedience"]):
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 3.0
                if "ownership" in tags:
                    mem["relevance_score"] = mem.get("relevance_score", 0) + 2.0
            
            # Adjust by confidence
            confidence = mem.get("confidence", 1.0)
            mem["relevance_score"] = mem.get("relevance_score", 0) * confidence
        
        return memories
    
    async def propagate_memory(self, conn, memory_text: str, tags: List[str], significance: int, emotional_intensity: float):
        """
        Propagate important memories to related NPCs as secondhand information.
        Enhanced for femdom contexts.
        
        Args:
            conn: Database connection
            memory_text: The memory text
            tags: Tags for the memory
            significance: Importance of the memory
            emotional_intensity: Emotional intensity of the memory
        """
        try:
            cursor = conn.cursor()
            
            # Find NPCs connected to this NPC
            cursor.execute("""
                SELECT entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND entity1_type = 'npc'
                  AND entity1_id = %s
                  AND entity2_type = 'npc'
            """, (self.user_id, self.conversation_id, self.npc_id))
            
            rows = cursor.fetchall()
            related_npcs = [(row[0], row[1], row[2]) for row in rows]
            
            # Get this NPC's name
            cursor.execute("""
                SELECT npc_name
                FROM NPCStats
                WHERE user_id = %s 
                  AND conversation_id = %s
                  AND npc_id = %s
            """, (self.user_id, self.conversation_id, self.npc_id))
            
            row = cursor.fetchone()
            npc_name = row[0] if row else f"NPC_{self.npc_id}"
            
            # Check if memory has femdom context
            femdom_tags = [
                "dominance_dynamic", "power_exchange", "discipline", 
                "service", "submission", "humiliation", "ownership"
            ]
            
            has_femdom_context = any(tag in femdom_tags for tag in tags)
            
            # Create secondhand memories for each related NPC
            for related_id, link_type, link_level in related_npcs:
                # Distort the message slightly based on relationship
                distortion_severity = 0.3
                
                # Less distortion for close relationships
                if link_level > 75:
                    distortion_severity = 0.1
                elif link_level > 50:
                    distortion_severity = 0.2
                # More distortion for hostile relationships
                elif link_level < 25:
                    distortion_severity = 0.5
                
                distorted_text = self.distort_text(memory_text, severity=distortion_severity)
                secondhand_text = f"I heard that {npc_name} {distorted_text}"
                
                # Lower significance and intensity for secondhand information
                secondhand_significance = max(1, significance - 2)
                secondhand_intensity = max(0, emotional_intensity - 20)
                
                # If femdom context, make the distortion more extreme for certain relationships
                secondhand_tags = tags + ["secondhand", "rumor"]
                
                if has_femdom_context:
                    # Check relationship type
                    if link_type == "submissive":
                        # Submissives might exaggerate dominance displays
                        if "dominance_dynamic" in tags or "control" in tags:
                            secondhand_text = f"I heard that {npc_name} was extremely dominant when {distorted_text}"
                            secondhand_tags.append("exaggerated")
                    
                    elif link_type == "dominant":
                        # Dominants might diminish other dominants' displays
                        if "dominance_dynamic" in tags or "control" in tags:
                            secondhand_text = f"I heard that {npc_name} tried to act dominant by {distorted_text}"
                            secondhand_tags.append("diminished")
                
                # Add the secondhand memory
                cursor.execute("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type, tags,
                        emotional_intensity, significance, status,
                        confidence, is_consolidated
                    )
                    VALUES (
                        %s, %s, 'secondhand', %s,
                        %s, %s, 'active',
                        0.7, FALSE
                    )
                """, (
                    related_id, 
                    secondhand_text, 
                    secondhand_tags,
                    secondhand_intensity,
                    secondhand_significance
                ))
            
            logger.debug(f"Propagated memory to {len(related_npcs)} related NPCs")
        except Exception as e:
            logger.error(f"Error propagating memory: {e}")
    
    def distort_text(self, original_text: str, severity=0.3) -> str:
        """
        Word-level partial rewrite for rumor distortion.
        Enhanced for femdom contexts.
        
        Args:
            original_text: The original text to distort
            severity: How much to distort the text (0.0-1.0)
            
        Returns:
            Distorted version of the text
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
        
        # Femdom-specific distortions
        femdom_synonyms = {
            "dominated": ["controlled completely", "took full control of", "overpowered"],
            "commanded": ["ordered", "instructed strictly", "demanded"],
            "punished": ["disciplined", "corrected", "taught a lesson to"],
            "submitted": ["obeyed", "yielded", "surrendered"],
            "praised": ["rewarded", "showed approval to", "acknowledged"],
            "humiliated": ["embarrassed", "shamed", "put in their place"]
        }
        
        # Merge the maps
        all_synonyms = {**synonyms_map, **femdom_synonyms}
        
        words = original_text.split()
        for i in range(len(words)):
            # Chance to modify a word based on severity
            if random.random() < severity:
                word_lower = words[i].lower()
                # Replace with synonym if available
                if word_lower in all_synonyms:
                    words[i] = random.choice(all_synonyms[word_lower])
                # Small chance to delete a word
                elif random.random() < 0.2:
                    words[i] = ""
        
        # Reconstruct text, removing empty strings
        return " ".join([w for w in words if w])
