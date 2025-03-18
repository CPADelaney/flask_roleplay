# context/memory_manager.py

import asyncio
import logging
import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import hashlib
from collections import defaultdict

from context.unified_cache import context_cache
from context.context_config import get_config
from context.vector_service import get_vector_service

logger = logging.getLogger(__name__)

class Memory:
    """Unified memory representation with metadata"""
    
    def __init__(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "observation",
        created_at: Optional[datetime] = None,
        importance: float = 0.5,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.memory_id = memory_id
        self.content = content
        self.memory_type = memory_type
        self.created_at = created_at or datetime.now()
        self.importance = importance
        self.access_count = access_count
        self.last_accessed = last_accessed or self.created_at
        self.tags = tags or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        last_accessed = datetime.fromisoformat(data["last_accessed"]) if isinstance(data["last_accessed"], str) else data["last_accessed"]
        
        memory = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            created_at=timestamp,
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        return memory
    
    def access(self) -> None:
        """Record an access to this memory"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_importance(self) -> float:
        """Calculate and update the importance score"""
        # Base score by type
        type_scores = {
            "observation": 0.3,
            "event": 0.5,
            "scene": 0.4,
            "decision": 0.6,
            "character_development": 0.7,
            "game_mechanic": 0.5,
            "quest": 0.8,
            "summary": 0.7
        }
        type_score = type_scores.get(self.memory_type, 0.4)
        
        # Recency factor (newer = more important)
        age_days = (datetime.now() - self.created_at).days
        recency = 1.0 if age_days < 1 else max(0.0, 1.0 - (age_days / 30))
        
        # Access score (frequently accessed = more important)
        access_score = min(0.3, 0.05 * math.log(1 + self.access_count))
        
        # Calculate final score
        self.importance = min(1.0, (type_score * 0.5) + (recency * 0.3) + (access_score * 0.2))
        
        return self.importance


class MemoryManager:
    """Integrated memory manager for storage, retrieval, and consolidation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.memories = {}  # In-memory cache of recent/important memories
        self.memory_index = {}  # Indices for efficient lookup
        self.is_initialized = False
        self.vector_service = VectorService()
        self.db = Database()
        self.memory_indices = {
            "by_type": defaultdict(list),
            "by_importance": defaultdict(list),
            "by_recency": defaultdict(list)
        }
        self.type_scores = {
            "event": 1.0,
            "relationship": 0.8,
            "knowledge": 0.6,
            "emotion": 0.4
        }
        self.importance_weights = {
            "recency": 0.4,
            "access_frequency": 0.3,
            "type": 0.3
        }
        self.recency_weights = {
            "hour": 1.0,
            "day": 0.8,
            "week": 0.6,
            "month": 0.4
        }
        self.include_rules = set()
        self.exclude_rules = set()
        self.importance_threshold = 0.5
        
        # NEW: Nyx directive handling
        self.nyx_directives = {}
        self.nyx_overrides = {}
        self.nyx_prohibitions = {}
        
        # NEW: Nyx governance integration
        self.governance = None
        self.directive_handler = None
    
    async def initialize(self, user_id: int, conversation_id: int):
        """Initialize the memory manager."""
        try:
            # Store user and conversation IDs
            self.user_id = user_id
            self.conversation_id = conversation_id
            
            # Initialize Nyx integration
            await self._initialize_nyx_integration(user_id, conversation_id)
            
            # Load important memories
            await self._load_important_memories()
            
            # Build memory indices
            await self._build_memory_indices()
            
            logger.info("Memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memory manager: {e}")
            raise

    async def _initialize_nyx_integration(self, user_id: int, conversation_id: int):
        """Initialize Nyx governance integration."""
        try:
            from nyx.integrate import get_central_governance
            from nyx.directive_handler import DirectiveHandler
            from nyx.nyx_governance import AgentType
            
            # Get governance system
            self.governance = await get_central_governance(user_id, conversation_id)
            
            # Initialize directive handler
            self.directive_handler = DirectiveHandler(
                user_id,
                conversation_id,
                AgentType.MEMORY_MANAGER,
                "memory_manager"
            )
            
            # Register handlers
            self.directive_handler.register_handler("action", self._handle_action_directive)
            self.directive_handler.register_handler("override", self._handle_override_directive)
            self.directive_handler.register_handler("prohibition", self._handle_prohibition_directive)
            
            logger.info("Initialized Nyx integration for memory manager")
        except Exception as e:
            logger.error(f"Error initializing Nyx integration: {e}")

    async def _handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx."""
        instruction = directive.get("instruction", "")
        logging.info(f"[MemoryManager] Processing action directive: {instruction}")
        
        if "consolidate_memories" in instruction.lower():
            # Apply memory consolidation
            params = directive.get("parameters", {})
            consolidation_rules = params.get("consolidation_rules", {})
            
            # Consolidate memories
            await self._consolidate_memories(consolidation_rules)
            
            return {"result": "memories_consolidated"}
            
        elif "prioritize_memories" in instruction.lower():
            # Apply memory prioritization
            params = directive.get("parameters", {})
            priority_rules = params.get("priority_rules", {})
            
            # Update priority rules
            self._update_priority_rules(priority_rules)
            
            # Re-prioritize memories
            await self._prioritize_memories()
            
            return {"result": "memories_prioritized"}
            
        elif "filter_memories" in instruction.lower():
            # Apply memory filtering
            params = directive.get("parameters", {})
            filter_rules = params.get("filter_rules", {})
            
            # Update filter rules
            self._update_filter_rules(filter_rules)
            
            # Apply filters
            await self._apply_memory_filters()
            
            return {"result": "memories_filtered"}
        
        return {"result": "action_not_recognized"}

    async def _handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx."""
        logging.info(f"[MemoryManager] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        applies_to = directive.get("applies_to", [])
        
        # Store override
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_overrides[directive_id] = {
                "action": override_action,
                "applies_to": applies_to
            }
        
        return {"result": "override_stored"}

    async def _handle_prohibition_directive(self, directive: dict) -> dict:
        """Handle a prohibition directive from Nyx."""
        logging.info(f"[MemoryManager] Processing prohibition directive")
        
        # Extract prohibition details
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        # Store prohibition
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_prohibitions[directive_id] = {
                "prohibited_actions": prohibited_actions,
                "reason": reason
            }
        
        return {"result": "prohibition_stored"}

    def _update_priority_rules(self, rules: Dict[str, Any]) -> None:
        """Update priority rules based on Nyx directive."""
        try:
            # Update type scores
            if "type_scores" in rules:
                self.type_scores.update(rules["type_scores"])
            
            # Update importance weights
            if "importance_weights" in rules:
                self.importance_weights.update(rules["importance_weights"])
            
            # Update recency weights
            if "recency_weights" in rules:
                self.recency_weights.update(rules["recency_weights"])
            
            logger.info("Updated priority rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating priority rules: {e}")

    def _update_filter_rules(self, rules: Dict[str, Any]) -> None:
        """Update filter rules based on Nyx directive."""
        try:
            # Update inclusion rules
            if "include_only" in rules:
                self.include_rules = set(rules["include_only"])
            
            # Update exclusion rules
            if "exclude" in rules:
                self.exclude_rules = set(rules["exclude"])
            
            # Update importance threshold
            if "importance_threshold" in rules:
                self.importance_threshold = rules["importance_threshold"]
            
            logger.info("Updated filter rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating filter rules: {e}")

    async def _consolidate_memories(self, rules: Dict[str, Any]) -> None:
        """Consolidate memories based on Nyx rules."""
        try:
            # Get memories to consolidate
            memories = await self._get_memories_to_consolidate(rules)
            
            # Group memories
            grouped_memories = self._group_memories(memories, rules)
            
            # Generate summaries
            summaries = await self._generate_memory_summaries(grouped_memories, rules)
            
            # Store consolidated memories
            await self._store_consolidated_memories(summaries)
            
            logger.info("Consolidated memories based on Nyx rules")
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")

    async def _get_memories_to_consolidate(self, rules: Dict[str, Any]) -> List[Memory]:
        """Get memories that should be consolidated based on rules."""
        try:
            query = """
                SELECT * FROM memories
                WHERE user_id = $1 AND conversation_id = $2
                AND timestamp >= $3
                AND importance >= $4
            """
            
            # Get parameters from rules
            time_window = rules.get("time_window", 86400)  # 24 hours default
            min_importance = rules.get("min_importance", 0.5)
            
            # Calculate timestamp threshold
            threshold = datetime.now() - timedelta(seconds=time_window)
            
            # Execute query
            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    self.user_id,
                    self.conversation_id,
                    threshold,
                    min_importance
                )
            
            # Convert to Memory objects
            return [Memory.from_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting memories to consolidate: {e}")
            return []

    def _group_memories(self, memories: List[Memory], 
                       rules: Dict[str, Any]) -> Dict[str, List[Memory]]:
        """Group memories based on rules."""
        try:
            grouped = defaultdict(list)
            
            # Get grouping key from rules
            group_by = rules.get("group_by", "type")
            
            # Group memories
            for memory in memories:
                if group_by == "type":
                    key = memory.type
                elif group_by == "importance":
                    key = self._get_importance_bracket(memory.importance)
                elif group_by == "recency":
                    key = self._get_recency_bracket(memory.timestamp)
                else:
                    key = getattr(memory, group_by, "unknown")
                
                grouped[key].append(memory)
            
            return grouped
        except Exception as e:
            logger.error(f"Error grouping memories: {e}")
            return {}

    def _get_importance_bracket(self, importance: float) -> str:
        """Get importance bracket for grouping."""
        if importance >= 0.8:
            return "critical"
        elif importance >= 0.6:
            return "high"
        elif importance >= 0.4:
            return "medium"
        else:
            return "low"

    def _get_recency_bracket(self, timestamp: datetime) -> str:
        """Get recency bracket for grouping."""
        age = (datetime.now() - timestamp).total_seconds()
        
        if age < 3600:  # 1 hour
            return "recent"
        elif age < 86400:  # 24 hours
            return "today"
        elif age < 604800:  # 1 week
            return "week"
        else:
            return "old"

    async def _generate_memory_summaries(self, grouped_memories: Dict[str, List[Memory]], 
                                       rules: Dict[str, Any]) -> List[Memory]:
        """Generate summaries for groups of memories."""
        try:
            summaries = []
            
            for key, memories in grouped_memories.items():
                # Sort memories by importance and recency
                sorted_memories = sorted(
                    memories,
                    key=lambda m: (m.importance, m.timestamp),
                    reverse=True
                )
                
                # Get top memories
                top_memories = sorted_memories[:rules.get("max_memories_per_group", 3)]
                
                # Generate summary
                summary = Memory(
                    id=f"summary_{key}_{datetime.now().timestamp()}",
                    type="summary",
                    content=self._generate_summary_content(top_memories),
                    importance=self._calculate_group_importance(top_memories),
                    timestamp=datetime.now(),
                    metadata={
                        "group_key": key,
                        "memory_count": len(memories),
                        "time_span": {
                            "start": min(m.timestamp for m in memories),
                            "end": max(m.timestamp for m in memories)
                        }
                    }
                )
                
                summaries.append(summary)
            
            return summaries
        except Exception as e:
            logger.error(f"Error generating memory summaries: {e}")
            return []

    def _generate_summary_content(self, memories: List[Memory]) -> str:
        """Generate content for a memory summary."""
        try:
            # Get key points from memories
            key_points = []
            for memory in memories:
                if memory.type == "event":
                    key_points.append(f"- {memory.content}")
                elif memory.type == "relationship":
                    key_points.append(f"* {memory.content}")
                else:
                    key_points.append(f"• {memory.content}")
            
            # Generate summary
            summary = "Summary of related memories:\n"
            summary += "\n".join(key_points)
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary content: {e}")
            return "Error generating summary"

    def _calculate_group_importance(self, memories: List[Memory]) -> float:
        """Calculate importance for a group of memories."""
        try:
            if not memories:
                return 0.0
            
            # Calculate weighted average importance
            total_weight = 0.0
            weighted_sum = 0.0
            
            for memory in memories:
                # Weight by recency
                age = (datetime.now() - memory.timestamp).total_seconds()
                weight = 1.0 / (1.0 + age / 86400)  # Decay over 24 hours
                
                weighted_sum += memory.importance * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating group importance: {e}")
            return 0.0

    async def _store_consolidated_memories(self, summaries: List[Memory]) -> None:
        """Store consolidated memory summaries."""
        try:
            query = """
                INSERT INTO memories (
                    id, type, content, importance, timestamp, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            async with self.db.pool.acquire() as conn:
                for summary in summaries:
                    await conn.execute(
                        query,
                        summary.id,
                        summary.type,
                        summary.content,
                        summary.importance,
                        summary.timestamp,
                        json.dumps(summary.metadata)
                    )
            
            logger.info(f"Stored {len(summaries)} consolidated memories")
        except Exception as e:
            logger.error(f"Error storing consolidated memories: {e}")

    async def _prioritize_memories(self) -> None:
        """Re-prioritize memories based on current rules."""
        try:
            # Get all memories
            memories = await self._get_all_memories()
            
            # Calculate new importance scores
            for memory in memories:
                memory.importance = self._calculate_memory_importance(memory)
            
            # Update memory importance in database
            await self._update_memory_importance(memories)
            
            logger.info("Re-prioritized memories")
        except Exception as e:
            logger.error(f"Error prioritizing memories: {e}")

    def _calculate_memory_importance(self, memory: Memory) -> float:
        """Calculate importance score for a memory."""
        try:
            importance = 0.0
            
            # Type-based importance
            type_score = self.type_scores.get(memory.type, 0.5)
            importance += type_score * self.importance_weights["type"]
            
            # Recency-based importance
            age = (datetime.now() - memory.timestamp).total_seconds()
            recency_score = 1.0 / (1.0 + age / 86400)  # Decay over 24 hours
            importance += recency_score * self.importance_weights["recency"]
            
            # Access frequency importance
            access_count = memory.metadata.get("access_count", 0)
            access_score = min(1.0, access_count / 10)  # Cap at 10 accesses
            importance += access_score * self.importance_weights["access_frequency"]
            
            return min(1.0, importance)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error calculating memory importance: {e}")
            return 0.5

    async def _update_memory_importance(self, memories: List[Memory]) -> None:
        """Update importance scores for memories in database."""
        try:
            query = """
                UPDATE memories
                SET importance = $1
                WHERE id = $2
            """
            
            async with self.db.pool.acquire() as conn:
                for memory in memories:
                    await conn.execute(
                        query,
                        memory.importance,
                        memory.id
                    )
            
            logger.info(f"Updated importance for {len(memories)} memories")
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")

    async def _apply_memory_filters(self) -> None:
        """Apply current filter rules to memories."""
        try:
            # Get all memories
            memories = await self._get_all_memories()
            
            # Apply filters
            filtered_memories = []
            for memory in memories:
                if self._should_include_memory(memory):
                    filtered_memories.append(memory)
            
            # Update memory indices
            self.memory_indices = {
                "by_type": defaultdict(list),
                "by_importance": defaultdict(list),
                "by_recency": defaultdict(list)
            }
            
            for memory in filtered_memories:
                self._add_to_indices(memory)
            
            logger.info("Applied memory filters")
        except Exception as e:
            logger.error(f"Error applying memory filters: {e}")

    def _should_include_memory(self, memory: Memory) -> bool:
        """Check if memory should be included based on filter rules."""
        try:
            # Check inclusion rules
            if self.include_rules and memory.type not in self.include_rules:
                return False
            
            # Check exclusion rules
            if self.exclude_rules and memory.type in self.exclude_rules:
                return False
            
            # Check importance threshold
            if memory.importance < self.importance_threshold:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking memory inclusion: {e}")
            return False

    async def _load_important_memories(self, user_id: int, conversation_id: int):
        """Load important memories into local cache"""
        # Use cache key for important memories
        cache_key = f"important_memories:{user_id}:{conversation_id}"
        
        async def fetch_important_memories():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Get important memories (high importance or recently accessed)
                    rows = await conn.fetch("""
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2
                          AND (importance >= 0.7 OR last_accessed > NOW() - INTERVAL '7 days')
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT 50
                    """, user_id, conversation_id)
                    
                    # Convert to Memory objects
                    memories = {}
                    for row in rows:
                        # Parse metadata JSON
                        metadata = {}
                        if row["entry_metadata"]:
                            try:
                                metadata = json.loads(row["entry_metadata"])
                            except:
                                pass
                        
                        # Parse tags
                        tags = []
                        if row["tags"]:
                            try:
                                tags = json.loads(row["tags"])
                            except:
                                pass
                        
                        memory = Memory(
                            memory_id=str(row["id"]),
                            content=row["entry_text"],
                            memory_type=row["entry_type"],
                            created_at=row["created_at"],
                            importance=row["importance"],
                            access_count=row["access_count"],
                            last_accessed=row["last_accessed"],
                            tags=tags,
                            metadata=metadata
                        )
                        
                        memories[str(row["id"])] = memory
                    
                    return memories
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error loading important memories: {e}")
                return {}
        
        # Get from cache or fetch (1 hour TTL, medium importance)
        memories = await context_cache.get(
            cache_key, 
            fetch_important_memories, 
            cache_level=2,  # L2 cache
            importance=0.6,
            ttl_override=3600  # 1 hour
        )
        
        # Store in local cache
        self.memories = memories
        
        # Build indices
        self._build_memory_indices()
        
        return len(memories)
    
    def _build_memory_indices(self):
        """Build memory indices for efficient lookup"""
        # Reset indices
        self.memory_index = {
            "type": {},
            "tag": {},
            "timestamp": []
        }
        
        # Build type index
        for memory_id, memory in self.memories.items():
            # Type index
            memory_type = memory.memory_type
            if memory_type not in self.memory_index["type"]:
                self.memory_index["type"][memory_type] = set()
            self.memory_index["type"][memory_type].add(memory_id)
            
            # Tag index
            for tag in memory.tags:
                if tag not in self.memory_index["tag"]:
                    self.memory_index["tag"][tag] = set()
                self.memory_index["tag"][tag].add(memory_id)
            
            # Timestamp index (sorted by created_at)
            self.memory_index["timestamp"].append((memory.created_at, memory_id))
        
        # Sort timestamp index
        self.memory_index["timestamp"].sort(reverse=True)
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "observation",
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_vector: bool = True
    ) -> str:
        """Add a new memory with vector storage integration"""
        # Generate unique memory ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_base = f"{self.user_id}:{self.conversation_id}:{content}:{timestamp}"
        memory_id = f"{timestamp}-{hashlib.md5(hash_base.encode()).hexdigest()[:8]}"
        
        # Create memory object
        memory = Memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Auto-calculate importance if not provided
        if importance is None:
            importance = memory.calculate_importance()
        memory.importance = importance
        
        # Store in database
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Insert into database
                result = await conn.fetchrow("""
                    INSERT INTO PlayerJournal(
                        user_id, conversation_id, entry_type, entry_text, 
                        entry_metadata, importance, access_count, last_accessed, tags
                    )
                    VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, 
                    self.user_id, 
                    self.conversation_id,
                    memory_type,
                    content,
                    json.dumps(metadata) if metadata else None,
                    importance,
                    0,  # access_count
                    datetime.now(),  # last_accessed
                    json.dumps(tags) if tags else None
                )
                
                # Get the database ID
                db_id = result["id"]
                memory.memory_id = str(db_id)
                
                # If the memory is important, store in local cache
                if importance >= 0.5:
                    self.memories[memory.memory_id] = memory
                    self._build_memory_indices()
                
                # Store vector embedding if vector search is enabled
                if store_vector and self.config.is_enabled("use_vector_search"):
                    # Get vector service
                    vector_service = await get_vector_service(self.user_id, self.conversation_id)
                    
                    # Add to vector database
                    await vector_service.add_memory(
                        memory_id=memory.memory_id,
                        content=content,
                        memory_type=memory_type,
                        importance=importance,
                        tags=tags
                    )
                
                return memory.memory_id
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return ""
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID"""
        # Check local cache first
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            return memory
        
        # Use cache for other memories
        cache_key = f"memory:{self.user_id}:{self.conversation_id}:{memory_id}"
        
        async def fetch_memory():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Get memory from database
                    row = await conn.fetchrow("""
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2 AND id = $3
                    """, self.user_id, self.conversation_id, int(memory_id))
                    
                    if not row:
                        return None
                    
                    # Parse metadata JSON
                    metadata = {}
                    if row["entry_metadata"]:
                        try:
                            metadata = json.loads(row["entry_metadata"])
                        except:
                            pass
                    
                    # Parse tags
                    tags = []
                    if row["tags"]:
                        try:
                            tags = json.loads(row["tags"])
                        except:
                            pass
                    
                    # Create Memory object
                    memory = Memory(
                        memory_id=str(row["id"]),
                        content=row["entry_text"],
                        memory_type=row["entry_type"],
                        created_at=row["created_at"],
                        importance=row["importance"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                        tags=tags,
                        metadata=metadata
                    )
                    
                    # Update access count in database
                    await conn.execute("""
                        UPDATE PlayerJournal
                        SET access_count = access_count + 1, last_accessed = NOW()
                        WHERE id = $1
                    """, int(memory_id))
                    
                    # If important, store in local cache
                    if memory.importance >= 0.5:
                        self.memories[memory.memory_id] = memory
                        self._build_memory_indices()
                    
                    return memory
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error getting memory: {e}")
                return None
        
        # Get from cache or fetch (5 minute TTL, medium importance)
        memory = await context_cache.get(
            cache_key, 
            fetch_memory, 
            cache_level=1,  # L1 cache
            importance=0.5,
            ttl_override=300  # 5 minutes
        )
        
        if memory:
            memory.access()
            
        return memory
    
    async def search_memories(
        self,
        query_text: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
        use_vector: bool = True
    ) -> List[Memory]:
        """Search memories by query text with vector search option"""
        cache_key = f"memory_search:{self.user_id}:{self.conversation_id}:{query_text}:{memory_types}:{tags}:{limit}:{use_vector}"
        
        async def perform_search():
            results = []
            
            # Use vector search if enabled
            if use_vector and self.config.is_enabled("use_vector_search"):
                # Get vector service
                vector_service = await get_vector_service(self.user_id, self.conversation_id)
                
                # Search for memories
                vector_results = await vector_service.search_entities(
                    query_text=query_text,
                    entity_types=["memory"],
                    top_k=limit
                )
                
                # Convert results to Memory objects
                for result in vector_results:
                    metadata = result.get("metadata", {})
                    if metadata.get("entity_type") == "memory":
                        memory_id = metadata.get("memory_id")
                        if memory_id:
                            # Get full memory
                            memory = await self.get_memory(memory_id)
                            if memory:
                                results.append(memory)
            
            # If we didn't get enough results or vector search is disabled,
            # fall back to database search
            if len(results) < limit:
                db_limit = limit - len(results)
                db_results = await self._search_memories_in_db(
                    query_text, memory_types, tags, db_limit
                )
                
                # Add unique results
                existing_ids = {m.memory_id for m in results}
                for memory in db_results:
                    if memory.memory_id not in existing_ids:
                        results.append(memory)
                        existing_ids.add(memory.memory_id)
            
            # Record access for retrieved memories
            for memory in results:
                memory.access()
            
            return results
        
        # Get from cache or search (30 second TTL, importance based on query)
        importance = min(0.7, 0.3 + (len(query_text) / 100))
        return await context_cache.get(
            cache_key, 
            perform_search, 
            cache_level=1,  # L1 cache
            importance=importance,
            ttl_override=30  # 30 seconds
        )
    
    async def _search_memories_in_db(
        self,
        query_text: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Memory]:
        """Search memories in database using text search"""
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Prepare query
                query = """
                    SELECT id, entry_type, entry_text, entry_metadata, 
                           created_at, importance, access_count, last_accessed, tags
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                """
                params = [self.user_id, self.conversation_id]
                
                # Add text search condition
                if query_text:
                    query += f" AND entry_text ILIKE ${len(params) + 1}"
                    params.append(f"%{query_text}%")
                
                # Add memory type filter
                if memory_types:
                    placeholders = ', '.join(f'${i}' for i in range(len(params) + 1, len(params) + len(memory_types) + 1))
                    query += f" AND entry_type IN ({placeholders})"
                    params.extend(memory_types)
                
                # Add ordering and limit
                query += " ORDER BY importance DESC, last_accessed DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)
                
                # Execute query
                rows = await conn.fetch(query, *params)
                
                # Process results
                results = []
                for row in rows:
                    # Parse metadata JSON
                    metadata = {}
                    if row["entry_metadata"]:
                        try:
                            metadata = json.loads(row["entry_metadata"])
                        except:
                            pass
                    
                    # Parse tags
                    row_tags = []
                    if row["tags"]:
                        try:
                            row_tags = json.loads(row["tags"])
                        except:
                            pass
                    
                    # Apply tag filter if needed
                    if tags and not any(tag in row_tags for tag in tags):
                        continue
                    
                    # Create Memory object
                    memory = Memory(
                        memory_id=str(row["id"]),
                        content=row["entry_text"],
                        memory_type=row["entry_type"],
                        created_at=row["created_at"],
                        importance=row["importance"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                        tags=row_tags,
                        metadata=metadata
                    )
                    
                    results.append(memory)
                
                return results
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error searching memories in DB: {e}")
            return []
    
    async def get_recent_memories(
        self,
        days: int = 3,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Get recent memories"""
        cache_key = f"recent_memories:{self.user_id}:{self.conversation_id}:{days}:{memory_types}:{limit}"
        
        async def fetch_recent_memories():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Prepare query
                    query = """
                        SELECT id, entry_type, entry_text, entry_metadata, 
                               created_at, importance, access_count, last_accessed, tags
                        FROM PlayerJournal
                        WHERE user_id = $1 AND conversation_id = $2
                          AND created_at > NOW() - INTERVAL '1 day' * $3
                    """
                    params = [self.user_id, self.conversation_id, days]
                    
                    # Add memory type filter
                    if memory_types:
                        placeholders = ', '.join(f'${i}' for i in range(len(params) + 1, len(params) + len(memory_types) + 1))
                        query += f" AND entry_type IN ({placeholders})"
                        params.extend(memory_types)
                    
                    # Add ordering and limit
                    query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
                    params.append(limit)
                    
                    # Execute query
                    rows = await conn.fetch(query, *params)
                    
                    # Process results
                    results = []
                    for row in rows:
                        # Parse metadata JSON
                        metadata = {}
                        if row["entry_metadata"]:
                            try:
                                metadata = json.loads(row["entry_metadata"])
                            except:
                                pass
                        
                        # Parse tags
                        tags = []
                        if row["tags"]:
                            try:
                                tags = json.loads(row["tags"])
                            except:
                                pass
                        
                        # Create Memory object
                        memory = Memory(
                            memory_id=str(row["id"]),
                            content=row["entry_text"],
                            memory_type=row["entry_type"],
                            created_at=row["created_at"],
                            importance=row["importance"],
                            access_count=row["access_count"],
                            last_accessed=row["last_accessed"],
                            tags=tags,
                            metadata=metadata
                        )
                        
                        results.append(memory)
                    
                    return results
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error getting recent memories: {e}")
                return []
        
        # Get from cache or fetch (2 minute TTL, medium importance)
        return await context_cache.get(
            cache_key, 
            fetch_recent_memories, 
            cache_level=1,  # L1 cache
            importance=0.4,
            ttl_override=120  # 2 minutes
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance tasks"""
        # Determine if consolidation is needed based on configuration
        should_consolidate = self.config.get("memory_consolidation", "enabled", True)
        
        if not should_consolidate:
            return {
                "consolidated": False,
                "reason": "Memory consolidation disabled in config"
            }
            
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Count old, unconsolidated memories
                days_threshold = self.config.get("memory_consolidation", "days_threshold", 7)
                min_memories = self.config.get("memory_consolidation", "min_memories_to_consolidate", 5)
                
                cutoff_date = datetime.now() - timedelta(days=days_threshold)
                
                count_row = await conn.fetchrow("""
                    SELECT COUNT(*) as count
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at < $3
                      AND consolidated = FALSE
                """, self.user_id, self.conversation_id, cutoff_date)
                
                if not count_row or count_row["count"] < min_memories:
                    return {
                        "consolidated": False,
                        "reason": f"Not enough old memories to consolidate: {count_row['count'] if count_row else 0} < {min_memories}"
                    }
                
                # At this point, we would perform memory consolidation
                # Simplified version just marks memories as consolidated
                await conn.execute("""
                    UPDATE PlayerJournal
                    SET consolidated = TRUE
                    WHERE user_id = $1 AND conversation_id = $2
                      AND created_at < $3
                      AND consolidated = FALSE
                """, self.user_id, self.conversation_id, cutoff_date)
                
                return {
                    "consolidated": True,
                    "count": count_row["count"],
                    "threshold_days": days_threshold
                }
            
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error during memory maintenance: {e}")
            return {
                "consolidated": False,
                "error": str(e)
            }

    async def _store_consolidated_memories(self, consolidated: Dict[str, Any]) -> None:
        """Store consolidated memories in the database."""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                for mem_type, summary in consolidated.items():
                    await conn.execute("""
                        INSERT INTO ConsolidatedMemories 
                        (user_id, conversation_id, memory_type, summary, created_at)
                        VALUES ($1, $2, $3, $4, NOW())
                    """, self.user_id, self.conversation_id, mem_type, json.dumps(summary))
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error storing consolidated memories: {e}")

    async def _generate_group_summary(self, memories: List[Memory]) -> Dict[str, Any]:
        """Generate a summary for a group of memories."""
        try:
            # Sort by importance and recency
            sorted_memories = sorted(
                memories,
                key=lambda x: (x.importance, x.last_accessed),
                reverse=True
            )
            
            # Get top memories
            top_memories = sorted_memories[:5]
            
            # Generate summary
            summary = {
                "type": memories[0].memory_type,
                "count": len(memories),
                "time_span": {
                    "start": min(m.created_at for m in memories),
                    "end": max(m.created_at for m in memories)
                },
                "key_memories": [m.to_dict() for m in top_memories],
                "importance_score": sum(m.importance for m in memories) / len(memories),
                "tags": list(set(tag for m in memories for tag in m.tags))
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error generating group summary: {e}")
            return {}

    async def consolidate_memories(self, time_window: int = 7) -> None:
        """Consolidate and summarize memories within a time window."""
        try:
            # Get memories in time window
            memories = await self.get_recent_memories(days=time_window)
            
            # Group by type and importance
            grouped_memories = defaultdict(list)
            for memory in memories:
                grouped_memories[memory.memory_type].append(memory)
                
            # Generate summaries for each group
            consolidated = {}
            for mem_type, mems in grouped_memories.items():
                if len(mems) > 5:  # Only consolidate if enough memories
                    summary = await self._generate_group_summary(mems)
                    consolidated[mem_type] = summary
                    
            # Store consolidated memories
            await self._store_consolidated_memories(consolidated)
            
            # Update local cache
            self.memories.update(consolidated)
            self._build_memory_indices()
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")


# Global manager registry
_memory_managers = {}

async def get_memory_manager(user_id: int, conversation_id: int) -> MemoryManager:
    """Get or create a memory manager instance"""
    key = f"{user_id}:{conversation_id}"
    
    if key not in _memory_managers:
        manager = MemoryManager(user_id, conversation_id)
        await manager.initialize()
        _memory_managers[key] = manager
    
    return _memory_managers[key]

async def cleanup_memory_managers():
    """Close all memory managers"""
    global _memory_managers
    
    # Close each manager
    for manager in _memory_managers.values():
        await manager.close()
    
    # Clear registry
    _memory_managers.clear()
