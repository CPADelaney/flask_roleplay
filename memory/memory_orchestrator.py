# memory/memory_orchestrator.py

"""
Memory System Orchestrator

Single access point for all memory operations in the system.
Provides a unified interface for the narrative generator and other components
to interact with the complex memory subsystems.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import json

# Core memory components
from memory.core import (
    UnifiedMemoryManager, 
    Memory, 
    MemoryType, 
    MemoryStatus,
    MemorySignificance,
    MemoryCache
)

# Specialized managers
from memory.managers import (
    NPCMemoryManager,
    NyxMemoryManager,
    PlayerMemoryManager,
    ConflictMemoryManager,
    LoreMemoryManager,
    ContextEvolutionManager
)

# Advanced features
from memory.emotional import EmotionalMemoryManager
from memory.masks import ProgressiveRevealManager, RevealType, RevealSeverity
from memory.flashbacks import FlashbackManager
from memory.interference import MemoryInterferenceManager
from memory.reconsolidation import ReconsolidationManager
from memory.schemas import MemorySchemaManager
from memory.semantic import SemanticMemoryManager

# Memory services
from memory.memory_service import MemoryEmbeddingService
from memory.memory_retriever import MemoryRetrieverAgent

# Integration and wrapper
from memory.integrated import IntegratedMemorySystem, init_memory_system
from memory.wrapper import MemorySystem

# Database and configuration
from memory.connection import get_connection_context
from memory.config import load_config, get_memory_config
from memory.telemetry import MemoryTelemetry
from memory.maintenance import MemoryMaintenance

# Utilities
from logic.game_time_helper import get_game_datetime, get_game_iso_string

logger = logging.getLogger("memory_orchestrator")


class EntityType(str, Enum):
    """Supported entity types in the memory system."""
    PLAYER = "player"
    NPC = "npc"
    NYX = "nyx"  # The DM/narrator
    LOCATION = "location"
    ITEM = "item"
    LORE = "lore"
    CONFLICT = "conflict"
    CONTEXT = "context"


class MemoryOrchestrator:
    """
    Central orchestrator for all memory operations.
    Provides a unified interface for storing, retrieving, and analyzing memories
    across all entity types and subsystems.
    """
    
    _instances = {}  # Singleton pattern per user/conversation
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'MemoryOrchestrator':
        """
        Get or create an orchestrator instance for the given user/conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            MemoryOrchestrator instance
        """
        key = (user_id, conversation_id)
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the orchestrator."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Configuration
        self.config = load_config()
        self.memory_config = get_memory_config()
        
        # Core systems (to be initialized)
        self.integrated_system = None
        self.memory_wrapper = None
        
        # Specialized managers
        self.managers = {}
        self.emotional_manager = None
        self.mask_manager = None
        self.flashback_manager = None
        self.interference_manager = None
        self.reconsolidation_manager = None
        self.schema_manager = None
        self.semantic_manager = None
        
        # Services
        self.embedding_service = None
        self.retriever_agent = None
        
        # Maintenance
        self.maintenance = None
        
        # State
        self.initialized = False
        self.active_context = {}
        self.cache = MemoryCache()
        
    async def initialize(self):
        """Initialize all memory subsystems."""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing Memory Orchestrator for user {self.user_id}, conversation {self.conversation_id}")
            
            # Initialize core systems
            self.integrated_system = await init_memory_system(self.user_id, self.conversation_id)
            self.memory_wrapper = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            
            # Initialize specialized managers
            self.emotional_manager = EmotionalMemoryManager(self.user_id, self.conversation_id)
            self.mask_manager = ProgressiveRevealManager(self.user_id, self.conversation_id)
            self.flashback_manager = FlashbackManager(self.user_id, self.conversation_id)
            self.interference_manager = MemoryInterferenceManager(self.user_id, self.conversation_id)
            self.reconsolidation_manager = ReconsolidationManager(self.user_id, self.conversation_id)
            self.schema_manager = MemorySchemaManager(self.user_id, self.conversation_id)
            self.semantic_manager = SemanticMemoryManager(self.user_id, self.conversation_id)
            
            # Initialize memory services
            self.embedding_service = MemoryEmbeddingService(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                vector_store_type=self.memory_config["vector_store"]["type"],
                embedding_model=self.memory_config["embedding"]["type"],
                config=self.memory_config
            )
            await self.embedding_service.initialize()
            
            self.retriever_agent = MemoryRetrieverAgent(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                llm_type=self.memory_config["llm"]["type"],
                memory_service=self.embedding_service,
                config=self.memory_config
            )
            await self.retriever_agent.initialize()
            
            # Initialize maintenance
            self.maintenance = MemoryMaintenance()
            
            self.initialized = True
            logger.info("Memory Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Orchestrator: {e}")
            raise
    
    # ========================================================================
    # Core Memory Operations
    # ========================================================================
    
    async def store_memory(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        memory_text: str,
        importance: Union[str, MemorySignificance] = "medium",
        emotional: bool = True,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Store a new memory for any entity type.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            memory_text: The memory content
            importance: Importance level
            emotional: Whether to analyze emotional content
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional parameters for specific memory types
            
        Returns:
            Created memory information including ID and analysis results
        """
        if not self.initialized:
            await self.initialize()
        
        # Normalize entity type
        if isinstance(entity_type, str):
            entity_type = entity_type.lower()
        else:
            entity_type = entity_type.value
        
        # Start telemetry
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze emotional content if requested
            emotion_analysis = None
            if emotional and entity_type in ["npc", "player"]:
                emotion_analysis = await self.emotional_manager.analyze_emotional_content(
                    text=memory_text,
                    context=self.active_context.get("emotional_context")
                )
            
            # Use integrated system for comprehensive processing
            result = await self.integrated_system.add_memory(
                entity_type=entity_type,
                entity_id=entity_id,
                memory_text=memory_text,
                memory_kwargs={
                    "significance": importance if isinstance(importance, int) else self._parse_importance(importance),
                    "tags": tags or [],
                    "metadata": metadata or {},
                    "emotional_intensity": int(emotion_analysis["intensity"] * 100) if emotion_analysis else 0,
                    "apply_schemas": kwargs.get("apply_schemas", True),
                    "check_interference": kwargs.get("check_interference", True),
                    "generate_semantic": kwargs.get("generate_semantic", False)
                }
            )
            
            # Add to vector store for semantic search
            if self.embedding_service:
                await self.embedding_service.add_memory(
                    text=memory_text,
                    metadata={
                        "memory_id": result.get("memory_id"),
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "importance": importance,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    },
                    entity_type=entity_type
                )
            
            # Record telemetry
            duration = asyncio.get_event_loop().time() - start_time
            await MemoryTelemetry.record(
                self.user_id,
                self.conversation_id,
                operation="store_memory",
                success=True,
                duration=duration,
                data_size=len(memory_text),
                metadata={"entity_type": entity_type, "entity_id": entity_id}
            )
            
            # Update result with emotion analysis
            if emotion_analysis:
                result["emotion_analysis"] = emotion_analysis
            
            return result
            
        except Exception as e:
            # Record failure telemetry
            duration = asyncio.get_event_loop().time() - start_time
            await MemoryTelemetry.record(
                self.user_id,
                self.conversation_id,
                operation="store_memory",
                success=False,
                duration=duration,
                error=str(e)
            )
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def retrieve_memories(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        query: str = None,
        context: Dict[str, Any] = None,
        limit: int = 5,
        include_analysis: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve memories for an entity with comprehensive analysis.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            query: Optional search query
            context: Current context for retrieval
            limit: Maximum memories to return
            include_analysis: Whether to include memory analysis
            **kwargs: Additional retrieval parameters
            
        Returns:
            Retrieved memories with analysis and context
        """
        if not self.initialized:
            await self.initialize()
        
        # Normalize entity type
        if isinstance(entity_type, str):
            entity_type = entity_type.lower()
        else:
            entity_type = entity_type.value
        
        # Check cache
        cache_key = f"retrieve_{entity_type}_{entity_id}_{query}_{limit}"
        cached = await self.cache.get(cache_key)
        if cached and not kwargs.get("bypass_cache", False):
            return cached
        
        # Use integrated system for comprehensive retrieval
        result = await self.integrated_system.retrieve_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            current_context=context or self.active_context,
            retrieval_kwargs={
                "limit": limit,
                "use_emotional_context": kwargs.get("use_emotional_context", True),
                "simulate_competition": kwargs.get("simulate_competition", False),
                "allow_flashbacks": kwargs.get("allow_flashbacks", True),
                "include_schema_interpretation": kwargs.get("include_schemas", True),
                "check_for_intrusions": kwargs.get("check_intrusions", True)
            }
        )
        
        # Add semantic search results if query provided
        if query and self.retriever_agent:
            semantic_results = await self.retriever_agent.retrieve_and_analyze(
                query=query,
                entity_types=[entity_type] if entity_type != "all" else None,
                top_k=limit,
                threshold=kwargs.get("similarity_threshold", 0.7)
            )
            
            if semantic_results.get("found_memories"):
                result["semantic_search"] = semantic_results
        
        # Include analysis if requested
        if include_analysis:
            result["analysis"] = await self.analyze_memory_set(
                memories=result.get("memories", []),
                entity_type=entity_type,
                entity_id=entity_id
            )
        
        # Cache result
        await self.cache.set(cache_key, result)
        
        return result
    
    # ========================================================================
    # Narrative Context Operations
    # ========================================================================
    
    async def get_narrative_context(
        self,
        focus_entities: List[Tuple[str, int]] = None,
        time_window: timedelta = None,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive narrative context for the narrative generator.
        
        Args:
            focus_entities: List of (entity_type, entity_id) tuples to focus on
            time_window: Time window for recent memories
            include_predictions: Whether to include future predictions
            
        Returns:
            Comprehensive narrative context
        """
        if not self.initialized:
            await self.initialize()
        
        context = {
            "timestamp": await get_game_iso_string(self.user_id, self.conversation_id),
            "entities": {},
            "relationships": {},
            "active_schemas": [],
            "emotional_landscape": {},
            "recent_events": [],
            "narrative_threads": [],
            "potential_conflicts": [],
            "predictions": []
        }
        
        # Default to all main entities if none specified
        if not focus_entities:
            focus_entities = await self._get_active_entities()
        
        # Gather context for each entity
        for entity_type, entity_id in focus_entities:
            entity_context = await self._get_entity_context(
                entity_type, entity_id, time_window
            )
            context["entities"][f"{entity_type}_{entity_id}"] = entity_context
            
            # Add to emotional landscape
            if entity_context.get("emotional_state"):
                context["emotional_landscape"][f"{entity_type}_{entity_id}"] = {
                    "current": entity_context["emotional_state"].get("current_emotion"),
                    "mood": entity_context["emotional_state"].get("mood")
                }
        
        # Get relationships between entities
        context["relationships"] = await self._get_entity_relationships(focus_entities)
        
        # Get active schemas across all entities
        for entity_type, entity_id in focus_entities:
            schemas = await self.schema_manager.get_entity_schemas(
                entity_type=entity_type,
                entity_id=entity_id
            )
            context["active_schemas"].extend(schemas)
        
        # Get recent significant events
        context["recent_events"] = await self._get_recent_significant_events(time_window)
        
        # Identify narrative threads
        context["narrative_threads"] = await self._identify_narrative_threads()
        
        # Detect potential conflicts
        context["potential_conflicts"] = await self._detect_potential_conflicts(focus_entities)
        
        # Generate predictions if requested
        if include_predictions:
            context["predictions"] = await self._generate_narrative_predictions(context)
        
        # Update active context
        self.active_context = context
        
        return context
    
    async def get_entity_memory_summary(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        include_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary of an entity's memory state.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            include_stats: Whether to include statistics
            
        Returns:
            Memory summary for the entity
        """
        if not self.initialized:
            await self.initialize()
        
        # Get appropriate manager
        manager = await self._get_entity_manager(entity_type, entity_id)
        
        summary = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "total_memories": 0,
            "memory_types": {},
            "significance_distribution": {},
            "emotional_profile": {},
            "key_memories": [],
            "beliefs": [],
            "schemas": [],
            "recent_activity": []
        }
        
        # Get all memories for analysis
        all_memories = await manager.retrieve_memories(limit=1000)
        summary["total_memories"] = len(all_memories)
        
        # Analyze memory types
        for memory in all_memories:
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
            summary["memory_types"][mem_type] = summary["memory_types"].get(mem_type, 0) + 1
            
            # Significance distribution
            sig = memory.significance
            sig_key = f"level_{sig}"
            summary["significance_distribution"][sig_key] = summary["significance_distribution"].get(sig_key, 0) + 1
        
        # Get emotional profile
        if entity_type in ["npc", "player"]:
            emotional_state = await self.emotional_manager.get_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id
            )
            summary["emotional_profile"] = emotional_state
        
        # Get key memories (high significance)
        key_memories = await manager.retrieve_memories(
            min_significance=MemorySignificance.HIGH,
            limit=10
        )
        summary["key_memories"] = [
            {
                "id": m.id,
                "text": m.text[:100] + "..." if len(m.text) > 100 else m.text,
                "significance": m.significance,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }
            for m in key_memories
        ]
        
        # Get beliefs
        beliefs = await self.semantic_manager.get_beliefs(
            entity_type=entity_type,
            entity_id=entity_id,
            limit=5
        )
        summary["beliefs"] = beliefs
        
        # Get schemas
        schemas = await self.schema_manager.get_entity_schemas(
            entity_type=entity_type,
            entity_id=entity_id
        )
        summary["schemas"] = [
            {"name": s.get("schema_name"), "confidence": s.get("confidence", 0.5)}
            for s in schemas[:5]
        ]
        
        # Include statistics if requested
        if include_stats:
            summary["statistics"] = await self._calculate_memory_statistics(all_memories)
        
        return summary
    
    # ========================================================================
    # Specialized Operations
    # ========================================================================
    
    async def trigger_npc_revelation(
        self,
        npc_id: int,
        trigger: str = None,
        severity: int = None
    ) -> Dict[str, Any]:
        """
        Trigger a mask slippage event for an NPC.
        
        Args:
            npc_id: NPC ID
            trigger: What triggered the revelation
            severity: Severity level (1-5)
            
        Returns:
            Revelation details
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.mask_manager.generate_mask_slippage(
            npc_id=npc_id,
            trigger=trigger,
            severity=severity
        )
    
    async def generate_flashback(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an entity based on context.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            context: Current context
            
        Returns:
            Flashback information or None
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.flashback_manager.generate_flashback(
            entity_type=entity_type,
            entity_id=entity_id,
            current_context=context
        )
    
    async def create_false_memory(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        false_text: str,
        base_memory_ids: List[int] = None
    ) -> Dict[str, Any]:
        """
        Create a false memory for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            false_text: The false memory text
            base_memory_ids: IDs of real memories to base it on
            
        Returns:
            Created false memory information
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.interference_manager.create_false_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            false_memory_text=false_text,
            related_true_memory_ids=base_memory_ids
        )
    
    async def reconsolidate_memory(
        self,
        memory_id: int,
        entity_type: Union[str, EntityType],
        entity_id: int,
        alteration_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Reconsolidate a memory with potential alterations.
        
        Args:
            memory_id: Memory ID
            entity_type: Type of entity
            entity_id: Entity ID
            alteration_strength: How much to alter (0.0-1.0)
            
        Returns:
            Reconsolidation results
        """
        if not self.initialized:
            await self.initialize()
        
        emotional_context = None
        if entity_type in ["npc", "player"]:
            emotional_state = await self.emotional_manager.get_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id
            )
            emotional_context = emotional_state.get("current_emotion")
        
        return await self.reconsolidation_manager.reconsolidate_memory(
            memory_id=memory_id,
            entity_type=entity_type,
            entity_id=entity_id,
            emotional_context=emotional_context,
            alteration_strength=alteration_strength
        )
    
    # ========================================================================
    # Analysis Operations
    # ========================================================================
    
    async def analyze_memory_patterns(
        self,
        entity_type: Union[str, EntityType] = None,
        entity_id: int = None,
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Analyze patterns across memories.
        
        Args:
            entity_type: Optional entity type filter
            entity_id: Optional entity ID filter
            topic: Optional topic filter
            
        Returns:
            Pattern analysis results
        """
        if not self.initialized:
            await self.initialize()
        
        if entity_type and entity_id:
            # Analyze specific entity
            return await self.semantic_manager.find_patterns_across_memories(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic
            )
        else:
            # Analyze across all entities
            patterns = []
            entities = await self._get_active_entities()
            
            for ent_type, ent_id in entities:
                entity_patterns = await self.semantic_manager.find_patterns_across_memories(
                    entity_type=ent_type,
                    entity_id=ent_id,
                    topic=topic
                )
                patterns.append({
                    "entity": f"{ent_type}_{ent_id}",
                    "patterns": entity_patterns
                })
            
            return {"cross_entity_patterns": patterns}
    
    async def generate_lore(
        self,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate lore based on accumulated memories.
        
        Args:
            context: Optional context for lore generation
            
        Returns:
            Generated lore
        """
        if not self.initialized:
            await self.initialize()
        
        lore_manager = LoreMemoryManager(self.user_id, self.conversation_id)
        return await lore_manager.generate_lore_from_memories(
            context=context or self.active_context
        )
    
    # ========================================================================
    # Maintenance Operations
    # ========================================================================
    
    async def run_maintenance(
        self,
        operations: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run maintenance operations on the memory system.
        
        Args:
            operations: Specific operations to run, or None for all
            
        Returns:
            Maintenance results
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        operations = operations or [
            "consolidation",
            "decay",
            "schema_maintenance",
            "reconsolidation",
            "cleanup"
        ]
        
        # Run requested operations
        if "consolidation" in operations:
            results["consolidation"] = await self._run_consolidation()
        
        if "decay" in operations:
            results["decay"] = await self._run_decay()
        
        if "schema_maintenance" in operations:
            results["schemas"] = await self._run_schema_maintenance()
        
        if "reconsolidation" in operations:
            results["reconsolidation"] = await self._run_reconsolidation_maintenance()
        
        if "cleanup" in operations:
            results["cleanup"] = await self.maintenance.cleanup_old_memories()
        
        return results
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _parse_importance(self, importance: str) -> int:
        """Parse importance string to MemorySignificance value."""
        mapping = {
            "trivial": MemorySignificance.TRIVIAL,
            "low": MemorySignificance.LOW,
            "medium": MemorySignificance.MEDIUM,
            "high": MemorySignificance.HIGH,
            "critical": MemorySignificance.CRITICAL
        }
        result = mapping.get(importance.lower(), MemorySignificance.MEDIUM)
        return result.value if hasattr(result, 'value') else result
    
    async def _get_entity_manager(
        self,
        entity_type: str,
        entity_id: int
    ) -> UnifiedMemoryManager:
        """Get the appropriate memory manager for an entity."""
        key = f"{entity_type}_{entity_id}"
        
        if key not in self.managers:
            if entity_type == "npc":
                self.managers[key] = NPCMemoryManager(
                    entity_id, self.user_id, self.conversation_id
                )
            elif entity_type == "player":
                # Assuming player name can be retrieved
                self.managers[key] = PlayerMemoryManager(
                    "Chase", self.user_id, self.conversation_id  
                )
            elif entity_type == "nyx":
                self.managers[key] = NyxMemoryManager(
                    self.user_id, self.conversation_id
                )
            else:
                self.managers[key] = UnifiedMemoryManager(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
        
        return self.managers[key]
    
    async def _get_active_entities(self) -> List[Tuple[str, int]]:
        """Get list of active entities in the current conversation."""
        entities = []
        
        async with get_connection_context() as conn:
            # Get active NPCs
            npc_rows = await conn.fetch("""
                SELECT npc_id FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
            """, self.user_id, self.conversation_id)
            
            for row in npc_rows:
                entities.append(("npc", row["npc_id"]))
            
            # Get player
            player_row = await conn.fetchrow("""
                SELECT DISTINCT player_name FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2
                LIMIT 1
            """, self.user_id, self.conversation_id)
            
            if player_row:
                entities.append(("player", self.user_id))
            
            # Always include Nyx
            entities.append(("nyx", 0))
        
        return entities
    
    async def _get_entity_context(
        self,
        entity_type: str,
        entity_id: int,
        time_window: timedelta = None
    ) -> Dict[str, Any]:
        """Get comprehensive context for a single entity."""
        manager = await self._get_entity_manager(entity_type, entity_id)
        
        context = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "recent_memories": [],
            "emotional_state": None,
            "active_schemas": [],
            "beliefs": [],
            "key_relationships": []
        }
        
        # Get recent memories
        context["recent_memories"] = await manager.retrieve_memories(
            limit=10,
            conn=None
        )
        
        # Get emotional state if applicable
        if entity_type in ["npc", "player"]:
            context["emotional_state"] = await self.emotional_manager.get_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id
            )
        
        # Get schemas
        schemas = await self.schema_manager.get_entity_schemas(
            entity_type=entity_type,
            entity_id=entity_id
        )
        context["active_schemas"] = schemas[:5]  # Top 5 schemas
        
        # Get beliefs
        context["beliefs"] = await self.semantic_manager.get_beliefs(
            entity_type=entity_type,
            entity_id=entity_id,
            limit=5
        )
        
        return context
    
    async def _get_entity_relationships(
        self,
        entities: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """Get relationships between entities."""
        relationships = {}
        
        async with get_connection_context() as conn:
            for i, (type1, id1) in enumerate(entities):
                for type2, id2 in entities[i+1:]:
                    # Check for social link
                    row = await conn.fetchrow("""
                        SELECT link_type, link_level
                        FROM SocialLinks
                        WHERE user_id = $1 AND conversation_id = $2
                        AND ((entity1_type = $3 AND entity1_id = $4 
                              AND entity2_type = $5 AND entity2_id = $6)
                          OR (entity1_type = $5 AND entity1_id = $6 
                              AND entity2_type = $3 AND entity2_id = $4))
                    """, self.user_id, self.conversation_id,
                        type1, id1, type2, id2)
                    
                    if row:
                        key = f"{type1}_{id1}_to_{type2}_{id2}"
                        relationships[key] = {
                            "type": row["link_type"],
                            "strength": row["link_level"]
                        }
        
        return relationships
    
    async def _get_recent_significant_events(
        self,
        time_window: timedelta = None
    ) -> List[Dict[str, Any]]:
        """Get recent significant events across all entities."""
        if not time_window:
            time_window = timedelta(hours=24)
        
        cutoff = await get_game_datetime(self.user_id, self.conversation_id) - time_window
        events = []
        
        async with get_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT id, entity_type, entity_id, memory_text, 
                       significance, emotional_intensity, timestamp
                FROM unified_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > $3
                AND significance >= $4
                ORDER BY timestamp DESC
                LIMIT 20
            """, self.user_id, self.conversation_id, cutoff, MemorySignificance.HIGH)
            
            for row in rows:
                events.append({
                    "id": row["id"],
                    "entity": f"{row['entity_type']}_{row['entity_id']}",
                    "text": row["memory_text"],
                    "significance": row["significance"],
                    "timestamp": row["timestamp"].isoformat()
                })
        
        return events
    
    async def _identify_narrative_threads(self) -> List[Dict[str, Any]]:
        """Identify active narrative threads from memories."""
        # This would use pattern recognition across memories
        # Simplified version for now
        return []
    
    async def _detect_potential_conflicts(
        self,
        entities: List[Tuple[str, int]]
    ) -> List[Dict[str, Any]]:
        """Detect potential conflicts between entities."""
        conflicts = []
        
        # Check for opposing beliefs or goals
        for i, (type1, id1) in enumerate(entities):
            beliefs1 = await self.semantic_manager.get_beliefs(
                entity_type=type1,
                entity_id=id1,
                limit=3
            )
            
            for type2, id2 in entities[i+1:]:
                beliefs2 = await self.semantic_manager.get_beliefs(
                    entity_type=type2,
                    entity_id=id2,
                    limit=3
                )
                
                # Simple conflict detection - would be more sophisticated
                for b1 in beliefs1:
                    for b2 in beliefs2:
                        if self._beliefs_conflict(b1, b2):
                            conflicts.append({
                                "entity1": f"{type1}_{id1}",
                                "entity2": f"{type2}_{id2}",
                                "belief1": b1["belief"],
                                "belief2": b2["belief"],
                                "conflict_type": "belief"
                            })
        
        return conflicts
    
    def _beliefs_conflict(self, belief1: Dict, belief2: Dict) -> bool:
        """Check if two beliefs conflict (simplified)."""
        # This would use NLP to detect opposing beliefs
        # Simplified check for now
        text1 = belief1.get("belief", "").lower()
        text2 = belief2.get("belief", "").lower()
        
        # Check for obvious opposites
        opposites = [
            ("should", "should not"),
            ("must", "must not"),
            ("good", "bad"),
            ("right", "wrong")
        ]
        
        for pos, neg in opposites:
            if pos in text1 and neg in text2:
                return True
            if neg in text1 and pos in text2:
                return True
        
        return False
    
    async def _generate_narrative_predictions(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate predictions about future narrative developments."""
        # This would use the context to predict likely future events
        # Simplified version for now
        predictions = []
        
        # Check for high emotional intensity
        for entity, data in context["entities"].items():
            if data.get("emotional_state"):
                emotion = data["emotional_state"].get("current_emotion", {})
                if emotion.get("intensity", 0) > 0.7:
                    predictions.append({
                        "entity": entity,
                        "prediction": f"High emotional intensity may lead to impulsive action",
                        "confidence": 0.7
                    })
        
        return predictions
    
    async def analyze_memory_set(
        self,
        memories: List[Memory],
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """Analyze a set of memories for patterns and insights."""
        if not memories:
            return {"message": "No memories to analyze"}
        
        analysis = {
            "total_count": len(memories),
            "average_significance": sum(m.significance for m in memories) / len(memories),
            "average_emotional_intensity": sum(m.emotional_intensity for m in memories) / len(memories),
            "common_tags": {},
            "temporal_distribution": {},
            "memory_types": {}
        }
        
        # Analyze tags
        for memory in memories:
            for tag in (memory.tags or []):
                analysis["common_tags"][tag] = analysis["common_tags"].get(tag, 0) + 1
        
        # Analyze memory types
        for memory in memories:
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
            analysis["memory_types"][mem_type] = analysis["memory_types"].get(mem_type, 0) + 1
        
        return analysis
    
    async def _calculate_memory_statistics(
        self,
        memories: List[Memory]
    ) -> Dict[str, Any]:
        """Calculate detailed statistics for a set of memories."""
        if not memories:
            return {}
        
        stats = {
            "total": len(memories),
            "recall_statistics": {
                "total_recalls": sum(m.times_recalled for m in memories),
                "average_recalls": sum(m.times_recalled for m in memories) / len(memories),
                "most_recalled": max(memories, key=lambda m: m.times_recalled).id if memories else None
            },
            "age_statistics": {},
            "consolidation_rate": sum(1 for m in memories if m.is_consolidated) / len(memories)
        }
        
        # Calculate age statistics
        now = datetime.now()
        ages = []
        for memory in memories:
            if memory.timestamp:
                age_days = (now - memory.timestamp).days
                ages.append(age_days)
        
        if ages:
            stats["age_statistics"] = {
                "average_age_days": sum(ages) / len(ages),
                "oldest_days": max(ages),
                "newest_days": min(ages)
            }
        
        return stats
    
    async def _run_consolidation(self) -> Dict[str, Any]:
        """Run memory consolidation for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            manager = await self._get_entity_manager(entity_type, entity_id)
            consolidated = await manager.consolidate_memories()
            results[f"{entity_type}_{entity_id}"] = len(consolidated)
        
        return results
    
    async def _run_decay(self) -> Dict[str, Any]:
        """Run memory decay for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            manager = await self._get_entity_manager(entity_type, entity_id)
            decayed = await manager.apply_memory_decay()
            results[f"{entity_type}_{entity_id}"] = decayed
        
        return results
    
    async def _run_schema_maintenance(self) -> Dict[str, Any]:
        """Run schema maintenance for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            maintenance_result = await self.schema_manager.run_schema_maintenance(
                entity_type=entity_type,
                entity_id=entity_id
            )
            results[f"{entity_type}_{entity_id}"] = maintenance_result
        
        return results
    
    async def _run_reconsolidation_maintenance(self) -> Dict[str, Any]:
        """Run reconsolidation checks for all entities."""
        results = {}
        entities = await self._get_active_entities()
        
        for entity_type, entity_id in entities:
            reconsolidated = await self.reconsolidation_manager.check_memories_for_reconsolidation(
                entity_type=entity_type,
                entity_id=entity_id,
                max_memories=3
            )
            results[f"{entity_type}_{entity_id}"] = len(reconsolidated)
        
        return results
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    async def close(self):
        """Clean up resources."""
        if self.embedding_service:
            await self.embedding_service.close()
        
        if self.retriever_agent:
            await self.retriever_agent.close()
        
        # Clear caches
        await self.cache.clear()
        
        self.initialized = False


# ============================================================================
# Convenience Functions
# ============================================================================

async def get_memory_orchestrator(user_id: int, conversation_id: int) -> MemoryOrchestrator:
    """
    Get or create a Memory Orchestrator instance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        MemoryOrchestrator instance
    """
    return await MemoryOrchestrator.get_instance(user_id, conversation_id)
