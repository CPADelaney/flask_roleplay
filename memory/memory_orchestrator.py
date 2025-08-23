# memory/memory_orchestrator.py

"""
Memory System Orchestrator

Single access point for ALL memory operations in the system.
Provides a unified interface for the narrative generator and other components
to interact with the complex memory subsystems.

This orchestrator routes all memory operations to the appropriate subsystem,
ensuring consistent access patterns and centralized management.
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

# Memory services and agents
from memory.memory_service import MemoryEmbeddingService
from memory.memory_retriever import MemoryRetrieverAgent
from memory.memory_agent_sdk import create_memory_agent, MemorySystemContext
from memory.memory_agent_wrapper import MemoryAgentWrapper

# Integration layers
from memory.integrated import IntegratedMemorySystem, init_memory_system
from memory.wrapper import MemorySystem
from memory.memory_nyx_integration import MemoryNyxBridge, get_memory_nyx_bridge
from memory.memory_integration import (
    MemoryIntegration,
    get_memory_service,
    get_memory_retriever,
    enrich_context_with_memories
)

# Initialization and setup
from memory.init import (
    initialize_database,
    setup_npc,
    setup_player,
    setup_nyx
)

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
    MEMORY = "memory"  # Generic memory type
    NARRATIVE = "narrative"


class MemoryOrchestrator:
    """
    Central orchestrator for ALL memory operations.
    
    This class serves as the single access point for:
    - Core memory operations (store, retrieve, update, delete)
    - Specialized memory managers (NPC, Player, Nyx, etc.)
    - Advanced features (schemas, emotional, masks, flashbacks, etc.)
    - Memory agents and LLM-based operations
    - Vector store and embedding operations
    - Integration with Nyx governance
    - Telemetry and maintenance
    - Setup and initialization
    
    All memory operations in the system should go through this orchestrator.
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
        
        # Services and agents
        self.embedding_service = None
        self.retriever_agent = None
        self.memory_agent = None
        self.memory_agent_wrapper = None
        
        # Integration layers
        self.nyx_bridge = None
        self.memory_integration = None
        
        # Maintenance and telemetry
        self.maintenance = None
        self.telemetry = MemoryTelemetry
        
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
            
            # Initialize database if needed
            await initialize_database()
            
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
            self.embedding_service = await get_memory_service(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                vector_store_type=self.memory_config.get("vector_store", {}).get("type", "chroma"),
                embedding_model=self.memory_config.get("embedding", {}).get("type", "local"),
                config=self.memory_config
            )
            
            self.retriever_agent = await get_memory_retriever(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                llm_type=self.memory_config.get("llm", {}).get("type", "openai"),
                vector_store_type=self.memory_config.get("vector_store", {}).get("type", "chroma"),
                embedding_model=self.memory_config.get("embedding", {}).get("type", "local"),
                config=self.memory_config
            )
            
            # Initialize memory agent
            memory_context = MemorySystemContext(self.user_id, self.conversation_id)
            base_agent = create_memory_agent(self.user_id, self.conversation_id)
            self.memory_agent = base_agent
            self.memory_agent_wrapper = MemoryAgentWrapper(base_agent, memory_context)

            if not hasattr(self, '_canon_synced'):
                self._canon_synced = False
            
            # Run canon sync on first initialization
            if not self._canon_synced:
                try:
                    sync_result = await self.sync_canon_to_memory()
                    logger.info(f"Initial canon sync completed: {sync_result}")
                    self._canon_synced = True
                except Exception as e:
                    logger.warning(f"Canon sync failed during initialization: {e}")
            
            # Initialize integration layers
            self.nyx_bridge = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
            self.memory_integration = MemoryIntegration(self.user_id, self.conversation_id)
            await self.memory_integration.initialize()
            
            # Initialize maintenance
            self.maintenance = MemoryMaintenance()
            
            self.initialized = True
            logger.info("Memory Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Orchestrator: {e}")
            raise
    
    # ========================================================================
    # Setup and Initialization Operations
    # ========================================================================

    async def store_canonical_entity(
        self,
        entity_type: str,
        entity_id: int,
        entity_name: str,
        entity_data: Dict[str, Any],
        significance: int = 5
    ) -> Dict[str, Any]:
        """
        Store a canonical entity creation as a memory.
        This is called by canon.py when creating new entities.
        """
        if not self.initialized:
            await self.initialize()
        
        # Create descriptive text for the entity
        description_parts = [f"Created {entity_type}: {entity_name}"]
        for key, value in entity_data.items():
            if key not in ['embedding', 'user_id', 'conversation_id']:
                description_parts.append(f"{key}: {value}")
        
        memory_text = ". ".join(description_parts[:5])  # Limit to avoid too long text
        
        # Map significance to importance
        importance_map = {
            1: "trivial", 2: "trivial", 3: "low",
            4: "low", 5: "medium", 6: "medium",
            7: "high", 8: "high", 9: "critical", 10: "critical"
        }
        importance = importance_map.get(significance, "medium")
        
        # Store as a memory
        result = await self.store_memory(
            entity_type=EntityType.LORE,
            entity_id=0,  # Use 0 for general lore
            memory_text=memory_text,
            importance=importance,
            tags=[entity_type.lower(), "canon", "creation"],
            metadata={
                "canonical_entity_type": entity_type,
                "canonical_entity_id": entity_id,
                "canonical_entity_name": entity_name,
                **entity_data
            }
        )
        
        # Also add to vector store for searchability
        await self.add_to_vector_store(
            text=f"{entity_type}: {entity_name} - {memory_text}",
            metadata={
                "entity_type": entity_type.lower(),
                "entity_id": entity_id,
                "entity_name": entity_name,
                "canonical": True,
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            },
            entity_type=entity_type.lower()
        )
        
        return result
    
    async def search_canonical_entities(
        self,
        query: str,
        entity_types: List[str] = None,
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Search for canonical entities using the vector store.
        Used by canon.py for duplicate detection.
        """
        if not self.initialized:
            await self.initialize()
        
        filter_dict = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "canonical": True
        }
        
        results = []
        
        if entity_types:
            # Search each entity type separately for better accuracy
            for entity_type in entity_types:
                type_results = await self.search_vector_store(
                    query=query,
                    entity_type=entity_type.lower(),
                    top_k=3,
                    filter_dict=filter_dict
                )
                
                # Filter by similarity threshold
                for result in type_results:
                    if result.get("similarity", 0) >= similarity_threshold:
                        result["entity_type"] = entity_type
                        results.append(result)
        else:
            # Search all canonical entities
            all_results = await self.search_vector_store(
                query=query,
                top_k=5,
                filter_dict=filter_dict
            )
            
            # Filter by similarity threshold
            results = [r for r in all_results if r.get("similarity", 0) >= similarity_threshold]
        
        # Sort by similarity
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return results
    
    async def get_canonical_context(
        self,
        entity_types: List[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get canonical context for narrative generation.
        Combines canonical entities with memories.
        """
        if not self.initialized:
            await self.initialize()
        
        context = {
            "canonical_entities": {},
            "recent_canonical_events": [],
            "active_locations": [],
            "active_npcs": [],
            "world_state": {},
            "narrative_context": await self.get_narrative_context()
        }
        
        from logic.game_time_helper import get_game_datetime
        from db.connection import get_db_connection_context
        
        current_time = await get_game_datetime(self.user_id, self.conversation_id)
        cutoff_time = current_time - timedelta(hours=time_window_hours)
        
        async with get_db_connection_context() as conn:
            # Get recent canonical events
            events = await conn.fetch("""
                SELECT event_text, tags, significance, timestamp
                FROM CanonicalEvents
                WHERE user_id = $1 AND conversation_id = $2
                AND timestamp > $3
                ORDER BY timestamp DESC
                LIMIT 20
            """, self.user_id, self.conversation_id, cutoff_time)
            
            context["recent_canonical_events"] = [
                {
                    "text": e["event_text"],
                    "tags": json.loads(e["tags"]) if e["tags"] else [],
                    "significance": e["significance"],
                    "timestamp": e["timestamp"].isoformat()
                }
                for e in events
            ]
            
            # Get active NPCs
            npcs = await conn.fetch("""
                SELECT npc_id, npc_name, role, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND introduced = TRUE
            """, self.user_id, self.conversation_id)
            
            context["active_npcs"] = [
                {
                    "id": npc["npc_id"],
                    "name": npc["npc_name"],
                    "role": npc["role"],
                    "location": npc["current_location"]
                }
                for npc in npcs
            ]
            
            # Get active locations
            locations = await conn.fetch("""
                SELECT location_name, location_type, description
                FROM Locations
                WHERE user_id = $1 AND conversation_id = $2
                LIMIT 10
            """, self.user_id, self.conversation_id)
            
            context["active_locations"] = [
                {
                    "name": loc["location_name"],
                    "type": loc["location_type"],
                    "description": loc["description"]
                }
                for loc in locations
            ]
            
            # Get current roleplay state
            roleplay_state = await conn.fetch("""
                SELECT key, value
                FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            for state in roleplay_state:
                context["world_state"][state["key"]] = state["value"]
        
        # Enhance with memory-based insights
        if context["active_npcs"]:
            for npc in context["active_npcs"]:
                # Get NPC's recent memories
                npc_memories = await self.retrieve_memories(
                    entity_type="npc",
                    entity_id=npc["id"],
                    limit=3
                )
                npc["recent_memories"] = npc_memories.get("memories", [])
                
                # Get NPC's emotional state
                emotional_state = await self.get_emotional_state(
                    entity_type="npc",
                    entity_id=npc["id"]
                )
                npc["emotional_state"] = emotional_state
        
        return context

    async def ensure_canon_synced(self) -> bool:
        """
        Ensure canon is synced to memory system, with proper error handling.
        This is called by canon.py when needed.
        
        Returns:
            True if sync successful or already synced
        """
        if hasattr(self, '_canon_synced') and self._canon_synced:
            return True
        
        try:
            from db.connection import get_db_connection_context
            
            # Check if canon tables exist
            async with get_db_connection_context() as conn:
                tables_exist = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name IN ('npcstats', 'locations', 'events', 'playerjournal')
                    AND table_schema = 'public'
                """)
                
                if tables_exist == 0:
                    logger.info("No canon tables found, skipping sync")
                    self._canon_synced = True
                    return True
                
                # Check if we have any data to sync
                has_data = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM NPCStats 
                        WHERE user_id = $1 AND conversation_id = $2
                        LIMIT 1
                    )
                """, self.user_id, self.conversation_id)
                
                if has_data:
                    logger.info("Starting canon to memory sync...")
                    sync_result = await self.sync_canon_to_memory_safe()
                    logger.info(f"Canon sync completed: {sync_result}")
                else:
                    logger.info("No canon data to sync")
                
                self._canon_synced = True
                return True
                
        except Exception as e:
            logger.warning(f"Canon sync check failed (non-critical): {e}")
            # Don't fail initialization over this
            self._canon_synced = False
            return False
    
    async def validate_canonical_consistency(
        self,
        entity_type: str,
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that a new entity is consistent with existing canon.
        Uses memory analysis to detect conflicts.
        """
        if not self.initialized:
            await self.initialize()
        
        conflicts = []
        warnings = []
        
        # Search for similar entities
        search_text = f"{entity_type}: {entity_data.get('name', '')} {entity_data.get('description', '')}"
        similar_entities = await self.search_canonical_entities(
            query=search_text,
            entity_types=[entity_type],
            similarity_threshold=0.7
        )
        
        if similar_entities:
            # Use LLM to check for conflicts
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            prompt = f"""
            Check if this new {entity_type} conflicts with existing canon:
            
            New Entity:
            {json.dumps(entity_data, indent=2)}
            
            Similar Existing Entities:
            {json.dumps(similar_entities[:3], indent=2)}
            
            Identify:
            1. Direct conflicts (contradictions)
            2. Potential issues (inconsistencies)
            3. Warnings (things to be careful about)
            
            Return JSON:
            {{
                "has_conflicts": true/false,
                "conflicts": ["conflict1", ...],
                "warnings": ["warning1", ...],
                "suggestions": ["suggestion1", ...]
            }}
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a canon consistency validator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if result.get("has_conflicts"):
                    conflicts.extend(result.get("conflicts", []))
                warnings.extend(result.get("warnings", []))
                
            except Exception as e:
                logger.error(f"Error validating canonical consistency: {e}")
        
        # Check narrative consistency
        narrative_context = await self.get_narrative_context()
        if narrative_context.get("potential_conflicts"):
            for conflict in narrative_context["potential_conflicts"]:
                warnings.append(f"Potential narrative conflict: {conflict.get('conflict_type')}")
        
        return {
            "is_consistent": len(conflicts) == 0,
            "conflicts": conflicts,
            "warnings": warnings,
            "similar_entities": similar_entities
        }
    
    async def sync_canon_to_memory(self) -> Dict[str, Any]:
        """
        Synchronize all canonical data to the memory system.
        This ensures the memory system has full knowledge of the world state.
        """
        if not self.initialized:
            await self.initialize()
        
        from db.connection import get_db_connection_context
        from lore.core.canon import ensure_canonical_context
        
        ctx = ensure_canonical_context({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        sync_stats = {
            "npcs_synced": 0,
            "locations_synced": 0,
            "events_synced": 0,
            "journal_entries_synced": 0,
            "errors": []
        }
        
        try:
            async with get_db_connection_context() as conn:
                # Sync NPCs
                npcs = await conn.fetch("""
                    SELECT npc_id, npc_name, role, affiliations, introduced
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                for npc in npcs:
                    try:
                        await self.store_canonical_entity(
                            entity_type="npc",
                            entity_id=npc["npc_id"],
                            entity_name=npc["npc_name"],
                            entity_data={
                                "role": npc["role"],
                                "affiliations": npc["affiliations"],
                                "introduced": npc["introduced"]
                            }
                        )
                        sync_stats["npcs_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"NPC {npc['npc_id']}: {str(e)}")
                
                # Sync Locations
                locations = await conn.fetch("""
                    SELECT id, location_name, location_type, description
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                for loc in locations:
                    try:
                        await self.store_canonical_entity(
                            entity_type="location",
                            entity_id=loc["id"],
                            entity_name=loc["location_name"],
                            entity_data={
                                "type": loc["location_type"],
                                "description": loc["description"]
                            }
                        )
                        sync_stats["locations_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"Location {loc['id']}: {str(e)}")
                
                # Sync Journal Entries
                journal_entries = await conn.fetch("""
                    SELECT id, entry_type, entry_text, importance, tags
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 100
                """, self.user_id, self.conversation_id)
                
                for entry in journal_entries:
                    try:
                        importance = "high" if entry["importance"] > 0.7 else "medium" if entry["importance"] > 0.3 else "low"
                        tags = json.loads(entry["tags"]) if entry["tags"] else []
                        
                        await self.store_memory(
                            entity_type=EntityType.PLAYER,
                            entity_id=self.user_id,
                            memory_text=entry["entry_text"],
                            importance=importance,
                            tags=tags + ["journal_sync"],
                            metadata={
                                "journal_id": entry["id"],
                                "entry_type": entry["entry_type"]
                            }
                        )
                        sync_stats["journal_entries_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"Journal {entry['id']}: {str(e)}")
            
            logger.info(f"Canon sync completed: {sync_stats}")
            return sync_stats
            
        except Exception as e:
            logger.error(f"Error during canon sync: {e}")
            sync_stats["errors"].append(f"General sync error: {str(e)}")
            return sync_stats

    async def sync_canon_to_memory_safe(self) -> Dict[str, Any]:
        """
        Safer version of canon sync that avoids circular imports.
        
        Returns:
            Sync statistics
        """
        from db.connection import get_db_connection_context
        
        sync_stats = {
            "npcs_synced": 0,
            "locations_synced": 0,
            "events_synced": 0,
            "journal_entries_synced": 0,
            "errors": []
        }
        
        try:
            async with get_db_connection_context() as conn:
                # Sync NPCs
                npcs = await conn.fetch("""
                    SELECT npc_id, npc_name, role, affiliations, introduced
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 100
                """, self.user_id, self.conversation_id)
                
                for npc in npcs:
                    try:
                        # Check if already synced
                        existing = await self.search_vector_store(
                            query=f"npc:{npc['npc_id']}",
                            entity_type="npc",
                            top_k=1,
                            filter_dict={
                                "entity_id": npc["npc_id"],
                                "canonical": True
                            }
                        )
                        
                        if not existing:
                            await self.store_canonical_entity(
                                entity_type="npc",
                                entity_id=npc["npc_id"],
                                entity_name=npc["npc_name"],
                                entity_data={
                                    "role": npc["role"],
                                    "affiliations": npc["affiliations"],
                                    "introduced": npc["introduced"]
                                },
                                significance=5
                            )
                            sync_stats["npcs_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"NPC {npc['npc_id']}: {str(e)[:100]}")
                
                # Sync Locations
                locations = await conn.fetch("""
                    SELECT id, location_name, location_type, description
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 100
                """, self.user_id, self.conversation_id)
                
                for loc in locations:
                    try:
                        existing = await self.search_vector_store(
                            query=f"location:{loc['id']}",
                            entity_type="location",
                            top_k=1,
                            filter_dict={
                                "entity_id": loc["id"],
                                "canonical": True
                            }
                        )
                        
                        if not existing:
                            await self.store_canonical_entity(
                                entity_type="location",
                                entity_id=loc["id"],
                                entity_name=loc["location_name"],
                                entity_data={
                                    "type": loc["location_type"],
                                    "description": loc["description"]
                                },
                                significance=5
                            )
                            sync_stats["locations_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"Location {loc['id']}: {str(e)[:100]}")
                
                # Sync recent journal entries
                journal_entries = await conn.fetch("""
                    SELECT id, entry_type, entry_text, importance, tags
                    FROM PlayerJournal
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY created_at DESC
                    LIMIT 50
                """, self.user_id, self.conversation_id)
                
                for entry in journal_entries:
                    try:
                        # Check if already synced
                        existing = await self.search_vector_store(
                            query=f"journal:{entry['id']}",
                            entity_type="player",
                            top_k=1,
                            filter_dict={
                                "journal_id": entry["id"]
                            }
                        )
                        
                        if not existing:
                            importance = "high" if entry["importance"] > 0.7 else "medium" if entry["importance"] > 0.3 else "low"
                            tags = json.loads(entry["tags"]) if entry["tags"] else []
                            
                            await self.store_memory(
                                entity_type=EntityType.PLAYER,
                                entity_id=self.user_id,
                                memory_text=entry["entry_text"],
                                importance=importance,
                                tags=tags + ["journal_sync"],
                                metadata={
                                    "journal_id": entry["id"],
                                    "entry_type": entry["entry_type"]
                                },
                                add_to_vector_store=True
                            )
                            sync_stats["journal_entries_synced"] += 1
                    except Exception as e:
                        sync_stats["errors"].append(f"Journal {entry['id']}: {str(e)[:100]}")
            
            # Trim errors if too many
            if len(sync_stats["errors"]) > 10:
                error_count = len(sync_stats["errors"])
                sync_stats["errors"] = sync_stats["errors"][:10]
                sync_stats["errors"].append(f"... and {error_count - 10} more errors")
            
            return sync_stats
            
        except Exception as e:
            logger.error(f"Error during canon sync: {e}")
            sync_stats["errors"].append(f"General sync error: {str(e)}")
            return sync_stats
    
    async def setup_entity(
        self,
        entity_type: Union[str, EntityType],
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set up a new entity with initial data.
        
        Args:
            entity_type: Type of entity to set up
            entity_data: Initial data for the entity
            
        Returns:
            Setup results
        """
        if not self.initialized:
            await self.initialize()
        
        # Normalize entity type
        if isinstance(entity_type, EntityType):
            entity_type = entity_type.value
        
        # Route to appropriate setup function
        if entity_type == "npc":
            npc_id = await setup_npc(
                self.user_id,
                self.conversation_id,
                entity_data
            )
            return {"entity_type": "npc", "entity_id": npc_id, "success": True}
            
        elif entity_type == "player":
            success = await setup_player(
                self.user_id,
                self.conversation_id,
                entity_data
            )
            return {"entity_type": "player", "success": success}
            
        elif entity_type == "nyx":
            success = await setup_nyx(
                self.user_id,
                self.conversation_id,
                entity_data
            )
            return {"entity_type": "nyx", "success": success}
            
        else:
            return {"error": f"Unsupported entity type for setup: {entity_type}"}
    
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
        use_governance: bool = False,
        check_canon_consistency: bool = True,
        enforce_canon: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Store a new memory with optional canon consistency checking.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            memory_text: The memory content
            importance: Importance level
            emotional: Whether to analyze emotional content
            tags: Optional tags
            metadata: Optional metadata
            use_governance: Whether to use Nyx governance
            check_canon_consistency: Whether to check canon consistency
            enforce_canon: If True, reject memories that violate canon
            **kwargs: Additional parameters
            
        Returns:
            Created memory information
        """
        if not self.initialized:
            await self.initialize()
        
        # Normalize entity type
        if isinstance(entity_type, str):
            entity_type = entity_type.lower()
        else:
            entity_type = entity_type.value
        
        # Check canon consistency if requested
        if check_canon_consistency and entity_type in ['npc', 'player', 'location', 'event']:
            consistency = await self.validate_canonical_consistency(
                entity_type=entity_type,
                entity_data={
                    "memory": memory_text,
                    "entity_id": entity_id,
                    "tags": tags
                }
            )
            
            if not consistency["is_consistent"]:
                logger.warning(
                    f"Memory may conflict with canon for {entity_type} {entity_id}: "
                    f"{consistency['conflicts']}"
                )
                
                if enforce_canon:
                    raise ValueError(
                        f"Memory violates established canon: {consistency['conflicts']}"
                    )
                
                # Add warning to metadata
                if metadata is None:
                    metadata = {}
                metadata["canon_warnings"] = consistency["conflicts"]
        
        # Track telemetry
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use governance if requested
            if use_governance and self.nyx_bridge:
                result = await self.nyx_bridge.remember(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_text=memory_text,
                    importance=importance,
                    emotional=emotional,
                    tags=tags
                )
            else:
                # Use wrapper for simplified interface
                result = await self.memory_wrapper.remember(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_text=memory_text,
                    importance=importance,
                    emotional=emotional,
                    tags=tags
                )
            
            # Add to vector store
            if self.embedding_service and kwargs.get("add_to_vector_store", True):
                await self.embedding_service.add_memory(
                    text=memory_text,
                    metadata={
                        "memory_id": result.get("memory_id"),
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "importance": importance,
                        "timestamp": datetime.now().isoformat(),
                        "canonical": False,  # Regular memories are not canonical
                        **(metadata or {})
                    },
                    entity_type=entity_type
                )
            
            # Record telemetry
            duration = asyncio.get_event_loop().time() - start_time
            await self.telemetry.record(
                self.user_id,
                self.conversation_id,
                operation="store_memory",
                success=True,
                duration=duration,
                data_size=len(memory_text),
                metadata={
                    "entity_type": entity_type, 
                    "entity_id": entity_id,
                    "canon_checked": check_canon_consistency
                }
            )
            
            return result
            
        except Exception as e:
            # Record failure telemetry
            duration = asyncio.get_event_loop().time() - start_time
            await self.telemetry.record(
                self.user_id,
                self.conversation_id,
                operation="store_memory",
                success=False,
                duration=duration,
                error=str(e)
            )
            logger.error(f"Error storing memory: {e}")
            raise

    async def get_canon_aware_narrative_context(
        self,
        include_canon: bool = True,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get narrative context that includes both memories and canonical facts.
        
        Args:
            include_canon: Include canonical world state
            time_window_hours: Time window for recent events
            
        Returns:
            Combined narrative and canonical context
        """
        # Get memory-based narrative context
        narrative_context = await self.get_narrative_context(
            time_window=timedelta(hours=time_window_hours),
            include_predictions=True
        )
        
        if not include_canon:
            return narrative_context
        
        # Enhance with canonical context
        canonical_context = await self.get_canonical_context(
            time_window_hours=time_window_hours
        )
        
        # Merge contexts intelligently
        merged_context = {
            **narrative_context,
            "canonical": {
                "entities": canonical_context.get("canonical_entities", {}),
                "events": canonical_context.get("recent_canonical_events", []),
                "locations": canonical_context.get("active_locations", []),
                "npcs": canonical_context.get("active_npcs", []),
                "world_state": canonical_context.get("world_state", {})
            }
        }
        
        # Cross-reference memories with canon
        for entity_key, entity_data in narrative_context.get("entities", {}).items():
            # Find corresponding canonical data
            canon_match = None
            for npc in canonical_context.get("active_npcs", []):
                if f"npc_{npc['id']}" == entity_key:
                    canon_match = npc
                    break
            
            if canon_match:
                entity_data["canonical_info"] = canon_match
                
                # Check for discrepancies
                if canon_match.get("location") != entity_data.get("last_known_location"):
                    entity_data["location_discrepancy"] = {
                        "canonical": canon_match.get("location"),
                        "memory": entity_data.get("last_known_location")
                    }
        
        return merged_context

    async def validate_memory_canon_consistency(
        self,
        memory_text: str,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Validate that a memory is consistent with established canon.
        
        Args:
            memory_text: Memory to validate
            entity_type: Type of entity
            entity_id: Entity ID
            
        Returns:
            Validation results
        """
        from logic.chatgpt_integration import get_openai_client
        
        # Get canonical facts about the entity
        canonical_facts = await self.search_canonical_entities(
            query=f"{entity_type} {entity_id}",
            entity_types=[entity_type],
            similarity_threshold=0.5
        )
        
        if not canonical_facts:
            return {
                "is_consistent": True,
                "message": "No canonical facts to check against"
            }
        
        # Use LLM to check consistency
        try:
            client = get_openai_client()
            
            facts_summary = "\n".join([
                f"- {fact.get('text', '')}"
                for fact in canonical_facts[:5]
            ])
            
            prompt = f"""Check if this memory is consistent with established canon:
    
    Memory: {memory_text}
    
    Established canonical facts:
    {facts_summary}
    
    Identify any contradictions or inconsistencies.
    
    Return JSON:
    {{
        "is_consistent": true/false,
        "conflicts": ["conflict1", ...],
        "severity": "minor|moderate|severe"
    }}
    """
            
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You validate narrative consistency."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            result = json.loads(response.choices[0].message.content)
            result["canonical_facts_checked"] = len(canonical_facts)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating memory consistency: {e}")
            return {
                "is_consistent": True,
                "message": "Could not validate",
                "error": str(e)
            }
        
    async def retrieve_memories(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        query: str = None,
        context: Dict[str, Any] = None,
        limit: int = 5,
        include_analysis: bool = True,
        use_governance: bool = False,
        use_llm_analysis: bool = False,
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
            use_governance: Whether to use Nyx governance
            use_llm_analysis: Whether to use LLM for analysis
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
        
        try:
            # Use governance if requested
            if use_governance and self.nyx_bridge:
                result = await self.nyx_bridge.recall(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    query=query,
                    context=str(context) if context else None,
                    limit=limit
                )
            # Use LLM analysis if requested
            elif use_llm_analysis and self.retriever_agent:
                result = await self.retriever_agent.retrieve_and_analyze(
                    query=query or "",
                    entity_types=[entity_type],
                    top_k=limit,
                    threshold=kwargs.get("similarity_threshold", 0.7)
                )
            else:
                # Use wrapper for standard retrieval
                result = await self.memory_wrapper.recall(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    query=query,
                    context=context,
                    limit=limit
                )
            
            # Include analysis if requested
            if include_analysis and result.get("memories"):
                result["analysis"] = await self.analyze_memory_set(
                    memories=result["memories"],
                    entity_type=entity_type,
                    entity_id=entity_id
                )
            
            # Cache result
            await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"error": str(e), "memories": [], "count": 0}
    
    # ========================================================================
    # Agent-Based Operations
    # ========================================================================
    
    async def agent_remember(
        self,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Store a memory using the memory agent wrapper.
        
        This provides LLM-enhanced memory storage with automatic
        categorization and analysis.
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_agent_wrapper.remember(
            run_context=None,  # Context handled internally
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def agent_recall(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories using the memory agent wrapper.
        
        This provides LLM-enhanced memory retrieval with automatic
        synthesis and analysis.
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_agent_wrapper.recall(
            run_context=None,  # Context handled internally
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            context=context,
            limit=limit
        )
    
    # ========================================================================
    # Schema Operations
    # ========================================================================
    
    async def create_schema(
        self,
        entity_type: str,
        entity_id: int,
        schema_name: str,
        description: str,
        category: str = "general",
        attributes: Dict[str, Any] = None,
        example_memory_ids: List[int] = None
    ) -> Dict[str, Any]:
        """Create a new memory schema."""
        if not self.initialized:
            await self.initialize()
        
        return await self.schema_manager.create_schema(
            entity_type=entity_type,
            entity_id=entity_id,
            schema_name=schema_name,
            description=description,
            category=category,
            attributes=attributes,
            example_memory_ids=example_memory_ids
        )
    
    async def detect_schemas(
        self,
        entity_type: str,
        entity_id: int,
        memory_ids: List[int] = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Detect schemas from memory patterns."""
        if not self.initialized:
            await self.initialize()
        
        return await self.schema_manager.detect_schema_from_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_ids=memory_ids,
            tags=tags
        )
    
    async def apply_schema(
        self,
        memory_id: int,
        entity_type: str,
        entity_id: int,
        schema_id: int = None,
        auto_detect: bool = False
    ) -> Dict[str, Any]:
        """Apply a schema to a memory."""
        if not self.initialized:
            await self.initialize()
        
        return await self.schema_manager.apply_schema_to_memory(
            memory_id=memory_id,
            entity_type=entity_type,
            entity_id=entity_id,
            schema_id=schema_id,
            auto_detect=auto_detect
        )
    
    async def evolve_schema(
        self,
        schema_id: int,
        entity_type: str,
        entity_id: int,
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evolve a schema based on conflicting memories."""
        if not self.initialized:
            await self.initialize()
        
        return await self.schema_manager.evolve_schema_from_conflicts(
            schema_id=schema_id,
            entity_type=entity_type,
            entity_id=entity_id,
            conflicts=conflicts
        )
    
    # ========================================================================
    # Semantic Memory Operations
    # ========================================================================
    
    async def create_belief(
        self,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        supporting_memory_ids: Optional[List[int]] = None,
        confidence: float = 0.5,
        use_governance: bool = False
    ) -> Dict[str, Any]:
        """Create a belief for an entity."""
        if not self.initialized:
            await self.initialize()
        
        if use_governance and self.nyx_bridge:
            return await self.nyx_bridge.create_belief(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=belief_text,
                confidence=confidence
            )
        else:
            return await self.semantic_manager.create_belief(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=belief_text,
                supporting_memory_ids=supporting_memory_ids,
                confidence=confidence
            )
    
    async def get_beliefs(
        self,
        entity_type: str,
        entity_id: int,
        topic: Optional[str] = None,
        min_confidence: float = 0.0,
        use_governance: bool = False
    ) -> List[Dict[str, Any]]:
        """Get beliefs held by an entity."""
        if not self.initialized:
            await self.initialize()
        
        if use_governance and self.nyx_bridge:
            return await self.nyx_bridge.get_beliefs(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic
            )
        else:
            return await self.semantic_manager.get_beliefs(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic,
                min_confidence=min_confidence
            )
    
    async def generate_counterfactual(
        self,
        memory_id: int,
        entity_type: str,
        entity_id: int,
        variation_type: str = "alternative"
    ) -> Dict[str, Any]:
        """Generate a counterfactual memory."""
        if not self.initialized:
            await self.initialize()
        
        return await self.semantic_manager.generate_counterfactual(
            memory_id=memory_id,
            entity_type=entity_type,
            entity_id=entity_id,
            variation_type=variation_type
        )
    
    async def build_semantic_network(
        self,
        entity_type: str,
        entity_id: int,
        central_topic: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Build a semantic network around a topic."""
        if not self.initialized:
            await self.initialize()
        
        return await self.semantic_manager.build_semantic_network(
            entity_type=entity_type,
            entity_id=entity_id,
            central_topic=central_topic,
            depth=depth
        )
    
    # ========================================================================
    # NPC-Specific Operations
    # ========================================================================
    
    async def trigger_npc_revelation(
        self,
        npc_id: int,
        trigger: str = None,
        severity: int = None
    ) -> Dict[str, Any]:
        """Trigger a mask slippage event for an NPC."""
        if not self.initialized:
            await self.initialize()
        
        return await self.mask_manager.generate_mask_slippage(
            npc_id=npc_id,
            trigger=trigger,
            severity=severity
        )
    
    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """Get current mask state for an NPC."""
        if not self.initialized:
            await self.initialize()
        
        return await self.mask_manager.get_npc_mask(npc_id)
    
    async def update_npc_mask_integrity(
        self,
        npc_id: int,
        integrity_change: float,
        reason: str = None
    ) -> Dict[str, Any]:
        """Update an NPC's mask integrity."""
        if not self.initialized:
            await self.initialize()
        
        return await self.mask_manager.update_mask_integrity(
            npc_id=npc_id,
            integrity_change=integrity_change,
            reason=reason
        )
    
    # ========================================================================
    # Player-Specific Operations
    # ========================================================================
    
    async def add_journal_entry(
        self,
        player_name: str,
        entry_text: str,
        entry_type: str = "observation",
        fantasy_flag: bool = False,
        intensity_level: int = 0
    ) -> int:
        """Add a journal entry for a player."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_wrapper.add_journal_entry(
            player_name=player_name,
            entry_text=entry_text,
            entry_type=entry_type,
            fantasy_flag=fantasy_flag,
            intensity_level=intensity_level
        )
    
    async def get_journal_history(
        self,
        player_name: str,
        entry_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get journal history for a player."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_wrapper.get_journal_history(
            player_name=player_name,
            entry_type=entry_type,
            limit=limit
        )
    
    async def get_player_profile(self, player_name: str) -> Dict[str, Any]:
        """Get comprehensive player profile."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_wrapper.player_profile(player_name)
    
    # ========================================================================
    # Emotional Memory Operations
    # ========================================================================
    
    async def update_emotional_state(
        self,
        entity_type: str,
        entity_id: int,
        emotion: str,
        intensity: float = 0.5,
        trauma_event: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an entity's emotional state."""
        if not self.initialized:
            await self.initialize()
        
        current_emotion = {
            "primary_emotion": emotion,
            "intensity": intensity,
            "secondary_emotions": {},
            "valence": 0.0,
            "arousal": 0.0
        }
        
        return await self.emotional_manager.update_entity_emotional_state(
            entity_type=entity_type,
            entity_id=entity_id,
            current_emotion=current_emotion,
            trauma_event=trauma_event
        )
    
    async def get_emotional_state(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """Get an entity's current emotional state."""
        if not self.initialized:
            await self.initialize()
        
        return await self.emotional_manager.get_entity_emotional_state(
            entity_type=entity_type,
            entity_id=entity_id
        )
    
    async def process_trauma_triggers(
        self,
        entity_type: str,
        entity_id: int,
        text: str
    ) -> Dict[str, Any]:
        """Check if text triggers traumatic memories."""
        if not self.initialized:
            await self.initialize()
        
        return await self.emotional_manager.process_traumatic_triggers(
            entity_type=entity_type,
            entity_id=entity_id,
            text=text
        )
    
    # ========================================================================
    # Flashback and Interference Operations
    # ========================================================================
    
    async def generate_flashback(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a flashback for an entity."""
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
        """Create a false memory for an entity."""
        if not self.initialized:
            await self.initialize()
        
        return await self.interference_manager.create_false_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            false_memory_text=false_text,
            related_true_memory_ids=base_memory_ids
        )
    
    async def detect_memory_interference(
        self,
        entity_type: str,
        entity_id: int,
        memory_id: int
    ) -> Dict[str, Any]:
        """Detect interference for a memory."""
        if not self.initialized:
            await self.initialize()
        
        return await self.interference_manager.detect_memory_interference(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_id=memory_id
        )
    
    # ========================================================================
    # Vector Store and Embedding Operations
    # ========================================================================
    
    async def add_to_vector_store(
        self,
        text: str,
        metadata: Dict[str, Any],
        entity_type: str = "memory"
    ) -> str:
        """Add content directly to the vector store."""
        if not self.initialized:
            await self.initialize()
        
        return await self.embedding_service.add_memory(
            text=text,
            metadata=metadata,
            entity_type=entity_type
        )
    
    async def search_vector_store(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the vector store directly."""
        if not self.initialized:
            await self.initialize()
        
        return await self.embedding_service.search_memories(
            query_text=query,
            entity_type=entity_type,
            top_k=top_k,
            filter_dict=filter_dict,
            fetch_content=True
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text."""
        if not self.initialized:
            await self.initialize()
        
        return await self.embedding_service.generate_embedding(text)
    
    # ========================================================================
    # Context Enrichment Operations
    # ========================================================================
    
    async def enrich_context(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich context with relevant memories."""
        if not self.initialized:
            await self.initialize()
        
        return await enrich_context_with_memories(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            user_input=user_input,
            context=context
        )
    
    # ========================================================================
    # Narrative Context Operations
    # ========================================================================
    
    async def get_narrative_context(
        self,
        focus_entities: List[Tuple[str, int]] = None,
        time_window: timedelta = None,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive narrative context."""
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
    
    # ========================================================================
    # Agent Creation Methods
    # ========================================================================
    
    async def create_memory_analysis_agent(self, custom_instructions: str = "") -> Any:
        """
        Create a specialized memory analysis agent following the Nyx pattern.
        
        Args:
            custom_instructions: Additional instructions for the agent
            
        Returns:
            Configured memory analysis agent
        """
        try:
            from agents import Agent, ModelSettings, function_tool, handoff
            
            # Check for preset story constraints (similar to Nyx example)
            preset_constraints = ""
            if "preset_story_id" in custom_instructions:
                preset_constraints = """
==== PRESET STORY ACTIVE ====
A preset story is active. You must follow all established lore and consistency rules.
Do not contradict any pre-established facts about this story world.
"""
            
            combined_instructions = f"""{custom_instructions}
{preset_constraints}

As a Memory Analysis Agent, you must:
1. Analyze memory patterns and relationships
2. Identify narrative threads and developments
3. Detect conflicts and inconsistencies
4. Generate insights from memory clusters
5. Predict future narrative directions
6. Maintain consistency with established lore

Core analytical capabilities:
- Pattern recognition across temporal and semantic dimensions
- Emotional trajectory analysis
- Belief system conflict detection
- Schema evolution tracking
- Narrative coherence assessment

Remember: You are analyzing complex narrative memories. Be thorough, insightful, and maintain consistency with any active preset stories.
"""
            
            # Create the agent
            agent = Agent(
                name="Memory Analyzer",
                instructions=combined_instructions,
                tools=[
                    self._create_analysis_tool("analyze_memory_patterns"),
                    self._create_analysis_tool("identify_narrative_threads"),
                    self._create_analysis_tool("detect_belief_conflicts"),
                    self._create_analysis_tool("predict_developments"),
                    self._create_analysis_tool("assess_coherence")
                ],
                model="gpt-4",
                model_settings=ModelSettings(
                    max_tokens=2000,
                    temperature=0.7
                )
            )
            
            return agent
            
        except ImportError:
            logger.warning("Agents SDK not available, using direct OpenAI calls")
            return None
    
    def _create_analysis_tool(self, tool_name: str):
        """Create a function tool for memory analysis."""
        from agents import function_tool
        
        @function_tool
        async def analysis_tool(context: Dict[str, Any]) -> Dict[str, Any]:
            """Generic analysis tool that routes to appropriate method."""
            if tool_name == "analyze_memory_patterns":
                return await self.analyze_memory_patterns(
                    entity_type=context.get("entity_type"),
                    entity_id=context.get("entity_id"),
                    topic=context.get("topic")
                )
            elif tool_name == "identify_narrative_threads":
                return {"threads": await self._identify_narrative_threads()}
            elif tool_name == "detect_belief_conflicts":
                entities = await self._get_active_entities()
                return {"conflicts": await self._detect_potential_conflicts(entities)}
            elif tool_name == "predict_developments":
                return {"predictions": await self._generate_narrative_predictions(context)}
            elif tool_name == "assess_coherence":
                return await self._assess_narrative_coherence(context)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        analysis_tool.__name__ = tool_name
        return analysis_tool
    
    async def _assess_narrative_coherence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the coherence of the current narrative using LLM analysis.
        
        Args:
            context: Current narrative context
            
        Returns:
            Coherence assessment results
        """
        try:
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Prepare narrative summary
            narrative_summary = {
                "entity_count": len(context.get("entities", {})),
                "relationship_count": len(context.get("relationships", {})),
                "recent_events": len(context.get("recent_events", [])),
                "active_threads": len(await self._identify_narrative_threads()),
                "conflicts": len(context.get("potential_conflicts", [])),
                "emotional_states": {
                    entity: state.get("current", {}).get("primary_emotion", "neutral")
                    for entity, state in context.get("emotional_landscape", {}).items()
                }
            }
            
            prompt = f"""Assess the narrative coherence of this roleplay scenario:

{json.dumps(narrative_summary, indent=2)}

Evaluate:
1. Logical consistency of events
2. Character motivation alignment
3. Temporal consistency
4. Emotional trajectory believability
5. World-building consistency

Provide scores (0.0-1.0) for each dimension and an overall coherence score.

Return JSON:
{{
    "logical_consistency": 0.0-1.0,
    "character_motivation": 0.0-1.0,
    "temporal_consistency": 0.0-1.0,
    "emotional_believability": 0.0-1.0,
    "world_consistency": 0.0-1.0,
    "overall_coherence": 0.0-1.0,
    "issues": ["issue1", ...],
    "recommendations": ["recommendation1", ...]
}}
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a narrative coherence analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            result["assessed_at"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing narrative coherence: {e}")
            return {
                "error": str(e),
                "overall_coherence": 0.5,
                "assessed_at": datetime.now().isoformat()
            }
    
    # ========================================================================
    # Analysis Operations
    # ========================================================================
    
    async def analyze_entity_memories(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int
    ) -> Dict[str, Any]:
        """Comprehensive analysis of an entity's memories."""
        if not self.initialized:
            await self.initialize()
        
        if use_governance := False:  # Can be parameterized
            return await self.nyx_bridge.analyze_memories(
                entity_type=entity_type,
                entity_id=entity_id
            )
        else:
            return await self.integrated_system.analyze_entity_memories(
                entity_type=entity_type,
                entity_id=entity_id
            )
    
    async def analyze_memory_patterns(
        self,
        entity_type: Union[str, EntityType] = None,
        entity_id: int = None,
        topic: str = None
    ) -> Dict[str, Any]:
        """Analyze patterns across memories."""
        if not self.initialized:
            await self.initialize()
        
        return await self.semantic_manager.find_patterns_across_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            topic=topic
        ) if entity_type and entity_id else await self._analyze_cross_entity_patterns(topic)
    
    # ========================================================================
    # Maintenance and Telemetry Operations
    # ========================================================================
    
    async def run_maintenance(
        self,
        entity_type: str = None,
        entity_id: int = None,
        operations: List[str] = None,
        use_governance: bool = False
    ) -> Dict[str, Any]:
        """Run maintenance operations."""
        if not self.initialized:
            await self.initialize()
        
        if entity_type and entity_id and use_governance and self.nyx_bridge:
            return await self.nyx_bridge.run_maintenance(
                entity_type=entity_type,
                entity_id=entity_id
            )
        elif entity_type and entity_id:
            return await self.integrated_system.run_memory_maintenance(
                entity_type=entity_type,
                entity_id=entity_id
            )
        else:
            # Run global maintenance
            results = {}
            operations = operations or [
                "consolidation",
                "decay",
                "schema_maintenance",
                "reconsolidation",
                "cleanup"
            ]
            
            entities = await self._get_active_entities()
            
            for op in operations:
                if op == "consolidation":
                    results["consolidation"] = await self._run_consolidation()
                elif op == "decay":
                    results["decay"] = await self._run_decay()
                elif op == "schema_maintenance":
                    results["schemas"] = await self._run_schema_maintenance()
                elif op == "reconsolidation":
                    results["reconsolidation"] = await self._run_reconsolidation_maintenance()
                elif op == "cleanup":
                    results["cleanup"] = await self.maintenance.cleanup_old_memories()
                elif op == "telemetry_cleanup":
                    results["telemetry_cleanup"] = await self.telemetry.cleanup_old_telemetry(
                        self.user_id, self.conversation_id
                    )
            
            return results
    
    async def get_telemetry_metrics(
        self,
        time_window_minutes: int = 15
    ) -> Dict[str, Any]:
        """Get telemetry metrics for memory operations."""
        if not self.initialized:
            await self.initialize()
        
        return await self.telemetry.get_recent_metrics(
            self.user_id,
            self.conversation_id,
            time_window_minutes
        )
    
    async def get_slow_operations(
        self,
        threshold_ms: float = 500,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent slow memory operations."""
        if not self.initialized:
            await self.initialize()
        
        return await self.telemetry.get_slow_operations(
            threshold_ms=threshold_ms,
            limit=limit
        )
    
    # ========================================================================
    # Advanced LLM-Enhanced Operations
    # ========================================================================
    
    async def generate_memory_prompt(
        self,
        entity_type: str,
        entity_id: int,
        context: Dict[str, Any],
        prompt_type: str = "recall"
    ) -> str:
        """
        Generate an optimized prompt for memory operations using LLM.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            context: Current context
            prompt_type: Type of prompt (recall/reflection/analysis)
            
        Returns:
            Generated prompt string
        """
        try:
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Get entity's recent memories and state
            manager = await self._get_entity_manager(entity_type, entity_id)
            recent_memories = await manager.retrieve_memories(limit=5, conn=None)
            
            memory_summary = [
                m.text[:100] if hasattr(m, 'text') else str(m)[:100]
                for m in recent_memories
            ]
            
            emotional_state = None
            if entity_type in ["npc", "player"]:
                emotional_state = await self.emotional_manager.get_entity_emotional_state(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
            
            prompt = f"""Generate an optimal memory {prompt_type} prompt for this entity:

Entity: {entity_type} (ID: {entity_id})
Recent memories: {json.dumps(memory_summary)}
Emotional state: {json.dumps(emotional_state) if emotional_state else "N/A"}
Current context: {json.dumps(context, default=str)[:500]}

Generate a prompt that will:
1. Trigger relevant memory recall
2. Maintain narrative consistency
3. Respect the entity's current emotional state
4. Encourage rich, detailed responses
5. Stay true to the entity's established patterns

Return only the prompt text, no explanation.
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You generate memory prompts for narrative AI systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating memory prompt: {e}")
            # Fallback to simple prompt
            if prompt_type == "recall":
                return f"What does {entity_type} {entity_id} remember about recent events?"
            elif prompt_type == "reflection":
                return f"How does {entity_type} {entity_id} feel about what has happened?"
            else:
                return f"Analyze the memories of {entity_type} {entity_id}"
    
    async def synthesize_memory_narrative(
        self,
        entity_type: str,
        entity_id: int,
        memories: List[Any],
        perspective: str = "first_person"
    ) -> str:
        """
        Synthesize multiple memories into a coherent narrative using LLM.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            memories: List of memories to synthesize
            perspective: Narrative perspective (first_person/third_person)
            
        Returns:
            Synthesized narrative text
        """
        try:
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Format memories for synthesis
            memory_texts = []
            for m in memories[:10]:  # Limit to 10 memories
                if isinstance(m, dict):
                    memory_texts.append({
                        "text": m.get("text", ""),
                        "significance": m.get("significance", "medium"),
                        "emotion": m.get("emotional_intensity", 0)
                    })
                else:
                    memory_texts.append({
                        "text": getattr(m, "text", str(m)),
                        "significance": getattr(m, "significance", "medium"),
                        "emotion": getattr(m, "emotional_intensity", 0)
                    })
            
            perspective_instruction = (
                "Use first-person perspective (I, me, my)" if perspective == "first_person"
                else "Use third-person perspective"
            )
            
            prompt = f"""Synthesize these memories into a coherent narrative:

Entity: {entity_type} (ID: {entity_id})
Memories: {json.dumps(memory_texts, indent=2)}

Instructions:
1. {perspective_instruction}
2. Maintain chronological order where apparent
3. Emphasize high-significance memories
4. Reflect emotional content appropriately
5. Create smooth transitions between memories
6. Keep the narrative engaging and character-consistent
7. Length: 150-250 words

Generate only the narrative text, no meta-commentary.
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a narrative synthesizer for memory systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing memory narrative: {e}")
            # Fallback to simple concatenation
            texts = []
            for m in memories[:5]:
                if isinstance(m, dict):
                    texts.append(m.get("text", ""))
                else:
                    texts.append(getattr(m, "text", str(m)))
            
            return " ".join(texts)
    
    async def generate_memory_questions(
        self,
        entity_type: str,
        entity_id: int,
        purpose: str = "exploration"
    ) -> List[str]:
        """
        Generate questions to explore an entity's memories using LLM.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            purpose: Purpose of questions (exploration/therapy/investigation)
            
        Returns:
            List of generated questions
        """
        try:
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Get entity's memory summary
            summary = await self.analyze_entity_memories(entity_type, entity_id)
            
            prompt = f"""Generate questions to explore this entity's memories:

Entity: {entity_type} (ID: {entity_id})
Memory summary:
- Total memories: {summary.get('total_memories', 0)}
- Common themes: {list(summary.get('common_tags', {}).keys())[:5]}
- Emotional profile: {summary.get('emotional_profile', {}).get('current_emotion', 'unknown')}

Purpose: {purpose}

Generate 5 insightful questions that will:
1. Uncover hidden connections between memories
2. Explore emotional depths
3. Reveal character motivations
4. Challenge assumptions
5. Encourage self-reflection

Return as JSON array of strings.
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You generate insightful questions for memory exploration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            return result if isinstance(result, list) else result.get("questions", [])
            
        except Exception as e:
            logger.error(f"Error generating memory questions: {e}")
            # Fallback questions
            return [
                "What is your earliest memory?",
                "What memory brings you the most joy?",
                "What do you try not to think about?",
                "What pattern do you see in your experiences?",
                "What would you change if you could?"
            ]
    
    # ========================================================================
    # Governance and Integration Operations
    # ========================================================================
    
    async def get_memory_state(self) -> Dict[str, Any]:
        """Get current memory system state for governance."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_integration.get_state()
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for the memory system."""
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_integration.get_health_metrics()
    
    async def process_memory_directive(
        self,
        directive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a directive from Nyx governance."""
        if not self.initialized:
            await self.initialize()
        
        return await self.nyx_bridge.process_memory_directive(directive_data)
    
    # ========================================================================
    # Helper Methods (kept private as before)
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
        context["active_schemas"] = schemas[:5]
        
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
        """Identify active narrative threads from memories using LLM analysis."""
        try:
            # Get recent significant memories across all entities
            recent_events = await self._get_recent_significant_events()
            if not recent_events:
                return []
            
            # Get OpenAI client
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Format memories for analysis
            events_text = "\n".join([
                f"- [{event['entity']}] {event['text']} (significance: {event['significance']})"
                for event in recent_events[:20]  # Limit to 20 most recent
            ])
            
            prompt = f"""Analyze these recent narrative events and identify active story threads:

{events_text}

Identify 3-5 major narrative threads that connect these events. For each thread:
1. Give it a descriptive name
2. Identify the central conflict or tension
3. List the entities involved
4. Assess the current state (building/climax/resolving)
5. Predict likely next development

Return as JSON array with this structure:
[{{
    "thread_name": "...",
    "central_conflict": "...",
    "involved_entities": ["entity_type_id", ...],
    "current_state": "building|climax|resolving",
    "next_development": "...",
    "urgency": "low|medium|high"
}}]
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a narrative analyst identifying story threads in roleplay scenarios."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            threads = result if isinstance(result, list) else result.get("threads", [])
            
            # Add metadata to each thread
            for thread in threads:
                thread["identified_at"] = datetime.now().isoformat()
                thread["source_event_count"] = len(recent_events)
            
            return threads
            
        except Exception as e:
            logger.error(f"Error identifying narrative threads: {e}")
            return []
    
    async def _detect_potential_conflicts(
        self,
        entities: List[Tuple[str, int]]
    ) -> List[Dict[str, Any]]:
        """Detect potential conflicts between entities."""
        conflicts = []
        
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
        """Check if two beliefs conflict using LLM analysis."""
        try:
            # Get OpenAI client
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            text1 = belief1.get("belief", "")
            text2 = belief2.get("belief", "")
            
            # Quick rule-based check first
            opposites = [
                ("should", "should not"),
                ("must", "must not"),
                ("good", "bad"),
                ("right", "wrong"),
                ("always", "never")
            ]
            
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            for pos, neg in opposites:
                if (pos in text1_lower and neg in text2_lower) or (neg in text1_lower and pos in text2_lower):
                    return True
            
            # Use LLM for more nuanced conflict detection
            prompt = f"""Analyze if these two beliefs are in conflict:

Belief 1: {text1}
Belief 2: {text2}

Consider:
1. Direct contradictions
2. Incompatible values or goals
3. Mutually exclusive worldviews
4. Conflicting behavioral prescriptions

Return JSON with structure:
{{
    "conflicts": true/false,
    "conflict_type": "direct|values|worldview|behavioral|none",
    "severity": "minor|moderate|severe",
    "explanation": "..."
}}
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a belief conflict analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Return true if conflict detected and severity is moderate or severe
            return result.get("conflicts", False) and result.get("severity", "minor") in ["moderate", "severe"]
            
        except Exception as e:
            logger.error(f"Error in LLM belief conflict detection: {e}")
            # Fallback to simple rule-based check
            text1 = belief1.get("belief", "").lower()
            text2 = belief2.get("belief", "").lower()
            
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
        """Generate predictions about future narrative developments using LLM analysis."""
        try:
            # Get OpenAI client
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            # Prepare context summary
            context_summary = {
                "entities": {},
                "relationships": context.get("relationships", {}),
                "recent_events": context.get("recent_events", [])[:10],
                "emotional_landscape": context.get("emotional_landscape", {}),
                "active_schemas": [
                    {"name": s.get("name"), "description": s.get("description")}
                    for s in context.get("active_schemas", [])[:5]
                ],
                "potential_conflicts": context.get("potential_conflicts", [])
            }
            
            # Simplify entity data
            for entity_key, entity_data in context.get("entities", {}).items():
                context_summary["entities"][entity_key] = {
                    "emotional_state": entity_data.get("emotional_state", {}).get("current_emotion"),
                    "recent_memories": [
                        m.get("text", "")[:100] if isinstance(m, dict) else str(m)[:100]
                        for m in entity_data.get("recent_memories", [])[:3]
                    ],
                    "beliefs": [
                        b.get("belief", "") if isinstance(b, dict) else str(b)
                        for b in entity_data.get("beliefs", [])[:3]
                    ]
                }
            
            prompt = f"""Based on this narrative context, predict likely future developments:

CURRENT STATE:
{json.dumps(context_summary, indent=2)}

Analyze the current narrative state and predict 3-5 likely developments. Consider:
1. Emotional tensions and their likely resolutions
2. Unresolved conflicts that may escalate
3. Character arcs and their trajectories
4. Schema patterns suggesting future events
5. Relationship dynamics and potential changes

For each prediction, provide:
- The predicted event/development
- Which entities are involved
- Confidence level (0.0-1.0)
- Timeframe (immediate/soon/eventual)
- Trigger conditions
- Narrative impact (low/medium/high)

Return as JSON array:
[{{
    "prediction": "...",
    "involved_entities": ["entity_type_id", ...],
    "confidence": 0.0-1.0,
    "timeframe": "immediate|soon|eventual",
    "trigger_conditions": ["condition1", ...],
    "narrative_impact": "low|medium|high",
    "reasoning": "..."
}}]
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a narrative prediction system analyzing roleplay scenarios to predict future developments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group())
            else:
                # Fallback: create basic predictions from emotional states
                predictions = []
                for entity, data in context.get("entities", {}).items():
                    if data.get("emotional_state"):
                        emotion = data["emotional_state"].get("current_emotion", {})
                        if emotion.get("intensity", 0) > 0.7:
                            predictions.append({
                                "prediction": f"High emotional intensity may lead to impulsive action",
                                "involved_entities": [entity],
                                "confidence": 0.7,
                                "timeframe": "immediate",
                                "trigger_conditions": ["continued high emotion"],
                                "narrative_impact": "medium",
                                "reasoning": "Intense emotions often drive immediate action"
                            })
            
            # Add metadata
            for pred in predictions:
                pred["generated_at"] = datetime.now().isoformat()
                pred["context_hash"] = hash(json.dumps(context_summary, sort_keys=True))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating narrative predictions: {e}")
            # Return simple rule-based predictions as fallback
            predictions = []
            for entity, data in context.get("entities", {}).items():
                if data.get("emotional_state"):
                    emotion = data["emotional_state"].get("current_emotion", {})
                    if emotion.get("intensity", 0) > 0.7:
                        predictions.append({
                            "prediction": f"High emotional intensity may lead to impulsive action",
                            "involved_entities": [entity],
                            "confidence": 0.7,
                            "timeframe": "immediate",
                            "trigger_conditions": ["continued high emotion"],
                            "narrative_impact": "medium",
                            "reasoning": "Intense emotions often drive immediate action"
                        })
            return predictions
    
    async def analyze_memory_set(
        self,
        memories: List[Any],
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """Analyze a set of memories for patterns and insights."""
        if not memories:
            return {"message": "No memories to analyze"}
        
        # Handle both Memory objects and dicts
        analysis = {
            "total_count": len(memories),
            "average_significance": 0,
            "average_emotional_intensity": 0,
            "common_tags": {},
            "temporal_distribution": {},
            "memory_types": {}
        }
        
        for memory in memories:
            # Handle dict format
            if isinstance(memory, dict):
                significance = memory.get("significance", 0)
                emotional_intensity = memory.get("emotional_intensity", 0)
                tags = memory.get("tags", [])
                memory_type = memory.get("type", "unknown")
            else:
                # Handle Memory object
                significance = getattr(memory, "significance", 0)
                emotional_intensity = getattr(memory, "emotional_intensity", 0)
                tags = getattr(memory, "tags", [])
                memory_type = getattr(memory, "memory_type", "unknown")
                if hasattr(memory_type, 'value'):
                    memory_type = memory_type.value
            
            analysis["average_significance"] += significance
            analysis["average_emotional_intensity"] += emotional_intensity
            
            for tag in tags:
                analysis["common_tags"][tag] = analysis["common_tags"].get(tag, 0) + 1
            
            analysis["memory_types"][str(memory_type)] = analysis["memory_types"].get(str(memory_type), 0) + 1
        
        if len(memories) > 0:
            analysis["average_significance"] /= len(memories)
            analysis["average_emotional_intensity"] /= len(memories)
        
        return analysis
    
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
    
    async def _analyze_cross_entity_patterns(self, topic: str = None) -> Dict[str, Any]:
        """Analyze patterns across all entities using LLM."""
        try:
            entities = await self._get_active_entities()
            all_patterns = []
            entity_memories = {}
            
            # Collect memories from each entity
            for entity_type, entity_id in entities:
                manager = await self._get_entity_manager(entity_type, entity_id)
                memories = await manager.retrieve_memories(
                    query=topic,
                    limit=10,
                    conn=None
                )
                
                entity_key = f"{entity_type}_{entity_id}"
                entity_memories[entity_key] = [
                    {
                        "text": m.text if hasattr(m, 'text') else m.get('text', ''),
                        "significance": getattr(m, 'significance', 0) if hasattr(m, 'significance') else m.get('significance', 0),
                        "type": str(getattr(m, 'memory_type', 'unknown')) if hasattr(m, 'memory_type') else m.get('type', 'unknown')
                    }
                    for m in memories[:5]  # Limit to 5 per entity
                ]
            
            # Use LLM to find cross-entity patterns
            from logic.chatgpt_integration import get_openai_client
            client = get_openai_client()
            
            prompt = f"""Analyze these memories from different entities to find cross-entity patterns:

{json.dumps(entity_memories, indent=2)}

{"Topic focus: " + topic if topic else "No specific topic focus"}

Identify:
1. Shared themes across entities
2. Complementary or conflicting perspectives
3. Causal relationships between entity experiences
4. Emergent narrative patterns
5. Systemic behaviors or cycles

Return JSON:
{{
    "shared_themes": [
        {{
            "theme": "...",
            "entities_involved": ["entity_key", ...],
            "description": "...",
            "significance": "low|medium|high"
        }}
    ],
    "relationships": [
        {{
            "type": "causal|complementary|conflicting",
            "entity1": "entity_key",
            "entity2": "entity_key",
            "description": "..."
        }}
    ],
    "emergent_patterns": [
        {{
            "pattern": "...",
            "description": "...",
            "entities_affected": ["entity_key", ...],
            "implications": "..."
        }}
    ],
    "systemic_behaviors": [
        {{
            "behavior": "...",
            "cycle_description": "...",
            "participating_entities": ["entity_key", ...]
        }}
    ]
}}
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cross-entity pattern analyzer for narrative memory systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result["analysis_timestamp"] = datetime.now().isoformat()
            result["entity_count"] = len(entities)
            result["topic"] = topic
            result["total_memories_analyzed"] = sum(len(mems) for mems in entity_memories.values())
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing cross-entity patterns: {e}")
            # Fallback to simple pattern collection
            patterns = []
            entities = await self._get_active_entities()
            
            for entity_type, entity_id in entities:
                entity_patterns = await self.semantic_manager.find_patterns_across_memories(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    topic=topic
                )
                patterns.append({
                    "entity": f"{entity_type}_{entity_id}",
                    "patterns": entity_patterns
                })
            
            return {
                "cross_entity_patterns": patterns,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": "LLM analysis failed, using fallback"
            }
    
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
    
    This is the primary entry point for all memory operations in the system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        MemoryOrchestrator instance
    """
    return await MemoryOrchestrator.get_instance(user_id, conversation_id)
