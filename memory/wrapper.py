# memory/wrapper.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .integrated import IntegratedMemorySystem, init_memory_system
from .core import Memory, MemoryType, MemorySignificance
from .managers import NPCMemoryManager, NyxMemoryManager, PlayerMemoryManager
from .masks import ProgressiveRevealManager, RevealType, RevealSeverity
from .emotional import EmotionalMemoryManager
from .flashbacks import FlashbackManager

logger = logging.getLogger("memory_wrapper")

class MemorySystem:
    """
    A simplified interface to the complex memory system.
    Provides easy-to-use methods for common operations while
    still allowing access to the advanced features.
    """
    
    _instances = {}  # Singleton pattern to avoid duplicate initialization
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int):
        """Get or create a memory system instance for this user/conversation."""
        key = (user_id, conversation_id)
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the memory system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.integrated = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory system and its components."""
        if self.initialized:
            return
            
        # Initialize the integrated memory system
        self.integrated = await init_memory_system(self.user_id, self.conversation_id)
        self.initialized = True
        logger.info(f"Memory system initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    # =========================================================================
    # High-level memory operations (simplified for common use)
    # =========================================================================
    
    async def remember(self, entity_type: str, entity_id: int, memory_text: str, 
                      importance: str = "medium", emotional: bool = True, 
                      tags: List[str] = None) -> Dict[str, Any]:
        """
        Record a new memory for an entity (NPC, player, or DM).
        
        Args:
            entity_type: Type of entity ("npc", "player", "nyx")
            entity_id: ID of the entity
            memory_text: The memory text to record
            importance: Importance level ("trivial", "low", "medium", "high", "critical")
            emotional: Whether to analyze emotional content
            tags: Optional tags for the memory
            
        Returns:
            Information about the created memory
        """
        if not self.initialized:
            await self.initialize()
        
        # Map importance string to MemorySignificance enum
        significance_map = {
            "trivial": MemorySignificance.TRIVIAL,
            "low": MemorySignificance.LOW,
            "medium": MemorySignificance.MEDIUM,
            "high": MemorySignificance.HIGH,
            "critical": MemorySignificance.CRITICAL
        }
        significance = significance_map.get(importance.lower(), MemorySignificance.MEDIUM)
        
        # Add memory with integrated processing
        result = await self.integrated.add_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            memory_kwargs={
                "significance": significance,
                "tags": tags or [],
                "apply_schemas": True,
                "check_interference": True
            }
        )
        
        return result
    
    async def recall(self, entity_type: str, entity_id: int, query: str = None, 
                    context: Dict[str, Any] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Recall memories for an entity, optionally filtered by a query.
        
        Args:
            entity_type: Type of entity ("npc", "player", "nyx")
            entity_id: ID of the entity
            query: Optional search query
            context: Current context that might influence recall
            limit: Maximum number of memories to return
            
        Returns:
            Retrieved memories and related information
        """
        if not self.initialized:
            await self.initialize()
        
        # Format context if provided as string
        if isinstance(context, str):
            context = {"text": context}
        elif context is None:
            context = {}
        
        # Retrieve memories with integrated processing
        result = await self.integrated.retrieve_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            current_context=context,
            retrieval_kwargs={
                "limit": limit,
                "use_emotional_context": True,
                "simulate_competition": True,
                "allow_flashbacks": True,
                "include_schema_interpretation": True
            }
        )
        
        return result
    
    async def maintain(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """
        Run maintenance tasks on an entity's memories (consolidation, decay, etc.).
        
        Args:
            entity_type: Type of entity ("npc", "player", "nyx")
            entity_id: ID of the entity
            
        Returns:
            Results of maintenance operations
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.integrated.run_memory_maintenance(
            entity_type=entity_type,
            entity_id=entity_id
        )
    
    # =========================================================================
    # Specialized NPC memory functions
    # =========================================================================
    
    async def npc_memory(self, npc_id: int) -> NPCMemoryManager:
        """Get a specialized memory manager for an NPC."""
        return NPCMemoryManager(npc_id, self.user_id, self.conversation_id)
    
    async def reveal_npc_trait(self, npc_id: int, trigger: str = None, 
                              severity: int = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event for an NPC to reveal aspects of their true nature.
        
        Args:
            npc_id: ID of the NPC
            trigger: What triggered the reveal (optional)
            severity: How significant the reveal is (1-5, optional)
            
        Returns:
            Details of the revelation
        """
        if not self.initialized:
            await self.initialize()
        
        mask_manager = ProgressiveRevealManager(self.user_id, self.conversation_id)
        return await mask_manager.generate_mask_slippage(
            npc_id=npc_id,
            trigger=trigger,
            severity=severity
        )
    
    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get information about an NPC's mask (presented vs. hidden traits).
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Mask information
        """
        if not self.initialized:
            await self.initialize()
        
        mask_manager = ProgressiveRevealManager(self.user_id, self.conversation_id)
        return await mask_manager.get_npc_mask(npc_id)
    
    async def npc_flashback(self, npc_id: int, context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC based on current context.
        
        Args:
            npc_id: ID of the NPC
            context: Current context that might trigger a flashback
            
        Returns:
            Flashback information or None
        """
        if not self.initialized:
            await self.initialize()
        
        flashback_manager = FlashbackManager(ctx.context["user_id"], ctx.context["conversation_id"])
        return await flashback_manager.generate_flashback(
            entity_type="npc",
            entity_id=npc_id,
            current_context=context
        )
    
    async def update_npc_emotion(self, npc_id: int, emotion: str, 
                               intensity: float = 0.5) -> Dict[str, Any]:
        """
        Update an NPC's emotional state.
        
        Args:
            npc_id: ID of the NPC
            emotion: Primary emotion (e.g., "anger", "joy", "fear")
            intensity: Intensity of the emotion (0.0-1.0)
            
        Returns:
            Updated emotional state
        """
        if not self.initialized:
            await self.initialize()
        
        emotional_manager = EmotionalMemoryManager(self.user_id, self.conversation_id)
        
        current_emotion = {
            "primary_emotion": emotion,
            "intensity": intensity,
            "secondary_emotions": {},
            "valence": 0.0,  # Will be calculated by the manager
            "arousal": 0.0    # Will be calculated by the manager
        }
        
        return await emotional_manager.update_entity_emotional_state(
            entity_type="npc",
            entity_id=npc_id,
            current_emotion=current_emotion
        )
    
    async def get_npc_emotion(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Current emotional state
        """
        if not self.initialized:
            await self.initialize()
        
        emotional_manager = EmotionalMemoryManager(self.user_id, self.conversation_id)
        return await emotional_manager.get_entity_emotional_state(
            entity_type="npc",
            entity_id=npc_id
        )
    
    # =========================================================================
    # Specialized DM (Nyx) memory functions
    # =========================================================================
    
    async def dm_memory(self) -> NyxMemoryManager:
        """Get a specialized memory manager for the DM (Nyx)."""
        return NyxMemoryManager(self.user_id, self.conversation_id)
    
    async def add_narrative_reflection(self, reflection: str, 
                                     reflection_type: str = "general",
                                     importance: str = "medium") -> int:
        """
        Add a reflection for the DM about the narrative or game state.
        
        Args:
            reflection: The reflection text
            reflection_type: Type of reflection
            importance: Importance level ("trivial", "low", "medium", "high", "critical")
            
        Returns:
            ID of the created reflection memory
        """
        if not self.initialized:
            await self.initialize()
        
        # Map importance string to MemorySignificance enum
        significance_map = {
            "trivial": MemorySignificance.TRIVIAL,
            "low": MemorySignificance.LOW,
            "medium": MemorySignificance.MEDIUM,
            "high": MemorySignificance.HIGH,
            "critical": MemorySignificance.CRITICAL
        }
        significance = significance_map.get(importance.lower(), MemorySignificance.MEDIUM)
        
        nyx_manager = NyxMemoryManager(self.user_id, self.conversation_id)
        return await nyx_manager.add_reflection(
            reflection=reflection,
            reflection_type=reflection_type,
            significance=significance
        )
    
    async def get_narrative_state(self) -> Dict[str, Any]:
        """
        Get the current narrative state from the DM's perspective.
        
        Returns:
            Narrative state information
        """
        if not self.initialized:
            await self.initialize()
        
        nyx_manager = NyxMemoryManager(self.user_id, self.conversation_id)
        return await nyx_manager.get_narrative_state()
    
    async def generate_plot_hooks(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate potential plot hooks based on current game state.
        
        Args:
            count: Number of hooks to generate
            
        Returns:
            List of plot hook information
        """
        if not self.initialized:
            await self.initialize()
        
        nyx_manager = NyxMemoryManager(self.user_id, self.conversation_id)
        return await nyx_manager.generate_plot_hooks(count=count)
    
    # =========================================================================
    # Specialized player memory functions
    # =========================================================================
    
    async def player_memory(self, player_name: str) -> PlayerMemoryManager:
        """Get a specialized memory manager for a player."""
        return PlayerMemoryManager(player_name, self.user_id, self.conversation_id)
    
    async def add_journal_entry(self, player_name: str, entry_text: str,
                              entry_type: str = "observation",
                              fantasy_flag: bool = False,
                              intensity_level: int = 0) -> int:
        """
        Add a journal entry to a player's memory.
        
        Args:
            player_name: Name of the player
            entry_text: The journal entry text
            entry_type: Type of entry
            fantasy_flag: Whether this is a fantasy/dream
            intensity_level: Emotional intensity (0-5)
            
        Returns:
            ID of the created journal entry
        """
        if not self.initialized:
            await self.initialize()
        
        player_manager = PlayerMemoryManager(player_name, self.user_id, self.conversation_id)
        return await player_manager.add_journal_entry(
            entry_text=entry_text,
            entry_type=entry_type,
            fantasy_flag=fantasy_flag,
            intensity_level=intensity_level
        )
    
    async def get_journal_history(self, player_name: str, entry_type: str = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a player's journal entries.
        
        Args:
            player_name: Name of the player
            entry_type: Optional filter by entry type
            limit: Maximum number of entries to return
            
        Returns:
            List of journal entries
        """
        if not self.initialized:
            await self.initialize()
        
        player_manager = PlayerMemoryManager(player_name, self.user_id, self.conversation_id)
        return await player_manager.get_journal_history(
            entry_type=entry_type,
            limit=limit
        )
    
    async def player_profile(self, player_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive player profile with stats, history, etc.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Player profile information
        """
        if not self.initialized:
            await self.initialize()
        
        player_manager = PlayerMemoryManager(player_name, self.user_id, self.conversation_id)
        return await player_manager.compile_player_profile()
    
    # =========================================================================
    # Advanced memory operations (for more complex use cases)
    # =========================================================================
    
    async def analyze_entity_memory(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of an entity's memories.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Memory analysis results
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.integrated.analyze_entity_memories(
            entity_type=entity_type,
            entity_id=entity_id
        )
    
    async def generate_schemas(self, entity_type: str, entity_id: int) -> Dict[str, Any]:
        """
        Generate schemas by analyzing memory patterns.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Generated schemas information
        """
        if not self.initialized:
            await self.initialize()
        
        return await self.integrated.generate_schemas_from_memories(
            entity_type=entity_type,
            entity_id=entity_id
        )
    
    async def create_false_memory(self, entity_type: str, entity_id: int,
                                false_memory_text: str,
                                related_true_memory_ids: List[int] = None) -> Dict[str, Any]:
        """
        Create a fabricated false memory for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            false_memory_text: Text of the false memory
            related_true_memory_ids: Optional IDs of related true memories
            
        Returns:
            Created false memory information
        """
        if not self.initialized:
            await self.initialize()
        
        interference_manager = self.integrated.interference_manager
        return await interference_manager.create_false_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            false_memory_text=false_memory_text,
            related_true_memory_ids=related_true_memory_ids
        )
    
    async def create_belief(self, entity_type: str, entity_id: int,
                          belief_text: str,
                          confidence: float = 0.7) -> Dict[str, Any]:
        """
        Create a belief for an entity based on their experiences.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            belief_text: The belief statement
            confidence: Confidence in this belief (0.0-1.0)
            
        Returns:
            Created belief information
        """
        if not self.initialized:
            await self.initialize()
        
        semantic_manager = self.integrated.semantic_manager
        return await semantic_manager.create_belief(
            entity_type=entity_type,
            entity_id=entity_id,
            belief_text=belief_text,
            confidence=confidence
        )
    
    async def get_beliefs(self, entity_type: str, entity_id: int,
                        topic: str = None) -> List[Dict[str, Any]]:
        """
        Get beliefs held by an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            topic: Optional topic filter
            
        Returns:
            List of beliefs
        """
        if not self.initialized:
            await self.initialize()
        
        semantic_manager = self.integrated.semantic_manager
        return await semantic_manager.get_beliefs(
            entity_type=entity_type,
            entity_id=entity_id,
            topic=topic
        )
    
    # Direct access to specialized components for advanced usage
    @property
    def core(self):
        """Direct access to the integrated memory system."""
        return self.integrated
