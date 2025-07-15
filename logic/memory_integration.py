# logic/memory_integration.py

"""
Memory integration module connecting the enhanced memory system from IntegratedNPCSystem
with the existing game memory functions.
"""

import logging
import json
import random
import asyncio
import time
import asyncpg
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from db.connection import get_db_connection_context
from logic.fully_integrated_npc_system import IntegratedNPCSystem

class MemoryIntegration:
    """
    Bridge class that integrates the sophisticated memory management from 
    IntegratedNPCSystem with the existing memory handling.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize memory integration.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_system = IntegratedNPCSystem(user_id, conversation_id)

    async def initialize(self) -> "MemoryIntegration":
        """
        Initialise the bridge with deep, timestamped tracing **and**
        defensive timeouts so any step that stalls becomes an explicit
        `asyncio.TimeoutError` in the logs.

        Returns:
            self (so you can `return await bridge.initialize()` in callers)
        """
        # ------------------------------------------------------------
        # ❶  One-time setup / sanity checks
        # ------------------------------------------------------------
        import time, asyncio, logging                         # local import keeps globals tidy
        from nyx.nyx_governance import (
            AgentType,
            DirectiveType,
            DirectivePriority,
        )
        from memory.core.memory_system import MemorySystem    # ← adjust if your path differs
        from memory.memory_agent import (
            MemoryAgentWrapper,
            create_memory_agent,
            MemoryContext,            # class that holds shared state
        )

        if not hasattr(self, "logger"):
            # guarantee a logger even if outer scope changed
            self.logger = logging.getLogger(__name__)

        # Initialise a MemoryContext if parent didn’t do it
        if not hasattr(self, "memory_context") or self.memory_context is None:
            self.memory_context = MemoryContext()

        # Call parent initialisation if the base-class needs it
        # (safe even if MemoryIntegration doesn’t ultimately subclass anything).
        try:
            base_init = super().initialize          # may raise AttributeError
        except AttributeError:
            pass
        else:
            await base_init()                       # no timeout here – base class should be cheap

        # ------------------------------------------------------------
        # ❷  Timed / traced steps
        # ------------------------------------------------------------
        t0 = time.perf_counter()
        log = lambda step: self.logger.debug(
            "[MI-init +%.3fs] %s", time.perf_counter() - t0, step
        )

        try:
            # 1) Create underlying OpenAI-agent-based memory agent
            log("1) create_memory_agent()")
            base_agent = create_memory_agent(self.user_id, self.conversation_id)

            # 2) Wrap in the compatibility layer
            log("2) MemoryAgentWrapper()")
            self.memory_agent = MemoryAgentWrapper(base_agent, self.memory_context)

            # 3) Fetch / build MemorySystem instance   (10-second watchdog)
            log("3) MemorySystem.get_instance()")
            self.memory_system = await asyncio.wait_for(
                MemorySystem.get_instance(self.user_id, self.conversation_id),
                timeout=10.0,
            )
            self.memory_context.memory_system = self.memory_system

            # 4) Register with Nyx governance         (10-second watchdog)
            log("4) governor.register_agent()")
            await asyncio.wait_for(
                self.governor.register_agent(
                    agent_type=AgentType.MEMORY_MANAGER,
                    agent_id=f"memory_manager:{self.conversation_id}",
                    agent_instance=self.memory_agent,
                ),
                timeout=10.0,
            )

            # 5) Issue default memory-maintenance directive (10-second watchdog)
            log("5) governor.issue_directive()")
            await asyncio.wait_for(
                self.governor.issue_directive(
                    agent_type=AgentType.MEMORY_MANAGER,
                    agent_id=f"memory_manager:{self.conversation_id}",
                    directive_type=DirectiveType.ACTION,
                    directive_data={
                        "instruction": (
                            "Maintain entity memories and ensure proper consolidation."
                        ),
                        "scope": "global",
                    },
                    priority=DirectivePriority.MEDIUM,
                    duration_minutes=24 * 60,
                ),
                timeout=10.0,
            )

            # ----- Success -------------------------------------------------
            log("✅ initialise() finished successfully")
            self._track_state_change("initialization", {"status": "success"})
            return self

        # ------------------------------------------------------------
        # ❸  Error handling & telemetry
        # ------------------------------------------------------------
        except asyncio.TimeoutError as te:
            log(f"⏰ TIMEOUT → {te}")
            self._track_error("initialization", "timeout")
            raise

        except Exception as exc:
            log(f"❌ FAILED → {exc}")
            self._track_error("initialization", str(exc))
            raise
    
    async def add_memory(self, npc_id: int, memory_text: str, 
                       memory_type: str = "interaction",
                       significance: int = 3,
                       emotional_valence: int = 0,
                       tags: List[str] = None) -> bool:
        """
        Add a memory to an NPC using the integrated system.
        
        Args:
            npc_id: ID of the NPC
            memory_text: Text of the memory
            memory_type: Type of memory
            significance: Significance level (1-10)
            emotional_valence: Emotional impact (-10 to +10)
            tags: List of tags for the memory
            
        Returns:
            True if successful, False otherwise
        """
        return await self.npc_system.add_memory_to_npc(
            npc_id, memory_text, memory_type, significance, emotional_valence, tags
        )
    
    async def retrieve_memories(self, npc_id: int, context: str = None, 
                              tags: List[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a context.
        
        Args:
            npc_id: ID of the NPC
            context: Context to retrieve memories for
            tags: List of tags to filter by
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory objects
        """
        memories = await self.npc_system.retrieve_relevant_memories(
            npc_id, context, tags, limit
        )
        
        # Format the memories for consistent API
        return [
            {
                "text": memory.text if hasattr(memory, "text") else str(memory),
                "type": memory.type if hasattr(memory, "type") else "unknown",
                "significance": memory.significance if hasattr(memory, "significance") else 3,
                "emotional_valence": memory.emotional_valence if hasattr(memory, "emotional_valence") else 0,
                "tags": memory.tags if hasattr(memory, "tags") else []
            }
            for memory in memories
        ]
    
    async def propagate_memory(self, source_npc_id: int, memory_text: str,
                             memory_type: str = "emotional",
                             significance: int = 5,
                             emotional_valence: int = 0) -> bool:
        """
        Propagate a memory to related NPCs.
        
        Args:
            source_npc_id: ID of the source NPC
            memory_text: Text of the memory
            memory_type: Type of memory
            significance: Significance level (1-10)
            emotional_valence: Emotional impact (-10 to +10)
            
        Returns:
            True if successful, False otherwise
        """
        return await self.npc_system.propagate_significant_memory(
            source_npc_id, memory_text, memory_type, significance, emotional_valence
        )
    
    async def record_event(self, npc_id: int, event_description: str) -> bool:
        """
        Record a memory event for an NPC.
        
        Args:
            npc_id: ID of the NPC
            event_description: Description of the event
            
        Returns:
            True if successful
        """
        return await self.npc_system.record_memory_event(npc_id, event_description)
    
    async def generate_flashback(self, npc_id: int, current_context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC.
        
        Args:
            npc_id: ID of the NPC
            current_context: Current context that may trigger a flashback
            
        Returns:
            Flashback data or None if no flashback was generated
        """
        return await self.npc_system.generate_flashback(npc_id, current_context)
    
    async def get_shared_memory(self, npc_id: int, target: str = "player", 
                              target_name: str = "Chase", rel_type: str = "related") -> str:
        """
        Generate a shared memory between an NPC and another entity.
        
        Args:
            npc_id: ID of the NPC
            target: Target type
            target_name: Target name
            rel_type: Relationship type
            
        Returns:
            Generated memory text
        """
        return await self.npc_system.generate_shared_memory(npc_id, target, target_name, rel_type)
    
    async def create_group_memory(self, npc_ids: List[int], event_description: str, 
                                significance: int = 5) -> bool:
        """
        Create a shared memory for a group of NPCs.
        
        Args:
            npc_ids: List of NPC IDs
            event_description: Description of the event
            significance: Significance level (1-10)
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        for npc_id in npc_ids:
            result = await self.add_memory(
                npc_id, 
                event_description, 
                memory_type="shared_experience",
                significance=significance,
                tags=["group_memory"]
            )
            
            if not result:
                success = False
        
        # If successful for at least one NPC, propagate to other NPCs
        if success and len(npc_ids) > 0:
            await self.propagate_memory(
                npc_ids[0],
                f"Group experience: {event_description}",
                memory_type="shared_experience",
                significance=significance
            )
        
        return success
    
    async def get_memory_summary(self, npc_id: int) -> Dict[str, Any]:
        """
        Get a summary of an NPC's memories.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with memory summary
        """
        memories = await self.retrieve_memories(npc_id, limit=50)
        
        # Extract memory stats
        total_memories = len(memories)
        significant_memories = sum(1 for m in memories if m.get("significance", 0) >= 7)
        emotional_memories = sum(1 for m in memories if abs(m.get("emotional_valence", 0)) >= 7)
        
        # Get most significant memories
        significant_memories_list = sorted(
            [m for m in memories if m.get("significance", 0) >= 5],
            key=lambda x: x.get("significance", 0),
            reverse=True
        )[:5]
        
        # Group memories by type
        memory_types = {}
        for memory in memories:
            memory_type = memory.get("type", "unknown")
            if memory_type not in memory_types:
                memory_types[memory_type] = 0
            memory_types[memory_type] += 1
        
        return {
            "npc_id": npc_id,
            "total_memories": total_memories,
            "significant_memories_count": significant_memories,
            "emotional_memories_count": emotional_memories,
            "most_significant": significant_memories_list,
            "memory_types": memory_types
        }
    
    async def fade_old_memories(self, npc_id: int, days_old: int = 14, 
                              significance_threshold: int = 3) -> int:
        """
        Fade old, less significant memories for an NPC.
        
        Args:
            npc_id: ID of the NPC
            days_old: Age threshold in days
            significance_threshold: Significance threshold
            
        Returns:
            Number of memories faded
        """
        # Get the memory functions from the NPC system
        from logic.npc_agents.memory_manager import EnhancedMemoryManager
        
        memory_manager = EnhancedMemoryManager(npc_id, self.user_id, self.conversation_id)
        count = await memory_manager.prune_old_memories(
            age_days=days_old,
            significance_threshold=significance_threshold,
            intensity_threshold=15
        )
        
        return count
    
    async def record_player_journal_entry(self, entry_type: str, entry_text: str, 
                                        fantasy_flag: bool = False, 
                                        intensity_level: int = 0) -> int:
        """
        Record an entry in the player's journal.
        
        Args:
            entry_type: Type of entry
            entry_text: Text of the entry
            fantasy_flag: Whether the entry is fantasy
            intensity_level: Intensity level
            
        Returns:
            ID of the created entry
        """
        try:
            async with get_db_connection_context() as conn:
                # Insert the journal entry and return the ID
                row = await conn.fetchrow("""
                    INSERT INTO PlayerJournal (
                        user_id, conversation_id, entry_type, entry_text, 
                        fantasy_flag, intensity_level, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    RETURNING id
                """, 
                self.user_id, self.conversation_id, entry_type, entry_text,
                fantasy_flag, intensity_level)
                
                if row:
                    return row['id']
                return 0
                
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logging.error(f"DB Error recording player journal entry: {db_err}", exc_info=True)
            return 0
        except Exception as e:
            logging.error(f"Error recording player journal entry: {e}", exc_info=True)
            return 0
    
    async def record_dream_sequence(self, dream_text: str, 
                                  intensity_level: int = 3) -> int:
        """
        Record a dream sequence in the player's journal.
        
        Args:
            dream_text: Text of the dream
            intensity_level: Intensity level
            
        Returns:
            ID of the created entry
        """
        return await self.record_player_journal_entry(
            "dream_sequence", dream_text, fantasy_flag=True, intensity_level=intensity_level
        )
    
    async def record_moment_of_clarity(self, clarity_text: str, 
                                     intensity_level: int = 2) -> int:
        """
        Record a moment of clarity in the player's journal.
        
        Args:
            clarity_text: Text of the clarity moment
            intensity_level: Intensity level
            
        Returns:
            ID of the created entry
        """
        return await self.record_player_journal_entry(
            "moment_of_clarity", clarity_text, fantasy_flag=False, intensity_level=intensity_level
        )
