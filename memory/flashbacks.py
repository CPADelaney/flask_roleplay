# memory/flashbacks.py

import random
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .connection import with_transaction
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_flashbacks")

class FlashbackManager:
    """
    Manages memory flashbacks for NPCs and players.
    Simulates how significant memories can be triggered by context.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def generate_flashback(self, 
                               entity_type: str,
                               entity_id: int, 
                               current_context: str,
                               emotional_intensity: int = None,
                               conn = None) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback moment for an entity based on the current context.
        
        Args:
            entity_type: Type of entity (npc, player)
            entity_id: ID of the entity
            current_context: The current context that might trigger a flashback
            emotional_intensity: Optional filter for emotional intensity
            
        Returns:
            A flashback object with the memory and presentation text
        """
        # Create memory manager for this entity
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Retrieve candidate memories that might become flashbacks
        query_params = {
            "query": current_context,
            "memory_types": ["observation", "emotional"],
            "limit": 5,
            "min_significance": MemorySignificance.MEDIUM,
            "conn": conn
        }
        
        # Filter by emotional_intensity if provided
        if emotional_intensity is not None:
            query_params["tags"] = ["emotional"]
        
        memories = await memory_manager.retrieve_memories(**query_params)
        
        if not memories:
            return None
            
        # For flashbacks, prefer emotional or traumatic memories
        emotional_memories = [m for m in memories if 
                             "emotional" in m.tags or 
                             "traumatic" in m.tags or 
                             m.emotional_intensity > 50]
        
        selected_memory = random.choice(emotional_memories) if emotional_memories else random.choice(memories)
        
        # Get entity name
        entity_name = ""
        if entity_type == "npc":
            row = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, self.user_id, self.conversation_id, entity_id)
            entity_name = row[0] if row else "the NPC"
        else:
            entity_name = "Chase"  # Default player name
        
        # Format the flashback
        flashback_text = self._format_flashback_text(entity_name, selected_memory)
            
        # Record that this memory was recalled as a flashback
        await memory_manager.reconsolidate_memory(
            memory_id=selected_memory.id,
            alteration_strength=0.05,  # Slight alteration during flashback
            conn=conn
        )
        
        # Return the flashback
        return {
            "type": "flashback",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "text": flashback_text,
            "memory": selected_memory.text,
            "memory_id": selected_memory.id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_flashback_text(self, entity_name: str, memory: Memory) -> str:
        """
        Format a memory into a narrative flashback text.
        Varies the presentation based on emotional intensity.
        """
        intensity = memory.emotional_intensity
        transitions = [
            f"{entity_name}'s mind flashes to a memory: ",
            f"{entity_name} suddenly recalls: ",
            f"A memory surfaces in {entity_name}'s mind: ",
            f"{entity_name}'s expression shifts as they remember: ",
            f"For a brief moment, {entity_name} is lost in a memory: "
        ]
        
        transition = random.choice(transitions)
        flashback_text = f"{transition}\"{memory.text}\""
        
        # Add emotional reaction based on intensity
        if intensity > 70:
            intense_reactions = [
                f" {entity_name}'s hands begin to tremble at the vivid recollection.",
                f" The memory hits {entity_name} like a physical blow.",
                f" {entity_name}'s breath catches as the powerful memory resurfaces."
            ]
            flashback_text += random.choice(intense_reactions)
        elif intensity > 40:
            moderate_reactions = [
                f" A shadow crosses {entity_name}'s face at the memory.",
                f" {entity_name}'s eyes unfocus briefly as they process the memory.",
                f" The memory clearly affects {entity_name}'s demeanor."
            ]
            flashback_text += random.choice(moderate_reactions)
        else:
            mild_reactions = [
                f" {entity_name} blinks as the memory passes through their mind.",
                f" The memory brings a subtle change to {entity_name}'s expression.",
                f" {entity_name} pauses momentarily as the memory surfaces."
            ]
            flashback_text += random.choice(mild_reactions)
        
        return flashback_text
    
    @with_transaction
    async def check_for_triggered_flashback(self, 
                                         entity_type: str,
                                         entity_id: int,
                                         trigger_words: List[str],
                                         chance: float = 0.3,
                                         conn = None) -> Optional[Dict[str, Any]]:
        """
        Check if a flashback should be triggered based on trigger words.
        
        Args:
            entity_type: Type of entity (npc, player)
            entity_id: ID of the entity
            trigger_words: List of words that might trigger a flashback
            chance: Base chance of a flashback occurring (0.0-1.0)
            
        Returns:
            A flashback object if triggered, None otherwise
        """
        # Check if a flashback should occur
        if random.random() > chance:
            return None
            
        # Create context from trigger words
        context = " ".join(trigger_words)
        
        # Generate flashback
        return await self.generate_flashback(
            entity_type=entity_type,
            entity_id=entity_id,
            current_context=context,
            conn=conn
        )
    
    @with_transaction
    async def record_flashback_reaction(self,
                                      flashback_id: str,
                                      entity_type: str,
                                      entity_id: int,
                                      reaction_text: str,
                                      reaction_type: str = "neutral",
                                      conn = None) -> bool:
        """
        Record how an entity reacted to a flashback.
        This helps model how flashbacks affect behavior.
        
        Args:
            flashback_id: Unique ID of the flashback
            entity_type: Type of entity (npc, player)
            entity_id: ID of the entity
            reaction_text: Description of the reaction
            reaction_type: Type of reaction (positive, negative, neutral)
            
        Returns:
            Success status
        """
        # Store in FlashbackHistory table
        await conn.execute("""
            INSERT INTO FlashbackHistory (
                user_id, conversation_id, entity_type, entity_id, 
                flashback_id, reaction_text, reaction_type, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
        """, self.user_id, self.conversation_id, entity_type, entity_id,
            flashback_id, reaction_text, reaction_type)
        
        # Create memory of this reaction
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        significance = {
            "positive": MemorySignificance.MEDIUM,
            "negative": MemorySignificance.HIGH,
            "neutral": MemorySignificance.MEDIUM
        }.get(reaction_type, MemorySignificance.MEDIUM)
        
        memory_text = f"I reacted to a memory flashback: {reaction_text}"
        
        await memory_manager.add_memory(
            Memory(
                text=memory_text,
                memory_type=MemoryType.REFLECTION,
                significance=significance,
                emotional_intensity=70 if reaction_type == "negative" else 40,
                tags=["flashback_reaction", reaction_type],
                metadata={
                    "flashback_id": flashback_id,
                    "reaction_type": reaction_type
                },
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        return True

# Create the necessary table if it doesn't exist
async def create_flashback_tables():
    """Create the necessary tables for the flashback system if they don't exist."""
    from .connection import TransactionContext
    
    async with TransactionContext() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS FlashbackHistory (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                flashback_id TEXT NOT NULL,
                reaction_text TEXT,
                reaction_type TEXT,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_flashback_entity 
            ON FlashbackHistory(user_id, conversation_id, entity_type, entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_flashback_timestamp 
            ON FlashbackHistory(timestamp);
        """)
