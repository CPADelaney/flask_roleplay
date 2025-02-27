# logic.enhanced_memory.py

import logging
import json
import random
from datetime import datetime, timedelta
from db.connection import get_db_connection

class MemoryType:
    INTERACTION = "interaction"  # Direct interaction between player and NPC
    OBSERVATION = "observation"  # NPC observes player doing something
    EMOTIONAL = "emotional"      # Emotionally significant event
    TRAUMATIC = "traumatic"      # Highly negative event
    INTIMATE = "intimate"        # Deeply personal or sexual event
    
class MemorySignificance:
    LOW = 1      # Routine interaction
    MEDIUM = 3   # Notable but not remarkable
    HIGH = 5     # Important, remembered clearly
    CRITICAL = 10 # Life-changing, unforgettable

class EnhancedMemory:
    """
    Enhanced memory class that tracks emotional impact, decay rates,
    and determines when memories should be recalled.
    """
    def __init__(self, text, memory_type=MemoryType.INTERACTION, significance=MemorySignificance.MEDIUM):
        self.text = text
        self.timestamp = datetime.now().isoformat()
        self.memory_type = memory_type
        self.significance = significance
        self.recall_count = 0
        self.last_recalled = None
        self.emotional_valence = 0  # -10 to +10, negative to positive emotion
        self.tags = []
        
    def to_dict(self):
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "significance": self.significance,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data):
        memory = cls(data["text"], data["memory_type"], data["significance"])
        memory.timestamp = data["timestamp"]
        memory.recall_count = data["recall_count"]
        memory.last_recalled = data["last_recalled"]
        memory.emotional_valence = data["emotional_valence"]
        memory.tags = data["tags"]
        return memory

class MemoryManager:
    """
    Manages creation, storage, retrieval, and sharing of memories between NPCs.
    Implements advanced features like emotional weighting, memory decay, and
    contextual recall.
    """
    
    @staticmethod
    async def add_memory(user_id, conversation_id, entity_id, entity_type, 
                        memory_text, memory_type=MemoryType.INTERACTION, 
                        significance=MemorySignificance.MEDIUM, 
                        emotional_valence=0, tags=None):
        """Add a new memory to an entity (NPC or player)"""
        tags = tags or []
        
        # Create the memory object
        memory = EnhancedMemory(memory_text, memory_type, significance)
        memory.emotional_valence = emotional_valence
        memory.tags = tags
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current memories
            if entity_type == "npc":
                cursor.execute("""
                    SELECT memory FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    SELECT memories FROM PlayerStats
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (user_id, conversation_id, entity_id))
                
            row = cursor.fetchone()
            
            memories = []
            if row and row[0]:
                if isinstance(row[0], str):
                    try:
                        memories = json.loads(row[0])
                    except json.JSONDecodeError:
                        memories = []
                else:
                    memories = row[0]
            
            # Add new memory
            memories.append(memory.to_dict())
            
            # Update the database
            if entity_type == "npc":
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(memories), user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    UPDATE PlayerStats
                    SET memories = %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (json.dumps(memories), user_id, conversation_id, entity_id))
                
            conn.commit()
            
            # Determine if this memory should propagate to other NPCs
            if memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC] and significance >= MemorySignificance.HIGH:
                await MemoryManager.propagate_significant_memory(user_id, conversation_id, entity_id, entity_type, memory)
                
            return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding memory: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def propagate_significant_memory(user_id, conversation_id, source_entity_id, source_entity_type, memory):
        """Propagate significant memories to related NPCs based on social links"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Find strong social links
            cursor.execute("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s AND link_level >= 50
                AND ((entity1_type=%s AND entity1_id=%s) OR (entity2_type=%s AND entity2_id=%s))
            """, (user_id, conversation_id, 
                 source_entity_type, source_entity_id, 
                 source_entity_type, source_entity_id))
                
            links = cursor.fetchall()
            
            for link in links:
                e1_type, e1_id, e2_type, e2_id, link_type, link_level = link
                
                # Determine the target entity
                if e1_type == source_entity_type and e1_id == source_entity_id:
                    target_type = e2_type
                    target_id = e2_id
                else:
                    target_type = e1_type
                    target_id = e1_id
                
                # Skip if target is not an NPC (don't propagate to player)
                if target_type != "npc":
                    continue
                
                # Modify the memory for the target's perspective
                # This creates a "heard about" memory rather than direct experience
                target_memory = EnhancedMemory(
                    f"I heard that {memory.text}",
                    memory_type="observation",
                    significance=max(MemorySignificance.LOW, memory.significance - 2)
                )
                target_memory.emotional_valence = memory.emotional_valence * 0.7  # Reduced emotional impact
                target_memory.tags = memory.tags + ["secondhand"]
                
                # Add the modified memory to the target
                await MemoryManager.add_memory(
                    user_id, conversation_id, 
                    target_id, target_type,
                    target_memory.text,
                    target_memory.memory_type,
                    target_memory.significance,
                    target_memory.emotional_valence,
                    target_memory.tags
                )
                
            return True
        except Exception as e:
            logging.error(f"Error propagating memory: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def retrieve_relevant_memories(user_id, conversation_id, entity_id, entity_type, 
                                       context=None, tags=None, limit=5):
        """
        Retrieve memories relevant to the given context or tags.
        Applies weighting based on significance, recency, and emotional impact.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get all memories
            if entity_type == "npc":
                cursor.execute("""
                    SELECT memory FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    SELECT memories FROM PlayerStats
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (user_id, conversation_id, entity_id))
                
            row = cursor.fetchone()
            
            if not row or not row[0]:
                return []
                
            memories = []
            if isinstance(row[0], str):
                try:
                    memories = json.loads(row[0])
                except json.JSONDecodeError:
                    memories = []
            else:
                memories = row[0]
            
            # Convert to EnhancedMemory objects
            memory_objects = [EnhancedMemory.from_dict(m) for m in memories]
            
            # Filter by tags if provided
            if tags:
                memory_objects = [m for m in memory_objects if any(tag in m.tags for tag in tags)]
            
            # Score memories based on relevance
            scored_memories = []
            for memory in memory_objects:
                score = memory.significance  # Base score is significance
                
                # Recency bonus
                try:
                    memory_date = datetime.fromisoformat(memory.timestamp)
                    days_old = (datetime.now() - memory_date).days
                    recency_score = max(0, 10 - days_old/30)  # Higher score for more recent memories
                    score += recency_score
                except (ValueError, TypeError):
                    pass
                
                # Context relevance if context is provided
                if context:
                    context_words = context.lower().split()
                    memory_words = memory.text.lower().split()
                    common_words = set(context_words) & set(memory_words)
                    context_score = len(common_words) * 0.5
                    score += context_score
                
                # Emotional impact bonus
                emotion_score = abs(memory.emotional_valence) * 0.3
                score += emotion_score
                
                # Penalize frequently recalled memories slightly to ensure variety
                recall_penalty = min(memory.recall_count * 0.2, 2)
                score -= recall_penalty
                
                scored_memories.append((memory, score))
            
            # Sort by score and take top 'limit'
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [m[0] for m in scored_memories[:limit]]
            
            # Update recall count for selected memories
            for memory in top_memories:
                memory.recall_count += 1
                memory.last_recalled = datetime.now().isoformat()
            
            # Update the stored memories
            updated_memories = []
            for memory in memory_objects:
                # Check if this memory is in top_memories
                matching_memory = next((m for m in top_memories if m.timestamp == memory.timestamp), None)
                if matching_memory:
                    updated_memories.append(matching_memory.to_dict())
                else:
                    updated_memories.append(memory.to_dict())
            
            # Save back to database
            if entity_type == "npc":
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(updated_memories), user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    UPDATE PlayerStats
                    SET memories = %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (json.dumps(updated_memories), user_id, conversation_id, entity_id))
                
            conn.commit()
            
            return top_memories
        except Exception as e:
            conn.rollback()
            logging.error(f"Error retrieving memories: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
            
    @staticmethod
    async def generate_flashback(user_id, conversation_id, npc_id, current_context):
        """
        Generate a flashback moment for an NPC to reference a significant past memory
        that relates to the current context.
        """
        # First, retrieve relevant memories
        memories = await MemoryManager.retrieve_relevant_memories(
            user_id, conversation_id, npc_id, "npc", 
            context=current_context, limit=3
        )
        
        if not memories:
            return None
            
        # Select a memory, favoring emotional or traumatic ones
        emotional_memories = [m for m in memories if m.memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC]]
        selected_memory = random.choice(emotional_memories) if emotional_memories else random.choice(memories)
        
        # Get the NPC's name
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        row = cursor.fetchone()
        conn.close()
        
        npc_name = row[0] if row else "the NPC"
        
        # Format the flashback
        flashback_text = f"{npc_name}'s expression shifts momentarily, a distant look crossing their face. \"This reminds me of {selected_memory.text}\""
        
        if selected_memory.emotional_valence < -5:
            flashback_text += " A shadow crosses their face at the memory."
        elif selected_memory.emotional_valence > 5:
            flashback_text += " Their eyes light up at the pleasant memory."
            
        return {
            "type": "flashback",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "text": flashback_text,
            "memory": selected_memory.text
        }
