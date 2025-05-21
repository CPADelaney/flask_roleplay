# memory/reconsolidation.py

import logging
import random
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import openai

from .connection import with_transaction
from .core import Memory, MemoryType, MemorySignificance

logger = logging.getLogger("memory_reconsolidation")

class ReconsolidationManager:
    """
    Advanced memory reconsolidation that simulates how human memories 
    change each time they're recalled.
    
    Features:
    - Emotional state influences memory alterations
    - More frequent recalls can either strengthen or distort memories
    - GPT-based alterations for natural language changes
    - Memory schemas influence how memories are altered
    - Source confusion between similar memories
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def reconsolidate_memory(self,
                                 memory_id: int,
                                 entity_type: str,
                                 entity_id: int,
                                 emotional_context: Dict[str, Any] = None,
                                 recall_context: str = None,
                                 alteration_strength: float = 0.1,
                                 conn = None) -> Dict[str, Any]:
        """
        Reconsolidate a memory with advanced human-like distortion effects.
        
        Args:
            memory_id: ID of the memory to reconsolidate
            entity_type: Type of entity that owns the memory
            entity_id: ID of the entity
            emotional_context: Current emotional state of the entity
            recall_context: Context in which the memory was recalled
            alteration_strength: How much to alter the memory (0.0-1.0)
            
        Returns:
            Updated memory information
        """
        # Fetch the memory
        row = await conn.fetchrow("""
            SELECT id, memory_text, memory_type, significance, emotional_intensity,
                   tags, metadata, timestamp, times_recalled, last_recalled
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Memory {memory_id} not found"}
            
        # Create memory object from row
        memory = Memory.from_db_row(dict(row))
        
        # Skip reconsolidation for certain memory types or high significance
        if memory.memory_type in [MemoryType.SEMANTIC, MemoryType.CONSOLIDATED]:
            return {"message": "Semantic or consolidated memories are not reconsolidated", "memory_id": memory_id}
            
        if memory.significance >= MemorySignificance.CRITICAL:
            return {"message": "Critical memories are not reconsolidated", "memory_id": memory_id}
        
        # Initialize metadata properly
        if not memory.metadata:
            memory.metadata = {}
            
        # Get original form if available, otherwise use current text
        original_form = memory.metadata.get("original_form", memory.text)
        
        # Save original form if not already saved
        if "original_form" not in memory.metadata:
            memory.metadata["original_form"] = original_form
            
        # Initialize reconsolidation history if needed
        if "reconsolidation_history" not in memory.metadata:
            memory.metadata["reconsolidation_history"] = []
            
        # Add current version to history before altering
        memory.metadata["reconsolidation_history"].append({
            "previous_text": memory.text,
            "timestamp": datetime.now().isoformat(),
            "emotional_context": emotional_context,
            "recall_context": recall_context
        })
        
        # Keep only last 3 versions to avoid metadata bloat
        if len(memory.metadata["reconsolidation_history"]) > 3:
            memory.metadata["reconsolidation_history"] = memory.metadata["reconsolidation_history"][-3:]
        
        # Get memory schemas for this entity (patterns that influence memory)
        schemas = await self._get_entity_memory_schemas(entity_type, entity_id, conn)
        
        # Check for source confusion with similar memories
        similar_memories = await self._find_similar_memories(
            memory_id, entity_type, entity_id, memory.text, conn
        )
        
        # Adjust alteration strength based on factors
        times_recalled = memory.times_recalled or 0
        
        # Memories recalled many times become more rigid in their altered form
        if times_recalled > 5:
            if "fidelity" in memory.metadata and memory.metadata["fidelity"] < 0.8:
                # Already distorted and frequently recalled - becomes more rigid
                alteration_strength *= 0.5
        else:
            # Less recalled memories are more susceptible to change
            alteration_strength *= min(1.5, max(0.5, 1.0 + (3 - times_recalled) * 0.1))
            
        # Emotional intensity influences alterations
        if memory.emotional_intensity > 70:
            # High emotional intensity can lead to exaggeration
            alteration_strength *= 1.2
            
        # Memory age affects reconsolidation
        age_days = (datetime.now() - memory.timestamp).days if memory.timestamp else 0
        if age_days > 30:
            # Older memories are more susceptible to change
            alteration_strength *= min(2.0, 1.0 + (age_days / 100))
            
        # Apply emotional context influence
        if emotional_context:
            current_emotion = emotional_context.get("primary_emotion", "neutral")
            intensity = emotional_context.get("intensity", 0.5)
            
            # Current emotions can color memory reconsolidation
            if current_emotion in ["anger", "fear", "sadness"]:
                # Negative emotions can enhance negative aspects
                alteration_strength *= 1.0 + (intensity * 0.3)
        
        # Generate the altered text
        altered_text = await self._alter_memory_text(
            memory.text,
            original_form,
            alteration_strength,
            emotional_context,
            schemas,
            similar_memories
        )
        
        # Update memory fidelity
        current_fidelity = memory.metadata.get("fidelity", 1.0)
        fidelity_change = -0.05 * alteration_strength
        
        # Frequently recalled memories can either strengthen or weaken
        if times_recalled > 3:
            # 70% chance of strengthening if recalled often
            if random.random() < 0.7:
                fidelity_change = abs(fidelity_change) * 0.5
        
        new_fidelity = max(0.1, min(1.0, current_fidelity + fidelity_change))
        memory.metadata["fidelity"] = new_fidelity
        
        # Generate embedding for the altered text
        embedding = None
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=altered_text
            )
            embedding = response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding for reconsolidated memory: {e}")
        
        # Update the memory in the database
        await conn.execute("""
            UPDATE unified_memories
            SET memory_text = $1,
                metadata = $2,
                embedding = COALESCE($3, embedding)
            WHERE id = $4
        """, altered_text, json.dumps(memory.metadata), embedding, memory_id)
        
        # Return the updated memory info
        return {
            "memory_id": memory_id,
            "original_text": memory.text,
            "reconsolidated_text": altered_text,
            "fidelity": new_fidelity,
            "alteration_strength": alteration_strength,
            "source_confusion": len(similar_memories) > 0
        }
    
    async def _get_entity_memory_schemas(self, entity_type: str, entity_id: int, conn) -> List[Dict[str, Any]]:
        """
        Get memory schemas for an entity.
        Schemas are patterns that influence how memories are altered.
        """
        # Try to get from MemorySchemas table
        rows = await conn.fetch("""
            SELECT schema_name, pattern, influence_strength
            FROM MemorySchemas
            WHERE user_id = $1 
              AND conversation_id = $2
              AND entity_type = $3
              AND entity_id = $4
        """, self.user_id, self.conversation_id, entity_type, entity_id)
        
        if rows:
            return [dict(row) for row in rows]
            
        # If no schemas found, create default ones based on entity type
        default_schemas = []
        
        if entity_type == "npc":
            # Get NPC traits
            npc_row = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, self.user_id, self.conversation_id, entity_id)
            
            if npc_row:
                dominance = npc_row["dominance"]
                cruelty = npc_row["cruelty"]
                
                # High dominance NPCs tend to remember themselves as more in control
                if dominance > 60:
                    default_schemas.append({
                        "schema_name": "control_bias",
                        "pattern": "remember being in control",
                        "influence_strength": (dominance - 60) / 40
                    })
                
                # Cruel NPCs tend to downplay others' suffering
                if cruelty > 60:
                    default_schemas.append({
                        "schema_name": "cruelty_bias",
                        "pattern": "downplay others' pain",
                        "influence_strength": (cruelty - 60) / 40
                    })
        
        elif entity_type == "player":
            # Player might have self-serving bias
            default_schemas.append({
                "schema_name": "self_serving_bias",
                "pattern": "remember successes as personal skill, failures as external circumstances",
                "influence_strength": 0.3
            })
        
        # Add universal schemas
        default_schemas.append({
            "schema_name": "positivity_bias",
            "pattern": "remember positive emotions more vividly than negative ones",
            "influence_strength": 0.2
        })
        
        default_schemas.append({
            "schema_name": "consistency_bias",
            "pattern": "alter memories to be consistent with current beliefs",
            "influence_strength": 0.3
        })
        
        return default_schemas
    
    async def _find_similar_memories(self, 
                                    memory_id: int, 
                                    entity_type: str, 
                                    entity_id: int, 
                                    memory_text: str,
                                    conn) -> List[Dict[str, Any]]:
        """
        Find memories similar to the given one that could cause source confusion.
        """
        # Generate embedding for comparison
        embedding = None
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=memory_text
            )
            embedding = response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding for similar memory search: {e}")
            # If embedding fails, try keyword search as fallback
            if embedding is None:
                # Extract keywords from memory text
                words = set(memory_text.lower().split())
                significant_words = [w for w in words if len(w) > 4]
                
                if significant_words:
                    # Use OR query with the significant words
                    conditions = " OR ".join([f"memory_text ILIKE '%' || $" + str(i+5) + " || '%'" for i, _ in enumerate(significant_words[:5])])
                    params = [memory_id, entity_type, entity_id, self.user_id, self.conversation_id] + significant_words[:5]
                    
                    query = f"""
                        SELECT id, memory_text, significance, emotional_intensity
                        FROM unified_memories
                        WHERE id != $1
                          AND entity_type = $2
                          AND entity_id = $3
                          AND user_id = $4
                          AND conversation_id = $5
                          AND ({conditions})
                        LIMIT 3
                    """
                    
                    rows = await conn.fetch(query, *params)
                    return [dict(row) for row in rows]
                
                return []
        
        # If we have an embedding, use vector search
        if embedding:
            rows = await conn.fetch("""
                SELECT id, memory_text, significance, emotional_intensity,
                       embedding <-> $1 AS similarity
                FROM unified_memories
                WHERE id != $2
                  AND entity_type = $3
                  AND entity_id = $4
                  AND user_id = $5
                  AND conversation_id = $6
                ORDER BY similarity
                LIMIT 3
            """, embedding, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
            
            similar_memories = []
            for row in rows:
                # Only include if similarity is high enough
                if row["similarity"] < 0.25:  # Lower is more similar for PostgreSQL vector operators
                    similar_memories.append(dict(row))
            
            return similar_memories
        
        return []
    
    async def _alter_memory_text(self,
                               memory_text: str,
                               original_form: str,
                               alteration_strength: float,
                               emotional_context: Optional[Dict[str, Any]],
                               schemas: List[Dict[str, Any]],
                               similar_memories: List[Dict[str, Any]]) -> str:
        """
        Alter memory text based on various factors.
        Uses GPT for more natural alterations.
        """
        # For minimal changes, use simple word-level alterations
        if alteration_strength < 0.1 and random.random() < 0.7:
            return self._simple_text_alteration(memory_text, alteration_strength)
        
        # For more significant changes, use GPT
        try:
            # Format schemas for prompt
            schema_text = ""
            if schemas:
                schema_text = "Memory schemas (cognitive biases) affecting this memory:\n"
                for schema in schemas:
                    schema_text += f"- {schema['schema_name']}: {schema['pattern']} (strength: {schema['influence_strength']:.2f})\n"
            
            # Format similar memories for source confusion
            confusion_text = ""
            if similar_memories:
                confusion_text = "Similar memories that may cause source confusion:\n"
                for i, mem in enumerate(similar_memories):
                    confusion_text += f"- Similar memory {i+1}: {mem['memory_text']}\n"
            
            # Format emotional context
            emotion_text = ""
            if emotional_context:
                emotion_text = f"Current emotional state: {emotional_context.get('primary_emotion', 'neutral')} (intensity: {emotional_context.get('intensity', 0.5):.2f})\n"
            
            # Create the prompt
            prompt = f"""
            Alter this memory to simulate human memory reconsolidation effects. The memory is being recalled and should change slightly.
            
            Original form of memory: {original_form}
            Current form of memory: {memory_text}
            Alteration strength (0.0-1.0): {alteration_strength:.2f}
            
            {emotion_text}
            {schema_text}
            {confusion_text}
            
            Instructions:
            1. Keep the same core meaning and key facts
            2. Make subtle changes to wording, emphasis, or details based on the alteration strength
            3. If schemas are present, subtly alter the memory to align with those biases
            4. If similar memories exist, incorporate small elements from them (source confusion)
            5. If there's emotional context, slightly color the memory with that emotion
            
            Return only the altered memory text with no explanation.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are simulating human memory reconsolidation effects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            altered_text = response.choices[0].message.content.strip()
            return altered_text
            
        except Exception as e:
            logger.error(f"Error in GPT-based memory alteration: {e}")
            # Fall back to simple text alteration
            return self._simple_text_alteration(memory_text, alteration_strength)
    
    def _simple_text_alteration(self, text: str, alteration_strength: float) -> str:
        """
        Perform simple word-level alterations when GPT is unavailable.
        """
        words = text.split()
        changes_made = 0
        max_changes = max(1, int(len(words) * alteration_strength * 0.5))
        
        for i in range(len(words)):
            if changes_made >= max_changes:
                break
                
            if len(words[i]) <= 3 or random.random() > alteration_strength:
                continue
                
            # Word alterations
            alterations = [
                lambda w: f"very {w}" if len(w) > 3 else w,
                lambda w: f"quite {w}" if len(w) > 3 else w,
                lambda w: f"{w}ly" if not w.endswith("ly") and len(w) > 4 else w,
                lambda w: w.replace("very ", "") if w.startswith("very ") else w,
                lambda w: w.replace("quite ", "") if w.startswith("quite ") else w,
                lambda w: w.replace("extremely ", "") if w.startswith("extremely ") else w,
                lambda w: w.replace("slightly ", "") if w.startswith("slightly ") else w,
                # Word substitutions for common adjectives
                lambda w: "good" if w == "nice" else w,
                lambda w: "great" if w == "good" else w,
                lambda w: "bad" if w == "poor" else w,
                lambda w: "unhappy" if w == "sad" else w,
                lambda w: "angry" if w == "upset" else w,
                # Intensity adjustments
                lambda w: "furious" if w == "angry" else w,
                lambda w: "upset" if w == "furious" else w,
                lambda w: "scared" if w == "afraid" else w,
                lambda w: "terrified" if w == "scared" else w
            ]
            
            # Choose a random alteration
            alteration = random.choice(alterations)
            new_word = alteration(words[i])
            
            if new_word != words[i]:
                words[i] = new_word
                changes_made += 1
        
        return " ".join(words)
    
    @with_transaction
    async def check_memories_for_reconsolidation(self,
                                              entity_type: str,
                                              entity_id: int,
                                              max_memories: int = 5,
                                              conn = None) -> List[int]:
        """
        Find memories that are due for reconsolidation.
        Used for background maintenance.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            max_memories: Maximum number of memories to reconsolidate
            
        Returns:
            List of memory IDs that were reconsolidated
        """
        # Find candidate memories for reconsolidation
        rows = await conn.fetch("""
            SELECT id, significance, times_recalled,
                   EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400 AS days_old,
                   EXTRACT(EPOCH FROM (NOW() - COALESCE(last_recalled, timestamp))) / 86400 AS days_since_recall
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND memory_type NOT IN ('semantic', 'consolidated')
              AND significance < $5
              AND is_archived = FALSE
            ORDER BY RANDOM()
            LIMIT $6
        """, entity_type, entity_id, self.user_id, self.conversation_id, 
            MemorySignificance.CRITICAL, max_memories * 2)
        
        # Filter candidates based on reconsolidation criteria
        candidates = []
        for row in rows:
            memory_id = row["id"]
            days_old = row["days_old"]
            days_since_recall = row["days_since_recall"]
            times_recalled = row["times_recalled"]
            
            # Different reconsolidation strategies based on memory properties
            score = 0
            
            # Older memories are more likely to be reconsolidated
            score += min(10, days_old / 10)
            
            # Memories recalled long ago are due for reconsolidation
            if days_since_recall > 7:
                score += 5
                
            # Memories recalled frequently change more
            if times_recalled > 2:
                score += times_recalled * 0.5
            
            # Add some randomness
            score += random.random() * 3
            
            candidates.append((memory_id, score))
        
        # Sort by score and pick top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c[0] for c in candidates[:max_memories]]
        
        # Reconsolidate these memories
        reconsolidated = []
        for memory_id in top_candidates:
            try:
                # Use lower alteration strength for background reconsolidation
                result = await self.reconsolidate_memory(
                    memory_id=memory_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    alteration_strength=0.05 + (random.random() * 0.1),
                    conn=conn
                )
                
                if "error" not in result:
                    reconsolidated.append(memory_id)
            except Exception as e:
                logger.error(f"Error in background reconsolidation for memory {memory_id}: {e}")
        
        return reconsolidated

# Create the necessary tables if they don't exist
async def create_reconsolidation_tables():
    """Create the necessary tables for the reconsolidation system if they don't exist."""
    from db.connection import get_db_connection_context
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS MemorySchemas (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                schema_name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                influence_strength FLOAT NOT NULL DEFAULT 0.5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_memory_schemas_entity 
            ON MemorySchemas(user_id, conversation_id, entity_type, entity_id);
        """)
