# logic/memory_manager.py - New module based on advice

import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

from db.connection import get_db_connection
import asyncpg
import openai  # Make sure openai is imported for embeddings

class EnhancedMemoryManager:
    """Manages memories for an individual NPC with advanced features"""
    
    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def add_memory(self, memory_text, memory_type="observation", 
                         significance=3, emotional_valence=0, tags=None):
        """Add memory with rich metadata and auto-tagging"""
        tags = tags or []
        
        # Extract additional tags from content analysis
        content_tags = await self.analyze_memory_content(memory_text)
        tags.extend(content_tags)
        
        # Calculate emotional intensity
        emotional_intensity = await self.calculate_emotional_intensity(
            memory_text, emotional_valence)
        
        # Store memory with embedding
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Generate embedding
            try:
                response = await openai.AsyncOpenAI().embeddings.create(
                    model="text-embedding-ada-002",
                    input=memory_text
                )
                embedding_data = response.data[0].embedding
            except Exception as e:
                logging.error(f"Error generating embedding: {e}")
                embedding_data = None
            
            # Insert memory
            memory_id = await conn.fetchval("""
                INSERT INTO NPCMemories (
                    npc_id, memory_text, memory_type, tags, 
                    emotional_intensity, significance, embedding,
                    associated_entities, is_consolidated
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, 
                self.npc_id, memory_text, memory_type, tags,
                emotional_intensity, significance, embedding_data,
                json.dumps({}), False
            )
            
            # Propagate significant memories to related NPCs
            if significance >= 4:
                await self.propagate_memory(memory_text, tags, significance, emotional_intensity)
            
            return memory_id
            
        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            return None
        finally:
            await conn.close()
            
    async def retrieve_relevant_memories(self, context, limit=5):
        """Get memories relevant to current context with enhanced recall"""
        query_text = context if isinstance(context, str) else context.get("description", "")
        
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Generate embedding for query
            try:
                response = await openai.AsyncOpenAI().embeddings.create(
                    model="text-embedding-ada-002",
                    input=query_text
                )
                query_vector = response.data[0].embedding
                
                # Vector search
                rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type, tags, 
                           emotional_intensity, significance,
                           times_recalled, timestamp
                    FROM NPCMemories
                    WHERE npc_id = $1
                    ORDER BY embedding <-> $2
                    LIMIT $3
                """, self.npc_id, query_vector, limit)
                
            except Exception as e:
                logging.error(f"Error generating embedding for query: {e}")
                
                # Fallback to simple keyword matching
                words = query_text.lower().split()
                query = " OR ".join([f"LOWER(memory_text) LIKE '%' || $" + str(i+2) + " || '%'" for i in range(len(words))])
                params = [self.npc_id] + words + [limit]
                
                rows = await conn.fetch(f"""
                    SELECT id, memory_text, memory_type, tags, 
                           emotional_intensity, significance,
                           times_recalled, timestamp
                    FROM NPCMemories
                    WHERE npc_id = $1
                    AND ({query})
                    ORDER BY timestamp DESC
                    LIMIT ${len(params)}
                """, *params)
            
            memories = []
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "memory_text": row["memory_text"],
                    "memory_type": row["memory_type"],
                    "tags": row["tags"] or [],
                    "emotional_intensity": row["emotional_intensity"],
                    "significance": row["significance"],
                    "times_recalled": row["times_recalled"],
                    "timestamp": row["timestamp"],
                    "relevance_score": 0  # Will be filled in with recency and emotional biases
                })
            
            # Apply recency bias
            memories = await self.apply_recency_bias(memories)
            
            # Apply emotional bias
            memories = await self.apply_emotional_bias(memories)
            
            # Update recall stats
            memory_ids = [m["id"] for m in memories]
            await self.update_memory_retrieval_stats(memory_ids)
            
            return memories
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {e}")
            return []
        finally:
            await conn.close()
    
    async def update_memory_retrieval_stats(self, memory_ids):
        """Update recall statistics for retrieved memories"""
        if not memory_ids:
            return
        
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Update times_recalled and last_recalled for all memories in one query
            await conn.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = CURRENT_TIMESTAMP
                WHERE id = ANY($1)
            """, memory_ids)
            
        except Exception as e:
            logging.error(f"Error updating memory retrieval stats: {e}")
        finally:
            await conn.close()
    
    async def analyze_memory_content(self, memory_text):
        """Extract meaningful tags from memory content"""
        tags = []
        
        # Simple keyword extraction
        if "angry" in memory_text.lower() or "upset" in memory_text.lower():
            tags.append("negative_emotion")
        if "happy" in memory_text.lower() or "pleased" in memory_text.lower():
            tags.append("positive_emotion")
        if "player" in memory_text.lower() or "chase" in memory_text.lower():
            tags.append("player_related")
            
        return tags
    
    async def calculate_emotional_intensity(self, memory_text, base_valence):
        """Calculate emotional intensity based on text and base valence"""
        # Simple implementation - could be enhanced with sentiment analysis
        intensity = abs(base_valence) * 10  # Scale from -10...+10 to 0...100
        
        # Boost intensity based on emotional keywords
        emotion_words = {
            "furious": 20, "ecstatic": 20, "devastated": 20, "thrilled": 20,
            "angry": 15, "delighted": 15, "sad": 15, "happy": 15,
            "annoyed": 10, "pleased": 10, "upset": 10, "glad": 10,
            "concerned": 5, "fine": 5, "worried": 5, "okay": 5
        }
        
        for word, boost in emotion_words.items():
            if word in memory_text.lower():
                intensity += boost
                break  # Only apply the highest matching boost
                
        return min(100, intensity)  # Cap at 100
    
    async def apply_recency_bias(self, memories):
        """Boost relevance of recent memories"""
        for memory in memories:
            timestamp = memory.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            days_ago = (datetime.now() - timestamp).days
            recency_factor = max(0, 30 - days_ago) / 30  # 0.0 to 1.0
            memory["relevance_score"] = memory.get("relevance_score", 0) + (recency_factor * 5)
        
        # Sort by relevance score
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return memories
    
    async def apply_emotional_bias(self, memories):
        """Boost relevance of emotionally charged memories"""
        for memory in memories:
            emotional_intensity = memory.get("emotional_intensity", 0)
            significance = memory.get("significance", 0)
            
            # Calculate emotion factor (0.0 to 1.0)
            emotion_factor = emotional_intensity / 100
            
            # Calculate significance factor (0.0 to 1.0)
            signif_factor = significance / 10
            
            # Combine for emotional bias
            emotion_bias = (emotion_factor * 3) + (signif_factor * 2)
            
            memory["relevance_score"] = memory.get("relevance_score", 0) + emotion_bias
        
        # Sort by relevance score
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return memories
    
    async def propagate_memory(self, memory_text, tags, significance, emotional_intensity):
        """Propagate significant memories to related NPCs"""
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Find NPCs with relationships to this NPC
            rows = await conn.fetch("""
                SELECT entity2_id 
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2
                AND entity1_type='npc' AND entity1_id=$3
                AND entity2_type='npc'
            """, self.user_id, self.conversation_id, self.npc_id)
            
            related_npc_ids = [row["entity2_id"] for row in rows]
            
            # Get the current NPC's name
            row = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
            """, self.user_id, self.conversation_id, self.npc_id)
            
            if not row:
                return
                
            npc_name = row["npc_name"]
            
            # Create a secondhand memory for each related NPC
            for related_npc_id in related_npc_ids:
                secondhand_memory = f"I heard that {npc_name} {memory_text}"
                secondhand_tags = tags + ["secondhand"]
                
                # Reduce significance and emotional intensity for secondhand memories
                secondhand_significance = max(1, significance - 2)
                secondhand_intensity = max(0, emotional_intensity - 20)
                
                # Create a new MemoryManager for this NPC
                related_memory_manager = EnhancedMemoryManager(related_npc_id, self.user_id, self.conversation_id)
                await related_memory_manager.add_memory(
                    secondhand_memory,
                    memory_type="secondhand",
                    significance=secondhand_significance,
                    emotional_valence=secondhand_intensity / 10,  # Convert back to -10...+10 scale
                    tags=secondhand_tags
                )
                
        except Exception as e:
            logging.error(f"Error propagating memory: {e}")
        finally:
            await conn.close()
    
    async def consolidate_memories(self, threshold=10, max_days=30):
        """Create summary memories from similar memories to prevent overflow"""
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Find memory clusters by tag
            rows = await conn.fetch("""
                SELECT tags, COUNT(*) as count
                FROM NPCMemories
                WHERE npc_id = $1
                AND timestamp > NOW() - INTERVAL '$2 days'
                AND is_consolidated = FALSE
                GROUP BY tags
                HAVING COUNT(*) >= $3
            """, self.npc_id, max_days, threshold)
            
            for row in rows:
                tags = row["tags"]
                count = row["count"]
                
                if not tags:
                    continue
                    
                # Get all memories with these tags
                mem_rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type, emotional_intensity, significance
                    FROM NPCMemories
                    WHERE npc_id = $1 AND tags = $2
                    AND is_consolidated = FALSE
                    ORDER BY timestamp DESC
                """, self.npc_id, tags)
                
                memories = [(row["id"], row["memory_text"], row["memory_type"], 
                             row["emotional_intensity"], row["significance"]) 
                           for row in mem_rows]
                
                # Skip if not enough memories
                if len(memories) < threshold:
                    continue
                
                # Get the NPC's name
                npc_row = await conn.fetchrow("""
                    SELECT npc_name FROM NPCStats
                    WHERE npc_id = $1
                """, self.npc_id)
                
                if not npc_row:
                    continue
                    
                npc_name = npc_row["npc_name"]
                
                # Prepare text for summarization
                memory_texts = "\n".join([f"- {mem[1]}" for mem in memories[:20]])
                tags_str = ", ".join(tags)
                
                # Get a summary using OpenAI
                prompt = f"""
                Summarize these related memories from {npc_name}'s perspective into one consolidated memory.
                The memories are all tagged with: {tags_str}
                
                Memories:
                {memory_texts}
                
                Create a single paragraph that captures the essence of these memories
                from {npc_name}'s first-person perspective, beginning with "I remember..."
                """
                
                try:
                    client = openai.AsyncOpenAI()
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.7
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    
                    # Calculate average emotional intensity and max significance
                    avg_intensity = sum(mem[3] for mem in memories) / len(memories)
                    max_significance = max(mem[4] for mem in memories)
                    
                    # Create the consolidated memory
                    await self.add_memory(
                        summary,
                        memory_type="consolidated",
                        significance=max_significance,
                        emotional_valence=avg_intensity / 10,
                        tags=tags + ["consolidated"]
                    )
                    
                    # Mark the original memories as consolidated
                    memory_ids = [mem[0] for mem in memories]
                    if memory_ids:
                        await conn.execute("""
                            UPDATE NPCMemories
                            SET is_consolidated = TRUE
                            WHERE id = ANY($1)
                        """, memory_ids)
                        
                except Exception as e:
                    logging.error(f"Error consolidating memories: {e}")
                    
        except Exception as e:
            logging.error(f"Error in consolidate_memories: {e}")
        finally:
            await conn.close()
    
    async def apply_memory_decay(self, decay_rate=0.05, max_days=90):
        """Apply gradual forgetting based on time and recall frequency"""
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Find old memories that have been recalled infrequently
            rows = await conn.fetch("""
                SELECT id, emotional_intensity, times_recalled, 
                       EXTRACT(DAY FROM NOW() - timestamp) as age_days
                FROM NPCMemories
                WHERE npc_id = $1
                AND timestamp < NOW() - INTERVAL '$2 days'
            """, self.npc_id, max_days)
            
            for row in rows:
                memory_id = row["id"]
                intensity = row["emotional_intensity"]
                recalled = row["times_recalled"]
                age_days = row["age_days"]
                
                # Calculate decay based on age and recall frequency
                # Memories recalled often decay more slowly
                recall_factor = min(1.0, recalled / 10)  # 0.0 to 1.0
                age_factor = min(1.0, age_days / 365)    # 0.0 to 1.0
                
                # Calculate how much to reduce emotional intensity
                decay_amount = intensity * decay_rate * age_factor * (1 - recall_factor)
                
                # Apply decay if significant
                if decay_amount >= 1:
                    new_intensity = max(0, intensity - int(decay_amount))
                    
                    await conn.execute("""
                        UPDATE NPCMemories
                        SET emotional_intensity = $1
                        WHERE id = $2
                    """, new_intensity, memory_id)
            
        except Exception as e:
            logging.error(f"Error applying memory decay: {e}")
        finally:
            await conn.close()
