# logic/memory_manager.py

import json
import logging
import os
import asyncpg
import openai
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")

class EnhancedMemoryManager:
    """Manages memories for an individual NPC with advanced features"""

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def add_memory(self,
                         memory_text: str,
                         memory_type: str = "observation",
                         significance: int = 3,
                         emotional_valence: int = 0,
                         tags: Optional[List[str]] = None) -> Optional[int]:
        """Add memory with metadata & optional embedding."""
        tags = tags or []
        content_tags = await self.analyze_memory_content(memory_text)
        tags.extend(content_tags)
        emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)

        embedding_data = None
        try:
            # Use async OpenAI call
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=memory_text
            )
            embedding_data = response["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"[EnhancedMemoryManager] Error generating embedding: {e}")

        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            memory_id = await conn.fetchval("""
                INSERT INTO NPCMemories (
                    npc_id, memory_text, memory_type, tags,
                    emotional_intensity, significance, embedding,
                    associated_entities, is_consolidated
                )
                VALUES ($1, $2, $3, $4,
                        $5, $6, $7,
                        $8, $9)
                RETURNING id
            """,
                self.npc_id,
                memory_text,
                memory_type,
                tags,
                emotional_intensity,
                significance,
                embedding_data,
                json.dumps({}),
                False
            )
            if significance >= 4:
                await self.propagate_memory(memory_text, tags, significance, emotional_intensity)
            return memory_id
        except Exception as e:
            logging.error(f"[EnhancedMemoryManager] add_memory error: {e}")
            return None
        finally:
            if conn:
                await conn.close()

    async def retrieve_relevant_memories(self, context: Any, limit: int = 5) -> List[dict]:
        """Retrieve relevant memories, using embeddings or fallback keyword search."""
        query_text = context if isinstance(context, str) else context.get("description", "")
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                response = await openai.Embedding.acreate(
                    model="text-embedding-ada-002",
                    input=query_text
                )
                query_vector = response["data"][0]["embedding"]
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
                logging.error(f"[EnhancedMemoryManager] Embedding error for query: {e}")
                # Fallback to keyword
                words = query_text.lower().split()
                conditions = " OR ".join([f"LOWER(memory_text) LIKE '%'||${i+2}||'%'" for i in range(len(words))])
                params = [self.npc_id] + words + [limit]
                q = f"""
                    SELECT id, memory_text, memory_type, tags,
                           emotional_intensity, significance,
                           times_recalled, timestamp
                    FROM NPCMemories
                    WHERE npc_id = $1
                    AND ({conditions})
                    ORDER BY timestamp DESC
                    LIMIT ${len(params)}
                """
                rows = await conn.fetch(q, *params)

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
                    "relevance_score": 0
                })
            memories = await self.apply_recency_bias(memories)
            memories = await self.apply_emotional_bias(memories)

            await self.update_memory_retrieval_stats([m["id"] for m in memories])
            return memories
        except Exception as e:
            logging.error(f"[EnhancedMemoryManager] retrieve_relevant_memories error: {e}")
            return []
        finally:
            if conn:
                await conn.close()

    async def update_memory_retrieval_stats(self, memory_ids: List[int]):
        if not memory_ids:
            return
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            await conn.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = NOW()
                WHERE id = ANY($1)
            """, memory_ids)
        except Exception as e:
            logging.error(f"[EnhancedMemoryManager] update_memory_retrieval_stats error: {e}")
        finally:
            if conn:
                await conn.close()

    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        tags = []
        # Example naive approach
        text_lower = memory_text.lower()
        if "angry" in text_lower or "upset" in text_lower:
            tags.append("negative_emotion")
        if "happy" in text_lower or "pleased" in text_lower:
            tags.append("positive_emotion")
        if "player" in text_lower or "chase" in text_lower:
            tags.append("player_related")
        return tags

    async def calculate_emotional_intensity(self, memory_text: str, base_valence: float) -> float:
        intensity = abs(base_valence)*10
        # Optionally add logic for keywords
        emotion_words = {
            "furious": 20, "ecstatic": 20, "devastated": 20, "thrilled": 20,
            "angry": 15, "delighted": 15, "sad": 15, "happy": 15,
            "annoyed": 10, "pleased": 10, "upset": 10, "glad": 10,
            "concerned": 5, "fine": 5, "worried": 5, "okay": 5
        }
        lw = memory_text.lower()
        for word, boost in emotion_words.items():
            if word in lw:
                intensity += boost
                break
        return float(min(100, intensity))

    async def apply_recency_bias(self, memories: List[dict]) -> List[dict]:
        now = datetime.now()
        for mem in memories:
            ts = mem.get("timestamp")
            if ts and isinstance(ts, datetime):
                days_ago = (now - ts).days
            elif ts and isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    days_ago = (now - dt).days
                except:
                    days_ago = 30
            else:
                days_ago = 30
            recency_factor = max(0, 30 - days_ago)/30
            mem["relevance_score"] += recency_factor*5
        # Re-sort
        memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return memories

    async def apply_emotional_bias(self, memories: List[dict]) -> List[dict]:
        for mem in memories:
            emo_int = mem.get("emotional_intensity", 0)
            sig = mem.get("significance", 0)
            emotion_factor = emo_int/100
            signif_factor = sig/10
            mem["relevance_score"] += (emotion_factor*3 + signif_factor*2)
        memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return memories

    async def propagate_memory(self, memory_text: str, tags: List[str], significance: int, emotional_intensity: float):
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            rel_rows = await conn.fetch("""
                SELECT entity2_id
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2
                  AND entity1_type='npc' AND entity1_id=$3
                  AND entity2_type='npc'
            """, self.user_id, self.conversation_id, self.npc_id)
            related_npcs = [r["entity2_id"] for r in rel_rows]
            npc_row = await conn.fetchrow("""
                SELECT npc_name
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
                  AND npc_id=$3
            """, self.user_id, self.conversation_id, self.npc_id)
            if not npc_row:
                return
            npc_name = npc_row["npc_name"]
        except Exception as e:
            logging.error(f"[EnhancedMemoryManager] propagate_memory fetch error: {e}")
            return
        finally:
            if conn:
                await conn.close()

        # For each related NPC, create secondhand memory
        for rid in related_npcs:
            secondhand = f"I heard that {npc_name} {memory_text}"
            secondhand_tags = tags + ["secondhand"]
            secondhand_signif = max(1, significance-2)
            secondhand_intensity = max(0, emotional_intensity-20)
            # Re-use manager
            related_mem_mgr = EnhancedMemoryManager(rid, self.user_id, self.conversation_id)
            await related_mem_mgr.add_memory(
                secondhand,
                memory_type="secondhand",
                significance=secondhand_signif,
                emotional_valence=secondhand_intensity/10,
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
