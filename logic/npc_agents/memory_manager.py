# logic/memory_manager.py

import json
import logging
import os
import asyncpg
import openai
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")

logger = logging.getLogger(__name__)

class EnhancedMemoryManager:
    """
    Manages memories for an individual NPC with advanced features:
      - significance (1..10 or 1..100) to indicate importance
      - status in ('active','summarized','archived')
      - human-like fading, summarization, recall mechanics
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    # -----------------------------------------------------------
    # A) ADDING A NEW MEMORY
    # -----------------------------------------------------------
    async def add_memory(
        self,
        memory_text: str,
        memory_type: str = "observation",
        significance: int = 3,
        emotional_valence: int = 0,
        tags: Optional[List[str]] = None,
        status: str = "active"
    ) -> Optional[int]:
        """
        Add a memory row with optional embedding. By default:
         - significance=3
         - emotional_valence => converted to emotional_intensity [0..100]
         - status='active'
         - times_recalled=0
         - last_recalled=NULL
         - is_consolidated=FALSE
        """
        tags = tags or []
        # Possibly parse content to add extra tags
        content_tags = await self.analyze_memory_content(memory_text)
        tags.extend(content_tags)

        emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)

        # Generate embedding (optional step)
        embedding_data = None
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=memory_text
            )
            embedding_data = response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] Error generating embedding: {e}")

        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            # Insert row with the new columns significance & status
            memory_id = await conn.fetchval("""
                INSERT INTO NPCMemories (
                    npc_id, memory_text, memory_type, tags,
                    emotional_intensity, significance, embedding,
                    associated_entities, is_consolidated, status
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, FALSE, $9
                )
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
                status
            )

            # Optionally propagate if significance is high
            if significance >= 4:
                await self.propagate_memory(memory_text, tags, significance, emotional_intensity)

            return memory_id
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] add_memory error: {e}")
            return None
        finally:
            if conn:
                await conn.close()

    # -----------------------------------------------------------
    # B) RETRIEVING MEMORIES
    # -----------------------------------------------------------
    async def retrieve_relevant_memories(self, context: Any, limit: int = 5) -> List[dict]:
        """
        Retrieve relevant memories, using embeddings if possible. 
        Only search 'active' or 'summarized' memories by default 
        to exclude 'archived' from normal results (unless you want them).
        """
        query_text = context if isinstance(context, str) else context.get("description", "")
        conn = None
        memories = []
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)

            # Try embeddings
            try:
                response = await openai.Embedding.acreate(
                    model="text-embedding-ada-002",
                    input=query_text
                )
                query_vector = response["data"][0]["embedding"]
                rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type, tags,
                           emotional_intensity, significance,
                           times_recalled, timestamp, status
                    FROM NPCMemories
                    WHERE npc_id = $1
                      AND status IN ('active','summarized')
                    ORDER BY embedding <-> $2
                    LIMIT $3
                """, self.npc_id, query_vector, limit)

            except Exception as e:
                logger.error(f"[EnhancedMemoryManager] Embedding error for query: {e}")
                # Fallback: naive keyword search
                words = query_text.lower().split()
                if words:
                    conditions = " OR ".join(
                        [f"LOWER(memory_text) LIKE '%'||${i+2}||'%'" for i in range(len(words))]
                    )
                    # param order: [npc_id] + words + [limit]
                    params = [self.npc_id] + words + [limit]
                    q = f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status
                        FROM NPCMemories
                        WHERE npc_id = $1
                          AND status IN ('active','summarized')
                          AND ({conditions})
                        ORDER BY timestamp DESC
                        LIMIT ${len(params)}
                    """
                    rows = await conn.fetch(q, *params)
                else:
                    rows = []

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
                    "status": row["status"],
                    "relevance_score": 0
                })

            # Optionally apply recency or emotional biases
            memories = await self.apply_recency_bias(memories)
            memories = await self.apply_emotional_bias(memories)

            # Update retrieval stats
            await self.update_memory_retrieval_stats([m["id"] for m in memories])
            return memories

        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] retrieve_relevant_memories error: {e}")
            return []
        finally:
            if conn:
                await conn.close()

    async def update_memory_retrieval_stats(self, memory_ids: List[int]):
        """Increment times_recalled, update last_recalled=NOW() for retrieved memories."""
        if not memory_ids:
            return
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            await conn.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = CURRENT_TIMESTAMP
                WHERE id = ANY($1)
            """, memory_ids)
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] update_memory_retrieval_stats error: {e}")
        finally:
            if conn:
                await conn.close()

    # -----------------------------------------------------------
    # C) RECALL & LIFECYCLE HELPER METHODS
    # -----------------------------------------------------------
    async def recall_memory(self, memory_id: int):
        """
        Mark a memory as recalled => times_recalled++, last_recalled=NOW().
        Optionally bump significance if we want repeated referencing to 
        keep it more relevant.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            await conn.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = CURRENT_TIMESTAMP,
                    significance = LEAST(significance + 1, 10)
                WHERE id = $1
                  AND npc_id = $2
            """, memory_id, self.npc_id)
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] recall_memory error: {e}")
        finally:
            if conn:
                await conn.close()

    async def prune_old_memories(self,
                                 age_days: int = 14,
                                 significance_threshold: int = 3,
                                 intensity_threshold: int = 15):
        """
        1) Delete truly trivial memories older than age_days 
           (where significance < significance_threshold and emotional_intensity < intensity_threshold).
        2) For the rest older than age_days, if status='active', set status='summarized'.
        """
        cutoff = datetime.now() - timedelta(days=age_days)
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)

            # Step 1: Delete truly trivial
            del_result = await conn.execute("""
                DELETE FROM NPCMemories
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND significance < $3
                  AND emotional_intensity < $4
            """, self.npc_id, cutoff, significance_threshold, intensity_threshold)
            logger.info(f"[EnhancedMemoryManager] prune_old_memories => {del_result} trivial removed.")

            # Step 2: For older memories, set status='summarized' if they are 'active'
            upd_result = await conn.execute("""
                UPDATE NPCMemories
                SET status='summarized'
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND status='active'
            """, self.npc_id, cutoff)
            logger.info(f"[EnhancedMemoryManager] prune_old_memories => {upd_result} set to summarized.")
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] prune_old_memories error: {e}")
        finally:
            if conn:
                await conn.close()

    async def apply_memory_decay(self, age_days: int = 30, decay_rate: float = 0.2):
        """
        Decrease emotional_intensity and significance for older memories 
        that are not recalled recently. 
        This simulates forgetting or diminishing emotional punch.
        """
        cutoff = datetime.now() - timedelta(days=age_days)
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            # For memories older than age_days
            rows = await conn.fetch("""
                SELECT id, emotional_intensity, significance, times_recalled
                FROM NPCMemories
                WHERE npc_id=$1
                  AND timestamp < $2
                  AND status IN ('active','summarized')
            """, self.npc_id, cutoff)
            for row in rows:
                mem_id = row["id"]
                intensity = row["emotional_intensity"]
                signif = row["significance"]
                recalled = row["times_recalled"]

                # If recalled a lot, we degrade less
                recall_factor = min(1.0, recalled / 10.0)
                new_intensity = max(0, int(intensity * (1 - decay_rate*(1 - recall_factor))))
                new_signif = max(1, int(signif * (1 - decay_rate*(1 - recall_factor))))

                await conn.execute("""
                    UPDATE NPCMemories
                    SET emotional_intensity=$1,
                        significance=$2
                    WHERE id=$3
                """, new_intensity, new_signif, mem_id)

            logger.info(f"[EnhancedMemoryManager] apply_memory_decay => done for NPC {self.npc_id}.")
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] apply_memory_decay error: {e}")
        finally:
            if conn:
                await conn.close()

    # -----------------------------------------------------------
    # D) REPETITIVE MEMORIES -> SUMMARIZATION
    # -----------------------------------------------------------
    async def summarize_repetitive_memories(self, lookback_days: int = 7, min_count: int = 3):
        """
        Merges multiple repeated or similar memory_text entries into a single summary memory.
        Then sets the old ones as is_consolidated=TRUE or possibly removes them entirely.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            rows = await conn.fetch("""
                SELECT memory_text, COUNT(*) AS cnt, array_agg(id) AS mem_ids
                FROM NPCMemories
                WHERE npc_id=$1
                  AND timestamp > (NOW() - $2::INTERVAL)
                  AND status='active'
                  AND is_consolidated=FALSE
                GROUP BY memory_text
                HAVING COUNT(*) >= $3
            """, self.npc_id, f"{lookback_days} days", min_count)

            for row in rows:
                text = row["memory_text"]
                count = row["cnt"]
                mem_ids = row["mem_ids"]
                if not mem_ids or count < min_count:
                    continue

                summary_text = f"I recall {count} similar moments: '{text}' repeated multiple times."

                # Insert a new consolidated memory with moderate significance
                new_id = await conn.fetchval("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type,
                        emotional_intensity, significance, status,
                        is_consolidated
                    )
                    VALUES ($1, $2, 'consolidated', 10, 5, 'summarized', FALSE)
                    RETURNING id
                """, self.npc_id, summary_text)
                logger.debug(f"[EnhancedMemoryManager] Summarized {count} mems into {new_id}")

                # Mark old ones as consolidated or remove them
                await conn.execute("""
                    UPDATE NPCMemories
                    SET is_consolidated=TRUE, status='summarized'
                    WHERE id=ANY($1)
                """, mem_ids)
            logger.info("[EnhancedMemoryManager] summarize_repetitive_memories completed.")
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] summarize_repetitive_memories error: {e}")
        finally:
            if conn:
                await conn.close()

    # -----------------------------------------------------------
    # E) EMBEDDING / BIAS / PROPAGATION
    # -----------------------------------------------------------
    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        """Naive approach to auto-tagging memory content."""
        tags = []
        lower_text = memory_text.lower()
        if "angry" in lower_text or "upset" in lower_text:
            tags.append("negative_emotion")
        if "happy" in lower_text or "pleased" in lower_text:
            tags.append("positive_emotion")
        if "player" in lower_text or "chase" in lower_text:
            tags.append("player_related")
        return tags

    async def calculate_emotional_intensity(self, memory_text: str, base_valence: float) -> float:
        """Convert valence [-10..10] plus keywords => [0..100]."""
        intensity = (base_valence + 10)*5  # baseline
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
        return float(min(100, max(0, intensity)))

    async def apply_recency_bias(self, memories: List[dict]) -> List[dict]:
        """Boost recent memories' relevance_score slightly."""
        now = datetime.now()
        for mem in memories:
            ts = mem.get("timestamp")
            if isinstance(ts, datetime):
                days_ago = (now - ts).days
            elif isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    days_ago = (now - dt).days
                except:
                    days_ago = 30
            else:
                days_ago = 30

            recency_factor = max(0, 30 - days_ago)/30
            mem["relevance_score"] += recency_factor*5
        memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return memories

    async def apply_emotional_bias(self, memories: List[dict]) -> List[dict]:
        """Raise relevance_score based on emotional_intensity + significance."""
        for mem in memories:
            emo_int = mem.get("emotional_intensity", 0) / 100.0
            sig = mem.get("significance", 0) / 10.0
            mem["relevance_score"] += (emo_int*3 + sig*2)
        memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return memories

    async def propagate_memory(self, memory_text: str, tags: List[str], significance: int, emotional_intensity: float):
        """
        If significance is high, share a 'secondhand' memory with other NPCs connected to this NPC.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            rows = await conn.fetch("""
                SELECT entity2_id
                FROM SocialLinks
                WHERE user_id=$1
                  AND conversation_id=$2
                  AND entity1_type='npc'
                  AND entity1_id=$3
                  AND entity2_type='npc'
            """, self.user_id, self.conversation_id, self.npc_id)
            related_npcs = [r["entity2_id"] for r in rows]

            row2 = await conn.fetchrow("""
                SELECT npc_name
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
                  AND npc_id=$3
            """, self.user_id, self.conversation_id, self.npc_id)
            npc_name = row2["npc_name"] if row2 else f"NPC_{self.npc_id}"

        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] propagate_memory fetch error: {e}")
            return
        finally:
            if conn:
                await conn.close()

        # Insert the secondhand memories
        for rid in related_npcs:
            secondhand_text = f"I heard that {npc_name} {memory_text}"
            secondhand_signif = max(1, significance - 2)
            secondhand_int = max(0, emotional_intensity - 20)
            # Use a new manager for each related NPC
            r_mgr = EnhancedMemoryManager(rid, self.user_id, self.conversation_id)
            await r_mgr.add_memory(
                secondhand_text,
                memory_type="secondhand",
                significance=secondhand_signif,
                emotional_valence=(secondhand_int / 10),
                tags=tags + ["secondhand"],
                status="active"
            )

        logger.debug("[EnhancedMemoryManager] propagate_memory => secondhand memories inserted")

    # -----------------------------------------------------------
    # F) CONSOLIDATION (optional)
    # -----------------------------------------------------------
    async def consolidate_memories(self, threshold=10, max_days=30):
        """
        Example: Merges clusters of memories with identical tags or 
        other criteria into a single consolidated memory, setting 
        'is_consolidated=TRUE' for the originals.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            rows = await conn.fetch("""
                SELECT tags, COUNT(*) as count
                FROM NPCMemories
                WHERE npc_id=$1
                  AND timestamp > NOW() - INTERVAL '$2 days'
                  AND is_consolidated=FALSE
                  AND status IN ('active','summarized')
                GROUP BY tags
                HAVING COUNT(*) >= $3
            """, self.npc_id, max_days, threshold)

            for row in rows:
                group_tags = row["tags"]
                cnt = row["count"]
                if not group_tags:
                    continue
                # Grab the full set
                mem_rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type,
                           emotional_intensity, significance
                    FROM NPCMemories
                    WHERE npc_id=$1
                      AND tags=$2
                      AND is_consolidated=FALSE
                      AND status IN ('active','summarized')
                """, self.npc_id, group_tags)

                if len(mem_rows) < threshold:
                    continue

                # Summarize them
                joined_text = "\n".join([r["memory_text"] for r in mem_rows])
                summary_text = (
                    f"I recall {len(mem_rows)} similar events with tags {group_tags}: \n{joined_text[:200]}..."
                )

                # Insert new consolidated memory
                new_id = await conn.fetchval("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type,
                        emotional_intensity, significance,
                        status, is_consolidated
                    )
                    VALUES ($1, $2, 'consolidated', 10, 5, 'summarized', FALSE)
                    RETURNING id
                """, self.npc_id, summary_text)

                # Mark old ones
                old_ids = [r["id"] for r in mem_rows]
                await conn.execute("""
                    UPDATE NPCMemories
                    SET is_consolidated=TRUE, status='summarized'
                    WHERE id=ANY($1)
                """, old_ids)

            logger.info("[EnhancedMemoryManager] consolidate_memories => done.")
        except Exception as e:
            logger.error(f"[EnhancedMemoryManager] consolidate_memories error: {e}")
        finally:
            if conn:
                await conn.close()

    # -----------------------------------------------------------
    # G) ARCHIVING STALE MEMORIES (Optional)
    # -----------------------------------------------------------
    async def archive_stale_memories(
        self,
        older_than_days: int = 60,
        max_significance: int = 4
    ):
        """
        Instead of deleting older memories, set status='archived' for
        those older than 'older_than_days' with significance <= max_significance.
        So 'active' or 'summarized' memories that are low-value become 'archived'.
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            result = await conn.execute("""
                UPDATE NPCMemories
                SET status='archived'
                WHERE npc_id=$1
                  AND timestamp < $2
                  AND significance <= $3
                  AND status IN ('active','summarized')
            """, self.npc_id, cutoff, max_significance)
            logger.info(f"[EnhancedMemoryManager] archive_stale_memories => {result} set to archived.")
        except Exception as e:
            logger.error(f"Error archiving stale memories: {e}")
        finally:
            if conn:
                await conn.close()

import asyncpg
import openai
import random
from datetime import datetime

class EnhancedMemoryManager:
    def __init__(
        self, 
        user_id: int, 
        npc_id: int, 
        db_pool,
        npc_personality: str = "neutral",
        npc_intelligence: float = 1.0
    ):
        """
        user_id: the player's ID
        npc_id: which NPC we're managing memories for
        db_pool: asyncpg pool
        npc_personality: 'gullible', 'skeptical', 'paranoid', or 'neutral'
        npc_intelligence: numeric factor (0.5..2.0) controlling memory decay speed
        """
        self.user_id = user_id
        self.npc_id = npc_id
        self.db_pool = db_pool
        self.npc_personality = npc_personality
        self.npc_intelligence = npc_intelligence

        self.memory_cache = {}

    async def generate_embedding(self, text: str) -> list:
        """
        One-off call to OpenAI’s embedding API for storing or comparing text with pgvector.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]

    async def add_memory(self, memory_text: str, significance: int=3, emotional_intensity: int=0):
        """
        Insert a new memory with an embedding.
        Applies personality bias, rumor propagation, reputation & future outcomes.
        """
        conn = await self.db_pool.acquire()
        try:
            embedding = await self.generate_embedding(memory_text)
            row = await conn.fetchrow("""
                INSERT INTO NPCMemories (
                    npc_id,
                    memory_text,
                    significance,
                    emotional_intensity,
                    embedding,
                    confidence,
                    status,
                    timestamp
                )
                VALUES ($1, $2, $3, $4, $5, 1.0, 'active', NOW())
                RETURNING id
            """, self.npc_id, memory_text, significance, emotional_intensity, embedding)
            mem_id = row["id"]

            # NPC personality bias for initial confidence
            await self.apply_npc_memory_bias(conn, mem_id)

            # Social rumor propagation
            await self.propagate_memory(conn, memory_text, significance, emotional_intensity)

            # Reputation & future outcomes
            await self.update_reputation(conn, memory_text)
            await self.update_future_outcomes(conn, memory_text)

        finally:
            await self.db_pool.release(conn)

    async def propagate_memory(self, conn, memory_text: str, significance: int, emotional_intensity: int):
        """
        If significance >=5, partially rewrite rumor text,
        then propagate to allied NPCs.
        """
        if significance < 5:
            return

        allies = await conn.fetch("""
            SELECT entity2_id AS other_npc
            FROM SocialLinks
            WHERE entity1_id=$1 AND link_type='ally'
        """, self.npc_id)

        for row in allies:
            other_npc_id = row["other_npc"]
            distorted_text = self.distort_text(memory_text, severity=0.3)
            await conn.execute("""
                INSERT INTO NPCMemories (
                    npc_id,
                    memory_text,
                    significance,
                    emotional_intensity,
                    embedding,
                    confidence,
                    status,
                    timestamp
                )
                VALUES ($1, $2, $3, $4, null, 0.8, 'active', NOW())
            """, other_npc_id, distorted_text, max(1, significance - 1), emotional_intensity)

    def distort_text(self, original_text: str, severity=0.3) -> str:
        """
        Word-level partial rewrite for rumor distortion.
        """
        synonyms_map = {
            "attacked": ["assaulted", "ambushed", "jumped"],
            "betrayed": ["backstabbed", "double-crossed"],
            "stole": ["looted", "swiped", "snatched"],
            "helped": ["assisted", "saved", "bailed out"],
            "rescued": ["freed", "saved", "pulled out"]
        }
        words = original_text.split()
        for i in range(len(words)):
            if random.random() < severity:
                w_lower = words[i].lower()
                if w_lower in synonyms_map:
                    words[i] = random.choice(synonyms_map[w_lower])
                elif random.random() < 0.2:
                    words[i] = ""
        return " ".join([w for w in words if w])

    async def apply_npc_memory_bias(self, conn, memory_id: int):
        """
        Adjust memory confidence by personality type at creation.
        gullible=1.2, skeptical=0.8, paranoid=1.5, neutral=1.0
        """
        personality_factors = {
            "gullible": 1.2,
            "skeptical": 0.8,
            "paranoid": 1.5,
            "neutral": 1.0
        }
        factor = personality_factors.get(self.npc_personality, 1.0)
        await conn.execute("""
            UPDATE NPCMemories
            SET confidence = LEAST(confidence * $1, 1.0)
            WHERE id=$2
        """, factor, memory_id)

    async def reinforce_memory(self, memory_text: str, reinforcement_base: float=0.1):
        """
        Vector similarity-based reinforcement, factoring in:
         - trust level
         - emotional intensity
         - existing confidence
         - multiple reinforcements for high confidence
        """
        conn = await self.db_pool.acquire()
        try:
            query_embedding = await self.generate_embedding(memory_text)
            row = await conn.fetchrow("""
                SELECT id, confidence, emotional_intensity, (embedding <-> $2) AS distance
                FROM NPCMemories
                WHERE npc_id=$1 AND status='active'
                ORDER BY embedding <-> $2
                LIMIT 1
            """, self.npc_id, query_embedding)

            if row:
                mem_id = row["id"]
                distance = row["distance"]
                current_conf = row["confidence"]
                emo = row["emotional_intensity"]

                # Get trust level from PlayerReputation (range -100..100)
                trust_level = await conn.fetchval("""
                    SELECT reputation_score
                    FROM PlayerReputation
                    WHERE user_id=$1 AND npc_id=$2
                """, self.user_id, self.npc_id) or 0

                # Turn trust into a multiplier
                trust_multiplier = 1.0 + (trust_level / 100.0)

                # Higher confidence => more scaling down => multiple reinforcements needed
                # e.g. if current_conf=0.8, reduce base by 0.24 => 0.1*(1-0.8*0.3)=some fraction
                conf_scaling = 1 - (current_conf * 0.3)
                # Final reinforcement factor
                reinforcement_factor = max(0.05, reinforcement_base * conf_scaling * trust_multiplier)

                # Dynamic threshold factoring in confidence & emotional intensity
                threshold = max(0.10, min(0.30, 0.30 - (current_conf * 0.05) - (emo * 0.01)))

                if distance < threshold:
                    await conn.execute("""
                        UPDATE NPCMemories
                        SET confidence = LEAST(confidence + $1, 1.0)
                        WHERE id=$2
                    """, reinforcement_factor, mem_id)
        finally:
            await self.db_pool.release(conn)

    async def retrieve_relevant_memories(self, query_text: str, limit: int=5):
        """
        Multi-trait personality weighting for retrieval:
         - vector similarity
         - emotional_intensity
         - recency
         - paranoia factor for 'betrayed' memories
         - skepticism factor for 'rumor' memories
        """
        personality_weights = {
            "gullible": {
                "emotion_factor": 0.15,
                "recency_factor": 0.05,
                "skepticism_factor": 0.9,
                "paranoia_factor": 0.8
            },
            "skeptical": {
                "emotion_factor": 0.05,
                "recency_factor": 0.15,
                "skepticism_factor": 1.2,
                "paranoia_factor": 1.0
            },
            "paranoid": {
                "emotion_factor": 0.25,
                "recency_factor": 0.10,
                "skepticism_factor": 1.0,
                "paranoia_factor": 1.5
            },
            "neutral": {
                "emotion_factor": 0.10,
                "recency_factor": 0.10,
                "skepticism_factor": 1.0,
                "paranoia_factor": 1.0
            }
        }
        w = personality_weights.get(self.npc_personality, personality_weights["neutral"])

        query_embedding = await self.generate_embedding(query_text)
        conn = await self.db_pool.acquire()
        try:
            rows = await conn.fetch(f"""
                SELECT
                    id,
                    memory_text,
                    significance,
                    emotional_intensity,
                    confidence,
                    timestamp,
                    (embedding <-> $2) AS similarity
                FROM NPCMemories
                WHERE npc_id=$1
                  AND status='active'
                ORDER BY (
                    similarity
                    - (emotional_intensity * {w["emotion_factor"]})
                    - ((EXTRACT(EPOCH FROM (NOW() - timestamp)) / 1000000.0) * {w["recency_factor"]})
                    - (CASE WHEN memory_text ILIKE '%betrayed%' THEN {w["paranoia_factor"]} ELSE 0 END)
                    - (CASE WHEN memory_text ILIKE '%rumor%' THEN {w["skepticism_factor"]} ELSE 0 END)
                )
                LIMIT {limit}
            """, self.npc_id, query_embedding)
            return rows
        finally:
            await self.db_pool.release(conn)

    async def update_reputation(self, conn, memory_text: str):
        """
        Subtle +/- 5 changes for references to 'helped' or 'betrayed'.
        Clamped at [-100..100].
        """
        lower_text = memory_text.lower()
        if "betrayed" in lower_text:
            await conn.execute("""
                UPDATE PlayerReputation
                SET reputation_score = GREATEST(reputation_score - 5, -100)
                WHERE user_id=$1 AND npc_id=$2
            """, self.user_id, self.npc_id)
        elif "helped" in lower_text:
            await conn.execute("""
                UPDATE PlayerReputation
                SET reputation_score = LEAST(reputation_score + 5, 100)
                WHERE user_id=$1 AND npc_id=$2
            """, self.user_id, self.npc_id)

    async def update_future_outcomes(self, conn, memory_text: str):
        """
        Probability-based changes for future outcomes, e.g. 'NPC seeks revenge'.
        """
        lt = memory_text.lower()
        if "betrayed" in lt:
            await conn.execute("""
                INSERT INTO FutureOutcomes (npc_id, event, probability)
                VALUES ($1, 'NPC seeks revenge', 0.5)
                ON CONFLICT DO NOTHING
            """, self.npc_id)
            await conn.execute("""
                UPDATE FutureOutcomes
                SET probability = LEAST(probability + 0.1, 1.0)
                WHERE npc_id=$1 AND event='NPC seeks revenge'
            """, self.npc_id)
        elif "helped" in lt:
            await conn.execute("""
                INSERT INTO FutureOutcomes (npc_id, event, probability)
                VALUES ($1, 'NPC allies with player', 0.5)
                ON CONFLICT DO NOTHING
            """, self.npc_id)
            await conn.execute("""
                UPDATE FutureOutcomes
                SET probability = GREATEST(probability - 0.1, 0.0)
                WHERE npc_id=$1 AND event='NPC allies with player'
            """, self.npc_id)

    async def fade_memories(self):
        """
        Intelligence-based memory decay + personality factor.
        E.g. 'smart' NPC => fade slower, 'skeptical' => fade faster, etc.
        """
        personality_decay = {
            "gullible": 0.8,
            "skeptical": 1.2,
            "paranoid": 0.6,
            "neutral": 1.0
        }
        base_factor = personality_decay.get(self.npc_personality, 1.0)

        # E.g. npc_intelligence=2 => fade 2x slower, 0.5 => fade 1.5x faster
        intelligence_factor = max(0.5, min(1.5, 1.5 - (self.npc_intelligence * 0.05)))
        decay_factor = base_factor * intelligence_factor

        conn = await self.db_pool.acquire()
        try:
            rows = await conn.fetch("""
                SELECT id, significance, emotional_intensity, confidence
                FROM NPCMemories
                WHERE npc_id=$1
                  AND status='active'
            """, self.npc_id)

            for row in rows:
                mem_id = row["id"]
                sig = row["significance"]
                emo = row["emotional_intensity"]
                conf = row["confidence"]

                fade_modifier = 1.0
                if emo > 5:
                    fade_modifier -= 0.2
                if conf > 0.7:
                    fade_modifier -= 0.3
                fade_modifier = max(0.1, fade_modifier) * decay_factor

                new_sig = max(1, sig - (0.5 * fade_modifier))
                new_conf = max(0, conf - (0.05 * fade_modifier))
                new_status = "active" if new_sig > 1 else "archived"

                await conn.execute("""
                    UPDATE NPCMemories
                    SET significance=$1, confidence=$2, status=$3
                    WHERE id=$4
                """, new_sig, new_conf, new_status, mem_id)
        finally:
            await self.db_pool.release(conn)

    # Optional local cache for high-frequency queries

    def preload_memories_into_cache(self, memory_rows):
        """
        E.g. store memory rows for quick local access.
        """
        self.memory_cache["cached_relevant_memories"] = memory_rows
        self.memory_cache["last_cached"] = datetime.utcnow()

    def get_cached_memories(self):
        """
        Return preloaded memories if still fresh.
        """
        return self.memory_cache.get("cached_relevant_memories", [])

    async def dynamic_npc_memory_preloading(
        self, 
        query_threshold: int = 10,
        time_window_minutes: int = 5,
        cache_timeout_minutes: int = 30
    ):
        """
        Checks if this NPC is heavily queried. If yes, 
        preload top 5 memories, refresh if stale.
        """
        conn = await self.db_pool.acquire()
        try:
            query_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM InteractionLogs
                WHERE npc_id=$1
                  AND timestamp > (NOW() - INTERVAL '$2 minutes')
            """, self.npc_id, time_window_minutes)

            current_time = datetime.utcnow()
            last_cached = self.memory_cache.get("last_cached", None)
            is_cache_stale = False
            if last_cached:
                delta_sec = (current_time - last_cached).total_seconds()
                if delta_sec > cache_timeout_minutes * 60:
                    is_cache_stale = True

            if query_count and query_count >= query_threshold and (not last_cached or is_cache_stale):
                print(f"[INFO] High-frequency NPC {self.npc_id}: refreshing memory cache.")
                query_embedding = await self.generate_embedding("general recall")
                cached_memories = await conn.fetch(f"""
                    SELECT id, memory_text
                    FROM NPCMemories
                    WHERE npc_id=$1 AND status='active'
                    ORDER BY embedding <-> $2
                    LIMIT 5
                """, self.npc_id, query_embedding)
                self.preload_memories_into_cache(cached_memories)

        finally:
            await self.db_pool.release(conn)

import asyncio

async def train_reflection_model(db_pool):
    """
    Adjust reflection settings based on reflection accuracy,
    possibly factoring in 'confidence_weight' if stored in ReflectionLogs.
    """
    conn = await db_pool.acquire()
    try:
        # Suppose we store reflection data in ReflectionLogs with a "confidence_weight" column
        row = await conn.fetchrow("""
            SELECT
                SUM(CASE WHEN was_accurate=TRUE THEN confidence_weight ELSE 0 END) AS correct_conf_sum,
                SUM(confidence_weight) AS total_conf_sum,
                COUNT(*) FILTER (WHERE was_accurate=TRUE) AS correct_count,
                COUNT(*) AS total_count
            FROM ReflectionLogs
        """)
        if not row or row["total_count"] == 0:
            return

        correct_count = row["correct_count"]
        total_count = row["total_count"]
        accuracy_rate = correct_count / total_count

        # Weighted average confidence of correct reflections (if desired)
        if row["total_conf_sum"] and row["total_conf_sum"] > 0:
            weighted_accuracy = row["correct_conf_sum"] / row["total_conf_sum"]
        else:
            weighted_accuracy = 0

        # Combine or just use accuracy_rate
        final_metric = (accuracy_rate + weighted_accuracy) / 2  # simple average

        await conn.execute("""
            UPDATE AIReflectionSettings
            SET temperature = GREATEST(0.2, 1.0 - ($1 * 0.8)),
                max_tokens = LEAST(4000, 2000 + ($1 * 2000))
            WHERE id = 1
        """, final_metric)
    finally:
        await db_pool.release(conn)

async def periodic_reflection_training(db_pool, interval_hours=12):
    while True:
        print("[INFO] Running reflection training (weighted approach).")
        await train_reflection_model(db_pool)
        await asyncio.sleep(interval_hours * 3600)
