"""
Enhanced Nyx Memory Manager - Comprehensive implementation for human-like memory systems
with metacognitive awareness, narrative construction, and advanced psychological features.
"""

import json
import logging
import os
import asyncpg
import openai
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")

logger = logging.getLogger(__name__)

class NyxMemoryManager:
    """
    Advanced memory manager for Nyx (the DM) with enhanced human-like memory features:
    - Episodic and semantic memory stores
    - Memory reconstruction and narrative formation
    - Metacognitive awareness and reflection
    - Contextual retrieval based on environment
    - Memory reconsolidation with subtle alterations
    - Interference between similar memories
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Memory retrieval cache
        self.memory_cache = {}
        self.cache_timestamp = None
        self.cache_valid_seconds = 300  # 5 minutes
        
        # Narrative structures
        self.current_narrative_arcs = []
        self.player_model = {}

    # -----------------------------------------------------------
    # MEMORY ADDITION & ORGANIZATION
    # -----------------------------------------------------------
    async def add_memory(
        self,
        memory_text: str,
        memory_type: str = "observation",
        significance: int = 3,
        emotional_intensity: int = 0,
        tags: Optional[List[str]] = None,
        related_entities: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Add a new memory with rich contextual information.
        
        Args:
            memory_text: The actual text of the memory
            memory_type: Type of memory (observation, reflection, insight, prediction)
            significance: How important this memory is (1-10)
            emotional_intensity: Emotional impact (0-100)
            tags: List of tags for categorization
            related_entities: Dict mapping entity types to IDs (player, npc, location)
            context: Additional contextual data (location, time, etc)
        
        Returns:
            The memory ID if successful, None otherwise
        """
        tags = tags or []
        related_entities = related_entities or {}
        context = context or {}
        
        # Enhance tags with content analysis
        content_tags = await self.analyze_memory_content(memory_text)
        tags.extend(content_tags)
        
        # Generate embedding for similarity search
        embedding_data = await self.generate_embedding(memory_text)
        
        # Prepare contextual data
        current_time = datetime.now()
        
        # Extract environmental context if available
        location = context.get("location", "unknown")
        time_of_day = context.get("time_of_day", "unknown")
        environmental_context = {
            "location": location,
            "time_of_day": time_of_day,
            "year": context.get("year", 1),
            "month": context.get("month", 1),
            "day": context.get("day", 1)
        }
        
        # Complexity factor - determines how detailed this memory will be stored
        complexity = min(10, significance + (emotional_intensity // 20))
        
        # Get details about related entities for richer context
        entity_context = await self.gather_entity_context(related_entities)
        
        # Combined metadata for storage
        metadata = {
            "environmental_context": environmental_context,
            "entity_context": entity_context,
            "complexity": complexity,
            "original_form": memory_text  # Save original form for tracking reconsolidation changes
        }
        
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Insert the memory with extended metadata
            memory_id = await conn.fetchval("""
                INSERT INTO NyxMemories (
                    user_id, conversation_id, memory_text, memory_type,
                    significance, embedding, timestamp,
                    tags, times_recalled, is_archived,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 0, FALSE, $9)
                RETURNING id
            """,
                self.user_id,
                self.conversation_id,
                memory_text,
                memory_type,
                significance,
                embedding_data,
                current_time,
                tags,
                json.dumps(metadata)
            )
            
            # For high significance memories, create a semantic abstraction
            if significance >= 7 and memory_type == "observation":
                await self.create_semantic_abstraction(memory_text, tags, memory_id, conn)
            
            # If this memory relates to player behavior, update player model
            if "player" in related_entities or "player_action" in tags:
                await self.update_player_model(memory_text, context, conn)
            
            # Check for narrative implications
            if significance >= 5:
                await self.evaluate_narrative_impact(memory_text, significance, tags, conn)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] add_memory error: {e}")
            return None
        finally:
            if conn:
                await conn.close()

    async def gather_entity_context(self, related_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather additional context about referenced entities to enrich the memory.
        """
        entity_context = {}
        conn = None
        
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Get player context if present
            if "player" in related_entities:
                player_row = await conn.fetchrow("""
                    SELECT player_name, corruption, confidence, willpower, 
                           obedience, dependency, lust
                    FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if player_row:
                    entity_context["player"] = dict(player_row)
            
            # Get NPC context if present
            if "npc" in related_entities:
                npc_id = related_entities["npc"]
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, dominance, cruelty, closeness, trust, respect
                    FROM NPCStats
                    WHERE npc_id = $1
                    LIMIT 1
                """, npc_id)
                
                if npc_row:
                    entity_context["npc"] = dict(npc_row)
            
            # Get location context if present
            if "location" in related_entities:
                location_name = related_entities["location"]
                loc_row = await conn.fetchrow("""
                    SELECT location_name, description
                    FROM Locations
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                    LIMIT 1
                """, location_name, self.user_id, self.conversation_id)
                
                if loc_row:
                    entity_context["location"] = dict(loc_row)
            
            return entity_context
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] gather_entity_context error: {e}")
            return {}
        finally:
            if conn:
                await conn.close()

    async def create_semantic_abstraction(self, memory_text: str, tags: List[str], source_id: int, conn):
        """
        Create a semantic memory (higher-level abstraction) from an episodic memory.
        This converts concrete experiences into generalized knowledge.
        """
        # Create a prompt for generating a semantic abstraction
        prompt = f"""
        Convert this specific observation into a general insight or pattern:
        
        Observation: {memory_text}
        
        Create a concise semantic memory that:
        1. Extracts the general principle or pattern from this specific event
        2. Forms a higher-level abstraction that could apply to similar situations
        3. Phrases it as a generalized insight rather than a specific event
        4. Keeps it under 50 words
        
        Example transformation:
        Observation: "Chase hesitated when Monica asked him about his past, changing the subject quickly."
        Semantic abstraction: "Chase appears uncomfortable discussing his past and employs deflection when questioned about it."
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI that extracts semantic meaning from specific observations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=100
            )
            
            abstraction = response.choices[0].message.content.strip()
            
            # Store the semantic memory with a reference to its source
            await conn.execute("""
                INSERT INTO NyxMemories (
                    user_id, conversation_id, memory_text, memory_type,
                    significance, embedding, timestamp,
                    tags, times_recalled, is_archived,
                    metadata
                )
                VALUES ($1, $2, $3, 'semantic', $4, $5, CURRENT_TIMESTAMP, $6, 0, FALSE, $7)
            """,
                self.user_id,
                self.conversation_id,
                abstraction,
                5,  # Moderate significance for semantic memories
                await self.generate_embedding(abstraction),
                tags + ["semantic", "abstraction"],
                json.dumps({"source_memory_id": source_id})
            )
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] create_semantic_abstraction error: {e}")

    # -----------------------------------------------------------
    # MEMORY RETRIEVAL & RECONSTRUCTION
    # -----------------------------------------------------------
    async def retrieve_memories(
        self, 
        query: str,
        memory_types: List[str] = None,
        limit: int = 10,
        min_significance: int = 0,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories with contextual weighting.
        
        Args:
            query: The query text to search for
            memory_types: Optional list of memory types to filter by
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold
            context: Optional contextual information to enhance retrieval
        
        Returns:
            List of memory objects sorted by relevance
        """
        memory_types = memory_types or ["observation", "reflection", "semantic", "insight"]
        context = context or {}
        
        # Check cache first
        cache_key = f"{query}:{'-'.join(memory_types)}:{min_significance}"
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)
            
            # Get current environmental context for contextual retrieval
            current_location = context.get("location", "")
            current_time_of_day = context.get("time_of_day", "")
            
            # Get raw memory candidates 
            rows = await conn.fetch("""
                SELECT 
                    id, memory_text, memory_type, significance, 
                    tags, times_recalled, last_recalled, timestamp,
                    metadata,
                    embedding <-> $1 AS semantic_distance
                FROM NyxMemories
                WHERE user_id = $2 
                AND conversation_id = $3
                AND is_archived = FALSE
                AND memory_type = ANY($4)
                AND significance >= $5
                ORDER BY semantic_distance
                LIMIT 50  -- Get more candidates than needed for post-filtering
            """, 
                query_embedding, 
                self.user_id, 
                self.conversation_id, 
                memory_types,
                min_significance
            )
            
            # Convert to a list of dictionaries
            memories = []
            for row in rows:
                memory = dict(row)
                
                # Parse metadata
                if memory.get("metadata"):
                    memory["metadata"] = json.loads(memory["metadata"])
                else:
                    memory["metadata"] = {}
                
                # Calculate additional relevance factors
                memory["relevance"] = await self.calculate_memory_relevance(
                    memory, 
                    semantic_distance=memory.pop("semantic_distance"),
                    query=query,
                    context=context
                )
                
                memories.append(memory)
            
            # Sort by final relevance score and limit results
            memories.sort(key=lambda x: x["relevance"], reverse=True)
            result = memories[:limit]
            
            # Before returning, record this retrieval event for the memories
            memory_ids = [m["id"] for m in result]
            await self.update_retrieval_stats(memory_ids, conn)
            
            # Cache the result
            self.add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] retrieve_memories error: {e}")
            return []
        finally:
            if conn:
                await conn.close()

    async def calculate_memory_relevance(
        self,
        memory: Dict[str, Any],
        semantic_distance: float,
        query: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate an overall relevance score for a memory considering multiple factors.
        Lower score = more relevant.
        """
        # Start with embedding similarity (already the primary sort)
        relevance = semantic_distance * 10  # Scale up for comparison with other factors
        
        # Adjust for memory significance
        # More significant memories should be boosted
        significance_factor = 1.0 - (memory["significance"] / 10.0)  # 0.0-0.9 
        relevance *= (1 + significance_factor * 0.3)  # Modest adjustment
        
        # Adjust for recency (for episodic memories)
        if memory["memory_type"] == "observation":
            # Calculate how long ago this memory was formed
            days_old = (datetime.now() - memory["timestamp"]).days
            # Recency bonus if less than 7 days old, penalty if older
            recency_factor = min(1.0, max(-0.5, 1.0 - (days_old / 7.0)))
            relevance *= (1 - recency_factor * 0.2)
        
        # Boost for frequently recalled memories
        times_recalled = memory.get("times_recalled", 0)
        if times_recalled > 0:
            recall_boost = min(0.3, times_recalled / 10.0)  # Cap at 0.3
            relevance *= (1 - recall_boost)
        
        # Context-based boosting 
        metadata = memory.get("metadata", {})
        env_context = metadata.get("environmental_context", {})
        
        # Location context match
        current_location = context.get("location", "")
        memory_location = env_context.get("location", "")
        if current_location and memory_location and current_location == memory_location:
            relevance *= 0.8  # 20% boost for same location
        
        # Time of day context match
        current_tod = context.get("time_of_day", "")
        memory_tod = env_context.get("time_of_day", "")
        if current_tod and memory_tod and current_tod == memory_tod:
            relevance *= 0.9  # 10% boost for same time of day
            
        # Emotional state match would go here (if implemented)
        
        # Tag matching - direct keyword relevance 
        query_words = set(query.lower().split())
        for tag in memory.get("tags", []):
            if tag.lower() in query_words:
                relevance *= 0.7  # 30% boost for each matching tag
        
        # Return inverse of score so higher is better
        return 1.0 / max(0.001, relevance)

    async def update_retrieval_stats(self, memory_ids: List[int], conn):
        """
        Update the retrieval statistics for accessed memories.
        Each retrieval reinforces the memory.
        """
        if not memory_ids:
            return
            
        try:
            # Update each memory with new recall count and timestamp
            for memory_id in memory_ids:
                await conn.execute("""
                    UPDATE NyxMemories
                    SET times_recalled = times_recalled + 1,
                        last_recalled = CURRENT_TIMESTAMP,
                        significance = LEAST(significance + 0.2, 10)
                    WHERE id = $1
                """, memory_id)
        except Exception as e:
            logger.error(f"[NyxMemoryManager] update_retrieval_stats error: {e}")

    async def construct_narrative(
        self, 
        topic: str, 
        context: Dict[str, Any] = None,
        limit: int = 5,
        require_chronological: bool = True
    ) -> Dict[str, Any]:
        """
        Construct a coherent narrative from related memories.
        This simulates how humans construct stories from memory fragments.
        
        Args:
            topic: The topic to construct a narrative about
            context: Optional context information to enhance retrieval
            limit: Maximum number of memories to incorporate
            require_chronological: Whether to enforce chronological ordering
            
        Returns:
            A narrative object with processed story and source memories
        """
        context = context or {}
        
        # 1. Retrieve relevant memories
        memories = await self.retrieve_memories(
            query=topic,
            memory_types=["observation", "semantic", "reflection"],
            limit=limit,
            min_significance=2,
            context=context
        )
        
        if not memories:
            return {
                "narrative": f"I don't have any significant memories about {topic}.",
                "sources": [],
                "confidence": 0.2
            }
        
        # 2. Sort chronologically if required
        if require_chronological:
            memories.sort(key=lambda x: x["timestamp"])
            
        # 3. Extract memory texts and calculate narrative confidence
        memory_texts = [m["memory_text"] for m in memories]
        source_ids = [m["id"] for m in memories]
        
        # Calculate overall confidence based on memory significance and recall frequency
        avg_significance = sum(m["significance"] for m in memories) / len(memories)
        avg_recalled = sum(m.get("times_recalled", 0) for m in memories) / len(memories)
        
        # Confidence is higher when based on significant, frequently recalled memories
        base_confidence = min(0.9, (avg_significance / 10.0) * 0.7 + (min(1.0, avg_recalled / 5.0) * 0.3))
        
        # 4. Adjust for internal consistency among memories
        consistency_factor = await self.evaluate_memory_consistency(memories)
        final_confidence = base_confidence * consistency_factor
        
        # 5. Generate the narrative (either using GPT or a template approach)
        narrative = await self.generate_narrative_from_memories(memories, topic, final_confidence)
        
        # Metacognitive awareness - include confidence and sources
        return {
            "narrative": narrative,
            "sources": source_ids,
            "confidence": final_confidence,
            "memory_count": len(memories)
        }

    async def evaluate_memory_consistency(self, memories: List[Dict[str, Any]]) -> float:
        """
        Evaluate how consistent the memories are with each other.
        This affects the confidence in the constructed narrative.
        
        Returns a factor between 0.5 and 1.0 (1.0 = completely consistent)
        """
        if len(memories) <= 1:
            return 1.0  # Single memory is consistent with itself
            
        # Simplified approach: check for time consistency and contradictions
        timestamps = [m["timestamp"] for m in memories]
        timestamp_order = sorted(timestamps)
        
        # If timestamps aren't in order, that's a consistency issue
        if timestamps != timestamp_order:
            return 0.9  # Small penalty for timestamp inconsistency
            
        # More sophisticated implementations would check for contradictory content
        # or inconsistent entity attributes across memories
        
        return 1.0  # Default: assume consistent

    async def generate_narrative_from_memories(
        self, 
        memories: List[Dict[str, Any]], 
        topic: str,
        confidence: float
    ) -> str:
        """
        Generate a coherent narrative text from a set of memories.
        Uses GPT to weave memories into a story with appropriate
        confidence markers and metacognitive awareness.
        """
        # Format memories for the prompt
        memory_list = []
        for i, memory in enumerate(memories):
            date_str = memory["timestamp"].strftime("%Y-%m-%d")
            memory_list.append(f"Memory {i+1} [{date_str}]: {memory['memory_text']}")
        
        memories_text = "\n".join(memory_list)
        
        # Prepare confidence expression
        confidence_phrase = "I'm certain" if confidence > 0.8 else \
                            "I believe" if confidence > 0.6 else \
                            "I think" if confidence > 0.4 else \
                            "I vaguely recall"
        
        # Create the prompt
        prompt = f"""
        As Nyx, construct a coherent narrative about "{topic}" based on these memories:
        
        {memories_text}
        
        Confidence level: {confidence:.2f}
        
        1. Begin the narrative with "{confidence_phrase}" or similar confidence marker appropriate to the confidence level.
        2. Weave the memories into a coherent story, filling minimal gaps as needed.
        3. If memories seem contradictory, acknowledge the uncertainty.
        4. Keep it concise (under 200 words).
        5. Write in first person as Nyx.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Nyx, constructing a narrative from memories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            narrative = response.choices[0].message.content.strip()
            return narrative
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] generate_narrative_from_memories error: {e}")
            return f"{confidence_phrase} the following about {topic}: " + " ".join([m["memory_text"] for m in memories[:3]])

    # -----------------------------------------------------------
    # MEMORY MAINTENANCE & LIFECYCLE
    # -----------------------------------------------------------
    async def consolidate_memories(self):
        """
        Consolidate related memories into higher-level semantic memories.
        This simulates how episodic memories transform into semantic knowledge over time.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Find clusters of related memories to consolidate
            memory_rows = await conn.fetch("""
                SELECT id, memory_text, tags, embedding
                FROM NyxMemories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND memory_type = 'observation'
                AND is_archived = FALSE
                AND timestamp < NOW() - INTERVAL '3 days'
                AND times_recalled >= 2
            """, self.user_id, self.conversation_id)
            
            if not memory_rows:
                return
                
            # Convert to easily workable format
            memories = []
            for row in memory_rows:
                memories.append({
                    "id": row["id"],
                    "text": row["memory_text"],
                    "tags": row["tags"],
                    "embedding": row["embedding"],
                })
            
            # Find clusters using embedding similarity
            clusters = self.cluster_memories_by_similarity(memories)
            logger.info(f"[NyxMemoryManager] Found {len(clusters)} memory clusters for consolidation")
            
            # For each significant cluster, create a consolidated memory
            for cluster in clusters:
                if len(cluster) >= 3:  # Only consolidate clusters with several memories
                    cluster_ids = [m["id"] for m in cluster]
                    cluster_texts = [m["text"] for m in cluster]
                    
                    # Extract all tags from the cluster
                    all_tags = []
                    for m in cluster:
                        all_tags.extend(m.get("tags", []))
                    unique_tags = list(set(all_tags))
                    
                    # Generate a consolidated summary
                    summary = await self.generate_consolidated_summary(cluster_texts)
                    
                    # Store the consolidated memory
                    consolidated_id = await conn.fetchval("""
                        INSERT INTO NyxMemories (
                            user_id, conversation_id, memory_text, memory_type,
                            significance, embedding, timestamp,
                            tags, times_recalled, is_archived,
                            metadata
                        )
                        VALUES ($1, $2, $3, 'consolidated', 6, $4, CURRENT_TIMESTAMP, 
                                $5, 0, FALSE, $6)
                        RETURNING id
                    """,
                        self.user_id,
                        self.conversation_id,
                        summary,
                        await self.generate_embedding(summary),
                        unique_tags + ["consolidated"],
                        json.dumps({"source_memory_ids": cluster_ids})
                    )
                    
                    # Update the original memories to mark them as consolidated
                    # But don't archive them yet - they're still valuable
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET is_consolidated = TRUE
                        WHERE id = ANY($1)
                    """, cluster_ids)
                    
                    logger.info(f"[NyxMemoryManager] Created consolidated memory {consolidated_id} from {len(cluster_ids)} memories")
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] consolidate_memories error: {e}")
        finally:
            if conn:
                await conn.close()

    def cluster_memories_by_similarity(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group memories into clusters based on embedding similarity.
        This is a simplified version of clustering.
        """
        # We'll use a greedy approach for simplicity
        clusters = []
        unclustered = memories.copy()
        
        while unclustered:
            # Take the first memory as a seed
            seed = unclustered.pop(0)
            current_cluster = [seed]
            
            # Find all similar memories
            i = 0
            while i < len(unclustered):
                memory = unclustered[i]
                
                # Calculate cosine similarity
                similarity = np.dot(seed["embedding"], memory["embedding"])
                if similarity > 0.85:  # Threshold for similarity
                    current_cluster.append(memory)
                    unclustered.pop(i)
                else:
                    i += 1
            
            # If we found a significant cluster, add it
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        return clusters

    async def generate_consolidated_summary(self, memory_texts: List[str]) -> str:
        """
        Generate a consolidated summary of related memories.
        """
        joined_texts = "\n".join(memory_texts)
        
        prompt = f"""
        Consolidate these related memory fragments into a single coherent memory:
        
        {joined_texts}
        
        Create a single paragraph that:
        1. Captures the essential pattern or theme across these memories
        2. Generalizes the specific details into broader understanding
        3. Retains the most significant elements
        4. Begins with "I've observed that..." or similar phrase
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You consolidate memory fragments into coherent patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[NyxMemoryManager] generate_consolidated_summary error: {e}")
            # Fallback
            return f"I've observed a pattern across several memories: {memory_texts[0]}..."

    async def apply_memory_decay(self):
        """
        Apply decay to memories based on age, significance, and recall frequency.
        This simulates how human memories fade over time, especially less important ones.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Get memories that haven't been recalled recently
            memory_rows = await conn.fetch("""
                SELECT id, significance, times_recalled, 
                       EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400 AS days_old,
                       EXTRACT(EPOCH FROM (NOW() - COALESCE(last_recalled, timestamp))) / 86400 AS days_since_recall
                FROM NyxMemories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND is_archived = FALSE
                AND memory_type = 'observation'  -- Only decay episodic memories
            """, self.user_id, self.conversation_id)
            
            for row in memory_rows:
                memory_id = row["id"]
                significance = row["significance"]
                times_recalled = row["times_recalled"]
                days_old = row["days_old"]
                days_since_recall = row["days_since_recall"]
                
                # Calculate decay factors
                age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
                recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
                
                # How much to reduce significance
                # Memories decay faster if they're old AND haven't been recalled recently
                decay_rate = 0.1 * age_factor * recall_factor
                if days_since_recall > 7:
                    decay_rate *= 1.5  # Extra decay for memories not recalled in a week
                
                # Apply decay with a floor of 1 for significance
                new_significance = max(1.0, significance - decay_rate)
                
                # If significance drops below threshold, archive the memory
                if new_significance < 2.0 and days_old > 14:
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET is_archived = TRUE
                        WHERE id = $1
                    """, memory_id)
                else:
                    # Otherwise just update the significance
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET significance = $1
                        WHERE id = $2
                    """, new_significance, memory_id)
            
            logger.info(f"[NyxMemoryManager] Memory decay applied to {len(memory_rows)} memories")
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] apply_memory_decay error: {e}")
        finally:
            if conn:
                await conn.close()
                
    async def reconsolidate_memory(self, memory_id: int, context: Dict[str, Any] = None):
        """
        Reconsolidate (slightly alter) a memory when it's recalled.
        This simulates how human memories change slightly each time they're accessed.
        
        Args:
            memory_id: The ID of the memory to reconsolidate
            context: Current context (emotional state, etc.) that might influence reconsolidation
        """
        context = context or {}
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Fetch the memory
            row = await conn.fetchrow("""
                SELECT memory_text, metadata, significance, memory_type, embedding
                FROM NyxMemories
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """, memory_id, self.user_id, self.conversation_id)
            
            if not row:
                return
                
            memory_text = row["memory_text"]
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            significance = row["significance"]
            memory_type = row["memory_type"]
            
            # Only reconsolidate episodic memories with low/medium significance
            # High significance memories are more stable
            if memory_type != "observation" or significance >= 8:
                return
                
            # Get original form if available
            original_form = metadata.get("original_form", memory_text)
            
            # Current emotional state can influence reconsolidation
            current_emotion = context.get("emotional_state", "neutral")
            
            # Reconsolidation varies by memory age
            reconsolidation_strength = min(0.3, significance / 10.0)  # Cap at 0.3
            
            # Generate a slightly altered version
            altered_memory = await self.alter_memory_text(
                memory_text, 
                original_form,
                reconsolidation_strength,
                current_emotion
            )
            
            # Update metadata to track changes
            if "reconsolidation_history" not in metadata:
                metadata["reconsolidation_history"] = []
                
            metadata["reconsolidation_history"].append({
                "previous_text": memory_text,
                "timestamp": datetime.now().isoformat(),
                "emotional_context": current_emotion
            })
            
            # Only store last 3 versions to avoid metadata bloat
            if len(metadata["reconsolidation_history"]) > 3:
                metadata["reconsolidation_history"] = metadata["reconsolidation_history"][-3:]
            
            # Update the memory
            await conn.execute("""
                UPDATE NyxMemories
                SET memory_text = $1, metadata = $2, embedding = $3
                WHERE id = $4
            """, 
                altered_memory, 
                json.dumps(metadata),
                await self.generate_embedding(altered_memory),
                memory_id
            )
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] reconsolidate_memory error: {e}")
        finally:
            if conn:
                await conn.close()
                
    async def alter_memory_text(
        self, 
        memory_text: str, 
        original_form: str,
        alteration_strength: float,
        emotional_context: str
    ) -> str:
        """
        Slightly alter a memory text based on emotional context.
        The closer to the original form, the less alteration.
        """
        # For minimal changes, we'll use GPT with a specific prompt
        prompt = f"""
        Slightly alter this memory to simulate memory reconsolidation effects.
        
        Original memory: {original_form}
        Current memory: {memory_text}
        Emotional context: {emotional_context}
        
        Create a very slight alteration that:
        1. Maintains the same core information and meaning
        2. Makes subtle changes to wording or emphasis ({int(alteration_strength * 100)}% alteration)
        3. Slightly enhances aspects that align with the "{emotional_context}" emotional state
        4. Never changes key facts, names, or locations
        
        Return only the altered text with no explanation.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You subtly alter memories to simulate reconsolidation effects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            altered_text = response.choices[0].message.content.strip()
            return altered_text
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] alter_memory_text error: {e}")
            # If GPT fails, make minimal random changes
            words = memory_text.split()
            for i in range(len(words)):
                if random.random() < alteration_strength * 0.2:
                    # Minimal changes like adding "very" or changing emphasis words
                    if words[i] in ["a", "the", "was", "is"]:
                        continue  # Skip essential words
                    if "good" in words[i]:
                        words[i] = "very " + words[i]
                    elif "bad" in words[i]:
                        words[i] = "quite " + words[i]
            
            return " ".join(words)
            
    # -----------------------------------------------------------
    # PLAYER MODELING & NARRATIVE PLANNING
    # -----------------------------------------------------------
    async def update_player_model(self, memory_text: str, context: Dict[str, Any], conn):
        """
        Update Nyx's model of the player based on observed behavior.
        This powers adaptative narrative and personalization.
        """
        # Extract player action information
        action_types = self.detect_player_action_types(memory_text)
        
        if not action_types:
            return
            
        # Get current player model or initialize new one
        row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxPlayerModel'
        """, self.user_id, self.conversation_id)
        
        if row and row["value"]:
            player_model = json.loads(row["value"])
        else:
            # Initialize with default values
            player_model = {
                "play_style": {
                    "aggressive": 0,
                    "cautious": 0,
                    "curious": 0,
                    "submissive": 0,
                    "dominant": 0
                },
                "interests": {},
                "narrative_preferences": {
                    "action": 0,
                    "romance": 0,
                    "intrigue": 0,
                    "exploration": 0
                },
                "recurring_choices": {},
                "key_decisions": [],
                "response_to_femdom": 0,  # -100 to 100 scale
                "comfort_zones": {
                    "physical": 0,  # 0-100 scale
                    "emotional": 0,
                    "power_dynamics": 0
                }
            }
        
        # Update player model based on action types
        for action_type in action_types:
            # Update play style
            if action_type == "aggressive":
                player_model["play_style"]["aggressive"] += 1
            elif action_type == "cautious":
                player_model["play_style"]["cautious"] += 1
            elif action_type == "curious":
                player_model["play_style"]["curious"] += 1
            elif action_type == "submissive":
                player_model["play_style"]["submissive"] += 1
                # Also update power dynamics comfort
                player_model["comfort_zones"]["power_dynamics"] = min(100, player_model["comfort_zones"]["power_dynamics"] + 2)
                # And femdom response
                player_model["response_to_femdom"] = min(100, player_model["response_to_femdom"] + 3)
            elif action_type == "dominant":
                player_model["play_style"]["dominant"] += 1
                # Negative adjustment to femdom response
                player_model["response_to_femdom"] = max(-100, player_model["response_to_femdom"] - 3)
            
            # For other categories like interests, narrative preferences, etc.
            # We would add similar logic based on detected categories
        
        # Check for recurring choices
        if "player_choice" in context:
            choice = context["player_choice"]
            if choice in player_model["recurring_choices"]:
                player_model["recurring_choices"][choice] += 1
            else:
                player_model["recurring_choices"][choice] = 1
        
        # Add key decisions if this is a significant choice
        if context.get("is_key_decision", False):
            player_model["key_decisions"].append({
                "decision": memory_text,
                "context": context.get("decision_context", ""),
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 10 key decisions
            player_model["key_decisions"] = player_model["key_decisions"][-10:]
        
        # Save updated model
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'NyxPlayerModel', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = $3
        """, self.user_id, self.conversation_id, json.dumps(player_model))

    def detect_player_action_types(self, text: str) -> List[str]:
        """
        Analyze a memory text to detect player action types.
        Simple keyword-based approach for demonstration.
        """
        text_lower = text.lower()
        action_types = []
        
        # Simple keyword matching
        if any(word in text_lower for word in ["attack", "fight", "confront", "demand"]):
            action_types.append("aggressive")
            
        if any(word in text_lower for word in ["wait", "hesitate", "careful", "cautious"]):
            action_types.append("cautious")
            
        if any(word in text_lower for word in ["ask", "question", "explore", "investigate"]):
            action_types.append("curious")
            
        if any(word in text_lower for word in ["obey", "follow", "agree", "accept", "submit"]):
            action_types.append("submissive")
            
        if any(word in text_lower for word in ["command", "lead", "control", "direct"]):
            action_types.append("dominant")
        
        return action_types

    async def evaluate_narrative_impact(self, memory_text: str, significance: int, tags: List[str], conn):
        """
        Evaluate how a new memory impacts ongoing narrative arcs.
        Updates Nyx's narrative plans based on player actions.
        """
        # Only process significant memories
        if significance < 5:
            return
            
        # Get current narrative arcs
        row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
        """, self.user_id, self.conversation_id)
        
        if row and row["value"]:
            narrative_arcs = json.loads(row["value"])
        else:
            # Initialize with default structure
            narrative_arcs = {
                "active_arcs": [],
                "completed_arcs": [],
                "planned_arcs": [],
                "narrative_adaption_history": []
            }
        
        # Check each active arc for impact
        for arc in narrative_arcs["active_arcs"]:
            arc_keywords = arc.get("keywords", [])
            arc_npcs = arc.get("involved_npcs", [])
            
            # Simple impact detection - keyword overlap
            impact_score = 0
            for keyword in arc_keywords:
                if keyword.lower() in memory_text.lower():
                    impact_score += 1
            
            # Tag overlap
            for tag in tags:
                if tag in arc_keywords:
                    impact_score += 1
            
            # If significant impact detected
            if impact_score >= 2:
                # Update arc progress
                if "progress" not in arc:
                    arc["progress"] = 0
                arc["progress"] += min(25, significance * 5)  # Cap at 25% progress per event
                
                # Record impact
                if "key_events" not in arc:
                    arc["key_events"] = []
                arc["key_events"].append({
                    "memory_text": memory_text,
                    "impact_score": impact_score,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for arc completion
                if arc["progress"] >= 100:
                    arc["status"] = "completed"
                    arc["completion_date"] = datetime.now().isoformat()
                    narrative_arcs["completed_arcs"].append(arc)
                    narrative_arcs["active_arcs"].remove(arc)
                    
                    # Add record of adaptation
                    narrative_arcs["narrative_adaption_history"].append({
                        "event": f"Arc completed: {arc['name']}",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Activate a planned arc if available
                    if narrative_arcs["planned_arcs"]:
                        new_arc = narrative_arcs["planned_arcs"].pop(0)
                        new_arc["status"] = "active"
                        new_arc["start_date"] = datetime.now().isoformat()
                        narrative_arcs["active_arcs"].append(new_arc)
                        
                        narrative_arcs["narrative_adaption_history"].append({
                            "event": f"New arc activated: {new_arc['name']}",
                            "timestamp": datetime.now().isoformat()
                        })
        
        # Save updated narrative arcs
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'NyxNarrativeArcs', $3)
            ON CONFLICT (user_id, conversation_id, key) 
            DO UPDATE SET value = $3
        """, self.user_id, self.conversation_id, json.dumps(narrative_arcs))

    # -----------------------------------------------------------
    # METACOGNITIVE FUNCTIONS
    # -----------------------------------------------------------
    async def generate_introspection(self) -> Dict[str, Any]:
        """
        Generate Nyx's introspection about her own memory and knowledge.
        This adds metacognitive awareness to the DM character.
        """
        conn = None
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Count memories by type and significance
            stats = await conn.fetch("""
                SELECT 
                    memory_type, 
                    COUNT(*) as count,
                    AVG(significance) as avg_significance,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
                GROUP BY memory_type
            """, self.user_id, self.conversation_id)
            
            # Get most frequently recalled memories
            top_memories = await conn.fetch("""
                SELECT memory_text, times_recalled
                FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
                ORDER BY times_recalled DESC
                LIMIT 3
            """, self.user_id, self.conversation_id)
            
            # Get player model
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxPlayerModel'
            """, self.user_id, self.conversation_id)
            
            player_model = json.loads(row["value"]) if row and row["value"] else {}
            
            # Calculate memory health metrics
            memory_health = {
                "total_memories": sum(r["count"] for r in stats),
                "episodic_ratio": next((r["count"] for r in stats if r["memory_type"] == "observation"), 0) / 
                                sum(r["count"] for r in stats) if sum(r["count"] for r in stats) > 0 else 0,
                "average_significance": sum(r["count"] * r["avg_significance"] for r in stats) / 
                                      sum(r["count"] for r in stats) if sum(r["count"] for r in stats) > 0 else 0,
                "memory_span_days": (max(r["newest"] for r in stats) - min(r["oldest"] for r in stats)).days 
                                   if stats else 0,
                "top_recalled_memories": [{"text": m["memory_text"], "recalled": m["times_recalled"]} for m in top_memories]
            }
            
            # Generate introspection text using GPT
            introspection = await self.generate_introspection_text(memory_health, player_model)
            
            # Return combination of metrics and generated text
            return {
                "memory_stats": {r["memory_type"]: {"count": r["count"], "avg_significance": r["avg_significance"]} for r in stats},
                "memory_health": memory_health,
                "player_understanding": player_model.get("play_style", {}),
                "introspection": introspection,
                "confidence": min(1.0, memory_health["total_memories"] / 100)  # More memories = more confidence
            }
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] generate_introspection error: {e}")
            return {
                "introspection": "I'm having trouble accessing my memories right now.",
                "confidence": 0.1
            }
        finally:
            if conn:
                await conn.close()
                
    async def generate_introspection_text(self, memory_health: Dict[str, Any], player_model: Dict[str, Any]) -> str:
        """
        Generate natural language introspection about Nyx's own memory state.
        """
        # Format key metrics for the prompt
        memory_count = memory_health["total_memories"]
        span_days = memory_health["memory_span_days"]
        avg_sig = memory_health["average_significance"]
        
        # Format player understanding
        play_style = player_model.get("play_style", {})
        play_style_str = ", ".join([f"{style}: {count}" for style, count in play_style.items() if count > 0])
        
        # Top recalled memories
        top_memories = memory_health.get("top_recalled_memories", [])
        top_memories_str = "\n".join([f"- {m['text']} (recalled {m['recalled']} times)" for m in top_memories])
        
        prompt = f"""
        As Nyx, generate an introspective reflection on your memory state using these metrics:
        
        - Total memories: {memory_count}
        - Memory span: {span_days} days
        - Average significance: {avg_sig:.1f}/10
        - Player tendencies: {play_style_str}
        
        Most frequently recalled memories:
        {top_memories_str}
        
        Create a first-person introspection that:
        1. Reflects on your understanding of Chase (the player)
        2. Notes any gaps or uncertainties in your knowledge
        3. Acknowledges how your perspective might be biased or incomplete
        4. Expresses metacognitive awareness of your role as the narrative guide
        
        Keep it natural and conversational, as if you're thinking to yourself.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Nyx, the dungeon master, reflecting on your memories and understanding."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"[NyxMemoryManager] generate_introspection_text error: {e}")
            
            # Fallback template if GPT fails
            return f"""
            I've accumulated {memory_count} memories about Chase and our interactions over {span_days} days.
            I notice that he tends to be {max(play_style.items(), key=lambda x: x[1])[0]} in his approach.
            My understanding feels {'strong' if avg_sig > 7 else 'moderate' if avg_sig > 4 else 'limited'}, 
            though I wonder what I might be missing or misinterpreting.
            """

    # -----------------------------------------------------------
    # UTILITY FUNCTIONS
    # -----------------------------------------------------------
    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        """
        Analyze memory content for automatic tagging.
        """
        tags = []
        # Simple keyword-based tagging
        text_lower = memory_text.lower()
        
        if "chase" in text_lower or "player" in text_lower:
            tags.append("player_related")
            
        if any(word in text_lower for word in ["angry", "upset", "furious", "annoyed"]):
            tags.append("negative_emotion")
            
        if any(word in text_lower for word in ["happy", "pleased", "excited", "joyful"]):
            tags.append("positive_emotion")
            
        if any(word in text_lower for word in ["confused", "unsure", "uncertain"]):
            tags.append("uncertainty")
            
        # Check for action types
        if any(word in text_lower for word in ["attack", "fight", "confront"]):
            tags.append("aggressive_action")
            
        if any(word in text_lower for word in ["help", "assist", "support"]):
            tags.append("helpful_action")
            
        if any(word in text_lower for word in ["romance", "flirt", "kiss"]):
            tags.append("romantic")
            
        # Add more patterns as needed
            
        return tags

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a text using OpenAI's API.
        """
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"[NyxMemoryManager] generate_embedding error: {e}")
            # Return zeros vector as fallback
            return [0.0] * 1536  # ada-002 uses 1536 dimensions

    # Cache management functions
    def add_to_cache(self, key: str, value: Any):
        """Add a result to the in-memory cache with timestamp."""
        self.memory_cache[key] = value
        self.cache_timestamp = datetime.now()
        
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get a result from cache if it exists and is not expired."""
        if not self.cache_timestamp:
            return None
            
        # Check cache freshness
        elapsed_seconds = (datetime.now() - self.cache_timestamp).total_seconds()
        if elapsed_seconds > self.cache_valid_seconds:
            return None
            
        return self.memory_cache.get(key)
        
    def clear_cache(self):
        """Clear the memory cache."""
        self.memory_cache = {}
        self.cache_timestamp = None

# Main maintenance function for scheduled execution
async def perform_memory_maintenance(user_id: int, conversation_id: int):
    """
    Perform regular memory maintenance tasks:
    - Consolidation of related memories
    - Applying memory decay
    - Archiving old memories
    
    Should be run periodically (e.g., daily)
    """
    memory_manager = NyxMemoryManager(user_id, conversation_id)
    
    # Apply memory decay
    await memory_manager.apply_memory_decay()
    
    # Consolidate memories
    await memory_manager.consolidate_memories()
    
    logger.info(f"[MemoryMaintenance] Completed for user_id={user_id}, conversation_id={conversation_id}")
