# memory/semantic.py

import logging
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
import asyncio

# Import the OpenAI helper functions from chatgpt_integration

from .connection import with_transaction, TransactionContext
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_semantic")

def _parse_json_block(text: str) -> Any:
    """Strip ```json fences and parse JSON; return None on failure."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```json") and t.endswith("```"):
        t = t[7:-3].strip()
    elif t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    try:
        return json.loads(t)
    except Exception:
        return None

def _safe_block(txt: str) -> str:
    """
    Wrap untrusted text in a triple-quoted block for LLM prompts and
    neuter any embedded triple quotes to reduce prompt injection risk.
    """
    if txt is None:
        txt = ""
    # collapse backticks that models sometimes treat as code fences
    # (optional; remove if undesired)
    txt = txt.replace("```", "¸¸¸")  # harmless chars
    # escape literal triple double-quotes so we can safely embed
    txt = txt.replace('"""', '\\"""')
    return f'"""\n{txt}\n"""'

class SemanticMemoryManager:
    """
    Manages semantic memory generation and organization.
    Transforms concrete episodic memories into abstract knowledge and beliefs.
    
    Features:
    - Pattern recognition across similar memories
    - Belief formation based on experiences
    - Generalized knowledge extraction
    - Semantic network building
    - Counterfactual reasoning
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def generate_semantic_memory(self,
                                     source_memory_id: int,
                                     entity_type: str,
                                     entity_id: int,
                                     abstraction_level: float = 0.5,
                                     conn = None) -> Dict[str, Any]:
        """
        Generate a semantic memory (abstract knowledge) from an episodic memory.
        
        Args:
            source_memory_id: ID of the source episodic memory
            entity_type: Type of entity that owns the memory
            entity_id: ID of the entity
            abstraction_level: How abstract the semantic memory should be (0.0-1.0)
            
        Returns:
            Information about the created semantic memory
        """
        # Fetch the source memory
        row = await conn.fetchrow("""
            SELECT memory_text, tags, significance, timestamp
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, source_memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Source memory {source_memory_id} not found"}
            
        memory_text = row["memory_text"]
        tags = row["tags"] or []
        # Handle tags if stored as JSON string
        if isinstance(tags, str):
            try:
                tags = json.loads(tags) or []
            except Exception:
                tags = []
        significance = row["significance"]
        timestamp = row["timestamp"]
        
        # Generate semantic abstraction
        abstraction = await self._generate_semantic_abstraction(
            memory_text, abstraction_level
        )
        
        # Create memory manager for this entity
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Create the semantic memory
        # Handle significance properly if it's an enum
        try:
            medium_val = MemorySignificance.MEDIUM.value
        except AttributeError:
            medium_val = MemorySignificance.MEDIUM
        new_significance = max(medium_val, (significance - 1))
            
        semantic_memory = Memory(
            text=abstraction,
            memory_type=MemoryType.SEMANTIC,
            significance=new_significance,
            emotional_intensity=0,  # Semantic memories are emotionally neutral
            tags=tags + ["semantic", "abstraction"],
            metadata={
                "source_memory_id": source_memory_id,
                "abstraction_level": abstraction_level,
                "generated_at": datetime.now().isoformat()
            },
            timestamp=timestamp  # Use same timestamp as source for chronology
        )
        
        semantic_id = await memory_manager.add_memory(semantic_memory, conn=conn)
        
        return {
            "semantic_memory_id": semantic_id,
            "source_memory_id": source_memory_id,
            "abstraction": abstraction,
            "abstraction_level": abstraction_level
        }

    def _strip_markdown_and_parse_json(self, text: str) -> Any:
        return _parse_json_block(text)
    
    async def _generate_semantic_abstraction(
        self,
        memory_text: str,
        abstraction_level: float,
    ) -> str:
        """Create a semantic abstraction (minimal/moderate/high) via OpenAI API."""
        from logic.chatgpt_integration import get_async_openai_client, get_openai_client
        client = get_async_openai_client()
        level = (
            "minimal" if abstraction_level < 0.3
            else "high" if abstraction_level > 0.7
            else "moderate"
        )
    
        obs_block = _safe_block(memory_text)
        prompt = (
            f"Observation:\n{obs_block}\n\n"
            f"Produce a {level} abstraction (≤ 50 words) capturing the general insight.\n"
            "Return ONLY the abstraction."
        )
    
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You extract semantic abstractions from memories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=100
            )
            
            txt = response.choices[0].message.content.strip()
            if txt:
                return txt
            logger.warning("Empty response from OpenAI for semantic abstraction")
        except Exception as e:
            logger.error("Semantic abstraction failed: %s", e)
    
        # fallback
        return (
            f"General pattern observed: {memory_text[:100]}..."
            if len(memory_text) > 100 else f"General pattern observed: {memory_text}"
        )
        
    @with_transaction
    async def find_patterns_across_memories(self,
                                         entity_type: str,
                                         entity_id: int,
                                         topic: Optional[str] = None,
                                         min_memories: int = 3,
                                         conn = None) -> Dict[str, Any]:
        """
        Find patterns across multiple episodic memories and generate higher-level semantic insights.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            topic: Optional topic to focus on
            min_memories: Minimum number of memories needed to find patterns
            
        Returns:
            Generated pattern insights
        """
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Retrieve episodic memories
        query_params = {
            "memory_types": ["observation", "episodic"],
            "min_significance": MemorySignificance.LOW,
            "limit": 20,
            "conn": conn
        }
        
        if topic:
            query_params["query"] = topic
            
        episodic_memories = await memory_manager.retrieve_memories(**query_params)
        
        if len(episodic_memories) < min_memories:
            return {
                "error": f"Not enough memories to find patterns (found {len(episodic_memories)}, need {min_memories})"
            }
            
        # Group memories by similarity
        clusters = await self._cluster_memories_by_similarity(episodic_memories)
        
        # Find patterns in each cluster
        patterns = []
        for cluster in clusters:
            if len(cluster) >= min_memories:
                pattern = await self._extract_pattern_from_cluster(cluster, topic)
                if pattern:
                    # Store the pattern as a semantic memory
                    semantic_memory = Memory(
                        text=pattern["pattern"],
                        memory_type=MemoryType.SEMANTIC,
                        significance=MemorySignificance.HIGH,
                        emotional_intensity=0,
                        tags=["semantic", "pattern", "insight"] + (["topic_" + topic] if topic else []),
                        metadata={
                            "source_memory_ids": [m.id for m in cluster],
                            "pattern_confidence": pattern["confidence"],
                            "generated_at": datetime.now().isoformat()
                        },
                        timestamp=datetime.now()
                    )
                    
                    pattern_id = await memory_manager.add_memory(semantic_memory, conn=conn)
                    pattern["memory_id"] = pattern_id
                    patterns.append(pattern)
        
        return {
            "patterns_found": len(patterns),
            "patterns": patterns,
            "memories_analyzed": len(episodic_memories),
            "clusters_found": len(clusters)
        }
       
    async def _cluster_memories_by_similarity(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Group memories into clusters by semantic similarity.
        """
        if not memories:
            return []
            
        # Simple approach: use embedding similarity
        clusters = []
        unprocessed = memories.copy()
        
        while unprocessed:
            seed = unprocessed.pop(0)
            current_cluster = [seed]
            seed_embedding = seed.embedding
            
            if seed_embedding is None:
                continue  # Skip memories without embeddings
                
            i = 0
            while i < len(unprocessed):
                memory = unprocessed[i]
                memory_embedding = memory.embedding
                
                if memory_embedding is None:
                    i += 1
                    continue
                
                # Calculate cosine similarity
                similarity = 0
                try:
                    import numpy as np
                    vec1 = np.array(seed_embedding)
                    vec2 = np.array(memory_embedding)
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")
                    i += 1
                    continue
                
                if similarity > 0.8:  # High similarity threshold
                    current_cluster.append(memory)
                    unprocessed.pop(i)
                else:
                    i += 1
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        return clusters
    
    async def _extract_pattern_from_cluster(
        self,
        cluster: List[Memory],
        topic: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract a pattern from a cluster of Memory objects."""
        from logic.chatgpt_integration import get_async_openai_client, get_openai_client
        client = get_async_openai_client()
    
        memories_text = "\n".join(f"- {m.text}" for m in cluster)
        mem_block = _safe_block(memories_text)
        topic_line = f"TOPIC FOCUS:\n{_safe_block(topic)}\n" if topic else ""
        
        prompt = (
            f"MEMORIES:\n{mem_block}\n\n"
            f"{topic_line}"
            "Identify pattern, confidence (0-1) and implications.\n"
            "Respond JSON:\n"
            "{\n"
            '  "pattern": "...",\n'
            '  "confidence": 0.x,\n'
            '  "implications": "..."\n'
            "}"
        )
    
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You extract patterns from memory clusters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
    
            if not response.choices[0].message.content:
                logger.warning("Empty response from OpenAI for pattern extraction")
                return None
    
            output_text = response.choices[0].message.content.strip()
            if not output_text:
                logger.warning("Empty output_text from OpenAI for pattern extraction")
                return None
    
            data = _parse_json_block(output_text)
            if data and isinstance(data, dict):
                return data
            if data and isinstance(data, list):
                return {
                    "pattern": "; ".join(map(str, data)),
                    "confidence": 0.5,
                    "implications": "",
                }
    
            logger.error("Failed to parse JSON in pattern extraction. Response: %s", output_text[:100])
            return None
    
        except Exception as e:
            logger.error("Cluster pattern extraction failed: %s", e)
            return None
            
    @with_transaction
    async def create_belief(self,
                         entity_type: str,
                         entity_id: int,
                         belief_text: str,
                         supporting_memory_ids: Optional[List[int]] = None,
                         confidence: float = 0.5,
                         tags: Optional[List[str]] = None,
                         conn = None) -> Dict[str, Any]:
        """
        Create a belief based on experiences.
        Beliefs are a special type of semantic memory that represent conclusions.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            belief_text: The belief statement
            supporting_memory_ids: IDs of memories supporting this belief
            confidence: Confidence level in this belief (0.0-1.0)
            tags: Tags for categorizing the belief
            
        Returns:
            Information about the created belief
        """
        # Format the belief
        formatted_belief = (
            f"I believe that {belief_text}" 
            if not belief_text.startswith("I believe") 
            else belief_text
        )
        
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Verify supporting memories exist (fixed: don't mutate list while iterating)
        if supporting_memory_ids:
            valid_ids = []
            for memory_id in supporting_memory_ids:
                row = await conn.fetchrow("""
                    SELECT id FROM unified_memories
                    WHERE id = $1
                      AND entity_type = $2
                      AND entity_id = $3
                      AND user_id = $4
                      AND conversation_id = $5
                """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
                
                if row:
                    valid_ids.append(memory_id)
                else:
                    logger.warning("Supporting memory %s not found", memory_id)
            supporting_memory_ids = valid_ids
        
        # Create the belief as a semantic memory
        belief_memory = Memory(
            text=formatted_belief,
            memory_type=MemoryType.SEMANTIC,
            significance=MemorySignificance.HIGH,
            emotional_intensity=0,
            tags=(tags or []) + ["belief", "semantic"],
            metadata={
                "belief_type": "explicit",
                "confidence": confidence,
                "supporting_memory_ids": supporting_memory_ids or [],
                "generated_at": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        
        belief_id = await memory_manager.add_memory(belief_memory, conn=conn)
        
        return {
            "belief_id": belief_id,
            "belief_text": formatted_belief,
            "confidence": confidence,
            "supporting_memories": supporting_memory_ids or []
        }
    
    @with_transaction
    async def get_beliefs(self,
                        entity_type: str,
                        entity_id: int,
                        topic: Optional[str] = None,
                        min_confidence: float = 0.0,
                        limit: int = 10,
                        conn = None) -> List[Dict[str, Any]]:
        """
        Get beliefs held by an entity, optionally filtered by topic.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            topic: Optional topic to filter beliefs
            min_confidence: Minimum confidence level
            limit: Maximum number of beliefs to return
            
        Returns:
            List of beliefs
        """
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Query parameters
        query_params = {
            "tags": ["belief"],
            "memory_types": [MemoryType.SEMANTIC],
            "limit": limit,
            "conn": conn
        }
        
        if topic:
            query_params["query"] = topic
            
        # Retrieve beliefs
        belief_memories = await memory_manager.retrieve_memories(**query_params)
        
        # Format results
        beliefs = []
        for memory in belief_memories:
            confidence = memory.metadata.get("confidence", 0.5)
            
            if confidence >= min_confidence:
                beliefs.append({
                    "id": memory.id,
                    "belief": memory.text,
                    "confidence": confidence,
                    "supporting_memories": memory.metadata.get("supporting_memory_ids", []),
                    "created_at": memory.timestamp.isoformat() if memory.timestamp else None
                })
        
        return beliefs
    
    @with_transaction
    async def update_belief_confidence(self,
                                     belief_id: int,
                                     entity_type: str,
                                     entity_id: int,
                                     new_confidence: float,
                                     reason: Optional[str] = None,
                                     conn = None) -> Dict[str, Any]:
        """
        Update the confidence level in a belief.
        
        Args:
            belief_id: ID of the belief to update
            entity_type: Type of entity
            entity_id: ID of the entity
            new_confidence: New confidence level (0.0-1.0)
            reason: Reason for the confidence change
            
        Returns:
            Updated belief information
        """
        # Fetch the belief
        row = await conn.fetchrow("""
            SELECT memory_text, metadata
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
              AND 'belief' = ANY(tags)
        """, belief_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Belief {belief_id} not found"}
            
        # Parse metadata
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        # Update confidence
        old_confidence = metadata.get("confidence", 0.5)
        metadata["confidence"] = max(0.0, min(1.0, new_confidence))
        
        # Track confidence history
        if "confidence_history" not in metadata:
            metadata["confidence_history"] = []
            
        metadata["confidence_history"].append({
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })
        
        # Update in database
        await conn.execute("""
            UPDATE unified_memories
            SET metadata = $1
            WHERE id = $2
        """, json.dumps(metadata), belief_id)
        
        return {
            "belief_id": belief_id,
            "belief_text": row["memory_text"],
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "reason": reason
        }
    
    @with_transaction
    async def generate_counterfactual(self,
                                    memory_id: int,
                                    entity_type: str,
                                    entity_id: int,
                                    variation_type: str = "alternative",
                                    conn = None) -> Dict[str, Any]:
        """
        Generate a counterfactual memory (what could have happened differently).
        This helps with abstract reasoning and understanding causality.
        
        Args:
            memory_id: ID of the base memory
            entity_type: Type of entity
            entity_id: ID of the entity
            variation_type: Type of counterfactual ('alternative', 'opposite', 'exaggeration')
            
        Returns:
            The counterfactual memory
        """
        # Fetch the base memory
        row = await conn.fetchrow("""
            SELECT memory_text, memory_type, significance, tags
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Memory {memory_id} not found"}
            
        memory_text = row["memory_text"]
        memory_type = row["memory_type"]
        significance = row["significance"]
        tags = row["tags"] or []
        # Handle tags if stored as JSON string
        if isinstance(tags, str):
            try:
                tags = json.loads(tags) or []
            except Exception:
                tags = []
        
        # Generate counterfactual
        counterfactual = await self._generate_counterfactual_variation(
            memory_text, variation_type
        )
        
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Create the counterfactual memory
        cf_memory = Memory(
            text=counterfactual,
            memory_type=MemoryType.SEMANTIC,
            significance=significance,
            emotional_intensity=0,
            tags=tags + ["counterfactual", variation_type],
            metadata={
                "source_memory_id": memory_id,
                "counterfactual_type": variation_type,
                "is_real": False,
                "generated_at": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        
        cf_id = await memory_manager.add_memory(cf_memory, conn=conn)
        
        return {
            "counterfactual_id": cf_id,
            "source_memory_id": memory_id,
            "counterfactual_text": counterfactual,
            "variation_type": variation_type
        }
    
    async def _generate_counterfactual_variation(
        self,
        memory_text: str,
        variation_type: str,
    ) -> str:
        """Counterfactual variation (alternative/opposite/exaggeration) via OpenAI API."""
        from logic.chatgpt_integration import get_async_openai_client, get_openai_client
        client = get_async_openai_client()
    
        description = {
            "alternative": "what might have gone differently",
            "opposite": "the opposite outcome",
            "exaggeration": "an exaggerated version of events",
        }.get(variation_type, "an alternative version")
    
        orig_block = _safe_block(memory_text)
        prompt = (
            f"Original memory:\n{orig_block}\n\n"
            f"Generate {description}. Begin with 'What if...' and return ONLY the variation."
        )
    
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You generate counterfactual variations of memories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            txt = response.choices[0].message.content.strip()
            if txt:
                return txt
            logger.warning("Empty response from OpenAI for counterfactual generation")
        except Exception as e:
            logger.error("Counterfactual generation failed: %s", e)
    
        # fallback
        prefix = {
            "alternative": "What if instead ",
            "opposite": "What if the opposite happened and ",
            "exaggeration": "What if, to a much greater extent, ",
        }.get(variation_type, "What if ")
        return prefix + memory_text
        
    @with_transaction
    async def build_semantic_network(self,
                                   entity_type: str,
                                   entity_id: int,
                                   central_topic: str,
                                   depth: int = 2,
                                   conn = None) -> Dict[str, Any]:
        """
        Build a semantic network around a central topic.
        Maps relationships between abstract concepts.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            central_topic: The central topic to build the network around
            depth: How many levels deep to build the network
            
        Returns:
            Semantic network structure
        """
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Start with the central topic
        semantic_network = {
            "central_topic": central_topic,
            "nodes": [],
            "edges": []
        }
        
        # Track processed topics to avoid duplication
        processed_topics = set([central_topic.lower()])
        current_topics = [central_topic]
        
        # Build the network level by level
        for level in range(depth):
            next_level_topics = []
            
            for topic in current_topics:
                # Find semantic memories related to this topic
                memories = await memory_manager.retrieve_memories(
                    query=topic,
                    memory_types=[MemoryType.SEMANTIC],
                    limit=5,
                    conn=conn
                )
                
                if not memories:
                    continue
                    
                # Create a node for this topic if not exists
                topic_node = {
                    "id": f"topic_{len(semantic_network['nodes'])}",
                    "label": topic,
                    "type": "topic",
                    "level": level
                }
                
                if not any(n.get("label") == topic for n in semantic_network["nodes"]):
                    semantic_network["nodes"].append(topic_node)
                else:
                    # Find existing node
                    for node in semantic_network["nodes"]:
                        if node.get("label") == topic:
                            topic_node = node
                            break
                
                # Process each memory
                for memory in memories:
                    # Create a node for this memory
                    memory_node = {
                        "id": f"memory_{memory.id}",
                        "label": memory.text[:50] + ("..." if len(memory.text) > 50 else ""),
                        "type": "memory",
                        "memory_id": memory.id,
                        "level": level + 0.5
                    }
                    
                    semantic_network["nodes"].append(memory_node)
                    
                    # Create an edge between topic and memory
                    semantic_network["edges"].append({
                        "source": topic_node["id"],
                        "target": memory_node["id"],
                        "type": "contains"
                    })
                    
                    # Extract related topics from this memory
                    related_topics = await self._extract_related_topics(memory.text, topic)
                    
                    for related_topic in related_topics:
                        if related_topic.lower() not in processed_topics:
                            processed_topics.add(related_topic.lower())
                            next_level_topics.append(related_topic)
                            
                            # Create a node for this related topic
                            related_node = {
                                "id": f"topic_{len(semantic_network['nodes'])}",
                                "label": related_topic,
                                "type": "topic",
                                "level": level + 1
                            }
                            
                            semantic_network["nodes"].append(related_node)
                            
                            # Create an edge between memory and related topic
                            semantic_network["edges"].append({
                                "source": memory_node["id"],
                                "target": related_node["id"],
                                "type": "relates_to"
                            })
            
            # Move to next level
            current_topics = next_level_topics
            
            if not current_topics:
                break
        
        return semantic_network
    
    async def _extract_related_topics(
        self,
        memory_text: str,
        current_topic: str,
    ) -> List[str]:
        """
        Extract 2-3 related topics from memory_text via the OpenAI API.
        """
        from logic.chatgpt_integration import get_async_openai_client, get_openai_client
        client = get_async_openai_client()
    
        mem_block = _safe_block(memory_text)
        cur_block = _safe_block(current_topic)
        
        prompt = (
            f"Memory:\n{mem_block}\n\n"
            f"Current topic:\n{cur_block}\n\n"
            "Extract 2-3 related topics as a JSON array. "
            "Topics must be distinct and different from the current topic.\n"
            "Return ONLY a JSON array of strings."
        )
    
        last_err = None
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": "You extract related topics from memory text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
                
                raw = response.choices[0].message.content.strip()
                try:
                    topics = json.loads(raw)
                except Exception:
                    topics = _parse_json_block(raw)
                    if topics is None:
                        raise ValueError("Invalid JSON in topic extraction")
    
                if not isinstance(topics, list):
                    raise ValueError(f"Expected list, got {type(topics)}")
    
                cleaned = []
                seen = set()
                cur_lower = current_topic.strip().lower()
                for t in topics:
                    if not isinstance(t, str):
                        continue
                    s = t.strip()
                    if not s:
                        continue
                    if s.lower() == cur_lower:
                        continue
                    if s.lower() in seen:
                        continue
                    seen.add(s.lower())
                    cleaned.append(s)
    
                if cleaned:
                    return cleaned[:3]
                raise ValueError("No usable topics in model output.")
            except Exception as e:
                last_err = e
                logger.warning("Topic extraction attempt %s failed: %s", attempt + 1, e)
                await asyncio.sleep(0.75 * (attempt + 1))
    
        logger.error("Topic extraction failed after retries; using fallback. Last error: %s", last_err)
        words = [
            w.capitalize()
            for w in (memory_text or "").split()
            if len(w) > 5 and w.strip() and w.strip().lower() != current_topic.lower()
        ]
        if not words:
            return []
        return random.sample(words, min(3, len(words)))

async def create_semantic_tables():
    """Create the necessary tables for the semantic memory system if they don't exist."""
    try:
        async with TransactionContext() as conn:
            
            # Create SemanticNetworks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS SemanticNetworks (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    central_topic TEXT NOT NULL,
                    network_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for SemanticNetworks
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_networks_entity 
                ON SemanticNetworks(user_id, conversation_id, entity_type, entity_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_networks_topic 
                ON SemanticNetworks(central_topic)
            """)
            
            # Create SemanticPatterns table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS SemanticPatterns (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    pattern_text TEXT NOT NULL,
                    confidence FLOAT NOT NULL DEFAULT 0.5,
                    source_memory_ids INTEGER[],
                    pattern_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for SemanticPatterns
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_patterns_entity 
                ON SemanticPatterns(user_id, conversation_id, entity_type, entity_id)
            """)
            
            # Create CounterfactualMemories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS CounterfactualMemories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id INTEGER NOT NULL,
                    source_memory_id INTEGER NOT NULL,
                    counterfactual_text TEXT NOT NULL,
                    variation_type VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for CounterfactualMemories
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_counterfactual_entity 
                ON CounterfactualMemories(user_id, conversation_id, entity_type, entity_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_counterfactual_source 
                ON CounterfactualMemories(source_memory_id)
            """)
            
            logger.info("Semantic memory tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating semantic memory tables: {e}")
        raise
