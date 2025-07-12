# memory/interference.py

import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio
import openai
import numpy as np

from .connection import with_transaction, TransactionContext
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_interference")

class MemoryInterferenceManager:
    """
    Systems for modeling memory interference, intrusive thoughts, and faulty recall.
    
    Features:
    - Proactive/retroactive interference
    - Intrusive memory surfacing
    - Memory competition during recall
    - Source confusion
    - Memory blending
    - False memory generation
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def detect_memory_interference(self,
                                       entity_type: str,
                                       entity_id: int,
                                       memory_id: int,
                                       interference_threshold: float = 0.8,
                                       conn = None) -> Dict[str, Any]:
        """
        Detect potential interference between memories.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            memory_id: ID of the memory to check for interference
            interference_threshold: Similarity threshold for interference
            
        Returns:
            Interference analysis
        """
        # Get the target memory
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # First get the memory details
        row = await conn.fetchrow("""
            SELECT memory_text, embedding, timestamp
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
        memory_embedding = row["embedding"]
        memory_timestamp = row["timestamp"]
        
        memory_embedding = row["embedding"]

        # Guard: pgvector can give us a NumPy array; its truth-value is ambiguous.
        if (
            memory_embedding is None or                                   # missing
            (isinstance(memory_embedding, np.ndarray) and memory_embedding.size == 0) or  # empty ndarray
            (isinstance(memory_embedding, (list, tuple)) and len(memory_embedding) == 0)   # empty list/tuple
        ):
            return {
                "error": "Memory doesn't have an embedding for similarity comparison"
            }
            
        # Find similar memories that might cause interference
        # We need both newer memories (retroactive interference) 
        # and older memories (proactive interference)
        
        # Retroactive interference - newer memories that might interfere with this one
        retroactive_rows = await conn.fetch("""
            SELECT id, memory_text, embedding, timestamp,
                   embedding <-> $1 AS similarity
            FROM unified_memories
            WHERE entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
              AND id != $6
              AND timestamp > $7
            ORDER BY similarity
            LIMIT 5
        """, memory_embedding, entity_type, entity_id, self.user_id, self.conversation_id,
            memory_id, memory_timestamp)
        
        # Proactive interference - older memories that might interfere with this one
        proactive_rows = await conn.fetch("""
            SELECT id, memory_text, embedding, timestamp,
                   embedding <-> $1 AS similarity
            FROM unified_memories
            WHERE entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
              AND id != $6
              AND timestamp < $7
            ORDER BY similarity
            LIMIT 5
        """, memory_embedding, entity_type, entity_id, self.user_id, self.conversation_id,
            memory_id, memory_timestamp)
        
        # Process the results
        retroactive_interference = []
        for row in retroactive_rows:
            similarity = row["similarity"]
            # In pgvector, lower values mean higher similarity
            normalized_similarity = 1.0 - min(1.0, similarity)
            
            if normalized_similarity >= interference_threshold:
                retroactive_interference.append({
                    "memory_id": row["id"],
                    "memory_text": row["memory_text"],
                    "similarity": normalized_similarity,
                    "timestamp": row["timestamp"].isoformat()
                })
        
        proactive_interference = []
        for row in proactive_rows:
            similarity = row["similarity"]
            normalized_similarity = 1.0 - min(1.0, similarity)
            
            if normalized_similarity >= interference_threshold:
                proactive_interference.append({
                    "memory_id": row["id"],
                    "memory_text": row["memory_text"],
                    "similarity": normalized_similarity,
                    "timestamp": row["timestamp"].isoformat()
                })
        
        # Calculate overall interference risk
        retroactive_risk = sum([i["similarity"] for i in retroactive_interference]) / max(1, len(retroactive_interference))
        proactive_risk = sum([i["similarity"] for i in proactive_interference]) / max(1, len(proactive_interference))
        
        # Combined risk, weighted towards retroactive (newer memory interference is usually stronger)
        combined_risk = (retroactive_risk * 0.7) + (proactive_risk * 0.3) if retroactive_interference or proactive_interference else 0.0
        
        # Update memory metadata with interference information
        row = await conn.fetchrow("""
            SELECT metadata
            FROM unified_memories
            WHERE id = $1
        """, memory_id)
        
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        # Add interference data to metadata
        metadata["interference"] = {
            "retroactive_risk": retroactive_risk,
            "proactive_risk": proactive_risk,
            "combined_risk": combined_risk,
            "retroactive_count": len(retroactive_interference),
            "proactive_count": len(proactive_interference),
            "analysis_date": datetime.now().isoformat()
        }
        
        await conn.execute("""
            UPDATE unified_memories
            SET metadata = $1
            WHERE id = $2
        """, json.dumps(metadata), memory_id)
        
        return {
            "memory_id": memory_id,
            "retroactive_interference": retroactive_interference,
            "proactive_interference": proactive_interference,
            "retroactive_risk": retroactive_risk,
            "proactive_risk": proactive_risk,
            "combined_risk": combined_risk,
            "high_interference_risk": combined_risk > 0.7
        }
    
    @with_transaction
    async def simulate_memory_competition(self,
                                       entity_type: str,
                                       entity_id: int,
                                       query: str,
                                       competition_count: int = 3,
                                       conn = None) -> Dict[str, Any]:
        """
        Simulate memory competition during recall.
        When recalling a memory, similar memories compete for activation.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            query: Query to retrieve competing memories
            competition_count: Number of memories to include in competition
            
        Returns:
            Results of memory competition
        """
        # Get competing memories
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Get more memories than needed so we can simulate competition
        competing_memories = await memory_manager.retrieve_memories(
            query=query,
            limit=competition_count * 2,
            conn=conn
        )
        
        if len(competing_memories) < 2:
            return {
                "query": query,
                "winner": competing_memories[0].to_dict() if competing_memories else None,
                "competition_occurred": False
            }
            
        # Limit to the number we want in competition
        competing_memories = competing_memories[:competition_count]
        
        # Simulate competition
        # Factors that influence which memory wins:
        # 1. Relevance to query (already factored in by retrieve_memories)
        # 2. Recency
        # 3. Emotional intensity
        # 4. Significance
        # 5. Times recalled (strengthens memory)
        # 6. Random noise (stochasticity in recall)
        
        scored_memories = []
        for memory in competing_memories:
            # Calculate memory strength factors
            # Base score from retrieval relevance (position in results)
            position_score = 1.0 - (competing_memories.index(memory) / len(competing_memories))
            
            # Recency factor
            recency_score = 0.0
            if memory.timestamp:
                days_old = (datetime.now() - memory.timestamp).days
                recency_score = max(0.0, 1.0 - (days_old / 90.0))  # Memories up to ~3 months old
                
            # Emotional intensity factor
            emotion_score = memory.emotional_intensity / 100.0
            
            # Significance factor
            significance_score = memory.significance / 5.0  # Normalized to 0.0-1.0
            
            # Recall strength factor
            recall_score = min(1.0, (memory.times_recalled or 0) / 10.0)
            
            # Calculate total score with weights
            total_score = (
                (position_score * 0.3) +
                (recency_score * 0.2) +
                (emotion_score * 0.15) +
                (significance_score * 0.2) +
                (recall_score * 0.15)
            )
            
            # Add randomness (stochasticity in recall)
            noise = random.uniform(-0.1, 0.1)
            final_score = total_score + noise
            
            scored_memories.append((memory, final_score))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Winner is the highest-scoring memory
        winner = scored_memories[0][0]
        
        # Update recall count for the winner
        await conn.execute("""
            UPDATE unified_memories
            SET times_recalled = COALESCE(times_recalled, 0) + 1,
                last_recalled = CURRENT_TIMESTAMP
            WHERE id = $1
        """, winner.id)
        
        # Check if we should create a blended memory
        create_blend = False
        if len(scored_memories) >= 2:
            top_score = scored_memories[0][1]
            second_score = scored_memories[1][1]
            
            # If top two memories are very close in activation, they might blend
            if (top_score - second_score) < 0.1:
                create_blend = True
        
        result = {
            "query": query,
            "winner": winner.to_dict(),
            "competition_occurred": True,
            "competition_size": len(competing_memories),
            "competing_memories": [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "score": score,
                    "position": i+1
                }
                for i, (memory, score) in enumerate(scored_memories)
            ]
        }
        
        # Create a blended memory if appropriate
        if create_blend:
            top_memory = scored_memories[0][0]
            second_memory = scored_memories[1][0]
            
            blended_memory = await self.generate_blended_memory(
                entity_type=entity_type,
                entity_id=entity_id,
                memory1_id=top_memory.id,
                memory2_id=second_memory.id,
                conn=conn
            )
            
            result["memory_blend"] = {
                "created": True,
                "blended_memory_id": blended_memory["blended_memory_id"],
                "blended_text": blended_memory["blended_text"]
            }
        
        return result
    
    @with_transaction
    async def generate_blended_memory(self,
                                    entity_type: str,
                                    entity_id: int,
                                    memory1_id: int,
                                    memory2_id: int,
                                    blend_method: str = "auto",
                                    conn = None) -> Dict[str, Any]:
        """
        Generate a blended memory from two source memories.
        This simulates source confusion and false memory creation.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            memory1_id: ID of first source memory
            memory2_id: ID of second source memory
            blend_method: How to blend the memories
            
        Returns:
            Blended memory information
        """
        # Get the source memories
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        rows = await conn.fetch("""
            SELECT id, memory_text, significance, emotional_intensity, tags, metadata, timestamp
            FROM unified_memories
            WHERE id IN ($1, $2)
              AND entity_type = $3
              AND entity_id = $4
              AND user_id = $5
              AND conversation_id = $6
        """, memory1_id, memory2_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if len(rows) < 2:
            return {"error": "One or both source memories not found"}
            
        # Parse the memories
        memories = []
        for row in rows:
            memory = {
                "id": row["id"],
                "text": row["memory_text"],
                "significance": row["significance"],
                "emotional_intensity": row["emotional_intensity"],
                "tags": row["tags"],
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
                "timestamp": row["timestamp"]
            }
            memories.append(memory)
        
        # Sort by id to ensure consistent order
        memories.sort(key=lambda x: x["id"])
        memory1 = memories[0]
        memory2 = memories[1]
        
        # Determine blend method if auto
        if blend_method == "auto":
            # Choose based on memory properties
            if abs(memory1["emotional_intensity"] - memory2["emotional_intensity"]) > 30:
                # Large emotional difference - emotion-weighted blend
                blend_method = "emotion_weighted"
            elif abs((memory1["timestamp"] - memory2["timestamp"]).total_seconds()) > 86400 * 30:
                # Large time difference (>30 days) - temporal_confusion
                blend_method = "temporal_confusion"
            else:
                # Default to element_mixing
                blend_method = "element_mixing"
        
        # Generate the blended memory text
        blended_text = ""
        
        if blend_method == "element_mixing":
            # Mix elements from both memories
            blended_text = await self._blend_memory_elements(memory1["text"], memory2["text"])
        elif blend_method == "temporal_confusion":
            # Confuse the time ordering
            blended_text = await self._blend_temporal_elements(memory1["text"], memory2["text"])
        elif blend_method == "emotion_weighted":
            # Weight by emotional intensity
            total_emotion = memory1["emotional_intensity"] + memory2["emotional_intensity"]
            weight1 = memory1["emotional_intensity"] / total_emotion if total_emotion > 0 else 0.5
            weight2 = memory2["emotional_intensity"] / total_emotion if total_emotion > 0 else 0.5
            blended_text = await self._blend_weighted_memories(memory1["text"], memory2["text"], weight1, weight2)
        else:
            # Default blend with GPT
            blended_text = await self._blend_with_gpt(memory1["text"], memory2["text"])
        
        # Calculate blended properties
        blended_significance = max(memory1["significance"], memory2["significance"])
        blended_emotion = max(memory1["emotional_intensity"], memory2["emotional_intensity"])
        
        # Combine tags
        blended_tags = list(set(memory1["tags"] + memory2["tags"]))
        if "blended" not in blended_tags:
            blended_tags.append("blended")
        if "false_memory" not in blended_tags:
            blended_tags.append("false_memory")
        
        # Create metadata
        blended_metadata = {
            "blend_type": blend_method,
            "source_memory_ids": [memory1["id"], memory2["id"]],
            "blend_date": datetime.now().isoformat(),
            "is_false_memory": True
        }
        
        # Create the blended memory
        blended_memory = Memory(
            text=blended_text,
            memory_type=MemoryType.OBSERVATION,  # Same as original memories
            significance=blended_significance,
            emotional_intensity=blended_emotion,
            tags=blended_tags,
            metadata=blended_metadata,
            timestamp=max(memory1["timestamp"], memory2["timestamp"])  # Use the more recent timestamp
        )
        
        blended_id = await memory_manager.add_memory(blended_memory, conn=conn)
        
        # Mark source memories as contributing to a blend
        for memory in memories:
            metadata = memory["metadata"]
            if "contributed_to_blends" not in metadata:
                metadata["contributed_to_blends"] = []
                
            metadata["contributed_to_blends"].append({
                "blend_id": blended_id,
                "blend_type": blend_method,
                "date": datetime.now().isoformat()
            })
            
            await conn.execute("""
                UPDATE unified_memories
                SET metadata = $1
                WHERE id = $2
            """, json.dumps(metadata), memory["id"])
        
        return {
            "blended_memory_id": blended_id,
            "source_memory_ids": [memory1["id"], memory2["id"]],
            "blended_text": blended_text,
            "blend_method": blend_method
        }
    
    async def _blend_memory_elements(self, text1: str, text2: str) -> str:
        """
        Blend two memories via Responses API.
        """
        client = get_openai_client()
    
        prompt = f"""
        Blend elements from these memories into a plausible but false single memory.
    
        Memory 1: {text1}
        Memory 2: {text2}
    
        Return ONLY the blended memory in first person.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You create blended false memories.",
                input=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            return resp.output_text.strip()
        except Exception as e:
            logger.error("Blend failed: %s", e)
            return f"I remember that {text1} … and somehow {text2} too."
    
    async def _blend_temporal_elements(self, text1: str, text2: str) -> str:
        """
        Build a single false memory that *confuses the order* of two memories.
        """
        client = get_openai_client()
    
        prompt = f"""
        You are creating a false memory that mixes timelines.
    
        MEMORY A: {text1}
        MEMORY B: {text2}
    
        Task:
          • Merge these memories into ONE narrative that scrambles their temporal order.
          • The final memory must be in first-person and feel coherent to the rememberer.
          • Begin the memory naturally (no meta preamble) and DO NOT add any explanation.
    
        Return ONLY the blended memory text.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You create temporally-confused false memories.",
                input=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            return resp.output_text.strip()
    
        except Exception as e:
            logger.error("Temporal blend failed: %s", e)
            # simple fallback – prepend a confusion marker
            return f"I keep mixing them up: first {text2}, then somehow {text1}…"
    
    async def _blend_weighted_memories(
        self,
        text1: str,
        text2: str,
        weight1: float,
        weight2: float,
    ) -> str:
        """
        Produce a blended memory, favouring the higher-weighted source.
        """
        client = get_openai_client()
    
        prompt = f"""
        MEMORY 1 (weight {weight1:.2f}): {text1}
        MEMORY 2 (weight {weight2:.2f}): {text2}
    
        Create ONE false memory in first-person:
          • Use more details from the memory with the HIGHER weight.
          • Still weave in subtle elements from the lower-weighted memory.
          • The tone/emotional colour should match the dominant (higher weight) memory.
          • No prefaces or explanations — just the memory text.
    
        Return ONLY the blended memory text.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You craft emotionally weighted blended memories.",
                input=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            return resp.output_text.strip()
    
        except Exception as e:
            logger.error("Weighted blend failed: %s", e)
            # naive fallback – whichever has higher weight comes first
            primary, secondary = (text1, text2) if weight1 >= weight2 else (text2, text1)
            return f"I vividly remember that {primary}, and—though it’s hazy—{secondary} seems mixed in there too."

    
    async def _blend_with_gpt(self, text1: str, text2: str) -> str:
        """
        Generic “blend two memories” helper using the Responses API.
        """
        client = get_openai_client()
    
        prompt = f"""
        MEMORY 1: {text1}
        MEMORY 2: {text2}
    
        Combine them into ONE plausible but incorrect memory, first-person POV.
        Subtly merge details so it feels authentic yet slightly inconsistent.
        Return ONLY the blended memory.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You produce generic blended false memories.",
                input=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            return resp.output_text.strip()
    
        except Exception as e:
            logger.error("Generic blend failed: %s", e)
            # basic sentence join fallback
            return f"{text1.rstrip('.')} … and somehow {text2.lstrip().capitalize()}."
    
    @with_transaction
    async def generate_intrusive_memory(self,
                                     entity_type: str,
                                     entity_id: int,
                                     context: str,
                                     intrusion_type: str = "emotional",
                                     conn = None) -> Dict[str, Any]:
        """
        Generate an intrusive memory based on the current context.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            context: Current context that might trigger intrusion
            intrusion_type: Type of intrusion (emotional, traumatic, random)
            
        Returns:
            Intrusive memory information
        """
        # Find candidate intrusive memories
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Different query strategies based on intrusion type
        query_params = {
            "conn": conn
        }
        
        if intrusion_type == "emotional":
            # For emotional intrusions, find high emotional intensity memories
            query_params["query"] = context
            query_params["min_significance"] = MemorySignificance.MEDIUM
            query_params["limit"] = 10
            
            candidates = await memory_manager.retrieve_memories(**query_params)
            
            # Sort by emotional intensity
            candidates.sort(key=lambda m: m.emotional_intensity, reverse=True)
            candidates = candidates[:3]  # Take top 3
            
        elif intrusion_type == "traumatic":
            # For traumatic intrusions, look for traumatic tags
            query_params["tags"] = ["traumatic"]
            query_params["limit"] = 5
            
            candidates = await memory_manager.retrieve_memories(**query_params)
            
        elif intrusion_type == "random":
            # For random intrusions, get recent memories
            query_params["limit"] = 20
            
            candidates = await memory_manager.retrieve_memories(**query_params)
            
            # Select randomly
            if len(candidates) > 3:
                candidates = random.sample(candidates, 3)
        else:
            # Default to context-based
            query_params["query"] = context
            query_params["limit"] = 3
            
            candidates = await memory_manager.retrieve_memories(**query_params)
        
        if not candidates:
            return {"intrusion_generated": False}
            
        # Select one memory to become intrusive
        intrusive_memory = random.choice(candidates)
        
        # Calculate intrusion probability
        intrusion_probability = 0.3  # Base probability
        
        # Factors that increase intrusion probability:
        # - High emotional intensity
        if intrusive_memory.emotional_intensity > 70:
            intrusion_probability += 0.2
            
        # - Memory is traumatic
        if "traumatic" in intrusive_memory.tags:
            intrusion_probability += 0.3
            
        # - Similar context
        if context and len(context) > 3:
            if any(word in intrusive_memory.text.lower() for word in context.lower().split()):
                intrusion_probability += 0.2
        
        # Decision to generate intrusion
        if random.random() > intrusion_probability:
            return {"intrusion_generated": False}
            
        # Generate the intrusion
        intrusion_format = await self._format_intrusive_memory(
            intrusive_memory.text, 
            intrusion_type,
            intrusive_memory.emotional_intensity
        )
        
        # Record the intrusion event
        intrusion_id = f"intrusion_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        metadata = intrusive_memory.metadata or {}
        if "intrusions" not in metadata:
            metadata["intrusions"] = []
            
        metadata["intrusions"].append({
            "intrusion_id": intrusion_id,
            "intrusion_type": intrusion_type,
            "context": context[:100] if context else "",
            "probability": intrusion_probability,
            "timestamp": datetime.now().isoformat()
        })
        
        await conn.execute("""
            UPDATE unified_memories
            SET metadata = $1
            WHERE id = $2
        """, json.dumps(metadata), intrusive_memory.id)
        
        return {
            "intrusion_generated": True,
            "intrusion_id": intrusion_id,
            "intrusion_type": intrusion_type,
            "source_memory_id": intrusive_memory.id,
            "source_memory_text": intrusive_memory.text,
            "intrusion_text": intrusion_format["text"],
            "intrusion_style": intrusion_format["style"],
            "emotional_intensity": intrusive_memory.emotional_intensity,
            "probability": intrusion_probability
        }
    
    async def _format_intrusive_memory(self, memory_text: str, intrusion_type: str, emotional_intensity: int) -> Dict[str, Any]:
        """
        Format a memory as an intrusive thought.
        """
        intrusion_styles = {
            "emotional": [
                "Suddenly, I'm reminded of {memory}",
                "A memory flashes into my mind: {memory}",
                "I can't help but remember {memory}",
                "The feeling when {memory} comes rushing back to me",
                "I'm unexpectedly reminded of the time {memory}"
            ],
            "traumatic": [
                "I'm overwhelmed by the memory of {memory}",
                "I'm suddenly back there again - {memory}",
                "The vivid image invades my thoughts: {memory}",
                "I can't push away the memory of {memory}",
                "My mind betrays me with the memory: {memory}"
            ],
            "random": [
                "Strangely, I think of {memory}",
                "For no reason, I remember {memory}",
                "A random memory surfaces: {memory}",
                "Out of nowhere, I recall {memory}",
                "My mind wanders to {memory}"
            ]
        }
        
        # Default to emotional if type not found
        available_styles = intrusion_styles.get(intrusion_type, intrusion_styles["emotional"])
        
        # Choose a style
        style = random.choice(available_styles)
        
        # Format the memory text
        if memory_text.startswith("I remember") or memory_text.startswith("I recall"):
            # Remove the prefix for cleaner integration
            for prefix in ["I remember ", "I remember that ", "I recall ", "I recall that "]:
                if memory_text.startswith(prefix):
                    memory_text = memory_text[len(prefix):]
                    break
        
        # Truncate if too long
        if len(memory_text) > 150:
            memory_text = memory_text[:147] + "..."
            
        # Format the intrusion
        intrusion_text = style.format(memory=memory_text)
        
        # Add emotional indicators for high intensity
        if emotional_intensity > 80:
            intense_additions = [
                " My heart races.",
                " I feel my breath catching.",
                " My hands tremble slightly.",
                " A wave of emotion washes over me.",
                " I feel it as intensely as when it happened."
            ]
            intrusion_text += random.choice(intense_additions)
        
        return {
            "text": intrusion_text,
            "style": style
        }
    
    @with_transaction
    async def create_false_memory(self,
                                entity_type: str,
                                entity_id: int,
                                false_memory_text: str,
                                significance: int = MemorySignificance.MEDIUM,
                                emotional_intensity: int = 50,
                                tags: List[str] = None,
                                related_true_memory_ids: List[int] = None,
                                conn = None) -> Dict[str, Any]:
        """
        Create a false memory.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            false_memory_text: Text of the false memory
            significance: Memory significance
            emotional_intensity: Emotional intensity
            tags: Additional tags
            related_true_memory_ids: IDs of true memories this false one is related to
            
        Returns:
            Created false memory information
        """
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Ensure false memory tags
        all_tags = (tags or []) + ["false_memory"]
        if "fabricated" not in all_tags:
            all_tags.append("fabricated")
        
        # Create metadata
        metadata = {
            "is_false_memory": True,
            "creation_date": datetime.now().isoformat(),
            "related_true_memory_ids": related_true_memory_ids or [],
            "fabrication_method": "explicit_creation"
        }
        
        # Create the false memory
        false_memory = Memory(
            text=false_memory_text,
            memory_type=MemoryType.OBSERVATION,
            significance=significance,
            emotional_intensity=emotional_intensity,
            tags=all_tags,
            metadata=metadata,
            timestamp=datetime.now() - timedelta(days=random.randint(1, 7))  # Backdate slightly
        )
        
        false_memory_id = await memory_manager.add_memory(false_memory, conn=conn)
        
        # Update related true memories if provided
        if related_true_memory_ids:
            for related_id in related_true_memory_ids:
                row = await conn.fetchrow("""
                    SELECT metadata
                    FROM unified_memories
                    WHERE id = $1
                      AND entity_type = $2
                      AND entity_id = $3
                      AND user_id = $4
                      AND conversation_id = $5
                """, related_id, entity_type, entity_id, self.user_id, self.conversation_id)
                
                if row:
                    metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
                    
                    if "related_false_memories" not in metadata:
                        metadata["related_false_memories"] = []
                        
                    metadata["related_false_memories"].append({
                        "false_memory_id": false_memory_id,
                        "creation_date": datetime.now().isoformat()
                    })
                    
                    await conn.execute("""
                        UPDATE unified_memories
                        SET metadata = $1
                        WHERE id = $2
                    """, json.dumps(metadata), related_id)
        
        return {
            "false_memory_id": false_memory_id,
            "false_memory_text": false_memory_text,
            "related_true_memory_ids": related_true_memory_ids
        }
    
    @with_transaction
    async def run_interference_maintenance(self,
                                        entity_type: str,
                                        entity_id: int,
                                        conn = None) -> Dict[str, Any]:
        """
        Run maintenance tasks related to memory interference.
        Automatically processes memory interference and occasionally creates blends.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Results of maintenance operations
        """
        # Find memory pairs with potential interference
        rows = await conn.fetch("""
            WITH memory_pairs AS (
                SELECT m1.id AS id1, m2.id AS id2, 
                       m1.embedding <-> m2.embedding AS similarity
                FROM unified_memories m1
                JOIN unified_memories m2 ON 
                    m1.entity_type = m2.entity_type AND
                    m1.entity_id = m2.entity_id AND
                    m1.user_id = m2.user_id AND
                    m1.conversation_id = m2.conversation_id AND
                    m1.id < m2.id  -- Avoid duplicates and self-pairs
                WHERE m1.entity_type = $1
                  AND m1.entity_id = $2
                  AND m1.user_id = $3
                  AND m1.conversation_id = $4
                  AND m1.is_archived = FALSE
                  AND m2.is_archived = FALSE
                ORDER BY similarity
                LIMIT 10
            )
            SELECT id1, id2, similarity
            FROM memory_pairs
            WHERE similarity < 0.3  -- Higher similarity (lower value in pgvector)
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        # Process the results
        interference_updates = 0
        blends_created = 0
        
        for row in rows:
            memory1_id = row["id1"]
            memory2_id = row["id2"]
            similarity = row["similarity"]
            
            # Mark interference for both memories
            for memory_id in [memory1_id, memory2_id]:
                try:
                    await self.detect_memory_interference(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        memory_id=memory_id,
                        conn=conn
                    )
                    interference_updates += 1
                except Exception as e:
                    logger.error(f"Error processing interference for memory {memory_id}: {e}")
            
            # Occasionally create a blended memory
            # Higher chance for more similar memories
            blend_probability = 0.1 * (1.0 - similarity)
            
            if random.random() < blend_probability:
                try:
                    await self.generate_blended_memory(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        memory1_id=memory1_id,
                        memory2_id=memory2_id,
                        conn=conn
                    )
                    blends_created += 1
                except Exception as e:
                    logger.error(f"Error creating blend for memories {memory1_id}/{memory2_id}: {e}")
        
        return {
            "interference_updates": interference_updates,
            "blends_created": blends_created,
            "pairs_processed": len(rows)
        }
    
    @with_transaction
    async def get_false_memory_status(self,
                                    entity_type: str,
                                    entity_id: int,
                                    belief_in_false_memories: bool = True,
                                    conn = None) -> Dict[str, Any]:
        """
        Get status information about false memories.
        Useful for meta-awareness of memory reliability.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            belief_in_false_memories: Whether the entity believes their false memories are true
            
        Returns:
            False memory statistics and information
        """
        # Get counts of different types of false memories
        fabricated_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND 'fabricated' = ANY(tags)
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        blended_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND 'blended' = ANY(tags)
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        false_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND 'false_memory' = ANY(tags)
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        total_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        # Get most recent false memories
        recent_false = []
        if belief_in_false_memories:
            # If they believe false memories, don't tag them differently
            rows = await conn.fetch("""
                SELECT id, memory_text, timestamp
                FROM unified_memories
                WHERE entity_type = $1
                  AND entity_id = $2
                  AND user_id = $3
                  AND conversation_id = $4
                  AND 'false_memory' = ANY(tags)
                ORDER BY timestamp DESC
                LIMIT 3
            """, entity_type, entity_id, self.user_id, self.conversation_id)
            
            for row in rows:
                recent_false.append({
                    "id": row["id"],
                    "text": row["memory_text"],
                    "timestamp": row["timestamp"].isoformat()
                })
        
        # Calculate false memory percentage
        false_percentage = (false_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "fabricated_count": fabricated_count,
            "blended_count": blended_count,
            "total_false_count": false_count,
            "total_memories": total_count,
            "false_percentage": round(false_percentage, 1),
            "belief_in_false_memories": belief_in_false_memories,
            "recent_false_memories": recent_false if belief_in_false_memories else []
        }

# Create the necessary tables if they don't exist
async def create_interference_tables():
    """Create the necessary tables for the memory interference system if they don't exist."""
    # No additional tables needed - uses existing unified_memories table
    pass
