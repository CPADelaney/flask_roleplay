# nyx/core/memory_core.py

import asyncio
import datetime
import json
import logging
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)

class MemoryCore:
    """
    Unified memory storage and retrieval system that consolidates all memory-related functionality.
    Handles storage, retrieval, embedding, decay, consolidation, and archival of memories.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Memory storage containers
        self.memories = {}  # Main memory storage: memory_id -> memory_data
        self.memory_embeddings = {}  # memory_id -> embedding vector
        
        # Indices for efficient retrieval
        self.type_index = {}  # memory_type -> [memory_ids]
        self.tag_index = {}  # tag -> [memory_ids]
        self.scope_index = {}  # memory_scope -> [memory_ids]
        self.entity_index = {}  # entity_id -> [memory_ids]
        self.emotional_index = {}  # emotion -> [memory_ids]
        
        # Temporal index for chronological retrieval
        self.temporal_index = []  # [(timestamp, memory_id)]
        
        # Significance tracking for memory importance
        self.significance_index = {}  # significance_level -> [memory_ids]
        
        # Archive status
        self.archived_memories = set()  # Set of archived memory_ids
        self.consolidated_memories = set()  # Set of memory_ids that have been consolidated
        
        # Configuration
        self.embed_dim = 1536  # Default embedding dimension
        self.max_memory_age_days = 90  # Default max memory age before archival consideration
        self.decay_rate = 0.1  # Default decay rate per day
        self.consolidation_threshold = 0.85  # Similarity threshold for memory consolidation
        
        # Caches for performance
        self.query_cache = {}  # query -> (timestamp, results)
        self.cache_ttl = 300  # Cache TTL in seconds
        
        # Initialized flag
        self.initialized = False
        
        # Initialization timestamp
        self.init_time = datetime.datetime.now()
    
    async def initialize(self):
        """Initialize memory system and load existing memories"""
        if self.initialized:
            return
        
        logger.info(f"Initializing memory system for user {self.user_id}, conversation {self.conversation_id}")
        
        # This is where you would load memories from a database
        # For this implementation, we'll just initialize empty structures
        
        # Initialize indices
        self.type_index = {
            "observation": set(),
            "reflection": set(),
            "abstraction": set(),
            "experience": set(),
            "consolidated": set()
        }
        
        self.scope_index = {
            "game": set(),
            "user": set(),
            "global": set()
        }
        
        # Initialize significance index
        self.significance_index = {level: set() for level in range(1, 11)}
        
        self.initialized = True
        logger.info("Memory system initialized")
    
    async def add_memory(self,
                        memory_text: str,
                        memory_type: str = "observation",
                        memory_scope: str = "game",
                        significance: int = 5,
                        tags: List[str] = None,
                        metadata: Dict[str, Any] = None,
                        embedding: List[float] = None) -> str:
        """
        Add a new memory to the system
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory (observation, reflection, abstraction, experience)
            memory_scope: Scope of memory (game, user, global)
            significance: Importance level (1-10)
            tags: Optional tags for categorization
            metadata: Additional metadata
            embedding: Optional pre-computed embedding vector
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            await self.initialize()
        
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Normalize tags
        if tags is None:
            tags = []
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
        
        # Set timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self._generate_embedding(memory_text)
        
        # Create memory object
        memory = {
            "id": memory_id,
            "memory_text": memory_text,
            "memory_type": memory_type,
            "memory_scope": memory_scope,
            "significance": significance,
            "tags": tags,
            "metadata": metadata,
            "embedding": embedding,
            "created_at": datetime.datetime.now().isoformat(),
            "times_recalled": 0,
            "is_archived": False,
            "is_consolidated": False
        }
        
        # Store memory
        self.memories[memory_id] = memory
        self.memory_embeddings[memory_id] = embedding
        
        # Update indices
        if memory_type in self.type_index:
            self.type_index[memory_type].add(memory_id)
        else:
            self.type_index[memory_type] = {memory_id}
        
        if memory_scope in self.scope_index:
            self.scope_index[memory_scope].add(memory_id)
        else:
            self.scope_index[memory_scope] = {memory_id}
        
        # Update tag index
        for tag in tags:
            if tag in self.tag_index:
                self.tag_index[tag].add(memory_id)
            else:
                self.tag_index[tag] = {memory_id}
        
        # Update temporal index
        timestamp = datetime.datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
        self.temporal_index.append((timestamp, memory_id))
        self.temporal_index.sort(key=lambda x: x[0])  # Keep sorted by timestamp
        
        # Update significance index
        sig_level = min(10, max(1, int(significance)))
        self.significance_index[sig_level].add(memory_id)
        
        # Update entity index if entities are specified
        entities = metadata.get("entities", [])
        for entity_id in entities:
            if entity_id in self.entity_index:
                self.entity_index[entity_id].add(memory_id)
            else:
                self.entity_index[entity_id] = {memory_id}
        
        # Update emotional index if emotional context is specified
        if "emotional_context" in metadata:
            emotional_context = metadata["emotional_context"]
            primary_emotion = emotional_context.get("primary_emotion")
            if primary_emotion:
                if primary_emotion in self.emotional_index:
                    self.emotional_index[primary_emotion].add(memory_id)
                else:
                    self.emotional_index[primary_emotion] = {memory_id}
        
        # Clear query cache since memory store has changed
        self.query_cache = {}
        
        logger.debug(f"Added memory {memory_id} of type {memory_type}")
        return memory_id
    
    async def retrieve_memories(self,
                              query: str,
                              memory_types: List[str] = None,
                              scopes: List[str] = None,
                              limit: int = 5,
                              min_significance: int = 3,
                              context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query and filters
        
        Args:
            query: Search query text
            memory_types: Types of memories to retrieve
            scopes: Memory scopes to search in
            limit: Maximum number of memories to return
            min_significance: Minimum significance level
            context: Additional context for search
            
        Returns:
            List of matching memories with relevance scores
        """
        if not self.initialized:
            await self.initialize()
        
        # Set defaults
        memory_types = memory_types or ["observation", "reflection", "abstraction", "experience"]
        scopes = scopes or ["game", "user", "global"]
        context = context or {}
        
        # Check cache first
        cache_key = f"{query}_{','.join(memory_types)}_{','.join(scopes)}_{limit}_{min_significance}"
        if cache_key in self.query_cache:
            cache_time, cache_results = self.query_cache[cache_key]
            cache_age = (datetime.datetime.now() - cache_time).total_seconds()
            if cache_age < self.cache_ttl:
                # Return cached results
                return [self.memories[mid] for mid in cache_results if mid in self.memories]
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Get IDs of memories matching type and scope filters
        type_matches = set()
        for memory_type in memory_types:
            if memory_type in self.type_index:
                type_matches.update(self.type_index[memory_type])
        
        scope_matches = set()
        for scope in scopes:
            if scope in self.scope_index:
                scope_matches.update(self.scope_index[scope])
        
        significance_matches = set()
        for sig_level in range(min_significance, 11):
            significance_matches.update(self.significance_index[sig_level])
        
        # Find intersection of all filters
        candidate_ids = type_matches.intersection(scope_matches, significance_matches)
        
        # Remove archived memories unless explicitly requested
        if not context.get("include_archived", False):
            candidate_ids = candidate_ids - self.archived_memories
        
        # Calculate relevance for candidates
        scored_candidates = []
        for memory_id in candidate_ids:
            if memory_id in self.memory_embeddings:
                embedding = self.memory_embeddings[memory_id]
                relevance = self._cosine_similarity(query_embedding, embedding)
                
                # Apply entity relevance boost if context includes entities
                entity_boost = 0.0
                if "entities" in context and self.memories[memory_id].get("metadata", {}).get("entities"):
                    memory_entities = self.memories[memory_id]["metadata"]["entities"]
                    context_entities = context["entities"]
                    matching_entities = set(memory_entities).intersection(set(context_entities))
                    entity_boost = len(matching_entities) * 0.05  # 5% boost per matching entity
                
                # Apply emotional relevance boost if context includes emotional state
                emotional_boost = 0.0
                if "emotional_state" in context and self.memories[memory_id].get("metadata", {}).get("emotional_context"):
                    memory_emotion = self.memories[memory_id]["metadata"]["emotional_context"]
                    context_emotion = context["emotional_state"]
                    emotional_boost = self._calculate_emotional_relevance(memory_emotion, context_emotion)
                
                # Apply temporal recency boost
                temporal_boost = 0.0
                memory_timestamp = datetime.datetime.fromisoformat(
                    self.memories[memory_id].get("metadata", {}).get("timestamp", self.init_time.isoformat()).replace("Z", "+00:00")
                )
                days_old = (datetime.datetime.now() - memory_timestamp).days
                if days_old < 7:  # Boost recent memories
                    temporal_boost = 0.1 * (1 - (days_old / 7))
                
                # Calculate final relevance score
                final_relevance = min(1.0, relevance + entity_boost + emotional_boost + temporal_boost)
                
                scored_candidates.append((memory_id, final_relevance))
        
        # Sort by relevance
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_memory_ids = [mid for mid, _ in scored_candidates[:limit]]
        
        # Prepare results with relevance scores
        results = []
        for memory_id, relevance in scored_candidates[:limit]:
            if memory_id in self.memories:
                memory = self.memories[memory_id].copy()
                memory["relevance"] = relevance
                
                # Update recall count
                await self._update_memory_recall(memory_id)
                
                results.append(memory)
        
        # Cache results
        self.query_cache[cache_key] = (datetime.datetime.now(), top_memory_ids)
        
        return results
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        if not self.initialized:
            await self.initialize()
        
        if memory_id in self.memories:
            return self.memories[memory_id].copy()
        
        return None
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        if not self.initialized:
            await self.initialize()
        
        if memory_id not in self.memories:
            return False
        
        # Get existing memory
        memory = self.memories[memory_id]
        
        # Track indices to update
        index_updates = {
            "type": False,
            "scope": False,
            "tags": False,
            "significance": False,
            "entities": False,
            "emotional": False
        }
        
        # Update fields
        for key, value in updates.items():
            if key in memory:
                # Check if indices need updating
                if key == "memory_type" and value != memory["memory_type"]:
                    index_updates["type"] = True
                elif key == "memory_scope" and value != memory["memory_scope"]:
                    index_updates["scope"] = True
                elif key == "tags" and set(value) != set(memory["tags"]):
                    index_updates["tags"] = True
                elif key == "significance" and value != memory["significance"]:
                    index_updates["significance"] = True
                elif key == "metadata":
                    # Check for entity changes
                    old_entities = memory.get("metadata", {}).get("entities", [])
                    new_entities = value.get("entities", old_entities)
                    if set(new_entities) != set(old_entities):
                        index_updates["entities"] = True
                    
                    # Check for emotional context changes
                    old_emotion = memory.get("metadata", {}).get("emotional_context", {}).get("primary_emotion")
                    new_emotion = value.get("emotional_context", {}).get("primary_emotion", old_emotion)
                    if new_emotion != old_emotion:
                        index_updates["emotional"] = True
                
                # Update the field
                memory[key] = value
        
        # Update embedding if text changed
        if "memory_text" in updates:
            memory["embedding"] = await self._generate_embedding(updates["memory_text"])
            self.memory_embeddings[memory_id] = memory["embedding"]
        
        # Update indices if needed
        if index_updates["type"]:
            # Remove from old type index
            old_type = memory["memory_type"]
            if old_type in self.type_index and memory_id in self.type_index[old_type]:
                self.type_index[old_type].remove(memory_id)
            
            # Add to new type index
            new_type = updates["memory_type"]
            if new_type in self.type_index:
                self.type_index[new_type].add(memory_id)
            else:
                self.type_index[new_type] = {memory_id}
        
        if index_updates["scope"]:
            # Remove from old scope index
            old_scope = memory["memory_scope"]
            if old_scope in self.scope_index and memory_id in self.scope_index[old_scope]:
                self.scope_index[old_scope].remove(memory_id)
            
            # Add to new scope index
            new_scope = updates["memory_scope"]
            if new_scope in self.scope_index:
                self.scope_index[new_scope].add(memory_id)
            else:
                self.scope_index[new_scope] = {memory_id}
        
        if index_updates["tags"]:
            # Remove from old tag indices
            old_tags = memory["tags"]
            for tag in old_tags:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
            
            # Add to new tag indices
            new_tags = updates["tags"]
            for tag in new_tags:
                if tag in self.tag_index:
                    self.tag_index[tag].add(memory_id)
                else:
                    self.tag_index[tag] = {memory_id}
        
        if index_updates["significance"]:
            # Remove from old significance index
            old_sig = min(10, max(1, int(memory["significance"])))
            if old_sig in self.significance_index and memory_id in self.significance_index[old_sig]:
                self.significance_index[old_sig].remove(memory_id)
            
            # Add to new significance index
            new_sig = min(10, max(1, int(updates["significance"])))
            if new_sig in self.significance_index:
                self.significance_index[new_sig].add(memory_id)
            else:
                self.significance_index[new_sig] = {memory_id}
        
        if index_updates["entities"]:
            # Update entity index
            metadata = memory.get("metadata", {})
            old_entities = metadata.get("entities", [])
            new_entities = updates.get("metadata", {}).get("entities", old_entities)
            
            # Remove from old entity indices
            for entity_id in old_entities:
                if entity_id in self.entity_index and memory_id in self.entity_index[entity_id]:
                    self.entity_index[entity_id].remove(memory_id)
            
            # Add to new entity indices
            for entity_id in new_entities:
                if entity_id in self.entity_index:
                    self.entity_index[entity_id].add(memory_id)
                else:
                    self.entity_index[entity_id] = {memory_id}
        
        if index_updates["emotional"]:
            # Update emotional index
            metadata = memory.get("metadata", {})
            old_emotion = metadata.get("emotional_context", {}).get("primary_emotion")
            new_emotion = updates.get("metadata", {}).get("emotional_context", {}).get("primary_emotion", old_emotion)
            
            # Remove from old emotional index
            if old_emotion and old_emotion in self.emotional_index and memory_id in self.emotional_index[old_emotion]:
                self.emotional_index[old_emotion].remove(memory_id)
            
            # Add to new emotional index
            if new_emotion:
                if new_emotion in self.emotional_index:
                    self.emotional_index[new_emotion].add(memory_id)
                else:
                    self.emotional_index[new_emotion] = {memory_id}
        
        # Update archived status if specified
        if "is_archived" in updates:
            if updates["is_archived"] and memory_id not in self.archived_memories:
                self.archived_memories.add(memory_id)
            elif not updates["is_archived"] and memory_id in self.archived_memories:
                self.archived_memories.remove(memory_id)
        
        # Update consolidated status if specified
        if "is_consolidated" in updates:
            if updates["is_consolidated"] and memory_id not in self.consolidated_memories:
                self.consolidated_memories.add(memory_id)
            elif not updates["is_consolidated"] and memory_id in self.consolidated_memories:
                self.consolidated_memories.remove(memory_id)
        
        # Clear query cache as memory store has changed
        self.query_cache = {}
        
        logger.debug(f"Updated memory {memory_id}")
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the system"""
        if not self.initialized:
            await self.initialize()
        
        if memory_id not in self.memories:
            return False
        
        # Get memory for index cleanup
        memory = self.memories[memory_id]
        
        # Remove from type index
        memory_type = memory["memory_type"]
        if memory_type in self.type_index and memory_id in self.type_index[memory_type]:
            self.type_index[memory_type].remove(memory_id)
        
        # Remove from scope index
        memory_scope = memory["memory_scope"]
        if memory_scope in self.scope_index and memory_id in self.scope_index[memory_scope]:
            self.scope_index[memory_scope].remove(memory_id)
        
        # Remove from tag index
        for tag in memory["tags"]:
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        # Remove from significance index
        sig_level = min(10, max(1, int(memory["significance"])))
        if sig_level in self.significance_index and memory_id in self.significance_index[sig_level]:
            self.significance_index[sig_level].remove(memory_id)
        
        # Remove from entity index
        entities = memory.get("metadata", {}).get("entities", [])
        for entity_id in entities:
            if entity_id in self.entity_index and memory_id in self.entity_index[entity_id]:
                self.entity_index[entity_id].remove(memory_id)
        
        # Remove from emotional index
        emotion = memory.get("metadata", {}).get("emotional_context", {}).get("primary_emotion")
        if emotion and emotion in self.emotional_index and memory_id in self.emotional_index[emotion]:
            self.emotional_index[emotion].remove(memory_id)
        
        # Remove from temporal index
        self.temporal_index = [(ts, mid) for ts, mid in self.temporal_index if mid != memory_id]
        
        # Remove from archived or consolidated sets if present
        if memory_id in self.archived_memories:
            self.archived_memories.remove(memory_id)
        if memory_id in self.consolidated_memories:
            self.consolidated_memories.remove(memory_id)
        
        # Delete memory and embedding
        del self.memories[memory_id]
        if memory_id in self.memory_embeddings:
            del self.memory_embeddings[memory_id]
        
        # Clear query cache as memory store has changed
        self.query_cache = {}
        
        logger.debug(f"Deleted memory {memory_id}")
        return True
    
    async def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory"""
        return await self.update_memory(memory_id, {"is_archived": True})
    
    async def unarchive_memory(self, memory_id: str) -> bool:
        """Unarchive a memory"""
        return await self.update_memory(memory_id, {"is_archived": False})
    
    async def mark_as_consolidated(self, memory_id: str, consolidated_into: str) -> bool:
        """Mark a memory as consolidated into another memory"""
        return await self.update_memory(memory_id, {
            "is_consolidated": True,
            "metadata": {
                "consolidated_into": consolidated_into,
                "consolidation_date": datetime.datetime.now().isoformat()
            }
        })
    
    async def _update_memory_recall(self, memory_id: str):
        """Update recall count and timestamp for a memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory["times_recalled"] = memory.get("times_recalled", 0) + 1
            memory["metadata"] = memory.get("metadata", {})
            memory["metadata"]["last_recalled"] = datetime.datetime.now().isoformat()
    
    async def apply_memory_decay(self) -> Dict[str, Any]:
        """Apply decay to memories based on age, significance, and recall frequency"""
        if not self.initialized:
            await self.initialize()
        
        decayed_count = 0
        archived_count = 0
        
        for memory_id, memory in self.memories.items():
            # Skip non-episodic memories
            if memory["memory_type"] != "observation":
                continue
            
            # Skip already archived memories
            if memory_id in self.archived_memories:
                continue
            
            significance = memory["significance"]
            times_recalled = memory.get("times_recalled", 0)
            
            # Calculate age
            timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            days_old = (datetime.datetime.now() - timestamp).days
            
            # Calculate decay factors
            age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
            recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
            
            # Calculate decay amount
            decay_amount = self.decay_rate * age_factor * recall_factor
            
            # Apply extra decay to memories never recalled
            if times_recalled == 0 and days_old > 7:
                decay_amount *= 1.5
            
            # Skip if decay is negligible
            if decay_amount < 0.05:
                continue
            
            # Store original significance if this is first decay
            if "original_significance" not in memory.get("metadata", {}):
                memory["metadata"]["original_significance"] = significance
            
            # Apply decay with minimum of 1
            new_significance = max(1, significance - decay_amount)
            
            # Only update if change is significant
            if abs(new_significance - significance) >= 0.5:
                await self.update_memory(memory_id, {
                    "significance": new_significance,
                    "metadata": {
                        **memory.get("metadata", {}),
                        "last_decay": datetime.datetime.now().isoformat(),
                        "decay_amount": decay_amount
                    }
                })
                decayed_count += 1
            
            # Archive if significance is very low and memory is old
            if new_significance < 2 and days_old > 30:
                await self.archive_memory(memory_id)
                archived_count += 1
        
        return {
            "memories_decayed": decayed_count,
            "memories_archived": archived_count
        }
    
    async def find_memory_clusters(self) -> List[List[str]]:
        """Find clusters of similar memories based on embedding similarity"""
        if not self.initialized:
            await self.initialize()
        
        # Get memories that are candidates for clustering
        candidate_ids = set()
        for memory_type in ["observation", "experience"]:
            if memory_type in self.type_index:
                candidate_ids.update(self.type_index[memory_type])
        
        # Remove already consolidated or archived memories
        candidate_ids = candidate_ids - self.consolidated_memories - self.archived_memories
        
        # Group memories into clusters
        clusters = []
        unclustered = list(candidate_ids)
        
        while unclustered:
            # Take the first memory as a seed
            seed_id = unclustered.pop(0)
            if seed_id not in self.memory_embeddings:
                continue
                
            seed_embedding = self.memory_embeddings[seed_id]
            current_cluster = [seed_id]
            
            # Find similar memories
            i = 0
            while i < len(unclustered):
                memory_id = unclustered[i]
                
                if memory_id not in self.memory_embeddings:
                    i += 1
                    continue
                
                # Calculate similarity
                similarity = self._cosine_similarity(seed_embedding, self.memory_embeddings[memory_id])
                
                # Add to cluster if similar enough
                if similarity > self.consolidation_threshold:
                    current_cluster.append(memory_id)
                    unclustered.pop(i)
                else:
                    i += 1
            
            # Only keep clusters with multiple memories
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        return clusters
    
    async def consolidate_memory_clusters(self) -> Dict[str, Any]:
        """Consolidate clusters of similar memories"""
        if not self.initialized:
            await self.initialize()
        
        # Find clusters
        clusters = await self.find_memory_clusters()
        
        consolidated_count = 0
        affected_memories = 0
        
        for cluster in clusters:
            # Skip small clusters
            if len(cluster) < 3:
                continue
            
            try:
                # Get memory texts
                memory_texts = []
                for memory_id in cluster:
                    if memory_id in self.memories:
                        memory_texts.append(self.memories[memory_id]["memory_text"])
                
                # Generate consolidated text
                consolidated_text = await self._generate_consolidated_text(memory_texts)
                
                # Calculate average significance (weighted by individual memory significance)
                total_significance = sum(self.memories[mid]["significance"] for mid in cluster if mid in self.memories)
                avg_significance = total_significance / len(cluster)
                
                # Combine tags
                all_tags = set()
                for memory_id in cluster:
                    if memory_id in self.memories:
                        all_tags.update(self.memories[memory_id].get("tags", []))
                
                # Create consolidated memory
                consolidated_id = await self.add_memory(
                    memory_text=consolidated_text,
                    memory_type="consolidated",
                    memory_scope="game",
                    significance=min(10, int(avg_significance) + 1),  # Slightly higher significance than average
                    tags=list(all_tags) + ["consolidated"],
                    metadata={
                        "source_memory_ids": cluster,
                        "consolidation_date": datetime.datetime.now().isoformat()
                    }
                )
                
                # Mark original memories as consolidated
                for memory_id in cluster:
                    await self.mark_as_consolidated(memory_id, consolidated_id)
                
                consolidated_count += 1
                affected_memories += len(cluster)
                
            except Exception as e:
                logger.error(f"Error consolidating cluster: {e}")
        
        return {
            "clusters_consolidated": consolidated_count,
            "memories_affected": affected_memories
        }
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run full memory maintenance"""
        if not self.initialized:
            await self.initialize()
        
        # Apply memory decay
        decay_result = await self.apply_memory_decay()
        
        # Consolidate similar memories
        consolidation_result = await self.consolidate_memory_clusters()
        
        # Archive old memories with low recall
        archive_count = 0
        for memory_id, memory in self.memories.items():
            # Skip already archived memories
            if memory_id in self.archived_memories:
                continue
            
            # Skip non-observation memories
            if memory["memory_type"] not in ["observation", "experience"]:
                continue
            
            times_recalled = memory.get("times_recalled", 0)
            
            # Get age
            timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            days_old = (datetime.datetime.now() - timestamp).days
            
            # Archive very old memories that haven't been recalled
            if days_old > self.max_memory_age_days and times_recalled == 0:
                await self.archive_memory(memory_id)
                archive_count += 1
        
        return {
            "memories_decayed": decay_result["memories_decayed"],
            "clusters_consolidated": consolidation_result["clusters_consolidated"],
            "memories_archived": decay_result["memories_archived"] + archive_count
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        if not self.initialized:
            await self.initialize()
        
        # Count memories by type
        type_counts = {}
        for memory_type, memory_ids in self.type_index.items():
            type_counts[memory_type] = len(memory_ids)
        
        # Count by scope
        scope_counts = {}
        for scope, memory_ids in self.scope_index.items():
            scope_counts[scope] = len(memory_ids)
        
        # Get top tags
        tag_counts = {}
        for tag, memory_ids in self.tag_index.items():
            tag_counts[tag] = len(memory_ids)
        
        # Sort tags by count
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Get total memory count
        total_memories = len(self.memories)
        
        # Get archive and consolidation counts
        archived_count = len(self.archived_memories)
        consolidated_count = len(self.consolidated_memories)
        
        # Get average significance
        if total_memories > 0:
            avg_significance = sum(m["significance"] for m in self.memories.values()) / total_memories
        else:
            avg_significance = 0
        
        # Calculate memory age stats
        now = datetime.datetime.now()
        ages = []
        for memory in self.memories.values():
            timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age_days = (now - timestamp).days
            ages.append(age_days)
        
        oldest_memory = max(ages) if ages else 0
        newest_memory = min(ages) if ages else 0
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "total_memories": total_memories,
            "type_counts": type_counts,
            "scope_counts": scope_counts,
            "top_tags": dict(top_tags),
            "archived_count": archived_count,
            "consolidated_count": consolidated_count,
            "avg_significance": avg_significance,
            "oldest_memory_days": oldest_memory,
            "newest_memory_days": newest_memory,
            "avg_age_days": avg_age
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 * norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_emotional_relevance(self, memory_emotion: Dict[str, Any], current_emotion: Dict[str, Any]) -> float:
        """Calculate emotional relevance boost based on emotional context"""
        # Get primary emotions
        memory_primary = memory_emotion.get("primary_emotion", "neutral")
        current_primary = current_emotion.get("primary_emotion", "neutral")
        
        # Primary emotion match gives highest boost
        if memory_primary == current_primary:
            return 0.2
        
        # Secondary emotion matches give smaller boost
        memory_secondary = memory_emotion.get("secondary_emotions", {})
        if current_primary in memory_secondary:
            return 0.1
        
        # Valence match (positive/negative alignment)
        memory_valence = memory_emotion.get("valence", 0.0)
        current_valence = current_emotion.get("valence", 0.0)
        
        # Similar valence gives small boost
        if abs(memory_valence - current_valence) < 0.5:
            return 0.05
        
        return 0.0
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text
        
        In a real implementation, this would call an embedding API
        like OpenAI's text-embedding-ada-002. For this implementation,
        we'll return a simple hash-based pseudo-embedding.
        """
        # This is just a placeholder - in a real implementation you'd call an embedding API
        # For demo purposes, we'll use a simple hash-based approach
        
        # Hash the text to a fixed-length number
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Convert to a pseudo-embedding (this is NOT a proper embedding, just a placeholder)
        random.seed(hash_val)
        return [random.uniform(-1, 1) for _ in range(self.embed_dim)]
    
    async def _generate_consolidated_text(self, memory_texts: List[str]) -> str:
        """Generate a consolidated memory text from multiple memory texts
        
        In a real implementation, this would use an LLM to generate
        a coherent consolidated memory.
        """
        # This is just a placeholder - in a real implementation you'd call an LLM
        
        # For demo purposes, create a simple summary
        combined = " ".join(memory_texts)
        # Truncate if too long
        if len(combined) > 500:
            combined = combined[:497] + "..."
        
        return f"I've observed a pattern in several memories: {combined}"
    
    def export_memories(self, format_type: str = "json") -> str:
        """Export memories to the specified format"""
        if format_type.lower() == "json":
            return json.dumps({
                "memories": list(self.memories.values()),
                "stats": {
                    "total": len(self.memories),
                    "archived": len(self.archived_memories),
                    "consolidated": len(self.consolidated_memories)
                },
                "export_date": datetime.datetime.now().isoformat()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def import_memories(self, data: str, format_type: str = "json") -> Dict[str, Any]:
        """Import memories from the specified format"""
        if format_type.lower() == "json":
            import_data = json.loads(data)
            imported_count = 0
            
            for memory in import_data.get("memories", []):
                memory_id = memory.get("id", str(uuid.uuid4()))
                self.memories[memory_id] = memory
                
                # Update indices
                memory_type = memory.get("memory_type", "observation")
                if memory_type in self.type_index:
                    self.type_index[memory_type].add(memory_id)
                else:
                    self.type_index[memory_type] = {memory_id}
                
                # Update other indices
                memory_scope = memory.get("memory_scope", "game")
                if memory_scope in self.scope_index:
                    self.scope_index[memory_scope].add(memory_id)
                else:
                    self.scope_index[memory_scope] = {memory_id}
                
                # Update embedding index
                if "embedding" in memory:
                    self.memory_embeddings[memory_id] = memory["embedding"]
                
                # Update archived status
                if memory.get("is_archived", False):
                    self.archived_memories.add(memory_id)
                
                # Update consolidated status
                if memory.get("is_consolidated", False):
                    self.consolidated_memories.add(memory_id)
                
                imported_count += 1
            
            # Clear cache
            self.query_cache = {}
            
            return {
                "memories_imported": imported_count
            }
        else:
            raise ValueError(f"Unsupported import format: {format_type}")
# Add these methods to the MemoryCore class

async def retrieve_memories_with_formatting(self, query: str, memory_types: List[str] = None, 
                                          limit: int = 5) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with formatted results for agent consumption.
    
    Args:
        query: Search query
        memory_types: Types of memories to include
        limit: Maximum number of results
    
    Returns:
        List of formatted memories with confidence markers
    """
    memories = await self.retrieve_memories(
        query=query,
        memory_types=memory_types or ["observation", "reflection", "abstraction", "experience"],
        limit=limit
    )
    
    # Format memories with confidence markers
    formatted_memories = []
    for memory in memories:
        confidence = memory.get("confidence_marker", "remember")
        formatted = {
            "id": memory["id"],
            "text": memory["memory_text"],
            "type": memory["memory_type"],
            "significance": memory["significance"],
            "confidence": confidence,
            "relevance": memory.get("relevance", 0.5),
            "tags": memory.get("tags", [])
        }
        formatted_memories.append(formatted)
    
    return formatted_memories
    
async def create_reflection_from_memories(self, topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a reflection on a specific topic using memories.
    
    Args:
        topic: Optional topic to reflect on
    
    Returns:
        Reflection result dictionary
    """
    # Get relevant memories for reflection
    query = topic if topic else "important memories"
    memories = await self.retrieve_memories(
        query=query,
        memory_types=["observation", "experience"],
        limit=5
    )
    
    if not memories:
        return {
            "reflection": "I don't have enough memories to form a meaningful reflection yet.",
            "confidence": 0.3,
            "topic": topic,
            "reflection_id": None
        }
    
    # Generate reflection text
    # This would normally call the reflection engine, but for compatibility we'll keep it here
    from nyx.core.reflection_engine import ReflectionEngine
    reflection_engine = ReflectionEngine()
    
    reflection_text, confidence = await reflection_engine.generate_reflection(memories, topic)
    
    # Store reflection as a memory
    reflection_id = await self.add_memory(
        memory_text=reflection_text,
        memory_type="reflection",
        memory_scope="game",
        significance=6,
        tags=["reflection"] + ([topic] if topic else []),
        metadata={
            "confidence": confidence,
            "source_memory_ids": [m["id"] for m in memories],
            "emotional_context": {},  # This would normally come from emotional_core
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return {
        "reflection": reflection_text,
        "confidence": confidence,
        "topic": topic,
        "reflection_id": reflection_id
    }
    
async def create_abstraction_from_memories(self, memory_ids: List[str], 
                                        pattern_type: str = "behavior") -> Dict[str, Any]:
    """
    Create a higher-level abstraction from specific memories.
    
    Args:
        memory_ids: IDs of memories to abstract from
        pattern_type: Type of pattern to identify
        
    Returns:
        Abstraction result dictionary
    """
    # Retrieve the specified memories
    memories = []
    for memory_id in memory_ids:
        memory = await self.get_memory(memory_id)
        if memory:
            memories.append(memory)
    
    if not memories:
        return {
            "abstraction": "I couldn't find the specified memories to form an abstraction.",
            "confidence": 0.1,
            "pattern_type": pattern_type
        }
    
    # Generate abstraction
    from nyx.core.reflection_engine import ReflectionEngine
    reflection_engine = ReflectionEngine()
    
    abstraction_text, pattern_data = await reflection_engine.create_abstraction(
        memories=memories,
        pattern_type=pattern_type
    )
    
    # Store abstraction as a memory
    abstraction_id = await self.add_memory(
        memory_text=abstraction_text,
        memory_type="abstraction",
        memory_scope="game",
        significance=7,
        tags=["abstraction", pattern_type],
        metadata={
            "pattern_data": pattern_data,
            "source_memory_ids": memory_ids,
            "emotional_context": {},  # This would normally come from emotional_core
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return {
        "abstraction": abstraction_text,
        "pattern_type": pattern_type,
        "confidence": pattern_data.get("confidence", 0.5),
        "abstraction_id": abstraction_id
    }
    
async def construct_narrative_from_memories(self, topic: str, chronological: bool = True, 
                                          limit: int = 5) -> Dict[str, Any]:
    """
    Construct a narrative from memories about a topic.
    
    Args:
        topic: Topic for narrative construction
        chronological: Whether to maintain chronological order
        limit: Maximum number of memories to include
    
    Returns:
        Narrative result dictionary
    """
    # Retrieve memories for the topic
    memories = await self.retrieve_memories(
        query=topic,
        memory_types=["observation", "experience"],
        limit=limit
    )
    
    if not memories:
        return {
            "narrative": f"I don't have enough memories about {topic} to construct a narrative.",
            "confidence": 0.2,
            "experience_count": 0
        }
    
    # Sort memories chronologically if requested
    if chronological:
        memories.sort(key=lambda m: m.get("metadata", {}).get("timestamp", ""))
    
    # In a real implementation, we would use a more sophisticated narrative generation
    # For now, a simple concatenation with transitions
    narrative_parts = []
    for i, memory in enumerate(memories):
        if i == 0:
            narrative_parts.append(f"I remember that {memory['memory_text']}")
        else:
            transition = "After that" if chronological else "Also"
            narrative_parts.append(f"{transition}, {memory['memory_text']}")
    
    narrative = " ".join(narrative_parts)
    
    # Calculate confidence based on memory count and relevance
    avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
    confidence = 0.4 + (len(memories) / 10) + (avg_relevance * 0.3)  # Scale based on count and relevance
    
    return {
        "narrative": narrative,
        "confidence": min(1.0, confidence),
        "experience_count": len(memories)
    }
