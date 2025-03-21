# nyx/unified_memory_system.py

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import json
import numpy as np
import uuid
import re

from agents import Agent, function_tool, Runner, trace, RunContextWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class MemoryCreate(BaseModel):
    """Model for creating a new memory"""
    memory_text: str = Field(..., description="Text content of the memory")
    memory_type: str = Field("observation", description="Type of memory (observation, reflection, abstraction, experience)")
    significance: int = Field(5, description="Importance of the memory (1-10)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    emotional_context: Optional[Dict[str, Any]] = Field(None, description="Emotional context of the memory")
    
class MemoryQuery(BaseModel):
    """Model for memory query parameters"""
    query: str = Field(..., description="Search query text")
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "abstraction", "experience"], 
                                  description="Types of memories to include in search")
    scopes: List[str] = Field(default_factory=lambda: ["game", "user"], 
                            description="Memory scopes to search in")
    min_significance: int = Field(3, description="Minimum significance level")
    limit: int = Field(5, description="Maximum number of results")
    include_emotional_context: bool = Field(True, description="Include emotional context in results")
    
class ExperienceQuery(BaseModel):
    """Model for querying experiences with additional parameters"""
    query: str = Field(..., description="Search query text")
    scenario_type: Optional[str] = Field(None, description="Type of scenario/experience")
    emotional_state: Optional[Dict[str, Any]] = Field(None, description="Current emotional state")
    entities: List[str] = Field(default_factory=list, description="Entities involved in the experience")
    min_relevance: float = Field(0.6, description="Minimum relevance score (0.0-1.0)")
    limit: int = Field(3, description="Maximum number of results")
    
class ReflectionCreate(BaseModel):
    """Model for creating a reflection"""
    topic: str = Field(..., description="Reflection topic")
    source_memories: List[str] = Field(default_factory=list, description="Memory IDs used as sources")
    confidence: float = Field(0.7, description="Confidence in the reflection (0.0-1.0)")
    
class NarrativeRequest(BaseModel):
    """Model for requesting a narrative from memories"""
    topic: str = Field(..., description="Topic for narrative construction")
    chronological: bool = Field(True, description="Whether to maintain chronological order")
    emotional_context: Optional[Dict[str, Any]] = Field(None, description="Emotional context to influence narrative")
    limit: int = Field(5, description="Maximum number of memories to include")

# --- Main Unified Memory System ---

class UnifiedMemorySystem:
    """
    Unified system that combines memory storage, experience retrieval, 
    reflection, and narrative generation.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core storage layers
        self.raw_storage = None  # Will be initialized to internal storage implementation
        self.vector_storage = None  # Will be initialized to vector storage
        
        # Specialized subsystems
        self.experience_retriever = None  # Will be initialized to ExperienceRetriever
        self.reflection_generator = None  # Will be initialized to ReflectionGenerator
        self.abstraction_generator = None  # Will be initialized to AbstractionGenerator
        
        # Integration status
        self.initialized = False
        
        # LLM models/clients
        self.llm_client = None
        
        # Caching
        self.memory_cache = {}
        self.last_queries = {}
        
        # Initialization counts (for singleton pattern)
        self._instance_count = 0
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'UnifiedMemorySystem':
        """Get or create a singleton instance for the specified user and conversation"""
        # Use a key for the specific user/conversation
        key = f"memory_{user_id}_{conversation_id}"
        
        # Check if instance exists in a global registry
        # This is a simplified approach - you'd need a proper registry in production
        if not hasattr(cls, '_instances'):
            cls._instances = {}
            
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        
        return cls._instances[key]
    
    async def initialize(self):
        """Initialize all subsystems"""
        if self.initialized:
            return
        
        # Increment initialization counter
        self._instance_count += 1
        logger.info(f"Initializing UnifiedMemorySystem {self._instance_count} for user {self.user_id}")
        
        # Initialize raw storage
        from memory.storage import MemoryStorage
        self.raw_storage = MemoryStorage(self.user_id, self.conversation_id)
        await self.raw_storage.initialize()
        
        # Initialize vector storage
        from memory.vector_storage import VectorStorage
        self.vector_storage = VectorStorage(self.user_id, self.conversation_id)
        await self.vector_storage.initialize()
        
        # Initialize specialized subsystems
        from nyx.eternal.experience_retriever import ExperienceRetriever
        self.experience_retriever = ExperienceRetriever(self.user_id, self.conversation_id)
        await self.experience_retriever.initialize()
        
        # Initialize reflection and abstraction generators
        from memory.experience_abstractor import ReflectionGenerator, AbstractionGenerator
        self.reflection_generator = ReflectionGenerator()
        self.abstraction_generator = AbstractionGenerator()
        
        # Initialize LLM client
        import openai
        self.llm_client = openai.AsyncOpenAI()
        
        self.initialized = True
        logger.info(f"UnifiedMemorySystem initialized for user {self.user_id}")
    
    async def add_memory(self, 
                       memory_text: str,
                       memory_type: str = "observation",
                       memory_scope: str = "game",
                       significance: int = 5,
                       tags: List[str] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Create and store a new memory.
        
        Args:
            memory_text: Text content of memory
            memory_type: Type of memory
            memory_scope: Scope of memory (game, user, global)
            significance: Importance level (1-10)
            tags: Optional tags for categorization
            metadata: Additional metadata for memory
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            await self.initialize()
        
        # Set defaults
        tags = tags or []
        metadata = metadata or {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Create memory entry
        memory_id = str(uuid.uuid4())
        
        # Generate embedding for vector search
        embedding = await self._generate_embedding(memory_text)
        
        # Store memory
        memory_data = {
            "id": memory_id,
            "memory_text": memory_text,
            "memory_type": memory_type,
            "memory_scope": memory_scope,
            "significance": significance,
            "tags": tags,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "times_recalled": 0,
            "is_archived": False
        }
        
        # Store in raw storage
        await self.raw_storage.store_memory(memory_data)
        
        # Store in vector storage for semantic search
        await self.vector_storage.store_vector(memory_id, embedding, {
            "memory_text": memory_text,
            "memory_type": memory_type,
            "memory_scope": memory_scope,
            "tags": tags,
            "significance": significance
        })
        
        # Add to cache
        self.memory_cache[memory_id] = memory_data
        
        # Process emotional context if exists in metadata
        if "emotional_context" in metadata:
            # Special processing for emotional memories - update indices
            await self._update_emotional_indices(memory_id, metadata["emotional_context"])
        
        # Process experience memory if applicable
        if memory_type == "experience" or "experience" in tags:
            # Special processing for experience memories
            await self._process_experience_memory(memory_id, memory_data)
        
        return memory_id
    
    async def retrieve_memories(self, 
                             query: str,
                             memory_types: List[str] = None,
                             scopes: List[str] = None,
                             limit: int = 5,
                             min_significance: int = 3,
                             context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories matching query and filters.
        
        Args:
            query: Search query text
            memory_types: Types of memories to retrieve
            scopes: Memory scopes to search in
            limit: Maximum number of memories to return
            min_significance: Minimum significance level
            context: Additional context for search
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            await self.initialize()
        
        # Set defaults
        memory_types = memory_types or ["observation", "reflection", "abstraction", "experience"]
        scopes = scopes or ["game", "user"]
        context = context or {}
        
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        
        # First search vector database for semantic similarity
        vector_results = await self.vector_storage.search_vectors(
            query_embedding, 
            limit=limit*2,  # Get more than needed for filtering
            filters={
                "memory_type": memory_types,
                "memory_scope": scopes,
                "min_significance": min_significance
            }
        )
        
        # Get full memory data for each result
        memories = []
        for result in vector_results:
            memory_id = result["id"]
            
            # Get from cache or storage
            if memory_id in self.memory_cache:
                memory = self.memory_cache[memory_id]
            else:
                memory = await self.raw_storage.get_memory(memory_id)
                if memory:
                    self.memory_cache[memory_id] = memory
                else:
                    continue  # Skip if not found
            
            # Add similarity score from vector search
            memory["relevance"] = result["similarity"]
            
            # Add to results if it meets criteria
            if (memory["memory_type"] in memory_types and
                memory["memory_scope"] in scopes and
                memory["significance"] >= min_significance):
                memories.append(memory)
        
        # Further filter and enhance with context
        enhanced_memories = await self._enhance_memories_with_context(memories, context)
        
        # Sort by relevance
        enhanced_memories.sort(key=lambda m: m.get("relevance", 0), reverse=True)
        
        # Limit results
        results = enhanced_memories[:limit]
        
        # Update recall count for returned memories
        for memory in results:
            await self._update_memory_recall(memory["id"])
        
        return results
    
    async def retrieve_experiences(self,
                                current_context: Dict[str, Any],
                                limit: int = 3,
                                min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to the current context using 
        specialized experience retrieval logic.
        
        Args:
            current_context: Current conversation context including:
                - query: Search query or current topic
                - scenario_type: Type of scenario
                - emotional_state: Current emotional state
                - entities: Entities involved in current context
            limit: Maximum number of experiences to return
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant experiences with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        # Use specialized experience retriever
        experiences = await self.experience_retriever.retrieve_relevant_experiences(
            current_context=current_context,
            limit=limit,
            min_relevance=min_relevance
        )
        
        # Add recall updates
        for experience in experiences:
            if "id" in experience:
                await self._update_memory_recall(experience["id"])
        
        return experiences
    
    async def create_reflection(self,
                             topic: str,
                             query: str = None,
                             source_memories: List[str] = None) -> Dict[str, Any]:
        """
        Create a reflection based on memories.
        
        Args:
            topic: Reflection topic
            query: Optional query to find memories if source_memories not provided
            source_memories: Optional list of specific memory IDs to use as sources
            
        Returns:
            Created reflection data
        """
        if not self.initialized:
            await self.initialize()
        
        # Get source memories if not provided
        if not source_memories and query:
            memory_results = await self.retrieve_memories(
                query=query,
                memory_types=["observation", "experience"],
                limit=5
            )
            source_memories = [m["id"] for m in memory_results]
        
        # Require source memories
        if not source_memories:
            raise ValueError("Either source_memories or query must be provided")
        
        # Get full memory data for sources
        memories = []
        for memory_id in source_memories:
            memory = await self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        # Create reflection using specialized generator
        reflection_text, confidence = await self.reflection_generator.generate_reflection(
            memories=memories,
            topic=topic
        )
        
        # Store reflection as a memory
        reflection_id = await self.add_memory(
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game",
            significance=6,
            tags=["reflection", topic],
            metadata={
                "source_memories": source_memories,
                "confidence": confidence,
                "topic": topic
            }
        )
        
        # Return the created reflection
        reflection = await self.get_memory(reflection_id)
        return reflection
    
    async def create_abstraction(self,
                              source_memories: List[str],
                              pattern_type: str = "behavior") -> Dict[str, Any]:
        """
        Create a higher-level abstraction from specific memories.
        
        Args:
            source_memories: Memory IDs to use as sources
            pattern_type: Type of pattern to identify
            
        Returns:
            Created abstraction data
        """
        if not self.initialized:
            await self.initialize()
        
        # Get full memory data for sources
        memories = []
        for memory_id in source_memories:
            memory = await self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        # Create abstraction using specialized generator
        abstraction_text, pattern_data = await self.abstraction_generator.create_abstraction(
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
                "source_memories": source_memories,
                "pattern_type": pattern_type,
                "pattern_data": pattern_data
            }
        )
        
        # Return the created abstraction
        abstraction = await self.get_memory(abstraction_id)
        return abstraction
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        if not self.initialized:
            await self.initialize()
        
        # Check cache first
        if memory_id in self.memory_cache:
            return self.memory_cache[memory_id]
        
        # Get from storage
        memory = await self.raw_storage.get_memory(memory_id)
        
        # Cache for future use
        if memory:
            self.memory_cache[memory_id] = memory
        
        return memory
    
    async def construct_narrative(self,
                               topic: str,
                               chronological: bool = True,
                               emotional_context: Dict[str, Any] = None,
                               limit: int = 5) -> Dict[str, Any]:
        """
        Construct a coherent narrative from related memories.
        
        Args:
            topic: Topic to construct narrative about
            chronological: Whether to enforce chronological ordering
            emotional_context: Optional emotional context
            limit: Maximum number of memories to include
            
        Returns:
            Narrative data including text and source memories
        """
        if not self.initialized:
            await self.initialize()
        
        # Get relevant memories
        query_context = {"query": topic}
        if emotional_context:
            query_context["emotional_state"] = emotional_context
        
        memories = await self.retrieve_memories(
            query=topic,
            memory_types=["observation", "experience", "reflection"],
            limit=limit,
            context=query_context
        )
        
        if not memories:
            return {
                "narrative": f"I don't have any significant memories about {topic}.",
                "sources": [],
                "confidence": 0.2
            }
        
        # Sort chronologically if required
        if chronological:
            memories.sort(key=lambda m: m.get("metadata", {}).get("timestamp", ""))
        
        # Generate narrative using specialized formatting
        narrative = await self._generate_narrative(memories, topic, emotional_context)
        
        return narrative
    
    async def reconsolidate_memory(self, 
                                memory_id: str, 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reconsolidate (slightly alter) a memory when it's recalled.
        
        Args:
            memory_id: ID of memory to reconsolidate
            context: Current context including emotional state
            
        Returns:
            Updated memory data
        """
        if not self.initialized:
            await self.initialize()
        
        # Get memory
        memory = await self.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")
        
        # Only reconsolidate episodic memories with low/medium significance
        if memory["memory_type"] != "observation" or memory["significance"] >= 8:
            return memory
        
        # Current emotional state can influence reconsolidation
        current_emotion = context.get("emotional_state", {}) if context else {}
        
        # Reconsolidation varies by memory age and significance
        reconsolidation_strength = min(0.3, memory["significance"] / 10.0)
        
        # Generate altered version using LLM
        memory_text = memory["memory_text"]
        altered_memory = await self._generate_reconsolidated_memory(
            memory_text, reconsolidation_strength, current_emotion
        )
        
        # Update memory with altered text
        memory["memory_text"] = altered_memory
        memory["metadata"]["original_form"] = memory.get("metadata", {}).get("original_form", memory_text)
        
        # Add reconsolidation history
        if "reconsolidation_history" not in memory["metadata"]:
            memory["metadata"]["reconsolidation_history"] = []
            
        memory["metadata"]["reconsolidation_history"].append({
            "previous_text": memory_text,
            "timestamp": datetime.now().isoformat(),
            "emotional_context": current_emotion,
            "strength": reconsolidation_strength
        })
        
        # Only store last 3 versions to avoid metadata bloat
        if len(memory["metadata"]["reconsolidation_history"]) > 3:
            memory["metadata"]["reconsolidation_history"] = memory["metadata"]["reconsolidation_history"][-3:]
        
        # Update embedding
        memory["embedding"] = await self._generate_embedding(altered_memory)
        
        # Update in storage
        await self.raw_storage.update_memory(memory_id, memory)
        await self.vector_storage.update_vector(memory_id, memory["embedding"])
        
        # Update cache
        self.memory_cache[memory_id] = memory
        
        return memory
    
    async def merge_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        Merge multiple related memories into a consolidated memory.
        
        Args:
            memory_ids: List of memory IDs to merge
            
        Returns:
            Created consolidated memory
        """
        if not self.initialized:
            await self.initialize()
        
        # Get memories to merge
        memories = []
        for memory_id in memory_ids:
            memory = await self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        if len(memories) < 2:
            raise ValueError("At least two valid memories required for merging")
        
        # Generate consolidated memory text
        consolidated_text = await self._generate_consolidated_memory(memories)
        
        # Calculate average significance (weighted by individual memory significance)
        total_significance = sum(m["significance"] for m in memories)
        avg_significance = total_significance / len(memories)
        
        # Combine tags
        all_tags = set()
        for memory in memories:
            all_tags.update(memory.get("tags", []))
        
        # Create consolidated memory
        consolidated_id = await self.add_memory(
            memory_text=consolidated_text,
            memory_type="consolidated",
            memory_scope="game",
            significance=min(10, int(avg_significance) + 1),  # Slightly higher significance than average
            tags=list(all_tags) + ["consolidated"],
            metadata={
                "source_memory_ids": memory_ids,
                "merge_date": datetime.now().isoformat(),
                "source_memory_types": [m["memory_type"] for m in memories]
            }
        )
        
        # Mark original memories as consolidated
        for memory in memories:
            memory["metadata"]["is_consolidated"] = True
            memory["metadata"]["consolidated_into"] = consolidated_id
            await self.raw_storage.update_memory(memory["id"], memory)
            
            # Update cache
            self.memory_cache[memory["id"]] = memory
        
        # Return the created consolidated memory
        consolidated_memory = await self.get_memory(consolidated_id)
        return consolidated_memory
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Perform memory system maintenance.
        - Apply decay to memories
        - Consolidate related memories
        - Archive old, low-significance memories
        
        Returns:
            Maintenance results
        """
        if not self.initialized:
            await self.initialize()
        
        # Apply memory decay
        decay_result = await self._apply_memory_decay()
        
        # Identify and consolidate clusters
        consolidation_result = await self._consolidate_memory_clusters()
        
        # Archive old memories
        archive_result = await self._archive_old_memories()
        
        return {
            "decay_applied": decay_result["memories_affected"],
            "consolidation_performed": consolidation_result["clusters_consolidated"],
            "memories_archived": archive_result["memories_archived"],
            "maintenance_date": datetime.now().isoformat()
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        if not self.initialized:
            await self.initialize()
        
        # Get counts by type
        type_counts = await self.raw_storage.get_memory_counts_by_type()
        
        # Get recency stats
        recent_counts = await self.raw_storage.get_recent_memory_counts()
        
        # Get top memories by significance
        top_memories = await self.raw_storage.get_top_memories_by_significance(limit=3)
        
        # Calculate decay rate
        all_memories = await self.raw_storage.get_all_memories()
        decayed_count = sum(1 for m in all_memories if "original_significance" in m.get("metadata", {}))
        decay_rate = decayed_count / len(all_memories) if all_memories else 0
        
        # Calculate consolidation stats
        consolidated_count = sum(1 for m in all_memories if m.get("metadata", {}).get("is_consolidated", False))
        consolidation_rate = consolidated_count / len(all_memories) if all_memories else 0
        
        return {
            "total_memories": len(all_memories),
            "type_counts": type_counts,
            "recent_additions": recent_counts,
            "top_memories": [
                {
                    "id": m["id"],
                    "text": m["memory_text"],
                    "significance": m["significance"]
                }
                for m in top_memories
            ],
            "decay_rate": decay_rate,
            "consolidation_rate": consolidation_rate,
            "stats_as_of": datetime.now().isoformat()
        }
    
    # --- Internal helper methods ---
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using LLM API"""
        try:
            response = await self.llm_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zeroed embedding vector as fallback
            return [0.0] * 1536  # Standard embedding size
    
    async def _update_memory_recall(self, memory_id: str):
        """Update recall count and timestamp for a memory"""
        try:
            memory = await self.get_memory(memory_id)
            if memory:
                # Increment recall count
                memory["times_recalled"] = memory.get("times_recalled", 0) + 1
                
                # Update last recalled timestamp
                memory["metadata"] = memory.get("metadata", {})
                memory["metadata"]["last_recalled"] = datetime.now().isoformat()
                
                # Update in storage
                await self.raw_storage.update_memory(memory_id, memory)
                
                # Update cache
                self.memory_cache[memory_id] = memory
        except Exception as e:
            logger.error(f"Error updating memory recall: {e}")
    
    async def _enhance_memories_with_context(self, 
                                         memories: List[Dict[str, Any]],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance memory results with context-specific information"""
        enhanced_memories = []
        
        for memory in memories:
            enhanced = memory.copy()
            
            # Add confidence marker based on relevance
            relevance = enhanced.get("relevance", 0.5)
            enhanced["confidence_marker"] = self._get_confidence_marker(relevance)
            
            # Add emotional context if requested and available
            if context.get("include_emotional_context", True):
                emotional_context = enhanced.get("metadata", {}).get("emotional_context")
                if emotional_context:
                    enhanced["emotional_context"] = emotional_context
                    
                    # Add emotional resonance if current emotional state is provided
                    if "emotional_state" in context:
                        enhanced["emotional_resonance"] = self._calculate_emotional_resonance(
                            emotional_context, context["emotional_state"]
                        )
            
            # Add recency information
            if "metadata" in enhanced and "timestamp" in enhanced["metadata"]:
                timestamp = enhanced["metadata"]["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                days_ago = (datetime.now() - timestamp).days
                enhanced["recency"] = self._get_recency_text(days_ago)
            
            enhanced_memories.append(enhanced)
        
        return enhanced_memories
    
    def _get_confidence_marker(self, relevance: float) -> str:
        """Get confidence marker text based on relevance score"""
        if relevance > 0.8:
            return "vividly recall"
        elif relevance > 0.6:
            return "clearly remember"
        elif relevance > 0.4:
            return "remember"
        elif relevance > 0.2:
            return "think I recall"
        else:
            return "vaguely remember"
    
    def _get_recency_text(self, days_ago: int) -> str:
        """Get human-friendly recency text"""
        if days_ago < 1:
            return "today"
        elif days_ago < 2:
            return "yesterday"
        elif days_ago < 7:
            return f"{days_ago} days ago"
        elif days_ago < 14:
            return "last week"
        elif days_ago < 30:
            return "a few weeks ago"
        elif days_ago < 60:
            return "last month"
        elif days_ago < 365:
            return f"{days_ago // 30} months ago"
        else:
            return f"{days_ago // 365} years ago"
    
    def _calculate_emotional_resonance(self, 
                                     memory_emotions: Dict[str, Any],
                                     current_emotions: Dict[str, Any]) -> float:
        """Calculate how strongly a memory's emotions resonate with current state"""
        # Default if structures don't match expectations
        if not isinstance(memory_emotions, dict) or not isinstance(current_emotions, dict):
            return 0.5
        
        # Extract primary emotion from memory
        memory_primary = memory_emotions.get("primary_emotion", "neutral")
        memory_intensity = memory_emotions.get("primary_intensity", 0.5)
        memory_valence = memory_emotions.get("valence", 0.0)
        
        # Calculate primary emotion match
        current_primary = current_emotions.get("primary_emotion", "neutral")
        current_intensity = current_emotions.get("intensity", 0.5)
        
        # Primary match - exact emotion match gives highest resonance
        primary_match = 1.0 if memory_primary == current_primary else 0.3
        
        # Intensity match - similar intensity gives higher resonance
        intensity_match = 1.0 - abs(memory_intensity - current_intensity)
        
        # Calculate valence match
        current_valence = current_emotions.get("valence", 0.0)
        valence_match = 1.0 - min(1.0, abs(memory_valence - current_valence) / 2.0)
        
        # Combined weighted resonance
        resonance = (
            primary_match * 0.6 +
            intensity_match * 0.2 +
            valence_match * 0.2
        )
        
        return max(0.0, min(1.0, resonance))
    
    async def _update_emotional_indices(self, memory_id: str, emotional_context: Dict[str, Any]):
        """Update emotional indices for improved emotional retrieval"""
        # This would implement specialized indices for emotional content
        # For now, just updating the memory's metadata is sufficient
        pass
    
    async def _process_experience_memory(self, memory_id: str, memory_data: Dict[str, Any]):
        """Process a memory marked as an experience for specialized handling"""
        # Additional processing for experience memories
        # This would connect with the experience retriever system
        pass
    
    async def _generate_narrative(self, 
                               memories: List[Dict[str, Any]], 
                               topic: str,
                               emotional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a coherent narrative from memories"""
        # Extract memory texts and IDs
        memory_texts = [m["memory_text"] for m in memories]
        source_ids = [m["id"] for m in memories]
        
        # Calculate confidence based on memory quality
        avg_significance = sum(m["significance"] for m in memories) / len(memories)
        avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
        
        base_confidence = min(0.9, (avg_significance / 10.0) * 0.5 + avg_relevance * 0.5)
        
        # Prepare emotions string if available
        emotions_str = ""
        if emotional_context:
            emotions_str = f"Current dominant emotion: {emotional_context.get('primary_emotion', 'neutral')}"
            if "valence" in emotional_context:
                valence = emotional_context["valence"]
                emotions_str += f", valence: {valence:+.2f}"
        
        # Use LLM to generate narrative
        try:
            prompt = f"""
            As Nyx, construct a coherent narrative about "{topic}" based on these memories:
            
            {memory_texts}
            
            {emotions_str}
            
            Confidence level: {base_confidence:.2f}
            
            1. Begin the narrative with an appropriate confidence marker.
            2. Weave the memories into a coherent story, filling minimal gaps as needed.
            3. If memories seem contradictory, acknowledge the uncertainty.
            4. Keep it concise (under 200 words).
            5. Write in first person as Nyx.
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Nyx, constructing a narrative from memories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            narrative_text = response.choices[0].message.content.strip()
            
            return {
                "narrative": narrative_text,
                "sources": source_ids,
                "confidence": base_confidence,
                "memory_count": len(memories)
            }
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return {
                "narrative": f"I recall {len(memories)} memories about {topic}, but am having trouble forming a coherent narrative.",
                "sources": source_ids,
                "confidence": base_confidence * 0.5,
                "error": str(e)
            }
    
    async def _generate_reconsolidated_memory(self, 
                                         memory_text: str, 
                                         alteration_strength: float,
                                         emotional_context: Dict[str, Any]) -> str:
        """Generate a slightly altered version of a memory"""
        # Try using LLM
        try:
            emotional_state = ""
            if emotional_context:
                primary_emotion = emotional_context.get("primary_emotion", "neutral")
                emotional_state = f"Current emotional state: {primary_emotion}"
            
            prompt = f"""
            Slightly alter this memory to simulate memory reconsolidation effects.
            
            Original memory: {memory_text}
            {emotional_state}
            
            Create a very slight alteration that:
            1. Maintains the same core information and meaning
            2. Makes subtle changes to wording or emphasis ({int(alteration_strength * 100)}% alteration)
            3. Slightly enhances aspects that align with the current emotional state
            4. Never changes key facts, names, or locations
            
            Return only the altered text with no explanations.
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You subtly alter memories to simulate memory reconsolidation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=len(memory_text) + 50
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error reconsolidating memory: {e}")
            
            # Fallback: minimal random changes
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
    
    async def _generate_consolidated_memory(self, memories: List[Dict[str, Any]]) -> str:
        """Generate a consolidated memory from multiple related memories"""
        # Extract memory texts
        memory_texts = [m["memory_text"] for m in memories]
        
        # Use LLM to generate consolidated text
        try:
            prompt = f"""
            Consolidate these related memory fragments into a single coherent memory:
            
            {memory_texts}
            
            Create a single paragraph that:
            1. Captures the essential pattern or theme across these memories
            2. Generalizes the specific details into broader understanding
            3. Retains the most significant elements
            4. Begins with "I've observed that..." or similar phrase
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You consolidate memory fragments into coherent patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating consolidated memory: {e}")
            return f"I've observed that {memories[0]['memory_text']} and similar patterns in related memories."
    
    async def _apply_memory_decay(self) -> Dict[str, Any]:
        """Apply decay to memories based on age, significance, and recall frequency"""
        # Get all memories
        memories = await self.raw_storage.get_all_memories()
        
        decayed_count = 0
        
        for memory in memories:
            memory_id = memory["id"]
            
            # Skip if not appropriate for decay
            if memory["memory_type"] not in ["observation", "experience"]:
                continue
                
            significance = memory["significance"]
            times_recalled = memory.get("times_recalled", 0)
            
            # Calculate age
            timestamp = memory.get("metadata", {}).get("timestamp")
            if not timestamp:
                continue
                
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
            days_old = (datetime.now() - timestamp).days
            
            # Calculate decay factors
            age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
            recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
            
            # Calculate decay rate
            decay_rate = 0.2 * age_factor * recall_factor
            if days_old > 7 and times_recalled == 0:
                decay_rate *= 1.5  # Extra decay for old memories never recalled
            
            # Skip if decay is negligible
            if decay_rate < 0.05:
                continue
            
            # Store original significance if this is first decay
            if "original_significance" not in memory.get("metadata", {}):
                memory["metadata"] = memory.get("metadata", {})
                memory["metadata"]["original_significance"] = significance
            
            # Apply decay with minimum of 1
            new_significance = max(1, significance - decay_rate)
            
            # Only update if change is significant
            if abs(new_significance - significance) >= 0.5:
                memory["significance"] = new_significance
                memory["metadata"]["last_decay"] = datetime.now().isoformat()
                
                # Update in storage
                await self.raw_storage.update_memory(memory_id, memory)
                
                # Update cache
                self.memory_cache[memory_id] = memory
                
                decayed_count += 1
        
        return {
            "memories_affected": decayed_count,
            "total_memories": len(memories)
        }
    
    async def _find_memory_clusters(self) -> List[List[Dict[str, Any]]]:
        """Group memories into clusters based on embedding similarity"""
        # Get all memories
        memories = await self.raw_storage.get_all_memories()
        
        # We'll use a greedy approach for simplicity
        clusters = []
        unclustered = list(memories)
        
        while unclustered:
            # Take the first memory as a seed
            seed = unclustered.pop(0)
            current_cluster = [seed]
            
            # Skip if already consolidated
            if seed.get("metadata", {}).get("is_consolidated", False):
                continue
                
            # Skip if not an appropriate type for consolidation
            if seed["memory_type"] not in ["observation", "experience"]:
                continue
            
            # Find all similar memories
            i = 0
            while i < len(unclustered):
                memory = unclustered[i]
                
                # Skip if inappropriate type
                if memory["memory_type"] not in ["observation", "experience"]:
                    i += 1
                    continue
                    
                # Skip if already consolidated
                if memory.get("metadata", {}).get("is_consolidated", False):
                    i += 1
                    continue
                
                # Calculate cosine similarity if embeddings exist
                embedding1 = seed.get("embedding")
                embedding2 = memory.get("embedding")
                
                if embedding1 and embedding2:
                    similarity = self._cosine_similarity(embedding1, embedding2)
                    
                    # Add to cluster if similar enough
                    if similarity > 0.85:  # High similarity threshold
                        current_cluster.append(memory)
                        unclustered.pop(i)
                    else:
                        i += 1
                else:
                    i += 1
            
            # Only keep clusters with multiple memories
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        return clusters
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 * norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    async def _consolidate_memory_clusters(self) -> Dict[str, Any]:
        """Identify and consolidate clusters of related memories"""
        # Find clusters
        clusters = await self._find_memory_clusters()
        
        consolidated_count = 0
        
        # Process each cluster
        for cluster in clusters:
            # Skip small clusters
            if len(cluster) < 3:
                continue
                
            # Get memory IDs
            memory_ids = [m["id"] for m in cluster]
            
            try:
                # Merge memories
                await self.merge_memories(memory_ids)
                consolidated_count += 1
            except Exception as e:
                logger.error(f"Error consolidating cluster: {e}")
        
        return {
            "clusters_found": len(clusters),
            "clusters_consolidated": consolidated_count
        }
    
    async def _archive_old_memories(self) -> Dict[str, Any]:
        """Archive old, low-significance memories"""
        # Get all memories
        memories = await self.raw_storage.get_all_memories()
        
        archived_count = 0
        
        for memory in memories:
            memory_id = memory["id"]
            
            # Skip if already archived
            if memory.get("is_archived", False):
                continue
                
            significance = memory["significance"]
            times_recalled = memory.get("times_recalled", 0)
            
            # Calculate age
            timestamp = memory.get("metadata", {}).get("timestamp")
            if not timestamp:
                continue
                
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
            days_old = (datetime.now() - timestamp).days
            
            # Archive if:
            # 1. Low significance and old
            # 2. Never recalled and very old
            # 3. Consolidated into another memory
            if ((significance < 3 and days_old > 30) or
                (times_recalled == 0 and days_old > 60) or
                memory.get("metadata", {}).get("is_consolidated", False)):
                
                # Mark as archived
                memory["is_archived"] = True
                memory["metadata"] = memory.get("metadata", {})
                memory["metadata"]["archived_date"] = datetime.now().isoformat()
                
                # Update in storage
                await self.raw_storage.update_memory(memory_id, memory)
                
                # Update cache
                self.memory_cache[memory_id] = memory
                
                archived_count += 1
        
        return {
            "memories_archived": archived_count,
            "total_memories": len(memories)
        }
    
    # --- OpenAI Agents SDK function tools ---
    
    @function_tool
    async def add_memory_tool(ctx: RunContextWrapper[Any], 
                       memory_text: str,
                       memory_type: str = "observation",
                       significance: int = 5,
                       tags: Optional[List[str]] = None) -> str:
        """
        Add a new memory.
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory (observation, reflection, abstraction, experience)
            significance: Importance level (1-10)
            tags: Optional list of tags for categorization
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        memory_id = await memory_system.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            significance=significance,
            tags=tags or []
        )
        
        return json.dumps({
            "success": True,
            "memory_id": memory_id,
            "memory_type": memory_type
        })
    
    @function_tool
    async def retrieve_memories_tool(ctx: RunContextWrapper[Any],
                             query: str,
                             memory_types: Optional[List[str]] = None,
                             limit: int = 5) -> str:
        """
        Retrieve memories based on query.
        
        Args:
            query: Search query
            memory_types: Types of memories to include
            limit: Maximum number of results
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        memories = await memory_system.retrieve_memories(
            query=query,
            memory_types=memory_types or ["observation", "reflection", "abstraction", "experience"],
            limit=limit
        )
        
        # Format memories for output
        formatted_memories = []
        for memory in memories:
            confidence = memory.get("confidence_marker", "remember")
            formatted = {
                "id": memory["id"],
                "text": memory["memory_text"],
                "type": memory["memory_type"],
                "significance": memory["significance"],
                "confidence": confidence,
                "tags": memory.get("tags", [])
            }
            formatted_memories.append(formatted)
        
        return json.dumps(formatted_memories)
    
    @function_tool
    async def retrieve_experiences_tool(ctx: RunContextWrapper[Any],
                                query: str,
                                scenario_type: Optional[str] = None,
                                emotional_state: Optional[Dict[str, Any]] = None,
                                limit: int = 3) -> str:
        """
        Retrieve experiences relevant to current context.
        
        Args:
            query: Search query
            scenario_type: Optional scenario type
            emotional_state: Optional emotional state
            limit: Maximum number of results
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        context = {
            "query": query,
            "scenario_type": scenario_type
        }
        
        if emotional_state:
            context["emotional_state"] = emotional_state
        
        experiences = await memory_system.retrieve_experiences(
            current_context=context,
            limit=limit
        )
        
        # Format experiences for output
        formatted_experiences = []
        for exp in experiences:
            formatted = {
                "content": exp.get("content", exp.get("memory_text", "")),
                "scenario_type": exp.get("scenario_type", ""),
                "emotional_context": exp.get("emotional_context", {}),
                "relevance_score": exp.get("relevance_score", exp.get("relevance", 0.5)),
                "experiential_richness": exp.get("experiential_richness", 0.5)
            }
            formatted_experiences.append(formatted)
        
        return json.dumps(formatted_experiences)
    
    @function_tool
    async def create_reflection_tool(ctx: RunContextWrapper[Any],
                            topic: str,
                            query: Optional[str] = None) -> str:
        """
        Create a reflection on a specific topic.
        
        Args:
            topic: Topic to reflect on
            query: Optional query to find memories if needed
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        reflection = await memory_system.create_reflection(
            topic=topic,
            query=query
        )
        
        return json.dumps({
            "success": True,
            "reflection_id": reflection["id"],
            "reflection_text": reflection["memory_text"],
            "confidence": reflection.get("metadata", {}).get("confidence", 0.7)
        })
    
    @function_tool
    async def construct_narrative_tool(ctx: RunContextWrapper[Any],
                               topic: str,
                               chronological: bool = True,
                               limit: int = 5) -> str:
        """
        Construct a narrative from memories about a topic.
        
        Args:
            topic: Topic for narrative construction
            chronological: Whether to maintain chronological order
            limit: Maximum number of memories to include
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        narrative = await memory_system.construct_narrative(
            topic=topic,
            chronological=chronological,
            limit=limit
        )
        
        return json.dumps({
            "narrative": narrative["narrative"],
            "confidence": narrative["confidence"],
            "memory_count": narrative["memory_count"]
        })
    
    @function_tool
    async def get_memory_stats_tool(ctx: RunContextWrapper[Any]) -> str:
        """
        Get statistics about the memory system.
        """
        memory_system = ctx.context
        
        if not isinstance(memory_system, UnifiedMemorySystem):
            return json.dumps({"error": "Invalid context type"})
        
        stats = await memory_system.get_memory_stats()
        
        return json.dumps(stats)
