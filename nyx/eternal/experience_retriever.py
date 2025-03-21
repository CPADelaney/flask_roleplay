# nyx/eternal/experience_retriever.py 

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import random

from nyx.core.nyx_brain import NyxBrain
from memory.core import Memory, MemoryType, MemorySignificance
from memory.emotional import EmotionalMemoryManager
from memory.integrated import IntegratedMemorySystem

logger = logging.getLogger("experience_retriever")

class ExperienceRetriever:
    """
    Specialized component for retrieving relevant past experiences.
    Interfaces with the new consolidated architecture while maintaining
    the same functionality and API.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems (will be initialized lazily)
        self.brain = None
        self.memory_system = None  # For backward compatibility
        self.emotional_system = None  # For backward compatibility
        self.initialized = False
        
        # Caching
        self.experience_cache = {}
        self.last_retrieval_time = datetime.now()
        self.relevance_cache = {}
        
        # Confidence mapping 
        self.confidence_markers = {
            (0.8, 1.0): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
    
    async def initialize(self):
        """Initialize the experience retriever."""
        if self.initialized:
            return
            
        # Initialize NyxBrain - the central integration point
        self.brain = await NyxBrain.get_instance(self.user_id, self.conversation_id)
        
        # For backward compatibility with systems that expect these properties
        self.memory_system = await IntegratedMemorySystem.get_instance(
            self.user_id, self.conversation_id
        )
        
        self.emotional_system = await EmotionalMemoryManager(
            self.user_id, self.conversation_id
        )
        
        self.initialized = True
        logger.info(f"Experience retriever initialized for user {self.user_id}")
    
    async def retrieve_relevant_experiences(self, 
                                          current_context: Dict[str, Any],
                                          limit: int = 3,
                                          min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to the current conversation context.
        
        Args:
            current_context: Current conversation context including:
                - query: Search query or current topic
                - scenario_type: Type of scenario (e.g., "teasing", "dark")
                - emotional_state: Current emotional state
                - entities: Entities involved in current context
            limit: Maximum number of experiences to return
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant experiences with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        # Check cache for identical request
        cache_key = f"{current_context.get('query', '')}_{current_context.get('scenario_type', '')}_{limit}_{min_relevance}"
        if cache_key in self.experience_cache:
            cache_time, cache_result = self.experience_cache[cache_key]
            # Use cache if less than 5 minutes old
            cache_age = (datetime.now() - cache_time).total_seconds()
            if cache_age < 300:
                return cache_result
        
        # Get base memories from memory system
        query = current_context.get("query", "")
        base_memories = await self.memory_system.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "episodic"],
            limit=limit*3,  # Get more to filter later
            min_significance=MemorySignificance.MEDIUM
        )
        
        if not base_memories:
            logger.info("No base memories found for experience retrieval")
            return []
        
        # Score memories for relevance
        scored_memories = []
        for memory in base_memories:
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                memory, current_context
            )
            
            # Skip low-relevance memories
            if relevance_score < min_relevance:
                continue
            
            # Get emotional context for this memory
            emotional_context = await self._get_memory_emotional_context(memory)
            
            # Calculate experiential richness (how much detail/emotion it has)
            experiential_richness = self._calculate_experiential_richness(
                memory, emotional_context
            )
            
            # Add to scored memories
            scored_memories.append({
                "memory": memory,
                "relevance_score": relevance_score,
                "emotional_context": emotional_context,
                "experiential_richness": experiential_richness,
                "final_score": relevance_score * 0.7 + experiential_richness * 0.3
            })
        
        # Sort by final score
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Process top memories into experiences
        experiences = []
        for item in scored_memories[:limit]:
            experience = await self._convert_memory_to_experience(
                item["memory"],
                item["emotional_context"],
                item["relevance_score"],
                item["experiential_richness"]
            )
            experiences.append(experience)
        
        # Cache the result
        self.experience_cache[cache_key] = (datetime.now(), experiences)
        self.last_retrieval_time = datetime.now()
        
        return experiences
    
    async def generate_conversational_recall(self, 
                                          experience: Dict[str, Any],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a natural, conversational recall of an experience.
        
        Args:
            experience: The experience to recall
            context: Current conversation context
            
        Returns:
            Conversational recall with reflection
        """
        if not self.initialized:
            await self.initialize()
        
        # Use the brain's experience engine for this
        return await self.brain.experience_engine.generate_conversational_recall(
            experience=experience,
            context=context
        )
    
    async def handle_experience_sharing_request(self,
                                             user_query: str,
                                             context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user request to share experiences.
        
        Args:
            user_query: User's query text
            context_data: Additional context data
            
        Returns:
            Experience sharing response
        """
        if not self.initialized:
            await self.initialize()
        
        # Use the brain's experience engine for this
        return await self.brain.experience_engine.handle_experience_sharing_request(
            user_query=user_query,
            context_data=context_data
        )
    
    async def store_experience(self,
                            memory_text: str,
                            scenario_type: str = "general",
                            entities: List[str] = None,
                            emotional_context: Dict[str, Any] = None,
                            significance: int = 5,
                            tags: List[str] = None) -> Dict[str, Any]:
        """
        Store a new experience in the memory system.
        
        Args:
            memory_text: The memory text
            scenario_type: Type of scenario
            entities: List of entity IDs involved
            emotional_context: Emotional context data
            significance: Memory significance
            tags: Additional tags
            
        Returns:
            Stored experience information
        """
        if not self.initialized:
            await self.initialize()
        
        # Set default tags if not provided
        tags = tags or []
        
        # Add scenario type to tags if not already present
        if scenario_type not in tags:
            tags.append(scenario_type)
        
        # Add experience tag
        if "experience" not in tags:
            tags.append("experience")
            
        # Prepare metadata
        metadata = {
            "scenario_type": scenario_type,
            "entities": entities or [],
            "is_experience": True
        }
        
        # Add emotional context to metadata if provided
        if emotional_context:
            metadata["emotional_context"] = emotional_context
        else:
            # If no emotional context is provided, get current emotional state
            emotional_context = self.brain.emotional_core.get_formatted_emotional_state()
            metadata["emotional_context"] = emotional_context
        
        # Create memory using memory system
        memory = Memory(
            text=memory_text,
            memory_type=MemoryType.OBSERVATION,
            significance=significance,
            emotional_intensity=emotional_context.get("primary_intensity", 0.5) * 100 if emotional_context else 50,
            tags=tags,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        # Store memory using the brain's memory core
        memory_id = await self.brain.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="experience",
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata=metadata
        )
        
        return {
            "memory_id": memory_id,
            "memory_text": memory_text,
            "scenario_type": scenario_type,
            "tags": tags,
            "significance": significance
        }
    
    async def generate_personality_reflection(self,
                                           experiences: List[Dict[str, Any]],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a personality-driven reflection based on experiences.
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Personality-driven reflection
        """
        if not self.initialized:
            await self.initialize()
        
        # Use the brain's experience engine for this
        return await self.brain.experience_engine.generate_personality_reflection(
            experiences=experiences,
            context=context
        )
    
    async def construct_narrative(self,
                               experiences: List[Dict[str, Any]],
                               topic: str,
                               chronological: bool = True) -> Dict[str, Any]:
        """
        Construct a coherent narrative from multiple experiences.
        
        Args:
            experiences: List of experiences to include in narrative
            topic: Topic of the narrative
            chronological: Whether to maintain chronological order
            
        Returns:
            Narrative data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use the brain's experience engine for this
        return await self.brain.experience_engine.construct_narrative(
            experiences=experiences,
            topic=topic,
            chronological=chronological
        )
    
    async def _calculate_relevance_score(self, 
                                      memory: Dict[str, Any], 
                                      context: Dict[str, Any]) -> float:
        """
        Calculate how relevant a memory is to the current context.
        
        Args:
            memory: Memory to evaluate
            context: Current conversation context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Check cache for this calculation
        memory_id = memory.get("id", "")
        context_hash = hash(str(context))
        cache_key = f"{memory_id}_{context_hash}"
        
        if cache_key in self.relevance_cache:
            return self.relevance_cache[cache_key]
        
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        memory_metadata = memory.get("metadata", {})
        
        # Initialize score components
        semantic_score = 0.0
        tag_score = 0.0
        temporal_score = 0.0
        emotional_score = 0.0
        entity_score = 0.0
        
        # Semantic relevance (via embedding similarity)
        # This would ideally use the embedding similarity from your memory system
        if "relevance" in memory:
            semantic_score = memory.get("relevance", 0.0)
        elif "embedding" in memory and "query_embedding" in context:
            # Calculate cosine similarity if both embeddings available
            memory_embedding = np.array(memory["embedding"])
            query_embedding = np.array(context["query_embedding"])
            
            # Cosine similarity
            dot_product = np.dot(memory_embedding, query_embedding)
            norm_product = np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding)
            semantic_score = dot_product / norm_product if norm_product > 0 else 0.0
        else:
            # Fallback: keyword matching
            query = context.get("query", "").lower()
            text = memory_text.lower()
            
            # Simple word overlap
            query_words = set(query.split())
            text_words = set(text.split())
            
            if query_words:
                matches = query_words.intersection(text_words)
                semantic_score = len(matches) / len(query_words)
            else:
                semantic_score = 0.1  # Minimal score for empty query
        
        # Tag matching
        scenario_type = context.get("scenario_type", "").lower()
        context_tags = context.get("tags", [])
        
        # Count matching tags
        matching_tags = 0
        for tag in memory_tags:
            if tag.lower() in scenario_type or tag in context_tags:
                matching_tags += 1
        
        tag_score = min(1.0, matching_tags / 3) if memory_tags else 0.0
        
        # Temporal relevance (newer memories score higher)
        if "timestamp" in memory:
            memory_time = datetime.fromisoformat(memory["timestamp"]) if isinstance(memory["timestamp"], str) else memory["timestamp"]
            age_in_days = (datetime.now() - memory_time).total_seconds() / 86400
            # Newer memories get higher scores, but not too dominant
            temporal_score = max(0.0, 1.0 - (age_in_days / 180))  # 6 month scale
        
        # Emotional relevance
        memory_emotions = memory_metadata.get("emotions", {})
        context_emotions = context.get("emotional_state", {})
        
        if memory_emotions and context_emotions:
            # Compare primary emotions
            memory_primary = memory_emotions.get("primary", {}).get("name", "neutral")
            context_primary = context_emotions.get("primary_emotion", "neutral")
            
            # Emotion match bonus
            if memory_primary == context_primary:
                emotional_score += 0.5
                
            # Emotional intensity comparison
            memory_intensity = memory_emotions.get("primary", {}).get("intensity", 0.5)
            context_intensity = context_emotions.get("intensity", 0.5)
            
            # Similar intensity bonus
            emotional_score += 0.5 * (1.0 - abs(memory_intensity - context_intensity))
        
        # Entity relevance
        memory_entities = memory_metadata.get("entities", [])
        context_entities = context.get("entities", [])
        
        if memory_entities and context_entities:
            matching_entities = len(set(memory_entities).intersection(set(context_entities)))
            entity_score = min(1.0, matching_entities / len(context_entities)) if context_entities else 0.0
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.35 +
            tag_score * 0.20 +
            temporal_score * 0.10 +
            emotional_score * 0.20 +
            entity_score * 0.15
        )
        
        # Cache the result
        self.relevance_cache[cache_key] = final_score
        
        return min(1.0, max(0.0, final_score))
    
    async def _get_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or infer emotional context for a memory.
        
        Args:
            memory: Memory to analyze
            
        Returns:
            Emotional context information
        """
        # Check if memory already has emotional data
        metadata = memory.get("metadata", {})
        if "emotions" in metadata:
            return metadata["emotions"]
        
        if "emotional_context" in metadata:
            return metadata["emotional_context"]
        
        # If memory has emotional intensity but no detailed emotions, infer them
        emotional_intensity = memory.get("emotional_intensity", 0)
        if emotional_intensity > 0:
            # Get memory tags for context
            tags = memory.get("tags", [])
            
            # For best results, analyze the text directly
            # This uses your emotional system's analysis capabilities
            try:
                analysis = await self.emotional_system.analyze_emotional_content(
                    memory.get("memory_text", "")
                )
                return {
                    "primary_emotion": analysis.get("primary_emotion", "neutral"),
                    "primary_intensity": analysis.get("intensity", 0.5),
                    "secondary_emotions": analysis.get("secondary_emotions", {}),
                    "valence": analysis.get("valence", 0.0),
                    "arousal": analysis.get("arousal", 0.0)
                }
            except Exception as e:
                logger.error(f"Error analyzing emotional content: {e}")
                
                # Fallback: infer from intensity and tags
                primary_emotion = "neutral"
                for tag in tags:
                    if tag in ["joy", "sadness", "anger", "fear", "disgust", 
                              "surprise", "anticipation", "trust"]:
                        primary_emotion = tag
                        break
                
                return {
                    "primary_emotion": primary_emotion,
                    "primary_intensity": emotional_intensity / 100,  # Convert to 0-1 scale
                    "secondary_emotions": {},
                    "valence": 0.5 if primary_emotion in ["joy", "anticipation", "trust"] else -0.5,
                    "arousal": 0.7 if emotional_intensity > 50 else 0.3
                }
        
        # Default emotional context if nothing else available
        return {
            "primary_emotion": "neutral",
            "primary_intensity": 0.3,
            "secondary_emotions": {},
            "valence": 0.0,
            "arousal": 0.2
        }
    
    def _calculate_experiential_richness(self, 
                                       memory: Dict[str, Any], 
                                       emotional_context: Dict[str, Any]) -> float:
        """
        Calculate how rich and detailed the experience is.
        Higher scores mean the memory has more emotional and sensory detail.
        
        Args:
            memory: Memory to evaluate
            emotional_context: Emotional context of the memory
            
        Returns:
            Experiential richness score (0.0-1.0)
        """
        # Extract memory attributes
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        significance = memory.get("significance", 3)
        
        # Initialize richness factors
        detail_score = 0.0
        emotional_depth = 0.0
        sensory_richness = 0.0
        significance_score = 0.0
        
        # Text length as a proxy for detail (longer memories might have more detail)
        # Capped at reasonable limits to avoid overly long memories dominating
        word_count = len(memory_text.split())
        detail_score = min(1.0, word_count / 100)  # Cap at 100 words
        
        # Emotional depth from context
        if emotional_context:
            # Primary emotion intensity
            primary_intensity = emotional_context.get("primary_intensity", 0.0)
            
            # Count secondary emotions
            secondary_count = len(emotional_context.get("secondary_emotions", {}))
            
            # Combine for emotional depth
            emotional_depth = 0.7 * primary_intensity + 0.3 * min(1.0, secondary_count / 3)
        
        # Sensory richness via presence of sensory words
        sensory_words = ["see", "saw", "look", "hear", "heard", "sound", 
                        "feel", "felt", "touch", "smell", "scent", "taste"]
        
        sensory_count = sum(1 for word in sensory_words if word in memory_text.lower())
        sensory_richness = min(1.0, sensory_count / 5)  # Cap at 5 sensory words
        
        # Significance as a direct factor
        significance_score = significance / 10.0  # Convert to 0-1 scale
        
        # Combine scores with weights
        richness_score = (
            detail_score * 0.3 +
            emotional_depth * 0.4 +
            sensory_richness * 0.2 +
            significance_score * 0.1
        )
        
        return min(1.0, max(0.0, richness_score))
    
    async def _convert_memory_to_experience(self,
                                         memory: Dict[str, Any],
                                         emotional_context: Dict[str, Any],
                                         relevance_score: float,
                                         experiential_richness: float) -> Dict[str, Any]:
        """
        Convert a raw memory into a rich experience format.
        
        Args:
            memory: The base memory
            emotional_context: Emotional context information
            relevance_score: How relevant this memory is
            experiential_richness: How rich and detailed the experience is
            
        Returns:
            Formatted experience
        """
        # Get memory participants/entities
        metadata = memory.get("metadata", {})
        entities = metadata.get("entities", [])
        
        # Get scenario type from tags
        tags = memory.get("tags", [])
        scenario_types = [tag for tag in tags if tag in [
            "teasing", "dark", "indulgent", "psychological", "nurturing",
            "training", "discipline", "service", "worship", "punishment"
        ]]
        
        scenario_type = scenario_types[0] if scenario_types else "general"
        
        # Format the experience
        experience = {
            "id": memory.get("id"),
            "content": memory.get("memory_text", ""),
            "emotional_context": emotional_context,
            "scenario_type": scenario_type,
            "entities": entities,
            "timestamp": memory.get("timestamp"),
            "relevance_score": relevance_score,
            "experiential_richness": experiential_richness,
            "tags": tags,
            "significance": memory.get("significance", 3)
        }
        
        # Add confidence marker based on relevance
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance_score < max_val:
                experience["confidence_marker"] = marker
                break
                
        if "confidence_marker" not in experience:
            experience["confidence_marker"] = "remember"  # Default
        
        return experience
