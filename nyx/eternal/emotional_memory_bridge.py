# nyx/eternal/emotional_memory_bridge.py

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import json

from nyx.eternal.emotional_framework import EmotionCore, MemoryIntegration, ReflectionLayer
from nyx.memory_integration_sdk import MemoryIntegrationSDK
from agents import Agent, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class EmotionalMemoryBridge:
    """
    Bridge between Nyx's emotional system and memory system, creating
    bidirectional influence between emotions and memories.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems
        self.emotion_core = EmotionCore()
        self.memory_integration = MemoryIntegration()
        self.reflection_layer = ReflectionLayer(self.emotion_core, self.memory_integration)
        
        # Memory SDK
        self.memory_sdk = None
        
        # Emotion-memory mappings
        self.emotion_memory_valence_map = {
            "Joy": 0.8,
            "Trust": 0.6,
            "Anticipation": 0.5,
            "Love": 0.9,
            "Surprise": 0.2,
            "Sadness": -0.6,
            "Fear": -0.7,
            "Anger": -0.8,
            "Disgust": -0.7,
            "Frustration": -0.5
        }
        
        # Memory emotion intensity thresholds
        self.emotion_intensity_thresholds = {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }
        
        # Emotional memory context cache
        self.emotional_context_cache = {}
        
        # Last emotional update time
        self.last_emotional_update = datetime.now()
        
        # Initialize connection pools
        self.connection_initialized = False
    
    async def initialize(self):
        """Initialize the bridge with necessary connections"""
        if self.connection_initialized:
            return
        
        # Initialize memory SDK
        self.memory_sdk = MemoryIntegrationSDK(self.user_id, self.conversation_id)
        await self.memory_sdk.initialize()
        
        # Load initial emotional state from memory if available
        await self._load_emotional_state_from_memory()
        
        self.connection_initialized = True
        logger.info(f"Emotional Memory Bridge initialized for user {self.user_id}")
    
    async def _load_emotional_state_from_memory(self):
        """Load initial emotional state from memory if available"""
        try:
            # Query for emotional state memories
            emotional_memories = await self.memory_sdk.query_memories({
                "tags": ["emotional_state"],
                "limit": 1,
                "sort_by": "timestamp",
                "sort_direction": "desc"
            })
            
            if emotional_memories and len(emotional_memories) > 0:
                memory = emotional_memories[0]
                if "metadata" in memory and "emotional_state" in memory["metadata"]:
                    # Extract emotional state
                    state = memory["metadata"]["emotional_state"]
                    
                    # Update emotion core
                    for emotion, value in state.items():
                        if emotion in self.emotion_core.emotions:
                            self.emotion_core.update_emotion(emotion, value - self.emotion_core.emotions[emotion])
                    
                    logger.info("Loaded emotional state from memory")
        except Exception as e:
            logger.error(f"Error loading emotional state from memory: {e}")
    
    async def retrieve_memories_with_emotional_context(self, 
                                                  query: str, 
                                                  current_emotions: Optional[Dict[str, float]] = None,
                                                  limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories with emotional context influence.
        
        Args:
            query: Search query
            current_emotions: Current emotional state (optional)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories with emotional context
        """
        if not self.connection_initialized:
            await self.initialize()
        
        # Use current emotions if provided, otherwise get from emotion core
        emotions = current_emotions or self.emotion_core.get_emotional_state()
        
        # Calculate emotional relevance boosters
        relevance_boosters = self._calculate_emotional_relevance_boosters(emotions)
        
        # Enhance query with emotional context
        emotional_query_params = {
            "query": query,
            "relevance_modifiers": relevance_boosters,
            "emotion_state_context": emotions,
            "limit": limit
        }
        
        # Retrieve memories with emotional context
        memories = await self.memory_sdk.query_memories(emotional_query_params)
        
        # Process and enhance memories with emotional context
        enhanced_memories = await self._enhance_memories_with_emotional_context(memories, emotions)
        
        # Update emotion core based on retrieved memories
        await self._update_emotions_from_memories(enhanced_memories)
        
        return enhanced_memories
    
    def _calculate_emotional_relevance_boosters(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate memory relevance boost factors based on emotional state.
        
        Higher values for a tag mean memories with that tag will be boosted in relevance.
        """
        relevance_boosters = {}
        
        # Get dominant emotion
        dominant_emotion, dominant_value = max(emotions.items(), key=lambda x: x[1])
        
        # Set boosters based on emotional state
        # Memories matching dominant emotion get highest boost
        relevance_boosters[dominant_emotion.lower()] = 1.5
        
        # Calculate valence (positive/negative) of current emotional state
        valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                     for emotion, value in emotions.items())
        
        # Boost memories with matching valence
        if valence > 0.3:  # Positive emotional state
            relevance_boosters["positive"] = 1.3
            relevance_boosters["negative"] = 0.7
        elif valence < -0.3:  # Negative emotional state
            relevance_boosters["negative"] = 1.3
            relevance_boosters["positive"] = 0.7
        
        # Add boosters for specific emotions above threshold
        for emotion, value in emotions.items():
            if value >= self.emotion_intensity_thresholds["medium"]:
                relevance_boosters[emotion.lower()] = 1.0 + value
        
        return relevance_boosters
    
    async def _enhance_memories_with_emotional_context(self, 
                                                 memories: List[Dict[str, Any]],
                                                 current_emotions: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Enhance retrieved memories with emotional context.
        
        Adds confidence markers, emotional resonance, and contextual notes.
        """
        enhanced_memories = []
        
        for memory in memories:
            # Get or calculate memory emotional context
            emotional_context = await self._get_memory_emotional_context(memory)
            
            # Calculate emotional resonance with current state
            resonance = self._calculate_emotional_resonance(emotional_context, current_emotions)
            
            # Calculate confidence marker based on resonance and memory attributes
            confidence_marker = self._calculate_confidence_marker(memory, resonance)
            
            # Add enhanced attributes
            enhanced_memory = memory.copy()
            enhanced_memory["emotional_context"] = emotional_context
            enhanced_memory["emotional_resonance"] = resonance
            enhanced_memory["confidence_marker"] = confidence_marker
            
            # Add contextual notes based on resonance
            if resonance > 0.7:
                enhanced_memory["context_note"] = "This memory strongly resonates with your current emotional state"
            elif resonance < 0.3:
                enhanced_memory["context_note"] = "This memory feels distant from your current emotional state"
            
            enhanced_memories.append(enhanced_memory)
        
        # Sort by resonance and relevance
        enhanced_memories.sort(
            key=lambda m: (m.get("emotional_resonance", 0) * 0.6 + m.get("relevance", 0) * 0.4),
            reverse=True
        )
        
        return enhanced_memories
    
    async def _get_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get or calculate emotional context for a memory"""
        memory_id = memory.get("id")
        
        # Check cache first
        if memory_id in self.emotional_context_cache:
            return self.emotional_context_cache[memory_id]
        
        # Extract from memory if available
        if "metadata" in memory and "emotional_context" in memory["metadata"]:
            context = memory["metadata"]["emotional_context"]
            self.emotional_context_cache[memory_id] = context
            return context
        
        # Calculate from memory content
        context = await self._calculate_memory_emotional_context(memory)
        
        # Cache result
        if memory_id:
            self.emotional_context_cache[memory_id] = context
        
        return context
    
    async def _calculate_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emotional context from memory content"""
        # Extract memory content
        memory_text = memory.get("memory_text", "")
        memory_tags = memory.get("tags", [])
        
        # Simple heuristic to determine emotion from content and tags
        context = {
            "primary_emotion": "neutral",
            "primary_intensity": 0.5,
            "secondary_emotions": {},
            "valence": 0.0,
            "arousal": 0.3
        }
        
        # Check for emotion tags
        emotion_tags = [tag for tag in memory_tags if tag.lower() in 
                        [e.lower() for e in self.emotion_core.emotions.keys()]]
        
        if emotion_tags:
            context["primary_emotion"] = emotion_tags[0].capitalize()
            context["primary_intensity"] = 0.7
            
            # Add secondary emotions
            for tag in emotion_tags[1:]:
                context["secondary_emotions"][tag.capitalize()] = 0.5
        
        # Calculate valence from tags
        positive_tags = ["positive", "happy", "joy", "love", "trust", "anticipation"]
        negative_tags = ["negative", "sad", "fear", "anger", "disgust", "frustration"]
        
        positive_count = sum(1 for tag in memory_tags if any(p in tag.lower() for p in positive_tags))
        negative_count = sum(1 for tag in memory_tags if any(n in tag.lower() for n in negative_tags))
        
        if positive_count > negative_count:
            context["valence"] = 0.5
        elif negative_count > positive_count:
            context["valence"] = -0.5
        
        # Calculate arousal from memory significance
        if "significance" in memory:
            context["arousal"] = min(1.0, memory["significance"] / 10.0)
        
        return context
    
    def _calculate_emotional_resonance(self, 
                                    memory_emotions: Dict[str, Any],
                                    current_emotions: Dict[str, float]) -> float:
        """
        Calculate how strongly a memory's emotional context resonates with current emotions.
        
        Returns a value from 0.0 (no resonance) to 1.0 (perfect resonance).
        """
        # Extract primary emotion from memory
        memory_primary = memory_emotions.get("primary_emotion", "neutral")
        memory_intensity = memory_emotions.get("primary_intensity", 0.5)
        memory_valence = memory_emotions.get("valence", 0.0)
        
        # Calculate primary emotion match
        primary_match = 0.0
        if memory_primary in current_emotions:
            primary_match = current_emotions[memory_primary] * memory_intensity
        
        # Calculate valence match (how well positive/negative alignment matches)
        current_valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                            for emotion, value in current_emotions.items())
        
        valence_match = 1.0 - min(1.0, abs(current_valence - memory_valence))
        
        # Calculate secondary emotion matches
        secondary_match = 0.0
        secondary_emotions = memory_emotions.get("secondary_emotions", {})
        
        if secondary_emotions:
            matches = []
            for emotion, intensity in secondary_emotions.items():
                if emotion in current_emotions:
                    matches.append(current_emotions[emotion] * intensity)
            
            if matches:
                secondary_match = sum(matches) / len(matches)
        
        # Combined weighted resonance
        resonance = (
            primary_match * 0.5 +
            valence_match * 0.3 +
            secondary_match * 0.2
        )
        
        return max(0.0, min(1.0, resonance))
    
    def _calculate_confidence_marker(self, memory: Dict[str, Any], resonance: float) -> str:
        """Calculate confidence marker for memory display based on resonance and attributes"""
        # Base confidence on resonance and memory attributes
        base_confidence = resonance * 0.6
        
        # Add confidence from memory attributes
        if "significance" in memory:
            base_confidence += min(0.3, memory["significance"] / 30.0)
        
        if "times_recalled" in memory:
            base_confidence += min(0.1, memory["times_recalled"] / 10.0)
        
        # Map confidence to marker
        if base_confidence > 0.8:
            return "vividly recall"
        elif base_confidence > 0.6:
            return "clearly remember"
        elif base_confidence > 0.4:
            return "remember"
        elif base_confidence > 0.2:
            return "think I recall"
        else:
            return "vaguely remember"
    
    async def _update_emotions_from_memories(self, memories: List[Dict[str, Any]]):
        """Update emotion core based on retrieved memories"""
        # Skip if no memories or too soon since last update
        if not memories:
            return
            
        now = datetime.now()
        seconds_since_update = (now - self.last_emotional_update).total_seconds()
        if seconds_since_update < 60:  # Limit to once per minute
            return
        
        # Calculate emotional stimuli from memories
        stimuli = {}
        
        for memory in memories:
            # Get memory emotional context
            emotional_context = memory.get("emotional_context", {})
            resonance = memory.get("emotional_resonance", 0.3)
            
            # Primary emotion influence
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if primary_emotion and primary_emotion in self.emotion_core.emotions:
                # Scale by resonance and memory relevance
                relevance = memory.get("relevance", 0.5)
                influence = primary_intensity * resonance * relevance * 0.1
                
                if primary_emotion not in stimuli:
                    stimuli[primary_emotion] = 0
                    
                stimuli[primary_emotion] += influence
            
            # Secondary emotions influence
            secondary_emotions = emotional_context.get("secondary_emotions", {})
            for emotion, intensity in secondary_emotions.items():
                if emotion in self.emotion_core.emotions:
                    # Scaled down influence for secondary emotions
                    influence = intensity * resonance * 0.05
                    
                    if emotion not in stimuli:
                        stimuli[emotion] = 0
                        
                    stimuli[emotion] += influence
        
        # Apply emotional stimuli
        if stimuli:
            self.emotion_core.update_from_stimuli(stimuli)
            self.last_emotional_update = now
    
    async def store_memory_with_emotional_context(self,
                                             memory_text: str,
                                             memory_type: str = "observation",
                                             significance: int = 5,
                                             tags: List[str] = None) -> str:
        """
        Store a new memory with current emotional context.
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory (observation, reflection, abstraction)
            significance: Importance level (1-10)
            tags: Optional list of tags to associate with memory
            
        Returns:
            Memory ID
        """
        if not self.connection_initialized:
            await self.initialize()
        
        # Get current emotional state
        current_emotions = self.emotion_core.get_emotional_state()
        
        # Calculate emotional context for the memory
        emotional_context = {
            "primary_emotion": self.emotion_core.get_dominant_emotion()[0],
            "primary_intensity": self.emotion_core.get_dominant_emotion()[1],
            "secondary_emotions": {
                k: v for k, v in current_emotions.items()
                if v >= self.emotion_intensity_thresholds["medium"] and k != self.emotion_core.get_dominant_emotion()[0]
            },
            "valence": sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                          for emotion, value in current_emotions.items()),
            "arousal": sum(value for value in current_emotions.values()) / len(current_emotions)
        }
        
        # Add emotional tags
        if tags is None:
            tags = []
            
        # Add dominant emotion as tag
        dom_emotion = self.emotion_core.get_dominant_emotion()[0].lower()
        if dom_emotion not in [t.lower() for t in tags]:
            tags.append(dom_emotion)
            
        # Add valence tag
        valence = emotional_context["valence"]
        if valence > 0.3 and "positive" not in tags:
            tags.append("positive")
        elif valence < -0.3 and "negative" not in tags:
            tags.append("negative")
        
        # Store memory with emotional context in metadata
        memory_id = await self.memory_sdk.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata={
                "emotional_context": emotional_context,
                "creation_emotional_state": current_emotions,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return memory_id
    
    async def generate_emotional_reflection(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a reflection based on emotions and memories.
        
        Args:
            topic: Optional topic to focus reflection on
            
        Returns:
            Reflection data
        """
        if not self.connection_initialized:
            await self.initialize()
        
        # Get recent memories for reflection context
        query = topic if topic else "recent important memories"
        
        memories = await self.retrieve_memories_with_emotional_context(
            query=query,
            limit=5
        )
        
        # Generate reflection using the reflection layer
        reflection = self.reflection_layer.generate_reflection()
        
        # Store reflection as a memory
        reflection_text = reflection.get("reflection", "")
        if reflection_text:
            await self.store_memory_with_emotional_context(
                memory_text=reflection_text,
                memory_type="reflection",
                significance=7,
                tags=["reflection", "emotional"] + ([topic] if topic else [])
            )
        
        return reflection
    
    async def update_emotional_state(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Update the emotional state directly and record in memory.
        
        Args:
            emotions: Dictionary of emotions and their values
            
        Returns:
            Updated emotional state
        """
        if not self.connection_initialized:
            await self.initialize()
        
        # Update emotion core
        for emotion, value in emotions.items():
            if emotion in self.emotion_core.emotions:
                # Update as absolute value, not delta
                current = self.emotion_core.emotions[emotion]
                self.emotion_core.update_emotion(emotion, value - current)
        
        # Get updated state
        updated_state = self.emotion_core.get_emotional_state()
        
        # Store emotional state update in memory
        await self.memory_sdk.add_memory(
            memory_text=f"Emotional state updated: {', '.join(f'{e}: {v:.2f}' for e, v in updated_state.items())}",
            memory_type="system",
            memory_scope="game",
            significance=4,
            tags=["emotional_state", "system"],
            metadata={
                "emotional_state": updated_state,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "emotional_state": updated_state,
            "dominant_emotion": self.emotion_core.get_dominant_emotion()
        }

    # OpenAI Agents SDK function tools
    
    @function_tool
    async def retrieve_emotional_memories(ctx, query: str, limit: int = 5) -> str:
        """
        Retrieve memories with emotional context influence.
        
        Args:
            query: Search query
            limit: Maximum number of memories to retrieve
        """
        bridge = ctx.context
        if not isinstance(bridge, EmotionalMemoryBridge):
            return json.dumps({"error": "Invalid context type"})
        
        memories = await bridge.retrieve_memories_with_emotional_context(query, limit=limit)
        
        formatted_memories = []
        for memory in memories:
            formatted = {
                "text": memory.get("memory_text", ""),
                "confidence": memory.get("confidence_marker", "remember"),
                "resonance": memory.get("emotional_resonance", 0.5),
                "context_note": memory.get("context_note", ""),
                "emotional_context": memory.get("emotional_context", {})
            }
            formatted_memories.append(formatted)
        
        return json.dumps(formatted_memories)
    
    @function_tool
    async def store_emotional_memory(ctx, memory_text: str, 
                             memory_type: str = "observation",
                             significance: int = 5,
                             tags: Optional[List[str]] = None) -> str:
        """
        Store a new memory with emotional context.
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory (observation, reflection, abstraction)
            significance: Importance level (1-10)
            tags: Optional list of tags
        """
        bridge = ctx.context
        if not isinstance(bridge, EmotionalMemoryBridge):
            return json.dumps({"error": "Invalid context type"})
        
        memory_id = await bridge.store_memory_with_emotional_context(
            memory_text=memory_text,
            memory_type=memory_type,
            significance=significance,
            tags=tags or []
        )
        
        return json.dumps({
            "success": True,
            "memory_id": memory_id,
            "emotional_context": bridge.emotion_core.get_emotional_state()
        })
    
    @function_tool
    async def get_emotional_state(ctx) -> str:
        """
        Get the current emotional state.
        """
        bridge = ctx.context
        if not isinstance(bridge, EmotionalMemoryBridge):
            return json.dumps({"error": "Invalid context type"})
        
        emotional_state = bridge.emotion_core.get_emotional_state()
        dominant_emotion = bridge.emotion_core.get_dominant_emotion()
        
        return json.dumps({
            "emotional_state": emotional_state,
            "dominant_emotion": {
                "emotion": dominant_emotion[0],
                "intensity": dominant_emotion[1]
            }
        })
    
    @function_tool
    async def update_emotion(ctx, emotion: str, value: float) -> str:
        """
        Update a specific emotion value.
        
        Args:
            emotion: The emotion to update
            value: The delta change in emotion value (-1.0 to 1.0)
        """
        bridge = ctx.context
        if not isinstance(bridge, EmotionalMemoryBridge):
            return json.dumps({"error": "Invalid context type"})
        
        # Validate input
        if not -1.0 <= value <= 1.0:
            return json.dumps({
                "error": "Value must be between -1.0 and 1.0"
            })
        
        if emotion not in bridge.emotion_core.emotions:
            return json.dumps({
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(bridge.emotion_core.emotions.keys())
            })
        
        # Update emotion
        bridge.emotion_core.update_emotion(emotion, value)
        
        # Get updated state
        updated_state = bridge.emotion_core.get_emotional_state()
        
        return json.dumps({
            "success": True,
            "updated_emotion": emotion,
            "change": value,
            "new_value": updated_state[emotion],
            "emotional_state": updated_state
        })
