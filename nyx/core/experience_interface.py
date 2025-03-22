# nyx/core/experience_interface.py

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import random
from pydantic import BaseModel, Field

# Import OpenAI Agents SDK components
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool
)

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class ExperienceOutput(BaseModel):
    """Schema for experience retrieval output"""
    experience_text: str = Field(..., description="The formatted experience text")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    source_id: Optional[str] = Field(None, description="Source memory ID")
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    emotional_context: Optional[Dict[str, Any]] = Field(None, description="Emotional context data")

class ReflectionOutput(BaseModel):
    """Schema for reflection output"""
    reflection_text: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    experience_ids: List[str] = Field(default_factory=list, description="IDs of experiences used")
    insight_level: float = Field(..., description="Depth of insight (0.0-1.0)")

class NarrativeOutput(BaseModel):
    """Schema for narrative output"""
    narrative_text: str = Field(..., description="The narrative text")
    experiences_included: int = Field(..., description="Number of experiences included")
    coherence_score: float = Field(..., description="Narrative coherence (0.0-1.0)")
    chronological: bool = Field(..., description="Whether narrative is chronological")

class ExperienceContextData(BaseModel):
    """Context data for experience operations"""
    query: str = Field(..., description="Query or topic")
    scenario_type: Optional[str] = Field(None, description="Type of scenario")
    emotional_state: Dict[str, Any] = Field(default_factory=dict, description="Current emotional state")
    entities: List[str] = Field(default_factory=list, description="Entities involved")
    user_id: Optional[str] = Field(None, description="User ID")
    timestamp: Optional[str] = Field(None, description="Operation timestamp")

class ExperienceInterface:
    """
    Agent-based interface for managing, retrieving, and formatting experience-related memories.
    Provides natural language experience sharing and recall functionality using OpenAI Agents SDK.
    """
    
    def __init__(self, memory_core, emotional_core):
        """
        Initialize the experience interface.
        
        Args:
            memory_core: The memory core for storing and retrieving memories
            emotional_core: The emotional core for emotional processing
        """
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        
        # Set up agents
        self.experience_agent = self._create_experience_agent()
        self.reflection_agent = self._create_reflection_agent()
        self.narrative_agent = self._create_narrative_agent()
        self.recall_agent = self._create_recall_agent()
        
        # Caching
        self.experience_cache = {}
        self.last_retrieval_time = datetime.now()
        
        # Template patterns for experience recall formatting
        self.recall_templates = {
            # Basic recall templates
            "standard": [
                "That reminds me of {timeframe} when {brief_summary}... {detail}",
                "I recall {timeframe} when {brief_summary}. {reflection}",
                "This is similar to {timeframe} when {brief_summary}... {detail}"
            ],
            
            # For emotionally positive experiences
            "positive": [
                "Mmm, I remember {timeframe} when {brief_summary}. {reflection}",
                "That brings back a delicious memory of {timeframe} when {brief_summary}... {detail}",
                "I quite enjoyed {timeframe} when {brief_summary}. {reflection}"
            ],
            
            # For emotionally negative experiences
            "negative": [
                "I recall {timeframe} dealing with someone who {brief_summary}. {reflection}",
                "This reminds me of a frustrating time when {brief_summary}... {detail}",
                "I once had to handle someone who {brief_summary}. {reflection}"
            ],
            
            # For intense experiences
            "intense": [
                "Mmm, that reminds me of an *intense* experience where {brief_summary}... {detail}",
                "I vividly remember when {brief_summary}. {reflection}",
                "I'll never forget when {brief_summary}... {detail}"
            ],
            
            # For teasing experiences
            "teasing": [
                "Oh, this reminds me of {timeframe} when I teased someone until {brief_summary}... {reflection}",
                "I once had such fun teasing someone who {brief_summary}. {reflection}",
                "There was this delicious time when I {brief_summary}... {detail}"
            ],
            
            # For disciplinary experiences
            "disciplinary": [
                "I remember having to discipline someone who {brief_summary}. {reflection}",
                "This reminds me of {timeframe} when I had to correct someone who {brief_summary}... {detail}",
                "I once dealt with someone who needed strict handling when they {brief_summary}. {reflection}"
            ]
        }
        
        # Confidence mapping
        self.confidence_markers = {
            (0.8, 1.0): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
        
    def _create_experience_agent(self):
        """Create the main experience agent"""
        return Agent(
            name="Experience Agent",
            instructions="""
            You are the Experience Agent, responsible for managing and retrieving experiences.
            Your role is to determine the best way to handle experience-related requests and
            coordinate with specialized agents when appropriate.
            """,
            handoffs=[
                handoff(self._create_reflection_agent()),
                handoff(self._create_narrative_agent()),
                handoff(self._create_recall_agent())
            ],
            tools=[
                function_tool(self._retrieve_experiences),
                function_tool(self._get_emotional_context),
                function_tool(self._store_experience)
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._experience_request_guardrail)
            ]
        )
    
    def _create_reflection_agent(self):
        """Create the reflection agent for generating reflections from experiences"""
        return Agent(
            name="Reflection Agent",
            handoff_description="Specialist agent for generating reflections from experiences",
            instructions="""
            You are the Reflection Agent, specialized in generating insightful reflections
            based on experiences. Create thoughtful, introspective reflections that show
            personality and emotional growth.
            """,
            tools=[
                function_tool(self._retrieve_experiences),
                function_tool(self._get_emotional_context)
            ],
            output_type=ReflectionOutput
        )
    
    def _create_narrative_agent(self):
        """Create the narrative agent for constructing coherent narratives"""
        return Agent(
            name="Narrative Agent",
            handoff_description="Specialist agent for constructing narratives from experiences",
            instructions="""
            You are the Narrative Agent, specialized in constructing coherent narratives
            from multiple experiences. Weave experiences together in a way that creates
            a compelling and meaningful story.
            """,
            tools=[
                function_tool(self._retrieve_experiences),
                function_tool(self._get_timeframe_text)
            ],
            output_type=NarrativeOutput
        )
    
    def _create_recall_agent(self):
        """Create the recall agent for conversational experience recall"""
        return Agent(
            name="Recall Agent",
            handoff_description="Specialist agent for conversational recall of experiences",
            instructions="""
            You are the Recall Agent, specialized in conversational recall of experiences.
            Generate natural, engaging, and emotionally appropriate recollections of experiences.
            """,
            tools=[
                function_tool(self._get_emotional_tone),
                function_tool(self._get_scenario_tone),
                function_tool(self._get_timeframe_text),
                function_tool(self._get_confidence_marker)
            ],
            output_type=ExperienceOutput
        )
    
    # Guardrail functions
    
    async def _experience_request_guardrail(self, ctx, agent, input_data):
        """Guardrail to validate experience requests"""
        # Check for minimum context
        if isinstance(input_data, str) and len(input_data.strip()) < 3:
            return GuardrailFunctionOutput(
                output_info={"error": "Request too short"},
                tripwire_triggered=True
            )
        
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )
    
    # Tool functions
    
    @function_tool
    async def _retrieve_experiences(self, ctx: RunContextWrapper, 
                                query: str,
                                scenario_type: Optional[str] = None,
                                limit: int = 3,
                                min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve relevant experiences based on query and scenario type
        
        Args:
            query: Search query
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of experiences to return
            min_relevance: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant experiences with metadata
        """
        # Create context dict
        context = {
            "query": query,
            "scenario_type": scenario_type or "",
            "emotional_state": self.emotional_core.get_formatted_emotional_state(),
            "entities": []
        }
        
        # Check cache for identical request
        cache_key = f"{query}_{scenario_type}_{limit}_{min_relevance}"
        if cache_key in self.experience_cache:
            cache_time, cache_result = self.experience_cache[cache_key]
            # Use cache if less than 5 minutes old
            cache_age = (datetime.now() - cache_time).total_seconds()
            if cache_age < 300:
                return cache_result
        
        # Get base memories from memory system
        base_memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "episodic", "experience"],
            limit=limit*3,  # Get more to filter later
            context=context
        )
        
        if not base_memories:
            logger.info("No base memories found for experience retrieval")
            return []
        
        # Score memories for relevance
        scored_memories = []
        for memory in base_memories:
            # Get emotional context for this memory
            emotional_context = await self._get_memory_emotional_context(memory)
            
            # Calculate experiential richness
            experiential_richness = self._calculate_experiential_richness(
                memory, emotional_context
            )
            
            # Add to scored memories
            scored_memories.append({
                "memory": memory,
                "relevance_score": memory.get("relevance", 0.5),
                "emotional_context": emotional_context,
                "experiential_richness": experiential_richness,
                "final_score": memory.get("relevance", 0.5) * 0.7 + experiential_richness * 0.3
            })
        
        # Sort by final score
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Process top memories into experiences
        experiences = []
        for item in scored_memories[:limit]:
            if item["final_score"] >= min_relevance:
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
    
    @function_tool
    async def _get_emotional_context(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """Get current emotional context"""
        return self.emotional_core.get_formatted_emotional_state()
    
    @function_tool
    async def _store_experience(self, ctx: RunContextWrapper,
                             memory_text: str,
                             scenario_type: str = "general",
                             entities: List[str] = None,
                             emotional_context: Dict[str, Any] = None,
                             significance: int = 5,
                             tags: List[str] = None) -> Dict[str, Any]:
        """
        Store a new experience in the memory system
        
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
        elif self.emotional_core:
            # If no emotional context is provided, get current emotional state
            emotional_context = self.emotional_core.get_formatted_emotional_state()
            metadata["emotional_context"] = emotional_context
        
        # Store memory using the memory core
        memory_id = await self.memory_core.add_memory(
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
    
    @function_tool
    async def _get_emotional_tone(self, ctx: RunContextWrapper, 
                             emotional_context: Dict[str, Any]) -> str:
        """
        Determine the emotional tone for recall based on emotions
        
        Args:
            emotional_context: Emotional context data
            
        Returns:
            Emotional tone string
        """
        if not emotional_context:
            return "standard"
            
        primary = emotional_context.get("primary_emotion", "neutral")
        intensity = emotional_context.get("primary_intensity", 0.5)
        valence = emotional_context.get("valence", 0.0)
        
        # High intensity experiences
        if intensity > 0.8:
            return "intense"
            
        # Positive emotions
        if valence > 0.3 or primary in ["Joy", "Anticipation", "Trust", "Love"]:
            return "positive"
            
        # Negative emotions
        if valence < -0.3 or primary in ["Anger", "Fear", "Disgust", "Sadness", "Frustration"]:
            return "negative"
            
        # Default to standard
        return "standard"
    
    @function_tool
    async def _get_scenario_tone(self, ctx: RunContextWrapper, 
                              scenario_type: str) -> str:
        """
        Get tone based on scenario type
        
        Args:
            scenario_type: Type of scenario
            
        Returns:
            Scenario tone string
        """
        scenario_type = scenario_type.lower()
        
        if scenario_type in ["teasing", "indulgent"]:
            return "teasing"
        elif scenario_type in ["discipline", "punishment", "training"]:
            return "disciplinary"
        elif scenario_type in ["dark", "fear"]:
            return "intense"
        
        # No specific tone for this scenario type
        return "standard"
    
    @function_tool
    async def _get_timeframe_text(self, ctx: RunContextWrapper, 
                             timestamp: Optional[str]) -> str:
        """
        Get conversational timeframe text from timestamp
        
        Args:
            timestamp: ISO timestamp string
            
        Returns:
            Natural language timeframe text
        """
        if not timestamp:
            return "a while back"
            
        try:
            if isinstance(timestamp, str):
                memory_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                memory_time = timestamp
                
            days_ago = (datetime.now() - memory_time).days
            
            if days_ago < 1:
                return "earlier today"
            elif days_ago < 2:
                return "yesterday"
            elif days_ago < 7:
                return f"{days_ago} days ago"
            elif days_ago < 14:
                return "last week"
            elif days_ago < 30:
                return "a couple weeks ago"
            elif days_ago < 60:
                return "a month ago"
            elif days_ago < 365:
                return f"{days_ago // 30} months ago"
            else:
                return "a while back"
                
        except Exception as e:
            logger.error(f"Error processing timestamp: {e}")
            return "a while back"
    
    @function_tool
    async def _get_confidence_marker(self, ctx: RunContextWrapper, 
                                relevance: float) -> str:
        """
        Get confidence marker text based on relevance score
        
        Args:
            relevance: Relevance score (0.0-1.0)
            
        Returns:
            Confidence marker text
        """
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance < max_val:
                return marker
        return "remember"  # Default
    
    # Helper functions
    
    async def _get_memory_emotional_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or infer emotional context for a memory
        
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
            
            # Use emotional_core if available
            if self.emotional_core:
                try:
                    analysis = self.emotional_core.analyze_text_sentiment(memory.get("memory_text", ""))
                    
                    # Get primary emotion
                    emotions = {k: v for k, v in analysis.items() if v > 0}
                    if emotions:
                        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                        primary_intensity = emotions[primary_emotion]
                        
                        # Other emotions as secondary
                        secondary_emotions = {k: v for k, v in emotions.items() if k != primary_emotion}
                        
                        # Calculate valence
                        positive_emotions = ["Joy", "Trust", "Anticipation", "Love"]
                        negative_emotions = ["Sadness", "Fear", "Anger", "Disgust", "Frustration"]
                        
                        valence = 0.0
                        if primary_emotion in positive_emotions:
                            valence = 0.5 + (primary_intensity * 0.5)
                        elif primary_emotion in negative_emotions:
                            valence = -0.5 - (primary_intensity * 0.5)
                        
                        return {
                            "primary_emotion": primary_emotion,
                            "primary_intensity": primary_intensity,
                            "secondary_emotions": secondary_emotions,
                            "valence": valence,
                            "arousal": primary_intensity
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
        Calculate how rich and detailed the experience is
        
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
        
        # Text length as a proxy for detail
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
        Convert a raw memory into a rich experience format
        
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
    
    # Public API methods
    
    async def retrieve_experiences_enhanced(self, 
                                         query: str,
                                         scenario_type: Optional[str] = None,
                                         limit: int = 3) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval of experiences based on query and scenario type
        
        Args:
            query: Search query
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of experiences to return
            
        Returns:
            List of relevant experiences with metadata
        """
        # Use the Agent SDK to run the experience agent
        with trace(workflow_name="ExperienceRetrieval"):
            context = ExperienceContextData(
                query=query,
                scenario_type=scenario_type,
                emotional_state=self.emotional_core.get_formatted_emotional_state(),
                entities=[],
                timestamp=datetime.now().isoformat()
            )
            
            result = await Runner.run(
                self.experience_agent,
                f"Retrieve {limit} experiences related to: {query}" + 
                (f" with scenario type: {scenario_type}" if scenario_type else ""),
                context=context
            )
            
            # The agent may have reformatted experiences as part of its output
            if hasattr(result, "experiences") and result.experiences:
                return result.experiences
            
            # Fall back to direct retrieval if agent didn't return experiences
            return await self._retrieve_experiences(
                RunContextWrapper(context=context),
                query=query,
                scenario_type=scenario_type,
                limit=limit
            )
    
    async def generate_conversational_recall(self, 
                                          experience: Dict[str, Any],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a natural, conversational recall of an experience
        
        Args:
            experience: The experience to recall
            context: Current conversation context
            
        Returns:
            Conversational recall with reflection
        """
        # Use the Recall Agent for this task
        with trace(workflow_name="ConversationalRecall"):
            context_data = ExperienceContextData(
                query="recall experience",
                scenario_type=experience.get("scenario_type", "general"),
                emotional_state=self.emotional_core.get_formatted_emotional_state(),
                entities=experience.get("entities", []),
                timestamp=datetime.now().isoformat()
            )
            
            # Create the agent input
            agent_input = {
                "role": "user",
                "content": f"Generate a conversational recall of this experience: {experience.get('content', '')}",
                "experience": experience
            }
            
            result = await Runner.run(
                self.recall_agent,
                agent_input,
                context=context_data
            )
            
            # Return the agent's structured output
            recall_output = result.final_output_as(ExperienceOutput)
            
            return {
                "recall_text": recall_output.experience_text,
                "confidence": recall_output.confidence,
                "experience": experience
            }
    
    async def handle_experience_sharing_request(self,
                                             user_query: str,
                                             context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user request to share experiences
        
        Args:
            user_query: User's query text
            context_data: Additional context data
            
        Returns:
            Experience sharing response
        """
        context_data = context_data or {}
        
        # Use the agent system for handling this request
        with trace(workflow_name="ExperienceSharing"):
            # Prepare context
            context = ExperienceContextData(
                query=user_query,
                scenario_type=context_data.get("scenario_type", ""),
                emotional_state=self.emotional_core.get_formatted_emotional_state(),
                entities=context_data.get("entities", []),
                timestamp=datetime.now().isoformat()
            )
            
            # Run the experience agent
            result = await Runner.run(
                self.experience_agent,
                f"Share an experience related to: {user_query}",
                context=context
            )
            
            # Process the result
            if hasattr(result, "final_output") and result.final_output:
                # Check if we have a structured output
                if isinstance(result.final_output, ExperienceOutput):
                    return {
                        "has_experience": True,
                        "response_text": result.final_output.experience_text,
                        "confidence": result.final_output.confidence,
                        "experience": {"id": result.final_output.source_id}
                    }
                # Otherwise use the text output
                else:
                    return {
                        "has_experience": True,
                        "response_text": str(result.final_output),
                        "confidence": 0.5,  # Default confidence
                        "experience": None
                    }
            
            # No experiences found
            return {
                "has_experience": False,
                "response_text": None
            }
    
    async def generate_personality_reflection(self,
                                           experiences: List[Dict[str, Any]],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a personality-driven reflection based on experiences
        
        Args:
            experiences: List of experiences to reflect on
            context: Current conversation context
            
        Returns:
            Personality-driven reflection
        """
        if not experiences:
            return {
                "reflection": "I don't have specific experiences to reflect on yet.",
                "confidence": 0.3
            }
        
        # Use the Reflection Agent for this task
        with trace(workflow_name="PersonalityReflection"):
            context_data = ExperienceContextData(
                query="reflect on experiences",
                scenario_type="",
                emotional_state=self.emotional_core.get_formatted_emotional_state(),
                timestamp=datetime.now().isoformat()
            )
            
            # Create input with experiences
            agent_input = {
                "role": "user",
                "content": "Generate a reflection based on these experiences",
                "experiences": experiences
            }
            
            result = await Runner.run(
                self.reflection_agent,
                agent_input,
                context=context_data
            )
            
            # Get the reflection output
            reflection_output = result.final_output_as(ReflectionOutput)
            
            return {
                "reflection": reflection_output.reflection_text,
                "confidence": reflection_output.confidence,
                "experience_ids": reflection_output.experience_ids,
                "insight_level": reflection_output.insight_level
            }
    
    async def construct_narrative(self,
                               experiences: List[Dict[str, Any]],
                               topic: str,
                               chronological: bool = True) -> Dict[str, Any]:
        """
        Construct a coherent narrative from multiple experiences
        
        Args:
            experiences: List of experiences to include in narrative
            topic: Topic of the narrative
            chronological: Whether to maintain chronological order
            
        Returns:
            Narrative data
        """
        # Use the Narrative Agent for this task
        with trace(workflow_name="NarrativeConstruction"):
            context_data = ExperienceContextData(
                query=topic,
                emotional_state=self.emotional_core.get_formatted_emotional_state(),
                timestamp=datetime.now().isoformat()
            )
            
            # Create input with experiences and options
            agent_input = {
                "role": "user",
                "content": f"Construct a narrative about {topic} from these experiences",
                "experiences": experiences,
                "chronological": chronological
            }
            
            result = await Runner.run(
                self.narrative_agent,
                agent_input,
                context=context_data
            )
            
            # Get the narrative output
            narrative_output = result.final_output_as(NarrativeOutput)
            
            return {
                "narrative": narrative_output.narrative_text,
                "confidence": narrative_output.coherence_score,
                "experience_count": narrative_output.experiences_included,
                "chronological": narrative_output.chronological
            }
