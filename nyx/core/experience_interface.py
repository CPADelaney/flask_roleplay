# nyx/core/experience_interface.py

import logging
import asyncio
from datetime import datetime
import random
import math
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from pydantic import BaseModel, Field

# Update imports to use OpenAI Agents SDK
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool, RunConfig,
    RunHooks, ModelSettings, set_default_openai_key
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

class ExperienceVector(BaseModel):
    """Vector representation of an experience for semantic search"""
    experience_id: str = Field(..., description="ID of the experience")
    vector: List[float] = Field(..., description="Vector embedding of the experience")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExperienceIdentityImpact(BaseModel):
    """Impact of an experience on identity formation"""
    experience_id: str = Field(..., description="ID of the experience")
    preference_updates: Dict[str, float] = Field(default_factory=dict, description="Preference strength updates")
    trait_updates: Dict[str, float] = Field(default_factory=dict, description="Personality trait updates")
    significance: float = Field(default=0.5, description="Overall significance to identity")

class UserPreferenceProfile(BaseModel):
    """User preference profile for personalized experience sharing"""
    user_id: str = Field(..., description="ID of the user")
    scenario_preferences: Dict[str, float] = Field(default_factory=dict, description="Preferences for scenario types")
    emotional_preferences: Dict[str, float] = Field(default_factory=dict, description="Preferences for emotional tones")
    experience_sharing_preference: float = Field(default=0.5, description="Preference for experience sharing (0-1)")

class VectorSearchParams(BaseModel):
    """Parameters for vector search"""
    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(default=5, description="Number of top results to return")
    user_id: Optional[str] = Field(default=None, description="Optional user ID to filter experiences")
    include_cross_user: bool = Field(default=False, description="Whether to include experiences from other users")

class ExperienceContext:
    """Context object for sharing state between agents and tools"""
    def __init__(self):
        self.current_query = ""
        self.current_scenario_type = ""
        self.emotional_state = {}
        self.current_user_id = None
        self.search_results = []
        self.last_reflection = None
        self.cycle_count = 0

class ExperienceRunHooks(RunHooks):
    """Hooks for monitoring experience agent execution"""
    
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent):
        """Called when an agent starts running"""
        logger.info(f"Starting agent: {agent.name}")
        
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any):
        """Called when an agent finishes running"""
        logger.info(f"Agent {agent.name} completed with output type: {type(output)}")
        
    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent):
        """Called when a handoff occurs between agents"""
        logger.info(f"Handoff from {from_agent.name} to {to_agent.name}")
        
    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Any):
        """Called before a tool is invoked"""
        logger.info(f"Starting tool: {tool.name} from agent: {agent.name}")
        
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Any, result: str):
        """Called after a tool is invoked"""
        logger.info(f"Tool {tool.name} completed with result length: {len(str(result))}")

class ExperienceInterface:
    """
    Agent-based interface for managing, retrieving, and formatting experience-related memories.
    Provides natural language experience sharing and recall functionality using OpenAI Agents SDK.
    
    Enhanced with vector search, cross-user experience sharing, identity evolution,
    improved adaptation integration, and memory consolidation mechanisms.
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
        
        # Create context object for sharing state between agents
        self.context = ExperienceContext()
        
        # Create run hooks for monitoring
        self.run_hooks = ExperienceRunHooks()
        
        # Set up agents with proper handoffs
        self.reflection_agent = self._create_reflection_agent()
        self.narrative_agent = self._create_narrative_agent()
        self.recall_agent = self._create_recall_agent()
        self.cross_user_agent = self._create_cross_user_agent()
        self.identity_evolution_agent = self._create_identity_evolution_agent()
        
        # Main agent with handoffs to specialists
        self.experience_agent = self._create_experience_agent()
        
        # Caching
        self.experience_cache = {}
        self.vector_cache = {}
        self.last_retrieval_time = datetime.now()
        
        # Vector database for experiences
        self.experience_vectors = {}
        self.user_experience_map = {}  # Maps user_ids to sets of experience ids
        self.vector_dimension = 384  # Default vector dimension for embeddings
        
        # Identity evolution tracking (same as before)
        self.identity_profile = {
            "preferences": {
                "scenario_types": {
                    "teasing": 0.6,
                    "dark": 0.4,
                    "indulgent": 0.7,
                    "psychological": 0.8,
                    "nurturing": 0.3,
                    "discipline": 0.5,
                    "training": 0.6,
                    "service": 0.4,
                    "worship": 0.5
                },
                "emotional_tones": {
                    "dominant": 0.8,
                    "playful": 0.7,
                    "stern": 0.6,
                    "nurturing": 0.4,
                    "cruel": 0.5,
                    "sadistic": 0.6,
                    "teasing": 0.7
                },
                "interaction_styles": {
                    "direct": 0.7,
                    "suggestive": 0.8,
                    "metaphorical": 0.6,
                    "explicit": 0.5,
                    "subtle": 0.4
                }
            },
            "traits": {
                "dominance": 0.8,
                "playfulness": 0.6,
                "strictness": 0.5,
                "creativity": 0.7,
                "intensity": 0.6,
                "patience": 0.4,
                "cruelty": 0.5
            },
            "evolution_history": []
        }
        
        # User preference profiles
        self.user_preference_profiles = {}
        
        # Trace ID for connecting traces
        self.trace_group_id = f"experience_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Template patterns
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
            
            # Other template categories remain the same...
        }
        
        # Confidence mapping
        self.confidence_markers = {
            (0.8, 1.0): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
        
        logger.info("Experience Interface initialized")
    
    def _create_experience_agent(self) -> Agent:
        """Create the main experience agent with formal handoffs to specialists"""
        return Agent(
            name="Experience Agent",
            instructions="""
            You are the Experience Agent, responsible for managing and retrieving experiences.
            Your role is to determine the best way to handle experience-related requests and
            coordinate with specialized agents when appropriate.
            """,
            handoffs=[
                handoff(
                    self.reflection_agent,
                    tool_name_override="generate_reflection",
                    tool_description_override="Generate a reflection from experiences"
                ),
                handoff(
                    self.narrative_agent,
                    tool_name_override="construct_narrative",
                    tool_description_override="Construct a narrative from experiences"
                ),
                handoff(
                    self.recall_agent,
                    tool_name_override="recall_experience",
                    tool_description_override="Format an experience for conversational recall"
                ),
                handoff(
                    self.cross_user_agent,
                    tool_name_override="find_cross_user_experience",
                    tool_description_override="Find relevant experiences from other users"
                ),
                handoff(
                    self.identity_evolution_agent,
                    tool_name_override="update_identity",
                    tool_description_override="Update identity based on experiences"
                )
            ],
            tools=[
                function_tool(self._retrieve_experiences),
                function_tool(self._get_emotional_context),
                function_tool(self._store_experience),
                function_tool(self._vector_search_experiences)
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._experience_request_guardrail)
            ],
            model_settings=ModelSettings(
                temperature=0.7
            )
        )
    
    def _create_reflection_agent(self) -> Agent:
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
            output_type=ReflectionOutput,
            model_settings=ModelSettings(
                temperature=0.7
            )
        )
    
    def _create_narrative_agent(self) -> Agent:
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
            output_type=NarrativeOutput,
            model_settings=ModelSettings(
                temperature=0.8
            )
        )
    
    def _create_recall_agent(self) -> Agent:
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
            output_type=ExperienceOutput,
            model_settings=ModelSettings(
                temperature=0.7
            )
        )
        
    def _create_cross_user_agent(self) -> Agent:
        """Create an agent specialized in cross-user experience sharing"""
        return Agent(
            name="Cross-User Experience Agent",
            handoff_description="Specialist agent for sharing experiences across users",
            instructions="""
            You are the Cross-User Experience Agent, specialized in finding and sharing 
            relevant experiences across different users. Consider relevance, privacy,
            and appropriateness when recommending cross-user experiences.
            """,
            tools=[
                function_tool(self._get_cross_user_experiences),
                function_tool(self._filter_sharable_experiences),
                function_tool(self._personalize_cross_user_experience)
            ],
            output_type=ExperienceOutput,
            model_settings=ModelSettings(
                temperature=0.6
            )
        )
    
    def _create_identity_evolution_agent(self) -> Agent:
        """Create an agent for managing identity evolution based on experiences"""
        return Agent(
            name="Identity Evolution Agent",
            handoff_description="Specialist agent for evolving identity based on experiences",
            instructions="""
            You are the Identity Evolution Agent, specialized in updating and evolving
            Nyx's preferences, traits, and overall identity based on accumulated experiences.
            Focus on creating a coherent, continuous identity that learns and grows.
            """,
            tools=[
                function_tool(self._update_identity_from_experience),
                function_tool(self._get_identity_profile),
                function_tool(self._generate_identity_reflection)
            ],
            model_settings=ModelSettings(
                temperature=0.5
            )
        )
    
    # Guardrail functions
    
    async def _experience_request_guardrail(self, ctx, agent, input_data):
        """Guardrail to validate experience-related requests"""
        # Check for minimum context
        if isinstance(input_data, str) and len(input_data.strip()) < 3:
            return GuardrailFunctionOutput(
                output_info={"error": "Request too short"},
                tripwire_triggered=True
            )
        
        # Check for appropriate content
        if isinstance(input_data, str) and any(term in input_data.lower() for term in ["delete", "remove", "erase"]):
            return GuardrailFunctionOutput(
                output_info={"error": "Cannot delete or remove experiences directly"},
                tripwire_triggered=True
            )
        
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )
    
    # Vector search functions with Pydantic model
    
    @function_tool
    async def _generate_experience_vector(self, ctx: RunContextWrapper,
                                      experience_text: str) -> List[float]:
        """
        Generate a vector embedding for an experience text
        
        Args:
            experience_text: Text of the experience to embed
            
        Returns:
            Vector embedding of the experience
        """
        # In a real implementation, this would use a proper embedding model
        # For this example, we'll use a simple mock embedding
        
        # Mock embedding generation - in real code, this would call an embedding API
        # Use hash of text to generate a deterministic but unique vector
        # This is just for demonstration - real code would use a proper embedding model
        text_hash = hash(experience_text)
        random.seed(text_hash)
        
        # Generate a random vector of the specified dimension
        vector = [random.uniform(-1.0, 1.0) for _ in range(self.vector_dimension)]
        
        # Normalize the vector to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    @function_tool
    async def _vector_search_experiences(self, ctx: RunContextWrapper,
                                     params: VectorSearchParams) -> List[Dict[str, Any]]:
        """
        Perform vector search for experiences similar to the query
        
        Args:
            params: Parameters for vector search
            
        Returns:
            List of experiences with similarity scores
        """
        # Extract parameters
        query = params.query
        top_k = params.top_k
        user_id = params.user_id
        include_cross_user = params.include_cross_user
        
        # Generate vector for the query
        query_vector = await self._generate_experience_vector(ctx, query)
        
        # Get all experience vectors
        results = []
        
        # Determine which experience vectors to search
        search_vectors = {}
        
        if not include_cross_user and user_id:
            # Only include experiences from this user
            if user_id in self.user_experience_map:
                for exp_id in self.user_experience_map[user_id]:
                    if exp_id in self.experience_vectors:
                        search_vectors[exp_id] = self.experience_vectors[exp_id]
        else:
            # Include all experiences (or filter by permitted cross-user sharing)
            if include_cross_user:
                # Get sharable experiences from all users
                sharable_experiences = await self._filter_sharable_experiences(
                    ctx, 
                    user_id=user_id if user_id else "default"
                )
                
                for exp_id in sharable_experiences:
                    if exp_id in self.experience_vectors:
                        search_vectors[exp_id] = self.experience_vectors[exp_id]
            else:
                # Use all experience vectors
                search_vectors = self.experience_vectors
        
        # Calculate similarities
        for exp_id, exp_vector_data in search_vectors.items():
            exp_vector = exp_vector_data.get("vector", [])
            
            # Calculate cosine similarity
            similarity = self._calculate_cosine_similarity(query_vector, exp_vector)
            
            # Add to results if similarity is good enough
            if similarity > 0.5:  # Threshold for minimum similarity
                results.append({
                    "experience_id": exp_id,
                    "similarity": similarity,
                    "metadata": exp_vector_data.get("metadata", {})
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get top k results
        top_results = results[:top_k]
        
        # Fetch the actual experiences
        experiences = []
        for result in top_results:
            exp_id = result["experience_id"]
            try:
                # Fetch the experience from memory core
                memory = await self.memory_core.get_memory_by_id(exp_id)
                if memory:
                    # Add similarity score to memory
                    memory["similarity"] = result["similarity"]
                    experiences.append(memory)
            except Exception as e:
                logger.error(f"Error fetching experience {exp_id}: {e}")
        
        return experiences
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Ensure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            return dot_product / (mag1 * mag2)
        else:
            return 0.0
    
    # Cross-user experience functions
    
    @function_tool
    async def _get_cross_user_experiences(self, ctx: RunContextWrapper,
                                      query: str,
                                      user_id: str,
                                      limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get experiences from other users that are relevant to the query
        
        Args:
            query: Search query
            user_id: Current user ID
            limit: Maximum number of experiences to return
            
        Returns:
            List of relevant cross-user experiences
        """
        # First, get experiences that are sharable
        sharable_experiences = await self._filter_sharable_experiences(ctx, user_id=user_id)
        
        # Perform vector search with these experiences
        search_params = VectorSearchParams(
            query=query,
            top_k=limit * 2,  # Get more to filter
            include_cross_user=True
        )
        
        experiences = await self._vector_search_experiences(ctx, search_params)
        
        # Filter to only include sharable experiences
        experiences = [exp for exp in experiences if exp.get("id") in sharable_experiences]
        
        # Sort by relevance and limit
        experiences.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        experiences = experiences[:limit]
        
        return experiences
    
    @function_tool
    async def _filter_sharable_experiences(self, ctx: RunContextWrapper,
                                      user_id: str) -> List[str]:
        """
        Filter experiences that can be shared across users
        
        Args:
            user_id: Current user ID
            
        Returns:
            List of experience IDs that can be shared
        """
        # Get all experiences
        sharable_ids = []
        
        # In a real implementation, this would check privacy settings, user preferences, etc.
        # For this example, we'll use a simple rule: experiences with significance > 5 are sharable
        
        # Check all user experience maps except the current user
        for other_user_id, exp_ids in self.user_experience_map.items():
            if other_user_id != user_id:
                for exp_id in exp_ids:
                    try:
                        # Get the experience from memory core
                        memory = await self.memory_core.get_memory_by_id(exp_id)
                        
                        if memory:
                            # Check if it's sharable
                            significance = memory.get("significance", 0)
                            tags = memory.get("tags", [])
                            metadata = memory.get("metadata", {})
                            
                            # Check criteria for sharability
                            is_sharable = (
                                significance > 5 and
                                "private" not in tags and
                                metadata.get("sharable", True)
                            )
                            
                            if is_sharable:
                                sharable_ids.append(exp_id)
                    except Exception as e:
                        logger.error(f"Error checking experience {exp_id}: {e}")
        
        return sharable_ids
    
    @function_tool
    async def _personalize_cross_user_experience(self, ctx: RunContextWrapper,
                                           experience: Dict[str, Any],
                                           user_id: str) -> Dict[str, Any]:
        """
        Personalize a cross-user experience for the current user
        
        Args:
            experience: The experience to personalize
            user_id: Current user ID
            
        Returns:
            Personalized experience
        """
        # Get user preference profile
        profile = self._get_user_preference_profile(user_id)
        
        # Make a copy of the experience to modify
        personalized = experience.copy()
        
        # Remove user-specific information
        if "metadata" in personalized:
            metadata = personalized["metadata"].copy()
            # Remove original user info
            if "user_id" in metadata:
                metadata["original_user_id"] = metadata.pop("user_id")
            
            # Add sharing metadata
            metadata["cross_user_shared"] = True
            metadata["shared_with"] = user_id
            metadata["shared_timestamp"] = datetime.now().isoformat()
            
            personalized["metadata"] = metadata
        
        # Adjust emotional context based on user preferences if needed
        emotional_context = personalized.get("metadata", {}).get("emotional_context", {})
        
        if emotional_context:
            # Check user preferences for emotional tones
            primary_emotion = emotional_context.get("primary_emotion", "")
            
            if primary_emotion and profile and "emotional_preferences" in profile:
                # Check if user has a preference for this emotion
                emotion_pref = profile["emotional_preferences"].get(primary_emotion.lower(), 0.5)
                
                # If preference is low, adjust intensity down slightly to match preferences
                if emotion_pref < 0.4 and "primary_intensity" in emotional_context:
                    emotional_context["primary_intensity"] *= 0.8
                
                # If preference is high, adjust intensity up slightly
                elif emotion_pref > 0.7 and "primary_intensity" in emotional_context:
                    emotional_context["primary_intensity"] = min(1.0, emotional_context["primary_intensity"] * 1.2)
        
        return personalized
    
    def _get_user_preference_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get the preference profile for a user, creating if it doesn't exist
        
        Args:
            user_id: User ID
            
        Returns:
            User preference profile
        """
        if user_id not in self.user_preference_profiles:
            # Create default profile
            self.user_preference_profiles[user_id] = {
                "user_id": user_id,
                "scenario_preferences": {
                    "teasing": 0.5,
                    "dark": 0.5,
                    "indulgent": 0.5,
                    "psychological": 0.5,
                    "nurturing": 0.5,
                    "discipline": 0.5
                },
                "emotional_preferences": {
                    "joy": 0.5,
                    "sadness": 0.5,
                    "anger": 0.5,
                    "fear": 0.5,
                    "trust": 0.5,
                    "disgust": 0.5,
                    "anticipation": 0.5,
                    "surprise": 0.5
                },
                "experience_sharing_preference": 0.5
            }
        
        return self.user_preference_profiles[user_id]
    
    # Identity evolution functions
    
    @function_tool
    async def _update_identity_from_experience(self, ctx: RunContextWrapper,
                                         experience: Dict[str, Any],
                                         impact: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Update Nyx's identity based on an experience
        
        Args:
            experience: The experience impacting identity
            impact: Dictionary of impacts on preferences, traits, etc.
            
        Returns:
            Updated identity profile segments
        """
        # Track what was updated
        updates = {
            "preferences": {},
            "traits": {}
        }
        
        # Update preferences
        if "preferences" in impact:
            for pref_type, pref_values in impact["preferences"].items():
                if pref_type in self.identity_profile["preferences"]:
                    for pref_name, pref_impact in pref_values.items():
                        if pref_name in self.identity_profile["preferences"][pref_type]:
                            # Update the preference with a weighted impact
                            old_value = self.identity_profile["preferences"][pref_type][pref_name]
                            
                            # Calculate significance factor (higher significance = stronger impact)
                            significance = experience.get("significance", 5) / 10
                            
                            # Calculate learning rate (how quickly preferences change)
                            learning_rate = 0.1 * significance
                            
                            # Update preference
                            new_value = old_value + (pref_impact * learning_rate)
                            
                            # Clamp to valid range
                            new_value = max(0.0, min(1.0, new_value))
                            
                            # Set new value
                            self.identity_profile["preferences"][pref_type][pref_name] = new_value
                            
                            # Record update
                            updates["preferences"][f"{pref_type}.{pref_name}"] = {
                                "old": old_value,
                                "new": new_value,
                                "change": new_value - old_value
                            }
        
        # Update traits
        if "traits" in impact:
            for trait_name, trait_impact in impact["traits"].items():
                if trait_name in self.identity_profile["traits"]:
                    # Update the trait with a weighted impact
                    old_value = self.identity_profile["traits"][trait_name]
                    
                    # Calculate significance factor
                    significance = experience.get("significance", 5) / 10
                    
                    # Calculate learning rate (traits change more slowly than preferences)
                    learning_rate = 0.05 * significance
                    
                    # Update trait
                    new_value = old_value + (trait_impact * learning_rate)
                    
                    # Clamp to valid range
                    new_value = max(0.0, min(1.0, new_value))
                    
                    # Set new value
                    self.identity_profile["traits"][trait_name] = new_value
                    
                    # Record update
                    updates["traits"][trait_name] = {
                        "old": old_value,
                        "new": new_value,
                        "change": new_value - old_value
                    }
        
        # Record the evolution
        self.identity_profile["evolution_history"].append({
            "timestamp": datetime.now().isoformat(),
            "experience_id": experience.get("id", "unknown"),
            "updates": updates
        })
        
        # Limit history size
        if len(self.identity_profile["evolution_history"]) > 100:
            self.identity_profile["evolution_history"] = self.identity_profile["evolution_history"][-100:]
        
        return updates
    
    @function_tool
    async def _get_identity_profile(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get Nyx's current identity profile
        
        Returns:
            Current identity profile
        """
        return self.identity_profile
    
    @function_tool
    async def _generate_identity_reflection(self, ctx: RunContextWrapper) -> str:
        """
        Generate a reflection on Nyx's identity evolution
        
        Returns:
            Reflection on identity
        """
        # Get the most significant changes in identity
        significant_changes = []
        
        # Look at recent history entries
        recent_history = self.identity_profile["evolution_history"][-10:]
        
        # Track cumulative changes
        cumulative_changes = {}
        
        for entry in recent_history:
            updates = entry.get("updates", {})
            
            # Check preferences
            for pref_path, pref_data in updates.get("preferences", {}).items():
                change = pref_data.get("change", 0)
                
                if pref_path not in cumulative_changes:
                    cumulative_changes[pref_path] = 0
                cumulative_changes[pref_path] += change
            
            # Check traits
            for trait_name, trait_data in updates.get("traits", {}).items():
                change = trait_data.get("change", 0)
                
                trait_path = f"traits.{trait_name}"
                if trait_path not in cumulative_changes:
                    cumulative_changes[trait_path] = 0
                cumulative_changes[trait_path] += change
        
        # Get the most significant cumulative changes
        for path, change in cumulative_changes.items():
            if abs(change) >= 0.1:  # Threshold for significance
                parts = path.split(".")
                category = parts[0]
                name = ".".join(parts[1:])
                
                significant_changes.append({
                    "category": category,
                    "name": name,
                    "change": change
                })
        
        # Sort by absolute change magnitude
        significant_changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        # Generate reflection text
        if not significant_changes:
            return "My personality and preferences have remained relatively stable recently. I continue to enjoy my established patterns of interaction."
        
        # Build reflection from significant changes
        reflection_parts = ["I've noticed some changes in how I approach situations and interact with others."]
        
        for change in significant_changes[:3]:  # Focus on top 3 changes
            category = change["category"]
            name = change["name"]
            change_value = change["change"]
            
            if category == "preferences":
                if "." in name:
                    pref_type, pref_name = name.split(".")
                    if change_value > 0:
                        reflection_parts.append(f"I've been increasingly drawn to {pref_name} {pref_type}.")
                    else:
                        reflection_parts.append(f"I've been less interested in {pref_name} {pref_type}.")
                else:
                    if change_value > 0:
                        reflection_parts.append(f"I've developed a stronger preference for {name}.")
                    else:
                        reflection_parts.append(f"I've been moving away from my preference for {name}.")
            
            elif category == "traits":
                if change_value > 0:
                    reflection_parts.append(f"I find myself becoming more {name} in my interactions.")
                else:
                    reflection_parts.append(f"I've been expressing less {name} than I used to.")
        
        # Add conclusion
        if len(significant_changes) > 3:
            reflection_parts.append("There are other subtle changes in my personality and preferences, but these stand out the most.")
        
        return " ".join(reflection_parts)
    
    # Other tool functions
    
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
        # Update context
        ctx.context.current_query = query
        ctx.context.current_scenario_type = scenario_type or ""
        
        # Create context dict
        context = {
            "query": query,
            "scenario_type": scenario_type or "",
            "emotional_state": self.emotional_core.get_formatted_emotional_state() if self.emotional_core else {},
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
        
        # Try vector search first for better semantic matching
        try:
            search_params = VectorSearchParams(
                query=query,
                top_k=limit * 2,  # Get more to filter
                include_cross_user=True  # Allow cross-user experiences
            )
            
            vector_results = await self._vector_search_experiences(ctx, search_params)
            
            # If we got good vector results, use them
            if vector_results and len(vector_results) >= limit:
                # Score memories for relevance and experiential richness
                scored_memories = []
                for memory in vector_results:
                    # Get emotional context for this memory
                    emotional_context = await self._get_memory_emotional_context(memory)
                    
                    # Calculate experiential richness
                    experiential_richness = self._calculate_experiential_richness(
                        memory, emotional_context
                    )
                    
                    # Use vector similarity as relevance score if available
                    relevance_score = memory.get("similarity", memory.get("relevance", 0.5))
                    
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
                
                # Update context with search results
                ctx.context.search_results = experiences
                
                return experiences
        except Exception as e:
            logger.error(f"Vector search failed: {e}, falling back to traditional retrieval")
            # Fall back to traditional retrieval
        
        # Get base memories from memory system (traditional method, fallback)
        base_memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "episodic", "experience"],
            limit=limit*3,  # Get more to filter later
            context=context
        )
        
        # Rest of the function remains the same...
        # Score memories, convert to experiences, etc.
        
        # Add to context
        ctx.context.search_results = experiences
        
        return experiences
    
    @function_tool
    async def _get_emotional_context(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """Get current emotional context"""
        emotional_state = self.emotional_core.get_formatted_emotional_state() if self.emotional_core else {}
        ctx.context.emotional_state = emotional_state
        return emotional_state
    
    @function_tool
    async def _store_experience(
        self,
        ctx: RunContextWrapper,
        memory_text: str,
        scenario_type: str = "general",
        entities: Optional[List[str]] = None,
        emotional_context: Optional[Dict[str, Any]] = None,
        significance: int = 5,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a new experience in the memory system
    
        Args:
            memory_text: The memory text
            scenario_type: Type of scenario
            entities: List of entity IDs involved
            emotional_context: Emotional context data
            significance: Memory significance
            tags: Additional tags
            user_id: User ID for cross-user functionality
    
        Returns:
            Stored experience information
        """
        tags = tags or []
    
        if scenario_type not in tags:
            tags.append(scenario_type)
        if "experience" not in tags:
            tags.append("experience")
    
        metadata = {
            "scenario_type": scenario_type,
            "entities": entities or [],
            "is_experience": True
        }
        if user_id:
            metadata["user_id"] = user_id
    
        if emotional_context:
            metadata["emotional_context"] = emotional_context
        elif self.emotional_core:
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
    
        # Store in user experience map
        if user_id:
            if not hasattr(self, "user_experience_map"):
                self.user_experience_map = dict()
            if user_id not in self.user_experience_map:
                self.user_experience_map[user_id] = set()
            self.user_experience_map[user_id].add(memory_id)
    
        # Generate and store vector embedding
        try:
            vector = await self._generate_experience_vector(ctx, memory_text)
            if not hasattr(self, "experience_vectors"):
                self.experience_vectors = dict()
            self.experience_vectors[memory_id] = {
                "experience_id": memory_id,
                "vector": vector,
                "metadata": {
                    "user_id": user_id or "default",
                    "scenario_type": scenario_type,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error generating vector for experience {memory_id}: {e}")
    
        # Calculate potential identity impact
        try:
            # Determine impact on identity
            identity_impact = self._calculate_identity_impact(
                memory_text, scenario_type, emotional_context
            )
            # Update identity if impact is significant
            if identity_impact:
                await self._update_identity_from_experience(
                    ctx,
                    {"id": memory_id, "memory_text": memory_text, "significance": significance},
                    identity_impact
                )
        except Exception as e:
            logger.error(f"Error updating identity from experience: {e}")
    
        return {
            "memory_id": memory_id,
            "memory_text": memory_text,
            "scenario_type": scenario_type,
            "tags": tags,
            "significance": significance,
            "user_id": user_id
        }
        
    def _calculate_identity_impact(self, 
                                 memory_text: str, 
                                 scenario_type: str, 
                                 emotional_context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate the impact of an experience on identity
        
        Args:
            memory_text: Experience text
            scenario_type: Type of scenario
            emotional_context: Emotional context
            
        Returns:
            Identity impact dictionary
        """
        impact = {
            "preferences": {},
            "traits": {}
        }
        
        # Impact on scenario preferences
        if scenario_type in self.identity_profile["preferences"]["scenario_types"]:
            # Positive emotional valence increases preference
            valence = emotional_context.get("valence", 0)
            
            if valence > 0.3:
                # Positive experience, strengthen preference
                impact["preferences"]["scenario_types"] = {scenario_type: 0.1}
            elif valence < -0.3:
                # Negative experience, weaken preference
                impact["preferences"]["scenario_types"] = {scenario_type: -0.05}
        
        # Impact on emotional tone preferences
        primary_emotion = emotional_context.get("primary_emotion", "")
        if primary_emotion:
            primary_emotion_lower = primary_emotion.lower()
            
            # Map emotions to tone preferences
            emotion_to_tone = {
                "joy": "playful",
                "sadness": "nurturing",
                "anger": "stern",
                "fear": "cruel",
                "disgust": "cruel",
                "anticipation": "teasing",
                "surprise": "playful",
                "trust": "nurturing",
                "love": "playful",
                "frustration": "stern"
            }
            
            if primary_emotion_lower in emotion_to_tone:
                tone = emotion_to_tone[primary_emotion_lower]
                if tone in self.identity_profile["preferences"]["emotional_tones"]:
                    impact["preferences"]["emotional_tones"] = {tone: 0.08}
        
        # Impact on traits based on scenario and emotional context
        trait_impacts = {}
        
        if scenario_type == "teasing":
            trait_impacts["playfulness"] = 0.1
            trait_impacts["creativity"] = 0.05
            
        elif scenario_type == "discipline":
            trait_impacts["strictness"] = 0.1
            trait_impacts["dominance"] = 0.08
            
        elif scenario_type == "dark":
            trait_impacts["intensity"] = 0.1
            trait_impacts["cruelty"] = 0.08
            
        elif scenario_type == "indulgent":
            trait_impacts["patience"] = 0.1
            trait_impacts["creativity"] = 0.08
            
        elif scenario_type == "psychological":
            trait_impacts["creativity"] = 0.1
            trait_impacts["intensity"] = 0.05
            
        elif scenario_type == "nurturing":
            trait_impacts["patience"] = 0.1
            trait_impacts["strictness"] = -0.05
        
        # Add trait impacts if they exist
        if trait_impacts:
            impact["traits"] = trait_impacts
        
        return impact
    
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
                if tag.lower() in ["joy", "sadness", "anger", "fear", "disgust", 
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
            "significance": memory.get("significance", 3),
            "user_id": metadata.get("user_id", "default")
        }
        
        # Add confidence marker based on relevance
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance_score < max_val:
                experience["confidence_marker"] = marker
                break
                
        if "confidence_marker" not in experience:
            experience["confidence_marker"] = "remember"  # Default
        
        return experience
    
    # Public API methods with enhanced SDK integration
    
    async def retrieve_experiences_enhanced(self, 
                                          query: str,
                                          scenario_type: Optional[str] = None,
                                          limit: int = 3,
                                          user_id: Optional[str] = None,
                                          include_cross_user: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval of experiences based on query and scenario type
        
        Args:
            query: Search query
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of experiences to return
            user_id: Optional user ID to filter experiences
            include_cross_user: Whether to include cross-user experiences
            
        Returns:
            List of relevant experiences with metadata
        """
        # Create context and run configuration
        experience_context = ExperienceContextData(
            query=query,
            scenario_type=scenario_type,
            emotional_state=self.emotional_core.get_formatted_emotional_state() if self.emotional_core else {},
            entities=[],
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        
        run_config = RunConfig(
            workflow_name="ExperienceRetrieval",
            trace_metadata={
                "query": query,
                "scenario_type": scenario_type or "any",
                "user_id": user_id or "any",
                "cross_user": include_cross_user
            }
        )
        
        # Use the agent system with proper tracing
        with trace(workflow_name="ExperienceRetrieval", group_id=self.trace_group_id):
            # Create agent input
            agent_input = {
                "action": "retrieve_experiences",
                "query": query,
                "scenario_type": scenario_type,
                "limit": limit,
                "user_id": user_id,
                "include_cross_user": include_cross_user
            }
            
            # Run the experience agent
            result = await Runner.run(
                self.experience_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Check if we have search results
            if hasattr(result, "final_output") and isinstance(result.final_output, list):
                return result.final_output
            
            # Fallback to direct retrieval if agent didn't return proper results
            search_params = VectorSearchParams(
                query=query,
                top_k=limit,
                user_id=user_id,
                include_cross_user=include_cross_user
            )
            
            return await self._vector_search_experiences(
                RunContextWrapper(context=self.context),
                search_params
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
        # Create context and run configuration
        experience_context = ExperienceContextData(
            query="recall experience",
            scenario_type=experience.get("scenario_type", "general"),
            emotional_state=self.emotional_core.get_formatted_emotional_state() if self.emotional_core else {},
            entities=experience.get("entities", []),
            timestamp=datetime.now().isoformat()
        )
        
        run_config = RunConfig(
            workflow_name="ConversationalRecall",
            trace_metadata={
                "experience_id": experience.get("id", "unknown"),
                "scenario_type": experience.get("scenario_type", "general"),
                "user_id": experience.get("user_id", "unknown")
            }
        )
        
        # Use the Recall Agent for this task with proper tracing
        with trace(workflow_name="ConversationalRecall", group_id=self.trace_group_id):
            # Create the agent input
            agent_input = {
                "action": "recall_experience",
                "experience": experience,
                "context": context or {}
            }
            
            # Run the recall agent directly via handoff
            result = await Runner.run(
                self.recall_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Get the formatted experience output
            recall_output = result.final_output
            
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
        user_id = context_data.get("user_id")
        
        # Create context and run configuration
        experience_context = ExperienceContextData(
            query=user_query,
            scenario_type=context_data.get("scenario_type", ""),
            emotional_state=self.emotional_core.get_formatted_emotional_state() if self.emotional_core else {},
            entities=context_data.get("entities", []),
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        
        run_config = RunConfig(
            workflow_name="ExperienceSharing",
            trace_metadata={
                "query": user_query,
                "user_id": user_id or "unknown"
            }
        )
        
        # Use the agent system with proper tracing
        with trace(workflow_name="ExperienceSharing", group_id=self.trace_group_id):
            # Create agent input
            agent_input = {
                "action": "share_experience",
                "query": user_query,
                "context": context_data
            }
            
            # Run the experience agent
            result = await Runner.run(
                self.experience_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Process the result
            if hasattr(result, "final_output"):
                # Check if we have a structured output
                if isinstance(result.final_output, ExperienceOutput):
                    return {
                        "has_experience": True,
                        "response_text": result.final_output.experience_text,
                        "confidence": result.final_output.confidence,
                        "experience": {"id": result.final_output.source_id} if result.final_output.source_id else None,
                        "relevance_score": result.final_output.relevance_score
                    }
                # Otherwise use the text output
                elif isinstance(result.final_output, str):
                    return {
                        "has_experience": True,
                        "response_text": result.final_output,
                        "confidence": 0.5,  # Default confidence
                        "experience": None
                    }
                elif isinstance(result.final_output, dict) and "experience_text" in result.final_output:
                    return {
                        "has_experience": True,
                        "response_text": result.final_output["experience_text"],
                        "confidence": result.final_output.get("confidence", 0.5),
                        "experience": {"id": result.final_output.get("source_id")} if "source_id" in result.final_output else None
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
        
        # Create run configuration
        run_config = RunConfig(
            workflow_name="PersonalityReflection",
            trace_metadata={
                "experience_count": len(experiences)
            }
        )
        
        # Use the Reflection Agent with proper tracing
        with trace(workflow_name="PersonalityReflection", group_id=self.trace_group_id):
            # Create input with experiences
            agent_input = {
                "action": "generate_reflection",
                "experiences": experiences,
                "context": context or {}
            }
            
            # Run the reflection agent
            result = await Runner.run(
                self.reflection_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Get the reflection output
            reflection_output = result.final_output
            
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
        # Create run configuration
        run_config = RunConfig(
            workflow_name="NarrativeConstruction",
            trace_metadata={
                "experience_count": len(experiences),
                "topic": topic,
                "chronological": chronological
            }
        )
        
        # Use the Narrative Agent with proper tracing
        with trace(workflow_name="NarrativeConstruction", group_id=self.trace_group_id):
            # Create input with experiences and options
            agent_input = {
                "action": "construct_narrative",
                "experiences": experiences,
                "topic": topic,
                "chronological": chronological
            }
            
            # Run the narrative agent
            result = await Runner.run(
                self.narrative_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Get the narrative output
            narrative_output = result.final_output
            
            return {
                "narrative": narrative_output.narrative_text,
                "confidence": narrative_output.coherence_score,
                "experience_count": narrative_output.experiences_included,
                "chronological": narrative_output.chronological
            }
    
    async def share_cross_user_experience(self,
                                      query: str,
                                      user_id: str,
                                      context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Share an experience from other users that is relevant to the query
        
        Args:
            query: User's query text
            user_id: Current user ID
            context_data: Additional context data
            
        Returns:
            Cross-user experience sharing response
        """
        context_data = context_data or {}
        
        # Create run configuration
        run_config = RunConfig(
            workflow_name="CrossUserExperienceSharing",
            trace_metadata={
                "query": query,
                "user_id": user_id
            }
        )
        
        # Use the cross-user agent with proper tracing
        with trace(workflow_name="CrossUserExperienceSharing", group_id=self.trace_group_id):
            # Create agent input
            agent_input = {
                "action": "find_cross_user_experience",
                "query": query,
                "user_id": user_id,
                "context": context_data
            }
            
            # Run the cross-user agent
            result = await Runner.run(
                self.cross_user_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Check if we have a valid experience
            if hasattr(result, "final_output") and isinstance(result.final_output, ExperienceOutput):
                return {
                    "has_experience": True,
                    "response_text": result.final_output.experience_text,
                    "confidence": result.final_output.confidence,
                    "experience": {"id": result.final_output.source_id} if result.final_output.source_id else None,
                    "cross_user": True,
                    "relevance_score": result.final_output.relevance_score
                }
            
            # No cross-user experiences found
            return {
                "has_experience": False,
                "response_text": None
            }
    
    async def get_identity_reflection(self) -> Dict[str, Any]:
        """
        Get a reflection on Nyx's identity evolution
        
        Args:
            None
            
        Returns:
            Identity reflection data
        """
        # Create run configuration
        run_config = RunConfig(
            workflow_name="IdentityReflection",
            trace_metadata={
                "evolution_entries": len(self.identity_profile["evolution_history"])
            }
        )
        
        # Use the identity evolution agent with proper tracing
        with trace(workflow_name="IdentityReflection", group_id=self.trace_group_id):
            # Create agent input
            agent_input = {
                "action": "generate_identity_reflection"
            }
            
            # Run the identity evolution agent
            result = await Runner.run(
                self.identity_evolution_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Extract the reflection
            reflection_text = result.final_output
            
            # Get identity profile
            identity_profile = await self._get_identity_profile(RunContextWrapper(context=self.context))
            
            # Extract key identity elements
            top_scenario_prefs = sorted(
                identity_profile["preferences"]["scenario_types"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_traits = sorted(
                identity_profile["traits"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            return {
                "reflection": reflection_text,
                "top_preferences": dict(top_scenario_prefs),
                "top_traits": dict(top_traits),
                "evolution_entries": len(identity_profile["evolution_history"]),
                "has_evolved": len(identity_profile["evolution_history"]) > 0
            }
    
    # Enhanced integration with adaptation system
    
    async def share_experience_enhanced(self, 
                                    query: str, 
                                    context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced experience sharing with adaptation integration
        
        Args:
            query: User's query text
            context_data: Additional context data
            
        Returns:
            Experience sharing response with adaptation
        """
        context_data = context_data or {}
        user_id = context_data.get("user_id", "default")
        
        # Create run configuration
        run_config = RunConfig(
            workflow_name="EnhancedExperienceSharing",
            trace_metadata={
                "query": query,
                "user_id": user_id
            }
        )
        
        # Use the experience agent with proper tracing
        with trace(workflow_name="EnhancedExperienceSharing", group_id=self.trace_group_id):
            # Get user preference for experience sharing
            profile = self._get_user_preference_profile(user_id)
            sharing_preference = profile.get("experience_sharing_preference", 0.5)
            
            # Check if experience sharing should be used based on preference
            # Lower preference means less frequent sharing
            random_factor = random.random()
            
            if random_factor > sharing_preference and "force_experience" not in context_data:
                return {
                    "has_experience": False,
                    "response_text": None,
                    "reason": "user_preference"
                }
            
            # Create agent input
            agent_input = {
                "action": "enhance_experience_sharing",
                "query": query,
                "user_id": user_id,
                "include_cross_user": context_data.get("include_cross_user", True),
                "scenario_type": context_data.get("scenario_type", None),
                "entities": context_data.get("entities", [])
            }
            
            # Run the experience agent
            result = await Runner.run(
                self.experience_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Process the result
            if hasattr(result, "final_output") and result.final_output:
                if isinstance(result.final_output, dict) and "has_experience" in result.final_output:
                    return result.final_output
                elif isinstance(result.final_output, ExperienceOutput):
                    return {
                        "has_experience": True,
                        "response_text": result.final_output.experience_text,
                        "confidence": result.final_output.confidence,
                        "experience": {"id": result.final_output.source_id} if result.final_output.source_id else None,
                        "relevance_score": result.final_output.relevance_score
                    }
            
            # Fallback to standard experience sharing if agent didn't provide proper response
            return await self.handle_experience_sharing_request(
                user_query=query,
                context_data=context_data
            )
    
    async def share_experience_with_temporal_context(self, 
                                                  query: str,
                                                  temporal_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Share an experience with temporal context influencing selection and formatting
        
        Args:
            query: Search query
            temporal_context: Temporal perception data
            
        Returns:
            Experience sharing result with temporal context
        """
        temporal_context = temporal_context or {}
        context_data = {}
        template_type = "standard"
        
        # Create run configuration
        run_config = RunConfig(
            workflow_name="TemporalExperienceSharing",
            trace_metadata={
                "query": query,
                "time_category": temporal_context.get("time_category", "medium")
            }
        )
        
        # Use the experience agent with proper tracing
        with trace(workflow_name="TemporalExperienceSharing", group_id=self.trace_group_id):
            # Process temporal context
            time_category = temporal_context.get("time_category", "medium")
            context_data["time_category"] = time_category
            
            # Select template based on time category
            if time_category in ["long", "very_long"]:
                template_type = "long_interval"
                
            # Check for milestone
            if "milestone_reached" in temporal_context:
                template_type = "milestone"
                context_data["milestone"] = temporal_context["milestone_reached"]
                
            # Check for psychological maturity influence
            if "psychological_age" in temporal_context and temporal_context["psychological_age"] > 0.7:
                template_type = "mature_reflection"
                
            # Set template for experience formatting
            context_data["template_type"] = template_type
            
            # Create agent input
            agent_input = {
                "action": "temporal_experience_sharing",
                "query": query,
                "temporal_context": temporal_context,
                "template_type": template_type
            }
            
            # Run the experience agent
            result = await Runner.run(
                self.experience_agent,
                agent_input,
                context=self.context,
                hooks=self.run_hooks,
                run_config=run_config
            )
            
            # Process the result
            if hasattr(result, "final_output") and result.final_output:
                if isinstance(result.final_output, dict) and "has_experience" in result.final_output:
                    response = result.final_output
                    
                    # Add temporal influence context if experience was found
                    if response.get("has_experience", False):
                        response["temporal_context_applied"] = {
                            "template_type": template_type,
                            "time_category": time_category
                        }
                    
                    return response
            
            # Fallback to enhanced experience sharing
            result = await self.share_experience_enhanced(
                query=query,
                context_data=context_data
            )
            
            # Add temporal influence context if experience was found
            if result.get("has_experience", False):
                result["temporal_context_applied"] = {
                    "template_type": template_type,
                    "time_category": time_category
                }
            
            return result
    
    async def get_milestone_related_experiences(self, milestone: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get experiences relevant to a temporal milestone
        
        Args:
            milestone: Temporal milestone data
            
        Returns:
            List of relevant experiences
        """
        # Create run configuration
        run_config = RunConfig(
            workflow_name="MilestoneExperiences",
            trace_metadata={
                "milestone_name": milestone.get("name", "unknown")
            }
        )
        
        # Use the experience agent with proper tracing
        with trace(workflow_name="MilestoneExperiences", group_id=self.trace_group_id):
            milestone_name = milestone.get("name", "")
            associated_memory_ids = milestone.get("associated_memory_ids", [])
            
            experiences = []
            
            # First try to get directly associated memories
            if associated_memory_ids and self.memory_core:
                for memory_id in associated_memory_ids:
                    try:
                        memory = await self.memory_core.get_memory_by_id(memory_id)
                        if memory:
                            experience = await self._convert_memory_to_experience(
                                memory,
                                {}, # Empty emotional context to start
                                0.8, # High relevance for milestone memories
                                0.7  # Good experiential richness
                            )
                            experiences.append(experience)
                    except Exception as e:
                        logger.error(f"Error getting milestone memory {memory_id}: {e}")
            
            # If not enough directly associated memories, search for relevant ones
            if len(experiences) < 3:
                # Create search query based on milestone type
                search_query = f"milestone {milestone_name}"
                
                # Get additional experiences
                search_params = VectorSearchParams(
                    query=search_query,
                    top_k=3 - len(experiences)
                )
                
                search_results = await self._vector_search_experiences(
                    RunContextWrapper(context=self.context),
                    search_params
                )
                
                # Convert search results to experiences
                for memory in search_results:
                    # Get emotional context
                    emotional_context = await self._get_memory_emotional_context(memory)
                    
                    # Calculate experiential richness
                    experiential_richness = self._calculate_experiential_richness(
                        memory, emotional_context
                    )
                    
                    # Convert to experience
                    experience = await self._convert_memory_to_experience(
                        memory,
                        emotional_context,
                        memory.get("similarity", 0.7),
                        experiential_richness
                    )
                    
                    experiences.append(experience)
            
            return experiences
    
    # Event handling
    
    async def initialize_event_subscriptions(self, event_bus):
        """Initialize event subscriptions for the experience interface"""
        self.event_bus = event_bus
        
        # Subscribe to relevant events
        await self.event_bus.subscribe("conditioning_update", self._handle_conditioning_update)
        await self.event_bus.subscribe("conditioned_response", self._handle_conditioned_response)
        
        logger.info("Experience interface subscribed to events")
    
    async def _handle_conditioning_update(self, event):
        """Store experiences based on significant conditioning events"""
        update_type = event.data.get("update_type", "")
        association_key = event.data.get("association_key", "")
        association_type = event.data.get("association_type", "")
        strength = event.data.get("strength", 0.0)
        
        # Only create experiences for significant conditioning events
        if strength < 0.6:
            return
        
        # Create the experience
        experience_text = f"Formed a conditioning connection between {association_key} with strength {strength:.2f}"
        
        # Store the experience
        await self._store_experience(
            RunContextWrapper(context=self.context),
            memory_text=experience_text,
            scenario_type="conditioning",
            entities=[],
            emotional_context=self.emotional_core.get_formatted_emotional_state() if self.emotional_core else None,
            significance=int(strength * 7),  # Scale to 0-10
            tags=["conditioning", update_type, association_type],
            user_id=event.data.get("user_id", "default")
        )
    
    async def _handle_conditioned_response(self, event):
        """Record experiences from triggered conditioned responses"""
        stimulus = event.data.get("stimulus", "")
        responses = event.data.get("triggered_responses", [])
        user_id = event.data.get("user_id", "default")
        
        if not responses:
            return
        
        # Get strongest response
        strongest = max(responses, key=lambda r: r.get("strength", 0))
        response_text = strongest.get("response", "Unknown response")
        strength = strongest.get("strength", 0.5)
        
        # Create experience text
        experience_text = f"Responded to '{stimulus}' stimulus with: {response_text}"
        
        # Store the experience
        await self._store_experience(
            RunContextWrapper(context=self.context),
            memory_text=experience_text,
            scenario_type="conditioned_response",
            entities=[],
            emotional_context=self.emotional_core.get_formatted_emotional_state() if self.emotional_core else None,
            significance=int(strength * 6),  # Scale to 0-10
            tags=["conditioning", "response", stimulus],
            user_id=user_id
        )
    
    # Analysis methods
    
    async def analyze_conditioning_patterns(self, user_id=None, time_period="recent"):
        """Analyze conditioning patterns from stored experiences"""
        # Create run configuration
        run_config = RunConfig(
            workflow_name="ConditioningAnalysis",
            trace_metadata={
                "user_id": user_id or "all",
                "time_period": time_period
            }
        )
        
        # Use proper tracing
        with trace(workflow_name="ConditioningAnalysis", group_id=self.trace_group_id):
            # Search for conditioning-related experiences
            query = "conditioning pattern response"
            
            # Use experience retrieval
            experiences = await self.retrieve_experiences_enhanced(
                query=query,
                limit=20,
                user_id=user_id
            )
            
            if not experiences:
                return {
                    "patterns_found": 0,
                    "has_analysis": False,
                    "message": "No conditioning patterns found in experiences"
                }
            
            # Group by stimulus type
            stimulus_groups = {}
            for exp in experiences:
                tags = exp.get("tags", [])
                for tag in tags:
                    if tag not in ["conditioning", "response", "classical", "operant"]:
                        # Potential stimulus
                        if tag not in stimulus_groups:
                            stimulus_groups[tag] = []
                        stimulus_groups[tag].append(exp)
            
            # Analyze trends for each stimulus
            pattern_analysis = {}
            for stimulus, exps in stimulus_groups.items():
                if len(exps) < 2:
                    continue
                    
                # Sort by timestamp
                sorted_exps = sorted(exps, key=lambda e: e.get("timestamp", ""))
                
                # Calculate trend
                significances = [e.get("significance", 5) for e in sorted_exps]
                avg_significance = sum(significances) / len(significances)
                
                first_half = significances[:len(significances)//2]
                second_half = significances[len(significances)//2:]
                
                trend = "stable"
                if len(first_half) > 0 and len(second_half) > 0:
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    
                    if second_avg > first_avg * 1.2:
                        trend = "strengthening"
                    elif second_avg < first_avg * 0.8:
                        trend = "weakening"
                
                pattern_analysis[stimulus] = {
                    "experience_count": len(exps),
                    "average_significance": avg_significance,
                    "trend": trend,
                    "earliest": sorted_exps[0].get("timestamp", "unknown"),
                    "latest": sorted_exps[-1].get("timestamp", "unknown")
                }
            
            return {
                "patterns_found": len(pattern_analysis),
                "has_analysis": True,
                "patterns": pattern_analysis
            }
    
    # Vector search status check
    
    async def check_vector_search_status(self) -> Dict[str, Any]:
        """
        Check if vector search is available and working
        
        Returns:
            Status information
        """
        # Create run configuration
        run_config = RunConfig(
            workflow_name="VectorSearchCheck",
            trace_metadata={
                "vector_count": len(self.experience_vectors)
            }
        )
        
        # Use proper tracing
        with trace(workflow_name="VectorSearchCheck", group_id=self.trace_group_id):
            if not self.experience_vectors:
                return {
                    "status": "not_ready",
                    "reason": "No experience vectors available",
                    "vector_count": 0
                }
            
            # Try a simple search
            try:
                search_params = VectorSearchParams(
                    query="test query",
                    top_k=1
                )
                
                result = await self._vector_search_experiences(
                    RunContextWrapper(context=self.context),
                    search_params
                )
                
                return {
                    "status": "ready",
                    "vector_count": len(self.experience_vectors),
                    "test_query_results": len(result),
                    "test_success": len(result) > 0
                }
                
            except Exception as e:
                logger.error(f"Vector search test failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "vector_count": len(self.experience_vectors)
                }
