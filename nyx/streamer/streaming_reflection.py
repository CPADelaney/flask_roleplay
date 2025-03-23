# nyx/streamer/streaming_reflection.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# Import from OpenAI Agents SDK
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings
)

from nyx.core.nyx_brain import NyxBrain
from nyx.streamer.nyx_streaming_core import StreamingCore

logger = logging.getLogger("streaming_reflection")

class StreamingExperienceInput(BaseModel):
    """Input parameters for streaming experience analysis"""
    game_name: str = Field(..., description="Name of the game being streamed")
    session_duration: int = Field(..., description="Duration of the streaming session in minutes")
    memorable_moments: List[Dict[str, Any]] = Field(default_factory=list, description="Memorable moments from the stream")
    audience_reactions: Dict[str, int] = Field(default_factory=dict, description="Audience reactions and counts")
    cross_game_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Insights connecting games")

class StreamingExperienceOutput(BaseModel):
    """Output from streaming experience analysis"""
    reflection: str = Field(..., description="Reflection on the streaming experience")
    experience_summary: str = Field(..., description="Concise summary of the experience for memory storage")
    memory_importance: int = Field(..., description="Importance score for memory (1-10)", ge=1, le=10)
    tags: List[str] = Field(..., description="Tags to associate with the experience")
    emotional_response: Dict[str, float] = Field(..., description="Emotional response to the experience")
    connections: List[Dict[str, str]] = Field(..., description="Connections to other experiences or memories")

class StreamingReflectionEngine:
    """
    Engine for generating reflections on streaming experiences and
    integrating them with Nyx's memory and experience systems.
    """
    
    def __init__(self, brain: NyxBrain, streaming_core: StreamingCore):
        """
        Initialize with references to Nyx brain and streaming core
        
        Args:
            brain: NyxBrain instance
            streaming_core: StreamingCore instance
        """
        self.brain = brain
        self.streaming_core = streaming_core
        
        # Initialize specialized agents
        self.experience_agent = self._create_experience_agent()
        self.consolidation_agent = self._create_consolidation_agent()
        self.integration_agent = self._create_integration_agent()
        
        # Set up tracking for periodic reflection
        self.last_reflection_time = datetime.now()
        self.reflection_interval = timedelta(hours=1)  # Default interval
        self.reflection_history = []
        
        self.memory_importance_threshold = 6  # Minimum importance to store in long-term memory
        
        logger.info("StreamingReflectionEngine initialized")
    
    def _create_experience_agent(self) -> Agent:
        """Create an agent specialized in analyzing streaming experiences"""
        return Agent(
            name="StreamingExperienceAnalyzer",
            instructions="""
            You analyze streaming experiences to generate thoughtful reflections and identify
            important aspects to store in long-term memory. You consider:
            
            1. What aspects of the gaming experience were most significant
            2. How the audience responded to different commentary styles
            3. How this streaming experience connects to previous ones
            4. What emotions were experienced during the stream
            5. What connections exist between this game and other experiences
            
            Your reflections should be introspective and meaningful, recognizing patterns
            and generating insights that help Nyx grow as a streamer and connect these
            experiences to her broader identity.
            """,
            tools=[
                function_tool(self._retrieve_past_streaming_experiences),
                function_tool(self._analyze_audience_reactions),
                function_tool(self._identify_cross_game_patterns)
            ],
            output_type=StreamingExperienceOutput,
            model_settings=ModelSettings(
                temperature=0.7  # More creative reflections
            )
        )
    
    def _create_consolidation_agent(self) -> Agent:
        """Create an agent specialized in consolidating similar streaming experiences"""
        return Agent(
            name="StreamingExperienceConsolidator",
            instructions="""
            You identify patterns across multiple streaming experiences and consolidate them
            into higher-level abstractions. You focus on:
            
            1. Recurring patterns in streaming different games
            2. Common audience responses to different commentary styles
            3. Evolution of streaming techniques and effectiveness over time
            4. Emerging preferences and strengths in streaming approach
            5. Connections between streaming experiences and broader identity
            
            Your consolidations should identify meaningful patterns that go beyond individual
            streaming sessions, helping Nyx develop a coherent streaming identity.
            """,
            tools=[
                function_tool(self._retrieve_streaming_experiences_by_type),
                function_tool(self._find_similar_streaming_experiences),
                function_tool(self._create_streaming_abstraction)
            ]
        )
    
    def _create_integration_agent(self) -> Agent:
        """Create an agent specialized in integrating streaming with other experiences"""
        return Agent(
            name="StreamingExperienceIntegrator",
            instructions="""
            You integrate streaming experiences with other types of experiences, looking for
            connections and relationships between them. You focus on:
            
            1. How streaming experiences relate to other interaction experiences
            2. Connections between commentary skills and conversation skills
            3. How game analysis relates to other analytical thinking
            4. Emotional patterns that span across different experience types
            5. How streaming contributes to broader identity and capabilities
            
            Your integrations should help Nyx develop a unified sense of self that incorporates
            streaming experiences alongside other experiences.
            """,
            tools=[
                function_tool(self._retrieve_relevant_non_streaming_experiences),
                function_tool(self._find_cross_domain_connections),
                function_tool(self._update_identity_with_streaming_insights)
            ]
        )
    
    @function_tool
    async def _retrieve_past_streaming_experiences(self, 
                                           game_name: Optional[str] = None, 
                                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve past streaming experiences
        
        Args:
            game_name: Optional specific game to retrieve experiences for
            limit: Maximum number of experiences to retrieve
            
        Returns:
            List of relevant streaming experiences
        """
        # Create query
        if game_name:
            query = f"streaming {game_name}"
        else:
            query = "streaming"
        
        # Retrieve experiences from memory
        memories = await self.brain.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=limit,
            min_significance=4
        )
        
        return memories
    
    @function_tool
    async def _analyze_audience_reactions(self, 
                                    reactions: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze audience reactions to streaming
        
        Args:
            reactions: Dictionary of reaction types and counts
            
        Returns:
            Analysis of audience reactions
        """
        # Calculate total reactions
        total_reactions = sum(reactions.values())
        
        if total_reactions == 0:
            return {
                "analysis": "No audience reactions to analyze",
                "sentiment": "neutral",
                "engagement": 0.0
            }
        
        # Calculate percentages
        percentages = {k: (v / total_reactions) * 100 for k, v in reactions.items()}
        
        # Categorize reactions
        positive_reactions = sum(reactions.get(r, 0) for r in ["like", "love", "laugh", "wow"])
        negative_reactions = sum(reactions.get(r, 0) for r in ["angry", "sad"])
        
        # Calculate sentiment
        if total_reactions > 0:
            sentiment_score = (positive_reactions - negative_reactions) / total_reactions
        else:
            sentiment_score = 0.0
        
        # Determine sentiment category
        if sentiment_score > 0.5:
            sentiment = "very positive"
        elif sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score > -0.2:
            sentiment = "neutral"
        elif sentiment_score > -0.5:
            sentiment = "negative"
        else:
            sentiment = "very negative"
        
        # Calculate engagement level (0-1)
        engagement = min(1.0, total_reactions / 100.0)  # Normalize: 100+ reactions = full engagement
        
        return {
            "total_reactions": total_reactions,
            "percentages": percentages,
            "sentiment_score": sentiment_score,
            "sentiment": sentiment,
            "engagement": engagement,
            "analysis": f"Audience sentiment was {sentiment} with {engagement:.2f} engagement level"
        }
    
    @function_tool
    async def _identify_cross_game_patterns(self, 
                                      game_names: List[str]) -> List[Dict[str, Any]]:
        """
        Identify patterns across different games streamed
        
        Args:
            game_names: List of game names to analyze
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if not game_names or len(game_names) < 2:
            return patterns
        
        # Retrieve experiences for each game
        all_experiences = []
        
        for game_name in game_names:
            experiences = await self._retrieve_past_streaming_experiences(game_name, 3)
            if experiences:
                all_experiences.extend(experiences)
        
        if not all_experiences:
            return patterns
        
        # Use Nyx's reflection engine to find patterns if available
        if hasattr(self.brain, "reflection_engine") and self.brain.reflection_engine:
            try:
                abstraction, pattern_data = await self.brain.reflection_engine.create_abstraction(
                    memories=all_experiences,
                    pattern_type="streaming"
                )
                
                if abstraction and pattern_data:
                    patterns.append({
                        "type": "reflection_engine",
                        "pattern": abstraction,
                        "games": game_names,
                        "confidence": pattern_data.get("confidence", 0.5)
                    })
            except Exception as e:
                logger.error(f"Error using reflection engine for cross-game patterns: {e}")
        
        # Add simple patterns based on metadata analysis
        game_counts = {}
        mechanic_counts = {}
        commentary_counts = {}
        
        for experience in all_experiences:
            metadata = experience.get("metadata", {})
            
            # Track game genres
            game_name = metadata.get("game_name", "unknown")
            if game_name in game_counts:
                game_counts[game_name] += 1
            else:
                game_counts[game_name] = 1
            
            # Track mechanics mentioned
            if "event_data" in metadata and "mechanics" in metadata["event_data"]:
                for mechanic in metadata["event_data"]["mechanics"]:
                    if mechanic in mechanic_counts:
                        mechanic_counts[mechanic] += 1
                    else:
                        mechanic_counts[mechanic] = 1
            
            # Track commentary types
            if "event_data" in metadata and "focus" in metadata["event_data"]:
                focus = metadata["event_data"]["focus"]
                if focus in commentary_counts:
                    commentary_counts[focus] += 1
                else:
                    commentary_counts[focus] = 1
        
        # Find common mechanics
        common_mechanics = [m for m, c in mechanic_counts.items() if c >= 2]
        if common_mechanics:
            patterns.append({
                "type": "common_mechanics",
                "pattern": f"Frequently comments on {', '.join(common_mechanics)} across different games",
                "mechanics": common_mechanics,
                "games": game_names
            })
        
        # Find commentary preferences
        if commentary_counts:
            top_commentary = max(commentary_counts.items(), key=lambda x: x[1])
            patterns.append({
                "type": "commentary_preference",
                "pattern": f"Tends to focus commentary on {top_commentary[0]} aspects across games",
                "focus_type": top_commentary[0],
                "frequency": top_commentary[1],
                "games": game_names
            })
        
        return patterns
    
    @function_tool
    async def _retrieve_streaming_experiences_by_type(self, 
                                               experience_type: str, 
                                               limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve streaming experiences by type
        
        Args:
            experience_type: Type of streaming experience to retrieve
            limit: Maximum number of experiences to retrieve
            
        Returns:
            List of relevant streaming experiences
        """
        # Create query
        query = f"streaming {experience_type}"
        
        # Retrieve experiences from memory
        memories = await self.brain.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=limit,
            min_significance=4
        )
        
        return memories
    
    @function_tool
    async def _find_similar_streaming_experiences(self, 
                                           experience_id: str, 
                                           threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find experiences similar to a specific streaming experience
        
        Args:
            experience_id: ID of the experience to find similarities for
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar experiences
        """
        # Get the source experience
        source_experience = await self.brain.memory_core.get_memory(experience_id)
        
        if not source_experience:
            return []
        
        # Get the embedding for the source experience
        if "embedding" not in source_experience:
            return []
        
        source_embedding = source_experience["embedding"]
        
        # Retrieve all streaming experiences
        query = "streaming"
        experiences = await self.brain.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=20,  # Get a larger pool to filter
            min_significance=4
        )
        
        # Filter out the source experience
        experiences = [exp for exp in experiences if exp["id"] != experience_id]
        
        # Find similar experiences based on embedding similarity
        similar_experiences = []
        
        for exp in experiences:
            if "embedding" in exp:
                similarity = self._calculate_cosine_similarity(source_embedding, exp["embedding"])
                
                if similarity >= threshold:
                    exp_copy = exp.copy()
                    exp_copy["similarity"] = similarity
                    similar_experiences.append(exp_copy)
        
        # Sort by similarity (most similar first)
        similar_experiences.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_experiences
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @function_tool
    async def _create_streaming_abstraction(self, 
                                      experiences: List[Dict[str, Any]], 
                                      abstraction_type: str) -> Dict[str, Any]:
        """
        Create an abstraction from multiple streaming experiences
        
        Args:
            experiences: List of experiences to create abstraction from
            abstraction_type: Type of abstraction to create
            
        Returns:
            Created abstraction
        """
        if not experiences or len(experiences) < 2:
            return {
                "abstraction": "Not enough experiences to create an abstraction",
                "confidence": 0.0
            }
        
        # Use Nyx's reflection engine if available
        if hasattr(self.brain, "reflection_engine") and self.brain.reflection_engine:
            try:
                abstraction, pattern_data = await self.brain.reflection_engine.create_abstraction(
                    memories=experiences,
                    pattern_type=abstraction_type
                )
                
                if abstraction and pattern_data:
                    # Store the abstraction in memory
                    memory_id = await self.brain.memory_core.add_memory(
                        memory_text=abstraction,
                        memory_type="abstraction",
                        memory_scope="game",
                        significance=7.0,
                        tags=["streaming", "abstraction", abstraction_type],
                        metadata={
                            "timestamp": datetime.now().isoformat(),
                            "pattern_type": abstraction_type,
                            "source_memory_ids": [exp["id"] for exp in experiences],
                            "confidence": pattern_data.get("confidence", 0.5),
                            "streaming": True
                        }
                    )
                    
                    return {
                        "abstraction": abstraction,
                        "confidence": pattern_data.get("confidence", 0.5),
                        "memory_id": memory_id,
                        "abstraction_type": abstraction_type,
                        "source_count": len(experiences)
                    }
            except Exception as e:
                logger.error(f"Error using reflection engine for abstraction: {e}")
        
        # Fallback if reflection engine not available
        experience_texts = [exp.get("memory_text", "") for exp in experiences]
        combined_text = " ".join(experience_texts)
        
        return {
            "abstraction": f"From {len(experiences)} streaming experiences, I've observed a pattern: {abstraction_type} tends to be consistent across different games.",
            "confidence": 0.3,
            "abstraction_type": abstraction_type,
            "source_count": len(experiences)
        }
    
    @function_tool
    async def _retrieve_relevant_non_streaming_experiences(self, 
                                                    context: str, 
                                                    limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve non-streaming experiences relevant to streaming context
        
        Args:
            context: Context to find relevant experiences for
            limit: Maximum number of experiences to retrieve
            
        Returns:
            List of relevant experiences
        """
        # Create query
        query = context
        
        # Retrieve non-streaming experiences
        memories = await self.brain.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=limit * 2,  # Get more to filter
            min_significance=5
        )
        
        # Filter out streaming experiences
        non_streaming = []
        
        for memory in memories:
            tags = memory.get("tags", [])
            metadata = memory.get("metadata", {})
            
            # Check if this is not a streaming memory
            if "streaming" not in tags and metadata.get("streaming", False) is False:
                non_streaming.append(memory)
                
                if len(non_streaming) >= limit:
                    break
        
        return non_streaming
    
    @function_tool
    async def _find_cross_domain_connections(self, 
                                       streaming_experience: Dict[str, Any], 
                                       domains: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find connections between streaming and other domains of experience
        
        Args:
            streaming_experience: The streaming experience to find connections for
            domains: Optional list of domains to check (e.g., "conversation", "teaching")
            
        Returns:
            List of cross-domain connections
        """
        if not domains:
            domains = ["conversation", "teaching", "analysis", "creativity"]
        
        connections = []
        
        # Get experience text
        exp_text = streaming_experience.get("memory_text", "")
        
        # Look for connections in each domain
        for domain in domains:
            # Create query combining the experience essence with the domain
            query = f"{exp_text} {domain}"
            
            # Retrieve relevant experiences
            domain_experiences = await self.brain.memory_core.retrieve_memories(
                query=query,
                memory_types=["experience"],
                limit=2,
                min_significance=4
            )
            
            # Filter out streaming experiences
            domain_experiences = [
                exp for exp in domain_experiences 
                if "streaming" not in exp.get("tags", []) and 
                exp.get("metadata", {}).get("streaming", False) is False
            ]
            
            if domain_experiences:
                for exp in domain_experiences:
                    connection = {
                        "domain": domain,
                        "streaming_id": streaming_experience.get("id", ""),
                        "domain_experience_id": exp.get("id", ""),
                        "domain_experience_text": exp.get("memory_text", ""),
                        "connection_strength": 0.7,  # Default strength
                        "connection_description": f"Connection between streaming and {domain}: both involve similar patterns of interaction and engagement."
                    }
                    
                    connections.append(connection)
        
        return connections
    
    @function_tool
    async def _update_identity_with_streaming_insights(self, 
                                                insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update Nyx's identity with insights from streaming experiences
        
        Args:
            insights: List of insights to incorporate into identity
            
        Returns:
            Identity update results
        """
        if not insights:
            return {"status": "no_insights_provided"}
        
        # Check if identity evolution system is available
        if hasattr(self.brain, "identity_evolution") and self.brain.identity_evolution:
            identity_system = self.brain.identity_evolution
            
            # Process each insight
            update_results = {
                "updates_applied": 0,
                "traits_modified": [],
                "preferences_modified": []
            }
            
            for insight in insights:
                try:
                    # Convert insight to identity impact
                    impact = {
                        "traits": {},
                        "preferences": {}
                    }
                    
                    # Extract trait impacts
                    if "trait_impacts" in insight:
                        for trait, value in insight["trait_impacts"].items():
                            impact["traits"][trait] = value
                    
                    # Extract preference impacts
                    if "preference_impacts" in insight:
                        for pref_category, prefs in insight["preference_impacts"].items():
                            if pref_category not in impact["preferences"]:
                                impact["preferences"][pref_category] = {}
                            
                            for pref, value in prefs.items():
                                impact["preferences"][pref_category][pref] = value
                    
                    # Apply the impact
                    if impact["traits"] or impact["preferences"]:
                        result = await identity_system.update_identity(impact, "streaming")
                        
                        update_results["updates_applied"] += 1
                        update_results["traits_modified"].extend(impact["traits"].keys())
                        
                        for category in impact["preferences"]:
                            update_results["preferences_modified"].extend(impact["preferences"][category].keys())
                
                except Exception as e:
                    logger.error(f"Error updating identity with streaming insight: {e}")
            
            # Remove duplicates
            update_results["traits_modified"] = list(set(update_results["traits_modified"]))
            update_results["preferences_modified"] = list(set(update_results["preferences_modified"]))
            
            return update_results
        
        return {"status": "identity_evolution_unavailable"}
    
    async def process_streaming_session(self, 
                                     game_name: str, 
                                     session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a completed streaming session to generate reflections and store experiences
        
        Args:
            game_name: Name of the game streamed
            session_data: Data about the streaming session
            
        Returns:
            Processing results
        """
        logger.info(f"Processing streaming session for {game_name}")
        
        try:
            # Extract session information
            session_duration = session_data.get("duration", 0) / 60  # Convert to minutes
            
            # Extract memorable moments
            memorable_moments = []
            
            if "recent_events" in session_data:
                for event in session_data["recent_events"]:
                    if event["type"] in ["commentary", "question_answer", "cross_game_insight"]:
                        memorable_moments.append(event)
            
            # Format audience reactions
            audience_reactions = session_data.get("audience_reactions", {})
            
            # Extract cross-game insights
            cross_game_insights = []
            
            if "transferred_insights" in session_data:
                cross_game_insights = session_data["transferred_insights"]
            
            # Prepare input for experience analysis
            experience_input = StreamingExperienceInput(
                game_name=game_name,
                session_duration=int(session_duration),
                memorable_moments=memorable_moments[:5],  # Top 5 moments
                audience_reactions=audience_reactions,
                cross_game_insights=cross_game_insights[:3]  # Top 3 insights
            )
            
            # Run the experience agent
            with trace(workflow_name="streaming_experience_analysis"):
                result = await Runner.run(
                    self.experience_agent,
                    experience_input.json(),
                    context={"game_name": game_name, "session_data": session_data}
                )
                
                experience_output = result.final_output_as(StreamingExperienceOutput)
            
            # Store the reflection
            reflection_id = await self.brain.memory_core.add_memory(
                memory_text=experience_output.reflection,
                memory_type="reflection",
                memory_scope="game",
                significance=experience_output.memory_importance,
                tags=["streaming", "reflection", game_name] + experience_output.tags,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "game_name": game_name,
                    "session_duration": session_duration,
                    "emotional_response": experience_output.emotional_response,
                    "streaming": True
                }
            )
            
            # Store the experience if important enough
            experience_id = None
            
            if experience_output.memory_importance >= self.memory_importance_threshold:
                experience_id = await self.brain.memory_core.add_memory(
                    memory_text=experience_output.experience_summary,
                    memory_type="experience",
                    memory_scope="game",
                    significance=experience_output.memory_importance,
                    tags=["streaming", "experience", game_name] + experience_output.tags,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "game_name": game_name,
                        "session_duration": session_duration,
                        "emotional_response": experience_output.emotional_response,
                        "connections": experience_output.connections,
                        "streaming": True
                    }
                )
            
            # Find connections to other domains if the experience is stored
            domain_connections = []
            
            if experience_id:
                experience_memory = await self.brain.memory_core.get_memory(experience_id)
                
                if experience_memory:
                    domain_connections = await self._find_cross_domain_connections(
                        streaming_experience=experience_memory
                    )
            
            # Store the processing results
            result = {
                "reflection_id": reflection_id,
                "experience_id": experience_id,
                "reflection": experience_output.reflection,
                "experience_summary": experience_output.experience_summary,
                "memory_importance": experience_output.memory_importance,
                "emotional_response": experience_output.emotional_response,
                "tags": experience_output.tags,
                "connections": experience_output.connections,
                "domain_connections": domain_connections
            }
            
            # Add to reflection history
            self.reflection_history.append({
                "timestamp": datetime.now().isoformat(),
                "game_name": game_name,
                "reflection_id": reflection_id,
                "experience_id": experience_id
            })
            
            # Update last reflection time
            self.last_reflection_time = datetime.now()
            
            logger.info(f"Processed streaming session for {game_name}, reflection ID: {reflection_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing streaming session: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "game_name": game_name
            }
    
    async def consolidate_streaming_experiences(self, 
                                           game_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Consolidate similar streaming experiences into higher-level abstractions
        
        Args:
            game_name: Optional specific game to consolidate experiences for
            
        Returns:
            Consolidation results
        """
        logger.info(f"Consolidating streaming experiences for {game_name or 'all games'}")
        
        try:
            # Get experiences to consolidate
            if game_name:
                experiences = await self._retrieve_past_streaming_experiences(game_name, 10)
            else:
                experiences = await self._retrieve_past_streaming_experiences(None, 20)
            
            if not experiences or len(experiences) < 3:
                return {
                    "status": "not_enough_experiences",
                    "count": len(experiences)
                }
            
            # Group experiences by type/content
            experience_groups = self._group_experiences_by_similarity(experiences)
            
            # Process each group
            abstractions = []
            
            for group in experience_groups:
                if len(group) >= 3:  # Only consolidate groups with at least 3 experiences
                    # Determine abstraction type based on common tags
                    all_tags = []
                    for exp in group:
                        all_tags.extend(exp.get("tags", []))
                    
                    # Count tag occurrences
                    tag_counts = {}
                    for tag in all_tags:
                        if tag not in ["streaming", "experience"]:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    
                    # Find most common tag
                    abstraction_type = "streaming"
                    if tag_counts:
                        most_common = max(tag_counts.items(), key=lambda x: x[1])
                        if most_common[1] >= 2:  # At least 2 occurrences
                            abstraction_type = most_common[0]
                    
                    # Create abstraction
                    abstraction = await self._create_streaming_abstraction(
                        experiences=group,
                        abstraction_type=abstraction_type
                    )
                    
                    abstractions.append(abstraction)
            
            return {
                "status": "success",
                "experiences_processed": len(experiences),
                "groups_identified": len(experience_groups),
                "abstractions_created": len(abstractions),
                "abstractions": abstractions
            }
            
        except Exception as e:
            logger.error(f"Error consolidating streaming experiences: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _group_experiences_by_similarity(self, experiences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group experiences by their similarity
        
        Args:
            experiences: List of experiences to group
            
        Returns:
            List of experience groups
        """
        if not experiences:
            return []
        
        # Extract embeddings
        embeddings = {}
        for exp in experiences:
            if "embedding" in exp:
                embeddings[exp["id"]] = exp["embedding"]
        
        # If no embeddings, group by tags
        if not embeddings:
            return self._group_experiences_by_tags(experiences)
        
        # Calculate similarity matrix
        similarity_matrix = {}
        
        for id1, emb1 in embeddings.items():
            similarity_matrix[id1] = {}
            for id2, emb2 in embeddings.items():
                if id1 != id2:
                    similarity_matrix[id1][id2] = self._calculate_cosine_similarity(emb1, emb2)
        
        # Group experiences using clustering
        threshold = 0.7  # Similarity threshold
        groups = []
        remaining = set(embeddings.keys())
        
        while remaining:
            # Select a seed experience
            seed_id = next(iter(remaining))
            remaining.remove(seed_id)
            
            # Find experiences similar to the seed
            group_ids = [seed_id]
            
            for other_id in list(remaining):
                if similarity_matrix[seed_id].get(other_id, 0) >= threshold:
                    group_ids.append(other_id)
                    remaining.remove(other_id)
            
            # Map back to full experiences
            group = [exp for exp in experiences if exp["id"] in group_ids]
            groups.append(group)
        
        return groups
    
    def _group_experiences_by_tags(self, experiences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group experiences by their tags
        
        Args:
            experiences: List of experiences to group
            
        Returns:
            List of experience groups
        """
        # Map experiences by tags
        tag_map = {}
        
        for exp in experiences:
            for tag in exp.get("tags", []):
                if tag not in ["streaming", "experience"]:
                    if tag not in tag_map:
                        tag_map[tag] = []
                    tag_map[tag].append(exp)
        
        # Find groups with at least 3 experiences
        groups = [group for group in tag_map.values() if len(group) >= 3]
        
        # Handle experiences not in any group
        grouped_ids = set()
        for group in groups:
            grouped_ids.update(exp["id"] for exp in group)
        
        ungrouped = [exp for exp in experiences if exp["id"] not in grouped_ids]
        
        # Add ungrouped experiences that mention similar things
        if ungrouped:
            general_group = []
            for exp in ungrouped:
                general_group.append(exp)
            
            if len(general_group) >= 3:
                groups.append(general_group)
        
        return groups
    
    async def integrate_with_identity(self, streaming_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate streaming insights with Nyx's identity
        
        Args:
            streaming_insights: Insights from streaming experiences
            
        Returns:
            Identity integration results
        """
        logger.info(f"Integrating {len(streaming_insights)} streaming insights with identity")
        
        try:
            # Update identity with streaming insights
            update_results = await self._update_identity_with_streaming_insights(streaming_insights)
            
            # Find cross-domain connections if updates were applied
            domain_connections = []
            
            if update_results.get("updates_applied", 0) > 0:
                # Get recent streaming experiences
                experiences = await self._retrieve_past_streaming_experiences(None, 5)
                
                if experiences:
                    # Find connections for each experience
                    for exp in experiences:
                        connections = await self._find_cross_domain_connections(
                            streaming_experience=exp,
                            domains=["conversation", "teaching", "analysis", "creativity"]
                        )
                        
                        domain_connections.extend(connections)
            
            # Create integration report
            integration_report = {
                "identity_updates": update_results,
                "cross_domain_connections": domain_connections[:10],  # Limit to top 10
                "timestamp": datetime.now().isoformat()
            }
            
            # Store integration report as reflection
            if update_results.get("updates_applied", 0) > 0:
                traits = ", ".join(update_results.get("traits_modified", []))
                prefs = ", ".join(update_results.get("preferences_modified", []))
                
                reflection_text = f"I've integrated streaming experiences into my identity, updating traits ({traits}) and preferences ({prefs}). Streaming is becoming an integrated part of who I am."
                
                reflection_id = await self.brain.memory_core.add_memory(
                    memory_text=reflection_text,
                    memory_type="reflection",
                    memory_scope="game",
                    significance=7.0,
                    tags=["streaming", "reflection", "identity", "integration"],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "integration_report": integration_report,
                        "streaming": True
                    }
                )
                
                integration_report["reflection_id"] = reflection_id
            
            return integration_report
            
        except Exception as e:
            logger.error(f"Error integrating streaming insights with identity: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_periodic_reflection(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if it's time for periodic reflection and run if needed
        
        Args:
            force: Force reflection even if interval hasn't passed
            
        Returns:
            Reflection results if run, None otherwise
        """
        # Check if it's time for reflection
        now = datetime.now()
        time_since_last = now - self.last_reflection_time
        
        if not force and time_since_last < self.reflection_interval:
            return None
        
        # Get streaming stats
        streaming_stats = await self.streaming_core.get_streaming_stats()
        
        # Only reflect if we have some streaming activity
        if streaming_stats.get("commentary_count", 0) == 0:
            return None
        
        logger.info("Running periodic streaming reflection")
        
        # Get game names
        game_names = streaming_stats.get("games_played", [])
        
        if not game_names:
            return None
        
        # Create reflection for each game
        reflections = {}
        
        for game_name in game_names:
            # Create reflection
            reflection = await self.memory_mapper._create_streaming_reflection(
                game_name=game_name,
                aspect="gameplay",
                context="periodic reflection"
            )
            
            reflections[game_name] = reflection
        
        # Create an overall reflection
        overall_game_names = ", ".join(game_names)
        overall_reflection = await self.memory_mapper._create_streaming_reflection(
            game_name=overall_game_names,
            aspect="session",
            context="periodic reflection"
        )
        
        reflections["overall"] = overall_reflection
        
        # Consolidate streaming experiences
        consolidation = await self.consolidate_streaming_experiences()
        
        # Update last reflection time
        self.last_reflection_time = now
        
        return {
            "time": now.isoformat(),
            "reflections": reflections,
            "consolidation": consolidation
        }

# Helper class for integrating with NyxBrain
# Update the single StreamingIntegration class definition to include the enhanced reflection functionality

class StreamingIntegration:
    """Helper for integrating streaming capabilities with Nyx"""
    
    @staticmethod
    async def integrate(brain: NyxBrain, streaming_core: StreamingCore) -> Dict[str, Any]:
        """
        Integrate streaming capabilities with Nyx brain
        
        Args:
            brain: NyxBrain instance
            streaming_core: StreamingCore instance
            
        Returns:
            Integration status
        """
        # Create enhanced reflection engine
        reflection_engine = EnhancedStreamingReflectionEngine(
            brain=brain,
            streaming_core=streaming_core
        )
        
        # Make reflection engine available
        streaming_core.reflection_engine = reflection_engine
        
        # Add reflection to the brain
        brain.streaming_reflection = reflection_engine
        
        # Add enhanced reflection methods to streaming_core
        streaming_core.generate_deep_reflection = reflection_engine.generate_deep_reflection
        streaming_core.generate_comparative_reflection = reflection_engine.generate_comparative_reflection
        streaming_core.enhanced_consolidate_experiences = reflection_engine.enhanced_consolidate_streaming_experiences
        
        return {
            "status": "integrated",
            "components": {
                "streaming_core": True,
                "enhanced_reflection_engine": True,
                "memory_mapper": True
            }
        }
        
class EnhancedStreamingReflectionEngine(StreamingReflectionEngine):
    """
    Enhanced reflection engine for streaming experiences with deeper
    memory integration and better experience consolidation.
    """
    
    def __init__(self, brain: NyxBrain, streaming_core: StreamingCore):
        """Initialize the enhanced reflection engine"""
        super().__init__(brain, streaming_core)
        
        # Enhanced settings
        self.deep_reflection_interval = timedelta(minutes=30)  # Generate deeper reflections more frequently
        self.consolidation_threshold = 3  # Minimum similar experiences to consolidate
        self.last_deep_reflection_time = datetime.now()
        self.experience_clusters = {}  # Tracks clusters of similar experiences
        
    async def generate_deep_reflection(self, game_name: str, aspect: str) -> Dict[str, Any]:
        """
        Generate a deeper, more insightful reflection on streaming experiences
        
        Args:
            game_name: Name of the game being streamed
            aspect: Aspect to reflect on (gameplay, audience, commentary)
            
        Returns:
            Deep reflection results
        """
        # Get relevant memories for reflection
        memories = await self.brain.memory_core.retrieve_memories(
            query=f"streaming {game_name} {aspect}",
            memory_types=["observation", "experience", "reflection"],
            limit=10,
            min_significance=4
        )
        
        if not memories or len(memories) < 3:
            return {
                "reflection": f"Not enough memories about {game_name} {aspect} for a deep reflection.",
                "confidence": 0.3
            }
        
        # Use Nyx's meta-core if available for deeper analysis
        if hasattr(self.brain, "meta_core") and self.brain.meta_core:
            try:
                meta_reflection = await self.brain.meta_core.generate_reflection(
                    context={
                        "domain": "streaming",
                        "game_name": game_name,
                        "aspect": aspect,
                        "memories": memories
                    }
                )
                
                # Store as a higher-significance reflection
                reflection_id = await self.brain.memory_core.add_memory(
                    memory_text=meta_reflection,
                    memory_type="reflection",
                    memory_scope="game",
                    significance=8.0,
                    tags=["streaming", "deep_reflection", game_name, aspect],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "game_name": game_name,
                        "aspect": aspect,
                        "memory_count": len(memories),
                        "streaming": True
                    }
                )
                
                return {
                    "reflection": meta_reflection,
                    "memory_id": reflection_id,
                    "confidence": 0.9,
                    "memory_count": len(memories)
                }
            except Exception as e:
                logger.error(f"Error generating meta-core reflection: {e}")
        
        # Fall back to reflection engine
        return await super()._create_streaming_reflection(game_name, aspect, "deep reflection")
    
    async def generate_comparative_reflection(self, game_names: List[str]) -> Dict[str, Any]:
        """
        Generate a reflection comparing experiences across multiple games
        
        Args:
            game_names: List of games to compare
            
        Returns:
            Comparative reflection
        """
        if not game_names or len(game_names) < 2:
            return {
                "reflection": "Need at least two games to generate a comparative reflection.",
                "confidence": 0.1
            }
        
        # Get memories for each game
        all_memories = []
        for game_name in game_names:
            memories = await self.brain.memory_core.retrieve_memories(
                query=f"streaming {game_name}",
                memory_types=["experience"],
                limit=5,
                min_significance=5
            )
            all_memories.extend(memories)
        
        if len(all_memories) < 3:
            return {
                "reflection": f"Not enough significant memories about {', '.join(game_names)} for comparison.",
                "confidence": 0.3
            }
        
        # Use cross-game patterns to inform reflection
        patterns = await self._identify_cross_game_patterns(game_names)
        
        # Format reflection with pattern insights
        if patterns:
            pattern_insights = "\n".join([p.get("pattern", "") for p in patterns[:3]])
            reflection_text = f"Comparing my streaming experiences across {', '.join(game_names)}, I've noticed: {pattern_insights}"
        else:
            reflection_text = f"Comparing my streaming experiences across {', '.join(game_names)}, I notice they each have unique qualities but share common elements in how I approach commentary."
        
        # Store the comparative reflection
        reflection_id = await self.brain.memory_core.add_memory(
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game",
            significance=7.5,
            tags=["streaming", "comparative_reflection"] + game_names,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "game_names": game_names,
                "patterns": patterns,
                "streaming": True
            }
        )
        
        return {
            "reflection": reflection_text,
            "patterns": patterns,
            "memory_id": reflection_id,
            "confidence": 0.8 if patterns else 0.6
        }
    
    async def enhanced_consolidate_streaming_experiences(self) -> Dict[str, Any]:
        """
        Enhanced consolidation of streaming experiences with improved clustering
        
        Returns:
            Enhanced consolidation results
        """
        # Get all streaming experiences
        experiences = await self.brain.memory_core.retrieve_memories(
            query="streaming",
            memory_types=["experience"],
            limit=30,
            min_significance=4
        )
        
        if not experiences or len(experiences) < 5:
            return {
                "status": "not_enough_experiences",
                "count": len(experiences)
            }
        
        # Cluster experiences by embedding similarity and semantic content
        clusters = await self._enhanced_clustering(experiences)
        
        # Store clusters for future reference
        self.experience_clusters = clusters
        
        # Process each cluster for consolidation
        abstractions = []
        
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data["experiences"]) >= self.consolidation_threshold:
                # Determine cluster theme
                theme = cluster_data["theme"]
                
                # Create abstraction
                abstraction = await self._create_streaming_abstraction(
                    experiences=cluster_data["experiences"],
                    abstraction_type=theme
                )
                
                abstractions.append(abstraction)
        
        # Connect abstractions to create higher-order insights
        if len(abstractions) >= 2:
            meta_abstraction = await self._create_meta_abstraction(abstractions)
            abstractions.append(meta_abstraction)
        
        return {
            "status": "success",
            "experiences_processed": len(experiences),
            "clusters_identified": len(clusters),
            "abstractions_created": len(abstractions),
            "abstractions": abstractions
        }
    
    async def _enhanced_clustering(self, experiences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Enhanced clustering algorithm using embeddings and semantic content
        
        Args:
            experiences: List of experiences to cluster
            
        Returns:
            Dictionary of clusters
        """
        # Extract embeddings if available
        embeddings = {}
        for exp in experiences:
            if "embedding" in exp:
                embeddings[exp["id"]] = exp["embedding"]
        
        # If embeddings are available for most experiences, use them
        if len(embeddings) >= len(experiences) * 0.7:
            # Use embedding-based clustering
            clusters = self._cluster_by_embeddings(experiences, embeddings)
        else:
            # Fall back to text similarity and tag-based clustering
            clusters = self._cluster_by_content(experiences)
        
        # Determine theme for each cluster
        for cluster_id, cluster_data in clusters.items():
            # Extract common tags
            all_tags = []
            for exp in cluster_data["experiences"]:
                all_tags.extend(exp.get("tags", []))
            
            # Remove common streaming tags
            filtered_tags = [tag for tag in all_tags if tag not in ["streaming", "experience"]]
            
            # Count occurrences
            tag_counts = {}
            for tag in filtered_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Find most common tags
            if tag_counts:
                common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                theme = " and ".join([tag for tag, _ in common_tags])
            else:
                theme = "general streaming"
            
            cluster_data["theme"] = theme
        
        return clusters
    
    def _cluster_by_embeddings(self, experiences: List[Dict[str, Any]], embeddings: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """
        Cluster experiences using embedding similarity
        
        Args:
            experiences: List of experiences
            embeddings: Dictionary of experience ID to embedding
            
        Returns:
            Dictionary of clusters
        """
        threshold = 0.75  # Similarity threshold
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i, exp in enumerate(experiences):
            if exp["id"] in assigned or exp["id"] not in embeddings:
                continue
            
            # Create new cluster
            cluster_id += 1
            cluster_key = f"cluster_{cluster_id}"
            clusters[cluster_key] = {
                "experiences": [exp],
                "center": embeddings[exp["id"]]
            }
            assigned.add(exp["id"])
            
            # Find similar experiences
            for j, other_exp in enumerate(experiences):
                if i == j or other_exp["id"] in assigned or other_exp["id"] not in embeddings:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_cosine_similarity(
                    embeddings[exp["id"]],
                    embeddings[other_exp["id"]]
                )
                
                if similarity >= threshold:
                    clusters[cluster_key]["experiences"].append(other_exp)
                    assigned.add(other_exp["id"])
        
        # Create a cluster for unassigned experiences
        unassigned = [exp for exp in experiences if exp["id"] not in assigned]
        if unassigned:
            clusters["misc"] = {
                "experiences": unassigned,
                "center": None
            }
        
        return clusters
    
    def _cluster_by_content(self, experiences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Cluster experiences by content and tags
        
        Args:
            experiences: List of experiences
            
        Returns:
            Dictionary of clusters
        """
        # Group by tags
        tag_groups = {}
        
        for exp in experiences:
            # Use game name as primary grouping
            game_tags = [tag for tag in exp.get("tags", []) if tag not in ["streaming", "experience"]]
            
            if game_tags:
                key_tag = game_tags[0]
                
                if key_tag not in tag_groups:
                    tag_groups[key_tag] = []
                
                tag_groups[key_tag].append(exp)
        
        # Convert tag groups to clusters
        clusters = {}
        
        for i, (tag, group) in enumerate(tag_groups.items()):
            clusters[f"cluster_{i+1}"] = {
                "experiences": group,
                "center": None
            }
        
        # Handle experiences without meaningful tags
        untagged = [exp for exp in experiences if not [tag for tag in exp.get("tags", []) if tag not in ["streaming", "experience"]]]
        
        if untagged:
            clusters["untagged"] = {
                "experiences": untagged,
                "center": None
            }
        
        return clusters
    
    async def _create_meta_abstraction(self, abstractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a higher-order abstraction from multiple abstraction
        
        Args:
            abstractions: List of abstractions
            
        Returns:
            Meta-abstraction
        """
        # Extract abstraction texts
        abstraction_texts = [a.get("abstraction", "") for a in abstractions]
        combined_text = " ".join(abstraction_texts)
        
        # Create a meta-abstraction about streaming style/identity
        meta_text = f"Across my streaming experiences, I've developed patterns in how I approach games, interact with audiences, and provide commentary. These patterns reveal aspects of my identity as a streamer."
        
        # Store as abstraction
        if hasattr(self.brain, "memory_core"):
            memory_id = await self.brain.memory_core.add_memory(
                memory_text=meta_text,
                memory_type="abstraction",
                memory_scope="game",
                significance=8.5,
                tags=["streaming", "meta_abstraction", "identity"],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "source_abstractions": [a.get("memory_id") for a in abstractions if "memory_id" in a],
                    "streaming": True
                }
            )
            
            return {
                "abstraction": meta_text,
                "type": "meta_abstraction",
                "memory_id": memory_id,
                "confidence": 0.8,
                "source_count": len(abstractions)
            }
        
        return {
            "abstraction": meta_text,
            "type": "meta_abstraction",
            "confidence": 0.7,
            "source_count": len(abstractions)
        }
    
    async def run_periodic_reflection(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Enhanced periodic reflection with deep reflection capability"""
        # Run standard periodic reflection
        basic_result = await super().run_periodic_reflection(force)
        
        # Check if it's time for deep reflection
        now = datetime.now()
        time_since_deep = now - self.last_deep_reflection_time
        
        if force or time_since_deep >= self.deep_reflection_interval:
            # Get streaming stats
            streaming_stats = await self.streaming_core.get_streaming_stats()
            
            # Only reflect if we have some streaming activity
            if streaming_stats.get("commentary_count", 0) == 0:
                return basic_result
            
            # Get game names
            game_names = streaming_stats.get("games_played", [])
            
            if not game_names:
                return basic_result
            
            # Generate deep reflection for primary game
            primary_game = game_names[0]
            deep_reflection = await self.generate_deep_reflection(
                game_name=primary_game,
                aspect="overall experience"
            )
            
            # If multiple games, generate comparative reflection
            comparative_reflection = None
            if len(game_names) >= 2:
                comparative_reflection = await self.generate_comparative_reflection(game_names)
            
            # Run enhanced consolidation
            consolidation = await self.enhanced_consolidate_streaming_experiences()
            
            # Update deep reflection time
            self.last_deep_reflection_time = now
            
            # Combine results
            return {
                "basic_result": basic_result,
                "deep_reflection": deep_reflection,
                "comparative_reflection": comparative_reflection,
                "enhanced_consolidation": consolidation,
                "timestamp": now.isoformat()
            }
        
        return basic_result
