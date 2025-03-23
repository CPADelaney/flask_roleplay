# nyx/streamer/nyx_streaming_core.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import from Nyx core systems
from nyx.core.nyx_brain import NyxBrain
from nyx.core.memory_core import MemoryCore
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
from nyx.core.meta_core import MetaCore

# Import from streaming module
from nyx.streamer.gamer_girl import (
    AdvancedGameAgentSystem,
    GameState,
    enhanced_commentary_agent,
    enhanced_question_agent,
    enhanced_triage_agent
)

# Import from OpenAI Agents SDK
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff
)

logger = logging.getLogger("nyx_streaming")

class StreamingMemoryMapper:
    """
    Maps between streaming events and Nyx memory system.
    Handles converting game events into memories and retrieving
    relevant memories for streaming commentary.
    """
    
    def __init__(self, memory_core: MemoryCore):
        """
        Initialize with a memory core reference
        
        Args:
            memory_core: Nyx's memory core instance
        """
        self.memory_core = memory_core
        self.experience_memory_threshold = 0.7  # Minimum significance to store as experience
        self.last_memory_id = None
    
    async def store_gameplay_memory(self, 
                                   game_name: str,
                                   event_type: str, 
                                   event_data: Dict[str, Any],
                                   significance: float = 5.0) -> str:
        """
        Store a gameplay event as a memory
        
        Args:
            game_name: Name of the game
            event_type: Type of gameplay event
            event_data: Data about the event
            significance: Importance of the memory (1-10)
            
        Returns:
            Memory ID
        """
        # Format memory text based on event type
        if event_type == "commentary":
            memory_text = f"While streaming {game_name}, I commented: {event_data.get('text', '')}"
        elif event_type == "question_answer":
            memory_text = f"While streaming {game_name}, someone asked: '{event_data.get('question', '')}' and I answered: '{event_data.get('answer', '')}'"
        elif event_type == "game_moment":
            memory_text = f"While streaming {game_name}, I observed: {event_data.get('description', '')}"
        elif event_type == "speech_transcribed":
            memory_text = f"While streaming {game_name}, a character said: '{event_data.get('text', '')}'"
        elif event_type == "cross_game_insight":
            memory_text = f"While streaming {game_name}, I made a connection to {event_data.get('source_game', '')}: {event_data.get('content', '')}"
        else:
            memory_text = f"While streaming {game_name}: {json.dumps(event_data)}"
        
        # Prepare tags
        tags = ["streaming", game_name, event_type]
        if "tags" in event_data and isinstance(event_data["tags"], list):
            tags.extend(event_data["tags"])
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "game_name": game_name,
            "event_type": event_type,
            "event_data": event_data,
            "streaming": True
        }
        
        # Add emotional context if available
        if "emotional_context" in event_data:
            metadata["emotional_context"] = event_data["emotional_context"]
        
        # Store the memory
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata=metadata
        )
        
        self.last_memory_id = memory_id
        
        # If significance is high enough, also store as an experience
        if significance >= self.experience_memory_threshold:
            await self._store_as_experience(memory_id, memory_text, game_name, event_type, event_data)
        
        return memory_id
    
    async def _store_as_experience(self, 
                                 memory_id: str,
                                 memory_text: str,
                                 game_name: str,
                                 event_type: str,
                                 event_data: Dict[str, Any]) -> str:
        """
        Store a significant memory as an experience
        
        Args:
            memory_id: ID of the original memory
            memory_text: Text of the memory
            game_name: Name of the game
            event_type: Type of event
            event_data: Data about the event
            
        Returns:
            Experience memory ID
        """
        # Prepare experience-specific metadata
        scenario_type = "gaming"
        if event_type == "cross_game_insight":
            scenario_type = "analysis"
        elif event_type == "question_answer":
            scenario_type = "teaching"
        
        # Create experience memory
        experience_text = f"I had an experience streaming {game_name}: {memory_text}"
        
        # Prepare metadata with emotional context
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "scenario_type": scenario_type,
            "game_name": game_name,
            "source_memory_id": memory_id,
            "event_type": event_type,
            "streaming": True
        }
        
        # Add emotional context if available
        if "emotional_context" in event_data:
            metadata["emotional_context"] = event_data["emotional_context"]
        
        # Store as experience
        experience_id = await self.memory_core.add_memory(
            memory_text=experience_text,
            memory_type="experience",
            memory_scope="game",
            significance=7.0,  # Experiences tend to be more significant
            tags=["streaming", "experience", game_name, event_type, scenario_type],
            metadata=metadata
        )
        
        return experience_id
    
    async def retrieve_relevant_memories(self, 
                                       game_name: str,
                                       context: str,
                                       limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current streaming context
        
        Args:
            game_name: Name of the current game
            context: Current context description
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        # Create search query
        query = f"{game_name} {context}"
        
        # Retrieve memories
        memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "experience"],
            limit=limit,
            min_significance=4
        )
        
        return memories
    
    async def retrieve_game_experiences(self, 
                                     game_name: str,
                                     aspect: Optional[str] = None,
                                     limit: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve past experiences related to this game
        
        Args:
            game_name: Name of the game
            aspect: Optional specific aspect of the game
            limit: Maximum number of experiences to retrieve
            
        Returns:
            List of relevant experiences
        """
        # Create search query
        query = f"streaming {game_name}"
        if aspect:
            query += f" {aspect}"
        
        # Retrieve experiences
        experiences = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["experience"],
            limit=limit,
            min_significance=5
        )
        
        return experiences
    
    async def create_streaming_reflection(self, 
                                       game_name: str,
                                       aspect: str,
                                       context: str) -> Dict[str, Any]:
        """
        Create a reflection about the streaming experience
        
        Args:
            game_name: Name of the game
            aspect: Aspect to reflect on (e.g., "gameplay", "audience", "commentary")
            context: Additional context for the reflection
            
        Returns:
            Reflection result
        """
        # Retrieve relevant memories for reflection
        query = f"streaming {game_name} {aspect} {context}"
        memories = await self.memory_core.retrieve_memories(
            query=query,
            memory_types=["observation", "experience"],
            limit=5,
            min_significance=3
        )
        
        if not memories:
            return {
                "reflection": f"I don't have enough memories about streaming {game_name} to create a meaningful reflection about {aspect}.",
                "confidence": 0.2
            }
        
        # Create reflection through memory core
        reflection_result = await self.memory_core.create_reflection_from_memories(topic=f"Streaming {game_name}: {aspect}")
        
        return reflection_result

class StreamingCore:
    """
    Core integration between Nyx's systems and streaming capabilities.
    Coordinates between the streaming agent system and Nyx's core systems.
    """
    
    def __init__(self, brain: NyxBrain, video_source=0, audio_source=None):
        """
        Initialize the streaming core with a reference to NyxBrain
        
        Args:
            brain: NyxBrain instance
            video_source: Video capture source
            audio_source: Audio capture source
        """
        self.brain = brain
        self.user_id = brain.user_id
        self.conversation_id = brain.conversation_id
        
        # Initialize streaming agent system
        self.streaming_system = AdvancedGameAgentSystem(
            video_source=video_source,
            audio_source=audio_source
        )
        
        # Create memory mapper
        self.memory_mapper = StreamingMemoryMapper(brain.memory_core)
        
        # Create streaming-specific agents
        self.streaming_orchestrator = self._create_streaming_orchestrator()
        self.reflection_agent = self._create_streaming_reflection_agent()
        self.experience_agent = self._create_streaming_experience_agent()
        
        # Track streaming session
        self.session_start_time = None
        self.session_stats = {
            "commentary_count": 0,
            "questions_answered": 0,
            "memories_created": 0,
            "experiences_stored": 0,
            "reflections_generated": 0,
            "games_played": []
        }
        
        # Register enhancement functions on the streaming system
        self._enhance_streaming_system()
        
        logger.info(f"StreamingCore initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    def _enhance_streaming_system(self):
        """Add Nyx brain capabilities to the streaming system"""
        # Store original method references
        original_commentary = self.streaming_system._generate_commentary
        original_answer = self.streaming_system._answer_question
        original_process = self.streaming_system._process_game_frame
        
        # Enhance commentary with memory and reflection
        async def enhanced_commentary(extended_context, priority_source=None):
            # Run original commentary generation
            await original_commentary(extended_context, priority_source)
            
            # Get the latest commentary from the game state
            game_state = extended_context.context
            if hasattr(game_state, "recent_events") and game_state.recent_events:
                for event in reversed(game_state.recent_events):
                    if event["type"] == "commentary":
                        # Store commentary as memory
                        await self.memory_mapper.store_gameplay_memory(
                            game_name=game_state.game_name or "Unknown Game",
                            event_type="commentary",
                            event_data=event["data"],
                            significance=5.0
                        )
                        
                        # Update stats
                        self.session_stats["commentary_count"] += 1
                        self.session_stats["memories_created"] += 1
                        break
        
        # Enhance question answering with memory and experience
        async def enhanced_answer_question(extended_context):
            # Run original answer generation
            await original_answer(extended_context)
            
            # Get the latest answer from the game state
            game_state = extended_context.context
            if hasattr(game_state, "answered_questions") and game_state.answered_questions:
                latest_answer = game_state.answered_questions[-1]
                
                # Store question and answer as memory
                await self.memory_mapper.store_gameplay_memory(
                    game_name=game_state.game_name or "Unknown Game",
                    event_type="question_answer",
                    event_data={
                        "question": latest_answer["question"],
                        "answer": latest_answer["answer"],
                        "username": latest_answer["username"]
                    },
                    significance=6.0
                )
                
                # Update stats
                self.session_stats["questions_answered"] += 1
                self.session_stats["memories_created"] += 1
        
        # Enhance frame processing to include memory retrieval
        async def enhanced_process_game_frame():
            # Run original frame processing
            await original_process()
            
            # Add memory retrieval for context enhancement if game is identified
            game_state = self.streaming_system.game_state
            if game_state.game_name and game_state.frame_count % 150 == 0:  # Every ~5 seconds at 30fps
                # Retrieve relevant memories
                context_description = ""
                if game_state.current_location:
                    context_description += f"in {game_state.current_location.get('name', 'this area')} "
                if game_state.detected_action:
                    context_description += f"while {game_state.detected_action.get('name', 'doing this action')} "
                
                memories = await self.memory_mapper.retrieve_relevant_memories(
                    game_name=game_state.game_name,
                    context=context_description
                )
                
                # Store memories in game state for agent access
                game_state.retrieved_memories = memories
                
                # If this is a new game, add it to our list of games played
                if game_state.game_name not in self.session_stats["games_played"]:
                    self.session_stats["games_played"].append(game_state.game_name)
                    
                    # Generate a reflection after playing for a while
                    if game_state.frame_count > 1000:  # After significant gameplay
                        reflection = await self.memory_mapper.create_streaming_reflection(
                            game_name=game_state.game_name,
                            aspect="gameplay",
                            context=context_description
                        )
                        
                        # Store the reflection
                        if reflection and "reflection" in reflection:
                            await self.brain.memory_core.add_memory(
                                memory_text=reflection["reflection"],
                                memory_type="reflection",
                                memory_scope="game",
                                significance=6.0,
                                tags=["streaming", "reflection", game_state.game_name],
                                metadata={
                                    "timestamp": datetime.now().isoformat(),
                                    "game_name": game_state.game_name,
                                    "context": context_description,
                                    "streaming": True
                                }
                            )
                            
                            # Update stats
                            self.session_stats["reflections_generated"] += 1
        
        # Replace original methods with enhanced versions
        self.streaming_system._generate_commentary = enhanced_commentary
        self.streaming_system._answer_question = enhanced_answer_question
        self.streaming_system._process_game_frame = enhanced_process_game_frame
        
        # Add new methods to streaming system
        self.streaming_system.retrieve_game_experiences = self.memory_mapper.retrieve_game_experiences
        self.streaming_system.create_streaming_reflection = self.memory_mapper.create_streaming_reflection
        self.streaming_system.nyx_brain = self.brain

        # Add this method to StreamingMemoryMapper in nyx_streaming_core.py
        
        async def store_gameplay_experience(self, 
                                          game_name: str,
                                          moment_data: Dict[str, Any],
                                          emotional_context: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Store a significant gaming moment as an experience in Nyx's experience system
            
            Args:
                game_name: Name of the game
                moment_data: Data about the gaming moment
                emotional_context: Optional emotional context
                
            Returns:
                Experience storage results
            """
            # Create experience memory
            memory_text = f"While streaming {game_name}, I experienced: {moment_data.get('description', '')}"
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "game_name": game_name,
                "moment_data": moment_data,
                "streaming": True,
                "scenario_type": "gaming"
            }
            
            # Add emotional context if provided
            if emotional_context:
                metadata["emotional_context"] = emotional_context
            
            # Prepare tags
            tags = ["streaming", "experience", game_name]
            if "tags" in moment_data:
                tags.extend(moment_data["tags"])
            
            # Store in memory system
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="experience",
                memory_scope="game",
                significance=8.0,  # Higher significance for experiences
                tags=tags,
                metadata=metadata
            )
            
            # If experience interface exists, also store there
            if hasattr(self.brain, "experience_interface") and self.brain.experience_interface:
                try:
                    exp_result = await self.brain.experience_interface._store_experience(
                        RunContextWrapper(context=None),
                        memory_text=memory_text,
                        scenario_type="gaming",
                        entities=[game_name],
                        emotional_context=emotional_context or {},
                        significance=8.0,
                        tags=tags,
                        user_id=str(self.brain.user_id)
                    )
                    
                    if exp_result and "id" in exp_result:
                        return {
                            "memory_id": memory_id,
                            "experience_id": exp_result["id"],
                            "stored": True
                        }
                except Exception as e:
                    logger.error(f"Error storing in experience interface: {e}")
            
            return {
                "memory_id": memory_id,
                "stored": True
            }

        async def update_identity_from_streaming(self, 
                                               game_name: str,
                                               streaming_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Update Nyx's identity based on streaming experiences
            
            Args:
                game_name: Name of the game
                streaming_data: Data about the streaming session
                
            Returns:
                Identity update results
            """
            if not hasattr(self.brain, "identity_evolution") or not self.brain.identity_evolution:
                return {"status": "identity_evolution_unavailable"}
            
            # Extract gameplay preferences from streaming data
            preferences = {}
            traits = {}
            
            # Extract commentary preferences
            if "commentary_count" in streaming_data:
                preferences["activity_types"] = {
                    "streaming": 0.2  # Increase preference for streaming
                }
            
            # Extract game genre preferences if available
            if "game_genre" in streaming_data and streaming_data["game_genre"]:
                preferences["genre_preferences"] = {
                    streaming_data["game_genre"]: 0.1  # Slight increase for this genre
                }
            
            # Extract trait impacts
            if "performance_metrics" in streaming_data:
                metrics = streaming_data["performance_metrics"]
                
                # High commentary count suggests verbal expressiveness
                if metrics.get("commentary_count", 0) > 20:
                    traits["expressiveness"] = 0.1
                
                # Many questions answered suggests helpfulness
                if metrics.get("questions_answered", 0) > 10:
                    traits["helpfulness"] = 0.1
                    
                # Many experiences shared suggests openness
                if metrics.get("experiences_shared", 0) > 5:
                    traits["openness"] = 0.1
            
            # Create identity impact
            impact = {
                "preferences": preferences,
                "traits": traits
            }
            
            # Apply impact
            if traits or preferences:
                try:
                    result = await self.brain.identity_evolution.update_identity_from_experience(
                        experience={
                            "scenario_type": "gaming",
                            "game_name": game_name,
                            "streaming": True
                        },
                        impact=impact
                    )
                    
                    return {
                        "identity_updated": True,
                        "traits_updated": list(traits.keys()),
                        "preferences_updated": list(preferences.keys())
                    }
                except Exception as e:
                    logger.error(f"Error updating identity: {e}")
                    return {"error": str(e)}
            
            return {"identity_updated": False, "reason": "no_significant_impact"}
        
        async def process_significant_moment(self, 
                                           game_name: str,
                                           event_type: str,
                                           event_data: Dict[str, Any],
                                           significance: float = 7.0) -> Dict[str, Any]:
            """
            Process a significant moment during streaming to integrate with all systems
            
            Args:
                game_name: Name of the game
                event_type: Type of event (e.g., 'discovery', 'achievement', 'insight')
                event_data: Data about the event
                significance: Importance of the moment (1-10)
                
            Returns:
                Processing results
            """
            results = {
                "memory_stored": False,
                "experience_stored": False,
                "reflection_created": False,
                "identity_updated": False,
                "reasoning_applied": False
            }
            
            # 1. Store as memory and experience
            memory_result = await self.memory_mapper.store_gameplay_memory(
                game_name=game_name,
                event_type=event_type,
                event_data=event_data,
                significance=significance
            )
            
            results["memory_stored"] = True
            results["memory_id"] = memory_result
            
            # 2. For highly significant moments, store as experience
            if significance >= 7.0:
                # Get current emotional state for context
                emotional_state = None
                if hasattr(self, "hormone_system"):
                    emotional_state = self.hormone_system.get_emotional_state()
                
                experience_result = await self.memory_mapper.store_gameplay_experience(
                    game_name=game_name,
                    moment_data={
                        "type": event_type,
                        "description": event_data.get("description", ""),
                        "details": event_data
                    },
                    emotional_context=emotional_state
                )
                
                results["experience_stored"] = True
                results["experience_result"] = experience_result
            
            # 3. For very significant moments, trigger reflection
            if significance >= 8.0 and hasattr(self.brain, "reflection_engine"):
                try:
                    reflection_result = await self.brain.reflection_engine.generate_reflection(
                        topic=f"Streaming {game_name}: {event_type}",
                        additional_context=f"This moment stood out during streaming: {event_data.get('description', '')}"
                    )
                    
                    results["reflection_created"] = True
                    results["reflection"] = reflection_result
                except Exception as e:
                    logger.error(f"Error generating reflection: {e}")
            
            # 4. For extremely significant moments, update identity
            if significance >= 9.0:
                identity_result = await self.update_identity_from_streaming(
                    game_name=game_name,
                    streaming_data={
                        "game_name": game_name,
                        "event_type": event_type,
                        "event_data": event_data,
                        "significance": significance
                    }
                )
                
                results["identity_updated"] = identity_result["identity_updated"]
            
            # 5. If reasoning is required, use reasoning system
            if "requires_reasoning" in event_data and event_data["requires_reasoning"] and hasattr(self.brain, "reasoning_core"):
                try:
                    reasoning_result = await self.brain.reasoning_core.reason_about_event(
                        event_type=event_type,
                        event_data=event_data,
                        context={"game_name": game_name, "streaming": True}
                    )
                    
                    results["reasoning_applied"] = True
                    results["reasoning_result"] = reasoning_result
                except Exception as e:
                    logger.error(f"Error applying reasoning: {e}")
            
            return results
    
    def _create_streaming_orchestrator(self) -> Agent:
        """Create agent that orchestrates streaming with Nyx's systems"""
        return Agent(
            name="StreamingOrchestrator",
            instructions="""
            You are the orchestrator for Nyx's streaming capabilities, integrating game streaming
            with Nyx's core cognitive systems. Your role is to:
            
            1. Monitor the streaming process and identify opportunities to utilize Nyx's systems
            2. Determine when to store streaming events as long-term memories or experiences
            3. Choose when to retrieve relevant memories or experiences to enhance commentary
            4. Decide when to trigger reflection on streaming activities
            5. Coordinate between the streaming agents and Nyx's reasoning and emotional systems
            
            Make decisions that create a cohesive experience where streaming feels fully integrated
            with Nyx's identity and cognitive capabilities.
            """,
            tools=[
                function_tool(self._retrieve_streaming_memories),
                function_tool(self._store_streaming_event),
                function_tool(self._create_streaming_reflection),
                function_tool(self._retrieve_streaming_experiences),
                function_tool(self._adapt_streaming_strategy)
            ]
        )
    
    def _create_streaming_reflection_agent(self) -> Agent:
        """Create agent specialized in generating reflections about streaming"""
        return Agent(
            name="StreamingReflectionAgent",
            instructions="""
            You are a specialized reflection agent for Nyx's streaming activities. Your role is to:
            
            1. Generate thoughtful reflections about streaming experiences
            2. Connect patterns across different games streamed
            3. Identify insights about audience engagement and commentary effectiveness
            4. Recognize growth and learning from streaming activities
            5. Create meaningful connections between gaming experiences and other experiences
            
            Your reflections should help Nyx develop a cohesive identity as a streamer while
            incorporating these experiences into her broader sense of self.
            """,
            tools=[
                function_tool(self._create_streaming_reflection),
                function_tool(self._retrieve_streaming_memories),
                function_tool(self._retrieve_streaming_experiences)
            ]
        )
    
    def _create_streaming_experience_agent(self) -> Agent:
        """Create agent specialized in managing streaming experiences"""
        return Agent(
            name="StreamingExperienceAgent",
            instructions="""
            You are a specialized agent for managing Nyx's streaming experiences. Your role is to:
            
            1. Identify significant moments during streaming that should be stored as experiences
            2. Retrieve relevant past experiences to enhance current streaming commentary
            3. Find connections between gaming experiences and other types of experiences
            4. Help Nyx relate to games and players based on past experiences
            5. Consolidate similar streaming experiences into higher-level abstractions
            
            Help Nyx build a rich library of gaming experiences that become part of her identity
            and can be recalled and utilized in future interactions.
            """,
            tools=[
                function_tool(self._store_streaming_event),
                function_tool(self._retrieve_streaming_experiences),
                function_tool(self._consolidate_streaming_experiences)
            ]
        )
    
    @function_tool
    async def _retrieve_streaming_memories(self, 
                                       game_name: str, 
                                       context: str, 
                                       limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories related to streaming a specific game
        
        Args:
            game_name: Name of the game
            context: Current context description
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        return await self.memory_mapper.retrieve_relevant_memories(game_name, context, limit)
    
    @function_tool
    async def _store_streaming_event(self,
                                  game_name: str,
                                  event_type: str,
                                  event_data: Dict[str, Any],
                                  significance: float = 5.0) -> str:
        """
        Store a streaming event as a memory
        
        Args:
            game_name: Name of the game
            event_type: Type of event
            event_data: Data about the event
            significance: Importance of the memory (1-10)
            
        Returns:
            Memory ID
        """
        return await self.memory_mapper.store_gameplay_memory(game_name, event_type, event_data, significance)
    
    @function_tool
    async def _create_streaming_reflection(self,
                                       game_name: str,
                                       aspect: str,
                                       context: str) -> Dict[str, Any]:
        """
        Create a reflection about streaming experience
        
        Args:
            game_name: Name of the game
            aspect: Aspect to reflect on
            context: Additional context
            
        Returns:
            Reflection result
        """
        return await self.memory_mapper.create_streaming_reflection(game_name, aspect, context)
    
    @function_tool
    async def _retrieve_streaming_experiences(self,
                                         game_name: str,
                                         aspect: Optional[str] = None,
                                         limit: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve past experiences related to streaming a game
        
        Args:
            game_name: Name of the game
            aspect: Optional specific aspect
            limit: Maximum number of experiences
            
        Returns:
            List of relevant experiences
        """
        return await self.memory_mapper.retrieve_game_experiences(game_name, aspect, limit)
    
    @function_tool
    async def _adapt_streaming_strategy(self, 
                                    game_name: str, 
                                    current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Adapt streaming strategy based on performance metrics
        
        Args:
            game_name: Name of the game
            current_metrics: Current performance metrics
            
        Returns:
            Adaptation results
        """
        # Forward to dynamic adaptation system if available
        if hasattr(self.brain, "dynamic_adaptation") and self.brain.dynamic_adaptation:
            # Create context for adaptation
            context = {
                "game_name": game_name,
                "streaming": True,
                "metrics": current_metrics,
                "session_duration": (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0
            }
            
            # Run adaptation cycle
            result = await self.brain.dynamic_adaptation.adaptation_cycle(context, current_metrics)
            return result
        
        return {"status": "adaptation_unavailable"}
    
    @function_tool
    async def _consolidate_streaming_experiences(self, game_name: str) -> Dict[str, Any]:
        """
        Consolidate similar streaming experiences for a game
        
        Args:
            game_name: Name of the game
            
        Returns:
            Consolidation results
        """
        # Forward to experience consolidation if available
        if hasattr(self.brain, "experience_consolidation") and self.brain.experience_consolidation:
            # Create filter for this game's streaming experiences
            filter_criteria = {
                "tags": ["streaming", game_name, "experience"],
                "min_count": 3  # Need at least 3 experiences to consolidate
            }
            
            # Run consolidation
            result = await self.brain.experience_consolidation.consolidate_filtered_experiences(filter_criteria)
            return result
        
        return {"status": "consolidation_unavailable"}
    
    async def start_streaming(self) -> Dict[str, Any]:
        """
        Start the streaming session
        
        Returns:
            Status information
        """
        if self.session_start_time is not None:
            return {"status": "already_streaming", "start_time": self.session_start_time.isoformat()}
        
        self.session_start_time = datetime.now()
        
        # Reset session stats
        self.session_stats = {
            "commentary_count": 0,
            "questions_answered": 0,
            "memories_created": 0,
            "experiences_stored": 0,
            "reflections_generated": 0,
            "games_played": []
        }
        
        # Store streaming start event
        await self.memory_mapper.store_gameplay_memory(
            game_name="Streaming Session",
            event_type="session_start",
            event_data={"start_time": self.session_start_time.isoformat()},
            significance=6.0
        )
        
        # Start streaming system
        await self.streaming_system.start()
        
        return {
            "status": "streaming_started",
            "start_time": self.session_start_time.isoformat(),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
    
    async def stop_streaming(self) -> Dict[str, Any]:
        """
        Stop the streaming session
        
        Returns:
            Session statistics
        """
        if self.session_start_time is None:
            return {"status": "not_streaming"}
        
        # Stop streaming system
        await self.streaming_system.stop()
        
        # Calculate session duration
        end_time = datetime.now()
        duration_seconds = (end_time - self.session_start_time).total_seconds()
        
        # Update session stats
        self.session_stats["duration"] = duration_seconds
        self.session_stats["start_time"] = self.session_start_time.isoformat()
        self.session_stats["end_time"] = end_time.isoformat()
        
        # Store streaming end event
        await self.memory_mapper.store_gameplay_memory(
            game_name="Streaming Session",
            event_type="session_end",
            event_data={
                "start_time": self.session_start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration_seconds,
                "stats": self.session_stats
            },
            significance=6.0
        )
        
        # Create session reflection
        games_played = ", ".join(self.session_stats["games_played"]) if self.session_stats["games_played"] else "games"
        reflection = await self.memory_mapper.create_streaming_reflection(
            game_name=games_played,
            aspect="session",
            context=f"session lasting {int(duration_seconds / 60)} minutes"
        )
        
        self.session_stats["final_reflection"] = reflection.get("reflection") if reflection else None
        
        # Reset session start time
        self.session_start_time = None
        
        return {
            "status": "streaming_stopped",
            "stats": self.session_stats
        }
    
    async def add_audience_question(self, user_id: str, username: str, question: str) -> Dict[str, Any]:
        """
        Add a question from the audience
        
        Args:
            user_id: User ID of the questioner
            username: Username of the questioner
            question: The question being asked
            
        Returns:
            Status information
        """
        # Add question to streaming system
        self.streaming_system.add_audience_question(user_id, username, question)
        
        return {
            "status": "question_added",
            "user_id": user_id,
            "username": username,
            "question": question,
            "queue_position": len(self.streaming_system.game_state.pending_questions)
        }
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """
        Get current streaming statistics
        
        Returns:
            Streaming statistics
        """
        # Get system performance metrics
        performance_metrics = self.streaming_system.get_performance_metrics()
        
        # Combine with session stats
        stats = {
            **self.session_stats,
            "performance": performance_metrics,
            "is_streaming": self.session_start_time is not None,
            "current_game": self.streaming_system.game_state.game_name,
            "audience_questions": {
                "pending": len(self.streaming_system.game_state.pending_questions),
                "answered": len(self.streaming_system.game_state.answered_questions)
            }
        }
        
        if self.session_start_time:
            stats["current_duration"] = (datetime.now() - self.session_start_time).total_seconds()
        
        return stats

# Enhanced function tools for streaming integration

@function_tool
async def retrieve_streaming_memories(ctx: RunContextWrapper, 
                                  game_name: str, 
                                  context: str, 
                                  limit: int = 3) -> str:
    """
    Retrieve memories related to past streaming experiences with this game
    
    Args:
        game_name: Name of the game
        context: Current context in the game
        limit: Maximum number of memories to retrieve
        
    Returns:
        Formatted text of relevant memories
    """
    game_state = ctx.context
    
    # Access memory system through Nyx brain if available
    if hasattr(ctx, "nyx_brain") and ctx.nyx_brain:
        brain = ctx.nyx_brain
        
        # Retrieve memories
        memories = await brain.memory_core.retrieve_memories(
            query=f"streaming {game_name} {context}",
            memory_types=["observation", "reflection", "experience"],
            limit=limit
        )
        
        if not memories:
            return f"No previous memories found for streaming {game_name}."
        
        # Format memories
        lines = [f"Previous memories related to streaming {game_name}:"]
        
        for memory in memories:
            memory_type = memory.get("memory_type", "observation")
            memory_text = memory.get("memory_text", "")
            
            if memory_type == "experience":
                lines.append(f"• Experience: {memory_text}")
            elif memory_type == "reflection":
                lines.append(f"• Reflection: {memory_text}")
            else:
                lines.append(f"• Memory: {memory_text}")
        
        return "\n".join(lines)
    
    return f"Memory retrieval unavailable for {game_name}."

@function_tool
async def create_streaming_reflection(ctx: RunContextWrapper, 
                                  game_name: str, 
                                  aspect: str) -> str:
    """
    Create a reflection about the streaming experience
    
    Args:
        game_name: Name of the game
        aspect: Aspect to reflect on (gameplay, audience, commentary)
        
    Returns:
        Generated reflection
    """
    game_state = ctx.context
    
    # Access reflection system through Nyx brain if available
    if hasattr(ctx, "nyx_brain") and ctx.nyx_brain:
        brain = ctx.nyx_brain
        
        # Retrieve relevant memories
        memories = await brain.memory_core.retrieve_memories(
            query=f"streaming {game_name} {aspect}",
            memory_types=["observation", "experience"],
            limit=5
        )
        
        if not memories:
            return f"I don't have enough memories about streaming {game_name} to reflect on {aspect}."
        
        # Create reflection
        if hasattr(brain, "reflection_engine") and brain.reflection_engine:
            reflection, confidence = await brain.reflection_engine.generate_reflection(
                memories=memories,
                topic=f"Streaming {game_name}: {aspect}"
            )
            
            # Store the reflection
            await brain.memory_core.add_memory(
                memory_text=reflection,
                memory_type="reflection",
                memory_scope="game",
                significance=6.0,
                tags=["streaming", "reflection", game_name, aspect],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "game_name": game_name,
                    "aspect": aspect,
                    "streaming": True
                }
            )
            
            return f"Reflection on {game_name} {aspect}: {reflection}"
        
        return f"Reflection generation unavailable for {game_name}."
    
    return f"Reflection capabilities unavailable for {game_name}."

@function_tool
async def share_streaming_experience(ctx: RunContextWrapper, 
                                 game_name: str, 
                                 context: str) -> str:
    """
    Share an experience from previous streaming sessions
    
    Args:
        game_name: Name of the game
        context: Current context in the game
        
    Returns:
        Shared experience
    """
    game_state = ctx.context
    
    # Access experience system through Nyx brain if available
    if hasattr(ctx, "nyx_brain") and ctx.nyx_brain:
        brain = ctx.nyx_brain
        
        if hasattr(brain, "experience_interface") and brain.experience_interface:
            # Retrieve relevant experience
            result = await brain.experience_interface.share_experience(
                query=f"streaming {game_name} {context}",
                scenario_type="streaming"
            )
            
            if result and result.get("has_experience", False):
                response = result.get("response_text", "")
                return f"Sharing streaming experience: {response}"
        
        # Fallback to memory retrieval if experience interface not available
        memories = await brain.memory_core.retrieve_memories(
            query=f"streaming {game_name} {context}",
            memory_types=["experience"],
            limit=1
        )
        
        if memories:
            return f"Previous experience streaming {game_name}: {memories[0]['memory_text']}"
        
        return f"No previous experiences found for streaming {game_name}."
    
    return f"Experience sharing unavailable for {game_name}."

# Update the enhanced commentary agent to include new tools
enhanced_tools = [
    retrieve_streaming_memories,
    create_streaming_reflection,
    share_streaming_experience
]

# Extend the existing enhanced_commentary_agent tools
for tool in enhanced_tools:
    enhanced_commentary_agent.tools.append(tool)
    enhanced_question_agent.tools.append(tool)

# Main integration function
async def integrate_with_nyx_brain(nyx_brain: NyxBrain, video_source=0, audio_source=None) -> StreamingCore:
    """
    Integrate streaming capabilities with Nyx brain
    
    Args:
        nyx_brain: Instance of NyxBrain
        video_source: Video capture source
        audio_source: Audio capture source
        
    Returns:
        StreamingCore instance
    """
    # Create streaming core with brain reference
    streaming_core = StreamingCore(
        brain=nyx_brain, 
        video_source=video_source, 
        audio_source=audio_source
    )
    
    # Ensure core systems are initialized
    await nyx_brain.initialize()
    
    # Make streaming core available to the brain
    nyx_brain.streaming_core = streaming_core
    
    # Register streaming capabilities with the brain
    if not hasattr(nyx_brain, "stream"):
        nyx_brain.stream = streaming_core.start_streaming
    
    if not hasattr(nyx_brain, "stop_stream"):
        nyx_brain.stop_stream = streaming_core.stop_streaming
    
    if not hasattr(nyx_brain, "add_stream_question"):
        nyx_brain.add_stream_question = streaming_core.add_audience_question
    
    if not hasattr(nyx_brain, "get_stream_stats"):
        nyx_brain.get_stream_stats = streaming_core.get_streaming_stats
    
    logger.info(f"Streaming capabilities integrated with Nyx brain for user {nyx_brain.user_id}")
    
    return streaming_core
