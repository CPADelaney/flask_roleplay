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
            memory_text = f"While streaming {game_name}: {event_data.get('description', json.dumps(event_data))}"
        
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
        if hasattr(self.memory_core, "create_reflection_from_memories"):
            reflection_result = await self.memory_core.create_reflection_from_memories(
                topic=f"Streaming {game_name}: {aspect}", 
                memories=memories
            )
            return reflection_result
        
        # Fallback if create_reflection_from_memories is not available
        try:
            # Use retrieve_reflection if available
            reflection_result = await self.memory_core.retrieve_reflection(
                topic=f"Streaming {game_name}: {aspect}",
                context={
                    "game_name": game_name,
                    "aspect": aspect,
                    "context": context
                }
            )
            return reflection_result
        except Exception as e:
            # Basic fallback
            return {
                "reflection": f"Reflecting on {game_name} {aspect}: Based on my memories, I've enjoyed streaming this game and learned from the experience.",
                "confidence": 0.4
            }

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
    
    async def process_significant_moment(self, 
                                      game_name: str,
                                      event_type: str, 
                                      event_data: Dict[str, Any],
                                      significance: float = 7.0) -> Dict[str, Any]:
        """
        Process a significant gaming moment and store it in memory
        
        Args:
            game_name: Name of the game
            event_type: Type of gameplay event
            event_data: Data about the event
            significance: Importance of the moment (1-10)
            
        Returns:
            Processing results
        """
        results = {}
        
        # 1. Store in memory
        memory_id = await self.memory_mapper.store_gameplay_memory(
            game_name=game_name, 
            event_type=event_type, 
            event_data=event_data, 
            significance=significance
        )
        
        results["memory_id"] = memory_id
        
        # 2. For significant events, store as experience
        if significance >= 7.0 and hasattr(self.brain, "experience_interface"):
            # Format description
            description = event_data.get("description", "")
            if not description and "text" in event_data:
                description = event_data["text"]
                
            experience_text = f"While streaming {game_name}, I experienced: {description}"
            
            # Get emotional context if available
            emotional_context = {}
            if hasattr(self, "hormone_system"):
                emotional_context = self.hormone_system.get_emotional_state()
                
            # Store in experience system
            try:
                experience_result = await self.brain.experience_interface.store_experience(
                    text=experience_text,
                    scenario_type="gaming",
                    entities=[game_name],
                    emotional_context=emotional_context,
                    significance=significance,
                    tags=["streaming", game_name, event_type],
                    user_id=str(self.brain.user_id)
                )
                
                results["experience_id"] = experience_result.get("id")
                results["experience_stored"] = True
            except Exception as e:
                logger.error(f"Error storing in experience interface: {e}")
        
        # 3. For very significant events, use reasoning system
        if significance >= 8.0 and hasattr(self.brain, "reasoning_core"):
            try:
                reasoning_result = await self.brain.reasoning_core.analyze_event(
                    event_data=event_data, 
                    context={
                        "context": "streaming", 
                        "game_name": game_name
                    }
                )
                
                results["reasoning_result"] = reasoning_result
                results["reasoning_applied"] = True
            except Exception as e:
                logger.error(f"Error applying reasoning: {e}")
        
        # Update memory stats
        self.session_stats["memories_created"] += 1
        if results.get("experience_stored", False):
            self.session_stats["experiences_stored"] += 1
        
        return results
    
    async def recall_streaming_experience(self, 
                                       query: str, 
                                       game_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Recall streaming experience using both memory and experience systems
        
        Args:
            query: Search query
            game_name: Optional specific game to search for
            
        Returns:
            Recalled experiences
        """
        # Format full query
        full_query = f"streaming {query}"
        if game_name:
            full_query = f"streaming {game_name} {query}"
        
        # Try experience system first if available
        if hasattr(self.brain, "experience_interface"):
            try:
                experience_result = await self.brain.experience_interface.recall_experience(
                    query=full_query,
                    scenario_type="gaming",
                    confidence_threshold=0.6
                )
                
                if experience_result and experience_result.get("has_experience", False):
                    return {
                        "text": experience_result.get("text", ""),
                        "confidence": experience_result.get("confidence", 0.0),
                        "source": "experience_interface",
                        "has_experience": True
                    }
            except Exception as e:
                logger.error(f"Error recalling from experience interface: {e}")
        
        # Fall back to memory system
        try:
            memories = await self.brain.memory_core.retrieve_memories(
                query=full_query,
                memory_types=["experience", "observation"],
                limit=3,
                min_significance=5.0
            )
            
            if memories:
                return {
                    "memories": memories,
                    "text": memories[0]["memory_text"],
                    "confidence": memories[0].get("relevance", 0.7),
                    "source": "memory_core",
                    "has_experience": True
                }
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
        
        return {
            "text": "",
            "confidence": 0.0,
            "has_experience": False
        }
    
    async def update_identity_from_streaming(self, 
                                          game_name: str,
                                          streaming_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update identity based on streaming experiences
        
        Args:
            game_name: Name of the game
            streaming_data: Data about streaming
            
        Returns:
            Identity update results
        """
        if not hasattr(self.brain, "identity_evolution"):
            return {
                "updated": False,
                "reason": "identity_evolution_unavailable"
            }
        
        # Extract identity impacts
        preferences = {}
        traits = {}
        
        # Game preferences
        if "game_genre" in streaming_data and streaming_data["game_genre"]:
            genre = streaming_data["game_genre"]
            preferences["genre_preferences"] = {
                genre: 0.1  # Small increase in preference for this genre
            }
        
        # Streaming activity preference
        if "session_duration" in streaming_data and streaming_data["session_duration"] > 1800:  # 30+ minutes
            preferences["activity_preferences"] = {
                "streaming": 0.1  # Small increase in streaming preference
            }
        
        # Commentary style traits
        if "commentary_style" in streaming_data:
            style = streaming_data["commentary_style"]
            if style == "analytical":
                traits["analytical"] = 0.1
            elif style == "humorous":
                traits["humorous"] = 0.1
            elif style == "educational":
                traits["informative"] = 0.1
        
        # Impact based on audience interaction
        if streaming_data.get("questions_answered", 0) > 5:
            traits["helpful"] = 0.1
        
        # Create impact
        impact = {
            "preferences": preferences,
            "traits": traits
        }
        
        # Only update if we have meaningful impacts
        if preferences or traits:
            try:
                result = await self.brain.identity_evolution.update_identity_from_experience(
                    experience={
                        "type": "streaming",
                        "game_name": game_name,
                        "streaming_data": streaming_data
                    },
                    impact=impact
                )
                
                return {
                    "updated": True,
                    "preferences_updated": list(preferences.keys()),
                    "traits_updated": list(traits.keys())
                }
            except Exception as e:
                logger.error(f"Error updating identity: {e}")
                return {"error": str(e)}
        
        return {"updated": False, "reason": "no_significant_impact"}
    
    async def reason_about_streaming_event(self, 
                                        event_data: Dict[str, Any], 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply reasoning to streaming events
        
        Args:
            event_data: Data about the event
            context: Additional context
            
        Returns:
            Reasoning results
        """
        if not hasattr(self.brain, "reasoning_core"):
            return {
                "reasoned": False,
                "reason": "reasoning_core_unavailable"
            }
        
        try:
            result = await self.brain.reasoning_core.analyze_event(
                event_data=event_data,
                context=context
            )
            
            # Store reasoning as memory if significant
            if result and result.get("significance", 0) >= 6.0:
                game_name = context.get("game_name", "Unknown Game")
                event_type = context.get("event_type", "event")
                
                memory_text = f"While streaming {game_name}, I reasoned about {event_type}: {result.get('conclusion', '')}"
                
                memory_id = await self.brain.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="reflection",
                    memory_scope="game",
                    significance=result.get("significance", 6.0),
                    tags=["streaming", "reasoning", game_name, event_type],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "game_name": game_name,
                        "event_type": event_type,
                        "reasoning_process": result,
                        "streaming": True
                    }
                )
                
                result["memory_id"] = memory_id
            
            return {
                "reasoned": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error applying reasoning to streaming event: {e}")
            return {
                "reasoned": False,
                "error": str(e)
            }

    def is_streaming(self) -> bool:
        """Check if a streaming session is active"""
        return self.session_start_time is not None

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

# Add these tools to the existing agents
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
    Integrate streaming capabilities with Nyx brain using optimized streaming core
    
    Args:
        nyx_brain: Instance of NyxBrain
        video_source: Video capture source
        audio_source: Audio capture source
        
    Returns:
        StreamingCore instance
    """
    # Create optimized streaming core with brain reference
    streaming_core = OptimizedStreamingCore(
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
    
    # Register experience access
    if not hasattr(nyx_brain, "retrieve_streaming_experience"):
        nyx_brain.retrieve_streaming_experience = streaming_core.recall_streaming_experience
    
    # Register memory creation
    if not hasattr(nyx_brain, "create_streaming_memory"):
        nyx_brain.create_streaming_memory = streaming_core.memory_mapper.store_gameplay_memory
    
    # Register reflection creation
    if not hasattr(nyx_brain, "create_streaming_reflection"):
        nyx_brain.create_streaming_reflection = streaming_core.memory_mapper.create_streaming_reflection
    
    # Register reasoning if available
    if hasattr(streaming_core, "reason_about_streaming_event"):
        nyx_brain.reason_about_streaming_event = streaming_core.reason_about_streaming_event

    nyx_brain.process_frame_optimized = streaming_core.process_frame_optimized
    nyx_brain.get_performance_metrics = streaming_core.get_performance_metrics
    
    logger.info(f"Streaming capabilities integrated with Nyx brain for user {nyx_brain.user_id}")
    
    return streaming_core

# Add to nyx/streamer/integration.py
class CrossGameKnowledgeIntegration:
    """Deep integration between streaming and cross-game knowledge systems"""
    
    @staticmethod
    async def integrate(brain, streaming_core):
        """
        Integrate cross-game knowledge with streaming capabilities
        
        Args:
            brain: NyxBrain instance
            streaming_core: StreamingCore instance
            
        Returns:
            Integration status
        """
        from nyx.streamer.cross_game_knowledge import EnhancedCrossGameKnowledgeSystem
        
        # Create cross-game knowledge system if not exists
        if not hasattr(streaming_core, "cross_game_knowledge"):
            streaming_core.cross_game_knowledge = EnhancedCrossGameKnowledgeSystem()
            
            # Initialize with some data
            streaming_core.cross_game_knowledge.seed_initial_knowledge()
        
        # Add real-time cross-game insight generation during gameplay
        original_process = streaming_core.streaming_system._process_game_frame
        
        async def knowledge_enhanced_frame():
            # Run original processing
            await original_process()
            
            # Generate cross-game insights periodically
            game_state = streaming_core.streaming_system.game_state
            if game_state.game_id and game_state.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                # Get current game info
                game_name = game_state.game_name
                current_context = ""
                
                if game_state.current_location:
                    current_context += f"in {game_state.current_location.get('name', '')}"
                if game_state.detected_action:
                    current_context += f" while {game_state.detected_action.get('name', '')}"
                
                # Get insights from cross-game knowledge
                insights = streaming_core.cross_game_knowledge.get_applicable_insights(
                    target_game=game_name,
                    context=current_context
                )
                
                # Process relevant insights
                if insights:
                    for insight in insights[:1]:  # Use top insight
                        # Add to game state for commentary
                        game_state.add_event(
                            "cross_game_insight", 
                            {
                                "source_game": insight.get("source_game_name", insight.get("source_game", "")),
                                "target_game": game_name,
                                "mechanic": insight.get("mechanic_name", insight.get("mechanic", "")),
                                "content": insight.get("content", insight.get("insight", "")),
                                "relevance": insight.get("relevance", 0.7)
                            }
                        )
                        
                        # Store in memory system if available
                        if hasattr(streaming_core, "memory_mapper"):
                            await streaming_core.memory_mapper.store_gameplay_memory(
                                game_name=game_name,
                                event_type="cross_game_insight",
                                event_data={
                                    "source_game": insight.get("source_game_name", insight.get("source_game", "")),
                                    "content": insight.get("content", insight.get("insight", "")),
                                    "relevance": insight.get("relevance", 0.7)
                                },
                                significance=7.0
                            )
                        
                        # Add to brain's experiences if available
                        if hasattr(brain, "experience_interface"):
                            try:
                                await brain.experience_interface.store_experience(
                                    text=f"While streaming {game_name}, I made a connection to {insight.get('source_game_name', insight.get('source_game', ''))}: {insight.get('content', insight.get('insight', ''))}",
                                    scenario_type="analysis",
                                    entities=[game_name, insight.get("source_game_name", insight.get("source_game", ""))],
                                    significance=7.0,
                                    tags=["streaming", "cross_game_insight", game_name],
                                    user_id=str(brain.user_id)
                                )
                            except Exception as e:
                                logger.error(f"Error storing cross-game insight in experience system: {e}")
        
        # Replace frame processing method
        streaming_core.streaming_system._process_game_frame = knowledge_enhanced_frame
        
        # Add methods to streaming core
        streaming_core.get_cross_game_insights = streaming_core.cross_game_knowledge.get_applicable_insights
        streaming_core.generate_game_insight = streaming_core.cross_game_knowledge.generate_insight
        streaming_core.discover_game_patterns = streaming_core.cross_game_knowledge.discover_patterns
        
        # Add periodic knowledge consolidation
        async def run_knowledge_consolidation():
            result = await streaming_core.cross_game_knowledge.consolidate_knowledge()
            
            # Store consolidation as reflection if memory system available
            if hasattr(streaming_core, "memory_mapper"):
                combined_games = ", ".join(streaming_core.streaming_system.game_state.session_stats.get("games_played", []))
                if combined_games:
                    await streaming_core.memory_mapper.create_streaming_reflection(
                        game_name=combined_games,
                        aspect="cross_game_knowledge",
                        context="consolidation"
                    )
            
            return result
        
        streaming_core.run_knowledge_consolidation = run_knowledge_consolidation
        
        return {
            "status": "integrated",
            "components": {
                "cross_game_knowledge": True,
                "real_time_insights": True,
                "knowledge_consolidation": True
            }
        }

# Enhance setup_enhanced_streaming to include the new integration
async def setup_enhanced_streaming(brain: NyxBrain, 
                                video_source=0, 
                                audio_source=None) -> StreamingCore:
    """
    Set up deeply integrated streaming system utilizing all of Nyx's cognitive systems
    
    Args:
        brain: NyxBrain instance
        video_source: Video source
        audio_source: Audio source
        
    Returns:
        Fully integrated StreamingCore
    """
    # 1. First create the base streaming core
    streaming_core = await integrate_with_nyx_brain(brain, video_source, audio_source)
    
    # 2. Add hormone system integration
    await StreamingHormoneIntegration.integrate(brain, streaming_core)
    
    # 3. Add reflection engine integration
    await StreamingIntegration.integrate(brain, streaming_core)
    
    # 4. Add cross-game knowledge integration - NEW!
    await CrossGameKnowledgeIntegration.integrate(brain, streaming_core)
    
    # 5. Enhance with reasoning system integration
    streaming_core = await enhance_with_reasoning(brain, streaming_core)
    
    # 6. Enhance with experience system integration
    streaming_core = await enhance_with_experience(brain, streaming_core)
    
    # 7. Connect to meta-cognitive system for ongoing integration
    streaming_core = await connect_to_metacognition(brain, streaming_core)
    
    # 8. Set up bi-directional identity influence
    streaming_core = await setup_identity_integration(brain, streaming_core)
    
    # 9. Enable periodic cross-system tasks
    task = asyncio.create_task(_run_enhanced_periodic_tasks(brain, streaming_core))
    streaming_core._integrated_task = task
    
    # 10. Register functions in the brain
    _register_brain_functions(brain, streaming_core)
    
    return streaming_core
class OptimizedStreamingCore(StreamingCore):
    """
    Performance-optimized streaming core with better asynchronous processing
    and resource management for real-time performance.
    """
    
    def __init__(self, brain: NyxBrain, video_source=0, audio_source=None):
        """Initialize the optimized streaming core"""
        super().__init__(brain, video_source, audio_source)
        
        # Processing optimizations
        self.frame_buffer = deque(maxlen=5)  # Buffer recent frames
        self.parallel_processing = True  # Enable parallel processing
        self.skip_frames = 2  # Process every nth frame for heavy operations
        self.priority_queue = asyncio.PriorityQueue()  # Queue for prioritized tasks
        self.processing_tasks = set()  # Track active tasks
        self.resource_monitor = ResourceMonitor()  # Monitor system resources
        
        # Performance metrics
        self.processing_times = {
            "visual": deque(maxlen=50),
            "audio": deque(maxlen=50),
            "speech": deque(maxlen=50),
            "memory": deque(maxlen=50),
            "commentary": deque(maxlen=50),
            "total": deque(maxlen=50)
        }
        
        logger.info("Initialized OptimizedStreamingCore with performance enhancements")
    
    async def process_frame_optimized(self):
        """
        Process a frame with optimized resource usage
        
        Returns:
            Processing results
        """
        if not self.streaming_system or not self.streaming_system.game_state:
            return {"status": "not_initialized"}
        
        # Get current frame
        frame = self.streaming_system.game_state.current_frame
        frame_count = self.streaming_system.game_state.frame_count
        
        if frame is None:
            return {"status": "no_frame_available"}
        
        # Store in buffer
        self.frame_buffer.append(frame)
        
        # Skip frames for heavy processing based on current load
        resource_usage = self.resource_monitor.get_usage()
        high_load = resource_usage.get("cpu", 0) > 70 or resource_usage.get("memory", 0) > 70
        
        # Adjust skip rate based on resource usage
        skip_rate = self.skip_frames * 2 if high_load else self.skip_frames
        
        # Check if this frame should be processed
        if frame_count % skip_rate != 0:
            return {"status": "frame_skipped", "reason": "resource_management"}
        
        start_time = time.time()
        tasks = []
        results = {}
        
        try:
            # Create context
            extended_context = RunContextWrapper(context=self.streaming_system.game_state)
            
            if hasattr(self.streaming_system, "audio_processor"):
                extended_context.audio_processor = self.streaming_system.audio_processor
            
            if hasattr(self.streaming_system, "speech_recognition"):
                extended_context.speech_recognition = self.streaming_system.speech_recognition
            
            if hasattr(self, "cross_game_knowledge"):
                extended_context.cross_game_knowledge = self.cross_game_knowledge
            
            # Process in parallel with prioritization
            if self.parallel_processing:
                # Always identify game first if needed
                if not self.streaming_system.game_state.game_id:
                    game_start = time.time()
                    game_result = await identify_game(extended_context)
                    self.processing_times["visual"].append(time.time() - game_start)
                    results["game_identification"] = game_result
                
                # Create prioritized tasks
                priority_tasks = [
                    (1, analyze_current_frame(extended_context)),  # High priority
                    (3, analyze_speech(extended_context)),          # Medium priority
                    (5, get_player_location(extended_context))      # Lower priority
                ]
                
                # Process every task by priority
                for priority, coro in sorted(priority_tasks, key=lambda x: x[0]):
                    try:
                        # Use asyncio.shield to prevent cancellation
                        task_start = time.time()
                        result = await asyncio.shield(coro)
                        task_time = time.time() - task_start
                        
                        # Store timing
                        if priority == 1:
                            self.processing_times["visual"].append(task_time)
                        elif priority == 3:
                            self.processing_times["speech"].append(task_time)
                        
                        # Store result
                        results[f"task_{priority}"] = result
                    except Exception as e:
                        logger.error(f"Error in task with priority {priority}: {e}")
            else:
                # Sequential processing for resource-constrained systems
                visual_start = time.time()
                results["visual"] = await analyze_current_frame(extended_context)
                self.processing_times["visual"].append(time.time() - visual_start)
                
                speech_start = time.time()
                results["speech"] = await analyze_speech(extended_context)
                self.processing_times["speech"].append(time.time() - speech_start)
            
            # Always process memory asynchronously
            if frame_count % (skip_rate * 2) == 0:
                memory_start = time.time()
                # Create a lightweight task for memory operations
                asyncio.create_task(self._process_memory_async(extended_context))
                self.processing_times["memory"].append(time.time() - memory_start)
            
            # Check if it's time for commentary
            if self._should_generate_commentary():
                commentary_start = time.time()
                # Process commentary in a separate task to not block the main loop
                task = asyncio.create_task(self._generate_commentary_async(extended_context))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.remove)
                self.processing_times["commentary"].append(time.time() - commentary_start)
        
        except Exception as e:
            logger.error(f"Error in optimized frame processing: {e}")
            results["error"] = str(e)
        
        finally:
            # Calculate total processing time
            total_time = time.time() - start_time
            self.processing_times["total"].append(total_time)
            
            results["processing_time"] = total_time
            results["frame_count"] = frame_count
            
            return results

    async def summarize_session_learnings(self) -> Dict[str, Any]:
        """
        Summarize what was learned during the streaming session
        
        Returns:
            Learning summary
        """
        if not hasattr(self, "learning_manager"):
            return {
                "status": "learning_manager_unavailable",
                "summary": "Learning analysis system not available."
            }
        
        # Get session data
        session_data = {
            "game_name": self.streaming_system.game_state.game_name,
            "recent_events": list(self.streaming_system.game_state.recent_events),
            "dialog_history": self.streaming_system.game_state.dialog_history,
            "answered_questions": list(self.streaming_system.game_state.answered_questions),
            "transferred_insights": self.streaming_system.game_state.transferred_insights,
            "duration": (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0
        }
        
        # Analyze session learnings
        analysis_result = await self.learning_manager.analyze_session_learnings(session_data)
        
        # Generate final learning summary
        summary_result = await self.learning_manager.generate_learning_summary()
        
        # Add functionality needs assessment
        needs_assessment = await self.learning_manager.assess_functionality_needs(
            {"category_counts": summary_result.get("categories", {}), "total_learnings": summary_result.get("total_learnings", 0)}
        )
        summary_result["functionality_needs"] = needs_assessment
        
        return summary_result
    
    async def stop_streaming(self) -> Dict[str, Any]:
        """
        Stop the streaming session
        
        Returns:
            Session statistics
        """
        if self.session_start_time is None:
            return {"status": "not_streaming"}
        
        # Generate learning summary
        learning_summary = None
        if hasattr(self, "learning_manager"):
            try:
                learning_summary = await self.summarize_session_learnings()
            except Exception as e:
                logger.error(f"Error generating learning summary: {e}")
        
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
                "stats": self.session_stats,
                "learning_summary": learning_summary
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
            "stats": self.session_stats,
            "learning_summary": learning_summary
        }
    
    async def _process_memory_async(self, context):
        """Process memory operations asynchronously"""
        try:
            # Retrieve relevant memories
            game_name = context.context.game_name
            if not game_name:
                return
            
            current_context = ""
            if context.context.current_location:
                current_context += f"in {context.context.current_location.get('name', '')}"
            if context.context.detected_action:
                current_context += f" while {context.context.detected_action.get('name', '')}"
            
            # Use memory mapper
            await self.memory_mapper.retrieve_relevant_memories(
                game_name=game_name,
                context=current_context,
                limit=3
            )
        except Exception as e:
            logger.error(f"Error in async memory processing: {e}")
    
    async def _generate_commentary_async(self, context):
        """Generate commentary asynchronously"""
        try:
            # Determine priority source based on recent events
            priority_source = self._determine_priority_source(context.context)
            
            # Generate commentary
            await self.streaming_system._generate_commentary(context, priority_source)
        except Exception as e:
            logger.error(f"Error in async commentary generation: {e}")
    
    def _should_generate_commentary(self):
        """Determine if it's time to generate commentary"""
        if not hasattr(self, "last_commentary_time"):
            self.last_commentary_time = 0
        
        current_time = time.time()
        time_since_last = current_time - self.last_commentary_time
        
        # Adjust commentary frequency based on system load
        resource_usage = self.resource_monitor.get_usage()
        high_load = resource_usage.get("cpu", 0) > 70
        
        base_cooldown = self.streaming_system.commentary_cooldown if hasattr(self.streaming_system, "commentary_cooldown") else 5.0
        adjusted_cooldown = base_cooldown * 1.5 if high_load else base_cooldown
        
        return time_since_last >= adjusted_cooldown
    
    def _determine_priority_source(self, game_state):
        """Determine the priority source for commentary"""
        # Check recent events
        if hasattr(game_state, "recent_events") and game_state.recent_events:
            for event in reversed(game_state.recent_events):
                if event["type"] == "character_dialog":
                    return "speech"
                elif event["type"] == "gameplay_event" and event["data"].get("significance", 0) >= 7.0:
                    return "visual"
                elif event["type"] == "cross_game_insight":
                    return "cross_game"
        
        # Default to visual
        return "visual"
    
    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        metrics = {
            "average_times": {
                "visual": sum(self.processing_times["visual"]) / max(len(self.processing_times["visual"]), 1),
                "speech": sum(self.processing_times["speech"]) / max(len(self.processing_times["speech"]), 1),
                "memory": sum(self.processing_times["memory"]) / max(len(self.processing_times["memory"]), 1),
                "commentary": sum(self.processing_times["commentary"]) / max(len(self.processing_times["commentary"]), 1),
                "total": sum(self.processing_times["total"]) / max(len(self.processing_times["total"]), 1)
            },
            "resource_usage": self.resource_monitor.get_usage(),
            "frame_buffer_size": len(self.frame_buffer),
            "active_tasks": len(self.processing_tasks),
            "fps": 1.0 / max(sum(self.processing_times["total"]) / max(len(self.processing_times["total"]), 1), 0.001)
        }
        
        # Add base stats
        base_stats = super().get_streaming_stats()
        metrics.update(base_stats)
        
        return metrics

class ResourceMonitor:
    """Monitor system resources for adaptive processing"""
    
    def __init__(self):
        """Initialize the resource monitor"""
        self.last_check_time = 0
        self.cached_usage = {
            "cpu": 0,
            "memory": 0,
            "gpu": 0
        }
        self.cache_duration = 5  # seconds
    
    def get_usage(self):
        """Get current resource usage"""
        current_time = time.time()
        
        # Use cached values if recent
        if current_time - self.last_check_time < self.cache_duration:
            return self.cached_usage
        
        try:
            # CPU usage
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # GPU usage (if available)
            gpu_usage = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass
            
            # Update cache
            self.cached_usage = {
                "cpu": cpu_usage,
                "memory": memory_usage,
                "gpu": gpu_usage
            }
            
            self.last_check_time = current_time
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
        
        return self.cached_usage

# Add to nyx/streamer/nyx_streaming_core.py

