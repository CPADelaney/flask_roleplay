# nyx/core/nyx_brain.py

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.emotional_core import EmotionalCore
from nyx.core.memory_core import MemoryCore
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.experience_interface import ExperienceInterface

logger = logging.getLogger(__name__)

class NyxBrain:
    """
    Central integration point for all Nyx systems.
    Handles cross-component communication and provides a unified API.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core components
        self.emotional_core = EmotionalCore()
        self.memory_core = None  # Initialized in initialize()
        self.reflection_engine = ReflectionEngine()
        self.experience_interface = None  # Initialized in initialize()
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.now()
        self.interaction_count = 0
        
        # Bidirectional influence settings
        self.memory_to_emotion_influence = 0.3  # How much memories influence emotions
        self.emotion_to_memory_influence = 0.4  # How much emotions influence memory retrieval
        
        # Performance monitoring
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "response_times": []
        }
        
        # Caching
        self.context_cache = {}
        
        # Singleton registry
        self._instance_count = 0
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxBrain':
        """Get or create a singleton instance for the specified user and conversation"""
        # Use a key for the specific user/conversation
        key = f"brain_{user_id}_{conversation_id}"
        
        # Check if instance exists in a global registry
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
        logger.info(f"Initializing NyxBrain {self._instance_count} for user {self.user_id}")
        
        # Initialize memory system
        self.memory_core = MemoryCore(self.user_id, self.conversation_id)
        await self.memory_core.initialize()
        
        # Initialize experience interface with memory core
        self.experience_interface = ExperienceInterface(self.memory_core)
        
        # Initialize emotional core with default state
        # No async initialization needed for emotional_core
        
        # Reflection engine doesn't need async initialization
        
        self.initialized = True
        logger.info(f"NyxBrain initialized for user {self.user_id}")
    
    async def process_input(self, 
                          user_input: str, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input and update all systems.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        # Update interaction tracking
        self.last_interaction = datetime.now()
        self.interaction_count += 1
        
        # Initialize context
        context = context or {}
        
        # Process emotional impact of input
        emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
        emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
        
        # Add emotional state to context for memory retrieval
        context["emotional_state"] = emotional_state
        
        # Retrieve relevant memories
        memories = await self.memory_core.retrieve_memories(
            query=user_input,
            context=context
        )
        self.performance_metrics["memory_operations"] += 1
        
        # Update emotional state based on retrieved memories
        if memories:
            memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
            # Apply memory-to-emotion influence
            for emotion, value in memory_emotional_impact.items():
                self.emotional_core.update_emotion(emotion, value * self.memory_to_emotion_influence)
            
            # Get updated emotional state
            emotional_state = self.emotional_core.get_emotional_state()
        
        self.performance_metrics["emotion_updates"] += 1
        
        # Check if experience sharing is requested
        should_share_experience = self._should_share_experience(user_input, context)
        experience_result = None
        
        if should_share_experience:
            # Retrieve and format experience
            experience_result = await self.experience_interface.handle_experience_sharing_request(
                user_query=user_input,
                context_data=context
            )
            self.performance_metrics["experiences_shared"] += 1
        
        # Add memory of this interaction
        memory_text = f"User said: {user_input}"
        
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=5,
            tags=["interaction", "user_input"],
            metadata={
                "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Calculate response time
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        self.performance_metrics["response_times"].append(response_time)
        
        # Return processing results
        return {
            "user_input": user_input,
            "emotional_state": emotional_state,
            "memories": memories,
            "memory_count": len(memories),
            "has_experience": experience_result["has_experience"] if experience_result else False,
            "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
            "memory_id": memory_id,
            "response_time": response_time
        }
    
    async def generate_response(self, 
                             user_input: str, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data including main message and supporting information
        """
        # Process the input first
        processing_result = await self.process_input(user_input, context)
        
        # Determine if experience response should be used
        if processing_result["has_experience"]:
            main_response = processing_result["experience_response"]
            response_type = "experience"
        else:
            # No specific experience to share, generate standard response
            # In a real implementation, this would call an LLM or other response generation system
            main_response = "I acknowledge your message."
            response_type = "standard"
        
        # Determine if emotion should be expressed
        should_express_emotion = self.emotional_core.should_express_emotion()
        emotional_expression = None
        
        if should_express_emotion:
            emotional_expression = self.emotional_core.get_expression_for_emotion()
        
        # Package the response
        response_data = {
            "message": main_response,
            "response_type": response_type,
            "emotional_state": processing_result["emotional_state"],
            "emotional_expression": emotional_expression,
            "memories_used": [m["id"] for m in processing_result["memories"]],
            "memory_count": processing_result["memory_count"]
        }
        
        # Add memory of this response
        await self.memory_core.add_memory(
            memory_text=f"I responded: {main_response}",
            memory_type="observation",
            memory_scope="game",
            significance=5,
            tags=["interaction", "nyx_response"],
            metadata={
                "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                "timestamp": datetime.now().isoformat(),
                "response_type": response_type
            }
        )
        
        return response_data
    
    async def create_reflection(self, 
                             topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reflection on memories.
        
        Args:
            topic: Optional topic to focus reflection on
            
        Returns:
            Reflection data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's reflection creation
        reflection_result = await self.memory_core.create_reflection_from_memories(topic=topic)
        self.performance_metrics["reflections_generated"] += 1
        
        return reflection_result
    
    async def create_abstraction(self,
                              memory_ids: List[str],
                              pattern_type: str = "behavior") -> Dict[str, Any]:
        """
        Create a higher-level abstraction from specific memories.
        
        Args:
            memory_ids: IDs of memories to abstract from
            pattern_type: Type of pattern to identify
            
        Returns:
            Abstraction data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's abstraction creation
        return await self.memory_core.create_abstraction_from_memories(
            memory_ids=memory_ids,
            pattern_type=pattern_type
        )
    
    async def retrieve_experiences(self,
                                query: str,
                                scenario_type: Optional[str] = None,
                                limit: int = 3) -> Dict[str, Any]:
        """
        Retrieve experiences relevant to a query.
        
        Args:
            query: Search query
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of experiences to return
            
        Returns:
            Experience retrieval results
        """
        if not self.initialized:
            await self.initialize()
        
        # Use experience interface for retrieving experiences
        experiences = await self.experience_interface.retrieve_experiences_enhanced(
            query=query,
            scenario_type=scenario_type,
            limit=limit
        )
        
        return {
            "experiences": experiences,
            "count": len(experiences),
            "query": query,
            "scenario_type": scenario_type
        }
    
    async def construct_narrative(self,
                               topic: str,
                               chronological: bool = True,
                               limit: int = 5) -> Dict[str, Any]:
        """
        Construct a narrative from memories about a topic.
        
        Args:
            topic: Topic for narrative
            chronological: Whether to maintain chronological order
            limit: Maximum number of memories to include
            
        Returns:
            Narrative data
        """
        if not self.initialized:
            await self.initialize()
        
        # Use memory core's narrative construction
        return await self.memory_core.construct_narrative_from_memories(
            topic=topic,
            chronological=chronological,
            limit=limit
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all subsystems"""
        if not self.initialized:
            await self.initialize()
        
        # Run memory maintenance
        memory_result = await self.memory_core.run_maintenance()
        
        # Perform additional maintenance as needed
        # (other components don't need routine maintenance)
        
        return {
            "memory_maintenance": memory_result,
            "maintenance_time": datetime.now().isoformat()
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about all systems"""
        if not self.initialized:
            await self.initialize()
        
        # Get memory stats
        memory_stats = await self.memory_core.get_memory_stats()
        
        # Get emotional state
        emotional_state = self.emotional_core.get_emotional_state()
        dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
        
        # Get performance metrics
        avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
        
        # Generate introspection text
        introspection = await self.reflection_engine.generate_introspection(
            memory_stats=memory_stats,
            player_model=None  # Player model would be provided in real implementation
        )
        
        return {
            "memory_stats": memory_stats,
            "emotional_state": {
                "emotions": emotional_state,
                "dominant_emotion": dominant_emotion,
                "dominant_value": dominant_value,
                "valence": self.emotional_core.get_emotional_valence(),
                "arousal": self.emotional_core.get_emotional_arousal()
            },
            "interaction_stats": {
                "interaction_count": self.interaction_count,
                "last_interaction": self.last_interaction.isoformat()
            },
            "performance_metrics": {
                "memory_operations": self.performance_metrics["memory_operations"],
                "emotion_updates": self.performance_metrics["emotion_updates"],
                "reflections_generated": self.performance_metrics["reflections_generated"],
                "experiences_shared": self.performance_metrics["experiences_shared"],
                "avg_response_time": avg_response_time
            },
            "introspection": introspection
        }
    
    def _should_share_experience(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Determine if we should share an experience based on input and context"""
        # Check for explicit experience requests
        explicit_request = any(phrase in user_input.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience"])
        
        if explicit_request:
            return True
        
        # Check if it's a question that could benefit from experience sharing
        is_question = user_input.endswith("?") or user_input.lower().startswith(("what", "how", "when", "where", "why", "who", "can", "could", "do", "did"))
        
        if is_question and "share_experiences" in context and context["share_experiences"]:
            return True
        
        # Default to not sharing experiences unless explicitly requested
        return False
    
    async def _calculate_memory_emotional_impact(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate emotional impact from relevant memories"""
        impact = {}
        
        for memory in memories:
            # Extract emotional context
            emotional_context = memory.get("metadata", {}).get("emotional_context", {})
            
            if not emotional_context:
                continue
                
            # Get primary emotion
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if primary_emotion:
                # Calculate impact based on relevance and recency
                relevance = memory.get("relevance", 0.5)
                
                # Get timestamp if available
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                recency_factor = 1.0
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    days_old = (datetime.now() - timestamp).days
                    recency_factor = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days, minimum 0.5
                
                # Calculate final impact value
                impact_value = primary_intensity * relevance * recency_factor * 0.1
                
                # Add to impact dict
                if primary_emotion not in impact:
                    impact[primary_emotion] = 0
                impact[primary_emotion] += impact_value
        
        return impact
    
    # Enhanced system function implementations
    
    async def process_user_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced processing of user input with comprehensive results.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Comprehensive processing results
        """
        # Use existing process_input but enhance the result
        result = await self.process_input(user_input, context)
        
        # Add additional processing information
        system_stats = await self.get_system_stats()
        
        # Return enhanced result
        return {
            "input": user_input,
            "emotional_state": result["emotional_state"],
            "memories": result["memories"],
            "memory_count": result["memory_count"],
            "has_experience": result["has_experience"],
            "experience_response": result["experience_response"],
            "memory_id": result["memory_id"],
            "response_time": result["response_time"],
            "system_stats": {
                "memory_stats": system_stats["memory_stats"],
                "emotional_state": system_stats["emotional_state"],
                "performance_metrics": system_stats["performance_metrics"]
            }
        }
    
    async def generate_enhanced_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an enhanced response to user input with adaptation.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Enhanced response data with adaptation
        """
        # Generate standard response
        response_data = await self.generate_response(user_input, context)
        
        # Add adaptive behavior from dynamic adaptation system
        if hasattr(self, 'dynamic_adaptation'):
            # Create adaptable context
            adaptable_context = {
                "user_input": user_input,
                "response": response_data["message"],
                "interaction_type": context.get("interaction_type", "general") if context else "general",
            }
            
            # Detect context change
            change_result = await self.dynamic_adaptation.detect_context_change(adaptable_context)
            
            # Monitor performance
            performance = await self.dynamic_adaptation.monitor_performance({
                "success_rate": context.get("success_rate", 0.5) if context else 0.5,
                "user_satisfaction": context.get("user_satisfaction", 0.5) if context else 0.5,
                "efficiency": context.get("efficiency", 0.5) if context else 0.5,
                "response_quality": context.get("response_quality", 0.5) if context else 0.5
            })
            
            # Add adaptation data to response
            response_data["adaptation"] = {
                "context_change": change_result,
                "performance": performance
            }
            
            # If significant change, select strategy
            if change_result[0]:  # significant change
                strategy = await self.dynamic_adaptation.select_strategy(adaptable_context, performance)
                response_data["adaptation"]["strategy"] = strategy
        
        return response_data
