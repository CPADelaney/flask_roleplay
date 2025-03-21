# nyx/eternal/experience_retriever.py (Refactored)

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.nyx_brain import NyxBrain
from nyx.core.experience_engine import ExperienceEngine

logger = logging.getLogger("experience_retriever")

class ExperienceRetriever:
    """
    Specialized component for retrieving relevant past experiences.
    Now interfaces with the new consolidated architecture.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems (will be initialized lazily)
        self.brain = None
        self.experience_engine = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the experience retriever."""
        if self.initialized:
            return
            
        # Get instances of required systems using the new architecture
        self.brain = await NyxBrain.get_instance(self.user_id, self.conversation_id)
        self.experience_engine = self.brain.experience_engine
        
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
        
        # Use the new ExperienceEngine implementation
        experiences = await self.experience_engine.retrieve_relevant_experiences(
            current_context=current_context,
            limit=limit,
            min_relevance=min_relevance
        )
        
        return experiences
    
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
        
        # Use the new ExperienceEngine implementation
        return await self.experience_engine.handle_experience_sharing_request(
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
        
        # Use the new MemoryCore to store the experience
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
