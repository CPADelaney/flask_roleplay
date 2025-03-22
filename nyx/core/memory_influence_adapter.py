# nyx/core/memory_influence_adapter.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.integration_utils import ComponentIntegration, NyxMemoryIntegration

logger = logging.getLogger(__name__)

class MemoryInfluenceAdapter:
    """
    Adapter that integrates the component influence matrix with memory operations.
    This allows dynamic weighting of memory operations based on context and
    interaction with other components.
    """
    
    def __init__(self, memory_core, influence_matrix=None):
        """
        Initialize the memory influence adapter
        
        Args:
            memory_core: The memory core instance
            influence_matrix: Optional component influence matrix
        """
        self.memory_core = memory_core
        self.influence_matrix = influence_matrix
        self.context_cache = {}
        
        # Tracking for component activity
        self.memory_activity = 0.0
        self.interaction_count = 0
    
    async def retrieve_memories_with_influence(self,
                                       query: str,
                                       context: Dict[str, Any] = None,
                                       memory_types: List[str] = None,
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories with influence-aware parameters
        
        Args:
            query: Search query
            context: Context data
            memory_types: Optional memory types to include
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of retrieved memories
        """
        # Track interaction
        self.interaction_count += 1
        
        # Default context
        if context is None:
            context = {}
        
        # Enhance retrieval parameters with influence-based adjustments
        enhanced_params = await NyxMemoryIntegration.enhance_memory_retrieval(
            query, self.memory_core, self.influence_matrix, context
        )
        
        # Override with explicit parameters if provided
        if memory_types:
            enhanced_params["memory_types"] = memory_types
        if limit:
            enhanced_params["limit"] = limit
        
        # Retrieve memories with enhanced parameters
        try:
            memories = await self.memory_core.retrieve_memories(
                query=enhanced_params["query"],
                memory_types=enhanced_params["memory_types"],
                limit=enhanced_params["limit"],
                min_significance=enhanced_params["min_significance"],
                include_archived=context.get("include_archived", False),
                entities=context.get("entities"),
                emotional_state=context.get("emotional_state")
            )
            
            # Track memory activity based on success
            memory_activity = 0.3  # Base activity
            if memories:
                # Scale activity based on number and relevance of memories
                avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories) if memories else 0.5
                memory_activity += min(0.7, (len(memories) / 10) + (avg_relevance * 0.5))
            
            self.memory_activity = memory_activity
            
            # Update context cache
            self.context_cache = {
                "last_query": query,
                "last_context": context,
                "last_result_count": len(memories),
                "last_interaction": datetime.now().isoformat(),
                "memory_activity": memory_activity
            }
            
            return memories
        except Exception as e:
            logger.error(f"Error in enhanced memory retrieval: {e}")
            # Fallback to basic retrieval
            return await self.memory_core.retrieve_memories(
                query=query,
                memory_types=memory_types or ["observation", "reflection", "abstraction", "experience"],
                limit=limit
            )
    
    async def add_memory_with_influence(self,
                                  memory_text: str,
                                  memory_type: str = "observation",
                                  memory_scope: str = "game",
                                  context: Dict[str, Any] = None,
                                  tags: List[str] = None,
                                  significance: int = None) -> str:
        """
        Add a memory with dynamically adjusted significance based on context
        
        Args:
            memory_text: Text of the memory
            memory_type: Type of memory
            memory_scope: Scope of memory
            context: Context data
            tags: Optional tags
            significance: Optional explicit significance (1-10)
            
        Returns:
            Memory ID
        """
        if context is None:
            context = {}
        
        # Calculate dynamic significance if not explicitly provided
        if significance is None:
            # Default significance
            dynamic_significance = 5
            
            # Get relevant influence weights
            if self.influence_matrix:
                # Main components that influence memory significance
                components = ["emotion", "reasoning", "experience"]
                
                # Calculate weighted significance based on component influence
                weighted_sum = 0.0
                total_weight = 0.0
                
                for component in components:
                    influence = await ComponentIntegration.get_influence_weight(
                        self.influence_matrix, component, "memory", context
                    )
                    
                    # Component-specific significance adjustments
                    if component == "emotion" and "emotional_state" in context:
                        # More significant if high emotional intensity
                        arousal = context["emotional_state"].get("arousal", 0.5)
                        component_significance = 5 + int(arousal * 4)
                    elif component == "reasoning" and self._is_reasoning_related(memory_text):
                        # More significant if reasoning-related
                        component_significance = 7
                    elif component == "experience" and memory_type == "experience":
                        # Experience memories tend to be more significant
                        component_significance = 6
                    else:
                        component_significance = 5
                    
                    weighted_sum += influence * component_significance
                    total_weight += influence
                
                # Calculate final significance
                if total_weight > 0:
                    dynamic_significance = int(weighted_sum / total_weight)
                    # Ensure in valid range
                    dynamic_significance = max(1, min(10, dynamic_significance))
            
            significance = dynamic_significance
        
        # Enhance tags based on context if needed
        if tags is None:
            tags = []
        
        # Add memory using core method
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            memory_scope=memory_scope,
            significance=significance,
            tags=tags,
            metadata=context.get("metadata", {})
        )
        
        # Track activity
        self.memory_activity = 0.7
        
        return memory_id
    
    async def create_reflection_with_influence(self, 
                                         topic: Optional[str] = None,
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a reflection with influence-aware parameters
        
        Args:
            topic: Optional topic to reflect on
            context: Context data
            
        Returns:
            Reflection result
        """
        if context is None:
            context = {}
        
        # Get memory influence on reflection
        memory_reflection_influence = 0.6  # Default
        if self.influence_matrix:
            memory_reflection_influence = await ComponentIntegration.get_influence_weight(
                self.influence_matrix, "memory", "reflection", context
            )
        
        # Adjust reflection parameters based on influence
        memory_limit = 5  # Default
        if memory_reflection_influence > 0.7:
            # Higher influence means use more memories for reflection
            memory_limit = 8
        elif memory_reflection_influence < 0.3:
            # Lower influence means use fewer memories
            memory_limit = 3
        
        try:
            # Create reflection with adjusted parameters
            reflection_result = await self.memory_core.create_reflection_from_memories(topic)
            
            # Track activity
            self.memory_activity = 0.6
            
            return reflection_result
        except Exception as e:
            logger.error(f"Error in reflection with influence: {e}")
            # Fallback to basic reflection
            return await self.memory_core.create_reflection_from_memories(topic)
    
    def get_component_activity(self) -> float:
        """Get current memory component activity level"""
        return self.memory_activity
    
    def _is_reasoning_related(self, text: str) -> bool:
        """Check if text is related to reasoning processes"""
        reasoning_indicators = [
            "because", "therefore", "since", "due to", "consequently",
            "reason", "logic", "analysis", "conclude", "deduction",
            "inference", "think", "understand", "recognize", "connect"
        ]
        
        return any(indicator in text.lower() for indicator in reasoning_indicators)
