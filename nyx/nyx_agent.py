# nyx/nyx_agent.py

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.nyx_decision_engine import NyxDecisionEngine
from nyx.nyx_memory_system import NyxMemorySystem
from nyx.nyx_model_manager import UserModelManager
from logic.nyx_enhancements_integration import enhance_context_with_memories
from logic.nyx_memory import NyxMemoryManager, perform_memory_maintenance

logger = logging.getLogger(__name__)

class NyxAgent:
    """
    Integrated agent for Nyx that combines:
    - Enhanced memory system
    - Decision engine
    - User modeling
    - Narrative management
    - Social dynamics
    
    This serves as the central point of integration for all Nyx systems.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the Nyx agent with required components."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems
        self.decision_engine = NyxDecisionEngine(user_id, conversation_id)
        self.memory_system = NyxMemorySystem(user_id, conversation_id)
        self.user_model = UserModelManager(user_id, conversation_id)
        
        # Legacy memory manager for compatibility
        self.legacy_memory = NyxMemoryManager(user_id, conversation_id)
        
        # Agent state
        self.last_response_time = None
        self.conversation_context = {}
        
    async def process_input(
        self,
        user_input: str,
        context: Dict[str, Any] = None,
        system_directives: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process user input and generate Nyx's response with full context integration.
        
        Args:
            user_input: User's message
            context: Environmental context (location, NPCs present, etc.)
            system_directives: Optional system directives for response behavior
            
        Returns:
            Complete response object with text and metadata
        """
        start_time = datetime.now()
        context = context or {}
        system_directives = system_directives or {}
        
        # Update conversation context
        self.conversation_context.update(context)
        
        # 1. Process input through memory systems
        memory_enhancement = await self._process_memory_systems(user_input, context)
        
        # 2. Update user model and get guidance
        await self._update_user_model(user_input, context)
        response_guidance = await self.user_model.get_response_guidance()
        
        # 3. Apply narrative considerations
        narrative_context = await self._get_narrative_context(context)
        
        # 4. Get decision from decision engine
        enhanced_context = self._enhance_context_with_memories(
            context, memory_enhancement, narrative_context
        )
        response = await self.decision_engine.get_response(user_input, enhanced_context)
        
        # 5. Record interaction in memory systems
        await self._record_interaction(user_input, response["text"], context)
        
        # 6. Check for maintenance needs
        await self._check_maintenance_needs()
        
        # Track response time
        self.last_response_time = datetime.now()
        response["processing_time"] = (self.last_response_time - start_time).total_seconds()
        
        return response
    
    async def _process_memory_systems(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process user input through memory systems."""
        # 1. Retrieve relevant memories
        memories = await self.memory_system.retrieve_memories(
            query=user_input,
            scopes=["game", "user"],
            memory_types=["observation", "reflection", "abstraction"],
            limit=5,
            context=context
        )
        
        # 2. Get legacy memory enhancement (for backward compatibility)
        legacy_enhancement = await enhance_context_with_memories(
            self.user_id,
            self.conversation_id,
            user_input,
            context,
            self.legacy_memory
        )
        
        # 3. Store user input in memory system
        await self.memory_system.add_memory(
            memory_text=f"Player said: {user_input}",
            memory_type="observation",
            memory_scope="game",
            significance=4,
            tags=["player_input"],
            metadata={
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Combine both memory systems' results
        return {
            "memories": memories,
            "legacy_enhancement": legacy_enhancement,
            "memory_context": self._format_memories_for_context(memories)
        }
    
    def _format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for inclusion in context."""
        if not memories:
            return ""
        
        memory_texts = []
        for memory in memories:
            relevance = memory.get("relevance", 0.5)
            confidence_marker = "vividly recall" if relevance > 0.8 else \
                              "remember" if relevance > 0.6 else \
                              "think I recall" if relevance > 0.4 else \
                              "vaguely remember"
            
            memory_texts.append(f"I {confidence_marker}: {memory['memory_text']}")
        
        return "\n".join(memory_texts)
    
    async def _update_user_model(self, user_input: str, context: Dict[str, Any]):
        """Update user model based on input."""
        # Analyze input for preference revelations
        revelations = await self.decision_engine._detect_user_revelations(user_input, context)
        
        # Process each revelation
        for revelation in revelations:
            if revelation["type"] == "kink_preference":
                await self.user_model.track_kink_preference(
                    kink_name=revelation["kink"],
                    intensity=revelation["intensity"],
                    detected_from=revelation["source"]
                )
            elif revelation["type"] == "behavior_pattern":
                await self.user_model.track_behavior_pattern(
                    pattern_type=revelation["pattern"],
                    pattern_value=revelation["pattern"],
                    intensity=revelation["intensity"]
                )
    
    async def _get_narrative_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current narrative context."""
        # This could be expanded to include more sophisticated narrative tracking
        return context.get("narrative_context", {})
    
    def _enhance_context_with_memories(
        self,
        context: Dict[str, Any],
        memory_enhancement: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance context with memory information."""
        enhanced_context = context.copy()
        
        # Add memory context
        enhanced_context["memory_context"] = memory_enhancement.get("memory_context", "")
        
        # Add legacy enhancement for backwards compatibility
        legacy_text = memory_enhancement.get("legacy_enhancement", {}).get("text", "")
        if legacy_text:
            enhanced_context["legacy_memory"] = legacy_text
        
        # Add narrative context
        enhanced_context["narrative_context"] = narrative_context
        
        return enhanced_context
    
    async def _record_interaction(
        self,
        user_input: str,
        nyx_response: str,
        context: Dict[str, Any]
    ):
        """Record the interaction in memory systems."""
        # Record in new memory system
        await self.memory_system.add_memory(
            memory_text=f"I responded to player: {nyx_response[:200]}...",
            memory_type="observation",
            memory_scope="game",
            significance=4,
            tags=["nyx_response"],
            metadata={
                "user_input": user_input,
                "full_response": nyx_response,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Record in legacy memory system for compatibility
        await self.legacy_memory.add_memory(
            memory_text=f"I responded: {nyx_response[:200]}...",
            memory_type="observation",
            significance=4,
            tags=["nyx_response"],
            related_entities={"player": "Chase"},
            context=context
        )
        
        # Track conversation in user model
        await self.user_model.track_conversation_response(
            user_message=user_input,
            nyx_response=nyx_response,
            conversation_context=context
        )
    
    async def _check_maintenance_needs(self):
        """Check if memory maintenance is needed."""
        # Simple time-based check (could be made more sophisticated)
        if not self.last_response_time:
            return
            
        # Run maintenance every 20 interactions or after long periods
        hours_since_last = (datetime.now() - self.last_response_time).total_seconds() / 3600
        if hours_since_last > 6:
            asyncio.create_task(self._run_maintenance())
    
    async def _run_maintenance(self):
        """Run maintenance tasks for memory systems."""
        try:
            # Run maintenance on legacy memory system
            await perform_memory_maintenance(self.user_id, self.conversation_id)
            
            # Run maintenance on new memory system
            await self.memory_system.run_maintenance()
            
            logger.info(f"Completed memory maintenance for user_id={self.user_id}, conversation_id={self.conversation_id}")
        except Exception as e:
            logger.error(f"Error in memory maintenance: {str(e)}")
    
    async def generate_reflection(self, topic: str = None) -> Dict[str, Any]:
        """
        Generate a reflection on a topic or the player.
        
        Args:
            topic: Optional topic to reflect on
            
        Returns:
            Reflection data
        """
        return await self.memory_system.generate_reflection(
            topic=topic,
            context=self.conversation_context
        )
    
    async def get_introspection(self) -> Dict[str, Any]:
        """
        Get Nyx's introspection about her memories and understanding.
        
        Returns:
            Introspection data
        """
        # Get legacy introspection
        legacy_introspection = await self.legacy_memory.generate_introspection()
        
        # Get new system reflection
        reflection = await self.memory_system.generate_reflection("self_understanding")
        
        # Combine the results
        return {
            "introspection": reflection.get("reflection", ""),
            "confidence": reflection.get("confidence", 0.5),
            "legacy_introspection": legacy_introspection.get("introspection", ""),
            "memory_stats": legacy_introspection.get("memory_stats", {})
        }
