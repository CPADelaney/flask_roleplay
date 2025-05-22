# nyx/core/brain/context_aware_examples.py

"""
Example implementations showing how to enhance existing modules 
with context distribution capabilities
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .integration_layer import ContextAwareModule
from .context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareEmotionalCore(ContextAwareModule):
    """
    Example: Enhanced EmotionalCore with context distribution
    """
    
    def __init__(self, original_emotional_core):
        super().__init__("emotional_core")
        self.original_core = original_emotional_core
        self.context_subscriptions = [
            "relationship_updates", "goal_updates", "memory_updates", "sensory_input"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize emotional processing for this context"""
        logger.debug(f"EmotionalCore received context for user: {context.user_id}")
        
        # Extract emotional cues from user input
        emotional_cues = await self._analyze_emotional_cues(context.user_input)
        
        # Send initial emotional assessment to other modules
        await self.send_context_update(
            update_type="emotional_assessment",
            data={
                "emotional_cues": emotional_cues,
                "current_emotional_state": self.original_core.get_emotional_state(),
                "valence": self.original_core.get_emotional_valence(),
                "arousal": self.original_core.get_emotional_arousal()
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect emotions"""
        if update.update_type == "relationship_state_change":
            # Relationship changes affect emotional state
            relationship_data = update.data.get("relationship_context", {})
            trust_level = relationship_data.get("trust", 0.5)
            intimacy_level = relationship_data.get("intimacy", 0.5)
            
            # Adjust emotions based on relationship state
            if trust_level > 0.8:
                self.original_core.update_emotion("Security", 0.2)
            if intimacy_level > 0.7:
                self.original_core.update_emotion("Affection", 0.15)
            
            # Send emotional response back
            await self.send_context_update(
                update_type="emotional_response_to_relationship",
                data={"emotional_adjustment": "positive_relationship_feedback"},
                target_modules=["relationship_manager"],
                scope=ContextScope.TARGETED
            )
        
        elif update.update_type == "goal_progress":
            # Goal progress affects emotions
            goal_data = update.data.get("goal_context", {})
            if goal_data.get("goal_completed"):
                self.original_core.update_emotion("Satisfaction", 0.3)
                self.original_core.update_emotion("Pride", 0.2)
            elif goal_data.get("goal_blocked"):
                self.original_core.update_emotion("Frustration", 0.2)
        
        elif update.update_type == "memory_emotional_trigger":
            # Emotional memories affect current state
            memory_data = update.data.get("memory_context", {})
            emotional_intensity = memory_data.get("emotional_intensity", 0.0)
            emotional_valence = memory_data.get("emotional_valence", 0.0)
            
            if emotional_intensity > 0.5:
                # Strong emotional memory - adjust current state
                if emotional_valence > 0:
                    self.original_core.update_emotion("Nostalgia", emotional_intensity * 0.3)
                else:
                    self.original_core.update_emotion("Melancholy", emotional_intensity * 0.2)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness"""
        # Get cross-module messages for emotional context
        messages = await self.get_cross_module_messages()
        
        # Analyze input with full context
        emotional_response = await self._process_input_with_context(
            context.user_input, context, messages
        )
        
        # Update context with emotional state
        await self.send_context_update(
            update_type="emotional_state_update",
            data={
                "emotional_state": self.original_core.get_emotional_state(),
                "emotional_response": emotional_response,
                "dominant_emotion": self.original_core.get_dominant_emotion(),
                "emotional_context_integration": True
            }
        )
        
        return emotional_response
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize emotional components for final response"""
        # Get all relevant context
        emotional_state = self.original_core.get_emotional_state()
        messages = await self.get_cross_module_messages()
        
        # Consider relationship context for emotional expression
        relationship_context = context.relationship_context
        emotional_expression_level = self._calculate_expression_level(relationship_context)
        
        # Generate emotional component of response
        emotional_synthesis = {
            "emotional_tone": self._determine_response_tone(emotional_state),
            "emotional_expression_level": emotional_expression_level,
            "suggested_emotional_markers": self._suggest_emotional_markers(emotional_state),
            "emotional_coherence_check": self._check_emotional_coherence(context, messages)
        }
        
        return emotional_synthesis
    
    async def _analyze_emotional_cues(self, user_input: str) -> Dict[str, Any]:
        """Analyze emotional cues in user input"""
        if hasattr(self.original_core, 'analyze_text_sentiment'):
            return self.original_core.analyze_text_sentiment(user_input)
        return {"sentiment": 0.0, "emotional_words": []}
    
    async def _process_input_with_context(self, user_input: str, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Process input considering full context"""
        # Use original processing
        if hasattr(self.original_core, 'process_emotional_input'):
            base_response = await self.original_core.process_emotional_input(user_input)
        else:
            base_response = {}
        
        # Enhance with context
        context_enhanced_response = {
            **base_response,
            "context_aware": True,
            "relationship_informed": bool(context.relationship_context),
            "goal_informed": bool(context.goal_context),
            "memory_informed": bool(context.memory_context),
            "cross_module_messages_count": len(messages)
        }
        
        return context_enhanced_response
    
    def _calculate_expression_level(self, relationship_context: Dict) -> float:
        """Calculate how much emotion to express based on relationship"""
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        return min(1.0, (trust + intimacy) / 2)
    
    def _determine_response_tone(self, emotional_state: Dict) -> str:
        """Determine the emotional tone for responses"""
        if not emotional_state:
            return "neutral"
        
        dominant_emotion, strength = max(emotional_state.items(), key=lambda x: x[1])
        
        if strength > 0.7:
            return f"strong_{dominant_emotion.lower()}"
        elif strength > 0.4:
            return f"moderate_{dominant_emotion.lower()}"
        else:
            return "mild_emotional"
    
    def _suggest_emotional_markers(self, emotional_state: Dict) -> List[str]:
        """Suggest emotional markers for response text"""
        markers = []
        for emotion, strength in emotional_state.items():
            if strength > 0.5:
                markers.append(f"express_{emotion.lower()}")
        return markers[:3]  # Limit to top 3
    
    def _check_emotional_coherence(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Check emotional coherence across modules"""
        coherence_score = 1.0
        issues = []
        
        # Check consistency with relationship context
        relationship_context = context.relationship_context
        if relationship_context:
            conflict_level = relationship_context.get("conflict", 0.0)
            current_positive_emotions = sum(
                v for k, v in self.original_core.get_emotional_state().items() 
                if k in ["Joy", "Love", "Satisfaction"]
            )
            
            if conflict_level > 0.6 and current_positive_emotions > 0.5:
                coherence_score -= 0.3
                issues.append("high_conflict_vs_positive_emotions")
        
        return {
            "coherence_score": coherence_score,
            "issues": issues,
            "emotional_consistency": coherence_score > 0.7
        }


class ContextAwareMemoryCore(ContextAwareModule):
    """
    Example: Enhanced MemoryCore with context distribution
    """
    
    def __init__(self, original_memory_core):
        super().__init__("memory_core")
        self.original_core = original_memory_core
        self.context_subscriptions = [
            "emotional_updates", "goal_updates", "relationship_updates"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize memory processing for this context"""
        logger.debug(f"MemoryCore received context for input: {context.user_input[:50]}...")
        
        # Retrieve relevant memories based on context
        relevant_memories = await self._retrieve_contextual_memories(context)
        
        # Send memory context to other modules
        await self.send_context_update(
            update_type="memory_context_available",
            data={
                "relevant_memories": relevant_memories,
                "memory_count": len(relevant_memories),
                "memory_types": list(set(m.get("memory_type") for m in relevant_memories))
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that should be stored in memory"""
        if update.update_type == "emotional_state_update":
            # Store emotional experiences
            emotional_data = update.data
            await self._store_emotional_memory(emotional_data)
        
        elif update.update_type == "goal_completion":
            # Store goal achievements
            goal_data = update.data.get("goal_context", {})
            await self._store_goal_memory(goal_data)
        
        elif update.update_type == "relationship_milestone":
            # Store relationship developments
            relationship_data = update.data.get("relationship_context", {})
            await self._store_relationship_memory(relationship_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with memory retrieval and context awareness"""
        # Retrieve memories with full context
        memory_retrieval = await self._contextual_memory_retrieval(context)
        
        # Store this interaction
        memory_id = await self._store_interaction_memory(context)
        
        # Send memory updates
        await self.send_context_update(
            update_type="memory_retrieval_complete",
            data={
                "retrieved_memories": memory_retrieval,
                "new_memory_id": memory_id,
                "memory_integration_ready": True
            }
        )
        
        return {
            "memory_retrieval": memory_retrieval,
            "memory_id": memory_id,
            "context_aware_retrieval": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize memory components for response"""
        messages = await self.get_cross_module_messages()
        
        # Create memory-informed synthesis
        memory_synthesis = {
            "relevant_experiences": await self._extract_relevant_experiences(context),
            "memory_based_insights": await self._generate_memory_insights(context, messages),
            "autobiographical_elements": await self._extract_autobiographical_elements(context),
            "memory_coherence_check": await self._check_memory_coherence(context)
        }
        
        return memory_synthesis
    
    async def _retrieve_contextual_memories(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Retrieve memories considering full context"""
        if hasattr(self.original_core, 'retrieve_memories'):
            # Enhanced retrieval with context
            query = context.user_input
            
            # Add context-based query enhancement
            if context.session_context.get("task_purpose"):
                query += f" {context.session_context['task_purpose']}"
            
            memories = await self.original_core.retrieve_memories(
                query=query,
                limit=10,
                memory_types=["observation", "experience", "reflection"]
            )
            
            return memories
        return []
    
    async def _contextual_memory_retrieval(self, context: SharedContext) -> Dict[str, Any]:
        """Enhanced memory retrieval with cross-module context"""
        base_memories = await self._retrieve_contextual_memories(context)
        
        # Get context from other modules
        messages = await self.get_cross_module_messages()
        
        # Enhance retrieval based on emotional context
        emotional_context = context.emotional_state
        if emotional_context:
            emotional_memories = await self._retrieve_emotional_memories(emotional_context)
            base_memories.extend(emotional_memories)
        
        # Enhance based on goal context
        goal_context = context.goal_context
        if goal_context:
            goal_memories = await self._retrieve_goal_related_memories(goal_context)
            base_memories.extend(goal_memories)
        
        return {
            "base_memories": base_memories,
            "context_enhanced": True,
            "total_memories": len(base_memories),
            "context_sources": list(messages.keys())
        }
    
    async def _store_interaction_memory(self, context: SharedContext) -> str:
        """Store this interaction with full context"""
        if hasattr(self.original_core, 'add_memory'):
            memory_text = f"User interaction: {context.user_input}"
            
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "user_id": context.user_id,
                "active_modules": list(context.active_modules),
                "context_updates_count": len(context.context_updates),
                "emotional_context": context.emotional_state,
                "goal_context": context.goal_context,
                "relationship_context": context.relationship_context
            }
            
            memory_id = await self.original_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                significance=5,
                metadata=metadata
            )
            
            return memory_id
        return "mock_memory_id"
    
    async def _retrieve_emotional_memories(self, emotional_context: Dict) -> List[Dict[str, Any]]:
        """Retrieve memories related to current emotional state"""
        # Implementation would query for memories with similar emotional context
        return []
    
    async def _retrieve_goal_related_memories(self, goal_context: Dict) -> List[Dict[str, Any]]:
        """Retrieve memories related to current goals"""
        # Implementation would query for memories related to active goals
        return []
    
    async def _store_emotional_memory(self, emotional_data: Dict):
        """Store emotional experiences as memories"""
        # Implementation would create memory entries for significant emotional events
        pass
    
    async def _store_goal_memory(self, goal_data: Dict):
        """Store goal-related memories"""
        # Implementation would create memory entries for goal progress/completion
        pass
    
    async def _store_relationship_memory(self, relationship_data: Dict):
        """Store relationship development memories"""
        # Implementation would create memory entries for relationship milestones
        pass
    
    async def _extract_relevant_experiences(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract relevant past experiences for current context"""
        return []
    
    async def _generate_memory_insights(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate insights based on memory patterns"""
        return ["memory_pattern_insight_1", "memory_pattern_insight_2"]
    
    async def _extract_autobiographical_elements(self, context: SharedContext) -> Dict[str, Any]:
        """Extract autobiographical elements relevant to response"""
        return {"personal_experiences": [], "identity_relevant_memories": []}
    
    async def _check_memory_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Check coherence between memories and current context"""
        return {"coherence_score": 0.8, "consistency_issues": []}


class ContextAwareGoalManager(ContextAwareModule):
    """
    Example: Enhanced GoalManager with context distribution
    """
    
    def __init__(self, original_goal_manager):
        super().__init__("goal_manager")
        self.original_manager = original_goal_manager
        self.context_subscriptions = [
            "emotional_updates", "memory_updates", "relationship_updates", "need_updates"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize goal processing for this context"""
        # Get active goals and assess relevance to current context
        active_goals = await self._get_contextually_relevant_goals(context)
        
        # Send goal context to other modules
        await self.send_context_update(
            update_type="goal_context_available",
            data={
                "active_goals": active_goals,
                "goal_priorities": await self._calculate_contextual_priorities(context, active_goals),
                "goal_relevance_to_input": await self._assess_input_goal_relevance(context)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that affect goal management"""
        if update.update_type == "emotional_state_update":
            # Emotional state affects goal motivation
            await self._adjust_goal_motivation_from_emotion(update.data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Past experiences inform goal strategies
            await self._inform_goals_from_memory(update.data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect social goals
            await self._adjust_social_goals(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for goal-relevant actions"""
        # Analyze input for goal implications
        goal_analysis = await self._analyze_input_for_goals(context)
        
        # Execute relevant goal steps
        execution_result = await self._execute_contextual_goal_steps(context)
        
        # Send goal progress updates
        await self.send_context_update(
            update_type="goal_progress",
            data={
                "goal_analysis": goal_analysis,
                "execution_result": execution_result,
                "goal_state_changes": await self._get_goal_state_changes()
            }
        )
        
        return {
            "goal_analysis": goal_analysis,
            "execution_result": execution_result,
            "context_informed_execution": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize goal-related response components"""
        messages = await self.get_cross_module_messages()
        
        # Create goal-informed synthesis
        goal_synthesis = {
            "goal_aligned_suggestions": await self._generate_goal_aligned_suggestions(context),
            "progress_updates": await self._generate_progress_updates(context),
            "next_steps": await self._suggest_next_steps(context, messages),
            "goal_coherence": await self._check_goal_coherence(context, messages)
        }
        
        return goal_synthesis
    
    # Helper methods (implementations would be specific to your goal system)
    async def _get_contextually_relevant_goals(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get goals relevant to current context"""
        if hasattr(self.original_manager, 'get_all_goals'):
            all_goals = await self.original_manager.get_all_goals(status_filter=["active"])
            # Filter based on context relevance
            return all_goals[:5]  # Return top 5 for example
        return []
    
    async def _calculate_contextual_priorities(self, context: SharedContext, goals: List) -> Dict[str, float]:
        """Calculate goal priorities based on current context"""
        priorities = {}
        for goal in goals:
            goal_id = goal.get('id', 'unknown')
            # Calculate priority based on context (simplified)
            base_priority = goal.get('priority', 0.5)
            context_boost = 0.0
            
            # Boost priority if goal relates to current emotional state
            if context.emotional_state:
                # Simple heuristic - boost achievement goals when feeling confident
                dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
                if dominant_emotion == "Confidence" and "achieve" in goal.get('description', '').lower():
                    context_boost += 0.2
            
            priorities[goal_id] = min(1.0, base_priority + context_boost)
        
        return priorities
    
    async def _assess_input_goal_relevance(self, context: SharedContext) -> Dict[str, float]:
        """Assess how relevant user input is to active goals"""
        relevance_scores = {}
        user_input_lower = context.user_input.lower()
        
        active_goals = await self._get_contextually_relevant_goals(context)
        for goal in active_goals:
            goal_id = goal.get('id', 'unknown')
            goal_desc = goal.get('description', '').lower()
            
            # Simple relevance scoring
            common_words = set(user_input_lower.split()) & set(goal_desc.split())
            relevance_scores[goal_id] = len(common_words) / max(1, len(goal_desc.split()))
        
        return relevance_scores
    
    async def _adjust_goal_motivation_from_emotion(self, emotional_data: Dict):
        """Adjust goal motivation based on emotional state"""
        # Implementation would modify goal priorities/motivation based on emotions
        pass
    
    async def _inform_goals_from_memory(self, memory_data: Dict):
        """Inform goal strategies from retrieved memories"""
        # Implementation would use past experiences to inform current goal execution
        pass
    
    async def _adjust_social_goals(self, relationship_data: Dict):
        """Adjust social goals based on relationship state changes"""
        # Implementation would modify relationship/social goals based on relationship context
        pass
    
    async def _analyze_input_for_goals(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze user input for goal-related implications"""
        return {
            "contains_goal_keywords": any(kw in context.user_input.lower() for kw in ["goal", "want", "achieve", "plan"]),
            "suggests_new_goal": "want to" in context.user_input.lower(),
            "references_progress": any(kw in context.user_input.lower() for kw in ["progress", "status", "how am i"])
        }
    
    async def _execute_contextual_goal_steps(self, context: SharedContext) -> Dict[str, Any]:
        """Execute goal steps considering full context"""
        if hasattr(self.original_manager, 'execute_next_step'):
            result = await self.original_manager.execute_next_step()
            return {"execution_result": result, "context_informed": True}
        return {"no_execution": True}
    
    async def _get_goal_state_changes(self) -> Dict[str, Any]:
        """Get recent goal state changes"""
        return {"changes": [], "last_update": datetime.now().isoformat()}
    
    async def _generate_goal_aligned_suggestions(self, context: SharedContext) -> List[str]:
        """Generate suggestions aligned with active goals"""
        return ["Continue working toward your current objectives", "Consider the next steps in your plan"]
    
    async def _generate_progress_updates(self, context: SharedContext) -> List[str]:
        """Generate progress updates for active goals"""
        return ["Goal progress update 1", "Goal progress update 2"]
    
    async def _suggest_next_steps(self, context: SharedContext, messages: Dict) -> List[str]:
        """Suggest next steps based on context and cross-module input"""
        return ["Next step suggestion 1", "Next step suggestion 2"]
    
    async def _check_goal_coherence(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Check coherence between goals and other module outputs"""
        return {"coherence_score": 0.9, "alignment_issues": []}
