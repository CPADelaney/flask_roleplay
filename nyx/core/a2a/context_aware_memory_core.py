# nyx/core/a2a/context_aware_memory_core.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareMemoryCore(ContextAwareModule):
    """
    Enhanced MemoryCore with full context distribution capabilities
    """
    
    def __init__(self, original_memory_core):
        super().__init__("memory_core")
        self.original_core = original_memory_core
        self.context_subscriptions = [
            "emotional_state_update", "goal_progress", "knowledge_processing_complete",
            "relationship_milestone", "significant_event", "memory_trigger",
            "consolidation_needed", "reflection_requested", "experience_query"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize memory processing for this context"""
        logger.debug(f"MemoryCore received context for user: {context.user_id}")
        
        # Retrieve memories relevant to current input
        relevant_memories = await self._retrieve_contextual_memories(context)
        
        # Check for memory formation opportunities
        memory_opportunities = await self._identify_memory_opportunities(context)
        
        # Assess need for consolidation or reflection
        maintenance_needs = await self._assess_maintenance_needs(context)
        
        # Send initial memory context
        await self.send_context_update(
            update_type="memory_context_available",
            data={
                "relevant_memories": relevant_memories,
                "memory_count": len(relevant_memories),
                "memory_types": self._categorize_memory_types(relevant_memories),
                "memory_opportunities": memory_opportunities,
                "maintenance_needs": maintenance_needs,
                "recall_confidence": await self._calculate_recall_confidence(relevant_memories)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "emotional_state_update":
            # Store emotional context with current experience
            await self._store_emotional_memory(update.data)
            
        elif update.update_type == "goal_progress":
            # Record goal-related memories
            await self._record_goal_memory(update.data)
            
        elif update.update_type == "knowledge_processing_complete":
            # Link knowledge to memories
            await self._link_knowledge_to_memories(update.data)
            
        elif update.update_type == "relationship_milestone":
            # Store significant relationship memories
            await self._store_relationship_memory(update.data)
            
        elif update.update_type == "significant_event":
            # Store with high significance
            await self._store_significant_event(update.data)
            
        elif update.update_type == "memory_trigger":
            # Handle explicit memory triggers
            await self._process_memory_trigger(update.data)
            
        elif update.update_type == "consolidation_needed":
            # Run consolidation process
            await self._run_targeted_consolidation(update.data)
            
        elif update.update_type == "reflection_requested":
            # Create reflection
            await self._create_contextual_reflection(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with memory awareness"""
        # Analyze input for memory implications
        memory_analysis = await self._analyze_input_for_memories(context)
        
        # Retrieve relevant memories with reconsolidation
        retrieved_memories = await self._retrieve_and_reconsolidate(context)
        
        # Create new memory if appropriate
        new_memory = None
        if memory_analysis["should_store"]:
            new_memory = await self._create_contextual_memory(context)
        
        # Check for pattern emergence
        patterns = await self._detect_memory_patterns(context, retrieved_memories)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Send memory processing update
        await self.send_context_update(
            update_type="memory_processing_complete",
            data={
                "memory_analysis": memory_analysis,
                "retrieved_memories": retrieved_memories,
                "new_memory_id": new_memory["id"] if new_memory else None,
                "patterns_detected": patterns,
                "cross_module_context": len(messages)
            }
        )
        
        return {
            "memories_processed": True,
            "analysis": memory_analysis,
            "retrieved": retrieved_memories,
            "created": new_memory,
            "patterns": patterns
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze memory patterns and health"""
        # Get memory statistics
        memory_stats = await self.original_core.get_memory_stats()
        
        # Analyze memory distribution
        distribution_analysis = await self._analyze_memory_distribution(context)
        
        # Identify consolidation opportunities
        consolidation_candidates = await self._identify_consolidation_candidates(context)
        
        # Assess memory system health
        health_assessment = await self._assess_memory_health(context)
        
        # Analyze cross-module memory usage
        messages = await self.get_cross_module_messages()
        usage_patterns = await self._analyze_cross_module_usage(messages)
        
        # Check for important memories needing crystallization
        crystallization_candidates = await self._identify_crystallization_candidates(context)
        
        return {
            "memory_stats": memory_stats,
            "distribution": distribution_analysis,
            "consolidation_candidates": consolidation_candidates,
            "health_assessment": health_assessment,
            "usage_patterns": usage_patterns,
            "crystallization_candidates": crystallization_candidates
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize memory components for response"""
        messages = await self.get_cross_module_messages()
        
        # Create memory-informed synthesis
        memory_synthesis = {
            "experiential_context": await self._synthesize_experiential_context(context),
            "autobiographical_elements": await self._extract_autobiographical_elements(context),
            "memory_based_insights": await self._generate_memory_insights(context, messages),
            "narrative_suggestions": await self._suggest_narrative_elements(context),
            "memory_coherence": await self._check_memory_coherence(context)
        }
        
        # Check for experience recall opportunity
        recall_opportunity = await self._identify_recall_opportunity(context, messages)
        if recall_opportunity["should_recall"]:
            await self.send_context_update(
                update_type="experience_recall_suggested",
                data=recall_opportunity,
                priority=ContextPriority.HIGH
            )
        
        return memory_synthesis
    
    # ========================================================================================
    # MEMORY-SPECIFIC HELPER METHODS
    # ========================================================================================
    
    async def _retrieve_contextual_memories(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Retrieve memories with full context awareness"""
        # Build retrieval parameters from context
        retrieval_params = {
            "query": context.user_input,
            "memory_types": ["observation", "experience", "reflection", "abstraction"],
            "limit": 10,
            "emotional_state": context.emotional_state,
            "min_fidelity": 0.3
        }
        
        # Add entity filter if available
        if context.session_context.get("entities"):
            retrieval_params["entities"] = context.session_context["entities"]
        
        # Retrieve with contextual boosting
        memories = await self.original_core.context.retrieve_memories(**retrieval_params)
        
        # Apply additional context-based scoring
        for memory in memories:
            memory["context_score"] = await self._calculate_contextual_relevance(memory, context)
        
        # Re-sort by combined relevance
        memories.sort(key=lambda m: m.get("relevance", 0.5) * m.get("context_score", 1.0), reverse=True)
        
        return memories[:5]  # Top 5 most contextually relevant
    
    async def _identify_memory_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities to form new memories"""
        opportunities = []
        
        # Check for significant emotional intensity
        if context.emotional_state:
            primary_intensity = max(v for v in context.emotional_state.values() if isinstance(v, (int, float)))
            if primary_intensity > 0.7:
                opportunities.append({
                    "type": "emotional_significance",
                    "reason": "High emotional intensity experience",
                    "suggested_type": "experience",
                    "significance": int(5 + primary_intensity * 5)
                })
        
        # Check for goal-related events
        if context.goal_context:
            for goal in context.goal_context.get("active_goals", []):
                if goal.get("progress", 0) > 0.8 or goal.get("status") == "completed":
                    opportunities.append({
                        "type": "goal_milestone",
                        "reason": f"Goal progress: {goal.get('description', 'unknown')}",
                        "suggested_type": "observation",
                        "significance": 7
                    })
        
        # Check for learning moments
        if context.session_context.get("knowledge_gained"):
            opportunities.append({
                "type": "learning_moment",
                "reason": "New knowledge acquired",
                "suggested_type": "observation",
                "significance": 6
            })
        
        # Check for relationship developments
        if context.relationship_context:
            trust_change = context.relationship_context.get("trust_delta", 0)
            if abs(trust_change) > 0.1:
                opportunities.append({
                    "type": "relationship_development",
                    "reason": f"Significant trust change: {trust_change:+.2f}",
                    "suggested_type": "experience",
                    "significance": 8
                })
        
        return opportunities
    
    async def _store_emotional_memory(self, emotional_data: Dict[str, Any]):
        """Store memory with emotional context"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Only store if intensity is significant
        if intensity < 0.5:
            return
        
        # Create emotional context
        emotional_context = {
            "primary_emotion": emotion_name,
            "primary_intensity": intensity,
            "secondary_emotions": {k: v for k, v in emotional_state.items() if k != emotion_name},
            "valence": emotional_data.get("valence", 0.0),
            "arousal": emotional_data.get("arousal", 0.0)
        }
        
        # Determine memory text
        memory_text = f"Experienced {emotion_name.lower()} (intensity: {intensity:.2f}) during interaction"
        if self.current_context:
            memory_text += f" about: {self.current_context.user_input[:50]}..."
        
        # Store memory
        memory_id = await self.original_core.add_memory(
            memory_text=memory_text,
            memory_type="experience",
            memory_scope="user",
            significance=int(5 + intensity * 5),
            tags=[emotion_name.lower(), "emotional_experience"],
            metadata={
                "emotional_context": emotional_context,
                "timestamp": datetime.now().isoformat(),
                "context_id": self.current_context.conversation_id if self.current_context else None
            }
        )
        
        logger.debug(f"Stored emotional memory {memory_id} for {emotion_name}")
    
    async def _create_contextual_memory(self, context: SharedContext) -> Dict[str, Any]:
        """Create a memory with full context awareness"""
        # Determine memory type based on context
        memory_type = await self._determine_memory_type(context)
        
        # Calculate significance from multiple factors
        significance = await self._calculate_memory_significance(context)
        
        # Generate comprehensive metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": context.conversation_id,
            "user_id": context.user_id,
            "active_modules": list(context.active_modules),
            "processing_stage": context.processing_stage
        }
        
        # Add emotional context if present
        if context.emotional_state:
            metadata["emotional_context"] = await self._create_emotional_context(context.emotional_state)
        
        # Add knowledge links if present
        knowledge_links = []
        for update in context.context_updates:
            if update.source_module == "knowledge_core" and update.data.get("knowledge_added"):
                knowledge_links.append(update.data["knowledge_added"])
        
        if knowledge_links:
            metadata["knowledge_links"] = knowledge_links
        
        # Extract entities
        entities = await self._extract_entities(context.user_input)
        metadata["entities"] = entities
        
        # Generate memory text
        memory_text = await self._generate_memory_text(context, memory_type)
        
        # Create the memory
        memory_id = await self.original_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            memory_scope="user",
            significance=significance,
            tags=await self._generate_memory_tags(context),
            metadata=metadata
        )
        
        # Check if this memory should be crystallized
        if significance >= 8 or await self._is_identity_relevant(memory_text):
            await self.original_core.crystallize_memory(
                memory_id=memory_id,
                reason="high_significance",
                importance_data={"auto_crystallized": True, "significance": significance}
            )
        
        return {
            "id": memory_id,
            "type": memory_type,
            "significance": significance,
            "crystallized": significance >= 8
        }
    
    async def _detect_memory_patterns(self, context: SharedContext, retrieved_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns across memories"""
        patterns = []
        
        if len(retrieved_memories) < 3:
            return patterns
        
        # Emotional patterns
        emotional_pattern = await self._detect_emotional_pattern(retrieved_memories)
        if emotional_pattern:
            patterns.append(emotional_pattern)
            
            # Send pattern notification
            await self.send_context_update(
                update_type="pattern_detected",
                data={
                    "pattern_type": "emotional",
                    "pattern": emotional_pattern,
                    "source_memories": [m["id"] for m in retrieved_memories[:5]]
                }
            )
        
        # Behavioral patterns
        behavioral_pattern = await self._detect_behavioral_pattern(retrieved_memories)
        if behavioral_pattern:
            patterns.append(behavioral_pattern)
        
        # Temporal patterns
        temporal_pattern = await self._detect_temporal_pattern(retrieved_memories)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        return patterns
    
    async def _retrieve_and_reconsolidate(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Retrieve memories with potential reconsolidation"""
        # Retrieve base memories
        memories = await self._retrieve_contextual_memories(context)
        
        # Apply reconsolidation to eligible memories
        reconsolidated = []
        for memory in memories:
            # Check reconsolidation eligibility
            if await self._should_reconsolidate(memory, context):
                # Apply reconsolidation
                updated_memory = await self._apply_contextual_reconsolidation(memory, context)
                reconsolidated.append(updated_memory)
            else:
                reconsolidated.append(memory)
        
        return reconsolidated
    
    async def _should_reconsolidate(self, memory: Dict[str, Any], context: SharedContext) -> bool:
        """Determine if a memory should be reconsolidated"""
        # Don't reconsolidate crystallized memories
        if memory.get("metadata", {}).get("is_crystallized", False):
            return False
        
        # Don't reconsolidate very recent memories
        timestamp = memory.get("metadata", {}).get("timestamp", "")
        if timestamp:
            memory_age = (datetime.now() - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).days
            if memory_age < 7:
                return False
        
        # Reconsolidate if emotional context has shifted significantly
        if context.emotional_state and memory.get("metadata", {}).get("emotional_context"):
            emotional_shift = await self._calculate_emotional_shift(
                memory["metadata"]["emotional_context"],
                context.emotional_state
            )
            if emotional_shift > 0.5:
                return True
        
        # Random chance based on memory age and recall count
        import random
        recall_count = memory.get("times_recalled", 0)
        reconsolidation_chance = 0.1 + (memory_age / 365) * 0.2 - (recall_count / 20) * 0.1
        
        return random.random() < reconsolidation_chance
    
    async def _apply_contextual_reconsolidation(self, memory: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Apply reconsolidation with context awareness"""
        # Get original text
        original_text = memory["memory_text"]
        
        # Apply contextual modifications
        modified_text = original_text
        
        # Adjust based on emotional reframing
        if context.emotional_state:
            emotional_reframe = await self._generate_emotional_reframe(original_text, context.emotional_state)
            if emotional_reframe:
                modified_text = emotional_reframe
        
        # Update the memory
        updated_memory = memory.copy()
        updated_memory["memory_text"] = modified_text
        
        # Track reconsolidation
        metadata = updated_memory.get("metadata", {})
        if "reconsolidation_history" not in metadata:
            metadata["reconsolidation_history"] = []
        
        metadata["reconsolidation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "original_text": original_text,
            "context_type": "emotional_reframe" if context.emotional_state else "general"
        })
        
        # Reduce fidelity
        current_fidelity = metadata.get("fidelity", 1.0)
        metadata["fidelity"] = max(0.3, current_fidelity - 0.1)
        
        updated_memory["metadata"] = metadata
        
        # Update in storage
        await self.original_core.update_memory(
            memory_id=memory["id"],
            updates={"memory_text": modified_text, "metadata": metadata}
        )
        
        return updated_memory
    
    async def _identify_recall_opportunity(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Identify if this is a good opportunity for experience recall"""
        # Check for direct experience questions
        if any(q in context.user_input.lower() for q in ["remember when", "recall", "that time", "reminds me"]):
            return {
                "should_recall": True,
                "reason": "direct_recall_request",
                "confidence": 0.9
            }
        
        # Check for thematic similarity to past experiences
        relevant_experiences = await self.original_core.retrieve_experiences(
            query=context.user_input,
            limit=3
        )
        
        if relevant_experiences and relevant_experiences[0].get("relevance_score", 0) > 0.7:
            return {
                "should_recall": True,
                "reason": "high_relevance_experience",
                "confidence": relevant_experiences[0]["relevance_score"],
                "experience": relevant_experiences[0]
            }
        
        # Check emotional resonance
        if context.emotional_state:
            emotional_memories = await self._find_emotional_resonance(context.emotional_state)
            if emotional_memories:
                return {
                    "should_recall": True,
                    "reason": "emotional_resonance",
                    "confidence": 0.7,
                    "experience": emotional_memories[0]
                }
        
        return {"should_recall": False}
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _categorize_memory_types(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize retrieved memories by type"""
        type_counts = {}
        for memory in memories:
            mem_type = memory.get("memory_type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        return type_counts
    
    async def _calculate_recall_confidence(self, memories: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in recall"""
        if not memories:
            return 0.0
        
        # Factors: relevance, fidelity, count
        avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
        avg_fidelity = sum(m.get("metadata", {}).get("fidelity", 1.0) for m in memories) / len(memories)
        count_factor = min(1.0, len(memories) / 5)  # More memories = higher confidence
        
        confidence = (avg_relevance * 0.4 + avg_fidelity * 0.4 + count_factor * 0.2)
        
        return confidence
    
    async def _calculate_contextual_relevance(self, memory: Dict[str, Any], context: SharedContext) -> float:
        """Calculate how relevant a memory is to current context"""
        relevance = 1.0  # Base relevance (already filtered by retrieval)
        
        # Boost for goal alignment
        if context.goal_context and memory.get("metadata", {}).get("goal_links"):
            goal_overlap = len(set(memory["metadata"]["goal_links"]) & 
                             set([g["id"] for g in context.goal_context.get("active_goals", [])]))
            relevance *= (1.0 + goal_overlap * 0.1)
        
        # Boost for emotional alignment
        if context.emotional_state and memory.get("metadata", {}).get("emotional_context"):
            emotional_similarity = await self._calculate_emotional_similarity(
                memory["metadata"]["emotional_context"],
                context.emotional_state
            )
            relevance *= (1.0 + emotional_similarity * 0.2)
        
        # Boost for entity overlap
        memory_entities = set(memory.get("metadata", {}).get("entities", []))
        context_entities = set(context.session_context.get("entities", []))
        if memory_entities and context_entities:
            entity_overlap = len(memory_entities & context_entities) / max(len(memory_entities), len(context_entities))
            relevance *= (1.0 + entity_overlap * 0.15)
        
        return min(2.0, relevance)  # Cap at 2x boost
    
    async def _determine_memory_type(self, context: SharedContext) -> str:
        """Determine appropriate memory type from context"""
        # Check for high emotional content
        if context.emotional_state:
            max_emotion = max(context.emotional_state.values())
            if max_emotion > 0.7:
                return "experience"
        
        # Check for reflection indicators
        if any(word in context.user_input.lower() for word in ["realize", "understand", "think that", "believe"]):
            return "reflection"
        
        # Check for abstract thinking
        if any(word in context.user_input.lower() for word in ["pattern", "always", "never", "generally"]):
            return "abstraction"
        
        # Default to observation
        return "observation"
    
    async def _calculate_memory_significance(self, context: SharedContext) -> int:
        """Calculate memory significance from multiple factors"""
        significance = 5  # Base significance
        
        # Emotional intensity boost
        if context.emotional_state:
            max_emotion = max(context.emotional_state.values())
            significance += int(max_emotion * 3)
        
        # Goal relevance boost
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            high_priority_goals = [g for g in active_goals if g.get("priority", 0) > 0.7]
            if high_priority_goals:
                significance += 2
        
        # Relationship significance
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            if trust > 0.8 or trust < 0.2:  # Very high or very low trust is significant
                significance += 1
        
        # Knowledge creation boost
        knowledge_updates = [u for u in context.context_updates if u.source_module == "knowledge_core"]
        if any(u.data.get("knowledge_added") for u in knowledge_updates):
            significance += 1
        
        # Cap at 10
        return min(10, significance)
    
    async def _is_identity_relevant(self, text: str) -> bool:
        """Check if memory is relevant to identity/self-concept"""
        identity_keywords = [
            "i am", "i'm", "my nature", "my personality", "defines me",
            "core to who", "fundamental", "essential", "identity"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in identity_keywords)
    
    async def _generate_emotional_reframe(self, original_text: str, current_emotion: Dict[str, Any]) -> Optional[str]:
        """Generate emotionally reframed version of memory"""
        # Simple reframing based on current emotional state
        dominant_emotion = max(current_emotion.items(), key=lambda x: x[1])[0] if current_emotion else None
        
        if not dominant_emotion:
            return None
        
        # Apply emotional coloring to memory
        if dominant_emotion in ["Joy", "Love", "Gratitude"]:
            # Positive reframe
            return original_text.replace("difficult", "challenging").replace("hard", "growth opportunity")
        elif dominant_emotion in ["Sadness", "Melancholy"]:
            # Melancholic reframe
            return original_text.replace("happy", "bittersweet").replace("exciting", "poignant")
        
        return None
    
    # Delegate to original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original core"""
        return getattr(self.original_core, name)
