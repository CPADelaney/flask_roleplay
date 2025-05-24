# nyx/core/a2a/context_aware_reflection_engine.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareReflectionEngine(ContextAwareModule):
    """
    Enhanced ReflectionEngine with full context distribution capabilities
    """
    
    def __init__(self, original_reflection_engine):
        super().__init__("reflection_engine")
        self.original_engine = original_reflection_engine
        self.context_subscriptions = [
            "memory_retrieval_complete", "emotional_state_update", "goal_completion",
            "relationship_milestone", "pattern_detected", "abstraction_needed",
            "introspection_request", "observation_accumulation", "communication_summary"
        ]
        
        # Context-aware reflection parameters
        self.reflection_urgency = 0.5
        self.reflection_depth = 0.7
        self.reflection_focus_areas = []
        self.pending_reflection_topics = []
        
    async def on_context_received(self, context: SharedContext):
        """Initialize reflection processing for this context"""
        logger.debug(f"ReflectionEngine received context for user: {context.user_id}")
        
        # Check if it's time for reflection
        should_reflect = await self._check_reflection_timing(context)
        
        # Analyze what type of reflection might be needed
        reflection_needs = await self._analyze_reflection_needs(context)
        
        # Send initial reflection context
        await self.send_context_update(
            update_type="reflection_system_ready",
            data={
                "should_reflect": should_reflect,
                "reflection_needs": reflection_needs,
                "last_reflection": self.original_engine.reflection_intervals.get("last_reflection").isoformat() if hasattr(self.original_engine, 'reflection_intervals') else None,
                "reflection_depth": self.reflection_depth
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that might trigger reflection"""
        
        if update.update_type == "memory_retrieval_complete":
            # New memories might need reflection
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            if len(memories) > 5:  # Significant memory retrieval
                self.pending_reflection_topics.append({
                    "topic": "recent_memories",
                    "data": memories,
                    "urgency": 0.6
                })
                
        elif update.update_type == "emotional_state_update":
            # Significant emotional changes warrant reflection
            emotional_data = update.data
            if self._is_significant_emotional_change(emotional_data):
                self.pending_reflection_topics.append({
                    "topic": "emotional_experience",
                    "data": emotional_data,
                    "urgency": 0.7
                })
                
        elif update.update_type == "goal_completion":
            # Goal completions deserve reflection
            goal_data = update.data
            self.pending_reflection_topics.append({
                "topic": "goal_achievement",
                "data": goal_data,
                "urgency": 0.8
            })
            
        elif update.update_type == "relationship_milestone":
            # Relationship developments need reflection
            relationship_data = update.data
            self.pending_reflection_topics.append({
                "topic": "relationship_development",
                "data": relationship_data,
                "urgency": 0.7
            })
            
        elif update.update_type == "introspection_request":
            # Direct request for introspection
            self.reflection_urgency = 0.9
            self.pending_reflection_topics.append({
                "topic": "system_introspection",
                "data": update.data,
                "urgency": 0.9
            })
            
        elif update.update_type == "observation_accumulation":
            # Observations have accumulated enough for reflection
            observation_data = update.data
            if observation_data.get("observation_count", 0) > 10:
                self.pending_reflection_topics.append({
                    "topic": "observation_patterns",
                    "data": observation_data,
                    "urgency": 0.5
                })
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for reflection needs"""
        messages = await self.get_cross_module_messages()
        
        # Check if input suggests need for reflection
        reflection_triggers = await self._check_input_for_reflection_triggers(context.user_input)
        
        # Process any pending reflections if appropriate
        reflections_generated = []
        
        if reflection_triggers or self.pending_reflection_topics:
            # Sort pending topics by urgency
            self.pending_reflection_topics.sort(key=lambda x: x["urgency"], reverse=True)
            
            # Process top priority reflection
            if self.pending_reflection_topics:
                top_topic = self.pending_reflection_topics.pop(0)
                
                reflection_result = await self._generate_contextual_reflection(
                    topic_data=top_topic,
                    context=context,
                    messages=messages
                )
                
                if reflection_result:
                    reflections_generated.append(reflection_result)
                    
                    # Send reflection update
                    await self.send_context_update(
                        update_type="reflection_generated",
                        data={
                            "reflection_text": reflection_result["text"],
                            "reflection_type": reflection_result["type"],
                            "confidence": reflection_result["confidence"],
                            "topic": top_topic["topic"]
                        }
                    )
        
        return {
            "reflections_generated": reflections_generated,
            "pending_topics": len(self.pending_reflection_topics),
            "reflection_triggers": reflection_triggers,
            "context_integrated": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze need for various types of reflection"""
        messages = await self.get_cross_module_messages()
        
        # Analyze memory patterns for abstraction needs
        abstraction_analysis = await self._analyze_abstraction_needs(context, messages)
        
        # Analyze emotional patterns
        emotional_analysis = await self._analyze_emotional_reflection_needs(context, messages)
        
        # Analyze system state for introspection
        introspection_analysis = await self._analyze_introspection_needs(context, messages)
        
        # Analyze cross-module patterns
        integration_analysis = await self._analyze_integration_patterns(messages)
        
        return {
            "abstraction_needed": abstraction_analysis["needed"],
            "emotional_reflection_needed": emotional_analysis["needed"],
            "introspection_needed": introspection_analysis["needed"],
            "integration_insights": integration_analysis,
            "reflection_priorities": await self._prioritize_reflections(context, messages)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reflection components for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine if reflections should be included in response
        include_reflections = await self._should_include_reflections(context, messages)
        
        reflection_synthesis = {
            "include_in_response": include_reflections,
            "reflection_tone": await self._determine_reflection_tone(context),
            "reflection_depth": self.reflection_depth,
            "integration_level": await self._calculate_integration_level(messages)
        }
        
        # If we have generated reflections that should be surfaced
        if include_reflections and hasattr(self, '_recent_reflections'):
            recent = self._recent_reflections[-1] if self._recent_reflections else None
            if recent:
                await self.send_context_update(
                    update_type="surface_reflection",
                    data={
                        "reflection_text": recent["text"],
                        "reflection_type": recent["type"],
                        "integration_suggestion": "weave_into_response",
                        "tone": reflection_synthesis["reflection_tone"]
                    },
                    priority=ContextPriority.HIGH
                )
        
        return reflection_synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _check_reflection_timing(self, context: SharedContext) -> bool:
        """Check if it's appropriate time for reflection"""
        if hasattr(self.original_engine, 'should_reflect'):
            base_should_reflect = self.original_engine.should_reflect()
        else:
            # Default timing check
            base_should_reflect = True
        
        # Context-based adjustments
        if context.emotional_state:
            # High emotional intensity might trigger reflection sooner
            intensity = max(v for v in context.emotional_state.values() if isinstance(v, (int, float)))
            if intensity > 0.8:
                return True
        
        # Check if conversation is winding down (good time for reflection)
        if context.user_input.lower() in ["goodbye", "bye", "see you", "talk later"]:
            return True
        
        return base_should_reflect
    
    async def _analyze_reflection_needs(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze what types of reflection might be needed"""
        needs = {
            "memory_reflection": False,
            "emotional_reflection": False,
            "goal_reflection": False,
            "relationship_reflection": False,
            "observation_reflection": False,
            "communication_reflection": False
        }
        
        # Check each subsystem's state
        if context.memory_context:
            memory_count = context.memory_context.get("memory_count", 0)
            if memory_count > 5:
                needs["memory_reflection"] = True
        
        if context.emotional_state:
            needs["emotional_reflection"] = True
        
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals:
                needs["goal_reflection"] = True
        
        if context.relationship_context:
            needs["relationship_reflection"] = True
        
        return needs
    
    def _is_significant_emotional_change(self, emotional_data: Dict[str, Any]) -> bool:
        """Check if emotional change is significant enough for reflection"""
        # Check for emotion intensity
        if "dominant_emotion" in emotional_data:
            emotion, intensity = emotional_data["dominant_emotion"]
            if intensity > 0.7:
                return True
        
        # Check for valence shift
        if "valence" in emotional_data:
            # Would need to track previous valence to detect shift
            # For now, extreme valences trigger reflection
            valence = emotional_data["valence"]
            if abs(valence) > 0.7:
                return True
        
        return False
    
    async def _check_input_for_reflection_triggers(self, user_input: str) -> List[str]:
        """Check if user input contains reflection triggers"""
        triggers = []
        
        reflection_keywords = [
            "think about", "reflect", "consider", "wondering",
            "looking back", "realize", "understand", "meaning",
            "pattern", "insight", "learned", "growth"
        ]
        
        input_lower = user_input.lower()
        for keyword in reflection_keywords:
            if keyword in input_lower:
                triggers.append(keyword)
        
        # Questions often trigger reflection
        if "?" in user_input and any(word in input_lower for word in ["why", "how", "what does it mean"]):
            triggers.append("reflective_question")
        
        return triggers
    
    async def _generate_contextual_reflection(self, 
                                            topic_data: Dict[str, Any],
                                            context: SharedContext,
                                            messages: Dict) -> Optional[Dict[str, Any]]:
        """Generate reflection using full context"""
        topic = topic_data["topic"]
        data = topic_data["data"]
        
        try:
            # Route to appropriate reflection type
            if topic == "recent_memories":
                memories = data if isinstance(data, list) else []
                reflection_text, confidence = await self.original_engine.generate_reflection(
                    memories=memories,
                    topic="recent experiences",
                    neurochemical_state=self._extract_neurochemical_state(context)
                )
                return {
                    "text": reflection_text,
                    "type": "memory_reflection",
                    "confidence": confidence
                }
                
            elif topic == "emotional_experience":
                # Generate emotional reflection
                if hasattr(self.original_engine, 'process_emotional_state'):
                    result = await self.original_engine.process_emotional_state(
                        emotional_state=data.get("emotional_state", {}),
                        neurochemical_state=self._extract_neurochemical_state(context)
                    )
                    return {
                        "text": result.get("processing_text", ""),
                        "type": "emotional_reflection",
                        "confidence": result.get("insight_level", 0.5)
                    }
                    
            elif topic == "observation_patterns":
                # Generate observation reflection
                if hasattr(self.original_engine, 'generate_observation_reflection'):
                    observations = data.get("observations", [])
                    reflection_text, confidence = await self.original_engine.generate_observation_reflection(
                        observations=observations,
                        topic="my observation patterns"
                    )
                    return {
                        "text": reflection_text,
                        "type": "observation_reflection",
                        "confidence": confidence
                    }
                    
            elif topic == "system_introspection":
                # Generate introspection
                if hasattr(self.original_engine, 'generate_introspection'):
                    memory_stats = await self._get_memory_stats()
                    result = await self.original_engine.generate_introspection(
                        memory_stats=memory_stats,
                        neurochemical_state=self._extract_neurochemical_state(context)
                    )
                    return {
                        "text": result.get("introspection", ""),
                        "type": "introspection",
                        "confidence": result.get("confidence", 0.5)
                    }
                    
        except Exception as e:
            logger.error(f"Error generating contextual reflection: {e}")
            
        return None
    
    def _extract_neurochemical_state(self, context: SharedContext) -> Dict[str, float]:
        """Extract neurochemical state from context"""
        # Check if it's in emotional state
        if context.emotional_state and "neurochemical_influence" in context.emotional_state:
            return context.emotional_state["neurochemical_influence"]
        
        # Default balanced state
        return {
            "nyxamine": 0.5,
            "seranix": 0.5,
            "oxynixin": 0.5,
            "cortanyx": 0.3,
            "adrenyx": 0.3
        }
    
    async def _should_include_reflections(self, context: SharedContext, messages: Dict) -> bool:
        """Determine if reflections should be included in response"""
        # Check if user is asking for reflection
        if any(word in context.user_input.lower() for word in ["reflect", "think", "insight", "meaning"]):
            return True
        
        # Check if conversation is at a natural reflection point
        if context.user_input.lower() in ["goodbye", "bye", "thank you", "see you"]:
            return True
        
        # Check if other modules suggest reflection
        for module_messages in messages.values():
            for msg in module_messages:
                if msg.get("type") == "request_reflection":
                    return True
        
        # Default: include if urgency is high
        return self.reflection_urgency > 0.7
    
    async def _determine_reflection_tone(self, context: SharedContext) -> str:
        """Determine appropriate tone for reflection"""
        # Base tone on emotional state
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                emotion_name = dominant_emotion[0] if isinstance(dominant_emotion, tuple) else dominant_emotion
                
                tone_mapping = {
                    "Joy": "warm",
                    "Sadness": "contemplative",
                    "Fear": "reassuring",
                    "Anger": "understanding",
                    "Love": "intimate",
                    "Curiosity": "exploratory"
                }
                
                return tone_mapping.get(emotion_name, "thoughtful")
        
        return "thoughtful"
    
    # Delegate other methods to original engine
    def __getattr__(self, name):
        """Delegate any missing methods to the original engine"""
        return getattr(self.original_engine, name)
