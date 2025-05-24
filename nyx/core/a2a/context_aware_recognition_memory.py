# nyx/core/a2a/context_aware_recognition_memory.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareRecognitionMemory(ContextAwareModule):
    """
    Enhanced RecognitionMemorySystem with full context distribution capabilities
    """
    
    def __init__(self, original_recognition_memory):
        super().__init__("recognition_memory")
        self.original_system = original_recognition_memory
        self.context_subscriptions = [
            "emotional_state_update", "goal_progress", "attention_shift",
            "memory_retrieval_complete", "needs_assessment", "mode_change",
            "relationship_state_change", "urgency_detected"
        ]
        
        # Context-influenced recognition parameters
        self.context_recognition_boost = {}  # module -> boost factor
        self.attention_filter = set()  # Current attention focus areas
        self.emotional_priming = {}  # Emotion -> memory types to prioritize
        
    async def on_context_received(self, context: SharedContext):
        """Initialize recognition memory processing for this context"""
        logger.debug(f"RecognitionMemory received context for user: {context.user_id}")
        
        # Analyze context for recognition influences
        recognition_context = await self._analyze_context_for_recognition(context)
        
        # Set up contextual triggers based on current state
        contextual_triggers = await self._generate_contextual_triggers(context)
        
        # Send initial recognition context to other modules
        await self.send_context_update(
            update_type="recognition_context_available",
            data={
                "recognition_sensitivity": recognition_context.get("sensitivity", 0.7),
                "active_trigger_count": len(contextual_triggers),
                "attention_areas": list(self.attention_filter),
                "emotional_priming": self.emotional_priming,
                "recognition_ready": True
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect recognition"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state affects what memories are likely to be recognized
            emotional_data = update.data
            await self._adjust_emotional_priming(emotional_data)
            
        elif update.update_type == "attention_shift":
            # Attention changes what we're likely to recognize
            attention_data = update.data
            new_focus = attention_data.get("new_focus", [])
            self.attention_filter = set(new_focus)
            
            # Adjust recognition triggers based on attention
            await self._adjust_triggers_for_attention(new_focus)
            
        elif update.update_type == "goal_progress":
            # Goal context can prime recognition of goal-relevant memories
            goal_data = update.data
            active_goals = goal_data.get("active_goals", [])
            await self._prime_goal_relevant_recognition(active_goals)
            
        elif update.update_type == "urgency_detected":
            # High urgency situations need faster recognition
            urgency_data = update.data
            urgency_level = urgency_data.get("urgency_score", 0.5)
            
            if urgency_level > 0.8:
                # Lower recognition thresholds for faster response
                await self._enable_rapid_recognition_mode()
                
        elif update.update_type == "relationship_state_change":
            # Relationship context affects social memory recognition
            relationship_data = update.data
            await self._adjust_social_recognition(relationship_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context-aware recognition"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Determine recognition parameters based on context
        recognition_params = await self._calculate_recognition_parameters(context, messages)
        
        # Process conversation turn with context-aware parameters
        if hasattr(self.original_system, 'process_conversation_turn'):
            # Adjust system parameters based on context
            self.original_system.context.max_recognitions_per_turn = recognition_params.get("max_recognitions", 3)
            self.original_system.context.recognition_cooldown = recognition_params.get("cooldown", 600)
            
            # Process with original system
            recognition_results = await self.original_system.process_conversation_turn(
                conversation_text=context.user_input,
                current_context=context.session_context
            )
        else:
            recognition_results = []
        
        # Filter results based on context
        filtered_results = await self._filter_by_context(recognition_results, context, messages)
        
        # Send recognition updates
        if filtered_results:
            await self.send_context_update(
                update_type="memories_recognized",
                data={
                    "recognized_memories": [
                        {
                            "memory_id": r.memory_id,
                            "memory_text": r.memory_text,
                            "relevance_score": r.relevance_score,
                            "activation_trigger": r.activation_trigger
                        } for r in filtered_results
                    ],
                    "recognition_count": len(filtered_results),
                    "recognition_context": recognition_params
                }
            )
        
        return {
            "recognition_results": filtered_results,
            "context_parameters": recognition_params,
            "cross_module_influence": len(messages),
            "context_aware_recognition": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze recognition patterns in context"""
        messages = await self.get_cross_module_messages()
        
        # Analyze recognition patterns
        pattern_analysis = await self._analyze_recognition_patterns(context, messages)
        
        # Analyze context influences on recognition
        influence_analysis = await self._analyze_context_influences(context, messages)
        
        # Identify recognition biases
        bias_analysis = await self._analyze_recognition_biases(context)
        
        return {
            "pattern_analysis": pattern_analysis,
            "influence_analysis": influence_analysis,
            "bias_analysis": bias_analysis,
            "recognition_health": await self._assess_recognition_health(context),
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize recognition insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Create recognition synthesis
        recognition_synthesis = {
            "recognition_summary": await self._summarize_recognitions(context),
            "recognition_insights": await self._generate_recognition_insights(context, messages),
            "memory_connections": await self._identify_memory_connections(context),
            "recognition_recommendations": await self._suggest_recognition_adjustments(context)
        }
        
        # Check if we should surface any recognized memories in response
        surface_memories = await self._should_surface_memories(context, messages)
        if surface_memories:
            await self.send_context_update(
                update_type="surface_recognized_memories",
                data={
                    "memories_to_surface": surface_memories,
                    "surface_reason": "contextually_relevant",
                    "integration_suggestions": await self._suggest_memory_integration(surface_memories, context)
                },
                priority=ContextPriority.HIGH
            )
        
        return recognition_synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_context_for_recognition(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context to determine recognition parameters"""
        sensitivity = 0.7  # Base sensitivity
        
        # Adjust based on emotional state
        if context.emotional_state:
            arousal = context.emotional_state.get("arousal", 0.5)
            # Higher arousal = more sensitive recognition
            sensitivity += (arousal - 0.5) * 0.2
        
        # Adjust based on task purpose
        if context.task_purpose == "remember":
            sensitivity += 0.2
        elif context.task_purpose == "create":
            sensitivity -= 0.1  # Less recognition during creative tasks
        
        # Adjust based on active modules
        if "memory_core" in context.active_modules:
            sensitivity += 0.1
        
        return {
            "sensitivity": max(0.3, min(1.0, sensitivity)),
            "context_factors": {
                "emotional_influence": bool(context.emotional_state),
                "task_influence": context.task_purpose,
                "module_influence": len(context.active_modules)
            }
        }
    
    async def _generate_contextual_triggers(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Generate triggers based on current context"""
        triggers = []
        
        # Extract entities from input
        words = context.user_input.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                triggers.append({
                    "trigger_type": "entity",
                    "trigger_value": word,
                    "source": "user_input"
                })
        
        # Add emotional triggers if present
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                triggers.append({
                    "trigger_type": "emotion",
                    "trigger_value": dominant_emotion[0] if isinstance(dominant_emotion, tuple) else dominant_emotion,
                    "source": "emotional_state"
                })
        
        # Add goal-based triggers
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals[:2]:  # Top 2 goals
                if "description" in goal:
                    # Extract key terms from goal
                    key_terms = [w for w in goal["description"].split() if len(w) > 4]
                    for term in key_terms[:2]:
                        triggers.append({
                            "trigger_type": "concept",
                            "trigger_value": term.lower(),
                            "source": "active_goal"
                        })
        
        return triggers
    
    async def _adjust_emotional_priming(self, emotional_data: Dict[str, Any]):
        """Adjust recognition based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name = dominant_emotion[0] if isinstance(dominant_emotion, tuple) else dominant_emotion
        
        # Map emotions to memory types they prime
        emotion_memory_priming = {
            "Joy": ["positive_experience", "achievement", "connection"],
            "Sadness": ["loss", "disappointment", "reflection"],
            "Fear": ["threat", "warning", "safety"],
            "Anger": ["conflict", "boundary", "injustice"],
            "Love": ["attachment", "care", "intimacy"],
            "Curiosity": ["discovery", "question", "exploration"]
        }
        
        self.emotional_priming = emotion_memory_priming.get(emotion_name, [])
        
        # Add contextual cues for emotional memories
        if hasattr(self.original_system, 'register_contextual_cue'):
            await self.original_system.register_contextual_cue(
                cue_type="emotion",
                cue_value=emotion_name,
                salience=0.8,
                source_text=f"Emotional state: {emotion_name}"
            )
    
    async def _calculate_recognition_parameters(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Calculate recognition parameters based on full context"""
        params = {
            "max_recognitions": 3,
            "cooldown": 600,
            "threshold_adjustment": 0.0,
            "priority_types": []
        }
        
        # Check for memory module messages
        memory_messages = messages.get("memory_core", [])
        for msg in memory_messages:
            if msg["type"] == "memory_retrieval_complete":
                # If memory core just retrieved memories, reduce recognitions
                params["max_recognitions"] = 2
                params["cooldown"] = 300  # Shorter cooldown
        
        # Check emotional context
        emotional_messages = messages.get("emotional_core", [])
        for msg in emotional_messages:
            if msg["type"] == "emotional_state_update":
                emotional_data = msg.get("data", {})
                if emotional_data.get("arousal", 0.5) > 0.7:
                    # High arousal = more recognitions
                    params["max_recognitions"] += 1
                    params["threshold_adjustment"] = -0.1
        
        # Check goal context
        goal_messages = messages.get("goal_manager", [])
        if goal_messages:
            params["priority_types"].append("goal_relevant")
        
        return params
    
    async def _filter_by_context(self, results: List[Any], context: SharedContext, messages: Dict) -> List[Any]:
        """Filter recognition results based on context"""
        if not results:
            return results
        
        filtered = []
        
        for result in results:
            # Check attention filter
            if self.attention_filter:
                # Check if memory relates to attention areas
                memory_text = result.memory_text.lower()
                if any(area.lower() in memory_text for area in self.attention_filter):
                    # Boost relevance for attention-matched memories
                    result.relevance_score = min(1.0, result.relevance_score * 1.2)
                    filtered.append(result)
                    continue
            
            # Check emotional priming
            if self.emotional_priming:
                memory_type = getattr(result, 'memory_type', '')
                if any(primed_type in memory_type for primed_type in self.emotional_priming):
                    filtered.append(result)
                    continue
            
            # Default filtering by relevance
            if result.relevance_score > 0.6:
                filtered.append(result)
        
        # Sort by relevance
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply max recognitions from parameters
        max_recognitions = self.original_system.context.max_recognitions_per_turn
        return filtered[:max_recognitions]
    
    async def _enable_rapid_recognition_mode(self):
        """Enable rapid recognition for urgent situations"""
        if hasattr(self.original_system, 'context'):
            # Reduce cooldown for faster re-recognition
            self.original_system.context.recognition_cooldown = 60  # 1 minute instead of 10
            # Increase max recognitions
            self.original_system.context.max_recognitions_per_turn = 5
            # Lower trigger thresholds
            for trigger in self.original_system.context.active_triggers.values():
                trigger.relevance_threshold = max(0.5, trigger.relevance_threshold - 0.2)
    
    async def _analyze_recognition_patterns(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze patterns in recognition behavior"""
        patterns = {
            "recognition_frequency": "normal",
            "trigger_effectiveness": {},
            "memory_type_distribution": {},
            "temporal_patterns": []
        }
        
        if hasattr(self.original_system, 'context'):
            # Analyze recent recognitions
            recent = self.original_system.context.recent_recognitions
            if len(recent) > 5:
                # Check frequency
                time_diffs = []
                for i in range(1, len(recent)):
                    diff = (recent[i][1] - recent[i-1][1]).total_seconds()
                    time_diffs.append(diff)
                
                avg_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                if avg_diff < 120:  # Less than 2 minutes average
                    patterns["recognition_frequency"] = "high"
                elif avg_diff > 600:  # More than 10 minutes average
                    patterns["recognition_frequency"] = "low"
            
            # Analyze trigger effectiveness
            for trigger_id, trigger in self.original_system.context.active_triggers.items():
                patterns["trigger_effectiveness"][trigger.trigger_type] = {
                    "activations": getattr(trigger, 'activation_count', 0),
                    "threshold": trigger.relevance_threshold
                }
        
        return patterns
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
