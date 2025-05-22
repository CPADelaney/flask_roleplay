# nyx/core/a2a/context_aware_emotional_core.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareEmotionalCore(ContextAwareModule):
    """
    Context-aware wrapper for EmotionalCore with full A2A integration
    """
    
    def __init__(self, original_emotional_core):
        super().__init__("emotional_core")
        self.original_core = original_emotional_core
        self.context_subscriptions = [
            "goal_context_available", "goal_progress", "goal_completion_announcement",
            "memory_context_available", "memory_retrieval_complete",
            "relationship_state_change", "relationship_milestone",
            "mode_adjustment", "attention_focus_change",
            "hormone_influence_update", "hormone_cycle_complete",
            "user_input_analysis", "response_generation_request"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize emotional processing for this context"""
        logger.debug(f"EmotionalCore received context for user: {context.user_id}")
        
        # Analyze emotional implications of the input
        emotional_analysis = await self._analyze_input_emotions(context.user_input)
        
        # Get current emotional state
        current_state = await self._get_current_emotional_state()
        
        # Check for hormone influences
        hormone_influences = await self._check_hormone_influences()
        
        # Send initial emotional context to other modules
        await self.send_context_update(
            update_type="emotional_state_available",
            data={
                "current_state": current_state,
                "input_emotions": emotional_analysis,
                "dominant_emotion": current_state.get("primary_emotion", {}).get("name", "Neutral"),
                "intensity": current_state.get("primary_emotion", {}).get("intensity", 0.5),
                "valence": current_state.get("valence", 0.0),
                "arousal": current_state.get("arousal", 0.5),
                "hormone_influences": hormone_influences,
                "neurochemicals": self._get_neurochemical_summary()
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "goal_progress":
            # Goal progress affects emotions
            goal_data = update.data
            if goal_data.get("goals_executed", 0) > 0:
                # Success in goal execution increases positive chemicals
                await self._adjust_neurochemicals_for_success()
            
        elif update.update_type == "goal_completion_announcement":
            # Goal completion triggers satisfaction
            completed_goals = update.data.get("completed_goals", [])
            if completed_goals:
                await self._process_goal_completion_emotions(completed_goals)
                
        elif update.update_type == "memory_retrieval_complete":
            # Memories can trigger emotional responses
            memory_data = update.data
            emotional_memories = memory_data.get("retrieved_memories", [])
            await self._process_emotional_memories(emotional_memories)
            
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect oxytocin and trust-related emotions
            relationship_data = update.data
            await self._adjust_for_relationship_state(relationship_data)
            
        elif update.update_type == "hormone_influence_update":
            # Apply hormone influences to neurochemicals
            hormone_data = update.data
            await self._apply_hormone_influences(hormone_data)
            
        elif update.update_type == "attention_focus_change":
            # Attention changes can affect arousal
            attention_data = update.data
            await self._adjust_arousal_from_attention(attention_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with emotional awareness"""
        # Process through original emotional core
        result = await self.original_core.process_emotional_input(context.user_input)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Enhance emotional processing with context
        enhanced_result = await self._enhance_with_context(result, messages)
        
        # Update emotional state in context
        await self.send_context_update(
            update_type="emotional_processing_complete",
            data={
                "processed_emotions": enhanced_result,
                "neurochemical_changes": self._get_neurochemical_changes(),
                "emotion_transitions": self._detect_emotion_transitions(),
                "context_influences": self._analyze_context_influences(messages)
            }
        )
        
        return enhanced_result
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze emotional patterns in context"""
        # Get emotion history
        emotion_history = self.original_core.emotional_state_history
        
        # Analyze patterns
        patterns = await self._analyze_emotional_patterns(emotion_history)
        
        # Check coherence with other modules
        coherence = await self._check_emotional_coherence(context)
        
        # Predict emotional trajectory
        trajectory = await self._predict_emotional_trajectory(context)
        
        return {
            "emotional_patterns": patterns,
            "coherence_analysis": coherence,
            "predicted_trajectory": trajectory,
            "stability_index": self._calculate_emotional_stability(),
            "recommendation": self._suggest_emotional_regulation(patterns, trajectory)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize emotional components for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate emotional coloring for response
        emotional_tone = await self._determine_response_tone(context)
        
        # Check if we need to express specific emotions
        expression_needs = await self._analyze_expression_needs(context, messages)
        
        # Generate emotional markers
        emotional_markers = {
            "tone": emotional_tone,
            "expression_level": self._calculate_expression_level(context),
            "emotional_words": self._suggest_emotional_vocabulary(emotional_tone),
            "nonverbal_cues": self._suggest_nonverbal_expressions(emotional_tone),
            "intensity_modulation": self._calculate_intensity_modulation(context)
        }
        
        # Check for emotional urgency
        if self._detect_emotional_urgency(context):
            await self.send_context_update(
                update_type="emotional_urgency_detected",
                data={
                    "urgency_type": "emotional_expression",
                    "required_emotions": expression_needs,
                    "urgency_level": self._calculate_urgency_level()
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "emotional_synthesis": emotional_markers,
            "expression_requirements": expression_needs,
            "coherence_verified": True
        }
    
    # Helper methods
    async def _analyze_input_emotions(self, user_input: str) -> Dict[str, Any]:
        """Analyze emotional content in user input"""
        # Use emotion tools to analyze text
        from nyx.core.emotions.tools.emotion_tools import EmotionTools
        emotion_tools = EmotionTools(self.original_core)
        
        # Create a mock context for the tool
        from nyx.core.emotions.context import EmotionalContext
        from agents import RunContextWrapper
        
        mock_context = EmotionalContext()
        mock_context.set_value("emotion_tools_instance", emotion_tools)
        ctx = RunContextWrapper(context=mock_context)
        
        analysis = await emotion_tools.analyze_text_sentiment(ctx, user_input)
        
        return {
            "detected_emotions": analysis.derived_emotions,
            "dominant_emotion": analysis.dominant_emotion,
            "intensity": analysis.intensity,
            "valence": analysis.valence,
            "chemicals_affected": analysis.chemicals_affected
        }
    
    def _get_neurochemical_summary(self) -> Dict[str, float]:
        """Get current neurochemical levels"""
        return {
            chem: data["value"] 
            for chem, data in self.original_core.neurochemicals.items()
        }
    
    async def _get_current_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state matrix"""
        return self.original_core._get_emotional_state_matrix_sync()
    
    async def _check_hormone_influences(self) -> Dict[str, float]:
        """Check current hormone influences"""
        if hasattr(self.original_core, 'hormone_influences'):
            return self.original_core.hormone_influences
        return {}
    
    async def _adjust_neurochemicals_for_success(self):
        """Adjust chemicals for success/achievement"""
        # Boost dopamine (nyxamine) for reward
        self.original_core.neurochemicals["nyxamine"]["value"] = min(
            1.0, self.original_core.neurochemicals["nyxamine"]["value"] + 0.15
        )
        
        # Slight serotonin boost for satisfaction
        self.original_core.neurochemicals["seranix"]["value"] = min(
            1.0, self.original_core.neurochemicals["seranix"]["value"] + 0.1
        )
    
    async def _process_goal_completion_emotions(self, completed_goals: List[Dict]):
        """Process emotions from completed goals"""
        # Major achievement - significant neurochemical response
        for goal in completed_goals:
            priority = goal.get("priority", 0.5)
            
            # Scale response by goal priority
            nyxamine_boost = 0.2 * priority
            seranix_boost = 0.15 * priority
            oxynixin_boost = 0.1 * priority  # Social reward aspect
            
            self.original_core.neurochemicals["nyxamine"]["value"] = min(
                1.0, self.original_core.neurochemicals["nyxamine"]["value"] + nyxamine_boost
            )
            self.original_core.neurochemicals["seranix"]["value"] = min(
                1.0, self.original_core.neurochemicals["seranix"]["value"] + seranix_boost
            )
            self.original_core.neurochemicals["oxynixin"]["value"] = min(
                1.0, self.original_core.neurochemicals["oxynixin"]["value"] + oxynixin_boost
            )
    
    # Delegate missing methods to original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original core"""
        return getattr(self.original_core, name)
