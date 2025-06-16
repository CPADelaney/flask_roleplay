# nyx/core/a2a/context_aware_emotional_core.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareEmotionalCore(ContextAwareModule):
    """
    Context-aware wrapper for EmotionalCore with full A2A integration.
    This wrapper adapts the EmotionalCore to work with the A2A SharedContext system.
    """
    
    def __init__(self, original_emotional_core):
        """
        Initialize the context-aware wrapper.
        
        Args:
            original_emotional_core: Instance of EmotionalCore from nyx.core.emotions
        """
        super().__init__("emotional_core")
        self.emotional_core = original_emotional_core
        self.context_subscriptions = [
            "goal_context_available", "goal_progress", "goal_completion_announcement",
            "memory_context_available", "memory_retrieval_complete",
            "relationship_state_change", "relationship_milestone",
            "mode_adjustment", "attention_focus_change",
            "hormone_influence_update", "hormone_cycle_complete",
            "user_input_analysis", "response_generation_request"
        ]
        
        # Track last processed input to avoid reprocessing
        self._last_processed_input = None
        self._last_process_time = None
    
    async def on_context_received(self, context: SharedContext):
        """Initialize emotional processing for this context"""
        logger.debug(f"EmotionalCore received context for user: {context.user_id}")
        
        # Analyze emotional implications of the input
        emotional_analysis = await self._analyze_input_emotions(context.user_input)
        
        # Get current emotional state
        current_state = await self._get_current_emotional_state()
        
        # Check for hormone influences
        hormone_influences = self._get_hormone_influences()
        
        # Send initial emotional context to other modules
        await self.send_context_update(
            update_type="emotional_state_available",
            data={
                "current_state": current_state,
                "input_emotions": emotional_analysis,
                "dominant_emotion": current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("name", "Neutral"),
                "intensity": current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("intensity", 0.5),
                "valence": current_state.get("emotional_state_matrix", {}).get("valence", 0.0),
                "arousal": current_state.get("emotional_state_matrix", {}).get("arousal", 0.5),
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
        # Avoid reprocessing the same input
        if (self._last_processed_input == context.user_input and 
            self._last_process_time and 
            (datetime.now() - self._last_process_time).seconds < 1):
            return self._get_cached_response()
        
        # Process through emotional core
        result = await self.emotional_core.process_emotional_input(context.user_input)
        
        # Update tracking
        self._last_processed_input = context.user_input
        self._last_process_time = datetime.now()
        
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
        # Get emotion history from the actual emotional core
        emotion_history = self.emotional_core.emotional_state_history
        
        # Analyze patterns
        patterns = self._analyze_emotional_patterns(emotion_history)
        
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
        # Process the input through the emotional core
        result = await self.emotional_core.process_emotional_input(user_input)
        
        # Extract relevant emotional data
        return {
            "detected_emotions": result.get("primary_emotion", {}).get("name", "Neutral"),
            "dominant_emotion": result.get("primary_emotion", {}).get("name", "Neutral"),
            "intensity": result.get("intensity", 0.5),
            "valence": result.get("valence", 0.0),
            "arousal": result.get("arousal", 0.5),
            "chemicals_affected": result.get("neurochemical_changes", {})
        }
    
    def _get_neurochemical_summary(self) -> Dict[str, float]:
        """Get current neurochemical levels"""
        return {
            chem: data["value"] 
            for chem, data in self.emotional_core.neurochemicals.items()
        }
    
    async def _get_current_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return await self.emotional_core.get_emotional_state()
    
    def _get_hormone_influences(self) -> Dict[str, float]:
        """Get current hormone influences"""
        if hasattr(self.emotional_core, 'hormone_influences'):
            return self.emotional_core.hormone_influences
        return {}
    
    async def _adjust_neurochemicals_for_success(self):
        """Adjust chemicals for success/achievement"""
        # Use the emotional core's update method
        await self.emotional_core.update_neurochemical("nyxamine", 0.15, "goal_success")
        await self.emotional_core.update_neurochemical("seranix", 0.1, "goal_success")
    
    async def _process_goal_completion_emotions(self, completed_goals: List[Dict]):
        """Process emotions from completed goals"""
        for goal in completed_goals:
            priority = goal.get("priority", 0.5)
            
            # Scale response by goal priority
            nyxamine_boost = 0.2 * priority
            seranix_boost = 0.15 * priority
            oxynixin_boost = 0.1 * priority  # Social reward aspect
            
            await self.emotional_core.update_neurochemical("nyxamine", nyxamine_boost, "goal_completion")
            await self.emotional_core.update_neurochemical("seranix", seranix_boost, "goal_completion")
            await self.emotional_core.update_neurochemical("oxynixin", oxynixin_boost, "goal_completion")
    
    async def _process_emotional_memories(self, memories: List[Dict]):
        """Process emotional responses from memories"""
        for memory in memories:
            # Extract emotional content from memory
            emotion = memory.get("dominant_emotion", "Neutral")
            intensity = memory.get("emotional_intensity", 0.3)
            
            # Adjust neurochemicals based on memory emotion
            if emotion in ["Joy", "Happiness", "Love"]:
                await self.emotional_core.update_neurochemical("nyxamine", 0.1 * intensity, "memory_recall")
                await self.emotional_core.update_neurochemical("oxynixin", 0.05 * intensity, "memory_recall")
            elif emotion in ["Sadness", "Grief", "Loss"]:
                await self.emotional_core.update_neurochemical("cortanyx", 0.1 * intensity, "memory_recall")
                await self.emotional_core.update_neurochemical("seranix", -0.05 * intensity, "memory_recall")
    
    async def _adjust_for_relationship_state(self, relationship_data: Dict):
        """Adjust emotions based on relationship state changes"""
        relationship_quality = relationship_data.get("quality", 0.5)
        trust_level = relationship_data.get("trust", 0.5)
        
        # Oxytocin response to relationship quality
        oxynixin_change = (relationship_quality - 0.5) * 0.2
        await self.emotional_core.update_neurochemical("oxynixin", oxynixin_change, "relationship_change")
        
        # Seranix response to trust
        seranix_change = (trust_level - 0.5) * 0.1
        await self.emotional_core.update_neurochemical("seranix", seranix_change, "trust_adjustment")
    
    async def _apply_hormone_influences(self, hormone_data: Dict):
        """Apply hormone influences to emotional state"""
        # This would integrate with the hormone system if available
        if hasattr(self.emotional_core, 'hormone_system') and self.emotional_core.hormone_system:
            # The hormone system handles its own influences
            pass
    
    async def _adjust_arousal_from_attention(self, attention_data: Dict):
        """Adjust arousal based on attention changes"""
        attention_level = attention_data.get("focus_level", 0.5)
        attention_target = attention_data.get("target", "general")
        
        # High attention increases arousal chemicals
        if attention_level > 0.7:
            await self.emotional_core.update_neurochemical("adrenyx", 0.1, "high_attention")
        elif attention_level < 0.3:
            await self.emotional_core.update_neurochemical("adrenyx", -0.1, "low_attention")
    
    def _get_cached_response(self) -> Dict[str, Any]:
        """Get cached response for duplicate requests"""
        current_state = self.emotional_core._get_emotional_state_matrix_sync()
        return {
            "primary_emotion": current_state.get("primary_emotion", {}),
            "intensity": current_state.get("primary_emotion", {}).get("intensity", 0.5),
            "response_text": "Processing...",
            "neurochemical_changes": {},
            "valence": current_state.get("valence", 0.0),
            "arousal": current_state.get("arousal", 0.5),
            "cached": True
        }
    
    async def _enhance_with_context(self, result: Dict[str, Any], messages: List[Dict]) -> Dict[str, Any]:
        """Enhance emotional result with cross-module context"""
        enhanced = result.copy()
        
        # Add context influences
        enhanced["context_influences"] = []
        
        for message in messages:
            if message.get("module") == "goal_system":
                enhanced["context_influences"].append({
                    "type": "goal_influence",
                    "impact": "positive" if message.get("progress", 0) > 0.5 else "neutral"
                })
            elif message.get("module") == "memory_system":
                enhanced["context_influences"].append({
                    "type": "memory_influence",
                    "impact": "contextual"
                })
        
        return enhanced
    
    def _get_neurochemical_changes(self) -> Dict[str, float]:
        """Calculate recent neurochemical changes"""
        changes = {}
        # This would track actual changes over time
        # For now, return current deviations from baseline
        for chem, data in self.emotional_core.neurochemicals.items():
            changes[chem] = data["value"] - data["baseline"]
        return changes
    
    def _detect_emotion_transitions(self) -> List[Dict[str, Any]]:
        """Detect recent emotion transitions"""
        transitions = []
        history = self.emotional_core.emotional_state_history
        
        if len(history) >= 2:
            for i in range(1, min(5, len(history))):
                prev = history[-i-1].get("primary_emotion", {}).get("name", "Unknown")
                curr = history[-i].get("primary_emotion", {}).get("name", "Unknown")
                
                if prev != curr:
                    transitions.append({
                        "from": prev,
                        "to": curr,
                        "index": i
                    })
        
        return transitions
    
    def _analyze_context_influences(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze how context messages influence emotions"""
        influences = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        for message in messages:
            # Simple sentiment analysis of message impact
            if any(word in str(message).lower() for word in ["success", "complete", "achieve"]):
                influences["positive"] += 1
            elif any(word in str(message).lower() for word in ["fail", "error", "problem"]):
                influences["negative"] += 1
            else:
                influences["neutral"] += 1
        
        return influences
    
    def _analyze_emotional_patterns(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in emotional history"""
        if len(history) < 2:
            return {"message": "Insufficient history for pattern analysis"}
        
        patterns = {
            "dominant_emotions": {},
            "transitions": {},
            "stability": 0.0
        }
        
        # Count emotion frequencies
        for state in history[-20:]:  # Last 20 states
            emotion = state.get("primary_emotion", {}).get("name", "Unknown")
            patterns["dominant_emotions"][emotion] = patterns["dominant_emotions"].get(emotion, 0) + 1
        
        # Calculate stability
        changes = 0
        for i in range(1, min(20, len(history))):
            if history[-i].get("primary_emotion", {}).get("name") != history[-i-1].get("primary_emotion", {}).get("name"):
                changes += 1
        
        patterns["stability"] = 1.0 - (changes / min(20, len(history)))
        
        return patterns
    
    async def _check_emotional_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Check if emotional state is coherent with context"""
        current_state = await self._get_current_emotional_state()
        current_emotion = current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("name", "Neutral")
        
        coherence = {
            "is_coherent": True,
            "factors": []
        }
        
        # Check if emotion matches input sentiment
        if hasattr(context, 'sentiment') and context.sentiment:
            expected_valence = 1 if context.sentiment == "positive" else -1 if context.sentiment == "negative" else 0
            actual_valence = current_state.get("emotional_state_matrix", {}).get("valence", 0)
            
            if abs(expected_valence - actual_valence) > 0.5:
                coherence["is_coherent"] = False
                coherence["factors"].append("Emotion doesn't match input sentiment")
        
        return coherence
    
    async def _predict_emotional_trajectory(self, context: SharedContext) -> Dict[str, Any]:
        """Predict future emotional trajectory"""
        current_state = await self._get_current_emotional_state()
        current_chemicals = self._get_neurochemical_summary()
        
        trajectory = {
            "direction": "stable",
            "confidence": 0.5,
            "factors": []
        }
        
        # Analyze chemical trends
        high_stress = current_chemicals.get("cortanyx", 0) > 0.6
        high_joy = current_chemicals.get("nyxamine", 0) > 0.6
        
        if high_stress:
            trajectory["direction"] = "negative"
            trajectory["factors"].append("High stress levels")
        elif high_joy:
            trajectory["direction"] = "positive"
            trajectory["factors"].append("High joy chemicals")
        
        return trajectory
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate overall emotional stability index"""
        history = self.emotional_core.emotional_state_history[-10:]
        
        if len(history) < 2:
            return 0.5
        
        # Calculate variance in valence
        valences = [h.get("valence", 0) for h in history]
        avg_valence = sum(valences) / len(valences)
        variance = sum((v - avg_valence) ** 2 for v in valences) / len(valences)
        
        # Lower variance = higher stability
        stability = max(0, 1 - (variance * 2))
        
        return stability
    
    def _suggest_emotional_regulation(self, patterns: Dict, trajectory: Dict) -> str:
        """Suggest emotional regulation strategy"""
        stability = patterns.get("stability", 0.5)
        direction = trajectory.get("direction", "stable")
        
        if stability < 0.3:
            return "Focus on stabilizing emotions through consistent routines"
        elif direction == "negative":
            return "Engage in positive activities to shift emotional trajectory"
        elif direction == "positive":
            return "Maintain current positive momentum"
        else:
            return "Continue balanced emotional processing"
    
    async def _determine_response_tone(self, context: SharedContext) -> str:
        """Determine appropriate emotional tone for response"""
        current_state = await self._get_current_emotional_state()
        primary_emotion = current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("name", "Neutral")
        
        tone_map = {
            "Joy": "cheerful",
            "Sadness": "empathetic",
            "Anger": "assertive",
            "Fear": "reassuring",
            "Trust": "warm",
            "Neutral": "balanced"
        }
        
        return tone_map.get(primary_emotion, "balanced")
    
    async def _analyze_expression_needs(self, context: SharedContext, messages: List[Dict]) -> List[str]:
        """Analyze what emotions need to be expressed"""
        needs = []
        
        current_state = await self._get_current_emotional_state()
        intensity = current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("intensity", 0)
        
        # High intensity emotions need expression
        if intensity > 0.7:
            emotion = current_state.get("emotional_state_matrix", {}).get("primary_emotion", {}).get("name", "Neutral")
            needs.append(f"express_{emotion.lower()}")
        
        # Check for suppressed emotions
        for chem, data in self.emotional_core.neurochemicals.items():
            if data["value"] < data["baseline"] - 0.3:
                needs.append(f"acknowledge_{chem}_deficit")
        
        return needs
    
    def _calculate_expression_level(self, context: SharedContext) -> float:
        """Calculate how much emotion to express"""
        # Base expression level on context factors
        base_level = 0.5
        
        # Adjust based on user familiarity if available
        if hasattr(context, 'user_familiarity'):
            base_level += context.user_familiarity * 0.2
        
        # Adjust based on interaction length
        if hasattr(context, 'interaction_count'):
            base_level += min(0.2, context.interaction_count * 0.02)
        
        return min(1.0, base_level)
    
    def _suggest_emotional_vocabulary(self, tone: str) -> List[str]:
        """Suggest emotion-appropriate vocabulary"""
        vocabulary_map = {
            "cheerful": ["wonderful", "delightful", "exciting", "fantastic"],
            "empathetic": ["understand", "feel", "appreciate", "recognize"],
            "assertive": ["important", "need", "must", "clearly"],
            "reassuring": ["safe", "okay", "together", "support"],
            "warm": ["care", "value", "appreciate", "grateful"],
            "balanced": ["consider", "think", "perhaps", "might"]
        }
        
        return vocabulary_map.get(tone, vocabulary_map["balanced"])
    
    def _suggest_nonverbal_expressions(self, tone: str) -> List[str]:
        """Suggest nonverbal emotional expressions"""
        expression_map = {
            "cheerful": ["*smiles*", "*brightens*", "*excited energy*"],
            "empathetic": ["*gentle nod*", "*soft expression*", "*understanding look*"],
            "assertive": ["*firm stance*", "*direct gaze*", "*confident posture*"],
            "reassuring": ["*calming presence*", "*steady voice*", "*supportive gesture*"],
            "warm": ["*warm smile*", "*open posture*", "*welcoming energy*"],
            "balanced": ["*thoughtful pause*", "*considered response*", "*neutral presence*"]
        }
        
        return expression_map.get(tone, expression_map["balanced"])
    
    def _calculate_intensity_modulation(self, context: SharedContext) -> float:
        """Calculate how to modulate emotional intensity"""
        # Start with current intensity
        current_state = self.emotional_core._get_emotional_state_matrix_sync()
        base_intensity = current_state.get("primary_emotion", {}).get("intensity", 0.5)
        
        # Modulate based on context
        if hasattr(context, 'requires_calm') and context.requires_calm:
            return base_intensity * 0.7  # Reduce intensity
        elif hasattr(context, 'requires_energy') and context.requires_energy:
            return min(1.0, base_intensity * 1.3)  # Increase intensity
        
        return base_intensity
    
    def _detect_emotional_urgency(self, context: SharedContext) -> bool:
        """Detect if there's emotional urgency requiring immediate response"""
        # Check for extreme chemical levels
        chemicals = self._get_neurochemical_summary()
        
        # High stress or very low mood
        if chemicals.get("cortanyx", 0) > 0.8 or chemicals.get("seranix", 0) < 0.2:
            return True
        
        # Check for rapid emotional changes
        if len(self.emotional_core.emotional_state_history) >= 2:
            recent = self.emotional_core.emotional_state_history[-2:]
            valence_change = abs(recent[1].get("valence", 0) - recent[0].get("valence", 0))
            if valence_change > 0.5:
                return True
        
        return False
    
    def _calculate_urgency_level(self) -> float:
        """Calculate the level of emotional urgency"""
        urgency = 0.0
        
        # Check chemical extremes
        for chem, data in self.emotional_core.neurochemicals.items():
            deviation = abs(data["value"] - data["baseline"])
            if deviation > 0.4:
                urgency = max(urgency, deviation)
        
        return min(1.0, urgency)
    
    # Delegate any other method calls to the emotional core
    def __getattr__(self, name):
        """Delegate any missing methods to the emotional core"""
        return getattr(self.emotional_core, name)
