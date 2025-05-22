# nyx/core/a2a/context_aware_context_system.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority
from nyx.core.context_awareness import InteractionContext, ContextDistribution

logger = logging.getLogger(__name__)

class ContextAwareContextSystem(ContextAwareModule):
    """
    Enhanced ContextAwarenessSystem with A2A context distribution capabilities
    """
    
    def __init__(self, original_context_system):
        super().__init__("context_awareness")
        self.original_system = original_context_system
        self.context_subscriptions = [
            "user_input", "emotional_state_update", "goal_context_available",
            "relationship_state_change", "memory_retrieval_complete",
            "conditioning_triggered", "mode_change"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize context awareness for this interaction"""
        logger.debug(f"ContextAwareness received initial context")
        
        # Process the user input to detect context
        context_result = await self.original_system.process_message(context.user_input)
        
        # Send the detected context to all other modules
        await self.send_context_update(
            update_type="context_detection_update",
            data={
                "context_distribution": context_result["context_distribution"],
                "primary_context": context_result["primary_context"],
                "active_contexts": context_result["active_contexts"],
                "confidence": context_result["overall_confidence"],
                "context_changed": context_result.get("context_changed", False)
            },
            priority=ContextPriority.CRITICAL  # Context detection is critical
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that might affect context detection"""
        
        if update.update_type == "emotional_state_update":
            # Strong emotions might indicate context
            emotional_data = update.data
            await self._adjust_context_from_emotion(emotional_data)
        
        elif update.update_type == "goal_context_available":
            # Active goals influence context
            goal_data = update.data
            await self._adjust_context_from_goals(goal_data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect interaction context
            relationship_data = update.data
            await self._adjust_context_from_relationship(relationship_data)
        
        elif update.update_type == "conditioning_triggered":
            # Triggered conditioning might indicate context
            conditioning_data = update.data
            await self._infer_context_from_conditioning(conditioning_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with enhanced context awareness"""
        # Get cross-module signals for context detection
        messages = await self.get_cross_module_messages()
        
        # Detect context with cross-module awareness
        detection_result = await self._detect_context_with_modules(context, messages)
        
        # Update the system's context distribution
        self.original_system.context_distribution = detection_result["distribution"]
        self.original_system.overall_confidence = detection_result["confidence"]
        
        # Send refined context update
        await self.send_context_update(
            update_type="context_refined",
            data={
                "context_distribution": detection_result["distribution"].dict(),
                "confidence": detection_result["confidence"],
                "cross_module_signals": detection_result["module_signals"]
            }
        )
        
        return {
            "context_detected": True,
            "distribution": detection_result["distribution"].dict(),
            "confidence": detection_result["confidence"],
            "module_enhanced": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context coherence across modules"""
        messages = await self.get_cross_module_messages()
        
        # Analyze context coherence
        coherence_analysis = await self._analyze_cross_module_coherence(context, messages)
        
        # Analyze context stability
        stability_analysis = await self._analyze_context_stability()
        
        # Identify context conflicts
        conflicts = await self._identify_context_conflicts(messages)
        
        return {
            "coherence_analysis": coherence_analysis,
            "stability_analysis": stability_analysis,
            "context_conflicts": conflicts,
            "recommendation": await self._recommend_context_adjustments(coherence_analysis, conflicts)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize context recommendations for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine optimal context blend for response
        optimal_blend = await self._calculate_optimal_context_blend(context, messages)
        
        # Suggest context-appropriate response characteristics
        response_suggestions = {
            "tone_distribution": await self._suggest_tone_distribution(optimal_blend),
            "formality_level": await self._calculate_formality_level(optimal_blend),
            "emotional_expression": await self._suggest_emotional_expression(optimal_blend),
            "interaction_style": await self._determine_interaction_style(optimal_blend)
        }
        
        # Send synthesis results
        await self.send_context_update(
            update_type="context_synthesis",
            data={
                "optimal_blend": optimal_blend.dict(),
                "response_suggestions": response_suggestions
            },
            priority=ContextPriority.HIGH
        )
        
        return {
            "optimal_context_blend": optimal_blend.dict(),
            "response_suggestions": response_suggestions,
            "synthesis_complete": True
        }
    
    # Helper methods
    
    async def _detect_context_with_modules(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Detect context using cross-module signals"""
        # Start with base detection
        base_result = await self.original_system.process_message(context.user_input)
        base_distribution = ContextDistribution(**base_result["context_distribution"])
        
        module_signals = []
        
        # Enhance with emotional signals
        if "emotional_core" in messages:
            emotional_distribution = await self._infer_distribution_from_emotions(messages["emotional_core"])
            base_distribution = base_distribution.blend_with(emotional_distribution, 0.2)
            module_signals.append({"source": "emotional_core", "influence": 0.2})
        
        # Enhance with goal signals
        if "goal_manager" in messages:
            goal_distribution = await self._infer_distribution_from_goals(messages["goal_manager"])
            base_distribution = base_distribution.blend_with(goal_distribution, 0.15)
            module_signals.append({"source": "goal_manager", "influence": 0.15})
        
        # Enhance with memory signals
        if "memory_core" in messages:
            memory_distribution = await self._infer_distribution_from_memory(messages["memory_core"])
            base_distribution = base_distribution.blend_with(memory_distribution, 0.1)
            module_signals.append({"source": "memory_core", "influence": 0.1})
        
        # Normalize the final distribution
        final_distribution = base_distribution.normalize()
        
        # Calculate enhanced confidence
        base_confidence = base_result["overall_confidence"]
        module_boost = len(module_signals) * 0.1
        final_confidence = min(1.0, base_confidence + module_boost)
        
        return {
            "distribution": final_distribution,
            "confidence": final_confidence,
            "module_signals": module_signals,
            "base_confidence": base_confidence
        }
    
    async def _infer_distribution_from_emotions(self, emotional_messages: List[Dict]) -> ContextDistribution:
        """Infer context distribution from emotional state"""
        distribution = ContextDistribution()
        
        for msg in emotional_messages:
            if msg["type"] == "emotional_state_update":
                emotion_data = msg["data"]
                dominant = emotion_data.get("dominant_emotion")
                
                if dominant:
                    emotion_name, strength = dominant
                    
                    # Map emotions to context weights
                    if emotion_name in ["Frustration", "Anger"]:
                        distribution = distribution.increase_context("dominant", strength * 0.5)
                    elif emotion_name in ["Joy", "Excitement"]:
                        distribution = distribution.increase_context("playful", strength * 0.6)
                    elif emotion_name in ["Sadness", "Fear"]:
                        distribution = distribution.increase_context("empathic", strength * 0.7)
                    elif emotion_name in ["Curiosity"]:
                        distribution = distribution.increase_context("intellectual", strength * 0.8)
        
        return distribution
    
    async def _analyze_cross_module_coherence(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze if context is coherent across modules"""
        coherence_scores = {}
        
        # Check if emotional state matches detected context
        if context.emotional_state:
            emotional_coherence = self._calculate_emotional_context_coherence(
                self.original_system.context_distribution,
                context.emotional_state
            )
            coherence_scores["emotional"] = emotional_coherence
        
        # Check if goals match detected context
        if context.goal_context:
            goal_coherence = self._calculate_goal_context_coherence(
                self.original_system.context_distribution,
                context.goal_context
            )
            coherence_scores["goals"] = goal_coherence
        
        # Calculate overall coherence
        if coherence_scores:
            overall_coherence = sum(coherence_scores.values()) / len(coherence_scores)
        else:
            overall_coherence = 0.7  # Default moderate coherence
        
        return {
            "overall_coherence": overall_coherence,
            "module_coherence": coherence_scores,
            "is_coherent": overall_coherence > 0.6
        }
    
    def _calculate_emotional_context_coherence(self, distribution: ContextDistribution, 
                                             emotional_state: Dict) -> float:
        """Calculate coherence between context and emotional state"""
        valence = emotional_state.get("valence", 0)
        arousal = emotional_state.get("arousal", 0)
        
        # High arousal + positive valence suggests playful/dominant contexts
        if arousal > 0.6 and valence > 0.5:
            expected_contexts = ["playful", "dominant"]
            coherence = sum(getattr(distribution, ctx, 0) for ctx in expected_contexts)
        # Low arousal + positive valence suggests casual/intellectual
        elif arousal < 0.4 and valence > 0.5:
            expected_co
