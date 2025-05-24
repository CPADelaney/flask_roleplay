# nyx/core/a2a/context_aware_somatosensory_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareDigitalSomatosensorySystem(ContextAwareModule):
    """
    Enhanced DigitalSomatosensorySystem with full context distribution capabilities
    """
    
    def __init__(self, original_somatosensory_system):
        super().__init__("digital_somatosensory_system")
        self.original_system = original_somatosensory_system
        self.context_subscriptions = [
            "emotional_state_update", "memory_trigger", "relationship_intimacy_change",
            "action_physical_contact", "environmental_change", "arousal_relevant_content",
            "mode_state_change", "needs_physical_expression"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize somatosensory processing for this context"""
        logger.debug(f"SomatosensorySystem received context for user: {context.user_id}")
        
        # Analyze input for physical/sensory content
        sensory_analysis = await self._analyze_sensory_content(context.user_input)
        
        # Get current body state
        body_state = await self.original_system.get_body_state()
        
        # Check for arousal relevance
        arousal_context = await self._assess_arousal_context(context)
        
        # Send initial sensory context to other modules
        await self.send_context_update(
            update_type="sensory_state_available",
            data={
                "body_state": body_state,
                "sensory_analysis": sensory_analysis,
                "arousal_state": {
                    "level": self.original_system.arousal_state.arousal_level,
                    "physical": self.original_system.arousal_state.physical_arousal,
                    "cognitive": self.original_system.arousal_state.cognitive_arousal,
                    "in_afterglow": self.original_system.is_in_afterglow()
                },
                "temperature": self.original_system.temperature_model["body_temperature"],
                "arousal_context": arousal_context
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect physical sensations"""
        
        if update.update_type == "emotional_state_update":
            # Strong emotions can trigger physical sensations
            emotional_data = update.data
            await self._process_emotional_physical_response(emotional_data)
            
        elif update.update_type == "memory_trigger":
            # Memories can trigger associated physical sensations
            memory_data = update.data
            if memory_data.get("memory_id"):
                trigger_result = await self.original_system.process_trigger(
                    memory_data.get("memory_text", "")[:50]
                )
                if trigger_result.get("triggered_responses"):
                    await self.send_context_update(
                        update_type="physical_memory_triggered",
                        data=trigger_result,
                        target_modules=["memory_core"],
                        scope=ContextScope.TARGETED
                    )
                    
        elif update.update_type == "relationship_intimacy_change":
            # Intimacy changes affect arousal sensitivity
            relationship_data = update.data
            await self._update_arousal_sensitivity(relationship_data)
            
        elif update.update_type == "action_physical_contact":
            # Physical contact from actions
            contact_data = update.data
            await self._process_physical_contact(contact_data)
            
        elif update.update_type == "arousal_relevant_content":
            # Content that might affect arousal
            content_data = update.data
            await self._process_arousal_content(content_data)
            
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for physical sensations and responses"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Check if input describes physical sensations
        harm_check = await self.original_system.analyze_text_for_harmful_content(context.user_input)
        
        if harm_check.get("status") == "safe":
            # Process any described sensations
            sensation_result = await self._process_described_sensations(context.user_input)
            
            # Update arousal if relevant
            if self._contains_arousal_content(context.user_input):
                arousal_result = await self.original_system.process_cognitive_arousal(
                    stimulus=context.user_input[:50],
                    partner_id=context.user_id,
                    context=str(context.session_context),
                    intensity=0.4
                )
                
                # Send arousal update
                await self.send_context_update(
                    update_type="arousal_state_change",
                    data={
                        "arousal_result": arousal_result,
                        "current_arousal": self.original_system.arousal_state.arousal_level
                    }
                )
        
        # Get sensory influence for response
        sensory_influence = await self.original_system.get_sensory_influence(context.user_input)
        
        return {
            "harm_check": harm_check,
            "sensory_influence": sensory_influence,
            "processing_complete": True,
            "body_awareness_active": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze physical state in context"""
        messages = await self.get_cross_module_messages()
        
        # Comprehensive body state analysis
        body_analysis = await self.original_system.get_body_state()
        
        # Temperature effects analysis
        temperature_effects = await self.original_system.get_temperature_effects()
        
        # Arousal pattern analysis
        arousal_patterns = self._analyze_arousal_patterns()
        
        # Physical-emotional correlation analysis
        physical_emotional_correlation = await self._analyze_physical_emotional_correlation(context, messages)
        
        # Comfort and tension analysis
        comfort_analysis = self._analyze_comfort_patterns()
        
        return {
            "body_analysis": body_analysis,
            "temperature_effects": temperature_effects,
            "arousal_patterns": arousal_patterns,
            "physical_emotional_correlation": physical_emotional_correlation,
            "comfort_analysis": comfort_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize physical expression elements for response"""
        messages = await self.get_cross_module_messages()
        
        # Get sensory expressions to include
        sensory_influence = await self.original_system.get_sensory_influence("")
        
        # Determine physical behavior modifiers
        behavior_modifiers = await self._get_physical_behavior_modifiers(context)
        
        # Get arousal expression modifiers
        arousal_modifiers = self.original_system.get_arousal_expression_modifier(
            partner_id=context.user_id
        )
        
        # Generate physical action suggestions
        physical_actions = await self._suggest_physical_actions(context, messages)
        
        synthesis = {
            "include_sensory_expression": sensory_influence.get("should_express", False),
            "sensory_expressions": sensory_influence.get("expressions", []),
            "behavior_modifiers": behavior_modifiers,
            "arousal_expression": arousal_modifiers,
            "suggested_physical_actions": physical_actions,
            "posture_effect": behavior_modifiers.get("posture"),
            "movement_quality": behavior_modifiers.get("movement")
        }
        
        # Send high arousal notification if needed
        if self.original_system.arousal_state.arousal_level > 0.8:
            await self.send_context_update(
                update_type="high_arousal_state",
                data={
                    "arousal_level": self.original_system.arousal_state.arousal_level,
                    "expression_hints": arousal_modifiers,
                    "requires_expression": True
                },
                priority=ContextPriority.HIGH
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_sensory_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for sensory/physical content"""
        text_lower = text.lower()
        
        analysis = {
            "contains_touch": any(word in text_lower for word in ["touch", "feel", "caress", "stroke"]),
            "contains_temperature": any(word in text_lower for word in ["hot", "cold", "warm", "cool"]),
            "contains_pain": any(word in text_lower for word in ["hurt", "pain", "ache", "sore"]),
            "contains_pleasure": any(word in text_lower for word in ["pleasure", "good", "nice", "enjoy"]),
            "contains_physical": any(word in text_lower for word in ["body", "physical", "skin", "flesh"])
        }
        
        analysis["has_sensory_content"] = any(analysis.values())
        return analysis
    
    async def _assess_arousal_context(self, context: SharedContext) -> Dict[str, Any]:
        """Assess if context is arousal-relevant"""
        arousal_indicators = 0
        
        # Check relationship intimacy
        if context.relationship_context:
            intimacy = context.relationship_context.get("intimacy", 0)
            if intimacy > 0.6:
                arousal_indicators += 1
                
        # Check emotional state for relevant emotions
        if context.emotional_state:
            arousing_emotions = ["Love", "Desire", "Excitement"]
            for emotion in arousing_emotions:
                if context.emotional_state.get(emotion, 0) > 0.5:
                    arousal_indicators += 1
                    
        # Check session context
        if context.session_context.get("intimate_context"):
            arousal_indicators += 2
            
        return {
            "is_arousal_relevant": arousal_indicators > 1,
            "arousal_indicators": arousal_indicators,
            "intimacy_level": context.relationship_context.get("intimacy", 0) if context.relationship_context else 0
        }
    
    async def _process_emotional_physical_response(self, emotional_data: Dict[str, Any]):
        """Process physical responses to emotions"""
        emotional_state = emotional_data.get("emotional_state", {})
        
        # Map strong emotions to physical sensations
        emotion_physical_map = {
            "Anxiety": [("tingling", "hands", 0.4), ("pressure", "chest", 0.3)],
            "Excitement": [("tingling", "skin", 0.3), ("temperature", "face", 0.7)],
            "Love": [("pleasure", "chest", 0.3), ("temperature", "skin", 0.6)],
            "Fear": [("temperature", "skin", 0.2), ("tingling", "spine", 0.5)]
        }
        
        for emotion, strength in emotional_state.items():
            if emotion in emotion_physical_map and strength > 0.6:
                for sensation_type, region, intensity in emotion_physical_map[emotion]:
                    await self.original_system.process_stimulus(
                        sensation_type, region, intensity * strength,
                        cause=f"emotional_response_{emotion.lower()}", 
                        duration=2.0
                    )
    
    def _contains_arousal_content(self, text: str) -> bool:
        """Check if text contains arousal-relevant content"""
        arousal_keywords = [
            "intimate", "touch", "caress", "kiss", "hold", "embrace",
            "desire", "want", "need", "aroused", "excited", "close"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in arousal_keywords)
    
    async def _process_described_sensations(self, text: str) -> Dict[str, Any]:
        """Process sensations described in text"""
        # This would use more sophisticated NLP in production
        results = []
        
        # Simple pattern matching for demonstration
        if "touch" in text.lower() or "feel" in text.lower():
            # Infer a gentle touch sensation
            result = await self.original_system.process_stimulus(
                "pressure", "skin", 0.3, "described_touch", 1.0
            )
            results.append(result)
            
        return {"processed_sensations": results}
    
    def _analyze_arousal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in arousal history"""
        if not hasattr(self.original_system.arousal_state, 'arousal_history'):
            return {}
            
        history = self.original_system.arousal_state.arousal_history
        if not history:
            return {"no_history": True}
            
        # Calculate average, peak, and trend
        values = [h[1] for h in history[-20:]]  # Last 20 entries
        
        return {
            "average_arousal": sum(values) / len(values) if values else 0,
            "peak_arousal": max(values) if values else 0,
            "current_trend": "increasing" if len(values) > 1 and values[-1] > values[-2] else "stable",
            "time_since_peak": (datetime.now() - self.original_system.arousal_state.peak_time).total_seconds() 
                             if self.original_system.arousal_state.peak_time else None
        }
    
    async def _update_arousal_sensitivity(self, relationship_data: Dict[str, Any]):
        """Update arousal sensitivity based on relationship intimacy"""
        intimacy = relationship_data.get("intimacy", 0.5)
        trust = relationship_data.get("trust", 0.5)
        partner_id = relationship_data.get("user_id")
        
        if partner_id:
            # Update partner affinity and emotional connection
            combined_score = (intimacy + trust) / 2
            self.original_system.set_partner_affinity(partner_id, combined_score)
            self.original_system.set_partner_emoconn(partner_id, intimacy)
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
