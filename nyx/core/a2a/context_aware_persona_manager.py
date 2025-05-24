# nyx/core/a2a/context_aware_persona_manager.py

import logging
from typing import Dict, List, Any, Optional

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwarePersonaManager(ContextAwareModule):
    """
    Enhanced Persona Manager with full context distribution capabilities
    """
    
    def __init__(self, original_persona_manager):
        super().__init__("persona_manager")
        self.original_manager = original_persona_manager
        self.context_subscriptions = [
            "emotional_state_update", "dominance_level_change", "relationship_state_change",
            "submission_metric_update", "user_interaction", "femdom_mode_change",
            "scenario_change", "mood_update", "goal_context_available",
            "memory_retrieval_complete", "intensity_adjustment_needed"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize persona processing for this context"""
        logger.debug(f"PersonaManager received context for user: {context.user_id}")
        
        # Get active persona if any
        active_persona = await self.original_manager.get_active_persona(context.user_id)
        
        # Analyze context for persona relevance
        persona_analysis = await self._analyze_persona_context(context, active_persona)
        
        # Send initial persona context
        await self.send_context_update(
            update_type="persona_context_available",
            data={
                "user_id": context.user_id,
                "active_persona": active_persona.get("persona_id") if active_persona.get("active") else None,
                "persona_name": active_persona.get("persona_name") if active_persona.get("active") else None,
                "intensity": active_persona.get("intensity", 0.5) if active_persona.get("active") else None,
                "persona_analysis": persona_analysis,
                "persona_suitable": persona_analysis.get("current_suitable", True)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates affecting persona management"""
        
        if update.update_type == "emotional_state_update":
            # Adjust persona based on emotional state
            emotional_data = update.data
            await self._adjust_persona_from_emotion(emotional_data)
        
        elif update.update_type == "dominance_level_change":
            # Adjust persona intensity
            dominance_data = update.data
            new_level = dominance_data.get("new_level", 0.5)
            user_id = dominance_data.get("user_id")
            
            if user_id:
                await self._adjust_persona_intensity_from_dominance(user_id, new_level)
        
        elif update.update_type == "relationship_state_change":
            # Consider persona change for relationship progression
            relationship_data = update.data
            await self._consider_persona_progression(relationship_data)
        
        elif update.update_type == "scenario_change":
            # Recommend persona for new scenario
            scenario_data = update.data
            await self._recommend_scenario_persona(scenario_data)
        
        elif update.update_type == "intensity_adjustment_needed":
            # Adjust current persona intensity
            adjustment_data = update.data
            await self._process_intensity_adjustment(adjustment_data)
        
        elif update.update_type == "submission_metric_update":
            # Track for persona preferences
            submission_data = update.data
            await self._update_persona_preferences_from_submission(submission_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with persona awareness"""
        messages = await self.get_cross_module_messages()
        user_id = context.user_id
        
        # Check if persona change needed
        change_needed = await self._check_persona_change_needed(context, messages)
        
        if change_needed:
            # Recommend and potentially activate new persona
            recommendation = await self._get_contextual_recommendation(user_id, context, messages)
            
            if recommendation.get("auto_activate", False):
                # Activate recommended persona
                activation_result = await self.original_manager.activate_persona(
                    user_id=user_id,
                    persona_id=recommendation["persona_id"],
                    intensity=recommendation.get("recommended_intensity", 0.7),
                    reason="contextual_recommendation"
                )
                
                # Send activation update
                await self.send_context_update(
                    update_type="persona_activated",
                    data={
                        "user_id": user_id,
                        "persona_id": recommendation["persona_id"],
                        "persona_name": recommendation["persona_name"],
                        "intensity": recommendation.get("recommended_intensity", 0.7),
                        "reason": "contextual_activation"
                    },
                    priority=ContextPriority.HIGH
                )
        
        # Get behavior guidelines for active persona
        if await self._has_active_persona(user_id):
            guidelines = await self.original_manager.get_behavior_guidelines(user_id)
            
            # Send guidelines update
            await self.send_context_update(
                update_type="persona_guidelines_available",
                data={
                    "user_id": user_id,
                    "guidelines": guidelines,
                    "context_adjusted": True
                }
            )
        
        return {
            "persona_processing_complete": True,
            "change_activated": change_needed and recommendation.get("auto_activate", False)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze persona effectiveness and suitability"""
        user_id = context.user_id
        messages = await self.get_cross_module_messages()
        
        # Get persona history and preferences
        history = await self.original_manager.get_persona_history(user_id)
        
        # Analyze current persona effectiveness
        effectiveness = await self._analyze_persona_effectiveness(user_id, messages)
        
        # Get contextual recommendations
        recommendations = await self._generate_persona_recommendations(context, messages, history)
        
        # Analyze persona coherence with other systems
        coherence = await self._analyze_persona_coherence(messages)
        
        return {
            "persona_history": history,
            "effectiveness_analysis": effectiveness,
            "recommendations": recommendations,
            "coherence_analysis": coherence,
            "optimization_suggestions": await self._generate_optimization_suggestions(
                effectiveness, coherence
            )
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize persona elements for response"""
        messages = await self.get_cross_module_messages()
        user_id = context.user_id
        
        # Get active persona
        active_persona = await self.original_manager.get_active_persona(user_id)
        
        synthesis = {
            "use_persona": active_persona.get("active", False),
            "persona_elements": {}
        }
        
        if active_persona.get("active"):
            persona_id = active_persona.get("persona_id")
            
            # Get language patterns
            patterns = await self.original_manager.get_language_patterns(user_id)
            
            # Get adjusted intensity based on context
            adjusted_intensity = await self._calculate_contextual_intensity(
                active_persona.get("intensity", 0.7),
                messages
            )
            
            synthesis["persona_elements"] = {
                "persona_id": persona_id,
                "persona_name": active_persona.get("persona_name"),
                "base_intensity": active_persona.get("intensity"),
                "adjusted_intensity": adjusted_intensity,
                "language_patterns": patterns.get("patterns", {}),
                "communication_style": patterns.get("communication_style", {}),
                "trait_emphasis": await self._determine_trait_emphasis(persona_id, messages)
            }
            
            # Check if persona change suggested
            if self._should_suggest_persona_change(messages):
                synthesis["suggest_persona_change"] = True
                synthesis["suggested_change_reason"] = self._get_change_reason(messages)
        
        return synthesis
    
    # Helper methods
    async def _analyze_persona_context(self, context: SharedContext, active_persona: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for persona suitability"""
        analysis = {
            "current_suitable": True,
            "intensity_appropriate": True,
            "scenario_match": True
        }
        
        if active_persona.get("active"):
            persona_id = active_persona.get("persona_id")
            
            # Check scenario suitability
            if persona_id in self.original_manager.personas:
                persona = self.original_manager.personas[persona_id]
                
                # Check if current interaction matches avoided activities
                input_lower = context.user_input.lower()
                for avoided in persona.avoid_activities:
                    if avoided.lower() in input_lower:
                        analysis["current_suitable"] = False
                        analysis["scenario_match"] = False
                        analysis["mismatch_reason"] = f"mentions_avoided_activity_{avoided}"
                        break
            
            # Check intensity appropriateness
            if context.relationship_context:
                trust = context.relationship_context.get("trust", 0.5)
                current_intensity = active_persona.get("intensity", 0.5)
                
                if current_intensity > trust + 0.3:
                    analysis["intensity_appropriate"] = False
                    analysis["intensity_issue"] = "too_high_for_trust"
        
        return analysis
    
    async def _adjust_persona_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust persona based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name = dominant_emotion[0] if isinstance(dominant_emotion, tuple) else dominant_emotion
        
        # Map emotions to persona adjustments
        if emotion_name in ["Sadistic", "Cruel"]:
            # Consider switching to sadistic persona
            await self.send_context_update(
                update_type="persona_recommendation",
                data={
                    "recommended_persona": "sadistic_dominant",
                    "reason": f"emotional_state_{emotion_name.lower()}",
                    "auto_activate": False
                }
            )
        elif emotion_name in ["Playful", "Amused"]:
            # Consider playful persona
            await self.send_context_update(
                update_type="persona_recommendation",
                data={
                    "recommended_persona": "playful_tease",
                    "reason": "playful_mood",
                    "auto_activate": False
                }
            )
    
    async def _check_persona_change_needed(self, context: SharedContext, 
                                         messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Check if persona change is needed"""
        user_id = context.user_id
        
        # Get active persona
        active_persona = await self.original_manager.get_active_persona(user_id)
        
        if not active_persona.get("active"):
            # No active persona, might need one
            return True
        
        # Check for explicit scenario change
        for module, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("type") == "scenario_change":
                    return True
        
        # Check for significant mood shift
        mood_shift = self._detect_mood_shift(messages)
        if mood_shift > 0.5:
            return True
        
        # Check for relationship milestone
        if self._detect_relationship_milestone(messages):
            return True
        
        return False
    
    async def _calculate_contextual_intensity(self, base_intensity: float, 
                                            messages: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate adjusted intensity based on context"""
        adjusted = base_intensity
        
        # Check submission level
        for module, module_messages in messages.items():
            if module == "submission_progression":
                for msg in module_messages:
                    if msg.get("type") == "submission_level":
                        level = msg.get("data", {}).get("level", 1)
                        # Higher submission allows higher intensity
                        adjusted = min(1.0, base_intensity + (level - 1) * 0.1)
        
        # Check arousal/frustration
        for module, module_messages in messages.items():
            if module == "somatosensory_system":
                for msg in module_messages:
                    if msg.get("type") == "arousal_state":
                        arousal = msg.get("data", {}).get("arousal_level", 0)
                        if arousal > 0.7:
                            # High arousal, can increase intensity
                            adjusted = min(1.0, adjusted + 0.1)
        
        return adjusted
    
    async def _determine_trait_emphasis(self, persona_id: str, 
                                      messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Determine which persona traits to emphasize based on context"""
        if persona_id not in self.original_manager.personas:
            return []
        
        persona = self.original_manager.personas[persona_id]
        emphasized_traits = []
        
        # Check context for trait relevance
        for trait_name, trait in persona.traits.items():
            # Default emphasis based on trait intensity
            if trait.intensity > 0.7:
                emphasized_traits.append(trait_name)
        
        # Adjust based on messages
        for module, module_messages in messages.items():
            for msg in module_messages:
                # Emphasize cruelty if punishment needed
                if msg.get("type") == "punishment_needed" and "cruel" in persona.traits:
                    if "cruel" not in emphasized_traits:
                        emphasized_traits.append("cruel")
                
                # Emphasize support if user struggling
                if msg.get("type") == "user_struggling" and "supportive" in persona.traits:
                    if "supportive" not in emphasized_traits:
                        emphasized_traits.append("supportive")
        
        return emphasized_traits[:3]  # Limit to top 3 traits
    
    # Delegate other methods to original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
