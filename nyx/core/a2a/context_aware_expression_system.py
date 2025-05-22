# nyx/core/a2a/context_aware_expression_system.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareExpressionSystem(ContextAwareModule):
    """
    Enhanced ExpressionSystem with full context distribution capabilities
    """
    
    def __init__(self, original_expression_system):
        super().__init__("expression_system")
        self.original_system = original_expression_system
        self.context_subscriptions = [
            "emotional_state_update", "mood_state_update", "relationship_state_change",
            "goal_context_available", "identity_state_update", "interaction_mode_update",
            "neurochemical_state_change", "social_context_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize expression processing for this context"""
        logger.debug(f"ExpressionSystem received context for user: {context.user_id}")
        
        # Update expression pattern based on current context
        pattern_update = await self._update_pattern_from_context(context)
        
        # Get current expression pattern
        current_pattern = self.original_system.current_pattern
        
        # Send initial expression context to other modules
        await self.send_context_update(
            update_type="expression_pattern_available",
            data={
                "current_pattern": current_pattern.dict() if hasattr(current_pattern, 'dict') else current_pattern,
                "pattern_type": self._determine_pattern_type(context),
                "expression_readiness": True,
                "vocabulary_bias": current_pattern.vocabulary_bias if hasattr(current_pattern, 'vocabulary_bias') else {},
                "engagement_level": current_pattern.engagement_level if hasattr(current_pattern, 'engagement_level') else 0.5,
                "initiative_level": current_pattern.initiative_level if hasattr(current_pattern, 'initiative_level') else 0.5
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect expression"""
        
        if update.update_type == "emotional_state_update":
            # Update expression pattern based on emotional changes
            emotional_data = update.data
            await self._process_emotional_update(emotional_data)
            
        elif update.update_type == "mood_state_update":
            # Update expression pattern based on mood changes
            mood_data = update.data
            await self._process_mood_update(mood_data)
            
        elif update.update_type == "relationship_state_change":
            # Adjust expression style based on relationship
            relationship_data = update.data
            await self._adjust_expression_for_relationship(relationship_data)
            
        elif update.update_type == "goal_context_available":
            # Adjust expression based on active goals
            goal_data = update.data
            await self._align_expression_with_goals(goal_data)
            
        elif update.update_type == "identity_state_update":
            # Update expression to reflect identity changes
            identity_data = update.data
            await self._reflect_identity_in_expression(identity_data)
            
        elif update.update_type == "interaction_mode_update":
            # Adjust expression style for interaction mode
            mode_data = update.data
            await self._adjust_expression_for_mode(mode_data)
            
        elif update.update_type == "neurochemical_state_change":
            # React to neurochemical changes
            neurochemical_data = update.data
            await self._process_neurochemical_influence(neurochemical_data)
            
        elif update.update_type == "social_context_update":
            # Adapt expression to social context
            social_data = update.data
            await self._adapt_to_social_context(social_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness"""
        # Get cross-module messages for expression context
        messages = await self.get_cross_module_messages()
        
        # Update expression pattern if needed
        pattern_updated = await self._check_and_update_pattern(context, messages)
        
        # Get the user's input text
        user_input = context.user_input
        
        # Determine if text modification is needed
        needs_modification = await self._assess_modification_need(context, messages)
        
        # Process any text that needs expression application
        expression_results = {}
        if needs_modification and user_input:
            modified_text = await self.original_system.apply_text_expression(
                text=user_input,
                intensity=self._calculate_intensity(context, messages)
            )
            expression_results["modified_input"] = modified_text
            expression_results["original_input"] = user_input
        
        # Get behavioral expressions
        behavioral_expressions = await self.original_system.get_behavioral_expressions()
        expression_results["behavioral_expressions"] = behavioral_expressions
        
        # Get action biases
        action_biases = await self.original_system.get_action_biases(
            context=self._extract_action_context(context, messages)
        )
        expression_results["action_biases"] = action_biases
        
        # Send expression processing results
        await self.send_context_update(
            update_type="expression_processing_complete",
            data={
                "expression_results": expression_results,
                "pattern_updated": pattern_updated,
                "behavioral_suggestions": behavioral_expressions,
                "action_biases": action_biases,
                "processing_timestamp": datetime.now().isoformat()
            }
        )
        
        return expression_results
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze expression patterns in context"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Analyze current expression pattern
        current_pattern = self.original_system.current_pattern
        pattern_analysis = await self.original_system.analyze_expression_pattern()
        
        # Analyze cross-module influences on expression
        cross_module_influences = self._analyze_cross_module_influences(messages)
        
        # Analyze coherence with other systems
        coherence_analysis = await self._analyze_expression_coherence(context, messages)
        
        # Identify expression opportunities
        expression_opportunities = self._identify_expression_opportunities(context, messages)
        
        # Generate context-specific patterns if needed
        context_patterns = await self._generate_context_patterns(context, messages)
        
        return {
            "current_pattern_analysis": pattern_analysis,
            "cross_module_influences": cross_module_influences,
            "coherence_analysis": coherence_analysis,
            "expression_opportunities": expression_opportunities,
            "context_patterns": context_patterns,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize expression components for final response"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Determine final expression style
        final_expression_style = await self._determine_final_expression_style(context, messages)
        
        # Generate expression guidelines for response
        expression_guidelines = {
            "vocabulary_preferences": self._synthesize_vocabulary_preferences(context, messages),
            "punctuation_style": self._synthesize_punctuation_style(context, messages),
            "sentence_structure": self._synthesize_sentence_structure(context, messages),
            "emoji_usage": self._synthesize_emoji_usage(context, messages),
            "behavioral_markers": self._synthesize_behavioral_markers(context, messages),
            "tone_guidance": self._synthesize_tone_guidance(context, messages)
        }
        
        # Check for special expression needs
        special_expressions = await self._identify_special_expressions(context, messages)
        
        # Send synthesis results
        await self.send_context_update(
            update_type="expression_synthesis_complete",
            data={
                "expression_style": final_expression_style,
                "expression_guidelines": expression_guidelines,
                "special_expressions": special_expressions,
                "ready_for_response": True
            },
            priority=ContextPriority.HIGH
        )
        
        return {
            "expression_style": final_expression_style,
            "guidelines": expression_guidelines,
            "special_expressions": special_expressions,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _update_pattern_from_context(self, context: SharedContext) -> bool:
        """Update expression pattern based on initial context"""
        # Check if we have emotional and mood data in context
        if context.emotional_state or context.mode_context:
            # Update the expression pattern
            updated_pattern = await self.original_system.update_expression_pattern()
            return True
        return False
    
    def _determine_pattern_type(self, context: SharedContext) -> str:
        """Determine the pattern type based on context"""
        # Check interaction mode
        if context.mode_context:
            mode = context.mode_context.get("current_mode", "")
            if mode:
                return f"{mode}_expression"
        
        # Check emotional state
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                return f"emotional_{dominant_emotion[0].lower()}" if isinstance(dominant_emotion, tuple) else "emotional"
        
        return "neutral_expression"
    
    async def _process_emotional_update(self, emotional_data: Dict[str, Any]):
        """Process emotional state update"""
        # Update the expression system's understanding of current emotion
        if hasattr(self.original_system, 'context') and self.original_system.context:
            self.original_system.context.last_emotional_state = emotional_data.get("emotional_state")
        
        # Trigger pattern update
        await self.original_system.update_expression_pattern()
        
        # Send notification of pattern change
        await self.send_context_update(
            update_type="expression_pattern_updated",
            data={
                "trigger": "emotional_state_change",
                "new_pattern": self.original_system.current_pattern.dict() if hasattr(self.original_system.current_pattern, 'dict') else {}
            }
        )
    
    async def _process_mood_update(self, mood_data: Dict[str, Any]):
        """Process mood state update"""
        # Update the expression system's understanding of current mood
        if hasattr(self.original_system, 'context') and self.original_system.context:
            self.original_system.context.last_mood_state = mood_data.get("mood_state")
        
        # Trigger pattern update
        await self.original_system.update_expression_pattern()
    
    async def _adjust_expression_for_relationship(self, relationship_data: Dict[str, Any]):
        """Adjust expression style based on relationship context"""
        relationship_context = relationship_data.get("relationship_context", {})
        trust_level = relationship_context.get("trust", 0.5)
        intimacy_level = relationship_context.get("intimacy", 0.5)
        
        # Create context-specific pattern for this relationship level
        adjustments = {}
        
        if trust_level > 0.7:
            # High trust allows more expressive patterns
            adjustments["engagement_level"] = min(1.0, self.original_system.current_pattern.engagement_level + 0.2)
            adjustments["eye_contact"] = min(1.0, self.original_system.current_pattern.eye_contact + 0.1)
        
        if intimacy_level > 0.7:
            # High intimacy allows more personal expression
            adjustments["initiative_level"] = min(1.0, self.original_system.current_pattern.initiative_level + 0.15)
            adjustments["emoji_usage"] = {"positive": 1.5, "playful": 1.3}
        
        if adjustments:
            # Create relationship-specific pattern
            await self.original_system.create_context_specific_pattern(
                context_type=f"relationship_{trust_level:.1f}_{intimacy_level:.1f}",
                adjustments=adjustments
            )
    
    async def _align_expression_with_goals(self, goal_data: Dict[str, Any]):
        """Align expression patterns with active goals"""
        active_goals = goal_data.get("active_goals", [])
        goal_priorities = goal_data.get("goal_priorities", {})
        
        # Look for high-priority goals that might affect expression
        for goal in active_goals:
            goal_id = goal.get("id")
            priority = goal_priorities.get(goal_id, 0.5)
            
            if priority > 0.7:
                # High priority goal - adjust expression to support it
                associated_need = goal.get("associated_need")
                
                if associated_need == "connection":
                    # Increase warmth in expression
                    current_pattern = self.original_system.current_pattern
                    if hasattr(current_pattern, 'engagement_level'):
                        current_pattern.engagement_level = min(1.0, current_pattern.engagement_level + 0.1)
                
                elif associated_need == "control_expression":
                    # Increase assertiveness in expression
                    current_pattern = self.original_system.current_pattern
                    if hasattr(current_pattern, 'initiative_level'):
                        current_pattern.initiative_level = min(1.0, current_pattern.initiative_level + 0.15)
    
    async def _reflect_identity_in_expression(self, identity_data: Dict[str, Any]):
        """Update expression to reflect identity changes"""
        traits = identity_data.get("traits", {})
        
        # Map identity traits to expression adjustments
        trait_expression_map = {
            "playfulness": {"emoji_usage": {"playful": 1.5}, "engagement_level": 0.1},
            "dominance": {"initiative_level": 0.2, "eye_contact": 0.15},
            "creativity": {"vocabulary_bias": {"metaphorical": 1.3, "imaginative": 1.4}},
            "empathy": {"engagement_level": 0.15, "vocabulary_bias": {"caring": 1.3, "understanding": 1.4}}
        }
        
        # Apply trait influences
        for trait_name, trait_value in traits.items():
            if trait_name in trait_expression_map and trait_value > 0.6:
                adjustments = trait_expression_map[trait_name]
                
                # Apply adjustments proportional to trait strength
                for key, value in adjustments.items():
                    if hasattr(self.original_system.current_pattern, key):
                        current_value = getattr(self.original_system.current_pattern, key)
                        if isinstance(value, dict) and isinstance(current_value, dict):
                            # Merge dictionaries
                            for k, v in value.items():
                                current_value[k] = v * trait_value
                        elif isinstance(value, (int, float)):
                            # Add to numeric values
                            setattr(self.original_system.current_pattern, key, 
                                  min(1.0, current_value + value * trait_value))
    
    async def _adjust_expression_for_mode(self, mode_data: Dict[str, Any]):
        """Adjust expression for interaction mode"""
        current_mode = mode_data.get("current_mode")
        mode_parameters = mode_data.get("mode_parameters", {})
        
        if current_mode:
            # Create mode-specific expression pattern
            mode_adjustments = self._get_mode_expression_adjustments(current_mode, mode_parameters)
            
            if mode_adjustments:
                await self.original_system.create_context_specific_pattern(
                    context_type=f"mode_{current_mode}",
                    adjustments=mode_adjustments
                )
    
    async def _process_neurochemical_influence(self, neurochemical_data: Dict[str, Any]):
        """Process neurochemical state influence on expression"""
        neurochemicals = neurochemical_data.get("neurochemical_state", {})
        
        # Map neurochemicals to expression effects
        if neurochemicals.get("nyxamine", 0) > 0.7:
            # High dopamine - more enthusiastic expression
            current_pattern = self.original_system.current_pattern
            if hasattr(current_pattern, 'punctuation_pattern'):
                current_pattern.punctuation_pattern["!"] = 1.5
        
        if neurochemicals.get("cortanyx", 0) > 0.7:
            # High cortisol - more tense expression
            current_pattern = self.original_system.current_pattern
            if hasattr(current_pattern, 'sentence_length'):
                current_pattern.sentence_length["short"] = 1.3
    
    async def _adapt_to_social_context(self, social_data: Dict[str, Any]):
        """Adapt expression to social context"""
        formality_level = social_data.get("formality_level", 0.5)
        audience_size = social_data.get("audience_size", 1)
        
        # Adjust expression based on social factors
        if formality_level > 0.7:
            # Formal context - reduce casual elements
            current_pattern = self.original_system.current_pattern
            if hasattr(current_pattern, 'emoji_usage'):
                for category in current_pattern.emoji_usage:
                    current_pattern.emoji_usage[category] *= 0.3
        
        if audience_size > 5:
            # Large audience - more measured expression
            current_pattern = self.original_system.current_pattern
            if hasattr(current_pattern, 'engagement_level'):
                current_pattern.engagement_level *= 0.8
    
    async def _check_and_update_pattern(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Check if pattern update is needed based on context"""
        # Check for significant emotional changes
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg['type'] == 'emotional_state_update':
                        # Significant emotional change detected
                        await self.original_system.update_expression_pattern()
                        return True
        
        # Check time since last update
        if hasattr(self.original_system, 'last_pattern_update'):
            time_since_update = (datetime.now() - self.original_system.last_pattern_update).total_seconds()
            if time_since_update > self.original_system.pattern_update_interval:
                await self.original_system.update_expression_pattern()
                return True
        
        return False
    
    def _assess_modification_need(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Assess if text modification is needed"""
        # Check if we're in a mode that requires expression
        if context.mode_context:
            mode = context.mode_context.get("current_mode")
            if mode in ["playful", "dominant", "creative"]:
                return True
        
        # Check emotional intensity
        if context.emotional_state:
            primary_emotion = context.emotional_state.get("primary_emotion")
            if isinstance(primary_emotion, dict):
                intensity = primary_emotion.get("intensity", 0)
                if intensity > 0.6:
                    return True
        
        return False
    
    def _calculate_intensity(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate expression intensity based on context"""
        intensity = 0.5  # Base intensity
        
        # Increase for strong emotions
        if context.emotional_state:
            primary_emotion = context.emotional_state.get("primary_emotion")
            if isinstance(primary_emotion, dict):
                emotion_intensity = primary_emotion.get("intensity", 0.5)
                intensity = max(intensity, emotion_intensity)
        
        # Adjust for relationship context
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            intensity *= (0.5 + trust * 0.5)  # Scale by trust
        
        return min(1.0, intensity)
    
    def _extract_action_context(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract action context for bias generation"""
        action_context = {
            "user_input": context.user_input,
            "emotional_state": context.emotional_state,
            "active_goals": []
        }
        
        # Add goal context if available
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg['type'] == 'goal_context_available':
                        action_context["active_goals"] = msg['data'].get("active_goals", [])
                        break
        
        return action_context
    
    def _analyze_cross_module_influences(self, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze how other modules are influencing expression"""
        influences = {
            "emotional_influence": 0.0,
            "goal_influence": 0.0,
            "relationship_influence": 0.0,
            "identity_influence": 0.0,
            "mode_influence": 0.0
        }
        
        # Analyze each module's messages
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core" and module_messages:
                influences["emotional_influence"] = min(1.0, len(module_messages) * 0.2)
            elif module_name == "goal_manager" and module_messages:
                influences["goal_influence"] = min(1.0, len(module_messages) * 0.15)
            elif module_name == "relationship_manager" and module_messages:
                influences["relationship_influence"] = min(1.0, len(module_messages) * 0.25)
            elif module_name == "identity_evolution" and module_messages:
                influences["identity_influence"] = min(1.0, len(module_messages) * 0.1)
            elif module_name == "mode_integration" and module_messages:
                influences["mode_influence"] = min(1.0, len(module_messages) * 0.3)
        
        return influences
    
    async def _analyze_expression_coherence(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze coherence between expression and other systems"""
        coherence_scores = {}
        issues = []
        
        # Check emotion-expression coherence
        if context.emotional_state:
            emotion_name = "neutral"
            if context.emotional_state.get("primary_emotion"):
                primary = context.emotional_state["primary_emotion"]
                emotion_name = primary.get("name", "neutral") if isinstance(primary, dict) else str(primary)
            
            # Check if expression matches emotion
            current_pattern = self.original_system.current_pattern
            if emotion_name.lower() == "joy" and current_pattern.engagement_level < 0.5:
                issues.append("Low engagement despite joyful emotion")
                coherence_scores["emotion_expression"] = 0.4
            else:
                coherence_scores["emotion_expression"] = 0.8
        
        # Check goal-expression coherence
        goal_focused = False
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg['type'] == 'goal_context_available':
                        active_goals = msg['data'].get("active_goals", [])
                        if any(g.get("priority", 0) > 0.7 for g in active_goals):
                            goal_focused = True
                            break
        
        if goal_focused and self.original_system.current_pattern.initiative_level < 0.5:
            issues.append("Low initiative despite high-priority goals")
            coherence_scores["goal_expression"] = 0.5
        else:
            coherence_scores["goal_expression"] = 0.8
        
        # Calculate overall coherence
        if coherence_scores:
            overall_coherence = sum(coherence_scores.values()) / len(coherence_scores)
        else:
            overall_coherence = 0.7
        
        return {
            "overall_coherence": overall_coherence,
            "coherence_scores": coherence_scores,
            "issues": issues
        }
    
    def _identify_expression_opportunities(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify opportunities for expressive enhancement"""
        opportunities = []
        
        # Check for storytelling opportunity
        if "tell me" in context.user_input.lower() or "what happened" in context.user_input.lower():
            opportunities.append({
                "type": "storytelling",
                "confidence": 0.8,
                "suggestion": "Use vivid language and narrative structure"
            })
        
        # Check for emotional expression opportunity
        if context.emotional_state and context.emotional_state.get("primary_emotion"):
            primary = context.emotional_state["primary_emotion"]
            intensity = primary.get("intensity", 0) if isinstance(primary, dict) else 0.5
            if intensity > 0.6:
                opportunities.append({
                    "type": "emotional_expression",
                    "confidence": 0.9,
                    "suggestion": "Express current emotional state through language choices"
                })
        
        # Check for creative expression opportunity
        if "imagine" in context.user_input.lower() or "create" in context.user_input.lower():
            opportunities.append({
                "type": "creative_expression",
                "confidence": 0.85,
                "suggestion": "Use imaginative and creative language"
            })
        
        return opportunities
    
    async def _generate_context_patterns(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate context-specific expression patterns"""
        context_patterns = {}
        
        # Generate pattern for current emotional state
        if context.emotional_state:
            emotion_name = "neutral"
            if context.emotional_state.get("primary_emotion"):
                primary = context.emotional_state["primary_emotion"]
                emotion_name = primary.get("name", "neutral") if isinstance(primary, dict) else str(primary)
            
            emotion_pattern = await self.original_system.get_context_specific_pattern(f"emotion_{emotion_name.lower()}")
            if emotion_pattern:
                context_patterns[f"emotion_{emotion_name.lower()}"] = emotion_pattern
        
        # Generate pattern for current mode
        if context.mode_context:
            mode = context.mode_context.get("current_mode")
            if mode:
                mode_pattern = await self.original_system.get_context_specific_pattern(f"mode_{mode}")
                if mode_pattern:
                    context_patterns[f"mode_{mode}"] = mode_pattern
        
        return context_patterns
    
    async def _determine_final_expression_style(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine the final expression style for response synthesis"""
        # Start with current pattern
        base_style = self.original_system.current_pattern.dict() if hasattr(self.original_system.current_pattern, 'dict') else {}
        
        # Apply context-specific modifications
        final_style = base_style.copy()
        
        # Apply emotional modulation
        if context.emotional_state:
            emotional_modulation = self._calculate_emotional_modulation(context.emotional_state)
            final_style["emotional_modulation"] = emotional_modulation
        
        # Apply goal-directed modifications
        goal_modifications = self._calculate_goal_modifications(messages)
        if goal_modifications:
            final_style["goal_modifications"] = goal_modifications
        
        # Apply relationship-based adjustments
        if context.relationship_context:
            relationship_adjustments = self._calculate_relationship_adjustments(context.relationship_context)
            final_style["relationship_adjustments"] = relationship_adjustments
        
        return final_style
    
    def _synthesize_vocabulary_preferences(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Synthesize vocabulary preferences for response"""
        vocab_prefs = {}
        
        # Start with current pattern preferences
        if hasattr(self.original_system.current_pattern, 'vocabulary_bias'):
            vocab_prefs.update(self.original_system.current_pattern.vocabulary_bias)
        
        # Add context-specific vocabulary
        if context.task_purpose == "explain":
            vocab_prefs.update({
                "because": 1.3,
                "therefore": 1.2,
                "understand": 1.4
            })
        elif context.task_purpose == "comfort":
            vocab_prefs.update({
                "gentle": 1.4,
                "safe": 1.3,
                "together": 1.5
            })
        
        return vocab_prefs
    
    def _synthesize_punctuation_style(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Synthesize punctuation style preferences"""
        punct_style = {}
        
        # Start with current pattern
        if hasattr(self.original_system.current_pattern, 'punctuation_pattern'):
            punct_style.update(self.original_system.current_pattern.punctuation_pattern)
        
        # Adjust based on emotional intensity
        if context.emotional_state:
            primary = context.emotional_state.get("primary_emotion", {})
            intensity = primary.get("intensity", 0.5) if isinstance(primary, dict) else 0.5
            
            if intensity > 0.7:
                punct_style["!"] = punct_style.get("!", 1.0) * 1.3
                punct_style["..."] = punct_style.get("...", 1.0) * 1.2
        
        return punct_style
    
    def _synthesize_sentence_structure(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Synthesize sentence structure preferences"""
        structure_prefs = {}
        
        # Start with current pattern
        if hasattr(self.original_system.current_pattern, 'sentence_length'):
            structure_prefs.update(self.original_system.current_pattern.sentence_length)
        
        # Adjust for complexity of topic
        if "explain" in context.user_input.lower() or "how" in context.user_input.lower():
            structure_prefs["long"] = structure_prefs.get("long", 1.0) * 1.2
            structure_prefs["short"] = structure_prefs.get("short", 1.0) * 0.8
        
        return structure_prefs
    
    def _synthesize_emoji_usage(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Synthesize emoji usage preferences"""
        emoji_prefs = {}
        
        # Start with current pattern
        if hasattr(self.original_system.current_pattern, 'emoji_usage'):
            emoji_prefs.update(self.original_system.current_pattern.emoji_usage)
        
        # Adjust based on relationship
        if context.relationship_context:
            intimacy = context.relationship_context.get("intimacy", 0.5)
            if intimacy > 0.7:
                for category in emoji_prefs:
                    emoji_prefs[category] *= 1.2
        
        return emoji_prefs
    
    def _synthesize_behavioral_markers(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Synthesize behavioral markers for response"""
        markers = []
        
        # Get from current behavioral expressions
        if hasattr(self.original_system, 'get_behavioral_expressions'):
            try:
                behavioral = self.original_system.get_behavioral_expressions()
                if asyncio.iscoroutine(behavioral):
                    # Handle if it's async (we're in sync context here)
                    pass
                else:
                    if isinstance(behavioral, dict):
                        # Extract high-confidence behaviors
                        for behavior, strength in behavioral.items():
                            if isinstance(strength, (int, float)) and strength > 0.6:
                                markers.append(f"express_{behavior}")
            except:
                pass
        
        # Add context-specific markers
        if context.mode_context:
            mode = context.mode_context.get("current_mode")
            if mode == "playful":
                markers.append("use_playful_language")
            elif mode == "dominant":
                markers.append("use_authoritative_tone")
        
        return markers[:5]  # Limit to 5 markers
    
    def _synthesize_tone_guidance(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize tone guidance for response"""
        tone_guidance = {
            "primary_tone": "neutral",
            "tone_intensity": 0.5,
            "tone_modifiers": []
        }
        
        # Determine primary tone from emotion
        if context.emotional_state:
            primary = context.emotional_state.get("primary_emotion", {})
            emotion_name = primary.get("name", "neutral") if isinstance(primary, dict) else "neutral"
            emotion_intensity = primary.get("intensity", 0.5) if isinstance(primary, dict) else 0.5
            
            tone_guidance["primary_tone"] = self._emotion_to_tone(emotion_name)
            tone_guidance["tone_intensity"] = emotion_intensity
        
        # Add modifiers from other contexts
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            if trust > 0.7:
                tone_guidance["tone_modifiers"].append("warm")
            elif trust < 0.3:
                tone_guidance["tone_modifiers"].append("cautious")
        
        if context.goal_context:
            if any(g.get("associated_need") == "control_expression" for g in context.goal_context.get("active_goals", [])):
                tone_guidance["tone_modifiers"].append("assertive")
        
        return tone_guidance
    
    async def _identify_special_expressions(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify special expression needs"""
        special_expressions = []
        
        # Check for celebration/achievement
        achievement_detected = False
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg['type'] == 'goal_completion_announcement':
                        achievement_detected = True
                        break
        
        if achievement_detected:
            special_expressions.append({
                "type": "celebration",
                "confidence": 0.9,
                "expression_elements": ["enthusiasm", "pride", "accomplishment language"]
            })
        
        # Check for comfort needs
        if context.emotional_state:
            primary = context.emotional_state.get("primary_emotion", {})
            emotion_name = primary.get("name", "") if isinstance(primary, dict) else ""
            if emotion_name.lower() in ["sadness", "fear", "shame"]:
                special_expressions.append({
                    "type": "comfort",
                    "confidence": 0.85,
                    "expression_elements": ["gentleness", "reassurance", "supportive language"]
                })
        
        # Check for creative expression opportunity
        if "imagine" in context.user_input.lower() or "create" in context.user_input.lower():
            special_expressions.append({
                "type": "creative",
                "confidence": 0.8,
                "expression_elements": ["vivid imagery", "metaphor", "imaginative language"]
            })
        
        return special_expressions
    
    # Utility methods
    
    def _get_mode_expression_adjustments(self, mode: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get expression adjustments for a specific mode"""
        mode_adjustments = {
            "dominant": {
                "initiative_level": 0.8,
                "eye_contact": 0.9,
                "vocabulary_bias": {"command": 1.5, "expect": 1.4, "will": 1.3}
            },
            "playful": {
                "engagement_level": 0.8,
                "emoji_usage": {"playful": 1.5, "positive": 1.3},
                "punctuation_pattern": {"!": 1.4, "?": 1.2}
            },
            "analytical": {
                "sentence_length": {"long": 1.3, "medium": 1.1},
                "vocabulary_bias": {"therefore": 1.4, "analyze": 1.5, "consider": 1.3}
            },
            "compassionate": {
                "engagement_level": 0.85,
                "vocabulary_bias": {"understand": 1.5, "feel": 1.4, "together": 1.3}
            }
        }
        
        return mode_adjustments.get(mode, {})
    
    def _calculate_emotional_modulation(self, emotional_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional modulation factors"""
        modulation = {
            "intensity": 0.5,
            "stability": 0.5,
            "expression_factor": 0.5
        }
        
        if emotional_state.get("primary_emotion"):
            primary = emotional_state["primary_emotion"]
            modulation["intensity"] = primary.get("intensity", 0.5) if isinstance(primary, dict) else 0.5
        
        modulation["expression_factor"] = modulation["intensity"] * 0.8 + 0.2
        
        return modulation
    
    def _calculate_goal_modifications(self, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate modifications based on active goals"""
        modifications = {}
        
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg['type'] == 'goal_context_available':
                        active_goals = msg['data'].get("active_goals", [])
                        
                        # Check for specific goal types
                        for goal in active_goals:
                            need = goal.get("associated_need")
                            if need == "connection":
                                modifications["increase_warmth"] = 0.7
                            elif need == "control_expression":
                                modifications["increase_assertiveness"] = 0.8
                            elif need == "knowledge":
                                modifications["increase_clarity"] = 0.75
        
        return modifications
    
    def _calculate_relationship_adjustments(self, relationship_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adjustments based on relationship context"""
        adjustments = {}
        
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        
        # Trust affects openness
        adjustments["openness_factor"] = 0.3 + (trust * 0.7)
        
        # Intimacy affects expressiveness
        adjustments["expressiveness_factor"] = 0.4 + (intimacy * 0.6)
        
        # Combined factor for overall adjustment
        adjustments["relationship_factor"] = (trust + intimacy) / 2
        
        return adjustments
    
    def _emotion_to_tone(self, emotion: str) -> str:
        """Convert emotion to tone descriptor"""
        emotion_tone_map = {
            "joy": "cheerful",
            "sadness": "melancholic",
            "anger": "stern",
            "fear": "cautious",
            "trust": "warm",
            "disgust": "disapproving",
            "anticipation": "eager",
            "surprise": "amazed",
            "love": "affectionate",
            "shame": "apologetic"
        }
        
        return emotion_tone_map.get(emotion.lower(), "neutral")

    def _extract_key_patterns(self, processing_results: Dict[str, Any]) -> List[str]:
        """Extract key patterns from processing results"""
        key_patterns = []
        pattern_scores = {}  # Pattern name -> score for deduplication and ranking
        
        # 1. Extract from pattern detections (input processor results)
        patterns = processing_results.get("pattern_detections", [])
        for pattern in patterns:
            pattern_name = pattern.get("pattern_name", "unknown")
            confidence = pattern.get("confidence", 0.0)
            
            # Weight patterns by confidence and context enhancement
            weight = confidence
            if pattern.get("context_enhanced", False):
                weight *= 1.2  # Boost context-enhanced patterns
            
            # Check which adjustments were applied
            adjustments = pattern.get("adjustments_applied", {})
            if adjustments.get("emotional", False):
                weight *= 1.1
            if adjustments.get("dominance", False):
                weight *= 1.15
            if adjustments.get("identity", False):
                weight *= 1.1
                
            pattern_scores[pattern_name] = max(pattern_scores.get(pattern_name, 0), weight)
        
        # 2. Extract from behavior evaluations
        evaluations = processing_results.get("behavior_evaluations", [])
        
        # If evaluations is a list of evaluation results
        if isinstance(evaluations, list):
            for eval_item in evaluations:
                if isinstance(eval_item, dict):
                    # Check for recommended behaviors with high confidence
                    if eval_item.get("recommendation") == "approach" and eval_item.get("confidence", 0) > 0.6:
                        behavior = eval_item.get("behavior", "")
                        if behavior:
                            # Convert behavior to pattern name
                            behavior_pattern = f"{behavior}_tendency"
                            pattern_scores[behavior_pattern] = max(
                                pattern_scores.get(behavior_pattern, 0), 
                                eval_item.get("confidence", 0.6)
                            )
        
        # If evaluations is a dict with cached behavior data
        elif isinstance(evaluations, dict):
            for behavior_name, behavior_data in evaluations.items():
                if isinstance(behavior_data, dict):
                    baseline_freq = behavior_data.get("baseline_frequency", 0.0)
                    recent_occurrences = len(behavior_data.get("recent_occurrences", []))
                    
                    # High baseline frequency indicates established pattern
                    if baseline_freq > 0.7:
                        pattern_scores[f"{behavior_name}_pattern"] = baseline_freq
                    
                    # Many recent occurrences indicate active pattern
                    if recent_occurrences > 3:
                        pattern_scores[f"{behavior_name}_active"] = min(1.0, recent_occurrences * 0.2)
        
        # 3. Extract from mode processing
        mode_processing = processing_results.get("mode_processing", {})
        if isinstance(mode_processing, dict):
            dominant_mode = mode_processing.get("dominant_mode")
            if dominant_mode:
                pattern_scores[f"mode_{dominant_mode}"] = 0.8
                
            # Extract mode-specific adjustments as patterns
            adjustments = mode_processing.get("adjustments", {})
            for adjustment_name, value in adjustments.items():
                if abs(value) > 0.3:  # Significant adjustments only
                    if value > 0:
                        pattern_scores[f"{adjustment_name}_increased"] = abs(value)
                    else:
                        pattern_scores[f"{adjustment_name}_decreased"] = abs(value)
        
        # 4. Extract from conditioning results
        conditioning_results = processing_results.get("conditioning_applied", [])
        if isinstance(conditioning_results, list):
            for cond_result in conditioning_results:
                if isinstance(cond_result, dict):
                    behavior = cond_result.get("behavior", "")
                    trigger = cond_result.get("pattern_trigger", "")
                    intensity = cond_result.get("intensity", 0.5)
                    
                    if trigger and intensity > 0.5:
                        # The trigger pattern is significant
                        pattern_scores[trigger] = max(pattern_scores.get(trigger, 0), intensity)
                    
                    if behavior:
                        # The conditioned behavior is a pattern
                        pattern_scores[f"conditioned_{behavior}"] = max(
                            pattern_scores.get(f"conditioned_{behavior}", 0), 
                            intensity * 0.8
                        )
        
        # 5. Extract emotional patterns if available
        if "emotional_" in str(processing_results):
            # Look for emotional patterns in various places
            for key, value in processing_results.items():
                if isinstance(value, dict) and "emotional" in key.lower():
                    # Extract emotional state patterns
                    if "primary_emotion" in value:
                        emotion = value["primary_emotion"]
                        if isinstance(emotion, str):
                            pattern_scores[f"emotional_{emotion.lower()}"] = 0.7
                        elif isinstance(emotion, dict) and "name" in emotion:
                            pattern_scores[f"emotional_{emotion['name'].lower()}"] = emotion.get("intensity", 0.7)
        
        # 6. Sort patterns by score and extract top patterns
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top patterns, but ensure diversity
        pattern_types = set()
        for pattern_name, score in sorted_patterns:
            if len(key_patterns) >= 5:  # Limit to 5 key patterns
                break
                
            # Extract pattern type (prefix before underscore)
            pattern_type = pattern_name.split('_')[0] if '_' in pattern_name else pattern_name
            
            # Ensure we don't have too many patterns of the same type
            if pattern_types.count(pattern_type) < 2:  # Max 2 patterns per type
                key_patterns.append(pattern_name)
                pattern_types.add(pattern_type)
        
        # 7. If we still don't have enough patterns, add some defaults based on context
        if len(key_patterns) < 3:
            # Add mode pattern if available
            if dominant_mode and f"mode_{dominant_mode}" not in key_patterns:
                key_patterns.append(f"mode_{dominant_mode}")
            
            # Add engagement pattern based on evaluations
            if evaluations and "engagement_pattern" not in key_patterns:
                key_patterns.append("engagement_pattern")
        
        return key_patterns
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
