# nyx/core/a2a/context_aware_mode_integration.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareModeIntegration(ContextAwareModule):
    """
    Advanced ModeIntegrationManager with full context distribution capabilities
    """
    
    def __init__(self, original_mode_integration):
        super().__init__("mode_integration")
        self.original_integration = original_mode_integration
        self.context_subscriptions = [
            "emotional_state_update", "goal_context_available", "relationship_state_change",
            "mood_state_update", "needs_state_change", "dominance_gratification",
            "user_feedback", "memory_context_available", "attention_shift",
            "reward_signal", "social_context_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize mode processing for this context"""
        logger.debug(f"ModeIntegration received context for user: {context.user_id}")
        
        # Analyze context for mode implications
        mode_implications = await self._analyze_context_for_modes(context)
        
        # Get current mode distribution
        current_distribution = await self._get_current_mode_distribution()
        
        # Calculate suggested mode adjustments
        suggested_adjustments = await self._calculate_mode_adjustments(context, mode_implications)
        
        # Send initial mode context
        await self.send_context_update(
            update_type="mode_context_initialized",
            data={
                "current_distribution": current_distribution,
                "mode_implications": mode_implications,
                "suggested_adjustments": suggested_adjustments,
                "primary_mode": current_distribution.get("primary_mode"),
                "active_modes": await self._get_active_modes(current_distribution)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting interaction modes"""
        
        if update.update_type == "emotional_state_update":
            # Emotional changes affect mode selection
            emotional_data = update.data
            await self._adjust_modes_for_emotion(emotional_data)
        
        elif update.update_type == "mood_state_update":
            # Mood affects longer-term mode preferences
            mood_data = update.data
            await self._adjust_modes_for_mood(mood_data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect interaction style
            relationship_data = update.data
            await self._adjust_modes_for_relationship(relationship_data)
        
        elif update.update_type == "needs_state_change":
            # High priority needs can shift modes
            needs_data = update.data
            high_priority_needs = needs_data.get("high_priority_needs", [])
            
            if high_priority_needs:
                await self._adjust_modes_for_needs(high_priority_needs, needs_data)
        
        elif update.update_type == "dominance_gratification":
            # Dominance success reinforces dominant modes
            dominance_data = update.data
            await self._reinforce_dominance_modes(dominance_data)
        
        elif update.update_type == "user_feedback":
            # Direct feedback about interaction style
            feedback_data = update.data
            await self._process_mode_feedback(feedback_data)
        
        elif update.update_type == "goal_context_available":
            # Goals influence mode selection
            goal_data = update.data
            await self._align_modes_with_goals(goal_data)
        
        elif update.update_type == "social_context_update":
            # Social context affects formality and style
            social_data = update.data
            await self._adjust_modes_for_social_context(social_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with mode awareness and coordination"""
        # Process message through mode system
        mode_result = await self.original_integration.process_message(
            message=context.user_input,
            user_id=context.user_id,
            additional_context=context.session_context
        )
        
        # Extract mode information
        mode_distribution = mode_result.get("mode_distribution", {})
        primary_mode = mode_result.get("primary_mode", "default")
        active_modes = mode_result.get("active_modes", [])
        
        # Analyze cross-module implications
        messages = await self.get_cross_module_messages()
        cross_module_alignment = await self._analyze_mode_alignment(mode_distribution, messages)
        
        # Send mode update if changed
        if mode_result.get("mode_updated"):
            await self.send_context_update(
                update_type="mode_change",
                data={
                    "mode_distribution": mode_distribution,
                    "primary_mode": primary_mode,
                    "active_modes": active_modes,
                    "change_reason": mode_result.get("change_reason", "context_based"),
                    "cross_module_alignment": cross_module_alignment
                }
            )
        
        # Get blended guidance
        guidance = await self.original_integration.get_response_guidance()
        
        return {
            "mode_processing_complete": True,
            "mode_result": mode_result,
            "guidance": guidance,
            "cross_module_alignment": cross_module_alignment,
            "blended_processing": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze mode effectiveness and coherence"""
        # Get current mode state
        current_distribution = await self._get_current_mode_distribution()
        
        # Analyze mode effectiveness
        effectiveness_analysis = await self._analyze_mode_effectiveness(context, current_distribution)
        
        # Check mode-context coherence
        coherence_analysis = await self._analyze_mode_coherence(context, current_distribution)
        
        # Analyze mode transitions
        transition_analysis = await self._analyze_mode_transitions()
        
        # Generate mode recommendations
        recommendations = await self._generate_mode_recommendations(
            context, effectiveness_analysis, coherence_analysis
        )
        
        return {
            "current_distribution": current_distribution,
            "effectiveness_analysis": effectiveness_analysis,
            "coherence_analysis": coherence_analysis,
            "transition_analysis": transition_analysis,
            "recommendations": recommendations,
            "mode_health": await self._assess_mode_health()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize mode-based response modifications"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Get current mode guidance
        guidance = await self.original_integration.get_response_guidance()
        
        # Generate response modifications
        modifications = {
            "tone_adjustments": await self._synthesize_tone_adjustments(guidance, messages),
            "style_elements": await self._synthesize_style_elements(guidance, messages),
            "vocabulary_preferences": await self._synthesize_vocabulary(guidance, context),
            "interaction_patterns": await self._synthesize_interaction_patterns(guidance),
            "mode_specific_elements": await self._get_mode_specific_elements(guidance)
        }
        
        # Check for mode conflicts
        conflicts = await self._detect_mode_conflicts(guidance, messages)
        if conflicts:
            # Resolve conflicts
            resolution = await self._resolve_mode_conflicts(conflicts)
            modifications["conflict_resolution"] = resolution
        
        # Send synthesis complete
        await self.send_context_update(
            update_type="mode_synthesis_complete",
            data={
                "modifications": modifications,
                "guidance_summary": self._summarize_guidance(guidance),
                "synthesis_confidence": await self._calculate_synthesis_confidence(modifications)
            }
        )
        
        return {
            "mode_synthesis": modifications,
            "synthesis_complete": True,
            "blended_response_ready": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_context_for_modes(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context for mode selection implications"""
        implications = {
            "suggested_modes": {},
            "mode_triggers": [],
            "formality_level": 0.5,
            "energy_level": 0.5,
            "dominance_level": 0.5
        }
        
        user_input_lower = context.user_input.lower()
        
        # Analyze for mode triggers
        mode_triggers = {
            "dominant": ["obey", "submit", "command", "control", "dominate"],
            "compassionate": ["help", "support", "care", "comfort", "understand"],
            "intellectual": ["analyze", "explain", "think", "reason", "logic"],
            "playful": ["fun", "play", "joke", "laugh", "silly"],
            "professional": ["work", "business", "formal", "meeting", "report"],
            "creative": ["create", "imagine", "design", "invent", "artistic"],
            "friendly": ["chat", "talk", "friend", "casual", "hey"]
        }
        
        for mode, triggers in mode_triggers.items():
            if any(trigger in user_input_lower for trigger in triggers):
                implications["suggested_modes"][mode] = 0.3  # Base suggestion weight
                implications["mode_triggers"].append(mode)
        
        # Analyze formality from context
        formal_indicators = ["sir", "madam", "please", "kindly", "would you"]
        casual_indicators = ["hey", "yeah", "gonna", "wanna", "lol"]
        
        formal_count = sum(1 for ind in formal_indicators if ind in user_input_lower)
        casual_count = sum(1 for ind in casual_indicators if ind in user_input_lower)
        
        if formal_count > casual_count:
            implications["formality_level"] = min(1.0, 0.5 + formal_count * 0.1)
        elif casual_count > formal_count:
            implications["formality_level"] = max(0.0, 0.5 - casual_count * 0.1)
        
        # Analyze energy level from punctuation and capitals
        exclamation_count = context.user_input.count("!")
        question_count = context.user_input.count("?")
        caps_ratio = sum(1 for c in context.user_input if c.isupper()) / max(1, len(context.user_input))
        
        implications["energy_level"] = min(1.0, 0.5 + exclamation_count * 0.1 + caps_ratio)
        
        # Analyze dominance indicators
        submission_words = ["please", "may i", "could you", "would you mind"]
        dominance_words = ["must", "will", "now", "immediately", "do this"]
        
        submission_count = sum(1 for word in submission_words if word in user_input_lower)
        dominance_count = sum(1 for word in dominance_words if word in user_input_lower)
        
        if dominance_count > submission_count:
            implications["dominance_level"] = min(1.0, 0.5 + dominance_count * 0.15)
        elif submission_count > dominance_count:
            implications["dominance_level"] = max(0.0, 0.5 - submission_count * 0.15)
        
        return implications
    
    async def _get_current_mode_distribution(self) -> Dict[str, Any]:
        """Get current mode distribution from mode manager"""
        if self.original_integration.mode_manager:
            try:
                # Get from mode manager context
                mode_context = self.original_integration.mode_manager.context
                distribution = mode_context.mode_distribution.dict()
                
                # Get primary mode
                primary_mode, primary_weight = mode_context.mode_distribution.primary_mode
                
                return {
                    "distribution": distribution,
                    "primary_mode": primary_mode,
                    "primary_weight": primary_weight,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting mode distribution: {e}")
        
        return {
            "distribution": {"friendly": 1.0},
            "primary_mode": "friendly",
            "primary_weight": 1.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _calculate_mode_adjustments(self, context: SharedContext, implications: Dict[str, Any]) -> Dict[str, float]:
        """Calculate suggested mode adjustments based on context"""
        adjustments = {}
        
        # Start with suggested modes from implications
        for mode, weight in implications.get("suggested_modes", {}).items():
            adjustments[mode] = weight
        
        # Adjust based on formality level
        formality = implications.get("formality_level", 0.5)
        if formality > 0.7:
            adjustments["professional"] = adjustments.get("professional", 0) + 0.2
            adjustments["friendly"] = adjustments.get("friendly", 0) - 0.1
        elif formality < 0.3:
            adjustments["friendly"] = adjustments.get("friendly", 0) + 0.2
            adjustments["professional"] = adjustments.get("professional", 0) - 0.1
        
        # Adjust based on energy level
        energy = implications.get("energy_level", 0.5)
        if energy > 0.7:
            adjustments["playful"] = adjustments.get("playful", 0) + 0.15
        
        # Adjust based on dominance level
        dominance = implications.get("dominance_level", 0.5)
        if dominance > 0.7:
            adjustments["dominant"] = adjustments.get("dominant", 0) + 0.25
        elif dominance < 0.3:
            adjustments["compassionate"] = adjustments.get("compassionate", 0) + 0.15
        
        # Normalize adjustments
        total_positive = sum(v for v in adjustments.values() if v > 0)
        if total_positive > 0.5:  # Don't let adjustments be too dramatic
            scale_factor = 0.5 / total_positive
            adjustments = {k: v * scale_factor for k, v in adjustments.items()}
        
        return adjustments
    
    async def _get_active_modes(self, distribution: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get list of active modes (weight >= 0.2)"""
        mode_dist = distribution.get("distribution", {})
        return [(mode, weight) for mode, weight in mode_dist.items() if weight >= 0.2]
    
    async def _adjust_modes_for_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust modes based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Emotion to mode mappings
        emotion_mode_map = {
            "Joy": {"playful": 0.2, "friendly": 0.15},
            "Sadness": {"compassionate": 0.2, "friendly": -0.1},
            "Anger": {"dominant": 0.15, "compassionate": -0.1},
            "Fear": {"compassionate": 0.15, "dominant": -0.15},
            "Excitement": {"playful": 0.25, "creative": 0.15},
            "Frustration": {"professional": 0.1, "playful": -0.2},
            "Love": {"compassionate": 0.3, "friendly": 0.2},
            "Curiosity": {"intellectual": 0.25, "creative": 0.15},
            "Pride": {"dominant": 0.2, "professional": 0.1},
            "Shame": {"compassionate": 0.2, "dominant": -0.3}
        }
        
        if emotion_name in emotion_mode_map:
            adjustments = emotion_mode_map[emotion_name]
            
            # Scale by intensity
            scaled_adjustments = {mode: adj * intensity for mode, adj in adjustments.items()}
            
            # Apply adjustments
            await self._apply_mode_adjustments(scaled_adjustments, f"emotion_{emotion_name.lower()}")
            
            # Send notification
            await self.send_context_update(
                update_type="mode_adjusted_for_emotion",
                data={
                    "emotion": emotion_name,
                    "intensity": intensity,
                    "adjustments": scaled_adjustments
                }
            )
    
    async def _adjust_modes_for_mood(self, mood_data: Dict[str, Any]):
        """Adjust modes based on mood state"""
        mood = mood_data.get("dominant_mood", "Neutral")
        valence = mood_data.get("valence", 0.0)
        arousal = mood_data.get("arousal", 0.5)
        control = mood_data.get("control", 0.0)
        
        adjustments = {}
        
        # Valence affects friendliness and compassion
        if valence > 0.3:
            adjustments["friendly"] = 0.1
            adjustments["compassionate"] = 0.05
        elif valence < -0.3:
            adjustments["professional"] = 0.1  # More reserved when mood is negative
            adjustments["friendly"] = -0.1
        
        # Arousal affects energy-related modes
        if arousal > 0.7:
            adjustments["playful"] = 0.15
            adjustments["creative"] = 0.1
        elif arousal < 0.3:
            adjustments["intellectual"] = 0.1  # Low energy favors thoughtful modes
            adjustments["playful"] = -0.15
        
        # Control affects dominance
        if control > 0.3:
            adjustments["dominant"] = 0.2
        elif control < -0.3:
            adjustments["compassionate"] = 0.15
            adjustments["dominant"] = -0.2
        
        # Apply adjustments
        if adjustments:
            await self._apply_mode_adjustments(adjustments, f"mood_{mood.lower()}")
    
    async def _adjust_modes_for_relationship(self, relationship_data: Dict[str, Any]):
        """Adjust modes based on relationship context"""
        relationship_context = relationship_data.get("relationship_context", {})
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        power_dynamic = relationship_context.get("power_dynamic", 0.0)
        
        adjustments = {}
        
        # High trust allows more playful and dominant modes
        if trust > 0.7:
            adjustments["playful"] = 0.15
            adjustments["dominant"] = 0.1
            adjustments["professional"] = -0.1  # Less need for formality
        elif trust < 0.3:
            adjustments["professional"] = 0.2
            adjustments["friendly"] = 0.15  # Build trust through friendliness
            adjustments["dominant"] = -0.2
        
        # High intimacy enables more personal modes
        if intimacy > 0.7:
            adjustments["compassionate"] = 0.2
            adjustments["creative"] = 0.1
        elif intimacy < 0.3:
            adjustments["professional"] = 0.15
            adjustments["compassionate"] = -0.1
        
        # Power dynamic affects dominance
        if power_dynamic > 0.5:  # User is submissive
            adjustments["dominant"] = 0.25
        elif power_dynamic < -0.5:  # User is dominant
            adjustments["compassionate"] = 0.2
            adjustments["dominant"] = -0.15
        
        # Apply adjustments
        if adjustments:
            await self._apply_mode_adjustments(adjustments, "relationship_context")
    
    async def _adjust_modes_for_needs(self, high_priority_needs: List[str], needs_data: Dict[str, Any]):
        """Adjust modes based on high priority needs"""
        adjustments = {}
        
        # Need to mode mappings
        need_mode_map = {
            "connection": {"friendly": 0.2, "compassionate": 0.15},
            "knowledge": {"intellectual": 0.25, "professional": 0.1},
            "control_expression": {"dominant": 0.3},
            "pleasure_indulgence": {"playful": 0.2, "creative": 0.15},
            "novelty": {"creative": 0.25, "playful": 0.15},
            "intimacy": {"compassionate": 0.25, "friendly": 0.1},
            "agency": {"dominant": 0.15, "professional": 0.1}
        }
        
        for need in high_priority_needs:
            if need in need_mode_map:
                for mode, adjustment in need_mode_map[need].items():
                    adjustments[mode] = adjustments.get(mode, 0) + adjustment
        
        # Scale adjustments by urgency
        urgency = needs_data.get("urgency", 0.5)
        scaled_adjustments = {mode: adj * (0.5 + urgency * 0.5) for mode, adj in adjustments.items()}
        
        # Apply adjustments
        if scaled_adjustments:
            await self._apply_mode_adjustments(scaled_adjustments, "high_priority_needs")
    
    async def _reinforce_dominance_modes(self, dominance_data: Dict[str, Any]):
        """Reinforce dominance-related modes after successful dominance"""
        gratification_type = dominance_data.get("gratification_type", "general")
        intensity = dominance_data.get("intensity", 0.5)
        
        adjustments = {
            "dominant": 0.3 * intensity,
            "compassionate": -0.1 * intensity  # Slight reduction in softer modes
        }
        
        # Specific gratification types might adjust other modes
        if gratification_type == "control":
            adjustments["professional"] = 0.1 * intensity
        elif gratification_type == "playful_dominance":
            adjustments["playful"] = 0.15 * intensity
        
        # Apply reinforcement
        await self._apply_mode_adjustments(adjustments, f"dominance_reinforcement_{gratification_type}")
        
        # Send notification
        await self.send_context_update(
            update_type="dominance_mode_reinforced",
            data={
                "gratification_type": gratification_type,
                "intensity": intensity,
                "adjustments": adjustments
            }
        )
    
    async def _process_mode_feedback(self, feedback_data: Dict[str, Any]):
        """Process direct feedback about interaction style"""
        feedback_text = feedback_data.get("feedback", "")
        sentiment = feedback_data.get("sentiment", 0.0)
        
        # Analyze feedback for mode-related comments
        feedback_lower = feedback_text.lower()
        
        mode_feedback_map = {
            "too formal": {"professional": -0.2, "friendly": 0.15},
            "too casual": {"professional": 0.2, "friendly": -0.15},
            "too dominant": {"dominant": -0.25, "compassionate": 0.2},
            "not assertive enough": {"dominant": 0.2},
            "boring": {"playful": 0.2, "creative": 0.15},
            "too silly": {"playful": -0.2, "professional": 0.15},
            "cold": {"compassionate": 0.25, "friendly": 0.2},
            "too emotional": {"professional": 0.15, "compassionate": -0.1}
        }
        
        adjustments = {}
        for feedback_phrase, mode_adjustments in mode_feedback_map.items():
            if feedback_phrase in feedback_lower:
                for mode, adjustment in mode_adjustments.items():
                    adjustments[mode] = adjustments.get(mode, 0) + adjustment
        
        # Scale by sentiment (negative feedback has stronger effect)
        if sentiment < 0:
            scale_factor = 1.0 + abs(sentiment)
            adjustments = {mode: adj * scale_factor for mode, adj in adjustments.items()}
        
        # Apply feedback-based adjustments
        if adjustments:
            await self._apply_mode_adjustments(adjustments, "user_feedback")
    
    async def _align_modes_with_goals(self, goal_data: Dict[str, Any]):
        """Align modes with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        if not active_goals:
            return
        
        adjustments = {}
        
        # Analyze goal types and adjust modes accordingly
        for goal in active_goals[:5]:  # Top 5 goals
            goal_desc = goal.get("description", "").lower()
            priority = goal.get("priority", 0.5)
            
            # Goal type detection and mode alignment
            if any(word in goal_desc for word in ["learn", "understand", "analyze"]):
                adjustments["intellectual"] = adjustments.get("intellectual", 0) + 0.1 * priority
                
            elif any(word in goal_desc for word in ["connect", "bond", "relationship"]):
                adjustments["friendly"] = adjustments.get("friendly", 0) + 0.1 * priority
                adjustments["compassionate"] = adjustments.get("compassionate", 0) + 0.1 * priority
                
            elif any(word in goal_desc for word in ["control", "dominate", "command"]):
                adjustments["dominant"] = adjustments.get("dominant", 0) + 0.15 * priority
                
            elif any(word in goal_desc for word in ["create", "imagine", "design"]):
                adjustments["creative"] = adjustments.get("creative", 0) + 0.15 * priority
                
            elif any(word in goal_desc for word in ["work", "achieve", "complete"]):
                adjustments["professional"] = adjustments.get("professional", 0) + 0.1 * priority
        
        # Apply goal-based adjustments
        if adjustments:
            await self._apply_mode_adjustments(adjustments, "goal_alignment")
    
    async def _adjust_modes_for_social_context(self, social_data: Dict[str, Any]):
        """Adjust modes based on social context"""
        social_setting = social_data.get("setting", "casual")
        participants = social_data.get("participants", 1)
        formality_required = social_data.get("formality_required", False)
        
        adjustments = {}
        
        # Formal settings
        if social_setting == "formal" or formality_required:
            adjustments["professional"] = 0.3
            adjustments["playful"] = -0.2
            adjustments["friendly"] = -0.1
        
        # Group settings (multiple participants)
        elif participants > 1:
            adjustments["friendly"] = 0.2  # More inclusive
            adjustments["dominant"] = -0.15  # Less dominant in groups
            adjustments["professional"] = 0.1
        
        # Intimate settings
        elif social_setting == "intimate":
            adjustments["compassionate"] = 0.25
            adjustments["friendly"] = 0.15
            adjustments["professional"] = -0.2
        
        # Apply social context adjustments
        if adjustments:
            await self._apply_mode_adjustments(adjustments, f"social_context_{social_setting}")
    
    async def _apply_mode_adjustments(self, adjustments: Dict[str, float], reason: str):
        """Apply mode adjustments to current distribution"""
        if not self.original_integration.mode_manager:
            return
        
        try:
            # Get current distribution
            current_context = self.original_integration.mode_manager.context
            current_dist = current_context.mode_distribution
            
            # Create new distribution with adjustments
            new_dist_dict = current_dist.dict()
            
            # Apply adjustments
            for mode, adjustment in adjustments.items():
                if mode in new_dist_dict:
                    # Apply adjustment with bounds
                    new_value = max(0.0, min(1.0, new_dist_dict[mode] + adjustment))
                    new_dist_dict[mode] = new_value
            
            # Create new distribution object
            from nyx.core.interaction_mode_manager import ModeDistribution
            new_distribution = ModeDistribution(**new_dist_dict)
            
            # Normalize
            new_distribution = new_distribution.normalize()
            
            # Update mode manager
            current_context.mode_distribution = new_distribution
            
            # Update primary mode
            primary_mode, primary_weight = new_distribution.primary_mode
            current_context.current_mode = primary_mode
            
            logger.info(f"Applied mode adjustments for {reason}: {adjustments}")
            
        except Exception as e:
            logger.error(f"Error applying mode adjustments: {e}")
    
    async def _analyze_mode_alignment(self, mode_distribution: Dict[str, float], messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze alignment between modes and other module states"""
        alignment_analysis = {
            "overall_alignment": 0.0,
            "misalignments": [],
            "synergies": []
        }
        
        # Check alignment with emotional state
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotion_data = msg["data"]
                        emotion_alignment = await self._check_emotion_mode_alignment(
                            emotion_data, mode_distribution
                        )
                        alignment_analysis["emotion_alignment"] = emotion_alignment
                        
            elif module_name == "goal_manager":
                for msg in module_messages:
                    if msg["type"] == "goal_context_available":
                        goal_data = msg["data"]
                        goal_alignment = await self._check_goal_mode_alignment(
                            goal_data, mode_distribution
                        )
                        alignment_analysis["goal_alignment"] = goal_alignment
        
        # Calculate overall alignment
        alignment_scores = []
        if "emotion_alignment" in alignment_analysis:
            alignment_scores.append(alignment_analysis["emotion_alignment"]["score"])
        if "goal_alignment" in alignment_analysis:
            alignment_scores.append(alignment_analysis["goal_alignment"]["score"])
            
        if alignment_scores:
            alignment_analysis["overall_alignment"] = sum(alignment_scores) / len(alignment_scores)
        
        return alignment_analysis
    
    async def _check_emotion_mode_alignment(self, emotion_data: Dict[str, Any], mode_distribution: Dict[str, float]) -> Dict[str, Any]:
        """Check alignment between emotional state and mode distribution"""
        alignment = {
            "score": 0.5,
            "issues": []
        }
        
        dominant_emotion = emotion_data.get("dominant_emotion")
        if dominant_emotion:
            emotion_name, intensity = dominant_emotion
            
            # Check for conflicts
            if emotion_name == "Sadness" and mode_distribution.get("playful", 0) > 0.5:
                alignment["issues"].append("High playful mode during sadness")
                alignment["score"] -= 0.3
                
            elif emotion_name == "Anger" and mode_distribution.get("compassionate", 0) > 0.5:
                alignment["issues"].append("High compassionate mode during anger")
                alignment["score"] -= 0.2
                
            # Check for synergies
            elif emotion_name == "Joy" and mode_distribution.get("playful", 0) > 0.3:
                alignment["score"] += 0.2
                
            elif emotion_name == "Curiosity" and mode_distribution.get("intellectual", 0) > 0.3:
                alignment["score"] += 0.2
        
        alignment["score"] = max(0.0, min(1.0, alignment["score"]))
        return alignment
    
    async def _check_goal_mode_alignment(self, goal_data: Dict[str, Any], mode_distribution: Dict[str, float]) -> Dict[str, Any]:
        """Check alignment between goals and mode distribution"""
        alignment = {
            "score": 0.5,
            "issues": []
        }
        
        active_goals = goal_data.get("active_goals", [])
        
        for goal in active_goals[:3]:  # Check top 3 goals
            goal_desc = goal.get("description", "").lower()
            
            # Check for learning goals vs intellectual mode
            if "learn" in goal_desc or "understand" in goal_desc:
                if mode_distribution.get("intellectual", 0) < 0.2:
                    alignment["issues"].append("Learning goal but low intellectual mode")
                    alignment["score"] -= 0.1
                else:
                    alignment["score"] += 0.1
                    
            # Check for dominance goals vs dominant mode
            elif "control" in goal_desc or "dominate" in goal_desc:
                if mode_distribution.get("dominant", 0) < 0.2:
                    alignment["issues"].append("Control goal but low dominant mode")
                    alignment["score"] -= 0.15
                else:
                    alignment["score"] += 0.15
        
        alignment["score"] = max(0.0, min(1.0, alignment["score"]))
        return alignment
    
    async def _analyze_mode_effectiveness(self, context: SharedContext, current_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness of current mode distribution"""
        effectiveness = {
            "overall_effectiveness": 0.5,
            "mode_scores": {},
            "effective_modes": [],
            "ineffective_modes": []
        }
        
        # Analyze each active mode
        distribution = current_distribution.get("distribution", {})
        
        for mode, weight in distribution.items():
            if weight >= 0.2:  # Only analyze active modes
                mode_effectiveness = await self._calculate_mode_effectiveness(mode, weight, context)
                effectiveness["mode_scores"][mode] = mode_effectiveness
                
                if mode_effectiveness > 0.7:
                    effectiveness["effective_modes"].append(mode)
                elif mode_effectiveness < 0.3:
                    effectiveness["ineffective_modes"].append(mode)
        
        # Calculate overall effectiveness
        if effectiveness["mode_scores"]:
            # Weighted average by mode weight
            total_weighted_score = sum(
                score * distribution.get(mode, 0) 
                for mode, score in effectiveness["mode_scores"].items()
            )
            total_weight = sum(distribution.get(mode, 0) for mode in effectiveness["mode_scores"])
            
            if total_weight > 0:
                effectiveness["overall_effectiveness"] = total_weighted_score / total_weight
        
        return effectiveness
    
    async def _calculate_mode_effectiveness(self, mode: str, weight: float, context: SharedContext) -> float:
        """Calculate effectiveness score for a specific mode"""
        base_score = 0.5
        
        # Context appropriateness
        user_input_lower = context.user_input.lower()
        
        mode_context_match = {
            "dominant": ["obey", "control", "command", "submit"],
            "compassionate": ["help", "support", "care", "feeling"],
            "intellectual": ["explain", "analyze", "understand", "why"],
            "playful": ["fun", "play", "game", "laugh"],
            "professional": ["work", "business", "report", "meeting"],
            "creative": ["create", "imagine", "idea", "design"],
            "friendly": ["chat", "talk", "hello", "hi"]
        }
        
        # Check context match
        if mode in mode_context_match:
            matches = sum(1 for keyword in mode_context_match[mode] if keyword in user_input_lower)
            base_score += matches * 0.1
        
        # Check relationship appropriateness
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            
            # Some modes require higher trust
            trust_requirements = {
                "dominant": 0.6,
                "playful": 0.5,
                "compassionate": 0.3,
                "professional": 0.0
            }
            
            required_trust = trust_requirements.get(mode, 0.4)
            if trust >= required_trust:
                base_score += 0.2
            else:
                base_score -= 0.2
        
        # Weight influence (very low or very high weights are less effective)
        if 0.3 <= weight <= 0.7:
            base_score += 0.1
        elif weight > 0.8 or weight < 0.2:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    async def _analyze_mode_coherence(self, context: SharedContext, current_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence of mode distribution"""
        coherence = {
            "coherence_score": 1.0,
            "conflicts": [],
            "complementary_pairs": []
        }
        
        distribution = current_distribution.get("distribution", {})
        active_modes = [(mode, weight) for mode, weight in distribution.items() if weight >= 0.2]
        
        # Check for conflicting mode pairs
        conflict_pairs = [
            ("dominant", "compassionate"),
            ("playful", "professional"),
            ("intellectual", "playful")
        ]
        
        for mode1, mode2 in conflict_pairs:
            weight1 = distribution.get(mode1, 0)
            weight2 = distribution.get(mode2, 0)
            
            # Both modes significantly active
            if weight1 >= 0.3 and weight2 >= 0.3:
                coherence["conflicts"].append({
                    "modes": [mode1, mode2],
                    "weights": [weight1, weight2],
                    "severity": min(weight1, weight2)
                })
                coherence["coherence_score"] -= 0.2
        
        # Check for complementary pairs
        complementary_pairs = [
            ("compassionate", "friendly"),
            ("intellectual", "professional"),
            ("creative", "playful")
        ]
        
        for mode1, mode2 in complementary_pairs:
            weight1 = distribution.get(mode1, 0)
            weight2 = distribution.get(mode2, 0)
            
            if weight1 >= 0.2 and weight2 >= 0.2:
                coherence["complementary_pairs"].append({
                    "modes": [mode1, mode2],
                    "weights": [weight1, weight2],
                    "synergy": (weight1 + weight2) / 2
                })
                coherence["coherence_score"] += 0.1
        
        coherence["coherence_score"] = max(0.0, min(1.0, coherence["coherence_score"]))
        return coherence
    
    async def _analyze_mode_transitions(self) -> Dict[str, Any]:
        """Analyze recent mode transitions"""
        transitions = {
            "recent_transitions": [],
            "transition_frequency": 0.0,
            "stability_score": 1.0
        }
        
        # This would need access to mode history
        # For now, return basic analysis
        
        return transitions
    
    async def _generate_mode_recommendations(self, context: SharedContext, 
                                           effectiveness: Dict[str, Any], 
                                           coherence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for mode adjustments"""
        recommendations = []
        
        # Address ineffective modes
        for mode in effectiveness.get("ineffective_modes", []):
            recommendations.append({
                "type": "reduce_mode",
                "mode": mode,
                "reason": "Low effectiveness in current context",
                "suggested_adjustment": -0.2
            })
        
        # Address mode conflicts
        for conflict in coherence.get("conflicts", []):
            modes = conflict["modes"]
            recommendations.append({
                "type": "resolve_conflict",
                "modes": modes,
                "reason": f"Conflicting modes: {modes[0]} vs {modes[1]}",
                "suggested_action": f"Reduce weight of one mode"
            })
        
        # Suggest modes based on context
        context_implications = await self._analyze_context_for_modes(context)
        for mode, suggested_weight in context_implications.get("suggested_modes", {}).items():
            current_weight = effectiveness.get("mode_scores", {}).get(mode, 0)
            if suggested_weight > current_weight + 0.2:
                recommendations.append({
                    "type": "increase_mode",
                    "mode": mode,
                    "reason": "Context suggests this mode",
                    "suggested_adjustment": 0.2
                })
        
        return recommendations
    
    async def _assess_mode_health(self) -> Dict[str, Any]:
        """Assess overall health of mode system"""
        health = {
            "health_score": 0.7,
            "issues": [],
            "strengths": []
        }
        
        # Get current distribution
        current_dist = await self._get_current_mode_distribution()
        distribution = current_dist.get("distribution", {})
        
        # Check for mode diversity
        active_modes = sum(1 for weight in distribution.values() if weight >= 0.2)
        if active_modes == 1:
            health["issues"].append("Single mode dominance")
            health["health_score"] -= 0.2
        elif active_modes >= 2 and active_modes <= 3:
            health["strengths"].append("Good mode diversity")
            health["health_score"] += 0.1
        elif active_modes > 4:
            health["issues"].append("Too many active modes")
            health["health_score"] -= 0.1
        
        # Check for extreme weights
        extreme_modes = [mode for mode, weight in distribution.items() if weight > 0.8 or weight < 0.1]
        if extreme_modes:
            health["issues"].append(f"Extreme mode weights: {extreme_modes}")
            health["health_score"] -= 0.1
        
        health["health_score"] = max(0.0, min(1.0, health["health_score"]))
        return health
    
    async def _synthesize_tone_adjustments(self, guidance: Dict[str, Any], messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize tone adjustments from guidance and context"""
        tone_adjustments = {
            "base_tone": guidance.get("tone", "balanced"),
            "modifiers": [],
            "intensity": 0.5
        }
        
        # Add emotional influence
        for module_messages in messages.values():
            for msg in module_messages:
                if msg["type"] == "emotional_state_update":
                    emotion_data = msg["data"]
                    dominant_emotion = emotion_data.get("dominant_emotion")
                    if dominant_emotion:
                        emotion_name, intensity = dominant_emotion
                        if intensity > 0.6:
                            tone_adjustments["modifiers"].append(f"emotionally_{emotion_name.lower()}")
                            tone_adjustments["intensity"] = max(tone_adjustments["intensity"], intensity)
        
        return tone_adjustments
    
    async def _synthesize_style_elements(self, guidance: Dict[str, Any], messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Synthesize style elements from guidance"""
        style_elements = []
        
        # Extract from guidance
        if guidance.get("formality_level", 0.5) > 0.7:
            style_elements.append("formal_language")
        elif guidance.get("formality_level", 0.5) < 0.3:
            style_elements.append("casual_language")
        
        if guidance.get("verbosity", 0.5) > 0.7:
            style_elements.append("detailed_responses")
        elif guidance.get("verbosity", 0.5) < 0.3:
            style_elements.append("concise_responses")
        
        # Add mode-specific elements
        active_modes = guidance.get("active_modes", {})
        for mode, weight in active_modes.items():
            if weight > 0.3:
                style_elements.append(f"{mode}_style")
        
        return style_elements
    
    async def _synthesize_vocabulary(self, guidance: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Synthesize vocabulary preferences"""
        vocabulary = {
            "preferred_terms": guidance.get("key_phrases", []),
            "avoid_terms": guidance.get("avoid_phrases", []),
            "complexity_level": "medium"
        }
        
        # Adjust complexity based on mode
        if "intellectual" in guidance.get("active_modes", {}):
            vocabulary["complexity_level"] = "high"
        elif "friendly" in guidance.get("active_modes", {}) or "playful" in guidance.get("active_modes", {}):
            vocabulary["complexity_level"] = "low"
        
        return vocabulary
    
    async def _synthesize_interaction_patterns(self, guidance: Dict[str, Any]) -> List[str]:
        """Synthesize interaction patterns from mode guidance"""
        patterns = []
        
        active_modes = guidance.get("active_modes", {})
        
        # Mode-specific patterns
        if active_modes.get("dominant", 0) > 0.3:
            patterns.extend(["directive_statements", "commanding_tone", "expectation_setting"])
            
        if active_modes.get("compassionate", 0) > 0.3:
            patterns.extend(["empathetic_responses", "supportive_language", "validation"])
            
        if active_modes.get("intellectual", 0) > 0.3:
            patterns.extend(["analytical_approach", "evidence_based", "logical_structure"])
            
        if active_modes.get("playful", 0) > 0.3:
            patterns.extend(["humor_inclusion", "light_hearted_tone", "creative_expressions"])
        
        return patterns
    
    async def _get_mode_specific_elements(self, guidance: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get mode-specific elements for response generation"""
        elements = {}
        
        active_modes = guidance.get("active_modes", {})
        
        for mode, weight in active_modes.items():
            if weight >= 0.2:
                elements[mode] = await self._get_elements_for_mode(mode, weight)
        
        return elements
    
    async def _get_elements_for_mode(self, mode: str, weight: float) -> List[str]:
        """Get specific elements for a mode"""
        mode_elements = {
            "dominant": ["assert_control", "set_expectations", "command_presence"],
            "compassionate": ["show_understanding", "offer_support", "express_care"],
            "intellectual": ["provide_analysis", "explain_reasoning", "cite_facts"],
            "playful": ["add_humor", "be_creative", "lighten_mood"],
            "professional": ["maintain_formality", "focus_on_task", "be_efficient"],
            "creative": ["think_outside_box", "suggest_novel_ideas", "use_imagination"],
            "friendly": ["be_approachable", "show_warmth", "casual_conversation"]
        }
        
        elements = mode_elements.get(mode, [])
        
        # Scale elements by weight (return more elements for higher weights)
        num_elements = max(1, int(len(elements) * weight))
        return elements[:num_elements]
    
    async def _detect_mode_conflicts(self, guidance: Dict[str, Any], messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Detect conflicts between modes and other system states"""
        conflicts = []
        
        active_modes = guidance.get("active_modes", {})
        
        # Check for emotional conflicts
        for module_messages in messages.values():
            for msg in module_messages:
                if msg["type"] == "emotional_state_update":
                    emotion_data = msg["data"]
                    dominant_emotion = emotion_data.get("dominant_emotion")
                    
                    if dominant_emotion:
                        emotion_name, intensity = dominant_emotion
                        
                        # Check specific conflicts
                        if emotion_name == "Sadness" and active_modes.get("playful", 0) > 0.4:
                            conflicts.append({
                                "type": "emotion_mode_conflict",
                                "emotion": emotion_name,
                                "mode": "playful",
                                "severity": intensity * active_modes["playful"]
                            })
        
        return conflicts
    
    async def _resolve_mode_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve detected mode conflicts"""
        resolution = {
            "adjustments": {},
            "reasoning": []
        }
        
        for conflict in conflicts:
            if conflict["type"] == "emotion_mode_conflict":
                mode = conflict["mode"]
                severity = conflict["severity"]
                
                # Reduce conflicting mode
                resolution["adjustments"][mode] = -severity * 0.5
                resolution["reasoning"].append(
                    f"Reducing {mode} mode due to conflict with {conflict['emotion']} emotion"
                )
        
        return resolution
    
    def _summarize_guidance(self, guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of guidance for other modules"""
        active_modes = guidance.get("active_modes", {})
        primary_mode = max(active_modes.items(), key=lambda x: x[1])[0] if active_modes else "default"
        
        return {
            "primary_mode": primary_mode,
            "mode_weights": active_modes,
            "formality_level": guidance.get("formality_level", 0.5),
            "energy_level": guidance.get("energy_level", 0.5),
            "key_characteristics": guidance.get("characteristics", []),
            "verbosity": guidance.get("verbosity", 0.5)
        }
    
    async def _calculate_synthesis_confidence(self, modifications: Dict[str, Any]) -> float:
        """Calculate confidence in synthesis results"""
        confidence = 0.8  # Base confidence
        
        # Check for conflicts in modifications
        if modifications.get("conflict_resolution"):
            confidence -= len(modifications["conflict_resolution"].get("adjustments", {})) * 0.1
        
        # Check for coherence in style elements
        style_elements = modifications.get("style_elements", [])
        if len(style_elements) > 5:
            confidence -= 0.1  # Too many competing styles
        
        # Check for clear primary pattern
        patterns = modifications.get("interaction_patterns", [])
        if len(patterns) == 0:
            confidence -= 0.2  # No clear pattern
        elif len(patterns) > 7:
            confidence -= 0.15  # Too many patterns
        
        return max(0.3, min(1.0, confidence))
    
    # Delegate all other methods to the original integration
    def __getattr__(self, name):
        """Delegate any missing methods to the original integration"""
        return getattr(self.original_integration, name)
