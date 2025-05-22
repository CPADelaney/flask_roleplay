# nyx/core/a2a/context_aware_interaction_mode_manager.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareInteractionModeManager(ContextAwareModule):
    """
    Enhanced InteractionModeManager with full context distribution capabilities
    """
    
    def __init__(self, original_mode_manager):
        super().__init__("mode_manager")
        self.original_manager = original_mode_manager
        self.context_subscriptions = [
            "context_distribution_update", "emotional_state_update", "goal_context_available",
            "relationship_state_change", "needs_assessment", "reward_signal",
            "dominance_context_update", "user_preference_detected", "memory_context_available"
        ]
        
        # Advanced mode management state
        self.mode_transition_history = []
        self.mode_stability_scores = {}
        self.context_mode_correlations = {}
        
    async def on_context_received(self, context: SharedContext):
        """Initialize mode processing for this context"""
        logger.debug(f"ModeManager received context for user: {context.user_id}")
        
        # Extract context distribution from shared context
        context_info = self._extract_context_info(context)
        
        # Update mode based on initial context
        mode_result = await self._update_mode_from_context(context_info, context)
        
        # Send initial mode context to other modules
        await self.send_context_update(
            update_type="mode_distribution_update",
            data={
                "mode_distribution": mode_result.get("mode_distribution", {}),
                "primary_mode": mode_result.get("primary_mode", "default"),
                "confidence": mode_result.get("confidence", 0.5),
                "active_modes": mode_result.get("active_modes", []),
                "mode_parameters": await self._get_blended_parameters_for_distribution(mode_result.get("mode_distribution", {}))
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect modes"""
        
        if update.update_type == "context_distribution_update":
            # Direct context distribution update from context awareness system
            await self._handle_context_distribution_update(update.data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional changes can influence mode
            await self._handle_emotional_influence(update.data)
        
        elif update.update_type == "goal_context_available":
            # Active goals influence mode selection
            await self._handle_goal_influence(update.data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect social modes
            await self._handle_relationship_influence(update.data)
        
        elif update.update_type == "needs_assessment":
            # High drive needs influence mode
            await self._handle_needs_influence(update.data)
        
        elif update.update_type == "dominance_context_update":
            # Special handling for dominance mode triggers
            await self._handle_dominance_triggers(update.data)
        
        elif update.update_type == "user_preference_detected":
            # User preferences for interaction style
            await self._handle_user_preferences(update.data)
        
        elif update.update_type == "reward_signal":
            # Reward signals reinforce current mode
            await self._handle_reward_reinforcement(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with mode awareness"""
        # Extract mode cues from user input
        mode_cues = await self._analyze_input_for_mode_cues(context.user_input, context)
        
        # Get cross-module messages for mode context
        messages = await self.get_cross_module_messages()
        
        # Check if mode adjustment is needed
        adjustment_analysis = await self._analyze_mode_adjustment_need(mode_cues, messages, context)
        
        if adjustment_analysis["adjustment_needed"]:
            # Update mode based on comprehensive context
            mode_result = await self._comprehensive_mode_update(context, mode_cues, messages)
            
            # Send mode update to other modules
            await self.send_context_update(
                update_type="mode_adjusted_from_input",
                data={
                    "previous_distribution": adjustment_analysis["previous_distribution"],
                    "new_distribution": mode_result["mode_distribution"],
                    "adjustment_reason": adjustment_analysis["reason"],
                    "confidence": mode_result["confidence"]
                }
            )
        
        return {
            "mode_cues": mode_cues,
            "adjustment_performed": adjustment_analysis["adjustment_needed"],
            "current_mode_distribution": self._get_current_distribution(),
            "mode_stability": await self._calculate_mode_stability()
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze mode effectiveness and coherence"""
        current_distribution = self._get_current_distribution()
        
        # Analyze mode coherence
        coherence_analysis = await self._analyze_mode_coherence(current_distribution, context)
        
        # Analyze mode effectiveness
        effectiveness_analysis = await self._analyze_mode_effectiveness(context)
        
        # Analyze cross-module alignment
        messages = await self.get_cross_module_messages()
        alignment_analysis = await self._analyze_cross_module_alignment(current_distribution, messages)
        
        # Generate mode recommendations
        recommendations = await self._generate_mode_recommendations(
            coherence_analysis, effectiveness_analysis, alignment_analysis
        )
        
        return {
            "current_distribution": current_distribution,
            "coherence_analysis": coherence_analysis,
            "effectiveness_analysis": effectiveness_analysis,
            "alignment_analysis": alignment_analysis,
            "recommendations": recommendations,
            "mode_history": self._get_recent_mode_history()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize mode guidance for response generation"""
        # Get comprehensive mode guidance
        mode_guidance = await self._generate_comprehensive_guidance(context)
        
        # Get blended parameters
        blended_parameters = await self._get_blended_parameters_for_distribution(
            self._get_current_distribution()
        )
        
        # Get conversation style guidance
        style_guidance = await self._generate_style_guidance(context)
        
        # Check for mode expression requirements
        expression_requirements = await self._check_mode_expression_requirements(context)
        
        # Send synthesis results
        synthesis_data = {
            "mode_guidance": mode_guidance,
            "blended_parameters": blended_parameters,
            "style_guidance": style_guidance,
            "expression_requirements": expression_requirements,
            "vocalization_patterns": await self._get_blended_vocalization_patterns()
        }
        
        # Check if dominant mode requires special expression
        if expression_requirements.get("dominant_expression_required"):
            await self.send_context_update(
                update_type="dominant_mode_expression_required",
                data={
                    "dominance_level": expression_requirements["dominance_level"],
                    "expression_type": expression_requirements["expression_type"],
                    "linguistic_patterns": style_guidance.get("dominant_patterns", [])
                },
                priority=ContextPriority.HIGH
            )
        
        return synthesis_data
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    def _extract_context_info(self, context: SharedContext) -> Dict[str, Any]:
        """Extract relevant context information for mode determination"""
        # Get context distribution from session context
        context_dist = context.session_context.get("context_distribution", {})
        
        # Get additional context factors
        context_info = {
            "context_distribution": context_dist,
            "emotional_state": context.emotional_state,
            "relationship_context": context.relationship_context,
            "goal_context": context.goal_context,
            "user_input": context.user_input,
            "processing_stage": context.processing_stage
        }
        
        # Add any mode-specific context if present
        if context.mode_context:
            context_info.update(context.mode_context)
        
        return context_info
    
    async def _update_mode_from_context(self, context_info: Dict[str, Any], shared_context: SharedContext) -> Dict[str, Any]:
        """Update mode based on context information"""
        if hasattr(self.original_manager, 'update_interaction_mode'):
            # Use original manager's update method
            result = await self.original_manager.update_interaction_mode(context_info)
            
            # Store in shared context
            shared_context.mode_context = {
                "mode_distribution": result.get("mode_distribution", {}),
                "primary_mode": result.get("primary_mode", "default"),
                "confidence": result.get("confidence", 0.5),
                "active_modes": result.get("active_modes", [])
            }
            
            # Track transition
            self._track_mode_transition(result)
            
            return result
        
        # Fallback
        return {
            "mode_distribution": {"default": 1.0},
            "primary_mode": "default",
            "confidence": 0.5,
            "active_modes": [("default", 1.0)]
        }
    
    async def _handle_context_distribution_update(self, update_data: Dict[str, Any]):
        """Handle direct context distribution updates"""
        context_distribution = update_data.get("context_distribution", {})
        confidence = update_data.get("overall_confidence", 0.5)
        
        # Update mode to match context distribution
        context_info = {
            "context_distribution": context_distribution,
            "overall_confidence": confidence
        }
        
        mode_result = await self.original_manager.update_interaction_mode(context_info)
        
        # Send mode update to other modules
        await self.send_context_update(
            update_type="mode_distribution_synchronized",
            data={
                "mode_distribution": mode_result.get("mode_distribution", {}),
                "source_context": context_distribution,
                "synchronization_confidence": confidence
            }
        )
    
    async def _handle_emotional_influence(self, emotional_data: Dict[str, Any]):
        """Handle emotional state influence on modes"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Map emotions to mode influences
        emotion_mode_influences = {
            "Joy": {"playful": 0.3, "friendly": 0.2, "creative": 0.2},
            "Excitement": {"playful": 0.3, "creative": 0.2, "friendly": 0.1},
            "Confidence": {"dominant": 0.3, "professional": 0.2, "intellectual": 0.1},
            "Frustration": {"dominant": 0.2, "professional": 0.1},
            "Anxiety": {"compassionate": 0.2, "friendly": 0.2},
            "Curiosity": {"intellectual": 0.3, "creative": 0.2},
            "Affection": {"compassionate": 0.3, "friendly": 0.2, "playful": 0.1},
            "Loneliness": {"compassionate": 0.3, "friendly": 0.3}
        }
        
        influences = emotion_mode_influences.get(emotion_name, {})
        
        if influences and strength > 0.5:
            # Apply emotional influence to current distribution
            current_dist = self._get_current_distribution()
            
            # Blend with emotional influences
            for mode, influence in influences.items():
                adjustment = influence * strength * 0.3  # Scale adjustment
                current_dist[mode] = min(1.0, current_dist.get(mode, 0.0) + adjustment)
            
            # Normalize distribution
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            # Update mode
            await self._apply_mode_distribution(current_dist, f"emotional_influence_{emotion_name}")
    
    async def _handle_goal_influence(self, goal_data: Dict[str, Any]):
        """Handle goal context influence on modes"""
        active_goals = goal_data.get("active_goals", [])
        goal_priorities = goal_data.get("goal_priorities", {})
        
        if not active_goals:
            return
        
        # Extract mode preferences from goals
        mode_weights = {}
        total_priority = 0.0
        
        for goal in active_goals:
            source_mode = goal.get("source_mode", goal.get("source", "").replace("_mode", ""))
            priority = goal_priorities.get(goal.get("id", goal.get("description", "")), goal.get("priority", 0.5))
            
            if source_mode and source_mode != "unknown":
                mode_weights[source_mode] = mode_weights.get(source_mode, 0.0) + priority
                total_priority += priority
        
        # Normalize to get distribution influence
        if total_priority > 0:
            mode_influence = {k: v/total_priority for k, v in mode_weights.items()}
            
            # Blend with current distribution (goals have moderate influence)
            current_dist = self._get_current_distribution()
            blend_factor = 0.3
            
            for mode, influence in mode_influence.items():
                current_dist[mode] = current_dist.get(mode, 0.0) * (1 - blend_factor) + influence * blend_factor
            
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            await self._apply_mode_distribution(current_dist, "goal_influence")
    
    async def _handle_relationship_influence(self, relationship_data: Dict[str, Any]):
        """Handle relationship state influence on modes"""
        relationship_context = relationship_data.get("relationship_context", {})
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        conflict = relationship_context.get("conflict", 0.0)
        
        # Calculate relationship strength
        relationship_strength = (trust + intimacy) / 2 - conflict
        
        # Influence mode distribution based on relationship
        mode_adjustments = {}
        
        if relationship_strength > 0.7:
            # Strong relationship - enable more intimate/playful modes
            mode_adjustments = {
                "playful": 0.2,
                "friendly": 0.15,
                "compassionate": 0.15,
                "dominant": 0.1  # Can be more dominant with trust
            }
        elif relationship_strength > 0.4:
            # Moderate relationship - balanced approach
            mode_adjustments = {
                "friendly": 0.2,
                "compassionate": 0.1,
                "playful": 0.1
            }
        else:
            # Weak relationship - more cautious
            mode_adjustments = {
                "professional": 0.2,
                "intellectual": 0.15,
                "friendly": 0.1
            }
        
        # Apply adjustments
        current_dist = self._get_current_distribution()
        
        for mode, adjustment in mode_adjustments.items():
            current_dist[mode] = min(1.0, current_dist.get(mode, 0.0) + adjustment * 0.4)
        
        # Normalize
        total = sum(current_dist.values())
        if total > 0:
            current_dist = {k: v/total for k, v in current_dist.items()}
        
        await self._apply_mode_distribution(current_dist, "relationship_influence")
    
    async def _handle_needs_influence(self, needs_data: Dict[str, Any]):
        """Handle needs assessment influence on modes"""
        high_drive_needs = needs_data.get("high_drive_needs", [])
        most_urgent_need = needs_data.get("most_urgent_need", {})
        
        # Map needs to mode preferences
        need_mode_mapping = {
            "knowledge": {"intellectual": 0.4, "professional": 0.2},
            "connection": {"friendly": 0.3, "compassionate": 0.3},
            "intimacy": {"compassionate": 0.4, "playful": 0.2},
            "control_expression": {"dominant": 0.5, "professional": 0.1},
            "pleasure_indulgence": {"playful": 0.4, "creative": 0.2},
            "creativity": {"creative": 0.5, "playful": 0.2},
            "agency": {"dominant": 0.2, "intellectual": 0.2},
            "safety": {"compassionate": 0.3, "professional": 0.2}
        }
        
        # Calculate mode influences from needs
        mode_influences = {}
        
        for need in high_drive_needs:
            if need in need_mode_mapping:
                for mode, weight in need_mode_mapping[need].items():
                    mode_influences[mode] = mode_influences.get(mode, 0.0) + weight
        
        # Extra weight for most urgent need
        urgent_need_name = most_urgent_need.get("name")
        if urgent_need_name and urgent_need_name in need_mode_mapping:
            for mode, weight in need_mode_mapping[urgent_need_name].items():
                mode_influences[mode] = mode_influences.get(mode, 0.0) + weight * 0.5
        
        if mode_influences:
            # Normalize influences
            total_influence = sum(mode_influences.values())
            if total_influence > 0:
                mode_influences = {k: v/total_influence for k, v in mode_influences.items()}
            
            # Blend with current distribution
            current_dist = self._get_current_distribution()
            blend_factor = 0.35  # Needs have moderate-high influence
            
            for mode, influence in mode_influences.items():
                current_dist[mode] = current_dist.get(mode, 0.0) * (1 - blend_factor) + influence * blend_factor
            
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            await self._apply_mode_distribution(current_dist, "needs_influence")
    
    async def _handle_dominance_triggers(self, dominance_data: Dict[str, Any]):
        """Handle specific dominance mode triggers"""
        trigger_type = dominance_data.get("trigger_type")
        trigger_strength = dominance_data.get("strength", 0.5)
        
        if trigger_type and trigger_strength > 0.6:
            # Strong dominance trigger - boost dominant mode significantly
            current_dist = self._get_current_distribution()
            
            # Calculate dominance boost
            boost = min(0.4, trigger_strength * 0.5)
            current_dist["dominant"] = min(1.0, current_dist.get("dominant", 0.0) + boost)
            
            # Reduce conflicting modes
            conflicting_modes = ["compassionate", "friendly"]
            for mode in conflicting_modes:
                current_dist[mode] = max(0.0, current_dist.get(mode, 0.0) - boost * 0.5)
            
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            await self._apply_mode_distribution(current_dist, f"dominance_trigger_{trigger_type}")
            
            # Send dominance activation notice
            await self.send_context_update(
                update_type="dominance_mode_activated",
                data={
                    "trigger": trigger_type,
                    "strength": trigger_strength,
                    "new_dominance_weight": current_dist.get("dominant", 0.0)
                },
                priority=ContextPriority.HIGH
            )
    
    async def _handle_user_preferences(self, preference_data: Dict[str, Any]):
        """Handle detected user preferences for interaction style"""
        preferred_modes = preference_data.get("preferred_modes", [])
        preference_strength = preference_data.get("strength", 0.5)
        
        if preferred_modes and preference_strength > 0.4:
            current_dist = self._get_current_distribution()
            
            # Boost preferred modes
            boost_per_mode = preference_strength * 0.3 / len(preferred_modes)
            
            for mode in preferred_modes:
                if mode in ["dominant", "friendly", "intellectual", "compassionate", "playful", "creative", "professional"]:
                    current_dist[mode] = min(1.0, current_dist.get(mode, 0.0) + boost_per_mode)
            
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            await self._apply_mode_distribution(current_dist, "user_preference")
    
    async def _handle_reward_reinforcement(self, reward_data: Dict[str, Any]):
        """Handle reward signals to reinforce current mode"""
        reward_value = reward_data.get("reward_value", 0.0)
        reward_type = reward_data.get("reward_type", "general")
        
        if reward_value > 0.5:
            # Positive reward - reinforce current distribution
            current_dist = self._get_current_distribution()
            
            # Find primary mode to reinforce
            primary_mode = max(current_dist.items(), key=lambda x: x[1])[0] if current_dist else "default"
            
            # Slight boost to primary mode
            reinforcement = min(0.1, reward_value * 0.15)
            current_dist[primary_mode] = min(1.0, current_dist.get(primary_mode, 0.0) + reinforcement)
            
            # Normalize
            total = sum(current_dist.values())
            if total > 0:
                current_dist = {k: v/total for k, v in current_dist.items()}
            
            # Update stability score for primary mode
            self.mode_stability_scores[primary_mode] = self.mode_stability_scores.get(primary_mode, 0.5) + 0.1
            
            # Log the reinforcement
            logger.info(f"Mode {primary_mode} reinforced by {reward_type} reward (value: {reward_value})")
    
    async def _analyze_input_for_mode_cues(self, user_input: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze user input for mode-relevant cues"""
        input_lower = user_input.lower()
        
        mode_cues = {
            "dominant": self._detect_dominance_cues(input_lower),
            "friendly": self._detect_friendly_cues(input_lower),
            "intellectual": self._detect_intellectual_cues(input_lower),
            "compassionate": self._detect_compassionate_cues(input_lower),
            "playful": self._detect_playful_cues(input_lower),
            "creative": self._detect_creative_cues(input_lower),
            "professional": self._detect_professional_cues(input_lower)
        }
        
        # Calculate cue strengths
        total_cues = sum(mode_cues.values())
        cue_distribution = {k: v/total_cues for k, v in mode_cues.items()} if total_cues > 0 else {}
        
        # Identify strongest cue
        strongest_cue = max(mode_cues.items(), key=lambda x: x[1]) if mode_cues else ("default", 0)
        
        return {
            "mode_cues": mode_cues,
            "cue_distribution": cue_distribution,
            "strongest_cue": strongest_cue,
            "total_cue_strength": total_cues,
            "input_length": len(user_input.split())
        }
    
    def _detect_dominance_cues(self, text: str) -> float:
        """Detect dominance mode cues in text"""
        cue_words = [
            "control", "command", "dominate", "obey", "submit", "discipline",
            "strict", "authority", "power", "serve", "master", "mistress"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Check for explicit requests
        explicit_patterns = ["dominate me", "control me", "be strict", "command me"]
        explicit_count = sum(1 for pattern in explicit_patterns if pattern in text)
        
        return min(1.0, cue_count * 0.2 + explicit_count * 0.5)
    
    def _detect_friendly_cues(self, text: str) -> float:
        """Detect friendly mode cues in text"""
        cue_words = [
            "friend", "chat", "talk", "casual", "hey", "hi", "hello",
            "how are you", "what's up", "nice", "cool", "awesome"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Informal language patterns
        informal_markers = ["!", "lol", "haha", ":)", "😊", "😄"]
        informal_count = sum(1 for marker in informal_markers if marker in text)
        
        return min(1.0, cue_count * 0.15 + informal_count * 0.2)
    
    def _detect_intellectual_cues(self, text: str) -> float:
        """Detect intellectual mode cues in text"""
        cue_words = [
            "explain", "understand", "why", "how", "theory", "concept",
            "analyze", "think", "reason", "logic", "evidence", "research"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Question patterns
        question_words = ["what", "why", "how", "when", "where", "which"]
        question_count = sum(1 for word in question_words if text.startswith(word))
        
        return min(1.0, cue_count * 0.2 + question_count * 0.3)
    
    def _detect_compassionate_cues(self, text: str) -> float:
        """Detect compassionate mode cues in text"""
        cue_words = [
            "feel", "feeling", "help", "support", "understand", "comfort",
            "care", "worry", "concern", "emotional", "heart", "difficult"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Emotional expression patterns
        emotion_patterns = ["i feel", "i'm feeling", "makes me", "i'm worried", "i need support"]
        emotion_count = sum(1 for pattern in emotion_patterns if pattern in text)
        
        return min(1.0, cue_count * 0.15 + emotion_count * 0.3)
    
    def _detect_playful_cues(self, text: str) -> float:
        """Detect playful mode cues in text"""
        cue_words = [
            "play", "fun", "game", "laugh", "joke", "silly", "funny",
            "humor", "witty", "tease", "amusing", "entertain"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Playful patterns
        playful_markers = ["haha", "lol", "😂", "🤣", "jk", "xD", ":P"]
        playful_count = sum(1 for marker in playful_markers if marker in text)
        
        return min(1.0, cue_count * 0.2 + playful_count * 0.25)
    
    def _detect_creative_cues(self, text: str) -> float:
        """Detect creative mode cues in text"""
        cue_words = [
            "create", "imagine", "story", "idea", "invent", "design",
            "artistic", "creative", "inspiration", "dream", "fantasy", "write"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Creative request patterns
        creative_patterns = ["let's create", "tell me a story", "imagine if", "what if"]
        creative_count = sum(1 for pattern in creative_patterns if pattern in text)
        
        return min(1.0, cue_count * 0.2 + creative_count * 0.3)
    
    def _detect_professional_cues(self, text: str) -> float:
        """Detect professional mode cues in text"""
        cue_words = [
            "professional", "work", "business", "formal", "efficient",
            "task", "project", "deadline", "meeting", "report", "analysis"
        ]
        
        cue_count = sum(1 for word in cue_words if word in text)
        
        # Formal language indicators
        formal_patterns = ["could you please", "would you mind", "i would like", "kindly"]
        formal_count = sum(1 for pattern in formal_patterns if pattern in text)
        
        return min(1.0, cue_count * 0.2 + formal_count * 0.25)
    
    async def _analyze_mode_adjustment_need(self, mode_cues: Dict[str, Any], 
                                          messages: Dict[str, List[Dict]], 
                                          context: SharedContext) -> Dict[str, Any]:
        """Analyze whether mode adjustment is needed"""
        current_dist = self._get_current_distribution()
        cue_dist = mode_cues.get("cue_distribution", {})
        
        # Calculate divergence between current and cued distribution
        divergence = 0.0
        for mode in set(current_dist.keys()).union(set(cue_dist.keys())):
            current_weight = current_dist.get(mode, 0.0)
            cue_weight = cue_dist.get(mode, 0.0)
            divergence += abs(current_weight - cue_weight)
        
        # Check for strong mode cues
        strongest_cue_mode, strongest_cue_strength = mode_cues.get("strongest_cue", ("default", 0))
        
        # Determine if adjustment needed
        adjustment_needed = False
        reason = "stable"
        
        if divergence > 0.5:
            adjustment_needed = True
            reason = "significant_divergence"
        elif strongest_cue_strength > 0.7 and current_dist.get(strongest_cue_mode, 0.0) < 0.4:
            adjustment_needed = True
            reason = f"strong_{strongest_cue_mode}_cue"
        elif self._check_critical_updates(messages):
            adjustment_needed = True
            reason = "critical_module_update"
        
        return {
            "adjustment_needed": adjustment_needed,
            "reason": reason,
            "divergence": divergence,
            "previous_distribution": current_dist.copy(),
            "suggested_distribution": cue_dist
        }
    
    def _check_critical_updates(self, messages: Dict[str, List[Dict]]) -> bool:
        """Check for critical updates from other modules"""
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("priority") == "CRITICAL":
                    return True
        return False
    
    async def _comprehensive_mode_update(self, context: SharedContext, 
                                       mode_cues: Dict[str, Any], 
                                       messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform comprehensive mode update considering all factors"""
        # Start with cue distribution as base
        new_distribution = mode_cues.get("cue_distribution", {}).copy()
        
        # Blend with current distribution for stability
        current_dist = self._get_current_distribution()
        stability_factor = 0.3  # How much to preserve current state
        
        for mode in set(current_dist.keys()).union(set(new_distribution.keys())):
            current_weight = current_dist.get(mode, 0.0)
            new_weight = new_distribution.get(mode, 0.0)
            new_distribution[mode] = current_weight * stability_factor + new_weight * (1 - stability_factor)
        
        # Apply influences from cross-module messages
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg["type"] == "mode_influence":
                    influence_data = msg["data"]
                    for mode, influence in influence_data.items():
                        if mode in new_distribution:
                            new_distribution[mode] = min(1.0, new_distribution[mode] + influence * 0.2)
        
        # Normalize distribution
        total = sum(new_distribution.values())
        if total > 0:
            new_distribution = {k: v/total for k, v in new_distribution.items()}
        else:
            new_distribution = {"default": 1.0}
        
        # Update using original manager
        update_result = await self._apply_mode_distribution(new_distribution, "comprehensive_update")
        
        return update_result
    
    async def _apply_mode_distribution(self, distribution: Dict[str, float], reason: str) -> Dict[str, Any]:
        """Apply a new mode distribution"""
        # Create mode distribution object if needed
        if hasattr(self.original_manager, 'context') and hasattr(self.original_manager.context, 'mode_distribution'):
            # Update the distribution directly
            for mode, weight in distribution.items():
                if hasattr(self.original_manager.context.mode_distribution, mode):
                    setattr(self.original_manager.context.mode_distribution, mode, weight)
            
            # Normalize
            self.original_manager.context.mode_distribution = self.original_manager.context.mode_distribution.normalize()
            
            # Get the result
            result = {
                "mode_distribution": self.original_manager.context.mode_distribution.dict(),
                "primary_mode": self.original_manager.context.mode_distribution.primary_mode[0],
                "confidence": 0.7,  # Default confidence
                "active_modes": self.original_manager.context.mode_distribution.active_modes
            }
        else:
            # Fallback
            result = {
                "mode_distribution": distribution,
                "primary_mode": max(distribution.items(), key=lambda x: x[1])[0] if distribution else "default",
                "confidence": 0.7,
                "active_modes": [(k, v) for k, v in distribution.items() if v > 0.2]
            }
        
        # Track the transition
        self._track_mode_transition(result)
        
        return result
    
    def _get_current_distribution(self) -> Dict[str, float]:
        """Get current mode distribution"""
        if hasattr(self.original_manager, 'context') and hasattr(self.original_manager.context, 'mode_distribution'):
            return self.original_manager.context.mode_distribution.dict()
        
        # Fallback
        return {"default": 1.0}
    
    def _track_mode_transition(self, mode_result: Dict[str, Any]):
        """Track mode transitions for analysis"""
        transition_record = {
            "timestamp": datetime.now().isoformat(),
            "distribution": mode_result.get("mode_distribution", {}),
            "primary_mode": mode_result.get("primary_mode", "default"),
            "confidence": mode_result.get("confidence", 0.5)
        }
        
        self.mode_transition_history.append(transition_record)
        
        # Limit history size
        if len(self.mode_transition_history) > 50:
            self.mode_transition_history = self.mode_transition_history[-50:]
    
    async def _calculate_mode_stability(self) -> float:
        """Calculate how stable the current mode is"""
        if len(self.mode_transition_history) < 2:
            return 1.0  # Assume stable if no history
        
        # Look at recent transitions
        recent_transitions = self.mode_transition_history[-5:]
        
        # Calculate variance in primary mode
        primary_modes = [t["primary_mode"] for t in recent_transitions]
        mode_changes = sum(1 for i in range(1, len(primary_modes)) if primary_modes[i] != primary_modes[i-1])
        
        # Calculate distribution stability
        distribution_changes = []
        for i in range(1, len(recent_transitions)):
            prev_dist = recent_transitions[i-1]["distribution"]
            curr_dist = recent_transitions[i]["distribution"]
            
            change = sum(abs(curr_dist.get(mode, 0) - prev_dist.get(mode, 0)) 
                        for mode in set(prev_dist.keys()).union(set(curr_dist.keys())))
            distribution_changes.append(change)
        
        avg_distribution_change = sum(distribution_changes) / len(distribution_changes) if distribution_changes else 0
        
        # Calculate stability score (1.0 = very stable, 0.0 = very unstable)
        mode_stability = 1.0 - (mode_changes / len(primary_modes))
        distribution_stability = 1.0 - min(1.0, avg_distribution_change)
        
        return (mode_stability + distribution_stability) / 2
    
    async def _analyze_mode_coherence(self, distribution: Dict[str, float], context: SharedContext) -> Dict[str, Any]:
        """Analyze coherence of current mode distribution"""
        # Check for conflicting modes
        conflicts = []
        
        conflict_pairs = [
            ("dominant", "compassionate"),
            ("playful", "professional"),
            ("dominant", "friendly")  # To some extent
        ]
        
        for mode1, mode2 in conflict_pairs:
            weight1 = distribution.get(mode1, 0.0)
            weight2 = distribution.get(mode2, 0.0)
            
            if weight1 > 0.3 and weight2 > 0.3:
                conflicts.append({
                    "modes": (mode1, mode2),
                    "weights": (weight1, weight2),
                    "severity": min(weight1, weight2)
                })
        
        # Check mode concentration
        active_modes = [(k, v) for k, v in distribution.items() if v > 0.2]
        concentration = len(active_modes)
        
        # Calculate coherence score
        coherence_score = 1.0
        coherence_score -= sum(c["severity"] for c in conflicts) * 0.5
        coherence_score -= max(0, concentration - 3) * 0.1  # Penalty for too many active modes
        
        return {
            "coherence_score": max(0.0, coherence_score),
            "conflicts": conflicts,
            "active_mode_count": concentration,
            "is_coherent": coherence_score > 0.6
        }
    
    async def _analyze_mode_effectiveness(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze effectiveness of current mode selection"""
        # This would ideally use feedback and reward signals
        # For now, we'll use heuristics
        
        current_dist = self._get_current_distribution()
        effectiveness_factors = {
            "alignment_with_context": 0.7,  # Default
            "user_satisfaction": 0.6,  # Would come from feedback
            "goal_achievement": 0.5,  # Would come from goal manager
            "emotional_appropriateness": 0.7  # Would come from emotional core
        }
        
        # Check if we have feedback data
        if hasattr(context, "session_context") and "user_satisfaction" in context.session_context:
            effectiveness_factors["user_satisfaction"] = context.session_context["user_satisfaction"]
        
        # Calculate overall effectiveness
        overall_effectiveness = sum(effectiveness_factors.values()) / len(effectiveness_factors)
        
        return {
            "overall_effectiveness": overall_effectiveness,
            "factors": effectiveness_factors,
            "recommendations": self._generate_effectiveness_recommendations(effectiveness_factors)
        }
    
    def _generate_effectiveness_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Generate recommendations based on effectiveness analysis"""
        recommendations = []
        
        if factors["alignment_with_context"] < 0.5:
            recommendations.append("Consider adjusting mode to better match conversation context")
        
        if factors["user_satisfaction"] < 0.5:
            recommendations.append("Monitor user responses for satisfaction indicators")
        
        if factors["emotional_appropriateness"] < 0.6:
            recommendations.append("Ensure mode aligns with emotional tone of interaction")
        
        return recommendations
    
    async def _analyze_cross_module_alignment(self, distribution: Dict[str, float], 
                                            messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze alignment with other modules"""
        alignments = {}
        
        # Check emotional alignment
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotional_state = msg["data"].get("dominant_emotion")
                        if emotional_state:
                            alignments["emotional"] = self._calculate_emotion_mode_alignment(
                                distribution, emotional_state
                            )
        
        # Check goal alignment
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg["type"] == "goal_context_available":
                        active_goals = msg["data"].get("active_goals", [])
                        alignments["goals"] = self._calculate_goal_mode_alignment(
                            distribution, active_goals
                        )
        
        # Calculate overall alignment
        if alignments:
            overall_alignment = sum(alignments.values()) / len(alignments)
        else:
            overall_alignment = 0.5  # Neutral if no data
        
        return {
            "alignments": alignments,
            "overall_alignment": overall_alignment,
            "is_well_aligned": overall_alignment > 0.6
        }
    
    def _calculate_emotion_mode_alignment(self, distribution: Dict[str, float], 
                                        emotional_state: Tuple[str, float]) -> float:
        """Calculate alignment between mode and emotional state"""
        emotion_name, strength = emotional_state
        
        # Define emotion-mode affinities
        affinities = {
            "Joy": {"playful": 0.9, "friendly": 0.8, "creative": 0.7},
            "Confidence": {"dominant": 0.9, "professional": 0.7},
            "Anxiety": {"compassionate": 0.8, "friendly": 0.7},
            "Curiosity": {"intellectual": 0.9, "creative": 0.7}
        }
        
        emotion_affinities = affinities.get(emotion_name, {})
        
        # Calculate weighted alignment
        alignment = 0.0
        total_weight = 0.0
        
        for mode, weight in distribution.items():
            if weight > 0.1:
                affinity = emotion_affinities.get(mode, 0.5)  # Default neutral affinity
                alignment += affinity * weight
                total_weight += weight
        
        return alignment / total_weight if total_weight > 0 else 0.5
    
    def _calculate_goal_mode_alignment(self, distribution: Dict[str, float], 
                                     active_goals: List[Dict[str, Any]]) -> float:
        """Calculate alignment between mode and active goals"""
        if not active_goals:
            return 0.5  # Neutral if no goals
        
        # Extract goal modes
        goal_modes = {}
        for goal in active_goals:
            mode = goal.get("source_mode", goal.get("source", "").replace("_mode", ""))
            if mode:
                goal_modes[mode] = goal_modes.get(mode, 0) + 1
        
        # Calculate alignment
        alignment = 0.0
        for mode, count in goal_modes.items():
            mode_weight = distribution.get(mode, 0.0)
            alignment += mode_weight * (count / len(active_goals))
        
        return alignment
    
    async def _generate_mode_recommendations(self, coherence: Dict[str, Any], 
                                           effectiveness: Dict[str, Any], 
                                           alignment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for mode improvement"""
        recommendations = []
        
        # Coherence recommendations
        if not coherence["is_coherent"]:
            for conflict in coherence["conflicts"]:
                modes = conflict["modes"]
                recommendations.append(f"Reduce conflict between {modes[0]} and {modes[1]} modes")
        
        if coherence["active_mode_count"] > 4:
            recommendations.append("Consider focusing on fewer modes for clearer interaction")
        
        # Effectiveness recommendations
        recommendations.extend(effectiveness["recommendations"])
        
        # Alignment recommendations
        if not alignment["is_well_aligned"]:
            if "emotional" in alignment["alignments"] and alignment["alignments"]["emotional"] < 0.5:
                recommendations.append("Adjust mode to better match emotional state")
            
            if "goals" in alignment["alignments"] and alignment["alignments"]["goals"] < 0.5:
                recommendations.append("Align mode selection with active goals")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_recent_mode_history(self) -> List[Dict[str, Any]]:
        """Get recent mode transition history"""
        return self.mode_transition_history[-10:] if self.mode_transition_history else []
    
    async def _generate_comprehensive_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Generate comprehensive mode guidance"""
        if hasattr(self.original_manager, 'get_current_mode_guidance'):
            # Use original manager's guidance generation
            return await self.original_manager.get_current_mode_guidance()
        
        # Fallback guidance
        current_dist = self._get_current_distribution()
        primary_mode = max(current_dist.items(), key=lambda x: x[1])[0] if current_dist else "default"
        
        return {
            "primary_mode": primary_mode,
            "mode_distribution": current_dist,
            "mode_description": f"Operating in {primary_mode} mode",
            "behavioral_guidelines": self._get_mode_guidelines(primary_mode)
        }
    
    def _get_mode_guidelines(self, mode: str) -> List[str]:
        """Get behavioral guidelines for a mode"""
        guidelines = {
            "dominant": [
                "Project confidence and authority",
                "Use direct, commanding language",
                "Set clear expectations",
                "Maintain control of the interaction"
            ],
            "friendly": [
                "Be warm and approachable",
                "Use casual, conversational tone",
                "Show genuine interest",
                "Build rapport naturally"
            ],
            "intellectual": [
                "Focus on clarity and precision",
                "Provide thorough explanations",
                "Encourage analytical thinking",
                "Use evidence-based reasoning"
            ],
            "compassionate": [
                "Show empathy and understanding",
                "Validate emotions",
                "Offer support and comfort",
                "Listen actively"
            ],
            "playful": [
                "Be light-hearted and fun",
                "Use appropriate humor",
                "Encourage creativity",
                "Keep things engaging"
            ],
            "creative": [
                "Think outside the box",
                "Use vivid descriptions",
                "Encourage imagination",
                "Explore possibilities"
            ],
            "professional": [
                "Maintain formal tone",
                "Focus on efficiency",
                "Provide clear information",
                "Respect boundaries"
            ],
            "default": [
                "Balance warmth and professionalism",
                "Adapt to user's needs",
                "Maintain helpful stance",
                "Be consistently reliable"
            ]
        }
        
        return guidelines.get(mode, guidelines["default"])
    
    async def _get_blended_parameters_for_distribution(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """Get blended parameters for current distribution"""
        if hasattr(self.original_manager, 'get_blended_parameters'):
            return await self.original_manager.get_blended_parameters()
        
        # Fallback - simple parameter blending
        blended = {
            "formality": 0.5,
            "assertiveness": 0.5,
            "warmth": 0.6,
            "vulnerability": 0.4,
            "directness": 0.6,
            "depth": 0.5,
            "humor": 0.4,
            "emotional_expression": 0.5
        }
        
        # Adjust based on distribution
        for mode, weight in distribution.items():
            if weight > 0.1:
                if mode == "dominant":
                    blended["assertiveness"] += weight * 0.3
                    blended["formality"] -= weight * 0.2
                elif mode == "friendly":
                    blended["warmth"] += weight * 0.3
                    blended["formality"] -= weight * 0.3
                elif mode == "intellectual":
                    blended["depth"] += weight * 0.3
                    blended["formality"] += weight * 0.2
                elif mode == "compassionate":
                    blended["warmth"] += weight * 0.3
                    blended["vulnerability"] += weight * 0.2
                elif mode == "playful":
                    blended["humor"] += weight * 0.4
                    blended["formality"] -= weight * 0.3
                elif mode == "professional":
                    blended["formality"] += weight * 0.3
                    blended["directness"] += weight * 0.2
        
        # Normalize values
        for param in blended:
            blended[param] = max(0.0, min(1.0, blended[param]))
        
        return blended
    
    async def _generate_style_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Generate conversation style guidance"""
        current_dist = self._get_current_distribution()
        
        # Collect style elements from active modes
        style_elements = {
            "tone": [],
            "language_patterns": [],
            "topics_to_emphasize": [],
            "topics_to_avoid": []
        }
        
        # Add elements based on active modes
        for mode, weight in current_dist.items():
            if weight > 0.2:
                if mode == "dominant":
                    style_elements["tone"].append("commanding")
                    style_elements["language_patterns"].append("direct imperatives")
                    style_elements["dominant_patterns"] = [
                        "You will...", "I expect...", "Do as I say"
                    ]
                elif mode == "friendly":
                    style_elements["tone"].append("warm")
                    style_elements["language_patterns"].append("casual conversation")
                elif mode == "intellectual":
                    style_elements["tone"].append("thoughtful")
                    style_elements["language_patterns"].append("analytical reasoning")
                elif mode == "compassionate":
                    style_elements["tone"].append("supportive")
                    style_elements["language_patterns"].append("empathetic validation")
                elif mode == "playful":
                    style_elements["tone"].append("light-hearted")
                    style_elements["language_patterns"].append("humor and wit")
                elif mode == "creative":
                    style_elements["tone"].append("imaginative")
                    style_elements["language_patterns"].append("vivid descriptions")
                elif mode == "professional":
                    style_elements["tone"].append("formal")
                    style_elements["language_patterns"].append("clear and concise")
        
        return style_elements
    
    async def _get_blended_vocalization_patterns(self) -> Dict[str, Any]:
        """Get blended vocalization patterns"""
        current_dist = self._get_current_distribution()
        
        vocalization = {
            "pronouns": ["I", "you", "we"],
            "key_phrases": [],
            "address_forms": []
        }
        
        # Add vocalization elements from active modes
        for mode, weight in current_dist.items():
            if weight > 0.2:
                if mode == "dominant":
                    vocalization["key_phrases"].extend([
                        "You will obey", "I command you", "Submit to me"
                    ])
                    vocalization["address_forms"].extend(["pet", "dear one"])
                elif mode == "friendly":
                    vocalization["key_phrases"].extend([
                        "That's great!", "I'm happy to help", "Let's chat"
                    ])
                elif mode == "intellectual":
                    vocalization["key_phrases"].extend([
                        "Let me explain", "Consider this", "The evidence suggests"
                    ])
                elif mode == "compassionate":
                    vocalization["key_phrases"].extend([
                        "I understand", "That must be difficult", "I'm here for you"
                    ])
                elif mode == "playful":
                    vocalization["key_phrases"].extend([
                        "How fun!", "Let's play", "That's hilarious"
                    ])
        
        # Limit phrases to avoid repetition
        vocalization["key_phrases"] = vocalization["key_phrases"][:6]
        
        return vocalization
    
    async def _check_mode_expression_requirements(self, context: SharedContext) -> Dict[str, Any]:
        """Check if specific modes require expression"""
        current_dist = self._get_current_distribution()
        requirements = {}
        
        # Check dominant mode requirements
        dominant_weight = current_dist.get("dominant", 0.0)
        if dominant_weight > 0.6:
            requirements["dominant_expression_required"] = True
            requirements["dominance_level"] = dominant_weight
            requirements["expression_type"] = "establish_control" if context.processing_stage == "input" else "maintain_authority"
        
        # Check other mode requirements
        primary_mode = max(current_dist.items(), key=lambda x: x[1])[0] if current_dist else "default"
        
        if primary_mode == "compassionate" and current_dist[primary_mode] > 0.7:
            requirements["compassionate_expression_required"] = True
            requirements["expression_focus"] = "emotional_support"
        
        elif primary_mode == "intellectual" and current_dist[primary_mode] > 0.7:
            requirements["intellectual_expression_required"] = True
            requirements["expression_focus"] = "analytical_depth"
        
        return requirements
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original mode manager"""
        return getattr(self.original_manager, name)
