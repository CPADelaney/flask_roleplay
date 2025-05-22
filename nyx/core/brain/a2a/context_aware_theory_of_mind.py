# nyx/core/brain/a2a/context_aware_theory_of_mind.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority
from nyx.core.theory_of_mind import TheoryOfMind, UserMentalState, SubmissionMarkers

logger = logging.getLogger(__name__)

class ContextAwareTheoryOfMind(ContextAwareModule):
    """
    Enhanced Theory of Mind with A2A context distribution capabilities
    """
    
    def __init__(self, original_theory_of_mind: TheoryOfMind):
        super().__init__("theory_of_mind")
        self.original_tom = original_theory_of_mind
        self.context_subscriptions = [
            "emotional_updates",
            "relationship_updates", 
            "memory_updates",
            "goal_updates",
            "sensory_input",
            "dominance_updates"
        ]
        
    async def on_context_received(self, context: SharedContext):
        """Initialize mental state inference for this context"""
        logger.debug(f"TheoryOfMind received context for user: {context.user_id}")
        
        # Get or create user model
        user_id = context.user_id
        if user_id:
            current_model = await self.original_tom.get_user_model(user_id)
            if not current_model:
                # Create initial model
                await self.original_tom._get_or_create_user_model(user_id)
            
            # Perform initial mental state analysis
            initial_analysis = await self._analyze_initial_mental_state(context)
            
            # Send initial mental state assessment to other modules
            await self.send_context_update(
                update_type="user_mental_state_assessment",
                data={
                    "user_id": user_id,
                    "initial_analysis": initial_analysis,
                    "confidence": initial_analysis.get("confidence", 0.5),
                    "detected_intentions": initial_analysis.get("intentions", []),
                    "emotional_indicators": initial_analysis.get("emotional_indicators", {})
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that inform mental state modeling"""
        
        if update.update_type == "emotional_state_update":
            # Emotional core's assessment can inform user emotion inference
            await self._integrate_emotional_context(update.data)
            
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect trust/familiarity estimates
            await self._integrate_relationship_context(update.data)
            
        elif update.update_type == "memory_retrieval_complete":
            # Past interactions inform current mental state
            await self._integrate_memory_context(update.data)
            
        elif update.update_type == "goal_progress":
            # User's reaction to goal progress reveals their priorities
            await self._integrate_goal_context(update.data)
            
        elif update.update_type == "dominance_action_taken":
            # Track mental state changes from dominance interactions
            await self._track_dominance_response(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context for mental state inference"""
        user_id = context.user_id
        if not user_id:
            return {"error": "No user_id in context"}
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Prepare interaction data with full context
        interaction_data = {
            "user_input": context.user_input,
            "nyx_response": context.session_context.get("last_nyx_response", ""),
            "emotional_context": context.emotional_state,
            "relationship_context": context.relationship_context,
            "goal_context": context.goal_context,
            "cross_module_insights": self._extract_relevant_insights(messages)
        }
        
        # Update user model with context-aware inference
        update_result = await self.original_tom.update_user_model(user_id, interaction_data)
        
        # Detect special psychological states
        submission_analysis = await self.original_tom.detect_submission_signals(
            context.user_input, user_id
        )
        
        # Check for subspace if submission detected
        subspace_analysis = None
        if submission_analysis.get("overall_submission", 0) > 0.5:
            if hasattr(self.original_tom, "subspace_detection_system"):
                subspace_analysis = await self.original_tom.subspace_detection_system.analyze_message(
                    user_id, context.user_input
                )
        
        # Send comprehensive mental state update
        await self.send_context_update(
            update_type="user_mental_state_update",
            data={
                "user_id": user_id,
                "mental_state": update_result,
                "submission_analysis": submission_analysis,
                "subspace_analysis": subspace_analysis,
                "context_integrated": True
            },
            priority=ContextPriority.HIGH if submission_analysis.get("overall_submission", 0) > 0.7 
                     else ContextPriority.NORMAL
        )
        
        return {
            "mental_state_update": update_result,
            "submission_detected": submission_analysis.get("overall_submission", 0) > 0.5,
            "subspace_detected": subspace_analysis.get("in_subspace", False) if subspace_analysis else False,
            "context_aware_inference": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize mental state insights for response generation"""
        user_id = context.user_id
        if not user_id:
            return {}
        
        # Get current user model
        user_model = await self.original_tom.get_user_model(user_id)
        if not user_model:
            return {}
        
        # Get cross-module insights
        messages = await self.get_cross_module_messages()
        
        # Create synthesis recommendations
        synthesis = {
            "response_emotional_tone": self._recommend_emotional_tone(user_model),
            "cognitive_complexity": self._recommend_cognitive_level(user_model),
            "trust_based_openness": self._calculate_openness_level(user_model),
            "attention_focus_alignment": self._suggest_focus_alignment(user_model, context),
            "psychological_safety_check": self._assess_psychological_safety(user_model, messages)
        }
        
        # Special handling for submission/subspace states
        if hasattr(self, "_last_submission_analysis"):
            submission_data = self._last_submission_analysis.get(user_id, {})
            if submission_data.get("overall_submission", 0) > 0.5:
                synthesis["dominance_response_style"] = self._recommend_dominance_style(
                    submission_data, user_model
                )
        
        return synthesis
    
    async def _analyze_initial_mental_state(self, context: SharedContext) -> Dict[str, Any]:
        """Perform initial mental state analysis from context"""
        # Use linguistic patterns to infer initial state
        linguistic_analysis = await self.original_tom.get_linguistic_patterns(context.user_input)
        emotional_markers = await self.original_tom.get_emotional_markers(context.user_input)
        
        # Infer intentions from patterns
        intentions = []
        if linguistic_analysis.get("inquiry"):
            intentions.append("seeking_information")
        if linguistic_analysis.get("directive"):
            intentions.append("requesting_action")
        if linguistic_analysis.get("self_focus_ratio", 0) > 0.7:
            intentions.append("self_expression")
        
        # Calculate initial confidence
        confidence = 0.5
        if emotional_markers.get("has_strong_markers"):
            confidence += 0.2
        if len(intentions) > 0:
            confidence += 0.1
        
        return {
            "intentions": intentions,
            "emotional_indicators": emotional_markers,
            "linguistic_patterns": linguistic_analysis,
            "confidence": min(confidence, 0.8)
        }
    
    async def _integrate_emotional_context(self, emotional_data: Dict[str, Any]):
        """Integrate emotional assessment from EmotionalCore"""
        # Store for cross-reference during inference
        self._emotional_context = emotional_data
        
        # If strong Nyx emotions detected, it might influence user state
        nyx_valence = emotional_data.get("valence", 0)
        nyx_arousal = emotional_data.get("arousal", 0.5)
        
        # High arousal from Nyx might increase user arousal (emotional contagion)
        if nyx_arousal > 0.7:
            # Note this for next user model update
            if not hasattr(self, "_context_modifiers"):
                self._context_modifiers = {}
            self._context_modifiers["arousal_boost"] = 0.1
    
    async def _integrate_relationship_context(self, relationship_data: Dict[str, Any]):
        """Integrate relationship state into mental modeling"""
        # Relationship context directly affects trust/familiarity estimates
        self._relationship_context = relationship_data.get("relationship_context", {})
    
    async def _integrate_memory_context(self, memory_data: Dict[str, Any]):
        """Use retrieved memories to inform mental state"""
        # Extract patterns from memories
        memories = memory_data.get("retrieved_memories", [])
        if memories:
            # Look for emotional patterns in past interactions
            emotional_patterns = {}
            for memory in memories:
                # Simple pattern extraction
                if "happy" in str(memory).lower():
                    emotional_patterns["positive_associations"] = emotional_patterns.get("positive_associations", 0) + 1
    
    async def _integrate_goal_context(self, goal_data: Dict[str, Any]):
        """Integrate goal progress to understand user priorities"""
        # User's reaction to goal progress reveals what they care about
        if goal_data.get("goal_completed"):
            # User likely satisfied if they contributed to goal completion
            if not hasattr(self, "_context_modifiers"):
                self._context_modifiers = {}
            self._context_modifiers["satisfaction_boost"] = 0.2
    
    async def _track_dominance_response(self, dominance_data: Dict[str, Any]):
        """Track mental state changes from dominance interactions"""
        user_id = dominance_data.get("user_id")
        if not user_id:
            return
        
        # Store submission analysis for synthesis
        if not hasattr(self, "_last_submission_analysis"):
            self._last_submission_analysis = {}
        
        submission_level = dominance_data.get("submission_level", 0)
        self._last_submission_analysis[user_id] = {
            "overall_submission": submission_level,
            "timestamp": datetime.now(),
            "action_type": dominance_data.get("action_type")
        }
    
    def _extract_relevant_insights(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Extract insights relevant to mental state from cross-module messages"""
        insights = {
            "emotional_consistency": True,
            "goal_alignment": True,
            "attention_focus_modules": []
        }
        
        # Check for emotional consistency across modules
        emotional_assessments = []
        for module, module_messages in messages.items():
            for msg in module_messages:
                if "emotion" in msg.get("type", "").lower():
                    emotional_assessments.append(msg.get("data", {}))
        
        # Simple consistency check
        if len(emotional_assessments) > 1:
            # Check if assessments agree
            valences = [a.get("valence", 0) for a in emotional_assessments if "valence" in a]
            if valences and max(valences) - min(valences) > 0.5:
                insights["emotional_consistency"] = False
        
        return insights
    
    def _recommend_emotional_tone(self, user_model: Dict[str, Any]) -> str:
        """Recommend emotional tone based on user's mental state"""
        valence = user_model.get("valence", 0)
        arousal = user_model.get("arousal", 0.5)
        emotion = user_model.get("inferred_emotion", "neutral")
        
        # Map to response tone
        if valence > 0.5 and arousal > 0.5:
            return "enthusiastic_supportive"
        elif valence > 0.5 and arousal <= 0.5:
            return "calm_appreciative"
        elif valence < -0.5 and arousal > 0.5:
            return "concerned_engaged"
        elif valence < -0.5 and arousal <= 0.5:
            return "gentle_supportive"
        else:
            return "balanced_attentive"
    
    def _recommend_cognitive_level(self, user_model: Dict[str, Any]) -> str:
        """Recommend cognitive complexity based on user state"""
        knowledge_level = user_model.get("knowledge_level", 0.5)
        attention_focus = user_model.get("attention_focus")
        
        if knowledge_level > 0.7 and attention_focus:
            return "high_complexity"
        elif knowledge_level < 0.3:
            return "simple_clear"
        else:
            return "moderate_complexity"
    
    def _calculate_openness_level(self, user_model: Dict[str, Any]) -> float:
        """Calculate how open Nyx should be based on trust"""
        perceived_trust = user_model.get("perceived_trust", 0.5)
        perceived_familiarity = user_model.get("perceived_familiarity", 0.1)
        
        # Weighted combination
        openness = (perceived_trust * 0.7) + (perceived_familiarity * 0.3)
        return min(1.0, openness)
    
    def _suggest_focus_alignment(self, user_model: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Suggest how to align with user's attention focus"""
        user_focus = user_model.get("attention_focus")
        user_goals = user_model.get("inferred_goals", [])
        
        suggestions = {
            "primary_focus": user_focus or "conversation",
            "align_with_goals": user_goals[:2] if user_goals else [],
            "cognitive_load": "low" if user_model.get("arousal", 0.5) > 0.8 else "moderate"
        }
        
        return suggestions
    
    def _assess_psychological_safety(self, user_model: Dict[str, Any], messages: Dict) -> Dict[str, Any]:
        """Assess psychological safety for various interaction types"""
        trust = user_model.get("perceived_trust", 0.5)
        emotional_state = user_model.get("inferred_emotion", "neutral")
        
        safety_assessment = {
            "overall_safety": trust > 0.6 and emotional_state not in ["fear", "anger", "disgust"],
            "vulnerability_safe": trust > 0.7,
            "playful_teasing_safe": trust > 0.6 and emotional_state in ["joy", "anticipation", "trust"],
            "deep_questions_safe": trust > 0.8,
            "dominance_play_safe": trust > 0.8 and user_model.get("overall_confidence", 0.5) > 0.6
        }
        
        return safety_assessment
    
    def _recommend_dominance_style(self, submission_data: Dict[str, Any], user_model: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend dominance interaction style based on mental state"""
        submission_level = submission_data.get("overall_submission", 0)
        submission_type = submission_data.get("submission_type", "none")
        trust = user_model.get("perceived_trust", 0.5)
        
        # Safety first
        if trust < 0.7:
            return {
                "style": "gentle_guidance",
                "intensity": 0.3,
                "techniques": ["positive_reinforcement", "clear_structure"]
            }
        
        # Match style to submission type
        if submission_type == "deference":
            return {
                "style": "benevolent_control",
                "intensity": submission_level * 0.8,
                "techniques": ["clear_commands", "earned_praise", "structure"]
            }
        elif submission_type == "vulnerability":
            return {
                "style": "protective_dominance",
                "intensity": submission_level * 0.6,
                "techniques": ["reassurance", "gentle_control", "aftercare_focus"]
            }
        elif submission_type == "subspace":
            return {
                "style": "mindful_guidance",
                "intensity": 0.5,  # Lower intensity for safety
                "techniques": ["simple_commands", "grounding", "careful_monitoring"],
                "caution": "User in altered state - prioritize safety"
            }
        else:
            return {
                "style": "exploratory_dominance",
                "intensity": 0.4,
                "techniques": ["test_boundaries", "build_trust", "clear_communication"]
            }
