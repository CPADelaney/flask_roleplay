# nyx/core/a2a/context_aware_conditioning.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareConditioningSystem(ContextAwareModule):
    """
    Enhanced ConditioningSystem with full context distribution capabilities
    """
    
    def __init__(self, original_conditioning_system):
        super().__init__("conditioning_system")
        self.original_system = original_conditioning_system
        self.context_subscriptions = [
            "emotional_state_update", "behavior_evaluation", "reward_signal",
            "goal_progress", "relationship_state_change", "dominance_action",
            "memory_retrieval_complete", "context_detection_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize conditioning processing for this context"""
        logger.debug(f"ConditioningSystem received context for user: {context.user_id}")
        
        # Analyze user input for potential conditioning triggers
        conditioning_implications = await self._analyze_input_for_conditioning(context.user_input)
        
        # Check for active conditioned associations
        active_associations = await self._get_relevant_associations(context)
        
        # Send initial conditioning context to other modules
        await self.send_context_update(
            update_type="conditioning_context_available",
            data={
                "conditioning_implications": conditioning_implications,
                "active_associations": active_associations,
                "total_classical": len(self.original_system.context.classical_associations),
                "total_operant": len(self.original_system.context.operant_associations),
                "conditioning_ready": True
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect conditioning"""
        
        if update.update_type == "emotional_state_update":
            # Emotional states can trigger conditioning
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                await self._process_emotional_conditioning(dominant_emotion, emotional_data)
        
        elif update.update_type == "behavior_evaluation":
            # Behavior evaluations inform operant conditioning
            behavior_data = update.data
            behavior = behavior_data.get("behavior")
            outcome = behavior_data.get("outcome")
            
            if behavior and outcome:
                await self._process_behavior_conditioning(behavior, outcome)
        
        elif update.update_type == "reward_signal":
            # Direct reward signals for conditioning
            reward_data = update.data
            await self._process_reward_conditioning(reward_data)
        
        elif update.update_type == "goal_progress":
            # Goal progress can reinforce behaviors
            goal_data = update.data
            if goal_data.get("goal_completed"):
                await self._reinforce_goal_behaviors(goal_data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changes affect social conditioning
            relationship_data = update.data
            await self._adjust_social_conditioning(relationship_data)
        
        elif update.update_type == "dominance_action":
            # Dominance actions trigger specific conditioning
            dominance_data = update.data
            await self._process_dominance_conditioning(dominance_data)
        
        elif update.update_type == "context_detection_update":
            # Context changes affect conditioning priorities
            context_data = update.data
            await self._adjust_conditioning_priorities(context_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context-aware conditioning"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Check for conditioning triggers in input
        triggers = await self._detect_conditioning_triggers(context.user_input, messages)
        
        # Process any triggered responses
        triggered_responses = []
        for trigger in triggers:
            response = await self.original_system.trigger_conditioned_response(
                stimulus=trigger["stimulus"],
                context={
                    "user_id": context.user_id,
                    "session_context": context.session_context,
                    "cross_module_context": messages
                }
            )
            if response:
                triggered_responses.append(response)
        
        # Apply conditioning based on current context
        conditioning_results = await self._apply_contextual_conditioning(context, messages)
        
        # Send conditioning updates
        if triggered_responses or conditioning_results:
            await self.send_context_update(
                update_type="conditioning_triggered",
                data={
                    "triggered_responses": triggered_responses,
                    "conditioning_results": conditioning_results,
                    "trigger_count": len(triggers)
                }
            )
        
        return {
            "conditioning_processed": True,
            "triggers_detected": len(triggers),
            "responses_triggered": len(triggered_responses),
            "contextual_conditioning": conditioning_results
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze conditioning state in current context"""
        # Get active associations relevant to context
        relevant_associations = await self._analyze_contextual_associations(context)
        
        # Analyze conditioning opportunities
        opportunities = await self._identify_conditioning_opportunities(context)
        
        # Check conditioning coherence with other systems
        coherence = await self._check_conditioning_coherence(context)
        
        return {
            "relevant_associations": relevant_associations,
            "conditioning_opportunities": opportunities,
            "coherence_analysis": coherence,
            "conditioning_health": await self._assess_conditioning_health()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize conditioning influences for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine conditioning influences on response
        conditioning_influence = {
            "behavioral_biases": await self._calculate_behavioral_biases(context),
            "emotional_triggers": await self._identify_active_emotional_triggers(context),
            "reinforcement_suggestions": await self._suggest_reinforcements(context, messages),
            "response_modulations": await self._calculate_response_modulations(context)
        }
        
        # Check for conditioning opportunities in response
        response_conditioning = await self._plan_response_conditioning(context, messages)
        
        return {
            "conditioning_influence": conditioning_influence,
            "response_conditioning": response_conditioning,
            "synthesis_complete": True
        }
    
    # Helper methods
    
    async def _analyze_input_for_conditioning(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for conditioning implications"""
        implications = {
            "contains_triggers": False,
            "suggests_reinforcement": False,
            "indicates_punishment": False,
            "conditioning_opportunity": False
        }
        
        input_lower = user_input.lower()
        
        # Check for submission language (positive reinforcement opportunity)
        if any(phrase in input_lower for phrase in ["yes mistress", "as you wish", "i obey"]):
            implications["contains_triggers"] = True
            implications["suggests_reinforcement"] = True
            implications["trigger_type"] = "submission"
        
        # Check for defiance (potential punishment)
        elif any(phrase in input_lower for phrase in ["no way", "won't do", "refuse"]):
            implications["contains_triggers"] = True
            implications["indicates_punishment"] = True
            implications["trigger_type"] = "defiance"
        
        # Check for learning opportunities
        if "?" in user_input or "teach me" in input_lower:
            implications["conditioning_opportunity"] = True
            implications["opportunity_type"] = "learning"
        
        return implications
    
    async def _get_relevant_associations(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get associations relevant to current context"""
        relevant = []
        
        # Check classical associations
        for key, assoc in self.original_system.context.classical_associations.items():
            relevance = self._calculate_association_relevance(assoc, context)
            if relevance > 0.3:
                relevant.append({
                    "key": key,
                    "type": "classical",
                    "stimulus": assoc.stimulus,
                    "response": assoc.response,
                    "strength": assoc.association_strength,
                    "relevance": relevance
                })
        
        # Check operant associations
        for key, assoc in self.original_system.context.operant_associations.items():
            relevance = self._calculate_association_relevance(assoc, context)
            if relevance > 0.3:
                relevant.append({
                    "key": key,
                    "type": "operant",
                    "behavior": assoc.stimulus,
                    "consequence": assoc.response,
                    "strength": assoc.association_strength,
                    "relevance": relevance
                })
        
        # Sort by relevance
        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant[:10]  # Top 10 most relevant
    
    def _calculate_association_relevance(self, association, context: SharedContext) -> float:
        """Calculate how relevant an association is to current context"""
        relevance = 0.0
        
        # Check if association matches current emotional state
        if context.emotional_state:
            valence_match = 1.0 - abs(association.valence - context.emotional_state.get("valence", 0))
            relevance += valence_match * 0.3
        
        # Check if association matches current goals
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if goal.get("associated_need") and goal["associated_need"] in association.stimulus.lower():
                    relevance += 0.3
        
        # Check context key matches
        if association.context_keys:
            context_matches = sum(1 for key in association.context_keys 
                                if key in context.session_context.get("context_keys", []))
            relevance += (context_matches / len(association.context_keys)) * 0.4
        
        return min(1.0, relevance)
    
    async def _process_emotional_conditioning(self, emotion: tuple, emotional_data: Dict[str, Any]):
        """Process conditioning based on emotional state"""
        emotion_name, strength = emotion
        
        # Create emotion-based conditioning
        if strength > 0.6:
            # Strong emotions create conditioning opportunities
            if emotion_name in ["Joy", "Satisfaction"]:
                # Positive emotions reinforce recent behaviors
                await self._reinforce_recent_behaviors(strength)
            elif emotion_name in ["Frustration", "Anger"]:
                # Negative emotions may punish recent behaviors
                await self._punish_recent_behaviors(strength * 0.7)
    
    async def _reinforce_recent_behaviors(self, strength: float):
        """Reinforce recently expressed behaviors"""
        # This would track recent behaviors and apply reinforcement
        # For now, send update about reinforcement opportunity
        await self.send_context_update(
            update_type="reinforcement_opportunity",
            data={
                "type": "positive_emotion",
                "strength": strength,
                "target": "recent_behaviors"
            },
            priority=ContextPriority.NORMAL
        )
    
    async def _detect_conditioning_triggers(self, user_input: str, messages: Dict) -> List[Dict[str, Any]]:
        """Detect conditioning triggers in input and context"""
        triggers = []
        
        # Pattern detection from original system
        patterns = self.original_system._detect_patterns(user_input)
        
        for pattern in patterns:
            triggers.append({
                "stimulus": pattern,
                "source": "input_pattern",
                "confidence": 0.8
            })
        
        # Check for triggers from other modules
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotion = msg["data"].get("dominant_emotion")
                        if emotion and emotion[1] > 0.7:  # Strong emotion
                            triggers.append({
                                "stimulus": f"strong_{emotion[0].lower()}",
                                "source": "emotional_state",
                                "confidence": emotion[1]
                            })
        
        return triggers
    
    async def _apply_contextual_conditioning(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Apply conditioning based on full context"""
        results = {
            "classical_conditioning": [],
            "operant_conditioning": [],
            "contextual_adjustments": []
        }
        
        # Check if we should create new associations based on context
        if context.emotional_state and context.emotional_state.get("valence", 0) > 0.7:
            # Positive emotional context - good time for conditioning
            if "goal_progress" in messages:
                # Associate current context with goal progress
                goal_msg = messages["goal_progress"][-1] if messages["goal_progress"] else None
                if goal_msg and goal_msg["data"].get("progress_made"):
                    # Create positive association
                    result = await self.original_system.process_classical_conditioning(
                        unconditioned_stimulus="goal_progress",
                        conditioned_stimulus=context.user_input[:50],  # First 50 chars as stimulus
                        response="satisfaction",
                        intensity=0.7,
                        context={"source": "contextual_conditioning"}
                    )
                    results["classical_conditioning"].append(result)
        
        return results
    
    async def _identify_conditioning_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify opportunities for new conditioning"""
        opportunities = []
        
        # Check for unassociated positive experiences
        if context.emotional_state and context.emotional_state.get("valence", 0) > 0.6:
            opportunities.append({
                "type": "positive_experience",
                "target": "current_interaction",
                "method": "classical_conditioning",
                "priority": 0.7
            })
        
        # Check for goal-related opportunities
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if goal.get("priority", 0) > 0.7:
                    opportunities.append({
                        "type": "goal_reinforcement",
                        "target": goal["description"],
                        "method": "operant_conditioning",
                        "priority": goal["priority"]
                    })
        
        return opportunities
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original conditioning system"""
        return getattr(self.original_system, name)
