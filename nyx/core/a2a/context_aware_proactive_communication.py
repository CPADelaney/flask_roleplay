# nyx/core/a2a/context_aware_proactive_communication.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareProactiveCommunication(ContextAwareModule):
    """
    Advanced ProactiveCommunicationEngine with full context distribution capabilities
    """
    
    def __init__(self, original_engine):
        super().__init__("proactive_communication")
        self.original_engine = original_engine
        self.context_subscriptions = [
            "emotional_state_update", "mood_state_change", "relationship_milestone",
            "goal_progress", "goal_completion", "need_expression", "urgent_need_expression",
            "memory_retrieval_complete", "reflection_complete", "dominance_gratification",
            "creative_expression", "temporal_milestone", "user_inactivity_detected",
            "conversation_momentum", "identity_shift", "interaction_mode_change"
        ]
        
        # Track context-informed intent generation
        self.context_informed_intents = {}
        self.suppressed_intents = {}  # Intents suppressed due to context
        
    async def on_context_received(self, context: SharedContext):
        """Initialize proactive communication processing for this context"""
        logger.debug(f"ProactiveCommunication received context for user: {context.user_id}")
        
        # Analyze context for communication opportunities
        comm_opportunities = await self._analyze_communication_opportunities(context)
        
        # Check if proactive communication is appropriate given context
        appropriateness = await self._assess_communication_appropriateness(context)
        
        # Get current active intents
        active_intents = await self.original_engine.get_active_intents()
        
        # Analyze how context affects existing intents
        intent_adjustments = await self._analyze_intent_context_fit(active_intents, context)
        
        # Send initial assessment to other modules
        await self.send_context_update(
            update_type="proactive_comm_assessment",
            data={
                "communication_opportunities": comm_opportunities,
                "appropriateness_score": appropriateness["score"],
                "appropriateness_factors": appropriateness["factors"],
                "active_intent_count": len(active_intents),
                "intent_adjustments": intent_adjustments,
                "suppression_recommended": appropriateness["score"] < 0.3
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect proactive communication"""
        
        if update.update_type == "emotional_state_update":
            # Emotional changes might trigger or suppress communication
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            emotional_state = emotional_data.get("emotional_state", {})
            
            await self._process_emotional_trigger(dominant_emotion, emotional_state)
            
        elif update.update_type == "mood_state_change":
            # Mood changes affect communication style and frequency
            mood_data = update.data
            mood_state = mood_data.get("mood_state")
            mood_intensity = mood_data.get("intensity", 0.5)
            
            await self._adjust_communication_for_mood(mood_state, mood_intensity)
            
        elif update.update_type == "relationship_milestone":
            # Relationship milestones are important communication triggers
            relationship_data = update.data
            milestone_type = relationship_data.get("milestone_type")
            user_id = relationship_data.get("user_id")
            
            if milestone_type and user_id:
                await self._create_milestone_intent(user_id, milestone_type, relationship_data)
                
        elif update.update_type == "goal_completion":
            # Goal completions might warrant celebratory messages
            goal_data = update.data
            completed_goals = goal_data.get("completed_goals", [])
            
            for goal in completed_goals:
                if goal.get("significant", False):
                    await self._create_achievement_intent(goal)
                    
        elif update.update_type in ["need_expression", "urgent_need_expression"]:
            # Need expressions might trigger supportive communication
            need_data = update.data
            expressed_needs = need_data.get("expressed_needs", [])
            urgency = need_data.get("urgency", 0.5)
            
            await self._process_need_expression(expressed_needs, urgency)
            
        elif update.update_type == "memory_retrieval_complete":
            # Retrieved memories might inspire communication
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            significant_memories = [m for m in memories if m.get("significance", 0) > 7]
            if significant_memories:
                await self._create_memory_inspired_intent(significant_memories)
                
        elif update.update_type == "reflection_complete":
            # Reflections might generate insights to share
            reflection_data = update.data
            insights = reflection_data.get("insights", [])
            
            if insights:
                await self._create_insight_sharing_intent(insights)
                
        elif update.update_type == "user_inactivity_detected":
            # User inactivity might trigger check-in messages
            inactivity_data = update.data
            user_id = inactivity_data.get("user_id")
            days_inactive = inactivity_data.get("days_inactive", 0)
            
            if user_id and days_inactive > 3:
                await self._create_checkin_intent(user_id, days_inactive)
                
        elif update.update_type == "conversation_momentum":
            # Conversation momentum affects timing
            momentum_data = update.data
            momentum_score = momentum_data.get("momentum_score", 0.5)
            
            # High momentum might suppress proactive messages
            if momentum_score > 0.8:
                await self._suppress_non_urgent_intents("high_conversation_momentum")
                
        elif update.update_type == "interaction_mode_change":
            # Mode changes affect communication style
            mode_data = update.data
            new_mode = mode_data.get("new_mode")
            
            await self._adjust_communication_for_mode(new_mode)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context awareness for proactive communication"""
        # Check if the input affects our proactive communication plans
        input_impact = await self._analyze_input_impact_on_intents(context.user_input)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Process any action-driven intent opportunities
        action_intents = []
        for module_name, module_messages in messages.items():
            if module_name == "agentic_action_generator":
                for msg in module_messages:
                    if msg['type'] == 'action_executed':
                        action_data = msg['data']
                        intent = await self._consider_action_driven_intent(action_data, context)
                        if intent:
                            action_intents.append(intent)
        
        # Re-evaluate active intents based on new context
        active_intents = await self.original_engine.get_active_intents()
        reevaluated_intents = await self._reevaluate_intents_with_context(active_intents, context, messages)
        
        # Send update about communication adjustments
        if input_impact["requires_adjustment"] or action_intents or reevaluated_intents["changes_made"]:
            await self.send_context_update(
                update_type="proactive_comm_adjusted",
                data={
                    "input_impact": input_impact,
                    "action_driven_intents": len(action_intents),
                    "reevaluation_results": reevaluated_intents,
                    "total_active_intents": len(await self.original_engine.get_active_intents())
                }
            )
        
        return {
            "proactive_comm_processed": True,
            "input_impact": input_impact,
            "action_intents_created": len(action_intents),
            "intents_adjusted": reevaluated_intents["changes_made"]
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze proactive communication in context"""
        # Get current communication state
        active_intents = await self.original_engine.get_active_intents()
        recent_sent = await self.original_engine.get_recent_sent_intents()
        
        # Analyze communication patterns
        pattern_analysis = await self._analyze_communication_patterns(active_intents, recent_sent)
        
        # Assess communication effectiveness
        effectiveness = await self._assess_communication_effectiveness(recent_sent, context)
        
        # Identify communication gaps
        gaps = await self._identify_communication_gaps(context, active_intents)
        
        # Generate recommendations
        recommendations = await self._generate_communication_recommendations(
            pattern_analysis, effectiveness, gaps, context
        )
        
        return {
            "active_intents": len(active_intents),
            "recent_communications": len(recent_sent),
            "pattern_analysis": pattern_analysis,
            "effectiveness_assessment": effectiveness,
            "identified_gaps": gaps,
            "recommendations": recommendations,
            "context_informed_analysis": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize proactive communication components for response"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine if we should mention upcoming communications
        should_preview = await self._should_preview_communications(context, messages)
        
        # Get relevant active intents for preview
        preview_intents = []
        if should_preview:
            active_intents = await self.original_engine.get_active_intents()
            preview_intents = await self._select_intents_for_preview(active_intents, context)
        
        # Check if we should send a message now
        immediate_send = await self._check_immediate_send_opportunity(context, messages)
        
        # Generate communication influence on response
        comm_influence = {
            "should_preview_communications": should_preview,
            "preview_intents": preview_intents,
            "immediate_send_recommended": immediate_send["should_send"],
            "immediate_send_reason": immediate_send.get("reason"),
            "communication_tone_adjustment": await self._suggest_tone_adjustment(context, messages),
            "timing_considerations": await self._get_timing_considerations(context)
        }
        
        # Send synthesis results
        await self.send_context_update(
            update_type="proactive_comm_synthesis",
            data={
                "synthesis_results": comm_influence,
                "preview_count": len(preview_intents),
                "immediate_opportunity": immediate_send["should_send"]
            },
            priority=ContextPriority.NORMAL
        )
        
        return {
            "communication_influence": comm_influence,
            "synthesis_complete": True,
            "context_integrated": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_communication_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze context for communication opportunities"""
        opportunities = []
        
        # Check emotional state for expression opportunities
        if context.emotional_state:
            emotional_opportunity = self._evaluate_emotional_opportunity(context.emotional_state)
            if emotional_opportunity:
                opportunities.append(emotional_opportunity)
        
        # Check for unfinished conversations
        if context.session_context.get("unfinished_conversation"):
            opportunities.append({
                "type": "continuation",
                "urgency": 0.7,
                "reason": "unfinished_conversation",
                "context": context.session_context.get("last_topic")
            })
        
        # Check temporal factors
        temporal_opportunity = self._evaluate_temporal_opportunity(context)
        if temporal_opportunity:
            opportunities.append(temporal_opportunity)
        
        # Check goal context for achievement sharing
        if context.goal_context and context.goal_context.get("recent_completion"):
            opportunities.append({
                "type": "achievement_sharing",
                "urgency": 0.6,
                "reason": "goal_completed",
                "context": context.goal_context
            })
        
        # Check memory context for nostalgia
        if context.memory_context and context.memory_context.get("significant_memory_retrieved"):
            opportunities.append({
                "type": "memory_recollection",
                "urgency": 0.4,
                "reason": "significant_memory",
                "context": context.memory_context
            })
        
        return opportunities
    
    async def _assess_communication_appropriateness(self, context: SharedContext) -> Dict[str, Any]:
        """Assess if proactive communication is appropriate given context"""
        appropriateness_score = 1.0
        factors = []
        
        # Check relationship context
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            if trust < 0.3:
                appropriateness_score *= 0.5
                factors.append("low_trust")
            elif trust > 0.7:
                appropriateness_score *= 1.2
                factors.append("high_trust")
        
        # Check conversation momentum
        if context.session_context.get("high_engagement"):
            appropriateness_score *= 0.7  # Reduce during high engagement
            factors.append("high_current_engagement")
        
        # Check emotional readiness
        if context.emotional_state:
            arousal = context.emotional_state.get("arousal", 0.5)
            if arousal > 0.8:
                appropriateness_score *= 0.6  # Reduce during high arousal
                factors.append("high_emotional_arousal")
        
        # Check mode context
        if context.mode_context and context.mode_context.get("mode") == "task_focused":
            appropriateness_score *= 0.4  # Significantly reduce during task focus
            factors.append("task_focused_mode")
        
        # Check recent message frequency
        recent_count = context.session_context.get("recent_proactive_count", 0)
        if recent_count > 2:
            appropriateness_score *= 0.3
            factors.append("high_recent_frequency")
        
        return {
            "score": min(1.0, appropriateness_score),
            "factors": factors
        }
    
    async def _analyze_intent_context_fit(self, active_intents: List[Dict[str, Any]], 
                                         context: SharedContext) -> Dict[str, Any]:
        """Analyze how well active intents fit current context"""
        adjustments = {
            "suppress": [],
            "boost": [],
            "modify": []
        }
        
        for intent in active_intents:
            intent_type = intent.get("intent_type")
            intent_id = intent.get("intent_id")
            
            # Check mood fit
            if context.mode_context and context.mode_context.get("mode") == "analytical":
                if intent_type in ["mood_expression", "creative_expression"]:
                    adjustments["suppress"].append({
                        "intent_id": intent_id,
                        "reason": "analytical_mode_mismatch"
                    })
                elif intent_type == "insight_sharing":
                    adjustments["boost"].append({
                        "intent_id": intent_id,
                        "reason": "analytical_mode_match"
                    })
            
            # Check emotional fit
            if context.emotional_state and intent_type == "mood_expression":
                current_valence = context.emotional_state.get("valence", 0)
                intent_valence = intent.get("content_guidelines", {}).get("emotional_valence", 0)
                
                if abs(current_valence - intent_valence) > 0.5:
                    adjustments["modify"].append({
                        "intent_id": intent_id,
                        "reason": "emotional_mismatch",
                        "suggestion": "update_emotional_tone"
                    })
            
            # Check relationship fit
            if context.relationship_context:
                relationship_level = context.relationship_context.get("intimacy", 0.5)
                
                if intent_type == "milestone_recognition" and relationship_level < 0.4:
                    adjustments["suppress"].append({
                        "intent_id": intent_id,
                        "reason": "insufficient_relationship_depth"
                    })
        
        return adjustments
    
    async def _process_emotional_trigger(self, dominant_emotion: Optional[tuple], 
                                       emotional_state: Dict[str, float]):
        """Process emotional triggers for communication"""
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Map emotions to communication triggers
        emotion_triggers = {
            "Joy": ("mood_expression", 0.7),
            "Excitement": ("creative_expression", 0.6),
            "Curiosity": ("insight_sharing", 0.5),
            "Loneliness": ("connection", 0.8),
            "Pride": ("achievement_sharing", 0.6),
            "Nostalgia": ("memory_recollection", 0.5)
        }
        
        if emotion_name in emotion_triggers and strength > 0.6:
            intent_type, base_urgency = emotion_triggers[emotion_name]
            urgency = base_urgency * strength
            
            # Create context for intent
            intent_context = {
                "emotional_trigger": True,
                "emotion": emotion_name,
                "strength": strength,
                "full_state": emotional_state
            }
            
            # Add to engine's intent generation queue
            await self._queue_emotional_intent(intent_type, urgency, intent_context)
    
    async def _create_milestone_intent(self, user_id: str, milestone_type: str, 
                                     relationship_data: Dict[str, Any]):
        """Create intent for relationship milestone"""
        # Map milestone types to intent configurations
        milestone_configs = {
            "trust_increase": {
                "intent_type": "relationship_maintenance",
                "urgency": 0.6,
                "tone": "warm",
                "content_focus": "appreciation"
            },
            "intimacy_milestone": {
                "intent_type": "milestone_recognition",
                "urgency": 0.7,
                "tone": "celebratory",
                "content_focus": "connection"
            },
            "duration_milestone": {
                "intent_type": "milestone_recognition",
                "urgency": 0.5,
                "tone": "reflective",
                "content_focus": "journey"
            }
        }
        
        config = milestone_configs.get(milestone_type, {
            "intent_type": "relationship_maintenance",
            "urgency": 0.5,
            "tone": "friendly",
            "content_focus": "general"
        })
        
        # Create intent through original engine
        intent_id = await self.original_engine.add_proactive_intent(
            intent_type=config["intent_type"],
            user_id=user_id,
            content_guidelines={
                "tone": config["tone"],
                "content_focus": config["content_focus"],
                "milestone_type": milestone_type,
                "relationship_context": relationship_data
            },
            urgency=config["urgency"]
        )
        
        # Track as context-informed
        if intent_id:
            self.context_informed_intents[intent_id] = {
                "trigger": "relationship_milestone",
                "milestone_type": milestone_type,
                "created_at": datetime.now()
            }
    
    async def _create_checkin_intent(self, user_id: str, days_inactive: int):
        """Create check-in intent for inactive user"""
        # Calculate urgency based on inactivity duration
        urgency = min(0.9, 0.4 + (days_inactive * 0.1))
        
        # Adjust tone based on relationship
        tone = "caring"
        if days_inactive > 7:
            tone = "concerned"
        elif days_inactive > 14:
            tone = "gentle"
        
        # Create intent
        intent_id = await self.original_engine.add_proactive_intent(
            intent_type="check_in",
            user_id=user_id,
            content_guidelines={
                "tone": tone,
                "days_inactive": days_inactive,
                "acknowledge_absence": True,
                "no_pressure": True
            },
            urgency=urgency
        )
        
        # Track as context-informed
        if intent_id:
            self.context_informed_intents[intent_id] = {
                "trigger": "user_inactivity",
                "days_inactive": days_inactive,
                "created_at": datetime.now()
            }
    
    async def _suppress_non_urgent_intents(self, reason: str):
        """Suppress non-urgent intents due to context"""
        active_intents = await self.original_engine.get_active_intents()
        
        for intent in active_intents:
            if intent.get("urgency", 0.5) < 0.7:
                intent_id = intent.get("intent_id")
                
                # Track suppression
                self.suppressed_intents[intent_id] = {
                    "reason": reason,
                    "suppressed_at": datetime.now(),
                    "original_urgency": intent.get("urgency", 0.5)
                }
                
                # Update configuration to delay sending
                await self.original_engine.update_configuration({
                    "processed_intent_id": intent_id
                })
    
    async def _reevaluate_intents_with_context(self, active_intents: List[Dict[str, Any]], 
                                              context: SharedContext,
                                              messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Re-evaluate active intents based on current context"""
        changes_made = False
        adjustments = []
        
        for intent in active_intents:
            intent_id = intent.get("intent_id")
            original_urgency = intent.get("urgency", 0.5)
            adjusted_urgency = original_urgency
            
            # Check emotional alignment
            if context.emotional_state:
                emotional_alignment = self._calculate_emotional_alignment(intent, context.emotional_state)
                if emotional_alignment < 0.3:
                    adjusted_urgency *= 0.5
                    adjustments.append({
                        "intent_id": intent_id,
                        "reason": "poor_emotional_alignment",
                        "adjustment": -0.5
                    })
                elif emotional_alignment > 0.7:
                    adjusted_urgency *= 1.2
                    adjustments.append({
                        "intent_id": intent_id,
                        "reason": "strong_emotional_alignment",
                        "adjustment": 0.2
                    })
            
            # Check goal alignment
            if context.goal_context and "goal" in messages:
                goal_messages = messages["goal"]
                if any(msg['type'] == 'goal_urgency_detected' for msg in goal_messages):
                    if intent.get("intent_type") not in ["goal_related", "achievement_sharing"]:
                        adjusted_urgency *= 0.7
                        adjustments.append({
                            "intent_id": intent_id,
                            "reason": "urgent_goals_take_priority",
                            "adjustment": -0.3
                        })
            
            # Apply adjustments if changed
            if adjusted_urgency != original_urgency:
                changes_made = True
                # Would update intent urgency through engine if method available
        
        return {
            "changes_made": changes_made,
            "adjustments": adjustments,
            "total_adjusted": len(adjustments)
        }
    
    def _calculate_emotional_alignment(self, intent: Dict[str, Any], 
                                     emotional_state: Dict[str, float]) -> float:
        """Calculate how well an intent aligns with current emotional state"""
        intent_type = intent.get("intent_type", "")
        
        # Get dominant emotion
        if not emotional_state:
            return 0.5
        
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else None
        
        # Define alignments
        alignments = {
            "mood_expression": {
                "Joy": 0.9, "Excitement": 0.8, "Sadness": 0.7, "Anxiety": 0.6
            },
            "insight_sharing": {
                "Curiosity": 0.9, "Interest": 0.8, "Contemplation": 0.7
            },
            "memory_recollection": {
                "Nostalgia": 0.9, "Melancholy": 0.7, "Joy": 0.6
            },
            "check_in": {
                "Concern": 0.8, "Care": 0.8, "Loneliness": 0.7
            }
        }
        
        intent_alignments = alignments.get(intent_type, {})
        return intent_alignments.get(dominant_emotion, 0.5)
    
    async def _analyze_communication_patterns(self, active_intents: List[Dict[str, Any]], 
                                            recent_sent: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in communication"""
        # Analyze intent type distribution
        intent_types = {}
        for intent in active_intents + recent_sent:
            intent_type = intent.get("intent_type", "unknown")
            intent_types[intent_type] = intent_types.get(intent_type, 0) + 1
        
        # Analyze timing patterns
        if recent_sent:
            # Calculate average time between messages
            timestamps = []
            for intent in recent_sent:
                if "created_at" in intent:
                    timestamps.append(datetime.fromisoformat(intent["created_at"]))
            
            time_deltas = []
            if len(timestamps) > 1:
                timestamps.sort()
                for i in range(1, len(timestamps)):
                    delta = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                    time_deltas.append(delta)
            
            avg_hours_between = sum(time_deltas) / len(time_deltas) if time_deltas else 0
        else:
            avg_hours_between = 0
        
        # Analyze context triggers
        context_triggers = {}
        for intent_id in self.context_informed_intents:
            trigger = self.context_informed_intents[intent_id].get("trigger", "unknown")
            context_triggers[trigger] = context_triggers.get(trigger, 0) + 1
        
        return {
            "intent_type_distribution": intent_types,
            "average_hours_between_messages": avg_hours_between,
            "context_trigger_distribution": context_triggers,
            "total_active": len(active_intents),
            "total_recent_sent": len(recent_sent),
            "context_informed_percentage": len(self.context_informed_intents) / max(1, len(active_intents + recent_sent))
        }
    
    async def _identify_communication_gaps(self, context: SharedContext, 
                                         active_intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gaps in communication coverage"""
        gaps = []
        
        # Check if emotional needs are addressed
        if context.emotional_state:
            emotional_coverage = self._check_emotional_coverage(active_intents, context.emotional_state)
            if emotional_coverage < 0.5:
                gaps.append({
                    "type": "emotional_expression",
                    "severity": 1.0 - emotional_coverage,
                    "suggestion": "Consider expressing current emotional state"
                })
        
        # Check if relationship needs are addressed
        if context.relationship_context:
            relationship_coverage = self._check_relationship_coverage(active_intents, context.relationship_context)
            if relationship_coverage < 0.5:
                gaps.append({
                    "type": "relationship_maintenance",
                    "severity": 1.0 - relationship_coverage,
                    "suggestion": "Consider relationship maintenance communication"
                })
        
        # Check if goals are acknowledged
        if context.goal_context and context.goal_context.get("active_goals"):
            goal_coverage = self._check_goal_coverage(active_intents, context.goal_context)
            if goal_coverage < 0.3:
                gaps.append({
                    "type": "goal_acknowledgment",
                    "severity": 1.0 - goal_coverage,
                    "suggestion": "Consider acknowledging active goals or progress"
                })
        
        return gaps
    
    def _check_emotional_coverage(self, active_intents: List[Dict[str, Any]], 
                                emotional_state: Dict[str, float]) -> float:
        """Check how well active intents cover emotional expression needs"""
        emotional_intent_types = ["mood_expression", "emotional_sharing", "feeling_expression"]
        
        emotional_intents = [i for i in active_intents if i.get("intent_type") in emotional_intent_types]
        
        if not emotional_intents:
            return 0.0
        
        # Calculate coverage based on emotional intensity
        max_emotion_strength = max(emotional_state.values()) if emotional_state else 0
        coverage = min(1.0, len(emotional_intents) / max(1, max_emotion_strength * 2))
        
        return coverage
    
    def _check_relationship_coverage(self, active_intents: List[Dict[str, Any]], 
                                   relationship_context: Dict[str, Any]) -> float:
        """Check how well active intents cover relationship needs"""
        relationship_intent_types = ["relationship_maintenance", "milestone_recognition", "check_in"]
        
        relationship_intents = [i for i in active_intents if i.get("intent_type") in relationship_intent_types]
        
        # Consider relationship strength
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        relationship_strength = (trust + intimacy) / 2
        
        # Higher relationship strength needs more maintenance
        needed_intents = max(1, int(relationship_strength * 3))
        coverage = min(1.0, len(relationship_intents) / needed_intents)
        
        return coverage
    
    def _check_goal_coverage(self, active_intents: List[Dict[str, Any]], 
                           goal_context: Dict[str, Any]) -> float:
        """Check how well active intents cover goal-related communication"""
        goal_intent_types = ["goal_sharing", "achievement_sharing", "progress_update"]
        
        goal_intents = [i for i in active_intents if i.get("intent_type") in goal_intent_types]
        active_goals = goal_context.get("active_goals", [])
        
        if not active_goals:
            return 1.0  # No goals to cover
        
        # Calculate coverage
        high_priority_goals = [g for g in active_goals if g.get("priority", 0) > 0.7]
        needed_intents = max(1, len(high_priority_goals))
        coverage = min(1.0, len(goal_intents) / needed_intents)
        
        return coverage
    
    # Forward all other methods to the original engine
    async def start(self):
        """Start the background task"""
        return await self.original_engine.start()
    
    async def stop(self):
        """Stop the background process"""
        return await self.original_engine.stop()
    
    async def create_intent_from_action(self, action: Dict[str, Any], user_id: str) -> Optional[str]:
        """Create intent from action with context awareness"""
        # Add context tracking
        intent_id = await self.original_engine.create_intent_from_action(action, user_id)
        
        if intent_id:
            self.context_informed_intents[intent_id] = {
                "trigger": "action_driven",
                "action": action.get("name", "unknown"),
                "created_at": datetime.now()
            }
        
        return intent_id
    
    def __getattr__(self, name):
        """Delegate any missing methods to the original engine"""
        return getattr(self.original_engine, name)
