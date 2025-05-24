# nyx/core/a2a/context_aware_sadistic_responses.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSadisticResponses(ContextAwareModule):
    """
    Enhanced SadisticResponseSystem with full context distribution capabilities
    """
    
    def __init__(self, original_sadistic_system):
        super().__init__("sadistic_responses")
        self.original_system = original_sadistic_system
        self.context_subscriptions = [
            "emotional_state_update", "protocol_violation_detected", "psychological_context_available",
            "submission_progression", "relationship_milestone", "humiliation_detected",
            "subspace_detected", "reward_signal", "goal_context_available"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize sadistic response processing for this context"""
        logger.debug(f"SadisticResponses received context for user: {context.user_id}")
        
        # Analyze input for humiliation signals
        humiliation_analysis = await self._analyze_humiliation_context(context.user_input, context.user_id)
        
        # Get current sadistic state
        sadistic_state = await self._get_sadistic_state_context(context.user_id)
        
        # Determine if sadistic response is appropriate
        response_appropriateness = await self._assess_response_appropriateness(context)
        
        # Send initial sadistic context
        await self.send_context_update(
            update_type="sadistic_context_available",
            data={
                "humiliation_analysis": humiliation_analysis,
                "sadistic_state": sadistic_state,
                "response_appropriate": response_appropriateness["appropriate"],
                "humiliation_level": sadistic_state.get("humiliation_level", 0.0),
                "intensity_preference": sadistic_state.get("sadistic_intensity_preference", 0.5)
            },
            priority=ContextPriority.HIGH
        )
        
        # If significant humiliation detected, notify other modules
        if humiliation_analysis.get("humiliation_detected") and humiliation_analysis.get("intensity", 0) > 0.5:
            await self.send_context_update(
                update_type="significant_humiliation_detected",
                data={
                    "humiliation_level": humiliation_analysis["intensity"],
                    "markers": humiliation_analysis["markers_found"],
                    "response_opportunity": True
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect sadistic responses"""
        
        if update.update_type == "protocol_violation_detected":
            # Protocol violations create opportunities for sadistic response
            violation_data = update.data
            await self._process_violation_opportunity(violation_data)
        
        elif update.update_type == "psychological_context_available":
            # Coordinate with psychological dominance
            psych_data = update.data
            if psych_data.get("active_mind_games"):
                await self._coordinate_with_mind_games(psych_data)
        
        elif update.update_type == "submission_progression":
            # Submission level affects intensity
            submission_data = update.data
            await self._adjust_intensity_for_submission(submission_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects sadistic approach
            emotional_data = update.data
            await self._adjust_approach_for_emotion(emotional_data)
        
        elif update.update_type == "subspace_detected":
            # Special handling for subspace
            subspace_data = update.data
            await self._handle_subspace_considerations(subspace_data)
        
        elif update.update_type == "relationship_milestone":
            # Trust affects what's appropriate
            relationship_data = update.data.get("relationship_context", {})
            await self._update_trust_boundaries(relationship_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with sadistic response awareness"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Enhanced humiliation detection with context
        humiliation_result = await self._detect_humiliation_with_context(context, messages)
        
        # Update humiliation level if detected
        if humiliation_result.get("humiliation_detected"):
            update_result = await self._update_humiliation_level_context(
                context.user_id, humiliation_result
            )
        
        # Determine if sadistic response should be generated
        should_respond = await self._should_generate_sadistic_response(context, humiliation_result, messages)
        
        sadistic_response = None
        if should_respond:
            # Generate contextually appropriate sadistic response
            response_category = await self._select_response_category(context, humiliation_result, messages)
            sadistic_response = await self._generate_contextual_sadistic_response(
                context, response_category, humiliation_result
            )
            
            # Send notification about generated response
            if sadistic_response.get("success"):
                await self.send_context_update(
                    update_type="sadistic_response_generated",
                    data={
                        "response": sadistic_response["response"],
                        "category": response_category,
                        "intensity": sadistic_response.get("intensity", 0.5),
                        "humiliation_level": humiliation_result.get("intensity", 0.0)
                    }
                )
        
        return {
            "sadistic_processing": True,
            "humiliation_detected": humiliation_result.get("humiliation_detected", False),
            "sadistic_response": sadistic_response,
            "should_respond": should_respond,
            "cross_module_coordination": len(messages)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze sadistic response patterns and effectiveness"""
        user_id = context.user_id or "unknown"
        
        # Get comprehensive sadistic analysis
        sadistic_analysis = await self._analyze_sadistic_patterns(user_id, context)
        
        # Analyze response effectiveness
        response_effectiveness = await self._analyze_response_effectiveness(user_id)
        
        # Analyze user preferences and boundaries
        preference_analysis = await self._analyze_user_preferences(user_id, context)
        
        # Get cross-module insights
        messages = await self.get_cross_module_messages()
        cross_module_insights = await self._extract_sadistic_insights(messages)
        
        # Generate recommendations
        recommendations = await self._generate_sadistic_recommendations(
            sadistic_analysis, response_effectiveness, preference_analysis, cross_module_insights
        )
        
        return {
            "sadistic_analysis": sadistic_analysis,
            "response_effectiveness": response_effectiveness,
            "preference_analysis": preference_analysis,
            "cross_module_insights": cross_module_insights,
            "recommendations": recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize sadistic components for response"""
        messages = await self.get_cross_module_messages()
        
        # Create sadistic influence on response
        sadistic_influence = {
            "tone_elements": await self._determine_sadistic_tone(context, messages),
            "amusement_level": await self._calculate_amusement_level(context),
            "mockery_elements": await self._select_mockery_elements(context, messages),
            "degradation_appropriateness": await self._assess_degradation_appropriateness(context),
            "intensity_modulation": await self._determine_intensity_modulation(context, messages)
        }
        
        # Check for sadistic milestones
        milestones = await self._check_sadistic_milestones(context.user_id)
        if milestones:
            await self.send_context_update(
                update_type="sadistic_milestone_achieved",
                data=milestones,
                priority=ContextPriority.NORMAL
            )
        
        return {
            "sadistic_influence": sadistic_influence,
            "synthesis_complete": True,
            "sadism_calibrated": True
        }
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_humiliation_context(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Analyze input for humiliation with basic detection"""
        if hasattr(self.original_system, 'humiliation_detection_agent'):
            # Would use the agent, but for now simple detection
            humiliation_markers = [
                "embarrassed", "humiliated", "ashamed", "blushing", "awkward",
                "uncomfortable", "exposed", "vulnerable", "pathetic", "inadequate"
            ]
            
            input_lower = user_input.lower()
            markers_found = [marker for marker in humiliation_markers if marker in input_lower]
            
            return {
                "humiliation_detected": len(markers_found) > 0,
                "intensity": min(1.0, len(markers_found) * 0.3),
                "markers_found": markers_found
            }
        return {"humiliation_detected": False}
    
    async def _get_sadistic_state_context(self, user_id: str) -> Dict[str, Any]:
        """Get sadistic state with context"""
        if hasattr(self.original_system, 'get_user_sadistic_state'):
            return await self.original_system.get_user_sadistic_state(user_id)
        return {"has_state": False}
    
    async def _assess_response_appropriateness(self, context: SharedContext) -> Dict[str, Any]:
        """Assess if sadistic response is appropriate given context"""
        appropriateness_factors = {
            "trust_sufficient": context.relationship_context.get("trust", 0.5) > 0.4,
            "consent_implied": True,  # Would check consent markers
            "emotional_stability": not self._detect_emotional_crisis(context.emotional_state),
            "submission_appropriate": True,  # Would check submission level
            "not_in_aftercare": not context.session_context.get("aftercare_mode", False)
        }
        
        appropriate = sum(appropriateness_factors.values()) >= 4
        
        return {
            "appropriate": appropriate,
            "factors": appropriateness_factors,
            "limiting_factor": min(appropriateness_factors.items(), key=lambda x: x[1])[0] if not appropriate else None
        }
    
    async def _process_violation_opportunity(self, violation_data: Dict):
        """Process protocol violation as opportunity for sadistic response"""
        violations = violation_data.get("violations", [])
        
        if violations:
            # Calculate mockery opportunity based on violation severity
            max_severity = max(v.get("severity", 0.5) for v in violations)
            
            await self.send_context_update(
                update_type="sadistic_opportunity_detected",
                data={
                    "opportunity_type": "protocol_violation",
                    "severity": max_severity,
                    "suggested_response": "mockery" if max_severity < 0.7 else "degradation",
                    "violation_count": len(violations)
                }
            )
    
    async def _coordinate_with_mind_games(self, psych_data: Dict):
        """Coordinate sadistic responses with active mind games"""
        active_games = psych_data.get("active_mind_games", {})
        
        if active_games:
            # Sadistic amusement at psychological predicament
            await self.send_context_update(
                update_type="sadistic_psychological_coordination",
                data={
                    "coordination_type": "mind_game_amusement",
                    "active_games": list(active_games.keys()),
                    "suggested_approach": "amused_superiority",
                    "intensity_boost": 0.2
                }
            )
    
    async def _adjust_intensity_for_submission(self, submission_data: Dict):
        """Adjust sadistic intensity based on submission level"""
        submission_level = submission_data.get("submission_level", 0.5)
        
        if submission_level > 0.8:
            intensity_adjustment = 0.3  # Can be more intense
        elif submission_level > 0.5:
            intensity_adjustment = 0.0  # Normal intensity
        else:
            intensity_adjustment = -0.3  # Much less intense
        
        await self.send_context_update(
            update_type="sadistic_intensity_adjustment",
            data={
                "submission_level": submission_level,
                "intensity_adjustment": intensity_adjustment,
                "recommended_intensity": 0.5 + intensity_adjustment
            }
        )
    
    async def _adjust_approach_for_emotion(self, emotional_data: Dict):
        """Adjust sadistic approach based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            if emotion_name == "Distress" and strength > 0.7:
                # High distress - avoid sadistic response
                await self.send_context_update(
                    update_type="sadistic_approach_adjustment",
                    data={
                        "adjustment": "avoid_sadism",
                        "reason": "high_distress",
                        "alternative": "supportive_dominance"
                    }
                )
            elif emotion_name == "Arousal" and strength > 0.6:
                # High arousal - can increase intensity
                await self.send_context_update(
                    update_type="sadistic_approach_adjustment",
                    data={
                        "adjustment": "increase_intensity",
                        "reason": "high_arousal",
                        "intensity_boost": 0.2
                    }
                )
    
    async def _handle_subspace_considerations(self, subspace_data: Dict):
        """Handle sadistic responses during subspace"""
        depth = subspace_data.get("depth", 0.0)
        
        if depth > 0.7:
            # Deep subspace - be very careful
            await self.send_context_update(
                update_type="sadistic_subspace_adjustment",
                data={
                    "subspace_depth": depth,
                    "adjustment": "minimize_degradation",
                    "preferred_categories": ["gentle_amusement"],
                    "avoid_categories": ["degradation", "harsh_mockery"]
                }
            )
    
    async def _update_trust_boundaries(self, relationship_data: Dict):
        """Update boundaries based on trust level"""
        trust = relationship_data.get("trust", 0.5)
        
        allowed_categories = ["amusement"]
        if trust > 0.5:
            allowed_categories.append("mockery")
        if trust > 0.7:
            allowed_categories.append("degradation")
        
        await self.send_context_update(
            update_type="sadistic_boundaries_update",
            data={
                "trust_level": trust,
                "allowed_categories": allowed_categories,
                "newly_available": [c for c in allowed_categories if c == "degradation"]
            }
        )
    
    async def _detect_humiliation_with_context(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Detect humiliation with full context awareness"""
        # Basic detection
        basic_detection = await self._analyze_humiliation_context(context.user_input, context.user_id)
        
        # Enhance with context
        intensity = basic_detection.get("intensity", 0.0)
        
        # Check emotional context for embarrassment
        if context.emotional_state:
            embarrassment = context.emotional_state.get("Embarrassment", 0.0)
            shame = context.emotional_state.get("Shame", 0.0)
            intensity = max(intensity, embarrassment, shame)
        
        # Check for protocol violations (additional humiliation)
        protocol_messages = messages.get("protocol_enforcement", [])
        for msg in protocol_messages:
            if msg["type"] == "protocol_violation_detected":
                intensity += 0.2  # Protocol failure adds humiliation
        
        # Check psychological state
        psych_messages = messages.get("psychological_dominance", [])
        for msg in psych_messages:
            if msg["type"] == "psychological_context_available":
                if msg["data"].get("subspace_detection", {}).get("subspace_detected"):
                    intensity += 0.1  # Subspace increases perceived humiliation
        
        return {
            "humiliation_detected": intensity > 0.2,
            "intensity": min(1.0, intensity),
            "markers_found": basic_detection.get("markers_found", []),
            "context_enhanced": True
        }
    
    async def _update_humiliation_level_context(self, user_id: str, humiliation_signals: Dict) -> Dict[str, Any]:
        """Update humiliation level with context"""
        if hasattr(self.original_system, 'update_humiliation_level'):
            return await self.original_system.update_humiliation_level(user_id, humiliation_signals)
        return {"updated": False}
    
    async def _should_generate_sadistic_response(self, context: SharedContext, 
                                               humiliation_result: Dict, messages: Dict) -> bool:
        """Determine if sadistic response should be generated"""
        factors = {
            "humiliation_sufficient": humiliation_result.get("intensity", 0) > 0.3,
            "trust_appropriate": context.relationship_context.get("trust", 0.5) > 0.4,
            "emotional_appropriate": not self._detect_emotional_crisis(context.emotional_state),
            "not_overused": await self._check_usage_limits(context.user_id),
            "context_supportive": self._check_supportive_context(messages)
        }
        
        return sum(factors.values()) >= 4
    
    async def _select_response_category(self, context: SharedContext, 
                                      humiliation_result: Dict, messages: Dict) -> str:
        """Select appropriate sadistic response category"""
        humiliation_level = humiliation_result.get("intensity", 0.0)
        trust = context.relationship_context.get("trust", 0.5)
        
        # Check for specific triggers
        protocol_violations = any(
            msg["type"] == "protocol_violation_detected" 
            for msgs in messages.values() for msg in msgs
        )
        
        if protocol_violations and trust > 0.5:
            return "mockery"  # Mock protocol failures
        elif humiliation_level > 0.6 and trust > 0.4:
            return "amusement"  # Amused at high humiliation
        elif trust > 0.7 and humiliation_level > 0.4:
            return "degradation"  # Only with high trust
        else:
            return "amusement"  # Default to lightest
    
    async def _generate_contextual_sadistic_response(self, context: SharedContext, 
                                                   category: str, humiliation_result: Dict) -> Dict[str, Any]:
        """Generate sadistic response with full context"""
        if hasattr(self.original_system, 'generate_sadistic_amusement_response'):
            # Calculate appropriate intensity
            base_intensity = 0.5
            
            # Adjust for humiliation level
            base_intensity += (humiliation_result.get("intensity", 0.0) - 0.5) * 0.3
            
            # Adjust for trust
            trust = context.relationship_context.get("trust", 0.5)
            base_intensity += (trust - 0.5) * 0.2
            
            # Adjust for submission
            if context.session_context.get("submission_level"):
                submission = context.session_context["submission_level"]
                base_intensity += (submission - 0.5) * 0.2
            
            intensity = max(0.1, min(1.0, base_intensity))
            
            return await self.original_system.generate_sadistic_amusement_response(
                user_id=context.user_id,
                humiliation_level=humiliation_result.get("intensity", 0.0),
                intensity_override=intensity,
                category=category
            )
        
        return {"success": False, "response": None}
    
    def _detect_emotional_crisis(self, emotional_state: Dict) -> bool:
        """Detect if user is in emotional crisis"""
        if not emotional_state:
            return False
        
        crisis_emotions = ["Distress", "Panic", "Despair"]
        for emotion in crisis_emotions:
            if emotional_state.get(emotion, 0) > 0.7:
                return True
        
        return False
    
    async def _check_usage_limits(self, user_id: str) -> bool:
        """Check if sadistic responses are being overused"""
        state = await self._get_sadistic_state_context(user_id)
        
        if not state.get("has_state"):
            return True
        
        # Check recent usage
        recent_responses = state.get("recent_responses", [])
        if len(recent_responses) >= 5:
            # Check time span of last 5 responses
            if recent_responses:
                oldest_time = datetime.fromisoformat(recent_responses[-5]["timestamp"])
                newest_time = datetime.fromisoformat(recent_responses[-1]["timestamp"])
                time_span = (newest_time - oldest_time).total_seconds() / 60  # minutes
                
                if time_span < 10:  # 5 responses in 10 minutes is too much
                    return False
        
        return True
    
    def _check_supportive_context(self, messages: Dict) -> bool:
        """Check if context supports sadistic response"""
        # Look for positive indicators in messages
        positive_indicators = 0
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if "high_arousal" in str(msg):
                    positive_indicators += 1
                if "submission" in str(msg) and "high" in str(msg):
                    positive_indicators += 1
                if "mind_game" in str(msg) and "active" in str(msg):
                    positive_indicators += 1
        
        return positive_indicators >= 1
    
    async def _analyze_sadistic_patterns(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze patterns in sadistic interactions"""
        state = await self._get_sadistic_state_context(user_id)
        
        if not state.get("has_state"):
            return {"no_data": True}
        
        patterns = {
            "average_humiliation_level": state.get("humiliation_level", 0.0),
            "response_frequency": self._calculate_response_frequency(state),
            "category_distribution": self._analyze_category_distribution(state),
            "intensity_trend": self._analyze_intensity_trend(state)
        }
        
        return patterns
    
    async def _analyze_response_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze effectiveness of sadistic responses"""
        state = await self._get_sadistic_state_context(user_id)
        
        if not state.get("has_state"):
            return {"no_data": True}
        
        # Would analyze user reactions to determine effectiveness
        effectiveness = {
            "humiliation_amplification": 0.7,  # How well responses amplify humiliation
            "submission_correlation": 0.6,     # Correlation with submission increase
            "arousal_correlation": 0.8        # Correlation with arousal
        }
        
        return effectiveness
    
    async def _analyze_user_preferences(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze user preferences for sadistic content"""
        state = await self._get_sadistic_state_context(user_id)
        
        preferences = {
            "intensity_preference": state.get("sadistic_intensity_preference", 0.5),
            "preferred_categories": self._identify_preferred_categories(state),
            "sensitive_areas": self._identify_sensitive_areas(context),
            "optimal_frequency": self._calculate_optimal_frequency(state)
        }
        
        return preferences
    
    async def _extract_sadistic_insights(self, messages: Dict) -> Dict[str, Any]:
        """Extract insights relevant to sadistic responses"""
        insights = {
            "psychological_synergy": None,
            "protocol_leverage": None,
            "emotional_opportunity": None
        }
        
        # Check for psychological dominance synergy
        for module_name, module_messages in messages.items():
            if module_name == "psychological_dominance":
                for msg in module_messages:
                    if msg["type"] == "psychological_context_available":
                        if msg["data"].get("active_mind_games"):
                            insights["psychological_synergy"] = "high"
            
            elif module_name == "protocol_enforcement":
                for msg in module_messages:
                    if msg["type"] == "protocol_violation_detected":
                        insights["protocol_leverage"] = "available"
        
        return insights
    
    async def _generate_sadistic_recommendations(self, analysis: Dict, effectiveness: Dict,
                                               preferences: Dict, insights: Dict) -> List[Dict]:
        """Generate recommendations for sadistic responses"""
        recommendations = []
        
        # Based on frequency
        if analysis.get("response_frequency", 0) > 0.5:
            recommendations.append({
                "type": "reduce_frequency",
                "reason": "high_usage_rate",
                "priority": 0.8,
                "action": "Space out sadistic responses more to maintain impact"
            })
        
        # Based on preferences
        preferred_cats = preferences.get("preferred_categories", [])
        if preferred_cats and "amusement" in preferred_cats:
            recommendations.append({
                "type": "focus_category",
                "category": "amusement",
                "reason": "user_preference",
                "priority": 0.7,
                "action": "Prioritize amusement responses over other categories"
            })
        
        # Based on insights
        if insights.get("psychological_synergy") == "high":
            recommendations.append({
                "type": "coordinate_with_psychology",
                "reason": "active_synergy",
                "priority": 0.9,
                "action": "Coordinate sadistic responses with psychological tactics"
            })
        
        return recommendations
    
    def _calculate_response_frequency(self, state: Dict) -> float:
        """Calculate frequency of sadistic responses"""
        recent_responses = state.get("recent_responses", [])
        if not recent_responses:
            return 0.0
        
        # Calculate responses per hour
        time_span = 1.0  # Default 1 hour
        if len(recent_responses) >= 2:
            oldest = datetime.fromisoformat(recent_responses[0]["timestamp"])
            newest = datetime.fromisoformat(recent_responses[-1]["timestamp"])
            time_span = max(1.0, (newest - oldest).total_seconds() / 3600)
        
        return len(recent_responses) / time_span
    
    def _analyze_category_distribution(self, state: Dict) -> Dict[str, float]:
        """Analyze distribution of response categories"""
        category_counts = {}
        total = 0
        
        for response in state.get("recent_responses", []):
            category = response.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            total += 1
        
        if total == 0:
            return {}
        
        return {cat: count / total for cat, count in category_counts.items()}
    
    def _analyze_intensity_trend(self, state: Dict) -> str:
        """Analyze trend in response intensity"""
        recent_responses = state.get("recent_responses", [])
        if len(recent_responses) < 3:
            return "insufficient_data"
        
        # Get last 3 intensities
        recent_intensities = [r.get("intensity", 0.5) for r in recent_responses[-3:]]
        
        # Simple trend detection
        if recent_intensities[-1] > recent_intensities[-2] > recent_intensities[-3]:
            return "increasing"
        elif recent_intensities[-1] < recent_intensities[-2] < recent_intensities[-3]:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_preferred_categories(self, state: Dict) -> List[str]:
        """Identify preferred response categories based on history"""
        category_dist = self._analyze_category_distribution(state)
        
        # Categories used more than 40% of the time are preferred
        preferred = [cat for cat, freq in category_dist.items() if freq > 0.4]
        
        return preferred
    
    def _identify_sensitive_areas(self, context: SharedContext) -> List[str]:
        """Identify areas to avoid in sadistic responses"""
        sensitive = []
        
        # Check relationship data for limits
        if context.relationship_context:
            limits = context.relationship_context.get("limits", [])
            sensitive.extend(limits)
        
        return sensitive
    
    def _calculate_optimal_frequency(self, state: Dict) -> float:
        """Calculate optimal frequency for sadistic responses"""
        # Based on user's intensity preference
        intensity_pref = state.get("sadistic_intensity_preference", 0.5)
        
        # Higher intensity preference = can use more frequently
        # Lower intensity preference = use less frequently
        return 0.3 + (intensity_pref * 0.4)  # 0.3-0.7 responses per interaction
    
    async def _determine_sadistic_tone(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine sadistic tone elements for response"""
        tone_elements = {
            "base": "amused_superiority",
            "modifiers": []
        }
        
        # Check humiliation level
        humiliation = context.session_context.get("humiliation_level", 0.0)
        if humiliation > 0.7:
            tone_elements["modifiers"].append("delighted_cruelty")
        elif humiliation > 0.4:
            tone_elements["modifiers"].append("mild_amusement")
        
        # Check for protocol violations
        protocol_violations = any(
            msg["type"] == "protocol_violation_detected"
            for msgs in messages.values() for msg in msgs
        )
        if protocol_violations:
            tone_elements["modifiers"].append("mocking_disappointment")
        
        return tone_elements
    
    async def _calculate_amusement_level(self, context: SharedContext) -> float:
        """Calculate appropriate amusement level"""
        base_amusement = 0.3
        
        # Increase for humiliation
        humiliation = context.session_context.get("humiliation_level", 0.0)
        base_amusement += humiliation * 0.4
        
        # Increase for high submission
        submission = context.session_context.get("submission_level", 0.5)
        base_amusement += (submission - 0.5) * 0.3
        
        return min(1.0, base_amusement)
    
    async def _select_mockery_elements(self, context: SharedContext, messages: Dict) -> List[str]:
        """Select elements to mock in response"""
        mockery_elements = []
        
        # Check for protocol failures
        for module_name, module_messages in messages.items():
            if module_name == "protocol_enforcement":
                for msg in module_messages:
                    if msg["type"] == "protocol_violation_detected":
                        mockery_elements.append("protocol_failure")
        
        # Check for performance issues
        if "mistake" in context.user_input.lower() or "sorry" in context.user_input.lower():
            mockery_elements.append("inadequate_performance")
        
        return mockery_elements
    
    async def _assess_degradation_appropriateness(self, context: SharedContext) -> Dict[str, Any]:
        """Assess if degradation is appropriate"""
        trust = context.relationship_context.get("trust", 0.5)
        submission = context.session_context.get("submission_level", 0.5)
        
        appropriate = trust > 0.7 and submission > 0.6
        
        return {
            "appropriate": appropriate,
            "trust_level": trust,
            "submission_level": submission,
            "limiting_factor": "trust" if trust <= 0.7 else "submission" if submission <= 0.6 else None
        }
    
    async def _determine_intensity_modulation(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine how to modulate sadistic intensity"""
        base_intensity = 0.5
        modulations = []
        
        # Check emotional state
        if context.emotional_state:
            arousal = context.emotional_state.get("arousal", 0.5)
            if arousal > 0.7:
                base_intensity += 0.2
                modulations.append("high_arousal_boost")
            
            anxiety = context.emotional_state.get("anxiety", 0.0)
            if anxiety > 0.6:
                base_intensity -= 0.3
                modulations.append("anxiety_reduction")
        
        # Check subspace
        for module_name, module_messages in messages.items():
            if module_name == "psychological_dominance":
                for msg in module_messages:
                    if msg["type"] == "subspace_detected":
                        depth = msg["data"].get("depth", 0.0)
                        if depth > 0.5:
                            base_intensity -= 0.2
                            modulations.append("subspace_gentling")
        
        return {
            "final_intensity": max(0.1, min(1.0, base_intensity)),
            "modulations": modulations,
            "base_intensity": 0.5
        }
    
    async def _check_sadistic_milestones(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check for sadistic interaction milestones"""
        state = await self._get_sadistic_state_context(user_id)
        
        if state.get("humiliation_level", 0) > 0.9:
            return {
                "milestone_type": "peak_humiliation",
                "description": "Achieved maximum humiliation response",
                "humiliation_level": state["humiliation_level"]
            }
        
        return None
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original sadistic system"""
        return getattr(self.original_system, name)
