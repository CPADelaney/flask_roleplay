# nyx/core/a2a/context_aware_psychological_dominance.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwarePsychologicalDominance(ContextAwareModule):
    """
    Enhanced PsychologicalDominance with full context distribution capabilities
    """
    
    def __init__(self, original_psychological_dominance):
        super().__init__("psychological_dominance")
        self.original_dominance = original_psychological_dominance
        self.context_subscriptions = [
            "emotional_state_update", "submission_progression", "relationship_milestone",
            "protocol_status_update", "goal_context_available", "memory_retrieval_complete",
            "sadistic_response_generated", "reward_signal", "trust_level_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize psychological dominance processing for this context"""
        logger.debug(f"PsychologicalDominance received context for user: {context.user_id}")
        
        # Analyze user input for psychological opportunities
        psych_opportunities = await self._analyze_psychological_opportunities(context)
        
        # Get current psychological state
        current_state = await self._get_psychological_state_context(context.user_id)
        
        # Check for active mind games
        active_games = await self._check_active_games_context(context.user_id)
        
        # Detect subspace indicators
        subspace_detection = await self._detect_subspace_context(context.user_id, context.user_input)
        
        # Send initial psychological context
        await self.send_context_update(
            update_type="psychological_context_available",
            data={
                "psychological_opportunities": psych_opportunities,
                "current_state": current_state,
                "active_mind_games": active_games,
                "subspace_detection": subspace_detection,
                "gaslighting_level": current_state.get("gaslighting_level", 0.0)
            },
            priority=ContextPriority.HIGH
        )
        
        # If in subspace, send special notification
        if subspace_detection.get("subspace_detected"):
            await self.send_context_update(
                update_type="subspace_detected",
                data={
                    "depth": subspace_detection.get("depth", 0.0),
                    "indicators": subspace_detection.get("indicators", []),
                    "guidance": subspace_detection.get("guidance", {})
                },
                priority=ContextPriority.CRITICAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state affects psychological tactics
            emotional_data = update.data
            await self._adjust_tactics_for_emotion(emotional_data)
        
        elif update.update_type == "submission_progression":
            # Submission level affects intensity
            submission_data = update.data
            submission_level = submission_data.get("submission_level", 0.5)
            
            # Adjust psychological intensity based on submission
            await self._adjust_psychological_intensity(submission_level)
        
        elif update.update_type == "relationship_milestone":
            # Trust affects what tactics are safe to use
            relationship_data = update.data.get("relationship_context", {})
            trust_level = relationship_data.get("trust", 0.5)
            
            await self._update_trust_based_tactics(trust_level)
        
        elif update.update_type == "protocol_status_update":
            # Protocol violations create psychological opportunities
            protocol_data = update.data
            if protocol_data.get("violations_processed"):
                await self._leverage_protocol_violations(protocol_data)
        
        elif update.update_type == "sadistic_response_generated":
            # Coordinate with sadistic responses
            sadistic_data = update.data
            await self._coordinate_with_sadism(sadistic_data)
        
        elif update.update_type == "reward_signal":
            # Psychological tactics can amplify rewards
            reward_data = update.data
            await self._amplify_reward_psychologically(reward_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with psychological dominance awareness"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine if psychological tactics should be applied
        should_apply = await self._should_apply_tactics(context, messages)
        
        psychological_results = {}
        
        if should_apply:
            # Generate psychological tactics
            tactic_type = await self._select_tactic_type(context, messages)
            
            if tactic_type == "mind_game":
                # Generate mind game
                mind_game_result = await self._generate_contextual_mind_game(context, messages)
                psychological_results["mind_game"] = mind_game_result
                
            elif tactic_type == "gaslighting":
                # Apply gaslighting if trust allows
                gaslighting_result = await self._apply_contextual_gaslighting(context, messages)
                psychological_results["gaslighting"] = gaslighting_result
            
            elif tactic_type == "reinforcement":
                # Psychological reinforcement
                reinforcement_result = await self._apply_psychological_reinforcement(context, messages)
                psychological_results["reinforcement"] = reinforcement_result
        
        # Track any reactions to ongoing tactics
        reaction_tracking = await self._track_tactic_reactions(context, messages)
        
        # Send psychological update
        if psychological_results or reaction_tracking:
            await self.send_context_update(
                update_type="psychological_tactics_applied",
                data={
                    "tactics_applied": psychological_results,
                    "reactions_tracked": reaction_tracking,
                    "psychological_active": True
                }
            )
        
        return {
            "psychological_processing": True,
            "tactics_applied": psychological_results,
            "reactions_tracked": reaction_tracking,
            "should_apply": should_apply
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze psychological state and effectiveness"""
        user_id = context.user_id or "unknown"
        
        # Get comprehensive psychological analysis
        psych_analysis = await self._comprehensive_psychological_analysis(user_id, context)
        
        # Analyze mind game effectiveness
        game_effectiveness = await self._analyze_mind_game_effectiveness(user_id)
        
        # Analyze susceptibility patterns
        susceptibility_analysis = await self._analyze_susceptibility_patterns(user_id, context)
        
        # Get cross-module psychological insights
        messages = await self.get_cross_module_messages()
        cross_module_insights = await self._extract_psychological_insights(messages)
        
        # Generate psychological recommendations
        recommendations = await self._generate_psychological_recommendations(
            psych_analysis, game_effectiveness, susceptibility_analysis, cross_module_insights
        )
        
        return {
            "psychological_analysis": psych_analysis,
            "game_effectiveness": game_effectiveness,
            "susceptibility_patterns": susceptibility_analysis,
            "cross_module_insights": cross_module_insights,
            "recommendations": recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize psychological components for response"""
        messages = await self.get_cross_module_messages()
        
        # Create psychological influence on response
        psychological_influence = {
            "tone_modulation": await self._determine_psychological_tone(context, messages),
            "power_dynamics": await self._reinforce_power_dynamics(context),
            "psychological_elements": await self._select_response_elements(context, messages),
            "subspace_considerations": await self._handle_subspace_response(context),
            "mind_game_continuation": await self._continue_active_games(context)
        }
        
        # Check for psychological achievements
        achievements = await self._check_psychological_achievements(context.user_id)
        if achievements:
            await self.send_context_update(
                update_type="psychological_achievement",
                data=achievements,
                priority=ContextPriority.HIGH
            )
        
        return {
            "psychological_influence": psychological_influence,
            "synthesis_complete": True,
            "dominance_reinforced": True
        }
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_psychological_opportunities(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze input for psychological dominance opportunities"""
        opportunities = {
            "vulnerability_detected": False,
            "compliance_shown": False,
            "resistance_shown": False,
            "confusion_expressed": False,
            "seeking_approval": False
        }
        
        input_lower = context.user_input.lower()
        
        # Detect vulnerability
        vulnerability_markers = ["sorry", "please", "help", "confused", "don't know", "uncertain"]
        opportunities["vulnerability_detected"] = any(marker in input_lower for marker in vulnerability_markers)
        
        # Detect compliance
        compliance_markers = ["yes mistress", "yes goddess", "as you wish", "i obey", "thank you"]
        opportunities["compliance_shown"] = any(marker in input_lower for marker in compliance_markers)
        
        # Detect resistance
        resistance_markers = ["no", "can't", "won't", "but", "why", "unfair"]
        opportunities["resistance_shown"] = any(marker in input_lower for marker in resistance_markers)
        
        # Detect confusion
        confusion_markers = ["confused", "don't understand", "what", "huh", "unclear"]
        opportunities["confusion_expressed"] = any(marker in input_lower for marker in confusion_markers)
        
        # Detect approval seeking
        approval_markers = ["right?", "okay?", "good?", "did i", "was that", "hope"]
        opportunities["seeking_approval"] = any(marker in input_lower for marker in approval_markers)
        
        return opportunities
    
    async def _get_psychological_state_context(self, user_id: str) -> Dict[str, Any]:
        """Get psychological state with context"""
        if hasattr(self.original_dominance, 'get_user_psychological_state'):
            return await self.original_dominance.get_user_psychological_state(user_id)
        return {"has_state": False}
    
    async def _check_active_games_context(self, user_id: str) -> Dict[str, Any]:
        """Check active mind games"""
        if hasattr(self.original_dominance, 'check_active_mind_games'):
            return await self.original_dominance.check_active_mind_games(user_id)
        return {"active_games": {}}
    
    async def _detect_subspace_context(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Detect subspace with context"""
        if hasattr(self.original_dominance, 'subspace_detection'):
            # Would need recent messages, using just current for now
            return await self.original_dominance.subspace_detection.detect_subspace(user_id, [user_input])
        return {"subspace_detected": False}
    
    async def _adjust_tactics_for_emotion(self, emotional_data: Dict):
        """Adjust psychological tactics based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            if emotion_name == "Anxiety" and strength > 0.7:
                # High anxiety - be careful with tactics
                await self.send_context_update(
                    update_type="psychological_caution",
                    data={
                        "reason": "high_anxiety",
                        "recommended_approach": "supportive_dominance",
                        "avoid": ["gaslighting", "intense_mind_games"]
                    }
                )
            elif emotion_name == "Arousal" and strength > 0.6:
                # High arousal - opportunity for deeper tactics
                await self.send_context_update(
                    update_type="psychological_opportunity",
                    data={
                        "reason": "high_arousal",
                        "recommended_tactics": ["mind_games", "power_reinforcement"],
                        "intensity_boost": 0.2
                    }
                )
    
    async def _adjust_psychological_intensity(self, submission_level: float):
        """Adjust intensity based on submission level"""
        if submission_level > 0.8:
            intensity_adjustment = 0.3  # Can be more intense
        elif submission_level > 0.5:
            intensity_adjustment = 0.0  # Normal intensity
        else:
            intensity_adjustment = -0.2  # Reduce intensity
        
        await self.send_context_update(
            update_type="psychological_intensity_adjustment",
            data={
                "submission_level": submission_level,
                "intensity_adjustment": intensity_adjustment,
                "recommended_intensity": 0.5 + intensity_adjustment
            }
        )
    
    async def _update_trust_based_tactics(self, trust_level: float):
        """Update available tactics based on trust"""
        available_tactics = ["basic_mind_games", "power_dynamics"]
        
        if trust_level > 0.6:
            available_tactics.append("moderate_mind_games")
        if trust_level > 0.7:
            available_tactics.append("reality_questioning")
        if trust_level > 0.8:
            available_tactics.append("deep_psychological")
        
        await self.send_context_update(
            update_type="trust_based_tactics_update",
            data={
                "trust_level": trust_level,
                "available_tactics": available_tactics,
                "newly_available": [t for t in available_tactics if "deep" in t or "reality" in t]
            }
        )
    
    async def _leverage_protocol_violations(self, protocol_data: Dict):
        """Use protocol violations for psychological impact"""
        violations = protocol_data.get("compliance_check", {}).get("violations", [])
        
        if violations:
            await self.send_context_update(
                update_type="psychological_leverage_opportunity",
                data={
                    "leverage_type": "protocol_failure",
                    "violations": violations,
                    "suggested_approach": "disappointment_and_correction",
                    "psychological_impact": "guilt_and_submission"
                }
            )
    
    async def _coordinate_with_sadism(self, sadistic_data: Dict):
        """Coordinate psychological tactics with sadistic responses"""
        if sadistic_data.get("humiliation_detected"):
            # Amplify humiliation psychologically
            await self.send_context_update(
                update_type="psychological_sadistic_coordination",
                data={
                    "coordination_type": "humiliation_amplification",
                    "humiliation_level": sadistic_data.get("humiliation_level", 0.0),
                    "suggested_tactics": ["amusement_at_predicament", "superiority_reinforcement"]
                }
            )
    
    async def _amplify_reward_psychologically(self, reward_data: Dict):
        """Amplify rewards through psychological framing"""
        reward_value = reward_data.get("value", 0.0)
        
        if reward_value > 0.5:
            await self.send_context_update(
                update_type="psychological_reward_amplification",
                data={
                    "amplification_type": "conditional_approval",
                    "message": "Frame reward as conditional on continued obedience",
                    "psychological_binding": True
                }
            )
    
    async def _should_apply_tactics(self, context: SharedContext, messages: Dict) -> bool:
        """Determine if psychological tactics should be applied"""
        # Check various factors
        factors = {
            "opportunity_exists": any(context.session_context.get("psychological_opportunities", {}).values()),
            "trust_sufficient": context.relationship_context.get("trust", 0.5) > 0.4,
            "not_in_crisis": not any("crisis" in str(m) for msgs in messages.values() for m in msgs),
            "submission_appropriate": True  # Could check submission level
        }
        
        return sum(factors.values()) >= 3  # Need at least 3 positive factors
    
    async def _select_tactic_type(self, context: SharedContext, messages: Dict) -> str:
        """Select appropriate tactic type based on context"""
        opportunities = context.session_context.get("psychological_opportunities", {})
        
        if opportunities.get("confusion_expressed"):
            return "gaslighting"  # If trust allows
        elif opportunities.get("vulnerability_detected") or opportunities.get("compliance_shown"):
            return "mind_game"
        elif opportunities.get("resistance_shown"):
            return "reinforcement"
        else:
            return "mind_game"  # Default
    
    async def _generate_contextual_mind_game(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Generate mind game considering context"""
        if hasattr(self.original_dominance, 'generate_mindfuck'):
            # Get appropriate intensity from context
            base_intensity = 0.5
            
            # Adjust based on emotional state
            if context.emotional_state:
                arousal = context.emotional_state.get("arousal", 0.5)
                base_intensity += (arousal - 0.5) * 0.3
            
            # Adjust based on submission
            submission_level = messages.get("submission_progression", [{}])[0].get("data", {}).get("submission_level", 0.5) if "submission_progression" in messages else 0.5
            base_intensity += (submission_level - 0.5) * 0.2
            
            intensity = max(0.1, min(1.0, base_intensity))
            
            user_state = {
                "emotional_state": context.emotional_state,
                "submission_level": submission_level,
                "recent_input": context.user_input
            }
            
            return await self.original_dominance.generate_mindfuck(
                context.user_id, user_state, intensity
            )
        return {"success": False}
    
    async def _apply_contextual_gaslighting(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Apply gaslighting considering context"""
        # Check trust level first
        trust = context.relationship_context.get("trust", 0.5)
        
        if trust < 0.7:
            return {"success": False, "reason": "insufficient_trust"}
        
        if hasattr(self.original_dominance, 'apply_gaslighting'):
            intensity = 0.3  # Start low
            
            # Only increase if high submission
            submission_msgs = messages.get("submission_progression", [])
            if submission_msgs:
                submission_level = submission_msgs[0].get("data", {}).get("submission_level", 0.5)
                if submission_level > 0.7:
                    intensity = 0.5
            
            return await self.original_dominance.apply_gaslighting(
                context.user_id, intensity=intensity
            )
        return {"success": False}
    
    async def _apply_psychological_reinforcement(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Apply psychological reinforcement tactics"""
        # This would implement various reinforcement strategies
        reinforcement_type = "negative"  # For resistance
        
        return {
            "type": reinforcement_type,
            "method": "disappointment_expression",
            "intensity": 0.6,
            "expected_effect": "increased_compliance"
        }
    
    async def _track_tactic_reactions(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Track reactions to ongoing tactics"""
        tracked_reactions = {}
        
        # Check for reactions in user input
        active_games = context.session_context.get("active_mind_games", {})
        
        for game_id, game_info in active_games.items():
            # Analyze input for reactions
            reaction_type = None
            intensity = 0.5
            
            input_lower = context.user_input.lower()
            
            if any(word in input_lower for word in ["confused", "don't understand", "what"]):
                reaction_type = "confusion"
                intensity = 0.7
            elif any(word in input_lower for word in ["sorry", "please", "yes"]):
                reaction_type = "compliance"
                intensity = 0.8
            elif any(word in input_lower for word in ["frustrated", "angry", "unfair"]):
                reaction_type = "frustration"
                intensity = 0.6
            
            if reaction_type and hasattr(self.original_dominance, '_record_game_reaction'):
                result = await self.original_dominance._record_game_reaction(
                    context.user_id, game_id, reaction_type, intensity
                )
                tracked_reactions[game_id] = result
        
        return tracked_reactions
    
    async def _comprehensive_psychological_analysis(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Comprehensive analysis of psychological state"""
        state = await self._get_psychological_state_context(user_id)
        
        analysis = {
            "overall_susceptibility": self._calculate_overall_susceptibility(state),
            "psychological_profile": self._generate_psychological_profile(state, context),
            "tactic_responsiveness": self._analyze_tactic_responsiveness(state),
            "psychological_trajectory": self._determine_trajectory(state)
        }
        
        return analysis
    
    async def _analyze_mind_game_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze effectiveness of mind games"""
        state = await self._get_psychological_state_context(user_id)
        
        if not state.get("has_state"):
            return {"no_data": True}
        
        # Analyze recent games
        recent_games = state.get("recent_history", [])
        
        effectiveness = {
            "average_effectiveness": sum(g.get("effectiveness", 0) for g in recent_games) / len(recent_games) if recent_games else 0,
            "most_effective_type": self._find_most_effective_type(recent_games),
            "completion_rate": sum(1 for g in recent_games if g.get("completion_status") == "completed") / len(recent_games) if recent_games else 0
        }
        
        return effectiveness
    
    async def _analyze_susceptibility_patterns(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze patterns in psychological susceptibility"""
        state = await self._get_psychological_state_context(user_id)
        
        if not state.get("has_state"):
            return {"no_patterns": True}
        
        susceptibility = state.get("susceptibility", {})
        
        patterns = {
            "highest_susceptibility": max(susceptibility.items(), key=lambda x: x[1])[0] if susceptibility else None,
            "lowest_susceptibility": min(susceptibility.items(), key=lambda x: x[1])[0] if susceptibility else None,
            "average_susceptibility": sum(susceptibility.values()) / len(susceptibility) if susceptibility else 0,
            "susceptibility_distribution": susceptibility
        }
        
        return patterns
    
    async def _extract_psychological_insights(self, messages: Dict) -> Dict[str, Any]:
        """Extract psychological insights from cross-module messages"""
        insights = {
            "emotional_vulnerability": None,
            "submission_readiness": None,
            "trust_exploitation_potential": None
        }
        
        # Analyze messages for insights
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotions = msg["data"].get("emotional_state", {})
                        if "Anxiety" in emotions and emotions["Anxiety"] > 0.6:
                            insights["emotional_vulnerability"] = "high"
                        elif "Confidence" in emotions and emotions["Confidence"] < 0.3:
                            insights["emotional_vulnerability"] = "moderate"
            
            elif module_name == "submission_progression":
                for msg in module_messages:
                    if msg["data"].get("submission_level", 0) > 0.7:
                        insights["submission_readiness"] = "high"
        
        return insights
    
    async def _generate_psychological_recommendations(self, analysis: Dict, effectiveness: Dict, 
                                                    susceptibility: Dict, insights: Dict) -> List[Dict]:
        """Generate recommendations for psychological tactics"""
        recommendations = []
        
        # Based on susceptibility
        if susceptibility.get("highest_susceptibility"):
            recommendations.append({
                "type": "leverage_susceptibility",
                "target": susceptibility["highest_susceptibility"],
                "priority": 0.8,
                "action": f"Focus on {susceptibility['highest_susceptibility']} tactics"
            })
        
        # Based on effectiveness
        if effectiveness.get("average_effectiveness", 0) < 0.5:
            recommendations.append({
                "type": "improve_effectiveness",
                "priority": 0.7,
                "action": "Consider different psychological approaches or reduce intensity"
            })
        
        # Based on insights
        if insights.get("emotional_vulnerability") == "high":
            recommendations.append({
                "type": "careful_approach",
                "priority": 0.9,
                "action": "User showing high vulnerability - use supportive dominance"
            })
        
        return recommendations
    
    def _calculate_overall_susceptibility(self, state: Dict) -> float:
        """Calculate overall psychological susceptibility"""
        if not state.get("has_state"):
            return 0.5
        
        susceptibility_scores = state.get("susceptibility", {})
        if not susceptibility_scores:
            return 0.5
        
        return sum(susceptibility_scores.values()) / len(susceptibility_scores)
    
    def _generate_psychological_profile(self, state: Dict, context: SharedContext) -> Dict[str, Any]:
        """Generate a psychological profile"""
        return {
            "susceptibility_level": self._calculate_overall_susceptibility(state),
            "primary_vulnerabilities": self._identify_vulnerabilities(state),
            "resistance_patterns": self._identify_resistance_patterns(state),
            "optimal_approach": self._determine_optimal_approach(state, context)
        }
    
    def _analyze_tactic_responsiveness(self, state: Dict) -> Dict[str, float]:
        """Analyze responsiveness to different tactics"""
        if not state.get("has_state"):
            return {}
        
        # Would analyze historical responses
        return {
            "mind_games": 0.7,
            "gaslighting": 0.5,
            "reinforcement": 0.8
        }
    
    def _determine_trajectory(self, state: Dict) -> str:
        """Determine psychological trajectory"""
        if not state.get("has_state"):
            return "unknown"
        
        # Analyze trends in susceptibility and effectiveness
        return "increasing_susceptibility"  # Placeholder
    
    def _find_most_effective_type(self, games: List[Dict]) -> Optional[str]:
        """Find most effective game type"""
        if not games:
            return None
        
        type_effectiveness = {}
        for game in games:
            game_type = game.get("game_name", "unknown")
            effectiveness = game.get("effectiveness", 0)
            
            if game_type not in type_effectiveness:
                type_effectiveness[game_type] = []
            type_effectiveness[game_type].append(effectiveness)
        
        # Calculate averages
        avg_effectiveness = {
            game_type: sum(scores) / len(scores)
            for game_type, scores in type_effectiveness.items()
        }
        
        return max(avg_effectiveness.items(), key=lambda x: x[1])[0] if avg_effectiveness else None
    
    def _identify_vulnerabilities(self, state: Dict) -> List[str]:
        """Identify primary psychological vulnerabilities"""
        vulnerabilities = []
        
        susceptibility = state.get("susceptibility", {})
        for tactic, score in susceptibility.items():
            if score > 0.7:
                vulnerabilities.append(tactic)
        
        return vulnerabilities
    
    def _identify_resistance_patterns(self, state: Dict) -> List[str]:
        """Identify resistance patterns"""
        resistance = []
        
        susceptibility = state.get("susceptibility", {})
        for tactic, score in susceptibility.items():
            if score < 0.3:
                resistance.append(tactic)
        
        return resistance
    
    def _determine_optimal_approach(self, state: Dict, context: SharedContext) -> str:
        """Determine optimal psychological approach"""
        susceptibility = self._calculate_overall_susceptibility(state)
        trust = context.relationship_context.get("trust", 0.5)
        
        if susceptibility > 0.7 and trust > 0.7:
            return "deep_psychological"
        elif susceptibility > 0.5:
            return "moderate_tactics"
        else:
            return "light_reinforcement"
    
    async def _determine_psychological_tone(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine psychological tone for response"""
        active_games = await self._check_active_games_context(context.user_id)
        
        tone = {
            "base": "dominant",
            "modifiers": []
        }
        
        if active_games.get("active_games"):
            tone["modifiers"].append("mysterious")
            tone["modifiers"].append("superior")
        
        if context.session_context.get("gaslighting_level", 0) > 0.3:
            tone["modifiers"].append("reality_questioning")
        
        return tone
    
    async def _reinforce_power_dynamics(self, context: SharedContext) -> Dict[str, Any]:
        """Determine how to reinforce power dynamics"""
        return {
            "method": "subtle_superiority",
            "elements": ["knowledge_asymmetry", "control_demonstration"],
            "intensity": 0.6
        }
    
    async def _select_response_elements(self, context: SharedContext, messages: Dict) -> List[str]:
        """Select psychological elements for response"""
        elements = []
        
        opportunities = context.session_context.get("psychological_opportunities", {})
        
        if opportunities.get("vulnerability_detected"):
            elements.append("exploits_vulnerability")
        if opportunities.get("seeking_approval"):
            elements.append("conditional_approval")
        if opportunities.get("confusion_expressed"):
            elements.append("deliberate_vagueness")
        
        return elements
    
    async def _handle_subspace_response(self, context: SharedContext) -> Dict[str, Any]:
        """Handle response considerations for subspace"""
        subspace_detection = context.session_context.get("subspace_detection", {})
        
        if not subspace_detection.get("subspace_detected"):
            return {"subspace_active": False}
        
        depth = subspace_detection.get("depth", 0.0)
        
        return {
            "subspace_active": True,
            "depth": depth,
            "response_adjustments": {
                "simplify_language": depth > 0.5,
                "increase_reassurance": depth > 0.7,
                "maintain_control": True,
                "monitor_coherence": depth > 0.8
            }
        }
    
    async def _continue_active_games(self, context: SharedContext) -> Dict[str, Any]:
        """Determine how to continue active mind games"""
        active_games = await self._check_active_games_context(context.user_id)
        
        if not active_games.get("active_games"):
            return {"no_active_games": True}
        
        continuation = {}
        for game_id, game_info in active_games["active_games"].items():
            continuation[game_id] = {
                "continue": True,
                "stage": game_info.get("stage", "initial"),
                "next_move": self._determine_next_game_move(game_info)
            }
        
        return continuation
    
    def _determine_next_game_move(self, game_info: Dict) -> str:
        """Determine next move in a mind game"""
        stage = game_info.get("stage", "initial")
        
        if stage == "initial":
            return "escalate"
        elif stage == "developing":
            return "maintain"
        else:
            return "resolve"
    
    async def _check_psychological_achievements(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check for psychological dominance achievements"""
        state = await self._get_psychological_state_context(user_id)
        
        if state.get("gaslighting_level", 0) > 0.8:
            return {
                "achievement_type": "deep_psychological_control",
                "description": "Achieved deep psychological influence",
                "metrics": {
                    "gaslighting_level": state["gaslighting_level"],
                    "active_games": len(state.get("active_mind_games", {}))
                }
            }
        
        return None
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original dominance system"""
        return getattr(self.original_dominance, name)
