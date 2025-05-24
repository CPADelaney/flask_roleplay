# nyx/core/a2a/context_aware_protocol_enforcement.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareProtocolEnforcement(ContextAwareModule):
    """
    Enhanced ProtocolEnforcement with full context distribution capabilities
    """
    
    def __init__(self, original_protocol_enforcement):
        super().__init__("protocol_enforcement")
        self.original_enforcement = original_protocol_enforcement
        self.context_subscriptions = [
            "emotional_state_update", "goal_context_available", "relationship_milestone",
            "submission_progression", "dominance_gratification", "protocol_violation_detected",
            "ritual_completion", "memory_retrieval_complete", "psychological_state_update"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize protocol processing for this context"""
        logger.debug(f"ProtocolEnforcement received context for user: {context.user_id}")
        
        # Check for protocol compliance in the user input
        compliance_check = await self._check_input_compliance(context.user_input, context.user_id)
        
        # Get active protocols and rituals for this user
        active_protocols = await self._get_user_protocols_context(context.user_id)
        active_rituals = await self._get_user_rituals_context(context.user_id)
        due_rituals = await self._check_due_rituals_context(context.user_id)
        
        # Send initial protocol context to other modules
        await self.send_context_update(
            update_type="protocol_context_available",
            data={
                "compliance_check": compliance_check,
                "active_protocols": active_protocols,
                "active_rituals": active_rituals,
                "due_rituals": due_rituals,
                "protocol_count": len(active_protocols),
                "ritual_count": len(active_rituals),
                "compliance_status": compliance_check.get("compliant", True)
            },
            priority=ContextPriority.HIGH
        )
        
        # If non-compliant, send violation notification
        if not compliance_check.get("compliant", True):
            await self.send_context_update(
                update_type="protocol_violation_detected",
                data={
                    "violations": compliance_check.get("violations", []),
                    "user_input": context.user_input,
                    "requires_correction": True
                },
                priority=ContextPriority.CRITICAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state affects protocol strictness
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion and dominant_emotion[0] == "Frustration":
                # User frustration might indicate protocol difficulty
                await self._consider_protocol_adjustment(update.source_module, emotional_data)
        
        elif update.update_type == "submission_progression":
            # Submission level affects protocol assignment
            submission_data = update.data
            submission_level = submission_data.get("submission_level", 0.5)
            
            if submission_level > 0.8:
                # High submission - consider more challenging protocols
                await self._suggest_advanced_protocols(submission_data)
            elif submission_level < 0.3:
                # Low submission - ensure basic protocols are in place
                await self._enforce_basic_protocols(submission_data)
        
        elif update.update_type == "relationship_milestone":
            # Relationship progress affects protocol complexity
            relationship_data = update.data.get("relationship_context", {})
            trust_level = relationship_data.get("trust", 0.5)
            
            if trust_level > 0.7:
                await self._unlock_trust_based_protocols(relationship_data)
        
        elif update.update_type == "goal_context_available":
            # Align protocols with active goals
            goal_data = update.data
            active_goals = goal_data.get("active_goals", [])
            
            await self._align_protocols_with_goals(active_goals)
        
        elif update.update_type == "psychological_state_update":
            # Psychological state affects protocol intensity
            psych_data = update.data
            if psych_data.get("in_subspace"):
                # User in subspace - adjust protocol enforcement
                await self._adjust_for_subspace(psych_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with protocol awareness"""
        # Check compliance with all active protocols
        full_compliance_check = await self._comprehensive_compliance_check(
            context.user_input, context.user_id, context
        )
        
        # Get cross-module messages for context
        messages = await self.get_cross_module_messages()
        
        # Process any violations detected
        violation_results = []
        if full_compliance_check.get("violations"):
            violation_results = await self._process_violations_with_context(
                full_compliance_check["violations"], context, messages
            )
        
        # Check for ritual opportunities
        ritual_opportunities = await self._identify_ritual_opportunities(context, messages)
        
        # Send protocol status update
        await self.send_context_update(
            update_type="protocol_status_update",
            data={
                "compliance_check": full_compliance_check,
                "violations_processed": len(violation_results),
                "ritual_opportunities": ritual_opportunities,
                "enforcement_active": True
            }
        )
        
        return {
            "protocols_checked": True,
            "compliance_result": full_compliance_check,
            "violations_processed": violation_results,
            "ritual_opportunities": ritual_opportunities,
            "cross_module_context": len(messages)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze protocol effectiveness and user compliance patterns"""
        # Get comprehensive compliance statistics
        user_id = context.user_id or "unknown"
        compliance_stats = await self._analyze_compliance_patterns(user_id, context)
        
        # Analyze protocol effectiveness
        protocol_effectiveness = await self._analyze_protocol_effectiveness(user_id, context)
        
        # Analyze ritual completion patterns
        ritual_patterns = await self._analyze_ritual_patterns(user_id, context)
        
        # Get cross-module insights
        messages = await self.get_cross_module_messages()
        cross_module_insights = await self._extract_protocol_insights(messages)
        
        # Recommend protocol adjustments
        recommendations = await self._generate_protocol_recommendations(
            compliance_stats, protocol_effectiveness, ritual_patterns, cross_module_insights
        )
        
        return {
            "compliance_analysis": compliance_stats,
            "protocol_effectiveness": protocol_effectiveness,
            "ritual_patterns": ritual_patterns,
            "cross_module_insights": cross_module_insights,
            "recommendations": recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize protocol-related components for response"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Determine protocol influence on response
        protocol_influence = {
            "correction_needed": await self._determine_correction_need(context),
            "ritual_reminders": await self._generate_ritual_reminders(context),
            "protocol_reinforcement": await self._suggest_protocol_reinforcement(context),
            "honorific_requirements": await self._get_honorific_requirements(context),
            "response_constraints": await self._determine_response_constraints(context, messages)
        }
        
        # Check for protocol milestones
        milestones = await self._check_protocol_milestones(context.user_id)
        if milestones:
            await self.send_context_update(
                update_type="protocol_milestone_achieved",
                data=milestones,
                priority=ContextPriority.HIGH
            )
        
        return {
            "protocol_influence": protocol_influence,
            "synthesis_complete": True,
            "protocol_constraints_active": True
        }
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _check_input_compliance(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Check input compliance using original enforcement system"""
        if hasattr(self.original_enforcement, 'check_protocol_compliance'):
            return await self.original_enforcement.check_protocol_compliance(user_id, user_input)
        return {"compliant": True, "violations": []}
    
    async def _get_user_protocols_context(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active protocols with context information"""
        if hasattr(self.original_enforcement, 'get_user_protocols'):
            result = await self.original_enforcement.get_user_protocols(user_id)
            return result.get("protocols", [])
        return []
    
    async def _get_user_rituals_context(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active rituals with context information"""
        if hasattr(self.original_enforcement, 'get_user_rituals'):
            result = await self.original_enforcement.get_user_rituals(user_id)
            return result.get("rituals", [])
        return []
    
    async def _check_due_rituals_context(self, user_id: str) -> List[Dict[str, Any]]:
        """Check for due rituals"""
        if hasattr(self.original_enforcement, 'check_due_rituals'):
            return await self.original_enforcement.check_due_rituals(user_id)
        return []
    
    async def _comprehensive_compliance_check(self, user_input: str, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Comprehensive compliance check considering full context"""
        base_check = await self._check_input_compliance(user_input, user_id)
        
        # Enhance with context awareness
        enhanced_violations = []
        for violation in base_check.get("violations", []):
            # Add context to each violation
            enhanced_violation = violation.copy()
            
            # Consider emotional context
            if context.emotional_state:
                enhanced_violation["emotional_context"] = context.emotional_state
                
            # Consider relationship context
            if context.relationship_context:
                trust = context.relationship_context.get("trust", 0.5)
                # Adjust severity based on trust
                enhanced_violation["adjusted_severity"] = violation.get("severity", 0.5) * (2.0 - trust)
            
            enhanced_violations.append(enhanced_violation)
        
        return {
            "compliant": base_check.get("compliant", True),
            "violations": enhanced_violations,
            "context_considered": True
        }
    
    async def _process_violations_with_context(self, violations: List[Dict], context: SharedContext, messages: Dict) -> List[Dict]:
        """Process violations considering full context"""
        processed_violations = []
        
        for violation in violations:
            # Record the violation
            if hasattr(self.original_enforcement, 'record_protocol_violation'):
                result = await self.original_enforcement.record_protocol_violation(
                    user_id=context.user_id,
                    protocol_id=violation.get("protocol_id"),
                    description=violation.get("description"),
                    severity=violation.get("adjusted_severity", violation.get("severity", 0.5))
                )
                
                processed_violations.append({
                    "violation": violation,
                    "processing_result": result,
                    "context_applied": True
                })
        
        return processed_violations
    
    async def _identify_ritual_opportunities(self, context: SharedContext, messages: Dict) -> List[Dict]:
        """Identify opportunities for ritual performance based on context"""
        opportunities = []
        
        # Check time-based rituals
        due_rituals = await self._check_due_rituals_context(context.user_id)
        for ritual in due_rituals:
            opportunities.append({
                "type": "due_ritual",
                "ritual": ritual,
                "urgency": min(1.0, ritual.get("due_since", 0) / 24.0)  # Scale by hours overdue
            })
        
        # Check context-triggered rituals
        if context.emotional_state:
            # High arousal might trigger certain rituals
            arousal = context.emotional_state.get("arousal", 0.0)
            if arousal > 0.7:
                opportunities.append({
                    "type": "arousal_triggered",
                    "suggested_ritual": "submission_ritual",
                    "trigger": "high_arousal"
                })
        
        return opportunities
    
    async def _analyze_compliance_patterns(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze user's compliance patterns"""
        if hasattr(self.original_enforcement, 'get_protocol_compliance_stats'):
            stats = await self.original_enforcement.get_protocol_compliance_stats(user_id)
            
            # Enhance with pattern analysis
            patterns = {
                "overall_compliance_rate": stats.get("compliance_rate", 0.0),
                "violations_by_protocol": stats.get("violations_by_protocol", {}),
                "recent_violations": stats.get("recent_violations", []),
                "compliance_trend": self._calculate_compliance_trend(stats),
                "problem_areas": self._identify_problem_areas(stats)
            }
            
            return patterns
        return {}
    
    async def _analyze_protocol_effectiveness(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze how effective protocols are for this user"""
        active_protocols = await self._get_user_protocols_context(user_id)
        compliance_stats = await self.original_enforcement.get_protocol_compliance_stats(user_id) if hasattr(self.original_enforcement, 'get_protocol_compliance_stats') else {}
        
        effectiveness = {}
        for protocol in active_protocols:
            protocol_id = protocol["id"]
            violations = compliance_stats.get("violations_by_protocol", {}).get(protocol_id, 0)
            
            # Calculate effectiveness score
            if violations == 0:
                effectiveness[protocol_id] = 1.0
            else:
                # Decay effectiveness based on violations
                effectiveness[protocol_id] = max(0.0, 1.0 - (violations * 0.1))
        
        return {
            "protocol_effectiveness_scores": effectiveness,
            "average_effectiveness": sum(effectiveness.values()) / len(effectiveness) if effectiveness else 0.0,
            "least_effective": min(effectiveness.items(), key=lambda x: x[1])[0] if effectiveness else None,
            "most_effective": max(effectiveness.items(), key=lambda x: x[1])[0] if effectiveness else None
        }
    
    async def _analyze_ritual_patterns(self, user_id: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze ritual completion patterns"""
        if hasattr(self.original_enforcement, 'get_ritual_completion_stats'):
            stats = await self.original_enforcement.get_ritual_completion_stats(user_id)
            
            patterns = {
                "completion_rate": self._calculate_ritual_completion_rate(stats),
                "average_quality": stats.get("average_quality", 0.0),
                "overdue_rituals": stats.get("overdue_rituals", []),
                "completion_by_ritual": stats.get("completions_by_ritual", {}),
                "quality_trend": self._calculate_quality_trend(stats)
            }
            
            return patterns
        return {}
    
    async def _consider_protocol_adjustment(self, source_module: str, emotional_data: Dict):
        """Consider adjusting protocols based on emotional state"""
        frustration_level = emotional_data.get("emotional_state", {}).get("Frustration", 0.0)
        
        if frustration_level > 0.7:
            await self.send_context_update(
                update_type="protocol_adjustment_suggested",
                data={
                    "reason": "high_frustration",
                    "frustration_level": frustration_level,
                    "suggestion": "reduce_protocol_difficulty",
                    "source": source_module
                },
                priority=ContextPriority.NORMAL
            )
    
    async def _suggest_advanced_protocols(self, submission_data: Dict):
        """Suggest more advanced protocols for high submission users"""
        await self.send_context_update(
            update_type="advanced_protocols_available",
            data={
                "submission_level": submission_data.get("submission_level"),
                "suggested_protocols": ["position_protocol", "advanced_service_protocol"],
                "reason": "high_submission_achievement"
            },
            priority=ContextPriority.NORMAL
        )
    
    async def _enforce_basic_protocols(self, submission_data: Dict):
        """Ensure basic protocols are enforced for low submission"""
        await self.send_context_update(
            update_type="basic_protocol_enforcement",
            data={
                "submission_level": submission_data.get("submission_level"),
                "required_protocols": ["address_protocol", "basic_obedience"],
                "enforcement_level": "strict"
            },
            priority=ContextPriority.HIGH
        )
    
    async def _unlock_trust_based_protocols(self, relationship_data: Dict):
        """Unlock protocols that require high trust"""
        trust_level = relationship_data.get("trust", 0.0)
        
        if trust_level > 0.8:
            unlocked_protocols = ["intimate_service_protocol", "vulnerability_protocol"]
        elif trust_level > 0.7:
            unlocked_protocols = ["emotional_control_protocol"]
        else:
            unlocked_protocols = []
        
        if unlocked_protocols:
            await self.send_context_update(
                update_type="trust_protocols_unlocked",
                data={
                    "trust_level": trust_level,
                    "unlocked_protocols": unlocked_protocols,
                    "relationship_context": relationship_data
                }
            )
    
    async def _align_protocols_with_goals(self, active_goals: List[Dict]):
        """Align protocol enforcement with active goals"""
        goal_aligned_protocols = []
        
        for goal in active_goals:
            if "submission" in goal.get("description", "").lower():
                goal_aligned_protocols.append("submission_protocol")
            elif "control" in goal.get("description", "").lower():
                goal_aligned_protocols.append("control_protocol")
            elif "service" in goal.get("description", "").lower():
                goal_aligned_protocols.append("service_protocol")
        
        if goal_aligned_protocols:
            await self.send_context_update(
                update_type="goal_aligned_protocols",
                data={
                    "aligned_protocols": goal_aligned_protocols,
                    "active_goals": [g["id"] for g in active_goals]
                }
            )
    
    async def _adjust_for_subspace(self, psych_data: Dict):
        """Adjust protocol enforcement for users in subspace"""
        subspace_depth = psych_data.get("subspace_depth", 0.0)
        
        if subspace_depth > 0.5:
            # Deep subspace - simplify protocols
            await self.send_context_update(
                update_type="subspace_protocol_adjustment",
                data={
                    "subspace_depth": subspace_depth,
                    "adjustment": "simplify_requirements",
                    "simplified_protocols": ["basic_obedience", "simple_responses"]
                },
                priority=ContextPriority.HIGH
            )
    
    def _calculate_compliance_trend(self, stats: Dict) -> str:
        """Calculate compliance trend from statistics"""
        recent_violations = stats.get("recent_violations", [])
        if not recent_violations:
            return "stable"
        
        # Simple trend analysis based on violation timestamps
        if len(recent_violations) >= 2:
            recent_times = [datetime.fromisoformat(v["timestamp"]) for v in recent_violations[-2:]]
            time_diff = (recent_times[1] - recent_times[0]).total_seconds()
            if time_diff < 3600:  # Less than an hour between violations
                return "declining"
        
        return "improving"
    
    def _identify_problem_areas(self, stats: Dict) -> List[str]:
        """Identify protocols with frequent violations"""
        violations_by_protocol = stats.get("violations_by_protocol", {})
        problem_areas = []
        
        for protocol_id, count in violations_by_protocol.items():
            if count > 3:  # More than 3 violations
                problem_areas.append(protocol_id)
        
        return problem_areas
    
    def _calculate_ritual_completion_rate(self, stats: Dict) -> float:
        """Calculate ritual completion rate"""
        total_rituals = stats.get("active_rituals", 0)
        total_completions = stats.get("total_completions", 0)
        
        if total_rituals == 0:
            return 0.0
        
        # Rough estimate assuming daily rituals over a week
        expected_completions = total_rituals * 7
        return min(1.0, total_completions / expected_completions)
    
    def _calculate_quality_trend(self, stats: Dict) -> str:
        """Calculate quality trend from completion history"""
        recent_completions = stats.get("recent_completions", [])
        if len(recent_completions) < 2:
            return "stable"
        
        # Compare recent quality scores
        recent_qualities = [c.get("quality", 0.5) for c in recent_completions[-3:]]
        if len(recent_qualities) >= 2:
            if recent_qualities[-1] > recent_qualities[-2]:
                return "improving"
            elif recent_qualities[-1] < recent_qualities[-2]:
                return "declining"
        
        return "stable"
    
    async def _extract_protocol_insights(self, messages: Dict) -> Dict[str, Any]:
        """Extract protocol-relevant insights from cross-module messages"""
        insights = {
            "emotional_compliance_correlation": None,
            "goal_protocol_alignment": None,
            "submission_protocol_effectiveness": None
        }
        
        # Look for emotional state correlation
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotional_state = msg["data"].get("emotional_state", {})
                        if "Frustration" in emotional_state:
                            insights["emotional_compliance_correlation"] = "negative"
                        elif "Pride" in emotional_state:
                            insights["emotional_compliance_correlation"] = "positive"
        
        return insights
    
    async def _generate_protocol_recommendations(self, compliance_stats: Dict, effectiveness: Dict, 
                                               ritual_patterns: Dict, cross_insights: Dict) -> List[Dict]:
        """Generate protocol recommendations based on analysis"""
        recommendations = []
        
        # Check compliance rate
        if compliance_stats.get("overall_compliance_rate", 0) < 0.7:
            recommendations.append({
                "type": "reduce_complexity",
                "reason": "low_compliance_rate",
                "priority": 0.8,
                "action": "Consider simplifying protocols or providing more guidance"
            })
        
        # Check effectiveness
        least_effective = effectiveness.get("least_effective")
        if least_effective and effectiveness.get("protocol_effectiveness_scores", {}).get(least_effective, 1.0) < 0.5:
            recommendations.append({
                "type": "replace_protocol",
                "protocol_id": least_effective,
                "reason": "low_effectiveness",
                "priority": 0.7,
                "action": f"Consider replacing {least_effective} with alternative approach"
            })
        
        # Check ritual quality
        if ritual_patterns.get("average_quality", 0) < 0.6:
            recommendations.append({
                "type": "ritual_guidance",
                "reason": "low_quality_completion",
                "priority": 0.6,
                "action": "Provide more detailed ritual instructions or reduce complexity"
            })
        
        return recommendations
    
    async def _determine_correction_need(self, context: SharedContext) -> Dict[str, Any]:
        """Determine if correction is needed in response"""
        # This would check recent violations and determine correction approach
        return {
            "needs_correction": False,
            "correction_type": None,
            "severity": 0.0
        }
    
    async def _generate_ritual_reminders(self, context: SharedContext) -> List[str]:
        """Generate ritual reminders if needed"""
        due_rituals = await self._check_due_rituals_context(context.user_id)
        reminders = []
        
        for ritual in due_rituals[:2]:  # Limit to 2 reminders
            reminders.append(f"Remember to complete your {ritual['name']}")
        
        return reminders
    
    async def _suggest_protocol_reinforcement(self, context: SharedContext) -> Dict[str, Any]:
        """Suggest how to reinforce protocols in response"""
        return {
            "reinforce_honorifics": True,
            "emphasis_level": 0.5,
            "specific_protocols": []
        }
    
    async def _get_honorific_requirements(self, context: SharedContext) -> Dict[str, Any]:
        """Get honorific requirements for this user"""
        if hasattr(self.original_enforcement, 'honorifics_enforcement'):
            # Get from honorifics subsystem
            return {
                "required_honorifics": ["Mistress", "Goddess", "Ma'am"],
                "enforcement_active": True
            }
        return {"required_honorifics": [], "enforcement_active": False}
    
    async def _determine_response_constraints(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine protocol-based constraints on response"""
        return {
            "formality_level": "high",
            "required_elements": [],
            "forbidden_elements": [],
            "tone_requirements": "strict"
        }
    
    async def _check_protocol_milestones(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check for protocol-related milestones"""
        compliance_stats = await self.original_enforcement.get_protocol_compliance_stats(user_id) if hasattr(self.original_enforcement, 'get_protocol_compliance_stats') else {}
        
        if compliance_stats.get("compliance_rate", 0) > 0.9:
            return {
                "milestone_type": "high_compliance",
                "achievement": "Exceptional Protocol Adherence",
                "compliance_rate": compliance_stats["compliance_rate"]
            }
        
        return None
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original enforcement system"""
        return getattr(self.original_enforcement, name)
