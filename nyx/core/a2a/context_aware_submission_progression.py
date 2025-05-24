# nyx/core/a2a/context_aware_submission_progression.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareSubmissionProgression(ContextAwareModule):
    """
    Context-aware wrapper for SubmissionProgression with full A2A capabilities
    """
    
    def __init__(self, original_submission_progression):
        super().__init__("submission_progression")
        self.original_system = original_submission_progression
        self.context_subscriptions = [
            "task_completion", "task_assignment", "compliance_record",
            "emotional_state_update", "relationship_milestone", 
            "dominance_interaction", "protocol_violation",
            "reward_signal", "punishment_applied",
            "goal_context_available", "needs_state_change"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize submission tracking for this context"""
        logger.debug(f"SubmissionProgression received context for user: {context.user_id}")
        
        # Analyze input for submission-related implications
        submission_implications = await self._analyze_input_for_submission(context.user_input)
        
        # Get user's current submission state
        user_submission_state = await self._get_user_submission_context(context.user_id)
        
        # Send initial submission context to other modules
        await self.send_context_update(
            update_type="submission_state_available",
            data={
                "user_id": context.user_id,
                "current_level": user_submission_state.get("current_level"),
                "submission_score": user_submission_state.get("submission_score", 0.0),
                "compliance_rate": user_submission_state.get("compliance_rate", 1.0),
                "active_path": user_submission_state.get("active_path"),
                "submission_implications": submission_implications,
                "recent_compliance": user_submission_state.get("recent_compliance", [])
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "task_completion":
            # Task completion affects submission metrics
            task_data = update.data
            user_id = task_data.get("user_id")
            completed = task_data.get("completed", False)
            task_difficulty = task_data.get("difficulty", 0.5)
            
            if user_id:
                # Record compliance based on task completion
                await self._record_task_compliance(user_id, task_data, completed)
                
                # Check for milestone progress if on a path
                await self._check_milestone_progress_from_task(user_id, task_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional states can affect submission dynamics
            emotional_data = update.data
            await self._process_emotional_influence_on_submission(emotional_data, context.user_id)
        
        elif update.update_type == "dominance_interaction":
            # Direct dominance interactions affect submission
            dominance_data = update.data
            interaction_type = dominance_data.get("interaction_type")
            intensity = dominance_data.get("intensity", 0.5)
            
            await self._process_dominance_interaction(context.user_id, interaction_type, intensity)
        
        elif update.update_type == "protocol_violation":
            # Protocol violations affect submission metrics
            violation_data = update.data
            user_id = violation_data.get("user_id")
            severity = violation_data.get("severity", 0.5)
            
            if user_id:
                await self._process_protocol_violation(user_id, severity, violation_data)
        
        elif update.update_type == "relationship_milestone":
            # Relationship milestones can unlock new submission paths
            relationship_data = update.data
            await self._check_path_unlocks(context.user_id, relationship_data)
        
        elif update.update_type == "goal_context_available":
            # Goals related to submission should be tracked
            goal_data = update.data
            await self._track_submission_goals(context.user_id, goal_data)
        
        elif update.update_type == "needs_state_change":
            # Submission can satisfy certain needs
            needs_data = update.data
            await self._check_submission_need_satisfaction(context.user_id, needs_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with submission awareness"""
        # Analyze input for submission-related content
        submission_analysis = await self._analyze_input_for_submission(context.user_input)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Check if we need to process any submission-related requests
        processing_results = {}
        
        if submission_analysis.get("requests_submission_info"):
            # User is asking about their submission state
            user_state = await self._get_detailed_submission_report(context.user_id)
            processing_results["submission_report"] = user_state
        
        if submission_analysis.get("expresses_submission"):
            # User is expressing submission - record and reward
            await self._process_submission_expression(context.user_id, context.user_input)
            processing_results["submission_acknowledged"] = True
        
        if submission_analysis.get("requests_task") or submission_analysis.get("requests_training"):
            # User wants tasks or training - coordinate with task system
            await self.send_context_update(
                update_type="submission_requests_task",
                data={
                    "user_id": context.user_id,
                    "current_level": await self._get_user_level(context.user_id),
                    "preferred_difficulty": submission_analysis.get("difficulty_preference", "moderate")
                },
                target_modules=["task_assignment_system"],
                scope=ContextScope.TARGETED
            )
            processing_results["task_request_forwarded"] = True
        
        # Update submission context based on interaction
        if submission_analysis.get("submission_relevant"):
            await self.send_context_update(
                update_type="submission_interaction",
                data={
                    "analysis": submission_analysis,
                    "processing_results": processing_results,
                    "cross_module_context": len(messages)
                }
            )
        
        return {
            "submission_processed": True,
            "analysis": submission_analysis,
            "results": processing_results,
            "context_aware": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze submission state in current context"""
        user_id = context.user_id
        
        # Get comprehensive submission analysis
        current_state = await self._get_user_submission_context(user_id)
        
        # Analyze progression trajectory
        trajectory = await self._analyze_progression_trajectory(user_id)
        
        # Get path-specific analysis if on a path
        path_analysis = {}
        if current_state.get("active_path"):
            path_analysis = await self._analyze_path_progress(user_id, current_state["active_path"])
        
        # Cross-reference with other module states
        messages = await self.get_cross_module_messages()
        
        # Analyze coherence with emotional state
        emotional_coherence = await self._analyze_emotional_submission_coherence(
            context.emotional_state, current_state
        )
        
        # Analyze alignment with active goals
        goal_alignment = await self._analyze_goal_submission_alignment(
            context.goal_context, current_state
        )
        
        # Generate recommendations
        recommendations = await self._generate_progression_recommendations(
            current_state, trajectory, messages
        )
        
        return {
            "current_state": current_state,
            "trajectory_analysis": trajectory,
            "path_progress": path_analysis,
            "emotional_coherence": emotional_coherence,
            "goal_alignment": goal_alignment,
            "recommendations": recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize submission-related response elements"""
        user_id = context.user_id
        messages = await self.get_cross_module_messages()
        
        # Get current submission state for response framing
        current_state = await self._get_user_submission_context(user_id)
        
        # Determine appropriate dominance level for response
        dominance_level = self._calculate_response_dominance_level(current_state, context)
        
        # Check if we should reinforce submission
        reinforce_submission = await self._should_reinforce_submission(context, messages)
        
        # Check for opportunities to progress submission
        progression_opportunity = await self._identify_progression_opportunity(context, current_state)
        
        # Generate submission-aware response elements
        synthesis = {
            "dominance_level": dominance_level,
            "reinforce_submission": reinforce_submission,
            "submission_elements": await self._generate_submission_response_elements(
                current_state, dominance_level
            ),
            "progression_opportunity": progression_opportunity,
            "protocol_requirements": await self._get_active_protocols(user_id),
            "reward_suggestions": await self._suggest_submission_rewards(context, current_state)
        }
        
        # Send synthesis update if significant
        if reinforce_submission or progression_opportunity:
            await self.send_context_update(
                update_type="submission_synthesis",
                data=synthesis,
                priority=ContextPriority.HIGH
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_submission(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for submission-related content"""
        input_lower = user_input.lower()
        
        analysis = {
            "submission_relevant": False,
            "expresses_submission": any(phrase in input_lower for phrase in [
                "yes mistress", "yes goddess", "i obey", "i submit", "as you wish",
                "your wish", "i serve", "at your command", "i'm yours"
            ]),
            "requests_submission_info": any(phrase in input_lower for phrase in [
                "my level", "my submission", "my progress", "my training",
                "my obedience", "my score", "my compliance"
            ]),
            "shows_defiance": any(phrase in input_lower for phrase in [
                "i won't", "i refuse", "no way", "not doing", "i can't",
                "don't want to", "make me"
            ]),
            "requests_task": any(phrase in input_lower for phrase in [
                "give me a task", "what should i do", "assign me", "i need a task",
                "something to do", "train me", "test me"
            ]),
            "requests_training": any(phrase in input_lower for phrase in [
                "train me", "teach me", "help me submit", "make me better",
                "improve my", "develop my"
            ]),
            "expresses_struggle": any(phrase in input_lower for phrase in [
                "it's hard", "struggling", "difficult", "can't do", "too much",
                "need help", "failing"
            ])
        }
        
        # Set submission_relevant if any aspect is true
        analysis["submission_relevant"] = any(analysis.values())
        
        # Extract difficulty preference if requesting task
        if analysis["requests_task"]:
            if "easy" in input_lower or "simple" in input_lower:
                analysis["difficulty_preference"] = "easy"
            elif "hard" in input_lower or "difficult" in input_lower or "challenging" in input_lower:
                analysis["difficulty_preference"] = "challenging"
            elif "extreme" in input_lower or "impossible" in input_lower:
                analysis["difficulty_preference"] = "extreme"
            else:
                analysis["difficulty_preference"] = "moderate"
        
        return analysis
    
    async def _get_user_submission_context(self, user_id: str) -> Dict[str, Any]:
        """Get current submission context for user"""
        try:
            # Use the original system's method
            user_data = await self.original_system.get_user_submission_data(
                None,  # ctx parameter 
                user_id, 
                include_history=False
            )
            
            if user_data:
                return {
                    "current_level": user_data.get("submission_level", {}),
                    "submission_score": user_data.get("submission_score", 0.0),
                    "compliance_rate": user_data.get("compliance_rate", 1.0),
                    "metrics": user_data.get("metrics", {}),
                    "active_path": user_data.get("assigned_path"),
                    "recent_compliance": []  # Would need to get from history
                }
        except Exception as e:
            logger.error(f"Error getting submission context: {e}")
        
        return {
            "current_level": {"id": 1, "name": "Curious"},
            "submission_score": 0.0,
            "compliance_rate": 1.0,
            "metrics": {},
            "active_path": None,
            "recent_compliance": []
        }
    
    async def _record_task_compliance(self, user_id: str, task_data: Dict[str, Any], completed: bool):
        """Record task completion as compliance"""
        try:
            task_title = task_data.get("title", "Unknown task")
            difficulty = task_data.get("difficulty", 0.5)
            
            # Convert to float if string
            if isinstance(difficulty, str):
                difficulty_map = {
                    "trivial": 0.1, "easy": 0.3, "moderate": 0.5,
                    "challenging": 0.7, "difficult": 0.85, "extreme": 1.0
                }
                difficulty = difficulty_map.get(difficulty, 0.5)
            
            result = await self.original_system.record_compliance(
                None,  # ctx
                user_id=user_id,
                instruction=f"Complete task: {task_title}",
                complied=completed,
                difficulty=difficulty,
                context_info={"source": "task_system", "task_id": task_data.get("task_id")},
                defiance_reason="Task not completed" if not completed else None
            )
            
            # Send update about compliance record
            if result.get("level_changed"):
                await self.send_context_update(
                    update_type="submission_level_changed",
                    data=result.get("level_change_details", {}),
                    priority=ContextPriority.HIGH
                )
                
        except Exception as e:
            logger.error(f"Error recording task compliance: {e}")
    
    async def _check_milestone_progress_from_task(self, user_id: str, task_data: Dict[str, Any]):
        """Check if task completion progresses milestones"""
        try:
            # Check milestone progress
            result = await self.original_system.check_milestone_progress(None, user_id)
            
            if result.get("newly_completed_milestones"):
                # Send update about milestone completion
                await self.send_context_update(
                    update_type="submission_milestone_completed",
                    data={
                        "user_id": user_id,
                        "milestones": result["newly_completed_milestones"],
                        "from_task": task_data.get("task_id")
                    },
                    priority=ContextPriority.HIGH
                )
                
        except Exception as e:
            logger.error(f"Error checking milestone progress: {e}")
    
    async def _process_emotional_influence_on_submission(self, emotional_data: Dict[str, Any], user_id: str):
        """Process how emotional state affects submission"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Map emotions to submission metric effects
        emotion_effects = {
            "Fear": ("receptiveness", 0.1),
            "Arousal": ("surrender", 0.15),
            "Shame": ("depth", 0.1),
            "Pride": ("reverence", -0.1),  # Pride reduces reverence
            "Frustration": ("consistency", -0.1),
            "Love": ("reverence", 0.2),
            "Submission": ("surrender", 0.2)
        }
        
        if emotion_name in emotion_effects:
            metric_name, base_change = emotion_effects[emotion_name]
            change = base_change * intensity
            
            try:
                await self.original_system.update_submission_metric(
                    None, user_id, metric_name, change,
                    reason=f"emotional_influence_{emotion_name.lower()}"
                )
            except Exception as e:
                logger.error(f"Error updating submission metric from emotion: {e}")
    
    async def _process_dominance_interaction(self, user_id: str, interaction_type: str, intensity: float):
        """Process dominance interaction effects on submission"""
        # Map interaction types to metric changes
        interaction_effects = {
            "command": ("obedience", 0.05 * intensity),
            "control": ("surrender", 0.08 * intensity),
            "humiliation": ("depth", 0.1 * intensity),
            "praise": ("reverence", 0.07 * intensity),
            "punishment": ("receptiveness", 0.06 * intensity),
            "training": ("consistency", 0.05 * intensity)
        }
        
        if interaction_type in interaction_effects:
            metric_name, change = interaction_effects[interaction_type]
            
            try:
                await self.original_system.update_submission_metric(
                    None, user_id, metric_name, change,
                    reason=f"dominance_{interaction_type}"
                )
                
                # Send update about dominance processing
                await self.send_context_update(
                    update_type="submission_dominance_processed",
                    data={
                        "interaction_type": interaction_type,
                        "intensity": intensity,
                        "metric_affected": metric_name,
                        "change": change
                    }
                )
            except Exception as e:
                logger.error(f"Error processing dominance interaction: {e}")
    
    async def _process_protocol_violation(self, user_id: str, severity: float, violation_data: Dict[str, Any]):
        """Process protocol violation effects"""
        try:
            # Record as non-compliance
            await self.original_system.record_compliance(
                None,
                user_id=user_id,
                instruction=f"Follow protocol: {violation_data.get('protocol', 'Unknown')}",
                complied=False,
                difficulty=0.3,  # Protocols should be easy to follow
                context_info={"violation_type": "protocol", "severity": severity},
                defiance_reason=violation_data.get("reason", "Protocol violation")
            )
            
            # Additional metric penalty for protocol violations
            await self.original_system.update_submission_metric(
                None, user_id, "protocol_adherence", -0.15 * severity,
                reason="protocol_violation"
            )
            
        except Exception as e:
            logger.error(f"Error processing protocol violation: {e}")
    
    async def _get_user_level(self, user_id: str) -> int:
        """Get user's current submission level"""
        context = await self._get_user_submission_context(user_id)
        return context.get("current_level", {}).get("id", 1)
    
    async def _process_submission_expression(self, user_id: str, user_input: str):
        """Process when user expresses submission"""
        try:
            # Small metric boosts for verbal submission
            await self.original_system.update_submission_metric(
                None, user_id, "reverence", 0.05,
                reason="verbal_submission_expression"
            )
            
            await self.original_system.update_submission_metric(
                None, user_id, "initiative", 0.03,
                reason="spontaneous_submission"
            )
            
            # Send reward signal
            await self.send_context_update(
                update_type="submission_expression_reward",
                data={
                    "user_id": user_id,
                    "expression": user_input[:50],
                    "reward_value": 0.3
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing submission expression: {e}")
    
    async def _get_detailed_submission_report(self, user_id: str) -> Dict[str, Any]:
        """Get detailed submission report for user"""
        try:
            report = await self.original_system.generate_progression_report(None, user_id)
            return report
        except Exception as e:
            logger.error(f"Error getting submission report: {e}")
            return {}
    
    async def _analyze_progression_trajectory(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's progression trajectory"""
        try:
            # Get current data
            current = await self._get_user_submission_context(user_id)
            
            # Simple trajectory analysis
            trajectory = "stable"
            momentum = 0.0
            
            compliance_rate = current.get("compliance_rate", 1.0)
            if compliance_rate > 0.8:
                trajectory = "ascending"
                momentum = 0.2
            elif compliance_rate < 0.5:
                trajectory = "declining"  
                momentum = -0.2
                
            return {
                "trajectory": trajectory,
                "momentum": momentum,
                "compliance_trend": compliance_rate,
                "recommendation": "maintain" if trajectory == "stable" else 
                                "encourage" if trajectory == "ascending" else "intervene"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trajectory: {e}")
            return {"trajectory": "unknown", "momentum": 0.0}
    
    async def _analyze_path_progress(self, user_id: str, path_id: str) -> Dict[str, Any]:
        """Analyze progress on assigned path"""
        try:
            progress = await self.original_system.check_milestone_progress(None, user_id)
            return progress
        except Exception as e:
            logger.error(f"Error analyzing path progress: {e}")
            return {}
    
    async def _check_path_unlocks(self, user_id: str, relationship_data: Dict[str, Any]):
        """Check if relationship milestone unlocks new paths"""
        trust_level = relationship_data.get("trust", 0.5)
        intimacy_level = relationship_data.get("intimacy", 0.5)
        
        # High trust/intimacy might unlock more intense paths
        if trust_level > 0.8 and intimacy_level > 0.7:
            await self.send_context_update(
                update_type="submission_paths_available",
                data={
                    "user_id": user_id,
                    "unlocked_paths": ["psychological", "strict_discipline"],
                    "reason": "high_trust_intimacy"
                }
            )
    
    def _calculate_response_dominance_level(self, current_state: Dict[str, Any], 
                                          context: SharedContext) -> float:
        """Calculate appropriate dominance level for response"""
        base_level = 0.3  # Base dominance
        
        # Scale by submission level
        level_id = current_state.get("current_level", {}).get("id", 1)
        base_level += (level_id - 1) * 0.15
        
        # Adjust for emotional state
        if context.emotional_state:
            if "Fear" in context.emotional_state and context.emotional_state["Fear"] > 0.5:
                base_level -= 0.1  # Reduce dominance if user is fearful
            if "Arousal" in context.emotional_state and context.emotional_state["Arousal"] > 0.5:
                base_level += 0.1  # Increase if aroused
                
        return max(0.1, min(1.0, base_level))
    
    async def _should_reinforce_submission(self, context: SharedContext, 
                                         messages: Dict[str, List[Dict]]) -> bool:
        """Determine if submission should be reinforced in response"""
        # Check if recent interactions suggest reinforcement needed
        for module_messages in messages.values():
            for msg in module_messages:
                if msg['type'] in ['protocol_violation', 'defiance_detected', 'compliance_failure']:
                    return True
                    
        # Reinforce if user expressed struggle
        submission_analysis = await self._analyze_input_for_submission(context.user_input)
        if submission_analysis.get("expresses_struggle"):
            return True
            
        return False
    
    async def _generate_submission_response_elements(self, current_state: Dict[str, Any], 
                                                   dominance_level: float) -> Dict[str, Any]:
        """Generate submission-aware response elements"""
        elements = {
            "use_titles": dominance_level > 0.3,
            "enforce_protocols": dominance_level > 0.5,
            "dominance_markers": [],
            "submission_reinforcement": []
        }
        
        # Add dominance markers based on level
        if dominance_level > 0.7:
            elements["dominance_markers"] = ["strict_tone", "commands", "expectations"]
            elements["submission_reinforcement"] = ["praise_obedience", "reinforce_position"]
        elif dominance_level > 0.4:
            elements["dominance_markers"] = ["firm_tone", "guidance", "structure"]
            elements["submission_reinforcement"] = ["encourage_compliance", "gentle_reminder"]
        else:
            elements["dominance_markers"] = ["supportive_tone", "suggestions"]
            elements["submission_reinforcement"] = ["positive_encouragement"]
            
        return elements
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
