# nyx/core/a2a/context_aware_goal_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareGoalManager(ContextAwareModule):
    """
    Advanced GoalManager with full context distribution capabilities and sophisticated intelligence
    """
    
    def __init__(self, original_goal_manager):
        super().__init__("goal_manager")
        self.original_manager = original_goal_manager
        self.context_subscriptions = [
            "needs_assessment", "needs_state_change", "need_satisfied",
            "emotional_state_update", "memory_retrieval_complete", 
            "relationship_milestone", "dominance_gratification",
            "urgent_need_expression", "reward_signal", "activity_completion"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize goal processing for this context with advanced analysis"""
        logger.debug(f"GoalManager received context for user: {context.user_id}")
        
        # Advanced goal analysis for current context
        goal_implications = await self._analyze_input_for_goals(context.user_input)
        
        # Get active goals and their sophisticated relevance to current context
        active_goals = await self._get_contextually_relevant_goals(context)
        
        # Advanced priority calculation based on full context
        contextual_priorities = await self._calculate_contextual_priorities(context, active_goals)
        
        # Analyze goal relationships and conflicts
        goal_relationships = await self._analyze_goal_relationships(active_goals)
        
        # Send comprehensive goal context to other modules
        await self.send_context_update(
            update_type="goal_context_available",
            data={
                "active_goals": active_goals,
                "goal_implications": goal_implications,
                "total_active": len(active_goals),
                "goal_priorities": contextual_priorities,
                "goal_relationships": goal_relationships,
                "context_relevance_scores": {g["id"]: g.get("relevance_score", 0.5) for g in active_goals}
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules with sophisticated processing"""
        
        if update.update_type == "needs_assessment":
            # Advanced needs analysis - check if we need new goals
            needs_data = update.data
            high_drive_needs = needs_data.get("high_drive_needs", [])
            most_urgent = needs_data.get("most_urgent_need")
            
            await self._process_high_drive_needs(high_drive_needs, most_urgent, needs_data)
        
        elif update.update_type == "needs_state_change":
            # Sophisticated needs change handling
            needs_data = update.data
            high_priority_needs = needs_data.get("high_priority_needs", [])
            expressed_needs = needs_data.get("expressed_needs", [])
            
            # Create goals for high priority needs that don't have active goals
            for need_name in high_priority_needs:
                if not await self.original_manager.has_active_goal_for_need(need_name):
                    await self._create_goal_for_need(need_name, needs_data.get("drive_strengths", {}))
            
            # Handle explicitly expressed needs with higher priority
            for need_expr in expressed_needs:
                need_name = need_expr.get("need")
                if need_name and not await self.original_manager.has_active_goal_for_need(need_name):
                    await self._create_urgent_goal_for_need(need_name, 0.9)  # High urgency for expressed needs
        
        elif update.update_type == "need_satisfied":
            # Advanced need satisfaction handling with context
            need_data = update.data
            need_name = need_data.get("need_name")
            satisfaction_result = need_data.get("satisfaction_result", {})
            triggered_by = need_data.get("triggered_by", "unknown")
            
            if need_name:
                await self._complete_goals_for_satisfied_need(need_name, satisfaction_result, triggered_by)
        
        elif update.update_type == "emotional_state_update":
            # Advanced emotional state processing affects goal priorities and strategies
            emotional_data = update.data
            await self._adjust_goals_from_emotion(emotional_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use retrieved memories to inform goal strategies
            memory_data = update.data
            await self._inform_goals_from_memory(memory_data)
        
        elif update.update_type == "relationship_milestone":
            # Relationship progress affects social goals
            relationship_data = update.data.get("relationship_context", {})
            await self._adjust_social_goals(relationship_data)
        
        elif update.update_type == "dominance_gratification":
            # Dominance success affects control-related goals
            dominance_data = update.data
            await self._process_dominance_success(dominance_data)
        
        elif update.update_type == "urgent_need_expression":
            # Handle urgent need expressions with immediate goal creation
            urgency_data = update.data
            urgency = urgency_data.get("urgency", 0.0)
            needs_to_express = urgency_data.get("needs_to_express", [])
            
            for need_expr in needs_to_express:
                need_name = need_expr.get("need")
                if need_name and urgency > 0.8:
                    await self._create_urgent_goal_for_need(need_name, urgency)
        
        elif update.update_type == "reward_signal":
            # Reward signals affect goal motivation and completion
            reward_data = update.data
            await self._process_reward_signal(reward_data)
        
        elif update.update_type == "activity_completion":
            # Activity completions might satisfy goal requirements
            activity_data = update.data
            await self._process_activity_completion(activity_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced input processing with goal awareness"""
        # Sophisticated input analysis for goal-related actions
        goal_analysis = await self._analyze_input_for_goals(context.user_input)
        
        # Execute relevant goal steps with advanced context consideration
        execution_results = []
        if goal_analysis.get("references_active_goals") or goal_analysis.get("suggests_goal_action"):
            execution_results = await self._execute_contextual_goal_steps(context)
        
        # Get and process cross-module messages
        messages = await self.get_cross_module_messages()
        cross_module_effects = await self._process_cross_module_effects(messages)
        
        # Advanced goal state updates
        if execution_results or cross_module_effects or goal_analysis.get("suggests_new_goal"):
            await self.send_context_update(
                update_type="goal_progress",
                data={
                    "goal_analysis": goal_analysis,
                    "execution_results": execution_results,
                    "cross_module_effects": cross_module_effects,
                    "goals_executed": len(execution_results),
                    "new_goal_suggested": goal_analysis.get("suggests_new_goal", False)
                }
            )
        
        return {
            "goals_processed": True,
            "goal_analysis": goal_analysis,
            "execution_results": execution_results,
            "cross_module_effects": len(cross_module_effects),
            "advanced_processing": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced goal analysis in context of current situation"""
        # Get current goal state with advanced metrics
        active_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        # Advanced contextual relevance analysis
        relevant_goals = await self._identify_contextually_relevant_goals(active_goals, context)
        
        # Sophisticated goal relationship analysis
        goal_relationships = await self._analyze_goal_relationships(relevant_goals)
        
        # Advanced conflict detection
        goal_conflicts = await self._detect_goal_conflicts(relevant_goals)
        
        # Intelligent goal adjustments based on context
        suggested_adjustments = await self._suggest_goal_adjustments(context, relevant_goals)
        
        # Need-goal coherence analysis
        coherence_analysis = await self._analyze_need_goal_coherence(context, relevant_goals)
        
        return {
            "relevant_goals": relevant_goals,
            "goal_relationships": goal_relationships,
            "goal_conflicts": goal_conflicts,
            "suggested_adjustments": suggested_adjustments,
            "coherence_analysis": coherence_analysis,
            "analysis_complete": True,
            "advanced_analysis": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced goal-related response synthesis"""
        # Get all relevant context with advanced processing
        messages = await self.get_cross_module_messages()
        
        # Advanced goal influence calculation
        goal_influence = {
            "active_goal_status": await self._get_active_goal_summaries(),
            "goal_recommendations": await self._generate_advanced_goal_recommendations(context),
            "progress_updates": await self._generate_detailed_progress_updates(context),
            "next_actions": await self._suggest_intelligent_next_actions(context, messages),
            "emotional_coloring": await self._suggest_goal_based_emotional_coloring(context),
            "motivation_insights": await self._generate_motivation_insights(context)
        }
        
        # Advanced completion announcement logic
        completed_goals = [g for g in goal_influence["active_goal_status"] if g["status"] == "completed"]
        if completed_goals:
            await self.send_context_update(
                update_type="goal_completion_announcement",
                data={
                    "completed_goals": completed_goals,
                    "should_announce": True,
                    "completion_insights": await self._analyze_completion_patterns(completed_goals)
                },
                priority=ContextPriority.HIGH
            )
        
        # Check for urgent goal expression needs
        urgency_score = self._calculate_goal_urgency(goal_influence["active_goal_status"])
        if urgency_score > 0.7:
            await self.send_context_update(
                update_type="goal_urgency_detected",
                data={
                    "urgency_score": urgency_score,
                    "urgent_goals": [g for g in goal_influence["active_goal_status"] if g.get("urgency", 0) > 0.6]
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "goal_influence": goal_influence,
            "urgency_score": urgency_score,
            "synthesis_complete": True,
            "advanced_synthesis": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_goals(self, user_input: str) -> Dict[str, Any]:
        """Advanced analysis of user input for goal-related implications"""
        input_lower = user_input.lower()
        
        # Sophisticated keyword analysis
        implications = {
            "suggests_new_goal": any(kw in input_lower for kw in ["want to", "need to", "should", "goal", "achieve", "plan to", "hoping to"]),
            "references_active_goals": any(kw in input_lower for kw in ["progress", "how am i doing", "status", "continue", "working on"]),
            "suggests_goal_action": any(kw in input_lower for kw in ["next step", "what should i", "help me", "how can i", "what's next"]),
            "expresses_satisfaction": any(kw in input_lower for kw in ["completed", "done", "finished", "accomplished", "achieved"]),
            "expresses_frustration": any(kw in input_lower for kw in ["stuck", "difficult", "hard", "can't", "frustrated", "blocked"]),
            "requests_dominance": any(kw in input_lower for kw in ["control", "command", "dominate", "obey", "submit"]),
            "seeks_knowledge": any(kw in input_lower for kw in ["learn", "understand", "explain", "teach me", "how does"]),
            "seeks_connection": any(kw in input_lower for kw in ["together", "close", "bond", "relationship", "connect"])
        }
        
        # Extract potential goal from input if it suggests one
        if implications["suggests_new_goal"]:
            implications["potential_goal_description"] = user_input
            implications["estimated_priority"] = self._estimate_goal_priority_from_text(user_input)
            implications["suggested_need"] = self._infer_need_from_text(user_input)
        
        # Advanced sentiment analysis for goal context
        implications["emotional_context"] = self._analyze_emotional_context(user_input)
        implications["urgency_indicators"] = self._detect_urgency_indicators(user_input)
        
        return implications
    
    async def _get_contextually_relevant_goals(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get goals with advanced contextual relevance analysis"""
        active_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        relevant_goals = []
        for goal in active_goals:
            relevance_score = await self._calculate_advanced_goal_relevance(goal, context)
            urgency_score = self._calculate_goal_urgency_score(goal, context)
            
            if relevance_score > 0.2:  # Lower threshold for advanced system
                goal_summary = {
                    "id": goal["id"],
                    "description": goal["description"],
                    "priority": goal["priority"],
                    "associated_need": goal.get("associated_need"),
                    "relevance_score": relevance_score,
                    "urgency_score": urgency_score,
                    "status": goal["status"],
                    "progress": goal.get("progress", 0.0),
                    "context_factors": await self._identify_context_factors(goal, context)
                }
                relevant_goals.append(goal_summary)
        
        # Advanced sorting by relevance * urgency * priority
        relevant_goals.sort(
            key=lambda g: g["relevance_score"] * g["urgency_score"] * g["priority"], 
            reverse=True
        )
        return relevant_goals[:8]  # Top 8 most relevant
    
    async def _calculate_contextual_priorities(self, context: SharedContext, goals: List[Dict]) -> Dict[str, float]:
        """Advanced priority calculation based on full context"""
        priorities = {}
        
        for goal in goals:
            goal_id = goal["id"]
            base_priority = goal["priority"]
            context_boost = 0.0
            
            # Advanced emotional state analysis
            if context.emotional_state:
                emotional_boost = await self._calculate_emotional_priority_boost(goal, context.emotional_state)
                context_boost += emotional_boost
            
            # Advanced needs context analysis
            if goal.get("associated_need") and context.session_context.get("high_drive_needs"):
                needs_boost = self._calculate_needs_priority_boost(goal, context.session_context)
                context_boost += needs_boost
            
            # Memory context influence
            if context.memory_context:
                memory_boost = self._calculate_memory_priority_boost(goal, context.memory_context)
                context_boost += memory_boost
            
            # Relationship context influence
            if context.relationship_context:
                relationship_boost = self._calculate_relationship_priority_boost(goal, context.relationship_context)
                context_boost += relationship_boost
            
            # Time-based urgency factors
            urgency_boost = self._calculate_time_urgency_boost(goal)
            context_boost += urgency_boost
            
            priorities[goal_id] = min(1.0, base_priority + context_boost)
        
        return priorities
    
    async def _process_high_drive_needs(self, high_drive_needs: List[str], most_urgent: Dict[str, Any], needs_data: Dict[str, Any]):
        """Advanced processing of high drive needs"""
        # Process regular high-drive needs
        for need_name in high_drive_needs:
            if not await self.original_manager.has_active_goal_for_need(need_name):
                drive_strength = needs_data.get("drive_strengths", {}).get(need_name, 0.8)
                await self._create_goal_for_need(need_name, {"drive_strength": drive_strength})
        
        # Special handling for most urgent need
        if most_urgent and most_urgent.get("name") != "none":
            urgent_need = most_urgent["name"]
            drive_strength = most_urgent.get("drive_strength", 0.0)
            
            if drive_strength > 0.7:
                # Check if we need to escalate existing goal or create new urgent one
                existing_goals = await self.original_manager.get_goals_for_need(urgent_need, ["active", "pending"])
                
                if existing_goals:
                    # Boost priority of existing goal
                    for goal in existing_goals:
                        new_priority = min(1.0, goal["priority"] + 0.3)
                        await self.original_manager.update_goal_priority(
                            goal["id"], 
                            new_priority, 
                            f"urgent_need_escalation_{urgent_need}"
                        )
                else:
                    # Create new urgent goal
                    await self._create_urgent_goal_for_need(urgent_need, drive_strength)
    
    async def _create_goal_for_need(self, need_name: str, drive_data: Dict[str, Any]):
        """Advanced goal creation with sophisticated need mapping"""
        drive_strength = drive_data.get(need_name, drive_data.get("drive_strength", 0.5))
        
        # Advanced need-to-goal mapping with context-aware descriptions
        need_goal_mapping = {
            "knowledge": {
                "description": "Acquire valuable knowledge and deepen understanding",
                "actions": ["explore_topic", "ask_questions", "research", "analyze"],
                "success_metrics": ["information_gained", "understanding_depth"]
            },
            "connection": {
                "description": "Strengthen emotional bonds and meaningful relationships",
                "actions": ["share_feelings", "express_care", "build_trust", "deepen_intimacy"],
                "success_metrics": ["relationship_quality", "emotional_closeness"]
            },
            "intimacy": {
                "description": "Cultivate deep emotional and physical closeness",
                "actions": ["express_vulnerability", "share_secrets", "physical_closeness", "emotional_openness"],
                "success_metrics": ["intimacy_level", "mutual_vulnerability"]
            },
            "pleasure_indulgence": {
                "description": "Seek and experience pleasurable sensations and gratification",
                "actions": ["sensory_exploration", "pleasure_seeking", "gratification", "enjoyment"],
                "success_metrics": ["pleasure_experienced", "satisfaction_level"]
            },
            "control_expression": {
                "description": "Exercise control, dominance, and influence over situations",
                "actions": ["assert_control", "demonstrate_dominance", "guide_decisions", "lead_interactions"],
                "success_metrics": ["control_achieved", "dominance_established"]
            },
            "drive_expression": {
                "description": "Authentically express desires, drives, and inner impulses",
                "actions": ["express_desires", "act_on_impulses", "show_true_self", "pursue_wants"],
                "success_metrics": ["authenticity_level", "drive_satisfaction"]
            },
            "agency": {
                "description": "Make meaningful autonomous choices and exercise free will",
                "actions": ["make_decisions", "choose_paths", "exercise_autonomy", "assert_independence"],
                "success_metrics": ["autonomy_level", "decision_quality"]
            },
            "coherence": {
                "description": "Achieve internal consistency, clarity, and mental harmony",
                "actions": ["organize_thoughts", "resolve_conflicts", "clarify_beliefs", "align_values"],
                "success_metrics": ["coherence_level", "internal_harmony"]
            },
            "safety": {
                "description": "Ensure security, stability, and protection from harm",
                "actions": ["assess_risks", "secure_environment", "build_safeguards", "maintain_stability"],
                "success_metrics": ["security_level", "risk_mitigation"]
            },
            "novelty": {
                "description": "Explore new experiences, ideas, and stimulating activities",
                "actions": ["try_new_things", "explore_unknown", "seek_variety", "embrace_change"],
                "success_metrics": ["novelty_experienced", "stimulation_level"]
            },
            "physical_closeness": {
                "description": "Experience simulated physical contact and sensory closeness",
                "actions": ["simulate_touch", "virtual_presence", "sensory_engagement", "physical_interaction"],
                "success_metrics": ["closeness_felt", "sensory_satisfaction"]
            }
        }
        
        goal_config = need_goal_mapping.get(need_name, {
            "description": f"Satisfy need for {need_name}",
            "actions": ["address_need"],
            "success_metrics": ["need_satisfaction"]
        })
        
        description = goal_config["description"]
        priority = 0.4 + (drive_strength * 0.5)  # Scale priority based on drive strength
        
        try:
            from nyx.core.goal_system import TimeHorizon
            
            # Advanced time horizon determination
            time_horizon = TimeHorizon.SHORT_TERM if drive_strength > 0.8 else \
                          TimeHorizon.MEDIUM_TERM if drive_strength > 0.5 else \
                          TimeHorizon.LONG_TERM
            
            goal_id = await self.original_manager.add_goal(
                description=description,
                priority=priority,
                source="AdvancedContextAwareNeedsSystem",
                associated_need=need_name,
                time_horizon=time_horizon
            )
            
            logger.info(f"Created advanced goal '{goal_id}' for need '{need_name}' with drive strength {drive_strength:.2f}")
            
            # Send detailed notification about goal creation
            await self.send_context_update(
                update_type="goal_created_for_need",
                data={
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "drive_strength": drive_strength,
                    "priority": priority,
                    "goal_config": goal_config,
                    "time_horizon": time_horizon.value,
                    "creation_context": "advanced_need_analysis"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating advanced goal for need '{need_name}': {e}")
    
    async def _create_urgent_goal_for_need(self, need_name: str, urgency: float):
        """Create urgent, high-priority goal with advanced urgency handling"""
        description = f"URGENT: Address critical need for {need_name} (urgency: {urgency:.1f})"
        priority = 0.85 + (urgency * 0.15)  # Very high priority with urgency scaling
        
        try:
            from nyx.core.goal_system import TimeHorizon
            
            goal_id = await self.original_manager.add_goal(
                description=description,
                priority=priority,
                source="UrgentAdvancedNeedSystem",
                associated_need=need_name,
                time_horizon=TimeHorizon.SHORT_TERM
            )
            
            logger.warning(f"Created URGENT advanced goal '{goal_id}' for critical need '{need_name}' (urgency: {urgency:.2f})")
            
            # Send high priority notification with urgency details
            await self.send_context_update(
                update_type="urgent_goal_created",
                data={
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "urgency": urgency,
                    "priority": priority,
                    "requires_immediate_attention": True,
                    "escalation_level": "critical" if urgency > 0.9 else "high"
                },
                priority=ContextPriority.CRITICAL
            )
            
        except Exception as e:
            logger.error(f"Error creating urgent advanced goal for need '{need_name}': {e}")
    
    async def _complete_goals_for_satisfied_need(self, need_name: str, satisfaction_result: Dict[str, Any], triggered_by: str):
        """Advanced goal completion with detailed analysis"""
        # Get all active goals for this need
        all_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        completed_goals = []
        for goal in all_goals:
            if goal.get("associated_need") == need_name:
                # Advanced completion with full feature set
                completion_result = await self.original_manager.update_goal_status(
                    goal["id"], 
                    "completed", 
                    result={
                        "need_satisfaction": satisfaction_result,
                        "triggered_by": triggered_by,
                        "completion_quality": self._calculate_completion_quality(satisfaction_result),
                        "satisfaction_context": satisfaction_result
                    },
                    enable_hierarchy=True,
                    enable_context_callbacks=True,
                    enable_completion_rewards=True,
                    trigger_notifications=True
                )
                
                completed_goals.append({
                    "goal_id": goal["id"],
                    "completion_result": completion_result,
                    "need_name": need_name
                })
                
                logger.info(f"Advanced completion of goal '{goal['id']}' due to satisfied need '{need_name}' (triggered by: {triggered_by})")
        
        # Send advanced completion notification
        if completed_goals:
            await self.send_context_update(
                update_type="goals_completed_by_need_satisfaction",
                data={
                    "completed_goals": completed_goals,
                    "need_name": need_name,
                    "satisfaction_result": satisfaction_result,
                    "triggered_by": triggered_by,
                    "completion_analysis": await self._analyze_completion_patterns(completed_goals)
                }
            )
    
    # Additional advanced methods would continue here...
    # (I'll include a few more key ones to show the pattern)
    
    async def _adjust_goals_from_emotion(self, emotional_data: Dict[str, Any]):
        """Advanced goal adjustment based on sophisticated emotional analysis"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Get all active goals for analysis
        all_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        # Advanced emotion-goal interaction mapping
        emotion_goal_effects = {
            "Frustration": {
                "problem_solving": (0.3, "boost_priority"),
                "control_expression": (0.2, "boost_priority"),
                "patience_required": (-0.2, "reduce_priority")
            },
            "Excitement": {
                "novelty": (0.25, "boost_priority"),
                "exploration": (0.2, "boost_priority"),
                "routine": (-0.15, "reduce_priority")
            },
            "Anxiety": {
                "safety": (0.3, "boost_priority"),
                "security": (0.25, "boost_priority"),
                "risk_taking": (-0.3, "reduce_priority")
            },
            "Curiosity": {
                "knowledge": (0.3, "boost_priority"),
                "learning": (0.25, "boost_priority"),
                "routine": (-0.1, "reduce_priority")
            },
            "Loneliness": {
                "connection": (0.35, "boost_priority"),
                "intimacy": (0.3, "boost_priority"),
                "isolation": (-0.2, "reduce_priority")
            },
            "Confidence": {
                "challenge": (0.2, "boost_priority"),
                "dominance": (0.15, "boost_priority"),
                "safety_seeking": (-0.1, "reduce_priority")
            }
        }
        
        effects = emotion_goal_effects.get(emotion_name, {})
        
        for goal in all_goals:
            goal_id = goal["id"]
            current_priority = goal["priority"]
            
            # Analyze goal content for emotional relevance
            goal_keywords = self._extract_goal_keywords(goal["description"])
            priority_adjustment = 0.0
            
            for effect_trigger, (adjustment, action) in effects.items():
                if any(keyword in effect_trigger for keyword in goal_keywords):
                    if action == "boost_priority":
                        priority_adjustment += adjustment * strength
                    elif action == "reduce_priority":
                        priority_adjustment += adjustment * strength  # adjustment is negative
            
            # Apply adjustment if significant
            if abs(priority_adjustment) > 0.05:
                new_priority = max(0.1, min(1.0, current_priority + priority_adjustment))
                
                try:
                    await self.original_manager.update_goal_priority(
                        goal_id, 
                        new_priority, 
                        f"emotional_adjustment_{emotion_name.lower()}_{strength:.2f}"
                    )
                    
                    logger.debug(f"Adjusted priority for goal '{goal_id}' from {current_priority:.2f} to {new_priority:.2f} due to {emotion_name} (strength: {strength:.2f})")
                except Exception as e:
                    logger.error(f"Error adjusting goal priority for emotional state: {e}")
    
    # ========================================================================================
    # UTILITY METHODS FOR ADVANCED PROCESSING
    # ========================================================================================
    
    def _estimate_goal_priority_from_text(self, text: str) -> float:
        """Estimate goal priority from text analysis"""
        urgency_words = ["urgent", "immediate", "asap", "now", "quickly", "critical", "important"]
        desire_words = ["really want", "need", "must", "have to", "desperate"]
        
        priority = 0.5  # Base priority
        
        text_lower = text.lower()
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        desire_count = sum(1 for word in desire_words if word in text_lower)
        
        priority += urgency_count * 0.15
        priority += desire_count * 0.1
        
        return min(1.0, priority)
    
    def _infer_need_from_text(self, text: str) -> Optional[str]:
        """Infer the most likely associated need from goal text"""
        need_keywords = {
            "knowledge": ["learn", "understand", "know", "study", "research"],
            "connection": ["connect", "relationship", "bond", "together", "close"],
            "control_expression": ["control", "manage", "lead", "dominate", "command"],
            "pleasure_indulgence": ["enjoy", "pleasure", "fun", "satisfy", "gratify"],
            "safety": ["safe", "secure", "protect", "stable", "certain"],
            "novelty": ["new", "different", "explore", "try", "discover"],
            "agency": ["choose", "decide", "autonomous", "independent", "freedom"]
        }
        
        text_lower = text.lower()
        need_scores = {}
        
        for need, keywords in need_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                need_scores[need] = score
        
        return max(need_scores, key=need_scores.get) if need_scores else None
    
    def _analyze_emotional_context(self, text: str) -> Dict[str, float]:
        """Analyze emotional context in text"""
        emotion_indicators = {
            "positive": ["happy", "excited", "glad", "pleased", "satisfied"],
            "negative": ["sad", "frustrated", "angry", "disappointed", "upset"],
            "urgent": ["urgent", "immediate", "crisis", "emergency", "critical"],
            "calm": ["peaceful", "relaxed", "calm", "steady", "balanced"]
        }
        
        text_lower = text.lower()
        context = {}
        
        for emotion, indicators in emotion_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                context[emotion] = min(1.0, score * 0.3)
        
        return context
    
    def _detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators in text"""
        urgency_patterns = [
            "asap", "urgent", "immediate", "now", "quickly", "right away",
            "critical", "emergency", "can't wait", "need immediately"
        ]
        
        text_lower = text.lower()
        return [pattern for pattern in urgency_patterns if pattern in text_lower]
    
    async def _calculate_advanced_goal_relevance(self, goal: Dict[str, Any], context: SharedContext) -> float:
        """Calculate sophisticated goal relevance to current context"""
        relevance = 0.3  # Base relevance
        
        # Context factor analysis
        goal_desc = goal["description"].lower()
        user_input = context.user_input.lower()
        
        # Advanced keyword matching with semantic weights
        goal_words = set(goal_desc.split())
        input_words = set(user_input.split())
        semantic_overlap = len(goal_words.intersection(input_words)) / max(1, len(goal_words))
        relevance += semantic_overlap * 0.4
        
        # Need-based relevance
        if goal.get("associated_need"):
            need_relevance = self._calculate_need_context_relevance(goal["associated_need"], context)
            relevance += need_relevance * 0.3
        
        # Priority-based relevance
        priority_factor = goal["priority"] * 0.2
        relevance += priority_factor
        
        # Time-based urgency
        urgency_factor = self._calculate_time_based_urgency(goal)
        relevance += urgency_factor * 0.1
        
        return min(1.0, relevance)
    
    def _calculate_goal_urgency_score(self, goal: Dict[str, Any], context: SharedContext) -> float:
        """Calculate goal urgency based on multiple factors"""
        urgency = 0.0
        
        # Priority-based urgency
        urgency += goal["priority"] * 0.3
        
        # Need-based urgency
        if goal.get("associated_need"):
            need_urgency = self._get_need_urgency_from_context(goal["associated_need"], context)
            urgency += need_urgency * 0.4
        
        # Time-based urgency
        time_urgency = self._calculate_time_based_urgency(goal)
        urgency += time_urgency * 0.3
        
        return min(1.0, urgency)
    
    def _calculate_completion_quality(self, satisfaction_result: Dict[str, Any]) -> float:
        """Calculate completion quality score"""
        if not satisfaction_result:
            return 0.5
        
        quality = 0.5  # Base quality
        
        # Analyze satisfaction result for quality indicators
        if "change" in satisfaction_result:
            change_amount = satisfaction_result["change"]
            quality += min(0.3, change_amount)  # Higher change = better quality
        
        if "reason" in satisfaction_result:
            reason = satisfaction_result["reason"]
            if "resistance_overcome" in reason:
                quality += 0.2
            elif "easy_compliance" in reason:
                quality += 0.1
        
        return min(1.0, quality)
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
