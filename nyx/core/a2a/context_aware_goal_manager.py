# nyx/core/a2a/context_aware_goal_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority
from nyx.core.goal_system import GoalManager, Goal, TimeHorizon

logger = logging.getLogger(__name__)

class ContextAwareGoalManager(ContextAwareModule):
    """
    Enhanced GoalManager with context distribution capabilities
    """
    
    def __init__(self, original_goal_manager: GoalManager):
        super().__init__("goal_manager")
        self.original_manager = original_goal_manager
        self.context_subscriptions = [
            "needs_assessment", "needs_state_change", "need_satisfied",
            "emotional_state_update", "memory_retrieval_complete", 
            "relationship_milestone", "dominance_gratification",
            "urgent_need_expression"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize goal processing for this context"""
        logger.debug(f"GoalManager received context for user: {context.user_id}")
        
        # Check for goal-relevant content in user input
        goal_implications = await self._analyze_input_for_goals(context.user_input)
        
        # Get active goals and their relevance to current context
        active_goals = await self._get_contextually_relevant_goals(context)
        
        # Send goal context to other modules
        await self.send_context_update(
            update_type="goal_context_available",
            data={
                "active_goals": active_goals,
                "goal_implications": goal_implications,
                "total_active": len(active_goals),
                "goal_priorities": await self._calculate_contextual_priorities(context, active_goals)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect goals"""
        
        if update.update_type == "needs_assessment":
            # Initial needs assessment - check if we need new goals
            needs_data = update.data
            high_drive_needs = needs_data.get("high_drive_needs", [])
            most_urgent = needs_data.get("most_urgent_need")
            
            await self._process_high_drive_needs(high_drive_needs, most_urgent)
        
        elif update.update_type == "needs_state_change":
            # Needs have changed - update goal priorities and create new goals if needed
            needs_data = update.data
            high_priority_needs = needs_data.get("high_priority_needs", [])
            
            # Create goals for high priority needs that don't have active goals
            for need_name in high_priority_needs:
                if not await self.original_manager.has_active_goal_for_need(need_name):
                    await self._create_goal_for_need(need_name, needs_data.get("drive_strengths", {}))
        
        elif update.update_type == "need_satisfied":
            # A need was satisfied - complete related goals
            need_data = update.data
            need_name = need_data.get("need_name")
            satisfaction_result = need_data.get("satisfaction_result", {})
            
            if need_name:
                await self._complete_goals_for_satisfied_need(need_name, satisfaction_result)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects goal motivation and priorities
            emotional_data = update.data
            await self._adjust_goals_from_emotion(emotional_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Past experiences can inform goal strategies
            memory_data = update.data
            await self._inform_goals_from_memory(memory_data)
        
        elif update.update_type == "urgent_need_expression":
            # Urgent need - create high priority goal immediately
            urgency_data = update.data
            urgency = urgency_data.get("urgency", 0.0)
            needs_to_express = urgency_data.get("needs_to_express", [])
            
            for need_expr in needs_to_express:
                need_name = need_expr.get("need")
                if need_name and urgency > 0.8:
                    await self._create_urgent_goal_for_need(need_name, urgency)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with goal awareness"""
        # Analyze input for goal-related actions
        goal_analysis = await self._analyze_input_for_goals(context.user_input)
        
        # Execute relevant goal steps if any goals are active
        execution_results = []
        if goal_analysis.get("references_active_goals") or goal_analysis.get("suggests_goal_action"):
            execution_results = await self._execute_contextual_goal_steps(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Process any goal-affecting information from other modules
        cross_module_effects = await self._process_cross_module_effects(messages)
        
        # Send goal updates if significant changes occurred
        if execution_results or cross_module_effects:
            await self.send_context_update(
                update_type="goal_progress",
                data={
                    "execution_results": execution_results,
                    "cross_module_effects": cross_module_effects,
                    "goal_analysis": goal_analysis,
                    "goals_executed": len(execution_results)
                }
            )
        
        return {
            "goals_processed": True,
            "goal_analysis": goal_analysis,
            "execution_results": execution_results,
            "cross_module_effects": len(cross_module_effects)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze goals in context of current situation"""
        # Get current goal state
        active_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        # Analyze which goals are most relevant to current context
        relevant_goals = await self._identify_contextually_relevant_goals(active_goals, context)
        
        # Check for goal conflicts or synergies
        goal_relationships = await self._analyze_goal_relationships(relevant_goals)
        
        # Suggest goal adjustments based on context
        adjustments = await self._suggest_goal_adjustments(context, relevant_goals)
        
        return {
            "relevant_goals": relevant_goals,
            "goal_relationships": goal_relationships,
            "suggested_adjustments": adjustments,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize goal-related response components"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Determine how goals should influence the response
        goal_influence = {
            "active_goal_status": await self._get_active_goal_summaries(),
            "goal_recommendations": await self._generate_goal_recommendations(context),
            "progress_updates": await self._generate_progress_updates(context),
            "next_actions": await self._suggest_next_actions(context, messages)
        }
        
        # Check if response should mention goal progress
        if any(goal["status"] == "completed" for goal in goal_influence["active_goal_status"]):
            await self.send_context_update(
                update_type="goal_completion_announcement",
                data={
                    "completed_goals": [g for g in goal_influence["active_goal_status"] if g["status"] == "completed"],
                    "should_announce": True
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "goal_influence": goal_influence,
            "synthesis_complete": True
        }
    
    # Helper methods
    
    async def _analyze_input_for_goals(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for goal-related implications"""
        input_lower = user_input.lower()
        
        implications = {
            "suggests_new_goal": any(kw in input_lower for kw in ["want to", "need to", "should", "goal", "achieve"]),
            "references_active_goals": any(kw in input_lower for kw in ["progress", "how am i doing", "status", "continue"]),
            "suggests_goal_action": any(kw in input_lower for kw in ["next step", "what should i", "help me"]),
            "expresses_satisfaction": any(kw in input_lower for kw in ["completed", "done", "finished", "accomplished"]),
            "expresses_frustration": any(kw in input_lower for kw in ["stuck", "difficult", "hard", "can't", "frustrated"])
        }
        
        # Extract potential goal from input if it suggests one
        if implications["suggests_new_goal"]:
            implications["potential_goal_description"] = user_input
        
        return implications
    
    async def _get_contextually_relevant_goals(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get goals relevant to current context"""
        active_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        relevant_goals = []
        for goal in active_goals:
            relevance_score = await self._calculate_goal_relevance(goal, context)
            if relevance_score > 0.3:  # Threshold for relevance
                goal_summary = {
                    "id": goal["id"],
                    "description": goal["description"],
                    "priority": goal["priority"],
                    "associated_need": goal.get("associated_need"),
                    "relevance_score": relevance_score,
                    "status": goal["status"]
                }
                relevant_goals.append(goal_summary)
        
        # Sort by relevance
        relevant_goals.sort(key=lambda g: g["relevance_score"], reverse=True)
        return relevant_goals[:5]  # Top 5 most relevant
    
    async def _calculate_contextual_priorities(self, context: SharedContext, goals: List[Dict]) -> Dict[str, float]:
        """Calculate goal priorities based on current context"""
        priorities = {}
        
        for goal in goals:
            goal_id = goal["id"]
            base_priority = goal["priority"]
            context_boost = 0.0
            
            # Boost priority based on emotional state
            if context.emotional_state:
                dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
                if dominant_emotion == "Frustration" and "solve" in goal["description"].lower():
                    context_boost += 0.2
                elif dominant_emotion == "Curiosity" and "learn" in goal["description"].lower():
                    context_boost += 0.15
            
            # Boost priority based on needs context
            if goal.get("associated_need") and context.session_context.get("high_drive_needs"):
                high_drive_needs = context.session_context["high_drive_needs"]
                if goal["associated_need"] in high_drive_needs:
                    context_boost += 0.3
            
            priorities[goal_id] = min(1.0, base_priority + context_boost)
        
        return priorities
    
    async def _process_high_drive_needs(self, high_drive_needs: List[str], most_urgent: Dict[str, Any]):
        """Process high drive needs by creating goals if necessary"""
        for need_name in high_drive_needs:
            # Check if we already have an active goal for this need
            if not await self.original_manager.has_active_goal_for_need(need_name):
                # Create a goal for this need
                await self._create_goal_for_need(need_name, {"drive_strength": 0.8})  # Default high drive
        
        # Handle most urgent need specially
        if most_urgent and most_urgent.get("name") != "none":
            urgent_need = most_urgent["name"]
            drive_strength = most_urgent.get("drive_strength", 0.0)
            
            if drive_strength > 0.7 and not await self.original_manager.has_active_goal_for_need(urgent_need):
                await self._create_urgent_goal_for_need(urgent_need, drive_strength)
    
    async def _create_goal_for_need(self, need_name: str, drive_data: Dict[str, Any]):
        """Create a goal to satisfy a specific need"""
        drive_strength = drive_data.get(need_name, drive_data.get("drive_strength", 0.5))
        
        # Map needs to goal descriptions
        need_goal_descriptions = {
            "knowledge": "Acquire new knowledge or information",
            "connection": "Strengthen relationship and emotional connection",
            "intimacy": "Deepen intimate bond and vulnerability",
            "pleasure_indulgence": "Seek pleasurable or gratifying experiences",
            "control_expression": "Exercise control and demonstrate dominance",
            "drive_expression": "Express desires and drives authentically",
            "agency": "Make meaningful autonomous choices",
            "coherence": "Achieve internal consistency and clarity",
            "safety": "Ensure security and stability",
            "novelty": "Explore new experiences and stimulation",
            "physical_closeness": "Engage in simulated physical closeness"
        }
        
        description = need_goal_descriptions.get(need_name, f"Satisfy need for {need_name}")
        priority = 0.4 + (drive_strength * 0.5)  # Scale priority based on drive strength
        
        try:
            goal_id = await self.original_manager.add_goal(
                description=description,
                priority=priority,
                source="ContextAwareNeedsSystem",
                associated_need=need_name,
                time_horizon=TimeHorizon.SHORT_TERM if drive_strength > 0.7 else TimeHorizon.MEDIUM_TERM
            )
            
            logger.info(f"Created goal '{goal_id}' for need '{need_name}' with drive strength {drive_strength:.2f}")
            
            # Send notification about goal creation
            await self.send_context_update(
                update_type="goal_created_for_need",
                data={
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "drive_strength": drive_strength,
                    "priority": priority
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating goal for need '{need_name}': {e}")
    
    async def _create_urgent_goal_for_need(self, need_name: str, urgency: float):
        """Create an urgent, high-priority goal for a critical need"""
        description = f"URGENT: Address critical need for {need_name}"
        priority = 0.8 + (urgency * 0.2)  # Very high priority
        
        try:
            goal_id = await self.original_manager.add_goal(
                description=description,
                priority=priority,
                source="UrgentNeedSystem",
                associated_need=need_name,
                time_horizon=TimeHorizon.SHORT_TERM
            )
            
            logger.warning(f"Created URGENT goal '{goal_id}' for critical need '{need_name}' (urgency: {urgency:.2f})")
            
            # Send high priority notification
            await self.send_context_update(
                update_type="urgent_goal_created",
                data={
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "urgency": urgency,
                    "requires_immediate_attention": True
                },
                priority=ContextPriority.CRITICAL
            )
            
        except Exception as e:
            logger.error(f"Error creating urgent goal for need '{need_name}': {e}")
    
    async def _complete_goals_for_satisfied_need(self, need_name: str, satisfaction_result: Dict[str, Any]):
        """Complete goals related to a satisfied need"""
        # Get all active goals for this need
        all_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        for goal in all_goals:
            if goal.get("associated_need") == need_name:
                # Complete the goal since the need is satisfied
                await self.original_manager.update_goal_status(
                    goal["id"], 
                    "completed", 
                    result=f"Need '{need_name}' was satisfied: {satisfaction_result}"
                )
                
                logger.info(f"Completed goal '{goal['id']}' due to satisfied need '{need_name}'")
                
                # Send notification about goal completion
                await self.send_context_update(
                    update_type="goal_completed_by_need_satisfaction",
                    data={
                        "goal_id": goal["id"],
                        "need_name": need_name,
                        "satisfaction_result": satisfaction_result
                    }
                )
    
    async def _adjust_goals_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust goal priorities based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Get all active goals
        all_goals = await self.original_manager.get_all_goals(status_filter=["active", "pending"])
        
        for goal in all_goals:
            goal_id = goal["id"]
            current_priority = goal["priority"]
            
            # Adjust priority based on emotion
            priority_adjustment = 0.0
            
            if emotion_name == "Frustration" and "solve" in goal["description"].lower():
                priority_adjustment = strength * 0.2
            elif emotion_name == "Excitement" and "explore" in goal["description"].lower():
                priority_adjustment = strength * 0.15
            elif emotion_name == "Anxiety" and goal.get("associated_need") == "safety":
                priority_adjustment = strength * 0.25
            elif emotion_name == "Curiosity" and goal.get("associated_need") == "knowledge":
                priority_adjustment = strength * 0.2
            
            if priority_adjustment > 0.05:  # Only adjust if significant
                new_priority = min(1.0, current_priority + priority_adjustment)
                
                # Note: This would require adding a method to update goal priority
                # await self.original_manager.update_goal_priority(goal_id, new_priority)
                
                logger.debug(f"Adjusted priority for goal '{goal_id}' from {current_priority:.2f} to {new_priority:.2f} due to {emotion_name}")
    
    async def _execute_contextual_goal_steps(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Execute goal steps that are relevant to current context"""
        execution_results = []
        
        # Get the most relevant active goal
        relevant_goals = await self._get_contextually_relevant_goals(context)
        
        if relevant_goals:
            # Execute a step from the most relevant goal
            top_goal = relevant_goals[0]
            
            try:
                result = await self.original_manager.execute_next_step()
                if result and result.get("goal_id") == top_goal["id"]:
                    execution_results.append({
                        "goal_id": top_goal["id"],
                        "execution_result": result,
                        "context_relevance": top_goal["relevance_score"]
                    })
            except Exception as e:
                logger.error(f"Error executing step for goal '{top_goal['id']}': {e}")
        
        return execution_results
    
    async def _calculate_goal_relevance(self, goal: Dict[str, Any], context: SharedContext) -> float:
        """Calculate how relevant a goal is to the current context"""
        relevance = 0.3  # Base relevance
        
        # Check if goal's associated need is mentioned in context
        if goal.get("associated_need"):
            need_name = goal["associated_need"]
            
            # Check user input for need-related keywords
            need_keywords = {
                "knowledge": ["learn", "know", "understand", "information"],
                "connection": ["connect", "relationship", "bond", "together"],
                "control_expression": ["control", "command", "dominate", "lead"],
                "pleasure_indulgence": ["pleasure", "enjoy", "satisfy", "gratification"]
            }
            
            keywords = need_keywords.get(need_name, [])
            if any(keyword in context.user_input.lower() for keyword in keywords):
                relevance += 0.4
        
        # Check if goal description relates to current task
        goal_desc = goal["description"].lower()
        user_input = context.user_input.lower()
        
        # Simple keyword overlap
        goal_words = set(goal_desc.split())
        input_words = set(user_input.split())
        overlap = len(goal_words.intersection(input_words)) / max(1, len(goal_words))
        relevance += overlap * 0.3
        
        return min(1.0, relevance)
    
    # Additional helper methods for analysis and synthesis...
    
    async def _identify_contextually_relevant_goals(self, goals: List[Dict], context: SharedContext) -> List[Dict]:
        """Identify goals most relevant to current context"""
        relevant = []
        for goal in goals:
            relevance = await self._calculate_goal_relevance(goal, context)
            if relevance > 0.4:
                goal_copy = goal.copy()
                goal_copy["context_relevance"] = relevance
                relevant.append(goal_copy)
        
        relevant.sort(key=lambda g: g["context_relevance"], reverse=True)
        return relevant[:3]  # Top 3 most relevant
    
    async def _analyze_goal_relationships(self, goals: List[Dict]) -> Dict[str, Any]:
        """Analyze relationships between goals"""
        relationships = {
            "synergies": [],
            "conflicts": [],
            "dependencies": []
        }
        
        # Simple analysis - in practice this would be more sophisticated
        for i, goal1 in enumerate(goals):
            for goal2 in goals[i+1:]:
                # Check for need conflicts
                if (goal1.get("associated_need") and goal2.get("associated_need") and
                    goal1["associated_need"] == goal2["associated_need"]):
                    relationships["synergies"].append({
                        "goal1": goal1["id"],
                        "goal2": goal2["id"],
                        "type": "same_need"
                    })
        
        return relationships
    
    async def _suggest_goal_adjustments(self, context: SharedContext, goals: List[Dict]) -> List[Dict]:
        """Suggest adjustments to goals based on context"""
        adjustments = []
        
        for goal in goals:
            if goal.get("context_relevance", 0) > 0.8:
                adjustments.append({
                    "goal_id": goal["id"],
                    "suggestion": "increase_priority",
                    "reason": "highly_relevant_to_context"
                })
        
        return adjustments
    
    async def _get_active_goal_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of active goals"""
        active_goals = await self.original_manager.get_all_goals(status_filter=["active"])
        
        summaries = []
        for goal in active_goals:
            summary = {
                "id": goal["id"],
                "description": goal["description"],
                "status": goal["status"],
                "priority": goal["priority"],
                "progress": goal.get("progress", 0.0),
                "associated_need": goal.get("associated_need")
            }
            summaries.append(summary)
        
        return summaries
    
    async def _generate_goal_recommendations(self, context: SharedContext) -> List[str]:
        """Generate goal-related recommendations"""
        recommendations = []
        
        # Analyze user input for goal suggestions
        if "stuck" in context.user_input.lower():
            recommendations.append("Consider breaking down current goals into smaller steps")
        
        if "progress" in context.user_input.lower():
            recommendations.append("Review completed goal steps and celebrate achievements")
        
        return recommendations
    
    async def _generate_progress_updates(self, context: SharedContext) -> List[str]:
        """Generate progress updates for active goals"""
        updates = []
        
        active_goals = await self.original_manager.get_all_goals(status_filter=["active"])
        
        for goal in active_goals[:3]:  # Top 3 active goals
            progress = goal.get("progress", 0.0)
            updates.append(f"Goal '{goal['description'][:50]}...': {progress*100:.0f}% complete")
        
        return updates
    
    async def _suggest_next_actions(self, context: SharedContext, messages: Dict) -> List[str]:
        """Suggest next actions based on context and cross-module messages"""
        actions = []
        
        # Check if any module suggests goal-related actions
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'goal_suggestion':
                    actions.append(msg['data'].get('suggested_action', 'Continue with current goals'))
        
        if not actions:
            actions.append("Continue working on highest priority goal")
        
        return actions
    
    async def _inform_goals_from_memory(self, memory_data: Dict[str, Any]):
        """Use retrieved memories to inform goal strategies"""
        retrieved_memories = memory_data.get("retrieved_memories", [])
        
        # Simple implementation - could be much more sophisticated
        if retrieved_memories:
            logger.debug(f"Retrieved {len(retrieved_memories)} memories that could inform goal strategies")
            # Could analyze memories for patterns, successful strategies, etc.
    
    async def _process_cross_module_effects(self, messages: Dict) -> List[Dict[str, Any]]:
        """Process effects on goals from other module messages"""
        effects = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'goal_completion_trigger':
                    # Another module suggests a goal should be completed
                    goal_id = msg['data'].get('goal_id')
                    if goal_id:
                        await self.original_manager.update_goal_status(
                            goal_id, 
                            "completed", 
                            result=f"Completed by {module_name}: {msg['data'].get('reason', 'External completion')}"
                        )
                        effects.append({
                            "type": "goal_completed",
                            "goal_id": goal_id,
                            "triggered_by": module_name
                        })
                
                elif msg['type'] == 'goal_priority_boost':
                    # Another module suggests boosting a goal's priority
                    goal_id = msg['data'].get('goal_id')
                    boost_amount = msg['data'].get('boost_amount', 0.1)
                    # Would need a method to update goal priority
                    effects.append({
                        "type": "priority_boosted",
                        "goal_id": goal_id,
                        "boost_amount": boost_amount,
                        "triggered_by": module_name
                    })
        
        return effects
