# nyx/core/a2a/context_aware_task_assignment.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareTaskAssignment(ContextAwareModule):
    """
    Context-aware wrapper for TaskAssignmentSystem with full A2A capabilities
    """
    
    def __init__(self, original_task_system):
        super().__init__("task_assignment_system")
        self.original_system = original_task_system
        self.context_subscriptions = [
            "submission_requests_task", "submission_state_available",
            "emotional_state_update", "goal_context_available",
            "relationship_state_change", "dominance_interaction",
            "needs_state_change", "memory_retrieval_complete",
            "submission_milestone_completed", "protocol_violation",
            "reward_signal", "activity_completion"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize task context for this interaction"""
        logger.debug(f"TaskAssignment received context for user: {context.user_id}")
        
        # Analyze input for task-related requests
        task_implications = await self._analyze_input_for_tasks(context.user_input)
        
        # Get user's current task state
        user_task_state = await self._get_user_task_context(context.user_id)
        
        # Check for overdue tasks
        overdue_tasks = await self._check_overdue_tasks(context.user_id)
        
        # Send initial task context to other modules
        await self.send_context_update(
            update_type="task_context_available",
            data={
                "user_id": context.user_id,
                "active_tasks": user_task_state.get("active_tasks", []),
                "task_capacity": user_task_state.get("capacity_available", True),
                "overdue_tasks": overdue_tasks,
                "task_implications": task_implications,
                "completion_rate": user_task_state.get("completion_rate", 1.0),
                "preferred_categories": user_task_state.get("preferred_categories", [])
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "submission_requests_task":
            # Submission system is requesting a task assignment
            request_data = update.data
            user_id = request_data.get("user_id")
            submission_level = request_data.get("current_level", 1)
            difficulty_pref = request_data.get("preferred_difficulty", "moderate")
            
            if user_id:
                # Assign appropriate task based on submission level
                await self._assign_submission_appropriate_task(
                    user_id, submission_level, difficulty_pref
                )
        
        elif update.update_type == "submission_state_available":
            # Update our understanding of user's submission state
            submission_data = update.data
            user_id = submission_data.get("user_id")
            
            if user_id:
                # Store submission context for task customization
                await self._update_task_submission_context(user_id, submission_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects task selection
            emotional_data = update.data
            await self._adjust_tasks_for_emotional_state(context.user_id, emotional_data)
        
        elif update.update_type == "goal_context_available":
            # Align tasks with active goals
            goal_data = update.data
            await self._align_tasks_with_goals(context.user_id, goal_data)
        
        elif update.update_type == "submission_milestone_completed":
            # Milestone completion might unlock new task types
            milestone_data = update.data
            await self._unlock_milestone_tasks(milestone_data)
        
        elif update.update_type == "needs_state_change":
            # Tasks can help satisfy needs
            needs_data = update.data
            await self._suggest_need_satisfying_tasks(context.user_id, needs_data)
        
        elif update.update_type == "protocol_violation":
            # Assign corrective tasks for violations
            violation_data = update.data
            await self._assign_corrective_task(violation_data)
        
        elif update.update_type == "activity_completion":
            # Activity completion might be task completion
            activity_data = update.data
            await self._check_task_completion_from_activity(activity_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with task awareness"""
        # Analyze input for task-related content
        task_analysis = await self._analyze_input_for_tasks(context.user_input)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Process task-related requests
        processing_results = {}
        
        if task_analysis.get("requests_new_task"):
            # User wants a new task
            task_result = await self._process_task_request(context)
            processing_results["task_assigned"] = task_result
        
        if task_analysis.get("reports_completion"):
            # User is reporting task completion
            completion_result = await self._process_completion_report(context, task_analysis)
            processing_results["completion_processed"] = completion_result
        
        if task_analysis.get("requests_task_info"):
            # User wants info about their tasks
            task_info = await self._get_task_information(context.user_id)
            processing_results["task_info"] = task_info
        
        if task_analysis.get("requests_extension"):
            # User wants to extend a deadline
            extension_result = await self._process_extension_request(context, task_analysis)
            processing_results["extension_processed"] = extension_result
        
        # Send update about task processing
        if task_analysis.get("task_relevant"):
            await self.send_context_update(
                update_type="task_interaction",
                data={
                    "analysis": task_analysis,
                    "processing_results": processing_results,
                    "user_id": context.user_id
                }
            )
        
        return {
            "tasks_processed": True,
            "analysis": task_analysis,
            "results": processing_results,
            "context_aware": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze task state in current context"""
        user_id = context.user_id
        
        # Get comprehensive task analysis
        current_tasks = await self._get_user_task_context(user_id)
        
        # Analyze task performance
        performance_analysis = await self._analyze_task_performance(user_id)
        
        # Check task-submission alignment
        submission_alignment = await self._analyze_task_submission_alignment(
            current_tasks, context.session_context.get("submission_state", {})
        )
        
        # Analyze task-goal coherence
        goal_coherence = await self._analyze_task_goal_coherence(
            current_tasks, context.goal_context
        )
        
        # Get task recommendations based on full context
        recommendations = await self._generate_task_recommendations(
            user_id, context, performance_analysis
        )
        
        # Check for task conflicts or overload
        workload_analysis = await self._analyze_task_workload(current_tasks)
        
        return {
            "current_tasks": current_tasks,
            "performance": performance_analysis,
            "submission_alignment": submission_alignment,
            "goal_coherence": goal_coherence,
            "recommendations": recommendations,
            "workload": workload_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize task-related response elements"""
        user_id = context.user_id
        messages = await self.get_cross_module_messages()
        
        # Get current task state
        current_tasks = await self._get_user_task_context(user_id)
        
        # Determine if we should mention tasks in response
        mention_tasks = await self._should_mention_tasks(context, current_tasks)
        
        # Check for urgent task reminders
        urgent_reminders = await self._get_urgent_task_reminders(current_tasks)
        
        # Generate task-related response elements
        task_elements = await self._generate_task_response_elements(
            current_tasks, context, messages
        )
        
        # Check if we should assign a spontaneous task
        spontaneous_task = await self._check_spontaneous_task_opportunity(
            context, messages, current_tasks
        )
        
        synthesis = {
            "mention_tasks": mention_tasks,
            "urgent_reminders": urgent_reminders,
            "task_elements": task_elements,
            "spontaneous_task": spontaneous_task,
            "active_task_count": len(current_tasks.get("active_tasks", [])),
            "task_suggestions": await self._generate_contextual_task_suggestions(context)
        }
        
        # Send synthesis update if needed
        if urgent_reminders or spontaneous_task:
            await self.send_context_update(
                update_type="task_synthesis",
                data=synthesis,
                priority=ContextPriority.HIGH
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_tasks(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for task-related content"""
        input_lower = user_input.lower()
        
        analysis = {
            "task_relevant": False,
            "requests_new_task": any(phrase in input_lower for phrase in [
                "give me a task", "assign me", "what should i do",
                "i need a task", "new task", "another task",
                "something to do", "train me", "test me"
            ]),
            "reports_completion": any(phrase in input_lower for phrase in [
                "i completed", "i finished", "task done", "did the task",
                "completed the", "finished the", "i did it"
            ]),
            "requests_task_info": any(phrase in input_lower for phrase in [
                "my tasks", "what tasks", "current tasks", "active tasks",
                "show tasks", "list tasks", "task status"
            ]),
            "requests_extension": any(phrase in input_lower for phrase in [
                "more time", "extend", "deadline", "can't finish",
                "need extension", "postpone", "delay"
            ]),
            "expresses_difficulty": any(phrase in input_lower for phrase in [
                "too hard", "can't do", "impossible", "too difficult",
                "struggling with", "help with task"
            ]),
            "task_feedback": any(phrase in input_lower for phrase in [
                "the task", "that task", "this task", "about the task"
            ])
        }
        
        # Set task_relevant if any aspect is true
        analysis["task_relevant"] = any(v for k, v in analysis.items() if k != "task_relevant")
        
        # Try to extract task references
        if analysis["reports_completion"] or analysis["task_feedback"]:
            # Look for task identifiers or titles
            analysis["mentioned_task_hints"] = self._extract_task_hints(input_lower)
        
        return analysis
    
    async def _get_user_task_context(self, user_id: str) -> Dict[str, Any]:
        """Get current task context for user"""
        try:
            # Get active tasks
            active_result = await self.original_system.get_active_tasks(user_id)
            
            # Get task statistics
            stats_result = await self.original_system.get_user_task_statistics(user_id)
            
            # Get user settings if available
            settings = self.original_system.user_settings.get(user_id, {})
            
            return {
                "active_tasks": active_result.get("active_tasks", []),
                "active_count": active_result.get("count", 0),
                "max_concurrent": active_result.get("max_concurrent", 3),
                "capacity_available": active_result.get("count", 0) < active_result.get("max_concurrent", 3),
                "completion_rate": stats_result.get("statistics", {}).get("completion_rate", 1.0),
                "preferred_categories": [
                    cat for cat, _ in stats_result.get("preferred_categories", [])[:3]
                ],
                "statistics": stats_result.get("statistics", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting task context: {e}")
            return {
                "active_tasks": [],
                "active_count": 0,
                "capacity_available": True,
                "completion_rate": 1.0,
                "preferred_categories": []
            }
    
    async def _check_overdue_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """Check for user's overdue tasks"""
        try:
            all_overdue = await self.original_system.get_expired_tasks()
            user_overdue = [task for task in all_overdue if task.get("user_id") == user_id]
            return user_overdue
        except Exception as e:
            logger.error(f"Error checking overdue tasks: {e}")
            return []
    
    async def _assign_submission_appropriate_task(self, user_id: str, 
                                                submission_level: int, 
                                                difficulty_pref: str):
        """Assign a task appropriate to submission level"""
        try:
            # Map submission level to task difficulty
            level_difficulty_map = {
                1: "easy",
                2: "moderate", 
                3: "moderate",
                4: "challenging",
                5: "difficult"
            }
            
            base_difficulty = level_difficulty_map.get(submission_level, "moderate")
            
            # Use preference if it's not too extreme for level
            if submission_level <= 2 and difficulty_pref in ["difficult", "extreme"]:
                difficulty = base_difficulty  # Override to appropriate level
            else:
                difficulty = difficulty_pref
            
            # Assign task
            result = await self.original_system.assign_task(
                user_id=user_id,
                difficulty_override=difficulty,
                due_in_hours=24
            )
            
            if result.get("success"):
                # Send notification about task assignment
                await self.send_context_update(
                    update_type="task_assigned",
                    data={
                        "user_id": user_id,
                        "task": result.get("task"),
                        "from_submission_request": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error assigning submission task: {e}")
    
    async def _update_task_submission_context(self, user_id: str, submission_data: Dict[str, Any]):
        """Update task context with submission information"""
        # Store submission context for future task assignments
        # This helps customize tasks based on submission level
        if not hasattr(self, '_submission_contexts'):
            self._submission_contexts = {}
            
        self._submission_contexts[user_id] = {
            "level": submission_data.get("current_level"),
            "score": submission_data.get("submission_score"),
            "path": submission_data.get("active_path"),
            "updated": datetime.now()
        }
    
    async def _adjust_tasks_for_emotional_state(self, user_id: str, emotional_data: Dict[str, Any]):
        """Adjust task recommendations based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
            
        emotion_name, intensity = dominant_emotion
        
        # Don't assign difficult tasks when user is struggling emotionally
        if emotion_name in ["Frustration", "Anxiety", "Fear"] and intensity > 0.6:
            # Check if user has active difficult tasks
            active_tasks = await self._get_user_task_context(user_id)
            
            for task in active_tasks.get("active_tasks", []):
                if task.get("difficulty") in ["difficult", "extreme", "challenging"]:
                    # Suggest easier alternatives or support
                    await self.send_context_update(
                        update_type="task_emotional_adjustment",
                        data={
                            "user_id": user_id,
                            "emotion": emotion_name,
                            "suggestion": "consider_easier_tasks",
                            "affected_tasks": [task["task_id"]]
                        }
                    )
                    break
    
    async def _align_tasks_with_goals(self, user_id: str, goal_data: Dict[str, Any]):
        """Ensure tasks align with user's active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        if not active_goals:
            return
            
        # Extract goal themes
        goal_themes = []
        for goal in active_goals:
            if "knowledge" in goal.get("description", "").lower():
                goal_themes.append("learning")
            elif "connection" in goal.get("description", "").lower():
                goal_themes.append("social")
            elif "control" in goal.get("description", "").lower():
                goal_themes.append("dominance")
                
        # Store goal themes for task selection
        if not hasattr(self, '_user_goal_themes'):
            self._user_goal_themes = {}
            
        self._user_goal_themes[user_id] = goal_themes
    
    async def _process_task_request(self, context: SharedContext) -> Dict[str, Any]:
        """Process a request for a new task"""
        user_id = context.user_id
        
        try:
            # Get submission context if available
            submission_level = 1
            if hasattr(self, '_submission_contexts') and user_id in self._submission_contexts:
                submission_level = self._submission_contexts[user_id].get("level", {}).get("id", 1)
            
            # Get goal themes if available
            preferred_categories = []
            if hasattr(self, '_user_goal_themes') and user_id in self._user_goal_themes:
                themes = self._user_goal_themes[user_id]
                # Map themes to task categories
                if "learning" in themes:
                    preferred_categories.append("self_improvement")
                if "dominance" in themes:
                    preferred_categories.extend(["obedience", "protocol"])
            
            # Assign task with context awareness
            result = await self.original_system.assign_task(
                user_id=user_id,
                difficulty_override=self._get_contextual_difficulty(submission_level),
                due_in_hours=24
            )
            
            if result.get("success"):
                # Send cross-module update
                await self.send_context_update(
                    update_type="task_assigned",
                    data={
                        "user_id": user_id,
                        "task": result.get("task"),
                        "context_aware": True
                    }
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing task request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_completion_report(self, context: SharedContext, 
                                       task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task completion report"""
        user_id = context.user_id
        
        try:
            # Get active tasks to find which one they're completing
            active_tasks = await self._get_user_task_context(user_id)
            
            if not active_tasks.get("active_tasks"):
                return {"success": False, "message": "No active tasks to complete"}
            
            # Try to identify which task from hints
            task_hints = task_analysis.get("mentioned_task_hints", [])
            identified_task = None
            
            # If only one active task, assume that's it
            if len(active_tasks["active_tasks"]) == 1:
                identified_task = active_tasks["active_tasks"][0]
            else:
                # Try to match based on hints
                for task in active_tasks["active_tasks"]:
                    for hint in task_hints:
                        if hint in task["title"].lower() or hint in task.get("description", "").lower():
                            identified_task = task
                            break
                    if identified_task:
                        break
            
            if not identified_task:
                # Ask for clarification
                return {
                    "success": False,
                    "needs_clarification": True,
                    "active_tasks": [{"id": t["task_id"], "title": t["title"]} 
                                   for t in active_tasks["active_tasks"]]
                }
            
            # Process completion
            result = await self.original_system.complete_task(
                task_id=identified_task["task_id"],
                verification_data={"reported_via": "context_aware_system"},
                completion_notes="Completed via conversational report"
            )
            
            if result.get("success"):
                # Send completion update
                await self.send_context_update(
                    update_type="task_completion",
                    data={
                        "user_id": user_id,
                        "task_id": identified_task["task_id"],
                        "completed": result.get("verified", False),
                        "title": identified_task["title"],
                        "category": identified_task.get("category"),
                        "difficulty": identified_task.get("difficulty")
                    },
                    priority=ContextPriority.HIGH
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing completion report: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_task_hints(self, input_lower: str) -> List[str]:
        """Extract potential task references from input"""
        hints = []
        
        # Common task keywords
        task_keywords = ["ritual", "service", "writing", "position", "mantra", 
                        "photo", "video", "report", "exercise", "practice"]
        
        for keyword in task_keywords:
            if keyword in input_lower:
                hints.append(keyword)
                
        return hints
    
    def _get_contextual_difficulty(self, submission_level: int) -> str:
        """Get appropriate difficulty based on submission level"""
        if submission_level <= 1:
            return "easy"
        elif submission_level <= 2:
            return "moderate"
        elif submission_level <= 3:
            return "moderate"
        elif submission_level <= 4:
            return "challenging"
        else:
            return "difficult"
    
    async def _should_mention_tasks(self, context: SharedContext, 
                                  current_tasks: Dict[str, Any]) -> bool:
        """Determine if tasks should be mentioned in response"""
        # Mention if user asked about tasks
        if "task" in context.user_input.lower():
            return True
            
        # Mention if there are urgent/overdue tasks
        for task in current_tasks.get("active_tasks", []):
            if task.get("time_remaining") and task["time_remaining"].get("overdue"):
                return True
                
        return False
    
    async def _generate_task_response_elements(self, current_tasks: Dict[str, Any],
                                             context: SharedContext,
                                             messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate task-related response elements"""
        elements = {
            "mention_count": len(current_tasks.get("active_tasks", [])),
            "urgency_level": "normal",
            "encouragement_needed": False,
            "task_reminders": []
        }
        
        # Check for overdue tasks
        overdue_count = 0
        for task in current_tasks.get("active_tasks", []):
            if task.get("time_remaining", {}).get("overdue"):
                overdue_count += 1
                
        if overdue_count > 0:
            elements["urgency_level"] = "high"
            elements["encouragement_needed"] = True
            
        # Check completion rate
        if current_tasks.get("completion_rate", 1.0) < 0.5:
            elements["encouragement_needed"] = True
            
        return elements
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
