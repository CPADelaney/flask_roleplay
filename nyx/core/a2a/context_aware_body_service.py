# nyx/core/a2a/context_aware_body_service.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareBodyService(ContextAwareModule):
    """
    Enhanced Body Service System with full context distribution capabilities
    """
    
    def __init__(self, original_body_service):
        super().__init__("body_service")
        self.original_service = original_body_service
        self.context_subscriptions = [
            "dominance_level_change", "submission_metric_update", "emotional_state_update",
            "punishment_needed", "reward_signal", "task_completion", "protocol_violation",
            "relationship_milestone", "physical_state_update", "femdom_mode_change"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize body service processing for this context"""
        logger.debug(f"BodyService received context for user: {context.user_id}")
        
        # Check if any active tasks or positions
        user_id = context.user_id
        active_status = await self._check_active_assignments(user_id)
        
        # Analyze input for service-related keywords
        service_implications = await self._analyze_input_for_service(context.user_input)
        
        # Send initial body service context
        await self.send_context_update(
            update_type="body_service_context",
            data={
                "active_task": active_status.get("active_task"),
                "active_position": active_status.get("active_position"),
                "service_implications": service_implications,
                "available_for_assignment": not active_status.get("active_task"),
                "user_training_level": await self._get_user_training_level(user_id)
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "dominance_level_change":
            # Adjust task difficulty based on dominance level
            dominance_data = update.data
            new_level = dominance_data.get("new_level", 0.5)
            user_id = dominance_data.get("user_id")
            
            if user_id and new_level > 0.7:
                # Consider assigning more challenging tasks
                await self._adjust_task_difficulty(user_id, new_level)
        
        elif update.update_type == "submission_metric_update":
            # Track submission for service rewards
            submission_data = update.data
            metric = submission_data.get("metric")
            user_id = submission_data.get("user_id")
            
            if metric == "obedience" and submission_data.get("value", 0) > 0.8:
                # High obedience - consider rewarding with easier tasks
                await self._consider_service_reward(user_id, "high_obedience")
        
        elif update.update_type == "punishment_needed":
            # Assign punishment tasks
            punishment_data = update.data
            user_id = punishment_data.get("user_id")
            severity = punishment_data.get("severity", 0.5)
            
            await self._assign_punishment_task(user_id, severity)
        
        elif update.update_type == "emotional_state_update":
            # Consider emotional state for task selection
            emotional_data = update.data
            if emotional_data.get("dominant_emotion") == ("Sadistic", None):
                # Increase humiliation factor in tasks
                await self._adjust_task_selection_criteria("increase_humiliation")
        
        elif update.update_type == "protocol_violation":
            # Assign corrective service
            violation_data = update.data
            user_id = violation_data.get("user_id")
            
            await self._assign_corrective_service(user_id, violation_data)
        
        elif update.update_type == "physical_state_update":
            # Adjust for physical limitations
            physical_data = update.data
            fatigue = physical_data.get("fatigue", 0)
            
            if fatigue > 0.7:
                # Reduce endurance requirements
                await self._adjust_endurance_requirements(physical_data.get("user_id"), fatigue)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context awareness"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Check for task/position completion attempts
        completion_check = await self._check_for_completion_attempt(context.user_input)
        
        if completion_check.get("is_completion_attempt"):
            # Process completion
            result = await self._process_completion_with_context(
                context.user_id, 
                completion_check,
                messages
            )
            
            # Send completion update
            await self.send_context_update(
                update_type="service_task_completed",
                data={
                    "user_id": context.user_id,
                    "task_type": completion_check.get("type"),
                    "quality": result.get("quality_rating", 0.5),
                    "reward_earned": result.get("reward_result") is not None
                },
                priority=ContextPriority.HIGH
            )
        
        # Check for new assignment requests
        assignment_check = await self._check_for_assignment_request(context.user_input)
        
        if assignment_check.get("requests_assignment"):
            # Process assignment with context
            result = await self._process_assignment_with_context(
                context.user_id,
                assignment_check,
                messages
            )
            
            # Send assignment update
            await self.send_context_update(
                update_type="service_task_assigned",
                data={
                    "user_id": context.user_id,
                    "task_id": result.get("task_id"),
                    "difficulty": result.get("difficulty", 0.5),
                    "duration_minutes": result.get("duration_minutes", 10)
                }
            )
        
        return {
            "processed": True,
            "completion_processed": completion_check.get("is_completion_attempt", False),
            "assignment_processed": assignment_check.get("requests_assignment", False)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze service context"""
        user_id = context.user_id
        
        # Get user's service record
        service_record = await self.original_service.get_user_service_record(user_id)
        
        # Analyze training needs
        training_analysis = await self._analyze_training_needs(service_record)
        
        # Get recommended tasks based on context
        recommended_tasks = await self._get_contextual_task_recommendations(
            context, service_record
        )
        
        # Analyze position proficiency
        position_analysis = await self._analyze_position_proficiency(service_record)
        
        return {
            "service_record": service_record,
            "training_analysis": training_analysis,
            "recommended_tasks": recommended_tasks,
            "position_analysis": position_analysis,
            "ready_for_advancement": training_analysis.get("ready_for_advancement", False)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize body service components for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine if service instruction should be included
        include_service = await self._should_include_service_instruction(context, messages)
        
        synthesis = {
            "include_service_element": include_service,
            "service_suggestions": []
        }
        
        if include_service:
            # Generate appropriate service suggestions
            synthesis["service_suggestions"] = await self._generate_service_suggestions(
                context, messages
            )
            
            # Add position recommendations if appropriate
            if self._should_suggest_position(context, messages):
                synthesis["position_suggestion"] = await self._suggest_position(
                    context.user_id, messages
                )
            
            # Check if praise or correction needed
            recent_completion = self._get_recent_completion(messages)
            if recent_completion:
                if recent_completion.get("quality_rating", 0) > 0.8:
                    synthesis["include_praise"] = True
                    synthesis["praise_focus"] = recent_completion.get("task_name")
                elif recent_completion.get("quality_rating", 0) < 0.4:
                    synthesis["include_correction"] = True
                    synthesis["correction_focus"] = recent_completion.get("task_name")
        
        return synthesis
    
    # Helper methods
    async def _check_active_assignments(self, user_id: str) -> Dict[str, Any]:
        """Check for active tasks or positions"""
        if user_id not in self.original_service.context.user_training:
            return {"active_task": None, "active_position": None}
        
        user_training = self.original_service.context.user_training[user_id]
        return {
            "active_task": user_training.current_task,
            "active_position": user_training.current_position
        }
    
    async def _analyze_input_for_service(self, user_input: str) -> Dict[str, Any]:
        """Analyze input for service-related content"""
        input_lower = user_input.lower()
        
        return {
            "mentions_service": any(kw in input_lower for kw in ["serve", "service", "task", "position"]),
            "requests_task": any(kw in input_lower for kw in ["give me", "assign", "what should i do", "task"]),
            "reports_completion": any(kw in input_lower for kw in ["finished", "completed", "done", "maintained"]),
            "expresses_difficulty": any(kw in input_lower for kw in ["hard", "difficult", "can't", "struggling"])
        }
    
    async def _get_user_training_level(self, user_id: str) -> str:
        """Get user's training level"""
        if user_id not in self.original_service.context.user_training:
            return "beginner"
        
        user_training = self.original_service.context.user_training[user_id]
        total_tasks = sum(stats["completion_count"] for stats in user_training.tasks.values())
        
        if total_tasks < 5:
            return "beginner"
        elif total_tasks < 20:
            return "intermediate"
        else:
            return "advanced"
    
    async def _assign_punishment_task(self, user_id: str, severity: float):
        """Assign a punishment task based on severity"""
        # Select appropriate punishment task
        if severity > 0.7:
            task_type = "humbling_display"
        elif severity > 0.4:
            task_type = "extended_kneeling"
        else:
            task_type = "recite_rules"
        
        # Assign the task
        result = await self.original_service.assign_service_task(
            user_id=user_id,
            task_type=task_type,
            duration=None  # Use default duration
        )
        
        if result.get("success"):
            # Send notification
            await self.send_context_update(
                update_type="punishment_task_assigned",
                data={
                    "user_id": user_id,
                    "task_id": result.get("task_id"),
                    "task_name": result.get("task_name"),
                    "severity": severity
                },
                priority=ContextPriority.HIGH
            )
    
    async def _analyze_training_needs(self, service_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what training the user needs"""
        if not service_record.get("success"):
            return {"needs_basic_training": True}
        
        task_stats = service_record.get("task_statistics", {})
        position_stats = service_record.get("position_statistics", {})
        
        # Identify weak areas
        weak_tasks = []
        for task_id, stats in task_stats.items():
            if stats.get("average_rating", 1.0) < 0.6:
                weak_tasks.append(task_id)
        
        weak_positions = []
        for pos_id, stats in position_stats.items():
            if stats.get("maintained_rate", 1.0) < 0.7:
                weak_positions.append(pos_id)
        
        # Check for advancement readiness
        avg_quality = service_record.get("overall_stats", {}).get("average_quality", 0)
        total_tasks = service_record.get("overall_stats", {}).get("completed_tasks", 0)
        
        ready_for_advancement = avg_quality > 0.7 and total_tasks > 10
        
        return {
            "weak_tasks": weak_tasks,
            "weak_positions": weak_positions,
            "needs_practice": len(weak_tasks) > 0 or len(weak_positions) > 0,
            "ready_for_advancement": ready_for_advancement,
            "recommended_focus": weak_tasks[0] if weak_tasks else None
        }
    
    # Delegate other methods to original service
    def __getattr__(self, name):
        """Delegate any missing methods to the original service"""
        return getattr(self.original_service, name)
