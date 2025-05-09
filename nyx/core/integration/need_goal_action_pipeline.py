# nyx/core/integration/need_goal_action_pipeline.py

import logging
import asyncio
import datetime
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import deque

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method
from nyx.core.integration.action_selector import ActionPriority

logger = logging.getLogger(__name__)

class NeedGoalActionPipeline:
    """
    Integration pipeline connecting needs to goals to actions.
    
    This module creates a cohesive pipeline from need detection to action execution,
    ensuring that unmet needs generate appropriate goals, which then lead to concrete
    actions, and these actions are prioritized and executed to satisfy the needs.
    
    Key functions:
    1. Monitors need states and triggers goal creation when needs exceed thresholds
    2. Coordinates between goal manager and needs system
    3. Transforms high-level goals into concrete action plans
    4. Prioritizes actions based on need urgency and goal importance
    5. Provides feedback to update need satisfaction based on action outcomes
    """
    
    def __init__(self, 
                brain_reference=None, 
                needs_system=None, 
                goal_manager=None, 
                action_selector=None,
                reward_system=None):
        """Initialize the need-goal-action pipeline."""
        self.brain = brain_reference
        self.needs_system = needs_system
        self.goal_manager = goal_manager
        self.action_selector = action_selector
        self.reward_system = reward_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Pipeline configuration
        self.need_threshold_for_goal = 0.4  # Minimum need drive to trigger a goal
        self.goal_priority_multiplier = 1.5  # How strongly need drive affects goal priority
        self.action_priority_mapping = {
            "high": ActionPriority.HIGH,
            "medium": ActionPriority.MEDIUM,
            "low": ActionPriority.LOW
        }
        
        # Pipeline state tracking
        self.need_goal_mappings = {}  # need_name -> list of goal_ids
        self.goal_action_mappings = {}  # goal_id -> list of action_ids
        self.action_outcomes = {}  # action_id -> outcome data
        
        # Recent pipeline activity
        self.recent_need_triggers = deque(maxlen=20)  # Recent need triggers
        self.recent_goal_creations = deque(maxlen=20)  # Recent goal creations
        self.recent_action_submissions = deque(maxlen=20)  # Recent action submissions
        
        # Monitoring
        self.pipeline_metrics = {
            "need_triggers": 0,
            "goals_created": 0,
            "actions_submitted": 0,
            "completion_rate": 0.0,
            "average_latency": 0.0
        }
        
        # Integration event subscriptions
        self._subscribed = False
        
        # Timestamp tracking
        self.last_pipeline_run = datetime.datetime.now()
        self.startup_time = datetime.datetime.now()
        
        logger.info("NeedGoalActionPipeline initialized")
    
    async def initialize(self) -> bool:
        """Initialize the pipeline and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.needs_system and hasattr(self.brain, "needs_system"):
                self.needs_system = self.brain.needs_system
                
            if not self.goal_manager and hasattr(self.brain, "goal_manager"):
                self.goal_manager = self.brain.goal_manager
                
            if not self.action_selector and hasattr(self.brain, "action_selector"):
                self.action_selector = self.brain.action_selector
                
            if not self.reward_system and hasattr(self.brain, "reward_system"):
                self.reward_system = self.brain.reward_system
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("need_state_change", self._handle_need_state_change)
                self.event_bus.subscribe("goal_status_change", self._handle_goal_status_change)
                self.event_bus.subscribe("action_completed", self._handle_action_completed)
                self._subscribed = True
            
            logger.info("NeedGoalActionPipeline successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing NeedGoalActionPipeline: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="NeedGoalAction")
    async def run_pipeline_cycle(self) -> Dict[str, Any]:
        """
        Run a full cycle of the need-goal-action pipeline.
        
        This method checks need states, generates goals for unmet needs,
        and submits actions for active goals.
        
        Returns:
            Cycle results with statistics
        """
        cycle_results = {
            "needs_checked": 0,
            "goals_created": 0,
            "actions_submitted": 0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # 1. Check need states and generate goals for unmet needs
            if self.needs_system:
                # Update needs first
                drive_strengths = await self.needs_system.update_needs()
                
                # Get detailed needs state
                needs_state = await self.needs_system.get_needs_state_async()
                cycle_results["needs_checked"] = len(needs_state)
                
                # Find unmet needs that exceed threshold
                unmet_needs = []
                for need_name, need_data in needs_state.items():
                    drive_strength = need_data.get("drive_strength", 0.0)
                    
                    if drive_strength >= self.need_threshold_for_goal:
                        # Check if there's already an active goal for this need
                        existing_goals = self.need_goal_mappings.get(need_name, [])
                        active_goals = []
                        
                        if existing_goals and self.goal_manager:
                            # Check which goals are still active
                            for goal_id in existing_goals:
                                goal_status = await self.goal_manager.get_goal_status(goal_id)
                                if goal_status and goal_status.get("status") in ["active", "pending"]:
                                    active_goals.append(goal_id)
                        
                        if not active_goals:
                            # No active goals for this need, add to unmet needs
                            unmet_needs.append((need_name, need_data))
                
                # Generate goals for unmet needs
                if unmet_needs and self.goal_manager:
                    for need_name, need_data in unmet_needs:
                        # Calculate priority based on drive strength
                        priority = min(1.0, need_data.get("drive_strength", 0.5) * self.goal_priority_multiplier)
                        
                        # Generate goal
                        goal_id = await self._create_goal_for_need(need_name, need_data, priority)
                        
                        if goal_id:
                            # Add to need-goal mappings
                            if need_name not in self.need_goal_mappings:
                                self.need_goal_mappings[need_name] = []
                            self.need_goal_mappings[need_name].append(goal_id)
                            
                            cycle_results["goals_created"] += 1
            
            # 2. Check active goals and submit actions
            if self.goal_manager and self.action_selector:
                # Get active goals
                active_goals = await self.goal_manager.get_all_goals(status_filter=["active"])
                
                for goal in active_goals:
                    goal_id = goal.get("id")
                    
                    # Check if goal has current step
                    current_step = goal.get("current_step")
                    if current_step:
                        # Prepare and submit action
                        action_submitted = await self._submit_action_for_goal_step(goal_id, current_step)
                        
                        if action_submitted:
                            cycle_results["actions_submitted"] += 1
            
            # 3. Update pipeline metrics
            self._update_pipeline_metrics(cycle_results)
            
            # Record when pipeline was last run
            self.last_pipeline_run = datetime.datetime.now()
            
            return cycle_results
        except Exception as e:
            logger.error(f"Error running pipeline cycle: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="NeedGoalAction")
    async def create_goal_for_need(self, 
                                need_name: str, 
                                priority: Optional[float] = None) -> Dict[str, Any]:
        """
        Manually create a goal for a specific need.
        
        Args:
            need_name: Name of the need to create goal for
            priority: Optional priority override (0.0-1.0)
            
        Returns:
            Goal creation result
        """
        if not self.needs_system:
            return {"status": "error", "message": "Needs system not available"}
        
        if not self.goal_manager:
            return {"status": "error", "message": "Goal manager not available"}
        
        try:
            # Get need state
            needs_state = await self.needs_system.get_needs_state_async()
            
            if need_name not in needs_state:
                return {"status": "error", "message": f"Unknown need: {need_name}"}
            
            need_data = needs_state[need_name]
            
            # Use provided priority or calculate from drive strength
            if priority is None:
                priority = min(1.0, need_data.get("drive_strength", 0.5) * self.goal_priority_multiplier)
            
            # Create goal
            goal_id = await self._create_goal_for_need(need_name, need_data, priority)
            
            if goal_id:
                # Add to need-goal mappings
                if need_name not in self.need_goal_mappings:
                    self.need_goal_mappings[need_name] = []
                self.need_goal_mappings[need_name].append(goal_id)
                
                return {
                    "status": "success",
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "priority": priority
                }
            else:
                return {"status": "error", "message": "Failed to create goal"}
        except Exception as e:
            logger.error(f"Error creating goal for need {need_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NeedGoalAction")
    async def submit_action_for_goal(self, 
                                  goal_id: str, 
                                  action_type: str,
                                  parameters: Dict[str, Any],
                                  priority: str = "medium") -> Dict[str, Any]:
        """
        Manually submit an action for a goal.
        
        Args:
            goal_id: ID of the goal to submit action for
            action_type: Type of action to submit
            parameters: Action parameters
            priority: Action priority (high, medium, low)
            
        Returns:
            Action submission result
        """
        if not self.action_selector:
            return {"status": "error", "message": "Action selector not available"}
        
        try:
            # Map string priority to ActionPriority enum
            action_priority = self.action_priority_mapping.get(priority.lower(), ActionPriority.MEDIUM)
            
            # Create action
            action_id = await self.action_selector.create_action(
                action_type=action_type,
                source_module="need_goal_action_pipeline",
                parameters=parameters,
                priority=action_priority
            )
            
            if action_id:
                # Record mapping
                if goal_id not in self.goal_action_mappings:
                    self.goal_action_mappings[goal_id] = []
                self.goal_action_mappings[goal_id].append(action_id)
                
                # Record in recent submissions
                self.recent_action_submissions.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "goal_id": goal_id,
                    "action_id": action_id,
                    "action_type": action_type,
                    "priority": priority
                })
                
                # Update metrics
                self.pipeline_metrics["actions_submitted"] += 1
                
                return {
                    "status": "success",
                    "action_id": action_id,
                    "goal_id": goal_id,
                    "priority": priority
                }
            else:
                return {"status": "error", "message": "Failed to create action"}
        except Exception as e:
            logger.error(f"Error submitting action for goal {goal_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NeedGoalAction")
    async def provide_action_feedback(self, 
                                   action_id: str, 
                                   success: bool,
                                   result: Any = None) -> Dict[str, Any]:
        """
        Provide feedback about an action's outcome to update need satisfaction.
        
        Args:
            action_id: ID of the action to provide feedback for
            success: Whether the action was successful
            result: Optional result data
            
        Returns:
            Feedback processing result
        """
        if not self.action_selector:
            return {"status": "error", "message": "Action selector not available"}
        
        try:
            # Get action status
            action_status = await self.action_selector.get_action_status(action_id)
            
            if not action_status:
                return {"status": "error", "message": f"Unknown action: {action_id}"}
            
            # Find associated goal
            goal_id = None
            for g_id, actions in self.goal_action_mappings.items():
                if action_id in actions:
                    goal_id = g_id
                    break
            
            if not goal_id:
                return {
                    "status": "partial", 
                    "message": "Action found but not associated with a goal"
                }
            
            # Find need associated with goal
            need_name = None
            for n_name, goals in self.need_goal_mappings.items():
                if goal_id in goals:
                    need_name = n_name
                    break
            
            # Record outcome
            self.action_outcomes[action_id] = {
                "action_id": action_id,
                "goal_id": goal_id,
                "need_name": need_name,
                "success": success,
                "result": result,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Update need satisfaction if need found and needs system available
            if need_name and self.needs_system:
                satisfaction_change = 0.0
                
                if success:
                    # Successful action increases need satisfaction
                    # The amount depends on action parameters, but we use a simple approach here
                    satisfaction_change = 0.15  # Base satisfaction from successful action
                    
                    # Adjust based on action priority
                    priority_str = action_status.get("priority", "MEDIUM")
                    if hasattr(priority_str, "name"):  # If it's an enum
                        priority_str = priority_str.name
                    
                    if isinstance(priority_str, str):
                        if "HIGH" in priority_str:
                            satisfaction_change += 0.1  # Higher satisfaction for high priority actions
                        elif "LOW" in priority_str:
                            satisfaction_change -= 0.05  # Lower satisfaction for low priority actions
                    
                    # Satisfy the need
                    await self.needs_system.satisfy_need(
                        need_name, 
                        satisfaction_change,
                        context={"action_id": action_id, "goal_id": goal_id}
                    )
                else:
                    # Failed action might slightly decrease satisfaction
                    await self.needs_system.decrease_need(
                        need_name,
                        0.05,
                        reason=f"action_failure_{action_id}"
                    )
                
                # Send reward signal if available
                if self.reward_system and hasattr(self.reward_system, "process_reward_signal"):
                    # Create reward signal
                    from nyx.core.reward_system import RewardSignal
                    
                    reward_value = 0.2 if success else -0.1  # Simple reward/penalty
                    
                    reward_signal = RewardSignal(
                        value=reward_value,
                        source="action_feedback",
                        context={
                            "action_id": action_id,
                            "goal_id": goal_id,
                            "need_name": need_name,
                            "success": success
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    
                    await self.reward_system.process_reward_signal(reward_signal)
            
            return {
                "status": "success",
                "action_id": action_id,
                "goal_id": goal_id,
                "need_name": need_name,
                "success": success
            }
        except Exception as e:
            logger.error(f"Error providing action feedback: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NeedGoalAction")
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the need-goal-action pipeline.
        
        Returns:
            Current pipeline status
        """
        try:
            # Get need states if available
            need_states = {}
            if self.needs_system:
                need_states = await self.needs_system.get_needs_state_async()
            
            # Get active goals if available
            active_goals = []
            if self.goal_manager:
                active_goals = await self.goal_manager.get_all_goals(status_filter=["active", "pending"])
            
            # Get active actions if available
            active_actions = {}
            if self.action_selector:
                action_status = await self.action_selector.get_queue_status()
                active_actions = {
                    "queued": action_status.get("queued_actions", 0),
                    "executing": action_status.get("executing_actions", 0),
                    "next_actions": action_status.get("next_actions", [])
                }
            
            # Count mappings
            need_goal_count = sum(len(goals) for goals in self.need_goal_mappings.values())
            goal_action_count = sum(len(actions) for actions in self.goal_action_mappings.values())
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "needs": {
                    "count": len(need_states),
                    "high_drive_needs": [
                        {
                            "name": name, 
                            "drive": data.get("drive_strength", 0.0),
                            "level": data.get("level", 0.0)
                        }
                        for name, data in need_states.items() 
                        if data.get("drive_strength", 0.0) > 0.5
                    ]
                },
                "goals": {
                    "active_count": len(active_goals),
                    "active_goals": [
                        {
                            "id": goal.get("id"),
                            "description": goal.get("description", ""),
                            "priority": goal.get("priority", 0.5)
                        }
                        for goal in active_goals[:5]  # Limit to 5 for brevity
                    ]
                },
                "actions": active_actions,
                "mappings": {
                    "need_goal_mappings": need_goal_count,
                    "goal_action_mappings": goal_action_count
                },
                "metrics": self.pipeline_metrics,
                "last_run": self.last_pipeline_run.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _create_goal_for_need(self, need_name: str, need_data: Dict[str, Any], priority: float) -> Optional[str]:
        """Create a goal for an unmet need."""
        if not self.goal_manager:
            return None
        
        try:
            # Prepare goal description
            level = need_data.get("level", 0.0)
            drive = need_data.get("drive_strength", 0.0)
            description = f"Satisfy '{need_name}' need (Level: {level:.2f}, Drive: {drive:.2f})"
            
            # Specific descriptions for known needs
            if need_name == "knowledge":
                description = "Explore and acquire new knowledge to satisfy curiosity"
            elif need_name == "coherence":
                description = "Increase internal consistency and clarity"
            elif need_name == "agency":
                description = "Express autonomy through meaningful choices"
            elif need_name == "connection":
                description = "Develop meaningful connection with user"
            elif need_name == "intimacy":
                description = "Deepen emotional intimacy with user"
            elif need_name == "safety":
                description = "Enhance safety and stability of operations"
            elif need_name == "novelty":
                description = "Seek out new experiences and stimulation"
            elif need_name == "physical_closeness":
                description = "Engage in simulated physical/sensory experiences"
            elif need_name == "drive_expression":
                description = "Express drives and desires appropriately"
            elif need_name == "control_expression":
                description = "Exercise appropriate control and guidance"
            
            # Create and record the goal trigger
            trigger_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "need_name": need_name,
                "priority": priority,
                "drive_strength": drive,
                "level": level
            }
            
            self.recent_need_triggers.append(trigger_data)
            self.pipeline_metrics["need_triggers"] += 1
            
            # Create goal
            goal_id = await self.goal_manager.add_goal(
                description=description,
                priority=priority,
                source="NeedGoalActionPipeline",
                associated_need=need_name
            )
            
            # Record goal creation
            if goal_id:
                self.recent_goal_creations.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "goal_id": goal_id,
                    "need_name": need_name,
                    "priority": priority,
                    "description": description
                })
                
                self.pipeline_metrics["goals_created"] += 1
                
                logger.info(f"Created goal {goal_id} for need '{need_name}' with priority {priority:.2f}")
            
            return goal_id
        except Exception as e:
            logger.error(f"Error creating goal for need '{need_name}': {e}")
            return None
    
    async def _submit_action_for_goal_step(self, goal_id: str, step_data: Dict[str, Any]) -> bool:
        """Submit an action based on a goal step."""
        if not self.action_selector:
            return False
        
        try:
            # Extract step information
            action = step_data.get("action")
            parameters = step_data.get("parameters", {})
            
            if not action:
                logger.warning(f"Cannot submit action for goal {goal_id}: Missing action in step")
                return False
            
            # Determine priority based on goal priority
            priority = ActionPriority.MEDIUM  # Default
            
            if self.goal_manager:
                goal_status = await self.goal_manager.get_goal_status(goal_id)
                if goal_status:
                    goal_priority = goal_status.get("priority", 0.5)
                    
                    # Map goal priority to action priority
                    if goal_priority > 0.8:
                        priority = ActionPriority.HIGH
                    elif goal_priority < 0.3:
                        priority = ActionPriority.LOW
            
            # Create action
            action_id = await self.action_selector.create_action(
                action_type=action,
                source_module="need_goal_action_pipeline",
                parameters=parameters,
                priority=priority
            )
            
            if action_id:
                # Record mapping
                if goal_id not in self.goal_action_mappings:
                    self.goal_action_mappings[goal_id] = []
                self.goal_action_mappings[goal_id].append(action_id)
                
                # Record in recent submissions
                self.recent_action_submissions.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "goal_id": goal_id,
                    "action_id": action_id,
                    "action_type": action,
                    "priority": priority.name if hasattr(priority, "name") else str(priority)
                })
                
                # Update metrics
                self.pipeline_metrics["actions_submitted"] += 1
                
                logger.info(f"Submitted action {action_id} of type '{action}' for goal {goal_id}")
                return True
            else:
                logger.warning(f"Failed to create action for goal {goal_id}")
                return False
        except Exception as e:
            logger.error(f"Error submitting action for goal {goal_id}: {e}")
            return False
    
    def _update_pipeline_metrics(self, cycle_results: Dict[str, Any]) -> None:
        """Update pipeline metrics based on cycle results."""
        # Update goals created and actions submitted from cycle
        self.pipeline_metrics["goals_created"] += cycle_results.get("goals_created", 0)
        self.pipeline_metrics["actions_submitted"] += cycle_results.get("actions_submitted", 0)
        
        # Calculate completion rate (success rate of submitted actions)
        successful_actions = sum(1 for outcome in self.action_outcomes.values() if outcome.get("success", False))
        total_actions = len(self.action_outcomes)
        
        self.pipeline_metrics["completion_rate"] = (
            successful_actions / total_actions if total_actions > 0 else 0.0
        )
        
        # Calculate average latency between need triggers and action submissions
        need_triggers = list(self.recent_need_triggers)
        action_submissions = list(self.recent_action_submissions)
        
        if need_triggers and action_submissions:
            # Simple approach - average time between oldest need trigger and newest action submission
            try:
                oldest_need = datetime.datetime.fromisoformat(need_triggers[0]["timestamp"])
                newest_action = datetime.datetime.fromisoformat(action_submissions[-1]["timestamp"])
                
                if len(need_triggers) > 1:
                    latency = (newest_action - oldest_need).total_seconds() / len(need_triggers)
                    self.pipeline_metrics["average_latency"] = latency
            except (ValueError, KeyError):
                # Failed to calculate, keep existing value
                pass
    
    async def _handle_need_state_change(self, event: Event) -> None:
        """
        Handle need state change events from the event bus.
        
        Args:
            event: Need state change event
        """
        try:
            # Extract event data
            need_name = event.data.get("need_name")
            level = event.data.get("level")
            drive_strength = event.data.get("drive_strength")
            
            if not need_name or level is None or drive_strength is None:
                return
            
            # Check if need requires attention
            if drive_strength >= self.need_threshold_for_goal:
                # Check if already triggered
                for trigger in self.recent_need_triggers:
                    if (trigger.get("need_name") == need_name and 
                        (datetime.datetime.now() - datetime.datetime.fromisoformat(trigger["timestamp"])).total_seconds() < 600):  # 10 minutes
                        # Already recently triggered
                        return
                
                # Check if there's an active goal
                if self.goal_manager:
                    has_active_goal = False
                    
                    # Check if goal already exists for this need
                    for goal_id in self.need_goal_mappings.get(need_name, []):
                        goal_status = await self.goal_manager.get_goal_status(goal_id)
                        if goal_status and goal_status.get("status") in ["active", "pending"]:
                            has_active_goal = True
                            break
                    
                    if not has_active_goal:
                        # Calculate priority based on drive strength
                        priority = min(1.0, drive_strength * self.goal_priority_multiplier)
                        
                        # Create a need trigger task
                        asyncio.create_task(
                            self._create_goal_for_need(
                                need_name, 
                                {"level": level, "drive_strength": drive_strength}, 
                                priority
                            )
                        )
        except Exception as e:
            logger.error(f"Error handling need state change event: {e}")
    
    async def _handle_goal_status_change(self, event: Event) -> None:
        """
        Handle goal status change events from the event bus.
        
        Args:
            event: Goal status change event
        """
        try:
            # Extract event data
            goal_id = event.data.get("goal_id")
            status = event.data.get("status")
            
            if not goal_id or not status:
                return
            
            # Handle completed goals
            if status == "completed":
                # Check if need should be satisfied
                need_name = None
                for n_name, goals in self.need_goal_mappings.items():
                    if goal_id in goals:
                        need_name = n_name
                        break
                
                if need_name and self.needs_system:
                    # Satisfy need
                    await self.needs_system.satisfy_need(
                        need_name,
                        0.3,  # Substantial satisfaction from completing a goal
                        context={"goal_id": goal_id, "reason": "goal_completion"}
                    )
                    
                    logger.info(f"Satisfied need '{need_name}' due to completion of goal {goal_id}")
        except Exception as e:
            logger.error(f"Error handling goal status change event: {e}")
    
    async def _handle_action_completed(self, event: Event) -> None:
        """
        Handle action completed events from the event bus.
        
        Args:
            event: Action completed event
        """
        try:
            # Extract event data
            action_id = event.data.get("action_id")
            success = event.data.get("success", False)
            result = event.data.get("result")
            
            if not action_id:
                return
            
            # Provide feedback for action
            asyncio.create_task(
                self.provide_action_feedback(action_id, success, result)
            )
        except Exception as e:
            logger.error(f"Error handling action completed event: {e}")

# Function to create the need-goal-action pipeline
def create_need_goal_action_pipeline(brain_reference=None):
    """Create a need-goal-action pipeline for the given brain."""
    return NeedGoalActionPipeline(brain_reference=brain_reference)
