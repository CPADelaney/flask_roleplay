# nyx/core/integration/adaptation_goal_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method
from nyx.core.integration.action_selector import ActionPriority

logger = logging.getLogger(__name__)

class CoreSystemsIntegrationBridge:
    """
    Integration bridge connecting core systems (DynamicAdaptationSystem, GoalSystem, 
    and IssueTrackingSystem) with the rest of the Nyx architecture.
    
    This bridge facilitates communication between these core systems and other modules
    through the event bus, providing standardized interfaces and event handling.
    
    Key functions:
    1. Connects core systems to the event bus
    2. Facilitates cross-module communication and data transformation
    3. Provides unified interfaces for accessing core system functionality
    4. Propagates events between core systems and other modules
    5. Handles synchronization and state management between systems
    """
    
    def __init__(self, 
                dynamic_adaptation_system=None,
                goal_system=None,
                issue_tracking_system=None):
        """Initialize the core systems integration bridge."""
        self.dynamic_adaptation_system = dynamic_adaptation_system
        self.goal_system = goal_system
        self.issue_tracking_system = issue_tracking_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.adaptation_cycle_interval = 60  # Seconds between adaptation cycles
        self.goal_execution_interval = 5     # Seconds between goal execution steps
        self.issue_report_threshold = 0.7    # Threshold for auto-reporting issues
        
        # Integration state tracking
        self._subscribed = False
        self._adaptation_task = None
        self._goal_execution_task = None
        
        # Cached state
        self.current_adaptation_strategy = None
        self.active_goals = {}  # goal_id -> goal_data
        self.recent_issues = []  # List of recent issues
        
        logger.info("CoreSystemsIntegrationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("context_change", self._handle_context_change)
                self.event_bus.subscribe("performance_metrics", self._handle_performance_metrics)
                self.event_bus.subscribe("system_error", self._handle_system_error)
                self.event_bus.subscribe("need_state_change", self._handle_need_state_change)
                self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
                self._subscribed = True
            
            # Start background tasks
            if self.dynamic_adaptation_system:
                self._adaptation_task = asyncio.create_task(self._run_adaptation_cycle())
            
            if self.goal_system:
                self._goal_execution_task = asyncio.create_task(self._run_goal_execution())
            
            logger.info("CoreSystemsIntegrationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing CoreSystemsIntegrationBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def adapt_to_context(self, 
                          context: Dict[str, Any],
                          performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Adapt to a new context using the DynamicAdaptationSystem.
        
        Args:
            context: Current context information
            performance_metrics: Optional performance metrics
            
        Returns:
            Adaptation results
        """
        if not self.dynamic_adaptation_system:
            return {"status": "error", "message": "DynamicAdaptationSystem not available"}
        
        try:
            # Set default performance metrics if not provided
            if not performance_metrics:
                performance_metrics = {
                    "success_rate": self.system_context.get_value("success_rate", 0.5),
                    "response_time": self.system_context.get_value("avg_response_time", 1.0),
                    "user_satisfaction": self.system_context.get_value("user_satisfaction", 0.5)
                }
            
            # Run adaptation cycle
            result = await self.dynamic_adaptation_system.adaptation_cycle(context, performance_metrics)
            
            # Update cached strategy
            if "selected_strategy" in result:
                self.current_adaptation_strategy = result["selected_strategy"]
                
                # Update system context
                self.system_context.set_value("current_adaptation_strategy", self.current_adaptation_strategy)
                
                # Broadcast event about strategy change
                event = Event(
                    event_type="adaptation_strategy_changed",
                    source="core_systems_integration_bridge",
                    data={
                        "strategy_id": result["selected_strategy"]["id"],
                        "strategy_name": result["selected_strategy"]["name"],
                        "context_summary": {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
                    }
                )
                await self.event_bus.publish(event)
            
            return {
                "status": "success",
                "adaptation_result": result,
                "strategy_changed": "selected_strategy" in result
            }
        except Exception as e:
            logger.error(f"Error adapting to context: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def create_goal_from_need(self, 
                               need_name: str,
                               priority: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a goal to satisfy a need using the GoalSystem.
        
        Args:
            need_name: Name of the need to create goal for
            priority: Optional priority override (0.0-1.0)
            
        Returns:
            Goal creation results
        """
        if not self.goal_system:
            return {"status": "error", "message": "GoalSystem not available"}
        
        try:
            # Check if there's already an active goal for this need
            if self.goal_system.has_active_goal_for_need(need_name):
                return {
                    "status": "skipped",
                    "message": f"An active goal already exists for need '{need_name}'"
                }
            
            # Get need information from system context
            need_states = self.system_context.need_states
            need_info = need_states.get(need_name, {})
            
            # Set default priority if not provided
            if priority is None:
                # Calculate from need drive
                drive_strength = need_info.get("drive_strength", 0.5)
                priority = min(1.0, drive_strength * 1.2)  # Scale drive to priority
            
            # Create goal description based on need
            description = f"Satisfy {need_name} need"
            
            # Add goal to goal system
            goal_id = await self.goal_system.add_goal(
                description=description,
                priority=priority,
                source="CoreSystemsIntegrationBridge",
                associated_need=need_name
            )
            
            if goal_id:
                # Add to tracked goals
                goal_status = await self.goal_system.get_goal_status(goal_id)
                if goal_status:
                    self.active_goals[goal_id] = goal_status
                
                # Broadcast event about goal creation
                event = Event(
                    event_type="goal_created_for_need",
                    source="core_systems_integration_bridge",
                    data={
                        "goal_id": goal_id,
                        "need_name": need_name,
                        "priority": priority,
                        "description": description
                    }
                )
                await self.event_bus.publish(event)
                
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
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def report_issue(self, 
                       title: str,
                       description: str,
                       category: str,
                       priority: int = 3,
                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Report an issue using the IssueTrackingSystem.
        
        Args:
            title: Issue title
            description: Issue description
            category: Issue category (bug, efficiency, enhancement, etc.)
            priority: Issue priority (1-5)
            context: Optional context information
            
        Returns:
            Issue reporting results
        """
        if not self.issue_tracking_system:
            return {"status": "error", "message": "IssueTrackingSystem not available"}
        
        try:
            # Add issue directly
            result = await self.issue_tracking_system.add_issue_directly(
                title=title,
                description=description,
                category=category,
                priority=priority,
                tags=["auto_reported", "core_systems_bridge"]
            )
            
            if result.success:
                # Add to recent issues
                self.recent_issues.append({
                    "issue_id": result.issue_id,
                    "title": title,
                    "category": category,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Trim recent issues list
                if len(self.recent_issues) > 10:
                    self.recent_issues = self.recent_issues[-10:]
                
                # Broadcast event about issue creation
                event = Event(
                    event_type="issue_reported",
                    source="core_systems_integration_bridge",
                    data={
                        "issue_id": result.issue_id,
                        "title": title,
                        "category": category,
                        "priority": priority
                    }
                )
                await self.event_bus.publish(event)
                
                return {
                    "status": "success",
                    "issue_id": result.issue_id,
                    "message": result.message
                }
            else:
                return {"status": "error", "message": result.message}
        except Exception as e:
            logger.error(f"Error reporting issue: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def process_observation_for_issues(self, 
                                        observation: str,
                                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an observation to detect and report issues.
        
        Args:
            observation: Observation text
            context: Optional context information
            
        Returns:
            Processing results
        """
        if not self.issue_tracking_system:
            return {"status": "error", "message": "IssueTrackingSystem not available"}
        
        try:
            # Process the observation
            result = await self.issue_tracking_system.process_observation(observation, context)
            
            # Return results
            return {
                "status": "success",
                "observation_processed": True,
                "analysis": result.get("analysis"),
                "processing_result": result.get("processing_result")
            }
        except Exception as e:
            logger.error(f"Error processing observation for issues: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def execute_goal_step(self, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a step for a goal, or the highest priority goal if none specified.
        
        Args:
            goal_id: Optional specific goal ID to execute a step for
            
        Returns:
            Step execution results
        """
        if not self.goal_system:
            return {"status": "error", "message": "GoalSystem not available"}
        
        try:
            # If no goal specified, let goal system select one
            if goal_id is None:
                step_result = await self.goal_system.execute_next_step()
            else:
                # Set as active goal and execute
                # This is a simplification - in a real implementation, you'd want to
                # set the selected goal and then call execute_next_step
                step_result = await self.goal_system.execute_next_step()
            
            if not step_result:
                return {
                    "status": "no_action",
                    "message": "No step executed, possibly no active goals"
                }
            
            # Broadcast event about step execution
            event = Event(
                event_type="goal_step_executed",
                source="core_systems_integration_bridge",
                data=step_result
            )
            await self.event_bus.publish(event)
            
            # Update active goals
            executed_goal_id = step_result.get("goal_id")
            if executed_goal_id:
                # Get updated goal status
                goal_status = await self.goal_system.get_goal_status(executed_goal_id)
                if goal_status:
                    self.active_goals[executed_goal_id] = goal_status
            
            return {
                "status": "success",
                "step_result": step_result
            }
        except Exception as e:
            logger.error(f"Error executing goal step: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="CoreSystems")
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the current status of core systems integration.
        
        Returns:
            Status information for all core systems
        """
        try:
            # Get dynamic adaptation system status
            adaptation_status = {}
            if self.dynamic_adaptation_system:
                # Get current strategy
                strategy_id = self.dynamic_adaptation_system.context.current_strategy_id
                strategy = self.dynamic_adaptation_system.context.strategies.get(strategy_id, {})
                
                adaptation_status = {
                    "current_strategy": {
                        "id": strategy_id,
                        "name": strategy.get("name", "Unknown"),
                        "description": strategy.get("description", "")
                    },
                    "cycle_count": self.dynamic_adaptation_system.context.cycle_count,
                    "available_strategies": list(self.dynamic_adaptation_system.context.strategies.keys())
                }
            
            # Get goal system status
            goal_status = {}
            if self.goal_system:
                # Get active goals
                active_goals = await self.goal_system.get_all_goals(status_filter=["active", "pending"])
                
                # Get statistics
                goal_stats = await self.goal_system.get_goal_statistics()
                
                goal_status = {
                    "active_goals_count": len(active_goals),
                    "active_goals": [
                        {
                            "id": goal.get("id"),
                            "description": goal.get("description", ""),
                            "status": goal.get("status", "unknown"),
                            "priority": goal.get("priority", 0)
                        }
                        for goal in active_goals[:5]  # Top 5 for brevity
                    ],
                    "success_rate": goal_stats.get("success_rate", 0),
                    "total_goals": goal_stats.get("total_goals_created", 0)
                }
            
            # Get issue tracking system status
            issue_status = {}
            if self.issue_tracking_system:
                # Get summary
                summary = await self.issue_tracking_system.get_issue_summary()
                
                issue_status = {
                    "total_issues": summary.get("stats", {}).get("total_issues", 0),
                    "open_issues": summary.get("stats", {}).get("open_issues", 0),
                    "issues_by_category": summary.get("stats", {}).get("by_category", {}),
                    "recent_issues": self.recent_issues
                }
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "adaptation_system": adaptation_status,
                "goal_system": goal_status,
                "issue_tracking_system": issue_status,
                "event_subscriptions": self._subscribed
            }
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _run_adaptation_cycle(self) -> None:
        """Run periodic adaptation cycles."""
        while True:
            try:
                # Get current context from system context
                context = {
                    "affective_state": {
                        "primary_emotion": self.system_context.affective_state.primary_emotion,
                        "valence": self.system_context.affective_state.valence,
                        "arousal": self.system_context.affective_state.arousal
                    },
                    "active_goals": len(self.active_goals),
                    "cycle_time": datetime.datetime.now().isoformat()
                }
                
                # Get performance metrics from system context
                performance_metrics = {
                    "success_rate": self.system_context.get_value("success_rate", 0.5),
                    "response_time": self.system_context.get_value("avg_response_time", 1.0),
                    "user_satisfaction": self.system_context.get_value("user_satisfaction", 0.5)
                }
                
                # Run adaptation cycle
                await self.adapt_to_context(context, performance_metrics)
                
            except Exception as e:
                logger.error(f"Error in adaptation cycle: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.adaptation_cycle_interval)
    
    async def _run_goal_execution(self) -> None:
        """Run periodic goal execution steps."""
        while True:
            try:
                # Execute next goal step
                await self.execute_goal_step()
                
            except Exception as e:
                logger.error(f"Error in goal execution: {e}")
            
            # Wait for next execution
            await asyncio.sleep(self.goal_execution_interval)
    
    async def _handle_context_change(self, event: Event) -> None:
        """
        Handle context change events.
        
        Args:
            event: Context change event
        """
        try:
            # Extract context data
            context = event.data
            
            # Run adaptation cycle with new context
            await self.adapt_to_context(context)
            
        except Exception as e:
            logger.error(f"Error handling context change: {e}")
    
    async def _handle_performance_metrics(self, event: Event) -> None:
        """
        Handle performance metrics events.
        
        Args:
            event: Performance metrics event
        """
        try:
            # Extract metrics data
            metrics = event.data
            
            # Get current context
            context = {
                "affective_state": {
                    "primary_emotion": self.system_context.affective_state.primary_emotion,
                    "valence": self.system_context.affective_state.valence,
                    "arousal": self.system_context.affective_state.arousal
                },
                "performance_event": True,
                "cycle_time": datetime.datetime.now().isoformat()
            }
            
            # Run adaptation cycle with metrics
            await self.adapt_to_context(context, metrics)
            
        except Exception as e:
            logger.error(f"Error handling performance metrics: {e}")
    
    async def _handle_system_error(self, event: Event) -> None:
        """
        Handle system error events.
        
        Args:
            event: System error event
        """
        try:
            # Extract error data
            error = event.data.get("error")
            source = event.data.get("source", "unknown")
            details = event.data.get("details", {})
            
            if not error:
                return
            
            # Determine issue severity based on error details
            severity = details.get("severity", 3)  # Default medium severity
            
            # Generate error title and description
            title = f"Error in {source}: {error[:50]}"
            description = f"""Error detected in module: {source}

Error message: {error}

Details: {json.dumps(details, indent=2)}

Timestamp: {datetime.datetime.now().isoformat()}
"""
            
            # Report issue
            await self.report_issue(
                title=title,
                description=description,
                category="bug",
                priority=severity
            )
            
        except Exception as e:
            logger.error(f"Error handling system error: {e}")
    
    async def _handle_need_state_change(self, event: Event) -> None:
        """
        Handle need state change events.
        
        Args:
            event: Need state change event
        """
        try:
            # Extract need data
            need_name = event.data.get("need_name")
            level = event.data.get("level")
            drive_strength = event.data.get("drive_strength")
            
            if not need_name or level is None or drive_strength is None:
                return
            
            # Check if drive is high enough to trigger goal creation
            if drive_strength >= 0.7:  # High drive
                # Create goal for need
                await self.create_goal_from_need(need_name, priority=drive_strength)
            
        except Exception as e:
            logger.error(f"Error handling need state change: {e}")
    
    async def _handle_user_feedback(self, event: Event) -> None:
        """
        Handle user feedback events.
        
        Args:
            event: User feedback event
        """
        try:
            # Extract feedback data
            feedback = event.data.get("feedback")
            rating = event.data.get("rating")
            
            if not feedback:
                return
            
            # Check if feedback contains issue reports
            if "bug" in feedback.lower() or "issue" in feedback.lower() or "problem" in feedback.lower():
                # Process as observation for issues
                await self.process_observation_for_issues(
                    observation=feedback,
                    context="User feedback"
                )
            
            # Update user satisfaction in system context for adaptation
            if rating is not None:
                # Convert to normalized value (0.0-1.0)
                if isinstance(rating, (int, float)):
                    # Direct value
                    normalized_rating = max(0.0, min(1.0, rating))
                elif isinstance(rating, str):
                    # String rating
                    rating_mapping = {
                        "excellent": 1.0, "good": 0.8, "positive": 0.7,
                        "neutral": 0.5,
                        "negative": 0.3, "bad": 0.2, "terrible": 0.0
                    }
                    normalized_rating = rating_mapping.get(rating.lower(), 0.5)
                else:
                    normalized_rating = 0.5  # Default neutral
                
                # Update system context
                self.system_context.set_value("user_satisfaction", normalized_rating)
            
        except Exception as e:
            logger.error(f"Error handling user feedback: {e}")

# Function to create the bridge
def create_core_systems_integration_bridge(nyx_brain):
    """Create a core systems integration bridge for the given brain."""
    return CoreSystemsIntegrationBridge(
        dynamic_adaptation_system=nyx_brain.dynamic_adaptation_system if hasattr(nyx_brain, "dynamic_adaptation_system") else None,
        goal_system=nyx_brain.goal_manager if hasattr(nyx_brain, "goal_manager") else None,
        issue_tracking_system=nyx_brain.issue_tracking_system if hasattr(nyx_brain, "issue_tracking_system") else None
    )
