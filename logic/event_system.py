"""
Event System

This module provides a comprehensive event system that integrates with other game systems
including NPCs, lore, conflicts, and artifacts. It handles event generation, processing,
and propagation throughout the game world.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime

import asyncpg  # Added explicit import for asyncpg

from agents import function_tool, RunContextWrapper, Agent, Runner, trace
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from db.connection import get_db_connection_context  # Updated to use context manager

from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
from logic.artifact_system.artifact_manager import ArtifactManager
from logic.lore.core.system import LoreSystem
from npcs.npc_learning_adaptation import NPCLearningManager
from npcs.belief_system_integration import NPCBeliefSystemIntegration

logger = logging.getLogger(__name__)

class EventSystem:
    """
    Comprehensive event system that handles event generation, processing,
    and propagation throughout the game world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the event system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.is_initialized = False
        
        # Core systems
        self.conflict_resolution = None
        self.artifact_manager = None
        self.lore_system = None
        self.npc_learning = None
        self.belief_system = None
        
        # Event tracking
        self.active_events = {}
        self.event_history = []
        self.event_queue = asyncio.Queue()
        
        # Event processing state
        self.processing_lock = asyncio.Lock()
        self.event_handlers = {}
        
        # Agentic components
        self.event_agent = None
        self.analysis_agent = None
        self.propagation_agent = None
        self.agent_context = None
        self.agent_performance = {}
        self.agent_learning = {}
        
    async def initialize(self):
        """Initialize the event system and its dependencies."""
        if not self.is_initialized:
            # Initialize core systems
            self.conflict_resolution = ConflictResolutionSystem(self.user_id, self.conversation_id)
            await self.conflict_resolution.initialize()
            
            self.artifact_manager = ArtifactManager(self.user_id, self.conversation_id)
            await self.artifact_manager.initialize()
            
            self.lore_system = LoreSystem()
            
            self.npc_learning = NPCLearningManager(self.user_id, self.conversation_id)
            
            self.belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
            
            # Initialize agentic components
            await self._initialize_agents()
            
            # Register default event handlers
            self._register_default_handlers()
            
            # Start event processing loop
            asyncio.create_task(self._process_event_queue())
            
            self.is_initialized = True
            logger.info(f"Event system initialized for user {self.user_id}")
        return self
        
    async def _initialize_agents(self):
        """Initialize the event system agents."""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            
            # Create event agent
            self.event_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_MANAGER,
                agent_id="event_manager",
                capabilities=["event_processing", "event_prioritization", "event_coordination"]
            )
            
            # Create analysis agent
            self.analysis_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_ANALYZER,
                agent_id="event_analyzer",
                capabilities=["event_analysis", "impact_assessment", "pattern_recognition"]
            )
            
            # Create propagation agent
            self.propagation_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_PROPAGATOR,
                agent_id="event_propagator",
                capabilities=["event_propagation", "npc_notification", "system_integration"]
            )
            
            # Initialize agent context
            self.agent_context = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "active_events": self.active_events,
                "event_history": self.event_history,
                "performance_metrics": self.agent_performance,
                "learning_state": self.agent_learning
            }
            
            logger.info("Event system agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing event system agents: {e}")
            raise
            
    def _register_default_handlers(self):
        """Register default event handlers for common event types."""
        self.register_handler("conflict_event", self._handle_conflict_event)
        self.register_handler("artifact_event", self._handle_artifact_event)
        self.register_handler("lore_event", self._handle_lore_event)
        self.register_handler("npc_event", self._handle_npc_event)
        self.register_handler("relationship_event", self._handle_relationship_event)
        self.register_handler("time_event", self._handle_time_event)
        
    def register_handler(self, event_type: str, handler: callable):
        """Register a handler for a specific event type."""
        self.event_handlers[event_type] = handler
        
    @with_governance(
        agent_type=AgentType.EVENT_MANAGER,
        action_type="create_event",
        action_description="Creating a new event in the system",
        id_from_context=lambda ctx: "event_manager"
    )
    async def create_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        priority: int = 0
    ) -> Dict[str, Any]:
        """
        Create and queue a new event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            priority: Event priority (higher numbers = higher priority)
            
        Returns:
            Created event details
        """
        try:
            # Generate event ID
            event_id = f"event_{len(self.active_events) + 1}"
            
            # Create event object
            event = {
                "id": event_id,
                "type": event_type,
                "data": event_data,
                "priority": priority,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "processed_at": None,
                "results": {}
            }
            
            # Analyze event with analysis agent
            analysis_result = await self._analyze_event(event)
            event["analysis"] = analysis_result
            
            # Add to active events
            self.active_events[event_id] = event
            
            # Queue for processing
            await self.event_queue.put((priority, event))
            
            # Update agent performance
            self._update_agent_performance("event_creation", True)
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            self._update_agent_performance("event_creation", False)
            return {"error": str(e)}
            
    async def _analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an event using the analysis agent."""
        try:
            # Create analysis context
            analysis_context = {
                "event": event,
                "system_state": self.agent_context,
                "historical_patterns": self._get_historical_patterns(event["type"])
            }
            
            # Get analysis from agent
            analysis = await self.analysis_agent.analyze(
                context=analysis_context,
                capabilities=["event_analysis", "impact_assessment"]
            )
            
            # Update agent learning
            self._update_agent_learning("event_analysis", analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            return {"error": str(e)}
            
    async def _process_event_queue(self):
        """Process events from the queue."""
        while True:
            try:
                # Get next event from queue
                priority, event = await self.event_queue.get()
                
                # Process event with event agent
                await self._process_event(event)
                
                # Propagate event with propagation agent
                await self._propagate_event(event)
                
                # Mark as processed
                event["processed_at"] = datetime.utcnow().isoformat()
                event["status"] = "processed"
                
                # Add to history
                self.event_history.append(event)
                
                # Clean up old history
                if len(self.event_history) > 1000:
                    self.event_history = self.event_history[-1000:]
                    
                # Update agent performance
                self._update_agent_performance("event_processing", True)
                    
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
                self._update_agent_performance("event_processing", False)
                
            finally:
                self.event_queue.task_done()
                
    async def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event using the event agent."""
        try:
            # Get event handler
            handler = self.event_handlers.get(event["type"])
            if not handler:
                logger.warning(f"No handler registered for event type: {event['type']}")
                return {"error": f"No handler for event type: {event['type']}"}
                
            # Process with governance and event agent
            async with self.processing_lock:
                # Create processing context
                processing_context = {
                    "event": event,
                    "system_state": self.agent_context,
                    "handler": handler.__name__
                }
                
                # Get processing guidance from event agent
                guidance = await self.event_agent.guide(
                    context=processing_context,
                    capabilities=["event_processing", "event_coordination"]
                )
                
                # Execute handler with guidance
                result = await handler(event, guidance)
                
            # Store results
            event["results"] = result
            
            # Update agent learning
            self._update_agent_learning("event_processing", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return {"error": str(e)}
            
    async def _propagate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate an event using the propagation agent."""
        try:
            # Create propagation context
            propagation_context = {
                "event": event,
                "system_state": self.agent_context,
                "affected_systems": self._get_affected_systems(event)
            }
            
            # Get propagation plan from agent
            propagation_plan = await self.propagation_agent.plan(
                context=propagation_context,
                capabilities=["event_propagation", "system_integration"]
            )
            
            # Execute propagation
            result = await self._execute_propagation(event, propagation_plan)
            
            # Update agent learning
            self._update_agent_learning("event_propagation", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error propagating event: {e}")
            return {"error": str(e)}
            
    def _get_affected_systems(self, event: Dict[str, Any]) -> List[str]:
        """Determine which systems are affected by an event."""
        affected_systems = []
        
        # Check event type and data
        event_type = event["type"]
        event_data = event["data"]
        
        if event_type in ["conflict_event", "artifact_event", "lore_event", "npc_event"]:
            affected_systems.append(event_type.split("_")[0])
            
        # Check for cross-system effects
        if "cross_system_effects" in event_data:
            affected_systems.extend(event_data["cross_system_effects"])
            
        return list(set(affected_systems))
        
    def _get_historical_patterns(self, event_type: str) -> List[Dict[str, Any]]:
        """Get historical patterns for an event type."""
        patterns = []
        
        # Filter history by event type
        type_history = [e for e in self.event_history if e["type"] == event_type]
        
        # Analyze patterns
        if type_history:
            # Get success patterns
            success_patterns = [e for e in type_history if "error" not in e.get("results", {})]
            if success_patterns:
                patterns.append({
                    "type": "success",
                    "count": len(success_patterns),
                    "examples": success_patterns[-3:]
                })
                
            # Get failure patterns
            failure_patterns = [e for e in type_history if "error" in e.get("results", {})]
            if failure_patterns:
                patterns.append({
                    "type": "failure",
                    "count": len(failure_patterns),
                    "examples": failure_patterns[-3:]
                })
                
        return patterns
        
    def _update_agent_performance(self, action: str, success: bool):
        """Update agent performance metrics."""
        if action not in self.agent_performance:
            self.agent_performance[action] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0
            }
            
        metrics = self.agent_performance[action]
        metrics["total"] += 1
        
        if success:
            metrics["successful"] += 1
        else:
            metrics["failed"] += 1
            
        metrics["success_rate"] = metrics["successful"] / metrics["total"]
        
    def _update_agent_learning(self, action: str, result: Dict[str, Any]):
        """Update agent learning state."""
        if action not in self.agent_learning:
            self.agent_learning[action] = {
                "patterns": [],
                "strategies": {},
                "adaptations": []
            }
            
        learning = self.agent_learning[action]
        
        # Extract patterns
        if "patterns" in result:
            learning["patterns"].extend(result["patterns"])
            
        # Update strategies
        if "strategy" in result:
            strategy = result["strategy"]
            if strategy not in learning["strategies"]:
                learning["strategies"][strategy] = {
                    "uses": 0,
                    "successes": 0,
                    "failures": 0
                }
            learning["strategies"][strategy]["uses"] += 1
            if "error" not in result:
                learning["strategies"][strategy]["successes"] += 1
            else:
                learning["strategies"][strategy]["failures"] += 1
                
        # Record adaptations
        if "adaptation" in result:
            learning["adaptations"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "details": result["adaptation"]
            })
            
    async def _handle_conflict_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a conflict-related event."""
        try:
            event_data = event["data"]
            conflict_id = event_data.get("conflict_id")
            
            if not conflict_id:
                return {"error": "No conflict ID provided"}
                
            # Get conflict details
            conflict = await self.conflict_resolution.get_conflict_details(conflict_id)
            if not conflict:
                return {"error": f"Conflict {conflict_id} not found"}
                
            # Process conflict event
            result = await self.conflict_resolution.process_conflict_event(
                conflict_id,
                event_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling conflict event: {e}")
            return {"error": str(e)}
            
    async def _handle_artifact_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an artifact-related event."""
        try:
            event_data = event["data"]
            artifact_id = event_data.get("artifact_id")
            
            if not artifact_id:
                return {"error": "No artifact ID provided"}
                
            # Get artifact details
            artifact = self.artifact_manager.active_artifacts.get(artifact_id)
            if not artifact:
                return {"error": f"Artifact {artifact_id} not found"}
                
            # Process artifact event
            result = await self.artifact_manager.process_artifact_event(
                artifact_id,
                event_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling artifact event: {e}")
            return {"error": str(e)}
            
    async def _handle_lore_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a lore-related event."""
        try:
            event_data = event["data"]
            
            # Process lore event
            result = await self.lore_system.process_lore_event(
                self.user_id,
                self.conversation_id,
                event_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling lore event: {e}")
            return {"error": str(e)}
            
    async def _handle_npc_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an NPC-related event."""
        try:
            event_data = event["data"]
            npc_ids = event_data.get("npc_ids", [])
            
            if not npc_ids:
                return {"error": "No NPC IDs provided"}
                
            # Process learning
            learning_result = await self.npc_learning.process_event_for_learning(
                event_data.get("event_text", ""),
                event_data.get("event_type", "unknown"),
                npc_ids,
                event_data.get("player_response")
            )
            
            # Process beliefs
            belief_result = await self.belief_system.process_event_for_beliefs(
                event_data.get("event_text", ""),
                event_data.get("event_type", "unknown"),
                npc_ids,
                event_data.get("factuality", 1.0)
            )
            
            return {
                "learning": learning_result,
                "beliefs": belief_result
            }
            
        except Exception as e:
            logger.error(f"Error handling NPC event: {e}")
            return {"error": str(e)}
            
    async def _handle_relationship_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a relationship-related event."""
        try:
            event_data = event["data"]
            
            # Use the new relationship system
            manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            
            # Process interaction or update
            if event_data.get("interaction_type"):
                result = await manager.process_interaction(
                    entity1_type="player",
                    entity1_id=1,  # Assuming player ID is 1
                    entity2_type=event_data.get("entity_type"),
                    entity2_id=event_data.get("entity_id"),
                    interaction={"type": event_data.get("interaction_type")}
                )
            else:
                # Direct relationship update
                state = await manager.get_relationship_state(
                    entity1_type="player",
                    entity1_id=1,
                    entity2_type=event_data.get("entity_type"),
                    entity2_id=event_data.get("entity_id")
                )
                # Apply dimension changes
                if event_data.get("dimensions"):
                    for dim, value in event_data["dimensions"].items():
                        if hasattr(state.dimensions, dim):
                            setattr(state.dimensions, dim, value)
                    state.dimensions.clamp()
                    await manager._queue_update(state)
                result = {"success": True, "state": state.to_summary()}
            
    async def _handle_time_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a time-related event."""
        try:
            event_data = event["data"]
            time_amount = event_data.get("time_amount", 0)
            time_unit = event_data.get("time_unit", "hours")
            
            if not time_amount:
                return {"error": "No time amount provided"}
                
            # Process time event
            result = await self.lore_system.advance_time(
                self.user_id,
                self.conversation_id,
                time_amount,
                time_unit
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling time event: {e}")
            return {"error": str(e)}
            
    async def get_active_events(self) -> List[Dict[str, Any]]:
        """Get all currently active events."""
        return list(self.active_events.values())
        
    async def get_event_history(
        self,
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get event history with optional filtering.
        
        Args:
            limit: Maximum number of events to return
            event_type: Optional event type to filter by
            
        Returns:
            List of historical events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e["type"] == event_type]
            
        return events[-limit:]
        
    async def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific event."""
        return self.active_events.get(event_id)
        
    async def cancel_event(self, event_id: str) -> Dict[str, Any]:
        """Cancel a queued event."""
        try:
            event = self.active_events.get(event_id)
            if not event:
                return {"error": f"Event {event_id} not found"}
                
            if event["status"] != "queued":
                return {"error": f"Event {event_id} is not queued"}
                
            # Remove from active events
            del self.active_events[event_id]
            
            return {"success": True, "message": f"Event {event_id} cancelled"}
            
        except Exception as e:
            logger.error(f"Error cancelling event: {e}")
            return {"error": str(e)}
            
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about event processing."""
        try:
            total_events = len(self.event_history)
            event_types = {}
            
            # Count events by type
            for event in self.event_history:
                event_type = event["type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1
                
            # Calculate average processing time
            processing_times = []
            for event in self.event_history:
                if event["processed_at"]:
                    created = datetime.fromisoformat(event["created_at"])
                    processed = datetime.fromisoformat(event["processed_at"])
                    processing_times.append((processed - created).total_seconds())
                    
            avg_processing_time = (
                sum(processing_times) / len(processing_times)
                if processing_times else 0
            )
            
            return {
                "total_events": total_events,
                "events_by_type": event_types,
                "average_processing_time": avg_processing_time,
                "active_events": len(self.active_events),
                "queued_events": self.event_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {"error": str(e)}
