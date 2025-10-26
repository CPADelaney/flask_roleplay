"""
Event System

This module provides a comprehensive event system that integrates with other game systems
including NPCs, lore, conflicts (via new synthesizer), and artifacts.

REFACTORED: Updated to use new ConflictSynthesizer instead of ConflictResolutionSystem
REFACTORED: Added proper cleanup and shutdown handling for worker tasks
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
import asyncio
from datetime import datetime
from itertools import count
from uuid import uuid4
from contextlib import asynccontextmanager
import weakref

import asyncpg

from agents import function_tool, RunContextWrapper, Agent, Runner, trace
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from db.connection import get_db_connection_context

# NEW: Import the conflict synthesizer
from logic.conflict_system.conflict_synthesizer import get_synthesizer, ConflictSynthesizer

from logic.dynamic_relationships import OptimizedRelationshipManager
from logic.artifact_system.artifact_manager import ArtifactManager
from logic.event_logging import log_event
from logic.game_time_helper import GameTimeContext
from npcs.npc_learning_adaptation import NPCLearningManager
from npcs.belief_system_integration import NPCBeliefSystemIntegration

logger = logging.getLogger(__name__)

# Global registry for active event systems (for cleanup during app shutdown)
_active_event_systems: Set[weakref.ref] = set()

def register_event_system(event_system: 'EventSystem'):
    """Register an active event system for cleanup."""
    _active_event_systems.add(weakref.ref(event_system))

def unregister_event_system(event_system: 'EventSystem'):
    """Unregister an event system after cleanup."""
    _active_event_systems.discard(weakref.ref(event_system))

async def cleanup_all_event_systems():
    """Clean up all active event systems during app shutdown."""
    tasks = []
    for ref in list(_active_event_systems):
        event_system = ref()
        if event_system:
            tasks.append(event_system.shutdown(drain=False, timeout=1.0))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    _active_event_systems.clear()
    logger.info("All event systems cleaned up")


class EventSystem:
    """
    Comprehensive event system that handles event generation, processing,
    and propagation throughout the game world, now with new conflict synthesizer.

    Public API (unchanged):
      - await initialize()
      - await create_event(event_type, event_data, priority=0) -> Dict[str, Any]
      - register_handler(event_type, handler)
      - await get_active_events()
      - await get_event_history(limit=100, event_type=None)
      - await get_event_details(event_id)
      - await cancel_event(event_id)
      - await get_event_statistics()
      - await shutdown(drain: bool = False)
    """

    # ---- construction & core state -------------------------------------------------

    def __init__(self, user_id: int, conversation_id: int, *, num_workers: int = 1, max_history: int = 1000):
        self.user_id = user_id
        self.conversation_id = conversation_id

        # init flags/locks
        self.is_initialized = False
        self._init_lock = asyncio.Lock()
        
        # Shutdown management
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False

        # worker pool
        self.num_workers = max(1, int(num_workers))
        self._worker_tasks: List[asyncio.Task] = []

        # core systems - UPDATED
        self.conflict_synthesizer: Optional[ConflictSynthesizer] = None  # NEW
        self.artifact_manager: Optional[ArtifactManager] = None
        self.lore_system = None
        self.npc_learning: Optional[NPCLearningManager] = None
        self.belief_system: Optional[NPCBeliefSystemIntegration] = None
        self.relationship_manager: Optional[OptimizedRelationshipManager] = None

        # event tracking
        self.active_events: Dict[str, Dict[str, Any]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = max_history

        # priority queue (higher priority first)
        self.event_queue: asyncio.PriorityQueue[Tuple[int, int, Dict[str, Any]]] = asyncio.PriorityQueue()
        self._counter = count()  # tie-breaker for stable ordering

        # processing coordination
        self.processing_lock = asyncio.Lock()
        self.event_handlers: Dict[str, Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any]] = {}

        # agentic parts
        self.event_agent = None
        self.analysis_agent = None
        self.propagation_agent = None
        self.agent_context = None
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.agent_learning: Dict[str, Dict[str, Any]] = {}

    def __del__(self):
        """Ensure workers are cancelled if object is destroyed."""
        if hasattr(self, '_worker_tasks') and self._worker_tasks:
            for task in self._worker_tasks:
                if not task.done():
                    try:
                        task.cancel()
                    except Exception:
                        pass

    # ---- lifecycle ----------------------------------------------------------------

    async def initialize(self) -> "EventSystem":
        """Initialize the event system and its dependencies, and start workers."""
        async with self._init_lock:
            if self.is_initialized:
                return self

            # Initialize conflict synthesizer - NEW
            self.conflict_synthesizer = await get_synthesizer(self.user_id, self.conversation_id)
            logger.info(f"Initialized conflict synthesizer for event system")

            # Initialize artifact manager
            self.artifact_manager = ArtifactManager(self.user_id, self.conversation_id)
            await self.artifact_manager.initialize()

            # Import LoreSystem lazily to avoid circular deps
            try:
                from logic.lore.core.system import LoreSystem
                self.lore_system = LoreSystem()
            except ImportError:
                logger.warning("LoreSystem not available")
                self.lore_system = None

            # Initialize relationship and NPC systems
            self.relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            self.npc_learning = NPCLearningManager(self.user_id, self.conversation_id)
            self.belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)

            # Initialize agents (best effort)
            await self._initialize_agents()

            # Register handlers
            self._register_default_handlers()

            # Spawn worker tasks (guarded; idempotent)
            self._start_workers()
            
            # Register for global cleanup
            register_event_system(self)

            self.is_initialized = True
            logger.info(f"Event system initialized for user={self.user_id} convo={self.conversation_id} workers={len(self._worker_tasks)}")
        return self

    def _start_workers(self):
        """Start exactly num_workers workers if not already running."""
        alive = [t for t in self._worker_tasks if not t.done()]
        missing = self.num_workers - len(alive)
        for i in range(max(0, missing)):
            task = asyncio.create_task(
                self._process_event_queue(), 
                name=f"EventSystemWorker-{self.user_id}-{self.conversation_id}-{i}"
            )
            self._worker_tasks.append(task)

    async def shutdown(self, drain: bool = False, timeout: float = 5.0):
        """
        Stop worker tasks. If drain=True, wait for the queue to empty (best effort).
        Call this during application shutdown.
        """
        if self._is_shutting_down:
            return  # Prevent double shutdown
        
        self._is_shutting_down = True
        self._shutdown_event.set()  # Signal workers to stop
        
        logger.info(f"Shutting down EventSystem for user={self.user_id} convo={self.conversation_id}")
        
        if drain and not self.event_queue.empty():
            try:
                logger.info(f"Draining {self.event_queue.qsize()} pending events...")
                await asyncio.wait_for(self.event_queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"EventSystem drain timeout after {timeout}s; continuing shutdown")

        # Cancel workers gracefully
        for t in self._worker_tasks:
            if not t.done():
                t.cancel()

        # Wait for cancellation to complete
        if self._worker_tasks:
            results = await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error(f"Worker {i} ended with error during shutdown: {result}")

        self._worker_tasks.clear()
        
        # Unregister from global cleanup
        unregister_event_system(self)
        
        logger.info(f"EventSystem shutdown complete for user={self.user_id} convo={self.conversation_id}")

    # ---- agents -------------------------------------------------------------------

    async def _initialize_agents(self):
        """Initialize the event system agents (best effort)."""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)

            self.event_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_MANAGER,
                agent_id="event_manager",
                model="gpt-5-nano",
            )

            self.analysis_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_ANALYZER,
                agent_id="event_analyzer",
                model="gpt-5-nano",
            )

            self.propagation_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_PROPAGATOR,
                agent_id="event_propagator",
                model="gpt-5-nano",
            )

            self.agent_context = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "active_events": self.active_events,
                "event_history": self.event_history,
                "performance_metrics": self.agent_performance,
                "learning_state": self.agent_learning,
            }

            logger.info("Event system agents initialized")
        except Exception as e:
            logger.error(f"Error initializing event system agents: {e}")
            logger.warning("Continuing without event system agents")

    # ---- handlers -----------------------------------------------------------------

    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_handler("conflict_event", self._handle_conflict_event)
        self.register_handler("conflict_created", self._handle_conflict_created)
        self.register_handler("conflict_resolved", self._handle_conflict_resolved)
        self.register_handler("conflict_scene", self._handle_conflict_scene)
        self.register_handler("artifact_event", self._handle_artifact_event)
        self.register_handler("lore_event", self._handle_lore_event)
        self.register_handler("npc_event", self._handle_npc_event)
        self.register_handler("relationship_event", self._handle_relationship_event)
        self.register_handler("time_event", self._handle_time_event)

    def register_handler(self, event_type: str, handler: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any]):
        """Register a handler for a specific event type."""
        self.event_handlers[event_type] = handler

    # ---- event creation / queueing ------------------------------------------------

    async def create_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        Create and queue a new event.

        Args:
            event_type: Type of event
            event_data: Event payload
            priority: Higher numbers = higher priority

        Returns:
            Created event details
        """
        try:
            event_id = f"{event_type}-{uuid4().hex[:10]}"

            base_event = {
                "id": event_id,
                "data": event_data,
                "priority": int(priority),
                "status": "queued",
                "processed_at": None,
                "results": {},
            }
            event = await log_event(
                self.user_id,
                self.conversation_id,
                event_type,
                base_event,
            )
            event["type"] = event.pop("event_type")
            event["created_at"] = event["timestamp"].isoformat()

            if self.analysis_agent:
                analysis_result = await self._analyze_event(event)
                event["analysis"] = analysis_result

            self.active_events[event_id] = event

            # True priority: negative so greater priority pops first; counter breaks ties
            await self.event_queue.put((-int(priority), next(self._counter), event))

            self._update_agent_performance("event_creation", True)
            return event

        except Exception as e:
            logger.error(f"Error creating event: {e}")
            self._update_agent_performance("event_creation", False)
            return {"error": str(e)}

    async def _analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an event using the analysis agent (best effort)."""
        try:
            if not self.analysis_agent:
                return {"skipped": "No analysis agent available"}

            analysis_context = {
                "event": event,
                "system_state": self.agent_context,
                "historical_patterns": self._get_historical_patterns(event["type"]),
            }

            analysis = await self.analysis_agent.analyze(
                context=analysis_context,
                capabilities=["event_analysis", "impact_assessment"],
            )

            self._update_agent_learning("event_analysis", analysis or {})
            return analysis or {}
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            return {"error": str(e)}

    # ---- worker loop --------------------------------------------------------------

    async def _process_event_queue(self):
        """Worker: process events from the priority queue (cancellation-safe)."""
        worker_name = asyncio.current_task().get_name() if asyncio.current_task() else "EventSystemWorker"
        logger.debug(f"{worker_name} started")
        
        try:
            while not self._is_shutting_down:
                item = None
                try:
                    # Use wait_for with a timeout to periodically check shutdown status
                    try:
                        item = await asyncio.wait_for(
                            self.event_queue.get(), 
                            timeout=1.0  # Check for shutdown every second
                        )
                    except asyncio.TimeoutError:
                        # Check if we should shut down
                        if self._is_shutting_down:
                            break
                        continue  # Try again
                        
                except asyncio.CancelledError:
                    logger.info(f"{worker_name} cancelled; exiting gracefully")
                    break

                if item is None:
                    continue

                # Process the item
                try:
                    _neg_prio, _seq, event = item

                    # Skip cancelled events
                    if event.get("status") == "cancelled":
                        logger.debug(f"{worker_name} skipping cancelled event {event.get('id')}")
                        continue

                    # Check for shutdown before processing
                    if self._is_shutting_down:
                        logger.debug(f"{worker_name} shutting down, skipping event {event.get('id')}")
                        break

                    # Process + propagate
                    await self._process_event(event)
                    await self._propagate_event(event)

                    # Mark processed
                    async with GameTimeContext(self.user_id, self.conversation_id) as game_time:
                        event["processed_at"] = (await game_time.to_datetime()).isoformat()
                    event["status"] = "processed"

                    # History trim
                    self.event_history.append(event)
                    if len(self.event_history) > self.max_history:
                        self.event_history = self.event_history[-self.max_history :]

                    self._update_agent_performance("event_processing", True)

                except asyncio.CancelledError:
                    # Re-raise cancellation
                    raise
                except Exception as e:
                    logger.error(f"Error processing event: {e}", exc_info=True)
                    self._update_agent_performance("event_processing", False)
                finally:
                    # Always mark task as done if we got an item
                    if item is not None:
                        self.event_queue.task_done()
                        
        except asyncio.CancelledError:
            logger.info(f"{worker_name} cancelled during shutdown")
        except Exception as e:
            logger.error(f"{worker_name} unexpected error: {e}", exc_info=True)
        finally:
            logger.debug(f"{worker_name} exited cleanly")

    async def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event."""
        try:
            handler = self.event_handlers.get(event["type"])
            if not handler:
                msg = f"No handler registered for event type: {event['type']}"
                logger.warning(msg)
                return {"error": msg}

            # Optional agent guidance
            async with self.processing_lock:
                if self.event_agent:
                    processing_context = {
                        "event": event,
                        "system_state": self.agent_context,
                        "handler": handler.__name__,
                    }
                    guidance = await self.event_agent.guide(
                        context=processing_context,
                        capabilities=["event_processing", "event_coordination"],
                    )
                    result = await handler(event, guidance)
                else:
                    result = await handler(event, None)

            event["results"] = result or {}
            self._update_agent_learning("event_processing", result or {})
            return result or {}

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return {"error": str(e)}

    async def _propagate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate an event (best effort)."""
        try:
            if not self.propagation_agent:
                return {"skipped": "No propagation agent available"}

            propagation_context = {
                "event": event,
                "system_state": self.agent_context,
                "affected_systems": self._get_affected_systems(event),
            }

            plan = await self.propagation_agent.plan(
                context=propagation_context,
                capabilities=["event_propagation", "system_integration"],
            )

            result = await self._execute_propagation(event, plan or {})
            self._update_agent_learning("event_propagation", result or {})
            return result or {}

        except Exception as e:
            logger.error(f"Error propagating event: {e}")
            return {"error": str(e)}

    async def _execute_propagation(self, event: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute event propagation plan (placeholder for concrete integrations)."""
        return {"propagated": True, "plan": plan}

    # ---- helpers / analytics ------------------------------------------------------

    def _get_affected_systems(self, event: Dict[str, Any]) -> List[str]:
        affected: List[str] = []
        etype = event["type"]
        data = event.get("data", {})

        if etype in {"conflict_event", "artifact_event", "lore_event", "npc_event", "relationship_event"}:
            affected.append(etype.split("_", 1)[0])

        if isinstance(data, dict) and "cross_system_effects" in data:
            try:
                affected.extend(list(data["cross_system_effects"]))
            except Exception:
                pass

        # dedupe
        return list(dict.fromkeys(affected))

    def _get_historical_patterns(self, event_type: str) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        type_history = [e for e in self.event_history if e.get("type") == event_type]

        if type_history:
            success = [e for e in type_history if "error" not in (e.get("results") or {})]
            if success:
                patterns.append({"type": "success", "count": len(success), "examples": success[-3:]})

            failure = [e for e in type_history if "error" in (e.get("results") or {})]
            if failure:
                patterns.append({"type": "failure", "count": len(failure), "examples": failure[-3:]})

        return patterns

    def _update_agent_performance(self, action: str, success: bool):
        metrics = self.agent_performance.setdefault(
            action, {"total": 0, "successful": 0, "failed": 0, "success_rate": 0.0}
        )
        metrics["total"] += 1
        if success:
            metrics["successful"] += 1
        else:
            metrics["failed"] += 1
        metrics["success_rate"] = (metrics["successful"] / metrics["total"]) if metrics["total"] else 0.0

    def _update_agent_learning(self, action: str, result: Dict[str, Any]):
        learning = self.agent_learning.setdefault(action, {"patterns": [], "strategies": {}, "adaptations": []})

        if isinstance(result, dict):
            # patterns
            if "patterns" in result and isinstance(result["patterns"], list):
                learning["patterns"].extend(result["patterns"])

            # strategies
            if "strategy" in result:
                strategy = result["strategy"]
                strat = learning["strategies"].setdefault(strategy, {"uses": 0, "successes": 0, "failures": 0})
                strat["uses"] += 1
                if "error" in result:
                    strat["failures"] += 1
                else:
                    strat["successes"] += 1

            # adaptation
            if "adaptation" in result:
                learning["adaptations"].append({"timestamp": datetime.utcnow().isoformat(), "details": result["adaptation"]})

    # ---- NEW: Updated Conflict Handlers ----

    async def _handle_conflict_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        """Generic conflict event handler - routes to synthesizer"""
        try:
            if not self.conflict_synthesizer:
                return {"error": "Conflict synthesizer not initialized"}
            
            data = event.get("data", {})
            
            # Determine what kind of conflict operation this is
            if "create" in data:
                # Create new conflict
                conflict_type = data.get("conflict_type", "slice")
                context = data.get("context", {})
                result = await self.conflict_synthesizer.create_conflict(conflict_type, context)
                return {"created": True, "conflict": result}
                
            elif "resolve" in data:
                # Resolve conflict
                conflict_id = data.get("conflict_id")
                if not conflict_id:
                    return {"error": "No conflict_id provided for resolution"}
                resolution_type = data.get("resolution_type", "natural")
                context = data.get("context", {})
                result = await self.conflict_synthesizer.resolve_conflict(
                    conflict_id, resolution_type, context
                )
                return {"resolved": True, "resolution": result}
                
            elif "process_scene" in data:
                # Process a scene through conflicts
                scene_context = data.get("scene_context", {})
                result = await self.conflict_synthesizer.process_scene(scene_context)
                return {"processed": True, "scene_result": result}
                
            else:
                # Get conflict state
                conflict_id = data.get("conflict_id")
                if conflict_id:
                    state = await self.conflict_synthesizer.get_conflict_state(conflict_id)
                    return {"state": state}
                else:
                    system_state = await self.conflict_synthesizer.get_system_state()
                    return {"system_state": system_state}
                    
        except Exception as e:
            logger.error(f"Error handling conflict event: {e}", exc_info=True)
            return {"error": str(e)}

    async def _handle_conflict_created(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle a conflict creation event"""
        try:
            data = event.get("data", {})
            conflict_type = data.get("conflict_type", "slice")
            
            # Extract context for conflict creation
            context = {
                "description": data.get("description", ""),
                "participants": data.get("participants", []),
                "location": data.get("location"),
                "intensity": data.get("intensity", "moderate"),
                "is_multiparty": data.get("is_multiparty", False),
                "party_count": data.get("party_count"),
                "trigger_event": event.get("id")
            }
            
            # Check if it should be multiparty based on participants
            if len(context["participants"]) > 2 and not context["is_multiparty"]:
                context["is_multiparty"] = True
                context["party_count"] = len(context["participants"])
            
            result = await self.conflict_synthesizer.create_conflict(conflict_type, context)
            
            # Create follow-up events if needed
            if result.get("narrative_hooks"):
                for hook in result["narrative_hooks"][:3]:  # Limit to 3 hooks
                    await self.create_event(
                        event_type="narrative_hook",
                        event_data={
                            "hook": hook,
                            "conflict_id": result.get("conflict_id"),
                            "source": "conflict_creation"
                        },
                        priority=3
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling conflict creation: {e}", exc_info=True)
            return {"error": str(e)}

    async def _handle_conflict_resolved(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle a conflict resolution event"""
        try:
            data = event.get("data", {})
            conflict_id = data.get("conflict_id")
            
            if not conflict_id:
                return {"error": "No conflict_id provided"}
            
            resolution_type = data.get("resolution_type", "natural")
            context = {
                "reason": data.get("reason"),
                "victor": data.get("victor"),
                "consequences": data.get("consequences", {}),
                "trigger_event": event.get("id")
            }
            
            result = await self.conflict_synthesizer.resolve_conflict(
                conflict_id, resolution_type, context
            )
            
            # Create follow-up events for consequences
            if result.get("new_conflicts_created"):
                for new_conflict_id in result["new_conflicts_created"]:
                    await self.create_event(
                        event_type="conflict_escalation",
                        event_data={
                            "parent_conflict": conflict_id,
                            "new_conflict": new_conflict_id,
                            "reason": "resolution_consequence"
                        },
                        priority=5
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling conflict resolution: {e}", exc_info=True)
            return {"error": str(e)}

    async def _handle_conflict_scene(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a scene through the conflict system"""
        try:
            data = event.get("data", {})
            scene_context = {
                "scene_type": data.get("scene_type", "interaction"),
                "location": data.get("location"),
                "participants": data.get("participants", []),
                "activity": data.get("activity"),
                "mood": data.get("mood"),
                "player_action": data.get("player_action")
            }
            
            result = await self.conflict_synthesizer.process_scene(scene_context)
            
            # Check for emergent conflicts
            if result.get("conflicts_detected"):
                for conflict_id in result["conflicts_detected"]:
                    await self.create_event(
                        event_type="conflict_manifestation",
                        event_data={
                            "conflict_id": conflict_id,
                            "scene_id": data.get("scene_id"),
                            "manifestations": result.get("manifestations", [])
                        },
                        priority=4
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing conflict scene: {e}", exc_info=True)
            return {"error": str(e)}

    # ---- Helper Methods for Conflict Integration ----

    async def create_conflict_event(
        self,
        conflict_type: str,
        context: Dict[str, Any],
        priority: int = 5
    ) -> Dict[str, Any]:
        """Convenience method to create a conflict through the event system"""
        return await self.create_event(
            event_type="conflict_created",
            event_data={
                "conflict_type": conflict_type,
                **context
            },
            priority=priority
        )

    async def resolve_conflict_event(
        self,
        conflict_id: int,
        resolution_type: str,
        context: Dict[str, Any],
        priority: int = 5
    ) -> Dict[str, Any]:
        """Convenience method to resolve a conflict through the event system"""
        return await self.create_event(
            event_type="conflict_resolved",
            event_data={
                "conflict_id": conflict_id,
                "resolution_type": resolution_type,
                **context
            },
            priority=priority
        )

    async def check_conflict_triggers(
        self,
        scene_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if current scene should trigger any conflicts"""
        if not self.conflict_synthesizer:
            return []
        
        # Get system state
        system_state = await self.conflict_synthesizer.get_system_state()
        
        triggers = []
        
        # Check tension levels
        if "metrics" in system_state:
            metrics = system_state["metrics"]
            if metrics.get("complexity_score", 0) > 0.7:
                triggers.append({
                    "type": "high_complexity",
                    "suggested_conflict": "social",
                    "reason": "System complexity is high"
                })
        
        # Check active conflicts for escalation
        active_conflicts = system_state.get("active_conflicts", [])
        if len(active_conflicts) > 3:
            triggers.append({
                "type": "conflict_overload", 
                "suggested_action": "resolve_some",
                "reason": f"{len(active_conflicts)} active conflicts"
            })
        
        return triggers

    # ---- Existing handlers (kept as-is) ----

    async def _handle_artifact_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if not self.artifact_manager:
                return {"error": "Artifact manager not initialized"}
                
            data = event["data"]
            artifact_id = data.get("artifact_id")
            if not artifact_id:
                return {"error": "No artifact ID provided"}

            artifact = self.artifact_manager.active_artifacts.get(artifact_id)
            if not artifact:
                return {"error": f"Artifact {artifact_id} not found"}

            return await self.artifact_manager.process_artifact_event(artifact_id, data)
        except Exception as e:
            logger.error(f"Error handling artifact event: {e}")
            return {"error": str(e)}

    async def _handle_lore_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if not self.lore_system:
                return {"error": "Lore system not initialized"}
                
            data = event["data"]
            return await self.lore_system.process_lore_event(self.user_id, self.conversation_id, data)
        except Exception as e:
            logger.error(f"Error handling lore event: {e}")
            return {"error": str(e)}

    async def _handle_npc_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            data = event["data"]
            npc_ids = data.get("npc_ids", [])
            if not npc_ids:
                return {"error": "No NPC IDs provided"}

            results = {}
            
            if self.npc_learning:
                learning_result = await self.npc_learning.process_event_for_learning(
                    data.get("event_text", ""),
                    data.get("event_type", "unknown"),
                    npc_ids,
                    data.get("player_response"),
                )
                results["learning"] = learning_result

            if self.belief_system:
                belief_result = await self.belief_system.process_event_for_beliefs(
                    data.get("event_text", ""),
                    data.get("event_type", "unknown"),
                    npc_ids,
                    data.get("factuality", 1.0),
                )
                results["beliefs"] = belief_result

            return results
        except Exception as e:
            logger.error(f"Error handling NPC event: {e}")
            return {"error": str(e)}

    async def _handle_relationship_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            data = event["data"]

            if not self.relationship_manager:
                self.relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)

            if data.get("interaction_type"):
                return await self.relationship_manager.process_interaction(
                    entity1_type=data.get("entity1_type", "player"),
                    entity1_id=data.get("entity1_id", 1),
                    entity2_type=data.get("entity2_type"),
                    entity2_id=data.get("entity2_id"),
                    interaction={"type": data.get("interaction_type")},
                )

            # Direct update path
            state = await self.relationship_manager.get_relationship_state(
                entity1_type=data.get("entity1_type", "player"),
                entity1_id=data.get("entity1_id", 1),
                entity2_type=data.get("entity2_type"),
                entity2_id=data.get("entity2_id"),
            )

            if data.get("dimensions"):
                for dim, value in data["dimensions"].items():
                    if hasattr(state.dimensions, dim):
                        setattr(state.dimensions, dim, value)
                state.dimensions.clamp()
                await self.relationship_manager._queue_update(state)
                await self.relationship_manager._flush_updates()

            return {"success": True, "state": state.to_summary()}
        except Exception as e:
            logger.error(f"Error handling relationship event: {e}")
            return {"error": str(e)}

    async def _handle_time_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if not self.lore_system:
                return {"error": "Lore system not initialized"}
                
            data = event["data"]
            time_amount = data.get("time_amount", 0)
            time_unit = data.get("time_unit", "hours")
            if not time_amount:
                return {"error": "No time amount provided"}

            return await self.lore_system.advance_time(self.user_id, self.conversation_id, time_amount, time_unit)
        except Exception as e:
            logger.error(f"Error handling time event: {e}")
            return {"error": str(e)}

    # ---- queries / admin ----------------------------------------------------------

    async def get_active_events(self) -> List[Dict[str, Any]]:
        return list(self.active_events.values())

    async def get_event_history(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        events = self.event_history
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
        return events[-int(limit):]

    async def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self.active_events.get(event_id)

    async def cancel_event(self, event_id: str) -> Dict[str, Any]:
        """
        Mark a queued event as cancelled. It will be skipped when dequeued.
        (There is no safe way to remove arbitrary items from asyncio queues.)
        """
        try:
            event = self.active_events.get(event_id)
            if not event:
                return {"error": f"Event {event_id} not found"}

            status = event.get("status")
            if status != "queued":
                return {"error": f"Event {event_id} is not queued (status={status})"}

            event["status"] = "cancelled"
            return {"success": True, "message": f"Event {event_id} marked as cancelled"}
        except Exception as e:
            logger.error(f"Error cancelling event: {e}")
            return {"error": str(e)}

    async def get_event_statistics(self) -> Dict[str, Any]:
        try:
            total_events = len(self.event_history)
            by_type: Dict[str, int] = {}
            for e in self.event_history:
                et = e.get("type", "unknown")
                by_type[et] = by_type.get(et, 0) + 1

            times = []
            for e in self.event_history:
                if e.get("processed_at"):
                    try:
                        created = datetime.fromisoformat(e["created_at"])
                        processed = datetime.fromisoformat(e["processed_at"])
                        times.append((processed - created).total_seconds())
                    except Exception:
                        pass

            avg_processing_time = (sum(times) / len(times)) if times else 0.0

            queued_size = self.event_queue.qsize()
            active_count = sum(1 for e in self.active_events.values() if e.get("status") == "queued")

            return {
                "total_events": total_events,
                "events_by_type": by_type,
                "average_processing_time": avg_processing_time,
                "active_events": active_count,
                "queued_events": queued_size,
                "workers": len([t for t in self._worker_tasks if not t.done()]),
                "is_shutting_down": self._is_shutting_down,
            }
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {"error": str(e)}


# ---- Context Manager for EventSystem ----

@asynccontextmanager
async def create_event_system(user_id: int, conversation_id: int, **kwargs):
    """
    Context manager for EventSystem that ensures proper cleanup.
    
    Usage:
        async with create_event_system(user_id, conversation_id) as event_system:
            await event_system.create_event(...)
            # EventSystem will be properly shut down when exiting the context
    """
    event_system = EventSystem(user_id, conversation_id, **kwargs)
    try:
        await event_system.initialize()
        yield event_system
    finally:
        await event_system.shutdown(drain=True, timeout=2.0)


