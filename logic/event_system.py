"""
Event System

This module provides a comprehensive event system that integrates with other game systems
including NPCs, lore, conflicts, and artifacts. It handles event generation, processing,
and propagation throughout the game world.

Refactor highlights:
- Cancellation-safe worker loop (no stray task_done calls)
- PriorityQueue with true priority ordering
- Single-init guard + configurable worker pool
- Graceful shutdown with optional drain
- Effective cancel semantics for queued events
- Hardened metrics, IDs, logging

Updated to properly use the new dynamic relationship system.
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import asyncio
from datetime import datetime
from itertools import count
from uuid import uuid4

import asyncpg  # explicit import

from agents import function_tool, RunContextWrapper, Agent, Runner, trace
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from db.connection import get_db_connection_context

from logic.dynamic_relationships import OptimizedRelationshipManager
from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
from logic.artifact_system.artifact_manager import ArtifactManager
from npcs.npc_learning_adaptation import NPCLearningManager
from npcs.belief_system_integration import NPCBeliefSystemIntegration

logger = logging.getLogger(__name__)


class EventSystem:
    """
    Comprehensive event system that handles event generation, processing,
    and propagation throughout the game world.

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

        # worker pool
        self.num_workers = max(1, int(num_workers))
        self._worker_tasks: List[asyncio.Task] = []

        # core systems
        self.conflict_resolution: Optional[ConflictResolutionSystem] = None
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

    # ---- lifecycle ----------------------------------------------------------------

    async def initialize(self) -> "EventSystem":
        """Initialize the event system and its dependencies, and start workers."""
        async with self._init_lock:
            if self.is_initialized:
                return self

            # Initialize core systems
            self.conflict_resolution = ConflictResolutionSystem(self.user_id, self.conversation_id)
            await self.conflict_resolution.initialize()

            self.artifact_manager = ArtifactManager(self.user_id, self.conversation_id)
            await self.artifact_manager.initialize()

            # Import LoreSystem lazily to avoid circular deps
            from logic.lore.core.system import LoreSystem
            self.lore_system = LoreSystem()

            self.relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            self.npc_learning = NPCLearningManager(self.user_id, self.conversation_id)
            self.belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)

            # Initialize agents (best effort)
            await self._initialize_agents()

            # Handlers
            self._register_default_handlers()

            # Spawn worker tasks (guarded; idempotent)
            self._start_workers()

            self.is_initialized = True
            logger.info(f"Event system initialized for user={self.user_id} convo={self.conversation_id} workers={len(self._worker_tasks)}")
        return self

    def _start_workers(self):
        # start exactly num_workers workers if not already running
        alive = [t for t in self._worker_tasks if not t.done()]
        missing = self.num_workers - len(alive)
        for _ in range(max(0, missing)):
            task = asyncio.create_task(self._process_event_queue(), name=f"EventSystemWorker-{self.user_id}-{self.conversation_id}")
            self._worker_tasks.append(task)

    async def shutdown(self, drain: bool = False, timeout: float = 5.0):
        """
        Stop worker tasks. If drain=True, wait for the queue to empty (best effort).
        Call this during application shutdown.
        """
        if drain:
            try:
                await asyncio.wait_for(self.event_queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("EventSystem drain timeout; continuing shutdown")

        # cancel workers
        for t in self._worker_tasks:
            if not t.done():
                t.cancel()

        for t in self._worker_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Worker ended with error during shutdown: {e}")

        self._worker_tasks.clear()
        logger.info("EventSystem shutdown complete")

    # ---- agents -------------------------------------------------------------------

    async def _initialize_agents(self):
        """Initialize the event system agents (best effort)."""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)

            self.event_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_MANAGER,
                agent_id="event_manager",
                model="gpt-4.1-nano",
            )

            self.analysis_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_ANALYZER,
                agent_id="event_analyzer",
                model="gpt-4.1-nano",
            )

            self.propagation_agent = await governance.create_agent(
                agent_type=AgentType.EVENT_PROPAGATOR,
                agent_id="event_propagator",
                model="gpt-4.1-nano",
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
        self.register_handler("conflict_event", self._handle_conflict_event)
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

            event: Dict[str, Any] = {
                "id": event_id,
                "type": event_type,
                "data": event_data,
                "priority": int(priority),
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "processed_at": None,
                "results": {},
            }

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
        while True:
            try:
                # Only call task_done for items successfully fetched
                item = await self.event_queue.get()
            except asyncio.CancelledError:
                logger.info(f"{worker_name} cancelled; exiting")
                break

            try:
                _neg_prio, _seq, event = item

                # Skip cancelled events
                if event.get("status") == "cancelled":
                    logger.debug(f"{worker_name} skipping cancelled event {event.get('id')}")
                    continue

                # Process + propagate
                await self._process_event(event)
                await self._propagate_event(event)

                # Mark processed
                event["processed_at"] = datetime.utcnow().isoformat()
                event["status"] = "processed"

                # History trim
                self.event_history.append(event)
                if len(self.event_history) > self.max_history:
                    self.event_history = self.event_history[-self.max_history :]

                self._update_agent_performance("event_processing", True)

            except asyncio.CancelledError:
                # We fetched an item; pair the task_done with it, then exit
                raise
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
                self._update_agent_performance("event_processing", False)
            finally:
                # Pair with the 'get' above — safe even if processing failed
                self.event_queue.task_done()

        logger.debug(f"{worker_name} exited")

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

    # ---- default handlers ---------------------------------------------------------

    async def _handle_conflict_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            data = event["data"]
            conflict_id = data.get("conflict_id")
            if not conflict_id:
                return {"error": "No conflict ID provided"}

            conflict = await self.conflict_resolution.get_conflict_details(conflict_id)
            if not conflict:
                return {"error": f"Conflict {conflict_id} not found"}

            return await self.conflict_resolution.process_conflict_event(conflict_id, data)
        except Exception as e:
            logger.error(f"Error handling conflict event: {e}")
            return {"error": str(e)}

    async def _handle_artifact_event(self, event: Dict[str, Any], guidance: Optional[Dict] = None) -> Dict[str, Any]:
        try:
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

            learning_result = await self.npc_learning.process_event_for_learning(
                data.get("event_text", ""),
                data.get("event_type", "unknown"),
                npc_ids,
                data.get("player_response"),
            )

            belief_result = await self.belief_system.process_event_for_beliefs(
                data.get("event_text", ""),
                data.get("event_type", "unknown"),
                npc_ids,
                data.get("factuality", 1.0),
            )

            return {"learning": learning_result, "beliefs": belief_result}
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
            }
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {"error": str(e)}
