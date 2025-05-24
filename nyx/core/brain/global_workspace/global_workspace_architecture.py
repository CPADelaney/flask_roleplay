"""Global Workspace Architecture for Nyx

This file contains the minimal, high‑performance async implementation of the
core infrastructure that replaces the legacy event‑bus system.  It introduces:

• Proposal – lightweight dataclass for module contributions.
• GlobalContextWorkspace – shared async blackboard.
• AttentionMechanism – selects highest‑salience proposal.
• GoalManager – placeholder hierarchical goal handler.
• Coordinator – merges proposals into a single decision.
• ReflectionLoop – meta layer for last‑second coherence checks.
• WorkspaceModule – base class for all Nyx modules.
• NyxEngine – orchestrator that ties everything together.

The code is intentionally lean so existing modules can migrate incrementally.
Drop this file into ``flask_roleplay/nyx/core/global_workspace`` and adapt
imports accordingly.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Core datatypes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Proposal:
    """A unit of information contributed by a module.

    Parameters
    ----------
    source : str
        Unique name of the contributing module.
    content : Any
        Arbitrary payload – text, action plan, etc.
    salience : float, optional
        Relative importance (0‑1).  Higher means more likely to gain focus.
    context_tag : str, optional
        Quick label so other modules can filter by type/category.
    timestamp : float, optional
        Unix time recorded automatically if omitted.
    """

    source: str
    content: Any
    salience: float = 0.5
    context_tag: str = "general"
    timestamp: float = field(default_factory=time.time)

    def boost(self, factor: float = 1.2, cap: float = 1.0) -> None:
        """Increase salience (goal alignment, recency, etc.)."""
        self.salience = min(self.salience * factor, cap)


# ---------------------------------------------------------------------------
# Global blackboard
# ---------------------------------------------------------------------------

class GlobalContextWorkspace:
    """Thread‑safe async blackboard shared by all modules."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._new_proposal_event: asyncio.Event = asyncio.Event()

        self.proposals: List[Proposal] = []           # transient inbox
        self.focus: Optional[Proposal] = None         # spotlight of attention

        # Global persistent state buckets (extend as needed)
        self.state: Dict[str, Any] = {}
        self.active_goals: List[Dict[str, Any]] = []

    # ----- proposal handling -------------------------------------------------
    async def submit_proposal(self, proposal: Proposal) -> None:
        """Add proposal and notify attention mechanism."""
        async with self._lock:
            self.proposals.append(proposal)
            # keep proposals bounded (basic hygiene)
            if len(self.proposals) > 1_000:
                self.proposals.pop(0)
        self._new_proposal_event.set()

    async def get_proposals(self) -> List[Proposal]:
        async with self._lock:
            return list(self.proposals)

    async def clear_proposals(self) -> None:
        async with self._lock:
            self.proposals.clear()

    # ----- attention focus ---------------------------------------------------
    def set_focus(self, proposal: Proposal) -> None:
        self.focus = proposal

    def get_focus(self) -> Optional[Proposal]:
        return self.focus

    # ----- signalling --------------------------------------------------------
    async def wait_for_new_proposal(self) -> None:
        """Suspend until at least one new proposal arrives."""
        await self._new_proposal_event.wait()
        # auto‑clear for next wait cycle
        self._new_proposal_event.clear()


# ---------------------------------------------------------------------------
# Goal hierarchy (placeholder)
# ---------------------------------------------------------------------------

class GoalManager:
    """Tracks active goals and biases salience/decisions."""

    def __init__(self) -> None:
        self.active: List[Dict[str, Any]] = []

    # — simple API —
    def add(self, desc: str, goal_id: str | None = None, priority: float = 1.0):
        goal = {"id": goal_id or desc, "desc": desc, "priority": priority}
        self.active.append(goal)

    def list(self) -> Sequence[Dict[str, Any]]:
        return tuple(self.active)

    # — hooks used by other subsystems —
    def boost_for_goal(self, proposal: Proposal) -> None:
        for g in self.active:
            if g["id"] in proposal.context_tag or g["id"] in str(proposal.content):
                proposal.boost()

    def veto(self, decision: Any) -> bool:
        """Return True if decision must be blocked (placeholder logic)."""
        for g in self.active:
            if g["desc"].lower().startswith("stay quiet") and isinstance(decision, str):
                return True
        return False


# ---------------------------------------------------------------------------
# Attention mechanism
# ---------------------------------------------------------------------------

class AttentionMechanism:
    """Select the most salient proposal whenever new info arrives."""

    def __init__(self, ws: GlobalContextWorkspace, goals: GoalManager):
        self.ws, self.goals = ws, goals
        self._task: Optional[asyncio.Task] = None

    async def _select(self) -> None:
        while True:
            await self.ws.wait_for_new_proposal()

            # snapshot proposals
            proposals = await self.ws.get_proposals()
            if not proposals:
                continue

            # goal‑based salience adjustment
            for p in proposals:
                self.goals.boost_for_goal(p)

            # pick highest salience (ties -> newest)
            top = max(proposals, key=lambda p: (p.salience, p.timestamp))
            self.ws.set_focus(top)
            await self.ws.clear_proposals()

    # public API --------------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._select())


# ---------------------------------------------------------------------------
# Decision integrator
# ---------------------------------------------------------------------------

class Coordinator:
    """Merge attention focus (and possibly other data) into final decision."""

    def __init__(self, ws: GlobalContextWorkspace, goals: GoalManager):
        self.ws, self.goals = ws, goals

    async def decide(self) -> Any | None:
        focus = self.ws.get_focus()
        if focus is None:
            return None

        decision = focus.content  # placeholder: accept focus content directly

        # goal veto example
        if self.goals.veto(decision):
            decision = ""  # empty output if vetoed
        return decision


# ---------------------------------------------------------------------------
# Meta‑cognitive reflection
# ---------------------------------------------------------------------------

class ReflectionLoop:
    """Optional post‑processing for coherence/safety checks."""

    def __init__(self, ws: GlobalContextWorkspace):
        self.ws = ws

    async def refine(self, decision: Any) -> Any:
        # trivial placeholder – remove banned marker if present
        if isinstance(decision, str) and "<incoherent>" in decision:
            decision = decision.replace("<incoherent>", "")
        return decision


# ---------------------------------------------------------------------------
# Base module class
# ---------------------------------------------------------------------------

class WorkspaceModule:
    """Base helper for modules interacting with the workspace."""

    def __init__(self, name: str, ws: GlobalContextWorkspace):
        self.name, self.ws = name, ws

    def submit(self, content: Any, salience: float = 0.5, tag: str = "general") -> None:
        prop = Proposal(source=self.name, content=content, salience=salience, context_tag=tag)
        # fire‑and‑forget so module isn't blocked
        asyncio.create_task(self.ws.submit_proposal(prop))

    def observe(self, key: str | None = None) -> Any:
        ctx = {"focus": self.ws.get_focus(), **self.ws.state}
        return ctx.get(key) if key else ctx

    async def contribute(self) -> None:
        """Override in subclass to submit proposals."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Engine orchestrator
# ---------------------------------------------------------------------------

class NyxEngine:
    """Ties workspace, attention, coordinator, reflection, & modules together."""

    def __init__(self, modules: List[WorkspaceModule]):
        self.ws = GlobalContextWorkspace()
        self.goals = GoalManager()
        self.attn = AttentionMechanism(self.ws, self.goals)
        self.coord = Coordinator(self.ws, self.goals)
        self.reflect = ReflectionLoop(self.ws)
        self.modules = modules

    async def process_input(self, user_input: str) -> Any:
        # treat user input as a special proposal so modules see it
        await self.ws.submit_proposal(Proposal("user", user_input, salience=1.0, context_tag="user_input"))

        # let modules react (single round)
        await asyncio.gather(*(m.contribute() for m in self.modules))

        # allow attention mechanism to update focus
        await asyncio.sleep(0)  # yield to let attn pick up proposals

        # produce decision
        decision = await self.coord.decide()
        decision = await self.reflect.refine(decision)
        return decision

    async def start(self) -> None:
        self.attn.start()


# ---------------------------------------------------------------------------
# Example stub module (remove in production)
# ---------------------------------------------------------------------------

class EchoModule(WorkspaceModule):
    """Dummy module that mirrors user input with low salience."""

    async def contribute(self):
        focus = self.observe("focus")
        if focus and focus.context_tag == "user_input":
            self.submit(f"Echo: {focus.content}", salience=0.2, tag="reply")


# ---------------------------------------------------------------------------
# Quick test harness (``python -m global_workspace_architecture``)
# ---------------------------------------------------------------------------

async def _demo():
    engine = NyxEngine([EchoModule("echo", None)])  # module will get ws later
    # patch module workspace
    for m in engine.modules:
        m.ws = engine.ws
    await engine.start()

    resp = await engine.process_input("Hello, Nyx!")
    print("Nyx decision:", resp)

if __name__ == "__main__":
    asyncio.run(_demo())
