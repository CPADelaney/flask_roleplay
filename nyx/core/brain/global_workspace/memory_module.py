"""MemoryModule – bridge Nyx MemoryCoreAgents to the Global Workspace

Drop this file next to `global_workspace_architecture.py` and import it when
constructing the NyxEngine.  It wraps the existing, heavy‑duty MemoryCore in a
thin WorkspaceModule–compatible shell so the new architecture can pull
contextually relevant memories.

The first objective is *demonstration* – a real deployment should refine the
retrieval/summary logic and error handling.  All interaction with the shared
blackboard uses `submit()` and `observe()` from the base class.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List

from nyx.core.brain.global_workspace.global_workspace_architecture import WorkspaceModule, Proposal  # noqa: relative import ok
from nyx.core.memory_core import MemoryCoreAgents  # adjust path if needed

logger = logging.getLogger(__name__)


class MemoryModule(WorkspaceModule):
    """Expose Nyx’s MemoryCore as a contributor to the global workspace."""

    def __init__(self, ws):
        super().__init__(name="memory", ws=ws)
        self._core = MemoryCoreAgents(user_id=None, conversation_id=None)
        self._ready: bool = False
        # throttle to avoid excessive queries in fast chat loops
        self._cooldown_sec = 1.0
        self._last_call: float = 0.0

    async def _ensure_ready(self):
        if not self._ready:
            await self._core.initialize()
            self._ready = True

    # ---------------------------------------------------------------------
    # Contribution logic
    # ---------------------------------------------------------------------

    async def contribute(self):
        """React to new user input by surfacing top memories."""
        await self._ensure_ready()

        focus: Proposal | None = self.observe("focus")
        if not focus or focus.context_tag != "user_input":
            return  # nothing to do

        # simple cooldown guard
        import time as _time
        if _time.time() - self._last_call < self._cooldown_sec:
            return
        self._last_call = _time.time()

        query = str(focus.content)[:200]  # limit length for retrieval

        # Retrieve top 3 memories (fast, no extra filters)
        try:
            memories: List[dict[str, Any]] = await self._core.retrieve_memories(query=query, limit=3)
        except Exception as e:  # pragma: no cover
            logger.error("Memory retrieval failed: %s", e)
            return

        if not memories:
            return  # nothing to surface

        # Create a compact summary for other modules
        summary_snippets = "; ".join(m.get("memory_text", "")[:60] for m in memories)

        # Boost salience a bit to compete with language modules
        self.submit(
            content={"memories": memories, "summary": summary_snippets},
            salience=0.6,
            tag="memory_recall",
        )

    # The module could expose helper APIs (e.g., explicit memory add) if needed.


# -------------------------------------------------------------------------
# Quick verification (optional): run this module standalone with a minimal
# engine to ensure it does not crash.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    from global_workspace_architecture import NyxEngine, EchoModule

    async def _test():
        eng = NyxEngine(modules=[EchoModule("echo", None)])  # placeholder; memory added below
        # inject workspace into memory module after engine builds ws
        mem_mod = MemoryModule(eng.ws)
        eng.modules.append(mem_mod)
        for m in eng.modules:
            if getattr(m, "ws", None) is None:
                m.ws = eng.ws
        await eng.start()
        print(await eng.process_input("Tell me about last night"))

    asyncio.run(_test())
