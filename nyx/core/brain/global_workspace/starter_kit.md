Below is a concise “starter kit” for hooking Nyx Brain up to the Global‑Workspace runtime. Everything lives in a single file so you can paste‑in quickly, then break it into multiple modules later if you prefer.

1 Where the code lives
pgsql
Copy
Edit
flask_roleplay/
└─ nyx/
   └─ core/
      └─ global_workspace/
         ├─ global_workspace_architecture.py   ← the whole runtime (already in canvas)
         ├─ memory_module.py                   ← wrapper for MemoryCore (already in canvas)
         └─ … (add more wrappers here)
If you want cleaner separation, split global_workspace_architecture.py into:

bash
Copy
Edit
workspace.py        # Proposal + GlobalContextWorkspace
attention.py        # AttentionMechanism
goals.py            # GoalManager
coordinator.py      # Coordinator
reflection.py       # ReflectionLoop
engine.py           # NyxEngine (imports the above)
base_module.py      # WorkspaceModule
The classes already have no circular imports, so a straight copy‑and‑paste split works.

2 Minimal NyxBrain wiring
python
Copy
Edit
# nyx/core/brain/base.py  (or wherever NyxBrain lives)

from nyx.core.global_workspace.global_workspace_architecture import (
    NyxEngine,
)
from nyx.core.global_workspace.memory_module import MemoryModule
# TODO: import wrappers for emotions, reasoning, expression, …

class NyxBrain:
    def __init__(self):
        self.engine: NyxEngine | None = None

    async def initialize(self) -> None:
        """Instantiate modules and fire up the workspace engine."""
        modules = [
            MemoryModule(None),          # memory
            # EmotionModule(None),
            # ReasoningModule(None),
            # ExpressionModule(None),
            # … add each wrapper here …
        ]
        self.engine = NyxEngine(modules)
        await self.engine.start()

    # --------------------------------------------------------------
    # External API Nyx already exposes – now just delegate to engine
    # --------------------------------------------------------------
    async def process_input(self, user_input: str) -> dict[str, any]:
        if self.engine is None:
            await self.initialize()

        # engine returns `decision` (string or action object); wrap into
        # your existing response schema if needed.
        decision = await self.engine.process_input(user_input)
        return {
            "message": decision,
            "engine": "GWA",
        }
Pass None for ws when constructing each module; NyxEngine sets the shared workspace reference during startup so every wrapper sees the same object.

3 Running a smoke test
python
Copy
Edit
import asyncio
from nyx.core.brain.base import NyxBrain

brain = NyxBrain()

async def main():
    print(await brain.process_input("Hello, Nyx!"))
    print(await brain.process_input("Tell me about your day."))

asyncio.run(main())
With only MemoryModule and the built‑in EchoModule, you should see the echo plus memory summaries surfacing through the workspace. Add each new wrapper and test again; nothing else in Nyx needs to change.

4 When to split files
Keep one file until:

Unit‑tests start mocking individual subsystems.

You add more sophisticated attention or consensus algorithms.

You want separate teams maintaining different layers.

Then cut‑and‑paste the classes into the structure shown in §1; import paths inside the classes remain the same.

You now have:

global_workspace_architecture.py – the full working runtime

memory_module.py – sample wrapper

A skeleton for wiring NyxBrain to the engine

Drop these in, start adding wrappers, and Nyx will run on the new cognitive backbone. Ping me whenever you’re ready to split files, tune salience logic, or wrap another subsystem!
