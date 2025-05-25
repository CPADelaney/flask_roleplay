from ..global_workspace.types import Proposal
from ..global_workspace.engine_v2 import WorkspaceModule
import asyncio, random, time

class MemoryGateway(WorkspaceModule):
    def __init__(self, memory_core):
        super().__init__()
        self.mem = memory_core
    async def on_phase(self, phase):
        if phase == 0 and random.random() < 0.2:              # sample occasionally
            recent = await self.mem.peek_recent(n=3)
            if recent:
                await self.submit(recent[-1], salience=0.4, context_tag="memory")

class EmotionGateway(WorkspaceModule):
    def __init__(self, emo, hormones, mood):
        super().__init__()
        self.ecore, self.hs, self.mm = emo, hormones, mood
    async def on_phase(self, phase):
        if phase == 0:
            levels = self.hs.get_levels()
            await self.submit(levels, salience=0.7, context_tag="affect")

class ReasoningGateway(WorkspaceModule):
    def __init__(self, rc):
        super().__init__()
        self.rc = rc
    async def on_phase(self, phase):
        if phase == 1:
            q = self.rc.pop_pending_question()
            if q:
                await self.submit(q, salience=0.6, context_tag="reason")

class ExpressionGateway(WorkspaceModule):
    def __init__(self, ag):
        super().__init__()
        self.ag = ag
    async def on_phase(self, phase):
        # after coordinator picks content, finalise wording
        pass   # handled by NyxBrain downstream – can stay empty

class MultimodalGateway(WorkspaceModule):
    def __init__(self, mm):
        super().__init__()
        self.mm = mm
    async def on_phase(self, phase):
        if phase == 2:
            fused = self.mm.get_current_binding()
            if fused:
                await self.submit(fused, salience=0.5, context_tag="percept")
