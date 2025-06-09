import asyncio
import datetime
import re
from agents import Agent

class ReflectionAgent(Agent):
    """Simple agent producing reflective insights from event logs."""
    SYSTEM_PROMPT = "You are Nyx's inner critic. Provide concise insights."

    async def run(self, events: list):
        txt = "\n".join(f"{e.role}:{e.content}" for e in events[-100:])
        insight = await self.chat(txt)
        priority = _extract_priority(insight)
        from nyx.core.memory.memory_manager import MemoryManager
        await MemoryManager.add(insight, {"type": "insight", "priority": priority})
        if priority == "high":
            try:
                from strategy import manager as strategy_manager
                asyncio.create_task(strategy_manager.apply(insight))
            except Exception:
                pass
        return insight

def _extract_priority(text: str) -> str:
    m = re.search(r"priority\s*:\s*(\w+)", text, re.IGNORECASE)
    return m.group(1).lower() if m else "normal"

log_buffer: list = []
last_reflection_time: datetime.datetime = datetime.datetime.utcnow()
reflection_agent = ReflectionAgent(name="ReflectionAgent", instructions=ReflectionAgent.SYSTEM_PROMPT)

async def schedule_reflection() -> None:
    """Run reflection when buffer is large or too much time has passed."""
    global last_reflection_time
    if len(log_buffer) >= 100 or (datetime.datetime.utcnow() - last_reflection_time) > datetime.timedelta(hours=2):
        asyncio.create_task(reflection_agent.run(log_buffer))
        log_buffer.clear()
        last_reflection_time = datetime.datetime.utcnow()
