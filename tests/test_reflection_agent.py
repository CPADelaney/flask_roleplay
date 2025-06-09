import asyncio
import importlib
import importlib.util
import sys
import types
import datetime

def test_reflection_schedule(monkeypatch):
    async def inner():
        # Fake agents module
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass
            async def chat(self, txt):
                return "insight\npriority: high"
        sys.modules['agents'] = types.SimpleNamespace(Agent=FakeAgent)

        # Fake MemoryManager
        added = []
        class FakeMM:
            @staticmethod
            async def add(text, meta):
                added.append((text, meta))
        sys.modules['nyx.core.memory.memory_manager'] = types.SimpleNamespace(MemoryManager=FakeMM)

        # Fake strategy
        applied = []
        async def apply(insight):
            applied.append(insight)
        sys.modules['strategy'] = types.SimpleNamespace(manager=types.SimpleNamespace(apply=apply))

        spec = importlib.util.spec_from_file_location(
            'reflection.reflection_agent',
            'reflection/reflection_agent.py',
        )
        ra = importlib.util.module_from_spec(spec)
        sys.modules['reflection.reflection_agent'] = ra
        spec.loader.exec_module(ra)

        tasks = []
        monkeypatch.setattr(ra.asyncio, 'create_task', lambda coro: tasks.append(coro))

        # fill buffer and set old timestamp
        ra.log_buffer[:] = [types.SimpleNamespace(role='user', content='x')] * 100
        ra.last_reflection_time = ra.last_reflection_time - datetime.timedelta(hours=3)

        await ra.schedule_reflection()
        assert tasks, "reflection should schedule task"

        await tasks[0]
        assert added, "MemoryManager.add called"
        assert added[0][1]['priority'] == 'high'
        # apply should also have been scheduled
        assert len(tasks) > 1
        await tasks[1]
        assert applied and applied[0] == "insight\npriority: high"

    asyncio.run(inner())
