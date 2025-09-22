import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from logic import addiction_system_sdk as sdk


@pytest.mark.asyncio
async def test_thematic_messages_deferred_generation(monkeypatch, tmp_path):
    sdk.ThematicMessages._instance = None

    generated_timeouts = []

    async def fake_generate(user_id, conversation_id, addiction_types=sdk.ADDICTION_TYPES, governor=None, timeout=None):
        generated_timeouts.append(timeout)
        return {
            addiction: {str(level): f"{addiction}-{level}" for level in range(1, 5)}
            for addiction in addiction_types
        }

    monkeypatch.setattr(sdk, "generate_thematic_messages_via_agent", fake_generate)
    monkeypatch.setattr(sdk, "THEMATIC_MESSAGES_FILE", str(tmp_path / "thematic.json"))

    thematic = await sdk.ThematicMessages.get(
        user_id=1,
        conversation_id=2,
        refresh=True,
        defer_generation=True,
    )

    assert generated_timeouts == []

    await thematic.ensure_generated(generation_timeout=0.1)

    assert generated_timeouts == [0.1]
    assert thematic._generated is True
    assert thematic.get_for("feet", 1) == "feet-1"


@pytest.mark.asyncio
async def test_addiction_context_lazy_generation(monkeypatch):
    sdk.ThematicMessages._instance = None

    generate_timeouts = []

    async def fake_generate(user_id, conversation_id, addiction_types=sdk.ADDICTION_TYPES, governor=None, timeout=None):
        generate_timeouts.append(timeout)
        return {
            addiction: {str(level): f"{addiction}-{level}" for level in range(1, 5)}
            for addiction in addiction_types
        }

    monkeypatch.setattr(sdk, "generate_thematic_messages_via_agent", fake_generate)

    from nyx import integrate as nyx_integrate

    async def fake_get_central_governance(user_id, conversation_id):
        class DummyGovernor:
            async def check_action_permission(self, *args, **kwargs):
                return {"approved": True}

        return DummyGovernor()

    monkeypatch.setattr(nyx_integrate, "get_central_governance", fake_get_central_governance)

    async def fake_get_instance(cls, user_id, conversation_id):
        class DummyLore:
            pass

        return DummyLore()

    monkeypatch.setattr(sdk.LoreSystem, "get_instance", classmethod(fake_get_instance))

    class DummyDirectiveHandler:
        def __init__(self, *args, **kwargs):
            self.handlers = []

        def register_handler(self, *args, **kwargs):
            self.handlers.append((args, kwargs))

        def start_background_processing(self, interval=60.0):
            return None

    monkeypatch.setattr(sdk, "DirectiveHandler", DummyDirectiveHandler)

    ctx = sdk.AddictionContext(user_id=3, conversation_id=4)
    await ctx.initialize()

    assert generate_timeouts == []

    thematic = await ctx.get_thematic_messages(require_generation=True)
    assert generate_timeouts == [ctx.thematic_generation_timeout]
    assert thematic._generated is True

    # Subsequent calls without force should reuse the cached generation
    await ctx.get_thematic_messages(require_generation=True)
    assert len(generate_timeouts) == 1

    assert thematic.get_for("feet", 2) == "feet-2"
