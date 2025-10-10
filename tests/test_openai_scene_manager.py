import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_integration.scene_manager import SceneManager  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_rotate_if_needed_persists_scene(monkeypatch):
    calls: list[Dict[str, Any]] = []
    expected_scene = {"id": 11, "scene_number": 2, "scene_title": "New"}

    async def _fake_rotate(
        *,
        conversation_id: int,
        new_scene: Dict[str, Any],
        closing_scene: Optional[Dict[str, Any]] = None,
        conn=None,
    ):
        calls.append(
            {
                "conversation_id": conversation_id,
                "new_scene": new_scene,
                "closing_scene": closing_scene,
                "conn": conn,
            }
        )
        return expected_scene

    monkeypatch.setattr(
        "openai_integration.scene_manager.rotate_conversation_scene",
        _fake_rotate,
    )

    manager = SceneManager(conversation_id=42)
    new_scene = {"scene_title": "New"}
    closing_scene = {"scene_summary": "Ended"}

    result = await manager.rotate_if_needed(
        new_scene=new_scene,
        closing_scene=closing_scene,
    )

    assert calls == [
        {
            "conversation_id": 42,
            "new_scene": new_scene,
            "closing_scene": closing_scene,
            "conn": None,
        }
    ]
    assert result == expected_scene
    assert manager.current_scene == expected_scene


@pytest.mark.anyio("asyncio")
async def test_rotate_if_needed_handles_missing_new_scene(monkeypatch):
    async def _fake_rotate(**kwargs):  # pragma: no cover - should not be hit
        raise AssertionError("rotate_conversation_scene should not be called")

    monkeypatch.setattr(
        "openai_integration.scene_manager.rotate_conversation_scene",
        _fake_rotate,
    )

    manager = SceneManager(conversation_id=1)
    result = await manager.rotate_if_needed(new_scene=None)

    assert result is None
    assert manager.current_scene is None
