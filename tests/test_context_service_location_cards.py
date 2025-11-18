import asyncio
import json
import os
import types

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DB_DSN", "postgresql://user:pass@localhost/testdb")

from context.context_service import ContextService  # noqa: E402  # pylint: disable=wrong-import-position
import context.context_service as context_service_module  # noqa: E402


def test_context_service_handles_stringified_location_cards(monkeypatch):
    async def fake_read_entity_cards(*_args, **_kwargs):
        return [
            {
                "entity_id": "loc-77",
                "card": json.dumps(
                    {
                        "location_id": 77,
                        "location_name": "Azure Library",
                        "description": "Stacks of knowledge reaching the ceiling.",
                        "connected_locations": ["Atrium"],
                    }
                ),
            }
        ]

    monkeypatch.setattr(context_service_module, "read_entity_cards", fake_read_entity_cards)

    service = ContextService(user_id=1, conversation_id=2)
    service.vector_service = None
    service.config = types.SimpleNamespace(is_enabled=lambda *_: False)

    result = asyncio.run(service._get_location_details("Azure Library"))

    assert result.location_id == "77"
    assert result.location_name == "Azure Library"
    assert result.description.startswith("Stacks of knowledge")
    assert result.connected_locations == ["Atrium"]
