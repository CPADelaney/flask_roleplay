import types
import asyncio
import json
import sys

import pytest

from nyx.core.logging import summarizer


@pytest.mark.asyncio
async def test_nightly_rollup(monkeypatch, tmp_path):
    log_dir = tmp_path
    file = log_dir / "2024-01-01.jsonl"
    file.write_text(json.dumps({"a":1}) + "\n" + json.dumps({"b":2}))

    monkeypatch.setattr(summarizer, "LOG_DIR", str(log_dir))

    added = []
    class FakeMM:
        @staticmethod
        async def add(text, meta):
            added.append((text, meta))
    monkeypatch.setattr(summarizer, "MemoryManager", FakeMM)

    async def fake_acreate(*args, **kwargs):
        async def gen():
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="summary"))])
        return gen()
    monkeypatch.setattr(summarizer.openai.ChatCompletion, "acreate", fake_acreate)

    await summarizer.nightly_rollup("2024-01-01")

    assert added
    assert len(added) == 1
    assert added[0][0] == "summary"
    assert added[0][1]["type"] == "daily_summary"
    assert added[0][1]["date"] == "2024-01-01"
