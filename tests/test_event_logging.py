import sys
from pathlib import Path
import types
from datetime import datetime

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

dummy = types.ModuleType("logic.game_time_helper")


class DummyGameTimeContext:
    def __init__(self, user_id, conversation_id):
        self.year = 2
        self.month = 3
        self.day = 4
        self.time_of_day = "Evening"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def to_datetime(self):
        return datetime(2000, 1, 1)

    async def to_string(self, include_date=True, include_time=True):
        return "dummy"


dummy.GameTimeContext = DummyGameTimeContext
sys.modules["logic.game_time_helper"] = dummy

from logic import event_logging

@pytest.mark.asyncio
async def test_log_event(monkeypatch):
    event = await event_logging.log_event(1, 1, "test", {"foo": "bar"})
    assert event["event_type"] == "test"
    assert event["game_time"] == {"year": 2, "month": 3, "day": 4, "time_of_day": "Evening"}
    assert "time_string" in event
    assert event["foo"] == "bar"
