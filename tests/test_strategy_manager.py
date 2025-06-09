import json
import pytest
from strategy.manager import StrategyManager

@pytest.mark.asyncio
async def test_apply_updates_and_saves(tmp_path):
    path = tmp_path / "nyx_state.json"
    mgr = StrategyManager(path)
    await mgr.apply("Too many math errors in output")
    assert mgr.current().precision_focus == 0.6
    with open(path) as f:
        data = json.load(f)
    assert data["precision_focus"] == 0.6


def test_load_existing(tmp_path):
    path = tmp_path / "nyx_state.json"
    with open(path, "w") as f:
        json.dump({"precision_focus": 0.7, "exploration_rate": 0.1,
                   "risk_tolerance": 0.3, "creativity": 0.4}, f)
    mgr = StrategyManager(path)
    assert pytest.approx(mgr.current().precision_focus, 0.0001) == 0.7
