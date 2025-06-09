"""Manage strategy parameters across runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .params import StrategyParams

# Default location for strategy state
STATE_PATH = Path("nyx_state.json")


class StrategyManager:
    """Load, modify and persist :class:`StrategyParams`."""

    def __init__(self, path: Path | str = STATE_PATH):
        self._path = Path(path)
        self._params = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> StrategyParams:
        if self._path.exists():
            try:
                with self._path.open("r") as f:
                    data = json.load(f)
                # if strategy params stored under key 'strategy', handle
                if set(data.keys()) <= set(StrategyParams.model_fields.keys()):
                    return StrategyParams.model_validate(data)
                if "strategy" in data:
                    return StrategyParams.model_validate(data["strategy"])
            except Exception:
                pass
        return StrategyParams()

    def _save(self) -> None:
        data = self._params.model_dump()
        # if file contains other data keep them
        if self._path.exists():
            try:
                with self._path.open("r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
            if isinstance(existing, dict) and "strategy" not in existing:
                existing.update(data)
                data = existing
            else:
                existing["strategy"] = data
                data = existing
        with self._path.open("w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    def update_tool_bias(self, tool_name: str, success_rate: float) -> None:
        """Boost bias for a tool if its success rate is high."""
        if success_rate <= 0.8:
            return
        cur = self._params.tool_biases.get(tool_name, 0.0)
        new_val = min(cur + 0.1, 1.0)
        if new_val != cur:
            self._params.tool_biases[tool_name] = new_val
            self._save()

    # ------------------------------------------------------------------
    def current(self) -> StrategyParams:
        return self._params

    async def apply(self, insight: str) -> None:
        """Adjust parameters based on a reflective insight."""
        text = insight.lower()
        changed = False
        if "math error" in text:
            new_val = min(self._params.precision_focus + 0.1, 1.0)
            if new_val != self._params.precision_focus:
                self._params.precision_focus = new_val
                changed = True

        if changed:
            self._save()


# Default manager instance used by agents
_manager = StrategyManager()


def current() -> StrategyParams:
    """Return current strategy parameters."""
    return _manager.current()


async def apply(insight: str) -> None:
    """Public wrapper for :meth:`StrategyManager.apply`."""
    await _manager.apply(insight)

def update_tool_bias(tool_name: str, success_rate: float) -> None:
    """Public wrapper for :meth:`StrategyManager.update_tool_bias`."""
    _manager.update_tool_bias(tool_name, success_rate)
