"""Lightweight Nyx settings facade."""

from __future__ import annotations

import os
from typing import Final


def _is_truthy(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on", "enabled"}


def _read_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value is None:
        return False
    return _is_truthy(value)


CONFLICT_EAGER_WARMUP: Final[bool] = _read_bool("NYX_CONFLICT_EAGER_WARMUP", "0")
"""Gate conflict eager warm-up behaviour for context warmers."""


__all__ = ["CONFLICT_EAGER_WARMUP"]
