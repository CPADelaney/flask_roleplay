from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlaceQuery:
    raw_text: str
    normalized: str
    is_travel: bool = False
    target: Optional[str] = None
    transport_hint: Optional[str] = None


__all__ = ["PlaceQuery"]
