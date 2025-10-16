# nyx/locations/real_world_resolver.py
from __future__ import annotations
from typing import Any, Dict
from nyx.location.types import PlaceQuery, SettingProfile, ResolutionResult
from nyx.location.search import resolve_real as _canonical_resolve_real

async def resolve_real(query: PlaceQuery, setting: SettingProfile, meta: Dict[str, Any]) -> ResolutionResult:
    """
    Back-compat wrapper. Prefer nyx.location.search.resolve_real going forward.
    """
    return await _canonical_resolve_real(query, setting, meta)
