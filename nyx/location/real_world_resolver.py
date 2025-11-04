# nyx/locations/real_world_resolver.py
from __future__ import annotations
from typing import Any, Dict

from nyx.location.query import PlaceQuery
from nyx.location.types import Anchor, ResolveResult
from nyx.location.search import resolve_real as _canonical_resolve_real

async def resolve_real(
    query: PlaceQuery,
    anchor: Anchor,
    meta: Dict[str, Any],
    *,
    skip_gemini: bool = False,
) -> ResolveResult:
    """
    Back-compat wrapper. Prefer nyx.location.search.resolve_real going forward.
    """
    return await _canonical_resolve_real(query, anchor, meta, skip_gemini=skip_gemini)
