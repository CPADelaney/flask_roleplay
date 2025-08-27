# nyx/scene_keys.py  
import hashlib
from typing import Any, Iterable
from nyx.nyx_agent.context import json_dumps  # orjson-backed when available

def _as_list(x: Any) -> list:
    if x is None: return []
    if isinstance(x, (set, tuple, list)): return list(x)
    return [x]

def generate_scene_cache_key(scope_like: Any) -> str:
    data = {
        "location": getattr(scope_like, "location_id", None) or getattr(scope_like, "location", None),
        "npcs": sorted(_as_list(getattr(scope_like, "npc_ids", []))),
        "topics": sorted(_as_list(getattr(scope_like, "topics", []))),
        "lore": sorted(_as_list(getattr(scope_like, "lore_tags", []))),
        "conflicts": sorted(_as_list(getattr(scope_like, "conflict_ids", []))),
        "anchors": sorted(_as_list(getattr(scope_like, "memory_anchors", []))),
        "nations": sorted(_as_list(getattr(scope_like, "nation_ids", []))),
    }
    return hashlib.md5(json_dumps(data).encode("utf-8")).hexdigest()
