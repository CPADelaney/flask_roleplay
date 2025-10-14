"""Helpers for working with scene projection views.

This module provides utilities to normalize the JSON payloads returned by
``public.v_scene_context`` and related projections.  The projections surface
aggregated context that used to be assembled from many write tables.  The
helpers here coerce those JSON blobs back into Python-native structures so the
renderer stack can continue to operate without being aware of the underlying
storage.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _decode_value(value: Any) -> Any:
    """Attempt to coerce database projection values into native Python types."""

    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_decode_value(item) for item in value]

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    if _NUMBER_RE.match(text):
        try:
            return float(text) if "." in text else int(text)
        except ValueError:  # pragma: no cover - defensive
            logger.debug("Failed to coerce numeric string '%s'", text)

    return value


def _decode_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _decode_value(value) for key, value in mapping.items()}


def _normalize_personality_traits(value: Any) -> List[str]:
    decoded = _decode_value(value)
    if decoded is None:
        return []
    if isinstance(decoded, str):
        return [decoded]
    if isinstance(decoded, Iterable) and not isinstance(decoded, (str, bytes)):
        return [str(item) for item in decoded if item is not None]
    return []


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except ValueError:  # pragma: no cover - defensive
            return None
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            try:
                return int(text)
            except ValueError:  # pragma: no cover - defensive
                return None
    return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    sanitized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(sanitized)
    except ValueError:
        return None

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class SceneProjection:
    """Normalized view of the ``v_scene_context`` payload."""

    current_roleplay: Dict[str, Any]
    player_stats: Dict[str, Any]
    npcs: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    quests: List[Dict[str, Any]]

    @classmethod
    def empty(cls) -> "SceneProjection":
        return cls({}, {}, [], [], [])

    def current_location(self) -> Optional[str]:
        location = self.current_roleplay.get("CurrentLocation") or self.current_roleplay.get(
            "current_location"
        )
        result = _coerce_str(location)
        if result:
            return result
        return None

    def time_of_day(self) -> Optional[str]:
        return _coerce_str(self.current_roleplay.get("TimeOfDay"))

    def current_day(self) -> Optional[int]:
        return _coerce_int(self.current_roleplay.get("CurrentDay"))

    def roleplay_dict(self) -> Dict[str, Any]:
        return dict(self.current_roleplay)

    def player_stats_dict(self) -> Dict[str, Optional[float]]:
        return {key: _coerce_float(value) for key, value in self.player_stats.items()}

    def npc_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for npc in self.npcs:
            normalized = _decode_mapping(npc)
            normalized["personality_traits"] = _normalize_personality_traits(
                normalized.get("personality_traits")
            )
            normalized["dominance"] = _coerce_float(normalized.get("dominance"))
            normalized["cruelty"] = _coerce_float(normalized.get("cruelty"))
            normalized["closeness"] = _coerce_float(normalized.get("closeness"))
            normalized["trust"] = _coerce_float(normalized.get("trust"))
            normalized["respect"] = _coerce_float(normalized.get("respect"))
            normalized["intensity"] = _coerce_float(normalized.get("intensity"))
            normalized["affection"] = _coerce_float(normalized.get("affection"))
            normalized["introduced"] = bool(normalized.get("introduced", True))
            rows.append(normalized)

        rows.sort(
            key=lambda item: (
                -(item.get("closeness") or 0.0),
                -(item.get("trust") or 0.0),
                item.get("npc_name") or "",
            )
        )
        return rows

    def active_events(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        current_day = self.current_day()
        current_time = (self.time_of_day() or "").strip().lower()

        results: List[Dict[str, Any]] = []
        for event in self.events:
            data = _decode_mapping(event)
            event_day = _coerce_int(data.get("day"))
            event_time = _coerce_str(data.get("time_of_day"))
            if (
                current_day is not None
                and event_day == current_day
                and current_time
                and (event_time or "").strip().lower() == current_time
            ):
                results.append(
                    {
                        "event_name": data.get("event_name"),
                        "description": data.get("description"),
                        "location": data.get("location"),
                        "fantasy_level": data.get("fantasy_level"),
                    }
                )
                continue

            start_time = _parse_timestamp(data.get("start_time"))
            end_time = _parse_timestamp(data.get("end_time"))
            if start_time and end_time and start_time <= now <= end_time:
                results.append(
                    {
                        "event_name": data.get("event_name"),
                        "description": data.get("description"),
                        "location": data.get("location"),
                        "fantasy_level": data.get("fantasy_level"),
                    }
                )

        return results

    def active_quests(self) -> List[Dict[str, Any]]:
        active_status = {"active", "in_progress"}
        quests: List[Dict[str, Any]] = []
        for quest in self.quests:
            data = _decode_mapping(quest)
            status = (_coerce_str(data.get("status")) or "").lower()
            if status not in active_status:
                continue
            quests.append(data)
        return quests


def parse_scene_projection_row(row: Dict[str, Any]) -> SceneProjection:
    raw_scene_context = row.get("scene_context") or {}
    scene_context = _decode_value(raw_scene_context)
    if not isinstance(scene_context, dict):
        scene_context = {}
    current_roleplay = _decode_mapping(scene_context.get("current_roleplay") or {})
    player_stats = _decode_mapping(scene_context.get("player_stats") or {})
    npcs = [_decode_mapping(npc or {}) for npc in scene_context.get("npcs_present") or []]
    events = [_decode_mapping(evt or {}) for evt in scene_context.get("events") or []]
    quests = [_decode_mapping(quest or {}) for quest in scene_context.get("quests") or []]
    return SceneProjection(
        current_roleplay=current_roleplay,
        player_stats=player_stats,
        npcs=npcs,
        events=events,
        quests=quests,
    )

