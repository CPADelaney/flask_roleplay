"""Lightweight registry for conversation-scoped version counters."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from nyx.conversation.snapshot_store import ConversationSnapshotStore

logger = logging.getLogger(__name__)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class VersionRegistry:
    """Fetch version counters for conversations and conflicts."""

    def __init__(self) -> None:
        self._store = ConversationSnapshotStore()

    def get_counters(
        self,
        user_id: int,
        conversation_id: int,
        *,
        conflict_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Return available version counters for the conversation."""

        counters: Dict[str, int] = {}
        snapshot: Dict[str, Any] = {}
        try:
            snapshot = self._store.get(str(user_id), str(conversation_id)) or {}
        except Exception:  # pragma: no cover - defensive (redis/local store failures)
            logger.exception(
                "Failed to fetch snapshot for version counters: user_id=%s conversation_id=%s",
                user_id,
                conversation_id,
            )

        world_version = _coerce_int(snapshot.get("world_version"))
        if world_version is not None:
            counters["world"] = world_version

        if conflict_id is not None:
            conflict_version = self._extract_conflict_version(snapshot, conflict_id)
            if conflict_version is not None:
                counters["conflict"] = conflict_version

        return counters

    def _extract_conflict_version(
        self, snapshot: Dict[str, Any], conflict_id: str
    ) -> Optional[int]:
        if not conflict_id:
            return None
        conflict_key = str(conflict_id)
        mappings = []

        direct = snapshot.get("conflict_versions")
        if isinstance(direct, dict):
            mappings.append(direct)

        conflict_state = snapshot.get("conflict_state") or snapshot.get("conflict")
        if isinstance(conflict_state, dict):
            state_version = conflict_state.get("version")
            if state_version is not None:
                owner = conflict_state.get("id") or conflict_state.get("conflict_id")
                if owner is None or str(owner) == conflict_key:
                    coerced = _coerce_int(state_version)
                    if coerced is not None:
                        return coerced
            nested = conflict_state.get("versions")
            if isinstance(nested, dict):
                mappings.append(nested)

        conflicts_payload = snapshot.get("conflicts")
        if isinstance(conflicts_payload, dict):
            mappings.append(conflicts_payload)
            nested = conflicts_payload.get("by_id")
            if isinstance(nested, dict):
                mappings.append(nested)
        elif isinstance(conflicts_payload, list):
            for entry in conflicts_payload:
                if isinstance(entry, dict):
                    entry_id = entry.get("id") or entry.get("conflict_id")
                    if entry_id is not None and str(entry_id) == conflict_key:
                        for key in ("version", "counter", "turn", "turn_id"):
                            maybe = entry.get(key)
                            coerced = _coerce_int(maybe)
                            if coerced is not None:
                                return coerced

        history = snapshot.get("conflict_history")
        if isinstance(history, list):
            for entry in reversed(history):
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("payload")
                for candidate in (entry, payload):
                    if not isinstance(candidate, dict):
                        continue
                    entry_id = candidate.get("conflict_id") or candidate.get("id")
                    if entry_id is not None and str(entry_id) == conflict_key:
                        for key in ("version", "turn_id", "counter"):
                            maybe = candidate.get(key)
                            coerced = _coerce_int(maybe)
                            if coerced is not None:
                                return coerced
                turn_hint = entry.get("turn_id")
                coerced = _coerce_int(turn_hint)
                if coerced is not None:
                    return coerced

        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            for key_variant in (conflict_key, conflict_key.lstrip("#")):
                value = mapping.get(key_variant)
                if value is None and key_variant.isdigit():
                    value = mapping.get(int(key_variant))  # type: ignore[index]
                if value is None:
                    continue
                if isinstance(value, dict):
                    for nested_key in ("version", "counter", "turn", "turn_id"):
                        nested_value = value.get(nested_key)
                        coerced = _coerce_int(nested_value)
                        if coerced is not None:
                            return coerced
                coerced = _coerce_int(value)
                if coerced is not None:
                    return coerced

        return None


version_registry = VersionRegistry()

__all__ = ["VersionRegistry", "version_registry"]
