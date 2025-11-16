"""Shared helpers for conflict-related Celery signal producers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from conflict.signals import ConflictSignal, ConflictSignalType

if TYPE_CHECKING:
    from nyx.nyx_agent.context import SceneScope as SceneScopeType
else:  # pragma: no cover - runtime fallback for testing environments
    try:
        from nyx.nyx_agent.context import SceneScope as SceneScopeType
    except Exception:  # pragma: no cover - simplified stub
        @dataclass
        class SceneScopeType:  # type: ignore[override]
            location_id: Optional[int] = None
            npc_ids: set[int] = field(default_factory=set)
            topics: set[str] = field(default_factory=set)

            def to_cache_key(self) -> str:
                return "stub_scope"

            def to_dict(self) -> Dict[str, Any]:
                return {
                    'location_id': self.location_id,
                    'npc_ids': sorted(self.npc_ids),
                    'topics': sorted(self.topics),
                }

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'SceneScopeType':
                npc_ids = data.get('npc_ids') or []
                topics = data.get('topics') or []
                return cls(
                    location_id=data.get('location_id'),
                    npc_ids=set(npc_ids),
                    topics=set(topics),
                )


SceneScope = SceneScopeType

logger = logging.getLogger(__name__)


def _build_scene_scope(raw_scope: Dict[str, Any]) -> SceneScope:
    scope_fields = SceneScope.__dataclass_fields__.keys()
    payload: Dict[str, Any] = {}

    for field_name in scope_fields:
        if field_name == 'npc_ids':
            source_value = raw_scope.get('npc_ids', raw_scope.get('npcs'))
        elif field_name == 'topics':
            source_value = raw_scope.get('topics', raw_scope.get('conversation_topics'))
        else:
            source_value = raw_scope.get(field_name)

        if source_value is None:
            continue

        if isinstance(source_value, set):
            payload[field_name] = list(source_value)
        else:
            payload[field_name] = source_value

    try:
        return SceneScope.from_dict(payload)
    except Exception:
        return SceneScope()


async def generate_scene_conflict_context(
    synthesizer,
    user_id: int,
    conversation_id: int,
    scene_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a rich conflict context bundle without emitting signals."""

    scope_mapping: Dict[str, Any] = scene_info or {}

    fast_context = await synthesizer.get_fast_conflict_context_for_scene(scope_mapping)

    processor = getattr(synthesizer, "processor", None)
    if processor is not None:
        try:
            background_bundle = await processor.process_scene_relevant_updates(scope_mapping)
        except Exception:  # pragma: no cover - defensive background hook
            logger.exception(
                "Conflict background processor failed during scene context refresh",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
            )
        else:
            if background_bundle:
                metadata = fast_context.setdefault('metadata', {})
                metadata['background_refresh'] = background_bundle

    return fast_context


async def dispatch_tension_update_signal(
    synthesizer,
    user_id: int,
    conversation_id: int,
    payload_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Send a targeted time-tick to refresh tension metrics."""

    payload = {
        'update_tensions': True,
        'target_subsystems': ['tension'],
    }
    if payload_overrides:
        payload.update(payload_overrides)

    signal = ConflictSignal(
        type=ConflictSignalType.TIME_TICK,
        user_id=user_id,
        conversation_id=conversation_id,
        payload=payload,
    )

    await synthesizer.handle_signal(signal)
