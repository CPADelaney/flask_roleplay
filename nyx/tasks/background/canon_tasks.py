"""Background tasks for conflict canon slow-path operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

from nyx.tasks.base import NyxTask, app

from db.connection import get_db_connection_context
from infra.cache import cache_key, set_json
from logic.conflict_system.conflict_canon import (
    CanonEventType,
    CanonicalEvent,
    ConflictCanonSubsystem,
)
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from lore.core.canon import log_canonical_event
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent
from nyx.gateway.llm_gateway import execute, execute_stream, LLMRequest, LLMOperation

logger = logging.getLogger(__name__)


def _require_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer") from exc


def _idempotency_key_canonize(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    resolution_hash = hash(str(payload.get("resolution", {})))
    return f"canonize:{conflict_id}:{resolution_hash}"


def _idempotency_key_references(payload: Dict[str, Any]) -> str:
    cache_id = payload.get("cache_id")
    event_id = payload.get("event_id")
    return f"canon_refs:{cache_id}:{event_id}"


def _idempotency_key_suggestions(payload: Dict[str, Any]) -> str:
    cache_id = payload.get("cache_id")
    conflict_type = payload.get("conflict_type")
    return f"canon_suggestions:{cache_id}:{conflict_type}"


def _idempotency_key_mythology(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    return f"canon_myth:{conflict_id}"


async def _evaluate_canon_pipeline(
    subsystem: ConflictCanonSubsystem,
    conflict_id: int,
    resolution: Dict[str, Any],
) -> Dict[str, Any]:
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow(
            """
            SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
            """,
            conflict_id,
        )
        if not conflict:
            return {
                "became_canonical": False,
                "reason": "Conflict not found",
                "significance": 0.0,
                "tags": [],
                "canonical_event_id": 0,
                "pending": False,
            }
        stakeholders = await conn.fetch(
            """
            SELECT * FROM ConflictStakeholders WHERE conflict_id = $1
            """,
            conflict_id,
        )

    prompt = f"""
Evaluate if this conflict resolution should become canonical:

Conflict: {conflict['conflict_name']}
Type: {conflict['conflict_type']}
Resolution: {json.dumps(resolution)}
Stakeholders: {len(stakeholders)}

Return JSON:
{{
  "should_be_canonical": true/false,
  "reason": "Why this matters (or doesn't)",
  "event_type": "historical_precedent|cultural_shift|relationship_milestone|power_restructuring|social_evolution|legendary_moment|tradition_born|taboo_broken",
  "significance": 0.0
}}
"""
    request = LLMRequest(
        prompt=prompt,
        agent=subsystem.lore_integrator,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "canon_evaluation",
        },
    )
    result = await execute(request)
    response = result.raw
    data = json.loads(extract_runner_response(response))
    should = bool(data.get("should_be_canonical", False))
    reason = data.get("reason", "")
    significance = float(data.get("significance", 0.0) or 0.0)

    if not should:
        return {
            "became_canonical": False,
            "reason": reason,
            "significance": significance,
            "tags": [],
            "canonical_event_id": 0,
            "pending": False,
        }

    event_type_str = data.get("event_type", CanonEventType.LEGENDARY_MOMENT.value)
    event_type = CanonEventType(event_type_str)
    event, tags = await _create_canonical_event_pipeline(
        subsystem,
        conflict_id,
        dict(conflict),
        resolution,
        event_type,
        significance,
    )

    return {
        "became_canonical": True,
        "reason": reason,
        "significance": significance,
        "tags": tags,
        "canonical_event_id": int(event.event_id),
        "legacy": event.legacy,
        "name": event.name,
        "creates_precedent": event.creates_precedent,
        "pending": False,
    }


async def _create_canonical_event_pipeline(
    subsystem: ConflictCanonSubsystem,
    conflict_id: int,
    conflict: Dict[str, Any],
    resolution: Dict[str, Any],
    event_type: CanonEventType,
    significance: float,
) -> Tuple[CanonicalEvent, List[str]]:
    prompt = f"""
Create a canonical description for this event:

Conflict: {conflict['conflict_name']}
Resolution: {json.dumps(resolution)}
Event Type: {event_type.value}
Significance: {significance:.2f}

Return JSON:
{{
  "canonical_name": "How history will remember this",
  "canonical_description": "2-3 sentence historical record",
  "cultural_impact": {{
    "immediate": "How society reacts",
    "long_term": "Cultural changes over time",
    "traditions_affected": ["existing traditions impacted"],
    "new_traditions": ["potential new traditions"]
  }},
  "creates_precedent": true/false,
  "precedent_description": "What precedent if any"
}}
"""
    request = LLMRequest(
        prompt=prompt,
        agent=subsystem.cultural_interpreter,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "canon_event",
        },
    )
    result = await execute(request)
    response = result.raw
    event_payload = json.loads(extract_runner_response(response))

    legacy = await _generate_legacy_pipeline(subsystem, conflict, resolution, event_payload)

    tags = [
        "conflict",
        "resolution",
        event_type.value,
        conflict.get("conflict_type", "unknown"),
        f"conflict_id_{conflict_id}",
        "precedent" if event_payload.get("creates_precedent") else "event",
    ]

    async with get_db_connection_context() as conn:
        await log_canonical_event(
            subsystem.ctx,
            conn,
            f"{event_payload['canonical_name']}: {event_payload['canonical_description']}",
            tags=tags,
            significance=int(max(1, min(10, round(significance * 10)))),
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conflict_canon_details (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                conflict_id INTEGER,
                event_type TEXT,
                cultural_impact JSONB,
                creates_precedent BOOLEAN,
                legacy TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        event_id = await conn.fetchval(
            """
            INSERT INTO conflict_canon_details
                (user_id, conversation_id, conflict_id, event_type, cultural_impact,
                 creates_precedent, legacy, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            subsystem.user_id,
            subsystem.conversation_id,
            conflict_id,
            event_type.value,
            json.dumps(event_payload.get("cultural_impact", {})),
            bool(event_payload.get("creates_precedent", False)),
            legacy,
            json.dumps({"name": event_payload.get("canonical_name", "")}),
        )

    canonical_event = CanonicalEvent(
        event_id=event_id,
        conflict_id=conflict_id,
        event_type=event_type,
        name=event_payload.get("canonical_name", ""),
        description=event_payload.get("canonical_description", ""),
        significance=significance,
        cultural_impact=event_payload.get("cultural_impact", {}),
        referenced_by=[],
        creates_precedent=bool(event_payload.get("creates_precedent", False)),
        legacy=legacy,
    )

    await subsystem._persist_canon_summary(conflict_id, canonical_event, tags)

    return canonical_event, tags


async def _generate_legacy_pipeline(
    subsystem: ConflictCanonSubsystem,
    conflict: Dict[str, Any],
    resolution: Dict[str, Any],
    cultural_data: Dict[str, Any],
) -> str:
    prompt = f"""
Write the lasting legacy of this canonical event:

Event: {cultural_data.get('canonical_name','')}
Description: {cultural_data.get('canonical_description','')}
Cultural Impact: {json.dumps(cultural_data.get('cultural_impact', {}))}
"""
    request = LLMRequest(
        prompt=prompt,
        agent=subsystem.legacy_writer,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "canon_legacy",
        },
    )
    result = await execute(request)
    response = result.raw
    return extract_runner_response(response)


async def _build_reference_cache_pipeline(
    subsystem: ConflictCanonSubsystem,
    cache_id: int,
    event_id: int,
    context: str,
) -> None:
    try:
        async with get_db_connection_context() as conn:
            event = await conn.fetchrow(
                """
                SELECT * FROM CanonicalEvents
                 WHERE user_id = $1 AND conversation_id = $2
                   AND id = $3
                """,
                subsystem.user_id,
                subsystem.conversation_id,
                event_id,
            )

            if not event:
                details = await conn.fetchrow(
                    """
                    SELECT * FROM conflict_canon_details WHERE id = $1
                    """,
                    event_id,
                )
                if not details:
                    await subsystem.update_reference_cache(cache_id, [], status="failed")
                    return
                event_dict = {
                    "event_text": (details.get("metadata") or {}).get("name", "Unknown Event"),
                    "tags": ["conflict", details.get("event_type", "")],
                    "significance": 5,
                }
            else:
                event_dict = dict(event)

        prompt = f"""
Generate NPC references to this canonical event:

Event: {event_dict.get('event_text','')}
Tags: {event_dict.get('tags', [])}
Significance: {event_dict.get('significance', 5)}
Context: {context}

Return JSON:
{{ "references": [{{"text": "..."}}] }}
"""
        request = LLMRequest(
            prompt=prompt,
            agent=subsystem.reference_generator,
            metadata={
                "operation": LLMOperation.ORCHESTRATION.value,
                "stage": "canon_references",
            },
        )
        result = await execute(request)
        response = result.raw
        data = json.loads(extract_runner_response(response))
        refs = [ref.get("text", "") for ref in data.get("references", [])]
        await subsystem.update_reference_cache(cache_id, refs, status="ready")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to build reference cache", exc_info=exc)
        await subsystem.update_reference_cache(cache_id, [], status="failed")


async def _generate_mythology_pipeline(
    subsystem: ConflictCanonSubsystem,
    conflict_id: int,
) -> str:
    async with get_db_connection_context() as conn:
        conflict = await conn.fetchrow(
            """
            SELECT *, id as conflict_id FROM Conflicts WHERE id = $1
            """,
            conflict_id,
        )
        canonical_events = await conn.fetch(
            """
            SELECT event_text, significance FROM CanonicalEvents
             WHERE user_id = $1 AND conversation_id = $2
               AND tags ? $3
             ORDER BY significance DESC
            """,
            subsystem.user_id,
            subsystem.conversation_id,
            f"conflict_id_{conflict_id}",
        )

    if not canonical_events:
        return "This conflict has not yet become part of the canonical lore."

    prompt = f"""
Generate the mythological interpretation of this conflict:

Conflict: {conflict['conflict_name'] if conflict else f'Conflict {conflict_id}'}
Type: {conflict['conflict_type'] if conflict else 'Unknown'}
Canonical Events: {json.dumps([dict(e) for e in canonical_events])}

Write 2-3 paragraphs of authentic folklore.
"""
    request = LLMRequest(
        prompt=prompt,
        agent=subsystem.cultural_interpreter,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "canon_mythology",
        },
    )
    result = await execute(request)
    response = result.raw
    return extract_runner_response(response)


async def _canonize_background(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    resolution: Dict[str, Any],
    snapshot_id: int | None,
) -> Dict[str, Any]:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)

    try:
        result = await _evaluate_canon_pipeline(subsystem, conflict_id, resolution)
        if snapshot_id:
            status = 'completed' if not result.get('pending') else 'pending'
            await subsystem._mark_snapshot_status(snapshot_id, status, result=result)
        return result
    except Exception as exc:
        logger.exception("Canonization failed for conflict %s", conflict_id)
        if snapshot_id:
            await subsystem._mark_snapshot_status(
                snapshot_id,
                'failed',
                result={'became_canonical': False},
                error=str(exc),
            )
        raise


@app.task(base=NyxTask, name="canon.canonize_conflict",
    bind=True,
    max_retries=3,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_canonize)
def canonize_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the slow canon evaluation pipeline."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    conflict_id = _require_int(payload.get("conflict_id"), "conflict_id")
    snapshot_id = payload.get("snapshot_id")
    if snapshot_id is not None:
        snapshot_id = _require_int(snapshot_id, "snapshot_id")
    resolution = payload.get("resolution") or {}

    logger.info(
        "Canonizing conflict %s for user=%s conversation=%s", conflict_id, user_id, conversation_id
    )

    result = run_coro(
        _canonize_background(user_id, conversation_id, conflict_id, resolution, snapshot_id)
    )

    return {
        "status": "completed" if result.get("became_canonical") else "queued",
        "conflict_id": conflict_id,
        "result": result,
    }


async def _reference_background(
    user_id: int,
    conversation_id: int,
    cache_id: int,
    event_id: int,
    context: str,
) -> None:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    await _build_reference_cache_pipeline(subsystem, cache_id, event_id, context)


@app.task(base=NyxTask, name="canon.generate_canon_references",
    bind=True,
    max_retries=3,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_references)
def generate_canon_references(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and persist canon reference cache entries."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    cache_id = _require_int(payload.get("cache_id"), "cache_id")
    event_id = _require_int(payload.get("event_id"), "event_id")
    context = str(payload.get("context", "casual"))

    logger.info(
        "Generating canon references for event %s (cache=%s, user=%s conv=%s)",
        event_id,
        cache_id,
        user_id,
        conversation_id,
    )

    run_coro(_reference_background(user_id, conversation_id, cache_id, event_id, context))

    return {
        "status": "queued",
        "event_id": event_id,
        "cache_id": cache_id,
    }


async def _generate_compliance_suggestions(
    subsystem: ConflictCanonSubsystem,
    cache_id: int,
    conflict_type: str,
    conflict_context: Dict[str, Any],
    matching_event_ids: List[int],
) -> None:
    try:
        related_events: List[Any] = []
        if matching_event_ids:
            async with get_db_connection_context() as conn:
                related_events = await conn.fetch(
                    """
                    SELECT id, event_text, tags, significance
                      FROM CanonicalEvents
                     WHERE user_id = $1 AND conversation_id = $2
                       AND id = ANY($3::int[])
                    """,
                    subsystem.user_id,
                    subsystem.conversation_id,
                    matching_event_ids,
                )

        prompt = f"""
Assess lore guidance for the following conflict.

Conflict Type: {conflict_type}
Context: {json.dumps(conflict_context)}
Matching Canonical Events: {json.dumps([dict(row) for row in related_events])}

Return JSON: {{"suggestions": ["specific player-facing suggestion"]}}
"""
        request = LLMRequest(
            prompt=prompt,
            agent=subsystem.precedent_analyzer,
            metadata={
                "operation": LLMOperation.ORCHESTRATION.value,
                "stage": "canon_compliance",
            },
        )
        result = await execute(request)
        response = result.raw
        data = json.loads(extract_runner_response(response))
        suggestions = [str(s) for s in (data.get("suggestions") or [])]
        await subsystem.update_compliance_suggestions(
            cache_id,
            suggestions,
            status="ready",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build compliance suggestions", exc_info=exc)
        await subsystem.update_compliance_suggestions(
            cache_id,
            [],
            status="failed",
            error=str(exc),
        )


async def _suggestions_background(
    user_id: int,
    conversation_id: int,
    cache_id: int,
    conflict_type: str,
    conflict_context: Dict[str, Any],
    matching_event_ids: List[int],
) -> None:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    await _generate_compliance_suggestions(
        subsystem,
        cache_id,
        conflict_type,
        conflict_context,
        matching_event_ids,
    )


@app.task(base=NyxTask, name="canon.generate_lore_suggestions",
    bind=True,
    max_retries=3,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_suggestions)
def generate_lore_suggestions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate lore compliance suggestions asynchronously."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    cache_id = _require_int(payload.get("cache_id"), "cache_id")
    conflict_type = str(payload.get("conflict_type", "unknown"))
    conflict_context = payload.get("conflict_context") or {}
    matching_event_ids = [int(e) for e in (payload.get("matching_event_ids") or [])]

    logger.info(
        "Generating lore suggestions for cache %s (user=%s conv=%s type=%s)",
        cache_id,
        user_id,
        conversation_id,
        conflict_type,
    )

    run_coro(
        _suggestions_background(
            user_id,
            conversation_id,
            cache_id,
            conflict_type,
            conflict_context,
            matching_event_ids,
        )
    )

    return {"status": "queued", "cache_id": cache_id}


async def _mythology_background(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
) -> str:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    mythology_text = await _generate_mythology_pipeline(subsystem, conflict_id)

    cache_payload = {
        "text": mythology_text,
        "created_at": datetime.utcnow().isoformat(),
    }
    set_json(cache_key("canon", "mythology", conflict_id), cache_payload, ex=3600)
    return mythology_text


@app.task(base=NyxTask, name="canon.generate_mythology",
    bind=True,
    max_retries=3,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_mythology)
def generate_mythology_reinterpretation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate mythological reinterpretation text for a conflict."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    conflict_id = _require_int(payload.get("conflict_id"), "conflict_id")

    logger.info(
        "Generating mythology reinterpretation for conflict %s (user=%s conv=%s)",
        conflict_id,
        user_id,
        conversation_id,
    )

    mythology_text = run_coro(_mythology_background(user_id, conversation_id, conflict_id))

    return {
        "status": "generated",
        "conflict_id": conflict_id,
        "mythology": mythology_text,
    }


__all__ = [
    "canonize_conflict",
    "generate_canon_references",
    "generate_lore_suggestions",
    "generate_mythology_reinterpretation",
]
