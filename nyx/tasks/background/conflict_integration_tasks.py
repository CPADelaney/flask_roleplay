"""Celery tasks for enhanced conflict integration slow-path work."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from celery import shared_task

from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_tension(payload: Dict[str, Any]) -> str:
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    scope_key = payload.get("scope_key")
    return f"conflict_integration:tension:{user_id}:{conversation_id}:{scope_key}"


def _idempotency_key_contextual(payload: Dict[str, Any]) -> str:
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    context_key = payload.get("context_key")
    return f"conflict_integration:contextual:{user_id}:{conversation_id}:{context_key}"


def _idempotency_key_activity(payload: Dict[str, Any]) -> str:
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    integration_key = payload.get("integration_key")
    activity = payload.get("activity")
    return f"conflict_integration:activity:{user_id}:{conversation_id}:{integration_key}:{activity}"


async def _analyze_scene_tension_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    from logic.conflict_system.enhanced_conflict_integration import (
        EnhancedIntegrationSubsystem,
    )

    subsystem = EnhancedIntegrationSubsystem(
        payload["user_id"], payload["conversation_id"]
    )
    scope_key = payload.get("scope_key", "")
    scene_context = payload.get("scene_context") or {}

    summary = await subsystem._analyze_scene_tension(scene_context)
    if scope_key:
        await subsystem._persist_tension_summary(scope_key, summary)
    return {"scope_key": scope_key, "summary": summary}


async def _generate_contextual_conflict_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    from logic.conflict_system.enhanced_conflict_integration import (
        EnhancedIntegrationSubsystem,
    )

    subsystem = EnhancedIntegrationSubsystem(
        payload["user_id"], payload["conversation_id"]
    )
    context_key = payload.get("context_key", "")
    tension_data = payload.get("tension_data") or {}
    npcs: List[int] = payload.get("npcs") or []

    conflict = await subsystem._generate_contextual_conflict(tension_data, npcs)
    if context_key:
        await subsystem._persist_contextual_conflict(context_key, conflict)
    return {"context_key": context_key, "conflict": conflict}


async def _integrate_activity_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    from logic.conflict_system.enhanced_conflict_integration import (
        EnhancedIntegrationSubsystem,
    )

    subsystem = EnhancedIntegrationSubsystem(
        payload["user_id"], payload["conversation_id"]
    )
    integration_key = payload.get("integration_key", "")
    activity = payload.get("activity", "")
    conflicts = payload.get("conflicts") or []

    integration = await subsystem._integrate_conflicts_with_activity(
        activity, conflicts
    )
    if integration_key:
        await subsystem._persist_activity_integration(integration_key, integration)
    return {"integration_key": integration_key, "integration": integration}


@shared_task(
    name="nyx.tasks.background.conflict_integration.analyze_scene_tension",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_tension)
def analyze_scene_tension(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run slow LLM tension analysis and persist the result."""
    logger.info(
        "Running scene tension analysis for conversation=%s scope=%s",
        payload.get("conversation_id"),
        payload.get("scope_key"),
    )
    result = run_coro(_analyze_scene_tension_async(payload))
    return {"status": "ok", **result}


@shared_task(
    name="nyx.tasks.background.conflict_integration.generate_contextual_conflict",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_contextual)
def generate_contextual_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a contextual conflict from tension data (slow LLM call)."""
    logger.info(
        "Generating contextual conflict for conversation=%s context=%s",
        payload.get("conversation_id"),
        payload.get("context_key"),
    )
    result = run_coro(_generate_contextual_conflict_async(payload))
    return {"status": "ok", **result}


@shared_task(
    name="nyx.tasks.background.conflict_integration.integrate_activity",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_activity)
def integrate_activity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate conflicts into an activity (slow LLM call)."""
    logger.info(
        "Integrating conflicts with activity=%s conversation=%s",
        payload.get("activity"),
        payload.get("conversation_id"),
    )
    result = run_coro(_integrate_activity_async(payload))
    return {"status": "ok", **result}

