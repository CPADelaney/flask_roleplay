"""Celery tasks for conflict edge-case recovery generation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import redis
from celery import shared_task

from agents import Agent
from logic.conflict_system.dynamic_conflict_template import (
    extract_runner_response,
)
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

logger = logging.getLogger(__name__)


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RECOVERY_CACHE_TTL = int(os.getenv("EDGE_RECOVERY_CACHE_TTL", "900"))
RECOVERY_LOCK_TTL = int(os.getenv("EDGE_RECOVERY_LOCK_TTL", "300"))

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def recovery_cache_key(
    user_id: int | str,
    conversation_id: int | str,
    case_type: str,
    case_ref: str,
) -> str:
    """Build the redis cache key for a recovery payload."""

    return f"edge_case_recovery:{user_id}:{conversation_id}:{case_type}:{case_ref}"


def recovery_lock_key(
    user_id: int | str,
    conversation_id: int | str,
    case_type: str,
    case_ref: str,
) -> str:
    """Build the redis lock key used to dedupe queueing."""

    return f"lock:{recovery_cache_key(user_id, conversation_id, case_type, case_ref)}"


def _store_result(cache_key: str, payload: Dict[str, Any]) -> None:
    client = _get_redis()
    client.set(cache_key, json.dumps(payload), ex=RECOVERY_CACHE_TTL)


def _release_lock(lock_key: str) -> None:
    try:
        _get_redis().delete(lock_key)
    except Exception:
        logger.exception("Failed to release recovery lock", exc_info=True)


def _run_runner(agent: Agent, prompt: str) -> Any:
    async def _run() -> Any:
        return await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
            )
        )

    return asyncio.run(_run())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_base(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "options": options,
        "generated_at": _timestamp(),
    }


def _orphan_agent() -> Agent:
    return Agent(
        name="Recovery Strategist",
        instructions="""
        Design recovery strategies for conflict system problems.

        Create strategies that:
        - Preserve narrative continuity
        - Minimize player disruption
        - Maintain conflict integrity
        - Enable graceful recovery
        - Prevent recurrence

        Turn problems into opportunities when possible.
        """,
        model="gpt-5-nano",
    )


def _healer_agent() -> Agent:
    return Agent(
        name="Narrative Healer",
        instructions="""
        Heal narrative breaks and continuity issues.

        Create fixes that:
        - Explain inconsistencies
        - Bridge narrative gaps
        - Justify sudden changes
        - Maintain immersion
        - Feel intentional

        Make broken stories whole again.
        """,
        model="gpt-5-nano",
    )


def _degrader_agent() -> Agent:
    return Agent(
        name="Graceful Degradation Manager",
        instructions="""
        Manage graceful degradation when systems fail.

        Ensure:
        - Core experience preserved
        - Fallbacks feel natural
        - Complexity reduced smoothly
        - Player experience maintained
        - Recovery paths clear

        When things break, break beautifully.
        """,
        model="gpt-5-nano",
    )


def _build_orphan_prompt(conflict: Dict[str, Any]) -> str:
    return f"""
    Generate recovery options for orphaned conflict:

    Conflict: {conflict.get('conflict_name', 'Unknown')}
    Description: {conflict.get('description', '')}

    Create 3 recovery options:
    1. Graceful closure
    2. NPC assignment
    3. Player-centric pivot

    Return JSON:
    {{
        "options": [
            {{
                "strategy": "close/assign/pivot",
                "description": "How to recover",
                "narrative": "Story explanation",
                "implementation": ["steps to take"],
                "risk": "low/medium/high"
            }}
        ]
    }}
    """


def _build_stale_prompt(conflict: Dict[str, Any]) -> str:
    return f"""
    Generate recovery for stale conflict:

    Conflict: {conflict.get('conflict_name', 'Unknown')}
    Phase: {conflict.get('phase', 'unknown')}
    Progress: {conflict.get('progress', 0)}%

    Create options to:
    - Revitalize with new development
    - Gracefully conclude
    - Transform into something new

    Return JSON:
    {{
        "options": [
            {{
                "strategy": "revitalize/conclude/transform",
                "description": "Recovery approach",
                "narrative_event": "What happens in story",
                "player_hook": "How to re-engage player",
                "expected_outcome": "What this achieves"
            }}
        ]
    }}
    """


def _build_loop_prompt(conflicts: List[int]) -> str:
    return f"""
    Generate recovery for infinite conflict loop:

    Conflicts involved: {conflicts}

    Create solutions that:
    - Break the cycle
    - Preserve both conflicts if possible
    - Maintain narrative sense

    Return JSON:
    {{
        "options": [
            {{
                "strategy": "break/merge/prioritize",
                "description": "How to break loop",
                "preserved_conflicts": [conflict_ids],
                "narrative_bridge": "Story explanation",
                "prevention": "How to prevent recurrence"
            }}
        ]
    }}
    """


def _build_overload_prompt(count: int) -> str:
    return f"""
    Generate recovery for conflict overload:

    Active conflicts: {count}

    Create strategies to:
    - Reduce complexity gracefully
    - Merge related conflicts
    - Prioritize important conflicts
    - Create breathing room

    Return JSON:
    {{
        "options": [
            {{
                "strategy": "consolidate/prioritize/pause/resolve",
                "target_count": ideal number of conflicts,
                "selection_criteria": "How to choose which conflicts",
                "narrative_framing": "Story explanation",
                "player_communication": "How to explain to player"
            }}
        ]
    }}
    """


def _build_contradiction_prompt(contradiction: Dict[str, Any]) -> str:
    return f"""
    Generate recovery for contradictory NPC positions:

    Conflict 1: {contradiction.get('name1', 'Unknown')}
    Conflict 2: {contradiction.get('name2', 'Unknown')}

    NPC has contradictory roles in these conflicts.

    Create solutions that:
    - Explain the contradiction
    - Resolve the inconsistency
    - Maintain character integrity

    Return JSON:
    {{
        "options": [
            {{
                "strategy": "explain/choose/split",
                "narrative_explanation": "How to explain in story",
                "character_development": "How this develops NPC",
                "conflict_impact": {{
                    "conflict1": "impact on first conflict",
                    "conflict2": "impact on second conflict"
                }}
            }}
        ]
    }}
    """


def _handle_generation(
    cache_key: str,
    lock_key: str,
    prompt: str,
    agent_factory,
) -> Dict[str, Any] | None:
    try:
        response = _run_runner(agent_factory(), prompt)
        data = json.loads(extract_runner_response(response))
        options = data.get("options", [])
        if not isinstance(options, list):
            options = []
        payload = _payload_base(options)
        _store_result(cache_key, payload)
        return payload
    except Exception:
        logger.exception("Failed to generate recovery options")
        return None
    finally:
        _release_lock(lock_key)


@shared_task(
    name="nyx.tasks.background.conflict_edge_tasks.generate_orphan_recovery",
    queue="background",
    acks_late=True,
)
def generate_orphan_recovery(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    conflict = payload.get("conflict", {})
    cache_key = payload.get("cache_key", "")
    lock_key = payload.get("lock_key", "")
    return _handle_generation(cache_key, lock_key, _build_orphan_prompt(conflict), _orphan_agent)


@shared_task(
    name="nyx.tasks.background.conflict_edge_tasks.generate_stale_recovery",
    queue="background",
    acks_late=True,
)
def generate_stale_recovery(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    conflict = payload.get("conflict", {})
    cache_key = payload.get("cache_key", "")
    lock_key = payload.get("lock_key", "")
    return _handle_generation(cache_key, lock_key, _build_stale_prompt(conflict), _orphan_agent)


@shared_task(
    name="nyx.tasks.background.conflict_edge_tasks.generate_loop_recovery",
    queue="background",
    acks_late=True,
)
def generate_loop_recovery(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    conflicts = payload.get("conflicts", [])
    cache_key = payload.get("cache_key", "")
    lock_key = payload.get("lock_key", "")
    return _handle_generation(cache_key, lock_key, _build_loop_prompt(conflicts), _orphan_agent)


@shared_task(
    name="nyx.tasks.background.conflict_edge_tasks.generate_overload_recovery",
    queue="background",
    acks_late=True,
)
def generate_overload_recovery(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    count = int(payload.get("active_count", 0))
    cache_key = payload.get("cache_key", "")
    lock_key = payload.get("lock_key", "")
    return _handle_generation(cache_key, lock_key, _build_overload_prompt(count), _degrader_agent)


@shared_task(
    name="nyx.tasks.background.conflict_edge_tasks.generate_contradiction_recovery",
    queue="background",
    acks_late=True,
)
def generate_contradiction_recovery(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    contradiction = payload.get("contradiction", {})
    cache_key = payload.get("cache_key", "")
    lock_key = payload.get("lock_key", "")
    return _handle_generation(cache_key, lock_key, _build_contradiction_prompt(contradiction), _healer_agent)


def dispatch_recovery_generation(case_type: str, payload: Dict[str, Any]) -> None:
    """Dispatch the correct Celery task based on case type."""

    task_map = {
        "orphaned_conflict": generate_orphan_recovery,
        "stale_conflict": generate_stale_recovery,
        "infinite_loop": generate_loop_recovery,
        "complexity_overload": generate_overload_recovery,
        "contradiction": generate_contradiction_recovery,
    }

    task = task_map.get(case_type)
    if not task:
        logger.warning("Unknown recovery case type: %s", case_type)
        return

    try:
        task.apply_async(kwargs={"payload": payload}, queue="background")
    except Exception:
        logger.exception("Failed to enqueue recovery generation task")


__all__ = [
    "dispatch_recovery_generation",
    "recovery_cache_key",
    "recovery_lock_key",
    "RECOVERY_CACHE_TTL",
    "RECOVERY_LOCK_TTL",
]
