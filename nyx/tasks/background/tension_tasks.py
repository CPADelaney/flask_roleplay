"""Background tasks for the tension subsystem.

These tasks perform the slow LLM work needed for tension manifestations and
escalation narration. The hot-path logic queues these tasks and serves cached
results immediately, allowing latency-sensitive flows to remain responsive.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from nyx.tasks.base import NyxTask, app

from agents import Agent
from nyx.config import WARMUP_MODEL
from infra.cache import cache_key, set_json
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from nyx.tasks.utils import with_retry, run_coro
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_TTL = 1800
DEFAULT_ESCALATION_TTL = 3600


def _idempotency_key_tension_bundle(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for tension bundle generation."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    scene_hash = payload.get("scene_hash") or (payload.get("scene_context") or {}).get("scene_hash", "")
    return f"tension_bundle:{user_id}:{conversation_id}:{scene_hash}"


def _idempotency_key_manifestation(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for manifestation generation."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    scene_hash = payload.get("scene_hash") or (payload.get("scene_context") or {}).get("scene_hash", "")
    dominant_type = payload.get("dominant_type", "")
    return f"tension_manifestation:{user_id}:{conversation_id}:{scene_hash}:{dominant_type}"


def _idempotency_key_escalation(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for escalation narration."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    breaking_tension = (payload.get("escalation_event") or {}).get("breaking_tension", "")
    return f"tension_escalation:{user_id}:{conversation_id}:{breaking_tension}"


def _fallback_manifestation(dominant_type: str, dominant_level: float) -> Dict[str, Any]:
    """Provide a deterministic manifestation when LLM generation fails."""

    readable_type = dominant_type.replace("_", " ")
    return {
        "tension_type": dominant_type,
        "level": float(round(dominant_level, 3)),
        "physical_cues": [f"Subtle {readable_type} tension lingers in the air."],
        "dialogue_modifications": ["Speech becomes measured and cautious."],
        "environmental_changes": ["The atmosphere grows noticeably taut."],
        "player_sensations": ["A prickle of unease settles in."],
    }


async def _generate_manifestation_slow(
    scene_context: Dict[str, Any],
    current_tensions: Dict[str, float],
    dominant_type: str,
    dominant_level: float,
) -> Dict[str, Any]:
    """Run the LLM to produce rich tension manifestations."""

    agent = Agent(
        name="Tension Manifestation Generator",
        instructions="""
        Generate specific, sensory manifestations of tension in scenes.
        Focus on:
        - Physical cues that players can observe
        - Dialogue modifications that reflect tension
        - Environmental changes that enhance atmosphere
        - Player sensations that create immersion

        Make manifestations subtle and realistic.
        Return valid JSON.
        """,
        model="gpt-5-nano",
    )

    prompt = (
        "Generate tension manifestations for this scene.\n\n"
        f"Dominant tension: {dominant_type} (level {dominant_level:.2f}).\n"
        f"Scene context: {json.dumps(scene_context, indent=2, default=str)}\n\n"
        "Current tensions (0-1 scale):\n"
        f"{json.dumps(current_tensions, indent=2, default=str)}\n\n"
        "Respond with JSON in the shape:\n"
        "{\n"
        "  \"physical_cues\": [..],\n"
        "  \"dialogue_modifications\": [..],\n"
        "  \"environmental_changes\": [..],\n"
        "  \"player_sensations\": [..]\n"
        "}\n"
    )

    try:
        run_result = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                model_override=WARMUP_MODEL,
            )
        )
        raw = extract_runner_response(run_result) or "{}"
        parsed = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to generate tension manifestation via LLM: %s", exc)
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    manifestation = {
        "tension_type": dominant_type,
        "level": float(round(dominant_level, 3)),
        "physical_cues": list(parsed.get("physical_cues", [])),
        "dialogue_modifications": list(parsed.get("dialogue_modifications", [])),
        "environmental_changes": list(parsed.get("environmental_changes", [])),
        "player_sensations": list(parsed.get("player_sensations", [])),
    }

    generated_lists = [
        manifestation["physical_cues"],
        manifestation["dialogue_modifications"],
        manifestation["environmental_changes"],
        manifestation["player_sensations"],
    ]

    if not any(generated_lists):
        return _fallback_manifestation(dominant_type, dominant_level)

    # Fill missing list fields with fallbacks
    for key, fallback in (
        ("physical_cues", ["Tension permeates body language."]),
        ("dialogue_modifications", ["Voices tighten and words come slowly."]),
        ("environmental_changes", ["The surroundings feel charged and brittle."]),
        ("player_sensations", ["A knot of anticipation forms in the gut."]),
    ):
        if not manifestation.get(key):
            manifestation[key] = fallback

    return manifestation


async def _generate_escalation_narrative(escalation_event: Dict[str, Any]) -> Dict[str, Any]:
    """Run the LLM to narrate an escalation event."""

    agent = Agent(
        name="Tension Escalation Narrator",
        instructions="Narrate how tensions build, peak, and release in dramatic moments.",
        model="gpt-5-nano",
    )

    prompt = (
        "A tension breaking point has been reached.\n"
        f"Event details: {json.dumps(escalation_event, indent=2, default=str)}\n\n"
        "Return JSON with keys:\n"
        "{\n"
        "  \"trigger\": str,\n"
        "  \"consequences\": [str, ...],\n"
        "  \"player_choices\": [str, ...]\n"
        "}\n"
    )

    try:
        run_result = await llm_gateway.execute(
            LLMRequest(
                prompt=prompt,
                agent=agent,
                model_override=WARMUP_MODEL,
            )
        )
        raw = extract_runner_response(run_result) or "{}"
        parsed = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to generate escalation narration via LLM: %s", exc)
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    return {
        "trigger": parsed.get(
            "trigger",
            f"The {escalation_event.get('breaking_tension', 'emotional')} tension snaps, forcing action.",
        ),
        "consequences": list(
            parsed.get(
                "consequences",
                [
                    "Relationships shift immediately.",
                    "Hidden motives surface in the aftermath.",
                ],
            )
        ),
        "player_choices": list(
            parsed.get(
                "player_choices",
                [
                    "Confront the core of the conflict.",
                    "Attempt to defuse the situation diplomatically.",
                    "Withdraw to reassess your options.",
                ],
            )
        ),
    }


def _cache_tension_bundle(
    user_id: int,
    conversation_id: int,
    scene_hash: str,
    bundle: Dict[str, Any],
    ttl: int,
) -> str:
    key = cache_key("tension_bundle", user_id, conversation_id, scene_hash)
    set_json(key, bundle, ex=ttl)
    return key


def _cache_escalation_narrative(
    user_id: int,
    conversation_id: int,
    breaking_tension: str,
    payload: Dict[str, Any],
    ttl: int,
) -> str:
    key = cache_key("tension_escalation", user_id, conversation_id, breaking_tension)
    set_json(key, payload, ex=ttl)
    return key


@app.task(base=NyxTask, name="nyx.tasks.background.tension_tasks.update_tension_bundle_cache",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_tension_bundle)
def update_tension_bundle_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility shim that delegates to manifestation generation."""

    return generate_tension_manifestations(payload)


@app.task(base=NyxTask, name="nyx.tasks.background.tension_tasks.generate_tension_manifestations",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_manifestation)
def generate_tension_manifestations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tension manifestations for a scene (slow LLM call)."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    scene_hash = payload.get("scene_hash") or (payload.get("scene_context") or {}).get("scene_hash")
    scene_context = payload.get("scene_context", {})
    current_tensions = payload.get("current_tensions") or {}
    dominant_type = (payload.get("dominant_type") or "emotional").lower()
    dominant_level = float(payload.get("dominant_level") or 0.0)
    ttl = int(payload.get("ttl", DEFAULT_BUNDLE_TTL))

    if user_id is None or conversation_id is None or not scene_hash:
        raise ValueError("user_id, conversation_id, and scene_hash are required")

    logger.info(
        "Generating tension manifestations for user=%s conversation=%s scene=%s",
        user_id,
        conversation_id,
        scene_hash,
    )

    manifestation = run_coro(
        _generate_manifestation_slow(scene_context, current_tensions, dominant_type, dominant_level)
    )

    bundle = {
        "manifestation": manifestation,
        "current_tensions": current_tensions,
        "dominant_type": dominant_type,
        "dominant_level": float(round(dominant_level, 3)),
        "status": "completed",
        "generated_at": datetime.utcnow().isoformat(),
    }

    cache_key_name = _cache_tension_bundle(user_id, conversation_id, scene_hash, bundle, ttl)

    logger.info(
        "Cached tension bundle for user=%s conversation=%s scene=%s at %s",
        user_id,
        conversation_id,
        scene_hash,
        cache_key_name,
    )

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": cache_key_name,
        "dominant_type": dominant_type,
        "manifestation_level": manifestation.get("level", dominant_level),
    }


@app.task(base=NyxTask, name="nyx.tasks.background.tension_tasks.narrate_escalation",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_escalation)
def narrate_escalation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate escalation narration (slow LLM call)."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    escalation_event = payload.get("escalation_event") or {}
    breaking_tension = escalation_event.get("breaking_tension") or "emotional"
    ttl = int(payload.get("ttl", DEFAULT_ESCALATION_TTL))

    if user_id is None or conversation_id is None:
        raise ValueError("user_id and conversation_id are required")

    logger.info(
        "Generating escalation narration for user=%s conversation=%s tension=%s",
        user_id,
        conversation_id,
        breaking_tension,
    )

    narrative = run_coro(_generate_escalation_narrative(escalation_event))

    payload_to_cache = {
        "trigger": narrative.get("trigger"),
        "consequences": narrative.get("consequences", []),
        "player_choices": narrative.get("player_choices", []),
        "event": escalation_event,
        "generated_at": datetime.utcnow().isoformat(),
    }

    cache_key_name = _cache_escalation_narrative(
        user_id,
        conversation_id,
        breaking_tension,
        payload_to_cache,
        ttl,
    )

    logger.info(
        "Cached escalation narration for user=%s conversation=%s tension=%s at %s",
        user_id,
        conversation_id,
        breaking_tension,
        cache_key_name,
    )

    return {
        "status": "narrated",
        "cache_key": cache_key_name,
        "breaking_tension": breaking_tension,
    }


__all__ = [
    "update_tension_bundle_cache",
    "generate_tension_manifestations",
    "narrate_escalation",
]
