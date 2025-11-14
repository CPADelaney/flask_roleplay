"""Hot-path helpers for tension system (no blocking LLM calls).

This module provides fast, cache-first functions for tension:
- Retrieve cached tension bundles from Redis
- Queue tension manifestation generation to background
- Fast tension level calculations (rule-based)

The slow LLM calls have been moved to nyx/tasks/background/tension_tasks.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from infra.cache import cache_key, get_json
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


def _compute_scene_hash(scene_context: Dict[str, Any]) -> str:
    """Compute stable hash for scene context."""
    stable_keys = ["scene_id", "location", "npcs_present", "conflict_ids"]
    stable_context = {k: scene_context.get(k) for k in stable_keys if k in scene_context}
    serialized = json.dumps(stable_context, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compute_scene_hash(scene_context: Dict[str, Any]) -> str:
    """Public helper for computing a scene hash."""
    return _compute_scene_hash(scene_context)


def get_tension_bundle(
    user_id: int,
    conversation_id: int,
    scene_hash: str,
) -> Dict[str, Any]:
    """Get cached tension bundle for a scene (hot path).

    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        scene_hash: Scene hash identifier

    Returns:
        Tension bundle dict with level, manifestations, atmosphere
    """
    key = cache_key("tension_bundle", user_id, conversation_id, scene_hash)
    cached = get_json(key)

    if cached:
        logger.debug("Cache hit for tension bundle: %s/%s/%s", user_id, conversation_id, scene_hash)
        return cached

    logger.debug(
        "Cache miss for tension bundle: user=%s conversation=%s scene=%s", user_id, conversation_id, scene_hash
    )

    # Return fallback while background generation runs
    return {
        "tension_level": "baseline",
        "tension_score": 0.3,
        "manifestations": [],
        "atmosphere": "calm",
        "status": "generating",
    }


def calculate_tension_score(
    conflict_intensity: float,
    num_active_conflicts: int,
    recent_escalations: int
) -> float:
    """Fast rule-based tension score calculation (no LLM).

    Args:
        conflict_intensity: Average intensity of active conflicts (0.0-1.0)
        num_active_conflicts: Number of active conflicts
        recent_escalations: Number of recent escalation events

    Returns:
        Tension score (0.0-1.0)
    """
    # Base score from conflict intensity
    base_score = conflict_intensity * 0.6

    # Add contribution from number of conflicts (capped)
    conflict_factor = min(num_active_conflicts * 0.1, 0.3)

    # Add contribution from recent escalations (capped)
    escalation_factor = min(recent_escalations * 0.05, 0.2)

    total_score = min(1.0, base_score + conflict_factor + escalation_factor)

    logger.debug(
        f"Calculated tension score: {total_score:.2f} "
        f"(intensity={conflict_intensity:.2f}, "
        f"conflicts={num_active_conflicts}, "
        f"escalations={recent_escalations})"
    )

    return total_score


def tension_level_from_score(score: float) -> str:
    """Convert tension score to categorical level (fast).

    Args:
        score: Tension score (0.0-1.0)

    Returns:
        Tension level: baseline, simmering, rising, high, critical
    """
    if score < 0.2:
        return "baseline"
    elif score < 0.4:
        return "simmering"
    elif score < 0.6:
        return "rising"
    elif score < 0.8:
        return "high"
    else:
        return "critical"


def queue_manifestation_generation(
    user_id: int,
    conversation_id: int,
    scene_context: Dict[str, Any],
    current_tensions: Dict[str, float],
    dominant_type: str,
    dominant_level: float,
) -> str:
    """Queue background task to generate tension manifestations.

    Callers should route through :meth:`logic.conflict_system.tension.TensionSubsystem.
    _trigger_manifestation_generation` so eager warm-up toggles and orchestration
    policies remain centralized.

    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        scene_context: Scene context
        current_tensions: Mapping of tension type to level
        dominant_type: Dominant tension type (string value)
        dominant_level: Dominant tension level
    """
    try:
        from nyx.tasks.background.tension_tasks import generate_tension_manifestations

        scene_hash = compute_scene_hash(scene_context)
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "scene_hash": scene_hash,
            "scene_context": scene_context,
            "current_tensions": current_tensions,
            "dominant_type": dominant_type,
            "dominant_level": dominant_level,
            "timestamp": datetime.utcnow().isoformat(),
        }

        generate_tension_manifestations.delay(payload)
        logger.debug(
            "Queued manifestation generation: user=%s conversation=%s scene=%s",
            user_id,
            conversation_id,
            scene_hash,
        )
    except Exception as e:
        logger.warning(f"Failed to queue manifestation generation: {e}")

    return scene_hash


def queue_escalation_narration(
    user_id: int,
    conversation_id: int,
    escalation_event: Dict[str, Any],
) -> None:
    """Queue background task to narrate escalation.

    Args:
        user_id: User identifier
        conversation_id: Conversation identifier
        escalation_event: Escalation event data
    """
    try:
        from nyx.tasks.background.tension_tasks import narrate_escalation

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "escalation_event": escalation_event,
            "timestamp": datetime.utcnow().isoformat(),
        }

        narrate_escalation.delay(payload)
        logger.debug(
            "Queued escalation narration for user=%s conversation=%s",
            user_id,
            conversation_id,
        )
    except Exception as e:
        logger.warning(f"Failed to queue escalation narration: {e}")


async def get_cached_manifestations(scene_hash: str) -> List[str]:
    """Get cached tension manifestations for scene (hot path).

    Args:
        scene_hash: Scene hash

    Returns:
        List of manifestation strings
    """
    key = cache_key("manifestations", scene_hash)
    cached = get_json(key, default=[])

    if isinstance(cached, list):
        logger.debug(f"Cache hit for manifestations: {scene_hash}")
        return cached

    return []


async def get_tension_metrics(
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """Get tension metrics for user session (hot path - DB only).

    Args:
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        Metrics dict with average tension, recent escalations
    """
    try:
        async with get_db_connection_context() as conn:
            # Get average conflict intensity
            intensity_row = await conn.fetchrow(
                """
                SELECT AVG(intensity) as avg_intensity
                FROM conflicts
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND status = 'active'
                """,
                user_id,
                conversation_id
            )

            avg_intensity = float(intensity_row["avg_intensity"] or 0.3) if intensity_row else 0.3

            # Get active conflict count
            count_row = await conn.fetchrow(
                """
                SELECT COUNT(*) as num_conflicts
                FROM conflicts
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND status = 'active'
                """,
                user_id,
                conversation_id
            )

            num_conflicts = count_row["num_conflicts"] if count_row else 0

            # Calculate tension score
            tension_score = calculate_tension_score(
                conflict_intensity=avg_intensity,
                num_active_conflicts=num_conflicts,
                recent_escalations=0  # Could query tension_events table if it exists
            )

            return {
                "tension_score": tension_score,
                "tension_level": tension_level_from_score(tension_score),
                "avg_intensity": avg_intensity,
                "active_conflicts": num_conflicts,
            }
    except Exception as e:
        logger.error(f"Failed to get tension metrics: {e}")
        return {
            "tension_score": 0.3,
            "tension_level": "baseline",
            "avg_intensity": 0.3,
            "active_conflicts": 0,
        }


def should_trigger_escalation(
    tension_score: float,
    time_since_last_escalation: float
) -> bool:
    """Fast rule-based check if escalation should be triggered (no LLM).

    Args:
        tension_score: Current tension score (0.0-1.0)
        time_since_last_escalation: Time in hours since last escalation

    Returns:
        True if escalation should be triggered
    """
    # High tension + enough time passed
    if tension_score > 0.7 and time_since_last_escalation > 2.0:
        return True

    # Critical tension can trigger immediately
    if tension_score > 0.9:
        return True

    return False


def get_atmosphere_description(tension_level: str) -> str:
    """Get fast atmosphere description based on tension level (no LLM).

    Args:
        tension_level: Tension level string

    Returns:
        Atmosphere description
    """
    atmospheres = {
        "baseline": "The atmosphere is calm and routine.",
        "simmering": "There's a subtle undercurrent of unease.",
        "rising": "Tension hangs in the air like a coming storm.",
        "high": "The atmosphere crackles with barely contained conflict.",
        "critical": "The tension is suffocating, ready to explode at any moment."
    }

    return atmospheres.get(tension_level, "The atmosphere is neutral.")
