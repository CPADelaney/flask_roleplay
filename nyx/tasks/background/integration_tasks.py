"""Background tasks for conflict integration helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from celery import shared_task

from agents import Agent, Runner
from logic.conflict_system.integration import IntegrationMode
from logic.conflict_system.mode_recommendation import (
    MODE_OPTIMIZER_INSTRUCTIONS,
    MODE_OPTIMIZER_MODEL,
    MODE_OPTIMIZER_NAME,
    build_mode_recommendation_prompt,
    normalize_mode_context,
    store_mode_recommendation,
)
from nyx.tasks.utils import run_coro

logger = logging.getLogger(__name__)


def _extract_mode_name(response: Any) -> Optional[str]:
    """Attempt to extract a raw mode name string from a Runner response."""

    if response is None:
        return None

    candidates = []
    for attr in ("output", "final_output", "output_text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    if not candidates and hasattr(response, "output_json"):
        try:
            candidates.append(json.dumps(response.output_json))
        except Exception:  # pragma: no cover - defensive
            pass
    if not candidates:
        try:
            candidates.append(str(response).strip())
        except Exception:  # pragma: no cover - defensive
            return None
    raw = candidates[0].splitlines()[0].strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    return raw or None


@shared_task(name="nyx.tasks.background.integration_tasks.recommend_mode", bind=True)
def recommend_mode(
    self,
    user_id: int,
    conversation_id: int,
    context_signature: str,
    context: Dict[str, Any],
    current_mode: str,
    current_quality: float,
    heuristic_mode: Optional[str] = None,
) -> Optional[str]:
    """Run the LLM-based mode recommendation and persist the result."""

    context = normalize_mode_context(context)
    agent = Agent(
        name=MODE_OPTIMIZER_NAME,
        instructions=MODE_OPTIMIZER_INSTRUCTIONS,
        model=MODE_OPTIMIZER_MODEL,
    )
    prompt = build_mode_recommendation_prompt(current_mode, current_quality, context)

    try:
        response = run_coro(Runner.run(agent, prompt))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Mode recommendation run failed", exc_info=exc)
        raise

    raw_mode = _extract_mode_name(response)
    if not raw_mode:
        logger.warning("Mode recommendation returned empty result for signature %s", context_signature)
        return None

    try:
        recommended = IntegrationMode[raw_mode.upper()]
    except KeyError:
        logger.warning("Invalid mode '%s' from recommendation task", raw_mode)
        return None

    metadata = {
        "raw_output": raw_mode,
        "heuristic_mode": heuristic_mode,
        "current_mode": current_mode,
        "current_quality": current_quality,
        "context": normalize_mode_context(context),
    }

    run_coro(
        store_mode_recommendation(
            int(user_id),
            int(conversation_id),
            context_signature,
            recommended.value,
            source="task",
            confidence=None,
            metadata=metadata,
        )
    )

    logger.info(
        "Stored mode recommendation signature=%s mode=%s (heuristic=%s)",
        context_signature,
        recommended.value,
        heuristic_mode,
    )

    return recommended.value


__all__ = ["recommend_mode"]
