"""Background tasks for conflict integration helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from nyx.tasks.base import NyxTask, app

from agents import Agent
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
from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest, LLMResult

logger = logging.getLogger(__name__)


def _extract_mode_name(response: Any) -> Optional[str]:
    """Attempt to extract a raw mode name string from an LLM execution result."""

    if isinstance(response, LLMResult):
        payload = response.raw
    else:
        payload = response

    if payload is None:
        return None

    candidates = []
    for attr in ("output", "final_output", "output_text"):
        value = getattr(payload, attr, None)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    if not candidates and hasattr(payload, "output_json"):
        try:
            candidates.append(json.dumps(payload.output_json))
        except Exception:  # pragma: no cover - defensive
            pass
    if not candidates:
        try:
            candidates.append(str(payload).strip())
        except Exception:  # pragma: no cover - defensive
            return None
    raw = candidates[0].splitlines()[0].strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    return raw or None


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.integration_tasks.recommend_mode",
)
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
        response = run_coro(
            llm_gateway.execute(
                LLMRequest(
                    prompt=prompt,
                    agent=agent,
                )
            )
        )
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
