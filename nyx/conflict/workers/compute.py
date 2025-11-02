"""Background compute helpers for conflict synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List

from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from nyx.gateway.llm_gateway import LLMOperation, LLMRequest, execute

logger = logging.getLogger(__name__)


async def llm_route_scene_subsystems(
    synthesizer: Any,
    prompt: str,
    *,
    timeout: float,
) -> List[str]:
    """Run the conflict orchestrator agent and decode subsystem names."""

    orchestrator = getattr(synthesizer, "_orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("Conflict synthesizer does not expose an orchestrator agent")

    request = LLMRequest(
        prompt=prompt,
        agent=orchestrator,
        metadata={
            "operation": LLMOperation.ORCHESTRATION.value,
            "stage": "conflict_route",
        },
    )

    result = await asyncio.wait_for(execute(request), timeout=timeout)
    payload = extract_runner_response(result)

    try:
        parsed = json.loads(payload)
    except Exception as exc:  # pragma: no cover - defensive guard for malformed responses
        logger.exception("Failed to decode conflict routing response")
        raise ValueError("Invalid routing response payload") from exc

    if not isinstance(parsed, list):
        raise ValueError("Routing response must be a list")

    return [name for name in parsed if isinstance(name, str)]


__all__ = ["llm_route_scene_subsystems"]
