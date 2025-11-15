"""Utility helpers for routing conflict signals from request handlers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_LEVERAGE_VERB_KEYWORDS: Dict[str, List[str]] = {
    "reveal_secret": ["reveal", "expose", "confess", "spill"],
    "threaten": ["threat", "blackmail", "menace", "intimidate"],
    "hint_at_leverage": ["hint", "remind", "allude", "suggest"],
    "press_advantage": ["pressure", "force", "demand", "coerce"],
}


def _infer_player_action_metadata(
    activity_type: str,
    user_input: str,
    context: Dict[str, Any],
    involved_npcs: List[int],
) -> Dict[str, Optional[str]]:
    """Infer structured verb/actor/target hints for downstream leverage logic."""

    inferred_verb: Optional[str] = None
    normalized_input = (user_input or "").lower()

    intent_block = context.get("player_intent")
    if isinstance(intent_block, dict):
        inferred_verb = intent_block.get("verb") or intent_block.get("intent")
        if isinstance(inferred_verb, str):
            inferred_verb = inferred_verb.strip().lower() or None

    if not inferred_verb:
        for canonical, keywords in _LEVERAGE_VERB_KEYWORDS.items():
            if any(keyword in normalized_input for keyword in keywords):
                inferred_verb = canonical
                break

    if not inferred_verb and activity_type:
        normalized_activity = activity_type.lower()
        if normalized_activity in _LEVERAGE_VERB_KEYWORDS:
            inferred_verb = normalized_activity

    target_identifier: Optional[str] = None
    candidate_target: Optional[int] = None
    if isinstance(intent_block, dict):
        candidate_target = intent_block.get("target_npc_id")
        if candidate_target is None:
            candidate_target = intent_block.get("npc_id")

    target_hints = [
        context.get("target_npc_id"),
        context.get("focus_npc_id"),
        context.get("primary_npc_id"),
        candidate_target,
    ]

    for hint in target_hints:
        try:
            if hint is not None:
                candidate_target = int(hint)
                break
        except (TypeError, ValueError):  # pragma: no cover - safety guard
            continue
    else:
        candidate_target = None

    if candidate_target is None and involved_npcs:
        candidate_target = involved_npcs[0]

    if candidate_target is not None:
        target_identifier = f"npc_{candidate_target}"

    return {
        "verb": inferred_verb,
        "actor": "player",
        "target": target_identifier,
    }

from conflict.signals import ConflictSignal, ConflictSignalType


async def emit_daily_update_signal(
    synthesizer,
    user_id: int,
    conversation_id: int,
    time_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Send a daily update tick when the time system advances."""

    if not time_result.get("time_advanced", False):
        return None

    new_time = time_result.get("new_time", {}) or {}
    payload = {
        "type": "daily_update",
        "new_day": new_time.get("day"),
        "time_of_day": new_time.get("time_of_day"),
    }

    signal = ConflictSignal(
        type=ConflictSignalType.TIME_TICK,
        user_id=int(user_id),
        conversation_id=int(conversation_id),
        payload=payload,
    )

    await synthesizer.handle_signal(signal)
    return {"daily_update_triggered": True}


async def emit_player_action_signal(
    synthesizer,
    user_id: int,
    conversation_id: int,
    activity_type: str,
    user_input: str,
    context: Dict[str, Any],
    npc_responses: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Send a player action signal describing the most recent turn."""

    involved_npcs = [
        resp.get("npc_id")
        for resp in (npc_responses or [])
        if resp.get("npc_id") is not None
    ]

    metadata = _infer_player_action_metadata(
        activity_type,
        user_input,
        context or {},
        involved_npcs,
    )

    payload = {
        "activity_type": activity_type,
        "user_input": user_input,
        "location": context.get("location"),
        "involved_npcs": involved_npcs,
        "verb": metadata.get("verb"),
        "actor": metadata.get("actor"),
        "target": metadata.get("target"),
    }

    signal = ConflictSignal(
        type=ConflictSignalType.PLAYER_ACTION,
        user_id=int(user_id),
        conversation_id=int(conversation_id),
        payload=payload,
    )

    await synthesizer.handle_signal(signal)
    return {"processed": True, "responses": 0}


async def emit_end_of_day_signal(
    synthesizer,
    user_id: int,
    conversation_id: int,
    year: int,
    month: int,
    day: int,
) -> None:
    """Signal an end-of-day transition to downstream subsystems."""

    signal = ConflictSignal(
        type=ConflictSignalType.TIME_TICK,
        user_id=int(user_id),
        conversation_id=int(conversation_id),
        payload={
            "type": "end_of_day",
            "year": year,
            "month": month,
            "day": day,
        },
    )

    await synthesizer.handle_signal(signal)
