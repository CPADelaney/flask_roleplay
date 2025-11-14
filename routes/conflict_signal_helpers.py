"""Utility helpers for routing conflict signals from request handlers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

    signal = ConflictSignal(
        type=ConflictSignalType.PLAYER_ACTION,
        user_id=int(user_id),
        conversation_id=int(conversation_id),
        payload={
            "activity_type": activity_type,
            "user_input": user_input,
            "location": context.get("location"),
            "involved_npcs": involved_npcs,
        },
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
