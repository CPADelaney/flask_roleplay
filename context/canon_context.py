"""Utilities for packaging canonical context envelopes."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from context.vector_service import VectorService


def _estimate_tokens_from_string(payload: str) -> int:
    """Rough token estimate (~4 characters per token)."""
    if not payload:
        return 0
    return max(1, (len(payload) + 3) // 4)


def format_canon_context(
    entity_cards: Sequence[Dict[str, Any]],
    episodic_chunks: Sequence[Dict[str, Any]],
    token_cap: int = 1200,
    top_episodic: int | None = None,
) -> str:
    """Format canonical context ensuring we respect a soft token cap."""

    lines: List[str] = []
    lines.append("CANON SEAL")
    lines.append("Reuse exact canonical strings unless an explicit delta revises them.")
    lines.append("")
    lines.append("ENTITY CARDS:")

    running_tokens = sum(_estimate_tokens_from_string(line) for line in lines)

    for card in entity_cards:
        entity_type = card.get("entity_type", "entity")
        entity_id = card.get("entity_id", "?")
        card_payload = card.get("card", {})
        snapshot = json.dumps(card_payload, sort_keys=True, ensure_ascii=False)
        entry = f"- {entity_type}:{entity_id} {snapshot}"
        est = _estimate_tokens_from_string(entry)
        if running_tokens + est > token_cap:
            break
        lines.append(entry)
        running_tokens += est

    lines.append("")
    header = "TOP EPISODIC" if top_episodic is None else f"TOP EPISODIC (limit {top_episodic})"
    lines.append(header + ":")
    running_tokens += _estimate_tokens_from_string(lines[-1])

    included = 0
    for idx, chunk in enumerate(episodic_chunks, start=1):
        if top_episodic is not None and included >= top_episodic:
            break
        chunk_payload = chunk.get("chunk", {})
        snapshot = json.dumps(chunk_payload, sort_keys=True, ensure_ascii=False)
        entry = f"{idx}. {snapshot}"
        est = _estimate_tokens_from_string(entry)
        if running_tokens + est > token_cap:
            break
        lines.append(entry)
        running_tokens += est
        included += 1

    return "\n".join(lines).strip()


async def build_canon_context(
    vector_service: VectorService,
    query_text: str,
    token_cap: int = 1200,
    top_cards: int = 5,
    top_episodic: int = 3,
) -> str:
    """Helper that pulls entity cards and recent chunks then formats them."""

    cards = await vector_service.get_entity_cards(query_text, top_k=top_cards)
    chunks = await vector_service.fetch_recent_chunks(limit=top_episodic * 2)
    return format_canon_context(cards, chunks, token_cap=token_cap, top_episodic=top_episodic)


__all__ = [
    "build_canon_context",
    "format_canon_context",
]
