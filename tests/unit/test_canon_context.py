from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from context.canon_context import format_canon_context
from context.vector_service import VectorService


def _estimate_tokens(text: str) -> int:
    return sum(max(1, (len(line) + 3) // 4) for line in text.splitlines())


def test_format_canon_context_respects_token_cap():
    entity_cards = [
        {
            "entity_type": "npc",
            "entity_id": str(idx),
            "card": {"npc_name": f"NPC {idx}", "description": "abc" * 10},
        }
        for idx in range(10)
    ]
    episodic_chunks = [
        {
            "chunk": {"content": "chunk" * 12, "memory_type": "observation"},
        }
        for _ in range(5)
    ]

    formatted = format_canon_context(entity_cards, episodic_chunks, token_cap=120)
    assert _estimate_tokens(formatted) <= 120
    assert "Reuse exact canonical strings" in formatted


def test_format_canon_context_limits_top_episodic():
    entity_cards = []
    episodic_chunks = [{"chunk": {"content": f"chunk {idx}"}} for idx in range(6)]
    formatted = format_canon_context(entity_cards, episodic_chunks, token_cap=300, top_episodic=2)
    episodic_lines = [line for line in formatted.splitlines() if line and line[0].isdigit()]
    assert len(episodic_lines) == 2


def test_hybrid_ranking_prefers_weighted_scores():
    service = VectorService(user_id=1, conversation_id=1)
    now = datetime.now(timezone.utc)
    rows = [
        {
            "entity_type": "npc",
            "entity_id": "alpha",
            "card": {},
            "vector_score": 0.9,
            "text_score": 0.2,
            "updated_at": now,
        },
        {
            "entity_type": "npc",
            "entity_id": "beta",
            "card": {},
            "vector_score": 0.6,
            "text_score": 0.8,
            "updated_at": now - timedelta(days=5),
        },
    ]

    scored = service._score_entity_rows(rows)
    assert scored[0]["entity_id"] == "alpha"
    assert scored[0]["score"] > scored[1]["score"]
