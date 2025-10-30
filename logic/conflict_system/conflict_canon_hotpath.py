"""Hot-path helpers for conflict canon system (no blocking LLM calls).

This module provides fast, cache-first functions for canon:
- Fast lore compliance checks (rule-based)
- Queue canonization to background tasks
- Retrieve cached canon references

The slow LLM calls have been moved to nyx/tasks/background/canon_tasks.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from infra.cache import redis_client, cache_key, get_json, set_json
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


def lore_compliance_fast(
    similar_items: List[Dict[str, Any]],
    threshold: float = 0.85
) -> Dict[str, Any]:
    """Fast rule-based lore compliance check (no LLM).

    Uses vector similarity scores to detect conflicts.
    For detailed semantic analysis, use background tasks.

    Args:
        similar_items: List of similar lore items with similarity scores
        threshold: Similarity threshold for conflicts (0.0-1.0)

    Returns:
        Dict with is_compliant, conflicts (if any)
    """
    if not similar_items:
        return {"is_compliant": True, "conflicts": []}

    # Find highest similarity
    top_item = max(similar_items, key=lambda x: x.get("similarity", 0.0))
    top_similarity = top_item.get("similarity", 0.0)

    if top_similarity > threshold:
        logger.debug(
            f"Lore conflict detected: similarity={top_similarity:.3f}, "
            f"threshold={threshold:.3f}"
        )
        return {
            "is_compliant": False,
            "conflicts": [top_item.get("text", "Unknown conflict")],
            "similarity": top_similarity
        }

    return {"is_compliant": True, "conflicts": []}


def queue_canonization(
    conflict_id: int,
    resolution: Dict[str, Any]
) -> None:
    """Queue background task to canonize conflict resolution.

    Args:
        conflict_id: Conflict ID
        resolution: Resolution data
    """
    try:
        from nyx.tasks.background.canon_tasks import canonize_conflict

        payload = {
            "conflict_id": conflict_id,
            "resolution": resolution,
            "timestamp": datetime.utcnow().isoformat(),
        }

        canonize_conflict.delay(payload)
        logger.debug(f"Queued canonization for conflict {conflict_id}")
    except Exception as e:
        logger.warning(f"Failed to queue canonization: {e}")


def queue_canon_reference_generation(
    conflict_id: int,
    context: Dict[str, Any]
) -> None:
    """Queue background task to generate canon references.

    Args:
        conflict_id: Conflict ID
        context: Context for reference generation
    """
    try:
        from nyx.tasks.background.canon_tasks import generate_canon_references

        payload = {
            "conflict_id": conflict_id,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
        }

        generate_canon_references.delay(payload)
        logger.debug(f"Queued canon reference generation for conflict {conflict_id}")
    except Exception as e:
        logger.warning(f"Failed to queue canon reference generation: {e}")


async def get_cached_canon_record(conflict_id: int) -> Optional[Dict[str, Any]]:
    """Get cached canon record for conflict (hot path).

    Args:
        conflict_id: Conflict ID

    Returns:
        Canon record dict or None
    """
    key = cache_key("canon", "conflict", conflict_id)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for canon record: conflict={conflict_id}")
        return cached

    # Try DB fallback
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    canon_id,
                    conflict_id,
                    canon_text,
                    significance_score,
                    cultural_impact,
                    created_at
                FROM conflict_canon
                WHERE conflict_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                conflict_id
            )

            if row:
                canon_record = {
                    "canon_id": row["canon_id"],
                    "conflict_id": row["conflict_id"],
                    "canon_text": row["canon_text"],
                    "significance_score": float(row["significance_score"] or 0.0),
                    "cultural_impact": row["cultural_impact"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }

                # Cache for next time
                set_json(key, canon_record, ex=3600)
                return canon_record

        return None
    except Exception as e:
        logger.error(f"Failed to get canon record for conflict {conflict_id}: {e}")
        return None


async def get_cached_canon_references(conflict_id: int) -> List[str]:
    """Get cached canon references for conflict (hot path).

    Args:
        conflict_id: Conflict ID

    Returns:
        List of canon reference strings
    """
    key = cache_key("canon", "references", conflict_id)
    cached = get_json(key, default=[])

    if isinstance(cached, list) and cached:
        logger.debug(f"Cache hit for canon references: conflict={conflict_id}")
        return cached

    return []


def should_canonize(
    conflict: Any,
    resolution: Dict[str, Any]
) -> bool:
    """Fast rule-based check if conflict should be canonized (no LLM).

    Args:
        conflict: Conflict object
        resolution: Resolution data

    Returns:
        True if should be canonized
    """
    # Simple heuristics
    intensity = getattr(conflict, "intensity", 0.5)
    significance = resolution.get("significance_score", 0.0)

    # High-intensity conflicts with meaningful outcomes
    if intensity > 0.7 and significance > 0.6:
        return True

    # Player-involved critical moments
    if resolution.get("player_involved", False) and significance > 0.5:
        return True

    # Major world events
    if resolution.get("world_impact", 0.0) > 0.7:
        return True

    return False


async def check_lore_conflicts_fast(
    proposed_content: str,
    category: str = "general"
) -> Dict[str, Any]:
    """Fast check for lore conflicts using cached similar items (hot path).

    This is a simplified version that uses pre-computed similarity scores.
    For detailed semantic analysis, use the background task.

    Args:
        proposed_content: Content to check
        category: Lore category

    Returns:
        Dict with is_compliant, conflicts, needs_review
    """
    # Quick cache check for recent similar checks
    import hashlib
    content_hash = hashlib.sha256(proposed_content.encode()).hexdigest()[:16]
    key = cache_key("lore_check", category, content_hash)

    cached = get_json(key)
    if cached:
        logger.debug(f"Cache hit for lore check: {content_hash}")
        return cached

    # For hot path, we can't do full vector search without blocking
    # Return a "needs_review" status and queue background check
    try:
        from nyx.tasks.background.canon_tasks import check_lore_compliance

        check_lore_compliance.delay({
            "content": proposed_content,
            "category": category,
            "content_hash": content_hash,
        })
        logger.debug(f"Queued lore compliance check: {content_hash}")
    except Exception as e:
        logger.warning(f"Failed to queue lore compliance check: {e}")

    # Return optimistic result for hot path
    result = {
        "is_compliant": True,
        "conflicts": [],
        "needs_review": True,
        "status": "pending"
    }

    # Cache briefly to avoid duplicate checks
    set_json(key, result, ex=60)

    return result


async def get_canon_summary(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Get summary of canon records (hot path - DB only).

    Args:
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        Summary dict with counts and recent items
    """
    try:
        async with get_db_connection_context() as conn:
            # Get counts
            count_row = await conn.fetchrow(
                """
                SELECT COUNT(*) as total
                FROM conflict_canon
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id,
                conversation_id
            )

            # Get recent items
            recent_rows = await conn.fetch(
                """
                SELECT
                    canon_id,
                    conflict_id,
                    canon_text,
                    significance_score,
                    created_at
                FROM conflict_canon
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY created_at DESC
                LIMIT 5
                """,
                user_id,
                conversation_id
            )

            recent_items = []
            for row in recent_rows:
                recent_items.append({
                    "canon_id": row["canon_id"],
                    "conflict_id": row["conflict_id"],
                    "canon_text": row["canon_text"][:200] + "..." if len(row["canon_text"]) > 200 else row["canon_text"],
                    "significance_score": float(row["significance_score"] or 0.0),
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })

            return {
                "total_canon_records": count_row["total"] if count_row else 0,
                "recent_canon": recent_items,
            }
    except Exception as e:
        logger.error(f"Failed to get canon summary: {e}")
        return {"total_canon_records": 0, "recent_canon": []}
