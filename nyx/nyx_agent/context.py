# nyx/nyx_agent/context.py
"""
Enhanced NyxContext with optimized context assembly pipeline for speed and richness

Schema Version: 3
Production-Ready Features:
- Parallel orchestrator initialization and fetching
- Scene-scoped context bundles with smart caching
- LRU memory cache (128 scenes) + Redis distributed cache
- Per-section TTLs and delta refresh
- Token-budget aware packing with summarization fallback
- Link hints for emergent connections (used in memory, lore, conflicts)
- Strong DTOs for type safety (NPCSectionData, MemorySectionData, LoreSectionData)
- Metrics with p95 tracking and throttled logging
- Redis with exponential backoff and proper byte accounting
- Fast JSON via orjson when available
- Schema versioning for clean cache upgrades
- Payload size capping (256KB) with smart trimming
- Word-boundary NPC name matching
- Configurable token budgets with guard rails
- JSON-safe serialization for all data types

Performance Targets:
- Cache hits: <200ms response time
- Cache misses: 2-5s with parallel fetching
- Memory usage: Bounded at ~20MB (128 scenes)
- Token efficiency: 40-60% reduction via smart packing
"""

import json
import time
import asyncio
import hashlib
import logging
import random
import os
import re
import dataclasses
import contextlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
    Callable,
    Awaitable,
    Iterable,
)
from enum import Enum
from collections import defaultdict, OrderedDict
import redis.asyncio as redis  # Modern redis async client
import lore.core.canon as canon
from lore.version_registry import with_lore_version_suffix

from logic.conflict_system.conflict_synthesizer import get_synthesizer
from logic.conflict_system.background_processor import get_conflict_scheduler
from logic.conflict_system.enhanced_conflict_integration_hotpath import (
    get_cached_tension_result,
)
from infra.cache import get_redis_client

# Try to use faster JSON library
try:
    import orjson
    def json_dumps(obj): 
        return orjson.dumps(obj).decode('utf-8')
    def json_loads(data): 
        return orjson.loads(data if isinstance(data, bytes) else data.encode('utf-8'))
except ImportError:
    json_dumps = json.dumps
    json_loads = json.loads

# Set up logger FIRST
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logic.universal_updater_agent import UniversalUpdaterContext

_UNIVERSAL_UPDATER_MODULE = None


def _get_universal_updater_module():
    """Lazily import the universal updater to avoid circular imports."""

    global _UNIVERSAL_UPDATER_MODULE
    if _UNIVERSAL_UPDATER_MODULE is None:
        from logic import universal_updater_agent as _universal_updater_module

        _UNIVERSAL_UPDATER_MODULE = _universal_updater_module
    return _UNIVERSAL_UPDATER_MODULE


def _build_universal_updater_context(user_id: int, conversation_id: int):
    module = _get_universal_updater_module()
    return module.UniversalUpdaterContext(user_id, conversation_id)


def _convert_updates_for_database(updates: Dict[str, Any]):
    module = _get_universal_updater_module()
    return module.convert_updates_for_database(updates)


async def _apply_universal_updates_async(*args, **kwargs):
    module = _get_universal_updater_module()
    return await module.apply_universal_updates_async(*args, **kwargs)

# Schema version for cache invalidation
SCHEMA_VERSION = 3

CANONICAL_RULES_RESERVED = 512
CONFLICT_RESERVED = 512

from db.connection import get_db_connection_context
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.response_filter import ResponseFilter
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.integrate import get_central_governance
from lore.core.canon import ensure_canonical_context
from logic.aggregator_sdk import get_comprehensive_context

_SNAPSHOT_STORE = ConversationSnapshotStore()

_PLACEHOLDER_LOCATION_TOKENS = {
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "undefined",
    "tbd",
    "-",
}


def build_canonical_snapshot_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the minimal snapshot fields that should be persisted canonically."""

    if not isinstance(snapshot, dict):
        return {}

    payload: Dict[str, Any] = {}

    for key in (
        "scene_id",
        "location_name",
        "region_id",
        "world_version",
        "conflict_id",
        "conflict_active",
        "time_window",
        "updated_at",
        "turns_at_location",
        "conflict_intensity",
    ):
        if key not in snapshot:
            continue
        value = snapshot.get(key)
        if value is None and key not in {"world_version", "conflict_active"}:
            continue
        payload[key] = value

    participants = snapshot.get("participants")
    if isinstance(participants, (list, tuple, set)):
        payload["participants"] = [str(p) for p in participants]
    elif participants is not None:
        payload["participants"] = [str(participants)]

    return payload


def _compute_enhanced_scope_key(scene_context: Dict[str, Any]) -> str:
    location = str(scene_context.get('location') or 'unknown')
    scene_type = str(scene_context.get('scene_type') or 'unknown')
    npcs = ','.join(str(n) for n in scene_context.get('present_npcs', []))
    topics = ','.join(str(t) for t in scene_context.get('topics', []))
    return f"{location}|{scene_type}|{npcs}|{topics}"


async def persist_canonical_snapshot(
    user_id: int,
    conversation_id: int,
    snapshot_payload: Dict[str, Any],
) -> None:
    """Upsert the reduced snapshot payload into the canonical CurrentRoleplay table."""

    if not snapshot_payload:
        return

    try:
        canonical_ctx = ensure_canonical_context(
            {"user_id": user_id, "conversation_id": conversation_id}
        )
        async with get_db_connection_context() as conn:
            await canon.update_current_roleplay(
                canonical_ctx,
                conn,
                "CurrentSnapshot",
                json_dumps(snapshot_payload),
            )
    except Exception:
        logger.warning(
            "Failed to persist canonical snapshot for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
            exc_info=True,
        )


async def fetch_canonical_snapshot(
    user_id: int, conversation_id: int
) -> Optional[Dict[str, Any]]:
    """Load the reduced snapshot payload from the canonical CurrentRoleplay table."""

    try:
        canonical_ctx = ensure_canonical_context(
            {"user_id": user_id, "conversation_id": conversation_id}
        )
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSnapshot'
                """,
                canonical_ctx.user_id,
                canonical_ctx.conversation_id,
            )
        if not row:
            return None
        raw_value = row.get("value")
        if raw_value is None:
            return None
        if isinstance(raw_value, (bytes, bytearray)):
            raw_value = raw_value.decode("utf-8")
        if isinstance(raw_value, str):
            data = json_loads(raw_value)
        elif isinstance(raw_value, dict):
            data = raw_value
        else:
            return None
        return dict(data) if isinstance(data, dict) else None
    except Exception:
        logger.warning(
            "Failed to fetch canonical snapshot for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
            exc_info=True,
        )
        return None

# Import NPC orchestrator and related types
from npcs.npc_orchestrator import NPCOrchestrator, NPCSnapshot, NPCStatus

# Import Memory orchestrator and related types
from memory.memory_orchestrator import (
    MemoryOrchestrator,
    EntityType,
    get_memory_orchestrator
)

# Import Conflict synthesizer and related types
from logic.conflict_system.conflict_synthesizer import (
    ConflictSynthesizer,
    get_synthesizer as get_conflict_synthesizer,
    ConflictContext,
    SubsystemType,
    EventType,
    SystemEvent
)

if TYPE_CHECKING:
    from lore.lore_orchestrator import LoreOrchestrator, OrchestratorConfig
    from story_agent.world_director_agent import (
        CompleteWorldDirector,
        WorldDirector,
        CompleteWorldDirectorContext,
        WorldDirectorContext,
    )

from .config import Config
from .world_orchestrator import WorldOrchestrator

try:
    from story_agent.world_simulation_models import (
        CompleteWorldState, WorldState, WorldMood, TimeOfDay,
        ActivityType, PowerDynamicType, PowerExchange,
        WorldTension, RelationshipDynamics, NPCRoutine,
        CurrentTimeData, VitalsData, AddictionCravingData,
        DreamData, RevelationData, ChoiceData, ChoiceProcessingResult,
    )
    from story_agent import world_director_agent as _world_director_agent  # noqa: F401
    WORLD_SIMULATION_AVAILABLE = True
except ImportError as e:
    import traceback
    logger.error(f"ACTUAL IMPORT ERROR: {e}")
    traceback.print_exc()  # This will show the full stack trace
    logger.warning("World simulation models not available - slice-of-life features disabled")
    WORLD_SIMULATION_AVAILABLE = False

# Section names constant to avoid typos
SECTION_NAMES = ('npcs', 'memories', 'lore', 'conflicts', 'world', 'narrative')

# ===== New Data Structures for Optimized Context Assembly =====

import re
from collections import OrderedDict
from nyx.scene_keys import generate_scene_cache_key

@dataclass
class SceneScope:
    location_id: Optional[Union[int, str]] = None
    location_name: Optional[str] = None
    npc_ids: Set[int] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)
    lore_tags: Set[str] = field(default_factory=set)
    conflict_ids: Set[int] = field(default_factory=set)
    memory_anchors: Set[str] = field(default_factory=set)
    nation_ids: Set[int] = field(default_factory=set)  # optional, helps lore anchoring
    time_window: Optional[int] = 24
    link_hints: Dict[str, List[Union[int, str]]] = field(default_factory=dict)
    turns_at_location: int = 0
    conflict_intensity: float = 0.0

    def to_key(self) -> str:
        # Single canonical key path for all systems
        return with_lore_version_suffix(generate_scene_cache_key(self))

    def to_cache_key(self) -> str:
        # Orchestrator expects this name; forward to our canonical key
        return self.to_key()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ('npc_ids', 'topics', 'lore_tags', 'conflict_ids', 'memory_anchors', 'nation_ids'):
            if isinstance(d.get(k), set):
                d[k] = sorted(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SceneScope':
        for k in ('npc_ids', 'topics', 'lore_tags', 'conflict_ids', 'memory_anchors', 'nation_ids'):
            if k in d and isinstance(d[k], list):
                d[k] = set(d[k])
        return cls(**d)

    @property
    def is_first_turn_at_location(self) -> bool:
        return self.turns_at_location == 0


def compute_lore_priority(
    user_input: Optional[str],
    intents: Optional[Iterable[Dict[str, Any]]],
    scope: Optional[SceneScope],
) -> float:
    """Estimate how urgently lore should be surfaced for the next turn.

    Heuristics blend the current user request, feasibility intent metadata,
    conflict intensity, whether we are at the first turn for the location, and
    any lore tags already attached to the scope. The output is clamped to the
    [0.1, 1.0] range so downstream packing always has a sane weight to work
    with even when data is sparse.
    """

    normalized_text = (user_input or "").strip().lower()
    score = 0.4 if normalized_text else 0.25

    tokens: Set[str] = set()
    if normalized_text:
        tokens = set(re.findall(r"\b\w+\b", normalized_text))

        inquiry_phrases = (
            "tell me about",
            "what is",
            "what's",
            "who is",
            "who's",
            "where is",
            "where's",
            "why is",
            "how did",
            "how does",
        )
        if any(phrase in normalized_text for phrase in inquiry_phrases):
            score += 0.15
            # Strong hint this is a lore-focused question
            score = max(score, 0.75)

        lore_keywords = {
            "lore",
            "history",
            "background",
            "legend",
            "myth",
            "culture",
            "tradition",
            "origin",
            "story",
            "stories",
        }
        if tokens & lore_keywords:
            score += 0.2
            score = max(score, 0.8)

        if normalized_text.endswith("?"):
            score += 0.05

    categories: Set[str] = set()
    if intents:
        for entry in intents:
            if not isinstance(entry, dict):
                continue
            raw_categories = entry.get("categories")
            if isinstance(raw_categories, (list, tuple, set)):
                categories.update(
                    str(category).lower()
                    for category in raw_categories
                    if category is not None
                )

    if categories:
        lore_categories = {
            "lore",
            "world_lore",
            "knowledge",
            "information",
            "investigation",
            "exploration",
            "history",
            "discovery",
            "research",
        }
        if categories & lore_categories:
            score += 0.2
            # Intents explicitly mark lore/knowledge – strongly favor lore
            score = max(score, 0.8)

        action_categories = {
            "combat",
            "violence",
            "attack",
            "fight",
            "stealth",
        }
        if categories & action_categories:
            score -= 0.1

    scope_obj = scope if isinstance(scope, SceneScope) else None
    if scope_obj:
        intensity = scope_obj.conflict_intensity
        if isinstance(intensity, (int, float)):
            clamped_intensity = max(0.0, min(float(intensity), 1.0))
            if clamped_intensity >= 0.75:
                score -= 0.1
            elif clamped_intensity >= 0.4:
                score -= 0.05

        if scope_obj.is_first_turn_at_location:
            score += 0.2

        if scope_obj.lore_tags:
            score += min(0.15, 0.03 * len(scope_obj.lore_tags))

    # Clamp and gently bias away from the exact threshold edges so small bumps
    # don’t cause unpredictable flip-flopping around 0.75.
    score = max(0.1, min(1.0, score))
    if 0.72 <= score < 0.75:
        score = 0.72
    if 0.75 < score < 0.78:
        score = 0.78
    return score


def infer_lore_aspects(user_input: str, scope: SceneScope) -> List[str]:
    """Infer which lore aspects are most relevant for the current turn."""

    normalized_text = (user_input or "").lower()
    aspects: List[str] = []

    npc_names: List[str] = []
    lore_tags: List[str] = []
    if scope is not None:
        hints_dict = getattr(scope, "link_hints", {}) or {}
        if isinstance(hints_dict, dict):
            for values in hints_dict.values():
                if isinstance(values, list):
                    for value in values:
                        if value is None:
                            continue
                        name = str(value).strip()
                        if name:
                            npc_names.append(name.lower())

        # Normalize lore tags for cheap keyword inference
        raw_tags = getattr(scope, "lore_tags", set()) or set()
        for tag in raw_tags:
            try:
                lore_tags.append(str(tag).lower())
            except Exception:
                continue

    npc_query_phrases = ("who is", "who's", "tell me about")
    npc_backstory_needed = False
    if normalized_text:
        for phrase in npc_query_phrases:
            idx = normalized_text.find(phrase)
            if idx == -1:
                continue
            if npc_names and any(name in normalized_text for name in npc_names):
                npc_backstory_needed = True
                break
            remainder = (user_input or "")[idx + len(phrase):]
            if re.findall(r"[A-Z][\w']+", remainder):
                npc_backstory_needed = True
                break
    if npc_backstory_needed:
        aspects.append("npc_backstory")

    if (
        "history" in normalized_text
        or "origin" in normalized_text
        or "backstory" in normalized_text
        or "why is this place" in normalized_text
    ):
        aspects.append("location_history")

    # Soft "what is this place / where are we" → at least history
    if (
        "what is this place" in normalized_text
        or "what's this place" in normalized_text
        or "where are we" in normalized_text
    ) and "location_history" not in aspects:
        aspects.append("location_history")

    # Religion / faith – either from text or from lore tags
    if (
        "religion" in normalized_text
        or "faith" in normalized_text
        or "temple" in normalized_text
        or any(tag in ("religion", "faith", "church", "cult") for tag in lore_tags)
    ):
        aspects.append("religious_context")

    # Politics / power structures
    if (
        "politic" in normalized_text
        or "government" in normalized_text
        or "regime" in normalized_text
        or "laws" in normalized_text
        or "who rules" in normalized_text
        or any(tag in ("politics", "government", "regime") for tag in lore_tags)
    ):
        aspects.append("political_context")

    if not aspects:
        aspects.append("location_flavor")

    return list(dict.fromkeys(aspects))


# ===== Strong DTOs for Section Data =====

@dataclass
class NPCSectionData:
    """Strongly typed NPC section data"""
    npcs: List[Dict[str, Any]]  # List format for JSON compatibility
    canonical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {'npcs': self.npcs, 'canonical_count': self.canonical_count}
    
    def compact(self, max_npcs: int = 5) -> 'NPCSectionData':
        """Compact to essential fields"""
        compacted = []
        for npc in self.npcs[:max_npcs]:
            compact_npc = {
                'id': npc.get('id'),
                'name': npc.get('name'),
                'role': npc.get('role'),
                'canonical': npc.get('canonical', False)
            }
            if 'relationship' in npc:
                compact_npc['relationship'] = {
                    k: v for k, v in npc['relationship'].items()
                    if k in ('trust', 'respect', 'closeness')
                }
            compacted.append(compact_npc)
        return NPCSectionData(npcs=compacted, canonical_count=self.canonical_count)

@dataclass
class MemorySectionData:
    """Strongly typed memory section data"""
    relevant: List[Dict[str, Any]]
    recent: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def compact(self) -> 'MemorySectionData':
        return MemorySectionData(
            relevant=self.relevant[:3],
            recent=self.recent[:2],
            patterns=self.patterns[:1]
        )

@dataclass
class LoreSectionData:
    """Strongly typed lore section data"""
    location: Dict[str, Any]
    world: Dict[str, Any]
    canonical_rules: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compact(self) -> 'LoreSectionData':
        return LoreSectionData(
            location={'description': self.location.get('description', '')[:200]},
            world={k: v for k, v in list(self.world.items())[:3]},
            canonical_rules=list(self.canonical_rules)
        )

@dataclass
class BundleSection:
    """A section of context with metadata"""
    data: Union[Dict[str, Any], NPCSectionData, MemorySectionData, LoreSectionData]
    canonical: bool = False
    priority: int = 0
    last_changed_at: float = 0.0
    ttl: float = 30.0  # seconds
    version: Optional[str] = None  # For precise staleness checking
    
    def is_stale(self) -> bool:
        return time.time() - self.last_changed_at > self.ttl
    
    @staticmethod
    def _to_json_safe(obj: Any) -> Any:
        """Recursively convert objects to JSON-safe types"""
        if hasattr(obj, 'to_dict'):
            return BundleSection._to_json_safe(obj.to_dict())
        if dataclasses.is_dataclass(obj):
            return BundleSection._to_json_safe(dataclasses.asdict(obj))
        if isinstance(obj, Enum):
            return getattr(obj, 'value', obj.name)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        if isinstance(obj, dict):
            return {str(k): BundleSection._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [BundleSection._to_json_safe(v) for v in obj]
        return obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dictionary"""
        # Convert to JSON-safe format
        data_dict = self._to_json_safe(self.data)
        
        # Handle numeric keys by converting to string keys for JSON
        if isinstance(data_dict, dict):
            data_dict = self._stringify_numeric_keys(data_dict)
        
        return {
            'data': data_dict,
            'canonical': self.canonical,
            'priority': self.priority,
            'last_changed_at': self.last_changed_at,
            'ttl': self.ttl,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BundleSection':
        """Deserialize from dictionary"""
        # Convert string keys back to ints where appropriate
        data = cls._restore_numeric_keys(d['data'])
        d['data'] = data
        return cls(**d)
    
    @staticmethod
    def _stringify_numeric_keys(obj: Any) -> Any:
        """Recursively convert numeric dict keys to strings"""
        if isinstance(obj, dict):
            return {str(k): BundleSection._stringify_numeric_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BundleSection._stringify_numeric_keys(item) for item in obj]
        return obj
    
    @staticmethod
    def _restore_numeric_keys(obj: Any) -> Any:
        """Restore numeric keys where appropriate"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Try to convert key to int if it looks numeric
                try:
                    if k.isdigit():
                        result[int(k)] = BundleSection._restore_numeric_keys(v)
                    else:
                        result[k] = BundleSection._restore_numeric_keys(v)
                except (AttributeError, ValueError):
                    result[k] = BundleSection._restore_numeric_keys(v)
            return result
        elif isinstance(obj, list):
            return [BundleSection._restore_numeric_keys(item) for item in obj]
        return obj

@dataclass
class ContextBundle:
    """Scene-scoped context bundle with all relevant data.
    Drop-in with:
      - sections alias map (e.g., 'memory' → 'memories', 'conflict' → 'conflicts')
      - pack(..., must_include=...) support (e.g., 'canon')
      - light helper methods used by tools.py (invalidate_section, add_event, etc.)
    """
    scene_scope: SceneScope
    npcs: BundleSection
    memories: BundleSection
    lore: BundleSection
    conflicts: BundleSection
    world: BundleSection
    narrative: BundleSection
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ─────────────────────────────
    # Packing
    # ─────────────────────────────
    def pack(self, token_budget: int = 8000, must_include: Optional[List[str]] = None) -> 'PackedContext':
        """Pack bundle into token-budget-aware context.
        - canonical sections are guaranteed first
        - 'must_include' forces specific keys into canonical (e.g. 'canon')
        """
        working_budget = token_budget
        packed = PackedContext(token_budget=working_budget)

        lore_priority = 0.4
        if isinstance(self.metadata, dict):
            raw_priority = self.metadata.get('lore_priority', lore_priority)
            try:
                lore_priority = float(raw_priority)
            except (TypeError, ValueError):
                lore_priority = 0.4
        lore_priority = max(0.1, min(1.0, lore_priority))

        lore_allocation = int(working_budget * 0.6 * lore_priority)
        npc_allocation = int(working_budget * 0.3)
        memory_allocation = max(0, working_budget - lore_allocation - npc_allocation)

        canonical_reserve = min(CANONICAL_RULES_RESERVED, working_budget)
        conflict_reserve = min(CONFLICT_RESERVED, max(0, working_budget - canonical_reserve))

        def has_capacity(tokens_needed: int, *, include_conflict: bool = False) -> bool:
            reserve = 0
            if not include_conflict:
                reserve = conflict_reserve
            available = max(0, working_budget - reserve)
            return packed.tokens_used + tokens_needed <= available

        def _add_payload(name: str, section: BundleSection, payload: Any, *, canonical_mode: bool = True) -> bool:
            if section.canonical and canonical_mode:
                packed.add_canonical(name, payload)
                return True
            return packed.try_add(name, payload)

        def _handle_primary(name: str, allocation: int) -> None:
            if name in added:
                return
            section = sec_map.get(name)
            if section is None:
                return
            payload = self._as_dict(section.data)
            compacted = packed._compact_if_needed(name, payload, hard=False)
            estimated = packed._estimate_tokens(compacted)
            has_room = has_capacity(estimated)
            exceeds_allocation = allocation <= 0 or estimated > allocation
            needs_summary = exceeds_allocation or not has_room
            if not needs_summary:
                if _add_payload(name, section, compacted):
                    added.add(name)
                return
            summary = packed._summarize(name, compacted)
            if summary is not None:
                summary_tokens = packed._estimate_tokens(summary)
                if has_capacity(summary_tokens) and packed.try_add(name, summary):
                    added.add(name)
                    return
            if section.canonical or has_room:
                if _add_payload(name, section, compacted):
                    added.add(name)

        def _handle_generic(name: str, *, include_conflict: bool = False) -> None:
            if name in added:
                return
            section = sec_map.get(name)
            if section is None:
                return
            payload = self._as_dict(section.data)
            compacted = packed._compact_if_needed(name, payload, hard=False)
            estimated = packed._estimate_tokens(compacted)
            has_room = has_capacity(estimated, include_conflict=include_conflict)
            if not has_room:
                summary = packed._summarize(name, compacted)
                if summary is not None:
                    summary_tokens = packed._estimate_tokens(summary)
                    if has_capacity(summary_tokens, include_conflict=include_conflict) and packed.try_add(name, summary):
                        added.add(name)
                        return
            if section.canonical or has_room:
                if _add_payload(name, section, compacted):
                    added.add(name)
        must_include = set(must_include or [])

        # Map of section name → BundleSection
        sec_map: Dict[str, BundleSection] = {
            'npcs': self.npcs,
            'memories': self.memories,
            'lore': self.lore,
            'conflicts': self.conflicts,
            'world': self.world,
            'narrative': self.narrative,
        }

        added: Set[str] = set()

        # Special-case 'canon' so tools can force-in canonical rules/facts.
        if 'canon' in must_include:
            canon_payload: Dict[str, Any] = {}
            # Pull canonical rules from lore (DTO-safe)
            lore_data = self._as_dict(self.lore.data)
            rules = lore_data.get('canonical_rules') or []
            if rules:
                canon_payload['rules'] = rules
            # Pull explicit world canon facts if present
            world_data = self._as_dict(self.world.data)
            if 'canon' in world_data:
                canon_payload['world'] = world_data['canon']
            if canon_payload:
                packed.add_canonical('canon', canon_payload)
                added.add('canon')

        # Force-include any sections named in must_include (except 'canon' which is handled above)
        for name in (must_include - {'canon'}):
            if name in sec_map and name not in added:
                packed.add_canonical(name, sec_map[name].data)
                added.add(name)

        # Pre-handle core sections with explicit allocations
        _handle_primary('lore', lore_allocation)
        _handle_primary('npcs', npc_allocation)
        _handle_primary('memories', memory_allocation)

        # Conflicts reserve their own pool before releasing remaining capacity
        _handle_generic('conflicts', include_conflict=True)
        conflict_reserve = 0

        # Retry primaries once conflict reserve is released to use any reclaimed space
        _handle_primary('lore', lore_allocation)
        _handle_primary('npcs', npc_allocation)
        _handle_primary('memories', memory_allocation)

        # Now add remaining sections: canonical ones first, then by priority desc
        sections = [
            ('npcs', self.npcs),
            ('memories', self.memories),
            ('lore', self.lore),
            ('conflicts', self.conflicts),
            ('world', self.world),
            ('narrative', self.narrative),
        ]
        sections.sort(key=lambda x: (not x[1].canonical, -x[1].priority))

        for name, _section in sections:
            if name in added or name in {'lore', 'npcs', 'memories', 'conflicts'}:
                continue
            _handle_generic(name)

        # Canonical rules are non-negotiable: always append them after other sections
        lore_dict = self._as_dict(self.lore.data)
        rules = list(lore_dict.get('canonical_rules') or [])
        if rules:
            payload = {'canonical_rules': rules}
            rule_tokens = packed._estimate_tokens(payload)
            budget_ceiling = working_budget + canonical_reserve
            if payload['canonical_rules'] and packed.tokens_used + rule_tokens <= budget_ceiling:
                packed.canonical['canonical_rules'] = payload['canonical_rules']
                packed.tokens_used += rule_tokens

        if isinstance(self.metadata, dict):
            packed.metadata = dict(self.metadata)
        else:
            packed.metadata = {}

        return packed

    # ─────────────────────────────
    # Serialization
    # ─────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serialize bundle to JSON-safe dictionary."""
        return {
            'scene_scope': self.scene_scope.to_dict(),
            'npcs': self.npcs.to_dict(),
            'memories': self.memories.to_dict(),
            'lore': self.lore.to_dict(),
            'conflicts': self.conflicts.to_dict(),
            'world': self.world.to_dict(),
            'narrative': self.narrative.to_dict(),
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ContextBundle':
        """Deserialize bundle from dictionary."""
        return cls(
            scene_scope=SceneScope.from_dict(d['scene_scope']),
            npcs=BundleSection.from_dict(d['npcs']),
            memories=BundleSection.from_dict(d['memories']),
            lore=BundleSection.from_dict(d['lore']),
            conflicts=BundleSection.from_dict(d['conflicts']),
            world=BundleSection.from_dict(d['world']),
            narrative=BundleSection.from_dict(d['narrative']),
            metadata=d.get('metadata', {}),
            created_at=d.get('created_at', time.time())
        )

    # ─────────────────────────────
    # Compatibility / Helper API
    # ─────────────────────────────
    @property
    def sections(self) -> Dict[str, Any]:
        """Legacy-friendly view over sections with aliases and synthetic subsections."""
        # Pull raw/DTO-safe views
        npcs_dict = self._as_dict(self.npcs.data)
        mem_dict = self._memory_view_dict()
        lore_dict = self._as_dict(self.lore.data)
        conflicts_dict = self._as_dict(self.conflicts.data)
        world_dict = self._as_dict(self.world.data)
        narrative_dict = self._as_dict(self.narrative.data)

        # Relationship / emotional / user_model / activities live under metadata as soft sections
        rel = self.metadata.setdefault('_relationships', {})
        emo = self.metadata.setdefault('_emotional', {})
        usr = self.metadata.setdefault('_user_model', {})
        acts = self.metadata.setdefault('_activities', {})
        visual_recent = self.metadata.get('visual_recent', [])

        out = {
            'npcs': npcs_dict,
            'memories': mem_dict,
            'memory': mem_dict,            # alias
            'lore': lore_dict,
            'conflicts': conflicts_dict,
            'conflict': conflicts_dict,    # alias
            'world': world_dict,
            'narrative': narrative_dict,
            'location': lore_dict.get('location', {}),  # convenience
            'relationships': rel,
            'emotional': emo,
            'user_model': usr,
            'activities': acts,
            'visual': {'recent': visual_recent},
        }
        return out

    @property
    def expanded_sections(self) -> Set[str]:
        """Names of sections that were explicitly expanded by the broker."""
        return set(self.metadata.get('expanded_sections', []))

    def mark_section_expanded(self, section: str) -> None:
        xs = set(self.metadata.get('expanded_sections', []))
        xs.add(section)
        self.metadata['expanded_sections'] = sorted(xs)

    def invalidate_section(self, section: str) -> None:
        """Force a section stale so the broker refreshes it on next load."""
        norm = {'memory': 'memories', 'conflict': 'conflicts'}.get(section, section)
        try:
            target: BundleSection = getattr(self, norm)
            target.last_changed_at = 0.0  # make stale
            # remove from expanded set (so tools re-request expansion if needed)
            xs = set(self.metadata.get('expanded_sections', []))
            xs.discard(section)
            xs.discard(norm)
            self.metadata['expanded_sections'] = sorted(xs)
        except AttributeError:
            # unknown section name; ignore
            pass

    def add_event(self, event: Dict[str, Any]) -> None:
        """Append an event into the world section and record a pending change."""
        wd = self._as_dict(self.world.data)
        events = wd.setdefault('events', [])
        events.append(event)
        # reflect back if the section holds a dataclass or dict
        self._assign_back(self.world, wd)
        self._pending('world').setdefault('events_added', []).append(event)

    def update_npc_states(self, actions: List[Dict[str, Any]]) -> None:
        """Apply NPC state deltas to the npcs section."""
        nd = self._as_dict(self.npcs.data)
        items = nd.get('npcs', [])
        by_id = {item.get('id'): item for item in items if isinstance(item, dict)}
        changed = []
        for act in actions or []:
            nid = act.get('id')
            if nid in by_id:
                by_id[nid].update(act)
                changed.append(nid)
        # write back
        if changed:
            self._assign_back(self.npcs, nd)
            self._pending('npcs').setdefault('updated_ids', []).extend(changed)

    def get_pending_changes(self) -> Dict[str, Any]:
        """Return and clear accumulated pending changes."""
        changes = self.metadata.get('_pending_changes', {})
        # reset after read
        self.metadata['_pending_changes'] = {}
        return changes

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Adapter so tools.check_performance_metrics has something sensible."""
        metrics = self.metadata.get('metrics', {})
        # Provide stable keys with defaults
        return {
            'response_time': metrics.get('response_time', 0),
            'tokens': metrics.get('tokens', 0),
            'cache_hits': metrics.get('cache_hits', 0),
            'cache_misses': metrics.get('cache_misses', 0),
            'parallel_fetches': metrics.get('parallel_fetches', 0),
            'bundle_size': metrics.get('bundle_size', 0),
            'sections_loaded': [k for k, v in {
                'npcs': bool(self._as_dict(self.npcs.data)),
                'memories': bool(self._as_dict(self.memories.data)),
                'lore': bool(self._as_dict(self.lore.data)),
                'conflicts': bool(self._as_dict(self.conflicts.data)),
                'world': bool(self._as_dict(self.world.data)),
                'narrative': bool(self._as_dict(self.narrative.data)),
            }.items() if v],
        }

    def get_active_themes(self) -> List[str]:
        nar = self._as_dict(self.narrative.data)
        if 'themes' in nar and isinstance(nar['themes'], list):
            return nar['themes']
        return self.metadata.get('active_themes', [])

    def get_tension_level(self) -> float:
        conf = self._as_dict(self.conflicts.data)
        tens = conf.get('tensions') or {}
        if isinstance(tens, dict) and 'overall' in tens:
            return float(tens['overall'])
        # fallback: avg of numeric values
        vals = [v for v in (tens.values() if isinstance(tens, dict) else []) if isinstance(v, (int, float))]
        return float(sum(vals) / len(vals)) if vals else 0.5

    def is_dialogue_heavy(self) -> bool:
        nar = self._as_dict(self.narrative.data)
        recent = nar.get('recent') or []
        if isinstance(recent, list) and len(recent) >= 3:
            return True
        atmos = (nar.get('atmosphere') or '').lower()
        return any(k in atmos for k in ('dialogue', 'banter', 'conversation', 'talky'))

    def mark_image_generated(self) -> None:
        """Record an image generation timestamp for pacing heuristics."""
        t = int(time.time())
        seq = self.metadata.get('visual_recent', [])
        if not isinstance(seq, list):
            seq = []
        seq.append(t)
        # keep last 10
        self.metadata['visual_recent'] = seq[-10:]

    def get_graph_context(self, entity_id: Union[str, int]) -> Dict[str, Any]:
        """Return graph context used by relationship updater."""
        links = self.metadata.get('graph_links', {})
        return links.get(str(entity_id), {}) if isinstance(links, dict) else {}

    def get_recent_interactions(self) -> List[Dict[str, Any]]:
        nar = self._as_dict(self.narrative.data)
        recent = nar.get('recent')
        if isinstance(recent, list):
            return recent
        return self.metadata.get('recent_interactions', [])

    def get_section_links(self, section: str) -> Dict[str, List[Union[int, str]]]:
        """Expose link hints per section when available."""
        hints = self.metadata.get('link_hints', {})
        entry = hints.get(section, {})
        return entry if isinstance(entry, dict) else {}

    # ─────────────────────────────
    # Internal utilities
    # ─────────────────────────────
    @staticmethod
    def _as_dict(data: Any) -> Dict[str, Any]:
        """Safely normalize BundleSection.data into a plain dict."""
        if hasattr(data, 'to_dict'):
            try:
                return data.to_dict()
            except Exception:
                pass
        if dataclasses.is_dataclass(data):
            try:
                return dataclasses.asdict(data)
            except Exception:
                pass
        if isinstance(data, dict):
            return data
        # Graceful fallback
        return {}

    @staticmethod
    def _assign_back(section: BundleSection, payload: Dict[str, Any]) -> None:
        """Write a dict payload back into section.data, preserving DTO if present."""
        # If section.data has a constructor shim, try to reconstruct;
        # otherwise just store the dict.
        section.data = payload  # lightweight & safe for JSON paths
        section.last_changed_at = time.time()

    def _pending(self, key: str) -> Dict[str, Any]:
        """Get (and create) a pending-changes bucket."""
        pc = self.metadata.setdefault('_pending_changes', {})
        return pc.setdefault(key, {})

    def _memory_view_dict(self) -> Dict[str, Any]:
        """Project the memory section to the shape tools expect:
           {'memories': [...], 'graph_links': {...}}.
        """
        raw = self._as_dict(self.memories.data)
        # If already in expected shape, return as-is.
        if 'memories' in raw:
            out = dict(raw)
        else:
            # Combine 'relevant' + 'recent' when using the DTO shape
            combined = []
            if isinstance(raw.get('relevant'), list):
                combined.extend(raw.get('relevant'))
            if isinstance(raw.get('recent'), list):
                combined.extend(raw.get('recent'))
            out = {'memories': combined}
        # Attach graph links from metadata if present
        if isinstance(self.metadata.get('graph_links'), dict):
            out['graph_links'] = self.metadata['graph_links']
        return out

@dataclass 
class PackedContext:
    """Token-budget-aware packed context"""
    token_budget: int
    canonical: Dict[str, Any] = field(default_factory=dict)
    optional: Dict[str, Any] = field(default_factory=dict)
    summarized: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    
    def add_canonical(self, key: str, data: Any):
        """Add canonical data with budget checks and compaction"""
        # Normalize dataclasses first
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        # Compact if needed
        compact = self._compact_if_needed(key, data, hard=True)
        tokens = self._estimate_tokens(compact)
        
        # If still too large, summarize
        if self.tokens_used + tokens > self.token_budget:
            compact = self._summarize(key, compact) or compact
            tokens = self._estimate_tokens(compact)
        
        self.canonical[key] = compact
        self.tokens_used += tokens
    
    def try_add(self, key: str, data: Any) -> bool:
        """Try to add optional data, with fallback to summarization"""
        # Normalize dataclasses and apply light compaction
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        data = self._compact_if_needed(key, data, hard=False)
        
        estimated = self._estimate_tokens(data)
        
        # Try full data first
        if self.tokens_used + estimated <= self.token_budget:
            self.optional[key] = data
            self.tokens_used += estimated
            return True
        
        # Try summarized version
        summary = self._summarize(key, data)
        if summary:
            summary_tokens = self._estimate_tokens(summary)
            if self.tokens_used + summary_tokens <= self.token_budget:
                self.summarized[key] = summary
                self.tokens_used += summary_tokens
                return True
        
        return False
    
    def _estimate_tokens(self, data: Any) -> int:
        """Improved token estimation - handle dataclasses and count only values"""
        # Handle dataclasses
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        if isinstance(data, str):
            return max(1, len(data) // 4)
        elif isinstance(data, dict):
            # Count only values, not keys (keys are structure)
            total = 0
            for key, value in data.items():
                if key == 'canonical_rules':
                    continue
                total += self._estimate_tokens(value)
            return total
        elif isinstance(data, list):
            return sum(self._estimate_tokens(item) for item in data)
        elif data is None:
            return 0
        else:
            return max(1, len(str(data)) // 4)
    
    def _compact_if_needed(self, key: str, data: Any, hard: bool = False) -> Any:
        """Compact data to essential fields"""
        if key == 'npcs':
            if isinstance(data, NPCSectionData):
                return data.compact(max_npcs=5 if hard else 8).to_dict()
            elif isinstance(data, dict) and 'npcs' in data:
                # List format
                items = data['npcs'][:5] if hard else data['npcs'][:8]
                for item in items:
                    # Remove verbose fields
                    item.pop('canonical_events', None)
                    if 'relationship' in item:
                        rel = item['relationship']
                        item['relationship'] = {
                            k: v for k, v in rel.items()
                            if k in ('trust', 'respect', 'closeness')
                        }
                return {'npcs': items}
            elif isinstance(data, dict):
                # Legacy dict format - convert to list
                items = list(data.values())[:5] if hard else list(data.values())[:8]
                return {'npcs': items}
        
        elif key == 'memories' and isinstance(data, MemorySectionData):
            return data.compact().to_dict()
        
        elif key == 'lore' and isinstance(data, LoreSectionData):
            return data.compact().to_dict()

        return data
    
    def _summarize(self, key: str, data: Any) -> Optional[Dict[str, Any]]:
        """Create a summary of data if too large"""
        if key == 'memories':
            if isinstance(data, MemorySectionData):
                return data.compact().to_dict()
            elif isinstance(data, dict):
                return {
                    'count': len(data.get('relevant', [])) + len(data.get('recent', [])),
                    'recent_summary': data.get('recent', [])[:2],
                    'patterns': data.get('patterns', [])[:1]
                }
        
        elif key == 'lore':
            if isinstance(data, LoreSectionData):
                return data.compact().to_dict()
            elif isinstance(data, dict):
                rules = data.get('canonical_rules', [])
                return {
                    'location': data.get('location', {}).get('description', '')[:100],
                    'canonical_rules': list(rules) if isinstance(rules, list) else rules
                }
        
        elif key == 'npcs':
            if isinstance(data, NPCSectionData):
                return data.compact(max_npcs=3).to_dict()
            elif isinstance(data, dict):
                if 'npcs' in data:
                    items = data['npcs'][:3]
                else:
                    items = list(data.values())[:3]
                
                summary_items = []
                for item in items:
                    summary_items.append({
                        'name': item.get('name'),
                        'role': item.get('role'),
                        'trust': item.get('relationship', {}).get('trust')
                    })
                return {'npcs': summary_items}
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Merge all sections into single dict"""
        result = {**self.canonical, **self.optional}
        if self.summarized:
            result['_summarized'] = self.summarized
        if self.metadata:
            result['_metadata'] = dict(self.metadata)
        return result

# ===== LRU Cache Implementation =====

class LRUCache(OrderedDict):
    """Simple LRU cache with capacity limit"""
    def __init__(self, capacity: int = 128):
        super().__init__()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]
    
    def set(self, key: str, value: Any):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)

# ===== Context Broker for Smart Assembly =====

class ContextBroker:
    """Manages scene-scoped context assembly with caching and parallel fetching"""
    
    def __init__(self, nyx_context: 'NyxContext'):
        self.ctx = nyx_context
        self.bundle_cache = LRUCache(capacity=128)  # LRU cache for scenes
        self.redis_client: Optional[redis.Redis] = None
        self._locks = defaultdict(asyncio.Lock)  # Per-scene locks
        self._npc_alias_cache: Optional[Dict[str, int]] = None  # name→id map
        self._alias_cache_updated: float = 0
        self._alias_cache_ttl: float = 300.0  # 5 minutes
        self._world_section_cache: Optional[BundleSection] = None
        self._world_section_cache_expires_at: float = 0.0
        
        # Redis backoff state
        self._redis_failures = 0
        self._redis_backoff_until = 0
        
        # Per-section TTLs (can be overridden)
        self.section_ttls = {
            'npcs': 30.0,
            'memories': 120.0,
            'lore': 300.0,  # 5 minutes
            'conflicts': 30.0,
            'world': 15.0,
            'narrative': 30.0
        }

        # Precomputed fallbacks per section to keep shapes stable during warmups
        self._fallback_builders: Dict[str, Callable[[], BundleSection]] = {
            'npcs': lambda: BundleSection(data=NPCSectionData(npcs=[], canonical_count=0)),
            'memories': lambda: BundleSection(
                data=MemorySectionData(relevant=[], recent=[], patterns=[])
            ),
            'lore': lambda: BundleSection(
                data=LoreSectionData(location={}, world={}, canonical_rules=[])
            ),
            'conflicts': lambda: BundleSection(data={}),
            'world': lambda: BundleSection(
                data={
                    'time': {'year': None, 'month': None, 'day': None, 'tod': None},
                    'locations': [],
                    'npcs': [],
                    'facts': {},
                },
                canonical=False,
                priority=0,
                last_changed_at=time.time(),
                version='world_empty',
            ),
            'narrative': lambda: BundleSection(data={}),
        }

        # Canonical fetch method registry + aliases
        self._section_fetchers: Dict[
            str, Callable[[SceneScope], Awaitable[BundleSection]]
        ] = {
            'npcs': self._fetch_npc_section,
            'memories': self._fetch_memory_section,
            'lore': self._fetch_lore_section,
            'conflicts': self._fetch_conflict_section,
            'world': self._fetch_world_section,
            'narrative': self._fetch_narrative_section,
        }
        self._section_aliases: Dict[str, str] = {
            'npc': 'npcs',
            'memory': 'memories',
            'conflict': 'conflicts',
        }

        # Optimized conflict components
        self.conflict_synthesizer = None
        self.conflict_scheduler = get_conflict_scheduler()
        self.conflict_processor = None
        try:
            self.conflict_processor = self.conflict_scheduler.get_processor(
                self.ctx.user_id,
                self.ctx.conversation_id
            )
        except Exception as e:
            logger.warning(f"Conflict processor unavailable: {e}")

        # Performance metrics
        self.metrics = {
            'cache_hits': defaultdict(int),
            'cache_misses': defaultdict(int),
            'fetch_times': defaultdict(list),
            'fetch_p95': defaultdict(float),
            'bytes_cached': defaultdict(int)
        }

        # Previous scene state for fast-path
        self._last_scene_key: Optional[str] = None
        self._last_packed: Optional[PackedContext] = None
        self._last_bundle: Optional[ContextBundle] = None

        # Section timing state (populated during bundle fetches)
        self._active_section_budgets: Optional[Dict[str, float]] = None
        self._active_fetch_cold_flag: Optional[bool] = None
        
        # Metrics logging throttle
        self._metrics_log_counter = 0
        self._metrics_log_interval = 5  # Log every 5th turn

        self._init_lock = asyncio.Lock()
        self._is_initialized = False

    def _build_section_fallback(self, section_name: str) -> BundleSection:
        builder = self._fallback_builders.get(section_name)
        if builder is not None:
            section = builder()
        else:
            section = BundleSection(data={})
        section.ttl = self.section_ttls.get(section_name, section.ttl)
        return section

    async def initialize(self):
        """Initialize broker resources such as Redis connections and caches."""

        async with self._init_lock:
            if self._is_initialized:
                return

            user_id = getattr(self.ctx, "user_id", "unknown")
            conversation_id = getattr(self.ctx, "conversation_id", "unknown")
            logger.info(
                "[CONTEXT_BROKER] Starting initialization for user %s, conversation %s",
                user_id,
                conversation_id,
            )

            await self._try_connect_redis()
            await self._build_npc_alias_cache()

            self._is_initialized = True

            logger.info(
                "[CONTEXT_BROKER] Initialization complete for user %s, conversation %s",
                user_id,
                conversation_id,
            )

    async def await_orchestrator(self, name: str) -> bool:
        """Delegate orchestrator readiness checks to the parent NyxContext."""

        awaiter = getattr(self.ctx, "await_orchestrator", None)
        if awaiter is None:
            logger.warning(
                "[CONTEXT_BROKER] NyxContext missing await_orchestrator; cannot await '%s'",
                name,
            )
            return False

        try:
            result = await awaiter(name)
            return bool(result)
        except Exception:
            logger.error(
                "[CONTEXT_BROKER] Await for orchestrator '%s' failed for user %s conversation %s",
                name,
                getattr(self.ctx, "user_id", "unknown"),
                getattr(self.ctx, "conversation_id", "unknown"),
                exc_info=True,
            )
            return False

    async def perform_fetch_and_cache(
        self,
        *,
        orchestrator_name: str,
        cache_attribute: str,
        expires_attribute: str,
        fetcher: Callable[[], Awaitable[BundleSection]],
        fallback_factory: Callable[[], BundleSection],
        ttl: Optional[float] = None,
    ) -> BundleSection:
        """Fetch a section, caching the result with an optional TTL."""

        now = time.time()
        cached: Optional[BundleSection] = getattr(self, cache_attribute, None)
        expires_at: float = getattr(self, expires_attribute, 0.0)
        ttl_value = ttl if ttl is not None else self.section_ttls.get(orchestrator_name, 0.0)

        if cached is not None and (ttl_value <= 0 or now < expires_at):
            return cached

        is_ready = await self.await_orchestrator(orchestrator_name)
        if not is_ready:
            return cached if cached is not None else fallback_factory()

        try:
            section = await fetcher()
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            reason = "timed out" if isinstance(exc, asyncio.TimeoutError) else "was cancelled"
            logger.info(
                "[CONTEXT_BROKER] Fetch for section '%s' %s; returning fallback",
                orchestrator_name,
                reason,
            )
            asyncio.create_task(
                self._refresh_section_in_background(
                    orchestrator_name=orchestrator_name,
                    cache_attribute=cache_attribute,
                    expires_attribute=expires_attribute,
                    fetcher=fetcher,
                    ttl_value=ttl_value,
                )
            )

            if cached is not None:
                logger.info(
                    "[CONTEXT_BROKER] Using stale %s section due to %s",
                    orchestrator_name,
                    reason,
                )
                return cached

            return fallback_factory()
        except Exception:
            logger.exception(
                "[CONTEXT_BROKER] Failed to fetch section '%s'", orchestrator_name
            )
            return cached if cached is not None else fallback_factory()

        setattr(self, cache_attribute, section)
        if ttl_value > 0:
            setattr(self, expires_attribute, time.time() + ttl_value)
        else:
            setattr(self, expires_attribute, 0.0)

        return section

    async def _refresh_section_in_background(
        self,
        *,
        orchestrator_name: str,
        cache_attribute: str,
        expires_attribute: str,
        fetcher: Callable[[], Awaitable[BundleSection]],
        ttl_value: float,
    ) -> None:
        """Refresh a cached section in the background without propagating failures."""

        try:
            section = await fetcher()
        except Exception:
            logger.warning(
                "[CONTEXT_BROKER] Background refresh for section '%s' failed", orchestrator_name,
                exc_info=True,
            )
            return

        setattr(self, cache_attribute, section)
        if ttl_value > 0:
            setattr(self, expires_attribute, time.time() + ttl_value)
        else:
            setattr(self, expires_attribute, 0.0)


    async def expand_bundle_section(self, bundle: ContextBundle, section: str) -> None:
        normalized = self._normalize_section_name(section)
        new_section = await self._fetch_section(normalized, bundle.scene_scope)
        setattr(bundle, normalized, new_section)

    async def expand_world(
        self,
        entities: Optional[List[str]] = None,
        aspects: Optional[List[str]] = None,
        depth: str = "summary",
        scene_scope: Optional[SceneScope] = None,
    ) -> Dict[str, Any]:
        """Expand world state details via the world orchestrator."""

        orchestrator = getattr(self.ctx, "world_orchestrator", None)
        if orchestrator is None:
            return {}

        if not self.ctx.is_orchestrator_ready("world"):
            return {}

        if not await self.ctx.await_orchestrator("world"):
            return {}

        base_timeout = getattr(self.ctx, "world_expand_timeout", None)
        if base_timeout is None:
            base_timeout = getattr(self.ctx, "world_fetch_timeout", 1.0)
        timeout_budget = self._get_section_timeout("world", base_timeout)

        try:
            expanded = await asyncio.wait_for(
                orchestrator.expand_state(
                    entities=entities,
                    aspects=aspects,
                    depth=depth,
                    scope=scene_scope,
                ),
                timeout=timeout_budget,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "World expand_state timed out after %.2fs", timeout_budget
            )
            return {}
        except Exception:
            logger.debug(
                "ContextBroker.expand_world: expand_state failed",
                exc_info=True,
            )
            return {}

        return expanded if isinstance(expanded, dict) else {}

    async def get_world_state(
        self,
        aspects: Optional[List[str]] = None,
        scene_scope: Optional[SceneScope] = None,
    ) -> Dict[str, Any]:
        """
        Return a structured world snapshot for tools like check_world_state.

        This is the world equivalent of the lore handle op:
        - Always uses the *fast* world bundle path when available
        - Falls back to CompleteWorldDirector.get_world_state()
        - Only returns the slices explicitly requested in `aspects`
        """
        aspects = aspects or ["time", "mood", "location", "tension"]

        world_director = getattr(self.ctx, "world_director", None)
        if not world_director:
            return {}

        # Try to grab a fast bundle; fall back to slow get_world_state
        bundle: Dict[str, Any] = {}
        ws = None

        try:
            ctx_obj = getattr(world_director, "context", None)
            get_bundle = getattr(ctx_obj, "get_world_bundle", None) if ctx_obj else None
            if callable(get_bundle):
                # Fast path: shallow bundle with summary
                raw_bundle = await get_bundle(fast=True)
                if isinstance(raw_bundle, dict):
                    bundle = raw_bundle
                    ws = bundle.get("world_state")
            if ws is None:
                # Fallback: full world state from director
                ws = await world_director.get_world_state()
        except Exception:
            logger.debug(
                "ContextBroker.get_world_state: bundle path failed; using fallback state",
                exc_info=True,
            )

        if ws is None:
            return {}

        # Normalize world_state for access
        try:
            if hasattr(ws, "model_dump"):
                ws_dict = ws.model_dump(mode="python")
            elif isinstance(ws, dict):
                ws_dict = dict(ws)
            else:
                ws_dict = {}
        except Exception:
            ws_dict = {}

        summary = bundle.get("summary", {}) if isinstance(bundle, dict) else {}
        conflict_state = bundle.get("conflict_state", {}) if isinstance(bundle, dict) else {}

        out: Dict[str, Any] = {}

        # TIME
        if "time" in aspects:
            time_obj = ws_dict.get("current_time")
            if isinstance(time_obj, dict):
                out["time"] = time_obj

        # MOOD
        if "mood" in aspects:
            mood = ws_dict.get("world_mood")
            if isinstance(mood, str):
                out["mood"] = mood
            elif hasattr(getattr(ws, "world_mood", None), "value"):
                out["mood"] = ws.world_mood.value

        # LOCATION
        if "location" in aspects:
            loc = ws_dict.get("location_data")
            if not isinstance(loc, str):
                # builder sometimes uses dict; normalize
                if isinstance(loc, dict):
                    loc = loc.get("current_location") or loc.get("name") or str(loc)
                else:
                    loc = str(loc) if loc is not None else ""
            out["location"] = loc or "unknown"

        # TENSION / WORLD_TENSION
        if "tension" in aspects or "world_tension" in aspects:
            wt = ws_dict.get("world_tension")
            if isinstance(wt, dict):
                out["world_tension"] = wt
            elif hasattr(getattr(ws, "world_tension", None), "model_dump"):
                out["world_tension"] = ws.world_tension.model_dump(mode="python")

        # ACTIVITIES
        if "activities" in aspects:
            acts = ws_dict.get("available_activities")
            if isinstance(acts, list):
                out["available_activities"] = acts

        # CONFLICT SNAPSHOT
        if "conflicts" in aspects:
            if isinstance(conflict_state, dict):
                out["conflicts"] = {
                    "active_conflicts": conflict_state.get("active_conflicts", []),
                    "metrics": conflict_state.get("metrics", {}),
                }

        # RAW / DEEP STATE (opt-in, can be big)
        if "deep_state" in aspects:
            out["world_state"] = ws_dict

        return out

    async def _try_connect_redis(self):
        """Try to connect to Redis with backoff on failure"""
        if self.redis_client:  # Already connected
            return
        if time.time() < self._redis_backoff_until:  # Still backing off
            return
        
        try:
            # Use environment variable with fallback to localhost
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=2.0
            )
            await self.redis_client.ping()
            logger.info("Redis cache connected for context broker")
            self._redis_failures = 0
            self._redis_backoff_until = 0
        except Exception as e:
            self._redis_failures += 1
            # Exponential backoff with jitter
            backoff = min(300, 2 ** self._redis_failures) + random.uniform(0, 1)
            self._redis_backoff_until = time.time() + backoff
            logger.warning(f"Redis not available (attempt {self._redis_failures}), "
                         f"retrying in {backoff:.1f}s: {e}")
            self.redis_client = None
    
    def _set_redis_backoff(self):
        """Set backoff after Redis operation failure"""
        self._redis_failures += 1
        backoff = min(300, 2 ** self._redis_failures) + random.uniform(0, 1)
        self._redis_backoff_until = time.time() + backoff

    async def invalidate_by_scope_keys(self, scope_keys: List[str], sections: Optional[List[str]] = None):
        """
        Purge cached bundles (LRU + Redis) for given scene keys.
        If NPC orchestrator is present, also purge its per-NPC scene bundles.
        """
        sections = sections or list(SECTION_NAMES)
        for sk in scope_keys:
            # memory LRU
            self.bundle_cache.pop(sk, None)
            # redis
            if self.redis_client and time.time() >= self._redis_backoff_until:
                try:
                    await self.redis_client.delete(f"nyx:bundle:{sk}")
                except Exception as e:
                    logger.warning(f"Redis delete failed for {sk}: {e}")
                    self._set_redis_backoff()
            # NPC bundles
            if self.ctx.npc_orchestrator:
                try:
                    self.ctx.npc_orchestrator.invalidate_npc_bundles_for_scene_key(sk)
                except Exception as e:
                    logger.warning(f"NPC invalidation failed for {sk}: {e}")
    
    async def _build_npc_alias_cache(self):
        """Build name→id mapping for efficient NPC lookup"""
        await self.ctx.await_orchestrator("npc")
        if not self.ctx.npc_orchestrator:
            return
        
        # Check if cache is still fresh
        if (self._npc_alias_cache and 
            time.time() - self._alias_cache_updated < self._alias_cache_ttl):
            return
        
        try:
            all_npcs = await self.ctx.npc_orchestrator.get_all_npcs()
            self._npc_alias_cache = {
                npc.get('name', '').lower(): npc['id']
                for npc in all_npcs if 'name' in npc and 'id' in npc
            }
            self._alias_cache_updated = time.time()
            logger.debug(f"Built NPC alias cache with {len(self._npc_alias_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not build NPC alias cache: {e}")
            self._npc_alias_cache = {}

    def _get_latest_snapshot(self) -> Dict[str, Any]:
        """Fetch the most recent lightweight snapshot for turn metadata."""

        user_id = getattr(self.ctx, "user_id", None)
        conversation_id = getattr(self.ctx, "conversation_id", None)
        if user_id is None or conversation_id is None:
            return {}

        try:
            snapshot = _SNAPSHOT_STORE.get(str(user_id), str(conversation_id))
        except Exception:
            logger.debug(
                "[CONTEXT_BROKER] Failed to read snapshot for scope computation",
                exc_info=True,
            )
            return {}

        return snapshot if isinstance(snapshot, dict) else {}

    @staticmethod
    def _coerce_turn_count(value: Any) -> Optional[int]:
        """Convert heterogeneous turn counters into a non-negative integer."""

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return max(int(round(value)), 0)

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = int(candidate)
            except ValueError:
                try:
                    parsed = int(float(candidate))
                except ValueError:
                    return None
            return max(parsed, 0)

        return None

    @staticmethod
    def _extract_turn_count(mapping: Any, keys: Iterable[str]) -> Optional[int]:
        if not isinstance(mapping, dict):
            return None

        for key in keys:
            if key not in mapping:
                continue
            count = ContextBroker._coerce_turn_count(mapping.get(key))
            if count is not None:
                return count

        return None

    @staticmethod
    def _lookup_turns_by_location(
        mapping: Any, location_tokens: Iterable[str]
    ) -> Optional[int]:
        if not isinstance(mapping, dict):
            return None

        for token in location_tokens:
            if not token:
                continue

            direct = ContextBroker._coerce_turn_count(mapping.get(token))
            if direct is not None:
                return direct

            if isinstance(token, str):
                lowered = token.casefold()
                for raw_key, raw_value in mapping.items():
                    if not isinstance(raw_key, str):
                        continue
                    if raw_key == token:
                        continue
                    if raw_key.casefold() == lowered:
                        count = ContextBroker._coerce_turn_count(raw_value)
                        if count is not None:
                            return count

        return None

    def _derive_turns_at_location(
        self,
        *,
        location_candidates: Iterable[Any],
        current_state: Optional[Dict[str, Any]],
        snapshot: Optional[Dict[str, Any]],
    ) -> int:
        """Determine how many turns have occurred at the active location."""

        normalized_tokens: List[str] = []
        for value in location_candidates:
            normalized = self.ctx._normalize_location_value(value)
            if normalized and normalized not in normalized_tokens:
                normalized_tokens.append(normalized)

        if not normalized_tokens:
            return 0

        state = current_state or {}
        direct_keys = (
            "turns_at_location",
            "turns_at_scene",
            "scene_turns",
            "turns_in_scene",
            "turn_count",
            "scene_turn_count",
        )

        count = self._extract_turn_count(state, direct_keys)
        if count is not None:
            return count

        location_turns = state.get("location_turns")
        count = self._lookup_turns_by_location(location_turns, normalized_tokens)
        if count is not None:
            return count

        for nested_key in (
            "scene",
            "current_scene",
            "currentRoleplay",
            "current_roleplay",
            "aggregator_data",
        ):
            nested = state.get(nested_key)
            if not isinstance(nested, dict):
                continue

            count = self._extract_turn_count(nested, direct_keys + ("turns",))
            if count is not None:
                return count

            nested_location_turns = nested.get("location_turns")
            count = self._lookup_turns_by_location(
                nested_location_turns, normalized_tokens
            )
            if count is not None:
                return count

        snap = snapshot or {}
        count = self._extract_turn_count(snap, ("turns_at_location",))
        if count is not None:
            scene_tokens: List[str] = []
            for key in ("scene_id", "location_name"):
                normalized_scene = self.ctx._normalize_location_value(snap.get(key))
                if normalized_scene:
                    scene_tokens.append(normalized_scene)

            if not scene_tokens:
                return count

            scene_matches = any(
                isinstance(candidate, str)
                and any(
                    candidate.casefold() == scene.casefold()
                    for scene in scene_tokens
                    if isinstance(scene, str)
                )
                for candidate in normalized_tokens
            )

            if scene_matches:
                return count

        snapshot_location_turns = snap.get("location_turn_counts")
        if not isinstance(snapshot_location_turns, dict):
            snapshot_location_turns = snap.get("location_turns")

        count = self._lookup_turns_by_location(
            snapshot_location_turns, normalized_tokens
        )
        if count is not None:
            return count

        turn_counts = snap.get("turn_counts")
        count = self._lookup_turns_by_location(turn_counts, normalized_tokens)
        if count is not None:
            return count

        return 0

    def _resolve_conflict_intensity(self) -> float:
        """Average conflict tension scores into a single intensity metric."""

        tensions = getattr(self.ctx, "conflict_tensions", None)
        if not isinstance(tensions, dict) or not tensions:
            return 0.0

        values: List[float] = []
        for value in tensions.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str):
                try:
                    values.append(float(value.strip()))
                except (TypeError, ValueError, AttributeError):
                    continue

        if not values:
            return 0.0

        return float(sum(values) / len(values))

    async def compute_scene_scope(self, user_input: str, current_state: Dict[str, Any]) -> SceneScope:
        """Analyze input and state to determine relevant scope (no orchestrator calls)"""
        scope = SceneScope()

        # Extract location (keep both ID and name)
        location_id = self.ctx._normalize_location_value(current_state.get('location_id'))
        if not location_id:
            location_id = self.ctx._normalize_location_value(current_state.get('location'))

        canonical_location = self.ctx._extract_canonical_location(current_state)
        scope.location_id = location_id or canonical_location or self.ctx.current_location

        location_name = self.ctx._normalize_location_value(current_state.get('location_name'))
        scope.location_name = location_name or canonical_location or self.ctx.current_location
        
        # Refresh NPC alias cache if stale
        await self._build_npc_alias_cache()
        
        # Use cached NPC alias map with word boundary matching
        if self._npc_alias_cache:
            for npc_name, npc_id in self._npc_alias_cache.items():
                # Use word boundaries and escape special regex chars
                if re.search(rf'\b{re.escape(npc_name)}\b', user_input, flags=re.IGNORECASE):
                    scope.npc_ids.add(npc_id)
        
        # Add NPCs already in scene
        scope.npc_ids.update(self.ctx.current_scene_npcs)
        
        # Extract topics from input using proper tokenization
        topic_keywords = ['combat', 'trade', 'romance', 'quest', 'explore', 
                         'craft', 'magic', 'politics', 'religion', 'history']
        tokens = set(re.findall(r'\b\w+\b', user_input.lower()))
        scope.topics.update(kw for kw in topic_keywords if kw in tokens)
        
        # Add active conflicts (limit to most relevant)
        if self.ctx.active_conflicts:
            scope.conflict_ids.update(self.ctx.active_conflicts[:5])
        
        # Expand scope with linked concepts (lightweight graph hop)
        await self._expand_scope_with_links(scope)

        snapshot = self._get_latest_snapshot()
        scope.turns_at_location = self._derive_turns_at_location(
            location_candidates=(
                scope.location_id,
                scope.location_name,
                current_state.get("location_id"),
                current_state.get("location_name"),
                self.ctx.current_location,
            ),
            current_state=current_state,
            snapshot=snapshot,
        )
        current_state["turns_at_location"] = scope.turns_at_location

        scope.conflict_intensity = self._resolve_conflict_intensity()
        current_state["conflict_intensity"] = scope.conflict_intensity

        return scope
    
    async def _expand_scope_with_links(self, scope: SceneScope):
        """Expand scope with directly linked entities for emergent connections"""
        # For each NPC, find directly related NPCs (limit 2 per source)
        if scope.npc_ids and len(scope.npc_ids) < 10:
            related_npcs = set()
            for npc_id in list(scope.npc_ids)[:3]:  # Top 3 NPCs
                # Could query relationship table or use cached links
                # For now, simple placeholder - in production, query relationships
                pass
            if related_npcs:
                scope.link_hints['related_npcs'] = list(related_npcs)[:5]
        
        # For topics, find related lore tags
        if scope.topics:
            # In production, use a topic→tag mapping or lightweight NLP
            related_tags = []
            tag_map = {
                'combat': ['warfare', 'conflict'],
                'trade': ['economy', 'merchants'],
                'romance': ['relationships', 'love'],
                'magic': ['arcane', 'mystical']
            }
            for topic in scope.topics:
                related_tags.extend(tag_map.get(topic, []))
            
            if related_tags:
                scope.lore_tags.update(related_tags[:5])
                scope.link_hints['related_tags'] = related_tags[:5]

    async def collect_universal_updates(self, bundle: ContextBundle) -> Dict[str, Any]:
        """Collect deltas from the bundle that should be persisted via the universal updater."""

        if not isinstance(bundle, ContextBundle):
            return {}

        scope = getattr(bundle, "scene_scope", None)
        if not isinstance(scope, SceneScope):
            return {}

        def _normalize_candidate(primary: Any = None, secondary: Any = None) -> Tuple[Optional[str], Optional[str]]:
            name = self.ctx._normalize_location_value(primary)
            ident = self.ctx._normalize_location_value(secondary) if secondary is not None else None

            if not name and secondary is not None:
                name = self.ctx._normalize_location_value(secondary)

            if name and self.ctx._is_placeholder_location_token(name):
                name = None
            if ident and self.ctx._is_placeholder_location_token(ident):
                ident = None

            if name and not ident:
                ident = name
            if not name and ident:
                name = ident

            return name, ident

        current_location = self.ctx._normalize_location_value(self.ctx.current_location)
        current_context: Dict[str, Any] = (
            self.ctx.current_context if isinstance(self.ctx.current_context, dict) else {}
        )
        current_location_id = self.ctx._normalize_location_value(current_context.get("location_id"))

        def _same_as_current(name: Optional[str], ident: Optional[str]) -> bool:
            if not name:
                return False
            if name != current_location:
                return False
            normalized_ident = ident or name
            if current_location_id:
                return normalized_ident == current_location_id
            return True

        candidate_name, candidate_id = _normalize_candidate(
            scope.location_name, scope.location_id
        )

        def _coerce_signal(signal: Any) -> Any:
            if signal is None:
                return None
            for attr in ("model_dump", "dict"):
                if hasattr(signal, attr):
                    try:
                        return getattr(signal, attr)()
                    except Exception:
                        pass
            if hasattr(signal, "items") and not isinstance(signal, dict):
                try:
                    return {item.key: item.value for item in signal.items}
                except Exception:
                    try:
                        return dict(signal.items())
                    except Exception:
                        pass
            return signal

        primary_keys = {
            "destination",
            "location",
            "location_name",
            "locationName",
            "locationSlug",
            "location_slug",
            "CurrentLocation",
            "current_location",
            "currentLocation",
            "CurrentScene",
            "currentScene",
            "scene",
            "scene_name",
            "sceneName",
            "venue",
        }
        nested_keys = {
            "per_intent",
            "intents",
            "intent",
            "normalized_intent",
            "resolved_locations",
            "operations",
            "updates",
            "payload",
            "data",
            "result",
            "results",
            "entries",
            "targets",
            "actions",
        }
        disqualifier_keys = {
            "categories",
            "strategy",
            "violations",
            "verb",
            "type",
            "rule",
        }

        def _extract_location_from_signal(signal: Any) -> Optional[Tuple[Any, Any]]:
            signal = _coerce_signal(signal)

            if isinstance(signal, str):
                stripped = signal.strip()
                return (stripped, None) if stripped else None

            if isinstance(signal, (int, float)):
                token = str(signal).strip()
                return (token, token) if token else None

            if isinstance(signal, dict):
                candidate_name: Optional[Any] = None
                candidate_ident: Optional[Any] = None

                name_keys = [
                    "location_name",
                    "location",
                    "destination",
                    "scene",
                    "scene_name",
                    "sceneName",
                    "venue",
                ]

                for key in name_keys:
                    if key not in signal:
                        continue
                    value = signal.get(key)
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float)):
                        candidate_name = value
                        break
                    nested = _extract_location_from_signal(value)
                    if nested:
                        return nested

                if candidate_name is None and "name" in signal:
                    if not (disqualifier_keys & set(signal.keys())):
                        candidate_name = signal.get("name")

                id_keys = [
                    "id",
                    "location_id",
                    "locationId",
                    "location_slug",
                    "locationSlug",
                    "slug",
                    "scene_id",
                    "sceneId",
                ]

                for key in id_keys:
                    if key not in signal:
                        continue
                    value = signal.get(key)
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float)):
                        candidate_ident = value
                        break
                    nested = _extract_location_from_signal(value)
                    if nested and nested[1] is not None:
                        return nested

                if candidate_name is not None or candidate_ident is not None:
                    return candidate_name, candidate_ident

                key_field = signal.get("key") or signal.get("field")
                if key_field:
                    key_norm = str(key_field).strip().lower()
                    if key_norm in {
                        "currentlocation",
                        "current_location",
                        "location",
                        "currentscene",
                        "current_scene",
                        "scene",
                    }:
                        value = signal.get("value") or signal.get("payload") or signal.get("data")
                        nested = _extract_location_from_signal(value)
                        if nested:
                            return nested

                for key in primary_keys:
                    if key in signal:
                        nested = _extract_location_from_signal(signal.get(key))
                        if nested:
                            return nested

                for key in nested_keys:
                    if key in signal:
                        nested = _extract_location_from_signal(signal.get(key))
                        if nested:
                            return nested

                return None

            if isinstance(signal, (list, tuple, set)):
                for item in signal:
                    nested = _extract_location_from_signal(item)
                    if nested:
                        return nested

            return None

        def _extract_from_text(text: Optional[str]) -> Optional[str]:
            if not isinstance(text, str):
                return None
            stripped = text.strip()
            if not stripped:
                return None

            movement_pattern = re.compile(
                r"(?:go|head|travel|walk|run|move|proceed|enter|drive|sail|march|ride|stroll|step|rush|dash|journey|make\s+(?:my|our|your)\s+way)\s+"
                r"(?:to|into|toward|towards)?\s*(?:the\s+|a\s+|an\s+)?(?P<loc>[A-Za-z][A-Za-z0-9'\- ]{2,})",
                flags=re.IGNORECASE,
            )

            match = movement_pattern.search(stripped)
            if not match:
                return None

            raw_location = match.group("loc").strip()
            raw_location = re.sub(r"[.!?,;:]+$", "", raw_location)
            normalized = self.ctx._normalize_location_value(raw_location)
            if normalized and not self.ctx._is_placeholder_location_token(normalized):
                return normalized
            return None

        def _extract_from_signals() -> Tuple[Optional[str], Optional[str]]:
            signals: List[Any] = []

            metadata = getattr(bundle, "metadata", {})
            if isinstance(metadata, dict) and metadata:
                signals.append(metadata)
                for key in (
                    "feasibility",
                    "feasibility_payload",
                    "fast_feasibility",
                    "intents",
                    "latest_intents",
                    "extras",
                ):
                    value = metadata.get(key)
                    if value is not None:
                        signals.append(value)

            if current_context:
                for key in (
                    "feasibility",
                    "fast_feasibility",
                    "feasibility_meta",
                    "feasibility_payload",
                    "intents",
                    "latest_intents",
                    "action_intents",
                    "intents_payload",
                ):
                    value = current_context.get(key)
                    if value is not None:
                        signals.append(value)

                processing_meta = current_context.get("processing_metadata")
                if isinstance(processing_meta, dict):
                    feas_meta = processing_meta.get("feasibility")
                    if feas_meta is not None:
                        signals.append(feas_meta)

                aggregator_data = current_context.get("aggregator_data")
                if isinstance(aggregator_data, dict):
                    for agg_key, agg_value in aggregator_data.items():
                        lowered = str(agg_key).lower()
                        if any(token in lowered for token in ("feas", "intent", "location")):
                            signals.append(agg_value)

            for attr in (
                "last_feasibility",
                "last_feasibility_result",
                "fast_feasibility",
                "latest_intents",
                "last_intents_payload",
                "last_intents",
            ):
                value = getattr(self.ctx, attr, None)
                if value is not None:
                    signals.append(value)

            for raw_signal in signals:
                extracted = _extract_location_from_signal(raw_signal)
                if not extracted:
                    continue
                name, ident = extracted
                normalized_name, normalized_ident = _normalize_candidate(name, ident)
                if not normalized_name:
                    continue
                if _same_as_current(normalized_name, normalized_ident):
                    continue
                return normalized_name, normalized_ident

            return None, None

        if not candidate_name or _same_as_current(candidate_name, candidate_id):
            signal_name, signal_id = _extract_from_signals()
            if signal_name:
                candidate_name, candidate_id = signal_name, signal_id

        if not candidate_name or _same_as_current(candidate_name, candidate_id):
            text_sources = [
                getattr(self.ctx, "last_user_input", None),
                current_context.get("last_user_input"),
            ]
            for text in text_sources:
                inferred = _extract_from_text(text)
                if not inferred:
                    continue
                normalized_name, normalized_ident = _normalize_candidate(inferred)
                if not normalized_name:
                    continue
                if _same_as_current(normalized_name, normalized_ident):
                    continue
                candidate_name, candidate_id = normalized_name, normalized_ident
                break

        if not candidate_name:
            return {}

        normalized_id = candidate_id or candidate_name

        roleplay_updates: Dict[str, Any] = {"CurrentLocation": candidate_name}
        if normalized_id and normalized_id != candidate_name:
            roleplay_updates["CurrentLocationId"] = normalized_id

        scene_payload: Dict[str, Any] = {"name": candidate_name}
        if normalized_id:
            scene_payload["id"] = normalized_id

        roleplay_updates["CurrentScene"] = json.dumps({"location": scene_payload})

        if _same_as_current(candidate_name, normalized_id):
            return {}

        return {"roleplay_updates": roleplay_updates}

    async def apply_updates_async(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Persist collected updates using the universal updater pipeline."""

        if not isinstance(updates, dict) or not updates:
            return {}

        try:
            db_ready_updates = _convert_updates_for_database(dict(updates))
        except Exception:
            logger.warning("Failed to normalize universal updates payload", exc_info=True)
            return {}

        roleplay_updates = db_ready_updates.get("roleplay_updates")
        if isinstance(roleplay_updates, list):
            coerced: Dict[str, Any] = {}
            for item in roleplay_updates:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                if not key:
                    continue
                coerced[key] = item.get("value")
            roleplay_updates = coerced
            db_ready_updates["roleplay_updates"] = coerced
        elif not isinstance(roleplay_updates, dict):
            roleplay_updates = {}

        if not roleplay_updates:
            return {}

        try:
            updater_context = _build_universal_updater_context(
                self.ctx.user_id, self.ctx.conversation_id
            )
            await updater_context.initialize()
        except Exception:
            logger.warning("Failed to initialize UniversalUpdaterContext", exc_info=True)
            return {}

        try:
            async with get_db_connection_context() as conn:
                result = await _apply_universal_updates_async(
                    updater_context,
                    self.ctx.user_id,
                    self.ctx.conversation_id,
                    db_ready_updates,
                    conn,
                )
        except Exception:
            logger.error("Failed to apply universal updates", exc_info=True)
            return {}

        new_location = self.ctx._normalize_location_value(roleplay_updates.get("CurrentLocation"))
        scene_location_id: Optional[str] = None
        raw_scene = roleplay_updates.get("CurrentScene")
        if isinstance(raw_scene, str):
            try:
                parsed_scene = json.loads(raw_scene)
            except json.JSONDecodeError:
                parsed_scene = None
            if isinstance(parsed_scene, dict):
                location_payload = parsed_scene.get("location")
                if isinstance(location_payload, dict):
                    scene_location_id = self.ctx._normalize_location_value(
                        location_payload.get("id") or location_payload.get("location_id")
                    )
                    if not new_location:
                        new_location = self.ctx._normalize_location_value(location_payload.get("name"))

        if new_location:
            self.ctx.current_location = new_location
            self.ctx.current_context["location_name"] = new_location
            self.ctx.current_context["current_location"] = new_location
            self.ctx.current_context["location"] = new_location
            if scene_location_id:
                self.ctx.current_context["location_id"] = scene_location_id
            else:
                self.ctx.current_context["location_id"] = new_location

            for key in ("currentRoleplay", "current_roleplay"):
                existing = self.ctx.current_context.get(key)
                payload = {
                    "id": scene_location_id or new_location,
                    "name": new_location,
                }
                if isinstance(existing, dict):
                    existing["CurrentLocation"] = payload
                else:
                    self.ctx.current_context[key] = {"CurrentLocation": payload}

            try:
                user_key = str(self.ctx.user_id)
                conversation_key = str(self.ctx.conversation_id)
                snapshot = _SNAPSHOT_STORE.get(user_key, conversation_key)
                snapshot["location_name"] = new_location
                snapshot["scene_id"] = scene_location_id or new_location
                _SNAPSHOT_STORE.put(user_key, conversation_key, snapshot)
            except Exception:
                logger.debug("Failed to update snapshot store after universal updates", exc_info=True)

        return result if isinstance(result, dict) else {}

    async def load_or_fetch_bundle(self, scene_scope: SceneScope) -> ContextBundle:
        """Load bundle from cache or fetch if needed, with per-section refresh"""
        scene_key = scene_scope.to_key()

        # Opportunistic Redis reconnect if needed
        if self.redis_client is None and time.time() >= self._redis_backoff_until:
            await self._try_connect_redis()
        
        # Fast path: if same scene as last time, refresh stale sections
        if scene_key == self._last_scene_key and self._last_bundle:
            refreshed = await self._refresh_stale_sections(self._last_bundle, scene_scope)
            if refreshed:
                await self._save_cache(scene_key, refreshed)
                self._last_bundle = refreshed  # Update last bundle pointer
                self.metrics['cache_hits']['bundle'] += 1
                return refreshed
        
        async with self._locks[scene_key]:
            # Check memory cache
            bundle = self.bundle_cache.get(scene_key)
            
            # Try Redis if no memory cache
            if not bundle and self.redis_client and time.time() >= self._redis_backoff_until:
                try:
                    cached_json = await self.redis_client.get(f"nyx:bundle:{scene_key}")
                    if cached_json:
                        bundle_dict = json_loads(cached_json)
                        
                        # Check schema version with robust parsing
                        raw = bundle_dict.get('metadata', {}).get('schema_version', 0)
                        try:
                            cached_version = int(raw)
                        except (TypeError, ValueError):
                            cached_version = 0
                        
                        if cached_version >= SCHEMA_VERSION:
                            bundle = ContextBundle.from_dict(bundle_dict)
                            self.bundle_cache.set(scene_key, bundle)
                            self.metrics['bytes_cached']['redis'] += len(cached_json)
                            logger.debug(f"Redis cache hit for scene {scene_key[:8]}")
                        else:
                            logger.debug(f"Skipping old schema version {cached_version} in Redis cache")
                            # Optionally delete old entry
                            await self.redis_client.delete(f"nyx:bundle:{scene_key}")
                except Exception as e:
                    logger.warning(f"Redis cache read failed: {e}")
                    self._set_redis_backoff()
            
            # If we have a bundle, refresh only stale sections
            if bundle:
                refreshed = await self._refresh_stale_sections(bundle, scene_scope)
                if refreshed:
                    # CRITICAL FIX: Save refreshed bundle
                    await self._save_cache(scene_key, refreshed)
                    self.metrics['cache_hits']['bundle'] += 1
                    return refreshed
            
            # No bundle or all sections stale - build shallow bundle fast, then trigger full build
            self.metrics['cache_misses']['bundle'] += 1

            cold_budget_ms = int(os.getenv("NYX_COLD_BUNDLE_BUDGET_MS", "2000"))
            shallow_bundle = await self._build_shallow_bundle(scene_scope, budget_ms=cold_budget_ms)

            # Cache shallow bundle immediately so callers can proceed
            await self._save_cache(scene_key, shallow_bundle)

            # Fire-and-forget full build so subsequent calls benefit from fresh data
            background_task = asyncio.create_task(
                self._background_full_fetch(scene_key, scene_scope)
            )

            tracker = getattr(self.ctx, "_track_background_task", None)
            if callable(tracker):
                tracker(
                    background_task,
                    task_name="context_full_bundle_refresh",
                    task_details={"scene_key": scene_key},
                )

            self._last_scene_key = scene_key
            self._last_bundle = shallow_bundle

            return shallow_bundle
    
    async def _refresh_stale_sections(self, bundle: ContextBundle, scope: SceneScope) -> Optional[ContextBundle]:
        """Refresh only stale sections of a bundle, preserving metadata and using orchestrator deltas for conflicts."""
        sections_to_refresh = []
    
        for name in SECTION_NAMES:
            section = getattr(bundle, name)
            if section.is_stale():
                sections_to_refresh.append(name)
                logger.debug(f"Section {name} is stale, will refresh")
    
        if not sections_to_refresh:
            # Everything is fresh
            return bundle
    
        # Special-case: conflicts can be refreshed via orchestrator delta (cheap incremental)
        if (
            "conflicts" in sections_to_refresh
            and self.conflict_synthesizer
            and hasattr(self.conflict_synthesizer, "get_scene_delta")
        ):
            try:
                since_ts = float(getattr(bundle.conflicts, "last_changed_at", 0.0) or 0.0)
            except Exception:
                since_ts = 0.0
    
            try:
                delta = await self.conflict_synthesizer.get_scene_delta(scope, since_ts)
            except Exception as e:
                delta = None
                logger.debug(f"Conflict delta fetch failed: {e}")
    
            if isinstance(delta, dict):
                # If delta returned actual changes, rebuild the conflicts section
                has_changes = any(
                    bool(delta.get(k))
                    for k in ("active", "conflicts", "tensions", "opportunities")
                )
                if has_changes:
                    new_data = {
                        "active": delta.get("active", delta.get("conflicts", [])) or [],
                        "tensions": delta.get("tensions", {}) or {},
                        "opportunities": delta.get("opportunities", []) or [],
                    }
                    wt = delta.get("world_tension")
                    if isinstance(wt, (int, float)):
                        new_data.setdefault("tensions", {})
                        new_data["tensions"]["overall"] = float(wt)
    
                    new_conflicts_section = BundleSection(
                        data=new_data,
                        canonical=True,
                        priority=bundle.conflicts.priority,
                        last_changed_at=float(delta.get("last_changed_at", time.time())),
                        ttl=bundle.conflicts.ttl,
                        version=f"conflict_delta_{int(time.time())}_{len(new_data['active'])}",
                    )
    
                    bdict = bundle.to_dict()
                    bdict["conflicts"] = new_conflicts_section.to_dict()
                    # Preserve metadata and link hints
                    bdict.setdefault("metadata", {}).update(bundle.metadata)
                    if scope.link_hints:
                        bdict["metadata"]["link_hints"] = scope.link_hints
    
                    bundle = ContextBundle.from_dict(bdict)
                    self.metrics["cache_hits"]["conflicts"] += 1
                    sections_to_refresh = [s for s in sections_to_refresh if s != "conflicts"]
                else:
                    # No changes since since_ts: just bump last_changed_at to mark fresh
                    bdict = bundle.to_dict()
                    conf_dict = bdict.get("conflicts", {})
                    if isinstance(conf_dict, dict):
                        conf_dict["last_changed_at"] = time.time()
                        bdict["conflicts"] = conf_dict
                        bdict.setdefault("metadata", {}).update(bundle.metadata)
                        if scope.link_hints:
                            bdict["metadata"]["link_hints"] = scope.link_hints
                        bundle = ContextBundle.from_dict(bdict)
                        self.metrics["cache_hits"]["conflicts"] += 1
                        sections_to_refresh = [s for s in sections_to_refresh if s != "conflicts"]
    
        # If everything was handled by delta, return
        if not sections_to_refresh:
            return bundle
    
        # Refresh remaining stale sections in parallel
        refresh_tasks = []
        for name in sections_to_refresh:
            refresh_tasks.append(self._fetch_section(name, scope))
    
        refreshed_sections = await asyncio.gather(*refresh_tasks, return_exceptions=True)
    
        # Update bundle with refreshed sections
        bundle_dict = bundle.to_dict()
        for name, result in zip(sections_to_refresh, refreshed_sections):
            if isinstance(result, Exception):
                logger.error(f"Failed to refresh {name}: {result}")
                # Keep old section on error
            else:
                bundle_dict[name] = result.to_dict()
                self.metrics["cache_hits"][name] += 1
    
        # Preserve metadata including link hints
        if "metadata" not in bundle_dict:
            bundle_dict["metadata"] = {}
    
        # Preserve existing metadata
        bundle_dict["metadata"].update(bundle.metadata)
    
        # Update link hints if scope has them
        if scope.link_hints:
            bundle_dict["metadata"]["link_hints"] = scope.link_hints
    
        return ContextBundle.from_dict(bundle_dict)
    
    def _normalize_section_name(self, section_name: str) -> str:
        """Return canonical section name for a given alias."""
        return self._section_aliases.get(section_name, section_name)

    def _get_section_timeout(self, section_name: str, default: float) -> float:
        """Return the active timeout budget for a section, falling back to default."""
        budgets = self._active_section_budgets
        if not budgets:
            return default
        return float(budgets.get(section_name, default))

    async def _fetch_section(self, section_name: str, scope: SceneScope) -> BundleSection:
        """Fetch a single section by name"""
        canonical_name = self._normalize_section_name(section_name)
        method = self._section_fetchers.get(canonical_name)
        if not method:
            logger.warning(f"Unknown section name: {section_name}")
            return self._build_section_fallback(canonical_name)

        start_time = time.time()
        try:
            result = await method(scope)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            reason = "timed out" if isinstance(exc, asyncio.TimeoutError) else "was cancelled"
            logger.info(
                "[CONTEXT_BROKER] Section '%s' %s before completion; returning fallback",
                canonical_name,
                reason,
            )
            return self._build_section_fallback(canonical_name)

        # Track fetch time
        fetch_time = time.time() - start_time
        self.metrics['fetch_times'][canonical_name].append(fetch_time)

        # Apply section-specific TTL
        result.ttl = self.section_ttls.get(canonical_name, 30.0)

        return result
    
    async def fetch_bundle(self, scene_scope: SceneScope) -> ContextBundle:
        """Fetch all context sections in parallel (full build)."""
        return await self.fetch_bundle_full(scene_scope)

    async def _background_full_fetch(self, scene_key: str, scene_scope: SceneScope) -> None:
        """Build and cache the full bundle without blocking the caller."""
        try:
            full_bundle = await self.fetch_bundle_full(scene_scope)
        except Exception:
            logger.debug("Background full-bundle fetch failed", exc_info=True)
            return

        try:
            await self._save_cache(scene_key, full_bundle)
        except Exception:
            logger.debug("Failed to save background bundle to cache", exc_info=True)

        self._last_scene_key = scene_key
        self._last_bundle = full_bundle

    async def fetch_bundle_full(self, scene_scope: SceneScope) -> ContextBundle:
        """Full bundle build that fetches all sections in parallel."""
        start_time = time.time()

        # Use semaphore to limit parallelism if configured
        max_parallel = self.ctx.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None

        # Determine whether we're still warming up slower orchestrators.
        is_cold = False
        ready_checker = getattr(self.ctx, "is_orchestrator_ready", None)
        if callable(ready_checker):
            readiness_samples: List[bool] = []
            for orchestrator in ("memory", "lore"):
                try:
                    readiness_samples.append(bool(ready_checker(orchestrator)))
                except Exception:
                    readiness_samples.append(False)
            if readiness_samples:
                is_cold = not all(readiness_samples)

        # Per-section budgets (seconds). Allow env overrides; bump slow paths.
        def _to_float(env: str, default: float) -> float:
            try:
                v = float(os.getenv(env, "").strip() or default)
                return max(0.25, v)
            except Exception:
                return default

        warm_budgets: Dict[str, float] = {
            'npcs':       _to_float("NYX_SECTION_TIMEOUT_NPCS",       1.5),
            'memories':   _to_float("NYX_SECTION_TIMEOUT_MEMORIES",   3.0),
            'lore':       _to_float("NYX_SECTION_TIMEOUT_LORE",       2.0),
            'conflicts':  _to_float("NYX_SECTION_TIMEOUT_CONFLICTS",  3.0),
            'world':      _to_float("NYX_SECTION_TIMEOUT_WORLD",      1.5),
            'narrative':  _to_float("NYX_SECTION_TIMEOUT_NARRATIVE",  1.0),
        }

        cold_budgets = dict(warm_budgets)
        cold_budgets['memories'] = _to_float(
            "NYX_SECTION_TIMEOUT_MEMORIES_COLD",
            max(warm_budgets.get('memories', 3.0), 5.0),
        )
        cold_budgets['lore'] = _to_float(
            "NYX_SECTION_TIMEOUT_LORE_COLD",
            max(warm_budgets.get('lore', 2.0), 5.0),
        )

        section_budgets = cold_budgets if is_cold else warm_budgets

        previous_budgets = self._active_section_budgets
        previous_cold_flag = self._active_fetch_cold_flag
        self._active_section_budgets = section_budgets
        self._active_fetch_cold_flag = is_cold

        async def fetch_with_semaphore(section_name: str, timeout: float):
            timeout = max(0.25, timeout)

            async def _timed_fetch() -> BundleSection:
                """
                Run the section fetch without letting the timeout cancel the task.
                If it doesn’t finish in time, return a fallback and let the
                task finish in the background.
                """
                task = asyncio.create_task(self._fetch_section(section_name, scene_scope))
                try:
                    # Use asyncio.wait to avoid cancelling the task on timeout.
                    done, _pending = await asyncio.wait({task}, timeout=timeout)
                    if task in done:
                        return task.result()
                    logger.info(
                        "[CONTEXT_BROKER] Section '%s' exceeded %.2fs; using fallback (will finish in background)",
                        section_name, timeout
                    )
                    # Leave the task running; it can warm caches on completion.
                    task.add_done_callback(lambda t: None)  # no-op, just avoid "never awaited" warnings
                    return self._build_section_fallback(section_name)
                except Exception:
                    logger.info(
                        "[CONTEXT_BROKER] Section '%s' failed during timed fetch; using fallback",
                        section_name,
                    )
                    return self._build_section_fallback(section_name)

            if semaphore:
                async with semaphore:
                    return await _timed_fetch()
            return await _timed_fetch()

        try:
            results = await asyncio.gather(
                *[
                    fetch_with_semaphore(name, section_budgets.get(name, 1.5))
                    for name in SECTION_NAMES
                ],
                return_exceptions=True
            )
        finally:
            self._active_section_budgets = previous_budgets
            self._active_fetch_cold_flag = previous_cold_flag

        bundle_data = {}
        for section_name, result in zip(SECTION_NAMES, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {section_name}: {result}")
                bundle_data[section_name] = BundleSection(
                    data={},
                    canonical=False,
                    priority=0,
                    last_changed_at=time.time(),
                    ttl=self.section_ttls.get(section_name, 30.0)
                )
            else:
                bundle_data[section_name] = result

        fetch_time = time.time() - start_time
        logger.debug(f"Fetched context bundle in {fetch_time:.2f}s")

        return ContextBundle(
            scene_scope=scene_scope,
            **bundle_data,
            metadata={
                'fetch_time': fetch_time,
                'schema_version': SCHEMA_VERSION,
                'link_hints': scene_scope.link_hints
            }
        )

    async def _build_shallow_bundle(self, scope: SceneScope, budget_ms: int = 2000) -> ContextBundle:
        """Build a fast, shallow bundle for cold misses within a tight budget."""

        start = time.time()

        def _remaining_seconds() -> float:
            elapsed = time.time() - start
            return max(0.1, budget_ms / 1000.0 - elapsed)

        # WORLD (fast snapshot via world orchestrator)
        world_section = BundleSection(data={}, canonical=False, priority=4,
                                      last_changed_at=time.time(),
                                      ttl=self.section_ttls.get('world', 15.0),
                                      version='world_shallow')
        try:
            orchestrator = getattr(self.ctx, "world_orchestrator", None)
            if orchestrator and _remaining_seconds() > 0.1:
                if await self.ctx.await_orchestrator("world"):
                    world_bundle = await asyncio.wait_for(
                        orchestrator.get_scene_bundle(scope),
                        timeout=min(1.0, _remaining_seconds()),
                    )
                    summary = world_bundle.get('summary', {}) if isinstance(world_bundle, dict) else {}
                    world_section = BundleSection(
                        data=summary,
                        canonical=False,
                        priority=4,
                        last_changed_at=time.time(),
                        ttl=self.section_ttls.get('world', 15.0),
                        version='world_shallow',
                    )
        except Exception:
            logger.debug("Shallow world fetch failed", exc_info=True)

        # Preload comprehensive context (best-effort) for NPCs and metadata
        aggregated_context: Optional[Dict[str, Any]] = None
        try:
            aggregated_context = await asyncio.wait_for(
                get_comprehensive_context(
                    user_id=self.ctx.user_id,
                    conversation_id=self.ctx.conversation_id,
                    input_text=getattr(self.ctx, 'last_user_input', '') or '',
                    location=scope.location_id or scope.location_name,
                    context_budget=1000,
                    use_delta=True,
                    summary_level=0,
                ),
                timeout=min(0.8, _remaining_seconds()),
            )
        except Exception:
            logger.debug("Shallow aggregated context fetch failed", exc_info=True)

        # NPCs (projection only)
        npc_data = []
        if aggregated_context:
            raw_npcs = aggregated_context.get('npcsPresent') or []
            for row in raw_npcs[:5]:
                if not isinstance(row, dict):
                    continue
                npc_data.append({
                    'id': row.get('npc_id') or row.get('id'),
                    'name': row.get('npc_name') or row.get('name'),
                    'role': row.get('role'),
                    'current_location': row.get('current_location') or row.get('location'),
                    'relationship': {
                        'trust': row.get('trust'),
                        'respect': row.get('respect'),
                        'closeness': row.get('closeness'),
                    }
                })
        else:
            for npc_id in self.ctx.current_scene_npcs[:5]:
                snapshot = self.ctx.npc_snapshots.get(npc_id, {})
                if not isinstance(snapshot, dict):
                    snapshot = {}
                npc_data.append({
                    'id': npc_id,
                    'name': snapshot.get('name'),
                    'role': snapshot.get('role'),
                    'current_location': snapshot.get('current_location'),
                    'relationship': snapshot.get('relationship', {}),
                })

        npcs_section = BundleSection(
            data=NPCSectionData(npcs=npc_data, canonical_count=0),
            canonical=False,
            priority=8,
            last_changed_at=time.time(),
            ttl=self.section_ttls.get('npcs', 30.0),
            version='npcs_shallow',
        )

        # Conflicts (cheap system state only)
        conflicts_section = BundleSection(
            data={'active': [], 'tensions': {}, 'opportunities': []},
            canonical=False,
            priority=5,
            last_changed_at=time.time(),
            ttl=self.section_ttls.get('conflicts', 30.0),
            version='conflicts_shallow',
        )
        if self.conflict_synthesizer and _remaining_seconds() > 0.2:
            try:
                state = await asyncio.wait_for(
                    self.conflict_synthesizer.get_system_state(),
                    timeout=min(0.7, _remaining_seconds()),
                )
            except Exception:
                logger.debug("Shallow conflict state fetch failed", exc_info=True)
            else:
                if isinstance(state, dict):
                    conflicts_section = BundleSection(
                        data={
                            'active': (state.get('active_conflicts') or [])[:2],
                            'tensions': state.get('metrics') or {},
                            'opportunities': [],
                        },
                        canonical=False,
                        priority=5,
                        last_changed_at=time.time(),
                        ttl=self.section_ttls.get('conflicts', 30.0),
                        version='conflicts_shallow',
                    )

        # Lore/Memories/Narrative placeholders
        lore_section = BundleSection(
            data=LoreSectionData(location={}, world={}, canonical_rules=[]),
            canonical=False,
            priority=6,
            last_changed_at=time.time(),
            ttl=self.section_ttls.get('lore', 300.0),
            version='lore_shallow',
        )
        memories_section = BundleSection(
            data=MemorySectionData(relevant=[], recent=[], patterns=[]),
            canonical=False,
            priority=7,
            last_changed_at=time.time(),
            ttl=self.section_ttls.get('memories', 120.0),
            version='mem_shallow',
        )
        narrative_section = BundleSection(
            data={},
            canonical=False,
            priority=3,
            last_changed_at=time.time(),
            ttl=self.section_ttls.get('narrative', 30.0),
            version='narr_shallow',
        )

        metadata = {
            'schema_version': SCHEMA_VERSION,
            'link_hints': scope.link_hints,
            'fetch_time': time.time() - start,
            'shallow': True,
        }
        if aggregated_context:
            metadata['source'] = 'aggregated_context'

        return ContextBundle(
            scene_scope=scope,
            npcs=npcs_section,
            memories=memories_section,
            lore=lore_section,
            conflicts=conflicts_section,
            world=world_section,
            narrative=narrative_section,
            metadata=metadata,
        )
    
    async def _save_cache(self, scene_key: str, bundle: ContextBundle):
        """Save bundle to both memory and Redis cache"""
        self.bundle_cache.set(scene_key, bundle)  # Use LRU set method
        
        # Try to reconnect if backoff expired
        if self.redis_client is None and time.time() >= self._redis_backoff_until:
            await self._try_connect_redis()
        
        if self.redis_client and time.time() >= self._redis_backoff_until:
            try:
                bundle_json = json_dumps(bundle.to_dict())
                
                # Convert to bytes for accurate accounting
                payload = bundle_json.encode('utf-8') if isinstance(bundle_json, str) else bundle_json
                
                # Cap Redis payload size to avoid memory bloat
                max_redis_size = 256 * 1024  # 256KB soft cap
                if len(payload) > max_redis_size:
                    # Drop lowest priority sections to fit
                    trimmed_bundle = self._trim_bundle_for_cache(bundle, max_redis_size)
                    bundle_json = json_dumps(trimmed_bundle.to_dict())
                    payload = bundle_json.encode('utf-8') if isinstance(bundle_json, str) else bundle_json
                
                # Use max section TTL for Redis
                ttl = max(self.section_ttls.values())
                await self.redis_client.setex(
                    f"nyx:bundle:{scene_key}",
                    int(ttl),
                    payload
                )
                self.metrics['bytes_cached']['redis'] += len(payload)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
                self._set_redis_backoff()
    
    def _trim_bundle_for_cache(self, bundle: ContextBundle, max_size: int) -> ContextBundle:
        """Trim bundle by dropping low-priority sections to fit size limit"""
        sections = [
            ('narrative', bundle.narrative, 3),
            ('world', bundle.world, 4),
            ('conflicts', bundle.conflicts, 5),
            ('lore', bundle.lore, 6),
            ('memories', bundle.memories, 7),
            ('npcs', bundle.npcs, 8),
        ]
        
        # Sort by priority (ascending - drop lowest first)
        sections.sort(key=lambda x: x[2])
        
        # Try dropping sections until it fits
        trimmed = bundle.to_dict()
        for name, section, priority in sections:
            if not section.canonical:  # Never drop canonical sections
                # Create minimal placeholder
                trimmed[name] = BundleSection(
                    data={}, 
                    canonical=False, 
                    priority=priority,
                    last_changed_at=section.last_changed_at,
                    ttl=section.ttl
                ).to_dict()
                
                # Check if it fits now
                test_json = json_dumps(trimmed)
                test_payload = test_json.encode('utf-8') if isinstance(test_json, str) else test_json
                if len(test_payload) <= max_size:
                    return ContextBundle.from_dict(trimmed)
        
        # If still too large after dropping all non-canonical, hard-compact remaining
        bundle_obj = ContextBundle.from_dict(trimmed)
        
        # Hard compact the largest sections
        if hasattr(bundle_obj.memories.data, 'compact'):
            bundle_obj.memories.data = bundle_obj.memories.data.compact()
        if hasattr(bundle_obj.npcs.data, 'compact'):
            bundle_obj.npcs.data = bundle_obj.npcs.data.compact(max_npcs=3)
        if hasattr(bundle_obj.lore.data, 'compact'):
            bundle_obj.lore.data = bundle_obj.lore.data.compact()
        
        # Re-check size after compaction
        test_json = json_dumps(bundle_obj.to_dict())
        test_payload = test_json.encode('utf-8') if isinstance(test_json, str) else test_json
        
        # If STILL too large, progressively drop optional fields
        if len(test_payload) > max_size:
            # Drop recent memories / patterns safely
            mem = bundle_obj.memories.data
            if isinstance(mem, MemorySectionData):
                mem.recent = []
                mem.patterns = []
            elif isinstance(mem, dict):
                mem.pop('recent', None)
                mem.pop('patterns', None)
        
            # Drop world events safely
            if isinstance(bundle_obj.world.data, dict):
                bundle_obj.world.data.pop('events', None)
        
        return bundle_obj
    
    async def _fetch_npc_section(self, scope: SceneScope) -> BundleSection:
        """
        Fetch NPC context for scene — prefer orchestrator scene bundle, then overlay
        dynamic relationship data (player↔NPC) including patterns/archetypes/momentum
        and the full dimensions vector. Falls back to per-NPC snapshots if needed.
        """
        fallback_section = BundleSection(data=NPCSectionData(npcs=[], canonical_count=0))

        if not self.ctx.is_orchestrator_ready("npc"):
            return fallback_section

        if not await self.ctx.await_orchestrator("npc"):
            return fallback_section

        if not self.ctx.npc_orchestrator:
            return BundleSection(
                data=NPCSectionData(npcs=[], canonical_count=0),
                canonical=False,
                priority=0,
            )
    
        npc_ids = list(scope.npc_ids)[:10] if getattr(scope, "npc_ids", None) else []
    
        # Fast path: orchestrator scene bundle (one RPC, includes scene/group dynamics)
        obundle = None
        if hasattr(self.ctx.npc_orchestrator, "get_scene_bundle"):
            try:
                obundle = await self.ctx.npc_orchestrator.get_scene_bundle(scope)
            except Exception as e:
                logger.debug(f"[NyxContext] Orchestrator scene bundle fast-path failed, fallback: {e}")
    
        canonical = False
        last_changed = time.time()
        version = f"npcs_{len(npc_ids)}_{last_changed}"
        npc_items: List[Dict[str, Any]] = []
        canonical_count = 0
    
        if obundle:
            data = obundle.get("data", {}) or {}
            npc_items = data.get("npcs", []) or []
            canonical = bool(obundle.get("canonical", False))
            last_changed = float(obundle.get("last_changed_at", last_changed))
            version = obundle.get("version") or version
            canonical_count = int(data.get("canonical_count", 0))
    
            # Persist orchestrator scene/group dynamics in context metadata for reuse
            try:
                self.ctx.current_context.setdefault("_npc_scene_meta", {})[scope.to_key()] = {
                    "scene_dynamics": data.get("scene_dynamics", {}),
                    "group_dynamics": data.get("group_dynamics", {})
                }
            except Exception:
                pass
    
            # If orchestrator bundle omitted explicit npc_ids, infer them from items
            if not npc_ids and npc_items:
                try:
                    npc_ids = [int(it.get("id")) for it in npc_items if isinstance(it, dict) and it.get("id") is not None]
                except Exception:
                    pass
        else:
            # Fallback: per-NPC snapshots
            async def _fetch_one(nid: int):
                try:
                    snap = await self.ctx.npc_orchestrator.get_npc_snapshot(nid, light=True)
                    entry = {
                        "id": nid,
                        "name": snap.name,
                        "role": snap.role,
                        "canonical": bool(snap.canonical_events),
                        "status": snap.status if not hasattr(snap.status, "value") else snap.status.value,
                        "relationship": {
                            "trust": snap.trust,
                            "respect": snap.respect,
                            "closeness": snap.closeness,
                        },
                        "current_intent": snap.emotional_state.get("intent") if snap.emotional_state else None,
                    }
                    return entry, bool(snap.canonical_events)
                except Exception as e:
                    logger.warning(f"[NyxContext] Snapshot fetch failed for NPC {nid}: {e}")
                    return None, False
    
            results = await asyncio.gather(*[_fetch_one(n) for n in npc_ids], return_exceptions=False)
            for it, is_canon in results:
                if isinstance(it, dict):
                    npc_items.append(it)
                    if is_canon:
                        canonical_count += 1
    
        # Overlay dynamic relationship data (player↔NPC) onto each entry
        # Includes patterns/archetypes, momentum (magnitude & direction), and full dimensions.
        relationships_meta: Dict[str, Any] = {}
        if npc_ids:
            try:
                from logic.dynamic_relationships import OptimizedRelationshipManager
                dyn = OptimizedRelationshipManager(self.ctx.user_id, self.ctx.conversation_id)
    
                async def _fetch_rel(nid: int):
                    try:
                        state = await dyn.get_relationship_state("player", 1, "npc", int(nid))
                        dims = state.dimensions
                        # Map to 0..100 UI scalars from [-100..100]
                        trust = int(round((float(dims.trust) + 100.0) / 2.0))
                        respect = int(round((float(dims.respect) + 100.0) / 2.0))
                        # Closeness from positive-affection, intimacy, frequency (0..100)
                        intimacy = max(0.0, float(dims.intimacy))
                        affection_pos = max(0.0, float(dims.affection))
                        frequency = max(0.0, float(dims.frequency))
                        closeness = int(round((intimacy + affection_pos + frequency) / 3.0))
    
                        return nid, {
                            "trust": trust, "respect": respect, "closeness": closeness,
                            "patterns": list(state.history.active_patterns),
                            "archetypes": list(state.active_archetypes),
                            "momentum_mag": round(state.momentum.get_magnitude(), 2),
                            "momentum_dir": state.momentum.get_direction(),
                            "dims": state.dimensions.to_dict()
                        }
                    except Exception as re:
                        logger.debug(f"[NyxContext] dynamic rel fetch failed for NPC {nid}: {re}")
                        return nid, None
    
                rel_results = await asyncio.gather(*[_fetch_rel(n) for n in npc_ids], return_exceptions=False)
                rel_map: Dict[int, Dict[str, Any]] = {
                    nid: payload for nid, payload in rel_results
                    if isinstance(nid, int) and isinstance(payload, dict)
                }
    
                # Patch items with dynamic numbers; store compact meta for tools
                for item in npc_items:
                    try:
                        nid = int(item.get("id"))
                    except Exception:
                        continue
                    payload = rel_map.get(nid)
                    if not payload:
                        continue
    
                    # Overlay relationship scalars with dynamic versions
                    item.setdefault("relationship", {})
                    item["relationship"]["trust"] = payload["trust"]
                    item["relationship"]["respect"] = payload["respect"]
                    item["relationship"]["closeness"] = payload["closeness"]
    
                    # Attach compact dynamic meta (trim patterns/archetypes)
                    item["relationship_dynamic"] = {
                        "patterns": payload["patterns"][:2],
                        "archetypes": payload["archetypes"][:2],
                        "momentum": {
                            "magnitude": payload["momentum_mag"],
                            "direction": payload["momentum_dir"]
                        },
                        "dimensions": payload["dims"]  # full vector: trust/affection/respect/familiarity/tension/intimacy/frequency
                    }
    
                    # Populate metadata._relationships for cross-section tools
                    relationships_meta[str(nid)] = {
                        "with_player": {
                            "trust": payload["trust"],
                            "respect": payload["respect"],
                            "closeness": payload["closeness"],
                            "momentum": {
                                "magnitude": payload["momentum_mag"],
                                "direction": payload["momentum_dir"]
                            }
                        }
                    }
    
            except Exception as e:
                logger.debug(f"[NyxContext] dynamic relationship overlay skipped: {e}")
    
        # Persist relationships meta in bundle-level metadata (broker merges later)
        if relationships_meta:
            try:
                # Stash in current_context; ContextBundle.sections exposes it as 'relationships'
                rel_meta = self.ctx.current_context.setdefault('_relationships', {})
                rel_meta.update(relationships_meta)
            except Exception:
                pass
    
        section_data = NPCSectionData(npcs=npc_items, canonical_count=canonical_count)
        return BundleSection(
            data=section_data,
            canonical=canonical or canonical_count > 0,
            priority=8,
            last_changed_at=last_changed,
            version=version
        )
    
    async def _fetch_memory_section(self, scope: SceneScope) -> BundleSection:
        """Fetch relevant memories using existing orchestrator API and link hints"""
        fallback_section = BundleSection(
            data=MemorySectionData(relevant=[], recent=[], patterns=[])
        )

        if not self.ctx.is_orchestrator_ready("memory"):
            return fallback_section

        if not await self.ctx.await_orchestrator("memory"):
            return fallback_section

        if not self.ctx.memory_orchestrator:
            return BundleSection(data=MemorySectionData(relevant=[], recent=[], patterns=[]),
                               canonical=False, priority=0)

        relevant_memories = []
        recent_memories = []
        patterns = []

        try:
            # Build NPC list including link hints
            npc_list = list(scope.npc_ids)[:3]
            # Add related NPCs from link hints
            hint_npcs = scope.link_hints.get('related_npcs', [])[:2]
            for hint_npc in hint_npcs:
                if hint_npc not in npc_list:
                    npc_list.append(hint_npc)

            orchestrator = self.ctx.memory_orchestrator
            base_timeout = getattr(self.ctx, 'memory_fetch_timeout', 1.5)
            timeout = self._get_section_timeout('memories', base_timeout)

            async def _await_with_timeout(coro, label: str):
                try:
                    return await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Memory fetch timed out for %s after %.2fs", label, timeout
                    )
                except Exception:
                    logger.error("Memory fetch failed for %s", label, exc_info=True)
                return None

            async def _fetch_npc_memories(npc_id: Union[int, str]):
                if hasattr(orchestrator, 'retrieve_memories'):
                    query = " ".join(list(scope.topics)[:5])
                    return await _await_with_timeout(
                        orchestrator.retrieve_memories(
                            entity_type=EntityType.NPC,
                            entity_id=npc_id,
                            query=query,
                            limit=3,
                            use_llm_analysis=False
                        ),
                        f"NPC {npc_id}"
                    )
                query = " ".join(scope.topics)
                return await _await_with_timeout(
                    orchestrator.recall(
                        entity_type=EntityType.NPC,
                        entity_id=npc_id,
                        query=query,
                        limit=3
                    ),
                    f"NPC {npc_id}"
                )

            async def _fetch_location_memories(location_id: Union[int, str]):
                if hasattr(orchestrator, 'retrieve_memories'):
                    return await _await_with_timeout(
                        orchestrator.retrieve_memories(
                            entity_type=EntityType.LOCATION,
                            entity_id=location_id,
                            limit=5,
                            use_llm_analysis=False
                        ),
                        f"location {location_id}"
                    )
                return await _await_with_timeout(
                    orchestrator.recall(
                        entity_type=EntityType.LOCATION,
                        entity_id=location_id,
                        query="",
                        limit=5
                    ),
                    f"location {location_id}"
                )

            async def _fetch_patterns():
                try:
                    return await asyncio.wait_for(
                        self.ctx.analyze_memory_patterns(
                            topic=", ".join(list(scope.topics)[:3])
                        ),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Memory pattern analysis timed out after %.2fs", timeout
                    )
                except Exception:
                    logger.error("Memory pattern analysis failed", exc_info=True)
                return None

            tasks: List[asyncio.Task] = []
            npc_tasks: List[asyncio.Task] = []

            for npc_id in npc_list:
                npc_task = asyncio.create_task(_fetch_npc_memories(npc_id))
                npc_tasks.append(npc_task)
                tasks.append(npc_task)

            location_task: Optional[asyncio.Task] = None
            if scope.location_id:
                location_task = asyncio.create_task(
                    _fetch_location_memories(scope.location_id)
                )
                tasks.append(location_task)

            pattern_task: Optional[asyncio.Task] = None
            if hasattr(self.ctx, 'analyze_memory_patterns') and scope.topics:
                pattern_task = asyncio.create_task(_fetch_patterns())
                tasks.append(pattern_task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

            for npc_task in npc_tasks:
                if npc_task.cancelled():
                    continue
                try:
                    result = npc_task.result()
                except Exception:
                    continue
                if result and 'memories' in result:
                    relevant_memories.extend(result['memories'][:3])

            if location_task and not location_task.cancelled():
                try:
                    location_result = location_task.result()
                except Exception:
                    location_result = None
                if location_result and 'memories' in location_result:
                    recent_memories.extend(location_result['memories'][:5])

            if pattern_task and not pattern_task.cancelled():
                try:
                    pattern_result = pattern_task.result()
                except Exception:
                    pattern_result = None
                if pattern_result and 'predictions' in pattern_result:
                    patterns = pattern_result['predictions'][:3]

        except Exception as e:
            logger.error(f"Memory fetch failed: {e}")
        
        # Use MemorySectionData for strong typing
        section_data = MemorySectionData(
            relevant=relevant_memories,
            recent=recent_memories,
            patterns=patterns
        )
        
        return BundleSection(
            data=section_data,
            canonical=False,
            priority=7,
            last_changed_at=time.time(),
            version=f"mem_{len(relevant_memories)}_{len(recent_memories)}"
        )
    
    async def _fetch_lore_section(self, scope: SceneScope) -> BundleSection:
        """Fetch lore context; prefer orchestrator scene bundle when available."""
        fallback_section = BundleSection(
            data=LoreSectionData(location={}, world={}, canonical_rules=[])
        )

        if not self.ctx.is_orchestrator_ready("lore"):
            return fallback_section

        if not await self.ctx.await_orchestrator("lore"):
            return fallback_section

        if not self.ctx.lore_orchestrator:
            return BundleSection(data=LoreSectionData(location={}, world={}, canonical_rules=[]),
                                 canonical=False, priority=0)
    
        base_timeout = getattr(self.ctx, 'lore_fetch_timeout', 2.0)
        timeout_budget = self._get_section_timeout('lore', base_timeout)

        # --- Fast path: one RPC for the whole scene ---
        if hasattr(self.ctx.lore_orchestrator, 'get_scene_bundle'):
            try:
                bundle = await asyncio.wait_for(
                    self.ctx.lore_orchestrator.get_scene_bundle(scope),
                    timeout=timeout_budget,
                )
                data = bundle.get('data', {}) or {}
                rules_raw = data.get('canonical_rules', []) or []
    
                # Coerce canonical_rules to List[str]
                rules: List[str] = []
                for r in rules_raw:
                    if isinstance(r, dict) and 'text' in r:
                        rules.append(str(r['text']))
                    elif isinstance(r, str):
                        rules.append(r)
    
                section_data = LoreSectionData(
                    location=data.get('location', {}) or {},
                    world=data.get('world', {}) or {},
                    canonical_rules=rules
                )
    
                return BundleSection(
                    data=section_data,
                    canonical=bool(bundle.get('canonical')) or bool(rules),
                    priority=6,
                    last_changed_at=float(bundle.get('last_changed_at', time.time())),
                    version=bundle.get('version') or f"lore_scene_{int(time.time())}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Lore scene bundle fetch timed out after %.2fs; falling back",
                    timeout_budget,
                )
            except Exception as e:
                logger.error(f"Lore scene bundle fetch failed, falling back: {e}")
    
        # --- Fallback path (legacy calls) ---
        location_lore, world_lore, canonical_rules = {}, {}, []
        location_task: Optional[asyncio.Task] = None
        canonical_task: Optional[asyncio.Task] = None
        try:
            location_coro: Optional[Awaitable[Dict[str, Any]]] = None
            if scope.location_id:
                if isinstance(scope.location_id, int) and hasattr(self.ctx.lore_orchestrator, '_fetch_location_lore_for_bundle'):
                    location_coro = self.ctx.lore_orchestrator._fetch_location_lore_for_bundle(scope.location_id)
                else:
                    location_coro = self.ctx.lore_orchestrator.get_location_context(str(scope.location_id))

            if location_coro is not None:
                location_task = asyncio.create_task(location_coro)

            if hasattr(self.ctx.lore_orchestrator, 'check_canonical_consistency'):
                canonical_task = asyncio.create_task(self.ctx.lore_orchestrator.check_canonical_consistency())

            location_result: Optional[Dict[str, Any]] = None
            if location_task is not None:
                try:
                    location_result = await asyncio.wait_for(
                        location_task, timeout=timeout_budget
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Location lore fetch timed out after %.2fs", timeout_budget
                    )
                    location_result = None

            if location_result:
                location_lore = {
                    'description': (location_result.get('description') or '')[:200],
                    'governance': location_result.get('governance') or {},
                    'culture': location_result.get('culture') or {},
                }
                if 'tags' in location_result:
                    scope.lore_tags.update(location_result['tags'][:10])

            tag_seed = list(scope.lore_tags)[:5]
            if 'related_tags' in scope.link_hints:
                tag_seed.extend(scope.link_hints['related_tags'][:3])

            if tag_seed and hasattr(self.ctx.lore_orchestrator, 'get_tagged_lore'):
                try:
                    tagged_lore = await asyncio.wait_for(
                        self.ctx.lore_orchestrator.get_tagged_lore(tags=tag_seed),
                        timeout=timeout_budget,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Tagged lore fetch timed out after %.2fs", timeout_budget
                    )
                    tagged_lore = None
                world_lore = tagged_lore or {}

            if canonical_task is not None:
                canonical_payload: Optional[Dict[str, Any]] = None
                try:
                    canonical_payload = await asyncio.wait_for(
                        canonical_task, timeout=timeout_budget
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Canonical lore consistency check timed out after %.2fs",
                        timeout_budget,
                    )
                if canonical_payload:
                    canonical_rules = list(canonical_payload.get('rules') or [])

        except Exception as e:
            if location_task is not None and not location_task.done():
                location_task.cancel()
                with contextlib.suppress(Exception):
                    await location_task
            if canonical_task is not None and not canonical_task.done():
                canonical_task.cancel()
                with contextlib.suppress(Exception):
                    await canonical_task
            logger.error(f"Lore fetch failed: {e}")
    
        section_data = LoreSectionData(
            location=location_lore,
            world=world_lore,
            canonical_rules=canonical_rules
        )
        return BundleSection(
            data=section_data,
            canonical=bool(canonical_rules),
            priority=6,
            last_changed_at=time.time(),
            version=f"lore_{scope.location_id}_{len(scope.lore_tags)}"
        )
    
    async def _fetch_conflict_section(self, scope: SceneScope) -> BundleSection:
        """Fetch conflict context using existing orchestrator API and link hints."""
        # Resolve synthesizer once and guard
        fallback_section = BundleSection(data={})

        if not self.ctx.is_orchestrator_ready("conflict"):
            return fallback_section

        if not await self.ctx.await_orchestrator("conflict"):
            return fallback_section

        synthesizer = getattr(self.ctx, "conflict_synthesizer", None)
        if not synthesizer:
            return BundleSection(data={}, canonical=False, priority=0)
    
        # Fast path: if the synthesizer exposes an optimized scene bundle, use it.
        if hasattr(synthesizer, "get_scene_bundle"):
            try:
                bundle = await synthesizer.get_scene_bundle(scope)
                bundle = bundle or {}
                last_changed = float(bundle.get("last_changed_at", time.time()))
                data = {
                    "active": bundle.get("active", bundle.get("conflicts", [])) or [],
                    "tensions": bundle.get("tensions", {}) or {},
                    "opportunities": bundle.get("opportunities", []) or [],
                }
                # Optional: surface overall/world tension for upstream packing/use
                wt = bundle.get("world_tension")
                if isinstance(wt, (int, float)):
                    data.setdefault("tensions", {})
                    data["tensions"]["overall"] = float(wt)
        
                return BundleSection(
                    data=data,
                    canonical=True,
                    priority=5,
                    last_changed_at=last_changed,             # use orchestrator’s timestamp
                    version=f"conflict_opt_{int(last_changed)}_{len(data['active'])}",
                )
            except Exception as e:
                logger.error("Optimized conflict fetch failed, falling back: %s", e)
    
        # ----- Fallback path (existing APIs) -----
        conflict_data = {
            "active": [],
            "tensions": {},
            "opportunities": [],
        }

        try:
            conflict_state_timeout = getattr(self.ctx, "conflict_state_timeout", 0.75)
            tension_timeout = getattr(self.ctx, "conflict_tension_timeout", 0.5)

            # Build NPC list including link hints for conflict filtering
            relevant_npcs = set(getattr(scope, "npc_ids", []) or [])
            link_hints = getattr(scope, "link_hints", {}) or {}
            if isinstance(link_hints, dict) and "related_npcs" in link_hints:
                related = link_hints.get("related_npcs") or []
                relevant_npcs.update(related[:3])

            timed_out_conflicts: List[int] = []
            failed_conflicts: List[int] = []
            conflict_tasks: List[asyncio.Task] = []
            tension_task: Optional[asyncio.Task] = None

            async def _fetch_conflict_state(conflict_id: int):
                try:
                    result = await asyncio.wait_for(
                        synthesizer.get_conflict_state(conflict_id),
                        timeout=conflict_state_timeout,
                    )
                    return conflict_id, result
                except asyncio.TimeoutError:
                    timed_out_conflicts.append(conflict_id)
                    logger.warning(
                        "Conflict state fetch for %s timed out after %.2fs",
                        conflict_id,
                        conflict_state_timeout,
                    )
                except Exception:
                    failed_conflicts.append(conflict_id)
                    logger.error(
                        "Conflict state fetch failed for %s", conflict_id, exc_info=True
                    )
                return conflict_id, None

            async def _fetch_tensions():
                if not hasattr(self.ctx, "calculate_conflict_tensions"):
                    return None
                try:
                    return await asyncio.wait_for(
                        self.ctx.calculate_conflict_tensions(),
                        timeout=tension_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Conflict tension calculation timed out after %.2fs",
                        tension_timeout,
                    )
                except Exception:
                    logger.error("Conflict tension calculation failed", exc_info=True)
                return None

            conflict_ids = list(getattr(scope, "conflict_ids", []))[:5]

            if conflict_ids or relevant_npcs:
                async with asyncio.TaskGroup() as tg:
                    for conflict_id in conflict_ids:
                        task = tg.create_task(_fetch_conflict_state(conflict_id))
                        conflict_tasks.append(task)
                    if relevant_npcs:
                        tension_task = tg.create_task(_fetch_tensions())

            for task in conflict_tasks:
                if task.cancelled():
                    continue
                try:
                    conflict_id, state = task.result()
                except Exception:
                    continue
                if not state:
                    continue
                subsystem = state.get("subsystem_data", {}) if isinstance(state, dict) else {}
                tension = subsystem.get("tension", {}) if isinstance(subsystem, dict) else {}
                stakeholder = subsystem.get("stakeholder", {}) if isinstance(subsystem, dict) else {}

                conflict_data["active"].append(
                    {
                        "id": conflict_id,
                        "type": state.get("conflict_type") if isinstance(state, dict) else None,
                        "intensity": tension.get("level", 0.5),
                        "stakeholders": stakeholder.get("stakeholders", []),
                    }
                )

            if tension_task and not tension_task.cancelled():
                try:
                    all_tensions = tension_task.result()
                except Exception:
                    all_tensions = None
                if relevant_npcs and all_tensions:
                    conflict_data["tensions"] = self._filter_tensions_for_npcs(
                        all_tensions,
                        relevant_npcs,
                    )

            if timed_out_conflicts or failed_conflicts:
                logger.info(
                    "Conflict section assembled with partial data (timed_out=%s, failed=%s)",
                    timed_out_conflicts,
                    failed_conflicts,
                )

        except Exception as e:
            logger.error("Conflict fetch failed: %s", e)

        return BundleSection(
            data=conflict_data,
            canonical=False,
            priority=5,
            last_changed_at=time.time(),
            version=f"conflict_{len(conflict_data['active'])}",
        )
    
    def _filter_tensions_for_npcs(self, all_tensions: Dict[str, float], 
                                  npc_ids: Set[int]) -> Dict[str, float]:
        """Filter tensions to only those involving specified NPCs"""
        filtered = {}
        npc_id_strs = {str(npc_id) for npc_id in npc_ids}
        
        for key, value in all_tensions.items():
            # Assume keys are formatted like "npc:12|npc:47" or "12_47"
            if '|' in key:
                # Format: "npc:12|npc:47"
                parts = key.split('|')
                if any(f"npc:{npc_id}" in parts for npc_id in npc_id_strs):
                    filtered[key] = value
            elif '_' in key:
                # Format: "12_47" - check exact matches
                parts = key.split('_')
                if any(part in npc_id_strs for part in parts):
                    filtered[key] = value
            else:
                # Single entity or unknown format - check if any NPC ID appears
                for npc_id in npc_id_strs:
                    # Use word boundary to avoid "12" matching "312"
                    if re.search(rf'\b{npc_id}\b', key):
                        filtered[key] = value
                        break
        
        return filtered
    
    async def _fetch_world_section(self, scope: SceneScope) -> BundleSection:
        """Fetch world state with safe field access and enum handling"""

        fallback_section = self._build_section_fallback('world')

        # Align readiness checks with lore orchestration – short circuit when not ready
        if not self.ctx.is_orchestrator_ready("world"):
            return fallback_section

        if not await self.ctx.await_orchestrator("world"):
            return fallback_section

        def _empty_section() -> BundleSection:
            return self._build_section_fallback('world')

        async def _fetch() -> BundleSection:
            orchestrator = getattr(self.ctx, "world_orchestrator", None)
            if not orchestrator:
                raise RuntimeError("World orchestrator unavailable")

            base_timeout = getattr(self.ctx, 'world_fetch_timeout', 1.0)
            timeout_budget = self._get_section_timeout('world', base_timeout)

            def _extract_bundle_payload(bundle: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(bundle, dict):
                    return {}
                data_block = bundle.get('data')
                if isinstance(data_block, dict):
                    return data_block
                summary = bundle.get('summary')
                if isinstance(summary, dict):
                    return summary
                return {
                    key: value
                    for key, value in bundle.items()
                    if key in {'time', 'mood', 'weather', 'events', 'vitals', 'location'}
                }

            async def _invoke_targeted(handler: Callable[..., Any], label: str) -> Any:
                """Invoke targeted orchestrator helper safely."""

                async def _call_with_scope(pass_scope: bool) -> Any:
                    try:
                        if pass_scope:
                            result = handler(scope)
                        else:
                            result = handler()
                    except TypeError as exc:
                        if pass_scope:
                            return exc
                        raise
                    except Exception:
                        logger.debug(
                            "World targeted fetch '%s' failed during call", label,
                            exc_info=True,
                        )
                        return None

                    if isinstance(result, Exception):
                        # Handler rejected scope argument; try without scope
                        return result

                    if asyncio.iscoroutine(result):
                        try:
                            return await asyncio.wait_for(result, timeout=timeout_budget)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "World targeted fetch '%s' timed out after %.2fs",
                                label,
                                timeout_budget,
                            )
                            return None
                        except Exception:
                            logger.debug(
                                "World targeted fetch '%s' await failed", label,
                                exc_info=True,
                            )
                            return None
                    return result

                call_result = await _call_with_scope(True)
                if isinstance(call_result, Exception):
                    call_result = await _call_with_scope(False)
                return call_result

            async def _fetch_via_targeted() -> Dict[str, Any]:
                targeted_data: Dict[str, Any] = {}
                targeted_specs = (
                    ('get_time_snapshot', 'time'),
                    ('get_weather', 'weather'),
                    ('get_world_mood', 'mood'),
                    ('get_active_events', 'events'),
                    ('get_player_vitals', 'vitals'),
                )

                tasks: List[Tuple[str, asyncio.Task[Any]]] = []
                for method_name, field in targeted_specs:
                    handler = getattr(orchestrator, method_name, None)
                    if not callable(handler):
                        continue
                    task = asyncio.create_task(_invoke_targeted(handler, method_name))
                    tasks.append((field, task))

                for field, task in tasks:
                    try:
                        value = await task
                    except Exception:
                        logger.debug(
                            "World targeted fetch '%s' task failed", field,
                            exc_info=True,
                        )
                        continue
                    if value is None:
                        continue
                    targeted_data[field] = value

                if not targeted_data and hasattr(orchestrator, 'expand_state'):
                    try:
                        expanded = await asyncio.wait_for(
                            orchestrator.expand_state(aspects=['summary'], scope=scope),
                            timeout=timeout_budget,
                        )
                        summary = {}
                        if isinstance(expanded, dict):
                            summary = expanded.get('summary') or {}
                            if not summary:
                                bundle = expanded.get('bundle') or {}
                                if isinstance(bundle, dict):
                                    summary = bundle.get('summary') or {}
                        if isinstance(summary, dict):
                            targeted_data.update(summary)
                    except Exception:
                        logger.debug("World expand_state fallback failed", exc_info=True)

                return targeted_data

            bundle: Dict[str, Any] = {}
            try:
                bundle = await asyncio.wait_for(
                    orchestrator.get_scene_bundle(scope),
                    timeout=timeout_budget,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "World scene bundle fetch timed out after %.2fs; attempting targeted fallback",
                    timeout_budget,
                )
            except Exception:
                logger.error(
                    "World scene bundle fetch failed; attempting targeted fallback",
                    exc_info=True,
                )

            if isinstance(bundle, dict) and bundle:
                payload = _extract_bundle_payload(bundle)
                last_changed = float(bundle.get('last_changed_at', time.time()))
                version = str(bundle.get('version') or bundle.get('world_version') or f"world_{int(last_changed)}")
                canonical = bool(bundle.get('canonical'))
                return BundleSection(
                    data=payload,
                    canonical=canonical,
                    priority=4,
                    last_changed_at=last_changed,
                    version=version,
                )

            targeted_data = await _fetch_via_targeted()
            if not targeted_data:
                raise RuntimeError("Targeted world fetch returned no data")

            return BundleSection(
                data=targeted_data,
                canonical=False,
                priority=4,
                last_changed_at=time.time(),
                version=f"world_fallback_{int(time.time())}",
            )

        return await self.perform_fetch_and_cache(
            orchestrator_name="world",
            cache_attribute="_world_section_cache",
            expires_attribute="_world_section_cache_expires_at",
            fetcher=_fetch,
            fallback_factory=_empty_section,
            ttl=self.section_ttls.get("world"),
        )
    
    async def _fetch_narrative_section(self, scope: SceneScope) -> BundleSection:
        """Fetch narrative context with safe method calls"""
        fallback_section = BundleSection(data={})

        if not self.ctx.is_orchestrator_ready("world"):
            return fallback_section

        if not await self.ctx.await_orchestrator("world"):
            return fallback_section
        # Narrator is part of world systems
        if not self.ctx.slice_of_life_narrator:
            return BundleSection(data={}, canonical=False, priority=0)
        
        narrative_data = {}
        
        try:
            # Safe method calls with hasattr checks
            if hasattr(self.ctx.slice_of_life_narrator, 'get_scene_atmosphere'):
                atmosphere = await self.ctx.slice_of_life_narrator.get_scene_atmosphere(
                    location=scope.location_id or scope.location_name
                )
                narrative_data['atmosphere'] = atmosphere
            
            if hasattr(self.ctx.slice_of_life_narrator, 'get_recent_narrations'):
                recent = await self.ctx.slice_of_life_narrator.get_recent_narrations(limit=2)
                narrative_data['recent'] = recent
                
        except Exception as e:
            logger.error(f"Narrative fetch failed: {e}")
        
        return BundleSection(
            data=narrative_data,
            canonical=False,
            priority=3,
            last_changed_at=time.time(),
            version=f"narr_{scope.location_id}"
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary with p95 and detailed stats"""
        summary = {
            'cache_hit_rate': {},
            'avg_fetch_time': {},
            'p95_fetch_time': {},
            'bytes_cached': self.metrics.get('bytes_cached', {})
        }
        
        for section in SECTION_NAMES:
            hits = self.metrics['cache_hits'][section]
            misses = self.metrics['cache_misses'][section]
            total = hits + misses
            
            if total > 0:
                summary['cache_hit_rate'][section] = round(hits / total, 2)
            
            fetch_times = self.metrics['fetch_times'].get(section, [])
            if fetch_times:
                summary['avg_fetch_time'][section] = round(sum(fetch_times) / len(fetch_times), 3)
                # Calculate p95
                sorted_times = sorted(fetch_times)
                p95_idx = int(len(sorted_times) * 0.95)
                summary['p95_fetch_time'][section] = round(sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1], 3)
        
        return summary
    
    def log_metrics_line(self, scene_key: str, packed_context: 'PackedContext'):
        """Log a compact metrics line for monitoring (throttled)"""
        self._metrics_log_counter += 1
        
        # Only log every Nth turn or on significant changes
        if self._metrics_log_counter % self._metrics_log_interval != 0:
            return
        
        metrics = self.get_metrics_summary()
        hit_rates = ' '.join(f"{k[:3]}={v:.2f}" for k, v in metrics['cache_hit_rate'].items())
        p95_times = metrics.get('p95_fetch_time', {})
        max_p95 = max(p95_times.values()) if p95_times else 0
        
        logger.info(f"ctx: key={scene_key[:8]} canon={len(packed_context.canonical)} "
                   f"opt={len(packed_context.optional)} sum={len(packed_context.summarized)} "
                   f"tok={packed_context.tokens_used} p95_fetch={max_p95*1000:.0f}ms "
                   f"hit[{hit_rates}]")

# ===== Main NyxContext Class =====

@dataclass
class NyxContext:
    """Enhanced NyxContext with optimized parallel context assembly"""
    
    # ────────── REQUIRED (no defaults) ──────────
    user_id: int
    conversation_id: int

    # ────────── SUB-SYSTEM HANDLES ──────────
    memory_orchestrator: Optional[MemoryOrchestrator] = None
    user_model: Optional[UserModelManager] = None
    task_integration: Optional[NyxTaskIntegration] = None
    response_filter: Optional[ResponseFilter] = None
    emotional_core: Optional[EmotionalCore] = None
    npc_orchestrator: Optional[NPCOrchestrator] = None
    conflict_synthesizer: Optional[ConflictSynthesizer] = None
    lore_orchestrator: Optional["LoreOrchestrator"] = None
    world_orchestrator: Optional[WorldOrchestrator] = None
    slice_of_life_narrator: Optional[Any] = None  # SliceOfLifeNarrator
    
    # ────────── CONTEXT BROKER (NEW) ──────────
    context_broker: Optional[ContextBroker] = None
    
    # ────────── CURRENT STATE ──────────
    current_context: Dict[str, Any] = field(default_factory=dict)
    scenario_state: Dict[str, Any] = field(default_factory=dict)
    current_location: Optional[str] = None
    current_world_state: Optional[Any] = None
    last_packed_context: Optional['PackedContext'] = None
    config: Optional[Any] = None  # For model context budget
    last_user_input: Optional[str] = None
    governance: Optional[Any] = field(default=None, init=False, repr=False)
    
    # ────────── NPC-SPECIFIC STATE ──────────
    npc_snapshots: Dict[int, NPCSnapshot] = field(default_factory=dict)
    current_scene_npcs: List[int] = field(default_factory=list)
    npc_perceptions: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    npc_decisions: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # ────────── MEMORY STATE ──────────
    recent_memories: Dict[str, List[Any]] = field(default_factory=dict)
    memory_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # ────────── CONFLICT STATE ──────────
    active_conflicts: List[int] = field(default_factory=list)
    conflict_states: Dict[int, str] = field(default_factory=dict)
    conflict_tensions: Dict[str, float] = field(default_factory=dict)
    
    # ────────── LORE STATE ──────────
    world_lore: Dict[str, Any] = field(default_factory=dict)
    active_religions: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    active_nations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # ────────── PERFORMANCE METRICS ──────────
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "response_times": [],
        "memory_queries": 0,
        "npc_updates": 0,
        "world_director_syncs": 0,
        "narrator_syncs": 0,
        "lore_syncs": 0,
        "context_assembly_times": []
    })
    
    # ────────── FEATURE FLAGS ──────────
    enable_parallel_fetch: bool = True
    enable_smart_caching: bool = True
    enable_background_workers: bool = True
    max_parallel_tasks: int = 6
    
    # ────────── TASK TRACKING ──────────
    last_task_runs: Dict[str, datetime] = field(default_factory=dict)
    task_intervals: Dict[str, float] = field(default_factory=lambda: {
        "npc_perception_update": 30,
        "memory_consolidation": 3600,
        "conflict_tension_calculation": 120,
        "lore_evolution": 1800,
        "world_sync": 30,
        "narrator_sync": 30
    })
    
    # ────────── ERROR LOGGING ──────────
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)

    _init_tasks: Dict[str, asyncio.Task[Any]] = field(default_factory=dict, init=False, repr=False)
    _init_failures: Dict[str, BaseException] = field(default_factory=dict, init=False, repr=False)
    _is_initialized: bool = field(default=False, init=False)
    _init_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _init_mode: str = field(default="none", init=False, repr=False)
    background_tasks: Set[asyncio.Task] = field(default_factory=set)
    _governance_pending_minimal: bool = field(default=False, init=False, repr=False)
    _tables_available: Dict[str, bool] = field(default_factory=lambda: {"scenario_states": True})

    def __post_init__(self):
        if self.scenario_state is None:
            self.scenario_state = {}

        if self._tables_available is None or not isinstance(self._tables_available, dict):
            self._tables_available = {"scenario_states": True}
        else:
            self._tables_available.setdefault("scenario_states", True)
    
    async def initialize(self, *, warm_minimal: bool = False):
        """Initialize orchestrators lazily.

        Args:
            warm_minimal: When True, skip heavy orchestrator bootstraps and only
                perform minimal wiring (governance, location hydration, context
                broker setup). Passing ``False`` performs the full bootstrap.
        """

        target_mode = "minimal" if warm_minimal else "full"
        if self._init_mode == "full" or self._init_mode == target_mode:
            return

        async with self._init_lock:
            if self._init_mode == "full" or self._init_mode == target_mode:
                return

            upgrade_from_minimal = self._init_mode == "minimal" and not warm_minimal

            mode_label = (
                "minimal"
                if warm_minimal and not upgrade_from_minimal
                else "full (upgrade)" if upgrade_from_minimal else "full"
            )
            logger.info(
                "[CONTEXT_INIT] Starting %s initialization for user %s, conversation %s...",
                mode_label,
                self.user_id,
                self.conversation_id,
            )
            init_start_time = time.time()

            # Try to load config if available
            try:
                from .config import Config

                self.config = Config
            except ImportError:
                pass

            user_key = str(self.user_id)
            conversation_key = str(self.conversation_id)
            snapshot = _SNAPSHOT_STORE.get(user_key, conversation_key)
            if not snapshot:
                canonical_snapshot = await fetch_canonical_snapshot(
                    self.user_id,
                    self.conversation_id,
                )
                if canonical_snapshot:
                    _SNAPSHOT_STORE.put(
                        user_key,
                        conversation_key,
                        canonical_snapshot,
                    )

            # Helper function to wrap and time each initialization task
            async def timed_init(name: str, coro: Any) -> None:
                t0 = time.time()
                logger.info("[CONTEXT_INIT] Starting sub-task '%s'...", name)
                try:
                    await coro
                    logger.info(
                        "[CONTEXT_INIT] ✔ Sub-task '%s' finished in %.3fs",
                        name,
                        time.time() - t0,
                    )
                except Exception as exc:
                    logger.error(
                        "[CONTEXT_INIT] ✖ Sub-task '%s' failed after %.3fs: %s",
                        name,
                        time.time() - t0,
                        exc,
                        exc_info=True,
                    )
                    raise

            self._init_tasks.clear()

            def _start_task(key: str, label: str, coro: Any) -> None:
                task = asyncio.create_task(
                    timed_init(label, coro),
                    name=f"nyx_init_{key}",
                )
                self._init_failures.pop(key, None)
                self._init_tasks[key] = task
                task.add_done_callback(
                    lambda t, task_key=key: self._on_init_task_done(task_key, t)
                )

            if not warm_minimal:
                _start_task(
                    "memory",
                    "Memory",
                    self._init_memory_orchestrator(warm_minimal=False),
                )
                _start_task(
                    "lore",
                    "Lore",
                    self._init_lore_orchestrator(warm_minimal=False),
                )
                _start_task(
                    "npc",
                    "NPC",
                    self._init_npc_orchestrator(warm_minimal=False),
                )
                _start_task(
                    "conflict",
                    "Conflict",
                    self._init_conflict_synthesizer(warm_minimal=False),
                )

                if WORLD_SIMULATION_AVAILABLE:
                    _start_task(
                        "world",
                        "World",
                        self._init_world_systems(warm_minimal=False),
                    )
            else:
                logger.info(
                    "[CONTEXT_INIT] Minimal warm requested; skipping heavy orchestrator bootstraps"
                )

            await self._hydrate_location_from_db()

            if not self.current_location:
                try:
                    fallback_context = await get_comprehensive_context(
                        self.user_id,
                        self.conversation_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch fallback context for user_id=%s conversation_id=%s",
                        self.user_id,
                        self.conversation_id,
                        exc_info=True,
                    )
                else:
                    if isinstance(fallback_context, dict):
                        previous_location_id = self.current_context.get("location_id")
                        self.current_context.update(fallback_context)
                        await self._refresh_location_from_context(
                            previous_location_id=previous_location_id
                        )

                        normalized_location = self._normalize_location_value(
                            self.current_location
                        )
                        if normalized_location:
                            self.current_location = normalized_location
                            try:
                                user_key = str(self.user_id)
                                conversation_key = str(self.conversation_id)
                                snapshot = _SNAPSHOT_STORE.get(
                                    user_key, conversation_key
                                )
                                snapshot["location_name"] = normalized_location
                                snapshot.setdefault("scene_id", normalized_location)
                                _SNAPSHOT_STORE.put(
                                    user_key, conversation_key, snapshot
                                )
                            except Exception as snapshot_exc:
                                logger.debug(
                                    "Snapshot store seed failed after fallback context: %s",
                                    snapshot_exc,
                                )

            self.context_broker = ContextBroker(self)
            _start_task(
                "context_broker",
                "ContextBroker",
                self.context_broker.initialize(),
            )

            governance_ready = False
            if warm_minimal or upgrade_from_minimal:
                governance_ready = await self._ensure_governance_ready(
                    minimal_warm=warm_minimal and not upgrade_from_minimal
                )
                if governance_ready:
                    logger.debug(
                        "Governance wiring completed during %s initialization for user %s conversation %s",
                        target_mode,
                        self.user_id,
                        self.conversation_id,
                    )

            self._is_initialized = True
            self._init_mode = target_mode if not upgrade_from_minimal else "full"

            final_label = (
                "minimal"
                if target_mode == "minimal" and not upgrade_from_minimal
                else "full (upgrade)" if upgrade_from_minimal else "full"
            )
            logger.info(
                "NyxContext %s initialization finished for user %s, conversation %s",
                final_label,
                self.user_id,
                self.conversation_id,
            )
            logger.info(
                "[CONTEXT_INIT] Total initialization took %.3fs",
                time.time() - init_start_time,
            )

    async def _ensure_governance_ready(self, *, minimal_warm: bool = False) -> bool:
        """Best-effort governance wiring for minimal warm flows."""

        if getattr(self, "governance", None) is not None:
            self._governance_pending_minimal = False
            return True

        if minimal_warm:
            self._governance_pending_minimal = True
            logger.debug(
                "Governance wiring deferred during minimal warm for user %s conversation %s",
                self.user_id,
                self.conversation_id,
            )
            return False

        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
        except Exception as exc:  # pragma: no cover - governance is optional in tests
            logger.debug(
                "Governance wiring skipped for user %s conversation %s: %s",
                self.user_id,
                self.conversation_id,
                exc,
                exc_info=True,
            )
            return False

        self.governance = governance
        self._governance_pending_minimal = False

        attach_hook = getattr(governance, "attach_context", None)
        if attach_hook is not None:
            try:
                maybe = attach_hook(self)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception as exc:  # pragma: no cover - best effort hook
                logger.debug(
                    "Governance attach_context failed for user %s conversation %s: %s",
                    self.user_id,
                    self.conversation_id,
                    exc,
                    exc_info=True,
                )

        return self.governance is not None

    def _on_init_task_done(self, name: str, task: asyncio.Task[Any]) -> None:
        """Ensure initialization task exceptions are surfaced and recorded."""

        if task.cancelled():
            logger.warning("Initialization task '%s' was cancelled", name)
            self._init_failures[name] = asyncio.CancelledError(
                f"Initialization task '{name}' was cancelled"
            )
            return

        try:
            task.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Initialization task '%s' failed post-completion: %s",
                name,
                exc,
                exc_info=True,
            )
            self._init_failures[name] = exc
            try:
                self.log_error(exc, {"task": f"init_{name}"})
            except Exception:
                logger.debug("Failed to record init task error for '%s'", name, exc_info=True)
        else:
            self._init_failures.pop(name, None)

    def get_init_task(self, name: str) -> Optional[asyncio.Task[Any]]:
        """Return the initialization task for a subsystem, if any."""

        return self._init_tasks.get(name)

    def is_orchestrator_ready(self, name: str) -> bool:
        """Return True if the orchestrator finished initializing without errors."""

        task = self._init_tasks.get(name)
        if task is None:
            return False

        if name in self._init_failures:
            return False

        if task.cancelled():
            return False

        if not task.done():
            return False

        exc = task.exception()
        if exc is not None:
            self._init_failures[name] = exc
            return False

        return True

    async def await_orchestrator(self, name: str) -> bool:
        """Await the initialization task for a subsystem and report readiness."""

        task = self._init_tasks.get(name)
        if not task:
            return False

        if name in self._init_failures:
            return False

        if task.done():
            if task.cancelled():
                return False
            exc = task.exception()
            if exc is not None:
                self._init_failures[name] = exc
                return False
            self._init_failures.pop(name, None)
            return True

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                self._init_failures[name] = exc
                return False
            raise
        except Exception as exc:
            self._init_failures[name] = exc
            return False

        if task.cancelled():
            self._init_failures[name] = asyncio.CancelledError(
                f"Initialization task '{name}' was cancelled"
            )
            return False

        exc = task.exception()
        if exc is not None:
            self._init_failures[name] = exc
            return False

        self._init_failures.pop(name, None)
        return True

    async def _hydrate_location_from_db(self) -> None:
        """Preload the current location from the CurrentRoleplay table."""
        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                    LIMIT 1
                    """,
                    self.user_id,
                    self.conversation_id,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to hydrate current location for user_id=%s conversation_id=%s: %s",
                self.user_id,
                self.conversation_id,
                exc,
            )
            return

        if not row:
            logger.info(
                "No CurrentLocation row found for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )
            return

        try:
            raw_value = row["value"]
        except (KeyError, IndexError):
            raw_value = None

        normalized_location = self._normalize_location_value(raw_value)
        if not normalized_location:
            logger.info(
                "CurrentLocation normalization failed for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )
            return

        self.current_location = normalized_location
        self.current_context["location"] = normalized_location
        self.current_context.setdefault("location_id", normalized_location)
        self.current_context.setdefault("location_name", normalized_location)
        self.current_context.setdefault("current_location", normalized_location)

        snapshot_cache_hit = False
        try:
            user_key = str(self.user_id)
            conversation_key = str(self.conversation_id)
            snapshot = _SNAPSHOT_STORE.get(user_key, conversation_key)
            snapshot_cache_hit = bool(snapshot)
            snapshot["location_name"] = normalized_location
            snapshot.setdefault("scene_id", str(normalized_location))
            _SNAPSHOT_STORE.put(user_key, conversation_key, snapshot)
        except Exception as exc:  # pragma: no cover - best effort cache seed
            logger.debug("Snapshot store seed failed: %s", exc)

        logger.info(
            (
                "Hydrated CurrentLocation for user_id=%s conversation_id=%s "
                "snapshot_cache_hit=%s"
            ),
            self.user_id,
            self.conversation_id,
            snapshot_cache_hit,
        )

    async def warm_minimal_context(
        self,
        *,
        prime_vector_store: bool = False,
        minimal_warm: bool = True,
    ) -> Dict[str, Any]:
        """Perform the minimal warm path without full bundle fetches."""

        await self.initialize(warm_minimal=True)
        await self.await_orchestrator("context_broker")

        governance_ready = await self._ensure_governance_ready(
            minimal_warm=minimal_warm
        )

        snapshot_seeded = False
        try:
            user_key = str(self.user_id)
            conversation_key = str(self.conversation_id)
            snapshot_seeded = bool(
                _SNAPSHOT_STORE.get(user_key, conversation_key)
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Snapshot inspection during minimal warm failed for user %s conversation %s: %s",
                self.user_id,
                self.conversation_id,
                exc,
                exc_info=True,
            )

        if prime_vector_store:
            try:
                await self._init_memory_orchestrator(warm_minimal=False)
            except Exception as exc:  # pragma: no cover - optional prime
                logger.warning(
                    "Vector store prime during minimal warm failed for user %s conversation %s: %s",
                    self.user_id,
                    self.conversation_id,
                    exc,
                    exc_info=True,
                )

        return {
            "status": "warmed",
            "mode": "minimal",
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "location": self.current_location,
            "snapshot_seeded": snapshot_seeded,
            "governance_ready": governance_ready,
        }

    async def _persist_location_to_db(self, canonical_location: str) -> None:
        """Persist the canonical location to the backing CurrentRoleplay row."""
        fallback_required = False

        try:
            canonical_ctx = ensure_canonical_context(
                {"user_id": self.user_id, "conversation_id": self.conversation_id}
            )
            async with get_db_connection_context() as conn:
                try:
                    await canon.update_current_roleplay(
                        canonical_ctx,
                        conn,
                        "CurrentLocation",
                        canonical_location,
                    )
                except Exception as canon_exc:
                    fallback_required = True
                    logger.warning(
                        "Primary CurrentLocation persist failed for user_id=%s conversation_id=%s: %s",
                        self.user_id,
                        self.conversation_id,
                        canon_exc,
                        exc_info=True,
                    )
                else:
                    logger.info(
                        "CurrentLocation persisted for user_id=%s conversation_id=%s location=%s (fallback=%s)",
                        self.user_id,
                        self.conversation_id,
                        canonical_location,
                        False,
                    )
                    return
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.warning(
                "Failed to persist CurrentLocation for user_id=%s conversation_id=%s: %s",
                self.user_id,
                self.conversation_id,
                exc,
                exc_info=True,
            )
            return

        if not fallback_required:
            return

        try:
            async with get_db_connection_context() as fallback_conn:
                await fallback_conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    self.user_id,
                    self.conversation_id,
                    "CurrentLocation",
                    canonical_location,
                )
                logger.info(
                    "CurrentLocation persisted for user_id=%s conversation_id=%s location=%s (fallback=%s)",
                    self.user_id,
                    self.conversation_id,
                    canonical_location,
                    True,
                )
        except Exception as fallback_exc:  # pragma: no cover - log and continue
            logger.error(
                "Fallback CurrentLocation persist failed for user_id=%s conversation_id=%s: %s",
                self.user_id,
                self.conversation_id,
                fallback_exc,
                exc_info=True,
            )

    def _track_background_task(
        self,
        task: asyncio.Task,
        *,
        task_name: str,
        task_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track background tasks and surface failures via logging."""

        if task_details is None:
            task_details = {}

        self.background_tasks.add(task)

        def _handle_completion(completed: asyncio.Task) -> None:
            self.background_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                logger.info(
                    "Background task '%s' cancelled", task_name,
                    extra={"task_details": task_details},
                )
            except Exception:
                logger.exception(
                    "Background task '%s' failed", task_name,
                    extra={"task_details": task_details},
                )

        task.add_done_callback(_handle_completion)

    @staticmethod
    def _is_placeholder_location_token(token: str) -> bool:
        if not isinstance(token, str):
            return False

        normalized = token.strip()
        if not normalized:
            return True

        return normalized.casefold() in _PLACEHOLDER_LOCATION_TOKENS

    @staticmethod
    def _normalize_location_value(value: Any) -> Optional[str]:
        """Normalize location values stored as TEXT/JSON into a simple identifier."""
        if value is None:
            return None

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                value = value.decode("utf-8", errors="ignore")

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                parsed = candidate

            if isinstance(parsed, str):
                parsed = parsed.strip()
                if NyxContext._is_placeholder_location_token(parsed):
                    return None
                return parsed or None
            if isinstance(parsed, dict):
                for key in ("name", "location", "location_name", "id", "scene_id"):
                    token = parsed.get(key)
                    if isinstance(token, str):
                        token_str = token.strip()
                        if NyxContext._is_placeholder_location_token(token_str):
                            continue
                        if token_str:
                            return token_str
                    if isinstance(token, (int, float)):
                        token_str = str(token).strip()
                        if token_str:
                            return token_str
                return None
            if isinstance(parsed, (int, float)):
                token = str(parsed).strip()
                if NyxContext._is_placeholder_location_token(token):
                    return None
                return token or None
            return None

        if isinstance(value, (int, float)):
            token = str(value).strip()
            if NyxContext._is_placeholder_location_token(token):
                return None
            return token or None

        if isinstance(value, dict):
            for key in ("name", "location", "location_name", "id", "scene_id"):
                token = value.get(key)
                if isinstance(token, str):
                    token_str = token.strip()
                    if NyxContext._is_placeholder_location_token(token_str):
                        continue
                    if token_str:
                        return token_str
                if isinstance(token, (int, float)):
                    token_str = str(token).strip()
                    if token_str:
                        return token_str

        return None

    def _extract_canonical_location(self, mapping: Optional[Dict[str, Any]]) -> Optional[str]:
        """Pull the best-effort canonical location string from a context mapping."""
        if not isinstance(mapping, dict):
            return None

        candidate_keys = (
            "location_name",
            "location",
            "currentLocation",
            "current_location",
            "location_id",
        )

        for key in candidate_keys:
            if key in mapping:
                normalized = self._normalize_location_value(mapping.get(key))
                if normalized:
                    return normalized

        for nested_key in ("currentRoleplay", "current_roleplay"):
            nested = mapping.get(nested_key)
            if not isinstance(nested, dict):
                continue
            for location_key in ("CurrentLocation", "currentLocation", "current_location"):
                normalized = self._normalize_location_value(nested.get(location_key))
                if normalized:
                    return normalized

        aggregator_data = mapping.get("aggregator_data")
        if isinstance(aggregator_data, dict):
            for key in candidate_keys:
                if key in aggregator_data:
                    normalized = self._normalize_location_value(
                        aggregator_data.get(key)
                    )
                    if normalized:
                        return normalized

            for nested_key in ("currentRoleplay", "current_roleplay"):
                nested = aggregator_data.get(nested_key)
                if not isinstance(nested, dict):
                    continue
                for location_key in (
                    "CurrentLocation",
                    "currentLocation",
                    "current_location",
                ):
                    normalized = self._normalize_location_value(
                        nested.get(location_key)
                    )
                    if normalized:
                        return normalized

        return None

    async def _refresh_location_from_context(
        self,
        previous_location_id: Optional[str] = None,
    ) -> None:
        """Update current location tracking based on the merged context payload."""
        canonical_location = self._extract_canonical_location(self.current_context)
        if not canonical_location:
            return

        if self._is_placeholder_location_token(canonical_location):
            return

        previous_location = self.current_location

        normalized_location_id = self._normalize_location_value(
            self.current_context.get("location_id")
        )
        if normalized_location_id and self._is_placeholder_location_token(normalized_location_id):
            normalized_location_id = None
        if not normalized_location_id:
            for nested_key in ("currentRoleplay", "current_roleplay"):
                nested = self.current_context.get(nested_key)
                if not isinstance(nested, dict):
                    continue
                current_location = (
                    nested.get("CurrentLocation")
                    or nested.get("currentLocation")
                    or nested.get("current_location")
                )
                if isinstance(current_location, dict):
                    normalized_location_id = self._normalize_location_value(
                        current_location.get("id")
                        or current_location.get("location_id")
                        or current_location.get("location")
                    )
                else:
                    normalized_location_id = self._normalize_location_value(current_location)
                if normalized_location_id:
                    break

        if not normalized_location_id:
            aggregator_data = self.current_context.get("aggregator_data")
            if isinstance(aggregator_data, dict):
                for nested_key in ("currentRoleplay", "current_roleplay"):
                    nested = aggregator_data.get(nested_key)
                    if not isinstance(nested, dict):
                        continue
                    current_location = (
                        nested.get("CurrentLocation")
                        or nested.get("currentLocation")
                        or nested.get("current_location")
                    )
                    if isinstance(current_location, dict):
                        normalized_location_id = self._normalize_location_value(
                            current_location.get("id")
                            or current_location.get("location_id")
                            or current_location.get("location")
                        )
                        if not normalized_location_id:
                            normalized_location_id = self._normalize_location_value(
                                current_location
                            )
                    else:
                        normalized_location_id = self._normalize_location_value(
                            current_location
                        )
                    if normalized_location_id:
                        break

        if not normalized_location_id and previous_location_id:
            normalized_location_id = self._normalize_location_value(previous_location_id)

        if not normalized_location_id:
            normalized_location_id = canonical_location

        self.current_location = canonical_location
        self.current_context["location"] = canonical_location
        self.current_context["location_name"] = canonical_location
        self.current_context["current_location"] = canonical_location
        self.current_context["location_id"] = normalized_location_id
        if "currentLocation" in self.current_context:
            self.current_context["currentLocation"] = canonical_location

        if canonical_location != previous_location:
            try:
                user_key = str(self.user_id)
                conversation_key = str(self.conversation_id)
                snapshot = _SNAPSHOT_STORE.get(user_key, conversation_key)
                snapshot["location_name"] = canonical_location
                snapshot.setdefault("scene_id", str(canonical_location))
                _SNAPSHOT_STORE.put(user_key, conversation_key, snapshot)
            except Exception as exc:  # pragma: no cover - best effort cache seed
                logger.debug("Snapshot store update failed: %s", exc)

            task = asyncio.create_task(
                self._persist_location_to_db(canonical_location)
            )
            try:
                task.set_name(
                    f"nyx-persist-location:{self.user_id}:{self.conversation_id}"
                )
            except AttributeError:
                pass
            self._track_background_task(
                task,
                task_name="persist_location",
                task_details={"location": canonical_location},
            )

    async def _init_memory_orchestrator(self, *, warm_minimal: bool = False):
        """Initialize memory orchestrator"""
        if warm_minimal:
            logger.info("Skipping memory orchestrator bootstrap during minimal warm phase")
            return
        try:
            self.memory_orchestrator = await get_memory_orchestrator(
                self.user_id,
                self.conversation_id
            )
            logger.info("Memory Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Memory Orchestrator: {e}")

    async def _init_lore_orchestrator(self, *, warm_minimal: bool = False):
        """Initialize lore orchestrator"""
        if warm_minimal:
            logger.info("Skipping lore orchestrator bootstrap during minimal warm phase")
            return
        try:
            from lore.lore_orchestrator import OrchestratorConfig, get_lore_orchestrator

            lore_config = OrchestratorConfig(
                enable_governance=True,
                enable_cache=True,
                auto_initialize=True
            )
            self.lore_orchestrator = await get_lore_orchestrator(
                self.user_id,
                self.conversation_id,
                lore_config
            )
            logger.info("Lore Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Lore Orchestrator: {e}")

    async def _init_npc_orchestrator(self, *, warm_minimal: bool = False):
        if warm_minimal:
            logger.info("Skipping NPC orchestrator bootstrap during minimal warm phase")
            return
        try:
            self.npc_orchestrator = NPCOrchestrator(
                self.user_id,
                self.conversation_id,
                config={"enable_canon": True}
            )
            await self.npc_orchestrator.initialize()
            logger.info("NPC Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NPC Orchestrator: {e}")

    async def _init_conflict_synthesizer(self, *, warm_minimal: bool = False):
        """Initialize conflict synthesizer"""
        if warm_minimal:
            logger.info("Skipping conflict synthesizer bootstrap during minimal warm phase")
            return
        try:
            self.conflict_synthesizer = await get_conflict_synthesizer(
                self.user_id,
                self.conversation_id
            )
            logger.info("Conflict Synthesizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Conflict Synthesizer: {e}")

    async def _init_world_systems(self, *, warm_minimal: bool = False):
        """Initialize world orchestrator and slice-of-life narrator."""

        if warm_minimal:
            logger.info("Skipping world systems bootstrap during minimal warm phase")
            return

        try:
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator

            cache_key = f"ctx:warmed:{self.user_id}:{self.conversation_id}"
            warmed = False
            redis_client = None

            try:
                redis_client = get_redis_client()
            except Exception as redis_exc:
                logger.debug(
                    "World system warm check failed to get Redis client: %s",
                    redis_exc,
                    exc_info=True,
                )

            if redis_client is not None:
                try:
                    warmed = bool(await asyncio.to_thread(redis_client.exists, cache_key))
                except Exception as redis_exc:
                    logger.debug(
                        "World system warm check failed for key %s: %s",
                        cache_key,
                        redis_exc,
                        exc_info=True,
                    )

            if warmed:
                logger.info(
                    "World systems warm cache hit for user_id=%s conversation_id=%s",
                    self.user_id,
                    self.conversation_id,
                )

            if self.world_orchestrator is not None:
                try:
                    await self.world_orchestrator.dispose()
                except Exception:
                    logger.debug("Previous world orchestrator dispose failed", exc_info=True)

            orchestrator = WorldOrchestrator(self.user_id, self.conversation_id)
            await orchestrator.initialize(warmed=warmed)
            self.world_orchestrator = orchestrator

            self.slice_of_life_narrator = SliceOfLifeNarrator(self.user_id, self.conversation_id)
            await self.slice_of_life_narrator.initialize()

            logger.info("World systems initialized")
        except Exception as e:
            logger.error(f"Failed to initialize world systems: {e}")

    async def handle_day_transition(self, new_day: int):
        """Handle game day transitions: notify conflicts & clear scene caches."""
        logger.info(f"NyxContext handling day transition to {new_day}")
        await self.await_orchestrator("conflict")
        await self.await_orchestrator("context_broker")
        try:
            if self.conflict_synthesizer and hasattr(self.conflict_synthesizer, 'handle_day_transition'):
                await self.conflict_synthesizer.handle_day_transition(new_day)
        except Exception as e:
            logger.warning(f"Conflict day transition hook failed: {e}")

        try:
            if self.context_broker:
                # reset the LRU and let Redis keys expire naturally
                self.context_broker.bundle_cache = LRUCache(capacity=128)
        except Exception as e:
            logger.warning(f"Failed to clear broker caches: {e}")

        logger.info("Day transition complete")

    async def _fetch_enhanced_conflict_summary(self, bundle: ContextBundle) -> Optional[Dict[str, Any]]:
        scene_scope = getattr(bundle, 'scene_scope', None)
        if scene_scope is None:
            return None

        present_npcs: List[str] = []
        npc_ids = getattr(scene_scope, 'npc_ids', None)
        if npc_ids:
            try:
                present_npcs = [str(int(n)) for n in npc_ids]
            except Exception:
                present_npcs = [str(n) for n in npc_ids]
            present_npcs.sort()

        topics: List[str] = []
        raw_topics = getattr(scene_scope, 'topics', None)
        if raw_topics:
            topics = sorted(str(t) for t in raw_topics if t is not None)

        metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
        location = (
            metadata.get('location')
            or metadata.get('location_id')
            or getattr(scene_scope, 'location_name', None)
            or getattr(scene_scope, 'location_id', None)
            or self.current_context.get('location_name')
            or self.current_context.get('location_id')
        )
        scene_type = (
            metadata.get('scene_type')
            or self.current_context.get('scene_type')
            or self.current_context.get('scene_descriptor')
        )

        normalized = {
            'location': location or 'unknown',
            'scene_type': scene_type or 'unknown',
            'present_npcs': present_npcs,
            'topics': topics,
        }
        scope_key = _compute_enhanced_scope_key(normalized)
        if not scope_key:
            return None

        summary = get_cached_tension_result(self.user_id, self.conversation_id, scope_key)

        if not summary:
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT summary
                        FROM conflict_scene_tension_summaries
                        WHERE user_id = $1 AND conversation_id = $2 AND scope_key = $3
                        """,
                        self.user_id,
                        self.conversation_id,
                        scope_key,
                    )
            except Exception as exc:
                logger.debug("Enhanced conflict summary DB fetch failed: %s", exc)
                row = None

            if row:
                summary = row.get('summary')
                if isinstance(summary, str):
                    try:
                        summary = json_loads(summary)
                    except Exception:
                        summary = None

        if isinstance(summary, dict):
            summary.setdefault('source', summary.get('source', 'cached'))
            summary.setdefault('manifestation', summary.get('manifestation', []))
            summary.setdefault('tensions', summary.get('tensions', []))
            summary.setdefault('suggested_type', summary.get('suggested_type'))
            summary.setdefault('should_generate_conflict', summary.get('should_generate_conflict', False))
            return summary
        return None

    async def _inject_enhanced_conflict_hints(self, bundle: ContextBundle) -> Optional[Dict[str, Any]]:
        try:
            summary = await self._fetch_enhanced_conflict_summary(bundle)
        except Exception as exc:
            logger.debug("Failed to fetch enhanced conflict summary: %s", exc)
            return None

        if not summary:
            return None

        hints = {
            'source': summary.get('source', 'cached'),
            'cached_at': summary.get('cached_at'),
            'should_generate_conflict': summary.get('should_generate_conflict', False),
            'suggested_type': summary.get('suggested_type'),
            'tensions': summary.get('tensions', []),
        }
        manifestations = list(summary.get('manifestation', []) or [])
        if manifestations:
            hints['manifestation'] = manifestations
        if isinstance(summary.get('context'), dict):
            hints['context'] = summary['context']
        if summary.get('choices'):
            hints['choices'] = summary['choices']

        bundle.metadata.setdefault('conflict_hints', {})['enhanced_integration'] = hints
        self.current_context['enhanced_conflict_hints'] = hints
        return summary

    async def build_context_for_input(self, user_input: str, context_data: Dict[str, Any] = None) -> PackedContext:
        """Main entry point: build optimized context for user input"""
        start_time = time.time()

        await self.await_orchestrator("context_broker")
        if not self.context_broker:
            raise RuntimeError("Context broker failed to initialize")

        # Merge provided context
        previous_location_id = self.current_context.get("location_id")
        if context_data:
            self.current_context.update(context_data)

        self.last_user_input = user_input
        self.current_context["last_user_input"] = user_input

        await self._refresh_location_from_context(
            previous_location_id=previous_location_id
        )

        # Compute scene scope
        scene_scope = await self.context_broker.compute_scene_scope(
            user_input,
            self.current_context
        )
        
        # Log scope for debugging
        logger.debug(f"Scene scope: location={scene_scope.location_id}, "
                    f"npcs={len(scene_scope.npc_ids)}, "
                    f"topics={len(scene_scope.topics)}")
        
        # Load or fetch the context bundle
        bundle = await self.context_broker.load_or_fetch_bundle(scene_scope)

        def _extract_per_intent(source: Any, depth: int = 0) -> Optional[List[Dict[str, Any]]]:
            if source is None or depth > 4:
                return None

            if isinstance(source, list):
                if source and all(isinstance(item, dict) for item in source):
                    return source
                return None

            if isinstance(source, dict):
                per_intent_value = source.get("per_intent")
                if isinstance(per_intent_value, list):
                    return per_intent_value

                for nested_key in (
                    "feasibility",
                    "fast_feasibility",
                    "feasibility_payload",
                    "payload",
                    "result",
                    "data",
                    "meta",
                ):
                    nested = source.get(nested_key)
                    extracted = _extract_per_intent(nested, depth + 1)
                    if extracted:
                        return extracted

            return None

        per_intent_payload: Optional[List[Dict[str, Any]]] = None
        candidate_sources: List[Any] = [
            self.current_context.get("feasibility"),
            self.current_context.get("feasibility_payload"),
            self.current_context.get("fast_feasibility"),
        ]

        processing_metadata = self.current_context.get("processing_metadata")
        if isinstance(processing_metadata, dict):
            candidate_sources.append(processing_metadata.get("feasibility"))

        aggregator_data = self.current_context.get("aggregator_data")
        if isinstance(aggregator_data, dict):
            candidate_sources.extend(list(aggregator_data.values())[:5])

        for attr in ("last_feasibility", "last_feasibility_result", "fast_feasibility"):
            candidate_sources.append(getattr(self, attr, None))

        for candidate in candidate_sources:
            per_intent_payload = _extract_per_intent(candidate)
            if per_intent_payload:
                break

        lore_priority = compute_lore_priority(user_input, per_intent_payload, scene_scope)
        if not isinstance(bundle.metadata, dict):
            bundle.metadata = {}
        bundle.metadata["lore_priority"] = lore_priority

        hints_payload = dict(bundle.metadata.get("hints") or {})
        hints_payload["lore_priority"] = lore_priority
        if lore_priority >= 0.75:
            hints_payload["lore_tool_recommended"] = True
            hints_payload["suggested_aspects"] = infer_lore_aspects(user_input, scene_scope)
        else:
            hints_payload.pop("lore_tool_recommended", None)
            hints_payload.pop("suggested_aspects", None)
        bundle.metadata["hints"] = hints_payload
        self.current_context["hints"] = hints_payload

        try:
            logger.info(
                "Nyx lore priority evaluated",
                extra={
                    "user_id": getattr(self, "user_id", None),
                    "conversation_id": getattr(self, "conversation_id", None),
                    "location_id": scene_scope.location_id,
                    "lore_priority": float(lore_priority),
                    "lore_tool_recommended": bool(hints_payload.get("lore_tool_recommended")),
                    "suggested_aspects": hints_payload.get("suggested_aspects") or [],
                },
            )
        except Exception:
            # Metrics should never break the turn
            logger.debug("Failed to log lore priority metrics", exc_info=True)

        # Propagate recent conversation turns into the bundle metadata
        raw_recent = self.current_context.get("recent_turns")
        if raw_recent is None:
            raw_recent = self.current_context.get("recent_interactions")

        canonical_recent: List[Dict[str, Any]] = []
        if isinstance(raw_recent, list):
            for entry in raw_recent:
                if not isinstance(entry, dict):
                    continue
                sender = entry.get("sender")
                content = entry.get("content")
                if sender is None and content is None:
                    continue
                turn: Dict[str, Any] = {}
                if sender is not None:
                    turn["sender"] = sender
                if content is not None:
                    turn["content"] = content
                canonical_recent.append(turn)

        self.current_context["recent_turns"] = canonical_recent
        bundle.metadata["recent_interactions"] = list(canonical_recent)

        narrative_data = bundle.narrative.data
        if isinstance(narrative_data, dict) and not narrative_data.get("recent"):
            narrative_data["recent"] = list(canonical_recent)

        enhanced_summary = await self._inject_enhanced_conflict_hints(bundle)

        # Determine token budget from config or use default
        token_budget = 8000  # Default
        if hasattr(self, 'config') and hasattr(self.config, 'model_context_budget'):
            # Reserve some tokens for system prompt and response
            reserved_tokens = getattr(self.config, 'reserved_system_tokens', 2000)
            # Guard against negative budget
            token_budget = max(1024, self.config.model_context_budget - reserved_tokens)
        
        # Reserve additional tokens for user_input
        user_input_reserve = 512  # Reserve for user input
        effective_budget = max(512, token_budget - user_input_reserve)
        
        # Pack it for the LLM
        packed = bundle.pack(token_budget=effective_budget)

        # Add user input as canonical (will fit in reserved space)
        packed.add_canonical('user_input', user_input)

        if enhanced_summary:
            enhanced_section = {
                'ambient': list(enhanced_summary.get('manifestation', []) or []),
                'tensions': enhanced_summary.get('tensions', []),
                'suggested_type': enhanced_summary.get('suggested_type'),
                'should_generate_conflict': enhanced_summary.get('should_generate_conflict', False),
            }
            if enhanced_summary.get('choices'):
                enhanced_section['choices'] = enhanced_summary['choices']
            packed.try_add('enhanced_conflict', enhanced_section)

        # Performance tracking
        elapsed = time.time() - start_time
        self.performance_metrics['response_times'].append(elapsed)
        self.performance_metrics['context_assembly_times'].append(elapsed)
        
        # Keep last 100 times for stats
        if len(self.performance_metrics['response_times']) > 100:
            self.performance_metrics['response_times'] = self.performance_metrics['response_times'][-100:]
        
        # Log compact metrics line (throttled)
        scene_key = scene_scope.to_key()
        self.context_broker.log_metrics_line(scene_key, packed)
        
        # Store for potential reuse
        self.context_broker._last_scene_key = scene_key
        self.context_broker._last_packed = packed
        self.context_broker._last_bundle = bundle
        
        return packed
    
    async def calculate_conflict_tensions(self) -> Dict[str, float]:
        """Calculate tensions between entities (adapter for existing code)"""
        await self.await_orchestrator("conflict")
        if not self.conflict_synthesizer:
            return {}
        
        try:
            # Use existing conflict synthesizer methods
            if hasattr(self.conflict_synthesizer, 'calculate_tensions'):
                tensions = await self.conflict_synthesizer.calculate_tensions()
            elif hasattr(self.conflict_synthesizer, 'get_all_tensions'):
                tensions = await self.conflict_synthesizer.get_all_tensions()
            else:
                # Fallback: build from conflict states
                tensions = {}
                for conflict_id in self.active_conflicts:
                    state = await self.conflict_synthesizer.get_conflict_state(conflict_id)
                    if state:
                        tension_data = state.get('subsystem_data', {}).get('tension', {})
                        level = tension_data.get('level', 0.5)
                        tensions[f"conflict_{conflict_id}"] = level
            
            self.conflict_tensions = tensions
            return tensions
            
        except Exception as e:
            logger.error(f"Failed to calculate tensions: {e}")
            return {}
    
    async def analyze_memory_patterns(self, topic: str = None) -> Dict[str, Any]:
        """Analyze memory patterns (adapter method)"""
        await self.await_orchestrator("memory")
        if not self.memory_orchestrator:
            return {'predictions': []}
        
        try:
            if hasattr(self.memory_orchestrator, 'analyze_patterns'):
                return await self.memory_orchestrator.analyze_patterns(
                    topic=topic,
                    time_window_hours=24
                )
            else:
                # Fallback: simple pattern detection
                memories = await self.memory_orchestrator.retrieve_memories(
                    entity_type=EntityType.PLAYER,
                    entity_id=self.user_id,
                    query=topic or "",
                    limit=20
                )
                
                # Extract basic patterns
                patterns = []
                if memories and 'memories' in memories:
                    # Count recurring themes
                    theme_counts = defaultdict(int)
                    for mem in memories['memories']:
                        if 'tags' in mem:
                            for tag in mem['tags']:
                                theme_counts[tag] += 1
                    
                    # Top themes are patterns
                    for theme, count in sorted(theme_counts.items(), 
                                              key=lambda x: x[1], reverse=True)[:3]:
                        patterns.append({
                            'theme': theme,
                            'frequency': count,
                            'prediction': f"Theme '{theme}' is recurring"
                        })
                
                return {'predictions': patterns}
                
        except Exception as e:
            logger.error(f"Failed to analyze memory patterns: {e}")
            return {'predictions': []}
    
    def should_run_task(self, task_name: str) -> bool:
        """Check if a periodic task should run"""
        if task_name not in self.task_intervals:
            return False
        
        interval = self.task_intervals[task_name]
        last_run = self.last_task_runs.get(task_name)
        
        if not last_run:
            return True
        
        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
        return elapsed >= interval
    
    def record_task_run(self, task_name: str):
        """Record that a task has run"""
        self.last_task_runs[task_name] = datetime.now(timezone.utc)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context"""
        error_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(error),
            'type': type(error).__name__,
            'context': context or {}
        }
        
        self.error_log.append(error_entry)
        self.error_counts[type(error).__name__] = self.error_counts.get(type(error).__name__, 0) + 1
        
        # Keep error log bounded
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

    def update_performance(self, metric_name: str, value: Any) -> None:
        """Update a performance metric value"""
        if metric_name in self.performance_metrics:
            # For lists, append the value
            if isinstance(self.performance_metrics[metric_name], list):
                self.performance_metrics[metric_name].append(value)
                # Keep lists bounded to reasonable sizes
                if metric_name == "response_times" and len(self.performance_metrics[metric_name]) > 100:
                    self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]
            # For counters, set the value directly
            else:
                self.performance_metrics[metric_name] = value
        else:
            # If metric doesn't exist, create it
            self.performance_metrics[metric_name] = value

    @property
    def world_director(self) -> Optional["CompleteWorldDirector"]:
        """Compatibility shim exposing the underlying director instance."""

        if self.world_orchestrator:
            return self.world_orchestrator.director
        return None

    # ────────── COMPATIBILITY METHODS ──────────

    def get_comprehensive_context_for_response(self) -> Dict[str, Any]:
        """Get comprehensive context (compatibility with existing code)"""
        return self.current_context.copy()
    
    def get_npc_context_for_response(self) -> Dict[str, Any]:
        """Get NPC context for response"""
        return {
            'npcs': {npc_id: snapshot for npc_id, snapshot in self.npc_snapshots.items()},
            'scene_npcs': self.current_scene_npcs,
            'perceptions': self.npc_perceptions
        }
    
    def get_conflict_context_for_response(self) -> Dict[str, Any]:
        """Get conflict context for response"""
        return {
            'active': self.active_conflicts,
            'states': self.conflict_states,
            'tensions': self.conflict_tensions
        }
    
    def get_lore_context_for_response(self) -> Dict[str, Any]:
        """Get lore context for response"""
        return {
            'world': self.world_lore,
            'religions': list(self.active_religions.values()),
            'nations': list(self.active_nations.values())
        }
