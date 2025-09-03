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
import re
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict, OrderedDict
import redis.asyncio as redis  # Modern redis async client

from logic.conflict_system.conflict_synthesizer import get_synthesizer
from logic.conflict_system.background_processor import get_conflict_scheduler

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

# Schema version for cache invalidation
SCHEMA_VERSION = 3

from db.connection import get_db_connection_context
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.response_filter import ResponseFilter
from nyx.core.emotions.emotional_core import EmotionalCore

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

# Import Lore Orchestrator
from lore.lore_orchestrator import (
    LoreOrchestrator,
    OrchestratorConfig,
    get_lore_orchestrator
)

from .config import Config

try:
    from story_agent.world_simulation_models import (
        CompleteWorldState, WorldState, WorldMood, TimeOfDay,
        ActivityType, PowerDynamicType, PowerExchange,
        WorldTension, RelationshipDynamics, NPCRoutine,
        CurrentTimeData, VitalsData, AddictionCravingData,
        DreamData, RevelationData, ChoiceData, ChoiceProcessingResult,
    )
    from story_agent.world_director_agent import (
        CompleteWorldDirector, WorldDirector,
        CompleteWorldDirectorContext, WorldDirectorContext,
    )
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

    def to_key(self) -> str:
        # Single canonical key path for all systems
        return generate_scene_cache_key(self)

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
            canonical_rules=self.canonical_rules[:3]
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
        packed = PackedContext(token_budget=token_budget)
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

        for name, section in sections:
            if name in added:
                continue
            if section.canonical:
                packed.add_canonical(name, section.data)
            else:
                packed.try_add(name, section.data)

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
            return sum(self._estimate_tokens(v) for v in data.values())
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
                return {
                    'location': data.get('location', {}).get('description', '')[:100],
                    'canonical_rules': data.get('canonical_rules', [])[:3]
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
        
        # Metrics logging throttle
        self._metrics_log_counter = 0
        self._metrics_log_interval = 5  # Log every 5th turn
    
    async def initialize(self):
        """Initialize broker with optional Redis for distributed caching"""
        await self._try_connect_redis()
        await self._build_npc_alias_cache()
        self.conflict_synthesizer = await get_synthesizer(
            self.ctx.user_id,
            self.ctx.conversation_id
        )
        logger.info("ContextBroker initialized with optimized conflict system")

    async def expand_bundle_section(self, bundle: ContextBundle, section: str) -> None:
        name_map = {'memory': 'memories', 'conflict': 'conflicts'}
        section = name_map.get(section, section)
        new_section = await self._fetch_section(section, bundle.scene_scope)
        setattr(bundle, section, new_section)
    
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
                redis_url,  # Now uses the environment variable
                encoding=None,
                decode_responses=False,
                socket_connect_timeout=2.0
            )
    
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
    
    async def compute_scene_scope(self, user_input: str, current_state: Dict[str, Any]) -> SceneScope:
        """Analyze input and state to determine relevant scope (no orchestrator calls)"""
        scope = SceneScope()
        
        # Extract location (keep both ID and name)
        scope.location_id = current_state.get('location_id') or current_state.get('location')
        scope.location_name = current_state.get('location_name')
        if not scope.location_id:
            scope.location_id = self.ctx.current_location
        
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
            
            # No bundle or all sections stale - fetch everything
            self.metrics['cache_misses']['bundle'] += 1
            bundle = await self.fetch_bundle(scene_scope)
            
            # Cache it
            await self._save_cache(scene_key, bundle)
            
            return bundle
    
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
    
    async def _fetch_section(self, section_name: str, scope: SceneScope) -> BundleSection:
        """Fetch a single section by name"""
        fetch_methods = {
            'npcs': self._fetch_npc_section,
            'memories': self._fetch_memory_section,
            'lore': self._fetch_lore_section,
            'conflicts': self._fetch_conflict_section,
            'world': self._fetch_world_section,
            'narrative': self._fetch_narrative_section
        }
        
        method = fetch_methods.get(section_name)
        if not method:
            logger.warning(f"Unknown section name: {section_name}")
            return BundleSection(data={}, canonical=False, priority=0)
        
        start_time = time.time()
        result = await method(scope)
        
        # Track fetch time
        fetch_time = time.time() - start_time
        self.metrics['fetch_times'][section_name].append(fetch_time)
        
        # Apply section-specific TTL
        result.ttl = self.section_ttls.get(section_name, 30.0)
        
        return result
    
    async def fetch_bundle(self, scene_scope: SceneScope) -> ContextBundle:
        """Fetch all context sections in parallel"""
        start_time = time.time()
        
        # Use semaphore to limit parallelism if configured
        max_parallel = self.ctx.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None
        
        async def fetch_with_semaphore(section_name: str):
            if semaphore:
                async with semaphore:
                    return await self._fetch_section(section_name, scene_scope)
            else:
                return await self._fetch_section(section_name, scene_scope)
        
        # Parallel fetch all sections
        results = await asyncio.gather(
            *[fetch_with_semaphore(name) for name in SECTION_NAMES],
            return_exceptions=True
        )
        
        # Handle results
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
        logger.info(f"Fetched context bundle in {fetch_time:.2f}s")
        
        # Add link hints and schema version to metadata
        return ContextBundle(
            scene_scope=scene_scope,
            **bundle_data,
            metadata={
                'fetch_time': fetch_time,
                'schema_version': SCHEMA_VERSION,  # Use constant
                'link_hints': scene_scope.link_hints
            }
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
        if not self.ctx.npc_orchestrator:
            return BundleSection(data=NPCSectionData(npcs=[], canonical_count=0), canonical=False, priority=0)
    
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
            
            # Fetch memories for NPCs
            for npc_id in npc_list:
                if hasattr(self.ctx.memory_orchestrator, 'retrieve_memories'):
                    result = await self.ctx.memory_orchestrator.retrieve_memories(
                        entity_type=EntityType.NPC,
                        entity_id=npc_id,
                        query=" ".join(list(scope.topics)[:5]),
                        limit=3,
                        use_llm_analysis=False  # Fast path
                    )
                else:
                    # Fallback to recall if available
                    result = await self.ctx.memory_orchestrator.recall(
                        entity_type=EntityType.NPC,
                        entity_id=npc_id,
                        query=" ".join(scope.topics),
                        limit=3
                    )
                
                if result and 'memories' in result:
                    relevant_memories.extend(result['memories'][:3])
            
            # Get location memories
            if scope.location_id:
                if hasattr(self.ctx.memory_orchestrator, 'retrieve_memories'):
                    result = await self.ctx.memory_orchestrator.retrieve_memories(
                        entity_type=EntityType.LOCATION,
                        entity_id=scope.location_id,
                        limit=5,
                        use_llm_analysis=False
                    )
                else:
                    result = await self.ctx.memory_orchestrator.recall(
                        entity_type=EntityType.LOCATION,
                        entity_id=scope.location_id,
                        query="",
                        limit=5
                    )
                
                if result and 'memories' in result:
                    recent_memories.extend(result['memories'][:5])
            
            # Get memory patterns if available
            if hasattr(self.ctx, 'analyze_memory_patterns') and scope.topics:
                pattern_result = await self.ctx.analyze_memory_patterns(
                    topic=", ".join(list(scope.topics)[:3])
                )
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
        if not self.ctx.lore_orchestrator:
            return BundleSection(data=LoreSectionData(location={}, world={}, canonical_rules=[]),
                                 canonical=False, priority=0)
    
        # --- Fast path: one RPC for the whole scene ---
        if hasattr(self.ctx.lore_orchestrator, 'get_scene_bundle'):
            try:
                bundle = await self.ctx.lore_orchestrator.get_scene_bundle(scope)
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
                    canonical_rules=rules[:5]
                )
    
                return BundleSection(
                    data=section_data,
                    canonical=bool(bundle.get('canonical')) or bool(rules),
                    priority=6,
                    last_changed_at=float(bundle.get('last_changed_at', time.time())),
                    version=bundle.get('version') or f"lore_scene_{int(time.time())}"
                )
            except Exception as e:
                logger.error(f"Lore scene bundle fetch failed, falling back: {e}")
    
        # --- Fallback path (legacy calls) ---
        location_lore, world_lore, canonical_rules = {}, {}, []
        try:
            # Location by id OR by name
            if scope.location_id:
                if isinstance(scope.location_id, int) and hasattr(self.ctx.lore_orchestrator, '_fetch_location_lore_for_bundle'):
                    location_result = await self.ctx.lore_orchestrator._fetch_location_lore_for_bundle(scope.location_id)
                else:
                    # Treat anything not int as a name
                    location_result = await self.ctx.lore_orchestrator.get_location_context(str(scope.location_id))
    
                if location_result:
                    location_lore = {
                        'description': (location_result.get('description') or '')[:200],
                        'governance': location_result.get('governance') or {},
                        'culture': location_result.get('culture') or {},
                    }
                    # Surface tags to scope so later calls can use them
                    if 'tags' in location_result:
                        scope.lore_tags.update(location_result['tags'][:10])
    
            # Topic/tag expansion + tag fetch
            tag_seed = list(scope.lore_tags)[:5]
            if 'related_tags' in scope.link_hints:
                tag_seed.extend(scope.link_hints['related_tags'][:3])
    
            if tag_seed and hasattr(self.ctx.lore_orchestrator, 'get_tagged_lore'):
                tagged_lore = await self.ctx.lore_orchestrator.get_tagged_lore(tags=tag_seed)
                world_lore = tagged_lore or {}
    
            # Canon rules
            if hasattr(self.ctx.lore_orchestrator, 'check_canonical_consistency'):
                canonical = await self.ctx.lore_orchestrator.check_canonical_consistency()
                canonical_rules = (canonical.get('rules') or [])[:5]
    
        except Exception as e:
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
            # Use get_conflict_state (existing API)
            conflict_ids = list(getattr(scope, "conflict_ids", []))[:5]
            if hasattr(synthesizer, "get_conflict_state"):
                for conflict_id in conflict_ids:
                    state = await synthesizer.get_conflict_state(conflict_id)
                    if state:
                        subsystem = state.get("subsystem_data", {}) if isinstance(state, dict) else {}
                        tension = subsystem.get("tension", {}) if isinstance(subsystem, dict) else {}
                        stakeholder = subsystem.get("stakeholder", {}) if isinstance(subsystem, dict) else {}
    
                        conflict_data["active"].append({
                            "id": conflict_id,
                            "type": state.get("conflict_type") if isinstance(state, dict) else None,
                            "intensity": tension.get("level", 0.5),
                            "stakeholders": stakeholder.get("stakeholders", []),
                        })
    
            # Build NPC list including link hints for conflict filtering
            relevant_npcs = set(getattr(scope, "npc_ids", []) or [])
            link_hints = getattr(scope, "link_hints", {}) or {}
            if isinstance(link_hints, dict) and "related_npcs" in link_hints:
                related = link_hints.get("related_npcs") or []
                relevant_npcs.update(related[:3])
    
            # Get tensions using existing calculate_conflict_tensions
            if hasattr(self.ctx, "calculate_conflict_tensions"):
                all_tensions = await self.ctx.calculate_conflict_tensions()
                if relevant_npcs and all_tensions:
                    conflict_data["tensions"] = self._filter_tensions_for_npcs(
                        all_tensions,
                        relevant_npcs,
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
        if not self.ctx.world_director:
            return BundleSection(data={}, canonical=False, priority=0)
        
        world_data = {}
        
        try:
            if hasattr(self.ctx.world_director, 'get_world_state'):
                world_state = await self.ctx.world_director.get_world_state()
                if world_state:
                    # Safe extraction with getattr and enum handling
                    mood = getattr(world_state, 'world_mood', None)
                    if isinstance(mood, Enum):
                        mood = mood.value
                    
                    weather = getattr(world_state, 'weather', None)
                    if isinstance(weather, Enum):
                        weather = weather.value
                    
                    world_data = {
                        'time': getattr(world_state, 'current_time', None),
                        'mood': mood,
                        'weather': weather,
                        'events': getattr(world_state, 'active_events', [])[:3]
                    }
                    
                    # Add player vitals if available
                    vitals = getattr(world_state, 'player_vitals', None)
                    if vitals:
                        world_data['vitals'] = {
                            'fatigue': getattr(vitals, 'fatigue', 0),
                            'hunger': getattr(vitals, 'hunger', 100),
                            'thirst': getattr(vitals, 'thirst', 100)
                        }
                    
        except Exception as e:
            logger.error(f"World fetch failed: {e}")
        
        return BundleSection(
            data=world_data,
            canonical=False,
            priority=4,
            last_changed_at=time.time(),
            version=f"world_{time.time()}"
        )
    
    async def _fetch_narrative_section(self, scope: SceneScope) -> BundleSection:
        """Fetch narrative context with safe method calls"""
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
    lore_orchestrator: Optional[LoreOrchestrator] = None
    world_director: Optional[Any] = None  # CompleteWorldDirector
    slice_of_life_narrator: Optional[Any] = None  # SliceOfLifeNarrator
    
    # ────────── CONTEXT BROKER (NEW) ──────────
    context_broker: Optional[ContextBroker] = None
    
    # ────────── CURRENT STATE ──────────
    current_context: Dict[str, Any] = field(default_factory=dict)
    current_location: Optional[str] = None
    current_world_state: Optional[Any] = None
    config: Optional[Any] = None  # For model context budget
    
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
    
    async def initialize(self):
        """Initialize all orchestrators and the context broker"""
        # Try to load config if available
        try:
            from .config import Config
            self.config = Config
        except ImportError:
            pass
        
        initialization_tasks = []
        
        # Initialize Memory Orchestrator
        initialization_tasks.append(self._init_memory_orchestrator())
        
        # Initialize Lore Orchestrator
        initialization_tasks.append(self._init_lore_orchestrator())
        
        # Initialize NPC Orchestrator
        initialization_tasks.append(self._init_npc_orchestrator())
        
        # Initialize Conflict Synthesizer
        initialization_tasks.append(self._init_conflict_synthesizer())
        
        # Initialize World Systems
        if WORLD_SIMULATION_AVAILABLE:
            initialization_tasks.append(self._init_world_systems())
        
        # Run all initializations in parallel
        if self.enable_parallel_fetch:
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Initialization task {i} failed: {result}")
                    self.log_error(result, {"task": f"init_{i}"})
        else:
            # Fallback to sequential initialization
            for task in initialization_tasks:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Initialization failed: {e}")
                    self.log_error(e, {"task": "init"})
        
        # Initialize the context broker
        self.context_broker = ContextBroker(self)
        await self.context_broker.initialize()
        
        logger.info(f"NyxContext initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    async def _init_memory_orchestrator(self):
        """Initialize memory orchestrator"""
        try:
            self.memory_orchestrator = await get_memory_orchestrator(
                self.user_id, 
                self.conversation_id
            )
            logger.info("Memory Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Memory Orchestrator: {e}")
    
    async def _init_lore_orchestrator(self):
        """Initialize lore orchestrator"""
        try:
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
    
    async def _init_npc_orchestrator(self):
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
    
    async def _init_conflict_synthesizer(self):
        """Initialize conflict synthesizer"""
        try:
            self.conflict_synthesizer = await get_conflict_synthesizer(
                self.user_id,
                self.conversation_id
            )
            logger.info("Conflict Synthesizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Conflict Synthesizer: {e}")
    
    async def _init_world_systems(self):
        """Initialize world director and narrator"""
        try:
            from story_agent.world_director_agent import CompleteWorldDirector
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
            
            self.world_director = CompleteWorldDirector(self.user_id, self.conversation_id)
            await self.world_director.initialize()
            
            self.slice_of_life_narrator = SliceOfLifeNarrator(self.user_id, self.conversation_id)
            await self.slice_of_life_narrator.initialize()
            
            logger.info("World systems initialized")
        except Exception as e:
            logger.error(f"Failed to initialize world systems: {e}")

    async def handle_day_transition(self, new_day: int):
        """Handle game day transitions: notify conflicts & clear scene caches."""
        logger.info(f"NyxContext handling day transition to {new_day}")
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
    
    async def build_context_for_input(self, user_input: str, context_data: Dict[str, Any] = None) -> PackedContext:
        """Main entry point: build optimized context for user input"""
        start_time = time.time()
        
        # Merge provided context
        if context_data:
            self.current_context.update(context_data)
        
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
