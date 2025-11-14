# logic/conflict_system/social_circle.py
"""
Social Circle Conflict System with LLM-generated dynamics.
Refactored to work as a ConflictSubsystem with the synthesizer.
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
import weakref
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from nyx.gateway.llm_gateway import LLMRequest, LLMOperation, execute
from logic.conflict_system.dynamic_conflict_template import extract_runner_response
from db.connection import get_db_connection_context
from logic.conflict_system.conflict_synthesizer import (
    ConflictSubsystem, SubsystemType, EventType,
    SystemEvent, SubsystemResponse
)

logger = logging.getLogger(__name__)

# ===============================================================================
# SOCIAL STRUCTURES (Preserved from original)
# ===============================================================================

class SocialRole(Enum):
    """Roles within social circles"""
    ALPHA = "alpha"
    BETA = "beta"
    CONFIDANT = "confidant"
    RIVAL = "rival"
    OUTSIDER = "outsider"
    GOSSIP = "gossip"
    MEDIATOR = "mediator"
    INFLUENCER = "influencer"
    FOLLOWER = "follower"
    WILDCARD = "wildcard"

class ReputationType(Enum):
    """Types of reputation"""
    TRUSTWORTHY = "trustworthy"
    SUBMISSIVE = "submissive"
    REBELLIOUS = "rebellious"
    MYSTERIOUS = "mysterious"
    INFLUENTIAL = "influential"
    SCANDALOUS = "scandalous"
    NURTURING = "nurturing"
    DANGEROUS = "dangerous"

class GossipType(Enum):
    """Types of gossip that spread"""
    RUMOR = "rumor"
    SECRET = "secret"
    SCANDAL = "scandal"
    PRAISE = "praise"
    WARNING = "warning"
    SPECULATION = "speculation"

@dataclass
class SocialCircle:
    """A social group with its own dynamics"""
    circle_id: int
    name: str
    description: str
    members: List[int]
    hierarchy: Dict[int, SocialRole]
    group_mood: str
    shared_values: List[str]
    current_gossip: List['GossipItem']
    tension_points: Dict[str, float]

@dataclass
class GossipItem:
    """A piece of gossip circulating"""
    gossip_id: int
    gossip_type: GossipType
    content: str
    about: List[int]
    spreaders: Set[int]
    believers: Set[int]
    deniers: Set[int]
    spread_rate: float
    truthfulness: float
    impact: Dict[str, Any]

@dataclass
class SocialConflict:
    """A conflict within social dynamics"""
    conflict_id: int
    conflict_type: str
    participants: List[int]
    stakes: str
    current_phase: str
    alliances: Dict[int, List[int]]
    public_opinion: Dict[int, float]

# ===============================================================================
# SOCIAL CIRCLE CONFLICT SUBSYSTEM
# ===============================================================================

class SocialCircleConflictSubsystem(ConflictSubsystem):
    """
    Social dynamics subsystem integrated with synthesizer.
    Manages social relationships, gossip, reputation, and group dynamics.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None
        
        # Components
        self.manager = SocialCircleManager(user_id, conversation_id)
        
        # State tracking
        self._active_gossip: Dict[int, GossipItem] = {}
        self._reputation_cache: Dict[int, Dict[ReputationType, float]] = {}
        self._social_circles: Dict[int, SocialCircle] = {}
        self._fact_gossip_map: Dict[str, int] = {}
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.SOCIAL
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'gossip_generation',
            'gossip_spreading',
            'reputation_tracking',
            'alliance_formation',
            'social_conflict_generation',
            'group_dynamics'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        return {SubsystemType.STAKEHOLDER}
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        return {
            EventType.CONFLICT_CREATED,
            EventType.STAKEHOLDER_ACTION,
            EventType.NPC_REACTION,
            EventType.STATE_SYNC,
            EventType.CONFLICT_RESOLVED,
            EventType.CANON_ESTABLISHED,
        }
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        """Initialize with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        self.manager.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle events from synthesizer"""
        try:
            if event.event_type == EventType.STATE_SYNC:
                return await self._handle_state_sync(event)
            elif event.event_type == EventType.CANON_ESTABLISHED:
                return await self._on_fact_became_public(event)
            elif event.event_type == EventType.STAKEHOLDER_ACTION:
                return await self._handle_stakeholder_action(event)
            elif event.event_type == EventType.NPC_REACTION:
                return await self._handle_npc_reaction(event)
            elif event.event_type == EventType.CONFLICT_CREATED:
                return await self._handle_conflict_created(event)
            elif event.event_type == EventType.CONFLICT_RESOLVED:
                return await self._handle_conflict_resolved(event)
            else:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'handled': False}
                )
        except Exception as e:
            logger.error(f"SocialCircle error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of social subsystem"""
        try:
            active_gossip = len(self._active_gossip)
            tracked_reputations = len(self._reputation_cache)
            
            return {
                'healthy': True,
                'active_gossip': active_gossip,
                'tracked_reputations': tracked_reputations,
                'social_circles': len(self._social_circles)
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get social-specific conflict data"""
        
        # Check if this conflict has social dimensions
        related_gossip = [
            g for g in self._active_gossip.values()
            if conflict_id in g.about
        ]
        
        return {
            'subsystem': 'social',
            'related_gossip': len(related_gossip),
            'social_stakes': self._analyze_social_stakes(conflict_id)
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of social subsystem"""
        return {
            'active_gossip_items': len(self._active_gossip),
            'reputation_tracking': len(self._reputation_cache),
            'social_tension': self._calculate_social_tension()
        }
    
    # ========== Event Handlers ==========
    
    async def _handle_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Handle scene state synchronization"""
        raw_payload = event.payload or {}
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        scene_payload = raw_payload.get('scene_context') if isinstance(raw_payload.get('scene_context'), dict) else None
        scene_context = dict(scene_payload or raw_payload)
        scene_context.setdefault('user_id', self.user_id)
        scene_context.setdefault('conversation_id', self.conversation_id)
        present_npcs = (
            scene_context.get('present_npcs')
            or scene_context.get('npcs_present')
            or []
        )
        if not isinstance(present_npcs, list):
            if isinstance(present_npcs, (set, tuple)):
                present_npcs = list(present_npcs)
            elif present_npcs:
                present_npcs = [present_npcs]
            else:
                present_npcs = []

        # HOT PATH: Use cached social data and dispatch background tasks
        from logic.conflict_system.social_circle_hotpath import (
            get_scene_bundle,
            schedule_gossip_generation,
            get_cached_reputation_scores,
        )

        # Compute scene hash for cache lookups
        scene_hash = hashlib.sha256(json.dumps(scene_context, sort_keys=True).encode()).hexdigest()[:16]
        scene_context['scene_hash'] = scene_hash

        # Get cached social bundle (fast)
        social_bundle = get_scene_bundle(
            scene_hash,
            scene_context=scene_context,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            target_npcs=present_npcs if present_npcs else None,
            eager=False,
        )

        side_effects = []

        # If bundle is generating, queue explicit gossip generation for these NPCs
        if present_npcs and len(present_npcs) >= 2 and social_bundle.get('status') == 'generating':
            if random.random() < 0.3:  # 30% chance of gossip
                schedule_gossip_generation(
                    scene_context,
                    present_npcs[:2],
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                )

        # Get cached reputations (fast)
        for npc_id in present_npcs:
            if npc_id not in self._reputation_cache:
                reputation = await get_cached_reputation_scores(npc_id)
                self._reputation_cache[npc_id] = reputation

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'social_bundle_status': social_bundle.get('status', 'cached'),
                'gossip_count': len(social_bundle.get('gossip', [])),
                'reputations_cached': len(present_npcs)
            },
            side_effects=side_effects
        )

    async def _on_fact_became_public(self, event: SystemEvent) -> SubsystemResponse:
        """Handle newly public facts by updating gossip state."""
        payload = event.payload or {}
        if not isinstance(payload, dict):
            payload = {}

        fact_id = payload.get('fact_id')
        holder_id = payload.get('holder_id')
        target_ids = payload.get('targets') or []
        visibility = payload.get('visibility', 'local')

        if not fact_id or not target_ids:
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'gossip_created': False, 'reason': 'missing_fact_or_targets'},
            )

        created_ids = await self._create_or_update_gossip_items(
            fact_id=fact_id,
            holder_id=holder_id,
            targets=target_ids,
            visibility=visibility,
            context=payload,
        )

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'gossip_created': bool(created_ids),
                'gossip_ids': created_ids,
                'targets': target_ids,
                'fact_id': fact_id,
            },
        )

    async def _handle_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions affecting social dynamics"""
        stakeholder_id = event.payload.get('stakeholder_id')
        action_type = event.payload.get('action_type')

        # HOT PATH: Fast reputation updates and task dispatch
        from logic.conflict_system.social_circle_hotpath import (
            apply_reputation_change,
            queue_reputation_narration,
            queue_alliance_formation,
        )

        side_effects = []

        # Actions can affect reputation (fast numeric update)
        if action_type in ['betray', 'support', 'oppose']:
            old_reputation = self._reputation_cache.get(stakeholder_id, {})
            new_reputation = await self._adjust_reputation_from_action(
                stakeholder_id, action_type
            )

            # Apply fast numeric reputation changes
            for rep_type, delta in self._calculate_reputation_delta(action_type).items():
                apply_reputation_change(stakeholder_id, rep_type, delta)

            # Queue background narration if significant
            if old_reputation:
                queue_reputation_narration(stakeholder_id, old_reputation, new_reputation)

        # Check for alliance formation (dispatch to background)
        if action_type == 'ally':
            target_id = event.payload.get('target_id')
            if target_id:
                queue_alliance_formation(stakeholder_id, target_id, 'mutual_benefit')

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'social_impact_processed': True, 'tasks_dispatched': True},
            side_effects=side_effects
        )
    
    async def _handle_npc_reaction(self, event: SystemEvent) -> SubsystemResponse:
        """Handle NPC reactions affecting social dynamics"""
        npc_id = event.payload.get('npc_id')
        reaction_type = event.payload.get('reaction_type')
        
        # NPCs can spread gossip
        if reaction_type == 'gossip' and self._active_gossip:
            gossip = random.choice(list(self._active_gossip.values()))
            listeners = event.payload.get('listeners', [])
            
            if listeners:
                spread_results = await self.manager.spread_gossip(
                    gossip, npc_id, listeners
                )
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={
                        'gossip_spread': True,
                        'new_believers': spread_results['new_believers'],
                        'new_spreaders': spread_results['new_spreaders']
                    }
                )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'reaction_noted': True}
        )
    
    async def _handle_conflict_created(self, event: SystemEvent) -> SubsystemResponse:
        """Handle new conflict creation with social dimensions"""
        conflict_id = event.payload.get('conflict_id')
        participants = event.payload.get('participants', [])

        # HOT PATH: Dispatch gossip generation to background
        if participants:
            from logic.conflict_system.social_circle_hotpath import schedule_gossip_generation

            scene_context = {
                'conflict_start': True,
                'conflict_id': conflict_id,
                'participants': participants,
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
            }
            scene_context['scene_hash'] = hashlib.sha256(
                json.dumps(scene_context, sort_keys=True).encode()
            ).hexdigest()[:16]

            schedule_gossip_generation(
                scene_context,
                participants,
                user_id=self.user_id,
                conversation_id=self.conversation_id,
            )

            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={'gossip_queued': True, 'message': 'Gossip generation dispatched to background'}
            )

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'no_social_dimension': True}
        )

    async def _create_or_update_gossip_items(
        self,
        fact_id: Any,
        holder_id: Any,
        targets: List[Any],
        visibility: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Create or update gossip entries for a newly public fact."""

        fact_key = str(fact_id)
        normalized_targets: List[int] = []
        for target in targets:
            if isinstance(target, int):
                normalized_targets.append(target)
            else:
                try:
                    normalized_targets.append(int(target))
                except (TypeError, ValueError):
                    continue

        holder_normalized: Optional[int]
        if isinstance(holder_id, int):
            holder_normalized = holder_id
        else:
            try:
                holder_normalized = int(holder_id)
            except (TypeError, ValueError):
                holder_normalized = None

        created_ids: List[int] = []
        existing_id = self._fact_gossip_map.get(fact_key)

        visibility_map = {
            'local': GossipType.RUMOR,
            'faction': GossipType.SECRET,
            'global': GossipType.SCANDAL,
        }
        gossip_type = visibility_map.get(str(visibility).lower(), GossipType.RUMOR)

        if existing_id and existing_id in self._active_gossip:
            gossip_item = self._active_gossip[existing_id]
            merged_targets = set(gossip_item.about or [])
            merged_targets.update(normalized_targets)
            gossip_item.about = list(merged_targets)

            if holder_normalized is not None:
                gossip_item.spreaders.add(holder_normalized)

            impact = gossip_item.impact or {}
            impact.setdefault('fact_id', fact_key)
            impact.setdefault('visibility', visibility)
            if normalized_targets:
                existing_targets = set(impact.get('targets', []))
                impact['targets'] = list(existing_targets.union(normalized_targets))
            gossip_item.impact = impact
        else:
            gossip_id = int(uuid.uuid4().int % 1_000_000_000)
            content_hint = ''
            if context:
                content_hint = context.get('fact_summary') or context.get('summary') or context.get('description') or ''

            if not content_hint:
                target_label = ', '.join(str(t) for t in normalized_targets[:3]) or 'others'
                content_hint = f"Word spreads about fact {fact_key} involving {target_label}."

            gossip_item = GossipItem(
                gossip_id=gossip_id,
                gossip_type=gossip_type,
                content=content_hint,
                about=list(set(normalized_targets)),
                spreaders={holder_normalized} if holder_normalized is not None else set(),
                believers=set(normalized_targets),
                deniers=set(),
                spread_rate=0.5 if str(visibility).lower() == 'global' else 0.35,
                truthfulness=0.9,
                impact={
                    'fact_id': fact_key,
                    'visibility': visibility,
                    'targets': list(set(normalized_targets)),
                    'holder_id': holder_normalized,
                },
            )

            self._active_gossip[gossip_id] = gossip_item
            self._fact_gossip_map[fact_key] = gossip_id
            created_ids.append(gossip_id)

            # v1 behavior: schedule background LLM gossip generation for new facts
            try:
                from logic.conflict_system.social_circle_hotpath import (
                    schedule_gossip_generation,
                )

                unique_targets = list(set(normalized_targets))
                context_snapshot = {
                    'fact_id': fact_key,
                    'holder_id': holder_normalized,
                    'targets': unique_targets,
                    'visibility': visibility,
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id,
                }
                scene_hash = hashlib.sha256(
                    json.dumps(context_snapshot, sort_keys=True).encode()
                ).hexdigest()[:16]
                context_snapshot['scene_hash'] = scene_hash

                schedule_gossip_generation(
                    context_snapshot,
                    unique_targets,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to queue gossip generation for fact %s: %s",
                    fact_key,
                    exc,
                )

        return created_ids

    def _calculate_reputation_delta(self, action_type: str) -> Dict[str, float]:
        """Fast rule-based reputation delta calculation (no LLM)."""
        deltas = {
            'betray': {'trustworthy': -0.3, 'scandalous': 0.2, 'dangerous': 0.1},
            'support': {'trustworthy': 0.2, 'nurturing': 0.1, 'influential': 0.1},
            'oppose': {'rebellious': 0.2, 'dangerous': 0.1, 'mysterious': 0.05},
        }
        return deltas.get(action_type, {})

    async def _handle_conflict_resolved(self, event: SystemEvent) -> SubsystemResponse:
        """Handle conflict resolution affecting social dynamics"""
        conflict_id = event.payload.get('conflict_id')
        resolution = event.payload.get('resolution', {})

        from logic.conflict_system.social_circle_hotpath import (
            get_cached_gossip_items,
            get_cached_reputation_scores,
            schedule_gossip_generation,
            schedule_reputation_calculation,
        )

        winners = resolution.get('winners', []) or []
        losers = resolution.get('losers', []) or []
        all_participants = [p for p in winners + losers if p is not None]

        side_effects: List[SystemEvent] = []

        scene_context = {
            'conflict_resolved': True,
            'conflict_id': conflict_id,
            'resolution': resolution,
            'participants': all_participants,
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
        }
        scene_context['scene_hash'] = hashlib.sha256(
            json.dumps(scene_context, sort_keys=True).encode()
        ).hexdigest()[:16]

        initial_gossip_cache = []
        if all_participants:
            schedule_gossip_generation(
                scene_context,
                all_participants,
                user_id=self.user_id,
                conversation_id=self.conversation_id,
            )
            initial_gossip_cache = await get_cached_gossip_items(
                scene_context['scene_hash'], limit=3
            )

        # Apply fast heuristic reputation adjustments
        for winner_id in winners:
            await self._adjust_reputation_from_action(winner_id, 'victory')
            schedule_reputation_calculation(
                self.user_id,
                self.conversation_id,
                winner_id,
                scene_context=scene_context,
            )

        for loser_id in losers:
            await self._adjust_reputation_from_action(loser_id, 'defeat')
            schedule_reputation_calculation(
                self.user_id,
                self.conversation_id,
                loser_id,
                scene_context=scene_context,
            )

        if all_participants:
            gossip_item = self._hydrate_cached_gossip(
                initial_gossip_cache[0] if initial_gossip_cache else None,
                all_participants,
            )

            if gossip_item:
                if gossip_item.gossip_id:
                    self._active_gossip[gossip_item.gossip_id] = gossip_item

                side_effects.append(SystemEvent(
                    event_id=f"resolution_gossip_{gossip_item.gossip_id or 'pending'}",
                    event_type=EventType.NPC_REACTION,
                    source_subsystem=self.subsystem_type,
                    payload={'gossip': gossip_item.content},
                    priority=7
                ))

            # Schedule follow-up checks for richer gossip once background tasks finish
            baseline_ids = {
                g.get('gossip_id') for g in initial_gossip_cache if g.get('gossip_id')
            }
            self._schedule_gossip_followup(
                scene_context['scene_hash'],
                all_participants,
                baseline_ids,
            )

        # Capture existing cached reputation for follow-up diffing
        initial_reputation = {}
        for npc_id in all_participants:
            cached_rep = await get_cached_reputation_scores(npc_id)
            if cached_rep:
                initial_reputation[npc_id] = cached_rep

        if all_participants:
            self._schedule_reputation_followup(all_participants, initial_reputation)

        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'social_aftermath_processed': True},
            side_effects=side_effects
        )
    
    # ========== Helper Methods ==========

    def _hydrate_cached_gossip(
        self,
        cached: Optional[Dict[str, Any]],
        fallback_participants: List[int],
    ) -> GossipItem:
        """Convert cached gossip payload into a GossipItem."""

        if not cached:
            return self.manager._create_fallback_gossip(fallback_participants)

        gossip_type_value = str(cached.get('gossip_type', 'rumor')).upper()
        gossip_type = GossipType.RUMOR
        try:
            gossip_type = GossipType[gossip_type_value]
        except KeyError:
            logger.debug("Unknown gossip type '%s' in cache", gossip_type_value)

        about = cached.get('about') or fallback_participants
        spreaders = set(cached.get('spreaders', []) or [])
        believers = set(cached.get('believers', []) or [])
        deniers = set(cached.get('deniers', []) or [])

        return GossipItem(
            gossip_id=int(cached.get('gossip_id') or 0),
            gossip_type=gossip_type,
            content=str(cached.get('content', 'People are whispering...')),
            about=about,
            spreaders=spreaders,
            believers=believers,
            deniers=deniers,
            spread_rate=float(cached.get('spread_rate', 0.3) or 0.3),
            truthfulness=float(cached.get('truthfulness', 0.5) or 0.5),
            impact=cached.get('impact', {}),
        )

    def _schedule_gossip_followup(
        self,
        scene_hash: str,
        participants: List[int],
        baseline_ids: Set[int],
    ) -> None:
        """Poll cache for new gossip and emit follow-up events when available."""

        synthesizer = self.synthesizer() if self.synthesizer else None
        if not synthesizer:
            return

        async def _poll_for_gossip() -> None:
            from logic.conflict_system.social_circle_hotpath import get_cached_gossip_items

            for attempt in range(4):
                await asyncio.sleep(min(8, 2 ** attempt))
                cached_items = await get_cached_gossip_items(scene_hash, limit=3)
                new_items = [
                    item for item in cached_items
                    if item.get('gossip_id') and item.get('gossip_id') not in baseline_ids
                ]

                if new_items:
                    gossip_item = self._hydrate_cached_gossip(new_items[0], participants)
                    if gossip_item.gossip_id:
                        self._active_gossip[gossip_item.gossip_id] = gossip_item

                    event = SystemEvent(
                        event_id=f"gossip_refresh_{scene_hash}",
                        event_type=EventType.NPC_REACTION,
                        source_subsystem=self.subsystem_type,
                        payload={'gossip': gossip_item.content},
                        priority=6,
                    )

                    try:
                        await synthesizer.emit_event(event)
                    except Exception as exc:  # pragma: no cover - defensive log
                        logger.warning("Failed to emit gossip follow-up: %s", exc)
                    break

        asyncio.create_task(_poll_for_gossip())

    def _schedule_reputation_followup(
        self,
        npc_ids: List[int],
        baseline: Dict[int, Dict[str, float]],
    ) -> None:
        """Poll cached reputation scores and refresh local cache when updated."""

        synthesizer = self.synthesizer() if self.synthesizer else None
        if not synthesizer:
            return

        async def _poll_for_reputation() -> None:
            from logic.conflict_system.social_circle_hotpath import get_cached_reputation_scores

            for attempt in range(4):
                await asyncio.sleep(min(8, 2 ** attempt))
                updates: Dict[int, Dict[str, float]] = {}

                for npc_id in npc_ids:
                    cached = await get_cached_reputation_scores(npc_id)
                    if not cached:
                        continue
                    if baseline.get(npc_id) != cached:
                        updates[npc_id] = cached

                if updates:
                    payload_updates: Dict[int, Dict[str, float]] = {}
                    for npc_id, rep_values in updates.items():
                        converted: Dict[ReputationType, float] = {}
                        for key, value in rep_values.items():
                            try:
                                enum_key = ReputationType[key.upper()]
                            except KeyError:
                                continue
                            converted[enum_key] = float(value)

                        if converted:
                            self._reputation_cache[npc_id] = converted
                            payload_updates[npc_id] = {
                                k.value: v for k, v in converted.items()
                            }

                    if payload_updates:
                        event = SystemEvent(
                            event_id=f"reputation_refresh_{uuid.uuid4()}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=self.subsystem_type,
                            payload={'updated_reputation': payload_updates},
                            priority=6,
                        )
                        try:
                            await synthesizer.emit_event(event)
                        except Exception as exc:  # pragma: no cover - defensive log
                            logger.warning("Failed to emit reputation follow-up: %s", exc)
                    break

        asyncio.create_task(_poll_for_reputation())

    def _analyze_social_stakes(self, conflict_id: int) -> str:
        """Analyze what's at stake socially in a conflict"""
        # Simple analysis - could be enhanced
        return "reputation and social standing"
    
    def _calculate_social_tension(self) -> float:
        """Calculate overall social tension"""
        if not self._active_gossip:
            return 0.0
        
        # More gossip = more tension
        base_tension = min(1.0, len(self._active_gossip) / 10)
        
        # Scandalous gossip increases tension
        scandal_count = sum(1 for g in self._active_gossip.values() 
                          if g.gossip_type == GossipType.SCANDAL)
        
        return min(1.0, base_tension + (scandal_count * 0.1))
    
    async def _adjust_reputation_from_action(
        self,
        entity_id: int,
        action_type: str
    ) -> Dict[ReputationType, float]:
        """Adjust reputation based on an action"""
        
        current = self._reputation_cache.get(entity_id, {})
        
        # Simple adjustments - could be more sophisticated
        adjustments = {
            'betray': {ReputationType.TRUSTWORTHY: -0.3, ReputationType.DANGEROUS: 0.2},
            'support': {ReputationType.TRUSTWORTHY: 0.1, ReputationType.NURTURING: 0.1},
            'oppose': {ReputationType.REBELLIOUS: 0.2, ReputationType.SUBMISSIVE: -0.2},
            'victory': {ReputationType.INFLUENTIAL: 0.2, ReputationType.DANGEROUS: 0.1},
            'defeat': {ReputationType.INFLUENTIAL: -0.1, ReputationType.SUBMISSIVE: 0.1}
        }
        
        changes = adjustments.get(action_type, {})
        
        new_reputation = current.copy() if current else {r: 0.5 for r in ReputationType}
        for rep_type, change in changes.items():
            if rep_type in new_reputation:
                new_reputation[rep_type] = max(0, min(1, new_reputation[rep_type] + change))
        
        self._reputation_cache[entity_id] = new_reputation
        return new_reputation

# ===============================================================================
# ORIGINAL MANAGER CLASS (Preserved with minor modifications)
# ===============================================================================

class SocialCircleManager:
    """
    Manages social dynamics using LLM for dynamic generation.
    Modified to work with synthesizer.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # Will be set by subsystem
        self._gossip_generator = None
        self._social_analyzer = None
        self._reputation_narrator = None
        self._alliance_strategist = None
    
    @property
    def gossip_generator(self) -> Agent:
        """Agent for generating dynamic gossip"""
        if self._gossip_generator is None:
            self._gossip_generator = Agent(
                name="Gossip Generator",
                instructions="""
                Generate realistic gossip for a matriarchal society setting.

                Return **only** a single-line minified JSON object with keys:
                {"type": "rumor|secret|scandal|praise|warning|speculation",
                 "content": string,
                 "truthfulness": number,
                 "spread_rate": number,
                 "impact": object}

                No markdown, no code fences, no commentary.
                """,
                model="gpt-5-nano",
            )
        return self._gossip_generator
    
    @property
    def social_analyzer(self) -> Agent:
        """Agent for analyzing social dynamics"""
        if self._social_analyzer is None:
            self._social_analyzer = Agent(
                name="Social Dynamics Analyzer",
                instructions="""
                Analyze complex social situations and group dynamics.
                
                Consider:
                - Power hierarchies and social roles
                - Alliances and rivalries
                - Group cohesion and fracture points
                - Information flow and influence
                - Cultural norms and violations
                
                Identify subtle social conflicts and tensions.
                Predict how social dynamics might evolve.
                """,
                model="gpt-5-nano",
            )
        return self._social_analyzer
    
    @property
    def reputation_narrator(self) -> Agent:
        """Agent for narrating reputation changes"""
        if self._reputation_narrator is None:
            self._reputation_narrator = Agent(
                name="Reputation Narrator",
                instructions="""
                Narrate how reputations shift and evolve.
                
                Focus on:
                - How whispers become accepted truths
                - The social cost of reputation changes
                - How different groups view the same person
                - The slow burn of reputation recovery or loss
                
                Create nuanced descriptions that show social complexity.
                """,
                model="gpt-5-nano",
            )
        return self._reputation_narrator
    
    @property
    def alliance_strategist(self) -> Agent:
        """Agent for generating alliance dynamics"""
        if self._alliance_strategist is None:
            self._alliance_strategist = Agent(
                name="Alliance Strategist",
                instructions="""
                Generate realistic alliance formations and betrayals.
                
                Consider:
                - Mutual benefits and shared enemies
                - Temporary vs permanent alliances
                - The cost of betrayal
                - Power consolidation strategies
                - Social pressure and group think
                
                Create complex webs of loyalty and opportunism.
                """,
                model="gpt-5-nano",
            )
        return self._alliance_strategist
    
    async def generate_gossip_background(
        self,
        context: Dict[str, Any],
        target_npcs: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run the slow gossip generation flow (for background tasks)."""

        gossip_item = await self._run_gossip_llm(context, target_npcs or [])
        return {
            'gossip_id': gossip_item.gossip_id,
            'gossip_type': gossip_item.gossip_type.value,
            'content': gossip_item.content,
            'about': gossip_item.about,
            'spreaders': list(gossip_item.spreaders),
            'believers': list(gossip_item.believers),
            'deniers': list(gossip_item.deniers),
            'spread_rate': gossip_item.spread_rate,
            'truthfulness': gossip_item.truthfulness,
            'impact': gossip_item.impact,
        }

    async def _run_gossip_llm(
        self,
        context: Dict[str, Any],
        target_npcs: List[int],
    ) -> GossipItem:
        """Run the original gossip generation prompt (slow)."""

        npc_details = await self._get_npc_social_details(target_npcs)

        prompt = f"""
        Generate a piece of gossip for this social context:

        Setting: Matriarchal society, slice-of-life game
        Context: {json.dumps(context, indent=2)}
        Potential Targets: {npc_details}

        Create:
        1. Type (rumor/secret/scandal/praise/warning/speculation)
        2. Content (1-2 sentences, conversational)
        3. Truthfulness (0.0-1.0)
        4. Why it would spread (what makes it juicy)
        5. Potential impact on relationships

        Make it feel like real gossip - not too dramatic, but interesting.
        Format as JSON.
        """

        result = await execute(
            LLMRequest(
                prompt=prompt,
                agent=self.gossip_generator,
                metadata={
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stage": "gossip_generation",
                },
            )
        )

        def _normalize_gossip(obj: Dict[str, Any]) -> Dict[str, Any]:
            """Tolerant coercion of model output into our schema."""
            if not isinstance(obj, dict):
                raise ValueError("gossip payload is not an object")

            inner = obj.get("gossip") if isinstance(obj.get("gossip"), dict) else obj

            content = (
                inner.get("content")
                or inner.get("text")
                or inner.get("message")
                or (
                    inner.get("content", {})
                    if isinstance(inner.get("content"), dict)
                    else None
                )
            )

            if isinstance(content, dict):
                content = content.get("content") or content.get("text")

            if not isinstance(content, str) or not content.strip():
                raise KeyError("content")

            gtype = inner.get("type") or inner.get("gossip_type") or "RUMOR"
            try:
                gtype_norm = GossipType[str(gtype).upper()]
            except Exception:
                gtype_norm = GossipType.RUMOR

            truth = (
                inner.get("truthfulness")
                or inner.get("truth")
                or inner.get("veracity")
                or 0.5
            )
            spread = (
                inner.get("spread_rate")
                or inner.get("spreadProbability")
                or 0.5
            )
            impact = inner.get("impact") or {}

            return {
                "content": content.strip(),
                "type": gtype_norm,
                "truthfulness": float(truth),
                "spread_rate": float(spread),
                "impact": impact if isinstance(impact, dict) else {},
            }

        payload: Optional[str] = None
        try:
            payload = extract_runner_response(result.raw)
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = {"content": payload.strip()}

            normalized = _normalize_gossip(parsed)

            async with get_db_connection_context() as conn:
                gossip_id = await conn.fetchval(
                    """
                    INSERT INTO social_gossip
                    (user_id, conversation_id, content, truthfulness, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    RETURNING gossip_id
                    """,
                    self.user_id,
                    self.conversation_id,
                    normalized["content"],
                    normalized["truthfulness"],
                )

            return GossipItem(
                gossip_id=gossip_id,
                gossip_type=normalized["type"],
                content=normalized["content"],
                about=target_npcs or [],
                spreaders=set(),
                believers=set(),
                deniers=set(),
                spread_rate=normalized["spread_rate"],
                truthfulness=normalized["truthfulness"],
                impact=normalized["impact"],
            )

        except Exception as e:
            preview = (
                (payload[:240] + "…")
                if isinstance(payload, str) and len(payload) > 240
                else payload
            )
            logger.warning(
                f"Failed to generate gossip (coercion): {e}; payload_preview={preview!r}"
            )
            return self._create_fallback_gossip(target_npcs)

    async def _run_reputation_llm(
        self,
        target_id: int,
        social_circle: Optional[SocialCircle] = None,
    ) -> Dict[ReputationType, float]:
        """Run the reputation analysis prompt (slow)."""

        factors = await self._gather_reputation_factors(target_id)

        prompt = f"""
        Calculate reputation scores based on these factors:

        Target: {"Player" if target_id == self.user_id else f"NPC {target_id}"}
        Recent Actions: {factors.get('actions', [])}
        Gossip About Them: {factors.get('gossip', [])}
        Social Role: {factors.get('role', social_circle.name if social_circle else 'unknown')}
        Relationships: {factors.get('relationships', {})}

        Score each reputation type (0.0-1.0):
        - trustworthy
        - submissive
        - rebellious
        - mysterious
        - influential
        - scandalous
        - nurturing
        - dangerous

        Consider how actions and gossip shape perception.
        Format as JSON with explanations.
        """

        result = await execute(
            LLMRequest(
                prompt=prompt,
                agent=self.social_analyzer,
                metadata={
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stage": "reputation_scoring",
                },
            )
        )

        try:
            payload = extract_runner_response(result.raw)
            result_json = json.loads(payload)

            reputation: Dict[ReputationType, float] = {}
            for rep_type in ReputationType:
                if rep_type.value in result_json:
                    reputation[rep_type] = float(result_json[rep_type.value])
                else:
                    reputation[rep_type] = 0.3

            return reputation

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to calculate reputation: {e}")
            return {rep_type: 0.3 for rep_type in ReputationType}
    
    async def spread_gossip(
        self,
        gossip: GossipItem,
        spreader_id: int,
        listeners: List[int]
    ) -> Dict[str, Any]:
        """Simulate gossip spreading with LLM reactions"""
        
        listener_details = await self._get_npc_social_details(listeners)
        
        prompt = f"""
        Determine how each listener reacts to gossip:
        
        Gossip: "{gossip.content}"
        Truthfulness: {gossip.truthfulness}
        Spreader: NPC {spreader_id}
        Listeners: {listener_details}
        
        For each listener, determine:
        1. Believes/Doubts/Denies
        2. Will they spread it further?
        3. How it affects their opinion of those involved
        4. Their reaction (quote or description)
        
        Consider personality and relationships.
        Format as JSON array.
        """
        
        result = await execute(
            LLMRequest(
                prompt=prompt,
                agent=self.social_analyzer,
                metadata={
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stage": "gossip_reaction",
                },
            )
        )

        try:
            payload = extract_runner_response(result.raw)
            reactions = json.loads(payload)

            spread_results = {
                'new_believers': [],
                'new_deniers': [],
                'new_spreaders': [],
                'reactions': {}
            }
            
            for i, listener_id in enumerate(listeners):
                if i < len(reactions):
                    reaction = reactions[i]
                    
                    if reaction.get('believes'):
                        gossip.believers.add(listener_id)
                        spread_results['new_believers'].append(listener_id)
                    elif reaction.get('denies'):
                        gossip.deniers.add(listener_id)
                        spread_results['new_deniers'].append(listener_id)
                    
                    if reaction.get('will_spread'):
                        gossip.spreaders.add(listener_id)
                        spread_results['new_spreaders'].append(listener_id)
                    
                    spread_results['reactions'][listener_id] = reaction.get(
                        'reaction', 
                        'listens with interest'
                    )
            
            return spread_results
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to process gossip spread: {e}")
            return {'new_believers': [], 'new_deniers': [], 'new_spreaders': [], 'reactions': {}}
    
    async def calculate_reputation_background(
        self,
        target_id: int,
        social_circle: Optional[SocialCircle] = None
    ) -> Dict[str, float]:
        """Run the slow reputation calculation (for background tasks)."""

        reputation = await self._run_reputation_llm(target_id, social_circle)
        await self._store_reputation(target_id, reputation)
        return {rep.value: score for rep, score in reputation.items()}
    
    async def narrate_reputation_change(
        self,
        target_id: int,
        old_reputation: Dict[ReputationType, float],
        new_reputation: Dict[ReputationType, float]
    ) -> str:
        """Generate narrative for reputation change"""
        
        changes = []
        for rep_type in ReputationType:
            delta = new_reputation[rep_type] - old_reputation[rep_type]
            if abs(delta) > 0.1:
                changes.append((rep_type, delta))
        
        if not changes:
            return ""
        
        prompt = f"""
        Narrate a reputation shift:
        
        Major Changes:
        {self._format_reputation_changes(changes)}
        
        Create a 2-3 sentence narrative about how social perception is shifting.
        Focus on the whispers, glances, and subtle social cues.
        Keep it slice-of-life and realistic.
        """
        
        result = await execute(
            LLMRequest(
                prompt=prompt,
                agent=self.reputation_narrator,
                metadata={
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stage": "reputation_narration",
                },
            )
        )
        payload = extract_runner_response(result.raw)
        return payload.strip()
    
    async def form_alliance(
        self,
        initiator_id: int,
        target_id: int,
        reason: str
    ) -> Dict[str, Any]:
        """Form an alliance with LLM-generated terms"""
        
        prompt = f"""
        Generate alliance details:
        
        Initiator: {"Player" if initiator_id == self.user_id else f"NPC {initiator_id}"}
        Target: {"Player" if target_id == self.user_id else f"NPC {target_id}"}
        Reason: {reason}
        
        Create:
        1. Alliance type (mutual support/against common enemy/temporary cooperation)
        2. Terms (what each party offers/expects)
        3. Duration (temporary/until condition/permanent)
        4. Secret or public?
        5. Potential weak points
        
        Make it realistic to social dynamics.
        Format as JSON.
        """
        
        result = await execute(
            LLMRequest(
                prompt=prompt,
                agent=self.alliance_strategist,
                metadata={
                    "operation": LLMOperation.ORCHESTRATION.value,
                    "stage": "alliance_formation",
                },
            )
        )

        try:
            payload = extract_runner_response(result.raw)
            result_json = json.loads(payload)

            async with get_db_connection_context() as conn:
                alliance_id = await conn.fetchval("""
                    INSERT INTO social_alliances
                    (user_id, conversation_id, party1_id, party2_id,
                     alliance_type, terms, is_secret)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING alliance_id
                """, self.user_id, self.conversation_id,
                initiator_id, target_id,
                result_json.get('type', 'cooperation'),
                json.dumps(result_json.get('terms', {})),
                result_json.get('secret', False))

            return {
                'alliance_id': alliance_id,
                'type': result_json.get('type'),
                'terms': result_json.get('terms'),
                'duration': result_json.get('duration'),
                'is_secret': result_json.get('secret', False),
                'weak_points': result_json.get('weak_points', [])
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to form alliance: {e}")
            return {'error': 'Failed to form alliance'}
    
    # Helper methods
    async def _get_npc_social_details(self, npc_ids: List[int]) -> str:
        """Get social details about NPCs for context"""
        
        if not npc_ids:
            return "No specific NPCs"
        
        details = []
        async with get_db_connection_context() as conn:
            for npc_id in npc_ids[:5]:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs WHERE npc_id = $1
                """, npc_id)
                if npc:
                    details.append(f"{npc['name']} ({npc.get('personality_traits', 'unknown')})")
        
        return ", ".join(details) if details else "Unknown NPCs"
    
    async def _gather_reputation_factors(self, target_id: int) -> Dict:
        """Gather factors affecting reputation"""
        
        factors = {'actions': [], 'gossip': [], 'relationships': {}}
        
        async with get_db_connection_context() as conn:
            actions = await conn.fetch("""
                SELECT memory_text FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_id = $3 AND entity_type = 'player'
                ORDER BY created_at DESC LIMIT 10
            """, self.user_id, self.conversation_id, target_id)
            
            factors['actions'] = [a['memory_text'] for a in actions]
            
            gossip = await conn.fetch("""
                SELECT content, truthfulness FROM social_gossip
                WHERE user_id = $1 AND conversation_id = $2
                AND $3 = ANY(about_ids)
                ORDER BY created_at DESC LIMIT 5
            """, self.user_id, self.conversation_id, target_id)
            
            factors['gossip'] = [
                {'content': g['content'], 'truth': g['truthfulness']} 
                for g in gossip
            ]
        
        return factors
    
    async def _store_reputation(
        self,
        target_id: int,
        reputation: Dict[ReputationType, float]
    ):
        """Store reputation scores"""
        
        async with get_db_connection_context() as conn:
            for rep_type, score in reputation.items():
                await conn.execute("""
                    INSERT INTO reputation_scores
                    (user_id, conversation_id, target_id, reputation_type, score)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, conversation_id, target_id, reputation_type)
                    DO UPDATE SET score = $5, updated_at = NOW()
                """, self.user_id, self.conversation_id, target_id,
                rep_type.value, score)
    
    def _format_reputation_changes(self, changes: List[Tuple]) -> str:
        """Format reputation changes for prompt"""
        
        formatted = []
        for rep_type, delta in changes:
            direction = "increased" if delta > 0 else "decreased"
            formatted.append(f"{rep_type.value} {direction} by {abs(delta):.2f}")
        return "\n".join(formatted)
    
    def _create_fallback_gossip(self, target_npcs: List[int]) -> GossipItem:
        """Create fallback gossip if LLM fails"""
        
        return GossipItem(
            gossip_id=0,
            gossip_type=GossipType.RUMOR,
            content="Something interesting was mentioned",
            about=target_npcs or [],
            spreaders=set(),
            believers=set(),
            deniers=set(),
            spread_rate=0.3,
            truthfulness=0.5,
            impact={}
        )
