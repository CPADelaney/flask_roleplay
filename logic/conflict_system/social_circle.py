# logic/conflict_system/social_circle.py
"""
Social Circle Conflict System with LLM-generated dynamics.
Refactored to work as a ConflictSubsystem with the synthesizer.
"""

import logging
import json
import random
import weakref
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
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
            EventType.CONFLICT_RESOLVED
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
        scene_context = event.payload
        present_npcs = scene_context.get('present_npcs', [])
        
        side_effects = []
        
        # Check for gossip opportunities
        if present_npcs and len(present_npcs) >= 2:
            if random.random() < 0.3:  # 30% chance of gossip
                gossip = await self.manager.generate_gossip(
                    {'scene': scene_context.get('activity', 'unknown')},
                    present_npcs[:2]
                )
                
                if gossip:
                    self._active_gossip[gossip.gossip_id] = gossip
                    
                    # Emit gossip event
                    side_effects.append(SystemEvent(
                        event_id=f"gossip_{gossip.gossip_id}",
                        event_type=EventType.NPC_REACTION,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'gossip': gossip.content,
                            'about': gossip.about,
                            'spreaders': list(gossip.spreaders)
                        },
                        priority=7
                    ))
        
        # Update reputation based on recent actions
        for npc_id in present_npcs:
            if npc_id not in self._reputation_cache:
                reputation = await self.manager.calculate_reputation(npc_id)
                self._reputation_cache[npc_id] = reputation
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'gossip_generated': len(side_effects) > 0,
                'reputations_updated': len(present_npcs)
            },
            side_effects=side_effects
        )
    
    async def _handle_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions affecting social dynamics"""
        stakeholder_id = event.payload.get('stakeholder_id')
        action_type = event.payload.get('action_type')
        
        side_effects = []
        
        # Actions can affect reputation
        if action_type in ['betray', 'support', 'oppose']:
            old_reputation = self._reputation_cache.get(stakeholder_id, {})
            new_reputation = await self._adjust_reputation_from_action(
                stakeholder_id, action_type
            )
            
            # Narrate reputation change if significant
            if old_reputation:
                narrative = await self.manager.narrate_reputation_change(
                    stakeholder_id, old_reputation, new_reputation
                )
                
                if narrative:
                    side_effects.append(SystemEvent(
                        event_id=f"reputation_{stakeholder_id}_{datetime.now().timestamp()}",
                        event_type=EventType.NPC_REACTION,
                        source_subsystem=self.subsystem_type,
                        payload={
                            'npc_id': stakeholder_id,
                            'narrative': narrative,
                            'reputation_change': True
                        },
                        priority=8
                    ))
        
        # Check for alliance formation/betrayal
        if action_type == 'ally':
            target_id = event.payload.get('target_id')
            if target_id:
                alliance = await self.manager.form_alliance(
                    stakeholder_id, target_id, 'mutual_benefit'
                )
                
                if alliance:
                    side_effects.append(SystemEvent(
                        event_id=f"alliance_{alliance['alliance_id']}",
                        event_type=EventType.STATE_SYNC,
                        source_subsystem=self.subsystem_type,
                        payload={'alliance': alliance},
                        priority=6
                    ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'social_impact_processed': True},
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
        
        # Generate gossip about the new conflict
        if participants:
            gossip = await self.manager.generate_gossip(
                {'conflict_start': True},
                participants
            )
            
            if gossip:
                self._active_gossip[gossip.gossip_id] = gossip
                
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'gossip_generated': True, 'gossip_id': gossip.gossip_id}
                )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'no_social_dimension': True}
        )
    
    async def _handle_conflict_resolved(self, event: SystemEvent) -> SubsystemResponse:
        """Handle conflict resolution affecting social dynamics"""
        conflict_id = event.payload.get('conflict_id')
        resolution = event.payload.get('resolution', {})
        
        # Resolution affects reputations
        winners = resolution.get('winners', [])
        losers = resolution.get('losers', [])
        
        side_effects = []
        
        for winner_id in winners:
            await self._adjust_reputation_from_action(winner_id, 'victory')
        
        for loser_id in losers:
            await self._adjust_reputation_from_action(loser_id, 'defeat')
        
        # Generate gossip about the resolution
        all_participants = winners + losers
        if all_participants:
            gossip = await self.manager.generate_gossip(
                {'conflict_resolved': True, 'outcome': resolution},
                all_participants
            )
            
            if gossip:
                self._active_gossip[gossip.gossip_id] = gossip
                
                side_effects.append(SystemEvent(
                    event_id=f"resolution_gossip_{gossip.gossip_id}",
                    event_type=EventType.NPC_REACTION,
                    source_subsystem=self.subsystem_type,
                    payload={'gossip': gossip.content},
                    priority=7
                ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'social_aftermath_processed': True},
            side_effects=side_effects
        )
    
    # ========== Helper Methods ==========
    
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
                
                Create gossip that:
                - Feels organic to the social dynamics
                - Has varying levels of truth and exaggeration
                - Reflects power structures and relationships
                - Creates interesting social consequences
                - Ranges from mundane to scandalous
                
                Consider the personalities of spreaders and targets.
                Make gossip feel like real social currency.
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
    
    # [Rest of the SocialCircleManager methods remain the same as in original]
    # Including: generate_gossip, spread_gossip, calculate_reputation,
    # narrate_reputation_change, form_alliance, betray_alliance, etc.
    
    async def generate_gossip(
        self,
        context: Dict[str, Any],
        target_npcs: Optional[List[int]] = None
    ) -> GossipItem:
        """Generate contextual gossip using LLM"""
        
        npc_details = await self._get_npc_social_details(target_npcs or [])
        
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
        
        response = await self.gossip_generator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            async with get_db_connection_context() as conn:
                gossip_id = await conn.fetchval("""
                    INSERT INTO social_gossip
                    (user_id, conversation_id, content, truthfulness, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    RETURNING gossip_id
                """, self.user_id, self.conversation_id,
                result['content'], result.get('truthfulness', 0.5))
            
            return GossipItem(
                gossip_id=gossip_id,
                gossip_type=GossipType[result.get('type', 'RUMOR').upper()],
                content=result['content'],
                about=target_npcs or [],
                spreaders=set(),
                believers=set(),
                deniers=set(),
                spread_rate=result.get('spread_rate', 0.5),
                truthfulness=result.get('truthfulness', 0.5),
                impact=result.get('impact', {})
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate gossip: {e}")
            return self._create_fallback_gossip(target_npcs)
    
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
        
        response = await self.social_analyzer.run(prompt)
        
        try:
            reactions = json.loads(response.content)
            
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
    
    async def calculate_reputation(
        self,
        target_id: int,
        social_circle: Optional[SocialCircle] = None
    ) -> Dict[ReputationType, float]:
        """Calculate reputation using LLM analysis"""
        
        factors = await self._gather_reputation_factors(target_id)
        
        prompt = f"""
        Calculate reputation scores based on these factors:
        
        Target: {"Player" if target_id == self.user_id else f"NPC {target_id}"}
        Recent Actions: {factors.get('actions', [])}
        Gossip About Them: {factors.get('gossip', [])}
        Social Role: {factors.get('role', 'unknown')}
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
        
        response = await self.social_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            reputation = {}
            for rep_type in ReputationType:
                if rep_type.value in result:
                    reputation[rep_type] = float(result[rep_type.value])
                else:
                    reputation[rep_type] = 0.3
            
            await self._store_reputation(target_id, reputation)
            return reputation
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to calculate reputation: {e}")
            return {rep_type: 0.3 for rep_type in ReputationType}
    
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
        
        response = await self.reputation_narrator.run(prompt)
        return response.content.strip()
    
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
        
        response = await self.alliance_strategist.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            async with get_db_connection_context() as conn:
                alliance_id = await conn.fetchval("""
                    INSERT INTO social_alliances
                    (user_id, conversation_id, party1_id, party2_id, 
                     alliance_type, terms, is_secret)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING alliance_id
                """, self.user_id, self.conversation_id,
                initiator_id, target_id,
                result.get('type', 'cooperation'),
                json.dumps(result.get('terms', {})),
                result.get('secret', False))
            
            return {
                'alliance_id': alliance_id,
                'type': result.get('type'),
                'terms': result.get('terms'),
                'duration': result.get('duration'),
                'is_secret': result.get('secret', False),
                'weak_points': result.get('weak_points', [])
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
