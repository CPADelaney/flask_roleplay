# logic/conflict_system/social_circle.py
"""
Social Circle Conflict System with LLM-generated dynamics.
Manages social relationships, gossip, reputation, and group dynamics.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# SOCIAL STRUCTURES
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
    members: List[int]  # NPC IDs
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
    about: List[int]  # Who it's about (NPC/Player IDs)
    spreaders: Set[int]  # Who's spreading it
    believers: Set[int]  # Who believes it
    deniers: Set[int]  # Who denies it
    spread_rate: float  # How fast it spreads
    truthfulness: float  # How true it is (0-1)
    impact: Dict[str, Any]  # Effects on relationships/reputation

@dataclass
class SocialConflict:
    """A conflict within social dynamics"""
    conflict_id: int
    conflict_type: str  # "exclusion", "rivalry", "scandal", etc.
    participants: List[int]
    stakes: str
    current_phase: str
    alliances: Dict[int, List[int]]  # Who sides with whom
    public_opinion: Dict[int, float]  # Support levels

# ===============================================================================
# SOCIAL CIRCLE MANAGER WITH LLM
# ===============================================================================

class SocialCircleManager:
    """
    Manages social dynamics using LLM for dynamic generation.
    Handles gossip, reputation, alliances, and social conflicts.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
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
    
    # ========== Gossip System ==========
    
    async def generate_gossip(
        self,
        context: Dict[str, Any],
        target_npcs: Optional[List[int]] = None
    ) -> GossipItem:
        """Generate contextual gossip using LLM"""
        
        # Get NPC details for richer gossip
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
            
            # Create gossip item
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
        
        # Get personalities to determine reactions
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
            
            # Update database
            await self._update_gossip_spread(gossip)
            
            return spread_results
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to process gossip spread: {e}")
            return {'new_believers': [], 'new_deniers': [], 'new_spreaders': [], 'reactions': {}}
    
    # ========== Reputation System ==========
    
    async def calculate_reputation(
        self,
        target_id: int,
        social_circle: Optional[SocialCircle] = None
    ) -> Dict[ReputationType, float]:
        """Calculate reputation using LLM analysis"""
        
        # Gather reputation factors
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
                    reputation[rep_type] = 0.3  # Neutral default
            
            # Store reputation
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
        
        # Find biggest changes
        changes = []
        for rep_type in ReputationType:
            delta = new_reputation[rep_type] - old_reputation[rep_type]
            if abs(delta) > 0.1:
                changes.append((rep_type, delta))
        
        if not changes:
            return "Your reputation remains stable."
        
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
    
    # ========== Alliance System ==========
    
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
            
            # Store alliance
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
    
    async def betray_alliance(
        self,
        betrayer_id: int,
        alliance_id: int,
        reason: str
    ) -> Dict[str, Any]:
        """Process alliance betrayal with consequences"""
        
        # Get alliance details
        alliance = await self._get_alliance_details(alliance_id)
        if not alliance:
            return {'error': 'Alliance not found'}
        
        prompt = f"""
        Generate betrayal consequences:
        
        Betrayer: {"Player" if betrayer_id == self.user_id else f"NPC {betrayer_id}"}
        Alliance Type: {alliance.get('type')}
        Reason for Betrayal: {reason}
        Was Secret: {alliance.get('is_secret')}
        
        Determine:
        1. Immediate consequences
        2. Reputation impact (specific changes)
        3. How others react (allies, neutrals, enemies)
        4. Long-term social costs
        5. Any unexpected benefits
        
        Consider the social dynamics of betrayal.
        Format as JSON.
        """
        
        response = await self.alliance_strategist.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Update alliance status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE social_alliances
                    SET status = 'betrayed', betrayer_id = $1, betrayal_reason = $2
                    WHERE alliance_id = $3
                """, betrayer_id, reason, alliance_id)
            
            # Process reputation impact
            if result.get('reputation_impact'):
                await self._apply_reputation_changes(
                    betrayer_id, 
                    result['reputation_impact']
                )
            
            return {
                'immediate_consequences': result.get('immediate_consequences', []),
                'reputation_changes': result.get('reputation_impact', {}),
                'social_reactions': result.get('reactions', {}),
                'long_term_costs': result.get('long_term_costs', []),
                'unexpected_benefits': result.get('benefits', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to process betrayal: {e}")
            return {'error': 'Failed to process betrayal'}
    
    # ========== Social Conflict System ==========
    
    async def generate_social_conflict(
        self,
        participants: List[int],
        conflict_seed: str
    ) -> SocialConflict:
        """Generate a social conflict using LLM"""
        
        participant_details = await self._get_npc_social_details(participants)
        
        prompt = f"""
        Generate a social conflict:
        
        Participants: {participant_details}
        Seed: {conflict_seed}
        
        Create:
        1. Conflict type (exclusion/rivalry/scandal/power struggle/ideological)
        2. What's at stake (be specific)
        3. Initial alliances (who sides with whom)
        4. Public opinion split
        5. Possible escalation path
        
        Keep it grounded in social dynamics, not physical conflict.
        Format as JSON.
        """
        
        response = await self.social_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Store conflict
            async with get_db_connection_context() as conn:
                conflict_id = await conn.fetchval("""
                    INSERT INTO social_conflicts
                    (user_id, conversation_id, conflict_type, stakes, current_phase)
                    VALUES ($1, $2, $3, $4, 'emerging')
                    RETURNING conflict_id
                """, self.user_id, self.conversation_id,
                result.get('type', 'rivalry'),
                result.get('stakes', 'social standing'))
            
            return SocialConflict(
                conflict_id=conflict_id,
                conflict_type=result.get('type', 'rivalry'),
                participants=participants,
                stakes=result.get('stakes', 'social standing'),
                current_phase='emerging',
                alliances=result.get('alliances', {}),
                public_opinion=result.get('public_opinion', {})
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate social conflict: {e}")
            return self._create_fallback_social_conflict(participants)
    
    # ========== Helper Methods ==========
    
    async def _get_npc_social_details(self, npc_ids: List[int]) -> str:
        """Get social details about NPCs for context"""
        
        if not npc_ids:
            return "No specific NPCs"
        
        details = []
        async with get_db_connection_context() as conn:
            for npc_id in npc_ids[:5]:  # Limit for prompt size
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
            # Get recent actions
            actions = await conn.fetch("""
                SELECT memory_text FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_id = $3 AND entity_type = 'player'
                ORDER BY created_at DESC LIMIT 10
            """, self.user_id, self.conversation_id, target_id)
            
            factors['actions'] = [a['memory_text'] for a in actions]
            
            # Get gossip
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
    
    def _create_fallback_social_conflict(self, participants: List[int]) -> SocialConflict:
        """Create fallback conflict if LLM fails"""
        
        return SocialConflict(
            conflict_id=0,
            conflict_type="disagreement",
            participants=participants,
            stakes="social standing",
            current_phase="emerging",
            alliances={},
            public_opinion={}
        )

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def create_gossip(
    ctx: RunContextWrapper,
    about_npcs: Optional[List[int]] = None,
    context_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new piece of gossip"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SocialCircleManager(user_id, conversation_id)
    
    context = {'hint': context_hint} if context_hint else {}
    gossip = await manager.generate_gossip(context, about_npcs)
    
    return {
        'gossip_id': gossip.gossip_id,
        'type': gossip.gossip_type.value,
        'content': gossip.content,
        'about': gossip.about,
        'truthfulness': gossip.truthfulness,
        'impact': gossip.impact
    }

@function_tool
async def spread_gossip_to_npcs(
    ctx: RunContextWrapper,
    gossip_id: int,
    spreader_npc: int,
    listener_npcs: List[int]
) -> Dict[str, Any]:
    """Spread gossip to NPCs and get their reactions"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SocialCircleManager(user_id, conversation_id)
    
    # Get gossip item
    async with get_db_connection_context() as conn:
        gossip_data = await conn.fetchrow("""
            SELECT * FROM social_gossip WHERE gossip_id = $1
        """, gossip_id)
    
    if not gossip_data:
        return {'error': 'Gossip not found'}
    
    gossip = GossipItem(
        gossip_id=gossip_id,
        gossip_type=GossipType.RUMOR,
        content=gossip_data['content'],
        about=[],
        spreaders=set(),
        believers=set(),
        deniers=set(),
        spread_rate=0.5,
        truthfulness=gossip_data['truthfulness'],
        impact={}
    )
    
    results = await manager.spread_gossip(gossip, spreader_npc, listener_npcs)
    return results

@function_tool
async def check_reputation(
    ctx: RunContextWrapper,
    target_id: Optional[int] = None
) -> Dict[str, Any]:
    """Check reputation scores"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SocialCircleManager(user_id, conversation_id)
    
    # Default to checking player's reputation
    if target_id is None:
        target_id = user_id
    
    reputation = await manager.calculate_reputation(target_id)
    
    # Format for response
    scores = {rep_type.value: score for rep_type, score in reputation.items()}
    
    # Find dominant reputation
    dominant = max(reputation.items(), key=lambda x: x[1])
    
    return {
        'target': "Player" if target_id == user_id else f"NPC {target_id}",
        'reputation_scores': scores,
        'dominant_reputation': dominant[0].value,
        'dominant_score': dominant[1]
    }

@function_tool
async def form_social_alliance(
    ctx: RunContextWrapper,
    with_npc: int,
    reason: str
) -> Dict[str, Any]:
    """Form an alliance with an NPC"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SocialCircleManager(user_id, conversation_id)
    
    result = await manager.form_alliance(user_id, with_npc, reason)
    return result

@function_tool
async def create_social_conflict(
    ctx: RunContextWrapper,
    involved_npcs: List[int],
    conflict_seed: str
) -> Dict[str, Any]:
    """Create a social conflict"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = SocialCircleManager(user_id, conversation_id)
    
    # Include player in participants
    participants = [user_id] + involved_npcs
    
    conflict = await manager.generate_social_conflict(participants, conflict_seed)
    
    return {
        'conflict_id': conflict.conflict_id,
        'type': conflict.conflict_type,
        'participants': conflict.participants,
        'stakes': conflict.stakes,
        'phase': conflict.current_phase,
        'alliances': conflict.alliances,
        'public_opinion': conflict.public_opinion
    }
