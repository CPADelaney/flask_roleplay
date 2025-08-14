# logic/conflict_system/leverage.py
"""
Leverage System with LLM-generated dynamic content.
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
# LEVERAGE STRUCTURES (Preserved from original)
# ===============================================================================

class LeverageType(Enum):
    """Types of leverage in conflicts"""
    INFORMATION = "information"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    MATERIAL = "material"
    POSITIONAL = "positional"
    BEHAVIORAL = "behavioral"
    VULNERABILITY = "vulnerability"
    DEPENDENCY = "dependency"

class LeverageStrength(Enum):
    """Strength levels of leverage"""
    TRIVIAL = 0.2
    MINOR = 0.4
    MODERATE = 0.6
    MAJOR = 0.8
    DECISIVE = 1.0

@dataclass
class LeverageItem:
    """A piece of leverage"""
    leverage_id: int
    leverage_type: LeverageType
    target_id: int
    holder_id: int
    description: str
    strength: float
    evidence: List[str]
    expiration: Optional[datetime]
    uses_remaining: int
    counters: List[str]
    discovery_context: str

@dataclass
class LeverageApplication:
    """An instance of using leverage"""
    application_id: int
    leverage_id: int
    context: str
    demand: str
    threat_level: float
    target_response: str
    success_level: float
    consequences: Dict[str, Any]

@dataclass
class CounterStrategy:
    """A strategy to counter leverage"""
    strategy_id: int
    leverage_id: int
    strategy_type: str
    description: str
    requirements: List[str]
    success_chance: float
    risks: List[str]

# ===============================================================================
# LEVERAGE SUBSYSTEM
# ===============================================================================

class LeverageSubsystem(ConflictSubsystem):
    """
    Leverage subsystem integrated with synthesizer.
    Manages leverage discovery, application, and counter-strategies.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None
        
        # Components
        self.manager = LeverageManager(user_id, conversation_id)
        
        # State tracking
        self._active_leverage: Dict[int, LeverageItem] = {}
        self._leverage_applications: Dict[int, LeverageApplication] = {}
        self._counter_strategies: Dict[int, List[CounterStrategy]] = {}
    
    # ========== ConflictSubsystem Interface Implementation ==========
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.LEVERAGE
    
    @property
    def capabilities(self) -> Set[str]:
        return {
            'leverage_discovery',
            'leverage_application',
            'counter_strategy_generation',
            'power_dynamics',
            'blackmail_management'
        }
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        return {SubsystemType.SOCIAL, SubsystemType.STAKEHOLDER}
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        return {
            EventType.STAKEHOLDER_ACTION,
            EventType.NPC_REACTION,
            EventType.CONFLICT_UPDATED,
            EventType.STATE_SYNC
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
            elif event.event_type == EventType.CONFLICT_UPDATED:
                return await self._handle_conflict_updated(event)
            else:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data={'handled': False}
                )
        except Exception as e:
            logger.error(f"LeverageSubsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of leverage subsystem"""
        try:
            active_leverage = len(self._active_leverage)
            expired_leverage = sum(
                1 for l in self._active_leverage.values()
                if l.expiration and l.expiration < datetime.now()
            )
            
            return {
                'healthy': expired_leverage < active_leverage / 2,
                'active_leverage': active_leverage,
                'expired_leverage': expired_leverage,
                'recent_applications': len(self._leverage_applications)
            }
        except Exception as e:
            return {'healthy': False, 'issue': str(e)}
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get leverage-specific conflict data"""
        
        # Find leverage related to this conflict
        conflict_leverage = []
        for leverage in self._active_leverage.values():
            # Check if leverage is relevant to conflict participants
            # This would need actual conflict participant lookup
            conflict_leverage.append(leverage.leverage_id)
        
        return {
            'subsystem': 'leverage',
            'related_leverage': len(conflict_leverage),
            'power_dynamics': self._analyze_power_dynamics()
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of leverage subsystem"""
        return {
            'active_leverage_items': len(self._active_leverage),
            'total_applications': len(self._leverage_applications),
            'counter_strategies_available': sum(
                len(strategies) for strategies in self._counter_strategies.values()
            )
        }
    
    # ========== Event Handlers ==========
    
    async def _handle_state_sync(self, event: SystemEvent) -> SubsystemResponse:
        """Handle scene state synchronization"""
        scene_context = event.payload
        present_npcs = scene_context.get('present_npcs', [])
        
        side_effects = []
        
        # Check for leverage discovery opportunities
        if present_npcs and random.random() < 0.2:  # 20% chance
            observer_id = self.user_id  # Player observing
            target_id = random.choice(present_npcs)
            
            leverage = await self.manager.discover_leverage(
                observer_id, target_id, scene_context
            )
            
            if leverage:
                self._active_leverage[leverage.leverage_id] = leverage
                
                side_effects.append(SystemEvent(
                    event_id=f"leverage_discovered_{leverage.leverage_id}",
                    event_type=EventType.INTENSITY_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'leverage_id': leverage.leverage_id,
                        'type': leverage.leverage_type.value,
                        'strength': leverage.strength,
                        'discovery_narrative': f"You notice something about the situation that could be useful..."
                    },
                    priority=6
                ))
        
        # Check for expired leverage
        expired = []
        for lid, leverage in self._active_leverage.items():
            if leverage.expiration and leverage.expiration < datetime.now():
                expired.append(lid)
        
        for lid in expired:
            del self._active_leverage[lid]
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={
                'leverage_discovered': len(side_effects) > 0,
                'expired_leverage': len(expired)
            },
            side_effects=side_effects
        )
    
    async def _handle_stakeholder_action(self, event: SystemEvent) -> SubsystemResponse:
        """Handle stakeholder actions that might involve leverage"""
        stakeholder_id = event.payload.get('stakeholder_id')
        action_type = event.payload.get('action_type')
        target_id = event.payload.get('target_id')
        
        side_effects = []
        
        # Check if stakeholder has leverage they might use
        if action_type in ['threaten', 'demand', 'negotiate']:
            relevant_leverage = self._find_relevant_leverage(stakeholder_id, target_id)
            
            if relevant_leverage:
                # Auto-apply leverage in the action
                application = await self.manager.apply_leverage(
                    relevant_leverage.leverage_id,
                    event.payload.get('demand', 'compliance'),
                    0.5  # Medium threat level
                )
                
                self._leverage_applications[application.application_id] = application
                
                # Emit consequence event
                side_effects.append(SystemEvent(
                    event_id=f"leverage_applied_{application.application_id}",
                    event_type=EventType.INTENSITY_CHANGED,
                    source_subsystem=self.subsystem_type,
                    payload={
                        'leverage_used': True,
                        'success': application.success_level > 0.5,
                        'target_response': application.target_response,
                        'consequences': application.consequences
                    },
                    priority=4
                ))
                
                # Generate counter-strategies for target
                if target_id == self.user_id:  # Player is target
                    strategies = await self.manager.generate_counter_strategies(
                        relevant_leverage.leverage_id,
                        {'player_resources': event.payload.get('player_resources', {})}
                    )
                    
                    self._counter_strategies[relevant_leverage.leverage_id] = strategies
                    
                    if strategies:
                        side_effects.append(SystemEvent(
                            event_id=f"counter_available_{relevant_leverage.leverage_id}",
                            event_type=EventType.PLAYER_CHOICE,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'choice_type': 'counter_leverage',
                                'strategies': [
                                    {
                                        'id': s.strategy_id,
                                        'type': s.strategy_type,
                                        'description': s.description,
                                        'success_chance': s.success_chance
                                    }
                                    for s in strategies[:3]
                                ]
                            },
                            priority=3
                        ))
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'leverage_processed': len(side_effects) > 0},
            side_effects=side_effects
        )
    
    async def _handle_npc_reaction(self, event: SystemEvent) -> SubsystemResponse:
        """Handle NPC reactions that might reveal leverage"""
        npc_id = event.payload.get('npc_id')
        reaction_type = event.payload.get('reaction_type')
        
        # Certain reactions might reveal vulnerabilities
        if reaction_type in ['nervous', 'defensive', 'secretive']:
            # Chance to discover leverage
            if random.random() < 0.3:
                leverage = await self.manager.discover_leverage(
                    self.user_id, npc_id,
                    {'reaction_observed': reaction_type}
                )
                
                if leverage:
                    self._active_leverage[leverage.leverage_id] = leverage
                    
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={
                            'leverage_discovered': True,
                            'leverage_type': leverage.leverage_type.value
                        }
                    )
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'reaction_noted': True}
        )
    
    async def _handle_conflict_updated(self, event: SystemEvent) -> SubsystemResponse:
        """Handle conflict updates that might affect leverage"""
        conflict_id = event.payload.get('conflict_id')
        update_type = event.payload.get('update_type')
        
        # Conflict progression might reveal or invalidate leverage
        if update_type == 'escalation':
            # Higher stakes might reveal more leverage
            for leverage in list(self._active_leverage.values()):
                # Increase strength of relevant leverage
                leverage.strength = min(1.0, leverage.strength + 0.1)
        
        elif update_type == 'de_escalation':
            # Lower stakes might reduce leverage effectiveness
            for leverage in list(self._active_leverage.values()):
                leverage.strength = max(0.0, leverage.strength - 0.1)
        
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={'leverage_adjusted': True}
        )
    
    # ========== Helper Methods ==========
    
    def _find_relevant_leverage(
        self,
        holder_id: int,
        target_id: Optional[int]
    ) -> Optional[LeverageItem]:
        """Find leverage that holder has over target"""
        for leverage in self._active_leverage.values():
            if leverage.holder_id == holder_id:
                if target_id is None or leverage.target_id == target_id:
                    if leverage.uses_remaining > 0:
                        return leverage
        return None
    
    def _analyze_power_dynamics(self) -> Dict[str, Any]:
        """Analyze power dynamics based on leverage"""
        dynamics = {
            'player_leverage': 0,
            'npc_leverage': 0,
            'balance': 'neutral'
        }
        
        for leverage in self._active_leverage.values():
            if leverage.holder_id == self.user_id:
                dynamics['player_leverage'] += leverage.strength
            else:
                dynamics['npc_leverage'] += leverage.strength
        
        if dynamics['player_leverage'] > dynamics['npc_leverage'] + 0.3:
            dynamics['balance'] = 'player_advantage'
        elif dynamics['npc_leverage'] > dynamics['player_leverage'] + 0.3:
            dynamics['balance'] = 'npc_advantage'
        
        return dynamics

# ===============================================================================
# ORIGINAL MANAGER CLASS (Preserved with synthesizer integration)
# ===============================================================================

class LeverageManager:
    """
    Manages leverage dynamics using LLM for intelligent generation.
    Modified to work with synthesizer.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.synthesizer = None  # Will be set by subsystem
        self._discovery_agent = None
        self._application_strategist = None
        self._counter_strategist = None
        self._consequence_narrator = None
    
    @property
    def discovery_agent(self) -> Agent:
        """Agent for discovering leverage"""
        if self._discovery_agent is None:
            self._discovery_agent = Agent(
                name="Leverage Discovery Agent",
                instructions="""
                Discover leverage based on observations and patterns.
                
                Look for:
                - Hidden vulnerabilities and dependencies
                - Secrets that could be exposed
                - Behavioral patterns that can be exploited
                - Emotional attachments that create pressure points
                - Social connections that matter
                
                Generate leverage that feels:
                - Organic to the relationships
                - Proportional (not too powerful)
                - Grounded in established facts
                - Interesting for gameplay
                
                Consider both obvious and subtle forms of leverage.
                """,
                model="gpt-5-nano",
            )
        return self._discovery_agent
    
    @property
    def application_strategist(self) -> Agent:
        """Agent for applying leverage strategically"""
        if self._application_strategist is None:
            self._application_strategist = Agent(
                name="Leverage Application Strategist",
                instructions="""
                Strategize how to apply leverage effectively.
                
                Consider:
                - The relationship between parties
                - The severity of the demand
                - The strength of the leverage
                - Potential backfire risks
                - Long-term consequences
                
                Generate applications that are:
                - Psychologically realistic
                - Proportional to the leverage strength
                - Contextually appropriate
                - Interesting narratively
                
                Balance effectiveness with relationship preservation.
                """,
                model="gpt-5-nano",
            )
        return self._application_strategist
    
    @property
    def counter_strategist(self) -> Agent:
        """Agent for generating counter-strategies"""
        if self._counter_strategist is None:
            self._counter_strategist = Agent(
                name="Counter-Strategy Developer",
                instructions="""
                Develop strategies to counter leverage.
                
                Consider approaches like:
                - Denial and discrediting
                - Deflection and misdirection  
                - Destroying the evidence
                - Reversing the leverage
                - Accepting and minimizing damage
                - Creating mutual destruction scenarios
                
                Generate counters that are:
                - Clever but realistic
                - Appropriate to character capabilities
                - Risk-aware
                - Dramatically interesting
                
                Consider both immediate and long-term strategies.
                """,
                model="gpt-5-nano",
            )
        return self._counter_strategist
    
    @property
    def consequence_narrator(self) -> Agent:
        """Agent for narrating leverage consequences"""
        if self._consequence_narrator is None:
            self._consequence_narrator = Agent(
                name="Leverage Consequence Narrator",
                instructions="""
                Narrate the consequences of leverage use.
                
                Focus on:
                - Immediate emotional impacts
                - Relationship shifts
                - Power dynamic changes
                - Unintended consequences
                - Long-term ramifications
                
                Create narratives that show:
                - The cost of using leverage
                - How it changes relationships
                - Ripple effects through social circles
                - Character growth or degradation
                
                Keep consequences proportional and realistic.
                """,
                model="gpt-5-nano",
            )
        return self._consequence_narrator
    
    # [Rest of LeverageManager methods remain mostly the same]
    # Including: discover_leverage, apply_leverage, generate_counter_strategies,
    # execute_counter_strategy, narrate_leverage_consequences, etc.
    
    async def discover_leverage(
        self,
        observer_id: int,
        target_id: int,
        context: Dict[str, Any]
    ) -> Optional[LeverageItem]:
        """Discover leverage through observation and analysis"""
        
        target_info = await self._gather_target_information(target_id)
        
        prompt = f"""
        Analyze for potential leverage:
        
        Observer: {"Player" if observer_id == self.user_id else f"NPC {observer_id}"}
        Target: {"Player" if target_id == self.user_id else f"NPC {target_id}"}
        
        Target Information:
        {json.dumps(target_info, indent=2)}
        
        Context:
        {json.dumps(context, indent=2)}
        
        Identify:
        1. Type of leverage (information/emotional/social/material/etc)
        2. Specific description (what exactly is the leverage)
        3. Strength (0.0-1.0, be conservative)
        4. Evidence/proof available
        5. How it could be countered
        6. Expiration conditions (when it becomes useless)
        
        Only identify leverage that:
        - Is supported by the information
        - Feels realistic and proportional
        - Would be interesting to use
        
        Format as JSON. Return null if no good leverage found.
        """
        
        response = await self.discovery_agent.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            if result and result != 'null':
                async with get_db_connection_context() as conn:
                    leverage_id = await conn.fetchval("""
                        INSERT INTO leverage_items
                        (user_id, conversation_id, holder_id, target_id,
                         leverage_type, description, strength, evidence,
                         discovery_context, uses_remaining)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING leverage_id
                    """, self.user_id, self.conversation_id,
                    observer_id, target_id,
                    result['type'], result['description'],
                    float(result['strength']), json.dumps(result.get('evidence', [])),
                    json.dumps(context), 3)
                
                # Emit discovery event through synthesizer if available
                if self.synthesizer and self.synthesizer():
                    synth = self.synthesizer()
                    await synth.emit_event(SystemEvent(
                        event_id=f"leverage_internal_{leverage_id}",
                        event_type=EventType.STATE_SYNC,
                        source_subsystem=SubsystemType.LEVERAGE,
                        payload={'leverage_discovered': leverage_id}
                    ))
                
                return LeverageItem(
                    leverage_id=leverage_id,
                    leverage_type=LeverageType[result['type'].upper()],
                    target_id=target_id,
                    holder_id=observer_id,
                    description=result['description'],
                    strength=float(result['strength']),
                    evidence=result.get('evidence', []),
                    expiration=self._calculate_expiration(result.get('expiration')),
                    uses_remaining=3,
                    counters=result.get('counters', []),
                    discovery_context=str(context)
                )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to discover leverage: {e}")
        
        return None
    
    async def apply_leverage(
        self,
        leverage_id: int,
        demand: str,
        threat_level: float = 0.5
    ) -> LeverageApplication:
        """Apply leverage to achieve a goal"""
        
        leverage = await self._get_leverage_details(leverage_id)
        if not leverage:
            raise ValueError(f"Leverage {leverage_id} not found")
        
        if leverage['uses_remaining'] <= 0:
            return self._create_failed_application("Leverage exhausted")
        
        prompt = f"""
        Strategize leverage application:
        
        Leverage: {leverage['description']}
        Type: {leverage['leverage_type']}
        Strength: {leverage['strength']}
        Target: {"Player" if leverage['target_id'] == self.user_id else f"NPC {leverage['target_id']}"}
        
        Demand: {demand}
        Threat Level: {threat_level} (0=subtle, 1=aggressive)
        
        Determine:
        1. How to present the leverage (exact approach)
        2. Target's likely response
        3. Success probability (0.0-1.0)
        4. Immediate consequences
        5. Relationship impact (-1.0 to 1.0)
        6. Potential backfire risks
        
        Consider personality and relationship dynamics.
        Format as JSON.
        """
        
        response = await self.application_strategist.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            success = random.random() < float(result.get('success_probability', 0.5))
            
            async with get_db_connection_context() as conn:
                application_id = await conn.fetchval("""
                    INSERT INTO leverage_applications
                    (user_id, conversation_id, leverage_id, demand,
                     threat_level, success, consequences)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING application_id
                """, self.user_id, self.conversation_id,
                leverage_id, demand, threat_level, success,
                json.dumps(result.get('consequences', {})))
                
                await conn.execute("""
                    UPDATE leverage_items
                    SET uses_remaining = uses_remaining - 1
                    WHERE leverage_id = $1
                """, leverage_id)
            
            return LeverageApplication(
                application_id=application_id,
                leverage_id=leverage_id,
                context=result.get('approach', 'Direct application'),
                demand=demand,
                threat_level=threat_level,
                target_response=result.get('target_response', 'Compliance'),
                success_level=1.0 if success else 0.3,
                consequences=result.get('consequences', {})
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to apply leverage: {e}")
            return self._create_failed_application("Application failed")
    
    async def generate_counter_strategies(
        self,
        leverage_id: int,
        defender_resources: Dict[str, Any]
    ) -> List[CounterStrategy]:
        """Generate strategies to counter leverage"""
        
        leverage = await self._get_leverage_details(leverage_id)
        if not leverage:
            return []
        
        prompt = f"""
        Generate counter-strategies for this leverage:
        
        Leverage: {leverage['description']}
        Type: {leverage['leverage_type']}
        Strength: {leverage['strength']}
        Evidence: {leverage.get('evidence', [])}
        
        Defender Resources:
        {json.dumps(defender_resources, indent=2)}
        
        Generate 2-4 counter-strategies:
        For each strategy provide:
        1. Type (deny/deflect/destroy/reverse/accept)
        2. Description (specific approach)
        3. Requirements (what's needed)
        4. Success chance (0.0-1.0, be realistic)
        5. Risks (what could go wrong)
        
        Make strategies varied and interesting.
        Format as JSON array.
        """
        
        response = await self.counter_strategist.run(prompt)
        
        try:
            strategies_data = json.loads(response.content)
            
            strategies = []
            for s_data in strategies_data:
                async with get_db_connection_context() as conn:
                    strategy_id = await conn.fetchval("""
                        INSERT INTO counter_strategies
                        (user_id, conversation_id, leverage_id,
                         strategy_type, description, success_chance)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING strategy_id
                    """, self.user_id, self.conversation_id,
                    leverage_id, s_data['type'], s_data['description'],
                    float(s_data['success_chance']))
                
                strategies.append(CounterStrategy(
                    strategy_id=strategy_id,
                    leverage_id=leverage_id,
                    strategy_type=s_data['type'],
                    description=s_data['description'],
                    requirements=s_data.get('requirements', []),
                    success_chance=float(s_data['success_chance']),
                    risks=s_data.get('risks', [])
                ))
            
            return strategies
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to generate counter-strategies: {e}")
            return []
    
    # Helper methods
    async def _gather_target_information(self, target_id: int) -> Dict:
        """Gather information about a target for leverage discovery"""
        
        info = {}
        
        async with get_db_connection_context() as conn:
            if target_id == self.user_id:
                info['type'] = 'player'
            else:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs
                    WHERE npc_id = $1
                """, target_id)
                info['type'] = 'npc'
                info['traits'] = npc.get('personality_traits', '') if npc else ''
            
            memories = await conn.fetch("""
                SELECT memory_text, emotional_valence
                FROM enhanced_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_id = $3
                ORDER BY created_at DESC LIMIT 10
            """, self.user_id, self.conversation_id, target_id)
            
            info['recent_events'] = [
                {'text': m['memory_text'], 'emotion': m['emotional_valence']}
                for m in memories
            ]
            
            relationships = await conn.fetch("""
                SELECT dimension, current_value
                FROM relationship_dimensions
                WHERE user_id = $1 AND conversation_id = $2
                AND (entity1_id = $3 OR entity2_id = $3)
            """, self.user_id, self.conversation_id, target_id)
            
            info['relationships'] = {
                r['dimension']: r['current_value']
                for r in relationships
            }
        
        return info
    
    async def _get_leverage_details(self, leverage_id: int) -> Optional[Dict]:
        """Get details of a leverage item"""
        
        async with get_db_connection_context() as conn:
            leverage = await conn.fetchrow("""
                SELECT * FROM leverage_items
                WHERE leverage_id = $1
            """, leverage_id)
        
        return dict(leverage) if leverage else None
    
    def _calculate_expiration(self, expiration_data: Any) -> Optional[datetime]:
        """Calculate when leverage expires"""
        
        if not expiration_data:
            return None
        
        if isinstance(expiration_data, str):
            if 'days' in expiration_data:
                days = int(''.join(filter(str.isdigit, expiration_data)) or 7)
                return datetime.now() + timedelta(days=days)
        
        return None
    
    def _create_failed_application(self, reason: str) -> LeverageApplication:
        """Create a failed application result"""
        
        return LeverageApplication(
            application_id=0,
            leverage_id=0,
            context=reason,
            demand="",
            threat_level=0,
            target_response="Rejection",
            success_level=0.0,
            consequences={'failure_reason': reason}
        )
