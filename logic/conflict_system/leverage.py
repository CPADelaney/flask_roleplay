# logic/conflict_system/leverage.py
"""
Leverage System with LLM-generated dynamic content.
Manages leverage discovery, application, and counter-strategies in conflicts.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# LEVERAGE STRUCTURES
# ===============================================================================

class LeverageType(Enum):
    """Types of leverage in conflicts"""
    INFORMATION = "information"  # Secrets, knowledge
    EMOTIONAL = "emotional"  # Feelings, attachments
    SOCIAL = "social"  # Reputation, relationships
    MATERIAL = "material"  # Resources, possessions
    POSITIONAL = "positional"  # Authority, access
    BEHAVIORAL = "behavioral"  # Habits, patterns
    VULNERABILITY = "vulnerability"  # Weaknesses, fears
    DEPENDENCY = "dependency"  # Needs, addictions

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
    target_id: int  # Who it's leverage over
    holder_id: int  # Who holds the leverage
    description: str
    strength: float  # 0.0-1.0
    evidence: List[str]  # Supporting evidence/proof
    expiration: Optional[datetime]  # When it becomes useless
    uses_remaining: int  # How many times it can be used
    counters: List[str]  # Known counter-strategies
    discovery_context: str  # How it was discovered

@dataclass
class LeverageApplication:
    """An instance of using leverage"""
    application_id: int
    leverage_id: int
    context: str  # Why it's being used
    demand: str  # What's being asked for
    threat_level: float  # How aggressively it's used
    target_response: str  # How target responds
    success_level: float  # How effective it was
    consequences: Dict[str, Any]  # Results of use

@dataclass
class CounterStrategy:
    """A strategy to counter leverage"""
    strategy_id: int
    leverage_id: int
    strategy_type: str  # "deny", "deflect", "destroy", "reverse"
    description: str
    requirements: List[str]  # What's needed to execute
    success_chance: float  # Probability of working
    risks: List[str]  # What could go wrong

# ===============================================================================
# LEVERAGE MANAGER WITH LLM
# ===============================================================================

class LeverageManager:
    """
    Manages leverage dynamics using LLM for intelligent generation.
    Handles discovery, application, and counter-strategies.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
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
    
    # ========== Leverage Discovery ==========
    
    async def discover_leverage(
        self,
        observer_id: int,
        target_id: int,
        context: Dict[str, Any]
    ) -> Optional[LeverageItem]:
        """Discover leverage through observation and analysis"""
        
        # Gather information about target
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
                # Store leverage
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
                    json.dumps(context), 3)  # Default 3 uses
                
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
    
    # ========== Leverage Application ==========
    
    async def apply_leverage(
        self,
        leverage_id: int,
        demand: str,
        threat_level: float = 0.5
    ) -> LeverageApplication:
        """Apply leverage to achieve a goal"""
        
        # Get leverage details
        leverage = await self._get_leverage_details(leverage_id)
        if not leverage:
            raise ValueError(f"Leverage {leverage_id} not found")
        
        # Check if leverage is still valid
        if leverage['uses_remaining'] <= 0:
            return self._create_failed_application("Leverage exhausted")
        
        # Generate application strategy
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
            
            # Determine success
            success = random.random() < float(result.get('success_probability', 0.5))
            
            # Create application record
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
                
                # Reduce uses remaining
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
    
    # ========== Counter-Strategies ==========
    
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
                # Store strategy
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
    
    async def execute_counter_strategy(
        self,
        strategy_id: int,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a counter-strategy"""
        
        # Get strategy details
        strategy = await self._get_strategy_details(strategy_id)
        if not strategy:
            return {'error': 'Strategy not found'}
        
        # Check requirements
        requirements_met = await self._check_requirements(
            strategy['requirements'],
            execution_context
        )
        
        if not requirements_met:
            return {
                'success': False,
                'reason': 'Requirements not met',
                'missing': strategy['requirements']
            }
        
        # Determine success with randomness
        base_chance = strategy['success_chance']
        modified_chance = self._modify_success_chance(base_chance, execution_context)
        success = random.random() < modified_chance
        
        # Generate consequences
        prompt = f"""
        Narrate counter-strategy execution:
        
        Strategy Type: {strategy['strategy_type']}
        Description: {strategy['description']}
        Success: {success}
        Context: {json.dumps(execution_context, indent=2)}
        
        Generate:
        1. How it plays out (2-3 sentences)
        2. If successful: How the leverage is neutralized
        3. If failed: What went wrong and consequences
        4. Impact on relationships
        5. Any unexpected outcomes
        
        Keep it grounded and realistic.
        Format as JSON.
        """
        
        response = await self.consequence_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Update leverage status if successful
            if success:
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE leverage_items
                        SET is_neutralized = true
                        WHERE leverage_id = $1
                    """, strategy['leverage_id'])
            
            return {
                'success': success,
                'narrative': result.get('narrative', 'The counter-strategy is executed'),
                'leverage_status': 'neutralized' if success else 'active',
                'relationship_impact': result.get('relationship_impact', {}),
                'unexpected_outcomes': result.get('unexpected', [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to execute counter-strategy: {e}")
            return {
                'success': success,
                'narrative': 'The counter-strategy plays out',
                'leverage_status': 'neutralized' if success else 'active'
            }
    
    # ========== Consequence System ==========
    
    async def narrate_leverage_consequences(
        self,
        application: LeverageApplication,
        long_term: bool = False
    ) -> str:
        """Generate narrative for leverage consequences"""
        
        prompt = f"""
        Narrate the {'long-term' if long_term else 'immediate'} consequences of leverage use:
        
        Leverage Applied: {application.context}
        Demand: {application.demand}
        Threat Level: {application.threat_level}
        Success: {application.success_level > 0.5}
        Target Response: {application.target_response}
        
        Create a narrative (2-4 sentences) that shows:
        {'- How relationships have permanently shifted' if long_term else '- The immediate emotional impact'}
        {'- Trust that may never return' if long_term else '- The power dynamic shift'}
        {'- Social ripple effects' if long_term else '- How others react'}
        
        Focus on the human cost of using leverage.
        Keep it realistic and emotionally grounded.
        """
        
        response = await self.consequence_narrator.run(prompt)
        return response.content.strip()
    
    # ========== Helper Methods ==========
    
    async def _gather_target_information(self, target_id: int) -> Dict:
        """Gather information about a target for leverage discovery"""
        
        info = {}
        
        async with get_db_connection_context() as conn:
            # Get basic info
            if target_id == self.user_id:
                info['type'] = 'player'
            else:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs
                    WHERE npc_id = $1
                """, target_id)
                info['type'] = 'npc'
                info['traits'] = npc.get('personality_traits', '') if npc else ''
            
            # Get recent memories
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
            
            # Get relationships
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
    
    async def _get_strategy_details(self, strategy_id: int) -> Optional[Dict]:
        """Get details of a counter-strategy"""
        
        async with get_db_connection_context() as conn:
            strategy = await conn.fetchrow("""
                SELECT * FROM counter_strategies
                WHERE strategy_id = $1
            """, strategy_id)
        
        return dict(strategy) if strategy else None
    
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
    
    async def _check_requirements(
        self,
        requirements: List[str],
        context: Dict[str, Any]
    ) -> bool:
        """Check if requirements are met"""
        
        # Simple check - could be enhanced
        if not requirements:
            return True
        
        # Check context for requirement keywords
        context_str = json.dumps(context).lower()
        for req in requirements:
            if req.lower() not in context_str:
                return False
        
        return True
    
    def _modify_success_chance(
        self,
        base_chance: float,
        context: Dict[str, Any]
    ) -> float:
        """Modify success chance based on context"""
        
        modified = base_chance
        
        # Boost for preparation
        if context.get('prepared', False):
            modified += 0.1
        
        # Penalty for rushed execution
        if context.get('rushed', False):
            modified -= 0.2
        
        # Bonus for resources
        if context.get('resources_available', 0) > 3:
            modified += 0.15
        
        return max(0.0, min(1.0, modified))

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def discover_leverage_opportunity(
    ctx: RunContextWrapper,
    target_id: int,
    observation: str
) -> Dict[str, Any]:
    """Discover potential leverage through observation"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = LeverageManager(user_id, conversation_id)
    
    context = {'observation': observation}
    leverage = await manager.discover_leverage(user_id, target_id, context)
    
    if leverage:
        return {
            'discovered': True,
            'leverage_id': leverage.leverage_id,
            'type': leverage.leverage_type.value,
            'description': leverage.description,
            'strength': leverage.strength,
            'uses_remaining': leverage.uses_remaining
        }
    else:
        return {
            'discovered': False,
            'message': 'No leverage discovered from this observation'
        }

@function_tool
async def use_leverage(
    ctx: RunContextWrapper,
    leverage_id: int,
    demand: str,
    aggressive: bool = False
) -> Dict[str, Any]:
    """Apply leverage to make a demand"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = LeverageManager(user_id, conversation_id)
    
    threat_level = 0.8 if aggressive else 0.4
    application = await manager.apply_leverage(leverage_id, demand, threat_level)
    
    # Generate narrative
    narrative = await manager.narrate_leverage_consequences(application, long_term=False)
    
    return {
        'application_id': application.application_id,
        'success': application.success_level > 0.5,
        'target_response': application.target_response,
        'narrative': narrative,
        'consequences': application.consequences
    }

@function_tool
async def defend_against_leverage(
    ctx: RunContextWrapper,
    leverage_id: int,
    available_resources: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate and optionally execute counter-strategies"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = LeverageManager(user_id, conversation_id)
    
    resources = available_resources or {}
    strategies = await manager.generate_counter_strategies(leverage_id, resources)
    
    return {
        'strategies_available': len(strategies),
        'options': [
            {
                'strategy_id': s.strategy_id,
                'type': s.strategy_type,
                'description': s.description,
                'requirements': s.requirements,
                'success_chance': s.success_chance,
                'risks': s.risks
            }
            for s in strategies
        ]
    }

@function_tool
async def execute_counter_strategy(
    ctx: RunContextWrapper,
    strategy_id: int,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a specific counter-strategy"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    manager = LeverageManager(user_id, conversation_id)
    
    context = additional_context or {}
    result = await manager.execute_counter_strategy(strategy_id, context)
    
    return result
