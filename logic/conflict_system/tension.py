# logic/conflict_system/tension.py
"""
Dynamic Tension System with LLM-generated content.
Manages the build-up, manifestation, and resolution of tensions in conflicts.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# TENSION TYPES AND STRUCTURES
# ===============================================================================

class TensionType(Enum):
    """Different types of tension in the game"""
    POWER = "power"
    SOCIAL = "social"
    SEXUAL = "sexual"
    EMOTIONAL = "emotional"
    ADDICTION = "addiction"
    VITAL = "vital"
    ECONOMIC = "economic"
    IDEOLOGICAL = "ideological"
    TERRITORIAL = "territorial"

class TensionLevel(Enum):
    """Levels of tension intensity"""
    ABSENT = 0.0
    SUBTLE = 0.2
    NOTICEABLE = 0.4
    PALPABLE = 0.6
    INTENSE = 0.8
    BREAKING = 1.0

@dataclass
class TensionSource:
    """A source contributing to tension"""
    source_type: str  # "npc", "conflict", "environment", "activity"
    source_id: Any
    contribution: float  # 0.0-1.0
    description: str

@dataclass
class TensionManifestation:
    """How tension manifests in a scene"""
    tension_type: TensionType
    level: float
    physical_cues: List[str]
    dialogue_modifications: List[str]
    environmental_changes: List[str]
    player_sensations: List[str]

# ===============================================================================
# TENSION SYSTEM WITH LLM
# ===============================================================================

class TensionSystem:
    """
    Manages tension dynamics using LLM for dynamic content generation.
    Replaces all hardcoded manifestations with contextual generation.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._tension_analyzer = None
        self._manifestation_generator = None
        self._escalation_narrator = None
        
    @property
    def tension_analyzer(self) -> Agent:
        """Agent for analyzing tension sources and levels"""
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="""
                Analyze various sources to determine tension levels and types.
                
                Consider:
                - Relationship dynamics and power imbalances
                - Recent conflicts and their intensity
                - Environmental factors and time of day
                - Player choices and their consequences
                - NPC emotional states and goals
                
                Identify subtle tensions that build gradually.
                Focus on psychological and emotional undercurrents.
                """,
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def manifestation_generator(self) -> Agent:
        """Agent for generating tension manifestations"""
        if self._manifestation_generator is None:
            self._manifestation_generator = Agent(
                name="Tension Manifestation Generator",
                instructions="""
                Generate specific, sensory manifestations of tension.
                
                Create:
                - Physical cues (body language, facial expressions)
                - Dialogue modifications (tone, word choice, pauses)
                - Environmental changes (atmosphere, sounds, lighting)
                - Player sensations (what they notice/feel)
                
                Make manifestations subtle and contextual.
                Layer multiple small details rather than obvious statements.
                Focus on show-don't-tell storytelling.
                """,
                model="gpt-5-nano",
            )
        return self._manifestation_generator
    
    @property
    def escalation_narrator(self) -> Agent:
        """Agent for narrating tension escalation"""
        if self._escalation_narrator is None:
            self._escalation_narrator = Agent(
                name="Tension Escalation Narrator",
                instructions="""
                Narrate how tensions build, peak, and release.
                
                Focus on:
                - Gradual accumulation of small moments
                - Tipping points and triggers
                - The moment before something breaks
                - Release and aftermath
                
                Create atmospheric, evocative descriptions.
                Use sensory details and emotional weight.
                """,
                model="gpt-5-nano",
            )
        return self._escalation_narrator
    
    # ========== Core Tension Management ==========
    
    async def calculate_current_tensions(self) -> Dict[TensionType, float]:
        """Calculate current tension levels from all sources"""
        
        tensions = {t: 0.0 for t in TensionType}
        
        # Get tension sources
        sources = await self._gather_tension_sources()
        
        # Use LLM to analyze and calculate tensions
        prompt = f"""
        Analyze these tension sources and calculate overall tension levels:
        
        Sources:
        {json.dumps([self._source_to_dict(s) for s in sources], indent=2)}
        
        For each tension type (power, social, sexual, emotional, addiction, vital, economic, ideological, territorial):
        Calculate a level from 0.0 to 1.0 based on the sources.
        
        Consider how sources compound or cancel each other.
        Format as JSON: {{"power": 0.X, "social": 0.X, ...}}
        """
        
        response = await self.tension_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            for tension_type in TensionType:
                if tension_type.value in result:
                    tensions[tension_type] = min(1.0, max(0.0, float(result[tension_type.value])))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse tension levels: {e}")
            # Fallback calculation
            for source in sources:
                tensions[self._map_source_to_tension_type(source)] += source.contribution * 0.2
        
        # Store in database
        await self._store_tension_levels(tensions)
        
        return tensions
    
    async def build_tension(
        self,
        tension_type: TensionType,
        amount: float,
        source: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build tension with contextual narrative"""
        
        current_level = await self._get_tension_level(tension_type)
        new_level = min(1.0, current_level + amount)
        
        # Generate build narrative with LLM
        prompt = f"""
        Narrate tension building:
        
        Type: {tension_type.value}
        Current Level: {current_level:.2f} → {new_level:.2f}
        Source: {source}
        Context: {json.dumps(context or {}, indent=2)}
        
        Create:
        1. A brief description of how the tension builds (1-2 sentences)
        2. 2-3 subtle physical/environmental cues
        3. Any changes in NPC behavior
        4. What the player might sense
        
        Keep it subtle and atmospheric.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            narrative = result
        except json.JSONDecodeError:
            narrative = {
                'description': f"The {tension_type.value} tension grows stronger",
                'cues': ['A subtle shift in the atmosphere'],
                'npc_changes': [],
                'player_sensation': 'You feel something building'
            }
        
        # Update database
        await self._update_tension_level(tension_type, new_level)
        
        # Record event
        await self._create_tension_memory(
            tension_type, 
            f"Tension increased: {narrative['description']}", 
            new_level
        )
        
        return {
            'tension_type': tension_type.value,
            'old_level': current_level,
            'new_level': new_level,
            'narrative': narrative,
            'threshold_crossed': self._check_threshold_crossed(current_level, new_level)
        }
    
    async def resolve_tension(
        self,
        tension_type: TensionType,
        amount: float,
        resolution_type: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Resolve tension with contextual narrative"""
        
        current_level = await self._get_tension_level(tension_type)
        new_level = max(0.0, current_level - amount)
        
        # Generate resolution narrative with LLM
        prompt = f"""
        Narrate tension resolution:
        
        Type: {tension_type.value}
        Current Level: {current_level:.2f} → {new_level:.2f}
        Resolution Type: {resolution_type}
        Context: {json.dumps(context or {}, indent=2)}
        
        Create:
        1. How the tension releases (1-2 sentences)
        2. Physical/emotional relief descriptions
        3. Environmental changes
        4. Aftermath mood
        
        Focus on the feeling of release and what changes.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            narrative = result
        except json.JSONDecodeError:
            narrative = {
                'release': f"The {tension_type.value} tension eases",
                'relief': ['A sense of lightness returns'],
                'changes': ['The atmosphere softens'],
                'aftermath': 'calm'
            }
        
        # Update database
        await self._update_tension_level(tension_type, new_level)
        
        return {
            'tension_type': tension_type.value,
            'old_level': current_level,
            'new_level': new_level,
            'narrative': narrative,
            'fully_resolved': new_level < 0.1
        }
    
    async def generate_tension_manifestation(
        self,
        scene_context: Dict[str, Any]
    ) -> TensionManifestation:
        """Generate how current tensions manifest in a scene"""
        
        # Get current tensions
        tensions = await self.calculate_current_tensions()
        
        # Find dominant tension
        dominant_type, dominant_level = max(
            tensions.items(), 
            key=lambda x: x[1]
        )
        
        if dominant_level < 0.1:
            return self._create_no_tension_manifestation()
        
        # Generate manifestation with LLM
        prompt = f"""
        Generate tension manifestations for this scene:
        
        Dominant Tension: {dominant_type.value} ({dominant_level:.2f})
        Other Tensions: {self._format_tensions_for_prompt(tensions)}
        Scene: {json.dumps(scene_context, indent=2)}
        
        Create specific, sensory details:
        1. 3-4 physical cues (body language, expressions)
        2. 2-3 dialogue modifications (how speech changes)
        3. 2-3 environmental changes (atmosphere, sounds)
        4. 2-3 player sensations (what they notice/feel)
        
        Make it subtle and layered. Show tension through details.
        Format as JSON with arrays for each category.
        """
        
        response = await self.manifestation_generator.run(prompt)
        
        try:
            result = json.loads(response.content)
            return TensionManifestation(
                tension_type=dominant_type,
                level=dominant_level,
                physical_cues=result.get('physical_cues', []),
                dialogue_modifications=result.get('dialogue_modifications', []),
                environmental_changes=result.get('environmental_changes', []),
                player_sensations=result.get('player_sensations', [])
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse manifestation: {e}")
            return self._create_fallback_manifestation(dominant_type, dominant_level)
    
    async def check_tension_breaking_point(self) -> Optional[Dict[str, Any]]:
        """Check if any tension has reached breaking point"""
        
        tensions = await self.calculate_current_tensions()
        breaking_tensions = {
            t: level for t, level in tensions.items() 
            if level >= TensionLevel.BREAKING.value
        }
        
        if not breaking_tensions:
            return None
        
        # Generate breaking point event with LLM
        breaking_type = max(breaking_tensions.items(), key=lambda x: x[1])[0]
        
        prompt = f"""
        A tension has reached its breaking point:
        
        Breaking Tension: {breaking_type.value}
        Level: {breaking_tensions[breaking_type]:.2f}
        All Tensions: {self._format_tensions_for_prompt(tensions)}
        
        Generate:
        1. The triggering moment (what finally breaks)
        2. Immediate consequences
        3. How it affects other tensions
        4. Choices available to player
        
        Make it dramatic but believable.
        Format as JSON.
        """
        
        response = await self.escalation_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            return {
                'breaking_tension': breaking_type.value,
                'trigger': result.get('trigger', 'The tension finally snaps'),
                'consequences': result.get('consequences', []),
                'tension_changes': result.get('tension_changes', {}),
                'player_choices': result.get('choices', [])
            }
        except json.JSONDecodeError:
            return {
                'breaking_tension': breaking_type.value,
                'trigger': 'The tension reaches a breaking point',
                'consequences': ['Things can no longer continue as they were'],
                'tension_changes': {},
                'player_choices': []
            }
    
    # ========== Helper Methods ==========
    
    async def _gather_tension_sources(self) -> List[TensionSource]:
        """Gather all current sources of tension"""
        
        sources = []
        
        async with get_db_connection_context() as conn:
            # Get conflict tensions
            conflicts = await conn.fetch("""
                SELECT conflict_id, conflict_type, intensity, progress
                FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND is_active = true
            """, self.user_id, self.conversation_id)
            
            for conflict in conflicts:
                sources.append(TensionSource(
                    source_type="conflict",
                    source_id=conflict['conflict_id'],
                    contribution=conflict['progress'] / 100 * 0.5,
                    description=f"{conflict['conflict_type']} conflict"
                ))
            
            # Get NPC tensions
            npc_tensions = await conn.fetch("""
                SELECT npc_id, emotional_state, stress_level
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND stress_level > 30
            """, self.user_id, self.conversation_id)
            
            for npc in npc_tensions:
                sources.append(TensionSource(
                    source_type="npc",
                    source_id=npc['npc_id'],
                    contribution=npc['stress_level'] / 100 * 0.3,
                    description=f"NPC stress: {npc['emotional_state']}"
                ))
        
        return sources
    
    async def _get_tension_level(self, tension_type: TensionType) -> float:
        """Get current level for a specific tension type"""
        
        async with get_db_connection_context() as conn:
            level = await conn.fetchval("""
                SELECT level FROM TensionLevels
                WHERE user_id = $1 AND conversation_id = $2
                AND tension_type = $3
            """, self.user_id, self.conversation_id, tension_type.value)
        
        return level or 0.0
    
    async def _update_tension_level(self, tension_type: TensionType, level: float):
        """Update tension level in database"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO TensionLevels (user_id, conversation_id, tension_type, level)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, tension_type)
                DO UPDATE SET level = $4, updated_at = NOW()
            """, self.user_id, self.conversation_id, tension_type.value, level)
    
    async def _store_tension_levels(self, tensions: Dict[TensionType, float]):
        """Store all tension levels"""
        
        for tension_type, level in tensions.items():
            await self._update_tension_level(tension_type, level)
    
    async def _create_tension_memory(
        self,
        tension_type: TensionType,
        description: str,
        level: float
    ):
        """Create a memory record of tension change"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO enhanced_memories
                (user_id, conversation_id, entity_type, entity_id, 
                 memory_text, importance, tags)
                VALUES ($1, $2, 'tension', $3, $4, $5, $6)
            """, self.user_id, self.conversation_id,
            tension_type.value, description,
            'medium' if level < 0.7 else 'high',
            ['tension', tension_type.value, f"level_{int(level*10)}"])
    
    def _source_to_dict(self, source: TensionSource) -> Dict:
        """Convert TensionSource to dict for JSON"""
        return {
            'type': source.source_type,
            'id': str(source.source_id),
            'contribution': source.contribution,
            'description': source.description
        }
    
    def _map_source_to_tension_type(self, source: TensionSource) -> TensionType:
        """Map a source to its primary tension type"""
        
        # Simple mapping - could be enhanced with LLM
        mappings = {
            'conflict': TensionType.POWER,
            'npc': TensionType.EMOTIONAL,
            'environment': TensionType.VITAL,
            'activity': TensionType.SOCIAL
        }
        return mappings.get(source.source_type, TensionType.EMOTIONAL)
    
    def _format_tensions_for_prompt(self, tensions: Dict[TensionType, float]) -> str:
        """Format tensions for LLM prompts"""
        
        formatted = []
        for t, level in tensions.items():
            if level > 0.1:
                formatted.append(f"{t.value}: {level:.2f}")
        return ", ".join(formatted) if formatted else "minimal tensions"
    
    def _check_threshold_crossed(self, old_level: float, new_level: float) -> Optional[str]:
        """Check if a significant threshold was crossed"""
        
        thresholds = [
            (0.2, "subtle"),
            (0.4, "noticeable"),
            (0.6, "palpable"),
            (0.8, "intense"),
            (1.0, "breaking")
        ]
        
        for threshold, name in thresholds:
            if old_level < threshold <= new_level:
                return f"entered_{name}"
            elif new_level < threshold <= old_level:
                return f"left_{name}"
        
        return None
    
    def _create_no_tension_manifestation(self) -> TensionManifestation:
        """Create manifestation for no/minimal tension"""
        
        return TensionManifestation(
            tension_type=TensionType.EMOTIONAL,
            level=0.0,
            physical_cues=["Relaxed postures", "Easy movements"],
            dialogue_modifications=["Natural speech patterns"],
            environmental_changes=["Comfortable atmosphere"],
            player_sensations=["A sense of ease"]
        )
    
    def _create_fallback_manifestation(
        self,
        tension_type: TensionType,
        level: float
    ) -> TensionManifestation:
        """Create fallback manifestation if LLM fails"""
        
        return TensionManifestation(
            tension_type=tension_type,
            level=level,
            physical_cues=[f"Subtle {tension_type.value} tension in body language"],
            dialogue_modifications=["Careful word choices"],
            environmental_changes=["The atmosphere feels charged"],
            player_sensations=["You sense an undercurrent of tension"]
        )

# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def analyze_scene_tensions(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    current_activity: str
) -> Dict[str, Any]:
    """Analyze tensions in current scene"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    system = TensionSystem(user_id, conversation_id)
    
    # Calculate current tensions
    tensions = await system.calculate_current_tensions()
    
    # Generate manifestation
    scene_context = {
        'description': scene_description,
        'npcs': npcs_present,
        'activity': current_activity
    }
    manifestation = await system.generate_tension_manifestation(scene_context)
    
    # Check for breaking points
    breaking_point = await system.check_tension_breaking_point()
    
    return {
        'tension_levels': {t.value: level for t, level in tensions.items()},
        'dominant_tension': manifestation.tension_type.value,
        'manifestation': {
            'physical_cues': manifestation.physical_cues,
            'dialogue_mods': manifestation.dialogue_modifications,
            'environment': manifestation.environmental_changes,
            'sensations': manifestation.player_sensations
        },
        'breaking_point': breaking_point
    }

@function_tool
async def modify_tension(
    ctx: RunContextWrapper,
    tension_type: str,
    change: float,
    reason: str
) -> Dict[str, Any]:
    """Modify a specific tension level"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    system = TensionSystem(user_id, conversation_id)
    
    try:
        t_type = TensionType(tension_type)
    except ValueError:
        return {'error': f'Invalid tension type: {tension_type}'}
    
    if change > 0:
        result = await system.build_tension(t_type, abs(change), reason)
    else:
        result = await system.resolve_tension(t_type, abs(change), reason)
    
    return result

@function_tool
async def get_tension_report(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Get comprehensive tension report"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    system = TensionSystem(user_id, conversation_id)
    
    tensions = await system.calculate_current_tensions()
    
    # Sort by level
    sorted_tensions = sorted(
        tensions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    report = {
        'current_tensions': {t.value: level for t, level in sorted_tensions},
        'highest': sorted_tensions[0] if sorted_tensions else None,
        'total_tension': sum(tensions.values()),
        'critical_tensions': [
            t.value for t, level in tensions.items() 
            if level >= 0.8
        ],
        'recommendations': []
    }
    
    # Generate recommendations
    if report['total_tension'] > 3.0:
        report['recommendations'].append("Multiple high tensions - consider resolution")
    if report['critical_tensions']:
        report['recommendations'].append("Critical tensions need immediate attention")
    
    return report
