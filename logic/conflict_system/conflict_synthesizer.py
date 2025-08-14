# logic/conflict_system/conflict_synthesizer.py
"""
Conflict Synthesizer with LLM-generated merging and complexity management
Intelligently combines multiple conflicts into coherent narrative experiences
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# SYNTHESIS TYPES
# ===============================================================================

class SynthesisType(Enum):
    """Ways conflicts can be synthesized"""
    MERGE = "merge"  # Conflicts become one
    LAYER = "layer"  # Conflicts stack on top of each other
    CHAIN = "chain"  # One conflict leads to another
    PARALLEL = "parallel"  # Conflicts run simultaneously
    NESTED = "nested"  # One conflict contains another
    COMPETING = "competing"  # Conflicts compete for attention
    AMPLIFYING = "amplifying"  # Conflicts make each other worse
    RESOLVING = "resolving"  # One conflict solves another


@dataclass
class SynthesizedConflict:
    """A synthesized multi-conflict narrative"""
    synthesis_id: int
    primary_conflict_id: int
    component_conflicts: List[int]
    synthesis_type: SynthesisType
    complexity_score: float
    narrative_structure: Dict[str, Any]
    interaction_points: List[Dict[str, Any]]
    emergent_properties: List[str]


# ===============================================================================
# CONFLICT SYNTHESIZER
# ===============================================================================

class ConflictSynthesizer:
    """Synthesizes multiple conflicts into coherent experiences"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._synthesis_architect = None
        self._complexity_manager = None
        self._interaction_designer = None
        self._emergence_detector = None
        self._narrative_weaver = None
    
    # ========== Agent Properties ==========
    
    @property
    def synthesis_architect(self) -> Agent:
        if self._synthesis_architect is None:
            self._synthesis_architect = Agent(
                name="Conflict Synthesis Architect",
                instructions="""
                Design how multiple conflicts combine into unified experiences.
                
                Create syntheses that:
                - Maintain each conflict's integrity
                - Create emergent complexity
                - Generate meaningful interactions
                - Build narrative coherence
                - Avoid overwhelming players
                
                Focus on creating richer experiences, not just more complicated ones.
                """,
                model="gpt-5-nano",
            )
        return self._synthesis_architect
    
    @property
    def complexity_manager(self) -> Agent:
        if self._complexity_manager is None:
            self._complexity_manager = Agent(
                name="Complexity Manager",
                instructions="""
                Manage the complexity of synthesized conflicts.
                
                Ensure that:
                - Complexity serves narrative purpose
                - Players can track what matters
                - Choices remain meaningful
                - Cognitive load is manageable
                - Depth doesn't become confusion
                
                Complexity should enhance, not obscure, the experience.
                """,
                model="gpt-5-nano",
            )
        return self._complexity_manager
    
    @property
    def interaction_designer(self) -> Agent:
        if self._interaction_designer is None:
            self._interaction_designer = Agent(
                name="Conflict Interaction Designer",
                instructions="""
                Design how conflicts interact with each other.
                
                Create interactions that:
                - Feel organic and logical
                - Generate interesting dynamics
                - Create cascade effects
                - Enable creative solutions
                - Produce unexpected outcomes
                
                Make conflicts feel like a living ecosystem.
                """,
                model="gpt-5-nano",
            )
        return self._interaction_designer
    
    @property
    def emergence_detector(self) -> Agent:
        if self._emergence_detector is None:
            self._emergence_detector = Agent(
                name="Emergent Property Detector",
                instructions="""
                Detect emergent properties from conflict combinations.
                
                Identify:
                - New dynamics that emerge
                - Unexpected opportunities
                - Hidden connections
                - Synergistic effects
                - Narrative possibilities
                
                Find the magic that happens when conflicts combine.
                """,
                model="gpt-5-nano",
            )
        return self._emergence_detector
    
    @property
    def narrative_weaver(self) -> Agent:
        if self._narrative_weaver is None:
            self._narrative_weaver = Agent(
                name="Narrative Weaver",
                instructions="""
                Weave multiple conflicts into coherent narratives.
                
                Create stories that:
                - Unite disparate conflicts
                - Build dramatic tension
                - Provide satisfying arcs
                - Maintain character consistency
                - Generate memorable moments
                
                Turn conflict chaos into narrative gold.
                """,
                model="gpt-5-nano",
            )
        return self._narrative_weaver
    
    # ========== Synthesis Methods ==========
    
    async def analyze_conflicts_for_synthesis(
        self,
        conflict_ids: List[int]
    ) -> Dict[str, Any]:
        """Analyze if and how conflicts should be synthesized"""
        
        # Get conflict details
        conflicts = []
        async with get_db_connection_context() as conn:
            for conflict_id in conflict_ids:
                conflict = await conn.fetchrow("""
                    SELECT * FROM Conflicts WHERE conflict_id = $1
                """, conflict_id)
                
                stakeholders = await conn.fetch("""
                    SELECT * FROM conflict_stakeholders
                    WHERE conflict_id = $1
                """, conflict_id)
                
                conflicts.append({
                    'id': conflict_id,
                    'type': conflict['conflict_type'],
                    'name': conflict['conflict_name'],
                    'intensity': conflict['intensity'],
                    'phase': conflict['phase'],
                    'stakeholder_count': len(stakeholders),
                    'stakeholder_ids': [s['npc_id'] for s in stakeholders]
                })
        
        # Check for synthesis potential
        prompt = f"""
        Analyze these conflicts for synthesis potential:
        
        Conflicts: {json.dumps(conflicts)}
        
        Determine:
        - Should they be synthesized?
        - What synthesis type works best?
        - What emerges from combination?
        - What complexity this creates?
        
        Return JSON:
        {{
            "should_synthesize": true/false,
            "reason": "Why or why not",
            "synthesis_type": "merge/layer/chain/parallel/nested/competing/amplifying",
            "expected_complexity": 0.0 to 1.0,
            "shared_elements": ["common stakeholders/themes/locations"],
            "interaction_potential": ["how they could interact"],
            "emergent_opportunities": ["what new possibilities emerge"],
            "risks": ["potential problems from synthesis"]
        }}
        """
        
        response = await Runner.run(self.synthesis_architect, prompt)
        return json.loads(response.output)
    
    async def synthesize_conflicts(
        self,
        conflict_ids: List[int],
        synthesis_type: SynthesisType
    ) -> SynthesizedConflict:
        """Synthesize multiple conflicts into one experience"""
        
        # Get detailed conflict data
        conflicts_data = await self._get_detailed_conflicts(conflict_ids)
        
        # Design the synthesis structure
        structure = await self._design_synthesis_structure(
            conflicts_data,
            synthesis_type
        )
        
        # Identify interaction points
        interactions = await self._identify_interaction_points(
            conflicts_data,
            structure
        )
        
        # Detect emergent properties
        emergent = await self._detect_emergent_properties(
            conflicts_data,
            structure,
            interactions
        )
        
        # Calculate complexity
        complexity = await self._calculate_synthesis_complexity(
            conflicts_data,
            structure,
            interactions,
            emergent
        )
        
        # Store synthesis
        synthesis_id = await self._store_synthesis(
            conflict_ids[0],  # Primary conflict
            conflict_ids,
            synthesis_type,
            complexity,
            structure,
            interactions,
            emergent
        )
        
        return SynthesizedConflict(
            synthesis_id=synthesis_id,
            primary_conflict_id=conflict_ids[0],
            component_conflicts=conflict_ids,
            synthesis_type=synthesis_type,
            complexity_score=complexity,
            narrative_structure=structure,
            interaction_points=interactions,
            emergent_properties=emergent
        )
    
    async def _get_detailed_conflicts(
        self,
        conflict_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get detailed data for conflicts"""
        
        conflicts = []
        async with get_db_connection_context() as conn:
            for conflict_id in conflict_ids:
                conflict = await conn.fetchrow("""
                    SELECT * FROM Conflicts WHERE conflict_id = $1
                """, conflict_id)
                
                stakeholders = await conn.fetch("""
                    SELECT cs.*, n.npc_name
                    FROM conflict_stakeholders cs
                    JOIN NPCStats n ON cs.npc_id = n.npc_id
                    WHERE cs.conflict_id = $1
                """, conflict_id)
                
                conflicts.append({
                    'conflict': dict(conflict),
                    'stakeholders': [dict(s) for s in stakeholders]
                })
        
        return conflicts
    
    async def _design_synthesis_structure(
        self,
        conflicts: List[Dict[str, Any]],
        synthesis_type: SynthesisType
    ) -> Dict[str, Any]:
        """Design the structure of synthesized conflicts"""
        
        prompt = f"""
        Design a synthesis structure for these conflicts:
        
        Conflicts: {json.dumps([c['conflict'] for c in conflicts])}
        Synthesis Type: {synthesis_type.value}
        
        Create a structure that:
        - Unifies the conflicts coherently
        - Maintains individual conflict identity
        - Creates natural flow
        - Enables meaningful choices
        
        Return JSON:
        {{
            "primary_thread": "Main narrative thread",
            "secondary_threads": ["supporting narratives"],
            "connection_points": ["where conflicts connect"],
            "narrative_flow": {{
                "opening": "How it begins",
                "development": "How it develops",
                "climax": "Peak moment",
                "resolution_paths": ["possible endings"]
            }},
            "player_agency_points": ["where player choices matter most"],
            "dramatic_beats": ["key dramatic moments"]
        }}
        """
        
        response = await Runner.run(self.narrative_weaver, prompt)
        return json.loads(response.output)
    
    async def _identify_interaction_points(
        self,
        conflicts: List[Dict[str, Any]],
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify where conflicts interact"""
        
        # Find shared stakeholders
        stakeholder_map = {}
        for conflict in conflicts:
            for stakeholder in conflict['stakeholders']:
                npc_id = stakeholder['npc_id']
                if npc_id not in stakeholder_map:
                    stakeholder_map[npc_id] = []
                stakeholder_map[npc_id].append(conflict['conflict']['conflict_id'])
        
        shared_stakeholders = {
            npc_id: conflicts_list
            for npc_id, conflicts_list in stakeholder_map.items()
            if len(conflicts_list) > 1
        }
        
        prompt = f"""
        Identify interaction points between conflicts:
        
        Conflicts: {json.dumps([c['conflict']['conflict_name'] for c in conflicts])}
        Structure: {json.dumps(structure)}
        Shared Stakeholders: {json.dumps(shared_stakeholders)}
        
        Create interaction points that:
        - Feel organic
        - Create interesting dynamics
        - Enable cascade effects
        - Generate drama
        
        Return JSON:
        {{
            "interactions": [
                {{
                    "type": "cascade/interference/resonance/contradiction",
                    "conflicts_involved": [conflict_ids],
                    "trigger": "What causes interaction",
                    "effect": "What happens",
                    "player_influence": "How player affects it",
                    "narrative_impact": "Story consequences"
                }}
            ]
        }}
        """
        
        response = await Runner.run(self.interaction_designer, prompt)
        data = json.loads(response.output)
        return data['interactions']
    
    async def _detect_emergent_properties(
        self,
        conflicts: List[Dict[str, Any]],
        structure: Dict[str, Any],
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Detect emergent properties from synthesis"""
        
        prompt = f"""
        Detect emergent properties from this conflict synthesis:
        
        Conflicts: {json.dumps([c['conflict']['conflict_name'] for c in conflicts])}
        Structure: {json.dumps(structure)}
        Interactions: {json.dumps(interactions)}
        
        Identify emergent properties - things that arise from
        the combination that don't exist in individual conflicts:
        
        - New dynamics
        - Unexpected opportunities
        - Hidden themes
        - Synergistic effects
        - Novel solutions
        
        Return JSON:
        {{
            "emergent_properties": [
                "List of emergent properties as strings"
            ]
        }}
        """
        
        response = await Runner.run(self.emergence_detector, prompt)
        data = json.loads(response.output)
        return data['emergent_properties']
    
    async def _calculate_synthesis_complexity(
        self,
        conflicts: List[Dict[str, Any]],
        structure: Dict[str, Any],
        interactions: List[Dict[str, Any]],
        emergent: List[str]
    ) -> float:
        """Calculate complexity score of synthesis"""
        
        prompt = f"""
        Calculate complexity of this conflict synthesis:
        
        Number of Conflicts: {len(conflicts)}
        Number of Stakeholders: {sum(len(c['stakeholders']) for c in conflicts)}
        Number of Interactions: {len(interactions)}
        Number of Emergent Properties: {len(emergent)}
        Structure Complexity: {len(structure.get('dramatic_beats', []))} beats
        
        Rate complexity from 0.0 to 1.0 considering:
        - Cognitive load on player
        - Narrative coherence
        - Decision complexity
        - Tracking difficulty
        - Emotional investment required
        
        Return JSON:
        {{
            "complexity_score": 0.0 to 1.0,
            "breakdown": {{
                "structural": 0.0 to 1.0,
                "interpersonal": 0.0 to 1.0,
                "narrative": 0.0 to 1.0,
                "decision": 0.0 to 1.0
            }},
            "player_burden": "low/medium/high",
            "recommendation": "proceed/simplify/abort"
        }}
        """
        
        response = await Runner.run(self.complexity_manager, prompt)
        data = json.loads(response.output)
        return data['complexity_score']
    
    async def _store_synthesis(
        self,
        primary_id: int,
        all_ids: List[int],
        synthesis_type: SynthesisType,
        complexity: float,
        structure: Dict[str, Any],
        interactions: List[Dict[str, Any]],
        emergent: List[str]
    ) -> int:
        """Store synthesis in database"""
        
        async with get_db_connection_context() as conn:
            synthesis_id = await conn.fetchval("""
                INSERT INTO conflict_synthesis
                (primary_conflict_id, component_conflicts, synthesis_type,
                 complexity_score, narrative_structure, interaction_points,
                 emergent_properties, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                RETURNING synthesis_id
            """, primary_id, json.dumps(all_ids), synthesis_type.value,
            complexity, json.dumps(structure), json.dumps(interactions),
            json.dumps(emergent))
        
        return synthesis_id
    
    async def manage_synthesis_progression(
        self,
        synthesis_id: int,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Manage how synthesized conflicts progress"""
        
        # Get synthesis data
        async with get_db_connection_context() as conn:
            synthesis = await conn.fetchrow("""
                SELECT * FROM conflict_synthesis WHERE synthesis_id = $1
            """, synthesis_id)
        
        component_conflicts = json.loads(synthesis['component_conflicts'])
        interactions = json.loads(synthesis['interaction_points'])
        
        prompt = f"""
        Manage progression of synthesized conflicts:
        
        Synthesis Type: {synthesis['synthesis_type']}
        Event: {json.dumps(event)}
        Component Conflicts: {component_conflicts}
        Current Interactions: {json.dumps(interactions)}
        
        Determine:
        - How event affects each conflict
        - New interactions triggered
        - Cascade effects
        - Complexity changes
        
        Return JSON:
        {{
            "conflict_updates": {{
                "conflict_id": {{
                    "progress_change": -10 to +10,
                    "phase_change": "new phase if applicable",
                    "intensity_change": "increase/decrease/maintain"
                }}
            }},
            "new_interactions": ["newly triggered interactions"],
            "cascade_effects": ["secondary effects"],
            "complexity_delta": -0.1 to +0.1,
            "narrative_impact": "How this changes the overall story"
        }}
        """
        
        response = await Runner.run(self.synthesis_architect, prompt)
        return json.loads(response.output)


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def check_for_conflict_synthesis(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Check if active conflicts should be synthesized"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictSynthesizer(user_id, conversation_id)
    
    # Get active conflicts
    async with get_db_connection_context() as conn:
        conflicts = await conn.fetch("""
            SELECT conflict_id FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND is_active = true
        """, user_id, conversation_id)
    
    if len(conflicts) < 2:
        return {'synthesis_possible': False, 'reason': 'Not enough active conflicts'}
    
    conflict_ids = [c['conflict_id'] for c in conflicts]
    
    # Analyze for synthesis
    analysis = await synthesizer.analyze_conflicts_for_synthesis(conflict_ids)
    
    if analysis['should_synthesize']:
        # Perform synthesis
        synthesis = await synthesizer.synthesize_conflicts(
            conflict_ids,
            SynthesisType(analysis['synthesis_type'])
        )
        
        return {
            'synthesis_possible': True,
            'synthesis_performed': True,
            'synthesis_id': synthesis.synthesis_id,
            'complexity': synthesis.complexity_score,
            'emergent_properties': synthesis.emergent_properties
        }
    
    return {
        'synthesis_possible': False,
        'reason': analysis['reason']
    }


@function_tool
async def manage_synthesized_conflict(
    ctx: RunContextWrapper,
    synthesis_id: int,
    player_action: Dict[str, Any]
) -> Dict[str, Any]:
    """Manage player interaction with synthesized conflicts"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    synthesizer = ConflictSynthesizer(user_id, conversation_id)
    
    # Process action through synthesis
    result = await synthesizer.manage_synthesis_progression(
        synthesis_id,
        player_action
    )
    
    # Apply updates to individual conflicts
    async with get_db_connection_context() as conn:
        for conflict_id, updates in result['conflict_updates'].items():
            await conn.execute("""
                UPDATE Conflicts
                SET progress = progress + $1,
                    phase = COALESCE($2, phase),
                    intensity = COALESCE($3, intensity)
                WHERE conflict_id = $4
            """, updates['progress_change'],
            updates.get('phase_change'),
            updates.get('intensity_change'),
            int(conflict_id))
    
    return {
        'updates_applied': True,
        'cascade_effects': result['cascade_effects'],
        'narrative_impact': result['narrative_impact'],
        'new_complexity': result.get('complexity_delta', 0)
    }
