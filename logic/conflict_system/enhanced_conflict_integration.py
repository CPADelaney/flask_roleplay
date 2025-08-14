# logic/conflict_system/enhanced_conflict_integration.py
"""
Enhanced conflict system integration with LLM-generated dynamic content.
Replaces hardcoded tension analysis and generation with intelligent LLM agents.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from db.connection import get_db_connection_context

# Import core conflict systems
from logic.conflict_system.slice_of_life_conflicts import (
    SliceOfLifeConflictType,
    ConflictIntensity,
    EmergentConflictDetector,
    SliceOfLifeConflictManager,
    PatternBasedResolution,
    ConflictDailyIntegration
)

# Import other game systems
from logic.time_cycle import get_current_time_model, ActivityType
from logic.dynamic_relationships import OptimizedRelationshipManager
from npcs.npc_handler import NPCHandler
from story_agent.world_director_agent import WorldDirector, WorldMood
from lore.matriarchal_lore_system import MatriarchalLoreSystem
from logic.narrative_events import check_for_narrative_moments
from context.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

# ===============================================================================
# MASTER INTEGRATION CLASS WITH LLM ENHANCEMENT
# ===============================================================================

class EnhancedConflictSystemIntegration:
    """
    Master integration class with LLM-powered dynamic content generation.
    Preserves all original functionality while replacing hardcoded analysis.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core conflict components
        self.detector = EmergentConflictDetector(user_id, conversation_id)
        self.manager = SliceOfLifeConflictManager(user_id, conversation_id)
        self.resolver = PatternBasedResolution(user_id, conversation_id)
        self.daily_integration = ConflictDailyIntegration(user_id, conversation_id)
        
        # Other system connections (lazy loaded)
        self._relationship_manager = None
        self._npc_handler = None
        self._world_director = None
        self._lore_system = None
        self._memory_manager = None
        
        # LLM agents
        self._tension_analyzer = None
        self._conflict_generator = None
        self._integration_narrator = None
        
    # ========== System Connections (Preserved) ==========
    
    @property
    def relationship_manager(self):
        if self._relationship_manager is None:
            self._relationship_manager = OptimizedRelationshipManager(
                self.user_id, self.conversation_id
            )
        return self._relationship_manager
    
    @property
    def npc_handler(self):
        if self._npc_handler is None:
            self._npc_handler = NPCHandler(self.user_id, self.conversation_id)
        return self._npc_handler
    
    @property
    def world_director(self):
        if self._world_director is None:
            self._world_director = WorldDirector(self.user_id, self.conversation_id)
        return self._world_director
    
    @property
    def lore_system(self):
        if self._lore_system is None:
            self._lore_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
        return self._lore_system
    
    @property
    def memory_manager(self):
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager(self.user_id, self.conversation_id)
        return self._memory_manager
    
    # ========== LLM Agents ==========
    
    @property
    def tension_analyzer(self) -> Agent:
        """Agent for analyzing tensions from various sources"""
        if self._tension_analyzer is None:
            self._tension_analyzer = Agent(
                name="Tension Analyzer",
                instructions="""
                Analyze game state for emerging tensions and conflicts.
                
                Consider multiple sources:
                - Relationship dynamics and power imbalances
                - NPC narrative progression and mask integrity
                - Matriarchal society lore and cultural tensions
                - Recent player patterns and behaviors
                
                Generate contextual, nuanced tensions that feel organic.
                Focus on slice-of-life conflicts rather than dramatic confrontations.
                Consider how different tension sources interact and compound.
                """,
                model="gpt-5-nano",
            )
        return self._tension_analyzer
    
    @property
    def conflict_generator(self) -> Agent:
        """Agent for generating conflicts from tensions"""
        if self._conflict_generator is None:
            self._conflict_generator = Agent(
                name="Conflict Generator",
                instructions="""
                Transform identified tensions into active conflicts.
                
                Create conflicts that:
                - Feel natural to the current scene and activity
                - Involve present NPCs meaningfully
                - Build on established patterns
                - Have clear but subtle stakes
                - Can manifest through daily activities
                
                Generate specific, contextual descriptions.
                Make conflicts feel like natural extensions of relationships.
                """,
                model="gpt-5-nano",
            )
        return self._conflict_generator
    
    @property
    def integration_narrator(self) -> Agent:
        """Agent for narrating conflict integration"""
        if self._integration_narrator is None:
            self._integration_narrator = Agent(
                name="Conflict Integration Narrator",
                instructions="""
                Narrate how conflicts weave into daily activities.
                
                Focus on:
                - Subtle manifestations in routine moments
                - NPC behaviors that reflect underlying tensions
                - Environmental cues and atmosphere
                - Player choice opportunities
                - The accumulation of small moments
                
                Keep narration grounded and slice-of-life.
                Make every detail feel purposeful but not heavy-handed.
                """,
                model="gpt-5-nano",
            )
        return self._integration_narrator
    
    # ========== Conflict Generation from World State (Enhanced with LLM) ==========
    
    async def generate_contextual_conflict(self, ctx: RunContextWrapper) -> Optional[Dict[str, Any]]:
        """Generate a conflict based on current world state using LLM analysis"""
        
        # Get current world state
        world_state = await self.world_director.get_world_state()
        current_time = await get_current_time_model(ctx)
        
        # Get active NPCs at current location
        player_location = await self._get_player_location()
        present_npcs = await self.npc_handler.get_npcs_at_location(player_location)
        
        # Gather all tension sources with LLM analysis
        all_tensions = await self._analyze_all_tension_sources_llm(
            present_npcs, world_state, current_time
        )
        
        if not all_tensions:
            return None
        
        # Select and generate conflict with LLM
        selected_tension = await self._select_best_tension_llm(
            all_tensions, world_state, current_time
        )
        
        if not selected_tension:
            return None
        
        # Generate the conflict with LLM
        return await self._create_conflict_from_tension_llm(
            selected_tension, present_npcs, world_state
        )
    
    async def _analyze_all_tension_sources_llm(
        self,
        npcs: List[int],
        world_state: Any,
        current_time: Any
    ) -> List[Dict]:
        """Analyze all tension sources using LLM"""
        
        # Gather data from all sources
        relationship_data = await self._gather_relationship_data(npcs)
        npc_data = await self._gather_npc_progression_data(npcs)
        lore_data = await self.lore_system.get_current_cultural_tensions()
        
        # Create comprehensive context for LLM
        context = {
            'time_of_day': current_time.time_of_day,
            'world_mood': getattr(world_state, 'world_mood', 'neutral'),
            'location': await self._get_player_location(),
            'npcs_present': len(npcs),
            'relationship_summary': self._summarize_for_llm(relationship_data),
            'npc_progression': self._summarize_for_llm(npc_data),
            'cultural_context': self._summarize_for_llm(lore_data)
        }
        
        # Use LLM to analyze tensions
        prompt = f"""
        Analyze the current game state for emerging tensions:
        
        Context:
        {json.dumps(context, indent=2)}
        
        Identify 1-5 potential tension points that could become conflicts.
        For each tension:
        1. Source category (relationship/npc_progression/lore/emergent)
        2. Conflict type (from SliceOfLifeConflictType enum)
        3. Tension level (0.0-1.0)
        4. Specific contextual description
        5. Why it's relevant now
        
        Format as JSON array. Focus on subtle, slice-of-life tensions.
        """
        
        response = await self.tension_analyzer.run(prompt)
        
        try:
            tensions = json.loads(response.content)
            return self._format_llm_tensions(tensions, npcs)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tension analysis: {e}")
            return []
    
    async def _select_best_tension_llm(
        self,
        tensions: List[Dict],
        world_state: Any,
        current_time: Any
    ) -> Optional[Dict]:
        """Use LLM to select the most appropriate tension"""
        
        if not tensions:
            return None
        
        # Create selection context
        context = {
            'current_time': current_time.time_of_day,
            'world_mood': str(getattr(world_state, 'world_mood', 'neutral')),
            'active_conflicts_count': await self._get_active_conflict_count(),
            'recent_conflict_types': await self._get_recent_conflict_types()
        }
        
        prompt = f"""
        Select the best tension to develop into a conflict:
        
        Current Context:
        {json.dumps(context, indent=2)}
        
        Available Tensions:
        {json.dumps(tensions[:5], indent=2)}
        
        Choose the tension that:
        - Best fits the current mood and time
        - Wouldn't feel repetitive
        - Has the most interesting potential
        - Feels natural to emerge now
        
        Return the index (0-based) of the best tension and explain why.
        Format as JSON: {{"index": N, "reason": "..."}}
        """
        
        response = await self.tension_analyzer.run(prompt)
        
        try:
            result = json.loads(response.content)
            index = result.get('index', 0)
            if 0 <= index < len(tensions):
                selected = tensions[index]
                selected['selection_reason'] = result.get('reason', '')
                return selected
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to select tension: {e}")
        
        # Fallback to first tension
        return tensions[0] if tensions else None
    
    async def _create_conflict_from_tension_llm(
        self,
        tension: Dict,
        present_npcs: List[int],
        world_state: Any
    ) -> Dict[str, Any]:
        """Create a conflict using LLM generation"""
        
        # Generate conflict details with LLM
        prompt = f"""
        Generate a slice-of-life conflict from this tension:
        
        Tension: {json.dumps(tension, indent=2)}
        NPCs Present: {present_npcs}
        World Mood: {getattr(world_state, 'world_mood', 'neutral')}
        
        Create:
        1. Conflict name (brief, evocative)
        2. Detailed description (2-3 sentences)
        3. Initial intensity (subtext/tension/passive)
        4. How it might manifest in daily activities
        5. Stakes for the player
        
        Format as JSON. Keep it subtle and realistic.
        """
        
        response = await self.conflict_generator.run(prompt)
        
        try:
            details = json.loads(response.content)
        except json.JSONDecodeError:
            details = {
                'name': "Emerging Tension",
                'description': "A subtle conflict begins to take shape",
                'intensity': 'tension',
                'manifestation': "Small moments of friction",
                'stakes': "Personal autonomy"
            }
        
        # Create conflict in database
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts
                (user_id, conversation_id, conflict_type, conflict_name,
                 description, intensity, phase, is_active, progress)
                VALUES ($1, $2, $3, $4, $5, $6, 'emerging', true, 0)
                RETURNING conflict_id
            """, self.user_id, self.conversation_id,
            tension.get('type', SliceOfLifeConflictType.SUBTLE_RIVALRY).value,
            details.get('name', 'Emerging Tension'),
            details.get('description', tension.get('description', '')),
            details.get('intensity', 'tension'))
            
            # Add stakeholders
            stakeholder_npcs = tension.get('npc_ids', present_npcs[:2])
            for npc_id in stakeholder_npcs:
                await conn.execute("""
                    INSERT INTO conflict_stakeholders
                    (conflict_id, npc_id, faction, involvement_level)
                    VALUES ($1, $2, 'neutral', 'primary')
                """, conflict_id, npc_id)
            
            # Create initial memory
            await self.memory_manager.create_memory(
                entity_type='conflict',
                entity_id=conflict_id,
                memory_text=f"A new tension emerges: {details.get('description', '')}",
                importance='medium',
                tags=['conflict', 'emerging', tension.get('source_category', 'unknown')]
            )
        
        return {
            'conflict_id': conflict_id,
            'type': tension.get('type', SliceOfLifeConflictType.SUBTLE_RIVALRY).value,
            'intensity': details.get('intensity', 'tension'),
            'source': tension.get('source_category', 'emergent'),
            'stakeholders': stakeholder_npcs,
            'manifestation': details.get('manifestation', ''),
            'stakes': details.get('stakes', ''),
            'initial_tension_level': tension.get('tension_level', 0.5)
        }
    
    # ========== Daily Activity Integration (Enhanced with LLM) ==========
    
    async def integrate_conflicts_with_daily_routine(
        self,
        ctx: RunContextWrapper,
        activity_type: str,
        activity_description: str
    ) -> Dict[str, Any]:
        """Integrate active conflicts into daily activities using LLM"""
        
        # Get current context
        current_time = await get_current_time_model(ctx)
        player_location = await self._get_player_location()
        
        # Get appropriate conflicts
        appropriate_conflicts = await self.daily_integration.get_conflicts_for_time_of_day(
            current_time.time_of_day
        )
        
        if not appropriate_conflicts:
            return {
                'conflicts_active': False,
                'activity_proceeds_normally': True
            }
        
        # Get NPCs present
        present_npcs = await self.npc_handler.get_npcs_at_location(player_location)
        
        # Filter conflicts with present stakeholders
        active_conflicts = await self._filter_conflicts_by_npcs(
            appropriate_conflicts, present_npcs
        )
        
        if not active_conflicts:
            return {
                'conflicts_active': False,
                'activity_proceeds_normally': True
            }
        
        # Generate integration narrative with LLM
        integration = await self._generate_conflict_integration_llm(
            activity_type, activity_description, active_conflicts, present_npcs
        )
        
        return integration
    
    async def _generate_conflict_integration_llm(
        self,
        activity_type: str,
        activity_description: str,
        conflicts: List[Dict],
        present_npcs: List[int]
    ) -> Dict[str, Any]:
        """Generate how conflicts integrate into activity using LLM"""
        
        # Get NPC details for richer context
        npc_details = await self._get_npc_details_for_llm(present_npcs[:3])
        
        prompt = f"""
        Integrate these conflicts into the current activity:
        
        Activity: {activity_type} - {activity_description}
        Active Conflicts: {self._summarize_conflicts_for_llm(conflicts)}
        NPCs Present: {npc_details}
        
        Generate:
        1. How conflicts subtly manifest (2-3 specific details)
        2. Environmental/atmospheric cues
        3. 2-3 player choice opportunities (with subtext)
        4. NPC behaviors reflecting tensions
        5. Narrative moments that might trigger
        
        Keep everything subtle and slice-of-life.
        Format as JSON.
        """
        
        response = await self.integration_narrator.run(prompt)
        
        try:
            result = json.loads(response.content)
            
            return {
                'conflicts_active': True,
                'manifestation': result.get('manifestations', ['Subtle tensions color the interaction']),
                'environmental_cues': result.get('environmental_cues', []),
                'player_choices': self._format_player_choices(result.get('choices', [])),
                'npc_reactions': result.get('npc_behaviors', {}),
                'narrative_moments': result.get('narrative_moments', []),
                'atmosphere': result.get('atmosphere', 'tense')
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to generate integration: {e}")
            return {
                'conflicts_active': True,
                'manifestation': "Underlying tensions affect the moment",
                'player_choices': [],
                'npc_reactions': {}
            }
    
    # ========== Helper Methods (Enhanced) ==========
    
    async def _gather_relationship_data(self, npcs: List[int]) -> Dict:
        """Gather relationship data for LLM context"""
        data = {}
        for npc_id in npcs[:5]:  # Limit for token efficiency
            relationship = await self.relationship_manager.get_relationship(
                'npc', npc_id, 'player', self.user_id
            )
            if relationship:
                data[npc_id] = relationship.get('dimensions', {})
        return data
    
    async def _gather_npc_progression_data(self, npcs: List[int]) -> Dict:
        """Gather NPC progression data for LLM context"""
        data = {}
        async with get_db_connection_context() as conn:
            for npc_id in npcs[:5]:
                progression = await conn.fetchrow("""
                    SELECT narrative_stage, relationship_level 
                    FROM NPCNarrativeProgression
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if progression:
                    data[npc_id] = dict(progression)
        return data
    
    def _summarize_for_llm(self, data: Any) -> str:
        """Create concise summary for LLM prompts"""
        if isinstance(data, dict):
            if not data:
                return "No significant data"
            items = []
            for key, value in list(data.items())[:5]:
                items.append(f"{key}: {value}")
            return "; ".join(items)
        elif isinstance(data, list):
            return "; ".join(str(item) for item in data[:5])
        else:
            return str(data)[:200]
    
    def _format_llm_tensions(self, tensions: List, npcs: List[int]) -> List[Dict]:
        """Format LLM output into internal tension format"""
        formatted = []
        for t in tensions:
            try:
                formatted.append({
                    'source_category': t.get('source', 'emergent'),
                    'type': SliceOfLifeConflictType[t.get('type', 'SUBTLE_RIVALRY').upper()],
                    'tension_level': float(t.get('tension_level', 0.5)),
                    'description': t.get('description', 'A tension emerges'),
                    'relevance': t.get('relevance', ''),
                    'npc_ids': npcs[:2] if not t.get('npc_id') else [t.get('npc_id')]
                })
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format tension: {e}")
                continue
        return formatted
    
    def _summarize_conflicts_for_llm(self, conflicts: List) -> str:
        """Summarize conflicts for LLM prompts"""
        summary = []
        for c in conflicts[:3]:
            summary.append(
                f"- {c.get('conflict_type', 'unknown')} "
                f"({c.get('intensity', 'tension')}, {c.get('progress', 0)}% progress)"
            )
        return "\n".join(summary)
    
    async def _get_npc_details_for_llm(self, npc_ids: List[int]) -> str:
        """Get NPC details for LLM context"""
        details = []
        async with get_db_connection_context() as conn:
            for npc_id in npc_ids:
                npc = await conn.fetchrow("""
                    SELECT name, personality_traits FROM NPCs WHERE npc_id = $1
                """, npc_id)
                if npc:
                    details.append(f"{npc['name']} ({npc.get('personality_traits', 'unknown')})")
        return ", ".join(details) if details else "Unknown NPCs"
    
    def _format_player_choices(self, choices: List) -> List[Dict]:
        """Format player choices from LLM output"""
        formatted = []
        for choice in choices[:3]:
            if isinstance(choice, dict):
                formatted.append({
                    'id': choice.get('id', f'choice_{len(formatted)}'),
                    'text': choice.get('text', 'Make a choice'),
                    'subtext': choice.get('subtext', ''),
                    'impact': choice.get('impact', {})
                })
            elif isinstance(choice, str):
                formatted.append({
                    'id': f'choice_{len(formatted)}',
                    'text': choice,
                    'subtext': 'Consider the implications',
                    'impact': {}
                })
        return formatted
    
    async def _get_active_conflict_count(self) -> int:
        """Get count of active conflicts"""
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2 AND is_active = true
            """, self.user_id, self.conversation_id)
        return count or 0
    
    async def _get_recent_conflict_types(self) -> List[str]:
        """Get types of recent conflicts"""
        async with get_db_connection_context() as conn:
            types = await conn.fetch("""
                SELECT DISTINCT conflict_type FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2
                AND created_at > NOW() - INTERVAL '7 days'
                LIMIT 5
            """, self.user_id, self.conversation_id)
        return [t['conflict_type'] for t in types]
    
    async def _filter_conflicts_by_npcs(
        self,
        conflicts: List[Dict],
        present_npcs: List[int]
    ) -> List[Dict]:
        """Filter conflicts to those with present stakeholders"""
        active = []
        for conflict in conflicts:
            stakeholder_npcs = conflict.get('stakeholder_npcs', [])
            if any(npc in present_npcs for npc in stakeholder_npcs):
                active.append(conflict)
        return active
    
    async def _get_player_location(self) -> str:
        """Get current player location"""
        async with get_db_connection_context() as conn:
            location = await conn.fetchval("""
                SELECT current_location FROM player_state
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
        return location or "Home"
    
    # ========== Pattern Detection and Resolution (Enhanced) ==========
    
    async def check_for_pattern_resolutions(self) -> List[Dict[str, Any]]:
        """Check all active conflicts for pattern-based resolution"""
        resolutions = []
        
        async with get_db_connection_context() as conn:
            active_conflicts = await conn.fetch("""
                SELECT conflict_id FROM Conflicts
                WHERE user_id = $1 
                AND conversation_id = $2
                AND is_active = true
                AND progress >= 50
            """, self.user_id, self.conversation_id)
        
        for conflict in active_conflicts:
            resolution = await self.resolver.check_resolution_by_pattern(
                conflict['conflict_id']
            )
            
            if resolution:
                resolutions.append({
                    'conflict_id': conflict['conflict_id'],
                    'resolution': resolution
                })
                
                # Create narrative moment with LLM
                await self._create_resolution_narrative_llm(
                    conflict['conflict_id'], resolution
                )
        
        return resolutions
    
    async def _create_resolution_narrative_llm(
        self,
        conflict_id: int,
        resolution: Dict
    ):
        """Create a narrative moment for conflict resolution using LLM"""
        
        prompt = f"""
        Generate a brief narrative moment for this conflict resolution:
        
        Resolution Type: {resolution.get('resolution_type', 'unknown')}
        Description: {resolution.get('description', '')}
        New Patterns: {resolution.get('new_patterns', [])}
        
        Create a single sentence that captures the subtle shift.
        Focus on the feeling of something settling into place.
        """
        
        response = await self.integration_narrator.run(prompt)
        narrative_text = response.content.strip()
        
        # Add to memories
        await self.memory_manager.create_memory(
            entity_type='narrative',
            entity_id=conflict_id,
            memory_text=narrative_text,
            importance='high',
            tags=['resolution', 'pattern', 'conflict_end']
        )
        
        # Update world state if significant
        if resolution.get('new_routines_established'):
            await self.world_director.register_routine_change(
                f"Conflict {conflict_id}: {narrative_text}"
            )

# ===============================================================================
# PUBLIC API FUNCTIONS (Enhanced with LLM)
# ===============================================================================

@function_tool
async def process_conflict_in_scene(
    ctx: RunContextWrapper,
    scene_type: str,
    activity: str,
    present_npcs: List[int]
) -> Dict[str, Any]:
    """Main function to process conflicts within a scene"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    integration = EnhancedConflictSystemIntegration(user_id, conversation_id)
    
    # Check for new conflict generation
    if random.random() < 0.15:  # 15% chance
        new_conflict = await integration.generate_contextual_conflict(ctx)
        if new_conflict:
            logger.info(f"Generated new conflict: {new_conflict['type']}")
    
    # Integrate existing conflicts
    conflict_integration = await integration.integrate_conflicts_with_daily_routine(
        ctx, activity, f"Player is {activity}"
    )
    
    # Check for resolutions
    resolutions = await integration.check_for_pattern_resolutions()
    
    return {
        'conflicts_active': conflict_integration.get('conflicts_active', False),
        'manifestation': conflict_integration.get('manifestation'),
        'player_choices': conflict_integration.get('player_choices', []),
        'npc_reactions': conflict_integration.get('npc_reactions', {}),
        'resolutions': resolutions,
        'narrative_moments': conflict_integration.get('narrative_moments', [])
    }

@function_tool
async def analyze_scene_for_conflict_potential(
    ctx: RunContextWrapper,
    scene_description: str,
    npcs_present: List[int],
    recent_events: List[str]
) -> Dict[str, Any]:
    """Analyze a scene for potential conflict generation using LLM"""
    
    from logic.conflict_system.slice_of_life_conflicts import analyze_conflict_subtext
    
    # Get subtext analysis with LLM
    subtext = await analyze_conflict_subtext(ctx, scene_description, npcs_present)
    
    # Use LLM to determine tension score
    agent = Agent(
        name="Tension Scorer",
        instructions="Analyze scene elements to determine conflict potential.",
        model="gpt-5-nano",
    )
    
    prompt = f"""
    Analyze conflict potential:
    
    Scene: {scene_description}
    Recent Events: {recent_events[:5]}
    Subtext Analysis: {json.dumps(subtext, indent=2)}
    
    Calculate:
    1. Tension score (0.0-1.0)
    2. Should generate conflict (true/false)
    3. Primary dynamic at play
    4. Suggested conflict types
    
    Format as JSON.
    """
    
    response = await agent.run(prompt)
    
    try:
        result = json.loads(response.content)
        return {
            'tension_score': float(result.get('tension_score', 0.5)),
            'should_generate_conflict': result.get('should_generate', False),
            'primary_dynamic': result.get('primary_dynamic', subtext.get('primary_dynamic')),
            'potential_conflict_types': result.get('suggested_types', []),
            'subtext_analysis': subtext
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to analyze scene: {e}")
        return {
            'tension_score': 0.5,
            'should_generate_conflict': False,
            'primary_dynamic': 'unclear',
            'potential_conflict_types': [],
            'subtext_analysis': subtext
        }
