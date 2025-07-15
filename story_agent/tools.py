# story_agent/tools.py
"""
Refactored tools for the Story Director agent with NPC-specific narrative progression
and generative agent integration for dynamic content generation.
"""

# Standard library imports
import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Literal

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper, Runner

# Database imports
from db.connection import get_db_connection_context

# Local application imports - NPC narrative progression
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    check_for_npc_revelation,
    NPC_NARRATIVE_STAGES,
    NPCNarrativeStage
)

# Local application imports - Legacy narrative progression
from logic.narrative_progression import (
    get_current_narrative_stage,  # Deprecated but kept
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity,
    get_relationship_overview
)

# Local application imports - Social links
from logic.social_links import (
    get_social_link_tool,
    get_relationship_summary_tool,
    check_for_crossroads_tool,
    check_for_ritual_tool,
    apply_crossroads_choice_tool
)

# Context system imports
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager, search_memories_tool, MemorySearchRequest
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

# Canon imports (was missing)
from lore.core import canon

# Config imports (was missing - adjust path as needed)
from context.context_config import get_config

# Initialize logger
logger = logging.getLogger(__name__)

# Type alias for context
ContextType = Any

# Simple context class for canon operations
class SimpleContext:
    """Simple context for operations that need user_id and conversation_id."""
    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id

# ============= GENERATIVE AGENTS =============

# Agent for generating dynamic personal revelations
revelation_generator = Agent(
    name="RevelationGenerator",
    instructions="""
    You generate psychologically realistic personal revelations for a player character
    in a femdom-themed narrative game. The revelations should reflect the player's
    growing awareness of their changing relationship dynamics with specific NPCs.
    
    Consider:
    - The NPC's current narrative stage (how open they are about control)
    - The type of revelation (dependency, obedience, corruption, etc.)
    - The player's history with this NPC
    - The psychological realism of the realization
    
    Generate revelations that are:
    - Internally focused (the player's thoughts)
    - Subtly disturbing or thought-provoking
    - Appropriate to the relationship's current stage
    - Written in first person
    """,
    model="gpt-4.1-nano",
    temperature=0.8
)

# Agent for creating dynamic dream sequences
dream_weaver = Agent(
    name="DreamWeaver",
    instructions="""
    You create symbolic, psychologically meaningful dream sequences for a player
    in a femdom-themed narrative game. Dreams should reflect the player's
    subconscious processing of their relationships with multiple NPCs.
    
    Consider:
    - Each NPC's narrative stage and role in the player's life
    - Symbolic representations of control, dependency, and transformation
    - The surreal logic of dreams
    - Psychological horror elements without being gratuitous
    
    Create dreams that are:
    - Rich in symbolism
    - Unsettling but not explicitly frightening
    - Reflective of multiple relationship dynamics
    - Open to interpretation
    """,
    model="gpt-4.1-nano",
    temperature=0.9
)

# Agent for suggesting contextual activities
activity_suggester = Agent(
    name="ActivitySuggester",
    instructions="""
    You suggest activities for NPCs to engage the player in, based on their
    personality, narrative stage, and current context. Activities should
    subtly reinforce power dynamics appropriate to each NPC's progression.
    
    Consider:
    - NPC's archetype and personality traits
    - Current narrative stage (how overt their control is)
    - Setting and time constraints
    - Resource implications of activities
    
    Suggest activities that:
    - Feel natural for the character and setting
    - Subtly advance the power dynamic
    - Vary in intensity based on narrative stage
    - Create memorable interactions
    """,
    model="gpt-4.1-nano",
    temperature=0.7
)

# Agent for analyzing manipulation opportunities
manipulation_analyst = Agent(
    name="ManipulationAnalyst",
    instructions="""
    You analyze situations to identify how NPCs might manipulate the player
    within ongoing conflicts. Consider each NPC's personality, narrative stage,
    and the specific conflict context.
    
    Analyze:
    - NPC's manipulation style based on personality
    - How their narrative stage affects their approach
    - Conflict stakes and NPC's goals
    - Player's current vulnerabilities
    
    Provide analysis that:
    - Respects each NPC's established character
    - Scales manipulation overtness with narrative stage
    - Identifies specific leverage points
    - Suggests believable manipulation tactics
    """,
    model="gpt-4.1-nano",
    temperature=0.6
)

conflict_beat_writer = Agent(
    name="ConflictBeatWriter",
    instructions="""
    You generate story beats for ongoing conflicts in a femdom-themed narrative game.
    Story beats should advance the conflict narrative while reflecting the game's themes
    of subtle control and shifting power dynamics.
    
    Given conflict context, generate:
    1. A compelling narrative beat that advances the conflict
    2. An appropriate progress value (0-100) based on:
       - Current phase: brewing (0-25), active (25-50), climax (50-75), resolution (75-100)
       - Narrative significance of the beat
       - Impact on player and stakeholders
    
    Consider:
    - The conflict's current phase and progress
    - Stakeholder motivations and hidden agendas
    - Recent player actions and their consequences
    - How NPCs might use the conflict to manipulate the player
    - The overall theme of gradual loss of autonomy
    
    Format your response as:
    BEAT: [narrative description]
    PROGRESS: [numeric value 0-100]
    IMPACT: [brief description of consequences]
    """,
    model="gpt-4.1-nano",
    temperature=0.7
)

dialogue_detector = Agent(
    name="DialogueDetector",
    instructions="""
    You analyze player input to determine if it's a conversational exchange
    that would benefit from quick dialogue mode rather than full narrative responses.
    
    Dialogue mode is appropriate when:
    - Player asks a direct question to an NPC
    - Player makes a short conversational statement
    - Player is clearly engaged in back-and-forth with an NPC
    - The input is under 20 words and conversational
    - Context shows recent dialogue exchanges
    
    Dialogue mode is NOT appropriate when:
    - Player describes actions or movements
    - Player addresses multiple NPCs
    - Significant time would pass
    - The situation requires environmental description
    - Combat or complex activities are involved
    
    Assess confidence level (0.0-1.0) based on how clear the signals are.
    """,
    model="gpt-4.1-nano",
    temperature=0.3
)

# ===== PYDANTIC MODELS =====

# Relationship Models
class DimensionChanges(BaseModel):
    """Changes to relationship dimensions."""
    money: Optional[int] = Field(None, ge=-100, le=100)
    trust: Optional[int] = Field(None, ge=-100, le=100)
    respect: Optional[int] = Field(None, ge=-100, le=100)
    obedience: Optional[int] = Field(None, ge=-100, le=100)
    closeness: Optional[int] = Field(None, ge=-100, le=100)
    dominance: Optional[int] = Field(None, ge=-100, le=100)
    
    model_config = ConfigDict(extra="forbid")

# Conflict Models
class FactionAffiliation(BaseModel):
    """NPC faction affiliation details."""
    faction_id: int
    faction_name: str
    
    model_config = ConfigDict(extra="forbid")

class MentionedNPCTyped(BaseModel):
    """Fully typed NPC mentioned in conflict analysis."""
    npc_id: int
    npc_name: str
    dominance: int
    faction_affiliations: List[FactionAffiliation] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class MentionedFaction(BaseModel):
    """Faction mentioned in conflict analysis."""
    faction_id: int
    faction_name: str
    
    model_config = ConfigDict(extra="forbid")

class NPCRelationship(BaseModel):
    """Relationship between NPCs in conflict analysis."""
    npc1_id: int
    npc1_name: str
    npc2_id: int
    npc2_name: str
    relationship_type: Literal["alliance", "rivalry", "unknown"]
    sentence: str
    
    model_config = ConfigDict(extra="forbid")

class InternalFactionConflict(BaseModel):
    """Internal faction conflict details."""
    faction_id: int
    challenger_npc_id: int
    target_npc_id: int
    prize: str
    approach: str
    
    model_config = ConfigDict(extra="forbid")

class ConflictAnalysis(BaseModel):
    """Analysis results from conflict potential detection."""
    conflict_intensity: int = Field(ge=0, le=10)
    matched_keywords: List[str]
    mentioned_npcs: List[MentionedNPCTyped]
    mentioned_factions: List[MentionedFaction]
    npc_relationships: List[NPCRelationship]
    recommended_conflict_type: Literal["major", "standard", "minor", "catastrophic"]
    potential_internal_faction_conflict: Optional[InternalFactionConflict] = None
    has_conflict_potential: bool
    
    model_config = ConfigDict(extra="forbid")

class ManipulationGoal(BaseModel):
    """Goal for NPC manipulation attempts."""
    faction: Literal["a", "b", "neutral"]
    involvement_level: Literal["none", "observing", "participating", "leading"]
    money_committed: int = Field(0, ge=0)
    supplies_committed: int = Field(0, ge=0)
    influence_committed: int = Field(0, ge=0)
    specific_action: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class ManipulationPotential(BaseModel):
    """Analysis of NPC manipulation potential."""
    overall_potential: int = Field(ge=0, le=100)
    most_effective_type: Literal["domination", "blackmail", "seduction", "gaslighting", "manipulation"]
    femdom_compatible: bool
    
    model_config = ConfigDict(extra="forbid")

class ManipulationOpportunity(BaseModel):
    """Suggested manipulation opportunity."""
    conflict_id: int
    conflict_name: str
    npc_id: int
    npc_name: str
    dominance: int
    manipulation_type: str
    potential: int
    
    model_config = ConfigDict(extra="forbid")

# Context/Memory Models
class MemorySearchParams(BaseModel):
    """Parameters for memory search operations."""
    query_text: str
    memory_type: Optional[str] = None
    limit: int = Field(5, ge=1, le=100)
    use_vector: bool = True
    
    model_config = ConfigDict(extra="forbid")

class ContextQueryParams(BaseModel):
    """Parameters for context retrieval."""
    query_text: str = ""
    use_vector: bool = True
    max_tokens: Optional[int] = Field(None, ge=100, le=10000)
    
    model_config = ConfigDict(extra="forbid")

class StoreMemoryParams(BaseModel):
    """Parameters for storing narrative memories."""
    content: str
    memory_type: str = "observation"
    importance: float = Field(0.6, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=lambda: ["story_director"])
    
    model_config = ConfigDict(extra="forbid")

class VectorSearchParams(BaseModel):
    """Parameters for vector search operations."""
    query_text: str
    entity_types: List[str] = Field(
        default_factory=lambda: ["npc", "location", "memory", "narrative"]
    )
    top_k: int = Field(5, ge=1, le=50)
    
    model_config = ConfigDict(extra="forbid")

# Resource Models
class ResourceCheck(BaseModel):
    """Resource requirements to check."""
    money: int = Field(0, ge=0)
    supplies: int = Field(0, ge=0)
    influence: int = Field(0, ge=0)
    
    model_config = ConfigDict(extra="forbid")

class ResourceCommitment(BaseModel):
    """Resources to commit to a conflict."""
    conflict_id: int
    money: int = Field(0, ge=0)
    supplies: int = Field(0, ge=0)
    influence: int = Field(0, ge=0)
    
    model_config = ConfigDict(extra="forbid")

# Activity Models
class ActivityAnalysisParams(BaseModel):
    """Parameters for activity analysis."""
    activity_text: str
    setting_context: Optional[str] = None
    apply_effects: bool = False
    
    model_config = ConfigDict(extra="forbid")

class ActivityFilterParams(BaseModel):
    """Parameters for filtering activities."""
    npc_archetypes: List[str] = Field(default_factory=list)
    meltdown_level: int = Field(0, ge=0, le=5)
    setting: str = ""
    
    model_config = ConfigDict(extra="forbid")

class ActivitySuggestionParams(BaseModel):
    """Parameters for activity suggestions."""
    npc_name: str
    intensity_level: int = Field(2, ge=1, le=5)
    archetypes: Optional[List[str]] = None
    
    model_config = ConfigDict(extra="forbid")

# Player Involvement Models
class PlayerInvolvementParams(BaseModel):
    """Parameters for setting player involvement in conflicts."""
    conflict_id: int
    involvement_level: Literal["none", "observing", "participating", "leading"]
    faction: Literal["a", "b", "neutral"] = "neutral"
    money_committed: int = Field(0, ge=0)
    supplies_committed: int = Field(0, ge=0)
    influence_committed: int = Field(0, ge=0)
    action: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class PlayerInvolvementData(BaseModel):
    """Current player involvement in a conflict."""
    involvement_level: Literal["none", "observing", "participating", "leading"]
    faction: Literal["a", "b", "neutral"]
    is_manipulated: bool = False
    manipulated_by: Optional[Dict[str, Any]] = None
    resources_committed: Optional[Dict[str, int]] = None
    
    model_config = ConfigDict(extra="forbid")

# Story Beat Models
class StoryBeatParams(BaseModel):
    """Parameters for tracking conflict story beats."""
    conflict_id: int
    path_id: str
    beat_description: str
    involved_npcs: List[int]
    progress_value: float = Field(5.0, ge=0.0, le=100.0)
    
    model_config = ConfigDict(extra="forbid")

class ChoiceOption(BaseModel):
    """One selectable option at a relationship crossroads."""
    option_id: int
    label: str
    consequence: str

    model_config = ConfigDict(extra="forbid")

class Requirement(BaseModel):
    """A single prerequisite for a ritual."""
    name: str
    value: Any

    model_config = ConfigDict(extra="forbid")

class Reward(BaseModel):
    """A single reward granted by a ritual."""
    name: str
    value: Any

    model_config = ConfigDict(extra="forbid")

# Story Director Models
class NPCInfo(BaseModel):
    """Basic information about an NPC."""
    npc_id: int
    name: str
    status: str = "active"
    relationship_level: int = Field(50, ge=0, le=100)
    location: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

class RelationshipCrossroads(BaseModel):
    """A decisive relationship fork with an NPC."""
    link_id: int
    npc_name: str
    crossroads_name: str
    description: str
    choices: List[ChoiceOption] = Field(default_factory=list)
    triggered_at: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

class RelationshipRitual(BaseModel):
    """A repeatable ritual that strengthens/alters a relationship."""
    link_id: int
    npc_name: str
    ritual_name: str
    description: str
    requirements: List[Requirement] = Field(default_factory=list)
    rewards: List[Reward] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

class TriggerEventData(BaseModel):
    """Payload passed to trigger_conflict_event."""
    description: str
    involved_npcs: List[int] = Field(default_factory=list)
    faction_impacts: Optional[List[Requirement]] = None
    severity: int = Field(5, ge=1, le=10)

    model_config = ConfigDict(extra="forbid")

class ConflictEvolutionData(BaseModel):
    """Payload passed to evolve_conflict."""
    description: str
    involved_npcs: List[int] = Field(default_factory=list)
    progress_change: float = Field(0.1, ge=0.0, le=1.0)
    phase_transition: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

# Narrative Event Content Models
class NarrativeEventContent(BaseModel):
    """Abstract base for event-content payloads."""
    title: str
    description: str

    model_config = ConfigDict(extra="forbid")

class NarrativeMomentContent(NarrativeEventContent):
    moment_type: Literal["tension", "revelation", "transition", "climax"]
    scene_text: str
    player_realization: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

class PersonalRevelationContent(NarrativeEventContent):
    revelation_type: Literal[
        "dependency", "obedience", "corruption",
        "willpower", "confidence", "insight"
    ]
    inner_monologue: str

    model_config = ConfigDict(extra="forbid")

class DreamSequenceContent(NarrativeEventContent):
    dream_text: str
    symbols: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

class NPCRevelationContent(NarrativeEventContent):
    npc_id: int
    npc_name: str
    revelation_type: Literal["secret", "past", "motivation", "truth"]
    revelation_text: str
    changes_relationship: bool = False

    model_config = ConfigDict(extra="forbid")

class ConflictBeatGenerationParams(BaseModel):
    """Parameters for generating a conflict story beat."""
    conflict_id: int
    recent_action: Optional[str] = None
    player_choice: Optional[str] = None
    involved_npcs: List[int] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class GeneratedConflictBeat(BaseModel):
    """A generated conflict story beat."""
    beat_description: str
    progress_value: float = Field(ge=0.0, le=100.0)
    impact_summary: str
    path_id: str
    involved_npcs: List[int]
    
    model_config = ConfigDict(extra="forbid")

class DialogueContext(BaseModel):
    """Context for dialogue exchanges"""
    in_dialogue: bool = False
    npc_id: Optional[int] = None
    npc_name: Optional[str] = None
    dialogue_depth: int = 0  # How many exchanges deep
    last_exchange: Optional[datetime] = None
    dialogue_type: Optional[str] = None  # "casual", "interrogation", "flirtation", etc.
    
    model_config = ConfigDict(extra="forbid")

class DialogueModeDetection(BaseModel):
    """Result of dialogue mode detection"""
    should_use_dialogue_mode: bool
    confidence: float = Field(ge=0.0, le=1.0)
    detected_type: Optional[str] = None
    reason: str
    suggested_npc_id: Optional[int] = None
    
    model_config = ConfigDict(extra="forbid")    

# ===== HELPER FUNCTIONS =====

def _get_stage_manipulation_modifier(stage_name: str) -> float:
    """Get manipulation effectiveness modifier based on stage."""
    modifiers = {
        "Innocent Beginning": 0.6,
        "First Doubts": 0.8,
        "Creeping Realization": 1.0,
        "Veil Thinning": 1.3,
        "Full Revelation": 1.5
    }
    return modifiers.get(stage_name, 1.0)

def _get_stage_appropriate_approach(stage_name: str) -> str:
    """Get appropriate manipulation approach for stage."""
    approaches = {
        "Innocent Beginning": "subtle_suggestion",
        "First Doubts": "gentle_guidance",
        "Creeping Realization": "confident_direction",
        "Veil Thinning": "open_manipulation",
        "Full Revelation": "direct_control"
    }
    return approaches.get(stage_name, "adaptive")

# ===== BASE NPC FUNCTIONS (needed by other functions) =====

@function_tool
@track_performance("get_available_npcs")
async def get_available_npcs(
    ctx: RunContextWrapper[ContextType],
    include_unintroduced: bool = False,
    min_dominance: Optional[int] = None,
    gender_filter: Optional[str] = None,
    min_stage: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get available NPCs with narrative stage information.
    
    Args:
        include_unintroduced: Include unintroduced NPCs
        min_dominance: Minimum dominance filter
        gender_filter: Gender filter
        min_stage: Minimum narrative stage filter
        
    Returns:
        List of NPCs with stage information
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get base NPC list
        async with get_db_connection_context() as conn:
            query = """
                SELECT 
                    n.npc_id, n.npc_name, n.dominance, n.cruelty, 
                    n.closeness, n.trust, n.respect, n.intensity, 
                    n.sex, n.faction_affiliations, n.archetype,
                    n.introduced, n.personality_traits
                FROM NPCStats n
                WHERE n.user_id = $1 AND n.conversation_id = $2
            """
            params = [user_id, conversation_id]
            
            conditions = []
            if not include_unintroduced:
                conditions.append("n.introduced = TRUE")
            if min_dominance is not None:
                conditions.append(f"n.dominance >= ${len(params) + 1}")
                params.append(min_dominance)
            if gender_filter:
                conditions.append(f"n.sex = ${len(params) + 1}")
                params.append(gender_filter)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY n.dominance DESC, n.introduced DESC"
            
            rows = await conn.fetch(query, *params)
            
        npcs = []
        for row in rows:
            npc = dict(row)
            npc_id = npc['npc_id']
            
            # Get narrative stage
            stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
            npc['narrative_stage'] = stage.name
            npc['stage_description'] = stage.description
            
            # Apply stage filter if specified
            if min_stage:
                stage_order = {s.name: i for i, s in enumerate(NPC_NARRATIVE_STAGES)}
                if stage_order.get(stage.name, 0) < stage_order.get(min_stage, 0):
                    continue
            
            # Parse JSON fields
            if isinstance(npc['faction_affiliations'], str):
                try:
                    npc['faction_affiliations'] = json.loads(npc['faction_affiliations'])
                except:
                    npc['faction_affiliations'] = []
            
            if isinstance(npc['archetype'], str):
                try:
                    archetype_data = json.loads(npc['archetype'])
                    npc['archetype_list'] = (
                        archetype_data if isinstance(archetype_data, list) 
                        else [archetype_data]
                    )
                except:
                    npc['archetype_list'] = [npc['archetype']]
            else:
                npc['archetype_list'] = []
            
            if isinstance(npc['personality_traits'], str):
                try:
                    npc['personality_traits'] = json.loads(npc['personality_traits'])
                except:
                    npc['personality_traits'] = []
            
            npcs.append(npc)
        
        return npcs
        
    except Exception as e:
        logger.error(f"Error getting available NPCs: {str(e)}", exc_info=True)
        return []

@function_tool
async def get_npc_details(
    ctx: RunContextWrapper[ContextType],
    npc_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about an NPC including their narrative stage.
    
    Args:
        npc_id: ID of the NPC
        
    Returns:
        NPC details with stage information or None if not found
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get base NPC data
        npcs = await get_available_npcs(ctx, include_unintroduced=True)
        npc_data = next((npc for npc in npcs if npc['npc_id'] == npc_id), None)
        
        if not npc_data:
            return None
        
        # Get narrative progression details
        async with get_db_connection_context() as conn:
            progression = await conn.fetchrow("""
                SELECT corruption, dependency, realization_level, 
                       stage_entered_at, stage_history
                FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
        
        if progression:
            npc_data['progression'] = dict(progression)
        
        # Get relationship with player
        from logic.fully_integrated_npc_system import IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, conversation_id)
        await npc_system.initialize()
        npc_data['relationship_with_player'] = await npc_system.get_relationship_with_player(npc_id)
        
        # Get active conflicts
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.conflict_id, c.conflict_name, cs.faction_name, cs.involvement_level
                FROM ConflictStakeholders cs
                JOIN Conflicts c ON cs.conflict_id = c.conflict_id
                WHERE cs.npc_id = $1 AND c.is_active = TRUE
                    AND c.user_id = $2 AND c.conversation_id = $3
            """, npc_id, user_id, conversation_id)
            
        npc_data['active_conflicts'] = [dict(c) for c in conflicts]
        
        return npc_data
        
    except Exception as e:
        logger.error(f"Error getting NPC details: {str(e)}", exc_info=True)
        return None

# ===== CONTEXT MANAGEMENT TOOLS =====

@function_tool
async def generate_conflict_beat(
    ctx: RunContextWrapper[ContextType],
    params: ConflictBeatGenerationParams
) -> Dict[str, Any]:
    """
    Generate a conflict story beat using AI based on current conflict state.
    
    Args:
        params: Conflict beat generation parameters
        
    Returns:
        Generated beat with progress value
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Get conflict details
        conflict = await conflict_integration.get_conflict_details(params.conflict_id)
        if not conflict:
            return {"success": False, "error": "Conflict not found"}
        
        # Get stakeholder information
        stakeholders = conflict.get('stakeholders', [])
        relevant_stakeholders = []
        
        for stakeholder in stakeholders:
            if stakeholder.get('entity_type') == 'npc':
                npc_id = stakeholder.get('npc_id')
                if npc_id in params.involved_npcs or not params.involved_npcs:
                    # Get NPC narrative stage for context
                    stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
                    stakeholder['narrative_stage'] = stage.name
                    relevant_stakeholders.append(stakeholder)
        
        # Get player involvement
        player_involvement = conflict.get('player_involvement', {})
        
        # Get resolution paths
        resolution_paths = conflict.get('resolution_paths', [])
        active_path = next(
            (p for p in resolution_paths if p.get('is_active')), 
            resolution_paths[0] if resolution_paths else None
        )
        
        # Build context for beat generation
        context_prompt = f"""
        Generate a story beat for this conflict:
        
        Conflict: {conflict['conflict_name']}
        Type: {conflict['conflict_type']}
        Current Phase: {conflict['phase']}
        Current Progress: {conflict['progress']}%
        Description: {conflict['description']}
        
        Active Resolution Path: {active_path['path_name'] if active_path else 'None'}
        Path Description: {active_path['description'] if active_path else 'N/A'}
        
        Player Involvement:
        - Level: {player_involvement.get('involvement_level', 'none')}
        - Faction: {player_involvement.get('faction', 'neutral')}
        - Is Manipulated: {player_involvement.get('is_manipulated', False)}
        
        Key Stakeholders:
        """
        
        for stakeholder in relevant_stakeholders[:3]:  # Limit to top 3
            context_prompt += f"""
        - {stakeholder['npc_name']} ({stakeholder['faction_name']}):
          Narrative Stage: {stakeholder.get('narrative_stage', 'Unknown')}
          Public Goal: {stakeholder.get('public_motivation')}
          Hidden Goal: {stakeholder.get('hidden_motivation')}
        """
        
        if params.recent_action:
            context_prompt += f"\n\nRecent Action: {params.recent_action}"
        
        if params.player_choice:
            context_prompt += f"\nPlayer Choice: {params.player_choice}"
        
        context_prompt += """
        
        Generate an appropriate story beat that:
        1. Advances the conflict naturally
        2. Reflects stakeholder agendas
        3. Subtly reinforces power dynamics
        4. Creates opportunities for manipulation
        """
        
        # Generate beat with agent
        result = await Runner.run(
            conflict_beat_writer,
            context_prompt
        )
        
        # Parse the response
        response_text = result.final_output
        lines = response_text.strip().split('\n')
        
        beat_description = ""
        progress_value = conflict['progress'] + 5.0  # Default increment
        impact_summary = ""
        
        for line in lines:
            if line.startswith("BEAT:"):
                beat_description = line.replace("BEAT:", "").strip()
            elif line.startswith("PROGRESS:"):
                try:
                    progress_value = float(line.replace("PROGRESS:", "").strip())
                except:
                    pass
            elif line.startswith("IMPACT:"):
                impact_summary = line.replace("IMPACT:", "").strip()
        
        # Ensure progress makes sense
        progress_increment = progress_value - conflict['progress']
        if progress_increment < 0 or progress_increment > 25:
            # Limit progress jumps
            progress_value = min(conflict['progress'] + 10, 100)
        
        # Create the beat data
        generated_beat = GeneratedConflictBeat(
            beat_description=beat_description or "The conflict continues to develop...",
            progress_value=progress_value,
            impact_summary=impact_summary or "Tensions shift subtly.",
            path_id=active_path['path_id'] if active_path else "default",
            involved_npcs=params.involved_npcs
        )
        
        # Now track the beat using the existing system
        track_result = await conflict_integration.track_story_beat(
            params.conflict_id,
            generated_beat.path_id,
            generated_beat.beat_description,
            generated_beat.involved_npcs,
            generated_beat.progress_value
        )
        
        # Create memory of the beat
        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(
                f"Conflict beat for '{conflict['conflict_name']}': {beat_description[:100]}...",
                "conflict_beat",
                0.7
            )
        
        return {
            "success": True,
            "conflict_id": params.conflict_id,
            "conflict_name": conflict['conflict_name'],
            "generated_beat": generated_beat.model_dump(),
            "track_result": track_result,
            "old_progress": conflict['progress'],
            "new_progress": progress_value
        }
        
    except Exception as e:
        logger.error(f"Error generating conflict beat: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
async def get_optimized_context(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ContextQueryParams] = None
) -> Dict[str, Any]:
    """
    Get optimized context using the comprehensive context system.

    Args:
        params: Query parameters for context retrieval

    Returns:
        Dictionary with comprehensive context information.
    """
    if params is None:
        params = ContextQueryParams()
    
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        context_service = await get_context_service(user_id, conversation_id)
        config = get_config()
        token_budget = params.max_tokens or config.get_token_budget("default")

        context_data = await context_service.get_context(
            input_text=params.query_text,
            context_budget=token_budget,
            use_vector_search=params.use_vector
        )

        # Safely access performance monitor
        perf_monitor = None
        if hasattr(context, 'performance_monitor'):
            perf_monitor = context.performance_monitor
        else:
            try:
                perf_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
            except Exception as pm_err:
                logger.warning(f"Could not get performance monitor instance: {pm_err}")

        if perf_monitor and "token_usage" in context_data:
            try:
                usage = context_data["token_usage"]
                if isinstance(usage, dict):
                    total_tokens = sum(usage.values())
                    perf_monitor.record_token_usage(total_tokens)
                else:
                    logger.warning(f"Unexpected format for token_usage: {type(usage)}")
            except Exception as token_err:
                logger.warning(f"Error recording token usage: {token_err}")

        return context_data
    except Exception as e:
        logger.error(f"Error getting optimized context: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "context_data": None
        }

@function_tool
async def detect_dialogue_mode(
    ctx: RunContextWrapper[ContextType],
    user_input: str,
    current_context: Optional[DialogueContext] = None
) -> DialogueModeDetection:
    """
    Detect whether the player input should trigger dialogue mode.
    
    Args:
        user_input: The player's input
        current_context: Current dialogue context if any
        
    Returns:
        Detection result with confidence and reasoning
    """
    context = ctx.context
    
    try:
        # Get nearby NPCs with error handling
        try:
            npcs = await get_available_npcs(ctx, include_unintroduced=False)
        except Exception as npc_error:
            logger.warning(f"Error getting NPCs for dialogue detection: {npc_error}")
            npcs = []
        
        # Get recent memories to check for ongoing conversation
        recent_memories = []
        recent_dialogue_content = []
        try:
            memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
            recent_memories = await memory_manager.search_memories(
                query_text="conversation dialogue",
                limit=3,
                use_vector=True
            )
            
            # Extract memory content safely
            for memory in recent_memories:
                if hasattr(memory, 'content'):
                    recent_dialogue_content.append(memory.content[:100] + '...')
                elif isinstance(memory, dict) and 'content' in memory:
                    recent_dialogue_content.append(memory['content'][:100] + '...')
        except Exception as mem_error:
            logger.warning(f"Error getting recent memories: {mem_error}")
        
        # Check if we're already in dialogue
        already_in_dialogue = False
        time_since_last = None
        if current_context:
            already_in_dialogue = (
                current_context.in_dialogue and 
                current_context.last_exchange is not None
            )
            if already_in_dialogue and current_context.last_exchange:
                time_since_last = (datetime.now() - current_context.last_exchange).seconds
                already_in_dialogue = time_since_last < 300  # 5 min timeout
        
        # Initialize detection variables
        should_use = False
        confidence = 0.0
        detected_type = "narrative"
        reason = "Default to narrative mode"
        
        # Analyze input patterns
        input_lower = user_input.lower().strip()
        word_count = len(user_input.split())
        
        # Pattern-based detection with confidence scores
        patterns = {
            # Direct questions
            (input_lower.endswith('?') and word_count < 15): {
                'use': True, 'confidence': 0.9, 'type': 'question',
                'reason': 'Direct question under 15 words'
            },
            # Direct speech commands
            (any(input_lower.startswith(prefix) for prefix in ['say ', 'tell ', 'ask ', 'reply ', 'answer '])): {
                'use': True, 'confidence': 0.85, 'type': 'direct_speech',
                'reason': 'Starts with speech verb'
            },
            # Quoted speech
            ('"' in user_input or "'" in user_input): {
                'use': True, 'confidence': 0.8, 'type': 'quoted_speech',
                'reason': 'Contains quoted text'
            },
            # Short conversational responses
            (word_count < 10 and not any(action in input_lower for action in [
                'walk', 'go', 'move', 'take', 'pick', 'grab', 'look', 'examine',
                'use', 'open', 'close', 'push', 'pull', 'attack', 'flee'
            ])): {
                'use': True, 'confidence': 0.7, 'type': 'casual',
                'reason': 'Short input without action verbs'
            },
            # Conversational keywords
            (any(word in input_lower for word in [
                'yes', 'no', 'maybe', 'okay', 'sure', 'fine', 'alright',
                'hello', 'hi', 'goodbye', 'bye', 'thanks', 'sorry'
            ]) and word_count < 5): {
                'use': True, 'confidence': 0.75, 'type': 'response',
                'reason': 'Brief conversational response'
            },
            # Continuing conversation
            (already_in_dialogue and word_count < 20): {
                'use': True, 'confidence': 0.8, 'type': 'continuation',
                'reason': 'Continuing existing dialogue'
            }
        }
        
        # Apply pattern matching
        for condition, result in patterns.items():
            if condition:
                should_use = result['use']
                confidence = max(confidence, result['confidence'])
                detected_type = result['type']
                reason = result['reason']
                break
        
        # Adjust confidence based on context
        if already_in_dialogue:
            confidence = min(1.0, confidence + 0.1)  # Boost confidence if continuing
            reason += f" (continuing conversation, {time_since_last}s ago)"
        
        # Check for multi-NPC or complex scenarios that should NOT use dialogue mode
        npc_count = len([npc for npc in npcs if npc['npc_name'].lower() in input_lower])
        if npc_count > 1:
            should_use = False
            confidence = 0.2
            detected_type = "multi_npc"
            reason = "Multiple NPCs addressed"
        
        # Check for time-passage indicators
        time_indicators = ['wait', 'rest', 'sleep', 'later', 'tomorrow', 'hour', 'minute']
        if any(indicator in input_lower for indicator in time_indicators):
            should_use = False
            confidence = 0.3
            detected_type = "time_passage"
            reason = "Input suggests time passage"
        
        # Find the most likely NPC if in dialogue mode
        suggested_npc_id = None
        if should_use and npcs:
            # First, check if an NPC name is directly mentioned
            mentioned_npc = None
            for npc in npcs:
                npc_name_lower = npc['npc_name'].lower()
                # Check for exact name or common variations
                if (npc_name_lower in input_lower or 
                    npc_name_lower.split()[0] in input_lower):  # First name only
                    mentioned_npc = npc
                    confidence = min(1.0, confidence + 0.1)
                    break
            
            if mentioned_npc:
                suggested_npc_id = mentioned_npc['npc_id']
            elif current_context and current_context.npc_id and already_in_dialogue:
                # Use the NPC from ongoing conversation
                suggested_npc_id = current_context.npc_id
            elif len(npcs) == 1:
                # Only one NPC nearby
                suggested_npc_id = npcs[0]['npc_id']
                confidence = min(1.0, confidence + 0.05)
            elif npcs:
                # Multiple NPCs, use the one with highest closeness/dominance
                sorted_npcs = sorted(
                    npcs, 
                    key=lambda n: (n.get('closeness', 0) + n.get('dominance', 0)), 
                    reverse=True
                )
                suggested_npc_id = sorted_npcs[0]['npc_id']
                confidence *= 0.9  # Reduce confidence when guessing
        
        # Final validation
        if should_use and not suggested_npc_id and not already_in_dialogue:
            # Can't determine NPC, fall back to narrative
            should_use = False
            confidence = 0.4
            reason = "No clear NPC target for dialogue"
        
        return DialogueModeDetection(
            should_use_dialogue_mode=should_use,
            confidence=confidence,
            detected_type=detected_type,
            reason=reason,
            suggested_npc_id=suggested_npc_id
        )
        
    except Exception as e:
        logger.error(f"Error detecting dialogue mode: {e}", exc_info=True)
        return DialogueModeDetection(
            should_use_dialogue_mode=False,
            confidence=0.0,
            detected_type="error",
            reason=f"Detection error: {str(e)}"
        )

@function_tool
async def generate_dialogue_exchange(
    ctx: RunContextWrapper[ContextType],
    user_input: str,
    npc_id: int,
    dialogue_context: Optional[DialogueContext] = None
) -> Dict[str, Any]:
    """
    Generate a quick dialogue exchange without full narrative wrapper.
    
    Args:
        user_input: Player's dialogue/input
        npc_id: ID of the NPC in conversation
        dialogue_context: Current dialogue context
        
    Returns:
        Quick dialogue response with metadata
    """
    context = ctx.context
    
    try:
        # Get NPC details
        npc_details = await get_npc_details(ctx, npc_id)
        if not npc_details:
            return {
                "success": False, 
                "error": "NPC not found",
                "fallback_to_narrative": True
            }
        
        # Get NPC's narrative stage
        stage = await get_npc_narrative_stage(
            context.user_id, 
            context.conversation_id, 
            npc_id
        )
        
        # Build context about recent exchanges
        dialogue_depth = 0
        recent_context = ""
        if dialogue_context:
            dialogue_depth = dialogue_context.dialogue_depth
            if dialogue_depth > 0:
                recent_context = f"\nConversation depth: {dialogue_depth} exchanges"
                if dialogue_depth > 3:
                    recent_context += "\n(Conversation getting long - consider wrapping up)"
        
        # Get recent memories with this NPC for better context
        memory_context = ""
        try:
            memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
            npc_memories = await memory_manager.search_memories(
                query_text=f"dialogue with {npc_details['npc_name']}",
                tags=[f"npc_{npc_id}", "dialogue"],
                limit=2,
                use_vector=True
            )
            if npc_memories:
                memory_context = "\nRecent exchanges:"
                for mem in npc_memories[:2]:
                    if hasattr(mem, 'content'):
                        memory_context += f"\n- {mem.content[:100]}..."
        except Exception as mem_error:
            logger.warning(f"Could not retrieve dialogue memories: {mem_error}")
        
        # Determine appropriate response style based on stage
        stage_guidance = {
            "Innocent Beginning": "Friendly and casual, no power dynamics",
            "First Doubts": "Subtly suggestive, gentle steering",
            "Creeping Realization": "More confident, occasional gentle commands",
            "Veil Thinning": "Openly manipulative, direct but not harsh",
            "Full Revelation": "Complete control, expects obedience"
        }
        
        style_guide = stage_guidance.get(stage.name, "Natural and character-appropriate")
        
        # Build the prompt
        prompt = f"""Generate a quick dialogue response for {npc_details['npc_name']}.

Player says: "{user_input}"

NPC Profile:
- Name: {npc_details['npc_name']}
- Personality: {', '.join(npc_details.get('personality_traits', ['confident']))}
- Narrative Stage: {stage.name}
- Response Style: {style_guide}
- Dominance: {npc_details.get('dominance', 50)}/100
{recent_context}
{memory_context}

Generate ONLY a dialogue line (1-3 sentences max). 
Keep it conversational and natural.
Match their personality and current narrative stage.
Do NOT include character names or action descriptions."""

        # Use the dialogue generator agent
        from story_agent.specialized_agents import Agent, Runner, ModelSettings
        
        dialogue_generator = Agent(
            name="QuickDialogue",
            instructions="""You generate brief, natural dialogue responses for NPCs.
Keep responses under 3 sentences. Match the character's personality and narrative stage.
Output only the spoken words, no names or actions.""",
            model="gpt-4",
            model_settings=ModelSettings(
                temperature=0.7,
                max_tokens=100
            )
        )
        
        result = await Runner.run(dialogue_generator, prompt)
        dialogue_line = result.final_output.strip()
        
        # Clean up the response
        # Remove character name if accidentally included
        if ':' in dialogue_line:
            parts = dialogue_line.split(':', 1)
            if len(parts) > 1:
                dialogue_line = parts[1].strip()
        
        # Remove quotes if wrapped
        dialogue_line = dialogue_line.strip('"\'')
        
        # Update or create dialogue context
        new_depth = dialogue_depth + 1
        new_context = DialogueContext(
            in_dialogue=True,
            npc_id=npc_id,
            npc_name=npc_details['npc_name'],
            dialogue_depth=new_depth,
            last_exchange=datetime.now(),
            dialogue_type=dialogue_context.dialogue_type if dialogue_context else "casual"
        )
        
        # Determine if we should suggest exiting dialogue
        should_exit = False
        exit_reason = None
        
        if new_depth > 6:
            should_exit = True
            exit_reason = "Conversation running long"
        elif any(farewell in dialogue_line.lower() for farewell in ['goodbye', 'farewell', 'see you', 'talk later']):
            should_exit = True
            exit_reason = "Natural conversation end"
        elif len(user_input.split()) > 20:
            should_exit = True
            exit_reason = "Player input suggests narrative action"
        
        # Create a memory of this exchange
        try:
            memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
            await memory_manager.add_memory(
                content=f"Dialogue: Player: '{user_input}' | {npc_details['npc_name']}: '{dialogue_line}'",
                memory_type="dialogue",
                importance=0.4,
                tags=["dialogue", "conversation", f"npc_{npc_id}", stage.name.lower().replace(" ", "_")],
                metadata={
                    "dialogue_mode": True,
                    "exchange_number": new_depth,
                    "npc_id": npc_id,
                    "stage": stage.name
                }
            )
        except Exception as mem_error:
            logger.warning(f"Could not store dialogue memory: {mem_error}")
        
        return {
            "success": True,
            "npc_id": npc_id,
            "npc_name": npc_details['npc_name'],
            "dialogue": dialogue_line,
            "dialogue_context": new_context.model_dump(),
            "stage": stage.name,
            "dominance": npc_details.get('dominance', 50),
            "should_exit_dialogue": should_exit,
            "exit_reason": exit_reason,
            "exchange_number": new_depth
        }
        
    except Exception as e:
        logger.error(f"Error generating dialogue exchange: {e}", exc_info=True)
        return {
            "success": False, 
            "error": str(e),
            "fallback_to_narrative": True
        }

@function_tool
async def exit_dialogue_mode(
    ctx: RunContextWrapper[ContextType],
    dialogue_context: DialogueContext,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Gracefully exit dialogue mode and return to full narrative.
    
    Args:
        dialogue_context: Current dialogue context
        reason: Optional reason for exiting
        
    Returns:
        Exit confirmation with transition text
    """
    context = ctx.context
    
    try:
        if not dialogue_context or not dialogue_context.in_dialogue:
            return {
                "success": True,
                "already_exited": True,
                "message": "Not currently in dialogue mode"
            }
        
        # Create a summary of the conversation
        summary_parts = [
            f"Ended {dialogue_context.dialogue_depth}-exchange conversation with {dialogue_context.npc_name}"
        ]
        
        if reason:
            summary_parts.append(f"Reason: {reason}")
        
        # Determine transition text based on how conversation ended
        transition_text = ""
        if dialogue_context.dialogue_depth > 5:
            transition_text = f"The conversation with {dialogue_context.npc_name} winds down."
        elif reason and "farewell" in reason.lower():
            transition_text = f"You part ways with {dialogue_context.npc_name}."
        else:
            transition_text = f"The moment passes."
        
        # Store conversation summary
        try:
            memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
            await memory_manager.add_memory(
                content=" ".join(summary_parts),
                memory_type="dialogue_end",
                importance=0.5,
                tags=["dialogue", "conversation_end", f"npc_{dialogue_context.npc_id}"],
                metadata={
                    "total_exchanges": dialogue_context.dialogue_depth,
                    "dialogue_type": dialogue_context.dialogue_type
                }
            )
        except Exception as mem_error:
            logger.warning(f"Could not store dialogue summary: {mem_error}")
        
        return {
            "success": True,
            "exited_dialogue": True,
            "message": "Returning to full narrative mode",
            "transition_text": transition_text,
            "conversation_summary": {
                "npc_name": dialogue_context.npc_name,
                "total_exchanges": dialogue_context.dialogue_depth,
                "duration_seconds": (
                    (datetime.now() - dialogue_context.last_exchange).seconds 
                    if dialogue_context.last_exchange else 0
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error exiting dialogue mode: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Error transitioning to narrative mode"
        }

@function_tool
async def get_dialogue_suggestions(
    ctx: RunContextWrapper[ContextType],
    npc_id: int,
    dialogue_context: Optional[DialogueContext] = None,
    context_hint: Optional[str] = None
) -> List[str]:
    """
    Get contextually appropriate dialogue suggestions for the player.
    
    Args:
        npc_id: ID of the NPC in conversation
        dialogue_context: Current dialogue context
        context_hint: Optional hint about conversation topic
        
    Returns:
        List of 3-5 suggested player dialogue options
    """
    context = ctx.context
    
    try:
        # Get NPC details and stage
        npc_details = await get_npc_details(ctx, npc_id)
        if not npc_details:
            return ["What?", "I see.", "I should go."]
            
        stage = await get_npc_narrative_stage(
            context.user_id, 
            context.conversation_id, 
            npc_id
        )
        
        # Base suggestions on narrative stage and context
        suggestions = []
        
        # Stage-specific suggestions
        stage_suggestions = {
            "Innocent Beginning": [
                "How have you been?",
                "What do you think about that?",
                "That's interesting...",
                "Want to hang out sometime?",
                "I should get going."
            ],
            "First Doubts": [
                "You seem different lately.",
                "Why do you say that?",
                "I'm not sure I understand.",
                "That's... an interesting way to put it.",
                "I need to think about this."
            ],
            "Creeping Realization": [
                "Yes, of course.",
                "Do I have to?",
                "What are you really asking?",
                "I... I'll do it.",
                "This feels strange."
            ],
            "Veil Thinning": [
                "As you wish.",
                "I understand.",
                "Yes, Ma'am.",
                "Please, I need a moment.",
                "Why does this feel so natural?"
            ],
            "Full Revelation": [
                "Yes, Mistress.",
                "I obey.",
                "Thank you for your guidance.",
                "How may I serve?",
                "I am yours."
            ]
        }
        
        # Get base suggestions for stage
        base_suggestions = stage_suggestions.get(stage.name, stage_suggestions["Innocent Beginning"])
        
        # Adjust based on dialogue depth if available
        if dialogue_context and dialogue_context.dialogue_depth > 3:
            # Add exit options for long conversations
            base_suggestions = base_suggestions[:-1] + ["I should go.", "We'll talk later."]
        
        # Add context-specific options if hint provided
        if context_hint:
            hint_lower = context_hint.lower()
            if "conflict" in hint_lower or "problem" in hint_lower:
                base_suggestions.insert(0, "What should we do about it?")
            elif "request" in hint_lower or "task" in hint_lower:
                base_suggestions.insert(0, "What do you need me to do?")
            elif "personal" in hint_lower:
                base_suggestions.insert(0, "You can tell me anything.")
        
        # Ensure variety and appropriateness
        suggestions = base_suggestions[:5]  # Limit to 5 options
        
        # Adjust formality based on dominance
        if npc_details.get('dominance', 50) > 70 and stage.name != "Innocent Beginning":
            # Make suggestions more respectful/submissive
            suggestions = [s.replace(".", ", Ma'am.") if not any(
                word in s.lower() for word in ['ma\'am', 'mistress', 'yes', 'no']
            ) else s for s in suggestions]
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting dialogue suggestions: {e}", exc_info=True)
        return ["What?", "I see.", "Goodbye."]

@function_tool
async def analyze_dialogue_flow(
    ctx: RunContextWrapper[ContextType],
    dialogue_history: List[Dict[str, str]],
    npc_id: int
) -> Dict[str, Any]:
    """
    Analyze the flow and dynamics of an ongoing dialogue.
    
    Args:
        dialogue_history: List of dialogue exchanges
        npc_id: ID of the NPC in conversation
        
    Returns:
        Analysis of dialogue progression and dynamics
    """
    context = ctx.context
    
    try:
        # Get NPC details
        npc_details = await get_npc_details(ctx, npc_id)
        if not npc_details:
            return {"success": False, "error": "NPC not found"}
        
        stage = await get_npc_narrative_stage(
            context.user_id, 
            context.conversation_id, 
            npc_id
        )
        
        # Analyze dialogue patterns
        total_exchanges = len(dialogue_history)
        player_word_count = sum(len(ex.get('player', '').split()) for ex in dialogue_history)
        npc_word_count = sum(len(ex.get('npc', '').split()) for ex in dialogue_history)
        
        # Detect power dynamics shifts
        submission_indicators = ['yes', 'okay', 'sorry', 'please', "ma'am", 'mistress']
        resistance_indicators = ['no', 'why', 'but', "don't", "can't", "won't"]
        
        player_submission_score = 0
        player_resistance_score = 0
        
        for exchange in dialogue_history:
            player_text = exchange.get('player', '').lower()
            for indicator in submission_indicators:
                if indicator in player_text:
                    player_submission_score += 1
            for indicator in resistance_indicators:
                if indicator in player_text:
                    player_resistance_score += 1
        
        # Determine dialogue dynamic
        if player_submission_score > player_resistance_score * 2:
            dynamic = "submissive"
        elif player_resistance_score > player_submission_score * 2:
            dynamic = "resistant"
        else:
            dynamic = "balanced"
        
        # Check for escalation
        escalation_detected = False
        if total_exchanges > 3:
            # Compare early vs late exchanges
            early_submission = sum(1 for ex in dialogue_history[:2] 
                                 for word in submission_indicators 
                                 if word in ex.get('player', '').lower())
            late_submission = sum(1 for ex in dialogue_history[-2:] 
                                for word in submission_indicators 
                                if word in ex.get('player', '').lower())
            escalation_detected = late_submission > early_submission
        
        # Recommendations
        recommendations = []
        if total_exchanges > 5:
            recommendations.append("Consider wrapping up - conversation getting long")
        if dynamic == "resistant" and stage.name in ["Veil Thinning", "Full Revelation"]:
            recommendations.append("Player resistance conflicts with established dynamic")
        if escalation_detected:
            recommendations.append("Power dynamic escalation detected")
        
        return {
            "success": True,
            "total_exchanges": total_exchanges,
            "average_player_words": player_word_count / max(1, total_exchanges),
            "average_npc_words": npc_word_count / max(1, total_exchanges),
            "dialogue_dynamic": dynamic,
            "submission_score": player_submission_score,
            "resistance_score": player_resistance_score,
            "escalation_detected": escalation_detected,
            "npc_stage": stage.name,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error analyzing dialogue flow: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
async def retrieve_relevant_memories(
    ctx: RunContextWrapper[ContextType],
    params: MemorySearchParams
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using vector search.
    
    Args:
        params: Memory search parameters
        
    Returns:
        List of relevant memories.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Create a MemorySearchRequest object
        request = MemorySearchRequest(
            query_text=params.query_text,
            memory_types=[params.memory_type] if params.memory_type else None,
            limit=params.limit,
            use_vector=params.use_vector
        )
        
        # Call the standalone function with the right parameters
        memory_result = await search_memories_tool(ctx, user_id, conversation_id, request)
        
        # Process the result
        memory_dicts = []
        if memory_result and hasattr(memory_result, 'memories'):
            for memory in memory_result.memories:
                if hasattr(memory, 'to_dict'):
                    memory_dicts.append(memory.to_dict())
                elif isinstance(memory, dict):
                    memory_dicts.append(memory)
                    
        return memory_dicts
    except Exception as e:
        logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
        return []

@function_tool
async def store_narrative_memory(
    ctx: RunContextWrapper[ContextType],
    params: StoreMemoryParams
) -> Dict[str, Any]:
    """
    Store a narrative memory in the memory system.

    Args:
        params: Memory storage parameters

    Returns:
        Stored memory information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle default tags
    tags = params.tags
    if not tags:
        tags = [params.memory_type, "story_director"]

    try:
        memory_manager = await get_memory_manager(user_id, conversation_id)

        memory_id = await memory_manager.add_memory(
            content=params.content,
            memory_type=params.memory_type,
            importance=params.importance,
            tags=tags,
            metadata={
                "source": "story_director_tool", 
                "timestamp": datetime.now().isoformat()
            }
        )

        # Safely check for narrative_manager
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            try:
                await context.narrative_manager.add_interaction(
                    content=params.content,
                    importance=params.importance,
                    tags=tags
                )
            except Exception as nm_err:
                logger.warning(f"Error calling narrative_manager.add_interaction: {nm_err}")

        return {
            "memory_id": memory_id,
            "content": params.content,
            "memory_type": params.memory_type,
            "importance": params.importance,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error storing narrative memory: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
async def search_by_vector(
    ctx: RunContextWrapper[ContextType],
    params: VectorSearchParams
) -> List[Dict[str, Any]]:
    """
    Search for entities by semantic similarity using vector search.

    Args:
        params: Vector search parameters

    Returns:
        List of semantically similar entities.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        vector_service = await get_vector_service(user_id, conversation_id)
        if not vector_service or not vector_service.enabled:
            logger.info("Vector service is not enabled or available.")
            return []

        results = await vector_service.search_entities(
            query_text=params.query_text,
            entity_types=params.entity_types,
            top_k=params.top_k,
            hybrid_ranking=True
        )
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}", exc_info=True)
        return []

@function_tool
async def get_summarized_narrative_context(
    ctx: RunContextWrapper[ContextType],
    query: str,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get automatically summarized narrative context using progressive summarization.

    Args:
        query: Query for relevance matching.
        max_tokens: Maximum tokens for context (default: 1000)

    Returns:
        Summarized narrative context or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle default
    actual_max_tokens = max_tokens if max_tokens is not None else 1000

    try:
        narrative_manager = None
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            narrative_manager = context.narrative_manager
        else:
            try:
                from story_agent.progressive_summarization import RPGNarrativeManager
                dsn = 'DATABASE_URL_NOT_FOUND'
                try:
                    async with get_db_connection_context() as conn:
                        pool = getattr(conn, '_pool', None)
                        connect_kwargs = getattr(pool, '_connect_kwargs', {}) if pool else {}
                        dsn = connect_kwargs.get('dsn', dsn)
                except Exception as db_conn_err:
                    logger.warning(f"Could not get DSN from DB connection: {db_conn_err}")

                narrative_manager = RPGNarrativeManager(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    db_connection_string=dsn
                )
                await narrative_manager.initialize()

                try:
                    if hasattr(context, '__dict__') or isinstance(context, object):
                        context.narrative_manager = narrative_manager
                    else:
                        logger.warning(
                            "Context object does not support attribute assignment."
                        )
                except Exception as assign_err:
                    logger.warning(
                        f"Could not store narrative_manager on context: {assign_err}"
                    )

            except ImportError:
                logger.error("Module 'story_agent.progressive_summarization' not found.")
                return {
                    "success": False,
                    "error": "Narrative manager component not available.", 
                    "memories": [], 
                    "arcs": []
                }
            except Exception as init_error:
                logger.error(
                    f"Error initializing narrative manager: {init_error}", 
                    exc_info=True
                )
                return {
                    "success": False,
                    "error": "Narrative manager initialization failed.", 
                    "memories": [], 
                    "arcs": []
                }

        if not narrative_manager:
            return {
                "success": False,
                "error": "Narrative manager could not be initialized.", 
                "memories": [], 
                "arcs": []
            }

        context_data = await narrative_manager.get_current_narrative_context(
            query,
            actual_max_tokens
        )
        return context_data
    except Exception as e:
        logger.error(
            f"Error getting summarized narrative context: {str(e)}", 
            exc_info=True
        )
        return {"success": False, "error": str(e), "memories": [], "arcs": []}

# ===== NPC NARRATIVE TOOLS =====

@function_tool
async def get_npc_narrative_overview(
    ctx: RunContextWrapper[ContextType]
) -> Dict[str, Any]:
    """
    Get a comprehensive overview of all NPC narrative progressions.
    
    Returns:
        Overview of narrative stages across all NPCs
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        overview = await get_relationship_overview(user_id, conversation_id)
        
        # Enhance with narrative-specific analysis
        stage_analysis = {
            "most_advanced_stage": None,
            "average_progression": 0,
            "npcs_by_stage": {},
            "ready_for_revelation": [],
            "manipulation_candidates": []
        }
        
        # Analyze NPC distributions
        for stage_name in [
            "Innocent Beginning", 
            "First Doubts", 
            "Creeping Realization", 
            "Veil Thinning", 
            "Full Revelation"
        ]:
            npcs_in_stage = overview['by_stage'].get(stage_name, [])
            stage_analysis['npcs_by_stage'][stage_name] = len(npcs_in_stage)
            
            if npcs_in_stage and not stage_analysis['most_advanced_stage']:
                stage_analysis['most_advanced_stage'] = stage_name
        
        # Calculate average progression
        total_progression = 0
        npc_count = 0
        
        for npc in overview.get('relationships', []):
            stage_index = 0
            for i, stage in enumerate(NPC_NARRATIVE_STAGES):
                if stage.name == npc['stage']:
                    stage_index = i
                    break
            total_progression += (stage_index / (len(NPC_NARRATIVE_STAGES) - 1)) * 100
            npc_count += 1
            
            # Check for revelation readiness
            if stage_index > 0:  # Not in Innocent Beginning
                stage_analysis['ready_for_revelation'].append({
                    'npc_id': npc['npc_id'],
                    'npc_name': npc['npc_name'],
                    'stage': npc['stage']
                })
            
            # Check for manipulation candidates
            if stage_index >= 2:  # Creeping Realization or later
                stage_analysis['manipulation_candidates'].append({
                    'npc_id': npc['npc_id'],
                    'npc_name': npc['npc_name'],
                    'stage': npc['stage']
                })
        
        if npc_count > 0:
            stage_analysis['average_progression'] = total_progression / npc_count
        
        overview['narrative_analysis'] = stage_analysis
        
        return overview
        
    except Exception as e:
        logger.error(f"Error getting NPC narrative overview: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "relationships": [], "narrative_analysis": {}}

@function_tool
async def check_all_npc_revelations(
    ctx: RunContextWrapper[ContextType]
) -> List[Dict[str, Any]]:
    """
    Check for potential revelations across all NPCs.
    
    Returns:
        List of potential NPC-specific revelations
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        npcs = await get_available_npcs(ctx, include_unintroduced=False)
        revelations = []
        
        for npc in npcs:
            npc_id = npc['npc_id']
            
            # Check narrative stage first
            stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
            
            # Only check for revelations if past Innocent Beginning
            if stage.name != "Innocent Beginning":
                revelation = await check_for_npc_revelation(user_id, conversation_id, npc_id)
                if revelation:
                    revelation['current_stage'] = stage.name
                    revelations.append(revelation)
        
        return revelations
        
    except Exception as e:
        logger.error(f"Error checking all NPC revelations: {str(e)}", exc_info=True)
        return []

@function_tool
async def generate_dynamic_personal_revelation(
    ctx: RunContextWrapper[ContextType],
    npc_id: int,
    revelation_type: str
) -> Dict[str, Any]:
    """
    Generate a dynamic personal revelation using AI based on NPC relationship context.
    
    Args:
        npc_id: ID of the NPC
        revelation_type: Type of revelation
        
    Returns:
        Generated revelation
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get NPC details
        npc_details = await get_npc_details(ctx, npc_id)
        if not npc_details:
            return {"success": False, "error": "NPC not found"}
        
        # Get narrative stage
        stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
        
        # Get recent memories with this NPC
        memory_manager = await get_memory_manager(user_id, conversation_id)
        npc_memories = await memory_manager.search_memories(
            query_text=f"{npc_details['npc_name']} interaction",
            tags=[f"npc_{npc_id}"],
            limit=5,
            use_vector=True
        )
        
        # Format memories for context
        memory_context = "\n".join([
            f"- {mem.content}" for mem in npc_memories[:3]
        ]) if npc_memories else "No specific memories"
        
        # Create prompt for revelation generator
        prompt = f"""
        Generate a personal revelation for the player about their relationship with {npc_details['npc_name']}.
        
        NPC Details:
        - Name: {npc_details['npc_name']}
        - Personality: {npc_details.get('personality_traits', [])}
        - Dominance: {npc_details.get('dominance', 50)}/100
        - Current Narrative Stage: {stage.name} ({stage.description})
        
        Revelation Type: {revelation_type}
        
        Recent Interactions:
        {memory_context}
        
        The revelation should be a first-person internal monologue that shows the player
        becoming aware of how their relationship with {npc_details['npc_name']} has changed.
        It should be appropriate to the current narrative stage and the specific revelation type.
        """
        
        # Generate using the agent
        result = await Runner.run(
            revelation_generator,
            prompt
        )
        
        inner_monologue = result.final_output
        
        # Create journal entry
        async with get_db_connection_context() as conn:
            canon_ctx = SimpleContext(user_id, conversation_id)
            
            journal_id = await canon.create_journal_entry(
                ctx=canon_ctx,
                conn=conn,
                entry_type='personal_revelation',
                entry_text=inner_monologue,
                revelation_types=revelation_type,
                narrative_moment=None,
                fantasy_flag=False,
                intensity_level=0,
                importance=0.8,
                tags=[
                    revelation_type, 
                    "revelation", 
                    f"npc_{npc_id}", 
                    stage.name.lower().replace(" ", "_")
                ]
            )
        
        return {
            "type": "personal_revelation",
            "npc_id": npc_id,
            "npc_name": npc_details['npc_name'],
            "narrative_stage": stage.name,
            "revelation_type": revelation_type,
            "inner_monologue": inner_monologue,
            "journal_id": journal_id,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error generating dynamic revelation: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
async def generate_multi_npc_dream(
    ctx: RunContextWrapper[ContextType],
    primary_npc_ids: List[int],
    dream_theme: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a dream sequence involving multiple NPCs at different narrative stages.
    
    Args:
        primary_npc_ids: List of NPC IDs to feature
        dream_theme: Optional theme for the dream
        
    Returns:
        Generated dream sequence
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get details for each NPC
        npc_contexts = []
        for npc_id in primary_npc_ids[:3]:  # Limit to 3 for coherence
            npc_details = await get_npc_details(ctx, npc_id)
            if npc_details:
                stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
                npc_contexts.append({
                    'name': npc_details['npc_name'],
                    'stage': stage.name,
                    'dominance': npc_details.get('dominance', 50),
                    'archetype': npc_details.get('archetype_list', [])
                })
        
        if not npc_contexts:
            return {"success": False, "error": "No valid NPCs found"}
        
        # Build context for dream generation
        npc_descriptions = "\n".join([
            f"- {npc['name']}: Stage '{npc['stage']}', Dominance {npc['dominance']}"
            for npc in npc_contexts
        ])
        
        prompt = f"""
        Create a symbolic dream sequence featuring these NPCs:
        {npc_descriptions}
        
        Dream Theme: {dream_theme or 'Control and transformation'}
        
        The dream should:
        - Feature all NPCs but reflect their different stages of control
        - Use surreal dream logic and symbolism
        - Show NPCs at different stages interacting in impossible ways
        - Include symbolic representations of the player's loss of autonomy
        - Be unsettling but not explicitly frightening
        
        Write the dream in second person, present tense.
        """
        
        # Generate dream
        result = await Runner.run(
            dream_weaver,
            prompt
        )
        
        dream_text = result.final_output
        
        # Extract symbols from the dream
        symbols = []
        for npc in npc_contexts:
            symbols.append(npc['name'])
        symbols.extend(['control', 'transformation', 'identity'])
        
        # Create journal entry
        async with get_db_connection_context() as conn:
            canon_ctx = SimpleContext(user_id, conversation_id)
            
            journal_id = await canon.create_journal_entry(
                ctx=canon_ctx,
                conn=conn,
                entry_type='dream_sequence',
                entry_text=dream_text,
                revelation_types=None,
                narrative_moment=True,
                fantasy_flag=True,
                intensity_level=0,
                importance=0.7,
                tags=["dream", "multi_npc"] + [f"npc_{npc_id}" for npc_id in primary_npc_ids]
            )
        
        return {
            "type": "dream_sequence",
            "text": dream_text,
            "symbols": symbols,
            "featured_npcs": npc_contexts,
            "journal_id": journal_id,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error generating multi-NPC dream: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

# ===== ACTIVITY TOOLS =====

@function_tool
async def suggest_stage_appropriate_activity(
    ctx: RunContextWrapper[ContextType],
    npc_id: int,
    setting: Optional[str] = None,
    intensity_override: Optional[int] = None
) -> Dict[str, Any]:
    """
    Suggest an activity appropriate to the NPC's current narrative stage.
    
    Args:
        npc_id: ID of the NPC
        setting: Current location/setting
        intensity_override: Optional intensity override
        
    Returns:
        Activity suggestion
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get NPC details and stage
        npc_details = await get_npc_details(ctx, npc_id)
        if not npc_details:
            return {"success": False, "error": "NPC not found"}
            
        stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
        
        # Map stage to intensity if not overridden
        if intensity_override is None:
            stage_intensity_map = {
                "Innocent Beginning": 1,
                "First Doubts": 2,
                "Creeping Realization": 3,
                "Veil Thinning": 4,
                "Full Revelation": 5
            }
            intensity = stage_intensity_map.get(stage.name, 2)
        else:
            intensity = intensity_override
        
        # Get current setting if not provided
        if not setting:
            comprehensive_context = await get_optimized_context(ctx)
            setting = comprehensive_context.get("current_location", "Default")
        
        # Build prompt for activity suggester
        prompt = f"""
        Suggest an activity for {npc_details['npc_name']} to engage the player in.
        
        NPC Profile:
        - Personality: {npc_details.get('personality_traits', [])}
        - Archetypes: {npc_details.get('archetype_list', [])}
        - Dominance: {npc_details.get('dominance', 50)}/100
        - Narrative Stage: {stage.name} - {stage.description}
        
        Context:
        - Setting: {setting}
        - Intensity Level: {intensity}/5
        - Stage Consideration: Activities should match the openness of control for this stage
        
        Provide:
        1. Activity name
        2. Brief description
        3. How it subtly reinforces the power dynamic
        4. Expected player response options
        """
        
        # Generate activity
        result = await Runner.run(
            activity_suggester,
            prompt
        )
        
        # Parse the generated content
        activity_text = result.final_output
        
        # Extract structured data (simplified - could use structured output)
        lines = activity_text.strip().split('\n')
        activity_data = {
            "npc_id": npc_id,
            "npc_name": npc_details['npc_name'],
            "narrative_stage": stage.name,
            "activity_name": lines[0] if lines else "Conversation",
            "description": activity_text,
            "intensity": intensity,
            "setting": setting,
            "stage_appropriate": True,
            "success": True
        }
        
        return activity_data
        
    except Exception as e:
        logger.error(f"Error suggesting stage-appropriate activity: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
async def analyze_manipulation_opportunities(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    narrative_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze manipulation opportunities for all NPCs in a conflict based on their stages.
    
    Args:
        conflict_id: ID of the conflict
        narrative_text: Optional narrative context
        
    Returns:
        Analysis of manipulation opportunities
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Get conflict details
        conflict = await conflict_integration.get_conflict_details(conflict_id)
        if not conflict:
            return {"success": False, "error": "Conflict not found", "opportunities": []}
        
        # Get all stakeholders
        stakeholders = conflict.get('stakeholders', [])
        opportunities = []
        
        for stakeholder in stakeholders:
            if stakeholder.get('entity_type') == 'npc':
                npc_id = stakeholder.get('npc_id')
                if not npc_id:
                    continue
                    
                # Get NPC stage
                stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
                
                # Skip NPCs in Innocent Beginning unless they have very high dominance
                npc_details = await get_npc_details(ctx, npc_id)
                if stage.name == "Innocent Beginning" and npc_details.get('dominance', 0) < 80:
                    continue
                
                # Build context for analysis
                analysis_prompt = f"""
                Analyze manipulation opportunity for {stakeholder.get('npc_name')} in conflict: {conflict['conflict_name']}
                
                NPC Profile:
                - Narrative Stage: {stage.name}
                - Faction: {stakeholder.get('faction_name')}
                - Public Motivation: {stakeholder.get('public_motivation')}
                - Hidden Motivation: {stakeholder.get('hidden_motivation')}
                
                Conflict Context:
                - Phase: {conflict.get('phase')}
                - Player Involvement: {conflict.get('player_involvement', {}).get('involvement_level', 'none')}
                
                Determine:
                1. Best manipulation approach for this stage
                2. Specific leverage points
                3. Likelihood of success
                4. Recommended tactics
                """
                
                # Analyze with agent
                result = await Runner.run(
                    manipulation_analyst,
                    analysis_prompt
                )
                
                # Create opportunity entry
                opportunity = {
                    'npc_id': npc_id,
                    'npc_name': stakeholder.get('npc_name'),
                    'narrative_stage': stage.name,
                    'analysis': result.final_output,
                    'stage_modifier': _get_stage_manipulation_modifier(stage.name),
                    'recommended_approach': _get_stage_appropriate_approach(stage.name)
                }
                
                opportunities.append(opportunity)
        
        # Sort by stage progression (more advanced stages first)
        stage_order = {stage.name: i for i, stage in enumerate(NPC_NARRATIVE_STAGES)}
        opportunities.sort(
            key=lambda x: stage_order.get(x['narrative_stage'], 0), 
            reverse=True
        )
        
        return {
            'conflict_id': conflict_id,
            'conflict_name': conflict['conflict_name'],
            'opportunities': opportunities,
            'total_opportunities': len(opportunities),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error analyzing manipulation opportunities: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "opportunities": []}

# ===== REMAINING TOOLS (Activity, Relationship, Conflict, Resource, Narrative) =====

@function_tool
@track_performance("analyze_activity")
async def analyze_activity(
    ctx: RunContextWrapper[ContextType],
    params: ActivityAnalysisParams
) -> Dict[str, Any]:
    """
    Analyze an activity to determine its resource effects.

    Args:
        params: Activity analysis parameters

    Returns:
        Dict with activity analysis and effects
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.activity_analyzer import ActivityAnalyzer
        analyzer = ActivityAnalyzer(user_id, conversation_id)
        result = await analyzer.analyze_activity(
            params.activity_text, 
            params.setting_context, 
            params.apply_effects
        )

        if hasattr(context, 'add_narrative_memory'):
            effects_description = []
            for resource_type, value in result.get("effects", {}).items():
                if value:
                    direction = "increased" if value > 0 else "decreased"
                    effects_description.append(
                        f"{resource_type} {direction} by {abs(value)}"
                    )
            effects_text = (
                ", ".join(effects_description) if effects_description 
                else "no significant effects"
            )
            memory_content = (
                f"Analyzed activity: {params.activity_text[:100]}... "
                f"with effects: {effects_text}"
            )
            await context.add_narrative_memory(
                memory_content, "activity_analysis", 0.5
            )

        return result
    except Exception as e:
        logger.error(f"Error analyzing activity: {str(e)}", exc_info=True)
        return {
            "activity_type": "unknown", 
            "activity_details": "", 
            "effects": {}, 
            "description": f"Error analyzing activity: {str(e)}", 
            "success": False,
            "error": str(e)
        }

@function_tool
@track_performance("get_filtered_activities")
async def get_filtered_activities(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ActivityFilterParams] = None
) -> List[Dict[str, Any]]:
    """
    Get a list of activities filtered by NPC archetypes, meltdown level, and setting.

    Args:
        params: Activity filter parameters

    Returns:
        List of filtered activities
    """
    if params is None:
        params = ActivityFilterParams()
    
    context = ctx.context
    try:
        from logic.activities_logic import (
            filter_activities_for_npc, 
            build_short_summary
        )

        user_stats = None
        if hasattr(context, 'resource_manager'):
            try:
                resources = await context.resource_manager.get_resources()
                vitals = await context.resource_manager.get_vitals()
                user_stats = {**resources, **vitals}
            except Exception as stats_error:
                logger.warning(f"Could not get user stats: {stats_error}")

        activities = await filter_activities_for_npc(
            npc_archetypes=params.npc_archetypes, 
            meltdown_level=params.meltdown_level, 
            user_stats=user_stats, 
            setting=params.setting
        )
        for activity in activities:
            activity["short_summary"] = build_short_summary(activity)

        return activities
    except Exception as e:
        logger.error(f"Error getting filtered activities: {str(e)}", exc_info=True)
        return []

@function_tool
@track_performance("generate_activity_suggestion")
async def generate_activity_suggestion(
    ctx: RunContextWrapper[ContextType],
    params: ActivitySuggestionParams
) -> Dict[str, Any]:
    """
    Generate a suggested activity for an NPC interaction based on archetypes and intensity.

    Args:
        params: Activity suggestion parameters

    Returns:
        Dict with suggested activity details
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        archetypes = params.archetypes
        if not archetypes:
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(
                        """SELECT archetype FROM NPCStats 
                        WHERE npc_name=$1 AND user_id=$2 AND conversation_id=$3""", 
                        params.npc_name, user_id, conversation_id
                    )
                    if row and row['archetype']:
                        if isinstance(row['archetype'], str):
                            try: 
                                archetype_data = json.loads(row['archetype'])
                                archetypes = (
                                    archetype_data if isinstance(archetype_data, list) 
                                    else (
                                        archetype_data.get("types") 
                                        if isinstance(archetype_data, dict) 
                                        else [row['archetype']]
                                    )
                                )
                            except: 
                                archetypes = [row['archetype']]
                        elif isinstance(row['archetype'], list): 
                            archetypes = row['archetype']
                        elif isinstance(row['archetype'], dict) and "types" in row['archetype']: 
                            archetypes = row['archetype']["types"]
            except Exception as archetype_error: 
                logger.warning(f"Error getting NPC archetypes: {archetype_error}")
        
        if not archetypes: 
            archetypes = ["Dominance", "Femdom"]

        setting = "Default"
        if hasattr(context, 'get_comprehensive_context'):
            try: 
                comprehensive_context = await context.get_comprehensive_context()
                current_location = comprehensive_context.get("current_location")
                setting = current_location or setting
            except Exception as context_error: 
                logger.warning(f"Error getting location from context: {context_error}")

        from logic.activities_logic import (
            filter_activities_for_npc, 
            build_short_summary, 
            get_all_activities as get_activities
        )
        activities = await filter_activities_for_npc(
            npc_archetypes=archetypes, 
            meltdown_level=max(0, params.intensity_level-1), 
            setting=setting
        )
        if not activities: 
            activities = await get_activities()
            activities = random.sample(activities, min(3, len(activities)))

        selected_activity = random.choice(activities) if activities else None
        if not selected_activity: 
            return {
                "npc_name": params.npc_name, 
                "success": False, 
                "error": "No suitable activities found"
            }

        intensity_tiers = selected_activity.get("intensity_tiers", [])
        tier_text = ""
        if intensity_tiers: 
            idx = min(params.intensity_level - 1, len(intensity_tiers) - 1)
            idx = max(0, idx)
            tier_text = intensity_tiers[idx]

        suggestion = {
            "npc_name": params.npc_name, 
            "activity_name": selected_activity.get("name", ""), 
            "purpose": (
                selected_activity.get("purpose", [])[0] 
                if selected_activity.get("purpose") else ""
            ), 
            "intensity_tier": tier_text, 
            "intensity_level": params.intensity_level, 
            "short_summary": build_short_summary(selected_activity), 
            "archetypes_used": archetypes, 
            "setting": setting, 
            "success": True
        }

        if hasattr(context, 'add_narrative_memory'):
            memory_content = (
                f"Generated activity suggestion for {params.npc_name}: "
                f"{suggestion['activity_name']} (Intensity: {params.intensity_level})"
            )
            await context.add_narrative_memory(
                memory_content, "activity_suggestion", 0.5
            )

        return suggestion
    except Exception as e:
        logger.error(f"Error generating activity suggestion: {str(e)}", exc_info=True)
        return {
            "npc_name": params.npc_name, 
            "success": False, 
            "error": str(e)
        }

# ===== RELATIONSHIP TOOLS =====

@function_tool
@track_performance("update_relationship_dimensions")
async def update_relationship_dimensions(
    ctx: RunContextWrapper[ContextType],
    link_id: int,
    dimension_changes: DimensionChanges,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update specific dimensions of a relationship.

    Args:
        link_id: ID of the relationship link
        dimension_changes: Dimension changes model
        reason: Reason for the changes

    Returns:
        Result of the update
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.relationship_integration import RelationshipIntegration
        integration = RelationshipIntegration(user_id, conversation_id)
        
        # Convert model to dict, excluding None values
        changes_dict = dimension_changes.model_dump(exclude_none=True)
        
        result = await integration.update_dimensions(link_id, changes_dict, reason)

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Updated relationship dimensions for link {link_id}: {changes_dict}"
            if reason: 
                memory_content += f" Reason: {reason}"
            await context.add_narrative_memory(
                memory_content, "relationship_update", 0.5
            )

        return result
    except Exception as e:
        logger.error(f"Error updating relationship dimensions: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "link_id": link_id}

# ===== RELATIONSHIP MILESTONE TOOLS =====

@function_tool
async def check_relationship_milestones(
    ctx: RunContextWrapper[ContextType],
    npc_id: int
) -> Dict[str, Any]:
    """
    Check for relationship milestones based on NPC's narrative stage.
    
    Args:
        npc_id: ID of the NPC
        
    Returns:
        Milestone information
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get current stage
        stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
        
        # Get progression stats
        async with get_db_connection_context() as conn:
            progression = await conn.fetchrow("""
                SELECT corruption, dependency, realization_level, stage_entered_at
                FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
        
        if not progression:
            return {"success": False, "error": "No progression found", "npc_id": npc_id}
        
        # Check proximity to next stage
        current_stage_index = next(
            (i for i, s in enumerate(NPC_NARRATIVE_STAGES) if s.name == stage.name), 
            0
        )
        next_stage = (
            NPC_NARRATIVE_STAGES[current_stage_index + 1] 
            if current_stage_index < len(NPC_NARRATIVE_STAGES) - 1 
            else None
        )
        
        milestones = {
            "current_stage": stage.name,
            "days_in_stage": (
                (datetime.now() - progression['stage_entered_at']).days 
                if progression['stage_entered_at'] else 0
            ),
            "progression_stats": {
                "corruption": progression['corruption'],
                "dependency": progression['dependency'],
                "realization": progression['realization_level']
            },
            "next_stage": next_stage.name if next_stage else None,
            "progress_to_next": None,
            "ready_for_advancement": False,
            "suggested_events": []
        }
        
        if next_stage:
            # Calculate progress to next stage
            progress_factors = []
            if next_stage.required_corruption > 0:
                progress_factors.append(
                    progression['corruption'] / next_stage.required_corruption
                )
            if next_stage.required_dependency > 0:
                progress_factors.append(
                    progression['dependency'] / next_stage.required_dependency
                )
            if next_stage.required_realization > 0:
                progress_factors.append(
                    progression['realization_level'] / next_stage.required_realization
                )
            
            milestones['progress_to_next'] = min(progress_factors) if progress_factors else 0
            milestones['ready_for_advancement'] = milestones['progress_to_next'] >= 0.9
            
            # Suggest events based on what's needed
            if progression['corruption'] < next_stage.required_corruption:
                milestones['suggested_events'].append("Activities that increase corruption")
            if progression['dependency'] < next_stage.required_dependency:
                milestones['suggested_events'].append("Interactions that build dependency")
            if progression['realization_level'] < next_stage.required_realization:
                milestones['suggested_events'].append("Moments of clarity or revelation")
        
        return milestones
        
    except Exception as e:
        logger.error(f"Error checking relationship milestones: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "npc_id": npc_id}

# ===== NARRATIVE MOMENT TOOLS =====

@function_tool
async def generate_stage_contrast_moment(
    ctx: RunContextWrapper[ContextType],
    npc_ids: List[int]
) -> Dict[str, Any]:
    """
    Generate a narrative moment highlighting the contrast between NPCs at different stages.
    
    Args:
        npc_ids: List of NPC IDs to contrast
        
    Returns:
        Generated narrative moment
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Get stages for each NPC
        npc_stages = []
        for npc_id in npc_ids[:3]:  # Limit to 3
            npc_details = await get_npc_details(ctx, npc_id)
            if npc_details:
                stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
                npc_stages.append({
                    'npc_id': npc_id,
                    'npc_name': npc_details['npc_name'],
                    'stage': stage.name,
                    'stage_index': next(
                        (i for i, s in enumerate(NPC_NARRATIVE_STAGES) if s.name == stage.name), 
                        0
                    )
                })
        
        if len(npc_stages) < 2:
            return {"success": False, "error": "Need at least 2 NPCs for contrast"}
        
        # Sort by stage progression
        npc_stages.sort(key=lambda x: x['stage_index'])
        
        # Generate contrast narrative
        most_advanced = npc_stages[-1]
        least_advanced = npc_stages[0]
        
        if most_advanced['stage_index'] - least_advanced['stage_index'] < 2:
            return {
                "success": False, 
                "error": "NPCs too similar in progression for meaningful contrast"
            }
        
        scene_text = f"""
        You notice {most_advanced['npc_name']} and {least_advanced['npc_name']} exchanging glances across the room. 
        
        {least_advanced['npc_name']}'s expression is warm, friendly, seemingly innocent - the same gentle demeanor 
        you've always known. But {most_advanced['npc_name']}'s smile carries a different weight entirely. There's 
        knowledge in it, a satisfaction that makes you shift uncomfortably.
        
        "{least_advanced['npc_name']} is lovely, isn't she?" {most_advanced['npc_name']} murmurs, close enough 
        that only you can hear. "So... genuine. So careful about maintaining appearances. I remember being like that."
        
        The implication hangs in the air between you. Your eyes dart between them, seeing the trajectory laid bare - 
        what {least_advanced['npc_name']} is, what {most_advanced['npc_name']} has become, and what that means 
        for you.
        """
        
        return {
            "type": "narrative_moment",
            "subtype": "stage_contrast",
            "scene_text": scene_text,
            "featured_npcs": npc_stages,
            "contrast_level": most_advanced['stage_index'] - least_advanced['stage_index'],
            "player_realization": (
                "The pattern is clear - they're all on the same path, "
                "just at different points."
            ),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error generating stage contrast moment: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

# ===== CONFLICT TOOLS =====

@function_tool
async def analyze_conflict_potential(
    ctx: RunContextWrapper[ContextType], 
    narrative_text: str
) -> Dict[str, Any]:
    """Analyze narrative text for conflict potential."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        conflict_keywords = [
            "argument", "disagreement", "tension", "rivalry", "competition",
            "dispute", "feud", "clash", "confrontation", "battle", "fight",
            "war", "conflict", "power struggle", "contest", "strife"
        ]
        
        matched_keywords = []
        for keyword in conflict_keywords:
            if keyword in narrative_text.lower():
                matched_keywords.append(keyword)
        
        # Calculate conflict intensity based on keyword matches
        conflict_intensity = min(10, len(matched_keywords) * 2)
        
        # Check for NPC mentions
        npcs = await get_available_npcs(ctx)
        
        mentioned_npcs = []
        for npc in npcs:
            if npc["npc_name"] in narrative_text:
                mentioned_npcs.append({
                    "npc_id": npc["npc_id"],
                    "npc_name": npc["npc_name"],
                    "dominance": npc["dominance"],
                    "faction_affiliations": npc.get("faction_affiliations", [])
                })
        
        # Look for faction mentions
        mentioned_factions = []
        for npc in mentioned_npcs:
            for affiliation in npc.get("faction_affiliations", []):
                faction_name = affiliation.get("faction_name")
                if faction_name and faction_name in narrative_text:
                    mentioned_factions.append({
                        "faction_id": affiliation.get("faction_id"),
                        "faction_name": faction_name
                    })
        
        # Check for relationship indicators between NPCs
        npc_relationships = []
        for i, npc1 in enumerate(mentioned_npcs):
            for npc2 in mentioned_npcs[i+1:]:
                # Look for both NPCs in the same sentence
                sentences = narrative_text.split('.')
                for sentence in sentences:
                    if npc1["npc_name"] in sentence and npc2["npc_name"] in sentence:
                        # Check for relationship indicators
                        relationship_type = "unknown"
                        for word in ["allies", "friends", "partners", "together"]:
                            if word in sentence.lower():
                                relationship_type = "alliance"
                                break
                        for word in ["enemies", "rivals", "hate", "against"]:
                            if word in sentence.lower():
                                relationship_type = "rivalry"
                                break
                        
                        npc_relationships.append({
                            "npc1_id": npc1["npc_id"],
                            "npc1_name": npc1["npc_name"],
                            "npc2_id": npc2["npc_id"],
                            "npc2_name": npc2["npc_name"],
                            "relationship_type": relationship_type,
                            "sentence": sentence.strip()
                        })
        
        # Determine appropriate conflict type based on analysis
        conflict_type = (
            "major" if conflict_intensity >= 8 
            else ("standard" if conflict_intensity >= 5 else "minor")
        )
        internal_faction_conflict = None

        return {
            "conflict_intensity": conflict_intensity, 
            "matched_keywords": matched_keywords, 
            "mentioned_npcs": mentioned_npcs, 
            "mentioned_factions": mentioned_factions, 
            "npc_relationships": npc_relationships, 
            "recommended_conflict_type": conflict_type, 
            "potential_internal_faction_conflict": internal_faction_conflict, 
            "has_conflict_potential": conflict_intensity >= 4
        }
    except Exception as e:
        logger.error(f"Error analyzing conflict potential: {e}")
        return {
            "conflict_intensity": 0, 
            "matched_keywords": [], 
            "mentioned_npcs": [], 
            "mentioned_factions": [], 
            "npc_relationships": [], 
            "recommended_conflict_type": "minor", 
            "potential_internal_faction_conflict": None, 
            "has_conflict_potential": False, 
            "success": False,
            "error": str(e)
        }

@function_tool
async def generate_conflict_from_analysis(
    ctx: RunContextWrapper[ContextType],
    analysis: ConflictAnalysis
) -> Dict[str, Any]:
    """Generate a conflict based on analysis provided by analyze_conflict_potential."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        if not analysis.has_conflict_potential:
            return {
                "generated": False, 
                "reason": "Insufficient conflict potential", 
                "analysis": analysis.model_dump()
            }

        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        conflict = await conflict_integration.generate_new_conflict(
            analysis.recommended_conflict_type
        )

        internal_faction_conflict = None
        if (analysis.potential_internal_faction_conflict and conflict 
            and conflict.get("conflict_id")):
            internal_data = analysis.potential_internal_faction_conflict
            try:
                internal_faction_conflict = await conflict_integration.initiate_faction_power_struggle(
                    conflict["conflict_id"],
                    internal_data.faction_id,
                    internal_data.challenger_npc_id,
                    internal_data.target_npc_id,
                    internal_data.prize,
                    internal_data.approach,
                    False
                )
            except Exception as e:
                logger.error(f"Error generating internal faction conflict: {e}")

        return {
            "generated": True,
            "conflict": conflict,
            "internal_faction_conflict": internal_faction_conflict
        }
    except Exception as e:
        logger.error(f"Error generating conflict from analysis: {e}", exc_info=True)
        return {
            "generated": False, 
            "reason": f"Error: {str(e)}", 
            "analysis": analysis.model_dump()
        }

@function_tool
async def analyze_npc_manipulation_potential(
    ctx: RunContextWrapper[ContextType], 
    conflict_id: int, 
    npc_id: int
) -> Dict[str, Any]:
    """Analyze an NPC's potential to manipulate the player within a conflict."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        potential = await conflict_integration.analyze_manipulation_potential(npc_id)
        conflict = await conflict_integration.get_conflict_details(conflict_id)
        involvement = conflict.get("player_involvement") if conflict else None

        makes_sense = True
        reason = "NPC could manipulate player"
        if (involvement and involvement.get("involvement_level") != "none" 
            and involvement.get("is_manipulated")):
            manipulator_id = involvement.get("manipulated_by", {}).get("npc_id")
            if manipulator_id == npc_id: 
                makes_sense = False
                reason = "NPC is already manipulating player"

        goal = {"faction": "neutral", "involvement_level": "observing"}
        if potential.get("femdom_compatible"): 
            goal["involvement_level"] = "participating"

        return {
            "npc_id": npc_id, 
            "conflict_id": conflict_id, 
            "manipulation_potential": potential, 
            "makes_sense": makes_sense, 
            "reason": reason, 
            "recommended_goal": goal, 
            "current_involvement": involvement
        }
    except Exception as e:
        logger.error(f"Error analyzing manipulation potential: {e}")
        return {
            "npc_id": npc_id, 
            "conflict_id": conflict_id, 
            "manipulation_potential": {}, 
            "makes_sense": False, 
            "reason": f"Error: {str(e)}", 
            "recommended_goal": {}, 
            "current_involvement": None
        }

@function_tool
async def generate_manipulation_attempt(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    npc_id: int,
    manipulation_type: str,
    goal: ManipulationGoal
) -> Dict[str, Any]:
    """Generate a manipulation attempt by an NPC in a conflict."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        # Convert model to dict
        goal_dict = goal.model_dump()

        suggestion = await conflict_integration.suggest_manipulation_content(
            npc_id, conflict_id, manipulation_type, goal_dict
        )
        attempt = await conflict_integration.create_manipulation_attempt(
            conflict_id,
            npc_id,
            manipulation_type,
            suggestion["content"],
            goal_dict,
            suggestion["leverage_used"],
            suggestion["intimacy_level"]
        )

        npc_name = suggestion.get("npc_name", "Unknown NPC")
        content = suggestion.get("content", "No content generated.")

        return {
            "generated": True,
            "attempt": attempt,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "manipulation_type": manipulation_type,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error generating manipulation attempt: {e}", exc_info=True)
        return {
            "generated": False,
            "reason": f"Error: {str(e)}",
            "npc_id": npc_id,
            "manipulation_type": manipulation_type
        }

@function_tool
async def set_player_involvement(
    ctx: RunContextWrapper[ContextType],
    params: PlayerInvolvementParams
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.

    Args:
        params: Player involvement parameters

    Returns:
        Updated conflict information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Use the conflict integration directly instead of expecting it on context
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    from logic.conflict_system.conflict_tools import (
        get_conflict_details, update_player_involvement as update_involvement
    )
    
    try:
        # Check if we have a resource manager
        if not hasattr(context, 'resource_manager'):
            logger.error("Context missing resource_manager")
            return {
                "conflict_id": params.conflict_id, 
                "success": False,
                "error": "Internal context setup error"
            }
            
        resource_manager = context.resource_manager

        # Check resources
        resource_check = await resource_manager.check_resources(
            params.money_committed, 
            params.supplies_committed, 
            params.influence_committed
        )
        if not resource_check.get('has_resources', False):
            resource_check['success'] = False
            resource_check['error'] = "Insufficient resources to commit"
            return resource_check

        # Get conflict info using the tools
        conflict_info = await get_conflict_details(ctx, params.conflict_id)
        
        # Update player involvement using the tools
        involvement_data = {
            "involvement_level": params.involvement_level,
            "faction": params.faction,
            "resources_committed": {
                "money": params.money_committed,
                "supplies": params.supplies_committed,
                "influence": params.influence_committed
            },
            "actions_taken": [params.action] if params.action else []
        }
        
        result = await update_involvement(ctx, params.conflict_id, involvement_data)

        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if params.money_committed > 0: 
                resources_text.append(f"{params.money_committed} money")
            if params.supplies_committed > 0: 
                resources_text.append(f"{params.supplies_committed} supplies")
            if params.influence_committed > 0: 
                resources_text.append(f"{params.influence_committed} influence")
            resources_committed = (
                ", ".join(resources_text) if resources_text else "no resources"
            )

            conflict_name = (
                conflict_info.get('conflict_name', f'ID: {params.conflict_id}') 
                if conflict_info else f'ID: {params.conflict_id}'
            )
            memory_content = (
                f"Player set involvement in conflict {conflict_name} "
                f"to {params.involvement_level}, supporting {params.faction} faction "
                f"with {resources_committed}."
            )
            if params.action: 
                memory_content += f" Action taken: {params.action}"
            await context.add_narrative_memory(
                memory_content, "conflict_involvement", 0.7
            )

        if isinstance(result, dict):
            result["success"] = True
        else:
            result = {
                "conflict_id": params.conflict_id,
                "involvement_level": params.involvement_level,
                "faction": params.faction,
                "resources_committed": {
                    "money": params.money_committed,
                    "supplies": params.supplies_committed,
                    "influence": params.influence_committed
                },
                "action": params.action,
                "success": True,
                "raw_result": result
            }

        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {str(e)}", exc_info=True)
        return {
            "conflict_id": params.conflict_id, 
            "success": False,
            "error": str(e)
        }

@function_tool
async def track_conflict_story_beat(
    ctx: RunContextWrapper[ContextType],
    params: StoryBeatParams
) -> Dict[str, Any]:
    """Track a story beat for a resolution path, advancing progress."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        result = await conflict_integration.track_story_beat(
            params.conflict_id,
            params.path_id,
            params.beat_description,
            params.involved_npcs,
            params.progress_value
        )

        if isinstance(result, dict):
            return {"tracked": True, "result": result}
        else:
            return {"tracked": True, "result": {"raw_output": result}}

    except Exception as e:
        logger.error(f"Error tracking story beat: {e}", exc_info=True)
        return {"tracked": False, "reason": f"Error: {str(e)}"}

@function_tool
async def suggest_potential_manipulation(
    ctx: RunContextWrapper[ContextType], 
    narrative_text: str
) -> Dict[str, Any]:
    """Analyze narrative text and suggest potential NPC manipulation opportunities."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        active_conflicts = await conflict_integration.get_active_conflicts()
        if not active_conflicts: 
            return {"opportunities": [], "reason": "No active conflicts"}

        # Only get introduced female NPCs with high dominance for manipulation
        npcs = await get_available_npcs(
            ctx, 
            include_unintroduced=False, 
            min_dominance=60, 
            gender_filter="female"
        )
        mentioned_npcs = [
            npc for npc in npcs if npc["npc_name"] in narrative_text
        ]
        if not mentioned_npcs: 
            return {"opportunities": [], "reason": "No NPCs mentioned in narrative"}

        opportunities = []
        for conflict in active_conflicts:
            conflict_id = conflict["conflict_id"]
            for npc in mentioned_npcs:
                if npc.get("sex", "female") == "female" and npc.get("dominance", 0) > 60:
                    is_stakeholder = any(
                        s["npc_id"] == npc["npc_id"] 
                        for s in conflict.get("stakeholders", [])
                    )
                    if is_stakeholder:
                        potential = await conflict_integration.analyze_manipulation_potential(
                            npc["npc_id"]
                        )
                        if potential.get("overall_potential", 0) > 60:
                            opportunities.append({
                                "conflict_id": conflict_id, 
                                "conflict_name": conflict["conflict_name"], 
                                "npc_id": npc["npc_id"], 
                                "npc_name": npc["npc_name"], 
                                "dominance": npc["dominance"], 
                                "manipulation_type": potential.get("most_effective_type"), 
                                "potential": potential.get("overall_potential")
                            })

        return {"opportunities": opportunities, "total_opportunities": len(opportunities)}
    except Exception as e:
        logger.error(f"Error suggesting potential manipulation: {e}")
        return {"opportunities": [], "reason": f"Error: {str(e)}"}

# ===== RESOURCE MANAGEMENT TOOLS =====

@function_tool
@track_performance("check_resources")
async def check_resources(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ResourceCheck] = None
) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.

    Args:
        params: Resource check parameters

    Returns:
        Dictionary with resource check results.
    """
    if params is None:
        params = ResourceCheck()
    
    context = ctx.context
    if not hasattr(context, 'resource_manager'):
        logger.error("Context missing resource_manager")
        return {
            "has_resources": False, 
            "success": False,
            "error": "Internal context setup error", 
            "current": {}
        }
    
    resource_manager = context.resource_manager

    try:
        result = await resource_manager.check_resources(
            params.money, params.supplies, params.influence
        )

        current_res = result.get('current', {})
        if current_res.get('money') is not None:
            try:
                formatted_money = await resource_manager.get_formatted_money(
                    current_res['money']
                )
                current_res['formatted_money'] = formatted_money
                result['current'] = current_res
            except Exception as format_err:
                logger.warning(f"Could not format money: {format_err}")

        if 'has_resources' not in result:
            result['has_resources'] = False
        if 'current' not in result:
            result['current'] = {}

        return result
    except Exception as e:
        logger.error(f"Error checking resources: {str(e)}", exc_info=True)
        return {
            "has_resources": False, 
            "success": False,
            "error": str(e), 
            "current": {}
        }

@function_tool
@track_performance("commit_resources_to_conflict")
async def commit_resources_to_conflict(
    ctx: RunContextWrapper[ContextType],
    params: ResourceCommitment
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.

    Args:
        params: Resource commitment parameters

    Returns:
        Result of committing resources or error dictionary.
    """
    context = ctx.context
    if not hasattr(context, 'resource_manager'):
        logger.error("Context missing resource_manager")
        return {"success": False, "error": "Internal context setup error"}
    
    resource_manager = context.resource_manager

    try:
        conflict_info = None
        if hasattr(context, 'conflict_manager') and context.conflict_manager:
            try:
                conflict_info = await context.conflict_manager.get_conflict(
                    params.conflict_id
                )
            except Exception as conflict_error:
                logger.warning(f"Could not get conflict info: {conflict_error}")
        else:
            logger.warning("Context missing conflict_manager")

        result = await resource_manager.commit_resources_to_conflict(
            params.conflict_id, params.money, params.supplies, params.influence
        )

        if params.money > 0 and result.get('success', False) and result.get('money_result'):
            money_result = result['money_result']
            if 'old_value' in money_result and 'new_value' in money_result:
                try:
                    old_formatted = await resource_manager.get_formatted_money(
                        money_result['old_value']
                    )
                    new_formatted = await resource_manager.get_formatted_money(
                        money_result['new_value']
                    )
                    change_val = money_result.get('change')
                    formatted_change = (
                        await resource_manager.get_formatted_money(change_val) 
                        if change_val is not None else None
                    )

                    money_result['formatted_old_value'] = old_formatted
                    money_result['formatted_new_value'] = new_formatted
                    if formatted_change is not None:
                        money_result['formatted_change'] = formatted_change
                    result['money_result'] = money_result
                except Exception as format_err:
                    logger.warning(f"Could not format money: {format_err}")

        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if params.money > 0: 
                resources_text.append(f"{params.money} money")
            if params.supplies > 0: 
                resources_text.append(f"{params.supplies} supplies")
            if params.influence > 0: 
                resources_text.append(f"{params.influence} influence")
            resources_committed = (
                ", ".join(resources_text) if resources_text else "No resources"
            )

            conflict_name = (
                conflict_info.get('conflict_name', f"ID: {params.conflict_id}") 
                if conflict_info else f"ID: {params.conflict_id}"
            )
            memory_content = f"Committed {resources_committed} to conflict {conflict_name}"
            await context.add_narrative_memory(
                memory_content, "resource_commitment", 0.6
            )

        if 'success' not in result:
            result['success'] = True

        return result
    except Exception as e:
        logger.error(f"Error committing resources: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
@track_performance("get_player_resources")
async def get_player_resources(
    ctx: RunContextWrapper[ContextType]
) -> Dict[str, Any]:
    """
    Get the current player resources and vitals.

    Returns:
        Current resource status
    """
    context = ctx.context
    resource_manager = context.resource_manager

    try:
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        formatted_money = await resource_manager.get_formatted_money()
        updated_at = resources.get('updated_at', datetime.now())
        updated_at_iso = (
            updated_at.isoformat() if isinstance(updated_at, datetime) 
            else str(updated_at)
        )

        return {
            "money": resources.get('money', 0), 
            "supplies": resources.get('supplies', 0), 
            "influence": resources.get('influence', 0), 
            "energy": vitals.get('energy', 0), 
            "hunger": vitals.get('hunger', 0), 
            "formatted_money": formatted_money, 
            "updated_at": updated_at_iso
        }
    except Exception as e:
        logger.error(f"Error getting player resources: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e), 
            "money": 0, 
            "supplies": 0, 
            "influence": 0, 
            "energy": 0, 
            "hunger": 0, 
            "formatted_money": "0"
        }

@function_tool
@track_performance("apply_activity_effects")
async def apply_activity_effects(
    ctx: RunContextWrapper[ContextType], 
    activity_text: str
) -> Dict[str, Any]:
    """
    Analyze and apply the effects of an activity to player resources.

    Args:
        activity_text: Description of the activity

    Returns:
        Results of applying activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer

    try:
        result = await activity_analyzer.analyze_activity(
            activity_text, apply_effects=True
        )

        if 'effects' in result and 'money' in result['effects']:
            resource_manager = context.resource_manager
            resources = await resource_manager.get_resources()
            result['formatted_money'] = await resource_manager.get_formatted_money(
                resources.get('money', 0)
            )

        if hasattr(context, 'add_narrative_memory'):
            effects = result.get('effects', {})
            effects_description = [
                f"{res} {('increased' if val > 0 else 'decreased')} by {abs(val)}" 
                for res, val in effects.items() if val
            ]
            effects_text = (
                ", ".join(effects_description) if effects_description 
                else "no significant effects"
            )
            memory_content = (
                f"Applied activity effects for: {activity_text[:100]}... "
                f"with {effects_text}"
            )
            await context.add_narrative_memory(
                memory_content, "activity_application", 0.5
            )

        return result
    except Exception as e:
        logger.error(f"Error applying activity effects: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e), 
            "activity_type": "unknown", 
            "activity_details": "", 
            "effects": {}
        }

@function_tool
@track_performance("get_resource_history")
async def get_resource_history(
    ctx: RunContextWrapper[ContextType],
    resource_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get the history of resource changes.

    Args:
        resource_type: Optional filter for specific resource type.
        limit: Maximum number of history entries to return. Defaults to 10.

    Returns:
        List of resource change history entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    resource_manager = getattr(context, 'resource_manager', None)

    # Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = """
                SELECT resource_type, old_value, new_value, amount_changed, 
                       source, description, timestamp 
                FROM ResourceHistoryLog 
                WHERE user_id=$1 AND conversation_id=$2
            """
            params = [user_id, conversation_id]

            if resource_type:
                base_query += " AND resource_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([resource_type, actual_limit])
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit)

            rows = await conn.fetch(base_query, *params)
            history = []

            for row in rows:
                formatted_old, formatted_new, formatted_change = None, None, None
                # Only format money if resource_manager is available
                if row['resource_type'] == "money" and resource_manager:
                    try:
                        formatted_old = await resource_manager.get_formatted_money(
                            row['old_value']
                        )
                        formatted_new = await resource_manager.get_formatted_money(
                            row['new_value']
                        )
                        formatted_change = await resource_manager.get_formatted_money(
                            row['amount_changed']
                        )
                    except Exception as format_err:
                        logger.warning(
                            f"Could not format money in get_resource_history: {format_err}"
                        )

                timestamp = row['timestamp']
                timestamp_iso = (
                    timestamp.isoformat() if isinstance(timestamp, datetime) 
                    else str(timestamp)
                )

                history.append({
                    "resource_type": row['resource_type'],
                    "old_value": row['old_value'],
                    "new_value": row['new_value'],
                    "amount_changed": row['amount_changed'],
                    "formatted_old_value": formatted_old,
                    "formatted_new_value": formatted_new,
                    "formatted_change": formatted_change,
                    "source": row['source'],
                    "description": row['description'],
                    "timestamp": timestamp_iso
                })
            return history
        except Exception as e:
            logger.error(f"Error getting resource history: {str(e)}", exc_info=True)
            return []

# ===== NARRATIVE ELEMENT TOOLS =====

@function_tool
@track_performance("generate_personal_revelation")
async def generate_personal_revelation(
    ctx: RunContextWrapper[ContextType], 
    npc_name: str, 
    revelation_type: str
) -> Dict[str, Any]:
    """
    Generate a personal revelation for the player about their relationship with an NPC.

    Args:
        npc_name: Name of the NPC involved in the revelation
        revelation_type: Type of revelation

    Returns:
        A personal revelation
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Define revelation templates based on type
    templates = {
        "dependency": [
            f"I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            f"I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            f"The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ],
        "obedience": [
            f"I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            f"Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            f"{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ],
        "corruption": [
            f"I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            f"Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            f"Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ],
        "willpower": [
            f"I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            f"I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            f"I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ],
        "confidence": [
            f"I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            f"I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            f"When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }

    try:
        revelation_templates = templates.get(
            revelation_type.lower(), 
            templates["dependency"]
        )
        inner_monologue = random.choice(revelation_templates)

        canon_ctx = SimpleContext(user_id, conversation_id)
        
        async with get_db_connection_context() as conn:
            try:
                # Use canon to create journal entry
                journal_id = await canon.create_journal_entry(
                    ctx=canon_ctx,
                    conn=conn,
                    entry_type='personal_revelation',
                    entry_text=inner_monologue,
                    revelation_types=revelation_type,
                    narrative_moment=None,
                    fantasy_flag=False,
                    intensity_level=0,
                    importance=0.8,
                    tags=[
                        revelation_type, 
                        "revelation", 
                        npc_name.lower().replace(" ", "_")
                    ]
                )

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Personal revelation about {npc_name}: {inner_monologue}", 
                        "personal_revelation", 
                        0.8, 
                        tags=[
                            revelation_type, 
                            "revelation", 
                            npc_name.lower().replace(" ", "_")
                        ]
                    )
                    
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    await context.narrative_manager.add_revelation(
                        content=inner_monologue, 
                        revelation_type=revelation_type, 
                        importance=0.8, 
                        tags=[revelation_type, "revelation"]
                    )

                return {
                    "type": "personal_revelation", 
                    "name": f"{revelation_type.capitalize()} Awareness", 
                    "inner_monologue": inner_monologue, 
                    "journal_id": journal_id, 
                    "success": True
                }
            except Exception as db_error:
                logger.error(f"Database error recording personal revelation: {db_error}")
                raise
    except Exception as e:
        logger.error(f"Error generating personal revelation: {str(e)}", exc_info=True)
        return {
            "type": "personal_revelation", 
            "name": f"{revelation_type.capitalize()} Awareness", 
            "inner_monologue": f"Error generating revelation: {str(e)}", 
            "success": False
        }

@function_tool
@track_performance("generate_dream_sequence")
async def generate_dream_sequence(
    ctx: RunContextWrapper[ContextType], 
    npc_names: List[str]
) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.

    Args:
        npc_names: List of NPC names to include in the dream

    Returns:
        A dream sequence
    """
    while len(npc_names) < 3: 
        npc_names.append(f"Unknown Figure {len(npc_names) + 1}")
    npc1, npc2, npc3 = npc_names[:3]
    
    # Dream templates
    dream_templates = [
        f"""You're sitting in a chair as {npc1} circles you slowly. "Show me your hands," she says. You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "Do you see who's holding them?" she asks. You look up, but the ceiling is mirrored, showing only your own face looking back down at you, smiling with an expression that isn't yours.""",
        
        f"""You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, doorways leading to impossible spaces. Your phone rings. It's {npc1}. "Where are you?" you ask desperately. "I'm right here," she says, her voice coming both from the phone and from behind you. "I've always been right here. You're the one who's lost." """,
        
        f"""You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "I don't understand why you're struggling," she says, not moving yet somehow keeping pace beside you. "You stopped walking on your own long ago." You look down to find your legs have merged with the floor entirely, indistinguishable from the material beneath.""",
        
        f"""You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, saying words you didn't intend. The audience nods approvingly. "Much better," whispers {npc2} from beside you. "Your ideas were never as good as hers anyway." """,
        
        f"""You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, all versions of her turn to look at you simultaneously. "Which one is real?" they ask in unison. "The one you needed, or the one who needed you?" """,
        
        f"""You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, looking up at you and conversing, their voices perfectly clear despite the water. "They still think they're above it all," says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."""
    ]

    try:
        dream_text = random.choice(dream_templates)
        context = ctx.context
        user_id = context.user_id
        conversation_id = context.conversation_id

        canon_ctx = SimpleContext(user_id, conversation_id)

        async with get_db_connection_context() as conn:
            try:
                # Use canon to create journal entry
                journal_id = await canon.create_journal_entry(
                    ctx=canon_ctx,
                    conn=conn,
                    entry_type='dream_sequence',
                    entry_text=dream_text,
                    revelation_types=None,
                    narrative_moment=True,
                    fantasy_flag=True,
                    intensity_level=0,
                    importance=0.7,
                    tags=["dream", "symbolic"] + [
                        npc.lower().replace(" ", "_") for npc in npc_names[:3]
                    ]
                )

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Dream sequence: {dream_text}", 
                        "dream_sequence", 
                        0.7, 
                        tags=["dream", "symbolic"] + [
                            npc.lower().replace(" ", "_") for npc in npc_names[:3]
                        ]
                    )
                    
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    await context.narrative_manager.add_dream_sequence(
                        content=dream_text, 
                        symbols=[npc1, npc2, npc3, "control", "manipulation"], 
                        importance=0.7, 
                        tags=["dream", "symbolic"]
                    )

                return {
                    "type": "dream_sequence", 
                    "text": dream_text, 
                    "journal_id": journal_id, 
                    "success": True
                }
            except Exception as db_error:
                logger.error(f"Database error recording dream sequence: {db_error}")
                raise
    except Exception as e:
        logger.error(f"Error generating dream sequence: {str(e)}", exc_info=True)
        return {
            "type": "dream_sequence", 
            "text": f"Error generating dream: {str(e)}", 
            "success": False
        }

@function_tool
@track_performance("check_relationship_events")
async def check_relationship_events(
    ctx: RunContextWrapper[ContextType]
) -> Dict[str, Any]:
    """
    Check for relationship events like crossroads or rituals.

    Returns:
        Dictionary with any triggered relationship events
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        crossroads = await check_for_crossroads_tool(user_id, conversation_id)
        ritual = await check_for_ritual_tool(user_id, conversation_id)

        if (crossroads or ritual) and hasattr(context, 'add_narrative_memory'):
            event_type = "crossroads" if crossroads else "ritual"
            npc_name = "Unknown"
            if crossroads: 
                npc_name = crossroads.get("npc_name", "Unknown")
            elif ritual: 
                npc_name = ritual.get("npc_name", "Unknown")
            memory_content = f"Relationship {event_type} detected with {npc_name}"
            await context.add_narrative_memory(
                memory_content, 
                f"relationship_{event_type}", 
                0.8, 
                tags=[event_type, "relationship", npc_name.lower().replace(" ", "_")]
            )

        return {
            "crossroads": crossroads, 
            "ritual": ritual, 
            "has_events": crossroads is not None or ritual is not None
        }
    except Exception as e:
        logger.error(f"Error checking relationship events: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e), 
            "crossroads": None, 
            "ritual": None, 
            "has_events": False
        }

@function_tool
async def get_npc_stage(
    ctx: RunContextWrapper[ContextType],
    npc_id: int
) -> Dict[str, Any]:
    """
    Get the narrative stage for a specific NPC relationship.
    
    Args:
        npc_id: ID of the NPC
        
    Returns:
        Dictionary with stage information
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
        
        return {
            "npc_id": npc_id,
            "stage_name": stage.name,
            "stage_description": stage.description,
            "required_corruption": stage.required_corruption,
            "required_dependency": stage.required_dependency,
            "required_realization": stage.required_realization
        }
        
    except Exception as e:
        logger.error(f"Error getting NPC stage: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "npc_id": npc_id
        }

@function_tool
@track_performance("apply_crossroads_choice")
async def apply_crossroads_choice(
    ctx: RunContextWrapper[ContextType],
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Apply a chosen effect from a triggered relationship crossroads.

    Args:
        link_id: ID of the social link
        crossroads_name: Name of the crossroads event
        choice_index: Index of the chosen option

    Returns:
        Result of applying the choice
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        result = await apply_crossroads_choice_tool(
            user_id, conversation_id, link_id, crossroads_name, choice_index
        )

        if hasattr(context, 'add_narrative_memory'):
            npc_name = "Unknown"
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(
                        """SELECT entity2_id FROM SocialLinks 
                        WHERE link_id = $1 AND entity2_type = 'npc'""", 
                        link_id
                    )
                    if row: 
                        npc_id = row['entity2_id']
                        npc_row = await conn.fetchrow(
                            "SELECT npc_name FROM NPCStats WHERE npc_id = $1", 
                            npc_id
                        )
                        npc_name = npc_row['npc_name'] if npc_row else npc_name
            except Exception as db_error: 
                logger.warning(f"Could not get NPC name for memory: {db_error}")

            memory_content = (
                f"Applied crossroads choice {choice_index} for '{crossroads_name}' "
                f"with {npc_name}"
            )
            await context.add_narrative_memory(
                memory_content, 
                "crossroads_choice", 
                0.8, 
                tags=["crossroads", "relationship", npc_name.lower().replace(" ", "_")]
            )
            if hasattr(context, 'narrative_manager') and context.narrative_manager:
                await context.narrative_manager.add_interaction(
                    content=memory_content, 
                    npc_name=npc_name, 
                    importance=0.8, 
                    tags=["crossroads", "relationship_choice"]
                )

        return result
    except Exception as e:
        logger.error(f"Error applying crossroads choice: {str(e)}", exc_info=True)
        return {
            "link_id": link_id, 
            "crossroads_name": crossroads_name, 
            "choice_index": choice_index, 
            "success": False, 
            "error": str(e)
        }

@function_tool
@track_performance("check_npc_relationship")
async def check_npc_relationship(
    ctx: RunContextWrapper[ContextType], 
    npc_id: int
) -> Dict[str, Any]:
    """
    Get the relationship between the player and an NPC.

    Args:
        npc_id: ID of the NPC

    Returns:
        Relationship summary
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        relationship = await get_relationship_summary_tool(
            user_id, conversation_id, "player", user_id, "npc", npc_id
        )
        if not relationship:
            try:
                from logic.social_links_agentic import create_social_link
                link_id = await create_social_link(
                    user_id, conversation_id, "player", user_id, "npc", npc_id
                )
                relationship = await get_relationship_summary_tool(
                    user_id, conversation_id, "player", user_id, "npc", npc_id
                )
            except Exception as link_error:
                logger.error(f"Error creating social link: {link_error}")
                return {
                    "success": False,
                    "error": f"Failed to create relationship: {str(link_error)}", 
                    "npc_id": npc_id
                }

        return relationship or {
            "success": False,
            "error": "Could not get or create relationship", 
            "npc_id": npc_id
        }
    except Exception as e:
        logger.error(f"Error checking NPC relationship: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "npc_id": npc_id}

@function_tool
@track_performance("add_moment_of_clarity")
async def add_moment_of_clarity(
    ctx: RunContextWrapper[ContextType], 
    realization_text: str
) -> Dict[str, Any]:
    """
    Add a moment of clarity where the player briefly becomes aware of their situation.

    Args:
        realization_text: The specific realization the player has

    Returns:
        The created moment of clarity
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.narrative_progression import add_moment_of_clarity as add_clarity
        result = await add_clarity(user_id, conversation_id, realization_text)

        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(
                f"Moment of clarity: {realization_text}", 
                "moment_of_clarity", 
                0.9, 
                tags=["clarity", "realization", "awareness"]
            )
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            await context.narrative_manager.add_revelation(
                content=realization_text, 
                revelation_type="clarity", 
                importance=0.9, 
                tags=["clarity", "realization"]
            )

        return {"type": "moment_of_clarity", "content": result, "success": True}
    except Exception as e:
        logger.error(f"Error adding moment of clarity: {str(e)}", exc_info=True)
        return {
            "type": "moment_of_clarity", 
            "content": None, 
            "success": False, 
            "error": str(e)
        }

@function_tool
@track_performance("get_player_journal_entries")
async def get_player_journal_entries(
    ctx: RunContextWrapper[ContextType],
    entry_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get entries from the player's journal.

    Args:
        entry_type: Optional filter for entry type.
        limit: Maximum number of entries to return. Defaults to 10.

    Returns:
        List of journal entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = """
                SELECT id, entry_type, entry_text, revelation_types, 
                       narrative_moment, fantasy_flag, intensity_level, timestamp 
                FROM PlayerJournal 
                WHERE user_id=$1 AND conversation_id=$2
            """
            params = [user_id, conversation_id]

            if entry_type:
                base_query += " AND entry_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([entry_type, actual_limit])
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit)

            rows = await conn.fetch(base_query, *params)
            entries = []
            for row in rows:
                timestamp = row['timestamp']
                timestamp_iso = (
                    timestamp.isoformat() if isinstance(timestamp, datetime) 
                    else str(timestamp)
                )
                entries.append({
                    "id": row.get('id'),
                    "entry_type": row.get('entry_type'),
                    "entry_text": row.get('entry_text'),
                    "revelation_types": row.get('revelation_types'),
                    "narrative_moment": row.get('narrative_moment'),
                    "fantasy_flag": row.get('fantasy_flag'),
                    "intensity_level": row.get('intensity_level'),
                    "timestamp": timestamp_iso
                })
            return entries
        except Exception as e:
            logger.error(f"Error getting player journal entries: {str(e)}", exc_info=True)
            return []

# ===== TOOL LISTS FOR EXPORT =====

dialogue_tools = [
    detect_dialogue_mode,
    generate_dialogue_exchange,
    exit_dialogue_mode,
    get_dialogue_suggestions,
    analyze_dialogue_flow
]

# Context management tools
context_tools = [
    get_optimized_context,
    retrieve_relevant_memories,
    store_narrative_memory,
    search_by_vector,
    get_summarized_narrative_context,
    get_available_npcs,
    get_npc_details
]

# NPC narrative tools
npc_narrative_tools = [
    get_npc_narrative_overview,
    check_all_npc_revelations,
    generate_dynamic_personal_revelation,
    generate_multi_npc_dream,
    check_relationship_milestones,
    generate_stage_contrast_moment,
    get_npc_stage
]

# Activity tools
activity_tools = [
    analyze_activity,
    get_filtered_activities,
    generate_activity_suggestion,
    suggest_stage_appropriate_activity
]

# Relationship tools
relationship_tools = [
    check_relationship_events,
    apply_crossroads_choice,
    check_npc_relationship,
    update_relationship_dimensions,
]

# Conflict management tools
conflict_tools = [
    analyze_conflict_potential,
    generate_conflict_from_analysis,
    analyze_npc_manipulation_potential,
    generate_manipulation_attempt,
    set_player_involvement,
    track_conflict_story_beat,
    suggest_potential_manipulation,
    analyze_manipulation_opportunities,
    generate_conflict_beat
]

# Resource management tools
resource_tools = [
    check_resources,
    commit_resources_to_conflict,
    get_player_resources,
    apply_activity_effects,
    get_resource_history
]

# Narrative element tools (now includes dialogue tools)
narrative_tools = [
    generate_personal_revelation,
    generate_dream_sequence,
    check_relationship_events,
    add_moment_of_clarity,
    get_player_journal_entries
] + npc_narrative_tools + dialogue_tools  # Add dialogue tools here

# All tools combined
all_tools = (
    context_tools +
    activity_tools +
    relationship_tools +
    conflict_tools +
    resource_tools +
    narrative_tools  # This now includes dialogue_tools
)

# Export for easy access
__all__ = [
    'context_tools',
    'activity_tools', 
    'relationship_tools',
    'conflict_tools',
    'resource_tools',
    'narrative_tools',
    'npc_narrative_tools',
    'dialogue_tools',  # Add this
    'all_tools'
]
