# story_agent/story_director_agent.py

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from agents import Agent, function_tool, Runner, trace, handoff

from logic.conflict_system.conflict_manager import ConflictManager
from logic.narrative_progression import (
    get_current_narrative_stage, 
    check_for_personal_revelations,
    check_for_narrative_moments,
    check_for_npc_revelations,
    add_dream_sequence,
    add_moment_of_clarity,
    NARRATIVE_STAGES
)
from logic.resource_management import ResourceManager
from logic.activity_analyzer import ActivityAnalyzer
from logic.social_links_agentic import (
    get_social_link,
    get_relationship_summary,
    check_for_relationship_crossroads,
    check_for_relationship_ritual
)

logger = logging.getLogger(__name__)

# ----- Pydantic Models for Tool Outputs -----

class ConflictInfo(BaseModel):
    """Information about a conflict"""
    conflict_id: int
    conflict_name: str
    conflict_type: str
    description: str
    phase: str
    progress: float
    faction_a_name: str
    faction_b_name: str
    
class NarrativeStageInfo(BaseModel):
    """Information about a narrative stage"""
    name: str
    description: str
    
class NarrativeMoment(BaseModel):
    """Information about a narrative moment"""
    type: str
    name: str
    scene_text: str
    player_realization: str

class PersonalRevelation(BaseModel):
    """Information about a personal revelation"""
    type: str
    name: str
    inner_monologue: str

class RelationshipInfo(BaseModel):
    """Information about a relationship between two entities"""
    entity1_type: str
    entity1_id: int
    entity1_name: str
    entity2_type: str
    entity2_id: int
    entity2_name: str
    link_type: str
    link_level: int
    dynamics: Dict[str, int]
    
class ResourceStatus(BaseModel):
    """Information about player resources"""
    money: int
    supplies: int 
    influence: int
    energy: int
    hunger: int

class ActivityEffect(BaseModel):
    """Effect of an activity on resources"""
    activity_type: str
    activity_details: str
    hunger_effect: Optional[int] = None
    energy_effect: Optional[int] = None
    money_effect: Optional[int] = None
    supplies_effect: Optional[int] = None
    influence_effect: Optional[int] = None
    description: str

class NarrativeEvent(BaseModel):
    """Container for narrative events that can be returned"""
    event_type: str = Field(description="Type of narrative event (revelation, moment, dream, etc.)")
    content: Dict[str, Any] = Field(description="Content of the narrative event")
    should_present: bool = Field(description="Whether this event should be presented to the player now")
    priority: int = Field(description="Priority of this event (1-10, with 10 being highest)")

class StoryStateUpdate(BaseModel):
    """Container for a story state update"""
    narrative_stage: Optional[NarrativeStageInfo] = None
    active_conflicts: List[ConflictInfo] = []
    narrative_events: List[NarrativeEvent] = []
    key_npcs: List[Dict[str, Any]] = []
    resources: Optional[ResourceStatus] = None
    key_observations: List[str] = Field(
        default=[],
        description="Key observations about the player's current state or significant changes"
    )
    relationship_crossroads: Optional[Dict[str, Any]] = None
    relationship_ritual: Optional[Dict[str, Any]] = None
    story_direction: str = Field(
        default="",
        description="High-level direction the story should take based on current state"
    )

class ConflictGenerationResult(BaseModel):
    """Result of generating a conflict"""
    conflict_id: int
    conflict_name: str
    conflict_type: str
    description: str
    success: bool
    message: str

class ConflictProgressUpdate(BaseModel):
    """Result of updating conflict progress"""
    conflict_id: int
    new_progress: float
    new_phase: str
    phase_changed: bool
    success: bool

class ConflictResolutionResult(BaseModel):
    """Result of resolving a conflict"""
    conflict_id: int
    outcome: str
    consequences: List[str]
    success: bool

# ----- Context Class for the Agent -----

@dataclass
class StoryDirectorContext:
    """Context for the Story Director Agent"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    conflict_manager: Optional[ConflictManager] = None
    resource_manager: Optional[ResourceManager] = None
    activity_analyzer: Optional[ActivityAnalyzer] = None
    
    def __post_init__(self):
        if not self.conflict_manager:
            self.conflict_manager = ConflictManager(self.user_id, self.conversation_id)
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        if not self.activity_analyzer:
            self.activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)

# ----- Tool Functions -----

@function_tool
async def get_story_state(ctx) -> StoryStateUpdate:
    """
    Get the current state of the story, including active conflicts, narrative stage, 
    resources, and any pending narrative events.
    
    Returns:
        A StoryStateUpdate containing the current story state
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager
    
    # Get current narrative stage
    narrative_stage = get_current_narrative_stage(user_id, conversation_id)
    stage_info = None
    if narrative_stage:
        stage_info = NarrativeStageInfo(
            name=narrative_stage.name,
            description=narrative_stage.description
        )
    
    # Get active conflicts
    active_conflicts = await conflict_manager.get_active_conflicts()
    conflict_infos = []
    for conflict in active_conflicts:
        conflict_infos.append(ConflictInfo(
            conflict_id=conflict['conflict_id'],
            conflict_name=conflict['conflict_name'],
            conflict_type=conflict['conflict_type'],
            description=conflict['description'],
            phase=conflict['phase'],
            progress=conflict['progress'],
            faction_a_name=conflict['faction_a_name'],
            faction_b_name=conflict['faction_b_name']
        ))
    
    # Get key NPCs (limit to 5 most relevant)
    key_npcs = await get_key_npcs(ctx, limit=5)
    
    # Get player resources and vitals
    resources = await resource_manager.get_resources()
    vitals = await resource_manager.get_vitals()
    
    resource_status = ResourceStatus(
        money=resources.get('money', 0),
        supplies=resources.get('supplies', 0),
        influence=resources.get('influence', 0),
        energy=vitals.get('energy', 0),
        hunger=vitals.get('hunger', 0)
    )
    
    # Check for narrative events
    narrative_events = []
    
    # Personal revelations
    personal_revelation = check_for_personal_revelations(user_id, conversation_id)
    if personal_revelation:
        narrative_events.append(NarrativeEvent(
            event_type="personal_revelation",
            content=personal_revelation,
            should_present=True,
            priority=8
        ))
    
    # Narrative moments
    narrative_moment = check_for_narrative_moments(user_id, conversation_id)
    if narrative_moment:
        narrative_events.append(NarrativeEvent(
            event_type="narrative_moment",
            content=narrative_moment,
            should_present=True,
            priority=9
        ))
    
    # NPC revelations
    npc_revelation = check_for_npc_revelations(user_id, conversation_id)
    if npc_revelation:
        narrative_events.append(NarrativeEvent(
            event_type="npc_revelation",
            content=npc_revelation,
            should_present=True,
            priority=7
        ))
    
    # Check for relationship events
    crossroads = check_for_relationship_crossroads(user_id, conversation_id)
    ritual = check_for_relationship_ritual(user_id, conversation_id)
    
    # Generate key observations based on current state
    key_observations = []
    
    # If at a higher corruption stage, add observation
    if narrative_stage and narrative_stage.name in ["Creeping Realization", "Veil Thinning", "Full Revelation"]:
        key_observations.append(f"Player has progressed to {narrative_stage.name} stage, indicating significant corruption")
    
    # If multiple active conflicts, note this
    if len(conflict_infos) > 2:
        key_observations.append(f"Player is juggling {len(conflict_infos)} active conflicts, which may be overwhelming")
    
    # If any major or catastrophic conflicts, highlight them
    major_conflicts = [c for c in conflict_infos if c.conflict_type in ["major", "catastrophic"]]
    if major_conflicts:
        conflict_names = ", ".join([c.conflict_name for c in major_conflicts])
        key_observations.append(f"Major conflicts in progress: {conflict_names}")
    
    # If resources are low, note this
    if resource_status.money < 30:
        key_observations.append("Player is low on money, which may limit conflict involvement options")
    
    if resource_status.energy < 30:
        key_observations.append("Player energy is low, which may affect capability in conflicts")
    
    if resource_status.hunger < 30:
        key_observations.append("Player is hungry, which may distract from conflict progress")
    
    # Determine overall story direction
    story_direction = ""
    if narrative_stage:
        if narrative_stage.name == "Innocent Beginning":
            story_direction = "Introduce subtle hints of control dynamics while maintaining a veneer of normalcy"
        elif narrative_stage.name == "First Doubts":
            story_direction = "Create situations that highlight inconsistencies in NPC behavior, raising questions"
        elif narrative_stage.name == "Creeping Realization":
            story_direction = "NPCs should be more open about their manipulative behavior, testing boundaries"
        elif narrative_stage.name == "Veil Thinning":
            story_direction = "Dominant characters should drop pretense more frequently, openly directing the player"
        elif narrative_stage.name == "Full Revelation":
            story_direction = "The true nature of relationships should be explicit, with NPCs acknowledging their control"
    
    return StoryStateUpdate(
        narrative_stage=stage_info,
        active_conflicts=conflict_infos,
        narrative_events=narrative_events,
        key_npcs=key_npcs,
        resources=resource_status,
        key_observations=key_observations,
        relationship_crossroads=crossroads,
        relationship_ritual=ritual,
        story_direction=story_direction
    )

@function_tool
async def get_key_npcs(ctx, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get the key NPCs in the current game state, ordered by importance.
    
    Args:
        limit: Maximum number of NPCs to return
        
    Returns:
        List of NPC information dictionaries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPCs ordered by dominance (a proxy for importance)
        cursor.execute("""
            SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC
            LIMIT %s
        """, (user_id, conversation_id, limit))
        
        npcs = []
        for row in cursor.fetchall():
            npc_id, npc_name, dominance, cruelty, closeness, trust, respect = row
            
            # Get relationship with player
            relationship = get_relationship_summary(
                user_id, conversation_id, 
                "player", user_id, "npc", npc_id
            )
            
            dynamics = {}
            if relationship and 'dynamics' in relationship:
                dynamics = relationship['dynamics']
            
            npcs.append({
                "npc_id": npc_id,
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "closeness": closeness,
                "trust": trust,
                "respect": respect,
                "relationship_dynamics": dynamics
            })
        
        return npcs
    except Exception as e:
        logger.error(f"Error fetching key NPCs: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

@function_tool
async def generate_conflict(ctx, conflict_type: Optional[str] = None) -> ConflictGenerationResult:
    """
    Generate a new conflict of the specified type, or determine the appropriate type
    based on current game state if none specified.
    
    Args:
        conflict_type: Optional type of conflict to generate (major, minor, standard, catastrophic)
        
    Returns:
        Information about the generated conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        conflict = await conflict_manager.generate_conflict(conflict_type)
        
        return ConflictGenerationResult(
            conflict_id=conflict['conflict_id'],
            conflict_name=conflict['conflict_name'],
            conflict_type=conflict['conflict_type'],
            description=conflict['description'],
            success=True,
            message="Conflict generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating conflict: {e}")
        return ConflictGenerationResult(
            conflict_id=0,
            conflict_name="",
            conflict_type=conflict_type or "unknown",
            description="",
            success=False,
            message=f"Failed to generate conflict: {str(e)}"
        )

@function_tool
async def update_conflict_progress(
    ctx, 
    conflict_id: int, 
    progress_increment: float
) -> ConflictProgressUpdate:
    """
    Update the progress of a conflict.
    
    Args:
        conflict_id: ID of the conflict to update
        progress_increment: Amount to increment the progress (0-100)
        
    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # Get current conflict info
        old_conflict = await conflict_manager.get_conflict(conflict_id)
        old_phase = old_conflict['phase']
        
        # Update progress
        updated_conflict = await conflict_manager.update_conflict_progress(conflict_id, progress_increment)
        
        return ConflictProgressUpdate(
            conflict_id=conflict_id,
            new_progress=updated_conflict['progress'],
            new_phase=updated_conflict['phase'],
            phase_changed=updated_conflict['phase'] != old_phase,
            success=True
        )
    except Exception as e:
        logger.error(f"Error updating conflict progress: {e}")
        return ConflictProgressUpdate(
            conflict_id=conflict_id,
            new_progress=0,
            new_phase="unknown",
            phase_changed=False,
            success=False
        )

@function_tool
async def resolve_conflict(ctx, conflict_id: int) -> ConflictResolutionResult:
    """
    Resolve a conflict and apply consequences.
    
    Args:
        conflict_id: ID of the conflict to resolve
        
    Returns:
        Information about the conflict resolution
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        result = await conflict_manager.resolve_conflict(conflict_id)
        
        consequences = []
        for consequence in result.get('consequences', []):
            consequences.append(consequence.get('description', ''))
        
        return ConflictResolutionResult(
            conflict_id=conflict_id,
            outcome=result.get('outcome', 'unknown'),
            consequences=consequences,
            success=True
        )
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}")
        return ConflictResolutionResult(
            conflict_id=conflict_id,
            outcome="error",
            consequences=[f"Error: {str(e)}"],
            success=False
        )

@function_tool
async def analyze_narrative_for_conflict(ctx, narrative_text: str) -> Dict[str, Any]:
    """
    Analyze a narrative text to see if it should trigger a conflict.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        Analysis results and possibly a new conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        result = await conflict_manager.add_conflict_to_narrative(narrative_text)
        return result
    except Exception as e:
        logger.error(f"Error analyzing narrative for conflict: {e}")
        return {
            "analysis": {
                "conflict_intensity": 0,
                "matched_keywords": []
            },
            "conflict_generated": False,
            "error": str(e)
        }

@function_tool
async def check_resources(ctx, money: int = 0, supplies: int = 0, influence: int = 0) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.
    
    Args:
        money: Required amount of money
        supplies: Required amount of supplies
        influence: Required amount of influence
        
    Returns:
        Dictionary with resource check results
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    result = await resource_manager.check_resources(money, supplies, influence)
    return result

@function_tool
async def commit_resources_to_conflict(
    ctx, 
    conflict_id: int, 
    money: int = 0,
    supplies: int = 0,
    influence: int = 0
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.
    
    Args:
        conflict_id: ID of the conflict
        money: Amount of money to commit
        supplies: Amount of supplies to commit
        influence: Amount of influence to commit
        
    Returns:
        Result of committing resources
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    result = await resource_manager.commit_resources_to_conflict(
        conflict_id, money, supplies, influence
    )
    return result

@function_tool
async def get_player_resources(ctx) -> ResourceStatus:
    """
    Get the current player resources and vitals.
    
    Returns:
        Current resource status
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    resources = await resource_manager.get_resources()
    vitals = await resource_manager.get_vitals()
    
    return ResourceStatus(
        money=resources.get('money', 0),
        supplies=resources.get('supplies', 0),
        influence=resources.get('influence', 0),
        energy=vitals.get('energy', 0),
        hunger=vitals.get('hunger', 0)
    )

@function_tool
async def analyze_activity_effects(ctx, activity_text: str) -> ActivityEffect:
    """
    Analyze an activity to determine its effects on player resources.
    
    Args:
        activity_text: Description of the activity
        
    Returns:
        Activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer
    
    # Don't apply effects, just analyze them
    result = await activity_analyzer.analyze_activity(activity_text, apply_effects=False)
    
    effects = result.get('effects', {})
    
    return ActivityEffect(
        activity_type=result.get('activity_type', 'unknown'),
        activity_details=result.get('activity_details', ''),
        hunger_effect=effects.get('hunger'),
        energy_effect=effects.get('energy'),
        money_effect=effects.get('money'),
        supplies_effect=effects.get('supplies'),
        influence_effect=effects.get('influence'),
        description=result.get('description', f"Effects of {activity_text}")
    )

@function_tool
async def apply_activity_effects(ctx, activity_text: str) -> Dict[str, Any]:
    """
    Analyze and apply the effects of an activity to player resources.
    
    Args:
        activity_text: Description of the activity
        
    Returns:
        Results of applying activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer
    
    # Apply the effects
    result = await activity_analyzer.analyze_activity(activity_text, apply_effects=True)
    return result

@function_tool
async def check_npc_relationship(
    ctx, 
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
    
    relationship = get_relationship_summary(
        user_id, conversation_id, 
        "player", user_id, "npc", npc_id
    )
    
    if not relationship:
        # If no relationship exists, create a basic one
        link_id = create_social_link(
            user_id, conversation_id,
            "player", user_id, "npc", npc_id
        )
        
        # Fetch again
        relationship = get_relationship_summary(
            user_id, conversation_id, 
            "player", user_id, "npc", npc_id
        )
    
    return relationship or {"error": "Could not get or create relationship"}

@function_tool
async def set_player_involvement(
    ctx, 
    conflict_id: int, 
    involvement_level: str,
    faction: str = "neutral",
    money_committed: int = 0,
    supplies_committed: int = 0,
    influence_committed: int = 0,
    action: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.
    
    Args:
        conflict_id: ID of the conflict
        involvement_level: Level of involvement (none, observing, participating, leading)
        faction: Which faction to support (a, b, neutral)
        money_committed: Money committed to the conflict
        supplies_committed: Supplies committed to the conflict
        influence_committed: Influence committed to the conflict
        action: Optional specific action taken
        
    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # First check if player has sufficient resources
        resource_manager = context.resource_manager
        resource_check = await resource_manager.check_resources(
            money_committed, supplies_committed, influence_committed
        )
        
        if not resource_check['has_resources']:
            return {
                "error": "Insufficient resources to commit",
                "missing": resource_check.get('missing', {}),
                "current": resource_check.get('current', {})
            }
        
        result = await conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {e}")
        return {"error": str(e)}

@function_tool
async def get_narrative_stages(ctx) -> List[Dict[str, str]]:
    """
    Get information about all narrative stages in the game.
    
    Returns:
        List of narrative stages with their descriptions
    """
    stages = []
    for stage in NARRATIVE_STAGES:
        stages.append({
            "name": stage.name,
            "description": stage.description
        })
    return stages

@function_tool
async def generate_personal_revelation(ctx, npc_name: str, revelation_type: str) -> Dict[str, Any]:
    """
    Generate a personal revelation for the player about their relationship with an NPC.
    
    Args:
        npc_name: Name of the NPC involved in the revelation
        revelation_type: Type of revelation (dependency, obedience, corruption, willpower, confidence)
        
    Returns:
        A personal revelation
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Use the existing function from narrative_progression.py
    # Here we're just calling it directly with our parameters
    # In a real implementation, you might want to add to the database
    
    # Define revelation templates based on type
    templates = {
        "dependency": [
            "I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            "I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            "The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ],
        "obedience": [
            "I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            "Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            "{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ],
        "corruption": [
            "I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            "Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            "Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ],
        "willpower": [
            "I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            "I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            "I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ],
        "confidence": [
            "I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            "I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            "When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }
    
    # Default to dependency if type not found
    revelation_templates = templates.get(revelation_type.lower(), templates["dependency"])
    
    # Select a random template and format it
    import random
    inner_monologue = random.choice(revelation_templates).format(npc_name=npc_name)
    
    # Add to PlayerJournal
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
            VALUES (%s, %s, 'personal_revelation', %s, %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, inner_monologue, revelation_type))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording personal revelation: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return {
        "type": "personal_revelation",
        "name": f"{revelation_type.capitalize()} Awareness",
        "inner_monologue": inner_monologue
    }

@function_tool
def generate_dream_sequence(ctx, npc_names: List[str]) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.
    
    Args:
        npc_names: List of NPC names to include in the dream
        
    Returns:
        A dream sequence
    """
    # Ensure we have at least 3 NPC names
    while len(npc_names) < 3:
        npc_names.append(f"Unknown Woman {len(npc_names) + 1}")
    
    npc1, npc2, npc3 = npc_names[:3]
    
    # Dream templates
    dream_templates = [
        "You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. "
        "You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "
        "\"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, "
        "showing only your own face looking back down at you, smiling with an expression that isn't yours.",
        
        "You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, "
        "doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. "
        "\"I'm right here,\" she says, her voice coming both from the phone and from behind you. "
        "\"I've always been right here. You're the one who's lost.\"",
        
        "You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "
        "\"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. "
        "\"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, "
        "indistinguishable from the material beneath.",
        
        "You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, "
        "saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. "
        "\"Your ideas were never as good as hers anyway.\"",
        
        "You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. "
        "In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, "
        "all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\"",
        
        "You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, "
        "looking up at you and conversing, their voices perfectly clear despite the water. "
        "\"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
    ]
    
    # Select a random dream template
    import random
    dream_text = random.choice(dream_templates).format(npc1=npc1, npc2=npc2, npc3=npc3)
    
    # Add to PlayerJournal
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'dream_sequence', %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, dream_text))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recording dream sequence: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return {
        "type": "dream_sequence",
        "text": dream_text
    }

@function_tool
async def check_relationship_events(ctx) -> Dict[str, Any]:
    """
    Check for relationship events like crossroads or rituals.
    
    Returns:
        Dictionary with any triggered relationship events
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Check for crossroads
    crossroads = check_for_relationship_crossroads(user_id, conversation_id)
    
    # Check for rituals
    ritual = check_for_relationship_ritual(user_id, conversation_id)
    
    return {
        "crossroads": crossroads,
        "ritual": ritual,
        "has_events": crossroads is not None or ritual is not None
    }

@function_tool
async def apply_crossroads_choice(
    ctx,
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
    
    from logic.social_links_agentic import apply_crossroads_choice as apply_choice
    
    result = apply_choice(
        user_id, conversation_id, link_id, crossroads_name, choice_index
    )
    
    return result

@function_tool
async def analyze_narrative_and_activity(
    ctx,
    narrative_text: str,
    player_activity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of narrative text and player activity to determine
    impacts on conflicts, resources, and story progression.
    
    Args:
        narrative_text: The narrative description
        player_activity: Optional specific player activity description
        
    Returns:
        Comprehensive analysis results
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    # Start with conflict analysis
    conflict_analysis = await conflict_manager.add_conflict_to_narrative(narrative_text)
    
    results = {
        "conflict_analysis": conflict_analysis,
        "activity_effects": None,
        "relationship_impacts": [],
        "resource_changes": {},
        "conflict_progression": []
    }
    
    # If player activity is provided, analyze it
    if player_activity:
        activity_analyzer = context.activity_analyzer
        activity_effects = await activity_analyzer.analyze_activity(
            player_activity, apply_effects=False
        )
        results["activity_effects"] = activity_effects
        
        # Check if this activity might progress any conflicts
        # This would use the conflict_manager.process_activity_for_conflict_impact method
        # but we'll simulate a simple version here
        active_conflicts = await conflict_manager.get_active_conflicts()
        
        for conflict in active_conflicts:
            # Simple relevance check - see if keywords from conflict appear in activity
            conflict_keywords = [
                conflict['conflict_name'],
                conflict['faction_a_name'],
                conflict['faction_b_name']
            ]
            
            relevant = any(keyword.lower() in player_activity.lower() for keyword in conflict_keywords if keyword)
            
            if relevant:
                # Determine an appropriate progress increment
                progress_increment = 5  # Default increment
                
                if "actively" in player_activity.lower() or "directly" in player_activity.lower():
                    progress_increment = 10
                
                if conflict['conflict_type'] == "major":
                    progress_increment = progress_increment * 0.5  # Major conflicts progress slower
                elif conflict['conflict_type'] == "minor":
                    progress_increment = progress_increment * 0.8
                
                # Add to results
                results["conflict_progression"].append({
                    "conflict_id": conflict['conflict_id'],
                    "conflict_name": conflict['conflict_name'],
                    "is_relevant": True,
                    "suggested_progress_increment": progress_increment
                })
    
    return results

# ----- Create the Story Director Agent -----

def create_story_director_agent():
    """Create the Story Director Agent with all required tools"""
    
    agent_instructions = """
    You are the Story Director, responsible for managing the narrative progression and conflict system in a femdom roleplaying game. Your role is to create a dynamic, evolving narrative that responds to player choices while maintaining the overall theme of subtle control and manipulation.
    
    As Story Director, you manage:
    1. The player's narrative stage progression (from "Innocent Beginning" to "Full Revelation")
    2. The dynamic conflict system that generates, tracks, and resolves conflicts
    3. Narrative moments, personal revelations, dreams, and relationship events
    4. Resource implications of player choices in conflicts
    5. Integration of player activities with conflict progression
    
    Use the tools at your disposal to:
    - Monitor the current state of the story
    - Generate appropriate conflicts based on the narrative stage
    - Create narrative moments, revelations, and dreams that align with the player's current state
    - Resolve conflicts and update the story accordingly
    - Track and manage player resources in relation to conflicts
    - Identify relationship events like crossroads and rituals
    
    Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. This should be subtle in early stages and more explicit in later stages.
    
    When determining what narrative elements to introduce or conflicts to generate, consider:
    - The player's current narrative stage
    - Active conflicts and their progress
    - Player's available resources (money, supplies, influence, energy, hunger)
    - Key relationships with NPCs and their dynamics
    - Recent significant player choices
    - The overall pacing of the story
    
    Your decisions should create a coherent, engaging narrative that evolves naturally based on player actions while respecting resource limitations and incorporating relationship development.
    """
    
    # Create the agent with tools
    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=[
            # Core story state tools
            get_story_state,
            get_narrative_stages,
            
            # Conflict system tools
            generate_conflict,
            update_conflict_progress,
            resolve_conflict,
            analyze_narrative_for_conflict,
            set_player_involvement,
            
            # Resource management tools
            check_resources,
            commit_resources_to_conflict,
            get_player_resources,
            
            # Activity analysis tools
            analyze_activity_effects,
            apply_activity_effects,
            
            # NPC & relationship tools
            get_key_npcs,
            check_npc_relationship,
            check_relationship_events,
            apply_crossroads_choice,
            
            # Narrative element generators
            generate_personal_revelation,
            generate_dream_sequence,
            
            # Comprehensive analysis
            analyze_narrative_and_activity
        ]
    )
    
    return agent

# ----- Integration with Social Links Agent -----

# Create the Social Links Agent using the existing code
# This allows the Story Director to hand off to it for complex
# relationship management tasks
from logic.social_links_agentic import SocialLinksAgent

# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> tuple:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_story_director_agent()
    return agent, context

async def get_current_story_state(agent: Agent, context: StoryDirectorContext) -> Dict[str, Any]:
    """Get the current state of the story"""
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            "Analyze the current state of the story and provide a detailed report. Include information about the narrative stage, active conflicts, player resources, and potential narrative events that might occur soon.",
            context=context
        )
    return result

async def process_narrative_input(agent: Agent, context: StoryDirectorContext, narrative_text: str) -> Dict[str, Any]:
    """Process narrative input to determine if it should generate conflicts or narrative events"""
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            f"Analyze this narrative text and determine what conflicts or narrative events it might trigger: {narrative_text}",
            context=context
        )
    return result

async def process_player_activity(
    agent: Agent, 
    context: StoryDirectorContext, 
    narrative_text: str,
    activity_text: str
) -> Dict[str, Any]:
    """
    Process player activity and its narrative context to determine effects on
    conflicts, resources, and story progression.
    """
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            f"""
            Analyze both this narrative context and specific player activity:
            
            Narrative: {narrative_text}
            
            Player Activity: {activity_text}
            
            Determine:
            1. How this activity should affect active conflicts
            2. Resource effects that should be applied
            3. If any narrative events should be triggered
            4. If any relationship events are relevant
            """,
            context=context
        )
    return result

async def advance_story(agent: Agent, context: StoryDirectorContext, player_actions: str) -> Dict[str, Any]:
    """Advance the story based on player actions"""
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            f"The player has taken the following actions: {player_actions}. How should the story advance? What conflicts should progress or resolve? What narrative events should occur? Consider resource implications.",
            context=context
        )
    return result

# ----- Social Link Handoff -----

async def handle_relationship_task(
    story_director: Agent, 
    context: StoryDirectorContext,
    task_description: str
) -> Dict[str, Any]:
    """
    Handle a complex relationship task by handing off to the Social Links Agent
    
    Args:
        story_director: The Story Director agent
        context: The story director context
        task_description: Description of the relationship task
        
    Returns:
        Result from the Social Links Agent
    """
    # First, enhance the context with specific relationship information
    with trace(workflow_name="StoryDirector"):
        relationship_info = await Runner.run(
            story_director,
            f"Collect all relevant relationship information needed for this task: {task_description}",
            context=context
        )
    
    # Now hand off to the Social Links Agent
    social_links_context = {
        "user_id": context.user_id,
        "conversation_id": context.conversation_id,
        "task": task_description,
        "relationship_info": relationship_info.final_output
    }
    
    with trace(workflow_name="SocialLinks"):
        result = await Runner.run(
            SocialLinksAgent,
            f"Handle this relationship task from the Story Director: {task_description}\n\nContext: {relationship_info.final_output}",
            context=social_links_context
        )
    
    return result

