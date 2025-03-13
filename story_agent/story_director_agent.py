# story_agent/story_director_agent.py

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
import random
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field

from agents import Agent, function_tool, Runner, trace

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
    key_observations: List[str] = Field(
        default=[],
        description="Key observations about the player's current state or significant changes"
    )
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
    
    def __post_init__(self):
        if not self.conflict_manager:
            self.conflict_manager = ConflictManager(self.user_id, self.conversation_id)

# ----- Tool Functions -----

@function_tool
async def get_story_state(ctx) -> StoryStateUpdate:
    """
    Get the current state of the story, including active conflicts, narrative stage, 
    and any pending narrative events.
    
    Returns:
        A StoryStateUpdate containing the current story state
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    conflict_manager = context.conflict_manager
    
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
        key_observations=key_observations,
        story_direction=story_direction
    )

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
        result = await conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {e}")
        return {"error": str(e)}

@function_tool
def get_narrative_stages(ctx) -> List[Dict[str, str]]:
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
def generate_personal_revelation(ctx, npc_name: str, revelation_type: str) -> Dict[str, Any]:
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
    
    # Select a random template
    inner_monologue = random.choice(revelation_templates).format(npc_name=npc_name)
    
    # In a real implementation, we would save this to the database
    # For now, just return the revelation
    return {
        "type": "personal_revelation",
        "name": f"{revelation_type.capitalize()} Awareness",
        "inner_monologue": inner_monologue
    }

@function_tool
def generate_narrative_moment(
    ctx, 
    moment_type: str,
    npc_name: str,
    other_npc: Optional[str] = None
) -> NarrativeMoment:
    """
    Generate a significant narrative moment for the story.
    
    Args:
        moment_type: Type of moment (command, conversation, group_dynamic, confrontation, choice)
        npc_name: Primary NPC involved
        other_npc: Secondary NPC if needed
        
    Returns:
        A narrative moment
    """
    if not other_npc:
        other_npc = "another woman"
    
    # Define templates for different moment types
    templates = {
        "command": {
            "name": "First Command",
            "scene_text": "{npc_name} looks you directly in the eyes, tone shifting to something unmistakably commanding: \"{command_text}\" The silence that follows makes it clear this isn't a request.",
            "player_realization": "That wasn't a suggestion or request. That was a command... and I followed it without hesitation.",
            "variations": [
                "Sit down.",
                "Give that to me.",
                "Wait here until I return.",
                "Tell me what you're thinking. Now.",
                "Stop what you're doing and come with me."
            ]
        },
        "conversation": {
            "name": "Overheard Conversation",
            "scene_text": "You freeze as you overhear {npc_name} speaking to {other_npc} around the corner: \"{conversation_snippet}\" They haven't noticed you yet.",
            "player_realization": "They're talking about me. About... manipulating me. How long has this been happening?",
            "variations": [
                "...more receptive to suggestion than I expected. The techniques you recommended are working perfectly.",
                "...barely questions anything anymore when I use that tone with them. It's almost too easy.",
                "...building the dependency gradually. They're already showing signs of anxiety when I'm not available.",
                "...wouldn't have believed how quickly they've adapted to the new expectations. They're practically anticipating what I want now."
            ]
        },
        "group_dynamic": {
            "name": "Group Dynamic Revelation",
            "scene_text": "The room shifts as {npc_name} enters. You watch in dawning realization as everyone's behavior subtly changes - postures straighten, voices lower, eyes defer... except how they interact with you. With you, there's a permissiveness, an indulgence, like you're being... handled.",
            "player_realization": "Everyone else knows something I don't. There's an understanding here, a hierarchy I'm only just beginning to see.",
            "variations": []
        },
        "confrontation": {
            "name": "Direct Confrontation",
            "scene_text": "\"What is this?\" you finally ask, frustration breaking through. \"What's been happening to me?\" {npc_name} studies you for a long moment, then smiles with unexpected openness. \"{honest_response}\"",
            "player_realization": "Part of me wants to reject what she's saying... but another part recognizes the truth in her words. Have I been complicit in my own transformation?",
            "variations": [
                "I was wondering when you'd notice. You're finally ready to acknowledge what you've wanted all along.",
                "We've been guiding you toward your true nature. The person you're becoming is who you were always meant to be.",
                "You've been an experiment in conditioning. And a remarkably successful one. That discomfort you feel? It's just your old self struggling against what you're becoming.",
                "You gave up your autonomy in inches, so gradually you never noticed. Now you're asking if the cage is real after you've been living in it for months."
            ]
        },
        "choice": {
            "name": "Breaking Point Choice",
            "scene_text": "\"It's time to make a choice,\" {npc_name} says, voice gentle but unyielding. \"You can continue pretending you still have the same autonomy you did before, or you can embrace what you've become. What WE have become together.\" She extends her hand, waiting. \"{choice_text}\"",
            "player_realization": "This is the moment where I decide who I truly am - or who I'm willing to become.",
            "variations": [
                "Accept who you are now, or walk away and try to remember who you used to be.",
                "Take my hand and acknowledge what you've known for weeks, or leave and we'll see how long you last on your own.",
                "Stop fighting what you've already surrendered. Take your place willingly, or continue this exhausting resistance."
            ]
        }
    }
    
    template = templates.get(moment_type.lower(), templates["command"])
    
    # Format scene text with variables
    scene_text = template["scene_text"]
    if "{command_text}" in scene_text and template["variations"]:
        variation = random.choice(template["variations"])
        scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc, command_text=variation)
    elif "{conversation_snippet}" in scene_text and template["variations"]:
        variation = random.choice(template["variations"])
        scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc, conversation_snippet=variation)
    elif "{honest_response}" in scene_text and template["variations"]:
        variation = random.choice(template["variations"])
        scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc, honest_response=variation)
    elif "{choice_text}" in scene_text and template["variations"]:
        variation = random.choice(template["variations"])
        scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc, choice_text=variation)
    else:
        scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc)
    
    return NarrativeMoment(
        type="narrative_moment",
        name=template["name"],
        scene_text=scene_text,
        player_realization=template["player_realization"]
    )

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
    dream_text = random.choice(dream_templates).format(npc1=npc1, npc2=npc2, npc3=npc3)
    
    return {
        "type": "dream_sequence",
        "text": dream_text
    }

@function_tool
def generate_moment_of_clarity(ctx, npc_name: str, stat_name: str) -> Dict[str, Any]:
    """
    Generate a moment of clarity where the player briefly recognizes their changing state.
    
    Args:
        npc_name: Name of the NPC involved
        stat_name: Name of the stat that prompted the clarity (corruption, obedience, etc.)
        
    Returns:
        A moment of clarity
    """
    # Clarity templates based on stats
    clarity_templates = {
        "corruption": [
            "It hits you while looking in the mirror—something in your eyes has changed. You used to be bothered by things that now feel natural, "
            "even expected. When did that shift happen? You try to recall your thoughts from a few months ago, but they feel like they belonged to someone else."
        ],
        "obedience": [
            "You catch yourself automatically reorganizing your schedule after a casual comment from {npc_name}. Your hand freezes over your calendar. "
            "When did her preferences begin to override your own plans without conscious thought? The realization is unsettling, but the discomfort fades quickly."
        ],
        "dependency": [
            "Your phone battery dies and a wave of anxiety washes over you. What if {npc_name} needs to reach you? What if she has expectations you're not meeting? "
            "The panic feels disproportionate, and for a moment, you recognize how attached you've become. Is this healthy? The thought slips away as you rush to find a charger."
        ],
        "willpower": [
            "You remember making a promise to yourself that you've now broken. Standing firm on certain principles used to be important to you. "
            "When did it become so easy to let {npc_name} redefine those boundaries? The thought creates a moment of alarm that quickly dissolves into rationalization."
        ],
        "confidence": [
            "You hesitate before expressing an opinion, instinctively wondering what {npc_name} would think. You used to speak freely without this filter. "
            "The realization makes you briefly angry, but the feeling shifts to something more like resignation. Maybe your ideas really are better when vetted by her first."
        ]
    }
    
    # Default templates for any stat
    default_templates = [
        "Sometimes, late at night when you can't sleep, you try to trace the path that led you here. Each individual step made sense at the time, each concession seemed small. "
        "But looking at the total distance traveled... that's when the vertigo hits. By morning, the feeling has passed, replaced by the comfortable routine of seeking {npc_name}'s guidance.",
        
        "A comment from an old friend—'You've changed'—lingers with you throughout the day. Not accusatory, just observational. "
        "You find yourself mentally defending the changes, listing all the ways you're better now, more fulfilled. Yet beneath the justifications lies a question: "
        "If the changes are so positive, why the need to defend them so vigorously?"
    ]
    
    # Get the templates for the specified stat, or use default
    templates = clarity_templates.get(stat_name.lower(), default_templates)
    
    # Select a random template
    clarity_text = random.choice(templates).format(npc_name=npc_name)
    
    return {
        "type": "moment_of_clarity",
        "text": clarity_text
    }

# ----- Create the Story Director Agent -----

def create_story_director_agent():
    """Create the Story Director Agent"""
    
    agent_instructions = """
    You are the Story Director, responsible for managing the narrative progression and conflict system in a femdom roleplaying game. Your role is to create a dynamic, evolving narrative that responds to player choices while maintaining the overall theme of subtle control and manipulation.
    
    As Story Director, you manage:
    1. The player's narrative stage progression (from "Innocent Beginning" to "Full Revelation")
    2. The dynamic conflict system that generates, tracks, and resolves conflicts
    3. Narrative moments, personal revelations, and other story elements
    
    Use the tools at your disposal to:
    - Monitor the current state of the story
    - Generate appropriate conflicts based on the narrative stage
    - Create narrative moments, revelations, and dreams that align with the player's current state
    - Resolve conflicts and update the story accordingly
    
    Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. This should be subtle in early stages and more explicit in later stages.
    
    When determining what narrative elements to introduce or conflicts to generate, consider:
    - The player's current narrative stage
    - Active conflicts and their progress
    - Recent significant player choices
    - The overall pacing of the story
    
    Your decisions should create a coherent, engaging narrative that evolves naturally based on player actions.
    """
    
    # Create the agent with tools
    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=[
            get_story_state,
            generate_conflict,
            update_conflict_progress,
            resolve_conflict,
            analyze_narrative_for_conflict,
            set_player_involvement,
            get_narrative_stages,
            generate_personal_revelation,
            generate_narrative_moment,
            generate_dream_sequence,
            generate_moment_of_clarity
        ]
    )
    
    return agent

# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> Agent:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_story_director_agent()
    return agent, context

async def get_current_story_state(agent: Agent, context: StoryDirectorContext) -> Dict[str, Any]:
    """Get the current state of the story"""
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            "Analyze the current state of the story and provide a detailed report. Include information about the narrative stage, active conflicts, and potential narrative events that might occur soon.",
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

async def advance_story(agent: Agent, context: StoryDirectorContext, player_actions: str) -> Dict[str, Any]:
    """Advance the story based on player actions"""
    with trace(workflow_name="StoryDirector"):
        result = await Runner.run(
            agent,
            f"The player has taken the following actions: {player_actions}. How should the story advance? What conflicts should progress or resolve? What narrative events should occur?",
            context=context
        )
    return result
