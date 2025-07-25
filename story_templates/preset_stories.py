# story_templates/preset_stories.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

@dataclass
class StoryBeat:
    """A key moment in the preset story"""
    id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]  # What needs to happen to trigger this
    required_npcs: List[str]  # NPCs that must exist
    required_locations: List[str]  # Locations that must exist
    narrative_stage: str  # Which narrative stage this should happen in
    outcomes: Dict[str, Any]  # What changes when this beat occurs
    dialogue_hints: List[str]  # Key phrases or themes for dynamic dialogue
    can_skip: bool = False  # Whether this beat is mandatory

@dataclass
class PresetStory:
    """A complete preset story with beats and requirements"""
    id: str
    name: str
    theme: str
    synopsis: str
    
    # Story structure
    acts: List[Dict[str, Any]]  # Act structure with beats
    story_beats: List[StoryBeat]
    
    # Required elements
    required_npcs: List[Dict[str, Any]]  # NPCs with specific traits
    required_locations: List[Dict[str, Any]]  # Key locations
    required_conflicts: List[Dict[str, Any]]  # Preset conflicts
    
    # Customization hooks
    dynamic_elements: Dict[str, Any]  # What can be dynamically generated
    player_choices_matter: List[str]  # Key decision points
    
    # Integration settings
    flexibility_level: float = 0.7  # How much deviation is allowed (0-1)
    enforce_ending: bool = True  # Whether to force the preset ending

# Example preset story - COMMENTED OUT AS IT'S INCOMPLETE
# EXAMPLE_PRESET_STORY = PresetStory(
#     id="the_gilded_cage",
#     name="The Gilded Cage",
#     theme="A seemingly perfect town hides a web of control",
#     synopsis="You arrive in Rosehaven seeking a fresh start, but find the town's warmth masks something darker.",
#     
#     acts=[
#         {
#             "act": 1,
#             "name": "Welcome to Paradise",
#             "beats": ["arrival", "first_job", "meet_council", "first_doubt"]
#         },
#         {
#             "act": 2, 
#             "name": "The Silk Threads",
#             "beats": ["obligation_web", "failed_escape", "deeper_involvement", "revelation"]
#         },
#         {
#             "act": 3,
#             "name": "Acceptance or Rebellion",
#             "beats": ["final_choice", "consequences", "ending"]
#         }
#     ],
#     
#     story_beats=[
#         StoryBeat(
#             id="arrival",
#             name="Fresh Start in Rosehaven",
#             description="Player arrives in town, greeted warmly by the Council",
#             trigger_conditions={"game_start": True},
#             required_npcs=["council_leader"],
#             required_locations=["town_square", "council_hall"],
#             narrative_stage="Innocent Beginning",
#             outcomes={"player_reputation": 50, "council_awareness": 10},
#             dialogue_hints=["welcome", "fresh start", "we've been expecting you"],
#             can_skip=False
#         ),
#         # ... more beats
#     ],
#     
#     required_npcs=[
#         {
#             "id": "council_leader",
#             "name": "Victoria Sterling",
#             "archetype": "Mentor",
#             "traits": ["charismatic", "controlling", "maternal"],
#             "role": "Town Council Leader",
#             "narrative_importance": "critical"
#         }
#     ],
#     
#     dynamic_elements={
#         "minor_npcs": True,
#         "side_quests": True,
#         "daily_activities": True,
#         "relationship_dynamics": True
#     }
# )
