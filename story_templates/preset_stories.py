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
    
    # Optional fields for advanced stories
    source_material: Optional[Dict[str, Any]] = None  # Poems, tone prompts, etc.
    special_mechanics: Optional[Dict[str, Any]] = None  # Story-specific mechanics
