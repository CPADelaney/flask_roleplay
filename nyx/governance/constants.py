# nyx/governance/constants.py
"""
Constants used across the governance system.
"""

class DirectiveType:
    """Constants for directive types."""
    ACTION = "action"
    MOVEMENT = "movement"
    DIALOGUE = "dialogue"
    RELATIONSHIP = "relationship"
    EMOTION = "emotion"
    PROHIBITION = "prohibition"
    SCENE = "scene"
    OVERRIDE = "override"
    INFORMATION = "information"


class DirectivePriority:
    """Constants for directive priorities."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class AgentType:
    """Constants for agent types."""
    NPC = "npc"
    WORLD_DIRECTOR = "world_director"
    # Deprecated alias for backward compatibility with linear story systems
    STORY_DIRECTOR = WORLD_DIRECTOR
    CONFLICT_ANALYST = "conflict_analyst"
    NARRATIVE_CRAFTER = "narrative_crafter"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    RELATIONSHIP_MANAGER = "relationship_manager"
    ACTIVITY_ANALYZER = "activity_analyzer"
    SCENE_MANAGER = "scene_manager"
    UNIVERSAL_UPDATER = "universal_updater"
    MEMORY_MANAGER = "memory_manager"
    
    # Event system agent types
    EVENT_MANAGER = "event_manager"
    EVENT_ANALYZER = "event_analyzer"
    EVENT_PROPAGATOR = "event_propagator"
