# story_templates/moth/lore/__init__.py
"""
SF Bay Area Moth and Flame Lore Package

This package contains all lore elements for The Moth and Flame story
set in San Francisco Bay Area, organized into modular components.
"""

# Import main classes from each module for easy access
from .education import SFEducationLore
from .geopolitical import SFGeopoliticalLore
from .local_lore import SFLocalLore
from .politics import SFPoliticsLore
from .religion import SFReligionLore

# Import the main preset manager
from .world_lore_manager import (
    SFBayMothFlamePreset,
    EnhancedMothFlameInitializer
)

# Define what's available when someone does "from lore import *"
__all__ = [
    # Module classes
    'SFEducationLore',
    'SFGeopoliticalLore',
    'SFLocalLore',
    'SFPoliticsLore', 
    'SFReligionLore',
    
    # Main managers
    'SFBayMothFlamePreset',
    'EnhancedMothFlameInitializer',
]

# Version info
__version__ = '1.0.0'
__author__ = 'The Moth and Flame Story Team'
