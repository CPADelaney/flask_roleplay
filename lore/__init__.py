# lore/__init__.py

"""
Lore Module

This module provides a comprehensive lore system for generating, 
evolving, and integrating lore into the game world.
"""

import warnings
import functools

# Main entry point for lore system - the consolidated implementation
from .lore_system import LoreSystem

# API routes
from .lore_routes import register_lore_routes

# Core components
from .lore_generator import DynamicLoreGenerator
from .integration import NPCLoreIntegration, ConflictIntegration, ContextEnhancer

# Error handling
from .error_manager import ErrorHandler, LoreError, handle_errors

# Utility components
from .data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess
from .validation import ValidationManager, ValidationResult
from .lore_directive_handler import LoreDirectiveHandler
from .lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

def _deprecated(func_or_class):
    """Mark a function or class as deprecated with a warning."""
    @functools.wraps(func_or_class)
    def wrapper(*args, **kwargs):
        name = func_or_class.__name__
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            f"Please use DynamicLoreGenerator instead. The consolidated system provides "
            f"all the functionality of the legacy components with improved "
            f"performance and better integration.",
            DeprecationWarning, 
            stacklevel=2
        )
        return func_or_class(*args, **kwargs)
    return wrapper

# Export the main interface
__all__ = [
    'LoreSystem',
    'register_lore_routes',
    'register_all_lore_modules_with_governance',
    'DynamicLoreGenerator',
    'NPCLoreIntegration',
    'ConflictIntegration',
    'ContextEnhancer',
    'ErrorHandler',
    'LoreError',
    'handle_errors',
    'NPCDataAccess',
    'LocationDataAccess',
    'FactionDataAccess',
    'LoreKnowledgeAccess',
    'ValidationManager',
    'ValidationResult',
    'LoreDirectiveHandler',
    'generate_foundation_lore',
    'generate_factions',
    'generate_cultural_elements',
    'generate_historical_events',
    'generate_locations',
    'generate_quest_hooks'
]
