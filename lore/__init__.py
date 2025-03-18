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
from lore.lore_api_routes import register_lore_api_routes

# Core components
from .lore_manager import LoreManager
from .npc_lore_integration import NPCLoreIntegration
from .dynamic_lore_generator import DynamicLoreGenerator
from .governance_registration import register_all_lore_modules_with_governance

from .generators import ComponentGeneratorFactory, ComponentConfig
from .resource_manager import ResourceManager, ResourceConfig
from .error_handler import ErrorHandler

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
    'register_lore_api_routes',
    'register_all_lore_modules_with_governance',
    'LoreManager',
    'NPCLoreIntegration',
    'DynamicLoreGenerator',
    'ComponentGeneratorFactory',
    'ComponentConfig',
    'ResourceManager',
    'ResourceConfig',
    'ErrorHandler'
] 