"""
Lore Module

This module provides a comprehensive lore system for generating, 
evolving, and integrating lore into the game world.
"""

import warnings
import functools

# Main entry point for lore system - the consolidated implementation
from lore.lore_system import LoreSystem

# API routes
from lore.lore_api_routes import register_lore_api_routes

# Legacy components (DEPRECATED)
# These are kept for backward compatibility but will be removed in a future release
# All new code should use LoreSystem instead
from lore.lore_manager import LoreManager
from lore.lore_integration import LoreIntegrationSystem
from lore.npc_lore_integration import NPCLoreIntegration
from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.enhanced_lore_consolidated import EnhancedLoreSystem
from lore.governance_registration import register_all_lore_modules_with_governance

def _deprecated(func_or_class):
    """Mark a function or class as deprecated with a warning."""
    @functools.wraps(func_or_class)
    def wrapper(*args, **kwargs):
        name = func_or_class.__name__
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            f"Please use LoreSystem instead.",
            DeprecationWarning, 
            stacklevel=2
        )
        return func_or_class(*args, **kwargs)
    return wrapper

# Apply deprecation warnings to legacy classes
LoreManager = _deprecated(LoreManager)
LoreIntegrationSystem = _deprecated(LoreIntegrationSystem)
NPCLoreIntegration = _deprecated(NPCLoreIntegration)
DynamicLoreGenerator = _deprecated(DynamicLoreGenerator)
EnhancedLoreSystem = _deprecated(EnhancedLoreSystem)
register_all_lore_modules_with_governance = _deprecated(register_all_lore_modules_with_governance)

__all__ = [
    # Main component (recommended)
    'LoreSystem',
    
    # API routes
    'register_lore_api_routes',
    
    # Legacy components (deprecated)
    'LoreManager',
    'LoreIntegrationSystem',
    'NPCLoreIntegration',
    'DynamicLoreGenerator',
    'EnhancedLoreSystem',
    'register_all_lore_modules_with_governance'
] 