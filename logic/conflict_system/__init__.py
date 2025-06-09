# logic/conflict_system/__init__.py
"""
Unified Conflict System for the RPG

This system manages character-driven conflicts with multiple stakeholders,
resolution paths, and dynamic evolution.
"""

from .core import ConflictCore
from .generation import ConflictGenerator
from .evolution import ConflictEvolution
from .resolution import ConflictResolver
from .stakeholders import StakeholderManager
from .integration import ConflictSystemIntegration
from .hooks import ConflictHooks

__all__ = [
    'ConflictCore',
    'ConflictGenerator', 
    'ConflictEvolution',
    'ConflictResolver',
    'StakeholderManager',
    'ConflictSystemIntegration',
    'ConflictHooks'
]
