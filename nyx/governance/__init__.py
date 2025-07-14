# nyx/governance/__init__.py
"""
Governance module initialization that maintains backward compatibility.
"""
from .core import NyxUnifiedGovernor
from .constants import DirectiveType, DirectivePriority, AgentType

# Maintain backward compatibility
__all__ = ['NyxUnifiedGovernor', 'DirectiveType', 'DirectivePriority', 'AgentType']
