# nyx/nyx_governance.py
"""
Legacy import support - maintains backward compatibility.
This file ensures existing imports continue to work.

All existing code that imports from nyx.nyx_governance will continue to work:
    from nyx.nyx_governance import NyxUnifiedGovernor, DirectiveType, AgentType
"""
from nyx.governance import (
    NyxUnifiedGovernor,
    DirectiveType,
    DirectivePriority,
    AgentType
)

# Re-export everything to maintain backward compatibility
__all__ = [
    'NyxUnifiedGovernor',
    'DirectiveType', 
    'DirectivePriority',
    'AgentType'
]
