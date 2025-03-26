# nyx/core/emotions/__init__.py

"""
Nyx Emotional Core - A modular, agent-based emotional simulation system.

This package implements a digital neurochemical model for emotional simulation
using the OpenAI Agents SDK.
"""

from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.core.emotions.hormone_system import HormoneSystem
from nyx.core.emotions.context import EmotionalContext

__all__ = ['EmotionalCore', 'HormoneSystem', 'EmotionalContext']
