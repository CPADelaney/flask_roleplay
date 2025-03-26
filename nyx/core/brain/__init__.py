# nyx/core/brain/__init__.py
from nyx.core.brain.base import NyxBrain
from nyx.core.brain.models import (
    UserInput, ProcessResult, ResponseResult, AdaptationResult,
    IdentityState, StimulusData, ReflexRegistrationInput, ReflexResponse
)

# Provide convenient imports, but don't expose all internals
__all__ = [
    "NyxBrain",
    "UserInput", "ProcessResult", "ResponseResult", "AdaptationResult",
    "IdentityState", "StimulusData", "ReflexRegistrationInput", "ReflexResponse"
]
