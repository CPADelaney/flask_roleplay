# config/pipeline_config.py:

import os
from typing import Optional

class PipelineConfig:
    """Central configuration for the unified pipeline"""
    
    # Pipeline settings
    USE_NYX_INTEGRATION = os.getenv("USE_NYX_INTEGRATION", "true").lower() == "true"
    USE_UNIVERSAL_UPDATER = os.getenv("USE_UNIVERSAL_UPDATER", "true").lower() == "true"
    USE_REFLECTION = os.getenv("USE_REFLECTION", "false").lower() == "true"
    
    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5-nano")
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the appropriate system prompt"""
        from logic.prompts import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    
    @classmethod
    def get_private_reflection(cls) -> str:
        """Get private reflection instructions"""
        from logic.prompts import PRIVATE_REFLECTION_INSTRUCTIONS
        return PRIVATE_REFLECTION_INSTRUCTIONS if cls.USE_REFLECTION else ""
