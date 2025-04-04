# nyx/creative/logging_utils.py

import datetime
import logging
from typing import Optional, Dict, Any
from nyx.creative.content_system import CreativeContentSystem, ContentType

logger = logging.getLogger(__name__)

class NyxLogger:
    def __init__(self, content_system: CreativeContentSystem):
        self.content_system = content_system

    async def log_thought(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an internal reflection, insight, or cognitive trace."""
        metadata = metadata or {}
        metadata["log_type"] = "thought"
        return await self.content_system.store_content(
            content_type=ContentType.JOURNAL,
            title=title,
            content=content,
            metadata=metadata
        )

    async def log_action(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an executed action and its reasoning."""
        metadata = metadata or {}
        metadata["log_type"] = "action"
        return await self.content_system.store_content(
            content_type=ContentType.ANALYSIS,
            title=title,
            content=content,
            metadata=metadata
        )

    async def log_evolution_suggestion(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a proposed self-improvement or capability match."""
        metadata = metadata or {}
        metadata["log_type"] = "evolution_suggestion"
        return await self.content_system.store_content(
            content_type=ContentType.ASSESSMENT,
            title=title,
            content=content,
            metadata=metadata
        )
