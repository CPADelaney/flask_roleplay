# nyx/directive_handler.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# We no longer import get_central_governance here to avoid circular import
# from nyx.integrate import get_central_governance  <-- REMOVED

from nyx.nyx_governance import AgentType, DirectiveType

logger = logging.getLogger(__name__)


class DirectiveHandler:
    """
    Standard handler for processing governance directives.
    This class can be used by any agent to standardize directive processing.
    """

    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        agent_type: str,
        agent_id: Union[int, str],
        governance: Optional["NyxUnifiedGovernor"] = None,
    ):
        """
        Initialize the directive handler.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            agent_type: Type of agent (use AgentType constants)
            agent_id: ID of agent instance
            governance: The NyxUnifiedGovernor object (or similar) that provides
                        get_npc_directives or get_agent_directives. If not provided,
                        you must set it later or do a local import to fetch it.
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.handlers: Dict[str, Callable] = {}
        self.last_check = None
        self.check_interval = 60  # seconds

        self.governance = governance  # store the governance reference if provided

    def register_handler(self, directive_type: str, handler: Callable):
        """
        Register a handler for a specific directive type.

        Args:
            directive_type: Type of directive to handle
            handler: Function (coroutine) to handle the directive
        """
        self.handlers[directive_type] = handler

    async def get_directives(self) -> List[Dict[str, Any]]:
        """
        Get all active directives for this agent.
        Returns:
            List of active directives
        """
        # Lazy‐initialize governance if it wasn’t provided at construction
        if not self.governance:
            from nyx.integrate import get_central_governance
            self.governance = await get_central_governance(self.user_id, self.conversation_id)
    
        try:
            if self.agent_type == AgentType.NPC:
                return await self.governance.get_npc_directives(int(self.agent_id))
            else:
                return await self.governance.get_agent_directives(self.agent_type, self.agent_id)
        except Exception as e:
            logger.error(f"Error getting directives: {e}", exc_info=True)
            return []

    async def process_directives(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Process all active directives for this agent.

        Args:
            force_check: Whether to force checking directives regardless of interval

        Returns:
            Dictionary with processing results
        """
        now = datetime.now()
        if not force_check and self.last_check and (now - self.last_check).total_seconds() < self.check_interval:
            return {"processed": 0, "skipped_time_check": True}

        self.last_check = now

        # 1. Get directives
        directives = await self.get_directives()
        if not directives:
            return {"processed": 0, "message": "No directives found"}

        # 2. Process each directive
        results = []
        for directive in directives:
            directive_type = directive.get("type")
            directive_id = directive.get("id")

            # Skip if no handler
            if directive_type not in self.handlers:
                results.append({
                    "directive_id": directive_id,
                    "processed": False,
                    "reason": f"No handler for directive type '{directive_type}'"
                })
                continue

            try:
                handler = self.handlers[directive_type]
                result = await handler(directive)
                results.append({
                    "directive_id": directive_id,
                    "processed": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing directive {directive_id}: {e}", exc_info=True)
                results.append({
                    "directive_id": directive_id,
                    "processed": False,
                    "error": str(e)
                })

        return {
            "processed": len([r for r in results if r.get("processed")]),
            "total": len(results),
            "results": results
        }

    async def start_background_processing(self, interval: float = 60.0):
        """
        Start a background task to periodically process directives.

        Args:
            interval: Time between checks in seconds
        """
        self.check_interval = interval

        async def background_task():
            while True:
                try:
                    await self.process_directives()
                except Exception as e:
                    logger.error(f"Error in background directive processing: {e}", exc_info=True)

                await asyncio.sleep(interval)

        task = asyncio.create_task(background_task())
        return task

    def _should_check_directives(self) -> bool:
        """Determine if enough time has passed to check directives again"""
        if not self.last_check:
            return True

        time_since_check = (datetime.now() - self.last_check).total_seconds()
        return time_since_check >= self.check_interval
