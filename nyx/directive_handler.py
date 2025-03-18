# nyx/directive_handler.py
# New file for standardized directive handling

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType

logger = logging.getLogger(__name__)

class DirectiveHandler:
    """
    Standard handler for processing governance directives.
    This class can be used by any agent to standardize directive processing.
    """
    
    def __init__(self, user_id: int, conversation_id: int, agent_type: str, agent_id: Union[int, str]):
        """
        Initialize the directive handler.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            agent_type: Type of agent (use AgentType constants)
            agent_id: ID of agent instance
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.handlers = {}
        self.last_check = None
        self.check_interval = 60  # seconds
    
    def register_handler(self, directive_type: str, handler: Callable):
        """
        Register a handler for a specific directive type.
        
        Args:
            directive_type: Type of directive to handle
            handler: Function to handle the directive
        """
        self.handlers[directive_type] = handler
    
    async def get_directives(self) -> List[Dict[str, Any]]:
        """
        Get all active directives for this agent.
        
        Returns:
            List of active directives
        """
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            
            if self.agent_type == AgentType.NPC:
                directives = await governance.get_npc_directives(int(self.agent_id))
            else:
                directives = await governance.get_agent_directives(self.agent_type, self.agent_id)
            
            return directives
        except Exception as e:
            logger.error(f"Error getting directives: {e}")
            return []
    
    async def process_directives(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Process all active directives for this agent.
        
        Args:
            force_check: Whether to force checking directives regardless of interval
            
        Returns:
            Dictionary with processing results
        """
        # Check if enough time has passed since last check
        now = datetime.now()
        if not force_check and self.last_check and (now - self.last_check).total_seconds() < self.check_interval:
            return {"processed": 0, "skipped_time_check": True}
        
        self.last_check = now
        
        # Get directives
        directives = await self.get_directives()
        
        if not directives:
            return {"processed": 0, "message": "No directives found"}
        
        # Process each directive
        results = []
        for directive in directives:
            directive_type = directive.get("type")
            directive_id = directive.get("id")
            
            # Skip if no handler for this directive type
            if directive_type not in self.handlers:
                results.append({
                    "directive_id": directive_id,
                    "processed": False,
                    "reason": f"No handler for directive type '{directive_type}'"
                })
                continue
            
            try:
                # Call the appropriate handler
                handler = self.handlers[directive_type]
                result = await handler(directive)
                
                results.append({
                    "directive_id": directive_id,
                    "processed": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing directive {directive_id}: {e}")
                results.append({
                    "directive_id": directive_id,
                    "processed": False,
                    "error": str(e)
                })
        
        return {
            "processed": len([r for r in results if r.get("processed", False)]),
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
                    logger.error(f"Error in background directive processing: {e}")
                
                await asyncio.sleep(interval)
        
        # Start the background task
        task = asyncio.create_task(background_task())
        return task

    def _should_check_directives(self) -> bool:
        """Determine if enough time has passed to check directives again"""
        if not self.last_check:
            return True
            
        time_since_check = (datetime.now() - self.last_check).total_seconds()
        return time_since_check >= self.check_interval
