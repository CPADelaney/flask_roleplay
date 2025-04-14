# memory/memory_agent_wrapper.py

from agents import Agent, Runner, function_tool
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class MemoryAgentWrapper:
    """
    Wrapper class that provides a compatibility layer between your existing memory system
    and the OpenAI Agents SDK.
    """
    
    def __init__(self, agent: Agent, context=None):
        self.agent = agent
        self.context = context
    
    async def recall(
        self,
        context,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context_text: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories for an entity by wrapping calls to the agent.
        """
        try:
            # Format a structured input for the agent
            input_message = {
                "role": "user",
                "content": f"Recall memories for {entity_type} {entity_id}",
                "metadata": {
                    "operation": "recall",
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "query": query,
                    "context": context_text,
                    "limit": limit
                }
            }
            
            # Run the agent with this input
            result = await Runner.run(
                self.agent,
                input=[input_message],
                context=context
            )
            
            # Extract and parse the result
            # Assuming the agent returns a structured output we can parse
            if isinstance(result.final_output, str):
                # If it returns text, try to parse it as JSON
                import json
                try:
                    return json.loads(result.final_output)
                except json.JSONDecodeError:
                    # If parsing fails, return a simple dict
                    return {"memories": [], "message": result.final_output}
            else:
                # If it's already structured data
                return result.final_output
        
        except Exception as e:
            logger.error(f"Error in recall: {str(e)}")
            return {"error": str(e), "memories": []}
    
    async def remember(
        self,
        context,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a memory for an entity by wrapping calls to the agent.
        """
        try:
            # Similar pattern to recall
            input_message = {
                "role": "user",
                "content": f"Create memory for {entity_type} {entity_id}: {memory_text[:50]}...",
                "metadata": {
                    "operation": "remember",
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "memory_text": memory_text,
                    "importance": importance,
                    "emotional": emotional,
                    "tags": tags or []
                }
            }
            
            result = await Runner.run(
                self.agent,
                input=[input_message],
                context=context
            )
            
            # Process result as in recall method
            if isinstance(result.final_output, str):
                import json
                try:
                    return json.loads(result.final_output)
                except json.JSONDecodeError:
                    return {"memory_id": None, "message": result.final_output}
            else:
                return result.final_output
        
        except Exception as e:
            logger.error(f"Error in remember: {str(e)}")
            return {"error": str(e), "memory_id": None}
