# memory/memory_agent_wrapper.py

from agents import Agent, Runner, function_tool, RunContextWrapper
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
        # Copy required Agent attributes
        self.handoffs = agent.handoffs if hasattr(agent, 'handoffs') else []
        self.output_type = agent.output_type if hasattr(agent, 'output_type') else None
        self.name = agent.name if hasattr(agent, 'name') else "MemoryAgent"
        self.instructions = agent.instructions if hasattr(agent, 'instructions') else ""
        self.tools = agent.tools if hasattr(agent, 'tools') else []
        self.input_guardrails = []
        self._hooks = None
    
    # Fix: Add proper property getters
    @property
    def hooks(self):
        return getattr(self.agent, "hooks", None)

    @property
    def model(self):
        return getattr(self.agent, "model", None)
        
    
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
            
    async def create_belief(
            self,
            context,
            entity_type: str,
            entity_id: int,
            belief_text: str,
            confidence: float = 0.7
        ) -> Dict[str, Any]:
            """
            Create a belief for an entity by wrapping calls to the agent.
            """
            try:
                input_message = {
                    "role": "user",
                    "content": f"Create belief for {entity_type} {entity_id}: {belief_text[:50]}...",
                    "metadata": {
                        "operation": "create_belief",
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "belief_text": belief_text,
                        "confidence": confidence
                    }
                }
                
                result = await Runner.run(
                    self.agent,
                    input=[input_message],
                    context=context
                )
                
                # Process the result
                if isinstance(result.final_output, str):
                    import json
                    try:
                        return json.loads(result.final_output)
                    except json.JSONDecodeError:
                        return {"belief_id": None, "message": result.final_output}
                else:
                    return result.final_output
                    
            except Exception as e:
                logger.error(f"Error in create_belief: {str(e)}")
                return {"error": str(e), "belief_id": None}

    # Fix: Update to pass run_context parameter
    async def get_system_prompt(self, run_context: RunContextWrapper):
        """
        Return the system prompt for the memory agent.
        This method is needed by the recall method or its dependencies.
        """
        if hasattr(self.agent, 'get_system_prompt'):
            return await self.agent.get_system_prompt(run_context)
        elif hasattr(self.agent, 'instructions'):
            if callable(self.agent.instructions):
                return await self.agent.instructions(run_context, self.agent)
            return self.agent.instructions
        else:
            return "You are a memory management assistant that helps manage and retrieve memories."

    def get_tools(self):
        """
        Get the tools from the underlying agent.
        """
        if hasattr(self.agent, 'get_tools'):
            return self.agent.get_tools()
        else:
            return self.tools
    
    def run(self, *args, **kwargs):
        """
        Run the agent with the given input.
        """
        if hasattr(self.agent, 'run'):
            return self.agent.run(*args, **kwargs)
        else:
            # Fallback implementation or raise an error
            raise NotImplementedError("Run method not implemented by underlying agent")
    
    def get_name(self):
        """
        Get the name of the agent.
        """
        if hasattr(self.agent, 'get_name'):
            return self.agent.get_name()
        else:
            return self.name
    
    async def get_beliefs(
        self,
        context,
        entity_type: str,
        entity_id: int,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get beliefs for an entity by wrapping calls to the agent.
        """
        try:
            input_message = {
                "role": "user",
                "content": f"Get beliefs for {entity_type} {entity_id}" + 
                          (f" about topic: {topic}" if topic else ""),
                "metadata": {
                    "operation": "get_beliefs",
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "topic": topic
                }
            }
            
            result = await Runner.run(
                self.agent,
                input=[input_message],
                context=context
            )
            
            # Process the result
            if isinstance(result.final_output, str):
                import json
                try:
                    parsed = json.loads(result.final_output)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "beliefs" in parsed:
                        return parsed["beliefs"]
                    else:
                        return []
                except json.JSONDecodeError:
                    return []
            elif isinstance(result.final_output, list):
                return result.final_output
            elif isinstance(result.final_output, dict) and "beliefs" in result.final_output:
                return result.final_output["beliefs"]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in get_beliefs: {str(e)}")
            return []
    async def run_maintenance(
            self,
            context,
            entity_type: str,
            entity_id: int
        ) -> Dict[str, Any]:
            """
            Run maintenance on memories by wrapping calls to the agent.
            """
            try:
                input_message = {
                    "role": "user",
                    "content": f"Run memory maintenance for {entity_type} {entity_id}",
                    "metadata": {
                        "operation": "run_maintenance",
                        "entity_type": entity_type,
                        "entity_id": entity_id
                    }
                }
                
                result = await Runner.run(
                    self.agent,
                    input=[input_message],
                    context=context
                )
                
                # Process result
                if isinstance(result.final_output, str):
                    import json
                    try:
                        return json.loads(result.final_output)
                    except json.JSONDecodeError:
                        return {"success": True, "message": result.final_output}
                else:
                    return result.final_output
                    
            except Exception as e:
                logger.error(f"Error in run_maintenance: {str(e)}")
                return {"error": str(e), "success": False}
    
    async def analyze_memories(
        self,
        context,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Analyze memories for an entity by wrapping calls to the agent.
        """
        try:
            input_message = {
                "role": "user",
                "content": f"Analyze memories for {entity_type} {entity_id}",
                "metadata": {
                    "operation": "analyze_memories",
                    "entity_type": entity_type,
                    "entity_id": entity_id
                }
            }
            
            result = await Runner.run(
                self.agent,
                input=[input_message],
                context=context
            )
            
            # Process result
            if isinstance(result.final_output, str):
                import json
                try:
                    return json.loads(result.final_output)
                except json.JSONDecodeError:
                    return {"analysis": result.final_output}
            else:
                return result.final_output
                
        except Exception as e:
            logger.error(f"Error in analyze_memories: {str(e)}")
            return {"error": str(e), "analysis": None}
    
    async def generate_schemas(
        self,
        context,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Generate memory schemas by wrapping calls to the agent.
        """
        try:
            input_message = {
                "role": "user",
                "content": f"Generate memory schemas for {entity_type} {entity_id}",
                "metadata": {
                    "operation": "generate_schemas",
                    "entity_type": entity_type,
                    "entity_id": entity_id
                }
            }
            
            result = await Runner.run(
                self.agent,
                input=[input_message],
                context=context
            )
            
            # Process result
            if isinstance(result.final_output, str):
                import json
                try:
                    return json.loads(result.final_output)
                except json.JSONDecodeError:
                    return {"schemas": [], "message": result.final_output}
            else:
                return result.final_output
                
        except Exception as e:
            logger.error(f"Error in generate_schemas: {str(e)}")
            return {"error": str(e), "schemas": []}
