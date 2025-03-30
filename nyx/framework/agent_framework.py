# nyx/framework/agent_framework.py

import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Type, TypeVar
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, ModelSettings, function_tool, handoff, trace,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    RunConfig, RunContextWrapper
)

T = TypeVar('T')

class NyxAgentFramework:
    """Base framework for Nyx agent modules."""
    
    def __init__(self, name: str):
        self.name = name
        self.trace_group_id = f"nyx_{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.main_agent = None
        self.sub_agents = {}
        
    async def initialize(self) -> bool:
        """Initialize the agent system."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """Execute the agent with provided input."""
        if not self.main_agent:
            raise ValueError(f"{self.name} main agent not initialized")
            
        with trace(workflow_name=f"{self.name}_Execution", group_id=self.trace_group_id):
            result = await Runner.run(
                self.main_agent,
                input_data,
                context=context,
                run_config=RunConfig(
                    workflow_name=f"{self.name}_Workflow",
                    trace_id=f"trace_{datetime.datetime.now().timestamp()}",
                    group_id=self.trace_group_id
                )
            )
            
            return result.final_output
            
    async def create_agent(
            self, 
            name: str, 
            instructions: str, 
            tools: List[Any] = None, 
            handoffs: List[Any] = None,
            input_guardrails: List[Any] = None,
            output_guardrails: List[Any] = None,
            output_type: Optional[Type[T]] = None,
            model: str = "gpt-4o",
            temp: float = 0.2
        ) -> Agent:
        """Helper method to create an agent with standard configuration."""
        return Agent(
            name=name,
            instructions=instructions,
            tools=tools or [],
            handoffs=handoffs or [],
            input_guardrails=input_guardrails or [],
            output_guardrails=output_guardrails or [],
            output_type=output_type,
            model=model,
            model_settings=ModelSettings(temperature=temp)
        )
        
    async def create_safety_guardrail(self, validation_schema: Type[BaseModel]) -> InputGuardrail:
        """Create a standard safety guardrail for inputs."""
        async def safety_check(ctx, agent, input_data):
            try:
                # Validate input against schema
                if isinstance(input_data, str):
                    # Parse JSON if string
                    import json
                    data = json.loads(input_data)
                else:
                    data = input_data
                    
                validated = validation_schema.model_validate(data)
                return GuardrailFunctionOutput(
                    output_info={"is_safe": True, "validated": validated.model_dump()},
                    tripwire_triggered=False
                )
            except Exception as e:
                return GuardrailFunctionOutput(
                    output_info={"is_safe": False, "error": str(e)},
                    tripwire_triggered=True
                )
                
        return InputGuardrail(guardrail_function=safety_check)
