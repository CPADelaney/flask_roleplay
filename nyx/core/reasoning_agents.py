# nyx/core/reasoning_agents.py
import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from agents import Agent, Runner, function_tool, handoff, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel, Field

# Import your existing reasoning core
from nyx.core.reasoning_core import (
    ReasoningCore, CausalModel, CausalNode, CausalRelation,
    ConceptSpace, ConceptualBlend, Intervention
)

# Initialize the reasoning core
reasoning_core = ReasoningCore()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- Pydantic Models for Tool Inputs/Outputs ---------------------

class CausalModelInput(BaseModel):
    name: str = Field(..., description="Name of the causal model")
    domain: str = Field(..., description="Domain of the causal model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the model")

class CausalNodeInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    name: str = Field(..., description="Name of the node")
    domain: str = Field(None, description="Domain of the node (optional)")
    node_type: str = Field("variable", description="Type of node (variable, event, action, state, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the node")

class CausalRelationInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    relation_type: str = Field("causal", description="Type of relation (causal, correlation, etc.)")
    strength: float = Field(0.5, description="Strength of the relation (0.0 to 1.0)")
    mechanism: str = Field("", description="Description of the causal mechanism")

class InterventionInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    target_node: str = Field(..., description="ID of the target node")
    target_value: Any = Field(..., description="Target value for the intervention")
    name: str = Field(..., description="Name of the intervention")
    description: str = Field("", description="Description of the intervention")

class CounterfactualInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    factual_values: Dict[str, Any] = Field({}, description="Current/factual values")
    counterfactual_values: Dict[str, Any] = Field(..., description="Counterfactual values to reason about")
    target_nodes: List[str] = Field([], description="Target nodes to analyze (optional)")

class ConceptSpaceInput(BaseModel):
    name: str = Field(..., description="Name of the concept space")
    domain: str = Field("", description="Domain of the concept space")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class ConceptInput(BaseModel):
    space_id: str = Field(..., description="ID of the concept space")
    name: str = Field(..., description="Name of the concept")
    properties: Dict[str, Any] = Field({}, description="Properties of the concept")

class ConceptRelationInput(BaseModel):
    space_id: str = Field(..., description="ID of the concept space")
    source_id: str = Field(..., description="ID of the source concept")
    target_id: str = Field(..., description="ID of the target concept")
    relation_type: str = Field(..., description="Type of relation")
    strength: float = Field(1.0, description="Strength of the relation")

class CreativeInterventionInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    target_node: str = Field(..., description="ID of the target node")
    description: str = Field("", description="Description of the intervention")
    use_blending: bool = Field(True, description="Whether to use conceptual blending")

# --------------------- Causal Reasoning Agent ---------------------

# Create tools for the causal reasoning agent
@function_tool
async def create_causal_model(input_data: CausalModelInput) -> str:
    """Create a new causal model."""
    model_id = await reasoning_core.create_causal_model(
        name=input_data.name,
        domain=input_data.domain,
        metadata=input_data.metadata
    )
    return f"Created causal model with ID: {model_id}"

@function_tool
async def add_node_to_model(input_data: CausalNodeInput) -> str:
    """Add a node to a causal model."""
    node_id = await reasoning_core.add_node_to_model(
        model_id=input_data.model_id,
        name=input_data.name,
        domain=input_data.domain or "",
        node_type=input_data.node_type,
        metadata=input_data.metadata
    )
    return f"Added node with ID: {node_id} to model: {input_data.model_id}"

@function_tool
async def add_relation_to_model(input_data: CausalRelationInput) -> str:
    """Add a causal relation to a model."""
    relation_id = await reasoning_core.add_relation_to_model(
        model_id=input_data.model_id,
        source_id=input_data.source_id,
        target_id=input_data.target_id,
        relation_type=input_data.relation_type,
        strength=input_data.strength,
        mechanism=input_data.mechanism
    )
    return f"Added relation with ID: {relation_id} to model: {input_data.model_id}"

@function_tool
async def get_causal_model(model_id: str) -> Dict[str, Any]:
    """Get a causal model by ID."""
    model = await reasoning_core.get_causal_model(model_id)
    return model

@function_tool
async def define_intervention(input_data: InterventionInput) -> str:
    """Define an intervention on a causal model."""
    intervention_id = await reasoning_core.define_intervention(
        model_id=input_data.model_id,
        target_node=input_data.target_node,
        target_value=input_data.target_value,
        name=input_data.name,
        description=input_data.description
    )
    return f"Created intervention with ID: {intervention_id}"

@function_tool
async def reason_counterfactually(input_data: CounterfactualInput) -> Dict[str, Any]:
    """Perform counterfactual reasoning using a causal model."""
    result = await reasoning_core.reason_counterfactually(
        model_id=input_data.model_id,
        query={
            "factual_values": input_data.factual_values,
            "counterfactual_values": input_data.counterfactual_values,
            "target_nodes": input_data.target_nodes
        }
    )
    return result

@function_tool
async def discover_causal_relations(model_id: str) -> Dict[str, Any]:
    """Discover causal relations in a model automatically."""
    result = await reasoning_core.discover_causal_relations(model_id)
    return result

# Define the Causal Reasoning Agent
causal_reasoning_agent = Agent(
    name="Causal Reasoning Agent",
    instructions="""You are a causal reasoning specialist agent. You help users build and work with causal models.
    
Your capabilities include:
- Creating causal models to represent variables and their relationships
- Adding nodes and relations to models
- Defining and analyzing interventions
- Performing counterfactual reasoning
- Automatic discovery of causal relations from data

Always explain your reasoning and how your causal analyses relate to the user's goals.
When working with causal models, help the user understand the implications of different causal structures.
    """,
    tools=[
        create_causal_model,
        add_node_to_model,
        add_relation_to_model,
        get_causal_model,
        define_intervention,
        reason_counterfactually,
        discover_causal_relations
    ]
)

# --------------------- Conceptual Reasoning Agent ---------------------

# Create tools for the conceptual reasoning agent
@function_tool
async def create_concept_space(input_data: ConceptSpaceInput) -> str:
    """Create a new conceptual space."""
    space_id = await reasoning_core.create_concept_space(
        name=input_data.name,
        domain=input_data.domain,
        metadata=input_data.metadata
    )
    return f"Created concept space with ID: {space_id}"

@function_tool
async def add_concept_to_space(input_data: ConceptInput) -> str:
    """Add a concept to a conceptual space."""
    concept_id = await reasoning_core.add_concept_to_space(
        space_id=input_data.space_id,
        name=input_data.name,
        properties=input_data.properties
    )
    return f"Added concept with ID: {concept_id} to space: {input_data.space_id}"

@function_tool
async def add_relation_to_space(input_data: ConceptRelationInput) -> str:
    """Add a relation between concepts in a space."""
    await reasoning_core.add_relation_to_space(
        space_id=input_data.space_id,
        source_id=input_data.source_id,
        target_id=input_data.target_id,
        relation_type=input_data.relation_type,
        strength=input_data.strength
    )
    return f"Added relation from {input_data.source_id} to {input_data.target_id} in space: {input_data.space_id}"

@function_tool
async def get_concept_space(space_id: str) -> Dict[str, Any]:
    """Get a concept space by ID."""
    space = await reasoning_core.get_concept_space(space_id)
    return space

class BlendInput(BaseModel):
    space_id_1: str = Field(..., description="ID of the first concept space")
    space_id_2: str = Field(..., description="ID of the second concept space")
    blend_type: str = Field("composition", description="Type of blend (composition, fusion, completion, elaboration, contrast)")

@function_tool
async def create_blend(input_data: BlendInput) -> Dict[str, Any]:
    """Create a blend between two conceptual spaces."""
    # This is a simplified version - in the actual implementation,
    # you would need to find mappings and call the appropriate blend function
    
    # For demonstration, we'll just pretend to create a blend
    space1 = await reasoning_core.get_concept_space(input_data.space_id_1)
    space2 = await reasoning_core.get_concept_space(input_data.space_id_2)
    
    # Find mappings between spaces (simplified)
    mappings = []
    blend_id = f"blend_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "blend_id": blend_id,
        "blend_type": input_data.blend_type,
        "input_spaces": [input_data.space_id_1, input_data.space_id_2],
        "status": "created"
    }

# Define the Conceptual Reasoning Agent
conceptual_reasoning_agent = Agent(
    name="Conceptual Reasoning Agent",
    instructions="""You are a conceptual reasoning specialist agent. You help users work with conceptual spaces and blending.
    
Your capabilities include:
- Creating conceptual spaces to represent concepts and their properties
- Adding concepts and relations between them
- Creating conceptual blends between different spaces
- Analyzing conceptual structures and mappings

Always explain your reasoning and help users understand how conceptual structures can lead to creative insights.
When working with conceptual blends, highlight emergent structures and novel properties.
    """,
    tools=[
        create_concept_space,
        add_concept_to_space,
        add_relation_to_space,
        get_concept_space,
        create_blend
    ]
)

# --------------------- Integrated Reasoning Agent ---------------------

# Create tools for the integrated reasoning agent
@function_tool
async def convert_blend_to_causal_model(blend_id: str, name: Optional[str] = None, domain: Optional[str] = None) -> str:
    """Convert a conceptual blend to a causal model."""
    model_id = await reasoning_core.convert_blend_to_causal_model(
        blend_id=blend_id,
        name=name,
        domain=domain
    )
    return f"Converted blend {blend_id} to causal model {model_id}"

@function_tool
async def convert_causal_model_to_concept_space(model_id: str, name: Optional[str] = None, domain: Optional[str] = None) -> str:
    """Convert a causal model to a conceptual space."""
    space_id = await reasoning_core.convert_causal_model_to_concept_space(
        model_id=model_id,
        name=name,
        domain=domain
    )
    return f"Converted causal model {model_id} to concept space {space_id}"

@function_tool
async def create_creative_intervention(input_data: CreativeInterventionInput) -> Dict[str, Any]:
    """Create a creative intervention using conceptual blending and causal reasoning."""
    result = await reasoning_core.create_creative_intervention(
        model_id=input_data.model_id,
        target_node=input_data.target_node,
        description=input_data.description,
        use_blending=input_data.use_blending
    )
    return result

@function_tool
async def create_integrated_model(domain: str, base_on_causal: bool = True) -> Dict[str, Any]:
    """Create an integrated model with both causal and conceptual reasoning."""
    result = await reasoning_core.create_integrated_model(
        domain=domain,
        base_on_causal=base_on_causal
    )
    return result

@function_tool
async def get_reasoning_stats() -> Dict[str, Any]:
    """Get statistics about the integrated reasoning system."""
    stats = await reasoning_core.get_stats()
    return stats

# Define the Integrated Reasoning Agent
integrated_reasoning_agent = Agent(
    name="Integrated Reasoning Agent",
    instructions="""You are an integrated reasoning specialist agent that combines causal and conceptual reasoning approaches.
    
Your capabilities include:
- Converting between causal models and conceptual spaces
- Creating creative interventions in causal models using conceptual blending
- Building integrated models that leverage both reasoning approaches
- Analyzing complex systems using complementary reasoning strategies

When helping users, look for opportunities to combine causal and conceptual approaches for deeper insights.
Always explain how the integration of different reasoning approaches enhances understanding.
    """,
    tools=[
        convert_blend_to_causal_model,
        convert_causal_model_to_concept_space,
        create_creative_intervention,
        create_integrated_model,
        get_reasoning_stats
    ],
    handoffs=[
        handoff(causal_reasoning_agent, 
                tool_description="Transfer to the causal reasoning agent for detailed causal modeling and analysis"),
        handoff(conceptual_reasoning_agent, 
                tool_description="Transfer to the conceptual reasoning agent for conceptual space creation and blending")
    ]
)

# --------------------- Triage Agent ---------------------

# Define a HomeworkCheck class for guardrails
class HomeworkCheck(BaseModel):
    is_homework: bool = Field(False, description="Whether the query is asking for homework help")
    reasoning: str = Field("", description="Reasoning for the determination")

# Create a homework detection agent
homework_detection_agent = Agent(
    name="Homework Check",
    instructions="Check if the user is asking for help with homework or academic assignments.",
    output_type=HomeworkCheck
)

# Create a guardrail function
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(homework_detection_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkCheck)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_homework
    )

# Define the Triage Agent that directs requests to appropriate specialists
triage_agent = Agent(
    name="Reasoning Triage Agent",
    instructions="""You are the entry point for a reasoning system that helps users work with causal models and conceptual spaces.

Based on the user's request, you should determine which specialist agent would be most appropriate:

1. Causal Reasoning Agent - For questions about cause and effect, interventions, counterfactuals, and structural models
2. Conceptual Reasoning Agent - For questions about concepts, conceptual spaces, mappings, and blending
3. Integrated Reasoning Agent - For complex problems that benefit from both causal and conceptual approaches

Always help the user understand which type of reasoning would be most beneficial for their problem and why.
""",
    handoffs=[
        handoff(causal_reasoning_agent, 
                tool_description="Transfer to the causal reasoning specialist for causal modeling and analysis"),
        handoff(conceptual_reasoning_agent, 
                tool_description="Transfer to the conceptual reasoning specialist for conceptual space creation and blending"),
        handoff(integrated_reasoning_agent, 
                tool_description="Transfer to the integrated reasoning specialist that combines causal and conceptual approaches")
    ],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail)
    ]
)

# --------------------- Main Entry Point ---------------------

async def process_reasoning_request(user_input: str):
    """Process a user request through the reasoning agents system."""
    result = await Runner.run(triage_agent, user_input)
    return result.final_output

# Run synchronously for simple use cases
def process_reasoning_request_sync(user_input: str):
    """Process a user request synchronously."""
    return asyncio.run(process_reasoning_request(user_input))

# Example usage
if __name__ == "__main__":
    user_query = "I want to build a causal model to understand climate change impacts on agriculture."
    response = process_reasoning_request_sync(user_query)
    print(response)
