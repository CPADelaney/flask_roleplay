# nyx/core/reasoning_agents.py
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from agents import (
    Agent, Runner, function_tool, handoff, InputGuardrail, GuardrailFunctionOutput, 
    RunContextWrapper, trace
)
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

# --------------------- Context and Type Definitions ---------------------

class CausalModelDump(BaseModel):
    model_id: str
    model_json: str
    model_config = {"extra": "forbid"}

class CounterfactualResult(BaseModel):
    model_id: str
    result_json: str
    model_config = {"extra": "forbid"}

class RelationsDiscoveryResult(BaseModel):
    model_id: str
    relations_json: str
    model_config = {"extra": "forbid"}

class ConceptSpaceDump(BaseModel):
    space_id: str
    space_json: str
    model_config = {"extra": "forbid"}

class BlendInfo(BaseModel):
    blend_id: str
    blend_json: str
    model_config = {"extra": "forbid"}

class CreativeInterventionResult(BaseModel):
    intervention_json: str
    model_config = {"extra": "forbid"}

class IntegratedModelResult(BaseModel):
    integrated_json: str
    model_config = {"extra": "forbid"}

class ReasoningStats(BaseModel):
    stats_json: str
    model_config = {"extra": "forbid"}

class ReasoningContext(BaseModel):
    """Context for all reasoning agents"""
    knowledge_core: Optional[Any] = None
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    
    # Tracking metrics
    total_calls: int = 0
    handoffs: int = 0
    
    # Additional context during execution
    current_domain: Optional[str] = None
    active_model_id: Optional[str] = None
    active_space_id: Optional[str] = None
    
    # History for multi-turn conversations
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

# --------------------- Pydantic Models for Tool Inputs/Outputs ---------------------

class CausalModelInput(BaseModel):
    name: str = Field(..., description="Name of the causal model")
    domain: str = Field(..., description="Domain of the causal model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the model")

class CausalModelOutput(BaseModel):
    model_id: str = Field(..., description="ID of the created causal model")
    message: str = Field(..., description="Status message")

class CausalNodeInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    name: str = Field(..., description="Name of the node")
    domain: Optional[str] = Field(None, description="Domain of the node (optional)")
    node_type: str = Field("variable", description="Type of node (variable, event, action, state, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the node")

class CausalNodeOutput(BaseModel):
    node_id: str = Field(..., description="ID of the created node")
    message: str = Field(..., description="Status message")

class CausalRelationInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    relation_type: str = Field("causal", description="Type of relation (causal, correlation, etc.)")
    strength: float = Field(0.5, description="Strength of the relation (0.0 to 1.0)")
    mechanism: str = Field("", description="Description of the causal mechanism")

class CausalRelationOutput(BaseModel):
    relation_id: str = Field(..., description="ID of the created relation")
    message: str = Field(..., description="Status message")

class InterventionInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    target_node: str = Field(..., description="ID of the target node")
    target_value: Any = Field(..., description="Target value for the intervention")
    name: str = Field(..., description="Name of the intervention")
    description: str = Field("", description="Description of the intervention")

class InterventionOutput(BaseModel):
    intervention_id: str = Field(..., description="ID of the created intervention")
    message: str = Field(..., description="Status message")

class CounterfactualInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    factual_values: Dict[str, Any] = Field({}, description="Current/factual values")
    counterfactual_values: Dict[str, Any] = Field(..., description="Counterfactual values to reason about")
    target_nodes: List[str] = Field([], description="Target nodes to analyze (optional)")

class ConceptSpaceInput(BaseModel):
    name: str = Field(..., description="Name of the concept space")
    domain: str = Field("", description="Domain of the concept space")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class ConceptSpaceOutput(BaseModel):
    space_id: str = Field(..., description="ID of the created concept space")
    message: str = Field(..., description="Status message")

class ConceptInput(BaseModel):
    space_id: str = Field(..., description="ID of the concept space")
    name: str = Field(..., description="Name of the concept")
    properties: Dict[str, Any] = Field({}, description="Properties of the concept")

class ConceptOutput(BaseModel):
    concept_id: str = Field(..., description="ID of the created concept")
    message: str = Field(..., description="Status message")

class ConceptRelationInput(BaseModel):
    space_id: str = Field(..., description="ID of the concept space")
    source_id: str = Field(..., description="ID of the source concept")
    target_id: str = Field(..., description="ID of the target concept")
    relation_type: str = Field(..., description="Type of relation")
    strength: float = Field(1.0, description="Strength of the relation")

class BlendInput(BaseModel):
    space_id_1: str = Field(..., description="ID of the first concept space")
    space_id_2: str = Field(..., description="ID of the second concept space")
    blend_type: str = Field("composition", description="Type of blend (composition, fusion, completion, elaboration, contrast)")

class CreativeInterventionInput(BaseModel):
    model_id: str = Field(..., description="ID of the causal model")
    target_node: str = Field(..., description="ID of the target node")
    description: str = Field("", description="Description of the intervention")
    use_blending: bool = Field(True, description="Whether to use conceptual blending")

class IntegratedModelInput(BaseModel):
    domain: str = Field(..., description="Domain for the integrated model")
    base_on_causal: bool = Field(True, description="Whether to base the integration on causal models")

# Homework check for guardrails
class HomeworkCheck(BaseModel):
    is_homework: bool = Field(False, description="Whether the query is asking for homework help")
    reasoning: str = Field("", description="Reasoning for the determination")

# --------------------- Causal Reasoning Agent Tools ---------------------

@function_tool
async def create_causal_model(
    ctx: RunContextWrapper[ReasoningContext], 
    input_data: CausalModelInput
) -> CausalModelOutput:
    """Create a new causal model."""
    # Update context stats
    ctx.context.total_calls += 1
    
    model_id = await reasoning_core.create_causal_model(
        name=input_data.name,
        domain=input_data.domain,
        metadata=input_data.metadata
    )
    
    # Set active model in context
    ctx.context.active_model_id = model_id
    ctx.context.current_domain = input_data.domain
    
    return CausalModelOutput(
        model_id=model_id,
        message=f"Created causal model with ID: {model_id}"
    )

@function_tool
async def add_node_to_model(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: CausalNodeInput
) -> CausalNodeOutput:
    """Add a node to a causal model."""
    # Update context stats
    ctx.context.total_calls += 1
    
    node_id = await reasoning_core.add_node_to_model(
        model_id=input_data.model_id,
        name=input_data.name,
        domain=input_data.domain or "",
        node_type=input_data.node_type,
        metadata=input_data.metadata
    )
    
    return CausalNodeOutput(
        node_id=node_id,
        message=f"Added node with ID: {node_id} to model: {input_data.model_id}"
    )

@function_tool
async def add_relation_to_model(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: CausalRelationInput
) -> CausalRelationOutput:
    """Add a causal relation to a model."""
    # Update context stats
    ctx.context.total_calls += 1
    
    relation_id = await reasoning_core.add_relation_to_model(
        model_id=input_data.model_id,
        source_id=input_data.source_id,
        target_id=input_data.target_id,
        relation_type=input_data.relation_type,
        strength=input_data.strength,
        mechanism=input_data.mechanism
    )
    
    return CausalRelationOutput(
        relation_id=relation_id,
        message=f"Added relation with ID: {relation_id} to model: {input_data.model_id}"
    )

@function_tool
async def get_causal_model(
    ctx: RunContextWrapper[ReasoningContext],
    model_id: str,
) -> CausalModelDump:
    ctx.context.total_calls += 1
    mdl = await reasoning_core.get_causal_model(model_id)
    return CausalModelDump(
        model_id=model_id,
        model_json=json.dumps(mdl, separators=(",", ":")),
    )

@function_tool
async def define_intervention(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: InterventionInput
) -> InterventionOutput:
    """Define an intervention on a causal model."""
    # Update context stats
    ctx.context.total_calls += 1
    
    intervention_id = await reasoning_core.define_intervention(
        model_id=input_data.model_id,
        target_node=input_data.target_node,
        target_value=input_data.target_value,
        name=input_data.name,
        description=input_data.description
    )
    
    return InterventionOutput(
        intervention_id=intervention_id,
        message=f"Created intervention with ID: {intervention_id}"
    )

@function_tool
async def reason_counterfactually(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: CounterfactualInput,
) -> CounterfactualResult:
    ctx.context.total_calls += 1
    res = await reasoning_core.reason_counterfactually(
        model_id=input_data.model_id,
        query={
            "factual_values": input_data.factual_values,
            "counterfactual_values": input_data.counterfactual_values,
            "target_nodes": input_data.target_nodes,
        },
    )
    return CounterfactualResult(
        model_id=input_data.model_id,
        result_json=json.dumps(res, separators=(",", ":")),
    )

@function_tool
async def discover_causal_relations(
    ctx: RunContextWrapper[ReasoningContext],
    model_id: str,
) -> RelationsDiscoveryResult:
    ctx.context.total_calls += 1
    rels = await reasoning_core.discover_causal_relations(model_id)
    return RelationsDiscoveryResult(
        model_id=model_id,
        relations_json=json.dumps(rels, separators=(",", ":")),
    )


# --------------------- Conceptual Reasoning Agent Tools ---------------------

@function_tool
async def create_concept_space(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: ConceptSpaceInput
) -> ConceptSpaceOutput:
    """Create a new conceptual space."""
    # Update context stats
    ctx.context.total_calls += 1
    
    space_id = await reasoning_core.create_concept_space(
        name=input_data.name,
        domain=input_data.domain,
        metadata=input_data.metadata
    )
    
    # Set active space in context
    ctx.context.active_space_id = space_id
    ctx.context.current_domain = input_data.domain
    
    return ConceptSpaceOutput(
        space_id=space_id,
        message=f"Created concept space with ID: {space_id}"
    )

@function_tool
async def add_concept_to_space(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: ConceptInput
) -> ConceptOutput:
    """Add a concept to a conceptual space."""
    # Update context stats
    ctx.context.total_calls += 1
    
    concept_id = await reasoning_core.add_concept_to_space(
        space_id=input_data.space_id,
        name=input_data.name,
        properties=input_data.properties
    )
    
    return ConceptOutput(
        concept_id=concept_id,
        message=f"Added concept with ID: {concept_id} to space: {input_data.space_id}"
    )

@function_tool
async def add_relation_to_space(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: ConceptRelationInput
) -> str:
    """Add a relation between concepts in a space."""
    # Update context stats
    ctx.context.total_calls += 1
    
    await reasoning_core.add_relation_to_space(
        space_id=input_data.space_id,
        source_id=input_data.source_id,
        target_id=input_data.target_id,
        relation_type=input_data.relation_type,
        strength=input_data.strength
    )
    
    return f"Added relation from {input_data.source_id} to {input_data.target_id} in space: {input_data.space_id}"

@function_tool
async def get_concept_space(
    ctx: RunContextWrapper[ReasoningContext],
    space_id: str,
) -> ConceptSpaceDump:
    ctx.context.total_calls += 1
    spc = await reasoning_core.get_concept_space(space_id)
    return ConceptSpaceDump(
        space_id=space_id,
        space_json=json.dumps(spc, separators=(",", ":")),
    )


@function_tool
async def create_blend(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: BlendInput,
) -> BlendInfo:
    ctx.context.total_calls += 1
    mappings = []  # (placeholder; your real mapping logic here)
    blend_id = f"blend_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    payload = {
        "blend_id": blend_id,
        "blend_type": input_data.blend_type,
        "input_spaces": [input_data.space_id_1, input_data.space_id_2],
        "mappings": mappings,
        "status": "created",
    }
    return BlendInfo(
        blend_id=blend_id,
        blend_json=json.dumps(payload, separators=(",", ":")),
    )
# --------------------- Integrated Reasoning Agent Tools ---------------------

@function_tool
async def convert_blend_to_causal_model(
    ctx: RunContextWrapper[ReasoningContext],
    blend_id: str, 
    name: Optional[str] = None, 
    domain: Optional[str] = None
) -> str:
    """Convert a conceptual blend to a causal model."""
    # Update context stats
    ctx.context.total_calls += 1
    
    model_id = await reasoning_core.convert_blend_to_causal_model(
        blend_id=blend_id,
        name=name,
        domain=domain
    )
    
    # Update context
    ctx.context.active_model_id = model_id
    
    return f"Converted blend {blend_id} to causal model {model_id}"

@function_tool
async def convert_causal_model_to_concept_space(
    ctx: RunContextWrapper[ReasoningContext],
    model_id: str, 
    name: Optional[str] = None, 
    domain: Optional[str] = None
) -> str:
    """Convert a causal model to a conceptual space."""
    # Update context stats
    ctx.context.total_calls += 1
    
    space_id = await reasoning_core.convert_causal_model_to_concept_space(
        model_id=model_id,
        name=name,
        domain=domain
    )
    
    # Update context
    ctx.context.active_space_id = space_id
    
    return f"Converted causal model {model_id} to concept space {space_id}"


@function_tool
async def create_creative_intervention(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: CreativeInterventionInput,
) -> CreativeInterventionResult:
    ctx.context.total_calls += 1
    res = await reasoning_core.create_creative_intervention(
        model_id=input_data.model_id,
        target_node=input_data.target_node,
        description=input_data.description,
        use_blending=input_data.use_blending,
    )
    return CreativeInterventionResult(
        intervention_json=json.dumps(res, separators=(",", ":")),
    )


@function_tool
async def create_integrated_model(
    ctx: RunContextWrapper[ReasoningContext],
    input_data: IntegratedModelInput,
) -> IntegratedModelResult:
    ctx.context.total_calls += 1
    res = await reasoning_core.create_integrated_model(
        domain=input_data.domain,
        base_on_causal=input_data.base_on_causal,
    )
    # update context if IDs returned
    ctx.context.active_model_id = res.get("causal_model_id")
    ctx.context.active_space_id = res.get("concept_space_id")
    return IntegratedModelResult(
        integrated_json=json.dumps(res, separators=(",", ":")),
    )





@function_tool
async def get_reasoning_stats(
    ctx: RunContextWrapper[ReasoningContext],
) -> ReasoningStats:
    ctx.context.total_calls += 1
    stats = await reasoning_core.get_stats()
    return ReasoningStats(
        stats_json=json.dumps(stats, separators=(",", ":")),
    )

# --------------------- Define the Agents ---------------------

# Define the Causal Reasoning Agent
causal_reasoning_agent = Agent[ReasoningContext](
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
    ],
    model="gpt-4.1-nano"
)

# Define the Conceptual Reasoning Agent
conceptual_reasoning_agent = Agent[ReasoningContext](
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
    ],
    model="gpt-4.1-nano"
)

# Define the Integrated Reasoning Agent
integrated_reasoning_agent = Agent[ReasoningContext](
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
                tool_description_override="Transfer to the causal reasoning agent for detailed causal modeling and analysis"),
        handoff(conceptual_reasoning_agent, 
                tool_description_override="Transfer to the conceptual reasoning agent for conceptual space creation and blending")
    ],
    model="gpt-4.1-nano"
)

# --------------------- Define Triage Agent and Guardrails ---------------------

# Create a homework detection agent
homework_detection_agent = Agent[ReasoningContext](
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
triage_agent = Agent[ReasoningContext](
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
                tool_description_override="Transfer to the causal reasoning specialist for causal modeling and analysis"),
        handoff(conceptual_reasoning_agent, 
                tool_description_override="Transfer to the conceptual reasoning specialist for conceptual space creation and blending"),
        handoff(integrated_reasoning_agent, 
                tool_description_override="Transfer to the integrated reasoning specialist that combines causal and conceptual approaches")
    ],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail)
    ],
    model="gpt-4.1-nano"
)

# --------------------- Main Entry Point ---------------------

async def process_reasoning_request(user_input: str, session_id: str, user_id: Optional[str] = None):
    """Process a user request through the reasoning agents system with proper tracing."""
    
    # Create context
    context = ReasoningContext(
        session_id=session_id,
        user_id=user_id,
        knowledge_core=reasoning_core.knowledge_core,
    )
    
    # Run the agent with tracing
    with trace(workflow_name=f"reasoning_request_{session_id}", group_id=session_id):
        result = await Runner.run(
            triage_agent, 
            user_input, 
            context=context,
            run_config={
                "workflow_name": "reasoning_system",
                "group_id": session_id,
                "trace_metadata": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    # Return the final output
    return result.final_output

# Run synchronously for simple use cases
def process_reasoning_request_sync(user_input: str, session_id: str, user_id: Optional[str] = None):
    """Process a user request synchronously with proper tracing."""
    return asyncio.run(process_reasoning_request(user_input, session_id, user_id))

# --------------------- Conversation Management ---------------------

class ReasoningConversation:
    """Manage a multi-turn conversation with the reasoning system."""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.context = ReasoningContext(
            session_id=session_id,
            user_id=user_id,
            knowledge_core=reasoning_core.knowledge_core,
        )
        self.history = []
        self.last_agent = triage_agent
    
    async def process_message(self, user_input: str):
        """Process a user message and maintain conversation context."""
        
        # Set up run configuration with tracing
        run_config = {
            "workflow_name": "reasoning_conversation",
            "group_id": self.session_id,
            "trace_metadata": {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "turn_number": len(self.history) + 1
            }
        }
        
        # Create input by appending to history
        if self.history:
            # Convert history to input format
            input_list = self.history[-1].to_input_list()
            # Add new user message
            input_list.append({"role": "user", "content": user_input})
            input_data = input_list
        else:
            input_data = user_input
        
        # Run the most appropriate agent (either last agent or triage)
        with trace(workflow_name=f"reasoning_turn_{len(self.history)+1}", group_id=self.session_id):
            result = await Runner.run(
                self.last_agent, 
                input_data, 
                context=self.context,
                run_config=run_config
            )
        
        # Update history and last agent
        self.history.append(result)
        self.last_agent = result.last_agent
        
        return result.final_output

# Example usage
if __name__ == "__main__":
    user_query = "I want to build a causal model to understand climate change impacts on agriculture."
    response = process_reasoning_request_sync(
        user_query, 
        session_id=f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )
    print(response)
