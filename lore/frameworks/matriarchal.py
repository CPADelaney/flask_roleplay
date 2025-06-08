# lore/frameworks/matriarchal.py

import json
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field

# Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    handoff,
    RunResultStreaming,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail
)
from agents.run_context import RunContextWrapper
from agents.run import RunConfig

# Project-specific import
from lore.managers.base_manager import BaseLoreManager

# ------------------------------------------------------------------
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ------------------------------------------------------------------

class PowerHierarchy(BaseModel):
    """Power structure within a matriarchal society."""
    dominant_gender: str = "female"
    power_expressions: List[str]
    leadership_positions: List[Dict[str, Any]]
    submissive_roles: List[Dict[str, Any]]

class PowerDynamics(BaseModel):
    """Power dynamics component."""
    dominant_gender: str = "female"
    power_expressions: List[str]
    authority_sources: List[str]
    submission_forms: List[str]

class SocietalNorms(BaseModel):
    """Societal norms component."""
    gender_roles: Dict[str, Any]
    behavioral_expectations: Dict[str, Any]
    social_structures: Dict[str, Any]

class SymbolicRepresentations(BaseModel):
    """Symbolic representations component."""
    symbols: Dict[str, Any]
    rituals: Dict[str, Any]
    ceremonial_elements: Dict[str, Any]

class CorePrinciples(BaseModel):
    """Core principles of a matriarchal power structure."""
    power_dynamics: Dict[str, Any]
    societal_norms: Dict[str, Any]
    symbolic_representations: Dict[str, Any]

class HierarchicalConstraint(BaseModel):
    """Hierarchical constraints in a matriarchal setting."""
    dominant_hierarchy_type: str
    description: str
    power_expressions: List[str]
    masculine_roles: List[str]
    leadership_domains: List[str]
    property_rights: Any
    status_markers: Any
    relationship_structure: str
    enforcement_mechanisms: List[str]

class PowerExpression(BaseModel):
    """Expression of power in a matriarchal society."""
    domain: str
    title: str
    description: str
    male_role: str

# Self-evaluation feedback model
class NarrativeEvaluation(BaseModel):
    """Evaluation of a narrative transformation."""
    matriarchal_strength: int = Field(..., ge=1, le=10)
    narrative_quality: int = Field(..., ge=1, le=10)
    consistency: int = Field(..., ge=1, le=10)
    engagement: int = Field(..., ge=1, le=10)
    improvements: List[str]

# ------------------------------------------------------------------
# MAIN FRAMEWORK CLASS
# ------------------------------------------------------------------

class MatriarchalPowerStructureFramework(BaseLoreManager):
    """
    Defines core principles for power dynamics in femdom/matriarchal settings,
    ensuring consistency across generated lore.
    Enhanced with structured outputs, handoffs, and self-evaluation.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)

        # Initialize the transformation agent with structured outputs
        self.transformation_agent = Agent(
            name="MatriarchalTransformationAgent",
            instructions=(
                "You are an expert at crafting narrative-rich, matriarchal (femdom) lore. "
                "Given instructions and context, you generate or rewrite text in a strongly matriarchal style. "
                "Your output should be immersive, cohesive, and consistent with the premise that "
                "women hold most or all power, and men occupy subordinate or service-based roles."
            ),
            model="gpt-4o-mini",  # Using valid model name
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Initialize specialized narrative transformation agents
        self._init_specialized_agents()
        
        # Initialize the evaluation agent for feedback loops
        self.evaluation_agent = Agent(
            name="NarrativeEvaluationAgent",
            instructions=(
                "You evaluate narratives for strength of matriarchal themes, narrative quality, "
                "consistency, and engagement. Provide constructive feedback for improvements."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.5),
            output_type=NarrativeEvaluation
        )

    def _init_specialized_agents(self):
        """Initialize specialized agents for different narrative domains."""
        # Political narrative specialist
        self.political_specialist = Agent(
            name="PoliticalMatriarchyAgent",
            instructions=(
                "You specialize in transforming political narratives to reflect matriarchal power structures. "
                "Focus on governance, authority, and policy formation led by women, with men in advisory "
                "or supportive roles only."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Religious narrative specialist
        self.religious_specialist = Agent(
            name="ReligiousMatriarchyAgent",
            instructions=(
                "You specialize in transforming religious narratives to center feminine divinity. "
                "Create matriarchal religious structures with goddesses as primary deities and "
                "priestesses as the dominant religious authorities."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Social/cultural narrative specialist
        self.cultural_specialist = Agent(
            name="CulturalMatriarchyAgent",
            instructions=(
                "You specialize in transforming social and cultural narratives to reflect matriarchal norms. "
                "Create customs, traditions, and social interactions that reinforce feminine authority "
                "and masculine deference."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Configure agent handoffs
        self.transformation_agent = self.transformation_agent.clone(
            handoffs=[
                handoff(
                    self.political_specialist,
                    tool_name_override="transform_political_narrative",
                    tool_description_override="Transform political narratives to reflect matriarchal structures"
                ),
                handoff(
                    self.religious_specialist,
                    tool_name_override="transform_religious_narrative",
                    tool_description_override="Transform religious narratives to center feminine divinity"
                ),
                handoff(
                    self.cultural_specialist,
                    tool_name_override="transform_cultural_narrative",
                    tool_description_override="Transform social and cultural narratives to reflect matriarchal norms"
                )
            ]
        )

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    async def _transform_text(self, original_text: str, context_desc: str) -> str:
        """
        Send `original_text` to the LLM-based transformation_agent, 
        requesting a matriarchal rewrite based on context_desc.
        """
        prompt = (
            f"CONTEXT:\n{context_desc}\n\n"
            f"ORIGINAL TEXT:\n{original_text}\n\n"
            "Rewrite or transform the text to reflect a strongly matriarchal society. "
            "Focus on narrative immersion and creative detail. Only output the final text."
        )

        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "matriarchal transformation"
        })
        run_cfg = RunConfig(
            workflow_name="MatriarchalRewrite",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(
            starting_agent=self.transformation_agent,
            input=prompt,
            context=run_ctx.context,
            run_config=run_cfg
        )
        
        # Get feedback on the transformation
        transformed_text = result.final_output
        await self._evaluate_transformation(original_text, transformed_text, context_desc)
        
        return transformed_text

    async def _evaluate_transformation(self, original: str, transformed: str, context: str) -> NarrativeEvaluation:
        """
        Evaluate the quality of a narrative transformation using the evaluation agent.
        This implements a feedback loop for continuous improvement.
        """
        prompt = (
            f"ORIGINAL TEXT:\n{original}\n\n"
            f"TRANSFORMED TEXT:\n{transformed}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Evaluate this transformation for matriarchal strength, narrative quality, "
            "consistency, and engagement. Provide specific suggestions for improvement."
        )
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "transformation evaluation"
        })
        
        result = await Runner.run(
            self.evaluation_agent,
            prompt,
            context=run_ctx.context
        )
        
        evaluation = result.final_output_as(NarrativeEvaluation)
        
        # Log the evaluation for future improvements
        logging.info(f"Transformation evaluation: {evaluation}")
        
        # If evaluation shows room for improvement, we could retry the transformation
        if evaluation.matriarchal_strength < 7 or evaluation.narrative_quality < 7:
            logging.info("Transformation quality below threshold, improvements needed")
            # In a real implementation, you might retry with more specific guidance
        
        return evaluation

    async def _call_transformation_agent_for_json(self, prompt_text: str, workflow_name: str) -> Any:
        """
        Helper function that sends a prompt requesting strictly valid JSON output. 
        If the response is invalid JSON, returns None or an empty structure.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "fetching JSON data for matriarchal lore"
        })
        run_cfg = RunConfig(
            workflow_name=workflow_name,
            trace_metadata=self.trace_metadata
        )

        result = await Runner.run(
            starting_agent=self.transformation_agent,
            input=prompt_text,
            context=run_ctx.context,
            run_config=run_cfg
        )

        try:
            return json.loads(result.final_output)
        except json.JSONDecodeError:
            return None  # or return an empty dict/list if you prefer

    # ------------------------------------------------------------------
    # 1) Generating Core Principles with Pydantic Model
    # ------------------------------------------------------------------
    @function_tool
    async def generate_core_principles(self) -> CorePrinciples:
        """
        Dynamically generate a set of 'core principles' for a femdom/matriarchal world,
        returning a structured CorePrinciples object.

        Returns:
            A CorePrinciples object representing the newly generated principles.
        """
        prompt = (
            "Generate a JSON object describing the core principles of a strongly matriarchal (femdom) world. "
            "Include sections like 'power_dynamics', 'societal_norms', and 'symbolic_representations'. "
            "Each section should be detailed, engaging, and suitable for a narrative-heavy setting. "
            "Your output must be valid JSON, with no extra text. Example structure:\n\n"
            "{\n"
            '  "power_dynamics": {\n'
            '      "dominant_gender": "female",\n'
            '      "power_expressions": ["..."],\n'
            '      ...\n'
            "  },\n"
            '  "societal_norms": {\n'
            '      ...\n'
            "  },\n"
            '  "symbolic_representations": {\n'
            '      ...\n'
            "  }\n"
            "}"
        )

        principles_agent = self.transformation_agent.clone(
            output_type=CorePrinciples
        )
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "generating core principles"
        })
        
        run_cfg = RunConfig(
            workflow_name="GenerateCorePrinciples",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(
            principles_agent,
            prompt,
            context=run_ctx.context,
            run_config=run_cfg
        )
        
        return result.final_output_as(CorePrinciples)

    # ------------------------------------------------------------------
    # 2) Generating Hierarchical Constraints with Pydantic Model
    # ------------------------------------------------------------------
    @function_tool
    async def generate_hierarchical_constraints(self) -> HierarchicalConstraint:
        """
        Use the LLM to produce an immersive, narrative-heavy HierarchicalConstraint object that 
        describes the hierarchical constraints in a matriarchal setting.

        Returns:
            A HierarchicalConstraint object describing hierarchy_type, power_expressions, roles, etc.
        """
        prompt = (
            "Produce a JSON object describing hierarchical constraints in a femdom/matriarchal world. "
            "It must include items such as:\n"
            "- 'dominant_hierarchy_type': a key string or short phrase.\n"
            "- 'description': a short narrative statement clarifying that hierarchy.\n"
            "- 'power_expressions': an array of unique ways female power is exerted.\n"
            "- 'masculine_roles': an array describing how men fit into each expression.\n"
            "- 'leadership_domains': an array listing spheres where women hold undisputed authority.\n"
            "- 'property_rights': an array or statement about how ownership is allocated.\n"
            "- 'status_markers': how men achieve or lose status.\n"
            "- 'relationship_structure': e.g., polygyny, polyandry, etc.\n"
            "- 'enforcement_mechanisms': how rules are enforced.\n\n"
            "Make it interesting, unique, and fully valid JSON with no wrapping text."
        )

        constraints_agent = self.transformation_agent.clone(
            output_type=HierarchicalConstraint
        )
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "generating hierarchical constraints"
        })
        
        run_cfg = RunConfig(
            workflow_name="GenerateHierarchicalConstraints",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(
            constraints_agent,
            prompt,
            context=run_ctx.context,
            run_config=run_cfg
        )
        
        return result.final_output_as(HierarchicalConstraint)

    # ------------------------------------------------------------------
    # 3) LENS APPLICATION WITH SPECIALIZED HANDOFFS
    # ------------------------------------------------------------------
    @function_tool(strict_mode=False)  # Disable strict schema to allow flexible data
    async def apply_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a matriarchal lens to the foundation lore, using specialized narrative
        transformation agents based on content type.
        
        Args:
            foundation_data: The foundation lore data to transform
            
        Returns:
            Transformed foundation data with matriarchal themes applied
        """
        result = foundation_data.copy()
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "applying power lens"
        })
        
        if "social_structure" in foundation_data:
            # Use the cultural specialist for social structures
            prompt = (
                f"ORIGINAL SOCIAL STRUCTURE:\n{foundation_data['social_structure']}\n\n"
                "Transform this social structure description to reflect a strongly "
                "matriarchal society. Focus on feminine authority, masculine deference, "
                "and gendered power dynamics."
            )
            
            social_result = await Runner.run(
                self.cultural_specialist,
                prompt,
                context=run_ctx.context
            )
            result["social_structure"] = social_result.final_output
        
        if "cosmology" in foundation_data:
            # Use the religious specialist for cosmology
            prompt = (
                f"ORIGINAL COSMOLOGY:\n{foundation_data['cosmology']}\n\n"
                "Transform this cosmology to center feminine divine power. Emphasize "
                "goddess figures, feminine creation principles, and matriarchal "
                "religious structures."
            )
            
            cosmology_result = await Runner.run(
                self.religious_specialist,
                prompt,
                context=run_ctx.context
            )
            result["cosmology"] = cosmology_result.final_output
        
        if "magic_system" in foundation_data:
            # General transformation for magic systems
            result["magic_system"] = await self._transform_text(
                foundation_data["magic_system"],
                context_desc="Highlight how women wield greater or central magical authority."
            )
        
        if "world_history" in foundation_data:
            # Use the political specialist for world history
            prompt = (
                f"ORIGINAL WORLD HISTORY:\n{foundation_data['world_history']}\n\n"
                "Transform this world history to center women as the primary historical "
                "actors, leaders, conquerors, and decision-makers. Men should appear in "
                "supportive roles or as subjects/conquered peoples."
            )
            
            history_result = await Runner.run(
                self.political_specialist,
                prompt,
                context=run_ctx.context
            )
            result["world_history"] = history_result.final_output
            
        if "calendar_system" in foundation_data:
            # General transformation for calendar systems
            result["calendar_system"] = await self._transform_text(
                foundation_data["calendar_system"],
                context_desc="Show feminine significance in months, lunar cycles, and symbolic rituals."
            )
            
        return result

    # ------------------------------------------------------------------
    # 4) Generating Power Expressions with Pydantic Models
    # ------------------------------------------------------------------
    @function_tool
    async def generate_power_expressions(self) -> List[PowerExpression]:
        """
        Generate a list of power expressions describing ways in which 
        female authority and male submission manifest in the world.
        Returns a list of PowerExpression objects.
        """
        prompt = (
            "Generate a JSON array, each item describing a unique 'power expression' in a "
            "femdom/matriarchal fantasy world. Include fields:\n"
            "- 'domain': (political, economic, religious, domestic, sexual, etc.)\n"
            "- 'title': short name or label for the expression.\n"
            "- 'description': a story-like explanation of how women's power is exercised.\n"
            "- 'male_role': how men specifically submit or contribute.\n\n"
            "Output strictly valid JSON with no additional formatting."
        )

        expressions_agent = self.transformation_agent.clone(
            output_type=List[PowerExpression]
        )
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "generating power expressions"
        })
        
        run_cfg = RunConfig(
            workflow_name="GeneratePowerExpressions",
            trace_metadata=self.trace_metadata
        )
        
        result = await Runner.run(
            expressions_agent,
            prompt,
            context=run_ctx.context,
            run_config=run_cfg
        )
        
        return result.final_output_as(List[PowerExpression])
    
    # ------------------------------------------------------------------
    # 5) Dialogue-based narrative development
    # ------------------------------------------------------------------
    @function_tool
    async def develop_narrative_through_dialogue(
        self, 
        narrative_theme: str, 
        initial_scene: str
    ) -> AsyncGenerator[str, None]:
        """
        Develop a narrative through iterative dialogue between specialized agents,
        streaming the results as they're generated.
        
        Args:
            narrative_theme: The theme of the narrative (e.g., "coming of age")
            initial_scene: Starting point for the narrative
            
        Yields:
            Narrative segments as they are developed
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "purpose": "narrative development"
        })
        
        # Initialize the narrative
        narrative = f"THEME: {narrative_theme}\n\n"
        narrative += f"INITIAL SCENE:\n{initial_scene}\n\n"
        
        yield narrative  # Send initial setup
        
        # Create agents for dialogue-based development
        plot_agent = Agent(
            name="PlotDevelopmentAgent",
            instructions=(
                "You develop plot elements in a matriarchal narrative. "
                "Continue from the existing narrative, adding new developments, "
                "conflicts, or revelations."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        character_agent = Agent(
            name="CharacterDevelopmentAgent",
            instructions=(
                "You develop characters in a matriarchal narrative. "
                "Focus on character growth, relationships, and internal struggles, "
                "while maintaining matriarchal power dynamics."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        setting_agent = Agent(
            name="SettingDevelopmentAgent",
            instructions=(
                "You develop settings and world elements in a matriarchal narrative. "
                "Enrich the existing narrative with vivid locations, cultural details, "
                "and environmental features that reinforce matriarchal themes."
            ),
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Dialogue-based development (alternating between agents)
        for step in range(5):  # 5 development steps
            # Plot development
            plot_prompt = f"Continue developing the plot for this narrative:\n\n{narrative}"
            plot_result = await Runner.run(plot_agent, plot_prompt, context=run_ctx.context)
            plot_development = plot_result.final_output
            
            narrative += f"\nPLOT DEVELOPMENT:\n{plot_development}\n"
            yield f"PLOT DEVELOPMENT:\n{plot_development}\n"  # Stream update
            
            # Character development
            char_prompt = f"Develop the characters in this narrative:\n\n{narrative}"
            char_result = await Runner.run(character_agent, char_prompt, context=run_ctx.context)
            char_development = char_result.final_output
            
            narrative += f"\nCHARACTER DEVELOPMENT:\n{char_development}\n"
            yield f"CHARACTER DEVELOPMENT:\n{char_development}\n"  # Stream update
            
            # Setting development
            setting_prompt = f"Enrich the setting in this narrative:\n\n{narrative}"
            setting_result = await Runner.run(setting_agent, setting_prompt, context=run_ctx.context)
            setting_development = setting_result.final_output
            
            narrative += f"\nSETTING DEVELOPMENT:\n{setting_development}\n"
            yield f"SETTING DEVELOPMENT:\n{setting_development}\n"  # Stream update
        
        # Final evaluation
        evaluation = await self._evaluate_transformation(
            initial_scene,
            narrative,
            f"Narrative development for theme: {narrative_theme}"
        )
        
        yield f"\nFINAL EVALUATION:\n"
        yield f"Matriarchal Strength: {evaluation.matriarchal_strength}/10\n"
        yield f"Narrative Quality: {evaluation.narrative_quality}/10\n"
        yield f"Consistency: {evaluation.consistency}/10\n"
        yield f"Engagement: {evaluation.engagement}/10\n"
        yield f"Improvement Suggestions: {', '.join(evaluation.improvements)}\n"
