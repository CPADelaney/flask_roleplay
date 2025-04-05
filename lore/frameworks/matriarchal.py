# lore/frameworks/matriarchal.py

import json
from typing import Dict, List, Any

# Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool
)
from agents.run_context import RunContextWrapper
from agents.run import RunConfig

# Project-specific import
from lore.core.base_manager import BaseLoreManager


class MatriarchalPowerStructureFramework(BaseLoreManager):
    """
    Defines core principles for power dynamics in femdom/matriarchal settings,
    ensuring consistency across generated lore. 
    This version is fully agent-ified so everything is dynamic and narrative-heavy.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)

        # We now rely on an Agent to generate or transform everything, 
        # instead of using a fixed dictionary or random sampling.
        self.transformation_agent = Agent(
            name="MatriarchalTransformationAgent",
            instructions=(
                "You are an expert at crafting narrative-rich, matriarchal (femdom) lore. "
                "Given instructions and context, you generate or rewrite text in a strongly matriarchal style. "
                "Your output should be immersive, cohesive, and consistent with the premise that "
                "women hold most or all power, and men occupy subordinate or service-based roles."
            ),
            model="o3-mini",  # You can change this to a more capable model if desired
            model_settings=ModelSettings(temperature=0.9)
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
        return result.final_output

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
    # 1) Generating Core Principles
    # ------------------------------------------------------------------
    @function_tool
    async def generate_core_principles(self) -> Dict[str, Any]:
        """
        Dynamically generate a set of 'core principles' for a femdom/matriarchal world,
        returning a JSON structure with sections like power_dynamics, societal_norms,
        symbolic_representations, etc.

        Returns:
            A dictionary representing the newly generated principles. 
            Example shape:
                {
                    "power_dynamics": {...},
                    "societal_norms": {...},
                    "symbolic_representations": {...}
                }
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

        data = await self._call_transformation_agent_for_json(
            prompt_text=prompt, 
            workflow_name="GenerateCorePrinciples"
        )
        if not data:
            return {}
        return data

    # ------------------------------------------------------------------
    # 2) Generating Hierarchical Constraints
    # ------------------------------------------------------------------
    @function_tool
    async def generate_hierarchical_constraints(self) -> Dict[str, Any]:
        """
        Use the LLM to produce an immersive, narrative-heavy JSON object that 
        describes the hierarchical constraints in a matriarchal setting. 
        This replaces the old random-sampling approach with agent-driven creativity.

        Returns:
            A dict describing hierarchy_type, power_expressions, roles, etc., 
            but in a dynamic, story-driven format.
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

        data = await self._call_transformation_agent_for_json(
            prompt_text=prompt, 
            workflow_name="GenerateHierarchicalConstraints"
        )
        if not data:
            return {}
        return data

    # ------------------------------------------------------------------
    # 3) LENS APPLICATION (foundation_data transformation)
    # ------------------------------------------------------------------
    @function_tool
    async def apply_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a matriarchal lens to the foundation lore, rewriting each 
        relevant field via the transformation agent for an engaging, 
        narrative-rich output.
        """
        if "social_structure" in foundation_data:
            foundation_data["social_structure"] = await self._transform_text(
                foundation_data["social_structure"],
                context_desc="Rewrite the social structure in a strongly matriarchal style."
            )
        
        if "cosmology" in foundation_data:
            foundation_data["cosmology"] = await self._transform_text(
                foundation_data["cosmology"],
                context_desc="Emphasize feminine primacy, goddess-centered beliefs, or gendered myth."
            )
        
        if "magic_system" in foundation_data:
            foundation_data["magic_system"] = await self._transform_text(
                foundation_data["magic_system"],
                context_desc="Highlight how women wield greater or central magical authority."
            )
        
        if "world_history" in foundation_data:
            foundation_data["world_history"] = await self._transform_text(
                foundation_data["world_history"],
                context_desc="Reflect matriarchal development, female-led conquests, or shifts in power."
            )
            
        if "calendar_system" in foundation_data:
            foundation_data["calendar_system"] = await self._transform_text(
                foundation_data["calendar_system"],
                context_desc="Show feminine significance in months, lunar cycles, and symbolic rituals."
            )
            
        return foundation_data

    # ------------------------------------------------------------------
    # 4) Generating Power Expressions
    # ------------------------------------------------------------------
    @function_tool
    async def generate_power_expressions(self) -> List[Dict[str, Any]]:
        """
        Generate a list of power expressions describing ways in which 
        female authority and male submission manifest in the world, 
        each accompanied by narrative detail. 
        Returns a parsed list from a JSON response.
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

        data = await self._call_transformation_agent_for_json(
            prompt_text=prompt,
            workflow_name="GeneratePowerExpressions"
        )
        if not data or not isinstance(data, list):
            return []
        return data
