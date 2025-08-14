# logic/conflict_system/dynamic_conflict_template.py
"""
Dynamic Conflict Template System with LLM-generated variations
Creates infinite conflict variations from base templates
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# TEMPLATE TYPES
# ===============================================================================

class TemplateCategory(Enum):
    """Categories of conflict templates"""
    POWER_DYNAMICS = "power_dynamics"
    SOCIAL_HIERARCHY = "social_hierarchy"
    RESOURCE_COMPETITION = "resource_competition"
    IDEOLOGICAL_CLASH = "ideological_clash"
    PERSONAL_BOUNDARIES = "personal_boundaries"
    LOYALTY_TESTS = "loyalty_tests"
    HIDDEN_AGENDAS = "hidden_agendas"
    TRANSFORMATION_RESISTANCE = "transformation_resistance"


@dataclass
class ConflictTemplate:
    """Base template for generating conflicts"""
    template_id: int
    category: TemplateCategory
    name: str
    base_structure: Dict[str, Any]
    variable_elements: List[str]
    contextual_modifiers: Dict[str, Any]
    complexity_range: Tuple[float, float]


@dataclass
class GeneratedConflict:
    """A conflict generated from a template"""
    conflict_id: int
    template_id: int
    variation_seed: str
    customization: Dict[str, Any]
    narrative_hooks: List[str]
    unique_elements: List[str]


# ===============================================================================
# DYNAMIC TEMPLATE GENERATOR
# ===============================================================================

class DynamicConflictTemplateSystem:
    """Generates infinite conflict variations from templates"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._template_creator = None
        self._variation_generator = None
        self._context_adapter = None
        self._uniqueness_engine = None
        self._hook_generator = None
    
    # ========== Agent Properties ==========
    
    @property
    def template_creator(self) -> Agent:
        if self._template_creator is None:
            self._template_creator = Agent(
                name="Template Creator",
                instructions="""
                Create flexible conflict templates that can generate countless variations.
                
                Design templates that:
                - Have clear core structures
                - Include variable elements
                - Allow contextual adaptation
                - Support complexity scaling
                - Enable emergent storytelling
                
                Templates should be seeds for infinite stories, not rigid patterns.
                """,
                model="gpt-5-nano",
            )
        return self._template_creator
    
    @property
    def variation_generator(self) -> Agent:
        if self._variation_generator is None:
            self._variation_generator = Agent(
                name="Variation Generator",
                instructions="""
                Generate unique variations from conflict templates.
                
                Create variations that:
                - Feel fresh and original
                - Respect template structure
                - Add surprising elements
                - Fit the context perfectly
                - Create memorable experiences
                
                Each variation should feel like a unique story, not a copy.
                """,
                model="gpt-5-nano",
            )
        return self._variation_generator
    
    @property
    def context_adapter(self) -> Agent:
        if self._context_adapter is None:
            self._context_adapter = Agent(
                name="Context Adapter",
                instructions="""
                Adapt conflict templates to specific contexts.
                
                Ensure adaptations:
                - Fit the current situation
                - Respect character personalities
                - Match location atmosphere
                - Align with ongoing narratives
                - Feel organic to the world
                
                Make templated conflicts feel bespoke to the moment.
                """,
                model="gpt-5-nano",
            )
        return self._context_adapter
    
    @property
    def uniqueness_engine(self) -> Agent:
        if self._uniqueness_engine is None:
            self._uniqueness_engine = Agent(
                name="Uniqueness Engine",
                instructions="""
                Ensure each generated conflict feels unique and memorable.
                
                Add elements that:
                - Create distinctive moments
                - Generate quotable lines
                - Produce unexpected twists
                - Build character-specific drama
                - Leave lasting impressions
                
                Every conflict should have something players remember.
                """,
                model="gpt-5-nano",
            )
        return self._uniqueness_engine
    
    @property
    def hook_generator(self) -> Agent:
        if self._hook_generator is None:
            self._hook_generator = Agent(
                name="Narrative Hook Generator",
                instructions="""
                Generate compelling hooks that draw players into conflicts.
                
                Create hooks that:
                - Grab immediate attention
                - Create emotional investment
                - Promise interesting outcomes
                - Connect to player history
                - Build anticipation
                
                Make players WANT to engage with the conflict.
                """,
                model="gpt-5-nano",
            )
        return self._hook_generator
    
    # ========== Template Creation ==========
    
    async def create_conflict_template(
        self,
        category: TemplateCategory,
        base_concept: str
    ) -> ConflictTemplate:
        """Create a new reusable conflict template"""
        
        prompt = f"""
        Create a flexible conflict template:
        
        Category: {category.value}
        Base Concept: {base_concept}
        
        Design a template that can generate hundreds of unique conflicts.
        
        Return JSON:
        {{
            "name": "Template name",
            "base_structure": {{
                "core_tension": "Fundamental conflict",
                "stakeholder_roles": ["role types needed"],
                "progression_phases": ["typical phases"],
                "resolution_conditions": ["ways it can end"]
            }},
            "variable_elements": [
                "List of 10+ elements that can change between instances"
            ],
            "contextual_modifiers": {{
                "personality_axes": ["relevant personality traits"],
                "environmental_factors": ["location/setting influences"],
                "cultural_variables": ["social/cultural elements"],
                "power_modifiers": ["hierarchy/authority factors"]
            }},
            "generation_rules": {{
                "required_elements": ["must-have components"],
                "optional_elements": ["can-have components"],
                "exclusions": ["incompatible elements"]
            }},
            "complexity_range": {{
                "minimum": 0.2,
                "maximum": 0.9
            }}
        }}
        """
        
        response = await Runner.run(self.template_creator, prompt)
        data = json.loads(response.output)
        
        # Store template
        async with get_db_connection_context() as conn:
            template_id = await conn.fetchval("""
                INSERT INTO conflict_templates
                (category, name, base_structure, variable_elements,
                 contextual_modifiers, complexity_min, complexity_max)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING template_id
            """, category.value, data['name'],
            json.dumps(data['base_structure']),
            json.dumps(data['variable_elements']),
            json.dumps(data['contextual_modifiers']),
            data['complexity_range']['minimum'],
            data['complexity_range']['maximum'])
        
        return ConflictTemplate(
            template_id=template_id,
            category=category,
            name=data['name'],
            base_structure=data['base_structure'],
            variable_elements=data['variable_elements'],
            contextual_modifiers=data['contextual_modifiers'],
            complexity_range=(
                data['complexity_range']['minimum'],
                data['complexity_range']['maximum']
            )
        )
    
    # ========== Conflict Generation ==========
    
    async def generate_conflict_from_template(
        self,
        template_id: int,
        context: Dict[str, Any]
    ) -> GeneratedConflict:
        """Generate a unique conflict from a template"""
        
        # Get template
        async with get_db_connection_context() as conn:
            template_data = await conn.fetchrow("""
                SELECT * FROM conflict_templates WHERE template_id = $1
            """, template_id)
        
        template = ConflictTemplate(
            template_id=template_id,
            category=TemplateCategory(template_data['category']),
            name=template_data['name'],
            base_structure=json.loads(template_data['base_structure']),
            variable_elements=json.loads(template_data['variable_elements']),
            contextual_modifiers=json.loads(template_data['contextual_modifiers']),
            complexity_range=(
                template_data['complexity_min'],
                template_data['complexity_max']
            )
        )
        
        # Generate variation
        variation = await self._generate_variation(template, context)
        
        # Adapt to context
        adapted = await self._adapt_to_context(variation, context)
        
        # Add unique elements
        unique = await self._add_unique_elements(adapted, context)
        
        # Generate hooks
        hooks = await self._generate_narrative_hooks(unique, context)
        
        # Create the conflict
        conflict_id = await self._create_conflict_from_generation(
            template,
            unique,
            hooks
        )
        
        return GeneratedConflict(
            conflict_id=conflict_id,
            template_id=template_id,
            variation_seed=unique['seed'],
            customization=unique['customization'],
            narrative_hooks=hooks,
            unique_elements=unique['unique_elements']
        )
    
    async def _generate_variation(
        self,
        template: ConflictTemplate,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a variation from template"""
        
        prompt = f"""
        Generate a unique variation from this template:
        
        Template: {template.name}
        Base Structure: {json.dumps(template.base_structure)}
        Variable Elements: {json.dumps(template.variable_elements)}
        Context: {json.dumps(context)}
        
        Create a variation that:
        - Uses the base structure
        - Varies 3-5 variable elements
        - Feels fresh and original
        - Fits the context
        
        Return JSON:
        {{
            "seed": "Unique identifier for this variation",
            "core_tension": "Specific tension for this instance",
            "stakeholder_configuration": {{
                "roles": ["specific roles"],
                "relationships": ["specific relationships"]
            }},
            "chosen_variables": {{
                "variable_name": "specific value"
            }},
            "progression_path": ["specific phases"],
            "resolution_options": ["specific endings"],
            "twist_potential": "Unexpected element"
        }}
        """
        
        response = await Runner.run(self.variation_generator, prompt)
        return json.loads(response.output)
    
    async def _adapt_to_context(
        self,
        variation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt variation to specific context"""
        
        prompt = f"""
        Adapt this conflict variation to the specific context:
        
        Variation: {json.dumps(variation)}
        Current Location: {context.get('location', 'unknown')}
        Present NPCs: {json.dumps(context.get('npcs', []))}
        Time of Day: {context.get('time', 'unknown')}
        Recent Events: {json.dumps(context.get('recent_events', []))}
        
        Adapt by:
        - Fitting the location perfectly
        - Using NPC personalities
        - Matching the time/mood
        - Building on recent events
        
        Return JSON:
        {{
            "location_integration": "How location shapes conflict",
            "npc_motivations": {{
                "npc_id": "specific motivation"
            }},
            "temporal_factors": "How timing affects it",
            "continuity_connections": ["links to recent events"],
            "environmental_obstacles": ["location-specific challenges"],
            "atmospheric_elements": ["mood and tone elements"]
        }}
        """
        
        response = await Runner.run(self.context_adapter, prompt)
        adapted = variation.copy()
        adapted['context_adaptation'] = json.loads(response.output)
        return adapted
    
    async def _add_unique_elements(
        self,
        adapted: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add unique memorable elements"""
        
        prompt = f"""
        Add unique elements to make this conflict memorable:
        
        Conflict: {json.dumps(adapted)}
        Player History: {json.dumps(context.get('player_history', []))}
        
        Add:
        - A memorable quirk or detail
        - An unexpected element
        - A quotable moment
        - A visual or sensory detail
        - Something players will talk about
        
        Return JSON:
        {{
            "unique_elements": [
                "List of 3-5 unique elements"
            ],
            "memorable_quote": "Something an NPC might say",
            "signature_moment": "A scene players will remember",
            "sensory_detail": "Something visceral",
            "conversation_piece": "What players will discuss later"
        }}
        """
        
        response = await Runner.run(self.uniqueness_engine, prompt)
        unique_data = json.loads(response.output)
        
        adapted['unique_elements'] = unique_data['unique_elements']
        adapted['signature_content'] = unique_data
        adapted['customization'] = {
            'base_variation': adapted.get('seed', 'unknown'),
            'context_layer': adapted.get('context_adaptation', {}),
            'unique_layer': unique_data
        }
        
        return adapted
    
    async def _generate_narrative_hooks(
        self,
        conflict_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate compelling narrative hooks"""
        
        prompt = f"""
        Generate narrative hooks for this conflict:
        
        Core Tension: {conflict_data.get('core_tension', '')}
        Unique Elements: {json.dumps(conflict_data.get('unique_elements', []))}
        Signature Moment: {conflict_data.get('signature_content', {}).get('signature_moment', '')}
        
        Create 3-5 hooks that:
        - Grab immediate attention
        - Promise interesting outcomes
        - Create emotional investment
        - Feel urgent or important
        - Connect to player experience
        
        Return JSON:
        {{
            "hooks": [
                "List of compelling one-sentence hooks"
            ]
        }}
        """
        
        response = await Runner.run(self.hook_generator, prompt)
        data = json.loads(response.output)
        return data['hooks']
    
    async def _create_conflict_from_generation(
        self,
        template: ConflictTemplate,
        generation_data: Dict[str, Any],
        hooks: List[str]
    ) -> int:
        """Create actual conflict from generated data"""
        
        # Calculate complexity
        complexity = random.uniform(
            template.complexity_range[0],
            template.complexity_range[1]
        )
        
        # Create conflict
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts
                (user_id, conversation_id, conflict_type, conflict_name,
                 description, intensity, phase, is_active, progress,
                 template_id, generation_data)
                VALUES ($1, $2, $3, $4, $5, $6, 'emerging', true, 0, $7, $8)
                RETURNING conflict_id
            """, self.user_id, self.conversation_id,
            template.category.value,
            generation_data.get('seed', 'Generated Conflict'),
            hooks[0] if hooks else 'A new tension emerges',
            self._calculate_intensity(complexity),
            template.template_id,
            json.dumps(generation_data))
        
        return conflict_id
    
    def _calculate_intensity(self, complexity: float) -> str:
        """Calculate intensity from complexity"""
        if complexity < 0.3:
            return "subtle"
        elif complexity < 0.5:
            return "tension"
        elif complexity < 0.7:
            return "friction"
        elif complexity < 0.9:
            return "opposition"
        else:
            return "confrontation"
    
    # ========== Template Evolution ==========
    
    async def evolve_template(
        self,
        template_id: int,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evolve template based on usage feedback"""
        
        # Get template and generation history
        async with get_db_connection_context() as conn:
            template = await conn.fetchrow("""
                SELECT * FROM conflict_templates WHERE template_id = $1
            """, template_id)
            
            generations = await conn.fetch("""
                SELECT generation_data, player_feedback
                FROM Conflicts
                WHERE template_id = $1
                LIMIT 20
            """, template_id)
        
        prompt = f"""
        Evolve this template based on feedback:
        
        Template: {template['name']}
        Current Structure: {template['base_structure']}
        Variable Elements: {template['variable_elements']}
        Feedback: {json.dumps(feedback)}
        Generation History: {json.dumps([dict(g) for g in generations[:5]])}
        
        Suggest improvements:
        
        Return JSON:
        {{
            "new_variables": ["suggested new variable elements"],
            "deprecated_variables": ["elements to remove"],
            "structure_adjustments": {{
                "add": {{}},
                "modify": {{}},
                "remove": []
            }},
            "complexity_adjustment": {{
                "new_min": 0.0 to 1.0,
                "new_max": 0.0 to 1.0
            }},
            "evolution_reason": "Why these changes improve the template"
        }}
        """
        
        response = await Runner.run(self.template_creator, prompt)
        evolution = json.loads(response.output)
        
        # Apply evolution
        await self._apply_template_evolution(template_id, evolution)
        
        return evolution
    
    async def _apply_template_evolution(
        self,
        template_id: int,
        evolution: Dict[str, Any]
    ):
        """Apply evolution to template"""
        
        async with get_db_connection_context() as conn:
            # Get current template
            template = await conn.fetchrow("""
                SELECT * FROM conflict_templates WHERE template_id = $1
            """, template_id)
            
            # Update structures
            base_structure = json.loads(template['base_structure'])
            variable_elements = json.loads(template['variable_elements'])
            
            # Apply changes
            if 'new_variables' in evolution:
                variable_elements.extend(evolution['new_variables'])
            if 'deprecated_variables' in evolution:
                variable_elements = [
                    v for v in variable_elements
                    if v not in evolution['deprecated_variables']
                ]
            
            # Update database
            await conn.execute("""
                UPDATE conflict_templates
                SET variable_elements = $1,
                    complexity_min = $2,
                    complexity_max = $3,
                    last_evolved = CURRENT_TIMESTAMP
                WHERE template_id = $4
            """, json.dumps(variable_elements),
            evolution['complexity_adjustment']['new_min'],
            evolution['complexity_adjustment']['new_max'],
            template_id)


# ===============================================================================
# INTEGRATION FUNCTIONS
# ===============================================================================

@function_tool
async def generate_templated_conflict(
    ctx: RunContextWrapper,
    category: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a conflict from a template category"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    system = DynamicConflictTemplateSystem(user_id, conversation_id)
    
    # Get or create template for category
    async with get_db_connection_context() as conn:
        template = await conn.fetchrow("""
            SELECT template_id FROM conflict_templates
            WHERE category = $1
            ORDER BY RANDOM()
            LIMIT 1
        """, category)
    
    if not template:
        # Create new template
        template_obj = await system.create_conflict_template(
            TemplateCategory(category),
            f"Dynamic {category} conflict"
        )
        template_id = template_obj.template_id
    else:
        template_id = template['template_id']
    
    # Generate conflict
    generated = await system.generate_conflict_from_template(
        template_id,
        context
    )
    
    return {
        'conflict_id': generated.conflict_id,
        'template_used': template_id,
        'hooks': generated.narrative_hooks,
        'unique_elements': generated.unique_elements
    }


@function_tool
async def create_custom_template(
    ctx: RunContextWrapper,
    category: str,
    concept: str
) -> Dict[str, Any]:
    """Create a custom conflict template"""
    
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    
    system = DynamicConflictTemplateSystem(user_id, conversation_id)
    
    template = await system.create_conflict_template(
        TemplateCategory(category),
        concept
    )
    
    return {
        'template_id': template.template_id,
        'name': template.name,
        'category': template.category.value,
        'variable_count': len(template.variable_elements),
        'complexity_range': template.complexity_range
    }
