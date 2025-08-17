# logic/conflict_system/dynamic_conflict_template.py
"""
Dynamic Conflict Template System with LLM-generated variations
Integrated with ConflictSynthesizer as the central orchestrator
REFACTORED: Fixed RunResult attribute access issues
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict, NotRequired
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# HELPER FUNCTION FOR RUNNER RESPONSE EXTRACTION
# ===============================================================================

def extract_runner_response(response) -> str:
    """
    Extract the actual response text from a Runner.run() result.
    Handles different possible RunResult structures.
    """
    # Try different attributes in order of likelihood
    if hasattr(response, 'output'):
        return response.output
    elif hasattr(response, 'data'):
        return response.data
    elif hasattr(response, 'result'):
        return response.result
    elif hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'messages') and response.messages:
        # If it's a list of messages, get the last one
        last_message = response.messages[-1]
        if hasattr(last_message, 'text'):
            return last_message.text
        elif hasattr(last_message, 'content'):
            return last_message.content
        else:
            return str(last_message)
    else:
        # Fallback to string representation
        return str(response)

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

class TemplateContextDTO(TypedDict, total=False):
    # Common conflict/context fields (extend as needed, all optional)
    participants: List[int]
    stakeholders: List[int]
    npcs: List[int]
    location: str
    location_id: int
    scene_type: str
    activity: str
    description: str
    intensity: str
    intensity_level: float  # 0..1
    hooks: List[str]
    complexity: float       # 0..1

class GenerateTemplatedConflictResponse(TypedDict):
    conflict_id: int
    status: str
    conflict_type: str
    template_used: int
    narrative_hooks: List[str]
    message: str
    error: str

class CreateTemplateResponse(TypedDict):
    template_id: int
    name: str
    category: str
    variable_count: int
    complexity_min: float
    complexity_max: float
    error: str


# ===============================================================================
# TEMPLATE SUBSYSTEM (Integrated with Synthesizer)
# ===============================================================================

class DynamicConflictTemplateSubsystem:
    """
    Template subsystem that integrates with ConflictSynthesizer.
    Generates infinite conflict variations from templates.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._template_creator = None
        self._variation_generator = None
        self._context_adapter = None
        self._uniqueness_engine = None
        self._hook_generator = None
        
        # Reference to synthesizer
        self.synthesizer = None
        
        # Template cache
        self._template_cache = {}
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.TEMPLATE
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'template_creation',
            'variation_generation',
            'context_adaptation',
            'uniqueness_injection',
            'hook_generation',
            'template_evolution'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.DETECTION,  # Use detection for template selection
            SubsystemType.FLOW  # Templates affect conflict flow
        }
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.TEMPLATE_GENERATED,
            EventType.CONFLICT_CREATED,
            EventType.HEALTH_CHECK,
            EventType.STATE_SYNC
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        
        # Create initial templates if none exist
        try:
            async with get_db_connection_context() as conn:
                template_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                if template_count == 0:
                    # Create base templates for each category
                    for category in list(TemplateCategory)[:3]:  # Start with 3 templates
                        await self.create_conflict_template(
                            category,
                            f"Base {category.value} template"
                        )
        except Exception as e:
            logger.error(f"Error initializing templates: {e}")
            # Continue even if templates can't be created
        
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                # Check if this conflict needs a template
                conflict_type = event.payload.get('conflict_type')
                if 'template' in conflict_type or event.payload.get('use_template'):
                    # Generate from template
                    template_id = event.payload.get('template_id')
                    if not template_id:
                        # Select appropriate template
                        template_id = await self._select_template(conflict_type)
                    
                    if template_id:
                        generated = await self.generate_conflict_from_template(
                            template_id,
                            event.payload.get('context', {})
                        )
                        
                        # Emit template generated event
                        side_effects = [SystemEvent(
                            event_id=f"template_gen_{event.event_id}",
                            event_type=EventType.TEMPLATE_GENERATED,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'template_id': template_id,
                                'conflict_id': generated.conflict_id,
                                'hooks': generated.narrative_hooks
                            },
                            priority=7
                        )]
                        
                        return SubsystemResponse(
                            subsystem=self.subsystem_type,
                            event_id=event.event_id,
                            success=True,
                            data={
                                'template_used': template_id,
                                'generated_conflict': generated.conflict_id,
                                'narrative_hooks': generated.narrative_hooks
                            },
                            side_effects=side_effects
                        )
                        
            elif event.event_type == EventType.STATE_SYNC:
                # Evolve templates based on usage
                if random.random() < 0.1:  # 10% chance
                    evolved = await self._evolve_random_template()
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'template_evolved': evolved}
                    )
                    
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check()
                )
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={}
            )
            
        except Exception as e:
            logger.error(f"Template subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        try:
            async with get_db_connection_context() as conn:
                template_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                # Check template usage
                used_count = await conn.fetchval("""
                    SELECT COUNT(DISTINCT template_id) FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND template_id IS NOT NULL
                """, self.user_id, self.conversation_id)
            
            return {
                'healthy': template_count > 0,
                'total_templates': template_count,
                'templates_used': used_count,
                'usage_ratio': used_count / template_count if template_count > 0 else 0,
                'issue': 'No templates available' if template_count == 0 else None
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                'healthy': False,
                'issue': str(e)
            }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get template-related data for a specific conflict"""
        try:
            async with get_db_connection_context() as conn:
                conflict = await conn.fetchrow("""
                    SELECT template_id, generation_data FROM Conflicts
                    WHERE conflict_id = $1
                """, conflict_id)
            
            if conflict and conflict['template_id']:
                template = await self._get_template(conflict['template_id'])
                generation_data = json.loads(conflict.get('generation_data', '{}'))
                
                return {
                    'template_used': template.name if template else 'Unknown',
                    'template_category': template.category.value if template else None,
                    'unique_elements': generation_data.get('unique_elements', []),
                    'variation_seed': generation_data.get('seed', 'default')
                }
        except Exception as e:
            logger.error(f"Error getting conflict data: {e}")
        
        return {'template_used': None}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of template system"""
        try:
            async with get_db_connection_context() as conn:
                # Get most used templates
                popular_templates = await conn.fetch("""
                    SELECT template_id, COUNT(*) as usage_count
                    FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND template_id IS NOT NULL
                    GROUP BY template_id
                    ORDER BY usage_count DESC
                    LIMIT 3
                """, self.user_id, self.conversation_id)
            
            return {
                'popular_templates': [
                    {'id': t['template_id'], 'usage': t['usage_count']}
                    for t in popular_templates
                ],
                'cache_size': len(self._template_cache)
            }
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return {'error': str(e)}
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if template system is relevant to scene"""
        # Templates are relevant when creating new conflicts
        if scene_context.get('creating_conflict'):
            return True
        
        # Or when scene calls for specific conflict types
        activity = scene_context.get('activity', '')
        if any(keyword in activity.lower() for keyword in ['dispute', 'challenge', 'competition']):
            return True
        
        return False
    
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
    
    # ========== Template Management Methods ==========
    
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
        
        try:
            response = await Runner.run(self.template_creator, prompt)
            response_text = extract_runner_response(response)
            data = json.loads(response_text)
            
            # Store template
            async with get_db_connection_context() as conn:
                template_id = await conn.fetchval("""
                    INSERT INTO conflict_templates
                    (user_id, conversation_id, category, name, base_structure, 
                     variable_elements, contextual_modifiers, complexity_min, complexity_max)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING template_id
                """, self.user_id, self.conversation_id, category.value, data['name'],
                json.dumps(data['base_structure']),
                json.dumps(data['variable_elements']),
                json.dumps(data['contextual_modifiers']),
                data['complexity_range']['minimum'],
                data['complexity_range']['maximum'])
            
            template = ConflictTemplate(
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
            
            # Cache template
            self._template_cache[template_id] = template
            
            return template
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise
    
    async def generate_conflict_from_template(
        self,
        template_id: int,
        context: Dict[str, Any]
    ) -> GeneratedConflict:
        """Generate a unique conflict from a template"""
        
        try:
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Generate variation
            variation = await self._generate_variation(template, context)
            
            # Adapt to context
            adapted = await self._adapt_to_context(variation, context)
            
            # Add unique elements
            unique = await self._add_unique_elements(adapted, context)
            
            # Generate hooks
            hooks = await self._generate_narrative_hooks(unique, context)
            
            # Create the conflict through synthesizer
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
            
        except Exception as e:
            logger.error(f"Error generating from template: {e}")
            raise
    
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
        response_text = extract_runner_response(response)
        return json.loads(response_text)
    
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
        response_text = extract_runner_response(response)
        adapted = variation.copy()
        adapted['context_adaptation'] = json.loads(response_text)
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
        response_text = extract_runner_response(response)
        unique_data = json.loads(response_text)
        
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
        response_text = extract_runner_response(response)
        data = json.loads(response_text)
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
        
        # Create conflict through synthesizer
        if self.synthesizer:
            synth = self.synthesizer()
            if synth:
                result = await synth.create_conflict(
                    template.category.value,
                    {
                        'template_id': template.template_id,
                        'generation_data': generation_data,
                        'hooks': hooks,
                        'complexity': complexity
                    }
                )
                return result.get('conflict_id', 0)
        
        # Fallback: create directly
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
    
    async def _get_template(self, template_id: int) -> Optional[ConflictTemplate]:
        """Get template from cache or database"""
        
        if template_id in self._template_cache:
            return self._template_cache[template_id]
        
        try:
            async with get_db_connection_context() as conn:
                template_data = await conn.fetchrow("""
                    SELECT * FROM conflict_templates WHERE template_id = $1
                """, template_id)
            
            if template_data:
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
                self._template_cache[template_id] = template
                return template
        except Exception as e:
            logger.error(f"Error getting template: {e}")
        
        return None
    
    async def _select_template(self, conflict_type: str) -> Optional[int]:
        """Select appropriate template for conflict type"""
        
        try:
            async with get_db_connection_context() as conn:
                template = await conn.fetchrow("""
                    SELECT template_id FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                    AND category LIKE $3
                    ORDER BY RANDOM()
                    LIMIT 1
                """, self.user_id, self.conversation_id, f"%{conflict_type}%")
            
            return template['template_id'] if template else None
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return None
    
    async def _evolve_random_template(self) -> bool:
        """Evolve a random template based on usage"""
        
        try:
            async with get_db_connection_context() as conn:
                # Get a template with usage
                template = await conn.fetchrow("""
                    SELECT t.*, COUNT(c.conflict_id) as usage_count
                    FROM conflict_templates t
                    LEFT JOIN Conflicts c ON t.template_id = c.template_id
                    WHERE t.user_id = $1 AND t.conversation_id = $2
                    GROUP BY t.template_id
                    HAVING COUNT(c.conflict_id) > 0
                    ORDER BY RANDOM()
                    LIMIT 1
                """, self.user_id, self.conversation_id)
            
            if template:
                # Simple evolution: adjust complexity range based on success
                new_min = max(0.1, template['complexity_min'] - 0.05)
                new_max = min(1.0, template['complexity_max'] + 0.05)
                
                await conn.execute("""
                    UPDATE conflict_templates
                    SET complexity_min = $1, complexity_max = $2,
                        last_evolved = CURRENT_TIMESTAMP
                    WHERE template_id = $3
                """, new_min, new_max, template['template_id'])
                
                return True
        except Exception as e:
            logger.error(f"Error evolving template: {e}")
        
        return False


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def generate_templated_conflict(
    ctx: RunContextWrapper,
    category: str,
    context: TemplateContextDTO,
) -> GenerateTemplatedConflictResponse:
    """Generate a conflict from a template category through synthesizer"""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import get_synthesizer
    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Flatten: pass a single context dict (no nested "context" field)
    merged_context: Dict = {
        'use_template': True,
        'template_category': category,
        **dict(context or {})
    }

    result = await synthesizer.create_conflict(
        f"template_{category}",
        merged_context,
    )

    # Coerce into strict response
    return {
        'conflict_id': int(result.get('conflict_id', 0) or 0),
        'status': str(result.get('status', 'created')),
        'conflict_type': str(result.get('conflict_type', f"template_{category}")),
        'template_used': int(result.get('template_used', 0) or 0),
        'narrative_hooks': [str(h) for h in (result.get('narrative_hooks') or [])],
        'message': str(result.get('message', "")),
        'error': "",
    }


@function_tool
async def create_custom_template(
    ctx: RunContextWrapper,
    category: str,
    concept: str,
) -> CreateTemplateResponse:
    """Create a custom conflict template via the TEMPLATE subsystem"""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    from logic.conflict_system.conflict_synthesizer import (
        get_synthesizer, SystemEvent, EventType, SubsystemType
    )
    synthesizer = await get_synthesizer(user_id, conversation_id)

    # Ask TEMPLATE subsystem to create a template (no direct _subsystems access)
    evt = SystemEvent(
        event_id=f"create_template_{category}_{datetime.now().timestamp()}",
        event_type=EventType.TEMPLATE_GENERATED,
        source_subsystem=SubsystemType.TEMPLATE,
        payload={'request': 'create_template', 'category': category, 'concept': concept},
        target_subsystems={SubsystemType.TEMPLATE},
        requires_response=True,
        priority=3,
    )

    template_id = 0
    name = ""
    cat = category
    variable_count = 0
    comp_min = 0.0
    comp_max = 0.0
    error = "Template subsystem did not respond"

    responses = await synthesizer.emit_event(evt)
    if responses:
        for r in responses:
            if r.subsystem == SubsystemType.TEMPLATE:
                data = r.data or {}
                t = data.get('template') or data  # allow either shape
                template_id = int(t.get('template_id', 0) or 0)
                name = str(t.get('name', "") or "")
                cat = str(t.get('category', category) or category)
                # variable elements could be a list or count
                ve = t.get('variable_elements')
                if isinstance(ve, list):
                    variable_count = len(ve)
                else:
                    variable_count = int(t.get('variable_count', 0) or 0)
                # complexity range could be pair or dict
                cr = t.get('complexity_range') or {}
                if isinstance(cr, (list, tuple)) and len(cr) == 2:
                    comp_min = float(cr[0] or 0.0)
                    comp_max = float(cr[1] or 0.0)
                else:
                    comp_min = float(cr.get('min', 0.0) if isinstance(cr, dict) else 0.0)
                    comp_max = float(cr.get('max', 0.0) if isinstance(cr, dict) else 0.0)
                error = ""

    return {
        'template_id': template_id,
        'name': name,
        'category': cat,
        'variable_count': variable_count,
        'complexity_min': comp_min,
        'complexity_max': comp_max,
        'error': error,
    }
