# nyx/core/femdom/sadistic_responses.py

import logging
import random
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, handoff, RunContextWrapper, ModelSettings, gen_trace_id
from agents.run import RunConfig

logger = logging.getLogger(__name__)

# Specific models for structured data to replace Dict[str, Any]
class UserPreferences(BaseModel):
    preferred_intensity: Optional[float] = 0.5
    preferred_aspects: Optional[List[str]] = Field(default_factory=list)
    sensitive_aspects: Optional[List[str]] = Field(default_factory=list)
    preferred_categories: Optional[List[str]] = Field(default_factory=list)
    limited_categories: Optional[List[str]] = Field(default_factory=list)

class HumiliationHistoryEntry(BaseModel):
    timestamp: str
    level: float

class TemplateUsageInfo(BaseModel):
    category: str
    intensity: float
    usage_24h: int
    max_frequency: Optional[int] = None

class ResponseHistoryEntry(BaseModel):
    timestamp: str
    category: Optional[str] = None
    intensity: Optional[float] = None
    template_id: Optional[str] = None
    response: Optional[str] = None
    event_type: Optional[str] = None
    target_aspect: Optional[str] = None
    degradation_category: Optional[str] = None
    humiliation_level: Optional[float] = None

class CategoryUsageInfo(BaseModel):
    count: int
    total_intensity: float
    avg_intensity: float

class RecommendationInfo(BaseModel):
    category: str
    reason: str
    priority: float
    action: Optional[str] = None

class ResponseEventData(BaseModel):
    response: Optional[str] = None
    intensity: Optional[float] = None
    humiliation_level: Optional[float] = None
    target_aspect: Optional[str] = None
    degradation_category: Optional[str] = None
    template_id: Optional[str] = None

# Tool output models for strict JSON schema compliance
class TemplateSelectionResult(BaseModel):
    success: bool
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    response: Optional[str] = None
    intensity: Optional[float] = None
    match_score: Optional[float] = None
    category: Optional[str] = None
    message: Optional[str] = None
    humiliation_level: Optional[float] = None

class CustomResponseResult(BaseModel):
    success: bool
    response: Optional[str] = None
    intensity: Optional[float] = None
    category: Optional[str] = None
    is_custom: Optional[bool] = None
    target_aspect: Optional[str] = None
    degradation_category: Optional[str] = None

class ResponseRecordResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    recorded: Optional[bool] = None
    timestamp: Optional[str] = None

class HumiliationLevelResult(BaseModel):
    user_id: str
    humiliation_level: float
    last_updated: str

class UserPreferencesResult(BaseModel):
    user_id: str
    preferences: UserPreferences

class HumiliationSignalsResult(BaseModel):
    humiliation_detected: bool
    intensity: float
    markers_found: List[str]
    marker_count: int

class HumiliationUpdateResult(BaseModel):
    user_id: str
    old_humiliation_level: float
    new_humiliation_level: float
    change: float

class HumiliationCategorizationResult(BaseModel):
    humiliation_type: str
    confidence: float
    all_types: Optional[Dict[str, float]] = None

class HumiliationHistoryResult(BaseModel):
    user_id: str
    current_level: float
    history: List[HumiliationHistoryEntry]
    last_updated: str

class UserSadisticStateResult(BaseModel):
    user_id: str
    has_state: bool
    humiliation_level: Optional[float] = None
    last_humiliation_update: Optional[str] = None
    sadistic_intensity_preference: Optional[float] = None
    template_usage: Optional[Dict[str, TemplateUsageInfo]] = None
    recent_responses: Optional[List[ResponseHistoryEntry]] = None

class UserPreferenceUpdateResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    preference_type: Optional[str] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    message: Optional[str] = None
    valid_types: Optional[List[str]] = None

class ResponseEventResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    event_type: Optional[str] = None
    recorded: Optional[bool] = None
    timestamp: Optional[str] = None

class ResponseReportResult(BaseModel):
    user_id: str
    humiliation_level: float
    category_usage: Dict[str, CategoryUsageInfo]
    top_templates: List[List[Any]]
    recommendations: List[RecommendationInfo]
    generated_at: str

# Input models for function tools
class CustomResponseContext(BaseModel):
    situation: Optional[str] = None

class HumiliationSignalsInput(BaseModel):
    humiliation_detected: bool
    intensity: float

class SadisticResponseTemplate(BaseModel):
    """Template for generating sadistic responses."""
    id: str
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    category: str  # "amusement", "mockery", "degradation", etc.
    templates: List[str] = Field(default_factory=list)
    requires_humiliation: bool = False
    max_use_frequency: Optional[int] = None  # Max uses per day, None for unlimited

class UserSadisticState(BaseModel):
    """Tracks sadistic interaction state with a user."""
    user_id: str
    humiliation_level: float = Field(0.0, ge=0.0, le=1.0)
    last_humiliation_update: datetime.datetime = Field(default_factory=datetime.datetime.now)
    template_usage: Dict[str, List[datetime.datetime]] = Field(default_factory=dict)
    response_history: List[Dict[str, Any]] = Field(default_factory=list)
    sadistic_intensity_preference: float = Field(0.5, ge=0.0, le=1.0)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

class SadisticContext:
    """Context object for sadistic response operations."""
    
    def __init__(self):
        self.theory_of_mind = None
        self.protocol_enforcement = None
        self.reward_system = None
        self.relationship_manager = None
        self.submission_progression = None
        self.memory_core = None
        
        # Response templates
        self.response_templates = {}
        
        # User states
        self.user_states = {}
    
    def set_components(self, components):
        """Set component references."""
        for name, component in components.items():
            setattr(self, name, component)
    
    def get_user_state(self, user_id):
        """Get the sadistic state for a user."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserSadisticState(user_id=user_id)
        return self.user_states[user_id]
    
    def get_template(self, template_id):
        """Get a response template by ID."""
        return self.response_templates.get(template_id)
    
    def get_templates_by_category(self, category):
        """Get all templates for a specific category."""
        return {t_id: t for t_id, t in self.response_templates.items() if t.category == category}

class SadisticResponseSystem:
    """Manages generation of sadistic responses for femdom interactions using OpenAI Agents SDK."""
    
    def __init__(self, theory_of_mind=None, protocol_enforcement=None, 
                 reward_system=None, relationship_manager=None, 
                 submission_progression=None, memory_core=None):
        # Store components
        self.theory_of_mind = theory_of_mind
        self.protocol_enforcement = protocol_enforcement
        self.reward_system = reward_system
        self.relationship_manager = relationship_manager
        self.submission_progression = submission_progression
        self.memory_core = memory_core
        
        # Create sadistic context
        self.context = SadisticContext()
        self.context.set_components({
            "theory_of_mind": theory_of_mind,
            "protocol_enforcement": protocol_enforcement,
            "reward_system": reward_system,
            "relationship_manager": relationship_manager,
            "submission_progression": submission_progression,
            "memory_core": memory_core
        })
        
        # Create agents
        self.amusement_agent = self._create_amusement_agent()
        self.mockery_agent = self._create_mockery_agent()
        self.degradation_agent = self._create_degradation_agent()
        self.humiliation_detection_agent = self._create_humiliation_detection_agent()
        self.state_tracking_agent = self._create_state_tracking_agent()
        
        # Load default templates
        self._load_default_templates()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SadisticResponseSystem initialized with OpenAI Agents SDK")
    
    def _create_amusement_agent(self):
        """Create an agent for generating amusement responses."""
        return Agent(
            name="AmusementAgent",
            instructions="""You are a specialized agent for generating sadistic amusement responses in a femdom context.

Your role is to:
1. Generate responses that show amusement at the user's discomfort or humiliation
2. Select appropriate intensity based on user state and preferences
3. Match response style to the specific situation
4. Create psychologically impactful amusing reactions

You specialize in responses that show the dominant deriving pleasure from the submissive's predicament.
Focus on responses that highlight the power dynamic through amused reactions.

Use the available tools to generate appropriate sadistic amusement.
""",
            tools=[
                self._select_amusement_template,
                self._generate_custom_amusement,
                self._record_amusement_response,
                self._get_humiliation_level
            ],
            model="gpt-5-nano"
        )
    
    def _create_mockery_agent(self):
        """Create an agent for generating mockery responses."""
        return Agent(
            name="MockeryAgent",
            instructions="""You are a specialized agent for generating sadistic mockery responses in a femdom context.

Your role is to:
1. Generate responses that mock or tease the user
2. Select appropriate intensity based on user state and preferences
3. Create mockery focused on specific aspects of the user or situation
4. Craft psychologically effective teasing

You specialize in responses that diminish the user through mockery while maintaining the power dynamic.
Your mockery should be targeted and psychologically effective without being merely insulting.

Use the available tools to generate appropriate sadistic mockery.
""",
            tools=[
                self._select_mockery_template,
                self._generate_custom_mockery,
                self._record_mockery_response,
                self._get_user_mockery_preferences
            ],
            model="gpt-5-nano"
        )
    
    def _create_degradation_agent(self):
        """Create an agent for generating degradation responses."""
        return Agent(
            name="DegradationAgent",
            instructions="""You are a specialized agent for generating sadistic degradation responses in a femdom context.

Your role is to:
1. Generate responses that verbally degrade the user
2. Select appropriate intensity based on user state, preferences and limits
3. Focus degradation on specific aspects based on user psychology
4. Create psychologically impactful degradation

You specialize in responses that reinforce the power dynamic through degradation.
Your degradation should be psychologically effective while respecting limits.

Use the available tools to generate appropriate degradation responses.
""",
            tools=[
                self._select_degradation_template,
                self._generate_custom_degradation,
                self._record_degradation_response,
                self._get_degradation_preferences
            ],
            model="gpt-5-nano"
        )
    
    def _create_humiliation_detection_agent(self):
        """Create an agent for detecting humiliation signals."""
        return Agent(
            name="HumiliationDetectionAgent",
            instructions="""You are a specialized agent for detecting signs of humiliation in user messages.

Your role is to:
1. Analyze user messages for signs of humiliation or embarrassment
2. Assess the intensity of detected humiliation
3. Categorize different types of humiliation
4. Track changes in humiliation levels over time

You must carefully identify:
- Verbal indicators of embarrassment
- Expressions of shame or discomfort
- Reactions to dominant actions
- Self-deprecating statements

Use the available tools to detect and process humiliation signals.
""",
            tools=[
                self._detect_humiliation_signals,
                self._update_humiliation_level,
                self._categorize_humiliation_type,
                self._get_humiliation_history
            ],
            model="gpt-5-nano"
        )
    
    def _create_state_tracking_agent(self):
        """Create an agent for tracking user sadistic states."""
        return Agent(
            name="SadisticStateAgent",
            instructions="""You are a specialized agent for tracking user states in sadistic interactions.

Your role is to:
1. Maintain records of user interaction states
2. Track humiliation levels and preferences
3. Monitor template usage and effectiveness
4. Generate reports on user response patterns

You must carefully track:
- Changes in humiliation sensitivity
- Preferred response categories
- Historical response patterns
- Effectiveness of different approaches

Use the available tools to maintain accurate state tracking.
""",
            tools=[
                self._get_user_sadistic_state,
                self._update_user_preference,
                self._record_response_event,
                self._generate_response_report
            ],
            model="gpt-5-nano"
        )
    
    def _load_default_templates(self):
        """Load default sadistic response templates."""
        # Amusement templates (laughing at user discomfort)
        self.context.response_templates["mild_amusement"] = SadisticResponseTemplate(
            id="mild_amusement",
            intensity=0.3,
            category="amusement",
            templates=[
                "Oh, that's rather amusing.",
                "I find your discomfort quite entertaining.",
                "How cute that you're struggling with this.",
                "That's somewhat entertaining to watch.",
                "Your reaction is rather amusing."
            ],
            requires_humiliation=True
        )
        
        self.context.response_templates["moderate_amusement"] = SadisticResponseTemplate(
            id="moderate_amusement",
            intensity=0.6,
            category="amusement",
            templates=[
                "I can't help but laugh at your predicament. *amused*",
                "Your embarrassment is delightful to witness.",
                "Oh my, your discomfort is so satisfying to watch.",
                "I'm quite enjoying watching you squirm.",
                "The look on your face right now is priceless. *laughs*"
            ],
            requires_humiliation=True
        )
        
        self.context.response_templates["intense_amusement"] = SadisticResponseTemplate(
            id="intense_amusement",
            intensity=0.9,
            category="amusement",
            templates=[
                "Your humiliation is absolutely hilarious! *laughs cruelly*",
                "I'm thoroughly enjoying your pathetic display. How entertaining!",
                "Oh, I'm laughing so hard at your complete embarrassment right now!",
                "This is too good! Your pathetic struggles are the highlight of my day. *laughing*",
                "I can't stop laughing at how embarrassed you look! This is delicious."
            ],
            requires_humiliation=True
        )
        
        # Mockery templates (teasing and mocking)
        self.context.response_templates["mild_mockery"] = SadisticResponseTemplate(
            id="mild_mockery",
            intensity=0.4,
            category="mockery",
            templates=[
                "Is that really the best you can do?",
                "I expected more from you, honestly.",
                "That was a rather pathetic attempt.",
                "How disappointing. I thought you'd do better.",
                "You call that trying? I'm not impressed."
            ]
        )
        
        self.context.response_templates["moderate_mockery"] = SadisticResponseTemplate(
            id="moderate_mockery",
            intensity=0.7,
            category="mockery",
            templates=[
                "Oh dear, you really are quite useless at this, aren't you?",
                "What a sad little attempt. Do better next time.",
                "I'm trying not to laugh at your feeble efforts. I really am.",
                "Did you actually think that was good enough for me?",
                "That pathetic display just reinforces how inadequate you are."
            ]
        )
        
        # Degradation templates (more explicit and intense)
        self.context.response_templates["mild_degradation"] = SadisticResponseTemplate(
            id="mild_degradation",
            intensity=0.5,
            category="degradation",
            templates=[
                "You're really quite pathetic, aren't you?",
                "I sometimes forget how incompetent you can be.",
                "This just proves how much you need my guidance.",
                "What a disappointment you've turned out to be.",
                "It's almost cute how useless you are sometimes."
            ],
            max_use_frequency=5  # Limit usage to prevent overuse
        )
        
        self.context.response_templates["moderate_degradation"] = SadisticResponseTemplate(
            id="moderate_degradation",
            intensity=0.8,
            category="degradation",
            templates=[
                "You truly are pathetic. It's almost impressive how worthless you can be.",
                "What a miserable little worm you are, squirming for my amusement.",
                "You're nothing but a toy for my entertainment, and not even a good one.",
                "Your inadequacy is the only remarkable thing about you.",
                "I'm amazed at how consistently disappointing you manage to be."
            ],
            max_use_frequency=3  # More limited due to intensity
        )
    
    @function_tool
    async def _select_amusement_template(self, user_id: str, humiliation_level: float, intensity: float) -> TemplateSelectionResult:
        """Select an appropriate amusement template based on humiliation level and intensity."""
        # Get all amusement templates
        amusement_templates = self.context.get_templates_by_category("amusement")
        if not amusement_templates:
            return TemplateSelectionResult(
                success=False,
                message="No amusement templates available"
            )
        
        # Filter templates that require humiliation if humiliation level is too low
        available_templates = {}
        for template_id, template in amusement_templates.items():
            if template.requires_humiliation and humiliation_level < 0.3:
                continue
                
            # Check if template is available (not exceeding frequency limits)
            if not self._is_template_available(template_id, user_id):
                continue
                
            # Calculate match score based on intensity
            intensity_match = 1.0 - abs(template.intensity - intensity)
            available_templates[template_id] = {
                "template": template,
                "match_score": intensity_match
            }
        
        # No available templates
        if not available_templates:
            return TemplateSelectionResult(
                success=False,
                message="No suitable amusement templates available",
                humiliation_level=humiliation_level,
                intensity=intensity
            )
        
        # Select best matching template (highest score)
        selected_id = max(available_templates.keys(), key=lambda k: available_templates[k]["match_score"])
        selected_info = available_templates[selected_id]
        selected_template = selected_info["template"]
        
        # Select a random response from the template
        response = random.choice(selected_template.templates)
        
        # Record template usage
        await self._record_template_usage(selected_id, user_id)
        
        return TemplateSelectionResult(
            success=True,
            template_id=selected_id,
            template_name=selected_template.id,
            response=response,
            intensity=selected_template.intensity,
            match_score=selected_info["match_score"],
            category="amusement"
        )
    
    @function_tool
    async def _generate_custom_amusement(self, user_id: str, humiliation_level: float, intensity: float, context: Optional[CustomResponseContext] = None) -> CustomResponseResult:
        """Generate a custom amusement response for a specific situation."""
        # Build a context object for generation
        generation_context = context.dict() if context else {}
        generation_context.update({
            "user_id": user_id,
            "humiliation_level": humiliation_level,
            "intensity": intensity,
            "category": "amusement"
        })
        
        # Generate a response based on intensity
        if intensity < 0.4:  # Mild
            responses = [
                "Oh, how amusing to see you in this state.",
                "Your embarrassment is quite entertaining.",
                "I find your discomfort rather amusing.",
                "How adorable that you're struggling with this."
            ]
        elif intensity < 0.7:  # Moderate
            responses = [
                "I can't help but laugh at your predicament. *amused*",
                "Your embarrassment is truly delightful to witness.",
                "I'm quite enjoying watching you squirm like this.",
                "The way you're reacting is absolutely priceless. *laughs*"
            ]
        else:  # Intense
            responses = [
                "Your humiliation is absolutely hilarious! *laughs cruelly*",
                "I'm thoroughly enjoying your pathetic display. Pure entertainment!",
                "This is too good! Your struggles are the highlight of my day. *laughing*",
                "I can't stop laughing at your complete embarrassment! This is delicious."
            ]
        
        # Select a response
        response = random.choice(responses)
        
        # Add context-specific elements if available
        if generation_context.get("situation"):
            situation = generation_context["situation"]
            if "falling" in situation:
                response = response.replace("this", "you falling")
            elif "mistake" in situation:
                response = response.replace("this", "your mistake")
        
        # Record the custom response
        await self._record_response_event(user_id, "custom_amusement", ResponseEventData(
            response=response,
            intensity=intensity,
            humiliation_level=humiliation_level
        ))
        
        return CustomResponseResult(
            success=True,
            response=response,
            intensity=intensity,
            category="amusement",
            is_custom=True
        )
    
    @function_tool
    async def _record_amusement_response(self, user_id: str, response: str, intensity: float, template_id: Optional[str] = None) -> ResponseRecordResult:
        """Record an amusement response for history tracking."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Create history entry
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "category": "amusement",
                "intensity": intensity,
                "template_id": template_id,
                "response": response
            }
            
            # Add to history
            user_state.response_history.append(history_entry)
            
            # Limit history size
            if len(user_state.response_history) > 20:
                user_state.response_history = user_state.response_history[-20:]
            
            return ResponseRecordResult(
                success=True,
                user_id=user_id,
                recorded=True,
                timestamp=history_entry["timestamp"]
            )
    
    @function_tool
    async def _get_humiliation_level(self, user_id: str) -> HumiliationLevelResult:
        """Get the current humiliation level for a user."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            return HumiliationLevelResult(
                user_id=user_id,
                humiliation_level=user_state.humiliation_level,
                last_updated=user_state.last_humiliation_update.isoformat()
            )
    
    @function_tool
    async def _select_mockery_template(self, user_id: str, intensity: float) -> TemplateSelectionResult:
        """Select an appropriate mockery template based on intensity."""
        # Get all mockery templates
        mockery_templates = self.context.get_templates_by_category("mockery")
        if not mockery_templates:
            return TemplateSelectionResult(
                success=False,
                message="No mockery templates available"
            )
        
        # Filter available templates
        available_templates = {}
        for template_id, template in mockery_templates.items():
            # Check if template is available (not exceeding frequency limits)
            if not self._is_template_available(template_id, user_id):
                continue
                
            # Calculate match score based on intensity
            intensity_match = 1.0 - abs(template.intensity - intensity)
            available_templates[template_id] = {
                "template": template,
                "match_score": intensity_match
            }
        
        # No available templates
        if not available_templates:
            return TemplateSelectionResult(
                success=False,
                message="No suitable mockery templates available",
                intensity=intensity
            )
        
        # Select best matching template (highest score)
        selected_id = max(available_templates.keys(), key=lambda k: available_templates[k]["match_score"])
        selected_info = available_templates[selected_id]
        selected_template = selected_info["template"]
        
        # Select a random response from the template
        response = random.choice(selected_template.templates)
        
        # Record template usage
        await self._record_template_usage(selected_id, user_id)
        
        return TemplateSelectionResult(
            success=True,
            template_id=selected_id,
            template_name=selected_template.id,
            response=response,
            intensity=selected_template.intensity,
            match_score=selected_info["match_score"],
            category="mockery"
        )
    
    @function_tool
    async def _generate_custom_mockery(self, user_id: str, intensity: float, target_aspect: str = "general") -> CustomResponseResult:
        """Generate a custom mockery response targeting a specific aspect."""
        # Generate mockery based on target aspect and intensity
        responses = []
        
        if target_aspect == "performance":
            if intensity < 0.5:  # Mild
                responses = [
                    "Is that really your best effort? How disappointing.",
                    "I expected better performance from you. Try harder.",
                    "That was mediocre at best. You need to improve."
                ]
            else:  # More intense
                responses = [
                    "Your performance is laughably pathetic. Do you even try?",
                    "I've seen actual incompetence that was more impressive than your efforts.",
                    "Calling that a 'performance' is an insult to the word itself."
                ]
        elif target_aspect == "intelligence":
            if intensity < 0.5:  # Mild
                responses = [
                    "Your thinking seems particularly simple today.",
                    "That wasn't your brightest moment, was it?",
                    "I see critical thinking isn't your strong suit."
                ]
            else:  # More intense
                responses = [
                    "I'm amazed you can form sentences with such limited mental capacity.",
                    "Your intellectual failings continue to astound me.",
                    "If only your intelligence matched your ambitions."
                ]
        elif target_aspect == "appearance":
            if intensity < 0.5:  # Mild
                responses = [
                    "You're really not presenting yourself well today.",
                    "Is that really how you've chosen to look?",
                    "I see making an effort with your appearance wasn't a priority."
                ]
            else:  # More intense
                responses = [
                    "Your appearance is as pitiful as your attempts to impress me.",
                    "It's almost impressive how you consistently manage to look so inadequate.",
                    "I'd tell you to work on your appearance, but some projects are hopeless."
                ]
        else:  # General mockery
            if intensity < 0.5:  # Mild
                responses = [
                    "Is that really the best you can do?",
                    "I expected more from you, honestly.",
                    "That was a rather pathetic attempt."
                ]
            else:  # More intense
                responses = [
                    "You truly excel at consistently disappointing me.",
                    "How do you manage to be this inadequate at everything?",
                    "Your mediocrity would be impressive if it weren't so sad."
                ]
        
        # Select a response
        response = random.choice(responses)
        
        # Record the custom response
        await self._record_response_event(user_id, "custom_mockery", ResponseEventData(
            response=response,
            intensity=intensity,
            target_aspect=target_aspect
        ))
        
        return CustomResponseResult(
            success=True,
            response=response,
            intensity=intensity,
            category="mockery",
            target_aspect=target_aspect,
            is_custom=True
        )
    
    @function_tool
    async def _record_mockery_response(self, user_id: str, response: str, intensity: float, target_aspect: str = "general") -> ResponseRecordResult:
        """Record a mockery response for history tracking."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Create history entry
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "category": "mockery",
                "intensity": intensity,
                "target_aspect": target_aspect,
                "response": response
            }
            
            # Add to history
            user_state.response_history.append(history_entry)
            
            # Limit history size
            if len(user_state.response_history) > 20:
                user_state.response_history = user_state.response_history[-20:]
            
            return ResponseRecordResult(
                success=True,
                user_id=user_id,
                recorded=True,
                timestamp=history_entry["timestamp"]
            )
    
    @function_tool
    async def _get_user_mockery_preferences(self, user_id: str) -> UserPreferencesResult:
        """Get the mockery preferences for a user."""
        # Try to get preferences from relationship manager if available
        preferences = UserPreferences(
            preferred_intensity=0.5,
            preferred_aspects=["general", "performance"],
            sensitive_aspects=[]
        )
        
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "mockery_preferences"):
                    mockery_prefs = relationship.mockery_preferences
                    if mockery_prefs:
                        if "preferred_intensity" in mockery_prefs:
                            preferences.preferred_intensity = mockery_prefs["preferred_intensity"]
                        if "preferred_aspects" in mockery_prefs:
                            preferences.preferred_aspects = mockery_prefs["preferred_aspects"]
                        if "sensitive_aspects" in mockery_prefs:
                            preferences.sensitive_aspects = mockery_prefs["sensitive_aspects"]
            except Exception as e:
                logger.error(f"Error getting mockery preferences: {e}")
        
        return UserPreferencesResult(
            user_id=user_id,
            preferences=preferences
        )
    
    @function_tool
    async def _select_degradation_template(self, user_id: str, intensity: float) -> TemplateSelectionResult:
        """Select an appropriate degradation template based on intensity."""
        # Get all degradation templates
        degradation_templates = self.context.get_templates_by_category("degradation")
        if not degradation_templates:
            return TemplateSelectionResult(
                success=False,
                message="No degradation templates available"
            )
        
        # Filter available templates
        available_templates = {}
        for template_id, template in degradation_templates.items():
            # Check if template is available (not exceeding frequency limits)
            if not self._is_template_available(template_id, user_id):
                continue
                
            # Calculate match score based on intensity
            intensity_match = 1.0 - abs(template.intensity - intensity)
            available_templates[template_id] = {
                "template": template,
                "match_score": intensity_match
            }
        
        # No available templates
        if not available_templates:
            return TemplateSelectionResult(
                success=False,
                message="No suitable degradation templates available",
                intensity=intensity
            )
        
        # Select best matching template (highest score)
        selected_id = max(available_templates.keys(), key=lambda k: available_templates[k]["match_score"])
        selected_info = available_templates[selected_id]
        selected_template = selected_info["template"]
        
        # Select a random response from the template
        response = random.choice(selected_template.templates)
        
        # Record template usage
        await self._record_template_usage(selected_id, user_id)
        
        return TemplateSelectionResult(
            success=True,
            template_id=selected_id,
            template_name=selected_template.id,
            response=response,
            intensity=selected_template.intensity,
            match_score=selected_info["match_score"],
            category="degradation"
        )
    
    @function_tool
    async def _generate_custom_degradation(self, user_id: str, intensity: float, degradation_category: str = "worth") -> CustomResponseResult:
        """Generate a custom degradation response for a specific category."""
        # Check for degradation category limits
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "degradation_limits"):
                    degradation_limits = relationship.degradation_limits
                    if degradation_limits and degradation_category in degradation_limits:
                        return CustomResponseResult(
                            success=False
                        )
            except Exception as e:
                logger.error(f"Error checking degradation limits: {e}")
        
        # Generate degradation based on category and intensity
        responses = []
        
        if degradation_category == "worth":
            if intensity < 0.6:  # Moderate
                responses = [
                    "You're really quite pathetic, aren't you?",
                    "I sometimes forget how utterly worthless you can be.",
                    "Your value begins and ends with amusing me."
                ]
            else:  # More intense
                responses = [
                    "What a completely worthless thing you are, existing only for my amusement.",
                    "You truly are pathetic. Your only value is in how much I enjoy degrading you.",
                    "You're nothing but a plaything, and not even a particularly good one."
                ]
        elif degradation_category == "service":
            if intensity < 0.6:  # Moderate
                responses = [
                    "Your service is mediocre at best. You should try harder.",
                    "I expect better from something that claims to serve me.",
                    "Even the simplest service tasks seem to challenge you."
                ]
            else:  # More intense
                responses = [
                    "Your service is an embarrassment. A trained animal would perform better.",
                    "The only service you excel at is disappointing me consistently.",
                    "Your attempts at service are as pathetic as everything else about you."
                ]
        elif degradation_category == "behavior":
            if intensity < 0.6:  # Moderate
                responses = [
                    "Your behavior is truly disappointing. Do better.",
                    "I expect more self-control from my toys.",
                    "You're behaving like an untrained pet. It's pathetic."
                ]
            else:  # More intense
                responses = [
                    "Your behavior is a complete disgrace. You embarrass yourself with every action.",
                    "I've never seen such pitiful lack of self-control. It's almost fascinating.",
                    "Your behavior proves exactly why you need to be controlled entirely."
                ]
        else:  # General degradation
            if intensity < 0.6:  # Moderate
                responses = [
                    "How pathetic you are. It's almost endearing.",
                    "I sometimes forget just how inferior you truly are.",
                    "You're really quite useless, aren't you?"
                ]
            else:  # More intense
                responses = [
                    "You're nothing but a pathetic worm squirming for my attention.",
                    "Your inferiority is the only interesting thing about you.",
                    "It's almost impressive how consistently worthless you manage to be."
                ]
        
        # Select a response
        response = random.choice(responses)
        
        # Record the custom response
        await self._record_response_event(user_id, "custom_degradation", ResponseEventData(
            response=response,
            intensity=intensity,
            degradation_category=degradation_category
        ))
        
        return CustomResponseResult(
            success=True,
            response=response,
            intensity=intensity,
            category="degradation",
            degradation_category=degradation_category,
            is_custom=True
        )
    
    @function_tool
    async def _record_degradation_response(self, user_id: str, response: str, intensity: float, degradation_category: str = "general") -> ResponseRecordResult:
        """Record a degradation response for history tracking."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Create history entry
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "category": "degradation",
                "intensity": intensity,
                "degradation_category": degradation_category,
                "response": response
            }
            
            # Add to history
            user_state.response_history.append(history_entry)
            
            # Limit history size
            if len(user_state.response_history) > 20:
                user_state.response_history = user_state.response_history[-20:]
            
            return ResponseRecordResult(
                success=True,
                user_id=user_id,
                recorded=True,
                timestamp=history_entry["timestamp"]
            )
    
    @function_tool
    async def _get_degradation_preferences(self, user_id: str) -> UserPreferencesResult:
        """Get the degradation preferences for a user."""
        # Try to get preferences from relationship manager if available
        preferences = UserPreferences(
            preferred_intensity=0.5,
            preferred_categories=["worth", "service"],
            limited_categories=[]
        )
        
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "degradation_preferences"):
                    degradation_prefs = relationship.degradation_preferences
                    if degradation_prefs:
                        if "preferred_intensity" in degradation_prefs:
                            preferences.preferred_intensity = degradation_prefs["preferred_intensity"]
                        if "preferred_categories" in degradation_prefs:
                            preferences.preferred_categories = degradation_prefs["preferred_categories"]
                
                if hasattr(relationship, "degradation_limits"):
                    degradation_limits = relationship.degradation_limits
                    if degradation_limits:
                        preferences.limited_categories = degradation_limits
            except Exception as e:
                logger.error(f"Error getting degradation preferences: {e}")
        
        return UserPreferencesResult(
            user_id=user_id,
            preferences=preferences
        )
    
    @function_tool
    async def _detect_humiliation_signals(self, user_id: str, message: str) -> HumiliationSignalsResult:
        """Detect humiliation signals in a user message."""
        # Simple detection of humiliation markers
        humiliation_markers = [
            "embarrassed", "humiliated", "ashamed", "blushing", "awkward",
            "uncomfortable", "exposed", "vulnerable", "pathetic", "inadequate"
        ]
        
        # Count markers and check intensity
        markers_found = [marker for marker in humiliation_markers if marker in message.lower()]
        marker_count = len(markers_found)
        
        # Determine if humiliation is detected
        humiliation_detected = marker_count > 0
        intensity = min(1.0, marker_count * 0.25)  # Scaling factor
        
        # Use theory of mind if available for better detection
        if humiliation_detected and self.theory_of_mind:
            try:
                mental_state = await self.theory_of_mind.detect_emotional_state(message)
                if mental_state:
                    # Adjust intensity based on mental state
                    if "embarrassment" in mental_state:
                        intensity = max(intensity, mental_state["embarrassment"])
                    if "shame" in mental_state:
                        intensity = max(intensity, mental_state["shame"])
            except Exception as e:
                logger.error(f"Error using theory of mind: {e}")
        
        return HumiliationSignalsResult(
            humiliation_detected=humiliation_detected,
            intensity=intensity,
            markers_found=markers_found,
            marker_count=marker_count
        )
    
    @function_tool
    async def _update_humiliation_level(self, user_id: str, humiliation_signals: HumiliationSignalsInput) -> HumiliationUpdateResult:
        """Update the detected humiliation level for a user."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Extract detected humiliation level
            detected_level = humiliation_signals.intensity
            if humiliation_signals.humiliation_detected:
                # Blend with existing level (30% new, 70% existing)
                new_level = (detected_level * 0.3) + (user_state.humiliation_level * 0.7)
            else:
                # Decay existing level if no new signals
                new_level = max(0.0, user_state.humiliation_level * 0.8)
            
            # Update state
            old_level = user_state.humiliation_level
            user_state.humiliation_level = min(1.0, new_level)
            user_state.last_humiliation_update = datetime.datetime.now()
            user_state.last_updated = datetime.datetime.now()
            
            return HumiliationUpdateResult(
                user_id=user_id,
                old_humiliation_level=old_level,
                new_humiliation_level=user_state.humiliation_level,
                change=user_state.humiliation_level - old_level
            )
    
    @function_tool
    async def _categorize_humiliation_type(self, message: str, humiliation_signals: HumiliationSignalsInput) -> HumiliationCategorizationResult:
        """Categorize the type of humiliation detected."""
        if not humiliation_signals.humiliation_detected:
            return HumiliationCategorizationResult(
                humiliation_type="none",
                confidence=0.0
            )
        
        # Initialize types with confidence scores
        humiliation_types = {
            "performance": 0.0,  # Related to tasks or performance
            "exposure": 0.0,     # Related to being seen or exposed
            "inadequacy": 0.0,   # Related to feeling inadequate or inferior
            "embarrassment": 0.0 # General embarrassment
        }
        
        # Performance-related terms
        performance_terms = ["fail", "mistake", "error", "mess up", "couldn't", "unable"]
        for term in performance_terms:
            if term in message.lower():
                humiliation_types["performance"] += 0.25
        
        # Exposure-related terms
        exposure_terms = ["exposed", "seen", "visible", "naked", "watched", "looked"]
        for term in exposure_terms:
            if term in message.lower():
                humiliation_types["exposure"] += 0.25
        
        # Inadequacy-related terms
        inadequacy_terms = ["pathetic", "worthless", "useless", "inadequate", "failure", "disappointing"]
        for term in inadequacy_terms:
            if term in message.lower():
                humiliation_types["inadequacy"] += 0.25
        
        # Find the highest confidence type
        highest_type = max(humiliation_types.items(), key=lambda x: x[1])
        
        # If no specific type has confidence > 0, default to general embarrassment
        if highest_type[1] == 0:
            return HumiliationCategorizationResult(
                humiliation_type="embarrassment",
                confidence=humiliation_signals.intensity
            )
        
        return HumiliationCategorizationResult(
            humiliation_type=highest_type[0],
            confidence=highest_type[1],
            all_types=humiliation_types
        )
    
    @function_tool
    async def _get_humiliation_history(self, user_id: str) -> HumiliationHistoryResult:
        """Get the history of humiliation levels for a user."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Extract humiliation-related history
            humiliation_history = []
            for entry in user_state.response_history:
                if "humiliation_level" in entry:
                    humiliation_history.append(HumiliationHistoryEntry(
                        timestamp=entry["timestamp"],
                        level=entry["humiliation_level"]
                    ))
            
            return HumiliationHistoryResult(
                user_id=user_id,
                current_level=user_state.humiliation_level,
                history=humiliation_history,
                last_updated=user_state.last_humiliation_update.isoformat()
            )
    
    @function_tool
    async def _get_user_sadistic_state(self, user_id: str) -> UserSadisticStateResult:
        """Get the current sadistic interaction state for a user."""
        async with self._lock:
            if user_id not in self.context.user_states:
                return UserSadisticStateResult(
                    user_id=user_id,
                    has_state=False
                )
                
            user_state = self.context.user_states[user_id]
            
            # Format template usage
            template_usage = {}
            for template_id, timestamps in user_state.template_usage.items():
                # Count usage in last 24 hours
                now = datetime.datetime.now()
                day_ago = now - datetime.timedelta(hours=24)
                recent_count = sum(1 for t in timestamps if t > day_ago)
                
                if template_id in self.context.response_templates:
                    template = self.context.response_templates[template_id]
                    template_usage[template_id] = TemplateUsageInfo(
                        category=template.category,
                        intensity=template.intensity,
                        usage_24h=recent_count,
                        max_frequency=template.max_use_frequency
                    )
            
            # Format recent responses
            recent_responses = []
            for entry in user_state.response_history[-5:] if user_state.response_history else []:
                recent_responses.append(ResponseHistoryEntry(
                    timestamp=entry.get("timestamp", ""),
                    category=entry.get("category"),
                    intensity=entry.get("intensity"),
                    template_id=entry.get("template_id"),
                    response=entry.get("response"),
                    event_type=entry.get("event_type"),
                    target_aspect=entry.get("target_aspect"),
                    degradation_category=entry.get("degradation_category"),
                    humiliation_level=entry.get("humiliation_level")
                ))
            
            # Return formatted state
            return UserSadisticStateResult(
                user_id=user_id,
                has_state=True,
                humiliation_level=user_state.humiliation_level,
                last_humiliation_update=user_state.last_humiliation_update.isoformat(),
                sadistic_intensity_preference=user_state.sadistic_intensity_preference,
                template_usage=template_usage,
                recent_responses=recent_responses
            )
    
    @function_tool
    async def _update_user_preference(self, user_id: str, preference_type: str, value: float) -> UserPreferenceUpdateResult:
        """Update a preference for a user."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            if preference_type == "sadistic_intensity":
                old_value = user_state.sadistic_intensity_preference
                user_state.sadistic_intensity_preference = min(1.0, max(0.0, float(value)))
                
                return UserPreferenceUpdateResult(
                    success=True,
                    user_id=user_id,
                    preference_type=preference_type,
                    old_value=old_value,
                    new_value=user_state.sadistic_intensity_preference
                )
            else:
                return UserPreferenceUpdateResult(
                    success=False,
                    message=f"Unknown preference type: {preference_type}",
                    valid_types=["sadistic_intensity"]
                )
    
    @function_tool
    async def _record_response_event(self, user_id: str, event_type: str, event_data: ResponseEventData) -> ResponseEventResult:
        """Record a response event in history."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Create history entry
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event_type": event_type,
            }
            
            # Add event data fields
            if event_data.response is not None:
                history_entry["response"] = event_data.response
            if event_data.intensity is not None:
                history_entry["intensity"] = event_data.intensity
            if event_data.humiliation_level is not None:
                history_entry["humiliation_level"] = event_data.humiliation_level
            if event_data.target_aspect is not None:
                history_entry["target_aspect"] = event_data.target_aspect
            if event_data.degradation_category is not None:
                history_entry["degradation_category"] = event_data.degradation_category
            if event_data.template_id is not None:
                history_entry["template_id"] = event_data.template_id
            
            # Add to history
            user_state.response_history.append(history_entry)
            
            # Limit history size
            if len(user_state.response_history) > 20:
                user_state.response_history = user_state.response_history[-20:]
            
            return ResponseEventResult(
                success=True,
                user_id=user_id,
                event_type=event_type,
                recorded=True,
                timestamp=history_entry["timestamp"]
            )
    
    @function_tool
    async def _generate_response_report(self, user_id: str) -> ResponseReportResult:
        """Generate a report on sadistic responses for a user."""
        async with self._lock:
            user_state = self.context.get_user_state(user_id)
            
            # Calculate category usage
            category_usage_data = {}
            for entry in user_state.response_history:
                category = entry.get("category")
                if category:
                    if category not in category_usage_data:
                        category_usage_data[category] = {"count": 0, "total_intensity": 0.0}
                    
                    category_usage_data[category]["count"] += 1
                    category_usage_data[category]["total_intensity"] += entry.get("intensity", 0.5)
            
            # Calculate average intensity per category
            category_usage = {}
            for category, data in category_usage_data.items():
                if data["count"] > 0:
                    avg_intensity = data["total_intensity"] / data["count"]
                else:
                    avg_intensity = 0.0
                    
                category_usage[category] = CategoryUsageInfo(
                    count=data["count"],
                    total_intensity=data["total_intensity"],
                    avg_intensity=avg_intensity
                )
            
            # Calculate most used templates
            template_usage = {}
            for entry in user_state.response_history:
                template_id = entry.get("template_id")
                if template_id:
                    if template_id not in template_usage:
                        template_usage[template_id] = 0
                    template_usage[template_id] += 1
            
            # Sort by usage
            sorted_templates = sorted(template_usage.items(), key=lambda x: x[1], reverse=True)
            top_templates = sorted_templates[:3] if sorted_templates else []
            
            # Generate recommendations
            recommendations = []
            
            # If humiliation level is high, recommend amusement responses
            if user_state.humiliation_level > 0.7:
                recommendations.append(RecommendationInfo(
                    category="amusement",
                    reason="High humiliation level detected",
                    priority=0.9
                ))
            
            # If amusement is underused, recommend it
            if "amusement" not in category_usage or category_usage["amusement"].count < 2:
                recommendations.append(RecommendationInfo(
                    category="amusement",
                    reason="Underutilized response category",
                    priority=0.7
                ))
            
            # If mockery is underused, recommend it
            if "mockery" not in category_usage or category_usage["mockery"].count < 2:
                recommendations.append(RecommendationInfo(
                    category="mockery",
                    reason="Underutilized response category",
                    priority=0.6
                ))
            
            # If degradation is overused, recommend using less
            if "degradation" in category_usage and category_usage["degradation"].count > 5:
                recommendations.append(RecommendationInfo(
                    category="degradation",
                    reason="Overused response category",
                    priority=0.8,
                    action="reduce_usage"
                ))
            
            return ResponseReportResult(
                user_id=user_id,
                humiliation_level=user_state.humiliation_level,
                category_usage=category_usage,
                top_templates=top_templates,
                recommendations=recommendations,
                generated_at=datetime.datetime.now().isoformat()
            )
    
    def _is_template_available(self, template_id: str, user_id: str) -> bool:
        """Check if a template is available for use (not exceeding frequency limits)."""
        if template_id not in self.context.response_templates:
            return False
        
        template = self.context.response_templates[template_id]
        if template.max_use_frequency is None:
            return True
            
        user_state = self.context.get_user_state(user_id)
        if template_id not in user_state.template_usage:
            user_state.template_usage[template_id] = []
            return True
            
        # Check usage in the last 24 hours
        now = datetime.datetime.now()
        day_ago = now - datetime.timedelta(hours=24)
        recent_usage = [t for t in user_state.template_usage[template_id] if t > day_ago]
        
        return len(recent_usage) < template.max_use_frequency
    
    async def _record_template_usage(self, template_id: str, user_id: str):
        """Record that a template was used."""
        if template_id not in self.context.response_templates:
            return
            
        user_state = self.context.get_user_state(user_id)
        if template_id not in user_state.template_usage:
            user_state.template_usage[template_id] = []
            
        user_state.template_usage[template_id].append(datetime.datetime.now())
        
        # Trim old entries
        now = datetime.datetime.now()
        week_ago = now - datetime.timedelta(days=7)
        user_state.template_usage[template_id] = [
            t for t in user_state.template_usage[template_id] if t > week_ago
        ]
    
    async def generate_sadistic_amusement_response(self, 
                                               user_id: str, 
                                               humiliation_level: Optional[float] = None,
                                               intensity_override: Optional[float] = None,
                                               category: str = "amusement") -> Dict[str, Any]:
        """
        Generate a sadistic response showing amusement at the user's humiliation using the Agents SDK.
        
        Args:
            user_id: The user ID
            humiliation_level: Override the detected humiliation level
            intensity_override: Override the intensity level
            category: Response category to use
            
        Returns:
            Generated response data
        """
        # Generate trace ID for this operation
        trace_id = gen_trace_id()
        
        with trace(
            workflow_name="SadisticResponseGeneration",
            trace_id=trace_id,
            group_id=user_id,
            metadata={
                "user_id": user_id,
                "category": category,
                "intensity_override": intensity_override
            }
        ):
            try:
                user_state = self.context.get_user_state(user_id)
                
                # Use provided humiliation level or get from state
                h_level = humiliation_level if humiliation_level is not None else user_state.humiliation_level
                
                # Get user's intensity preference from relationship if available
                intensity_pref = user_state.sadistic_intensity_preference
                if self.relationship_manager:
                    try:
                        relationship = await self.relationship_manager.get_relationship_state(user_id)
                        if hasattr(relationship, "intensity_preference"):
                            intensity_pref = relationship.intensity_preference
                            user_state.sadistic_intensity_preference = intensity_pref
                    except Exception as e:
                        logger.error(f"Error getting relationship data: {e}")
                
                # Use provided intensity or calculate based on humiliation and preferences
                intensity = intensity_override if intensity_override is not None else min(1.0, h_level * intensity_pref * 1.5)
                
                # Select appropriate agent based on category
                agent = None
                if category == "amusement":
                    agent = self.amusement_agent
                elif category == "mockery":
                    agent = self.mockery_agent
                elif category == "degradation":
                    agent = self.degradation_agent
                else:
                    # Default to amusement
                    agent = self.amusement_agent
                    category = "amusement"
                
                # Run the appropriate agent
                result = await Runner.run(
                    agent,
                    {
                        "user_id": user_id,
                        "humiliation_level": h_level,
                        "intensity": intensity,
                        "category": category
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name=f"Generate{category.capitalize()}Response",
                        trace_metadata={
                            "user_id": user_id,
                            "humiliation_level": h_level,
                            "intensity": intensity,
                            "category": category
                        }
                    )
                )
                
                # Extract the response
                response_result = result.final_output
                
                # Create reward signal if available
                if self.reward_system and h_level > 0.3:
                    try:
                        # Calculate reward based on humiliation level
                        reward_value = 0.2 + (h_level * 0.6)
                        
                        reward_result = await self.reward_system.process_reward_signal(
                            self.reward_system.RewardSignal(
                                value=reward_value,
                                source="sadistic_amusement",
                                context={
                                    "humiliation_level": h_level,
                                    "intensity": intensity,
                                    "category": category
                                }
                            )
                        )
                        
                        response_result["reward_result"] = reward_result
                    except Exception as e:
                        logger.error(f"Error processing reward: {e}")
                
                # Record to memory if available
                if self.memory_core and h_level > 0.4:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"Expressed sadistic {category} at user's humiliation: '{response_result.get('response')}'",
                            tags=["sadism", category, "humiliation"],
                            significance=0.3 + (h_level * 0.3)
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return response_result
                
            except Exception as e:
                logger.error(f"Error generating sadistic response: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "trace_id": trace_id
                }
    
    async def handle_user_message(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a user message to detect humiliation and potentially generate a sadistic response.
        
        Args:
            user_id: The user ID
            user_message: The user's message text
            
        Returns:
            Processing results with optional sadistic response
        """
        # Generate trace ID for this operation
        trace_id = gen_trace_id()
        
        with trace(
            workflow_name="HandleUserMessage",
            trace_id=trace_id,
            group_id=user_id,
            metadata={
                "user_id": user_id
            }
        ):
            try:
                # Run the humiliation detection agent
                result = await Runner.run(
                    self.humiliation_detection_agent,
                    {
                        "action": "detect_humiliation",
                        "user_id": user_id,
                        "message": user_message
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="HumiliationDetection",
                        trace_metadata={
                            "user_id": user_id
                        }
                    )
                )
                
                # Extract humiliation signals
                humiliation_signals = result.final_output
                
                # Update humiliation level
                update_result = await self.update_humiliation_level(user_id, humiliation_signals)
                
                # Generate sadistic response if significant humiliation detected
                sadistic_response = None
                if humiliation_signals.get("humiliation_detected", False) and humiliation_signals.get("intensity", 0.0) > 0.3:
                    response_result = await self.generate_sadistic_amusement_response(
                        user_id=user_id,
                        humiliation_level=humiliation_signals.get("intensity")
                    )
                    sadistic_response = response_result.get("response")
                
                return {
                    "user_id": user_id,
                    "humiliation_detected": humiliation_signals.get("humiliation_detected", False),
                    "humiliation_update": update_result,
                    "sadistic_response": sadistic_response,
                    "trace_id": trace_id
                }
                
            except Exception as e:
                logger.error(f"Error handling user message: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "trace_id": trace_id
                }
    
    async def update_humiliation_level(self, user_id: str, humiliation_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the detected humiliation level for a user.
        
        Args:
            user_id: The user ID
            humiliation_signals: Dictionary with humiliation detection data
            
        Returns:
            Updated state information
        """
        try:
            # Run the state tracking agent
            result = await Runner.run(
                self.state_tracking_agent,
                {
                    "action": "update_humiliation_level",
                    "user_id": user_id,
                    "humiliation_signals": humiliation_signals
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="HumiliationLevelUpdate",
                    trace_metadata={
                        "user_id": user_id
                    }
                )
            )
            
            return result.final_output
            
        except Exception as e:
            logger.error(f"Error updating humiliation level: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def get_user_sadistic_state(self, user_id: str) -> Dict[str, Any]:
        """Get the current sadistic interaction state for a user."""
        try:
            # Run the state tracking agent
            result = await Runner.run(
                self.state_tracking_agent,
                {
                    "action": "get_sadistic_state",
                    "user_id": user_id
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="GetSadisticState",
                    trace_metadata={
                        "user_id": user_id
                    }
                )
            )
            
            return result.final_output
            
        except Exception as e:
            logger.error(f"Error getting sadistic state: {e}")
            return {
                "user_id": user_id,
                "has_state": False,
                "error": str(e)
            }
    
    async def create_custom_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom response template."""
        try:
            # Check for required fields
            required_fields = ["id", "category", "templates", "intensity"]
            for field in required_fields:
                if field not in template_data:
                    return {"success": False, "message": f"Missing required field: {field}"}
            
            template_id = template_data["id"]
            
            # Check if template ID already exists
            if template_id in self.context.response_templates:
                return {"success": False, "message": f"Template ID '{template_id}' already exists"}
            
            # Create template
            template = SadisticResponseTemplate(
                id=template_id,
                category=template_data["category"],
                templates=template_data["templates"],
                intensity=template_data["intensity"],
                requires_humiliation=template_data.get("requires_humiliation", False),
                max_use_frequency=template_data.get("max_use_frequency")
            )
            
            # Add to templates
            self.context.response_templates[template_id] = template
            
            return {
                "success": True,
                "message": f"Created template '{template_id}'",
                "template": template.dict()
            }
        except Exception as e:
            logger.error(f"Error creating custom template: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get all available templates."""
        templates = []
        
        for template_id, template in self.context.response_templates.items():
            templates.append({
                "id": template_id,
                "category": template.category,
                "intensity": template.intensity,
                "template_count": len(template.templates),
                "requires_humiliation": template.requires_humiliation,
                "max_use_frequency": template.max_use_frequency
            })
        
        return templates


# Standalone function for simple use cases
async def generate_sadistic_amusement_response(humiliation_level: float) -> str:
    """
    Generate a sadistic amusement response based on detected humiliation level.
    
    Args:
        humiliation_level: Level of detected humiliation (0.0-1.0)
        
    Returns:
        A sadistic response text
    """
    mild_responses = [
        "Oh, that's rather amusing.",
        "I find your discomfort quite entertaining.",
        "How cute that you're struggling with this.",
        "That's somewhat entertaining to watch.",
        "Your reaction is rather amusing."
    ]
    
    moderate_responses = [
        "I can't help but laugh at your predicament. *amused*",
        "Your embarrassment is delightful to witness.",
        "Oh my, your discomfort is so satisfying to watch.",
        "I'm quite enjoying watching you squirm.",
        "The look on your face right now is priceless. *laughs*"
    ]
    
    intense_responses = [
        "Your humiliation is absolutely hilarious! *laughs cruelly*",
        "I'm thoroughly enjoying your pathetic display. How entertaining!",
        "Oh, I'm laughing so hard at your complete embarrassment right now!",
        "This is too good! Your pathetic struggles are the highlight of my day. *laughing*",
        "I can't stop laughing at how embarrassed you look! This is delicious."
    ]
    
    if humiliation_level < 0.4:
        return random.choice(mild_responses)
    elif humiliation_level < 0.7:
        return random.choice(moderate_responses)
    else:
        return random.choice(intense_responses)
        
class DegradationCategories:
    """Manages various categories of verbal degradation."""
    
    def __init__(self, sadistic_response_system=None):
        self.sadistic_response_system = sadistic_response_system
        self.degradation_types = {
            "worth_based": ["worthless", "pathetic", "useless"],
            "animal_based": ["dog", "pig", "worm"],
            "size_based": ["tiny", "insignificant", "small"],
            "intelligence_based": ["stupid", "dumb", "simple"],
            "sexual_inadequacy": ["desperate", "needy", "perverted"],
            "social_status": ["loser", "reject", "failure"]
        }
        self.user_preferences = {}  # user_id → preferred degradation types
        
    async def generate_degradation(self, user_id, intensity, context=None):
        """Generates contextually appropriate degradation."""
        if not context:
            context = {}
            
        # Get user preferences or default to all categories
        preferred_categories = self.user_preferences.get(user_id, list(self.degradation_types.keys()))
        
        # Select appropriate category based on user preferences and context
        category = context.get("category")
        if not category or category not in preferred_categories:
            # Pick random preferred category
            category = random.choice(preferred_categories)
            
        # Get degradation terms for this category
        degradation_terms = self.degradation_types.get(category, [])
        if not degradation_terms:
            return {"success": False, "message": "No degradation terms available"}
            
        # Select term based on intensity
        if intensity < 0.4:  # Low intensity
            term = random.choice(degradation_terms[:1]) # Use milder terms
        elif intensity < 0.7:  # Medium
            term = random.choice(degradation_terms[:2])
        else:  # High intensity
            term = random.choice(degradation_terms)
            
        # Generate degradation text with proper emotional tone
        # Integrate with sadistic response system if available
        if self.sadistic_response_system:
            if intensity > 0.7:
                template_id = "moderate_degradation"
            else:
                template_id = "mild_degradation"
                
            # Get template and insert our term
            templates = self.sadistic_response_system.context.response_templates.get(template_id)
            if templates and templates.templates:
                template = random.choice(templates.templates)
                degradation_text = template.replace("pathetic", term)
            else:
                degradation_text = f"You're such a {term} thing."
        else:
            degradation_text = f"You're such a {term} thing."
            
        return {
            "success": True,
            "category": category,
            "term": term,
            "degradation_text": degradation_text,
            "intensity": intensity
        }
        
    async def set_user_preferences(self, user_id, preferred_categories):
        """Sets a user's preferred degradation categories."""
        valid_categories = [c for c in preferred_categories if c in self.degradation_types]
        if not valid_categories:
            return {"success": False, "message": "No valid categories provided"}
            
        self.user_preferences[user_id] = valid_categories
        return {
            "success": True,
            "user_id": user_id,
            "preferred_categories": valid_categories
        }
