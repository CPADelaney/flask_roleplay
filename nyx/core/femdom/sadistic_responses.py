# nyx/core/femdom/sadistic_responses.py

import logging
import random
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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

class SadisticResponseSystem:
    """Manages generation of sadistic responses for femdom interactions."""
    
    def __init__(self, theory_of_mind=None, protocol_enforcement=None, 
                 reward_system=None, relationship_manager=None, 
                 submission_progression=None, memory_core=None):
        self.theory_of_mind = theory_of_mind
        self.protocol_enforcement = protocol_enforcement
        self.reward_system = reward_system
        self.relationship_manager = relationship_manager
        self.submission_progression = submission_progression
        self.memory_core = memory_core
        
        # User states
        self.user_states: Dict[str, UserSadisticState] = {}
        
        # Response templates
        self.response_templates: Dict[str, SadisticResponseTemplate] = {}
        
        # Load default templates
        self._load_default_templates()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SadisticResponseSystem initialized")
    
    def _load_default_templates(self):
        """Load default sadistic response templates."""
        # Amusement templates (laughing at user discomfort)
        self.response_templates["mild_amusement"] = SadisticResponseTemplate(
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
        
        self.response_templates["moderate_amusement"] = SadisticResponseTemplate(
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
        
        self.response_templates["intense_amusement"] = SadisticResponseTemplate(
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
        self.response_templates["mild_mockery"] = SadisticResponseTemplate(
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
        
        self.response_templates["moderate_mockery"] = SadisticResponseTemplate(
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
        self.response_templates["mild_degradation"] = SadisticResponseTemplate(
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
        
        self.response_templates["moderate_degradation"] = SadisticResponseTemplate(
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
    
    def _get_or_create_user_state(self, user_id: str) -> UserSadisticState:
        """Get or create a user's sadistic state tracking."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserSadisticState(user_id=user_id)
        return self.user_states[user_id]
    
    async def update_humiliation_level(self, user_id: str, humiliation_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the detected humiliation level for a user.
        
        Args:
            user_id: The user ID
            humiliation_signals: Dictionary with humiliation detection data
            
        Returns:
            Updated state information
        """
        async with self._lock:
            user_state = self._get_or_create_user_state(user_id)
            
            # Extract detected humiliation level
            detected_level = humiliation_signals.get("intensity", 0.0)
            if humiliation_signals.get("humiliation_detected", False):
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
            
            return {
                "user_id": user_id,
                "old_humiliation_level": old_level,
                "new_humiliation_level": user_state.humiliation_level,
                "change": user_state.humiliation_level - old_level
            }
    
    def _is_template_available(self, template_id: str, user_id: str) -> bool:
        """Check if a template is available for use (not exceeding frequency limits)."""
        if template_id not in self.response_templates:
            return False
        
        template = self.response_templates[template_id]
        if template.max_use_frequency is None:
            return True
            
        user_state = self._get_or_create_user_state(user_id)
        if template_id not in user_state.template_usage:
            user_state.template_usage[template_id] = []
            return True
            
        # Check usage in the last 24 hours
        now = datetime.datetime.now()
        day_ago = now - datetime.timedelta(hours=24)
        recent_usage = [t for t in user_state.template_usage[template_id] if t > day_ago]
        
        return len(recent_usage) < template.max_use_frequency
    
    def _record_template_usage(self, template_id: str, user_id: str):
        """Record that a template was used."""
        if template_id not in self.response_templates:
            return
            
        user_state = self._get_or_create_user_state(user_id)
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
        Generate a sadistic response showing amusement at the user's humiliation.
        
        Args:
            user_id: The user ID
            humiliation_level: Override the detected humiliation level
            intensity_override: Override the intensity level
            category: Response category to use
            
        Returns:
            Generated response data
        """
        async with self._lock:
            user_state = self._get_or_create_user_state(user_id)
            
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
            
            # Filter templates by category and availability
            filtered_templates = [
                t for t_id, t in self.response_templates.items()
                if t.category == category and self._is_template_available(t_id, user_id)
            ]
            
            if not filtered_templates:
                # Fallback if no templates available in category
                response = "I find that quite amusing."
                template_id = None
            else:
                # Find closest matching template by intensity
                closest_template = min(filtered_templates, key=lambda t: abs(t.intensity - intensity))
                
                # Select a random template string
                response = random.choice(closest_template.templates)
                template_id = closest_template.id
                
                # Record usage
                if template_id:
                    self._record_template_usage(template_id, user_id)
            
            # Record in history
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "category": category,
                "intensity": intensity,
                "humiliation_level": h_level,
                "template_id": template_id,
                "response": response
            }
            user_state.response_history.append(history_entry)
            
            # Limit history size
            if len(user_state.response_history) > 20:
                user_state.response_history = user_state.response_history[-20:]
            
            # Create reward signal if available
            reward_result = None
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
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
            
            # Record memory if available
            if self.memory_core and h_level > 0.4:
                try:
                    await self.memory_core.add_memory(
                        memory_type="experience",
                        content=f"Expressed sadistic amusement at user's humiliation: '{response}'",
                        tags=["sadism", "amusement", "humiliation", category],
                        significance=0.3 + (h_level * 0.3)
                    )
                except Exception as e:
                    logger.error(f"Error recording memory: {e}")
            
            return {
                "response": response,
                "humiliation_level": h_level,
                "intensity": intensity,
                "category": category,
                "template_id": template_id,
                "reward_result": reward_result
            }
    
    async def get_user_sadistic_state(self, user_id: str) -> Dict[str, Any]:
        """Get the current sadistic interaction state for a user."""
        async with self._lock:
            if user_id not in self.user_states:
                return {"user_id": user_id, "has_state": False}
                
            user_state = self.user_states[user_id]
            
            # Format template usage
            template_usage = {}
            for template_id, timestamps in user_state.template_usage.items():
                # Count usage in last 24 hours
                now = datetime.datetime.now()
                day_ago = now - datetime.timedelta(hours=24)
                recent_count = sum(1 for t in timestamps if t > day_ago)
                
                if template_id in self.response_templates:
                    template = self.response_templates[template_id]
                    template_usage[template_id] = {
                        "category": template.category,
                        "intensity": template.intensity,
                        "usage_24h": recent_count,
                        "max_frequency": template.max_use_frequency
                    }
            
            # Return formatted state
            return {
                "user_id": user_id,
                "has_state": True,
                "humiliation_level": user_state.humiliation_level,
                "last_humiliation_update": user_state.last_humiliation_update.isoformat(),
                "sadistic_intensity_preference": user_state.sadistic_intensity_preference,
                "template_usage": template_usage,
                "recent_responses": user_state.response_history[-5:] if user_state.response_history else []
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
            if template_id in self.response_templates:
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
            self.response_templates[template_id] = template
            
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
        
        for template_id, template in self.response_templates.items():
            templates.append({
                "id": template_id,
                "category": template.category,
                "intensity": template.intensity,
                "template_count": len(template.templates),
                "requires_humiliation": template.requires_humiliation,
                "max_use_frequency": template.max_use_frequency
            })
        
        return templates
    
    async def handle_user_message(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a user message to detect humiliation and potentially generate a sadistic response.
        
        Args:
            user_id: The user ID
            user_message: The user's message text
            
        Returns:
            Processing results with optional sadistic response
        """
        # Check for humiliation signals using Theory of Mind
        humiliation_signals = None
        if self.theory_of_mind:
            try:
                humiliation_signals = await self.theory_of_mind.detect_humiliation_signals(user_message)
            except Exception as e:
                logger.error(f"Error detecting humiliation: {e}")
                humiliation_signals = {"humiliation_detected": False, "intensity": 0.0}
        else:
            # Simple fallback if no Theory of Mind
            humiliation_markers = [
                "embarrassed", "humiliated", "ashamed", "blushing", "awkward",
                "uncomfortable", "exposed", "vulnerable", "pathetic", "inadequate"
            ]
            marker_count = sum(user_message.lower().count(marker) for marker in humiliation_markers)
            humiliation_signals = {
                "humiliation_detected": marker_count > 0,
                "intensity": min(1.0, marker_count * 0.2)
            }
        
        # Update humiliation level
        update_result = await self.update_humiliation_level(user_id, humiliation_signals)
        
        # Generate sadistic response if significant humiliation detected
        sadistic_response = None
        if humiliation_signals.get("humiliation_detected", False) and humiliation_signals.get("intensity", 0.0) > 0.3:
            sadistic_response = await self.generate_sadistic_amusement_response(
                user_id=user_id,
                humiliation_level=humiliation_signals.get("intensity")
            )
        
        return {
            "user_id": user_id,
            "humiliation_detected": humiliation_signals.get("humiliation_detected", False),
            "humiliation_update": update_result,
            "sadistic_response": sadistic_response
        }


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
            templates = self.sadistic_response_system.response_templates.get(template_id)
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
