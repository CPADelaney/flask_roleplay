# nyx/core/emotional_core.py

import datetime
import json
import logging
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field, validator

# Import OpenAI Agents SDK components
from agents import (
    Agent, Runner, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    function_tool, handoff, trace, RunContextWrapper, FunctionTool
)

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class EmotionValue(BaseModel):
    """Schema for individual emotion values"""
    value: float = Field(..., description="Emotion intensity (0.0-1.0)", ge=0.0, le=1.0)
    
    @validator('value')
    def validate_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("Emotion value must be between 0.0 and 1.0")
        return v

class EmotionalState(BaseModel):
    """Schema for the complete emotional state"""
    Joy: EmotionValue
    Sadness: EmotionValue
    Fear: EmotionValue
    Anger: EmotionValue
    Trust: EmotionValue
    Disgust: EmotionValue
    Anticipation: EmotionValue
    Surprise: EmotionValue
    Love: EmotionValue
    Frustration: EmotionValue
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    @validator('*', pre=True)
    def convert_to_emotion_value(cls, v):
        if isinstance(v, (int, float)):
            return EmotionValue(value=v)
        return v

class FormattedEmotionalState(BaseModel):
    """Schema for formatted emotional state output"""
    primary_emotion: str = Field(..., description="Dominant emotion")
    primary_intensity: float = Field(..., description="Intensity of dominant emotion", ge=0.0, le=1.0)
    secondary_emotions: Dict[str, float] = Field(..., description="Other significant emotions")
    valence: float = Field(..., description="Overall emotional valence (-1.0 to 1.0)", ge=-1.0, le=1.0)
    arousal: float = Field(..., description="Overall emotional intensity/arousal", ge=0.0, le=1.0)

class EmotionUpdateInput(BaseModel):
    """Schema for emotion update input"""
    emotion: str = Field(..., description="Emotion to update")
    value: float = Field(..., description="Change in emotion value (-1.0 to 1.0)", ge=-1.0, le=1.0)

class EmotionUpdateResult(BaseModel):
    """Schema for emotion update result"""
    success: bool = Field(..., description="Whether the update was successful")
    updated_emotion: str = Field(..., description="Emotion that was updated")
    old_value: float = Field(..., description="Previous emotion value")
    new_value: float = Field(..., description="New emotion value")
    emotional_state: Dict[str, float] = Field(..., description="Updated emotional state")

class EmotionSetInput(BaseModel):
    """Schema for emotion set input"""
    emotion: str = Field(..., description="Emotion to set")
    value: float = Field(..., description="Absolute emotion value (0.0 to 1.0)", ge=0.0, le=1.0)

class TextAnalysisOutput(BaseModel):
    """Schema for text sentiment analysis"""
    emotions: Dict[str, float] = Field(..., description="Detected emotions and intensities")
    dominant_emotion: str = Field(..., description="Dominant emotion in text")
    intensity: float = Field(..., description="Overall emotional intensity", ge=0.0, le=1.0)
    valence: float = Field(..., description="Overall emotional valence", ge=-1.0, le=1.0)

class EmotionExpressionOutput(BaseModel):
    """Schema for emotional expression"""
    expression_text: str = Field(..., description="Natural language expression of emotion")
    emotion: str = Field(..., description="Emotion being expressed")
    intensity: float = Field(..., description="Intensity of the emotion", ge=0.0, le=1.0)

class EmotionResonanceInput(BaseModel):
    """Schema for emotional resonance calculation input"""
    memory_emotions: Dict[str, Any] = Field(..., description="Memory's emotional data")
    current_emotions: Optional[Dict[str, float]] = Field(None, description="Current emotional state")

class EmotionalCore:
    """
    Agent-based emotion management system for Nyx.
    Handles emotion representation, intensity, decay, and updates from stimuli.
    Leverages OpenAI Agents SDK for coordinated emotional processing.
    """
    
    def __init__(self):
        # Initialize baseline emotions with default values
        self.emotions = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.1,
            "Anger": 0.1,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.3,
            "Surprise": 0.1,
            "Love": 0.3,
            "Frustration": 0.1
        }
        
        # Emotion decay rates (how quickly emotions fade without reinforcement)
        self.decay_rates = {
            "Joy": 0.05,
            "Sadness": 0.03,
            "Fear": 0.04,
            "Anger": 0.06,
            "Trust": 0.02,
            "Disgust": 0.05,
            "Anticipation": 0.04,
            "Surprise": 0.08,
            "Love": 0.01,
            "Frustration": 0.05
        }
        
        # Emotional baseline (personality tendency)
        self.baseline = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.2,
            "Anger": 0.2,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.4,
            "Surprise": 0.3,
            "Love": 0.4,
            "Frustration": 0.2
        }
        
        # Emotion-memory valence mapping
        self.emotion_memory_valence_map = {
            "Joy": 0.8,
            "Trust": 0.6,
            "Anticipation": 0.5,
            "Love": 0.9,
            "Surprise": 0.2,
            "Sadness": -0.6,
            "Fear": -0.7,
            "Anger": -0.8,
            "Disgust": -0.7,
            "Frustration": -0.5
        }
        
        # Emotional history
        self.emotional_history = []
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
        
        # Set up agents
        self.emotion_agent = self._create_emotion_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.expression_agent = self._create_expression_agent()
        self.resonance_agent = self._create_resonance_agent()
    
    def _create_emotion_agent(self):
        """Create the main emotion management agent"""
        return Agent(
            name="Emotion Agent",
            instructions="""
            You are the Emotion Agent, responsible for managing emotional state.
            Your role is to handle emotion updates, decay, and coordinate with
            specialized emotional processing agents.
            """,
            handoffs=[
                handoff(self._create_analysis_agent()),
                handoff(self._create_expression_agent()),
                handoff(self._create_resonance_agent())
            ],
            tools=[
                function_tool(self._update_emotion_tool),
                function_tool(self._set_emotion_tool),
                function_tool(self._apply_decay_tool),
                function_tool(self._get_emotional_state_tool)
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._emotion_request_guardrail)
            ]
        )
    
    def _create_analysis_agent(self):
        """Create the emotion analysis agent"""
        return Agent(
            name="Emotion Analysis Agent",
            handoff_description="Specialist agent for analyzing text for emotional content",
            instructions="""
            You are the Emotion Analysis Agent, specialized in detecting emotional
            content in text. Analyze text for emotional tones, sentiment, and specific emotions.
            """,
            tools=[
                function_tool(self._analyze_text_tool)
            ],
            output_type=TextAnalysisOutput
        )
    
    def _create_expression_agent(self):
        """Create the emotion expression agent"""
        return Agent(
            name="Emotion Expression Agent",
            handoff_description="Specialist agent for generating natural language expressions of emotions",
            instructions="""
            You are the Emotion Expression Agent, specialized in creating natural,
            authentic expressions of emotions in natural language.
            """,
            tools=[
                function_tool(self._get_emotional_state_tool),
                function_tool(self._get_expression_tool)
            ],
            output_type=EmotionExpressionOutput
        )
    
    def _create_resonance_agent(self):
        """Create the emotion resonance agent"""
        return Agent(
            name="Emotion Resonance Agent",
            handoff_description="Specialist agent for calculating emotional resonance between states",
            instructions="""
            You are the Emotion Resonance Agent, specialized in comparing emotional
            states and calculating how strongly they resonate with each other.
            """,
            tools=[
                function_tool(self._calculate_resonance_tool)
            ]
        )
    
    # Guardrail functions
    
    async def _emotion_request_guardrail(self, ctx, agent, input_data):
        """Guardrail to validate emotion-related requests"""
        # Check for minimum context
        if isinstance(input_data, str) and len(input_data.strip()) < 3:
            return GuardrailFunctionOutput(
                output_info={"error": "Request too short"},
                tripwire_triggered=True
            )
        
        # Check for valid emotion names if request contains emotion update
        if isinstance(input_data, dict) and "emotion" in input_data:
            if input_data["emotion"] not in self.emotions:
                return GuardrailFunctionOutput(
                    output_info={"error": f"Unknown emotion: {input_data['emotion']}",
                                "available_emotions": list(self.emotions.keys())},
                    tripwire_triggered=True
                )
        
        return GuardrailFunctionOutput(
            output_info={"valid": True},
            tripwire_triggered=False
        )
    
    # Tool functions
    
    @function_tool
    async def _update_emotion_tool(self, ctx: RunContextWrapper, 
                               emotion: str, 
                               value: float) -> Dict[str, Any]:
        """
        Update a specific emotion with a delta change.
        
        Args:
            emotion: The emotion to update (e.g., "Joy", "Fear")
            value: Delta value to apply (-1.0 to 1.0)
            
        Returns:
            Update result data
        """
        # Validate input
        if not -1.0 <= value <= 1.0:
            return {
                "error": "Value must be between -1.0 and 1.0"
            }
        
        if emotion not in self.emotions:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(self.emotions.keys())
            }
        
        # Get pre-update value
        old_value = self.emotions[emotion]
        
        # Update emotion
        self.emotions[emotion] = max(0, min(1, self.emotions[emotion] + value))
        
        # Update timestamp and record in history
        self.last_update = datetime.datetime.now()
        self._record_emotional_state()
        
        return {
            "success": True,
            "updated_emotion": emotion,
            "change": value,
            "old_value": old_value,
            "new_value": self.emotions[emotion],
            "emotional_state": self.get_emotional_state()
        }
    
    @function_tool
    async def _set_emotion_tool(self, ctx: RunContextWrapper, 
                            emotion: str, 
                            value: float) -> Dict[str, Any]:
        """
        Set a specific emotion to an absolute value.
        
        Args:
            emotion: The emotion to set (e.g., "Joy", "Fear")
            value: Absolute value (0.0 to 1.0)
            
        Returns:
            Update result data
        """
        # Validate input
        if not 0.0 <= value <= 1.0:
            return {
                "error": "Value must be between 0.0 and 1.0"
            }
        
        if emotion not in self.emotions:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(self.emotions.keys())
            }
        
        # Get pre-update value
        old_value = self.emotions[emotion]
        
        # Set emotion to absolute value
        self.emotions[emotion] = value
        
        # Update timestamp and record in history
        self.last_update = datetime.datetime.now()
        self._record_emotional_state()
        
        return {
            "success": True,
            "set_emotion": emotion,
            "old_value": old_value,
            "value": value,
            "emotional_state": self.get_emotional_state()
        }
    
    @function_tool
    async def _apply_decay_tool(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Apply emotional decay based on time elapsed.
        
        Returns:
            Updated emotional state after decay
        """
        self.apply_decay()
        return {
            "emotional_state": self.get_emotional_state(),
            "decay_applied": True,
            "last_update": self.last_update.isoformat()
        }
    
    @function_tool
    async def _get_emotional_state_tool(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get current emotional state.
        
        Returns:
            Current emotional state data
        """
        # Apply decay before returning state
        self.apply_decay()
        
        dominant_emotion, dominant_value = self.get_dominant_emotion()
        
        return {
            "emotions": self.get_emotional_state(),
            "dominant_emotion": dominant_emotion,
            "dominant_value": dominant_value,
            "valence": self.get_emotional_valence(),
            "arousal": self.get_emotional_arousal(),
            "formatted": self.get_formatted_emotional_state()
        }
    
    @function_tool
    async def _analyze_text_tool(self, ctx: RunContextWrapper, text: str) -> Dict[str, Any]:
        """
        Analyze text for emotional content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotional analysis of text
        """
        stimuli = self.analyze_text_sentiment(text)
        
        # Find dominant emotion
        dominant_emotion = "neutral"
        dominant_value = 0.0
        
        if stimuli:
            dominant_item = max(stimuli.items(), key=lambda x: x[1])
            dominant_emotion = dominant_item[0]
            dominant_value = dominant_item[1]
        
        # Calculate overall valence
        valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                     for emotion, value in stimuli.items())
        valence = max(-1.0, min(1.0, valence))  # Clamp between -1 and 1
        
        # Calculate overall intensity
        intensity = sum(stimuli.values())
        
        return {
            "emotions": stimuli,
            "dominant_emotion": dominant_emotion,
            "intensity": intensity,
            "valence": valence
        }
    
    @function_tool
    async def _get_expression_tool(self, ctx: RunContextWrapper,
                              emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a natural language expression for an emotion.
        
        Args:
            emotion: Specific emotion to express (or None for dominant emotion)
            
        Returns:
            Natural language expression data
        """
        # Use dominant emotion if none specified
        if emotion is None:
            emotion, intensity = self.get_dominant_emotion()
        else:
            intensity = self.emotions.get(emotion, 0.5)
        
        # Get expression
        expression = self.get_expression_for_emotion(emotion)
        
        return {
            "expression": expression,
            "emotion": emotion,
            "intensity": intensity
        }
    
    @function_tool
    async def _calculate_resonance_tool(self, ctx: RunContextWrapper,
                                    memory_emotions: Dict[str, Any],
                                    current_emotions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate emotional resonance between memory and current state.
        
        Args:
            memory_emotions: Memory's emotional data
            current_emotions: Current emotional state (or None to use system state)
            
        Returns:
            Resonance calculation results
        """
        # Use current emotions if not provided
        if current_emotions is None:
            current_emotions = self.get_emotional_state()
        
        # Calculate resonance
        resonance = self.calculate_emotional_resonance(memory_emotions, current_emotions)
        
        # Get additional data
        memory_primary = memory_emotions.get("primary_emotion", "neutral")
        memory_intensity = memory_emotions.get("primary_intensity", 0.5)
        memory_valence = memory_emotions.get("valence", 0.0)
        
        current_primary, current_intensity = self.get_dominant_emotion()
        current_valence = self.get_emotional_valence()
        
        return {
            "resonance": resonance,
            "memory_emotion": memory_primary,
            "current_emotion": current_primary,
            "memory_valence": memory_valence,
            "current_valence": current_valence,
            "memory_intensity": memory_intensity,
            "current_intensity": current_intensity
        }
    
    # Original core methods with some refinements
    
    def update_emotion(self, emotion: str, value: float) -> bool:
        """Update a specific emotion with a new intensity value (delta)"""
        if emotion in self.emotions:
            # Ensure emotion values stay between 0 and 1
            self.emotions[emotion] = max(0, min(1, self.emotions[emotion] + value))
            return True
        return False
    
    def set_emotion(self, emotion: str, value: float) -> bool:
        """Set a specific emotion to an absolute value (not delta)"""
        if emotion in self.emotions:
            # Ensure emotion values stay between 0 and 1
            self.emotions[emotion] = max(0, min(1, value))
            return True
        return False
    
    def update_from_stimuli(self, stimuli: Dict[str, float]) -> Dict[str, float]:
        """
        Update emotions based on received stimuli
        stimuli: dict of emotion adjustments
        """
        for emotion, adjustment in stimuli.items():
            self.update_emotion(emotion, adjustment)
        
        # Update timestamp
        self.last_update = datetime.datetime.now()
        
        # Record in history
        self._record_emotional_state()
        
        return self.get_emotional_state()
    
    def apply_decay(self):
        """Apply emotional decay based on time elapsed since last update"""
        now = datetime.datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 3600  # hours
        
        # Don't decay if less than a minute has passed
        if time_delta < 0.016:  # about 1 minute in hours
            return
        
        for emotion in self.emotions:
            # Calculate decay based on time passed
            decay_amount = self.decay_rates[emotion] * time_delta
            
            # Current emotion value
            current = self.emotions[emotion]
            
            # Decay toward baseline
            baseline = self.baseline[emotion]
            if current > baseline:
                self.emotions[emotion] = max(baseline, current - decay_amount)
            elif current < baseline:
                self.emotions[emotion] = min(baseline, current + decay_amount)
        
        # Update timestamp
        self.last_update = now
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Return the current emotional state"""
        self.apply_decay()  # Apply decay before returning state
        return self.emotions.copy()
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Return the most intense emotion"""
        self.apply_decay()
        return max(self.emotions.items(), key=lambda x: x[1])
    
    def get_emotional_valence(self) -> float:
        """Calculate overall emotional valence (positive/negative)"""
        valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                     for emotion, value in self.emotions.items())
        return max(-1.0, min(1.0, valence))  # Clamp between -1 and 1
    
    def get_emotional_arousal(self) -> float:
        """Calculate overall emotional arousal (intensity)"""
        return sum(value for value in self.emotions.values()) / len(self.emotions)
    
    def get_formatted_emotional_state(self) -> Dict[str, Any]:
        """Get a formatted emotional state suitable for memory storage"""
        dominant_emotion, dominant_value = self.get_dominant_emotion()
        
        # Get secondary emotions (high intensity but not dominant)
        secondary_emotions = {
            emotion: value for emotion, value in self.emotions.items()
            if value >= 0.4 and emotion != dominant_emotion
        }
        
        return {
            "primary_emotion": dominant_emotion,
            "primary_intensity": dominant_value,
            "secondary_emotions": secondary_emotions,
            "valence": self.get_emotional_valence(),
            "arousal": self.get_emotional_arousal()
        }
    
    def _record_emotional_state(self):
        """Record current emotional state in history"""
        state = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotions": self.emotions.copy(),
            "dominant_emotion": self.get_dominant_emotion()[0],
            "valence": self.get_emotional_valence(),
            "arousal": self.get_emotional_arousal()
        }
        self.emotional_history.append(state)
        
        # Limit history size
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Simple analysis of text sentiment to extract emotional stimuli.
        For a proper implementation, this would use an NLP model.
        """
        stimuli = {}
        text_lower = text.lower()
        
        # Very basic keyword matching
        if any(word in text_lower for word in ["happy", "good", "great", "love", "like"]):
            stimuli["Joy"] = 0.1
            stimuli["Trust"] = 0.1
        
        if any(word in text_lower for word in ["sad", "sorry", "miss", "lonely"]):
            stimuli["Sadness"] = 0.1
        
        if any(word in text_lower for word in ["worried", "scared", "afraid", "nervous"]):
            stimuli["Fear"] = 0.1
        
        if any(word in text_lower for word in ["angry", "mad", "frustrated", "annoyed"]):
            stimuli["Anger"] = 0.1
            stimuli["Frustration"] = 0.1
        
        if any(word in text_lower for word in ["surprised", "wow", "unexpected"]):
            stimuli["Surprise"] = 0.1
        
        if any(word in text_lower for word in ["gross", "disgusting", "nasty"]):
            stimuli["Disgust"] = 0.1
        
        if any(word in text_lower for word in ["hope", "future", "expect", "waiting"]):
            stimuli["Anticipation"] = 0.1
        
        if any(word in text_lower for word in ["trust", "believe", "faith", "reliable"]):
            stimuli["Trust"] = 0.1
        
        if any(word in text_lower for word in ["love", "adore", "cherish"]):
            stimuli["Love"] = 0.1
        
        # Return neutral if no matches
        if not stimuli:
            stimuli = {
                "Surprise": 0.05,
                "Anticipation": 0.05
            }
        
        return stimuli
    
    def calculate_emotional_resonance(self, 
                                     memory_emotions: Dict[str, Any],
                                     current_emotions: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate how strongly a memory's emotions resonate with current state.
        Returns a value from 0.0 (no resonance) to 1.0 (perfect resonance).
        """
        # Use current emotions if not explicitly provided
        if current_emotions is None:
            current_emotions = self.get_emotional_state()
        
        # Extract primary emotion from memory
        memory_primary = memory_emotions.get("primary_emotion", "neutral")
        memory_intensity = memory_emotions.get("primary_intensity", 0.5)
        memory_valence = memory_emotions.get("valence", 0.0)
        
        # Calculate primary emotion match
        primary_match = 0.0
        if memory_primary in current_emotions:
            primary_match = current_emotions[memory_primary] * memory_intensity
        
        # Calculate valence match (how well positive/negative alignment matches)
        current_valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                            for emotion, value in current_emotions.items())
        
        valence_match = 1.0 - min(1.0, abs(current_valence - memory_valence))
        
        # Calculate secondary emotion matches
        secondary_match = 0.0
        secondary_emotions = memory_emotions.get("secondary_emotions", {})
        
        if secondary_emotions:
            matches = []
            for emotion, intensity in secondary_emotions.items():
                if emotion in current_emotions:
                    matches.append(current_emotions[emotion] * intensity)
            
            if matches:
                secondary_match = sum(matches) / len(matches)
        
        # Combined weighted resonance
        resonance = (
            primary_match * 0.5 +
            valence_match * 0.3 +
            secondary_match * 0.2
        )
        
        return max(0.0, min(1.0, resonance))
    
    def should_express_emotion(self) -> bool:
        """Determine if Nyx should express emotion based on current state"""
        # Get dominant emotion and intensity
        dominant_emotion, dominant_value = self.get_dominant_emotion()
        
        # Higher intensity emotions are more likely to be expressed
        threshold = 0.7 - (dominant_value * 0.3)  # Adaptive threshold
        
        return random.random() > threshold
    
    def get_expression_for_emotion(self, emotion: Optional[str] = None) -> str:
        """Get a natural language expression for an emotion"""
        if emotion is None:
            emotion, _ = self.get_dominant_emotion()
        
        # Simple expression templates
        expressions = {
            "Joy": ["I'm feeling quite pleased right now.", 
                   "I'm in a good mood today.",
                   "I feel rather happy at the moment."],
            "Sadness": ["I'm feeling a bit melancholy.",
                       "I feel somewhat downcast.",
                       "I'm in a rather somber mood."],
            "Fear": ["I'm feeling somewhat anxious.",
                    "I feel a bit uneasy.",
                    "I'm rather on edge right now."],
            "Anger": ["I'm feeling rather irritated.",
                     "I'm a bit annoyed at the moment.",
                     "I feel somewhat vexed."],
            "Trust": ["I'm feeling quite comfortable with you.",
                     "I feel we have a good understanding.",
                     "I'm in a trusting mood."],
            "Disgust": ["I'm feeling a bit repulsed.",
                       "I find this rather distasteful.",
                       "I'm somewhat revolted by this."],
            "Anticipation": ["I'm looking forward to what happens next.",
                            "I'm curious about what's to come.",
                            "I feel expectant about our interaction."],
            "Surprise": ["I'm quite taken aback.",
                        "This has certainly surprised me.",
                        "I didn't expect that."],
            "Love": ["I'm feeling particularly fond of you.",
                    "I'm in a very affectionate mood.",
                    "I feel quite attached to you at the moment."],
            "Frustration": ["I'm feeling somewhat frustrated.",
                           "I'm a bit exasperated right now.",
                           "I feel rather irritated at the moment."]
        }
        
        # Select a random expression for the emotion
        if emotion in expressions:
            return random.choice(expressions[emotion])
        else:
            return "I'm experiencing a complex mix of emotions right now."
    
    def save_state(self, filename: str) -> bool:
        """Save the current emotional state to file"""
        state = {
            "emotions": self.get_emotional_state(),
            "last_update": self.last_update.isoformat(),
            "emotional_history": self.emotional_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save emotional state: {e}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """Load emotional state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore emotions
            for emotion, value in state["emotions"].items():
                if emotion in self.emotions:
                    self.emotions[emotion] = value
            
            # Restore timestamp
            self.last_update = datetime.datetime.fromisoformat(state["last_update"])
            
            # Restore history if available
            if "emotional_history" in state:
                self.emotional_history = state["emotional_history"]
            
            return True
        except Exception as e:
            logger.error(f"Failed to load emotional state: {e}")
            return False
    
    # Public API methods
    
    async def process_emotional_input(self, text: str) -> Dict[str, Any]:
        """
        Process input text and update emotional state accordingly
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results with updated emotional state
        """
        # Use the Agent SDK to run the emotion agent
        with trace(workflow_name="EmotionalInput"):
            agent_input = {
                "role": "user", 
                "content": f"Process this input and update emotions appropriately: {text}"
            }
            
            result = await Runner.run(
                self.emotion_agent,
                agent_input
            )
            
            # Extract final output
            if hasattr(result, "final_output") and result.final_output:
                return result.final_output
            
            # Fall back to direct processing
            stimuli = self.analyze_text_sentiment(text)
            self.update_from_stimuli(stimuli)
            
            return {
                "input_processed": True,
                "detected_emotions": stimuli,
                "updated_state": self.get_formatted_emotional_state()
            }
    
    async def generate_emotional_expression(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate an emotional expression based on current state
        
        Args:
            force: Whether to force expression even if below threshold
            
        Returns:
            Expression result data
        """
        # Check if emotion should be expressed
        if not force and not self.should_express_emotion():
            return {
                "expressed": False,
                "reason": "Below expression threshold"
            }
        
        # Use the Expression Agent
        with trace(workflow_name="EmotionalExpression"):
            agent_input = {
                "role": "user",
                "content": "Generate an appropriate emotional expression based on current state."
            }
            
            result = await Runner.run(
                self.expression_agent,
                agent_input
            )
            
            # Get the expression output
            if hasattr(result, "final_output") and result.final_output:
                if isinstance(result.final_output, EmotionExpressionOutput):
                    expression_output = result.final_output
                    return {
                        "expressed": True,
                        "expression": expression_output.expression_text,
                        "emotion": expression_output.emotion,
                        "intensity": expression_output.intensity
                    }
            
            # Fall back to direct expression
            dominant_emotion, dominant_value = self.get_dominant_emotion()
            expression = self.get_expression_for_emotion(dominant_emotion)
            
            return {
                "expressed": True,
                "expression": expression,
                "emotion": dominant_emotion,
                "intensity": dominant_value
            }
    
    async def analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for emotional content
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotional analysis result
        """
        # Use the Analysis Agent
        with trace(workflow_name="EmotionalAnalysis"):
            agent_input = {
                "role": "user",
                "content": f"Analyze this text for emotional content: {text}"
            }
            
            result = await Runner.run(
                self.analysis_agent,
                agent_input
            )
            
            # Get the analysis output
            if hasattr(result, "final_output") and result.final_output:
                if isinstance(result.final_output, TextAnalysisOutput):
                    analysis_output = result.final_output
                    return {
                        "emotions": analysis_output.emotions,
                        "dominant_emotion": analysis_output.dominant_emotion,
                        "intensity": analysis_output.intensity,
                        "valence": analysis_output.valence
                    }
            
            # Fall back to direct analysis
            stimuli = self.analyze_text_sentiment(text)
            
            # Find dominant emotion
            dominant_emotion = max(stimuli.items(), key=lambda x: x[1])[0] if stimuli else "neutral"
            intensity = sum(stimuli.values())
            
            # Calculate valence
            valence = sum(self.emotion_memory_valence_map.get(emotion, 0) * value 
                         for emotion, value in stimuli.items())
            
            return {
                "emotions": stimuli,
                "dominant_emotion": dominant_emotion,
                "intensity": intensity,
                "valence": max(-1.0, min(1.0, valence))
            }
    
    async def update_emotion_async(self, emotion: str, value: float) -> Dict[str, Any]:
        """
        Async wrapper for update_emotion with enhanced return format
        
        Args:
            emotion: The emotion to update
            value: The delta change in emotion value (-1.0 to 1.0)
        
        Returns:
            Dictionary with update results
        """
        with trace(workflow_name="EmotionUpdate"):
            # Run the emotion agent with an update request
            agent_input = {
                "role": "user",
                "content": f"Update emotion {emotion} by {value}"
            }
            
            result = await Runner.run(
                self.emotion_agent,
                agent_input
            )
            
            # Check if we have a structured result
            if hasattr(result, "final_output") and result.final_output:
                return result.final_output
            
            # Fall back to direct update
            return await self._update_emotion_tool(
                RunContextWrapper(context=None),
                emotion=emotion,
                value=value
            )
    
    async def set_emotion_async(self, emotion: str, value: float) -> Dict[str, Any]:
        """
        Async wrapper for set_emotion with enhanced return format
        
        Args:
            emotion: The emotion to set
            value: The absolute value (0.0 to 1.0)
        
        Returns:
            Dictionary with update results
        """
        with trace(workflow_name="EmotionSet"):
            # Run the emotion agent with a set request
            agent_input = {
                "role": "user",
                "content": f"Set emotion {emotion} to {value}"
            }
            
            result = await Runner.run(
                self.emotion_agent,
                agent_input
            )
            
            # Check if we have a structured result
            if hasattr(result, "final_output") and result.final_output:
                return result.final_output
            
            # Fall back to direct update
            return await self._set_emotion_tool(
                RunContextWrapper(context=None),
                emotion=emotion,
                value=value
            )
    
    async def get_formatted_emotional_state_async(self) -> Dict[str, Any]:
        """Async wrapper for get_formatted_emotional_state for function tool compatibility"""
        with trace(workflow_name="GetEmotionalState"):
            # Run the emotion agent for state retrieval
            agent_input = {
                "role": "user",
                "content": "Get current emotional state"
            }
            
            result = await Runner.run(
                self.emotion_agent,
                agent_input
            )
            
            # Check if we have a structured result
            if hasattr(result, "final_output") and result.final_output:
                return result.final_output
            
            # Fall back to direct retrieval
            return self.get_formatted_emotional_state()
