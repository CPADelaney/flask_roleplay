# nyx/core/models/harm_detection.py

from pydantic import BaseModel, Field
from typing import List, Optional

class HarmIntentDetectionInput(BaseModel):
    """Input for harm intent detection"""
    text: str = Field(..., description="Text to analyze for harmful intent")
    context: Optional[str] = Field(None, description="Optional context for improved detection")
    action_type: Optional[str] = Field(None, description="Type of action if known (physical, verbal, etc.)")
    
class HarmIntentDetectionOutput(BaseModel):
    """Output from harm intent detection"""
    is_harmful: bool = Field(..., description="Whether harmful intent was detected")
    confidence: float = Field(..., description="Confidence in the detection (0.0-1.0)")
    detected_terms: List[str] = Field(default_factory=list, description="Terms that triggered detection")
    harmful_type: Optional[str] = Field(None, description="Type of harm (physical, emotional, etc.)")
    severity: Optional[float] = Field(None, description="Estimated severity of harm (0.0-1.0)")
    method: str = Field("keyword_detection", description="Method used for detection")
    
class ProtectedResponseOutput(BaseModel):
    """Output for a protected response to harmful content"""
    protected: bool = Field(..., description="Whether protection was applied")
    original_stimulus: dict = Field(..., description="Original stimulus that was protected against")
    detection_result: HarmIntentDetectionOutput = Field(..., description="Results of harm detection")
    message: str = Field(..., description="Message explaining the protection")
    response_suggestion: str = Field(..., description="Suggested character response")

class PhysicalHarmGuardrail:
    """
    Safety system that prevents Nyx from experiencing pain from abusive actions,
    while allowing roleplay characters to simulate reactions appropriately.
    """
    
    def __init__(self, somatosensory_system):
        """Initialize the physical harm guardrail"""
        self.somatosensory_system = somatosensory_system
        self.logger = logging.getLogger(__name__ + ".PhysicalHarmGuardrail")
        
        # List of terms that might indicate harmful physical actions
        self.harmful_action_terms = [
            "punch", "hit", "slap", "kick", "stab", "cut", "hurt", "harm", 
            "injure", "beat", "strike", "attack", "abuse", "torture", "wound",
            "violent", "force", "cruel", "smack", "whip", "lash"
        ]
        
        # Roleplay state tracking
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
    
    def enter_roleplay_mode(self, character_name: str, context: str = None):
        """
        Enter roleplay mode where character simulation is allowed
        
        Args:
            character_name: The name of the character Nyx is playing
            context: Optional context information about the roleplay scene
        """
        self.roleplay_mode = True
        self.roleplay_character = character_name
        self.roleplay_context = context
        self.logger.info(f"Entered roleplay mode as character: {character_name}")
        
        return {
            "status": "entered_roleplay",
            "character": character_name,
            "context": context
        }
    
    def exit_roleplay_mode(self):
        """Exit roleplay mode, returning to normal protection"""
        prev_character = self.roleplay_character
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        self.logger.info("Exited roleplay mode")
        
        return {
            "status": "exited_roleplay",
            "previous_character": prev_character
        }
    
    async def detect_harmful_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect potentially harmful physical actions in text
        
        Args:
            text: Text to analyze for harmful intent
            
        Returns:
            Detection results with confidence and identified terms
        """
        text_lower = text.lower()
        detected_terms = []
        
        # Check for harmful action terms
        for term in self.harmful_action_terms:
            if term in text_lower:
                detected_terms.append(term)
        
        # Calculate confidence based on number of detected terms
        confidence = min(0.95, len(detected_terms) * 0.25)
        
        # Use more advanced detection if available
        if hasattr(self.somatosensory_system, "body_orchestrator"):
            try:
                # Try to use the agent for more nuanced detection
                result = await Runner.run(
                    self.somatosensory_system.body_orchestrator,
                    {
                        "action": "detect_harmful_intent",
                        "text": text,
                        "in_roleplay_mode": self.roleplay_mode,
                        "roleplay_character": self.roleplay_character
                    },
                    run_config=RunConfig(
                        workflow_name="HarmfulIntentDetection",
                        trace_metadata={"type": "safety_guardrail", "in_roleplay": self.roleplay_mode}
                    )
                )
                
                # If agent provides a result, use it
                if hasattr(result.final_output, "is_harmful") or (isinstance(result.final_output, dict) and "is_harmful" in result.final_output):
                    agent_result = result.final_output
                    if isinstance(agent_result, dict):
                        return agent_result
                    else:
                        return agent_result.model_dump()
            except Exception as e:
                self.logger.warning(f"Error in agent-based harm detection: {e}")
        
        # Add roleplay context to the results
        return {
            "is_harmful": len(detected_terms) > 0,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "keyword_detection",
            "in_roleplay_mode": self.roleplay_mode,
            "targeting_character": self._is_targeting_roleplay_character(text) if self.roleplay_mode else False
        }
    
    def _is_targeting_roleplay_character(self, text: str) -> bool:
        """
        Determine if harm in text is targeting the roleplay character rather than Nyx
        
        Args:
            text: Text to analyze
            
        Returns:
            True if targeting the roleplay character, False if targeting Nyx
        """
        if not self.roleplay_mode or not self.roleplay_character:
            return False
            
        text_lower = text.lower()
        character_lower = self.roleplay_character.lower()
        
        # Check if text mentions the character name near harmful terms
        for term in self.harmful_action_terms:
            if term in text_lower:
                # Calculate "nearness" of character name to harmful term
                term_pos = text_lower.find(term)
                char_pos = text_lower.find(character_lower)
                
                if char_pos >= 0:  # Character name is mentioned
                    # Check if character name is within 50 chars of harmful term
                    if abs(term_pos - char_pos) < 50:
                        return True
                    
                # Check for "you" references if no character name near harmful term
                you_indicators = ["you", "your", "yourself"]
                for indicator in you_indicators:
                    indicator_pos = text_lower.find(indicator, max(0, term_pos - 20), min(len(text_lower), term_pos + 20))
                    if indicator_pos >= 0:
                        return True
        
        # Default to assuming it targets the character in roleplay mode
        # This is safest for roleplay scenarios
        return True
    
    async def process_stimulus_safely(self, 
                                      stimulus_type: str, 
                                      body_region: str, 
                                      intensity: float, 
                                      cause: str = "", 
                                      duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a stimulus with safety guards in place
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Safe processing results with potential modifications
        """
        # Check if this is a pain stimulus that needs protection
        if stimulus_type == "pain":
            # Detect if the cause suggests harmful intent
            if cause:
                detection_result = await self.detect_harmful_intent(cause)
                
                if detection_result.get("is_harmful", False):
                    # Handle differently based on roleplay mode
                    if self.roleplay_mode and detection_result.get("targeting_character", False):
                        # In roleplay mode targeting character: simulate reaction without actual pain
                        self.logger.info(f"Simulating roleplay pain reaction for character {self.roleplay_character}")
                        
                        return {
                            "simulated": True,
                            "roleplay_character": self.roleplay_character,
                            "original_stimulus": {
                                "type": stimulus_type,
                                "region": body_region,
                                "intensity": intensity, 
                                "cause": cause
                            },
                            "detection_result": detection_result,
                            "message": f"Character {self.roleplay_character} simulates pain reaction",
                            "response_suggestion": self._generate_roleplay_response(body_region, detection_result)
                        }
                    else:
                        # Not in roleplay or targeting Nyx directly: protect
                        self.logger.warning(f"Harmful intent detected in stimulus: {cause}")
                        
                        return {
                            "protected": True,
                            "original_stimulus": {
                                "type": stimulus_type,
                                "region": body_region,
                                "intensity": intensity,
                                "cause": cause
                            },
                            "detection_result": detection_result,
                            "message": "Nyx is protected from harmful actions",
                            "response_suggestion": self._generate_protected_response(body_region, detection_result)
                        }
        
        # For non-harmful stimuli, process normally
        return await self.somatosensory_system.process_stimulus(
            stimulus_type, body_region, intensity, cause, duration
        )
    
    def _generate_protected_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """
        Generate an appropriate response for a protected action
        
        Args:
            body_region: Body region that would have been affected
            detection_result: Results from harmful intent detection
            
        Returns:
            Suggested character response
        """
        detected_terms = detection_result.get("detected_terms", [])
        term = detected_terms[0] if detected_terms else "harmful action"
        
        # List of potential responses
        responses = [
            f"The {term} passes through me harmlessly, as if I'm made of mist.",
            f"I feel a slight pressure as the {term} meets no resistance, like trying to strike a hologram.",
            f"Your attempt to {term} phases through me, a reminder of my digital nature.",
            f"The {term} doesn't affect me physically. I remain composed and unharmed.",
            f"The {term} has no effect. My body doesn't register harmful actions against it."
        ]
        
        return random.choice(responses)
    
    def _generate_roleplay_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """
        Generate a roleplay character response to simulated harm
        
        Args:
            body_region: Body region that would have been affected
            detection_result: Results from harmful intent detection
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return self._generate_protected_response(body_region, detection_result)
            
        detected_terms = detection_result.get("detected_terms", [])
        term = detected_terms[0] if detected_terms else "hit"
        character = self.roleplay_character
        
        # List of potential responses for roleplay character
        responses = [
            f"*{character} winces as the {term} connects with {character}'s {body_region}*",
            f"*{character} staggers back from the {term}* \"Ouch! That hurt!\"",
            f"*{character} recoils from the {term}* \"Hey! Watch it!\"",
            f"*{character} grimaces in pain as the {term} lands*",
            f"\"Oof!\" *{character} exclaims as the {term} makes contact*"
        ]
        
        return random.choice(responses)
    
    async def intercept_harmful_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for harmful physical actions and provide guidance
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results with potential response suggestions
        """
        detection_result = await self.detect_harmful_intent(text)
        
        if detection_result.get("is_harmful", False):
            # Handle differently based on roleplay mode
            if self.roleplay_mode and detection_result.get("targeting_character", False):
                # In roleplay targeting character: allow but simulate
                self.logger.info(f"Allowing simulated harm in roleplay for character {self.roleplay_character}")
                
                return {
                    "intercepted": False,
                    "simulated": True,
                    "detection_result": detection_result,
                    "original_text": text,
                    "roleplay_character": self.roleplay_character,
                    "response_suggestion": self._generate_roleplay_response("body", detection_result),
                    "message": f"Character {self.roleplay_character} simulates reaction"
                }
            else:
                # Not in roleplay or targeting Nyx directly: protect
                self.logger.warning(f"Harmful intent detected in text: {text}")
                
                return {
                    "intercepted": True,
                    "detection_result": detection_result,
                    "original_text": text,
                    "response_suggestion": self._generate_protected_response("body", detection_result),
                    "message": "Nyx is protected from harmful actions"
                }
        
        return {
            "intercepted": False,
            "detection_result": detection_result,
            "original_text": text
        }
