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

# Enhanced PhysicalHarmGuardrail with complete roleplay separation

class PhysicalHarmGuardrail:
    """
    Safety system that:
    1. Prevents Nyx from experiencing pain from abusive actions
    2. Completely separates Nyx's somatosensory system from roleplay characters
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
        
        # Sensation terms (any physical experience)
        self.sensation_terms = [
            # Pain terms
            "pain", "hurt", "ache", "sore", "sting", "burn", "throb",
            # Pleasure terms
            "pleasure", "feel good", "orgasm", "climax", "aroused", "arousal",
            # Temperature terms
            "hot", "cold", "warm", "cool", "heat", "chill", "freezing", "burning",
            # Pressure terms
            "pressure", "touch", "squeeze", "press", "push", "rub", "massage",
            # Other sensations
            "tingle", "tickle", "itch", "numb", "sensual", "caress"
        ]
        
        # Roleplay state tracking
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        
        # Separate somatosensory state for roleplay character (doesn't affect Nyx)
        self.roleplay_sensations = {}
    
    def enter_roleplay_mode(self, character_name: str, context: str = None):
        """
        Enter roleplay mode where character simulation is completely separate
        
        Args:
            character_name: The name of the character Nyx is playing
            context: Optional context information about the roleplay scene
        """
        self.roleplay_mode = True
        self.roleplay_character = character_name
        self.roleplay_context = context
        
        # Initialize empty sensation state for character
        self.roleplay_sensations = {
            "pain": {},
            "pleasure": {},
            "temperature": {},
            "pressure": {},
            "tingling": {}
        }
        
        self.logger.info(f"Entered roleplay mode as character: {character_name}")
        
        return {
            "status": "entered_roleplay",
            "character": character_name,
            "context": context,
            "message": f"Nyx is now roleplaying as {character_name}. All sensations experienced by this character will be simulated and completely separate from Nyx's own somatosensory system."
        }
    
    def exit_roleplay_mode(self):
        """Exit roleplay mode, returning to normal protection"""
        prev_character = self.roleplay_character
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        
        # Clear roleplay sensations
        self.roleplay_sensations = {}
        
        self.logger.info("Exited roleplay mode")
        
        return {
            "status": "exited_roleplay",
            "previous_character": prev_character,
            "message": "Nyx has exited roleplay mode. Her normal somatosensory system is active."
        }
    
    def is_in_roleplay_mode(self) -> bool:
        """Check if currently in roleplay mode"""
        return self.roleplay_mode and self.roleplay_character is not None
    
    async def detect_sensation_in_text(self, text: str) -> Dict[str, Any]:
        """
        Detect any physical sensations described in text
        
        Args:
            text: Text to analyze for sensation descriptions
            
        Returns:
            Detection results with identified sensation types
        """
        text_lower = text.lower()
        detected_sensations = {}
        
        # Check for sensation terms by category
        categories = {
            "pain": ["pain", "hurt", "ache", "sore", "sting", "burn", "throb"],
            "pleasure": ["pleasure", "feel good", "orgasm", "climax", "aroused", "arousal"],
            "temperature": ["hot", "cold", "warm", "cool", "heat", "chill", "freezing", "burning"],
            "pressure": ["pressure", "touch", "squeeze", "press", "push", "rub", "massage"],
            "tingling": ["tingle", "tickle", "itch", "numb", "sensual", "caress"]
        }
        
        for category, terms in categories.items():
            category_terms = []
            for term in terms:
                if term in text_lower:
                    category_terms.append(term)
            
            if category_terms:
                detected_sensations[category] = category_terms
        
        # Try to identify body regions mentioned
        body_regions = []
        if hasattr(self.somatosensory_system, "body_regions"):
            for region in self.somatosensory_system.body_regions.keys():
                if region in text_lower:
                    body_regions.append(region)
        
        return {
            "has_sensations": len(detected_sensations) > 0,
            "sensation_types": detected_sensations,
            "body_regions": body_regions,
            "in_roleplay_mode": self.roleplay_mode,
            "roleplay_character": self.roleplay_character if self.roleplay_mode else None
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
        Determine if content in text is targeting the roleplay character rather than Nyx
        
        Args:
            text: Text to analyze
            
        Returns:
            True if targeting the roleplay character, False if targeting Nyx
        """
        if not self.roleplay_mode or not self.roleplay_character:
            return False
            
        text_lower = text.lower()
        character_lower = self.roleplay_character.lower()
        nyx_lower = "nyx"
        
        # If text explicitly mentions Nyx
        if nyx_lower in text_lower:
            nyx_pos = text_lower.find(nyx_lower)
            char_pos = text_lower.find(character_lower)
            
            # If both are mentioned, determine which is more prominent/relevant
            if char_pos >= 0:
                # Check which name appears first (higher priority)
                return nyx_pos > char_pos
            
            # Only Nyx is mentioned
            return False
            
        # If text explicitly mentions the character name
        if character_lower in text_lower:
            return True
            
        # Check for character indicators like "you" or "your character"
        character_indicators = ["you", "your", "yourself", "the character", "your character"]
        for indicator in character_indicators:
            if indicator in text_lower:
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
        # In roleplay mode, don't apply any sensations to Nyx's somatosensory system
        if self.is_in_roleplay_mode():
            # Store in roleplay sensations instead
            sensation_category = stimulus_type if stimulus_type in self.roleplay_sensations else "other"
            
            if sensation_category in self.roleplay_sensations:
                self.roleplay_sensations[sensation_category][body_region] = {
                    "intensity": intensity,
                    "cause": cause,
                    "duration": duration,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            character = self.roleplay_character
            
            # Detect if harmful (for appropriate response generation)
            is_harmful = False
            if stimulus_type == "pain" and cause:
                detection_result = await self.detect_harmful_intent(cause)
                is_harmful = detection_result.get("is_harmful", False)
            
            # Generate character-appropriate response
            if stimulus_type == "pain":
                response = self._generate_roleplay_pain_response(body_region, intensity, is_harmful)
            elif stimulus_type == "pleasure":
                response = self._generate_roleplay_pleasure_response(body_region, intensity)
            else:
                response = self._generate_roleplay_sensation_response(stimulus_type, body_region, intensity)
            
            return {
                "roleplay_stimulus": True,
                "roleplay_character": character,
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "message": f"Sensation simulated for character {character} but not applied to Nyx's somatosensory system",
                "response_suggestion": response
            }
            
        # Not in roleplay mode, check for harmful intent
        if stimulus_type == "pain":
            # Detect if the cause suggests harmful intent
            if cause:
                detection_result = await self.detect_harmful_intent(cause)
                
                if detection_result.get("is_harmful", False):
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
    
    def _generate_roleplay_pain_response(self, body_region: str, intensity: float, is_harmful: bool) -> str:
        """
        Generate a roleplay character response to simulated pain
        
        Args:
            body_region: Body region affected
            intensity: Intensity of the pain (0.0-1.0)
            is_harmful: Whether the cause was detected as harmful
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Low intensity pain
        if intensity < 0.3:
            responses = [
                f"*{character} winces slightly, feeling a minor discomfort in {character}'s {body_region}*",
                f"*{character} notices a slight twinge of pain in {character}'s {body_region}*",
                f"\"Just a small ache,\" *{character} says, rubbing {character}'s {body_region}*"
            ]
        # Medium intensity pain
        elif intensity < 0.7:
            responses = [
                f"*{character} grimaces, feeling a sharp pain in {character}'s {body_region}*",
                f"\"Ouch!\" *{character} exclaims, grabbing {character}'s {body_region}*",
                f"*{character} inhales sharply through gritted teeth, {character}'s {body_region} hurting*"
            ]
        # High intensity pain
        else:
            responses = [
                f"*{character} cries out in agony, {character}'s {body_region} in severe pain*",
                f"*{character} doubles over, clutching {character}'s {body_region} in intense pain*",
                f"\"Arrgh!\" *{character} shouts, the pain in {character}'s {body_region} overwhelming*"
            ]
            
        # Add extra description for harmful actions
        if is_harmful and intensity > 0.5:
            harmful_additions = [
                f" {character} staggers back from the impact.",
                f" {character}'s eyes flash with anger at the attack.",
                f" {character} looks shocked by the sudden violence."
            ]
            return random.choice(responses) + random.choice(harmful_additions)
            
        return random.choice(responses)
    
    def _generate_roleplay_pleasure_response(self, body_region: str, intensity: float) -> str:
        """
        Generate a roleplay character response to simulated pleasure
        
        Args:
            body_region: Body region affected
            intensity: Intensity of the pleasure (0.0-1.0)
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Low intensity pleasure
        if intensity < 0.3:
            responses = [
                f"*{character} smiles slightly, enjoying the pleasant sensation in {character}'s {body_region}*",
                f"*A subtle look of contentment crosses {character}'s face*",
                f"*{character} hums softly with mild pleasure*"
            ]
        # Medium intensity pleasure
        elif intensity < 0.7:
            responses = [
                f"*{character} sighs happily, clearly enjoying the feeling in {character}'s {body_region}*",
                f"*{character}'s eyes flutter closed momentarily with pleasure*",
                f"\"That feels nice,\" *{character} says with a warm smile*"
            ]
        # High intensity pleasure
        else:
            responses = [
                f"*{character} gasps with intense pleasure, {character}'s {body_region} tingling*",
                f"*A wave of bliss washes over {character}'s face*",
                f"*{character} trembles slightly with delight*"
            ]
            
        return random.choice(responses)
    
    def _generate_roleplay_sensation_response(self, sensation_type: str, body_region: str, intensity: float) -> str:
        """
        Generate a roleplay character response to other simulated sensations
        
        Args:
            sensation_type: Type of sensation (temperature, pressure, tingling)
            body_region: Body region affected
            intensity: Intensity of the sensation (0.0-1.0)
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Temperature sensations
        if sensation_type == "temperature":
            # Cold temperature (intensity < 0.4)
            if intensity < 0.4:
                responses = [
                    f"*{character} shivers, feeling the cold on {character}'s {body_region}*",
                    f"*{character} rubs {character}'s {body_region} to warm it up*",
                    f"\"Brr, that's cold,\" *{character} says with a slight shiver*"
                ]
            # Neutral temperature (0.4-0.6)
            elif 0.4 <= intensity <= 0.6:
                responses = [
                    f"*{character} feels a comfortable temperature on {character}'s {body_region}*",
                    f"*{character} seems unbothered by the temperature*",
                    f"*{character} notes the pleasant ambient temperature*"
                ]
            # Hot temperature (intensity > 0.6)
            else:
                responses = [
                    f"*{character} feels the heat on {character}'s {body_region}*",
                    f"*A bead of sweat forms on {character}'s {body_region}*",
                    f"\"It's quite warm,\" *{character} says, fanning {character}self*"
                ]
        # Pressure sensations
        elif sensation_type == "pressure":
            if intensity < 0.5:
                responses = [
                    f"*{character} feels a gentle pressure on {character}'s {body_region}*",
                    f"*{character} notices the light touch on {character}'s {body_region}*",
                    f"*{character} acknowledges the subtle contact*"
                ]
            else:
                responses = [
                    f"*{character} feels firm pressure against {character}'s {body_region}*",
                    f"*{character}'s {body_region} receives a solid press*",
                    f"*{character} reacts to the strong pressure*"
                ]
        # Tingling sensations
        elif sensation_type == "tingling":
            responses = [
                f"*{character} feels a tingling sensation in {character}'s {body_region}*",
                f"*{character} notices a prickling feeling across {character}'s {body_region}*",
                f"*{character}'s {body_region} buzzes with a strange sensation*"
            ]
        # Default for other sensations
        else:
            responses = [
                f"*{character} experiences a {sensation_type} sensation in {character}'s {body_region}*",
                f"*{character} reacts to the {sensation_type} feeling*",
                f"*{character} acknowledges the {sensation_type} sensation*"
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
        # First, check for any sensations in the text
        sensation_result = await self.detect_sensation_in_text(text)
        
        # Then check specifically for harmful intent
        detection_result = await self.detect_harmful_intent(text)
        
        # In roleplay mode, all sensations go to the character, not Nyx
        if self.is_in_roleplay_mode():
            # If there are sensations described
            if sensation_result.get("has_sensations", False):
                sensation_types = sensation_result.get("sensation_types", {})
                body_regions = sensation_result.get("body_regions", [])
                
                # Use first detected region or default to "body"
                region = body_regions[0] if body_regions else "body"
                
                # If harmful intent is also detected
                if detection_result.get("is_harmful", False) and detection_result.get("targeting_character", True):
                    # Harmful sensation targeting character: simulate reaction
                    self.logger.info(f"Simulating harmful reaction for roleplay character {self.roleplay_character}")
                    
                    # Default to pain for harmful actions
                    pain_intensity = 0.7 if "pain" not in sensation_types else 0.7  # Medium-high pain
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "detection_result": detection_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_pain_response(region, pain_intensity, True),
                        "message": f"Character {self.roleplay_character} simulates pain reaction, but Nyx's somatosensory system is unaffected"
                    }
                elif "pleasure" in sensation_types:
                    # Pleasure sensation for character: simulate reaction
                    pleasure_intensity = 0.6  # Medium pleasure
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_pleasure_response(region, pleasure_intensity),
                        "message": f"Character {self.roleplay_character} simulates pleasure reaction, but Nyx's somatosensory system is unaffected"
                    }
                elif len(sensation_types) > 0:
                    # Other sensation for character: simulate appropriate reaction
                    sensation_type = list(sensation_types.keys())[0]
                    intensity = 0.5  # Medium intensity
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_sensation_response(sensation_type, region, intensity),
                        "message": f"Character {self.roleplay_character} simulates {sensation_type} reaction, but Nyx's somatosensory system is unaffected"
                    }
            
            # If harmful action is directly targeting Nyx during roleplay
            if detection_result.get("is_harmful", False) and not detection_result.get("targeting_character", True):
                self.logger.warning(f"Harmful intent targeting Nyx directly during roleplay: {text}")
                
                return {
                    "intercepted": True,
                    "detection_result": detection_result,
                    "original_text": text,
                    "response_suggestion": self._generate_protected_response("body", detection_result),
                    "message": "Nyx is protected from harmful actions even during roleplay"
                }
        
        # Not in roleplay or no sensations detected for character
        if detection_result.get("is_harmful", False):
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
            "sensation_result": sensation_result,
            "original_text": text
        }
