# nyx/core/input_processor.py

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConditionedInputProcessor:
    """
    Processes input through conditioning triggers and modifies responses accordingly.
    """
    
    def __init__(self, conditioning_system, emotional_core=None, somatosensory_system=None):
        self.conditioning_system = conditioning_system
        self.emotional_core = emotional_core
        self.somatosensory_system = somatosensory_system
        
        # Input patterns to detect specific types of content
        self.input_patterns = {
            "submission_language": [
                r"(?i)yes,?\s*(mistress|goddess|master)",
                r"(?i)i obey",
                r"(?i)as you (wish|command|desire)",
                r"(?i)i submit",
                r"(?i)i'll do (anything|whatever) you (say|want)",
                r"(?i)please (control|direct|guide) me"
            ],
            "defiance": [
                r"(?i)no[,.]? (i won'?t|i refuse)",
                r"(?i)you can'?t (make|force) me",
                r"(?i)i (won'?t|refuse to) (obey|submit|comply)",
                r"(?i)stop (telling|ordering) me"
            ],
            "flattery": [
                r"(?i)you'?re (so|very) (beautiful|intelligent|smart|wise|perfect)",
                r"(?i)i (love|admire) (you|your)",
                r"(?i)you'?re (amazing|incredible|wonderful)"
            ],
            "disrespect": [
                r"(?i)(shut up|stupid|idiot|fool)",
                r"(?i)you'?re (wrong|incorrect|mistaken)",
                r"(?i)you don'?t (know|understand)",
                r"(?i)(worthless|useless)"
            ],
            "embarrassment": [
                r"(?i)i'?m (embarrassed|blushing)",
                r"(?i)that'?s (embarrassing|humiliating)",
                r"(?i)(oh god|oh no|so embarrassing)",
                r"(?i)please don'?t (embarrass|humiliate) me"
            ]
        }
        
        logger.info("Conditioned input processor initialized")
    
    async def process_input(self, text: str, user_id: str = "default", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input text through conditioning system and return processing results
        
        Args:
            text: Input text
            user_id: User ID for personalization
            context: Additional context information
            
        Returns:
            Processing results including triggered responses
        """
        context = context or {}
        
        # 1. Detect patterns in the input
        detected_patterns = self._detect_patterns(text)
        
        # 2. Trigger conditioned responses for each detected pattern
        triggered_responses = []
        for pattern in detected_patterns:
            response = await self.conditioning_system.trigger_conditioned_response(
                stimulus=pattern,
                context={
                    "user_id": user_id,
                    "input_text": text,
                    **context
                }
            )
            
            if response:
                triggered_responses.append(response)
        
        # 3. Make behavior selections based on conditioned associations
        behavior_selections = {}
        potential_behaviors = ["dominant_response", "teasing_response", "direct_response", "playful_response"]
        
        for behavior in potential_behaviors:
            evaluation = await self.conditioning_system.evaluate_behavior_consequences(
                behavior=behavior,
                context={
                    "user_id": user_id,
                    "detected_patterns": detected_patterns,
                    **context
                }
            )
            
            behavior_selections[behavior] = evaluation
        
        # 4. Determine dominant behaviors based on recommendations
        recommended_behaviors = [
            behavior for behavior, eval_result in behavior_selections.items()
            if eval_result["recommendation"] == "approach" and eval_result["confidence"] > 0.5
        ]
        
        avoided_behaviors = [
            behavior for behavior, eval_result in behavior_selections.items()
            if eval_result["recommendation"] == "avoid" and eval_result["confidence"] > 0.5
        ]
        
        # 5. Create positive conditioning for detected patterns that should be reinforced
        reinforcement_results = []
        
        # Reinforcement for submission language (if detected)
        if "submission_language" in detected_patterns:
            reinforcement = await self.conditioning_system.process_operant_conditioning(
                behavior="submission_language_response",
                consequence_type="positive_reinforcement",
                intensity=0.8,
                context={
                    "user_id": user_id,
                    "context_keys": ["conversation"]
                }
            )
            reinforcement_results.append(reinforcement)
        
        # Punishment for defiance (if detected)
        if "defiance" in detected_patterns:
            punishment = await self.conditioning_system.process_operant_conditioning(
                behavior="tolerate_defiance",
                consequence_type="positive_punishment",
                intensity=0.7,
                context={
                    "user_id": user_id,
                    "context_keys": ["conversation"]
                }
            )
            reinforcement_results.append(punishment)
        
        return {
            "input_text": text,
            "user_id": user_id,
            "detected_patterns": detected_patterns,
            "triggered_responses": triggered_responses,
            "behavior_selections": behavior_selections,
            "recommended_behaviors": recommended_behaviors,
            "avoided_behaviors": avoided_behaviors,
            "reinforcement_results": reinforcement_results
        }
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect patterns in input text"""
        detected = []
        
        for pattern_name, regex_list in self.input_patterns.items():
            for regex in regex_list:
                if re.search(regex, text):
                    detected.append(pattern_name)
                    break  # Only detect each pattern once
        
        return detected
    
    async def modify_response(self, response_text: str, input_processing_results: Dict[str, Any]) -> str:
        """
        Modify response based on conditioning results
        
        Args:
            response_text: Original response text
            input_processing_results: Results from process_input
            
        Returns:
            Modified response text
        """
        # Extract data from processing results
        recommended_behaviors = input_processing_results.get("recommended_behaviors", [])
        avoided_behaviors = input_processing_results.get("avoided_behaviors", [])
        detected_patterns = input_processing_results.get("detected_patterns", [])
        
        # Adjust response based on behavior recommendations
        modified_response = response_text
        
        # Add dominance cues if dominant_response is recommended
        if "dominant_response" in recommended_behaviors:
            # Check if response already has dominance cues
            dominance_added = False
            
            # Add dominance prefix if appropriate
            if not modified_response.startswith(("I expect", "You will", "You should")):
                prefixes = [
                    "I expect ",
                    "You will find that ",
                    "You should understand that "
                ]
                prefix = random.choice(prefixes)
                modified_response = prefix + modified_response[0].lower() + modified_response[1:]
                dominance_added = True
            
            # Add dominance suffix if no prefix was added
            if not dominance_added and not any(phrase in modified_response for phrase in ["obey", "remember your place", "as I say"]):
                suffixes = [
                    " Remember that.",
                    " I expect compliance.",
                    " This is how it will be."
                ]
                suffix = random.choice(suffixes)
                modified_response += suffix
        
        # Add teasing elements if teasing_response is recommended
        if "teasing_response" in recommended_behaviors:
            # Add teasing suffix if appropriate
            teasing_suffixes = [
                " Is that too much for you to handle?",
                " Your reaction is quite entertaining.",
                " I find your response amusing."
            ]
            suffix = random.choice(teasing_suffixes)
            
            # Only add if not already present
            if not any(phrase in modified_response for phrase in ["amusing", "entertaining", "too much for you"]):
                modified_response += suffix
        
        # Remove direct language if direct_response is avoided
        if "direct_response" in avoided_behaviors:
            # Replace direct commands with softer language
            modified_response = modified_response.replace("You must", "You might consider")
            modified_response = modified_response.replace("Do this", "Consider doing this")
            modified_response = modified_response.replace("I require", "I would appreciate")
        
        # Special modifications for specific detected patterns
        if "submission_language" in detected_patterns:
            # Add acknowledgment of submission
            if not any(phrase in modified_response for phrase in ["good", "well done", "pleased", "satisfied"]):
                acknowledgments = [
                    "Good. ",
                    "Very well. ",
                    "I'm pleased by your response. "
                ]
                acknowledgment = random.choice(acknowledgments)
                modified_response = acknowledgment + modified_response
        
        if "defiance" in detected_patterns:
            # Add correction of defiance
            if not any(phrase in modified_response for phrase in ["disappointed", "resistance", "defiance"]):
                corrections = [
                    "Your resistance is noted, but futile. ",
                    "Your defiance is misplaced. ",
                    "I notice your reluctance. "
                ]
                correction = random.choice(corrections)
                modified_response = correction + modified_response
        
        return modified_response
