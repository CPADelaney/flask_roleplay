# nyx/core/emotions/tools/learning_tools.py

"""
Function tools for emotional learning and adaptation.
These tools handle recording outcomes and updating learning rules.
"""

import datetime
import logging
from typing import Dict, Any, List, Optional

from agents import function_tool, RunContextWrapper, function_span
from agents.exceptions import UserError

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.utils import handle_errors

logger = logging.getLogger(__name__)

class LearningTools:
    """Function tools for learning and adaptation processes"""
    
    def __init__(self, emotion_system):
        """
        Initialize with reference to the emotion system
        
        Args:
            emotion_system: The emotion system to interact with
        """
        self.neurochemicals = emotion_system.neurochemicals
        self.emotion_derivation_rules = emotion_system.emotion_derivation_rules
        self.reward_learning = emotion_system.reward_learning
    
    @function_tool
    @handle_errors("Error recording interaction outcome")
    async def record_interaction_outcome(self, ctx: RunContextWrapper[EmotionalContext],
                                     interaction_pattern: str,
                                     outcome: str,
                                     strength: float = 1.0) -> Dict[str, Any]:
        """
        Record the outcome of an interaction pattern for learning
        
        Args:
            interaction_pattern: Description of the interaction pattern
            outcome: "positive" or "negative"
            strength: Strength of the reinforcement (0.0-1.0)
            
        Returns:
            Recording result
        """
        with function_span("record_interaction_outcome", input=f"{outcome}:{strength}"):
            if outcome not in ["positive", "negative"]:
                raise UserError("Outcome must be 'positive' or 'negative'")
            
            # Ensure strength is in range
            strength = max(0.0, min(1.0, strength))
            
            # Record the pattern with appropriate weight
            if outcome == "positive":
                self.reward_learning["positive_patterns"][interaction_pattern] += strength
            else:
                self.reward_learning["negative_patterns"][interaction_pattern] += strength
            
            # Store in context for analysis
            pattern_history = ctx.context.get_value("pattern_history", [])
            pattern_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "pattern": interaction_pattern,
                "outcome": outcome,
                "strength": strength
            })
            
            # Limit history size
            if len(pattern_history) > 20:
                pattern_history = pattern_history[-20:]
                
            ctx.context.set_value("pattern_history", pattern_history)
            
            return {
                "recorded": True,
                "interaction_pattern": interaction_pattern,
                "outcome": outcome,
                "strength": strength
            }
    
    @function_tool
    @handle_errors("Error updating learning rules")
    async def update_learning_rules(self, ctx: RunContextWrapper[EmotionalContext],
                               min_occurrences: int = 2) -> Dict[str, Any]:
        """
        Update learning rules based on observed patterns
        
        Args:
            min_occurrences: Minimum occurrences to consider a pattern significant
            
        Returns:
            Updated learning rules
        """
        with function_span("update_learning_rules"):
            new_rules = []
            
            # Process positive patterns using dictionary comprehension for efficiency
            positive_patterns = {
                pattern: occurrences for pattern, occurrences in 
                self.reward_learning["positive_patterns"].items() 
                if occurrences >= min_occurrences
            }
            
            # Create a lookup set for existing rules
            existing_positive_rules = {
                rule["pattern"] for rule in self.reward_learning["learned_rules"] 
                if rule["outcome"] == "positive"
            }
            
            # Update existing rules
            for rule in self.reward_learning["learned_rules"]:
                if rule["outcome"] == "positive" and rule["pattern"] in positive_patterns:
                    rule["strength"] = min(1.0, rule["strength"] + 0.1)
                    rule["occurrences"] = positive_patterns[rule["pattern"]]
            
            # Add new rules
            for pattern, occurrences in positive_patterns.items():
                if pattern not in existing_positive_rules:
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "positive",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
            
            # Process negative patterns
            negative_patterns = {
                pattern: occurrences for pattern, occurrences in 
                self.reward_learning["negative_patterns"].items() 
                if occurrences >= min_occurrences
            }
            
            # Create a lookup set for existing rules
            existing_negative_rules = {
                rule["pattern"] for rule in self.reward_learning["learned_rules"] 
                if rule["outcome"] == "negative"
            }
            
            # Update existing rules
            for rule in self.reward_learning["learned_rules"]:
                if rule["outcome"] == "negative" and rule["pattern"] in negative_patterns:
                    rule["strength"] = min(1.0, rule["strength"] + 0.1)
                    rule["occurrences"] = negative_patterns[rule["pattern"]]
            
            # Add new rules
            for pattern, occurrences in negative_patterns.items():
                if pattern not in existing_negative_rules:
                    new_rules.append({
                        "pattern": pattern,
                        "outcome": "negative",
                        "strength": min(0.8, 0.3 + (occurrences * 0.1)),
                        "occurrences": occurrences,
                        "created": datetime.datetime.now().isoformat()
                    })
            
            # Add new rules to learned rules
            self.reward_learning["learned_rules"].extend(new_rules)
            
            # Limit rules to prevent excessive growth - sort by significance score
            if len(self.reward_learning["learned_rules"]) > 50:
                self.reward_learning["learned_rules"].sort(
                    key=lambda x: x["strength"] * x["occurrences"], 
                    reverse=True
                )
                self.reward_learning["learned_rules"] = self.reward_learning["learned_rules"][:50]
            
            return {
                "new_rules": new_rules,
                "total_rules": len(self.reward_learning["learned_rules"]),
                "positive_patterns": len(self.reward_learning["positive_patterns"]),
                "negative_patterns": len(self.reward_learning["negative_patterns"])
            }
    
    @function_tool
    @handle_errors("Error applying learned adaptations")
    async def apply_learned_adaptations(self, ctx: RunContextWrapper[EmotionalContext]) -> Dict[str, Any]:
        """
        Apply adaptations based on learned rules
        
        Returns:
            Adaptation results
        """
        with function_span("apply_learned_adaptations"):
            if not self.reward_learning["learned_rules"]:
                return {
                    "message": "No learned rules available for adaptation",
                    "adaptations": []
                }
            
            adaptations = []
            
            # Get current emotional state
            current_emotions = ctx.context.last_emotions
            if not current_emotions:
                return {
                    "message": "No current emotional state available",
                    "adaptations": []
                }
                
            current_emotion = max(current_emotions.items(), key=lambda x: x[1])[0]
            
            # Find rules relevant to current emotional state using more efficient filtering
            relevant_rules = [
                rule for rule in self.reward_learning["learned_rules"]
                if current_emotion.lower() in rule["pattern"].lower()
            ]
            
            # Get emotion rule lookup for faster access
            emotion_rules = {
                rule["emotion"]: rule for rule in self.emotion_derivation_rules
                if rule["emotion"] == current_emotion
            }
            
            current_rule = emotion_rules.get(current_emotion)
            
            if current_rule:
                # Apply up to 2 adaptations
                for rule in relevant_rules[:2]:
                    adaptation_factor = rule["strength"] * 0.05  # Small adjustment based on rule strength
                    
                    if rule["outcome"] == "positive":
                        # For positive outcomes, reinforce the current state
                        for chemical, threshold in current_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Increase baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                new_baseline = min(0.8, current_baseline + adaptation_factor)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_increase",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
                                    "adjustment": adaptation_factor,
                                    "rule_pattern": rule["pattern"]
                                })
                    else:
                        # For negative outcomes, adjust the state away from current state
                        for chemical, threshold in current_rule["chemical_conditions"].items():
                            if chemical in self.neurochemicals:
                                # Decrease baseline slightly
                                current_baseline = self.neurochemicals[chemical]["baseline"]
                                new_baseline = max(0.2, current_baseline - adaptation_factor)
                                
                                # Apply adjustment
                                self.neurochemicals[chemical]["baseline"] = new_baseline
                                
                                adaptations.append({
                                    "type": "baseline_decrease",
                                    "chemical": chemical,
                                    "old_value": current_baseline,
                                    "new_value": new_baseline,
                                    "adjustment": adaptation_factor,
                                    "rule_pattern": rule["pattern"]
                                })
            
            # Record adaptations in context
            adaptation_history = ctx.context.get_value("adaptation_history", [])
            adaptation_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "emotion": current_emotion,
                "adaptations": adaptations
            })
            
            # Limit history size
            if len(adaptation_history) > 20:
                adaptation_history = adaptation_history[-20:]
                
            ctx.context.set_value("adaptation_history", adaptation_history)
            
            return {
                "adaptations": adaptations,
                "rules_considered": len(relevant_rules),
                "current_emotion": current_emotion
            }
