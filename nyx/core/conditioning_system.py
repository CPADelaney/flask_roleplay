# nyx/core/conditioning_system.py

import logging
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

class ConditionedAssociation(BaseModel):
    """Represents a conditioned association between stimuli and responses"""
    stimulus: str = Field(..., description="The triggering stimulus")
    response: str = Field(..., description="The conditioned response")
    association_strength: float = Field(0.0, description="Strength of the association (0.0-1.0)")
    formation_date: str = Field(..., description="When this association was formed")
    last_reinforced: str = Field(..., description="When this association was last reinforced")
    reinforcement_count: int = Field(0, description="Number of times this association has been reinforced")
    valence: float = Field(0.0, description="Emotional valence of this association (-1.0 to 1.0)")
    context_keys: List[str] = Field(default_factory=list, description="Contextual keys where this association applies")
    decay_rate: float = Field(0.05, description="Rate at which this association decays if not reinforced")

class ConditioningSystem:
    """
    System for implementing classical and operant conditioning mechanisms
    to shape AI personality, preferences, and behaviors.
    """
    
    def __init__(self, reward_system, emotional_core=None, memory_core=None, somatosensory_system=None):
        self.reward_system = reward_system
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        self.somatosensory_system = somatosensory_system
        
        # Classical conditioning associations (stimulus → response)
        self.classical_associations = {}  # stimulus → ConditionedAssociation
        
        # Operant conditioning associations (behavior → consequences)
        self.operant_associations = {}  # behavior → ConditionedAssociation
        
        # Association strength thresholds
        self.weak_association_threshold = 0.3
        self.moderate_association_threshold = 0.6
        self.strong_association_threshold = 0.8
        
        # Learning parameters
        self.association_learning_rate = 0.1
        self.extinction_rate = 0.05
        self.generalization_factor = 0.3
        
        # Tracking metrics
        self.total_associations = 0
        self.total_reinforcements = 0
        self.successful_associations = 0
        
        logger.info("Conditioning system initialized")
    
    async def process_classical_conditioning(self, 
                                           unconditioned_stimulus: str,
                                           conditioned_stimulus: str,
                                           response: str,
                                           intensity: float = 1.0,
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a classical conditioning event where an unconditioned stimulus 
        is paired with a conditioned stimulus to create an association.
        
        Args:
            unconditioned_stimulus: The natural stimulus that triggers the response
            conditioned_stimulus: The neutral stimulus to be conditioned
            response: The response to be conditioned
            intensity: The intensity of the unconditioned stimulus (0.0-1.0)
            context: Additional contextual information
            
        Returns:
            Processing results
        """
        context = context or {}
        
        # Create a unique key for this association
        association_key = f"{conditioned_stimulus}→{response}"
        
        # Check if this association already exists
        if association_key in self.classical_associations:
            # Get existing association
            association = self.classical_associations[association_key]
            
            # Update association strength based on intensity and learning rate
            old_strength = association.association_strength
            new_strength = min(1.0, old_strength + (intensity * self.association_learning_rate))
            
            # Update association data
            association.association_strength = new_strength
            association.last_reinforced = datetime.datetime.now().isoformat()
            association.reinforcement_count += 1
            
            # Extract context keys if provided
            if context and "context_keys" in context:
                for key in context["context_keys"]:
                    if key not in association.context_keys:
                        association.context_keys.append(key)
            
            # Record reinforcement
            self.total_reinforcements += 1
            
            logger.info(f"Reinforced classical association: {association_key} ({old_strength:.2f} → {new_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "reinforcement",
                "old_strength": old_strength,
                "new_strength": new_strength,
                "reinforcement_count": association.reinforcement_count
            }
        else:
            # Create new association
            association = ConditionedAssociation(
                stimulus=conditioned_stimulus,
                response=response,
                association_strength=intensity * self.association_learning_rate,
                formation_date=datetime.datetime.now().isoformat(),
                last_reinforced=datetime.datetime.now().isoformat(),
                reinforcement_count=1,
                valence=context.get("valence", 0.0),
                context_keys=context.get("context_keys", [])
            )
            
            # Store the association
            self.classical_associations[association_key] = association
            self.total_associations += 1
            
            logger.info(f"Created new classical association: {association_key} ({association.association_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "new_association",
                "strength": association.association_strength,
                "reinforcement_count": 1
            }
    
    async def process_operant_conditioning(self,
                                        behavior: str,
                                        consequence_type: str,
                                        intensity: float = 1.0,
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process an operant conditioning event where a behavior 
        is reinforced or punished based on its consequences.
        
        Args:
            behavior: The behavior being conditioned
            consequence_type: Type of consequence (positive_reinforcement, negative_reinforcement, 
                             positive_punishment, negative_punishment)
            intensity: The intensity of the consequence (0.0-1.0)
            context: Additional contextual information
            
        Returns:
            Processing results
        """
        context = context or {}
        
        # Determine if this is reinforcement or punishment
        is_reinforcement = "reinforcement" in consequence_type
        is_positive = "positive" in consequence_type
        
        # Calculate the valence and reward value
        valence = intensity * (1.0 if is_reinforcement else -1.0)
        reward_value = intensity * (1.0 if is_reinforcement else -0.8)  # Punishments slightly less impactful
        
        # Create a unique key for this association
        association_key = f"{behavior}→{consequence_type}"
        
        # Generate reward signal for this conditioning event
        if self.reward_system:
            reward_signal = RewardSignal(
                value=reward_value,
                source="operant_conditioning",
                context={
                    "behavior": behavior,
                    "consequence_type": consequence_type,
                    "intensity": intensity,
                    **context
                }
            )
            await self.reward_system.process_reward_signal(reward_signal)
        
        # Update or create the association
        if association_key in self.operant_associations:
            # Get existing association
            association = self.operant_associations[association_key]
            
            # Calculate strength change based on intensity and whether it's reinforcement or punishment
            strength_change = intensity * self.association_learning_rate
            if not is_reinforcement:
                strength_change *= -1  # Decrease strength for punishment
            
            # Update association strength
            old_strength = association.association_strength
            new_strength = max(0.0, min(1.0, old_strength + strength_change))
            
            # Update association data
            association.association_strength = new_strength
            association.last_reinforced = datetime.datetime.now().isoformat()
            association.reinforcement_count += 1
            association.valence = (association.valence + valence) / 2  # Average valence
            
            # Extract context keys if provided
            if context and "context_keys" in context:
                for key in context["context_keys"]:
                    if key not in association.context_keys:
                        association.context_keys.append(key)
            
            # Record reinforcement
            self.total_reinforcements += 1
            
            logger.info(f"Updated operant association: {association_key} ({old_strength:.2f} → {new_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "update",
                "old_strength": old_strength,
                "new_strength": new_strength,
                "reinforcement_count": association.reinforcement_count,
                "is_reinforcement": is_reinforcement,
                "is_positive": is_positive
            }
        else:
            # Create new association with initial strength based on intensity
            initial_strength = intensity * self.association_learning_rate
            if not is_reinforcement:
                initial_strength *= 0.5  # Start weaker for punishment
            
            association = ConditionedAssociation(
                stimulus=behavior,  # For operant, the behavior is the stimulus
                response=consequence_type,  # And the consequence is the response
                association_strength=initial_strength,
                formation_date=datetime.datetime.now().isoformat(),
                last_reinforced=datetime.datetime.now().isoformat(),
                reinforcement_count=1,
                valence=valence,
                context_keys=context.get("context_keys", [])
            )
            
            # Store the association
            self.operant_associations[association_key] = association
            self.total_associations += 1
            
            logger.info(f"Created new operant association: {association_key} ({association.association_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "new_association",
                "strength": association.association_strength,
                "reinforcement_count": 1,
                "is_reinforcement": is_reinforcement,
                "is_positive": is_positive
            }
    
    async def trigger_conditioned_response(self, 
                                        stimulus: str,
                                        context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Trigger conditioned responses based on a stimulus
        
        Args:
            stimulus: The stimulus that might trigger conditioned responses
            context: Additional contextual information
            
        Returns:
            Dictionary with triggered responses, or None if no responses were triggered
        """
        context = context or {}
        
        # Check for classical conditioning associations
        matched_associations = []
        for key, association in self.classical_associations.items():
            if association.stimulus == stimulus:
                # Check context match if context keys are present
                context_match = True
                if association.context_keys:
                    # Context keys are required but not provided
                    if not context:
                        context_match = False
                    else:
                        # Check if any required context keys are missing
                        for required_key in association.context_keys:
                            if required_key not in context:
                                context_match = False
                                break
                
                # Only include if context matches and strength is above threshold
                if context_match and association.association_strength >= self.weak_association_threshold:
                    matched_associations.append((key, association))
        
        if not matched_associations:
            return None
            
        # Sort by association strength (strongest first)
        matched_associations.sort(key=lambda x: x[1].association_strength, reverse=True)
        
        # Prepare responses
        responses = []
        for key, association in matched_associations:
            # Determine if association is triggered based on strength
            # Stronger associations are more likely to be triggered
            trigger_threshold = random.random() * (1.0 - self.weak_association_threshold) + self.weak_association_threshold
            
            if association.association_strength >= trigger_threshold:
                responses.append({
                    "association_key": key,
                    "response": association.response,
                    "strength": association.association_strength,
                    "valence": association.valence
                })
                
                # Record successful association
                self.successful_associations += 1
                
                # Apply effects based on association (emotional, physical, etc.)
                await self._apply_association_effects(association)
        
        if not responses:
            return None
            
        return {
            "stimulus": stimulus,
            "triggered_responses": responses,
            "context": context
        }
    
    async def evaluate_behavior_consequences(self,
                                          behavior: str,
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate the likely consequences of a behavior based on operant conditioning history
        
        Args:
            behavior: The behavior to evaluate
            context: Additional contextual information
            
        Returns:
            Evaluation of behavior consequences
        """
        context = context or {}
        
        # Find all associations related to this behavior
        behavior_associations = []
        for key, association in self.operant_associations.items():
            if association.stimulus == behavior:
                # Check context match if context keys are present
                context_match = True
                if association.context_keys:
                    # Context keys are required but not provided
                    if not context:
                        context_match = False
                    else:
                        # Check if any required context keys are missing
                        for required_key in association.context_keys:
                            if required_key not in context:
                                context_match = False
                                break
                
                # Only include if context matches
                if context_match:
                    behavior_associations.append((key, association))
        
        # If no associations found, return neutral evaluation
        if not behavior_associations:
            return {
                "behavior": behavior,
                "has_associations": False,
                "expected_valence": 0.0,
                "confidence": 0.1,
                "recommendation": "neutral"
            }
        
        # Calculate the expected outcomes
        total_strength = sum(assoc.association_strength for _, assoc in behavior_associations)
        weighted_valence = sum(assoc.association_strength * assoc.valence for _, assoc in behavior_associations)
        
        if total_strength > 0:
            expected_valence = weighted_valence / total_strength
        else:
            expected_valence = 0.0
        
        # Calculate confidence based on total strength and number of reinforcements
        total_reinforcements = sum(assoc.reinforcement_count for _, assoc in behavior_associations)
        confidence = min(0.9, (total_strength / len(behavior_associations)) * 0.7 + (min(10, total_reinforcements) / 10) * 0.3)
        
        # Generate recommendation
        if expected_valence > 0.3 and confidence > 0.4:
            recommendation = "approach"  # Positive expected outcome
        elif expected_valence < -0.3 and confidence > 0.4:
            recommendation = "avoid"  # Negative expected outcome
        else:
            recommendation = "neutral"  # Neutral or uncertain
        
        return {
            "behavior": behavior,
            "has_associations": True,
            "associations_count": len(behavior_associations),
            "expected_valence": expected_valence,
            "total_strength": total_strength,
            "confidence": confidence,
            "total_reinforcements": total_reinforcements,
            "recommendation": recommendation
        }
    
    async def _apply_association_effects(self, association: ConditionedAssociation) -> None:
        """Apply effects from a triggered association"""
        # Determine intensity based on association strength
        intensity = association.association_strength * 0.8  # Scale to avoid maximum intensity
        
        # Apply emotional effects if emotional core is available
        if self.emotional_core and association.valence != 0.0:
            # Determine which neurochemicals to update based on valence
            if association.valence > 0:
                # Positive association - increase pleasure/reward chemicals
                await self.emotional_core.update_neurochemical("nyxamine", intensity * 0.7)
                await self.emotional_core.update_neurochemical("seranix", intensity * 0.3)
            else:
                # Negative association - increase stress/defense chemicals
                await self.emotional_core.update_neurochemical("cortanyx", intensity * 0.6)
                await self.emotional_core.update_neurochemical("adrenyx", intensity * 0.4)
        
        # Apply physical effects if somatosensory system is available
        if self.somatosensory_system:
            if association.valence > 0:
                # Positive association - pleasure sensation
                await self.somatosensory_system.process_stimulus(
                    stimulus_type="pleasure",
                    body_region="core",  # Default region
                    intensity=intensity,
                    cause="conditioned_response"
                )
            elif association.valence < 0:
                # Negative association - discomfort/tension
                await self.somatosensory_system.process_stimulus(
                    stimulus_type="pressure",  # Use pressure for tension
                    body_region="core",
                    intensity=intensity * 0.8,
                    cause="conditioned_response"
                )
    
    async def apply_extinction(self, association_key: str, association_type: str = "classical") -> Dict[str, Any]:
        """
        Apply extinction to an association (weaken it over time without reinforcement)
        
        Args:
            association_key: Key of the association to extinguish
            association_type: Type of association (classical or operant)
            
        Returns:
            Extinction results
        """
        # Get the appropriate association dictionary
        associations = self.classical_associations if association_type == "classical" else self.operant_associations
        
        if association_key not in associations:
            return {
                "success": False,
                "message": f"Association {association_key} not found"
            }
        
        # Get the association
        association = associations[association_key]
        
        # Calculate time since last reinforcement
        last_reinforced = datetime.datetime.fromisoformat(association.last_reinforced.replace("Z", "+00:00"))
        time_since_reinforcement = (datetime.datetime.now() - last_reinforced).total_seconds() / 86400.0  # Days
        
        # Calculate extinction effect based on time and extinction rate
        extinction_effect = min(0.9, time_since_reinforcement * association.decay_rate)
        
        # Apply extinction
        old_strength = association.association_strength
        new_strength = max(0.0, old_strength - extinction_effect)
        
        # Update association
        association.association_strength = new_strength
        
        # Remove association if strength is too low
        if new_strength < 0.05:
            del associations[association_key]
            return {
                "success": True,
                "message": f"Association {association_key} removed due to extinction",
                "old_strength": old_strength,
                "extinction_effect": extinction_effect
            }
        
        return {
            "success": True,
            "message": f"Applied extinction to {association_key}",
            "old_strength": old_strength,
            "new_strength": new_strength,
            "extinction_effect": extinction_effect
        }
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on conditioning system - apply extinction, etc.
        
        Returns:
            Maintenance results
        """
        # Apply extinction to all associations
        classical_updates = 0
        classical_removals = 0
        
        for key in list(self.classical_associations.keys()):
            result = await self.apply_extinction(key, "classical")
            if result["success"]:
                if "removed" in result["message"]:
                    classical_removals += 1
                else:
                    classical_updates += 1
        
        operant_updates = 0
        operant_removals = 0
        
        for key in list(self.operant_associations.keys()):
            result = await self.apply_extinction(key, "operant")
            if result["success"]:
                if "removed" in result["message"]:
                    operant_removals += 1
                else:
                    operant_updates += 1
        
        return {
            "classical_updates": classical_updates,
            "classical_removals": classical_removals,
            "operant_updates": operant_updates,
            "operant_removals": operant_removals,
            "total_associations": len(self.classical_associations) + len(self.operant_associations)
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conditioning system"""
        return {
            "classical_associations": len(self.classical_associations),
            "operant_associations": len(self.operant_associations),
            "total_associations": self.total_associations,
            "total_reinforcements": self.total_reinforcements,
            "successful_associations": self.successful_associations,
            "learning_parameters": {
                "association_learning_rate": self.association_learning_rate,
                "extinction_rate": self.extinction_rate,
                "generalization_factor": self.generalization_factor
            }
        }
    
    async def condition_preference(self, 
                                 stimulus: str, 
                                 preference_type: str,
                                 value: float,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Condition a preference for or against a stimulus
        
        Args:
            stimulus: The stimulus to condition
            preference_type: Type of preference (like, dislike, want, avoid, etc.)
            value: Value of preference (-1.0 to 1.0, negative for aversion)
            context: Additional contextual information
            
        Returns:
            Conditioning results
        """
        context = context or {}
        
        # Determine if this is a positive or negative preference
        is_positive = value > 0
        
        # Choose appropriate response based on preference type and value
        if preference_type == "like" or preference_type == "dislike":
            response = "emotional_response"
        elif preference_type == "want" or preference_type == "avoid":
            response = "behavioral_response"
        else:
            response = "general_response"
        
        # Condition the appropriate association
        if is_positive:
            # For positive preferences, use positive reinforcement
            result = await self.process_operant_conditioning(
                behavior=f"encounter_{stimulus}",
                consequence_type="positive_reinforcement",
                intensity=abs(value),
                context={
                    "preference_type": preference_type,
                    "valence": value,
                    "context_keys": context.get("context_keys", [])
                }
            )
        else:
            # For negative preferences, use positive punishment
            result = await self.process_operant_conditioning(
                behavior=f"encounter_{stimulus}",
                consequence_type="positive_punishment",
                intensity=abs(value),
                context={
                    "preference_type": preference_type,
                    "valence": value,
                    "context_keys": context.get("context_keys", [])
                }
            )
        
        # Also create a classical association
        classical_result = await self.process_classical_conditioning(
            unconditioned_stimulus=preference_type,
            conditioned_stimulus=stimulus,
            response=response,
            intensity=abs(value),
            context={
                "valence": value,
                "context_keys": context.get("context_keys", [])
            }
        )
        
        # If identity evolution is available, update identity
        if self.identity_evolution and abs(value) > 0.7:
            try:
                # Update preference in identity
                await self.identity_evolution.update_preference(
                    category="stimuli",
                    preference=stimulus,
                    impact=value * 0.3  # Scale down the impact
                )
                
                # Update related trait if value is strong enough
                if abs(value) > 0.8:
                    trait = "openness" if is_positive else "caution"
                    await self.identity_evolution.update_trait(
                        trait=trait,
                        impact=value * 0.1  # Small trait impact
                    )
            except Exception as e:
                logger.error(f"Error updating identity from preference: {e}")
        
        return {
            "stimulus": stimulus,
            "preference_type": preference_type,
            "value": value,
            "operant_result": result,
            "classical_result": classical_result
        }
    
    async def condition_personality_trait(self,
                                       trait: str,
                                       value: float,
                                       behaviors: List[str] = None,
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Condition a personality trait through reinforcement of related behaviors
        
        Args:
            trait: The personality trait to condition
            value: The target value for the trait (-1.0 to 1.0)
            behaviors: List of behaviors associated with this trait
            context: Additional contextual information
            
        Returns:
            Conditioning results
        """
        context = context or {}
        
        # Default behaviors if none provided
        if not behaviors:
            behaviors = [f"{trait}_behavior"]
        
        results = []
        
        # Process each behavior
        for behavior in behaviors:
            # Determine reinforcement type based on desired trait value
            if value > 0:
                # Positive trait value - reinforce related behaviors
                result = await self.process_operant_conditioning(
                    behavior=behavior,
                    consequence_type="positive_reinforcement",
                    intensity=abs(value),
                    context={
                        "trait": trait,
                        "valence": value,
                        "context_keys": context.get("context_keys", [])
                    }
                )
            else:
                # Negative trait value - punish related behaviors
                result = await self.process_operant_conditioning(
                    behavior=behavior,
                    consequence_type="positive_punishment",
                    intensity=abs(value),
                    context={
                        "trait": trait,
                        "valence": value,
                        "context_keys": context.get("context_keys", [])
                    }
                )
            
            results.append(result)
        
        # If identity evolution is available, update the trait directly
        if self.identity_evolution and abs(value) > 0.6:
            try:
                await self.identity_evolution.update_trait(
                    trait=trait,
                    impact=value * 0.5  # Stronger direct impact
                )
            except Exception as e:
                logger.error(f"Error updating identity trait: {e}")
        
        return {
            "trait": trait,
            "value": value,
            "behaviors": behaviors,
            "results": results
        }
    
    async def create_emotion_trigger(self,
                                   trigger: str,
                                   emotion: str,
                                   intensity: float = 0.5,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a trigger for an emotional response
        
        Args:
            trigger: The stimulus that will trigger the emotion
            emotion: The emotion to be triggered
            intensity: The intensity of the emotion (0.0-1.0)
            context: Additional contextual information
            
        Returns:
            Results of creating the emotion trigger
        """
        context = context or {}
        
        # Use classical conditioning to associate trigger with emotion
        result = await self.process_classical_conditioning(
            unconditioned_stimulus="emotional_stimulus",
            conditioned_stimulus=trigger,
            response=f"emotion_{emotion}",
            intensity=intensity,
            context={
                "emotion": emotion,
                "valence": context.get("valence", 0.0),
                "context_keys": context.get("context_keys", [])
            }
        )
        
        # If emotional core is available, create a test activation
        if self.emotional_core:
            try:
                # Map emotion to neurochemicals
                chemical_map = {
                    "joy": "nyxamine",
                    "contentment": "seranix",
                    "trust": "oxynixin",
                    "fear": "adrenyx",
                    "anger": "cortanyx",
                    "sadness": "cortanyx"
                }
                
                # Update appropriate neurochemical if emotion is mapped
                emotion_lower = emotion.lower()
                if emotion_lower in chemical_map:
                    chemical = chemical_map[emotion_lower]
                    test_intensity = intensity * 0.1  # Very mild test activation
                    
                    await self.emotional_core.update_neurochemical(
                        chemical=chemical,
                        value=test_intensity
                    )
            except Exception as e:
                logger.error(f"Error creating test emotion activation: {e}")
        
        return {
            "trigger": trigger,
            "emotion": emotion,
            "intensity": intensity,
            "association_result": result
        }
