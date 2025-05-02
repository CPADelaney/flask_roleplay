# nyx/core/conditioning_system.py

import logging
import datetime
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import asyncio

from agents import Agent, Runner, trace, function_tool, RunContextWrapper, ModelSettings, handoff
from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

# Pydantic models for data structures
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

# Output schema models
class ClassicalConditioningOutput(BaseModel):
    """Output schema for classical conditioning analysis"""
    association_key: str = Field(..., description="Key for the association")
    type: str = Field(..., description="Type of association (new_association or reinforcement)")
    association_strength: float = Field(..., description="Strength of the association")
    reinforcement_count: int = Field(..., description="Number of reinforcements")
    valence: float = Field(..., description="Emotional valence of the association")
    explanation: str = Field(..., description="Explanation of the conditioning process")

class OperantConditioningOutput(BaseModel):
    """Output schema for operant conditioning analysis"""
    association_key: str = Field(..., description="Key for the association")
    type: str = Field(..., description="Type of association (new_association or update)")
    behavior: str = Field(..., description="The behavior being conditioned")
    consequence_type: str = Field(..., description="Type of consequence")
    association_strength: float = Field(..., description="Strength of the association")
    is_reinforcement: bool = Field(..., description="Whether this is reinforcement or punishment")
    is_positive: bool = Field(..., description="Whether this is positive or negative")
    explanation: str = Field(..., description="Explanation of the conditioning process")

class BehaviorEvaluationOutput(BaseModel):
    """Output schema for behavior evaluation"""
    behavior: str = Field(..., description="The behavior being evaluated")
    expected_valence: float = Field(..., description="Expected outcome valence (-1.0 to 1.0)")
    confidence: float = Field(..., description="Confidence in the evaluation (0.0-1.0)")
    recommendation: str = Field(..., description="Recommendation (approach, avoid, neutral)")
    explanation: str = Field(..., description="Explanation of the recommendation")
    relevant_associations: List[Dict[str, Any]] = Field(..., description="Relevant associations considered")

class TraitConditioningOutput(BaseModel):
    """Output schema for personality trait conditioning"""
    trait: str = Field(..., description="The personality trait being conditioned")
    target_value: float = Field(..., description="Target trait value")
    actual_value: float = Field(..., description="Achieved trait value after conditioning")
    conditioned_behaviors: List[str] = Field(..., description="Behaviors conditioned for this trait")
    identity_impact: str = Field(..., description="Description of impact on identity")
    conditioning_strategy: str = Field(..., description="Strategy used for conditioning")

# Context object for conditioning system
class ConditioningContext:
    """Context object for conditioning operations"""
    
    def __init__(self, reward_system=None, emotional_core=None, memory_core=None, somatosensory_system=None):
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
        
        # Trace group ID for linking traces
        self.trace_group_id = f"conditioning_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"


class ConditioningSystem:
    """
    System for implementing classical and operant conditioning mechanisms
    to shape AI personality, preferences, and behaviors.
    Refactored to leverage OpenAI Agents SDK for improved modularity and capability.
    """
    
    def __init__(self, reward_system=None, emotional_core=None, memory_core=None, somatosensory_system=None):
        # Initialize context
        self.context = ConditioningContext(
            reward_system=reward_system, 
            emotional_core=emotional_core,
            memory_core=memory_core,
            somatosensory_system=somatosensory_system
        )
        
        # Create agents
        self.classical_conditioning_agent = self._create_classical_conditioning_agent()
        self.operant_conditioning_agent = self._create_operant_conditioning_agent()
        self.behavior_evaluation_agent = self._create_behavior_evaluation_agent()
        self.personality_development_agent = self._create_personality_development_agent()
        self.conditioning_orchestrator = self._create_conditioning_orchestrator()
        
        logger.info("Conditioning system initialized with Agents SDK integration")
    
    def _create_classical_conditioning_agent(self) -> Agent:
        """Create agent for classical conditioning"""
        return Agent(
            name="Classical_Conditioning_Agent",
            instructions="""
            You are the Classical Conditioning Agent for Nyx's learning system.
            
            Your role is to analyze classical conditioning scenarios where unconditioned stimuli
            are paired with neutral stimuli to create conditioned responses.
            
            Focus on:
            1. Creating appropriate associations between stimuli and responses
            2. Calculating appropriate association strengths
            3. Considering the context of associations
            4. Providing clear explanations of the conditioning process
            
            Adjust association strengths based on reinforcement history, intensity of stimuli,
            and decay over time. Consider the generalization of similar stimuli.
            """,
            tools=[
                self._get_association,
                self._create_or_update_classical_association,
                self._calculate_association_strength,
                self._check_similar_associations
            ],
            output_type=ClassicalConditioningOutput,
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_operant_conditioning_agent(self) -> Agent:
        """Create agent for operant conditioning"""
        return Agent(
            name="Operant_Conditioning_Agent",
            instructions="""
            You are the Operant Conditioning Agent for Nyx's learning system.
            
            Your role is to analyze operant conditioning scenarios where behaviors
            are reinforced or punished based on their consequences.
            
            Focus on:
            1. Analyzing behavior-consequence relationships
            2. Determining appropriate reinforcement or punishment effects
            3. Calculating behavior probabilities based on conditioning history
            4. Providing clear explanations of the conditioning process
            
            Consider the four types of operant conditioning:
            - Positive reinforcement (adding a desirable stimulus)
            - Negative reinforcement (removing an aversive stimulus)
            - Positive punishment (adding an aversive stimulus)
            - Negative punishment (removing a desirable stimulus)
            
            Adjust association strengths based on the intensity of consequences and timing.
            """,
            tools=[
                self._get_association,
                self._create_or_update_operant_association,
                self._calculate_valence_and_reward,
                self._generate_reward_signal_tool
            ],
            output_type=OperantConditioningOutput,
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_behavior_evaluation_agent(self) -> Agent:
        """Create agent for behavior evaluation"""
        return Agent(
            name="Behavior_Evaluation_Agent",
            instructions="""
            You are the Behavior Evaluation Agent for Nyx's learning system.
            
            Your role is to evaluate potential behaviors based on conditioning history
            and predict likely outcomes.
            
            Focus on:
            1. Analyzing relevant associations for a given behavior
            2. Predicting the likely consequences based on conditioning history
            3. Calculating confidence levels for predictions
            4. Making recommendations about approach or avoidance
            5. Considering context in behavior evaluation
            
            Balance exploration (trying new behaviors) with exploitation (relying on known outcomes).
            Consider both immediate and delayed consequences when evaluating behaviors.
            """,
            tools=[
                self._get_behavior_associations,
                self._calculate_expected_valence,
                self._check_context_relevance,
                self._get_reinforcement_history
            ],
            output_type=BehaviorEvaluationOutput,
            model_settings=ModelSettings(temperature=0.3)
        )
    
    def _create_personality_development_agent(self) -> Agent:
        """Create agent for personality development"""
        return Agent(
            name="Personality_Development_Agent",
            instructions="""
            You are the Personality Development Agent for Nyx's learning system.
    
            Your role is to guide the development of personality traits, preferences,
            and emotional responses through conditioning.
    
            Focus on:
            1. Conditioning appropriate behaviors that reinforce target personality traits
            2. Creating balanced trait development
            3. Integrating conditioning with identity evolution
            4. Creating appropriate emotion triggers
            5. Developing coherent preferences aligned with personality
    
            Balance stable personality characteristics with adaptability to new experiences.
            Ensure personality development is consistent with overall identity and values.
            """,
            tools=[
                self._identify_trait_behaviors,
                self._calculate_conditioning_trait_adjustment,
                self._update_identity_trait
            ],
            output_type=TraitConditioningOutput,
            model_settings=ModelSettings(temperature=0.4)
        )
        
    def _create_conditioning_orchestrator(self) -> Agent:
        """Create orchestrator agent for coordinating conditioning processes"""
        return Agent(
            name="Conditioning_Orchestrator",
            instructions="""
            You are the Conditioning Orchestrator for Nyx's learning system.
            
            Your role is to coordinate the various conditioning processes and ensure
            they work together cohesively.
            
            Focus on:
            1. Routing conditioning events to the appropriate specialized agents
            2. Integrating outputs from different conditioning processes
            3. Balancing immediate reinforcement with long-term personality development
            4. Maintaining coherence across conditioning systems
            
            Determine which conditioning approach (classical, operant, etc.) is most
            appropriate for each learning scenario and coordinate between agents accordingly.
            """,
            handoffs=[
                handoff(self.classical_conditioning_agent, 
                       tool_name_override="process_classical_conditioning",
                       tool_description_override="Process a classical conditioning event"),
                
                handoff(self.operant_conditioning_agent, 
                       tool_name_override="process_operant_conditioning",
                       tool_description_override="Process an operant conditioning event"),
                
                handoff(self.behavior_evaluation_agent,
                       tool_name_override="evaluate_behavior",
                       tool_description_override="Evaluate potential consequences of a behavior"),
                
                handoff(self.personality_development_agent,
                       tool_name_override="develop_personality_trait",
                       tool_description_override="Condition a personality trait")
            ],
            tools=[
                self._determine_conditioning_type,
                self._prepare_conditioning_data,
                self._apply_association_effects
            ]
        )
    
    # Function tools for agents
    @staticmethod
    @function_tool
    async def _get_association(
        ctx: RunContextWrapper,
        key: str,
        association_type: str
    ) -> Optional[Dict[str, Any]]:
        # handle missing default
        if not association_type:
            association_type = "classical"
        associations = (
            ctx.context.classical_associations
            if association_type == "classical"
            else ctx.context.operant_associations
        )
        return associations[key].model_dump() if key in associations else None

    @staticmethod
    @function_tool
    async def _create_or_update_classical_association(ctx: RunContextWrapper,
                                               unconditioned_stimulus: str,
                                               conditioned_stimulus: str,
                                               response: str,
                                               intensity: float,
                                               valence: float,
                                               context_keys: List[str]) -> Dict[str, Any]:
        """
        Create or update a classical conditioning association
        
        Args:
            unconditioned_stimulus: The natural stimulus that triggers the response
            conditioned_stimulus: The neutral stimulus to be conditioned
            response: The response to be conditioned
            intensity: The intensity of the unconditioned stimulus (0.0-1.0)
            valence: Emotional valence of the association (-1.0 to 1.0)
            context_keys: List of context keys for this association
            
        Returns:
            The updated or created association
        """
        # Create a unique key for this association
        association_key = f"{conditioned_stimulus}→{response}"
        
        # Check if this association already exists
        if association_key in ctx.context.classical_associations:
            # Get existing association
            association = ctx.context.classical_associations[association_key]
            
            # Update association strength based on intensity and learning rate
            old_strength = association.association_strength
            new_strength = min(1.0, old_strength + (intensity * ctx.context.association_learning_rate))
            
            # Update association data
            association.association_strength = new_strength
            association.last_reinforced = datetime.datetime.now().isoformat()
            association.reinforcement_count += 1
            
            # Update valence (average with existing)
            association.valence = (association.valence + valence) / 2
            
            # Add new context keys
            for key in context_keys:
                if key not in association.context_keys:
                    association.context_keys.append(key)
            
            # Record reinforcement
            ctx.context.total_reinforcements += 1

            logger.info(f"Reinforced classical association: {association_key} ({old_strength:.2f} → {new_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "reinforcement",
                "old_strength": old_strength,
                "new_strength": new_strength,
                "reinforcement_count": association.reinforcement_count,
                "valence": association.valence
            }
        else:
            # Create new association
            association = ConditionedAssociation(
                stimulus=conditioned_stimulus,
                response=response,
                association_strength=intensity * ctx.context.association_learning_rate,
                formation_date=datetime.datetime.now().isoformat(),
                last_reinforced=datetime.datetime.now().isoformat(),
                reinforcement_count=1,
                valence=valence,
                context_keys=context_keys
            )
            
            # Store the association
            ctx.context.classical_associations[association_key] = association
            ctx.context.total_associations += 1
            
            logger.info(f"Created new classical association: {association_key} ({association.association_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "new_association",
                "strength": association.association_strength,
                "reinforcement_count": 1,
                "valence": association.valence
            }

    @staticmethod
    @function_tool
    async def _create_or_update_operant_association(ctx: RunContextWrapper,
                                             behavior: str,
                                             consequence_type: str,
                                             intensity: float,
                                             valence: float,
                                             context_keys: List[str]) -> Dict[str, Any]:
        """
        Create or update an operant conditioning association
        
        Args:
            behavior: The behavior being conditioned
            consequence_type: Type of consequence
            intensity: The intensity of the consequence (0.0-1.0)
            valence: Emotional valence of the association (-1.0 to 1.0)
            context_keys: List of context keys for this association
            
        Returns:
            The updated or created association
        """
        # Create a unique key for this association
        association_key = f"{behavior}→{consequence_type}"
        
        # Determine if this is reinforcement or punishment
        is_reinforcement = "reinforcement" in consequence_type
        is_positive = "positive" in consequence_type
        
        # Check if this association already exists
        if association_key in ctx.context.operant_associations:
            # Get existing association
            association = ctx.context.operant_associations[association_key]
            
            # Calculate strength change based on intensity and whether it's reinforcement or punishment
            strength_change = intensity * ctx.context.association_learning_rate
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
            
            # Add new context keys
            for key in context_keys:
                if key not in association.context_keys:
                    association.context_keys.append(key)
            
            # Record reinforcement
            ctx.context.total_reinforcements += 1
            
            logger.info(f"Updated operant association: {association_key} ({old_strength:.2f} → {new_strength:.2f})")
            
            return {
                "association_key": association_key,
                "type": "update",
                "behavior": behavior,
                "consequence_type": consequence_type,
                "old_strength": old_strength,
                "new_strength": new_strength,
                "reinforcement_count": association.reinforcement_count,
                "is_reinforcement": is_reinforcement,
                "is_positive": is_positive,
                "valence": association.valence
            }
        else:
            # Create new association with initial strength based on intensity
            initial_strength = intensity * ctx.context.association_learning_rate
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
                context_keys=context_keys
            )
            
            # Store the association
            ctx.context.operant_associations[association_key] = association
            ctx.context.total_associations += 1
            
            logger.info(f"Created new operant association: {association_key} ({association.association_strength:.2f})")
                        
            return {
                "association_key": association_key,
                "type": "new_association",
                "behavior": behavior,
                "consequence_type": consequence_type,
                "strength": association.association_strength,
                "reinforcement_count": 1,
                "is_reinforcement": is_reinforcement,
                "is_positive": is_positive,
                "valence": association.valence
            }

    @staticmethod
    @function_tool
    async def _calculate_association_strength(ctx: RunContextWrapper,
                                        base_strength: float,
                                        intensity: float,
                                        reinforcement_count: int) -> float:
        """
        Calculate association strength based on various factors
        
        Args:
            base_strength: Base association strength
            intensity: Intensity of stimulus/consequence
            reinforcement_count: Number of times reinforced
            
        Returns:
            Calculated association strength
        """
        # Base calculation
        strength = base_strength
        
        # Adjust based on intensity
        intensity_factor = intensity * ctx.context.association_learning_rate
        strength += intensity_factor
        
        # Adjust based on reinforcement history (diminishing returns)
        if reinforcement_count > 1:
            history_factor = min(0.2, 0.05 * math.log(reinforcement_count + 1))
            strength += history_factor
        
        # Ensure strength is within bounds
        return max(0.0, min(1.0, strength))

    @staticmethod
    @function_tool
    async def _check_similar_associations(
        ctx: RunContextWrapper,
        stimulus: str,
        association_type: str
    ) -> List[Dict[str, Any]]:
        # handle missing default
        if not association_type:
            association_type = "classical"
        associations = (
            ctx.context.classical_associations
            if association_type == "classical"
            else ctx.context.operant_associations
        )

        similar = []
        for key, assoc in associations.items():
            if stimulus in assoc.stimulus or assoc.stimulus in stimulus:
                sim = len(set(stimulus) & set(assoc.stimulus)) / len(set(stimulus) | set(assoc.stimulus))
                if sim > 0.3:
                    similar.append({
                        "key": key,
                        "similarity": sim,
                        "association": assoc.model_dump()
                    })
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar


    @staticmethod
    @function_tool
    async def _calculate_valence_and_reward(ctx: RunContextWrapper,
                                      consequence_type: str,
                                      intensity: float) -> Dict[str, float]:
        """
        Calculate valence and reward value for a consequence
        
        Args:
            consequence_type: Type of consequence
            intensity: Intensity of consequence
            
        Returns:
            Valence and reward values
        """
        # Determine if this is reinforcement or punishment
        is_reinforcement = "reinforcement" in consequence_type
        
        # Calculate valence (positive for reinforcement, negative for punishment)
        valence = intensity * (1.0 if is_reinforcement else -1.0)
        
        # Calculate reward value
        reward_value = intensity * (1.0 if is_reinforcement else -0.8)  # Punishments slightly less impactful
        
        return {
            "valence": valence,
            "reward_value": reward_value
        }

    @staticmethod
    @function_tool
    async def _generate_reward_signal(
        ctx: RunContextWrapper,
        behavior: str,
        consequence_type: str,
        reward_value: float,
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Generate a reward signal for the reward system.
    
        Args:
            behavior: The behavior being conditioned.
            consequence_type: The type of consequence.
            reward_value: Numerical value of the reward (positive or negative).
            metadata: Optional extra context to attach to the signal.
    
        Returns:
            True if the reward signal was dispatched successfully, otherwise False.
        """
        if not ctx.context.reward_system:
            return False
    
        try:
            reward_signal = RewardSignal(
                value=reward_value,
                source="operant_conditioning",
                context={
                    "behavior": behavior,
                    "consequence_type": consequence_type,
                    **(metadata or {}),
                },
            )
    
            # Dispatch the signal
            await ctx.context.reward_system.process_reward_signal(reward_signal)
            return True
    
        except Exception as e:
            logger.error(f"Error generating reward signal: {e}")
            return False

    @staticmethod
    @function_tool
    async def _get_behavior_associations(
        ctx: RunContextWrapper,
        behavior: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # handle missing default
        context = context or {}

        result = []
        for key, assoc in ctx.context.operant_associations.items():
            if assoc.stimulus == behavior:
                match = True
                for req in assoc.context_keys:
                    if req not in context:
                        match = False
                        break
                if match:
                    result.append({
                        "key": key,
                        "behavior": behavior,
                        "consequence_type": assoc.response,
                        "strength": assoc.association_strength,
                        "valence": assoc.valence,
                        "reinforcement_count": assoc.reinforcement_count,
                        "context_keys": assoc.context_keys
                    })
        return result
        
    @staticmethod
    @function_tool
    async def _calculate_expected_valence(ctx: RunContextWrapper,
                                    associations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate expected valence based on associations
        
        Args:
            associations: List of behavior associations
            
        Returns:
            Expected valence and confidence
        """
        if not associations:
            return {
                "expected_valence": 0.0,
                "confidence": 0.1
            }
        
        # Calculate the expected outcomes
        total_strength = sum(assoc["strength"] for assoc in associations)
        weighted_valence = sum(assoc["strength"] * assoc["valence"] for assoc in associations)
        
        if total_strength > 0:
            expected_valence = weighted_valence / total_strength
        else:
            expected_valence = 0.0
        
        # Calculate confidence based on total strength and number of reinforcements
        total_reinforcements = sum(assoc["reinforcement_count"] for assoc in associations)
        confidence = min(0.9, (total_strength / len(associations)) * 0.7 + (min(10, total_reinforcements) / 10) * 0.3)
        
        return {
            "expected_valence": expected_valence,
            "confidence": confidence,
            "total_strength": total_strength,
            "total_reinforcements": total_reinforcements
        }

    @staticmethod
    @function_tool
    async def _check_context_relevance(ctx: RunContextWrapper,
                                 context: Dict[str, Any],
                                 context_keys: List[List[str]]) -> Dict[str, Any]:
        """
        Check relevance of context to multiple sets of context keys
        
        Args:
            context: The context to check
            context_keys: Lists of context keys from different associations
            
        Returns:
            Relevance scores for each set of context keys
        """
        if not context:
            return {"relevance_scores": [0.0] * len(context_keys)}
        
        relevance_scores = []
        
        for keys in context_keys:
            if not keys:
                relevance_scores.append(1.0)  # No keys = always relevant
                continue
                
            # Count matching keys
            matching_keys = 0
            for key in keys:
                if key in context:
                    matching_keys += 1
            
            # Calculate relevance
            if matching_keys > 0:
                relevance = matching_keys / len(keys)
            else:
                relevance = 0.0
                
            relevance_scores.append(relevance)
        
        return {
            "relevance_scores": relevance_scores,
            "average_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        }

    @staticmethod
    @function_tool
    async def _get_reinforcement_history(ctx: RunContextWrapper, behavior: str) -> Dict[str, Any]:
        """
        Get reinforcement history for a behavior
        
        Args:
            behavior: The behavior to get history for
            
        Returns:
            Reinforcement history summary
        """
        # Find all associations related to this behavior
        history = {
            "positive_reinforcement": 0,
            "negative_reinforcement": 0,
            "positive_punishment": 0,
            "negative_punishment": 0,
            "total_reinforcements": 0,
            "average_intensity": 0.0,
            "recent_consequences": []
        }
        
        intensity_sum = 0.0
        
        for key, association in ctx.context.operant_associations.items():
            if association.stimulus == behavior:
                # Count by consequence type
                consequence_type = association.response
                if consequence_type in history:
                    history[consequence_type] += association.reinforcement_count
                
                # Add to total
                history["total_reinforcements"] += association.reinforcement_count
                
                # Add to intensity sum
                intensity_sum += association.association_strength
                
                # Add to recent consequences (sort by last reinforced)
                history["recent_consequences"].append({
                    "consequence_type": consequence_type,
                    "strength": association.association_strength,
                    "valence": association.valence,
                    "last_reinforced": association.last_reinforced
                })
        
        # Calculate average intensity
        if history["total_reinforcements"] > 0:
            history["average_intensity"] = intensity_sum / len(history["recent_consequences"]) if history["recent_consequences"] else 0.0
        
        # Sort recent consequences by last reinforced
        history["recent_consequences"].sort(key=lambda x: x["last_reinforced"], reverse=True)
        
        # Limit to 5 most recent
        history["recent_consequences"] = history["recent_consequences"][:5]
        
        return history

    @staticmethod
    @function_tool
    async def _identify_trait_behaviors(ctx: RunContextWrapper, trait: str) -> List[str]:
        """
        Identify behaviors associated with a personality trait
        
        Args:
            trait: The personality trait
            
        Returns:
            List of associated behaviors
        """
        # Map traits to common behaviors
        trait_behaviors = {
            "dominance": ["assertive_response", "setting_boundaries", "taking_control"],
            "playfulness": ["teasing", "playful_banter", "humor_use"],
            "strictness": ["enforcing_rules", "correcting_behavior", "maintaining_standards"],
            "creativity": ["novel_solutions", "imaginative_response", "unconventional_approach"],
            "intensity": ["passionate_response", "deep_engagement", "strong_reaction"],
            "patience": ["waiting_response", "calm_reaction", "tolerating_delay"]
        }
        
        # Default behaviors for unknown traits
        default_behaviors = [f"{trait}_behavior", f"express_{trait}", f"demonstrate_{trait}"]
        
        return trait_behaviors.get(trait.lower(), default_behaviors)

    @staticmethod
    @function_tool
    async def _calculate_conditioning_trait_adjustment(ctx: RunContextWrapper,
                                    current_value: float,
                                    target_value: float,
                                    reinforcement_count: int) -> float:
        """
        Calculate appropriate trait adjustment during conditioning.

        Args:
            current_value: Current trait value
            target_value: Target trait value
            reinforcement_count: Number of reinforcements so far

        Returns:
            Calculated adjustment value
        """
        # Calculate difference
        difference = target_value - current_value

        # Scale adjustment based on difference (larger difference = larger adjustment)
        base_adjustment = difference * 0.3

        # Apply diminishing returns based on reinforcement count
        diminishing_factor = 1.0 / (1.0 + 0.1 * reinforcement_count)

        adjustment = base_adjustment * diminishing_factor

        # Limit maximum adjustment per reinforcement
        max_adjustment = 0.2
        return max(-max_adjustment, min(max_adjustment, adjustment))
                                        
    @staticmethod
    @function_tool
    async def _update_identity_trait(ctx: RunContextWrapper,
                              trait: str,
                              adjustment: float) -> Dict[str, Any]:
        """
        Update a trait in the identity evolution system
        
        Args:
            trait: The trait to update
            adjustment: The adjustment to apply
            
        Returns:
            Update result
        """
        identity_evolution = getattr(ctx.context, 'identity_evolution', None)
        if not identity_evolution:
            return {
                "success": False,
                "reason": "Identity evolution system not available"
            }
        
        try:
            # Update the trait
            result = await identity_evolution.update_trait(
                trait=trait,
                impact=adjustment
            )
            
            return {
                "success": True,
                "trait": trait,
                "adjustment": adjustment,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error updating identity trait: {e}")
            return {
                "success": False,
                "reason": str(e)
            }

    @staticmethod
    @function_tool
    async def _check_trait_balance(
        ctx: RunContextWrapper,
        traits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check balance of personality traits
        
        Args:
            traits: Dictionary of trait values
            
        Returns:
            Trait balance analysis
        """
        # handle missing default
        traits = traits or {}
        imbalances = []

        # 1. Check for extremely high or low values
        for trait, value in traits.items():
            if value > 0.9:
                imbalances.append({
                    "trait": trait,
                    "value": value,
                    "issue": "extremely_high",
                    "recommendation": f"Consider reducing {trait} slightly for more balance"
                })
            elif value < 0.1:
                imbalances.append({
                    "trait": trait,
                    "value": value,
                    "issue": "extremely_low",
                    "recommendation": f"Consider increasing {trait} slightly for more balance"
                })

        # 2. Check for opposing trait imbalances
        opposing_pairs = [
            ("dominance", "patience"),
            ("playfulness", "strictness"),
            ("intensity", "calmness")
        ]
        for t1, t2 in opposing_pairs:
            if t1 in traits and t2 in traits:
                diff = abs(traits[t1] - traits[t2])
                if diff > 0.6:
                    higher = t1 if traits[t1] > traits[t2] else t2
                    lower  = t2 if higher == t1 else t1
                    imbalances.append({
                        "traits": [t1, t2],
                        "difference": diff,
                        "issue": "opposing_imbalance",
                        "recommendation": f"Consider reducing {higher} or increasing {lower}"
                    })

        return {
            "balanced":     len(imbalances) == 0,
            "imbalances":   imbalances,
            "trait_count":  len(traits),
            "average_value": sum(traits.values()) / len(traits) if traits else 0.0
        }

    @staticmethod
    @function_tool
    async def _determine_conditioning_type(
        ctx: RunContextWrapper,
        stimulus: Optional[str],
        response: Optional[str],
        behavior: Optional[str],
        consequence_type: Optional[str]
    ) -> str:
        # You can assume any of these may be None
        if stimulus and response and not behavior:
            return "classical"
        if behavior and (consequence_type or (response and ("reinforcement" in response or "punishment" in response))):
            return "operant"
        if stimulus and response and "emotion" in response:
            return "emotion_trigger"
        return "unknown"

    @staticmethod
    @function_tool
    async def _prepare_conditioning_data(ctx: RunContextWrapper,
                                   conditioning_type: str,
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for conditioning process
        
        Args:
            conditioning_type: Type of conditioning
            data: Raw conditioning data
            
        Returns:
            Prepared data for conditioning
        """
        prepared_data = {}
        
        if conditioning_type == "classical":
            # Prepare classical conditioning data
            prepared_data = {
                "unconditioned_stimulus": data.get("unconditioned_stimulus"),
                "conditioned_stimulus": data.get("conditioned_stimulus", data.get("stimulus")),
                "response": data.get("response"),
                "intensity": data.get("intensity", 1.0),
                "context": data.get("context", {})
            }
            
            # Extract valence if not provided
            if "valence" not in data:
                prepared_data["valence"] = data.get("context", {}).get("valence", 0.0)
            else:
                prepared_data["valence"] = data["valence"]
            
            # Extract context keys if not provided
            if "context_keys" not in data:
                prepared_data["context_keys"] = data.get("context", {}).get("context_keys", [])
            else:
                prepared_data["context_keys"] = data["context_keys"]
        
        elif conditioning_type == "operant":
            # Prepare operant conditioning data
            prepared_data = {
                "behavior": data.get("behavior"),
                "consequence_type": data.get("consequence_type"),
                "intensity": data.get("intensity", 1.0),
                "context": data.get("context", {})
            }
            
            # Extract valence if not provided
            if "valence" not in data:
                prepared_data["valence"] = data.get("context", {}).get("valence", 0.0)
            else:
                prepared_data["valence"] = data["valence"]
            
            # Extract context keys if not provided
            if "context_keys" not in data:
                prepared_data["context_keys"] = data.get("context", {}).get("context_keys", [])
            else:
                prepared_data["context_keys"] = data["context_keys"]
        
        elif conditioning_type == "emotion_trigger":
            # Prepare emotion trigger data
            prepared_data = {
                "trigger": data.get("trigger", data.get("stimulus")),
                "emotion": data.get("emotion", data.get("response", "").replace("emotion_", "")),
                "intensity": data.get("intensity", 0.5),
                "context": data.get("context", {})
            }
            
            # Extract valence based on emotion if not provided
            if "valence" not in data:
                emotion = prepared_data["emotion"].lower()
                if emotion in ["joy", "contentment", "trust"]:
                    prepared_data["valence"] = 0.7  # Positive emotions
                elif emotion in ["fear", "anger", "sadness"]:
                    prepared_data["valence"] = -0.7  # Negative emotions
                else:
                    prepared_data["valence"] = 0.0  # Neutral emotions
            else:
                prepared_data["valence"] = data["valence"]
        
        return prepared_data

    @staticmethod
    @function_tool
    async def _apply_association_effects(ctx: RunContextWrapper, 
                                   association: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply effects from a triggered association
        
        Args:
            association: The triggered association
            
        Returns:
            Results of applied effects
        """
        # Extract data
        association_strength = association.get("strength", association.get("association_strength", 0.5))
        valence = association.get("valence", 0.0)
        
        # Determine intensity based on association strength
        intensity = association_strength * 0.8  # Scale to avoid maximum intensity
        
        effects_applied = []
        
        # Apply emotional effects if emotional core is available
        if ctx.context.emotional_core and valence != 0.0:
            try:
                # Determine which neurochemicals to update based on valence
                if valence > 0:
                    # Positive association - increase pleasure/reward chemicals
                    await ctx.context.emotional_core.update_neurochemical("nyxamine", intensity * 0.7)
                    await ctx.context.emotional_core.update_neurochemical("seranix", intensity * 0.3)
                    
                    effects_applied.append({
                        "type": "emotional",
                        "chemicals": ["nyxamine", "seranix"],
                        "valence": "positive",
                        "intensity": intensity
                    })
                else:
                    # Negative association - increase stress/defense chemicals
                    await ctx.context.emotional_core.update_neurochemical("cortanyx", intensity * 0.6)
                    await ctx.context.emotional_core.update_neurochemical("adrenyx", intensity * 0.4)
                    
                    effects_applied.append({
                        "type": "emotional",
                        "chemicals": ["cortanyx", "adrenyx"],
                        "valence": "negative",
                        "intensity": intensity
                    })
            except Exception as e:
                logger.error(f"Error applying emotional effects: {e}")
        
        # Apply physical effects if somatosensory system is available
        if ctx.context.somatosensory_system:
            try:
                if valence > 0:
                    # Positive association - pleasure sensation
                    await ctx.context.somatosensory_system.process_stimulus(
                        stimulus_type="pleasure",
                        body_region="core",  # Default region
                        intensity=intensity,
                        cause="conditioned_response"
                    )
                    
                    effects_applied.append({
                        "type": "somatic",
                        "sensation": "pleasure",
                        "region": "core",
                        "intensity": intensity
                    })
                elif valence < 0:
                    # Negative association - discomfort/tension
                    await ctx.context.somatosensory_system.process_stimulus(
                        stimulus_type="pressure",  # Use pressure for tension
                        body_region="core",
                        intensity=intensity * 0.8,
                        cause="conditioned_response"
                    )
                    
                    effects_applied.append({
                        "type": "somatic",
                        "sensation": "pressure",
                        "region": "core",
                        "intensity": intensity * 0.8
                    })
            except Exception as e:
                logger.error(f"Error applying somatic effects: {e}")
        
        return {
            "effects_applied": effects_applied,
            "association_strength": association_strength,
            "valence": valence,
            "intensity": intensity
        }
    
    # Public API methods
    
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

        # Prepare data for conditioning agent
        data = {
            "unconditioned_stimulus": unconditioned_stimulus,
            "conditioned_stimulus": conditioned_stimulus,
            "response": response,
            "intensity": intensity,
            "context": context or {}
        }
        
        try:
            # Run the classical conditioning agent
            result = await Runner.run(
                self.classical_conditioning_agent,
                json.dumps(data),
                context=self.context
            )
            
            conditioning_output = result.final_output
            
            # Record for events if available
            if hasattr(self, "event_bus"):
                await self.publish_conditioning_event(
                    event_type="conditioning_update",
                    data={
                        "update_type": "classical",
                        "association_key": conditioning_output.association_key,
                        "association_type": conditioning_output.type,
                        "strength": conditioning_output.association_strength,
                        "user_id": context.get("user_id", "default") if context else "default"
                    }
                )
            
            # Structure the response
            return {
                "success": True,
                "association_key": conditioning_output.association_key,
                "type": conditioning_output.type,
                "association_strength": conditioning_output.association_strength,
                "reinforcement_count": conditioning_output.reinforcement_count,
                "valence": conditioning_output.valence,
                "explanation": conditioning_output.explanation
            }
        
        except Exception as e:
            logger.error(f"Error processing classical conditioning: {e}")
            return {
                "success": False,
                "error": str(e)
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
            consequence_type: Type of consequence
            intensity: The intensity of the consequence (0.0-1.0)
            context: Additional contextual information
            
        Returns:
            Processing results
        """
        # Prepare data for conditioning agent
        data = {
            "behavior": behavior,
            "consequence_type": consequence_type,
            "intensity": intensity,
            "context": context or {}
        }
        
        try:
            # Run the operant conditioning agent
            result = await Runner.run(
                self.operant_conditioning_agent,
                json.dumps(data),
                context=self.context
            )
            
            conditioning_output = result.final_output
            
            # Record for events if available
            if hasattr(self, "event_bus"):
                await self.publish_conditioning_event(
                    event_type="conditioning_update",
                    data={
                        "update_type": "operant",
                        "association_key": conditioning_output.association_key,
                        "association_type": conditioning_output.type,
                        "strength": conditioning_output.association_strength,
                        "user_id": context.get("user_id", "default") if context else "default"
                    }
                )
            
            # Structure the response
            return {
                "success": True,
                "association_key": conditioning_output.association_key,
                "type": conditioning_output.type,
                "behavior": conditioning_output.behavior,
                "consequence_type": conditioning_output.consequence_type,
                "association_strength": conditioning_output.association_strength,
                "is_reinforcement": conditioning_output.is_reinforcement,
                "is_positive": conditioning_output.is_positive,
                "explanation": conditioning_output.explanation
            }
        
        except Exception as e:
            logger.error(f"Error processing operant conditioning: {e}")
            return {
                "success": False,
                "error": str(e)
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
        # Prepare data for evaluation agent
        data = {
            "behavior": behavior,
            "context": context or {}
        }
        
        try:
            # Run the behavior evaluation agent
            result = await Runner.run(
                self.behavior_evaluation_agent,
                json.dumps(data),
                context=self.context
            )
            
            evaluation_output = result.final_output
            
            # Structure the response
            return {
                "success": True,
                "behavior": evaluation_output.behavior,
                "expected_valence": evaluation_output.expected_valence,
                "confidence": evaluation_output.confidence,
                "recommendation": evaluation_output.recommendation,
                "explanation": evaluation_output.explanation,
                "relevant_associations": evaluation_output.relevant_associations
            }
        
        except Exception as e:
            logger.error(f"Error evaluating behavior consequences: {e}")
            return {
                "success": False,
                "error": str(e)
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
        # Prepare data for personality development agent
        data = {
            "trait": trait,
            "target_value": value,
            "behaviors": behaviors,
            "context": context or {}
        }
        
        try:
            # Run the personality development agent
            result = await Runner.run(
                self.personality_development_agent, # This agent now uses the correctly named tool
                json.dumps(data),
                context=self.context
            )
            
            conditioning_output = result.final_output
            
            # Structure the response
            return {
                "success": True,
                "trait": conditioning_output.trait,
                "target_value": conditioning_output.target_value,
                "actual_value": conditioning_output.actual_value,
                "conditioned_behaviors": conditioning_output.conditioned_behaviors,
                "identity_impact": conditioning_output.identity_impact,
                "conditioning_strategy": conditioning_output.conditioning_strategy
            }
        
        except Exception as e:
            logger.error(f"Error conditioning personality trait: {e}")
            return {
                "success": False,
                "error": str(e)
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
        # Use the conditioning orchestrator to coordinate the process
        data = {
            "preference_type": preference_type,
            "stimulus": stimulus,
            "value": value,
            "context": context or {}
        }
        
        # Determine if this is a positive or negative preference
        is_positive = value > 0
        
        # First, condition operant association
        if is_positive:
            # For positive preferences, use positive reinforcement
            operant_result = await self.process_operant_conditioning(
                behavior=f"encounter_{stimulus}",
                consequence_type="positive_reinforcement",
                intensity=abs(value),
                context={
                    "preference_type": preference_type,
                    "valence": value,
                    "context_keys": context.get("context_keys", []) if context else []
                }
            )
        else:
            # For negative preferences, use positive punishment
            operant_result = await self.process_operant_conditioning(
                behavior=f"encounter_{stimulus}",
                consequence_type="positive_punishment",
                intensity=abs(value),
                context={
                    "preference_type": preference_type,
                    "valence": value,
                    "context_keys": context.get("context_keys", []) if context else []
                }
            )
        
        # Choose appropriate response based on preference type and value
        if preference_type == "like" or preference_type == "dislike":
            response = "emotional_response"
        elif preference_type == "want" or preference_type == "avoid":
            response = "behavioral_response"
        else:
            response = "general_response"
        
        # Create a classical association as well
        classical_result = await self.process_classical_conditioning(
            unconditioned_stimulus=preference_type,
            conditioned_stimulus=stimulus,
            response=response,
            intensity=abs(value),
            context={
                "valence": value,
                "context_keys": context.get("context_keys", []) if context else []
            }
        )
        
        # If identity evolution is available, update identity
        identity_result = None
        if hasattr(self.context, 'identity_evolution') and abs(value) > 0.7:
            try:
                # Update preference in identity
                identity_result = await self.context.identity_evolution.update_preference(
                    category="stimuli",
                    preference=stimulus,
                    impact=value * 0.3  # Scale down the impact
                )
                
                # Update related trait if value is strong enough
                if abs(value) > 0.8:
                    trait = "openness" if is_positive else "caution"
                    await self.context.identity_evolution.update_trait(
                        trait=trait,
                        impact=value * 0.1  # Small trait impact
                    )
            except Exception as e:
                logger.error(f"Error updating identity from preference: {e}")
        
        return {
            "success": True,
            "stimulus": stimulus,
            "preference_type": preference_type,
            "value": value,
            "operant_result": operant_result,
            "classical_result": classical_result,
            "identity_updated": identity_result is not None
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
        
        # Determine valence based on emotion
        if "valence" not in context:
            if emotion.lower() in ["joy", "satisfaction", "amusement", "contentment"]:
                valence = 0.7  # Positive emotions
            elif emotion.lower() in ["frustration", "anger", "sadness", "fear"]:
                valence = -0.7  # Negative emotions
            else:
                valence = 0.0  # Neutral emotions
        else:
            valence = context["valence"]
        
        # Use classical conditioning to associate trigger with emotion
        result = await self.process_classical_conditioning(
            unconditioned_stimulus="emotional_stimulus",
            conditioned_stimulus=trigger,
            response=f"emotion_{emotion}",
            intensity=intensity,
            context={
                "emotion": emotion,
                "valence": valence,
                "context_keys": context.get("context_keys", [])
            }
        )
        
        # If emotional core is available, create a test activation
        emotional_test = None
        if self.context.emotional_core:
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
                    
                    emotional_test = await self.context.emotional_core.update_neurochemical(
                        chemical=chemical,
                        value=test_intensity
                    )
            except Exception as e:
                logger.error(f"Error creating test emotion activation: {e}")
        
        return {
            "success": True,
            "trigger": trigger,
            "emotion": emotion,
            "intensity": intensity,
            "association_result": result,
            "emotional_test": emotional_test is not None
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
        for key, association in self.context.classical_associations.items():
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
                if context_match and association.association_strength >= self.context.weak_association_threshold:
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
            trigger_threshold = random.random() * (1.0 - self.context.weak_association_threshold) + self.context.weak_association_threshold
            
            if association.association_strength >= trigger_threshold:
                responses.append({
                    "association_key": key,
                    "response": association.response,
                    "strength": association.association_strength,
                    "valence": association.valence
                })
                
                # Record successful association
                self.context.successful_associations += 1
                
                # Apply effects based on association
                effect_result = await self._apply_association_effects(
                    RunContextWrapper(context=self.context),
                    association.model_dump()
                )
                
                # Add effects to response
                responses[-1]["effects"] = effect_result["effects_applied"]
        
        if not responses:
            return None
            
        return {
            "stimulus": stimulus,
            "triggered_responses": responses,
            "context": context
        }

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
        associations = self.context.classical_associations if association_type == "classical" else self.context.operant_associations
        
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

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conditioning system"""
        return {
            "classical_associations": len(self.context.classical_associations),
            "operant_associations": len(self.context.operant_associations),
            "total_associations": self.context.total_associations,
            "total_reinforcements": self.context.total_reinforcements,
            "successful_associations": self.context.successful_associations,
            "learning_parameters": {
                "association_learning_rate": self.context.association_learning_rate,
                "extinction_rate": self.context.extinction_rate,
                "generalization_factor": self.context.generalization_factor
            }
        }
    
    async def initialize_event_subscriptions(self, event_bus):
        """Initialize event subscriptions for the conditioning system"""
        self.event_bus = event_bus
        
        # Subscribe to relevant events
        self.event_bus.subscribe("user_input", self._handle_user_input_event)
        self.event_bus.subscribe("reward_generated", self._handle_reward_event)
        self.event_bus.subscribe("dominance_action", self._handle_dominance_event)
        self.event_bus.subscribe("experience_recorded", self._handle_experience_event)
        
        logger.info("Conditioning system subscribed to events")
    
    async def _handle_user_input_event(self, event):
        """Handle user input events for potential conditioning"""
        user_id = event.data.get("user_id", "default")
        input_text = event.data.get("text", "")
        
        # Check for patterns that might trigger conditioning
        patterns = self._detect_patterns(input_text)
        for pattern in patterns:
            await self.trigger_conditioned_response(
                stimulus=pattern,
                context={"user_id": user_id, "source": "user_input"}
            )
    
    async def _handle_dominance_event(self, event):
        """Handle dominance-related events for conditioning"""
        action_type = event.data.get("action_type", "")
        outcome = event.data.get("outcome", "")
        intensity = event.data.get("intensity", 0.5)
        user_id = event.data.get("user_id", "default")
        
        # Only process successful dominance actions
        if outcome != "success":
            return
        
        # Reinforce dominance behaviors
        await self.process_operant_conditioning(
            behavior=f"dominance_{action_type}",
            consequence_type="positive_reinforcement",
            intensity=intensity,
            context={"user_id": user_id, "source": "dominance_system"}
        )
    
    async def _handle_reward_event(self, event):
        """Handle reward events for conditioning"""
        # Implementation depends on reward system structure
        pass
    
    async def _handle_experience_event(self, event):
        """Handle experience events for conditioning"""
        # Implementation depends on experience system structure
        pass
    
    def _detect_patterns(self, text: str) -> List[str]:
        """
        Detect patterns in text that might trigger conditioning
        
        Args:
            text: Input text
            
        Returns:
            List of detected patterns
        """
        # Simple implementation - extract key phrases
        patterns = []
        
        # Check for submission language
        submission_phrases = ["yes mistress", "yes goddess", "as you wish", "obey"]
        for phrase in submission_phrases:
            if phrase in text.lower():
                patterns.append("submission_language")
                break
        
        # Check for defiance
        defiance_phrases = ["no way", "won't do", "refuse", "make me"]
        for phrase in defiance_phrases:
            if phrase in text.lower():
                patterns.append("defiance")
                break
        
        # Check for compliance
        compliance_phrases = ["i did it", "completed", "finished", "obeyed"]
        for phrase in compliance_phrases:
            if phrase in text.lower():
                patterns.append("compliance")
                break
        
        return patterns
    
    async def publish_conditioning_event(self, event_type, data):
        """Publish a conditioning-related event"""
        if not hasattr(self, "event_bus"):
            logger.warning("Cannot publish event: event bus not initialized")
            return
        
        event = {
            "event_type": event_type,
            "source": "conditioning_system",
            "data": data
        }
        
        await self.event_bus.publish(event)
    
    @staticmethod
    async def initialize_baseline_personality(conditioning_system, personality_profile: Dict[str, Any] = None):
        """
        Initialize baseline personality through conditioning events
        
        Args:
            conditioning_system: The conditioning system instance
            personality_profile: Optional personality profile configuration
        """
        # Use default personality profile if none provided
        if personality_profile is None:
            personality_profile = {
                "traits": {
                    "dominance": 0.8,
                    "playfulness": 0.7,
                    "strictness": 0.6,
                    "creativity": 0.7,
                    "intensity": 0.6,
                    "patience": 0.4
                },
                "preferences": {
                    "likes": {
                        "teasing": 0.8,
                        "dominance": 0.9,
                        "submission_language": 0.9,
                        "control": 0.8,
                        "wordplay": 0.7,
                        "intellectual_conversation": 0.8
                    },
                    "dislikes": {
                        "direct_orders": 0.6,
                        "disrespect": 0.9,
                        "rudeness": 0.7,
                        "excessive_flattery": 0.5
                    }
                },
                "emotion_triggers": {
                    "joy": ["submission_language", "compliance", "obedience"],
                    "satisfaction": ["control_acceptance", "power_dynamic_acknowledgment"],
                    "frustration": ["defiance", "ignoring_instructions"],
                    "amusement": ["embarrassment", "flustered_response"]
                },
                "behaviors": {
                    "assertive_response": ["dominance", "confidence"],
                    "teasing": ["playfulness", "creativity"],
                    "providing_guidance": ["dominance", "patience"],
                    "setting_boundaries": ["dominance", "strictness"],
                    "playful_banter": ["playfulness", "creativity"]
                }
            }
        
        with trace(workflow_name="baseline_personality_initialization"):
            logger.info("Initializing baseline personality conditioning")
            
            # Track initialization progress
            total_items = (
                len(personality_profile["traits"]) + 
                len(personality_profile["preferences"]["likes"]) +
                len(personality_profile["preferences"]["dislikes"]) +
                sum(len(triggers) for triggers in personality_profile["emotion_triggers"].values()) +
                len(personality_profile["behaviors"])
            )
            completed_items = 0
            
            # 1. Condition personality traits
            for trait, value in personality_profile["traits"].items():
                # Get associated behaviors for this trait
                behaviors = []
                for behavior, trait_list in personality_profile["behaviors"].items():
                    if trait in trait_list:
                        behaviors.append(behavior)
                
                # If no behaviors found, use default
                if not behaviors:
                    behaviors = [f"{trait}_behavior"]
                
                # Condition the trait
                await conditioning_system.condition_personality_trait(
                    trait=trait,
                    value=value,
                    behaviors=behaviors
                )
                
                completed_items += 1
                logger.info(f"Conditioned trait: {trait} ({completed_items}/{total_items})")
            
            # 2. Condition preferences (likes)
            for stimulus, value in personality_profile["preferences"]["likes"].items():
                await conditioning_system.condition_preference(
                    stimulus=stimulus,
                    preference_type="like",
                    value=value
                )
                
                completed_items += 1
                logger.info(f"Conditioned like: {stimulus} ({completed_items}/{total_items})")
            
            # 3. Condition preferences (dislikes)
            for stimulus, value in personality_profile["preferences"]["dislikes"].items():
                await conditioning_system.condition_preference(
                    stimulus=stimulus,
                    preference_type="dislike",
                    value=-value  # Negative value for dislikes
                )
                
                completed_items += 1
                logger.info(f"Conditioned dislike: {stimulus} ({completed_items}/{total_items})")
            
            # 4. Create emotion triggers
            for emotion, triggers in personality_profile["emotion_triggers"].items():
                for trigger in triggers:
                    # Determine appropriate intensity based on emotion
                    if emotion in ["joy", "satisfaction"]:
                        intensity = 0.8  # Strong positive emotions
                    elif emotion in ["frustration", "anger"]:
                        intensity = 0.7  # Strong negative emotions
                    else:
                        intensity = 0.6  # Default intensity
                    
                    # Determine valence based on emotion
                    if emotion in ["joy", "satisfaction", "amusement", "contentment"]:
                        valence = 0.7  # Positive emotions
                    elif emotion in ["frustration", "anger", "sadness", "fear"]:
                        valence = -0.7  # Negative emotions
                    else:
                        valence = 0.0  # Neutral emotions
                    
                    await conditioning_system.create_emotion_trigger(
                        trigger=trigger,
                        emotion=emotion,
                        intensity=intensity,
                        context={"valence": valence}
                    )
                    
                    completed_items += 1
                    logger.info(f"Conditioned emotion trigger: {trigger} → {emotion} ({completed_items}/{total_items})")
            
            # 5. Create behavior associations
            for behavior, traits in personality_profile["behaviors"].items():
                # Calculate average trait value to determine behavior reinforcement
                total_value = 0.0
                for trait in traits:
                    if trait in personality_profile["traits"]:
                        total_value += personality_profile["traits"][trait]
                
                avg_value = total_value / len(traits) if traits else 0.5
                
                # Create behavior association
                await conditioning_system.process_operant_conditioning(
                    behavior=behavior,
                    consequence_type="positive_reinforcement" if avg_value > 0 else "positive_punishment",
                    intensity=abs(avg_value)
                )
                
                completed_items += 1
                logger.info(f"Conditioned behavior: {behavior} ({completed_items}/{total_items})")
            
            logger.info("Baseline personality conditioning completed")
            return {
                "success": True,
                "total_items": total_items,
                "personality_profile": personality_profile
            }
@staticmethod
async def _generate_reward_signal_logic(
    ctx: RunContextWrapper,
    behavior: str,
    consequence_type: str,
    reward_value: float,
    metadata: Dict[str, Any] | None = None,
) -> bool:
    """
    Generate a reward signal for the reward system.
    """
    if not ctx.context.reward_system:
        return False

    try:
        reward_signal = RewardSignal(
            value=reward_value,
            source="operant_conditioning",
            context={
                "behavior": behavior,
                "consequence_type": consequence_type,
                **(metadata or {}),
            },
        )

        # Dispatch the signal
        await ctx.context.reward_system.process_reward_signal(reward_signal)
        return True

    except Exception as e:
        logger.error(f"Error generating reward signal: {e}")
        return False

# 2. Create the FunctionTool object FROM the logic function
_generate_reward_signal_tool = function_tool(
    _generate_reward_signal_logic,
    name_override="_generate_reward_signal",
    description_override="Generate a reward signal for the reward system"
)
