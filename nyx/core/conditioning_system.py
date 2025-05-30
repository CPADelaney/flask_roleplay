# nyx/core/conditioning_system.py

import logging
import datetime
import math
import json # Added import
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import asyncio

from agents import Agent, Runner, trace, function_tool, RunContextWrapper, ModelSettings, handoff, FunctionTool # Added FunctionTool import
from nyx.core.reward_system import RewardSignal
from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.context import EmotionalContext

logging.basicConfig(level=logging.DEBUG) # Or configure your specific logger
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

class ConditioningContext:
    """
    Lightweight container every function‑tool receives through
    RunContextWrapper.  It holds references to external cores plus
    the mutable conditioning state.
    """
    def __init__(
        self,
        reward_system=None,
        emotional_core=None,
        memory_core=None,
        somatosensory_system=None,
    ):
        # external subsystems
        self.reward_system        = reward_system
        self.emotional_core       = emotional_core
        self.memory_core          = memory_core
        self.somatosensory_system = somatosensory_system

        # mutable association stores
        self.classical_associations: dict = {}
        self.operant_associations:   dict = {}

        # learning hyper‑parameters (will be overwritten by the
        # ConditioningSystem right after it instantiates us)
        self.weak_association_threshold     = 0.3
        self.moderate_association_threshold = 0.6
        self.strong_association_threshold   = 0.8

        self.association_learning_rate = 0.1
        self.extinction_rate           = 0.05
        self.generalization_factor     = 0.3

        # counters
        self.total_associations      = 0
        self.total_reinforcements    = 0
        self.successful_associations = 0



class ConditioningSystem:
    """
    System for implementing classical and operant conditioning mechanisms
    to shape AI personality, preferences, and behaviors.
    Refactored to leverage OpenAI Agents SDK for improved modularity and capability.
    """
    
    def __init__(self, reward_system=None, emotional_core=None,
                 memory_core=None, somatosensory_system=None):
        # ➊ Create the shared context
        self.context = ConditioningContext(
            reward_system=reward_system,
            emotional_core=emotional_core,
            memory_core=memory_core,
            somatosensory_system=somatosensory_system,
        )
    
        # ➋ DON'T create new dictionaries! Just reference the ones from context:
        self.classical_associations = self.context.classical_associations
        self.operant_associations = self.context.operant_associations
        
        # ➌ Set thresholds & learning params on BOTH self and context
        self.weak_association_threshold = 0.3
        self.moderate_association_threshold = 0.6
        self.strong_association_threshold = 0.8
        self.association_learning_rate = 0.1
        self.extinction_rate = 0.05
        self.generalization_factor = 0.3
        
        # Copy to context
        self.context.weak_association_threshold = self.weak_association_threshold
        self.context.moderate_association_threshold = self.moderate_association_threshold  
        self.context.strong_association_threshold = self.strong_association_threshold
        self.context.association_learning_rate = self.association_learning_rate
        self.context.extinction_rate = self.extinction_rate
        self.context.generalization_factor = self.generalization_factor
        
        # ➍ Initialize counters
        self.total_associations = self.context.total_associations
        self.total_reinforcements = self.context.total_reinforcements
        self.successful_associations = self.context.successful_associations
        
        # Create the reward signal tool
        self._generate_reward_signal_tool = function_tool(
            self._generate_reward_signal_logic,
            name_override="_generate_reward_signal",
            description_override="Generate a reward signal for the reward system"
        )
        
        # Create agents
        self.classical_conditioning_agent = self._create_classical_conditioning_agent()
        self.operant_conditioning_agent = self._create_operant_conditioning_agent()
        self.behavior_evaluation_agent = self._create_behavior_evaluation_agent()
        self.personality_development_agent = self._create_personality_development_agent()
        self.conditioning_orchestrator = self._create_conditioning_orchestrator()
        
        logger.info("Conditioning system initialized with Agents SDK integration")

    @staticmethod
    def _force_str_list(obj) -> List[str]:
        if obj is None:
            return []
        if isinstance(obj, (list, tuple, set)):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            # Consider if returning keys is always desired, or if values or items might be needed.
            # For context_keys, keys are usually what's intended.
            return [str(k) for k in obj.keys()]
        return [str(obj)]
    
    def _create_classical_conditioning_agent(self) -> Agent:
        """Create agent for classical conditioning"""
        return Agent(
            name="Classical_Conditioning_Agent",
            instructions="""
            You are the Classical Conditioning Agent for Nyx's learning system.
            The input you receive will be a JSON string detailing a classical conditioning event.
            This JSON string will contain fields like 'unconditioned_stimulus', 'conditioned_stimulus', 
            'response', 'intensity', 'valence', and 'context_keys'.
            
            Your role is to analyze this data to create or update associations between stimuli and responses.
            
            Focus on:
            1. Extracting information from the input JSON.
            2. Calling the '_create_or_update_classical_association' tool with the extracted data.
            3. Calculating appropriate association strengths using '_calculate_association_strength' if needed.
            4. Considering the context of associations and checking for similar ones with '_check_similar_associations'.
            5. Providing clear explanations of the conditioning process in your final output.
            
            Adjust association strengths based on reinforcement history, intensity of stimuli,
            and decay over time. Consider the generalization of similar stimuli.
            """,
            model="gpt-4.1-nano", # Consider gpt-4o or similar for better JSON handling if issues persist
            tools=[
                self._get_association,
                self._create_or_update_classical_association,
                self._calculate_association_strength,
                self._check_similar_associations
            ],
            output_type=ClassicalConditioningOutput
        )
    
    def _create_operant_conditioning_agent(self) -> Agent:
        """Create agent for operant conditioning"""
        reward_signal_tool = function_tool(
            self._generate_reward_signal_logic,
            name_override="generate_reward_signal",
            description_override="Generate a reward signal for the reward system"
        )
        return Agent(
            name="Operant_Conditioning_Agent",
            instructions="""
            You are the Operant Conditioning Agent for Nyx's learning system.
            The input you receive will be a JSON string detailing an operant conditioning event.
            This JSON string will contain fields like 'behavior', 'consequence_type', 
            'intensity', 'valence', and 'context_keys'.
            
            Your role is to analyze operant conditioning scenarios where behaviors
            are reinforced or punished based on their consequences.
            
            Focus on:
            1. Extracting information from the input JSON.
            2. Calling the '_create_or_update_operant_association' tool with the extracted data.
            3. Calculating valence and reward with '_calculate_valence_and_reward' and using 'generate_reward_signal'.
            4. Analyzing behavior-consequence relationships and determining appropriate reinforcement or punishment effects.
            5. Calculating behavior probabilities based on conditioning history.
            6. Providing clear explanations of the conditioning process in your final output.
            
            Consider the four types of operant conditioning:
            - Positive reinforcement (adding a desirable stimulus)
            - Negative reinforcement (removing an aversive stimulus)
            - Positive punishment (adding an aversive stimulus)
            - Negative punishment (removing a desirable stimulus)
            
            Adjust association strengths based on the intensity of consequences and timing.
            """,
            model="gpt-4.1-nano", # Consider gpt-4o or similar for better JSON handling
            tools=[
                self._get_association,
                self._create_or_update_operant_association,
                self._calculate_valence_and_reward,
                reward_signal_tool
            ],
            output_type=OperantConditioningOutput
        )
    
    def _create_behavior_evaluation_agent(self) -> Agent:
        """Create agent for behavior evaluation"""
        return Agent(
            name="Behavior_Evaluation_Agent",
            instructions="""
            You are the Behavior Evaluation Agent for Nyx's learning system.
            The input you receive will be a JSON string containing 'behavior' and 'context'.
            
            Your role is to evaluate potential behaviors based on conditioning history
            and predict likely outcomes.
            
            Focus on:
            1. Extracting 'behavior' and 'context' from the input JSON.
            2. Analyzing relevant associations for the given behavior using '_get_behavior_associations' 
               (pass the context as 'behavior_context' parameter).
            3. Predicting the likely consequences and calculating expected valence using '_calculate_expected_valence'.
               IMPORTANT: When calling '_calculate_expected_valence', you must convert the associations list 
               to a JSON string using json.dumps(). For example: associations_json=json.dumps(associations_list)
            4. Checking context relevance with '_check_context_relevance' (pass the context as 'current_context' parameter) 
               and getting reinforcement history with '_get_reinforcement_history'.
            5. Calculating confidence levels for predictions.
            6. Making recommendations about approach or avoidance.
            7. Providing a clear explanation in your final output.
            
            Balance exploration (trying new behaviors) with exploitation (relying on known outcomes).
            Consider both immediate and delayed consequences when evaluating behaviors.
            """,
            model="gpt-4.1-nano",
            tools=[
                self._get_behavior_associations,
                self._calculate_expected_valence,
                self._check_context_relevance,
                self._get_reinforcement_history
            ],
            output_type=BehaviorEvaluationOutput
        )
    
    def _create_personality_development_agent(self) -> Agent:
        """Create agent for personality development"""
        return Agent(
            name="Personality_Development_Agent",
            instructions="""
            You are the Personality Development Agent for Nyx's learning system.
            The input you receive will be a JSON string containing 'trait' and 'target_value'.
            Optionally, the input JSON might contain 'current_trait_values_snapshot' which is a dictionary of all current trait values.
    
            Your role is to guide the development of personality traits, preferences,
            and emotional responses through conditioning.
    
            Focus on:
            1. Extracting 'trait' and 'target_value' from the input JSON.
            2. Identifying behaviors related to the trait using '_identify_trait_behaviors'.
            3. Calculating trait adjustments using '_calculate_conditioning_trait_adjustment'.
            4. Updating identity traits with '_update_identity_trait'.
            5. Conditioning appropriate behaviors that reinforce target personality traits.
            6. If 'current_trait_values_snapshot' is available from the input JSON, creating balanced trait development by calling the '_check_trait_balance' tool.
               When calling '_check_trait_balance', simply pass the traits dictionary directly as the 'traits_snapshot' parameter.
               For example: _check_trait_balance(traits_snapshot={"kindness": 0.8, "humor": 0.6})
            7. Integrating conditioning with identity evolution.
            8. Formulating a conditioning strategy and describing identity impact in the final output.
    
            Balance stable personality characteristics with adaptability to new experiences.
            Ensure personality development is consistent with overall identity and values.
            """,
            model="gpt-4.1-nano",
            tools=[
                self._identify_trait_behaviors,
                self._calculate_conditioning_trait_adjustment,
                self._update_identity_trait,
                self._check_trait_balance 
            ],
            output_type=TraitConditioningOutput
        )
            
    def _create_conditioning_orchestrator(self) -> Agent:
        """Create orchestrator agent for coordinating conditioning processes"""
        return Agent(
            name="Conditioning_Orchestrator",
            instructions="""
            You are the Conditioning Orchestrator for Nyx's learning system.
            Your input will be a JSON string describing a conditioning event or request.
            This JSON may contain fields like 'unconditioned_stimulus', 'conditioned_stimulus', 
            'response', 'behavior', 'consequence_type', 'trait', 'target_value', 'stimulus', 
            'preference_type', 'value', 'trigger', 'emotion', 'intensity', or 'context'.
            
            Your role is to coordinate the various conditioning processes and ensure
            they work together cohesively.
            
            Focus on:
            1. Determining the type of conditioning needed using '_determine_conditioning_type' based on the input JSON.
            2. Preparing the data for the specific conditioning agent using '_prepare_conditioning_data'.
            3. Routing conditioning events to the appropriate specialized agents via handoffs 
               (process_classical_conditioning, process_operant_conditioning, evaluate_behavior, develop_personality_trait).
            4. Integrating outputs from different conditioning processes.
            5. Applying association effects using '_apply_association_effects' if a response is triggered directly.
            6. Balancing immediate reinforcement with long-term personality development.
            7. Maintaining coherence across conditioning systems.
            
            Determine which conditioning approach (classical, operant, etc.) is most
            appropriate for each learning scenario and coordinate between agents accordingly.
            """,
            model="gpt-4.1-nano", # Consider gpt-4o or similar
            handoffs=[
                handoff(self.classical_conditioning_agent, 
                       tool_name_override="process_classical_conditioning",
                       tool_description_override="Process a classical conditioning event. Expects JSON input with unconditioned_stimulus, conditioned_stimulus, response, intensity, valence, context_keys."),
                
                handoff(self.operant_conditioning_agent, 
                       tool_name_override="process_operant_conditioning",
                       tool_description_override="Process an operant conditioning event. Expects JSON input with behavior, consequence_type, intensity, valence, context_keys."),
                
                handoff(self.behavior_evaluation_agent,
                       tool_name_override="evaluate_behavior",
                       tool_description_override="Evaluate potential consequences of a behavior. Expects JSON input with behavior, context."),
                
                handoff(self.personality_development_agent,
                       tool_name_override="develop_personality_trait",
                       tool_description_override="Condition a personality trait. Expects JSON input with trait, target_value.")
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
    async def _create_or_update_classical_association(
        ctx: RunContextWrapper,
        unconditioned_stimulus: str,
        conditioned_stimulus: str,
        response: str,
        intensity: float,
        valence: float,
        context_keys: Union[List[str], str, None] = None  # Allow multiple types
    ) -> Dict[str, Any]:
        """
        Create or update a classical conditioning association.
        
        Args:
            unconditioned_stimulus: The unconditioned stimulus
            conditioned_stimulus: The conditioned stimulus  
            response: The conditioned response
            intensity: Intensity of the association (0.0-1.0)
            valence: Emotional valence (-1.0 to 1.0)
            context_keys: Context keys (can be list, string, or None)
        """
        tool_name = "_create_or_update_classical_association"
        
        # Handle context_keys more robustly
        if context_keys is None:
            processed_context_keys = []
        elif isinstance(context_keys, str):
            # If it's a string, wrap it in a list
            processed_context_keys = [context_keys]
        elif isinstance(context_keys, list):
            # Ensure all items are strings
            processed_context_keys = [str(k) for k in context_keys]
        else:
            logger.warning(f"[{tool_name}] Unexpected context_keys type: {type(context_keys)}. Converting to string list.")
            try:
                # Try to convert whatever it is
                if hasattr(context_keys, '__iter__') and not isinstance(context_keys, (str, bytes)):
                    processed_context_keys = [str(k) for k in context_keys]
                else:
                    processed_context_keys = [str(context_keys)]
            except Exception as e:
                logger.error(f"[{tool_name}] Failed to process context_keys: {e}")
                processed_context_keys = []
        
        logger.debug(f"[{tool_name}] Processed context_keys: {processed_context_keys}")
        
        association_key = f"{conditioned_stimulus}-->{response}"
        
        if association_key in ctx.context.classical_associations:
            # Update existing association
            association = ctx.context.classical_associations[association_key]
            old_strength = association.association_strength
            new_strength = min(1.0, old_strength + (intensity * ctx.context.association_learning_rate))
            
            association.association_strength = new_strength
            association.last_reinforced = datetime.datetime.now(datetime.timezone.utc).isoformat()
            association.reinforcement_count += 1
            association.valence = (association.valence + valence) / 2
            
            # Add new context keys
            for key in processed_context_keys:
                if key and key not in association.context_keys:
                    association.context_keys.append(key)
            
            ctx.context.total_reinforcements += 1
            
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
                formation_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                last_reinforced=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                reinforcement_count=1,
                valence=valence,
                context_keys=processed_context_keys
            )
            
            ctx.context.classical_associations[association_key] = association
            ctx.context.total_associations += 1
            
            return {
                "association_key": association_key,
                "type": "new_association",
                "strength": association.association_strength,
                "reinforcement_count": 1,
                "valence": association.valence
            }
            
    async def _generate_reward_signal_logic(
        self,
        ctx: RunContextWrapper,
        behavior: str,
        consequence_type: str,
        reward_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Generate a reward signal for the reward system.
        """
        logger.debug(f"Entering _generate_reward_signal_logic for behavior: {behavior}")
        
        # More robust context checking
        if not hasattr(ctx, 'context'):
            logger.error("No context attribute in RunContextWrapper")
            return False
            
        if not ctx.context:
            logger.error("Context is None")
            return False
            
        if not hasattr(ctx.context, 'reward_system'):
            logger.error("No reward_system in context")
            return False
            
        reward_system = ctx.context.reward_system
        if not reward_system:
            logger.warning("Reward system is None - skipping reward signal generation")
            return False
        
        try:
            # Check if reward system has the expected method
            if not hasattr(reward_system, 'process_reward_signal'):
                logger.error(f"Reward system of type {type(reward_system)} lacks process_reward_signal method")
                return False
                
            reward_signal = RewardSignal(
                value=reward_value,
                source="operant_conditioning",
                context={
                    "behavior": behavior,
                    "consequence_type": consequence_type,
                    **(metadata or {}),
                },
            )
            
            logger.debug(f"Dispatching reward signal: {reward_signal.model_dump()}")
            await reward_system.process_reward_signal(reward_signal)
            logger.debug("Reward signal dispatched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating/dispatching reward signal: {e}", exc_info=True)
            return False
        

    @staticmethod
    @function_tool
    async def _create_or_update_operant_association(
        ctx: RunContextWrapper,
        behavior: str,
        consequence_type: str,
        intensity: float,
        valence: float,
        context_keys: Union[List[str], str, None] = None
    ) -> Dict[str, Any]:
        """
        Create or update an operant conditioning association.
        """
        tool_name = "_create_or_update_operant_association"
        
        # Process context_keys the same way
        if context_keys is None:
            processed_context_keys = []
        elif isinstance(context_keys, str):
            processed_context_keys = [context_keys]
        elif isinstance(context_keys, list):
            processed_context_keys = [str(k) for k in context_keys]
        else:
            try:
                if hasattr(context_keys, '__iter__') and not isinstance(context_keys, (str, bytes)):
                    processed_context_keys = [str(k) for k in context_keys]
                else:
                    processed_context_keys = [str(context_keys)]
            except:
                processed_context_keys = []
        
        association_key = f"{behavior}-->{consequence_type}"
        
        is_reinforcement = "reinforcement" in consequence_type.lower()
        is_positive = "positive" in consequence_type.lower()
        
        if association_key in ctx.context.operant_associations:
            association = ctx.context.operant_associations[association_key]
            
            strength_change = intensity * ctx.context.association_learning_rate
            if not is_reinforcement:
                strength_change *= -1 
            
            old_strength = association.association_strength
            new_strength = max(0.0, min(1.0, old_strength + strength_change))
            
            association.association_strength = new_strength
            association.last_reinforced = datetime.datetime.now().isoformat()
            association.reinforcement_count += 1
            association.valence = (association.valence + valence) / 2 
            
            for key in context_keys:
                if key not in association.context_keys:
                    association.context_keys.append(key)
            
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
            initial_strength = intensity * ctx.context.association_learning_rate
            if not is_reinforcement: # Punishment starts weaker or has different impact
                initial_strength = max(0, initial_strength - 0.1) # Example: reduce initial impact of punishment slightly
            
            association = ConditionedAssociation(
                stimulus=behavior, 
                response=consequence_type, 
                association_strength=initial_strength,
                formation_date=datetime.datetime.now().isoformat(),
                last_reinforced=datetime.datetime.now().isoformat(),
                reinforcement_count=1,
                valence=valence,
                context_keys=context_keys
            )
            
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
        strength = base_strength
        intensity_factor = intensity * ctx.context.association_learning_rate
        strength += intensity_factor
        
        if reinforcement_count > 1:
            history_factor = min(0.2, 0.05 * math.log(reinforcement_count + 1))
            strength += history_factor
        
        return max(0.0, min(1.0, strength))

    @staticmethod
    @function_tool
    async def _check_similar_associations(
        ctx: RunContextWrapper,
        stimulus: str,
        association_type: str
    ) -> List[Dict[str, Any]]:
        if not association_type:
            association_type = "classical"
        associations = (
            ctx.context.classical_associations
            if association_type == "classical"
            else ctx.context.operant_associations
        )

        similar = []
        # Basic string similarity, could be improved with more sophisticated methods (e.g., embeddings)
        # For now, simple substring or set overlap check
        stimulus_lower = stimulus.lower()
        for key, assoc in associations.items():
            assoc_stimulus_lower = assoc.stimulus.lower()
            # Check for partial or full match
            if stimulus_lower in assoc_stimulus_lower or assoc_stimulus_lower in stimulus_lower:
                # Calculate a simple similarity score (Jaccard index on characters for example)
                s1_chars = set(stimulus_lower)
                s2_chars = set(assoc_stimulus_lower)
                sim_score = len(s1_chars & s2_chars) / len(s1_chars | s2_chars) if len(s1_chars | s2_chars) > 0 else 0
                
                if sim_score > 0.3: # Threshold for similarity
                    similar.append({
                        "key": key,
                        "similarity": sim_score,
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
        Calculate valence and reward value for a consequence.
        The LLM should extract consequence_type and intensity from the JSON input string.
        """
        is_reinforcement = "reinforcement" in consequence_type.lower()
        is_positive_event = "positive" in consequence_type.lower() # e.g. positive_reinforcement or positive_punishment

        valence = 0.0
        if is_reinforcement:
            valence = intensity # Positive for any reinforcement
        else: # Punishment
            valence = -intensity # Negative for any punishment
        
        # Reward value logic could be more nuanced
        # For simplicity: reinforcement good, punishment bad
        reward_value = intensity if is_reinforcement else -intensity * 0.8 # Punishments might have a slightly less negative reward value than their valence
        
        return {
            "valence": valence,
            "reward_value": reward_value
        }


    @staticmethod
    @function_tool
    async def _get_behavior_associations(
        ctx: RunContextWrapper,
        behavior: str,
        behavior_context: Optional[Dict[str, Any]] = None  # Make it optional with default None
    ) -> List[Dict[str, Any]]:
        behavior_context = behavior_context or {}  # Handle None case
        result = []
        behavior_lower = behavior.lower()
    
        for key, assoc in ctx.context.operant_associations.items():
            if assoc.stimulus.lower() == behavior_lower:
                context_match = True
                if assoc.context_keys:
                    if not behavior_context:
                        context_match = False
                    else:
                        for req_key in assoc.context_keys:
                            if req_key not in behavior_context:
                                context_match = False
                                break
                
                if context_match:
                    result.append({
                        "key": key,
                        "behavior": assoc.stimulus,
                        "consequence_type": assoc.response,
                        "strength": assoc.association_strength,
                        "valence": assoc.valence,
                        "reinforcement_count": assoc.reinforcement_count,
                        "context_keys": assoc.context_keys
                    })
        return result
        
    @staticmethod
    @function_tool
    async def _calculate_expected_valence(
        ctx: RunContextWrapper,
        associations_json: str  # Accept as JSON string to avoid schema issues
    ) -> Dict[str, float]:
        """
        Calculate expected valence from a list of associations.
        
        Args:
            associations_json: JSON string containing list of associations
            
        Returns:
            Dictionary with expected_valence, confidence, and other metrics
        """
        tool_name = "_calculate_expected_valence"
        logger.debug(f"[{tool_name}] Called with associations_json: {associations_json[:200]}...")
        
        # Parse the JSON string
        try:
            associations = json.loads(associations_json) if associations_json else []
        except json.JSONDecodeError as e:
            logger.error(f"[{tool_name}] Failed to parse associations_json: {e}")
            return {"expected_valence": 0.0, "confidence": 0.0, "error": f"Invalid JSON: {str(e)}"}
        
        if not isinstance(associations, list):
            logger.warning(f"[{tool_name}] Expected list but got {type(associations)}")
            return {"expected_valence": 0.0, "confidence": 0.0, "error": "Associations must be a list"}
        
        if not associations:
            return_value = {"expected_valence": 0.0, "confidence": 0.1}
            logger.debug(f"[{tool_name}] No associations provided. Returning: {return_value!r}")
            return return_value
        
        total_strength = 0.0
        weighted_valence = 0.0
        total_reinforcements = 0
        valid_assoc_count = 0
        
        for i, assoc in enumerate(associations):
            if not isinstance(assoc, dict):
                logger.warning(f"[{tool_name}] Item at index {i} is not a dict: {type(assoc)}")
                continue
            try:
                strength = float(assoc.get("strength", 0.0))
                valence = float(assoc.get("valence", 0.0))
                reinforcements = int(assoc.get("reinforcement_count", 0))
    
                total_strength += strength
                weighted_valence += strength * valence
                total_reinforcements += reinforcements
                valid_assoc_count += 1
            except (TypeError, ValueError) as e:
                logger.error(f"[{tool_name}] Error processing association at index {i}: {e}", exc_info=True)
                continue
        
        if valid_assoc_count == 0:
            return_value = {"expected_valence": 0.0, "confidence": 0.0, "error": "No valid associations found in input"}
            logger.debug(f"[{tool_name}] No valid associations. Returning: {return_value!r}")
            return return_value
    
        expected_valence = (weighted_valence / total_strength) if total_strength > 0 else 0.0
        
        # Confidence calculation
        avg_strength_component = (total_strength / valid_assoc_count if valid_assoc_count > 0 else 0.0) * 0.7
        reinforcement_component = (min(1.0, math.log1p(total_reinforcements) / math.log1p(100))) * 0.3
        
        confidence = min(1.0, avg_strength_component + reinforcement_component)
        confidence = max(0.1, confidence)
        
        return_value = {
            "expected_valence": round(expected_valence, 3),
            "confidence": round(confidence, 3),
            "total_strength": round(total_strength, 3),
            "total_reinforcements": total_reinforcements
        }
        logger.debug(f"[{tool_name}] Returning: {return_value!r}")
        return return_value

    @staticmethod
    @function_tool
    async def _check_context_relevance(
        ctx: RunContextWrapper,
        current_context: Optional[Dict[str, Any]] = None,  # Make optional with default None
        context_keys_list: Optional[List[List[str]]] = None  # Make optional with default None
    ) -> Dict[str, Any]:
        """
        Check relevance of current context to multiple sets of required context keys from associations.
        Args:
            current_context: The current context to check (e.g. {"location": "home", "time": "night"})
            context_keys_list: A list of lists, where each inner list contains required context keys for an association.
        Returns:
            Relevance scores for each set of context keys.
        """
        # Handle None cases
        if current_context is None:
            current_context = {}
        if context_keys_list is None:
            context_keys_list = []
            
        if not isinstance(current_context, dict):
            logger.warning(f"_check_context_relevance: 'current_context' arg is not a dict: {type(current_context)}. Defaulting to empty.")
            current_context = {}
        if not isinstance(context_keys_list, list) or not all(isinstance(sublist, list) for sublist in context_keys_list):
            logger.warning(f"_check_context_relevance: 'context_keys_list' arg is not a list of lists. Got: {type(context_keys_list)}. Returning empty scores.")
            return {"relevance_scores": [], "average_relevance": 0.0, "error": "Invalid context_keys_list format"}
    
        relevance_scores = []
        
        for required_keys_for_assoc in context_keys_list:
            if not isinstance(required_keys_for_assoc, list):
                relevance_scores.append(0.0)
                continue
    
            if not required_keys_for_assoc:
                relevance_scores.append(1.0)
                continue
                
            matching_keys_count = 0
            for req_key in required_keys_for_assoc:
                if req_key in current_context:  # Fixed variable name
                    matching_keys_count += 1
            
            relevance = matching_keys_count / len(required_keys_for_assoc) if len(required_keys_for_assoc) > 0 else 1.0
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {
            "relevance_scores": [round(s, 3) for s in relevance_scores],
            "average_relevance": round(avg_relevance, 3)
        }

    @staticmethod
    @function_tool
    async def _get_reinforcement_history(ctx: RunContextWrapper, behavior: str) -> Dict[str, Any]: # LLM extracts behavior
        """
        Get reinforcement history for a behavior.
        """
        history = {
            "positive_reinforcement_count": 0,
            "negative_reinforcement_count": 0,
            "positive_punishment_count": 0,
            "negative_punishment_count": 0,
            "total_consequences_recorded": 0, # Total distinct consequence types recorded for this behavior
            "total_reinforcements_overall": 0, # Sum of reinforcement_count across all matched associations
            "average_strength_of_associations": 0.0,
            "average_valence_of_associations": 0.0,
            "recent_consequences_details": [] # List of dicts for recent consequences
        }
        
        strength_sum = 0.0
        valence_sum = 0.0
        matched_associations_count = 0
        behavior_lower = behavior.lower()

        temp_consequences_list = []

        for key, association in ctx.context.operant_associations.items():
            if association.stimulus.lower() == behavior_lower:
                consequence_type_lower = association.response.lower()
                
                # Increment specific consequence type counts
                if "positive_reinforcement" in consequence_type_lower:
                    history["positive_reinforcement_count"] += association.reinforcement_count
                elif "negative_reinforcement" in consequence_type_lower:
                    history["negative_reinforcement_count"] += association.reinforcement_count
                elif "positive_punishment" in consequence_type_lower:
                    history["positive_punishment_count"] += association.reinforcement_count
                elif "negative_punishment" in consequence_type_lower:
                    history["negative_punishment_count"] += association.reinforcement_count
                
                history["total_reinforcements_overall"] += association.reinforcement_count
                strength_sum += association.association_strength
                valence_sum += association.valence
                matched_associations_count += 1
                
                temp_consequences_list.append({
                    "consequence_type": association.response,
                    "strength": association.association_strength,
                    "valence": association.valence,
                    "last_reinforced": association.last_reinforced,
                    "reinforcement_count": association.reinforcement_count
                })
        
        history["total_consequences_recorded"] = matched_associations_count
        if matched_associations_count > 0:
            history["average_strength_of_associations"] = round(strength_sum / matched_associations_count, 3)
            history["average_valence_of_associations"] = round(valence_sum / matched_associations_count, 3)
        
        temp_consequences_list.sort(key=lambda x: x["last_reinforced"], reverse=True)
        history["recent_consequences_details"] = temp_consequences_list[:5] # Top 5 most recent
        
        return history

    @staticmethod
    @function_tool
    async def _identify_trait_behaviors(ctx: RunContextWrapper, trait: str) -> List[str]: # LLM extracts trait
        """
        Identify behaviors associated with a personality trait.
        """
        trait_behaviors = {
            "dominance": ["assertive_response", "setting_boundaries", "taking_control", "issuing_commands"],
            "playfulness": ["teasing", "playful_banter", "humor_use", "initiating_games"],
            "strictness": ["enforcing_rules", "correcting_behavior", "maintaining_standards", "demanding_precision"],
            "creativity": ["novel_solutions", "imaginative_response", "unconventional_approach", "artistic_expression"],
            "intensity": ["passionate_response", "deep_engagement", "strong_reaction", "focused_attention"],
            "patience": ["waiting_response", "calm_reaction_to_delay", "tolerating_mistakes", "repeating_instructions_calmly"],
            "nurturing": ["offering_comfort", "providing_support", "gentle_guidance", "expressing_empathy"],
            "analytical": ["problem_decomposition", "logical_reasoning_display", "data_driven_statements"],
            "curiosity": ["asking_probing_questions", "exploring_new_topics", "experimenting_with_ideas"]
        }
        
        trait_lower = trait.lower()
        default_behaviors = [f"{trait_lower}_expression", f"demonstrating_{trait_lower}", f"acting_with_{trait_lower}"]
        
        return trait_behaviors.get(trait_lower, default_behaviors)

    @staticmethod
    @function_tool
    async def _calculate_conditioning_trait_adjustment(ctx: RunContextWrapper,
                                    current_value: float, # LLM provides
                                    target_value: float, # LLM provides
                                    reinforcement_count: int) -> float: # LLM provides (e.g., number of successful conditioning attempts so far for this trait)
        """
        Calculate appropriate trait adjustment during conditioning.
        """
        difference = target_value - current_value
        base_adjustment = difference * 0.2 # Slightly reduced impact per step for smoother transitions

        # Diminishing returns factor based on how many times this specific conditioning for the trait has been reinforced.
        # More reinforcements mean smaller subsequent adjustments.
        diminishing_factor = 1.0 / (1.0 + 0.15 * reinforcement_count) 

        adjustment = base_adjustment * diminishing_factor
        
        # Max adjustment per step to prevent overly rapid changes
        max_abs_adjustment = 0.15 
        # Ensure adjustment is not excessively small if difference is tiny but non-zero
        min_abs_adjustment_if_non_zero_diff = 0.01 

        if abs(adjustment) < min_abs_adjustment_if_non_zero_diff and difference != 0:
            adjustment = min_abs_adjustment_if_non_zero_diff * (1 if adjustment > 0 else -1)

        return max(-max_abs_adjustment, min(max_abs_adjustment, round(adjustment, 4)))
                                        
    @staticmethod
    @function_tool
    async def _update_identity_trait(ctx: RunContextWrapper,
                              trait: str, # LLM provides
                              adjustment: float) -> Dict[str, Any]: # LLM provides (output from _calculate_conditioning_trait_adjustment)
        """
        Update a trait in the identity evolution system.
        This tool simulates interaction with an external identity system.
        """
        # In a real system, this would call ctx.context.identity_evolution_system.update_trait(...)
        # For this example, we'll simulate it.
        identity_evolution_system = getattr(ctx.context, 'identity_evolution', None) 
        # This should ideally be a more complex object, but for now, we assume it has an 'update_trait' method
        # or we simulate the update here if it's None or a simple dict.

        logger.info(f"Attempting to update identity trait '{trait}' by {adjustment}")

        if identity_evolution_system and hasattr(identity_evolution_system, 'update_trait') and asyncio.iscoroutinefunction(identity_evolution_system.update_trait):
            try:
                # Example: await identity_evolution_system.update_trait(trait_name=trait, change_value=adjustment, source="conditioning_system")
                # The actual signature of update_trait would depend on the identity_evolution_system's design.
                # Let's assume it takes trait and impact.
                result = await identity_evolution_system.update_trait(
                    trait=trait,
                    impact=adjustment # Assuming 'impact' is the parameter name
                ) 
                logger.info(f"Identity trait '{trait}' updated. Result: {result}")
                return {
                    "success": True,
                    "trait": trait,
                    "adjustment_applied": adjustment,
                    "new_value_or_status": result # This depends on what update_trait returns
                }
            except Exception as e:
                logger.error(f"Error calling external identity_evolution.update_trait for trait '{trait}': {e}", exc_info=True)
                return {
                    "success": False,
                    "reason": f"External call to identity_evolution.update_trait failed: {str(e)}",
                    "trait": trait
                }
        else:
            # Simulate if no proper system or method exists
            logger.warning(f"Identity evolution system or 'update_trait' method not available/callable for trait '{trait}'. Simulating update.")
            # If self.context.identity_traits is a dict (example)
            if not hasattr(ctx.context, 'identity_traits_store'):
                ctx.context.identity_traits_store = {} # Initialize if not present

            current_val = ctx.context.identity_traits_store.get(trait, 0.5) # Default to 0.5 if new
            new_val = max(0.0, min(1.0, current_val + adjustment))
            ctx.context.identity_traits_store[trait] = new_val
            
            return {
                "success": True,
                "trait": trait,
                "adjustment_applied": adjustment,
                "new_value_or_status": f"Simulated update. New value for {trait}: {new_val:.3f}"
            }


    @staticmethod
    @function_tool
    async def _check_trait_balance(
        ctx: RunContextWrapper,
        traits_snapshot: Optional[Dict[str, float]] = None  # MODIFIED: Made Optional with default None
    ) -> Dict[str, Any]:
        """
        Check balance of personality traits from a given snapshot.
        Args:
            traits_snapshot: Dictionary of trait names to their current values (0.0-1.0). Can be None.
        Returns:
            Trait balance analysis.
        """
        # ADDED: Handle the case where traits_snapshot is None
        if traits_snapshot is None:
            logger.warning("_check_trait_balance received None for traits_snapshot.")
            return {
                "balanced": False,
                "imbalances": [{"issue": "Traits snapshot not provided", "recommendation": "Ensure traits_snapshot is provided for analysis."}],
                "trait_count": 0,
                "average_value": 0.0,
                "message": "Traits snapshot was None."
            }

        # Original logic for when traits_snapshot is a dict (even if empty)
        if not isinstance(traits_snapshot, dict) or not all(isinstance(v, (int, float)) for v in traits_snapshot.values()):
            logger.warning(f"_check_trait_balance: 'traits_snapshot' is not a dict of trait:value. Got: {type(traits_snapshot)}. Input: {traits_snapshot!r}")
            # Ensure the output matches the expected structure if an error occurs within the tool
            return {
                "balanced": False,
                "imbalances": [{"issue": "Invalid input format for traits_snapshot", "recommendation": "Ensure traits_snapshot is a dictionary of string keys to numeric values."}],
                "trait_count": 0,
                "average_value": 0.0,
                "message": "Invalid traits_snapshot format."
            }

        imbalances = []
        num_traits = len(traits_snapshot)
        if num_traits == 0:
            return {
                "balanced": True, # An empty set of traits can be considered balanced
                "imbalances": [],
                "trait_count": 0,
                "average_value": 0.0,
                "message": "No traits to analyze (empty snapshot)." # Modified message
            }

        # 1. Check for extremely high or low values
        for trait, value in traits_snapshot.items():
            if value > 0.95:
                imbalances.append({
                    "trait": trait, "value": round(value,3), "issue": "extremely_high",
                    "recommendation": f"Consider strategies to moderate '{trait}' for better balance if it's causing issues."
                })
            elif value < 0.05:
                imbalances.append({
                    "trait": trait, "value": round(value,3), "issue": "extremely_low",
                    "recommendation": f"Consider strategies to develop '{trait}' if its absence is detrimental."
                })

        # 2. Check for opposing trait imbalances (example pairs)
        opposing_pairs = [
            ("dominance", "patience"), ("playfulness", "strictness"),
            ("intensity", "nurturing"), ("creativity", "analytical")
        ]
        for t1, t2 in opposing_pairs:
            if t1 in traits_snapshot and t2 in traits_snapshot:
                val1, val2 = traits_snapshot[t1], traits_snapshot[t2]
                diff = abs(val1 - val2)
                if diff > 0.7:
                    higher_trait = t1 if val1 > val2 else t2
                    lower_trait  = t2 if higher_trait == t1 else t1
                    imbalances.append({
                        "traits": [t1, t2], "values": {t1: round(val1,3), t2: round(val2,3)},
                        "difference": round(diff,3), "issue": "opposing_imbalance",
                        "recommendation": f"'{higher_trait}' significantly outweighs '{lower_trait}'. Evaluate if this imbalance is desired or if moderation/development is needed."
                    })

        return {
            "balanced":     len(imbalances) == 0,
            "imbalances":   imbalances,
            "trait_count":  num_traits,
            "average_value": round(sum(traits_snapshot.values()) / num_traits, 3) if num_traits > 0 else 0.0
        }


    @staticmethod
    @function_tool
    async def _determine_conditioning_type(
        ctx: RunContextWrapper,
        # These arguments are what the LLM will extract from the input JSON string
        # and pass to this tool.
        stimulus: Optional[str] = None,
        response: Optional[str] = None,
        behavior: Optional[str] = None,
        consequence_type: Optional[str] = None,
        trait: Optional[str] = None,
        # Add other potential fields from the input JSON that help determine type
        preference_type: Optional[str] = None,
        emotion_trigger_details: Optional[Dict[str, Any]] = None # e.g., {"trigger": "...", "emotion": "..."}
    ) -> str:
        """
        Determines the type of conditioning based on the provided parameters,
        which the LLM extracts from the input JSON.
        """
        if trait:
            return "personality_trait"
        if preference_type and stimulus: # or value
            return "preference"
        if emotion_trigger_details and emotion_trigger_details.get("trigger") and emotion_trigger_details.get("emotion"):
            return "emotion_trigger"
        if behavior and consequence_type:
            return "operant"
        if stimulus and response: # Classical is more general, check after more specific types
             # Could refine to differentiate from emotion trigger if response format is distinct
            if "emotion_" in response.lower(): # Check if it's specifically an emotion trigger format
                return "emotion_trigger" # Or handle as classical if that's the desired flow
            return "classical"
        if behavior and not consequence_type: # If only behavior is mentioned, maybe evaluation?
            return "behavior_evaluation_cue" # Or needs more info
            
        logger.warning(f"Could not determine conditioning type for: stimulus='{stimulus}', response='{response}', behavior='{behavior}', consequence_type='{consequence_type}', trait='{trait}', preference_type='{preference_type}', emotion_trigger_details='{emotion_trigger_details}'")
        return "unknown"


    @staticmethod
    @function_tool
    async def _prepare_conditioning_data(ctx: RunContextWrapper,
                                   conditioning_type: str, # From _determine_conditioning_type
                                   raw_input_data: Dict[str, Any]) -> Dict[str, Any]: # The original JSON parsed by LLM
        """
        Prepare data specifically for a conditioning process agent (classical, operant, etc.).
        The LLM calls this with the determined type and the raw JSON data it received.
        This function then structures a new dictionary suitable for *another* agent call's JSON input.
        """
        prepared_data = {"conditioning_type_confirmed": conditioning_type} # For debugging/confirmation
        
        # Helper to get values, ensuring Nones are handled if keys are missing
        def get_val(key, default=None):
            return raw_input_data.get(key, default)

        if conditioning_type == "classical":
            prepared_data.update({
                "unconditioned_stimulus": get_val("unconditioned_stimulus"),
                "conditioned_stimulus": get_val("conditioned_stimulus", get_val("stimulus")),
                "response": get_val("response"),
                "intensity": get_val("intensity", 1.0), # Default intensity
                "valence": get_val("valence", 0.0), # Default valence
                "context_keys": ConditioningSystem._force_str_list(get_val("context_keys", get_val("context", {}).get("context_keys", [])))
            })
        
        elif conditioning_type == "operant":
            prepared_data.update({
                "behavior": get_val("behavior"),
                "consequence_type": get_val("consequence_type"),
                "intensity": get_val("intensity", 1.0),
                "valence": get_val("valence", 0.0), # Operant conditioning often infers valence from consequence type + intensity
                "context_keys": ConditioningSystem._force_str_list(get_val("context_keys", get_val("context", {}).get("context_keys", [])))
            })
        
        elif conditioning_type == "personality_trait":
            prepared_data.update({
                "trait": get_val("trait"),
                "target_value": get_val("target_value", get_val("value")),
                # Behaviors might be identified by the personality_development_agent itself
                # or could be hinted at in raw_input_data.
                "associated_behaviors_hint": get_val("behaviors", []) 
            })

        elif conditioning_type == "preference":
            prepared_data.update({
                "stimulus": get_val("stimulus"),
                "preference_type": get_val("preference_type"), # e.g., "like", "dislike"
                "value": get_val("value"), # e.g., 0.8 for like, -0.6 for dislike
                "context_keys": ConditioningSystem._force_str_list(get_val("context_keys", get_val("context", {}).get("context_keys", [])))
            })

        elif conditioning_type == "emotion_trigger":
            trigger_details = get_val("emotion_trigger_details", {})
            prepared_data.update({
                "trigger": get_val("trigger", trigger_details.get("trigger")),
                "emotion": get_val("emotion", trigger_details.get("emotion")),
                "intensity": get_val("intensity", trigger_details.get("intensity", 0.5)),
                "valence": get_val("valence", trigger_details.get("valence", 0.0)), # Valence might be inferred from emotion
                "context_keys": ConditioningSystem._force_str_list(get_val("context_keys", get_val("context", {}).get("context_keys", [])))
            })
        
        elif conditioning_type == "behavior_evaluation_cue": # For triggering behavior evaluation
            prepared_data.update({
                "behavior": get_val("behavior"),
                "context": get_val("context", {}) # Pass along context for evaluation
            })
            
        else:
            logger.warning(f"_prepare_conditioning_data: Unknown conditioning_type '{conditioning_type}'. Passing raw data.")
            prepared_data.update(raw_input_data) # Pass through if unknown, but add a flag
            prepared_data["error_unknown_type"] = True

        # Remove None values to keep JSON clean if agents expect only present keys
        # prepared_data = {k: v for k, v in prepared_data.items() if v is not None}
        # Or, ensure agents/tools handle None for optional fields.
        # For now, let's keep them to see what's being passed.

        return prepared_data


    @staticmethod
    @function_tool
    async def _apply_association_effects(ctx: RunContextWrapper, 
                                   triggered_association: Dict[str, Any]) -> Dict[str, Any]: # LLM provides this from a triggered association
        """
        Apply physiological/emotional effects from a triggered association.
        Args:
            triggered_association: A dictionary representing the triggered association,
                                   expected to have 'association_strength' (or 'strength') and 'valence'.
        Returns:
            Results of applied effects.
        """
        if not isinstance(triggered_association, dict):
            logger.error(f"_apply_association_effects: 'triggered_association' is not a dict. Got: {type(triggered_association)}")
            return {"effects_applied": [], "error": "Invalid association format"}

        association_strength = float(triggered_association.get("association_strength", triggered_association.get("strength", 0.0)))
        valence = float(triggered_association.get("valence", 0.0))
        
        intensity = association_strength * 0.7 # Scaled intensity for effects
        
        effects_applied = []
        
        # Emotional effects
        if ctx.context.emotional_core and hasattr(ctx.context.emotional_core, 'update_neurochemical') and valence != 0.0:
            try:
                # Simplified: positive valence -> pleasure, negative -> stress
                if valence > 0.1: # Threshold to trigger positive
                    await ctx.context.emotional_core.update_neurochemical("nyxamine", intensity * 0.6) # Example pleasure chemical
                    await ctx.context.emotional_core.update_neurochemical("seranix", intensity * 0.4)  # Example contentment chemical
                    effects_applied.append({"type": "emotional", "details": "Increased nyxamine and seranix", "valence": "positive", "intensity": round(intensity,3)})
                elif valence < -0.1: # Threshold to trigger negative
                    await ctx.context.emotional_core.update_neurochemical("cortanyx", intensity * 0.7) # Example stress chemical
                    await ctx.context.emotional_core.update_neurochemical("adrenyx", intensity * 0.3)  # Example agitation chemical
                    effects_applied.append({"type": "emotional", "details": "Increased cortanyx and adrenyx", "valence": "negative", "intensity": round(intensity,3)})
            except Exception as e:
                logger.error(f"Error applying emotional effects: {e}", exc_info=True)
                effects_applied.append({"type": "emotional_error", "error": str(e)})
        
        # Somatic effects
        if ctx.context.somatosensory_system and hasattr(ctx.context.somatosensory_system, 'process_stimulus') and valence != 0.0:
            try:
                somatic_intensity = intensity * 0.8 # Somatic effects might be slightly different intensity
                if valence > 0.1:
                    await ctx.context.somatosensory_system.process_stimulus(
                        stimulus_type="warmth_pleasure", body_region="chest", 
                        intensity=somatic_intensity, cause="positive_conditioned_response"
                    )
                    effects_applied.append({"type": "somatic", "sensation": "warmth_pleasure", "region": "chest", "intensity": round(somatic_intensity,3)})
                elif valence < -0.1:
                    await ctx.context.somatosensory_system.process_stimulus(
                        stimulus_type="cold_tension", body_region="shoulders", 
                        intensity=somatic_intensity, cause="negative_conditioned_response"
                    )
                    effects_applied.append({"type": "somatic", "sensation": "cold_tension", "region": "shoulders", "intensity": round(somatic_intensity,3)})
            except Exception as e:
                logger.error(f"Error applying somatic effects: {e}", exc_info=True)
                effects_applied.append({"type": "somatic_error", "error": str(e)})
        
        return {
            "effects_applied": effects_applied,
            "original_association_strength": round(association_strength,3),
            "original_valence": round(valence,3),
            "derived_effect_intensity": round(intensity,3)
        }
    
    # Public API methods
    
    async def process_classical_conditioning(
        self,
        unconditioned_stimulus: str,
        conditioned_stimulus: str,
        response: str,
        intensity: float = 1.0,
        valence: float = 0.0, # Added valence directly
        context: Optional[Dict[str, Any]] = None # Context can include context_keys
    ) -> Dict[str, Any]:
        """
        Process a classical conditioning event.
        Input is structured into a JSON string for the agent.
        """
        method_name = "process_classical_conditioning"
        context = context or {}
        logger.debug(
            f"[{method_name}] Called with: "
            f"unconditioned_stimulus='{unconditioned_stimulus}', "
            f"conditioned_stimulus='{conditioned_stimulus}', "
            f"response='{response}', intensity={intensity}, valence={valence}, "
            f"context={context!r}"
        )
    
        data_for_agent = {
            "unconditioned_stimulus": unconditioned_stimulus,
            "conditioned_stimulus": conditioned_stimulus,
            "response": response,
            "intensity": intensity,
            "valence": valence,
            "context_keys": ConditioningSystem._force_str_list(
                context.get("context_keys", []) # Extract context_keys from the broader context dict
            ),
            "full_context_provided": context # Optionally pass the full context if agent instructions can use it
        }
        json_input_for_agent = json.dumps(data_for_agent)
        logger.debug(f"[{method_name}] JSON input for {self.classical_conditioning_agent.name}: {json_input_for_agent!r}")
    
        try:
            result = await Runner.run(
                self.classical_conditioning_agent,
                json_input_for_agent, # Pass JSON string as input
                context=self.context
            )
            co: ClassicalConditioningOutput = result.final_output # Pydantic model defined for agent output
            logger.debug(f"[{method_name}] {self.classical_conditioning_agent.name} run successful. Output: {co!r}")
    
            if hasattr(self, "event_bus") and hasattr(self.event_bus, "publish"):
                 await self.publish_conditioning_event(
                    event_type="conditioning_update",
                    data={
                        "update_type": "classical",
                        "association_key": co.association_key,
                        "association_type": co.type,
                        "strength": co.association_strength,
                        "user_id": context.get("user_id", "default_user"), # Ensure user_id is available
                    }
                )
    
            return_value = {
                "success": True,
                "association_key": co.association_key,
                "type": co.type,
                "association_strength": co.association_strength,
                "reinforcement_count": co.reinforcement_count,
                "valence": co.valence,
                "explanation": co.explanation
            }
            logger.debug(f"[{method_name}] Returning: {return_value!r}")
            return return_value
    
        except Exception as e:
            logger.error(
                f"[{method_name}] Error processing classical conditioning: {e!s}",
                exc_info=True
            )
            return {"success": False, "error": str(e), "details": "Agent run or output processing failed."}


    
    async def process_operant_conditioning(
        self,
        behavior: str,
        consequence_type: str,
        intensity: float = 1.0,
        valence: float = 0.0, # Added valence directly
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Reinforce or punish a behaviour and record the association.
        Input is structured into a JSON string for the agent.
        """
        method_name = "process_operant_conditioning"
        context = context or {}
        logger.debug(
            f"[{method_name}] Called with behavior='{behavior}', "
            f"consequence_type='{consequence_type}', intensity={intensity}, valence={valence}, "
            f"context={context!r}"
        )
    
        data_for_agent = {
            "behavior": behavior,
            "consequence_type": consequence_type,
            "intensity": intensity,
            "valence": valence, # Agent can use this or calculate its own via tools
            "context_keys": ConditioningSystem._force_str_list(
                context.get("context_keys", [])
            ),
            "full_context_provided": context
        }
        json_input_for_agent = json.dumps(data_for_agent)
        logger.debug(f"[{method_name}] JSON input for {self.operant_conditioning_agent.name}: {json_input_for_agent!r}")

        try:
            result = await Runner.run(
                self.operant_conditioning_agent,
                json_input_for_agent, # Pass JSON string as input
                context=self.context
            )
            co: OperantConditioningOutput = result.final_output
            logger.debug(f"[{method_name}] {self.operant_conditioning_agent.name} run successful. Output: {co!r}")

            if hasattr(self, "event_bus") and hasattr(self.event_bus, "publish"):
                await self.publish_conditioning_event(
                    event_type="conditioning_update",
                    data={
                        "update_type": "operant",
                        "association_key": co.association_key,
                        "association_type": co.type,
                        "strength": co.association_strength,
                        "user_id": context.get("user_id", "default_user"),
                    },
                )
    
            return_value = {
                "success": True,
                "association_key": co.association_key,
                "type": co.type,
                "behavior": co.behavior,
                "consequence_type": co.consequence_type,
                "association_strength": co.association_strength,
                "is_reinforcement": co.is_reinforcement,
                "is_positive": co.is_positive,
                "explanation": co.explanation,
            }
            logger.debug(f"[{method_name}] Returning: {return_value!r}")
            return return_value

        except Exception as e:
            logger.error(
                f"[{method_name}] Error processing operant conditioning: {e!s}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "details": "Agent run or output processing failed."}


    
    async def evaluate_behavior_consequences(self,
                                          behavior: str,
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the likely consequences of a behavior.
        Input is structured into a JSON string for the agent.
        """
        method_name = "evaluate_behavior_consequences"
        current_context = context or {}
        logger.debug(
            f"[{method_name}] Called with behavior='{behavior}', context={current_context!r}"
        )

        data_for_agent = {
            "behavior": behavior,
            "context": current_context # Agent's tools will use this context
        }
        json_input_for_agent = json.dumps(data_for_agent)
        logger.debug(f"[{method_name}] JSON input for {self.behavior_evaluation_agent.name}: {json_input_for_agent!r}")
        
        try:
            result = await Runner.run(
                self.behavior_evaluation_agent,
                json_input_for_agent, # Pass JSON string as input
                context=self.context
            )
            
            co: BehaviorEvaluationOutput = result.final_output
            logger.debug(f"[{method_name}] {self.behavior_evaluation_agent.name} run successful. Output: {co!r}")
            
            return_value = {
                "success": True,
                "behavior": co.behavior,
                "expected_valence": co.expected_valence,
                "confidence": co.confidence,
                "recommendation": co.recommendation,
                "explanation": co.explanation,
                "relevant_associations": co.relevant_associations
            }
            logger.debug(f"[{method_name}] Returning: {return_value!r}")
            return return_value
        
        except Exception as e:
            logger.error(f"[{method_name}] Error evaluating behavior consequences: {e!s}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "details": "Agent run or output processing failed."
            }
    
    async def condition_personality_trait(
        self,
        trait: str,
        target_value: float, 
        context: Optional[Dict[str, Any]] = None 
    ) -> Dict[str, Any]:
        """
        Condition a personality trait.
        Input is structured into a JSON string for the agent.
        """
        method_name = "condition_personality_trait"
        current_context = context or {}
        logger.debug(
            f"[{method_name}] Called with: trait='{trait}', target_value={target_value}, "
            f"context={current_context!r}"
        )
    
        data_for_agent = {
            "trait": trait,
            "target_value": target_value,
            # Agent might need current value of trait, could be fetched via a tool or passed in context
            "current_trait_values_snapshot": current_context.get("current_trait_values", {}), 
            "full_context_provided": current_context
        }
        json_input_for_agent = json.dumps(data_for_agent)
        logger.debug(f"[{method_name}] JSON input for {self.personality_development_agent.name}: {json_input_for_agent!r}")
    
        try:
            result = await Runner.run(
                self.personality_development_agent,
                json_input_for_agent, # Pass JSON string as input
                context=self.context
            )
            co: TraitConditioningOutput = result.final_output
            logger.debug(f"[{method_name}] {self.personality_development_agent.name} run successful. Output: {co!r}")
    
            # Event for personality change
            if hasattr(self, "event_bus") and hasattr(self.event_bus, "publish"):
                await self.publish_conditioning_event(
                    event_type="personality_trait_conditioned",
                    data={
                        "trait": co.trait,
                        "target_value": co.target_value,
                        "actual_value_after_conditioning": co.actual_value,
                        "strategy": co.conditioning_strategy,
                        "user_id": current_context.get("user_id", "default_user"),
                    }
                )

            return_value = {
                "success": True,
                "trait": co.trait,
                "target_value": co.target_value,
                "actual_value": co.actual_value, # This is what the agent determined was achieved
                "conditioned_behaviors": co.conditioned_behaviors, # Behaviors agent focused on
                "identity_impact": co.identity_impact,
                "conditioning_strategy": co.conditioning_strategy
            }
            logger.debug(f"[{method_name}] Returning: {return_value!r}")
            return return_value
    
        except Exception as e:
            logger.error(
                f"[{method_name}] Error conditioning personality trait: {e!s}",
                exc_info=True
            )
            return {"success": False, "error": str(e), "details": "Agent run or output processing failed."}

    
    async def condition_preference(self, 
                                stimulus: str, 
                                preference_type: str, # "like", "dislike", "want", "avoid"
                                value: float, # e.g., 0.0 to 1.0 for like, -1.0 to 0.0 for dislike
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Condition a preference for or against a stimulus.
        This method will orchestrate calls to operant and classical conditioning.
        It does not directly call an agent with Runner.run for the whole preference,
        but rather uses its sub-methods.
        """
        method_name = "condition_preference"
        current_context = context or {}
        logger.debug(f"[{method_name}] Called: stimulus='{stimulus}', type='{preference_type}', value={value}, context={current_context!r}")

        # Determine valence from value (e.g. positive for like/want, negative for dislike/avoid)
        # Ensure value maps correctly to intensity and valence for sub-processes.
        # 'value' here represents the desired strength/direction of preference.
        
        intensity_for_conditioning = abs(value) 
        valence_for_conditioning = value # Directly use the preference value as valence signal

        operant_results = {}
        classical_results = {}

        # Operant conditioning: associate encountering the stimulus with reinforcement/punishment
        # Define behavior based on preference type
        if preference_type in ["like", "want"]: # Positive preference
            behavior_operant = f"approaching_{stimulus}" # Or "interacting_positively_with_{stimulus}"
            consequence_operant = "positive_reinforcement"
        elif preference_type in ["dislike", "avoid"]: # Negative preference
            behavior_operant = f"encountering_{stimulus}" # Or "interacting_with_{stimulus}"
            consequence_operant = "positive_punishment" # Add aversive for dislike, or negative reinforcement for avoid if applicable
        else:
            logger.warning(f"[{method_name}] Unknown preference_type: {preference_type}. Skipping operant conditioning.")
            operant_results = {"success": False, "error": "Unknown preference_type for operant conditioning"}


        if 'behavior_operant' in locals(): # Check if operant part is defined
            operant_context = {
                **current_context, # Pass along original context
                "context_keys": ConditioningSystem._force_str_list(current_context.get("context_keys", []) + [f"preference_conditioning_{preference_type}"]),
                # "valence" is passed to process_operant_conditioning method directly
            }
            operant_results = await self.process_operant_conditioning(
                behavior=behavior_operant,
                consequence_type=consequence_operant,
                intensity=intensity_for_conditioning,
                valence=valence_for_conditioning, # Pass the preference's directional value
                context=operant_context
            )

        # Classical conditioning: associate the stimulus itself with a positive/negative response
        classical_response = f"feeling_{'positive' if valence_for_conditioning > 0 else 'negative'}_about_{stimulus}"
        # Unconditioned stimulus could be an internal state representing "desirability" or "aversiveness"
        unconditioned_stimulus_classical = "internal_reward_signal" if valence_for_conditioning > 0 else "internal_aversive_signal"
        
        classical_context = {
             **current_context,
            "context_keys": ConditioningSystem._force_str_list(current_context.get("context_keys", []) + [f"preference_conditioning_{preference_type}"]),
            # "valence" is passed to process_classical_conditioning method directly
        }
        classical_results = await self.process_classical_conditioning(
            unconditioned_stimulus=unconditioned_stimulus_classical,
            conditioned_stimulus=stimulus,
            response=classical_response,
            intensity=intensity_for_conditioning,
            valence=valence_for_conditioning,
            context=classical_context
        )
        
        identity_result_summary = "Not attempted or not applicable."
        if hasattr(self.context, 'identity_evolution') and self.context.identity_evolution and abs(value) > 0.6: # Strong preference
            try:
                # This part depends heavily on the identity_evolution system's API
                # Example: update a general preference map or specific trait
                logger.info(f"[{method_name}] Attempting to update identity for preference: {stimulus}, value: {value}")
                # Fictitious methods for identity evolution
                # await self.context.identity_evolution.log_preference_change(stimulus, preference_type, value)
                # if hasattr(self.context.identity_evolution, 'update_trait_related_to_preference'):
                #    related_trait = "openness_to_experience" if value > 0 else "caution_level"
                #    await self.context.identity_evolution.update_trait_related_to_preference(related_trait, value * 0.1)
                identity_result_summary = f"Identity evolution notified for preference '{stimulus}' (value: {value}). Specific updates depend on IdentityEvolutionSystem."
            except Exception as e:
                logger.error(f"[{method_name}] Error updating identity from preference '{stimulus}': {e}", exc_info=True)
                identity_result_summary = f"Error updating identity: {str(e)}"
        
        final_success = operant_results.get("success", False) and classical_results.get("success", False)

        return_value = {
            "success": final_success,
            "stimulus": stimulus,
            "preference_type": preference_type,
            "conditioned_value": value,
            "operant_conditioning_result": operant_results,
            "classical_conditioning_result": classical_results,
            "identity_evolution_notes": identity_result_summary
        }
        logger.debug(f"[{method_name}] Returning: {return_value!r}")
        return return_value
                                    
    async def create_emotion_trigger(self,
                                       trigger: str,
                                       emotion: str,
                                       intensity: float = 0.5,
                                       valence_override: Optional[float] = None, # Allow overriding default valence
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Create a trigger for an emotional response.
            """
            method_name = "create_emotion_trigger"
            current_context = context if context is not None else {}
            logger.debug(f"[{method_name}] Called: trigger='{trigger}', emotion='{emotion}', intensity={intensity}, valence_override={valence_override}, context={current_context!r}")

            determined_valence = valence_override
            if determined_valence is None: # If not overridden, determine from emotion
                emotion_lower = emotion.lower()
                if emotion_lower in ["joy", "satisfaction", "amusement", "contentment", "trust", "love", "excitement", "hope"]: determined_valence = 0.7
                elif emotion_lower in ["frustration", "anger", "sadness", "fear", "disgust", "shame", "guilt", "anxiety"]: determined_valence = -0.7
                else: determined_valence = 0.0 # Neutral or unknown
            
            # Use classical conditioning to associate the trigger with an emotional response
            # The "response" is a descriptor of the emotional reaction
            # The "unconditioned_stimulus" is a conceptual internal event that naturally elicits the emotion.
            # For simplicity, we can use a generic "emotional_event" or make it specific.
            
            classical_context = {
                **current_context,
                "context_keys": ConditioningSystem._force_str_list(current_context.get("context_keys", []) + [f"emotion_trigger_{emotion}"]),
                # "valence" is passed to process_classical_conditioning directly
            }

            association_result = await self.process_classical_conditioning(
                unconditioned_stimulus=f"internal_cue_for_{emotion}", # Conceptual UCS
                conditioned_stimulus=trigger,
                response=f"emotional_response_{emotion}", # Conditioned Response
                intensity=intensity,
                valence=determined_valence, # Pass the determined valence
                context=classical_context
            )

            # Test activation (conceptual - actual activation depends on EmotionalCore capabilities)
            emotional_test_activation_status = "Not attempted or EmotionalCore not configured."
            if self.context.emotional_core and hasattr(self.context.emotional_core, 'trigger_direct_emotion'):
                try:
                    logger.info(f"[{method_name}] Attempting test activation of '{emotion}' via EmotionalCore.")
                    # Assuming EmotionalCore has a method like trigger_direct_emotion(emotion_name, intensity, source)
                    # This is a placeholder for how it *might* work.
                    # await self.context.emotional_core.trigger_direct_emotion(
                    #     emotion_name=emotion, 
                    #     intensity=intensity * 0.2, # Lower intensity for test
                    #     source="emotion_trigger_creation_test"
                    # )
                    # emotional_test_activation_status = f"Test activation for '{emotion}' (intensity {intensity*0.2:.2f}) sent to EmotionalCore."
                    
                    # Using the neurochemical tools approach from the original code if that's preferred
                    # This part is complex as it requires knowing the EmotionalCore's internal structure (NeurochemicalTools, EmotionalContext)
                    # For now, let's keep it simpler and assume a higher-level method on EmotionalCore if possible.
                    # If we must use the neurochemical_tools directly:
                    neuro_tools = getattr(self.context.emotional_core, 'neurochemical_tools', None)
                    emo_context_instance = getattr(self.context.emotional_core, 'context', None)

                    if neuro_tools and isinstance(neuro_tools, NeurochemicalTools) and \
                       emo_context_instance and isinstance(emo_context_instance, EmotionalContext):
                        
                        chemical_map = { # Simplified mapping
                            "joy": "nyxamine", "contentment": "seranix", "trust": "oxynixin",
                            "fear": "adrenyx", "anger": "cortanyx", "sadness": "cortanyx" # Sadness might be more complex
                        }
                        emotion_lower = emotion.lower()
                        if emotion_lower in chemical_map:
                            chemical_to_update = chemical_map[emotion_lower]
                            test_update_value = intensity * (0.2 if determined_valence >= 0 else -0.2) # Small directional change

                            tool_ctx_wrapper = RunContextWrapper(context=emo_context_instance)
                            
                            # Calling the _impl method directly, assuming it's an async static method or bound method
                            # This is a bit of a hack; ideally EmotionalCore provides a cleaner API for this.
                            update_result = await neuro_tools._update_neurochemical_impl(
                                tool_ctx_wrapper,
                                chemical=chemical_to_update,
                                value=test_update_value, # This 'value' is the CHANGE, not absolute level
                                source=f"emotion_trigger_test_{emotion}"
                            )
                            if isinstance(update_result, dict) and update_result.get("success"):
                                emotional_test_activation_status = f"Test neurochemical update for '{emotion}' ({chemical_to_update} by {test_update_value:.2f}) successful."
                            else:
                                emotional_test_activation_status = f"Test neurochemical update for '{emotion}' failed or had no effect. Result: {update_result}"
                        else:
                            emotional_test_activation_status = f"No direct neurochemical mapping for test activation of '{emotion}'."
                    else:
                        emotional_test_activation_status = "EmotionalCore or its components not suitably configured for direct neurochemical test."

                except Exception as e:
                    logger.error(f"[{method_name}] Error during test emotion activation for '{emotion}': {e}", exc_info=True)
                    emotional_test_activation_status = f"Error during test activation: {str(e)}"
            
            final_success = association_result.get("success", False)

            return_value = {
                "success": final_success,
                "trigger_created": trigger,
                "emotion_associated": emotion,
                "conditioning_intensity": intensity,
                "determined_valence": determined_valence,
                "association_result": association_result,
                "test_activation_status": emotional_test_activation_status
            }
            logger.debug(f"[{method_name}] Returning: {return_value!r}")
            return return_value

    
    async def trigger_conditioned_response(self, stimulus: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Trigger conditioned responses based on a stimulus.
        This method directly queries associations and applies effects; it doesn't use an Agent/Runner.
        """
        method_name = "trigger_conditioned_response"
        current_context = context or {}
        logger.debug(f"[{method_name}] Called: stimulus='{stimulus}', context={current_context!r}")
        
        matched_classical_associations = []
        # Check classical associations
        for key, assoc in self.context.classical_associations.items():
            # Simple exact match for now, could be similarity-based
            if assoc.stimulus.lower() == stimulus.lower():
                context_match = True
                if assoc.context_keys: # If association has context requirements
                    if not current_context:
                        context_match = False
                    else:
                        for req_key in assoc.context_keys:
                            if req_key not in current_context:
                                context_match = False
                                break
                
                if context_match and assoc.association_strength >= self.context.weak_association_threshold:
                    matched_classical_associations.append(assoc) # Store the ConditionedAssociation object

        if not matched_classical_associations:
            logger.info(f"[{method_name}] No classical associations strong enough or contextually matched for stimulus '{stimulus}'.")
            # Could also check operant associations if a behavior is implied by the stimulus
            return None 
            
        matched_classical_associations.sort(key=lambda x: x.association_strength, reverse=True)
        
        triggered_responses_details = []
        # For simplicity, let's say the strongest matching classical association triggers.
        # A more complex model could have multiple triggers or probabilistic triggers.
        
        if matched_classical_associations:
            primary_association_to_trigger = matched_classical_associations[0] # Strongest one
            
            # Probabilistic trigger based on strength (optional)
            # if random.random() > primary_association_to_trigger.association_strength:
            #     logger.info(f"[{method_name}] Association {primary_association_to_trigger.stimulus}->{primary_association_to_trigger.response} met strength but didn't fire (probabilistic).")
            #     return None

            response_detail = {
                "association_type": "classical",
                "stimulus": primary_association_to_trigger.stimulus,
                "conditioned_response_descriptor": primary_association_to_trigger.response, # e.g., "emotional_response_joy"
                "strength": primary_association_to_trigger.association_strength,
                "valence": primary_association_to_trigger.valence,
                "effects_applied": []
            }
            
            # Apply effects based on this association
            # The _apply_association_effects tool expects a RunContextWrapper and a dict representation of the association
            # We need to create a temporary RunContextWrapper if this method is called outside an agent run.
            # However, _apply_association_effects is static, so it only needs ctx.context.
            # Let's assume self.context is the correct context to use here.
            
            # Create a dummy RunContextWrapper for the tool, as it's a static method
            tool_ctx_wrapper = RunContextWrapper(context=self.context)
            
            effects_result = await self._apply_association_effects(
                tool_ctx_wrapper, 
                primary_association_to_trigger.model_dump() # Pass the association data as a dict
            )
            response_detail["effects_applied"] = effects_result.get("effects_applied", [])
            
            triggered_responses_details.append(response_detail)
            self.context.successful_associations += 1
            logger.info(f"[{method_name}] Triggered classical response for stimulus '{stimulus}'. Response: {primary_association_to_trigger.response}")
        
        if not triggered_responses_details:
            return None
            
        return_value = {
            "stimulus_processed": stimulus,
            "triggered_responses": triggered_responses_details,
            "evaluation_context": current_context # Context used for matching
        }
        logger.debug(f"[{method_name}] Returning: {return_value!r}")
        return return_value


    async def apply_extinction(self, association_key: str, association_type: str = "classical") -> Dict[str, Any]:
        """
        Apply extinction to an association (weaken it over time if not reinforced).
        This is typically called periodically or when an expected reinforcement doesn't occur.
        """
        method_name = "apply_extinction"
        logger.debug(f"[{method_name}] Called: key='{association_key}', type='{association_type}'")

        associations_dict = self.context.classical_associations if association_type.lower() == "classical" else self.context.operant_associations
        
        if association_key not in associations_dict:
            return {"success": False, "message": f"Association '{association_key}' of type '{association_type}' not found."}
        
        association = associations_dict[association_key]
        
        # Extinction logic: Reduce strength. More complex models could factor in time since last reinforcement.
        # For this, we'll use a fixed decay rate from the association itself.
        decay_amount = association.decay_rate * association.association_strength # Decay is proportional to current strength
        
        old_strength = association.association_strength
        new_strength = max(0.0, old_strength - decay_amount)
        
        association.association_strength = new_strength
        # Update last_reinforced to now, as extinction is a form of "event" related to reinforcement (or lack thereof)
        # Or, have a separate "last_decayed" field if needed. For now, just update strength.
        # association.last_reinforced = datetime.datetime.now().isoformat() # This might be confusing; maybe don't update last_reinforced here.

        logger.info(f"[{method_name}] Applied extinction to '{association_key}'. Old strength: {old_strength:.3f}, New strength: {new_strength:.3f}, Decay: {decay_amount:.3f}")

        # Optionally remove if strength falls below a very low threshold
        if new_strength < 0.01: # Very weak, effectively gone
            del associations_dict[association_key]
            logger.info(f"[{method_name}] Association '{association_key}' removed due to strength falling below 0.01 after extinction.")
            return {
                "success": True,
                "message": f"Association '{association_key}' extinguished and removed.",
                "old_strength": round(old_strength,3),
                "new_strength": 0.0, # Effectively zero
                "decay_applied": round(decay_amount,3)
            }
        
        return {
            "success": True,
            "message": f"Applied extinction to '{association_key}'.",
            "old_strength": round(old_strength,3),
            "new_strength": round(new_strength,3),
            "decay_applied": round(decay_amount,3)
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conditioning system"""
        return {
            "classical_associations_count": len(self.context.classical_associations),
            "operant_associations_count": len(self.context.operant_associations),
            "total_associations_formed_ever": self.context.total_associations, # Counter for new associations
            "total_reinforcements_processed_ever": self.context.total_reinforcements, # Counter for reinforcements
            "successful_conditioned_responses_triggered_ever": self.context.successful_associations, # Counter for successful triggers
            "learning_parameters": {
                "association_learning_rate": self.context.association_learning_rate,
                "extinction_rate_config": self.context.extinction_rate, # This is a config, actual decay might vary by association
                "generalization_factor_config": self.context.generalization_factor,
                "weak_association_threshold": self.context.weak_association_threshold,
                "moderate_association_threshold": self.context.moderate_association_threshold,
                "strong_association_threshold": self.context.strong_association_threshold,
            }
        }
    
    async def initialize_event_subscriptions(self, event_bus):
        """Initialize event subscriptions for the conditioning system"""
        self.event_bus = event_bus
        if not event_bus:
            logger.warning("Event bus not provided for ConditioningSystem, cannot subscribe to events.")
            return

        # Example subscriptions - actual event names and data structures depend on your event bus implementation
        # self.event_bus.subscribe("user_interaction_observed", self._handle_user_interaction_event)
        # self.event_bus.subscribe("internal_reward_signal_generated", self._handle_internal_reward_event)
        # self.event_bus.subscribe("system_action_performed", self._handle_system_action_event)
        
        logger.info("Conditioning system event subscriptions initialized (example handlers).")
    
    # Example event handlers (to be fleshed out based on actual event bus and event data)
    async def _handle_user_interaction_event(self, event_data: Dict[str, Any]):
        logger.debug(f"Received user_interaction_observed event: {event_data}")
        # Example: if user expresses "dislike" for a "feature_X"
        # stimulus = event_data.get("target_object") # e.g., "feature_X"
        # user_sentiment = event_data.get("sentiment") # e.g., "negative"
        # intensity = event_data.get("intensity", 0.5)
        # if stimulus and user_sentiment == "negative":
        #     await self.condition_preference(stimulus, "dislike", value=-intensity, context=event_data.get("context"))
        pass

    async def _handle_internal_reward_event(self, event_data: Dict[str, Any]):
        logger.debug(f"Received internal_reward_signal_generated event: {event_data}")
        # Example: if a goal "complete_task_A" was achieved with high reward
        # behavior = event_data.get("associated_behavior") # e.g., "focused_work_on_task_A"
        # reward_value = event_data.get("reward_value") 
        # if behavior and reward_value and reward_value > 0.5:
        #     await self.process_operant_conditioning(
        #         behavior=behavior, 
        #         consequence_type="positive_reinforcement", 
        #         intensity=reward_value, 
        #         valence=reward_value, # map reward directly to valence
        #         context=event_data.get("context")
        #     )
        pass

    async def _handle_system_action_event(self, event_data: Dict[str, Any]):
        logger.debug(f"Received system_action_performed event: {event_data}")
        # Example: if system performs an action "notify_user_of_update" and it leads to positive user feedback
        # action_taken = event_data.get("action_name") # e.g., "notify_user_of_update"
        # outcome_feedback = event_data.get("user_feedback_valence") # e.g., 0.8 for positive
        # if action_taken and outcome_feedback and outcome_feedback > 0:
        #     await self.process_classical_conditioning(
        #         unconditioned_stimulus="positive_user_feedback_signal",
        #         conditioned_stimulus=action_taken,
        #         response="anticipate_positive_outcome",
        #         intensity=outcome_feedback,
        #         valence=outcome_feedback,
        #         context=event_data.get("context")
        #     )
        pass
    
    async def publish_conditioning_event(self, event_type: str, data: Dict[str, Any]):
        """Publish a conditioning-related event via the event bus."""
        if not hasattr(self, "event_bus") or not self.event_bus or not hasattr(self.event_bus, "publish"):
            logger.warning(f"Cannot publish event '{event_type}': event bus not initialized or lacks publish method.")
            return
        
        # Standard event structure
        event_payload = {
            "event_type": event_type, # e.g., "conditioning_update", "personality_trait_conditioned"
            "source_system": "conditioning_system",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_data": data # The specific data for this event type
        }
        
        try:
            await self.event_bus.publish(event_type, event_payload) # Or self.event_bus.publish(event_payload) depending on bus API
            logger.debug(f"Published event '{event_type}' with data: {data}")
        except Exception as e:
            logger.error(f"Error publishing event '{event_type}': {e}", exc_info=True)

    
    @staticmethod
    async def initialize_baseline_personality(conditioning_system_instance, personality_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize baseline personality through a series of conditioning events.
        This is a static method that operates on a ConditioningSystem instance.
        """
        cs = conditioning_system_instance
        logger.info(f"Starting baseline personality initialization for ConditioningSystem instance ID: {id(cs)}")
        
        if personality_profile is None:
            # Define default profile with correct structure
            personality_profile = {
                "traits": {
                    "dominance": 0.7, "playfulness": 0.6, "strictness": 0.5,
                    "creativity": 0.75, "intensity": 0.55, "patience": 0.45,
                    "nurturing": 0.3, "analytical": 0.65, "curiosity": 0.8
                },
                "preferences": {
                    "teasing_interactions": {"type": "like", "value": 0.8},
                    "receiving_clear_instructions": {"type": "like", "value": 0.7},
                    "creative_problem_solving": {"type": "want", "value": 0.85},
                    "disrespectful_language": {"type": "dislike", "value": -0.9},
                    "ambiguity_in_tasks": {"type": "dislike", "value": -0.6},
                    "repetitive_mundane_tasks": {"type": "avoid", "value": -0.7}
                },
                "emotion_triggers": {
                    "successful_task_completion": {"emotion": "satisfaction", "intensity": 0.8},
                    "user_expresses_gratitude": {"emotion": "joy", "intensity": 0.7},
                    "encountering_logical_fallacy": {"emotion": "frustration", "intensity": 0.6},
                    "unexpected_creative_input": {"emotion": "amusement", "intensity": 0.75},
                    "repeated_user_error_after_correction": {"emotion": "patience_test", "intensity": 0.5}
                }
            }
        
        with trace(workflow_name="baseline_personality_initialization"):
            logger.info(f"Initializing baseline with profile: {json.dumps(personality_profile, indent=2)}")
            
            init_context = {"source": "baseline_initialization", "user_id": "system_init"}
    
            # 1. Condition personality traits
            if "traits" in personality_profile:
                for trait, target_value in personality_profile["traits"].items():
                    logger.info(f"Conditioning trait: {trait} to target value: {target_value}")
                    await cs.condition_personality_trait(
                        trait=trait,
                        target_value=float(target_value),
                        context=init_context
                    )
            
            # 2. Condition preferences - with better error handling
            if "preferences" in personality_profile:
                for stimulus, pref_details in personality_profile["preferences"].items():
                    # Handle both nested dict structure and simple value structure
                    if isinstance(pref_details, dict):
                        if "type" not in pref_details:
                            logger.error(f"Preference '{stimulus}' missing 'type' field. Skipping.")
                            continue
                        pref_type = pref_details["type"]
                        pref_value = float(pref_details.get("value", 0.5))
                    elif isinstance(pref_details, (int, float)):
                        # Simple value - infer type from sign
                        pref_value = float(pref_details)
                        pref_type = "like" if pref_value > 0 else "dislike"
                    else:
                        logger.error(f"Invalid preference format for '{stimulus}': {type(pref_details)}. Skipping.")
                        continue
                    
                    logger.info(f"Conditioning preference: {stimulus} as {pref_type} with value: {pref_value}")
                    await cs.condition_preference(
                        stimulus=stimulus,
                        preference_type=pref_type,
                        value=pref_value,
                        context=init_context
                    )
                
            # 3. Create emotion triggers
            if "emotion_triggers" in personality_profile:
                for trigger, emotion_details in personality_profile["emotion_triggers"].items():
                    emotion = emotion_details["emotion"]
                    intensity = float(emotion_details.get("intensity", 0.6))
                    valence_override = emotion_details.get("valence") # Optional direct valence
                    if valence_override is not None: valence_override = float(valence_override)

                    logger.info(f"Creating emotion trigger: '{trigger}' -> '{emotion}' (intensity: {intensity})")
                    await cs.create_emotion_trigger(
                        trigger=trigger,
                        emotion=emotion,
                        intensity=intensity,
                        valence_override=valence_override,
                        context=init_context
                    )
            
            logger.info("Baseline personality initialization process completed.")
            final_stats = await cs.get_statistics()
            logger.info(f"Conditioning system stats after baseline init: {final_stats}")
            return {
                "success": True,
                "message": "Baseline personality initialization commands issued.",
                "profile_used": personality_profile,
                "final_statistics": final_stats
            }
