# nyx/core/conditioning_system.py

import json
import logging
import random
import time
from typing import Dict, List, Any, Optional

from agents import Agent, Runner, trace, ModelSettings, handoff
from nyx.core.conditioning_models import *
from nyx.core.conditioning_tools import *
from nyx.core.reward.reward_buffer import RewardBuffer
from nyx.core.reward.evaluator import adjust_association_strengths

logger = logging.getLogger(__name__)

class ConditioningSystem:
    """
    System for implementing classical and operant conditioning mechanisms
    to shape AI personality, preferences, and behaviors.
    """
    
    def __init__(self, reward_system=None, emotional_core=None, memory_core=None, somatosensory_system=None):
        # Create shared context
        self.context = ConditioningContext(
            reward_system=reward_system,
            emotional_core=emotional_core,
            memory_core=memory_core,
            somatosensory_system=somatosensory_system
        )

        # Reward evaluation helpers
        self.reward_buffer = RewardBuffer()
        self._event_count = 0
        self._weak_epochs: Dict[str, int] = {}
        self._last_flush = time.time()

        # Reference associations from context
        self.classical_associations = self.context.classical_associations
        self.operant_associations = self.context.operant_associations
        
        # Create agents
        self._create_agents()
        
        logger.info("Conditioning system initialized with Agents SDK")
    
    def _create_agents(self):
        """Create all conditioning agents"""
        
        # Classical Conditioning Agent
        self.classical_conditioning_agent = Agent(
            name="Classical_Conditioning_Agent",
            instructions="""
            You process classical conditioning events. Your input is a JSON string with:
            - unconditioned_stimulus, conditioned_stimulus, response
            - intensity, valence, context_keys
            
            Create or update associations between stimuli and responses.
            Use the tools to manage associations and calculate strengths.
            Output a ClassicalConditioningOutput with your analysis.
            """,
            tools=[
                get_association,
                create_or_update_classical_association,
                calculate_association_strength,
                check_similar_associations
            ],
            output_type=ClassicalConditioningOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Operant Conditioning Agent
        self.operant_conditioning_agent = Agent(
            name="Operant_Conditioning_Agent",
            instructions="""
            You process operant conditioning events. Your input is a JSON string with:
            - behavior, consequence_type, intensity, valence, context_keys
            
            Analyze behavior-consequence relationships and apply appropriate
            reinforcement or punishment effects. Use the tools to manage associations
            and generate reward signals. Output an OperantConditioningOutput.
            """,
            tools=[
                get_association,
                create_or_update_operant_association,
                calculate_valence_and_reward,
                generate_reward_signal
            ],
            output_type=OperantConditioningOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Behavior Evaluation Agent
        self.behavior_evaluation_agent = Agent(
            name="Behavior_Evaluation_Agent",
            instructions="""
            You evaluate potential behaviors based on conditioning history.
            Your input is a JSON string with: behavior, context
            
            Analyze relevant associations and predict likely outcomes.
            Important: When calling calculate_expected_valence, convert
            associations to JSON string using json.dumps().
            Output a BehaviorEvaluationOutput with recommendations.
            """,
            tools=[
                get_behavior_associations,
                calculate_expected_valence,
                check_context_relevance,
                get_reinforcement_history
            ],
            output_type=BehaviorEvaluationOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Personality Development Agent
        self.personality_development_agent = Agent(
            name="Personality_Development_Agent",
            instructions="""
            You condition personality traits. Your input is a JSON string with:
            - trait, target_value, current_trait_values_snapshot (optional)
            
            Guide trait development through conditioning. Identify relevant behaviors,
            calculate adjustments, and update identity. When calling check_trait_balance,
            pass the traits dictionary directly as 'traits_snapshot'.
            Output a TraitConditioningOutput.
            """,
            tools=[
                identify_trait_behaviors,
                calculate_conditioning_trait_adjustment,
                update_identity_trait,
                check_trait_balance
            ],
            output_type=TraitConditioningOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Conditioning Orchestrator
        self.conditioning_orchestrator = Agent(
            name="Conditioning_Orchestrator",
            instructions="""
            You coordinate conditioning processes. Your input is a JSON string
            describing a conditioning event. Determine the appropriate type and
            route to specialized agents via handoffs.
            
            Use determine_conditioning_type and prepare_conditioning_data to
            prepare for handoffs. Apply effects directly when needed.
            """,
            handoffs=[
                handoff(self.classical_conditioning_agent,
                       tool_name_override="process_classical_conditioning"),
                handoff(self.operant_conditioning_agent,
                       tool_name_override="process_operant_conditioning"),
                handoff(self.behavior_evaluation_agent,
                       tool_name_override="evaluate_behavior"),
                handoff(self.personality_development_agent,
                       tool_name_override="develop_personality_trait")
            ],
            tools=[
                determine_conditioning_type,
                prepare_conditioning_data,
                apply_association_effects
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # ==================== Public API Methods ====================

    async def record_event(self, event_type: str) -> None:
        """Record a simple event for reward evaluation."""
        await self.reward_buffer.add_event(event_type)
        self._event_count += 1
        now = time.time()
        if self._event_count % 50 == 0 or now - self._last_flush > 600:
            batch = await self.reward_buffer.next_batch()
            if batch:
                adjust_association_strengths(self, batch)
            self._last_flush = now

    async def process_classical_conditioning(
        self,
        unconditioned_stimulus: str,
        conditioned_stimulus: str,
        response: str,
        intensity: float = 1.0,
        valence: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a classical conditioning event"""
        context = context or {}
        
        data = {
            "unconditioned_stimulus": unconditioned_stimulus,
            "conditioned_stimulus": conditioned_stimulus,
            "response": response,
            "intensity": intensity,
            "valence": valence,
            "context_keys": context.get("context_keys", [])
        }
        
        try:
            result = await Runner.run(
                self.classical_conditioning_agent,
                json.dumps(data),
                context=self.context
            )
            
            output = result.final_output
            return {
                "success": True,
                "association_key": output.association_key,
                "type": output.type,
                "association_strength": output.association_strength,
                "reinforcement_count": output.reinforcement_count,
                "valence": output.valence,
                "explanation": output.explanation
            }
        except Exception as e:
            logger.error(f"Error in classical conditioning: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_operant_conditioning(
        self,
        behavior: str,
        consequence_type: str,
        intensity: float = 1.0,
        valence: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an operant conditioning event"""
        context = context or {}
        
        data = {
            "behavior": behavior,
            "consequence_type": consequence_type,
            "intensity": intensity,
            "valence": valence,
            "context_keys": context.get("context_keys", [])
        }
        
        try:
            result = await Runner.run(
                self.operant_conditioning_agent,
                json.dumps(data),
                context=self.context
            )
            
            output = result.final_output
            return {
                "success": True,
                "association_key": output.association_key,
                "type": output.type,
                "behavior": output.behavior,
                "consequence_type": output.consequence_type,
                "association_strength": output.association_strength,
                "is_reinforcement": output.is_reinforcement,
                "is_positive": output.is_positive,
                "explanation": output.explanation
            }
        except Exception as e:
            logger.error(f"Error in operant conditioning: {e}")
            return {"success": False, "error": str(e)}
    
    async def evaluate_behavior_consequences(
        self,
        behavior: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate the likely consequences of a behavior"""
        current_context = context or {}
        
        data = {
            "behavior": behavior,
            "context": current_context
        }
        
        try:
            result = await Runner.run(
                self.behavior_evaluation_agent,
                json.dumps(data),
                context=self.context
            )
            
            output = result.final_output
            return {
                "success": True,
                "behavior": output.behavior,
                "expected_valence": output.expected_valence,
                "confidence": output.confidence,
                "recommendation": output.recommendation,
                "explanation": output.explanation,
                "relevant_associations": output.relevant_associations
            }
        except Exception as e:
            logger.error(f"Error evaluating behavior: {e}")
            return {"success": False, "error": str(e)}
    
    async def condition_personality_trait(
        self,
        trait: str,
        target_value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Condition a personality trait"""
        current_context = context or {}
        
        data = {
            "trait": trait,
            "target_value": target_value,
            "current_trait_values_snapshot": current_context.get("current_trait_values", {})
        }
        
        try:
            result = await Runner.run(
                self.personality_development_agent,
                json.dumps(data),
                context=self.context
            )
            
            output = result.final_output
            return {
                "success": True,
                "trait": output.trait,
                "target_value": output.target_value,
                "actual_value": output.actual_value,
                "conditioned_behaviors": output.conditioned_behaviors,
                "identity_impact": output.identity_impact,
                "conditioning_strategy": output.conditioning_strategy
            }
        except Exception as e:
            logger.error(f"Error conditioning trait: {e}")
            return {"success": False, "error": str(e)}
    
    async def condition_preference(
        self,
        stimulus: str,
        preference_type: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Condition a preference for or against a stimulus"""
        current_context = context or {}
        
        # Orchestrate classical and operant conditioning
        intensity = abs(value)
        valence = value
        
        # Operant conditioning
        if preference_type in ["like", "want"]:
            behavior = f"approaching_{stimulus}"
            consequence = "positive_reinforcement"
        else:
            behavior = f"encountering_{stimulus}"
            consequence = "positive_punishment"
        
        operant_result = await self.process_operant_conditioning(
            behavior=behavior,
            consequence_type=consequence,
            intensity=intensity,
            valence=valence,
            context=current_context
        )
        
        # Classical conditioning
        classical_response = f"feeling_{'positive' if valence > 0 else 'negative'}_about_{stimulus}"
        unconditioned = "internal_reward_signal" if valence > 0 else "internal_aversive_signal"
        
        classical_result = await self.process_classical_conditioning(
            unconditioned_stimulus=unconditioned,
            conditioned_stimulus=stimulus,
            response=classical_response,
            intensity=intensity,
            valence=valence,
            context=current_context
        )
        
        success = operant_result.get("success", False) and classical_result.get("success", False)
        
        return {
            "success": success,
            "stimulus": stimulus,
            "preference_type": preference_type,
            "conditioned_value": value,
            "operant_conditioning_result": operant_result,
            "classical_conditioning_result": classical_result
        }
    
    async def create_emotion_trigger(
        self,
        trigger: str,
        emotion: str,
        intensity: float = 0.5,
        valence_override: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a trigger for an emotional response"""
        current_context = context or {}
        
        # Determine valence from emotion if not overridden
        if valence_override is None:
            positive_emotions = ["joy", "satisfaction", "amusement", "contentment", "trust"]
            negative_emotions = ["frustration", "anger", "sadness", "fear", "disgust"]
            
            if emotion.lower() in positive_emotions:
                valence = 0.7
            elif emotion.lower() in negative_emotions:
                valence = -0.7
            else:
                valence = 0.0
        else:
            valence = valence_override
        
        # Use classical conditioning
        result = await self.process_classical_conditioning(
            unconditioned_stimulus=f"internal_cue_for_{emotion}",
            conditioned_stimulus=trigger,
            response=f"emotional_response_{emotion}",
            intensity=intensity,
            valence=valence,
            context=current_context
        )
        
        return {
            "success": result.get("success", False),
            "trigger_created": trigger,
            "emotion_associated": emotion,
            "conditioning_intensity": intensity,
            "determined_valence": valence,
            "association_result": result
        }
    
    async def trigger_conditioned_response(
        self,
        stimulus: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Trigger conditioned responses based on a stimulus"""
        current_context = context or {}
        
        # Find matching associations
        matched_associations = []
        
        for key, assoc in self.classical_associations.items():
            if assoc.stimulus.lower() == stimulus.lower():
                # Check context match
                context_match = True
                if assoc.context_keys:
                    for req_key in assoc.context_keys:
                        if req_key not in current_context:
                            context_match = False
                            break
                
                if context_match and assoc.association_strength >= self.context.parameters.weak_association_threshold:
                    matched_associations.append(assoc)
        
        if not matched_associations:
            return None
        
        # Trigger strongest association
        matched_associations.sort(key=lambda x: x.association_strength, reverse=True)
        primary_association = matched_associations[0]
        
        # Apply effects
        ctx_wrapper = RunContextWrapper(context=self.context)
        effects_result = await apply_association_effects(
            ctx_wrapper,
            primary_association.model_dump()
        )
        
        self.context.successful_associations += 1
        
        return {
            "stimulus_processed": stimulus,
            "triggered_responses": [{
                "association_type": "classical",
                "stimulus": primary_association.stimulus,
                "response": primary_association.response,
                "strength": primary_association.association_strength,
                "valence": primary_association.valence,
                "effects_applied": effects_result.get("effects_applied", [])
            }]
        }
    
    async def apply_extinction(
        self,
        association_key: str,
        association_type: str = "classical"
    ) -> Dict[str, Any]:
        """Apply extinction to an association"""
        associations = self.classical_associations if association_type == "classical" else self.operant_associations
        
        if association_key not in associations:
            return {"success": False, "message": f"Association '{association_key}' not found"}
        
        association = associations[association_key]
        decay_amount = association.decay_rate * association.association_strength
        
        old_strength = association.association_strength
        new_strength = max(0.0, old_strength - decay_amount)
        association.association_strength = new_strength
        
        # Remove if too weak
        if new_strength < 0.01:
            del associations[association_key]
            return {
                "success": True,
                "message": f"Association '{association_key}' removed",
                "old_strength": round(old_strength, 3),
                "new_strength": 0.0,
                "decay_applied": round(decay_amount, 3)
            }
        
        return {
            "success": True,
            "message": f"Applied extinction to '{association_key}'",
            "old_strength": round(old_strength, 3),
            "new_strength": round(new_strength, 3),
            "decay_applied": round(decay_amount, 3)
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conditioning system"""
        return {
            "classical_associations_count": len(self.classical_associations),
            "operant_associations_count": len(self.operant_associations),
            "total_associations_formed_ever": self.context.total_associations,
            "total_reinforcements_processed_ever": self.context.total_reinforcements,
            "successful_conditioned_responses_triggered_ever": self.context.successful_associations,
            "learning_parameters": self.context.parameters.model_dump()
        }
    
    @staticmethod
    async def initialize_baseline_personality(
        conditioning_system_instance,
        personality_profile: Optional[Dict[str, Any]] = None
    ):
        """Initialize baseline personality through conditioning"""
        cs = conditioning_system_instance
        
        if personality_profile is None:
            personality_profile = {
                "traits": {
                    "dominance": 0.7,
                    "playfulness": 0.6,
                    "strictness": 0.5,
                    "creativity": 0.75,
                    "intensity": 0.55
                },
                "preferences": {
                    "teasing_interactions": {"type": "like", "value": 0.8},
                    "creative_problem_solving": {"type": "want", "value": 0.85}
                },
                "emotion_triggers": {
                    "successful_task_completion": {"emotion": "satisfaction", "intensity": 0.8},
                    "user_expresses_gratitude": {"emotion": "joy", "intensity": 0.7}
                }
            }
        
        with trace(workflow_name="baseline_personality_initialization"):
            init_context = {"source": "baseline_initialization"}
            
            # Condition traits
            if "traits" in personality_profile:
                for trait, value in personality_profile["traits"].items():
                    await cs.condition_personality_trait(trait, float(value), init_context)
            
            # Condition preferences
            if "preferences" in personality_profile:
                for stimulus, details in personality_profile["preferences"].items():
                    if isinstance(details, dict):
                        await cs.condition_preference(
                            stimulus,
                            details["type"],
                            float(details.get("value", 0.5)),
                            init_context
                        )
            
            # Create emotion triggers
            if "emotion_triggers" in personality_profile:
                for trigger, details in personality_profile["emotion_triggers"].items():
                    await cs.create_emotion_trigger(
                        trigger,
                        details["emotion"],
                        float(details.get("intensity", 0.6)),
                        details.get("valence"),
                        init_context
                    )
            
            stats = await cs.get_statistics()
            return {
                "success": True,
                "profile_used": personality_profile,
                "final_statistics": stats
            }
