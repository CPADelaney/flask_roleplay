# nyx/nyx_agent/compatability.py

"""Legacy compatibility layer for Nyx Agent SDK"""

import time
import json
import logging
from typing import Dict, List, Any, Optional

from agents import RunContextWrapper

from .context import NyxContext
from .config import Config
from .models import *
from .tools_part3 import *
from .utils import _calculate_variance

logger = logging.getLogger(__name__)

# Legacy AgentContext for full backward compatibility
class AgentContext:
    """Full backward compatibility with original AgentContext"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._nyx_context = None
        
        # Initialize all legacy attributes
        self.memory_system = None
        self.user_model = None
        self.task_integration = None
        self.belief_system = None
        self.emotional_system = None
        self.current_goals = []
        self.active_tasks = []
        self.decision_history = []
        self.state_history = []
        self.last_action = None
        self.last_result = None
        self.current_emotional_state = {}
        self.beliefs = {}
        self.intentions = []
        self.action_success_rate = 0.0
        self.decision_confidence = 0.0
        self.goal_progress = {}
        self.performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_decision_time": 0.0,
            "adaptation_rate": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "response_times": [],
            "error_rates": {
                "total": 0,
                "recovered": 0,
                "unrecovered": 0
            }
        }
        self.learned_patterns = {}
        self.strategy_effectiveness = {}
        self.adaptation_history = []
        self.learning_metrics = {
            "pattern_recognition_rate": 0.0,
            "strategy_improvement_rate": 0.0,
            "adaptation_success_rate": 0.0
        }
        self.resource_pools = {}
        self.resource_usage = {
            "memory": 0,
            "cpu": 0,
            "network": 0
        }
        self.context_cache = {}
        self.communication_history = []
        self.error_log = []
    
    @classmethod
    async def create(cls, user_id: int, conversation_id: int):
        """Async factory method for compatibility"""
        instance = cls(user_id, conversation_id)
        instance._nyx_context = NyxContext(user_id, conversation_id)
        await instance._nyx_context.initialize()
        
        # Map to legacy attributes
        instance.memory_system = instance._nyx_context.memory_system
        instance.user_model = instance._nyx_context.user_model
        instance.task_integration = instance._nyx_context.task_integration
        instance.belief_system = instance._nyx_context.belief_system
        instance.current_emotional_state = instance._nyx_context.emotional_state
        instance.performance_metrics.update(instance._nyx_context.performance_metrics)
        instance.learned_patterns = instance._nyx_context.learned_patterns
        instance.strategy_effectiveness = instance._nyx_context.strategy_effectiveness
        instance.adaptation_history = instance._nyx_context.adaptation_history
        instance.learning_metrics = instance._nyx_context.learning_metrics
        instance.error_log = instance._nyx_context.error_log
        
        # Load initial state
        await instance._load_initial_state()
        
        return instance
    
    async def _initialize_systems(self):
        """Legacy compatibility method"""
        pass
    
    async def _load_initial_state(self):
        """Load initial state for agent context"""
        pass
    
    async def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision using the decision scoring engine"""
        # Convert options to DecisionOption objects
        decision_options = [DecisionOption(
            id=str(i),
            description=str(opt),
            metadata=DecisionMetadata(data=dict_to_kvlist(opt) if isinstance(opt, dict) else DecisionMetadata().data)
        ) for i, opt in enumerate(options)]
        
        payload_model = ScoreDecisionOptionsInput(
            options=decision_options,
            decision_context=dict_to_kvlist(context)
        )

        result = await score_decision_options(
            RunContextWrapper(context=self._nyx_context),
            payload_model
        )
        decision_data = json.loads(result)
        
        # Update decision history
        self.decision_history.append({
            "timestamp": time.time(),
            "selected_option": decision_data["best_option"],
            "score": decision_data["confidence"],
            "context": context
        })
        
        # Update confidence
        self.decision_confidence = decision_data["confidence"]
        
        return {
            "decision": decision_data["best_option"],
            "confidence": decision_data["confidence"],
            "components": decision_data["scored_options"][0]["components"]
        }
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from experience and update patterns"""
        await self._nyx_context.learn_from_interaction(
            action=experience.get("action", "unknown"),
            outcome=experience.get("outcome", "unknown"),
            success=experience.get("success", False)
        )
        
        # Update local attributes from nyx context
        self.learned_patterns = self._nyx_context.learned_patterns
        self.adaptation_history = self._nyx_context.adaptation_history
        self.learning_metrics = self._nyx_context.learning_metrics
    
    async def process_emotional_state(self, context: Dict[str, Any], user_emotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and update emotional state - Fixed to actually update the state"""
        # Add user emotion to context if provided
        if user_emotion:
            context["user_emotion"] = user_emotion
        
        # Use the composite tool that both calculates AND updates
        result = await calculate_and_update_emotional_state(
            RunContextWrapper(context=self._nyx_context),
            CalculateEmotionalStateInput(context=dict_to_kvlist(context))
        )
        
        emotional_data = json.loads(result)
        # Update local state to match
        self.current_emotional_state = {
            "valence": emotional_data["valence"],
            "arousal": emotional_data["arousal"],
            "dominance": emotional_data["dominance"],
            "primary_emotion": emotional_data["primary_emotion"]
        }
        
        return self.current_emotional_state
    
    async def manage_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_scenario function"""
        from ..nyx_agent_sdk import manage_scenario
        scenario_data["user_id"] = self.user_id
        scenario_data["conversation_id"] = self.conversation_id
        return await manage_scenario(scenario_data)
    
    async def manage_relationships(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_relationships function"""
        from ..nyx_agent_sdk import manage_relationships
        interaction_data["user_id"] = self.user_id
        interaction_data["conversation_id"] = self.conversation_id
        return await manage_relationships(interaction_data)
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return self.current_emotional_state
    
    async def update_emotional_state(self, new_state: Dict[str, Any]):
        """Update emotional state"""
        self.current_emotional_state.update(new_state)
        self._nyx_context.emotional_state.update(new_state)
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update context - compatibility method"""
        self.context_cache.update(new_context)
        self._nyx_context.current_context.update(new_context)
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        return {
            "goals": self.current_goals,
            "active_tasks": self.active_tasks,
            "emotional_state": self.current_emotional_state,
            "beliefs": self.beliefs,
            "performance": {
                "action_success_rate": self.action_success_rate,
                "decision_confidence": self.decision_confidence,
                "goal_progress": self.goal_progress,
                "metrics": self.performance_metrics,
                "resource_usage": self.resource_usage
            },
            "learning": {
                "learned_patterns": self.learned_patterns,
                "strategy_effectiveness": self.strategy_effectiveness,
                "adaptation_history": self.adaptation_history[-5:],
                "metrics": self.learning_metrics
            },
            "errors": {
                "total": self.performance_metrics["error_rates"]["total"],
                "recovered": self.performance_metrics["error_rates"]["recovered"],
                "unrecovered": self.performance_metrics["error_rates"]["unrecovered"]
            }
        }
    
    # Additional compatibility methods
    def _calculate_emotional_weight(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate emotional weight for decisions"""
        intensity = max(abs(emotional_state.get("valence", 0)), abs(emotional_state.get("arousal", 0)))
        return min(1.0, intensity * 2.0)
    
    def _calculate_pattern_weight(self, context: Dict[str, Any]) -> float:
        """Calculate pattern weight for decisions"""
        relevant_patterns = sum(1 for p in self.learned_patterns.values()
                               if any(k in str(context) for k in str(p).split()))
        return min(1.0, relevant_patterns * 0.2)
    
    def _should_run_task(self, task_id: str) -> bool:
        """Check if task should run"""
        return self._nyx_context.should_run_task(task_id)

# Legacy function mappings for backward compatibility
retrieve_memories_impl = retrieve_memories
add_memory_impl = add_memory
get_user_model_guidance_impl = get_user_model_guidance
detect_user_revelations_impl = detect_user_revelations
generate_image_from_scene_impl = generate_image_from_scene
calculate_emotional_impact_impl = calculate_emotional_impact
calculate_and_update_emotional_state_impl = calculate_and_update_emotional_state
manage_beliefs_impl = manage_beliefs
score_decision_options_impl = score_decision_options
detect_conflicts_and_instability_impl = detect_conflicts_and_instability

async def get_emotional_state_impl(ctx) -> str:
    """Get current emotional state"""
    if hasattr(ctx, 'emotional_state'):
        return json.dumps(ctx.emotional_state, ensure_ascii=False)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        return json.dumps(ctx.context.emotional_state, ensure_ascii=False)
    else:
        # Default state
        return json.dumps({
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.7
        }, ensure_ascii=False)

async def update_emotional_state_impl(ctx, emotional_state: Dict[str, Any]) -> str:
    """Update emotional state"""
    if hasattr(ctx, 'emotional_state'):
        ctx.emotional_state.update(emotional_state)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        ctx.context.emotional_state.update(emotional_state)
    return "Emotional state updated"

async def enhance_context_with_strategies_impl(context: Dict[str, Any], conn) -> Dict[str, Any]:
    """Enhance context with active strategies"""
    from nyx.core.sync.strategy_controller import get_active_strategies
    strategies = await get_active_strategies(conn)
    context["nyx2_strategies"] = strategies
    return context

async def determine_image_generation_impl(ctx, response_text: str) -> str:
    """Compatibility wrapper for image generation decision"""
    visual_ctx = NyxContext(ctx.user_id, ctx.conversation_id)
    await visual_ctx.initialize()

    try:
        result = await decide_image_generation(
            RunContextWrapper(context=visual_ctx),
            DecideImageInput(scene_text=response_text)
        )
        return result
    except Exception as e:
        logger.debug(f"decide_image_generation failed: {e}", exc_info=True)
        try:
            from agents import Runner, RunConfig
            from .agents import visual_agent
            result = await Runner.run(
                visual_agent,
                f"Should an image be generated for this scene? {response_text}",
                context=visual_ctx,
                run_config=RunConfig(workflow_name="Nyx Visual Decision"),
            )
            decision = result.final_output_as(ImageGenerationDecision)
            return decision.model_dump_json()
        except Exception as e2:
            logger.warning(f"Visual agent failed: {e2}", exc_info=True)
            return ImageGenerationDecision(
                should_generate=False, score=0, image_prompt=None,
                reasoning="Unable to determine image generation need"
            ).model_dump_json()

async def generate_base_response(ctx: NyxContext, user_input: str, context: Dict[str, Any]) -> NarrativeResponse:
    """Generate base narrative response - for compatibility"""
    from .utils import add_nyx_hosting_style, calculate_world_tension, should_generate_image_for_scene, detect_emergent_opportunities

    # 1. Check world state
    world_state = await ctx.world_director.context.current_world_state if ctx.world_director else None

    # 2. Process through slice-of-life narrator
    narrator_response = await ctx.slice_of_life_narrator.process_player_input(user_input) if ctx.slice_of_life_narrator else ""

    # 3. Let Nyx add her hosting personality
    nyx_enhanced = await add_nyx_hosting_style(narrator_response, world_state) if world_state else {"narrative": narrator_response}

    return NarrativeResponse(
        narrative=nyx_enhanced['narrative'],
        tension_level=calculate_world_tension(world_state) if world_state else 0,
        generate_image=should_generate_image_for_scene(world_state) if world_state else False,
        world_mood=getattr(getattr(world_state, 'world_mood', None), 'value', None) if world_state else None,
        time_of_day=getattr(getattr(getattr(world_state, 'current_time', None), 'time_of_day', None), 'value', None) if world_state else None,
        ongoing_events=[getattr(e, 'title', str(e)) for e in getattr(world_state, 'ongoing_events', [])] if world_state else [],
        available_activities=[getattr(a, 'value', str(a)) for a in getattr(world_state, 'available_activities', [])] if world_state else [],
        emergent_opportunities=detect_emergent_opportunities(world_state) if world_state else []
    )

async def mark_strategy_for_review(conn, strategy_id: int, user_id: int, reason: str):
    """Mark a strategy for review"""
    await conn.execute("""
        INSERT INTO strategy_reviews (strategy_id, user_id, reason, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
    """, strategy_id, user_id, reason)

# Export compatibility mappings
get_emotional_state = get_emotional_state_impl
update_emotional_state = update_emotional_state_impl
enhance_context_with_strategies = enhance_context_with_strategies_impl
determine_image_generation = determine_image_generation_impl

# Compatibility with existing code (direct imports)
async def process_user_input_with_openai(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input using the OpenAI integration"""
    from ..nyx_agent_sdk import process_user_input
    return await process_user_input(user_id, conversation_id, user_input, context_data)

async def process_user_input_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Process user input standalone"""
    from ..nyx_agent_sdk import process_user_input
    return await process_user_input(user_id, conversation_id, user_input, context_data)

async def get_emotional_state(ctx) -> str:
    """Get current emotional state"""
    if hasattr(ctx, 'emotional_state'):
        return json.dumps(ctx.emotional_state, ensure_ascii=False)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        return json.dumps(ctx.context.emotional_state, ensure_ascii=False)
    else:
        # Default state
        return json.dumps({
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.7
        }, ensure_ascii=False)

async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """Update emotional state"""
    if hasattr(ctx, 'emotional_state'):
        ctx.emotional_state.update(emotional_state)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        ctx.context.emotional_state.update(emotional_state)
    return "Emotional state updated"
